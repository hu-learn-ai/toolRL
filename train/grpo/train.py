"""GRPO 训练入口（design_zh.md §4.3，TRL 备选路径）。

奖励函数复用 envs 的 judge + DEFAULT_REWARD_WEIGHTS（见 reward.py）。TRL 的 GRPOTrainer
奖励回调拿到的是 (prompts, completions) 单轮响应，这里做**单轮判定**的简化路径；
完整的多轮工具闭环（模型中途调用工具、环境回传结果再继续）由 verl 侧 reward manager
（``RewardManager.score_one``，内部跑满 env loop）承担。

GPU 侧依赖（torch/transformers/trl/datasets）均延迟到函数内导入，本模块顶层无 torch。
"""

from __future__ import annotations

from envs.api_sandbox import judge as api_judge
from envs.task_schema import Task
from envs.text2sql import judge as sql_judge

from .config import GRPOConfig
from .prompt import build_prompt, extract_task_id
from .reward import total_reward


def judge_for_task(task: Task, trajectory: list[dict]):
    """按 task_id 前缀分派到对应环境的 judge。"""
    if task.task_id.startswith("sql_"):
        return sql_judge(task, trajectory)
    return api_judge(task, trajectory)


def make_reward_func(task_map: dict[str, Task]):
    """构造 TRL 风格奖励函数 ``(prompts, completions) -> rewards``（单轮判定）。

    说明：单轮把每个 completion 当一条 assistant 响应交给 judge；多轮工具任务的
    完整判定请走 verl + ``RewardManager``。
    """

    def reward_func(prompts: list[str], completions: list[str]) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            task = task_map.get(extract_task_id(prompt) or "")
            if task is None:
                rewards.append(0.0)
                continue
            jr = judge_for_task(task, [{"raw": completion}])
            rewards.append(total_reward(jr))
        return rewards

    return reward_func


def load_policy(cfg: GRPOConfig):
    """加载 SFT checkpoint 作为 GRPO 初始策略。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )
    return model, tokenizer


def train(cfg: GRPOConfig) -> None:
    """用 TRL GRPOTrainer 跑 §4.3 的 GRPO（group_size / β / ε 等从 cfg 映射）。"""
    from datasets import Dataset
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    from data.io import load_rl_pool

    tasks = load_rl_pool(cfg.data_path)
    if not tasks:
        raise RuntimeError(f"RL prompt 池为空：{cfg.data_path}")
    task_map = {t.task_id: t for t in tasks}
    dataset = Dataset.from_list(
        [{"task_id": t.task_id, "prompt": build_prompt(t)} for t in tasks]
    )

    model, tokenizer = load_policy(cfg)

    trl_args = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        num_generations=cfg.group_size,          # G = 8
        max_completion_length=cfg.max_response_len,
        max_prompt_length=cfg.max_prompt_len,
        learning_rate=cfg.lr,
        beta=cfg.beta,                           # KL 系数
        epsilon=cfg.epsilon,                     # clip 系数
        per_device_train_batch_size=1,           # 每 prompt 组内 G 条
        gradient_accumulation_steps=cfg.num_prompts // 1,  # 每轮约 num_prompts 条 prompt
        num_train_epochs=1,
        bf16=cfg.bf16,
        logging_steps=10,
        seed=cfg.seed,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=trl_args,
        train_dataset=dataset,
        reward_funcs=make_reward_func(task_map),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


__all__ = ["judge_for_task", "make_reward_func", "load_policy", "train"]