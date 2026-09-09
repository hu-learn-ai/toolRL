"""GRPO 训练入口（design_zh.md §4.3，TRL 备选路径）。

奖励函数复用 envs 的 judge + DEFAULT_REWARD_WEIGHTS（见 reward.py）。TRL 的 GRPOTrainer
奖励回调拿到的是 (prompts, completions) 单轮响应，这里做**单轮判定**的简化路径；
完整的多轮工具闭环（模型中途调用工具、环境回传结果再继续）由 verl 侧 reward manager
（``RewardManager.score_one``，内部跑满 env loop）承担。

GPU 侧依赖（torch/transformers/trl/datasets）均延迟到函数内导入，本模块顶层无 torch。
"""

from __future__ import annotations

import inspect

from envs.api_sandbox import judge as api_judge
from envs.task_schema import Task
from envs.text2sql import judge as sql_judge

from .config import GRPOConfig
from .prompt import build_prompt, extract_task_id
from .reward import first_turn_reward, total_reward

# torch 2.5.1+cu124 的 wheel 缺 FSDPModule export，而 trl 1.12 顶层 import 会用到。
# 必须在 trl 首次被 import 之前补上（trl 在 train() 内延迟 import，这里顶层打补丁）。
# try/except 保护：无 torch 的环境（纯单测机）import 本模块不炸。
try:
    import torch.distributed.fsdp as _fsdp_pkg

    if not hasattr(_fsdp_pkg, "FSDPModule"):
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullyShardedDataParallel as _FSDP_legacy,
        )

        _fsdp_pkg.FSDPModule = _FSDP_legacy
except ImportError:
    pass


def judge_for_task(task: Task, trajectory: list[dict]):
    """按 task_id 前缀分派到对应环境的 judge。"""
    if task.task_id.startswith("sql_"):
        return sql_judge(task, trajectory)
    return api_judge(task, trajectory)


def make_reward_func(task_map: dict[str, Task]):
    """构造 TRL 风格奖励函数 ``(prompts, completions) -> rewards``（单轮判定）。

    说明：api 任务用 ``first_turn_reward``（只评第一回合 tool_call，上限 0.7，
    杜绝"跳过工具直接 answer 得 0.6"的 hacking 向量，见 reward.py 文档字符串）；
    sql 任务暂沿用完整 judge 的单轮简化。多轮工具任务的完整判定请走 verl +
    ``RewardManager``。
    """

    def reward_func(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            task = task_map.get(extract_task_id(prompt) or "")
            if task is None:
                rewards.append(0.0)
                continue
            if task.task_id.startswith("sql_"):
                jr = judge_for_task(task, [{"raw": completion}])
                rewards.append(total_reward(jr))
            else:
                rewards.append(first_turn_reward(task, completion))
        return rewards

    return reward_func


def load_policy(cfg: GRPOConfig):
    """加载 SFT checkpoint 作为 GRPO 初始策略。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation="sdpa",
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

    # TRL 要求 batch size 与 num_generations 对齐（否则报"不能整除"），
    # 这里让 per_device batch = G，梯度累积补足到约 num_prompts 条 prompt。
    per_device_batch = cfg.group_size
    trl_args = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        num_generations=cfg.group_size,          # G = 8
        max_completion_length=cfg.max_response_len,
        # 注意：trl 1.12.0 已移除 max_prompt_length，超长靠 tokenizer 截断兜底
        learning_rate=cfg.lr,
        beta=cfg.beta,                           # KL 系数
        epsilon=cfg.epsilon,                     # clip 系数（下界）
        epsilon_high=cfg.epsilon,                # clip 上界（对称 clip）
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=max(1, cfg.num_prompts // per_device_batch),
        num_train_epochs=cfg.epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        mask_truncated_completions=True,         # 被截断的 rollout 不计入 loss
        log_completions=False,
        bf16=cfg.bf16,
        logging_steps=10,
        seed=cfg.seed,
        report_to=cfg.report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=trl_args,
        train_dataset=dataset,
        reward_funcs=make_reward_func(task_map),
    )
    # transformers 4.49+ 的 Trainer 会调用 self._get_train_sampler(dataset)，
    # 而 trl 0.16 的方法签名只有 self；仅在旧签名下包一层兼容（忽略 dataset）。
    if len(inspect.signature(type(trainer)._get_train_sampler).parameters) == 1:
        _orig_sampler = trainer._get_train_sampler

        def _sampler_compat(_dataset=None):
            return _orig_sampler()

        trainer._get_train_sampler = _sampler_compat
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


__all__ = ["judge_for_task", "make_reward_func", "load_policy", "train"]
