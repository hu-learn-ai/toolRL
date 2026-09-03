"""GRPO 强化学习配置（design_zh.md §4.3）。

初始超参为待调值，均标注于字段注释。默认从 SFT checkpoint 起步（冷启动后微调）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GRPOConfig:
    # 模型 / 数据 / 输出
    model_name: str = "checkpoints/sft"        # 从 SFT checkpoint 起步
    data_path: str = "data_out/rl_pool.jsonl"  # RL prompt 池（完整 Task）
    output_dir: str = "checkpoints/grpo"
    seed: int = 42
    # §4.3 超参
    group_size: int = 8             # 每条 prompt 采样数 G
    lr: float = 1e-6
    epsilon: float = 0.2            # clip 系数
    beta: float = 0.04              # KL 系数（TRL 默认档；verl 习惯 ~1e-3）
    max_prompt_len: int = 2048
    max_response_len: int = 1024
    num_prompts: int = 128          # 每轮 prompt 数（128 × 8 = 1024 条轨迹）
    max_steps: int = 500            # 训练步数上限（至 reward 平台期）
    # 精度 / 显存
    bf16: bool = True
    report_to: str = "wandb"   # 训练日志上报后端；无 wandb 账号可改为 "none"
    # 安全：是否信任模型仓库的自定义建模代码（transformers 的 trust_remote_code）。
    # 开启会执行仓库内的任意 Python 代码，仅对官方可信权重（如 Qwen3）放行，默认关闭。
    trust_remote_code: bool = False


__all__ = ["GRPOConfig"]