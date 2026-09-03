"""SFT 冷启动配置（design_zh.md §4.2）。

0.6B 全参 / 1.7B LoRA 两档超参以 classmethod 预设提供，CLI 可在其基础上覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTConfig:
    # 模型 / 数据 / 输出
    model_name: str = "Qwen/Qwen3-0.6B"
    data_path: str = "data_out/sft_trajectories.jsonl"
    output_dir: str = "checkpoints/sft"
    val_ratio: float = 0.05
    seed: int = 42
    # §4.2 训练超参
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 1e-5
    warmup_ratio: float = 0.1
    max_seq_len: int = 4096
    bf16: bool = True
    report_to: str = "wandb"   # 训练日志上报后端；无 wandb 账号可改为 "none"
    # LoRA（1.7B 档）
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    # 安全：是否信任模型仓库的自定义建模代码（transformers 的 trust_remote_code）。
    # 开启会执行仓库内的任意 Python 代码，仅对官方可信权重（如 Qwen3）放行，默认关闭。
    trust_remote_code: bool = False

    @classmethod
    def full_06b(cls, **kw) -> "SFTConfig":
        """0.6B 全参：lr 1e-5，不用 LoRA。"""
        return cls(model_name="Qwen/Qwen3-0.6B", lr=1e-5, use_lora=False, **kw)

    @classmethod
    def lora_17b(cls, **kw) -> "SFTConfig":
        """1.7B LoRA：lr 2e-4，rank=16 / alpha=32。"""
        return cls(
            model_name="Qwen/Qwen3-1.7B",
            lr=2e-4,
            use_lora=True,
            lora_rank=16,
            lora_alpha=32,
            **kw,
        )


__all__ = ["SFTConfig"]