"""SFT 冷启动 CLI（design_zh.md §7.1 sft 阶段）。

用法：

    python -m train.sft --data data_out/sft_trajectories.jsonl --out checkpoints/sft
    python -m train.sft --lora                      # 1.7B LoRA 档
"""

from __future__ import annotations

import argparse

from .config import SFTConfig
from .train import train


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SFT 冷启动（Qwen3 + 自实现 loss 掩码）")
    p.add_argument("--model", default=None, help="覆盖模型名（默认 0.6B / --lora 1.7B）")
    p.add_argument("--data", default="data_out/sft_trajectories.jsonl")
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--lora", action="store_true", help="1.7B LoRA 档（lr 2e-4, rank=16）")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="覆盖预设学习率")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = SFTConfig.lora_17b() if args.lora else SFTConfig.full_06b()
    if args.model:
        cfg.model_name = args.model
    cfg.data_path = args.data
    cfg.output_dir = args.out
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.grad_accum = args.grad_accum
    cfg.max_seq_len = args.max_seq_len
    cfg.val_ratio = args.val_ratio
    cfg.seed = args.seed
    if args.lr is not None:
        cfg.lr = args.lr

    print(f"[sft] model={cfg.model_name} lr={cfg.lr} lora={cfg.use_lora} "
          f"epochs={cfg.epochs} batch={cfg.batch_size} data={cfg.data_path}")
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())