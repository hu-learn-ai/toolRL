"""GRPO 训练 CLI（design_zh.md §7.1 grpo 阶段）。

用法：

    python -m train.grpo --model checkpoints/sft --data data_out/rl_pool.jsonl
"""

from __future__ import annotations

import argparse

from .config import GRPOConfig
from .train import train


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GRPO 强化学习（§4.3，TRL 备选路径）")
    p.add_argument("--model", default="checkpoints/sft", help="初始策略（SFT checkpoint）")
    p.add_argument("--data", default="data_out/rl_pool.jsonl")
    p.add_argument("--out", default="checkpoints/grpo")
    p.add_argument("--group-size", type=int, default=8, help="每条 prompt 采样数 G")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04, help="KL 系数")
    p.add_argument("--epsilon", type=float, default=0.2, help="clip 系数")
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1, help="训练轮数（正式跑建议 2-3）")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-remote-code", action="store_true",
                   help="信任模型仓库的自定义建模代码（Qwen3 官方权重需开启；有任意代码执行风险）")
    args = p.parse_args(argv)

    cfg = GRPOConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.out,
        group_size=args.group_size,
        lr=args.lr,
        beta=args.beta,
        epsilon=args.epsilon,
        num_prompts=args.num_prompts,
        epochs=args.epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"[grpo] model={cfg.model_name} G={cfg.group_size} lr={cfg.lr} "
          f"beta={cfg.beta} eps={cfg.epsilon} data={cfg.data_path}")
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())