"""大规模 SFT 合成 CLI（design_zh.md §7.1 gen 阶段）。

用法（先配置 TEACHER_API_KEY，或项目根目录放 .env）：

    python scripts/gen_sft.py --n-per-env 2500 --out data_out

两个环境（api_sandbox / text2sql）各生成 n_per_env 条，真实 teacher（LLM）产出轨迹，
拒绝采样过滤后写 ``sft_trajectories.jsonl`` + ``rl_pool.jsonl``，checkpoint 支持断点续跑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.config import TeacherConfig
from data.driver import run_synthesis
from data.llm_teacher import LLMTeacher
from envs.api_sandbox import ApiSandboxEnv
from envs.text2sql import Text2SQLEnv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="大规模 SFT 合成（真实 LLM teacher）")
    p.add_argument("--n-per-env", type=int, default=2500, help="每个环境生成的任务数（§3.3：SFT 3-5k 条）")
    p.add_argument("--out", default="data_out", help="输出目录（sft_trajectories.jsonl / rl_pool.jsonl / checkpoint.jsonl）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true", help="禁用断点续跑，从头重跑")
    p.add_argument("--max-calls", type=int, default=None, help="LLM 调用上限（预算保护，默认不限）")
    args = p.parse_args(argv)

    cfg = TeacherConfig.from_env()
    envs = [ApiSandboxEnv(), Text2SQLEnv()]
    stats = run_synthesis(
        envs=envs,
        teacher_factory=lambda task: LLMTeacher(cfg),
        n_per_env=args.n_per_env,
        out_dir=args.out,
        seed=args.seed,
        resume=not args.no_resume,
        max_calls=args.max_calls,
    )

    total = args.n_per_env * len(envs)
    print(f"[gen_sft] teacher={cfg.model} 新增 {len(stats.results)} 条 SFT 轨迹，"
          f"LLM 调用 {stats.calls} 次，续跑跳过 {stats.skipped}/{total}，"
          f"失败 {stats.failed} 条，输出目录 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())