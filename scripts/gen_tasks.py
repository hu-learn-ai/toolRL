"""生成任务池 JSONL（design_zh.md §7.1 gen_tasks 阶段）。

固定种子生成 api_sandbox + text2sql 任务并落盘，产物为统一 Task schema 的
JSONL（复用 ``data.io.write_rl_pool`` 的写入路径），可作数据合成 / RL prompt 池 /
评测的共享任务来源。
"""

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.io import write_rl_pool
from envs.api_sandbox import task_generator as api_tasks
from envs.text2sql import task_generator as sql_tasks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成任务池 JSONL（固定种子可复现）")
    p.add_argument("--n", type=int, default=400, help="每个环境生成的任务数（默认 400）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--out", default="data_out/tasks.jsonl", help="输出路径")
    args = p.parse_args(argv)

    tasks = [
        t
        for env in (api_tasks(args.seed), sql_tasks(args.seed + 1))
        for t in islice(env, args.n)
    ]
    write_rl_pool(tasks, args.out)
    print(f"[gen_tasks] 生成 {len(tasks)} 条任务 -> {args.out}（seed={args.seed}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
