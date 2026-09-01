"""大规模合成驱动（design_zh.md §3.3 / §7.1 gen 阶段）。

在 ``synthesize_one`` 之上补齐大规模合成所需工程能力：
- 断点续跑：每个处理过的 task_id 写入 checkpoint，重跑自动跳过（省 API 预算）；
- 增量落盘：通过的轨迹追加写 SFT JSONL 与 RL prompt 池，崩溃不丢已产出的数据；
- 调用计数：统计 teacher ``generate`` 次数，配合 ``max_calls`` 做预算上限（§3.3 ≤100 元）。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

from envs.base_env import BaseToolEnv
from envs.task_schema import Task

from .filter import filter_ok
from .io import write_rl_pool, write_sft
from .synthesis import SynthesisResult, synthesize_one
from .teacher import Teacher

TeacherFactory = Callable[[Task], Teacher]


class _CountingTeacher(Teacher):
    """包装 teacher 统计 ``generate`` 调用次数（用于 API 预算）。"""

    def __init__(self, inner: Teacher, counter: dict):
        self._inner = inner
        self._counter = counter

    def generate(self, messages: list[dict]) -> str:
        self._counter["calls"] += 1
        return self._inner.generate(messages)


@dataclass
class RunStats:
    results: list[SynthesisResult]   # 本次新增、通过过滤的轨迹
    calls: int                       # 本次 LLM 调用次数（teacher.generate 次数）
    skipped: int                     # 断点续跑跳过的任务数


def run_synthesis(
    envs: Iterable[BaseToolEnv],
    teacher_factory: TeacherFactory,
    n_per_env: int,
    out_dir: str | Path,
    seed: int = 0,
    resume: bool = True,
    max_calls: int | None = None,
) -> RunStats:
    """批量合成并落盘，返回本次运行的统计。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "sft_trajectories.jsonl"
    rl_path = out_dir / "rl_pool.jsonl"
    ckpt_path = out_dir / "checkpoint.jsonl"

    done_ids: set[str] = set()
    if resume and ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(line.strip())

    counter = {"calls": 0}
    results: list[SynthesisResult] = []
    skipped = 0

    for i, env in enumerate(envs):
        for task in islice(env.task_generator(seed + i), n_per_env):
            if task.task_id in done_ids:
                skipped += 1
                continue
            if max_calls is not None and counter["calls"] >= max_calls:
                break
            r = synthesize_one(task, env, _CountingTeacher(teacher_factory(task), counter))
            if filter_ok(r.task, r.trajectory, r.judge):
                results.append(r)
                write_sft([r], sft_path, append=True)
                write_rl_pool([r.task], rl_path, append=True)
            # checkpoint 记录"已处理"（无论是否通过过滤），避免重跑重复付费
            with ckpt_path.open("a", encoding="utf-8") as f:
                f.write(task.task_id + "\n")

    return RunStats(results=results, calls=counter["calls"], skipped=skipped)


__all__ = ["RunStats", "run_synthesis"]