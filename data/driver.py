"""大规模合成驱动（design_zh.md §3.3 / §7.1 gen 阶段）。

在 ``synthesize_one`` 之上补齐大规模合成所需工程能力：
- 断点续跑：每个处理过的 task_id 写入 checkpoint，重跑自动跳过（省 API 预算）；
- 增量落盘：通过的轨迹追加写 SFT JSONL 与 RL prompt 池，崩溃不丢已产出的数据；
- 调用计数：统计 teacher ``generate`` 次数，配合 ``max_calls`` 做预算上限（§3.3 ≤100 元）；
- 失败隔离：单任务 LLM 故障重试限次后跳过并记账，不中断整批（§3.3 大规模跑量）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

from envs.base_env import BaseToolEnv
from envs.task_schema import Task

from .filter import filter_ok
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
    failed: int = 0                  # 重试后仍失败、被跳过的任务数


def _synthesize_with_retry(
    task: Task,
    env: BaseToolEnv,
    teacher_factory: TeacherFactory,
    counter: dict,
    max_attempts: int,
) -> SynthesisResult | None:
    """带限次重试的合成：LLM 瞬时故障（超时/5xx/网络抖动）不中断整批。

    重试会重新构造 teacher 并重跑 rollout；``counter`` 跨尝试累计，预算口径包含失败调用。
    ``max_attempts`` 次都失败则返回 ``None``（记入 ``failed``，不写 checkpoint，续跑时
    该任务会被重新处理）。
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return synthesize_one(task, env, _CountingTeacher(teacher_factory(task), counter))
        except Exception as e:  # 兜底：单任务失败不应中断整批
            last_exc = e
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 30.0))  # 指数退避，封顶 30s
    print(f"[synth] 任务 {task.task_id} 失败（重试 {max_attempts} 次后跳过）: {last_exc}")
    return None


def run_synthesis(
    envs: Iterable[BaseToolEnv],
    teacher_factory: TeacherFactory,
    n_per_env: int,
    out_dir: str | Path,
    seed: int = 0,
    resume: bool = True,
    max_calls: int | None = None,
    max_attempts: int = 3,
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
    failed = 0

    # 三个文件各打开一次，避免每结果 open/close 的 churn；先 flush 数据文件再写
    # checkpoint，保证"标记已处理"时轨迹数据已落盘（崩溃续跑不丢数据）。
    # resume=False 走 "w" 清空旧产物（从头重跑）；resume=True 走 "a" 增量续写。
    mode = "a" if resume else "w"
    with (
        sft_path.open(mode, encoding="utf-8") as sft_f,
        rl_path.open(mode, encoding="utf-8") as rl_f,
        ckpt_path.open(mode, encoding="utf-8") as ckpt_f,
    ):
        for i, env in enumerate(envs):
            for task in islice(env.task_generator(seed + i), n_per_env):
                if task.task_id in done_ids:
                    skipped += 1
                    continue
                if max_calls is not None and counter["calls"] >= max_calls:
                    return RunStats(
                        results=results, calls=counter["calls"], skipped=skipped, failed=failed
                    )
                r = _synthesize_with_retry(task, env, teacher_factory, counter, max_attempts)
                if r is None:
                    failed += 1
                    continue
                if filter_ok(r.task, r.trajectory, r.judge):
                    results.append(r)
                    sft_f.write(r.trajectory.model_dump_json() + "\n")
                    rl_f.write(r.task.model_dump_json() + "\n")
                    sft_f.flush()
                    rl_f.flush()
                # checkpoint 记录"已处理"（无论是否通过过滤），避免重跑重复付费
                ckpt_f.write(task.task_id + "\n")
                ckpt_f.flush()

    return RunStats(results=results, calls=counter["calls"], skipped=skipped, failed=failed)


__all__ = ["RunStats", "run_synthesis"]