"""Benchmark 构建（design_zh.md §5.1）。

确定性生成 300-500 条 api_sandbox 任务，按维度占比：

    单工具 25% / 多工具 30% / 多轮 20% / 长程 10% / 干扰项 15%

实现要点：不从 ``task_generator`` 均匀采样（那按 17 个模板均匀，与维度占比不符），
而是按类别分组模板后按占比配额逐类采样；干扰项维度用 ``generate_task(distractor_ratio=1.0)``
强制注入干扰工具并打 ``meta["distractor"] = True`` 标签。每条任务都带 ``meta["bench_dim"]``
标记所属维度，供分维度报告聚合。
"""

from __future__ import annotations

import random

from envs.api_sandbox.task_generator import TEMPLATES, generate_task
from envs.task_schema import Task

# §5.1 维度占比（和为 1.0）
PROPORTIONS: dict[str, float] = {
    "single_tool": 0.25,
    "multi_tool": 0.30,
    "multi_turn": 0.20,
    "long_horizon": 0.10,
    "distractor": 0.15,
}

DEFAULT_SIZE = 400
_BASE_CATS = ["single_tool", "multi_tool", "multi_turn", "long_horizon"]


def _templates_by_category() -> dict[str, list]:
    by: dict[str, list] = {}
    for cat, fn in TEMPLATES:
        by.setdefault(cat, []).append(fn)
    return by


def build_benchmark(seed: int = 42, size: int = DEFAULT_SIZE) -> list[Task]:
    """按维度占比确定性生成 ``size`` 条任务（给定 seed 结果完全可复现，§5.3）。"""
    if size <= 0:
        return []
    rng = random.Random(seed)
    by_cat = _templates_by_category()

    # 基础维度按占比取整，余数归干扰项，保证总数恰为 size
    counts = {cat: round(size * PROPORTIONS[cat]) for cat in _BASE_CATS}
    counts["distractor"] = size - sum(counts.values())

    tasks: list[Task] = []
    i = 0

    def _next_id() -> str:
        nonlocal i
        tid = f"bench_{i:04d}"
        i += 1
        return tid

    for cat in _BASE_CATS:
        for _ in range(counts[cat]):
            fn = rng.choice(by_cat[cat])
            task = fn(rng, _next_id())
            task.meta["bench_dim"] = cat
            tasks.append(task)

    for _ in range(counts["distractor"]):
        task = generate_task(rng, _next_id(), distractor_ratio=1.0)
        task.meta["bench_dim"] = "distractor"
        task.meta["distractor"] = True
        tasks.append(task)

    return tasks


__all__ = ["PROPORTIONS", "DEFAULT_SIZE", "build_benchmark"]