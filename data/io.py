"""数据落盘与读取（design_zh.md §1.3 / §3.3）。

SFT 轨迹 JSONL 每行一个 ``Trajectory``（§1.3）；RL 任务池 JSONL 每行一个**完整** Task
（含 gold/meta/max_steps/min_steps）——奖励函数判定需要 gold，故不裁剪为 prompt 子集。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from envs.task_schema import Task, Trajectory

from .synthesis import SynthesisResult


def write_sft(results: Iterable[SynthesisResult], path: str | Path, append: bool = False) -> int:
    """把合成结果里的轨迹按行写入 JSONL，返回写入条数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.trajectory.model_dump_json() + "\n")
            n += 1
    return n


def load_sft(path: str | Path) -> list[Trajectory]:
    """从 JSONL 读回 SFT 轨迹。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SFT 轨迹文件不存在：{path}（请先运行 scripts/gen_sft.py 合成）")
    out: list[Trajectory] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Trajectory.model_validate_json(line))
    return out


def write_rl_pool(tasks: Iterable[Task], path: str | Path, append: bool = False) -> int:
    """把任务写成 RL 采样池（每行一个完整 ``Task``，奖励函数据此判定）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")
            n += 1
    return n


def load_rl_pool(path: str | Path) -> list[Task]:
    """从 RL 采样池 JSONL 读回完整任务。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RL 任务池文件不存在：{path}（请先运行 scripts/gen_tasks.py 或 scripts/gen_sft.py）")
    out: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Task.model_validate_json(line))
    return out


__all__ = ["write_sft", "load_sft", "write_rl_pool", "load_rl_pool"]