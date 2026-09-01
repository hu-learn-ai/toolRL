"""RL prompt 组装（design_zh.md §4.3 任务 prompt 数据格式）。

复用 ``data.system_prompt`` 保证与 SFT 的 system 提示完全一致（口径一致原则），
并把 task_id 埋进 prompt 首行，供奖励函数回查 gold 判定。
"""

from __future__ import annotations

import re

from data.synthesis import system_prompt
from envs.task_schema import Task

_TASK_ID_RE = re.compile(r"<task_id>(.*?)</task_id>", re.DOTALL)


def build_prompt(task: Task, obs: dict | None = None) -> str:
    """组装 prompt：``<task_id>…</task_id>`` + system（工具+格式）+ user 指令。"""
    if obs is None:
        obs = {"tools": [t.model_dump() for t in task.tools]}
    sys = system_prompt(obs)
    return f"<task_id>{task.task_id}</task_id>\n\n{sys}\n\n{task.instruction}"


def extract_task_id(prompt: str) -> str | None:
    """从 prompt 里解析回 task_id（奖励函数用）。"""
    m = _TASK_ID_RE.search(prompt)
    return m.group(1).strip() if m else None


__all__ = ["build_prompt", "extract_task_id"]