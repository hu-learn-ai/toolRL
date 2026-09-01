"""拒绝采样过滤（design_zh.md §3.2）。

合成得到的轨迹要满足全部硬性条件才进入 SFT 数据集：
1. ``judge.success``（格式 + 正确 + 答案全对）；
2. 步数落在 ``[min_steps, max_steps]``；
3. 同一个工具没有被调用 ≥3 次（抑制穷举试错）；
4. ``R_answer == 1.0``（答案事实匹配，冗余于 success，但显式写出）。
"""

from __future__ import annotations

from envs.api_sandbox.judge import parse_turn
from envs.base_env import JudgeResult
from envs.task_schema import Task, Trajectory


def count_tool_calls(messages: list) -> dict[str, int]:
    """统计 assistant 回合里每个工具被调用的次数。"""
    counts: dict[str, int] = {}
    for m in messages:
        if getattr(m, "role", None) != "assistant":
            continue
        p = parse_turn(m.content)
        if p.kind == "tool_call" and p.tool_name:
            counts[p.tool_name] = counts.get(p.tool_name, 0) + 1
    return counts


def filter_ok(task: Task, trajectory: Trajectory, judge: JudgeResult) -> bool:
    """一条合成轨迹是否通过拒绝采样过滤（全部硬性条件）。"""
    if not judge.success:
        return False
    if not (task.min_steps <= trajectory.steps <= task.max_steps):
        return False
    if any(c >= 3 for c in count_tool_calls(trajectory.messages).values()):
        return False
    if judge.answer != 1.0:
        return False
    return True


__all__ = ["count_tool_calls", "filter_ok"]