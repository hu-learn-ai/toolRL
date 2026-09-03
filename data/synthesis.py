"""数据合成主流程（design_zh.md §3）。

teacher 模型 + 环境 rollout 产出 SFT 轨迹：
- ``reset`` 得到初始观察 → 组装 system prompt（工具列表 + 格式指令 + [Text2SQL 表结构]）；
- 逐回合让 teacher 生成，tool_call 则喂给 ``env.step``、把观察回填，answer 则结束；
- 用 ``env.judge`` 对原始输出打分，过滤后落盘。

轨迹形如 ``Trajectory``（§1.3），``messages`` 含 system/user/assistant/tool 四类角色。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from typing import Iterable

from envs.api_sandbox.judge import parse_turn
from envs.base_env import BaseToolEnv, JudgeResult
from envs.task_schema import Message, Task, Trajectory

from .filter import filter_ok
from .teacher import GoldTeacher, Teacher

_MAX_SANITY = 32  # 单条轨迹回合数安全上限，防死循环


def system_prompt(obs: dict) -> str:
    """由初始观察组装 system prompt：工具列表 JSON + Qwen3 格式指令 + [表结构]。

    格式指令对齐 §4.1：先 <think> 再 <tool_call>/<answer>，一次只输出一个动作。
    """
    parts = [
        "你是工具调用助手，按以下格式输出，先思考再行动：",
        "<think>简要说明下一步要调用哪个工具及原因</think>",
        '<tool_call>{"name": "...", "arguments": {...}}</tool_call>',
        "收到工具返回后继续；任务完成时最终输出：",
        "<answer>...</answer>",
        "每回合只能输出一个 <tool_call> 或一个 <answer>，不要一次输出多个动作。",
        "可用工具（JSON）：",
        json.dumps(obs.get("tools", []), ensure_ascii=False),
    ]
    schema = obs.get("schema")
    if schema:
        parts += ["数据库表结构：", schema]
    return "\n".join(parts)


@dataclass
class SynthesisResult:
    task: Task
    trajectory: Trajectory
    judge: JudgeResult


def synthesize_one(task: Task, env: BaseToolEnv, teacher: Teacher) -> SynthesisResult:
    """对单个任务 rollout 一条轨迹并打分。"""
    obs = env.reset(task)
    messages = [
        Message(role="system", content=system_prompt(obs)),
        Message(role="user", content=task.instruction),
    ]
    raw_turns: list[str] = []
    for _ in range(_MAX_SANITY):
        text = teacher.generate([m.model_dump() for m in messages])
        if not isinstance(text, str):
            # 兜底：teacher 返回 None / 非字符串（异常输出）按无效回合处理，避免
            # Message(content=...) 触发 pydantic 校验崩溃
            text = ""
        parsed = parse_turn(text)
        if parsed.kind == "tool_call":
            raw_turns.append(text)
            messages.append(Message(role="assistant", content=text))
            action = {"api": parsed.tool_name, "params": parsed.arguments or {}}
            out = env.step(action)
            messages.append(
                Message(role="tool", content=json.dumps(out["observation"], ensure_ascii=False))
            )
            if out.get("done"):
                break
        else:
            # answer（或 invalid）都视为结束回合
            raw_turns.append(text)
            messages.append(Message(role="assistant", content=text))
            break

    trajectory = Trajectory(
        task_id=task.task_id,
        messages=messages,
        success=False,
        steps=len(raw_turns),
    )
    result = env.judge([{"raw": r} for r in raw_turns])
    trajectory.success = result.success
    return SynthesisResult(task=task, trajectory=trajectory, judge=result)


def synthesize_dataset(
    envs: Iterable[BaseToolEnv],
    n_per_env: int,
    seed: int = 0,
    teacher_factory=GoldTeacher,
) -> list[SynthesisResult]:
    """对多个环境批量合成，只保留通过拒绝采样过滤的轨迹。"""
    results: list[SynthesisResult] = []
    for i, env in enumerate(envs):
        for task in islice(env.task_generator(seed + i), n_per_env):
            r = synthesize_one(task, env, teacher_factory(task))
            if filter_ok(r.task, r.trajectory, r.judge):
                results.append(r)
    return results


__all__ = [
    "system_prompt",
    "SynthesisResult",
    "synthesize_one",
    "synthesize_dataset",
]