"""最小 Agent Runtime（design_zh.md §6.2）。

手写工具闭环，不依赖 LangGraph/AutoGen（§6.2 要求循环 + 注册合计 <200 行）：

    1. 拼 system（任务工具列表 + 格式说明）+ user 指令 → 模型生成
    2. 含 <tool_call>：解析 JSON → 执行 → 结果作 tool 消息回填 → 回到 1
    3. 含 <answer>：返回答案
    4. 步数 > max_steps 或解析失败 → 终止并返回错误

复用 ``data.system_prompt``（与 SFT 的 system 口径一致）与 ``envs.api_sandbox.judge.parse_turn``
（Qwen3 输出格式解析），执行走 ``ToolRegistry.call``（带超时 + 参数校验）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from data.synthesis import system_prompt
from data.teacher import Teacher
from envs.api_sandbox.judge import parse_turn
from envs.task_schema import Task

from .registry import ToolRegistry


@dataclass
class AgentResult:
    """Agent 循环产物：答案或错误，以及完整对话（便于审计）。"""

    answer: str | None = None
    error: str | None = None
    messages: list[dict] = field(default_factory=list)
    steps: int = 0


def run_agent(
    task: Task,
    teacher: Teacher,
    registry: ToolRegistry,
    max_steps: int | None = None,
) -> AgentResult:
    """跑最小工具闭环（§6.2），返回答案或错误 + 完整对话。"""
    limit = max_steps if max_steps is not None else task.max_steps
    obs = {"tools": [t.model_dump() for t in task.tools]}
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(obs)},
        {"role": "user", "content": task.instruction},
    ]
    steps = 0
    while steps < limit:
        text = teacher.generate(messages)
        messages.append({"role": "assistant", "content": text})
        parsed = parse_turn(text)

        if parsed.kind == "tool_call":
            steps += 1
            result = registry.call(parsed.tool_name, parsed.arguments or {})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
            continue

        if parsed.kind == "answer":
            steps += 1
            return AgentResult(answer=parsed.answer, messages=messages, steps=steps)

        # invalid：解析失败，终止并返回结构化错误
        steps += 1
        return AgentResult(
            error=f"第 {steps} 回合输出非法：{parsed.error}",
            messages=messages,
            steps=steps,
        )

    return AgentResult(error=f"超过 max_steps={limit} 步仍未完成", messages=messages, steps=steps)


__all__ = ["AgentResult", "run_agent"]