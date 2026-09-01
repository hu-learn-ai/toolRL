"""Teacher 模型抽象（design_zh.md §3）。

SFT 冷启动的教师：给定对话上下文，产出下一回合的原始输出。默认用
``GoldTeacher``（直接按 gold 轨迹生成，用于验收闭环、跑通合成管线）；
正式合成时替换为真实 LLM（如 Qwen3-32B / DeepSeek）。
"""

from __future__ import annotations

import abc
import json

from envs.task_schema import Task


class Teacher(abc.ABC):
    """教师模型接口：``generate(messages)`` 返回下一回合原始输出。"""

    @abc.abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """``messages`` 为 OpenAI 风格 ``[{role, content}]``，返回 <tool_call>/<answer> 文本。"""


class GoldTeacher(Teacher):
    """黄金教师：按 ``task.gold`` 依次吐出 tool_call，最后吐 <answer>。

    只用于验证数据合成管线（黄金轨迹必过拒绝采样），不是真实教师模型。
    """

    def __init__(self, task: Task):
        self._calls = list(task.gold.calls)
        self._answer = task.gold.answer
        self._i = 0

    def generate(self, messages: list[dict]) -> str:
        if self._i < len(self._calls):
            c = self._calls[self._i]
            self._i += 1
            payload = json.dumps(
                {"name": c.api, "arguments": c.params}, ensure_ascii=False
            )
            return f"<tool_call>{payload}</tool_call>"
        return f"<answer>{self._answer}</answer>"


__all__ = ["Teacher", "GoldTeacher"]