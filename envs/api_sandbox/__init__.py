"""模拟 API 沙箱环境（design_zh.md §2.2）。

MVP 提供 10 个 mock API（天气/出行/差旅/效率），固定种子确定性返回；
任务生成器（模板 + 随机槽位 + 干扰工具注入）与分级判定（R_format/correct/answer/steps）
均已实现。
"""

from .env import ApiSandboxEnv
from .judge import ParsedTurn, judge, parse_turn
from .task_generator import generate_tasks, task_generator
from .tools import TOOLS, Tool, all_specs, call

__all__ = [
    "TOOLS",
    "Tool",
    "all_specs",
    "call",
    "task_generator",
    "generate_tasks",
    "ParsedTurn",
    "parse_turn",
    "judge",
    "ApiSandboxEnv",
]
