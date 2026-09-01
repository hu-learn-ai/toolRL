"""工具注册表：汇总各领域工具，提供进程内调用入口。"""

from __future__ import annotations

from ..task_schema import ToolSpec
from .calendar import TOOLS as calendar_tools
from .common import Tool
from .flight import TOOLS as flight_tools
from .hotel import TOOLS as hotel_tools
from .train import TOOLS as train_tools
from .weather import TOOLS as weather_tools

# 顺序即注册顺序，也决定 all_specs() 的导出顺序
_ALL_TOOLS: tuple[Tool, ...] = (
    *weather_tools,
    *flight_tools,
    *train_tools,
    *hotel_tools,
    *calendar_tools,
)

TOOLS: dict[str, Tool] = {t.name: t for t in _ALL_TOOLS}


def call(name: str, params: dict) -> dict:
    """进程内调用工具；未知工具返回 {"error": ...}。"""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"未知工具: {name}"}
    return tool.handler(params)


def all_specs() -> list[ToolSpec]:
    """导出全部工具 schema（供环境注入系统提示词 / MCP 注册）。"""
    return [t.to_spec() for t in _ALL_TOOLS]


__all__ = ["TOOLS", "Tool", "call", "all_specs"]
