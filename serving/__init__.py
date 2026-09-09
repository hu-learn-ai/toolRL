"""部署（design_zh.md §6）：MCP Server + 最小 Agent Runtime。"""

from .registry import (
    DEFAULT_TIMEOUT,
    RegisteredTool,
    ToolRegistry,
    default_tools,
    to_mcp_tool,
)
from .runtime import AgentResult, run_agent

# api 模块依赖 fastapi，import 时才检查（避免无 fastapi 环境炸）
try:
    from .api import REGISTRY, app  # noqa: F401
except ImportError:
    app = None  # type: ignore[assignment]
    REGISTRY = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_TIMEOUT",
    "RegisteredTool",
    "ToolRegistry",
    "default_tools",
    "to_mcp_tool",
    "AgentResult",
    "run_agent",
    "app",
    "REGISTRY",
]