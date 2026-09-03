"""工具注册与执行（design_zh.md §6.1 / §6.2）。

从环境导出的工具 schema（§1.2 ``tools`` 字段）构建注册表，提供带超时 + 参数校验的
进程内调用入口；MCP Server 与 Agent Runtime 都复用这一份，保证"注册 / 执行 / 报错"
口径一致。

MCP SDK 不在本模块导入：schema → MCP tool 的转换是纯函数（``to_mcp_tool``），SDK
只在 ``mcp_server.py`` 内延迟导入。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Callable

from envs.api_sandbox.common import validate_params
from envs.api_sandbox.tools import TOOLS as API_TOOLS
from envs.task_schema import ToolSpec

DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class RegisteredTool:
    """注册表条目：schema（导出 MCP tool / 注入提示词）+ 执行器。"""

    spec: ToolSpec
    handler: Callable[[dict], dict]


def default_tools() -> list[RegisteredTool]:
    """从 api_sandbox 环境导出全部 mock 工具（10 个，§6.1）。"""
    return [RegisteredTool(spec=t.to_spec(), handler=t.handler) for t in API_TOOLS.values()]


def to_mcp_tool(spec: ToolSpec) -> dict:
    """把 ToolSpec 转成 MCP tool 的 JSON Schema 描述（§6.1）。"""
    return {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": {
            "type": "object",
            "properties": spec.parameters.properties,
            "required": spec.parameters.required,
        },
    }


def _run_with_timeout(handler: Callable[[dict], dict], params: dict, timeout: float) -> dict:
    """在独立线程执行 handler，超时返回结构化错误（§6.1 默认 10s）。"""
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(handler, params)
        return fut.result(timeout=timeout)
    except FuturesTimeout:
        return {"error": f"工具调用超时（>{timeout:g}s）"}
    except Exception as e:  # handler 异常也收敛为结构化错误，不外抛（§6.1）
        return {"error": f"工具执行失败: {e}"}
    finally:
        # 不等待卡死的 worker，直接回收 executor（cancel 未启动的 future）
        ex.shutdown(wait=False, cancel_futures=True)


class ToolRegistry:
    """工具注册表：schema 导出 + 带超时/校验的执行。"""

    def __init__(
        self,
        tools: list[RegisteredTool] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._tools = {t.spec.name: t for t in (tools if tools is not None else default_tools())}
        self.timeout = timeout

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def mcp_tools(self) -> list[dict]:
        """导出全部工具的 MCP tool 描述（§6.1）。"""
        return [to_mcp_tool(t.spec) for t in self._tools.values()]

    def call(self, name: str, params: dict) -> dict:
        """执行一次工具调用：未知工具/参数非法/超时都返回结构化 ``{"error": ...}``。"""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"未知工具: {name}"}
        err = validate_params(params, tool.spec.parameters)
        if err is not None:
            return {"error": err}
        return _run_with_timeout(tool.handler, params, self.timeout)


__all__ = [
    "DEFAULT_TIMEOUT",
    "RegisteredTool",
    "default_tools",
    "to_mcp_tool",
    "ToolRegistry",
]