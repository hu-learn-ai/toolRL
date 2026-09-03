"""MCP Server（design_zh.md §6.1）。

把 ``ToolRegistry`` 里的工具注册为 MCP tools，通过 MCP Python SDK 提供 stdio transport。
SDK（``mcp``）延迟导入：无 SDK 环境 import 本模块不报错，只有 ``create_server`` /
``run_stdio`` 才会真正加载 SDK。

工具调用带超时 + 参数校验（由 ``ToolRegistry.call`` 负责），错误以结构化 JSON 文本
返回，不抛未处理异常（§6.1）。
"""

from __future__ import annotations

import json

from .registry import ToolRegistry


def create_server(registry: ToolRegistry, name: str = "toolrl-tools"):
    """创建并返回注册了全部工具的 MCP Server 实例（惰性导入 mcp SDK）。

    注意：MCP Python SDK 的 ``Tool`` / ``TextContent`` 类型随版本略有差异，若升级 SDK
    后字段不符，调整此处的模型构造即可（schema 转换逻辑在 ``registry.to_mcp_tool``）。
    """
    from mcp.server import Server
    from mcp.types import TextContent
    from mcp.types import Tool as McpTool

    server = Server(name)
    tools = [McpTool(**t) for t in registry.mcp_tools()]

    @server.list_tools()
    async def list_tools() -> list[McpTool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = registry.call(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    return server


async def run_stdio(registry: ToolRegistry, name: str = "toolrl-tools") -> None:
    """以 stdio transport 启动 MCP Server（§6.1）。"""
    from mcp.server.stdio import stdio_server

    server = create_server(registry, name)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


__all__ = ["create_server", "run_stdio"]