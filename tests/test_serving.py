"""部署测试（design_zh.md §6）。

覆盖：ToolRegistry 注册 / 参数校验 / 未知工具 / 超时、``to_mcp_tool`` schema 转换、
Agent Runtime 循环（GoldTeacher 全对 / BadTeacher 失败 / 步数超限）。
MCP SDK 未安装，故不测 SDK 集成，只测纯逻辑（schema 转换与执行）。
"""

import time

from data import GoldTeacher
from data.teacher import Teacher
from envs.api_sandbox.task_generator import generate_tasks as api_tasks
from envs.task_schema import ToolParameterSchema, ToolSpec
from serving import (
    DEFAULT_TIMEOUT,
    RegisteredTool,
    ToolRegistry,
    run_agent,
    to_mcp_tool,
)


class BadTeacher(Teacher):
    def generate(self, messages):
        return "没有 tool_call 也没有 answer 的乱输出"


class LoopTeacher(Teacher):
    """永远吐 tool_call，用于测步数超限。"""

    def generate(self, messages):
        return '<tool_call>{"name":"weather.query","arguments":{"city":"北京"}}</tool_call>'


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_default_tools_count_and_names():
    reg = ToolRegistry()
    assert len(reg.names()) == 10
    assert "weather.query" in reg.names()
    assert "flight.search" in reg.names()
    assert "meeting.create" in reg.names()


def test_default_timeout():
    assert DEFAULT_TIMEOUT == 10.0
    assert ToolRegistry().timeout == 10.0


def test_registry_call_ok():
    reg = ToolRegistry()
    out = reg.call("weather.query", {"city": "北京"})
    assert "error" not in out
    assert out["city"] == "北京"
    assert {"temperature", "condition", "humidity"} <= set(out)


def test_registry_call_missing_required():
    reg = ToolRegistry()
    out = reg.call("weather.query", {})
    assert "error" in out
    assert "缺少必填参数" in out["error"]


def test_registry_call_unknown_tool():
    reg = ToolRegistry()
    out = reg.call("no.such.tool", {})
    assert out == {"error": "未知工具: no.such.tool"}


def test_registry_call_timeout():
    def slow(params):
        time.sleep(0.3)
        return {"ok": True}

    spec = ToolSpec(
        name="slow.op",
        description="慢工具",
        parameters=ToolParameterSchema(properties={}, required=[]),
    )
    reg = ToolRegistry([RegisteredTool(spec=spec, handler=slow)], timeout=0.05)
    out = reg.call("slow.op", {})
    assert "error" in out
    assert "超时" in out["error"]


# ---------------------------------------------------------------------------
# to_mcp_tool / mcp_tools schema 转换（§6.1）
# ---------------------------------------------------------------------------


def test_to_mcp_tool_schema():
    reg = ToolRegistry()
    spec = next(s for s in reg.specs() if s.name == "weather.query")
    t = to_mcp_tool(spec)
    assert t["name"] == "weather.query"
    assert t["description"]
    assert t["inputSchema"]["type"] == "object"
    assert t["inputSchema"]["properties"]["city"]["type"] == "string"
    assert "city" in t["inputSchema"]["required"]


def test_mcp_tools_cover_all():
    reg = ToolRegistry()
    tools = reg.mcp_tools()
    assert len(tools) == 10
    assert {t["name"] for t in tools} == set(reg.names())
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


# ---------------------------------------------------------------------------
# Agent Runtime（§6.2）
# ---------------------------------------------------------------------------


def test_run_agent_gold_success():
    task = api_tasks(0, 1)[0]
    reg = ToolRegistry()
    result = run_agent(task, GoldTeacher(task), reg)
    assert result.error is None
    assert result.answer is not None
    assert result.steps == task.min_steps
    # 对话含 system/user/assistant/tool 四类角色
    roles = {m["role"] for m in result.messages}
    assert {"system", "user", "assistant", "tool"} <= roles


def test_run_agent_bad_invalid_turn():
    task = api_tasks(0, 1)[0]
    reg = ToolRegistry()
    result = run_agent(task, BadTeacher(), reg)
    assert result.answer is None
    assert result.error is not None
    assert "非法" in result.error
    assert result.steps == 1


def test_run_agent_step_limit():
    task = api_tasks(0, 1)[0]
    reg = ToolRegistry()
    result = run_agent(task, LoopTeacher(), reg, max_steps=3)
    assert result.answer is None
    assert result.error is not None
    assert "max_steps" in result.error
    assert result.steps == 3