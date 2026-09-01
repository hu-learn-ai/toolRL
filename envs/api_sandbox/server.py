"""模拟 API 沙箱的 FastAPI 服务（design_zh.md §2.2）。

把 tools.py 注册表里的 10 个 mock API 暴露成 HTTP 端点，启动方式：
    uvicorn envs.api_sandbox.server:app --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .tools import TOOLS, all_specs, call

app = FastAPI(title="ToolRL-Lite Mock API Sandbox")


@app.get("/tools")
def list_tools() -> list[dict]:
    """返回全部工具 schema（对应 design_zh.md §1.2 的 tools 字段）。"""
    return [spec.model_dump() for spec in all_specs()]


@app.post("/tools/{name}")
def invoke_tool(name: str, params: dict) -> dict:
    """执行一次工具调用。参数非法返回 {"error": ...}（200）；未知工具返回 404。"""
    if name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"未知工具: {name}")
    return call(name, params)
