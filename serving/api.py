"""FastAPI 部署入口（design_zh.md §6.3）。

把 serving/registry + serving/runtime 暴露成 HTTP 接口，让云端 GPU 实例能给任何
HTTP 客户端（含 curl、Claude Desktop via MCP-HTTP-proxy、自写前端）使用。

端点
----
- ``GET  /health``             健康检查 + 已加载模型清单
- ``GET  /tools``              列出 ToolRegistry 里的全部工具 schema
- ``POST /chat/{model}``       单轮对话（``model ∈ {teacher,sft,grpo_v3}``）
- ``GET  /models``             列出可用的 ``{model}`` 名字与对应 checkpoint 路径

请求体（``POST /chat/{model}``）::

    {
      "instruction": "帮我查一下深圳未来3天的天气",
      "max_steps": 6,            # 可选，默认 task.max_steps
      "seed": 42                 # 可选，默认 42
    }

响应体::

    {
      "answer": "...",            # 成功时
      "error": null,             # 失败时
      "steps": 3,
      "messages": [...]           # 完整对话，供审计
      "model": "sft"
    }

启动::

    uvicorn serving.api:app --host 0.0.0.0 --port 8000
    # 或
    python -m serving --http
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from envs.api_sandbox.task_generator import generate_tasks as api_tasks
from envs.task_schema import Task

from .model_loader import get_teacher, warmup
from .registry import ToolRegistry
from .runtime import run_agent

log = logging.getLogger("serving.api")

CHECKPOINT_SFT = os.environ.get("CHECKPOINT_SFT", "checkpoints/sft")
CHECKPOINT_GRPO_V3 = os.environ.get("CHECKPOINT_GRPO_V3", "checkpoints/grpo_v3")
DEFAULT_TEMPERATURE = float(os.environ.get("SERVING_TEMPERATURE", "0.7"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("SERVING_MAX_NEW_TOKENS", "1024"))

# 任务种子/下标池：HTTP /chat 没有 task schema，只给 instruction 时用一份固定工具列表
_DEFAULT_TASK_SEED = 0
_DEFAULT_TASK_POOL_SIZE = 100

# === FastAPI 懒加载：app 在脚本被 import 时构造，路由立即可用 ===
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # 让上层（__main__.py / docker build）有明确报错
    raise ImportError(
        "FastAPI 未安装，请先 `pip install -e \".[env]\"`（含 fastapi + uvicorn）"
    ) from e


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动时可选预热（``SERVING_WARMUP=1``）。失败不阻塞启动。"""
    if os.environ.get("SERVING_WARMUP", "0") == "1":
        for name, path in [("sft", CHECKPOINT_SFT), ("grpo_v3", CHECKPOINT_GRPO_V3)]:
            try:
                log.info("warmup %s from %s ...", name, path)
                warmup(name, path)
            except Exception as e:  # noqa: BLE001
                log.warning("warmup %s 失败（%s），请求时会再尝试", name, e)
    yield


app = FastAPI(title="ToolRL-Lite Serving", version="0.1.0", lifespan=_lifespan)
REGISTRY = ToolRegistry()


# ----------------------- Schema -----------------------

class ChatRequest(BaseModel):
    instruction: str = Field(..., description="用户指令")
    max_steps: int | None = Field(None, description="工具调用最大步数，默认 task.max_steps")
    seed: int | None = Field(42, description="采样种子，默认 42")
    temperature: float | None = Field(None, description="解码温度，默认 0.7")


class ChatResponse(BaseModel):
    answer: str | None = None
    error: str | None = None
    steps: int = 0
    messages: list[dict] = Field(default_factory=list)
    model: str


# ----------------------- 工具：构造 task + teacher -----------------------

def _resolve_task(instruction: str) -> Task:
    """从固定 task pool 里挑一条 instruction 最相近的，复用其工具列表 + max_steps。

    HTTP 部署没有用户输入 task schema，只能从默认任务池推断可用工具集合；
    按 instruction 子串匹配第一个 task。匹配不到时退到 pool[0]。
    """
    pool = api_tasks(_DEFAULT_TASK_SEED, _DEFAULT_TASK_POOL_SIZE)
    for task in pool:
        if task.instruction in instruction or instruction in task.instruction:
            return task
    return pool[0]


def _resolve_teacher(model: str, req: ChatRequest):
    """根据 model 名取 Teacher；teacher 模式返回 GoldTeacher（按当前 task）。"""
    seed = req.seed if req.seed is not None else 42
    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    if model == "teacher":
        task = _resolve_task(req.instruction)
        from data.teacher import GoldTeacher
        return GoldTeacher(task), task
    if model == "sft":
        teacher = get_teacher(
            "sft",
            CHECKPOINT_SFT,
            temperature=temperature,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            seed=seed,
        )
        task = _resolve_task(req.instruction)
        return teacher, task
    if model == "grpo_v3":
        teacher = get_teacher(
            "grpo_v3",
            CHECKPOINT_GRPO_V3,
            temperature=temperature,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            seed=seed,
        )
        task = _resolve_task(req.instruction)
        return teacher, task
    raise HTTPException(status_code=404, detail=f"未知 model: {model}")


# ----------------------- 路由 -----------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "tools": REGISTRY.names(),
        "checkpoints": {
            "sft": CHECKPOINT_SFT,
            "grpo_v3": CHECKPOINT_GRPO_V3,
        },
    }


@app.get("/tools")
def list_tools() -> list[dict]:
    return REGISTRY.mcp_tools()


@app.get("/models")
def list_models() -> dict[str, str]:
    return {
        "teacher": "(GoldTeacher, 规则，无 checkpoint)",
        "sft": CHECKPOINT_SFT,
        "grpo_v3": CHECKPOINT_GRPO_V3,
    }


@app.post("/chat/{model}", response_model=ChatResponse)
def chat(model: str, req: ChatRequest) -> ChatResponse:
    try:
        teacher, task = _resolve_teacher(model, req)
    except HTTPException:
        raise

    # teacher 是 GoldTeacher（无 GPU 也能跑）；其它走 HuggingFaceTeacher（需 GPU/内存）
    result = run_agent(task, teacher, REGISTRY, max_steps=req.max_steps)
    return ChatResponse(
        answer=result.answer,
        error=result.error,
        steps=result.steps,
        messages=result.messages,
        model=model,
    )


# ----------------------- 启动钩子（已迁移到 lifespan，见上方） -----------------------


__all__ = ["app", "REGISTRY"]