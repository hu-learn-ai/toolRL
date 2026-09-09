"""部署 CLI（design_zh.md §6）。

三种模式：

    python -m serving --mcp                 # stdio 启动 MCP 工具服务器（§6.1）
    python -m serving --task-seed 0         # 用 GoldTeacher 演示最小 Agent Runtime（§6.2）
    python -m serving --http                # FastAPI HTTP 服务，监听 0.0.0.0:8000（§6.3）
"""

from __future__ import annotations

import argparse
import asyncio

from data import GoldTeacher
from envs.api_sandbox.task_generator import generate_tasks as api_tasks

from .mcp_server import run_stdio
from .registry import DEFAULT_TIMEOUT, ToolRegistry
from .runtime import run_agent


def _demo_agent(registry: ToolRegistry, seed: int, idx: int) -> int:
    task = api_tasks(seed, idx + 1)[idx]
    result = run_agent(task, GoldTeacher(task), registry)
    if result.error:
        print(f"[agent] 错误：{result.error}")
    else:
        print(f"[agent] 答案：{result.answer}")
    print(f"[agent] 任务={task.task_id} 类别={task.category} "
          f"步数={result.steps} 消息数={len(result.messages)}")
    return 0


def _serve_http(host: str, port: int) -> int:
    import uvicorn
    uvicorn.run(
        "serving.api:app",
        host=host,
        port=port,
        log_level="info",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="部署（§6）：MCP Server / Agent Runtime / HTTP")
    p.add_argument("--mcp", action="store_true", help="stdio 启动 MCP 工具服务器")
    p.add_argument("--http", action="store_true", help="FastAPI HTTP 服务（云端部署主入口）")
    p.add_argument("--host", default="0.0.0.0", help="HTTP 监听地址，默认 0.0.0.0")
    p.add_argument("--port", type=int, default=8000, help="HTTP 监听端口，默认 8000")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="工具调用超时秒数")
    p.add_argument("--task-seed", type=int, default=0, help="agent 演示任务 seed")
    p.add_argument("--task-idx", type=int, default=0, help="agent 演示任务下标")
    args = p.parse_args(argv)

    registry = ToolRegistry(timeout=args.timeout)

    if args.mcp:
        asyncio.run(run_stdio(registry))
        return 0
    if args.http:
        return _serve_http(args.host, args.port)

    return _demo_agent(registry, args.task_seed, args.task_idx)


if __name__ == "__main__":
    raise SystemExit(main())