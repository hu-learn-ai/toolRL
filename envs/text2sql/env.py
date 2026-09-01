"""Text2SQLEnv：BaseToolEnv 实现（design_zh.md §2.3）。"""

from __future__ import annotations

from typing import Iterator

from ..base_env import BaseToolEnv, JudgeResult
from ..task_schema import Task
from .db import Database, SQLBlocked, build_database
from .judge import judge as _judge
from .task_generator import task_generator as _task_generator


class Text2SQLEnv(BaseToolEnv):
    """Text2SQL 环境：单步 SQL 查询，成功与否看结果集一致性。"""

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout
        self._task: Task | None = None
        self._db: Database | None = None
        self._num_calls = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, task: Task) -> dict:
        """重置环境，返回观察（指令 + 工具 schema + 数据库列结构）。"""
        self._task = task
        self._db = build_database(task.meta["db"])
        self._num_calls = 0
        self._done = False
        return {
            "instruction": task.instruction,
            "tools": [t.model_dump() for t in task.tools],
            "schema": self._db.schema_ddl(),
        }

    def step(self, action: dict) -> dict:
        """执行一次 SQL 查询，返回 ``{"observation": {columns, rows}|{error}, "done": bool}``。"""
        if self._task is None or self._db is None:
            raise RuntimeError("step 之前必须先 reset(task)")
        if self._done:
            return {"observation": {}, "done": True}
        query = (action.get("params") or {}).get("query")
        try:
            columns, rows = self._db.execute(query)
            observation = {"columns": columns, "rows": [list(r) for r in rows]}
        except SQLBlocked as e:
            observation = {"error": str(e)}
        except Exception as e:
            observation = {"error": f"SQL 执行失败: {e}"}
        self._num_calls += 1
        self._done = self._num_calls >= self._task.max_steps - 1
        return {"observation": observation, "done": self._done}

    def judge(self, trajectory: list[dict]) -> JudgeResult:
        if self._task is None or self._db is None:
            raise RuntimeError("judge 之前必须先 reset(task)")
        return _judge(self._task, trajectory, self._db)

    def task_generator(self, seed: int) -> Iterator[Task]:
        return _task_generator(seed)


__all__ = ["Text2SQLEnv"]