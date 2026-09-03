"""ApiSandboxEnv：BaseToolEnv 的进程内实现（design_zh.md §2.1 / §2.2）。

把 tools.py 的 mock 工具执行、task_generator 的任务生成、judge 的分级判定拼成一个
gym 风格闭环：:

    reset(task)  → 观察（指令 + 工具 schema，JSON 可序列化）
    step(action) → 执行一次工具调用，返回 {observation, done}
    judge(轨迹)  → 委托给 judge.judge（分级奖励）

命名约定（训练循环负责转换）：Qwen3 原始输出里的 ``{"name", "arguments"}`` 在进入
step 前转成 action 的 ``{"api", "params"}``；judge 的轨迹仍是逐回合原始输出
``{"raw": str}``（与 §1.3 的 SFT 轨迹不同，见 base_env.judge）。
"""

from __future__ import annotations

from typing import Iterator

from ..base_env import BaseToolEnv, JudgeResult
from ..task_schema import Task
from .judge import judge as _judge
from .task_generator import DEFAULT_DISTRACTOR_RATIO
from .task_generator import task_generator as _task_generator
from .tools import call as _call


class ApiSandboxEnv(BaseToolEnv):
    """模拟 API 沙箱环境。"""

    def __init__(self, distractor_ratio: float = DEFAULT_DISTRACTOR_RATIO):
        self._distractor_ratio = distractor_ratio
        self._task: Task | None = None
        self._num_calls = 0
        self._done = False

    @property
    def done(self) -> bool:
        """当前回合是否已终止（达到步数上限）。"""
        return self._done

    def reset(self, task: Task) -> dict:
        """重置环境，返回初始观察（指令 + 工具 schema）。"""
        self._task = task
        self._num_calls = 0
        self._done = False
        return {
            "instruction": task.instruction,
            "tools": [t.model_dump() for t in task.tools],
        }

    def step(self, action: dict) -> dict:
        """执行一次工具调用，返回 ``{"observation": dict, "done": bool}``。

        ``done`` 在工具调用次数达到 ``max_steps - 1`` 时置真（最后一回合留给
        <answer>）；此后重复调用直接返回 ``done=True``，供训练循环安全截断。
        参数非法不终止（返回 ``{"error": ...}`` 观察，模型应学会从错误中恢复）。
        """
        if self._task is None:
            raise RuntimeError("step 之前必须先 reset(task)")
        if self._done:
            return {"observation": {}, "done": True}
        observation = _call(action.get("api"), action.get("params", {}))
        self._num_calls += 1
        self._done = self._num_calls >= self._task.max_steps - 1
        return {"observation": observation, "done": self._done}

    def judge(self, trajectory: list[dict]) -> JudgeResult:
        """委托给 judge.judge 做分级判定（需先 reset 以携带 task）。"""
        if self._task is None:
            raise RuntimeError("judge 之前必须先 reset(task)")
        return _judge(self._task, trajectory)

    def task_generator(self, seed: int) -> Iterator[Task]:
        """按 seed 确定性生成任务（干扰工具比例见 __init__）。"""
        return _task_generator(seed, self._distractor_ratio)


__all__ = ["ApiSandboxEnv"]