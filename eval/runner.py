"""评测执行器（design_zh.md §5.3）。

复用 ``train.sft.eval.evaluate_tasks_detailed``（指标计算）与 ``train.grpo.RewardManager``
的加权核对四组模型在同一个 benchmark 上跑 rollout + judge，返回每组 ``EvalMetrics``
与全量逐任务结果（含失败 case 清单）。

加权口径与 GRPO 训练一致：默认权取 ``RewardManager(env).weights``（即
``base_env.DEFAULT_REWARD_WEIGHTS``），保证评测的 ``reward_mean`` 与 RL 奖励同源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from data.teacher import Teacher
from envs.api_sandbox import ApiSandboxEnv
from envs.base_env import BaseToolEnv
from envs.task_schema import Task
from train.grpo.reward import RewardManager
from train.sft.eval import EvalMetrics, TaskResult, evaluate_tasks_detailed

TeacherFactory = Callable[[Task], Teacher]


@dataclass
class GroupResult:
    metrics: EvalMetrics
    results: list[TaskResult]   # 全量逐任务结果（含成功与失败）

    @property
    def failures(self) -> list[TaskResult]:
        return [r for r in self.results if not r.success]


def run_group(
    teacher_factory: TeacherFactory,
    tasks: list[Task],
    env: BaseToolEnv,
    weights: dict[str, float] | None = None,
) -> GroupResult:
    """对一组"模型"（teacher_factory）在给定任务集上评测，返回指标 + 逐任务结果。"""
    if weights is None:
        weights = RewardManager(env).weights  # 与 GRPO 训练同权
    metrics, results = evaluate_tasks_detailed(teacher_factory, tasks, env, weights)
    return GroupResult(metrics=metrics, results=results)


def run_all(
    groups: dict[str, TeacherFactory],
    tasks: list[Task],
    env: BaseToolEnv | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, GroupResult]:
    """多组模型在同一 benchmark 上评测，返回 ``{组名: GroupResult}``（保持插入序）。"""
    env = env or ApiSandboxEnv()
    return {name: run_group(factory, tasks, env, weights) for name, factory in groups.items()}


__all__ = ["GroupResult", "run_group", "run_all"]