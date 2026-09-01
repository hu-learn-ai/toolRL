"""SFT 验证指标（design_zh.md §4.2 验证目标 / §5.2 指标定义）。

复用 ``data.synthesize_one``：把待验证模型包装成 Teacher 跑 rollout + judge，再聚合
§5.2 六个指标：

- 格式合规率（judge.format == 1.0）；
- 工具选择准确率（首步 API 与 gold 首步一致）；
- 参数正确率（judge.matched_params / judge.param_total 全任务聚合）；
- 端到端成功率（judge.success）；
- 平均步数（trajectory.steps 均值）；
- 平均 token（assistant 生成文本的粗略估算，评估部署成本）。

``reward_mean`` 是 RL 口径加权奖励（复用 ``train.grpo.reward.total_reward``，与 GRPO
的 ``RewardManager`` 同一加权核），用于跨组横向对比。

无 torch 依赖，可用 GoldTeacher 或 mock 生成器做确定性单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterable

from data.synthesis import SynthesisResult, synthesize_one
from data.teacher import Teacher
from envs.api_sandbox.judge import parse_turn
from envs.base_env import BaseToolEnv
from envs.task_schema import Task, Trajectory
from train.grpo.reward import total_reward

TeacherFactory = Callable[[Task], Teacher]


@dataclass
class EvalMetrics:
    n: int
    format_rate: float      # 格式合规率
    tool_acc: float         # 工具选择准确率（首步 API 匹配）
    param_acc: float        # 参数正确率（matched_params / param_total 聚合）
    success_rate: float     # 端到端成功率
    avg_steps: float        # 平均步数
    avg_tokens: float       # 平均生成 token（估算）
    reward_mean: float      # RL 口径加权奖励均值


@dataclass
class TaskResult:
    """单任务评测结果（失败 case 清单 / 分维度报告用）。"""

    task_id: str
    instruction: str
    category: str
    success: bool
    format_ok: bool
    reward: float
    dim: str = ""          # 所属评测维度（bench_dim，缺省回退 category）


def first_tool(trajectory: Trajectory) -> str | None:
    """轨迹中第一个 assistant tool_call 的工具名；无则 None。"""
    for m in trajectory.messages:
        if m.role != "assistant":
            continue
        p = parse_turn(m.content)
        if p.kind == "tool_call":
            return p.tool_name
    return None


def estimate_tokens(text: str) -> int:
    """无 tokenizer 的粗略 token 估算：CJK 按 1 字符、ASCII 按 4 字符折 1 token。

    仅用于 §5.2 的部署成本量级对比，非精确 tokenize。
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = len(text) - ascii_chars
    return cjk_chars + (ascii_chars + 3) // 4


class _Accumulator:
    """逐任务累加 §5.2 指标，最后归一出 EvalMetrics。"""

    def __init__(self) -> None:
        self.n = 0
        self.fmt = 0
        self.tool = 0
        self.succ = 0
        self.matched_params = 0
        self.param_total = 0
        self.steps_sum = 0
        self.tokens_sum = 0
        self.reward_sum = 0.0

    def add(self, task: Task, r: SynthesisResult, weights: dict | None) -> None:
        self.n += 1
        self.steps_sum += r.trajectory.steps
        self.fmt += 1 if r.judge.format == 1.0 else 0
        self.succ += 1 if r.judge.success else 0
        self.tool += 1 if first_tool(r.trajectory) == task.gold.calls[0].api else 0
        self.matched_params += r.judge.matched_params
        self.param_total += r.judge.param_total
        self.reward_sum += total_reward(r.judge, weights)
        self.tokens_sum += sum(
            estimate_tokens(m.content)
            for m in r.trajectory.messages
            if m.role == "assistant"
        )

    def metrics(self) -> EvalMetrics:
        n = self.n or 1
        return EvalMetrics(
            n=self.n,
            format_rate=self.fmt / n,
            tool_acc=self.tool / n,
            param_acc=self.matched_params / self.param_total if self.param_total else 0.0,
            success_rate=self.succ / n,
            avg_steps=self.steps_sum / n,
            avg_tokens=self.tokens_sum / n,
            reward_mean=self.reward_sum / n,
        )


def evaluate_tasks_detailed(
    teacher_factory: TeacherFactory,
    tasks: Iterable[Task],
    env: BaseToolEnv,
    weights: dict[str, float] | None = None,
) -> tuple[EvalMetrics, list[TaskResult]]:
    """对显式任务列表跑 rollout + judge，返回聚合指标与逐任务结果（失败 case 清单）。"""
    acc = _Accumulator()
    results: list[TaskResult] = []
    for task in tasks:
        r = synthesize_one(task, env, teacher_factory(task))
        acc.add(task, r, weights)
        results.append(
            TaskResult(
                task_id=task.task_id,
                instruction=task.instruction,
                category=task.category,
                success=r.judge.success,
                format_ok=r.judge.format == 1.0,
                reward=total_reward(r.judge, weights),
                dim=task.meta.get("bench_dim", task.category),
            )
        )
    return acc.metrics(), results


def evaluate_tasks(
    teacher_factory: TeacherFactory,
    tasks: Iterable[Task],
    env: BaseToolEnv,
    weights: dict[str, float] | None = None,
) -> EvalMetrics:
    """对显式任务列表聚合指标（丢弃逐任务结果）。"""
    metrics, _ = evaluate_tasks_detailed(teacher_factory, tasks, env, weights)
    return metrics


def evaluate(
    teacher_factory: TeacherFactory,
    envs: Iterable[BaseToolEnv],
    n_per_env: int,
    seed: int = 0,
    weights: dict[str, float] | None = None,
) -> EvalMetrics:
    """对给定"模型"（teacher_factory 产出的 Teacher）在 envs 上采样并聚合指标。"""
    acc = _Accumulator()
    for i, env in enumerate(envs):
        for task in islice(env.task_generator(seed + i), n_per_env):
            acc.add(task, synthesize_one(task, env, teacher_factory(task)), weights)
    return acc.metrics()


__all__ = [
    "EvalMetrics",
    "TaskResult",
    "first_tool",
    "estimate_tokens",
    "evaluate",
    "evaluate_tasks",
    "evaluate_tasks_detailed",
]