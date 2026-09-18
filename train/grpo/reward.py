"""GRPO 奖励函数（design_zh.md §2.4 / §4.3）。

复用 ``envs.api_sandbox.judge``（或 text2sql.judge）产出的 ``JudgeResult``，按
``base_env.DEFAULT_REWARD_WEIGHTS`` 加权组装标量奖励：

    R_total = w_format·R_format + w_correct·R_correct + w_answer·R_answer − w_steps·R_steps

各分项由 judge 先归一化到 [0,1]（steps 为惩罚项），满足 §4.3"归一化后再加权"的注意点。
组内相对优势 ``A_i = (r_i − mean) / std``（DeepSeekMath），std=0 时优势置 0（防除零）。

rollout 复用 ``data.synthesize_one``：把策略包装成 Teacher 跑满工具闭环后由 env.judge 判定。
"""

from __future__ import annotations

from data.synthesis import synthesize_one
from data.teacher import Teacher
from envs.base_env import DEFAULT_REWARD_WEIGHTS, BaseToolEnv, JudgeResult
from envs.task_schema import Task


def total_reward(jr: JudgeResult, weights: dict[str, float] | None = None) -> float:
    """把 JudgeResult 四个分项加权成标量奖励（§2.4）。"""
    w = weights or DEFAULT_REWARD_WEIGHTS
    return (
        w["format"] * jr.format
        + w["correct"] * jr.correct
        + w["answer"] * jr.answer
        - w["steps"] * jr.steps
    )


def group_advantages(rewards: list[float]) -> list[float]:
    """组内相对优势（DeepSeekMath）：A_i = (r_i − mean) / std；std≈0 → 全 0。"""
    if not rewards:
        return []
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    # 容差判定：避免浮点 sqrt 微小误差时被除零路径跳过；同时防止未来 std=1e-300
    # 这种诡异值进入除法分支。
    if std < 1e-9:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def compute_trajectory_reward(
    env: BaseToolEnv,
    task: Task,
    teacher: Teacher,
    weights: dict[str, float] | None = None,
) -> float:
    """跑一条完整 rollout（env loop + judge）并返回加权总奖励。"""
    r = synthesize_one(task, env, teacher)
    return total_reward(r.judge, weights)


class RewardManager:
    """GRPO 奖励管理器（§4.3）：跑 rollout 返回奖励 + 组内优势。

    verl 侧 reward manager 的核心即 ``score_one``；``score_group`` 对同一 prompt 采样
    G 条响应并组内标准化，直接对应 §4.3 的 group_size=8 采样流程。
    """

    def __init__(self, env: BaseToolEnv, weights: dict[str, float] | None = None):
        self.env = env
        self.weights = weights or DEFAULT_REWARD_WEIGHTS

    def score_one(self, task: Task, teacher: Teacher) -> float:
        return compute_trajectory_reward(self.env, task, teacher, self.weights)

    def score_group(self, task: Task, teacher_factory, group_size: int = 8) -> list[float]:
        """同一 prompt 采样 group_size 条（策略随机采样），返回各条奖励。"""
        return [self.score_one(task, teacher_factory(task)) for _ in range(group_size)]

    def advantages(self, task: Task, teacher_factory, group_size: int = 8) -> list[float]:
        return group_advantages(self.score_group(task, teacher_factory, group_size))


def first_turn_reward(
    task: Task, completion: str, weights: dict[str, float] | None = None
) -> float:
    """TRL 单轮路径专用奖励：只评第一回合 tool_call，不要求轨迹终态 ``<answer>``。

    为什么不能用完整 judge：``_judge_format`` 要求轨迹恰好以一个 ``<answer>`` 收尾，
    单轮 completion 永远给不出（工具结果要等环境回传）。若沿用完整 judge：
    - 正确的 tool_call 上限只有 ``w_correct = 0.3``（format/answer 恒 0）；
    - 而"跳过工具、直接输出含 gold 事实的 ``<answer>``"反而可得 ``0.4 + 0.2 = 0.6``
      ——负向激励（reward hacking 向量），加大 lr 后策略会学会不调工具。

    单轮路径改为：格式（合法 tool_call + schema 校验通过）+ 首步对齐（API 名与
    参数对 gold 调用逐条比对、分级给分），answer 维度一律 0。上限 0.7。
    完整的多轮闭环判定仍由 verl 侧 ``RewardManager`` 承担。
    """
    from envs.api_sandbox.common import validate_params
    from envs.api_sandbox.judge import _judge_correct, _schema_of, parse_turn

    w = weights or DEFAULT_REWARD_WEIGHTS
    p = parse_turn(completion)
    if p.kind != "tool_call":
        # answer / invalid 在单轮路径一律 0 分：第一回合就该调工具
        return 0.0
    schema = _schema_of(p.tool_name)
    fmt = (
        1.0
        if (schema is not None and validate_params(p.arguments, schema) is None)
        else 0.0
    )
    r_correct, _, _ = _judge_correct(task, [p])
    return w["format"] * fmt + w["correct"] * r_correct


__all__ = [
    "total_reward",
    "first_turn_reward",
    "group_advantages",
    "compute_trajectory_reward",
    "RewardManager",
]