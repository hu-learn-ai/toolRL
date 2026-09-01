"""GRPO 强化学习（design_zh.md §4.3）。"""

from .config import GRPOConfig
from .prompt import build_prompt, extract_task_id
from .reward import (
    RewardManager,
    compute_trajectory_reward,
    group_advantages,
    total_reward,
)
from .train import judge_for_task, load_policy, make_reward_func, train

__all__ = [
    "GRPOConfig",
    "build_prompt",
    "extract_task_id",
    "total_reward",
    "group_advantages",
    "compute_trajectory_reward",
    "RewardManager",
    "judge_for_task",
    "make_reward_func",
    "load_policy",
    "train",
]