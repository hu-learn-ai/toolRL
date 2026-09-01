"""任务环境统一接口（design_zh.md §2.1）。

两个环境（api_sandbox / text2sql）都实现 BaseToolEnv，让训练/评测端与具体环境解耦。
环境不感知模型与训练框架：训练端通过 HTTP/进程内调用 ``step`` 完成 rollout，
评测端在采样后批量调用 ``judge``。
"""

from __future__ import annotations

import abc
from typing import Iterator

from pydantic import BaseModel, Field

from .task_schema import Task

# design_zh.md §2.4：环境端只输出原始分项，训练端组装加权总奖励。
#   R_total = w1·R_format + w2·R_correct + w3·R_answer − w4·R_steps
# 权重为初始值，W5 奖励消融以它为中心做 ± 调整。
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "format": 0.40,   # 格式合规：轨迹全程 JSON schema 合法
    "correct": 0.30,  # 调用正确：工具选择 + 参数逐项比对（部分得分）
    "answer": 0.20,   # 最终答案正确
    "steps": 0.10,    # 步数惩罚（负权，抑制穷举试错）
}


class JudgeResult(BaseModel):
    """``judge`` 的返回值：原始奖励分项 + 是否正确（design_zh.md §2.4）。

    各分项应先归一化到 [0, 1]（``steps`` 为 [0, +∞) 的惩罚项），由训练端加权求和。
    """

    format: float = Field(ge=0.0, le=1.0)   # R_format 格式合规
    correct: float = Field(ge=0.0, le=1.0)  # R_correct 工具 + 参数正确
    answer: float = Field(ge=0.0, le=1.0)   # R_answer 最终答案正确
    steps: float = Field(ge=0.0)            # R_steps = max(0, steps−min_steps)/max_steps
    success: bool = False                   # 端到端是否成功（评测用）
    matched_params: int = Field(default=0, ge=0, description="参数正确项数（评测 §5.2 参数正确率聚合用）")
    param_total: int = Field(default=0, ge=0, description="gold 参数总项数（评测 §5.2 参数正确率聚合用）")


class BaseToolEnv(abc.ABC):
    """工具调用环境的统一抽象接口（gym 风格）。"""

    @abc.abstractmethod
    def reset(self, task: Task) -> dict:
        """重置环境，返回初始观察（工具列表 + 指令）。"""

    @abc.abstractmethod
    def step(self, action: dict) -> dict:
        """执行一步工具调用。

        ``action = {"api": str, "params": dict}``
        返回 ``{"observation": dict, "done": bool}``。
        """

    @abc.abstractmethod
    def judge(self, trajectory: list[dict]) -> JudgeResult:
        """程序化判定，返回各奖励分项与是否正确（不做加权）。

        ``trajectory`` 为模型逐回合的原始输出序列，每个元素形如 ``{"raw": str}``
        （含 <tool_call>/<answer> 标签），与 §1.3 的 SFT 轨迹（``Trajectory``）不是
        同一个概念 —— 格式判定需要原始文本，光有 ``{action, observation}`` 不够。
        """

    @abc.abstractmethod
    def task_generator(self, seed: int) -> Iterator[Task]:
        """参数化批量生成任务，固定种子保证可复现。"""
