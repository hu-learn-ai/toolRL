"""统一任务 JSON schema（design_zh.md §1.2 / §1.3）。

环境生成、数据合成、训练、评测四端共用同一任务格式，保证
"环境 → 数据 → 训练 → 评测" 闭环无需转换。本模块用 pydantic 建模该契约，
任何一端从 JSONL 读取任务时统一走 ``Task.model_validate(...)``。

注意：这里只定义"契约"（数据长什么样），不含任何执行/判定逻辑。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ToolParameterSchema(BaseModel):
    """工具参数 schema（OpenAI Function Calling 的 object 根类型子集）。"""

    type: Literal["object"] = "object"
    properties: dict[str, dict[str, Any]] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """单个工具定义，与任务 JSON 的 ``tools[]`` 元素对应（design_zh.md §1.2）。"""

    name: str
    description: str
    parameters: ToolParameterSchema = Field(default_factory=ToolParameterSchema)


class ToolCall(BaseModel):
    """一次工具调用：``gold.calls`` 的元素，也是 rollout 中 ``action`` 的形状。"""

    api: str
    params: dict[str, Any] = Field(default_factory=dict)


class Gold(BaseModel):
    """程序化判定的唯一依据（design_zh.md §1.2：不允许人工/LLM 裁判打分）。"""

    calls: list[ToolCall]
    answer: str


class Task(BaseModel):
    """统一任务对象（design_zh.md §1.2 的 JSON 1:1 建模）。"""

    task_id: str
    category: str
    instruction: str
    tools: list[ToolSpec]
    gold: Gold
    meta: dict[str, Any] = Field(default_factory=dict, description="环境自留附加信息（如 Text2SQL 的 db 名），训练端一般不使用")
    max_steps: int = Field(ge=1, description="环境执行步数上限")
    min_steps: int = Field(ge=1, description="最优步数，步数惩罚依赖它")

    @model_validator(mode="after")
    def _check_steps(self) -> "Task":
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps 不能大于 max_steps")
        return self


class Message(BaseModel):
    """轨迹中的一条消息（design_zh.md §1.3 ``messages[]``）。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Trajectory(BaseModel):
    """SFT 轨迹（design_zh.md §1.3 JSONL 的一行）。"""

    task_id: str
    messages: list[Message]
    success: bool
    steps: int = Field(ge=0)
