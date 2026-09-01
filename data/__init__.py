"""数据合成（design_zh.md §3）：teacher + 环境 → SFT 轨迹（含拒绝采样过滤）。"""

from .config import TeacherConfig
from .driver import RunStats, run_synthesis
from .filter import count_tool_calls, filter_ok
from .io import load_rl_pool, load_sft, write_rl_pool, write_sft
from .llm_teacher import LLMTeacher, to_api_messages
from .synthesis import (
    SynthesisResult,
    synthesize_dataset,
    synthesize_one,
    system_prompt,
)
from .teacher import GoldTeacher, Teacher

__all__ = [
    "Teacher",
    "GoldTeacher",
    "LLMTeacher",
    "TeacherConfig",
    "to_api_messages",
    "count_tool_calls",
    "filter_ok",
    "system_prompt",
    "SynthesisResult",
    "synthesize_one",
    "synthesize_dataset",
    "run_synthesis",
    "RunStats",
    "write_sft",
    "load_sft",
    "write_rl_pool",
    "load_rl_pool",
]