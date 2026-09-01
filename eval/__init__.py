"""评测（design_zh.md §5）：四组模型在 benchmark 上按 §5.2 指标对比。"""

from .benchmark import DEFAULT_SIZE, PROPORTIONS, build_benchmark
from .model import HuggingFaceTeacher, load_model_factory
from .report import render_markdown
from .runner import GroupResult, run_all, run_group

__all__ = [
    "DEFAULT_SIZE",
    "PROPORTIONS",
    "build_benchmark",
    "HuggingFaceTeacher",
    "load_model_factory",
    "GroupResult",
    "run_group",
    "run_all",
    "render_markdown",
]