"""模型加载器（部署侧复用 eval/model.py）。

把 checkpoint 路径懒加载为 ``Teacher``，并按 name 缓存以避免 FastAPI 多 worker / 多
请求重复读权重。torch/transformers 仍延迟到 ``HuggingFaceTeacher.generate`` 内部，
保证 ``import serving`` 在无 GPU 环境也不会炸。

用法::

    from serving.model_loader import get_teacher
    teacher = get_teacher("sft", path="checkpoints/sft")  # 第二次同 name 命中缓存
    text = teacher.generate(messages)
"""

from __future__ import annotations

from data.teacher import Teacher
from eval.model import HuggingFaceTeacher

_CACHE: dict[str, Teacher] = {}


def get_teacher(
    name: str,
    path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    seed: int | None = 42,
    trust_remote_code: bool = False,
) -> Teacher:
    """按 name 获取 Teacher 实例。同 name 第二次调用返回缓存（路径也一致才命中）。

    ``name`` 通常是 ``"sft"`` / ``"grpo_v3"`` / ``"base"`` / ``"teacher"``；
    ``"teacher"`` 走 ``GoldTeacher``，其余走 ``HuggingFaceTeacher``。
    """
    if name == "teacher":
        # GoldTeacher 需要 task 实例，不能缓存；返回 None 让上层自行构造
        return None  # type: ignore[return-value]

    cache_key = f"{name}:{path}"
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    teacher = HuggingFaceTeacher(
        model_name_or_path=path,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        trust_remote_code=trust_remote_code,
    )
    _CACHE[cache_key] = teacher
    return teacher


def warmup(name: str, path: str) -> None:
    """预热：服务启动时主动加载权重，避免首请求 latency 爆炸。"""
    teacher = get_teacher(name, path)
    teacher.generate([{"role": "user", "content": "hi"}])  # 触发 _load


__all__ = ["get_teacher", "warmup"]