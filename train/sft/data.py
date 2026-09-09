"""SFT 数据加载 / 切分 / loss 掩码（design_zh.md §1.3 / §4.2）。

读取合成好的 ``sft_trajectories.jsonl``（Trajectory），转 chat 格式，确定性切
train/val；``compute_labels`` 用"前缀 apply_chat_template"定位每个 assistant 消息的
token 区间，只对 assistant 内容计算 loss（system/user/tool 及角色标记都 mask 成 -100）。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from data.io import load_sft
from envs.task_schema import Trajectory


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """从 SFT JSONL 读回轨迹（复用 data.load_sft）。"""
    return load_sft(path)


def to_chat_messages(traj: Trajectory) -> list[dict]:
    """Trajectory.messages → OpenAI chat 格式 ``[{role, content}]``。"""
    return [{"role": m.role, "content": m.content} for m in traj.messages]


def train_val_split(
    trajs: list[Trajectory],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """确定性切分（固定种子 shuffle），val 至少 1 条（非空时）。"""
    rng = random.Random(seed)
    idx = list(range(len(trajs)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(trajs) * val_ratio))) if trajs else 0
    val_set = set(idx[:n_val])
    train = [t for i, t in enumerate(trajs) if i not in val_set]
    val = [t for i, t in enumerate(trajs) if i in val_set]
    return train, val


def compute_labels(input_ids: list[int], messages: list[dict], tokenizer) -> list[int]:
    """返回与 ``input_ids`` 等长的 labels，仅 assistant 内容 token 参与 loss。

    定位方法（Qwen 官方微调脚本同款）：对第 i 条 assistant 消息，
    ``start`` = 前缀到 i-1 加 generation prompt 的长度（模板补出的 assistant 起始标记），
    ``end``   = 前缀到 i（含）不加 generation prompt 的长度。
    两者差恰好是 assistant 内容 token 区间；角色标记本身落在 start 之前，被 mask。
    """
    labels = [-100] * len(input_ids)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        start = len(
            tokenizer.apply_chat_template(
                messages[:i], tokenize=True, add_generation_prompt=True,
                return_dict=False,
            )
        )
        end = len(
            tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=True, add_generation_prompt=False,
                return_dict=False,
            )
        )
        labels[start:end] = input_ids[start:end]
    return labels


class SFTDataCollator:
    """把 chat messages 转成 (input_ids, attention_mask, labels) 并 pad（训练用）。"""

    def __init__(self, tokenizer, max_seq_len: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: Iterable[dict]) -> dict:
        import torch  # 延迟导入：跑训练时才需要 torch

        rows = []
        for f in features:
            messages = f["messages"]
            # transformers 5.x 默认返回 BatchEncoding，这里显式要 list[int]
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False
            )
            labels = compute_labels(ids, messages, self.tokenizer)
            rows.append((ids[: self.max_seq_len], labels[: self.max_seq_len]))

        max_len = max(len(ids) for ids, _ in rows)
        input_ids, attn, labels = [], [], []
        for ids, lab in rows:
            n = len(ids)
            input_ids.append(ids + [self.pad_token_id] * (max_len - n))
            attn.append([1] * n + [0] * (max_len - n))
            labels.append(lab + [-100] * (max_len - n))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


__all__ = [
    "load_trajectories",
    "to_chat_messages",
    "train_val_split",
    "compute_labels",
    "SFTDataCollator",
]
