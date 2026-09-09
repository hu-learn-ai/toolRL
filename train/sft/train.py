"""SFT 冷启动训练入口（design_zh.md §4.2）。

用 transformers.Trainer + 自实现 loss 掩码 collator 训练 Qwen3；1.7B 走 PEFT LoRA。
所有 GPU 侧依赖（torch/transformers/peft/datasets）都延迟到函数内导入，
本模块顶层不 import torch，保证不装 GPU 依赖也能 import 本包并跑单测。
"""

from __future__ import annotations

import inspect

from .config import SFTConfig
from .data import SFTDataCollator, load_trajectories, to_chat_messages, train_val_split


def load_model_tokenizer(cfg: SFTConfig):
    """加载 Qwen3 模型与 tokenizer，use_lora 时套 PEFT LoRA。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code
    )
    torch_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch_dtype, trust_remote_code=cfg.trust_remote_code
    )
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    return model, tokenizer


def _to_dataset(trajs, tokenizer):
    """把轨迹列表转成 datasets.Dataset（字段 ``messages`` = chat 格式列表）。"""
    from datasets import Dataset

    return Dataset.from_list(
        [{"messages": to_chat_messages(t)} for t in trajs]
    )


def train(cfg: SFTConfig) -> None:
    """按 §4.2 超参跑 SFT，产出 checkpoint 到 cfg.output_dir。"""
    from transformers import Trainer, TrainingArguments

    trajs = load_trajectories(cfg.data_path)
    train_trajs, val_trajs = train_val_split(trajs, cfg.val_ratio, cfg.seed)
    if not train_trajs:
        raise RuntimeError(f"训练集为空：{cfg.data_path} 里没有可用轨迹")

    model, tokenizer = load_model_tokenizer(cfg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = _to_dataset(train_trajs, tokenizer)
    val_ds = _to_dataset(val_trajs, tokenizer)

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    args_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        # collator 需要原始 messages 列自行转 token，阻止 Trainer 预先删列
        remove_unused_columns=False,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        bf16=cfg.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=cfg.seed,
        report_to=cfg.report_to,
    )
    # transformers 5.x 部分版本移除了 warmup_ratio 字段，只在该参数存在时传入，
    # 兼容 4.x 与 5.x（缺失时按无 warmup 处理，不影响训练启动）。
    if "warmup_ratio" in ta_params:
        args_kwargs["warmup_ratio"] = cfg.warmup_ratio
    args = TrainingArguments(**args_kwargs)

    # transformers 5.x 把 tokenizer 参数改名为 processing_class，按版本兼容。
    trainer_kwargs = {"data_collator": SFTDataCollator(tokenizer, cfg.max_seq_len)}
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **trainer_kwargs,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


__all__ = ["load_model_tokenizer", "train"]
