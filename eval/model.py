"""评测用模型加载器（design_zh.md §5.3）。

把本地 checkpoint（基座 / SFT / GRPO）包装成 ``Teacher``，供 ``evaluate_tasks`` 复用
``data.synthesize_one`` 的 rollout 闭环。教师组（§5.3 上限参照）直接用 ``GoldTeacher``，
不经过此模块。

GPU 侧依赖（torch/transformers）延迟到 ``HuggingFaceTeacher.generate`` 内导入，
模块顶层无 torch，保证无 GPU 环境也能 import 并单测 benchmark/runner/report。
解码参数固定 ``temperature=0.7, top_p=0.9``（§5.3 口径），随机种子由调用方通过
``model_kwargs["seed"]`` 传入以保证可复现。
"""

from __future__ import annotations

from data.teacher import Teacher


class HuggingFaceTeacher(Teacher):
    """把 HuggingFace 因果模型包装成 Teacher：``generate(messages)`` 返回原始文本。"""

    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
        seed: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name_or_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        # 安全：是否信任模型仓库的自定义建模代码，默认关闭（见 _load 注释）
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # trust_remote_code 会执行仓库内的自定义 Python 代码，仅对官方可信权重放行
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    def generate(self, messages: list[dict]) -> str:
        import torch

        if self._model is None:
            self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen_kwargs: dict = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        if self.seed is not None:
            # 新版 transformers 的 generate() 不接受 generator= kwarg（会被
            # _validate_model_kwargs 拒绝），改用全局播种保证同 seed 可复现
            torch.manual_seed(self.seed)
        out = self._model.generate(**inputs, **gen_kwargs)
        new = out[0][inputs["input_ids"].shape[-1]:]
        # skip_special_tokens=False + 手动剥离终止符：对 Qwen3-0.6B 实测，
        # <tool_call>/<think> 的 special=False（skip=True 也不会剥掉它们），但
        # 显式保留 + 手动去 <|im_end|> 的写法不依赖 tokenizer 版本行为，更稳。
        text = self._tokenizer.decode(new, skip_special_tokens=False)
        for stop in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(stop, "")
        return text.strip()


def load_model_factory(
    model_name_or_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    seed: int | None = None,
    trust_remote_code: bool = False,
):
    """返回一个 ``TeacherFactory``（``Task -> HuggingFaceTeacher``），供评测复用。

    ``HuggingFaceTeacher.generate`` 不依赖具体任务，故在闭包外只构造一个实例，让同一组
    内所有任务共享同一份已加载的模型——避免 400 条 benchmark 逐任务重复从磁盘加载权重。
    """
    teacher = HuggingFaceTeacher(
        model_name_or_path,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        trust_remote_code=trust_remote_code,
    )

    def factory(task):
        return teacher

    return factory


__all__ = ["HuggingFaceTeacher", "load_model_factory"]