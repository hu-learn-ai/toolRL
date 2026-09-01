"""真实 teacher（LLM）配置（design_zh.md §3.1 / §3.3）。

DeepSeek / Qwen-Max 均走 OpenAI 兼容接口，配好 base_url/api_key/model 即可切换。
配置从环境变量（或项目根目录 ``.env``）读取，缺省指向 DeepSeek。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    """极简 ``.env`` 加载：不覆盖已存在的环境变量，避免引入 python-dotenv 依赖。"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass


@dataclass
class TeacherConfig:
    """teacher LLM 的接入配置，``from_env()`` 从环境变量 / .env 装配。"""

    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.7      # 拒绝采样需要多样性，但太高易出格式错误
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "TeacherConfig":
        _load_dotenv()
        return cls(
            base_url=os.getenv("TEACHER_BASE_URL", cls.base_url),
            api_key=os.getenv("TEACHER_API_KEY", ""),
            model=os.getenv("TEACHER_MODEL", cls.model),
            temperature=float(os.getenv("TEACHER_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("TEACHER_MAX_TOKENS", cls.max_tokens)),
            timeout=float(os.getenv("TEACHER_TIMEOUT", cls.timeout)),
            max_retries=int(os.getenv("TEACHER_MAX_RETRIES", cls.max_retries)),
        )


__all__ = ["TeacherConfig"]