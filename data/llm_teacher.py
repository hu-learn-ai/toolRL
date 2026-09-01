"""真实 LLM 教师（design_zh.md §3.1）：OpenAI 兼容 API 的 ReAct 教师。

与 ``GoldTeacher`` 的区别：GoldTeacher 直接按 gold 轨迹吐答案（只用于验证闭环），
``LLMTeacher`` 携带工具列表真正调用 DeepSeek / Qwen-Max，让模型自己规划工具调用，
产出"带试错、可被拒绝采样过滤"的真实轨迹，用于大规模 SFT 合成。

- 用 openai 官方 SDK（OpenAI 兼容），DeepSeek / DashScope(Qwen-Max) / vLLM 皆可；
- 文本式工具调用（非原生 function calling）：模型按 §4.1 输出 <tool_call>/<answer>，
  工具返回以 ``user`` 角色回填（跨端点最稳，规避原生 ``tool`` 角色需配对 tool_calls 的约束）。
"""

from __future__ import annotations

from .config import TeacherConfig
from .teacher import Teacher

_TOOL_PREFIX = "工具返回：\n"


def to_api_messages(messages: list[dict]) -> list[dict]:
    """把内部消息（system/user/assistant/tool 四角色）转成 OpenAI 兼容消息。

    ``tool`` 角色改写成 ``user`` 并加前缀——因为我们走文本式工具调用而非原生
    function calling，直接发 ``tool`` 角色会在多数端点报"缺少前置 assistant tool_calls"。
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "tool":
            out.append({"role": "user", "content": _TOOL_PREFIX + content})
        elif role in ("system", "user", "assistant"):
            out.append({"role": role, "content": content})
        # 其余角色忽略
    return out


class LLMTeacher(Teacher):
    """OpenAI 兼容 API 教师。``client`` 可注入假客户端用于单测（此时跳过鉴权与 SDK 导入）。"""

    def __init__(self, config: TeacherConfig | None = None, client=None):
        self._config = config or TeacherConfig.from_env()
        if client is not None:
            self._client = client
        else:
            if not self._config.api_key:
                raise ValueError("缺少 TEACHER_API_KEY（真实 teacher 需要 API key）")
            from openai import OpenAI  # 延迟导入：不装 openai 也能跑 GoldTeacher

            self._client = OpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key,
                timeout=self._config.timeout,
                max_retries=self._config.max_retries,
            )

    def generate(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._config.model,
            messages=to_api_messages(messages),
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


__all__ = ["LLMTeacher", "to_api_messages"]