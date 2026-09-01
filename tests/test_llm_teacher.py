"""真实 LLM 教师与大规模合成驱动测试（design_zh.md §3）。

LLMTeacher 用注入的假客户端测试（不触发 openai SDK 导入、不发起真实请求）；
run_synthesis 用 GoldTeacher 测试确定性统计与断点续跑。
"""

import itertools
from types import SimpleNamespace

import pytest

from data import (
    GoldTeacher,
    LLMTeacher,
    TeacherConfig,
    run_synthesis,
    to_api_messages,
)
from envs.api_sandbox import ApiSandboxEnv

# ---------------------------------------------------------------------------
# 假 OpenAI 客户端
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def _resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


# ---------------------------------------------------------------------------
# to_api_messages
# ---------------------------------------------------------------------------


def test_to_api_messages_maps_tool_to_user():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "<tool_call>x</tool_call>"},
        {"role": "tool", "content": '{"ok": 1}'},
        {"role": "weird", "content": "drop"},
    ]
    out = to_api_messages(msgs)
    assert out[0] == {"role": "system", "content": "s"}
    assert out[3] == {"role": "user", "content": '工具返回：\n{"ok": 1}'}
    assert len(out) == 4  # weird 被丢弃


# ---------------------------------------------------------------------------
# LLMTeacher
# ---------------------------------------------------------------------------


def test_llm_teacher_returns_content():
    client = FakeClient([_resp('<tool_call>{"name":"x","arguments":{}}</tool_call>')])
    t = LLMTeacher(TeacherConfig(model="m"), client=client)
    out = t.generate([{"role": "user", "content": "hi"}])
    assert out == '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    req = client.requests[0]
    assert req["model"] == "m"
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" in req and "max_tokens" in req


def test_llm_teacher_strips_whitespace():
    client = FakeClient([_resp("   ")])
    t = LLMTeacher(TeacherConfig(model="m"), client=client)
    assert t.generate([{"role": "user", "content": "hi"}]) == ""


def test_llm_teacher_requires_api_key():
    with pytest.raises(ValueError):
        LLMTeacher(TeacherConfig(api_key=""))


# ---------------------------------------------------------------------------
# TeacherConfig.from_env
# ---------------------------------------------------------------------------


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TEACHER_BASE_URL", "http://x")
    monkeypatch.setenv("TEACHER_API_KEY", "sk")
    monkeypatch.setenv("TEACHER_MODEL", "qwen-max")
    monkeypatch.setenv("TEACHER_TEMPERATURE", "0.3")
    monkeypatch.setenv("TEACHER_MAX_TOKENS", "512")
    monkeypatch.setenv("TEACHER_TIMEOUT", "30")
    monkeypatch.setenv("TEACHER_MAX_RETRIES", "5")
    cfg = TeacherConfig.from_env()
    assert cfg.base_url == "http://x"
    assert cfg.api_key == "sk"
    assert cfg.model == "qwen-max"
    assert cfg.temperature == 0.3
    assert cfg.max_tokens == 512
    assert cfg.timeout == 30.0
    assert cfg.max_retries == 5


# ---------------------------------------------------------------------------
# run_synthesis（用 GoldTeacher 保证确定性）
# ---------------------------------------------------------------------------


def test_run_synthesis_writes_and_counts(tmp_path):
    env = ApiSandboxEnv()
    n = 3
    tasks = list(itertools.islice(env.task_generator(0), n))
    expected_calls = sum(t.min_steps for t in tasks)

    stats = run_synthesis([env], GoldTeacher, n_per_env=n, out_dir=tmp_path, seed=0, resume=False)

    assert len(stats.results) == n  # 黄金轨迹全部通过过滤
    assert stats.calls == expected_calls
    sft = (tmp_path / "sft_trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rl = (tmp_path / "rl_pool.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sft) == n
    assert len(rl) == n


def test_run_synthesis_resume_skips_done(tmp_path):
    env = ApiSandboxEnv()
    n = 3
    s1 = run_synthesis([env], GoldTeacher, n_per_env=n, out_dir=tmp_path, seed=0, resume=False)
    assert len(s1.results) == n

    s2 = run_synthesis([env], GoldTeacher, n_per_env=n, out_dir=tmp_path, seed=0, resume=True)
    assert len(s2.results) == 0
    assert s2.skipped == n
    assert s2.calls == 0
    # 文件不被重复追加
    sft = (tmp_path / "sft_trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sft) == n


def test_run_synthesis_max_calls_zero(tmp_path):
    env = ApiSandboxEnv()
    stats = run_synthesis([env], GoldTeacher, n_per_env=5, out_dir=tmp_path, seed=0, resume=False, max_calls=0)
    assert stats.calls == 0
    assert stats.results == []