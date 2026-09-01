"""数据合成测试（design_zh.md §3）：teacher + 环境 → SFT 轨迹 + 拒绝采样过滤。"""

import json

from data import (
    GoldTeacher,
    count_tool_calls,
    filter_ok,
    load_sft,
    synthesize_dataset,
    synthesize_one,
    system_prompt,
    write_rl_pool,
    write_sft,
)
from envs.api_sandbox import ApiSandboxEnv
from envs.api_sandbox.task_generator import generate_tasks as api_tasks
from envs.base_env import JudgeResult
from envs.task_schema import Message, Trajectory
from envs.text2sql import Text2SQLEnv
from envs.text2sql import generate_tasks as sql_tasks


def _assistant(text: str) -> Message:
    return Message(role="assistant", content=text)


def _tool_msg(name: str) -> Message:
    return Message(role="tool", content=json.dumps({"ok": True}))


# ---------------------------------------------------------------------------
# GoldTeacher
# ---------------------------------------------------------------------------


def test_gold_teacher_emits_calls_then_answer():
    task = api_tasks(0, 1)[0]
    t = GoldTeacher(task)
    turns = [t.generate([]) for _ in range(task.min_steps + 1)]
    # 前 min_steps-1 回合是 tool_call，最后一回合是 <answer>
    assert all("<tool_call>" in x for x in turns[: task.min_steps - 1])
    assert "<answer>" in turns[task.min_steps - 1]


# ---------------------------------------------------------------------------
# system_prompt
# ---------------------------------------------------------------------------


def test_system_prompt_without_schema():
    obs = {"tools": [{"name": "weather.query", "description": "d", "parameters": {}}]}
    p = system_prompt(obs)
    assert "weather.query" in p
    assert "<tool_call>" in p and "<answer>" in p
    assert "数据库表结构" not in p


def test_system_prompt_with_schema():
    obs = {"tools": [], "schema": "CREATE TABLE users(id INT);"}
    p = system_prompt(obs)
    assert "数据库表结构" in p
    assert "CREATE TABLE users" in p


# ---------------------------------------------------------------------------
# synthesize_one（黄金轨迹必成功）
# ---------------------------------------------------------------------------


def test_synthesize_api_sandbox_gold_succeeds():
    env = ApiSandboxEnv()
    task = api_tasks(0, 1)[0]
    r = synthesize_one(task, env, GoldTeacher(task))
    assert r.judge.success is True
    assert r.trajectory.success is True
    assert r.trajectory.steps == task.min_steps


def test_synthesize_text2sql_gold_succeeds():
    env = Text2SQLEnv()
    task = sql_tasks(0, 1)[0]
    r = synthesize_one(task, env, GoldTeacher(task))
    assert r.judge.success is True
    assert r.trajectory.success is True
    assert r.trajectory.steps == task.min_steps


def test_synthesize_messages_have_four_roles():
    env = ApiSandboxEnv()
    task = api_tasks(0, 1)[0]
    r = synthesize_one(task, env, GoldTeacher(task))
    roles = [m.role for m in r.trajectory.messages]
    assert roles[0] == "system" and roles[1] == "user"
    assert "tool" in roles and "assistant" in roles


# ---------------------------------------------------------------------------
# count_tool_calls
# ---------------------------------------------------------------------------


def test_count_tool_calls():
    msgs = [
        _assistant('<tool_call>{"name":"weather.query","arguments":{"city":"北京"}}</tool_call>'),
        _assistant('<tool_call>{"name":"weather.query","arguments":{"city":"上海"}}</tool_call>'),
        _assistant('<tool_call>{"name":"meeting.create","arguments":{"title":"x"}}</tool_call>'),
        _assistant("<answer>ok</answer>"),
    ]
    counts = count_tool_calls(msgs)
    assert counts == {"weather.query": 2, "meeting.create": 1}


# ---------------------------------------------------------------------------
# filter_ok（拒绝采样硬性条件）
# ---------------------------------------------------------------------------


def _traj(messages, steps):
    return Trajectory(task_id="t", messages=messages, success=True, steps=steps)


def _judge(success=True, answer=1.0):
    return JudgeResult(format=1.0, correct=1.0, answer=answer, steps=0.0, success=success)


def test_filter_ok_passes_gold():
    env = ApiSandboxEnv()
    task = api_tasks(0, 1)[0]
    r = synthesize_one(task, env, GoldTeacher(task))
    assert filter_ok(r.task, r.trajectory, r.judge) is True


def test_filter_ok_rejects_failure():
    env = ApiSandboxEnv()
    task = api_tasks(0, 1)[0]
    r = synthesize_one(task, env, GoldTeacher(task))
    bad = _judge(success=False)
    assert filter_ok(task, r.trajectory, bad) is False


def test_filter_ok_rejects_out_of_range_steps():
    task = api_tasks(0, 1)[0]
    msgs = [_assistant('<tool_call>{"name":"weather.query","arguments":{"city":"北京"}}</tool_call>'),
            _tool_msg("weather.query"), _assistant("<answer>ok</answer>")]
    ok = _judge()
    assert filter_ok(task, _traj(msgs, steps=task.max_steps + 1), ok) is False  # 超 max
    assert filter_ok(task, _traj(msgs, steps=task.min_steps - 1), ok) is False  # 低于 min


def test_filter_ok_rejects_repeated_tool():
    task = api_tasks(0, 1)[0]
    msgs = [
        _assistant('<tool_call>{"name":"weather.query","arguments":{"city":"北京"}}</tool_call>'),
        _tool_msg("weather.query"),
        _assistant('<tool_call>{"name":"weather.query","arguments":{"city":"上海"}}</tool_call>'),
        _tool_msg("weather.query"),
        _assistant('<tool_call>{"name":"weather.query","arguments":{"city":"广州"}}</tool_call>'),
        _tool_msg("weather.query"),
        _assistant("<answer>ok</answer>"),
    ]
    ok = _judge()
    assert filter_ok(task, _traj(msgs, steps=4), ok) is False  # 同工具 3 次


def test_filter_ok_rejects_bad_answer():
    task = api_tasks(0, 1)[0]
    msgs = [_assistant('<tool_call>{"name":"weather.query","arguments":{"city":"北京"}}</tool_call>'),
            _tool_msg("weather.query"), _assistant("<answer>ok</answer>")]
    bad = _judge(answer=0.5)
    assert filter_ok(task, _traj(msgs, steps=2), bad) is False


# ---------------------------------------------------------------------------
# synthesize_dataset + 落盘
# ---------------------------------------------------------------------------


def test_synthesize_dataset_all_gold_pass():
    results = synthesize_dataset([ApiSandboxEnv(), Text2SQLEnv()], n_per_env=5, seed=0)
    assert len(results) == 10  # 黄金轨迹全部通过过滤
    assert all(r.judge.success for r in results)


def test_synthesize_dataset_deterministic():
    a = synthesize_dataset([ApiSandboxEnv()], n_per_env=5, seed=7)
    b = synthesize_dataset([ApiSandboxEnv()], n_per_env=5, seed=7)
    assert [r.task.task_id for r in a] == [r.task.task_id for r in b]


def test_write_and_load_sft_roundtrip(tmp_path):
    results = synthesize_dataset([ApiSandboxEnv()], n_per_env=3, seed=0)
    path = tmp_path / "sft.jsonl"
    n = write_sft(results, path)
    assert n == 3
    loaded = load_sft(path)
    assert [t.task_id for t in loaded] == [r.task.task_id for r in results]
    assert loaded[0].success is True


def test_write_rl_pool(tmp_path):
    tasks = api_tasks(0, 4)
    path = tmp_path / "rl.jsonl"
    n = write_rl_pool(tasks, path)
    assert n == 4
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    first = json.loads(lines[0])
    # 完整 Task：含 gold/meta/max_steps/min_steps，奖励函数据此判定
    assert set(first) == {"task_id", "category", "instruction", "tools", "gold", "meta", "max_steps", "min_steps"}