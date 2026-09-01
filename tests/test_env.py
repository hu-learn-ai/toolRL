"""ApiSandboxEnv 的测试：reset/step 工具执行闭环、步数截断、judge 委托。"""

import json

import pytest

from envs.api_sandbox import ApiSandboxEnv, call, generate_tasks


def _gold_trajectory(task):
    def tool(api, params):
        return {"raw": "<tool_call>" + json.dumps({"name": api, "arguments": params}, ensure_ascii=False) + "</tool_call>"}

    def answer(text):
        return {"raw": "<answer>" + text + "</answer>"}

    return [tool(c.api, c.params) for c in task.gold.calls] + [answer(task.gold.answer)]


def test_reset_returns_instruction_and_tools():
    env = ApiSandboxEnv()
    task = generate_tasks(0, 1)[0]
    obs = env.reset(task)
    assert obs["instruction"] == task.instruction
    assert [t["name"] for t in obs["tools"]] == [t.name for t in task.tools]
    assert obs["tools"][0]["parameters"]["type"] == "object"


def test_step_executes_tool():
    env = ApiSandboxEnv()
    task = generate_tasks(0, 1)[0]
    env.reset(task)
    c = task.gold.calls[0]
    out = env.step({"api": c.api, "params": c.params})
    assert out["observation"] == call(c.api, c.params)
    assert out["done"] is False
    assert "error" not in out["observation"]


def test_step_invalid_params_returns_error():
    env = ApiSandboxEnv()
    task = generate_tasks(0, 1)[0]
    env.reset(task)
    out = env.step({"api": task.gold.calls[0].api, "params": {}})
    assert "error" in out["observation"]
    assert out["done"] is False


def test_step_done_at_limit():
    env = ApiSandboxEnv()
    task = generate_tasks(0, 1)[0]  # single_tool: max_steps = 5
    env.reset(task)
    api, params = task.gold.calls[0].api, task.gold.calls[0].params
    limit = task.max_steps - 1
    for i in range(1, limit + 1):
        out = env.step({"api": api, "params": params})
        assert out["done"] is (i >= limit)
    # 达到上限后再调用 → 直接 done，返回空观察
    assert env.step({"api": api, "params": params}) == {"observation": {}, "done": True}


def test_step_requires_reset():
    env = ApiSandboxEnv()
    with pytest.raises(RuntimeError):
        env.step({"api": "weather.query", "params": {"city": "北京"}})


def test_judge_delegates_to_module():
    env = ApiSandboxEnv()
    task = generate_tasks(0, 1)[0]
    env.reset(task)
    r = env.judge(_gold_trajectory(task))
    assert r.success is True
    assert r.format == r.correct == r.answer == 1.0
    assert r.steps == 0.0


def test_judge_requires_reset():
    env = ApiSandboxEnv()
    with pytest.raises(RuntimeError):
        env.judge([])


def test_task_generator_matches_module():
    env = ApiSandboxEnv()
    assert next(env.task_generator(0)) == generate_tasks(0, 1)[0]


def test_reset_clears_step_state():
    env = ApiSandboxEnv()
    task = generate_tasks(0, 1)[0]
    env.reset(task)
    api, params = task.gold.calls[0].api, task.gold.calls[0].params
    for _ in range(task.max_steps):
        env.step({"api": api, "params": params})
    assert env.done is True
    env.reset(task)
    assert env.done is False
    assert env.step({"api": api, "params": params})["done"] is False