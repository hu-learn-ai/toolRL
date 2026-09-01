"""judge 的测试：格式判定、分级给分、以及"gold 轨迹必满分"的黄金路径。"""

import json

import pytest

from envs.api_sandbox import TOOLS, generate_tasks, judge
from envs.api_sandbox.judge import parse_turn
from envs.task_schema import Gold, Task, ToolCall

# 每个工具的合法参数（与 test_api_sandbox.py 保持一致，用于构造格式合法但内容错误的轨迹）
VALID_PARAMS = {
    "weather.query": {"city": "北京"},
    "weather.forecast": {"city": "北京", "days": 3},
    "flight.search": {"from": "北京", "to": "上海", "date": "2026-09-01"},
    "flight.book": {"flight_no": "CA123", "date": "2026-09-01", "passenger": "张三"},
    "train.search": {"from": "北京", "to": "上海", "date": "2026-09-01"},
    "train.book": {"train_no": "G123", "date": "2026-09-01", "passenger": "张三"},
    "hotel.search": {"city": "上海", "check_in": "2026-09-01", "check_out": "2026-09-03"},
    "hotel.book": {"hotel": "全季上海1店", "check_in": "2026-09-01", "check_out": "2026-09-03", "guest": "张三"},
    "meeting.create": {"title": "周会", "time": "10:00", "attendees": ["张三", "李四"]},
    "alarm.add": {"time": "09:00", "content": "去机场"},
}


def _tool(api, params):
    return {"raw": "<think>...</think><tool_call>" + json.dumps({"name": api, "arguments": params}, ensure_ascii=False) + "</tool_call>"}


def _answer(text):
    return {"raw": "<think>...</think><answer>" + text + "</answer>"}


def _gold_trajectory(task):
    return [_tool(c.api, c.params) for c in task.gold.calls] + [_answer(task.gold.answer)]


def _task(api, params, answer, extra_calls=()):
    """手工构造单步任务（tools 不影响判定，judge 用全局 TOOLS 校验 schema）。"""
    calls = [ToolCall(api=api, params=params)]
    for c in extra_calls:
        calls.append(ToolCall(api=c[0], params=c[1]))
    k = len(calls)
    return Task(
        task_id="t",
        category="single_tool",
        instruction="x",
        tools=[],
        gold=Gold(calls=calls, answer=answer),
        max_steps=k + 4,
        min_steps=k + 1,
    )


# ---------------------------------------------------------------------------
# 黄金路径：gold 轨迹必须满分（最关键的不变量）
# ---------------------------------------------------------------------------


def test_gold_trajectory_scores_full():
    for task in generate_tasks(0, 300):
        r = judge(task, _gold_trajectory(task))
        assert r.format == 1.0, task.task_id
        assert r.correct == 1.0, task.task_id
        assert r.answer == 1.0, task.task_id
        assert r.steps == 0.0, task.task_id
        assert r.success is True, task.task_id


# ---------------------------------------------------------------------------
# 格式判定（R_format）
# ---------------------------------------------------------------------------


def test_format_invalid_json():
    task = generate_tasks(0, 1)[0]
    r = judge(task, [{"raw": "<tool_call>{not json}</tool_call>"}, _answer(task.gold.answer)])
    assert r.format == 0.0


def test_format_empty_arguments_rejected():
    # §4.4：空 arguments 骗格式分 → 判为格式不合规
    task = generate_tasks(0, 1)[0]
    api = task.gold.calls[0].api
    r = judge(task, [_tool(api, {}), _answer(task.gold.answer)])
    assert r.format == 0.0


def test_format_unpaired_tag():
    task = generate_tasks(0, 1)[0]
    raw = '<tool_call>{"name":"weather.query","arguments":{"city":"北京"}}'
    r = judge(task, [{"raw": raw}, _answer(task.gold.answer)])
    assert r.format == 0.0


def test_format_answer_not_last():
    task = generate_tasks(0, 1)[0]
    r = judge(task, [_answer(task.gold.answer), _tool(task.gold.calls[0].api, task.gold.calls[0].params)])
    assert r.format == 0.0


def test_format_multiple_answers():
    task = generate_tasks(0, 1)[0]
    r = judge(task, [_answer("第一条"), _answer(task.gold.answer)])
    assert r.format == 0.0


def test_format_missing_action_block():
    task = generate_tasks(0, 1)[0]
    r = judge(task, [{"raw": "<think>只有思考没有动作</think>"}, _answer(task.gold.answer)])
    assert r.format == 0.0


# ---------------------------------------------------------------------------
# 调用正确（R_correct）
# ---------------------------------------------------------------------------


def test_wrong_api_zero_correct():
    task = _task("weather.query", {"city": "北京"}, "北京今天多云，气温 25℃，湿度 60%")
    r = judge(task, [_tool("weather.forecast", VALID_PARAMS["weather.forecast"]), _answer("北京今天多云，气温 25℃，湿度 60%")])
    assert r.format == 1.0  # 格式仍合法
    assert r.correct == 0.0


def test_wrong_param_value_partial_credit():
    task = _task("weather.query", {"city": "北京"}, "北京今天多云，气温 25℃，湿度 60%")
    r = judge(task, [_tool("weather.query", {"city": "广州"}), _answer("北京今天多云，气温 25℃，湿度 60%")])
    assert r.format == 1.0
    assert r.correct == 0.5  # r_tool=1, r_param=0 → 0.5


def test_param_partial_credit_two_of_three():
    task = _task("flight.search", {"from": "北京", "to": "上海", "date": "明天"}, "已为您查询到 3 个航班")
    traj = [_tool("flight.search", {"from": "北京", "to": "广州", "date": "明天"}), _answer("已为您查询到 3 个航班")]
    r = judge(task, traj)
    assert r.format == 1.0
    assert r.correct == pytest.approx(0.5 + 0.5 * (2 / 3))


def test_extra_call_penalized():
    task = _task("weather.query", {"city": "北京"}, "北京今天多云，气温 25℃，湿度 60%")
    traj = [
        _tool("weather.query", {"city": "北京"}),
        _tool("weather.query", {"city": "北京"}),
        _answer("北京今天多云，气温 25℃，湿度 60%"),
    ]
    r = judge(task, traj)
    assert r.format == 1.0
    assert r.correct == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)  # r_tool=1/2, r_param=1
    assert r.success is False


# ---------------------------------------------------------------------------
# 最终答案（R_answer）与步数（R_steps）
# ---------------------------------------------------------------------------


def test_answer_fact_subset():
    task = _task("flight.book", VALID_PARAMS["flight.book"], "已为您预订航班 CA123，订单号 FL123456")
    assert judge(task, [_answer("已预订 CA123，订单 FL123456")]).answer == 1.0
    assert judge(task, [_answer("已预订航班 CA123")]).answer == 0.0  # 缺订单号


def test_answer_time_normalized():
    task = _task("alarm.add", {"time": "09:00", "content": "去机场"}, "已为您设置 09:00 的提醒")
    assert judge(task, [_answer("9:00 提醒已设置")]).answer == 1.0


def test_steps_penalty_zero_when_optimal():
    task = _task("weather.query", {"city": "北京"}, "北京今天多云，气温 25℃，湿度 60%")
    r = judge(task, [_tool("weather.query", {"city": "北京"}), _answer("北京今天多云，气温 25℃，湿度 60%")])
    assert r.steps == 0.0  # 步数 == min_steps
    assert r.success is True


def test_steps_penalty_grows_with_wasted_turns():
    task = _task("weather.query", {"city": "北京"}, "北京今天多云，气温 25℃，湿度 60%")
    optimal = [_tool("weather.query", {"city": "北京"}), _answer("北京今天多云，气温 25℃，湿度 60%")]
    wasted = [_tool("weather.query", {"city": "北京"})] + optimal
    r = judge(task, wasted)
    assert r.steps == pytest.approx((len(wasted) - task.min_steps) / task.max_steps)
    assert r.steps > 0.0


# ---------------------------------------------------------------------------
# parse_turn 单元测试
# ---------------------------------------------------------------------------


def test_parse_turn_tool_call():
    p = parse_turn('<think>x</think><tool_call>{"name":"weather.query","arguments":{"city":"北京"}}</tool_call>')
    assert p.kind == "tool_call"
    assert p.tool_name == "weather.query"
    assert p.arguments == {"city": "北京"}


def test_parse_turn_answer():
    p = parse_turn("<think>x</think><answer>你好</answer>")
    assert p.kind == "answer"
    assert p.answer == "你好"


def test_parse_turn_invalid_json():
    assert parse_turn("<tool_call>oops</tool_call>").kind == "invalid"


def test_parse_turn_both_blocks_invalid():
    assert parse_turn("<tool_call>{}</tool_call><answer>x</answer>").kind == "invalid"


def test_parse_turn_no_block_invalid():
    assert parse_turn("<think>只有思考</think>").kind == "invalid"


def test_all_tools_covered_by_valid_params():
    assert set(VALID_PARAMS) == set(TOOLS)