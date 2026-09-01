"""api_sandbox 工具内核的测试：确定性、容错、注册表完整性。"""

import pytest

from envs.api_sandbox import TOOLS, all_specs, call

EXPECTED_TOOLS = [
    "weather.query",
    "weather.forecast",
    "flight.search",
    "flight.book",
    "train.search",
    "train.book",
    "hotel.search",
    "hotel.book",
    "meeting.create",
    "alarm.add",
]


def test_all_ten_tools_registered():
    assert list(TOOLS.keys()) == EXPECTED_TOOLS


def test_search_is_deterministic():
    params = {"from": "北京", "to": "上海", "date": "明天"}
    assert call("flight.search", params) == call("flight.search", params)


def test_different_params_give_different_results():
    a = call("flight.search", {"from": "北京", "to": "上海", "date": "明天"})
    b = call("flight.search", {"from": "北京", "to": "广州", "date": "明天"})
    assert a != b


@pytest.mark.parametrize(
    ("tool", "params"),
    [
        ("weather.query", {}),
        ("flight.search", {"from": "北京"}),
        ("alarm.add", {"time": "09:00"}),
        ("weather.forecast", {"city": "北京", "days": "三天"}),
    ],
)
def test_invalid_params_return_error(tool, params):
    result = call(tool, params)
    assert "error" in result


def test_unknown_tool_returns_error():
    assert call("no.such.tool", {}) == {"error": "未知工具: no.such.tool"}


def test_every_tool_valid_params_are_deterministic_and_error_free():
    valid = {
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
    for tool, params in valid.items():
        first = call(tool, params)
        assert "error" not in first, f"{tool} 意外返回错误: {first}"
        assert call(tool, params) == first, f"{tool} 结果不确定"


def test_all_specs_have_schema_shape():
    specs = all_specs()
    assert len(specs) == 10
    for spec in specs:
        assert spec.name
        assert spec.parameters.type == "object"
        assert spec.parameters.required
