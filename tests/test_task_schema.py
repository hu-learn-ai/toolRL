"""task_schema 的契约测试：校验统一任务/轨迹 JSON 的解析与约束。"""

import pytest
from pydantic import ValidationError

from envs import Gold, Message, Task, ToolCall, ToolSpec, Trajectory


def test_task_roundtrip_from_doc_example(example_task_dict):
    task = Task.model_validate(example_task_dict)
    assert task.task_id == "api_0001"
    assert task.category == "multi_tool"
    assert task.instruction.startswith("帮我订")

    assert len(task.tools) == 1
    assert task.tools[0].name == "flight.search"
    assert task.tools[0].parameters.required == ["from", "to", "date"]

    assert len(task.gold.calls) == 2
    assert task.gold.calls[0].api == "flight.search"
    assert task.gold.calls[0].params == {"from": "北京", "to": "上海", "date": "明天"}
    assert task.gold.answer == "已为您查询到 3 个航班，并设置 09:00 日程提醒"

    assert task.max_steps == 5
    assert task.min_steps == 2


def test_task_rejects_min_steps_greater_than_max(example_task_dict):
    with pytest.raises(ValidationError):
        Task.model_validate({**example_task_dict, "min_steps": 9})


@pytest.mark.parametrize("field", ["max_steps", "min_steps"])
def test_step_fields_must_be_positive(example_task_dict, field):
    with pytest.raises(ValidationError):
        Task.model_validate({**example_task_dict, field: 0})


def test_task_dumps_back_to_equivalent_dict(example_task_dict):
    task = Task.model_validate(example_task_dict)
    dumped = task.model_dump()
    assert dumped["task_id"] == example_task_dict["task_id"]
    assert dumped["gold"]["calls"][0]["api"] == "flight.search"


def test_tool_call_defaults_to_empty_params():
    call = ToolCall(api="weather.query")
    assert call.params == {}


def test_tool_spec_defaults_to_empty_object_schema():
    spec = ToolSpec(name="alarm.add", description="添加日程提醒")
    assert spec.parameters.type == "object"
    assert spec.parameters.properties == {}
    assert spec.parameters.required == []


def test_message_role_must_be_valid():
    with pytest.raises(ValidationError):
        Message(role="bogus", content="hi")


def test_trajectory_roundtrip_from_doc_example(example_trajectory_dict):
    traj = Trajectory.model_validate(example_trajectory_dict)
    assert traj.task_id == "api_0001"
    assert traj.success is True
    assert traj.steps == 2
    assert [m.role for m in traj.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_gold_holds_programmatic_judgement_basis(example_task_dict):
    gold = Gold.model_validate(example_task_dict["gold"])
    assert gold.calls[1].api == "alarm.add"
    assert gold.calls[1].params["time"] == "09:00"
