"""效率领域工具：meeting.create / alarm.add。"""

from __future__ import annotations

from ..task_schema import ToolParameterSchema
from .common import Tool, prop, rand_id, rng_for, validate_params

_MEETING_SCHEMA = ToolParameterSchema(
    properties={
        "title": prop("string"),
        "time": prop("string"),
        "attendees": prop("array"),
    },
    required=["title", "time", "attendees"],
)


def _meeting_create(params: dict) -> dict:
    err = validate_params(params, _MEETING_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("meeting.create", params)
    return {
        "meeting_id": rand_id(rng, "MT"),
        "title": params["title"],
        "time": params["time"],
        "attendees": params["attendees"],
        "status": "scheduled",
    }


_ALARM_SCHEMA = ToolParameterSchema(
    properties={"time": prop("string"), "content": prop("string")},
    required=["time", "content"],
)


def _alarm_add(params: dict) -> dict:
    err = validate_params(params, _ALARM_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("alarm.add", params)
    return {
        "alarm_id": rand_id(rng, "AL"),
        "time": params["time"],
        "content": params["content"],
        "status": "set",
    }


TOOLS = (
    Tool("meeting.create", "创建会议，参数：title, time, attendees", _MEETING_SCHEMA, _meeting_create),
    Tool("alarm.add", "添加日程提醒，参数：time, content", _ALARM_SCHEMA, _alarm_add),
)
