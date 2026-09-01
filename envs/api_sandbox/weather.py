"""天气领域工具：weather.query / weather.forecast。"""

from __future__ import annotations

from ..task_schema import ToolParameterSchema
from .common import CONDITIONS, Tool, prop, rng_for, validate_params

_QUERY_SCHEMA = ToolParameterSchema(
    properties={"city": prop("string")},
    required=["city"],
)


def _query(params: dict) -> dict:
    err = validate_params(params, _QUERY_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("weather.query", params)
    return {
        "city": params["city"],
        "temperature": rng.randint(-5, 38),
        "condition": rng.choice(CONDITIONS),
        "humidity": rng.randint(20, 95),
    }


_FORECAST_SCHEMA = ToolParameterSchema(
    properties={"city": prop("string"), "days": prop("integer")},
    required=["city", "days"],
)


def _forecast(params: dict) -> dict:
    err = validate_params(params, _FORECAST_SCHEMA)
    if err:
        return {"error": err}
    if not 1 <= params["days"] <= 7:
        return {"error": "days 应在 1-7 之间"}
    rng = rng_for("weather.forecast", params)
    forecast = [
        {
            "day": i + 1,
            "condition": rng.choice(CONDITIONS),
            "high": rng.randint(10, 38),
            "low": rng.randint(-5, 20),
        }
        for i in range(params["days"])
    ]
    return {"city": params["city"], "forecast": forecast}


TOOLS = (
    Tool("weather.query", "查询城市当前天气，参数：city", _QUERY_SCHEMA, _query),
    Tool("weather.forecast", "查询城市未来 N 天天气，参数：city, days", _FORECAST_SCHEMA, _forecast),
)
