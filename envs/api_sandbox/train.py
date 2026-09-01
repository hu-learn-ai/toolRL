"""出行领域工具：train.search / train.book。"""

from __future__ import annotations

from ..task_schema import ToolParameterSchema
from .common import Tool, prop, rand_id, rng_for, validate_params

_SEARCH_SCHEMA = ToolParameterSchema(
    properties={"from": prop("string"), "to": prop("string"), "date": prop("string")},
    required=["from", "to", "date"],
)


def _search(params: dict) -> dict:
    err = validate_params(params, _SEARCH_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("train.search", params)
    trains = [
        {
            "train_no": f"G{rng.randint(1, 9999)}",
            "departure": f"{rng.randint(6, 22):02d}:{rng.choice(['00', '15', '30', '45'])}",
            "arrival": f"{rng.randint(6, 23):02d}:{rng.choice(['00', '15', '30', '45'])}",
            "duration": f"{rng.randint(2, 10)}h{rng.choice(['00', '15', '30', '45'])}m",
            "price": rng.randint(50, 800),
        }
        for _ in range(rng.randint(2, 5))
    ]
    return {"from": params["from"], "to": params["to"], "date": params["date"], "trains": trains}


_BOOK_SCHEMA = ToolParameterSchema(
    properties={"train_no": prop("string"), "date": prop("string"), "passenger": prop("string")},
    required=["train_no", "date", "passenger"],
)


def _book(params: dict) -> dict:
    err = validate_params(params, _BOOK_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("train.book", params)
    return {
        "booking_id": rand_id(rng, "TR"),
        "train_no": params["train_no"],
        "date": params["date"],
        "passenger": params["passenger"],
        "status": "confirmed",
    }


TOOLS = (
    Tool("train.search", "搜索火车票，参数：from, to, date", _SEARCH_SCHEMA, _search),
    Tool("train.book", "预订火车票，参数：train_no, date, passenger", _BOOK_SCHEMA, _book),
)
