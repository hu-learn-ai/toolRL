"""出行领域工具：flight.search / flight.book。"""

from __future__ import annotations

from ..task_schema import ToolParameterSchema
from .common import AIRLINES, Tool, prop, rand_id, rng_for, validate_params

_SEARCH_SCHEMA = ToolParameterSchema(
    properties={"from": prop("string"), "to": prop("string"), "date": prop("string")},
    required=["from", "to", "date"],
)


def _search(params: dict) -> dict:
    err = validate_params(params, _SEARCH_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("flight.search", params)
    flights = [
        {
            "flight_no": f"{rng.choice(['CA', 'MU', 'CZ', 'HU', '3U'])}{rng.randint(100, 999)}",
            "airline": rng.choice(AIRLINES),
            "departure": f"{rng.randint(6, 22):02d}:{rng.choice(['00', '15', '30', '45'])}",
            "arrival": f"{rng.randint(6, 23):02d}:{rng.choice(['00', '15', '30', '45'])}",
            "price": rng.randint(300, 3000),
        }
        for _ in range(rng.randint(2, 5))
    ]
    return {"from": params["from"], "to": params["to"], "date": params["date"], "flights": flights}


_BOOK_SCHEMA = ToolParameterSchema(
    properties={
        "flight_no": prop("string"),
        "date": prop("string"),
        "passenger": prop("string"),
    },
    required=["flight_no", "date", "passenger"],
)


def _book(params: dict) -> dict:
    err = validate_params(params, _BOOK_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("flight.book", params)
    return {
        "booking_id": rand_id(rng, "FL"),
        "flight_no": params["flight_no"],
        "date": params["date"],
        "passenger": params["passenger"],
        "status": "confirmed",
    }


TOOLS = (
    Tool("flight.search", "搜索航班，参数：from, to, date", _SEARCH_SCHEMA, _search),
    Tool("flight.book", "预订航班，参数：flight_no, date, passenger", _BOOK_SCHEMA, _book),
)
