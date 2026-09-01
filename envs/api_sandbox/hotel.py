"""差旅领域工具：hotel.search / hotel.book。"""

from __future__ import annotations

from ..task_schema import ToolParameterSchema
from .common import HOTEL_CHAINS, Tool, prop, rand_id, rng_for, validate_params

_SEARCH_SCHEMA = ToolParameterSchema(
    properties={
        "city": prop("string"),
        "check_in": prop("string"),
        "check_out": prop("string"),
    },
    required=["city", "check_in", "check_out"],
)


def _search(params: dict) -> dict:
    err = validate_params(params, _SEARCH_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("hotel.search", params)
    hotels = [
        {
            "name": f"{rng.choice(HOTEL_CHAINS)}{params['city']}{rng.randint(1, 99)}店",
            "address": f"{params['city']}市{rng.choice(['中心', '高新区', '火车站', '机场'])}",
            "price_per_night": rng.randint(150, 1200),
            "rating": round(rng.uniform(3.5, 5.0), 1),
        }
        for _ in range(rng.randint(2, 5))
    ]
    return {
        "city": params["city"],
        "check_in": params["check_in"],
        "check_out": params["check_out"],
        "hotels": hotels,
    }


_BOOK_SCHEMA = ToolParameterSchema(
    properties={
        "hotel": prop("string"),
        "check_in": prop("string"),
        "check_out": prop("string"),
        "guest": prop("string"),
    },
    required=["hotel", "check_in", "check_out", "guest"],
)


def _book(params: dict) -> dict:
    err = validate_params(params, _BOOK_SCHEMA)
    if err:
        return {"error": err}
    rng = rng_for("hotel.book", params)
    return {
        "booking_id": rand_id(rng, "HT"),
        "hotel": params["hotel"],
        "check_in": params["check_in"],
        "check_out": params["check_out"],
        "guest": params["guest"],
        "status": "confirmed",
    }


TOOLS = (
    Tool("hotel.search", "搜索酒店，参数：city, check_in, check_out", _SEARCH_SCHEMA, _search),
    Tool("hotel.book", "预订酒店，参数：hotel, check_in, check_out, guest", _BOOK_SCHEMA, _book),
)
