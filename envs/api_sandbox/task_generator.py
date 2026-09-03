"""任务生成器（design_zh.md §2.2）。

模板 + 随机槽位：从一组任务模板中确定性采样，随机填充城市/日期/时间等槽位，
按比例注入干扰工具（任务本不需要的工具），训练模型学会不调用无关工具。

确定性：给定 seed，生成的任务序列完全可复现（§7.1 gen_tasks 依赖这一点）。

实现要点：
- 多轮/长程任务的后续 gold.calls 依赖前一步结果（如"订第一个航班"的 flight_no 来自
  flight.search 的确定性返回），所以模板会实际执行 gold 序列来解析依赖，保证 gold 自洽；
- gold.answer 由同一份确定性结果拼装，与 mock 数据严格一致。

步数口径：步数 = 助手回合数 = 工具调用数 + 最终回答那一轮（对齐 §1.3 示例：1 次调用 +
1 次回答 → steps=2）。故 min_steps = 工具调用数 + 1，max_steps = min_steps + 3（留 3 步
试错余量）。judge 实现时需与此口径保持一致。
"""

from __future__ import annotations

import itertools
import random
from typing import Callable, Iterator

from ..task_schema import Gold, Task, ToolCall, ToolSpec
from .tools import TOOLS, call

# 步数口径（见模块 docstring）：min_steps = 工具调用数 + 最终回答 1 步，max_steps 再留
# _MARGIN_STEPS 步试错余量。
_ANSWER_STEP = 1
_MARGIN_STEPS = 3

# 干扰工具默认注入概率（§5.1 干扰项维度）。
DEFAULT_DISTRACTOR_RATIO = 0.3

# ---------------------------------------------------------------------------
# 槽位素材与取值
# ---------------------------------------------------------------------------

_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京", "武汉", "重庆"]
_DATES = ["明天", "后天", "下周一", "下周三", "下周五"]
_TIMES = ["09:00", "10:30", "14:00", "16:30", "19:00", "20:30"]
_NAMES = ["张三", "李四", "王五", "赵六", "孙七", "周八"]
_ALARM_CONTENTS = ["去机场", "开会", "吃药", "取快递", "接孩子", "提交周报"]
_MEETING_TITLES = ["周会", "项目评审", "一对一沟通", "产品评审", "需求对齐"]
_DAYS = [3, 5, 7]


def _city(rng: random.Random) -> str:
    return rng.choice(_CITIES)


def _two_cities(rng: random.Random) -> tuple[str, str]:
    frm, to = _city(rng), _city(rng)
    while to == frm:
        to = _city(rng)
    return frm, to


def _date(rng: random.Random) -> str:
    return rng.choice(_DATES)


def _time(rng: random.Random) -> str:
    return rng.choice(_TIMES)


def _name(rng: random.Random) -> str:
    return rng.choice(_NAMES)


def _content(rng: random.Random) -> str:
    return rng.choice(_ALARM_CONTENTS)


def _title(rng: random.Random) -> str:
    return rng.choice(_MEETING_TITLES)


def _days(rng: random.Random) -> int:
    return rng.choice(_DAYS)


def _attendees(rng: random.Random) -> list[str]:
    return rng.sample(_NAMES, k=2)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


def _build(
    task_id: str,
    category: str,
    instruction: str,
    gold_calls: list[dict],
    answer: str,
) -> Task:
    """把模板产物组装成 Task；tools 只含 gold 用到的工具，干扰工具由上层注入。"""
    apis = list(dict.fromkeys(c["api"] for c in gold_calls))
    return Task(
        task_id=task_id,
        category=category,
        instruction=instruction,
        tools=[TOOLS[name].to_spec() for name in apis],
        gold=Gold(
            calls=[ToolCall(api=c["api"], params=c["params"]) for c in gold_calls],
            answer=answer,
        ),
        # 步数 = 助手回合 = 工具调用数 + 最终回答那一轮；max 再留 _MARGIN_STEPS 步试错
        max_steps=len(gold_calls) + _ANSWER_STEP + _MARGIN_STEPS,
        min_steps=len(gold_calls) + _ANSWER_STEP,
    )


# ---------------------------------------------------------------------------
# 单工具（single_tool）：一次调用即完成
# ---------------------------------------------------------------------------


def _tpl_weather_query(rng: random.Random, task_id: str) -> Task:
    city = _city(rng)
    r = call("weather.query", {"city": city})
    return _build(
        task_id, "single_tool",
        f"帮我查一下{city}今天的天气",
        [{"api": "weather.query", "params": {"city": city}}],
        f"{city}今天{r['condition']}，气温 {r['temperature']}℃，湿度 {r['humidity']}%",
    )


def _tpl_weather_forecast(rng: random.Random, task_id: str) -> Task:
    city, days = _city(rng), _days(rng)
    return _build(
        task_id, "single_tool",
        f"帮我查一下{city}未来{days}天的天气预报",
        [{"api": "weather.forecast", "params": {"city": city, "days": days}}],
        f"已为您查询{city}未来{days}天的天气",
    )


def _tpl_flight_search(rng: random.Random, task_id: str) -> Task:
    frm, to = _two_cities(rng)
    date = _date(rng)
    r = call("flight.search", {"from": frm, "to": to, "date": date})
    return _build(
        task_id, "single_tool",
        f"查{date}从{frm}到{to}的航班",
        [{"api": "flight.search", "params": {"from": frm, "to": to, "date": date}}],
        f"已为您查询到 {len(r['flights'])} 个从{frm}到{to}的航班",
    )


def _tpl_train_search(rng: random.Random, task_id: str) -> Task:
    frm, to = _two_cities(rng)
    date = _date(rng)
    r = call("train.search", {"from": frm, "to": to, "date": date})
    return _build(
        task_id, "single_tool",
        f"查{date}从{frm}到{to}的火车票",
        [{"api": "train.search", "params": {"from": frm, "to": to, "date": date}}],
        f"已为您查询到 {len(r['trains'])} 趟从{frm}到{to}的火车",
    )


def _tpl_hotel_search(rng: random.Random, task_id: str) -> Task:
    city = _city(rng)
    check_in, check_out = _date(rng), _date(rng)
    r = call("hotel.search", {"city": city, "check_in": check_in, "check_out": check_out})
    return _build(
        task_id, "single_tool",
        f"查{city}{check_in}入住、{check_out}退房的酒店",
        [{"api": "hotel.search", "params": {"city": city, "check_in": check_in, "check_out": check_out}}],
        f"已为您查询到 {len(r['hotels'])} 家{city}的酒店",
    )


def _tpl_meeting_create(rng: random.Random, task_id: str) -> Task:
    title, time = _title(rng), _time(rng)
    attendees = _attendees(rng)
    return _build(
        task_id, "single_tool",
        f"帮我创建一个「{title}」会议，时间 {time}，参会人 {'、'.join(attendees)}",
        [{"api": "meeting.create", "params": {"title": title, "time": time, "attendees": attendees}}],
        f"已为您创建会议「{title}」，时间 {time}",
    )


def _tpl_alarm_add(rng: random.Random, task_id: str) -> Task:
    time, content = _time(rng), _content(rng)
    return _build(
        task_id, "single_tool",
        f"帮我设置一个 {time} 的提醒：{content}",
        [{"api": "alarm.add", "params": {"time": time, "content": content}}],
        f"已为您设置 {time} 的提醒：{content}",
    )


# ---------------------------------------------------------------------------
# 多工具（multi_tool）：≥2 个独立工具顺序调用
# ---------------------------------------------------------------------------


def _tpl_flight_alarm(rng: random.Random, task_id: str) -> Task:
    frm, to = _two_cities(rng)
    date, time = _date(rng), _time(rng)
    r = call("flight.search", {"from": frm, "to": to, "date": date})
    return _build(
        task_id, "multi_tool",
        f"帮我查{date}从{frm}到{to}的航班，并设置一个 {time} 的出发提醒",
        [
            {"api": "flight.search", "params": {"from": frm, "to": to, "date": date}},
            {"api": "alarm.add", "params": {"time": time, "content": "去机场"}},
        ],
        f"已为您查询到 {len(r['flights'])} 个航班，并设置 {time} 提醒",
    )


def _tpl_train_alarm(rng: random.Random, task_id: str) -> Task:
    frm, to = _two_cities(rng)
    date, time = _date(rng), _time(rng)
    r = call("train.search", {"from": frm, "to": to, "date": date})
    return _build(
        task_id, "multi_tool",
        f"帮我查{date}从{frm}到{to}的火车票，并设置一个 {time} 的出发提醒",
        [
            {"api": "train.search", "params": {"from": frm, "to": to, "date": date}},
            {"api": "alarm.add", "params": {"time": time, "content": "去火车站"}},
        ],
        f"已为您查询到 {len(r['trains'])} 趟火车，并设置 {time} 提醒",
    )


def _tpl_weather_meeting(rng: random.Random, task_id: str) -> Task:
    city, time = _city(rng), _time(rng)
    title = _title(rng)
    attendees = _attendees(rng)
    return _build(
        task_id, "multi_tool",
        f"帮我查一下{city}的天气，然后创建会议「{title}」，时间 {time}，参会人 {'、'.join(attendees)}",
        [
            {"api": "weather.query", "params": {"city": city}},
            {"api": "meeting.create", "params": {"title": title, "time": time, "attendees": attendees}},
        ],
        f"已为您查询{city}天气并创建会议「{title}」",
    )


def _tpl_hotel_alarm(rng: random.Random, task_id: str) -> Task:
    city = _city(rng)
    check_in, check_out = _date(rng), _date(rng)
    time = _time(rng)
    r = call("hotel.search", {"city": city, "check_in": check_in, "check_out": check_out})
    return _build(
        task_id, "multi_tool",
        f"帮我查{city}的酒店（{check_in}入住），并设置一个 {time} 的入住提醒",
        [
            {"api": "hotel.search", "params": {"city": city, "check_in": check_in, "check_out": check_out}},
            {"api": "alarm.add", "params": {"time": time, "content": "办理入住"}},
        ],
        f"已为您查询到 {len(r['hotels'])} 家酒店，并设置 {time} 提醒",
    )


# ---------------------------------------------------------------------------
# 多轮（multi_turn）：依赖前一步返回结果决策
# ---------------------------------------------------------------------------


def _tpl_flight_roundtrip(rng: random.Random, task_id: str) -> Task:
    frm, to = _two_cities(rng)
    date, name = _date(rng), _name(rng)
    search = call("flight.search", {"from": frm, "to": to, "date": date})
    flight_no = search["flights"][0]["flight_no"]
    book = call("flight.book", {"flight_no": flight_no, "date": date, "passenger": name})
    return _build(
        task_id, "multi_turn",
        f"查{date}从{frm}到{to}的航班，并预订第一个航班，乘客 {name}",
        [
            {"api": "flight.search", "params": {"from": frm, "to": to, "date": date}},
            {"api": "flight.book", "params": {"flight_no": flight_no, "date": date, "passenger": name}},
        ],
        f"已为您预订航班 {flight_no}，订单号 {book['booking_id']}",
    )


def _tpl_train_roundtrip(rng: random.Random, task_id: str) -> Task:
    frm, to = _two_cities(rng)
    date, name = _date(rng), _name(rng)
    search = call("train.search", {"from": frm, "to": to, "date": date})
    train_no = search["trains"][0]["train_no"]
    book = call("train.book", {"train_no": train_no, "date": date, "passenger": name})
    return _build(
        task_id, "multi_turn",
        f"查{date}从{frm}到{to}的火车票，并预订第一趟，乘客 {name}",
        [
            {"api": "train.search", "params": {"from": frm, "to": to, "date": date}},
            {"api": "train.book", "params": {"train_no": train_no, "date": date, "passenger": name}},
        ],
        f"已为您预订火车 {train_no}，订单号 {book['booking_id']}",
    )


def _tpl_hotel_roundtrip(rng: random.Random, task_id: str) -> Task:
    city, name = _city(rng), _name(rng)
    check_in, check_out = _date(rng), _date(rng)
    search = call("hotel.search", {"city": city, "check_in": check_in, "check_out": check_out})
    hotel = search["hotels"][0]["name"]
    book = call("hotel.book", {"hotel": hotel, "check_in": check_in, "check_out": check_out, "guest": name})
    return _build(
        task_id, "multi_turn",
        f"查{city}的酒店（{check_in}入住），并预订第一家，客人 {name}",
        [
            {"api": "hotel.search", "params": {"city": city, "check_in": check_in, "check_out": check_out}},
            {"api": "hotel.book", "params": {"hotel": hotel, "check_in": check_in, "check_out": check_out, "guest": name}},
        ],
        f"已为您预订酒店 {hotel}，订单号 {book['booking_id']}",
    )


# ---------------------------------------------------------------------------
# 长程（long_horizon）：≥3 步
# ---------------------------------------------------------------------------


def _tpl_flight_trip(rng: random.Random, task_id: str) -> Task:
    frm, to = _two_cities(rng)
    date, name, time = _date(rng), _name(rng), _time(rng)
    search = call("flight.search", {"from": frm, "to": to, "date": date})
    flight_no = search["flights"][0]["flight_no"]
    book = call("flight.book", {"flight_no": flight_no, "date": date, "passenger": name})
    return _build(
        task_id, "long_horizon",
        f"帮我订{date}从{frm}到{to}的机票（乘客 {name}），并设置 {time} 的出发提醒",
        [
            {"api": "flight.search", "params": {"from": frm, "to": to, "date": date}},
            {"api": "flight.book", "params": {"flight_no": flight_no, "date": date, "passenger": name}},
            {"api": "alarm.add", "params": {"time": time, "content": "去机场"}},
        ],
        f"已为您预订航班 {flight_no}（订单号 {book['booking_id']}），并设置 {time} 提醒",
    )


def _tpl_hotel_trip(rng: random.Random, task_id: str) -> Task:
    city, name = _city(rng), _name(rng)
    check_in, check_out, time = _date(rng), _date(rng), _time(rng)
    search = call("hotel.search", {"city": city, "check_in": check_in, "check_out": check_out})
    hotel = search["hotels"][0]["name"]
    book = call("hotel.book", {"hotel": hotel, "check_in": check_in, "check_out": check_out, "guest": name})
    return _build(
        task_id, "long_horizon",
        f"帮我订{city}的酒店（{check_in}入住，客人 {name}），并设置 {time} 的入住提醒",
        [
            {"api": "hotel.search", "params": {"city": city, "check_in": check_in, "check_out": check_out}},
            {"api": "hotel.book", "params": {"hotel": hotel, "check_in": check_in, "check_out": check_out, "guest": name}},
            {"api": "alarm.add", "params": {"time": time, "content": "办理入住"}},
        ],
        f"已为您预订酒店 {hotel}（订单号 {book['booking_id']}），并设置 {time} 提醒",
    )


def _tpl_forecast_meeting_alarm(rng: random.Random, task_id: str) -> Task:
    city, days, time = _city(rng), _days(rng), _time(rng)
    title = _title(rng)
    attendees = _attendees(rng)
    return _build(
        task_id, "long_horizon",
        f"查{city}未来{days}天的天气，然后创建会议「{title}」（时间 {time}），并设置 {time} 提醒",
        [
            {"api": "weather.forecast", "params": {"city": city, "days": days}},
            {"api": "meeting.create", "params": {"title": title, "time": time, "attendees": attendees}},
            {"api": "alarm.add", "params": {"time": time, "content": title}},
        ],
        f"已为您查询{city}未来{days}天天气，创建会议「{title}」，并设置 {time} 提醒",
    )


# ---------------------------------------------------------------------------
# 注册表与对外接口
# ---------------------------------------------------------------------------

TemplateFn = Callable[[random.Random, str], Task]

TEMPLATES: list[tuple[str, TemplateFn]] = [
    ("single_tool", _tpl_weather_query),
    ("single_tool", _tpl_weather_forecast),
    ("single_tool", _tpl_flight_search),
    ("single_tool", _tpl_train_search),
    ("single_tool", _tpl_hotel_search),
    ("single_tool", _tpl_meeting_create),
    ("single_tool", _tpl_alarm_add),
    ("multi_tool", _tpl_flight_alarm),
    ("multi_tool", _tpl_train_alarm),
    ("multi_tool", _tpl_weather_meeting),
    ("multi_tool", _tpl_hotel_alarm),
    ("multi_turn", _tpl_flight_roundtrip),
    ("multi_turn", _tpl_train_roundtrip),
    ("multi_turn", _tpl_hotel_roundtrip),
    ("long_horizon", _tpl_flight_trip),
    ("long_horizon", _tpl_hotel_trip),
    ("long_horizon", _tpl_forecast_meeting_alarm),
]


def _distractors(used_apis: set[str], rng: random.Random) -> list[ToolSpec]:
    """从未被任务使用的工具中挑 1-2 个作为干扰项。"""
    candidates = [name for name in TOOLS if name not in used_apis]
    k = rng.randint(1, 2)
    return [TOOLS[name].to_spec() for name in rng.sample(candidates, k=min(k, len(candidates)))]


def generate_task(rng: random.Random, task_id: str, distractor_ratio: float = DEFAULT_DISTRACTOR_RATIO) -> Task:
    """按给定 RNG 生成单个任务；以 distractor_ratio 概率注入干扰工具。"""
    _, fn = rng.choice(TEMPLATES)
    task = fn(rng, task_id)
    if rng.random() < distractor_ratio:
        used = {c.api for c in task.gold.calls}
        task.tools.extend(_distractors(used, rng))
    return task


def task_generator(seed: int, distractor_ratio: float = DEFAULT_DISTRACTOR_RATIO) -> Iterator[Task]:
    """按 seed 确定性无限生成任务；配合 islice 取固定数量。"""
    rng = random.Random(seed)
    i = 0
    while True:
        yield generate_task(rng, f"api_{i:04d}", distractor_ratio)
        i += 1


def generate_tasks(seed: int, n: int, distractor_ratio: float = DEFAULT_DISTRACTOR_RATIO) -> list[Task]:
    """生成固定数量 n 的任务（§7.1 gen_tasks 用）。"""
    return list(itertools.islice(task_generator(seed, distractor_ratio), n))
