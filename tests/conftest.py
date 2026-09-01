"""共享测试夹具：来自 design_zh.md §1.2 / §1.3 的示例数据。"""

import pytest


@pytest.fixture
def example_task_dict() -> dict:
    """design_zh.md §1.2 的示例任务 JSON。"""
    return {
        "task_id": "api_0001",
        "category": "multi_tool",
        "instruction": "帮我订明天北京到上海的机票，并添加一条日程提醒",
        "tools": [
            {
                "name": "flight.search",
                "description": "搜索航班，参数：from, to, date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["from", "to", "date"],
                },
            }
        ],
        "gold": {
            "calls": [
                {"api": "flight.search", "params": {"from": "北京", "to": "上海", "date": "明天"}},
                {"api": "alarm.add", "params": {"time": "09:00", "content": "去机场"}},
            ],
            "answer": "已为您查询到 3 个航班，并设置 09:00 日程提醒",
        },
        "max_steps": 5,
        "min_steps": 2,
    }


@pytest.fixture
def example_trajectory_dict() -> dict:
    """design_zh.md §1.3 的示例轨迹 JSONL 一行。"""
    return {
        "task_id": "api_0001",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "帮我订..."},
            {
                "role": "assistant",
                "content": '<think>...</think>\n<tool_call>{"name":"flight.search","arguments":{...}}</tool_call>',
            },
            {"role": "tool", "content": '{"flights": [...]}'},
            {
                "role": "assistant",
                "content": "<think>...</think>\n<answer>已为您查询到 3 个航班...</answer>",
            },
        ],
        "success": True,
        "steps": 2,
    }
