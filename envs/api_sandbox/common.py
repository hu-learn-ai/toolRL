"""api_sandbox 的共享基础设施：确定性随机、参数校验、工具注册条目。

各领域模块（weather/flight/train/hotel/calendar）从这里复用：
- ``rng_for``：把 (工具名, 参数) 哈希成随机种子，保证同参数必同结果；
- ``validate_params``：基于 ToolParameterSchema 做必填 + 类型粗校验；
- ``Tool``：注册表条目（schema + handler）。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Callable

from ..task_schema import ToolParameterSchema, ToolSpec

# 生成"看起来像真的"确定性数据的素材
AIRLINES = ["国航", "东航", "南航", "海航", "川航"]
CONDITIONS = ["晴", "多云", "小雨", "阴", "小雪"]
HOTEL_CHAINS = ["如家", "汉庭", "全季", "亚朵", "希尔顿"]


def seed_from(tool: str, params: dict) -> int:
    """参数哈希 → 稳定随机种子（用 md5 而非内置 hash()，跨进程/平台稳定）。"""
    payload = tool + "|" + json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return int(hashlib.md5(payload.encode("utf-8")).hexdigest(), 16) % (2**32)


def rng_for(tool: str, params: dict) -> random.Random:
    return random.Random(seed_from(tool, params))


def validate_params(params: dict, schema: ToolParameterSchema) -> str | None:
    """校验 params 是否符合 schema；合法返回 None，否则返回错误描述。"""
    if not isinstance(params, dict):
        return "params 必须是对象"
    for key in schema.required:
        if key not in params:
            return f"缺少必填参数: {key}"
    for key, value in params.items():
        prop = schema.properties.get(key)
        if prop is None:
            continue  # 额外参数宽松处理
        expected = prop.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"参数 {key} 应为字符串"
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return f"参数 {key} 应为整数"
        if expected == "number" and not isinstance(value, (int, float)):
            return f"参数 {key} 应为数字"
        if expected == "array" and not isinstance(value, list):
            return f"参数 {key} 应为数组"
        if expected == "boolean" and not isinstance(value, bool):
            return f"参数 {key} 应为布尔值"
    return None


def prop(type_: str, description: str = "") -> dict:
    p: dict = {"type": type_}
    if description:
        p["description"] = description
    return p


def rand_id(rng: random.Random, prefix: str) -> str:
    return f"{prefix}{rng.randint(100000, 999999)}"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: ToolParameterSchema
    handler: Callable[[dict], dict]

    def to_spec(self) -> ToolSpec:
        """导出为注入系统提示词的函数定义（design_zh.md §1.2 的 tools 字段）。"""
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)
