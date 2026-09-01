"""Text2SQL 的三个数据库 schema 与确定性种子数据（design_zh.md §2.3）。

- 电商 ecommerce：users / products / orders / order_items
- 员工 employee：departments / employees
- 库存 inventory：warehouses / inventory

种子数据用固定 ``DB_SEED`` 确定性生成，同一 schema 每次重建得到相同数据（§7.1
可复现）。列间外键关系自洽（orders.user_id ∈ users、order_items.product_id ∈
products 等），保证模板里的 join 查询都能命中。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# 固定种子：同一 schema 的数据永远一致，judge 据此重建 DB 复现 gold 结果。
DB_SEED = 42

SCHEMA_NAMES = ("ecommerce", "employee", "inventory")


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[tuple[str, str], ...]  # (列名, SQLite 类型)


TABLES: dict[str, tuple[Table, ...]] = {
    "ecommerce": (
        Table("users", (("user_id", "INTEGER"), ("name", "TEXT"), ("city", "TEXT"), ("age", "INTEGER"), ("reg_date", "TEXT"))),
        Table("products", (("product_id", "INTEGER"), ("name", "TEXT"), ("category", "TEXT"), ("price", "REAL"))),
        Table("orders", (("order_id", "INTEGER"), ("user_id", "INTEGER"), ("amount", "REAL"), ("order_date", "TEXT"), ("status", "TEXT"))),
        Table("order_items", (("item_id", "INTEGER"), ("order_id", "INTEGER"), ("product_id", "INTEGER"), ("quantity", "INTEGER"), ("price", "REAL"))),
    ),
    "employee": (
        Table("departments", (("dept_id", "INTEGER"), ("name", "TEXT"), ("location", "TEXT"))),
        Table("employees", (("emp_id", "INTEGER"), ("name", "TEXT"), ("dept_id", "INTEGER"), ("salary", "INTEGER"), ("hire_date", "TEXT"))),
    ),
    "inventory": (
        Table("warehouses", (("warehouse_id", "INTEGER"), ("name", "TEXT"), ("city", "TEXT"), ("capacity", "INTEGER"))),
        Table("inventory", (("item_id", "INTEGER"), ("warehouse_id", "INTEGER"), ("product_name", "TEXT"), ("quantity", "INTEGER"), ("unit_price", "REAL"))),
    ),
}


def schema_ddl(schema_name: str) -> str:
    """把 schema 渲染成可读的列结构描述，供注入提示词（reset 观察用）。"""
    lines = []
    for t in TABLES[schema_name]:
        cols = ", ".join(f"{c} {ty}" for c, ty in t.columns)
        lines.append(f"{t.name}({cols})")
    return "; ".join(lines)


# ---------------------------------------------------------------------------
# 确定性种子数据
# ---------------------------------------------------------------------------

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京", "武汉", "重庆"]
_SURNAMES = "赵钱孙李周吴郑王冯陈"
_GIVEN = "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀霞"
_CATEGORIES = ["电子产品", "图书", "服装", "家居", "食品", "运动", "美妆", "办公"]
_DEPT_NAMES = ["技术部", "产品部", "市场部", "销售部", "人事部", "财务部", "行政部", "客服部"]
_PRODUCT_NAMES = ["笔记本", "手机", "耳机", "键盘", "显示器", "充电器", "水杯", "台灯", "背包", "文具"]


def _name(rng: random.Random) -> str:
    return rng.choice(_SURNAMES) + rng.choice(_GIVEN)


def _date(rng: random.Random) -> str:
    return f"202{rng.randint(0, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _gen_ecommerce(rng: random.Random) -> dict[str, list[tuple]]:
    products = [
        (i, f"{_PRODUCT_NAMES[(i - 1) % len(_PRODUCT_NAMES)]}{i}", rng.choice(_CATEGORIES), round(rng.uniform(9.9, 999.9), 2))
        for i in range(1, 16)
    ]
    # 城市循环赋值 + 打乱，保证 10 个城市都有用户（过滤模板必命中）
    cities = CITIES[:]
    rng.shuffle(cities)
    users = [
        (i, _name(rng), cities[(i - 1) % len(cities)], rng.randint(18, 65), _date(rng))
        for i in range(1, 21)
    ]
    orders = [
        (i, rng.randint(1, 20), round(rng.uniform(20.0, 5000.0), 2), _date(rng), rng.choice(["已完成", "待支付", "已取消", "配送中"]))
        for i in range(1, 41)
    ]
    order_items = []
    item_id = 1
    for order_id in range(1, 41):
        for _ in range(rng.randint(2, 3)):
            order_items.append((item_id, order_id, rng.randint(1, 15), rng.randint(1, 5), round(rng.uniform(9.9, 999.9), 2)))
            item_id += 1
    return {"users": users, "products": products, "orders": orders, "order_items": order_items}


def _gen_employee(rng: random.Random) -> dict[str, list[tuple]]:
    departments = [(i, _DEPT_NAMES[i - 1], rng.choice(CITIES)) for i in range(1, 9)]
    employees = [(i, _name(rng), rng.randint(1, 8), rng.randint(4000, 20000), _date(rng)) for i in range(1, 81)]
    return {"departments": departments, "employees": employees}


def _gen_inventory(rng: random.Random) -> dict[str, list[tuple]]:
    warehouses = [
        (i, f"{rng.choice(['华东', '华南', '华北', '西南'])}{i}号仓", rng.choice(CITIES), rng.choice([500, 1000, 2000, 5000]))
        for i in range(1, 9)
    ]
    inventory = [
        (i, rng.randint(1, 8), f"{_PRODUCT_NAMES[(i - 1) % len(_PRODUCT_NAMES)]}{i}", rng.randint(0, 500), round(rng.uniform(1.0, 500.0), 2))
        for i in range(1, 101)
    ]
    return {"warehouses": warehouses, "inventory": inventory}


_GENERATORS = {
    "ecommerce": _gen_ecommerce,
    "employee": _gen_employee,
    "inventory": _gen_inventory,
}


def generate_rows(schema_name: str, rng: random.Random) -> dict[str, list[tuple]]:
    """生成某 schema 的种子数据，返回 {表名: 行列表}。"""
    return _GENERATORS[schema_name](rng)