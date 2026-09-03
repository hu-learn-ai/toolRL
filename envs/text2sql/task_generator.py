"""Text2SQL 任务生成器（design_zh.md §2.3）。

三类 schema × 四类问题模板（单表过滤 / 多表 join / 聚合分组 / 排序分页），确定性
采样填充槽位；gold SQL 生成时实际执行一遍以校验合法性并渲染 gold.answer（参考用，
判定走结果集一致性）。任务复用统一 Task schema，``meta={"db": schema}`` 标记所属库。
"""

from __future__ import annotations

import itertools
import random
from typing import Callable, Iterator

from ..task_schema import Gold, Task, ToolCall, ToolParameterSchema, ToolSpec
from .db import Database
from .schemas import CITIES, SCHEMA_NAMES

# 步数口径（同 api_sandbox）：1 次 SQL 调用 + 最终回答 1 步，max 再留 3 步试错余量。
_ANSWER_STEP = 1
_MARGIN_STEPS = 3

SQL_TOOL = "sql.execute"
SQL_SPEC = ToolSpec(
    name=SQL_TOOL,
    description="在给定数据库上执行一条只读 SQL 查询，返回 (列名, 行) 结果集。",
    parameters=ToolParameterSchema(properties={"query": {"type": "string"}}, required=["query"]),
)


def _render(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "（0 行）"
    header = ", ".join(columns)
    body = "; ".join(", ".join(str(v) for v in row) for row in rows)
    return f"{len(rows)} 行 [{header}] => {body}"


def _sql_task(task_id: str, category: str, instruction: str, sql: str, db: Database) -> Task:
    cols, rows = db.execute(sql)  # 生成期即校验 SQL 合法
    return Task(
        task_id=task_id,
        category=category,
        instruction=instruction,
        tools=[SQL_SPEC],
        gold=Gold(calls=[ToolCall(api=SQL_TOOL, params={"query": sql})], answer=_render(cols, rows)),
        # 步数 = 1 次 SQL 调用 + 最终回答 1 步，max 再留 _MARGIN_STEPS 步试错
        max_steps=1 + _ANSWER_STEP + _MARGIN_STEPS,
        min_steps=1 + _ANSWER_STEP,
        meta={"db": db.schema_name},
    )


# ---------------------------------------------------------------------------
# 模板（每个 schema 覆盖 4 类问题）
# ---------------------------------------------------------------------------


def _eco_users_by_city(rng, db, task_id):
    city = rng.choice(CITIES)
    sql = f"SELECT name FROM users WHERE city = '{city}'"
    return _sql_task(task_id, "single_table_filter", f"查询来自{city}的用户姓名", sql, db)


def _eco_products_above(rng, db, task_id):
    price = rng.choice([100, 200, 300])
    sql = f"SELECT name FROM products WHERE price > {price}"
    return _sql_task(task_id, "single_table_filter", f"查询价格超过{price}元的商品名称", sql, db)


def _eco_orders_join_users(rng, db, task_id):
    sql = "SELECT o.order_id, u.name FROM orders o JOIN users u ON o.user_id = u.user_id"
    return _sql_task(task_id, "join", "查询每笔订单对应的下单用户姓名", sql, db)


def _eco_users_per_city(rng, db, task_id):
    sql = "SELECT city, COUNT(*) FROM users GROUP BY city"
    return _sql_task(task_id, "aggregation", "统计每个城市的用户数量", sql, db)


def _eco_top_orders(rng, db, task_id):
    top = rng.choice([3, 5, 10])
    sql = f"SELECT order_id FROM orders ORDER BY amount DESC LIMIT {top}"
    return _sql_task(task_id, "order_limit", f"查询消费金额最高的{top}笔订单号", sql, db)


def _emp_salary_above(rng, db, task_id):
    salary = rng.choice([5000, 8000, 10000])
    sql = f"SELECT name FROM employees WHERE salary > {salary}"
    return _sql_task(task_id, "single_table_filter", f"查询工资超过{salary}元的员工姓名", sql, db)


def _emp_dept_join(rng, db, task_id):
    sql = "SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.dept_id"
    return _sql_task(task_id, "join", "查询每位员工所在的部门名称", sql, db)


def _emp_avg_salary(rng, db, task_id):
    sql = "SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id"
    return _sql_task(task_id, "aggregation", "统计各部门的平均工资", sql, db)


def _emp_top_salary(rng, db, task_id):
    top = rng.choice([3, 5, 10])
    sql = f"SELECT name FROM employees ORDER BY salary DESC LIMIT {top}"
    return _sql_task(task_id, "order_limit", f"查询工资最高的{top}名员工姓名", sql, db)


def _inv_low_stock(rng, db, task_id):
    qty = rng.choice([50, 100, 150])
    sql = f"SELECT product_name FROM inventory WHERE quantity < {qty}"
    return _sql_task(task_id, "single_table_filter", f"查询库存量少于{qty}的商品名称", sql, db)


def _inv_warehouse_join(rng, db, task_id):
    sql = "SELECT i.product_name, w.name FROM inventory i JOIN warehouses w ON i.warehouse_id = w.warehouse_id"
    return _sql_task(task_id, "join", "查询每种商品所在的仓库名称", sql, db)


def _inv_warehouse_total(rng, db, task_id):
    sql = "SELECT warehouse_id, SUM(quantity) FROM inventory GROUP BY warehouse_id"
    return _sql_task(task_id, "aggregation", "统计每个仓库的商品总库存量", sql, db)


def _inv_lowest_stock(rng, db, task_id):
    top = rng.choice([3, 5, 10])
    sql = f"SELECT product_name FROM inventory ORDER BY quantity ASC LIMIT {top}"
    return _sql_task(task_id, "order_limit", f"查询库存量最少的{top}种商品名称", sql, db)


TEMPLATES: dict[str, list[Callable[[random.Random, Database, str], Task]]] = {
    "ecommerce": [_eco_users_by_city, _eco_products_above, _eco_orders_join_users, _eco_users_per_city, _eco_top_orders],
    "employee": [_emp_salary_above, _emp_dept_join, _emp_avg_salary, _emp_top_salary],
    "inventory": [_inv_low_stock, _inv_warehouse_join, _inv_warehouse_total, _inv_lowest_stock],
}


def generate_task(rng: random.Random, db: Database, task_id: str) -> Task:
    return rng.choice(TEMPLATES[db.schema_name])(rng, db, task_id)


def task_generator(seed: int) -> Iterator[Task]:
    """按 seed 确定性无限生成任务（跨三类 schema 轮换采样）。"""
    rng = random.Random(seed)
    dbs: dict[str, Database] = {}
    i = 0
    while True:
        schema = rng.choice(SCHEMA_NAMES)
        if schema not in dbs:
            dbs[schema] = Database(schema)
        yield generate_task(rng, dbs[schema], f"sql_{schema}_{i:04d}")
        i += 1


def generate_tasks(seed: int, n: int) -> list[Task]:
    return list(itertools.islice(task_generator(seed), n))