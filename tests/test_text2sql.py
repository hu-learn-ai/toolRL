"""Text2SQL 环境测试：DB 安全/确定性、结果集比对、任务生成、judge 黄金路径。"""

import json

import pytest

from envs.text2sql import (
    SQLBlocked,
    Text2SQLEnv,
    build_database,
    generate_tasks,
    judge,
    results_equal,
)
from envs.text2sql.task_generator import SQL_TOOL


def _tool(query):
    return {"raw": "<tool_call>" + json.dumps({"name": SQL_TOOL, "arguments": {"query": query}}, ensure_ascii=False) + "</tool_call>"}


def _answer(text="ok"):
    return {"raw": "<answer>" + text + "</answer>"}


def _gold_trajectory(task):
    return [_tool(task.gold.calls[0].params["query"]), _answer()]


# ---------------------------------------------------------------------------
# DB 安全 / 确定性
# ---------------------------------------------------------------------------


def test_database_deterministic():
    a = build_database("employee")
    b = build_database("employee")
    assert a.execute("SELECT COUNT(*) FROM employees") == b.execute("SELECT COUNT(*) FROM employees")


def test_database_rejects_unsafe():
    db = build_database("employee")
    unsafe = [
        "DROP TABLE employees",
        "DELETE FROM employees",
        "INSERT INTO employees VALUES (1,'x',1,1,'2020')",
        "PRAGMA table_info(employees)",
        "UPDATE employees SET salary=0",
        "SELECT name FROM employees; DROP TABLE employees",
    ]
    for sql in unsafe:
        with pytest.raises(SQLBlocked):
            db.execute(sql)


def test_database_select_works():
    db = build_database("employee")
    cols, rows = db.execute("SELECT name FROM employees WHERE salary > 15000")
    assert cols == ["name"]
    assert rows  # 非空


def test_database_schema_ddl():
    db = build_database("ecommerce")
    assert "users(" in db.schema_ddl()
    assert "orders(" in db.schema_ddl()


# ---------------------------------------------------------------------------
# 结果集比对
# ---------------------------------------------------------------------------


def test_results_equal_column_order():
    db = build_database("employee")
    c1, r1 = db.execute("SELECT emp_id, name FROM employees LIMIT 5")
    c2, r2 = db.execute("SELECT name, emp_id FROM employees LIMIT 5")
    assert results_equal(c1, r1, c2, r2)  # 列序归一化


def test_results_equal_row_order():
    db = build_database("employee")
    c, rows = db.execute("SELECT name FROM employees LIMIT 5")
    rev = list(reversed(rows))
    assert results_equal(c, rows, c, rev, order_matters=True) is (rows == rev)
    assert results_equal(c, rows, c, rev, order_matters=False)  # 行序归一化


def test_results_equal_type_normalization():
    assert results_equal(["x"], [(1,)], ["x"], [(1.0,)])  # int vs float
    assert results_equal(["x"], [(None,)], ["x"], [(None,)])


def test_results_equal_null_vs_string_distinct():
    db = build_database("employee")
    c1, r1 = db.execute("SELECT NULL AS x")
    c2, r2 = db.execute("SELECT 'null' AS x")
    assert not results_equal(c1, r1, c2, r2)


# ---------------------------------------------------------------------------
# 任务生成
# ---------------------------------------------------------------------------


def test_gold_sql_valid_and_nonempty():
    for task in generate_tasks(0, 200):
        db = build_database(task.meta["db"])
        cols, rows = db.execute(task.gold.calls[0].params["query"])
        assert rows, f"{task.task_id} gold 结果为空"
        assert task.gold.calls[0].api == SQL_TOOL


def test_categories_cover_all():
    cats = {t.category for t in generate_tasks(0, 400)}
    assert cats == {"single_table_filter", "join", "aggregation", "order_limit"}


def test_schemas_cover_all():
    schemas = {t.meta["db"] for t in generate_tasks(0, 400)}
    assert schemas == {"ecommerce", "employee", "inventory"}


def test_generator_deterministic():
    assert generate_tasks(0, 20) == generate_tasks(0, 20)
    assert generate_tasks(1, 20) != generate_tasks(2, 20)


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


def test_gold_trajectory_scores_full():
    for task in generate_tasks(0, 200):
        r = judge(task, _gold_trajectory(task))
        assert r.format == 1.0, task.task_id
        assert r.correct == 1.0, task.task_id
        assert r.answer == 1.0, task.task_id
        assert r.success is True, task.task_id


def test_wrong_sql_fails():
    task = generate_tasks(0, 1)[0]
    r = judge(task, [_tool("SELECT 1"), _answer()])
    assert r.correct == 0.0
    assert r.success is False


def test_unsafe_sql_correct_zero():
    task = generate_tasks(0, 1)[0]
    r = judge(task, [_tool("DROP TABLE users"), _answer()])
    assert r.correct == 0.0
    assert r.success is False


def test_format_requires_sql_and_answer():
    task = generate_tasks(0, 1)[0]
    assert judge(task, [_answer()]).format == 0.0  # 只有 answer、没有 SQL
    traj = [_tool(task.gold.calls[0].params["query"]), _answer(), _answer()]
    assert judge(task, traj).format == 0.0  # 两个 answer


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------


def test_env_reset_returns_schema():
    env = Text2SQLEnv()
    task = generate_tasks(0, 1)[0]
    obs = env.reset(task)
    assert obs["instruction"] == task.instruction
    assert obs["schema"]
    assert obs["tools"][0]["name"] == SQL_TOOL


def test_env_step_executes_sql():
    env = Text2SQLEnv()
    task = generate_tasks(0, 1)[0]
    env.reset(task)
    out = env.step({"api": SQL_TOOL, "params": {"query": task.gold.calls[0].params["query"]}})
    assert "error" not in out["observation"]
    assert "columns" in out["observation"]
    assert out["done"] is False


def test_env_step_blocked_sql():
    env = Text2SQLEnv()
    task = generate_tasks(0, 1)[0]
    env.reset(task)
    out = env.step({"api": SQL_TOOL, "params": {"query": "DELETE FROM users"}})
    assert "error" in out["observation"]


def test_env_judge_delegates():
    env = Text2SQLEnv()
    task = generate_tasks(0, 1)[0]
    env.reset(task)
    r = env.judge(_gold_trajectory(task))
    assert r.success is True