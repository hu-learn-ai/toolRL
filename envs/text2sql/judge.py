"""Text2SQL 判定：结果集一致性（design_zh.md §2.3）。

成功条件 = 模型 SQL 与 gold SQL 的结果集一致，比对前做：
- 列序归一化（按列名排序）；
- 行序归一化（按全部列排序，仅当 category != "order_limit" 时——排序分页题顺序即语义）；
- NULL 与类型归一化（None、bool、int/float、文本统一到规范串）。

R_format 复用 Qwen3 格式判定（parse_turn）；R_correct / R_answer 都映射到结果集
一致性（Text2SQL 的"结果"就是它的"答案"，对应 api_sandbox 的 correct+answer 两档）。
"""

from __future__ import annotations

from ..api_sandbox.common import validate_params
from ..api_sandbox.judge import parse_turn, raw_of, step_penalty
from ..base_env import JudgeResult
from ..task_schema import Task
from .db import Database, build_database
from .task_generator import SQL_SPEC, SQL_TOOL


def _norm_value(v) -> str:
    if v is None:
        return "\x00null"
    if isinstance(v, bool):
        return "\x01" + ("1" if v else "0")
    if isinstance(v, float):
        return "\x02" + (str(int(v)) if v.is_integer() else repr(v))
    if isinstance(v, int):
        return "\x02" + str(v)
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return "\x03" + str(v)


def _normalize(columns: list[str], rows: list[tuple]) -> list[tuple]:
    """每行按列名排序成 (列名, 归一化值) 元组，返回行列表（未排序）。"""
    norm = []
    for row in rows:
        pairs = sorted(zip(columns, row), key=lambda p: str(p[0]))
        norm.append(tuple((str(c), _norm_value(v)) for c, v in pairs))
    return norm


def results_equal(cols_a, rows_a, cols_b, rows_b, order_matters: bool = False) -> bool:
    a = _normalize(cols_a, rows_a)
    b = _normalize(cols_b, rows_b)
    if order_matters:
        return a == b
    return sorted(a) == sorted(b)


def _result_equality(db: Database, gold_sql: str, model_sql: str, order_matters: bool) -> float:
    try:
        gold_cols, gold_rows = db.execute(gold_sql)
    except Exception:
        return 0.0  # gold 不可执行（不应发生）
    try:
        model_cols, model_rows = db.execute(model_sql)
    except Exception:
        return 0.0  # 模型 SQL 非法 / 被拦截 → 错误
    return 1.0 if results_equal(gold_cols, gold_rows, model_cols, model_rows, order_matters) else 0.0


def judge(task: Task, trajectory: list, db: Database | None = None) -> JudgeResult:
    if db is None:
        db = build_database(task.meta.get("db", "ecommerce"))
    turns = [raw_of(x) for x in trajectory]
    parsed = [parse_turn(t) for t in turns]
    n = len(turns)

    # R_format：结构合法（每个 tool_call 都是 sql.execute 且 query 非空、恰好一个 answer 且末尾）
    format_ok = True
    sql_turns: list[str] = []
    for p in parsed:
        if p.kind == "invalid":
            format_ok = False
        elif p.kind == "tool_call":
            if p.tool_name != SQL_TOOL or validate_params(p.arguments, SQL_SPEC.parameters) is not None:
                format_ok = False
            else:
                sql_turns.append(p.arguments["query"])
    answer_idx = [i for i, p in enumerate(parsed) if p.kind == "answer"]
    if len(answer_idx) != 1 or answer_idx[0] != n - 1 or not sql_turns:
        format_ok = False
    r_format = 1.0 if format_ok else 0.0

    # R_correct / R_answer：执行最后一条 SQL，与 gold 结果集比对
    gold_sql = task.gold.calls[0].params["query"] if task.gold.calls else ""
    model_sql = sql_turns[-1] if sql_turns else ""
    order_matters = task.category == "order_limit"
    r_correct = _result_equality(db, gold_sql, model_sql, order_matters)
    r_answer = r_correct

    # R_steps：步数 = 助手回合数
    r_steps = step_penalty(n, task)

    success = r_format == 1.0 and r_correct == 1.0
    # 参数正确率（§5.2）：sql.execute 只有一个 "query" 参数，结果集一致即参数正确。
    param_total = 1
    matched_params = 1 if r_correct == 1.0 else 0
    return JudgeResult(
        format=r_format,
        correct=r_correct,
        answer=r_answer,
        steps=r_steps,
        success=success,
        matched_params=matched_params,
        param_total=param_total,
    )


__all__ = ["results_equal", "judge"]