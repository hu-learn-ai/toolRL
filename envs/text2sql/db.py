"""只读 SQLite 执行器（design_zh.md §2.3 安全约束）。

- 连接建库后开启 ``PRAGMA query_only = ON``（只读）；
- 执行前用黑名单拦截 DROP/DELETE/UPDATE/INSERT/PRAGMA 等危险语句，只放行 SELECT/WITH；
- 用 progress_handler 实现执行超时（默认 5 秒）。
"""

from __future__ import annotations

import random
import re
import sqlite3
import time

from .schemas import DB_SEED, TABLES, generate_rows, schema_ddl

# 危险关键字黑名单（词边界匹配，防止 ``SELECT ...; DROP ...`` 之类注入）
_FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|PRAGMA|ALTER|CREATE|ATTACH|DETACH|REPLACE|VACUUM|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK|TRUNCATE)\b",
    re.IGNORECASE,
)


class SQLBlocked(Exception):
    """SQL 被安全策略拦截。"""


def is_safe(sql: str) -> bool:
    if not isinstance(sql, str) or not sql.strip():
        return False
    if _FORBIDDEN.search(sql):
        return False
    stripped = sql.strip().lstrip(";").lstrip("(").strip()
    return stripped.upper().startswith("SELECT") or stripped.upper().startswith("WITH")


class Database:
    def __init__(self, schema_name: str, seed: int = DB_SEED, timeout: float = 5.0):
        if schema_name not in TABLES:
            raise ValueError(f"未知 schema: {schema_name}")
        self.schema_name = schema_name
        self._timeout = timeout
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA query_only = OFF")  # 建库阶段允许写
        for table in TABLES[schema_name]:
            cols = ", ".join(f"{c} {ty}" for c, ty in table.columns)
            self._conn.execute(f"CREATE TABLE {table.name} ({cols})")
        rows = generate_rows(schema_name, random.Random(seed))
        for table in TABLES[schema_name]:
            data = rows[table.name]
            placeholders = ", ".join("?" * len(table.columns))
            self._conn.executemany(f"INSERT INTO {table.name} VALUES ({placeholders})", data)
        self._conn.commit()
        self._conn.execute("PRAGMA query_only = ON")  # 之后只读

    def schema_ddl(self) -> str:
        return schema_ddl(self.schema_name)

    def execute(self, sql: str) -> tuple[list[str], list[tuple]]:
        """执行只读查询，返回 (列名列表, 行列表)。危险/非法 SQL 抛异常。"""
        if not is_safe(sql):
            raise SQLBlocked(f"禁止的 SQL 语句: {sql}")
        deadline = time.monotonic() + self._timeout

        def _abort() -> None:
            if time.monotonic() > deadline:
                raise sqlite3.OperationalError("SQL 执行超时")

        self._conn.set_progress_handler(_abort, 1000)
        try:
            cur = self._conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return columns, rows
        finally:
            self._conn.set_progress_handler(None, 0)


def build_database(schema_name: str) -> Database:
    """按固定种子重建数据库（judge 复现 gold 结果用）。"""
    return Database(schema_name)