"""Text2SQL 环境（design_zh.md §2.3）。"""

from .db import Database, SQLBlocked, build_database, is_safe
from .env import Text2SQLEnv
from .judge import judge, results_equal
from .schemas import SCHEMA_NAMES, TABLES, schema_ddl
from .task_generator import SQL_SPEC, SQL_TOOL, generate_tasks, task_generator

__all__ = [
    "Database",
    "SQLBlocked",
    "build_database",
    "is_safe",
    "Text2SQLEnv",
    "judge",
    "results_equal",
    "SCHEMA_NAMES",
    "TABLES",
    "schema_ddl",
    "SQL_SPEC",
    "SQL_TOOL",
    "generate_tasks",
    "task_generator",
]