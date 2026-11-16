"""Quest w5 / Text-to-SQL RAG: the LLM sees the schema, writes one SELECT, we validate it, run it on a
READ-ONLY connection, retry with the error message on failure, and hand the result rows to the answer step."""
from __future__ import annotations

import re
import sqlite3

from ..types import Chunk, Hit
from .base import Strategy

SQL_SYS = (
    "You write one SQLite SELECT query that answers the question using only the given schema. "
    "Reply with only the SQL, no explanation, no markdown."
)
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|truncate)\b", re.I)


def validate_sql(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.I | re.M).strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("only a single statement is allowed")
    if not re.match(r"(?is)^(select|with)\b", sql):
        raise ValueError("only SELECT queries are allowed")
    if FORBIDDEN.search(sql):
        raise ValueError("query contains a forbidden keyword")
    return sql


def schema_text(conn: sqlite3.Connection, sample_rows: int = 3) -> str:
    parts = []
    for name, ddl in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        rows = conn.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}').fetchall()
        parts.append(f"{ddl}\n-- sample rows: {rows}")
    return "\n\n".join(parts)


class SQLRAG(Strategy):
    name = "sql"

    def __init__(self, index=None, llm=None, k=5, db_path: str = "", max_rows=50, retries=2):
        super().__init__(index, llm, k)
        self.db_path, self.max_rows, self.retries = db_path, max_rows, retries

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)  # read-only at the database level too
        try:
            prompt = f"Schema:\n{schema_text(conn)}\n\nQuestion: {query}\nSQL:"
            for _ in range(self.retries + 1):
                raw = self.llm.generate(SQL_SYS, prompt, max_tokens=300)
                try:
                    sql = validate_sql(raw)
                    cur = conn.execute(sql)
                    rows = cur.fetchmany(self.max_rows)
                    cols = [d[0] for d in cur.description or []]
                    return [Hit(Chunk("sql::0", "sql", f"SQL: {sql}\nColumns: {cols}\nRows: {rows}"), 1.0, "sql")]
                except (ValueError, sqlite3.Error) as e:
                    prompt += f"\n\nThe previous query failed.\nSQL: {raw.strip()}\nError: {e}\nWrite a corrected query.\nSQL:"
            return []
        finally:
            conn.close()
