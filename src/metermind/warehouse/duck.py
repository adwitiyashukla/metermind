from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from metermind.config import PATHS, SETTINGS

logger = logging.getLogger(__name__)


@contextmanager
def connect(
    path: Path | str | None = None,
    read_only: bool = False,
    memory_limit: str | None = None,
    threads: int | None = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    target = Path(path) if path else PATHS.duckdb
    target.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(target), read_only=read_only)
    try:
        con.execute(f"SET memory_limit='{memory_limit or SETTINGS.duckdb_memory_limit}'")
        con.execute(f"SET threads={threads or SETTINGS.duckdb_threads}")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET enable_progress_bar=false")
        yield con
    finally:
        con.close()


def query(sql: str, params: list | None = None, path: Path | str | None = None) -> pd.DataFrame:
    with connect(path, read_only=True) as con:
        return con.execute(sql, params or []).df()


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    found = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(found and found[0])


def row_count(con: duckdb.DuckDBPyConnection, name: str) -> int:
    if not table_exists(con, name):
        return 0
    return int(con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0])


def summarise(path: Path | str | None = None) -> pd.DataFrame:
    with connect(path, read_only=True) as con:
        tables = (
            con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' ORDER BY table_name"
            )
            .df()["table_name"]
            .tolist()
        )
        return pd.DataFrame([{"table": t, "rows": row_count(con, t)} for t in tables])
