from __future__ import annotations

import logging
from pathlib import Path

from metermind.config import PATHS, SLOTS_COLUMNS
from metermind.warehouse.duck import connect

logger = logging.getLogger(__name__)

BRONZE_SUBDIR = "lcl_daily"


def _source_glob(source_dir: Path | None = None) -> str:
    root = Path(source_dir) if source_dir else PATHS.raw
    files = sorted(root.glob("chunk_*.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"No chunk_*.csv.gz found in {root}. "
            "Run scripts/extract_local.py on the machine holding the raw archive, "
            "then copy its output here."
        )
    logger.info("Reading %d chunk file(s) from %s", len(files), root)
    return str(root / "chunk_*.csv.gz").replace("\\", "/")


def ingest_lcl(source_dir: Path | None = None) -> dict[str, int]:
    PATHS.ensure()
    target_root = PATHS.bronze / BRONZE_SUBDIR
    target_root.mkdir(parents=True, exist_ok=True)

    glob = _source_glob(source_dir)
    slot_sums = ",\n            ".join(f"sum({c}) AS {c}" for c in SLOTS_COLUMNS)

    written: dict[str, int] = {}
    with connect() as con:
        con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _merged AS
        SELECT
            lcl_id,
            CAST(date_local AS DATE) AS date_local,
            any_value(tariff)        AS tariff,
            sum(n_intervals)         AS n_intervals,
            sum(kwh_total)           AS kwh_total,
            {slot_sums}
        FROM read_csv('{glob}', header = true, compression = 'gzip', union_by_name = true)
        GROUP BY lcl_id, CAST(date_local AS DATE)
        """)

        total, households = con.execute(
            "SELECT count(*), count(DISTINCT lcl_id) FROM _merged"
        ).fetchone()
        logger.info(
            "  %s household-days after merging chunk boundaries, %s households",
            f"{total:,}", f"{households:,}",
        )

        cohorts = con.execute(
            "SELECT tariff, count(DISTINCT lcl_id) AS n FROM _merged GROUP BY tariff ORDER BY tariff"
        ).fetchall()
        logger.info("  cohorts: %s", dict(cohorts))

        years = [r[0] for r in con.execute(
            "SELECT DISTINCT year(date_local) AS y FROM _merged ORDER BY y"
        ).fetchall()]

        for year in years:
            out = target_root / f"year={int(year)}"
            out.mkdir(parents=True, exist_ok=True)
            path = (out / "data.parquet").as_posix()
            con.execute(f"""
                COPY (SELECT * FROM _merged WHERE year(date_local) = {int(year)})
                TO '{path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            n = con.execute(
                f"SELECT count(*) FROM _merged WHERE year(date_local) = {int(year)}"
            ).fetchone()[0]
            written[str(int(year))] = int(n)
            logger.info("  %s  %s household-days", int(year), f"{n:,}")

    logger.info("Bronze written: %s household-days total", f"{sum(written.values()):,}")
    return written


def bronze_glob() -> str:
    return str(PATHS.bronze / BRONZE_SUBDIR / "**" / "*.parquet").replace("\\", "/")
