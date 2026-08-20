from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from metermind.config import PATHS
from metermind.warehouse.duck import connect, row_count

logger = logging.getLogger(__name__)

APP_TABLES = [
    "dim_household",
    "dim_date",
    "persona_summary",
    "persona_shapes",
    "household_persona",
    "treatment_effects",
    "parallel_trends",
    "qini_curve",
    "uplift_deciles",
    "uplift_scores",
    "propensity_diagnostics",
    "bill_forecasts",
    "dq_results",
    "dq_scorecard",
]

def export_for_app(destination: Path | None = None) -> Path:
    target = Path(destination) if destination else PATHS.app_duckdb
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".wal"):
        candidate = Path(str(target) + suffix)
        if candidate.exists():
            candidate.unlink()

    source = PATHS.duckdb
    if not source.exists():
        raise FileNotFoundError(f"Warehouse not found at {source}. Run `metermind build` first.")

    manifest: dict[str, int] = {}
    with connect(target) as con:
        con.execute(f"ATTACH '{source.as_posix()}' AS wh (READ_ONLY)")

        available = set(
            con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'wh'"
            ).df()["table_name"]
        )

        for table in APP_TABLES:
            if table in available:
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM wh.{table}")
                manifest[table] = row_count(con, table)

        con.execute("DETACH wh")
        con.execute("VACUUM")

    size_mb = target.stat().st_size / 1e6
    (PATHS.artifacts / "export_manifest.json").write_text(json.dumps({
        "database": target.name,
        "size_mb": round(size_mb, 2),
        "tables": manifest,
        "exported_at_utc": datetime.now(UTC).isoformat(),
    }, indent=2))

    logger.info("App database written: %s (%.1f MB)", target, size_mb)
    for table, count in manifest.items():
        logger.info("  %-26s %10s rows", table, f"{count:,}")
    return target
