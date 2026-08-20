from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class Paths:
    data: Path = field(default_factory=lambda: _resolve(_env("METERMIND_DATA_DIR", "data")))

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def bronze(self) -> Path:
        return self.data / "bronze"

    @property
    def gold(self) -> Path:
        return self.data / "gold"

    @property
    def duckdb(self) -> Path:
        return _resolve(_env("METERMIND_DUCKDB_PATH", "data/gold/metermind.duckdb"))

    @property
    def app_duckdb(self) -> Path:
        return _resolve(_env("METERMIND_APP_DB_PATH", "data/gold/metermind_app.duckdb"))

    @property
    def artifacts(self) -> Path:
        return _resolve(_env("METERMIND_ARTIFACTS_DIR", "artifacts"))

    def ensure(self) -> None:
        for p in (self.raw, self.bronze, self.gold, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


LONDON_TZ = "Europe/London"

INTERVALS_PER_DAY = 48
INTERVAL_MINUTES = 30

SLOTS_COLUMNS = [f"h{i}" for i in range(INTERVALS_PER_DAY)]

TARIFF_STANDARD = "Std"
TARIFF_TOU = "ToU"

PRE_PERIOD = (date(2012, 1, 1), date(2012, 12, 31))
TRIAL_PERIOD = (date(2013, 1, 1), date(2013, 12, 31))

TOU_PRICES_P_PER_KWH = {"Low": 3.99, "Normal": 11.76, "High": 67.20}
FLAT_PRICE_P_PER_KWH = 14.228


MIN_DAYS_HISTORY = 120

MAX_PLAUSIBLE_KWH_PER_HALFHOUR = 10.0

TARGET_COVERAGE = 0.80

BILLING_CYCLE_DAYS = 30
FORECAST_ELAPSED_DAYS = 12

MIN_CYCLE_ELAPSED_KWH = 5.0
MIN_CYCLE_TOTAL_KWH = 10.0

CYCLE_RATIO_BOUNDS = (1.0, 6.0)

RANDOM_SEED = 42


@dataclass(frozen=True)
class Settings:
    max_households: int = field(
        default_factory=lambda: int(_env("METERMIND_MAX_HOUSEHOLDS", "0"))
    )
    duckdb_memory_limit: str = field(
        default_factory=lambda: _env("METERMIND_DUCKDB_MEMORY", "3GB")
    )
    duckdb_threads: int = field(default_factory=lambda: int(_env("METERMIND_DUCKDB_THREADS", "2")))


SETTINGS = Settings()
