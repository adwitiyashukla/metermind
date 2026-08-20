from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("METERMIND_DUCKDB_MEMORY", "1GB")
os.environ.setdefault("METERMIND_DUCKDB_THREADS", "2")

SLOTS = [f"h{i}" for i in range(48)]


def _shape(kind: str, rng) -> np.ndarray:
    hours = np.arange(48) / 2
    if kind == "evening":
        curve = 0.3 + 1.6 * np.exp(-((hours - 19) ** 2) / 6) + 0.3 * np.exp(-((hours - 8) ** 2) / 4)
    elif kind == "twin":
        curve = 0.3 + 1.0 * np.exp(-((hours - 8) ** 2) / 3) + 1.2 * np.exp(-((hours - 19) ** 2) / 5)
    elif kind == "night":
        curve = 0.4 + 1.8 * np.exp(-((hours - 2.5) ** 2) / 8)
    elif kind == "daytime":
        curve = 0.3 + 1.3 * np.exp(-((hours - 13) ** 2) / 12)
    else:
        curve = np.ones(48)
    curve = curve * (1 + rng.normal(0, 0.06, 48))
    curve = np.clip(curve, 0.01, None)
    return curve / curve.sum()


@pytest.fixture(scope="session")
def synthetic_daily() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    archetypes = ["evening", "twin", "night", "daytime", "flat"]
    dates = pd.date_range("2012-01-01", "2013-12-31", freq="D")

    frames = []
    for i in range(60):
        kind = archetypes[i % len(archetypes)]
        treated = i % 3 == 0
        scale = rng.gamma(6, 1.6)

        profiles = np.vstack([_shape(kind, rng) for _ in range(len(dates))])
        doy = np.asarray(dates.dayofyear, dtype=float)
        daily_total = scale * (1 + 0.25 * np.cos(2 * np.pi * (doy - 20) / 365.25))
        daily_total = daily_total * (1 + rng.normal(0, 0.12, len(dates)))
        daily_total = np.clip(daily_total, 0.4, None)
        matrix = profiles * daily_total[:, None]

        frame = pd.DataFrame(matrix, columns=SLOTS)
        frame.insert(0, "lcl_id", f"MAC{i:06d}")
        frame.insert(1, "date_local", dates)
        frame["tariff_group"] = "ToU" if treated else "Std"
        frame["is_treated"] = treated
        frame["n_intervals"] = 48
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)

    panel.loc[panel["date_local"] == "2012-03-25", "n_intervals"] = 46
    panel.loc[panel["date_local"] == "2012-10-28", "n_intervals"] = 50
    panel.loc[panel["date_local"] == "2013-03-31", "n_intervals"] = 46
    panel.loc[panel["date_local"] == "2013-10-27", "n_intervals"] = 50

    vacant = (panel["lcl_id"] == "MAC000007") & (panel["date_local"].between("2013-06-01", "2013-06-20"))
    panel.loc[vacant, SLOTS] = 0.0

    panel["kwh_total"] = panel[SLOTS].sum(axis=1)
    panel["kwh_night"] = panel[[f"h{i}" for i in range(0, 14)]].sum(axis=1)
    panel["kwh_morning"] = panel[[f"h{i}" for i in range(14, 24)]].sum(axis=1)
    panel["kwh_afternoon"] = panel[[f"h{i}" for i in range(24, 32)]].sum(axis=1)
    panel["kwh_peak"] = panel[[f"h{i}" for i in range(32, 40)]].sum(axis=1)
    panel["kwh_late"] = panel[[f"h{i}" for i in range(40, 48)]].sum(axis=1)
    panel["kwh_peak_halfhour"] = panel[SLOTS].max(axis=1)
    panel["is_weekend"] = panel["date_local"].dt.dayofweek >= 5
    panel["flag_dst_day"] = panel["n_intervals"].isin([46, 50])
    panel["flag_zero_day"] = panel["kwh_total"] <= 0
    panel["flag_incomplete_day"] = panel["n_intervals"] != 48
    panel["is_modelling_day"] = ~(panel["flag_zero_day"] | panel["flag_incomplete_day"])
    panel["load_factor"] = panel["kwh_total"] / 48 / panel["kwh_peak_halfhour"].replace(0, np.nan)
    return panel


@pytest.fixture(scope="session")
def slots() -> list[str]:
    return list(SLOTS)
