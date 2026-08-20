from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metermind.models.billforecast import (
    FEATURE_COLUMNS,
    TARGET,
    BillForecaster,
    build_cycle_features,
    chronological_split,
    evaluate,
)


@pytest.fixture(scope="module")
def cycles(synthetic_daily):
    return build_cycle_features(synthetic_daily, elapsed_days=12, cycle_days=30)


def test_every_declared_feature_is_produced(cycles):
    missing = set(FEATURE_COLUMNS) - set(cycles.columns)
    assert not missing, f"declared but never built: {sorted(missing)}"


def test_target_is_the_whole_cycle_not_just_the_elapsed_part(cycles):
    assert (cycles[TARGET] >= cycles["elapsed_kwh"] - 1e-6).all()
    assert (cycles[TARGET] > cycles["elapsed_kwh"] * 1.5).mean() > 0.8


def test_features_do_not_read_past_the_cutoff(synthetic_daily):
    slots = [f"h{i}" for i in range(48)]
    baseline = build_cycle_features(synthetic_daily, elapsed_days=12, cycle_days=30)

    wall = pd.Timestamp("2013-06-01")
    corrupted = synthetic_daily.copy()
    corrupted["date_local"] = pd.to_datetime(corrupted["date_local"])
    after = corrupted["date_local"] > wall

    rng = np.random.default_rng(0)
    noise = rng.uniform(500, 900, size=int(after.sum()))
    corrupted.loc[after, "kwh_total"] = noise
    corrupted.loc[after, "kwh_peak"] = noise / 4
    corrupted.loc[after, "kwh_night"] = noise / 4
    corrupted.loc[after, "load_factor"] = 0.99
    for slot in slots:
        corrupted.loc[after, slot] = noise / 48

    rebuilt = build_cycle_features(corrupted, elapsed_days=12, cycle_days=30)

    key = ["lcl_id", "cycle_start"]
    safe = baseline["cutoff_date"] <= wall
    assert safe.sum() > 100, "need a decent number of unaffected cycles to test"

    left = baseline.loc[safe].set_index(key).sort_index()
    right = rebuilt.set_index(key).sort_index().loc[left.index]

    for column in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            left[column], right[column], check_names=False,
            obj=f"feature '{column}' changed when data after the cutoff was corrupted",
        )


def test_the_corruption_in_the_leakage_test_is_real(synthetic_daily):
    slots = [f"h{i}" for i in range(48)]
    baseline = build_cycle_features(synthetic_daily, elapsed_days=12, cycle_days=30)

    wall = pd.Timestamp("2013-06-01")
    corrupted = synthetic_daily.copy()
    corrupted["date_local"] = pd.to_datetime(corrupted["date_local"])
    after = corrupted["date_local"] > wall
    corrupted.loc[after, "kwh_total"] = 700.0
    for slot in slots:
        corrupted.loc[after, slot] = 700.0 / 48

    rebuilt = build_cycle_features(corrupted, elapsed_days=12, cycle_days=30)

    key = ["lcl_id", "cycle_start"]
    left = baseline.set_index(key).sort_index()
    right = rebuilt.set_index(key).sort_index()
    common = left.index.intersection(right.index)
    moved = ~np.isclose(left.loc[common, TARGET], right.loc[common, TARGET])
    assert moved.any(), "corruption never reached any target, so the leakage test is vacuous"


def test_history_features_never_use_the_current_cycle(cycles):
    first_cycles = cycles.sort_values(["lcl_id", "cycle_start"]).groupby("lcl_id").head(1)
    assert len(first_cycles) > 0


def test_chronological_split_never_overlaps(cycles):
    train, calib, test = chronological_split(cycles)
    assert train["cycle_start"].max() <= calib["cycle_start"].min()
    assert calib["cycle_start"].max() <= test["cycle_start"].min()
    assert len(train) + len(calib) + len(test) == len(cycles)


def test_split_leaves_enough_to_calibrate(cycles):
    _, calib, _ = chronological_split(cycles)
    assert len(calib) >= 50, "conformal calibration needs a meaningful held-out set"


def test_quantile_heads_never_cross(cycles):
    train, calib, test = chronological_split(cycles)
    model = BillForecaster().fit(train, quick=True)
    raw = model.predict_raw(test)
    assert (raw["upper"] >= raw["lower"] - 1e-9).all()


def test_conformal_intervals_are_wider_and_better_covered(cycles):
    train, calib, test = chronological_split(cycles)
    model = BillForecaster().fit(train, quick=True).calibrate(calib, groups=None)
    pred = model.predict(test, groups=None)

    assert (pred["upper_conformal"] - pred["lower_conformal"]
            >= pred["upper"] - pred["lower"] - 1e-9).all()

    truth = test[TARGET].to_numpy(dtype=float)
    inside_raw = ((truth >= pred["lower"]) & (truth <= pred["upper"])).mean() * 100
    inside_cal = ((truth >= pred["lower_conformal"])
                  & (truth <= pred["upper_conformal"])).mean() * 100
    assert inside_cal >= inside_raw


def test_model_beats_the_run_rate_baseline(cycles):
    train, calib, test = chronological_split(cycles)
    model = BillForecaster().fit(train, quick=True).calibrate(calib, groups=None)
    scores = evaluate(model, test)
    assert scores["mape_pct"] < scores["naive_runrate_mape_pct"], (
        "a model that cannot beat scaling up the elapsed days has not earned its place"
    )


def test_save_and_load_round_trip(cycles, tmp_path):
    train, calib, test = chronological_split(cycles)
    model = BillForecaster().fit(train, quick=True).calibrate(calib, groups=None)
    model.save(tmp_path / "bf")
    restored = BillForecaster.load(tmp_path / "bf")

    original = model.predict(test, groups=None)["point"].to_numpy()
    reloaded = restored.predict(test, groups=None)["point"].to_numpy()
    np.testing.assert_allclose(original, reloaded, rtol=1e-9)
    assert restored.calibration.q_hat == pytest.approx(model.calibration.q_hat)
