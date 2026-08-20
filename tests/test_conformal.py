from __future__ import annotations

import numpy as np
import pytest

from metermind.models.conformal import (
    ConformalCalibration,
    conformal_quantile,
    conformity_scores,
    coverage_report,
    empirical_coverage,
    mean_interval_width,
)


@pytest.fixture
def broken_interval():
    rng = np.random.default_rng(3)
    y = rng.normal(50, 12, 6000)
    lo = np.full_like(y, 48.0)
    hi = np.full_like(y, 52.0)
    return y, lo, hi


def test_conformity_score_is_negative_inside_the_interval():
    scores = conformity_scores([5.0], [0.0], [10.0])
    assert scores[0] < 0


def test_conformity_score_measures_the_miss_distance():
    assert conformity_scores([15.0], [0.0], [10.0])[0] == pytest.approx(5.0)
    assert conformity_scores([-4.0], [0.0], [10.0])[0] == pytest.approx(4.0)


def test_finite_sample_correction_is_applied():
    scores = np.arange(10, dtype=float)
    corrected = conformal_quantile(scores, alpha=0.2)
    naive = float(np.quantile(scores, 0.8, method="higher"))
    assert corrected > naive


def test_calibration_repairs_undercoverage(broken_interval):
    y, lo, hi = broken_interval
    split = len(y) // 2

    before = empirical_coverage(y[split:], lo[split:], hi[split:])
    cal = ConformalCalibration.fit(y[:split], lo[:split], hi[:split], alpha=0.2)
    new_lo, new_hi = cal.apply(lo[split:], hi[split:])
    after = empirical_coverage(y[split:], new_lo, new_hi)

    assert before < 40, "fixture should start badly undercovered"
    assert 76 <= after <= 84, f"conformal coverage {after:.1f}% missed the 80% target"


def test_coverage_guarantee_holds_across_many_seeds():
    achieved = []
    for seed in range(25):
        rng = np.random.default_rng(seed)
        y = rng.gamma(shape=4.0, scale=20.0, size=2000)
        centre = np.full_like(y, y.mean())
        lo, hi = centre - 3, centre + 3
        cal = ConformalCalibration.fit(y[:1000], lo[:1000], hi[:1000], alpha=0.2)
        nlo, nhi = cal.apply(lo[1000:], hi[1000:])
        achieved.append(empirical_coverage(y[1000:], nlo, nhi))

    mean_coverage = float(np.mean(achieved))
    assert mean_coverage >= 78.0, f"mean coverage {mean_coverage:.2f}% is below the guarantee"


def test_widening_never_shrinks_the_interval(broken_interval):
    y, lo, hi = broken_interval
    cal = ConformalCalibration.fit(y, lo, hi, alpha=0.2)
    nlo, nhi = cal.apply(lo, hi)
    assert (nhi - nlo >= hi - lo - 1e-9).all()


def test_grouped_calibration_fixes_conditional_coverage():
    rng = np.random.default_rng(5)
    n = 8000
    group = np.where(rng.random(n) < 0.5, "low", "high")
    y = np.where(group == "low", rng.normal(20, 2, n), rng.normal(60, 20, n))
    lo = np.where(group == "low", 19.0, 58.0)
    hi = np.where(group == "low", 21.0, 62.0)

    split = n // 2
    global_cal = ConformalCalibration.fit(y[:split], lo[:split], hi[:split], alpha=0.2)
    g_lo, g_hi = global_cal.apply(lo[split:], hi[split:])
    global_report = coverage_report(y[split:], g_lo, g_hi, group[split:])

    grouped_cal = ConformalCalibration.fit(
        y[:split], lo[:split], hi[:split], alpha=0.2, groups=group[:split]
    )
    m_lo, m_hi = grouped_cal.apply(lo[split:], hi[split:], groups=group[split:])
    grouped_report = coverage_report(y[split:], m_lo, m_hi, group[split:])

    def worst(report):
        per_group = report[report["group"] != "ALL"]
        return float(per_group["coverage_gap_pp"].abs().max())

    assert grouped_cal.is_grouped
    assert worst(grouped_report) < worst(global_report), (
        "grouped calibration should improve the worst-served group"
    )


def test_small_groups_fall_back_to_the_global_correction():
    rng = np.random.default_rng(9)
    n = 1200
    group = np.array(["big"] * (n - 10) + ["tiny"] * 10)
    y = rng.normal(10, 4, n)
    lo, hi = np.full(n, 9.0), np.full(n, 11.0)

    cal = ConformalCalibration.fit(y, lo, hi, alpha=0.2, groups=group)
    assert cal.group_counts["tiny"] == 10
    assert cal.group_q_hat["tiny"] == pytest.approx(cal.q_hat)


def test_round_trip_through_json(tmp_path):
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 500)
    cal = ConformalCalibration.fit(y, y - 0.1, y + 0.1, alpha=0.2)
    path = cal.save(tmp_path / "cal.json")
    restored = ConformalCalibration.load(path)
    assert restored.q_hat == pytest.approx(cal.q_hat)
    assert restored.n_calibration == cal.n_calibration


def test_coverage_report_flags_the_gap():
    y = np.array([1.0, 2.0, 3.0, 40.0])
    report = coverage_report(y, np.zeros(4), np.full(4, 5.0), target=80.0)
    row = report[report["group"] == "ALL"].iloc[0]
    assert row["coverage_pct"] == pytest.approx(75.0)
    assert row["coverage_gap_pp"] == pytest.approx(-5.0)


def test_mean_width_ignores_non_finite():
    assert mean_interval_width([0, np.nan], [2, np.inf]) == pytest.approx(2.0)
