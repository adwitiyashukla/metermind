from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metermind.models.uplift import (
    UpliftTLearner,
    estimate_att_did,
    estimate_att_ipw_did,
    estimate_att_naive,
    household_deltas,
    parallel_trends,
    pre_trend_divergence,
    qini_coefficient,
    qini_curve,
    uplift_by_decile,
)

TRUE_EFFECT_BASE = -0.06
HETEROGENEITY = 0.11


def _panel(seed: int = 7, n_treated: int = 700, n_control: int = 1800, bias: float = 0.35):
    rng = np.random.default_rng(seed)

    def cohort(n, treated):
        base = rng.gamma(4, 0.22, n) + (bias if treated else 0.0)
        return pd.DataFrame({
            "lcl_id": [f"{'T' if treated else 'C'}{i:05d}" for i in range(n)],
            "is_treated": treated,
            "base": base,
        })

    households = pd.concat([cohort(n_treated, True), cohort(n_control, False)], ignore_index=True)
    centred = households["base"] - households["base"].mean()
    households["true_effect"] = np.where(
        households["is_treated"], -(abs(TRUE_EFFECT_BASE) + HETEROGENEITY * centred), 0.0
    )

    frames = []
    for period, months, active in (
        ("pre", pd.date_range("2012-01-31", "2012-12-31", freq="ME"), 0.0),
        ("post", pd.date_range("2013-01-31", "2013-12-31", freq="ME"), 1.0),
    ):
        for month in months:
            seasonal = 0.10 * np.cos(2 * np.pi * (month.month - 1) / 12)
            frames.append(pd.DataFrame({
                "lcl_id": households["lcl_id"],
                "is_treated": households["is_treated"],
                "period": period,
                "date_local": month,
                "kwh_peak": households["base"] + seasonal
                            + active * households["true_effect"]
                            + rng.normal(0, 0.10, len(households)),
            }))
    return pd.concat(frames, ignore_index=True), households


@pytest.fixture(scope="module")
def panel_and_truth():
    return _panel()


@pytest.fixture(scope="module")
def covariates(panel_and_truth):
    panel, _ = panel_and_truth
    pre = panel[panel["period"] == "pre"]
    return (
        pre.groupby("lcl_id")["kwh_peak"]
        .agg(pre_mean="mean", pre_std="std")
        .reset_index()
    )


def test_naive_estimate_is_badly_biased(panel_and_truth):
    panel, households = panel_and_truth
    truth = households.loc[households["is_treated"], "true_effect"].mean()
    naive = estimate_att_naive(panel)
    assert truth < 0
    assert naive.estimate > 0, "fixture should produce a sign-flipped naive estimate"


def test_did_recovers_the_true_effect(panel_and_truth):
    panel, households = panel_and_truth
    truth = households.loc[households["is_treated"], "true_effect"].mean()
    did = estimate_att_did(panel)
    low, high = did.ci95
    assert low <= truth <= high, f"true ATT {truth:.4f} outside CI [{low:.4f}, {high:.4f}]"


def test_ipw_did_also_recovers_the_true_effect(panel_and_truth, covariates):
    panel, households = panel_and_truth
    truth = households.loc[households["is_treated"], "true_effect"].mean()
    ipw, weighted = estimate_att_ipw_did(panel, covariates)
    low, high = ipw.ci95
    assert low <= truth <= high
    assert (weighted["propensity"] > 0).all() and (weighted["propensity"] < 1).all()


def test_did_is_far_closer_to_truth_than_naive(panel_and_truth):
    panel, households = panel_and_truth
    truth = households.loc[households["is_treated"], "true_effect"].mean()
    naive = abs(estimate_att_naive(panel).estimate - truth)
    did = abs(estimate_att_did(panel).estimate - truth)
    assert did < naive / 5


def test_parallel_trends_passes_when_trends_are_parallel(panel_and_truth):
    panel, _ = panel_and_truth
    verdict = pre_trend_divergence(parallel_trends(panel))
    assert verdict["verdict"] == "parallel trends plausible"


def test_parallel_trends_detects_a_planted_divergence(panel_and_truth):
    panel, _ = panel_and_truth
    drifted = panel.copy()
    pre_treated = (drifted["period"] == "pre") & drifted["is_treated"]
    months = drifted.loc[pre_treated, "date_local"].dt.month.to_numpy()
    drifted.loc[pre_treated, "kwh_peak"] += 0.05 * months

    verdict = pre_trend_divergence(parallel_trends(drifted))
    assert verdict["verdict"] == "pre trends diverge"


def test_uplift_ranking_is_monotonic(panel_and_truth, covariates):
    panel, _ = panel_and_truth
    deltas = household_deltas(panel).merge(covariates, on="lcl_id")
    learner = UpliftTLearner(feature_names=["pre_mean", "pre_std"]).fit(deltas, quick=True)
    deltas["uplift"] = learner.predict_uplift(deltas)

    deciles = uplift_by_decile(
        deltas["uplift"].to_numpy(), deltas["delta"].to_numpy(), deltas["is_treated"].to_numpy()
    )
    observed = deciles["observed_uplift_kwh"].to_numpy()
    assert observed[0] > observed[-1], "decile 1 must out-save decile 10"
    correlation = np.corrcoef(deciles["decile"], observed)[0, 1]
    assert correlation < -0.8, f"decile ordering is not monotonic (r={correlation:.3f})"


def test_qini_is_positive_when_the_model_has_signal(panel_and_truth, covariates):
    panel, _ = panel_and_truth
    deltas = household_deltas(panel).merge(covariates, on="lcl_id")
    learner = UpliftTLearner(feature_names=["pre_mean", "pre_std"]).fit(deltas, quick=True)
    curve = qini_curve(
        learner.predict_uplift(deltas), deltas["delta"].to_numpy(),
        deltas["is_treated"].to_numpy(),
    )
    assert qini_coefficient(curve) > 0


def test_qini_is_near_zero_for_a_random_ranking(panel_and_truth):
    panel, _ = panel_and_truth
    deltas = household_deltas(panel)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, len(deltas))
    curve = qini_curve(noise, deltas["delta"].to_numpy(), deltas["is_treated"].to_numpy())
    assert abs(qini_coefficient(curve)) < 1.0


def test_uplift_sign_convention_is_reduction_positive(panel_and_truth, covariates):
    panel, households = panel_and_truth
    deltas = household_deltas(panel).merge(covariates, on="lcl_id")
    learner = UpliftTLearner(feature_names=["pre_mean", "pre_std"]).fit(deltas, quick=True)
    deltas["uplift"] = learner.predict_uplift(deltas)

    merged = deltas.merge(households[["lcl_id", "true_effect"]], on="lcl_id")
    treated = merged[merged["is_treated"]]
    correlation = np.corrcoef(treated["uplift"], -treated["true_effect"])[0, 1]
    assert correlation > 0.5, "predicted uplift should rise with the true reduction"


def test_household_deltas_require_both_periods(panel_and_truth):
    panel, _ = panel_and_truth
    truncated = panel[~((panel["lcl_id"] == "T00000") & (panel["period"] == "post"))]
    deltas = household_deltas(truncated)
    assert "T00000" not in set(deltas["lcl_id"])
