from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metermind.config import RANDOM_SEED

logger = logging.getLogger(__name__)

OUTCOME = "kwh_peak"


@dataclass
class TreatmentEffect:
    method: str
    estimate: float
    std_error: float
    n_treated: int
    n_control: int
    note: str = ""

    @property
    def ci95(self) -> tuple[float, float]:
        return (
            self.estimate - 1.96 * self.std_error,
            self.estimate + 1.96 * self.std_error,
        )

    @property
    def pct_of_baseline(self) -> float:
        return float("nan")

    def as_row(self, baseline: float | None = None) -> dict:
        lo, hi = self.ci95
        row = {
            "method": self.method,
            "att_kwh": round(self.estimate, 5),
            "ci_low": round(lo, 5),
            "ci_high": round(hi, 5),
            "std_error": round(self.std_error, 5),
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "note": self.note,
        }
        if baseline:
            row["att_pct_of_baseline"] = round(100 * self.estimate / baseline, 2)
        return row


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    diff = float(np.mean(a) - np.mean(b))
    se = float(np.sqrt(np.var(a, ddof=1) / a.size + np.var(b, ddof=1) / b.size))
    return diff, se


def estimate_att_naive(panel: pd.DataFrame, outcome: str = OUTCOME) -> TreatmentEffect:
    post = panel[panel["period"] == "post"]
    t = post.loc[post["is_treated"], outcome].to_numpy(dtype=float)
    c = post.loc[~post["is_treated"], outcome].to_numpy(dtype=float)
    diff, se = _welch(t, c)
    return TreatmentEffect(
        method="naive post-period difference",
        estimate=diff,
        std_error=se,
        n_treated=t.size,
        n_control=c.size,
        note="Confounded: cohorts were recruited, not randomised.",
    )


def household_deltas(panel: pd.DataFrame, outcome: str = OUTCOME) -> pd.DataFrame:
    wide = (
        panel.pivot_table(index=["lcl_id", "is_treated"], columns="period", values=outcome,
                          aggfunc="mean")
        .reset_index()
    )
    wide = wide.dropna(subset=["pre", "post"])
    wide["delta"] = wide["post"] - wide["pre"]
    return wide


def estimate_att_did(panel: pd.DataFrame, outcome: str = OUTCOME) -> TreatmentEffect:
    wide = household_deltas(panel, outcome)
    t = wide.loc[wide["is_treated"], "delta"].to_numpy(dtype=float)
    c = wide.loc[~wide["is_treated"], "delta"].to_numpy(dtype=float)
    diff, se = _welch(t, c)
    return TreatmentEffect(
        method="difference in differences",
        estimate=diff,
        std_error=se,
        n_treated=t.size,
        n_control=c.size,
        note="Cancels time invariant household differences. Assumes parallel trends.",
    )


def estimate_att_ipw_did(
    panel: pd.DataFrame, covariates: pd.DataFrame, outcome: str = OUTCOME, trim: float = 0.02
) -> tuple[TreatmentEffect, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    wide = household_deltas(panel, outcome).merge(covariates, on="lcl_id", how="inner")
    feature_cols = [c for c in covariates.columns if c != "lcl_id"]

    x = wide[feature_cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    treated = wide["is_treated"].to_numpy(dtype=bool)

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    model.fit(scaler.fit_transform(x), treated.astype(int))
    propensity = model.predict_proba(scaler.transform(x))[:, 1]

    keep = (propensity > trim) & (propensity < 1 - trim)
    dropped = int((~keep).sum())
    if dropped:
        logger.info("  IPW: trimmed %d household(s) outside the overlap region", dropped)

    wide = wide.loc[keep].copy()
    propensity = propensity[keep]
    treated = treated[keep]
    delta = wide["delta"].to_numpy(dtype=float)

    weights = np.where(treated, 1.0, propensity / (1.0 - propensity))

    wt = weights[treated]
    wc = weights[~treated]
    yt = delta[treated]
    yc = delta[~treated]

    mean_t = float(np.average(yt, weights=wt))
    mean_c = float(np.average(yc, weights=wc))
    estimate = mean_t - mean_c

    def _wse(y, w):
        m = np.average(y, weights=w)
        var = np.average((y - m) ** 2, weights=w)
        ess = w.sum() ** 2 / np.sum(w**2)
        return np.sqrt(var / ess)

    se = float(np.sqrt(_wse(yt, wt) ** 2 + _wse(yc, wc) ** 2))

    wide["propensity"] = propensity
    wide["ipw_weight"] = weights

    effect = TreatmentEffect(
        method="IPW difference in differences",
        estimate=estimate,
        std_error=se,
        n_treated=int(treated.sum()),
        n_control=int((~treated).sum()),
        note=f"Propensity from pre period behaviour; {dropped} household(s) trimmed for overlap.",
    )
    return effect, wide


def parallel_trends(
    panel: pd.DataFrame, outcome: str = OUTCOME, freq: str = "ME"
) -> pd.DataFrame:
    work = panel.copy()
    period_freq = freq[:-1] if freq.endswith("E") and len(freq) > 1 else freq
    work["bucket"] = pd.to_datetime(work["date_local"]).dt.to_period(period_freq).dt.to_timestamp()
    grouped = (
        work.groupby(["bucket", "period", "is_treated"], observed=True)[outcome]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": f"mean_{outcome}", "count": "n"})
    )
    grouped["cohort"] = np.where(grouped["is_treated"], "ToU (treated)", "Std (control)")
    return grouped.sort_values(["bucket", "cohort"]).reset_index(drop=True)


def pre_trend_divergence(trends: pd.DataFrame, outcome: str = OUTCOME) -> dict:
    pre = trends[trends["period"] == "pre"]
    if pre.empty:
        return {"pre_gap_slope_kwh_per_month": float("nan"), "n_buckets": 0}

    pivot = pre.pivot_table(index="bucket", columns="cohort", values=f"mean_{outcome}")
    pivot = pivot.dropna()
    if len(pivot) < 3:
        return {"pre_gap_slope_kwh_per_month": float("nan"), "n_buckets": len(pivot)}

    gap = (pivot["ToU (treated)"] - pivot["Std (control)"]).to_numpy(dtype=float)
    x = np.arange(len(gap), dtype=float)
    slope, intercept = np.polyfit(x, gap, 1)
    residual = gap - (slope * x + intercept)
    se = float(np.sqrt(np.sum(residual**2) / max(len(gap) - 2, 1)) / max(np.std(x), 1e-9) / np.sqrt(len(gap)))
    return {
        "pre_gap_slope_kwh_per_month": round(float(slope), 5),
        "pre_gap_slope_se": round(se, 5),
        "pre_gap_mean_kwh": round(float(np.mean(gap)), 5),
        "n_buckets": int(len(gap)),
        "verdict": "parallel trends plausible" if abs(slope) < 2 * se else "pre trends diverge",
    }


@dataclass
class UpliftTLearner:
    feature_names: list[str]
    model_treated: object = None
    model_control: object = None

    def fit(self, frame: pd.DataFrame, target: str = "delta", quick: bool = False):
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "l2",
            "num_leaves": 15 if quick else 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": RANDOM_SEED,
            "num_threads": 2,
        }
        rounds = 120 if quick else 350

        for label, mask in (("treated", frame["is_treated"]), ("control", ~frame["is_treated"])):
            subset = frame.loc[mask]
            dataset = lgb.Dataset(
                subset[self.feature_names], subset[target].to_numpy(dtype=float)
            )
            booster = lgb.train(params, dataset, num_boost_round=rounds)
            setattr(self, f"model_{label}", booster)
            logger.info("  uplift %s model fitted on %s household(s)", label, f"{len(subset):,}")
        return self

    def predict_uplift(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_names]
        treated_delta = self.model_treated.predict(x)
        control_delta = self.model_control.predict(x)
        return np.asarray(control_delta - treated_delta, dtype=float)


def qini_curve(
    uplift: np.ndarray, outcome: np.ndarray, treated: np.ndarray, n_points: int = 100
) -> pd.DataFrame:
    uplift = np.asarray(uplift, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    treated = np.asarray(treated, dtype=bool)

    order = np.argsort(-uplift)
    outcome, treated = outcome[order], treated[order]
    n = outcome.size

    depths = np.unique(np.linspace(1, n, num=min(n_points, n)).astype(int))
    rows = []
    for k in depths:
        yt, yc = outcome[:k][treated[:k]], outcome[:k][~treated[:k]]
        if yt.size < 5 or yc.size < 5:
            continue
        incremental = (yc.mean() - yt.mean()) * k
        rows.append({
            "depth": int(k),
            "depth_pct": round(100 * k / n, 2),
            "n_treated": int(yt.size),
            "n_control": int(yc.size),
            "cumulative_uplift_kwh": round(float(incremental), 4),
        })

    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve
    total = curve["cumulative_uplift_kwh"].iloc[-1]
    curve["random_baseline_kwh"] = (curve["depth"] / n * total).round(4)
    curve["gain_over_random_kwh"] = (
        curve["cumulative_uplift_kwh"] - curve["random_baseline_kwh"]
    ).round(4)
    return curve


def qini_coefficient(curve: pd.DataFrame) -> float:
    if curve.empty or len(curve) < 2:
        return float("nan")
    x = curve["depth_pct"].to_numpy(dtype=float) / 100
    model = curve["cumulative_uplift_kwh"].to_numpy(dtype=float)
    random = curve["random_baseline_kwh"].to_numpy(dtype=float)
    area_model = float(np.trapezoid(model, x))
    area_random = float(np.trapezoid(random, x))
    if abs(area_random) < 1e-12:
        return float("nan")
    return round((area_model - area_random) / abs(area_random), 4)


def uplift_by_decile(
    uplift: np.ndarray, outcome: np.ndarray, treated: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "uplift": np.asarray(uplift, dtype=float),
        "outcome": np.asarray(outcome, dtype=float),
        "treated": np.asarray(treated, dtype=bool),
    })
    frame["decile"] = pd.qcut(
        frame["uplift"].rank(method="first", ascending=False), 10, labels=range(1, 11)
    ).astype(int)

    rows = []
    for decile, group in frame.groupby("decile", observed=True):
        yt = group.loc[group["treated"], "outcome"]
        yc = group.loc[~group["treated"], "outcome"]
        rows.append({
            "decile": int(decile),
            "n_treated": int(yt.size),
            "n_control": int(yc.size),
            "predicted_uplift_kwh": round(float(group["uplift"].mean()), 4),
            "observed_uplift_kwh": round(float(yc.mean() - yt.mean()), 4)
            if yt.size and yc.size else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
