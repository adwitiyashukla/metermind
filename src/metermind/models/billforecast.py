from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metermind.config import (
    BILLING_CYCLE_DAYS,
    CYCLE_RATIO_BOUNDS,
    FORECAST_ELAPSED_DAYS,
    MIN_CYCLE_ELAPSED_KWH,
    MIN_CYCLE_TOTAL_KWH,
    PATHS,
    RANDOM_SEED,
    TARGET_COVERAGE,
)
from metermind.models.conformal import ConformalCalibration, coverage_report

logger = logging.getLogger(__name__)

TARGET = "kwh_cycle_total"
RATIO_TARGET = "cycle_ratio"

FEATURE_COLUMNS: list[str] = [
    "elapsed_kwh", "elapsed_mean_daily", "elapsed_std_daily",
    "elapsed_max_daily", "elapsed_min_daily", "elapsed_median_daily",
    "elapsed_peak_kwh", "elapsed_night_kwh", "elapsed_peak_share",
    "elapsed_weekend_mean", "elapsed_weekday_mean", "weekend_weekday_ratio",
    "elapsed_trend_slope", "elapsed_last3_mean", "elapsed_first3_mean",
    "recent_vs_early_ratio", "days_elapsed", "days_remaining",
    "hist_mean_daily", "hist_std_daily", "hist_cycle_mean",
    "elapsed_vs_hist_ratio", "load_factor_mean",
    "month", "day_of_year", "n_weekend_days_remaining",
]


def build_cycle_features(
    daily: pd.DataFrame,
    elapsed_days: int = FORECAST_ELAPSED_DAYS,
    cycle_days: int = BILLING_CYCLE_DAYS,
) -> pd.DataFrame:
    required = {"lcl_id", "date_local", "kwh_total", "kwh_peak", "kwh_night", "is_weekend"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"build_cycle_features is missing column(s): {sorted(missing)}")

    frame = daily.sort_values(["lcl_id", "date_local"]).copy()
    frame["date_local"] = pd.to_datetime(frame["date_local"])

    rows: list[dict] = []
    for lcl_id, group in frame.groupby("lcl_id", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < cycle_days * 2:
            continue

        for start in range(0, n - cycle_days + 1, cycle_days):
            cycle = group.iloc[start : start + cycle_days]
            if len(cycle) < cycle_days:
                continue

            elapsed = cycle.iloc[:elapsed_days]
            remaining = cycle.iloc[elapsed_days:]
            e_daily = elapsed["kwh_total"].to_numpy(dtype=float)
            if not np.isfinite(e_daily).all() or e_daily.sum() < MIN_CYCLE_ELAPSED_KWH:
                continue
            cycle_total = float(cycle["kwh_total"].sum())
            if not np.isfinite(cycle_total) or cycle_total < MIN_CYCLE_TOTAL_KWH:
                continue

            history = group.iloc[:start]
            if len(history):
                hist_mean = float(history["kwh_total"].mean())
                hist_std = float(history["kwh_total"].std())
                hist_cycle_mean = hist_mean * cycle_days
            else:
                hist_mean = np.nan
                hist_std = np.nan
                hist_cycle_mean = np.nan

            weekend_mask = elapsed["is_weekend"].to_numpy(dtype=bool)
            weekend_mean = float(e_daily[weekend_mask].mean()) if weekend_mask.any() else np.nan
            weekday_mean = float(e_daily[~weekend_mask].mean()) if (~weekend_mask).any() else np.nan

            slope = float(np.polyfit(np.arange(len(e_daily), dtype=float), e_daily, 1)[0])
            last3 = float(e_daily[-3:].mean())
            first3 = float(e_daily[:3].mean())

            rows.append({
                "lcl_id": lcl_id,
                "cycle_start": cycle["date_local"].iloc[0],
                "cycle_end": cycle["date_local"].iloc[-1],
                "cutoff_date": elapsed["date_local"].iloc[-1],
                "elapsed_kwh": float(e_daily.sum()),
                "elapsed_mean_daily": float(e_daily.mean()),
                "elapsed_std_daily": float(e_daily.std()),
                "elapsed_max_daily": float(e_daily.max()),
                "elapsed_min_daily": float(e_daily.min()),
                "elapsed_median_daily": float(np.median(e_daily)),
                "elapsed_peak_kwh": float(elapsed["kwh_peak"].sum()),
                "elapsed_night_kwh": float(elapsed["kwh_night"].sum()),
                "elapsed_peak_share": float(
                    elapsed["kwh_peak"].sum() / max(e_daily.sum(), 1e-9)
                ),
                "elapsed_weekend_mean": weekend_mean,
                "elapsed_weekday_mean": weekday_mean,
                "weekend_weekday_ratio": (
                    weekend_mean / weekday_mean if weekday_mean and weekday_mean > 0 else np.nan
                ),
                "elapsed_trend_slope": slope,
                "elapsed_last3_mean": last3,
                "elapsed_first3_mean": first3,
                "recent_vs_early_ratio": last3 / first3 if first3 > 0 else np.nan,
                "days_elapsed": int(len(elapsed)),
                "days_remaining": int(len(remaining)),
                "hist_mean_daily": hist_mean,
                "hist_std_daily": hist_std,
                "hist_cycle_mean": hist_cycle_mean,
                "elapsed_vs_hist_ratio": (
                    float(e_daily.mean()) / hist_mean
                    if np.isfinite(hist_mean) and hist_mean > 0 else np.nan
                ),
                "load_factor_mean": float(elapsed.get("load_factor", pd.Series([np.nan])).mean()),
                "month": int(cycle["date_local"].iloc[0].month),
                "day_of_year": int(cycle["date_local"].iloc[0].dayofyear),
                "n_weekend_days_remaining": int(remaining["is_weekend"].sum()),
                TARGET: cycle_total,
            })
            run_rate = cycle_days / elapsed_days
            rows[-1]["run_rate_multiplier"] = run_rate
            rows[-1][RATIO_TARGET] = rows[-1][TARGET] / rows[-1]["elapsed_kwh"] - run_rate

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    logger.info(
        "Cycle features: %s cycles across %s household(s), cutoff at day %d of %d",
        f"{len(out):,}", f"{out['lcl_id'].nunique():,}", elapsed_days, cycle_days,
    )
    return out.reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, calib_frac: float = 0.2, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("cycle_start").reset_index(drop=True)
    dates = ordered["cycle_start"]
    test_cut = dates.quantile(1 - test_frac)
    calib_cut = dates.quantile(1 - test_frac - calib_frac)

    train = ordered[dates <= calib_cut]
    calib = ordered[(dates > calib_cut) & (dates <= test_cut)]
    test = ordered[dates > test_cut]

    logger.info(
        "Split  train %s (to %s) | calib %s | test %s (from %s)",
        f"{len(train):,}", calib_cut.date(), f"{len(calib):,}",
        f"{len(test):,}", test_cut.date(),
    )
    return train, calib, test


@dataclass
class BillForecaster:
    point_model: object = None
    lower_model: object = None
    upper_model: object = None
    calibration: ConformalCalibration | None = None
    feature_names: list[str] | None = None
    alpha: float = 1.0 - TARGET_COVERAGE
    blend_weight: float = 1.0

    def fit(self, train: pd.DataFrame, quick: bool = False) -> BillForecaster:
        import lightgbm as lgb

        self.feature_names = list(FEATURE_COLUMNS)
        x = train[self.feature_names]
        y = train[RATIO_TARGET].to_numpy(dtype=float)
        run_rate = train["run_rate_multiplier"].to_numpy(dtype=float)
        low, high = CYCLE_RATIO_BOUNDS
        y = np.clip(y, low - run_rate, high - run_rate)

        base = {
            "boosting_type": "gbdt",
            "num_leaves": 31 if quick else 63,
            "learning_rate": 0.06 if quick else 0.035,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "min_data_in_leaf": 40,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": RANDOM_SEED,
            "num_threads": 2,
        }
        rounds = 250 if quick else 700

        dataset = lgb.Dataset(x, y)
        self.point_model = lgb.train(
            base | {"objective": "regression", "metric": "l1"}, dataset, num_boost_round=rounds
        )
        self.lower_model = lgb.train(
            base | {"objective": "quantile", "alpha": self.alpha / 2}, dataset,
            num_boost_round=rounds,
        )
        self.upper_model = lgb.train(
            base | {"objective": "quantile", "alpha": 1 - self.alpha / 2}, dataset,
            num_boost_round=rounds,
        )
        logger.info("  fitted point + P%d/P%d heads on %s cycles",
                    round(self.alpha / 2 * 100), round((1 - self.alpha / 2) * 100), f"{len(train):,}")
        return self

    def fit_blend(self, blend: pd.DataFrame) -> BillForecaster:
        self.blend_weight = 1.0
        raw = self.predict_raw(blend)
        elapsed = blend["elapsed_kwh"].to_numpy(dtype=float)
        run_rate = blend["run_rate_multiplier"].to_numpy(dtype=float)
        truth = blend[TARGET].to_numpy(dtype=float)

        baseline = run_rate * elapsed
        correction = raw["point"].to_numpy(dtype=float) - baseline
        mask = np.isfinite(truth) & (truth > 0)

        best_weight, best_error = 0.0, np.inf
        for weight in np.linspace(0.0, 1.0, 21):
            candidate = baseline + weight * correction
            error = float(np.mean(np.abs((truth[mask] - candidate[mask]) / truth[mask])))
            if error < best_error:
                best_weight, best_error = float(weight), error

        self.blend_weight = best_weight
        logger.info(
            "  blend weight %.2f chosen on %s held-out cycles (MAPE %.3f%%)",
            best_weight, f"{int(mask.sum()):,}", best_error * 100,
        )
        return self

    def calibrate(self, calib: pd.DataFrame, groups: str | None = "consumption_band"):
        raw = self.predict_raw(calib)
        group_values = calib[groups] if groups and groups in calib.columns else None
        self.calibration = ConformalCalibration.fit(
            calib[TARGET].to_numpy(dtype=float),
            raw["lower"], raw["upper"], alpha=self.alpha, groups=group_values,
        )
        logger.info(
            "  conformal q_hat = %.3f kWh from %d calibration cycle(s)%s",
            self.calibration.q_hat, self.calibration.n_calibration,
            f", {len(self.calibration.group_q_hat)} group correction(s)"
            if self.calibration.is_grouped else "",
        )
        return self

    def predict_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.feature_names]
        elapsed = frame["elapsed_kwh"].to_numpy(dtype=float)
        run_rate = frame["run_rate_multiplier"].to_numpy(dtype=float)
        lower = self.lower_model.predict(x)
        upper = self.upper_model.predict(x)
        stacked = np.sort(np.column_stack([lower, upper]), axis=1)
        return pd.DataFrame(
            {
                "point": (run_rate + self.blend_weight * self.point_model.predict(x)) * elapsed,
                "lower": (run_rate + self.blend_weight * stacked[:, 0]) * elapsed,
                "upper": (run_rate + self.blend_weight * stacked[:, 1]) * elapsed,
            },
            index=frame.index,
        )

    def predict(self, frame: pd.DataFrame, groups: str | None = "consumption_band") -> pd.DataFrame:
        out = self.predict_raw(frame)
        if self.calibration is not None:
            group_values = frame[groups] if groups and groups in frame.columns else None
            lo, hi = self.calibration.apply(out["lower"], out["upper"], groups=group_values)
            out["lower_conformal"] = np.maximum(lo, 0.0)
            out["upper_conformal"] = hi
        return out

    def save(self, directory: Path | None = None) -> Path:
        target = Path(directory) if directory else PATHS.artifacts / "billforecast"
        target.mkdir(parents=True, exist_ok=True)
        self.point_model.save_model(str(target / "point.txt"))
        self.lower_model.save_model(str(target / "lower.txt"))
        self.upper_model.save_model(str(target / "upper.txt"))
        (target / "meta.json").write_text(json.dumps({
            "feature_names": self.feature_names,
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "blend_weight": self.blend_weight,
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }, indent=2))
        logger.info("Bill forecaster written to %s", target)
        return target

    @classmethod
    def load(cls, directory: Path | None = None) -> BillForecaster:
        import lightgbm as lgb

        target = Path(directory) if directory else PATHS.artifacts / "billforecast"
        meta = json.loads((target / "meta.json").read_text())
        return cls(
            point_model=lgb.Booster(model_file=str(target / "point.txt")),
            lower_model=lgb.Booster(model_file=str(target / "lower.txt")),
            upper_model=lgb.Booster(model_file=str(target / "upper.txt")),
            calibration=ConformalCalibration.from_dict(meta["calibration"])
            if meta.get("calibration") else None,
            feature_names=meta["feature_names"],
            alpha=meta["alpha"],
            blend_weight=meta.get("blend_weight", 1.0),
        )


def evaluate(forecaster: BillForecaster, test: pd.DataFrame) -> dict:
    truth = test[TARGET].to_numpy(dtype=float)
    pred = forecaster.predict(test)
    point = pred["point"].to_numpy(dtype=float)

    mask = np.isfinite(truth) & np.isfinite(point) & (truth > 0)
    mape = float(np.mean(np.abs((truth[mask] - point[mask]) / truth[mask])) * 100)
    mae = float(np.mean(np.abs(truth[mask] - point[mask])))
    rmse = float(np.sqrt(np.mean((truth[mask] - point[mask]) ** 2)))

    groups = test["consumption_band"] if "consumption_band" in test.columns else None
    target_pct = 100 * (1 - forecaster.alpha)
    before = coverage_report(truth, pred["lower"], pred["upper"], groups, target=target_pct)
    after = (
        coverage_report(truth, pred["lower_conformal"], pred["upper_conformal"], groups,
                        target=target_pct)
        if "lower_conformal" in pred.columns else before
    )

    naive = (
        test["elapsed_kwh"].to_numpy(dtype=float)
        / test["days_elapsed"].to_numpy(dtype=float)
        * (test["days_elapsed"] + test["days_remaining"]).to_numpy(dtype=float)
    )
    naive_mape = float(np.mean(np.abs((truth[mask] - naive[mask]) / truth[mask])) * 100)

    return {
        "n_test_cycles": int(mask.sum()),
        "mape_pct": round(mape, 4),
        "mae_kwh": round(mae, 3),
        "rmse_kwh": round(rmse, 3),
        "naive_runrate_mape_pct": round(naive_mape, 4),
        "skill_vs_naive_pct": round(100 * (naive_mape - mape) / naive_mape, 2),
        "coverage_before_conformal": before.to_dict(orient="records"),
        "coverage_after_conformal": after.to_dict(orient="records"),
    }
