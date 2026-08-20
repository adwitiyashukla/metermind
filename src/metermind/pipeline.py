from __future__ import annotations

import gc
import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from metermind.config import (
    FLAT_PRICE_P_PER_KWH,
    PATHS,
    PRE_PERIOD,
    SLOTS_COLUMNS,
    TRIAL_PERIOD,
)
from metermind.models import billforecast, shapes, uplift
from metermind.warehouse.duck import connect, query

logger = logging.getLogger(__name__)


def _banner(title: str) -> None:
    logger.info("%s\n  %s\n%s", "=" * 70, title, "=" * 70)


SUMMARY_COLUMNS = """
    lcl_id, date_local, tariff_group, is_treated, n_intervals,
    kwh_total, kwh_night, kwh_morning, kwh_afternoon, kwh_peak, kwh_late,
    kwh_peak_halfhour, load_factor, peak_share_pct,
    is_weekend, is_holiday, is_business_day, season, month, year,
    day_of_week, day_of_year,
    flag_dst_day, flag_zero_day, flag_incomplete_day, is_modelling_day
"""

FLOAT32_COLUMNS = [
    "kwh_total", "kwh_night", "kwh_morning", "kwh_afternoon", "kwh_peak",
    "kwh_late", "kwh_peak_halfhour", "load_factor", "peak_share_pct",
]


def _household_filter(limit_households: int | None) -> str:
    if not limit_households:
        return ""
    return f"""
    WHERE lcl_id IN (
        SELECT lcl_id FROM dim_household
        ORDER BY n_clean_days DESC, lcl_id
        LIMIT {int(limit_households)}
    )
    """


def load_summary(limit_households: int | None = None) -> pd.DataFrame:
    frame = query(f"""
        SELECT {SUMMARY_COLUMNS}
        FROM fact_household_day
        {_household_filter(limit_households)}
        ORDER BY lcl_id, date_local
    """)
    frame["date_local"] = pd.to_datetime(frame["date_local"])
    for column in FLOAT32_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("float32")
    logger.info(
        "Loaded %s household-days across %s household(s), %s to %s",
        f"{len(frame):,}", f"{frame['lcl_id'].nunique():,}",
        frame["date_local"].min().date(), frame["date_local"].max().date(),
    )
    return frame


def load_profiles(
    limit_households: int | None = None, days_per_household: int = 90
) -> pd.DataFrame:
    slots = ", ".join(SLOTS_COLUMNS)
    where = _household_filter(limit_households).replace("WHERE", "AND", 1).strip()
    frame = query(f"""
        WITH ranked AS (
            SELECT lcl_id, date_local, tariff_group, is_treated, kwh_total,
                   is_modelling_day, flag_dst_day, {slots},
                   row_number() OVER (
                       PARTITION BY lcl_id ORDER BY hash(lcl_id || date_local::VARCHAR)
                   ) AS rn
            FROM fact_household_day
            WHERE is_modelling_day AND NOT flag_dst_day
            {where}
        )
        SELECT * EXCLUDE (rn) FROM ranked WHERE rn <= {int(days_per_household)}
    """)
    frame["date_local"] = pd.to_datetime(frame["date_local"])
    for column in SLOTS_COLUMNS + ["kwh_total"]:
        frame[column] = frame[column].astype("float32")
    logger.info(
        "Loaded %s sampled profiles across %s household(s)",
        f"{len(frame):,}", f"{frame['lcl_id'].nunique():,}",
    )
    return frame


def run_bill_forecast(daily: pd.DataFrame, quick: bool = False) -> dict:
    _banner("STAGE 1  Bill-shock forecasting with conformal intervals")

    cycles = billforecast.build_cycle_features(daily)
    if cycles.empty:
        raise RuntimeError("No billing cycles could be built. Check the warehouse.")

    band_basis = cycles["hist_mean_daily"].fillna(cycles["elapsed_mean_daily"])
    cycles["consumption_band"] = pd.qcut(
        band_basis, q=4, labels=["Q1_light", "Q2", "Q3", "Q4_heavy"], duplicates="drop",
    ).astype(str)

    train, calib, test = billforecast.chronological_split(cycles)

    midpoint = calib["cycle_start"].quantile(0.5)
    blend_split = calib[calib["cycle_start"] <= midpoint]
    conformal_split = calib[calib["cycle_start"] > midpoint]
    logger.info(
        "  calibration split: blend %s | conformal %s",
        f"{len(blend_split):,}", f"{len(conformal_split):,}",
    )

    model = billforecast.BillForecaster().fit(train, quick=quick)
    model.fit_blend(blend_split)
    model.calibrate(conformal_split, groups="consumption_band")
    model.save()

    scores = billforecast.evaluate(model, test)
    logger.info(
        "  MAPE %.3f%% vs naive run-rate %.3f%%  (skill %+.1f%%)",
        scores["mape_pct"], scores["naive_runrate_mape_pct"], scores["skill_vs_naive_pct"],
    )
    logger.info("  coverage BEFORE conformal:\n%s",
                pd.DataFrame(scores["coverage_before_conformal"]).to_string(index=False))
    logger.info("  coverage AFTER  conformal:\n%s",
                pd.DataFrame(scores["coverage_after_conformal"]).to_string(index=False))

    predictions = model.predict(test)
    out = test[["lcl_id", "cycle_start", "cycle_end", "cutoff_date", "consumption_band",
                "elapsed_kwh", "hist_cycle_mean", billforecast.TARGET]].copy()
    out["pred_kwh"] = predictions["point"].to_numpy()
    out["lower_raw"] = predictions["lower"].to_numpy()
    out["upper_raw"] = predictions["upper"].to_numpy()
    out["lower_kwh"] = predictions["lower_conformal"].to_numpy()
    out["upper_kwh"] = predictions["upper_conformal"].to_numpy()
    out["pred_bill_gbp"] = (out["pred_kwh"] * FLAT_PRICE_P_PER_KWH / 100).round(2)
    out["actual_bill_gbp"] = (out[billforecast.TARGET] * FLAT_PRICE_P_PER_KWH / 100).round(2)

    baseline = out["hist_cycle_mean"].to_numpy(dtype=float)
    out["is_bill_shock"] = out["lower_kwh"].to_numpy(dtype=float) > baseline * 1.25
    out["shock_headroom_kwh"] = (out["lower_kwh"].to_numpy(dtype=float) - baseline).round(2)

    with connect() as con:
        con.register("_bf", out)
        con.execute("CREATE OR REPLACE TABLE bill_forecasts AS SELECT * FROM _bf")
        con.unregister("_bf")

    (PATHS.artifacts / "billforecast_scores.json").write_text(json.dumps(scores, indent=2))
    return scores


def run_shape_personas(
    profiles: pd.DataFrame, quick: bool = False, reuse_model: bool = False
) -> dict:
    _banner("STAGE 2  Load-shape autoencoder and household personas")

    shape_matrix, index = shapes.normalise_profiles(profiles)
    if len(shape_matrix) < 1000:
        raise RuntimeError(f"Only {len(shape_matrix)} usable profiles; need at least 1000.")

    if reuse_model and (PATHS.artifacts / 'shape_autoencoder' / 'meta.json').exists():
        logger.info('  reusing the saved autoencoder')
        model = shapes.TrainedShapeModel.load()
    else:
        model = shapes.train_autoencoder(shape_matrix, quick=quick)
        model.save()

    codes = shapes.encode(model, shape_matrix)
    errors = shapes.reconstruction_error(model, shape_matrix)

    choice = shapes.choose_n_personas(codes)
    logger.info("  silhouette by k: %s -> chose k=%d", choice["scores"], choice["best_k"])

    labels, _, summary = shapes.assign_personas(
        codes, n_personas=choice["best_k"], shapes=shape_matrix
    )
    logger.info("  personas:\n%s",
                summary[["persona_id", "persona", "n_days", "share_pct",
                         "peak_hour_local"]].to_string(index=False))

    day_level = index[["lcl_id", "date_local", "tariff_group", "is_treated", "kwh_total"]].copy()
    day_level["persona_id"] = labels
    day_level["reconstruction_error"] = errors
    threshold = float(np.percentile(errors, 99.0))
    day_level["is_unusual_day"] = errors > threshold

    household = (
        day_level.groupby(["lcl_id", "persona_id"]).size().reset_index(name="n_days_in_persona")
        .sort_values(["lcl_id", "n_days_in_persona"], ascending=[True, False])
        .drop_duplicates("lcl_id")
        .merge(summary[["persona_id", "persona"]], on="persona_id", how="left")
    )
    embeddings = pd.DataFrame(codes, columns=[f"z{i}" for i in range(codes.shape[1])])
    embeddings["lcl_id"] = index["lcl_id"].to_numpy()
    household = household.merge(
        embeddings.groupby("lcl_id").mean().reset_index(), on="lcl_id", how="left"
    )

    mean_shapes = pd.DataFrame(
        [r["mean_shape"] for _, r in summary.iterrows()], columns=SLOTS_COLUMNS
    )
    mean_shapes.insert(0, "persona_id", summary["persona_id"].to_numpy())
    mean_shapes.insert(1, "persona", summary["persona"].to_numpy())

    with connect() as con:
        for name, frame in (
            ("persona_summary", summary.drop(columns=["mean_shape"])),
            ("persona_shapes", mean_shapes),
            ("household_persona", household),
            ("day_persona", day_level),
        ):
            con.register("_t", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
            con.unregister("_t")

    result = {
        "n_profiles": int(len(shape_matrix)),
        "latent_dim": model.latent_dim,
        "reconstruction_mse": round(model.best_loss, 8),
        "silhouette_scores": choice["scores"],
        "n_personas": choice["best_k"],
        "unusual_day_threshold": round(threshold, 8),
        "n_unusual_days": int(day_level["is_unusual_day"].sum()),
    }
    (PATHS.artifacts / "shape_scores.json").write_text(json.dumps(result, indent=2))
    return result


def run_causal_uplift(daily: pd.DataFrame, quick: bool = False) -> dict:
    _banner("STAGE 3  Causal demand-response effect and targeting")

    panel = daily[daily["is_modelling_day"]].copy()
    day = panel["date_local"].dt.date
    panel["period"] = np.select(
        [
            (day >= PRE_PERIOD[0]) & (day <= PRE_PERIOD[1]),
            (day >= TRIAL_PERIOD[0]) & (day <= TRIAL_PERIOD[1]),
        ],
        ["pre", "post"],
        default="other",
    )
    panel = panel[panel["period"] != "other"]
    logger.info("  panel:\n%s", panel.groupby(["period", "is_treated"]).size().to_string())

    trends = uplift.parallel_trends(panel)
    verdict = uplift.pre_trend_divergence(trends)
    logger.info("  parallel trends: %s", verdict)

    naive = uplift.estimate_att_naive(panel)
    did = uplift.estimate_att_did(panel)

    pre = panel[panel["period"] == "pre"]
    covariates = (
        pre.groupby("lcl_id")
        .agg(
            pre_mean_kwh=("kwh_total", "mean"),
            pre_std_kwh=("kwh_total", "std"),
            pre_peak_mean=("kwh_peak", "mean"),
            pre_peak_share=("peak_share_pct", "mean"),
            pre_night_mean=("kwh_night", "mean"),
            pre_load_factor=("load_factor", "mean"),
            pre_days=("kwh_total", "size"),
        )
        .reset_index()
    )

    ipw, weighted = uplift.estimate_att_ipw_did(panel, covariates)
    baseline_peak = float(pre.loc[pre["is_treated"], "kwh_peak"].mean())
    effects = pd.DataFrame([
        naive.as_row(baseline_peak), did.as_row(baseline_peak), ipw.as_row(baseline_peak)
    ])
    logger.info("  treatment effects:\n%s", effects.to_string(index=False))

    deltas = uplift.household_deltas(panel).merge(covariates, on="lcl_id", how="inner")
    feature_names = [c for c in covariates.columns if c != "lcl_id"]
    learner = uplift.UpliftTLearner(feature_names=feature_names).fit(deltas, quick=quick)
    deltas["predicted_uplift_kwh"] = learner.predict_uplift(deltas)

    u = deltas["predicted_uplift_kwh"].to_numpy()
    d = deltas["delta"].to_numpy()
    t = deltas["is_treated"].to_numpy()
    curve = uplift.qini_curve(u, d, t)
    qini = uplift.qini_coefficient(curve)
    deciles = uplift.uplift_by_decile(u, d, t)
    logger.info("  Qini coefficient %s", qini)
    logger.info("  uplift by decile:\n%s", deciles.to_string(index=False))

    with connect() as con:
        for name, frame in (
            ("treatment_effects", effects),
            ("parallel_trends", trends),
            ("uplift_scores", deltas),
            ("qini_curve", curve),
            ("uplift_deciles", deciles),
            ("propensity_diagnostics",
             weighted[["lcl_id", "is_treated", "propensity", "ipw_weight", "delta"]]),
        ):
            con.register("_t", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
            con.unregister("_t")

    result = {
        "baseline_peak_kwh": round(baseline_peak, 4),
        "parallel_trends": verdict,
        "effects": effects.to_dict(orient="records"),
        "qini_coefficient": qini,
        "n_households_scored": int(len(deltas)),
        "top_decile_observed_uplift_kwh": float(deciles.iloc[0]["observed_uplift_kwh"]),
        "bottom_decile_observed_uplift_kwh": float(deciles.iloc[-1]["observed_uplift_kwh"]),
    }
    (PATHS.artifacts / "uplift_scores.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def train_all(quick: bool = False, limit_households: int | None = None) -> dict:
    daily = load_summary(limit_households)
    if daily.empty:
        raise RuntimeError("Warehouse is empty. Run `metermind ingest` and `metermind build`.")

    n_households = int(daily["lcl_id"].nunique())
    n_days = int(len(daily))

    bill = run_bill_forecast(daily, quick=quick)
    causal = run_causal_uplift(daily, quick=quick)
    del daily
    gc.collect()

    profiles = load_profiles(limit_households)
    personas = run_shape_personas(profiles, quick=quick)
    del profiles
    gc.collect()

    results = {
        "bill_forecast": bill,
        "personas": personas,
        "uplift": causal,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_households": n_households,
        "n_household_days": n_days,
    }

    headline = _headline(results)
    (PATHS.artifacts / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    logger.info("HEADLINE\n%s", json.dumps(headline, indent=2, default=str))
    return results


def _headline(results: dict) -> dict:
    bf = results["bill_forecast"]
    up = results["uplift"]

    def _cov(records, group="ALL"):
        for row in records:
            if row["group"] == group:
                return row["coverage_pct"]
        return None

    did = next((e for e in up["effects"] if e["method"] == "difference in differences"), {})
    naive = next((e for e in up["effects"] if "naive" in e["method"]), {})

    return {
        "n_households": results["n_households"],
        "n_household_days": results["n_household_days"],
        "bill_mape_pct": bf["mape_pct"],
        "bill_naive_mape_pct": bf["naive_runrate_mape_pct"],
        "bill_skill_vs_naive_pct": bf["skill_vs_naive_pct"],
        "coverage_before_conformal_pct": _cov(bf["coverage_before_conformal"]),
        "coverage_after_conformal_pct": _cov(bf["coverage_after_conformal"]),
        "n_personas": results["personas"]["n_personas"],
        "att_naive_kwh": naive.get("att_kwh"),
        "att_did_kwh": did.get("att_kwh"),
        "att_did_pct": did.get("att_pct_of_baseline"),
        "qini_coefficient": up["qini_coefficient"],
        "parallel_trends_verdict": up["parallel_trends"].get("verdict"),
        "generated_at_utc": results["generated_at_utc"],
    }
