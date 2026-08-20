from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
ARTIFACTS = REPO_ROOT / "artifacts"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

METHOD_LABEL = {
    "naive post-period difference": "Naive group comparison",
    "difference in differences": "Difference in differences",
    "IPW difference in differences": "IPW difference in differences",
}


def load(name: str) -> dict:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def coverage_table(records: list[dict], after: list[dict]) -> str:
    by_group = {r["group"]: r for r in records}
    after_by_group = {r["group"]: r for r in after}
    lines = [
        "| Consumption band | Cycles | Coverage, quantile heads | Coverage, after conformal | Interval width (kWh) |",
        "|---|---|---|---|---|",
    ]
    for group in ["ALL", "Q1_light", "Q2", "Q3", "Q4_heavy"]:
        if group not in by_group:
            continue
        b = by_group[group]
        a = after_by_group.get(group, b)
        name = "**All cycles**" if group == "ALL" else group
        lines.append(
            f"| {name} | {b['n']:,} | {b['coverage_pct']:.2f}% | "
            f"**{a['coverage_pct']:.2f}%** | {a['mean_width']:,.1f} |"
        )
    return "\n".join(lines)


def effects_table(effects: list[dict]) -> str:
    lines = [
        "| Method | Effect on evening peak (kWh/day) | 95% CI | As % of baseline | Verdict |",
        "|---|---|---|---|---|",
    ]
    verdicts = {
        "naive post-period difference": "Confounded by recruitment",
        "difference in differences": "Cancels time-invariant differences",
        "IPW difference in differences": "Also balances observed covariates",
    }
    for row in effects:
        label = METHOD_LABEL.get(row["method"], row["method"])
        pct = row.get("att_pct_of_baseline")
        lines.append(
            f"| {label} | {row['att_kwh']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{pct:+.2f}% | {verdicts.get(row['method'], '')} |"
        )
    return "\n".join(lines)


def decile_table(deciles: list[dict]) -> str:
    lines = [
        "| Decile | Households (treated / control) | Predicted reduction (kWh) | Observed reduction (kWh) |",
        "|---|---|---|---|",
    ]
    for row in deciles:
        lines.append(
            f"| {row['decile']} | {row['n_treated']} / {row['n_control']} | "
            f"{row['predicted_uplift_kwh']:+.4f} | {row['observed_uplift_kwh']:+.4f} |"
        )
    return "\n".join(lines)


def build() -> str:
    head = load("headline.json")
    bill = load("billforecast_scores.json")
    causal = load("uplift_scores.json")
    shapes = load("shape_scores.json")

    if not head:
        raise SystemExit("No artifacts found. Run `metermind train` first.")

    parts: list[str] = []

    parts.append(
        f"Measured on **{head['n_household_days']:,} household-days** across "
        f"**{head['n_households']:,} London households**, 2012 pre-period and "
        f"2013 trial year.\n"
    )

    parts.append("### 1. Bill-shock forecasting\n")
    parts.append(
        f"Predicting the full billing cycle from the first 12 days of 30, on "
        f"{bill['n_test_cycles']:,} held-out cycles.\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| MAPE | **{bill['mape_pct']:.2f}%** |\n"
        f"| Naive run-rate baseline MAPE | {bill['naive_runrate_mape_pct']:.2f}% |\n"
        f"| Skill vs naive baseline | **{bill['skill_vs_naive_pct']:+.1f}%** |\n"
        f"| MAE | {bill['mae_kwh']:.1f} kWh |\n"
        f"| RMSE | {bill['rmse_kwh']:.1f} kWh |\n"
    )
    parts.append(
        "\nThe interval is the part that matters, and it is the part most models get "
        "wrong. Split conformal calibration is applied per consumption quartile:\n\n"
        + coverage_table(
            bill["coverage_before_conformal"], bill["coverage_after_conformal"]
        )
        + "\n"
    )

    parts.append("\n### 2. Load-shape personas\n")
    parts.append(
        f"A convolutional autoencoder with circular padding compresses each day's "
        f"48-point profile, normalised by its own total, into "
        f"{shapes['latent_dim']} numbers. Trained on {shapes['n_profiles']:,} daily "
        f"profiles, final reconstruction MSE {shapes['reconstruction_mse']:.2e}. "
        f"The cluster count was chosen by silhouette score, not by eye: "
        f"**k = {shapes['n_personas']}**.\n"
    )

    parts.append("\n### 3. Causal demand-response effect\n")
    parts.append(
        "The trial cohort was recruited, not randomised, so the three estimates below "
        "disagree by design. The distance between them is the finding.\n\n"
        + effects_table(causal["effects"])
        + "\n"
    )

    trends = causal.get("parallel_trends", {})
    if trends:
        parts.append(
            f"\nParallel trends check on the 2012 pre-period: gap slope "
            f"{trends.get('pre_gap_slope_kwh_per_month', float('nan')):+.5f} kWh per month "
            f"(SE {trends.get('pre_gap_slope_se', float('nan')):.5f}), verdict "
            f"**{trends.get('verdict', 'unknown')}**.\n"
        )

    parts.append(
        f"\n**Targeting.** Qini coefficient "
        f"{causal['qini_coefficient']:.3f} over {causal['n_households_scored']:,} "
        f"households. Observed effect by predicted decile:\n\n"
    )

    return "\n".join(parts)


def main() -> int:
    block = build()

    import duckdb

    db = REPO_ROOT / "data" / "gold" / "metermind_app.duckdb"
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        deciles = con.execute("SELECT * FROM uplift_deciles ORDER BY decile").df()
        con.close()
        block += decile_table(deciles.to_dict(orient="records")) + "\n"

    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README is missing the RESULTS markers.", file=sys.stderr)
        return 1
    before = text.split(START)[0]
    after = text.split(END)[1]
    README.write_text(f"{before}{START}\n{block}{END}{after}", encoding="utf-8")
    print("README results block updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
