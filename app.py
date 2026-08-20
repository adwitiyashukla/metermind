from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from metermind.warehouse.duck import connect

st.set_page_config(
    page_title="MeterMind | Residential Smart Meter Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
GRID = "rgba(195,194,183,0.14)"

S1 = "#3987e5"
S2 = "#d95926"
S3 = "#199e70"
S4 = "#c98500"
S5 = "#d55181"
S6 = "#008300"
SERIES = [S1, S2, S3, S4, S5, S6]

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
      h1, h2, h3 {{ letter-spacing: -0.02em; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.75rem; color: {S1}; }}
      div[data-testid="stMetricLabel"] {{ font-size: 0.74rem; text-transform: uppercase;
                                          letter-spacing: 0.07em; opacity: 0.75; }}
      .mm-hero {{ background: linear-gradient(120deg, rgba(57,135,229,0.12), rgba(25,158,112,0.10));
                  border: 1px solid rgba(57,135,229,0.28); border-radius: 12px;
                  padding: 1.3rem 1.7rem; margin-bottom: 1.3rem; }}
      .mm-hero h1 {{ margin: 0 0 0.35rem 0; font-size: 2rem; }}
      .mm-hero p {{ margin: 0; opacity: 0.82; font-size: 1rem; }}
      .mm-tag {{ display:inline-block; padding: 0.16rem 0.66rem; border-radius: 999px;
                 background: rgba(57,135,229,0.16); border: 1px solid rgba(57,135,229,0.32);
                 font-size: 0.75rem; margin-right: 0.4rem; margin-top: 0.7rem; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; }}
      .stTabs [data-baseweb="tab"] {{ padding: 0.5rem 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def database_path() -> Path:
    slim = ROOT / "data" / "gold" / "metermind_app.duckdb"
    return slim if slim.exists() else ROOT / "data" / "gold" / "metermind.duckdb"


@st.cache_data(ttl=1800, show_spinner=False)
def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    path = database_path()
    if not path.exists():
        return pd.DataFrame()
    try:
        with connect(path, read_only=True) as con:
            return con.execute(sql, list(params)).df()
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def artifact(name: str) -> dict:
    path = ROOT / "artifacts" / name
    return json.loads(path.read_text()) if path.exists() else {}


def style(figure: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    figure.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_2, size=12),
        margin=dict(l=12, r=16, t=28, b=12),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font_size=12),
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return figure


head = artifact("headline.json")

st.markdown(
    """
    <div class="mm-hero">
      <h1>MeterMind</h1>
      <p>Residential smart meter intelligence on 5,561 London households:
         calibrated bill-shock forecasting, learned load-shape personas,
         and causal demand-response targeting.</p>
      <div>
        <span class="mm-tag">Conformal prediction</span>
        <span class="mm-tag">Convolutional autoencoder</span>
        <span class="mm-tag">Difference in differences</span>
        <span class="mm-tag">Uplift modelling</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not database_path().exists():
    st.warning(
        "No warehouse found. Run `metermind all` locally and commit "
        "`data/gold/metermind_app.duckdb` plus the `artifacts/` directory."
    )
    st.stop()

with st.sidebar:
    st.header("Dataset")
    st.caption(
        "**Low Carbon London**  \n"
        "UK Power Networks smart meter trial, half-hourly readings from "
        "5,567 households, November 2011 to February 2014. "
        "This project uses the 2012 pre-period and the 2013 dynamic "
        "Time-of-Use trial year."
    )
    cohorts = run_query(
        "SELECT tariff_group, count(*) AS households FROM dim_household "
        "GROUP BY tariff_group ORDER BY tariff_group"
    )
    if not cohorts.empty:
        st.dataframe(cohorts, hide_index=True, width="stretch")
    st.divider()
    st.caption(
        "Built by **Adwitiya Shukla**  \n"
        "[GitHub repository](https://github.com/adwitiyashukla/metermind)  \n"
        "Data: UK Power Networks via London Datastore"
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Households", f"{head.get('n_households', 0):,}")
c2.metric("Household-days", f"{head.get('n_household_days', 0):,}")
before = head.get("coverage_before_conformal_pct")
after = head.get("coverage_after_conformal_pct")
c3.metric(
    "Interval coverage",
    f"{after:.1f}%" if after is not None else "-",
    delta=f"{after - before:+.1f} pp" if (after is not None and before is not None) else None,
    help="Share of actual bills falling inside the predicted interval. Target is 80 percent.",
)
c4.metric(
    "Bill forecast MAPE",
    f"{head.get('bill_mape_pct', 0):.2f}%" if head.get("bill_mape_pct") else "-",
    help="Error at day 12 of a 30 day billing cycle.",
)

tabs = st.tabs(
    ["Bill shock", "Personas", "Demand response", "Data quality", "Method"]
)


with tabs[0]:
    st.subheader("Calibrated bill-shock forecasting")
    st.caption(
        "Twelve days into a thirty day billing cycle, predict the final bill. "
        "The interval matters more than the point estimate. An interval that "
        "claims 80 percent and delivers barely half that is worse than no "
        "interval at all, because it still gets trusted."
    )

    scores = artifact("billforecast_scores.json")
    if not scores:
        st.info("No bill forecast scores yet. Run `metermind train`.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAPE", f"{scores['mape_pct']:.2f}%")
        m2.metric("Naive run-rate MAPE", f"{scores['naive_runrate_mape_pct']:.2f}%")
        m3.metric("Skill vs naive", f"{scores['skill_vs_naive_pct']:+.1f}%")
        m4.metric("Test cycles", f"{scores['n_test_cycles']:,}")

        cov_before = pd.DataFrame(scores["coverage_before_conformal"])
        cov_after = pd.DataFrame(scores["coverage_after_conformal"])
        merged = cov_before.merge(cov_after, on="group", suffixes=("_before", "_after"))

        st.markdown("**Interval coverage, before and after conformal calibration**")
        figure = go.Figure()
        figure.add_trace(go.Bar(
            x=merged["group"], y=merged["coverage_pct_before"], name="Quantile heads only",
            marker_color=S2, marker_line_color=SURFACE, marker_line_width=2,
            text=[f"{v:.1f}%" for v in merged["coverage_pct_before"]], textposition="outside",
            hovertemplate="%{x}<br>Coverage %{y:.2f}%<extra>Before</extra>",
        ))
        figure.add_trace(go.Bar(
            x=merged["group"], y=merged["coverage_pct_after"], name="After conformal calibration",
            marker_color=S3, marker_line_color=SURFACE, marker_line_width=2,
            text=[f"{v:.1f}%" for v in merged["coverage_pct_after"]], textposition="outside",
            hovertemplate="%{x}<br>Coverage %{y:.2f}%<extra>After</extra>",
        ))
        figure.add_hline(
            y=80, line_dash="dash", line_color=INK_2, opacity=0.7,
            annotation_text="80 percent target", annotation_position="top left",
            annotation_font_color=INK_2,
        )
        figure.update_layout(barmode="group", yaxis_title="Coverage (%)", xaxis_title=None)
        figure.update_yaxes(range=[0, 108])
        st.plotly_chart(style(figure, 420), width="stretch")

        st.caption(
            "Q1_light to Q4_heavy are quartiles of historical consumption. Calibration is "
            "Mondrian, meaning a separate correction per quartile, because one global "
            "correction serves light and heavy users badly at the same time."
        )

        with st.expander("Coverage table"):
            st.dataframe(
                merged[["group", "n_before", "coverage_pct_before", "coverage_pct_after",
                        "mean_width_before", "mean_width_after"]]
                .rename(columns={
                    "group": "Consumption band", "n_before": "Cycles",
                    "coverage_pct_before": "Coverage before (%)",
                    "coverage_pct_after": "Coverage after (%)",
                    "mean_width_before": "Interval width before (kWh)",
                    "mean_width_after": "Interval width after (kWh)",
                }),
                hide_index=True, width="stretch",
            )

        alerts = run_query("""
            SELECT lcl_id, cycle_start, consumption_band,
                   round(pred_kwh, 1) AS predicted_kwh,
                   round(lower_kwh, 1) AS lower_kwh,
                   round(upper_kwh, 1) AS upper_kwh,
                   round(kwh_cycle_total, 1) AS actual_kwh,
                   pred_bill_gbp, actual_bill_gbp, shock_headroom_kwh
            FROM bill_forecasts
            WHERE is_bill_shock
            ORDER BY shock_headroom_kwh DESC
            LIMIT 200
        """)
        alert_totals = run_query(
            "SELECT count(*) FILTER (WHERE is_bill_shock) AS fired, count(*) AS total "
            "FROM bill_forecasts"
        )
        if not alerts.empty and not alert_totals.empty:
            fired = int(alert_totals.iloc[0]["fired"])
            total_cycles = int(alert_totals.iloc[0]["total"])
            st.markdown(
                f"**Bill-shock alerts fired on {fired:,} of {total_cycles:,} test cycles "
                f"({100 * fired / total_cycles:.1f} percent).** "
                f"The {len(alerts):,} largest are shown below."
            )
            st.caption(
                "An alert fires when even the optimistic end of the calibrated interval "
                "exceeds the household's own historical cycle by more than 25 percent. "
                "Using the lower bound rather than the point estimate means the alert is "
                "conservative by construction."
            )
            st.dataframe(alerts, hide_index=True, width="stretch", height=320)


with tabs[1]:
    st.subheader("Load-shape personas")
    st.caption(
        "Each day is 48 half-hourly readings. Dividing by the day's own total removes "
        "how big the house is and keeps only when energy is used. A convolutional "
        "autoencoder with circular padding compresses that shape to 8 numbers, and the "
        "personas are clusters of those codes."
    )

    shapes_meta = artifact("shape_scores.json")
    personas = run_query("SELECT * FROM persona_summary ORDER BY persona_id")
    shapes_frame = run_query("SELECT * FROM persona_shapes ORDER BY persona_id")

    if personas.empty:
        st.info("No personas yet. Run `metermind train`.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Personas found", int(shapes_meta.get("n_personas", len(personas))))
        m2.metric("Profiles encoded", f"{shapes_meta.get('n_profiles', 0):,}")
        m3.metric("Reconstruction MSE", f"{shapes_meta.get('reconstruction_mse', 0):.2e}")

        slot_columns = [f"h{i}" for i in range(48)]
        hours = [i / 2 for i in range(48)]

        figure = go.Figure()
        for i, row in shapes_frame.iterrows():
            figure.add_trace(go.Scatter(
                x=hours, y=[row[c] for c in slot_columns],
                mode="lines", name=row["persona"],
                line=dict(color=SERIES[i % len(SERIES)], width=2),
                hovertemplate="%{y:.4f} of daily total at %{x:.1f}h<extra>"
                              + str(row["persona"]) + "</extra>",
            ))
        figure.update_layout(
            xaxis_title="Hour of day (local)", yaxis_title="Share of daily consumption",
        )
        figure.update_xaxes(tickmode="array", tickvals=[0, 4, 8, 12, 16, 20, 24])
        st.plotly_chart(style(figure, 430), width="stretch")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Persona composition**")
            st.dataframe(
                personas[["persona_id", "persona", "n_days", "share_pct",
                          "peak_hour_local", "night_share", "evening_share"]]
                .rename(columns={
                    "persona_id": "ID", "persona": "Persona", "n_days": "Days",
                    "share_pct": "Share (%)", "peak_hour_local": "Peak hour",
                    "night_share": "Night share", "evening_share": "Evening share",
                }),
                hide_index=True, width="stretch",
            )
        with right:
            st.markdown("**Cluster count chosen by silhouette score**")
            sil = shapes_meta.get("silhouette_scores", {})
            if sil:
                ks = [int(k) for k in sil]
                figure = go.Figure(go.Bar(
                    x=ks, y=[sil[str(k)] for k in ks],
                    marker_color=[S3 if k == shapes_meta.get("n_personas") else S1 for k in ks],
                    marker_line_color=SURFACE, marker_line_width=2,
                    text=[f"{sil[str(k)]:.3f}" for k in ks], textposition="outside",
                    hovertemplate="k=%{x}<br>silhouette %{y:.4f}<extra></extra>",
                ))
                figure.update_layout(xaxis_title="Number of clusters", yaxis_title="Silhouette")
                st.plotly_chart(style(figure, 300, legend=False), width="stretch")

        cross = run_query("""
            SELECT p.persona, h.tariff_group, count(*) AS households
            FROM household_persona hp
            JOIN dim_household h USING (lcl_id)
            JOIN persona_summary p ON p.persona_id = hp.persona_id
            GROUP BY p.persona, h.tariff_group
            ORDER BY p.persona, h.tariff_group
        """)
        if not cross.empty:
            st.markdown("**Persona mix by tariff cohort**")
            st.caption(
                "If the two cohorts had identical persona mixes, recruitment into the "
                "trial would be unrelated to how households actually use energy. They do "
                "not, which is the first visible sign that this is not a randomised trial."
            )
            pivot = cross.pivot(index="persona", columns="tariff_group", values="households").fillna(0)
            figure = go.Figure()
            for i, cohort in enumerate(pivot.columns):
                figure.add_trace(go.Bar(
                    x=pivot.index, y=pivot[cohort], name=str(cohort),
                    marker_color=SERIES[i], marker_line_color=SURFACE, marker_line_width=2,
                    hovertemplate="%{x}<br>%{y} households<extra>" + str(cohort) + "</extra>",
                ))
            figure.update_layout(barmode="group", yaxis_title="Households", xaxis_title=None)
            st.plotly_chart(style(figure, 340), width="stretch")


with tabs[2]:
    st.subheader("Did the tariff work, and who should be targeted next")

    causal = artifact("uplift_scores.json")
    effects = run_query("SELECT * FROM treatment_effects")

    if effects.empty:
        st.info("No causal results yet. Run `metermind train`.")
    else:
        st.caption(
            "The Low Carbon London dynamic Time-of-Use trial was not randomised. "
            "Households were recruited, so they differ from the flat-rate cohort before "
            "the trial begins. Three estimates are shown in increasing order of "
            "trustworthiness, and the distance between them is the point."
        )

        order = effects.sort_values("att_kwh", ascending=True).reset_index(drop=True)
        colours = [S2 if "naive" in m else (S1 if "IPW" in m else S3) for m in order["method"]]
        figure = go.Figure(go.Bar(
            x=order["att_kwh"], y=order["method"], orientation="h",
            marker_color=colours, marker_line_color=SURFACE, marker_line_width=2,
            error_x=dict(
                type="data", symmetric=False,
                array=(order["ci_high"] - order["att_kwh"]),
                arrayminus=(order["att_kwh"] - order["ci_low"]),
                color=INK_2, thickness=1.5, width=6,
            ),
            text=[f"{v:+.4f}" for v in order["att_kwh"]], textposition="outside",
            hovertemplate="%{y}<br>ATT %{x:.4f} kWh<extra></extra>",
        ))
        figure.add_vline(x=0, line_color=INK_2, opacity=0.6)
        figure.update_layout(
            xaxis_title="Effect on evening peak consumption (kWh per household-day)",
            yaxis_title=None,
        )
        st.plotly_chart(style(figure, 320, legend=False), width="stretch")

        verdict = (causal.get("parallel_trends") or {}).get("verdict", "not evaluated")
        st.markdown(f"**Parallel trends check: {verdict}**")
        st.caption(
            "Difference in differences is only valid if the two cohorts would have moved "
            "together without the tariff. The pre-period is where that is testable."
        )

        trends = run_query("SELECT * FROM parallel_trends ORDER BY bucket")
        if not trends.empty:
            figure = go.Figure()
            for i, cohort in enumerate(sorted(trends["cohort"].unique())):
                subset = trends[trends["cohort"] == cohort]
                figure.add_trace(go.Scatter(
                    x=subset["bucket"], y=subset["mean_kwh_peak"], mode="lines+markers",
                    name=cohort, line=dict(color=SERIES[i], width=2), marker=dict(size=6),
                    hovertemplate="%{x|%b %Y}<br>%{y:.3f} kWh<extra>" + cohort + "</extra>",
                ))
            figure.add_vline(
                x=pd.Timestamp("2013-01-01").timestamp() * 1000,
                line_dash="dash", line_color=INK_2, opacity=0.7,
                annotation_text="Trial starts", annotation_position="top right",
                annotation_font_color=INK_2,
            )
            figure.update_layout(
                yaxis_title="Mean evening peak (kWh per household-day)", xaxis_title=None
            )
            st.plotly_chart(style(figure, 360), width="stretch")

        left, right = st.columns(2)

        with left:
            st.markdown("**Qini curve**")
            st.caption(
                "Households ranked by predicted response. A model with no targeting skill "
                "traces the straight line."
            )
            curve = run_query("SELECT * FROM qini_curve ORDER BY depth")
            if not curve.empty:
                figure = go.Figure()
                figure.add_trace(go.Scatter(
                    x=curve["depth_pct"], y=curve["cumulative_uplift_kwh"],
                    mode="lines", name="Uplift model", line=dict(color=S1, width=2),
                    hovertemplate="Top %{x:.0f}%<br>%{y:.2f} kWh<extra>Model</extra>",
                ))
                figure.add_trace(go.Scatter(
                    x=curve["depth_pct"], y=curve["random_baseline_kwh"],
                    mode="lines", name="Random targeting",
                    line=dict(color=INK_2, width=2, dash="dash"),
                    hovertemplate="Top %{x:.0f}%<br>%{y:.2f} kWh<extra>Random</extra>",
                ))
                figure.update_layout(
                    xaxis_title="Share of households targeted (%)",
                    yaxis_title="Cumulative peak reduction (kWh)",
                )
                st.plotly_chart(style(figure, 340), width="stretch")
                if causal.get("qini_coefficient") is not None:
                    st.metric("Qini coefficient", f"{causal['qini_coefficient']:.4f}")

        with right:
            st.markdown("**Observed effect by predicted uplift decile**")
            st.caption(
                "The honest test of a targeting model. If the ranking is real, the "
                "observed reduction declines from decile 1 to decile 10."
            )
            deciles = run_query("SELECT * FROM uplift_deciles ORDER BY decile")
            if not deciles.empty:
                figure = go.Figure()
                figure.add_trace(go.Bar(
                    x=deciles["decile"], y=deciles["observed_uplift_kwh"],
                    name="Observed", marker_color=S3,
                    marker_line_color=SURFACE, marker_line_width=2,
                    hovertemplate="Decile %{x}<br>%{y:.4f} kWh<extra>Observed</extra>",
                ))
                figure.add_trace(go.Scatter(
                    x=deciles["decile"], y=deciles["predicted_uplift_kwh"],
                    mode="lines+markers", name="Predicted",
                    line=dict(color=S4, width=2), marker=dict(size=8),
                    hovertemplate="Decile %{x}<br>%{y:.4f} kWh<extra>Predicted</extra>",
                ))
                figure.update_layout(
                    xaxis_title="Predicted uplift decile (1 = most responsive)",
                    yaxis_title="Peak reduction (kWh)",
                )
                figure.update_xaxes(dtick=1)
                st.plotly_chart(style(figure, 340), width="stretch")

        with st.expander("Treatment effect table"):
            st.dataframe(effects, hide_index=True, width="stretch")


with tabs[3]:
    st.subheader("Data quality")
    st.caption(
        "Smart meter data fails in ways generic null checks miss: meters that stop "
        "reporting mid-trial, daylight saving days that look like missing data, and "
        "households that move out and report flat zero for a month."
    )

    checks = run_query("""
        SELECT check_name, dimension, severity, failed_rows, total_rows,
               failure_rate_pct, threshold_pct, passed, description
        FROM dq_results
        WHERE run_at_utc = (SELECT max(run_at_utc) FROM dq_results)
        ORDER BY passed, severity, check_name
    """)
    scorecard = run_query("SELECT * FROM dq_scorecard ORDER BY dimension")

    if checks.empty:
        st.info("No quality results yet. Run `metermind quality`.")
    else:
        passed = int(checks["passed"].sum())
        total = len(checks)
        m1, m2, m3 = st.columns(3)
        m1.metric("Checks passed", f"{passed}/{total}")
        m2.metric("Pass rate", f"{100 * passed / total:.0f}%")
        m3.metric(
            "Critical failures",
            int(((~checks["passed"]) & (checks["severity"] == "critical")).sum()),
        )

        if not scorecard.empty:
            figure = go.Figure(go.Bar(
                x=scorecard["dimension"], y=scorecard["pass_pct"],
                marker_color=S3, marker_line_color=SURFACE, marker_line_width=2,
                text=[f"{v:.0f}%" for v in scorecard["pass_pct"]], textposition="outside",
                hovertemplate="%{x}<br>%{y:.0f}% passed<extra></extra>",
            ))
            figure.update_layout(yaxis_title="Pass rate (%)", xaxis_title=None)
            figure.update_yaxes(range=[0, 112])
            st.plotly_chart(style(figure, 300, legend=False), width="stretch")

        display = checks.copy()
        display["Status"] = display["passed"].map({True: "PASS", False: "FAIL"})
        st.dataframe(
            display[["Status", "check_name", "dimension", "severity", "failed_rows",
                     "total_rows", "failure_rate_pct", "threshold_pct", "description"]],
            hide_index=True, width="stretch", height=440,
        )


with tabs[4]:
    st.subheader("Method")
    st.markdown(
        """
### The data

UK Power Networks ran the Low Carbon London trial between November 2011 and
February 2014, recording half-hourly electricity consumption for 5,567 London
households. About 1,100 of them were moved onto a dynamic Time-of-Use tariff
during 2013, with high, normal and low price signals sent a day ahead. The rest
stayed on a flat rate.

That structure is what makes causal work possible: 2012 is a clean pre-period
where both groups were on the same tariff, and 2013 is the trial.

### The pipeline

    167 million half-hourly readings
              |
              v
    reduced to one row per household per day, keeping the full 48 point profile
              |
              v
    bronze Parquet, partitioned by year
              |
              v
    silver: time-of-day bands, daylight saving flags, quality flags
              |
              v
    gold star schema: dim_household, dim_date, fact_household_day
              |
        +-----+-----+-----------------+
        v           v                 v
    bill forecast  shape autoencoder  causal uplift
    + conformal    + personas         + DiD + Qini

### Three choices worth defending

**Conformal prediction, not just quantile regression.** LightGBM quantile heads
give an interval and call it 80 percent. On this data they deliver far less than
that. Split conformal calibration measures the shortfall on held-out data and
widens the interval by exactly that much, which carries a finite-sample coverage
guarantee regardless of how badly the base model is calibrated. Calibration is
run per consumption quartile, because one global correction serves light and
heavy users badly at the same time.

**Shape, not level.** Clustering raw daily profiles just recovers large houses
and small houses. Dividing each day by its own total throws away level and keeps
only timing. The autoencoder uses circular padding because midnight is not an
edge: the half hour before 00:00 is adjacent to the half hour after it.

**Difference in differences, not a group comparison.** The trial cohort was
recruited, not randomised. Comparing the two groups during 2013 measures
recruitment as much as behaviour. Differencing against each household's own
pre-period cancels anything time-invariant, and the parallel trends assumption is
tested on the pre-period rather than assumed.

### Daylight saving

Europe/London has a 46 interval day each spring and a 50 interval day each
autumn. These are real measurements, not faults. On the autumn day the repeated
hour sums two readings into one slot, which distorts the pivoted profile, so
those days are flagged and excluded from shape modelling while remaining in
consumption totals where they are perfectly valid.
"""
    )

st.divider()
st.caption(
    "MeterMind. Data: Low Carbon London, UK Power Networks, via the London Datastore. "
    "Built by Adwitiya Shukla."
)
