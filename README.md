# MeterMind

Residential smart meter intelligence on 5,561 London households: calibrated
bill-shock forecasting, learned load-shape personas, and causal
demand-response targeting.

[![CI](https://github.com/adwitiyashukla/metermind/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/metermind/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## The question this answers

A utility running a time-of-use tariff trial wants to know three things. What
will this customer's bill be, and how confident should we be? What kind of
energy user are they? And if we enrol them in a demand-response programme, will
they actually change anything?

The third question is causal, and it is the one that is usually answered wrong.

## Where the data comes from

UK Power Networks ran the Low Carbon London trial between November 2011 and
February 2014, recording half-hourly electricity consumption for 5,567 London
households. Roughly 1,100 were moved onto a dynamic Time-of-Use tariff during
2013, receiving high, normal and low price signals a day ahead. The rest stayed
on a flat rate.

That structure is what makes causal work possible. 2012 is a clean pre-period
where both cohorts were on the same tariff, and 2013 is the trial.

| Source | What it provides |
|---|---|
| [Low Carbon London, London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/) | 167 million half-hourly readings, tariff cohort per household |

The raw archive is 168 CSV files. `scripts/extract_local.py` reduces it to one
row per household per day while keeping the full 48-point profile, so no
measurement is discarded, only rearranged. That takes 167 million rows down to
3.2 million.

## The pipeline

```
167 million half-hourly readings
          |
          v
one row per household-day, 48-point profile preserved
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
    +-----+------------------+------------------+
    v                        v                  v
bill forecast          shape autoencoder    causal uplift
+ conformal            + personas           + DiD + Qini
```

## Three models, and why each is built the way it is

### 1. Bill-shock forecasting with calibrated intervals

Twelve days into a thirty day billing cycle, predict the final bill. Utilities
care because a bill higher than the customer expected is the largest single
driver of inbound call volume, and a warning on day twelve is worth far more
than an explanation on day thirty.

Two decisions shaped this module, and both came from results rather than taste.

**The target is a correction, not a level.** Scaling the elapsed twelve days up
to thirty is already a strong estimator. A gradient booster asked to reproduce
it spends all of its capacity on arithmetic. Predicting the residual from that
baseline means a model with nothing to add predicts zero and reproduces the
baseline exactly. This mattered: predicting the absolute total scored 22.7%
MAPE against a baseline of 8.4%, because the chronological split trains on
winter and tests on summer and the absolute target inherits that level shift.

**The learned correction is shrunk toward the baseline by a weight fitted on
held-out data.** The weight came out at 0.50. At full strength the correction
made things worse on cycles unlike anything in training; at zero it would
reproduce the baseline. Fitting the weight out of sample bounds the downside by
construction.

**The interval is the point.** LightGBM quantile heads produce an interval and
call it 80%. Here they deliver 51.8%. Split conformal calibration measures that
shortfall on data the model has not seen and widens the interval by exactly the
observed miss, which carries a finite-sample coverage guarantee regardless of
how badly the base model is calibrated. Calibration is Mondrian, meaning a
separate correction per consumption quartile, because one global correction
serves light and heavy users badly at the same time.

### 2. Load-shape personas

A household's day is 48 numbers, and most of the variance in them is simply how
big the house is. Clustering raw profiles just recovers large user and small
user. Dividing each day by its own total throws away level and keeps only
timing, which is what actually distinguishes an electric-heating household from
one that is empty until six.

A convolutional autoencoder compresses those shapes to eight numbers. Every
convolution uses circular padding, because midnight is not an edge: the half
hour before 00:00 is adjacent to the half hour after it. Zero padding would
teach the model that consumption falls off a cliff at both ends of the day,
which is an artefact of where we chose to cut the axis.

The cluster count is chosen by silhouette score rather than by eye, and
reconstruction error doubles as a per-day anomaly score at no extra cost.

### 3. Causal demand-response effect and targeting

**The Low Carbon London trial was not randomised.** Households were recruited
onto the dynamic tariff, so they differ systematically from the flat-rate group
before the trial begins. Comparing the two cohorts during 2013 measures
recruitment as much as it measures behaviour. Any write-up that calls this an
RCT is wrong.

Three estimates are therefore reported side by side, in increasing order of
trustworthiness: a naive group comparison, difference in differences, and
difference in differences reweighted by the inverse probability of recruitment
estimated from pre-period behaviour only.

The parallel trends assumption that difference in differences depends on is
tested on the pre-period rather than assumed, and the test does not pass
cleanly. That is reported below rather than buried.

For targeting, a T-learner is fitted on the pre-to-post *change* in evening peak
consumption rather than its level, so every household acts as its own control
for anything that does not vary over time. Uplift is signed so that positive
means kWh removed from the evening peak, which is the direction a utility wants.

<!-- RESULTS:START -->
Measured on **3,203,781 household-days** across **5,561 London households**, 2012 pre-period and 2013 trial year.

### 1. Bill-shock forecasting

Predicting the full billing cycle from the first 12 days of 30, on 20,567 held-out cycles.

| Metric | Value |
|---|---|
| MAPE | **8.04%** |
| Naive run-rate baseline MAPE | 8.39% |
| Skill vs naive baseline | **+4.2%** |
| MAE | 22.3 kWh |
| RMSE | 45.5 kWh |


The interval is the part that matters, and it is the part most models get wrong. Split conformal calibration is applied per consumption quartile:

| Consumption band | Cycles | Coverage, quantile heads | Coverage, after conformal | Interval width (kWh) |
|---|---|---|---|---|
| **All cycles** | 20,567 | 51.83% | **78.35%** | 58.1 |
| Q1_light | 4,943 | 53.15% | **78.98%** | 23.4 |
| Q2 | 5,169 | 51.58% | **78.29%** | 38.5 |
| Q3 | 5,239 | 51.63% | **78.64%** | 57.1 |
| Q4_heavy | 5,216 | 51.02% | **77.53%** | 111.2 |


### 2. Load-shape personas

A convolutional autoencoder with circular padding compresses each day's 48-point profile, normalised by its own total, into 8 numbers. Trained on 497,373 daily profiles, final reconstruction MSE 6.06e-05. The cluster count was chosen by silhouette score, not by eye: **k = 5**.


### 3. Causal demand-response effect

The trial cohort was recruited, not randomised, so the three estimates below disagree by design. The distance between them is the finding.

| Method | Effect on evening peak (kWh/day) | 95% CI | As % of baseline | Verdict |
|---|---|---|---|---|
| Naive group comparison | -0.1481 | [-0.1557, -0.1404] | -6.69% | Confounded by recruitment |
| Difference in differences | -0.0713 | [-0.1090, -0.0336] | -3.22% | Cancels time-invariant differences |
| IPW difference in differences | -0.0623 | [-0.0995, -0.0251] | -2.82% | Also balances observed covariates |


Parallel trends check on the 2012 pre-period: gap slope -0.02729 kWh per month (SE 0.00506), verdict **pre trends diverge**.


**Targeting.** Qini coefficient 3.877 over 5,512 households. Observed effect by predicted decile:

| Decile | Households (treated / control) | Predicted reduction (kWh) | Observed reduction (kWh) |
|---|---|---|---|
| 1 | 108 / 444 | +0.7063 | +1.1630 |
| 2 | 103 / 448 | +0.3021 | +0.4101 |
| 3 | 118 / 433 | +0.1873 | +0.3231 |
| 4 | 112 / 439 | +0.1253 | +0.1502 |
| 5 | 125 / 426 | +0.0761 | +0.0510 |
| 6 | 96 / 455 | +0.0357 | +0.0445 |
| 7 | 122 / 429 | -0.0028 | -0.0541 |
| 8 | 109 / 442 | -0.0533 | -0.0779 |
| 9 | 112 / 439 | -0.1332 | -0.2657 |
| 10 | 107 / 445 | -0.4960 | -1.0132 |
<!-- RESULTS:END -->

## What these numbers do not say

**The causal estimate rests on an assumption the data partially contradicts.**
The pre-period trends test reports that the two cohorts were already diverging
before the tariff switched on. Difference in differences is therefore not
cleanly identified here, which is exactly why the IPW variant is reported
alongside it, and why the honest reading of the effect is a range of roughly 2.8
to 3.2 percent rather than a single number.

**Conformal coverage is guaranteed under exchangeability, and a chronological
split violates it.** Calibration data comes from an earlier window than test
data, so the guarantee is approximate here. The residual gap between the
achieved coverage and the 80 percent target is the price of forecasting forward
in time, and it is visible in the table above rather than hidden by calibrating
on a random split.

**Load shapes are a continuum, not clean clusters.** The best silhouette score
found was well below what a genuinely separable structure would produce. The
personas are a useful summary, not a claim that households fall into five
natural types.

**Cycles from empty properties are excluded.** A cycle metering under 5 kWh
across twelve days is a vacancy or a dead meter, not a small household. Left in,
their near-zero denominators make percentage error meaningless. They are 0.3
percent of cycles and their exclusion is enforced at the feature-build boundary.

## Running it

Python 3.11 to 3.13. No API keys, no accounts.

```
git clone https://github.com/adwitiyashukla/metermind.git
cd metermind

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps

streamlit run app.py
```

On macOS or Linux use `source .venv/bin/activate`.

The app database and the trained models are committed, so the dashboard works on
a fresh clone with nothing configured, and `pytest` needs no data and no network.

To rebuild everything from the raw archive, download it from the London
Datastore, then:

```
python scripts/extract_local.py 0 168
metermind ingest
metermind build --rebuild
metermind quality
metermind train
metermind export
```

## What is in the repo

```
metermind/
  src/metermind/
    config.py       cohorts, trial windows, thresholds, paths
    cli.py          one entry point per pipeline stage
    ingest/         reduced export into partitioned Parquet
    warehouse/      bronze to silver to gold in DuckDB, plus the app export
    quality/        12 checks across 5 dimensions
    models/
      billforecast.py  quantile LightGBM, baseline blending, cycle features
      conformal.py     split conformal prediction and coverage diagnostics
      shapes.py        circular-padded convolutional autoencoder, personas
      uplift.py        DiD, IPW-DiD, parallel trends, T-learner, Qini
    pipeline.py     orchestration and artifact writing
  app.py            Streamlit dashboard
  scripts/          raw extraction, README results generator
  tests/            no network, no data files required
  .github/workflows/
    ci.yml          lint, tests on three Python versions, dashboard render check
    keepalive.yml   six-hourly ping so the live demo never sleeps
```

## Automation

| Workflow | When | What it does |
|---|---|---|
| `ci.yml` | every push and pull request | ruff, the test suite on Python 3.11, 3.12 and 3.13, and a real render of the dashboard |
| `keepalive.yml` | every six hours | keeps the Hugging Face Space awake |

The keepalive exists because a free Space is suspended after 48 hours without
traffic, and the next visitor then waits about a minute on a loading screen. It
polls the Hub API for `runtime.stage` rather than just requesting the page,
because a sleeping Space serves a holding page while its container boots and a
plain HTTP 200 check would pass against that holding page without the app ever
having started.

## Tests

```
pytest -q
```

The ones worth reading first:

| Test | What it pins down |
|---|---|
| `test_billforecast.py::test_features_do_not_read_past_the_cutoff` | Corrupts every reading after a wall date and asserts no feature moves |
| `test_billforecast.py::test_the_corruption_in_the_leakage_test_is_real` | Proves the test above is not vacuous |
| `test_conformal.py::test_coverage_guarantee_holds_across_many_seeds` | Coverage meets the target across 25 independent draws |
| `test_conformal.py::test_grouped_calibration_fixes_conditional_coverage` | Mondrian calibration beats a global correction on the worst-served group |
| `test_conformal.py::test_finite_sample_correction_is_applied` | The quantile level is the corrected one, not the naive one |

The leakage test earned its place. It was written before the feature builder was
finished, and it immediately caught `hist_mean_daily` averaging over a
household's entire series including future billing cycles. That bug would have
inflated every accuracy number in this project and would not have been visible in
any score.

## Licence

MIT, see [LICENSE](LICENSE).
