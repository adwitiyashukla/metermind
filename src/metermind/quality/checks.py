from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import pandas as pd

from metermind.warehouse.duck import connect, table_exists

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Dimension(StrEnum):
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    PANEL = "panel"


@dataclass(frozen=True)
class Check:
    name: str
    dimension: Dimension
    severity: Severity
    description: str
    sql: str
    threshold: float = 0.0


@dataclass
class CheckResult:
    check: Check
    failed: int
    total: int
    error: str | None = None

    @property
    def failure_rate(self) -> float:
        return (self.failed / self.total) if self.total else 0.0

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return self.failure_rate <= self.check.threshold


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)
    run_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        return not any(
            r.check.severity is Severity.CRITICAL and not r.passed for r in self.results
        )

    @property
    def score(self) -> float:
        weights = {Severity.CRITICAL: 3.0, Severity.WARNING: 1.0, Severity.INFO: 0.5}
        total = sum(weights[r.check.severity] for r in self.results)
        earned = sum(weights[r.check.severity] for r in self.results if r.passed)
        return round(100 * earned / total, 1) if total else 100.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "run_at_utc": self.run_at,
            "check_name": r.check.name,
            "dimension": r.check.dimension.value,
            "severity": r.check.severity.value,
            "description": r.check.description,
            "failed_rows": r.failed,
            "total_rows": r.total,
            "failure_rate_pct": round(100 * r.failure_rate, 4),
            "threshold_pct": round(100 * r.check.threshold, 4),
            "passed": r.passed,
            "error": r.error,
        } for r in self.results])


CHECKS: list[Check] = [
    Check(
        name="grain_is_unique",
        dimension=Dimension.UNIQUENESS,
        severity=Severity.CRITICAL,
        description="One row per household per local day. Duplicates would double count consumption.",
        sql="""
            SELECT coalesce(sum(n - 1), 0) AS failed, count(*) AS total FROM (
                SELECT lcl_id, date_local, count(*) AS n
                FROM fact_household_day GROUP BY 1, 2
            )
        """,
    ),
    Check(
        name="consumption_non_negative",
        dimension=Dimension.VALIDITY,
        severity=Severity.CRITICAL,
        description="Domestic demand cannot be negative. A negative reading is a sign convention fault.",
        sql="""
            SELECT count(*) FILTER (WHERE kwh_total < 0) AS failed, count(*) AS total
            FROM fact_household_day
        """,
    ),
    Check(
        name="bands_reconcile_to_total",
        dimension=Dimension.CONSISTENCY,
        severity=Severity.CRITICAL,
        description=(
            "The five time-of-day bands must sum to the daily total. If they do not, "
            "the band boundaries are wrong and every peak statistic downstream is wrong."
        ),
        sql="""
            SELECT count(*) FILTER (
                WHERE abs((kwh_night + kwh_morning + kwh_afternoon + kwh_peak + kwh_late)
                          - kwh_total) > 0.01
            ) AS failed,
            count(*) AS total
            FROM fact_household_day
        """,
    ),
    Check(
        name="complete_days_have_48_intervals",
        dimension=Dimension.COMPLETENESS,
        severity=Severity.WARNING,
        description=(
            "A day should carry 48 half hours. The exceptions are the two daylight "
            "saving transitions each year, which are flagged separately and are real."
        ),
        sql="""
            SELECT count(*) FILTER (WHERE n_intervals <> 48 AND NOT flag_dst_day) AS failed,
                   count(*) AS total
            FROM fact_household_day
        """,
        threshold=0.06,
    ),
    Check(
        name="dst_days_are_recognised",
        dimension=Dimension.VALIDITY,
        severity=Severity.WARNING,
        description=(
            "Every 46 and 50 interval day must be flagged as a daylight saving day, "
            "so that shape modelling can exclude the distorted profile while consumption "
            "totals still use it."
        ),
        sql="""
            SELECT count(*) FILTER (WHERE n_intervals IN (46, 50) AND NOT flag_dst_day) AS failed,
                   count(*) FILTER (WHERE n_intervals IN (46, 50)) AS total
            FROM fact_household_day
        """,
    ),
    Check(
        name="both_cohorts_present",
        dimension=Dimension.PANEL,
        severity=Severity.CRITICAL,
        description="Both the flat rate and dynamic Time of Use cohorts must exist, or there is no comparison to make.",
        sql="""
            SELECT CASE WHEN count(DISTINCT tariff_group) < 2 THEN 1 ELSE 0 END AS failed,
                   1 AS total
            FROM fact_household_day
        """,
    ),
    Check(
        name="treated_cohort_large_enough",
        dimension=Dimension.PANEL,
        severity=Severity.CRITICAL,
        description=(
            "The treated cohort needs enough households for the causal estimate to have "
            "any power. Below a few hundred the confidence interval swamps the effect."
        ),
        sql="""
            SELECT CASE WHEN count(*) < 200 THEN 1 ELSE 0 END AS failed, 1 AS total
            FROM (SELECT DISTINCT lcl_id FROM fact_household_day WHERE is_treated)
        """,
    ),
    Check(
        name="pre_period_coverage",
        dimension=Dimension.PANEL,
        severity=Severity.CRITICAL,
        description=(
            "Difference in differences needs both cohorts observed before the trial. "
            "Without a pre period there is nothing to difference against."
        ),
        sql="""
            SELECT CASE WHEN count(*) < 2 THEN 1 ELSE 0 END AS failed, 1 AS total
            FROM (
                SELECT tariff_group FROM fact_household_day
                WHERE year = 2012 GROUP BY tariff_group HAVING count(*) > 1000
            )
        """,
    ),
    Check(
        name="referential_integrity_household",
        dimension=Dimension.CONSISTENCY,
        severity=Severity.CRITICAL,
        description="Every fact row resolves to a household in dim_household.",
        sql="""
            SELECT count(*) FILTER (WHERE h.lcl_id IS NULL) AS failed, count(*) AS total
            FROM fact_household_day f LEFT JOIN dim_household h USING (lcl_id)
        """,
    ),
    Check(
        name="referential_integrity_date",
        dimension=Dimension.CONSISTENCY,
        severity=Severity.CRITICAL,
        description="Every fact row resolves to a calendar row in dim_date.",
        sql="""
            SELECT count(*) FILTER (WHERE d.date_day IS NULL) AS failed, count(*) AS total
            FROM fact_household_day f LEFT JOIN dim_date d ON d.date_day = f.date_local
        """,
    ),
    Check(
        name="modelling_days_dominate",
        dimension=Dimension.COMPLETENESS,
        severity=Severity.WARNING,
        description=(
            "Most household-days should survive cleaning. If they do not, the cleaning "
            "rules are wrong rather than the data."
        ),
        sql="""
            SELECT count(*) FILTER (WHERE NOT is_modelling_day) AS failed, count(*) AS total
            FROM fact_household_day
        """,
        threshold=0.12,
    ),
    Check(
        name="no_implausible_halfhour_spikes",
        dimension=Dimension.VALIDITY,
        severity=Severity.WARNING,
        description=(
            "A single half hour above 10 kWh implies a sustained 20 kW draw, which is "
            "beyond a normal domestic supply. These are meter faults."
        ),
        sql="""
            SELECT count(*) FILTER (WHERE flag_implausible_spike) AS failed, count(*) AS total
            FROM fact_household_day
        """,
        threshold=0.005,
    ),
]


def run_quality_suite(persist: bool = True, database=None) -> QualityReport:
    report = QualityReport()

    with connect(database, read_only=not persist) as con:
        if not table_exists(con, "fact_household_day"):
            raise FileNotFoundError("Warehouse not built. Run `metermind build` first.")

        for check in CHECKS:
            try:
                row = con.execute(check.sql).fetchone()
                result = CheckResult(check, int(row[0] or 0), int(row[1] or 0))
            except Exception as exc:
                result = CheckResult(check, 0, 0, error=str(exc)[:300])
            report.results.append(result)

            marker = "PASS" if result.passed else (
                "FAIL" if check.severity is Severity.CRITICAL else "WARN"
            )
            logger.info(
                "  [%s] %-34s %-13s %10s/%-12s (%.3f%%)",
                marker, check.name, check.dimension.value,
                f"{result.failed:,}", f"{result.total:,}", 100 * result.failure_rate,
            )

        if persist:
            frame = report.to_frame()
            con.register("_dq", frame)
            con.execute("CREATE TABLE IF NOT EXISTS dq_results AS SELECT * FROM _dq LIMIT 0")
            con.execute("INSERT INTO dq_results SELECT * FROM _dq")
            con.unregister("_dq")

            con.execute("""
            CREATE OR REPLACE TABLE dq_scorecard AS
            WITH latest AS (SELECT max(run_at_utc) AS r FROM dq_results)
            SELECT dimension,
                   count(*) AS checks_run,
                   count(*) FILTER (WHERE passed) AS checks_passed,
                   round(100.0 * count(*) FILTER (WHERE passed) / count(*), 1) AS pass_pct,
                   max(run_at_utc) AS run_at_utc
            FROM dq_results, latest
            WHERE run_at_utc = latest.r
            GROUP BY dimension ORDER BY dimension
            """)

    passed_n = sum(r.passed for r in report.results)
    logger.info(
        "Quality score %.1f%%  (%d/%d checks passed, suite %s)",
        report.score, passed_n, len(report.results), "PASSED" if report.passed else "FAILED",
    )
    return report
