from __future__ import annotations

import logging

import pandas as pd

from metermind.config import (
    INTERVALS_PER_DAY,
    MAX_PLAUSIBLE_KWH_PER_HALFHOUR,
    PATHS,
    SLOTS_COLUMNS,
    TARIFF_TOU,
)
from metermind.warehouse.duck import connect, row_count

logger = logging.getLogger(__name__)

BANDS: dict[str, tuple[int, int]] = {
    "night": (0, 13),
    "morning": (14, 23),
    "afternoon": (24, 31),
    "peak": (32, 39),
    "late": (40, 47),
}

UK_BANK_HOLIDAYS_2012_2013 = [
    "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04",
    "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26",
    "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27",
    "2013-08-26", "2013-12-25", "2013-12-26",
]


def _band_sum_sql(name: str, lo: int, hi: int) -> str:
    cols = " + ".join(f"h{i}" for i in range(lo, hi + 1))
    return f"({cols}) AS kwh_{name}"


def _dim_date_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    holidays = set(pd.to_datetime(UK_BANK_HOLIDAYS_2012_2013))

    frame = pd.DataFrame({"date_day": days})
    frame["year"] = days.year
    frame["quarter"] = days.quarter
    frame["month"] = days.month
    frame["day_of_month"] = days.day
    frame["day_of_week"] = days.dayofweek
    frame["day_of_year"] = days.dayofyear
    frame["week_of_year"] = days.isocalendar().week.astype(int)
    frame["is_weekend"] = days.dayofweek >= 5
    frame["is_holiday"] = days.isin(holidays)
    frame["is_business_day"] = ~(frame["is_weekend"] | frame["is_holiday"])
    frame["season"] = days.month.map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )
    return frame


def build_warehouse(rebuild: bool = False) -> dict[str, int]:
    from metermind.ingest.lcl import bronze_glob

    PATHS.ensure()
    glob = bronze_glob()

    if not list((PATHS.bronze / "lcl_daily").rglob("*.parquet")):
        raise FileNotFoundError(
            "No bronze data found. Run `metermind ingest` before `metermind build`."
        )

    band_columns = ",\n            ".join(_band_sum_sql(n, lo, hi) for n, (lo, hi) in BANDS.items())
    profile_columns = ", ".join(SLOTS_COLUMNS)
    greatest_args = ", ".join(SLOTS_COLUMNS)

    with connect() as con:
        if rebuild:
            logger.info("Rebuild requested: dropping gold tables")
            for table in ("fact_household_day", "dim_household", "dim_date", "silver_household_day"):
                con.execute(f"DROP TABLE IF EXISTS {table}")

        logger.info("Building silver_household_day")
        con.execute(f"""
        CREATE OR REPLACE TABLE silver_household_day AS
        SELECT
            lcl_id,
            tariff,
            CAST(date_local AS DATE)                       AS date_local,
            n_intervals,
            kwh_total,
            {profile_columns},

            {band_columns},

            greatest({greatest_args})                      AS kwh_peak_halfhour,

            (n_intervals <> {INTERVALS_PER_DAY})           AS flag_incomplete_day,
            (n_intervals = 46 OR n_intervals = 50)         AS flag_dst_day,
            (kwh_total <= 0)                               AS flag_zero_day,
            (greatest({greatest_args}) > {MAX_PLAUSIBLE_KWH_PER_HALFHOUR})
                                                           AS flag_implausible_spike
        FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true)
        """)

        logger.info("Building dim_household")
        con.execute(f"""
        CREATE OR REPLACE TABLE dim_household AS
        SELECT
            lcl_id,
            any_value(tariff)                              AS tariff_group,
            (any_value(tariff) = '{TARIFF_TOU}')           AS is_treated,
            min(date_local)                                AS first_day,
            max(date_local)                                AS last_day,
            count(*)                                       AS n_days,
            count(*) FILTER (WHERE NOT flag_zero_day
                               AND NOT flag_incomplete_day) AS n_clean_days,
            round(avg(kwh_total), 4)                       AS mean_daily_kwh,
            round(median(kwh_total), 4)                    AS median_daily_kwh,
            round(avg(kwh_peak), 4)                        AS mean_peak_kwh
        FROM silver_household_day
        GROUP BY lcl_id
        """)

        span = con.execute(
            "SELECT min(date_local), max(date_local) FROM silver_household_day"
        ).fetchone()
        logger.info("Building dim_date for %s -> %s", span[0], span[1])
        con.register("_dim_date", _dim_date_frame(pd.Timestamp(span[0]), pd.Timestamp(span[1])))
        con.execute(
            "CREATE OR REPLACE TABLE dim_date AS "
            "SELECT CAST(date_day AS DATE) AS date_day, * EXCLUDE (date_day) FROM _dim_date"
        )
        con.unregister("_dim_date")

        logger.info("Building fact_household_day")
        con.execute("""
        CREATE OR REPLACE TABLE fact_household_day AS
        SELECT
            s.lcl_id,
            s.date_local,
            h.tariff_group,
            h.is_treated,
            s.n_intervals,
            s.kwh_total,
            s.kwh_night, s.kwh_morning, s.kwh_afternoon, s.kwh_peak, s.kwh_late,
            s.kwh_peak_halfhour,
            s.* EXCLUDE (lcl_id, tariff, date_local, n_intervals, kwh_total,
                         kwh_night, kwh_morning, kwh_afternoon, kwh_peak, kwh_late,
                         kwh_peak_halfhour,
                         flag_incomplete_day, flag_dst_day, flag_zero_day,
                         flag_implausible_spike),

            round(s.kwh_total / nullif(s.n_intervals, 0) / nullif(s.kwh_peak_halfhour, 0), 4)
                                                            AS load_factor,
            round(100.0 * s.kwh_peak / nullif(s.kwh_total, 0), 2) AS peak_share_pct,

            s.flag_incomplete_day,
            s.flag_dst_day,
            s.flag_zero_day,
            s.flag_implausible_spike,
            (NOT s.flag_incomplete_day
             AND NOT s.flag_zero_day
             AND NOT s.flag_implausible_spike)              AS is_modelling_day,

            d.day_of_week, d.is_weekend, d.is_holiday, d.is_business_day,
            d.season, d.month, d.year, d.day_of_year, d.week_of_year
        FROM silver_household_day s
        JOIN dim_household h USING (lcl_id)
        LEFT JOIN dim_date d ON d.date_day = s.date_local
        ORDER BY s.lcl_id, s.date_local
        """)

        counts = {
            t: row_count(con, t)
            for t in ("silver_household_day", "dim_household", "dim_date", "fact_household_day")
        }

    for table, n in counts.items():
        logger.info("  %-24s %10s rows", table, f"{n:,}")
    return counts
