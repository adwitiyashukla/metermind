from __future__ import annotations

import gzip
import io
import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_CANDIDATES = ["Partitioned LCL Data.zip", "PartitionedLCLData.zip", "LCL-FullData.zip"]
OUT_DIR = os.path.join(ROOT, "data", "local")
YEARS = ("2012", "2013")
SLOTS = 48
HEADER = "lcl_id,tariff,date_local,n_intervals,kwh_total," + ",".join(
    f"h{i}" for i in range(SLOTS)
)


def find_zip() -> str:
    for name in ZIP_CANDIDATES:
        candidate = os.path.join(ROOT, name)
        if os.path.exists(candidate):
            return candidate
    for name in os.listdir(ROOT):
        if name.lower().endswith(".zip") and "lcl" in name.lower():
            return os.path.join(ROOT, name)
    raise SystemExit(f"No Low Carbon London zip found in {ROOT}")


def main() -> int:
    start = int(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"chunk_{start:03d}.csv.gz")
    tmp_path = out_path + ".partial"

    tariff: dict[str, str] = {}
    rows_seen = 0
    rows_written = 0

    with zipfile.ZipFile(find_zip()) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith(".csv"))
        total = len(members)
        chunk = members[start : start + count]
        if not chunk:
            print(f"EMPTY start={start} is beyond the {total} members available")
            return 0

        with gzip.open(tmp_path, "wt", newline="", compresslevel=5) as out:
            out.write(HEADER + "\n")
            for member in chunk:
                buckets: dict[str, list] = {}
                with archive.open(member) as raw:
                    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    stream.readline()
                    for line in stream:
                        rows_seen += 1
                        parts = line.rstrip("\n").split(",")
                        if len(parts) < 4:
                            continue
                        stamp = parts[2]
                        if stamp[:4] not in YEARS:
                            continue
                        try:
                            kwh = float(parts[3])
                            hour = int(stamp[11:13])
                            minute = int(stamp[14:16])
                        except (ValueError, IndexError):
                            continue
                        if hour > 23:
                            continue
                        lcl_id = parts[0]
                        if lcl_id not in tariff:
                            tariff[lcl_id] = parts[1]
                        key = f"{lcl_id}|{stamp[:10]}"
                        entry = buckets.get(key)
                        if entry is None:
                            entry = buckets[key] = [[0.0] * SLOTS, 0]
                        entry[0][hour * 2 + (1 if minute >= 30 else 0)] += kwh
                        entry[1] += 1

                for key in sorted(buckets):
                    lcl_id, day = key.split("|")
                    profile, intervals = buckets[key]
                    values = ",".join(f"{value:.4f}" for value in profile)
                    out.write(
                        f"{lcl_id},{tariff.get(lcl_id, '')},{day},{intervals},"
                        f"{sum(profile):.4f},{values}\n"
                    )
                    rows_written += 1
                buckets = {}

    os.replace(tmp_path, out_path)
    treated = sum(1 for value in tariff.values() if value == "ToU")
    elapsed = time.time() - started
    print(
        f"OK chunk_{start:03d} files={start}..{start + len(chunk) - 1} "
        f"seen={rows_seen:,} rows={rows_written:,} households={len(tariff):,} "
        f"ToU={treated} in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
