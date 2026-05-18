"""
Fetch sister NWP forecasts от Open-Meteo Historical Forecast API:
ECMWF IFS025, GFS Global, ICON EU. Координаты Berdyansk/Primorsk Азовское побережье.

Coords: 46.83 N, 38.72 E (Бердянск/Приморск)
Time range: 2022-01-01 .. 2026-03-31, hourly, UTC, wind_speed_unit=ms

Каждый источник сохраняем в:
- data/raw/nwp_sister/<src>.json  (raw response, для воспроизводимости)
- data/processed/nwp_sister/<src>.parquet  (распаршенный hourly frame)

Single coord + 4 years = 1 запрос на источник, ~10 сек.
"""
import json, sys, time
from pathlib import Path
from urllib.parse import urlencode
import requests
import pandas as pd

ROOT = Path("/home/duck/wind_hackathon")
RAW = ROOT / "data/raw/nwp_sister"
PROC = ROOT / "data/processed/nwp_sister"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

LAT, LON = 46.83, 38.72
START, END = "2022-01-01", "2026-03-31"

# 15 NWP cols matched к train_with_physics.parquet + showers (Open-Meteo даёт)
HOURLY_VARS = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m", "temperature_120m",
    "pressure_msl",
    "rain", "showers", "snowfall",
    "cloud_cover_low",
]

SOURCES = {
    "ecmwf": "ecmwf_ifs025",
    "gfs": "gfs_global",
    "icon": "icon_eu",
}

BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_one(src_short: str, src_model: str, retries: int = 3):
    raw_path = RAW / f"{src_short}.json"
    if raw_path.exists():
        print(f"[{src_short}] raw cache hit {raw_path}")
        return json.loads(raw_path.read_text())

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "hourly": ",".join(HOURLY_VARS),
        "models": src_model,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    url = f"{BASE}?{urlencode(params)}"
    print(f"[{src_short}] fetching {src_model} ...")
    for i in range(retries):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            j = r.json()
            raw_path.write_text(json.dumps(j))
            print(f"[{src_short}] OK, raw -> {raw_path} ({len(r.content)/1024:.0f} KB)")
            return j
        except Exception as e:
            wait = 2 ** (i + 1)
            print(f"[{src_short}] try {i+1}/{retries} failed: {e}; sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {src_short}/{src_model} after {retries} retries")


def parse_to_parquet(src_short: str, j: dict):
    hourly = j["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "dt"})
    df["dt"] = pd.to_datetime(df["dt"], utc=False)
    # Sanity: rows count
    expected_hours = (pd.Timestamp(END) - pd.Timestamp(START)).total_seconds() / 3600 + 24
    print(f"[{src_short}] rows={len(df)} (expected ~{int(expected_hours)})")
    # NaN summary per col
    nan_pct = (df.isna().mean() * 100).round(2).to_dict()
    high_nan = {k: v for k, v in nan_pct.items() if v > 5}
    if high_nan:
        print(f"[{src_short}] WARN cols with >5% NaN: {high_nan}")
    parq = PROC / f"{src_short}.parquet"
    df.to_parquet(parq)
    print(f"[{src_short}] -> {parq} ({parq.stat().st_size/1024:.0f} KB)")
    return df


def main():
    t0 = time.time()
    summary = {}
    for short, model in SOURCES.items():
        j = fetch_one(short, model)
        df = parse_to_parquet(short, j)
        # Compute coverage
        n_total = len(df)
        wind_120 = df.get("wind_speed_120m")
        if wind_120 is not None:
            cov_120 = (1 - wind_120.isna().mean()) * 100
        else:
            cov_120 = 0
        summary[short] = {
            "model": model,
            "rows": n_total,
            "wind_120m_coverage_pct": round(float(cov_120), 2),
            "cols": list(df.columns),
        }
    sm = PROC / "_fetch_summary.json"
    sm.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDone in {time.time()-t0:.1f}s. Summary: {sm}")


if __name__ == "__main__":
    main()
