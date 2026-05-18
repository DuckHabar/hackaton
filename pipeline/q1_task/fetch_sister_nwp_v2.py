"""
fetch_sister_nwp_v2.py - добавляет ещё 3 NWP источника к существующим (gfs + icon):
- arpege_europe (Meteo France ARPEGE)
- meteofrance_arome_europe (Meteo France ARÔME high-res)
- ukmo_seamless (UK Met Office)

Цель: больше diversity для EXP-019 stacking. Текущий EXP-017 (gfs + icon) дал LB 7.6914.
Гипотеза: ещё 3 NWP -> LB 7.4-7.5.

Coords: 46.83 N, 38.72 E. Time range 2022-01-01 .. 2026-03-31, hourly UTC, ws units ms.

Сохраняем в data/processed/nwp_sister/{arpege,arome,ukmo}.parquet.
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

HOURLY_VARS = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m", "temperature_120m",
    "pressure_msl",
    "rain", "showers", "snowfall",
    "cloud_cover_low",
]

# 3 новых источника
SOURCES_V2 = {
    "arpege": "arpege_europe",
    "arome": "meteofrance_arome_europe",
    "ukmo": "ukmo_seamless",
}

BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_one(src_short: str, src_model: str, retries: int = 3):
    raw_path = RAW / f"{src_short}.json"
    if raw_path.exists():
        print(f"[{src_short}] raw cache hit")
        return json.loads(raw_path.read_text())

    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": START, "end_date": END,
        "hourly": ",".join(HOURLY_VARS),
        "models": src_model,
        "timezone": "UTC", "wind_speed_unit": "ms",
    }
    url = f"{BASE}?{urlencode(params)}"
    print(f"[{src_short}] fetching {src_model} ...")
    for i in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            j = r.json()
            raw_path.write_text(json.dumps(j))
            print(f"[{src_short}] OK ({len(r.content)/1024:.0f} KB)")
            return j
        except Exception as e:
            wait = 2 ** (i + 1)
            print(f"[{src_short}] try {i+1}/{retries} failed: {e}; sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {src_short}")


def parse_and_save(src_short, j):
    hourly = j["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    nan_pct = (df.isna().mean() * 100).round(1).to_dict()
    high_nan = {k: v for k, v in nan_pct.items() if v > 5 and k != "dt"}
    if high_nan:
        print(f"[{src_short}] WARN >5% NaN: {high_nan}")
    parq = PROC / f"{src_short}.parquet"
    df.to_parquet(parq)
    print(f"[{src_short}] -> {parq} ({len(df)} rows)")
    return df


def main():
    t0 = time.time()
    for short, model in SOURCES_V2.items():
        try:
            j = fetch_one(short, model)
            parse_and_save(short, j)
        except Exception as e:
            print(f"[{short}] FAILED: {e}; skipping")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
