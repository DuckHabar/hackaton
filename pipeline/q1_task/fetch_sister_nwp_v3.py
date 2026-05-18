"""
fetch_sister_nwp_v3.py - ещё 3 NWP source candidates:
- ecmwf_aifs025 (ECMWF AI-based forecast - diverse от physics-based)
- knmi_seamless (KNMI Netherlands MO)
- dmi_seamless (DMI Denmark MO)

Сохраняем в data/processed/nwp_sister/{aifs,knmi,dmi}.parquet.
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
    "wind_gusts_10m", "temperature_80m", "temperature_120m",
    "pressure_msl", "rain", "showers", "snowfall", "cloud_cover_low",
]

SOURCES_V3 = {
    "aifs": "ecmwf_aifs025",
    "knmi": "knmi_seamless",
    "dmi": "dmi_seamless",
}

BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_one(src_short, src_model, retries=3):
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
    raise RuntimeError(f"failed {src_short}")


def parse_and_save(src_short, j):
    df = pd.DataFrame(j["hourly"]).rename(columns={"time": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    # Coverage report
    ws_120 = df.get("wind_speed_120m")
    cov_120 = (1 - ws_120.isna().mean()) * 100 if ws_120 is not None else 0
    print(f"[{src_short}] wind_speed_120m coverage: {cov_120:.1f}%")
    parq = PROC / f"{src_short}.parquet"
    df.to_parquet(parq)
    print(f"[{src_short}] -> {parq} ({len(df)} rows)")
    return df


def main():
    t0 = time.time()
    for short, model in SOURCES_V3.items():
        try:
            j = fetch_one(short, model)
            parse_and_save(short, j)
        except Exception as e:
            print(f"[{short}] FAILED: {e}; skipping")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
