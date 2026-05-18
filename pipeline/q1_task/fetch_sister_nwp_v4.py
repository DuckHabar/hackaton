"""
fetch_sister_nwp_v4.py - попытаться ещё больше source diversity.

Кандидаты:
- met_no_seamless (Norway, MetNo) - European but Nordic-tuned
- bom_access_global (Australian global; may не покрывать Azov)
- jma_seamless (Japan)
- cma_grapes_global (China)
- ecmwf_ifs04 (ECMWF integrated 0.4 deg, отличный от ifs025)

Сохраняем в data/processed/nwp_sister/<short>.parquet.
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

SOURCES_V4 = {
    "metno": "metno_seamless",
    "bom": "bom_access_global",
    "jma": "jma_seamless",
    "cma": "cma_grapes_global",
    "ifs04": "ecmwf_ifs04",
}

BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_one(short, model, retries=3):
    raw_path = RAW / f"{short}.json"
    if raw_path.exists():
        print(f"[{short}] cache hit"); return json.loads(raw_path.read_text())
    params = {
        "latitude": LAT, "longitude": LON, "start_date": START, "end_date": END,
        "hourly": ",".join(HOURLY_VARS), "models": model,
        "timezone": "UTC", "wind_speed_unit": "ms",
    }
    url = f"{BASE}?{urlencode(params)}"
    print(f"[{short}] fetch {model}")
    for i in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            j = r.json()
            raw_path.write_text(json.dumps(j))
            print(f"[{short}] OK ({len(r.content)/1024:.0f} KB)")
            return j
        except Exception as e:
            print(f"[{short}] try {i+1} fail: {e}")
            time.sleep(2 ** (i+1))
    raise RuntimeError(f"failed {short}")


def parse_save(short, j):
    df = pd.DataFrame(j["hourly"]).rename(columns={"time": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    ws_120 = df.get("wind_speed_120m")
    cov = (1 - ws_120.isna().mean()) * 100 if ws_120 is not None else 0
    print(f"[{short}] wind_120m cov: {cov:.1f}%")
    parq = PROC / f"{short}.parquet"
    df.to_parquet(parq)
    print(f"[{short}] -> {parq}")


def main():
    t0 = time.time()
    for s, m in SOURCES_V4.items():
        try:
            j = fetch_one(s, m)
            parse_save(s, j)
        except Exception as e:
            print(f"[{s}] FAILED: {e}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
