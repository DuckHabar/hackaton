"""
Fetch spatial NWP features for HEFTcom24 GEB winner trick.
4 nearby points around farm + central point, dense NWP grid via Open-Meteo.

Coords:
- Central: 46.83, 38.72 (farm)
- N: 46.93, 38.72
- S: 46.73, 38.72
- E: 46.83, 38.82
- W: 46.83, 38.62
- NE: 46.93, 38.82 (corner sample for divergence)
- ... up to 8 (compass)

Source: gfs_global (100% coverage, ws_120 OK).
Range: 2022-01-01 -> 2026-05-19 (same as sister NWP).
Save: data/processed/spatial_nwp/<point>.parquet
"""
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT = Path("/home/duck/wind_hackathon")
OUT_DIR = ROOT / "data/processed/spatial_nwp"
RAW_DIR = ROOT / "data/raw/spatial_nwp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 8 compass points at radius 0.10 deg (≈11km) + center
POINTS = {
    "C": (46.83, 38.72),
    "N": (46.93, 38.72),
    "S": (46.73, 38.72),
    "E": (46.83, 38.82),
    "W": (46.83, 38.62),
    "NE": (46.93, 38.82),
    "NW": (46.93, 38.62),
    "SE": (46.73, 38.82),
    "SW": (46.73, 38.62),
}

HOURLY_VARS = [
    "wind_speed_120m", "wind_speed_80m", "wind_gusts_10m",
    "wind_direction_120m", "temperature_120m", "pressure_msl",
]

HIST_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FCAST_BASE = "https://api.open-meteo.com/v1/forecast"

HIST_START, HIST_END = "2022-01-01", "2026-05-12"
FCAST_START, FCAST_END = "2026-05-13", "2026-05-19"


def fetch(base, lat, lon, start, end, model="gfs_global", retries=3):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "models": model,
        "timezone": "UTC", "wind_speed_unit": "ms",
    }
    url = f"{base}?{urlencode(params)}"
    for i in range(retries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  fail try {i+1}: {e}", flush=True)
            time.sleep(2 ** (i + 1))
    return None


def main():
    summary = {}
    for name, (lat, lon) in POINTS.items():
        out = OUT_DIR / f"{name}.parquet"
        if out.exists():
            df_old = pd.read_parquet(out)
            if df_old["dt"].max() >= pd.Timestamp("2026-05-19"):
                print(f"[{name}] cache OK, last={df_old['dt'].max()}, skip")
                summary[name] = {"rows": len(df_old), "cached": True}
                continue
        print(f"[{name}] lat={lat:.2f} lon={lon:.2f}, fetching ...", flush=True)
        j_h = fetch(HIST_BASE, lat, lon, HIST_START, HIST_END)
        j_f = fetch(FCAST_BASE, lat, lon, FCAST_START, FCAST_END)
        parts = []
        if j_h:
            (RAW_DIR / f"{name}_hist.json").write_text(json.dumps(j_h))
            df_h = pd.DataFrame(j_h["hourly"]).rename(columns={"time": "dt"})
            df_h["dt"] = pd.to_datetime(df_h["dt"])
            parts.append(df_h)
        if j_f:
            (RAW_DIR / f"{name}_fcast.json").write_text(json.dumps(j_f))
            df_f = pd.DataFrame(j_f["hourly"]).rename(columns={"time": "dt"})
            df_f["dt"] = pd.to_datetime(df_f["dt"])
            parts.append(df_f)
        if not parts:
            print(f"[{name}] FAILED both endpoints")
            continue
        df_all = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["dt"], keep="first").sort_values("dt").reset_index(drop=True)
        df_all.to_parquet(out)
        print(f"[{name}] saved {len(df_all)} rows", flush=True)
        summary[name] = {"rows": len(df_all), "cached": False, "max_dt": str(df_all["dt"].max())}
        time.sleep(1)

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DONE: {len(summary)} points")


if __name__ == "__main__":
    main()
