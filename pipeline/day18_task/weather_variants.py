import yaml
import requests
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "pipeline" / "day18_task" / "config.yaml"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


CFG = _load_config()

OPENMETEO_HOURLY = [
    "temperature_2m", "pressure_msl",
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
    "wind_gusts_10m",
    "rain", "showers", "snowfall", "cloud_cover_low", "cloud_cover",
]

ARCHIVE_HOURLY = [
    "temperature_2m", "pressure_msl",
    "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m",
    "wind_gusts_10m",
    "rain", "showers", "snowfall", "cloud_cover_low", "cloud_cover",
]


def apply_weather_org(df):
    # org-ветка: погодные колонки берём из test_dataset как есть
    return df.copy()


def fetch_openmeteo_forecast(model, target_date, cache_dir=None):
    cache_dir = Path(cache_dir) if cache_dir is not None else (ROOT / CFG["openmeteo_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"openmeteo_forecast_{target_date}_{model}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["time"])

    params = {
        "latitude": CFG["openmeteo_lat"],
        "longitude": CFG["openmeteo_lon"],
        "start_date": target_date,
        "end_date": target_date,
        "timezone": "Europe/Moscow",
        "wind_speed_unit": "ms",
        "hourly": ",".join(OPENMETEO_HOURLY),
        "models": model,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=45)
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.to_csv(cache_file, index=False)
    return df


def fetch_openmeteo_archive(target_date, cache_dir=None):
    cache_dir = Path(cache_dir) if cache_dir is not None else (ROOT / CFG["openmeteo_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"openmeteo_archive_{target_date}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["time"])

    params = {
        "latitude": CFG["openmeteo_lat"],
        "longitude": CFG["openmeteo_lon"],
        "start_date": target_date,
        "end_date": target_date,
        "timezone": "Europe/Moscow",
        "wind_speed_unit": "ms",
        "hourly": ",".join(ARCHIVE_HOURLY),
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=45)
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.to_csv(cache_file, index=False)
    return df


DIRECTION_COLS = ["wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m"]


def _circular_mean_deg(matrix):
    out = np.full(matrix.shape[0], np.nan)
    for i, row in enumerate(matrix):
        row = row[np.isfinite(row)]
        if len(row) == 0:
            continue
        rad = np.deg2rad(row)
        s = float(np.mean(np.sin(rad)))
        c = float(np.mean(np.cos(rad)))
        out[i] = float(np.rad2deg(np.arctan2(s, c)) % 360.0)
    return out


def build_consensus(model_dfs):
    all_idx = sorted(set().union(*(set(df["time"]) for df in model_dfs)))
    consensus = pd.DataFrame({"time": all_idx})
    cols = [c for c in model_dfs[0].columns if c != "time"]
    for col in cols:
        stack = np.full((len(all_idx), len(model_dfs)), np.nan)
        for j, df in enumerate(model_dfs):
            sub = df.set_index("time")[col].reindex(all_idx).to_numpy()
            stack[:, j] = sub
        if col in DIRECTION_COLS:
            consensus[col] = _circular_mean_deg(stack)
        else:
            consensus[col] = np.nanmedian(stack, axis=1)
    return consensus


def overlay_archive(forecast, archive, cutoff):
    merged = forecast.copy()
    arch_idx = archive.set_index("time")
    mask = merged["time"] <= cutoff
    for col in merged.columns:
        if col == "time":
            continue
        if col in arch_idx.columns:
            vals = arch_idx[col].reindex(merged.loc[mask, "time"]).to_numpy()
            merged.loc[mask, col] = vals
    return merged


def apply_weather_om(df, target_date=None, target_dates=None, cutoff=None, cache_dir=None):
    if target_dates is None:
        target_dates = [target_date or CFG["target_date"]]
    if cutoff is None:
        cutoff = pd.Timestamp.now(tz="Europe/Moscow").tz_localize(None) - pd.Timedelta(hours=1)

    out = df.copy()
    for tdate in target_dates:
        out = _apply_om_single_day(out, str(tdate), cutoff, cache_dir)
    return out


def _apply_om_single_day(df, target_date, cutoff, cache_dir):
    model_dfs = []
    for m in CFG["openmeteo_models"]:
        try:
            model_dfs.append(fetch_openmeteo_forecast(m, target_date, cache_dir=cache_dir))
        except Exception:
            continue
    if not model_dfs:
        raise RuntimeError(f"ни одна модель open-meteo не отдала прогноз на {target_date}")
    consensus = build_consensus(model_dfs)

    try:
        archive = fetch_openmeteo_archive(target_date, cache_dir=cache_dir)
        merged = overlay_archive(consensus, archive, cutoff)
    except Exception:
        merged = consensus

    out = df.copy()
    target_mask = pd.to_datetime(out["METEOFORECASTHOUR_OPENM_Datetime"]).dt.normalize() == pd.Timestamp(target_date)
    merged_idx = merged.set_index("time")

    weather_cols = [
        "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
        "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
        "wind_gusts_10m", "rain", "showers", "snowfall", "cloud_cover_low",
        "pressure_msl",
    ]
    # температуры подбрасываем из t_2m с упрощённым градиентом
    if "temperature_2m" in merged_idx.columns:
        temp_at_target = pd.to_datetime(out.loc[target_mask, "METEOFORECASTHOUR_OPENM_Datetime"]).map(merged_idx["temperature_2m"])
        out.loc[target_mask, "temperature_80m"] = temp_at_target.to_numpy() - 0.5
        out.loc[target_mask, "temperature_120m"] = temp_at_target.to_numpy() - 0.8

    for col in weather_cols:
        if col in merged_idx.columns:
            mapped = pd.to_datetime(out.loc[target_mask, "METEOFORECASTHOUR_OPENM_Datetime"]).map(merged_idx[col])
            out.loc[target_mask, col] = mapped.to_numpy()

    # направление возвращаем в формат /1000 чтобы совпадал с organizer
    for col in DIRECTION_COLS:
        if col in out.columns:
            mask_fin = target_mask & out[col].notna()
            if mask_fin.any():
                vals = out.loc[mask_fin, col].astype(float)
                if vals.max() > 1.5:
                    out.loc[mask_fin, col] = vals / 1000.0

    return out
