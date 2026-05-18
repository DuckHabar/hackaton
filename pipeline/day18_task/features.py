import numpy as np
import pandas as pd
from pipeline.q1_task import empirical_pc as q1_pc

N_TURBINES = 26
R_AIR = 287.05
RHO_REF = 1.225


def _air_density(pressure_pa, temp_kelvin):
    return pressure_pa / (R_AIR * temp_kelvin)


def _wd_to_degrees(wd_series):
    # формат направления в датасете градусы поделённые на 1000
    arr = pd.to_numeric(wd_series, errors="coerce").to_numpy(dtype=float)
    if np.nanmax(np.abs(arr)) < 1.5:
        arr = arr * 1000.0
    return np.mod(arr, 360.0)


def build_core_features(df, pc_model=None):
    out = df.copy()
    ts = pd.to_datetime(out["METEOFORECASTHOUR_OPENM_Datetime"])

    # плотность воздуха
    t120_k = out["temperature_120m"].astype(float) + 273.15
    p_pa = out["pressure_msl"].astype(float) * 100.0
    out["rho"] = _air_density(p_pa, t120_k)
    out["ws_rho_120m"] = out["wind_speed_120m"].astype(float) * (out["rho"] / RHO_REF) ** (1 / 3)

    # ветер
    out["shear_180_80"] = out["wind_speed_180m"].astype(float) - out["wind_speed_80m"].astype(float)
    out["shear_120_10"] = out["wind_speed_120m"].astype(float) - out["wind_speed_10m"].astype(float)
    out["gust_ratio"] = out["wind_gusts_10m"].astype(float) / np.maximum(out["wind_speed_10m"].astype(float), 0.1)

    # направление ветра
    for h in [10, 80, 120, 180]:
        col = f"wind_direction_{h}m"
        deg = _wd_to_degrees(out[col])
        rad = np.deg2rad(deg)
        out[f"wd_sin_{h}m"] = np.sin(rad)
        out[f"wd_cos_{h}m"] = np.cos(rad)
        out[f"wd_sector_{h}m"] = (((deg + 22.5) % 360.0) // 45.0).astype(int)

    # календарь
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_num"] = month
    out["is_weekend"] = (dow >= 5).astype(int)
    for m in [1, 2, 3, 4, 5]:
        out[f"is_m{m}"] = (month == m).astype(int)

    # ремонт
    n_rep = pd.to_numeric(out["Кол-во_ВЭУ_в_ремонте"], errors="coerce").fillna(0).astype(int)
    out["n_repair"] = n_rep
    out["n_active"] = N_TURBINES - n_rep
    out["n_active_ratio"] = out["n_active"] / N_TURBINES

    # обледенение (на мае всегда 0, оставляем для совместимости)
    out["icing_flag"] = (
        (out["temperature_80m"] >= -5) & (out["temperature_80m"] <= 2)
    ).astype(int)

    # эмпирическая кривая мощности
    if pc_model is None:
        fit_df = out.dropna(subset=["Выработка. Результирующий расчет"])
        pc_model = q1_pc.fit_empirical_pc(fit_df)
    ws_arr = out["ws_rho_120m"].to_numpy()
    rho_arr = out["rho"].to_numpy()
    pc_per_turbine = q1_pc.pc(ws_arr, rho_arr, pc_model)
    out["pc_pred"] = pc_per_turbine * out["n_active"].to_numpy()
    out["pc_residual"] = out["Выработка. Результирующий расчет"].astype(float) - out["pc_pred"]

    # бин по скорости ветра
    bin_edges = np.arange(0, 26, 1.0)
    out["bin_idx"] = np.clip(
        np.digitize(out["wind_speed_120m"].astype(float), bin_edges) - 1,
        0, len(bin_edges) - 2,
    )

    return out


def build_lag_features(df):
    out = df.sort_values("METEOFORECASTHOUR_OPENM_Datetime").reset_index(drop=True).copy()
    target = out["Выработка. Результирующий расчет"].astype(float)

    # простые лаги
    out["lag_1d_power"] = target.shift(24)
    out["lag_7d_power"] = target.shift(168)

    # rolling по прошлому: shift(1) чтобы текущая строка не учитывалась
    shifted = target.shift(1)
    out["rolling_24h_mean"] = shifted.rolling(window=24, min_periods=24).mean()
    out["rolling_24h_std"] = shifted.rolling(window=24, min_periods=24).std()

    # среднее target в этот же час по 7 предыдущим суткам
    ts = pd.to_datetime(out["METEOFORECASTHOUR_OPENM_Datetime"])
    hour = ts.dt.hour
    work = pd.DataFrame({"target": target, "hour": hour})
    work["target_shifted"] = work.groupby("hour")["target"].shift(1)
    work["same_hour_7d_mean"] = (
        work.groupby("hour")["target_shifted"]
        .rolling(window=7, min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["same_hour_7d_mean"] = work["same_hour_7d_mean"]

    return out


def build_all_features(df, pc_model=None):
    out = build_core_features(df, pc_model=pc_model)
    out = build_lag_features(out)
    return out
