"""
Физ-фичи для wind hackathon. Используется в physics baseline и переиспользуется
для всех последующих ML-экспериментов (EXP-002+).

Источники: IEC 61400-12-1 (air density correction), Siemens Gamesa SG 3.4-132
datasheet (rated 3.465 МВт, rotor 132 м, hub 84 м), Wind Physics Cheatsheet.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Физ-константы
R_SPECIFIC = 287.05          # уд. газовая постоянная для сухого воздуха, Дж/(кг·К)
RHO_REF = 1.225              # станд. плотность воздуха, кг/м³ (IEC 61400-12-1)
G = 9.81                     # ускорение свободного падения

# Параметры станции
HUB_HEIGHT_M = 84.0
ROTOR_DIAM_M = 132.0
N_TURBINES = 26
TURBINE_RATED_MW = 3.465     # 90.09 / 26
P_INST_MW = 90.09            # теоретический максимум

# Колонки сырых данных (для удобной интроспекции)
TARGET_COL = "Выработка. Результирующий расчет"
REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"
DT_COL = "METEOFORECASTHOUR_OPENM_Datetime"


# Базовые физ-функции (векторизованные)
def air_density(pressure_msl_hpa, temperature_c):
    """ρ = p / (R · T). pressure в гПа (Open-Meteo даёт hPa), T в °C."""
    return (pressure_msl_hpa * 100.0) / (R_SPECIFIC * (temperature_c + 273.15))


def density_correction(ws, rho, rho_ref=RHO_REF):
    """IEC 61400-12-1: ws_corr = ws · (ρ/ρ_ref)^(1/3) для stall-regulated turbines.
    Эквивалентно P = P_ref · (ρ/ρ_ref) при том же ws, что переносится в shift по ws."""
    return ws * (rho / rho_ref) ** (1.0 / 3.0)


def shear_alpha(ws_low, ws_high, h_low, h_high):
    """power-law shear: v(h) ~ h^alpha, alpha = log(v_high/v_low) / log(h_high/h_low)."""
    ws_low = np.maximum(np.asarray(ws_low, dtype=float), 0.1)
    ws_high = np.maximum(np.asarray(ws_high, dtype=float), 0.1)
    return np.log(ws_high / ws_low) / np.log(h_high / h_low)


def power_law_extrap(ws_ref, h_ref, h_target, alpha):
    """Экстраполяция ws с h_ref на h_target по shear α."""
    return ws_ref * (h_target / h_ref) ** alpha


def rews_simple(ws_80, ws_120, ws_180, hub_h=HUB_HEIGHT_M):
    """Простой rotor-equivalent wind speed: средневзвешенный по 3 уровням 80/120/180
    с весами обратно пропорциональными расстоянию до hub. при hub=84 доминирует ws_80."""
    h_levels = np.array([80, 120, 180])
    w = 1.0 / np.maximum(np.abs(h_levels - hub_h), 1.0)
    w = w / w.sum()
    return w[0] * ws_80 + w[1] * ws_120 + w[2] * ws_180


def richardson_bulk(T_low_c, T_high_c, ws_low, ws_high, h_low, h_high):
    """Bulk Richardson: Ri = (g/T̄) · (ΔT/Δz) / (Δws/Δz)². Положит. = стабильно (инверсия)."""
    T_mean_K = (T_low_c + T_high_c) / 2.0 + 273.15
    dT = T_high_c - T_low_c
    dws = np.maximum(np.asarray(ws_high - ws_low, dtype=float), 0.01)
    dh = h_high - h_low
    return (G / T_mean_K) * (dT / dh) / (dws / dh) ** 2


def breeze_index(T_80_c, hour_of_day, sst_proxy_c=None):
    """Sea-breeze index для Азовского побережья. Без SST diurnal модулируется T_80.
    Пик днём (положит.), ночью отрицат. (земной бриз)."""
    diurnal = np.sin(2 * np.pi * hour_of_day / 24.0 - np.pi / 2.0)
    base = T_80_c if sst_proxy_c is None else (T_80_c - sst_proxy_c)
    return base * diurnal


def ti_proxy(ws_gust, ws_80):
    """Прокси турбулентности: (gust - mean)/mean. Стандартный proxy IEC."""
    ws_80 = np.maximum(np.asarray(ws_80, dtype=float), 0.1)
    return (ws_gust - ws_80) / ws_80


def icing_risk(T_80_c, rain, showers, snowfall):
    """Бинарный флаг: T ∈ [-5, 1] И влажные условия (precip > 0)."""
    cold = (T_80_c >= -5) & (T_80_c <= 1)
    wet = (rain > 0) | (showers > 0) | (snowfall > 0)
    return (cold & wet).astype(np.int8)


def available_fraction(n_repair, n_total=N_TURBINES):
    """Доля рабочих турбин."""
    return (n_total - n_repair) / n_total


def wd_normalize(wd_raw):
    """В данных wind_direction_*  нормирован: реальные градусы = wd × 1000."""
    return (wd_raw * 1000.0) % 360.0


def wd_sin_cos(wd_deg):
    rad = np.deg2rad(wd_deg)
    return np.sin(rad), np.cos(rad)


# Композитная функция: добавить все физ-фичи к DataFrame
def add_physics_features(df, copy=True):
    """
    Добавить все физ-фичи к датасету. Возвращает df с новыми колонками:
      rho_air, ws_corr_80, ws_corr_120, ws_corr_180
      alpha_80_120, alpha_10_180
      rews_simple
      ri_bulk_80_120
      breeze_idx
      ti_proxy
      icing_risk
      available_frac, n_avail
      wd_*_deg, wd_*_sin, wd_*_cos для уровней 10/80/120/180
    """
    if copy:
        df = df.copy()

    # Air density
    df["rho_air"] = air_density(df["pressure_msl"], df["temperature_80m"])

    # density-corrected ws (для всех уровней, пригодится для REWS и т.п.)
    for h in (10, 80, 120, 180):
        df[f"ws_corr_{h}"] = density_correction(df[f"wind_speed_{h}m"], df["rho_air"])

    # Shear coefficients
    df["alpha_80_120"] = shear_alpha(df["wind_speed_80m"], df["wind_speed_120m"], 80, 120)
    df["alpha_10_180"] = shear_alpha(df["wind_speed_10m"], df["wind_speed_180m"], 10, 180)

    # REWS
    df["rews_simple"] = rews_simple(
        df["wind_speed_80m"], df["wind_speed_120m"], df["wind_speed_180m"]
    )
    df["rews_corr"] = density_correction(df["rews_simple"], df["rho_air"])

    # Richardson bulk
    df["ri_bulk_80_120"] = richardson_bulk(
        df["temperature_80m"], df["temperature_120m"],
        df["wind_speed_80m"], df["wind_speed_120m"], 80, 120,
    )

    # Breeze index
    df["breeze_idx"] = breeze_index(df["temperature_80m"], df["hour_of_day"])

    # TI proxy
    df["ti_proxy"] = ti_proxy(df["wind_gusts_10m"], df["wind_speed_80m"])

    # Icing risk
    df["icing_risk"] = icing_risk(
        df["temperature_80m"], df["rain"], df["showers"], df["snowfall"]
    )

    # Availability
    df["available_frac"] = available_fraction(df[REPAIR_COL])
    df["n_avail"] = N_TURBINES - df[REPAIR_COL]

    # Wind direction: денормализация + sin/cos
    for h in (10, 80, 120, 180):
        deg = wd_normalize(df[f"wind_direction_{h}m"])
        df[f"wd_{h}_deg"] = deg
        s, c = wd_sin_cos(deg)
        df[f"wd_{h}_sin"] = s
        df[f"wd_{h}_cos"] = c

    return df


# Power curve для SG 3.4-132 (своя реализация, т.к. в windpowerlib нет точной)
def sg_3_4_132_power_curve(ws_grid=None, rated_speed=10.0, cut_in=3.0, cut_out=25.0):
    """
    Аппроксимация power curve станции (per turbine, Вт).

    Эмпирически подобран rated_speed=10.0 м/с через sweep (см. notes/physics_ti_sweep.json):
    при rated=13 (datasheet SG 3.4-132) CV nMAE=18.78%; при rated=9.5 CV=11.52%.
    Это связано с aggregated-curve эффектом (пространственный шум ws_120 vs hub) и/или
    yaw misalignment / blockage. Datasheet rated_speed=13 не воспроизводит data.

    между cut_in и rated_speed кубическая аппроксимация P ~ (v - cut_in)^3,
    нормированная так чтобы P(rated_speed) = rated_w.
    """
    if ws_grid is None:
        ws_grid = np.arange(0.0, 30.05, 0.5)
    rated_w = TURBINE_RATED_MW * 1e6  # 3.465 МВт в Вт

    P = np.zeros_like(ws_grid, dtype=float)
    rising = (ws_grid >= cut_in) & (ws_grid < rated_speed)
    plateau = (ws_grid >= rated_speed) & (ws_grid <= cut_out)
    P[rising] = rated_w * ((ws_grid[rising] - cut_in) / (rated_speed - cut_in)) ** 3
    P[plateau] = rated_w
    return pd.DataFrame({"wind_speed": ws_grid, "value": P})


def smooth_power_curve_ti(ti=0.20, ws_grid=None, wind_speed_range=15.0,
                          rated_speed=10.0):
    """
    Сглаженная power curve через windpowerlib.smooth_power_curve.
    Default ti=0.20 и rated_speed=10.0 эмпирически найдены через sweep
    (notes/physics_ti_sweep.json), best CV nMAE 11.52%.

    Возвращает pd.DataFrame[wind_speed, value (Вт)].
    """
    from windpowerlib.power_curves import smooth_power_curve

    pc = sg_3_4_132_power_curve(ws_grid=ws_grid, rated_speed=rated_speed)
    smoothed = smooth_power_curve(
        power_curve_wind_speeds=pd.Series(pc["wind_speed"].values),
        power_curve_values=pd.Series(pc["value"].values),
        standard_deviation_method="turbulence_intensity",
        turbulence_intensity=ti,
        wind_speed_range=wind_speed_range,
    )
    return smoothed


def physical_power_mw(ws_array, smoothed_pc):
    """Линейная интерполяция power curve по ws. Возвращает МВт per turbine."""
    p_w = np.interp(
        np.asarray(ws_array, dtype=float),
        smoothed_pc["wind_speed"].values,
        smoothed_pc["value"].values,
        left=0.0, right=0.0,
    )
    return p_w / 1e6


def predict_p_physical_total(df, smoothed_pc, ws_col="ws_corr_120"):
    """
    Применить power curve к (density-corrected) ws и умножить на n_avail.
    Жёсткий cap по available capacity.
    """
    p_per_turb_mw = physical_power_mw(df[ws_col].values, smoothed_pc)
    p_total_mw = p_per_turb_mw * df["n_avail"].values
    cap = df["n_avail"].values * TURBINE_RATED_MW
    return np.minimum(p_total_mw, cap)


# Метрика
def nmae(y_true, y_pred, p_inst=P_INST_MW):
    """nMAE = MAE / P_inst × 100, как в hackathon spec."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)) / p_inst * 100.0)
