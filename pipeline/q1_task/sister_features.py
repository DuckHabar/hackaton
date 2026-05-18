"""
sister_features.py - добавляет GFS + ICON sister NWP фичи к base train/valid frames.

ECMWF source НЕ берём из data/processed/nwp_sister/ecmwf.parquet (там только 10m wind,
80m/120m/180m все NaN от Open-Meteo для модели ecmwf_ifs025). Existing prepare_features()
данные используются как наш default "ECMWF" baseline (с extrapolated heights).

GFS - wind_speed_120m coverage 100%, чистый источник diversity.
ICON - coverage 79% (gaps в 2022), median-filled.

Каждый sister источник:
  1. Физика: wd sin/cos, alpha 80->120, ws_at_84, sector_8, rho_air, p_phys
  2. Bias correction wind_speed_120m per (sector, month, src)
  3. Long lags +-18/24/36/48 на key cols
  4. Все cols суффиксированы __gfs / __icon

Возвращает frames с добавленными колонками.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "pipeline"))
sys.path.insert(0, str(_ROOT / "pipeline" / "q1_task"))
from common.physics_features import TARGET_COL

ROOT = _ROOT
SISTER_DIR = ROOT / "data/processed/nwp_sister"

SOURCES = ["gfs", "icon"]  # ECMWF пропускаем - 80m+ всё NaN

EXTRA_LAG_HOURS = [-48, -36, -24, -18, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48]
EXTRA_LAG_COLS_BASE = ["wind_speed_120m", "wind_gusts_10m", "wd_120_sin", "wd_120_cos",
                       "alpha_80_120", "ws_corrected_120"]


def add_per_source_physics(df_nwp: pd.DataFrame) -> pd.DataFrame:
    """Per-source feature engineering - wd sin/cos, alpha, ws_at_84, sector, ws_180 fallback,
    rho_air, p_phys. БЕЗ суффикса - суффикс добавим позже."""
    d = df_nwp.copy()

    # wd sin/cos + sector_8
    if "wind_direction_120m" in d.columns:
        wd120 = d["wind_direction_120m"].fillna(180.0)
        rad = np.deg2rad(wd120)
        d["wd_120_sin"] = np.sin(rad)
        d["wd_120_cos"] = np.cos(rad)
        d["sector_8"] = ((wd120 // 45) % 8).astype("Int8")

    # alpha shear (80->120)
    if "wind_speed_80m" in d.columns and "wind_speed_120m" in d.columns:
        ws80 = d["wind_speed_80m"].clip(lower=0.1)
        ws120 = d["wind_speed_120m"].clip(lower=0.1)
        d["alpha_80_120"] = np.log(ws120 / ws80) / np.log(120 / 80)

    # ws_at_84
    if "wind_speed_80m" in d.columns:
        alpha = d["alpha_80_120"].fillna(0.143) if "alpha_80_120" in d.columns else 0.143
        d["ws_at_84"] = d["wind_speed_80m"] * (84 / 80) ** alpha

    # 180m fallback через power-law (если NaN или missing)
    if "wind_speed_180m" not in d.columns or d["wind_speed_180m"].isna().all():
        if "wind_speed_80m" in d.columns:
            alpha = d["alpha_80_120"].fillna(0.143) if "alpha_80_120" in d.columns else 0.143
            d["wind_speed_180m"] = d["wind_speed_80m"] * (180 / 80) ** alpha

    # rho_air (P/RT)
    if "temperature_80m" in d.columns and "pressure_msl" in d.columns:
        T = d["temperature_80m"].fillna(10) + 273.15  # C -> K
        P = d["pressure_msl"].fillna(1013) * 100  # hPa -> Pa
        d["rho_air"] = P / (287.05 * T)
    else:
        d["rho_air"] = 1.225

    # p_phys simple cube law without power curve (для diversity сигнала)
    if "ws_at_84" in d.columns:
        ws = d["ws_at_84"].clip(lower=0).fillna(0)
        d["p_phys"] = 0.5 * d["rho_air"] * (np.pi * (132/2)**2) * (ws ** 3) * 0.4 * 26 / 1e6
        d["p_phys"] = d["p_phys"].clip(0, 90.09)

    return d


def per_source_bias_correction(train_df: pd.DataFrame, valid_df: pd.DataFrame, src: str,
                                target_col: str = TARGET_COL):
    """Per (sector, month) bias correction для wind_speed_120m__<src>.
    Вычисляет implied_ws из target через inverse cube law, считает bias и применяет."""
    ws_col = f"wind_speed_120m__{src}"
    sec_col = f"sector_8__{src}"
    if ws_col not in train_df.columns or sec_col not in train_df.columns:
        print(f"[bias_corr {src}] missing cols ({ws_col} or {sec_col}), skip")
        return train_df, valid_df

    train_df = train_df.copy(); valid_df = valid_df.copy()
    train_df["_month"] = pd.to_datetime(train_df["dt"]).dt.month
    valid_df["_month"] = pd.to_datetime(valid_df["dt"]).dt.month

    # implied ws via inverse cube law (P -> ws_eff)
    tgt = train_df[target_col].clip(lower=0.001)
    train_df["_ws_implied"] = (tgt * 1e6 / (0.5 * 1.225 * np.pi * 66**2 * 0.4 * 26)) ** (1.0/3)
    train_df["_bias"] = train_df[ws_col] - train_df["_ws_implied"]

    bias_table = train_df.groupby([sec_col, "_month"])["_bias"].mean().reset_index().rename(
        columns={"_bias": "_bias_corr"})

    for d in (train_df, valid_df):
        merged = d.merge(bias_table, on=[sec_col, "_month"], how="left")
        d[f"ws_corrected_120__{src}"] = d[ws_col] - merged["_bias_corr"].fillna(0).values
        for col in ["_month", "_ws_implied", "_bias"]:
            if col in d.columns: d.drop(columns=[col], inplace=True)

    return train_df, valid_df


def add_extra_lags_per_source(combined: pd.DataFrame, src: str) -> pd.DataFrame:
    """Add long lags +-18/24/36/48 per source. combined уже sorted by dt."""
    combined = combined.sort_values("dt").reset_index(drop=True)
    for col_base in EXTRA_LAG_COLS_BASE:
        col = f"{col_base}__{src}"
        if col not in combined.columns:
            continue
        for h in EXTRA_LAG_HOURS:
            sign = "p" if h > 0 else "m"
            name = f"{col}_xlag_{sign}{abs(h)}"
            if name not in combined.columns:
                combined[name] = combined[col].shift(-h)
    return combined


def add_sister_features(train_base: pd.DataFrame, valid_base: pd.DataFrame, sources=None):
    """Главная функция: возвращает (train_base, valid_base) с добавленными __<src>__ фичами.

    sources - список short names (default SOURCES = ['gfs', 'icon']).
    EXP-019: ['gfs', 'icon', 'arpege']."""
    if sources is None:
        sources = SOURCES
    train_full = train_base.copy()
    valid_full = valid_base.copy()
    train_full["dt"] = pd.to_datetime(train_full["dt"])
    valid_full["dt"] = pd.to_datetime(valid_full["dt"])

    for src in sources:
        p = SISTER_DIR / f"{src}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run fetch_sister_nwp.py first")
        df = pd.read_parquet(p)
        df["dt"] = pd.to_datetime(df["dt"])

        # Per-source physics (без суффикса временно)
        df = add_per_source_physics(df)

        # Применить суффикс ко всем колонкам кроме dt
        rename_map = {c: f"{c}__{src}" for c in df.columns if c != "dt"}
        df = df.rename(columns=rename_map)

        n_cols_before = len(train_full.columns)
        train_full = train_full.merge(df, on="dt", how="left")
        valid_full = valid_full.merge(df, on="dt", how="left")
        print(f"  +{src}: {len(df.columns)-1} cols, train now {len(train_full.columns)} cols")

        # Bias correction для этого source
        train_full, valid_full = per_source_bias_correction(train_full, valid_full, src)

    # Extra lags on combined frame (чтобы lag границы train/valid были корректны)
    combined = pd.concat([train_full.assign(_src="train"), valid_full.assign(_src="valid")], ignore_index=True)
    for src in sources:
        combined = add_extra_lags_per_source(combined, src)
    train_full = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_full = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()

    return train_full, valid_full


if __name__ == "__main__":
    from lgbm_q50 import prepare_features
    tf, vf = prepare_features()
    print(f"base train: {tf.shape}, valid: {vf.shape}")
    tf, vf = add_sister_features(tf, vf)
    print(f"with sister train: {tf.shape}, valid: {vf.shape}")
    sample = [c for c in tf.columns if "__gfs" in c]
    print(f"GFS cols ({len(sample)}): {sample[:8]} ...")
    sample = [c for c in tf.columns if "__icon" in c]
    print(f"ICON cols ({len(sample)}): {sample[:8]} ...")
