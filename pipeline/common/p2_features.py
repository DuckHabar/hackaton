"""
P2 add-on: дополнительные interaction features для physics_features.py.
Идея: tree-модели могут выучить эти interactions через splits, но явные фичи
делают это БЫСТРЕЕ и оставляют depth для других паттернов.

Все 5 физических явлений из v16 mining + OOF cross-check:
  F1: stability * ws, bin2 night underest, bin3 day underest
  F2: density * evening, bin1 eve underest
  F3: empirical rated_speed=10, bin4 day overest, bin6 overest
  F4: n_rep * high_ws, bin7 n_rep3 overest
  F5: evening * high_wind, bin6 eve underest
"""
import numpy as np
import pandas as pd

def add_p2_interactions(df, copy=True):
    """Добавить interaction features для F1-F5."""
    if copy:
        df = df.copy()
    # Diurnal flags (более тонкие чем hour_of_day категоризация)
    h = df["hour_of_day"]
    df["is_evening"]      = ((h >= 16) & (h < 20)).astype(np.int8)   # F2, F5
    df["is_strict_day"]   = ((h >= 6) & (h < 16)).astype(np.int8)    # F1 CBL
    df["is_strict_night"] = ((h < 6) | (h >= 20)).astype(np.int8)    # F1 LLJ
    df["is_dawn"]         = ((h >= 4) & (h < 8)).astype(np.int8)     # transition
    df["is_dusk"]         = ((h >= 16) & (h < 20)).astype(np.int8)   # same as eve

    # F2: density × evening
    df["eve_x_rho"] = df["is_evening"] * df["rho_air"]
    # F2 expanded: density × ws^3 (theoretical power coefficient)
    df["rho_x_ws3_120"] = df["rho_air"] * df["wind_speed_120m"] ** 3 / 1000.0

    # F1: stability × wind (LLJ proxy)
    df["night_x_alpha"]   = df["is_strict_night"] * df["alpha_80_120"]
    df["day_x_alpha"]     = df["is_strict_day"] * df["alpha_80_120"]
    df["night_x_ws120"]   = df["is_strict_night"] * df["wind_speed_120m"]
    df["day_x_ws120"]     = df["is_strict_day"] * df["wind_speed_120m"]

    # F3: empirical rated_speed=10 (saturation)
    df["ws120_excess_10"] = np.clip(df["wind_speed_120m"] - 10.0, 0, None)
    df["ws120_deficit_10"] = np.clip(10.0 - df["wind_speed_120m"], 0, None)
    df["ws120_excess_14"] = np.clip(df["wind_speed_120m"] - 14.0, 0, None)

    # F4: n_repair × high wind
    df["n_rep_x_high_ws"] = df["Кол-во_ВЭУ_в_ремонте"] * df["ws120_excess_14"]
    df["is_n_rep_3"]      = (df["Кол-во_ВЭУ_в_ремонте"] == 3).astype(np.int8)
    df["is_n_rep_4"]      = (df["Кол-во_ВЭУ_в_ремонте"] == 4).astype(np.int8)
    df["rep3_x_high_ws"]  = df["is_n_rep_3"] * df["ws120_excess_14"]

    # F5: evening × high wind
    df["eve_x_ws120"]     = df["is_evening"] * df["wind_speed_120m"]
    df["eve_x_high_ws"]   = df["is_evening"] * df["ws120_excess_10"]

    # Temperature gradient (additional stability signal)
    df["temp_grad_80_120"] = df["temperature_80m"] - df["temperature_120m"]   # >0 = stable
    
    # wind shear at full range, 10 и 180
    df["alpha_10_120"] = np.log(df["wind_speed_120m"] / np.maximum(df["wind_speed_10m"], 0.1)) / np.log(120/10)

    return df

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
    
    train_path = "/home/duck/wind_hackathon/data/processed/train_with_physics.parquet"
    valid_path = "/home/duck/wind_hackathon/data/processed/valid_with_physics.parquet"
    
    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)
    print(f"train: {train.shape}, valid: {valid.shape}")
    
    train_p2 = add_p2_interactions(train)
    valid_p2 = add_p2_interactions(valid)
    
    new_cols = [c for c in train_p2.columns if c not in train.columns]
    print(f"Added {len(new_cols)} features:")
    for c in new_cols:
        nan_pct = train_p2[c].isna().mean() * 100
        print(f"  {c:25s} mean={train_p2[c].mean():+.3f} nan={nan_pct:.1f}%")
    
    train_p2.to_parquet("/home/duck/wind_hackathon/data/processed/train_with_p2.parquet")
    valid_p2.to_parquet("/home/duck/wind_hackathon/data/processed/valid_with_p2.parquet")
    print("\nSaved train_with_p2.parquet, valid_with_p2.parquet")
