import numpy as np
import pandas as pd
import pytest
from pipeline.day18_task import features as feat


@pytest.fixture
def small_df():
    rng = pd.date_range("2025-01-01", periods=48, freq="h")
    return pd.DataFrame({
        "METEOFORECASTHOUR_OPENM_Datetime": rng,
        "wind_speed_10m": np.linspace(2, 15, 48),
        "wind_speed_80m": np.linspace(3, 18, 48),
        "wind_speed_120m": np.linspace(3.5, 19, 48),
        "wind_speed_180m": np.linspace(4, 20, 48),
        "wind_direction_10m": np.linspace(0.0, 0.359, 48),
        "wind_direction_80m": np.linspace(0.0, 0.359, 48),
        "wind_direction_120m": np.linspace(0.0, 0.359, 48),
        "wind_direction_180m": np.linspace(0.0, 0.359, 48),
        "wind_gusts_10m": np.linspace(3, 22, 48),
        "temperature_80m": np.linspace(-5, 25, 48),
        "temperature_120m": np.linspace(-5.3, 24.7, 48),
        "pressure_msl": np.linspace(995, 1025, 48),
        "rain": np.zeros(48),
        "showers": np.zeros(48),
        "snowfall": np.zeros(48),
        "cloud_cover_low": np.linspace(0, 1, 48),
        "Кол-во_ВЭУ_в_ремонте": np.full(48, 3, dtype=int),
        "Выработка. Результирующий расчет": np.linspace(0, 80, 48),
    })


def test_core_features_returns_dataframe_with_expected_columns(small_df):
    out = feat.build_core_features(small_df)
    expected = {
        "rho", "ws_rho_120m",
        "shear_180_80", "shear_120_10", "gust_ratio",
        "wd_sin_120m", "wd_cos_120m",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_m1", "is_m2", "is_m3", "is_m4", "is_m5",
        "is_weekend",
        "n_active", "n_active_ratio",
        "icing_flag",
        "pc_pred", "pc_residual", "bin_idx",
    }
    assert expected.issubset(set(out.columns))
    assert len(out) == len(small_df)


def test_n_active_when_n_rep_3(small_df):
    out = feat.build_core_features(small_df)
    assert (out["n_active"] == 23).all()
    assert np.isclose(out["n_active_ratio"], 23 / 26).all()


def test_rho_in_physical_range(small_df):
    out = feat.build_core_features(small_df)
    assert (out["rho"] > 1.0).all()
    assert (out["rho"] < 1.4).all()


def test_icing_flag_only_in_cold_window(small_df):
    out = feat.build_core_features(small_df)
    expected_flag = (
        (small_df["temperature_80m"] >= -5) & (small_df["temperature_80m"] <= 2)
    ).astype(int).to_numpy()
    assert (out["icing_flag"].to_numpy() == expected_flag).all()


def test_wd_sin_cos_unit_circle(small_df):
    out = feat.build_core_features(small_df)
    norm = out["wd_sin_120m"] ** 2 + out["wd_cos_120m"] ** 2
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_pc_pred_monotone_in_wind_below_saturation(small_df):
    # ниже saturation (ws < 9 м/с) pc_pred должен расти монотонно по ветру
    out = feat.build_core_features(small_df)
    pc = out["pc_pred"].to_numpy()
    diffs = np.diff(pc[:16])
    assert (diffs >= -1e-6).all()


def test_lag_1d_equals_target_24h_back():
    rng = pd.date_range("2025-01-01", periods=72, freq="h")
    df = pd.DataFrame({
        "METEOFORECASTHOUR_OPENM_Datetime": rng,
        "Выработка. Результирующий расчет": np.arange(72, dtype=float),
    })
    out = feat.build_lag_features(df)
    assert out.loc[24, "lag_1d_power"] == 0.0
    assert out.loc[48, "lag_1d_power"] == 24.0
    assert pd.isna(out.loc[10, "lag_1d_power"])


def test_lag_7d_equals_target_168h_back():
    rng = pd.date_range("2025-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "METEOFORECASTHOUR_OPENM_Datetime": rng,
        "Выработка. Результирующий расчет": np.arange(200, dtype=float),
    })
    out = feat.build_lag_features(df)
    assert out.loc[168, "lag_7d_power"] == 0.0
    assert out.loc[199, "lag_7d_power"] == 31.0


def test_rolling_24h_mean_uses_past_only():
    rng = pd.date_range("2025-01-01", periods=48, freq="h")
    df = pd.DataFrame({
        "METEOFORECASTHOUR_OPENM_Datetime": rng,
        "Выработка. Результирующий расчет": np.arange(48, dtype=float),
    })
    out = feat.build_lag_features(df)
    # для t=24 ожидаем mean(0..23) = 11.5, current row (24) НЕ включён
    assert np.isclose(out.loc[24, "rolling_24h_mean"], 11.5)
    assert np.isclose(out.loc[25, "rolling_24h_mean"], 12.5)


def test_lag_features_dont_change_when_current_target_perturbed():
    rng = pd.date_range("2025-01-01", periods=72, freq="h")
    df = pd.DataFrame({
        "METEOFORECASTHOUR_OPENM_Datetime": rng,
        "Выработка. Результирующий расчет": np.arange(72, dtype=float),
    })
    out_orig = feat.build_lag_features(df)
    df_pert = df.copy()
    df_pert.loc[50, "Выработка. Результирующий расчет"] = 99999.0
    out_pert = feat.build_lag_features(df_pert)
    lag_cols = ["lag_1d_power", "lag_7d_power", "rolling_24h_mean", "rolling_24h_std", "same_hour_7d_mean"]
    for col in lag_cols:
        if col in out_orig.columns:
            a = out_orig.loc[50, col]
            b = out_pert.loc[50, col]
            assert (a == b) or (pd.isna(a) and pd.isna(b)), f"{col} изменилась при правке target[50]"


def test_same_hour_7d_mean_uses_past_same_hour_only():
    rng = pd.date_range("2025-01-01 00:00", periods=24 * 10, freq="h")
    df = pd.DataFrame({
        "METEOFORECASTHOUR_OPENM_Datetime": rng,
        "Выработка. Результирующий расчет": np.tile(np.arange(24, dtype=float), 10),
    })
    out = feat.build_lag_features(df)
    # на 8-й день (t=192, час 0): среднее target[час=0] по 7 предыдущим дням = 0
    assert np.isclose(out.loc[192, "same_hour_7d_mean"], 0.0)
    # на t=200 (8-й день, час 8): среднее target[час=8] по дням 1..7 = 8
    assert np.isclose(out.loc[200, "same_hour_7d_mean"], 8.0)
