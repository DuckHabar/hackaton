import numpy as np
import pandas as pd
import pytest
from pipeline.day18_task import weather_variants as wv


@pytest.fixture
def sample_test_df():
    rng = pd.date_range("2026-04-01", periods=1152, freq="h")
    df = pd.DataFrame({
        "METEOFORECASTHOUR_OPENM_Datetime": rng,
        "wind_speed_120m": np.linspace(2, 20, 1152),
        "wind_speed_80m": np.linspace(1.5, 18, 1152),
        "wind_direction_120m": np.linspace(0, 0.359, 1152),
        "temperature_80m": np.linspace(0, 25, 1152),
        "temperature_120m": np.linspace(-0.5, 24.5, 1152),
        "pressure_msl": np.full(1152, 1013.0),
        "rain": np.zeros(1152),
        "snowfall": np.zeros(1152),
        "showers": np.zeros(1152),
        "cloud_cover_low": np.zeros(1152),
        "wind_gusts_10m": np.linspace(3, 25, 1152),
        "wind_speed_10m": np.linspace(1, 15, 1152),
        "wind_speed_180m": np.linspace(3, 22, 1152),
        "wind_direction_10m": np.linspace(0, 0.359, 1152),
        "wind_direction_80m": np.linspace(0, 0.359, 1152),
        "wind_direction_180m": np.linspace(0, 0.359, 1152),
        "Кол-во_ВЭУ_в_ремонте": np.full(1152, 2, dtype=int),
        "Выработка. Результирующий расчет": np.concatenate([np.linspace(0, 80, 1128), [np.nan] * 24]),
    })
    return df


def test_org_branch_returns_unchanged_weather(sample_test_df):
    out = wv.apply_weather_org(sample_test_df)
    for col in ["wind_speed_120m", "wind_direction_120m", "temperature_80m", "rain"]:
        assert (out[col].to_numpy() == sample_test_df[col].to_numpy()).all()


def test_org_branch_preserves_row_count(sample_test_df):
    out = wv.apply_weather_org(sample_test_df)
    assert len(out) == len(sample_test_df)


from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_openmeteo_response():
    return {
        "hourly": {
            "time": [f"2026-05-18T{h:02d}:00" for h in range(24)],
            "temperature_2m": [15.0 + h * 0.5 for h in range(24)],
            "pressure_msl": [1013.0] * 24,
            "wind_speed_10m": [3.0 + h * 0.2 for h in range(24)],
            "wind_speed_80m": [5.0 + h * 0.3 for h in range(24)],
            "wind_speed_120m": [6.0 + h * 0.4 for h in range(24)],
            "wind_speed_180m": [7.0 + h * 0.5 for h in range(24)],
            "wind_direction_10m": [180.0] * 24,
            "wind_direction_80m": [185.0] * 24,
            "wind_direction_120m": [190.0] * 24,
            "wind_direction_180m": [195.0] * 24,
            "wind_gusts_10m": [4.0 + h * 0.3 for h in range(24)],
            "rain": [0.0] * 24,
            "showers": [0.0] * 24,
            "snowfall": [0.0] * 24,
            "cloud_cover_low": [50.0] * 24,
            "cloud_cover": [60.0] * 24,
        }
    }


def test_fetch_openmeteo_returns_24_rows(mock_openmeteo_response, tmp_path):
    with patch("pipeline.day18_task.weather_variants.requests.get") as mock_get:
        resp = MagicMock()
        resp.json.return_value = mock_openmeteo_response
        resp.status_code = 200
        mock_get.return_value = resp
        df = wv.fetch_openmeteo_forecast("gfs_seamless", "2026-05-18", cache_dir=tmp_path)
    assert len(df) == 24
    assert "wind_speed_120m" in df.columns
    assert df["wind_speed_120m"].iloc[0] == 6.0


def test_fetch_uses_cache_on_second_call(mock_openmeteo_response, tmp_path):
    with patch("pipeline.day18_task.weather_variants.requests.get") as mock_get:
        resp = MagicMock()
        resp.json.return_value = mock_openmeteo_response
        resp.status_code = 200
        mock_get.return_value = resp
        df1 = wv.fetch_openmeteo_forecast("gfs_seamless", "2026-05-18", cache_dir=tmp_path)
        df2 = wv.fetch_openmeteo_forecast("gfs_seamless", "2026-05-18", cache_dir=tmp_path)
    assert mock_get.call_count == 1
    pd.testing.assert_frame_equal(df1, df2)


def test_consensus_median_for_scalar():
    dfs = []
    for shift in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        df = pd.DataFrame({
            "time": pd.date_range("2026-05-18", periods=24, freq="h"),
            "wind_speed_120m": np.full(24, 10.0 + shift),
            "wind_direction_120m": np.full(24, 180.0),
        })
        dfs.append(df)
    consensus = wv.build_consensus(dfs)
    # median 7 значений [9..15] = 12
    assert np.allclose(consensus["wind_speed_120m"], 12.0)


def test_consensus_circular_mean_for_direction():
    dfs = []
    for deg in [0.0, 10.0, 350.0]:
        df = pd.DataFrame({
            "time": pd.date_range("2026-05-18", periods=24, freq="h"),
            "wind_speed_120m": np.full(24, 10.0),
            "wind_direction_120m": np.full(24, deg),
        })
        dfs.append(df)
    consensus = wv.build_consensus(dfs)
    # circular mean(0, 10, 350) ~ 0
    res = consensus["wind_direction_120m"].iloc[0]
    assert min(abs(res), abs(res - 360.0)) < 1.0


def test_overlay_archive_replaces_elapsed_hours():
    forecast = pd.DataFrame({
        "time": pd.date_range("2026-05-18", periods=24, freq="h"),
        "wind_speed_120m": np.full(24, 5.0),
    })
    archive = pd.DataFrame({
        "time": pd.date_range("2026-05-18", periods=24, freq="h"),
        "wind_speed_120m": np.full(24, 9.0),
    })
    cutoff = pd.Timestamp("2026-05-18 10:00")
    merged = wv.overlay_archive(forecast, archive, cutoff)
    assert (merged.loc[merged["time"] <= cutoff, "wind_speed_120m"] == 9.0).all()
    assert (merged.loc[merged["time"] > cutoff, "wind_speed_120m"] == 5.0).all()
