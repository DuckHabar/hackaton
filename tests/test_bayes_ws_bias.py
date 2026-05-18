"""Tests for WS-bias cell construction and posterior averaging."""

import numpy as np
import pandas as pd
import pytest

from pipeline.q1_task.empirical_pc import fit_empirical_pc, RHO_0
from pipeline.q1_task.bayes_ws_bias import (
    build_bias_cells,
    sector_index_from_wd,
    WS_MIN,
    WS_MAX,
    P_MIN,
    N_REP_STABLE,
    N_WS_BINS,
    N_SECTORS,
)


@pytest.fixture
def synthetic_train_with_dates():
    rng = np.random.default_rng(42)
    n = 3000
    ws = rng.uniform(2, 16, n)
    p_per_turbine = np.clip(0.0045 * ws ** 3, 0, 3.5)
    n_rep = np.full(n, 3)
    n_rep[n // 2:] = 4
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    wd = rng.uniform(0, 360, n)
    return pd.DataFrame(
        {
            "Кол-во_ВЭУ_в_ремонте": n_rep,
            "wind_speed_120m": ws,
            "wind_direction_120m": wd / 1000.0,
            "Выработка. Результирующий расчет": p_per_turbine * (26 - n_rep),
            "rho_air": np.full(n, RHO_0),
            "METEOFORECASTHOUR_OPENM_Datetime": dates,
        }
    )


def test_build_cells_no_nrep_filter(synthetic_train_with_dates):
    pc_model = fit_empirical_pc(synthetic_train_with_dates)
    cells = build_bias_cells(synthetic_train_with_dates, pc_model)
    assert (cells["year"] >= 2022).all()
    assert set(cells["n_rep"].unique()) == {3, 4}, "both n_rep values should remain"
    assert len(cells) > 0


def test_build_cells_ws_bin_range(synthetic_train_with_dates):
    pc_model = fit_empirical_pc(synthetic_train_with_dates)
    cells = build_bias_cells(synthetic_train_with_dates, pc_model)
    assert cells["ws_bin"].min() >= 0
    assert cells["ws_bin"].max() <= N_WS_BINS - 1


def test_build_cells_sector_range(synthetic_train_with_dates):
    pc_model = fit_empirical_pc(synthetic_train_with_dates)
    cells = build_bias_cells(synthetic_train_with_dates, pc_model)
    assert cells["sector"].min() >= 0
    assert cells["sector"].max() <= N_SECTORS - 1


def test_build_cells_columns(synthetic_train_with_dates):
    pc_model = fit_empirical_pc(synthetic_train_with_dates)
    cells = build_bias_cells(synthetic_train_with_dates, pc_model)
    expected_cols = {"year", "month", "ws_bin", "sector", "r", "ws_120", "ws_eff"}
    assert expected_cols.issubset(set(cells.columns))


def test_sector_index_basic():
    assert sector_index_from_wd(np.array([0.0]))[0] == 0
    assert sector_index_from_wd(np.array([90.0]))[0] == 2
    assert sector_index_from_wd(np.array([180.0]))[0] == 4
    assert sector_index_from_wd(np.array([270.0]))[0] == 6


def test_sector_index_denormalized():
    wd_denorm = np.array([0.090])
    sec = sector_index_from_wd(wd_denorm)
    assert sec[0] == 2
