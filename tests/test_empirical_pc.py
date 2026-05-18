"""Tests for empirical aggregated PC + inverse."""

import numpy as np
import pandas as pd
import pytest

from pipeline.q1_task.empirical_pc import (
    fit_empirical_pc,
    pc,
    pc_inverse,
    RHO_0,
    WS_SATURATION,
    WS_CUTOUT,
)


@pytest.fixture
def synthetic_train():
    """1000 synthetic rows: ws uniform [0, 22], cubic power up to 3.5 МВт rated."""
    rng = np.random.default_rng(42)
    ws = rng.uniform(0, 22, 1000)
    p_per_turbine = np.clip(0.0045 * ws ** 3, 0, 3.5)
    p_per_turbine[ws > WS_CUTOUT] = 0
    n_active = 23
    return pd.DataFrame(
        {
            "Кол-во_ВЭУ_в_ремонте": np.full(1000, 3),
            "wind_speed_120m": ws,
            "Выработка. Результирующий расчет": p_per_turbine * n_active,
            "rho_air": np.full(1000, RHO_0),
        }
    )


def test_pc_monotonic_below_saturation(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    ws = np.linspace(0, WS_SATURATION - 0.1, 50)
    rho = np.full_like(ws, RHO_0)
    p = pc(ws, rho, model)
    assert np.all(np.diff(p) >= -1e-9), "PC must be monotonic non-decreasing on [0, sat]"


def test_pc_zero_at_zero(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    p = pc(np.array([0.0]), np.array([RHO_0]), model)
    assert 0.0 <= p[0] < 0.5


def test_pc_zero_above_cutout(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    p = pc(np.array([WS_CUTOUT + 1.0]), np.array([RHO_0]), model)
    assert p[0] == 0.0


def test_pc_saturated_above_ws_sat(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    p_at_sat = pc(np.array([WS_SATURATION]), np.array([RHO_0]), model)[0]
    p_above = pc(np.array([15.0]), np.array([RHO_0]), model)[0]
    assert np.isclose(p_above, p_at_sat)


def test_pc_density_scaling(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    ws = np.array([7.0])
    p_low = pc(ws, np.array([1.0]), model)[0]
    p_high = pc(ws, np.array([1.5]), model)[0]
    assert p_high > p_low
    assert np.isclose(p_high / p_low, 1.5, rtol=1e-6)


def test_pc_inverse_roundtrip(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    ws_input = np.linspace(4.0, 9.0, 20)
    rho = np.full_like(ws_input, RHO_0)
    p_target = pc(ws_input, rho, model)
    ws_inv = pc_inverse(p_target, rho, model)
    assert np.max(np.abs(ws_input - ws_inv)) < 0.15, "Inverse must roundtrip within 0.15 м/с"


def test_pc_inverse_above_saturation(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    p_above_max = np.array([model["p_max"] * 2.0])
    rho = np.array([RHO_0])
    ws_inv = pc_inverse(p_above_max, rho, model)
    assert ws_inv[0] == WS_SATURATION


def test_pc_inverse_zero_power(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    p_zero = np.array([0.0])
    rho = np.array([RHO_0])
    ws_inv = pc_inverse(p_zero, rho, model)
    assert ws_inv[0] == 0.0


def test_pc_inverse_density_invariance(synthetic_train):
    model = fit_empirical_pc(synthetic_train)
    ws_input = np.array([7.0])
    p_at_rho_low = pc(ws_input, np.array([1.0]), model)
    p_at_rho_high = pc(ws_input, np.array([1.5]), model)
    ws_inv_low = pc_inverse(p_at_rho_low, np.array([1.0]), model)
    ws_inv_high = pc_inverse(p_at_rho_high, np.array([1.5]), model)
    assert abs(ws_inv_low[0] - ws_inv_high[0]) < 0.05
