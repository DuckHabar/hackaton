"""Empirical aggregated power curve for 26 турбин.

PC fitted via isotonic regression on bin means of (P/n_active) for rows where
n_repair=3 (stable capacity). Saturation at ws=9.5 m/s (physics_findings.md sec 11).
Density correction: PC(ws, rho) = PC_iso(ws) * (rho / rho_0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

RHO_0 = 1.225
WS_SATURATION = 9.5
WS_CUTOUT = 25.0
WS_BIN_STEP = 0.5


def fit_empirical_pc(train_df: pd.DataFrame) -> dict:
    """Fit isotonic PC on train rows with n_rep=3."""
    mask = train_df["Кол-во_ВЭУ_в_ремонте"] == 3
    sub = train_df.loc[mask].copy()
    n_active = 23
    sub["p_per_turbine"] = (
        sub["Выработка. Результирующий расчет"] / n_active
    )
    sub["ws_120"] = sub["wind_speed_120m"]
    sub["ws_bin"] = (sub["ws_120"] / WS_BIN_STEP).round() * WS_BIN_STEP
    fit_mask = (sub["ws_bin"] >= 0) & (sub["ws_bin"] <= WS_SATURATION)
    bin_means = (
        sub.loc[fit_mask]
        .groupby("ws_bin")["p_per_turbine"]
        .mean()
        .reset_index()
        .sort_values("ws_bin")
    )
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(bin_means["ws_bin"].to_numpy(), bin_means["p_per_turbine"].to_numpy())
    p_max = float(iso.predict([WS_SATURATION])[0])
    return {"iso": iso, "p_max": p_max}


def pc(ws: np.ndarray, rho: np.ndarray, model: dict) -> np.ndarray:
    """Apply PC: (ws, rho) -> P per turbine."""
    ws_arr = np.asarray(ws, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    out = np.zeros_like(ws_arr)
    iso = model["iso"]
    p_max = model["p_max"]
    in_iso = (ws_arr >= 0) & (ws_arr < WS_SATURATION)
    if in_iso.any():
        out[in_iso] = iso.predict(ws_arr[in_iso])
    sat = (ws_arr >= WS_SATURATION) & (ws_arr < WS_CUTOUT)
    out[sat] = p_max
    out *= rho_arr / RHO_0
    return out


def pc_inverse(
    p_target: np.ndarray,
    rho: np.ndarray,
    model: dict,
    ws_lo: float = 0.0,
    ws_hi: float = WS_SATURATION,
    tol: float = 0.005,
    max_iter: int = 60,
) -> np.ndarray:
    """Invert PC via bisection on [ws_lo, ws_hi]. Returns NaN if not invertible."""
    p_arr = np.asarray(p_target, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    n = p_arr.shape[0]
    out = np.full(n, np.nan)
    p_max = model["p_max"]
    p_at_rho0 = p_arr / (rho_arr / RHO_0)
    for i in range(n):
        p = p_at_rho0[i]
        if not np.isfinite(p):
            continue
        if p <= 0:
            out[i] = 0.0
            continue
        if p >= p_max:
            out[i] = WS_SATURATION
            continue
        lo, hi = ws_lo, ws_hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            p_mid = pc(np.array([mid]), np.array([RHO_0]), model)[0]
            if abs(p_mid - p) < tol:
                out[i] = mid
                break
            if p_mid < p:
                lo = mid
            else:
                hi = mid
        else:
            out[i] = 0.5 * (lo + hi)
    return out
