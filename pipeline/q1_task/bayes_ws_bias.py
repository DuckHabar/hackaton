"""Hierarchical Bayesian ws-bias model for Q1 forecast correction.

Cells: (year, month, ws_bin, sector). PyMC partial-pooling model fitted on
residuals r = ws_eff (from inverted PC) - ws_NWP_120.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.q1_task.empirical_pc import pc_inverse

WS_MIN = 4.0
WS_MAX = 14.0
P_MIN = 0.1
N_REP_STABLE = 3  # used only for fit_empirical_pc, not for bias cells
N_TURBINES = 26
N_WS_BINS = 10
N_SECTORS = 8


def sector_index_from_wd(wd_raw: np.ndarray) -> np.ndarray:
    """Compute 8-sector index from wind direction (handles denormalized convention)."""
    wd = np.asarray(wd_raw, dtype=float)
    if np.nanmax(np.abs(wd)) < 3.6:
        wd = wd * 1000.0
    wd = np.mod(wd, 360.0)
    return (((wd + 22.5) % 360.0) // 45.0).astype(int)


def build_bias_cells(train_df: pd.DataFrame, pc_model: dict) -> pd.DataFrame:
    """Filter train, compute ws_eff via PC inverse, return long-form cells DataFrame.

    n_rep filter intentionally removed: n_repair is monthly plan, filtering to n_rep=3
    drops Q1 2024+2025 (n_rep was 4/5). Bias on ws is invariant to capacity since
    p_per_turbine = P / n_active is already normalized.
    """
    df = train_df.copy()
    df["ws_120"] = df["wind_speed_120m"]
    df["rho"] = df["rho_air"]
    df["n_rep"] = df["Кол-во_ВЭУ_в_ремонте"]
    df["n_active"] = N_TURBINES - df["n_rep"]
    df["p_per_turbine"] = (
        df["Выработка. Результирующий расчет"] / df["n_active"].replace(0, np.nan)
    )
    df["dt"] = pd.to_datetime(df["METEOFORECASTHOUR_OPENM_Datetime"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    mask = (
        (df["ws_120"] >= WS_MIN)
        & (df["ws_120"] <= WS_MAX)
        & (df["p_per_turbine"] > P_MIN)
        & np.isfinite(df["rho"])
        & (df["n_active"] > 0)
    )
    sub = df.loc[mask].copy()
    sub["ws_eff"] = pc_inverse(
        sub["p_per_turbine"].to_numpy(),
        sub["rho"].to_numpy(),
        pc_model,
    )
    sub = sub.loc[np.isfinite(sub["ws_eff"])].copy()
    sub["r"] = sub["ws_eff"] - sub["ws_120"]
    sub["ws_bin"] = np.clip(
        ((sub["ws_120"] - WS_MIN) // 1).astype(int), 0, N_WS_BINS - 1
    )
    sub["sector"] = sector_index_from_wd(sub["wind_direction_120m"].to_numpy())
    cols = ["year", "month", "ws_bin", "sector", "r", "ws_120", "ws_eff", "rho", "n_rep"]
    return sub[cols].reset_index(drop=True)


def fit_bayes_ws_bias(
    cells: pd.DataFrame,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 42,
):
    """Fit hierarchical bias model on Q1-only cells."""
    import pymc as pm

    q1_cells = cells.loc[cells["month"].isin([1, 2, 3])].copy()
    years = sorted(q1_cells["year"].unique().tolist())
    months = [1, 2, 3]
    n_years = len(years)
    n_months = len(months)
    n_bins = N_WS_BINS
    n_sectors = N_SECTORS
    year_to_idx = {y: i for i, y in enumerate(years)}
    month_to_idx = {m: i for i, m in enumerate(months)}
    y_idx = q1_cells["year"].map(year_to_idx).to_numpy()
    m_idx = q1_cells["month"].map(month_to_idx).to_numpy()
    b_idx = q1_cells["ws_bin"].to_numpy()
    s_idx = q1_cells["sector"].to_numpy()
    r_obs = q1_cells["r"].to_numpy()
    with pm.Model() as _:
        sigma_obs = pm.HalfNormal("sigma_obs", 1.5)
        tau_year = pm.HalfNormal("tau_year", 0.5)
        tau_month = pm.HalfNormal("tau_month", 0.7)
        tau_bin = pm.HalfNormal("tau_bin", 0.5)
        tau_sector = pm.HalfNormal("tau_sector", 0.5)
        tau_ym = pm.HalfNormal("tau_ym", 0.3)
        tau_mb = pm.HalfNormal("tau_mb", 0.4)
        tau_ms = pm.HalfNormal("tau_ms", 0.3)
        mu_global = pm.Normal("mu_global", 0.0, 2.0)
        a_year = pm.Normal("a_year", 0.0, tau_year, shape=n_years)
        b_month = pm.Normal("b_month", 0.0, tau_month, shape=n_months)
        c_bin = pm.Normal("c_bin", 0.0, tau_bin, shape=n_bins)
        d_sector = pm.Normal("d_sector", 0.0, tau_sector, shape=n_sectors)
        ab_ym = pm.Normal("ab_ym", 0.0, tau_ym, shape=(n_years, n_months))
        bc_mb = pm.Normal("bc_mb", 0.0, tau_mb, shape=(n_months, n_bins))
        bs_ms = pm.Normal("bs_ms", 0.0, tau_ms, shape=(n_months, n_sectors))
        mu = (
            mu_global
            + a_year[y_idx]
            + b_month[m_idx]
            + c_bin[b_idx]
            + d_sector[s_idx]
            + ab_ym[y_idx, m_idx]
            + bc_mb[m_idx, b_idx]
            + bs_ms[m_idx, s_idx]
        )
        pm.Normal("likelihood", mu=mu, sigma=sigma_obs, observed=r_obs)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )
    return idata, {"years": years, "months": months}


def posterior_mu_2026(
    idata,
    meta: dict,
    year_weights: dict | None = None,
) -> np.ndarray:
    """Year-weighted posterior mean mu[m, b, s] for 2026 inference."""
    if year_weights is None:
        year_weights = {2022: 0.35, 2023: 0.35, 2024: 0.20, 2025: 0.10}
    years = meta["years"]
    months = meta["months"]
    post = idata.posterior
    mu_global = float(post["mu_global"].mean(dim=("chain", "draw")).values)
    a_year = post["a_year"].mean(dim=("chain", "draw")).values
    b_month = post["b_month"].mean(dim=("chain", "draw")).values
    c_bin = post["c_bin"].mean(dim=("chain", "draw")).values
    d_sector = post["d_sector"].mean(dim=("chain", "draw")).values
    ab_ym = post["ab_ym"].mean(dim=("chain", "draw")).values
    bc_mb = post["bc_mb"].mean(dim=("chain", "draw")).values
    bs_ms = post["bs_ms"].mean(dim=("chain", "draw")).values
    n_months = len(months)
    n_bins = N_WS_BINS
    n_sectors = N_SECTORS
    out = np.zeros((n_months, n_bins, n_sectors))
    total_w = sum(year_weights.get(y, 0.0) for y in years)
    if total_w <= 0:
        raise ValueError(f"All year weights zero for years {years}")
    for mi in range(n_months):
        for bi in range(n_bins):
            for si in range(n_sectors):
                m_mu = (
                    mu_global
                    + b_month[mi]
                    + c_bin[bi]
                    + d_sector[si]
                    + bc_mb[mi, bi]
                    + bs_ms[mi, si]
                )
                weighted_year_part = 0.0
                for yi, year in enumerate(years):
                    w = year_weights.get(year, 0.0) / total_w
                    weighted_year_part += w * (a_year[yi] + ab_ym[yi, mi])
                out[mi, bi, si] = m_mu + weighted_year_part
    return out


def posterior_diagnostics(idata) -> dict:
    """Return dict with r_hat_max, ess_min, divergent_pct, sigma diagnostics."""
    import arviz as az

    summary = az.summary(idata, var_names=["sigma_obs", "tau_year", "tau_month",
                                            "tau_bin", "tau_sector", "tau_ym",
                                            "tau_mb", "tau_ms", "mu_global"])
    r_hat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    n_divergent = int(idata.sample_stats.diverging.sum())
    n_total = int(idata.sample_stats.diverging.size)
    divergent_pct = 100.0 * n_divergent / max(n_total, 1)
    sigma_year = float(idata.posterior["tau_year"].mean().values)
    sigma_global = float(idata.posterior["sigma_obs"].mean().values)
    return {
        "r_hat_max": r_hat_max,
        "ess_min": ess_min,
        "divergent_pct": divergent_pct,
        "sigma_year": sigma_year,
        "sigma_obs": sigma_global,
        "sigma_year_ratio": sigma_year / max(sigma_global, 1e-6),
    }
