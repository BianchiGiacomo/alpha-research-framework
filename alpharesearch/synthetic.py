"""
Synthetic price panels for calibrating the research pipeline.

``null`` panels contain no predictable cross-sectional structure by
construction (common market factor + fat tails + volatility clustering), so
any "alpha" a pipeline finds on them is a false positive. ``planted`` panels
add a known short-term reversal effect of controllable strength, so they
measure whether a pipeline can find a real signal (power).

A methodology is only credible if it is quiet on the first and loud on the
second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PricePanel


def synthetic_panel(n_assets: int = 10, n_days: int = 1500, seed: int = 0,
                    signal_strength: float = 0.0, signal_lookback: int = 5,
                    df: float = 4.0, start: str = "2018-01-01") -> PricePanel:
    """
    Generate a daily OHLCV panel.

    Log returns follow
        r[t, i] = beta_i * m[t] + s[t] * sigma_i * (eps[t, i] - k * z[t-1, i])
    where ``m`` is a market factor, ``s`` a shared GARCH(1,1)-style volatility
    multiplier, ``eps`` Student-t noise and ``z[t-1, i]`` the cross-sectional
    z-score of each asset's past ``signal_lookback``-day return. With
    ``k = signal_strength = 0`` future returns are unpredictable from the past.
    With ``k > 0`` past winners underperform (reversal); the expected daily
    rank IC of the reversal signal is roughly ``k / sqrt(1 + k**2)``.
    """
    rng = np.random.default_rng(seed)
    t_scale = np.sqrt((df - 2) / df)  # unit-variance Student-t

    beta = rng.uniform(0.7, 1.3, n_assets)
    sigma = rng.uniform(0.02, 0.05, n_assets)  # idiosyncratic daily vol
    sigma_m = 0.025

    # Shared volatility regime (GARCH(1,1) on the market shock).
    omega, a, b = 0.05, 0.08, 0.90
    var = np.empty(n_days)
    var[0] = 1.0
    m = np.empty(n_days)
    for t in range(n_days):
        if t > 0:
            var[t] = omega + a * (m[t - 1] / sigma_m) ** 2 + b * var[t - 1]
        m[t] = sigma_m * np.sqrt(var[t]) * rng.standard_t(df) * t_scale
    s = np.sqrt(var)

    eps = rng.standard_t(df, size=(n_days, n_assets)) * t_scale
    logret = np.zeros((n_days, n_assets))
    for t in range(n_days):
        shock = eps[t]
        if signal_strength and t > signal_lookback:
            past = logret[t - signal_lookback:t].sum(axis=0)
            z = (past - past.mean()) / (past.std() + 1e-12)
            shock = shock - signal_strength * z
        logret[t] = beta * m[t] + s[t] * sigma * shock

    dates = pd.date_range(start, periods=n_days, freq="D", name="date")
    cols = [f"SYN{i:02d}" for i in range(n_assets)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(logret, axis=0)), index=dates, columns=cols)

    spread = np.abs(rng.normal(0, 0.01, size=(n_days, n_assets))) * s[:, None]
    high = close * np.exp(spread)
    low = close * np.exp(-spread)

    # Volume: independent log-AR(1), no information about returns.
    lv = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        lv[t] = 0.9 * lv[t - 1] + rng.normal(0, 0.3, n_assets)
    volume = pd.DataFrame(1e6 * np.exp(lv), index=dates, columns=cols)

    return PricePanel(close=close, volume=volume, high=high, low=low,
                      manifest={"synthetic": f"seed={seed},k={signal_strength}"})
