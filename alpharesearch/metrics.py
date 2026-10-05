"""
Evaluation statistics.

The IC is measured the way it is traded: one Spearman correlation across
assets per date (``rank_ic``). With an h-day label, consecutive daily ICs
share h-1 days of returns, so their naive t-statistic is inflated; the
headline statistic is therefore the Newey-West t-stat.

Sharpe ratios come with the Probabilistic and Deflated Sharpe Ratio
(Bailey & Lopez de Prado, 2012/2014) and the Probability of Backtest
Overfitting via CSCV (Bailey, Borwein, Lopez de Prado & Zhu, 2017), so that
the number of configurations tried is part of the evidence.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# Information coefficient
# --------------------------------------------------------------------------

def rank_ic(scores: pd.DataFrame, fwd: pd.DataFrame, min_assets: int = 3) -> pd.Series:
    """Per-date Spearman correlation between scores and forward returns."""
    s, f = scores.align(fwd, join="inner")
    valid = s.notna() & f.notna()
    s, f = s.where(valid), f.where(valid)
    rs = s.rank(axis=1)
    rf = f.rank(axis=1)
    rs = rs.sub(rs.mean(axis=1), axis=0)
    rf = rf.sub(rf.mean(axis=1), axis=0)
    num = (rs * rf).sum(axis=1)
    den = np.sqrt((rs ** 2).sum(axis=1) * (rf ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic[valid.sum(axis=1) >= min_assets].dropna()


def newey_west_tstat(x: Sequence[float], lags: Optional[int] = None) -> float:
    """t-statistic of the mean with a Bartlett-kernel HAC standard error."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    lrv = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    if lrv <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(lrv / n))


def ic_summary(ic: pd.Series, horizon: int) -> Dict[str, float]:
    """IC statistics; ``nw_lags`` covers the label overlap (h-1) with margin."""
    x = ic.to_numpy()
    n = len(x)
    if n < 3:
        return {"n_days": n}
    lags = max(horizon, int(np.floor(4 * (n / 100) ** (2 / 9))))
    t_nw = newey_west_tstat(x, lags)
    t_naive = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    return {
        "n_days": int(n),
        "mean_ic": float(x.mean()),
        "ic_std": float(x.std(ddof=1)),
        "icir_annual": float(x.mean() / x.std(ddof=1) * np.sqrt(365)),
        "hit_rate": float((x > 0).mean()),
        "t_naive": float(t_naive),
        "t_nw": t_nw,
        "nw_lags": int(lags),
        "p_nw": float(2 * stats.norm.sf(abs(t_nw))) if np.isfinite(t_nw) else float("nan"),
    }


# --------------------------------------------------------------------------
# Return statistics
# --------------------------------------------------------------------------

def sharpe(r: Sequence[float], periods_per_year: int = 365) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def max_drawdown(r: Sequence[float]) -> float:
    eq = np.cumprod(1 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(eq)
    return float(np.max(1 - eq / peak)) if len(eq) else float("nan")


def probabilistic_sharpe_ratio(r: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """
    P(true per-period Sharpe > ``sr_benchmark``), accounting for sample length,
    skewness and kurtosis of the returns. Both Sharpes are per period (not
    annualized).
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)))


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """Expected maximum per-period Sharpe among N independent zero-skill trials."""
    srs = np.asarray(trial_sharpes, dtype=float)
    srs = srs[np.isfinite(srs)]
    n = len(srs)
    if n < 2:
        return 0.0
    return float(np.sqrt(srs.var(ddof=1)) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe_ratio(r: Sequence[float], trial_sharpes: Sequence[float]) -> float:
    """PSR against the Sharpe the best of ``len(trial_sharpes)`` lucky trials would reach."""
    return probabilistic_sharpe_ratio(r, expected_max_sharpe(trial_sharpes))


def probability_of_backtest_overfitting(returns: np.ndarray, n_blocks: int = 16) -> Dict[str, float]:
    """
    CSCV estimate of PBO. ``returns`` is (T periods x N configurations).
    The rows are cut into ``n_blocks`` contiguous blocks; for every way of
    choosing half of them as in-sample, the configuration with the best
    in-sample Sharpe is located in the out-of-sample ranking. PBO is the share
    of splits where it lands in the bottom half.
    """
    M = np.asarray(returns, dtype=float)
    M = M[np.isfinite(M).all(axis=1)]
    T, N = M.shape
    if N < 2 or T < 2 * n_blocks:
        return {"pbo": float("nan"), "n_splits": 0}
    blocks = np.array_split(np.arange(T), n_blocks)
    s1 = np.array([M[b].sum(axis=0) for b in blocks])
    s2 = np.array([(M[b] ** 2).sum(axis=0) for b in blocks])
    cnt = np.array([len(b) for b in blocks])[:, None]

    def block_sharpe(sel):
        n = cnt[sel].sum()
        mu = s1[sel].sum(axis=0) / n
        var = s2[sel].sum(axis=0) / n - mu ** 2
        return mu / np.sqrt(np.maximum(var, 1e-18))

    logits = []
    for is_blocks in combinations(range(n_blocks), n_blocks // 2):
        is_sel = np.zeros(n_blocks, bool)
        is_sel[list(is_blocks)] = True
        best = int(np.argmax(block_sharpe(is_sel)))
        oos = block_sharpe(~is_sel)
        omega = stats.rankdata(oos)[best] / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": int(len(logits)),
            "median_logit": float(np.median(logits))}
