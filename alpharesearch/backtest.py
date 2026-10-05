"""
Portfolio construction from score panels.

Timing convention: weights decided with information up to the close of day t
earn the close-to-close return from t to t+1. Positions are rebalanced every
``rebalance_every`` days (default: the label horizon, so the holding period
matches what the model was trained to predict) and are held unchanged in
between. Costs are charged on traded notional: ``cost_bps * sum|w_t - w_{t-1}|``.
Weight drift between rebalances is ignored.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .metrics import max_drawdown, sharpe


def target_weights(scores: pd.DataFrame, top_k: int, mode: str = "long_short") -> pd.DataFrame:
    """
    Equal-weight the ``top_k`` highest scores (and short the ``top_k`` lowest
    for ``long_short``). Dates with fewer than ``2 * top_k`` scored assets
    (``top_k + 1`` for ``long_only``) get no position.
    """
    n = scores.notna().sum(axis=1)
    rank_desc = scores.rank(axis=1, ascending=False, method="first")
    rank_asc = scores.rank(axis=1, ascending=True, method="first")
    if mode == "long_short":
        w = (rank_desc <= top_k).astype(float) / top_k - (rank_asc <= top_k).astype(float) / top_k
        ok = n >= 2 * top_k
    elif mode == "long_only":
        w = (rank_desc <= top_k).astype(float) / top_k
        ok = n >= top_k + 1
    else:
        raise ValueError(mode)
    return w.where(scores.notna(), 0.0).mul(ok, axis=0)


def run_backtest(scores: pd.DataFrame, close: pd.DataFrame, top_k: int,
                 mode: str = "long_short", rebalance_every: int = 5,
                 cost_bps: float = 20.0, benchmark: Optional[str] = "equal_weight",
                 periods_per_year: int = 365) -> Dict:
    """
    Backtest over the dates where ``scores`` has at least one value.
    For ``long_only`` the reported statistics are on returns in excess of an
    equal-weight portfolio of all scored assets (costs on the benchmark are
    ignored, which is generous to the benchmark-relative strategy only by its
    own rebalancing cost).
    """
    active = scores.notna().any(axis=1)
    scores = scores.loc[active]
    rets = close.pct_change(fill_method=None).shift(-1).reindex(scores.index)  # t -> t+1

    target = target_weights(scores, top_k, mode)
    held = target.to_numpy(copy=True)
    unscored = scores.isna().to_numpy()
    for i in range(1, len(held)):
        # Rebalance on schedule, or early when the held book contains an asset
        # that lost its score (e.g. it dropped out of the universe).
        if i % rebalance_every != 0 and not (held[i - 1] != 0)[unscored[i]].any():
            held[i] = held[i - 1]
    held = pd.DataFrame(held, index=target.index, columns=target.columns)

    gross = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 1e4

    if mode == "long_only" and benchmark == "equal_weight":
        bench = rets.where(scores.notna()).mean(axis=1).fillna(0.0)
        net = net - bench
        gross = gross - bench

    net = net.iloc[:-1]  # last day has no realized t+1 return
    gross = gross.iloc[:-1]
    return {
        "returns": net,
        "sharpe_net": sharpe(net, periods_per_year),
        "sharpe_gross": sharpe(gross, periods_per_year),
        "ann_return_net": float(net.mean() * periods_per_year),
        "max_drawdown": max_drawdown(net),
        "avg_daily_turnover": float(turnover.mean()),
        "n_days": int(len(net)),
    }
