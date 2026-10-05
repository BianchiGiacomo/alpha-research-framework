"""
One evaluation = one configuration run end-to-end with the same protocol:
walk-forward scores -> rank IC with Newey-West t -> backtests with costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import PricePanel
from .metrics import ic_summary, rank_ic
from .pipeline import StudyConfig, build_dataset, predict_walk_forward


@dataclass(frozen=True)
class BacktestConfig:
    top_k: int = 2
    cost_bps: float = 20.0
    periods_per_year: int = 365


def evaluate(panel: PricePanel, cfg: StudyConfig, bt: BacktestConfig = BacktestConfig(),
             first_test: Optional[str] = None, last_test: Optional[str] = None,
             keep_scores: bool = False) -> Dict:
    """
    ``first_test`` / ``last_test`` are dates bounding the test windows
    (``last_test`` exclusive). Fixing them across configurations makes all
    trials comparable on exactly the same out-of-sample days.
    """
    ds = build_dataset(panel, cfg)
    pos = lambda d: int(np.searchsorted(ds.dates.values, np.datetime64(pd.Timestamp(d))))  # noqa: E731
    scores, folds = predict_walk_forward(
        ds, cfg,
        first_test=0 if first_test is None else pos(first_test),
        last_test=None if last_test is None else pos(last_test))

    ic = rank_ic(scores, ds.fwd)
    ls = run_backtest(scores, panel.close, bt.top_k, "long_short", cfg.horizon,
                      bt.cost_bps, periods_per_year=bt.periods_per_year)
    lo = run_backtest(scores, panel.close, bt.top_k, "long_only", cfg.horizon,
                      bt.cost_bps, periods_per_year=bt.periods_per_year)
    out = {
        "config": cfg.to_dict(),
        "n_folds": len(folds),
        "test_period": [folds[0]["test_start"], folds[-1]["test_end"]] if folds else None,
        "ic": ic_summary(ic, cfg.horizon),
        "long_short": {k: v for k, v in ls.items() if k != "returns"},
        "long_only_excess": {k: v for k, v in lo.items() if k != "returns"},
        "_ls_returns": ls["returns"],
        "_ic_series": ic,
        "_folds": folds,
    }
    if keep_scores:
        out["_scores"] = scores
        out["_fwd"] = ds.fwd
    return out


def public(result: Dict) -> Dict:
    """Drop the private (series-valued) fields before serialization."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
