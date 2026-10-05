"""
Backward-looking features on wide price panels.

Every feature at date t uses only prices up to and including t.
``check_no_lookahead`` verifies this empirically by corrupting the future and
checking that the past does not move; the test-suite runs it on every
feature set.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from .data import PricePanel

FeatureDict = Dict[str, pd.DataFrame]
EPS = 1e-12


def _ret(close, p):
    return close.pct_change(p, fill_method=None)


def _rsi(close, p):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(p, min_periods=p).mean()
    loss = (-delta.clip(upper=0)).rolling(p, min_periods=p).mean()
    rs = gain / (loss + EPS)
    return (100 - 100 / (1 + rs) - 50) / 50  # in [-1, 1]


def compact_features(panel: PricePanel) -> FeatureDict:
    """The 8 features of legacy experiment C7."""
    close, volume = panel.close, panel.volume
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    return {
        "ret_5d": _ret(close, 5),
        "ret_10d": _ret(close, 10),
        "ret_20d": _ret(close, 20),
        "vol_20d": _ret(close, 1).rolling(20, min_periods=20).std(),
        "sma_20_dist": close / sma20 - 1,
        "rsi_14": _rsi(close, 14),
        "volume_ratio": volume / (volume.rolling(20, min_periods=20).mean() + EPS),
        "trend": sma20 / sma50 - 1,
    }


def extended_features(panel: PricePanel) -> FeatureDict:
    """The 41 features of legacy experiment C6."""
    close, volume = panel.close, panel.volume
    high = panel.high if panel.high is not None else close
    low = panel.low if panel.low is not None else close
    f: FeatureDict = {}
    for p in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        f[f"ret_{p}d"] = _ret(close, p)
    f["ret_accel"] = f["ret_5d"] - f["ret_5d"].shift(5)

    daily = _ret(close, 1)
    for p in [5, 10, 20, 30]:
        f[f"vol_{p}d"] = daily.rolling(p, min_periods=p).std()
    f["vol_ratio_5_20"] = f["vol_5d"] / (f["vol_20d"] + EPS)
    f["vol_change"] = f["vol_5d"] - f["vol_5d"].shift(5)

    for p in [5, 10, 20, 50]:
        f[f"sma_{p}_dist"] = close / close.rolling(p, min_periods=p).mean() - 1
        f[f"ema_{p}_dist"] = close / close.ewm(span=p, min_periods=p).mean() - 1

    for p in [10, 20]:
        mid = close.rolling(p, min_periods=p).mean()
        sd = close.rolling(p, min_periods=p).std()
        f[f"bb_{p}_position"] = (close - mid) / (2 * sd + EPS)
        f[f"bb_{p}_width"] = 4 * sd / (mid + EPS)

    for p in [7, 14, 21]:
        f[f"rsi_{p}"] = _rsi(close, p)

    for fast, slow in [(8, 21), (12, 26)]:
        macd = (close.ewm(span=fast, min_periods=fast).mean()
                - close.ewm(span=slow, min_periods=slow).mean()) / close
        f[f"macd_{fast}_{slow}"] = macd
        f[f"macd_{fast}_{slow}_hist"] = macd - macd.ewm(span=9, min_periods=9).mean()

    for p in [10, 20]:
        hi = high.rolling(p, min_periods=p).max()
        lo = low.rolling(p, min_periods=p).min()
        f[f"price_position_{p}"] = (close - lo) / (hi - lo + EPS)

    vol_ma = volume.rolling(20, min_periods=20).mean()
    f["volume_ratio"] = volume / (vol_ma + EPS)
    f["volume_trend"] = volume.rolling(5, min_periods=5).mean() / (vol_ma + EPS)

    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    f["atr_ratio"] = tr.rolling(14, min_periods=14).mean() / close
    return f


FEATURE_SETS: Dict[str, Callable[[PricePanel], FeatureDict]] = {
    "compact": compact_features,
    "extended": extended_features,
}


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from close[t] to close[t + horizon]; NaN where not yet realized."""
    return close.shift(-horizon) / close - 1


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, centered on 0 (range ~[-0.5, 0.5])."""
    return df.rank(axis=1, pct=True) - 0.5


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return cs_demean(df).div(df.std(axis=1) + EPS, axis=0)


TRANSFORMS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "none": lambda df: df,
    "cs_rank": cs_rank,
    "cs_demean": cs_demean,
    "cs_zscore": cs_zscore,
}


def check_no_lookahead(builder: Callable[[PricePanel], FeatureDict],
                       panel: PricePanel, cut: int, seed: int = 0) -> Dict[str, float]:
    """
    Replace every price after position ``cut`` with random values and rebuild
    the features. Returns, per feature, the max absolute change on dates up to
    ``cut``; any non-zero value means the feature peeks into the future.
    """
    rng = np.random.default_rng(seed)

    def corrupt(df):
        if df is None:
            return None
        out = df.copy()
        out.iloc[cut + 1:] = rng.uniform(1, 1000, size=out.iloc[cut + 1:].shape)
        return out

    corrupted = PricePanel(corrupt(panel.close), corrupt(panel.volume),
                           corrupt(panel.high), corrupt(panel.low))
    base, alt = builder(panel), builder(corrupted)
    out = {}
    for name in base:
        a = base[name].iloc[:cut + 1].to_numpy(dtype=float)
        b = alt[name].iloc[:cut + 1].to_numpy(dtype=float)
        if (np.isnan(a) != np.isnan(b)).any():
            out[name] = float("inf")
        else:
            diff = np.abs(a - b)
            out[name] = float(np.nanmax(diff)) if (~np.isnan(diff)).any() else 0.0
    return out
