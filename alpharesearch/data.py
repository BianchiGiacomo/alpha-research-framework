"""
Price panels.

A ``PricePanel`` holds wide DataFrames (index = dates, columns = assets) for
open/high/low/close/volume. Downloads are cached to parquet and every file is
hashed, so a result file can state exactly which data produced it. The end
date is always explicit: ``datetime.now()`` is never used, otherwise two runs
of the same study silently see different data.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class PricePanel:
    close: pd.DataFrame
    volume: pd.DataFrame
    high: Optional[pd.DataFrame] = None
    low: Optional[pd.DataFrame] = None
    manifest: Dict[str, str] = field(default_factory=dict)

    @property
    def assets(self) -> List[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def select(self, assets: Iterable[str]) -> "PricePanel":
        assets = list(assets)
        return PricePanel(
            close=self.close[assets],
            volume=self.volume[assets],
            high=None if self.high is None else self.high[assets],
            low=None if self.low is None else self.low[assets],
            manifest={a: h for a, h in self.manifest.items() if a in assets},
        )

    def truncate(self, end) -> "PricePanel":
        """Keep only dates strictly before ``end`` (prices after it are unknown)."""
        mask = self.close.index < pd.Timestamp(end)

        def cut(df):
            return None if df is None else df.loc[mask]

        return PricePanel(cut(self.close), cut(self.volume), cut(self.high),
                          cut(self.low), dict(self.manifest))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df[[c for c in FIELDS if c in df.columns]].sort_index()


def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily: not needed for offline work

    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError(f"no data returned for {ticker}")
    return _normalize_ohlcv(df)


def load_prices(tickers: Iterable[str], start: str, end: str,
                cache_dir: str = "data/cache", refresh: bool = False,
                min_rows: int = 200) -> PricePanel:
    """
    Load daily OHLCV for ``tickers`` between ``start`` and ``end`` (exclusive).

    Each ticker is cached as ``{cache_dir}/{ticker}_{start}_{end}.parquet``.
    Files placed there by hand (e.g. exported from another vendor) are used
    as-is, which is how to run the studies without network access.
    Tickers that fail or have fewer than ``min_rows`` rows are dropped and
    reported; the caller decides whether the remaining universe is enough.
    """
    os.makedirs(cache_dir, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    manifest: Dict[str, str] = {}
    for t in tickers:
        path = os.path.join(cache_dir, f"{t}_{start}_{end}.parquet")
        try:
            if refresh or not os.path.exists(path):
                _download_yfinance(t, start, end).to_parquet(path)
            df = _normalize_ohlcv(pd.read_parquet(path))
        except Exception as e:  # network errors, delisted tickers, ...
            print(f"  {t}: skipped ({e})")
            continue
        if len(df) < min_rows:
            print(f"  {t}: skipped ({len(df)} rows < {min_rows})")
            continue
        frames[t] = df
        manifest[t] = sha256_file(path)

    if not frames:
        raise RuntimeError("no ticker could be loaded")
    return panel_from_frames(frames, manifest)


def panel_from_frames(frames: Dict[str, pd.DataFrame],
                      manifest: Optional[Dict[str, str]] = None) -> PricePanel:
    """Align per-asset OHLCV frames on the union of their dates."""
    def wide(field_name):
        if not all(field_name in df.columns for df in frames.values()):
            return None
        return pd.DataFrame({a: df[field_name] for a, df in frames.items()}).sort_index()

    close = wide("close")
    volume = wide("volume")
    if volume is None:
        volume = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    return PricePanel(close=close, volume=volume, high=wide("high"),
                      low=wide("low"), manifest=dict(manifest or {}))
