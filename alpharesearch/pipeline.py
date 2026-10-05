"""
Out-of-sample prediction panels.

``build_dataset`` turns a price panel into aligned feature / target arrays;
``predict_walk_forward`` fits one model per walk-forward split on pooled
(date, asset) rows and returns a wide panel of out-of-sample scores. Rows with
missing features are left without a prediction (never imputed with 0), and
every preprocessing step is fitted on the training window only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .cv import walk_forward_splits
from .data import PricePanel
from .features import FEATURE_SETS, TRANSFORMS, forward_returns


@dataclass(frozen=True)
class StudyConfig:
    feature_set: str = "compact"        # see features.FEATURE_SETS
    feature_transform: str = "cs_rank"  # see features.TRANSFORMS
    target_transform: str = "cs_rank"   # cross-sectional target: no market beta
    horizon: int = 5                    # label horizon in days
    ridge_alpha: float = 100.0
    train_days: int = 500
    test_days: int = 30
    purge: int = 10
    expanding: bool = False
    min_history: int = 60               # days an asset must exist before it is tradable
    liquidity_top_n: Optional[int] = None  # keep only the N most liquid assets per date

    def with_(self, **kw) -> "StudyConfig":
        return replace(self, **kw)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Dataset:
    dates: pd.DatetimeIndex
    assets: List[str]
    X: np.ndarray            # (T, N, F) transformed features
    y: np.ndarray            # (T, N) transformed target used for fitting
    fwd: pd.DataFrame        # (T, N) raw forward returns, for evaluation
    eligible: pd.DataFrame   # (T, N) bool tradability mask
    feature_names: List[str]


def eligibility_mask(panel: PricePanel, min_history: int,
                     liquidity_top_n: Optional[int] = None) -> pd.DataFrame:
    """
    An asset is eligible on date t if it has at least ``min_history`` prior
    observations and, optionally, ranks in the top N by trailing 30-day median
    dollar volume. This uses only information available at t and limits (but
    cannot remove) the survivorship bias of a hand-picked ticker list.
    """
    close = panel.close
    history = close.notna().cumsum()
    mask = (history >= min_history) & close.notna()
    if liquidity_top_n is not None:
        dv = (close * panel.volume).rolling(30, min_periods=10).median()
        rank = dv.where(mask).rank(axis=1, ascending=False)
        mask &= rank <= liquidity_top_n
    return mask


def build_dataset(panel: PricePanel, cfg: StudyConfig) -> Dataset:
    eligible = eligibility_mask(panel, cfg.min_history, cfg.liquidity_top_n)
    feats = FEATURE_SETS[cfg.feature_set](panel)
    tf = TRANSFORMS[cfg.feature_transform]
    names = sorted(feats)
    X = np.stack([tf(feats[n].where(eligible)).to_numpy(dtype=float) for n in names], axis=-1)
    X[~np.isfinite(X)] = np.nan

    fwd = forward_returns(panel.close, cfg.horizon).where(eligible)
    y = TRANSFORMS[cfg.target_transform](fwd).to_numpy(dtype=float)
    return Dataset(panel.dates, panel.assets, X, y, fwd, eligible, names)


def _rows(ds: Dataset, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = ds.X[idx].reshape(-1, ds.X.shape[-1])
    y = ds.y[idx].reshape(-1)
    return X, y


def predict_walk_forward(ds: Dataset, cfg: StudyConfig, first_test: int = 0,
                         last_test: Optional[int] = None,
                         min_train_rows: int = 200) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Returns ``(scores, folds)``: a (dates x assets) frame of out-of-sample
    scores (NaN outside test windows) and per-fold diagnostics.
    """
    T, N, _ = ds.X.shape
    scores = np.full((T, N), np.nan)
    folds: List[Dict] = []

    for split in walk_forward_splits(T, cfg.train_days, cfg.test_days, cfg.purge,
                                     cfg.horizon, cfg.expanding, first_test, last_test):
        Xtr, ytr = _rows(ds, split.train)
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        if ok.sum() < min_train_rows:
            continue
        scaler = StandardScaler().fit(Xtr[ok])
        model = Ridge(alpha=cfg.ridge_alpha).fit(scaler.transform(Xtr[ok]), ytr[ok])

        Xte = ds.X[split.test].reshape(-1, ds.X.shape[-1])
        good = np.isfinite(Xte).all(axis=1)
        pred = np.full(len(Xte), np.nan)
        if good.any():
            pred[good] = model.predict(scaler.transform(Xte[good]))
        scores[split.test] = pred.reshape(len(split.test), N)

        folds.append({
            "train_start": str(ds.dates[split.train[0]].date()),
            "test_start": str(ds.dates[split.test[0]].date()),
            "test_end": str(ds.dates[split.test[-1]].date()),
            "n_train_rows": int(ok.sum()),
            "coef": dict(zip(ds.feature_names, map(float, model.coef_))),
        })

    return pd.DataFrame(scores, index=ds.dates, columns=ds.assets), folds
