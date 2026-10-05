"""
Walk-forward splits with an enforced purge gap.

A training row at position t carries a label built from prices up to
t + horizon. For that label to be known when the test window opens at
position ``test_start``, we need ``t + horizon <= test_start`` for every
training row, i.e. ``purge >= horizon - 1`` (rows are dropped between the
last training row and the first test row). We require ``purge >= horizon``
to stay on the safe side; splits that would leak raise instead of warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray  # integer positions into the date index
    test: np.ndarray


def walk_forward_splits(n_dates: int, train_size: int, test_size: int,
                        purge: int, horizon: int, expanding: bool = False,
                        first_test: int = 0,
                        last_test: Optional[int] = None) -> Iterator[Split]:
    """
    Yield consecutive, non-overlapping test windows of ``test_size`` dates.

    Train window: the ``train_size`` dates ending ``purge`` dates before the
    test window (or all earlier dates if ``expanding``). Only test windows
    starting at or after ``first_test`` and ending before ``last_test``
    (exclusive) are produced, which is how a hold-out period is carved out.
    """
    if purge < horizon:
        raise ValueError(f"purge ({purge}) must be >= label horizon ({horizon}) "
                         "or training labels overlap the test window")
    last_test = n_dates if last_test is None else min(last_test, n_dates)
    start = max(train_size + purge, first_test)
    while start + test_size <= last_test:
        train_end = start - purge
        train_start = 0 if expanding else train_end - train_size
        yield Split(np.arange(train_start, train_end), np.arange(start, start + test_size))
        start += test_size
