import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alpharesearch.backtest import run_backtest, target_weights
from alpharesearch.cv import walk_forward_splits
from alpharesearch.data import PricePanel
from alpharesearch.features import FEATURE_SETS, check_no_lookahead
from alpharesearch.metrics import (deflated_sharpe_ratio, ic_summary, newey_west_tstat,
                                   probabilistic_sharpe_ratio,
                                   probability_of_backtest_overfitting, rank_ic)
from alpharesearch.pipeline import StudyConfig, build_dataset, predict_walk_forward
from alpharesearch.synthetic import synthetic_panel


@pytest.fixture(scope="module")
def null_panel():
    return synthetic_panel(n_assets=8, n_days=1100, seed=3)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FEATURE_SETS))
def test_features_do_not_look_ahead(null_panel, name):
    diffs = check_no_lookahead(FEATURE_SETS[name], null_panel, cut=700)
    leaking = {k: v for k, v in diffs.items() if v > 1e-12}
    assert not leaking, f"features using future prices: {leaking}"


def test_lookahead_check_catches_a_leaky_feature(null_panel):
    def leaky(panel):
        return {"peek": panel.close.shift(-1) / panel.close}
    assert check_no_lookahead(leaky, null_panel, cut=700)["peek"] > 0


def test_predictions_do_not_depend_on_future_prices(null_panel):
    cfg = StudyConfig(train_days=300, test_days=20, purge=10, horizon=5)
    cut = 800
    corrupted = PricePanel(
        null_panel.close.copy(), null_panel.volume.copy(),
        null_panel.high.copy(), null_panel.low.copy())
    rng = np.random.default_rng(0)
    for df in (corrupted.close, corrupted.high, corrupted.low, corrupted.volume):
        df.iloc[cut + 1:] = rng.uniform(1, 1000, size=df.iloc[cut + 1:].shape)

    s1, _ = predict_walk_forward(build_dataset(null_panel, cfg), cfg)
    s2, _ = predict_walk_forward(build_dataset(corrupted, cfg), cfg)
    pd.testing.assert_frame_equal(s1.iloc[:cut + 1], s2.iloc[:cut + 1])
    assert s1.iloc[:cut + 1].notna().any().any()


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def test_purge_shorter_than_horizon_is_rejected():
    with pytest.raises(ValueError):
        list(walk_forward_splits(1000, 300, 30, purge=3, horizon=5))


def test_splits_are_ordered_purged_and_non_overlapping():
    splits = list(walk_forward_splits(1000, 300, 30, purge=10, horizon=5))
    assert len(splits) > 10
    for s in splits:
        assert s.train.max() + 5 < s.test.min()
        assert len(s.train) == 300
    tests = np.concatenate([s.test for s in splits])
    assert len(tests) == len(np.unique(tests))


def test_hold_out_window_is_respected():
    in_sample = list(walk_forward_splits(1000, 300, 30, 10, 5, last_test=800))
    hold_out = list(walk_forward_splits(1000, 300, 30, 10, 5, first_test=800))
    assert max(s.test.max() for s in in_sample) < 800
    assert min(s.test.min() for s in hold_out) == 800


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_rank_ic_matches_scipy_spearman():
    rng = np.random.default_rng(1)
    s = pd.DataFrame(rng.normal(size=(50, 7)))
    f = pd.DataFrame(rng.normal(size=(50, 7)))
    s.iloc[3, 2] = np.nan
    ic = rank_ic(s, f)
    for d in [0, 3, 17]:
        ok = s.iloc[d].notna()
        expected = stats.spearmanr(s.iloc[d][ok], f.iloc[d][ok])[0]
        assert ic.loc[d] == pytest.approx(expected)


def test_newey_west_without_lags_is_the_plain_t_stat():
    x = np.random.default_rng(2).normal(0.1, 1, 500)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, lags=0) == pytest.approx(plain)


def test_newey_west_deflates_overlapping_observations():
    # 5-day sums of white noise: the overlap that daily ICs on 5-day labels have.
    e = np.random.default_rng(4).normal(size=5000)
    overlapping = np.convolve(e, np.ones(5), "valid") + 0.02
    s = ic_summary(pd.Series(overlapping), horizon=5)
    assert abs(s["t_nw"]) < abs(s["t_naive"]) / 1.5


def test_psr_and_dsr():
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.01, 1000)
    psr = probabilistic_sharpe_ratio(r)
    trials = rng.normal(0, 0.05, 200)  # per-period Sharpes of 200 tried configurations
    assert 0.5 < psr < 1
    assert deflated_sharpe_ratio(r, trials) < psr
    assert probabilistic_sharpe_ratio(r, sr_benchmark=r.mean() / r.std(ddof=1)) == pytest.approx(0.5)


def test_pbo_detects_noise_and_skill():
    # On pure noise the best in-sample configuration is a coin flip out of
    # sample: PBO ~ 0.5 on average (single estimates are very dispersed).
    pbos = [probability_of_backtest_overfitting(
        np.random.default_rng(s).normal(0, 0.01, size=(800, 20)), 10)["pbo"] for s in range(20)]
    assert 0.4 < np.mean(pbos) < 0.6
    noise = np.random.default_rng(6).normal(0, 0.01, size=(1600, 20))
    skill = noise.copy()
    skill[:, 0] += 0.003
    assert probability_of_backtest_overfitting(skill, 12)["pbo"] < 0.1


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

def test_long_short_weights_are_dollar_neutral():
    scores = pd.DataFrame(np.random.default_rng(7).normal(size=(10, 6)))
    w = target_weights(scores, top_k=2)
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.abs().sum(axis=1), 2)


def test_costs_follow_turnover_and_timing_is_next_day():
    dates = pd.date_range("2020-01-01", periods=11)
    # Asset A rises 1%/day, B falls 1%/day; constant scores -> one trade only.
    close = pd.DataFrame({"A": 100 * 1.01 ** np.arange(11), "B": 100 * 0.99 ** np.arange(11)},
                         index=dates)
    scores = pd.DataFrame({"A": 1.0, "B": -1.0}, index=dates)
    bt = run_backtest(scores, close, top_k=1, rebalance_every=1, cost_bps=100)
    r = bt["returns"]
    assert r.iloc[0] == pytest.approx(0.02 - 2 * 0.01)  # 2 units traded at 100 bps
    assert np.allclose(r.iloc[1:], 0.02)
    assert len(r) == 10


def test_pipeline_is_quiet_on_null_and_finds_planted_signal(null_panel):
    cfg = StudyConfig()
    planted = synthetic_panel(n_assets=8, n_days=1100, seed=3, signal_strength=0.15)
    t = {}
    for name, panel in [("null", null_panel), ("planted", planted)]:
        ds = build_dataset(panel, cfg)
        scores, _ = predict_walk_forward(ds, cfg)
        t[name] = ic_summary(rank_ic(scores, ds.fwd), cfg.horizon)["t_nw"]
    assert abs(t["null"]) < 3
    assert t["planted"] > 3
