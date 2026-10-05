# Alpha Research Framework

**Leakage-checked, multiple-testing-aware evaluation of ML alpha signals — and an honest record of how easy it is to fool yourself.**

[![tests](https://github.com/BianchiGiacomo/alpha-research-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/BianchiGiacomo/alpha-research-framework/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status (v0.2): under revision.** Version 1.0 claimed that comparing two
> experiments (C6 vs C7) *proved* that ML alpha on public price data is
> overfitting. That conclusion was withdrawn: the comparison changed seven
> settings at once and misread a significantly negative IC as "no signal".
> Every issue is listed in **[docs/ERRATA.md](docs/ERRATA.md)**. The legacy
> scripts are kept unchanged in `experiments/` for transparency.

---

## What this project is now

A small Python package, `alpharesearch`, plus studies built on it. The
package exists to answer one question credibly: **would this evaluation
protocol report alpha that is not there?**

| Component | What it guarantees |
|---|---|
| `features.check_no_lookahead` | Corrupts all prices after a date and checks that no feature before it changes. |
| `cv.walk_forward_splits` | Refuses to build splits whose purge gap is shorter than the label horizon. |
| test `test_predictions_do_not_depend_on_future_prices` | End-to-end: out-of-sample scores up to day *t* are bit-identical when every price after *t* is replaced by noise. |
| `metrics.rank_ic` + `newey_west_tstat` | Daily cross-sectional Spearman IC; t-stats that account for overlapping multi-day labels. |
| `metrics.deflated_sharpe_ratio`, `probability_of_backtest_overfitting` | The number of configurations tried becomes part of the evidence (Bailey & López de Prado). |
| `backtest.run_backtest` | Costs on traded notional, holding period = label horizon, next-day timing. |
| `data.load_prices` | Explicit end date, parquet snapshot, SHA-256 of every file recorded in the results. |
| `results.save_results` | Every number in the docs comes from a generated JSON with config, data hashes and code version. |
| `synthetic.synthetic_panel` | Null panels (fat tails, volatility clustering, market factor) and panels with a planted signal of known strength. |

## Studies

### 1. Null calibration — `studies/null_calibration.py` (runs offline)

Synthetic panels where the truth is known. On **null** data any detected
alpha is a false positive; on **planted** data a good protocol should find
the signal. Full results: [results/null_calibration.md](results/null_calibration.md).

200 null panels (10 assets, 1,800 days), nominal test level 5%:

| Evaluation protocol | Reports "significant alpha" on pure noise |
|---|---|
| **Corrected**: daily rank IC, Newey-West t, pre-registered config | **5.5%** (95% CI 2.8–9.6) |
| Same IC, naive t-stat ignoring overlapping 5-day labels | 32.0% |
| Legacy C7 protocol (pooled Pearson IC per fold, t-test across folds) | 22.5% |
| Best of 12 configurations, Newey-West t | 30.0% |
| Best of 12 configurations, Deflated Sharpe > 0.95 | 1.5% |

And on the legacy headline itself: on data with **no signal at all**, the
legacy C6 and C7 specifications produce ICs of opposite sign in **33.5%** of
panels (C6 positive and C7 negative in 19%). A sign flip between them is
therefore not evidence of anything.

Power of the corrected protocol on a planted reversal signal: 20% at
mean IC ≈ 0.007, 40% at IC ≈ 0.03, 100% at IC ≈ 0.08 — i.e. with 10 assets
and ~3 years of test data, signals of realistic size (IC 0.02–0.03) are
detected less than half the time. That limit is a result too.

### 2. Cross-sectional crypto, corrected — `studies/crypto_xs_ablation.py`

The replacement for legacy C6/C7: a pre-registered baseline, one factor
varied at a time (Ridge α, feature set, transforms, horizon, training
window, purge, universe), the legacy specifications re-run under the
corrected evaluation, Deflated Sharpe and PBO across all trials, and a
hold-out period evaluated once. It needs market data, and its results will
be published in `results/crypto_xs_ablation.md` once it has been run on a
fixed data snapshot.

### Legacy v1.0 experiments — `experiments/`

| ID | Topic | Still valid? |
|---|---|---|
| E1 | Neural net (sklearn MLP, not an LSTM) overfitting one stock | Yes, as an illustration |
| E2 | Absolute vs cross-sectional targets | Illustration only |
| E3 → E4 | Look-ahead bias in vendor fundamentals | **Yes** — the most useful legacy result |
| C1–C3 | Crypto direction / technical / "production" | Superseded |
| C6 vs C7 | "Regularization reveals no alpha" | **Withdrawn** — see [ERRATA](docs/ERRATA.md) |

---

## Quick start

```bash
git clone https://github.com/BianchiGiacomo/alpha-research-framework.git
cd alpha-research-framework
pip install -e .[data,test]

pytest                                        # leakage + statistics test-suite
python run_experiments.py --study null        # null calibration (~5 min, offline)
python run_experiments.py --study crypto      # corrected crypto study (downloads data)
python run_experiments.py --study crypto --synthetic   # same pipeline, no network
```

Market data is cached in `data/cache/{ticker}_{start}_{end}.parquet`. Files
put there by hand (from any vendor, columns `date, open, high, low, close,
volume`) are used instead of downloading.

Using the package directly:

```python
from alpharesearch.data import load_prices
from alpharesearch.pipeline import StudyConfig
from alpharesearch.study import evaluate

panel = load_prices(["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"], "2020-01-01", "2026-01-01")
res = evaluate(panel, StudyConfig(ridge_alpha=100, horizon=5), first_test="2022-03-01")
print(res["ic"])          # mean rank IC, Newey-West t, hit rate, ...
print(res["long_short"])  # net/gross Sharpe, turnover, drawdown
```

## Project structure

```
alpharesearch/      core package (data, features, cv, pipeline, backtest, metrics, results)
studies/            corrected studies; each writes results/<name>.json + .md
tests/              leakage, CV, statistics and backtest tests (run in CI)
experiments/        legacy v1.0 scripts, unchanged
docs/ERRATA.md      what was wrong in v1.0 and where it is fixed
docs/METHODOLOGY.md legacy methodology notes (see errata)
results/            generated results; documented_results.json is legacy
```

## Roadmap

1. **Credible core** (this release): leakage tests, correct statistics, null calibration.
2. Run the corrected crypto study on a published data snapshot; positive controls
   with documented factors (momentum, short-term reversal, low volatility).
3. Larger survivorship-free universes; purged K-fold / CPCV; more model families
   evaluated under the same protocol.

## Disclaimer

Research code for educational purposes. Not financial advice.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{alpha_research_framework,
  title  = {Alpha Research Framework: leakage-checked, multiple-testing-aware evaluation of ML alpha signals},
  author = {Bianchi, Giacomo},
  year   = {2026},
  url    = {https://github.com/BianchiGiacomo/alpha-research-framework}
}
```
