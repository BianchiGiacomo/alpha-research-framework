# Errata: legacy experiments C6/C7 and the original conclusions

The first version of this repository (v1.0, January 2025) claimed that the
C6 → C7 comparison *proves* that ML alpha on public price data is overfitting.
That claim does not follow from the experiments. This page lists what was
wrong, so that the legacy scripts can still be read in context. The corrected
work lives in `alpharesearch/` and `studies/`; the legacy scripts in
`experiments/` are kept unchanged for transparency and are **not** used for
any conclusion any more.

## 1. Errors in reasoning

| # | Original statement | Problem |
|---|---|---|
| R1 | "C7: IC −0.084, p = 0.001 → strong evidence there is NO alpha" | A significantly negative IC is a *signal*: the inverted model would have a positive IC. A small p-value rejects "IC = 0"; it can never be evidence for "IC = 0". |
| R2 | "The IC flips from positive to negative when overfitting is controlled" | C6 → C7 changes features (41→8), Ridge α (1→100), universe (10→5 coins), start date (2021→2020), training window (365→500), purge (5→10) and label horizon (3→5 days) at once. The flip cannot be attributed to regularization. |
| R3 | "C6's positive IC was overfitting" | C6 itself had p = 0.14: there was no significant result to explain away. |
| R4 | "Positive IC with negative validation R² = overfitting" | Out-of-sample R² of return forecasts is negative for almost every model, including ones with genuinely useful ICs; R² penalizes scale and bias, which ranking strategies ignore. It is not an overfitting test. |
| R5 | "Public price data has no exploitable alpha" | Not testable with 5–10 hand-picked coins and one specification. At most: *this* specification shows no robust edge. |

## 2. Implementation issues (legacy C7, same pattern in C6)

| # | Location | Problem | Fixed in |
|---|---|---|---|
| I1 | `C7_robust_final.py:138` | Mutual-information feature selection is defined but never called, although the README lists it as a key technique. | removed from claims |
| I2 | `C7_robust_final.py:133` | Target is the raw 5-day return, not the cross-sectional target the README describes. | `pipeline.StudyConfig.target_transform` |
| I3 | `C7_robust_final.py:250` | "IC" is one Pearson correlation over all (date, asset) pairs of a fold, mixing time-series and cross-sectional variation. | `metrics.rank_ic` (Spearman per date) |
| I4 | `C7_robust_final.py:271` | Transaction cost is `weight * bps / n_days`, independent of trading. | `backtest.run_backtest` (cost on turnover) |
| I5 | `C7_robust_final.py:236` | Missing test features are replaced by 0 before scaling (not a neutral value). | rows left unscored |
| I6 | C6/C7 | Model predicts 5-day (C6: 3-day) returns but the portfolio is rebalanced daily. | holding period = horizon |
| I7 | C6/C7 | Sharpe annualized with √252 on a 365-day market. | `periods_per_year=365` |
| I8 | C6/C7 | Universe chosen with hindsight (coins that are large today). | `min_history` / `liquidity_top_n` eligibility (mitigates, does not remove) |
| I9 | all | `end=datetime.now()`: every run sees different data; nothing is cached or hashed. | `data.load_prices` (explicit end, parquet snapshot, SHA-256) |
| I10 | `results/documented_results.json`, docs | Results typed in by hand and mutually inconsistent (C6 IC +0.036/p 0.14 in README, +0.02/p ~0.33 in C7's printout). | `results.save_results` (generated files only) |
| I11 | t-tests | Overlapping multi-day labels make consecutive observations dependent; plain t-stats overstate significance. | Newey-West t (`metrics.newey_west_tstat`) |
| I12 | whole project | No account of how many configurations were tried. | DSR and PBO over all trials |

## 3. Smaller issues

- E1 is described as an LSTM; it is an sklearn `MLPRegressor` on flattened windows.
- Experiments C4 and C5 are referenced by numbering but do not exist.
- E3 → E4 (look-ahead bias in Yahoo fundamentals) is a correct and useful
  demonstration; its conclusion stands, although E4's "point-in-time" data
  still comes from a vendor that restates history.

## 4. What replaces the main result

- `studies/null_calibration.py` measures, on synthetic data with and without
  a planted signal, how often each evaluation protocol reports alpha that is
  not there, and how often it finds alpha that is. Results:
  `results/null_calibration.md`.
- `studies/crypto_xs_ablation.py` re-runs the cross-sectional crypto question
  with one-factor-at-a-time ablation, a hold-out period and multiple-testing
  corrections. It needs market data; results appear in
  `results/crypto_xs_ablation.md` once run.
