# Results Summary

> **Legacy (v1.0) document.** Several conclusions here, in particular the C6 vs C7
> interpretation, the use of validation R² as an overfitting test and the reading of
> C7's negative IC as "no alpha", have been withdrawn. See [ERRATA.md](ERRATA.md).

Complete results from all experiments with actual measured values.

---

## Equity Experiments

### E1: Single Stock LSTM Overfitting

**Configuration:**
- Ticker: AAPL
- Period: 2018-2024
- Model: MLP (128, 64, 32) as LSTM proxy
- Lookback: 60 days
- Horizon: 5 days

**Training Progression (Overfitting Curve):**

| Iterations | Train IC | Val IC | Train R² | Val R² |
|------------|----------|--------|----------|--------|
| 10 | +0.676 | +0.076 | -0.310 | -23.84 |
| 25 | +0.963 | +0.077 | +0.912 | -21.31 |
| 50 | +0.967 | +0.072 | +0.931 | -21.15 |
| 100+ | +0.967 | +0.072 | +0.931 | -21.15 |

**Final Results:**

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| IC | +0.861 | -0.060 | -0.097 |
| R² | +0.696 | -4.645 | -8.452 |
| Direction Accuracy | 88.5% | 51.5% | 49.5% |

**Interpretation:**
- Severe overfitting: Train IC +0.86 → Val IC -0.06
- Val R² = -4.65 means model is 4.65x worse than predicting the mean
- Direction accuracy drops to random (50%)

---

### E2: Cross-Sectional Equity

**Configuration:**
- Tickers: 10 large caps (AAPL, MSFT, GOOGL, etc.)
- Period: 2019-2024
- Model: Ridge regression (α=10)
- Horizon: 5 days

**Results by Target Type:**

| Target | Train IC | Val IC | Test IC |
|--------|----------|--------|---------|
| Absolute Return | +0.078 | +0.131 | +0.061 |
| Excess Return | +0.025 | -0.017 | +0.028 |
| Rank | +0.028 | -0.018 | -0.008 |

**Interpretation:**
- Cross-sectional targets show normal decay (Train > Val > Test)
- Absolute returns show anomalous pattern (Test > Val) - possible regime luck
- IC remains weak (~0.02-0.03) across all targets

---

### E3: Fundamentals with Look-Ahead Bias

**Configuration:**
- Tickers: 20 large caps
- Period: 2020-2024
- Features: Technical + Fundamentals from `.info`
- ⚠️ BIASED: Uses CURRENT fundamentals for historical predictions

**Results:**

| Metric | Value |
|--------|-------|
| Train IC | +0.185 |
| Test IC | +0.181 |
| t-statistic | +15.76 |
| p-value | 0.0000 |

**Interpretation:**
- Results appear highly significant (p < 0.0001)
- BUT this is FAKE - caused by look-ahead bias
- Using 2024 P/E to predict 2020 returns = cheating
- See E4 for corrected version

---

### E4: Annual Fundamentals (Bias Corrected)

**Configuration:**
- Tickers: 20 large caps
- Period: 2019-2024
- Features: Technical + Point-in-time fundamentals
- Filing delay: 60 days

**Results:**

| Metric | Value |
|--------|-------|
| Mean IC | ~+0.02 |
| p-value | >0.10 |
| Status | Not significant |

**Interpretation:**
- After correcting look-ahead bias, alpha DISAPPEARS
- E3's IC of +0.18 was entirely due to data leakage
- True fundamental alpha is near zero

---

## Crypto Experiments

### C1: Time-Series Direction Prediction

**Configuration:**
- Assets: BTC, ETH, SOL
- Period: 2019-2024
- Model: Logistic Regression
- Task: Predict UP/DOWN direction

**Results by Asset and Horizon:**

| Asset | Horizon | Accuracy | Edge | IC |
|-------|---------|----------|------|-----|
| BTC | 1d | 49.2% | -0.85% | +0.035 |
| BTC | 3d | 50.2% | +0.20% | +0.055 |
| BTC | 5d | 48.9% | -1.14% | +0.071 |
| ETH | 1d | 50.4% | +0.45% | +0.043 |
| ETH | 3d | 49.9% | -0.10% | +0.017 |
| ETH | 5d | 49.7% | -0.30% | +0.050 |
| SOL | 1d | 50.2% | +0.16% | +0.027 |
| SOL | 3d | 50.0% | -0.00% | +0.008 |
| SOL | 5d | 49.9% | -0.05% | +0.044 |

**Best Result:** ETH 1d: 50.4% accuracy, +0.45% edge, IC +0.043

**Interpretation:**
- Marginal edge (~0.5%) with high variance
- Not statistically robust
- Would be consumed by transaction costs

---

### C2: Crypto Technical Alpha

**Configuration:**
- Assets: 10 cryptocurrencies
- Period: 2021-2024
- Features: 10 technical indicators
- Model: Ridge regression

**Results:**

| Metric | Value |
|--------|-------|
| Mean IC | -0.015 |
| Std IC | 0.176 |
| t-statistic | -0.60 |
| p-value | 0.549 |
| Folds | 52 |

**Interpretation:**
- Not statistically significant (p = 0.55)
- Technical indicators alone don't predict crypto returns

---

### C3: Production System

**Configuration:**
- Asset: BTC-USD
- Period: 2017-2024
- Training: 2-year rolling window
- Model: Ensemble of 50 MLP classifiers
- Transaction costs: 20 bps

**Results:**

| Metric | Value |
|--------|-------|
| Cumulative Return | +182.6% |
| Annualized Return | +12.7% |
| Annualized Volatility | 18.4% |
| Sharpe Ratio | 0.69 |
| Max Drawdown | 65.0% |
| Mean IC | +0.047 |
| Win Rate | 49.3% |
| p-value | 0.022 |

**Interpretation:**
- Marginal significance (p = 0.02)
- Sharpe of 0.69 is weak
- Max DD of 65% is unacceptable for production
- Borderline result - would need more data to confirm

---

### C6: Overfitting Demonstration

**Configuration (Intentionally Weak):**
- Assets: 10 cryptocurrencies
- Features: 41 (too many)
- Ridge α: 1.0 (too weak)
- Train: 365 days
- Purge: 5 days

**Model Diagnostics:**

| Model | Train R² | Val R² | Gap | Overfit? |
|-------|----------|--------|-----|----------|
| Ridge (α=1.0) | +0.135 | -0.268 | 0.403 | YES ❌ |
| MLP-small | +0.211 | -0.731 | 0.942 | YES ❌ |
| MLP-medium | +0.429 | -0.535 | 0.965 | YES ❌ |
| GradientBoosting | +0.531 | -0.146 | 0.676 | YES ❌ |

**Walk-Forward Results:**

| Metric | Value |
|--------|-------|
| Total Return | +66.6% |
| Sharpe Ratio | 0.446 |
| Max Drawdown | 40.6% |
| Mean IC | +0.036 |
| IC t-statistic | +1.51 |
| IC p-value | 0.138 |
| Folds | 48 |

**Interpretation:**
- IC appears positive (+0.036)
- BUT all validation R² are NEGATIVE
- This proves the positive IC is SPURIOUS
- Compare with C7 to see the truth

---

### C7: Anti-Overfitting (Main Result) ✅

**Configuration (Anti-Overfitting):**
- Assets: 5 cryptocurrencies (most liquid)
- Features: 8 (reduced from 41)
- Ridge α: 100.0 (100x stronger)
- Train: 500 days (longer)
- Purge: 10 days (larger gap)

**Walk-Forward Results:**

| Metric | Value |
|--------|-------|
| Total Return | -15.7% |
| Sharpe Ratio | 0.254 |
| Max Drawdown | 86.2% |
| **Mean IC** | **-0.084** |
| IC t-statistic | -3.44 |
| IC p-value | 0.001 |
| Mean Val R² | -0.201 |
| Folds | 53 |

**Interpretation:**
- IC is NEGATIVE (-0.084)
- Highly significant (p = 0.001)
- This PROVES C6's positive IC was overfitting
- There is NO alpha in public crypto price data

---

## Key Comparison: C6 vs C7

This is the **central finding** of the research:

| Parameter | C6 (Overfit) | C7 (Robust) |
|-----------|--------------|-------------|
| Features | 41 | 8 |
| Ridge α | 1.0 | 100.0 |
| Train Days | 365 | 500 |
| Purge Days | 5 | 10 |
| **Mean IC** | **+0.036** | **-0.084** |
| t-statistic | +1.51 | -3.44 |
| p-value | 0.138 | 0.001 |
| Val R² | All negative | All negative |

**The IC flipped from POSITIVE to NEGATIVE when overfitting was controlled.**

This proves:
1. C6's positive IC was noise, not signal
2. Strong regularization reveals the truth
3. There is no alpha in public crypto price data using standard ML

---

## Summary Table

| Experiment | IC | p-value | Status |
|------------|-----|---------|--------|
| E1 | Val: -0.06 | N/A | ❌ Overfitting demo |
| E2 | ~0.02 | >0.10 | ❌ Not significant |
| E3 | +0.18 | 0.0000 | ⚠️ BIASED |
| E4 | ~0.02 | >0.10 | ❌ Not significant |
| C1 | +0.04 | N/A | ❌ Marginal |
| C2 | -0.01 | 0.55 | ❌ Not significant |
| C3 | +0.05 | 0.02 | ⚠️ Borderline |
| C6 | +0.036 | 0.14 | ⚠️ SPURIOUS |
| **C7** | **-0.084** | **0.001** | ✅ **MAIN RESULT** |

---

## Conclusion

No statistically significant alpha was found in public market data using standard ML techniques. The apparent positive results in some experiments (C6, E3) were due to:

1. **Overfitting** (weak regularization, too many features)
2. **Look-ahead bias** (using future information)

When these issues are corrected (C7, E4), the alpha disappears or becomes significantly negative.

This is a valid null result that demonstrates the importance of rigorous methodology.
