# Research Journey: From Hypothesis to Null Result

> **Legacy (v1.0) document.** Several conclusions here, in particular the C6 vs C7
> interpretation, the use of validation R² as an overfitting test and the reading of
> C7's negative IC as "no alpha", have been withdrawn. See [ERRATA.md](ERRATA.md).

This document narrates the complete research journey, from initial hypothesis to final conclusion.

---

## Phase 1: The Initial Hypothesis

### Starting Point

The project began with an ambitious goal: build a hybrid TabPFN-LSTM system for stock alpha generation. The hypothesis was that combining:

1. **TabPFN** (fast, probabilistic, zero-shot) 
2. **LSTM** (deep, sequential, trainable)
3. **Gated Fusion** (regime-adaptive weighting)

...would produce robust alpha across market conditions.

**Target metrics:**
- Sharpe Ratio ≥ 1.3
- Information Coefficient ≥ 0.05
- Max Drawdown ≤ 15%

### Reality Check

Early experiments revealed fundamental issues with this approach.

---

## Phase 2: Equity Experiments (E1-E4)

### E1: Single Stock LSTM - The Overfitting Discovery

**Goal**: Train LSTM on AAPL to predict 5-day returns.

**Result**:
```
Train IC:  +0.86  (excellent!)
Val IC:    -0.06  (disaster)
Val R²:    -4.65  (worse than predicting mean)
Direction: 88.5% → 51.5% (random)
```

**Discovery**: The model memorized training patterns completely but learned nothing generalizable. This is textbook overfitting - the neural network found patterns in noise.

**Lesson**: High training performance is a WARNING sign, not success.

---

### E2: Cross-Sectional Targets

**Goal**: Use ranking targets (excess return, percentile rank) instead of absolute returns.

**Hypothesis**: Cross-sectional targets are more stationary and should reduce overfitting.

**Result**:
```
Target Type      | Train IC | Val IC  | Test IC
-----------------|----------|---------|--------
Absolute Return  | +0.078   | +0.131  | +0.061
Excess Return    | +0.025   | -0.017  | +0.028
Rank             | +0.028   | -0.018  | -0.008
```

**Discovery**: 
- Cross-sectional targets show NORMAL decay (Train > Val > Test)
- Absolute returns show ANOMALOUS pattern (Test > Val) - luck or regime
- IC remains weak (~0.02) regardless of target type

**Lesson**: Better targets reduce overfitting but don't create alpha where none exists.

---

### E3: Fundamentals - The Look-Ahead Bias Trap

**Goal**: Add fundamental features (P/E, ROE, profit margin) from Yahoo Finance.

**Method**: Used `yfinance.Ticker().info` to get fundamentals.

**Result**:
```
Train IC:  +0.185
Test IC:   +0.181
p-value:   0.0000  ← Highly significant!
```

**Wait... this looks too good!**

**Discovery**: The `.info` API returns CURRENT values, not historical! We were using 2024 P/E ratios to predict 2020-2024 returns. This is **look-ahead bias** - the ultimate data leakage.

**The Bias Mechanism**:
- AAPL's 2024 P/E encodes that it went up 140% since 2020
- Using this for 2020 predictions = time travel cheating
- The "alpha" is fake - it's just future information leaking into the past

**Lesson**: ALWAYS verify data timestamps. The `.info` API is dangerous for backtesting.

---

### E4: Fundamentals - Bias Corrected

**Goal**: Fix E3 by using point-in-time fundamentals with filing delay.

**Method**: 
- Use `.financials` (historical annual statements)
- Add 60-day filing delay (reports aren't instant)
- Only use data that was actually available at prediction time

**Result**:
```
Mean IC:   +0.02 (weak)
p-value:   >0.10 (not significant)
```

**Discovery**: After correcting the bias, the "alpha" DISAPPEARED. The entire signal in E3 was fake.

**Lesson**: Point-in-time data is essential. Without it, backtests are worthless.

---

## Phase 3: Pivot to Crypto (C1-C3)

### Why Crypto?

After equity experiments showed limited promise, we pivoted to crypto because:

1. Higher volatility = potentially larger signals
2. 24/7 markets = more data
3. Less efficient = more opportunity (in theory)
4. No fundamentals needed = simpler features

---

### C1: Time-Series Direction Prediction

**Goal**: Predict UP/DOWN direction for BTC, ETH, SOL.

**Result**:
```
Best Result:
  Asset:    ETH-USD
  Horizon:  1 day
  Accuracy: 50.4%
  Edge:     +0.45%
  IC:       +0.043
```

**Discovery**: Marginal edge exists (~0.5%) but:
- High variance across assets and horizons
- Not statistically robust
- Would be eaten by transaction costs

**Lesson**: Direction prediction is hard. Even crypto shows minimal edge.

---

### C2: Technical Alpha

**Goal**: Use technical indicators (RSI, MACD, Bollinger, momentum) across 10 cryptos.

**Result**:
```
Mean IC:     -0.015
Std IC:      0.176
t-statistic: -0.60
p-value:     0.55
```

**Discovery**: Not statistically significant. Technical indicators alone don't predict crypto returns.

**Lesson**: Standard technical analysis doesn't work on efficient(ish) markets.

---

### C3: Production System

**Goal**: Full production-ready system with:
- 2-year rolling training window
- Transaction cost modeling (20 bps)
- Ensemble of 50 MLP classifiers
- Regime filtering

**Result**:
```
Cumulative Return: +182%
Sharpe Ratio:      0.69
Max Drawdown:      65%
Mean IC:           +0.047
p-value:           0.02
```

**Assessment**: Borderline. Sharpe of 0.69 is weak. Max DD of 65% is unacceptable. The p=0.02 is marginal significance but with high variance.

**Lesson**: Even "production-ready" systems struggle to find robust alpha.

---

## Phase 4: The Critical Discovery (C6 vs C7)

### C6: The Overfit Version

**Configuration** (intentionally weak):
- Features: 41 (many indicators)
- Ridge α: 1.0 (weak regularization)
- Train: 365 days
- Purge: 5 days

**Result**:
```
Mean IC:       +0.036
p-value:       0.14
Val R²:        ALL NEGATIVE (Ridge: -0.27, MLP: -0.73, GB: -0.15)
```

**Red Flag**: Positive IC but ALL validation R² are negative!

This means the model is worse than predicting the mean on validation data. The positive IC is a fluke - random correlation in the test folds.

---

### C7: The Robust Version

**Configuration** (anti-overfitting):
- Features: 8 (only the best via mutual information)
- Ridge α: 100.0 (100x stronger regularization)
- Train: 500 days (longer)
- Purge: 10 days (larger gap)

**Result**:
```
Mean IC:       -0.084
t-statistic:   -3.44
p-value:       0.001
Val R²:        -0.20 average
```

**THE IC FLIPPED NEGATIVE!**

### The Critical Comparison

| Parameter | C6 (Overfit) | C7 (Robust) |
|-----------|--------------|-------------|
| Features | 41 | 8 |
| Ridge α | 1.0 | 100.0 |
| Mean IC | **+0.036** | **-0.084** |
| t-statistic | +1.51 | -3.44 |
| p-value | 0.14 | 0.001 |

**Interpretation**:
1. C6's positive IC was pure overfitting
2. C7's negative IC reveals the truth: there is no alpha
3. The signal didn't just disappear - it became significantly NEGATIVE
4. This proves the model was fitting noise, not signal

---

## Phase 5: Conclusions

### The Null Result

**Finding**: There is no statistically significant alpha in public crypto/equity price data using standard ML techniques.

This is a **valid scientific result**. Not finding alpha is just as important as finding it.

### Key Learnings

1. **Overfitting is the enemy**: Without strong regularization, any model will find spurious patterns.

2. **Look-ahead bias is subtle**: Even experienced researchers make this mistake. Always verify timestamps.

3. **Validation R² matters**: Positive IC with negative R² = overfitting. This metric should be checked BEFORE trusting IC.

4. **Simple beats complex**: 8 features with strong regularization reveals more than 41 features with weak regularization.

5. **Markets are efficient-ish**: Public price data is already well-analyzed. True alpha requires either:
   - Alternative data (sentiment, satellite, etc.)
   - Superior execution (latency, market making)
   - Unique insights (fundamental research)

### What This Research Proves

The value is in the **methodology**, not the outcome:

1. Walk-forward validation works
2. Purge gaps prevent leakage
3. Feature selection beats feature engineering
4. Regularization reveals truth
5. Statistical significance testing is essential

### Recommendations for Future Research

1. **Alternative Data**: Explore non-price data (social sentiment, on-chain metrics)
2. **Higher Frequency**: Daily data may be too efficient; try intraday
3. **Cross-Asset**: Multi-asset portfolios may have diversification benefits
4. **Regime Models**: Explicit regime detection might help
5. **Execution Alpha**: Focus on execution (market making, HFT) rather than prediction

---

## Reproducibility

All experiments are fully reproducible:

```bash
# Run the key comparison
python run_experiments.py --compare

# Run all experiments
python run_experiments.py --all
```

Each script is self-contained with no external dependencies beyond standard packages.

---

## Final Thoughts

This research journey started with ambitious goals and ended with a null result. But that null result, rigorously demonstrated, is valuable:

1. **For practitioners**: Don't waste time on approaches that don't work
2. **For academics**: Honest negative results should be published
3. **For learners**: The methodology transfers to any ML project

The framework developed here - walk-forward validation, overfitting detection, statistical testing - applies far beyond finance. Use it.
