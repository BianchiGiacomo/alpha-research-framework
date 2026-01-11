# Methodology: Anti-Overfitting Techniques for Financial ML

This document details the methodology used to ensure rigorous, reproducible results.

---

## 1. Walk-Forward Validation

### The Problem with Random Splits

Standard ML practice shuffles data randomly into train/test splits. This is **catastrophic** for time-series because:

1. **Temporal Leakage**: Future data "leaks" into training
2. **Auto-correlation**: Adjacent samples are not independent
3. **Non-stationarity**: Distribution changes over time

### The Solution: Walk-Forward

```
Time ─────────────────────────────────────────────────────────►

Fold 1: [=======TRAIN=======][PURGE][==TEST==]
Fold 2:      [=======TRAIN=======][PURGE][==TEST==]
Fold 3:           [=======TRAIN=======][PURGE][==TEST==]
...
```

**Rules**:
1. Train ONLY on past data
2. Test ONLY on future data
3. Never shuffle
4. Include purge gap

### Implementation

```python
def walk_forward_validation(df, train_days=500, test_days=30, purge_days=10):
    """
    Walk-forward validation with purge gap.
    
    Parameters:
    - train_days: Training window size
    - test_days: Test window size  
    - purge_days: Gap between train and test (prevents leakage)
    """
    results = []
    
    for start in range(0, len(df) - train_days - purge_days - test_days, test_days):
        train_end = start + train_days
        test_start = train_end + purge_days  # PURGE GAP
        test_end = test_start + test_days
        
        train = df.iloc[start:train_end]
        test = df.iloc[test_start:test_end]
        
        # Train model on train, evaluate on test
        model.fit(train)
        ic = evaluate(model, test)
        results.append(ic)
    
    return results
```

---

## 2. The Purge Gap

### Why It Matters

Even with walk-forward, adjacent train/test samples can be correlated:

- If we predict 5-day returns using 20-day features
- The last training sample and first test sample share 15 days of price data
- This creates subtle leakage

### Solution

Add a **purge gap** between train and test:

```
[TRAIN TRAIN TRAIN][PURGE GAP][TEST TEST TEST]
                   └─────────┘
                   No overlap allowed
```

**Recommended purge**: `max(feature_lookback, target_horizon) + buffer`

For our experiments:
- Feature lookback: 20 days
- Target horizon: 5 days
- Purge: 10 days (conservative)

---

## 3. Cross-Sectional Targets

### The Problem with Absolute Returns

Predicting absolute returns includes **market beta**:

```
AAPL return = alpha + beta * market_return + noise
```

If you predict AAPL goes up 2%, but the market goes up 3%, you underperformed!

### Solution: Cross-Sectional Targets

Rank stocks relative to each other:

```python
def compute_cross_sectional_target(df):
    """
    Convert absolute returns to cross-sectional ranks.
    """
    # Group by date
    grouped = df.groupby('date')
    
    # Excess return (vs market average)
    df['excess_ret'] = df['forward_ret'] - grouped['forward_ret'].transform('mean')
    
    # Rank (percentile within date)
    df['rank'] = grouped['forward_ret'].transform(lambda x: x.rank(pct=True))
    
    return df
```

**Benefits**:
1. Removes market trend (pure stock selection)
2. More stationary (ranks don't drift)
3. Robust to outliers (ordinal, not cardinal)

---

## 4. Feature Engineering

### The Paradox of Features

More features = more overfitting potential.

Our experiments proved this:
- C6 (41 features): IC = +0.036 (spurious)
- C7 (8 features): IC = -0.084 (truth revealed)

### Feature Selection

Use **Mutual Information** to select features:

```python
from sklearn.feature_selection import mutual_info_regression

def select_features(X, y, max_features=8):
    """
    Select top features by mutual information with target.
    """
    mi_scores = mutual_info_regression(X, y)
    top_indices = np.argsort(mi_scores)[-max_features:]
    return X[:, top_indices], top_indices
```

### Recommended Features (8 max)

| Feature | Rationale |
|---------|-----------|
| ret_5d | Short-term momentum |
| ret_20d | Medium-term momentum |
| vol_20d | Volatility regime |
| rsi_14 | Overbought/oversold |
| sma_20_dist | Mean reversion |
| volume_ratio | Volume confirmation |
| macd_hist | Trend strength |
| bb_position | Range position |

---

## 5. Regularization

### The Key Insight

C6 vs C7 comparison shows regularization reveals truth:

| Setting | C6 (Overfit) | C7 (Robust) |
|---------|--------------|-------------|
| Ridge α | 1.0 | 100.0 |
| Mean IC | +0.036 | -0.084 |

100x stronger regularization flipped the IC from positive to negative!

### Implementation

```python
from sklearn.linear_model import Ridge

# WRONG: Weak regularization (overfits)
model_weak = Ridge(alpha=1.0)

# RIGHT: Strong regularization (reveals truth)
model_strong = Ridge(alpha=100.0)
```

### How to Choose α

Use cross-validation on TRAINING data only:

```python
from sklearn.linear_model import RidgeCV

# Test multiple alphas, pick best
alphas = [1.0, 10.0, 100.0, 1000.0]
model = RidgeCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)
print(f"Best alpha: {model.alpha_}")
```

**Rule of thumb**: If best α is at the low end, your model is overfitting. Increase regularization.

---

## 6. Statistical Significance Testing

### The IC t-test

Information Coefficient (IC) is the correlation between predictions and actual returns.

To test if IC is significantly different from zero:

```python
def ic_significance_test(ics):
    """
    Test if mean IC is significantly different from zero.
    
    H0: mean IC = 0 (no predictive power)
    H1: mean IC ≠ 0 (some predictive power)
    """
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    n = len(ics)
    
    # t-statistic
    t_stat = mean_ic / (std_ic / np.sqrt(n))
    
    # p-value (two-sided)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return {
        'mean_ic': mean_ic,
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Interpretation

| p-value | Interpretation |
|---------|----------------|
| < 0.01 | Strong evidence of signal |
| 0.01-0.05 | Moderate evidence |
| 0.05-0.10 | Weak evidence |
| > 0.10 | No evidence (null result) |

**Our results**:
- C6: p = 0.14 → No evidence (despite positive IC!)
- C7: p = 0.001 → Strong evidence (that there's NO alpha!)

---

## 7. Overfitting Detection

### The Validation R² Check

**Critical insight**: Positive IC with negative validation R² = OVERFITTING.

```python
def check_overfitting(model, X_train, y_train, X_val, y_val):
    """
    Check for overfitting using R² comparison.
    """
    model.fit(X_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    val_r2 = r2_score(y_val, model.predict(X_val))
    
    gap = train_r2 - val_r2
    
    print(f"Train R²: {train_r2:+.4f}")
    print(f"Val R²:   {val_r2:+.4f}")
    print(f"Gap:      {gap:.4f}")
    
    if val_r2 < 0:
        print("⚠️  OVERFITTING: Val R² is negative!")
        print("   Model is worse than predicting the mean.")
    elif gap > 0.2:
        print("⚠️  WARNING: Large train/val gap, possible overfitting")
    else:
        print("✓ No obvious overfitting")
```

### C6 Model Diagnostics (from actual run)

```
Model              Train R²    Val R²      Gap    Overfit?
------------------------------------------------------------
Ridge (α=1.0)       +0.1354   -0.2678   0.4032      YES ❌
MLP-small           +0.2109   -0.7310   0.9420      YES ❌
MLP-medium          +0.4293   -0.5354   0.9647      YES ❌
GradientBoosting    +0.5307   -0.1455   0.6762      YES ❌
```

ALL models overfit. The positive IC later is spurious.

---

## 8. Anti-Overfitting Checklist

Before trusting any result, verify:

### Data Checks
- [ ] No future data in features (verify timestamps)
- [ ] Point-in-time fundamentals (add filing delay)
- [ ] No survivor bias (include delisted stocks)
- [ ] Train/test split is temporal (not random)

### Model Checks
- [ ] Validation R² is non-negative
- [ ] Train/Val gap < 0.2
- [ ] Feature count ≤ 10
- [ ] Regularization strength tested

### Statistical Checks
- [ ] IC p-value < 0.05
- [ ] Multiple folds (≥20)
- [ ] Consistent across time periods
- [ ] Robust to parameter changes

### Sanity Checks
- [ ] Result makes economic sense
- [ ] Transaction costs included
- [ ] Reasonable Sharpe (< 3.0)
- [ ] Not too good to be true

---

## 9. Transaction Cost Modeling

### Why It Matters

A strategy with IC = 0.02 might look profitable, but transaction costs can easily exceed the alpha:

```
Gross alpha:    +0.5% per trade
Bid-ask spread: -0.2%
Slippage:       -0.1%
Commission:     -0.05%
─────────────────────────
Net alpha:      +0.15%
```

### Implementation

```python
def compute_net_returns(signals, returns, cost_bps=20):
    """
    Compute returns after transaction costs.
    
    Parameters:
    - signals: Position signals (-1, 0, +1)
    - returns: Asset returns
    - cost_bps: Transaction cost in basis points
    """
    # Gross returns from positions
    gross_returns = signals * returns
    
    # Transaction costs when position changes
    position_changes = np.abs(np.diff(signals, prepend=0))
    costs = position_changes * (cost_bps / 10000)
    
    # Net returns
    net_returns = gross_returns - costs
    
    return net_returns
```

### Typical Costs

| Asset Class | Typical Cost (bps) |
|-------------|-------------------|
| US Large Cap Equity | 5-10 |
| US Small Cap Equity | 15-30 |
| Crypto (CEX) | 10-20 |
| Crypto (DEX) | 30-100 |

---

## 10. Reproducibility

### Random Seeds

Set all seeds for reproducibility:

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
```

### Version Tracking

Document all package versions:

```bash
pip freeze > requirements.txt
```

### Data Versioning

Save data with timestamps:

```python
import hashlib

def save_data_snapshot(df, name):
    timestamp = datetime.now().strftime('%Y%m%d')
    checksum = hashlib.md5(df.values.tobytes()).hexdigest()[:8]
    filename = f"data/{name}_{timestamp}_{checksum}.parquet"
    df.to_parquet(filename)
    return filename
```

---

## Summary

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| Walk-Forward | Prevent temporal leakage | Train past, test future |
| Purge Gap | Prevent correlation leakage | 10-day gap |
| Cross-Sectional | Remove beta | Rank targets |
| Feature Selection | Reduce overfitting | MI selection, max 8 |
| Regularization | Reveal truth | Ridge α ≥ 100 |
| IC t-test | Significance | p < 0.05 required |
| Val R² Check | Detect overfitting | Must be ≥ 0 |
| Cost Modeling | Reality check | Include 20 bps |

Follow this methodology and you'll know if your alpha is real or illusory.
