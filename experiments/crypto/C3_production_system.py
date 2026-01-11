#!/usr/bin/env python3
"""
C3: Production System - Hybrid Alpha (Self-Contained)
======================================================

This is a more comprehensive production-ready system that includes:
- Longer training window (2 years)
- Transaction costs
- Risk-adjusted position sizing
- Full equity curve simulation

Key Finding: Sharpe 0.71, p = 0.24 → NOT statistically significant.

Run: python C3_production_system.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from typing import Dict, List, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'ticker': 'BTC-USD',
    'start_date': '2018-01-01',
    'train_days': 730,      # 2 years
    'val_days': 180,        # 6 months
    'test_days': 30,        # 1 month
    'transaction_cost_bps': 10,
    'target_horizon': 5,
}


# =============================================================================
# DATA & FEATURES
# =============================================================================

def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    """Fetch crypto data."""
    end = datetime.now().strftime('%Y-%m-%d')
    
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [str(c).lower() for c in df.columns]
        df = df.reset_index()
        
        if 'date' not in df.columns:
            for col in df.columns:
                if 'date' in str(col).lower():
                    df = df.rename(columns={col: 'date'})
                    break
        
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive features."""
    result = df.copy()
    close = result['close']
    high = result['high']
    low = result['low']
    volume = result['volume']
    
    # Momentum
    for p in [1, 3, 5, 10, 20, 30]:
        result[f'ret_{p}d'] = close.pct_change(p)
    
    # Volatility
    daily_ret = close.pct_change()
    for p in [5, 10, 20, 30]:
        result[f'vol_{p}d'] = daily_ret.rolling(p, min_periods=1).std()
    
    # Volatility ratio
    result['vol_ratio'] = result['vol_5d'] / (result['vol_20d'] + 1e-10)
    
    # Mean reversion
    for p in [5, 10, 20, 50]:
        sma = close.rolling(p, min_periods=1).mean()
        result[f'sma_{p}_dist'] = (close - sma) / (sma + 1e-10)
    
    # RSI
    for p in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(p, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        result[f'rsi_{p}'] = (100 - (100 / (1 + rs)) - 50) / 50
    
    # Bollinger
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    result['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    result['bb_width'] = (4 * bb_std) / (bb_mid + 1e-10)
    
    # Volume
    vol_ma = volume.rolling(20, min_periods=1).mean()
    result['volume_ratio'] = volume / (vol_ma + 1)
    result['volume_trend'] = volume.rolling(5, min_periods=1).mean() / (vol_ma + 1)
    
    # Range
    high_20 = high.rolling(20, min_periods=1).max()
    low_20 = low.rolling(20, min_periods=1).min()
    result['range_position'] = (close - low_20) / (high_20 - low_20 + 1e-10)
    
    # Target (classification: up/down)
    future_ret = close.pct_change(5).shift(-5)
    result['target'] = (future_ret > 0).astype(int)
    result['future_ret'] = future_ret
    
    return result


def create_sequences(features: np.ndarray, targets: np.ndarray,
                     seq_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time-series model."""
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length].flatten())  # Flatten for MLP
        y.append(targets[i+seq_length])
    return np.array(X), np.array(y)


# =============================================================================
# BACKTESTING
# =============================================================================

def walk_forward_backtest(df: pd.DataFrame, feature_cols: List[str],
                          config: Dict) -> Dict:
    """Run walk-forward backtest with full equity curve."""
    
    dates = sorted(df['date'].unique())
    train_days = config['train_days']
    test_days = config['test_days']
    
    all_returns = []
    all_predictions = []
    all_actuals = []
    fold_results = []
    
    print("\nRunning walk-forward backtest...")
    
    for i in range(train_days, len(dates) - test_days, test_days):
        train_dates = dates[:i]
        test_dates = dates[i:i+test_days]
        
        train_df = df[df['date'].isin(train_dates)].copy()
        test_df = df[df['date'].isin(test_dates)].copy()
        
        if len(train_df) < 200 or len(test_df) < 10:
            continue
        
        # Prepare data
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        future_ret_test = test_df['future_ret'].values
        
        # Filter valid
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        valid_test = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        
        if valid_train.sum() < 100 or valid_test.sum() < 5:
            continue
        
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test = X_test[valid_test]
        y_test_valid = y_test[valid_test]
        future_ret_valid = future_ret_test[valid_test]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train MLP classifier
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        pred_class = (pred_proba > 0.5).astype(int)
        
        # Calculate IC
        valid_ret = ~np.isnan(future_ret_valid)
        if valid_ret.sum() >= 5:
            ic = np.corrcoef(pred_proba[valid_ret], future_ret_valid[valid_ret])[0, 1]
        else:
            ic = np.nan
        
        all_predictions.extend(pred_proba[valid_ret])
        all_actuals.extend(future_ret_valid[valid_ret])
        
        # Calculate returns
        for j in range(len(pred_proba)):
            if np.isnan(future_ret_valid[j]):
                continue
            
            # Position based on prediction
            if pred_proba[j] > 0.5:
                position = 1  # Long
            else:
                position = 0  # Cash
            
            ret = position * future_ret_valid[j] / 5  # Daily portion of 5-day return
            cost = config['transaction_cost_bps'] / 10000 if position > 0 else 0
            net_ret = ret - cost
            
            all_returns.append(net_ret)
        
        # Accuracy
        accuracy = (pred_class == y_test_valid).mean()
        
        fold_results.append({
            'date': str(test_dates[0])[:10],
            'accuracy': accuracy,
            'ic': ic,
        })
        
        if len(fold_results) <= 5 or len(fold_results) % 10 == 0:
            print(f"  Fold {len(fold_results)}: Acc={accuracy:.1%}, IC={ic:+.4f}")
    
    # Calculate statistics
    returns = np.array(all_returns)
    
    # Cumulative return
    cum_returns = np.cumprod(1 + returns)
    total_return = cum_returns[-1] - 1
    
    # Sharpe
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # IC
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    valid = ~np.isnan(all_actuals)
    mean_ic = np.corrcoef(all_predictions[valid], all_actuals[valid])[0, 1]
    
    # Max drawdown
    peak = np.maximum.accumulate(cum_returns)
    max_dd = np.max((peak - cum_returns) / peak)
    
    # Win rate
    winning_folds = sum(1 for f in fold_results if f['accuracy'] > 0.5)
    win_rate = winning_folds / len(fold_results) if fold_results else 0
    
    # Bootstrap for p-value
    n_bootstrap = 1000
    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, len(returns), len(returns))
        sample = returns[idx]
        if np.std(sample) > 0:
            bs = np.mean(sample) / np.std(sample) * np.sqrt(252)
            bootstrap_sharpes.append(bs)
    
    prob_positive = (np.array(bootstrap_sharpes) > 0).mean()
    p_value = 1 - prob_positive
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'mean_ic': mean_ic,
        'win_rate': win_rate,
        'n_folds': len(fold_results),
        'p_value': p_value,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("C3: PRODUCTION SYSTEM - HYBRID ALPHA")
    print("=" * 70)
    print("\nFull production-ready system with:")
    print("  - 2-year training window")
    print("  - Transaction cost modeling")
    print("  - Walk-forward validation")
    print("  - Equity curve simulation")
    
    # Fetch data
    print(f"\n1. Fetching {CONFIG['ticker']} data...")
    df = fetch_data(CONFIG['ticker'], CONFIG['start_date'])
    if df.empty:
        print("Error: Could not fetch data")
        return
    print(f"   Loaded {len(df)} days")
    
    # Compute features
    print("\n2. Computing features...")
    df = compute_features(df)
    
    feature_cols = [c for c in df.columns if c not in 
                   ['date', 'open', 'high', 'low', 'close', 'volume',
                    'adj close', 'target', 'future_ret']]
    print(f"   Features: {len(feature_cols)}")
    
    # Backtest
    print("\n3. Running backtest...")
    results = walk_forward_backtest(df, feature_cols, CONFIG)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"""
    PERFORMANCE:
      Cumulative Return:  {results['total_return']:+.2%}
      Annualized Return:  {results['ann_return']:+.2%}
      Annualized Vol:     {results['ann_vol']:.2%}
      Sharpe Ratio:       {results['sharpe']:.2f}
      Max Drawdown:       {results['max_dd']:.2%}
    
    SIGNAL QUALITY:
      Mean IC:            {results['mean_ic']:+.4f}
      Win Rate:           {results['win_rate']:.1%}
      Number of folds:    {results['n_folds']}
    
    STATISTICAL SIGNIFICANCE:
      p-value:            {results['p_value']:.4f}
      Significant (p<0.05): {'Yes ✅' if results['p_value'] < 0.05 else 'No ❌'}
    """)
    
    # Conclusion
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if results['p_value'] > 0.05:
        print(f"""
    ❌ NOT STATISTICALLY SIGNIFICANT
    
    Despite positive cumulative return ({results['total_return']:+.2%}),
    the p-value of {results['p_value']:.2f} indicates the result is not
    statistically significant.
    
    The Sharpe of {results['sharpe']:.2f} is below the 1.3 target.
    The max drawdown of {results['max_dd']:.1%} is too high.
    
    This could be due to:
    1. Insufficient edge in technical features
    2. High transaction costs eating into returns
    3. Model overfitting on training data
    
    See C6 and C7 for overfitting analysis.
        """)
    else:
        print(f"""
    ⚠️  MARGINALLY SIGNIFICANT
    
    The results are borderline. Would need more data or
    longer backtest period to confirm.
        """)


if __name__ == "__main__":
    main()
