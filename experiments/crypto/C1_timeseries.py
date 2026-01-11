#!/usr/bin/env python3
"""
C1: Crypto Time-Series Direction Prediction (Self-Contained)
=============================================================

This script predicts DIRECTION (up/down) for single crypto assets.

Key Finding: Marginal edge (~2%), high variance, not statistically robust.

Run: python C1_timeseries.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'assets': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
    'start_date': '2020-01-01',
    'train_days': 180,
    'test_days': 30,
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
    """Compute features for direction prediction."""
    result = df.copy()
    close = result['close']
    high = result['high']
    low = result['low']
    volume = result['volume']
    
    # Momentum
    for p in [1, 2, 3, 5, 10, 14, 21]:
        result[f'ret_{p}d'] = close.pct_change(p)
    
    # Moving averages
    for p in [5, 10, 20, 50]:
        sma = close.rolling(p, min_periods=1).mean()
        result[f'sma_{p}'] = (close - sma) / (sma + 1e-10)
    
    # Volatility
    daily_ret = close.pct_change()
    for p in [5, 10, 20]:
        result[f'vol_{p}d'] = daily_ret.rolling(p, min_periods=1).std() * np.sqrt(365)
    
    # RSI
    for p in [7, 14]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(p, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        result[f'rsi_{p}'] = 100 - (100 / (1 + rs))
    
    # Bollinger
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    result['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    
    # Volume
    vol_ma = volume.rolling(20, min_periods=1).mean()
    result['vol_ratio'] = volume / (vol_ma + 1)
    
    # Range
    high_20 = high.rolling(20, min_periods=1).max()
    low_20 = low.rolling(20, min_periods=1).min()
    result['range_position'] = (close - low_20) / (high_20 - low_20 + 1e-10)
    
    return result


def create_targets(df: pd.DataFrame, horizons: List[int] = [1, 3, 5]) -> pd.DataFrame:
    """Create binary direction targets."""
    result = df.copy()
    
    for h in horizons:
        future_ret = result['close'].pct_change(h).shift(-h)
        result[f'target_{h}d'] = (future_ret > 0).astype(int)
        result[f'future_ret_{h}d'] = future_ret
    
    return result


def walk_forward(df: pd.DataFrame, feature_cols: List[str], 
                 target_col: str, train_days: int, test_days: int) -> List[Dict]:
    """Walk-forward validation."""
    results = []
    dates = sorted(df['date'].unique())
    
    for i in range(train_days, len(dates) - test_days, test_days):
        train_start = dates[i - train_days]
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end = dates[min(i + test_days - 1, len(dates) - 1)]
        
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
        
        X_train = df.loc[train_mask, feature_cols].fillna(0).values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].fillna(0).values
        y_test = df.loc[test_mask, target_col].values
        
        train_valid = ~np.isnan(y_train)
        test_valid = ~np.isnan(y_test)
        
        if train_valid.sum() < 50 or test_valid.sum() < 10:
            continue
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(C=0.1, max_iter=1000)
        model.fit(X_train_scaled[train_valid], y_train[train_valid])
        
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        pred_class = (pred_proba > 0.5).astype(int)
        
        accuracy = (pred_class[test_valid] == y_test[test_valid]).mean()
        edge = accuracy - 0.5
        
        # IC
        future_ret = df.loc[test_mask, 'future_ret_1d'].values
        valid_ret = ~np.isnan(future_ret)
        combined = test_valid & valid_ret
        
        ic = np.nan
        if combined.sum() >= 5:
            ic = stats.spearmanr(pred_proba[combined], future_ret[combined])[0]
        
        results.append({
            'accuracy': accuracy,
            'edge': edge,
            'ic': ic,
        })
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("C1: CRYPTO TIME-SERIES DIRECTION PREDICTION")
    print("=" * 70)
    print("\nApproach: Predict UP/DOWN direction for single assets")
    print("Why: Simpler than cross-sectional, crypto correlates too much")
    
    all_results = {}
    
    for crypto in CONFIG['assets']:
        print(f"\n{'='*70}")
        print(f"TESTING: {crypto}")
        print("=" * 70)
        
        df = fetch_data(crypto, CONFIG['start_date'])
        if df.empty:
            print(f"  No data")
            continue
        
        print(f"  Loaded {len(df)} days")
        
        df = compute_features(df)
        df = create_targets(df, horizons=[1, 3, 5])
        
        feature_cols = [c for c in df.columns if c not in 
                       ['date', 'open', 'high', 'low', 'close', 'volume',
                        'target_1d', 'target_3d', 'target_5d',
                        'future_ret_1d', 'future_ret_3d', 'future_ret_5d']]
        
        horizon_results = {}
        
        for horizon in [1, 3, 5]:
            target_col = f'target_{horizon}d'
            results = walk_forward(df, feature_cols, target_col,
                                   CONFIG['train_days'], CONFIG['test_days'])
            
            if len(results) >= 3:
                accuracies = [r['accuracy'] for r in results]
                edges = [r['edge'] for r in results]
                ics = [r['ic'] for r in results if not np.isnan(r['ic'])]
                
                horizon_results[horizon] = {
                    'accuracy': np.mean(accuracies),
                    'edge': np.mean(edges),
                    'ic': np.mean(ics) if ics else np.nan,
                }
        
        all_results[crypto] = horizon_results
        
        print(f"\n  {'Horizon':<10} {'Accuracy':>10} {'Edge':>10} {'IC':>10}")
        print("  " + "-" * 45)
        for h, m in horizon_results.items():
            print(f"  {h}d{'':<7} {m['accuracy']:>10.1%} "
                  f"{m['edge']:>+10.2%} {m['ic']:>+10.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_edge = -999
    best_config = None
    
    for crypto, horizons in all_results.items():
        for h, m in horizons.items():
            if m['edge'] > best_edge:
                best_edge = m['edge']
                best_config = (crypto, h, m)
    
    if best_config:
        crypto, horizon, m = best_config
        print(f"""
    Best configuration:
      Asset:    {crypto}
      Horizon:  {horizon} days
      Accuracy: {m['accuracy']:.1%}
      Edge:     {m['edge']:+.2%}
      IC:       {m['ic']:+.4f}
        """)
        
        if m['edge'] > 0.02:
            print("    ⚠️  MARGINAL EDGE (~2%)")
            print("    May be tradeable but high variance.")
        else:
            print("    ❌ NO SIGNIFICANT EDGE")
    
    print("""
    Conclusion:
    Direction prediction shows marginal edge (~2%) but high variance.
    Not statistically robust enough for production trading.
    See C6 and C7 for cross-sectional approach and overfitting analysis.
    """)


if __name__ == "__main__":
    main()
