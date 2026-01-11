#!/usr/bin/env python3
"""
C2: Crypto Technical Alpha (Self-Contained)
============================================

This script tests technical indicators on multiple crypto assets.

Key Finding: IC = 0.025, p = 0.12 → NOT statistically significant.

Run: python C2_technical_alpha.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, List


# =============================================================================
# CONFIGURATION
# =============================================================================

CRYPTO_TICKERS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
    'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'LINK-USD',
]


# =============================================================================
# DATA & FEATURES
# =============================================================================

def fetch_data(tickers: List[str], start: str) -> Dict[str, pd.DataFrame]:
    """Fetch crypto data."""
    end = datetime.now().strftime('%Y-%m-%d')
    data = {}
    
    print("Fetching data...")
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df is not None and len(df) > 100:
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
                
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: FAILED")
    
    return data


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features."""
    result = df.copy()
    close = result['close']
    volume = result['volume']
    
    # Momentum
    for p in [1, 5, 10, 20]:
        result[f'ret_{p}d'] = close.pct_change(p)
    
    # Volatility
    daily_ret = close.pct_change()
    for p in [10, 20]:
        result[f'vol_{p}d'] = daily_ret.rolling(p, min_periods=1).std()
    
    # Mean reversion
    for p in [10, 20]:
        sma = close.rolling(p, min_periods=1).mean()
        result[f'sma_{p}_dist'] = (close - sma) / (sma + 1e-10)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    result['rsi_14'] = (100 - (100 / (1 + rs)) - 50) / 50
    
    # Volume
    vol_ma = volume.rolling(20, min_periods=1).mean()
    result['volume_ratio'] = volume / (vol_ma + 1)
    
    # Target
    result['future_ret_5d'] = close.pct_change(5).shift(-5)
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("C2: CRYPTO TECHNICAL ALPHA")
    print("=" * 70)
    print("\nTarget: Use technical indicators to predict 5-day returns")
    
    # Fetch data
    data = fetch_data(CRYPTO_TICKERS, '2020-01-01')
    
    # Compute features
    print("\nComputing features...")
    processed = {}
    for ticker, df in data.items():
        processed[ticker] = compute_features(df)
    
    # Get common dates
    common_dates = None
    for ticker, df in processed.items():
        dates = set(df['date'])
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = sorted(common_dates)
    
    # Feature columns
    sample_df = list(processed.values())[0]
    feature_cols = [c for c in sample_df.columns if c not in 
                   ['date', 'open', 'high', 'low', 'close', 'volume',
                    'adj close', 'future_ret_5d']]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Common dates: {len(common_dates)}")
    
    # Walk-forward
    print("\nWalk-forward validation...")
    
    train_days = 365
    test_days = 30
    
    all_ics = []
    
    for i in range(train_days, len(common_dates) - test_days, test_days):
        train_dates = common_dates[i-train_days:i]
        test_dates = common_dates[i:i+test_days]
        
        # Prepare data
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        for ticker, df in processed.items():
            train_mask = df['date'].isin(train_dates)
            test_mask = df['date'].isin(test_dates)
            
            if train_mask.sum() > 50:
                X_train.extend(df.loc[train_mask, feature_cols].values)
                y_train.extend(df.loc[train_mask, 'future_ret_5d'].values)
            
            if test_mask.sum() > 5:
                X_test.extend(df.loc[test_mask, feature_cols].values)
                y_test.extend(df.loc[test_mask, 'future_ret_5d'].values)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Filter valid
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        valid_test = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        
        if valid_train.sum() < 100 or valid_test.sum() < 20:
            continue
        
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]
        
        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        pred = model.predict(X_test_scaled)
        ic = np.corrcoef(pred, y_test)[0, 1]
        all_ics.append(ic)
        
        if len(all_ics) <= 5:
            print(f"  Fold {len(all_ics)}: IC = {ic:+.4f}")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    mean_ic = np.mean(all_ics)
    std_ic = np.std(all_ics)
    n = len(all_ics)
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    print(f"""
    Mean IC:     {mean_ic:+.4f}
    Std IC:      {std_ic:.4f}
    t-statistic: {t_stat:+.2f}
    p-value:     {p_value:.4f}
    Folds:       {n}
    
    Significant (p<0.05): {'Yes ✅' if p_value < 0.05 else 'No ❌'}
    """)
    
    if p_value > 0.05:
        print("""
    ❌ NOT STATISTICALLY SIGNIFICANT
    
    Technical indicators alone are not sufficient to generate
    statistically significant alpha on crypto assets.
    
    The IC of ~0.025 is borderline - might work with very
    low transaction costs, but not robust enough for production.
    """)
    
    print("\n  See C6 and C7 for overfitting analysis.")


if __name__ == "__main__":
    main()
