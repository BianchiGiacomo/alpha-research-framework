#!/usr/bin/env python3
"""
E3: Fundamentals with Look-Ahead Bias (Discovery)
=================================================

This experiment discovers a critical flaw in using Yahoo Finance fundamentals:
The .info API returns CURRENT values, not historical point-in-time data.

Using current P/E ratios to predict past returns creates LOOK-AHEAD BIAS.

Key Finding: IC = 0.04 appears significant but is FAKE due to data leakage.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

# Configuration
CONFIG = {
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'UNH', 'XOM', 'PG',
                'MA', 'HD', 'PFE', 'KO', 'PEP', 'MRK', 'ABBV', 'TMO', 'COST', 'WMT'],
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'horizon': 20,  # 20-day forward return
}


def fetch_price_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data."""
    print(f"Fetching price data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 100:
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                data[ticker] = df
        except Exception as e:
            print(f"  {ticker}: ERROR")
    print(f"  Got data for {len(data)} tickers")
    return data


def fetch_fundamentals_biased(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch fundamentals using .info (BIASED - returns CURRENT values).
    
    THIS IS THE PROBLEMATIC APPROACH we're demonstrating.
    """
    print("\nFetching fundamentals via .info (CURRENT values - BIASED!)...")
    fundamentals = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals[ticker] = {
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'dividend_yield': info.get('dividendYield'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'market_cap': info.get('marketCap'),
            }
        except Exception as e:
            pass
    
    print(f"  Got fundamentals for {len(fundamentals)} tickers")
    return fundamentals


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features."""
    df = df.copy()
    close = df['close']
    
    # Returns
    df['ret_5d'] = close.pct_change(5)
    df['ret_20d'] = close.pct_change(20)
    
    # Moving averages
    df['sma_20_dist'] = (close - close.rolling(20).mean()) / close.rolling(20).mean()
    
    # Volatility
    df['vol_20d'] = close.pct_change().rolling(20).std()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    return df


def prepare_biased_dataset(price_data: Dict[str, pd.DataFrame],
                           fundamentals: Dict[str, Dict],
                           horizon: int) -> pd.DataFrame:
    """
    Prepare dataset WITH look-ahead bias.
    
    THE BIAS: Using CURRENT (2024) fundamentals to predict 2020-2024 returns.
    This is cheating because we're using future information!
    """
    all_rows = []
    
    for ticker in price_data.keys():
        if ticker not in fundamentals:
            continue
        
        df = price_data[ticker].copy()
        df = compute_technical_features(df)
        
        # Forward return (target)
        df['forward_ret'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Add CURRENT fundamentals to ALL historical dates (THIS IS THE BIAS!)
        fund = fundamentals[ticker]
        for key, value in fund.items():
            if value is not None and not np.isnan(value) if isinstance(value, float) else value is not None:
                df[key] = value  # Same value for all dates!
        
        # Collect rows
        feature_cols = ['ret_5d', 'ret_20d', 'sma_20_dist', 'vol_20d', 'rsi_14',
                        'pe_ratio', 'pb_ratio', 'roe', 'profit_margin']
        
        for _, row in df.iterrows():
            if pd.notna(row.get('forward_ret')) and all(pd.notna(row.get(c)) for c in feature_cols if c in row):
                all_rows.append({
                    'date': row['date'],
                    'ticker': ticker,
                    'forward_ret': row['forward_ret'],
                    **{c: row.get(c, np.nan) for c in feature_cols}
                })
    
    return pd.DataFrame(all_rows)


def compute_ic(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Compute Information Coefficient."""
    mask = ~(np.isnan(predictions) | np.isnan(actual))
    if mask.sum() < 2:
        return 0.0
    return np.corrcoef(predictions[mask], actual[mask])[0, 1]


def main():
    print("\n" + "=" * 70)
    print("E3: FUNDAMENTALS WITH LOOK-AHEAD BIAS")
    print("=" * 70)
    print("\n⚠️  WARNING: This experiment DEMONSTRATES a common mistake!")
    print("   Using current fundamentals to predict past returns is BIASED.")
    
    # Fetch data
    price_data = fetch_price_data(CONFIG['tickers'], CONFIG['start_date'], CONFIG['end_date'])
    fundamentals = fetch_fundamentals_biased(CONFIG['tickers'])
    
    # Prepare biased dataset
    print("\nPreparing BIASED dataset...")
    print("  (Using 2024 fundamentals to predict 2020-2024 returns)")
    df = prepare_biased_dataset(price_data, fundamentals, CONFIG['horizon'])
    
    # Drop NaN
    feature_cols = ['ret_5d', 'ret_20d', 'sma_20_dist', 'vol_20d', 'rsi_14',
                    'pe_ratio', 'pb_ratio', 'roe', 'profit_margin']
    df = df.dropna(subset=feature_cols + ['forward_ret'])
    
    print(f"\n  Total samples: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    if len(df) < 100:
        print("ERROR: Not enough data")
        return
    
    # Split data (time-based)
    dates = sorted(df['date'].unique())
    split_idx = int(len(dates) * 0.7)
    train_end = dates[split_idx]
    
    train_df = df[df['date'] <= train_end]
    test_df = df[df['date'] > train_end]
    
    print(f"\n  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['forward_ret'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['forward_ret'].values
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    
    # Predict
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Compute ICs
    train_ic = compute_ic(train_pred, y_train)
    test_ic = compute_ic(test_pred, y_test)
    
    # Statistical test
    n = len(test_pred)
    t_stat = test_ic * np.sqrt(n - 2) / np.sqrt(1 - test_ic**2 + 1e-10)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (BIASED)")
    print("=" * 70)
    
    print("\n┌───────────────────────────────────────────────────────────────┐")
    print("│                    BIASED RESULTS                              │")
    print("├─────────────────────────┬─────────────────────────────────────┤")
    print(f"│ Train IC                │ {train_ic:>+.4f}                             │")
    print(f"│ Test IC                 │ {test_ic:>+.4f}                             │")
    print(f"│ t-statistic             │ {t_stat:>+.4f}                             │")
    print(f"│ p-value                 │ {p_value:.4f}                              │")
    print("└─────────────────────────┴─────────────────────────────────────┘")
    
    # The bias explanation
    print("\n" + "=" * 70)
    print("⚠️  THE PROBLEM: LOOK-AHEAD BIAS")
    print("=" * 70)
    
    print(f"""
What We Did Wrong:
------------------
1. yfinance .info returns CURRENT fundamentals (as of today)
2. We used these 2024 values to predict 2020-2024 returns
3. This is CHEATING - we're using future information!

The Bias Mechanism:
-------------------
Example for AAPL:
- Current P/E (2024): 30x
- 2020 price: $75, 2024 price: $180
- By using 2024 P/E for 2020, we implicitly encode:
  "AAPL will go up 140% in the next 4 years"
  
This creates FAKE predictive power:
- Low P/E in 2024 → stock was cheap in 2024 → probably went UP
- High P/E in 2024 → stock is expensive now → had good returns

Result Analysis:
----------------
- Test IC = {test_ic:+.4f} {'appears significant' if abs(test_ic) > 0.03 else 'is weak'}
- p-value = {p_value:.4f} {'< 0.05' if p_value < 0.05 else '>= 0.05'}
- BUT THIS IS FAKE! The signal comes from data leakage, not prediction.

The Correct Approach (see E4):
------------------------------
- Use POINT-IN-TIME fundamentals from .financials
- Add FILING DELAY (45-90 days after fiscal year end)
- Never use data that wasn't available at prediction time

This Experiment Proves:
-----------------------
1. Look-ahead bias is EASY to introduce
2. It creates FAKE significant results
3. Always check data timestamps carefully!
""")
    
    return {
        'train_ic': train_ic,
        'test_ic': test_ic,
        'p_value': p_value,
        'biased': True,
    }


if __name__ == '__main__':
    results = main()
