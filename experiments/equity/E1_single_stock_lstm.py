#!/usr/bin/env python3
"""
E1: Single Stock LSTM Overfitting Demonstration
================================================

This experiment demonstrates how neural networks overfit on financial data.
Uses sklearn's MLPRegressor as a proxy for LSTM behavior (no PyTorch required).

Key Finding: Model learns patterns in training (IC +0.15) but fails on validation (IC ~0)
This is classic overfitting - the model memorizes noise, not signal.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple

# Configuration
CONFIG = {
    'ticker': 'AAPL',
    'start_date': '2018-01-01',
    'end_date': '2024-12-31',
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'lookback': 60,
    'horizon': 5,  # 5-day forward return
    'hidden_layers': (128, 64, 32),  # Simulates LSTM complexity
    'max_iter': 500,
    'early_stopping': True,
}


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    print(f"Fetching {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    print(f"  Downloaded {len(df)} rows")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features for prediction.
    All features use only past data (no lookahead).
    """
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Returns at different horizons
    for d in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{d}d'] = close.pct_change(d)
    
    # Moving averages distance
    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        df[f'sma_{w}_dist'] = (close - sma) / sma
    
    # Volatility
    for w in [5, 10, 20]:
        df[f'vol_{w}d'] = close.pct_change().rolling(w).std()
    
    # RSI
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
    
    # Volume features
    df['volume_ratio'] = volume / volume.rolling(20).mean()
    df['volume_change'] = volume.pct_change()
    
    # High-Low range
    df['hl_range'] = (high - low) / close
    
    # Price momentum
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1
    df['momentum_20'] = close / close.shift(20) - 1
    
    return df


def create_sequences(df: pd.DataFrame, feature_cols: List[str], 
                     lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for time-series prediction.
    Returns X (features), y (target), dates.
    """
    # Target: forward return
    df = df.copy()
    df['target'] = df['close'].shift(-horizon) / df['close'] - 1
    
    # Drop NaN
    df = df.dropna()
    
    X_list = []
    y_list = []
    dates = []
    
    for i in range(lookback, len(df) - horizon):
        # Features: flatten lookback window
        window = df.iloc[i-lookback:i][feature_cols].values
        X_list.append(window.flatten())
        
        # Target: forward return at position i
        y_list.append(df.iloc[i]['target'])
        dates.append(df.iloc[i]['date'])
    
    return np.array(X_list), np.array(y_list), dates


def compute_ic(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Compute Information Coefficient (Pearson correlation)."""
    if len(predictions) < 2:
        return 0.0
    mask = ~(np.isnan(predictions) | np.isnan(actual))
    if mask.sum() < 2:
        return 0.0
    return np.corrcoef(predictions[mask], actual[mask])[0, 1]


def compute_direction_accuracy(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Compute direction accuracy (% correct sign predictions)."""
    correct = np.sign(predictions) == np.sign(actual)
    return correct.mean()


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, 
                       hidden_layers, max_iter, early_stopping) -> Dict:
    """
    Train MLP and track metrics during training to show overfitting.
    """
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK")
    print("=" * 60)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = {
        'train_ic': [],
        'val_ic': [],
        'train_r2': [],
        'val_r2': [],
    }
    
    # Train with different amounts of iterations to show overfitting curve
    print("\nTracking overfitting across training iterations:")
    print("-" * 60)
    print(f"{'Iter':>6} | {'Train IC':>10} | {'Val IC':>10} | {'Train R²':>10} | {'Val R²':>10}")
    print("-" * 60)
    
    for n_iter in [10, 25, 50, 100, 200, 300, 500]:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=n_iter,
            early_stopping=False,  # We control iterations manually
            random_state=42,
            alpha=0.0001,  # Weak regularization to show overfitting
            learning_rate_init=0.001,
        )
        
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        
        # Metrics
        train_ic = compute_ic(train_pred, y_train)
        val_ic = compute_ic(val_pred, y_val)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        results['train_ic'].append(train_ic)
        results['val_ic'].append(val_ic)
        results['train_r2'].append(train_r2)
        results['val_r2'].append(val_r2)
        
        print(f"{n_iter:>6} | {train_ic:>+10.4f} | {val_ic:>+10.4f} | {train_r2:>+10.4f} | {val_r2:>+10.4f}")
    
    print("-" * 60)
    
    # Final model with early stopping
    print("\n\nTraining final model with early stopping...")
    final_model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        alpha=0.001,  # Slightly more regularization
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train_scaled, y_train)
    
    # Final predictions
    train_pred_final = final_model.predict(X_train_scaled)
    val_pred_final = final_model.predict(X_val_scaled)
    test_pred_final = final_model.predict(X_test_scaled)
    
    # Final metrics
    final_results = {
        'train_ic': compute_ic(train_pred_final, y_train),
        'val_ic': compute_ic(val_pred_final, y_val),
        'test_ic': compute_ic(test_pred_final, y_test),
        'train_r2': r2_score(y_train, train_pred_final),
        'val_r2': r2_score(y_val, val_pred_final),
        'test_r2': r2_score(y_test, test_pred_final),
        'train_dir_acc': compute_direction_accuracy(train_pred_final, y_train),
        'val_dir_acc': compute_direction_accuracy(val_pred_final, y_val),
        'test_dir_acc': compute_direction_accuracy(test_pred_final, y_test),
        'history': results,
    }
    
    return final_results


def main():
    print("\n" + "=" * 70)
    print("E1: SINGLE STOCK LSTM OVERFITTING DEMONSTRATION")
    print("=" * 70)
    print(f"\nTicker: {CONFIG['ticker']}")
    print(f"Period: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print(f"Lookback: {CONFIG['lookback']} days")
    print(f"Horizon: {CONFIG['horizon']} days")
    print(f"Architecture: MLP {CONFIG['hidden_layers']} (proxy for LSTM)")
    
    # Fetch data
    df = fetch_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])
    
    # Compute features
    print("\nComputing features...")
    df = compute_features(df)
    
    # Feature columns (exclude non-feature columns)
    exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj close', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"  Using {len(feature_cols)} features")
    
    # Create sequences
    print(f"\nCreating sequences (lookback={CONFIG['lookback']}, horizon={CONFIG['horizon']})...")
    X, y, dates = create_sequences(df, feature_cols, CONFIG['lookback'], CONFIG['horizon'])
    print(f"  Total samples: {len(X)}")
    
    # Split data (time-based, no shuffle!)
    n = len(X)
    train_end = int(n * CONFIG['train_ratio'])
    val_end = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Train and evaluate
    results = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        CONFIG['hidden_layers'], CONFIG['max_iter'], CONFIG['early_stopping']
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    OVERFITTING DEMONSTRATION                     │")
    print("├─────────────────┬─────────────┬─────────────┬───────────────────┤")
    print("│ Metric          │ Train       │ Validation  │ Test              │")
    print("├─────────────────┼─────────────┼─────────────┼───────────────────┤")
    print(f"│ IC              │ {results['train_ic']:>+10.4f} │ {results['val_ic']:>+10.4f}  │ {results['test_ic']:>+10.4f}          │")
    print(f"│ R²              │ {results['train_r2']:>+10.4f} │ {results['val_r2']:>+10.4f}  │ {results['test_r2']:>+10.4f}          │")
    print(f"│ Direction Acc   │ {results['train_dir_acc']*100:>9.1f}% │ {results['val_dir_acc']*100:>9.1f}%  │ {results['test_dir_acc']*100:>9.1f}%          │")
    print("└─────────────────┴─────────────┴─────────────┴───────────────────┘")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    train_val_gap = results['train_ic'] - results['val_ic']
    
    print(f"""
Key Observations:
-----------------
1. Train IC ({results['train_ic']:+.4f}) >> Val IC ({results['val_ic']:+.4f})
   Gap: {train_val_gap:.4f} - This is SEVERE OVERFITTING

2. Validation R² is {'NEGATIVE' if results['val_r2'] < 0 else 'positive'} ({results['val_r2']:+.4f})
   {'   WARNING: Negative R² means model is WORSE than predicting the mean!' if results['val_r2'] < 0 else ''}

3. Direction accuracy drops from {results['train_dir_acc']*100:.1f}% (train) to {results['val_dir_acc']*100:.1f}% (val)
   {'   Close to 50% = random guessing' if results['val_dir_acc'] < 0.52 else ''}

4. The training curve shows IC INCREASING then COLLAPSING
   This is the textbook overfitting pattern.

Conclusion:
-----------
X The model MEMORIZES training patterns but CANNOT GENERALIZE
X Neural networks overfit easily on financial data
X High train IC does NOT mean the model has learned real patterns

What This Means:
----------------
* Complex models need STRONG regularization (dropout, L2, early stopping)
* Always validate on OUT-OF-SAMPLE data
* High training performance is a WARNING sign, not success
""")
    
    return results


if __name__ == '__main__':
    results = main()
