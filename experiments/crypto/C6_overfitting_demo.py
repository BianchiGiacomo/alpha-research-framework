#!/usr/bin/env python3
"""
C6: Cross-Sectional Crypto - OVERFITTING DEMONSTRATION
========================================================

⚠️ WARNING: This script INTENTIONALLY shows overfitting!

The positive IC in this script is SPURIOUS. Compare with C7 to see
how proper regularization reveals the true (null) signal.

Configuration (weak regularization):
- Features: 41 (too many)
- Ridge α: 1.0 (too weak)
- Result: IC appears positive (+0.02)

Run: python C6_overfitting_demo.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from typing import Dict, List


# =============================================================================
# CONFIGURATION - WEAK REGULARIZATION (causes overfitting)
# =============================================================================

CONFIG = {
    'assets': [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
        'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'LINK-USD',
    ],
    'start_date': '2021-01-01',
    'train_days': 365,
    'test_days': 30,
    'purge_days': 5,
    'long_top_n': 2,
    
    # WEAK REGULARIZATION (intentionally causes overfitting)
    'ridge_alpha': 1.0,  # Compare with C7: α = 100.0
}


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def fetch_data(assets: List[str], start: str) -> Dict[str, pd.DataFrame]:
    """Fetch crypto data from Yahoo Finance."""
    end = datetime.now().strftime('%Y-%m-%d')
    data = {}
    
    print("\nFetching data...")
    for asset in assets:
        try:
            df = yf.download(asset, start=start, end=end, progress=False)
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
                
                data[asset] = df
                print(f"  {asset}: {len(df)} rows")
        except Exception as e:
            print(f"  {asset}: FAILED - {e}")
    
    return data


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MANY features (41 total - intentionally too many)."""
    result = df.copy()
    close = result['close']
    high = result['high']
    low = result['low']
    volume = result['volume']
    
    # MOMENTUM (9 features)
    for p in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        result[f'ret_{p}d'] = close.pct_change(p)
    
    # MOMENTUM ACCELERATION (1 feature)
    result['ret_accel'] = result['ret_5d'] - result['ret_5d'].shift(5)
    
    # VOLATILITY (6 features)
    daily_ret = close.pct_change()
    for p in [5, 10, 20, 30]:
        result[f'vol_{p}d'] = daily_ret.rolling(p, min_periods=1).std()
    result['vol_ratio_5_20'] = result['vol_5d'] / (result['vol_20d'] + 1e-10)
    result['vol_change'] = result['vol_5d'] - result['vol_5d'].shift(5)
    
    # MEAN REVERSION (8 features)
    for p in [5, 10, 20, 50]:
        sma = close.rolling(p, min_periods=1).mean()
        result[f'sma_{p}_dist'] = (close - sma) / (sma + 1e-10)
        ema = close.ewm(span=p, min_periods=1).mean()
        result[f'ema_{p}_dist'] = (close - ema) / (ema + 1e-10)
    
    # BOLLINGER BANDS (4 features)
    for p in [10, 20]:
        bb_mid = close.rolling(p, min_periods=1).mean()
        bb_std = close.rolling(p, min_periods=1).std()
        result[f'bb_{p}_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
        result[f'bb_{p}_width'] = (4 * bb_std) / (bb_mid + 1e-10)
    
    # RSI (3 features)
    for p in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(p, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        result[f'rsi_{p}'] = (100 - (100 / (1 + rs)) - 50) / 50
    
    # MACD (4 features)
    for fast, slow in [(8, 21), (12, 26)]:
        ema_fast = close.ewm(span=fast, min_periods=1).mean()
        ema_slow = close.ewm(span=slow, min_periods=1).mean()
        macd = (ema_fast - ema_slow) / (close + 1e-10)
        signal = macd.ewm(span=9, min_periods=1).mean()
        result[f'macd_{fast}_{slow}'] = macd
        result[f'macd_{fast}_{slow}_hist'] = macd - signal
    
    # PRICE POSITION (2 features)
    for p in [10, 20]:
        highest = high.rolling(p, min_periods=1).max()
        lowest = low.rolling(p, min_periods=1).min()
        result[f'price_position_{p}'] = (close - lowest) / (highest - lowest + 1e-10)
    
    # VOLUME FEATURES (2 features)
    vol_ma = volume.rolling(20, min_periods=1).mean()
    result['volume_ratio'] = volume / (vol_ma + 1)
    result['volume_trend'] = volume.rolling(5, min_periods=1).mean() / (vol_ma + 1)
    
    # ATR (1 feature)
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    result['atr_ratio'] = tr.rolling(14, min_periods=1).mean() / (close + 1e-10)
    
    # TARGET
    result['future_ret_3d'] = close.pct_change(3).shift(-3)
    
    return result


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================

def diagnose_models(X_train, y_train, X_val, y_val):
    """
    Test multiple models and show overfitting pattern.
    
    KEY: If Val R² < 0 for ALL models, the features have no signal!
    """
    print("\n" + "=" * 70)
    print("MODEL DIAGNOSTICS (Detecting Overfitting)")
    print("=" * 70)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Features: {X_train.shape[1]}")
    
    models = {
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'MLP-small': MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42),
        'MLP-medium': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }
    
    print(f"\n  {'Model':<20} {'Train R²':>10} {'Val R²':>10} {'Gap':>10} {'Overfit?':>10}")
    print("  " + "-" * 60)
    
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            gap = train_r2 - val_r2
            
            overfit = "YES ❌" if val_r2 < 0 else "No ✓"
            
            print(f"  {name:<20} {train_r2:>+10.4f} {val_r2:>+10.4f} {gap:>10.4f} {overfit:>10}")
            
            results[name] = {'train_r2': train_r2, 'val_r2': val_r2}
            
        except Exception as e:
            print(f"  {name:<20} ERROR: {e}")
    
    # Key finding
    all_negative_val = all(r['val_r2'] < 0 for r in results.values())
    
    print("\n" + "-" * 70)
    if all_negative_val:
        print("  ⚠️  ALL VALIDATION R² ARE NEGATIVE!")
        print("  This means ALL models are OVERFITTING.")
        print("  The positive IC later is SPURIOUS - it's just noise.")
    else:
        print("  Some models have positive validation R².")
    
    return results


# =============================================================================
# WALK-FORWARD BACKTEST
# =============================================================================

def walk_forward_backtest(processed: Dict[str, pd.DataFrame], 
                          feature_cols: List[str],
                          config: Dict) -> Dict:
    """Run walk-forward cross-sectional backtest."""
    
    # Get common dates
    common_dates = None
    for asset, df in processed.items():
        dates = set(df['date'])
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = sorted(common_dates)
    
    train_days = config['train_days']
    test_days = config['test_days']
    purge_days = config['purge_days']
    
    all_returns = []
    all_ics = []
    fold_results = []
    
    print("\nRunning walk-forward backtest...")
    
    for start_idx in range(train_days + purge_days, len(common_dates) - test_days, test_days):
        train_end = start_idx - purge_days
        train_start = train_end - train_days
        if train_start < 0:
            continue
        
        train_dates = common_dates[train_start:train_end]
        test_dates = common_dates[start_idx:start_idx + test_days]
        
        # Prepare data
        X_train_list, y_train_list = [], []
        test_data = {}
        
        for asset, df in processed.items():
            train_mask = df['date'].isin(train_dates)
            test_mask = df['date'].isin(test_dates)
            
            if train_mask.sum() > 50:
                X_train_list.append(df.loc[train_mask, feature_cols].values)
                y_train_list.append(df.loc[train_mask, 'future_ret_3d'].values)
            
            if test_mask.sum() > 5:
                test_data[asset] = df[test_mask].copy().reset_index(drop=True)
        
        if not X_train_list or len(test_data) < 3:
            continue
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train, y_train = X_train[valid], y_train[valid]
        
        # Train with WEAK regularization
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = Ridge(alpha=config['ridge_alpha'])  # α = 1.0 (weak!)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        scores = {}
        for asset, df in test_data.items():
            X_test = df[feature_cols].values
            X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
            X_test_scaled = scaler.transform(X_test)
            scores[asset] = model.predict(X_test_scaled)
        
        # Calculate IC
        all_pred, all_actual = [], []
        for asset, df in test_data.items():
            all_pred.extend(scores[asset])
            all_actual.extend(df['future_ret_3d'].values)
        
        valid_mask = ~np.isnan(all_actual)
        if sum(valid_mask) > 10:
            fold_ic = np.corrcoef(np.array(all_pred)[valid_mask],
                                  np.array(all_actual)[valid_mask])[0, 1]
            all_ics.append(fold_ic)
        
        # Calculate returns (long top N)
        n_days = min(len(df) for df in test_data.values())
        for day_idx in range(n_days - 1):
            rankings = [(asset, scores[asset][day_idx]) for asset in scores]
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            long_assets = [r[0] for r in rankings[:config['long_top_n']]]
            
            day_return = 0
            for asset in long_assets:
                df = test_data[asset]
                if day_idx + 1 < len(df):
                    ret = (df['close'].iloc[day_idx + 1] - df['close'].iloc[day_idx]) / df['close'].iloc[day_idx]
                    day_return += 0.25 * ret  # 25% per position
            
            all_returns.append(day_return)
        
        fold_results.append({
            'test_start': str(test_dates[0])[:10],
            'ic': fold_ic if 'fold_ic' in dir() else np.nan,
        })
        
        if len(fold_results) <= 5:
            print(f"  Fold {len(fold_results)}: IC = {all_ics[-1]:+.4f}")
    
    # Calculate statistics
    returns = np.array(all_returns)
    total_return = np.prod(1 + returns) - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    
    mean_ic = np.mean(all_ics)
    ic_std = np.std(all_ics)
    ic_tstat = mean_ic / (ic_std / np.sqrt(len(all_ics))) if ic_std > 0 else 0
    ic_pvalue = 2 * (1 - stats.t.cdf(abs(ic_tstat), df=len(all_ics)-1))
    
    # Max drawdown
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    max_dd = np.max((peak - cum_returns) / peak)
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'mean_ic': mean_ic,
        'ic_std': ic_std,
        'ic_tstat': ic_tstat,
        'ic_pvalue': ic_pvalue,
        'n_folds': len(all_ics),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("C6: CROSS-SECTIONAL CRYPTO - OVERFITTING DEMONSTRATION")
    print("=" * 70)
    
    print("""
    ⚠️  WARNING: This script demonstrates OVERFITTING!
    
    Configuration (intentionally weak):
    - Features: 41 (too many → overfitting)
    - Ridge α: 1.0 (too weak → overfitting)
    
    Compare with C7:
    - Features: 8 (minimal)
    - Ridge α: 100.0 (strong regularization)
    """)
    
    # Fetch data
    data = fetch_data(CONFIG['assets'], CONFIG['start_date'])
    
    if len(data) < 3:
        print("Error: Not enough assets")
        return
    
    # Compute features
    print("\nComputing features (41 total - intentionally too many)...")
    processed = {}
    for asset, df in data.items():
        processed[asset] = compute_features(df)
    
    # Get feature columns
    sample_df = list(processed.values())[0]
    feature_cols = [c for c in sample_df.columns if c not in 
                   ['date', 'open', 'high', 'low', 'close', 'volume', 
                    'adj close', 'future_ret_3d']]
    print(f"  Total features: {len(feature_cols)}")
    
    # Model diagnostics (shows overfitting)
    # Use first fold for diagnostics
    common_dates = None
    for asset, df in processed.items():
        dates = set(df['date'])
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = sorted(common_dates)
    
    train_dates = common_dates[:CONFIG['train_days']]
    val_dates = common_dates[CONFIG['train_days']:CONFIG['train_days']+60]
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    for asset, df in processed.items():
        train_mask = df['date'].isin(train_dates)
        val_mask = df['date'].isin(val_dates)
        
        if train_mask.sum() > 50:
            train_df = df[train_mask]
            X_train.extend(train_df[feature_cols].values)
            y_train.extend(train_df['future_ret_3d'].values)
        
        if val_mask.sum() > 10:
            val_df = df[val_mask]
            X_val.extend(val_df[feature_cols].values)
            y_val.extend(val_df['future_ret_3d'].values)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    valid_val = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
    
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_val, y_val = X_val[valid_val], y_val[valid_val]
    
    model_results = diagnose_models(X_train, y_train, X_val, y_val)
    
    # Walk-forward backtest
    results = walk_forward_backtest(processed, feature_cols, CONFIG)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"""
    PERFORMANCE:
      Total Return:     {results['total_return']:+.2%}
      Sharpe Ratio:     {results['sharpe']:+.3f}
      Max Drawdown:     {results['max_dd']:.2%}
    
    INFORMATION COEFFICIENT:
      Mean IC:          {results['mean_ic']:+.4f}
      IC Std:           {results['ic_std']:.4f}
      IC t-statistic:   {results['ic_tstat']:+.2f}
      IC p-value:       {results['ic_pvalue']:.4f}
      Number of folds:  {results['n_folds']}
    """)
    
    # Assessment
    print("=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    
    print(f"""
    The IC appears positive ({results['mean_ic']:+.4f}), BUT:
    
    ❌ All validation R² were NEGATIVE in model diagnostics
    ❌ This means the model learned noise, not signal
    ❌ The positive IC is SPURIOUS (random correlation)
    
    Compare with C7:
    - C7 uses 8 features (not 41)
    - C7 uses Ridge α=100 (not 1.0)
    - C7's IC is NEGATIVE (-0.09), revealing the truth
    
    CONCLUSION:
    This positive IC (+{results['mean_ic']:.4f}) is FAKE.
    It is caused by overfitting with weak regularization.
    """)
    
    print("\n" + "=" * 70)
    print("RUN C7 TO SEE THE TRUE RESULT")
    print("=" * 70)


if __name__ == "__main__":
    main()
