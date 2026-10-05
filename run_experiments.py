#!/usr/bin/env python3
"""
Alpha Research Framework - Experiment Runner
=============================================

Unified runner for the corrected studies (studies/) and the legacy v1.0
experiments (experiments/). The legacy experiments are kept for reference
only; see docs/ERRATA.md for why their conclusions were withdrawn.

Usage:
    python run_experiments.py --study null    # Null calibration (offline, synthetic)
    python run_experiments.py --study crypto  # Corrected crypto study (needs data)
    python run_experiments.py --list          # List all experiments
    python run_experiments.py --show          # Show documented results
    python run_experiments.py --compare       # Run C6 vs C7 comparison
    python run_experiments.py -e E1           # Run specific experiment
    python run_experiments.py --all           # Run all experiments
"""

import argparse
import subprocess
import sys
import os
import json
from datetime import datetime

# Corrected studies built on the alpharesearch package
STUDIES = {
    'null': 'studies/null_calibration.py',
    'crypto': 'studies/crypto_xs_ablation.py',
}

# Legacy v1.0 experiment catalog ("expected" = numbers reported in v1.0)
EXPERIMENTS = {
    'E1': {
        'name': 'Single Stock LSTM',
        'file': 'experiments/equity/E1_single_stock_lstm.py',
        'description': 'LSTM overfitting demonstration on AAPL',
        'expected': 'Train IC +0.86 → Val IC -0.06 (severe overfitting)',
        'runtime': '~2 min'
    },
    'E2': {
        'name': 'Cross-Sectional Equity',
        'file': 'experiments/equity/E2_cross_sectional.py',
        'description': 'Cross-sectional ranking targets',
        'expected': 'IC ≈ 0.02, normal decay pattern',
        'runtime': '~3 min'
    },
    'E3': {
        'name': 'Fundamentals (Biased)',
        'file': 'experiments/equity/E3_fundamentals_bias.py',
        'description': 'Discovers look-ahead bias in Yahoo Finance fundamentals',
        'expected': 'IC ≈ 0.18 (BIASED - fake result due to data leakage)',
        'runtime': '~3 min'
    },
    'E4': {
        'name': 'Annual Fundamentals',
        'file': 'experiments/equity/E4_annual_fundamentals.py',
        'description': 'Point-in-time fundamentals (bias corrected)',
        'expected': 'IC ≈ 0.02, p > 0.10 (not significant after correction)',
        'runtime': '~3 min'
    },
    'C1': {
        'name': 'Crypto Time-Series',
        'file': 'experiments/crypto/C1_timeseries.py',
        'description': 'Direction prediction on BTC/ETH/SOL',
        'expected': 'Edge ~0.5%, IC ~0.04 (marginal, not robust)',
        'runtime': '~2 min'
    },
    'C2': {
        'name': 'Crypto Technical Alpha',
        'file': 'experiments/crypto/C2_technical_alpha.py',
        'description': 'Technical indicators on 10 cryptos',
        'expected': 'IC ≈ -0.01, p = 0.55 (not significant)',
        'runtime': '~3 min'
    },
    'C3': {
        'name': 'Production System',
        'file': 'experiments/crypto/C3_production_system.py',
        'description': 'Full hybrid system with costs',
        'expected': 'Sharpe 0.69, p = 0.02 (borderline)',
        'runtime': '~5 min'
    },
    'C6': {
        'name': 'Cross-Sectional v2 (Overfit Demo)',
        'file': 'experiments/crypto/C6_overfitting_demo.py',
        'description': '⚠️ Demonstrates overfitting with weak regularization',
        'expected': 'IC +0.036 (SPURIOUS - all Val R² negative)',
        'runtime': '~4 min'
    },
    'C7': {
        'name': 'Cross-Sectional v3 (Robust Final)',
        'file': 'experiments/crypto/C7_robust_final.py',
        'description': '✅ Main result with anti-overfitting techniques',
        'expected': 'IC -0.084, p = 0.001 (v1.0 reading withdrawn, see docs/ERRATA.md)',
        'runtime': '~4 min'
    }
}


def list_experiments():
    """List all available experiments."""
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 70)
    
    print("\n📊 EQUITY EXPERIMENTS")
    print("-" * 70)
    for eid in ['E1', 'E2', 'E3', 'E4']:
        exp = EXPERIMENTS[eid]
        print(f"\n  {eid}: {exp['name']}")
        print(f"      {exp['description']}")
        print(f"      Expected: {exp['expected']}")
        print(f"      Runtime: {exp['runtime']}")
    
    print("\n\n💰 CRYPTO EXPERIMENTS")
    print("-" * 70)
    for eid in ['C1', 'C2', 'C3', 'C6', 'C7']:
        exp = EXPERIMENTS[eid]
        print(f"\n  {eid}: {exp['name']}")
        print(f"      {exp['description']}")
        print(f"      Expected: {exp['expected']}")
        print(f"      Runtime: {exp['runtime']}")
    
    print("\n" + "=" * 70)
    print("\nLegacy experiments only. Start with: python run_experiments.py --study null")
    print()


def show_results():
    """Show documented results from JSON."""
    results_file = os.path.join(os.path.dirname(__file__), 'results', 'documented_results.json')
    
    if not os.path.exists(results_file):
        print("ERROR: Results file not found. Run experiments first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 70)
    print("DOCUMENTED RESULTS")
    print("=" * 70)
    
    # Key comparison
    print("\n🎯 KEY FINDING: C6 vs C7 Comparison")
    print("-" * 70)
    comp = results['key_comparison']
    print(f"""
    ┌───────────────────┬──────────────┬──────────────┐
    │ Parameter         │ C6 (Overfit) │ C7 (Robust)  │
    ├───────────────────┼──────────────┼──────────────┤
    │ Features          │ {comp['c6']['features']:<12} │ {comp['c7']['features']:<12} │
    │ Ridge α           │ {comp['c6']['ridge_alpha']:<12} │ {comp['c7']['ridge_alpha']:<12} │
    │ Mean IC           │ {comp['c6']['mean_ic']:>+12.3f} │ {comp['c7']['mean_ic']:>+12.3f} │
    │ p-value           │ {comp['c6']['p_value']:<12.3f} │ {comp['c7']['p_value']:<12.3f} │
    └───────────────────┴──────────────┴──────────────┘
    """)
    print(f"    Interpretation: {comp['interpretation']}")
    
    # All experiments summary
    print("\n\n📊 ALL EXPERIMENTS SUMMARY")
    print("-" * 70)
    print(f"\n    {'ID':<4} {'Name':<35} {'IC':<12} {'Status'}")
    print("    " + "-" * 65)
    
    for eid, exp in results['experiments'].items():
        name = exp['name'][:33]
        if 'mean_ic' in exp['results']:
            ic = f"{exp['results']['mean_ic']:+.3f}"
        elif 'val_ic' in exp['results']:
            ic = f"{exp['results']['val_ic']:+.3f}"
        elif 'test_ic' in exp['results']:
            ic = f"{exp['results']['test_ic']:+.3f}"
        else:
            ic = "N/A"
        
        status = "⚠️ BIASED" if exp.get('warning') else ("✅ MAIN" if exp.get('is_main_result') else "Done")
        print(f"    {eid:<4} {name:<35} {ic:<12} {status}")
    
    # Conclusions
    print("\n\n📝 CONCLUSIONS")
    print("-" * 70)
    print(f"\n    Main Finding: {results['conclusions']['main_finding']}")
    print("\n    Key Insights:")
    for insight in results['conclusions']['key_insights']:
        print(f"      • {insight}")
    
    print("\n" + "=" * 70)


def run_experiment(exp_id):
    """Run a single experiment."""
    if exp_id not in EXPERIMENTS:
        print(f"ERROR: Unknown experiment '{exp_id}'")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        return False
    
    exp = EXPERIMENTS[exp_id]
    script_path = os.path.join(os.path.dirname(__file__), exp['file'])
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    print("\n" + "=" * 70)
    print(f"Running Experiment {exp_id}: {exp['name']}")
    print("=" * 70)
    print(f"Description: {exp['description']}")
    print(f"Expected Result: {exp['expected']}")
    print(f"Estimated Runtime: {exp['runtime']}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_comparison():
    """Run the key C6 vs C7 comparison."""
    print("\n" + "=" * 70)
    print("KEY COMPARISON: C6 (Overfitting) vs C7 (Robust)")
    print("=" * 70)
    print("""
    LEGACY: the v1.0 conclusion drawn from this comparison is withdrawn
    (see docs/ERRATA.md). It is kept for reference only.
    
    C6: 41 features, Ridge α=1.0, 10 coins, 3-day horizon
    C7: 8 features, Ridge α=100.0, 5 coins, 5-day horizon
    (and different start date, training window and purge gap)

    Because seven settings change at once, a change in IC cannot be
    attributed to regularization. Use --study crypto for the controlled
    one-factor-at-a-time version.
    """)
    print("=" * 70)
    
    print("\n\n🔴 STEP 1: Running C6 (Weak Regularization)...")
    print("=" * 70)
    run_experiment('C6')
    
    print("\n\n🟢 STEP 2: Running C7 (Strong Regularization)...")
    print("=" * 70)
    run_experiment('C7')
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print("""
    Results reported in v1.0 (hand-written, not reproducible):

    C6:  IC ≈ +0.036, p ≈ 0.14
    C7:  IC ≈ -0.084, p ≈ 0.001

    A significantly negative IC is a signal with the opposite sign, not
    evidence of "no alpha". See docs/ERRATA.md.
    """)


def run_all():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    print("This will take approximately 15-20 minutes.\n")
    
    results = {}
    for exp_id in ['E1', 'E2', 'E3', 'E4', 'C1', 'C2', 'C3', 'C6', 'C7']:
        success = run_experiment(exp_id)
        results[exp_id] = 'SUCCESS' if success else 'FAILED'
        print("\n" + "=" * 70 + "\n")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print("\nResults:")
    for exp_id, status in results.items():
        symbol = "✅" if status == 'SUCCESS' else "❌"
        print(f"  {symbol} {exp_id}: {status}")
    
    print("\n" + "=" * 70)
    print("\n✨ Run 'python run_experiments.py --show' to see documented results")


def run_study(name, extra_args):
    """Run one of the corrected studies."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), STUDIES[name])
    return subprocess.run([sys.executable, script] + extra_args,
                          cwd=os.path.dirname(os.path.abspath(__file__))).returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Alpha Research Framework - Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --study null    Null calibration (offline)
  python run_experiments.py --study crypto  Corrected crypto study (needs data)
  python run_experiments.py --list          List all legacy experiments
  python run_experiments.py --show          Show documented results
  python run_experiments.py --compare       Run legacy C6 vs C7 comparison
  python run_experiments.py -e C7           Run specific experiment
  python run_experiments.py --all           Run all experiments
        """
    )
    
    parser.add_argument('--study', choices=sorted(STUDIES),
                        help='Run a corrected study (extra args are passed through)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available experiments')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Show documented results')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Run legacy C6 vs C7 comparison (withdrawn, see docs/ERRATA.md)')
    parser.add_argument('--experiment', '-e', type=str,
                        help='Run specific experiment (E1, E2, ..., C7)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run all experiments')
    
    args, extra = parser.parse_known_args()

    if args.study:
        sys.exit(0 if run_study(args.study, extra) else 1)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.list:
        list_experiments()
    elif args.show:
        show_results()
    elif args.compare:
        run_comparison()
    elif args.experiment:
        run_experiment(args.experiment.upper())
    elif args.all:
        run_all()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
