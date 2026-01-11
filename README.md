# Alpha Research Framework

**A rigorous, reproducible framework for testing ML-based alpha generation strategies.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Finding

This research demonstrates that **apparent alpha from ML models on public market data is primarily overfitting**.

The critical comparison between experiments C6 and C7 proves this:

| Metric | C6 (Weak Regularization) | C7 (Strong Regularization) |
|--------|--------------------------|----------------------------|
| Features | 41 | 8 |
| Ridge α | 1.0 | 100.0 |
| **Mean IC** | **+0.036** | **-0.084** |
| p-value | 0.14 | 0.001 |
| Validation R² | All negative | All negative | 

**The IC flips from positive to significantly negative when overfitting is controlled.**

This is the classic signature of spurious patterns: complex models find noise, simple models reveal truth.

---

## 📊 Experiment Results Summary

### Equity Experiments (E1-E4)

| ID | Description | Key Result | Status |
|----|-------------|------------|--------|
| E1 | LSTM Overfitting Demo | Train IC +0.86 → Val IC -0.06 | ✅ Severe overfitting demonstrated |
| E2 | Cross-Sectional Targets | IC ≈ 0.02-0.03, normal decay | ✅ Weak signal |
| E3 | Fundamentals (BIASED) | IC +0.18, p=0.0000 | ⚠️ FAKE - look-ahead bias |
| E4 | Fundamentals (Corrected) | IC ≈ 0.02, p > 0.10 | ✅ No significance after correction |

### Crypto Experiments (C1-C7)

| ID | Description | Key Result | Status |
|----|-------------|------------|--------|
| C1 | Time-Series Direction | Edge ~0.5%, IC ~0.04 | ✅ Marginal, not robust |
| C2 | Technical Alpha | IC -0.01, p=0.55 | ✅ Not significant |
| C3 | Production System | Sharpe 0.69, p=0.02 | ⚠️ Borderline |
| C6 | Overfitting Demo | IC +0.036, Val R² < 0 | ⚠️ SPURIOUS |
| C7 | **Anti-Overfitting** | **IC -0.084, p=0.001** | ✅ **MAIN RESULT** |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/alpha_research_framework.git
cd alpha_research_framework
pip install -r requirements.txt
```

### Run Key Comparison (Recommended)

```bash
# Run the critical C6 vs C7 comparison
python run_experiments.py --compare
```

### Run Individual Experiments

```bash
# List all experiments
python run_experiments.py --list

# Run specific experiment
python run_experiments.py -e C7    # Main result
python run_experiments.py -e E1    # Overfitting demo
python run_experiments.py -e E3    # Look-ahead bias demo

# Show documented results without running
python run_experiments.py --show
```

### Run All Experiments

```bash
# Full test suite (~15-20 minutes)
python run_experiments.py --all
```

---

## 📁 Project Structure

```
alpha_research_framework/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── run_experiments.py        # Unified experiment runner
├── experiments/
│   ├── equity/
│   │   ├── E1_single_stock_lstm.py    # LSTM overfitting
│   │   ├── E2_cross_sectional.py      # Cross-sectional targets
│   │   ├── E3_fundamentals_bias.py    # Look-ahead bias demo
│   │   └── E4_annual_fundamentals.py  # Bias-corrected
│   └── crypto/
│       ├── C1_timeseries.py           # Direction prediction
│       ├── C2_technical_alpha.py      # Technical indicators
│       ├── C3_production_system.py    # Full system
│       ├── C6_overfitting_demo.py     # ⚠️ Shows overfitting
│       └── C7_robust_final.py         # ✅ Main result
├── docs/
│   ├── RESEARCH_JOURNEY.md   # Full narrative
│   ├── METHODOLOGY.md        # Technical details
│   └── RESULTS_SUMMARY.md    # All results
└── results/
    └── documented_results.json
```

---

## 🔬 Methodology Highlights

### Anti-Overfitting Techniques

1. **Walk-Forward Validation**: Train on past, test on future (no shuffling)
2. **Purge Gap**: 5-10 day gap between train/test to prevent leakage
3. **Feature Reduction**: 8 features max (vs 41 in overfit version)
4. **Strong Regularization**: Ridge α = 100 (vs 1.0 in overfit version)
5. **Cross-Sectional Targets**: Rank-based to remove market trend

### Statistical Rigor

- **IC t-test**: Test if mean IC is significantly different from zero
- **Multiple Folds**: 40-70 walk-forward windows per experiment
- **Validation R²**: Must be non-negative (negative = overfitting)

---

## 📈 Key Insights

### What We Learned

1. **Neural networks overfit easily** on financial data (E1)
2. **Look-ahead bias creates fake alpha** - always check data timestamps (E3 vs E4)
3. **Positive IC with negative validation R² = overfitting** (C6)
4. **Strong regularization reveals truth** (C7)
5. **Public price data has no exploitable alpha** with standard ML

### What Works

- Walk-forward validation with purge gaps
- Cross-sectional (ranking) targets
- Minimal features (8-10 max)
- Strong regularization (Ridge α ≥ 100)
- Statistical significance testing

### What Doesn't Work

- Complex models (LSTM, deep MLP) without heavy regularization
- Many features (>20) without selection
- Absolute return targets (includes beta)
- Using current data for historical predictions

---

## 📚 Documentation

- **[RESEARCH_JOURNEY.md](docs/RESEARCH_JOURNEY.md)**: Full narrative from hypothesis to null result
- **[METHODOLOGY.md](docs/METHODOLOGY.md)**: Technical details and anti-overfitting checklist
- **[RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)**: Complete results tables

---

## 🎓 Academic Value

This framework demonstrates:

1. **Rigorous methodology** for financial ML research
2. **Honest null result** - finding no alpha is valid science
3. **Reproducible experiments** with clear documentation
4. **Overfitting detection** techniques applicable to any ML project

---

## ⚠️ Disclaimer

This is **research code** for educational purposes. Results are based on historical data and do not guarantee future performance. This is not financial advice.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please read the methodology documentation first to understand the anti-overfitting principles.

---

## 📖 Citation

```bibtex
@software{alpha_research_framework,
  title={Alpha Research Framework: A Rigorous Approach to ML-Based Alpha Generation},
  year={2025},
  url={https://github.com/yourusername/alpha_research_framework}
}
```
