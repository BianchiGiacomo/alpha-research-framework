#!/usr/bin/env python3
"""
Cross-sectional crypto study - the corrected replacement for legacy C6/C7.

What changed with respect to C6/C7 (see docs/ERRATA.md):
  * one factor varied at a time around a pre-registered baseline, instead of
    changing 7 settings at once
  * daily cross-sectional rank IC with Newey-West t-stats
  * cross-sectional (rank) target, as the original README claimed
  * costs charged on turnover; holding period = label horizon; 365-day year
  * no zero-imputation of missing features; assets tradable only after
    ``min_history`` days (limits hindsight in the ticker list)
  * every configuration is evaluated on the same out-of-sample days
  * multiple testing made explicit: Deflated Sharpe Ratio + PBO over all trials
  * a final hold-out period, touched once, for the baseline and for the best
    in-sample configuration
  * data snapshot hashed in the result file; explicit end date

Usage:
    python studies/crypto_xs_ablation.py                     # needs Yahoo Finance access
    python studies/crypto_xs_ablation.py --synthetic         # dry run, no network
    python studies/crypto_xs_ablation.py --cache-dir PATH    # use pre-downloaded parquet files
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpharesearch.data import load_prices  # noqa: E402
from alpharesearch.metrics import (deflated_sharpe_ratio, expected_max_sharpe,  # noqa: E402
                                   probability_of_backtest_overfitting)
from alpharesearch.pipeline import StudyConfig  # noqa: E402
from alpharesearch.results import save_results  # noqa: E402
from alpharesearch.study import BacktestConfig, evaluate, public  # noqa: E402
from alpharesearch.synthetic import synthetic_panel  # noqa: E402

UNIVERSES = {
    # Legacy C7 universe and legacy C6 universe (both hand-picked in hindsight).
    "c7_5": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"],
    "c6_10": ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
              "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD"],
}

BASELINE_UNIVERSE = "c6_10"
BASELINE = StudyConfig(feature_set="compact", feature_transform="cs_rank",
                       target_transform="cs_rank", horizon=5, ridge_alpha=100.0,
                       train_days=500, test_days=30, purge=10, min_history=60)

# One-factor-at-a-time ablation around the baseline.
ABLATION = {
    "ridge_alpha": [0.1, 1.0, 10.0, 1e3, 1e4, 1e5],
    "feature_set": ["extended"],
    "feature_transform": ["none"],
    "target_transform": ["none", "cs_demean"],
    "horizon": [1, 3, 10],
    "train_days": [365, 730],
    "purge": [20],
}

# The legacy specifications, re-run under the corrected evaluation.
LEGACY = {
    "legacy_C6_spec": ("c6_10", StudyConfig(feature_set="extended", feature_transform="none",
                                            target_transform="none", horizon=3, ridge_alpha=1.0,
                                            train_days=365, purge=5)),
    "legacy_C7_spec": ("c7_5", StudyConfig(feature_set="compact", feature_transform="none",
                                           target_transform="none", horizon=5, ridge_alpha=100.0,
                                           train_days=500, purge=10)),
}


def trials():
    yield "baseline", BASELINE_UNIVERSE, BASELINE
    for param, values in ABLATION.items():
        for v in values:
            cfg = BASELINE.with_(**{param: v})
            if param == "horizon":
                cfg = cfg.with_(purge=max(BASELINE.purge, v))
            yield f"{param}={v}", BASELINE_UNIVERSE, cfg
    yield "universe=c7_5", "c7_5", BASELINE
    for name, (u, cfg) in LEGACY.items():
        yield name, u, cfg


def row(name, universe, r):
    ic, ls, lo = r["ic"], r["long_short"], r["long_only_excess"]
    return {"trial": name, "universe": universe, "mean_ic": ic.get("mean_ic"),
            "t_nw": ic.get("t_nw"), "p_nw": ic.get("p_nw"), "t_naive": ic.get("t_naive"),
            "ls_sharpe_net": ls["sharpe_net"], "ls_sharpe_gross": ls["sharpe_gross"],
            "ls_turnover": ls["avg_daily_turnover"], "lo_excess_sharpe_net": lo["sharpe_net"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--eval-start", default="2022-03-01",
                    help="first out-of-sample day, common to all trials")
    ap.add_argument("--holdout-start", default="2025-07-01",
                    help="hold-out period start; never used during the ablation")
    ap.add_argument("--end", default="2026-09-30", help="exclusive end date of the data")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--synthetic", action="store_true",
                    help="run on a synthetic null panel (pipeline dry run, no network)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    all_tickers = sorted(set(sum(UNIVERSES.values(), [])))
    if args.synthetic:
        n_days = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days
        full = synthetic_panel(len(all_tickers), n_days, seed=0, start=args.start)
        full.close.columns = full.volume.columns = full.high.columns = full.low.columns = all_tickers
        name = "crypto_xs_ablation_SYNTHETIC_DRY_RUN"
    else:
        print("Loading prices...")
        full = load_prices(all_tickers, args.start, args.end, cache_dir=args.cache_dir)
        name = "crypto_xs_ablation"

    in_sample = full.truncate(args.holdout_start)  # prices after this date do not exist
    bt = BacktestConfig(top_k=args.top_k, cost_bps=args.cost_bps)

    results, rows, returns = {}, [], {}
    for tname, universe, cfg in trials():
        tickers = [t for t in UNIVERSES[universe] if t in full.assets]
        r = evaluate(in_sample.select(tickers), cfg, bt, first_test=args.eval_start)
        results[tname] = public(r) | {"universe": universe}
        rows.append(row(tname, universe, r))
        returns[tname] = r["_ls_returns"]
        print(f"  {tname:<28} IC {r['ic'].get('mean_ic', np.nan):+.4f}  "
              f"t_NW {r['ic'].get('t_nw', np.nan):+.2f}  "
              f"Sharpe(LS,net) {r['long_short']['sharpe_net']:+.2f}")

    # ---- multiple testing over everything that was tried --------------------
    R = pd.DataFrame(returns).dropna()
    sr = (R.mean() / R.std(ddof=1)).to_numpy()
    for rw in rows:
        rw["dsr"] = deflated_sharpe_ratio(R[rw["trial"]], sr)
    n_sig = sum(1 for rw in rows if rw["p_nw"] is not None and rw["p_nw"] < 0.05)
    multiple_testing = {
        "n_trials": len(rows),
        "n_significant_at_5pct": n_sig,
        "expected_by_chance": 0.05 * len(rows),
        "expected_max_sharpe_annual_under_null": expected_max_sharpe(sr) * np.sqrt(365),
        "pbo": probability_of_backtest_overfitting(R.to_numpy(), n_blocks=16),
        "note": "trials share data and differ in one setting, so they are strongly "
                "correlated; DSR/PBO treat them as the search space actually explored",
    }

    # ---- hold-out: touched once, for two configurations decided in advance ---
    best = max(rows, key=lambda rw: rw["ls_sharpe_net"])["trial"]
    trial_map = {n: (u, c) for n, u, c in trials()}
    holdout = {}
    for label, tname in [("pre_registered_baseline", "baseline"), ("best_in_sample", best)]:
        u, cfg = trial_map[tname]
        tickers = [t for t in UNIVERSES[u] if t in full.assets]
        r = evaluate(full.select(tickers), cfg, bt, first_test=args.holdout_start)
        holdout[label] = {"trial": tname, **public(r)}

    payload = {
        "description": "Corrected cross-sectional crypto study (replaces legacy C6/C7)",
        "data": {"start": args.start, "end": args.end, "eval_start": args.eval_start,
                 "holdout_start": args.holdout_start, "synthetic": args.synthetic,
                 "tickers_loaded": full.assets, "sha256": full.manifest},
        "universes": UNIVERSES,
        "backtest": vars(bt),
        "summary_table": rows,
        "multiple_testing": multiple_testing,
        "holdout": holdout,
        "trials": results,
    }
    path = save_results(name, payload, args.out)
    md = os.path.join(args.out, f"{name}.md")
    write_markdown(payload, md)
    print(open(md).read())
    print(f"saved {path}")


def write_markdown(p, path):
    f = lambda x, fmt: "n/a" if x is None or not np.isfinite(x) else format(x, fmt)  # noqa: E731
    d = p["data"]
    lines = ["# Cross-sectional crypto ablation (generated by `studies/crypto_xs_ablation.py`)", ""]
    if d["synthetic"]:
        lines += ["> **SYNTHETIC DRY RUN** - numbers below come from a null panel, not market data.", ""]
    lines += [
        f"In-sample out-of-sample period: {d['eval_start']} → {d['holdout_start']}; "
        f"hold-out: {d['holdout_start']} → {d['end']}. Long-short top/bottom "
        f"{p['backtest']['top_k']}, {p['backtest']['cost_bps']:.0f} bps on turnover.",
        "",
        "| Trial | Universe | Mean rank IC | t (NW) | t (naive) | LS Sharpe net | LS Sharpe gross | "
        "LO excess Sharpe net | DSR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p["summary_table"]:
        lines.append(f"| {r['trial']} | {r['universe']} | {f(r['mean_ic'], '+.4f')} | {f(r['t_nw'], '+.2f')} | "
                     f"{f(r['t_naive'], '+.2f')} | {f(r['ls_sharpe_net'], '+.2f')} | "
                     f"{f(r['ls_sharpe_gross'], '+.2f')} | {f(r['lo_excess_sharpe_net'], '+.2f')} | "
                     f"{f(r['dsr'], '.2f')} |")
    mt = p["multiple_testing"]
    lines += ["", "## Multiple testing", "",
              f"- Trials: {mt['n_trials']}; significant at 5% (NW): {mt['n_significant_at_5pct']} "
              f"(≈{mt['expected_by_chance']:.1f} expected by chance if independent)",
              f"- Best annual Sharpe expected from luck alone: "
              f"{f(mt['expected_max_sharpe_annual_under_null'], '.2f')}",
              f"- PBO (CSCV, {mt['pbo'].get('n_splits', 0)} splits): {f(mt['pbo'].get('pbo'), '.2f')}",
              "", "## Hold-out (evaluated once)", "",
              "| Config | Trial | Mean rank IC | t (NW) | LS Sharpe net |", "|---|---|---|---|---|"]
    for label, h in p["holdout"].items():
        lines.append(f"| {label} | {h['trial']} | {f(h['ic'].get('mean_ic'), '+.4f')} | "
                     f"{f(h['ic'].get('t_nw'), '+.2f')} | {f(h['long_short']['sharpe_net'], '+.2f')} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
