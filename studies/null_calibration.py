#!/usr/bin/env python3
"""
Null calibration: how often does each evaluation protocol "find alpha" when
there is none, and how often does it find alpha that is really there?

Runs entirely on synthetic panels (see ``alpharesearch.synthetic``), so it
needs no market data and its results are exactly reproducible from the seeds.

Measured on null panels (no predictable structure by construction):
  * the corrected protocol: daily rank IC + Newey-West t (pre-registered config)
  * the same IC with a naive t-stat that ignores overlapping labels
  * the legacy C6/C7 protocol: pooled Pearson IC per fold, t-test across folds
  * specification search: best |t| over a 12-configuration grid, with and
    without the Deflated Sharpe Ratio
  * how often the legacy C6 and C7 configurations disagree on the sign of the
    IC - the pattern the original README read as "proof of overfitting"
Measured on planted panels (a known reversal effect): detection power.

Usage:
    python studies/null_calibration.py                  # full run (~5 min, 4 cores)
    python studies/null_calibration.py --seeds 10       # quick run
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpharesearch.metrics import deflated_sharpe_ratio  # noqa: E402
from alpharesearch.pipeline import StudyConfig  # noqa: E402
from alpharesearch.results import save_results  # noqa: E402
from alpharesearch.study import evaluate  # noqa: E402
from alpharesearch.synthetic import synthetic_panel  # noqa: E402

N_ASSETS = 10
N_DAYS = 1800          # ~ 2020-2025 of daily crypto data
SYN_START = "2018-01-01"
FIRST_TEST = str((pd.Timestamp(SYN_START) + pd.Timedelta(days=520)).date())

BASELINE = StudyConfig()  # pre-registered: compact, cs_rank features & target, alpha=100

# Legacy specifications, run through the corrected pipeline (no imputation,
# purge enforced) but with raw features/targets like the original scripts.
LEGACY_C6 = StudyConfig(feature_set="extended", feature_transform="none", target_transform="none",
                        horizon=3, ridge_alpha=1.0, train_days=365, purge=5)
LEGACY_C7 = StudyConfig(feature_set="compact", feature_transform="none", target_transform="none",
                        horizon=5, ridge_alpha=100.0, train_days=500, purge=10)

GRID = [BASELINE.with_(ridge_alpha=a, feature_set=f, target_transform=t)
        for a, f, t in product([1.0, 100.0, 1e4], ["compact", "extended"], ["none", "cs_rank"])]


def legacy_fold_t(res):
    """C6/C7's statistic: pooled Pearson IC per fold, plain t-test across folds."""
    scores, fwd = res["_scores"], res["_fwd"]
    ics = []
    for f in res["_folds"]:
        s = scores.loc[f["test_start"]:f["test_end"]].to_numpy().ravel()
        y = fwd.loc[f["test_start"]:f["test_end"]].to_numpy().ravel()
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() > 10:
            ics.append(np.corrcoef(s[ok], y[ok])[0, 1])
    ics = np.array(ics)
    t = ics.mean() / (ics.std() / np.sqrt(len(ics)))
    return float(ics.mean()), float(t)


def run_null_seed(seed):
    panel = synthetic_panel(N_ASSETS, N_DAYS, seed=seed, start=SYN_START)
    base = evaluate(panel, BASELINE, first_test=FIRST_TEST)
    grid = [evaluate(panel, cfg, first_test=FIRST_TEST) for cfg in GRID]
    c6 = evaluate(panel, LEGACY_C6, first_test=FIRST_TEST, keep_scores=True)
    c7 = evaluate(panel, LEGACY_C7, first_test=FIRST_TEST, keep_scores=True)
    c6_ic, c6_t = legacy_fold_t(c6)
    c7_ic, c7_t = legacy_fold_t(c7)

    # Specification search: report the grid member with the best net Sharpe,
    # deflated by the dispersion of the Sharpes of all members tried.
    rets = [g["_ls_returns"] for g in grid]
    per_period_sr = [r.mean() / r.std(ddof=1) for r in rets]
    best = int(np.argmax(per_period_sr))
    return {
        "seed": seed,
        "baseline_t_nw": base["ic"]["t_nw"],
        "baseline_t_naive": base["ic"]["t_naive"],
        "baseline_sharpe": base["long_short"]["sharpe_net"],
        "grid_max_abs_t_nw": max(abs(g["ic"]["t_nw"]) for g in grid),
        "grid_best_sharpe": grid[best]["long_short"]["sharpe_net"],
        "grid_best_dsr": deflated_sharpe_ratio(rets[best], per_period_sr),
        "c6_ic": c6_ic, "c6_t": c6_t,
        "c7_ic": c7_ic, "c7_t": c7_t,
    }


def run_power_seed(args):
    seed, k = args
    panel = synthetic_panel(N_ASSETS, N_DAYS, seed=10_000 + seed, signal_strength=k, start=SYN_START)
    r = evaluate(panel, BASELINE, first_test=FIRST_TEST)
    return {"k": k, "seed": seed, "mean_ic": r["ic"]["mean_ic"], "t_nw": r["ic"]["t_nw"],
            "sharpe": r["long_short"]["sharpe_net"]}


def rate(x):
    x = np.asarray(x, dtype=bool)
    lo, hi = stats.binomtest(int(x.sum()), len(x)).proportion_ci(0.95)
    return {"rate": float(x.mean()), "ci95": [float(lo), float(hi)], "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--power-seeds", type=int, default=50)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.02, 0.04, 0.08])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with ProcessPoolExecutor(args.workers) as ex:
        null = pd.DataFrame(list(ex.map(run_null_seed, range(args.seeds))))
        power = pd.DataFrame(list(ex.map(run_power_seed,
                                         product(range(args.power_seeds), args.strengths))))

    z = 1.96
    fp = {
        "corrected_rank_ic_newey_west": rate(null.baseline_t_nw.abs() > z),
        "same_ic_naive_t_ignoring_overlap": rate(null.baseline_t_naive.abs() > z),
        "legacy_C7_fold_pearson_t": rate(null.c7_t.abs() > z),
        "legacy_C6_fold_pearson_t": rate(null.c6_t.abs() > z),
        "best_of_12_configs_t_nw": rate(null.grid_max_abs_t_nw > z),
        "best_of_12_configs_dsr_above_0.95": rate(null.grid_best_dsr > 0.95),
    }
    sign_flip = {
        "c6_c7_opposite_sign": rate(np.sign(null.c6_ic) != np.sign(null.c7_ic)),
        "c6_positive_and_c7_negative": rate((null.c6_ic > 0) & (null.c7_ic < 0)),
        "c7_negative_and_significant": rate((null.c7_ic < 0) & (null.c7_t.abs() > z)),
    }
    pw = {}
    for k, g in power.groupby("k"):
        pw[f"k={k}"] = {"mean_ic": float(g.mean_ic.mean()), "detection_t_nw>1.96": rate(g.t_nw > z),
                        "median_sharpe_net": float(g.sharpe.median())}

    payload = {
        "description": "False-positive rate and power of evaluation protocols on synthetic panels",
        "setup": {"n_assets": N_ASSETS, "n_days": N_DAYS, "first_test": FIRST_TEST,
                  "baseline": BASELINE.to_dict(), "legacy_c6": LEGACY_C6.to_dict(),
                  "legacy_c7": LEGACY_C7.to_dict(), "grid_size": len(GRID),
                  "null_seeds": args.seeds, "power_seeds": args.power_seeds,
                  "nominal_level": 0.05},
        "false_positive_rate": fp,
        "legacy_sign_flip_under_null": sign_flip,
        "power": pw,
        "null_distribution_summary": {
            "baseline_sharpe_net_p5_p50_p95": np.percentile(null.baseline_sharpe, [5, 50, 95]).tolist(),
            "best_of_grid_sharpe_net_p5_p50_p95": np.percentile(null.grid_best_sharpe, [5, 50, 95]).tolist(),
        },
    }
    path = save_results("null_calibration", payload, args.out)
    write_markdown(payload, os.path.join(args.out, "null_calibration.md"))
    print(open(os.path.join(args.out, "null_calibration.md")).read())
    print(f"saved {path}")


def write_markdown(p, path):
    pct = lambda r: f"{100 * r['rate']:.1f}% [{100 * r['ci95'][0]:.1f}–{100 * r['ci95'][1]:.1f}]"  # noqa: E731
    s = p["setup"]
    lines = [
        "# Null calibration (generated by `studies/null_calibration.py`)",
        "",
        f"Synthetic panels: {s['n_assets']} assets, {s['n_days']} days, "
        f"{s['null_seeds']} null seeds, {s['power_seeds']} seeds per signal strength. "
        "Nominal test level 5% (|t| > 1.96). Brackets: 95% CI.",
        "",
        "## False-positive rate on data with no signal",
        "",
        "| Protocol | Rejects H0 |",
        "|---|---|",
    ]
    lines += [f"| {k} | {pct(v)} |" for k, v in p["false_positive_rate"].items()]
    lines += ["", "## Legacy C6 vs C7 under the null", "", "| Event | Frequency |", "|---|---|"]
    lines += [f"| {k} | {pct(v)} |" for k, v in p["legacy_sign_flip_under_null"].items()]
    lines += ["", "## Power on planted reversal signal (pre-registered baseline)", "",
              "| Strength | Mean rank IC | Detected | Median net Sharpe |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean_ic']:+.3f} | {pct(v['detection_t_nw>1.96'])} | {v['median_sharpe_net']:+.2f} |"
              for k, v in p["power"].items()]
    lines += ["", "## Net Sharpe on null data (5th / 50th / 95th percentile)", ""]
    for k, v in p["null_distribution_summary"].items():
        lines.append(f"- {k}: " + " / ".join(f"{x:+.2f}" for x in v))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
