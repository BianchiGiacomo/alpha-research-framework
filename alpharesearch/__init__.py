"""
alpharesearch - shared core for the Alpha Research Framework.

All studies are built from the same small set of audited components:

- ``data``       price panels with an on-disk snapshot and content hashes
- ``synthetic``  null and planted-signal panels for calibration
- ``features``   backward-looking features + a look-ahead check
- ``cv``         walk-forward splits with an enforced purge gap
- ``pipeline``   out-of-sample prediction panels
- ``backtest``   portfolio construction with turnover-based costs
- ``metrics``    rank IC, Newey-West t-stats, PSR/DSR, PBO
- ``results``    machine-generated, self-describing result files
"""

__version__ = "0.2.0"
