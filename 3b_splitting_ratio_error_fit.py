"""
SCALABLE FIT OF THE SPLITTING-RATIO CHANNEL sigma_S
===================================================

Reads results_splitting_ratio_error/sensitivity_sigma_S_{small,large}_curves.csv (written by 2b_splitting_ratio_error_sweep.py),
fits  E = sqrt(2) sqrt(1 - exp(-u^2)),  u = s_S(N) sigma_S / sqrt(2),  s_S(N) = s_S0 (N/100)^p_S
to the SMALL range (pooled mean over all input types, all N, equal weight per point) and shows the same
parameters, without refitting, on the LARGE range.

Writes into results_splitting_ratio_error/:  fit_sigma_S_small.pdf, fit_sigma_S_large.pdf,
fit_results_sigma_S.json (parameters, bootstrap s.d./95% CI over unitaries, metrics), fit_split_points.csv,
jensen_gap_sigma_S.json.

All settings of the fit (bootstrap size and seed, figure size) are at the top of 3_common_fit_engine.py.
Requires 3_common_fit_engine.py in the same folder.   Run:  python 3b_splitting_ratio_error_fit.py   (add --quick for the smoke test)
"""
import importlib
import sys
from pathlib import Path

fit_engine = importlib.import_module("3_common_fit_engine")

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    fit_engine.run_fit("sigma_S", HERE, suffix="_quick" if "--quick" in sys.argv else "")      # --quick: read / write results_*_quick
