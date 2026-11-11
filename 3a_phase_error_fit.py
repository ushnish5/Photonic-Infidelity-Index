"""
SCALABLE FIT OF THE PHASE-SHIFTER CHANNEL sigma_rad
===================================================

Reads results_phase_error/sensitivity_sigma_rad_{small,large}_curves.csv (written by 2a_phase_error_sweep.py),
fits  E = sqrt(2) sqrt(1 - exp(-u^2)),  u = s_phi(N) sigma_rad / sqrt(2),  s_phi(N) = s_phi0 (N/100)^p_phi
to the SMALL range (pooled mean over all input types, all N, equal weight per point) and shows the same
parameters, without refitting, on the LARGE range.

Writes into results_phase_error/:  fit_sigma_rad_small.pdf, fit_sigma_rad_large.pdf,
fit_results_sigma_rad.json (parameters, bootstrap s.d./95% CI over unitaries, metrics), fit_phase_points.csv,
jensen_gap_sigma_rad.json.

All settings of the fit (bootstrap size and seed, figure size) are at the top of 3_common_fit_engine.py.
Requires 3_common_fit_engine.py in the same folder.   Run:  python 3a_phase_error_fit.py   (add --quick for the smoke test)
"""
import importlib
import sys
from pathlib import Path

fit_engine = importlib.import_module("3_common_fit_engine")

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    fit_engine.run_fit("sigma_rad", HERE, suffix="_quick" if "--quick" in sys.argv else "")      # --quick: read / write results_*_quick
