"""
SCALABLE FIT OF THE MEAN-LOSS CHANNEL mu_L  (loss spread sigma_L = 0)
=====================================================================

Reads results_mean_loss/sensitivity_mu_L_{small,large}_sigmaL0dB_curves.csv (written by 2c_mean_loss_sweep.py),
fits  E = sqrt(2) sqrt(1 - exp(-u^2)),  u = (s_L(N) mu_L + gamma_L(N) mu_L^2) / sqrt(2),
      s_L(N) = s_L0 (N/100)^p_L,  gamma_L(N) = gamma0 (N/100)^q_L
to the SMALL range (pooled mean over all input types, all N, equal weight per point) and shows the same
parameters, without refitting, on the LARGE range.

Writes into results_mean_loss/:  fit_mu_L_small.pdf, fit_mu_L_large.pdf,
fit_results_mu_L.json (parameters, bootstrap s.d./95% CI over unitaries, metrics), fit_loss_points.csv,
jensen_gap_mu_L.json.

>>> 3d_loss_spread_fit.py needs fit_results_mu_L.json: run this script BEFORE 3d. <<<

The quadratic term is only weakly constrained if the small range is short; the bootstrap s.d. / 95% CI
in the JSON and in the figure show how well gamma0 and q_L are determined.

All settings of the fit (bootstrap size and seed, figure size) are at the top of 3_common_fit_engine.py.
Requires 3_common_fit_engine.py in the same folder.   Run:  python 3c_mean_loss_fit.py   (add --quick for the smoke test)
"""
import importlib
import sys
from pathlib import Path

fit_engine = importlib.import_module("3_common_fit_engine")

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    fit_engine.run_fit("mu_L", HERE, suffix="_quick" if "--quick" in sys.argv else "")      # --quick: read / write results_*_quick
