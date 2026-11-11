"""
OUTPUT ERROR OF A CLEMENTS UNITARY MESH vs. EFFECTIVE INSERTION-LOSS SPREAD sigma_eff
=====================================================================================

Output error E of a Haar-random unitary Clements mesh vs. the EFFECTIVE (realised) per-arm loss spread sigma_eff (dB).
Only loss is active. L ~ TruncNorm(mu_L, sigma_L^2; 0, 2 mu_L) dB; the realised std of L is always below mu_L/sqrt(3).
The swept variable is sigma_eff = std(L); the nominal Gaussian width sigma_L of the underlying normal is derived from
it numerically for every mu_L (1_mzi_mesh.sigma_nominal_from_eff) before the losses are drawn.
DYNAMIC TOP LEVEL.  Every curve has 50 equally spaced sigma_eff points from 0 to
    sigma_eff_top(mu_L) = min(0.95 mu_L/sqrt(3), 0.263 dB)           (1_mzi_mesh.sigma_eff_top)
for ITS OWN mu_L, so every requested sigma_eff is reachable by construction (no clipping anywhere) and curves with a
small mu_L end early (intended).  x-axis of every figure = sigma_eff (dB), y-axis = output error E; "vs N" / "vs mu_L"
only says what the separate curves are.  Four panels (each its own figure / CSV / npz):
  small (calibration; used for fits and validation):
    Panel A : one curve per network size N (11 sizes) at fixed mu_L = 0.5 dB, each curve 0 -> 0.263 dB
    Panel B : one curve per mean loss mu_L at N = 100 (9 values), each curve 0 -> sigma_top(mu_L)
  large (exploratory / "big range"; qualitative only, evaluated without refit, never used in fits or validation):
    Panel A : one curve per N at mu_L = 1.0 dB, each curve 0 -> 0.95/sqrt(3) = 0.5485 dB   (0.263 dB cap LIFTED)
    Panel B : mu_L in {0.5, 0.75, 1.0} dB at N = 100, each curve 0 -> 0.95 mu_L/sqrt(3)     (0.263 dB cap LIFTED)
Besides the figures, the run writes an overlay check (CSV / JSON) between independent runs on their overlap.

This file is only the PARAMETER BLOCK; the engine (inputs, sweeps, checkpoints, CSV, figures,
validation) lives in 2_common_sweep_engine.py.  One run sweeps BOTH the small and the large range
(50 points each) and writes the figure (PDF), CSV and npz for both.

    python 2d_loss_spread_sweep.py            # full experiment
    python 2d_loss_spread_sweep.py --quick    # tiny smoke test (a minute or so), results in a *_quick folder

Requires: numpy, scipy, matplotlib, numba; 1_mzi_mesh.py and 2_common_sweep_engine.py in the same folder
Long runs are checkpointed per curve: re-running the same command resumes where it stopped.
"""

import importlib
sweep_engine = importlib.import_module("2_common_sweep_engine")

CHANNEL = "sigma_L"
N_VALUES = [20, 28, 38, 53, 72, 100, 138, 190, 263, 362, 500]   # 11 sizes, equally spaced in log N; N0 = 100 is the middle one
N_MATRICES = 25                    # Haar-random unitaries per N (decomposed once, reused for every curve)
CHIPS_PER_MATRIX = 30              # hardware realizations per unitary per sweep point
INPUT_TYPES = ["dense_complex", "dense_real", "binary_pm1", "phase_only", "sparse_10pct", "single_port", "gaussian_beam"]
INPUTS_PER_TYPE = 10               # random draws per input type and unitary; chip c uses draw (c mod 10) of every type
N_POINTS = 50                      # points per range

# --- exploratory ("big range") settings: they LIFT the 0.263 dB cap (top = 0.95 mu_L/sqrt3, the physical bound only).
#     Assumptions to be confirmed: the mu_L list of the exploratory spread-vs-mu_L panel.  Trivial to revert.
EXPLORATORY_MU_FOR_N_SWEEP = 1.0                 # exploratory spread-vs-N panel: fixed mu_L (dB)
EXPLORATORY_MU_LIST = [0.5, 0.75, 1.0]           # exploratory spread-vs-mu_L panel (N = 100); smaller mu_L would duplicate the small range
EXPLORATORY_APPLY_CAP = False                    # False: top = 0.95 mu_L/sqrt3 only;  True: min(0.95 mu_L/sqrt3, 0.263) as in the small range

# --- sweep ranges: "small" = calibration range (used for the fits / validation), "large" = exploratory "big range".
#     Each curve runs from 0 to sigma_eff_top(mu_L) (see the docstring); there is no common maximum.
RANGES = {
    "small": dict(apply_cap=True,
                  panel_A_mu=0.5,
                  panel_B_N=100,
                  panel_B_mu_values=[0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]),
    "large": dict(apply_cap=EXPLORATORY_APPLY_CAP,
                  panel_A_mu=EXPLORATORY_MU_FOR_N_SWEEP,
                  panel_B_N=100,
                  panel_B_mu_values=EXPLORATORY_MU_LIST),
}

# --- optional separability cross-check (default OFF; run only on request): is the N-scaling of the spread sensitivity
#     independent of mu_L?  Two mu_L x two N, small-range grids.  The fit script reports it if the CSV exists.
SEPARABILITY_CHECK = False
SEPARABILITY_MU = (0.1, 1.0)
SEPARABILITY_N = (20, 500)

# --- slope marker / end value ---
SLOPE_FIT_INDICES = (1, 2)         # first two nonzero points (index 0 is x = 0)
END_AVG_POINTS = 1                 # sigma_eff figures: E_max = E at sigma_top of the curve (last grid point; no plateau is expected)

SEED = 12345
RUN_VALIDATION = True
RESUME = True                      # reuse per-curve checkpoints if the settings are unchanged
SAVE_PER_CHIP = True               # also store every chip's E in the .npz
SHOW_PLOT = False                  # True: open all figures at the end (many windows)

FIG_SIZE = (16, 9)                 # 16:9
MARKERS = ["o", "^", "s", "D", "v", "<", ">", "P", "X", "*", "h"]
MARKER_SIZE = 5.0
LINE_WIDTH = 1.4
OUTPUT_DIR = "results_loss_spread"


def main():
    """Collect the settings above into one dictionary and run the sweep."""
    cfg = dict(
        CHANNEL=CHANNEL, N_VALUES=N_VALUES, N_MATRICES=N_MATRICES, CHIPS_PER_MATRIX=CHIPS_PER_MATRIX,
        INPUT_TYPES=INPUT_TYPES, INPUTS_PER_TYPE=INPUTS_PER_TYPE, N_POINTS=N_POINTS, RANGES=RANGES,
        SLOPE_FIT_INDICES=SLOPE_FIT_INDICES, END_AVG_POINTS=END_AVG_POINTS,
        SEED=SEED, RUN_VALIDATION=RUN_VALIDATION, RESUME=RESUME, SAVE_PER_CHIP=SAVE_PER_CHIP, SHOW_PLOT=SHOW_PLOT,
        FIG_SIZE=FIG_SIZE, MARKERS=MARKERS, MARKER_SIZE=MARKER_SIZE, LINE_WIDTH=LINE_WIDTH,
        OUTPUT_DIR=OUTPUT_DIR, SEPARABILITY_CHECK=SEPARABILITY_CHECK, SEPARABILITY_MU=SEPARABILITY_MU, SEPARABILITY_N=SEPARABILITY_N,
    )
    sweep_engine.main_entry(cfg)


if __name__ == "__main__":
    main()
