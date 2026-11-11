"""
OUTPUT ERROR OF A CLEMENTS UNITARY MESH vs. PHASE-SHIFTER ERROR sigma_rad
=========================================================================

Output error E of a Haar-random unitary Clements mesh vs. phase-shifter error sigma_rad.
Only phase errors are active; splitting ratio and loss are ideal (the output phase screen is always ideal).

This file is only the PARAMETER BLOCK; the engine (inputs, sweeps, checkpoints, CSV, figures,
validation) lives in 2_common_sweep_engine.py.  One run sweeps BOTH the small and the large range
(50 points each) and writes the figure (PDF), CSV and npz for both.

    python 2a_phase_error_sweep.py            # full experiment
    python 2a_phase_error_sweep.py --quick    # tiny smoke test (a minute or so), results in a *_quick folder

Requires: numpy, scipy, matplotlib, numba; 1_mzi_mesh.py and 2_common_sweep_engine.py in the same folder
Long runs are checkpointed per curve: re-running the same command resumes where it stopped.
"""

import importlib
sweep_engine = importlib.import_module("2_common_sweep_engine")

CHANNEL = "sigma_rad"
N_VALUES = [20, 28, 38, 53, 72, 100, 138, 190, 263, 362, 500]   # 11 sizes, equally spaced in log N; N0 = 100 is the middle one
N_MATRICES = 25                    # Haar-random unitaries per N (decomposed once, reused for every curve)
CHIPS_PER_MATRIX = 30              # hardware realizations per unitary per sweep point
INPUT_TYPES = ["dense_complex", "dense_real", "binary_pm1", "phase_only", "sparse_10pct", "single_port", "gaussian_beam"]
INPUTS_PER_TYPE = 10               # random draws per input type and unitary; chip c uses draw (c mod 10) of every type
N_POINTS = 50                      # points per range

# --- sweep ranges: the "small" range (near literature-typical values; used for the fits) and the "large" stress range
RANGES = {
    "small": dict(x_max=0.25),     # rad
    "large": dict(x_max=0.50),     # rad
}

# --- slope marker / saturation estimate ---
SLOPE_FIT_INDICES = (1, 2)         # first two nonzero points (index 0 is x = 0)

SEED = 12345
RUN_VALIDATION = True
RESUME = True                      # reuse per-curve checkpoints if the settings are unchanged
SAVE_PER_CHIP = True               # also store every chip's E in the .npz
SHOW_PLOT = False                  # True: open all figures at the end (many windows)

FIG_SIZE = (16, 9)                 # 16:9
MARKERS = ["o", "^", "s", "D", "v", "<", ">", "P", "X", "*", "h"]
MARKER_SIZE = 5.0
LINE_WIDTH = 1.4
OUTPUT_DIR = "results_phase_error"


def main():
    """Collect the settings above into one dictionary and run the sweep."""
    cfg = dict(
        CHANNEL=CHANNEL, N_VALUES=N_VALUES, N_MATRICES=N_MATRICES, CHIPS_PER_MATRIX=CHIPS_PER_MATRIX,
        INPUT_TYPES=INPUT_TYPES, INPUTS_PER_TYPE=INPUTS_PER_TYPE, N_POINTS=N_POINTS, RANGES=RANGES,
        SLOPE_FIT_INDICES=SLOPE_FIT_INDICES,
        SEED=SEED, RUN_VALIDATION=RUN_VALIDATION, RESUME=RESUME, SAVE_PER_CHIP=SAVE_PER_CHIP, SHOW_PLOT=SHOW_PLOT,
        FIG_SIZE=FIG_SIZE, MARKERS=MARKERS, MARKER_SIZE=MARKER_SIZE, LINE_WIDTH=LINE_WIDTH,
        OUTPUT_DIR=OUTPUT_DIR,
    )
    sweep_engine.main_entry(cfg)


if __name__ == "__main__":
    main()
