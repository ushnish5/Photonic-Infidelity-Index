WHAT IS IN THIS REPOSITORY 
==========================

The idea in one paragraph
-------------------------
We build a photonic Clements mesh that is supposed to do a matrix multiplication.
Real chips are not perfect: phases are a bit off, the beam splitters are not exactly 50/50, there is some
light loss, and the loss differs from arm to arm. This code simulates thousands of imperfect chips,
measures how wrong the output gets, fits a simple formula to that, checks that the formula still works when
all the errors act together, and finally checks how the matrix "fidelity" follows from it.

The files are numbered in the order you read them and the order you run them.
Every file also starts with its own short explanation, so you can open any of them and read the top.


THE FILES
---------

0_shared_settings.py
    The few numbers and tiny helpers that several files need (reference size 100, the square root of 2,
    the 1.2 cut-off for "error has saturated", number-to-text helpers, and the plot style).
    Nothing runs here. Everything else imports from it.

1_mzi_mesh.py
    The physics. Makes a random unitary matrix, turns it into the mesh of interferometers, adds the four
    kinds of hardware errors, sends light through many noisy copies of the chip, and measures the error.
    The speed-critical loops here are compiled with numba (marked "implemented by Claude" on the same line).
    Nothing runs here by itself.

2_common_sweep_engine.py
    The machinery behind the four sweeps: makes the test inputs, loops over chip sizes and error values,
    averages everything, saves the tables and the plot, and runs a few self-tests first.
    You never run this directly. The four 2x files use it.

2a_phase_error_sweep.py
    Sweep 1: how bad does the output get when the phase shifters are inaccurate?

2b_splitting_ratio_error_sweep.py
    Sweep 2: how bad when the beam splitters are not exactly 50/50?

2c_mean_loss_sweep.py
    Sweep 3: how bad when the arms lose some light (same loss in every arm, no spread)?

2d_loss_spread_sweep.py
    Sweep 4: how bad when the loss is different from arm to arm (the spread of the loss)?
    Two panels: panel A varies the chip size, panel B varies the average loss.

    Each sweep has a "small" range (used for the fits) and a "large" range (only to see
    what happens further out).

3_common_fit_engine.py
    The machinery behind the fits: the formula, the fitting, the error bars (bootstrap),
    quality numbers, and the fit plots. Also used by 3d. You never run this directly.

3a_phase_error_fit.py, 3b_splitting_ratio_error_fit.py, 3c_mean_loss_fit.py
    Fit the formula to sweep 1, 2 and 3. Each is only a few lines because the real work is in
    3_common_fit_engine.py. The fit uses the small range; the same numbers are then shown
    on the large range without refitting.
    Run 3c before 3d (3d needs its result).

3d_loss_spread_fit.py
    Fit for the loss-spread channel (sweep 4). It reuses the mean-loss fit and only fits the extra
    part caused by the spread. It also runs a few robustness checks (drop one curve and refit, etc.).

4_total_error_validation.py
    The "does it all add up" test. Turns on all four errors at the same time, simulates it,
    and compares with what the four separate fits predict when added together.
    Done once for chip sizes we fitted on, and once for in-between sizes we did not.
    Makes the agreement figure (simulated vs predicted, with residuals).

5_fidelity_validation.py
    The matrix-fidelity study. Test 1: does fidelity follow the infidelity index no matter which
    mix of errors you use? Test 2: does the formula with the 1/N^2 floor hold for different chip sizes?
    Makes the two fidelity figures. Can also be run with --selftest for a quick sanity check.

6_unit_checks.py
    Quick known-answer tests of the small building blocks. Takes a few seconds, independent of
    everything else. Prints PASS or FAIL for each check.

requirements.txt
    The Python packages needed.


HOW TO RUN IT
-------------

Quick test (about a minute in total, tiny simulations, results go into folders ending in _quick):
    python 2a_phase_error_sweep.py --quick     then   python 3a_phase_error_fit.py --quick
    python 2b_splitting_ratio_error_sweep.py --quick   then   python 3b_splitting_ratio_error_fit.py --quick
    python 2c_mean_loss_sweep.py --quick       then   python 3c_mean_loss_fit.py --quick
    python 2d_loss_spread_sweep.py --quick     then   python 3d_loss_spread_fit.py --quick
    python 4_total_error_validation.py --quick
    python 5_fidelity_validation.py --quick

Real run: the same commands without --quick. It takes many hours, but it can be stopped and
restarted; it picks up where it left off.

Order matters: each 2x before its 3x, 3c before 3d, and 4 and 5 last (they read the fit results).


WHAT YOU GET OUT
----------------

Result folders appear next to the scripts:
    results_phase_error, results_splitting_ratio_error, results_mean_loss, results_loss_spread,
    results_total_error_validation, results_fidelity_vs_pii

Inside: tables (csv), arrays (npz), fit results and summaries (json), and the figures as PDF.
The figures are:
    - for each sweep: the pooled (all input types) error curves for the small and large range,
      and the fit on the small range with the same parameters on the large range
    - for the loss spread: the same, separately for panel A and panel B
    - total_error_validation.pdf
    - F_vs_PII_N<size>.pdf and F_vs_N_floor.pdf

AI ASSISTANCE
-------------
Parts of this code were written or reorganized with AI assistance (Claude, Anthropic):
  - the numba-compiled kernels in 1_mzi_mesh.py; each is marked "implemented by Claude"
    on the same line
  - the organization of the scripts (section headings and comments)
  - the plotting and figure styling
The model, the simulation design, the choice of tests and the physical analysis are the
author's own. The decomposition and propagation code was checked against the PhaseShift
library.

REQUIREMENTS
------------
numpy==2.4.4
scipy==1.17.1
numba==0.68.1
llvmlite==0.50.0
matplotlib==3.10.8
pandas==3.0.2
