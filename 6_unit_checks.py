"""
6_unit_checks.py  --  unit-level checks of the physics helpers and the statistics
===================================================================================

Small tests with known answers, independent of the result folders.  Takes a few seconds.
Run:   python 6_unit_checks.py        Exit code 0 = all checks pass.

    1  sigma_eff_top against its reference table        5  E2_raw equals E_raw^2 per chip and input
    2  round trip  sigma_eff -> nominal width -> sigma_eff   6  spread grids run from 0 to sigma_top, 50 points
    3  Monte Carlo of the truncated Gaussian loss
    4  aggregation formulas (E_rms, s.e., I_rms)
"""
import copy
import importlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
mesh = importlib.import_module("1_mzi_mesh")  # noqa: E402
sweep_engine = importlib.import_module("2_common_sweep_engine")  # noqa: E402

all_passed = True


def report(name, passed, detail=""):
    """Print PASS / FAIL for one check and remember whether everything has passed so far."""
    global all_passed
    all_passed &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")


# ======================================================================================
# CHECKS
# ======================================================================================

def check_sigma_eff_top():
    """1. sigma_eff_top(mu_L) against a 4-decimal reference table; sigma_eff_top(0) = 0; arrays work like scalars."""
    table = {0.020: 0.0110, 0.05: 0.0274, 0.10: 0.0548, 0.15: 0.0823, 0.20: 0.1097, 0.25: 0.1371, 0.35: 0.1920, 0.40: 0.2194,
             0.50: 0.263, 0.51: 0.263, 0.75: 0.263, 1.00: 0.263}
    worst = max(abs(mesh.sigma_eff_top(mu) - top) for mu, top in table.items())
    report("sigma_eff_top reference table (4-decimal table)", worst < 5.0e-5, f"max dev {worst:.1e}")
    scalar_and_array_agree = np.allclose(mesh.sigma_eff_top(np.array([0.1, 1.0])), [mesh.sigma_eff_top(0.1), mesh.sigma_eff_top(1.0)])
    report("sigma_eff_top(0) = 0 and array == scalar", mesh.sigma_eff_top(0.0) == 0.0 and scalar_and_array_agree)


def check_round_trip():
    """2. sigma_eff -> nominal Gaussian width -> sigma_eff returns the starting value."""
    worst = 0.0
    for mu in [0.01, 0.0204, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 5.0]:
        for fraction in np.linspace(0, 0.99, 100):
            sigma_eff = fraction * mu / math.sqrt(3)
            back = mesh.sigma_eff_trunc(mu, mesh.sigma_nominal_from_eff(mu, sigma_eff))
            worst = max(worst, abs(back - sigma_eff))
    report("round trip sigma_eff_trunc(mu, nominal(mu, x)) = x", worst < 1e-10, f"worst {worst:.1e}")


def check_monte_carlo():
    """3. Draws of the truncated Gaussian loss have the requested mean mu_L and standard deviation sigma_eff."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for mu, sigma_eff in [(1.0, 0.263), (0.5, 0.2742), (0.3, 0.132), (0.05, 0.027)]:
        loss = mesh.truncnorm_sample(mu, mesh.sigma_nominal_from_eff(mu, sigma_eff), 0.0, 2 * mu, 2_000_000, rng)
        worst = max(worst, abs(loss.mean() / mu - 1), abs(loss.std() / sigma_eff - 1))
    report("Monte Carlo 2e6 draws: mean ~ mu_L, std ~ sigma_eff (rel. error < 0.3 %)", worst < 3e-3, f"worst {worst:.1e}")


def check_aggregation():
    """4. rms_statistics on a small example with a hand-computed answer."""
    m2 = np.array([[0.04, 0.16], [0.09, 0.25], [0.16, 0.36], [0.25, 0.49]])        # (4 unitaries, 2 points)
    m1 = np.sqrt(m2) * 0.95
    stats = sweep_engine.rms_statistics(m1, m2)
    E = np.sqrt(m2.mean(axis=0))
    se = m2.std(axis=0, ddof=1) / (2 * E * 2.0)
    I = -np.log(1 - E ** 2 / 2)
    report("E_rms = sqrt(mean m2)", np.allclose(stats["E_rms"], E, rtol=1e-14))
    report("delta-method s.e. of E_rms", np.allclose(stats["se_E_rms"], se, rtol=1e-14))
    report("sd of the per-unitary RMS values (ddof=1)", np.allclose(stats["sd_E_rms"], np.sqrt(m2).std(axis=0, ddof=1)))
    report("I_rms = -ln(1 - E_rms^2/2); Jensen gap >= 0", np.allclose(stats["I_rms"], I) and np.all(stats["jensen_gap_E"] >= 0))


def check_squared_error_accumulation():
    """5. E^2 is accumulated directly: E2_raw == E_raw^2 per chip and input."""
    n = 20
    bundle = sweep_engine.build_matrix(n, dict(INPUT_TYPES=list(sweep_engine.ALL_INPUT_TYPES), INPUTS_PER_TYPE=3, CHIPS_PER_MATRIX=12), np.random.default_rng(3))
    out = mesh.run_chips(bundle.dec, bundle.X3, 12, sigma_rad=0.1, sigma_S=0.05, mu_L=0.3, sigma_L=0.0,
                         rng=np.random.default_rng(4), Y_ideal=bundle.Y3, sigma_eff=0.1)
    rel = np.max(np.abs(out["E2_raw"] - out["E_raw"] ** 2) / np.maximum(out["E2_raw"], 1e-300))
    report("E2_raw equals E_raw^2 (direct squared norm)", rel < 1e-10, f"max rel diff {rel:.1e}")


def check_spread_grids():
    """6. The grids of the spread sweep run from 0 to sigma_top(mu_L) in 50 points (the cap is lifted only in the exploratory range)."""
    sys.argv = ["x"]
    spread_sweep = importlib.import_module("2d_loss_spread_sweep")  # noqa: E402
    cfg = dict(CHANNEL="sigma_L", N_VALUES=[20, 100], N_POINTS=50, RANGES=copy.deepcopy(spread_sweep.RANGES), SEPARABILITY_CHECK=True,
               SEPARABILITY_MU=(0.1, 1.0), SEPARABILITY_N=(20, 100))
    good = True
    for family in sweep_engine.build_families(cfg):
        for curve in family["curves"]:
            mu = curve["fixed"]["mu_L"]
            top = float(mesh.sigma_eff_top(mu)) if family["apply_cap"] else 0.95 * mu / math.sqrt(3)
            good &= bool(curve["x"][0] == 0 and abs(curve["x"][-1] - top) < 1e-12 and len(curve["x"]) == 50)
    report("every spread grid = 50 points from 0 to sigma_top (cap lifted only in the exploratory range)", good)


def main():
    """Run all checks and exit with code 0 only if all of them pass."""
    check_sigma_eff_top()
    check_round_trip()
    check_monte_carlo()
    check_aggregation()
    check_squared_error_accumulation()
    check_spread_grids()
    print("ALL PASS" if all_passed else "SOME FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
