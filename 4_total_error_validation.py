"""
4_total_error_validation.py  --  do the four fitted channels add up correctly when all are on together?
=====================================================================================================

The single-channel fits (3a-3d) describe each error channel alone.  Here all four channels act at the same time
on the SAME mesh, and the prediction of the combined surrogate (no refitting) is compared with a direct simulation.
The channels are combined as independent multiplicative overlaps:

    I_ph = u_phi^2 + u_S^2 + u_L^2 + u_sigma^2 = -ln(c_phi c_S c_L c_sigma),      E_tot = sqrt(2) sqrt(1 - exp(-I_ph))

Two validations (each with its own CSV / metrics; one figure shows both):

  1. GRID     tested network sizes (the N of the sweeps) x 81 error combinations = low / middle / high of every channel
              (3^4): sigma_rad, sigma_S, mu_L at points 2, 26, 50 of the 50-point small-range sweeps; the loss spread is a
              FRACTION f in {0, 0.5, 1} of the dynamic top level  sigma_eff = f * sigma_top(mu_L),
              sigma_top(mu_L) = min(0.95 mu_L/sqrt(3), 0.263 dB)  (1_mzi_mesh.sigma_eff_top).  Every combination is
              reachable by construction; nothing is clipped or dropped (asserted).
              Two predictions are compared with the simulation:
                - measured sum : single-channel simulated E from the sweep CSVs (phase, splitting, mean loss at
                                 sigma_eff = 0), combined as multiplicative overlaps; the spread overlap exp(-u_sigma^2)
                                 of the f > 0 levels comes from the fit (no measured spread curve at these exact levels)
                - fitted sum   : the four fitted functions (fit_results_*.json)
  2. RANDOM   UNTESTED network sizes placed half-way (geometric mean) between adjacent tested N, with RANDOM error values
              that lie BETWEEN the measured grid points of the small range (inside a sweep interval, never on a grid
              point), stratified so that low / middle / high values of all four channels are mixed (81 strata per N,
              one random draw per stratum).  Draw order: sigma_rad, sigma_S, mu_L inside their strata (mu_L > 0: the low
              stratum starts at the first non-zero sweep value); then top = sigma_top(mu_L); then the fraction f uniformly in the
              open interior of its stratum (0,1/3), (1/3,2/3), (2/3,1) and sigma_eff = f * top.
              Only the fitted sum can be evaluated here (the CSV tables exist on the grid only).

LOSS SPREAD = sigma_eff (realised std of the per-arm loss), always.  The simulation converts it to the nominal Gaussian
width numerically (1_mzi_mesh.sigma_nominal_from_eff).  Only a third of the combinations (low mu_L stratum) has
negligible spread (<= 0.011 dB on the grid): the spread channel is meaningfully exercised at mu_L >~ 0.3 dB (reported separately).

Every simulated point: N_MATRICES Haar unitaries x CHIPS hardware realizations (all four channels active, fresh draws),
each chip exposed to one input of EVERY input type (INPUTS_PER_TYPE random draws per type, chip c uses draw c mod D);
E_rms_sim = sqrt(mean of E^2 over unitaries, chips and input types (equal weight per type)); s.e. across unitaries.

Metrics are computed ONLY on points with simulated E_rms < E_CUT = 1.2 (all points are plotted; cut points are marked), with the
all-point block stored separately "for reference only" and a robustness check with the cut on the PREDICTED E_rms.  Reported (i) on the
retained set, (ii) separately for the literature-typical error box and for points beyond it.

NEEDED (from the previous steps; all result folders are next to this script):
    results_phase_error/sensitivity_sigma_rad_small_curves.csv          results_phase_error/fit_results_sigma_rad.json
    results_splitting_ratio_error/sensitivity_sigma_S_small_curves.csv  results_splitting_ratio_error/fit_results_sigma_S.json
    results_mean_loss/sensitivity_mu_L_small_sigmaL0dB_curves.csv       results_mean_loss/fit_results_mu_L.json
    results_loss_spread/sensitivity_sigma_L_small_panelB_*_curves.csv   results_loss_spread/fit_results_sigma_L.json

    python 4_total_error_validation.py            # full run (results in results_total_error_validation/)
    python 4_total_error_validation.py --quick    # tiny smoke test (results in results_total_error_validation_quick/)
Simulations are cached per (kind, N): re-running reuses them if the settings are unchanged.

The file is organised like a book:
    1 settings   2 reading the previous results   3 the two predictions   4 simulation of the combined circuit
    5 test points   6 one table row per test point   7 agreement metrics   8 figure   9 report and driver
"""
import hashlib
import importlib
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # so the numbered modules can be imported
shared = importlib.import_module("0_shared_settings")  # noqa: E402
mesh = importlib.import_module("1_mzi_mesh")  # noqa: E402
sweep_engine = importlib.import_module("2_common_sweep_engine")  # noqa: E402


# ======================================================================================
# 1. SETTINGS
# ======================================================================================
N_MATRICES = 10                         # Haar unitaries per point
CHIPS = 30                              # hardware realizations per unitary per point
INPUT_TYPES = list(sweep_engine.ALL_INPUT_TYPES)  # pooled with equal weight per type
N_INPUT_TYPES = len(INPUT_TYPES)
INPUTS_PER_TYPE = 10                    # chip c uses draw (c mod 10) of every type
SEED = 20261011                         # validation master seed (distinct from the calibration sweeps and from 5_fidelity_validation.py)
GRID_LEVEL_IDX = (1, 25, 49)            # "low / middle / high" = points 2, 26, 50 of the 50-point small-range sweeps
SPREAD_FRACTIONS = (0.0, 0.5, 1.0)      # grid: sigma_eff = f * sigma_top(mu_L), sigma_top = min(0.95 mu_L/sqrt3, 0.263 dB) (mesh.sigma_eff_top)
SPREAD_FROM_FIT = [f for f in SPREAD_FRACTIONS if f > 0.0]   # fractions whose spread overlap comes from the fit in the "measured sum"
SPREAD_STRATA_F = ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0))   # random test: open intervals of the fraction f
STRATA_IDX = ((1, 8), (20, 29), (40, 48))   # random test: sweep-interval indices of the low / middle / high strata
POS_IN_INTERVAL = (0.2, 0.8)            # random test: relative position inside the chosen sweep interval (never on a grid point)
TYPICAL_BOX = dict(sigma_rad=0.185, sigma_S=0.03, mu_L=0.5, sigma_eff=0.132)   # literature-typical upper bounds (sigma_eff: realised spread)
N0, SQRT2, E_CUT = shared.N0, shared.SQRT2, shared.E_CUT      # E_CUT: points with SIMULATED E_rms >= E_CUT are plotted (marked) but excluded from all agreement metrics
CHANNELS = ("sigma_rad", "sigma_S", "mu_L", "sigma_L")        # the four single-channel result folders / fits
LEVEL_NAMES = ["sigma_rad", "sigma_S", "mu_L", "sigma_eff_frac"]   # the four axes of the low / middle / high grid

OUT_NAME = "results_total_error_validation"
QUICK = "--quick" in sys.argv
if QUICK:
    N_MATRICES, CHIPS, INPUTS_PER_TYPE, OUT_NAME = 3, 10, 5, OUT_NAME + "_quick"
OUT = HERE / OUT_NAME
OUT.mkdir(exist_ok=True)


# ======================================================================================
# 2. READING THE PREVIOUS RESULTS (single-channel data and fits)
# ======================================================================================

def find_dir(name):
    """A result folder of a previous step (they all live next to this script)."""
    if not (HERE / name).is_dir():
        raise SystemExit(f"folder '{name}' not found in {HERE}: run the previous steps (2a-2d, 3a-3d) first")
    return HERE / name


def need(path):
    """The path, or stop with a clear message if the file does not exist."""
    if not path.exists():
        raise SystemExit(f"missing file: {path}\n(run the sweep / fit scripts first)")
    return path


def read_fit_parameters(folders):
    """The fitted parameters of the four channels, from fit_results_<channel>.json.

    Returns {channel: {parameter name: value}} in file order:  sigma_rad (s0, p),  sigma_S (s0, p),
    mu_L (s_L0, p_L, gamma0, q_L),  sigma_L (s_sigma0, p_sigma)."""
    parameters = {}
    for channel in CHANNELS:
        with open(need(folders[channel] / f"fit_results_{channel}.json")) as f:
            parameters[channel] = json.load(f)["params"]
    return parameters


def read_sweep_table(path, xcol):
    """A single-channel curves CSV, and the sorted values of its swept column."""
    d = pd.read_csv(path)
    return d, np.sort(d[xcol].unique())


def read_spread_sweep_mu_L(folder):
    """mu_L values of the spread sweep curves (random draws must not coincide with them); empty if the file is missing."""
    hits = sorted(folder.glob("sensitivity_sigma_L_small_panelB_*_curves.csv"))
    if not hits:
        return np.array([])
    return np.sort(pd.read_csv(hits[0])["fixed_mu_L"].unique())


def measured_levels(table, xcol, tested_N):
    """E_rms at the low / middle / high point of the sweep, for every tested N."""
    return {n: table[table.N == n].sort_values(xcol).E_rms.values[list(GRID_LEVEL_IDX)] for n in tested_N}


def read_setup():
    """Everything the validation needs from the previous steps, in one namespace:

    parameters     {channel: {name: value}} of the fits    (see read_fit_parameters; written into the metrics JSON)
    fit            {channel: list of values}               (the same, for the formulas)
    grid           sweep values of sigma_rad, sigma_S, mu_L (the random test draws between them)
    levels         low / middle / high values of the four axes of the grid test
    tested_N       sizes present in every CSV;   mid_N  = geometric means of adjacent tested sizes
    measured       {sigma_rad / sigma_S / mu_L: {N: E_rms at the three levels}}   (from the sweep CSVs)
    sweep_mu_L     mu_L values of the spread sweep
    grid_combos, grid_levels, grid_fractions   the 81 error combinations of the grid test"""
    folders = {k: find_dir(sweep_engine.RESULT_FOLDER[k] + ("_quick" if QUICK else "")) for k in CHANNELS}
    print("result folders used:")
    for k, v in folders.items():
        print(f"  {k:10s}: {v}")
    mean_loss_file = need(folders["mu_L"] / "sensitivity_mu_L_small_sigmaL0dB_curves.csv")      # deterministic-loss (sigma_eff = 0) curves are mandatory
    sr_file = need(folders["sigma_rad"] / "sensitivity_sigma_rad_small_curves.csv")
    sS_file = need(folders["sigma_S"] / "sensitivity_sigma_S_small_curves.csv")
    print(f"spread fractions f = {SPREAD_FROM_FIT}: in the 'measured sum' the spread overlap exp(-u_sigma^2) is taken from the fit "
          f"(mean-loss part stays measured, sigma_eff = 0)")
    parameters = read_fit_parameters(folders)

    d_sr, g_sr = read_sweep_table(sr_file, "sigma_rad")
    d_sS, g_sS = read_sweep_table(sS_file, "sigma_S")
    d_mL, g_mL = read_sweep_table(mean_loss_file, "mu_L")
    tested_N = sorted(int(n) for n in set.intersection(*[set(d.N.unique()) for d in (d_sr, d_sS, d_mL)]))      # tested N = N present in every CSV
    mid_N = [int(round(np.sqrt(a * b))) for a, b in zip(tested_N[:-1], tested_N[1:])]
    print(f"tested N : {tested_N}\nhalf-way (log scale) untested N : {mid_N}")

    levels = dict(sigma_rad=g_sr[list(GRID_LEVEL_IDX)], sigma_S=g_sS[list(GRID_LEVEL_IDX)], mu_L=g_mL[list(GRID_LEVEL_IDX)],
                  sigma_eff_frac=np.array(SPREAD_FRACTIONS))
    assert np.all(levels["mu_L"] > 0.0), "mu_L levels must be strictly positive (sigma_top(0) = 0)"
    grid_combos, grid_levels, grid_fractions = grid_test_points(levels)
    return SimpleNamespace(
        parameters=parameters, fit={k: list(v.values()) for k, v in parameters.items()}, grid=dict(sigma_rad=g_sr, sigma_S=g_sS, mu_L=g_mL), levels=levels, tested_N=tested_N, mid_N=mid_N,
        measured=dict(sigma_rad=measured_levels(d_sr, "sigma_rad", tested_N), sigma_S=measured_levels(d_sS, "sigma_S", tested_N),
                      mu_L=measured_levels(d_mL, "mu_L", tested_N)),
        sweep_mu_L=read_spread_sweep_mu_L(folders["sigma_L"]),
        grid_combos=grid_combos, grid_levels=grid_levels, grid_fractions=grid_fractions)


# ======================================================================================
# 3. THE TWO PREDICTIONS
# ======================================================================================

def channel_amplitudes(fit, sr, sS, mL, sE, n, with_sigma=True):
    """u_phi, u_S, u_L, u_sigma of the four channels at network size n.  sE = EFFECTIVE loss spread sigma_eff (dB), used as is (no conversion)."""
    (sp0, pp), (sS0, pS), (sL0, pL, g0, qL), (sG0, pG) = fit["sigma_rad"], fit["sigma_S"], fit["mu_L"], fit["sigma_L"]
    f = n / N0
    us = sG0 * f ** pG * sE / SQRT2 if with_sigma else 0.0
    return sp0 * f ** pp * sr / SQRT2, sS0 * f ** pS * sS / SQRT2, (sL0 * f ** pL * mL + g0 * f ** qL * mL ** 2) / SQRT2, us


def fitted_sum(fit, sr, sS, mL, sE, n, with_sigma=True):
    """Prediction of the four fitted functions: (E_rms, I_ph).  with_sigma=False leaves out the spread term."""
    I = sum(u ** 2 for u in channel_amplitudes(fit, sr, sS, mL, sE, n, with_sigma))
    return SQRT2 * np.sqrt(1 - np.exp(-I)), I


def measured_sum(e_phi, e_S, e_L, us=0.0):
    """Prediction from MEASURED single-channel errors, combined as multiplicative overlaps c = 1 - E^2/2: (E_rms, I_ph).
    us = u_sigma of the spread (from the fit); 0 if there is no spread."""
    c = np.prod([np.clip(1 - e ** 2 / 2, 1e-12, 1.0) for e in (e_phi, e_S, e_L)]) * np.exp(-us ** 2)
    return SQRT2 * np.sqrt(1 - c), -np.log(c)


# ======================================================================================
# 4. SIMULATION OF THE COMBINED CIRCUIT
# ======================================================================================
SIM_CFG = dict(CHIPS_PER_MATRIX=CHIPS, INPUTS_PER_TYPE=INPUTS_PER_TYPE, INPUT_TYPES=INPUT_TYPES)


def cache_hash(kind, n, combos):
    """Fingerprint of all settings of a simulation: a cache file is only reused if it carries the same fingerprint."""
    settings = [kind, int(n), N_MATRICES, CHIPS, INPUTS_PER_TYPE, SEED, INPUT_TYPES,
                np.round(combos, 12).tolist()]
    return hashlib.sha1(json.dumps(settings).encode()).hexdigest()[:16]


def simulate(kind, n, combos):
    """combos: (K, 4) array of (sigma_rad, sigma_S, mu_L, sigma_eff).  Returns (m1, m2), both (K, M, T): per unitary and input type the
    chip-mean of E and of E^2 (E^2 computed directly per chip and input); E_rms = sqrt(mean m2), E_avg = mean m1 (diagnostic)."""
    h = cache_hash(kind, n, combos)
    cache_file = OUT / f"cache_{kind}_N{n}.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        if str(z["h"]) == h:
            print(f"  [{kind}] N={n}: loaded cache", flush=True)
            return z["m1"], z["m2"]
    m1 = np.zeros((len(combos), N_MATRICES, N_INPUT_TYPES))
    m2 = np.zeros((len(combos), N_MATRICES, N_INPUT_TYPES))
    t0 = time.perf_counter()
    for mi in range(N_MATRICES):
        b = sweep_engine.build_matrix(n, SIM_CFG, np.random.default_rng([SEED, 0, n, mi]))
        rng = np.random.default_rng([SEED, 7000 + (0 if kind == "grid" else 1), n, mi])
        for ci, (sr, sS, mL, sE) in enumerate(combos):
            out = mesh.run_chips(b.dec, b.X3, CHIPS, sigma_rad=float(sr), sigma_S=float(sS), mu_L=float(mL),
                                 sigma_L=0.0, rng=rng, Y_ideal=b.Y3, sigma_eff=float(sE))
            m1[ci, mi] = out["E_raw"].mean(axis=0)
            m2[ci, mi] = out["E2_raw"].mean(axis=0)
        print(f"  [{kind}] N={n}  unitary {mi + 1}/{N_MATRICES}  [{time.perf_counter() - t0:6.1f}s]", flush=True)
    np.savez_compressed(cache_file, m1=m1, m2=m2, h=np.array(h))
    return m1, m2


# ======================================================================================
# 5. TEST POINTS
# ======================================================================================

def grid_test_points(levels):
    """The 81 combinations of the grid test: low / middle / high of (phase, split, mean loss, spread fraction).

    Returns (combos (81, 4) of sigma_rad, sigma_S, mu_L, sigma_eff;  level indices (i, j, k, l) of every combination;  spread fraction f of every combination)."""
    level_indices = list(itertools.product(range(3), repeat=4))          # (i, j, k, l) levels of (phase, split, mean loss, spread fraction)
    combos = np.array([[levels["sigma_rad"][i], levels["sigma_S"][j], levels["mu_L"][k],
                        levels["sigma_eff_frac"][l] * float(mesh.sigma_eff_top(levels["mu_L"][k]))] for i, j, k, l in level_indices])
    fractions = np.array([levels["sigma_eff_frac"][l] for i, j, k, l in level_indices])
    return combos, level_indices, fractions


def random_value_in_stratum(rng, grid_values, stratum, name):
    """A random value inside a sweep interval of the stratum (never on a grid point).  Two random draws: interval, then position."""
    k0, k1 = STRATA_IDX[stratum]
    k = int(rng.integers(k0, k1 + 1))
    u = rng.uniform(*POS_IN_INTERVAL)
    value = grid_values[k] + u * (grid_values[k + 1] - grid_values[k])
    assert np.min(np.abs(grid_values - value)) > 1e-9 * max(1.0, abs(value)), f"{name}: random value coincides with a sweep grid point"
    return value


def random_combos(n, setup):
    """81 strata (low/mid/high of every channel); one random draw per stratum.  sigma_rad, sigma_S, mu_L: inside a sweep interval of
    their stratum (relative position POS_IN_INTERVAL, never on a measured grid point; mu_L > 0).  Then top = sigma_top(mu_L) and the
    fraction f uniformly in the open interior of its stratum; sigma_eff = f * top.  Returns (combos (81, 4), strata, f (81,))."""
    rng = np.random.default_rng([SEED, 99, int(n)])
    out, fractions = [], []
    for strata in itertools.product(range(3), repeat=4):
        row = []
        for name, stratum in zip(LEVEL_NAMES[:3], strata[:3]):
            row.append(random_value_in_stratum(rng, setup.grid[name], stratum, name))
        assert row[2] > 0.0
        assert setup.sweep_mu_L.size == 0 or np.min(np.abs(setup.sweep_mu_L - row[2])) > 1e-9, "random mu_L coincides with a spread-sweep mu_L"
        lo, hi = SPREAD_STRATA_F[strata[3]]
        f = float(rng.uniform(lo, hi))
        assert lo < f < hi and np.min(np.abs(np.array(SPREAD_FRACTIONS) - f)) > 1e-9, "f must lie strictly inside its stratum and off the grid levels"
        row.append(f * float(mesh.sigma_eff_top(row[2])))
        out.append(row)
        fractions.append(f)
    return np.array(out), list(itertools.product(range(3), repeat=4)), np.array(fractions)


def in_typical(sr, sS, mL, sE):
    """Is the point inside the literature-typical error box?  (judged on the simulated values; the sigma_eff threshold is literal)"""
    return (sr <= TYPICAL_BOX["sigma_rad"] and sS <= TYPICAL_BOX["sigma_S"] and mL <= TYPICAL_BOX["mu_L"]
            and sE <= TYPICAL_BOX["sigma_eff"] + 1e-12)


def reachability_check(setup):
    """Every requested sigma_eff must be <= sigma_top(mu_L) (reachable by construction): asserts that 0 combinations are unreachable."""
    n_unreachable, total = 0, dict(grid=len(setup.grid_combos) * len(setup.tested_N), random=0)
    n_unreachable += int((setup.grid_combos[:, 3] > mesh.sigma_eff_top(setup.grid_combos[:, 2]) + 1e-12).sum())
    for n in setup.mid_N:
        combos, _, _ = random_combos(n, setup)
        total["random"] += len(combos)
        n_unreachable += int((combos[:, 3] > mesh.sigma_eff_top(combos[:, 2]) + 1e-12).sum())
    assert n_unreachable == 0, f"{n_unreachable} unreachable sigma_eff requests: the spread axis must be defined relative to sigma_top(mu_L)"
    print(f"reachability: 0 of {total['grid']} grid and 0 of {total['random']} random combinations exceed sigma_top(mu_L) = min(0.95 mu_L/sqrt3, 0.263 dB) "
          f"(asserted; no clipping)")
    return total


# ======================================================================================
# 6. ONE TABLE ROW PER TEST POINT
# ======================================================================================

def make_row(kind, n, ci, combo, level, fraction, stats, m2, setup):
    """All numbers of one test point: error values, level indices, simulation result and predictions."""
    sr, sS, mL, sE = combo
    sim_rms, sim_sd, sim_se, sim_avg = stats["E_rms"], stats["sd_E_rms"], stats["se_E_rms"], stats["E_avg"]
    e_fit, i_fit = fitted_sum(setup.fit, sr, sS, mL, sE, n)
    e_no_sigma, i_no_sigma = fitted_sum(setup.fit, sr, sS, mL, sE, n, False)
    row = dict(N=n, sigma_rad=sr, sigma_S=sS, mu_L=mL, sigma_eff_frac=float(fraction), sigma_eff_top_dB=float(mesh.sigma_eff_top(mL)),
               sigma_eff_dB=sE, sigma_nominal_dB=mesh.sigma_nominal_from_eff(float(mL), float(sE)),
               lvl_rad=level[0], lvl_S=level[1], lvl_L=level[2], lvl_seff=level[3], typical_box=in_typical(sr, sS, mL, sE),
               E_rms_sim=sim_rms[ci], sd_E_rms_sim=sim_sd[ci], se_E_rms_sim=sim_se[ci], E_avg_sim=sim_avg[ci],
               E_rms_pred_fit=e_fit, I_fit=i_fit, E_rms_pred_fit_no_sigma=e_no_sigma, I_fit_no_sigma=i_no_sigma)
    if kind == "grid":
        i, j, k, l = level
        measured = setup.measured
        if SPREAD_FRACTIONS[l] == 0.0:               # sigma_eff = 0: loss block fully measured (deterministic loss mu_L)
            e_csv, i_csv = measured_sum(measured["sigma_rad"][n][i], measured["sigma_S"][n][j], measured["mu_L"][n][k])
        else:                                        # measured mean loss (sigma_eff = 0) x fitted spread overlap at sigma_eff = f * sigma_top
            e_csv, i_csv = measured_sum(measured["sigma_rad"][n][i], measured["sigma_S"][n][j], measured["mu_L"][n][k],
                                        channel_amplitudes(setup.fit, sr, sS, mL, sE, n)[3])
        row.update(E_rms_pred_csv=e_csv, I_csv=i_csv)
    for ti, type_name in enumerate(INPUT_TYPES):
        row[f"E_rms_sim_{type_name}"] = float(np.sqrt(m2[ci, :, ti].mean()))
    return row


def build_rows(kind, setup):
    """Simulate every test point of the kind ("grid" or "random") and return the table (one row per point and N)."""
    rows = []
    for n in (setup.tested_N if kind == "grid" else setup.mid_N):
        if kind == "grid":
            combos, levels, fractions = setup.grid_combos, setup.grid_levels, setup.grid_fractions
        else:
            combos, levels, fractions = random_combos(n, setup)
        assert np.all(combos[:, 3] <= mesh.sigma_eff_top(combos[:, 2]) + 1e-12)      # reachable by construction
        m1, m2 = simulate(kind, n, combos)                   # (K, M, T)
        stats = sweep_engine.rms_statistics(m1.mean(axis=2).T, m2.mean(axis=2).T)      # per combination: E_rms, sd / se across unitaries, E_avg, ...
        for ci, (combo, level) in enumerate(zip(combos, levels)):
            rows.append(make_row(kind, n, ci, combo, level, fractions[ci], stats, m2, setup))
    return pd.DataFrame(rows)


# ======================================================================================
# 7. AGREEMENT METRICS
# ======================================================================================

def agreement_metrics(y, p):
    """How well do the predictions p describe the simulated values y?  RMSE, NRMSE (range-normalised), MAE, R2, Pearson r,
    MARE (mean relative error, y > 1e-3), MaxAbsDev, MeanBias."""
    r = y - p
    big = y > 1e-3
    return dict(n=int(y.size), RMSE=float(np.sqrt(np.mean(r ** 2))),
                NRMSE=float(np.sqrt(np.mean(r ** 2)) / max(y.max() - y.min(), 1e-12)), MAE=float(np.mean(np.abs(r))),
                R2=float(1 - np.sum(r ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)),
                Pearson=float(np.corrcoef(y, p)[0, 1]) if y.size > 2 else float("nan"),
                MARE=float(np.mean(np.abs(r[big]) / y[big])), MaxAbsDev=float(np.max(np.abs(r))), MeanBias=float(np.mean(r)))


def verdict(m):
    """Words for the metrics m, with the guideline thresholds printed in the figure."""
    if m["NRMSE"] < .03 and m["R2"] > .98 and m["MARE"] < .05:
        return "strong agreement"
    if m["NRMSE"] < .05 and m["R2"] > .95 and m["MARE"] < .10:
        return "quantitatively useful"
    return "thresholds not met"


def point_subsets(D):
    """Boolean masks of the groups of points for which metrics are reported."""
    return dict(all=np.ones(len(D), bool), typical_box=D.typical_box.values, beyond_typical=~D.typical_box.values,
                mu_low=D.lvl_L.values == 0, mu_mid=D.lvl_L.values == 1, mu_high=D.lvl_L.values == 2,        # per-mu_L-stratum breakdown
                mu_ge_0p3=D.mu_L.values >= 0.3)                                                           # spread channel meaningfully exercised


def metrics_per_subset(y, p, subsets, keep):
    """agreement_metrics for every subset, restricted to the points in `keep`; subsets with at most 3 points only report n."""
    result = {}
    for name, mask in subsets.items():
        chosen = mask & keep
        result[name] = agreement_metrics(y[chosen], p[chosen]) if chosen.sum() > 3 else dict(n=int(chosen.sum()))
    return result


def all_metrics(D, col):
    """Metrics ONLY on points with simulated E_rms < E_CUT (NRMSE normalised by the range of the RETAINED simulated E_rms).  Also stored:
    'all_points_reference_only' (including the plateau) and 'cut_on_predicted' (robustness: cut applied to the predicted E_rms instead)."""
    y, p = D.E_rms_sim.values, D[col].values
    subsets = point_subsets(D)
    keep_simulated, keep_predicted = y < E_CUT, p < E_CUT
    return dict(cut_on_simulated=metrics_per_subset(y, p, subsets, keep_simulated),
                all_points_reference_only=metrics_per_subset(y, p, subsets, np.ones(len(D), bool)),
                cut_on_predicted=metrics_per_subset(y, p, subsets, keep_predicted),
                n_retained=int(keep_simulated.sum()), n_excluded=int((~keep_simulated).sum()), n_total=int(len(D)))


def counts_per_N(D):
    """Retained / excluded points per N (simulated E_rms < E_CUT): at large N most mid/high-error combinations exceed the cut."""
    rows = []
    for n, q in D.groupby("N"):
        keep = q.E_rms_sim.values < E_CUT
        rows.append(dict(N=int(n), n_total=int(len(q)), n_retained=int(keep.sum()), n_excluded=int((~keep).sum()), frac_retained=float(keep.mean()),
                         n_retained_mu_ge_0p3=int((keep & (q.mu_L.values >= 0.3)).sum()), n_retained_typical_box=int((keep & q.typical_box.values).sum())))
    return pd.DataFrame(rows)


# ======================================================================================
# 8. FIGURE
# ======================================================================================

def draw_agreement_panels(ax_top, ax_residual, D, marker):
    """Simulated against predicted E_rms (top) and the residual simulated - predicted with +-1 s.e. bars (bottom).
    Colour = network size; grey open markers = points with simulated E_rms >= E_CUT (excluded from the metrics)."""
    sizes = sorted(D.N.unique())
    colours = {n: plt.cm.viridis(v) for n, v in zip(sizes, np.linspace(0.0, 0.92, len(sizes)))}
    predicted, simulated, se = D.E_rms_pred_fit.values, D.E_rms_sim.values, D.se_E_rms_sim.values
    residual = simulated - predicted
    kept = simulated < E_CUT
    for n in sizes:
        this_size = (D.N == n).values & kept
        ax_top.plot(predicted[this_size], simulated[this_size], marker, ms=3, color=colours[n], mew=0)
        ax_residual.errorbar(predicted[this_size], residual[this_size], yerr=se[this_size], fmt=marker, ms=3, color=colours[n],
                             mew=0, elinewidth=0.6, capsize=1.2)
    ax_top.plot(predicted[~kept], simulated[~kept], marker, ms=3.5, mfc="none", mec="0.6", mew=0.7)
    ax_residual.plot(predicted[~kept], residual[~kept], marker, ms=3.5, mfc="none", mec="0.6", mew=0.7)

    limit = max(1.5, 1.08 * max(predicted.max(), simulated.max()))
    ax_top.plot([0, limit], [0, limit], "k-", lw=0.8, zorder=1)
    ax_top.axhline(E_CUT, color="0.5", ls=":", lw=0.8)
    ax_top.text(0.02, E_CUT + 0.02, r"$E_{\mathrm{rms}}=%g$" % E_CUT, color="0.45", fontsize=7)
    ax_top.set_xlim(0, limit)
    ax_top.set_ylim(0, limit)
    ax_residual.axhline(0.0, color="k", lw=0.8)
    largest = 1.15 * np.max(np.abs(residual) + se)
    ax_residual.set_ylim(-largest, largest)
    ax_residual.set_xlabel(r"predicted $E_{\mathrm{rms}}$")


def make_agreement_figure(DG, DR):
    """The validation figure: predicted (fitted four-channel sum) against simulated E_rms for the tested sizes (left, circles)
    and the untested sizes (right, diamonds); top row the agreement, bottom row the residual."""
    shared.set_paper_style()
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex="col", gridspec_kw=dict(height_ratios=[3.4, 1], hspace=0.08, wspace=0.26))
    for column, (D, marker) in enumerate(((DG, "o"), (DR, "D"))):
        draw_agreement_panels(axs[0, column], axs[1, column], D, marker)
        shared.label_panel(axs[0, column], "ab"[column])
        shared.label_panel(axs[1, column], "cd"[column])
    axs[0, 0].set_ylabel(r"simulated $E_{\mathrm{rms}}$")
    axs[1, 0].set_ylabel("residual")
    return fig


# ======================================================================================
# 9. REPORT AND DRIVER
# ======================================================================================

def print_report(DG, DR, metrics, counts, setup):
    """Print the agreement metrics of both tests."""
    for kind, D in (("grid", DG), ("random", DR)):
        print(f"\n=== {kind.upper()}: {len(D)} points, E_rms,sim < {E_CUT:g} retained: {int((D.E_rms_sim < E_CUT).sum())}, excluded: {int((D.E_rms_sim >= E_CUT).sum())} ===")
        print("  retained / total per N: " + "  ".join(f"{int(r.N)}: {int(r.n_retained)}/{int(r.n_total)}" for r in counts[kind].itertuples()))
        for pred, mm_ in metrics[kind].items():
            for blk, lab in (("cut_on_simulated", "retained (sim cut)"), ("cut_on_predicted", "pred-side cut   "), ("all_points_reference_only", "ALL pts (ref only)")):
                for sub in ("all", "typical_box", "mu_ge_0p3"):
                    m = mm_[blk][sub]
                    if "RMSE" in m:
                        print(f"  {pred:13s} {lab} {sub:11s} n={m['n']:4d} RMSE={m['RMSE']:.4f} NRMSE={100 * m['NRMSE']:.2f}% R2={m['R2']:.4f} "
                              f"MARE={100 * m['MARE']:.2f}% max|dE|={m['MaxAbsDev']:.3f} bias={m['MeanBias']:+.4f}" + (f"  -> {verdict(m)}" if blk == "cut_on_simulated" else ""))
        R_ = D[D.E_rms_sim < E_CUT]
        print("  per-N RMSE (fit sum, retained):", {int(n): round(float(np.sqrt(np.mean((s_.E_rms_sim - s_.E_rms_pred_fit) ** 2))), 4) for n, s_ in R_.groupby("N")})
        print("  per-mu_L-stratum RMSE / bias (fit sum, retained):", {nm: (round(float(np.sqrt(np.mean((s_.E_rms_sim - s_.E_rms_pred_fit) ** 2))), 4), round(float(np.mean(s_.E_rms_sim - s_.E_rms_pred_fit)), 4))
                                                                  for nm, s_ in ((n_, R_[R_.lvl_L == i_]) for i_, n_ in enumerate(("low", "mid", "high"))) if len(s_)})
        q3 = R_[R_.mu_L >= 0.3]
        print(f"  NOTE: a third of the combinations (low mu_L stratum) has negligible spread (<= {mesh.sigma_eff_top(setup.levels['mu_L'][0]):.4f} dB on the grid): "
              f"the spread channel is meaningfully exercised only at mu_L >~ 0.3 dB.")
        if len(q3):
            print(f"  mu_L >= 0.3 dB ({len(q3)} retained pts): RMSE fit sum = {np.sqrt(np.mean((q3.E_rms_sim - q3.E_rms_pred_fit) ** 2)):.4f}  vs  spread term removed = "
                  f"{np.sqrt(np.mean((q3.E_rms_sim - q3.E_rms_pred_fit_no_sigma) ** 2)):.4f}")
        print("  NOTE: at large N most mid/high-error combinations exceed the cut: the large-N claim rests on the low-error combinations (see counts above).")


def write_metrics_json(metrics, counts, unreachable, setup):
    """total_error_validation_metrics.json: settings, fitted parameters, metrics and counts."""
    report = dict(settings=dict(N_MATRICES=N_MATRICES, CHIPS=CHIPS, INPUT_TYPES=INPUT_TYPES, INPUTS_PER_TYPE=INPUTS_PER_TYPE, SEED=SEED,
                                 N_tested=setup.tested_N, N_midpoints=setup.mid_N, sigma_eff_overlap_from_fit_in_measured_sum=SPREAD_FROM_FIT, E_cut=E_CUT, typical_box=TYPICAL_BOX,
                                 loss_spread_variable="sigma_eff (realised std of L, dB)",
                                 spread_design="sigma_eff = f * sigma_top(mu_L), sigma_top = min(0.95*mu_L/sqrt(3), 0.263 dB)",
                                 spread_fractions_grid=list(SPREAD_FRACTIONS), spread_strata_f_random=[list(v) for v in SPREAD_STRATA_F],
                                 unreachable_combinations=0, n_combinations_checked=unreachable,
                                 grid_levels={c: [float(v) for v in setup.levels[c]] for c in LEVEL_NAMES}),
                   parameters=setup.parameters, metrics=metrics, counts_per_N={k: v.to_dict(orient="records") for k, v in counts.items()})
    with open(OUT / "total_error_validation_metrics.json", "w") as f:
        json.dump(report, f, indent=1)


def main():
    """Simulate the grid and random tests, compare with the fitted sum, write tables, metrics and the figure."""
    t0 = time.perf_counter()
    setup = read_setup()
    unreachable = reachability_check(setup)
    DG = build_rows("grid", setup)
    DG.to_csv(OUT / "total_error_validation_grid_points.csv", index=False)
    DR = build_rows("random", setup)
    DR.to_csv(OUT / "total_error_validation_random_points.csv", index=False)

    metrics = dict(grid=dict(csv=all_metrics(DG, "E_rms_pred_csv"), fit=all_metrics(DG, "E_rms_pred_fit"), fit_no_sigma=all_metrics(DG, "E_rms_pred_fit_no_sigma")),
                   random=dict(fit=all_metrics(DR, "E_rms_pred_fit"), fit_no_sigma=all_metrics(DR, "E_rms_pred_fit_no_sigma")))
    counts = {k: counts_per_N(D) for k, D in (("grid", DG), ("random", DR))}
    for k, c in counts.items():
        c.to_csv(OUT / f"total_error_validation_{k}_counts_per_N.csv", index=False)
    print_report(DG, DR, metrics, counts, setup)
    write_metrics_json(metrics, counts, unreachable, setup)

    fig = make_agreement_figure(DG, DR)
    fig.savefig(OUT / "total_error_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved total_error_validation.pdf")
    print(f"\nall files in {OUT}   (total {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
