"""
5_fidelity_validation.py  --  does the Photonic Infidelity Index determine the matrix fidelity?  (Clements unitary mesh)
=======================================================================================================================

Per chip:  F = |Tr(U^dag V)|^2 / (N Tr V^dag V),   a = Re Tr(U^dag V)/N,   b = Im Tr(U^dag V)/N   (stored per chip as scalars).
All E-based quantities are E_rms = sqrt(mean E^2 over unitaries, chips, inputs);  I_ph = -ln(1 - E_rms^2/2).
Theory: <F> = <a>^2 + Var(a) + <b^2> = exp(-2 I_tau) + Gamma for unitary V (I_tau = -ln<a>); floor-aware formula
F ~ 1/N^2 + (1 - 1/N^2) exp(-2 I_ph).  Identity (lossless chips, asserted): mean F = mean(a)^2 + Var_a + mean b^2.

TEST 1  independence of the error mixture, at every N in MIXTURE_N_LIST (53, 100):
    sets A phase | B splitting | C phase+splitting (equal share) | D four channels (25 % loss, pair scaled from LOSS_REF, sigma_eff <= sigma_top),
    20 targets log-spaced in [0.01, 0.9997] + high targets from 1 to 0.95 x (largest reachable I_ph), same unitaries for all sets.
    A target above 0.95 x a set's maximum is SKIPPED (recorded, never clipped).
TEST 2  floor-aware formula vs N on the 11 calibration + 10 held-out sizes, levels s = 1 / 0.5 / 0.1 of (0.25 rad, 0.1, 1 dB, 0.263 dB),
    with loss (first) and lossless (second; tests the derivation itself).  Held-out sizes use their own master seed.
Figures: F_vs_PII_N<n>.pdf (Test 1, one per N: F against the realised I_ph for the four error sets, and the ratio to exp(-2 I_ph)) and
F_vs_N_floor.pdf (Test 2: F against N at the middle and worst level with loss, and the ratio of the simulation to the formulas).
Error bars: +-1 standard error of F across unitaries.

Needs (all next to this script): results_phase_error / splitting_ratio_error / mean_loss / loss_spread  fit_results_*.json.

    python 5_fidelity_validation.py             # everything  -> results_fidelity_vs_pii/   (resumable per unitary / per N)
    python 5_fidelity_validation.py --quick     # smoke test (reads results_*_quick) -> results_fidelity_vs_pii_quick/
    python 5_fidelity_validation.py --test1 | --test2 | --selftest     # only Test 1 / only Test 2 / bookkeeping self-tests

The file is organised like a book:
    1 settings   2 the fitted surrogate   3 error sets and reachable targets   4 simulation   5 statistics of one point
    6 analysis and tables   7 figures   8 self-tests   9 driver
"""
import csv
import hashlib
import importlib
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # so the numbered modules can be imported
shared = importlib.import_module("0_shared_settings")  # noqa: E402
mesh = importlib.import_module("1_mzi_mesh")  # noqa: E402
sweep_engine = importlib.import_module("2_common_sweep_engine")  # noqa: E402


# ======================================================================================
# 1. SETTINGS
# ======================================================================================
# --- Test 1 ---
MIXTURE_N_LIST = (53, 100)       # network sizes of the mixture test (calibration sizes; 20 may be added)
N_PII = 20                       # low targets, log-spaced in [PII_MIN, PII_MAX]
PII_MIN, PII_MAX = 0.01, 0.9997
N_PII_HIGH = 7                   # extra targets from I_ph = 1 to 0.95 x (largest reachable I_ph over the four sets)
N_UNITARIES = 40                 # Haar targets (same for every set / I_ph value)
CHIPS = 40
LOSS_SHARE = 0.25                # set D: share of I_ph taken by the loss channels (capped at the loss of LOSS_REF)
LOSS_REF = (1.0, 0.263)          # (mu_L dB, sigma_eff dB); the pair is scaled together; sigma_eff <= sigma_eff_top(mu_L) asserted
# --- Test 2 ---
CAL_N = (20, 28, 38, 53, 72, 100, 138, 190, 263, 362, 500)
HELD_OUT_N = (24, 33, 45, 62, 85, 117, 162, 224, 309, 425)
LAST_N = (425, 500)              # run last (most expensive)
TEST2_UNITARIES = 10
TEST2_CHIPS = 40                  # >= 400 chips per point
TEST2_MAX_VECTOR = (0.25, 0.1, 1.0, 0.263)      # (sigma_theta rad, sigma_S, mu_L dB, sigma_eff dB)
TEST2_LEVELS = {"worst": 1.0, "mid": 0.5, "low": 0.1}
# --- both tests ---
INPUT_TYPES = list(sweep_engine.ALL_INPUT_TYPES)   # all input families, equal weight (a subset can be given here)
INPUTS_PER_TYPE = 10
SEED = 20261013                  # Test 1 (distinct from calibration 12345, validation 20261011)
SEED_CAL = 20261014              # Test 2, calibration sizes
SEED_HELD = 20261017             # Test 2, held-out sizes (own master seed)
N0, SQRT2 = shared.N0, shared.SQRT2
CHANNELS = ("sigma_rad", "sigma_S", "mu_L", "sigma_L")    # the four single-channel fits

# --- what to run (command line) ---
QUICK = "--quick" in sys.argv
_only = [flag for flag in ("--test1", "--test2", "--selftest") if flag in sys.argv]
RUN_T1 = (not _only) or "--test1" in _only
RUN_T2 = (not _only) or "--test2" in _only
RUN_SELFTEST = "--selftest" in _only
if QUICK:
    MIXTURE_N_LIST, N_PII, N_PII_HIGH, N_UNITARIES, CHIPS = (38,), 6, 3, 3, 8
    CAL_N, HELD_OUT_N, LAST_N, TEST2_UNITARIES, TEST2_CHIPS = (20, 38), (28,), (38,), 3, 8
OUT = HERE / ("results_fidelity_vs_pii_quick" if QUICK else "results_fidelity_vs_pii")
OUT.mkdir(exist_ok=True)

# --- the four error sets of Test 1 ---
SETS = ["A", "B", "C", "D"]
SET_NAME = {"A": "A: phase only", "B": "B: splitting only", "C": "C: phase + splitting", "D": "D: all four channels"}
SET_MARKER = {"A": "o", "B": "s", "C": "^", "D": "D"}
SET_COL = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#e377c2"}


# ======================================================================================
# 2. THE FITTED SURROGATE (read from the fit results of the single channels)
# ======================================================================================

def find_dir(name):
    """A result folder of a previous step (they all live next to this script)."""
    if not (HERE / name).is_dir():
        raise SystemExit(f"folder '{name}' not found in {HERE}: run the previous steps (2a-2d, 3a-3d) first")
    return HERE / name


def read_fit_results():
    """Fitted parameters and fitted small range of every channel, from fit_results_<channel>.json.

    Returns ({channel: {parameter name: value}},  {channel: upper end of the fitted small range})."""
    parameters, fit_range = {}, {}
    for channel in CHANNELS:
        path = find_dir(sweep_engine.RESULT_FOLDER[channel] + ("_quick" if QUICK else "")) / f"fit_results_{channel}.json"
        if not path.exists():
            raise SystemExit(f"missing file: {path}\n(run the fit scripts first)")
        with open(path) as f:
            fit = json.load(f)
        parameters[channel] = fit["params"]
        fit_range[channel] = float(fit["fit_range"][1])      # upper end of the fitted small range
    print("fitted small ranges (upper ends):", fit_range)
    return parameters, fit_range


FIT_PARAMS, FIT_RANGE = read_fit_results()
(S_PHI0, P_PHI), (S_SPLIT0, P_SPLIT), (S_LOSS0, P_LOSS, GAMMA0, Q_LOSS), (S_SPREAD0, P_SPREAD) = [list(FIT_PARAMS[k].values()) for k in CHANNELS]


# ======================================================================================
# 3. ERROR SETS AND REACHABLE TARGETS
# ======================================================================================
# An error vector is (sigma_rad, sigma_S, mu_L, sigma_eff).  The surrogate gives its I_ph; the four sets of Test 1
# are different ways of spending a given target I_ph on the channels.

def phase_and_split_slopes(n):
    """s_phi(N) and s_S(N): sensitivity slopes of the phase and splitting channels at network size n."""
    f = n / N0
    return S_PHI0 * f ** P_PHI, S_SPLIT0 * f ** P_SPLIT


def loss_and_spread_terms(mu, sigma_eff, n):
    """(u_L^2, u_sigma^2): mean-loss part and spread part of I_ph.  sigma_eff = EFFECTIVE loss spread (dB), used as is (no conversion)."""
    f = n / N0
    u_loss = (S_LOSS0 * f ** P_LOSS * mu + GAMMA0 * f ** Q_LOSS * mu ** 2) / SQRT2
    u_spread = S_SPREAD0 * f ** P_SPREAD * sigma_eff / SQRT2
    return u_loss ** 2, u_spread ** 2


def surrogate_I_ph(par, n):
    """Surrogate I_ph (sum of the four channel contributions, E_rms fits) for (sigma_rad, sigma_S, mu_L, sigma_eff) at network size n."""
    sr, sS, mu, sg = par
    s_phi, s_spl = phase_and_split_slopes(n)
    uL2, uS2 = loss_and_spread_terms(mu, sg, n)
    return (s_phi * sr / SQRT2) ** 2 + (s_spl * sS / SQRT2) ** 2 + uL2 + uS2


def loss_I_ph(lam, n):
    """I_ph of the loss channels alone, with (mu_L, sigma_eff) = lam * LOSS_REF."""
    return sum(loss_and_spread_terms(lam * LOSS_REF[0], lam * LOSS_REF[1], n))


def error_vector_for_set(s, I, n):
    """The error vector that gives I_ph = I at size n for set s ("A" phase, "B" splitting, "C" both equally, "D" all four channels)."""
    s_phi, s_spl = phase_and_split_slopes(n)
    if s == "A":
        return (np.sqrt(2 * I) / s_phi, 0.0, 0.0, 0.0)
    if s == "B":
        return (0.0, np.sqrt(2 * I) / s_spl, 0.0, 0.0)
    if s == "C":
        return (np.sqrt(I) / s_phi, np.sqrt(I) / s_spl, 0.0, 0.0)
    cap = loss_I_ph(1.0, n)
    I_loss = min(LOSS_SHARE * I, cap)
    lam = brentq(lambda lam_: loss_I_ph(lam_, n) - I_loss, 1e-12, 1.0) if I_loss < cap else 1.0
    remainder = I - loss_I_ph(lam, n)
    return (np.sqrt(remainder) / s_phi, np.sqrt(remainder) / s_spl, lam * LOSS_REF[0], lam * LOSS_REF[1])


def inside_calibration_ranges(par):
    """Every error inside its fitted small range and sigma_eff reachable (<= sigma_eff_top(mu_L))."""
    ok = all(v <= FIT_RANGE[k] * (1 + 1e-9) for v, k in zip(par, CHANNELS))
    return bool(ok and par[3] <= float(mesh.sigma_eff_top(par[2])) + 1e-12)


def largest_reachable_I_ph(s, n):
    """Largest I_ph that set s reaches at size n with all its errors inside their calibration ranges (bisection on the construction itself)."""
    hi = 1.0
    while inside_calibration_ranges(error_vector_for_set(s, hi, n)) and hi < 1e3:
        hi *= 2.0
    lo = 1e-6
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if inside_calibration_ranges(error_vector_for_set(s, mid, n)):
            lo = mid
        else:
            hi = mid
    return lo


def targets_for_size(n):
    """(targets T, reach mask (S, P), I_max per set).  Low targets log-spaced in [PII_MIN, PII_MAX]; high targets from 1 to 0.95 x (largest set maximum).
    A target above 0.95 x a set's maximum is SKIPPED for that set (recorded; never clipped)."""
    imax = {s: largest_reachable_I_ph(s, n) for s in SETS}
    top = 0.95 * max(imax.values())
    high = np.geomspace(1.0, top, N_PII_HIGH) if top > 1.0 and N_PII_HIGH > 0 else np.array([])
    T = np.concatenate([np.geomspace(PII_MIN, PII_MAX, N_PII), high])
    reach = np.array([[t <= 0.95 * imax[s] for t in T] for s in SETS])
    return T, reach, imax


def build_mixture(n):
    """All error vectors of Test 1 at size n: PAR[set, target] = error vector (NaN where the target is skipped)."""
    T, reach, imax = targets_for_size(n)
    PAR = np.full((len(SETS), len(T), 4), np.nan)
    for si, s in enumerate(SETS):
        for pi, t in enumerate(T):
            if reach[si, pi]:
                p = error_vector_for_set(s, t, n)
                assert inside_calibration_ranges(p), f"N={n} set {s} target {t:.4g}: error outside its calibration range"
                assert abs(surrogate_I_ph(p, n) / t - 1) < 1e-6, f"set {s} misses its target I_ph"
                PAR[si, pi] = p
    skipped = [dict(N=n, set=s, target=float(T[pi])) for si, s in enumerate(SETS) for pi in range(len(T)) if not reach[si, pi]]
    return dict(n=n, T=T, reach=reach, imax=imax, PAR=PAR, skipped=skipped)


def test2_level_parameters():
    """Test 2 levels: fractions s of the maximum error vector, with loss ("four_channel") and without ("lossless")."""
    level_par = {}
    for level, s in TEST2_LEVELS.items():
        level_par[("four_channel", level)] = tuple(s * v for v in TEST2_MAX_VECTOR)
        level_par[("lossless", level)] = (s * TEST2_MAX_VECTOR[0], s * TEST2_MAX_VECTOR[1], 0.0, 0.0)
    return level_par


LEVEL_PAR = test2_level_parameters()


def check_settings():
    """The reference loss pair and all Test 2 levels must lie inside the calibration ranges."""
    assert LOSS_REF[0] <= FIT_RANGE["mu_L"] + 1e-12 and LOSS_REF[1] <= FIT_RANGE["sigma_L"] + 1e-12, "LOSS_REF is outside the fitted ranges"
    assert LOSS_REF[1] <= float(mesh.sigma_eff_top(LOSS_REF[0])) + 1e-12, "LOSS_REF sigma_eff above sigma_eff_top(mu_L)"
    for key, par in LEVEL_PAR.items():
        assert inside_calibration_ranges(par), f"Test 2 level {key} outside the calibration ranges / sigma_eff above sigma_top"


# ======================================================================================
# 4. SIMULATION (V formed for every chip; scalars F, a, b and per-family E^2 stored)
# ======================================================================================

def matrix_scalars(U, V, n):
    """V: (c, N, N).  Returns F = |Tr U^dag V|^2/(N Tr V^dag V), a = Re Tr(U^dag V)/N, b = Im Tr(U^dag V)/N (per chip)."""
    tr1 = np.einsum("ij,cij->c", U.conj(), V)
    tr2 = np.sum(np.abs(V) ** 2, axis=(1, 2))
    return np.abs(tr1) ** 2 / (n * tr2), tr1.real / n, tr1.imag / n


def chip_stats(dec, U, XT, Yid, fam, n, par, rng, chips):
    """Chips of ONE unitary at error vector par.  Returns F, a, b (chips,) and E2f (chips, T): mean over the inputs of each family of
    E^2 = ||y_hw_hat - y_ideal_hat||^2 (computed directly)."""
    sr, sS, mu, sg = par
    n_m = mesh.n_mzi(n)
    eye = np.eye(n, dtype=np.complex128)
    chunk = max(1, min(chips, int(1.5e8 // (48 * n_m + 4 * 16 * n * n))))      # chips per batch (limits the memory)
    F, A, B, E2 = [], [], [], []
    done = 0
    while done < chips:
        c = min(chunk, chips - done)
        errs = mesh.sample_chip_errors(c, n_m, sr, sS, mu, 0.0, rng, sigma_eff=sg)
        Y, ls = mesh.propagate(dec, eye, errs)                       # Y[c, k, :] = output for input e_k = column k of V_c
        V = (Y * np.exp(ls)[..., None]).transpose(0, 2, 1)         # (c, N, N)
        f_, a_, b_ = matrix_scalars(U, V, n)
        F.append(f_)
        A.append(a_)
        B.append(b_)
        Yh = V @ XT
        Yh = Yh / np.linalg.norm(Yh, axis=1, keepdims=True)
        e2 = np.sum(np.abs(Yh - Yid[None]) ** 2, axis=1)           # (c, K)
        E2.append(np.stack([e2[:, i0:i1].mean(axis=1) for i0, i1 in fam], axis=1))
        done += c
    return np.concatenate(F), np.concatenate(A), np.concatenate(B), np.concatenate(E2)


def unitary_bundle(n, mi, master, stream):
    """One Haar unitary with its decomposition and its inputs (INPUTS_PER_TYPE of every family) and ideal outputs.

    Returns U, dec, XT (inputs as columns), Yid (normalised ideal outputs), fam (column range of each input family)."""
    urng = np.random.default_rng([master, stream, n, mi])
    U = mesh.haar_unitary(n, urng)
    dec = mesh.decompose(U)
    blocks = [sweep_engine.gen_inputs(t, INPUTS_PER_TYPE, n, urng) for t in INPUT_TYPES]        # equal weight per family
    fam, first = [], 0
    for block in blocks:
        fam.append((first, first + len(block)))
        first += len(block)
    XT = np.concatenate(blocks).T
    Yid = U @ XT
    Yid = Yid / np.linalg.norm(Yid, axis=0, keepdims=True)
    return U, dec, XT, Yid, fam


def settings_hash(settings):
    """Fingerprint of the settings of a simulation: a cache file is only reused if it carries the same fingerprint."""
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def simulate_test1(mx):
    """Per-unitary caches t1_N<n>_u<mi>.npz (resumable).  Returns F, a, b (S, P, M, C) and m2f (S, P, M, T); NaN for skipped targets."""
    n = mx["n"]
    S, P, M, n_types = len(SETS), len(mx["T"]), N_UNITARIES, len(INPUT_TYPES)
    h = settings_hash(dict(n=n, M=M, C=CHIPS, seed=SEED, types=INPUT_TYPES, D=INPUTS_PER_TYPE, par=np.round(np.nan_to_num(mx["PAR"], nan=-1), 12).tolist(), v=4))
    F, A, B = (np.full((S, P, M, CHIPS), np.nan) for _ in range(3))
    M2 = np.full((S, P, M, n_types), np.nan)
    t0 = time.perf_counter()
    for mi in range(M):
        cache_file = OUT / f"t1_N{n}_u{mi:03d}.npz"
        if cache_file.exists() and str(np.load(cache_file)["h"]) == h:
            z = np.load(cache_file)
            F[:, :, mi], A[:, :, mi], B[:, :, mi], M2[:, :, mi] = z["F"], z["a"], z["b"], z["m2f"]
            continue
        U, dec, XT, Yid, fam = unitary_bundle(n, mi, SEED, 0)
        for si in range(S):
            for pi in range(P):
                if not mx["reach"][si, pi]:
                    continue
                rng = np.random.default_rng([SEED, 1, n, si, pi, mi])
                f_, a_, b_, e2 = chip_stats(dec, U, XT, Yid, fam, n, mx["PAR"][si, pi], rng, CHIPS)
                F[si, pi, mi], A[si, pi, mi], B[si, pi, mi], M2[si, pi, mi] = f_, a_, b_, e2.mean(axis=0)
        np.savez_compressed(cache_file, F=F[:, :, mi], a=A[:, :, mi], b=B[:, :, mi], m2f=M2[:, :, mi], h=np.array(h))
        elapsed = time.perf_counter() - t0
        print(f"  test 1, N={n}: unitary {mi + 1}/{M}  [{elapsed:7.1f}s, ~{elapsed / (mi + 1) * (M - mi - 1):6.0f}s left]", flush=True)
    return F, A, B, M2


def size_order():
    """All Test 2 sizes, with the most expensive ones (LAST_N) at the end."""
    all_sizes = sorted(set(CAL_N) | set(HELD_OUT_N))
    return [n for n in all_sizes if n not in LAST_N] + [n for n in LAST_N if n in all_sizes]


def simulate_test2(group, n):
    """group 'lossy' (four channels) or 'lossless' (loss channels zero).  Cache t2_<group>_N<n>.npz.  Returns dict(level -> (F, a, b (M, C), m2f (M, T)))."""
    series = "four_channel" if group == "lossy" else "lossless"
    master = SEED_HELD if n in HELD_OUT_N else SEED_CAL
    M, n_types = TEST2_UNITARIES, len(INPUT_TYPES)
    level_names = list(TEST2_LEVELS)
    h = settings_hash(dict(n=n, M=M, C=TEST2_CHIPS, master=master, group=group, types=INPUT_TYPES, D=INPUTS_PER_TYPE,
                           par={k: list(np.round(LEVEL_PAR[(series, k)], 12)) for k in level_names}, v=4))
    cache_file = OUT / f"t2_{group}_N{n}.npz"
    if cache_file.exists() and str(np.load(cache_file)["h"]) == h:
        z = np.load(cache_file)
        return {lv: (z["F"][i], z["a"][i], z["b"][i], z["m2f"][i]) for i, lv in enumerate(level_names)}
    F, A, B = (np.zeros((len(level_names), M, TEST2_CHIPS)) for _ in range(3))
    M2 = np.zeros((len(level_names), M, n_types))
    t0 = time.perf_counter()
    for mi in range(M):
        U, dec, XT, Yid, fam = unitary_bundle(n, mi, master, 2)
        for li, lv in enumerate(level_names):
            rng = np.random.default_rng([master, 3 + (group == "lossless"), li, n, mi])
            f_, a_, b_, e2 = chip_stats(dec, U, XT, Yid, fam, n, LEVEL_PAR[(series, lv)], rng, TEST2_CHIPS)
            F[li, mi], A[li, mi], B[li, mi], M2[li, mi] = f_, a_, b_, e2.mean(axis=0)
        print(f"  test 2 [{group}] N={n}: unitary {mi + 1}/{M}  [{time.perf_counter() - t0:7.1f}s]", flush=True)
    np.savez_compressed(cache_file, F=F, a=A, b=B, m2f=M2, h=np.array(h))
    return {lv: (F[i], A[i], B[i], M2[i]) for i, lv in enumerate(level_names)}


# ======================================================================================
# 5. STATISTICS OF ONE POINT
# ======================================================================================

def I_of_m2(m2):
    """I_ph = -ln(1 - m2/2) from the mean squared error m2 (NaN where m2 >= 2)."""
    m2 = np.asarray(m2, dtype=float)
    return np.where(m2 < 2.0, -np.log(np.clip(1.0 - m2 / 2.0, 1e-300, None)), np.nan)


def floor_formula(I, n):
    """Floor-aware fidelity: 1/N^2 + (1 - 1/N^2) exp(-2 I)."""
    return 1.0 / n ** 2 + (1.0 - 1.0 / n ** 2) * np.exp(-2.0 * I)


def boot_gamma(a, b, B=200, seed=0):
    """Bootstrap over unitaries (rows) of Gamma = Var(a) + <b^2> (population variance over the resampled (unitary, chip) pairs).  Returns s.e. of (Gamma, Var_a, <b^2>)."""
    rng = np.random.default_rng(seed)
    M = a.shape[0]
    out = np.zeros((B, 3))
    for i in range(B):
        idx = rng.integers(0, M, M)
        aa, bb = a[idx], b[idx]
        va, mb = aa.var(), np.mean(bb ** 2)
        out[i] = (va + mb, va, mb)
    return out.std(axis=0, ddof=1)


def point_stats(F, a, b, m2f, boot=True, seed=0):
    """All statistics of ONE point.  F, a, b: (M, C) per-chip values (M unitaries x C chips); m2f: (M, T) per-unitary, per-family chip-mean of E^2."""
    M = F.shape[0]
    Fu = F.mean(axis=1)
    m2u = m2f.mean(axis=1)
    E = float(np.sqrt(m2u.mean()))
    se_E = float(m2u.std(ddof=1) / (2 * E * np.sqrt(M))) if E > 0 else 0.0
    I = float(I_of_m2(m2u.mean()))
    se_I = float(E * se_E / (1 - E ** 2 / 2)) if np.isfinite(I) else float("nan")
    unresolved = bool((not np.isfinite(I)) or (1 - E ** 2 / 2) < 3 * E * se_E)
    mean_a = float(a.mean())
    var_a, mean_b2 = float(a.var()), float(np.mean(b ** 2))
    d = dict(F_mean=float(Fu.mean()), sd_F=float(Fu.std(ddof=1)), se_F=float(Fu.std(ddof=1) / np.sqrt(M)), E_rms=E, se_E_rms=se_E, I_rms=I, se_I_rms=se_I,
             I_unresolved=unresolved, mean_a=mean_a, I_tau=float(-np.log(mean_a)) if mean_a > 0 else float("nan"), Var_a=var_a, mean_b2=mean_b2,
             Gamma=var_a + mean_b2, identity_residual=float(F.mean() - (mean_a ** 2 + var_a + mean_b2)),
             I_rms_family=I_of_m2(m2f.mean(axis=0)).tolist(), E_rms_family=np.sqrt(m2f.mean(axis=0)).tolist())
    if boot:
        sg, sv, sb = boot_gamma(a, b, seed=seed)
        d.update(se_Gamma=float(sg), se_Var_a=float(sv), se_mean_b2=float(sb))
    return d


# ======================================================================================
# 6. ANALYSIS AND TABLES
# ======================================================================================

def x_of(p, lossless):
    """x-value of the I_ph axes: realised I_rms; where it is unresolved, I_tau for lossless runs (exact from the same matrices), NaN for runs with loss."""
    if not p["I_unresolved"]:
        return p["I_rms"], False
    if lossless and np.isfinite(p["I_tau"]):
        return p["I_tau"], True
    return float("nan"), False


def decorate(p, n, test, size_set, lossless, par):
    """Add the formula predictions, their ratios / relative deviations and the error vector to the statistics dict p of one point."""
    x, from_tau = x_of(p, lossless)
    p.update(test=test, N=int(n), size_set=size_set, lossless=bool(lossless), I_x=x, x_from_tau=from_tau, I_fit=float(surrogate_I_ph(par, n)),
             sigma_rad=float(par[0]), sigma_S=float(par[1]), mu_L=float(par[2]), sigma_eff=float(par[3]))
    p["F_floor_real"] = float(floor_formula(x, n)) if np.isfinite(x) else float("nan")
    p["F_floor_surr"] = float(floor_formula(p["I_fit"], n))
    p["F_simple_real"] = float(np.exp(-2 * x)) if np.isfinite(x) else float("nan")
    p["F_pure_floor"] = 1.0 / n ** 2
    for k in ("floor_real", "floor_surr", "simple_real"):
        f = p["F_" + k]
        p["ratio_" + k] = p["F_mean"] / f if np.isfinite(f) and f > 0 else float("nan")
        p["reldev_" + k] = (p["F_mean"] - f) / f if np.isfinite(f) and f > 0 else float("nan")
        p["within2se_" + k] = bool(abs(p["F_mean"] - f) <= 2 * p["se_F"]) if np.isfinite(f) else False
    if lossless:                                                                 # exact identity for unitary V
        assert abs(p["identity_residual"]) < 1e-9, f"identity mean F = mean(a)^2 + Var_a + mean_b2 violated: {p['identity_residual']:.3e}"
    return p


def slope_minus_lnF(m2u, Fu, si, selected_targets, unitary_idx):
    """Slope k of -ln F versus I_ph through the origin, for set si, over the selected targets and the unitaries unitary_idx."""
    I = np.array([float(I_of_m2(m2u[si, pi, unitary_idx].mean())) for pi in selected_targets])
    f = np.array([Fu[si, pi, unitary_idx].mean() for pi in selected_targets])
    ok = np.isfinite(I) & (f > 0)
    return float(np.sum(I[ok] * -np.log(f[ok])) / np.sum(I[ok] ** 2))


def analyse_test1(mx, F, A, B, M2):
    """Statistics of every reachable point of Test 1, and the slope k of -ln F vs I_ph (with bootstrap s.d. ksd) of every set."""
    n = mx["n"]
    pts = []
    for si, s in enumerate(SETS):
        for pi, t in enumerate(mx["T"]):
            if not mx["reach"][si, pi]:
                continue
            p = point_stats(F[si, pi], A[si, pi], B[si, pi], M2[si, pi], seed=1000 * si + pi)
            p.update(set=s, target=float(t), target_index=pi, high_target=bool(t >= 1.0))
            decorate(p, n, "test1", "calibration" if n in CAL_N else "held_out", s != "D", mx["PAR"][si, pi])
            pts.append(p)
    with np.errstate(all="ignore"), warnings.catch_warnings():                    # skipped targets are all-NaN slices
        warnings.simplefilter("ignore")
        Fu, m2u = np.nanmean(F, axis=3), np.nanmean(M2, axis=3)                                 # (S, P, M)
    rng = np.random.default_rng(SEED)
    k, ksd = {}, {}
    M = Fu.shape[2]
    for si, s in enumerate(SETS):                                                                # slope of -ln F vs I_ph (I_ph <= 1: floor negligible)
        selected = [pi for pi, t in enumerate(mx["T"]) if mx["reach"][si, pi] and t <= 1.0]
        k[s] = slope_minus_lnF(m2u, Fu, si, selected, np.arange(M))
        ksd[s] = float(np.std([slope_minus_lnF(m2u, Fu, si, selected, rng.integers(0, M, M)) for _ in range(150)], ddof=1))
    return pts, k, ksd


def analyse_test2(raw):
    """raw: {(group, n): {level: (F, a, b, m2f)}} -> list of point dicts."""
    rows = []
    for (group, n), d in raw.items():
        series = "four_channel" if group == "lossy" else "lossless"
        for lv, (F, a, b, m2f) in d.items():
            p = point_stats(F, a, b, m2f, seed=n)
            p.update(series=series, level=lv, s=TEST2_LEVELS[lv])
            decorate(p, n, "test2", "held_out" if n in HELD_OUT_N else "calibration", group == "lossless", LEVEL_PAR[(series, lv)])
            rows.append(p)
    return rows


def deviation_summary(rows):
    """Mean / max |relative deviation| of the three formulas and the number of points within 2 s.e., per series, level and size set."""
    out = {}
    for series in sorted({r["series"] for r in rows}):
        for lv in TEST2_LEVELS:
            for ss in ("calibration", "held_out"):
                q = [r for r in rows if r["series"] == series and r["level"] == lv and r["size_set"] == ss]
                if not q:
                    continue
                d = {}
                for k in ("floor_real", "floor_surr", "simple_real"):
                    v = np.array([r["reldev_" + k] for r in q], dtype=float)
                    ok = np.isfinite(v)
                    d[k] = dict(n=int(ok.sum()), mean_abs=float(np.mean(np.abs(v[ok]))) if ok.any() else None, max_abs=float(np.max(np.abs(v[ok]))) if ok.any() else None,
                                n_within_2se=int(sum(r["within2se_" + k] for r in q if np.isfinite(r["reldev_" + k]))))
                out[f"{series}/{lv}/{ss}"] = d
    return out


def write_points_csv(path, pts, extra_cols):
    """One CSV row per point; extra_cols are the leading columns that differ between the two tests."""
    cols = (extra_cols + ["N", "size_set", "sigma_rad", "sigma_S", "mu_L", "sigma_eff", "F_mean", "se_F", "sd_F", "E_rms", "se_E_rms", "I_rms", "se_I_rms", "I_unresolved",
                          "I_tau", "mean_a", "Var_a", "mean_b2", "Gamma", "se_Gamma", "I_fit", "I_x", "x_from_tau", "F_floor_real", "F_floor_surr", "F_simple_real",
                          "F_pure_floor", "ratio_floor_real", "ratio_floor_surr", "ratio_simple_real", "reldev_floor_real", "reldev_floor_surr", "reldev_simple_real",
                          "within2se_floor_real", "within2se_floor_surr", "within2se_simple_real", "identity_residual"]
            + [f"E_rms_family_{i + 1}" for i in range(len(INPUT_TYPES))] + [f"I_rms_family_{i + 1}" for i in range(len(INPUT_TYPES))])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in pts:
            row = []
            for c in cols:
                if c.startswith("E_rms_family_"):
                    v = p["E_rms_family"][int(c.rsplit("_", 1)[1]) - 1]
                elif c.startswith("I_rms_family_"):
                    v = p["I_rms_family"][int(c.rsplit("_", 1)[1]) - 1]
                else:
                    v = p.get(c, "")
                row.append(f"{v:.8g}" if isinstance(v, (float, np.floating)) else v)
            w.writerow(row)


def save_chip_npz(path, entries):
    """entries: list of (label, N, F (M, C), a, b).  Flat arrays: label_id, N, unitary_index, chip_index, F, a, b (+ labels legend)."""
    lab, nn, ui, ci, F, a, b = ([] for _ in range(7))
    labels = sorted({e[0] for e in entries})
    for l, n, f_, a_, b_ in entries:
        M, C = f_.shape
        uu, cc = np.meshgrid(np.arange(M), np.arange(C), indexing="ij")
        ok = np.isfinite(f_)
        lab.append(np.full(ok.sum(), labels.index(l)))
        nn.append(np.full(ok.sum(), n))
        ui.append(uu[ok])
        ci.append(cc[ok])
        F.append(f_[ok])
        a.append(a_[ok])
        b.append(b_[ok])
    np.savez_compressed(path, label_id=np.concatenate(lab).astype(np.int16), N=np.concatenate(nn).astype(np.int16), unitary_index=np.concatenate(ui).astype(np.int16),
                        chip_index=np.concatenate(ci).astype(np.int16), F=np.concatenate(F), a=np.concatenate(a), b=np.concatenate(b), labels=np.array(labels))


# ======================================================================================
# 7. FIGURES
# ======================================================================================
FITTED_COLOUR = "#d95f02"        # orange: the formula evaluated with the fitted I_ph


def save_figure(fig, stem):
    """Save the figure as <stem>.pdf in the result folder and close it."""
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def legend_data_first(ax, data_labels):
    """Lower-left legend with the entries of the data points (data_labels) first, then the lines."""
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] not in data_labels)
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", fontsize=6.5, handlelength=1.8)


def make_mixture_figure(n, pts):
    """Test 1, network size n.  (a) F against the realised I_ph for the four error sets, with exp(-2 I_ph) and the floor-aware formula;
    (b) the ratio F / exp(-2 I_ph)."""
    shared.set_paper_style()
    fig, (ax_f, ax_ratio) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.32))
    resolved = [p for p in pts if np.isfinite(p["I_x"])]
    I_line = np.geomspace(min(p["I_x"] for p in resolved), 1.15 * max(p["I_x"] for p in resolved), 200)
    for s in SETS:
        q = [p for p in resolved if p["set"] == s]
        x = np.array([p["I_x"] for p in q])
        F = np.array([p["F_mean"] for p in q])
        se = np.array([p["se_F"] for p in q])
        style = dict(fmt=SET_MARKER[s], ms=3.5, color=SET_COL[s], elinewidth=0.7, capsize=1.5, zorder=3)
        ax_f.errorbar(x, F, yerr=se, label=SET_NAME[s], **style)
        ax_ratio.errorbar(x, F / np.exp(-2 * x), yerr=se / np.exp(-2 * x), **style)
    ax_f.plot(I_line, np.exp(-2 * I_line), ":", color="k", lw=0.9, label=r"$e^{-2I_{\mathrm{ph}}}$", zorder=2)
    ax_f.plot(I_line, floor_formula(I_line, n), "--", color="k", lw=0.9, label="floor-aware formula", zorder=2)
    ax_ratio.axhline(1.0, color="k", ls=":", lw=0.9, zorder=2)
    ax_f.set_yscale("log")
    for ax in (ax_f, ax_ratio):
        ax.set_xscale("log")
        ax.set_xlabel(r"realised index $I_{\mathrm{ph}}$")
    ax_f.set_ylabel(r"matrix fidelity $F_{\mathrm{matrix}}$")
    ax_ratio.set_ylabel(r"$F_{\mathrm{matrix}}\,/\,e^{-2I_{\mathrm{ph}}}$")
    legend_data_first(ax_f, list(SET_NAME.values()))
    shared.label_panel(ax_f, "a", x=0.97, ha="right")
    shared.label_panel(ax_ratio, "b", x=0.97, ha="right")
    return fig


def draw_floor_panel(ax, q, with_legend):
    """F against the mesh size N for one level: simulation (calibration sizes filled, held-out sizes open) and the three formulas."""
    N = np.array([r["N"] for r in q], float)
    calibration = np.array([r["size_set"] == "calibration" for r in q])
    F = np.array([r["F_mean"] for r in q])
    se = np.array([r["se_F"] for r in q])
    style = dict(color="k", ms=3.5, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.errorbar(N[calibration], F[calibration], yerr=se[calibration], fmt="o", label="simulation, calibration sizes", **style)
    ax.errorbar(N[~calibration], F[~calibration], yerr=se[~calibration], fmt="D", mfc="white", label="simulation, held-out sizes", **style)
    formulas = (("F_floor_real", "-", "k", r"floor-aware formula, realised $I_{\mathrm{ph}}$"),
                ("F_floor_surr", "-.", FITTED_COLOUR, r"floor-aware formula, fitted $I_{\mathrm{ph}}$"),
                ("F_simple_real", ":", "k", r"$e^{-2I_{\mathrm{ph}}}$, realised $I_{\mathrm{ph}}$"))
    for key, line_style, colour, label in formulas:
        y = np.array([r[key] for r in q])
        ok = np.isfinite(y)
        ax.plot(N[ok], y[ok], line_style, color=colour, lw=1.0, label=label, zorder=2)
    N_line = np.geomspace(N.min(), N.max(), 100)
    ax.plot(N_line, 1.0 / N_line ** 2, "--", color="0.5", lw=0.9, label=r"floor $1/N^{2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if with_legend:
        legend_data_first(ax, ["simulation, calibration sizes", "simulation, held-out sizes"])


def draw_ratio_panel(ax, q, ratio_top=1.3):
    """Ratio of the simulated F to the floor-aware formula, with the realised I_ph (black) and the fitted I_ph (orange).
    Ratios above ratio_top are drawn as triangles on the upper edge."""
    for key, colour in (("F_floor_real", "k"), ("F_floor_surr", FITTED_COLOUR)):
        for size_set, marker, face in (("calibration", "o", colour), ("held_out", "D", "white")):
            r = [row for row in q if row["size_set"] == size_set and np.isfinite(row[key]) and row[key] > 0]
            N = np.array([row["N"] for row in r], float)
            ratio = np.array([row["F_mean"] / row[key] for row in r])
            error = np.array([row["se_F"] / row[key] for row in r])
            inside = ratio <= ratio_top
            ax.errorbar(N[inside], ratio[inside], yerr=error[inside], fmt=marker, ms=3, mfc=face, color=colour, elinewidth=0.7, capsize=1.5)
            ax.plot(N[~inside], np.full(int((~inside).sum()), ratio_top), "^", ms=3.5, mfc=face, color=colour, clip_on=False)
    ax.axhline(1.0, color="0.5", lw=0.8, zorder=0)
    ax.set_ylim(0.9, ratio_top)


def make_floor_figure(rows):
    """Test 2, four channels.  Left the middle level, right the worst level; top F against the mesh size N, bottom the ratio F_sim / F_formula."""
    shared.set_paper_style()
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col", gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08, wspace=0.28))
    for column, (level, name) in enumerate((("mid", "middle"), ("worst", "worst"))):
        q = sorted([r for r in rows if r["series"] == "four_channel" and r["level"] == level], key=lambda r: r["N"])
        draw_floor_panel(axs[0, column], q, with_legend=(column == 0))
        draw_ratio_panel(axs[1, column], q)
        sizes = [r["N"] for r in q]
        ticks = [t for t in (20, 50, 100, 200, 500) if 0.95 * min(sizes) <= t <= 1.05 * max(sizes)]
        axs[1, column].set_xticks(ticks, labels=[str(t) for t in ticks])
        axs[1, column].xaxis.set_minor_formatter(NullFormatter())
        axs[1, column].set_xlabel(r"mesh size $N$")
        axs[0, column].text(0.97, 0.95, rf"{name} level, $s={TEST2_LEVELS[level]:g}$", transform=axs[0, column].transAxes, ha="right", va="top")
        shared.label_panel(axs[0, column], "ab"[column])
        shared.label_panel(axs[1, column], "cd"[column])
    axs[0, 0].set_ylabel(r"matrix fidelity $F_{\mathrm{matrix}}$")
    axs[1, 0].set_ylabel(r"$F_{\mathrm{sim}}/F_{\mathrm{formula}}$")
    return fig


# ======================================================================================
# 8. SELF-TESTS (bookkeeping of F, a, b)
# ======================================================================================

def selftest():
    """Haar limit and the identity <F> = <a>^2 + Var(a) + <b^2>."""
    n = 20
    rng = np.random.default_rng(7)
    Fs, As, Bs = [], [], []
    for _ in range(6000):                                          # Haar limit: W independent of U (no mesh)
        U, W = mesh.haar_unitary(n, rng), mesh.haar_unitary(n, rng)
        f_, a_, b_ = matrix_scalars(U, W[None], n)
        Fs.append(f_[0])
        As.append(a_[0])
        Bs.append(b_[0])
    Fs, As, Bs = map(np.array, (Fs, As, Bs))
    n2g = n ** 2 * (As.var() + np.mean(Bs ** 2))
    print(f"  Haar limit (N={n}, 6000 draws): <F> N^2 = {Fs.mean() * n ** 2:.3f} (expect 1), N^2 Gamma = {n2g:.3f} (expect 1)")
    assert abs(Fs.mean() * n ** 2 - 1) < 0.08 and abs(n2g - 1) < 0.08
    U = mesh.haar_unitary(n, rng)
    dec = mesh.decompose(U)
    X = np.concatenate([sweep_engine.gen_inputs(t, 4, n, rng) for t in sweep_engine.ALL_INPUT_TYPES]).T
    Yid = U @ X
    Yid = Yid / np.linalg.norm(Yid, axis=0, keepdims=True)
    fam = [(i * 4, i * 4 + 4) for i in range(len(sweep_engine.ALL_INPUT_TYPES))]
    f_, a_, b_, e2 = chip_stats(dec, U, X, Yid, fam, n, (0.1, 0.05, 0.0, 0.0), np.random.default_rng(3), 60)
    res = f_.mean() - (a_.mean() ** 2 + a_.var() + np.mean(b_ ** 2))
    print(f"  identity <F> = <a>^2 + Var(a) + <b^2> on 60 lossless chips: residual {res:.2e} (expect ~1e-16)")
    assert abs(res) < 1e-10
    print("  self-tests passed")



# ======================================================================================
# 9. DRIVER
# ======================================================================================

def json_default(o):
    """How numpy objects are written to JSON."""
    return o.tolist() if hasattr(o, "tolist") else str(o)


def write_json(obj, path):
    """Write obj as indented JSON (numpy objects are converted)."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=json_default)


def print_test1_header(n, mx):
    """Print the set-up of Test 1 at size n: targets, reachable maxima per set, number of unitaries and chips."""
    print(f"\nTEST 1, N = {n}: {len(mx['T'])} targets ({N_PII} low + {len(mx['T']) - N_PII} high) x 4 sets x {N_UNITARIES} unitaries x {CHIPS} chips; "
          f"I_max per set: " + ", ".join(f"{s}: {mx['imax'][s]:.3g}" for s in SETS))
    if mx["skipped"]:
        by = {s: sum(1 for k in mx["skipped"] if k["set"] == s) for s in SETS}
        print(f"  skipped targets (above 0.95 x set maximum; recorded, never clipped): {by}")


def write_test1_summary(n, mx, pts, k, ksd):
    """F_vs_PII_N<n>_summary.json"""
    write_json(dict(N=n, targets=mx["T"], I_max_per_set=mx["imax"], skipped_targets=mx["skipped"], n_unitaries=N_UNITARIES, chips=CHIPS,
                slope_minus_lnF_vs_I_le_1={s: dict(k=k[s], sd=ksd[s]) for s in SETS}, max_se_F=max(p["se_F"] for p in pts),
                n_unresolved=int(sum(p["I_unresolved"] for p in pts)), identity_max_abs_residual_lossless=max(abs(p["identity_residual"]) for p in pts if p["lossless"]),
                sets={s: mx["PAR"][si].tolist() for si, s in enumerate(SETS)}, parameters=FIT_PARAMS), OUT / f"F_vs_PII_N{n}_summary.json")


def print_test1_slopes(pts, k, ksd):
    """Print the slope k of -ln F against I_ph for every set, with its bootstrap s.d."""
    print("  -ln F = k I_ph (I_ph <= 1): " + ", ".join(f"{s}: {k[s]:.3f}+-{ksd[s]:.3f}" for s in SETS) + "  (theory 2 for unitary V);  max se_F = %.4f" % max(p["se_F"] for p in pts))


def run_test1():
    """Test 1 for every N of MIXTURE_N_LIST: simulate, analyse, write the tables and one figure per N."""
    chip_entries = []
    for n in MIXTURE_N_LIST:
        mx = build_mixture(n)
        print_test1_header(n, mx)
        F, A, B, M2 = simulate_test1(mx)
        pts, k, ksd = analyse_test1(mx, F, A, B, M2)
        write_points_csv(OUT / f"F_vs_PII_N{n}_points.csv", [dict(p) for p in pts], ["test", "set", "target", "target_index", "high_target"])
        write_test1_summary(n, mx, pts, k, ksd)
        save_figure(make_mixture_figure(n, pts), f"F_vs_PII_N{n}")
        print_test1_slopes(pts, k, ksd)
        for si, s in enumerate(SETS):
            for pi in range(len(mx["T"])):
                if mx["reach"][si, pi]:
                    chip_entries.append((f"test1_N{n}_{s}_target{pi:02d}", n, F[si, pi], A[si, pi], B[si, pi]))
    save_chip_npz(OUT / "chip_scalars_test1.npz", chip_entries)


def write_test2_summary(rows, dsum):
    """F_vs_N_floor_summary.json"""
    write_json(dict(calibration_N=CAL_N, held_out_N=HELD_OUT_N, levels=TEST2_LEVELS, max_vector=TEST2_MAX_VECTOR,
                n_unitaries=TEST2_UNITARIES, chips=TEST2_CHIPS, seeds=dict(test1=SEED, calibration=SEED_CAL, held_out=SEED_HELD), deviation_summary=dsum,
                identity_max_abs_residual_lossless=max(abs(r["identity_residual"]) for r in rows if r["lossless"]),
                note="realised-I variants have no fitted parameters; the surrogate-I variant is in-sample at calibration sizes and out-of-sample at held-out sizes"), OUT / "F_vs_N_floor_summary.json")


def print_test2_deviations(dsum):
    """Print mean and maximum relative deviation of the floor-aware formulas for every series, level and size set."""
    print("\nTEST 2: relative deviation (F_sim - F_formula)/F_formula: mean |.| / max |.| and number of points within 2 s.e.")
    for key, v in dsum.items():
        print(f"  {key:34s} " + "  ".join(f"{k}: {100 * a['mean_abs']:.1f}%/{100 * a['max_abs']:.1f}% ({a['n_within_2se']}/{a['n']})" for k, a in v.items() if a["n"]))


def run_test2():
    """Test 2: simulate all sizes (with loss first, lossless second), analyse, write the tables and the figure."""
    raw = {}
    for group in ("lossy", "lossless"):                      # runs with loss first, lossless second; N = 500 and 425 last within each group
        for n in size_order():
            raw[(group, n)] = simulate_test2(group, n)
    rows = analyse_test2(raw)
    write_points_csv(OUT / "F_vs_N_floor_table.csv", [dict(r) for r in rows], ["test", "series", "level", "s"])
    dsum = deviation_summary(rows)
    write_test2_summary(rows, dsum)
    save_figure(make_floor_figure(rows), "F_vs_N_floor")
    print_test2_deviations(dsum)
    chip_entries = []
    for (group, n), d in raw.items():
        for lv, (F, a, b, m2f) in d.items():
            chip_entries.append((f"{group}_{lv}", n, F, a, b))
    save_chip_npz(OUT / "chip_scalars_test2.npz", chip_entries)


def main():
    """Run Test 1 and / or Test 2 (or the self-tests), as chosen on the command line."""
    t0 = time.perf_counter()
    check_settings()
    if RUN_SELFTEST:
        print("SELF-TESTS (F, a, b bookkeeping):")
        selftest()
        sys.exit(0)
    if RUN_T1:
        run_test1()
    if RUN_T2:
        run_test2()
    print(f"\nall files in {OUT}   (total {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
