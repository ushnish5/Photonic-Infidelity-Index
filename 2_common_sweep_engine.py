"""
2_common_sweep_engine.py  --  shared engine for the four error-sensitivity sweeps
========================================================================

The question: how large is the output error E of a Clements mesh when ONE hardware error
channel is switched on?  The four channels are

    sigma_rad   phase-shifter error [rad]
    sigma_S     splitting-ratio error
    mu_L        mean insertion loss per arm [dB]
    sigma_L     spread of the insertion loss.  We sweep the REALISED spread sigma_eff [dB];
                the nominal Gaussian width is computed from it (see 1_mzi_mesh.py).

The four scripts  2a_phase_error_sweep.py / 2b_splitting_ratio_error_sweep.py / 2c_mean_loss_sweep.py / 2d_loss_spread_sweep.py
only hold a block of settings and call  main_entry(cfg)  from this file.

What one run does
-----------------
Every run sweeps two ranges per channel ("small" = near realistic values, used for the fits;
"large" = stress range that shows saturation), 50 points each.  For each network size N:

    1. make M Haar-random unitaries (decomposed once, reused for every curve)
    2. for every unitary, simulate `chips` noisy chips
    3. every chip is fed one input of EVERY input type (dense complex, binary, sparse, ...)
    4. E = || y_hw/|y_hw| - y_ideal/|y_ideal| ||
    5. per sweep point:  E_rms = sqrt( mean of E^2 over unitaries, chips and inputs )

Then the results are written as CSV / npz files, plus one figure (PDF) per range: E_rms versus the swept error, all input types pooled.

The file is organised like a book:
    0 input vectors   1 channel labels   2 sweep families   3 per-N setup   4 one curve
    5 statistics      6 CSV / npz output   7 figures         8 self-tests     9 driver
"""
import importlib
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
shared = importlib.import_module("0_shared_settings")  # noqa: E402
mesh = importlib.import_module("1_mzi_mesh")  # noqa: E402


# ======================================================================================
# 0. INPUT VECTORS
# ======================================================================================
# Seven kinds of input are used, so the error is not tied to one special input.

INPUT_SHORT = {
    "dense_complex": "dense complex", "dense_real": "dense real", "binary_pm1": r"binary $\pm1$",
    "phase_only": "phase-only", "sparse_10pct": "sparse 10%", "single_port": "single port",
    "gaussian_beam": "Gaussian beam",
}
ALL_INPUT_TYPES = list(INPUT_SHORT)


def _make_dense_complex(n, N, rng):
    """n complex Gaussian vectors of length N."""
    return rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N))


def _make_dense_real(n, N, rng):
    """n real Gaussian vectors of length N."""
    return rng.standard_normal((n, N)).astype(np.complex128)


def _make_binary_pm1(n, N, rng):
    """n vectors with entries +1 or -1."""
    return rng.choice([-1.0, 1.0], size=(n, N)).astype(np.complex128)


def _make_phase_only(n, N, rng):
    """n vectors with unit-modulus entries and random phases."""
    return np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(n, N)))


def _make_sparse_10pct(n, N, rng):
    """n vectors with Gaussian entries on 10 % of the ports (at least 2) and zero elsewhere."""
    k = max(2, int(round(0.1 * N)))                       # number of non-zero ports
    X = np.zeros((n, N), dtype=np.complex128)
    for i in range(n):
        idx = rng.choice(N, size=k, replace=False)
        X[i, idx] = rng.standard_normal(k) + 1j * rng.standard_normal(k)
    return X


def _make_single_port(n, N, rng):
    """n one-hot vectors (light in a single random port)."""
    X = np.zeros((n, N), dtype=np.complex128)
    X[np.arange(n), rng.choice(N, size=n, replace=n > N)] = 1.0
    return X


def _make_gaussian_beam(n, N, rng):
    """n Gaussian beams across the ports with random centre, width (5 - 25 % of N) and tilt."""
    j = np.arange(N)
    X = np.empty((n, N), dtype=np.complex128)
    for i in range(n):
        centre = rng.uniform(0.0, N - 1.0)
        width = max(1.0, N * rng.uniform(0.05, 0.25))
        tilt = rng.uniform(-np.pi, np.pi)
        X[i] = np.exp(-(j - centre) ** 2 / (4.0 * width * width)) * np.exp(1j * tilt * j)
    return X


_INPUT_MAKERS = {
    "dense_complex": _make_dense_complex, "dense_real": _make_dense_real, "binary_pm1": _make_binary_pm1,
    "phase_only": _make_phase_only, "sparse_10pct": _make_sparse_10pct, "single_port": _make_single_port,
    "gaussian_beam": _make_gaussian_beam,
}


def gen_inputs(kind: str, n: int, N: int, rng: np.random.Generator):
    """n random input vectors of the given type, each normalised to length 1.  Shape (n, N)."""
    if kind not in _INPUT_MAKERS:
        raise ValueError(f"unknown input type {kind!r}")
    X = _INPUT_MAKERS[kind](n, N, rng)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ======================================================================================
# 1. CHANNEL DESCRIPTIONS  (only used for labels in figures and file names)
# ======================================================================================

_LOSS_MODEL = (r"Hardware model: per-arm loss $L\sim\mathrm{TruncNorm}(\mu_{L},\,\sigma_{L}^{2};\,0,\,2\mu_{L})$ dB, amplitude $10^{-L/20}$, "
               r"independent for each arm of every MZI and every chip; single Clements mesh, ideal phase shifters and 50:50 couplers, "
               r"ideal output phase screen")
_LOSS_MODEL_EFF = (r"Hardware model: per-arm loss $L\sim\mathrm{TruncNorm}(\mu_{L},\,\sigma_{L}^{2};\,0,\,2\mu_{L})$ dB, amplitude $10^{-L/20}$, independent for each arm of every MZI and chip; "
                   r"swept: realised std $\sigma_{\mathrm{eff}}$ of $L$ ($\sigma_{L}$ by inversion); single Clements mesh, ideal phase shifters, 50:50 couplers and output phase screen")


CHANNELS = {
    "sigma_rad": dict(
        xlabel=r"Phase-shifter error standard deviation, $\sigma_{\mathrm{rad}}$ (rad)",
        title="Output-error sensitivity of a Clements unitary mesh vs. phase-shifter error",
        slope_unit=r"rad$^{-1}$", sym=r"\sigma_{\mathrm{rad}}", xunit="rad",
        model=(r"Hardware model: i.i.d. Gaussian phase errors $\delta\theta,\,\delta\phi\sim\mathcal{N}(0,\sigma_{\mathrm{rad}}^{2})$ "
               r"on every MZI, independent for every chip; single Clements mesh, ideal 50:50 couplers, no loss, ideal output phase screen"),
    ),
    "sigma_S": dict(
        xlabel=r"Splitting-ratio error standard deviation, $\sigma_{S}$ (power splitting ratio $\eta$, ideal $\eta_0=1/2$)",
        title="Output-error sensitivity of a Clements unitary mesh vs. splitting-ratio error",
        slope_unit=r"(unit of $\sigma_S$)$^{-1}$", sym=r"\sigma_{S}", xunit="",
        model=(r"Hardware model: coupler splitting ratio $\eta\sim\mathrm{TruncNorm}(1/2,\,\sigma_{S}^{2};\,0,\,1)$, drawn independently for both "
               r"couplers of every MZI and every chip; single Clements mesh, ideal phase shifters, no loss, ideal output phase screen"),
    ),
    "mu_L": dict(
        xlabel=r"Mean per-arm insertion loss, $\mu_{L}$ (dB)",
        title="Output-error sensitivity of a Clements unitary mesh vs. mean insertion loss",
        slope_unit=r"dB$^{-1}$", sym=r"\mu_{L}", xunit="dB", model=_LOSS_MODEL,
    ),
    "sigma_L": dict(
        xlabel=r"Effective (realised) insertion-loss standard deviation, $\sigma_{\mathrm{eff}}$ (dB)",
        title="Output-error sensitivity of a Clements unitary mesh vs. effective insertion-loss spread",
        slope_unit=r"dB$^{-1}$", sym=r"\sigma_{\mathrm{eff}}", xunit="dB", model=_LOSS_MODEL_EFF, col="sigma_eff_dB",
    ),
}
# result folders (created next to the scripts; a --quick run adds "_quick").  The sweep AND the fit of a channel use the same folder.
RESULT_FOLDER = {"sigma_rad": "results_phase_error", "sigma_S": "results_splitting_ratio_error",
                 "mu_L": "results_mean_loss", "sigma_L": "results_loss_spread"}
BASE_STEMS = {"sigma_rad": "sensitivity_sigma_rad", "sigma_S": "sensitivity_sigma_S",
              "mu_L": "sensitivity_mu_L", "sigma_L": "sensitivity_sigma_L"}


fmt, tag = shared.fmt, shared.tag     # number -> label text / file-name text (defined in 0_shared_settings.py)


# ======================================================================================
# 2. SWEEP FAMILIES
# ======================================================================================
# One "family" = one set of figures and files = a list of curves that share one x-grid.
# One "curve"  = one network size N (and fixed values of the other parameters), swept over x.

def eff_grid(mu, n_points, apply_cap=True):
    """Spread grid for the loss-spread channel: n_points values from 0 up to sigma_eff_top(mu).
    apply_cap=False lifts the 0.263 dB cap (exploratory range); the physical limit still holds."""
    top = float(mesh.sigma_eff_top(mu)) if apply_cap else mesh.REACH_FRAC * mesh.sigma_eff_cap(mu)
    assert 0.0 < top < mesh.sigma_eff_cap(mu), f"mu_L = {mu}: no reachable sigma_eff range"
    x = np.linspace(0.0, top, n_points)
    assert x[0] == 0.0
    return x


def _families_one_curve_per_N(cfg, range_key, common):
    """sigma_rad and sigma_S: one family per range, one curve per network size."""
    ch = cfg["CHANNEL"]
    curves = [dict(N=int(n), fixed={}, head=rf"$N={n}$") for n in cfg["N_VALUES"]]
    return [dict(common, key=range_key, stem=f"{BASE_STEMS[ch]}_{range_key}", curves=curves,
                 title=f"{CHANNELS[ch]['title']} ({range_key} range)", legend_title=r"Network size $N$",
                 curve_txt="all other error channels ideal (zero)", label_style="slope")]


def _families_mean_loss(cfg, range_key, common):
    """mu_L: one family per fixed loss spread (only the value 0 is supported)."""
    ch = cfg["CHANNEL"]
    if any(float(sv) != 0.0 for sv in cfg["CURVE_VALUES"]):
        raise ValueError("sweep_mu_L: a fixed loss spread != 0 is not supported: the spread is specified as the EFFECTIVE "
                         "(realised) sigma_eff, which cannot be held fixed while mu_L is swept down to 0 (sigma_eff < mu_L/sqrt(3)). "
                         "Use CURVE_VALUES = [0.0] (deterministic loss mu_L).")
    families = []
    for sv in cfg["CURVE_VALUES"]:
        curves = [dict(N=int(n), fixed={"sigma_L": float(sv)}, head=rf"$N={n}$") for n in cfg["N_VALUES"]]
        families.append(dict(common, key=f"{range_key}_sigmaL{tag(sv)}", stem=f"{BASE_STEMS[ch]}_{range_key}_sigmaL{tag(sv)}dB",
                             curves=curves,
                             title=f"{CHANNELS[ch]['title']} ({range_key} range; fixed $\\sigma_L={fmt(sv)}$ dB)",
                             legend_title=rf"Network size $N$  |  loss spread $\sigma_L={fmt(sv)}$ dB (fixed)",
                             curve_txt=rf"fixed $\sigma_L$ = {fmt(sv)} dB, all other error channels ideal",
                             label_style="slope"))
    return families


def _families_loss_spread(cfg, range_key, range_cfg, common):
    """sigma_L: panel A (curves for different N at one mu_L) and panel B (curves for different mu_L at one N)."""
    capped = bool(range_cfg.get("apply_cap", True))
    n_points = cfg["N_POINTS"]
    range_txt = f"{range_key} range" + ("" if capped else ", exploratory")

    # panel A: fixed mu_L, one curve per N
    mu_a = float(range_cfg["panel_A_mu"])
    x_a = eff_grid(mu_a, n_points, capped)
    curves_a = [dict(N=int(n), fixed={"mu_L": mu_a}, head=rf"$N={n}$", x=x_a) for n in cfg["N_VALUES"]]
    panel_a = dict(common, key=f"{range_key}_panelA", stem=f"{BASE_STEMS['sigma_L']}_{range_key}_panelA_Ncurves_muL{tag(mu_a)}dB",
                   curves=curves_a, ncol=3, x=x_a, apply_cap=capped,
                   title=(f"Output-error sensitivity of a Clements mesh vs. effective insertion-loss spread "
                          f"({range_txt}; fixed $\\mu_L={fmt(mu_a)}$ dB, varying $N$)"),
                   legend_title=r"Network size $N$",
                   curve_txt=rf"curves at fixed $\mu_L$ = {fmt(mu_a)} dB, one per network size",
                   label_style="sat")

    # panel B: fixed N, one curve per mu_L (each with its own grid)
    N_b = int(range_cfg["panel_B_N"])
    grids_b = [eff_grid(float(m), n_points, capped) for m in range_cfg["panel_B_mu_values"]]
    curves_b = [dict(N=N_b, fixed={"mu_L": float(m)}, x=xb,
                     head=rf"$\mu_L={fmt(m)}$ dB  ($\sigma_{{\mathrm{{top}}}}={xb[-1]:.4g}$ dB)")
                for m, xb in zip(range_cfg["panel_B_mu_values"], grids_b)]
    panel_b = dict(common, key=f"{range_key}_panelB", stem=f"{BASE_STEMS['sigma_L']}_{range_key}_panelB_muLcurves_N{N_b}",
                   curves=curves_b, ncol=3, x=max(grids_b, key=lambda a: a[-1]), apply_cap=capped,
                   title=(f"Output-error sensitivity of a Clements mesh vs. effective insertion-loss spread "
                          f"({range_txt}; fixed $N={N_b}$, varying $\\mu_L$)"),
                   legend_title=r"Mean loss $\mu_L$ (fixed per curve)",
                   curve_txt=rf"fixed $N$ = {N_b}, one curve per mean loss $\mu_L$",
                   label_style="sat")
    return [panel_a, panel_b]


def _family_separability_check(cfg, n_existing):
    """Optional extra (off by default): is the N-dependence independent of mu_L?  Two mu_L x two N."""
    n_points = cfg["N_POINTS"]
    curves = []
    for mu_s in cfg["SEPARABILITY_MU"]:
        x_s = eff_grid(float(mu_s), n_points, True)
        for n_s in cfg["SEPARABILITY_N"]:
            curves.append(dict(N=int(n_s), fixed={"mu_L": float(mu_s)}, x=x_s, head=rf"$\mu_L={fmt(mu_s)}$ dB, $N={n_s}$"))
    return dict(range_key="small", x=curves[0]["x"], swept="sigma_L", ncol=2, key="small_separability",
                stem=f"{BASE_STEMS['sigma_L']}_small_separability_check", family_idx=n_existing, curves=curves, apply_cap=True,
                title="Separability cross-check of the loss-spread sensitivity (small range): two $\\mu_L$ x two $N$",
                legend_title=r"Mean loss $\mu_L$ and network size $N$",
                curve_txt="separability cross-check: one curve per ($\\mu_L$, $N$)", label_style="sat")


def build_families(cfg):
    """List of all families of this run (each a dict, see the helpers above)."""
    ch = cfg["CHANNEL"]
    families = []
    for range_key, range_cfg in cfg["RANGES"].items():
        # sigma_rad / sigma_S / mu_L share one x-grid per range; the spread channel has a grid per curve
        x = np.linspace(0.0, range_cfg["x_max"], cfg["N_POINTS"]) if "x_max" in range_cfg else None
        if x is not None:
            assert x[0] == 0.0
        common = dict(range_key=range_key, x=x, swept=ch, ncol=4)
        if ch in ("sigma_rad", "sigma_S"):
            new = _families_one_curve_per_N(cfg, range_key, common)
        elif ch == "mu_L":
            new = _families_mean_loss(cfg, range_key, common)
        elif ch == "sigma_L":
            new = _families_loss_spread(cfg, range_key, range_cfg, common)
        else:
            raise ValueError(ch)
        for fam in new:
            fam["family_idx"] = len(families)          # the family number goes into the random seeds
            families.append(fam)
    if ch == "sigma_L" and cfg.get("SEPARABILITY_CHECK", False):
        families.append(_family_separability_check(cfg, len(families)))

    # number the curves and check the spread grids
    for fam in families:
        for ci, curve in enumerate(fam["curves"]):
            curve["ci"] = ci
            if ch == "sigma_L":
                mu_c = curve["fixed"]["mu_L"]
                limit = float(mesh.sigma_eff_top(mu_c)) if fam["apply_cap"] else mesh.REACH_FRAC * mesh.sigma_eff_cap(mu_c)
                assert curve["x"][0] == 0.0 and np.all(curve["x"] <= limit * (1 + 1e-12)) \
                    and np.all(curve["x"] < mesh.sigma_eff_cap(mu_c)), (fam["key"], mu_c)
    return families


# ======================================================================================
# 3. SETUP FOR ONE NETWORK SIZE N
# ======================================================================================

class Bundle:
    """Everything that belongs to one random unitary: the matrix, its mesh, its inputs, its ideal outputs."""
    __slots__ = ("U", "dec", "X3", "Y3", "residual")


def build_matrix(N, cfg, rng):
    """Make one Haar unitary, decompose it, and prepare the inputs of every chip.

    X3[c, t] = the input of type t that chip c receives (chip c uses draw  c mod INPUTS_PER_TYPE  of every type).
    Y3       = the ideal outputs U x.
    residual = how well the error-free mesh reproduces U x (must be ~1e-15)."""
    chips, per_type, types = cfg["CHIPS_PER_MATRIX"], cfg["INPUTS_PER_TYPE"], cfg["INPUT_TYPES"]
    assert chips >= per_type, "need at least as many chips as inputs per type"
    b = Bundle()
    b.U = mesh.haar_unitary(N, rng)
    b.dec = mesh.decompose(b.U)
    inputs = np.stack([gen_inputs(t, per_type, N, rng) for t in types])      # (types, draws, N)
    draw_of_chip = np.arange(chips) % per_type
    b.X3 = np.ascontiguousarray(inputs[:, draw_of_chip, :].transpose(1, 0, 2))   # (chips, types, N)
    b.Y3 = b.X3 @ b.U.T
    # self-check with an error-free chip
    all_inputs = inputs.reshape(-1, N)
    Y, _ = mesh.propagate(b.dec, all_inputs, mesh.ideal_chip_errors(1, mesh.n_mzi(N)))
    b.residual = float(np.max(np.abs(Y[0] - all_inputs @ b.U.T)))
    return b


# ======================================================================================
# 4. ONE CURVE  (sweep one parameter over its grid, with checkpoints)
# ======================================================================================

def curve_signature(cfg, N, base_par, swept, x, seed_key):
    """Short fingerprint of all settings of a curve.  A checkpoint is only reused if the fingerprint matches."""
    d = dict(seed=cfg["SEED"], N=int(N), M=cfg["N_MATRICES"], chips=cfg["CHIPS_PER_MATRIX"], D=cfg["INPUTS_PER_TYPE"],
             types=list(cfg["INPUT_TYPES"]), base=base_par, swept=swept, x=[float(v) for v in x], key=list(seed_key))
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def sweep_curve(cfg, bundles, x, base_par, swept, seed_key, label):
    """Run one curve: for every unitary and every grid point, simulate the chips.
    Returns arrays  E, E2 (M, P, chips, T),  Ei, LT (M, P, T)   with M unitaries, P points, T input types."""
    M, P, chips, T = len(bundles), len(x), cfg["CHIPS_PER_MATRIX"], len(cfg["INPUT_TYPES"])
    E = np.zeros((M, P, chips, T), dtype=np.float32)
    E2 = np.zeros((M, P, chips, T))                         # E^2, kept in float64
    Ei = np.zeros((M, P, T))                                # phase-invariant error (diagnostic)
    LT = np.zeros((M, P, T))                                # log10 power transmission (diagnostic)
    sweeping_spread = swept == "sigma_L"                    # then the swept value is the realised spread sigma_eff
    t0 = time.perf_counter()
    for mi, b in enumerate(bundles):
        rng = np.random.default_rng([cfg["SEED"], 1] + list(seed_key) + [mi])
        for pi, x_value in enumerate(x):
            par = dict(base_par)
            par[swept] = float(x_value)
            out = mesh.run_chips(b.dec, b.X3, chips, sigma_rad=par["sigma_rad"], sigma_S=par["sigma_S"],
                               mu_L=par["mu_L"], sigma_L=0.0 if sweeping_spread else par["sigma_L"], rng=rng, Y_ideal=b.Y3,
                               sigma_eff=par["sigma_L"] if sweeping_spread else None)
            E[mi, pi] = out["E_raw"]
            E2[mi, pi] = out["E2_raw"]
            Ei[mi, pi] = out["E_inv"].mean(axis=0)
            LT[mi, pi] = out["log10_T"].mean(axis=0)
        elapsed = time.perf_counter() - t0
        print(f"  {label}  unitary {mi + 1:2d}/{M}  E@x_max = {E[mi, -1].mean():.4f}  "
              f"E@x_min = {E[mi, 0].mean():.2e}  [{elapsed:6.1f}s]", flush=True)
    return E, Ei, LT, E2


# ======================================================================================
# 5. STATISTICS
# ======================================================================================

def slope_through_anchor(x, y, idx):
    """Slope of the straight line that starts at the first point (x[0], y[0]) and is fitted
    (least squares) to the points `idx`:  slope = sum(dx*dy) / sum(dx^2)."""
    idx = list(idx)
    dx = x[idx] - x[0]
    dy = y[..., idx] - y[..., [0]]
    return (dy * dx).sum(axis=-1) / (dx * dx).sum()


def overlap_deficit(E):
    """I = -ln(1 - E^2/2).  (Clipped, because noise can push E slightly above sqrt(2).)"""
    c = np.clip(1.0 - np.asarray(E, dtype=float) ** 2 / 2.0, 1e-12, 1.0)
    return -np.log(c)


def rms_statistics(m1, m2):
    """Statistics per sweep point from the per-unitary averages
         m1 = mean of E,   m2 = mean of E^2          (both arrays of shape (n_unitaries, P)).
    E_rms = sqrt(mean over unitaries of m2).  The error bar plotted in the figures is the spread across
    unitaries (sd_E_rms); the standard error (se_E_rms, delta method) is used in the fits."""
    n_u = m1.shape[0]
    E_avg = m1.mean(axis=0)
    E_rms = np.sqrt(m2.mean(axis=0))
    rho = np.sqrt(m2)                                        # per-unitary RMS error
    with np.errstate(divide="ignore", invalid="ignore"):
        se_rms = np.where(E_rms > 0, m2.std(axis=0, ddof=1) / (2.0 * E_rms * np.sqrt(n_u)), 0.0)
        I_rms = np.where(E_rms ** 2 < 2.0, -np.log(1.0 - E_rms ** 2 / 2.0), np.nan)
        I_avg = np.where(E_avg ** 2 < 2.0, -np.log(1.0 - E_avg ** 2 / 2.0), np.nan)
    return dict(E_avg=E_avg, E_rms=E_rms, rho=rho, sd_E_rms=rho.std(axis=0, ddof=1), sd_E_avg=m1.std(axis=0, ddof=1),
                se_E_rms=se_rms, se_E_avg=m1.std(axis=0, ddof=1) / np.sqrt(n_u),
                E_sample_sd=np.sqrt(np.maximum(m2.mean(axis=0) - E_avg ** 2, 0.0)), I_rms=I_rms, I_avg=I_avg,
                jensen_gap_E=E_rms - E_avg)


def aggregate(cfg, fam, curve, x, E, Ei, LT, E2):
    """Turn the raw chip-level arrays of one curve into the numbers that are written and plotted."""
    M = E.shape[0]
    assert E.shape == E2.shape                               # equal sample counts for every unitary
    sem = lambda a, ax: a.std(axis=ax, ddof=1) / np.sqrt(M)  # noqa: E731   (standard error across unitaries)
    idx = cfg["SLOPE_FIT_INDICES"]

    # first average over chips, then over input types  ->  one number per unitary and point
    E_mt = E.astype(np.float64).mean(axis=2)                 # (M, P, T)
    pooled_mat = E_mt.mean(axis=2)                           # (M, P)  mean of E            (m1)
    E2_mt = E2.mean(axis=2)                                  # (M, P, T)
    m2_mat = E2_mt.mean(axis=2)                              # (M, P)  mean of E^2          (m2)

    r = dict(N=curve["N"], ci=curve["ci"], fixed=curve["fixed"], head=curve["head"],
             E_mt=E_mt, pooled_mat=pooled_mat,
             mean=pooled_mat.mean(axis=0), sem=sem(pooled_mat, 0),
             mean_t=E_mt.mean(axis=0), sem_t=sem(E_mt, 0),
             Ei_mean=Ei.mean(axis=2).mean(axis=0), Ei_sem=sem(Ei.mean(axis=2), 0), Ei_mean_t=Ei.mean(axis=0),
             LT_mean=LT.mean(axis=2).mean(axis=0))
    stats = rms_statistics(pooled_mat, m2_mat)
    r.update(m1=pooled_mat, m2=m2_mat, n_samples=int(E.shape[2] * E.shape[3]), stats=stats,
             rms=stats["E_rms"], sd_rms=stats["sd_E_rms"], rho=stats["rho"],
             rms_t=np.sqrt(E2_mt.mean(axis=0)))              # (P, T): E_rms for each input type

    # slopes of the E_rms curve (the main result of the "slope" families)
    r["slope"] = float(slope_through_anchor(x, r["rms"], idx))
    slope_per_unitary = slope_through_anchor(x, stats["rho"], idx)
    r["slope_mem_mean"], r["slope_mem_sem"] = float(slope_per_unitary.mean()), float(slope_per_unitary.std(ddof=1) / np.sqrt(M))
    r["slope_t"] = slope_through_anchor(x, r["rms_t"].T, idx)

    # numbers of the "sat" families (loss spread): value at the start and at the end of the grid
    if fam["label_style"] == "sat":
        k = cfg.get("END_AVG_POINTS", 1)
        end_per_unitary = stats["rho"][:, -k:].mean(axis=1)
        r["E0"] = float(r["rms"][0])
        r["E_max"], r["E_max_sem"] = float(end_per_unitary.mean()), float(end_per_unitary.std(ddof=1) / np.sqrt(M))
        r["E_max_sd"] = float(end_per_unitary.std(ddof=1))
        mu = curve["fixed"]["mu_L"]
        r["x"] = np.asarray(x, dtype=float)
        r["sig_eff"] = r["x"].copy()                         # the swept variable IS sigma_eff
        r["sig_nom"] = mesh.sigma_nominal_from_eff(mu, r["x"])
    return r


# ======================================================================================
# 6. WRITING CSV / NPZ FILES
# ======================================================================================

def _num(v):
    """Number -> text for CSV (10 significant digits; 'nan' for NaN / inf)."""
    return "nan" if not np.isfinite(v) else f"{v:.10g}"


def _fixed_text(result, fixed_keys):
    """The fixed parameters of a curve as CSV cells."""
    values = [result["fixed"].get(k, "") for k in fixed_keys] or [""]
    return [f"{v:.10g}" if v != "" else "" for v in values]


def _write_curves_csv(path, cfg, fam, results, spread):
    """One row per (curve, sweep point) with all statistics.  `spread` = loss-spread family."""
    x, types = fam["x"], cfg["INPUT_TYPES"]
    M = results[0]["pooled_mat"].shape[0]
    x_col = CHANNELS[cfg["CHANNEL"]].get("col", cfg["CHANNEL"])
    fixed_keys = sorted({k for c in fam["curves"] for k in c["fixed"]})
    fixed_cols = [f"fixed_{k}" for k in fixed_keys] or ["fixed_none"]
    extra_cols = ["sigma_nominal_dB", "sigma_eff_top_dB"] if spread else []
    header = (["curve_index", "N"] + fixed_cols + [x_col] + extra_cols
              + ["n_unitaries", "n_samples_per_unitary", "E_avg", "E_rms", "sd_E_rms", "sd_E_avg", "se_E_rms", "se_E_avg",
                 "E_sample_sd", "I_rms", "I_avg", "jensen_gap_E", "I_excess_rms_over_x0", "E_inv_mean", "log10_power_transmission_mean"]
              + [f"E_rms_{t}" for t in types] + [f"E_avg_{t}" for t in types] + [f"E_inv_{t}_mean" for t in types]
              + [f"E_mean_unitary_{k + 1:02d}" for k in range(M)] + [f"E2_mean_unitary_{k + 1:02d}" for k in range(M)])
    stat_names = ("E_avg", "E_rms", "sd_E_rms", "sd_E_avg", "se_E_rms", "se_E_avg", "E_sample_sd", "I_rms", "I_avg", "jensen_gap_E")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in results:
            st = r["stats"]
            I_excess = st["I_rms"] - st["I_rms"][0]
            for pi, x_value in enumerate(r["x"] if spread else x):
                row = [r["ci"], r["N"]] + _fixed_text(r, fixed_keys) + [f"{x_value:.10g}"]
                if spread:
                    row += [f"{r['sig_nom'][pi]:.10g}", f"{r['x'][-1]:.10g}"]
                row += [M, r["n_samples"]] + [_num(st[k][pi]) for k in stat_names]
                row += [_num(I_excess[pi]), f"{r['Ei_mean'][pi]:.10g}", f"{r['LT_mean'][pi]:.10g}"]
                row += [f"{r['rms_t'][pi, i]:.10g}" for i in range(len(types))]
                row += [f"{r['mean_t'][pi, i]:.10g}" for i in range(len(types))]
                row += [f"{r['Ei_mean_t'][pi, i]:.10g}" for i in range(len(types))]
                row += [f"{r['m1'][k, pi]:.10g}" for k in range(M)] + [f"{r['m2'][k, pi]:.10g}" for k in range(M)]
                w.writerow(row)


def _write_per_unitary_csv(path, cfg, fam, results, spread):
    """One row per (curve, sweep point, unitary): the raw numbers the fits are built from."""
    x, ch = fam["x"], cfg["CHANNEL"]
    M = results[0]["pooled_mat"].shape[0]
    x_col = CHANNELS[ch].get("col", ch)
    fixed_keys = sorted({k for c in fam["curves"] for k in c["fixed"]})
    fixed_cols = [f"fixed_{k}" for k in fixed_keys] or ["fixed_none"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel"] + ["curve_index", "N"] + fixed_cols + [x_col] + (["sigma_nominal_dB"] if spread else [])
                   + ["point_index", "unitary_index", "n_samples", "E_mean", "E2_mean"])
        for r in results:
            for pi, x_value in enumerate(r["x"] if spread else x):
                for k in range(M):
                    w.writerow([ch, r["ci"], r["N"]] + _fixed_text(r, fixed_keys) + [f"{x_value:.10g}"]
                               + ([f"{r['sig_nom'][pi]:.10g}"] if spread else [])
                               + [pi, k, r["n_samples"], f"{r['m1'][k, pi]:.12g}", f"{r['m2'][k, pi]:.12g}"])


def _write_slopes_csv(path, cfg, fam, results):
    """One row per curve: the slope of E_rms near zero (only for the "slope" families)."""
    x, types, idx = fam["x"], cfg["INPUT_TYPES"], cfg["SLOPE_FIT_INDICES"]
    fixed_keys = sorted({k for c in fam["curves"] for k in c["fixed"]})
    fixed_cols = [f"fixed_{k}" for k in fixed_keys] or ["fixed_none"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve_index", "N"] + fixed_cols + ["slope_of_E_rms_curve", "slope_per_unitary_mean", "slope_per_unitary_sem"]
                   + [f"slope_{t}" for t in types] + ["fit_point_indices", "fit_point_x", "slope_unit"])
        for r in results:
            w.writerow([r["ci"], r["N"]] + _fixed_text(r, fixed_keys)
                       + [f"{r['slope']:.10g}", f"{r['slope_mem_mean']:.10g}", f"{r['slope_mem_sem']:.10g}"]
                       + [f"{s:.10g}" for s in r["slope_t"]]
                       + [" ".join(str(i) for i in idx), " ".join(f"{x[i]:.6g}" for i in idx),
                          "E_rms per unit of the swept variable"])


def _write_summary_csv(path, cfg, fam, results):
    """One row per curve: E_rms at the start and at the end of the grid (only for the loss-spread families)."""
    fixed_keys = sorted({k for c in fam["curves"] for k in c["fixed"]})
    fixed_cols = [f"fixed_{k}" for k in fixed_keys] or ["fixed_none"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve_index", "N"] + fixed_cols + ["E_rms_at_sigma_eff_0", "E_rms_at_sigma_eff_max", "sd_E_rms_at_sigma_eff_max", "end_avg_points",
                    "I_rms_at_sigma_eff_0", "I_rms_at_sigma_eff_max", "I_rms_excess_at_sigma_eff_max", "sigma_eff_max_dB",
                    "sigma_nominal_at_sigma_eff_max_dB", "sigma_eff_requested_max_dB", "sigma_eff_cap_mu_over_sqrt3_dB",
                    "cap_active", "cap_applied"])
        for r in results:
            I0, I_end = float(overlap_deficit(r["E0"])), float(overlap_deficit(r["E_max"]))
            mu = r["fixed"]["mu_L"]
            w.writerow([r["ci"], r["N"]] + [f"{r['fixed'][k]:.10g}" for k in fixed_keys]
                       + [f"{r['E0']:.10g}", f"{r['E_max']:.10g}", f"{r['E_max_sd']:.10g}", cfg.get("END_AVG_POINTS", 1),
                          f"{I0:.10g}", f"{I_end:.10g}", f"{I_end - I0:.10g}", f"{r['sig_eff'][-1]:.10g}",
                          f"{r['sig_nom'][-1]:.10g}", f"{r['x'][-1]:.10g}", f"{mu / np.sqrt(3.0):.10g}",
                          bool(fam["apply_cap"] and mesh.SIGMA_EFF_CAP_DB < mesh.REACH_FRAC * mu / np.sqrt(3.0)), bool(fam["apply_cap"])])


def write_outputs(cfg, fam, results, outdir, raw_payload):
    """Write the curves CSV, per-unitary CSV, slopes (or summary) CSV and the npz of one family.  Returns the paths."""
    spread = fam["label_style"] == "sat"
    stem = fam["stem"]
    saved = []

    path = outdir / f"{stem}_curves.csv"
    _write_curves_csv(path, cfg, fam, results, spread)
    saved.append(path)

    path = outdir / f"{stem}_per_unitary.csv"
    _write_per_unitary_csv(path, cfg, fam, results, spread)
    saved.append(path)

    if not spread:
        path = outdir / f"{stem}_slopes.csv"
        _write_slopes_csv(path, cfg, fam, results)
    else:
        path = outdir / f"{stem}_summary.csv"
        _write_summary_csv(path, cfg, fam, results)
    saved.append(path)

    path = outdir / f"{stem}.npz"
    fixed_keys = sorted({k for c in fam["curves"] for k in c["fixed"]})
    payload = {"x": fam["x"], "input_types": np.array(cfg["INPUT_TYPES"]), "N_per_curve": np.array([c["N"] for c in fam["curves"]])}
    for k in fixed_keys:
        payload[f"fixed_{k}_per_curve"] = np.array([c["fixed"].get(k, np.nan) for c in fam["curves"]])
    if spread:
        payload["sigma_eff_per_curve"] = np.array([r["x"] for r in results])
        payload["sigma_nominal_per_curve"] = np.array([r["sig_nom"] for r in results])
    payload.update(raw_payload)
    np.savez_compressed(path, **payload)
    saved.append(path)
    return saved


# ======================================================================================
# 7. FIGURES
# ======================================================================================

def _style():
    """Matplotlib look: serif fonts, ticks on all four sides."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    })


def _viridis_colours(n):
    """n colours from the viridis map (hex strings), from dark to light."""
    return [plt.matplotlib.colors.to_hex(plt.cm.viridis(v)) for v in np.linspace(0.0, 0.92, max(n, 2))]


def _curves_to_plot(fam, results):
    """The curves of the figure as tuples (label head, E_rms, x, band): all input types pooled, one tuple per curve."""
    return [(r["head"], r["rms"], r.get("x", fam["x"]), r["sd_rms"]) for r in results]


def _draw_entries(ax, cfg, fam, entries):
    """Draw the curves, their bands and (spread families) the open circle at the end.  Returns the line handles."""
    x = fam["x"]
    ch = CHANNELS[cfg["CHANNEL"]]
    colours = _viridis_colours(len(entries))
    handles = []
    for k, (head, y, xk, band) in enumerate(entries):
        if fam["label_style"] == "slope":
            label = rf"{head}   (slope $={slope_through_anchor(x, y, cfg['SLOPE_FIT_INDICES']):.4g}$ {ch['slope_unit']})"
        else:
            label = rf"{head}   ($E_0={y[0]:.3f}$,  $E_{{\max}}={y[-cfg.get('END_AVG_POINTS', 1):].mean():.3f}$)"
        line, = ax.plot(xk, y, marker=cfg["MARKERS"][k % len(cfg["MARKERS"])], markersize=cfg["MARKER_SIZE"],
                        linewidth=cfg["LINE_WIDTH"], color=colours[k], markerfacecolor=colours[k], markeredgewidth=0.0,
                        label=label, zorder=3)
        handles.append(line)
        if band is not None:                                  # +-1 s.d. across the random unitaries
            ax.fill_between(xk, y - band, y + band, color=colours[k], alpha=0.12, linewidth=0, zorder=2)
        if fam["label_style"] == "sat":                       # open circle = intended end of the curve
            ax.plot([xk[-1]], [y[-1]], marker="o", markersize=cfg["MARKER_SIZE"] + 4, markerfacecolor="white",
                    markeredgecolor=colours[k], markeredgewidth=1.6, linestyle="none", zorder=4)
    return handles


def _caption_lines(cfg, fam, results, ch):
    """The five grey text lines under the figure title."""
    x, types = fam["x"], cfg["INPUT_TYPES"]
    M = results[0]["pooled_mat"].shape[0]
    chips, D, T = cfg["CHIPS_PER_MATRIX"], cfg["INPUTS_PER_TYPE"], len(types)
    spread = fam["label_style"] == "sat"
    n_txt = ", ".join(str(n) for n in dict.fromkeys(r["N"] for r in results))
    shown = f"shown: mean over all {T} types (equal weight)"
    types_txt = ", ".join(INPUT_SHORT[t] for t in types)
    reuse = f"{chips // D}" if chips % D == 0 else f"~{chips / D:.1f}"
    P = len(x)
    x_range = f"{fmt(x[0])}, {fmt(x[-1])}"
    if spread:
        x_range = r"0,\,\sigma_{\mathrm{top}}(\mu_{L})"
    unit = f" {ch['xunit']}" if ch["xunit"] else ""
    line1 = (rf"$N$ = {n_txt}  |  Haar-random unitary $U$ (single Clements mesh)  |  one forward pass $y=Ux$  |  "
             r"$\hat{y}$ = complex output field, normalised to unit L2 norm  |  "
             r"$E=\Vert\hat{y}^{\,\mathrm{hw}}-\hat{y}^{\,\mathrm{ideal}}\Vert_{2}$")
    line2 = f"Inputs per unitary: {T} types x {D} random draws ({types_txt})  |  {shown}"
    line3 = (rf"{chips} hardware realizations per unitary per point (each chip sees one input of every type; "
             rf"each input used on {reuse} chips)  |  {M} Haar unitaries  |  {P} points{' per curve' if spread else ''} in "
             rf"${ch['sym']}\in[{x_range}]${unit}  |  {fam['curve_txt']}")
    line4 = ch["model"]
    line5 = (r"Each point: $E_{\mathrm{rms}}=\sqrt{\langle E^{2}\rangle}$ over unitaries, chips and input types;  band: $\pm1$ s.d. across Haar-random targets (s.e. in the CSV)")
    if spread:
        line5 += (r"  |  $\sigma_{\mathrm{eff}}=0$: delta at $\mu_L$ (starts at $E_L(\mu_L,N)$)  |  "
                  + (r"$\sigma_{\mathrm{top}}=\min(0.95\,\mu_L/\sqrt{3},0.263)$ dB" if fam["apply_cap"]
                     else r"$\sigma_{\mathrm{top}}=0.95\,\mu_L/\sqrt{3}$, exploratory above 0.263 dB"))
    return line1, line2, line3, line4, line5


def make_figure(cfg, fam, results):
    """One figure: E_rms vs the swept parameter, all input types pooled, one curve per entry of the family."""
    _style()
    ch = CHANNELS[cfg["CHANNEL"]]
    x = fam["x"]
    fig, ax = plt.subplots(figsize=cfg["FIG_SIZE"])
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.245, top=0.775)
    spread = fam["label_style"] == "sat"

    # --- the curves ---
    entries = _curves_to_plot(fam, results)
    handles = _draw_entries(ax, cfg, fam, entries)

    # --- axis limits ---
    lo = min(h.get_ydata().min() for h in handles)
    hi = max(h.get_ydata().max() for h in handles)
    span = hi - min(0.0, lo)
    ax.set_ylim(min(0.0, lo - 0.03 * span), hi + 0.06 * span)
    ax.set_xlim(x[0], max(e[2][-1] for e in entries) * (1.015 if spread else 1.0))
    if spread and not fam["apply_cap"]:                       # exploratory range: shade the region above the validated cap
        cap = mesh.SIGMA_EFF_CAP_DB
        ax.axvline(cap, color="0.25", linestyle="--", linewidth=1.2, zorder=2)
        ax.axvspan(cap, ax.get_xlim()[1], color="0.5", alpha=0.10, zorder=1)
        ax.text(0.5 * (cap + ax.get_xlim()[1]), ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                r"exploratory, not validated" "\n" r"($\sigma_{\mathrm{eff}}>0.263$ dB)", ha="center", va="bottom", fontsize=11, color="0.25")

    # --- labels, ticks, grid ---
    ylab = (r"RMS output error $E_{\mathrm{rms}}$" "\n"
            r"($E_{\mathrm{rms}}^{2}$ = mean of $E^{2}=\Vert\hat{y}^{\,\mathrm{hw}}-\hat{y}^{\,\mathrm{ideal}}\Vert^{2}$" "\n"
            "over random unitaries, hardware realizations and inputs)")
    ax.set_xlabel(ch["xlabel"], fontsize=12, labelpad=8)
    ax.set_ylabel(ylab, fontsize=12, labelpad=8)
    ax.tick_params(which="both", labelsize=14)
    ax.tick_params(which="major", length=6, width=0.9)
    ax.tick_params(which="minor", length=3, width=0.7)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)

    # --- legend below the plot (entries ordered column by column) ---
    n_h = len(handles)
    ncol = min(fam["ncol"], n_h)
    nrow = -(-n_h // ncol)
    order = [r_ * ncol + c_ for c_ in range(ncol) for r_ in range(nrow) if r_ * ncol + c_ < n_h]
    legend_title = fam["legend_title"]
    if spread:
        rule = (r"$\sigma_{\mathrm{top}}=\min(0.95\,\mu_L/\sqrt{3},\,0.263\ \mathrm{dB})$" if fam["apply_cap"]
                else r"$\sigma_{\mathrm{top}}=0.95\,\mu_L/\sqrt{3}$ (0.263 dB cap lifted)")
        legend_title += (rf"  |  $E_0=E_{{\mathrm{{rms}}}}(\sigma_{{\mathrm{{eff}}}}{{=}}0)$,  $E_{{\max}}$ = $E_{{\mathrm{{rms}}}}$ at $\sigma_{{\mathrm{{top}}}}$ (open circle = end of curve);  " + rule)
    ax.legend(handles=[handles[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=ncol,
              fontsize=10, frameon=True, framealpha=0.95, edgecolor="0.4", fancybox=False,
              borderpad=0.7, labelspacing=0.55, columnspacing=1.8, handlelength=2.2,
              title=legend_title, title_fontsize=10)

    # --- title and caption ---
    fig.text(0.5, 0.975, fam["title"], ha="center", va="top", fontsize=20, fontweight="bold")
    for k, text in enumerate(_caption_lines(cfg, fam, results, ch)):
        fig.text(0.5, 0.925 - 0.0245 * k, text, ha="center", va="top", fontsize=10.5, color="0.25")
    return fig


def _overlay_pairs(results):
    """Pairs of independent runs that cover the same (mu_L, N): (label, reference run, other run)."""
    small_B, large_B, small_A = results["small_panelB"], results["large_panelB"], results["small_panelA"]
    pairs = []
    for rb in large_B:                                       # big range vs small range
        for rs in small_B:
            if abs(rb["fixed"]["mu_L"] - rs["fixed"]["mu_L"]) < 1e-12 and rb["N"] == rs["N"]:
                pairs.append((f"big vs small range, mu_L={rb['fixed']['mu_L']:g} dB, N={rb['N']}", rs, rb))
    for ra in small_A:                                       # N-panel vs mu_L-panel
        for rs in small_B:
            if abs(ra["fixed"]["mu_L"] - rs["fixed"]["mu_L"]) < 1e-12 and ra["N"] == rs["N"]:
                pairs.append((f"N-panel vs mu_L-panel (small range), mu_L={rs['fixed']['mu_L']:g} dB, N={ra['N']}", rs, ra))
    return pairs


def _overlay_row(label, ra, rb):
    """Compare two independent runs on the sigma_eff range they share; returns one row of the overlay-check table."""
    xa = ra["x"]
    in_overlap = xa <= min(xa[-1], rb["x"][-1]) * (1 + 1e-12)
    xo = xa[in_overlap]
    a_mat = ra["rho"][:, in_overlap]
    b_mat = np.array([np.interp(xo, rb["x"], rb["rho"][mi]) for mi in range(rb["rho"].shape[0])])
    d = a_mat - b_mat                                           # paired difference per unitary
    dm, dsem = d.mean(axis=0), d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])
    z = dm / np.where(dsem > 0, dsem, np.inf)
    return dict(comparison=label, overlap_sigma_eff_max_dB=float(xo[-1]), n_points=int(in_overlap.sum()),
                max_abs_dE=float(np.abs(dm).max()), max_abs_z=float(np.abs(z).max()), rms_z=float(np.sqrt(np.mean(z ** 2))),
                mean_abs_dE=float(np.abs(dm).mean()), n_unitaries=int(d.shape[0]),
                flagged=bool(np.sqrt(np.mean(z ** 2)) > 2.0 or np.abs(z).max() > 5.0))


def overlay_check(fams, results, outdir):
    """Consistency check between independent runs on the sigma_eff range they share.
    The longer grid is interpolated onto the shorter one.  The runs use the same unitaries but different
    hardware samples, so the paired difference per unitary measures sampling noise only (z = mean / s.e.m.).
    Writes sensitivity_sigma_L_overlay_check.csv / .json and prints the table."""
    by = {f["key"]: f for f in fams}
    if not all(k in by for k in ("small_panelA", "small_panelB", "large_panelB")):
        return None
    pairs = _overlay_pairs(results)
    if not pairs:
        return None
    rows = [_overlay_row(label, ra, rb) for label, ra, rb in pairs]

    with open(outdir / "sensitivity_sigma_L_overlay_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    note = "paired per-unitary difference z = mean/s.e.m.; flagged if RMS z > 2 (pure noise: ~1) or max |z| > 5; unreliable for < ~10 unitaries"
    with open(outdir / "sensitivity_sigma_L_overlay_check.json", "w") as f:
        json.dump(dict(comparisons=rows, note=note), f, indent=1)
    print("\nOVERLAY CHECK (independent runs on overlapping sigma_eff ranges):")
    for r in rows:
        flag = "   <-- FLAG: larger than a few standard errors" if r["flagged"] else ""
        print(f"  {r['comparison']}: overlap to {r['overlap_sigma_eff_max_dB']:.4g} dB ({r['n_points']} pts)  max |dE| = {r['max_abs_dE']:.3g}  "
              f"max |z| = {r['max_abs_z']:.2f}  RMS z = {r['rms_z']:.2f}{flag}")
    return rows


# ======================================================================================
# 8. SELF-TESTS  (small N, run before the sweeps)
# ======================================================================================

def _test_input_generators(rng):
    """V1: every input type has unit length and the right structure.  Returns the largest violation."""
    err = 0.0
    for N in (20, 57, 100):
        for t in ALL_INPUT_TYPES:
            X = gen_inputs(t, 10, N, rng)
            err = max(err, np.max(np.abs(np.linalg.norm(X, axis=1) - 1.0)))
            n_nonzero = (np.abs(X) > 0).sum(axis=1)
            if t == "binary_pm1":
                err = max(err, np.max(np.abs(np.abs(X) - 1 / np.sqrt(N))), np.max(np.abs(X.imag)))
            if t == "phase_only":
                err = max(err, np.max(np.abs(np.abs(X) - 1 / np.sqrt(N))))
            if t == "dense_real":
                err = max(err, np.max(np.abs(X.imag)))
            if t == "single_port":
                err = max(err, np.max(np.abs(n_nonzero - 1)))
            if t == "sparse_10pct":
                err = max(err, np.max(np.abs(n_nonzero - max(2, round(0.1 * N)))))
    return err


def run_validation_tests(cfg):
    """Quick self-tests.  Every error must be below 1e-9, otherwise the sweeps are not started."""
    print("=" * 78 + "\nVALIDATION TESTS\n" + "=" * 78)
    rng = np.random.default_rng(777)
    res = {}

    res["V1: input generators (unit norm, support size, binary/phase-only moduli)"] = _test_input_generators(rng)

    # V2: an error-free circuit must reproduce U x for every input type
    cfg_all = dict(cfg, CHIPS_PER_MATRIX=max(cfg["CHIPS_PER_MATRIX"], cfg["INPUTS_PER_TYPE"]), INPUT_TYPES=ALL_INPUT_TYPES)
    res["V2: error-free kernel reproduces U x for all input types"] = max(build_matrix(N, cfg_all, rng).residual for N in (12, 33, 60))

    # V3: E = 0 for the error-free circuit, through the full run_chips path
    b = build_matrix(40, cfg_all, rng)
    out = mesh.run_chips(b.dec, b.X3, cfg_all["CHIPS_PER_MATRIX"], rng=rng, Y_ideal=b.Y3)
    res["V3: E = 0 for the error-free circuit (run_chips path)"] = float(out["E_raw"].max())

    # V4: realised spread -> nominal width -> realised spread must give back the start value
    err = 0.0
    for mu_ in (0.05, 0.2, 0.5, 1.0, 3.0):
        for frac in (0.0, 0.1, 0.5, 0.8, 0.95, 0.99):
            se_ = frac * mesh.sigma_eff_cap(mu_)
            err = max(err, abs(mesh.sigma_eff_trunc(mu_, mesh.sigma_nominal_from_eff(mu_, se_)) - se_))
    res["V4: sigma_eff -> nominal sigma_L -> sigma_eff round trip"] = err

    # V5: sigma_eff_top = min(0.95 mu/sqrt3, 0.263) agrees with the reference table, and all grids lie inside [0, top]
    table = {0.020: 0.0110, 0.05: 0.0274, 0.10: 0.0548, 0.15: 0.0823, 0.20: 0.1097, 0.25: 0.1371, 0.35: 0.1920, 0.40: 0.2194,
             0.50: 0.263, 0.51: 0.263, 0.75: 0.263, 1.00: 0.263}
    e_table = max(abs(mesh.sigma_eff_top(k) - v) for k, v in table.items())
    e_formula = max(abs(mesh.sigma_eff_top(k) - min(0.95 * k / np.sqrt(3.0), 0.263)) for k in table)
    e_array = float(np.max(np.abs(mesh.sigma_eff_top(np.array(list(table))) - np.array([mesh.sigma_eff_top(k) for k in table]))))
    res["V5a: sigma_eff_top vs formula / array == scalar / mu = 0"] = max(e_formula, e_array, abs(mesh.sigma_eff_top(0.0)))
    res["V5b: sigma_eff_top vs reference table (table has 4 decimals)"] = max(0.0, e_table - 5.0e-5)
    if cfg["CHANNEL"] == "sigma_L":
        violation = 0.0
        for fam in build_families(cfg):
            for curve in fam["curves"]:
                mu_c = curve["fixed"]["mu_L"]
                limit = float(mesh.sigma_eff_top(mu_c)) if fam["apply_cap"] else mesh.REACH_FRAC * mesh.sigma_eff_cap(mu_c)
                violation = max(violation, float(curve["x"].max()) - limit, -float(curve["x"].min()),
                                float(curve["x"].max()) - 0.99 * mesh.sigma_eff_cap(mu_c))
        res["V5c: every spread grid satisfies 0 <= x <= top (< 0.99 mu/sqrt3)"] = max(0.0, violation)

    ok = True
    for name, e in res.items():
        flag = "OK" if e < 1e-9 else "FAIL"
        ok &= e < 1e-9
        print(f"  [{flag}] {name:<72s} max err = {e:.2e}")
    print("=" * 78)
    if not ok:
        raise AssertionError("Validation failed; not running the sweeps.")
    print("All validation tests passed.\n")


# ======================================================================================
# 9. DRIVER
# ======================================================================================

def _print_header(cfg, fams, outdir):
    """Print the summary of the planned run (curves, N values, unitaries, chips, input types, seed)."""
    ch = cfg["CHANNEL"]
    n_curves = sum(len(f["curves"]) for f in fams)
    print("=" * 78)
    print(f"CLEMENTS-MESH ERROR SENSITIVITY SWEEP: {ch}")
    print("=" * 78)
    for f in fams:
        print(f"family {f['key']:<28s}: {len(f['curves']):2d} curves, {CHANNELS[ch].get('col', ch)} in [0, {f['x'][-1]:g}], {len(f['x'])} points")
        if ch == "sigma_L":
            for c in (f["curves"][:1] if f["key"].endswith("panelA") else f["curves"]):
                mu_c = c["fixed"]["mu_L"]
                rule = "min(0.95 mu_L/sqrt3, 0.263)" if f["apply_cap"] else "0.95 mu_L/sqrt3, cap lifted"
                print(f"    mu_L = {mu_c:g} dB: sigma_eff in [0, {c['x'][-1]:.5g}] dB = sigma_top ({rule}; physical limit mu_L/sqrt3 = "
                      f"{mesh.sigma_eff_cap(mu_c):.5g});  nominal sigma_L at the end = {mesh.sigma_nominal_from_eff(mu_c, float(c['x'][-1])):.5g} dB")
    print(f"N values                : {cfg['N_VALUES']}")
    print(f"Unitaries x chips       : {cfg['N_MATRICES']} x {cfg['CHIPS_PER_MATRIX']}")
    print(f"Input types x draws     : {len(cfg['INPUT_TYPES'])} x {cfg['INPUTS_PER_TYPE']}  {cfg['INPUT_TYPES']}")
    print(f"Total curves            : {n_curves}   Seed: {cfg['SEED']}   Output: {outdir}")
    print("=" * 78)


def _make_bundles(cfg, N):
    """All random unitaries (with mesh and inputs) of one network size N."""
    print("Building Haar unitaries, Clements decompositions and inputs ...", flush=True)
    bundles = [build_matrix(N, cfg, np.random.default_rng([cfg["SEED"], 0, N, mi])) for mi in range(cfg["N_MATRICES"])]
    worst = max(b.residual for b in bundles)
    print(f"  worst ideal-circuit residual |U_mesh x - U x| over unitaries and inputs: {worst:.2e}")
    assert worst < 1e-8, "ideal circuit does not reproduce U x"
    return bundles


def _compute_all_curves(cfg, fams, ckdir):
    """Run (or load from checkpoint) every curve of every family.  Returns the aggregated results and the raw per-chip data."""
    ch = cfg["CHANNEL"]
    results = {f["key"]: [None] * len(f["curves"]) for f in fams}
    raw = {f["key"]: {} for f in fams}
    all_N = list(dict.fromkeys(c["N"] for f in fams for c in f["curves"]))
    for N in all_N:                                           # one network size at a time (the unitaries are built once per N)
        bundles = None
        print(f"\n{'-' * 78}\nNetwork size N = {N}\n{'-' * 78}")
        for f in fams:
            for c in f["curves"]:
                if c["N"] != N:
                    continue
                ci = c["ci"]
                base_par = dict(sigma_rad=0.0, sigma_S=0.0, mu_L=0.0, sigma_L=0.0)       # all channels off ...
                base_par.update(c["fixed"])                                              # ... except the fixed ones of this curve
                seed_key = (f["family_idx"], ci)
                x = c.get("x", f["x"])                    # per-curve grid (loss spread) or the family grid
                signature = curve_signature(cfg, N, base_par, f["swept"], x, seed_key)
                ckpt = ckdir / f"{f['stem']}_c{ci:02d}.npz"
                label = f"[{f['key']}] curve {ci + 1}/{len(f['curves'])} (N={N})"

                # 1. try to load a checkpoint
                loaded = False
                if cfg["RESUME"] and ckpt.exists():
                    z = np.load(ckpt)
                    if str(z["sig"]) == signature:
                        E, Ei, LT, E2 = z["E"], z["Ei"], z["LT"], z["E2"]
                        loaded = True
                        print(f"  {label}: loaded from checkpoint")
                # 2. otherwise simulate and save a checkpoint
                if not loaded:
                    if bundles is None:
                        bundles = _make_bundles(cfg, N)
                    print(f"\n{label}: swept {CHANNELS[ch].get('col', ch)}, fixed {c['fixed'] if c['fixed'] else '(none)'}")
                    E, Ei, LT, E2 = sweep_curve(cfg, bundles, x, base_par, f["swept"], seed_key, label)
                    np.savez_compressed(ckpt, E=E, Ei=Ei, LT=LT, E2=E2, sig=np.array(signature))

                # 3. statistics
                r = aggregate(cfg, f, c, x, E, Ei, LT, E2)
                results[f["key"]][ci] = r
                if f["label_style"] == "slope":
                    print(f"  {label}: slope (E_rms curve) = {r['slope']:.5g}   per-unitary: "
                          f"{r['slope_mem_mean']:.5g} +- {r['slope_mem_sem']:.2g} (s.e.m.)")
                else:
                    print(f"  {label}: E_rms(sigma_eff=0) = {r['E0']:.4f}   E_rms(sigma_eff={r['x'][-1]:.4g} dB, sigma_top) = "
                          f"{r['E_max']:.4f} +- {r['E_max_sem']:.2g}")
                if cfg["SAVE_PER_CHIP"]:
                    raw[f["key"]][f"E_chip_curve{ci}"] = E
    return results, raw


def _save_family(cfg, f, res, raw, outdir):
    """Write the CSV / npz files and the pooled figure (PDF) of one family."""
    for p in write_outputs(cfg, f, res, outdir, raw):
        print(f"Saved: {p}")
    fig = make_figure(cfg, f, res)
    pdf = outdir / f"{f['stem']}_pooled_all_types.pdf"
    fig.savefig(pdf)
    print(f"Saved: {pdf}")
    if not cfg["SHOW_PLOT"]:
        plt.close(fig)


def run(cfg):
    """The whole experiment: tests -> sweeps -> files and figures."""
    outdir = HERE / cfg["OUTPUT_DIR"]
    ckdir = outdir / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    bad = [t for t in cfg["INPUT_TYPES"] if t not in ALL_INPUT_TYPES]
    assert not bad, f"unknown input types: {bad}"
    if cfg["RUN_VALIDATION"]:
        run_validation_tests(cfg)

    fams = build_families(cfg)
    _print_header(cfg, fams, outdir)

    t_start = time.perf_counter()
    results, raw = _compute_all_curves(cfg, fams, ckdir)
    print(f"\nSweeps done in {(time.perf_counter() - t_start) / 60:.1f} min\n")

    for f in fams:
        _save_family(cfg, f, results[f["key"]], raw[f["key"]], outdir)
    if cfg["CHANNEL"] == "sigma_L":
        overlay_check(fams, results, outdir)
    print(f"\nTotal time: {(time.perf_counter() - t_start) / 60:.1f} min")
    if cfg["SHOW_PLOT"]:
        plt.show()


def apply_quick(cfg):
    """Tiny version of the experiment (about a minute) to check that everything runs."""
    import copy
    q = copy.deepcopy(cfg)
    q.update(N_VALUES=[20, 38], N_MATRICES=3, CHIPS_PER_MATRIX=10, INPUTS_PER_TYPE=5, N_POINTS=50,       # 50 points: later stages read points 2, 26, 50
             OUTPUT_DIR=cfg["OUTPUT_DIR"] + "_quick", RESUME=False)
    for rc in q["RANGES"].values():
        if "panel_B_N" in rc:
            rc["panel_B_N"] = 38
    if "SEPARABILITY_N" in q:
        q["SEPARABILITY_N"] = (20, 38)
    return q


def main_entry(cfg):
    """Called by the four sweep scripts:  python sweep_xxx.py [--quick]"""
    if "--quick" in sys.argv:
        print("QUICK MODE: tiny experiment, results go to a *_quick folder\n")
        cfg = apply_quick(cfg)
    run(cfg)
