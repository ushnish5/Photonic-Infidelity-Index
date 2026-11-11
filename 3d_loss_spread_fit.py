"""
3d_loss_spread_fit.py  --  scalable fit of the LOSS-SPREAD channel
==================================================================

The swept variable is the EFFECTIVE (realised) spread sigma_eff of the insertion loss
L ~ TruncNorm(mu_L, sigma_L^2; 0, 2 mu_L).  Model (same overlap-compatible form as the other channels):

    I_ph    = u_mu^2 + u_sigma^2                       (mean-loss term + spread term)
    E       = sqrt(2) * sqrt(1 - exp(-I_ph))
    u_mu    = (s_L(N) mu_L + gamma_L(N) mu_L^2) / sqrt(2)       FIXED: parameters of the mean-loss fit (3c_mean_loss_fit.py)
    u_sigma = s_sigma(N) * sigma_eff / sqrt(2),                  s_sigma(N) = s_sigma0 * (N / 100)^p_sigma

Only s_sigma0 and p_sigma are fitted.  sigma_eff is read directly from the column sigma_eff_dB of the CSV files of
2d_loss_spread_sweep.py (the sweep itself converts it to the nominal Gaussian width), so the fit needs no conversion.

What is fitted
    * the SMALL range, with BOTH small-range panels together (panel A: all N at mu_L = 0.5 dB, panel B: nine mu_L at N = 100),
      equal weight per point, in E space; the points with sigma_eff = 0 are not fitted (they carry no information on the spread);
    * every curve is ANCHORED at its own measured E(sigma_eff = 0), so the error of the mean-loss fit does not leak into the spread term;
    * every curve runs from 0 to its own sigma_top(mu_L) = min(0.95 mu_L / sqrt(3), 0.263 dB) and is fitted / drawn only on that domain;
    * the same parameters (no refit) are then shown on the LARGE range (the exploratory range: cap 0.263 dB lifted, qualitative only).
The full surrogate (fitted mean-loss term + spread term) is evaluated and plotted as well.

Diagnostics (printed and saved in the JSON): bootstrap over the unitaries, leave-one-curve-out, free exponent of the spread term,
the sigma_eff range each curve constrains, per-panel fits, collapse test of the mu_L curves, leave-one-mu_L-out and, if the optional
separability file of 2d exists, the separability cross-check.

Needs   results_mean_loss/fit_results_mu_L.json   (written by 3c_mean_loss_fit.py: run 3c before this script)
Reads   results_loss_spread/sensitivity_sigma_L_{small,large}_panel{A,B}_*_curves.csv   (from 2d_loss_spread_sweep.py)
Writes  results_loss_spread/:  fit_sigma_L_{small,large}_panel{A,B}.pdf,  fit_results_sigma_L.json,  fit_sigma_L_points.csv,
        jensen_gap_sigma_L.json.

Requires 3_common_fit_engine.py in the same folder.   Run:  python 3d_loss_spread_fit.py   (add --quick for the smoke test)

The file is organised like a book:
    1 settings and model   2 reading and checking the sweep files   3 the fit   4 bootstrap   5 diagnostics
    6 quality metrics   7 result files   8 figures   9 the driver  main()
"""
import csv
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

shared = importlib.import_module("0_shared_settings")  # noqa: E402
mesh = importlib.import_module("1_mzi_mesh")  # noqa: E402   (only used to verify the CSV files; NOT used by the model)
fit_engine = importlib.import_module("3_common_fit_engine")  # noqa: E402


# ======================================================================================
# 1. SETTINGS AND MODEL
# ======================================================================================
HERE = Path(__file__).resolve().parent
QUICK = "--quick" in sys.argv                                  # smoke test: read / write the *_quick folders
RESULT_FOLDER = HERE / ("results_loss_spread_quick" if QUICK else "results_loss_spread")
LOSS_FIT_JSON = HERE / ("results_mean_loss_quick" if QUICK else "results_mean_loss") / "fit_results_mu_L.json"
OUT_JSON = RESULT_FOLDER / "fit_results_sigma_L.json"

N0, SQRT2, E_CUT = shared.N0, shared.SQRT2, shared.E_CUT
SIGMA_EFF_RULE = "min(0.95*mu/sqrt(3), 0.263)"
FITTED_COLUMN = fit_engine.FITTED_COLUMN      # "E_rms"
N_BOOTSTRAP = fit_engine.N_BOOTSTRAP
BOOTSTRAP_SEED = fit_engine.BOOTSTRAP_SEED
INPUTS_PER_TYPE = fit_engine.INPUTS_PER_TYPE
FIGURE_SIZE = fit_engine.FIGURE_SIZE
MARKERS = fit_engine.MARKERS
PANELS = ("A", "B")                           # A: all N at fixed mu_L,  B: all mu_L at fixed N
RANGES = ("small", "large")


def E_of_I(I):
    """Error E_rms from the infidelity-like quantity I_ph."""
    return SQRT2 * np.sqrt(1.0 - np.exp(-I))


def I_of_E(E):
    """Inverse of E_of_I:  I_ph = -ln(1 - E^2 / 2)  (the argument is clipped to [1e-12, 1])."""
    return -np.log(np.clip(1.0 - np.asarray(E, float) ** 2 / 2.0, 1e-12, 1.0))


def mean_loss_term(mu_L, N, loss):
    """u_mu^2, the mean-loss part of I_ph, from the fixed parameters loss = (s_L0, p_L, gamma0, q_L)."""
    s_L0, p_L, gamma0, q_L = loss
    r = N / N0
    return ((s_L0 * r ** p_L * mu_L + gamma0 * r ** q_L * mu_L ** 2) / SQRT2) ** 2


def spread_term(sigma_eff, N, s0, p):
    """u_sigma^2, the spread part of I_ph."""
    return (s0 * (N / N0) ** p * sigma_eff / SQRT2) ** 2


def model_anchored(params, d, I_zero=None):
    """Spread term added on top of the MEASURED value I(sigma_eff = 0) of every curve."""
    base = d.I_zero.values if I_zero is None else I_zero
    return E_of_I(base + spread_term(d.sigma_eff.values, d.N.values, *params))


def model_full(params, d):
    """Full surrogate: fitted mean-loss term + spread term."""
    return E_of_I(d.I_mean_loss.values + spread_term(d.sigma_eff.values, d.N.values, *params))


# ======================================================================================
# 2. READING AND CHECKING THE SWEEP FILES
# ======================================================================================

def read_mean_loss_parameters():
    """The four fixed mean-loss parameters (s_L0, p_L, gamma0, q_L) from the 3c fit result."""
    if not LOSS_FIT_JSON.exists():
        sys.exit(f"missing {LOSS_FIT_JSON}\nrun 2c_mean_loss_sweep.py and 3c_mean_loss_fit.py first")
    with open(LOSS_FIT_JSON) as f:
        loss_fit = json.load(f)
    params = loss_fit["params"]
    loss = (params["s_{L,0}"], params["p_{L}"], params["\\gamma_{0}"], params["q_{L}"])
    print(f"mean-loss parameters used (fixed): s_L0={loss[0]:.5f}, p_L={loss[1]:.5f}, gamma0={loss[2]:.5f}, q_L={loss[3]:.5f}")
    return loss, params


def find_panel_csv(range_key, panel):
    """The one curves CSV of a range and panel."""
    hits = sorted(RESULT_FOLDER.glob(f"sensitivity_sigma_L_{range_key}_panel{panel}_*_curves.csv"))
    if not hits:
        raise SystemExit(f"missing file: {RESULT_FOLDER}/sensitivity_sigma_L_{range_key}_panel{panel}_*_curves.csv  (run 2d_loss_spread_sweep.py first)")
    if len(hits) > 1:
        raise SystemExit(f"more than one candidate for {range_key} panel {panel}: {[h.name for h in hits]}\n(keep only the files of one run in the folder)")
    return hits[0]


def check_sweep_rules(d, path, range_key):
    """Checks of the CSV (not part of the model): the stored nominal width reproduces the stored sigma_eff, and
    every curve runs exactly from 0 to its sigma_top(mu_L)."""
    sigma_eff = d["sigma_eff"].values
    back = np.array([mesh.sigma_eff_trunc(float(mu), float(sigma_nominal)) for mu, sigma_nominal in zip(d["mu_L"], d["sigma_nominal"])])
    assert np.max(np.abs(back - sigma_eff)) < 1e-8, f"{path.name}: nominal width in the CSV does not reproduce sigma_eff"
    if range_key == "small":
        top_rule = np.minimum(0.95 * d["mu_L"].values / np.sqrt(3.0), mesh.SIGMA_EFF_CAP_DB)       # calibration range: capped
    else:
        top_rule = 0.95 * d["mu_L"].values / np.sqrt(3.0)                                          # exploratory range: cap lifted
    rule_text = SIGMA_EFF_RULE if range_key == "small" else "0.95 mu/sqrt(3)"
    assert np.all(sigma_eff <= top_rule * (1 + 1e-9)), f"{path.name}: sigma_eff above sigma_top(mu_L) = {rule_text}"
    for curve, g in d.groupby("curve_index"):                                                      # each curve ends exactly at its sigma_top
        assert abs(g["sigma_eff"].max() - g["sigma_eff_top_dB"].iloc[0]) < 1e-9, f"{path.name}: curve {curve} does not end at sigma_top"
        assert abs(g["sigma_eff"].max() - top_rule[g.index[0]]) < 1e-9 * max(1.0, top_rule[g.index[0]]), f"{path.name}: curve {curve} grid is not 0..sigma_top"


def read_setup(path, d):
    """Number of chips and input types: from the npz file next to the CSV if it exists, else (30 chips, columns E_rms_<type>)."""
    chips, n_types = "30", len([c for c in d.columns if c.startswith("E_rms_")])
    npz_path = path.with_name(path.name.replace("_curves.csv", ".npz"))
    if npz_path.exists():
        npz = np.load(npz_path)
        chip_arrays = [name for name in npz.files if name.startswith("E_chip_curve")]
        if chip_arrays:
            chips, n_types = int(npz[chip_arrays[0]].shape[2]), int(npz[chip_arrays[0]].shape[3])
    return chips, n_types


def read_panel(range_key, panel):
    """One curves CSV, checked and with readable column names.

    Added / renamed columns:  mu_L, sigma_eff, sigma_nominal (renamed),  curve_id (panel + curve index),
    E_zero, I_zero (E_rms and I_ph of the curve at sigma_eff = 0: the anchor of the curve)."""
    path = find_panel_csv(range_key, panel)
    d = pd.read_csv(path)
    d = d.rename(columns={"fixed_mu_L": "mu_L", "sigma_eff_dB": "sigma_eff", "sigma_nominal_dB": "sigma_nominal"})
    unitary_columns = [c for c in d.columns if c.startswith("E2_mean_unitary_")]
    d["panel"] = panel
    d["curve_id"] = panel + d["curve_index"].astype(str)
    check_sweep_rules(d, path, range_key)
    d["E_zero"] = d["curve_id"].map(d[d.sigma_eff == 0].set_index("curve_id").E_rms)
    d["I_zero"] = I_of_E(d["E_zero"].values)
    chips, n_types = read_setup(path, d)
    setup = dict(unitary_columns=unitary_columns, M=len(unitary_columns), chips=chips, T=n_types, path=path)
    return d, setup


def read_range(range_key):
    """Both panels of a range, stacked into one table.  Returns (table, {panel: setup})."""
    tables, setups = [], {}
    for panel in PANELS:
        table, setups[panel] = read_panel(range_key, panel)
        tables.append(table)
    return pd.concat(tables, ignore_index=True), setups


# ======================================================================================
# 3. THE FIT
# ======================================================================================

def fit_spread(d, y=None, I_zero=None, start=None):
    """Fit (s_sigma0, p_sigma) to the anchored model; y, I_zero default to the data of the table d.

    Without `start` three starts (0.5, 1, 2 times [1, 0.5]) are tried and the best is kept."""
    y = d.E_rms.values if y is None else y
    if start is not None:
        starts = [np.array(start, float)]
    else:
        starts = [np.array([1.0, 0.5]) * factor for factor in (0.5, 1.0, 2.0)]

    def residuals(params):
        """Residuals (model minus data) for the parameter vector `params`."""
        return model_anchored(params, d, I_zero) - y

    best = None
    for s in starts:
        result = least_squares(residuals, s, bounds=([0, -2], [100, 3]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or result.cost < best.cost:
            best = result
    return best.x


def fit_s0_with_fixed_p(d, p_fixed, start_s0):
    """Fit s_sigma0 only, with the exponent held fixed (panel B has a single N, so it cannot determine p)."""
    def residuals(params):
        """Residuals (model minus data) for the parameter vector `params`."""
        return model_anchored([params[0], p_fixed], d) - d.E_rms.values

    result = least_squares(residuals, [start_s0], bounds=([0], [100]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return float(result.x[0])


# ======================================================================================
# 4. BOOTSTRAP OVER THE UNITARIES
# ======================================================================================

def bootstrap_spread_fits(small, n_unitaries, best):
    """Refit after resampling the unitaries with replacement.  The anchors E(sigma_eff = 0) are recomputed from every resample.

    Returns an array (N_BOOTSTRAP, 2) of (s_sigma0, p_sigma)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    m2 = small[[c for c in small.columns if c.startswith("E2_mean_unitary_")]].values.astype(float)     # per-unitary mean of E^2
    m2 = np.where(np.isnan(m2), 0.0, m2)
    at_zero = (small.sigma_eff.values == 0)
    positive = small.sigma_eff.values > 0
    fits = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, n_unitaries, size=n_unitaries)
        resampled_E = np.sqrt(m2[:, pick].mean(axis=1))                                             # E_rms of the resample
        anchor_E = pd.Series(resampled_E[at_zero], index=small.curve_id.values[at_zero])
        anchor_I = I_of_E(small.curve_id.map(anchor_E).values)
        fits.append(fit_spread(small[positive].reset_index(drop=True), y=resampled_E[positive], I_zero=anchor_I[positive], start=best))
    return np.array(fits)


# ======================================================================================
# 5. DIAGNOSTICS
# ======================================================================================

def leave_one_curve_out(fit_table):
    """Refit without one curve at a time; returns an array (n_curves, 2)."""
    result = np.array([fit_spread(fit_table[fit_table.curve_id != curve]) for curve in fit_table.curve_id.unique()])
    print(f"  leave-one-curve-out range: s_sigma,0 in [{result[:, 0].min():.4f}, {result[:, 0].max():.4f}], "
          f"p_sigma in [{result[:, 1].min():.4f}, {result[:, 1].max():.4f}]")
    return result


def free_exponents(fit_table):
    """Free exponent q of I_excess ~ sigma_eff^q for every curve (the model says q = 2)."""
    exponents = []
    for _, g in fit_table.groupby("curve_id"):
        exponents.append(np.polyfit(np.log(g.sigma_eff), np.log(np.clip(g.I_excess_rms_over_x0, 1e-9, None)), 1)[0])
    print(f"  free exponent of I_excess vs sigma_eff (model: 2): min {min(exponents):.2f}, max {max(exponents):.2f}, mean {np.mean(exponents):.2f}")
    return exponents


def curve_domains(small):
    """Which sigma_eff range does every curve constrain?  (printed and returned for the JSON)"""
    print("\nWHICH sigma_eff RANGE EACH CURVE CONSTRAINS (small range; every curve is fitted only on its own domain):")
    domain = {}
    for curve, g in small.groupby("curve_id", sort=False):
        positive = g[g.sigma_eff > 0]
        info = dict(panel=g.panel.iloc[0], N=int(g.N.iloc[0]), mu_L=float(g.mu_L.iloc[0]),
                    sigma_eff_min_dB=float(positive.sigma_eff.min()), sigma_eff_max_dB=float(positive.sigma_eff.max()),
                    n_points=int(len(positive)), reaches_0p263=bool(abs(positive.sigma_eff.max() - mesh.SIGMA_EFF_CAP_DB) < 1e-9))
        domain[curve] = info
        print(f"   panel {info['panel']}  N={info['N']:4d}  mu_L={info['mu_L']:5.2f} dB   sigma_eff in [{info['sigma_eff_min_dB']:.4f}, "
              f"{info['sigma_eff_max_dB']:.4f}] dB ({info['n_points']} pts){'   reaches 0.263 dB' if info['reaches_0p263'] else ''}")
    print("   NOTE: the short curves (small mu_L) constrain only the INITIAL slope; curvature / saturation is constrained only by the curves that reach 0.263 dB (mu_L >= 0.5 dB).")
    return domain


def per_panel_fits(fit_table, s0, p):
    """Fit panel A alone, and panel B alone (with p fixed to the joint value); the joint fit is the reference."""
    fit_A = fit_table[fit_table.panel == "A"].reset_index(drop=True)
    fit_B = fit_table[fit_table.panel == "B"].reset_index(drop=True)
    params_A = fit_spread(fit_A)
    s0_B = fit_s0_with_fixed_p(fit_B, p, s0)
    result = {"A_only": {"s_sigma0": float(params_A[0]), "p_sigma": float(params_A[1])},
              "B_only_p_fixed_to_joint": {"s_sigma0": s0_B, "p_sigma": float(p)}}
    print(f"\nPER-PANEL FITS (comparison with the joint fit s_sigma0 = {s0:.5f}, p_sigma = {p:.5f}):  panel A only: s0 = {params_A[0]:.5f}, p = {params_A[1]:.5f};  "
          f"panel B only (p fixed): s0 = {s0_B:.5f}")
    return result


def excess_curves(panel_B):
    """For every mu_L curve of panel B: the excess over its sigma_eff = 0 value, in E and in I_ph."""
    curves = []
    for _, g in panel_B.groupby("curve_id", sort=False):
        curves.append(dict(mu=float(g.mu_L.iloc[0]), x=g.sigma_eff.values, dE=g.E_rms.values - g.E_rms.values[0],
                           dI=I_of_E(g.E_rms.values) - I_of_E(g.E_rms.values[0])))
    return curves


def compare_curve_pairs(curves):
    """Compare all pairs of excess curves on their common sigma_eff range.  Returns {"E": [...], "I": [...]} of pair statistics."""
    pairs = {"E": [], "I": []}
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            x_max = min(curves[i]["x"][-1], curves[j]["x"][-1])
            x_grid = np.linspace(0.0, x_max, 200)
            for kind in ("E", "I"):
                yi = np.interp(x_grid, curves[i]["x"], curves[i]["d" + kind])
                yj = np.interp(x_grid, curves[j]["x"], curves[j]["d" + kind])
                reference = np.sqrt(np.mean(0.5 * (yi ** 2 + yj ** 2)))
                pairs[kind].append(dict(mu_a=curves[i]["mu"], mu_b=curves[j]["mu"], common_sigma_eff_max=float(x_max),
                                        rms_diff=float(np.sqrt(np.mean((yi - yj) ** 2))),
                                        rel_rms=float(np.sqrt(np.mean((yi - yj) ** 2)) / max(reference, 1e-12))))
    return pairs


def collapse_test(small):
    """Do the excess curves of different mu_L at N = 100 (panel B) lie on top of each other?"""
    panel_B = small[small.panel == "B"]
    curves = excess_curves(panel_B)
    pairs = compare_curve_pairs(curves)
    collapse = {kind: dict(mean_pair_rms=float(np.mean([q["rms_diff"] for q in v])), max_pair_rms=float(np.max([q["rms_diff"] for q in v])),
                           mean_pair_rel_rms=float(np.mean([q["rel_rms"] for q in v])), pairs=v) for kind, v in pairs.items()}
    print(f"\nCOLLAPSE TEST (N = {int(panel_B.N.iloc[0])}, {len(curves)} mu_L curves, pairs compared on their common sigma_eff range):")
    for kind, name in (("E", "E-excess  E(s)-E(0)"), ("I", "I-excess  I_ph(s)-I_ph(0) (the quantity the model says depends on sigma_eff only)")):
        print(f"   {name}: mean pairwise RMS diff = {collapse[kind]['mean_pair_rms']:.5f}, max = {collapse[kind]['max_pair_rms']:.5f}, "
              f"mean relative RMS = {100 * collapse[kind]['mean_pair_rel_rms']:.1f} %")
    print("   (E-excess need not collapse even for an exact model: E is a concave function of I_ph, so equal I-increments give larger E-increments at small baselines.)")
    return collapse


def leave_one_mu_out(fit_table, best):
    """Remove one panel-B curve from the joint fit, refit, and predict the removed curve."""
    fit_B = fit_table[fit_table.panel == "B"].reset_index(drop=True)
    results = []
    for curve in fit_B.curve_id.unique():
        g = fit_table[fit_table.curve_id == curve]
        refit = fit_spread(fit_table[fit_table.curve_id != curve])
        prediction = model_anchored(refit, g)
        rmse_left_out = float(np.sqrt(np.mean((prediction - g.E_rms.values) ** 2)))
        rmse_in_sample = float(np.sqrt(np.mean((model_anchored(best, g) - g.E_rms.values) ** 2)))
        results.append(dict(mu_L=float(g.mu_L.iloc[0]), sigma_eff_max_dB=float(g.sigma_eff.max()), s0=float(refit[0]), p=float(refit[1]),
                            rmse_left_out=rmse_left_out, rmse_in_sample=rmse_in_sample,
                            max_abs_dev_left_out=float(np.max(np.abs(prediction - g.E_rms.values)))))
    print("\nLEAVE-ONE-mu_L-OUT (panel-B curve removed from the joint fit, then predicted; panel A stays in):")
    for q in results:
        print(f"   mu_L = {q['mu_L']:5.2f} dB (to {q['sigma_eff_max_dB']:.4f} dB): refit s0 = {q['s0']:.4f}, p = {q['p']:.4f};  RMSE on left-out curve = "
              f"{q['rmse_left_out']:.5f}  (in-sample {q['rmse_in_sample']:.5f}),  max|dE| = {q['max_abs_dev_left_out']:.4f}")
    return results


def separability_check():
    """Optional cross-check, only if the separability CSV of 2d exists: is the spread sensitivity the same for two mu_L at every N?

    Returns a dict for the JSON, or None if the file does not exist."""
    files = sorted(RESULT_FOLDER.glob("sensitivity_sigma_L_small_separability_check_curves.csv"))
    if not files:
        print("\n(separability cross-check not run: set SEPARABILITY_CHECK = True in 2d_loss_spread_sweep.py to enable it)")
        return None
    d = pd.read_csv(files[0])
    slope = {}
    for (mu, N), g in d.groupby(["fixed_mu_L", "N"]):
        x_limit = d[d.fixed_mu_L == d.fixed_mu_L.min()].sigma_eff_dB.max()                          # common range = shortest curve (smallest mu_L)
        g = g[(g.sigma_eff_dB > 0) & (g.sigma_eff_dB <= x_limit * (1 + 1e-12))]
        x, y = g.sigma_eff_dB.values, g.I_excess_rms_over_x0.values
        slope[(float(mu), int(N))] = float((x ** 2 * y).sum() / (x ** 4).sum())                     # I_excess = a * sigma_eff^2 through the origin
    mus, sizes = sorted({k[0] for k in slope}), sorted({k[1] for k in slope})
    ratios = {int(n): slope[(mus[0], n)] / slope[(mus[-1], n)] for n in sizes}
    result = dict(mu_L=mus, N=sizes, a_I_excess_per_sigma_eff2={f"mu{k[0]:g}_N{k[1]}": v for k, v in slope.items()},
                  ratio_small_over_large_mu={str(k): float(v) for k, v in ratios.items()},
                  relative_change_of_ratio_across_N=float(abs(ratios[sizes[-1]] / ratios[sizes[0]] - 1.0)))
    print(f"\nSEPARABILITY CROSS-CHECK: ratio of spread sensitivities a(mu_L={mus[0]:g})/a(mu_L={mus[-1]:g}) = "
          + ", ".join(f"{v:.3f} at N={k}" for k, v in ratios.items()) + f"  -> relative change across N = {100 * result['relative_change_of_ratio_across_N']:.1f} %")
    return result


# ======================================================================================
# 6. QUALITY METRICS
# ======================================================================================

def spread_metrics(tables, best):
    """fit_engine.metrics for every range and panel ("all", "A", "B"):
    anchored (spread term only), full surrogate (all points and off-plateau) and mean-loss term only."""
    result = {}
    for range_key, D in tables.items():
        positive = D[D.sigma_eff > 0]
        result[range_key] = {}
        for panel in ("all", "A", "B"):
            fitted = positive if panel == "all" else positive[positive.panel == panel]       # points with sigma_eff > 0
            every = D if panel == "all" else D[D.panel == panel]                              # all points
            result[range_key][panel] = dict(
                anchored=fit_engine.metrics(fitted.E_rms.values, model_anchored(best, fitted)),
                full=fit_engine.metrics(every.E_rms.values, model_full(best, every)),
                full_off_plateau=fit_engine.metrics(every.E_rms.values, model_full(best, every), E_CUT),
                mean_loss_only=fit_engine.metrics(every.E_rms.values, E_of_I(every.I_mean_loss.values)))
        print(f"[{range_key}] sigma_eff max = {D.sigma_eff.max():.3f} dB; points with sigma_eff>0: {len(positive)}")
        for panel in ("all", "A", "B"):
            for name, values in result[range_key][panel].items():
                if "RMSE" in values:
                    print(f"   {panel:3s} {name:17s}", {a: round(b, 5) for a, b in values.items() if a in ("n", "RMSE", "MAE", "R2", "MaxAbsDev")})
    return result


# ======================================================================================
# 7. RESULT FILES
# ======================================================================================

def write_results_json(results):
    """fit_results_sigma_L.json; `results` holds everything that was computed."""
    s0, p, sd, ci_low, ci_high = results["s0"], results["p"], results["sd"], results["ci_low"], results["ci_high"]
    out = dict(params={"s_{\\sigma,0}": s0, "p_{\\sigma}": p},
               params_bootstrap_sd={"s_{\\sigma,0}": float(sd[0]), "p_{\\sigma}": float(sd[1])},
               params_ci95={"s_{\\sigma,0}": [float(ci_low[0]), float(ci_high[0])], "p_{\\sigma}": [float(ci_low[1]), float(ci_high[1])]},
               fit_space="E (unweighted; I_ph computed from the fitted E afterwards)", sigma_eff_rule=SIGMA_EFF_RULE,
               validated_domain=dict(sigma_eff_dB="0 <= sigma_eff <= min(0.95*mu_L/sqrt(3), 0.263)", mu_L_dB=[0.0, 1.0], N=[20, 500]),
               fit_domain_per_curve=results["domain"], per_panel_fit=results["per_panel"], collapse_test=results["collapse"],
               leave_one_mu_L_out=results["leave_one_mu_out"], separability=results["separability"],
               fit_range=[0.0, results["x_small_max"]], evaluated_unrefitted_up_to_sigma_eff_dB=results["x_large_max"],      # sigma_eff in dB
               fitted_column=FITTED_COLUMN, N0=N0, plateau_cut=E_CUT, n_bootstrap=N_BOOTSTRAP, n_unitaries=results["n_unitaries"],
               metrics=results["metrics"],
               leave_one_curve_out=dict(s0_min=float(results["loco"][:, 0].min()), s0_max=float(results["loco"][:, 0].max()),
                                        p_min=float(results["loco"][:, 1].min()), p_max=float(results["loco"][:, 1].max())),
               free_exponent_I_excess=dict(min=float(min(results["exponents"])), max=float(max(results["exponents"])),
                                           mean=float(np.mean(results["exponents"]))),
               loss_params_used=results["loss_params"], loss_json=str(LOSS_FIT_JSON.name), jensen_gap=results["jensen"])
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)


def write_points_csv(tables, best):
    """Data and fitted values of every point."""
    with open(RESULT_FOLDER / "fit_sigma_L_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["range", "panel", "curve_index", "N", "mu_L", "sigma_eff", "sigma_nominal", "E_rms_data", "E_rms_full_surrogate", "E_rms_anchored"])
        for range_key, D in tables.items():
            full, anchored = model_full(best, D), model_anchored(best, D)
            for (_, row), value_full, value_anchored in zip(D.iterrows(), full, anchored):
                writer.writerow([range_key, row.panel, int(row.curve_index), int(row.N), f"{row.mu_L:.10g}", f"{row.sigma_eff:.10g}",
                                 f"{row.sigma_nominal:.10g}", f"{row.E_rms:.10g}", f"{value_full:.10g}", f"{value_anchored:.10g}"])


# ======================================================================================
# 8. FIGURES
# ======================================================================================
EQUATION_TEXT = (r"$E(\mu_{L},\sigma_{\mathrm{eff}},N)=\sqrt{2}\,\sqrt{1-e^{-(u_{\mu}^{2}+u_{\sigma}^{2})}}$,   $u_{\sigma}=\dfrac{s_{\sigma}(N)\,\sigma_{\mathrm{eff}}}{\sqrt{2}}$,   "
                 r"$s_{\sigma}(N)=s_{\sigma,0}\,(N/N_{0})^{p_{\sigma}}$,   $N_{0}=100$")
SYMBOLS_TEXT = (r"$u_{\mu}$ = mean-loss term of the loss fit ($\mu_{L}$ only);   $\sigma_{\mathrm{eff}}$ = swept variable = realised std of $L\sim\mathrm{TruncNorm}(\mu_{L},\sigma_{L}^{2};0,2\mu_{L})$ dB "
                r"(nominal $\sigma_{L}$ derived in the sweep; no conversion and no fitted parameter here)")
PARAMETERS_TEXT = r"$s_{\sigma,0}$ = spread sensitivity at $N_{0}=100$ (dB$^{-1}$)   |   $p_{\sigma}$ = network-size exponent of $s_{\sigma}$"


def draw_panel_axes(ax, d, panel, range_key, best, loss, x_small_max):
    """Data (markers) and full surrogate (lines) of every curve of one panel.  Returns the legend handles."""
    s0, p = best
    curve_ids = list(dict.fromkeys(d.curve_id))
    colours = plt.cm.viridis(np.linspace(0, 0.95, len(curve_ids)))
    handles = []
    for i, curve in enumerate(curve_ids):
        g = d[d.curve_id == curve]
        N, mu = g.N.iloc[0], g.mu_L.iloc[0]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(g.sigma_eff, g.E_rms, marker, ms=5, color=colours[i], ls="none", zorder=3)
        x_line = np.linspace(0, g.sigma_eff.max(), 600)                         # each curve is drawn up to its own largest reachable sigma_eff
        ax.plot(x_line, E_of_I(mean_loss_term(mu, N, loss) + spread_term(x_line, N, s0, p)), "-", color=colours[i], lw=1.6, zorder=2)
        ax.plot([x_line[-1]], [E_of_I(mean_loss_term(mu, N, loss) + spread_term(x_line[-1], N, s0, p))], "o", ms=8.5, mfc="white", mec=colours[i], mew=1.5, zorder=4)   # end of the curve
        if panel == "A":
            label = r"$N=%d,\ s_{\sigma}=%.3f\ \mathrm{dB}^{-1}$" % (N, s0 * (N / N0) ** p)
        else:
            label = r"$\mu_{L}=%g\ \mathrm{dB},\ \sigma_{\mathrm{eff}}\in[0,\,%.4g]\ \mathrm{dB}$" % (mu, g.sigma_eff.max())
        handles.append(plt.Line2D([], [], color=colours[i], marker=marker, ms=5, lw=1.6, label=label))
    if range_key == "large":
        ax.axvline(x_small_max, color="0.35", ls="--", lw=1.2, zorder=1)
        ax.axvspan(x_small_max, d.sigma_eff.max() * 1.03, color="0.5", alpha=0.10, zorder=0)
        ax.text(x_small_max, 0.02, r"  validated domain ends ($\sigma_{\mathrm{eff}}=%.3f$ dB); right of it: exploratory, not validated" % x_small_max,
                fontsize=11, color="0.25", va="bottom", ha="left")
    ax.set_ylim(-0.03, max(d.E_rms.max(), 1.0) * 1.06)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"Effective (realised) insertion-loss standard deviation, $\sigma_{\mathrm{eff}}$ (dB)", fontsize=15)
    ax.set_ylabel("RMS output error $E_{\\mathrm{rms}}$\n($E_{\\mathrm{rms}}^{2}$ = mean of the squared difference of unit-normalised\noutputs over unitaries, chips and inputs)", fontsize=12.5)
    ax.tick_params(labelsize=14)
    return handles


def panel_text_lines(range_key, best, sd, quality, setup):
    """The (text, font size) lines under the figure title."""
    s0, p = best
    m_anchored, m_full, m_mean_loss, m_off_plateau = quality["anchored"], quality["full"], quality["mean_loss_only"], quality["full_off_plateau"]
    start = "Quality on this panel:  " if range_key == "small" else "Quality on this panel (parameters NOT refitted):  "
    full_text = start + r"full surrogate (lines): RMSE$=%.5f$, MAE$=%.5f$, $R^{2}=%.5f$, max$|\Delta E|=%.4f$" % (
        m_full["RMSE"], m_full["MAE"], m_full["R2"], m_full["MaxAbsDev"])
    if m_off_plateau["n"] >= 3:
        full_text += r";  off-plateau ($E\leq%g$, %d pts): RMSE$=%.5f$" % (E_CUT, m_off_plateau["n"], m_off_plateau["RMSE"])
    return [("Model (overlap-compatible, scalable in $N/N_{0}$):  " + EQUATION_TEXT, 13), (SYMBOLS_TEXT, 10.5), (PARAMETERS_TEXT, 12),
            (r"Best-fit values (small range, panels A+B, equal weight per point; $\pm$ = bootstrap s.d. over unitaries):   "
             r"$s_{\sigma,0}=%.5f\pm%.5f\ \mathrm{dB}^{-1}$,   $p_{\sigma}=%.5f\pm%.5f$" % (s0, sd[0], p, sd[1]), 12.5),
            (full_text, 11.5),
            (r"spread term alone, anchored at the measured $E(\sigma_{\mathrm{eff}}{=}0)$: RMSE$=%.5f$, $R^{2}=%.5f$, max$|\Delta E|=%.4f$   |   "
             r"mean-loss term only (no spread term): RMSE$=%.5f$, $R^{2}=%.5f$" % (m_anchored["RMSE"], m_anchored["R2"], m_anchored["MaxAbsDev"], m_mean_loss["RMSE"], m_mean_loss["R2"]), 11),
            (r"Haar-random unitary $U$ (single Clements mesh)  |  %s input types $\times$ %d draws, pooled  |  one forward pass $y=Ux$  |  "
             r"$E=\|\hat{y}^{\,\mathrm{hw}}-\hat{y}^{\,\mathrm{ideal}}\|_{2}$  |  %d unitaries $\times$ %s chips  |  50 points per curve"
             % (setup["T"], INPUTS_PER_TYPE, setup["M"], setup["chips"]), 10)]


def legend_title(d, panel, range_key):
    """Title of the legend box."""
    if panel == "A":
        first = r"Network size $N$  |  $\sigma_{\mathrm{eff}}\in[0,\sigma_{\mathrm{top}}]=[0,%.4g]$ dB, " % d.sigma_eff.max()
    else:
        first = r"Mean loss $\mu_{L}$ ($N=%d$)  |  each curve $\sigma_{\mathrm{eff}}\in[0,\sigma_{\mathrm{top}}(\mu_{L})]$, " % int(d.N.iloc[0])
    if range_key == "small":
        rule = r"$\sigma_{\mathrm{top}}=\min(0.95\,\mu_{L}/\sqrt{3},\,0.263\ \mathrm{dB})$"
    else:
        rule = r"$\sigma_{\mathrm{top}}=0.95\,\mu_{L}/\sqrt{3}$ (cap lifted)"
    return first + rule + r"  |  markers: data, lines: full surrogate, circles: end of curve"


def make_panel_figure(d, panel, range_key, best, sd, loss, quality, setup, x_small_max):
    """One figure: one panel of one range."""
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_axes([0.075, 0.27, 0.905, 0.395])
    handles = draw_panel_axes(ax, d, panel, range_key, best, loss, x_small_max)
    if panel == "A":
        fixed = r"fixed $\mu_{L}=%g$ dB, varying $N$" % d.mu_L.iloc[0]
    else:
        fixed = r"fixed $N=%d$, varying $\mu_{L}$" % int(d.N.iloc[0])
    fig.text(0.5, 0.985, "Scalable fit of the output-error sensitivity of a Clements unitary mesh vs. effective insertion-loss spread (%s)" % fixed,
             ha="center", va="top", fontsize=16, fontweight="bold")
    if range_key == "small":
        subtitle = r"FIT on the small range $\sigma_{\mathrm{eff}}\in[0,%.3f]$ dB (panels A and B together)" % x_small_max
    else:
        subtitle = r"large (exploratory) range $\sigma_{\mathrm{eff}}\in[0,%.4g]$ dB, 0.263 dB cap lifted: small-range parameters, no refit, qualitative only" % d.sigma_eff.max()
    fig.text(0.5, 0.953, subtitle, ha="center", va="top", fontsize=14, fontweight="bold")
    fit_engine.write_text_block(fig, panel_text_lines(range_key, best, sd, quality, setup))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, fontsize=12, frameon=True, edgecolor="#555555",
               fancybox=False, title=legend_title(d, panel, range_key), title_fontsize=12.5, columnspacing=2.2, handlelength=2.6)
    return fig


def make_all_figures(tables, setups, metrics_table, best, sd, loss, x_small_max):
    """The four figures fit_sigma_L_{small,large}_panel{A,B}.pdf."""
    fit_engine.set_plot_style()
    for range_key in RANGES:
        for panel in PANELS:
            d = tables[range_key][tables[range_key].panel == panel]
            fig = make_panel_figure(d, panel, range_key, best, sd, loss, metrics_table[range_key][panel], setups[range_key][panel], x_small_max)
            fit_engine.save_pdf(fig, RESULT_FOLDER, f"fit_sigma_L_{range_key}_panel{panel}")
            plt.close(fig)


# ======================================================================================
# 9. THE DRIVER
# ======================================================================================

def main():
    """Read the sweeps, fit the spread term, run all diagnostics and write the result files and figures."""
    loss, loss_params_in_json = read_mean_loss_parameters()

    # --- read both ranges; every curve gets the mean-loss term of the fixed 3c fit ---
    tables, setups = {}, {}
    for range_key in RANGES:
        tables[range_key], setups[range_key] = read_range(range_key)
    for D in tables.values():
        D["I_mean_loss"] = mean_loss_term(D["mu_L"].values, D["N"].values, loss)

    # --- fit on the small range (panels A + B together, sigma_eff > 0) ---
    small = tables["small"]
    fit_table = small[small.sigma_eff > 0].reset_index(drop=True)
    best = fit_spread(fit_table)
    s0, p = float(best[0]), float(best[1])
    x_small_max = float(small.sigma_eff.max())
    print(f"FIT ON SMALL RANGE (sigma_eff in [0, {x_small_max:g}] dB, panels A+B, {len(fit_table)} points):  "
          f"s_sigma,0 = {s0:.6f} dB^-1   p_sigma = {p:.6f}")

    # --- uncertainty and stability ---
    n_unitaries = setups["small"]["A"]["M"]
    boots = bootstrap_spread_fits(small, n_unitaries, best)
    sd = boots.std(axis=0, ddof=1)
    ci_low, ci_high = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    print(f"  bootstrap s.d.: s_sigma,0 +- {sd[0]:.5f}, p_sigma +- {sd[1]:.5f}")
    loco = leave_one_curve_out(fit_table)
    exponents = free_exponents(fit_table)

    # --- diagnostics ---
    domain = curve_domains(small)
    per_panel = per_panel_fits(fit_table, s0, p)
    collapse = collapse_test(small)
    leave_one_mu = leave_one_mu_out(fit_table, best)
    separability = separability_check()

    # --- quality metrics and result files ---
    metrics_table = spread_metrics(tables, best)
    jensen = fit_engine.jensen_report(RESULT_FOLDER, "sigma_L", tables)
    write_results_json(dict(s0=s0, p=p, sd=sd, ci_low=ci_low, ci_high=ci_high, domain=domain, per_panel=per_panel, collapse=collapse,
                            leave_one_mu_out=leave_one_mu, separability=separability, x_small_max=x_small_max,
                            x_large_max=float(tables["large"].sigma_eff.max()), n_unitaries=n_unitaries, metrics=metrics_table,
                            loco=loco, exponents=exponents, loss_params=loss_params_in_json, jensen=jensen))
    write_points_csv(tables, best)
    make_all_figures(tables, setups, metrics_table, best, sd, loss, x_small_max)
    print(f"saved: {OUT_JSON.name}, fit_sigma_L_points.csv, fit_sigma_L_{{small,large}}_panel{{A,B}}.pdf in {RESULT_FOLDER}")


if __name__ == "__main__":
    main()
