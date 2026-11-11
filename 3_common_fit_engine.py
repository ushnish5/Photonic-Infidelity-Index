"""
3_common_fit_engine.py  --  scalable single-channel fits of the sweep data
==========================================================================

The question: can the error curves of the single-channel sweeps (2a, 2b, 2c) be described by one
formula that is scalable in the network size N?  We fit

    E_rms(x, N) = sqrt(2) * sqrt(1 - exp(-u^2))

where x is the swept error (sigma_rad, sigma_S or mu_L) and u is small for small errors and grows with x:

    phase, splitting :  u = s(N) * x / sqrt(2),                          s(N) = s0 * (N / N0)^p
    mean loss        :  u = (s_L(N) * mu_L + gamma(N) * mu_L^2) / sqrt(2),
                         s_L(N) = s_L0 * (N / N0)^p_L,   gamma(N) = gamma0 * (N / N0)^q_L

(N0 = 100).  E_rms = sqrt(mean of E^2 over unitaries, chips and input types) is the column "E_rms" of the
sweep CSV files.  The fit is done IN E SPACE, unweighted, to the SMALL range of all N together; the SAME
parameters are then evaluated on the LARGE range without refitting.

The three scripts 3a / 3b / 3c only call  run_fit(channel, HERE, suffix)  from this file.
(3d_loss_spread_fit.py has its own model but re-uses  metrics, jensen_report, write_text_block, save_pdf  from here.)

Files written next to the sweep CSV files of the channel:
    fit_<channel>_small.pdf          fit on the small range
    fit_<channel>_large.pdf          same parameters on the large range (no refit)
    fit_results_<channel>.json       parameters, bootstrap uncertainties, metrics, Jensen-gap summary
    fit_<tag>_points.csv             data and fitted values point by point
    jensen_gap_<channel>.json        Jensen-gap summary (mean E versus root-mean-square E)

The file is organised like a book:
    1 the model   2 the fit   3 quality metrics   4 channel descriptions   5 reading the sweep files
    6 bootstrap   7 Jensen-gap summary   8 fit figures   9 the driver  run_fit
"""
import csv
import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

shared = importlib.import_module("0_shared_settings")  # noqa: E402
N0, SQRT2, E_CUT = shared.N0, shared.SQRT2, shared.E_CUT

# Settings of the fit (the same for all channels).
FITTED_COLUMN = "E_rms"       # the curve that is fitted
N_BOOTSTRAP = 200             # bootstrap resamples over the unitaries (parameter uncertainty)
BOOTSTRAP_SEED = 2026
INPUTS_PER_TYPE = 10          # only used in the figure description
FIGURE_SIZE = (17, 11.5)
MARKERS = ["o", "^", "s", "D", "v", "<", ">", "P", "X", "*", "h"]    # one marker per network size


def save_pdf(fig, folder, stem):
    """Save the figure as <stem>.pdf in the folder."""
    fig.savefig(folder / f"{stem}.pdf")


# ======================================================================================
# 1. THE MODEL
# ======================================================================================
# "lin"  = one error channel with a linear argument (phase, splitting)
# "loss" = mean loss, linear + quadratic argument

def E_of_u(u):
    """The saturating shape: E grows like sqrt(2) * u for small u and tends to sqrt(2)."""
    return SQRT2 * np.sqrt(1.0 - np.exp(-u ** 2))


def u_linear(x, N, s0, p):
    """u for the phase and splitting channels."""
    return s0 * (N / N0) ** p * x / SQRT2


def u_loss(x, N, s0, p, g0, q):
    """u for the mean-loss channel."""
    return (s0 * (N / N0) ** p * x + g0 * (N / N0) ** q * x ** 2) / SQRT2


def predict(kind, x, N, params):
    """Model value E_rms(x, N) for the parameter list `params`."""
    if kind == "loss":
        return E_of_u(u_loss(x, N, *params))
    return E_of_u(u_linear(x, N, *params))


# ======================================================================================
# 2. THE FIT
# ======================================================================================
# (typical start values, lower bounds, upper bounds) of the parameters
START_AND_BOUNDS = {
    "lin": ([10.0, 0.5], [0.0, -2.0], [100.0, 2.0]),                                       # s0, p
    "loss": ([0.19, 0.3, 0.08, 0.6], [0.0, -2.0, 0.0, -2.0], [10.0, 2.0, 10.0, 3.0]),      # s_L0, p_L, gamma0, q_L
}


def fit_params(kind, x, N, y, start=None):
    """Least-squares fit of the model to y (E space, unweighted); returns the best parameter vector.

    Without `start` three starts (0.5, 1, 2 times the typical values) are tried and the best is kept;
    with `start` (used by the bootstrap) only that start is used."""
    typical, lower, upper = START_AND_BOUNDS["loss" if kind == "loss" else "lin"]
    if start is not None:
        starts = [np.array(start, float)]
    else:
        starts = [np.array(typical) * factor for factor in (0.5, 1.0, 2.0)]

    def residuals(params):
        """Residuals (model minus data) for the parameter vector `params`."""
        return predict(kind, x, N, params) - y

    best = None
    for s in starts:
        result = least_squares(residuals, s, bounds=(lower, upper),
                               xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000)
        if best is None or result.cost < best.cost:
            best = result
    return best.x


# ======================================================================================
# 3. QUALITY METRICS
# ======================================================================================

def metrics(y, f, cut=None):
    """How well do the fitted values f describe the data y?  Optionally only for the points with y <= cut.

    RMSE, NRMSE (RMSE / range of y), MAE, R2, MaxAbsDev, MARE (mean relative error, only points with y > 1e-3)
    and MeanBias.  With fewer than 3 points only the point count n is returned."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    if cut is not None:
        keep = y <= cut
        y, f = y[keep], f[keep]
    if y.size < 3:
        return dict(n=int(y.size))
    r = y - f                                   # residuals
    big = y > 1e-3                              # points where a relative error makes sense
    rmse = np.sqrt(np.mean(r ** 2))
    return dict(n=int(y.size), RMSE=float(rmse),
                NRMSE=float(rmse / max(y.max() - y.min(), 1e-12)),
                MAE=float(np.mean(np.abs(r))),
                R2=float(1 - np.sum(r ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)),
                MaxAbsDev=float(np.max(np.abs(r))),
                MARE=float(np.mean(np.abs(r[big]) / y[big])) if big.any() else float("nan"),
                MeanBias=float(np.mean(r)))


# ======================================================================================
# 4. CHANNEL DESCRIPTIONS
# ======================================================================================
# One entry per fitted channel: where its files are, how the curve is named, and all texts of the figures.
#   kind            "lin" or "loss" (which model)
#   x_column        name of the swept column in the CSV
#   folder          result folder of the channel (written by the sweep, also receives the fit)
#   file_tag        short name used in fit_<file_tag>_points.csv
#   stem            sweep file name for a range, e.g. stem("small") + "_curves.csv"
#   x_label, unit   axis label and unit of the swept error
#   title, equation, parameter_lines, model_line   texts of the figure
#   parameter_names (LaTeX name, LaTeX unit) of each fitted parameter, in the order of the parameter vector
#   legend_label    legend text of one network size, from N and the fitted parameters
#   legend_columns  number of legend columns

CHANNEL_INFO = {
    "sigma_rad": dict(
        kind="lin", x_column="sigma_rad", folder="results_phase_error", file_tag="phase",
        stem=lambda rk: f"sensitivity_sigma_rad_{rk}",
        x_label=r"Phase-shifter error, $\sigma_{\mathrm{rad}}$ (rad)", unit="rad",
        title="Scalable fit of the output-error sensitivity of a Clements unitary mesh vs. phase-shifter error",
        equation=r"$E_{\phi}(\sigma_{\mathrm{rad}},N)=\sqrt{2}\,\sqrt{1-e^{-u_{\phi}^{2}}}$,   $u_{\phi}=\dfrac{s_{\phi}(N)\,\sigma_{\mathrm{rad}}}{\sqrt{2}}$,   $s_{\phi}(N)=s_{\phi,0}\,(N/N_{0})^{p_{\phi}}$,   $N_{0}=100$",
        parameter_lines=[r"$s_{\phi,0}$ = initial slope $dE/d\sigma_{\mathrm{rad}}$ at the reference size $N_{0}=100$ (rad$^{-1}$)",
                         r"$p_{\phi}$ = network-size scaling exponent of the sensitivity ($p_{\phi}\approx0.5$: $\sqrt{N}$ growth)"],
        parameter_names=[("s_{\\phi,0}", r"\mathrm{rad}^{-1}"), ("p_{\\phi}", "")],
        legend_label=lambda n, t: r"$N=%d,\ s_{\phi}=%.3f\ \mathrm{rad}^{-1}$" % (n, t[0] * (n / N0) ** t[1]), legend_columns=4,
        model_line=r"Hardware model: i.i.d. Gaussian phase errors $\delta\theta,\delta\phi\sim\mathcal{N}(0,\sigma_{\mathrm{rad}}^{2})$ on every MZI; all other channels ideal"),
    "sigma_S": dict(
        kind="lin", x_column="sigma_S", folder="results_splitting_ratio_error", file_tag="split",
        stem=lambda rk: f"sensitivity_sigma_S_{rk}",
        x_label=r"Splitting-ratio error, $\sigma_{S}$", unit="",
        title="Scalable fit of the output-error sensitivity of a Clements unitary mesh vs. splitting-ratio error",
        equation=r"$E_{S}(\sigma_{S},N)=\sqrt{2}\,\sqrt{1-e^{-u_{S}^{2}}}$,   $u_{S}=\dfrac{s_{S}(N)\,\sigma_{S}}{\sqrt{2}}$,   $s_{S}(N)=s_{S,0}\,(N/N_{0})^{p_{S}}$,   $N_{0}=100$",
        parameter_lines=[r"$s_{S,0}$ = initial slope $dE/d\sigma_{S}$ at the reference size $N_{0}=100$ (dimensionless)",
                         r"$p_{S}$ = network-size scaling exponent of the sensitivity ($p_{S}\approx0.5$: $\sqrt{N}$ growth)"],
        parameter_names=[("s_{S,0}", ""), ("p_{S}", "")],
        legend_label=lambda n, t: r"$N=%d,\ s_{S}=%.3f$" % (n, t[0] * (n / N0) ** t[1]), legend_columns=4,
        model_line=r"Hardware model: coupler splitting ratio $\eta\sim\mathrm{TruncNorm}(1/2,\sigma_{S}^{2};0,1)$ for both couplers of every MZI; all other channels ideal"),
    "mu_L": dict(
        kind="loss", x_column="mu_L", folder="results_mean_loss", file_tag="loss",
        stem=lambda rk: f"sensitivity_mu_L_{rk}_sigmaL0dB",
        x_label=r"Mean per-arm insertion loss, $\mu_{L}$ (dB)", unit="dB",
        title=r"Scalable fit of the output-error sensitivity of a Clements unitary mesh vs. mean insertion loss ($\sigma_{L}=0$)",
        equation=r"$E_{L}(\mu_{L},N)=\sqrt{2}\,\sqrt{1-e^{-u_{L}^{2}}}$,   $u_{L}=\dfrac{s_{L}(N)\,\mu_{L}+\gamma_{L}(N)\,\mu_{L}^{2}}{\sqrt{2}}$,   $s_{L}(N)=s_{L,0}\,(N/N_{0})^{p_{L}}$,   $\gamma_{L}(N)=\gamma_{0}\,(N/N_{0})^{q_{L}}$,   $N_{0}=100$",
        parameter_lines=[r"$s_{L,0}$ = initial linear sensitivity to loss at $N_{0}=100$ (dB$^{-1}$)   |   $p_{L}$ = network-size exponent of $s_{L}$",
                         r"$\gamma_{0}$ = quadratic (initial-convexity) coefficient at $N_{0}=100$ (dB$^{-2}$)   |   $q_{L}$ = network-size exponent of $\gamma_{L}$"],
        parameter_names=[("s_{L,0}", r"\mathrm{dB}^{-1}"), ("p_{L}", ""), ("\\gamma_{0}", r"\mathrm{dB}^{-2}"), ("q_{L}", "")],
        legend_label=lambda n, t: r"$N=%d,\ s_{L}=%.3f\ \mathrm{dB}^{-1},\ \gamma_{L}=%.3f\ \mathrm{dB}^{-2}$" % (n, t[0] * (n / N0) ** t[1], t[2] * (n / N0) ** t[3]), legend_columns=3,
        model_line=r"Hardware model: per-arm loss $L\sim\mathrm{TruncNorm}(\mu_{L},\sigma_{L}^{2};0,2\mu_{L})$ dB with $\sigma_{L}=0$ (deterministic loss); all other channels ideal"),
}


# ======================================================================================
# 5. READING THE SWEEP FILES
# ======================================================================================

def read_range(info, base, range_key, suffix):
    """Read the curves CSV of one range ("small" or "large") of a channel.

    Returns a dict with
        table             the whole CSV as a DataFrame
        unitary_columns   names of the columns E2_mean_unitary_XX (mean of E^2 of each unitary, used by the bootstrap)
        setup             number of unitaries M and input types T (and chips per unitary, if the npz file is there)
        path              path of the CSV
        x, N, y           swept error, network size and fitted curve of every row (float arrays)"""
    path = base / (info["folder"] + suffix) / f"{info['stem'](range_key)}_curves.csv"
    if not path.exists():
        raise SystemExit(f"missing file: {path}\n(run the corresponding sweep script first)")
    table = pd.read_csv(path)
    unitary_columns = [c for c in table.columns if c.startswith("E2_mean_unitary_")]
    setup = dict(M=len(unitary_columns), T=sum(1 for c in table.columns if c.startswith("E_rms_")))
    # the npz file next to the CSV knows the number of chips and input types exactly
    npz_path = path.with_name(path.name.replace("_curves.csv", ".npz"))
    if npz_path.exists():
        npz = np.load(npz_path)
        chip_arrays = [name for name in npz.files if name.startswith("E_chip_curve")]
        if chip_arrays:
            shape = npz[chip_arrays[0]].shape            # (M, P, chips, T)
            setup["chips"], setup["T"] = int(shape[2]), int(shape[3])
    return dict(table=table, unitary_columns=unitary_columns, setup=setup, path=path,
                x=table[info["x_column"]].values.astype(float), N=table["N"].values.astype(float),
                y=table[FITTED_COLUMN].values.astype(float))


# ======================================================================================
# 6. BOOTSTRAP OVER THE UNITARIES
# ======================================================================================

def bootstrap_fits(kind, small, best_params):
    """Refit after resampling the unitaries with replacement; returns an array (N_BOOTSTRAP, n_parameters).

    The E_rms of a resample is sqrt(mean of the resampled per-unitary E^2)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    m2 = small["table"][small["unitary_columns"]].values.astype(float)           # (rows, M)
    n_unitaries = m2.shape[1]
    fits = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, n_unitaries, size=n_unitaries)
        resampled_y = np.sqrt(m2[:, pick].mean(axis=1))
        fits.append(fit_params(kind, small["x"], small["N"], resampled_y, start=best_params))
    return np.array(fits)


# ======================================================================================
# 7. JENSEN-GAP SUMMARY
# ======================================================================================
# E_rms (root-mean-square) is never smaller than E_avg (plain mean); their difference is the "Jensen gap".

def jensen_summary(table):
    """Summary numbers of the Jensen gap of one range (largest and mean relative gap, ...)."""
    d = table
    valid = (d.E_rms > 1e-3) & np.isfinite(d.I_rms) & np.isfinite(d.I_avg)
    relative_gap = (d.jensen_gap_E / d.E_rms.where(d.E_rms > 1e-3)).where(valid)
    off_plateau = d.E_rms <= E_CUT
    return dict(n_points=int(len(d)), max_gap_E=float(d.jensen_gap_E.max()), max_rel_gap_E=float(np.nanmax(relative_gap)),
                max_rel_gap_E_off_plateau=float(np.nanmax(relative_gap.where(off_plateau))) if off_plateau.any() else float("nan"),
                mean_rel_gap_E_off_plateau=float(np.nanmean(relative_gap.where(off_plateau))) if off_plateau.any() else float("nan"),
                max_I_rms_minus_I_avg=float(np.nanmax((d.I_rms - d.I_avg).where(valid))),
                min_gap_E=float(d.jensen_gap_E.min()))


def jensen_report(outdir, channel, frames):
    """Jensen-gap summary of every range: how far is the root-mean-square E_rms above the plain mean E_avg?

    frames  {range name: DataFrame of the curves CSV}
    Saves jensen_gap_<channel>.json and returns the summary numbers per range."""
    summary = {range_key: jensen_summary(table) for range_key, table in frames.items()}
    with open(outdir / f"jensen_gap_{channel}.json", "w") as f:
        json.dump(dict(channel=channel, summary=summary), f, indent=1)
    worst = max(v["max_rel_gap_E_off_plateau"] for v in summary.values())
    print(f"  Jensen gap: largest relative gap (E_rms - E_avg)/E_rms off-plateau = {100 * worst:.1f} %   -> jensen_gap_{channel}.json")
    return summary


# ======================================================================================
# 8. FIT FIGURES
# ======================================================================================

def set_plot_style():
    """Serif fonts and inward ticks for all figures."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})


def setup_text(small):
    """One line that describes the simulation setup, from the small-range file."""
    setup = small["setup"]
    chips = setup.get("chips", "30")
    sizes = ",".join(str(int(n)) for n in np.unique(small["N"]))
    return (r"$N=\{%s\}$  |  Haar-random unitary $U$ (single Clements mesh)  |  %s input types $\times$ %d draws, pooled (equal weight)  |  one forward pass $y=Ux$  |  "
            r"$E_{\mathrm{rms}}=\sqrt{\langle\|\hat{y}^{\,\mathrm{hw}}-\hat{y}^{\,\mathrm{ideal}}\|_{2}^{2}\rangle}$  |  %d unitaries $\times$ %s chips  |  50 points per curve"
            % (sizes, setup["T"], INPUTS_PER_TYPE, setup["M"], chips))


def best_fit_text(info, params, sd):
    """The fitted parameters with their bootstrap s.d., as one line of text."""
    pieces = []
    for (name, unit), value, s in zip(info["parameter_names"], params, sd):
        pieces.append(r"$%s=%.5f\pm%.5f$" % (name, value, s) + (r"$\ %s$" % unit if unit else ""))
    return ",   ".join(pieces)


def quality_text(range_key, quality):
    """The 'fit quality' line: all points, and off-plateau points if there are at least 3."""
    all_points, off_plateau = quality["all"], quality["off_plateau"]
    start = "Fit quality (fit range):  " if range_key == "small" else "Quality on this range (parameters NOT refitted):  "
    text = start + r"all %d points: $\mathrm{RMSE}=%.5f$, $\mathrm{MAE}=%.5f$, $R^{2}=%.5f$, max$|\Delta E|=%.4f$" % (
        all_points["n"], all_points["RMSE"], all_points["MAE"], all_points["R2"], all_points["MaxAbsDev"])
    if off_plateau["n"] >= 3:
        text += r"   |   off-plateau ($E\leq%g$, %d points): $\mathrm{RMSE}=%.5f$, $R^{2}=%.5f$, mean rel. err. $=%.2f\%%$" % (
            E_CUT, off_plateau["n"], off_plateau["RMSE"], off_plateau["R2"], 100 * off_plateau["MARE"])
    return text


def write_text_block(fig, lines):
    """Write the (text, font size) lines centred under the figure title, from top to bottom."""
    y = 0.915
    for text, size in lines:
        fig.text(0.5, y, text, ha="center", va="top", fontsize=size, color="#333333")
        if text.startswith("Model"):
            y -= 0.052
        elif size >= 12.5:
            y -= 0.038
        else:
            y -= 0.031


def draw_fit_axes(ax, info, kind, params, data, small_x_max, range_key):
    """Markers (data) and solid lines (fit) of every network size, in one set of axes.  Returns the legend handles."""
    sizes = np.unique(data["N"])
    colours = plt.cm.viridis(np.linspace(0, 0.95, len(sizes)))
    x_line = np.linspace(0, data["x"].max(), 500)
    handles = []
    for i, n in enumerate(sizes):
        this_size = data["N"] == n
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(data["x"][this_size], data["y"][this_size], marker, ms=5, color=colours[i], ls="none", zorder=3)
        ax.plot(x_line, predict(kind, x_line, n, params), "-", color=colours[i], lw=1.6, zorder=2)
        handles.append(plt.Line2D([], [], color=colours[i], marker=marker, ms=5, lw=1.6, label=info["legend_label"](int(n), params)))
    if range_key == "large":
        ax.axvline(small_x_max, color="0.35", ls="--", lw=1.2, zorder=1)
        ax.text(small_x_max, 0.02, r"  fit range ends ($%g$ %s)" % (small_x_max, info["unit"]), fontsize=11, color="0.25", va="bottom", ha="left")
    ax.set_xlim(0, data["x"].max())
    ymax = 1.10 * max(data["y"].max(), data["fit"].max())       # 10% headroom above the highest data / fit value
    ax.set_ylim(-0.03 * ymax, ymax)
    ax.grid(alpha=0.25)
    ax.set_xlabel(info["x_label"], fontsize=15)
    ax.set_ylabel("RMS output error $E_{\\mathrm{rms}}$\n($E_{\\mathrm{rms}}^{2}$ = mean of the squared difference of unit-normalised\noutputs over unitaries, chips and inputs)", fontsize=12.5)
    ax.tick_params(labelsize=14)
    return handles


def make_fit_figure(info, params, sd, small, large, quality, range_key):
    """One figure: the data and the fit of one range, with the model description and the fit quality as text."""
    kind = info["kind"]
    data = small if range_key == "small" else large
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_axes([0.075, 0.27, 0.905, 0.395])
    handles = draw_fit_axes(ax, info, kind, params, data, small["x"].max(), range_key)

    if range_key == "small":
        range_text = r"FIT on the small range $[0,%g]$ %s" % (small["x"].max(), info["unit"])
    else:
        range_text = r"large range $[0,%g]$ %s with the small-range parameters (no refit)" % (large["x"].max(), info["unit"])
    fig.text(0.5, 0.985, info["title"], ha="center", va="top", fontsize=17, fontweight="bold", family="serif")
    fig.text(0.5, 0.953, range_text, ha="center", va="top", fontsize=14, fontweight="bold", family="serif")

    lines = [("Model (overlap-compatible, scalable in $N/N_{0}$):  " + info["equation"], 13)]
    lines += [(p, 12) for p in info["parameter_lines"]]
    lines.append(("Best-fit values (small range, all $N$, equal weight per point; $\\pm$ = bootstrap s.d. over unitaries):   " + best_fit_text(info, params, sd), 12.5))
    lines.append((quality_text(range_key, quality[range_key]), 11.5))
    lines.append((info["model_line"] + r"   |   markers = simulated $E_{\mathrm{rms}}$ (over unitaries, chips and input types), solid lines = scalable fit (fit in $E$ space)", 10.5))
    lines.append((setup_text(small), 10))
    write_text_block(fig, lines)

    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=info["legend_columns"], fontsize=12, frameon=True,
               edgecolor="#555555", fancybox=False,
               title=r"Network size $N$  |  scaled sensitivity from the fit (markers: data, lines: fit)", title_fontsize=12.5,
               columnspacing=2.2, handlelength=2.6)
    return fig


# ======================================================================================
# 9. THE DRIVER
# ======================================================================================

def write_points_csv(path, info, ranges):
    """Data and fitted value of every point: range, N, x, E_rms_data, E_rms_fit, residual."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["range", "N", info["x_column"], "E_rms_data", "E_rms_fit", "residual"])
        for range_key, data in ranges.items():
            for xv, nv, yv, fv in zip(data["x"], data["N"], data["y"], data["fit"]):
                writer.writerow([range_key, int(nv), f"{xv:.10g}", f"{yv:.10g}", f"{fv:.10g}", f"{yv - fv:.10g}"])


def run_fit(channel, base, suffix=""):
    """Fit one channel ("sigma_rad", "sigma_S" or "mu_L") and write all its result files.

    base    folder that holds the result folders (the folder of the scripts)
    suffix  "_quick" reads / writes the *_quick result folders (smoke test of the --quick sweeps)."""
    info = CHANNEL_INFO[channel]
    kind = info["kind"]
    outdir = base / (info["folder"] + suffix)
    set_plot_style()

    # --- read both ranges of the sweep ---
    ranges = {range_key: read_range(info, base, range_key, suffix) for range_key in ("small", "large")}
    small, large = ranges["small"], ranges["large"]

    # --- fit on the small range ---
    params = fit_params(kind, small["x"], small["N"], small["y"])
    names = [name for name, _ in info["parameter_names"]]
    print(f"[{channel}] best fit on the small range ({small['x'].max():g} {info['unit']}):",
          {n: round(float(v), 6) for n, v in zip(names, params)})

    # --- parameter uncertainty by bootstrap over the unitaries ---
    boots = bootstrap_fits(kind, small, params)
    sd = boots.std(axis=0, ddof=1)
    ci_low, ci_high = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    print("  bootstrap s.d.:", {n: round(float(v), 5) for n, v in zip(names, sd)})

    # --- evaluate the SAME parameters on both ranges ---
    quality = {}
    for range_key, data in ranges.items():
        data["fit"] = predict(kind, data["x"], data["N"], params)
        quality[range_key] = dict(all=metrics(data["y"], data["fit"]), off_plateau=metrics(data["y"], data["fit"], E_CUT))
        q = quality[range_key]
        print(f"  [{range_key}] all points: RMSE={q['all']['RMSE']:.5f} R2={q['all']['R2']:.5f} | "
              f"off-plateau (E<={E_CUT}, n={q['off_plateau']['n']}): "
              f"RMSE={q['off_plateau'].get('RMSE', float('nan')):.5f} R2={q['off_plateau'].get('R2', float('nan')):.5f}")

    # --- files: points CSV, Jensen-gap report, results JSON ---
    write_points_csv(outdir / f"fit_{info['file_tag']}_points.csv", info, ranges)
    jensen = jensen_report(outdir, channel, {range_key: data["table"] for range_key, data in ranges.items()})
    result = dict(channel=channel, params={n: float(v) for n, v in zip(names, params)},
                  params_bootstrap_sd={n: float(v) for n, v in zip(names, sd)},
                  params_ci95={n: [float(a), float(b)] for n, a, b in zip(names, ci_low, ci_high)},
                  fit_range=[0.0, float(small["x"].max())], evaluated_unrefitted_on=[0.0, float(large["x"].max())],
                  fitted_column=FITTED_COLUMN, fit_space="E (unweighted; I_ph is computed from the fitted E afterwards)",
                  median_se_E_rms=float(np.median(small["table"]["se_E_rms"])), N0=N0, plateau_cut=E_CUT, n_bootstrap=N_BOOTSTRAP,
                  n_unitaries=small["setup"]["M"], n_points_fit=int(small["x"].size), metrics=quality,
                  data_files=[small["path"].name, large["path"].name], jensen_gap=jensen)
    json_path = outdir / f"fit_results_{channel}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1)

    # --- figures ---
    for range_key in ("small", "large"):
        fig = make_fit_figure(info, params, sd, small, large, quality, range_key)
        save_pdf(fig, outdir, f"fit_{channel}_{range_key}")
        plt.close(fig)
    print(f"  saved: {json_path.name}, fit_{info['file_tag']}_points.csv, fit_{channel}_small/large.pdf  in {outdir}")
    return result
