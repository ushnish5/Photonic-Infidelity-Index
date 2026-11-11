"""
1_mzi_mesh.py  --  one Clements MZI mesh, with four kinds of hardware error
==================================================================================

What this file does, in the order you read it:

    1. Make a random unitary matrix U and random input vectors.
    2. Decompose U into a Clements mesh of MZIs (angles theta, phi + a final phase screen D).
    3. Draw random hardware errors for many "chips" (= many copies of the mesh).
    4. Push the inputs through every noisy chip.
    5. Compare each noisy output with the ideal output U @ x  ->  the error E.

The four error types (all injected while propagating, never in the decomposition):

    phase error     theta += theta_err / 2,  phi += phi_err      (both ~ N(0, sigma_rad^2))
                    (the internal shifter applies 2*theta, hence the factor 1/2)
    splitting error each coupler's power ratio eta ~ TruncNorm(1/2, sigma_S^2; 0, 1)
    mean loss       every arm loses L dB, L ~ TruncNorm(mu_L, sigma_L^2; 0, 2*mu_L)
    loss spread     the width of that same distribution.  We specify the REALISED spread
                    sigma_eff and compute the nominal width sigma_L from it.

Conventions (these fix the numbers, so they are never changed):
    MZI matrix       M(theta, phi) = [[gp*c, g*s], [gp*s, g*c]],
                     g = e^{i theta}, gp = g*e^{i phi}, c = cos(theta), s = i*sin(theta)
    Circuit          U = D * M_0 * M_1 * ... * M_{n-1}   (the kernel applies M_{n-1} first, D last)
    Output error     E = || y_hw/|y_hw|  -  y_ideal/|y_ideal| ||   (complex fields, no global-phase removal)
"""
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Optional

import numpy as np
from numba import njit, prange   # numba: implemented by Claude
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


# ======================================================================================
# 1. Containers and small helpers
# ======================================================================================

class Decomposition(NamedTuple):
    """The ideal Clements decomposition of one unitary.  MZI k acts on modes (m[k], m[k]+1)."""
    D: np.ndarray        # (N,)     final phase screen
    theta: np.ndarray    # (n_mzi,) internal phase of every MZI
    phi: np.ndarray      # (n_mzi,) external phase of every MZI
    m: np.ndarray        # (n_mzi,) upper mode index of every MZI


@dataclass
class ChipErrors:
    """Random errors for many chips.  Every array has shape (n_chips, n_mzi)."""
    theta_err: np.ndarray   # error of the internal shifter [rad]
    phi_err: np.ndarray     # error of the external shifter [rad]
    eta1: np.ndarray        # power splitting ratio, first coupler  (ideal 0.5)
    eta2: np.ndarray        # power splitting ratio, second coupler (ideal 0.5)
    amp1: np.ndarray        # amplitude transmission of the upper output arm
    amp2: np.ndarray        # amplitude transmission of the lower output arm

    @property
    def n_chips(self) -> int:
        """Number of chips described by the arrays."""
        return self.theta_err.shape[0]


def n_mzi(N: int) -> int:
    """Number of MZIs in an N-mode Clements mesh."""
    return N * (N - 1) // 2


def haar_unitary(N: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random N x N unitary (QR of a complex Gaussian matrix, with the phases of R fixed)."""
    Z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / math.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R)
    return Q * (d / np.abs(d))[np.newaxis, :]


# ======================================================================================
# 2. Clements decomposition  (compiled with numba, error-free)
# ======================================================================================
# Idea: multiply U from the right and from the left by 2x2 "nulling" blocks, diagonal by
# diagonal, until only a diagonal matrix D is left.  Every block becomes one MZI.

@njit(cache=True)   # numba: implemented by Claude
def _null_with_right_block(U, i, j, theta_list, phi_list, m_list, count):
    """Zero the entry U[i, j] by mixing columns j and j+1.  Stores the MZI angles."""
    two_pi = 2.0 * math.pi
    a = U[i, j]
    b = U[i, j + 1]
    if a == 0:
        return count                                   # already zero: no MZI needed
    if b == 0:
        phi = 0.0
    else:
        z = a / b
        phi = (math.atan2(z.imag, z.real) - math.pi / 2) % two_pi
    theta = math.atan2(abs(a), abs(b))
    theta_list[count] = theta
    phi_list[count] = phi
    m_list[count] = j

    # the 2x2 block that does the nulling
    g = complex(math.cos(theta), -math.sin(theta))               # e^{-i theta}
    gp = g * complex(math.cos(phi), -math.sin(phi))              # e^{-i theta} e^{-i phi}
    c = math.cos(theta)
    s = complex(0.0, -math.sin(theta))                           # -i sin(theta)
    h00 = gp * c
    h01 = gp * s
    h10 = g * s
    h11 = g * c
    for r in range(U.shape[0]):                                  # apply it to every row
        x0 = U[r, j]
        x1 = U[r, j + 1]
        U[r, j] = x0 * h00 + x1 * h10
        U[r, j + 1] = x0 * h01 + x1 * h11
    return count + 1


@njit(cache=True)   # numba: implemented by Claude
def _null_with_left_block(U, i, j, theta_list, phi_list, m_list, count):
    """Zero the entry U[i, j] by mixing rows i-1 and i.  Stores the MZI angles."""
    two_pi = 2.0 * math.pi
    a = U[i, j]
    b = U[i - 1, j]
    if a == 0:
        return count
    if b == 0:
        phi = 0.0
    else:
        z = a / b
        phi = (math.atan2(z.imag, z.real) + math.pi / 2) % two_pi
    theta = math.atan2(abs(a), abs(b))
    theta_list[count] = theta
    phi_list[count] = phi
    m_list[count] = i - 1

    g = complex(math.cos(theta), math.sin(theta))                # e^{i theta}
    gp = g * complex(math.cos(phi), math.sin(phi))               # e^{i theta} e^{i phi}
    c = math.cos(theta)
    s = complex(0.0, math.sin(theta))                            # i sin(theta)
    b00 = gp * c
    b01 = g * s
    b10 = gp * s
    b11 = g * c
    for col in range(U.shape[1]):                                # apply it to every column
        x0 = U[i - 1, col]
        x1 = U[i, col]
        U[i - 1, col] = b00 * x0 + b01 * x1
        U[i, col] = b10 * x0 + b11 * x1
    return count + 1


@njit(cache=True)   # numba: implemented by Claude
def _decompose_nb(U):
    """Returns D, theta, phi, m.  U is overwritten (it ends up diagonal)."""
    dim = U.shape[0]
    n_max = dim * (dim - 1) // 2
    two_pi = 2.0 * math.pi

    # scratch space for the MZIs found by right-multiplication (r_*) and left-multiplication (l_*)
    r_theta = np.empty(n_max)
    r_phi = np.empty(n_max)
    r_m = np.empty(n_max, dtype=np.int64)
    l_theta = np.empty(n_max)
    l_phi = np.empty(n_max)
    l_m = np.empty(n_max, dtype=np.int64)
    n_right = 0
    n_left = 0

    # --- sweep over the diagonals of U, alternating right and left blocks ---
    for k in range(1, dim):
        if k % 2 == 1:
            for t in range(k):
                n_right = _null_with_right_block(U, dim - 1 - t, k - 1 - t, r_theta, r_phi, r_m, n_right)
        else:
            for t in range(k):
                n_left = _null_with_left_block(U, dim - k + t, t, l_theta, l_phi, l_m, n_left)

    # --- what is left of U is the diagonal phase screen D ---
    D = np.empty(dim, dtype=np.complex128)
    for q in range(dim):
        D[q] = U[q, q]

    # --- move the left-hand MZIs through D so that D ends up on the output side ---
    new_l_phi = np.empty(n_left)
    for idx in range(n_left - 1, -1, -1):
        theta = l_theta[idx]
        phi = l_phi[idx]
        m = l_m[idx]
        n = m + 1
        angm = math.atan2(D[m].imag, D[m].real)
        angn = math.atan2(D[n].imag, D[n].real)
        new_l_phi[idx] = (math.pi + angm - angn) % two_pi
        new_beta = angn - 2.0 * theta
        new_alpha = angn - math.pi - phi - 2.0 * theta
        D[m] = complex(math.cos(new_alpha), math.sin(new_alpha))
        D[n] = complex(math.cos(new_beta), math.sin(new_beta))

    # --- final ordering: left MZIs in creation order, then right MZIs reversed ---
    n_total = n_left + n_right
    theta_out = np.empty(n_total)
    phi_out = np.empty(n_total)
    m_out = np.empty(n_total, dtype=np.int64)
    for idx in range(n_left):
        theta_out[idx] = l_theta[idx]
        phi_out[idx] = new_l_phi[idx]
        m_out[idx] = l_m[idx]
    for idx in range(n_right):
        theta_out[n_left + idx] = r_theta[n_right - 1 - idx]
        phi_out[n_left + idx] = r_phi[n_right - 1 - idx]
        m_out[n_left + idx] = r_m[n_right - 1 - idx]
    return D, theta_out, phi_out, m_out


def decompose(U: np.ndarray) -> Decomposition:
    """Ideal Clements decomposition:  U = D * M_0 * M_1 * ... * M_{n-1}."""
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be square.")
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
        raise ValueError("U must be a unitary matrix.")
    D, theta, phi, m = _decompose_nb(np.array(U, dtype=np.complex128, order="C"))
    return Decomposition(D=D, theta=theta, phi=phi, m=m)


# ======================================================================================
# 3. Random error samplers
# ======================================================================================

def truncnorm_sample(mu, sigma, lo, hi, size, rng):
    """Draw from N(mu, sigma^2) cut off to [lo, hi] (inverse-CDF method)."""
    if sigma == 0.0:
        return np.full(size, mu, dtype=np.float64)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    p_lo = ndtr(a)
    p_hi = ndtr(b)
    u = rng.random(size)
    z = np.clip(ndtri(p_lo + u * (p_hi - p_lo)), a, b)      # clip guards u -> 0 or 1
    return mu + sigma * z


# --- loss spread: converting between the nominal width and the realised spread -----------
# The loss L is a normal(mu, sigma_L) cut off to [0, 2*mu].  After the cut the realised
# standard deviation is smaller than sigma_L.  We want to SPECIFY the realised one.

REACH_FRAC = 0.95            # we never ask for more than 95 % of the largest possible spread
SIGMA_EFF_CAP_DB = 0.263     # absolute ceiling on the realised spread [dB] (about 3 % amplitude spread)
EFF_MAX_FRAC = 0.99          # above 99 % of the largest spread the inversion becomes ill-conditioned
_NOMINAL_MAX_RATIO = 50.0    # root search bracket: sigma_L in [0, 50*mu]


def sigma_eff_trunc(mu: float, sigma: float) -> float:
    """Realised std of N(mu, sigma^2) cut off to [0, 2*mu].  (Tends to mu/sqrt(3) for large sigma.)"""
    if sigma <= 0.0 or mu <= 0.0:
        return 0.0
    k = mu / sigma
    phi_k = math.exp(-0.5 * k * k) / math.sqrt(2.0 * math.pi)
    Z = 2.0 * float(ndtr(k)) - 1.0
    return sigma * math.sqrt(max(0.0, 1.0 - 2.0 * k * phi_k / Z))


def sigma_eff_cap(mu: float) -> float:
    """The largest realised spread that is possible for a given mean loss: mu / sqrt(3)."""
    return float(mu) / math.sqrt(3.0)


def sigma_eff_top(mu):
    """Largest realised spread we use for a given mean loss: min(0.95*mu/sqrt(3), 0.263 dB).
    Works for a number or an array.  Every spread f * sigma_eff_top (0 <= f <= 1) is reachable."""
    m = np.asarray(mu, dtype=np.float64)
    if not np.all(np.isfinite(m)) or np.any(m < 0.0):
        raise ValueError(f"mu_L must be finite and non-negative (got {mu!r})")
    top = np.minimum(REACH_FRAC * m / math.sqrt(3.0), SIGMA_EFF_CAP_DB)
    return float(top) if top.ndim == 0 else top


@lru_cache(maxsize=1 << 16)
def _find_nominal_width(mu: float, sigma_eff: float) -> float:
    """Solve sigma_eff_trunc(mu, s) = sigma_eff for s (cached, because it is called a lot)."""
    return float(brentq(lambda s: sigma_eff_trunc(mu, s) - sigma_eff, 0.0, _NOMINAL_MAX_RATIO * mu,
                        xtol=1e-14 * mu, rtol=1e-14, maxiter=500))


def _nominal_width(mu: float, sigma_eff: float) -> float:
    """One (mu, sigma_eff) pair -> nominal width, with sanity checks."""
    if not (math.isfinite(mu) and math.isfinite(sigma_eff)):
        raise ValueError(f"mu_L and sigma_eff must be finite (got mu_L={mu}, sigma_eff={sigma_eff})")
    if mu < 0.0 or sigma_eff < 0.0:
        raise ValueError(f"mu_L and sigma_eff must be non-negative (got mu_L={mu}, sigma_eff={sigma_eff})")
    if sigma_eff == 0.0:
        return 0.0                                   # no spread: every arm loses exactly mu_L
    cap = sigma_eff_cap(mu)
    if sigma_eff >= cap:
        raise ValueError(f"sigma_eff = {sigma_eff:g} dB is unreachable for mu_L = {mu:g} dB "
                         f"(the realised spread is always below mu_L/sqrt(3) = {cap:g} dB)")
    if sigma_eff > EFF_MAX_FRAC * cap:
        raise ValueError(f"sigma_eff = {sigma_eff:g} dB is above {EFF_MAX_FRAC:g} x mu_L/sqrt(3) = {EFF_MAX_FRAC * cap:g} dB "
                         f"for mu_L = {mu:g} dB: numerically ill-conditioned, request a smaller value")
    return _find_nominal_width(mu, sigma_eff)


def sigma_nominal_from_eff(mu, sigma_eff):
    """Nominal Gaussian width sigma_L that gives a realised loss spread sigma_eff (numbers or arrays)."""
    if np.ndim(mu) == 0 and np.ndim(sigma_eff) == 0:
        return _nominal_width(float(mu), float(sigma_eff))
    mu_arr, eff_arr = np.broadcast_arrays(np.asarray(mu, dtype=np.float64), np.asarray(sigma_eff, dtype=np.float64))
    out = np.empty(mu_arr.shape)
    for index in np.ndindex(mu_arr.shape):
        out[index] = _nominal_width(float(mu_arr[index]), float(eff_arr[index]))
    return out


# --- drawing the errors of many chips ----------------------------------------------------

def ideal_chip_errors(n_chips: int, n_mzi_: int) -> ChipErrors:
    """'Chips' without any error: zero phase error, eta = 1/2, no loss."""
    shape = (n_chips, n_mzi_)
    return ChipErrors(np.zeros(shape), np.zeros(shape), np.full(shape, 0.5), np.full(shape, 0.5),
                      np.ones(shape), np.ones(shape))


def sample_chip_errors(n_chips: int, n_mzi_: int, sigma_rad: float, sigma_S: float,
                       mu_L: float, sigma_L: float, rng: np.random.Generator,
                       *, sigma_eff: Optional[float] = None) -> ChipErrors:
    """Independent random errors for n_chips chips.
    Units: sigma_rad [rad], sigma_S [no unit], mu_L and sigma_L [dB].
    If sigma_eff (realised loss spread, dB) is given, sigma_L is computed from it and must be left at 0."""
    if sigma_eff is not None:
        if sigma_L != 0.0:
            raise ValueError("give either sigma_L (nominal width) or sigma_eff (realised spread), not both")
        sigma_L = sigma_nominal_from_eff(mu_L, sigma_eff)
    if min(sigma_rad, sigma_S, mu_L, sigma_L) < 0.0:
        raise ValueError("error parameters must be non-negative")
    shape = (n_chips, n_mzi_)

    # phase errors (the random draws are skipped when sigma_rad = 0)
    if sigma_rad > 0.0:
        theta_err = rng.normal(0.0, sigma_rad, size=shape)
        phi_err = rng.normal(0.0, sigma_rad, size=shape)
    else:
        theta_err = np.zeros(shape)
        phi_err = np.zeros(shape)

    # splitting ratios of the two couplers
    eta1 = truncnorm_sample(0.5, sigma_S, 0.0, 1.0, shape, rng)
    eta2 = truncnorm_sample(0.5, sigma_S, 0.0, 1.0, shape, rng)

    # per-arm loss in dB  ->  amplitude transmission
    if mu_L > 0.0:
        L1 = truncnorm_sample(mu_L, sigma_L, 0.0, 2.0 * mu_L, shape, rng)
        L2 = truncnorm_sample(mu_L, sigma_L, 0.0, 2.0 * mu_L, shape, rng)
        amp1 = 10.0 ** (-L1 / 20.0)
        amp2 = 10.0 ** (-L2 / 20.0)
    else:
        amp1 = np.ones(shape)
        amp2 = np.ones(shape)
    return ChipErrors(theta_err, phi_err, eta1, eta2, amp1, amp2)


# ======================================================================================
# 4. Propagation through the noisy chips  (numba, parallel over chips)
# ======================================================================================

@njit(parallel=True, cache=True)   # numba: implemented by Claude (parallel over chips)
def _propagate_kernel(theta, phi, mm, D, X3, th_err, ph_err, eta1, eta2, amp1, amp2,
                      rescale_every, out, logscale):
    """numba kernel: send the inputs through the noisy MZIs of every chip (parallel over chips), rescaling tiny vectors now and then."""
    n_chips = th_err.shape[0]
    n_m = theta.shape[0]
    K = X3.shape[1]
    N = X3.shape[2]
    for c in prange(n_chips):
        V = X3[c].copy()                      # the K input vectors of this chip
        ls = np.zeros(K)                      # log of the scale factor taken out of each vector
        since = 0
        for idx in range(n_m - 1, -1, -1):    # apply the MZIs, last one first
            # --- the 2x2 matrix of this noisy MZI ---
            th = theta[idx] + 0.5 * th_err[c, idx]
            ph = phi[idx] + ph_err[c, idx]
            e1 = eta1[c, idx]
            e2 = eta2[c, idx]
            a1 = math.sqrt(e1)
            b1 = math.sqrt(1.0 - e1)
            a2 = math.sqrt(e2)
            b2 = math.sqrt(1.0 - e2)
            e2t = complex(math.cos(2.0 * th), math.sin(2.0 * th))
            ep = complex(math.cos(ph), math.sin(ph))
            m00 = ep * (a1 * a2 * e2t + b1 * b2)
            m01 = a2 * b1 * e2t - a1 * b2
            m10 = ep * (a1 * b2 * e2t - a2 * b1)
            m11 = b1 * b2 * e2t + a1 * a2
            la = amp1[c, idx]                 # amplitude transmission of the two output arms
            lb = amp2[c, idx]
            m = mm[idx]
            n = m + 1
            # --- apply it to the two affected modes of every input vector ---
            for k in range(K):
                vm = V[k, m]
                vn = V[k, n]
                V[k, m] = la * (m00 * vm + m01 * vn)
                V[k, n] = lb * (m10 * vm + m11 * vn)
            # --- every so often, rescale tiny vectors so they cannot underflow to zero ---
            since += 1
            if since >= rescale_every:
                since = 0
                for k in range(K):
                    s2 = 0.0
                    for q in range(N):
                        s2 += V[k, q].real * V[k, q].real + V[k, q].imag * V[k, q].imag
                    nrm = math.sqrt(s2)
                    if nrm < 1e-40 and nrm > 0.0:
                        inv = 1.0 / nrm
                        for q in range(N):
                            V[k, q] = V[k, q] * inv
                        ls[k] += math.log(nrm)
        # --- finally the output phase screen D ---
        for k in range(K):
            for q in range(N):
                out[c, k, q] = D[q] * V[k, q]
            logscale[c, k] = ls[k]


def propagate(decomp: Decomposition, X: np.ndarray, errors: ChipErrors):
    """Send inputs through every chip in `errors`.
    X has shape (K, N) (same inputs for every chip) or (n_chips, K, N) (own inputs per chip).
    Returns (Y, logscale) with Y of shape (n_chips, K, N); the true output is Y * exp(logscale)."""
    X = np.asarray(X, dtype=np.complex128)
    n_chips = errors.n_chips
    if X.ndim == 2:
        X3 = np.ascontiguousarray(np.broadcast_to(X, (n_chips,) + X.shape))
    elif X.ndim == 3:
        if X.shape[0] != n_chips:
            raise ValueError("per-chip inputs must have shape (n_chips, K, N)")
        X3 = np.ascontiguousarray(X)
    else:
        raise ValueError("X must have 2 or 3 dimensions")
    K, N = X3.shape[1], X3.shape[2]
    n_m = decomp.theta.shape[0]
    if errors.theta_err.shape[1] != n_m:
        raise ValueError("error arrays do not match the number of MZIs")
    rescale_every = max(N, 64)                 # how often tiny vectors are rescaled (see the kernel)
    out = np.empty((n_chips, K, N), dtype=np.complex128)
    logscale = np.zeros((n_chips, K))
    _propagate_kernel(decomp.theta, decomp.phi, decomp.m, decomp.D, X3,
                      errors.theta_err, errors.phi_err, errors.eta1, errors.eta2,
                      errors.amp1, errors.amp2, rescale_every, out, logscale)
    return out, logscale


# ======================================================================================
# 5. Error metrics
# ======================================================================================

def output_metrics(Y: np.ndarray, logscale: np.ndarray, Y_ideal: np.ndarray) -> dict:
    """Compare noisy outputs Y (n_chips, K, N) with ideal outputs Y_ideal ((K, N) or (n_chips, K, N)).
    Returns arrays of shape (n_chips, K):
        E_raw    ||y_hw_hat - y_ideal_hat||            (the main error; sensitive to global phase)
        E2_raw   the same, squared (computed directly; E_rms is built from it)
        E_inv    same error but with the global phase removed
        log10_T  log10 of (output power / input power)
    where y_hat means "y divided by its length"."""
    Yi = np.asarray(Y_ideal)
    Yi3 = Yi[None, :, :] if Yi.ndim == 2 else Yi

    # normalise the noisy and the ideal outputs to unit length
    nrm = np.linalg.norm(Y, axis=2)
    nrm_safe = np.where(nrm > 0.0, nrm, 1.0)
    Y_hat = Y / nrm_safe[..., None]
    ideal_len = np.linalg.norm(Yi3, axis=2, keepdims=True)
    Ideal_hat = Yi3 / ideal_len

    # the error and its variants
    E_raw = np.linalg.norm(Y_hat - Ideal_hat, axis=2)
    E2_raw = np.sum(np.abs(Y_hat - Ideal_hat) ** 2, axis=2)
    overlap = np.abs(np.sum(Ideal_hat.conj() * Y_hat, axis=2))
    E_inv = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - np.minimum(overlap, 1.0))))
    with np.errstate(divide="ignore"):
        log10_T = 2.0 * (np.log10(nrm) + logscale / math.log(10.0)) - 2.0 * np.log10(ideal_len[..., 0])
    return {"E_raw": E_raw, "E2_raw": E2_raw, "E_inv": E_inv, "log10_T": log10_T}


MEMORY_BUDGET_BYTES = 1.2e8     # chips are processed in chunks so that their error arrays stay below this size


def run_chips(decomp: Decomposition, X: np.ndarray, n_chips: int,
              sigma_rad: float = 0.0, sigma_S: float = 0.0, mu_L: float = 0.0, sigma_L: float = 0.0,
              *, rng: np.random.Generator, Y_ideal: np.ndarray, sigma_eff: Optional[float] = None) -> dict:
    """Draw n_chips noisy copies of ONE unitary, send the inputs X through each, return the error metrics.
    X: (n_chips, K, N), the inputs of every chip;  Y_ideal: the ideal outputs U @ x, same shape.
    sigma_eff: realised loss spread [dB]; if given, sigma_L must be 0."""
    if sigma_eff is not None:
        if sigma_L != 0.0:
            raise ValueError("give either sigma_L (nominal width) or sigma_eff (realised spread), not both")
        sigma_L = sigma_nominal_from_eff(mu_L, sigma_eff)
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 3 or X.shape[0] != n_chips:
        raise ValueError("inputs must have shape (n_chips, K, N)")
    n_m = decomp.theta.shape[0]

    # process the chips in chunks so the error arrays stay within the memory budget
    chunk = max(1, int(MEMORY_BUDGET_BYTES // (48 * max(n_m, 1))))
    parts = []
    done = 0
    while done < n_chips:
        c = min(chunk, n_chips - done)
        errors = sample_chip_errors(c, n_m, sigma_rad, sigma_S, mu_L, sigma_L, rng)
        Y, logscale = propagate(decomp, X[done:done + c], errors)
        parts.append(output_metrics(Y, logscale, Y_ideal[done:done + c]))
        done += c
    return {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}
