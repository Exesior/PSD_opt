"""JIT-compiled batch functions for aggregation propensities.

The weighted-DSMC solver needs, once per accepted Monte-Carlo event, the
per-particle agglomeration propensity

    r_i = sum_j  W_j * beta(i, j)          (j != i)
        + (W_i - 1) * beta(i, i)           (self-collision, only if W_i > 1)

Evaluating that literally costs O(n^2) per event and, for the shear kernel at
n ~ 2000, accounted for ~88 % of total solver runtime.

Two families of implementations live here:

``*_pairwise``
    The literal double loop. Kept as the numerical reference and as the default
    so existing validation runs stay reproducible bit-for-bit. The outer loop is
    parallelised with ``prange``; because ``fastmath`` is off, LLVM may not
    reorder the inner floating-point reduction, so each ``r_out[i]`` is
    accumulated in exactly the original order and the results are bit-identical
    to the serial version.

``*_moment``
    Closed-form O(n) evaluation. Every built-in kernel is separable in
    ``(r_i, r_j)``, so the inner sum collapses onto a handful of weighted
    moments of the particle population:

        S0 = sum_j W_j            S1 = sum_j W_j r_j
        S2 = sum_j W_j r_j^2      S3 = sum_j W_j r_j^3

    These are *algebraically exact* - no approximation, no truncation. They are
    not bit-identical to the pairwise form because floating-point addition is
    not associative and the summation order differs; expect agreement to within
    a few ULP. See ``docs/`` for the measured deviation.

Complexity summary (n = number of computational particles):

    kernel     pairwise     moment
    --------   ----------   --------
    shear      O(n^2)       O(n)
    brownian   O(n^2)       O(n)
    sum        O(n^2)       O(n)
    constant   O(n)         O(n)      (already closed form)
    liquid_bridge  O(n^2)   -         (not separable; JIT batch only)
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Shear kernel (Chin et al. 1998):  beta(i,j) = corr_beta * g * (r_i + r_j)^3
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def rebuild_r_array_shear(corr_beta, g, R, W, r_out):
    """Exact pairwise reference for the shear kernel. O(n^2)."""
    a = len(W)
    for i in prange(a):
        ri = R[i]
        ri_sum = 0.0
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            beta = corr_beta * g * (ri + rj) ** 3
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_array_shear_moment(corr_beta, g, R, W, r_out):
    """Closed-form O(n) evaluation of the shear propensities.

    (r_i + r_j)^3 = r_i^3 + 3 r_i^2 r_j + 3 r_i r_j^2 + r_j^3, hence

        sum_j W_j beta(i,j) = c*g * (r_i^3 S0 + 3 r_i^2 S1 + 3 r_i S2 + S3)

    The self term is then corrected: the full sum above counts ``W_i beta_ii``,
    but the physical propensity uses ``(W_i - 1) beta_ii`` (a particle cannot
    collide with itself), and 0 when ``W_i <= 1``. With
    ``beta_ii = c*g*(2 r_i)^3 = 8 c g r_i^3`` the correction is a single term.
    """
    a = len(W)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    for j in range(a):
        rj = R[j]
        wj = W[j]
        rj2 = rj * rj
        s0 += wj
        s1 += wj * rj
        s2 += wj * rj2
        s3 += wj * rj2 * rj

    cg = corr_beta * g
    for i in range(a):
        ri = R[i]
        ri2 = ri * ri
        ri3 = ri2 * ri
        full = cg * (ri3 * s0 + 3.0 * ri2 * s1 + 3.0 * ri * s2 + s3)
        wi = W[i]
        self_w = (wi - 1.0) if wi > 1.0 else 0.0
        # beta_ii = 8 * cg * ri^3
        r_out[i] = full + (8.0 * cg * ri3) * (self_w - wi)


# ---------------------------------------------------------------------------
# Brownian kernel (Tsouris et al. 1995):
#     beta(i,j) = corr_beta * 2 kT (r_i + r_j)^2 / (3 mu r_i r_j)
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def rebuild_r_array_brownian(corr_beta, kT, viscosity, R, W, r_out):
    """Exact pairwise reference for the Brownian kernel. O(n^2)."""
    a = len(W)
    for i in prange(a):
        ri = R[i]
        ri_sum = 0.0
        if ri <= 0:
            r_out[i] = 0.0
            continue
        for j in range(a):
            rj = R[j]
            if rj <= 0:
                continue
            Wj = W[j]
            numerator = 2.0 * kT * (ri + rj) ** 2
            denominator = 3.0 * viscosity * ri * rj
            if denominator <= 0:
                continue
            beta = corr_beta * numerator / denominator
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_array_brownian_moment(corr_beta, kT, viscosity, R, W, r_out):
    """Closed-form O(n) evaluation of the Brownian propensities.

    (r_i + r_j)^2 / (r_i r_j) = r_i/r_j + 2 + r_j/r_i, so with
    ``K = 2 corr_beta kT / (3 mu)``

        sum_j W_j beta(i,j) = K * (r_i * T_inv + 2 T0 + T1 / r_i)

    where the moments run over the particles with ``r_j > 0`` only, matching the
    ``continue`` guards of the pairwise version:

        T_inv = sum W_j / r_j,  T0 = sum W_j,  T1 = sum W_j r_j
    """
    a = len(W)
    if viscosity <= 0.0:
        # Pairwise version skips every pair (denominator <= 0).
        for i in range(a):
            r_out[i] = 0.0
        return

    t_inv = 0.0
    t0 = 0.0
    t1 = 0.0
    for j in range(a):
        rj = R[j]
        if rj <= 0.0:
            continue
        wj = W[j]
        t_inv += wj / rj
        t0 += wj
        t1 += wj * rj

    K = 2.0 * corr_beta * kT / (3.0 * viscosity)
    for i in range(a):
        ri = R[i]
        if ri <= 0.0:
            r_out[i] = 0.0
            continue
        full = K * (ri * t_inv + 2.0 * t0 + t1 / ri)
        wi = W[i]
        self_w = (wi - 1.0) if wi > 1.0 else 0.0
        # beta_ii = K * (1 + 2 + 1) = 4K
        r_out[i] = full + (4.0 * K) * (self_w - wi)


# ---------------------------------------------------------------------------
# Sum kernel:  beta(i,j) = corr_beta * (v_i + v_j),  v = 4/3 pi r^3
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def rebuild_r_array_sum(corr_beta, R, W, r_out):
    """Exact pairwise reference for the volume-sum kernel. O(n^2)."""
    a = len(W)
    pi_factor = 4.0 / 3.0 * math.pi
    for i in prange(a):
        ri = R[i]
        ri_sum = 0.0
        vi = pi_factor * ri**3
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            vj = pi_factor * rj**3
            beta = corr_beta * (vi + vj)
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_array_sum_moment(corr_beta, R, W, r_out):
    """Closed-form O(n) evaluation of the volume-sum propensities.

        sum_j W_j beta(i,j) = corr_beta * (v_i * S0 + Sv),  Sv = sum_j W_j v_j
        beta_ii = 2 corr_beta v_i
    """
    a = len(W)
    pi_factor = 4.0 / 3.0 * math.pi
    s0 = 0.0
    sv = 0.0
    for j in range(a):
        wj = W[j]
        s0 += wj
        sv += wj * (pi_factor * R[j] ** 3)

    for i in range(a):
        vi = pi_factor * R[i] ** 3
        full = corr_beta * (vi * s0 + sv)
        wi = W[i]
        self_w = (wi - 1.0) if wi > 1.0 else 0.0
        r_out[i] = full + (2.0 * corr_beta * vi) * (self_w - wi)


# ---------------------------------------------------------------------------
# Constant kernel:  beta(i,j) = corr_beta
# Already closed form; pairwise and moment forms coincide.
# ---------------------------------------------------------------------------


@njit(cache=True)
def rebuild_r_array_constant(corr_beta, W, r_out):
    """Constant kernel propensities. O(n).

    ``sum_j W_j beta = corr_beta * sum_W``; the self-collision correction
    replaces ``W_i`` by ``W_i - 1`` when ``W_i > 1`` and by 0 otherwise.
    """
    a = len(W)
    sum_W = 0.0
    for j in range(a):
        sum_W += W[j]

    for i in range(a):
        Wi = W[i]
        if Wi > 1.0:
            r_out[i] = corr_beta * (sum_W - 1.0)
        else:
            r_out[i] = corr_beta * (sum_W - Wi)


rebuild_r_array_constant_moment = rebuild_r_array_constant


# ---------------------------------------------------------------------------
# Liquid-bridge kernel: shear base, modulated by the saturation of *both*
# partners and by their external liquid. Not separable in (i, j), so there is
# no O(n) form - but the previous implementation fell back to an O(n^2) loop of
# *Python* kernel calls, which is roughly three orders of magnitude slower than
# the compiled double loop below.
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def rebuild_r_array_liquid_bridge(
    corr_beta,
    g,
    s_opt,
    sigma_s,
    alpha_liq,
    R,
    W,
    saturation,
    v_liq_ext,
    r_out,
):
    """Exact pairwise evaluation of the liquid-bridge kernel. O(n^2), compiled.

    Mirrors ``LiquidBridgeKernel.compute_beta`` term for term:

        beta = corr_beta * g * (r1 + r2)^3
               * exp(-(s_eff - s_opt)^2 / (2 sigma_s^2))
               * (1 + alpha_liq * log1p(v_liq_total / v_ref))

    with ``s_eff = (s1 + s2)/2``, ``v_liq_total = v_ext1 + v_ext2`` and
    ``v_ref = 0.01 * 4/3 pi (r1^3 + r2^3)``. NaN saturation ("Vollkoerper",
    i.e. no pores) disables both enhancement factors, exactly as in the Python
    implementation.
    """
    a = len(W)
    cg = corr_beta * g
    pi_factor = 4.0 / 3.0 * math.pi
    two_sigma_sq = 2.0 * sigma_s * sigma_s

    for i in prange(a):
        ri = R[i]
        ri_sum = 0.0
        if ri <= 0.0:
            r_out[i] = 0.0
            continue
        si = saturation[i]
        ext_i = v_liq_ext[i]
        si_is_nan = np.isnan(si)

        for j in range(a):
            rj = R[j]
            if rj <= 0.0:
                continue
            beta = cg * (ri + rj) ** 3

            sj = saturation[j]
            if not (si_is_nan or np.isnan(sj)):
                s_eff = 0.5 * (si + sj)
                d = s_eff - s_opt
                bridge = math.exp(-(d * d) / two_sigma_sq)

                v_ref = (pi_factor * (ri**3 + rj**3)) * 0.01
                if v_ref < 1e-30:
                    v_ref = 1e-30
                liquid = 1.0 + alpha_liq * math.log1p((ext_i + v_liq_ext[j]) / v_ref)
                beta = beta * bridge * liquid

            if beta < 0.0:
                beta = 0.0

            Wj = W[j]
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


# ---------------------------------------------------------------------------
# Serial twins of the pairwise kernels.
#
# Numba's parallel dispatch costs tens of microseconds per call on Windows.
# For small populations the O(n^2) loop finishes faster than the threads can be
# handed their work, so below PARALLEL_MIN_N the serial variant wins. Both
# variants accumulate each r_out[i] in the same order and are bit-identical.
# ---------------------------------------------------------------------------

@njit(cache=True)
def rebuild_r_array_shear_serial(corr_beta, g, R, W, r_out):
    """Serial twin of the parallel kernel. Exact pairwise reference for the shear kernel. O(n^2)."""
    a = len(W)
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            beta = corr_beta * g * (ri + rj) ** 3
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_array_brownian_serial(corr_beta, kT, viscosity, R, W, r_out):
    """Serial twin of the parallel kernel. Exact pairwise reference for the Brownian kernel. O(n^2)."""
    a = len(W)
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        if ri <= 0:
            r_out[i] = 0.0
            continue
        for j in range(a):
            rj = R[j]
            if rj <= 0:
                continue
            Wj = W[j]
            numerator = 2.0 * kT * (ri + rj) ** 2
            denominator = 3.0 * viscosity * ri * rj
            if denominator <= 0:
                continue
            beta = corr_beta * numerator / denominator
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_array_sum_serial(corr_beta, R, W, r_out):
    """Serial twin of the parallel kernel. Exact pairwise reference for the volume-sum kernel. O(n^2)."""
    a = len(W)
    pi_factor = 4.0 / 3.0 * math.pi
    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        vi = pi_factor * ri**3
        for j in range(a):
            rj = R[j]
            Wj = W[j]
            vj = pi_factor * rj**3
            beta = corr_beta * (vi + vj)
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


@njit(cache=True)
def rebuild_r_array_liquid_bridge_serial(
    corr_beta,
    g,
    s_opt,
    sigma_s,
    alpha_liq,
    R,
    W,
    saturation,
    v_liq_ext,
    r_out,
):
    """Serial twin of the parallel kernel. Exact pairwise evaluation of the liquid-bridge kernel. O(n^2), compiled.

    Mirrors ``LiquidBridgeKernel.compute_beta`` term for term:

        beta = corr_beta * g * (r1 + r2)^3
               * exp(-(s_eff - s_opt)^2 / (2 sigma_s^2))
               * (1 + alpha_liq * log1p(v_liq_total / v_ref))

    with ``s_eff = (s1 + s2)/2``, ``v_liq_total = v_ext1 + v_ext2`` and
    ``v_ref = 0.01 * 4/3 pi (r1^3 + r2^3)``. NaN saturation ("Vollkoerper",
    i.e. no pores) disables both enhancement factors, exactly as in the Python
    implementation.
    """
    a = len(W)
    cg = corr_beta * g
    pi_factor = 4.0 / 3.0 * math.pi
    two_sigma_sq = 2.0 * sigma_s * sigma_s

    for i in range(a):
        ri = R[i]
        ri_sum = 0.0
        if ri <= 0.0:
            r_out[i] = 0.0
            continue
        si = saturation[i]
        ext_i = v_liq_ext[i]
        si_is_nan = np.isnan(si)

        for j in range(a):
            rj = R[j]
            if rj <= 0.0:
                continue
            beta = cg * (ri + rj) ** 3

            sj = saturation[j]
            if not (si_is_nan or np.isnan(sj)):
                s_eff = 0.5 * (si + sj)
                d = s_eff - s_opt
                bridge = math.exp(-(d * d) / two_sigma_sq)

                v_ref = (pi_factor * (ri**3 + rj**3)) * 0.01
                if v_ref < 1e-30:
                    v_ref = 1e-30
                liquid = 1.0 + alpha_liq * math.log1p((ext_i + v_liq_ext[j]) / v_ref)
                beta = beta * bridge * liquid

            if beta < 0.0:
                beta = 0.0

            Wj = W[j]
            if i == j:
                if Wj > 1.0:
                    ri_sum += (Wj - 1.0) * beta
            else:
                ri_sum += Wj * beta
        r_out[i] = ri_sum


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

#: Below this active-particle count the serial kernels are used: parallel
#: dispatch overhead exceeds the work for small populations. Measured
#: cross-over on an 8-core Windows box sits around 700-900 particles.
PARALLEL_MIN_N = 800

#: Kernel names that have a batch implementation here. Anything else falls back
#: to the generic (slow) per-pair Python path in ``MCPBEAgg``.
BATCH_KERNELS = frozenset(
    {"shear_chin1998", "brownian_tsouris1995", "constant", "sum", "liquid_bridge"}
)

#: Kernel names for which an exact O(n) moment form exists.
MOMENT_KERNELS = frozenset({"shear_chin1998", "brownian_tsouris1995", "constant", "sum"})
