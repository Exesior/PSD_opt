"""JIT-compiled batch functions for pair-delta corrected aggregation propensities.

What is computed
----------------
Once per accepted Monte-Carlo event the solver needs, for every active particle,
the *bias-corrected* partial agglomeration propensity (Ji & Rhein, Eqs. 33/36/37)

    R_i* = W_i * [ sum_{j!=i} W_j beta(i,j) / dW_ij
                 + (W_i - 1) beta(i,i) / dW_ii ]          (only if W_i > 1)

    dW_i  = min(dW_const, W_i)          effective batch size of i   (Eq. 33)
    dW_ij = min(dW_i, dW_j)             effective pair batch size
    dW_ii = min(dW_i, W_i / 2)          a self-collision consumes two particles

The ``1/dW_ij`` factor is what makes the scheme mean-unbiased: the batch actually
executed is capped by the *available* weight of both partners, so the sampling
probability has to be scaled by exactly that cap. Dividing the finished sum by
``dW_i`` alone (the pre-2026-08 form) is biased - see
``docs/Bias_Correction_und_Gewichtsdisziplin.md`` Sec. 2.

Two families of implementations
-------------------------------
``rebuild_r_pairdelta_pairwise{,_serial}``
    Literal O(n^2) double loop, structurally identical to the reference
    implementation in ``pbe_core.func.jit_mcpbe.nb_rebuild_ragg_weighted_pair_delta``.
    This is the numerical ground truth and works for every kernel.

``rebuild_r_pairdelta_moment``
    O(n log n) closed form for kernels that are *separable*, i.e. that can be
    written as ``beta(i,j) = sum_k f_k(i) g_k(j)``.

    The naive moment trick (a single set of weighted moments S0..S3) breaks down
    here, because ``min(dW_i, dW_j)`` couples both indices. The way around it:
    ``dW_i = min(dW_const, W_i)`` can only take two shapes, so the population
    splits into

        H = { j : W_j >= dW_const }   ->  dW_j = dW_const   (a constant!)
        L = { j : 0 < W_j < dW_const} ->  dW_j = W_j

    Three of the four blocks of ``min(dW_i, dW_j)`` are then separable outright
    (H x H gives the constant, H x L depends only on j, L x H only on i). The
    remaining L x L block splits at the sort position of W_i:

        sum_{j in L} W_j beta/min(W_i,W_j)
          = sum_{W_j <= W_i} beta(i,j)  +  (1/W_i) sum_{W_j > W_i} W_j beta(i,j)

    i.e. a prefix sum of ``g_k`` and a suffix sum of ``W_j g_k``, both O(1) to
    look up once L is sorted by weight. The ``1/W_i`` cancels against the outer
    ``W_i`` factor, so the light-particle branch ends up division-free.

    Full derivation, worked numeric example and the verification protocol:
    ``mcpbe/docs/Bias_Correction_und_Gewichtsdisziplin.md`` Sec. 4.

Complexity (n = number of computational particles):

    kernel          pairwise    moment
    -------------   ---------   -----------
    constant        O(n^2)      O(n log n)
    sum             O(n^2)      O(n log n)
    shear           O(n^2)      O(n log n)
    brownian        O(n^2)      O(n log n)
    eke             O(n^2)      -            (not separable)
    etm             O(n^2)      -            (not separable)

Why EKE/ETM have no moment form: their velocity factor is the square root of a
*sum*, ``sqrt(1/r_i^k + 1/r_j^k)``. A moment form needs
``beta(i,j) = sum_m f_m(i) g_m(j)``, and a square root does not distribute over
the sum, so no finite decomposition exists. They still get the compiled
pairwise path, because they are pure functions of ``(p0, r_i, r_j)`` - the same
shape as shear and brownian - and therefore fit into ``_beta_of`` directly.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

_PI43 = 4.0 / 3.0 * math.pi

# Kernel ids for the compiled dispatch (mirrors the COLEVAL switch upstream).
KID_CONSTANT = 0
KID_SUM = 1
KID_SHEAR = 2
KID_BROWNIAN = 3
KID_EKE = 4
KID_ETM = 5

#: Maps kernel name -> (kid, parameter extractor). ``p0`` folds every constant
#: prefactor into a single number so the compiled kernels stay branch-cheap.
KERNEL_IDS = {
    "constant": KID_CONSTANT,
    "sum": KID_SUM,
    "shear_chin1998": KID_SHEAR,
    "brownian_tsouris1995": KID_BROWNIAN,
    "eke_darelius2005": KID_EKE,
    "etm_darelius2005": KID_ETM,
}

#: Kernel names with a compiled batch implementation here.
BATCH_KERNELS = frozenset(
    {"shear_chin1998", "brownian_tsouris1995", "constant", "sum",
     "eke_darelius2005", "etm_darelius2005"}
)

#: Kernel names that are separable and therefore have an O(n log n) moment form.
MOMENT_KERNELS = frozenset({"shear_chin1998", "brownian_tsouris1995", "constant", "sum"})

#: Below this active-particle count the serial pairwise kernel is used: numba's
#: parallel dispatch costs tens of microseconds, which dominates for small n.
PARALLEL_MIN_N = 800


# ---------------------------------------------------------------------------
# Kernel evaluation
# ---------------------------------------------------------------------------


# These helpers are plain njit (no ``inline="always"``): njit-to-njit calls are
# compiled and LLVM inlines bodies this small on its own, so the hint bought
# nothing measurable. Local names are underscore-prefixed as a defensive habit
# against name capture should the inline hint ever be reintroduced.


@njit(cache=True)
def _beta_of(kid, p0, ri, rj):
    """``beta(i,j)`` for the built-in single-prefactor kernels.

    ``p0`` carries the folded prefactor:
        constant  -> corr_beta
        sum       -> corr_beta
        shear     -> corr_beta * g
        brownian  -> 2 * corr_beta * kT / (3 * viscosity)
        eke       -> corr_beta * n_mixer**c_mixer
        etm       -> corr_beta * n_mixer**c_mixer

    The first four are also separable and therefore have a moment form; EKE and
    ETM are not (see the module docstring) and only ever reach this function
    through the pairwise path.

    The EKE/ETM branches must stay bit-for-bit identical to ``_beta_eke_jit`` /
    ``_beta_etm_jit`` in the kernel modules, which is what those kernels'
    ``compute_beta`` calls. ``verify_jit_kernels`` asserts the equality, so keep
    the association order as written.
    """
    if kid == KID_CONSTANT:
        return p0
    elif kid == KID_SUM:
        return p0 * (_PI43 * ri * ri * ri + _PI43 * rj * rj * rj)
    elif kid == KID_SHEAR:
        _rsum = ri + rj
        return p0 * _rsum * _rsum * _rsum
    elif kid == KID_BROWNIAN:
        if ri <= 0.0 or rj <= 0.0:
            return 0.0
        _rsum = ri + rj
        return p0 * _rsum * _rsum / (ri * rj)
    elif kid == KID_EKE:
        if ri <= 0.0 or rj <= 0.0:
            return 0.0
        _rsum = ri + rj
        _inv = 1.0 / (ri * ri * ri) + 1.0 / (rj * rj * rj)
        return p0 * _rsum * _rsum * math.sqrt(_inv)
    elif kid == KID_ETM:
        if ri <= 0.0 or rj <= 0.0:
            return 0.0
        _rsum = ri + rj
        _ci = ri * ri * ri
        _cj = rj * rj * rj
        _inv = 1.0 / (_ci * _ci) + 1.0 / (_cj * _cj)
        return p0 * _rsum * _rsum * math.sqrt(_inv)
    return 0.0


@njit(cache=True)
def _self_delta(delta_i, Wi):
    """``dW_ii = min(dW_i, W_i/2)``.

    ``0.5 * W_i`` is exact in IEEE-754 (it only decrements the exponent), which
    is what lets the consuming side drain a particle to exactly zero.
    """
    _half = 0.5 * Wi
    return delta_i if delta_i < _half else _half


# ---------------------------------------------------------------------------
# Pairwise reference: O(n^2), valid for every separable kernel
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def rebuild_r_pairdelta_pairwise(kid, p0, R, W, delta, dW_const, r_out):
    """Exact O(n^2) pair-delta propensities. Numerical ground truth."""
    a = W.shape[0]
    for i in prange(a):
        di = delta[i]
        Wi = W[i]
        if di <= 0.0 or Wi <= 0.0:
            r_out[i] = 0.0
            continue
        ri = R[i]
        s = 0.0
        for j in range(a):
            if j == i:
                continue
            dj = delta[j]
            Wj = W[j]
            if dj <= 0.0 or Wj <= 0.0:
                continue
            bij = _beta_of(kid, p0, ri, R[j])
            if bij <= 0.0:
                continue
            pair_delta = di if di < dj else dj
            s += Wj * bij / pair_delta
        if Wi > 1.0:
            sd = _self_delta(di, Wi)
            if sd > 0.0:
                bii = _beta_of(kid, p0, ri, ri)
                if bii > 0.0:
                    s += (Wi - 1.0) * bii / sd
        val = Wi * s
        r_out[i] = val if val > 0.0 else 0.0


@njit(cache=True)
def rebuild_r_pairdelta_pairwise_serial(kid, p0, R, W, delta, dW_const, r_out):
    """Serial twin of :func:`rebuild_r_pairdelta_pairwise` (bit-identical)."""
    a = W.shape[0]
    for i in range(a):
        di = delta[i]
        Wi = W[i]
        if di <= 0.0 or Wi <= 0.0:
            r_out[i] = 0.0
            continue
        ri = R[i]
        s = 0.0
        for j in range(a):
            if j == i:
                continue
            dj = delta[j]
            Wj = W[j]
            if dj <= 0.0 or Wj <= 0.0:
                continue
            bij = _beta_of(kid, p0, ri, R[j])
            if bij <= 0.0:
                continue
            pair_delta = di if di < dj else dj
            s += Wj * bij / pair_delta
        if Wi > 1.0:
            sd = _self_delta(di, Wi)
            if sd > 0.0:
                bii = _beta_of(kid, p0, ri, ri)
                if bii > 0.0:
                    s += (Wi - 1.0) * bii / sd
        val = Wi * s
        r_out[i] = val if val > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Moment form: O(n log n) for separable kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _upper_bound(arr, m, x):
    """Index of the first entry of ``arr[:m]`` strictly greater than ``x``.

    Equivalent to ``np.searchsorted(arr[:m], x, side='right')``. Ties therefore
    fall into the "<= W_i" branch - which is exactly right, because for
    ``W_j == W_i`` both branches of the L x L split evaluate to the same value.
    """
    lo = 0
    hi = m
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def rebuild_r_pairdelta_moment(F, G, beta_ii, W, delta, dW_const, r_out):
    """O(n log n) pair-delta propensities for a separable kernel.

    Args:
        F: ``(n, K)`` table of ``f_k(i)``.
        G: ``(n, K)`` table of ``g_k(j)``, such that ``beta(i,j) = sum_k F[i,k] G[j,k]``.
        beta_ii: ``(n,)`` closed-form ``beta(i,i)``.
        W: active weights.
        delta: ``min(dW_const, W_i)``, zeroed for inactive particles.
        dW_const: the prescribed batch size ``dW``.
        r_out: output buffer, length n.
    """
    n = W.shape[0]
    K = F.shape[1]

    B_H = np.zeros(K)          # sum_{j in H} W_j g_k(j)
    A_L = np.zeros(K)          # sum_{j in L}       g_k(j)
    light_idx = np.empty(n, dtype=np.int64)
    m = 0

    # --- partition into heavy / light, accumulate the global moments ---
    for j in range(n):
        if delta[j] <= 0.0 or W[j] <= 0.0:
            continue
        wj = W[j]
        if wj >= dW_const:
            for k in range(K):
                B_H[k] += wj * G[j, k]
        else:
            for k in range(K):
                A_L[k] += G[j, k]
            light_idx[m] = j
            m += 1

    # --- sort the light subset ascending by weight ---
    W_light = np.empty(m)
    for t in range(m):
        W_light[t] = W[light_idx[t]]
    order = np.argsort(W_light)

    W_sort = np.empty(m)
    idx_sort = np.empty(m, dtype=np.int64)
    for t in range(m):
        s = order[t]
        idx_sort[t] = light_idx[s]
        W_sort[t] = W_light[s]

    # --- prefix sums over the sorted light subset ---
    # P[t]    = sum of g_k over the first t entries
    # Qpre[t] = sum of W_j g_k over the first t entries
    P = np.zeros((m + 1, K))
    Qpre = np.zeros((m + 1, K))
    for t in range(m):
        jj = idx_sort[t]
        wj = W_sort[t]
        for k in range(K):
            gk = G[jj, k]
            P[t + 1, k] = P[t, k] + gk
            Qpre[t + 1, k] = Qpre[t, k] + wj * gk

    # --- assemble R_i* ---
    for i in range(n):
        di = delta[i]
        Wi = W[i]
        if di <= 0.0 or Wi <= 0.0:
            r_out[i] = 0.0
            continue
        bii = beta_ii[i]

        if Wi >= dW_const:
            # i in H:  T_i = (1/dW) sum_k f_k B_k(H) + sum_k f_k A_k(L)
            acc = 0.0
            for k in range(K):
                acc += F[i, k] * (B_H[k] / dW_const + A_L[k])
            diag = Wi * Wi * bii / dW_const
            val = Wi * acc - diag
        else:
            # i in L: the 1/W_i terms cancel against the outer W_i factor.
            u = _upper_bound(W_sort, m, Wi)
            acc = 0.0
            for k in range(K):
                B_gt = Qpre[m, k] - Qpre[u, k]
                acc += F[i, k] * (B_H[k] + B_gt + Wi * P[u, k])
            diag = Wi * bii
            val = acc - diag

        # Cancellation guard. The diagonal j == i term is first accumulated into
        # the moments and then subtracted again, so a propensity that is exactly
        # zero (a lone particle, or one whose only partners have zero weight)
        # comes back as a rounding residue of order eps * diag - e.g. 1e-31 where
        # the pairwise path returns a clean 0. Anything that small is noise, not
        # a rate: it is ~15 orders of magnitude below the population total and
        # could never be drawn by the sampler. Cut it at a few ULP of the
        # cancelled magnitude so both paths agree bit-for-bit on the zero case.
        if val < 1e-14 * diag:
            val = 0.0

        if Wi > 1.0:
            sd = _self_delta(di, Wi)
            if sd > 0.0 and bii > 0.0:
                val += Wi * (Wi - 1.0) * bii / sd

        r_out[i] = val if val > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Separable decompositions:  beta(i,j) = sum_k f_k(i) g_k(j)
# ---------------------------------------------------------------------------


def separable_tables(name: str, kernel, R: np.ndarray):
    """Build ``(F, G, beta_ii, p0)`` for a separable built-in kernel.

    The decompositions (see docs Sec. 4.5):

    constant  beta = c
              f = (c,)                            g = (1,)
    sum       beta = c (v_i + v_j),  v = 4/3 pi r^3
              f = (c v_i, c)                      g = (1, v_j)
    shear     beta = C (r_i+r_j)^3,  C = corr_beta * g
              f = (C r^3, 3C r^2, 3C r, C)        g = (1, r, r^2, r^3)
    brownian  beta = KB (r_i+r_j)^2/(r_i r_j),  KB = 2 corr_beta kT / (3 mu)
              f = (KB r, 2KB, KB/r)               g = (1/r, 1, r)
    """
    n = R.shape[0]

    if name == "constant":
        c = float(kernel.corr_beta)
        F = np.full((n, 1), c)
        G = np.ones((n, 1))
        beta_ii = np.full(n, c)
        return F, G, beta_ii, c

    if name == "sum":
        c = float(kernel.corr_beta)
        v = _PI43 * R**3
        F = np.empty((n, 2))
        F[:, 0] = c * v
        F[:, 1] = c
        G = np.empty((n, 2))
        G[:, 0] = 1.0
        G[:, 1] = v
        return F, G, 2.0 * c * v, c

    if name == "shear_chin1998":
        C = float(kernel.corr_beta) * float(kernel.g)
        F = np.empty((n, 4))
        F[:, 0] = C * R**3
        F[:, 1] = 3.0 * C * R**2
        F[:, 2] = 3.0 * C * R
        F[:, 3] = C
        G = np.empty((n, 4))
        G[:, 0] = 1.0
        G[:, 1] = R
        G[:, 2] = R**2
        G[:, 3] = R**3
        return F, G, 8.0 * C * R**3, C

    if name == "brownian_tsouris1995":
        KB = 2.0 * float(kernel.corr_beta) * float(kernel.kT) / (3.0 * float(kernel.viscosity))
        F = np.empty((n, 3))
        F[:, 0] = KB * R
        F[:, 1] = 2.0 * KB
        F[:, 2] = KB / R
        G = np.empty((n, 3))
        G[:, 0] = 1.0 / R
        G[:, 1] = 1.0
        G[:, 2] = R
        return F, G, np.full(n, 4.0 * KB), KB

    raise ValueError(f"{name!r} has no separable decomposition")


def kernel_p0(name: str, kernel) -> float:
    """Folded scalar prefactor consumed by :func:`_beta_of`."""
    if name in ("constant", "sum"):
        return float(kernel.corr_beta)
    if name == "shear_chin1998":
        return float(kernel.corr_beta) * float(kernel.g)
    if name == "brownian_tsouris1995":
        return 2.0 * float(kernel.corr_beta) * float(kernel.kT) / (3.0 * float(kernel.viscosity))
    if name in ("eke_darelius2005", "etm_darelius2005"):
        # Read the value the kernel already computed rather than recomputing
        # corr_beta * n_mixer**c_mixer here. Recomputing would be correct to
        # within a rounding of the pow(), and a rounding difference in the
        # prefactor is exactly what breaks the bit-for-bit parity between
        # compute_beta and the compiled path.
        return float(kernel.p0)
    raise ValueError(f"{name!r} is not a built-in compiled kernel")


@njit(cache=True)
def pick_partner_pairdelta(kid, p0, i, R, W, delta, dW_const, partner_total, u_sel):
    """Draw partner ``j`` proportional to ``W_j beta(i,j) / dW_ij``.

    ``partner_total`` is ``R_i* / W_i`` - the same quantity the propensity
    rebuild accumulated, so the conditional distribution is consistent with the
    primary selection by construction.

    Returns ``(j, w_ij)``; ``j == -1`` signals a failed draw.
    """
    a = W.shape[0]
    if a <= 0 or i < 0 or i >= a:
        return -1, 0.0
    if partner_total <= 0.0:
        return -1, 0.0
    di = delta[i]
    if di <= 0.0:
        return -1, 0.0

    ri = R[i]
    thresh = u_sel * partner_total
    acc = 0.0
    last_j = -1
    last_w = 0.0

    for j in range(a):
        if j == i:
            Wi = W[i]
            if Wi <= 1.0:
                continue
            sd = _self_delta(di, Wi)
            if sd <= 0.0:
                continue
            bii = _beta_of(kid, p0, ri, ri)
            if bii <= 0.0:
                continue
            wij = (Wi - 1.0) * bii / sd
        else:
            Wj = W[j]
            dj = delta[j]
            if Wj <= 0.0 or dj <= 0.0:
                continue
            bij = _beta_of(kid, p0, ri, R[j])
            if bij <= 0.0:
                continue
            pair_delta = di if di < dj else dj
            wij = Wj * bij / pair_delta

        if wij <= 0.0:
            continue
        acc += wij
        last_j = j
        last_w = wij
        if acc > thresh:
            return j, wij

    # Numerical drift between partner_total and the freshly accumulated sum can
    # leave the threshold just out of reach; fall back to the last valid draw.
    if last_j < 0 or last_w <= 0.0:
        return -1, 0.0
    return last_j, last_w
