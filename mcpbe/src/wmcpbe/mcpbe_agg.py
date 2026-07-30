"""Agglomeration mixin for the weighted MC-PBE solver.

Responsibilities
----------------
* ``_rebuild_all_propensities`` - recompute ``r_i = sum_j W_j beta(i,j)`` for
  every active particle. Runs once per accepted event and used to dominate the
  solver runtime; see :mod:`wmcpbe.kernels.aggregation.jit_kernels`.
* ``_do_one_agg``               - execute a single agglomeration event:
  pick a pair, decide the packet size ``dW``, merge volumes / porosity /
  liquid, consume the parents' weight, refresh the samplers.

Volume semantics (see also ``.agents/informations.txt``)
-------------------------------------------------------
``V_flat[:dim]`` holds the **solid** volume per component, ``V_flat[-1]`` the
**dry** volume (solid + pores). ``V_solid = V_dry * (1 - porosity)``, and
``porosity`` is NaN for non-porous primary particles ("Vollkoerper"), for which
``V_solid == V_dry``. Solid volume is the conserved quantity:
``sum_i W_i * V_solid_i`` must not change during an agglomeration event.

Intensive vs. extensive
-----------------------
``liquid_volume``, ``porosity``, ``saturation`` and ``V_flat`` are stored **per
physical particle** (intensive). ``W`` counts how many physical particles a
computational particle represents (extensive). Reducing a parent's ``W`` must
therefore leave its intensive properties untouched.
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import njit

from .fenwick_new import rebuild_sampler
from .kernels.aggregation.jit_kernels import (
    MOMENT_KERNELS,
    PARALLEL_MIN_N,
    rebuild_r_array_brownian,
    rebuild_r_array_brownian_moment,
    rebuild_r_array_brownian_serial,
    rebuild_r_array_constant,
    rebuild_r_array_liquid_bridge,
    rebuild_r_array_liquid_bridge_serial,
    rebuild_r_array_shear,
    rebuild_r_array_shear_moment,
    rebuild_r_array_shear_serial,
    rebuild_r_array_sum,
    rebuild_r_array_sum_moment,
    rebuild_r_array_sum_serial,
)

# (parallel, serial) pairwise implementations per kernel name. Both members of
# a pair are bit-identical; only the dispatch threshold differs.
_PAIRWISE_IMPLS = {
    "shear_chin1998": (rebuild_r_array_shear, rebuild_r_array_shear_serial),
    "brownian_tsouris1995": (rebuild_r_array_brownian, rebuild_r_array_brownian_serial),
    "sum": (rebuild_r_array_sum, rebuild_r_array_sum_serial),
    "liquid_bridge": (rebuild_r_array_liquid_bridge, rebuild_r_array_liquid_bridge_serial),
}

#: Propensity evaluation modes for :attr:`MCPBEAgg.agg_propensity_mode`.
PROPENSITY_MODES = ("pairwise", "moment")


# =============================================================================
# JIT-compiled partner selection for the constant kernel
# =============================================================================


@njit(cache=True)
def pick_partner_constant_jit(
    i,
    W,
    R,
    V0,
    V1,
    dim,
    alpha1d,
    alpha4,
    SIZEEVAL,
    X_SEL,
    Y_SEL,
    Vmean2,
    u_sel,
    corr_beta,
):
    """Partner selection specialised for the constant kernel.

    Returns ``(j, pick_w)`` where ``j`` is the partner index (-1 if rejected)
    and ``pick_w`` is the pair propensity ``W[j] * beta_eff`` (or
    ``(W[i]-1) * beta_eff`` for a self-collision).
    """
    a = len(W)
    if a < 2 or i < 0 or i >= a:
        return -1, 0.0

    # Step 1: partner j drawn with probability proportional to W[j].
    total_W = 0.0
    for j in range(a):
        total_W += W[j]

    if total_W <= 0.0:
        return -1, 0.0

    target = u_sel * total_W
    cumsum = 0.0
    j = 0
    for j_idx in range(a):
        cumsum += W[j_idx]
        if cumsum > target:
            j = j_idx
            break

    if j >= a:
        j = a - 1

    # Step 2: beta(i,j) for the constant kernel.
    beta_ij = corr_beta
    if beta_ij <= 0.0:
        return -1, 0.0

    # Step 3: collision efficiency alpha(i,j).
    if dim == 1:
        alpha_ij = alpha1d
    else:
        Vi0 = V0[i]
        Vi1 = V1[i]
        Vti = Vi0 + Vi1
        Vj0 = V0[j]
        Vj1 = V1[j]
        Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            return -1, 0.0
        p0 = (Vi0 / Vti) * (Vj0 / Vtj)
        p1 = (Vi0 / Vti) * (Vj1 / Vtj)
        p2 = (Vi1 / Vti) * (Vj0 / Vtj)
        p3 = (Vi1 / Vti) * (Vj1 / Vtj)
        alpha_ij = p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3]

    if alpha_ij <= 0.0:
        return -1, 0.0

    beta_eff = beta_ij * alpha_ij

    # Step 4: pair propensity.
    if i == j:
        # Self-agglomeration offers (W[i] - 1) distinct partners.
        pick_w = max(0.0, W[i] - 1.0) * beta_eff
    else:
        pick_w = W[j] * beta_eff

    if pick_w <= 0.0:
        return -1, 0.0

    # Step 5: size-dependent acceptance (SIZEEVAL).
    if SIZEEVAL != 0:
        if dim == 1:
            Vi = V0[i]
            Vj = V0[j]
        else:
            Vi = V0[i] + V1[i]
            Vj = V0[j] + V1[j]

        V_target = np.sqrt(Vmean2) if Vmean2 > 0 else 1e-18
        sigma_V = V_target * Y_SEL

        diff_i = (Vi - V_target) / sigma_V if sigma_V > 0 else 0.0
        diff_j = (Vj - V_target) / sigma_V if sigma_V > 0 else 0.0
        size_factor = np.exp(-0.5 * diff_i * diff_i) * np.exp(-0.5 * diff_j * diff_j)

        if u_sel > size_factor:
            return -1, 0.0

    return j, pick_w


class MCPBEAgg:
    """Agglomeration physics and event execution."""

    #: How ``r_i = sum_j W_j beta(i,j)`` is evaluated.
    #:
    #: ``"pairwise"`` (default)
    #:     Literal O(n^2) double loop, parallelised across ``i``. Bit-identical
    #:     to the pre-refactor implementation, so recorded validation runs stay
    #:     reproducible.
    #: ``"moment"``
    #:     Algebraically exact O(n) closed form for the separable built-in
    #:     kernels (shear / brownian / sum / constant). 30-240x faster at
    #:     n = 500-2000 and the gap widens with n, at the cost of a different
    #:     floating-point summation order (agreement ~1e-15 relative). Kernels
    #:     without a closed form silently keep the pairwise path.
    agg_propensity_mode: str = "pairwise"

    # ------------------------------------------------------------------
    # Scratch buffers
    # ------------------------------------------------------------------
    def _agg_buffers(self, a: int) -> dict:
        """Return per-solver scratch arrays of length >= ``a``.

        Re-used across events so the hot path performs no allocation. Grown
        geometrically; contents are always fully overwritten before use.
        """
        buffers = getattr(self, "_agg_buf", None)
        if buffers is None or buffers["size"] < a:
            size = max(int(a * 1.25) + 8, 16, int(getattr(self, "_cap", 0)))
            buffers = {
                "size": size,
                "R": np.zeros(size, dtype=np.float64),
                "r": np.zeros(size, dtype=np.float64),
                "delta": np.zeros(size, dtype=np.float64),
                "zeros": np.zeros(size, dtype=np.float64),
                "cumsum": np.zeros(size, dtype=np.float64),
                "alpha4": np.zeros(4, dtype=np.float64),
            }
            self._agg_buf = buffers
        return buffers

    def _agg_radii(self, a: int) -> np.ndarray:
        """Active-particle radii ``X/2`` written into the scratch buffer."""
        buf = self._agg_buffers(a)
        R = buf["R"][:a]
        np.multiply(self.X[:a], 0.5, out=R)
        return R

    def _mean_initial_volume_sq(self, a: int) -> float:
        """``mean(V0_total)^2``, cached: ``V0`` is frozen after initialisation."""
        cached = getattr(self, "_vmean2_cache", None)
        if cached is not None:
            return cached
        if hasattr(self, "V0"):
            value = float(np.mean(self.V0[-1, :]) ** 2)
        else:
            # No initial snapshot (e.g. hand-built solver): fall back to the
            # current population. Not cached, since it keeps changing.
            return float(np.mean(self.V_flat[-1, :a]) ** 2)
        self._vmean2_cache = value
        return value

    # ------------------------------------------------------------------
    # Collision efficiency
    # ------------------------------------------------------------------
    def _alpha_ccm(self, idx1: int, idx2: int) -> float:
        """2D collision efficiency from component fractions and ``alpha_prim``."""
        if self.dim == 1:
            if np.ndim(self.alpha_prim) == 0:
                return float(self.alpha_prim)
            return float(np.mean(self.alpha_prim))

        V = self.V_flat
        Vi0 = V[0, idx1]
        Vi1 = V[1, idx1]
        Vti = Vi0 + Vi1
        Vj0 = V[0, idx2]
        Vj1 = V[1, idx2]
        Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            return 0.0
        p0 = (Vi0 / Vti) * (Vj0 / Vtj)
        p1 = (Vi0 / Vti) * (Vj1 / Vtj)
        p2 = (Vi1 / Vti) * (Vj0 / Vtj)
        p3 = (Vi1 / Vti) * (Vj1 / Vtj)
        ap = np.asarray(self.alpha_prim, dtype=float)
        if ap.size != 4:
            ap = np.ones(4, dtype=float)
        return float(p0 * ap[0] + p1 * ap[1] + p2 * ap[2] + p3 * ap[3])

    def _alpha_params(self, a: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(alpha1d, alpha4, V0, V1)`` views for partner selection."""
        buf = self._agg_buffers(a)
        alpha4 = buf["alpha4"]
        V0 = self.V_flat[0, :a]

        if self.dim == 1:
            if np.ndim(self.alpha_prim) == 0:
                alpha1d = float(self.alpha_prim)
            else:
                alpha1d = float(np.mean(self.alpha_prim))
            alpha4[:] = 0.0
            V1 = buf["zeros"][:a]
        else:
            alpha1d = 1.0
            ap = np.asarray(self.alpha_prim, dtype=np.float64)
            alpha4[:] = ap if ap.size == 4 else 1.0
            V1 = self.V_flat[1, :a]

        return alpha1d, alpha4, V0, V1

    def _beta(self, i: int, j: int) -> float:
        """Pair kernel ``beta(i, j)`` via the kernel framework."""
        if self.kernel_manager is None or self.kernel_manager.agg_kernel is None:
            raise RuntimeError("Aggregation kernel not initialized")

        return self.kernel_manager.compute_beta(
            float(self.X[i] * 0.5),
            float(self.X[j] * 0.5),
            particle1_idx=i,
            particle2_idx=j,
            solver=self,
        )

    # ------------------------------------------------------------------
    # Partner selection
    # ------------------------------------------------------------------
    def _pick_partner_kernel(
        self,
        i: int,
        R: np.ndarray,
        W: np.ndarray,
        V0: np.ndarray,
        V1: np.ndarray,
        dim: int,
        alpha1d: float,
        alpha4: np.ndarray,
        SIZEEVAL: int,
        X_SEL: float,
        Y_SEL: float,
        Vmean2: float,
        u_sel: float,
    ) -> tuple[int, float]:
        """Weight-proportional partner sampling using the configured kernel.

        Mirrors :func:`pick_partner_constant_jit` but obtains ``beta`` from
        ``kernel_manager`` so that partner selection stays consistent with
        :meth:`_rebuild_all_propensities`.

        Note on ``u_sel``: the same draw is used both to pick the partner and,
        when ``SIZEEVAL != 0``, as the acceptance variate. That correlation is
        inherited from the original implementation and is preserved here on
        purpose - changing it would alter every recorded trajectory. It is
        documented as a known statistical defect in ``docs/``.

        Returns:
            ``(j, pick_w)``; ``j == -1`` means the attempt was rejected.
        """
        a = len(W)
        if a < 2 or i < 0 or i >= a:
            return -1, 0.0

        # Step 1: partner j with probability proportional to W[j].
        total_W = np.sum(W)
        if total_W <= 0.0:
            return -1, 0.0

        cumsum_W = self._agg_buffers(a)["cumsum"][:a]
        np.cumsum(W, out=cumsum_W)
        j = int(np.searchsorted(cumsum_W, u_sel * total_W, side="right"))
        j = min(j, a - 1)

        # Step 2: beta(i,j) from the kernel framework.
        try:
            beta_ij = self.kernel_manager.compute_beta(
                float(R[i]), float(R[j]), particle1_idx=i, particle2_idx=j, solver=self
            )
        except Exception as exc:  # pragma: no cover - defensive, mirrors legacy
            self._warn_once(
                "agg_beta_failed",
                f"Aggregation kernel raised {type(exc).__name__}: {exc}. "
                "Treating the collision as rejected.",
            )
            beta_ij = 0.0

        if not np.isfinite(beta_ij) or beta_ij <= 0.0:
            return -1, 0.0

        # Step 3: collision efficiency alpha(i,j).
        if dim == 1:
            alpha_ij = alpha1d
        else:
            Vi0 = V0[i]
            Vi1 = V1[i]
            Vti = Vi0 + Vi1
            Vj0 = V0[j]
            Vj1 = V1[j]
            Vtj = Vj0 + Vj1
            if Vti <= 0.0 or Vtj <= 0.0:
                return -1, 0.0
            p0 = (Vi0 / Vti) * (Vj0 / Vtj)
            p1 = (Vi0 / Vti) * (Vj1 / Vtj)
            p2 = (Vi1 / Vti) * (Vj0 / Vtj)
            p3 = (Vi1 / Vti) * (Vj1 / Vtj)
            alpha_ij = float(
                p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3]
            )

        if alpha_ij <= 0.0:
            return -1, 0.0

        beta_eff = beta_ij * alpha_ij

        # Step 4: pair propensity.
        if i == j:
            pick_w = max(0.0, float(W[i] - 1)) * beta_eff
        else:
            pick_w = float(W[j]) * beta_eff

        if pick_w <= 0.0:
            return -1, 0.0

        # Step 5: size-dependent acceptance (SIZEEVAL).
        if SIZEEVAL != 0:
            if dim == 1:
                Vi = V0[i]
                Vj = V0[j]
            else:
                Vi = V0[i] + V1[i]
                Vj = V0[j] + V1[j]

            V_target = np.sqrt(Vmean2) if Vmean2 > 0 else 1e-18
            sigma_V = V_target * Y_SEL

            size_factor = np.exp(-0.5 * ((Vi - V_target) / sigma_V) ** 2) * np.exp(
                -0.5 * ((Vj - V_target) / sigma_V) ** 2
            )

            if u_sel > size_factor:
                return -1, 0.0

        return j, pick_w

    def _warn_once(self, key: str, message: str) -> None:
        """Emit ``message`` as a RuntimeWarning at most once per solver."""
        seen = getattr(self, "_warned_keys", None)
        if seen is None:
            seen = set()
            self._warned_keys = seen
        if key in seen:
            return
        seen.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # Propensity rebuild
    # ------------------------------------------------------------------
    def _rebuild_all_propensities(self) -> None:
        """Recompute ``self._r_agg[:a_tot]``, the per-particle event propensity.

        The raw kernel sum ``sum_j W_j beta(i,j)`` is divided by the packet size
        ``delta_i = min(agg_dW_max, W_i)`` so that the sampler draws packets, not
        single physical collisions.
        """
        a = self.a_tot
        cap = int(getattr(self, "_cap", max(8, a)))

        self._r_agg = _ensure_len(getattr(self, "_r_agg", None), cap)
        self._delta_agg = _ensure_len(getattr(self, "_delta_agg", None), cap)

        if a <= 0:
            self._r_agg[:] = 0.0
            self._delta_agg[:] = 0.0
            return

        buf = self._agg_buffers(a)
        R = self._agg_radii(a)
        W = self.W[:a]

        # delta_i = min(dW_const, W_i), zeroed where W is non-finite or <= 0.
        dW_const = float(getattr(self, "_agg_dW_const", None) or self._prepare_agg_delta_config())
        delta = buf["delta"][:a]
        np.minimum(W, dW_const, out=delta)
        np.copyto(delta, 0.0, where=~(np.isfinite(delta) & (delta > 0.0)))
        self._delta_agg[:a] = delta

        r = buf["r"][:a]
        self._compute_raw_propensities(R, W, r)

        # r_i / delta_i, guarding against delta_i == 0.
        np.divide(r, delta, out=r, where=delta > 0.0)
        np.copyto(r, 0.0, where=~(delta > 0.0))
        np.maximum(r, 0.0, out=r)

        self._r_agg[:a] = r
        self._r_agg[a:] = 0.0
        self._delta_agg[a:] = 0.0

    def _compute_raw_propensities(self, R: np.ndarray, W: np.ndarray, out: np.ndarray) -> None:
        """Fill ``out[i] = sum_j W_j beta(i,j)`` (with the self-pair correction)."""
        manager = getattr(self, "kernel_manager", None)
        kernel = getattr(manager, "agg_kernel", None) if manager is not None else None
        if kernel is None:
            raise RuntimeError("Aggregation kernel not initialized")

        name = getattr(kernel, "name", "")
        mode = str(getattr(self, "agg_propensity_mode", "pairwise")).lower()
        if mode not in PROPENSITY_MODES:
            raise ValueError(
                f"agg_propensity_mode must be one of {PROPENSITY_MODES}, got {mode!r}"
            )
        use_moment = mode == "moment" and name in MOMENT_KERNELS

        if name == "shear_chin1998":
            fn = rebuild_r_array_shear_moment if use_moment else _pairwise(name, out.shape[0])
            fn(float(kernel.corr_beta), float(kernel.g), R, W, out)
        elif name == "brownian_tsouris1995":
            fn = rebuild_r_array_brownian_moment if use_moment else _pairwise(name, out.shape[0])
            fn(float(kernel.corr_beta), float(kernel.kT), float(kernel.viscosity), R, W, out)
        elif name == "constant":
            # Already a closed form: pairwise and moment coincide.
            rebuild_r_array_constant(float(kernel.corr_beta), W, out)
        elif name == "sum":
            fn = rebuild_r_array_sum_moment if use_moment else _pairwise(name, out.shape[0])
            fn(float(kernel.corr_beta), R, W, out)
        elif name == "liquid_bridge":
            self._raw_propensities_liquid_bridge(kernel, R, W, out)
        else:
            self._raw_propensities_generic(R, W, out)

    def _raw_propensities_liquid_bridge(self, kernel, R, W, out) -> None:
        """Compiled O(n^2) batch path for the liquid-bridge kernel.

        The kernel is not separable in ``(i, j)``, so the quadratic loop stays -
        but it runs compiled instead of as ``n^2`` Python calls.
        """
        a = out.shape[0]
        saturation = self.saturation[:a]
        v_liq_ext = self.get_V_liquid_external()
        if v_liq_ext.shape[0] != a:  # pragma: no cover - defensive
            v_liq_ext = np.ascontiguousarray(v_liq_ext[:a])

        _pairwise("liquid_bridge", a)(
            float(kernel.corr_beta),
            float(kernel.g),
            float(kernel.s_opt),
            float(kernel.sigma_s),
            float(kernel.alpha_liq),
            R,
            W,
            np.ascontiguousarray(saturation),
            np.ascontiguousarray(v_liq_ext, dtype=np.float64),
            out,
        )

    def _raw_propensities_generic(self, R, W, out) -> None:
        """Fallback for kernels without a batch implementation.

        Costs ``n^2`` Python-level kernel calls per event; emits a one-time
        warning so the cost is visible rather than mysterious.
        """
        a = out.shape[0]
        self._warn_once(
            "agg_generic_fallback",
            f"Aggregation kernel {getattr(self.kernel_manager.agg_kernel, 'name', '?')!r} "
            "has no batch implementation; falling back to an O(n^2) Python loop "
            "in _rebuild_all_propensities(). Add a JIT batch function in "
            "wmcpbe/kernels/aggregation/jit_kernels.py to make this usable at scale.",
        )
        compute_beta = self.kernel_manager.compute_beta
        for i_idx in range(a):
            ri = 0.0
            r1 = float(R[i_idx])
            for j_idx in range(a):
                beta_ij = compute_beta(
                    r1,
                    float(R[j_idx]),
                    particle1_idx=i_idx,
                    particle2_idx=j_idx,
                    solver=self,
                )
                if i_idx == j_idx:
                    ri += max(0.0, float(W[j_idx] - 1)) * beta_ij
                else:
                    ri += float(W[j_idx]) * beta_ij
            out[i_idx] = ri

    # ------------------------------------------------------------------
    # Packet size
    # ------------------------------------------------------------------
    def _compute_agg_dW(self, i: int, j: int, pair_prop: float, sum_prop_before: float) -> float:
        """Packet size ``dW`` (number of physical collisions) for pair ``(i, j)``."""
        Wi = float(self.W[i])
        Wj = float(self.W[j])
        if Wi <= 0.0 or Wj <= 0.0:
            return 0.0

        dW_max = float(getattr(self, "agg_dW_max", 1.0))
        dW_min = max(0.0, float(getattr(self, "agg_dW_min", 1.0)))
        if dW_max <= 0.0:
            return 0.0

        mode = str(getattr(self, "agg_dW_mode", "const")).lower()
        if mode == "const":
            dW = dW_max
        else:
            f = 0.0 if sum_prop_before <= 0.0 else float(pair_prop) / float(sum_prop_before)
            f = float(np.clip(f, 0.0, 1.0))
            alpha = float(getattr(self, "agg_dW_alpha", 100.0))
            if alpha <= 0.0:
                alpha = 1.0
            shape = f**0.5 if mode == "sqrt" else f
            dW = dW_min + alpha * shape * (dW_max - dW_min)

        dW = min(max(dW, dW_min), dW_max)

        dW_const = float(getattr(self, "_agg_dW_const", dW_max))
        delta_i = self._update_delta_single(i, attr_name="_delta_agg", dW_const=dW_const)
        delta_j = self._update_delta_single(j, attr_name="_delta_agg", dW_const=dW_const)

        if i == j:
            # A self-collision consumes 2*dW from the same packet.
            if delta_i <= 0.0 or Wi <= 2.0 * delta_i:
                return 0.0
            dW = min(dW, delta_i)
        else:
            dW = min(dW, Wi, Wj)
            if delta_i > 0.0:
                dW = min(dW, delta_i)
            if delta_j > 0.0:
                dW = min(dW, delta_j)

        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)

    # ------------------------------------------------------------------
    # Single agglomeration event
    # ------------------------------------------------------------------
    def _do_one_agg(self) -> None:
        """Attempt one agglomeration event.

        Consumes exactly two random numbers per attempt (one for the first
        partner via the Fenwick sampler, one for partner selection /
        acceptance), plus whatever the optional acceptance kernel draws.
        """
        a = self.a_tot
        self._last_agg_dW = 0.0
        if a < 2:
            return

        pair = self._select_pair(a)
        if pair is None:
            return
        i, j, pick_w = pair

        # Packet size for this accepted event.
        if self._agg_sampler is not None:
            sum_prop_before = float(self._agg_sampler.total())
        else:
            sum_prop_before = float(np.sum(self._r_agg[:a]))

        # Distinct pair: W_i * (W_j * beta_ij); self pair: W_i * ((W_i-1) * beta_ii).
        pair_prop = float(self.W[i]) * pick_w
        dW = self._compute_agg_dW(i, j, pair_prop, sum_prop_before)
        if dW <= 0.0:
            return
        self._last_agg_dW = dW

        new_idx = self._merge_pair(i, j, dW)
        self._consume_parent_weight(i, j, dW)
        self._refresh_samplers_after_agg()
        del new_idx  # merged particle index is not needed by the caller

    def _select_pair(self, a: int) -> tuple[int, int, float] | None:
        """Draw a collision pair. Returns ``(i, j, pick_w)`` or ``None``."""
        # 1) first partner proportional to r_i
        i = self._agg_sampler.sample(self._rng)

        R = self._agg_radii(a)
        W = self.W[:a]
        alpha1d, alpha4, V0, V1 = self._alpha_params(a)

        SIZEEVAL = int(getattr(self, "SIZEEVAL", 1))
        X_SEL = float(getattr(self, "X_SEL", 0.31))
        Y_SEL = float(getattr(self, "Y_SEL", 1.06))
        Vmean2 = self._mean_initial_volume_sq(a)

        # 2) second partner proportional to W_j, plus SIZEEVAL acceptance
        u_sel = float(self._rng.random())

        kernel = self.kernel_manager.agg_kernel
        if getattr(kernel, "name", "") == "constant":
            j, pick_w = pick_partner_constant_jit(
                i, W, R, V0, V1, int(self.dim),
                float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2,
                u_sel, float(getattr(kernel, "corr_beta", 1.0)),
            )
        else:
            j, pick_w = self._pick_partner_kernel(
                i, R, W, V0, V1, int(self.dim),
                float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2,
                u_sel=u_sel,
            )

        if j < 0 or pick_w <= 0.0:
            return None

        # 3) optional physical acceptance criterion (e.g. Stokes, Braumann 2007)
        acceptance = self.kernel_manager.agglomeration_acceptance_kernel
        if acceptance is not None:
            accepted = acceptance.accept_collision(
                r1=float(R[i]),
                r2=float(R[j]),
                v_dry1=float(self.V_flat[-1, i]),
                v_dry2=float(self.V_flat[-1, j]),
                particle1_idx=i,
                particle2_idx=j,
                solver=self,
            )
            if not accepted:
                return None

        return i, j, pick_w

    def _merge_pair(self, i: int, j: int, dW: float) -> int:
        """Create the merged child particle and return its index.

        Mass conservation: ``V_solid_child = V_solid_i + V_solid_j``. The dry
        volume follows from the porosity growth kernel, and the liquid is
        redistributed between the internal (pore) and external (surface)
        reservoirs.
        """
        dim = self.dim
        Vi_comp = self.V_flat[:dim, i].copy()
        Vj_comp = self.V_flat[:dim, j].copy()
        Vi_dry = float(self.V_flat[-1, i])
        Vj_dry = float(self.V_flat[-1, j])

        poro_i = self.porosity[i]
        poro_j = self.porosity[j]

        # V_solid = V_dry for Vollkoerper (NaN porosity), else V_dry * (1 - poro).
        V_solid_i = Vi_dry if np.isnan(poro_i) else Vi_dry * (1.0 - poro_i)
        V_solid_j = Vj_dry if np.isnan(poro_j) else Vj_dry * (1.0 - poro_j)
        V_solid_merged = V_solid_i + V_solid_j

        self._append_particle_column(Vi_comp + Vj_comp)
        new_idx = self.a_tot - 1
        self.W[new_idx] = dW

        V_dry_merged, poro_merged = self._merged_porosity(i, j, Vi_dry, Vj_dry, poro_i, poro_j)
        self.porosity[new_idx] = poro_merged

        # V_flat[:dim] must hold V_solid, V_flat[-1] must hold V_dry.
        if dim == 1:
            self.V_flat[0, new_idx] = V_solid_merged
        else:
            # Distribute the merged solid volume over the components in the
            # parents' proportion. Writing the scalar total into every
            # component (the pre-refactor behaviour) multiplied the solid
            # volume by `dim` and destroyed the composition.
            comp = Vi_comp + Vj_comp
            comp_total = float(np.sum(comp))
            if comp_total > 0.0:
                self.V_flat[:dim, new_idx] = comp * (V_solid_merged / comp_total)
            else:
                self.V_flat[:dim, new_idx] = V_solid_merged / dim
        self.V_flat[-1, new_idx] = V_dry_merged

        self._merge_liquid(i, j, new_idx, Vi_dry, Vj_dry, poro_i, poro_j, V_dry_merged)
        return new_idx

    def _merged_porosity(self, i, j, Vi_dry, Vj_dry, poro_i, poro_j) -> tuple[float, float]:
        """Delegate porosity mixing to the porosity growth kernel."""
        lv_i = float(self.liquid_volume[i])
        lv_j = float(self.liquid_volume[j])
        sat_i = self.saturation[i] if not np.isnan(poro_i) else 0.0
        sat_j = self.saturation[j] if not np.isnan(poro_j) else 0.0

        # Simplified collision energy E ~ 0.5 m v^2 with v ~ G * d_eff.
        rho = 1000.0  # kg/m^3, assumed granule density
        d_eff = float(self.X[i] + self.X[j]) * 0.5
        v_rel = float(getattr(self, "G", 1000.0)) * d_eff
        E_coll = 0.5 * (rho * float(Vi_dry + Vj_dry)) * v_rel**2

        kernel = self._porosity_growth_kernel()
        return kernel.compute_merged_porosity(
            v_dry1=float(Vi_dry), poro1=poro_i,
            v_dry2=float(Vj_dry), poro2=poro_j,
            v_liq1=lv_i, v_liq2=lv_j,
            sat1=sat_i, sat2=sat_j,
            collision_energy=E_coll,
            solver=self,
        )

    def _porosity_growth_kernel(self):
        """Configured porosity growth kernel, or a cached ``volume_mixing`` default."""
        manager = getattr(self, "kernel_manager", None)
        kernel = getattr(manager, "porosity_growth_kernel", None) if manager else None
        if kernel is not None:
            return kernel

        kernel = getattr(self, "_default_porosity_kernel", None)
        if kernel is None:
            from .kernels.porosity_growth import get_porosity_growth_kernel

            kernel = get_porosity_growth_kernel("volume_mixing")
            self._default_porosity_kernel = kernel
        return kernel

    def _merge_liquid(
        self, i, j, new_idx, Vi_dry, Vj_dry, poro_i, poro_j, V_dry_merged
    ) -> None:
        """Combine the parents' liquid into the child and set its saturation.

        ``liquid_volume`` is intensive (per physical particle), so the child
        simply receives the *sum* of the parents' liquid - no ``dW/W`` scaling.
        A self-collision (``i == j``) merges two physical particles of the same
        computational particle, hence the factor 2.
        """
        sat_i = self.saturation[i] if not np.isnan(poro_i) else 0.0
        sat_j = self.saturation[j] if not np.isnan(poro_j) else 0.0

        V_pore_i = Vi_dry * poro_i if not np.isnan(poro_i) else 0.0
        V_pore_j = Vj_dry * poro_j if not np.isnan(poro_j) else 0.0

        V_liq_int_i = V_pore_i * sat_i if V_pore_i > 0 else 0.0
        V_liq_int_j = V_pore_j * sat_j if V_pore_j > 0 else 0.0

        liq_i = float(self.liquid_volume[i])
        liq_j = float(self.liquid_volume[j])
        V_liq_ext_i = liq_i - V_liq_int_i
        V_liq_ext_j = liq_j - V_liq_int_j

        # Braumann et al. (2007): surface liquid trapped in the newly formed
        # contact pores. Without a configured kernel this is 0 (no transfer).
        l_e_to_i = self.kernel_manager.compute_liquid_internalization_agglomeration(
            v_dry1=float(Vi_dry),
            v_dry2=float(Vj_dry),
            v_liq_ext1=float(V_liq_ext_i),
            v_liq_ext2=float(V_liq_ext_j),
            particle1_idx=i,
            particle2_idx=j,
            solver=self,
        )

        if i == j:
            V_liq_int_contrib = 2.0 * V_liq_int_i
            V_liq_ext_contrib = 2.0 * V_liq_ext_i
        else:
            V_liq_int_contrib = V_liq_int_i + V_liq_int_j
            V_liq_ext_contrib = V_liq_ext_i + V_liq_ext_j

        V_liq_int_merged = V_liq_int_contrib + l_e_to_i
        V_liq_ext_merged = V_liq_ext_contrib - l_e_to_i

        # Cannot internalise more than the available external liquid.
        if V_liq_ext_merged < 0.0:
            V_liq_int_merged = V_liq_int_contrib + V_liq_ext_contrib
            V_liq_ext_merged = 0.0

        poro_new = self.porosity[new_idx]
        V_pore_merged = V_dry_merged * poro_new if not np.isnan(poro_new) else 0.0

        sat_merged = V_liq_int_merged / V_pore_merged if V_pore_merged > 0 else 0.0

        # Pores cannot hold more than their volume: the excess becomes external.
        if sat_merged > 1.0:
            V_liq_ext_merged += V_pore_merged * (sat_merged - 1.0)
            V_liq_int_merged = V_pore_merged
            sat_merged = 1.0

        self.saturation[new_idx] = sat_merged
        self.liquid_volume[new_idx] = V_liq_int_merged + V_liq_ext_merged

    def _consume_parent_weight(self, i: int, j: int, dW: float) -> None:
        """Remove ``dW`` physical particles from each parent.

        Intensive properties (``liquid_volume``, ``porosity``, ``saturation``)
        stay untouched: the surviving physical particles are unchanged, only
        their count drops.
        """
        if i == j:
            # A self-collision consumes two physical particles per event.
            if i < self.a_tot:
                w_rem = float(self.W[i]) - 2.0 * dW
                if w_rem > 0.0:
                    self.W[i] = w_rem
                else:
                    self._remove_particle_column(i)
            return

        # Descending order so that swap-with-last removal cannot invalidate the
        # index we have not processed yet.
        for idx in sorted({int(i), int(j)}, reverse=True):
            if idx >= self.a_tot:
                continue
            w_rem = float(self.W[idx]) - dW
            if w_rem > 0.0:
                self.W[idx] = w_rem
            else:
                self._remove_particle_column(idx)

    def _refresh_samplers_after_agg(self) -> None:
        """Rebuild the agglomeration (and, in mix mode, breakage) samplers."""
        self._rebuild_all_propensities()
        self._agg_sampler = rebuild_sampler(self._agg_sampler, self._r_agg[: self.a_tot])

        if getattr(self, "process_break", False) or self.process_type in ("breakage", "mix"):
            self._calc_break_rates_full()
            self._break_sampler = rebuild_sampler(
                self._break_sampler, self._break_rate[: self.a_tot]
            )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _pairwise(kernel_name: str, n: int):
    """Pick the parallel or serial pairwise kernel for a population of size ``n``.

    Numba's parallel dispatch costs tens of microseconds per call. Below
    ``PARALLEL_MIN_N`` the O(n^2) loop finishes faster than the threads can be
    started, so the serial twin wins; above it the parallel version does. The
    two produce bit-identical output.
    """
    parallel_impl, serial_impl = _PAIRWISE_IMPLS[kernel_name]
    return parallel_impl if n >= PARALLEL_MIN_N else serial_impl


def _ensure_len(arr: np.ndarray | None, size: int) -> np.ndarray:
    """Return ``arr`` if it is at least ``size`` long, else a fresh zero array."""
    if arr is None or arr.shape[0] < size:
        return np.zeros(max(8, size), dtype=float)
    return arr
