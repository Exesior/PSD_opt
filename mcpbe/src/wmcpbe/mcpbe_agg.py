"""Agglomeration mixin for the weighted MC-PBE solver.

Responsibilities
----------------
* ``_rebuild_all_propensities`` - recompute the bias-corrected propensity
  ``R_i* = W_i sum_j W_j beta(i,j)/min(dW_i,dW_j)`` for every active particle.
  Runs once per accepted event and dominates the solver runtime; see
  :mod:`wmcpbe.kernels.aggregation.jit_kernels`.
* ``_do_one_agg``               - execute a single agglomeration event:
  pick a pair, decide the packet size ``dW``, merge volumes / porosity /
  liquid, consume the parents' weight, refresh the samplers.

Bias correction (Ji & Rhein, Eqs. 33/36-41)
-------------------------------------------
Because the executed batch is capped by the available weight of *both* partners
(``dW_ij = min(dW_i, dW_j)``, ``dW_i = min(dW, W_i)``), the sampling probability
has to be divided by exactly that cap, and the partial rate carries a ``W_i``
prefactor. Partner ``j`` is drawn proportional to ``W_j beta(i,j)/dW_ij``
(Eq. 40), not proportional to ``W_j`` alone. Derivation and verification:
``mcpbe/docs/Bias_Correction_und_Gewichtsdisziplin.md``.

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

from .fenwick_new import rebuild_sampler
from .particle_merger import ParticleMerger
from .kernels.aggregation.jit_kernels import (
    KERNEL_IDS,
    MOMENT_KERNELS,
    PARALLEL_MIN_N,
    kernel_p0,
    pick_partner_pairdelta,
    rebuild_r_pairdelta_liquid_bridge,
    rebuild_r_pairdelta_liquid_bridge_serial,
    rebuild_r_pairdelta_moment,
    rebuild_r_pairdelta_pairwise,
    rebuild_r_pairdelta_pairwise_serial,
    separable_tables,
)

#: Propensity evaluation modes for :attr:`MCPBEAgg.agg_propensity_mode`.
PROPENSITY_MODES = ("pairwise", "moment")


class MCPBEAgg:
    """
    Agglomerations-Physik und Event-Durchfuehrung.
    
    Diese Mixin-Klasse implementiert den Agglomerationsprozess fuer den gewichteten
    DSMC-PBE-Solver. Sie behandelt:
    
    * Berechnung der Kollisionsrate β(i,j) ueber modulare Kernel
    * Partnerauswahl via FenwickSampler (O(log n))
    * Groessenabhaengige Akzeptanz (SIZEEVAL-Kriterium)
    * Durchfuehrung von Agglomerationsereignissen mit Massenerhaltung
    
    Die Propensity-Berechnung (r_i = Σ_j W_j·β(i,j)) kann im paarweisen O(n²)-Modus
    oder im optimierten O(n)-Momentenmodus erfolgen (siehe agg_propensity_mode).
    
    Attributes
    ----------
    agg_propensity_mode : str
        Berechnungsmodus: 'pairwise' (O(n²), bitgenau) oder 'moment' (O(n), ~1e-15 Abweichung)
    kernel_manager : KernelManager
        Verwaltet alle Physik-Kernel (aggregation, acceptance, porosity_growth)
    
    Notes
    -----
    Volumina-Semantik:
    - V_flat[:dim]: Feststoffvolumen pro Komponente (erhalten bei Agglomeration)
    - V_flat[-1]: Trockenvolumen (V_solid + V_pore, aendert sich mit Porositaet)
    - liquid_volume, porosity, saturation: intensive Groessen (pro physical particle)
    
    Massenerhaltung:
    Bei Agglomeration gilt: V_solid_merged = V_solid_i + V_solid_j
    Das Kindpartikel erbt die Summe der Eltern-Feststoffvolumina.
    
    See Also
    --------
    kernels.aggregation.jit_kernels : JIT-kompilierte Kernel fuer Propensity-Rebuild
    fenwick_new.FenwickSampler : Effizientes gewichtetes Sampling
    MOMENT_MODE.md : Detaillierte Herleitung des O(n)-Momentenmodus
    """

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
    agg_propensity_mode: str = "moment"

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
    # Partner selection (paper Eq. 40)
    # ------------------------------------------------------------------
    def _compiled_kernel_spec(self) -> tuple[int, float] | None:
        """``(kernel_id, p0)`` if the compiled fast path applies, else ``None``.

        The compiled kernels fold the scalar collision efficiency ``alpha`` into
        ``p0``. That is only exact for ``dim == 1``, where ``alpha`` is a single
        number shared by every pair; in 2D it depends on the component mix of the
        pair and the generic Python path is used instead.
        """
        if int(self.dim) != 1:
            return None
        manager = getattr(self, "kernel_manager", None)
        kernel = getattr(manager, "agg_kernel", None) if manager is not None else None
        if kernel is None:
            return None
        name = getattr(kernel, "name", "")
        if name not in KERNEL_IDS:
            return None
        alpha = self._alpha_scalar()
        if alpha <= 0.0:
            return None
        return KERNEL_IDS[name], kernel_p0(name, kernel) * alpha

    def _alpha_scalar(self) -> float:
        """Scalar collision efficiency for ``dim == 1``."""
        if np.ndim(self.alpha_prim) == 0:
            return float(self.alpha_prim)
        return float(np.mean(self.alpha_prim))

    def _pick_partner_kernel(
        self, i: int, R: np.ndarray, W: np.ndarray, delta: np.ndarray,
        dW_const: float, partner_total: float, u_sel: float,
    ) -> tuple[int, float]:
        """Draw partner ``j`` proportional to ``W_j beta(i,j) alpha(i,j) / dW_ij``.

        This is the conditional stage of the two-stage sampler (paper Eqs. 17/40).
        ``partner_total`` is ``R_i* / W_i``, i.e. the very sum that
        :meth:`_rebuild_all_propensities` accumulated for this ``i``, so the
        conditional distribution is consistent with the primary draw by
        construction.

        Returns ``(j, w_ij)``; ``j == -1`` means the draw failed.
        """
        a = len(W)
        if a < 1 or i < 0 or i >= a or partner_total <= 0.0:
            return -1, 0.0
        di = float(delta[i])
        if di <= 0.0:
            return -1, 0.0

        compute_beta = self.kernel_manager.compute_beta
        dim = int(self.dim)
        ri = float(R[i])
        thresh = u_sel * partner_total
        acc = 0.0
        last_j, last_w = -1, 0.0

        for j in range(a):
            if j == i:
                Wi = float(W[i])
                if Wi <= 1.0:
                    continue
                sd = min(di, 0.5 * Wi)
                if sd <= 0.0:
                    continue
                b = self._safe_beta(compute_beta, ri, ri, i, i)
                if dim != 1:
                    b *= self._alpha_ccm(i, i)
                if b <= 0.0:
                    continue
                wij = (Wi - 1.0) * b / sd
            else:
                Wj = float(W[j])
                dj = float(delta[j])
                if Wj <= 0.0 or dj <= 0.0:
                    continue
                b = self._safe_beta(compute_beta, ri, float(R[j]), i, j)
                if dim != 1:
                    b *= self._alpha_ccm(i, j)
                if b <= 0.0:
                    continue
                wij = Wj * b / (di if di < dj else dj)

            if wij <= 0.0:
                continue
            acc += wij
            last_j, last_w = j, wij
            if acc > thresh:
                return j, wij

        if last_j < 0 or last_w <= 0.0:
            return -1, 0.0
        return last_j, last_w

    def _safe_beta(self, compute_beta, r1: float, r2: float, i: int, j: int) -> float:
        """``compute_beta`` with the kernel-failure guard, returning 0 on error."""
        try:
            b = compute_beta(r1, r2, particle1_idx=i, particle2_idx=j, solver=self)
        except Exception as exc:
            from .kernel_integration import KernelEvaluationError

            if isinstance(exc, KernelEvaluationError):
                self._warn_once(
                    "agg_beta_failed",
                    f"Aggregation kernel '{exc.kernel_name}' failed: {exc.message}. "
                    f"Collision rejected.",
                )
            else:
                self._warn_once(
                    "agg_beta_failed",
                    f"Aggregation kernel raised {type(exc).__name__}: {exc}. "
                    f"Treating the collision as rejected.",
                )
            return 0.0
        if not np.isfinite(b) or b <= 0.0:
            return 0.0
        return float(b)

    def _accept_sizeeval(self, i: int, j: int, u_acc: float) -> bool:
        """SIZEEVAL size-dependent acceptance.

        Uses its own random draw ``u_acc``. The previous implementation reused the
        partner-selection variate for this test, which correlated the two and was
        documented as a known statistical defect; the two-stage sampler makes that
        reuse impossible anyway.
        """
        if int(getattr(self, "SIZEEVAL", 1)) == 0:
            return True

        V = self.V_flat
        if int(self.dim) == 1:
            Vi, Vj = float(V[0, i]), float(V[0, j])
        else:
            Vi = float(np.sum(V[: self.dim, i]))
            Vj = float(np.sum(V[: self.dim, j]))

        Vmean2 = self._mean_initial_volume_sq(self.a_tot)
        V_target = np.sqrt(Vmean2) if Vmean2 > 0 else 1e-18
        sigma_V = V_target * float(getattr(self, "Y_SEL", 1.06))
        if sigma_V <= 0.0:
            return True

        size_factor = np.exp(-0.5 * ((Vi - V_target) / sigma_V) ** 2) * np.exp(
            -0.5 * ((Vj - V_target) / sigma_V) ** 2
        )
        return u_acc <= size_factor

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
        """
        Recompute per-particle agglomeration propensities after an event.

        Calculates r_i = sum_j(W_j * beta(i,j)) for all active particles,
        where beta(i,j) is the aggregation kernel rate. The propensity is
        normalized by packet size delta_i for weighted DSMC sampling.

        This method dominates solver runtime (>90% typically). Uses JIT-compiled
        batch kernels when available (O(n) moment mode or O(n²) pairwise).

        Returns
        -------
        None
            Updates self._r_agg[:a_tot] and self._delta_agg[:a_tot] in-place.

        Raises
        ------
        RuntimeError
            If aggregation kernel not initialized in kernel_manager.

        Warns
        -----
        RuntimeWarning
            Once if falling back to generic O(n²) Python loop for custom kernels.

        See Also
        --------
        _compute_raw_propensities : Kernel-specific propensity computation
        _raw_propensities_generic : Fallback for unoptimized kernels
        wmcpbe.kernels.aggregation.jit_kernels : Optimized batch implementations

        Notes
        -----
        **Algorithm:**

        1. Get active particle count a = a_tot
        2. Compute radii R from diameters X
        3. Set packet sizes: delta_i = min(dW_const, W_i)
        4. Compute raw propensities r_i = sum_j(W_j * beta(i,j))
           - Use moment mode O(n) if available and enabled
           - Otherwise use pairwise O(n²) JIT or Python fallback
        5. Normalize: r_i /= delta_i (for packet-based sampling)
        6. Zero out inactive entries (a_tot:_cap)

        **Performance Modes:**

        - **Moment mode** (agg_propensity_mode="moment", default):
          O(n) scaling, uses analytical moments for separable kernels.
          Available for: shear_chin1998, brownian_tsouris1995, sum, constant.
          Speedup: up to 275× at n=4000 particles.

        - **Pairwise mode** (agg_propensity_mode="pairwise"):
          O(n²) explicit summation over all pairs.
          Required for: liquid_bridge and custom kernels without moment form.
          Slower but exact for non-separable kernels.

        **Packet Size Normalization:**

        The propensity r_i is divided by delta_i = min(dW_max, W_i) because:
        - Fenwick sampler draws events proportional to r_i
        - Each event consumes a packet of size dW (not single collision)
        - Normalization ensures correct physical event rates

        **Memory Layout:**

        Buffers (_r_agg, _delta_agg) pre-allocated to capacity (_cap).
        Only first a_tot entries are valid; remainder zeroed for safety.
        Recycled across events to avoid allocation overhead.
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

        # delta_i = min(dW_const, W_i)  [paper Eq. 33], zeroed where W is
        # non-finite or non-positive. No epsilon threshold is needed: every
        # weight-consuming path drains a particle to exactly 0 (see
        # mcpbe_time_helper and docs/Bias_Correction_und_Gewichtsdisziplin.md).
        dW_const = float(getattr(self, "_agg_dW_const", None) or self._prepare_agg_delta_config())
        delta = buf["delta"][:a]
        np.minimum(W, dW_const, out=delta)
        np.copyto(delta, 0.0, where=~(np.isfinite(delta) & (delta > 0.0)))
        self._delta_agg[:a] = delta

        # The pair-delta division 1/min(delta_i, delta_j) and the W_i prefactor
        # are part of the kernel batch functions themselves - there is no
        # post-hoc division by delta_i any more (that was the biased form).
        r = buf["r"][:a]
        self._compute_raw_propensities(R, W, delta, dW_const, r)
        np.maximum(r, 0.0, out=r)

        self._r_agg[:a] = r
        self._r_agg[a:] = 0.0
        self._delta_agg[a:] = 0.0

    def _compute_raw_propensities(
        self,
        R: np.ndarray,
        W: np.ndarray,
        delta: np.ndarray,
        dW_const: float,
        out: np.ndarray,
    ) -> None:
        """Fill ``out[i] = R_i*``, the pair-delta corrected propensity (Eq. 36/37).

        ``R_i* = W_i [ sum_{j!=i} W_j beta(i,j)/min(dW_i,dW_j)
                     + (W_i-1) beta(i,i)/min(dW_i, W_i/2) ]``
        """
        manager = getattr(self, "kernel_manager", None)
        kernel = getattr(manager, "agg_kernel", None) if manager is not None else None
        if kernel is None:
            raise RuntimeError("Aggregation kernel not initialized")

        name = getattr(kernel, "name", "")
        mode = str(getattr(self, "agg_propensity_mode", "moment")).lower()
        if mode not in PROPENSITY_MODES:
            raise ValueError(
                f"agg_propensity_mode must be one of {PROPENSITY_MODES}, got {mode!r}"
            )

        a = out.shape[0]
        spec = self._compiled_kernel_spec()

        if spec is not None:
            kid, p0 = spec
            if mode == "moment" and name in MOMENT_KERNELS:
                alpha = self._alpha_scalar()
                F, G, beta_ii, _ = separable_tables(name, kernel, R)
                # alpha is a per-pair constant in 1D, so folding it into f_k
                # scales every beta(i,j) - and beta(i,i) - identically.
                rebuild_r_pairdelta_moment(
                    np.ascontiguousarray(F * alpha),
                    np.ascontiguousarray(G),
                    np.ascontiguousarray(beta_ii * alpha),
                    W, delta, dW_const, out,
                )
            else:
                fn = (
                    rebuild_r_pairdelta_pairwise
                    if a >= PARALLEL_MIN_N
                    else rebuild_r_pairdelta_pairwise_serial
                )
                fn(kid, p0, R, W, delta, dW_const, out)
            return

        if name == "liquid_bridge" and int(self.dim) == 1:
            self._raw_propensities_liquid_bridge(kernel, R, W, delta, dW_const, out)
            return

        self._raw_propensities_generic(R, W, delta, dW_const, out)

    def _raw_propensities_liquid_bridge(self, kernel, R, W, delta, dW_const, out) -> None:
        """Compiled O(n^2) pair-delta path for the liquid-bridge kernel.

        The Gaussian bridge factor contains a cross term ``s_i*s_j``, so the
        kernel is not separable and the quadratic loop stays - but it runs
        compiled instead of as ``n^2`` Python calls.
        """
        a = out.shape[0]
        alpha = self._alpha_scalar()
        v_liq_ext = self.get_V_liquid_external()
        if v_liq_ext.shape[0] != a:  # pragma: no cover - defensive
            v_liq_ext = v_liq_ext[:a]

        fn = (
            rebuild_r_pairdelta_liquid_bridge
            if a >= PARALLEL_MIN_N
            else rebuild_r_pairdelta_liquid_bridge_serial
        )
        fn(
            float(kernel.corr_beta) * alpha,
            float(kernel.g),
            float(kernel.s_opt),
            float(kernel.sigma_s),
            float(kernel.alpha_liq),
            R, W, delta, dW_const,
            np.ascontiguousarray(self.saturation[:a]),
            np.ascontiguousarray(v_liq_ext, dtype=np.float64),
            out,
        )

    def _raw_propensities_generic(self, R, W, delta, dW_const, out) -> None:
        """Fallback for kernels (or 2D setups) without a compiled batch path.

        Costs ``n^2`` Python-level kernel calls per event; emits a one-time
        warning so the cost is visible rather than mysterious. Applies the same
        pair-delta correction as the compiled paths.
        """
        a = out.shape[0]
        self._warn_once(
            "agg_generic_fallback",
            f"Aggregation kernel {getattr(self.kernel_manager.agg_kernel, 'name', '?')!r} "
            f"(dim={self.dim}) has no compiled batch path; falling back to an O(n^2) "
            "Python loop in _rebuild_all_propensities(). Add a JIT batch function in "
            "wmcpbe/kernels/aggregation/jit_kernels.py to make this usable at scale.",
        )
        compute_beta = self.kernel_manager.compute_beta
        dim = int(self.dim)

        for i in range(a):
            di = float(delta[i])
            Wi = float(W[i])
            if di <= 0.0 or Wi <= 0.0:
                out[i] = 0.0
                continue
            r1 = float(R[i])
            s = 0.0
            for j in range(a):
                if j == i:
                    continue
                dj = float(delta[j])
                Wj = float(W[j])
                if dj <= 0.0 or Wj <= 0.0:
                    continue
                b = self._safe_beta(compute_beta, r1, float(R[j]), i, j)
                if dim != 1:
                    b *= self._alpha_ccm(i, j)
                else:
                    b *= self._alpha_scalar()
                if b <= 0.0:
                    continue
                s += Wj * b / (di if di < dj else dj)
            if Wi > 1.0:
                sd = min(di, 0.5 * Wi)
                if sd > 0.0:
                    b = self._safe_beta(compute_beta, r1, r1, i, i)
                    b *= self._alpha_ccm(i, i) if dim != 1 else self._alpha_scalar()
                    if b > 0.0:
                        s += (Wi - 1.0) * b / sd
            val = Wi * s
            out[i] = val if val > 0.0 else 0.0

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
            # A self-collision consumes 2*dW physical particles from the same
            # packet, so the effective self batch is delta_ii = min(delta_i, W_i/2)
            # (paper Eq. 33). Capping - rather than rejecting when W_i <= 2*delta_i -
            # lets the final event drain the particle to exactly 0: W_i/2 is exact
            # in IEEE-754, hence 2*(W_i/2) == W_i bit for bit.
            delta_ii = min(delta_i, 0.5 * Wi)
            if delta_ii <= 0.0:
                return 0.0
            dW = min(dW, delta_ii)
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
        """
        Execute a single agglomeration event (if accepted).

        Core Monte Carlo step for agglomeration: selects a particle pair,
        computes packet size dW, merges volumes/porosity/liquid, updates weights,
        and refreshes samplers. Uses weighted DSMC algorithm.

        Returns
        -------
        None
            Modifies solver state in-place:
            - Creates new merged particle (child)
            - Reduces parent weights by dW
            - Updates propensity samplers
            - Increments real_agg_events counter

        Raises
        ------
        None
            Errors handled gracefully (rejected collisions return early)

        See Also
        --------
        _select_pair : Draw collision pair (i, j)
        _merge_pair : Create merged child particle
        _consume_parent_weight : Reduce parent weights
        _compute_agg_dW : Calculate packet size

        Notes
        -----
        **Event Sequence:**

        1. **Select pair**: Draw particle i proportional to propensity r_i,
           then draw j proportional to weight W_j. Apply SIZEEVAL filter
           and optional acceptance kernel (e.g., Stokes criterion).

        2. **Compute packet size**: dW = min(dW_max, W_i, W_j, delta_i, delta_j)
           where delta_i = min(dW_const, W_i). Ensures weight conservation.

        3. **Merge particles**: 
           - V_solid_child = V_solid_i + V_solid_j (conserved!)
           - V_dry_child from porosity growth kernel
           - Liquid redistributed (internal vs external)
           - Porosity recomputed from cone model or volume mixing

        4. **Update weights**: 
           - W_i -= dW, W_j -= dW (or W_i -= 2*dW for self-collision)
           - W_child = dW
           - Parents removed if W <= 0

        5. **Refresh samplers**: Rebuild propensity arrays for next event

        **Random Numbers:**

        Exactly 2 RNG calls per attempt:
        - u1: Fenwick sampler for particle i (proportional to r_i)
        - u2: Partner selection for j (proportional to W_j) + SIZEEVAL
        Plus additional calls if acceptance kernel requires them.

        **Rejection Criteria:**

        Event rejected (returns early) if:
        - a_tot < 2 (not enough particles)
        - Pair selection fails (j < 0 or pick_w <= 0)
        - Acceptance kernel rejects (Stokes criterion not met)
        - Packet size dW <= 0 (insufficient weight)

        **Mass Conservation:**

        Solid mass strictly conserved:
        sum(W × V_solid)_before = sum(W × V_solid)_after

        Verified in tests via validate_mass_conservation().
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
        """Two-stage pair draw (paper Eqs. 39/40).

        Stage 1 picks ``i`` proportional to ``R_i*`` via the Fenwick sampler,
        stage 2 picks ``j`` conditionally proportional to
        ``W_j beta(i,j)/dW_ij``. Returns ``(i, j, pick_w)`` or ``None``.
        """
        # 1) primary particle proportional to R_i*
        i = self._agg_sampler.sample(self._rng)

        R = self._agg_radii(a)
        W = self.W[:a]
        delta = self._delta_agg[:a]
        dW_const = float(getattr(self, "_agg_dW_const", None) or self._prepare_agg_delta_config())

        Wi = float(W[i]) if 0 <= i < a else 0.0
        if Wi <= 0.0:
            return None
        # partner_total == R_i*/W_i, i.e. the sum the rebuild accumulated for i.
        partner_total = float(self._r_agg[i]) / Wi

        # 2) conditional partner draw
        u_sel = float(self._rng.random())
        spec = self._compiled_kernel_spec()
        if spec is not None:
            kid, p0 = spec
            j, pick_w = pick_partner_pairdelta(
                kid, p0, i, R, W, delta, dW_const, partner_total, u_sel
            )
        else:
            j, pick_w = self._pick_partner_kernel(
                i, R, W, delta, dW_const, partner_total, u_sel
            )

        if j < 0 or pick_w <= 0.0:
            return None

        # 3) size-dependent acceptance (SIZEEVAL) with its own variate
        if int(getattr(self, "SIZEEVAL", 1)) != 0:
            if not self._accept_sizeeval(i, j, float(self._rng.random())):
                return None

        # 4) optional physical acceptance criterion (e.g. Stokes, Braumann 2007)
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
            # Track acceptance/rejection statistics
            if not hasattr(self, '_agg_accepted_count'):
                self._agg_accepted_count = 0
                self._agg_rejected_count = 0
            if accepted:
                self._agg_accepted_count += 1
            else:
                self._agg_rejected_count += 1
            if not accepted:
                return None

        return i, j, pick_w

    def _merge_pair(self, i: int, j: int, dW: float) -> int:
        """Create the merged child particle and return its index.

        Mass conservation: ``V_solid_child = V_solid_i + V_solid_j``. The dry
        volume follows from the porosity growth kernel, and the liquid is
        redistributed between the internal (pore) and external (surface)
        reservoirs.
        
        Uses ParticleMerger to find existing similar particles and merge weights
        instead of creating new particles (reduces n_comp).
        """
        # === DEBUG AGG: VOR MERGE ===
        if getattr(self, 'mcpbe_debug_mass', False):
            max_events = getattr(self, '_debug_max_events', 20)
            if not hasattr(self, '_debug_agg_count'):
                self._debug_agg_count = 0
            if self._debug_agg_count < max_events:
                self._debug_agg_count += 1
                print(f"\n[DEBUG AGG] Event #{self.real_agg_events+1:.0f}")
                print(f"  Parents: i={i}, j={j}")
                print(f"  W[i]={self.W[i]:.2f}, W[j]={self.W[j]:.2f}")
                print(f"  V_dry[i]={self.V_flat[-1,i]:.6e}, V_dry[j]={self.V_flat[-1,j]:.6e}")
                print(f"  poro[i]={self.porosity[i]:.4f}, poro[j]={self.porosity[j]:.4f}")
                
                # Berechne V_solid der Parents
                v_solid_i = self.V_flat[-1,i] * (1.0 - self.porosity[i]) if not np.isnan(self.porosity[i]) else self.V_flat[-1,i]
                v_solid_j = self.V_flat[-1,j] * (1.0 - self.porosity[j]) if not np.isnan(self.porosity[j]) else self.V_flat[-1,j]
                print(f"  V_solid[i]={v_solid_i:.6e}, V_solid[j]={v_solid_j:.6e}")
                print(f"  V_solid_SUM={v_solid_i + v_solid_j:.6e} (SOLLTE = Child V_solid sein)")
                print(f"  liq[i]={self.liquid_volume[i]:.6e}, liq[j]={self.liquid_volume[j]:.6e}")
                print(f"  liq_SUM={self.liquid_volume[i] + self.liquid_volume[j]:.6e}")
# ================================

        dim = self.dim
        Vi_comp = self.V_flat[:dim, i].copy()
        Vj_comp = self.V_flat[:dim, j].copy()
        Vi_dry = float(self.V_flat[-1, i])
        Vj_dry = float(self.V_flat[-1, j])

        poro_i = self.porosity[i]
        poro_j = self.porosity[j]

        # Universal formula: V_solid = V_dry * (1 - poro)
        # Works for both poro=0.0 (non-porous) and poro>0.0 (porous).
        # Legacy Vollkoerper (poro=NaN) have no pore space at all, so
        # V_solid = V_dry -- without this branch the NaN propagated straight
        # into V_flat[0] and destroyed that particle's mass.
        V_solid_i = Vi_dry if np.isnan(poro_i) else Vi_dry * (1.0 - poro_i)
        V_solid_j = Vj_dry if np.isnan(poro_j) else Vj_dry * (1.0 - poro_j)
        V_solid_merged = V_solid_i + V_solid_j
        
        # Compute component distribution for merged particle
        comp = Vi_comp + Vj_comp
        comp_total = float(np.sum(comp))
        if comp_total > 0.0:
            V_flat_dim = comp * (V_solid_merged / comp_total)
        else:
            V_flat_dim = np.full(dim, V_solid_merged / dim, dtype=float)
        
        # Compute merged porosity and V_dry BEFORE trying to find match
        V_dry_merged, poro_merged = self._merged_porosity(i, j, Vi_dry, Vj_dry, poro_i, poro_j)
        
        # Merged liquid and saturation. Needed BEFORE the ParticleMerger lookup,
        # because `liquid_target` is part of the match criteria -- which is why
        # this cannot be deferred to a post-creation helper (see B-09).
        liquid_merged, sat_merged = compute_merged_liquid(
            self, i, j,
            Vi_dry=Vi_dry, Vj_dry=Vj_dry,
            poro_i=poro_i, poro_j=poro_j,
            V_dry_merged=V_dry_merged, poro_merged=poro_merged,
        )

        # Try to find existing particle with matching properties using ParticleMerger
        if hasattr(self, '_particle_merger') and self._particle_merger is not None:
            new_idx, was_merged = self._particle_merger.find_or_create(
                V_solid_target=V_flat_dim,
                V_dry_target=V_dry_merged,
                liquid_target=liquid_merged,
                poro_target=poro_merged,
                sat_target=sat_merged,
                weight_to_add=dW,
                component_sum=V_flat_dim
            )
            
            if was_merged:
                # Existing particle found: ONLY W was updated -- that is the
                # ParticleMerger's contract and what makes the merge exactly
                # mass conserving (all other properties are intensive, so
                # adding another physical particle of the same kind does not
                # change them).
                #
                # This used to additionally overwrite saturation and
                # liquid_volume "to be safe". It was not safe: the match is
                # only within tolerance, so the overwrite applied that
                # difference to the target's ENTIRE existing weight, not just
                # to the dW being added -- and it silently changed the
                # particle's merger hash key behind the index's back. Breakage
                # never did this; the two paths had drifted.
                # See docs/Audit_2026-08-17.md, B-15.
                return new_idx
            # else: new particle created, continue with initialization below
        else:
            # Fallback: always create new particle (original behavior)
            self._append_particle_column(V_flat_dim)
            new_idx = self.a_tot - 1
            self.W[new_idx] = dW
        
        # Initialize properties for newly created particle
        self.porosity[new_idx] = poro_merged
        
        # V_flat[:dim] holds V_solid
        if dim == 1:
            self.V_flat[0, new_idx] = V_solid_merged
        else:
            self.V_flat[:dim, new_idx] = V_flat_dim
        
        # V_flat[-1] holds V_dry; X is derived from it and must follow.
        self.set_particle_dry_volume(new_idx, V_dry_merged)
        
        # Set liquid and saturation
        self.saturation[new_idx] = sat_merged
        self.liquid_volume[new_idx] = liquid_merged
        
        # === DEBUG AGG: NACH MERGE ===
        if getattr(self, 'mcpbe_debug_mass', False):
            max_events = getattr(self, '_debug_max_events', 20)
            if hasattr(self, '_debug_agg_count') and self._debug_agg_count <= max_events:
                print(f"  Child: idx={new_idx}, W={self.W[new_idx]:.2f}")
                print(f"  V_dry[child]={self.V_flat[-1,new_idx]:.6e}")
                print(f"  poro[child]={self.porosity[new_idx]:.4f}")
                
                v_solid_child = self.V_flat[-1,new_idx] * (1.0 - self.porosity[new_idx]) if not np.isnan(self.porosity[new_idx]) else self.V_flat[-1,new_idx]
                print(f"  V_solid[child]={v_solid_child:.6e}")
                print(f"  ΔV_solid={v_solid_child - V_solid_merged:.6e} (SOLLTE ≈ 0 sein)")
                
                print(f"  liq[child]={self.liquid_volume[new_idx]:.6e}")
                print(f"  Δliq={self.liquid_volume[new_idx] - liquid_merged:.6e}")
                print(f"  sat[child]={self.saturation[new_idx]:.4f}")
# ==============================
        
        return new_idx

    def _merged_porosity(self, i, j, Vi_dry, Vj_dry, poro_i, poro_j) -> tuple[float, float]:
        """Delegate porosity mixing to the porosity growth kernel."""
        lv_i = float(self.liquid_volume[i])
        lv_j = float(self.liquid_volume[j])
        sat_i = self.saturation[i]
        sat_j = self.saturation[j]

        kernel = self._porosity_growth_kernel()
        return kernel.compute_merged_porosity(
            v_dry1=float(Vi_dry), poro1=poro_i,
            v_dry2=float(Vj_dry), poro2=poro_j,
            v_liq1=lv_i, v_liq2=lv_j,
            sat1=sat_i, sat2=sat_j,
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

    # NOTE: `_merge_liquid()` used to live here. It was dead code -- the very
    # same liquid bookkeeping is done inline in `_merge_pair` (which needs the
    # values BEFORE the ParticleMerger lookup, to match on `liquid_target`,
    # while `_merge_liquid` wrote them AFTER the particle existed). Two copies
    # of one rule, one of them unreachable, is precisely the failure mode
    # documented in Fundamentals.md section 7: a fix applied to the readable,
    # documented method would have had no effect at all.
    # The shared, side-effect-free computation now lives in
    # `compute_merged_liquid()` at the bottom of this module and is called from
    # `_merge_pair`. See docs/Audit_2026-08-17.md, B-09.

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
                    # Remove from hash index first (if merger enabled)
                    if hasattr(self, '_particle_merger') and self._particle_merger is not None:
                        self._particle_merger.remove_from_hash_index(i)
                    self._remove_particle_column(i)
        else:
            # Descending order so that swap-with-last removal cannot invalidate the
            # index we have not processed yet.
            for idx in sorted({int(i), int(j)}, reverse=True):
                if idx >= self.a_tot:
                    continue
                w_rem = float(self.W[idx]) - dW
                if w_rem > 0.0:
                    self.W[idx] = w_rem
                else:
                    # Remove from hash index first (if merger enabled)
                    if hasattr(self, '_particle_merger') and self._particle_merger is not None:
                        self._particle_merger.remove_from_hash_index(idx)
                    self._remove_particle_column(idx)

        # === DEBUG AGG: NACH CONSUME ===
        if getattr(self, 'mcpbe_debug_mass', False):
            max_events = getattr(self, '_debug_max_events', 20)
            if hasattr(self, '_debug_agg_count') and self._debug_agg_count <= max_events:
                print(f"  After consume: a_tot={self.a_tot}")

                # Globale Massenbilanz
                v_dry = self.V_flat[-1, :self.a_tot]
                poro = self.porosity[:self.a_tot]
                w = self.W[:self.a_tot]
                valid = ~np.isnan(poro)
                v_solid = np.zeros_like(v_dry)
                v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
                v_solid[~valid] = v_dry[~valid]

                solid_total = np.sum(v_solid * w)
                liq_total = np.sum(self.liquid_volume[:self.a_tot] * w)
                n_phys = np.sum(w) / self.Vc

                print(f"  GLOBAL: V_solid_total={solid_total:.6e}, V_liq_total={liq_total:.6e}, n_phys={n_phys:.3e}")
# =================================

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
def compute_merged_liquid(
    solver, i: int, j: int, *,
    Vi_dry: float, Vj_dry: float,
    poro_i: float, poro_j: float,
    V_dry_merged: float, poro_merged: float,
) -> tuple[float, float]:
    """Liquid volume and saturation of the child of ``i`` and ``j``.

    Pure computation, no state is written -- deliberately, because the caller
    needs these values *before* the particle exists (``liquid_target`` is one of
    the ParticleMerger's match criteria). This replaces the dead
    ``MCPBEAgg._merge_liquid``, which held a second copy of the same rule and
    could never run. See docs/Audit_2026-08-17.md, B-09.

    ``liquid_volume`` is intensive, so the child receives the *sum* of the
    parents' liquid -- no ``dW/W`` scaling. A self-collision (``i == j``) merges
    two physical particles of the same computational particle, hence the
    factor 2.

    Returns ``(liquid_total, saturation)``.
    """
    sat_i = solver.saturation[i] if not np.isnan(poro_i) else 0.0
    sat_j = solver.saturation[j] if not np.isnan(poro_j) else 0.0

    V_pore_i = Vi_dry * poro_i if not np.isnan(poro_i) else 0.0
    V_pore_j = Vj_dry * poro_j if not np.isnan(poro_j) else 0.0

    V_liq_int_i = V_pore_i * sat_i if V_pore_i > 0 else 0.0
    V_liq_int_j = V_pore_j * sat_j if V_pore_j > 0 else 0.0

    liq_i = float(solver.liquid_volume[i])
    liq_j = float(solver.liquid_volume[j])
    V_liq_ext_i = liq_i - V_liq_int_i
    V_liq_ext_j = liq_j - V_liq_int_j

    # Braumann et al. (2007): surface liquid trapped in the newly formed
    # contact pores. Without a configured kernel this is 0 (no transfer).
    l_e_to_i = (
        solver.kernel_manager.compute_liquid_internalization_agglomeration(
            v_dry1=float(Vi_dry),
            v_dry2=float(Vj_dry),
            v_liq_ext1=float(V_liq_ext_i),
            v_liq_ext2=float(V_liq_ext_j),
            particle1_idx=i,
            particle2_idx=j,
            solver=solver,
        )
        if getattr(solver, "kernel_manager", None) is not None
        else 0.0
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

    V_pore_merged = V_dry_merged * poro_merged if not np.isnan(poro_merged) else 0.0
    sat_merged = V_liq_int_merged / V_pore_merged if V_pore_merged > 0 else 0.0

    # Pores cannot hold more than their volume: the excess becomes external.
    if sat_merged > 1.0:
        V_liq_ext_merged += V_pore_merged * (sat_merged - 1.0)
        V_liq_int_merged = V_pore_merged
        sat_merged = 1.0

    return V_liq_int_merged + V_liq_ext_merged, sat_merged


def _ensure_len(arr: np.ndarray | None, size: int) -> np.ndarray:
    """Return ``arr`` if it is at least ``size`` long, else a fresh zero array."""
    if arr is None or arr.shape[0] < size:
        return np.zeros(max(8, size), dtype=float)
    return arr
