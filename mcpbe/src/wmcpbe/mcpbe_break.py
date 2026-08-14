# Breakage mixin: breakage rates (full & single), two-level CDF builder, fragment production, single break event.
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# JIT helpers for CDF table generation (still needed for breakage)
from pbe_core.func.jit_mcpbe import _build_table_1d_jit, _build_tables_2d_jit

from .fenwick_new import rebuild_sampler
from .particle_merger import ParticleMerger

_ONE_SHOT_ADAPTER_NAMES = {"LMCRankAdapter", "LMCCopulaAdapter", "LMCFlowAdapter"}
_TABLE_ADAPTER_NAME = "LMCTableAdapter"
_LIVE_FALLBACK_ERROR = "LMCLiveFallback"
_LIVE_DISABLE_ERROR = "LMCLiveDisable"

class MCPBEBreak:
    """Breakage logic:
    - breakage rate table (full) from external JIT kernels
    - single-point breakage rate for incremental updates
    - two-level CDF builder (1D/2D) and discrete samplers
    - multi-fragment event by stochastic rounding of expected fragment count
    """

    @staticmethod
    def _adapter_type_name(adapter) -> str:
        return type(adapter).__name__ if adapter is not None else ""

    def _prepare_break_config(self) -> None:
        """Cache breakage configuration from kernel parameters."""
        self._bf_ready = False
        self._break_G = float(getattr(self, "G", 1000))
        self._break_pl_P1 = float(getattr(self, "pl_P1", 3e-2))
        self._break_pl_P2 = float(getattr(self, "pl_P2", 1.0))
        self._break_pl_P3 = float(getattr(self, "pl_P3", 1.0))
        self._break_pl_P4 = float(getattr(self, "pl_P4", 1.0))
        # Breakage FUNCTION parameters. `break_frag_v` / `break_frag_q` are the
        # canonical names; `pl_v` / `pl_q` remain the fallback for setups that
        # configure them directly. These are NOT the breakage rate exponent --
        # that one lives in the breakage kernel, also called `pl_v`.
        _frag_v = getattr(self, "break_frag_v", None)
        _frag_q = getattr(self, "break_frag_q", None)
        self._break_pl_v = float(_frag_v) if _frag_v is not None else float(getattr(self, "pl_v", 2.0))
        self._break_pl_q = float(_frag_q) if _frag_q is not None else float(getattr(self, "pl_q", 1.0))

        self._prepare_break_delta_config()
        self._break_dW_const = float(getattr(self, "_break_dW_const", 50.0))
        
        # Initialize particle merger for n_comp reduction
        # Enabled by default; can be disabled via enable_particle_merging=False
        if not hasattr(self, '_enable_particle_merging') or self._enable_particle_merging:
            self._particle_merger = ParticleMerger(
                self,
                use_hash_index=True,
                tol_rel=getattr(self, '_fragment_merge_tol', 1e-6),  # 0.0001% relative tolerance
                tol_abs_liquid=1e-30,
                bin_digits_volume=8,
                bin_digits_poro=4,
            )
        else:
            self._particle_merger = None

    # ------------------------------------------------------------------
    # Breakage rate (full table and single-point)
    # ------------------------------------------------------------------
    def _calc_break_rates_full(self):
        """Compute BREAKAGE PROPENSITIES for active slice.
    
        Stored in self._break_rate[:a] as:
            propensity_i = W[i] * S_i / delta_i
        with delta_i = min(break_dW_const, W[i]).
        where S_i is the single-particle breakage rate from Kernel/MLP/JIT.
        """
        a = self.a_tot
        if a <= 0:
            return
        cap = self._cap
        if (not hasattr(self, "_break_rate")
                or self._break_rate is None
                or self._break_rate.shape[0] < cap):
            self._break_rate = np.zeros(max(8, cap), dtype=float)
        if (not hasattr(self, "_delta_break")
                or self._delta_break is None
                or self._delta_break.shape[0] < cap):
            self._delta_break = np.zeros(max(8, cap), dtype=float)
        W = self.W[:a]
        delta = self._delta_from_weights(W, dW_const=float(self._break_dW_const))
        self._delta_break[:a] = delta
    
        # --------- Branch 0: Kernel Framework ---------
        if hasattr(self, 'kernel_manager') and self.kernel_manager is not None and self.kernel_manager.break_kernel is not None:
            self.V = self.V_flat[-1, :a]
            # Batch evaluation. Kernels whose rate is a pure function of volume
            # override compute_rate_array() with a vectorised expression; the
            # base class falls back to a per-particle loop, so state-dependent
            # kernels keep working unchanged.
            self.B_R = np.asarray(
                self.kernel_manager.break_kernel.compute_rate_array(
                    self.V_flat[-1, :a], solver=self
                ),
                dtype=float,
            )

            rates = self.B_R
            prop = np.divide(
                W * rates,
                delta,
                out=np.zeros_like(W, dtype=float),
                where=delta > 0.0,
            )
            np.maximum(prop, 0.0, out=prop)
            self._break_rate[:a] = prop
            if self._break_rate.shape[0] > a:
                self._break_rate[a:] = 0.0
                self._delta_break[a:] = 0.0
            return
    
        # --------- Branch 1: MLP model ---------
        use_mlp = bool(getattr(self, "lmc_use_breakage_model", False)) and (getattr(self, "lmc_breakage_adapter", None) is not None)
        if use_mlp:
            adapter = getattr(self, "lmc_breakage_adapter", None)
            if adapter is None:
                raise RuntimeError("MLP breakage model enabled but adapter not configured")
            rates = np.asarray(adapter.compute_rates_full(self), dtype=float)

            prop = np.divide(
                W * rates,
                delta,
                out=np.zeros_like(W, dtype=float),
                where=delta > 0.0,
            )
            np.maximum(prop, 0.0, out=prop)
            self._break_rate[:a] = prop
            if self._break_rate.shape[0] > a:
                self._break_rate[a:] = 0.0
                self._delta_break[a:] = 0.0
            return
    
        # --------- Branch 2: No breakage configured ---------
        # If no kernel and no MLP, assume breakage is disabled.
        # Set all rates to zero (no breakage events will occur).
        self._break_rate[:a] = 0.0
        if self._break_rate.shape[0] > a:
            self._break_rate[a:] = 0.0
            self._delta_break[a:] = 0.0
        return

    def _break_rate_single(self, i: int) -> float:
        """Single-particle BREAKAGE PROPENSITY.
    
        Returns:
            propensity_i = W[i] * S_i / delta_i
        with delta_i = min(break_dW_const, W[i]).
        where S_i is the single-particle breakage rate from MLP/JIT/Kernel.
        """
        a = self.a_tot
        if i < 0 or i >= a:
            return 0.0

        Wi = float(self.W[i])
        if Wi <= 0.0:
            if hasattr(self, "_delta_break") and self._delta_break is not None:
                self._delta_break[i] = 0.0
            return 0.0
        delta_i = self._update_delta_single(i, attr_name="_delta_break", dW_const=float(self._break_dW_const))
        if delta_i <= 0.0:
            return 0.0
    
        # --------- Branch 1: Kernel Framework ---------
        if hasattr(self, 'kernel_manager') and self.kernel_manager is not None and self.kernel_manager.break_kernel is not None:
            v_particle = float(self.V_flat[-1, i])
            Si = self.kernel_manager.compute_break_rate(
                v_particle,
                particle_idx=i,
                solver=self
            )
            val = Wi * Si / delta_i
            return float(val) if val > 0.0 else 0.0
    
        # --------- Branch 2: MLP model ---------
        use_mlp = bool(getattr(self, "lmc_use_breakage_model", False)) and (getattr(self, "lmc_breakage_adapter", None) is not None)
        if use_mlp:
            adapter = getattr(self, "lmc_breakage_adapter", None)
            if adapter is None:
                Si = 0.0
            else:
                Si = float(adapter.compute_rate_single(self, i))
            val = Wi * Si / delta_i
            return float(val) if val > 0.0 else 0.0
    
        # No kernel or MLP configured
        raise RuntimeError("Breakage kernel not initialized and no MLP model available")

    # ------------------------------------------------------------------
    # Two-level CDF builder (cached)
    # ------------------------------------------------------------------
    def _build_break_function(self, num_points: int = 1000):
        """
        Build two-level CDF tables for breakage fragment distributions.

        Generates cumulative distribution functions (CDFs) for sampling
        fragment sizes from power-law breakage models. Supports both
        1D (single-component) and 2D (bi-component) cases.

        Parameters
        ----------
        num_points : int, default=1000
            Number of grid points for CDF discretization.
            Higher values increase accuracy but memory usage.
            Typical range: 500 to 2000.

        Returns
        -------
        None
            Populates internal attributes:
            - dim=1: _bf1_rel, _bf1_cdf
            - dim=2: _bf2_rel1, _bf2_rel3, _bf2_rowsum_cdf, _bf2_row_cdf
            - Sets _bf_ready = True

        Raises
        ------
        None

        See Also
        --------
        _produce_one_frag_from_remaining : Sample fragments using built CDFs
        _get_break_tables_for_state : Retrieve CDFs for current state
        pbe_core.func.jit_mcpbe._build_table_1d_jit : JIT 1D CDF builder
        pbe_core.func.jit_mcpbe._build_tables_2d_jit : JIT 2D CDF builder

        Notes
        -----
        **CDF Structure:**

        For **1D** (single component):
        - _bf1_rel: Relative volume grid [0, 1] with num_points
        - _bf1_cdf: Cumulative probabilities P(V_frag/V_parent ≤ r)
        - Sampling: u ~ Uniform(0,1), find j where cdf[j-1] < u ≤ cdf[j]

        For **2D** (bi-component, e.g., solid + liquid):
        - _bf2_rel1: Grid for phase 1 fraction
        - _bf2_rel3: Grid for phase 2 fraction
        - _bf2_rowsum_cdf: Marginal CDF for phase 1 (row sums)
        - _bf2_row_cdf: Conditional CDF[i, :] for phase 2 given phase 1
        - Two-level sampling: First draw i from rowsum, then j from row[i]

        **Power-Law Model:**

        Fragment distribution based on parameters:
        - pl_v: Volume exponent (controls large vs small fragments)
        - pl_q: Shape parameter (distribution width)
        - BREAKFVAL: Fragment count model (default=1 for binary breakage)

        **Caching:**

        Generated CDFs cached by key = (dim, num_points, BREAKFVAL, pl_v, pl_q).
        Repeated calls with same parameters return cached tables instantly.
        Cache stored in self._bf_cache dictionary.

        **Performance:**

        - 1D table: ~0.1 ms (JIT compiled)
        - 2D table: ~1-5 ms (JIT compiled, O(n²))
        - Cached lookup: <1 μs
        - Memory: ~8 KB per 1D table, ~64 KB per 2D table (float64)

        **Usage Pattern:**

        Called once during initialization or before first breakage event.
        Subsequent events reuse cached tables via _get_break_tables_for_state().

        Examples
        --------
        >>> solver._break_pl_v = 2.0
        >>> solver._break_pl_q = 1.0
        >>> solver._build_break_function(num_points=1000)
        >>> # Tables now ready for fragment sampling
        >>> frag = solver._produce_one_frag_from_remaining(V_parent)
        """
        # BREAKFVAL is pinned to 1 here on purpose, and deliberately does NOT
        # follow `solver.BREAKFVAL` (base default 3), which drives the fragment
        # COUNT in mcpbe_base._compute_frag_num(). The two are allowed to
        # disagree: BREAKFVAL is a legacy switch that still carries other
        # meanings elsewhere, so aligning them would silently change the
        # fragment size distribution of every existing breakage run.
        #
        # Consequence worth knowing: with bf=1, breakage_func_1d() returns the
        # constant theta = 4.0 (see pbe-core/func/jit_kernel_break.py). The
        # constant normalises away in the CDF, so the fragment size distribution
        # is currently UNIFORM, and `_break_pl_v` / `_break_pl_q` do not enter
        # the formula at all -- they only take part in the cache key below.
        # Do not read them as active physics parameters.
        BREAKFVAL = 1
        key = (
            int(self.dim),
            int(num_points),
            BREAKFVAL,
            self._break_pl_v,
            self._break_pl_q,
        )
        cached = self._bf_cache.get(key, None)
        if cached is not None:
            # restore cached tables
            if self.dim == 1:
                self._bf1_rel, self._bf1_cdf = cached
            else:
                self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf = cached
            self._bf_ready = True
            return

        # grids
        if self.dim == 1:
            rel = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            v = self._break_pl_v
            q = self._break_pl_q
            cdf = _build_table_1d_jit(rel, v, q, BREAKFVAL)
            self._bf1_rel = rel
            self._bf1_cdf = cdf
            self._bf_cache[key] = (self._bf1_rel, self._bf1_cdf)
        else:
            rel1 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            rel3 = np.linspace(0.0, 1.0, num_points).astype(np.float64)
            v = self._break_pl_v
            q = self._break_pl_q
            rowsum_cdf, row_cdf = _build_tables_2d_jit(rel1, rel3, v, q, BREAKFVAL)
            self._bf2_rel1 = rel1
            self._bf2_rel3 = rel3
            self._bf2_rowsum_cdf = rowsum_cdf
            self._bf2_row_cdf = row_cdf
            self._bf_cache[key] = (self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf)

        self._bf_ready = True

    # [LMC-ADAPT] helpers
    def _state_AX1_from_Vrem(self, Vrem: np.ndarray) -> Tuple[float, float]:
        """Infer parent-particle state (A, X1) from remaining volume Vrem.

        A is total amount; X1 is phase-1 fraction.
        Uses a safe fallback when denominator is zero.
        """
        if self.dim == 1:
            A = float(Vrem[0])
            X1 = 1.0  # Single-component case; X1 is not used in practice.
        else:
            v1 = float(Vrem[0])
            v3 = float(Vrem[1])
            A = v1 + v3
            X1 = (v1 / A) if A > 0.0 else 0.5
        return A, X1
            
    def _get_break_tables_for_state(self, Vrem: np.ndarray):
        """
                Return CDF tables for the current parent-particle state.

                - If precomputed LMC adapter is enabled and available:
                    fetch 1D/2D tables from adapter.
                - Otherwise:
                    use JIT-generated tables via _build_break_function() and self._bf*.
        """
        use_lmc = bool(getattr(self, "use_lmc_pre_model", False) and getattr(self, "lmc_adapter", None) is not None)
        if not use_lmc:
            if not self._bf_ready:
                self._build_break_function()
            if self.dim == 1:
                return ("1d", self._bf1_rel, self._bf1_cdf, None, None, None, None, float(self.frag_num))
            else:
                return ("2d", self._bf2_rel1, self._bf2_rel3, self._bf2_rowsum_cdf, self._bf2_row_cdf, None, None, float(self.frag_num))

        # LMC table-driven path
        A, X1 = self._state_AX1_from_Vrem(Vrem)

        # Use 1D table for pure-phase/degenerate states; otherwise 2D.
        if self.dim == 2 and (Vrem[0] <= 0.0 or Vrem[1] <= 0.0):
            rel1d, cdf1d, zmin1d, pexp = self.lmc_adapter.get_1d(A, X1)
            return ("1d", rel1d, cdf1d, None, None, zmin1d, None, float(pexp))
        if self.dim == 1:
            rel1d, cdf1d, zmin1d, pexp = self.lmc_adapter.get_1d(A, X1)
            return ("1d", rel1d, cdf1d, None, None, zmin1d, None, float(pexp))
        else:
            rel1, rel3, rowsum_cdf, row_cdf, zmin1, zmin3, pexp = self.lmc_adapter.get_2d(A, X1)
            return ("2d", rel1, rel3, rowsum_cdf, row_cdf, zmin1, zmin3, float(pexp))
        
    # ------------------------------------------------------------------
    # Produce one fragment from remaining volume vector
    # ------------------------------------------------------------------
    def _produce_one_frag_from_remaining(self, Vrem: np.ndarray) -> np.ndarray:
        """
        Generate a single fragment from remaining parent volume via CDF sampling.

        Core fragment production method using pre-computed CDF tables (1D or 2D).
        Called iteratively to produce multiple fragments per breakage event.

        Parameters
        ----------
        Vrem : np.ndarray
            Remaining parent volume vector with shape (dim,):
            - dim=1: [V_total]
            - dim=2: [V_phase1, V_phase2]
            Must be positive (Vrem > 0).

        Returns
        -------
        np.ndarray
            Fragment volume vector with same shape as Vrem:
            - dim=1: [V_frag]
            - dim=2: [V_frag_phase1, V_frag_phase2]
            Satisfies: 0 ≤ V_frag ≤ Vrem (component-wise)

        Raises
        ------
        None

        See Also
        --------
        _build_break_function : Build CDF tables used for sampling
        _get_break_tables_for_state : Retrieve CDFs for current state
        _break_build_fragments : Generate all fragments via iterative calls

        Notes
        -----
        **Sampling Algorithm:**

        For **1D** (single component):
        1. Draw u ~ Uniform(0, 1)
        2. Binary search: find j where cdf[j-1] < u ≤ cdf[j]
        3. Relative fragment size: r = rel[j]
        4. Absolute volume: V_frag = r × Vrem

        For **2D** (bi-component):
        1. Draw u1, u2 ~ Uniform(0, 1) independently
        2. First level: draw i from rowsum_cdf using u1
        3. Second level: draw j from row_cdf[i, :] using u2
        4. Relative sizes: r1 = rel1[i], r3 = rel3[j]
        5. Handle degenerate cases (pure phase 1 or phase 2 only)

        **Fragment Size Distribution:**

        Determined by power-law parameters (pl_v, pl_q):
        - pl_v > 1: Favors larger fragments
        - pl_v < 1: Favors smaller fragments
        - pl_q controls distribution width

        **Usage in Breakage Events:**

        Called (n-1) times to generate n fragments:
        ```python
        Vrem = V_parent.copy()
        frags = []
        for _ in range(n-1):
            frag = _produce_one_frag_from_remaining(Vrem)
            frags.append(frag)
            Vrem -= frag  # Reduce remaining volume
        frags.append(Vrem)  # Last fragment gets remainder
        ```

        **Edge Cases:**

        - Pure phase 1 (Vrem[1]=0): Returns [r×Vrem[0], 0]
        - Pure phase 2 (Vrem[0]=0): Returns [0, r×Vrem[1]]
        - Both zero: Returns [0, 0] (should not occur in practice)

        **Performance:**

        - 1D sampling: ~0.5 μs (binary search on 1000 points)
        - 2D sampling: ~1 μs (two-level binary search)
        - Dominated by RNG calls, not table lookup

        Examples
        --------
        >>> # 1D case
        >>> Vrem = np.array([1e-9])  # 1 µm^3 particle
        >>> frag = solver._produce_one_frag_from_remaining(Vrem)
        >>> print(f"Fragment volume: {frag[0]:.3e} m^3")

        >>> # 2D case (solid + liquid)
        >>> Vrem = np.array([0.8e-9, 0.2e-9])  # 80% solid, 20% liquid
        >>> frag = solver._produce_one_frag_from_remaining(Vrem)
        >>> # Fragment maintains similar composition (statistically)
        """
        # Get CDF tables for current state
        mode, rA, rB, rowsum_cdf, row_cdf, *_ = self._get_break_tables_for_state(Vrem)

        if mode == "1d":
            u = float(self._rng.random())
            rel, cdf = rA, rB
            j = int(np.searchsorted(cdf, u, side="right"))
            if j >= rel.size: j = rel.size - 1
            r = float(rel[j])
            r = max(0.0, min(1.0, r))
            if self.dim == 1:
                return np.array([r * Vrem[0]], dtype=float)
            else:
                if Vrem[0] > 0.0 and Vrem[1] <= 0.0:
                    return np.array([r * Vrem[0], 0.0], dtype=float)
                elif Vrem[1] > 0.0 and Vrem[0] <= 0.0:
                    return np.array([0.0, r * Vrem[1]], dtype=float)
                return np.array([0.0, 0.0], dtype=float)

        else:
            # 2D two-level CDF
            u1 = float(self._rng.random())
            u2 = float(self._rng.random())
            i = int(np.searchsorted(rowsum_cdf, u1, side="right"))
            if i >= rA.size: i = rA.size - 1
            row = row_cdf[i]
            j = int(np.searchsorted(row, u2, side="right"))
            if j >= rB.size: j = rB.size - 1
            r1 = float(rA[i])
            r3 = float(rB[j])
            r1 = max(0.0, min(1.0, r1))
            r3 = max(0.0, min(1.0, r3))
            frag = np.array([r1 * Vrem[0], r3 * Vrem[1]], dtype=float)
            frag = np.minimum(frag, Vrem)
            frag = np.maximum(frag, 0.0)
            return frag

    # ------------------------------------------------------------------
    # Single breakage event (multi-fragment)
    # ------------------------------------------------------------------

    @staticmethod
    def _frag_total_volume(frag: np.ndarray) -> float:
        return float(np.sum(np.asarray(frag, dtype=float)))

    def _fragments_have_zero_volume(self, frags: list[np.ndarray]) -> bool:
        if not frags:
            return False
        for frag in frags:
            if self._frag_total_volume(frag) <= 0.0:
                return True
        return False

    def _filter_positive_volume_fragments(self, frags: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for frag in frags:
            if self._frag_total_volume(frag) > 0.0:
                out.append(np.asarray(frag, dtype=float))
        return out

    # Unified post-processing: apply fragments and maintain break/agg states.
    def _break_apply_and_maintain(self, k: int, frags: list[np.ndarray], dW: float, Vrem_k: np.ndarray) -> None:
        """Apply one *packet* breakage event.
    
        Interpretation:
          - Parent compute particle k represents W[k] real particles of volume Vk.
          - This call breaks dW of those real particles (dW can be non-integer).
          - The remaining (W[k]-dW) real particles stay at the same Vk (same compute particle k).
          - Broken products are represented by appending fragment compute particles,
            each with weight = dW and volume equal to the fragment volume of ONE real parent.
        """
        # DEBUG: Track fragment processing
        if not hasattr(self, '_break_debug_stats'):
            self._break_debug_stats = {
                'attempted': 0,
                'passed_weight_check': 0,
                'produced_fragments': 0,
                'passed_volume_filter': 0,
                'reached_merger': 0,
                'merged': 0,
                'created_new': 0,
            }
        self._break_debug_stats['attempted'] += 1
        
        if (not frags) or (dW <= 0.0):
            return
    
        w_parent_old = float(self.W[k])
        if w_parent_old <= 0.0:
            self._mark_unbreakable(k)
            return
    
        dW = float(min(dW, w_parent_old))
        if dW <= 0.0:
            self._mark_unbreakable(k)
            return
        
        self._break_debug_stats['passed_weight_check'] += 1

        # === DEBUG BREAK: VOR FRAGMENTIERUNG ===
        if getattr(self, 'mcpbe_debug_mass', False):
            max_events = getattr(self, '_debug_max_events', 20)
            if not hasattr(self, '_break_debug_counter'):
                self._break_debug_counter = 0
            if self._break_debug_counter < max_events:
                self._break_debug_counter += 1
                print(f"\n[DEBUG BREAK] Event #{self.real_break_events+1:.0f}")
                print(f"  Parent: k={k}")
                print(f"  W[k]={self.W[k]:.2f}, dW={dW:.2f}")
                print(f"  V_dry[k]={self.V_flat[-1,k]:.6e}")
                print(f"  poro[k]={self.porosity[k]:.4f}")
                
                v_solid_k = self.V_flat[-1,k] * (1.0 - self.porosity[k]) if not np.isnan(self.porosity[k]) else self.V_flat[-1,k]
                print(f"  V_solid[k]={v_solid_k:.6e}")
                print(f"  liq[k]={self.liquid_volume[k]:.6e}")
# ====================================

        Vrem_ref = np.asarray(Vrem_k, dtype=float).copy()
        resample_attempts = 0
        max_resample = 1000
        while self._fragments_have_zero_volume(frags) and resample_attempts < max_resample:
            status_retry, frags_retry = self._break_build_fragments(Vrem_ref.copy())
            if status_retry == "disable":
                self._mark_unbreakable(k)
                return
            if status_retry == "ok" and frags_retry:
                frags = frags_retry
            resample_attempts += 1

        frags_before_filter = len(frags)
        frags = self._filter_positive_volume_fragments(frags)
        if not frags:
            # Track filtered-out fragments
            self._break_debug_stats['filtered_out'] = self._break_debug_stats.get('filtered_out', 0) + (frags_before_filter - len(frags))
            return
        
        self._break_debug_stats['passed_volume_filter'] += len(frags)
        self._break_debug_stats['filtered_out'] = self._break_debug_stats.get('filtered_out', 0) + (frags_before_filter - len(frags))
    
        # Get parent liquid volume (PER PHYSICAL PARTICLE - intensive property)
        lv_parent = float(self.liquid_volume[k]) if hasattr(self, "liquid_volume") else 0.0
        V_parent_total = float(np.sum(Vrem_k)) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])

        new_indices: list[int] = []

        # Calculate total fragment volume for proportional liquid distribution
        frag_volumes = [float(np.sum(f)) for f in frags]
        total_frag_vol = sum(frag_volumes)

        # 1) Process ALL fragments: try to merge with existing particles or create new ones
        # IMPORTANT: liquid_volume is PER PHYSICAL PARTICLE (intensive).
        # When a particle breaks, its liquid is distributed among fragments.
        # Each fragment inherits a fraction of the parent's liquid proportional to its volume.
        # Total liquid is conserved: sum(fragment_liquid) = parent_liquid

        # Pre-compute fragment properties for all fragments (needed for deferred porosity)
        frag_props = []  # List of (V_solid, V_dry, V_intern_0, V_extern_0, poro, component, vol_fraction)
        parent_poro = self.porosity[k] if hasattr(self, "porosity") else np.nan
        parent_volume_total = float(np.sum(self.V_flat[:self.dim, k]))

        has_liquid_tracking = (hasattr(self, "liquid_volume") and hasattr(self, "porosity")
                                and hasattr(self, "saturation"))

        # Parent's pore volume and internal/external liquid split BEFORE breakage.
        # Needed to (a) distribute liquid like V_solid/V_pore (proportional to
        # fragment volume) and (b) compute how much internal liquid gets
        # externalized when pore volume is lost to new fracture surfaces
        # (see kernel_manager.compute_liquid_externalization_breakage below).
        if has_liquid_tracking and not np.isnan(parent_poro):
            V_dry_parent = float(self.V_flat[-1, k])
            S_p = float(self.saturation[k])
            if not np.isfinite(S_p):
                S_p = 0.0
            S_p = max(0.0, min(1.0, S_p))
            V_pore_p = V_dry_parent * parent_poro
            # Cap at the actually stored liquid: porosity/saturation/liquid_volume
            # are independently-set intensive properties elsewhere in the code
            # (e.g. nucleation, manual test setup) and are not guaranteed to be
            # mutually consistent. V_pore_p * S_p is only a theoretical capacity;
            # matches the same cap used by get_V_liquid_internal/external.
            V_intern_p = min(V_pore_p * S_p, lv_parent)
        else:
            S_p = 0.0
            V_pore_p = 0.0
            V_intern_p = 0.0
        V_extern_p = max(0.0, lv_parent - V_intern_p)

        # Get porosity kernel once for all fragments
        porosity_kernel = None
        if hasattr(self, "porosity"):
            if hasattr(self, 'kernel_manager') and self.kernel_manager is not None:
                porosity_kernel = self.kernel_manager.porosity_growth_kernel
            if porosity_kernel is None:
                from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
                porosity_kernel = get_porosity_growth_kernel('volume_mixing')

        breakage_energy = None
        use_deferred_poro = (hasattr(self, "porosity") and len(frags) > 1 and
                             porosity_kernel is not None and
                             hasattr(porosity_kernel, '_compute_fragment_porosity_multi'))

        for idx_frag, f in enumerate(frags):
            V_solid_frag = float(np.sum(f))
            vol_fraction = frag_volumes[idx_frag] / total_frag_vol if total_frag_vol > 0.0 else 0.0

            # Compute porosity (immediate or deferred)
            if hasattr(self, "porosity"):
                if use_deferred_poro:
                    frag_poro = parent_poro  # Placeholder, will be updated later
                else:
                    frag_poro = porosity_kernel.compute_fragment_porosity(
                        parent_porosity=parent_poro,
                        fragment_volume=V_solid_frag,
                        parent_volume=parent_volume_total,
                        breakage_energy=breakage_energy,
                        solver=self
                    ) if porosity_kernel is not None else parent_poro

                # Modern convention: poreless fragments get porosity=0.0, never
                # NaN (NaN is only the legacy "Vollkoerper" sentinel some
                # kernels still emit). Numerically identical for V_dry_frag,
                # since dividing by (1 - 0.0) == dividing by (1 - NaN-branch).
                if frag_poro is None or np.isnan(frag_poro):
                    frag_poro = 0.0
                V_dry_frag = V_solid_frag / (1.0 - frag_poro)
            else:
                frag_poro = None
                V_dry_frag = V_solid_frag

            # Distribute parent's internal/external liquid like V_solid/V_pore:
            # proportional to the fragment's volume fraction. The aggregate
            # pore-volume loss (and resulting externalization) is applied in a
            # second pass below, once all fragment pore volumes are known.
            V_intern_frag_0 = V_intern_p * vol_fraction
            V_extern_frag_0 = V_extern_p * vol_fraction

            frag_props.append((V_solid_frag, V_dry_frag, V_intern_frag_0, V_extern_frag_0,
                                frag_poro, f.copy(), vol_fraction))

        # Deferred porosity computation for multi-fragment kernels
        if use_deferred_poro:
            parent_volume_solid = float(np.sum(self.V_flat[:self.dim, k]))
            frag_solid_volumes = [fp[0] for fp in frag_props]
            poros_all = porosity_kernel._compute_fragment_porosity_multi(
                parent_porosity=parent_poro,
                fragment_volumes=frag_solid_volumes,
                parent_volume=parent_volume_solid,
                breakage_energy=breakage_energy,
                solver=self
            )
            # Update frag_props with final porosities and recompute V_dry
            frag_props_updated = []
            for i, fp in enumerate(frag_props):
                V_solid_frag, _, V_intern_frag_0, V_extern_frag_0, _, f_comp, vol_fraction = fp
                frag_poro = poros_all[i]
                if frag_poro is None or np.isnan(frag_poro):
                    frag_poro = 0.0
                V_dry_frag = V_solid_frag / (1.0 - frag_poro)
                frag_props_updated.append((V_solid_frag, V_dry_frag, V_intern_frag_0, V_extern_frag_0,
                                            frag_poro, f_comp, vol_fraction))
            frag_props = frag_props_updated

        # 2) Liquid externalization (reverse Braumann): pore volume lost to new
        # fracture surfaces pushes internal liquid out to the surface.
        # ΔV_pore = V_pore_p - Σ(frag_poro_i * V_dry_frag_i); V_liq_i2e = S_p * ΔV_pore
        V_liq_i2e = 0.0
        if has_liquid_tracking and V_pore_p > 0.0:
            V_pore_fragments_total = sum(
                fp[1] * fp[4] for fp in frag_props if fp[4] is not None
            )
            kmgr = getattr(self, 'kernel_manager', None)
            if kmgr is not None:
                V_liq_i2e = kmgr.compute_liquid_externalization_breakage(
                    v_pore_parent=V_pore_p,
                    v_pore_fragments_total=V_pore_fragments_total,
                    saturation_parent=S_p,
                    particle_idx=k,
                    solver=self
                )
                # Re-clamp against the ACTUAL (capped) internal liquid: the
                # kernel's own clamp uses S_p * V_pore_p, which can exceed
                # V_intern_p above when porosity/saturation/liquid_volume are
                # mutually inconsistent (see cap comment above).
                V_liq_i2e = max(0.0, min(V_liq_i2e, V_intern_p))

        # Finalize per-fragment liquid volume + saturation: redistribute the
        # externalized amount proportionally, same as everything else.
        # sum(liq_frag) == lv_parent exactly (only int/ext buckets shift).
        frag_props_final = []
        for (V_solid_frag, V_dry_frag, V_intern_frag_0, V_extern_frag_0,
             frag_poro, f_comp, vol_fraction) in frag_props:
            V_intern_frag = max(0.0, V_intern_frag_0 - V_liq_i2e * vol_fraction)
            V_extern_frag = V_extern_frag_0 + V_liq_i2e * vol_fraction
            liq_frag = V_intern_frag + V_extern_frag

            if frag_poro is not None:
                V_pore_frag = V_dry_frag * frag_poro
                sat_frag = (V_intern_frag / V_pore_frag) if V_pore_frag > 0.0 else 0.0
                sat_frag = max(0.0, min(1.0, sat_frag))
            else:
                sat_frag = None

            frag_props_final.append((V_solid_frag, V_dry_frag, liq_frag, frag_poro, sat_frag, f_comp))
        frag_props = frag_props_final

        # Now process each fragment: try to merge or create new
        for idx_frag, (V_solid_frag, V_dry_frag, liq_frag, frag_poro, sat_frag, f_comp) in enumerate(frag_props):
            # DEBUG: Track merger calls
            self._break_debug_stats['reached_merger'] += 1
            
            # Use particle merger to find existing or create new
            if self._particle_merger is not None:
                new_idx, was_merged = self._particle_merger.find_or_create(
                    V_solid_target=f_comp,
                    V_dry_target=V_dry_frag,
                    liquid_target=liq_frag,
                    poro_target=frag_poro,
                    sat_target=sat_frag,
                    weight_to_add=dW,
                    component_sum=f_comp
                )
                if was_merged:
                    self._break_debug_stats['merged'] += 1
                else:
                    self._break_debug_stats['created_new'] += 1
            else:
                # Fallback: always create new particle (original behavior)
                self._append_particle_column(f_comp)
                new_idx = self.a_tot - 1
                self.W[new_idx] = dW
                if hasattr(self, "liquid_volume"):
                    self.liquid_volume[new_idx] = liq_frag
                if hasattr(self, "porosity"):
                    self.porosity[new_idx] = frag_poro if frag_poro is not None else np.nan
                if hasattr(self, "saturation") and sat_frag is not None:
                    self.saturation[new_idx] = sat_frag
                was_merged = False
                self._break_debug_stats['created_new'] += 1
            
            new_indices.append(new_idx)
            
            # === DEBUG BREAK: NACH MERGER ===
            if getattr(self, 'mcpbe_debug_mass', False):
                max_events = getattr(self, '_debug_max_events', 20)
                if hasattr(self, '_break_debug_counter') and self._break_debug_counter <= max_events:
                    print(f"  Frag[{idx_frag}]: idx={new_idx}, was_merged={was_merged}")
                    if was_merged:
                        print(f"    MERGED: W[{new_idx}] increased by dW={dW:.2f}")
                        print(f"    Intensive properties UNCHANGED (correct!)")
                    else:
                        print(f"    CREATED NEW: W[{new_idx}]={self.W[new_idx]:.2f}")
                        V_dry_new = self.V_flat[-1, new_idx]
                        poro_new = self.porosity[new_idx] if hasattr(self, 'porosity') else np.nan
                        V_solid_new = V_dry_new * (1.0 - poro_new) if not np.isnan(poro_new) else V_dry_new
                        print(f"    V_dry={V_dry_new:.6e}, V_solid={V_solid_new:.6e}, liq={self.liquid_volume[new_idx]:.6e}")
# ==============================
            
            # Initialize delta and breakage rate (only for newly created particles)
            if not was_merged:
                self._update_delta_single(new_idx, attr_name="_delta_break", dW_const=float(self._break_dW_const))
                br_new = self._break_rate_single(new_idx)
                self._break_rate[new_idx] = br_new
                if self._break_sampler is not None:
                    self._break_sampler.update(new_idx, br_new)
            # If merged: W was already updated by merger, samplers will be refreshed below
        
        # 2) Reduce parent weight but keep its volume and liquid unchanged
        w_rem = w_parent_old - dW
        self.W[k] = w_rem
        
        # IMPORTANT: DO NOT scale liquid_volume!
        # liquid_volume is PER PHYSICAL PARTICLE (intensive property).
        # The remaining computational particles (weight w_rem) still represent
        # physical particles with the SAME liquid content per particle.
        # Only the NUMBER of represented particles changes, not their properties.
        
        if w_rem > 0.0:
            self._update_delta_single(k, attr_name="_delta_break", dW_const=float(self._break_dW_const))
            br_k = self._break_rate_single(k)  # uses new W[k]
            self._break_rate[k] = br_k
            if self._break_sampler is not None:
                self._break_sampler.update(k, br_k)
        else:
            # Parent population fully consumed -> remove compute particle k
            # Remove from hash index first (if merger enabled)
            if self._particle_merger is not None:
                self._particle_merger.remove_from_hash_index(k)
            self._remove_particle_column(k)

        # 3) Agglomeration maintenance (mix mode): full weighted rebuild for consistency
        pt = self.process_type
        if pt in ("agglomeration", "mix") and self._agg_sampler is not None:
            self._rebuild_all_propensities()
            self._agg_sampler = rebuild_sampler(self._agg_sampler, self._r_agg[:self.a_tot])
        
        # === DEBUG BREAK: NACH ANWENDUNG ===
        if getattr(self, 'mcpbe_debug_mass', False):
            max_events = getattr(self, '_debug_max_events', 20)
            if hasattr(self, '_break_debug_counter') and self._break_debug_counter <= max_events:
                # Summiere Fragment-Volumina
                v_solid_frags = 0.0
                liq_frags = 0.0
                
                # Hole Parent-Wert fuer Vergleich (falls noch existent)
                if k < self.a_tot:
                    v_solid_parent_remaining = self.V_flat[-1,k] * (1.0 - self.porosity[k]) if not np.isnan(self.porosity[k]) else self.V_flat[-1,k]
                    w_parent_remaining = self.W[k]
                else:
                    v_solid_parent_remaining = 0.0
                    w_parent_remaining = 0.0
                
                print(f"  After breakage: a_tot={self.a_tot}")
                print(f"  Parent remaining: W={w_parent_remaining:.2f}, V_solid={v_solid_parent_remaining * w_parent_remaining:.6e}")
# ================================
    
    # Mark particle as unbreakable: zero out breakage rate and update sampler.
    def _mark_unbreakable(self, k: int) -> None:
        self._break_rate[k] = 0.0
        if hasattr(self, "_delta_break") and self._delta_break is not None:
            self._delta_break[k] = 0.0
        if self._break_sampler is not None:
            self._break_sampler.update(k, 0.0)
    
    # Wrap legacy stepwise splitting into a helper that samples fragments
    # from table/builtin CDFs one by one.
    def _build_fragments_stepwise(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
        *_, pexp = self._get_break_tables_for_state(Vrem_k)
        p = float(pexp) if (pexp is not None and getattr(self, "use_lmc_pre_model", False)) else float(getattr(self, "frag_num", 2))
        fl = int(math.floor(p))
        ce = int(math.ceil(p))
        if fl <= 1:
            fl = 2
            ce = 2
        n = ce if (self._rng.random() < (p - fl)) else fl
    
        Vrem = Vrem_k.copy()
        frags = []
        for _ in range(n - 1):
            frag = self._produce_one_frag_from_remaining(Vrem)
            frags.append(frag)
            Vrem -= frag
        frags.append(Vrem)
        return frags
    
    # Live LMC small-particle fallback: generate NO_FRAG uniform fragments.
    def _build_uniform_live_fragments(self, Vrem_k: np.ndarray) -> list[np.ndarray]:
        """
        When live LMC raises LMCLiveFallback (particle too small to host
        the desired number of lattice cells), fall back to a simple,
        deterministic uniform split into NO_FRAG fragments.

        - For dim=1: split total volume V into NO_FRAG equal parts.
        - For dim=2: split each phase volume (VA, VB) evenly into NO_FRAG
          fragments, keeping the overall composition unchanged.
        """
        # Prefer NO_FRAG from the live LMC adapter; fall back to 2 if missing.
        n = int(getattr(self, "lmc_NO_FRAG", 2))
        if n < 2:
            n = 2

        if self.dim == 1:
            V = float(Vrem_k[0])
            v = V / float(n)
            return [np.array([v], dtype=float) for _ in range(n)]

        VA = float(Vrem_k[0])
        VB = float(Vrem_k[1])
        vA = VA / float(n)
        vB = VB / float(n)
        return [np.array([vA, vB], dtype=float) for _ in range(n)]
    
    # Unified fragment-source dispatcher: Rank one-shot / Live LMC / stepwise split.
    def _break_build_fragments(self, Vrem_k: np.ndarray) -> tuple[str, list[np.ndarray]]:
        """
        Dispatch order:
          1) Live LMC (if enabled): on Fallback/Disable, fall back or disable.
          2) Rank/Copula/Flow tables (if available): one-shot sampling.
          3) Marginal table or analytic function: stepwise splitting.

        Returns:
          ("ok", frags) or ("disable", [])
        """
        # 1) Live LMC first
        if getattr(self, "use_lmc_live", False) and (getattr(self, "lmc_live", None) is not None):
            try:
                frags, _E = self.lmc_live.sample_one_shot(Vrem_k, self._rng)
                return "ok", frags

            except Exception as exc:
                exc_name = type(exc).__name__
                if exc_name == _LIVE_FALLBACK_ERROR:
                    # Conditional fallback:
                    #   - if a table/rank model (or adapter) is available, keep the
                    #     original behavior and fall through to those models;
                    #   - otherwise, fall back to a simple uniform NO_FRAG split.
                    has_tables = bool(getattr(self, "use_lmc_pre_model", False))
                    has_adapter = getattr(self, "lmc_adapter", None) is not None

                    if has_tables or has_adapter:
                        # Old behavior: do nothing here and let the code fall through
                        # to the table / rank-based breakage models below.
                        pass
                    else:
                        # New behavior: no table/rank model available, so use a
                        # simple deterministic uniform split into NO_FRAG fragments.
                        frags = self._build_uniform_live_fragments(Vrem_k)
                        return "ok", frags
                elif exc_name == _LIVE_DISABLE_ERROR:
                    # Caller can mark this particle as unbreakable after receiving "disable".
                    return "disable", []
                else:
                    raise
    
        # 2) One-shot distribution adapters: rank / copula / flow
        lmc_ad = getattr(self, "lmc_adapter", None)
        if lmc_ad is None:
            # No adapter available, fall through to stepwise split
            return "ok", self._build_fragments_stepwise(Vrem_k)
        ad_name = self._adapter_type_name(lmc_ad)
        if ad_name in _ONE_SHOT_ADAPTER_NAMES:
        # if ad_name in {"LMCRankAdapter", "LMCCopulaAdapter"}:
            # --- Small-particle policy check (only required for "disable") ---
            if lmc_ad.small_particle_policy == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
            # Build A and X1
            if self.dim == 1:
                A = float(Vrem_k[0])
                X1 = 1.0
            else:
                A = float(Vrem_k[0] + Vrem_k[1])
                X1 = float(Vrem_k[0] / A) if A > 0.0 else 0.5
    
            # One-shot method signatures differ by adapter type.
            if ad_name == "LMCFlowAdapter":
                # flow: sample_one_shot(A, X1, rng, N=None)
                rA_list, rB_list = lmc_ad.sample_one_shot(A, X1, self._rng, N=None)
            else:
            # rank / copula: sample_one_shot(A, X1, rng, N=None, K_use=None, tail_strategy="equal")
                rA_list, rB_list = lmc_ad.sample_one_shot(
                    A, X1, self._rng, N=None, K_use=None, tail_strategy="equal"
                )
    
            # Reconstruct volume fragments by dimensionality.
            if self.dim == 1:
                frags = [np.array([r * Vrem_k[0]], dtype=float) for r in rA_list]
            else:
                frags = [
                    np.array([rA * Vrem_k[0], rB * Vrem_k[1]], dtype=float)
                    for (rA, rB) in zip(rA_list, rB_list)
                ]
            return "ok", frags
    
        # 3) Marginal-table / analytic-function path: stepwise split
        #    (LMCTableAdapter or pure JIT analytic model)
        if ad_name == _TABLE_ADAPTER_NAME:
            if getattr(lmc_ad, "small_particle_policy", None) == "disable":
                A = float(Vrem_k[0]) if self.dim == 1 else float(Vrem_k[0] + Vrem_k[1])
                if not lmc_ad.eligible_for_tables(A):
                    return "disable", []
    
        # Use original stepwise splitting logic.
        return "ok", self._build_fragments_stepwise(Vrem_k)

    def _compute_dW(self, k: int) -> float:
        """Compute packet size Î”W for breakage using a single constant event size."""
        Wk = float(self.W[k])
        if Wk <= 0.0 or not np.isfinite(Wk):
            return 0.0
        dW = min(float(self._break_dW_const), Wk)
    
        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)
    
    def _compute_dW_packet(self, k: int) -> float:
        """Compute packet size delta_i for breakage."""
        if k < 0 or k >= self.a_tot:
            return 0.0
        if hasattr(self, "_delta_break") and self._delta_break is not None:
            dW = float(self._delta_break[k])
        else:
            dW = self._update_delta_single(k, attr_name="_delta_break", dW_const=float(self._break_dW_const))
        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)

    def _do_one_break(self):
        """
        Execute a single breakage event (multi-fragment).

        Core Monte Carlo step for breakage: selects a particle, generates
        fragments via configured breakage model (MLP, LMC tables, or analytic),
        applies fragments to solver state, and updates samplers.

        Returns
        -------
        None
            Modifies solver state in-place:
            - Creates new fragment particles
            - Reduces parent weight by dW
            - Updates breakage propensity sampler
            - Increments real_break_events counter

        Raises
        ------
        None
            Errors handled gracefully (unbreakable particles marked and skipped)

        See Also
        --------
        _break_build_fragments : Generate fragments from parent volume
        _break_apply_and_maintain : Apply fragments to solver state
        _compute_dW_packet : Calculate packet size
        _mark_unbreakable : Mark particles that cannot break

        Notes
        -----
        **Fragment Generation Models (priority order):**

        1. **Live LMC** (use_lmc_live=True):
           On-the-fly lattice Monte Carlo simulation.
           Most accurate but computationally expensive.
           Falls back to uniform split if particle too small.

        2. **One-shot Adapters** (Rank/Copula/Flow):
           Pre-computed fragment distributions.
           Fast sampling from stored CDF tables.
           Requires eligibility check (particle size vs table range).

        3. **Marginal Tables / Analytic** (stepwise):
           Sequential fragment production.
           Default fallback when no advanced model available.
           Uses BREAKFVAL parameter for fragment count distribution.

        **Event Sequence:**

        1. **Select particle**: Draw k proportional to breakage propensity r_k.

        2. **Check breakability**: 
           - W_k > 0 (has weight)
           - dW_packet > 0 (sufficient packet size)
           - Not previously marked unbreakable

        3. **Generate fragments**:
           - Call _break_build_fragments(V_k)
           - Returns list of fragment volumes [V_frag1, V_frag2, ...]
           - Fragment count n determined by BREAKFVAL model

        4. **Apply fragments**:
           - Create n new particles with weight = dW each
           - Distribute parent liquid proportionally to fragment volume
           - Compute fragment porosities via porosity growth kernel
           - Reduce parent weight: W_k -= dW

        5. **Refresh samplers**: Update breakage propensity for next event

        **Packet-Based Breakage:**

        Unlike agglomeration (which breaks exactly 1 packet), this method
        can handle variable packet sizes via _compute_dW_packet(). The packet
        size determines how many *real* particles are broken in one event.

        **Liquid Distribution:**

        Parent liquid volume distributed among fragments proportionally:
        lv_frag_i = lv_parent × (V_frag_i / V_parent)

        Ensures total liquid conservation: sum(lv_frag) = lv_parent

        Internally, the parent's internal/external liquid split (V_intern_p,
        V_extern_p) is distributed the same way, then adjusted for pore
        volume lost to new fracture surfaces (reverse Braumann, see
        kernel_manager.compute_liquid_externalization_breakage): fragment
        saturation is recomputed from its own (possibly reduced) pore volume,
        never inherited or left unset. Only int/ext buckets shift -- the
        per-fragment total lv_frag_i and its conservation are unaffected.

        **Porosity Computation:**

        Fragment porosities computed via PorosityGrowthKernel:
        - Default: volume_mixing (inherit parent porosity)
        - Advanced: cone_model (pore collapse during breakage)
        - Multi-fragment kernels deferred until all fragments collected

        **Small Particle Handling:**

        Particles below table threshold handled via:
        - Fallback: Uniform split into NO_FRAG equal fragments
        - Disable: Mark as unbreakable (small_particle_policy="disable")

        **Mass Conservation:**

        Solid mass strictly conserved:
        sum(W × V_solid)_before = sum(W × V_solid)_after

        Verified via validate_mass_conservation() helper.
        """
    
        self._ensure_break_sampler()
    
        # DEBUG: Initialize stats EARLY (before ANY return)
        if not hasattr(self, '_break_debug_stats'):
            self._break_debug_stats = {
                'attempted': 0,
                'passed_weight_check': 0,
                'produced_fragments': 0,
                'passed_volume_filter': 0,
                'filtered_out': 0,
                'reached_merger': 0,
                'merged': 0,
                'created_new': 0,
            }
    
        # If no breakable weight exists, return directly.
        if self._break_sampler.total() <= 0.0:
            return
    
        for _ in range(max(1, self.a_tot)):
            if self._break_sampler.total() <= 0.0:
                return
    
            k = self._break_sampler.sample(self._rng)
            
            # DEBUG: Track entry into loop (after sample)
            self._break_debug_stats['attempted'] += 1
    
            # Available weight represented by this compute particle
            Wk0 = float(self.W[k])
            if Wk0 <= 0.0:
                self._mark_unbreakable(k)
                continue
    
            dW_total = self._compute_dW_packet(k)
            if dW_total <= 0.0:
                self._mark_unbreakable(k)
                continue

            # Store real-event count consumed by this packet event for dt update in solve().
            self._last_break_dW = float(dW_total)

            # Single packet event (no additional chunking, fixed N=1).
            if k >= self.a_tot:
                return

            Wk_now = float(self.W[k])
            if Wk_now <= 0.0:
                return

            dW = min(float(dW_total), Wk_now)
            if dW <= 0.0:
                return
            
            self._break_debug_stats['passed_weight_check'] += 1

            if self.dim == 1:
                Vrem_k = np.array([self.V_flat[0, k]], dtype=float)
            else:
                Vrem_k = np.array([self.V_flat[0, k], self.V_flat[1, k]], dtype=float)
            
            status, frags = self._break_build_fragments(Vrem_k)
            
            # Track fragments produced
            if status == "ok" and frags:
                self._break_debug_stats['produced_fragments'] += len(frags)

            if status == "disable":
                self._mark_unbreakable(k)
                return

            if status == "ok":
                self._break_apply_and_maintain(k, frags, dW, Vrem_k)
            else:
                return
    
            return  # Current breakage event completed.

        return


