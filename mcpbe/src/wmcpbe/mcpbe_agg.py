# Agglomeration mixin: beta/alpha hooks, r_agg rebuild (JIT), single agglomeration event.
from __future__ import annotations

import numpy as np
from numba import njit

from .fenwick_new import FenwickSampler

# JIT batch functions for new kernel framework
from .kernels.aggregation.jit_kernels import (
    rebuild_r_array_shear,
    rebuild_r_array_brownian,
    rebuild_r_array_constant,
    rebuild_r_array_sum,
)  # fmt: skip


# =============================================================================
# JIT-compiled partner selection for constant kernel
# =============================================================================

@njit(cache=True)
def pick_partner_constant_jit(
    i: int,
    W: np.ndarray,
    R: np.ndarray,
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
    corr_beta: float,
) -> tuple[int, float]:
    """
    JIT-compiled partner selection for CONSTANT kernel.
    
    This is a specialized version of _pick_partner_kernel() for the constant kernel,
    compiled with Numba for maximum performance.
    
    Returns:
        j: Selected partner index (-1 if rejected)
        pick_w: Pair propensity W[j]*beta or (W[i]-1)*beta for self-agglomeration
    """
    a = len(W)
    if a < 2 or i < 0 or i >= a:
        return -1, 0.0
    
    # Step 1: Select partner j with probability proportional to W[j]
    total_W = 0.0
    for j in range(a):
        total_W += W[j]
    
    if total_W <= 0.0:
        return -1, 0.0
    
    # Use cumulative weight distribution for sampling
    target = u_sel * total_W
    cumsum = 0.0
    j = 0
    for j_idx in range(a):
        cumsum += W[j_idx]
        if cumsum > target:
            j = j_idx
            break
    
    # Clamp to valid range
    if j >= a:
        j = a - 1
    
    # Step 2: Compute beta(i,j) for constant kernel
    beta_ij = corr_beta
    
    if beta_ij <= 0.0:
        return -1, 0.0
    
    # Step 3: Compute collision efficiency alpha(i,j)
    if dim == 1:
        alpha_ij = alpha1d
    else:
        # 2D: component-based mixing
        Vi0 = V0[i]; Vi1 = V1[i]; Vti = Vi0 + Vi1
        Vj0 = V0[j]; Vj1 = V1[j]; Vtj = Vj0 + Vj1
        if Vti <= 0.0 or Vtj <= 0.0:
            return -1, 0.0
        p0 = (Vi0 / Vti) * (Vj0 / Vtj)
        p1 = (Vi0 / Vti) * (Vj1 / Vtj)
        p2 = (Vi1 / Vti) * (Vj0 / Vtj)
        p3 = (Vi1 / Vti) * (Vj1 / Vtj)
        alpha_ij = float(p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3])
    
    if alpha_ij <= 0.0:
        return -1, 0.0
    
    # Step 4: Apply effective beta (beta * alpha)
    beta_eff = beta_ij * alpha_ij
    
    # Step 5: Compute pair propensity
    if i == j:
        # Self-agglomeration: (W[i] - 1) distinct pairs
        pick_w = max(0.0, float(W[i] - 1)) * beta_eff
    else:
        # Distinct particles: W[j] * beta_eff
        pick_w = float(W[j]) * beta_eff
    
    if pick_w <= 0.0:
        return -1, 0.0
    
    # Step 6: Size-dependent selection (SIZEEVAL)
    if SIZEEVAL != 0:
        if dim == 1:
            Vi = V0[i]
            Vj = V0[j]
        else:
            Vi = V0[i] + V1[i]
            Vj = V0[j] + V1[j]
        
        # Gaussian selection around target volume
        V_target = np.sqrt(Vmean2) if Vmean2 > 0 else 1e-18
        sigma_V = V_target * Y_SEL  # Width of selection window
        
        # Combined size factor for both particles
        diff_i = (Vi - V_target) / sigma_V if sigma_V > 0 else 0.0
        diff_j = (Vj - V_target) / sigma_V if sigma_V > 0 else 0.0
        size_factor = np.exp(-0.5 * diff_i * diff_i) * np.exp(-0.5 * diff_j * diff_j)
        
        # Reject if size factor too low
        if u_sel > size_factor:
            return -1, 0.0
    
    return j, pick_w


class MCPBEAgg:
    """Agglomeration logic:
    - _beta: wrapper to JIT kernel
    - _alpha_ccm: 2D alpha based on component fractions (kept for parity)
    - _rebuild_all_propensities: parallel JIT rebuild r_i = sum_j beta(i,j)
    - _do_one_agg: single event with incremental r updates + swap-pop removal
    """

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _alpha_ccm(self, idx1: int, idx2: int) -> float:
        """2D collision efficiency from component fractions and alpha_prim (length 4)."""
        if self.dim == 1:
            return float(self.alpha_prim if np.ndim(self.alpha_prim) == 0 else np.mean(self.alpha_prim))

        V = self.V_flat
        Vi0 = V[0, idx1]; Vi1 = V[1, idx1]; Vti = Vi0 + Vi1
        Vj0 = V[0, idx2]; Vj1 = V[1, idx2]; Vtj = Vj0 + Vj1
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

    def _pick_partner_kernel(
        self,
        i: int,
        R: np.ndarray,
        W: np.ndarray,
        delta: np.ndarray,
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
        u_acc: float = 0.0,  # Deprecated: acceptance now handled by separate kernel
    ) -> tuple[int, float]:
        """
        Weighted partner sampling with kernel-based beta calculation.
        
        This is a Python implementation of the partner selection algorithm,
        designed to be consistent with kernel-based propensity calculations.
        
        Design note: Implementation mirrors nb_pick_partner_weighted logic but uses
        kernel_manager.compute_beta() instead of hardcoded JIT kernel. This ensures
        consistency with _rebuild_all_propensities() which also uses kernel beta.
        
        Future optimization: This method can be JIT-compiled with numba
        by extracting it to a standalone function and adding @njit decorator.
        
        Args:
            i: Index of first particle (already selected from _agg_sampler)
            R: Particle radii array (X/2)
            W: Particle weights array
            delta: Delta values for propensity normalization
            V0, V1: Volume components (V1 unused in 1D)
            dim: Dimensionality (1 or 2)
            alpha1d: 1D collision efficiency
            alpha4: 2D collision efficiency array (length 4)
            SIZEEVAL: Size evaluation flag (0=disabled, 1=enabled)
            X_SEL, Y_SEL: Selection parameters
            Vmean2: Mean volume squared (for size-dependent selection)
            u_sel: Random number for partner selection [0,1]
            u_acc: Deprecated - acceptance now handled by separate kernel
        
        Returns:
            j: Selected partner index (-1 if rejected)
            pick_w: Pair propensity W[j]*beta(i,j) or (W[i]-1)*beta(i,i)
        """
        a = len(W)
        if a < 2 or i < 0 or i >= a:
            return -1, 0.0
        
        # Step 1: Select partner j with probability proportional to W[j]
        # This matches the logic in nb_pick_partner_weighted
        total_W = np.sum(W)
        if total_W <= 0.0:
            return -1, 0.0
        
        # Use cumulative weight distribution for sampling
        cumsum_W = np.cumsum(W)
        target = u_sel * total_W
        j = int(np.searchsorted(cumsum_W, target, side='right'))
        j = min(j, a - 1)  # Clamp to valid range
        
        # Step 2: Compute beta(i,j) using kernel framework (KEY DIFFERENCE FROM JIT!)
        r1 = float(R[i])
        r2 = float(R[j])
        
        try:
            beta_ij = self.kernel_manager.compute_beta(
                r1, r2,
                particle1_idx=i,
                particle2_idx=j,
                solver=self
            )
        except Exception:
            beta_ij = 0.0
        
        if not np.isfinite(beta_ij) or beta_ij <= 0.0:
            return -1, 0.0
        
        # Step 3: Compute collision efficiency alpha(i,j)
        if dim == 1:
            alpha_ij = alpha1d
        else:
            # 2D: component-based mixing
            Vi0 = V0[i]; Vi1 = V1[i]; Vti = Vi0 + Vi1
            Vj0 = V0[j]; Vj1 = V1[j]; Vtj = Vj0 + Vj1
            if Vti <= 0.0 or Vtj <= 0.0:
                return -1, 0.0
            p0 = (Vi0 / Vti) * (Vj0 / Vtj)
            p1 = (Vi0 / Vti) * (Vj1 / Vtj)
            p2 = (Vi1 / Vti) * (Vj0 / Vtj)
            p3 = (Vi1 / Vti) * (Vj1 / Vtj)
            alpha_ij = float(p0 * alpha4[0] + p1 * alpha4[1] + p2 * alpha4[2] + p3 * alpha4[3])
        
        if alpha_ij <= 0.0:
            return -1, 0.0
        
        # Step 4: Apply effective beta (beta * alpha)
        beta_eff = beta_ij * alpha_ij
        
        # Step 5: Compute pair propensity
        if i == j:
            # Self-agglomeration: (W[i] - 1) distinct pairs
            pick_w = max(0.0, float(W[i] - 1)) * beta_eff
        else:
            # Distinct particles: W[j] * beta_eff
            pick_w = float(W[j]) * beta_eff
        
        if pick_w <= 0.0:
            return -1, 0.0
        
        # Step 6: Size-dependent selection (SIZEEVAL)
        if SIZEEVAL != 0:
            if dim == 1:
                Vi = V0[i]
                Vj = V0[j]
            else:
                Vi = V0[i] + V1[i]
                Vj = V0[j] + V1[j]
            
            # Gaussian selection around target volume
            V_target = np.sqrt(Vmean2) if Vmean2 > 0 else 1e-18
            sigma_V = V_target * Y_SEL  # Width of selection window
            
            # Combined size factor for both particles
            size_factor = np.exp(-0.5 * ((Vi - V_target) / sigma_V) ** 2) * \
                         np.exp(-0.5 * ((Vj - V_target) / sigma_V) ** 2)
            
            # Reject if size factor too low
            if u_sel > size_factor:
                return -1, 0.0
        
        return j, pick_w

    def _beta(self, i: int, j: int) -> float:
        """Pair kernel β(i,j) via kernel framework or legacy JIT.
        
        Priority:
        1. Kernel framework (if kernel_manager exists)
        2. Legacy JIT kernel (backward compatibility)
        """
        # Use kernel framework
        if self.kernel_manager is None or self.kernel_manager.agg_kernel is None:
            raise RuntimeError("Aggregation kernel not initialized")
        
        r1 = float(self.X[i] * 0.5)
        r2 = float(self.X[j] * 0.5)
        return self.kernel_manager.compute_beta(
            r1, r2,
            particle1_idx=i,
            particle2_idx=j,
            solver=self
        )

    # ------------------------------------------------------------------
    # r_agg maintenance (full rebuild)
    # ------------------------------------------------------------------
    def _rebuild_all_propensities(self):
        """weighted agglomeration propensities with shared packet correction."""
        a = self.a_tot
        if a <= 0:
            if not hasattr(self, "_r_agg") or self._r_agg is None or self._r_agg.shape[0] < getattr(self, "_cap", a):
                self._r_agg = np.zeros(getattr(self, "_cap", max(8, a)), dtype=float)
            else:
                self._r_agg[:] = 0.0
            if hasattr(self, "_delta_agg") and self._delta_agg is not None:
                self._delta_agg[:] = 0.0
            return

        R = (self.X[:a] * 0.5).astype(np.float64)
        W = self.W[:a].astype(np.float64)
        dW_const = float(getattr(self, "_agg_dW_const", self._prepare_agg_delta_config()))
        delta = self._delta_from_weights(W, dW_const=dW_const)
        if (not hasattr(self, "_delta_agg")) or self._delta_agg is None or self._delta_agg.shape[0] < getattr(self, "_cap", a):
            self._delta_agg = np.zeros(getattr(self, "_cap", a), dtype=float)
        self._delta_agg[:a] = delta
        
        # Use kernel framework if available, otherwise legacy JIT
        if hasattr(self, 'kernel_manager') and self.kernel_manager is not None and self.kernel_manager.agg_kernel is not None:
            # Kernel framework: compute propensities via kernel
            # IMPORTANT: Must include self-agglomeration (i==j) for correctness!
            # For i==j: contribution is (W[i]-1)*beta_ii (not W[i]*beta_ii)
            # because a particle cannot collide with itself.
            
            kernel_name = getattr(self.kernel_manager.agg_kernel, 'name', '')
            
            # Use JIT batch functions for standard kernels (shear, brownian, constant, sum)
            # These are much faster than Python loops for large particle counts
            agg_kernel = self.kernel_manager.agg_kernel
            
            if kernel_name == 'shear_chin1998':
                r = np.zeros(a, dtype=float)
                rebuild_r_array_shear(
                    float(agg_kernel.corr_beta),
                    float(agg_kernel.g),
                    R, W, r
                )
            elif kernel_name == 'brownian_tsouris1995':
                r = np.zeros(a, dtype=float)
                rebuild_r_array_brownian(
                    float(agg_kernel.corr_beta),
                    float(agg_kernel.kT),
                    float(agg_kernel.viscosity),
                    R, W, r
                )
            elif kernel_name == 'constant':
                r = np.zeros(a, dtype=float)
                rebuild_r_array_constant(
                    float(agg_kernel.corr_beta),
                    W, r
                )
            elif kernel_name == 'sum':
                r = np.zeros(a, dtype=float)
                rebuild_r_array_sum(
                    float(agg_kernel.corr_beta),
                    R, W, r
                )
            else:
                # Fallback for custom kernels (e.g., liquid_bridge)
                # Uses Python loop - slower but flexible
                r = np.zeros(a, dtype=float)
                for i_idx in range(a):
                    ri = 0.0
                    r1 = float(R[i_idx])
                    for j_idx in range(a):
                        r2 = float(R[j_idx])
                        beta_ij = self.kernel_manager.compute_beta(
                            r1, r2,
                            particle1_idx=i_idx,
                            particle2_idx=j_idx,
                            solver=self
                        )
                        if i_idx == j_idx:
                            # Self-agglomeration: (W[i] - 1) distinct pairs
                            ri += max(0.0, float(W[j_idx] - 1)) * beta_ij
                        else:
                            # Distinct particles: W[j] * beta_ij
                            ri += float(W[j_idx]) * beta_ij
                    r[i_idx] = ri
        else:
            # No kernel configured - should not happen
            raise RuntimeError("Aggregation kernel not initialized")
        r = np.divide(r, delta, out=np.zeros_like(r, dtype=float), where=delta > 0.0)
        np.maximum(r, 0.0, out=r)

        if not hasattr(self, "_r_agg") or self._r_agg is None or self._r_agg.shape[0] < getattr(self, "_cap", a):
            self._r_agg = np.zeros(getattr(self, "_cap", a), dtype=float)
        self._r_agg[:a] = r
        if self._r_agg.shape[0] > a:
            self._r_agg[a:] = 0.0
        if self._delta_agg.shape[0] > a:
            self._delta_agg[a:] = 0.0

    def _compute_agg_dW(self, i: int, j: int, pair_prop: float, sum_prop_before: float) -> float:
        """Compute packet size Î”W for one agglomeration event on pair (i,j)."""
        Wi = float(self.W[i])
        Wj = float(self.W[j])
        if Wi <= 0.0 or Wj <= 0.0:
            return 0.0

        dW_max = float(getattr(self, "agg_dW_max", 1.0))
        dW_min = float(getattr(self, "agg_dW_min", 1.0))
        if dW_max <= 0.0:
            return 0.0
        if dW_min < 0.0:
            dW_min = 0.0

        mode = str(getattr(self, "agg_dW_mode", "const")).lower()
        if mode == "const":
            dW = dW_max
        else:
            f = 0.0 if sum_prop_before <= 0.0 else float(pair_prop) / float(sum_prop_before)
            f = float(np.clip(f, 0.0, 1.0))
            alpha = float(getattr(self, "agg_dW_alpha", 100.0))
            if alpha <= 0.0:
                alpha = 1.0
            if mode == "sqrt":
                dW = dW_min + alpha * (f ** 0.5) * (dW_max - dW_min)
            else:
                dW = dW_min + alpha * f * (dW_max - dW_min)

        if dW < dW_min:
            dW = dW_min
        if dW > dW_max:
            dW = dW_max
        delta_i = self._update_delta_single(i, attr_name="_delta_agg", dW_const=float(getattr(self, "_agg_dW_const", dW_max)))
        delta_j = self._update_delta_single(j, attr_name="_delta_agg", dW_const=float(getattr(self, "_agg_dW_const", dW_max)))
        if i == j:
            if delta_i <= 0.0 or Wi <= 2.0 * delta_i:
                return 0.0
            if dW > delta_i:
                dW = delta_i
        else:
            if dW > Wi:
                dW = Wi
            if dW > Wj:
                dW = Wj
            if delta_i > 0.0 and dW > delta_i:
                dW = delta_i
            if delta_j > 0.0 and dW > delta_j:
                dW = delta_j
        if not np.isfinite(dW) or dW <= 0.0:
            return 0.0
        return float(dW)

    # ------------------------------------------------------------------
    # Single agglomeration event
    # ------------------------------------------------------------------
    def _do_one_agg(self):
        a = self.a_tot
        if a < 2:
            self._last_agg_dW = 0.0
            return

        # default packet for rejected/empty attempts
        self._last_agg_dW = 0.0

        # 1) pick first partner by r_i
        i = self._agg_sampler.sample(self._rng)

        # 2) weighted partner sampling + acceptance
        R = (self.X[:a] * 0.5).astype(np.float64)
        W = self.W[:a].astype(np.float64)
        if self.dim == 1:
            alpha1d = float(self.alpha_prim if np.ndim(self.alpha_prim) == 0 else np.mean(self.alpha_prim))
            alpha4 = np.zeros(4, dtype=np.float64)
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = np.zeros_like(V0)
        else:
            alpha1d = 1.0
            ap = np.asarray(self.alpha_prim, dtype=np.float64)
            alpha4 = ap if ap.size == 4 else np.ones(4, dtype=np.float64)
            V0 = self.V_flat[0, :a].astype(np.float64)
            V1 = self.V_flat[1, :a].astype(np.float64)

        SIZEEVAL = int(getattr(self, "SIZEEVAL", 1))
        X_SEL = float(getattr(self, "X_SEL", 0.31))
        Y_SEL = float(getattr(self, "Y_SEL", 1.06))
        Vmean2 = float(np.mean(self.V0[-1, :]) ** 2) if hasattr(self, "V0") else float(np.mean(self.V_flat[-1, :a]) ** 2)

        u_sel = float(self._rng.random())
        
        # Use kernel framework for partner selection if available (ensures consistency with _rebuild_all_propensities)
        use_kernel = hasattr(self, 'kernel_manager') and self.kernel_manager is not None and self.kernel_manager.agg_kernel is not None
        
        if use_kernel:
            # Check if we can use JIT-compiled path for constant kernel
            kernel_name = getattr(self.kernel_manager.agg_kernel, 'name', '')
            corr_beta = getattr(self.kernel_manager.agg_kernel, 'corr_beta', 1.0)
            
            if kernel_name == 'constant':
                # FAST PATH: JIT-compiled partner selection for constant kernel
                j, pick_w = pick_partner_constant_jit(
                    i, W, R, V0, V1, int(self.dim),
                    float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2,
                    u_sel, float(corr_beta),
                )
            else:
                # Python path for other kernels (shear, brownian, sum, custom)
                j, pick_w = self._pick_partner_kernel(
                    i, R, W, self._delta_agg[:a].astype(np.float64), 
                    V0, V1, int(self.dim),
                    float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2,
                    u_sel=u_sel,
                )
        else:
            # Legacy JIT path (unchanged)
            j, pick_w = nb_pick_partner_weighted(
                i,
                int(self.COLEVAL), float(self.CORR_BETA), float(getattr(self, "G", 1.0)),
                R, W, self._delta_agg[:a].astype(np.float64), V0, V1, int(self.dim),
                float(alpha1d), alpha4, SIZEEVAL, X_SEL, Y_SEL, Vmean2,
                u_sel, 0.0,  # u_acc=0 for legacy (not used)
            )
        
        if j < 0 or pick_w <= 0.0:
            return
        
        # NEW: Agglomeration acceptance kernel (Braumann et al. 2007 Stokes criterion, etc.)
        # Called AFTER partner selection, BEFORE agglomeration execution
        if hasattr(self, 'kernel_manager') and self.kernel_manager.agglomeration_acceptance_kernel is not None:
            r_i = float(R[i])
            r_j = float(R[j])
            v_dry_i = float(self.V_flat[-1, i])
            v_dry_j = float(self.V_flat[-1, j])
            
            accepted = self.kernel_manager.agglomeration_acceptance_kernel.accept_collision(
                r1=r_i, r2=r_j,
                v_dry1=v_dry_i, v_dry2=v_dry_j,
                particle1_idx=i, particle2_idx=j,
                solver=self
            )
            
            if not accepted:
                return  # Rejected by acceptance criterion

        # 3) packet size Î"W for this accepted event
        sum_prop_before = float(self._agg_sampler.total()) if self._agg_sampler is not None else float(np.sum(self._r_agg[: self.a_tot]))
        Wi = float(self.W[i])
        Wj = float(self.W[j])
        pair_prop = Wi * pick_w  # distinct pair: W_i*(W_j*beta_ij); self pair: W_i*((W_i-1)*beta_ii)
        dW = self._compute_agg_dW(i, j, pair_prop, sum_prop_before)
        if dW <= 0.0:
            return
        self._last_agg_dW = dW

        # 4) create one new compute particle for merged products with weight dW
        # ================================================================
        # IMPORTANT: V_flat[:dim] semantics for mass conservation
        # ================================================================
        # V_flat[:dim] MUST represent V_solid (solid volume only), NOT V_dry!
        # This ensures mass is conserved during agglomeration.
        #
        # For dim=1:
        #   V_flat[0, :] = V_solid = V_dry × (1 - porosity)
        #   V_flat[-1,:] = V_dry (total dry volume)
        #
        # When merging: V_solid_new = V_solid_i + V_solid_j (mass conserved!)
        #               V_dry_new = V_solid_new / (1 - porosity_new)
        # ================================================================
        
        # Get parent volumes
        Vi_comp = self.V_flat[: self.dim, i].copy()  # Should be V_solid
        Vj_comp = self.V_flat[: self.dim, j].copy()  # Should be V_solid
        Vi_dry = self.V_flat[-1, i]  # V_dry of parent i
        Vj_dry = self.V_flat[-1, j]  # V_dry of parent j
        
        # Calculate V_solid from parents (explicit, for clarity)
        # If porosity exists, use it; otherwise assume V_flat[:dim] is already V_solid
        if hasattr(self, "porosity"):
            poro_i = self.porosity[i]
            poro_j = self.porosity[j]
            
            if np.isnan(poro_i):
                # Vollkörper: V_solid = V_dry
                V_solid_i = Vi_dry
            else:
                # Porous: V_solid = V_dry × (1 - porosity)
                V_solid_i = Vi_dry * (1.0 - poro_i)
            
            if np.isnan(poro_j):
                V_solid_j = Vj_dry
            else:
                V_solid_j = Vj_dry * (1.0 - poro_j)
        else:
            # No porosity tracking - assume V_flat[:dim] is V_solid
            V_solid_i = np.sum(Vi_comp)
            V_solid_j = np.sum(Vj_comp)
        
        # Merged solid volume (MASS CONSERVED!)
        V_solid_merged = V_solid_i + V_solid_j
        
        # Liquid volume: add from both parents (mass conservation)
        lv_i = float(self.liquid_volume[i]) if hasattr(self, "liquid_volume") else 0.0
        lv_j = float(self.liquid_volume[j]) if hasattr(self, "liquid_volume") else 0.0

        # Create merged V_flat array
        # _append_particle_column expects only V_flat[:dim] (component volumes)
        # V_flat[-1] will be set separately after porosity merge
        Vnew_comp = Vi_comp + Vj_comp  # Component volumes (V_solid for dim=1)
        
        self._append_particle_column(Vnew_comp)
        new_idx = self.a_tot - 1
        self.W[new_idx] = dW
        
        # Note: liquid_volume assignment moved below after saturation calculation
        # to ensure consistent internal/external partitioning
        
        # ================================================================
        # Handle porosity merge during agglomeration
        # ================================================================
        # ALWAYS use PorosityGrowthKernel for consistent physics!
        # Default: volume_mixing kernel if not configured
        # ================================================================
        # After _append_particle_column:
        #   V_flat[:dim, new_idx] = Vnew_comp (sum of parent component volumes)
        #   V_flat[-1, new_idx] = sum(Vnew_comp) (temporary, will be overwritten)
        # We need to update V_flat[-1] to V_dry after calculating porosity
        # ================================================================
        if hasattr(self, "porosity"):
            poro_i = self.porosity[i]
            poro_j = self.porosity[j]
            
            # Get liquid volumes and saturations for kernel
            lv_i = float(self.liquid_volume[i]) if hasattr(self, 'liquid_volume') else 0.0
            lv_j = float(self.liquid_volume[j]) if hasattr(self, 'liquid_volume') else 0.0
            sat_i = self.saturation[i] if hasattr(self, 'saturation') and not np.isnan(poro_i) else 0.0
            sat_j = self.saturation[j] if hasattr(self, 'saturation') and not np.isnan(poro_j) else 0.0
            
            # Estimate collision energy (simplified)
            # E_coll ~ 0.5 × m × v², v ~ G × d
            rho = 1000.0  # kg/m³ (assume density)
            d_eff = float(self.X[i] + self.X[j]) * 0.5
            g = float(getattr(self, 'G', 1000.0))
            v_rel = g * d_eff
            v_particle = float(Vi_dry + Vj_dry)  # Use V_dry from parents
            m = rho * v_particle
            E_coll = 0.5 * m * v_rel ** 2
            
            # Get porosity growth kernel (create default if not exists)
            porosity_kernel = None
            if hasattr(self, 'kernel_manager') and self.kernel_manager is not None:
                porosity_kernel = self.kernel_manager.porosity_growth_kernel
            
            if porosity_kernel is None:
                # Create default volume_mixing kernel
                from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
                porosity_kernel = get_porosity_growth_kernel('volume_mixing')
            
            # Compute merged porosity via kernel
            V_dry_merged, poro_merged = porosity_kernel.compute_merged_porosity(
                v_dry1=float(Vi_dry), poro1=poro_i,
                v_dry2=float(Vj_dry), poro2=poro_j,
                v_liq1=lv_i, v_liq2=lv_j,
                sat1=sat_i, sat2=sat_j,
                collision_energy=E_coll,
                solver=self
            )
            self.porosity[new_idx] = poro_merged
        else:
            # No porosity tracking - use simple volume mixing
            V_dry_merged = V_solid_merged
        
        # ================================================================
        # CRITICAL: Update V_flat to ensure mass conservation
        # ================================================================
        # V_flat[:dim] MUST be V_solid (not V_dry!)
        # V_flat[-1] MUST be V_dry
        # This is essential for mass conservation!
        # ================================================================
        # Set V_flat[:dim] to V_solid (mass conservation!)
        self.V_flat[:self.dim, new_idx] = V_solid_merged
        # Set V_flat[-1] to V_dry
        self.V_flat[-1, new_idx] = V_dry_merged
        
        # Merge saturation and liquid volume
        # IMPORTANT: liquid_volume stores properties PER PHYSICAL PARTICLE (intensive).
        # When n_comp_i and n_comp_j agglomerate, the child gets the COMBINED properties
        # of both parents (direct addition), NOT scaled by dW/W.
        # The weight W only tracks how many physical particles are represented,
        # not the magnitude of intensive properties like volume or liquid content.
        if hasattr(self, "saturation") and hasattr(self, "liquid_volume"):
            poro_i = self.porosity[i] if hasattr(self, 'porosity') else np.nan
            poro_j = self.porosity[j] if hasattr(self, 'porosity') else np.nan
            
            # Get parent saturations (stored state variable)
            sat_i = self.saturation[i] if not np.isnan(poro_i) else 0.0
            sat_j = self.saturation[j] if not np.isnan(poro_j) else 0.0
            
            # Pore volumes (per computational particle)
            V_pore_i = Vi_dry * poro_i if not np.isnan(poro_i) else 0.0
            V_pore_j = Vj_dry * poro_j if not np.isnan(poro_j) else 0.0
            
            # Internal liquid from stored saturation: V_liq_int = V_pore × S
            V_liq_int_i_full = V_pore_i * sat_i if V_pore_i > 0 else 0.0
            V_liq_int_j_full = V_pore_j * sat_j if V_pore_j > 0 else 0.0
            
            # Total liquid volumes (per computational particle)
            liq_i_full = float(self.liquid_volume[i]) if hasattr(self, 'liquid_volume') else 0.0
            liq_j_full = float(self.liquid_volume[j]) if hasattr(self, 'liquid_volume') else 0.0
            
            # External liquid: total - internal
            V_liq_ext_i_full = liq_i_full - V_liq_int_i_full
            V_liq_ext_j_full = liq_j_full - V_liq_int_j_full
            
            # ================================================================
            # BRAUMANN et al. 2007: Liquid internalization during agglomeration
            # ================================================================
            # When two wetted particles contact, external liquid on surfaces
            # becomes trapped in newly formed contact pores.
            # 
            # FALLBACK: If kernel not configured, l_e_to_i = 0.0 (no internalization)
            # ================================================================
            l_e_to_i = 0.0
            if hasattr(self, 'kernel_manager') and self.kernel_manager is not None:
                l_e_to_i = self.kernel_manager.compute_liquid_internalization_agglomeration(
                    v_dry1=float(Vi_dry),
                    v_dry2=float(Vj_dry),
                    v_liq_ext1=float(V_liq_ext_i_full),
                    v_liq_ext2=float(V_liq_ext_j_full),
                    particle1_idx=i,
                    particle2_idx=j,
                    solver=self
                )
            
            # DIRECT ADDITION: Child gets combined properties of both parents
            # No dW/W scaling needed - properties are per physical particle
            if i == j:
                # Self-agglomeration: particle merges with itself
                # Properties double (two particles become one)
                V_liq_int_contrib = 2.0 * V_liq_int_i_full
                V_liq_ext_contrib = 2.0 * V_liq_ext_i_full
            else:
                # Distinct particles: sum of both
                V_liq_int_contrib = V_liq_int_i_full + V_liq_int_j_full
                V_liq_ext_contrib = V_liq_ext_i_full + V_liq_ext_j_full
            
            # Merged internal/external liquids WITH Braumann internalization
            # Internal: parent sum + internalized amount
            # External: parent sum - internalized amount
            V_liq_int_merged = V_liq_int_contrib + l_e_to_i
            V_liq_ext_merged = V_liq_ext_contrib - l_e_to_i
            
            # Physical constraint: Cannot have negative external liquid
            if V_liq_ext_merged < 0.0:
                # Cap internalization at available external liquid
                l_e_to_i = V_liq_ext_contrib
                V_liq_int_merged = V_liq_int_contrib + l_e_to_i
                V_liq_ext_merged = 0.0
            
            # New pore volume (use V_dry_merged from porosity calculation above)
            V_pore_merged = V_dry_merged * self.porosity[new_idx] if not np.isnan(self.porosity[new_idx]) else 0.0
            
            # New saturation: S_merged = V_liq_int_merged / V_pore_merged
            if V_pore_merged > 0:
                sat_merged = V_liq_int_merged / V_pore_merged
            else:
                sat_merged = 0.0  # Vollkörper has no pores
            
            # If saturation exceeds 1.0, externalize the excess
            if sat_merged > 1.0:
                excess_internal = V_pore_merged * (sat_merged - 1.0)
                V_liq_int_merged = V_pore_merged  # Cap at full saturation
                V_liq_ext_merged += excess_internal  # Add excess to external
                sat_merged = 1.0  # Cap S at 1.0
            
            # Total merged liquid
            liq_merged = V_liq_int_merged + V_liq_ext_merged
            
            # Assign to new particle
            self.saturation[new_idx] = sat_merged
            self.liquid_volume[new_idx] = liq_merged

        # 5) consume dW from parents (self-agglomeration consumes 2*dW from one packet)
        # IMPORTANT: liquid_volume is PER PHYSICAL PARTICLE (intensive property).
        # When reducing parent weight W, the liquid_volume STAYS THE SAME because
        # remaining computational particles still represent physical particles with
        # the same properties. Only the NUMBER of represented particles changes.
        if i == j:
            if i < self.a_tot:
                w_now = float(self.W[i])
                w_rem = w_now - 2.0 * dW
                if w_rem > 0.0:
                    self.W[i] = w_rem
                    # DO NOT scale liquid_volume - it's per physical particle!
                    # Remaining particles have same liquid content per particle.
                else:
                    self._remove_particle_column(i)
        else:
            for idx in sorted({int(i), int(j)}, reverse=True):
                if idx >= self.a_tot:
                    continue
                w_now = float(self.W[idx])
                w_rem = w_now - dW
                if w_rem > 0.0:
                    self.W[idx] = w_rem
                    # DO NOT scale liquid_volume - it's per physical particle!
                    # Remaining particles have same liquid content per particle.
                else:
                    self._remove_particle_column(idx)

        # 6) full weighted agglomeration propensity refresh (simple and consistent)
        self._rebuild_all_propensities()
        self._agg_sampler = FenwickSampler(self._r_agg[: self.a_tot])

        # 7) breakage sampler refresh (if active)
        if getattr(self, "process_break", False) or getattr(self, "process_type", "agglomeration") in ("breakage", "mix"):
            self._calc_break_rates_full()  # provided by BreakageMixin
            self._break_sampler = FenwickSampler(self._break_rate[:self.a_tot])

