"""
Liquid Internalization/Externalization Kernel (Agglomeration ↔ Breakage).

Model for liquid moving between the internal (pore) and external (surface)
state as particles merge or break, symmetric to how the cone_model
porosity kernel handles pore-volume growth (agglomeration) and pore-volume
loss (breakage) in a single kernel.

Internalization formula (Braumann et al. 2007, Eq. for l_{e→i}), applied
PER AGGLOMERATION EVENT:
    l_{e→i} = { l_{e,j} × l_{e,k} ×
                [1 - √(1 - ((∛(v_j - l_{e,j})) / (∛v_j + ∛v_k))²)] ×
                [1 - √(1 - ((∛(v_k - l_{e,k})) / (∛v_j + ∛v_k))²)]
              }^(1/2)

where:
    - v_j = v_dry,j + v_liq_ext,j (hydrodynamic volume of wetted particle)
    - v_k = v_dry,k + v_liq_ext,k
    - l_{e,j}, l_{e,k}: External liquid volumes of parents

Externalization formula (reverse process), applied PER BREAKAGE EVENT:
    ΔV_pore = v_pore_parent - v_pore_fragments_total
    V_liq_int→ext = saturation_parent × ΔV_pore

Physical Background:
    Internalization: When two particles with wetted surfaces contact
    during agglomeration, a pore forms at the contact point. The external
    liquid that was on the previously wetted surfaces becomes trapped in
    this newly formed pore and is internalized. The formula is derived
    from geometric considerations of two spheres with liquid films
    contacting each other. The internalized amount depends on:
    1. Available external liquid on both particles
    2. Relative particle sizes (hydrodynamic volumes)
    3. Contact geometry

    Externalization: When a particle fragments, new fracture surfaces
    destroy part of the pore volume (see the porosity growth kernel, e.g.
    cone_model's ΔV term). The internal liquid that occupied that lost
    pore volume can no longer be held internally and is pushed out to
    become external liquid on the fragments' surfaces, proportional to the
    parent's saturation at the moment of breakage.

Usage:
    >>> kernel = LiquidInternalisationAgglomerationKernel()
    >>> l_e_to_i = kernel.compute_internalization(
    ...     v_dry1=1e-18, v_dry2=2e-18,
    ...     v_liq_ext1=1e-19, v_liq_ext2=2e-19
    ... )
    >>> print(f"Internalized liquid: {l_e_to_i:.3e} m³")
    >>> l_i_to_e = kernel.compute_externalization(
    ...     v_pore_parent=1e-18, v_pore_fragments_total=8e-19,
    ...     saturation_parent=0.5
    ... )
    >>> print(f"Externalized liquid: {l_i_to_e:.3e} m³")

References:
    [1] Braumann et al., "Modelling and validation of granulation with
        heterogeneous binder dispersion and chemical reaction",
        Chemical Engineering Science, 2007.
"""

import numpy as np
from typing import Optional, Any
from ..base import LiquidInternalizationAgglomerationKernel


class LiquidInternalisationAgglomerationKernel(LiquidInternalizationAgglomerationKernel):
    """
    Braumann et al. 2007 liquid internalization/externalization model.

    Computes the amount of external liquid that becomes trapped in
    contact pores when two wetted particles merge (`compute_internalization`,
    agglomeration), and the reverse process of internal liquid being pushed
    out when a particle fragments and pore volume is lost to new fracture
    surfaces (`compute_externalization`, breakage).

    Parameters:
        None (this kernel has no tunable parameters - pure physics model)
    
    Example:
        >>> kernel = LiquidInternalisationAgglomerationKernel()
        >>> # Two particles with external liquid films
        >>> l_e_to_i = kernel.compute_internalization(
        ...     v_dry1=1e-18, v_dry2=2e-18,
        ...     v_liq_ext1=1e-19, v_liq_ext2=2e-19
        ... )
        >>> print(f"Internalized: {l_e_to_i:.3e} m³")
    """
    
    @property
    def name(self) -> str:
        return 'liq_internalisation_agglomeration'
    
    def get_default_params(self) -> dict:
        return {}
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
    
    def validate_params(self, params: dict) -> dict:
        """No parameters to validate."""
        return params
    
    def compute_internalization(
        self,
        v_dry1: float, v_dry2: float,
        v_liq_ext1: float, v_liq_ext2: float,
        particle1_idx: Optional[int] = None,
        particle2_idx: Optional[int] = None,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute liquid internalization per Braumann et al. 2007.
        
        Args:
            v_dry1: Dry volume of particle 1 [m³]
            v_dry2: Dry volume of particle 2 [m³]
            v_liq_ext1: External liquid volume of particle 1 [m³]
            v_liq_ext2: External liquid volume of particle 2 [m³]
            particle1_idx: Not used (kept for API consistency)
            particle2_idx: Not used (kept for API consistency)
            solver: Not used (kept for API consistency)
        
        Returns:
            l_e_to_i: Amount of liquid internalized [m³]
                      (0.0 if no external liquid available)
        
        Note:
            IMPORTANT: v_j and v_k are HYDRODYNAMIC volumes:
            v_j = v_dry,j + v_liq_ext,j (NOT just v_dry!)
            
            This accounts for the liquid film on the particle surface
            when computing the contact geometry.
        """
        # ==========================================
        # Edge cases: No external liquid → No internalization
        # ==========================================
        if v_liq_ext1 <= 0.0 or v_liq_ext2 <= 0.0:
            return 0.0
        
        if not (np.isfinite(v_dry1) and np.isfinite(v_dry2) and
                np.isfinite(v_liq_ext1) and np.isfinite(v_liq_ext2)):
            return 0.0
        
        # ==========================================
        # Step 1: Compute hydrodynamic volumes
        # ==========================================
        # CRITICAL: v_j = v_dry + v_liq_ext (NOT just v_dry!)
        # This accounts for the liquid film thickness
        v_j = v_dry1 + v_liq_ext1  # Hydrodynamic volume of particle 1
        v_k = v_dry2 + v_liq_ext2  # Hydrodynamic volume of particle 2
        
        # ==========================================
        # Step 2: Compute cube roots (for geometric terms)
        # ==========================================
        # Cube root of volumes (proportional to particle radii)
        v_j_cbrt = np.cbrt(v_j)
        v_k_cbrt = np.cbrt(v_k)
        
        # Sum of radii (denominator for both terms)
        sum_radii = v_j_cbrt + v_k_cbrt
        
        if sum_radii <= 0.0:
            return 0.0
        
        # ==========================================
        # Step 3: Compute solid core volumes
        # ==========================================
        # v_solid = v_hydro - v_liq_ext = v_dry (since v_hydro = v_dry + v_liq_ext)
        # But we use: v_j - v_liq_ext1 = v_dry1 (solid core without liquid film)
        v_solid_j = v_j - v_liq_ext1  # = v_dry1
        v_solid_k = v_k - v_liq_ext2  # = v_dry2
        
        # ==========================================
        # Step 4: Compute Braumann formula terms
        # ==========================================
        # Term 1: [1 - √(1 - ((∛(v_j - l_{e,j})) / (∛v_j + ∛v_k))²)]
        # This represents the fraction of particle j's surface that contacts
        
        ratio_j = np.cbrt(v_solid_j) / sum_radii
        ratio_k = np.cbrt(v_solid_k) / sum_radii
        
        # Clamp ratios to [0, 1] for numerical stability
        ratio_j = max(0.0, min(1.0, ratio_j))
        ratio_k = max(0.0, min(1.0, ratio_k))
        
        # Compute square terms
        ratio_j_sq = ratio_j ** 2
        ratio_k_sq = ratio_k ** 2
        
        # Compute sqrt terms: √(1 - ratio²)
        # Clamp argument to [0, 1] to avoid NaN from negative values
        sqrt_arg_j = max(0.0, min(1.0, 1.0 - ratio_j_sq))
        sqrt_arg_k = max(0.0, min(1.0, 1.0 - ratio_k_sq))
        
        sqrt_j = np.sqrt(sqrt_arg_j)
        sqrt_k = np.sqrt(sqrt_arg_k)
        
        # Compute bracket terms
        bracket_j = 1.0 - sqrt_j
        bracket_k = 1.0 - sqrt_k
        
        # ==========================================
        # Step 5: Combine all terms
        # ==========================================
        # l_{e→i} = √(l_{e,j} × l_{e,k} × bracket_j × bracket_k)
        
        product = v_liq_ext1 * v_liq_ext2 * bracket_j * bracket_k
        
        if product <= 0.0:
            return 0.0
        
        l_e_to_i = np.sqrt(product)
        
        # ==========================================
        # Step 6: Physical constraints
        # ==========================================
        # Cannot internalize more than available external liquid
        max_internalizable = v_liq_ext1 + v_liq_ext2
        
        if l_e_to_i > max_internalizable:
            l_e_to_i = max_internalizable
        
        # Ensure non-negative and finite
        if not np.isfinite(l_e_to_i) or l_e_to_i < 0.0:
            return 0.0

        return float(l_e_to_i)

    def compute_externalization(
        self,
        v_pore_parent: float,
        v_pore_fragments_total: float,
        saturation_parent: float,
        particle_idx: Optional[int] = None,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute liquid externalization during breakage (reverse of Braumann).

        Formula:
            ΔV_pore = v_pore_parent - v_pore_fragments_total
            V_liq_int→ext = saturation_parent × ΔV_pore

        Physical background: new fracture surfaces destroy part of the
        parent's pore volume (see porosity growth kernel, e.g. cone_model's
        ΔV term). The internal liquid that occupied that lost pore volume
        can no longer be held internally and is pushed out to become
        external liquid on the fragments' surfaces.

        Args:
            v_pore_parent: Pore volume of the parent particle before
                breakage [m³]
            v_pore_fragments_total: Summed pore volume of all fragments
                after breakage [m³]
            saturation_parent: Saturation of the parent particle before
                breakage (S = V_liq_int / V_pore, in [0, 1])
            particle_idx: Not used (kept for API consistency)
            solver: Not used (kept for API consistency)

        Returns:
            V_liq_int_to_ext: Amount of liquid externalized [m³]
                               (0.0 if pore volume did not shrink, or if
                               inputs are invalid/non-finite)
        """
        # ==========================================
        # Edge cases: invalid or non-finite inputs
        # ==========================================
        if not (np.isfinite(v_pore_parent) and np.isfinite(v_pore_fragments_total)
                and np.isfinite(saturation_parent)):
            return 0.0

        if v_pore_parent <= 0.0:
            return 0.0

        saturation_parent = max(0.0, min(1.0, saturation_parent))

        # ==========================================
        # ΔV_pore: pore volume destroyed by new fracture surfaces
        # ==========================================
        delta_v_pore = v_pore_parent - v_pore_fragments_total

        if delta_v_pore <= 0.0:
            # Pore volume did not shrink (e.g. volume_mixing porosity
            # kernel, which conserves pore volume additively) -> nothing
            # to externalize.
            return 0.0

        v_liq_int_to_ext = saturation_parent * delta_v_pore

        # ==========================================
        # Physical constraint: cannot externalize more internal liquid
        # than the parent actually held
        # ==========================================
        v_liq_int_parent = saturation_parent * v_pore_parent
        if v_liq_int_to_ext > v_liq_int_parent:
            v_liq_int_to_ext = v_liq_int_parent

        if not np.isfinite(v_liq_int_to_ext) or v_liq_int_to_ext < 0.0:
            return 0.0

        return float(v_liq_int_to_ext)
