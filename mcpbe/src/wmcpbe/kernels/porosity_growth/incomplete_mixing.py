"""
Incomplete Mixing Porosity Kernel.

Models trapped pores from incomplete coalescence during agglomeration.

When two particles collide, they may not fully coalesce. Instead:
1. Some pore space gets trapped at the contact region
2. Surface roughness prevents complete merging
3. Binder viscosity limits restructuring time

This creates ADDITIONAL porosity beyond simple volume addition.

Formula:
    V_pore_merged = V_pore_1 + V_pore_2 + V_pore_trapped
    
    V_pore_trapped = f_trap × min(V_dry_1, V_dry_2)
    
    where f_trap is the trapped pore fraction (typically 0.05-0.3)

Physical Background:
    - Relevant for wet granulation with viscous binders
    - Depends on collision energy, binder viscosity, contact time
    - Higher f_trap → more porous granules

References:
    - Iveson et al., "Non-inertial droplet-particle collisions", 2001
    - Liu et al., "Wet granulation: mechanism and modeling", 2009
"""

import numpy as np
from ..base import PorosityGrowthKernel


class IncompleteMixingKernel(PorosityGrowthKernel):
    """
    Porosity growth kernel with trapped pores from incomplete coalescence.
    
    This kernel extends simple volume mixing by accounting for additional
    porosity created when particles don't fully merge. The trapped pore
    volume is proportional to the smaller particle's size.
    
    Physical mechanisms captured:
    1. **Contact porosity**: Pores trapped at particle-particle interface
    2. **Surface roughness**: Micro-scale voids from non-smooth surfaces
    3. **Viscous limitation**: Insufficient time for full restructuring
    
    Parameters:
        trapped_pore_fraction: Fraction of smaller particle volume that
                               becomes trapped pores (default: 0.1)
                               Typical range: 0.05 - 0.3
    
    Example:
        >>> kernel = IncompleteMixingKernel(trapped_pore_fraction=0.15)
        >>> v_dry, poro = kernel.compute_merged_porosity(
        ...     v_dry1=1e-18, poro1=0.4,
        ...     v_dry2=2e-18, poro2=0.5
        ... )
        >>> # poro will be higher than volume_mixing due to trapped pores
    """
    
    @property
    def name(self) -> str:
        return 'incomplete_mixing'
    
    def get_default_params(self) -> dict:
        return {
            'trapped_pore_fraction': 0.1,
            'nucleation_porosity': 0.0,  # Start with Vollkörper (NaN), porosity grows via agglomeration
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache values
        self.f_trap = float(self.params['trapped_pore_fraction'])
        self.nucleation_porosity = float(self.params['nucleation_porosity'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if not (0 <= params['trapped_pore_fraction'] <= 1):
            raise ValueError("trapped_pore_fraction must be in [0, 1]")
        if not (0 <= params['nucleation_porosity'] <= 0.999):
            raise ValueError("nucleation_porosity must be in [0, 0.999]")
        return params
    
    def compute_merged_porosity(self,
                                 v_dry1: float, poro1: float,
                                 v_dry2: float, poro2: float,
                                 v_liq1: float = None,
                                 v_liq2: float = None,
                                 sat1: float = None,
                                 sat2: float = None,
                                 solver = None
                                 ) -> tuple[float, float]:
        """
        Compute merged porosity with trapped pores.
        
        Args:
            v_dry1: Dry volume of particle 1 [m³]
            poro1: Porosity of particle 1 (NaN for Vollkörper)
            v_dry2: Dry volume of particle 2 [m³]
            poro2: Porosity of particle 2 (NaN for Vollkörper)
            v_liq1, v_liq2: Liquid volumes (optional, affects f_trap)
            sat1, sat2: Saturations (optional, affects coalescence)
            solver: Not used
        
        Returns:
            Tuple of (v_dry_merged, poro_merged)
            
        Note:
            Trapped pores are only created when at least one parent
            is already porous. Vollkörper + Vollkörper → Vollkörper.
        """
        # ==========================================
        # Decompose into solid and pore volumes
        # ==========================================
        
        # Particle 1
        if np.isnan(poro1):
            v_solid1 = v_dry1
            v_pore1 = 0.0
            is_vollkoerper_1 = True
        else:
            v_solid1 = v_dry1 * (1.0 - poro1)
            v_pore1 = v_dry1 * poro1
            is_vollkoerper_1 = False
        
        # Particle 2
        if np.isnan(poro2):
            v_solid2 = v_dry2
            v_pore2 = 0.0
            is_vollkoerper_2 = True
        else:
            v_solid2 = v_dry2 * (1.0 - poro2)
            v_pore2 = v_dry2 * poro2
            is_vollkoerper_2 = False
        
        # ==========================================
        # Base volumes (same as volume_mixing)
        # ==========================================
        v_solid_merged = v_solid1 + v_solid2
        v_pore_base = v_pore1 + v_pore2
        
        # ==========================================
        # Trapped pore formation
        # ==========================================
        v_pore_trapped = 0.0
        
        # Only create trapped pores if at least one particle is porous
        if not (is_vollkoerper_1 and is_vollkoerper_2):
            # Effective trapped pore fraction
            f_trap_eff = self.f_trap

            # Liquid-enhanced coalescence (more liquid → better merging)
            if v_liq1 is not None and v_liq2 is not None:
                v_liq_total = v_liq1 + v_liq2
                v_ref = (v_dry1 + v_dry2) * 0.1  # 10% liquid reference
                liq_factor = 1.0 / (1.0 + v_liq_total / max(v_ref, 1e-30))
                f_trap_eff *= liq_factor
            
            # Saturation effect (optimal saturation promotes coalescence)
            if sat1 is not None and sat2 is not None and not np.isnan(sat1) and not np.isnan(sat2):
                s_avg = (sat1 + sat2) / 2.0
                # High saturation (>80%) reduces trapped pores (better wetting)
                if s_avg > 0.8:
                    wetting_factor = 1.0 - (s_avg - 0.8)
                    f_trap_eff *= max(0.5, wetting_factor)
            
            # Trapped pore volume proportional to smaller particle
            v_smaller = min(v_dry1, v_dry2)
            v_pore_trapped = f_trap_eff * v_smaller
        
        # ==========================================
        # Total merged volumes
        # ==========================================
        v_pore_merged = v_pore_base + v_pore_trapped
        v_dry_merged = v_solid_merged + v_pore_merged
        
        # ==========================================
        # Compute merged porosity
        # ==========================================
        if v_dry_merged <= 0:
            return 0.0, np.nan
        
        if v_pore_merged <= 0:
            return v_dry_merged, np.nan
        
        poro_merged = v_pore_merged / v_dry_merged
        
        # Clamp to valid range
        poro_merged = max(0.0, min(1.0, poro_merged))
        
        return v_dry_merged, poro_merged
    
    def compute_nucleation_porosity(self,
                                     v_solid: float,
                                     v_liquid: float,
                                     nucleation_params: dict = None,
                                     solver = None
                                     ) -> tuple[float, float]:
        """
        Compute porosity for newly nucleated particle.
        
        For incomplete mixing model, nucleation starts with Vollkörper (poro=NaN).
        Porosity is CREATED during agglomeration via trapped_pore mechanism.
        
        IMPORTANT: Unlike volume_mixing, this kernel does NOT hardcode porosity!
        It starts with Vollkörper and porosity grows naturally through agglomeration.
        
        Args:
            v_solid: Solid volume of nucleus [m³]
            v_liquid: Liquid volume associated with nucleus [m³]
            nucleation_params: Additional parameters
                               - 'nucleation_porosity': Initial porosity (optional, default: 0.0=Vollkörper)
            solver: Not used
        
        Returns:
            Tuple of (v_dry, porosity)
            - Returns (v_solid, NaN) for Vollkörper when nucleation_porosity <= 0
        """
        if nucleation_params is None:
            nucleation_params = {}
        
        # Priority: 1) nucleation_params, 2) self.params (config), 3) 0.0 (Vollkörper!)
        initial_poro = 0.0  # DEFAULT: Start with Vollkörper
        
        if 'nucleation_porosity' in nucleation_params:
            initial_poro = float(nucleation_params['nucleation_porosity'])
        elif 'nucleation_porosity' in self.params:
            initial_poro = float(self.params['nucleation_porosity'])
        
        # If explicitly set to 0 or negative → Vollkörper
        if initial_poro <= 0:
            return float(v_solid), np.nan  # Vollkörper ✅
        
        # Clamp to valid range
        initial_poro = max(0.0, min(0.999, initial_poro))
        
        # Nucleus with initial porosity
        v_dry = float(v_solid) / (1.0 - initial_poro)
        
        return v_dry, float(initial_poro)
    
    def compute_fragment_porosity(self,
                                   parent_porosity: float,
                                   fragment_volume: float,
                                   parent_volume: float,
                                   solver = None
                                   ) -> float:
        """
        Compute porosity for fragment after breakage.
        
        Physical assumption for incomplete mixing model:
        Breakage causes PARTIAL PORE COLLAPSE due to mechanical stress.
        Fragments are denser than parent (lower porosity).
        
        Formula:
            poro_frag = poro_parent × (1 - collapse_factor)
            
            where collapse_factor depends on breakage energy:
            - Higher energy → more pore collapse
            - Vollkörper remains Vollkörper (NaN stays NaN)
        
        Args:
            parent_porosity: Parent particle porosity (NaN for Vollkörper)
            fragment_volume: Fragment solid volume [m³] (not used)
            parent_volume: Parent solid volume [m³] (not used)
            solver: Not used
        
        Returns:
            fragment_porosity: Reduced porosity (pore collapse)
                              or NaN if parent was Vollkörper
        """
        # Vollkörper stays Vollkörper
        if np.isnan(parent_porosity):
            return np.nan
        
        # Base collapse factor (10% pore reduction by default)
        collapse_factor = 0.1

        # Clamp collapse factor
        collapse_factor = max(0.0, min(0.5, collapse_factor))  # Max 50% collapse
        
        # Apply pore collapse
        frag_porosity = parent_porosity * (1.0 - collapse_factor)
        
        return frag_porosity
