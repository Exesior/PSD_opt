"""
Volume Mixing Porosity Kernel.

Simple porosity model: pores and solid volumes are additive.

This is the existing behavior from the original WMCPBE implementation.
When two particles agglomerate:
1. Solid volumes add: V_solid_merged = V_solid_1 + V_solid_2
2. Pore volumes add: V_pore_merged = V_pore_1 + V_pore_2
3. Dry volume: V_dry_merged = V_solid_merged + V_pore_merged
4. Porosity: poro_merged = V_pore_merged / V_dry_merged

Special cases:
- Vollkörper (NaN porosity): Treated as having zero pore volume
- If both parents are Vollkörper → child is Vollkörper
"""

import numpy as np
from ..base import PorosityGrowthKernel


class VolumeMixingKernel(PorosityGrowthKernel):
    """
    Simple volume-additive porosity mixing kernel.
    
    This kernel implements mass-conservative mixing of solid and pore volumes.
    It assumes that when two particles collide and merge, their internal
    structures (solid matrix and pores) simply add together.
    
    Physical interpretation:
    - No pore collapse during collision
    - No new pore formation beyond simple addition
    - Appropriate for gentle agglomeration or rigid particles
    
    Parameters:
        None (this kernel has no tunable parameters)
    
    Example:
        >>> kernel = VolumeMixingKernel()
        >>> v_dry, poro = kernel.compute_merged_porosity(
        ...     v_dry1=1e-18, poro1=0.4,
        ...     v_dry2=2e-18, poro2=0.5
        ... )
        >>> print(f"Merged: V_dry={v_dry:.3e}, porosity={poro:.3f}")
    """
    
    @property
    def name(self) -> str:
        return 'volume_mixing'
    
    def get_default_params(self) -> dict:
        return {}
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
    
    def validate_params(self, params: dict) -> dict:
        """No parameters to validate."""
        return params
    
    def compute_merged_porosity(self,
                                 v_dry1: float, poro1: float,
                                 v_dry2: float, poro2: float,
                                 v_liq1: float = None,
                                 v_liq2: float = None,
                                 sat1: float = None,
                                 sat2: float = None,
                                 collision_energy: float = None,
                                 solver = None
                                 ) -> tuple[float, float]:
        """
        Compute merged porosity by simple volume addition.
        
        Args:
            v_dry1: Dry volume of particle 1 [m³]
            poro1: Porosity of particle 1 (NaN for legacy Vollkörper, 0.0 for poreless)
            v_dry2: Dry volume of particle 2 [m³]
            poro2: Porosity of particle 2 (NaN for legacy Vollkörper, 0.0 for poreless)
            v_liq1, v_liq2: Liquid volumes (not used in this model)
            sat1, sat2: Saturations (not used in this model)
            collision_energy: Collision energy (not used in this model)
            solver: Not used
        
        Returns:
            Tuple of (v_dry_merged, poro_merged)
            - v_dry_merged: Sum of dry volumes
            - poro_merged: Merged porosity (0.0 if both parents poreless/Vollkörper)
            
        Note:
            Modern behavior: Returns 0.0 instead of NaN for poreless particles.
            Legacy NaN is normalized to 0.0 for consistent handling.
        """
        # ==========================================
        # Decompose into solid and pore volumes
        # ==========================================
        # Normalize NaN to 0.0 (treat legacy Vollkörper as poreless)
        # Formula works for BOTH poro=0.0 AND poro>0.0!
        poro1_norm = 0.0 if np.isnan(poro1) else poro1
        poro2_norm = 0.0 if np.isnan(poro2) else poro2
        
        # For poro=0.0: V_solid = V_dry × 1.0 = V_dry ✓, V_pore = 0.0 ✓
        v_solid1 = v_dry1 * (1.0 - poro1_norm)
        v_pore1 = v_dry1 * poro1_norm
        
        v_solid2 = v_dry2 * (1.0 - poro2_norm)
        v_pore2 = v_dry2 * poro2_norm
        
        # ==========================================
        # Mass-conservative addition
        # ==========================================
        v_solid_merged = v_solid1 + v_solid2
        v_pore_merged = v_pore1 + v_pore2
        v_dry_merged = v_solid_merged + v_pore_merged
        
        # ==========================================
        # Compute merged porosity
        # ==========================================
        if v_dry_merged <= 0:
            return 0.0, 0.0
        
        # MODERN: Return 0.0 instead of NaN for poreless particles
        # Both parents poreless (or legacy Vollkörper) → child is poreless (poro=0.0)
        if v_pore_merged <= 0:
            return v_dry_merged, 0.0  # Was: np.nan
        
        poro_merged = v_pore_merged / v_dry_merged
        
        # Clamp to valid range (numerical safety)
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
        
        IMPORTANT: Returns default porosity of 0.4 to enable porosity formation!
        
        Why not NaN (Vollkörper)?
        - With volume_mixing, porosity is ONLY additive (V_pore_new = V_pore_1 + V_pore_2)
        - If we start with Vollkörper (poro=NaN), agglomeration would NEVER create porosity
        - Default porosity=0.4 allows porous children from non-porous parents
        
        This is physically motivated by:
        - Surface roughness creating inter-particle voids
        - Incomplete coalescence even in "simple" mixing
        - Experimental observation that granules are rarely fully dense
        
        Args:
            v_solid: Solid volume of nucleus [m³]
            v_liquid: Liquid volume associated with nucleus [m³]
            nucleation_params: Can override default_porosity (optional)
            solver: Not used
        
        Returns:
            Tuple of (v_dry, porosity)
            - v_dry = v_solid / (1 - porosity)
            - porosity = 0.4 (default, enables porosity formation!)
        """
        # Get default porosity from params or use 0.4
        default_porosity = 0.4
        if nucleation_params and 'default_porosity' in nucleation_params:
            default_porosity = float(nucleation_params['default_porosity'])
        
        # Clamp to valid range
        porosity = max(0.0, min(0.999, default_porosity))
        
        # Compute dry volume from solid volume and porosity
        v_dry = v_solid / (1.0 - porosity)
        
        return v_dry, porosity
    
    def compute_fragment_porosity(self,
                                   parent_porosity: float,
                                   fragment_volume: float,
                                   parent_volume: float,
                                   breakage_energy: float = None,
                                   solver = None
                                   ) -> float:
        """
        Compute porosity for fragment after breakage.
        
        Physical assumption: Fragments inherit parent porosity.
        Breakage splits both solid and pore volumes proportionally.
        
        Special case: If parent is poreless (poro=0.0) or legacy Vollkörper (NaN),
        fragment inherits this (no pores to distribute).
        
        Args:
            parent_porosity: Parent particle porosity (NaN for legacy Vollkörper,
                            0.0 for poreless, >0.0 for porous)
            fragment_volume: Fragment solid volume [m³] (not used, kept for API consistency)
            parent_volume: Parent solid volume [m³] (not used, kept for API consistency)
            breakage_energy: Not used in this model
            solver: Not used
        
        Returns:
            fragment_porosity: Same as parent_porosity (inherited)
        """
        # Inherit parent porosity (works for both NaN and valid values)
        return parent_porosity
