"""
Liquid Bridge Agglomeration Kernel.

Extends shear-induced agglomeration with enhancement from capillary forces
due to liquid bridges between wet particles.

Formula:
    β = β_shear × f_bridge × f_capillary

Where:
    - β_shear = CORR_BETA × G × (r1 + r2)³  (base shear collision)
    - f_bridge = exp(-((s_eff - s_opt)²) / (2σ_s²))  (optimal saturation)
    - f_capillary = 1 + α × log(1 + V_liq_ext / V_ref)  (external liquid)

Physical Background:
    Liquid bridges between particles create capillary forces that:
    1. Increase collision efficiency (particles attract each other)
    2. Are strongest at intermediate saturation (~30-60%)
    3. Depend on surface tension and contact angle
    
    At very low saturation: Not enough liquid for bridges
    At very high saturation: Particles are fully coated, bridges less effective

References:
    - Soos et al. (2007): Size-dependent aggregation efficiency
    - Liquid bridge models from wet granulation literature
    - Capillary force theory (Rabinovich et al.)
"""

import numpy as np
from ..base import AggregationKernel


class LiquidBridgeKernel(AggregationKernel):
    """
    Agglomeration kernel with liquid bridge enhancement.
    
    This kernel extends the classical shear-induced agglomeration model
    by accounting for the effect of liquid content on collision efficiency.
    
    The enhancement has two components:
    
    1. **Bridge Formation Factor**: Models the probability of forming
       stable liquid bridges. Maximum at optimal saturation (~50%).
    
    2. **Capillary Enhancement**: Models the increased collision cross-section
       due to capillary attraction. Increases with external liquid volume.
    
    Parameters:
        corr_beta: Base collision efficiency factor (default: 1e-3)
        g: Shear rate [1/s] (default: 1.0)
        optimal_saturation: Saturation for maximum bridge formation (default: 0.5)
        saturation_width: Width of optimal saturation range (default: 0.2)
        liquid_enhancement: Strength of liquid enhancement (default: 1.0)
        surface_tension: Surface tension [N/m] (default: 0.072, water)
        contact_angle: Contact angle [rad] (default: 0.0, perfect wetting)
    
    Example:
        >>> kernel = LiquidBridgeKernel(
        ...     corr_beta=1e-3,
        ...     g=1000,
        ...     optimal_saturation=0.5,
        ...     liquid_enhancement=2.0
        ... )
        >>> beta = kernel.compute_beta(
        ...     r1=1e-6, r2=2e-6,
        ...     particle1_idx=0, particle2_idx=1,
        ...     solver=solver
        ... )
    """
    
    @property
    def name(self) -> str:
        return 'liquid_bridge'
    
    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-3,
            'g': 1.0,
            'optimal_saturation': 0.5,
            'saturation_width': 0.2,
            'liquid_enhancement': 1.0,
            'surface_tension': 0.072,  # N/m (water at 20°C)
            'contact_angle': 0.0,      # rad (perfect wetting)
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache commonly used values
        self.corr_beta = float(self.params['corr_beta'])
        self.g = float(self.params['g'])
        self.s_opt = float(self.params['optimal_saturation'])
        self.sigma_s = float(self.params['saturation_width'])
        self.alpha_liq = float(self.params['liquid_enhancement'])
        self.gamma = float(self.params['surface_tension'])
        self.theta = float(self.params['contact_angle'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['corr_beta'] <= 0:
            raise ValueError("corr_beta must be positive")
        if params['g'] < 0:
            raise ValueError("g must be non-negative")
        if not (0 <= params['optimal_saturation'] <= 1):
            raise ValueError("optimal_saturation must be in [0, 1]")
        if params['saturation_width'] <= 0:
            raise ValueError("saturation_width must be positive")
        if params['surface_tension'] < 0:
            raise ValueError("surface_tension must be non-negative")
        return params
    
    def compute_beta(self, r1: float, r2: float,
                     particle1_idx: int = None,
                     particle2_idx: int = None,
                     solver = None) -> float:
        """
        Compute collision frequency with liquid bridge enhancement.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx: Index of particle 1 in solver arrays
            particle2_idx: Index of particle 2 in solver arrays
            solver: Reference to solver for accessing saturation/liquid state
        
        Returns:
            beta: Enhanced collision frequency [m³/s]
            
        Note:
            If solver or particle indices are not provided, or if the solver
            doesn't have saturation/liquid attributes, returns the base
            shear-induced collision frequency (graceful degradation).
        """
        # Check for invalid radii
        if r1 <= 0 or r2 <= 0:
            return 0.0
        
        # Base shear-induced collision (Chin et al. 1998)
        beta_shear = self.corr_beta * self.g * (r1 + r2)**3
        
        # If no solver reference or indices, return base value
        # This allows the kernel to work with dry simulations too
        if solver is None or particle1_idx is None or particle2_idx is None:
            return float(beta_shear)
        
        # Check if solver has required attributes for liquid bridge calculation
        has_saturation = hasattr(solver, 'saturation')
        has_liquid = hasattr(solver, 'get_V_liquid_external')
        
        if not has_saturation:
            return float(beta_shear)
        
        # Get saturation values
        s1 = solver.saturation[particle1_idx]
        s2 = solver.saturation[particle2_idx]
        
        # Handle NaN (Vollkörper - no pores, no internal liquid)
        if np.isnan(s1) or np.isnan(s2):
            return float(beta_shear)
        
        # ==========================================
        # Bridge Formation Factor
        # ==========================================
        # Effective saturation (average of both particles)
        s_eff = (s1 + s2) / 2.0
        
        # Gaussian around optimal saturation
        # Maximum bridge formation at s_opt, decreasing on either side
        bridge_factor = np.exp(-((s_eff - self.s_opt)**2) / (2 * self.sigma_s**2))
        
        # ==========================================
        # Capillary Enhancement Factor
        # ==========================================
        if has_liquid:
            v_liq_ext1 = solver.get_V_liquid_external(particle1_idx)
            v_liq_ext2 = solver.get_V_liquid_external(particle2_idx)
            v_liq_total = v_liq_ext1 + v_liq_ext2
            
            # Reference volume: fraction of combined particle volume
            # This makes the enhancement scale-independent
            v_combined = 4.0/3.0 * np.pi * (r1**3 + r2**3)
            v_ref = v_combined * 0.01  # 1% threshold
            
            # Logarithmic enhancement (diminishing returns)
            liquid_factor = 1.0 + self.alpha_liq * np.log1p(v_liq_total / max(v_ref, 1e-30))
        else:
            liquid_factor = 1.0
        
        # ==========================================
        # Combined Enhancement
        # ==========================================
        beta_enhanced = beta_shear * bridge_factor * liquid_factor
        
        return max(0.0, float(beta_enhanced))
