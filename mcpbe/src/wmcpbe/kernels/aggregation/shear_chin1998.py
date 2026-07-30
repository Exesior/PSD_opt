"""
Shear-Induced Agglomeration Kernel (Chin et al. 1998).

Models collision frequency due to shear in stirred tanks.

Reference:
    Chin et al., "Shear-induced flocculation in stirred tanks", 1998.

Formula:
    β = CORR_BETA × G × (r1 + r2)³

Where:
    - CORR_BETA: Collision efficiency factor (dimensionless)
    - G: Shear rate [1/s]
    - r1, r2: Particle radii [m]

Implementation:
    Uses JIT-compiled function for maximum performance.
"""

import numpy as np
from numba import njit
from ..base import AggregationKernel


@njit(cache=True)
def _beta_shear_jit(corr_beta: float, g: float, r1: float, r2: float) -> float:
    """
    JIT-compiled shear kernel.
    
    Args:
        corr_beta: Collision efficiency factor
        g: Shear rate [1/s]
        r1: Radius of particle 1 [m]
        r2: Radius of particle 2 [m]
    
    Returns:
        beta: Collision frequency [m³/s]
    """
    return corr_beta * g * (r1 + r2)**3


class ShearChinKernel(AggregationKernel):
    """
    Shear-induced agglomeration kernel after Chin et al. (1998).
    
    This is the most commonly used kernel for high-shear granulation
    processes where turbulent shear dominates particle collisions.
    
    Parameters:
        corr_beta: Collision efficiency factor (default: 1e-3)
                   Accounts for collision efficiency < 100%
        g: Shear rate [1/s] (default: 1.0)
           Typical values: 100-10000 1/s for high-shear mixers
    
    Example:
        >>> kernel = ShearChinKernel(corr_beta=1e-3, g=1000)
        >>> beta = kernel.compute_beta(r1=1e-6, r2=2e-6)
        >>> print(f"Collision frequency: {beta:.3e} m³/s")
    """
    
    @property
    def name(self) -> str:
        return 'shear_chin1998'
    
    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-3,
            'g': 1.0,
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache commonly used values
        self.corr_beta = float(self.params['corr_beta'])
        self.g = float(self.params['g'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['corr_beta'] <= 0:
            raise ValueError("corr_beta must be positive")
        if params['g'] < 0:
            raise ValueError("g must be non-negative")
        return params
    
    def compute_beta(self, r1: float, r2: float,
                     particle1_idx: int = None,
                     particle2_idx: int = None,
                     solver = None) -> float:
        """
        Compute collision frequency for shear-induced agglomeration.
        
        Uses JIT-compiled function for performance.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx: Not used (for interface compatibility)
            particle2_idx: Not used (for interface compatibility)
            solver: Not used (for interface compatibility)
        
        Returns:
            beta: Collision frequency [m³/s]
        """
        # Check for invalid radii
        if r1 <= 0 or r2 <= 0:
            return 0.0
        
        # Use JIT-compiled function
        beta = _beta_shear_jit(self.corr_beta, self.g, r1, r2)
        
        return max(0.0, float(beta))
