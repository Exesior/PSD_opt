"""
Brownian Diffusion Agglomeration Kernel (Tsouris et al. 1995).

Models collision frequency due to Brownian motion.

Reference:
    Tsouris et al., "Brownian diffusion as controlling mechanism", 1995.

Formula:
    β = CORR_BETA × 2kT × (r1 + r2)² / (3μ × r1 × r2)

Where:
    - kT: Boltzmann constant × temperature ≈ 4.07e-21 J (at 293K)
    - μ: Dynamic viscosity of continuous phase [Pa·s]
    - r1, r2: Particle radii [m]

Implementation:
    Uses JIT-compiled function for performance.
"""

import numpy as np
from numba import njit
from ..base import AggregationKernel


@njit(cache=True)
def _beta_brownian_jit(corr_beta: float, kT: float, viscosity: float, 
                        r1: float, r2: float) -> float:
    """
    JIT-compiled Brownian kernel.
    
    Args:
        corr_beta: Collision efficiency factor
        kT: Boltzmann constant × temperature [J]
        viscosity: Dynamic viscosity [Pa·s]
        r1: Radius of particle 1 [m]
        r2: Radius of particle 2 [m]
    
    Returns:
        beta: Collision frequency [m³/s]
    """
    # Check for invalid radii
    if r1 <= 0 or r2 <= 0:
        return 0.0
    
    # Tsouris et al. (1995) Brownian kernel
    # β = 2kT(r1+r2)² / (3μ r1 r2)
    numerator = 2.0 * kT * (r1 + r2)**2
    denominator = 3.0 * viscosity * r1 * r2
    
    if denominator <= 0:
        return 0.0
    
    beta = corr_beta * numerator / denominator
    
    return max(0.0, beta)


class BrownianKernel(AggregationKernel):
    """
    Brownian diffusion agglomeration kernel after Tsouris et al. (1995).
    
    This kernel is relevant for small particles (< 1 μm) where Brownian
    motion dominates over shear-induced collisions.
    
    Parameters:
        corr_beta: Collision efficiency factor (default: 1.0)
        temperature: Temperature [K] (default: 293.0 = 20°C)
        viscosity: Dynamic viscosity [Pa·s] (default: 1e-3 = water)
    
    Example:
        >>> kernel = BrownianKernel(corr_beta=1.0, temperature=293, viscosity=1e-3)
        >>> beta = kernel.compute_beta(r1=0.1e-6, r2=0.2e-6)
    """
    
    @property
    def name(self) -> str:
        return 'brownian_tsouris1995'
    
    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1.0,
            'temperature': 293.0,  # 20°C
            'viscosity': 1e-3,     # Water at 20°C
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant [J/K]
        
        # Cache commonly used values
        self.corr_beta = float(self.params['corr_beta'])
        self.temperature = float(self.params['temperature'])
        self.viscosity = float(self.params['viscosity'])
        
        # Pre-compute kT
        self.kT = self.k_B * self.temperature
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['corr_beta'] <= 0:
            raise ValueError("corr_beta must be positive")
        if params['temperature'] <= 0:
            raise ValueError("temperature must be positive")
        if params['viscosity'] <= 0:
            raise ValueError("viscosity must be positive")
        return params
    
    def compute_beta(self, r1: float, r2: float,
                     particle1_idx: int = None,
                     particle2_idx: int = None,
                     solver = None) -> float:
        """
        Compute collision frequency for Brownian diffusion.
        
        Uses JIT-compiled function for performance.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx: Not used
            particle2_idx: Not used
            solver: Not used
        
        Returns:
            beta: Collision frequency [m³/s]
        """
        # Use JIT-compiled function
        return _beta_brownian_jit(
            self.corr_beta, self.kT, self.viscosity, r1, r2
        )
