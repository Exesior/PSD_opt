"""
Sum Agglomeration Kernel.

Collision frequency proportional to sum of particle volumes.

Formula:
    β = CORR_BETA × 4π/3 × (r1³ + r2³) = CORR_BETA × (V1 + V2)

Use case:
    - Validation against analytical solutions
    - Systems where collision rate scales with particle volume
"""

import math
from numba import njit
from ..base import AggregationKernel


@njit(cache=True)
def _beta_sum_jit(corr_beta: float, r1: float, r2: float) -> float:
    """
    JIT-compiled sum kernel.
    
    Args:
        corr_beta: Scaling factor [1/(m³·s)]
        r1: Radius of particle 1 [m]
        r2: Radius of particle 2 [m]
    
    Returns:
        beta: Collision frequency [m³/s]
    """
    # Sum kernel: β ∝ (V1 + V2)
    v1 = 4.0/3.0 * math.pi * r1**3
    v2 = 4.0/3.0 * math.pi * r2**3
    return corr_beta * (v1 + v2)


class SumKernel(AggregationKernel):
    """
    Sum kernel for agglomeration.
    
    The collision frequency is proportional to the sum of particle volumes.
    This kernel is sometimes used for validation because it leads to
    tractable analytical solutions of the Smoluchowski equation.
    
    Parameters:
        corr_beta: Scaling factor [1/(m³·s)]
    
    Example:
        >>> kernel = SumKernel(corr_beta=1e-9)
        >>> beta = kernel.compute_beta(r1=1e-6, r2=2e-6)
    """
    
    @property
    def name(self) -> str:
        return 'sum'
    
    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-9,
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache value
        self.corr_beta = float(self.params['corr_beta'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['corr_beta'] < 0:
            raise ValueError("corr_beta must be non-negative")
        return params
    
    def compute_beta(self, r1: float, r2: float,
                     particle1_idx: int = None,
                     particle2_idx: int = None,
                     solver = None) -> float:
        """
        Compute collision frequency proportional to volume sum.
        
        Uses JIT-compiled function for performance.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx, particle2_idx, solver: Not used
        
        Returns:
            beta: Collision frequency [m³/s]
        """
        if r1 <= 0 or r2 <= 0:
            return 0.0
        
        # Use JIT-compiled function
        beta = _beta_sum_jit(self.corr_beta, r1, r2)
        
        return max(0.0, float(beta))
