"""
Constant Agglomeration Kernel.

Simplest possible kernel: collision frequency is independent of particle size.

Formula:
    β = CORR_BETA (constant)

Use case:
    - Validation against analytical solutions
    - Simplified models where size dependence is negligible
    - Testing/debugging
"""

from numba import njit
from ..base import AggregationKernel


@njit(cache=True)
def _beta_constant_jit(corr_beta: float) -> float:
    """
    JIT-compiled constant kernel.
    
    Args:
        corr_beta: Constant collision frequency [m³/s]
    
    Returns:
        beta: Collision frequency (same as corr_beta)
    """
    return corr_beta


class ConstantKernel(AggregationKernel):
    """
    Constant collision frequency kernel.
    
    This kernel assumes all particle pairs collide at the same rate,
    regardless of their sizes. While physically unrealistic for most
    systems, it's useful for:
    
    1. Validation against analytical solutions (Smoluchowski equation)
    2. Testing the solver framework
    3. Simplified models where size effects are averaged out
    
    Parameters:
        corr_beta: Constant collision frequency [m³/s]
    
    Example:
        >>> kernel = ConstantKernel(corr_beta=1e-15)
        >>> beta = kernel.compute_beta(r1=1e-6, r2=2e-6)
        >>> print(f"Collision frequency: {beta:.3e} m³/s")  # Always 1e-15
    """
    
    @property
    def name(self) -> str:
        return 'constant'
    
    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-10,  # Typical order of magnitude
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
        Return constant collision frequency.
        
        Uses JIT-compiled function for performance.
        
        Args:
            r1, r2: Not used (size-independent)
            particle1_idx, particle2_idx, solver: Not used
        
        Returns:
            beta: Constant collision frequency [m³/s]
        """
        return _beta_constant_jit(self.corr_beta)
