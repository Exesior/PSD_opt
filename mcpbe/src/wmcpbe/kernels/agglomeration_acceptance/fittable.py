"""
Fittable Kernel: Simple Agglomeration Acceptance with Fit Parameter.

This kernel implements a simple probabilistic acceptance criterion:
- Draw random number u ~ Uniform(0, 1)
- Accept if u < u_acc
- Reject otherwise

The parameter u_acc ∈ [0, 1] can be fitted to experimental data.
This is useful when detailed physics (like Stokes criterion) is not needed
or when calibrating the model to match observed agglomeration rates.

Usage:
    >>> kernel = FittableKernel(u_acc=0.8)  # 80% acceptance rate
    >>> accepted = kernel.accept_collision(r1, r2, v_dry1, v_dry2, solver=solver)
"""

from typing import Any, Dict, Optional
import numpy as np

from ..base import AggAcceptanceKernel


class FittableKernel(AggAcceptanceKernel):
    """
    Simple fit-parameter-based agglomeration acceptance.
    
    Physical Model:
        No explicit physics. Agglomeration occurs with fixed probability u_acc.
        This is a phenomenological model for calibration purposes.
        
        - Draw u ~ Uniform(0, 1)
        - Accept if u < u_acc
        - Reject otherwise
        
        Effective agglomeration rate = β(i,j) × u_acc
    
    Parameters:
        u_acc: Acceptance probability [dimensionless]. Range: [0, 1].
               - u_acc = 1.0: Always accept (no rejection)
               - u_acc = 0.5: 50% acceptance rate
               - u_acc = 0.0: Never accept (no agglomeration)
    
    Note:
        This kernel uses the solver's RNG for reproducibility. If solver
        is not provided, falls back to np.random.default_rng().
    """
    
    @property
    def name(self) -> str:
        return 'fittable'
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'u_acc': 1.0,  # Default: always accept (equivalent to no acceptance kernel)
        }
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not (0.0 <= params['u_acc'] <= 1.0):
            raise ValueError(f"u_acc must be in [0, 1], got {params['u_acc']}")
        return params
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache frequently used values
        self.u_acc = float(self.params['u_acc'])
    
    def accept_collision(self,
                         r1: float, r2: float,
                         v_dry1: float, v_dry2: float,
                         particle1_idx: Optional[int] = None,
                         particle2_idx: Optional[int] = None,
                         solver: Optional[Any] = None
                         ) -> bool:
        """
        Determine if collision results in agglomeration using fit parameter.
        
        Args:
            r1: Radius of particle 1 [m] (unused)
            r2: Radius of particle 2 [m] (unused)
            v_dry1: Dry volume of particle 1 [m³] (unused)
            v_dry2: Dry volume of particle 2 [m³] (unused)
            particle1_idx: Index of particle 1 (unused)
            particle2_idx: Index of particle 2 (unused)
            solver: Reference to solver for accessing RNG
        
        Returns:
            accepted: True with probability u_acc, False otherwise
        
        Note:
            All physical parameters are ignored. Only u_acc matters.
        """
        # Edge cases
        if self.u_acc >= 1.0:
            return True  # Always accept
        if self.u_acc <= 0.0:
            return False  # Never accept
        
        # Get RNG from solver (for reproducibility) or use default
        if solver is not None and hasattr(solver, '_rng'):
            rng = solver._rng
        else:
            rng = np.random.default_rng()
        
        # Draw random number and compare
        u = float(rng.random())
        return u < self.u_acc
