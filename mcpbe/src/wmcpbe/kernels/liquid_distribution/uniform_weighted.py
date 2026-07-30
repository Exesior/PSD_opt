"""
Uniform Weighted Liquid Distribution Kernel.

Selects particles proportional to their computational weight W.
This ensures all PHYSICAL particles have equal collision probability.

Formula:
    P(select particle i) = W[i] / Σ_j W[j]

Physical Background:
    In DSMC (Direct Simulation Monte Carlo), each computational particle
    represents W[i] physical particles. To sample physical particles
    uniformly, we must sample computational particles with probability
    proportional to W.

Use case:
    - Default behavior for most systems
    - When droplet-particle collisions are size-independent
    - Validation against existing simulations
"""

import numpy as np
from ..base import LiquidDistributionKernel


class UniformWeightedKernel(LiquidDistributionKernel):
    """
    Uniform weighted particle selection for liquid distribution.
    
    This kernel selects computational particles with probability proportional
    to their weight W, which corresponds to uniform selection of PHYSICAL
    particles.
    
    The implementation uses a Fenwick tree sampler for O(log n) selection.
    
    Parameters:
        None (this kernel has no tunable parameters)
    
    Example:
        >>> kernel = UniformWeightedKernel()
        >>> idx = kernel.select_target_particle(solver, v_droplet=1e-18)
    """
    
    @property
    def name(self) -> str:
        return 'uniform_weighted'
    
    def get_default_params(self) -> dict:
        return {}
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
    
    def validate_params(self, params: dict) -> dict:
        """No parameters to validate."""
        return params
    
    def select_target_particle(self, solver,
                                v_droplet: float,
                                current_time: float = None) -> int:
        """
        Select particle proportional to weight W.
        
        Uses the solver's built-in weight-based sampler if available,
        otherwise falls back to manual sampling.
        
        Args:
            solver: Reference to solver instance
            v_droplet: Droplet volume [m³] (not used in this kernel)
            current_time: Simulation time [s] (not used)
        
        Returns:
            Index of selected particle, or -1 if no valid particle
        
        Note:
            This method delegates to solver._select_particle_uniform_physical()
            which uses a Fenwick tree for efficient weighted sampling.
        """
        # Delegate to solver's built-in method
        # This is the same method used by NucleationHandler
        if hasattr(solver, '_select_particle_uniform_physical'):
            return solver._select_particle_uniform_physical()
        
        # Fallback: manual weighted sampling
        a_tot = solver.a_tot
        if a_tot < 1:
            return -1
        
        W = solver.W[:a_tot]
        valid_mask = (W > 0) & np.isfinite(W)
        
        if not np.any(valid_mask):
            return -1
        
        # Normalize weights
        W_valid = W.copy()
        W_valid[~valid_mask] = 0
        W_sum = np.sum(W_valid)
        
        if W_sum <= 0:
            return -1
        
        W_norm = W_valid / W_sum
        
        # Sample using solver's RNG
        return int(solver._rng.choice(a_tot, p=W_norm))
