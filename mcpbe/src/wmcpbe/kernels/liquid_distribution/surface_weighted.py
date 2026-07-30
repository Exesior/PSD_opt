"""
Surface-Weighted Liquid Distribution Kernel.

Selects particles proportional to their surface area × weight.
Larger particles have higher collision cross-section for droplets.

Formula:
    P(select particle i) = (W[i] × A[i]) / Σ_j (W[j] × A[j])
    
    where A[i] = π × d[i]² is the particle surface area

Physical Background:
    In reality, droplet-particle collision frequency scales with the
    particle's geometric cross-section. Larger particles present a
    bigger target for incoming droplets.
    
    This kernel captures this size dependence while maintaining DSMC
    consistency through the W weighting.

Use case:
    - Systems with significant size disparity
    - When droplet mean free path >> particle size (geometric regime)
    - More physically accurate than uniform for many systems

References:
    - Collision theory from aerosol science
    - Droplet-particle collision models in spray drying literature
"""

import numpy as np
from ..base import LiquidDistributionKernel


class SurfaceWeightedKernel(LiquidDistributionKernel):
    """
    Surface-area-weighted particle selection for liquid distribution.
    
    This kernel selects particles with probability proportional to both
    their computational weight W AND their surface area A. This models
    the physical reality that larger particles have larger collision
    cross-sections for droplets.
    
    Parameters:
        None (this kernel has no tunable parameters)
    
    Example:
        >>> kernel = SurfaceWeightedKernel()
        >>> idx = kernel.select_target_particle(solver, v_droplet=1e-18)
    """
    
    @property
    def name(self) -> str:
        return 'surface_weighted'
    
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
        Select particle proportional to W × surface_area.
        
        Args:
            solver: Reference to solver instance
            v_droplet: Droplet volume [m³] (not used)
            current_time: Simulation time [s] (not used)
        
        Returns:
            Index of selected particle, or -1 if no valid particle
        """
        a_tot = solver.a_tot
        if a_tot < 1:
            return -1
        
        # Get weights
        W = solver.W[:a_tot]
        valid_mask = (W > 0) & np.isfinite(W)
        
        if not np.any(valid_mask):
            return -1
        
        # Get diameters and compute surface areas
        if hasattr(solver, 'X'):
            diameters = solver.X[:a_tot]
        else:
            # Compute from volume if X not available
            V_ges = solver.V_flat[-1, :a_tot]
            diameters = (6.0 * V_ges / np.pi) ** (1.0/3.0)
        
        # Surface area: A = π × d²
        areas = np.pi * diameters ** 2
        
        # Combined weight: W × A
        weights = W * areas
        weights[~valid_mask] = 0
        
        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            return -1
        
        probs = weights / weight_sum
        
        # Sample using solver's RNG
        return int(solver._rng.choice(a_tot, p=probs))
