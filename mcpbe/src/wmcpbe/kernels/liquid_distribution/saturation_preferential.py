"""
Saturation-Preferential Liquid Distribution Kernel.

Selects particles based on their saturation level.
Prefers particles with low saturation (far from target).

Formula:
    P(select particle i) ∝ W[i] × (1 + α × max(0, s_target - s[i]))^β
    
    where:
    - s_target: Target saturation (optimal wetting)
    - s[i]: Current saturation of particle i
    - α: Preference strength
    - β: Exponent for non-linear preference

Physical Background:
    In wet granulation, droplets may preferentially collide with or
    adhere to drier particles due to:
    1. Capillary suction (dry particles pull liquid)
    2. Wetting driving force (higher surface energy)
    3. Reduced rebound (dry surfaces capture droplets better)
    
    Conversely, very wet particles may repel additional droplets
    due to saturated surface conditions.

Use case:
    - Systems with strong wetting gradients
    - When liquid distribution is not uniform
    - Modeling preferential wetting phenomena

References:
    - Wetting dynamics in granulation literature
    - Capillary-driven liquid transport models
"""

import numpy as np
from ..base import LiquidDistributionKernel


class SaturationPreferentialKernel(LiquidDistributionKernel):
    """
    Saturation-preferential particle selection for liquid distribution.
    
    This kernel biases droplet distribution toward particles with lower
    saturation, modeling the physical tendency of liquid to wet drier
    surfaces preferentially.
    
    The preference can be tuned from mild (small α) to strong (large α).
    Particles at or above target saturation have baseline probability (W only).
    
    Parameters:
        target_saturation: Target/optimal saturation s_target (default: 0.5)
        preference_strength: Strength parameter α (default: 2.0)
                           Higher values → stronger preference for dry particles
        exponent: Non-linearity exponent β (default: 1.0)
                  >1: Super-linear preference
                  <1: Sub-linear preference
    
    Example:
        >>> kernel = SaturationPreferentialKernel(
        ...     target_saturation=0.6,
        ...     preference_strength=3.0
        ... )
        >>> idx = kernel.select_target_particle(solver, v_droplet=1e-18)
    """
    
    @property
    def name(self) -> str:
        return 'saturation_preferential'
    
    def get_default_params(self) -> dict:
        return {
            'target_saturation': 0.5,
            'preference_strength': 2.0,
            'exponent': 1.0,
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache values
        self.s_target = float(self.params['target_saturation'])
        self.alpha = float(self.params['preference_strength'])
        self.beta = float(self.params['exponent'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if not (0 <= params['target_saturation'] <= 1):
            raise ValueError("target_saturation must be in [0, 1]")
        if params['preference_strength'] < 0:
            raise ValueError("preference_strength must be non-negative")
        if params['exponent'] <= 0:
            raise ValueError("exponent must be positive")
        return params
    
    def select_target_particle(self, solver,
                                v_droplet: float,
                                current_time: float = None) -> int:
        """
        Select particle with preference for low saturation.
        
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
        
        # Get saturations
        if hasattr(solver, 'saturation'):
            saturations = solver.saturation[:a_tot].copy()
            # NaN (Vollkörper) treated as saturation = 0 (fully dry, high preference)
            saturations = np.nan_to_num(saturations, nan=0.0)
        else:
            # No saturation data: fall back to uniform weighted
            saturations = np.zeros(a_tot)
        
        # Compute saturation deficit (how far below target)
        # Positive for under-saturated, zero or negative for over-saturated
        sat_deficit = np.maximum(0.0, self.s_target - saturations)
        
        # Preference factor: (1 + α × deficit)^β
        preference = (1.0 + self.alpha * sat_deficit) ** self.beta
        
        # Combined weight: W × preference
        weights = W * preference
        weights[~valid_mask] = 0
        
        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            return -1
        
        probs = weights / weight_sum
        
        # Sample using solver's RNG
        return int(solver._rng.choice(a_tot, p=probs))
