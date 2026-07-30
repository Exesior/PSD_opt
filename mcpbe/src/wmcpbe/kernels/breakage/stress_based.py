"""
Stress-Based Breakage Kernel (Weibull Model).

Breakage rate based on mechanical stress exceeding particle strength.
Uses Weibull statistics for brittle fracture probability.

Formula:
    P(break) = 1 - exp(-(σ / σ_crit)^m)
    S(V) = P(break) × f_collision

Where:
    - σ: Applied stress [Pa]
    - σ_crit: Critical stress (particle strength) [Pa]
    - m: Weibull modulus (scatter parameter)
    - f_collision: Collision frequency with mixer walls/other particles

Physical Background:
    - Larger particles experience higher stress (σ ~ d for given shear)
    - Porous particles have lower effective strength
    - Weibull modulus m describes strength distribution:
      - Low m (~1-2): High scatter (brittle materials)
      - High m (>5): Low scatter (ductile materials)

References:
    - Weibull statistics for brittle fracture
    - Stress models from population balance literature
"""

import numpy as np
from ..base import BreakageKernel


class StressBasedBreakageKernel(BreakageKernel):
    """
    Stress-based breakage kernel using Weibull statistics.
    
    This kernel models breakage as a stochastic process where particles
    break when local stress exceeds their strength. The strength follows
    a Weibull distribution, accounting for material defects and heterogeneity.
    
    Key features:
    - Size-dependent stress (larger particles break more easily)
    - Porosity-dependent strength (porous particles are weaker)
    - Weibull statistics for fracture probability
    
    Parameters:
        critical_stress: Reference particle strength [Pa] (default: 1e6)
        weibull_modulus: Weibull modulus m (default: 2.0)
        g: Shear rate [1/s] (default: 1.0)
        stress_size_exp: Exponent for size-stress relation (default: 1.0)
        porosity_weakening: Exponent for porosity weakening (default: 2.0)
    
    Example:
        >>> kernel = StressBasedBreakageKernel(
        ...     critical_stress=1e6,
        ...     weibull_modulus=2.0,
        ...     g=1000
        ... )
        >>> rate = kernel.compute_rate(v_particle=1e-18, solver=solver)
    """
    
    @property
    def name(self) -> str:
        return 'stress_based'
    
    def get_default_params(self) -> dict:
        return {
            'critical_stress': 1e6,      # Pa (typical for brittle solids)
            'weibull_modulus': 2.0,       # Dimensionless
            'g': 1.0,                     # 1/s
            'stress_size_exp': 1.0,       # σ ~ d^exp
            'porosity_weakening': 2.0,    # σ_crit_eff = σ_crit × (1-p)^exp
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache values
        self.sigma_crit = float(self.params['critical_stress'])
        self.m = float(self.params['weibull_modulus'])
        self.g = float(self.params['g'])
        self.size_exp = float(self.params['stress_size_exp'])
        self.poro_exp = float(self.params['porosity_weakening'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['critical_stress'] <= 0:
            raise ValueError("critical_stress must be positive")
        if params['weibull_modulus'] <= 0:
            raise ValueError("weibull_modulus must be positive")
        if params['g'] < 0:
            raise ValueError("g must be non-negative")
        return params
    
    def compute_rate(self, v_particle: float,
                     particle_idx: int = None,
                     solver = None) -> float:
        """
        Compute breakage rate using stress-based Weibull model.
        
        Args:
            v_particle: Particle volume [m³]
            particle_idx: Index in solver for accessing porosity (optional)
            solver: Reference to solver for accessing porosity state
        
        Returns:
            rate: Breakage rate [1/s]
        """
        if v_particle <= 0:
            return 0.0
        
        # Calculate particle diameter
        d = (6.0 * v_particle / np.pi) ** (1.0/3.0)
        
        # ==========================================
        # Estimate local stress
        # ==========================================
        # Simple model: stress scales with particle size and shear rate
        # σ ~ μ × G × (d / d_ref)^size_exp
        mu = 1e-3  # Viscosity [Pa·s] (water)
        d_ref = 1e-6  # Reference size [m]
        
        sigma_applied = mu * self.g * (d / d_ref) ** self.size_exp
        
        # ==========================================
        # Effective critical stress (porosity weakening)
        # ==========================================
        sigma_eff = self.sigma_crit
        
        if solver is not None and particle_idx is not None:
            if hasattr(solver, 'porosity'):
                poro = solver.porosity[particle_idx]
                if not np.isnan(poro) and 0 < poro < 1:
                    # Porosity reduces effective strength
                    # σ_crit_eff = σ_crit × (1 - porosity)^poro_exp
                    sigma_eff = self.sigma_crit * (1.0 - poro) ** self.poro_exp
        
        # ==========================================
        # Weibull breakage probability
        # ==========================================
        if sigma_eff <= 0 or sigma_applied <= 0:
            return 0.0
        
        stress_ratio = sigma_applied / sigma_eff
        
        # P(break) = 1 - exp(-σ^m)
        break_prob = 1.0 - np.exp(-stress_ratio ** self.m)
        
        # Rate = collision frequency × breakage probability
        # Collision frequency ~ shear rate
        rate = self.g * break_prob
        
        return max(0.0, float(rate))
