"""
Stress-Dependent Compaction Kernel.

Porosity reduction based on local mechanical stress.

Formula:
    dε/dt = -k × (σ / σ_ref)^n × (ε - ε_min)^m

Where:
    - σ: Local stress [Pa]
    - σ_ref: Reference stress [Pa]
    - n: Stress exponent (typically 1-2)
    - m: Porosity exponent (typically 1)
    - k: Rate constant [1/s]

Physical Background:
    Unlike simple exponential decay, this model accounts for:
    1. Size-dependent stress (larger particles experience higher stress)
    2. Shear-rate-dependent stress (higher G → higher σ)
    3. Saturation effects (liquid can cushion or enhance compaction)

References:
    - Soil mechanics: Terzaghi consolidation theory
    - Granulation: Tardos et al., "Stress-induced compaction", 1997
"""

import numpy as np
from ..base import CompressionKernel


class StressCompactionKernel(CompressionKernel):
    """
    Stress-dependent porosity compaction kernel.
    
    This kernel models compaction as driven by local mechanical stress,
    which depends on particle size, shear rate, and material properties.
    
    Key features:
    - Larger particles compact faster (higher local stress)
    - Higher shear rate → faster compaction
    - Saturation can reduce effective stress (lubrication)
    
    Parameters:
        rate: Base rate constant k [1/s] (default: 0.01)
        min_porosity: Minimum achievable porosity (default: 0.3)
        stress_exponent: Exponent n for stress dependence (default: 1.0)
        reference_stress: Reference stress σ_ref [Pa] (default: 1e5)
        saturation_cushioning: If True, saturation reduces stress (default: True)
    
    Example:
        >>> kernel = StressCompactionKernel(
        ...     rate=0.01,
        ...     min_porosity=0.3,
        ...     stress_exponent=1.5
        ... )
        >>> poro_new = kernel.compute_porosity_decay(
        ...     poro=0.5, dt=1.0,
        ...     v_particle=1e-18, saturation=0.6
        ... )
    """
    
    @property
    def name(self) -> str:
        return 'stress_compaction'
    
    def get_default_params(self) -> dict:
        return {
            'rate': 0.01,
            'min_porosity': 0.3,
            'stress_exponent': 1.0,
            'reference_stress': 1e5,  # Pa (≈ 1 bar)
            'saturation_cushioning': True,
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache values
        self.k = float(self.params['rate'])
        self.eps_min = float(self.params['min_porosity'])
        self.n = float(self.params['stress_exponent'])
        self.sigma_ref = float(self.params['reference_stress'])
        self.use_cushioning = bool(self.params['saturation_cushioning'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['rate'] < 0:
            raise ValueError("rate must be non-negative")
        if not (0 <= params['min_porosity'] <= 1):
            raise ValueError("min_porosity must be in [0, 1]")
        if params['stress_exponent'] < 0:
            raise ValueError("stress_exponent must be non-negative")
        if params['reference_stress'] <= 0:
            raise ValueError("reference_stress must be positive")
        return params
    
    def compute(self,
                porosity: float,
                dt: float,
                v_particle: float = None,
                local_stress: float = None,
                saturation: float = None,
                solver = None
                ) -> float:
        """
        Compute new porosity after stress-dependent compaction.
        
        Args:
            porosity: Current porosity (0 to 1)
            dt: Time step [s]
            v_particle: Particle volume [m³] (for stress estimation)
            local_stress: Pre-computed local stress [Pa] (optional)
            saturation: Particle saturation (optional, for cushioning)
            solver: Reference to solver (for accessing G, etc.)
        
        Returns:
            porosity_new: Reduced porosity (≥ min_porosity)
        """
        # Vollkörper cannot be compressed
        if np.isnan(porosity):
            return porosity
        
        # Clamp input to valid range
        porosity = max(0.0, min(1.0, porosity))
        
        # If already at minimum, no further compression
        if porosity <= self.eps_min:
            return self.eps_min
        
        # ==========================================
        # Estimate local stress
        # ==========================================
        if local_stress is not None:
            # Use provided stress value
            sigma = float(local_stress)
        else:
            # Estimate from particle size and shear
            if v_particle is not None and v_particle > 0:
                d = (6.0 * v_particle / np.pi) ** (1.0/3.0)
                
                # Get shear rate from solver if available
                g = 1.0  # Default
                if solver is not None and hasattr(solver, 'G'):
                    g = float(solver.G)
                
                # Simple stress model: σ ~ μ × G × (d/d_ref)
                mu = 1e-3  # Viscosity [Pa·s]
                d_ref = 1e-6  # Reference size [m]
                sigma = mu * g * (d / d_ref)
            else:
                # Fallback to reference stress
                sigma = self.sigma_ref
        
        # ==========================================
        # Saturation cushioning effect
        # ==========================================
        stress_factor = 1.0
        
        if self.use_cushioning and saturation is not None:
            if not np.isnan(saturation) and 0 <= saturation <= 1:
                # High saturation cushions particles (liquid film)
                # Low saturation: direct solid-solid contact
                cushion_factor = 1.0 - 0.5 * saturation  # Up to 50% reduction
                stress_factor = cushion_factor
        
        # ==========================================
        # Stress-dependent compaction rate
        # ==========================================
        # Non-dimensional stress ratio
        stress_ratio = sigma / self.sigma_ref
        
        # Effective rate: k_eff = k × (σ/σ_ref)^n
        k_eff = self.k * (stress_ratio ** self.n)
        
        # ==========================================
        # Apply exponential decay with effective rate
        # ==========================================
        delta_poro = porosity - self.eps_min
        porosity_new = self.eps_min + delta_poro * np.exp(-k_eff * dt)
        
        # Ensure we don't go below minimum
        porosity_new = max(self.eps_min, porosity_new)
        
        return float(porosity_new)
