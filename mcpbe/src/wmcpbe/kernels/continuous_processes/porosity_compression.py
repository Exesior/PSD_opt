"""
Porosity Compression Kernel.

Classical model: porosity decays exponentially under constant shear.

Formula:
    ε(t) = ε_min + (ε_0 - ε_min) × exp(-k × t)
    
    dε/dt = -k × (ε - ε_min)

Where:
    - ε_0: Initial porosity
    - ε_min: Minimum achievable porosity (asymptote)
    - k: Decay rate constant [1/s]
    - t: Time under shear [s]

Physical Background:
    This model assumes that porosity reduction is:
    1. Proportional to current porosity (more pores → faster collapse)
    2. Limited by a minimum porosity (random close packing limit)
    3. Independent of particle size (mean-field approximation)

References:
    - Exponential consolidation models from soil mechanics
    - Applied to granulation in: 10.1103/PhysRevLett.64.2727
"""

import numpy as np
from ..base import CompressionKernel


class PorosityCompressionKernel(CompressionKernel):
    """
    Exponential porosity decay under constant shear.
    
    This is the standard compression model used in most WMCPBE simulations.
    It captures the empirical observation that porosity decreases rapidly
    at first, then asymptotically approaches a minimum value.
    
    Parameters:
        rate: Decay rate constant k [1/s] (default: 0.02)
              Higher values → faster compaction
        min_porosity: Minimum achievable porosity ε_min (default: 0.3)
                      Typical range: 0.25 - 0.4 for random close packing
    
    Example:
        >>> kernel = PorosityCompressionKernel(rate=0.02, min_porosity=0.3)
        >>> poro_new = kernel.compute(poro=0.5, dt=1.0)
        >>> print(f"Porosity after 1s: {poro_new:.3f}")
    """
    
    @property
    def name(self) -> str:
        return 'porosity_compression'
    
    def get_default_params(self) -> dict:
        return {
            'rate': 0.02,         # 1/s
            'min_porosity': 0.3,  # Dimensionless
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache values
        self.k = float(self.params['rate'])
        self.eps_min = float(self.params['min_porosity'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['rate'] < 0:
            raise ValueError("rate must be non-negative")
        if not (0 <= params['min_porosity'] <= 1):
            raise ValueError("min_porosity must be in [0, 1]")
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
        Compute new porosity after exponential decay over time dt.
        
        Args:
            porosity: Current porosity (0 to 1)
            dt: Time step [s]
            v_particle: Not used (size-independent model)
            local_stress: Not used (stress already in rate parameter)
            saturation: Not used (saturation-independent)
            solver: Not used
        
        Returns:
            porosity_new: Reduced porosity (≥ min_porosity)
            
        Note:
            Returns input porosity unchanged if it's NaN (Vollkörper).
        """
        # Vollkörper cannot be compressed
        if np.isnan(porosity):
            return porosity
        
        # Clamp input to valid range
        porosity = max(0.0, min(1.0, porosity))
        
        # If already at or below minimum, no further compression
        if porosity <= self.eps_min:
            return self.eps_min
        
        # ==========================================
        # Exponential decay formula
        # ==========================================
        # Analytical solution: ε(t+dt) = ε_min + (ε(t) - ε_min) × exp(-k×dt)
    
        delta_poro = porosity - self.eps_min
        porosity_new = self.eps_min + delta_poro * np.exp(-self.k * dt)

        # Ensure we don't go below minimum (numerical safety)
        porosity_new = max(self.eps_min, porosity_new)

        return float(porosity_new)

    def compute_array(self, porosity: np.ndarray, dt: float) -> np.ndarray:
        """Vectorised form of :meth:`compute`.

        Bit-identical to a loop of scalar calls: the decay factor
        ``exp(-k*dt)`` is a scalar, so only multiplications and clamps are
        applied elementwise.

        Args:
            porosity: Current porosities; NaN entries ("Vollkoerper") pass
                      through unchanged.
            dt:       Time step [s]

        Returns:
            Updated porosities, same shape as ``porosity``.
        """
        poro = np.asarray(porosity, dtype=float)
        out = poro.copy()

        finite = ~np.isnan(poro)
        if not np.any(finite):
            return out

        clipped = np.clip(poro[finite], 0.0, 1.0)
        decay = np.exp(-self.k * dt)
        updated = self.eps_min + (clipped - self.eps_min) * decay
        updated = np.maximum(self.eps_min, updated)
        # At or below the floor there is nothing left to compress.
        out[finite] = np.where(clipped <= self.eps_min, self.eps_min, updated)
        return out
