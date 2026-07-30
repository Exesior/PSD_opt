"""
Power-Law Breakage Kernel.

Classical breakage rate model where rate scales with particle size.

Formula:
    S(V) = P1 × G^P2 × V^(alpha)

Where:
    - P1, P2: Empirical parameters
    - G: Shear rate [1/s]
    - V: Particle volume [m³]
    - alpha: Size exponent (typically P2/3 for volume scaling)

This kernel wraps the existing JIT implementation from pbe_core.func.jit_kernel_break
for maximum performance.
"""

import numpy as np
from ..base import BreakageKernel


class PowerLawBreakageKernel(BreakageKernel):
    """
    Power-law breakage rate kernel.
    
    This is the classical breakage model used in most PBE simulations.
    The breakage rate increases with particle size and shear rate.
    
    Parameters:
        p1: Pre-factor [1/s·m^(-3*alpha)] (default: 3e-2)
        p2: Shear exponent (default: 1.0)
        g: Shear rate [1/s] (default: 1.0)
        breakrval: Breakage model variant (default: 1)
                   1: Constant rate
                   2: Linear in volume
                   4: Power law with pl_v
                   5: Modified power law
    
    Example:
        >>> kernel = PowerLawBreakageKernel(p1=3e-2, p2=1.0, g=1000)
        >>> rate = kernel.compute_rate(v_particle=1e-18)
    """
    
    @property
    def name(self) -> str:
        return 'power_law'
    
    def get_default_params(self) -> dict:
        return {
            'p1': 3e-2,      # Pre-factor
            'p2': 1.0,       # Shear exponent / size exponent  
            'g': 1.0,        # Shear rate [1/s]
            'breakrval': 1,  # Breakage model variant (1-5)
            'pl_v': 2.0,     # Volume exponent for BREAKFVAL=4 (legacy default)
            'pl_q': 1.0,     # Exponent for BREAKFVAL=3 (MUST BE 1.0 FOR STABILITY!)
                             # BREAKFVAL=3 with pl_q=0.5 causes NaN at z=0,1
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache values - EXACT names matching legacy solver attributes
        self.p1 = float(self.params['p1'])       # = pl_P1 in legacy
        self.p2 = float(self.params['p2'])       # = pl_P2 in legacy
        self.g = float(self.params['g'])         # = G in legacy
        self.breakrval = int(self.params['breakrval'])
        self.pl_v = float(self.params.get('pl_v', 2.0))   # = pl_v in legacy
        # CRITICAL: pl_q MUST be 1.0 for BREAKFVAL=3 (numerical stability)
        # pl_q != 1.0 creates singularities at z=0 or z=1
        self.pl_q = float(self.params.get('pl_q', 1.0))   # = pl_q in legacy
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        # Check required parameters
        required = ['p1', 'p2', 'g']
        for key in required:
            if key not in params:
                raise ValueError(
                    f"Missing required parameter '{key}' for power_law breakage kernel. "
                    f"Required parameters: {required}"
                )
        
        # Validate values
        if params['p1'] < 0:
            raise ValueError(f"p1 must be non-negative, got {params['p1']}")
        if params['g'] < 0:
            raise ValueError(f"g must be non-negative, got {params['g']}")
        
        # Validate breakrval
        breakrval = params.get('breakrval', 1)
        if breakrval not in (1, 2, 3, 4, 5):
            raise ValueError(
                f"Invalid breakrval={breakrval}. Must be one of: 1, 2, 3, 4, 5"
            )
        
        # Validate pl_v for BREAKRVAL=4
        if breakrval == 4 and 'pl_v' not in params:
            raise ValueError(
                f"Parameter 'pl_v' is required for breakrval=4 (volume-based power law)"
            )
        
        # Validate pl_q for BREAKRVAL=5
        if breakrval == 5 and 'pl_q' not in params:
            raise ValueError(
                f"Parameter 'pl_q' is required for breakrval=5 (modified power law)"
            )
        
        return params
    
    def compute_rate(self, v_particle: float,
                     particle_idx: int = None,
                     solver = None) -> float:
        """
        Compute breakage rate using power-law model.
        
        CRITICAL: This must match the legacy JIT implementation EXACTLY.
        The JIT function is: pbe_core.func.jit_kernel_break.calc_break_rate_1d
        
        Supports all BREAKRVAL variants (1-5):
            BREAKRVAL=1: S = P1 (constant)
            BREAKRVAL=2: S = P1 × V^(1/3) (linear in diameter)
            BREAKRVAL=3: NOT USED in 1D (2D only)
            BREAKRVAL=4: S = P1 × G^P2 × V^(pl_v/3) (volume-based power law)
            BREAKRVAL=5: S = P1 × G^P2 × (V/V_ref)^(pl_q) (modified power law)
        
        IMPORTANT for BREAKRVAL=4,5:
            Legacy JIT uses pl_v and pl_q from solver attributes!
            Our kernel receives these as p2 (for pl_v) or needs special handling.
        
        Args:
            v_particle: Particle volume [m³]
            particle_idx: Not used (for interface compatibility)
            solver: Not used (for interface compatibility)
        
        Returns:
            rate: Breakage rate [1/s]
        """
        if v_particle <= 0:
            return 0.0
        
        # Support different BREAKRVAL variants (matching legacy JIT behavior)
        if self.breakrval == 1:
            # Constant rate: S = P1
            rate = self.p1
        elif self.breakrval == 2:
            # Linear in diameter: S = P1 × V^(1/3)
            rate = self.p1 * (v_particle ** (1.0/3.0))
        elif self.breakrval == 3:
            # For 1D: Same as BREAKRVAL=1 (constant)
            # BREAKRVAL=3 is primarily for 2D cases
            rate = self.p1
        elif self.breakrval == 4:
            # Volume-based power law: S = P1 × G^P2 × V^(pl_v/3)
            # In legacy code: pl_v is stored separately, not in p2!
            # For backward compatibility, we use p2 as pl_v when breakrval=4
            pl_v = getattr(self, 'pl_v', self.p2)  # Fallback to p2 if pl_v not set
            alpha = pl_v / 3.0
            rate = self.p1 * (self.g ** self.p2) * (v_particle ** alpha)
        elif self.breakrval == 5:
            # Modified power law: S = P1 × G^P2 × (V/V_ref)^(pl_q)
            # pl_q is typically 0.5 in legacy code
            pl_q = getattr(self, 'pl_q', 0.5)  # Default matches legacy
            V_ref = 1e-18  # Reference volume (typical particle size)
            rate = self.p1 * (self.g ** self.p2) * ((v_particle / V_ref) ** pl_q)
        else:
            # Default to constant rate
            rate = self.p1
        
        return max(0.0, float(rate))
