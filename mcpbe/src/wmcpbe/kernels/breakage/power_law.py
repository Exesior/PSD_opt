"""
Power-Law Breakage Kernel.

Classical breakage rate model where rate scales with particle size.

Formula (BREAKRVAL=3, 4):
    S(V) = P1 × G × V^P2

Where:
    - P1: Pre-factor
    - P2: Volume exponent
    - G: Shear rate [1/s], enters linearly
    - V: Particle volume [m³]

The BREAKRVAL branches live in :mod:`._base_rate` and mirror
`pbe_core.func.jit_kernel_break.calc_break_rate_1d` exactly.
"""

from ..base import BreakageKernel
from ._base_rate import (
    compute_base_rate,
    compute_base_rate_array,
    validate_rate_params,
)


class PowerLawBreakageKernel(BreakageKernel):
    """
    Power-law breakage rate kernel.
    
    This is the classical breakage model used in most PBE simulations.
    The breakage rate increases with particle size and shear rate.
    
    Parameters:
        p1: Pre-factor [1/s·m^(-3*P2)] (default: 3e-2)
        p2: Volume exponent (default: 1.0)
        g: Shear rate [1/s] (default: 1000.0)
        breakrval: Breakage model variant (default: 1)
                   1: Constant rate
                   2: Linear in volume
                   3: Power law (Pandy & Spielmann)
                   4: Power law (identical to 3 in 1D)

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
            'p2': 1.0,       # Volume exponent
            'g': 1000.0,     # Shear rate [1/s]
            'breakrval': 1,  # Breakage model variant (1-4)
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
        
        # Validate breakrval and reject breakage-FUNCTION parameters
        validate_rate_params(params, self.name)

        return params
    
    def compute_rate(self, v_particle: float,
                     particle_idx: int = None,
                     solver = None) -> float:
        """
        Compute breakage rate using power-law model.
        
        CRITICAL: This must match the reference JIT implementation EXACTLY.
        The JIT function is: pbe_core.func.jit_kernel_break.calc_break_rate_1d

        Supports all BREAKRVAL variants (1-4):
            BREAKRVAL=1: S = P1 (constant)
            BREAKRVAL=2: S = P1 × V (linear in volume)
            BREAKRVAL=3: S = P1 × G × V^P2 (Pandy & Spielmann)
            BREAKRVAL=4: S = P1 × G × V^P2 (identical to 3 in 1D)

        Args:
            v_particle: Particle volume [m³]
            particle_idx: Not used (for interface compatibility)
            solver: Not used (for interface compatibility)

        Returns:
            rate: Breakage rate [1/s]
        """
        return float(compute_base_rate(
            float(v_particle), self.p1, self.p2, self.g, self.breakrval
        ))

    def compute_rate_array(self, v_particles, solver=None):
        """
        Vectorised form of :meth:`compute_rate`.

        The power-law rate depends only on the particle volume, so the whole
        slice can be evaluated with a handful of numpy operations instead of
        one Python call per particle. Same expressions, same order, so results
        match :meth:`compute_rate` element for element.
        """
        return compute_base_rate_array(
            v_particles, self.p1, self.p2, self.g, self.breakrval
        )
