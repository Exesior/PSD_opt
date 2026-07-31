"""
Liquid Internalization Kernel.

Model for capillary-driven liquid internalization during wet granulation.

Formula:
    dl_intern/dt = k × l_ex × (v_pore - l_intern)
    
    where:
        - l_intern: Internal liquid volume [m³]
        - l_ex: External liquid volume [m³] = l_total - l_intern
        - v_pore: Pore volume [m³]
        - k: Rate constant [1/(m³·s)]

Physical Background:
    This model describes the penetration of liquid binder into particle pores
    driven by capillary forces. The rate depends on:
    
    1. Available external liquid (l_ex): More external liquid → faster uptake
    2. Empty pore space (v_pore - l_intern): More empty pores → faster uptake
    
    The process saturates when:
    - All liquid is internalized (l_ex = 0), OR
    - Pores are completely filled (l_intern = v_pore, i.e., S = 1)

Integration:
    For small time steps dt, the change in internal liquid is:
    
    Δl_intern = k × l_ex × (v_pore - l_intern) × dt
    
    Updated saturation:
    S_new = l_intern_new / v_pore

References:
    [1] Braumann et al., "Modelling and validation of granulation with 
        heterogeneous binder dispersion and chemical reaction", 2007.
        Equation: r = k * l_ex * (v_pore - l_intern)

Integration with Nucleation:
    When this kernel is active, newly nucleated particles start with ALL liquid
    in the external state (saturation = 0). The internalization process then
    occurs naturally over time via the continuous process operator splitting.
    
    If this kernel is NOT active, the nucleation handler uses an empirical
    40/60 split (40% internal, 60% external) as fallback behavior.

Usage:
    >>> kernel = LiquidInternalizationKernel(k_int=1e12)
    >>> s_new = kernel.compute(saturation=0.3, v_pore=1e-18, l_total=5e-19, dt=0.01)
"""

import numpy as np
from ..base import LiquidInternalizationKernel


class LiquidInternalizationKernel(LiquidInternalizationKernel):
    """
    Capillary-driven liquid internalization model.
    
    Models the time-dependent penetration of liquid binder into particle pores.
    The rate is proportional to both available external liquid and empty pore space.
    
    Parameters:
        k_int: Internalization rate constant [1/(m³·s)] (default: 1e12)
               Higher values → faster liquid uptake
               Typical range: 1e10 - 1e14 1/(m³·s) depending on material properties
    
    Example:
        >>> kernel = LiquidInternalizationKernel(k_int=1e12)
        >>> # Particle with 30% saturated pores
        >>> s_new = kernel.compute(
        ...     saturation=0.3,
        ...     v_pore=1e-18,      # 1 µm³ pore volume
        ...     l_total=3e-19,     # Total liquid = initial internal
        ...     dt=0.01            # 10 ms time step
        ... )
        >>> print(f"New saturation: {s_new:.3f}")
    """
    
    @property
    def name(self) -> str:
        return 'liquid_internalization'
    
    def get_default_params(self) -> dict:
        return {
            'k_int': 1e12,  # 1/(m³·s), typical for porous granules
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache values
        self.k_int = float(self.params['k_int'])
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['k_int'] < 0:
            raise ValueError("k_int must be non-negative")
        if params['k_int'] == 0:
            # Warning: zero rate means no internalization
            pass  # Allow but user should be aware
        return params
    
    def compute(self,
                saturation: float,
                v_pore: float,
                l_total: float,
                dt: float,
                solver = None
                ) -> float:
        """
        Compute new saturation after liquid internalization over time dt.
        
        Uses exact analytical solution of the ODE over each time step.
        
        ODE: dl_intern/dt = k × (l_total - l_intern) × (v_pore - l_intern)
        
        Analytical solution for l_intern(t+dt) given l_intern(t):
        l_new = [l_total×(v_pore-l)×exp(α×dt) - v_pore×(l_total-l)] 
                / [(v_pore-l)×exp(α×dt) - (l_total-l)]
        
        where α = k × (v_pore - l_total)
        
        Args:
            saturation: Current saturation S = l_intern / v_pore (0 to 1)
            v_pore: Pore volume [m³]
            l_total: Total liquid volume [m³] (conserved during internalization)
            dt: Time step [s]
            solver: Not used
        
        Returns:
            saturation_new: Updated saturation (0 to 1)
        """
        # Edge cases
        if v_pore <= 0 or not np.isfinite(v_pore):
            return saturation
        if l_total <= 0 or not np.isfinite(l_total):
            return 0.0
        
        saturation = max(0.0, min(1.0, saturation))
        
        if saturation >= 1.0 or self.k_int <= 0:
            return saturation
        
        # Convert saturation to internal liquid volume
        l_intern = saturation * v_pore
        
        # Rate factor: α = k × (v_pore - l_total)
        alpha = self.k_int * (v_pore - l_total)
        
        # Exponential term
        exp_term = np.exp(alpha * dt)
        
        # Coefficients from current state
        # A = empty pore space = v_pore - l_intern
        # B = available external liquid = l_total - l_intern
        
        # Analytical solution (exact ODE solution over dt):
        # l_new = [v_pore × B - l_total × A × exp(α×dt)] / [B - A × exp(α×dt)]
        # where A = (v_pore - l_intern), B = (l_total - l_intern)
        
        A = v_pore - l_intern
        B = l_total - l_intern
        
        numerator = v_pore * B - l_total * A * exp_term
        denominator = B - A * exp_term
        
        # Compute new internal liquid volume
        # Use try-except to handle division by zero safely
        try:
            l_intern_new = numerator / denominator
            # Check for non-finite results (NaN, Inf) from division by ~0
            if not np.isfinite(l_intern_new):
                l_intern_new = min(l_total, v_pore)
        except (ZeroDivisionError, FloatingPointError):
            # Denominator is effectively zero - at equilibrium
            l_intern_new = min(l_total, v_pore)
        
        # Clamp to physical bounds
        l_max = min(l_total, v_pore)
        l_intern_new = max(0.0, min(l_intern_new, l_max))
        
        # Convert back to saturation
        saturation_new = l_intern_new / v_pore
        
        return float(saturation_new)
    
    def compute_array(self,
                      saturation: np.ndarray,
                      v_pore: np.ndarray,
                      l_total: np.ndarray,
                      dt: float,
                      ) -> np.ndarray:
        """Vectorised form of :meth:`compute` over whole particle arrays.

        Evaluates exactly the same analytical ODE solution, term for term, so
        that a batch call and a loop of scalar calls agree. Used by
        ``ContinuousProcessesHandler`` to replace an O(n) Python loop that ran
        once per Monte-Carlo event.

        Args:
            saturation: Current saturations S = l_intern / v_pore
            v_pore:     Pore volumes [m³]
            l_total:    Total liquid volumes [m³]
            dt:         Time step [s]

        Returns:
            Updated saturations, same shape as the inputs.
        """
        saturation = np.asarray(saturation, dtype=float)
        v_pore = np.asarray(v_pore, dtype=float)
        l_total = np.asarray(l_total, dtype=float)

        out = saturation.copy()

        # v_pore invalid -> saturation unchanged; no liquid -> saturation 0.
        pore_ok = (v_pore > 0) & np.isfinite(v_pore)
        liquid_ok = (l_total > 0) & np.isfinite(l_total)
        out[pore_ok & ~liquid_ok] = 0.0

        sat = np.clip(saturation, 0.0, 1.0)
        active = pore_ok & liquid_ok & (sat < 1.0) & (self.k_int > 0)
        # Already-saturated particles keep their clipped saturation.
        clipped_only = pore_ok & liquid_ok & ~active
        out[clipped_only] = sat[clipped_only]

        if not np.any(active):
            return out

        vp = v_pore[active]
        lt = l_total[active]
        l_intern = sat[active] * vp

        alpha = self.k_int * (vp - lt)
        exp_term = np.exp(alpha * dt)

        A = vp - l_intern
        B = lt - l_intern

        numerator = vp * B - lt * A * exp_term
        denominator = B - A * exp_term

        l_max = np.minimum(lt, vp)
        with np.errstate(divide="ignore", invalid="ignore"):
            l_new = numerator / denominator
        # Division by ~0 means the pair is at equilibrium.
        l_new = np.where(np.isfinite(l_new), l_new, l_max)
        l_new = np.maximum(0.0, np.minimum(l_new, l_max))

        out[active] = l_new / vp
        return out

    def compute_with_solver(self, particle_idx: int, dt: float, solver) -> float:
        """
        Convenience method to compute internalization using solver state.
        
        Extracts all required values from solver arrays for a given particle.
        
        Args:
            particle_idx: Index of particle in solver arrays
            dt: Time step [s]
            solver: Solver instance with liquid_volume, porosity, saturation arrays
        
        Returns:
            saturation_new: Updated saturation
        """
        # Get current state from solver
        saturation = solver.saturation[particle_idx]
        porosity = solver.porosity[particle_idx]
        v_dry = solver.V_flat[-1, particle_idx]
        l_total = solver.liquid_volume[particle_idx]
        
        # Skip Vollkörper (no pores)
        if np.isnan(porosity) or porosity <= 0:
            return saturation
        
        # Calculate pore volume: v_pore = v_dry × porosity
        v_pore = v_dry * porosity
        
        # Delegate to main compute method
        return self.compute(saturation, v_pore, l_total, dt)
