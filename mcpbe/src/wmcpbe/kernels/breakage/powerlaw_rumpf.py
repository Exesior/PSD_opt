"""
PowerLaw-Rumpf Breakage Kernel.

Breakage rate model incorporating particle strength σ as function of
porosity and saturation, based on Rumpf's theory for granule strength.

Formula:
    S(V,poro,saturation) = S_base(V) / σ(poro, saturation)

Where:
    - S_base: Breakage rate from :mod:`._base_rate` (BREAKRVAL 1-4, identical
      to pbe_core.func.jit_kernel_break.calc_break_rate_1d). For BREAKRVAL=3/4
      this is P1 × G × V^P2.
    - P1: Pre-factor, P2: Volume exponent
    - G: Shear rate [1/s], enters linearly
    - V: Particle volume [m³]
    - σ: Particle strength [Pa], depends on porosity and saturation

The 1/σ correction is a wet-granulation extension of this branch; the base
rate S_base itself is bit-for-bit the reference formula.

Particle Strength Model (Rumpf Theory, 3 Saturation Regimes):
    
    For S < 0.3 (dry regime):
        σ = (1-poro)/poro × k × γ / x_s
    
    For S > 0.8 (wet regime):
        σ = 6×α × (1-poro)/poro × γ×cos(δ) / x_s × S
    
    For 0.3 ≤ S ≤ 0.8 (transition):
        Linear interpolation between σ(S=0.3) and σ(S=0.8)

Parameters:
    - k: Fitting parameter dry regime [2.2-2.8] (default: 2.5)
    - alpha: Fitting parameter wet regime [1.0-1.33] (default: 1.15)
    - gamma: Surface tension of binder [N/m] (default: 0.072, water)
    - delta: Contact angle [rad] (default: 0.0, perfect wetting)
    - x_s: Sauter diameter [m] (computed from solver.X0 if not provided)

Note on Rumpf Theory:
    The particle strength model is based on Rumpf's theory for tensile
    strength of agglomerates, extended with saturation-dependent terms
    for capillary bridge forces.

Note on x_s:
    The Sauter diameter is computed from the initial particle size distribution
    in solver.X0. For monodisperse spheres, x_s equals the particle diameter.
    
    IMPORTANT: If X0 contains a distribution (multiple different values), the
    Sauter mean diameter is calculated as:
        x_s = Σ(n_i × d_i³) / Σ(n_i × d_i²)
    
    This assumes all particles contribute equally. For highly polydisperse
    systems, this may not accurately represent the true binder distribution.

Special Cases:
    - poro = NaN (Vollkörper) or poro <= 0: rate = 0.0. No pores means no
      capillary/pore-wall failure mechanism, and it is the limit of the formula
      itself (σ → ∞ as ε → 0). S(ε) is continuous and monotonically increasing
      on the whole of [0, 1].
    - poro > poro_max (default 0.9999): clamped to poro_max, so σ stays finite.
    - saturation = NaN: Treat as dry (S = 0.0)
    - sigma = 0 or inf: unreachable (see poro_max and the delta < π/2 check);
      guarded as rate = 0.0.

Example:
    >>> kernel = PowerLawRumpfBreakageKernel(
    ...     p1=3e-2, p2=1.0, g=1000,
    ...     k=2.5, alpha=1.15, gamma=0.072
    ... )
    >>> rate = kernel.compute_rate(v_particle=1e-18, particle_idx=0, solver=solver)
"""

import numpy as np
from ..base import BreakageKernel
from ._base_rate import (
    compute_base_rate,
    compute_base_rate_array,
    validate_rate_params,
)

# Try to import numba for JIT acceleration
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba not available
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


# =============================================================================
# JIT-compiled helper functions
# =============================================================================

# NOTE: deliberately WITHOUT fastmath. Two reasons:
#   1. `fastmath` let the compiler reassociate the products below, so this
#      function disagreed with its own `.py_func` -- and with the vectorised
#      `compute_rate_array` -- by about 5e-16 relative. Without it, scalar and
#      vector path agree exactly and their parity can be asserted as such.
#   2. The `saturation != saturation` NaN test relies on NaN semantics that
#      `fastmath` is formally allowed to assume away (cf. the same class of
#      problem in `_base_rate.py`).
# The cost is negligible: the hot path is `compute_rate_array`, which is pure
# numpy; this scalar helper only runs on single-particle incremental updates.
@njit(cache=True)
def _compute_sigma_jit(poro: float, saturation: float, x_s: float,
                       k: float, alpha: float, gamma: float, cos_delta: float,
                       sat_dry: float, sat_wet: float, sat_width: float) -> float:
    """
    JIT-compiled particle strength calculation.
    
    Returns sigma (particle strength) based on porosity and saturation.
    """
    # Handle NaN saturation → treat as dry
    if saturation != saturation:  # NaN check
        saturation = 0.0
    
    # Clamp saturation
    if saturation < 0.0:
        saturation = 0.0
    elif saturation > 1.0:
        saturation = 1.0
    
    # Common factor
    poro_factor = (1.0 - poro) / poro
    base_factor = poro_factor * gamma / x_s
    
    # Three regimes
    if saturation < sat_dry:
        # Dry regime
        sigma = base_factor * k
    elif saturation > sat_wet:
        # Wet regime
        sigma = 6.0 * alpha * base_factor * cos_delta * saturation
    else:
        # Transition: linear interpolation
        sigma_dry = base_factor * k
        sigma_wet = 6.0 * alpha * base_factor * cos_delta * sat_wet
        t = (saturation - sat_dry) / sat_width
        sigma = (1.0 - t) * sigma_dry + t * sigma_wet
    
    # Avoid division by zero
    if sigma < 1e-30:
        sigma = 1e-30
    
    return sigma


# =============================================================================
# Kernel Class
# =============================================================================

class PowerLawRumpfBreakageKernel(BreakageKernel):
    """
    PowerLaw-Rumpf breakage rate kernel.
    
    Extends the classical power-law model with a reciprocal dependence
    on particle strength σ, based on Rumpf's theory for granule strength.
    σ varies with porosity and saturation.
    
    Physical Background:
        - Dry particles (S < 0.3): Strength from capillary bridges at contact points
        - Wet particles (S > 0.8): Strength from liquid film viscosity and surface tension
        - Transition (0.3 ≤ S ≤ 0.8): Linear interpolation between regimes
    
    References:
        - Rumpf, H., "Grundlagen der Agglomeration", 1979
        - Capillary bridge theory: Rabinovich et al., Langmuir 2005
        - Granule strength: Pietsch, "Agglomeration Processes", 2002
        - Saturation regimes: Iveson et al., Powder Technology 2001
    
    Parameters:
        p1: Pre-factor [1/s·m^(-3*P2)·Pa] (default: 3e-2)
        p2: Volume exponent (default: 1.0)
        g: Shear rate [1/s] (default: 1000)
        breakrval: Breakage model variant 1-4 (default: 4)
                   4: Power law S = P1 × G × V^P2
        k: Fitting parameter dry regime [2.2-2.8] (default: 2.5)
        alpha: Fitting parameter wet regime [1.0-1.33] (default: 1.15)
        gamma: Surface tension [N/m] (default: 0.072 for water)
        delta: Contact angle [rad] (default: 0.0)
        x_s: Sauter diameter [m] (default: None, computed from solver.X0)
    
    Example:
        >>> kernel = PowerLawRumpfBreakageKernel(
        ...     p1=3e-2, p2=1.0, g=1000, breakrval=4,
        ...     k=2.5, alpha=1.15, gamma=0.072, delta=0.0
        ... )
    """
    
    # =========================================================================
    # Constants for saturation regimes
    # =========================================================================
    SAT_DRY_THRESHOLD = 0.3       # S < 0.3: dry regime
    SAT_WET_THRESHOLD = 0.8       # S > 0.8: wet regime
    SAT_TRANSITION_WIDTH = 0.5    # 0.8 - 0.3 = 0.5
    
    @property
    def name(self) -> str:
        return 'powerlaw_rumpf'
    
    def get_default_params(self) -> dict:
        return {
            # PowerLaw base parameters
            'p1': 3e-2,           # Pre-factor
            'p2': 1.0,            # Shear exponent
            'g': 1000.0,          # Shear rate [1/s]
            'breakrval': 4,       # Breakage model variant

            # Strength model parameters
            'k': 2.5,             # Fitting parameter dry [2.2-2.8]
            'alpha': 1.15,        # Fitting parameter wet [1.0-1.33]
            'gamma': 0.072,       # Surface tension [N/m] (water at 20°C)
            'delta': 0.0,         # Contact angle [rad] (0 = perfect wetting)
            'x_s': None,          # Sauter diameter [m] (None = compute from X0)

            # Upper bound for the porosity entering sigma. Kept identical to
            # ConeModelKernel.PORO_MAX so the two never disagree about where the
            # physically meaningful range ends. With cone_model active this
            # clamp therefore never fires; it guards against the porosity
            # kernels that do allow eps == 1.0 exactly (volume_mixing,
            # incomplete_mixing, blueprint).
            'poro_max': 0.9999,
        }
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache PowerLaw values
        self.p1 = float(self.params['p1'])
        self.p2 = float(self.params['p2'])
        self.g = float(self.params['g'])
        self.breakrval = int(self.params['breakrval'])

        # Cache strength model values
        self.k = float(self.params['k'])
        self.alpha = float(self.params['alpha'])
        self.gamma = float(self.params['gamma'])
        self.delta = float(self.params['delta'])
        self.x_s = self.params.get('x_s')  # May be None, computed later
        self.poro_max = float(self.params['poro_max'])
        
        # Pre-compute cos(delta) for efficiency
        self.cos_delta = np.cos(self.delta)
        
        # Placeholder for computed Sauter diameter
        self._computed_x_s = None
    
    def validate_params(self, params: dict) -> dict:
        """Validate parameters with physical bounds."""
        # Required PowerLaw parameters
        required = ['p1', 'p2', 'g']
        for key in required:
            if key not in params:
                raise ValueError(
                    f"Missing required parameter '{key}' for powerlaw_rumpf breakage kernel."
                )
        
        # Validate PowerLaw values
        if params['p1'] < 0:
            raise ValueError(f"p1 must be non-negative, got {params['p1']}")
        if params['g'] < 0:
            raise ValueError(f"g must be non-negative, got {params['g']}")
        
        # Validate breakrval and reject breakage-FUNCTION parameters
        validate_rate_params(params, self.name)

        # Validate strength model parameters
        k = params.get('k', 2.5)
        if not (2.2 <= k <= 2.8):
            raise ValueError(
                f"k must be in range [2.2, 2.8], got {k}. "
                "This fitting parameter represents capillary bridge geometry."
            )
        
        alpha = params.get('alpha', 1.15)
        if not (1.0 <= alpha <= 1.33):
            raise ValueError(
                f"alpha must be in range [1.0, 1.33], got {alpha}. "
                "This fitting parameter accounts for liquid film effects."
            )
        
        gamma = params.get('gamma', 0.072)
        if gamma <= 0:
            raise ValueError(f"gamma (surface tension) must be positive, got {gamma}")
        
        delta = params.get('delta', 0.0)
        # Strictly below pi/2: at exactly pi/2 cos(delta) == 0, so the wet-regime
        # strength sigma = 6*alpha*base_factor*cos(delta)*S collapses to 0 for
        # ANY porosity -- and S = S_base/sigma would diverge. This is the only
        # remaining route to sigma == 0 now that the porosity is clamped.
        if not (0 <= delta < np.pi/2):
            raise ValueError(
                f"delta (contact angle) must be in [0, π/2), got {delta}. "
                "delta = π/2 gives cos(delta) = 0 and thus zero granule "
                "strength in the wet regime, i.e. an infinite breakage rate. "
                "For perfect wetting, use delta=0."
            )

        poro_max = params.get('poro_max', 0.9999)
        if not (0.0 < poro_max < 1.0):
            raise ValueError(
                f"poro_max must be in (0, 1), got {poro_max}. It caps the "
                "porosity entering the strength model; sigma -> 0 as eps -> 1."
            )

        # x_s validation deferred until compute time (may be computed from X0)

        return params
    
    def _compute_sauter_diameter(self, solver) -> float:
        """
        Compute Sauter mean diameter from solver.X0.
        
        Sauter diameter: d32 = Σ(n_i × d_i³) / Σ(n_i × d_i²)
        
        For monodisperse systems, this equals the single particle diameter.
        For polydisperse systems, it's the volume-surface mean diameter.
        
        Args:
            solver: Reference to solver instance
        
        Returns:
            x_s: Sauter diameter [m]
        
        Raises:
            ValueError: If X0 is not available or invalid
        """
        if not hasattr(solver, 'X0') or solver.X0 is None:
            raise ValueError(
                "Solver has no X0 attribute. Cannot compute Sauter diameter. "
                "Please provide x_s parameter explicitly."
            )
        
        X0 = np.asarray(solver.X0, dtype=float).ravel()
        
        # Filter valid diameters
        valid = np.isfinite(X0) & (X0 > 0)
        if not np.any(valid):
            raise ValueError(
                "solver.X0 contains no valid (positive, finite) diameters. "
                "Cannot compute Sauter diameter."
            )
        
        X0_valid = X0[valid]
        
        # Check if monodisperse (all values equal within tolerance)
        if X0_valid.size == 1:
            # Single value - monodisperse
            return float(X0_valid[0])
        
        # Check if effectively monodisperse (variance very small)
        mean_d = np.mean(X0_valid)
        std_d = np.std(X0_valid)
        if std_d / mean_d < 1e-6:
            # Effectively monodisperse
            return float(mean_d)
        
        # Polydisperse: Compute Sauter mean diameter d32
        # d32 = Σ(d_i³) / Σ(d_i²)  (assuming equal number weighting)
        d2_sum = np.sum(X0_valid ** 2)
        d3_sum = np.sum(X0_valid ** 3)
        
        if d2_sum <= 0:
            raise ValueError(
                "Sum of squared diameters is zero or negative. "
                "Cannot compute Sauter diameter."
            )
        
        x_s = d3_sum / d2_sum
        
        # Store for reuse
        self._computed_x_s = x_s
        
        return x_s
    
    def _get_x_s(self, solver) -> float:
        """
        Get Sauter diameter, either from parameter or computed from X0.
        
        Args:
            solver: Reference to solver instance
        
        Returns:
            x_s: Sauter diameter [m]
        """
        # If x_s was provided as parameter, use it
        if self.x_s is not None:
            x_s_val = float(self.x_s)
            if x_s_val <= 0:
                raise ValueError(f"x_s must be positive, got {x_s_val}")
            return x_s_val
        
        # Otherwise compute from solver.X0
        if self._computed_x_s is not None:
            return self._computed_x_s
        
        return self._compute_sauter_diameter(solver)
    
    def _compute_sigma(self, poro: float, saturation: float, x_s: float) -> float:
        """
        Compute particle strength σ based on porosity and saturation.
        
        Uses JIT-compiled helper function for performance.
        
        Args:
            poro: Porosity [0, 1] (must be valid, not NaN)
            saturation: Saturation [0, 1] (NaN treated as 0.0)
            x_s: Sauter diameter [m]
        
        Returns:
            sigma: Particle strength [Pa]
        """
        return _compute_sigma_jit(
            poro, saturation, x_s,
            self.k, self.alpha, self.gamma, self.cos_delta,
            self.SAT_DRY_THRESHOLD, self.SAT_WET_THRESHOLD, self.SAT_TRANSITION_WIDTH
        )
    
    def _compute_base_rate_powerlaw(self, v_particle: float) -> float:
        """
        Compute base breakage rate using PowerLaw model (without σ correction).
        
        Uses JIT-compiled helper function for performance.
        
        Args:
            v_particle: Particle volume [m³]
        
        Returns:
            base_rate: PowerLaw breakage rate [1/s]
        """
        return compute_base_rate(
            v_particle, self.p1, self.p2, self.g, self.breakrval
        )
    
    def compute_rate(self, v_particle: float,
                     particle_idx: int = None,
                     solver = None) -> float:
        """
        Compute breakage rate with porosity/saturation-dependent strength.

        Formula: S(V) = S_base(V) / σ(poro, saturation)

        Special cases:
        - poro = NaN (Vollkörper) or poro <= 0: rate = 0.0 (no pores -> no break)
        - poro > poro_max: clamped to poro_max (keeps σ finite, S(ε) continuous)
        - saturation = NaN: Treat as dry (S = 0.0)
        - No solver/particle_idx, or x_s unavailable: Use base PowerLaw rate.
          There the σ correction is not *applicable*, which is a different
          situation from ε = 0, where it is applicable and evaluates to zero.

        Args:
            v_particle: Particle volume [m³]
            particle_idx: Index of particle in solver arrays
            solver: Reference to solver for accessing porosity/saturation
        
        Returns:
            rate: Breakage rate [1/s]
        """
        if v_particle <= 0:
            return 0.0
        
        # Compute base PowerLaw rate
        base_rate = self._compute_base_rate_powerlaw(v_particle)
        
        # If no solver or particle index, return base rate (no σ correction)
        if solver is None or particle_idx is None:
            return base_rate
        
        # Try to access porosity and saturation
        try:
            poro = solver.porosity[particle_idx]
        except (AttributeError, IndexError, TypeError):
            # Solver has no porosity attribute or invalid index
            return base_rate
        
        try:
            saturation = solver.saturation[particle_idx]
        except (AttributeError, IndexError, TypeError):
            # Solver has no saturation attribute
            saturation = 0.0  # Default to dry
        
        # Vollkoerper (poro = NaN) and pore-free particles (poro <= 0) have no
        # pores, hence no capillary/pore-wall failure mechanism: they do not
        # break. sigma = (1-eps)/eps * ... diverges as eps -> 0, so S = S_base/
        # sigma -> 0; returning 0.0 IS the limit of the formula.
        #
        # This used to `return base_rate`, which is not "the uncorrected case"
        # but implicitly sigma = 1 Pa -- an absurdly weak solid. It made the
        # pore-free (i.e. strongest) body the most fragile one in the whole
        # population, inverting the physics and putting a jump of many orders of
        # magnitude right at eps = 0.
        if np.isnan(poro) or poro <= 0.0:
            return 0.0

        # Upper edge: clamp instead of falling back. sigma stays finite and
        # S(eps) stays continuous. Previously `poro >= 1` returned base_rate,
        # which was a jump *downwards* by orders of magnitude exactly where the
        # granule is weakest.
        if poro > self.poro_max:
            poro = self.poro_max

        # Get Sauter diameter
        try:
            x_s = self._get_x_s(solver)
        except (ValueError, AttributeError) as e:
            # Cannot compute x_s → fallback to base rate
            # Log warning if verbose mode available
            if hasattr(solver, 'VERBOSE') and solver.VERBOSE:
                print(f"[powerlaw_rumpf] Warning: {e}, using base rate")
            return base_rate
        
        # Compute particle strength σ
        sigma = self._compute_sigma(poro, saturation, x_s)
        
        # Apply reciprocal correction: rate ∝ 1/σ
        # Higher strength → lower breakage rate
        if np.isfinite(sigma) and sigma > 1e-30:
            rate = base_rate / sigma
        else:
            # Unreachable since the porosity clamp above bounds sigma from
            # below and validate_params rules out cos(delta) == 0. Kept as a
            # division guard. Returns 0.0, not base_rate: a non-finite or
            # vanishing sigma means "infinitely strong", i.e. no breakage --
            # falling back to base_rate here would reintroduce a silent regime
            # switch of exactly the kind removed above.
            rate = 0.0

        return max(0.0, float(rate))

    def compute_rate_array(self, v_particles: np.ndarray,
                           solver = None) -> np.ndarray:
        """
        Vectorised form of :meth:`compute_rate` for a whole particle slice.

        Without this override the base class loops in Python over every
        particle on every event (`kernels/base.py`). σ is a pure function of
        ``(poro, saturation, x_s, …)``, so the whole slice can be done at once.

        Bit-identical to the scalar path by construction: every branch uses the
        same expressions in the same association order as
        :func:`_compute_sigma_jit`, and the branches are selected with
        ``np.where`` instead of ``if``. The parity is asserted by the
        verification suite, so keep the two in sync when changing either.

        Args:
            v_particles: Particle volumes (V_dry) [m³], shape (n,)
            solver: Reference to solver for porosity/saturation

        Returns:
            rates: Breakage rates [1/s], shape (n,)
        """
        v = np.asarray(v_particles, dtype=float)
        base = compute_base_rate_array(v, self.p1, self.p2, self.g,
                                       self.breakrval)

        # No context -> sigma correction not applicable (mirrors the scalar path)
        if solver is None or not hasattr(solver, 'porosity'):
            return base

        n = v.shape[0]
        poro = np.asarray(solver.porosity[:n], dtype=float)
        if hasattr(solver, 'saturation'):
            sat = np.asarray(solver.saturation[:n], dtype=float)
        else:
            sat = np.zeros(n, dtype=float)

        try:
            x_s = self._get_x_s(solver)
        except (ValueError, AttributeError) as e:
            if getattr(solver, 'VERBOSE', False):
                print(f"[powerlaw_rumpf] Warning: {e}, using base rate")
            return base

        # Vollkoerper (NaN) and pore-free particles do not break; everything
        # else is clamped to poro_max so sigma stays finite.
        valid = np.isfinite(poro) & (poro > 0.0)
        # Dummy 0.5 on the invalid entries keeps the arithmetic below free of
        # division-by-zero warnings; those entries are masked out at the end.
        poro_c = np.where(valid, np.minimum(poro, self.poro_max), 0.5)

        # Saturation: NaN -> dry, then clamp to [0, 1] (as in _compute_sigma_jit)
        sat = np.where(np.isnan(sat), 0.0, sat)
        sat = np.clip(sat, 0.0, 1.0)

        poro_factor = (1.0 - poro_c) / poro_c
        base_factor = poro_factor * self.gamma / x_s

        sigma_dry = base_factor * self.k
        sigma_wet = 6.0 * self.alpha * base_factor * self.cos_delta * sat
        sigma_wet_at_thr = (6.0 * self.alpha * base_factor * self.cos_delta
                            * self.SAT_WET_THRESHOLD)
        t = (sat - self.SAT_DRY_THRESHOLD) / self.SAT_TRANSITION_WIDTH
        sigma_trans = (1.0 - t) * sigma_dry + t * sigma_wet_at_thr

        sigma = np.where(
            sat < self.SAT_DRY_THRESHOLD, sigma_dry,
            np.where(sat > self.SAT_WET_THRESHOLD, sigma_wet, sigma_trans),
        )
        sigma = np.maximum(sigma, 1e-30)

        # `sigma > 1e-30` strictly, exactly like the scalar guard: a sigma that
        # had to be lifted onto the floor yields rate 0.0, not base/1e-30.
        ok = valid & np.isfinite(sigma) & (sigma > 1e-30)
        rate = np.where(ok, base / sigma, 0.0)
        return np.maximum(rate, 0.0)
