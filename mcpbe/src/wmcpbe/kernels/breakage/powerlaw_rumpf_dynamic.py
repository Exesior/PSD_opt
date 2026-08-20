"""
PowerLaw-Rumpf breakage kernel driven by the mixer speed.

Same breakage physics as :mod:`.powerlaw_rumpf` - a power-law base rate divided
by the Rumpf granule strength sigma(porosity, saturation) - but the driving
quantity is the mixer speed instead of a shear rate:

    S(V, eps, S_sat) = P1 * n_mixer^C_MIXER * V^P2 / sigma(eps, S_sat)

Self-contained:
    This file imports only the abstract base class, numpy, numba and the shared
    mixer-speed constants. It does NOT inherit from `powerlaw_rumpf` and does
    not import a shared rate helper: the base rate and the strength model are
    both computed here. Handing this one file (plus `base.py` and
    `mixer_speed.py`) to someone else gives a working kernel even if no other
    breakage kernel is present.

    The duplication against `powerlaw_rumpf` is intentional and guarded:
    `Trials/test_dry_mixer_kernels.py` asserts that this kernel and
    `powerlaw_rumpf` with `g = n_mixer**c_mixer` return bit-wise identical
    rates, so a change to one that is not carried over fails the suite.

Why not the shear rate:
    In `powerlaw_rumpf` the base rate is S_base = P1 * G * V^P2 (BREAKRVAL 3/4),
    where G is the velocity gradient of a continuous LIQUID phase. Here the
    continuum is a gas, so no such gradient exists to be measured; G could only
    ever be a fit parameter wearing a physical name.

    Note on the name: this kernel is NOT "dry". Wet agglomeration still happens,
    binder is still added, granules still have liquid bridges and saturation.
    What is dry is the CONTINUUM the granules move in.

Where C_MIXER comes from:
    G played the role of a STRESSING FREQUENCY - how often a granule is loaded
    per second - while the 1/sigma factor separately says how easily each
    loading breaks it. The mixer-speed analogue therefore has two parts:

        rate = (loadings per second) x P(break per loading)
             ~ n^0.0995            x  n^(2 * 0.2852)
             = n^0.6699

    The first factor is the DEM collision frequency, the second the impact
    energy (~ v_rel^2) from the DEM relative collision velocity. The default is
    ``mixer_speed.C_BREAK``; see that module for the data behind both exponents.

    Over the DEM range n = 5 to 40 this gives a factor 4.0 in breakage rate.

No mass term, and why that is a result rather than an omission:
    Under the EKE picture that drives the aggregation side, the fluctuating
    kinetic energy is equipartitioned: every granule carries the same E_0, so
    v' ~ m^(-1/2). The impact energy of a pair is then

        E_impact = 1/2 * mu * v_rel^2,   mu = m_i m_j / (m_i + m_j)
                 = 1/2 * mu * 2 E_0 (1/m_i + 1/m_j)
                 = E_0

    i.e. the mass cancels exactly and every collision delivers the same energy
    regardless of who collides with whom. Putting a mass dependence into the
    breakage energy would contradict the aggregation kernel it sits next to.

    A mass-dependent variant would mean adopting a different hypothesis about
    how the mixer shares its energy out:

        equal kinetic energy   v ~ m^-1/2   E ~ const   -> EKE  (this kernel)
        equal momentum         v ~ m^-1     E ~ 1/m     -> ETM
        equal speed            v ~ const    E ~ m       -> neither

    Such a variant belongs in its OWN kernel with its assumption stated. Be
    aware, too, that a mass exponent would be largely unidentifiable against the
    existing free volume exponent P2: with m = rho_s * V * (1 - eps), a factor
    m^c is P2-reparameterisation apart from the (1 - eps) part, and porosity
    already enters through sigma.

BREAKRVAL:
    The driving term only enters variants 3 and 4. With breakrval 1 (S = P1) or
    2 (S = P1 * V) the mixer speed has no effect here either - exactly as the
    shear rate had none in `powerlaw_rumpf`.

Particle strength (Rumpf theory, 3 saturation regimes):

    For S < 0.3 (dry regime):
        sigma = (1-eps)/eps * k * gamma / x_s

    For S > 0.8 (wet regime):
        sigma = 6*alpha * (1-eps)/eps * gamma*cos(delta) / x_s * S

    For 0.3 <= S <= 0.8 (transition):
        Linear interpolation between sigma(S=0.3) and sigma(S=0.8)

Special cases:
    - poro = NaN (Vollkoerper) or poro <= 0: rate = 0.0. No pores means no
      capillary/pore-wall failure mechanism, and it is the limit of the formula
      itself (sigma -> inf as eps -> 0).
    - poro > poro_max (default 0.9999): clamped, so sigma stays finite.
    - saturation = NaN: treated as dry (S = 0.0).
    - No solver/particle_idx, or x_s unavailable: base rate without the sigma
      correction. There the correction is not *applicable*, which differs from
      eps = 0, where it is applicable and evaluates to zero.

Example:
    >>> kernel = PowerLawRumpfDynamicBreakageKernel(
    ...     p1=4e13, p2=1.0, breakrval=4, n_mixer=20.0,
    ...     k=2.5, alpha=1.0, gamma=0.072, delta=0.0
    ... )
    >>> rate = kernel.compute_rate(v_particle=1e-18, particle_idx=0, solver=solver)
"""

import math

import numpy as np

from ..base import BreakageKernel
from ..mixer_speed import C_BREAK, N_MIXER_DEFAULT

# Try to import numba for JIT acceleration
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrapper(func):
            return func
        return wrapper


# =============================================================================
# Base rate S(V) - the BREAKRVAL switch
# =============================================================================
# This block is deliberately DUPLICATED in every breakage kernel instead of
# living in a shared module. A kernel has to work on its own: handing this one
# file to someone who does not have the rest of the package must still give a
# working kernel, so a kernel may not depend on another kernel or on a private
# helper module.
#
# The price is that the copies can drift apart. That is not left to discipline:
# `Trials/test_dry_mixer_kernels.py` asserts that every copy produces bit-wise
# identical rates for every BREAKRVAL, so a change to one of them that is not
# carried over to the others fails the suite.
#
# Mirrors `pbe_core.func.jit_kernel_break.calc_break_rate_1d` branch for branch:
#     BREAKRVAL=1: S = P1                (size independent, Leong2023 (10))
#     BREAKRVAL=2: S = P1 * V            (size dependent, Leong2023 (10))
#     BREAKRVAL=3: S = P1 * G * V^P2     (Pandy & Spielmann, Jeldres2018 (28))
#     BREAKRVAL=4: S = P1 * G * V^P2     (identical to 3 in 1D)
#
# P2 is the VOLUME exponent and G enters linearly. `pl_v` / `pl_q` belong to the
# breakage FUNCTION (fragment size distribution, BREAKFVAL), never to the rate,
# and are rejected below. There is deliberately no BREAKRVAL=5 - the reference
# implementation has no such branch.

_VALID_BREAKRVAL = (1, 2, 3, 4)


def _validate_rate_params(params: dict, kernel_name: str) -> None:
    """Reject invalid breakrval values and breakage-FUNCTION parameters.

    Raises a ValueError with a migration hint instead of silently ignoring
    them, so setups written against the old (divergent) formula fail loudly.
    """
    breakrval = params.get('breakrval')
    if breakrval == 5:
        raise ValueError(
            f"breakrval=5 is not a valid breakage rate model for "
            f"'{kernel_name}'. It never existed in the reference "
            f"implementation (pbe_core.func.jit_kernel_break."
            f"calc_break_rate_1d) and has been removed. "
            f"Valid values: {list(_VALID_BREAKRVAL)}."
        )
    if breakrval is not None and breakrval not in _VALID_BREAKRVAL:
        raise ValueError(
            f"Invalid breakrval={breakrval} for '{kernel_name}'. "
            f"Must be one of: {list(_VALID_BREAKRVAL)}"
        )

    for legacy in ('pl_v', 'pl_q'):
        if legacy in params:
            raise ValueError(
                f"Parameter '{legacy}' is not a breakage RATE parameter and is "
                f"no longer accepted by '{kernel_name}'. It belongs to the "
                f"breakage FUNCTION (fragment size distribution, BREAKFVAL). "
                f"Use 'p2' for the volume exponent of the rate, and set "
                f"'break_frag_v' / 'break_frag_q' on the solver to steer the "
                f"fragment distribution."
            )


@njit(cache=True, fastmath=True)
def _base_rate_scalar(v_particle: float, p1: float, p2: float, g: float,
                      breakrval: int) -> float:
    """Breakage rate S(V) for a single particle [1/s].

    Returns 0.0 for a non-positive volume, for NaN, and for an unknown
    breakrval.

    `not (v > 0.0)` rather than `v <= 0.0`: it treats NaN like a non-positive
    volume. With `fastmath=True` the compiler may assume NaN-freedom, so
    `v <= 0.0` let a NaN slip through into the arithmetic below - the JIT
    version then returned 0.0 while `.py_func` returned NaN.
    """
    if not (v_particle > 0.0):
        return 0.0

    if breakrval == 1:
        rate = p1
    elif breakrval == 2:
        rate = p1 * v_particle
    elif breakrval == 3:
        rate = p1 * g * v_particle ** p2
    elif breakrval == 4:
        rate = p1 * g * v_particle ** p2
    else:
        rate = 0.0

    if rate < 0.0:
        rate = 0.0

    return rate


def _base_rate_array(v_particles: np.ndarray, p1: float, p2: float, g: float,
                     breakrval: int) -> np.ndarray:
    """Vectorised form of :func:`_base_rate_scalar`.

    Same expressions in the same order, so results match element for element.
    `v > 0.0` is False for NaN, so NaN volumes fall into the same bucket as
    non-positive ones - deliberately identical to the scalar guard.
    """
    v = np.asarray(v_particles, dtype=float)
    positive = v > 0.0

    if breakrval == 1:
        rate = np.full(v.shape, p1, dtype=float)
    elif breakrval == 2:
        rate = p1 * v
    elif breakrval in (3, 4):
        rate = (p1 * g) * np.power(v, p2, where=positive, out=np.zeros_like(v))
    else:
        rate = np.zeros_like(v)

    return np.where(positive, np.maximum(rate, 0.0), 0.0)



# =============================================================================
# Particle strength sigma
# =============================================================================

# NOTE: deliberately WITHOUT fastmath. Two reasons:
#   1. `fastmath` let the compiler reassociate the products below, so this
#      function disagreed with its own `.py_func` -- and with the vectorised
#      `compute_rate_array` -- by about 5e-16 relative. Without it, scalar and
#      vector path agree exactly and their parity can be asserted as such.
#   2. The `saturation != saturation` NaN test relies on NaN semantics that
#      `fastmath` is formally allowed to assume away.
# The cost is negligible: the hot path is `compute_rate_array`, which is pure
# numpy; this scalar helper only runs on single-particle incremental updates.
@njit(cache=True)
def _sigma_jit(poro: float, saturation: float, x_s: float,
               k: float, alpha: float, gamma: float, cos_delta: float,
               sat_dry: float, sat_wet: float, sat_width: float) -> float:
    """JIT-compiled particle strength sigma [Pa] from porosity and saturation."""
    # Handle NaN saturation -> treat as dry
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
        sigma = base_factor * k
    elif saturation > sat_wet:
        sigma = 6.0 * alpha * base_factor * cos_delta * saturation
    else:
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

class PowerLawRumpfDynamicBreakageKernel(BreakageKernel):
    """
    PowerLaw-Rumpf breakage rate driven by the mixer speed instead of a shear
    rate.

    Parameters:
        p1: Pre-factor (default: 3e-2). Absorbs the DEM prefactor and the unit
            convention of `n_mixer`; NOT comparable to the `p1` of
            `powerlaw_rumpf`, because `g` (typically 1000) has been replaced by
            `n_mixer**c_mixer` (order 1-10).
        p2: Volume exponent (default: 1.0)
        n_mixer: Mixer speed (default: `mixer_speed.N_MIXER_DEFAULT` = 20.0).
                 Must match the `n_mixer` of every other mixer-speed-driven
                 kernel in the same run; `assert_consistent_mixer_speed`
                 enforces that at solver setup.
        c_mixer: Mixer speed exponent (default: `mixer_speed.C_BREAK` = 0.6699,
                 stressing frequency x impact energy). Set c_mixer=0.0 to
                 switch the mixer speed dependence off entirely.
        breakrval: Breakage model variant 1-4 (default: 4).
        k: Fitting parameter dry regime [2.2-2.8] (default: 2.5)
        alpha: Fitting parameter wet regime [1.0-1.33] (default: 1.15)
        gamma: Surface tension of binder [N/m] (default: 0.072, water)
        delta: Contact angle [rad] (default: 0.0, perfect wetting)
        x_s: Sauter diameter [m] (default: None, computed from solver.X0)
        poro_max: Upper bound for the porosity entering sigma (default 0.9999)

    Note:
        There is no `g` parameter. A caller who hands this kernel a shear rate
        gets a loud error from `reject_unknown_params` rather than a silently
        ignored value.
    """

    # =========================================================================
    # Constants for saturation regimes
    # =========================================================================
    SAT_DRY_THRESHOLD = 0.3       # S < 0.3: dry regime
    SAT_WET_THRESHOLD = 0.8       # S > 0.8: wet regime
    SAT_TRANSITION_WIDTH = 0.5    # 0.8 - 0.3 = 0.5

    @property
    def name(self) -> str:
        return 'powerlaw_rumpf_dynamic'

    def get_default_params(self) -> dict:
        return {
            # Base rate parameters. No 'g': the driving term is the mixer speed.
            'p1': 3e-2,
            'p2': 1.0,
            'n_mixer': N_MIXER_DEFAULT,
            'c_mixer': C_BREAK,
            'breakrval': 4,

            # Strength model parameters
            'k': 2.5,
            'alpha': 1.15,
            'gamma': 0.072,
            'delta': 0.0,
            'x_s': None,
            'poro_max': 0.9999,
        }

    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)

        # Mixer speed model
        self.n_mixer = float(self.params['n_mixer'])
        self.c_mixer = float(self.params['c_mixer'])

        # The driving term that replaces the shear rate. Kept under the name
        # `g` as well, so the base-rate helper reads the same argument as in
        # `powerlaw_rumpf` and the two can be compared directly - but it is NOT
        # a shear rate and carries no unit of 1/s.
        self.g = self.n_mixer ** self.c_mixer

        # Base rate values
        self.p1 = float(self.params['p1'])
        self.p2 = float(self.params['p2'])
        self.breakrval = int(self.params['breakrval'])

        # Strength model values
        self.k = float(self.params['k'])
        self.alpha = float(self.params['alpha'])
        self.gamma = float(self.params['gamma'])
        self.delta = float(self.params['delta'])
        self.x_s = self.params.get('x_s')
        self.poro_max = float(self.params['poro_max'])

        # Pre-compute cos(delta) for efficiency
        self.cos_delta = np.cos(self.delta)

        # Placeholder for computed Sauter diameter
        self._computed_x_s = None

    def validate_params(self, params: dict) -> dict:
        """Validate parameters with physical bounds."""
        required = ['p1', 'p2', 'n_mixer']
        for key in required:
            if key not in params:
                raise ValueError(
                    f"Missing required parameter '{key}' for "
                    f"powerlaw_rumpf_dynamic breakage kernel."
                )

        if params['p1'] < 0:
            raise ValueError(f"p1 must be non-negative, got {params['p1']}")

        if params['n_mixer'] <= 0:
            raise ValueError(
                f"n_mixer must be positive, got {params['n_mixer']}. "
                "A non-positive mixer speed makes n_mixer**c_mixer undefined "
                "for a fractional exponent. Use c_mixer=0.0 to disable the "
                "mixer speed dependence instead."
            )
        c_mixer = float(params['c_mixer'])
        if not math.isfinite(c_mixer):
            raise ValueError(f"c_mixer must be finite, got {c_mixer}")

        # Validate breakrval and reject breakage-FUNCTION parameters
        _validate_rate_params(params, self.name)

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
        # Strictly below pi/2: at exactly pi/2 cos(delta) == 0, so the
        # wet-regime strength collapses to 0 for ANY porosity and the rate
        # would diverge.
        if not (0 <= delta < np.pi / 2):
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

        return params

    # ------------------------------------------------------------------
    # Sauter diameter
    # ------------------------------------------------------------------
    def _compute_sauter_diameter(self, solver) -> float:
        """
        Compute Sauter mean diameter d32 = sum(n_i d_i^3) / sum(n_i d_i^2)
        from ``solver.X0``.

        For monodisperse spheres this equals the particle diameter.
        """
        if not hasattr(solver, 'X0') or solver.X0 is None:
            raise ValueError(
                "Solver has no X0 attribute. Cannot compute Sauter diameter. "
                "Please provide x_s parameter explicitly."
            )

        X0 = np.asarray(solver.X0, dtype=float).ravel()

        valid = np.isfinite(X0) & (X0 > 0)
        if not np.any(valid):
            raise ValueError(
                "solver.X0 contains no valid (positive, finite) diameters. "
                "Cannot compute Sauter diameter."
            )

        X0_valid = X0[valid]

        if X0_valid.size == 1:
            return float(X0_valid[0])

        mean_d = np.mean(X0_valid)
        std_d = np.std(X0_valid)
        if std_d / mean_d < 1e-6:
            return float(mean_d)

        d2_sum = np.sum(X0_valid ** 2)
        d3_sum = np.sum(X0_valid ** 3)

        if d2_sum <= 0:
            raise ValueError(
                "Sum of squared diameters is zero or negative. "
                "Cannot compute Sauter diameter."
            )

        x_s = d3_sum / d2_sum
        self._computed_x_s = x_s
        return x_s

    def _get_x_s(self, solver) -> float:
        """Sauter diameter [m], either from the parameter or from solver.X0."""
        if self.x_s is not None:
            x_s_val = float(self.x_s)
            if x_s_val <= 0:
                raise ValueError(f"x_s must be positive, got {x_s_val}")
            return x_s_val

        if self._computed_x_s is not None:
            return self._computed_x_s

        return self._compute_sauter_diameter(solver)

    def _compute_sigma(self, poro: float, saturation: float, x_s: float) -> float:
        """Particle strength sigma [Pa] from porosity and saturation."""
        return _sigma_jit(
            poro, saturation, x_s,
            self.k, self.alpha, self.gamma, self.cos_delta,
            self.SAT_DRY_THRESHOLD, self.SAT_WET_THRESHOLD,
            self.SAT_TRANSITION_WIDTH,
        )

    def _compute_base_rate_powerlaw(self, v_particle: float) -> float:
        """Base rate without the sigma correction [1/s]."""
        return _base_rate_scalar(
            v_particle, self.p1, self.p2, self.g, self.breakrval
        )

    # ------------------------------------------------------------------
    # Rate
    # ------------------------------------------------------------------
    def compute_rate(self, v_particle: float,
                     particle_idx: int = None,
                     solver = None) -> float:
        """
        Compute breakage rate with porosity/saturation-dependent strength.

        Formula: S(V) = S_base(V) / sigma(poro, saturation)

        Args:
            v_particle: Particle volume [m³]
            particle_idx: Index of particle in solver arrays
            solver: Reference to solver for accessing porosity/saturation

        Returns:
            rate: Breakage rate [1/s]
        """
        if v_particle <= 0:
            return 0.0

        base_rate = self._compute_base_rate_powerlaw(v_particle)

        if solver is None or particle_idx is None:
            return base_rate

        try:
            poro = solver.porosity[particle_idx]
        except (AttributeError, IndexError, TypeError):
            return base_rate

        try:
            saturation = solver.saturation[particle_idx]
        except (AttributeError, IndexError, TypeError):
            saturation = 0.0  # Default to dry

        # Vollkoerper (poro = NaN) and pore-free particles (poro <= 0) have no
        # pores, hence no capillary/pore-wall failure mechanism: they do not
        # break. sigma diverges as eps -> 0, so S = S_base/sigma -> 0;
        # returning 0.0 IS the limit of the formula.
        if np.isnan(poro) or poro <= 0.0:
            return 0.0

        # Upper edge: clamp instead of falling back, so sigma stays finite and
        # S(eps) stays continuous.
        if poro > self.poro_max:
            poro = self.poro_max

        try:
            x_s = self._get_x_s(solver)
        except (ValueError, AttributeError) as e:
            if hasattr(solver, 'VERBOSE') and solver.VERBOSE:
                print(f"[powerlaw_rumpf_dynamic] Warning: {e}, using base rate")
            return base_rate

        sigma = self._compute_sigma(poro, saturation, x_s)

        # Higher strength -> lower breakage rate
        if np.isfinite(sigma) and sigma > 1e-30:
            rate = base_rate / sigma
        else:
            # Unreachable: the porosity clamp bounds sigma from below and
            # validate_params rules out cos(delta) == 0. Kept as a division
            # guard. Returns 0.0, not base_rate: a vanishing sigma means
            # "infinitely strong", i.e. no breakage.
            rate = 0.0

        return max(0.0, float(rate))

    def compute_rate_array(self, v_particles: np.ndarray,
                           solver = None) -> np.ndarray:
        """
        Vectorised form of :meth:`compute_rate` for a whole particle slice.

        Bit-identical to the scalar path by construction: every branch uses the
        same expressions in the same association order as :func:`_sigma_jit`,
        and the branches are selected with ``np.where`` instead of ``if``.

        Args:
            v_particles: Particle volumes (V_dry) [m³], shape (n,)
            solver: Reference to solver for porosity/saturation

        Returns:
            rates: Breakage rates [1/s], shape (n,)
        """
        v = np.asarray(v_particles, dtype=float)
        base = _base_rate_array(v, self.p1, self.p2, self.g, self.breakrval)

        # No context -> sigma correction not applicable (mirrors scalar path)
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
                print(f"[powerlaw_rumpf_dynamic] Warning: {e}, using base rate")
            return base

        valid = np.isfinite(poro) & (poro > 0.0)
        # Dummy 0.5 on the invalid entries keeps the arithmetic below free of
        # division-by-zero warnings; those entries are masked out at the end.
        poro_c = np.where(valid, np.minimum(poro, self.poro_max), 0.5)

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

        # `sigma > 1e-30` strictly, exactly like the scalar guard.
        ok = valid & np.isfinite(sigma) & (sigma > 1e-30)
        rate = np.where(ok, base / sigma, 0.0)
        return np.maximum(rate, 0.0)
