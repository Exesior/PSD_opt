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

Self-contained:
    Everything this kernel needs is in this file. It imports only the abstract
    base class, numpy and numba - no other kernel and no shared rate helper.
    See the note above `_VALID_BREAKRVAL` for why.
"""

import numpy as np

from ..base import BreakageKernel

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
        _validate_rate_params(params, self.name)

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
        return float(_base_rate_scalar(
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
        return _base_rate_array(
            v_particles, self.p1, self.p2, self.g, self.breakrval
        )
