"""
Shared breakage rate base formula S(V) for all breakage kernels.

This is the single source of truth for the BREAKRVAL switch inside `wmcpbe`.
It mirrors `pbe_core.func.jit_kernel_break.calc_break_rate_1d` branch for
branch, so that `wmcpbe` produces the same breakage rates as the reference
solver (`dev_monorepo`, which calls that function directly).

Formulas (BREAKRVAL 1-4, identical to the reference):
    BREAKRVAL=1: S = P1                  (size independent, Leong2023 (10))
    BREAKRVAL=2: S = P1 * V              (size dependent, Leong2023 (10))
    BREAKRVAL=3: S = P1 * G * V^P2       (Pandy & Spielmann, Jeldres2018 (28))
    BREAKRVAL=4: S = P1 * G * V^P2       (identical to 3 in 1D)

Note on P2:
    P2 is the VOLUME exponent and G enters linearly. An earlier version of
    this code treated P2 as a shear exponent and used a separate `pl_v`
    parameter as the volume exponent (S = P1 * G^P2 * V^(pl_v/3)). That was a
    mix-up with BREAKFVAL, the fragment-size distribution switch, whose
    parameters really are called `pl_v` / `pl_q`. `pl_v` / `pl_q` belong to the
    breakage FUNCTION (see `mcpbe_break.py`: `break_frag_v` / `break_frag_q`),
    never to the breakage RATE.

    There is deliberately no BREAKRVAL=5 -- the reference implementation has
    no such branch.
"""

import numpy as np

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


#: BREAKRVAL values implemented here (and by the reference implementation).
VALID_BREAKRVAL = (1, 2, 3, 4)


def validate_rate_params(params: dict, kernel_name: str) -> None:
    """
    Reject breakrval values and parameters that no longer steer the rate.

    Raises a ValueError with a migration hint instead of silently ignoring
    them, so setups written against the old (divergent) formula fail loudly.

    Args:
        params: Parameter dictionary of the kernel
        kernel_name: Kernel name, used in the error messages

    Raises:
        ValueError: On breakrval=5 or on a supplied `pl_v` / `pl_q`.
    """
    breakrval = params.get('breakrval')
    if breakrval == 5:
        raise ValueError(
            f"breakrval=5 is not a valid breakage rate model for "
            f"'{kernel_name}'. It never existed in the reference "
            f"implementation (pbe_core.func.jit_kernel_break."
            f"calc_break_rate_1d) and has been removed. "
            f"Valid values: {list(VALID_BREAKRVAL)}."
        )
    if breakrval is not None and breakrval not in VALID_BREAKRVAL:
        raise ValueError(
            f"Invalid breakrval={breakrval} for '{kernel_name}'. "
            f"Must be one of: {list(VALID_BREAKRVAL)}"
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
def compute_base_rate(v_particle: float, p1: float, p2: float, g: float,
                      breakrval: int) -> float:
    """
    Breakage rate S(V) for a single particle.

    Args:
        v_particle: Particle volume [m³]
        p1: Pre-factor
        p2: Volume exponent (BREAKRVAL 3, 4)
        g: Shear rate [1/s]
        breakrval: Model variant (1-4)

    Returns:
        rate: Breakage rate [1/s], 0.0 for non-positive volume, NaN, or an
              unknown breakrval.
    """
    # `not (v > 0.0)` rather than `v <= 0.0`: it treats NaN like a non-positive
    # volume. With `fastmath=True` the compiler may assume NaN-freedom, so
    # `v <= 0.0` let a NaN slip through into the arithmetic below -- the JIT
    # version then returned 0.0 while `.py_func` returned NaN. Same expression
    # for both now. Genuine NaN corruption is caught loudly one level up, in
    # `mcpbe_break.py::_assert_finite_rates`.
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


def compute_base_rate_array(v_particles: np.ndarray, p1: float, p2: float,
                            g: float, breakrval: int) -> np.ndarray:
    """
    Vectorised form of :func:`compute_base_rate`.

    Same expressions in the same order, so results match element for element.

    Args:
        v_particles: Particle volumes [m³], shape (n,)
        p1, p2, g, breakrval: see :func:`compute_base_rate`

    Returns:
        rates: Breakage rates [1/s], shape (n,)
    """
    v = np.asarray(v_particles, dtype=float)
    # `v > 0.0` is False for NaN, so NaN volumes fall into the same bucket as
    # non-positive ones -- deliberately identical to the scalar guard above.
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
