"""
EKE Aggregation Kernel (Equipartition of Kinetic Energy).

Collision frequency for a mixer whose CONTINUUM is a gas, so that there is no
liquid velocity gradient G to drive the collisions. Instead the mixer keeps the
powder bed in a fluctuating, gas-like state and the collision frequency follows
from kinetic theory.

Note that this does NOT mean the process is dry: binder is still added, granules
still carry liquid bridges and saturation. What is dry is only the phase the
granules move THROUGH.

Reference:
    Darelius et al., "High shear wet granulation modelling - a mechanistic
    approach using population balances", Powder Technology 160 (2005) 209-218,
    Table 1 (EKE kernel), originally Hounslow, KONA 16 (1998) 179-193.

Formula:
    beta = CORR_BETA * n_mixer^C_MIXER * (r1 + r2)^2 * sqrt(1/r1^3 + 1/r2^3)

Where:
    - (r1 + r2)^2                   collision cross-section [m^2]
    - sqrt(1/r1^3 + 1/r2^3)         relative fluctuation velocity [m^-3/2]
      This is proportional to sqrt(1/m1 + 1/m2): the fluctuating kinetic energy
      is equally distributed over all granules ("equipartition"), so a light
      granule moves faster than a heavy one. Same idea as the thermal kT in
      `brownian_tsouris1995`, but in the ballistic instead of the diffusive
      limit.
    - n_mixer^C_MIXER               mixer speed factor (see below)
    - CORR_BETA                     fit prefactor [m^(5/2)/s]

Physical background:
    The EKE kernel is derived by writing the local velocity as a Reynolds
    decomposition, assuming that the *mean* velocity of a granule does not
    depend on its size but the *random* component does, and then treating the
    collisions like an ideal gas (instantaneous, fully elastic). Darelius found
    the EKE kernel to describe granulation best at the LOWER impeller speeds,
    while the ETM kernel (see `etm_darelius2005`) was better at higher ones.

Mixer speed factor:
    The classical form has no process variable at all - the intensity of the
    mixing sits entirely inside the fitted prefactor. To make the mixer speed a
    steerable input, a power law taken from a comparable DEM study is applied:

        collision frequency ~ n^C_FREQ   (= 0.0995)

    Only the EXPONENT is transferred; the DEM prefactor (~3e7) is not
    physically transferable and is absorbed into `corr_beta` together with all
    other unknown constants. `n_mixer` is therefore only defined up to a
    constant factor: changing its unit (m/s vs. rpm vs. 1/s) rescales the whole
    kernel by a constant, which `corr_beta` absorbs. What matters is the SHAPE
    of the dependence, and that is unit-free for a power law.

    Note that the exponent is very small: going from n = 5 to n = 40 changes
    beta by only 40^0.0995 / 5^0.0995 = 1.23, i.e. 23%. The mixer speed is a
    weak knob by construction, not by accident.

Size dependence (important when comparing against `shear_chin1998`):
    For two equally sized particles the EKE kernel scales as beta ~ sqrt(r),
    whereas the shear kernel scales as beta ~ r^3. Large granules are therefore
    far less favoured than under shear, and the size distribution stays
    narrower. Conversely, a very small partner is strongly favoured
    (beta ~ r_small^(-3/2) as r_small -> 0), which is physically sensible -
    fines are the fast movers - but means fines are scavenged quickly.

Implementation:
    beta is a pure function of (p0, r1, r2), so the compiled batch path in
    :mod:`.jit_kernels` applies (KID_EKE). The kernel is NOT separable - the
    square root of a *sum* cannot be written as sum_k f_k(i) g_k(j) - so there
    is no O(n log n) moment form and it is deliberately absent from
    ``MOMENT_KERNELS``.
"""

import math

from numba import njit

from ..base import AggregationKernel
from ..mixer_speed import C_FREQ, N_MIXER_DEFAULT


@njit(cache=True)
def _beta_eke_jit(p0: float, r1: float, r2: float) -> float:
    """
    JIT-compiled EKE kernel.

    Args:
        p0: Folded prefactor corr_beta * n_mixer^c_mixer [m^(5/2)/s]
        r1: Radius of particle 1 [m]
        r2: Radius of particle 2 [m]

    Returns:
        beta: Collision frequency [m^3/s]

    Note:
        The arithmetic must stay bit-for-bit identical to the ``_beta_of``
        branch ``KID_EKE`` in :mod:`.jit_kernels`; the verification suite
        asserts that the two agree exactly. Keep the association order as
        written.
    """
    if r1 <= 0.0 or r2 <= 0.0:
        return 0.0
    _rsum = r1 + r2
    _inv = 1.0 / (r1 * r1 * r1) + 1.0 / (r2 * r2 * r2)
    return p0 * _rsum * _rsum * math.sqrt(_inv)


class EKEDareliusKernel(AggregationKernel):
    """
    Equipartition-of-Kinetic-Energy collision frequency (Darelius et al. 2005).

    Intended for a gas continuum, where no shear rate can be given. The
    driving quantity is the mixer speed rather than a velocity gradient.

    Parameters:
        corr_beta: Fit prefactor [m^(5/2)/s] (default: 1e-11).
                   Absorbs the collision efficiency, the DEM prefactor and the
                   unit convention of `n_mixer`. Darelius fitted values around
                   1e-11 for mcc granulation; expect to refit this for any new
                   system - it is NOT comparable to the `corr_beta` of the
                   shear kernel (different units, different size scaling).
        n_mixer: Mixer speed (default: `mixer_speed.N_MIXER_DEFAULT` = 20.0,
                 the middle of the DEM range 5-40 m/s). Must match the
                 `n_mixer` of every other mixer-speed-driven kernel in the
                 same run; `assert_consistent_mixer_speed` enforces that at
                 solver setup. Only defined up to a constant factor.
        c_mixer: Mixer speed exponent (default: `mixer_speed.C_FREQ` =
                 0.0995, from the DEM fit y = 3e7 * x^0.0995). Set
                 c_mixer=0.0 to switch the dependence off entirely.

    Example:
        >>> kernel = EKEDareliusKernel(corr_beta=1e-11, n_mixer=20.0)
        >>> beta = kernel.compute_beta(r1=17e-6, r2=17e-6)
        >>> print(f"Collision frequency: {beta:.3e} m3/s")
    """

    @property
    def name(self) -> str:
        return 'eke_darelius2005'

    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-11,          # [m^(5/2)/s] fit prefactor
            'n_mixer': N_MIXER_DEFAULT,  # shared across all mixer kernels
            'c_mixer': C_FREQ,           # DEM collision frequency exponent
        }

    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)

        # Cache commonly used values
        self.corr_beta = float(self.params['corr_beta'])
        self.n_mixer = float(self.params['n_mixer'])
        self.c_mixer = float(self.params['c_mixer'])

        # Folded prefactor. Computed ONCE here and read back by
        # jit_kernels.kernel_p0, so the compiled path cannot end up with a
        # prefactor that differs from this one in the last bit.
        self.p0 = self.corr_beta * self.n_mixer ** self.c_mixer

    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['corr_beta'] <= 0:
            raise ValueError(
                f"corr_beta must be positive, got {params['corr_beta']}"
            )
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
        return params

    def compute_beta(self, r1: float, r2: float,
                     particle1_idx: int = None,
                     particle2_idx: int = None,
                     solver=None) -> float:
        """
        Compute the EKE collision frequency.

        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx: Not used (for interface compatibility)
            particle2_idx: Not used (for interface compatibility)
            solver: Not used (for interface compatibility)

        Returns:
            beta: Collision frequency [m^3/s]

        Note:
            beta diverges as either radius goes to zero. Non-finite results
            (which would require a radius below ~1e-100 m) are mapped to 0.0
            so a degenerate particle can never poison the propensity sum.
        """
        beta = _beta_eke_jit(self.p0, r1, r2)
        if not math.isfinite(beta) or beta < 0.0:
            return 0.0
        return float(beta)
