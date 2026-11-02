"""
Kinetic Blend Aggregation Kernel (equal speed <-> EKE <-> ETM).

One kernel for the whole family of kinetic collision frequencies in a gas
continuum. The three classical cases differ only in how the fluctuating motion
of the mixer is shared out over the particle sizes, i.e. in how the random
velocity of a particle scales with its mass:

    v ~ m^(-blend/2)

    blend = 0   every particle equally fast              (equal speed)
    blend = 1   equal kinetic energy, v ~ m^(-1/2)       (EKE, Hounslow 1998)
    blend = 2   equal translational momentum, v ~ 1/m    (ETM, Hounslow 1998)

`blend` is a fit factor; values in between interpolate continuously. Darelius
et al. (2005) found EKE better at low and ETM better at high impeller speed;
the equal-speed limit is the kinetic-theory case of one common granular
temperature in velocity units.

References:
    Hounslow, KONA 16 (1998) 179-193 (EKE and ETM kernels)
    Darelius et al., Powder Technology 160 (2005) 209-218, Table 1

Formula:
    beta = CORR_BETA * n_mixer^C_MIXER * (r1 + r2)^2
                     * sqrt(r1^(-3 blend) + r2^(-3 blend))

    - (r1 + r2)^2                        collision cross-section [m^2]
    - sqrt(r1^(-3 b) + r2^(-3 b))        relative velocity: with m ~ r^3,
                                         sqrt(v1^2 + v2^2) ~ sqrt(m1^-b + m2^-b)
    - n_mixer^C_MIXER                    mixer speed factor, as in EKE
    - CORR_BETA                          fit prefactor [m^(1 + 1.5 blend)/s]

    blend = 1 gives exactly `eke_darelius2005`, blend = 2 exactly
    `etm_darelius2005` (bit-for-bit, the same arithmetic is used for these two
    values). blend = 0 gives sqrt(2) * CORR_BETA * n^c * (r1 + r2)^2.

Size dependence (two equal particles, radius r):  beta ~ r^(2 - 1.5 blend)
    blend = 0: r^2  (large pairs favoured, only through the cross-section)
    blend = 1: r^0.5
    blend = 2: r^-1 (small pairs favoured)

Note on fitting:
    The unit of CORR_BETA changes with `blend`, so CORR_BETA and `blend` are
    correlated in a fit. Calibrate CORR_BETA for each blend against a reference
    pair if values are compared between different blends.

Implementation:
    beta is a function of (p0, k, r1, r2) with p0 = CORR_BETA * n^c and
    k = 3 * blend, so the compiled batch path in :mod:`.jit_kernels` applies
    (KID_KINETIC_BLEND, k is passed as the second scalar `p1`). Not separable
    (square root of a sum), hence no O(n log n) moment form.
"""

import math

from numba import njit

from ..base import AggregationKernel
from ..mixer_speed import C_FREQ, N_MIXER_DEFAULT


@njit(cache=True)
def _beta_kinetic_blend_jit(p0: float, k: float, r1: float, r2: float) -> float:
    """
    JIT-compiled kinetic blend kernel.

    Args:
        p0: Folded prefactor corr_beta * n_mixer^c_mixer
        k:  Velocity exponent in radius, k = 3 * blend
        r1: Radius of particle 1 [m]
        r2: Radius of particle 2 [m]

    Returns:
        beta: Collision frequency [m^3/s]

    Note:
        The arithmetic must stay bit-for-bit identical to the ``_beta_of``
        branch ``KID_KINETIC_BLEND`` in :mod:`.jit_kernels`. k == 3 and k == 6
        use exactly the EKE and ETM arithmetic, so blend = 1 and blend = 2
        reproduce those kernels bit-for-bit.
    """
    if r1 <= 0.0 or r2 <= 0.0:
        return 0.0
    _rsum = r1 + r2
    if k == 3.0:
        _inv = 1.0 / (r1 * r1 * r1) + 1.0 / (r2 * r2 * r2)
    elif k == 6.0:
        _ci = r1 * r1 * r1
        _cj = r2 * r2 * r2
        _inv = 1.0 / (_ci * _ci) + 1.0 / (_cj * _cj)
    else:
        _inv = r1 ** (-k) + r2 ** (-k)
    return p0 * _rsum * _rsum * math.sqrt(_inv)


class KineticBlendKernel(AggregationKernel):
    """
    Kinetic collision frequency between equal speed, EKE and ETM.

    Parameters:
        corr_beta: Fit prefactor [m^(1 + 1.5 blend)/s] (default: 1e-11, the
                   EKE default). Must be refitted when `blend` changes.
        blend: Fit factor for the velocity scaling v ~ m^(-blend/2)
               (default: 1.0 = EKE). Range [0, 2]: 0 equal speed, 1 EKE,
               2 ETM.
        n_mixer: Mixer speed (default: `mixer_speed.N_MIXER_DEFAULT` = 20.0).
                 Must match every other mixer-speed-driven kernel in the run.
        c_mixer: Mixer speed exponent (default: `mixer_speed.C_FREQ` = 0.0995).

    Example:
        >>> kernel = KineticBlendKernel(corr_beta=1e-11, blend=0.5, n_mixer=30.0)
        >>> beta = kernel.compute_beta(r1=17e-6, r2=100e-6)
    """

    BLEND_MIN = 0.0
    BLEND_MAX = 2.0

    @property
    def name(self) -> str:
        return 'kinetic_blend'

    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-11,          # [m^(1+1.5 blend)/s] fit prefactor
            'blend': 1.0,                # 0 equal speed, 1 EKE, 2 ETM
            'n_mixer': N_MIXER_DEFAULT,  # shared across all mixer kernels
            'c_mixer': C_FREQ,           # DEM collision frequency exponent
        }

    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)

        # Cache commonly used values
        self.corr_beta = float(self.params['corr_beta'])
        self.blend = float(self.params['blend'])
        self.n_mixer = float(self.params['n_mixer'])
        self.c_mixer = float(self.params['c_mixer'])

        # Folded scalars. Computed ONCE here and read back by
        # jit_kernels.kernel_p0 / kernel_p1, so the compiled path cannot end up
        # with values that differ from these in the last bit.
        self.p0 = self.corr_beta * self.n_mixer ** self.c_mixer
        self.p1 = 3.0 * self.blend

    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['corr_beta'] <= 0:
            raise ValueError(
                f"corr_beta must be positive, got {params['corr_beta']}"
            )
        blend = float(params['blend'])
        if not (self.BLEND_MIN <= blend <= self.BLEND_MAX):
            raise ValueError(
                f"blend must be in [{self.BLEND_MIN}, {self.BLEND_MAX}] "
                f"(0 equal speed, 1 EKE, 2 ETM), got {blend}"
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
        Compute the kinetic blend collision frequency.

        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx: Not used (for interface compatibility)
            particle2_idx: Not used (for interface compatibility)
            solver: Not used (for interface compatibility)

        Returns:
            beta: Collision frequency [m^3/s]; non-finite results are mapped
                  to 0.0, as in the EKE kernel.
        """
        beta = _beta_kinetic_blend_jit(self.p0, self.p1, r1, r2)
        if not math.isfinite(beta) or beta < 0.0:
            return 0.0
        return float(beta)
