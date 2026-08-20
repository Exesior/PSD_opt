"""
ETM Aggregation Kernel (Equipartition of Translational Momentum).

Sister kernel to :mod:`.eke_darelius2005`, built on the same picture of a
fluctuating powder bed in a gas continuum, but with a different assumption
about how the mixer shares its energy out among the granules.

Reference:
    Darelius et al., "High shear wet granulation modelling - a mechanistic
    approach using population balances", Powder Technology 160 (2005) 209-218,
    Table 1 (ETM kernel), originally Hounslow, KONA 16 (1998) 179-193.

Formula:
    beta = CORR_BETA * n_mixer^C_MIXER * (r1 + r2)^2 * sqrt(1/r1^6 + 1/r2^6)

Difference to EKE:
    EKE assumes every granule carries the same fluctuating kinetic ENERGY,
    hence v ~ 1/sqrt(m) and the velocity term sqrt(1/r1^3 + 1/r2^3).
    ETM assumes every granule receives the same random IMPULSE, so its
    fluctuating translational MOMENTUM is the same, hence v ~ 1/m and the
    velocity term sqrt(1/r1^6 + 1/r2^6).

    Darelius found ETM to describe the granulation better at the HIGHER
    impeller speeds and EKE better at the lower ones, without being able to
    say from first principles why. Having both available is the point: they
    bracket the plausible range of how mixing intensity is distributed over
    granule sizes.

Size dependence:
    For two equally sized particles beta ~ 1/r, i.e. the collision frequency
    *decreases* with granule size - an even stronger bias towards fines than
    EKE (which gives beta ~ sqrt(r)) and the opposite of the shear kernel
    (beta ~ r^3). Expect very little runaway growth from this kernel.

Units:
    CORR_BETA has units [m^4/s] here, NOT the [m^(5/2)/s] of the EKE kernel.
    The two prefactors are not interchangeable and must be fitted separately.

Mixer speed factor:
    Identical treatment to the EKE kernel - see that module's docstring. Only
    the exponent of the DEM correlation is transferred, the DEM prefactor is
    absorbed into `corr_beta`.

Implementation:
    Pure function of (p0, r1, r2), so the compiled batch path in
    :mod:`.jit_kernels` applies (KID_ETM). Not separable, therefore
    deliberately absent from ``MOMENT_KERNELS``.
"""

import math

from numba import njit

from ..base import AggregationKernel
from ..mixer_speed import C_FREQ, N_MIXER_DEFAULT


@njit(cache=True)
def _beta_etm_jit(p0: float, r1: float, r2: float) -> float:
    """
    JIT-compiled ETM kernel.

    Args:
        p0: Folded prefactor corr_beta * n_mixer^c_mixer [m^4/s]
        r1: Radius of particle 1 [m]
        r2: Radius of particle 2 [m]

    Returns:
        beta: Collision frequency [m^3/s]

    Note:
        The arithmetic must stay bit-for-bit identical to the ``_beta_of``
        branch ``KID_ETM`` in :mod:`.jit_kernels`; the verification suite
        asserts that the two agree exactly. ``r^6`` is evaluated as
        ``(r*r*r)**2`` rather than ``r**6`` so both sides round the same way.
    """
    if r1 <= 0.0 or r2 <= 0.0:
        return 0.0
    _rsum = r1 + r2
    _c1 = r1 * r1 * r1
    _c2 = r2 * r2 * r2
    _inv = 1.0 / (_c1 * _c1) + 1.0 / (_c2 * _c2)
    return p0 * _rsum * _rsum * math.sqrt(_inv)


class ETMDareliusKernel(AggregationKernel):
    """
    Equipartition-of-Translational-Momentum collision frequency
    (Darelius et al. 2005).

    Intended for a gas continuum, as the alternative hypothesis to
    :class:`~.eke_darelius2005.EKEDareliusKernel`.

    Parameters:
        corr_beta: Fit prefactor [m^4/s] (default: 1e-16).
                   Absorbs the collision efficiency, the DEM prefactor and the
                   unit convention of `n_mixer`. Not comparable to the
                   `corr_beta` of any other kernel.
        n_mixer: Mixer speed (default: `mixer_speed.N_MIXER_DEFAULT` = 20.0).
                 Must match every other mixer-speed-driven kernel in the run.
        c_mixer: Mixer speed exponent (default: `mixer_speed.C_FREQ` =
                 0.0995). Set to 0.0 to disable the dependence.

    Example:
        >>> kernel = ETMDareliusKernel(corr_beta=1e-16, n_mixer=20.0)
        >>> beta = kernel.compute_beta(r1=17e-6, r2=17e-6)
    """

    @property
    def name(self) -> str:
        return 'etm_darelius2005'

    def get_default_params(self) -> dict:
        return {
            'corr_beta': 1e-16,          # [m^4/s] fit prefactor
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

        # Folded prefactor; read back by jit_kernels.kernel_p0 so the compiled
        # path cannot drift from this value. See EKEDareliusKernel.
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
        Compute the ETM collision frequency.

        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx: Not used (for interface compatibility)
            particle2_idx: Not used (for interface compatibility)
            solver: Not used (for interface compatibility)

        Returns:
            beta: Collision frequency [m^3/s]

        Note:
            beta diverges as either radius goes to zero, faster than for EKE
            (r^-3 instead of r^-3/2). Non-finite results are mapped to 0.0.
        """
        beta = _beta_etm_jit(self.p0, r1, r2)
        if not math.isfinite(beta) or beta < 0.0:
            return 0.0
        return float(beta)
