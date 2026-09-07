"""
Mixer-speed-driven porosity compression kernel.

Same consolidation physics as :mod:`.porosity_compression` -- porosity decays
exponentially towards an asymptotic minimum -- but the rate constant is driven
by the mixer speed instead of being a fixed fit value:

    k          = rate * n_mixer ** c_mixer
    eps(t+dt)  = eps_min + (eps(t) - eps_min) * exp(-k * dt)

The analytical solution of ``deps/dt = -k * (eps - eps_min)`` with a *constant*
``k``; stable for any ``dt``. Only ``rate`` differs in meaning from
:mod:`.porosity_compression` -- it is the compaction rate at ``n_mixer = 1``,
not at the (unspecified) shear of the static kernel.

Self-contained
    Imports only numpy, the abstract base class and the shared mixer-speed
    constants. Handing this one file (plus ``base.py`` and ``mixer_speed.py``)
    to someone else gives a working kernel.

Where c_mixer comes from
    Consolidation is a plastic rearrangement of the primary-particle skeleton,
    driven per impact by the plastic work done on the granule -- i.e. the
    impact kinetic energy, which scales with the square of the relative
    collision velocity. The rate is that per-impact densification times the
    impact frequency:

        k ~ (impacts per second) x (plastic work per impact)
          ~ n ** C_FREQ          x  n ** (2 * C_VEL)
          = n ** C_BREAK

    This is the SAME reasoning the dynamic breakage kernel uses (see
    ``mixer_speed.C_BREAK``): breakage and consolidation are the same
    plastic-deformation-per-impact mechanism and differ only in outcome
    (rupture at a high deformation number vs. a permanent density increase at a
    low one). Driving both with the same exponent is the internally consistent
    choice. Only the EXPONENT transfers from the DEM study; its prefactor is
    physically not transferable and is absorbed into ``rate``.

    ``c_mixer = 0.0`` switches the mixer-speed dependence off -- the kernel then
    reduces exactly to :mod:`.porosity_compression` with ``k = rate``.

Interface
    ``compute(porosity, dt, ...) -> float``
    ``compute_array(porosity, dt, solver=None) -> np.ndarray``

    ``solver`` is accepted for a uniform call signature across compression
    kernels (see :meth:`CompressionKernel.compute_array`) but is not read: this
    kernel's ``k`` is a population-wide constant.
"""

import numpy as np

from ..base import CompressionKernel
from ..mixer_speed import C_BREAK, N_MIXER_DEFAULT


class PorosityCompressionDynamikKernel(CompressionKernel):
    """
    Exponential porosity decay with a mixer-speed-driven rate constant.

    Parameters:
        rate: Compaction rate constant at ``n_mixer = 1`` [1/s]
              (default: 2.688e-3). The effective rate is
              ``k = rate * n_mixer ** c_mixer``; the default is calibrated so
              that ``k`` equals the :mod:`.porosity_compression` default of
              0.02 at ``n_mixer = N_MIXER_DEFAULT`` and ``c_mixer = C_BREAK``.
              NOT comparable to the ``rate`` of :mod:`.porosity_compression`.
        min_porosity: Minimum achievable porosity eps_min (default: 0.3).
              Deliberately NOT mixer-speed dependent: eps_min is the
              maximum-packing limit set by particle shape, friction and binder
              film; a higher mixer speed reaches it faster (the rate), it does
              not move the floor.
        n_mixer: Mixer speed (default: ``mixer_speed.N_MIXER_DEFAULT`` = 20.0).
              Must match the ``n_mixer`` of every other mixer-speed-driven
              kernel in the same run; ``assert_consistent_mixer_speed``
              enforces that at solver setup. Only defined up to a constant
              factor (see ``mixer_speed``).
        c_mixer: Mixer-speed exponent (default: ``mixer_speed.C_BREAK`` =
              0.6699, stressing frequency x impact energy). Set ``c_mixer=0.0``
              to switch the mixer-speed dependence off entirely.

    Example:
        >>> kernel = PorosityCompressionDynamikKernel(rate=2.688e-3, n_mixer=20.0)
        >>> poro_new = kernel.compute(porosity=0.5, dt=1.0)
    """

    @property
    def name(self) -> str:
        return 'porosity_compression_dynamik'

    def get_default_params(self) -> dict:
        return {
            'rate': 2.688e-3,     # 1/s, at n_mixer = 1; see class docstring
            'min_porosity': 0.3,  # dimensionless
            'n_mixer': N_MIXER_DEFAULT,
            'c_mixer': C_BREAK,
        }

    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)

        self.rate = float(self.params['rate'])
        self.eps_min = float(self.params['min_porosity'])
        self.n_mixer = float(self.params['n_mixer'])
        self.c_mixer = float(self.params['c_mixer'])

        # Population-wide constant rate. Computed once, exactly as
        # ``eke_darelius2005`` folds its prefactor, so no code path can end up
        # with a different value.
        self.k = self.rate * self.n_mixer ** self.c_mixer

    def validate_params(self, params: dict) -> dict:
        """Validate parameters."""
        if params['rate'] < 0:
            raise ValueError(f"rate must be non-negative, got {params['rate']}")
        if not (0.0 <= params['min_porosity'] <= 1.0):
            raise ValueError(
                f"min_porosity must be in [0, 1], got {params['min_porosity']}"
            )
        if params['n_mixer'] <= 0:
            raise ValueError(
                f"n_mixer must be positive, got {params['n_mixer']}. "
                "A non-positive mixer speed makes n_mixer**c_mixer undefined "
                "for a fractional exponent. Use c_mixer=0.0 to disable the "
                "mixer speed dependence instead."
            )
        c_mixer = float(params['c_mixer'])
        if not np.isfinite(c_mixer):
            raise ValueError(f"c_mixer must be finite, got {c_mixer}")
        return params

    # ------------------------------------------------------------------
    # Scalar
    # ------------------------------------------------------------------
    def compute(self,
                porosity: float,
                dt: float,
                v_particle: float = None,
                local_stress: float = None,
                saturation: float = None,
                solver=None
                ) -> float:
        """
        New porosity after exponential decay over time ``dt``.

        Args:
            porosity: Current porosity (0 to 1); NaN ("Vollkoerper") passes
                      through unchanged, as does any value <= min_porosity
                      (compression is reductive only).
            dt: Time step [s].
            v_particle, local_stress, saturation, solver: Not used -- this
                      kernel's rate is a population-wide constant. Accepted for
                      interface compatibility.

        Returns:
            porosity_new: Reduced porosity (>= min_porosity).
        """
        if np.isnan(porosity):
            return porosity

        porosity = max(0.0, min(1.0, porosity))
        if porosity <= self.eps_min:
            return porosity

        porosity_new = self.eps_min + (porosity - self.eps_min) * np.exp(-self.k * dt)
        return float(max(self.eps_min, porosity_new))

    # ------------------------------------------------------------------
    # Vectorised
    # ------------------------------------------------------------------
    def compute_array(self, porosity: np.ndarray, dt: float, solver=None) -> np.ndarray:
        """Vectorised form of :meth:`compute`.

        Bit-identical to a loop of scalar calls: the decay factor
        ``exp(-k*dt)`` is a scalar, so only multiplications and clamps are
        applied elementwise.

        Args:
            porosity: Current porosities; NaN entries ("Vollkoerper") and
                      values <= min_porosity pass through unchanged.
            dt:       Time step [s].
            solver:   Not used (uniform signature only).

        Returns:
            Updated porosities, same shape as ``porosity``.
        """
        poro = np.asarray(porosity, dtype=float)
        out = poro.copy()

        finite = ~np.isnan(poro)
        if not np.any(finite):
            return out

        clipped = np.clip(poro[finite], 0.0, 1.0)
        can_compress = clipped > self.eps_min
        if np.any(can_compress):
            decay = np.exp(-self.k * dt)
            updated = self.eps_min + (clipped[can_compress] - self.eps_min) * decay
            updated = np.maximum(self.eps_min, updated)

            # Carry the mask back to full-array positions before the single
            # indexed write (see the note in porosity_compression.py: a chained
            # ``out[finite][mask] = ...`` writes into a throwaway copy).
            compressible = np.zeros(poro.shape, dtype=bool)
            compressible[finite] = can_compress
            out[compressible] = updated

        return out
