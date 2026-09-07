"""
Mixer-speed-driven porosity compression with a Rumpf-strength resistance.

Same consolidation picture as :mod:`.porosity_compression_dynamik` -- porosity
decays exponentially towards ``min_porosity``, the rate driven by the mixer
speed -- but the granule resists consolidation in proportion to its strength.
The rate is divided by the Rumpf granule strength ``sigma(eps, S)``:

    k(eps, S)  = rate * n_mixer ** c_mixer / sigma(eps, S)
    eps(t+dt)  = eps_min + (eps(t) - eps_min) * exp(-k(eps(t), S) * dt)

This mirrors the dynamic breakage kernel exactly: both breakage and
consolidation scale as ``n ** C_BREAK / sigma(eps, S)`` -- the same
plastic-deformation-per-impact mechanism, differing only in outcome (rupture at
a high impact-energy / strength ratio vs. a permanent density increase). Wetter
and denser granules resist consolidation, which is what consolidation
experiments report.

Why the closed-form solution still holds
    ``sigma`` depends on ``eps``, so ``deps/dt = -k(eps) * (eps - eps_min)`` is
    no longer linear. But the Rumpf strength factorises EXACTLY as

        sigma(eps, S) = phi(S) * (1 - eps) / eps

    in all three saturation regimes (verified numerically against
    ``powerlaw_rumpf``'s ``_sigma_jit`` in the Trials test). The ODE is then
    separable, with an implicit closed form -- but not one that can be solved
    for ``eps(t+dt)`` without a per-particle root find.

    Instead this kernel FREEZES ``sigma`` at ``eps(t)`` over the operator-split
    sub-step and applies the exact analytical solution of the resulting
    constant-coefficient linear ODE (exponential Euler). Justification:

    * The exponential consolidation law is itself the homogenisation of a
      DISCRETE process -- N impacts, each with a small densification increment.
      ``sigma`` enters per impact, evaluated at the porosity the granule has AT
      that impact. Over a sub-step spanning O(1) Monte-Carlo events, holding
      ``sigma`` at ``eps(t)`` is the natural discretisation, not an
      approximation.
    * The handler re-evaluates after every event, so the sub-step ``dt`` is
      small; the frozen-coefficient error is O(k*dt), and ``k*dt`` is ~1e-3 to
      1e-2 in practice (k ~ 0.02-0.1 /s, dt ~ 0.1 s). It is a consistent, tiny
      bias towards slightly faster compaction (the frozen ``k`` is the largest
      ``k`` of the step): ~1e-7 in porosity at dt = 0.1 s, ~6e-4 at dt = 1 s.
    * The scheme is unconditionally stable and keeps the ``eps_min`` asymptote
      and monotonicity exactly, for any ``dt`` -- the forward-Euler
      alternative does not.

Self-contained
    Imports only numpy, the abstract base class and the shared mixer-speed
    constants. The Rumpf strength and the Sauter-diameter helper are duplicated
    here (not imported from a breakage kernel), so this one file plus
    ``base.py`` and ``mixer_speed.py`` is a working kernel. The duplicate is
    pinned to ``powerlaw_rumpf``'s strength model by the Trials test.

Fail loud
    A missing granule size (no ``x_s`` and no ``solver.X0``), a non-positive or
    non-finite ``sigma``, or a call to :meth:`compute_array` without a solver
    all raise rather than fall back to a strength-free rate. A consolidation
    kernel with no strength model is meaningless, so a silent degrade would
    hide a misconfiguration.

Interface
    ``compute(porosity, dt, saturation=None, solver=None, ...) -> float``
    ``compute_array(porosity, dt, solver) -> np.ndarray``
"""

import numpy as np

from ..base import CompressionKernel
from ..mixer_speed import C_BREAK, N_MIXER_DEFAULT

#: Calibrated so that k(eps, S) equals the porosity_compression default of 0.02
#: at n_mixer = N_MIXER_DEFAULT, c_mixer = C_BREAK, and a reference strength
#: sigma(eps=0.4, S=0) with the default Rumpf parameters and x_s = 34e-6 m.
_RATE_DEFAULT = 21.35


class PorosityCompressionDynamikRumpfKernel(CompressionKernel):
    """
    Exponential porosity decay, mixer-speed driven, resisted by Rumpf strength.

    Parameters:
        rate: Compaction prefactor [Pa/s] (default: 21.35). The effective rate
              is ``k = rate * n_mixer ** c_mixer / sigma(eps, S)``. The default
              absorbs the ~8 kPa strength scale; see ``_RATE_DEFAULT``. NOT
              comparable to the ``rate`` of :mod:`.porosity_compression` or
              :mod:`.porosity_compression_dynamik`.
        min_porosity: Minimum achievable porosity eps_min (default: 0.3).
              Not mixer-speed dependent (see
              :mod:`.porosity_compression_dynamik`).
        n_mixer: Mixer speed (default: ``mixer_speed.N_MIXER_DEFAULT`` = 20.0).
              Must match every other mixer-speed-driven kernel in the run;
              ``assert_consistent_mixer_speed`` enforces that at solver setup.
        c_mixer: Mixer-speed exponent (default: ``mixer_speed.C_BREAK`` =
              0.6699). Set ``c_mixer=0.0`` to switch the mixer-speed dependence
              off (the strength resistance stays active).
        k: Rumpf dry-regime fit parameter [2.2-2.8] (default: 2.5). Same
              parameter and range as ``powerlaw_rumpf``; use the same value in
              both kernels.
        alpha: Rumpf wet-regime fit parameter [1.0-1.33] (default: 1.15).
        gamma: Binder surface tension [N/m] (default: 0.072, water).
        delta: Binder contact angle [rad] (default: 0.0, perfect wetting).
              Must be in [0, pi/2); at pi/2 the wet-regime strength is zero and
              the rate would diverge.
        x_s: Sauter diameter [m] (default: None -> the initial Sauter diameter
              from ``solver.X0``, computed once and held fixed as a length
              scale, exactly as ``powerlaw_rumpf_dynamic`` does).
        poro_max: Upper bound for the porosity entering ``sigma`` (default:
              0.9999), so ``(1 - eps)/eps`` stays finite as ``eps -> 1``.

    Example:
        >>> k = PorosityCompressionDynamikRumpfKernel(n_mixer=20.0, x_s=34e-6)
        >>> k.compute(porosity=0.5, dt=1.0, saturation=0.4)
    """

    SAT_DRY_THRESHOLD = 0.3       # S < 0.3: dry regime
    SAT_WET_THRESHOLD = 0.8       # S > 0.8: wet regime
    SAT_TRANSITION_WIDTH = 0.5    # 0.8 - 0.3

    @property
    def name(self) -> str:
        return 'porosity_compression_dynamik_rumpf'

    def get_default_params(self) -> dict:
        return {
            'rate': _RATE_DEFAULT,   # Pa/s; see module docstring
            'min_porosity': 0.3,
            'n_mixer': N_MIXER_DEFAULT,
            'c_mixer': C_BREAK,
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

        self.rate = float(self.params['rate'])
        self.eps_min = float(self.params['min_porosity'])
        self.n_mixer = float(self.params['n_mixer'])
        self.c_mixer = float(self.params['c_mixer'])

        # Folded mixer-speed prefactor: rate * n_mixer ** c_mixer. Divided by
        # sigma(eps, S) per particle to get the effective rate.
        self.rate_nc = self.rate * self.n_mixer ** self.c_mixer

        self.k_rumpf = float(self.params['k'])
        self.alpha = float(self.params['alpha'])
        self.gamma = float(self.params['gamma'])
        self.delta = float(self.params['delta'])
        self.cos_delta = float(np.cos(self.delta))
        self.x_s = self.params.get('x_s')
        self.poro_max = float(self.params['poro_max'])

        self._x_s_cache = None

    def validate_params(self, params: dict) -> dict:
        """Validate parameters with the same physical bounds as powerlaw_rumpf."""
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

        k = params['k']
        if not (2.2 <= k <= 2.8):
            raise ValueError(
                f"k must be in range [2.2, 2.8], got {k}. "
                "This fitting parameter represents capillary bridge geometry."
            )
        alpha = params['alpha']
        if not (1.0 <= alpha <= 1.33):
            raise ValueError(
                f"alpha must be in range [1.0, 1.33], got {alpha}. "
                "This fitting parameter accounts for liquid film effects."
            )
        gamma = params['gamma']
        if gamma <= 0:
            raise ValueError(
                f"gamma (surface tension) must be positive, got {gamma}"
            )
        delta = params['delta']
        if not (0 <= delta < np.pi / 2):
            raise ValueError(
                f"delta (contact angle) must be in [0, pi/2), got {delta}. "
                "delta = pi/2 gives cos(delta) = 0 and thus zero granule "
                "strength in the wet regime, i.e. an infinite compression rate."
            )
        poro_max = params['poro_max']
        if not (0.0 < poro_max < 1.0):
            raise ValueError(
                f"poro_max must be in (0, 1), got {poro_max}. It caps the "
                "porosity entering sigma; sigma -> 0 as eps -> 1."
            )
        x_s = params.get('x_s')
        if x_s is not None and float(x_s) <= 0:
            raise ValueError(f"x_s must be positive, got {x_s}")
        return params

    # ------------------------------------------------------------------
    # Sauter diameter (self-contained; matches powerlaw_rumpf)
    # ------------------------------------------------------------------
    def _get_x_s(self, solver) -> float:
        """Sauter diameter [m]: the ``x_s`` parameter, else the cached initial
        Sauter diameter from ``solver.X0``. Raises if neither is available."""
        if self.x_s is not None:
            return float(self.x_s)
        if self._x_s_cache is not None:
            return self._x_s_cache

        X0 = getattr(solver, 'X0', None) if solver is not None else None
        if X0 is None:
            raise ValueError(
                "porosity_compression_dynamik_rumpf needs the granule size: "
                "pass x_s=<Sauter diameter [m]> explicitly, or run it with a "
                "solver that exposes X0."
            )
        X0 = np.asarray(X0, dtype=float).ravel()
        valid = np.isfinite(X0) & (X0 > 0)
        if not np.any(valid):
            raise ValueError(
                "solver.X0 has no positive, finite diameter; cannot form the "
                "Sauter diameter for porosity_compression_dynamik_rumpf."
            )
        X0v = X0[valid]
        if X0v.size == 1:
            x_s = float(X0v[0])
        else:
            d2 = float(np.sum(X0v ** 2))
            d3 = float(np.sum(X0v ** 3))
            if d2 <= 0:
                raise ValueError(
                    "Sum of squared diameters is non-positive; cannot form the "
                    "Sauter diameter for porosity_compression_dynamik_rumpf."
                )
            x_s = d3 / d2
        self._x_s_cache = x_s
        return x_s

    # ------------------------------------------------------------------
    # Rumpf strength: sigma(eps, S) = phi(S) * (1 - eps) / eps
    # ------------------------------------------------------------------
    def _sigma(self, poro: np.ndarray, sat: np.ndarray, x_s: float) -> np.ndarray:
        """Rumpf granule strength [Pa], vectorised. ``poro`` is capped at
        ``poro_max`` for the ``(1 - eps)/eps`` factor only."""
        e = np.minimum(np.clip(np.asarray(poro, dtype=float), 0.0, 1.0), self.poro_max)
        s = np.asarray(sat, dtype=float)
        s = np.where(np.isnan(s), 0.0, np.clip(s, 0.0, 1.0))

        base = self.gamma / x_s                          # [Pa]
        phi_dry = base * self.k_rumpf
        wet_coeff = base * 6.0 * self.alpha * self.cos_delta
        phi_wet = wet_coeff * s
        phi_wet_at_thr = wet_coeff * self.SAT_WET_THRESHOLD
        t = (s - self.SAT_DRY_THRESHOLD) / self.SAT_TRANSITION_WIDTH
        phi_trans = (1.0 - t) * phi_dry + t * phi_wet_at_thr

        phi = np.where(
            s < self.SAT_DRY_THRESHOLD, phi_dry,
            np.where(s > self.SAT_WET_THRESHOLD, phi_wet, phi_trans),
        )
        return phi * (1.0 - e) / e

    def _new_porosity(self, poro_old: np.ndarray, sat: np.ndarray,
                      dt: float, x_s: float) -> np.ndarray:
        """Option C: freeze sigma at ``poro_old``, then the analytic exponential.

        Returns an array shaped like ``poro_old``. NaN entries and entries
        <= eps_min pass through unchanged.
        """
        poro = np.asarray(poro_old, dtype=float)
        sat = np.broadcast_to(np.asarray(sat, dtype=float), poro.shape)
        out = poro.copy()

        finite = ~np.isnan(poro)
        if not np.any(finite):
            return out

        clipped = np.clip(poro[finite], 0.0, 1.0)
        can_compress = clipped > self.eps_min
        if not np.any(can_compress):
            return out

        e = clipped[can_compress]
        s = sat[finite][can_compress]

        sigma = self._sigma(e, s, x_s)
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            bad = int(np.count_nonzero(~np.isfinite(sigma) | (sigma <= 0.0)))
            raise RuntimeError(
                f"porosity_compression_dynamik_rumpf: sigma non-positive or "
                f"non-finite for {bad} particle(s) (x_s={x_s:.3e}, "
                f"eps in [{e.min():.4f}, {e.max():.4f}]). Compression rate "
                f"would be infinite or undefined."
            )

        k_eff = self.rate_nc / sigma
        decay = np.exp(-k_eff * dt)
        updated = np.maximum(self.eps_min, self.eps_min + (e - self.eps_min) * decay)

        compressible = np.zeros(poro.shape, dtype=bool)
        compressible[finite] = can_compress
        out[compressible] = updated
        return out

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
        New porosity after strength-resisted compression over ``dt``.

        Args:
            porosity: Current porosity; NaN and values <= min_porosity pass
                      through unchanged.
            dt: Time step [s].
            saturation: Particle saturation. ``None`` is treated as dry
                      (S = 0.0), the conservative Rumpf assumption.
            solver: Used only to resolve ``x_s`` from ``solver.X0`` when no
                      ``x_s`` parameter was given.
            v_particle, local_stress: Not used.

        Returns:
            porosity_new (>= min_porosity).
        """
        if np.isnan(porosity):
            return porosity
        porosity = max(0.0, min(1.0, porosity))
        if porosity <= self.eps_min:
            return porosity

        x_s = self._get_x_s(solver)
        s = 0.0 if saturation is None else float(saturation)
        out = self._new_porosity(np.array([porosity], dtype=float),
                                 np.array([s], dtype=float), dt, x_s)
        return float(out[0])

    # ------------------------------------------------------------------
    # Vectorised
    # ------------------------------------------------------------------
    def compute_array(self, porosity: np.ndarray, dt: float, solver=None) -> np.ndarray:
        """Vectorised form of :meth:`compute`; matches a loop of scalar calls
        for every porosity in [0, 1] or NaN (the range the solver produces).

        Args:
            porosity: Current porosities (the active slice). NaN and
                      values <= min_porosity pass through unchanged.
            dt:       Time step [s].
            solver:   REQUIRED. Supplies ``saturation`` (index-aligned with
                      ``porosity``) and ``X0`` for the Sauter diameter. This
                      kernel raises without it -- the strength model is not
                      optional.

        Returns:
            Updated porosities, same shape as ``porosity``.
        """
        poro = np.asarray(porosity, dtype=float)
        if solver is None:
            raise ValueError(
                "porosity_compression_dynamik_rumpf.compute_array needs a "
                "solver: it reads saturation and the Sauter diameter to "
                "evaluate the Rumpf strength. Use porosity_compression_dynamik "
                "for a strength-free mixer-speed compression."
            )

        n = poro.shape[0]
        sat = getattr(solver, 'saturation', None)
        if sat is None:
            raise ValueError(
                "porosity_compression_dynamik_rumpf.compute_array: solver has "
                "no 'saturation' array."
            )
        sat = np.asarray(sat, dtype=float)[:n]

        x_s = self._get_x_s(solver)
        return self._new_porosity(poro, sat, dt, x_s)
