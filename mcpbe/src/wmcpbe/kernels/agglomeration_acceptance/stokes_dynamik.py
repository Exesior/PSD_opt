"""
Stokes-Dynamik Kernel: Stokes criterion with a mixer-speed-dependent collision
velocity.

Same acceptance physics as :mod:`.stokes_krit` (Braumann et al. 2007) - only
the collision velocity U is no longer a constant. With a gas continuum the
granules are set in motion by the impeller alone, so a faster mixer means
harder impacts, a larger Stokes number and therefore MORE rebounds.
`stokes_krit` cannot express that: its U_coll is a single fitted number, so
turning the mixer up changes the collision *rate* (via the aggregation kernel)
but not the collision *outcome*.

Self-contained:
    This file imports only the abstract base class, numpy and the shared
    mixer-speed constants. It does NOT inherit from `stokes_krit`: the masses,
    the effective restitution, the external liquid, the film thickness, St and
    St_crit are all computed here. Handing this one file (plus `base.py` and
    `mixer_speed.py`) to someone else gives a working kernel even if no other
    acceptance kernel is present.

    The duplication against `stokes_krit` is intentional and guarded:
    `Trials/test_dry_mixer_kernels.py` asserts that at `n_mixer == 1.0` this
    kernel returns exactly the same accept/reject decisions as `stokes_krit`
    with `U_coll = U_coll_ref` (since `n_mixer ** c_vel == 1.0` there for any
    `c_vel`), so a change to one that is not carried over to the other fails
    the suite.

Equations (Braumann et al. 2007):
    St      = (m_harm * U) / (3 * pi * eta * R_harm^2)      (Gl. 8)
    St_crit = (1 + 1/e_coag) * ln(h / h_a)                  (Gl. 10)

    where:
    - m_harm: harmonic mean of the particle masses
    - R_harm: harmonic mean of the particle radii
    - U: collision velocity [m/s], here mixer-speed dependent (see below)
    - eta: binder viscosity [Pa.s]
    - e_coag: effective restitution coefficient = sqrt(e_i * e_j)
    - e_i: m_solid_i / m_total_i (since e_solid = 1, e_liquid = 0)
    - h: liquid film thickness (from the external liquid volume)
    - h_a: minimum film thickness (surface roughness) [m]

    St < St_crit  -> particles agglomerate (viscous forces dominate)
    St > St_crit  -> particles bounce off (inertia dominates)

Velocity model:
    U_coll(n) = U_coll_ref * n_mixer^C_VEL

    Direct power law, no reference speed. The DEM study of a comparable mixer
    reports the mean relative particle-particle collision velocity as a
    function of the mixer speed:

        v_rel = 0.0794 * n^0.2852   (least-squares trendline over the DEM
                                      operating points, n in m/s, range 5-40)

    Unlike the aggregation kernels (`eke_darelius2005`, `etm_darelius2005`) and
    the dynamic breakage kernel, where only the EXPONENT transfers and the DEM
    prefactor is absorbed into an admittedly-unphysical fit constant, this
    kernel takes the DEM fit 1:1 - prefactor included - as `U_coll_ref`'s
    default. That is a deliberate, explicitly unphysical simplification, not
    an oversight:

    `U_coll_ref` was ALWAYS just a guess. `stokes_krit`'s `U_coll` has no
    experimental backing in this project either - Braumann et al. give no
    value transferable to this system, so 1.0 m/s was picked as a plausible
    order of magnitude and left for the eventual joint fit to correct. Once
    that is accepted, inventing a second, only-slightly-less-arbitrary
    reference-speed construction to keep `U_coll_ref` "physically clean" buys
    nothing: it hides one unidentifiable parameter (`n_ref`) behind another
    (`U_coll_ref`'s de-facto meaning already depended on which `n_ref` was
    picked). Taking the DEM number directly is more honest about being a
    placeholder, and it is one free parameter fewer for the eventual joint fit
    against experimental data - see `mcpbe/docs/historical/Nucleation_Propensity_Blockade.md`
    and `kernels/README.md` for the same reasoning applied to the mixer-speed
    exponents generally.

    Practical consequence: `n_mixer` must be read in the SAME units as the DEM
    study (m/s, roughly 5-40) for this kernel specifically, because there is no
    longer a free prefactor to absorb a unit mismatch. The aggregation kernels
    are unaffected by this - their prefactors stay free - but because
    `assert_consistent_mixer_speed` requires one shared `n_mixer` across every
    active kernel, the whole run is implicitly pinned to that scale the moment
    `stokes_dynamik` is active.

    Consequence of the exponent being ~0.29: over the DEM range n = 5 to 40 the
    collision velocity changes by a factor 8^0.2852 = 1.83, i.e. it not quite
    doubles. That is a considerably stronger process dependence than the
    collision *frequency* shows (exponent 0.0995, factor 1.23 over the same
    range), so the mixer speed acts mainly through the impact severity, not
    through how often particles meet.

Relation to the other mixer-speed kernels:
    `eke_darelius2005`, `etm_darelius2005`, `powerlaw_rumpf_dynamic` and this
    kernel all take an `n_mixer`, and all of them default to the same shared
    constant in :mod:`..mixer_speed`. Each still reads its own parameter, so a
    sweep must set the same value on every one of them - but a mismatch is no
    longer silent: `assert_consistent_mixer_speed` refuses a configuration in
    which two active kernels disagree, because collision frequency, collision
    severity and breakage would then describe different machines.

Usage:
    >>> kernel = StokesDynamikKernel(
    ...     U_coll_ref=0.0794,       # [m/s], DEM fit prefactor, taken 1:1
    ...     n_mixer=20.0,            # current mixer speed [m/s]
    ...     c_vel=0.2852,            # DEM exponent
    ...     binder_viscosity=0.1,    # [Pa.s]
    ...     rho_solid=2500.0,        # [kg/m^3]
    ...     rho_liquid=1000.0,       # [kg/m^3]
    ...     h_a=500e-9               # [m] surface roughness
    ... )
    >>> accepted = kernel.accept_collision(r1, r2, v_dry1, v_dry2, solver=solver)
"""

import math
from typing import Any, Dict, Optional

import numpy as np

from ..base import AggAcceptanceKernel
from ..mixer_speed import C_VEL, N_MIXER_DEFAULT


class StokesDynamikKernel(AggAcceptanceKernel):
    """
    Stokes acceptance criterion with a mixer-speed-dependent collision velocity.

    Parameters:
        U_coll_ref: Collision velocity prefactor [m/s] (default: 0.0794, the
                    DEM fit's prefactor, taken 1:1 - see module docstring for
                    why that is a deliberate simplification, not a real
                    calibration). `U_coll = U_coll_ref * n_mixer ** c_vel`.
        n_mixer: Current mixer speed [m/s] (default:
                 `mixer_speed.N_MIXER_DEFAULT` = 20.0). Must match every other
                 mixer-speed-driven kernel in the run. Unlike the aggregation
                 kernels, the unit here is NOT free: with `U_coll_ref` fixed at
                 the literal DEM value, `n_mixer` must be read on the same
                 scale as the DEM study (m/s, roughly 5-40).
        c_vel: Exponent of the velocity-over-mixer-speed power law
               (default: `mixer_speed.C_VEL` = 0.2852). Set c_vel=0.0 to make
               `U_coll` constant (= `U_coll_ref`) at any `n_mixer`.
        binder_viscosity: Viscosity of binder liquid [Pa.s]. Water: ~0.001.
        rho_solid: Solid material density [kg/m^3]. Typical: 1500-3000.
        rho_liquid: Liquid density [kg/m^3]. Water: 1000.
        h_a: Minimum film thickness / surface roughness [m]. Typical 100nm-1um.
        debug: Enable debug output for each collision check.

    Note:
        This kernel requires solver access to retrieve liquid_volume for
        computing the external liquid and thus the film thickness. Without a
        solver, or without liquid, it returns False - no liquid bridge, no
        agglomeration.

        At `n_mixer == 1.0` the decisions are identical to `stokes_krit` with
        `U_coll = U_coll_ref`, because `1.0 ** c == 1.0` in IEEE-754 for any
        finite c. That is a mathematical identity, not a calibration point -
        1.0 m/s carries no special physical meaning here.
    """

    # Pre-computed constants for performance
    C_H = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)  # For h calculation
    THREE_PI = 3.0 * np.pi

    @property
    def name(self) -> str:
        return 'stokes_dynamik'

    def get_default_params(self) -> Dict[str, Any]:
        return {
            # [m/s] DEM fit prefactor (y = 0.0794 * x^0.2852), taken 1:1.
            # Explicitly unphysical -- see module docstring. Not worse than
            # the 1.0 m/s guess `stokes_krit` used, and one free parameter
            # (n_ref) fewer for the eventual joint fit.
            'U_coll_ref': 0.0794,
            'n_mixer': N_MIXER_DEFAULT,  # shared across all mixer kernels, [m/s]
            'c_vel': C_VEL,              # DEM relative velocity exponent
            'binder_viscosity': 5,       # [Pa.s] typical for aqueous binders
            'rho_solid': 2500.0,         # [kg/m^3] typical solid density
            'rho_liquid': 1000.0,        # [kg/m^3] water density
            'h_a': 500e-9,               # [m] 500 nm surface roughness
            'debug': False,              # Debug output for each collision check
        }

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters."""
        if params['U_coll_ref'] <= 0:
            raise ValueError(
                f"U_coll_ref must be positive, got {params['U_coll_ref']}")
        if params['binder_viscosity'] <= 0:
            raise ValueError(
                f"binder_viscosity must be positive, got {params['binder_viscosity']}")
        if params['rho_solid'] <= 0:
            raise ValueError(f"rho_solid must be positive, got {params['rho_solid']}")
        if params['rho_liquid'] <= 0:
            raise ValueError(f"rho_liquid must be positive, got {params['rho_liquid']}")
        if params['h_a'] <= 0:
            raise ValueError(f"h_a must be positive, got {params['h_a']}")

        if params['n_mixer'] <= 0:
            raise ValueError(
                f"n_mixer must be positive, got {params['n_mixer']}. "
                "A non-positive speed makes n_mixer**c_vel undefined for a "
                "fractional exponent. Use c_vel=0.0 to disable the mixer "
                "speed dependence instead."
            )
        c_vel = float(params['c_vel'])
        if not math.isfinite(c_vel):
            raise ValueError(f"c_vel must be finite, got {c_vel}")

        return params

    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)

        # Speed model
        self.U_coll_ref = float(self.params['U_coll_ref'])
        self.n_mixer = float(self.params['n_mixer'])
        self.c_vel = float(self.params['c_vel'])

        # The single quantity the acceptance criterion below actually reads.
        self.U_coll = self.U_coll_ref * self.n_mixer ** self.c_vel

        # Cache frequently used values
        self.binder_viscosity = float(self.params['binder_viscosity'])
        self.rho_solid = float(self.params['rho_solid'])
        self.rho_liquid = float(self.params['rho_liquid'])
        self.h_a = float(self.params['h_a'])
        self.debug = bool(self.params.get('debug', False))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_particle_mass(self, v_dry: float, particle_idx: int,
                               solver: Any) -> tuple[float, float]:
        """
        Total mass and solid mass of a particle [kg].

        Returns:
            Tuple of (m_total, m_solid)
        """
        # Get porosity to compute solid volume
        if hasattr(solver, 'porosity') and particle_idx is not None:
            poro = solver.porosity[particle_idx]
            if np.isnan(poro):
                # Vollkoerper: V_solid = V_dry
                v_solid = v_dry
            else:
                # Porous: V_solid = V_dry * (1 - poro)
                v_solid = v_dry * (1.0 - poro)
        else:
            # No porosity info: assume Vollkoerper
            v_solid = v_dry

        # Get total liquid volume
        v_liquid_total = 0.0
        if hasattr(solver, 'liquid_volume') and particle_idx is not None:
            v_liquid_total = float(solver.liquid_volume[particle_idx])

        m_solid = self.rho_solid * v_solid
        m_liquid = self.rho_liquid * v_liquid_total
        m_total = m_solid + m_liquid

        return m_total, m_solid

    def _compute_external_liquid(self, v_dry: float, particle_idx: int,
                                 solver: Any) -> float:
        """
        External liquid volume of a particle [m^3].

            V_liq_external = V_liq_total - V_liq_internal
            V_liq_internal = V_pore * saturation
            V_pore         = V_dry * porosity

        For Vollkoerper (no pores, NaN porosity) all liquid is external.
        """
        if particle_idx is None:
            return 0.0

        if not hasattr(solver, 'liquid_volume'):
            return 0.0

        v_liq_total = float(solver.liquid_volume[particle_idx])

        if v_liq_total <= 0:
            return 0.0  # No liquid at all

        if not hasattr(solver, 'porosity'):
            # No porosity info: assume all liquid is external
            return v_liq_total

        poro = solver.porosity[particle_idx]

        if np.isnan(poro):
            # Vollkoerper (no pores): all liquid is external
            return v_liq_total

        if not hasattr(solver, 'saturation'):
            # No saturation info: assume all liquid is external (conservative)
            return v_liq_total

        saturation = solver.saturation[particle_idx]

        if np.isnan(saturation) or saturation <= 0:
            return v_liq_total

        v_pore = v_dry * poro
        v_liq_internal = v_pore * saturation

        return max(0.0, v_liq_total - v_liq_internal)

    def _compute_film_thickness(self, v_dry: float, v_liq_ext: float) -> float:
        """
        Liquid film thickness [m] from dry volume and external liquid.

        h = C_H * (V_wet^(1/3) - V_dry^(1/3)),  C_H = (3/(4*pi))^(1/3)
        """
        if v_liq_ext <= 0:
            return 0.0

        v_wet = v_dry + v_liq_ext

        h = self.C_H * (v_wet ** (1.0 / 3.0) - v_dry ** (1.0 / 3.0))
        return h

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------
    def accept_collision(self,
                         r1: float, r2: float,
                         v_dry1: float, v_dry2: float,
                         particle1_idx: Optional[int] = None,
                         particle2_idx: Optional[int] = None,
                         solver: Optional[Any] = None
                         ) -> bool:
        """
        Determine if a collision results in agglomeration (St < St_crit).

        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            v_dry1: Dry volume of particle 1 [m^3]
            v_dry2: Dry volume of particle 2 [m^3]
            particle1_idx: Index of particle 1 in solver arrays
            particle2_idx: Index of particle 2 in solver arrays
            solver: Reference to solver for accessing state arrays

        Returns:
            accepted: True if St < St_crit, False otherwise
        """
        if solver is None:
            return False  # Can't compute without solver

        if particle1_idx is None or particle2_idx is None:
            return False  # Need indices to access solver arrays

        # STEP 1: masses and solid fractions
        m1, m_solid1 = self._compute_particle_mass(v_dry1, particle1_idx, solver)
        m2, m_solid2 = self._compute_particle_mass(v_dry2, particle2_idx, solver)

        if m1 <= 0 or m2 <= 0:
            return False  # Invalid masses

        # STEP 2: effective restitution coefficients
        # e_i = m_solid / m_total (since e_solid = 1, e_liquid = 0)
        e1 = m_solid1 / m1
        e2 = m_solid2 / m2

        if e1 <= 0 or e2 <= 0:
            return False  # No solid content

        e_coag = np.sqrt(e1 * e2)

        # STEP 3: external liquid and film thickness
        v_liq_ext1 = self._compute_external_liquid(v_dry1, particle1_idx, solver)
        v_liq_ext2 = self._compute_external_liquid(v_dry2, particle2_idx, solver)

        h1 = self._compute_film_thickness(v_dry1, v_liq_ext1)
        h2 = self._compute_film_thickness(v_dry2, v_liq_ext2)

        # Only reject if BOTH particles are dry (asymmetric case allowed)
        if h1 + h2 <= 0:
            return False  # No liquid bridge possible

        # Arithmetic mean handles the asymmetric case (h1 = 0 or h2 = 0)
        h_harm = (h1 + h2) / 2.0

        # Early exit: film too thin -> ln(h/h_a) <= 0 -> St_crit <= 0
        if h_harm <= self.h_a:
            return False

        # STEP 4: Stokes number (Gl. 8)
        m_harm = 2.0 * m1 * m2 / (m1 + m2)
        R_harm = 2.0 * r1 * r2 / (r1 + r2)

        if R_harm <= 0:
            return False

        St = (m_harm * self.U_coll) / (self.THREE_PI * self.binder_viscosity * R_harm ** 2)

        if not np.isfinite(St) or St <= 0:
            return False

        # STEP 5: critical Stokes number (Gl. 10)
        St_crit = (1.0 + 1.0 / e_coag) * np.log(h_harm / self.h_a)

        if not np.isfinite(St_crit):
            return False

        # STEP 6: acceptance decision
        return St < St_crit
