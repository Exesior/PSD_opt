"""
Stokes-EKE-Dynamik Kernel: Stokes criterion with an EKE-consistent,
pair-dependent collision velocity.

Same acceptance physics as :mod:`.stokes_dynamik` (Braumann et al. 2007) - the
film, the restitution, St_crit and the mixer-speed law are unchanged. What
changes is the ONE assumption about how fast two particles hit each other:

    stokes_dynamik       every pair hits with the same velocity U
    stokes_eke_dynamik   every pair hits with the same ENERGY E

The second is the assumption the EKE aggregation kernel (`eke_darelius2005`)
already makes for the collision FREQUENCY. Running `eke_darelius2005` together
with `stokes_dynamik` means: fine particles are fast when we count how often
they meet, but slow when we decide whether they stick. This kernel removes
that inconsistency. `mixer_speed.py` (comment on C_BREAK) already derives that
under EKE the impact energy of a pair, 1/2 mu v_rel^2, is the same for every
pair, and the dynamic breakage kernel is built on that. This kernel applies the
same result to the acceptance.

Self-contained:
    This file imports only the abstract base class, numpy and the shared
    mixer-speed constants. It does NOT inherit from `stokes_dynamik`: masses,
    restitution, external liquid, film thickness, St and St_crit are all
    computed here (copied, see `kernels/README.md`, "Die Grundregel").

Why the velocity only enters through the momentum:
    The Stokes criterion is a momentum balance. The liquid film between two
    approaching particles brakes them with F = 3 pi eta R_harm^2 v / x (x = gap).
    Integrated from the film edge h to the asperity height h_a, the film can
    remove the momentum 3 pi eta R_harm^2 ln(h/h_a) per passage; with one
    rebound (restitution e) the pair stays together if

        m_harm * U  <  (1 + 1/e) * 3 pi eta R_harm^2 * ln(h/h_a)      (*)

    which is exactly St < St_crit below. U only appears in the momentum
    p = m_harm * U. If the collision ENERGY is known instead of the velocity,
    p follows from it, and so does U.

Equations (Braumann et al. 2007, Gl. 8 and 10, velocity per pair):
    mu_ij   = m_i m_j / (m_i + m_j)                    reduced mass
    m_harm  = 2 m_i m_j / (m_i + m_j) = 2 mu_ij        harmonic mean mass
    R_harm  = 2 r_i r_j / (r_i + r_j)
    U_ij    = U_coll(n) * sqrt(mu_ref / mu_ij)                       (EKE)
    St      = (m_harm * U_ij) / (3 * pi * eta * R_harm^2)       (Gl. 8)
    St_crit = (1 + 1/e_coag) * ln(h / h_a)                      (Gl. 10)

    St < St_crit  -> particles agglomerate
    St > St_crit  -> particles bounce off

Velocity model (equipartition of fluctuating kinetic energy):
    Every particle carries the same mean fluctuating kinetic energy,
    1/2 m_i <v_i'^2> = 3/2 Theta. For two independently moving particles
    <v_rel^2> = 3 Theta (1/m_i + 1/m_j) = 3 Theta / mu_ij, so the energy
    available in the centre-of-mass frame,

        E = 1/2 mu_ij <v_rel^2> = 3/2 Theta,

    is the SAME for every pair. Hence U_ij ~ mu_ij^(-1/2). The scale is set by
    a reference pair of two PRIMARY PARTICLES, taken from the initial
    particles the same way `powerlaw_rumpf(_dynamic)` takes its Sauter
    diameter:

        U_coll(n) = U_coll_ref * n_mixer^c_vel       velocity of the ref. pair
        x_s       = Sauter diameter of solver.X0      (as in Rumpf)
        eps_0     = initial porosity of the primary particles
        m_ref     = rho_solid * (1 - eps_0) * pi/6 * x_s^3
        mu_ref    = m_ref / 2
        E(n)      = 1/2 * mu_ref * U_coll(n)^2  ~  n^(2 c_vel)

    For a pair of two primary particles the kernel decides exactly like
    `stokes_dynamik`, and `U_coll_ref` reads as "impact velocity of two primary
    particles at n = 1". The reference is not a parameter: density and initial
    porosity are set per system anyway, and a new system needs a new fit.
    It is computed once, on the first collision that reaches the Stokes number.

    eps_0 comes from `solver.V0` (1 - V_solid/V_dry of the initial particles),
    not from `solver.porosity0`: the adapter and the Trials scripts set the
    porosity only after `_initialize_particles()`, so `porosity0` still holds
    0 for a porous start.

    There is deliberately NO upper velocity cap. A particle can be faster than
    the tool: a heavy blade at speed V knocks a light particle at rest to
    (1 + e) V. With the primary particle as reference the impact velocities in
    the AJ30 collision logs stay between 0.05 and 1.6 m/s anyway. The 2 mm DEM
    particles would be a poor reference: the DEM velocity law (0.21 m/s at
    30 m/s) extrapolated with EKE scaling would give ~95 m/s for 34 um fines.

Consequences (equal density, equal film; see the derivation document of
2026-10-01, "stokes_eke_dynamik"):
    - Equal pairs (radius r):   stokes_dynamik      St ~ r       -> stick BELOW r*
                                stokes_eke_dynamik  St ~ r^-1/2  -> stick ABOVE r*
      i.e. the classical growth limit by coalescence (Ennis et al. 1991,
      Adetayo & Ennis 1997) is reversed; a growth limit then has to come from
      breakage.
    - Fine r on granule R >> r: St depends on r, the pair film and e only, not
      on R (R enters the rate only through the collision frequency ~ R^2).
    - Mixer speed: critical size ~ n^(+2 c_vel) here, ~ n^(-c_vel) in
      stokes_dynamik.

Usage:
    >>> kernel = StokesEkeDynamikKernel(
    ...     U_coll_ref=0.4364,       # [m/s] two primary particles at n = 1 (fit)
    ...     n_mixer=30.0,            # [m/s] tool speed, same as all mixer kernels
    ...     c_vel=0.2852,            # DEM exponent
    ...     binder_viscosity=0.1,    # [Pa.s]
    ...     rho_solid=600.0,         # [kg/m^3]
    ...     rho_liquid=1000.0,       # [kg/m^3]
    ...     h_a=500e-9               # [m] surface roughness
    ... )
    >>> accepted = kernel.accept_collision(r1, r2, v_dry1, v_dry2,
    ...                                    i, j, solver=solver)

References:
    Braumann, Goodson, Kraft, Mort (2007), Chem. Eng. Sci. 62, 4717-4728,
        doi:10.1016/j.ces.2007.05.028 (Gl. 8, 10)
    Ennis, Tardos, Pfeffer (1991), Powder Technol. 65, 257-272 (St_crit)
    Hounslow (1998), KONA 16, 179-193, doi:10.14356/kona.1998021 (EKE)
"""

import math
from typing import Any, Dict, Optional

import numpy as np

from ..base import AggAcceptanceKernel
from ..mixer_speed import C_VEL, N_MIXER_DEFAULT


class StokesEkeDynamikKernel(AggAcceptanceKernel):
    """
    Stokes acceptance criterion with an EKE-consistent collision velocity.

    Parameters:
        U_coll_ref: Impact velocity of two PRIMARY PARTICLES at n_mixer = 1
                    [m/s] (default: 0.0794, as in `stokes_dynamik`). Fit
                    parameter. `U_coll = U_coll_ref * n_mixer ** c_vel` is the
                    velocity of that pair; other pairs scale with
                    sqrt(mu_ref / mu_ij). The primary particles are taken
                    from the solver (Sauter diameter of `solver.X0`, initial
                    porosity), not passed.
        n_mixer: Current mixer speed [m/s] (default:
                 `mixer_speed.N_MIXER_DEFAULT` = 20.0). Must match every other
                 mixer-speed-driven kernel in the run.
        c_vel: Exponent of the velocity-over-mixer-speed power law
               (default: `mixer_speed.C_VEL` = 0.2852). c_vel=0.0 makes the
               reference velocity independent of `n_mixer`.
        binder_viscosity: Viscosity of binder liquid [Pa.s]. Water: ~0.001.
        rho_solid: Solid material density [kg/m^3].
        rho_liquid: Liquid density [kg/m^3]. Water: 1000.
        h_a: Minimum film thickness / surface roughness [m].
        debug: Enable debug output for each collision check.

    Note:
        Requires solver access for liquid_volume, porosity, saturation and the
        initial particles (X0, V0). Without a solver, or without liquid, it
        returns False - no liquid bridge, no agglomeration. A solver without X0
        raises a ValueError (as in powerlaw_rumpf) once a collision reaches
        STEP 4.
    """

    # Pre-computed constants for performance
    C_H = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)  # For h calculation
    THREE_PI = 3.0 * np.pi

    @property
    def name(self) -> str:
        return 'stokes_eke_dynamik'

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'U_coll_ref': 0.0794,        # [m/s] reference pair at n = 1 (fit)
            'n_mixer': N_MIXER_DEFAULT,  # shared across all mixer kernels, [m/s]
            'c_vel': C_VEL,              # DEM relative velocity exponent
            'binder_viscosity': 5,       # [Pa.s] same default as stokes_dynamik
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

        # Speed model: velocity of the REFERENCE pair
        self.U_coll_ref = float(self.params['U_coll_ref'])
        self.n_mixer = float(self.params['n_mixer'])
        self.c_vel = float(self.params['c_vel'])
        self.U_coll = self.U_coll_ref * self.n_mixer ** self.c_vel

        # Cache frequently used values
        self.binder_viscosity = float(self.params['binder_viscosity'])
        self.rho_solid = float(self.params['rho_solid'])
        self.rho_liquid = float(self.params['rho_liquid'])
        self.h_a = float(self.params['h_a'])
        self.debug = bool(self.params.get('debug', False))

        # Reference pair: two primary particles, computed once on first use
        # (see _get_mu_ref), like x_s in powerlaw_rumpf(_dynamic).
        self.x_s = None            # [m]   Sauter diameter of the primaries
        self.eps_0 = None          # [-]   initial porosity of the primaries
        self.m_ref = None          # [kg]  mass of one primary particle
        self.mu_ref = None         # [kg]  reduced mass of the reference pair
        self.E_coll = None         # [J]   collision energy, same for every pair

    # ------------------------------------------------------------------
    # Helpers (copied from stokes_dynamik, see module docstring)
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

    def _compute_sauter_diameter(self, solver) -> float:
        """
        Sauter mean diameter d32 = sum(d_i^3) / sum(d_i^2) of ``solver.X0``.

        Copied from ``powerlaw_rumpf_dynamic`` (kernels are self-contained).
        For monodisperse spheres this equals the particle diameter.
        """
        if not hasattr(solver, 'X0') or solver.X0 is None:
            raise ValueError(
                "Solver has no X0 attribute. Cannot compute Sauter diameter "
                "of the primary particles.")

        X0 = np.asarray(solver.X0, dtype=float).ravel()

        valid = np.isfinite(X0) & (X0 > 0)
        if not np.any(valid):
            raise ValueError(
                "solver.X0 contains no valid (positive, finite) diameters. "
                "Cannot compute Sauter diameter.")

        X0_valid = X0[valid]

        if X0_valid.size == 1:
            return float(X0_valid[0])

        mean_d = np.mean(X0_valid)
        std_d = np.std(X0_valid)
        if std_d / mean_d < 1e-6:
            return float(mean_d)

        return float(np.sum(X0_valid ** 3) / np.sum(X0_valid ** 2))

    def _get_mu_ref(self, solver) -> float:
        """
        Reduced mass of the reference pair (two primary particles) [kg].

            m_ref = rho_solid * (1 - eps_0) * pi/6 * x_s^3,  mu_ref = m_ref / 2

        Computed once on first use, like x_s in powerlaw_rumpf(_dynamic).
        """
        if self.mu_ref is None:
            self.x_s = self._compute_sauter_diameter(solver)
            V0 = np.asarray(solver.V0, dtype=float)       # rows: solid..., dry
            self.eps_0 = 1.0 - float(V0[:-1].sum() / V0[-1].sum())
            self.m_ref = (self.rho_solid * (1.0 - self.eps_0)
                          * np.pi / 6.0 * self.x_s ** 3)
            self.mu_ref = 0.5 * self.m_ref
            self.E_coll = 0.5 * self.mu_ref * self.U_coll ** 2
        return self.mu_ref

    def pair_velocity(self, m1: float, m2: float) -> float:
        """
        EKE impact velocity of a pair [m/s].

            U_ij = U_coll * sqrt(mu_ref / mu_ij),  mu_ij = m1 m2 / (m1 + m2)

        Equivalent to sqrt(2 E_coll / mu_ij): every pair brings the same
        energy E_coll into the collision. Needs the reference pair, i.e. one
        call of _get_mu_ref(solver) first.
        """
        mu = m1 * m2 / (m1 + m2)
        return self.U_coll * math.sqrt(self.mu_ref / mu)

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

        Identical to `stokes_dynamik` except for STEP 4, where the impact
        velocity is computed per pair from the common collision energy.

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

        # STEP 4: Stokes number (Gl. 8) with the EKE pair velocity
        m_harm = 2.0 * m1 * m2 / (m1 + m2)
        R_harm = 2.0 * r1 * r2 / (r1 + r2)

        if R_harm <= 0:
            return False

        self._get_mu_ref(solver)
        U_pair = self.pair_velocity(m1, m2)
        St =(m_harm * U_pair) / (self.THREE_PI * self.binder_viscosity * R_harm ** 2)

        if not np.isfinite(St) or St <= 0:
            return False

        # STEP 5: critical Stokes number (Gl. 10)
        St_crit = (1.0 + 1.0 / e_coag) * np.log(h_harm / self.h_a)

        if not np.isfinite(St_crit):
            return False

        if self.debug:
            print(f"[stokes_eke_dynamik] r=({r1:.3e},{r2:.3e}) U={U_pair:.4f} m/s "
                  f"St={St:.4e} St_crit={St_crit:.4e}")

        # STEP 6: acceptance decision
        return St < St_crit
