"""
Stokes-Krit Kernel: Agglomeration Acceptance based on Stokes Criterion.

Implementation of Braumann et al. (2007) Stokes criterion for wet granulation.
Particles agglomerate only if St < St_crit, otherwise they bounce off.

Reference:
    Braumann et al., "Modelling and validation of granulation with heterogeneous 
    binder dispersion and chemical reaction", Chemical Engineering Science, 2007.
    
Equations:
    St = (m_harm * U) / (3 * π * η * R_harm²)           (Gl. 8)
    St_crit = (1 + 1/e_coag) * ln(h / h_a)              (Gl. 10)
    
    where:
    - m_harm: Harmonic mean of particle masses
    - R_harm: Harmonic mean of particle radii
    - U: Collision velocity [m/s]
    - η: Binder viscosity [Pa·s]
    - e_coag: Effective restitution coefficient = sqrt(e_i * e_j)
    - e_i: m_solid_i / m_total_i (since e_solid=1, e_liquid=0)
    - h: Liquid film thickness (from external liquid volume)
    - h_a: Minimum film thickness (surface roughness) [m]

Usage:
    >>> kernel = StokesKritKernel(
    ...     U_coll=1.0,              # [m/s]
    ...     binder_viscosity=0.1,    # [Pa·s]
    ...     rho_solid=2500.0,        # [kg/m³]
    ...     rho_liquid=1000.0,       # [kg/m³]
    ...     h_a=500e-9               # [m] surface roughness
    ... )
    >>> accepted = kernel.accept_collision(r1, r2, v_dry1, v_dry2, solver=solver)
"""

from typing import Any, Dict, Optional
import numpy as np

from ..base import AggAcceptanceKernel


class StokesKritKernel(AggAcceptanceKernel):
    """
    Stokes criterion for agglomeration acceptance (Braumann et al. 2007).
    
    Physical Model:
        Agglomeration succeeds only if inertial forces are overcome by viscous
        dissipation in the liquid bridge between particles. The dimensionless
        Stokes number compares these forces.
        
        - St < St_crit: Particles agglomerate (viscous forces dominate)
        - St > St_crit: Particles bounce off (inertia dominates)
    
    Parameters:
        U_coll: Collision velocity [m/s]. Typical: 0.5-2.0 m/s for high-shear.
        binder_viscosity: Viscosity of binder liquid [Pa·s]. Water: ~0.001 Pa·s.
        rho_solid: Solid material density [kg/m³]. Typical: 1500-3000 kg/m³.
        rho_liquid: Liquid density [kg/m³]. Water: 1000 kg/m³.
        h_a: Minimum film thickness / surface roughness [m]. Typical: 100nm-1µm.
    
    Note:
        This kernel requires solver access to retrieve liquid_volume for computing
        external liquid and thus film thickness. If solver is not provided or
        liquid_volume doesn't exist, returns False (no agglomeration without liquid).
    """
    
    # Pre-computed constants for performance
    C_H = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)  # For h calculation
    THREE_PI = 3.0 * np.pi
    
    @property
    def name(self) -> str:
        return 'stokes_krit'
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'U_coll': 1.0,           # [m/s] collision velocity
            'binder_viscosity': 5,   # [Pa·s] typical for aqueous binders
            'rho_solid': 2500.0,     # [kg/m³] typical solid density
            'rho_liquid': 1000.0,    # [kg/m³] water density
            'h_a': 500e-9,           # [m] 500 nm surface roughness
            'debug': False,          # Enable debug output for each collision check
        }
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params['U_coll'] <= 0:
            raise ValueError(f"U_coll must be positive, got {params['U_coll']}")
        if params['binder_viscosity'] <= 0:
            raise ValueError(f"binder_viscosity must be positive, got {params['binder_viscosity']}")
        if params['rho_solid'] <= 0:
            raise ValueError(f"rho_solid must be positive, got {params['rho_solid']}")
        if params['rho_liquid'] <= 0:
            raise ValueError(f"rho_liquid must be positive, got {params['rho_liquid']}")
        if params['h_a'] <= 0:
            raise ValueError(f"h_a must be positive, got {params['h_a']}")
        return params
    
    def __init__(self, **params):
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache frequently used values
        self.U_coll = float(self.params['U_coll'])
        self.binder_viscosity = float(self.params['binder_viscosity'])
        self.rho_solid = float(self.params['rho_solid'])
        self.rho_liquid = float(self.params['rho_liquid'])
        self.h_a = float(self.params['h_a'])
        self.debug = bool(self.params.get('debug', False))
    
    def _compute_particle_mass(self, v_dry: float, particle_idx: int, solver: Any) -> tuple[float, float]:
        """
        Compute total mass and solid mass of a particle.
        
        Args:
            v_dry: Dry volume [m³]
            particle_idx: Particle index in solver arrays
            solver: Solver instance
        
        Returns:
            Tuple of (m_total, m_solid) in [kg]
        """
        # Get porosity to compute solid volume
        if hasattr(solver, 'porosity') and particle_idx is not None:
            poro = solver.porosity[particle_idx]
            if np.isnan(poro):
                # Vollkörper: V_solid = V_dry
                v_solid = v_dry
            else:
                # Porous: V_solid = V_dry × (1 - poro)
                v_solid = v_dry * (1.0 - poro)
        else:
            # No porosity info: assume Vollkörper
            v_solid = v_dry
        
        # Get total liquid volume
        v_liquid_total = 0.0
        if hasattr(solver, 'liquid_volume') and particle_idx is not None:
            v_liquid_total = float(solver.liquid_volume[particle_idx])
        
        # Compute masses
        m_solid = self.rho_solid * v_solid
        m_liquid = self.rho_liquid * v_liquid_total
        m_total = m_solid + m_liquid
        
        return m_total, m_solid
    
    def _compute_external_liquid(self, v_dry: float, particle_idx: int, solver: Any) -> float:
        """
        Compute external liquid volume of a particle.
        
        External liquid = total liquid - internal liquid (in pores)
        
        Formula:
            V_liq_external = V_liq_total - V_liq_internal
            V_liq_internal = V_pore × saturation
            V_pore = V_dry × porosity
        
        For Vollkörper (no pores, NaN porosity): all liquid is external.
        
        Args:
            v_dry: Dry volume [m³]
            particle_idx: Particle index in solver arrays
            solver: Solver instance
        
        Returns:
            External liquid volume [m³]
        """
        if particle_idx is None:
            return 0.0
        
        # Get total liquid volume
        if not hasattr(solver, 'liquid_volume'):
            return 0.0
        
        v_liq_total = float(solver.liquid_volume[particle_idx])
        
        if v_liq_total <= 0:
            return 0.0  # No liquid at all
        
        # Check if particle has pores (porosity info)
        if not hasattr(solver, 'porosity'):
            # No porosity info: assume all liquid is external
            return v_liq_total
        
        poro = solver.porosity[particle_idx]
        
        if np.isnan(poro):
            # Vollkörper (no pores): all liquid is external
            return v_liq_total
        
        # Porous particle: compute internal liquid
        if not hasattr(solver, 'saturation'):
            # No saturation info: assume all liquid is external (conservative)
            return v_liq_total
        
        saturation = solver.saturation[particle_idx]
        
        if np.isnan(saturation) or saturation <= 0:
            # No saturation: all liquid is external
            return v_liq_total
        
        # Compute internal liquid (in pores)
        v_pore = v_dry * poro
        v_liq_internal = v_pore * saturation
        
        # External liquid = total - internal
        v_liq_external = max(0.0, v_liq_total - v_liq_internal)
        
        return v_liq_external
    
    def _compute_film_thickness(self, v_dry: float, v_liq_ext: float) -> float:
        """
        Compute liquid film thickness from dry volume and external liquid.
        
        Uses optimized formula: h = C_H * (V_wet^(1/3) - V_dry^(1/3))
        where C_H = (3/(4π))^(1/3)
        
        Args:
            v_dry: Dry volume [m³]
            v_liq_ext: External liquid volume [m³]
        
        Returns:
            Film thickness h [m]
        """
        if v_liq_ext <= 0:
            return 0.0
        
        v_wet = v_dry + v_liq_ext
        
        # Optimized: pre-computed constant, fewer operations
        h = self.C_H * (v_wet ** (1.0/3.0) - v_dry ** (1.0/3.0))
        return h
    
    def accept_collision(self,
                         r1: float, r2: float,
                         v_dry1: float, v_dry2: float,
                         particle1_idx: Optional[int] = None,
                         particle2_idx: Optional[int] = None,
                         solver: Optional[Any] = None
                         ) -> bool:
        """
        Determine if collision results in agglomeration using Stokes criterion.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            v_dry1: Dry volume of particle 1 [m³]
            v_dry2: Dry volume of particle 2 [m³]
            particle1_idx: Index of particle 1 in solver arrays
            particle2_idx: Index of particle 2 in solver arrays
            solver: Reference to solver for accessing state arrays
        
        Returns:
            accepted: True if St < St_crit, False otherwise
        """
        # Check if solver provides necessary data
        if solver is None:
            return False  # Can't compute without solver
        
        if particle1_idx is None or particle2_idx is None:
            return False  # Need indices to access solver arrays
        
        # STEP 1: Compute masses and solid fractions
        m1, m_solid1 = self._compute_particle_mass(v_dry1, particle1_idx, solver)
        m2, m_solid2 = self._compute_particle_mass(v_dry2, particle2_idx, solver)
        
        if m1 <= 0 or m2 <= 0:
            return False  # Invalid masses
        
        # STEP 2: Compute effective restitution coefficients
        # e_i = m_solid / m_total (since e_solid=1, e_liquid=0)
        e1 = m_solid1 / m1
        e2 = m_solid2 / m2
        
        if e1 <= 0 or e2 <= 0:
            return False  # No solid content
        
        # Geometric mean for pair
        e_coag = np.sqrt(e1 * e2)
        
        # STEP 3: Compute external liquid and film thickness
        v_liq_ext1 = self._compute_external_liquid(v_dry1, particle1_idx, solver)
        v_liq_ext2 = self._compute_external_liquid(v_dry2, particle2_idx, solver)
        
        h1 = self._compute_film_thickness(v_dry1, v_liq_ext1)
        h2 = self._compute_film_thickness(v_dry2, v_liq_ext2)
        
        # Check if at least one particle has liquid (asymmetric case allowed)
        # Only reject if BOTH particles are dry
        if h1 + h2 <= 0:
            return False  # No liquid bridge possible
        
        # Compute effective film thickness for liquid bridge
        # Use arithmetic mean to handle asymmetric case correctly
        # Arithmetic mean: (h1 + h2) / 2 works for all cases including h1=0 or h2=0
        h_harm = (h1 + h2) / 2.0
        
        # Early exit: film too thin
        if h_harm <= self.h_a:
            return False  # ln(h/h_a) <= 0 → St_crit <= 0
        
        # STEP 4: Compute Stokes number (Gl. 8)
        # Harmonic means
        m_harm = 2.0 * m1 * m2 / (m1 + m2)
        R_harm = 2.0 * r1 * r2 / (r1 + r2)
        
        if R_harm <= 0:
            return False
        
        # St = (m_harm * U) / (3 * π * η * R_harm²)
        St = (m_harm * self.U_coll) / (self.THREE_PI * self.binder_viscosity * R_harm ** 2)
        
        if not np.isfinite(St) or St <= 0:
            return False
        
        # STEP 5: Compute critical Stokes number (Gl. 10)
        # St_crit = (1 + 1/e_coag) * ln(h / h_a)
        St_crit = (1.0 + 1.0 / e_coag) * np.log(h_harm / self.h_a)
        
        if not np.isfinite(St_crit):
            return False
        
        # STEP 6: Acceptance decision
        return St < St_crit
