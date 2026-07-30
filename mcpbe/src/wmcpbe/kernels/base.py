"""
Base Classes for WMCPBE Kernel Framework.

This module defines abstract base classes for all physics kernels.
Each kernel category has its own base class specifying the required interface.

Kernels are bound to solver instances at initialization time and should not
be changed during simulation. This ensures reproducibility and performance.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import numpy as np


class KernelBase(ABC):
    """
    Abstract base class for all physics kernels.
    
    Kernels encapsulate physical models for specific processes.
    They are instantiated with parameters and bound to a solver at init.
    
    Attributes:
        params: Dictionary of kernel parameters (set in __init__)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this kernel (e.g., 'shear_chin1998')."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """
        Category of this kernel.
        
        One of: 'aggregation', 'breakage', 'porosity_growth', 
                'compression', 'liquid_distribution'
        """
        pass
    
    @abstractmethod
    def __init__(self, **params):
        """
        Initialize kernel with parameters.
        
        Subclasses should call super().__init__() and store params.
        """
        self.params = dict(params)
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default parameters for this kernel.
        
        Override in subclasses to specify defaults.
        """
        return {}
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and optionally modify parameters.
        
        Override in subclasses for custom validation.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Validated parameter dictionary
            
        Raises:
            ValueError: If parameters are invalid
        """
        return params
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


# =============================================================================
# Aggregation Kernels
# =============================================================================

class AggregationKernel(KernelBase):
    """
    Base class for agglomeration/collision kernels.
    
    Computes collision frequency β(i,j) between two particles.
    The collision frequency determines the rate at which particles collide
    and potentially agglomerate.
    
    Physical models include:
    - Shear-induced collisions (Smoluchowski)
    - Brownian diffusion
    - Differential settling
    - Turbulent inertia
    - Liquid bridge enhancement (for wet granulation)
    """
    
    @property
    def category(self) -> str:
        return 'aggregation'
    
    @abstractmethod
    def compute_beta(self, r1: float, r2: float,
                     particle1_idx: Optional[int] = None,
                     particle2_idx: Optional[int] = None,
                     solver: Optional[Any] = None) -> float:
        """
        Compute collision frequency β(i,j) for two particles.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            particle1_idx: Index of particle 1 in solver arrays (optional)
            particle2_idx: Index of particle 2 in solver arrays (optional)
            solver: Reference to solver for accessing state arrays (optional)
        
        Returns:
            beta: Collision frequency [m³/s]
            
        Note:
            The solver reference allows kernels to access additional particle
            properties like saturation, liquid content, porosity, etc.
            Kernels should handle cases where these attributes don't exist
            (backward compatibility with dry simulations).
        """
        pass


# =============================================================================
# Breakage Kernels
# =============================================================================

class BreakageKernel(KernelBase):
    """
    Base class for breakage/fragmentation kernels.
    
    Computes breakage rate S(V) and fragment distribution for particles.
    Breakage occurs when mechanical stress exceeds particle strength.
    
    Physical models include:
    - Power-law breakage rates
    - Stress-based Weibull models
    - Attrition models
    - Erosion models
    """
    
    @property
    def category(self) -> str:
        return 'breakage'
    
    @abstractmethod
    def compute_rate(self, v_particle: float,
                     particle_idx: Optional[int] = None,
                     solver: Optional[Any] = None) -> float:
        """
        Compute breakage rate S(V) for a particle.
        
        Args:
            v_particle: Particle volume [m³]
            particle_idx: Index of particle in solver arrays (optional)
            solver: Reference to solver for accessing state arrays (optional)
        
        Returns:
            rate: Breakage rate [1/s]
            
        Note:
            Higher rates mean faster breakage. The rate can depend on
            particle size, porosity, saturation, and local stress conditions.
        """
        pass
    
    def compute_fragment_distribution(self, v_parent: np.ndarray,
                                       n_fragments_target: float,
                                       solver: Optional[Any] = None
                                       ) -> Optional[List[np.ndarray]]:
        """
        Generate fragment volumes from parent particle.
        
        Override this method for custom fragment distributions.
        Default implementation returns None, which triggers the standard
        stepwise sampling from CDF tables.
        
        Args:
            v_parent: Parent particle volume array (dim,) [m³]
            n_fragments_target: Expected number of fragments
            solver: Reference to solver (optional)
        
        Returns:
            List of fragment volume arrays, or None to use default method
        """
        return None


# =============================================================================
# Porosity Growth Kernels (event-based)
# =============================================================================

class PorosityGrowthKernel(KernelBase):
    """
    Base class for porosity creation/growth during discrete events.
    
    Called during:
    - Agglomeration: pore formation between colliding particles
    - Nucleation: pore structure in newly formed primary particles
    
    These kernels model how porosity emerges and evolves during
    particle-particle interactions.
    
    Physical mechanisms:
    - Incomplete coalescence (pores trapped between particles)
    - Surface roughness effects
    - Binder-induced pore formation
    - Crystallization porosity
    """
    
    @property
    def category(self) -> str:
        return 'porosity_growth'
    
    @abstractmethod
    def compute_merged_porosity(self,
                                 v_dry1: float, poro1: float,
                                 v_dry2: float, poro2: float,
                                 v_liq1: Optional[float] = None,
                                 v_liq2: Optional[float] = None,
                                 sat1: Optional[float] = None,
                                 sat2: Optional[float] = None,
                                 collision_energy: Optional[float] = None,
                                 solver: Optional[Any] = None
                                 ) -> tuple[float, float]:
        """
        Compute merged dry volume and porosity after agglomeration.
        
        Args:
            v_dry1: Dry volume of particle 1 [m³]
            poro1: Porosity of particle 1 (NaN for Vollkörper)
            v_dry2: Dry volume of particle 2 [m³]
            poro2: Porosity of particle 2 (NaN for Vollkörper)
            v_liq1: Liquid volume of particle 1 [m³] (optional)
            v_liq2: Liquid volume of particle 2 [m³] (optional)
            sat1: Saturation of particle 1 (optional)
            sat2: Saturation of particle 2 (optional)
            collision_energy: Estimated collision energy [J] (optional)
            solver: Reference to solver (optional)
        
        Returns:
            Tuple of (v_dry_merged, poro_merged)
            - v_dry_merged: Merged dry volume [m³]
            - poro_merged: Merged porosity (NaN if Vollkörper)
        """
        pass
    
    def compute_nucleation_porosity(self,
                                     v_solid: float,
                                     v_liquid: float,
                                     nucleation_params: Optional[Dict] = None,
                                     solver: Optional[Any] = None
                                     ) -> tuple[float, float]:
        """
        Compute porosity for newly nucleated particle.
        
        Override for porous nucleation models. Default returns Vollkörper.
        
        IMPORTANT: For volume_mixing kernel, this should return a default porosity
        (e.g., 0.4) to enable porosity formation from non-porous parents.
        Without this, volume_mixing would never create porosity from Vollkörper!
        
        Args:
            v_solid: Solid volume of nucleus [m³]
            v_liquid: Liquid volume associated with nucleus [m³]
            nucleation_params: Additional nucleation parameters (optional)
            solver: Reference to solver (optional)
        
        Returns:
            Tuple of (v_dry, porosity)
            - v_dry: Dry volume [m³]
            - porosity: Initial porosity (NaN for Vollkörper)
        """
        # Default: no porosity (Vollkörper)
        return float(v_solid), np.nan
    
    def compute_fragment_porosity(self,
                                   parent_porosity: float,
                                   fragment_volume: float,
                                   parent_volume: float,
                                   breakage_energy: Optional[float] = None,
                                   solver: Optional[Any] = None
                                   ) -> float:
        """
        Compute porosity for fragment after breakage.
        
        Override for models where breakage changes porosity (e.g., pore collapse).
        Default: fragments inherit parent porosity (equal distribution).
        
        Args:
            parent_porosity: Parent particle porosity (NaN for Vollkörper)
            fragment_volume: Fragment solid volume [m³]
            parent_volume: Parent solid volume [m³]
            breakage_energy: Estimated breakage energy [J] (optional)
            solver: Reference to solver (optional)
        
        Returns:
            fragment_porosity: Porosity of fragment (same as parent by default)
        """
        # Default: inherit parent porosity (fragments have same structure as parent)
        return parent_porosity


# =============================================================================
# Compression Kernels (time-based)
# =============================================================================

class CompressionKernel(KernelBase):
    """
    Base class for porosity reduction over time.
    
    Applied continuously by CompressionHandler after each MC event.
    Models mechanical compaction due to shear forces in the mixer.
    
    Physical mechanisms:
    - Exponential decay under constant shear
    - Stress-dependent compaction
    - Saturation-dependent pore closure
    - Viscoelastic consolidation
    """
    
    @property
    def category(self) -> str:
        return 'compression'
    
    @abstractmethod
    def compute(self,
                porosity: float,
                dt: float,
                v_particle: Optional[float] = None,
                local_stress: Optional[float] = None,
                saturation: Optional[float] = None,
                solver: Optional[Any] = None
                ) -> float:
        """
        Compute new porosity after compression over time dt.
        
        Args:
            porosity: Current porosity (0 to 1)
            dt: Time step [s]
            v_particle: Particle volume [m³] (optional, for size-dependent models)
            local_stress: Local stress estimate [Pa] (optional)
            saturation: Particle saturation (optional)
            solver: Reference to solver (optional)
        
        Returns:
            porosity_new: Reduced porosity (clamped to min_porosity)
            
        Note:
            Should return the input porosity unchanged if porosity is NaN
            (Vollkörper should not be compressed).
        """
        pass


# =============================================================================
# Liquid Internalization Kernels (time-based)
# =============================================================================

class LiquidInternalizationKernel(KernelBase):
    """
    Base class for capillary-driven liquid internalization over time.
    
    Applied continuously during simulation via operator splitting.
    Models the penetration of liquid binder into particle pores.
    
    Physical mechanisms:
    - Capillary-driven pore filling
    - Viscous flow resistance
    - Saturation-dependent uptake rate
    
    The rate equation follows Braumann et al. (2007):
        dl_intern/dt = k × l_ex × (v_pore - l_intern)
    
    where:
        - l_intern: Internal liquid volume [m³]
        - l_ex: External liquid volume [m³]
        - v_pore: Pore volume [m³]
        - k: Rate constant [1/(m³·s)]
    """
    
    @property
    def category(self) -> str:
        return 'liquid_internalization'
    
    @abstractmethod
    def compute(self,
                saturation: float,
                v_pore: float,
                l_total: float,
                dt: float,
                solver: Optional[Any] = None
                ) -> float:
        """
        Compute new saturation after liquid internalization over time dt.
        
        Args:
            saturation: Current saturation S = l_intern / v_pore (0 to 1)
            v_pore: Pore volume [m³]
            l_total: Total liquid volume associated with particle [m³]
                     (l_total = l_intern + l_ex, conserved during internalization)
            dt: Time step [s]
            solver: Reference to solver (optional)
        
        Returns:
            saturation_new: Updated saturation after internalization (0 to 1)
            
        Note:
            This method conserves total liquid (l_total). It only redistributes
            between internal and external phases.
        """
        pass


# =============================================================================
# Liquid Distribution Kernels
# =============================================================================

class LiquidDistributionKernel(KernelBase):
    """
    Base class for droplet/particle selection during nucleation.
    
    Determines which computational particles receive liquid droplets
    during the nucleation process. Different selection strategies
    model different physical mechanisms of droplet-particle collision.
    
    Selection strategies:
    - Uniform weighted (by W): All physical particles equally likely
    - Surface-weighted: Larger particles more likely
    - Saturation-preferential: Unsaturated particles preferred
    - Size-dependent: Based on collision cross-section
    """
    
    @property
    def category(self) -> str:
        return 'liquid_distribution'
    
    @abstractmethod
    def select_target_particle(self, solver: Any,
                                v_droplet: float,
                                current_time: Optional[float] = None
                                ) -> int:
        """
        Select computational particle for droplet distribution.
        
        Args:
            solver: Reference to solver instance
            v_droplet: Volume of droplet to distribute [m³]
            current_time: Current simulation time [s] (optional)
        
        Returns:
            particle_index: Index of selected particle (-1 if none valid)
            
        Note:
            The selection probability should reflect the physical mechanism
            of droplet-particle collision. For example:
            - Large particles have larger collision cross-sections
            - Wet particles may attract more droplets (capillary forces)
            - Dry particles may be preferred (wetting driving force)
        """
        pass
    
    def compute_droplet_volume(self, solver: Any,
                                current_time: Optional[float] = None
                                ) -> float:
        """
        Compute volume of droplet to distribute.
        
        Override for time-varying or state-dependent droplet sizes.
        Default returns the configured droplet_volume from nucleation config.
        
        Args:
            solver: Reference to solver instance
            current_time: Current simulation time [s] (optional)
        
        Returns:
            v_droplet: Droplet volume [m³]
        """
        # Default: use configured droplet volume
        if hasattr(solver, 'nucleation') and solver.nucleation is not None:
            return solver.nucleation.config.droplet_volume
        return 1e-12  # Fallback default


# =============================================================================
# Liquid Internalization Agglomeration Kernels (event-based)
# =============================================================================

class LiquidInternalizationAgglomerationKernel(KernelBase):
    """
    Base class for liquid internalization DURING agglomeration events.
    
    Called when two particles merge to compute how much external liquid
    becomes trapped in the newly formed contact pores.
    
    Physical mechanism (Braumann et al. 2007):
    When two particles with wetted surfaces contact, pores form at the
    contact point. The external liquid on previously wetted surfaces
    becomes trapped in these newly formed pores.
    
    Formula:
        l_{e→i} = sqrt(
            l_{e,j} × l_{e,k} ×
            [1 - √(1 - ((∛(v_j - l_{e,j})) / (∛v_j + ∛v_k))²)] ×
            [1 - √(1 - ((∛(v_k - l_{e,k})) / (∛v_j + ∛v_k))²)]
        )
    
    where v_j = v_dry,j + v_liq_ext,j (hydrodynamic volume of wetted particle)
    """
    
    @property
    def category(self) -> str:
        return 'liquid_internalization_agglomeration'
    
    @abstractmethod
    def compute_internalization(
        self,
        v_dry1: float, v_dry2: float,
        v_liq_ext1: float, v_liq_ext2: float,
        particle1_idx: Optional[int] = None,
        particle2_idx: Optional[int] = None,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute amount of liquid internalized during agglomeration.
        
        Args:
            v_dry1: Dry volume of particle 1 [m³]
            v_dry2: Dry volume of particle 2 [m³]
            v_liq_ext1: External liquid volume of particle 1 [m³]
            v_liq_ext2: External liquid volume of particle 2 [m³]
            particle1_idx: Index of particle 1 in solver arrays (optional)
            particle2_idx: Index of particle 2 in solver arrays (optional)
            solver: Reference to solver for accessing additional state (optional)
        
        Returns:
            l_e_to_i: Amount of liquid internalized [m³]
                      (0.0 if no internalization occurs)
        
        Note:
            This method computes internalization PER AGGLOMERATION EVENT.
            It is called during _do_one_agg() after partner selection.
            The returned value is added to internal liquid and subtracted
            from external liquid of the merged particle.
        """
        pass


# =============================================================================
# Agglomeration Acceptance Kernels
# =============================================================================

class AggAcceptanceKernel(KernelBase):
    """
    Base class for agglomeration acceptance/rejection criteria.
    
    Called AFTER collision partner selection to determine if
    particles actually agglomerate or bounce off.
    
    Physical models include:
    - Stokes criterion (Braumann et al. 2007)
    - Simple fit parameter (fittable)
    - Future: Liquid bridge rupture, viscoelastic rebound, etc.
    
    Usage:
        >>> kernel = get_agglomeration_acceptance_kernel('stokes_krit',
        ...     U_coll=1.0,
        ...     binder_viscosity=0.1,
        ...     rho_solid=2500.0,
        ...     rho_liquid=1000.0,
        ...     h_a=500e-9
        ... )
        >>> accepted = kernel.accept_collision(r1, r2, v_dry1, v_dry2, ...)
    """
    
    @property
    def category(self) -> str:
        return 'agglomeration_acceptance'
    
    @abstractmethod
    def accept_collision(self,
                         r1: float, r2: float,
                         v_dry1: float, v_dry2: float,
                         particle1_idx: Optional[int] = None,
                         particle2_idx: Optional[int] = None,
                         solver: Optional[Any] = None
                         ) -> bool:
        """
        Determine if collision results in agglomeration.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            v_dry1: Dry volume of particle 1 [m³]
            v_dry2: Dry volume of particle 2 [m³]
            particle1_idx: Index of particle 1 in solver arrays (optional)
            particle2_idx: Index of particle 2 in solver arrays (optional)
            solver: Reference to solver for accessing state arrays (optional)
        
        Returns:
            accepted: True if agglomeration proceeds, False if rejected
        
        Note:
            The solver reference allows kernels to access additional particle
            properties like liquid_volume, porosity, saturation, etc.
            Kernels should handle cases where these attributes don't exist
            (backward compatibility with dry simulations).
        """
        pass
