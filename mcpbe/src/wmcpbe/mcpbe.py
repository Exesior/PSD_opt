from __future__ import annotations

from .mcpbe_base import MCPBEBase
from .mcpbe_agg import MCPBEAgg
from .mcpbe_break import MCPBEBreak
from .mcpbe_post import MCPBEPost
from .mcpbe_nucleation import NucleationHandler, NucleationConfig
from .mcpbe_compression import CompressionHandler, CompressionConfig
from .mcpbe_continuous_processes import (
    ContinuousProcessesHandler,
    ContinuousProcessesConfig,
)
from .reconstruction_mixin import ReconstructionMixin


class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
    """
    Weighted DSMC Monte Carlo Population Balance Equation (PBE) solver.
    
    Architecture:
        - Core: MCPBEBase (particle state, control volume, main loop)
        - Physics mixins: MCPBEAgg (agglomeration), MCPBEBreak (breakage)
        - Post-processing: MCPBEPost (moments, PSD, statistics)
        - Reconstruction: ReconstructionMixin (CAM, RS, 2PM, QMX methods)
        - Handlers (composition): NucleationHandler, CompressionHandler
    
    Why handlers? Nucleation and compression operate on separate time scales
    and don't intercept the main Monte Carlo event loop directly.
    
    MRO (Method Resolution Order):
        Post → Break → Agg → Base → Reconstruction
    
    Key Concepts:
        Computational Weight (W):
            Each computational particle represents W physical particles.
            W = N_physical / N_computational
            High W = many physical particles represented by one computational particle.
            
        n_comp vs n_phys:
            n_comp: Number of computational particles (simulation efficiency)
            n_phys: Number of physical particles (real-world count)
            Example: 1000 physical particles simulated with 100 computational
                     particles → W = 10, n_comp = 100, n_phys = 1000
        
        droplet_comp vs droplet_phys:
            droplet_comp: Computational droplets per MC event (typically 1)
            droplet_phys: Physical droplets represented = droplet_comp × (Vc_ref/Vc) × W
            DSMC scaling ensures physical flow rate remains constant despite
            changing control volume Vc during simulation.
    """
    
    def create_nucleation_handler(self, **kwargs) -> NucleationHandler:
        """
        Create nucleation handler for liquid addition during simulation.
        
        Purpose:
            Distributes liquid droplets onto particles via weight-based sampling.
            All particles have equal collision probability (uniform by weight W).
        
        Duration specification (mutually exclusive):
            Option A: Direct duration via `liquid_addition_duration` [s]
            Option B: Calculated from target_wt_percent + solid_mass_in_mixer + densities
        
        Critical parameters:
            volumetric_flow_rate: Physical flow rate [m³/s]. Used directly without
                                  DSMC scaling. The sampler accounts for Vc changes.
            droplet_diameter: Physical droplet size [m]. Determines volume per event.
            liquid_addition_start: When to begin addition [s], default 0.0
        
        Understanding computational vs physical quantities:
            
            Computational Weight (W):
                W = N_physical / N_computational
                Each computational particle represents W physical particles.
                W varies during simulation as particles agglomerate/break.
            
            Droplet distribution (droplet_comp vs droplet_phys):
                - One MC event adds 1 computational droplet (droplet_comp = 1)
                - Physical droplets added = (Vc_ref / Vc) × W_selected
                - Vc_ref: Reference control volume (initial Vc)
                - Vc: Current control volume (changes during simulation)
                - W_selected: Weight of the selected particle
                
                Example: Vc_ref=1e-6 m³, Vc=2e-6 m³, W=50
                         → Physical droplets = (1e-6/2e-6) × 50 = 25 droplets
                
                Why this matters: DSMC maintains Sum(W)/Vc = constant.
                As Vc grows, fewer physical droplets are added per event,
                but the physical flow rate (m³/s) remains correct.
        
        Consistency check:
            If both duration options provided, must agree within 0.1% tolerance.
        
        Returns:
            NucleationHandler instance attached to solver.nucleation
        
        Raises:
            ValueError: If parameter combination is invalid
        
        Example (direct duration):
            >>> solver.create_nucleation_handler(
            ...     enabled=True,
            ...     volumetric_flow_rate=1e-9,  # 1 µL/s
            ...     droplet_diameter=1e-6,      # 1 µm droplets
            ...     liquid_addition_duration=300.0,  # 5 minutes
            ... )
        
        Example (calculate from mass):
            >>> solver.create_nucleation_handler(
            ...     enabled=True,
            ...     volumetric_flow_rate=1e-9,
            ...     droplet_diameter=1e-6,
            ...     target_wt_percent=5.0,       # 5 wt% liquid
            ...     solid_mass_in_mixer=0.1,     # 100g solid
            ...     rho_solid=2500.0,            # kg/m³
            ...     rho_liquid=1000.0,           # kg/m³
            ... )
            # Calculates: duration = (0.1 × 0.05 / 1000) / 1e-9 = 5 seconds
        
        Note:
            - Calling multiple times replaces existing handler (statistics lost)
            - See NucleationConfig for all parameters and validation rules
        """
        import warnings
        
        # Guard: Warn if handler already exists
        if hasattr(self, 'nucleation') and self.nucleation is not None:
            warnings.warn(
                "NucleationHandler already exists and will be replaced. "
                "This may lead to loss of statistics. Consider checking "
                "if nucleation is already configured before calling this method.",
                UserWarning,
                stacklevel=2
            )
        
        self.nucleation = NucleationHandler(self, NucleationConfig(**kwargs))
        
        return self.nucleation
    
    def create_compression_handler(self, **kwargs) -> CompressionHandler:
        """
        Create and attach a compression handler to this solver.
        
        Convenience method that creates a CompressionHandler for modeling
        porosity reduction due to shear forces in high-shear mixers.
        
        Args:
            **kwargs: Arguments passed to CompressionConfig
                     (enabled, rate, min_porosity)
        
        Returns:
            Created CompressionHandler instance
        
        Example:
            solver.create_compression_handler(
                enabled=True,
                rate=0.02,  # 1/s
                min_porosity=0.3,
            )
        
        Reference:
            Exponential porosity decay model based on:
            10.1103/PhysRevLett.64.2727
        
        Note:
            Calling this method multiple times will replace the existing handler.
            A warning is issued if a handler already exists.
            
            For validation against analytical solutions, use process_type="compression"
            instead of (or in addition to) the handler. The handler applies compression
            after each MC event, while process_type="compression" uses fixed time steps.
        """
        import warnings
        
        # Guard: Warn if handler already exists
        if hasattr(self, 'compression') and self.compression is not None:
            warnings.warn(
                "CompressionHandler already exists and will be replaced. "
                "This may lead to loss of statistics. Consider checking "
                "if compression is already configured before calling this method.",
                UserWarning,
                stacklevel=2
            )
        
        self.compression = CompressionHandler(self, CompressionConfig(**kwargs))
        return self.compression
    
    def create_continuous_processes_handler(self, **kwargs) -> ContinuousProcessesHandler:
        """
        Create and attach a continuous processes handler to this solver.
        
        Convenience method that creates a ContinuousProcessesHandler for modeling:
        1. Liquid Internalization: Capillary-driven penetration of liquid into pores
           (Braumann et al. 2007: r = k * l_ex * (v_pore - l_intern))
        2. Porosity Compression: Exponential porosity decay under shear
        
        Process order per time step:
        1. Update liquid distribution (internalization)
           - Conserve total liquid per particle
           - Update saturation: S = l_intern / v_pore
        2. Update porosity (compression)
           - Reduce pore volume
           - Externalize excess liquid if S > 1
        
        Args:
            **kwargs: Arguments passed to ContinuousProcessesConfig
                     (enabled, k_int, compression_enabled, compression_rate, min_porosity)
        
        Returns:
            Created ContinuousProcessesHandler instance
        
        Example:
            >>> solver.create_continuous_processes_handler(
            ...     enabled=True,
            ...     k_int=1e12,           # 1/(m³·s), internalization rate
            ...     compression_enabled=True,
            ...     compression_rate=0.02,  # 1/s
            ...     min_porosity=0.3,
            ... )
        
        Reference:
            Liquid Internalization: Braumann et al. (2007)
            "Modelling and validation of granulation with heterogeneous binder
            dispersion and chemical reaction"
            
            Porosity Compression: Exponential model based on
            10.1103/PhysRevLett.64.2727
        
        Note:
            Calling this method multiple times will replace the existing handler.
            A warning is issued if a handler already exists.
            
            For validation against analytical solutions, use process_type="compression"
            instead of (or in addition to) the handler. The handler applies processes
            after each MC event, while process_type="compression" uses fixed time steps.
        """
        import warnings
        
        # Guard: Warn if handler already exists
        if hasattr(self, 'continuous_processes') and self.continuous_processes is not None:
            warnings.warn(
                "ContinuousProcessesHandler already exists and will be replaced. "
                "This may lead to loss of statistics. Consider checking "
                "if continuous processes are already configured before calling this method.",
                UserWarning,
                stacklevel=2
            )
        
        self.continuous_processes = ContinuousProcessesHandler(
            self, ContinuousProcessesConfig(**kwargs)
        )
        return self.continuous_processes

