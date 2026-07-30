"""
Continuous Processes Handler for WMCPBE Solver.

Handles time-continuous physical processes via operator splitting:
1. Liquid Internalization: Capillary-driven penetration of liquid into pores
2. Porosity Compression: Exponential porosity decay under shear

Applied after each Monte Carlo event using fixed or adaptive time steps.

Physical Models:
    1. Liquid Internalization (Braumann et al. 2007):
       dl_intern/dt = k × l_ex × (v_pore - l_intern)
       
       where:
           - l_intern: Internal liquid volume [m³]
           - l_ex: External liquid volume [m³]
           - v_pore: Pore volume [m³]
           - k: Rate constant [1/(m³·s)]
    
    2. Porosity Compression:
       dε/dt = -rate × (ε - ε_min)
       
       Solution: ε(t) = ε_min + (ε_0 - ε_min) × exp(-rate × t)

Process Order (per time step):
    1. Update liquid distribution (internalization)
       - Conserve total liquid per particle
       - Update saturation: S = l_intern / v_pore
    
    2. Update porosity (compression)
       - Reduce pore volume
       - Externalize excess liquid if S > 1

Example:
    >>> config = ContinuousProcessesConfig(
    ...     enabled=True,
    ...     k_int=1e12,        # Internalization rate
    ...     compression_rate=0.02,
    ...     min_porosity=0.3,
    ... )
    >>> solver.create_continuous_processes_handler(**config.__dict__)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ContinuousProcessesConfig:
    """
    Configuration for continuous processes handler.
    
    Attributes:
        enabled: Whether continuous processes are active
        k_int: Internalization rate constant [1/(m³·s)]
               Typical: 1e10 - 1e14 depending on material
        compression_enabled: Whether porosity compression is active
        compression_rate: Rate constant [1/s]. Typical: 0.01-0.1
        min_porosity: Minimum asymptotic porosity [0, 1). Typical: 0.25-0.4
    """
    enabled: bool = False
    k_int: float = 1e12  # 1/(m³·s)
    compression_enabled: bool = True
    compression_rate: float = 0.02  # 1/s
    min_porosity: float = 0.3
    
    def __post_init__(self) -> None:
        if self.enabled:
            if self.k_int < 0:
                raise ValueError("k_int must be non-negative")
            if self.compression_enabled:
                if self.compression_rate <= 0.0:
                    raise ValueError("compression_rate must be positive when enabled")
                if not (0.0 <= self.min_porosity < 1.0):
                    raise ValueError("min_porosity must be in [0, 1)")


class ContinuousProcessesHandler:
    """
    Applies continuous processes after MC events via operator splitting.
    
    Process sequence per time step dt:
    1. Liquid Internalization: Update saturation based on capillary uptake
    2. Porosity Compression: Reduce porosity, externalize excess liquid
    
    Kernels are pluggable for future extensions.
    """
    
    def __init__(self, solver, config: ContinuousProcessesConfig):
        """Initialize with parent solver and config."""
        self.solver = solver
        self.config = config
        
        # Internal state
        self._current_time = 0.0
        self._fixed_dt = 0.1  # Fixed dt for validation mode
        
        # Statistics
        self._total_steps = 0
        self._particles_internalized_total = 0
        self._particles_compressed_total = 0
        
        # Initialize kernels from solver's kernel manager
        self.internalization_kernel = None
        self.compression_kernel = None
        
        if hasattr(solver, 'kernel_manager'):
            km = solver.kernel_manager
            self.internalization_kernel = km.liquid_internalization_kernel
            self.compression_kernel = km.porosity_compression_kernel
    
    @property
    def fixed_dt(self) -> float:
        """Fixed time step for validation mode."""
        return self._fixed_dt
    
    @fixed_dt.setter
    def fixed_dt(self, value: float) -> None:
        """Set fixed time step for validation mode."""
        if value <= 0.0:
            raise ValueError("fixed_dt must be positive")
        self._fixed_dt = value
    
    def step(self, current_time: float, dt_event: float) -> None:
        """
        Apply continuous processes after MC event.
        
        Uses dt_event for agglomeration/breakage modes,
        fixed_dt for validation-only mode.
        
        Args:
            current_time: Current simulation time [s]
            dt_event: Time elapsed since last MC event [s]
        """
        if not self.config.enabled:
            return
        
        self._current_time = current_time
        
        # Determine time step
        process_type = getattr(self.solver, 'process_type', 'agglomeration')
        if process_type == 'compression':
            # Use fixed dt for validation against analytical solution
            dt = self._fixed_dt
        else:
            # Use actual elapsed time from MC events
            dt = dt_event
        
        if dt <= 0.0:
            return
        
        # Apply processes in sequence
        self._apply_internalization(dt)
        
        if self.config.compression_enabled:
            self._apply_compression(dt)
        
        # Update statistics
        self._total_steps += 1
    
    def _apply_internalization(self, dt: float) -> None:
        """
        Apply liquid internalization to all particles.
        
        Updates saturation based on capillary-driven uptake.
        Total liquid per particle is conserved.
        
        Args:
            dt: Time step [s]
        """
        a_tot = self.solver.a_tot
        if a_tot < 1:
            return
        
        particles_internalized = 0
        
        # Iterate over active particles
        for i in range(a_tot):
            porosity = self.solver.porosity[i]
            
            # Skip Vollkörper (no pores → no internalization)
            if np.isnan(porosity) or porosity <= 0:
                continue
            
            # Get current state
            saturation = self.solver.saturation[i]
            v_dry = self.solver.V_flat[-1, i]
            l_total = self.solver.liquid_volume[i]
            
            # Calculate pore volume: v_pore = v_dry × porosity
            v_pore = v_dry * porosity
            
            # Skip if pore volume invalid
            if v_pore <= 0 or not np.isfinite(v_pore):
                continue
            
            # Compute new saturation using kernel
            if self.internalization_kernel is not None:
                saturation_new = self.internalization_kernel.compute(
                    saturation=saturation,
                    v_pore=v_pore,
                    l_total=l_total,
                    dt=dt
                )
            else:
                # Fallback: use config parameter directly
                saturation_new = self._internalization_fallback(
                    saturation, v_pore, l_total, dt
                )
            
            # Update saturation
            self.solver.saturation[i] = saturation_new
            particles_internalized += 1
        
        self._particles_internalized_total += particles_internalized
    
    def _internalization_fallback(
        self,
        saturation: float,
        v_pore: float,
        l_total: float,
        dt: float
    ) -> float:
        """
        Fallback internalization computation without kernel.
        
        Uses Euler integration of Braumann equation:
            dl_intern/dt = k_int × l_ex × (v_pore - l_intern)
        """
        k_int = self.config.k_int
        
        # Edge cases
        if k_int <= 0 or v_pore <= 0 or l_total <= 0:
            return saturation
        
        saturation = max(0.0, min(1.0, saturation))
        
        if saturation >= 1.0:
            return 1.0
        
        # Current state
        l_intern = saturation * v_pore
        l_ex = l_total - l_intern
        v_empty = v_pore - l_intern
        
        # Rate equation
        rate = k_int * l_ex * v_empty
        dl = rate * dt
        
        # Update
        l_intern_new = l_intern + dl
        l_intern_new = max(0.0, min(l_intern_new, l_total, v_pore))
        
        return l_intern_new / v_pore
    
    def _apply_compression(self, dt: float) -> None:
        """
        Apply porosity compression to all particles.
        
        Reduces porosity according to compression model.
        Updates V_dry to conserve V_solid.
        Externalizes excess liquid if saturation exceeds 1.0.
        
        Args:
            dt: Time step [s]
        """
        a_tot = self.solver.a_tot
        if a_tot < 1:
            return
        
        min_poro = self.config.min_porosity
        particles_compressed = 0
        
        # Iterate over active particles
        for i in range(a_tot):
            current_poro = self.solver.porosity[i]
            
            # Skip Vollkörper (NaN porosity)
            if np.isnan(current_poro):
                continue
            
            # Only compress if above minimum
            if current_poro > min_poro:
                # Compute new porosity using kernel
                if self.compression_kernel is not None:
                    new_poro = self.compression_kernel.compute(
                        porosity=current_poro,
                        dt=dt
                    )
                else:
                    # Fallback: exponential decay
                    rate = self.config.compression_rate
                    new_poro = min_poro + (current_poro - min_poro) * np.exp(-rate * dt)
                
                self.solver.porosity[i] = new_poro
                
                # ==========================================
                # Compression: Conserve V_solid, reduce V_pore
                # ==========================================
                V_dry_old = self.solver.V_flat[-1, i]
                
                # SAFETY: Skip if V_dry is invalid
                if V_dry_old <= 0 or not np.isfinite(V_dry_old):
                    continue
                
                # Ensure (1 - new_poro) is not too close to zero
                one_minus_poro = 1.0 - new_poro
                if one_minus_poro < 1e-10:
                    continue
                
                if (1.0 - current_poro) > 1e-10:
                    # Step 1: Calculate V_solid (CONSERVED)
                    V_solid = V_dry_old * (1.0 - current_poro)
                    
                    # Step 2: Calculate old and new pore volumes
                    V_pore_old = V_dry_old * current_poro
                    
                    if one_minus_poro > 1e-10:
                        V_pore_new = V_solid * new_poro / one_minus_poro
                    else:
                        V_pore_new = 0.0
                    
                    # Step 3: Calculate new V_dry
                    V_dry_new = V_solid + V_pore_new
                    
                    if V_dry_new > 0 and np.isfinite(V_dry_new):
                        # Step 4: Update V_flat[-1] (V_dry changes)
                        self.solver.V_flat[-1, i] = V_dry_new
                    
                    # Step 5: Handle saturation increase and liquid externalization
                    if hasattr(self.solver, 'saturation') and hasattr(self.solver, 'liquid_volume'):
                        current_sat = self.solver.saturation[i]
                        
                        # Current internal liquid
                        V_liq_total = self.solver.liquid_volume[i]
                        V_liq_int_old = min(V_liq_total, V_pore_old * current_sat)
                        
                        # New saturation (internal liquid / new pore volume)
                        if V_pore_new > 0:
                            new_sat = V_liq_int_old / V_pore_new
                            
                            # If saturation exceeds 1, externalize the excess
                            if new_sat > 1.0:
                                new_sat = 1.0
                                # Excess liquid becomes external
                                # V_liq_total unchanged (intensive property per particle)
                            
                            self.solver.saturation[i] = new_sat
                
                particles_compressed += 1
        
        self._particles_compressed_total += particles_compressed
    
    def get_statistics(self) -> dict:
        """Return continuous processes statistics dict."""
        return {
            'current_time': self._current_time,
            'enabled': self.config.enabled,
            'k_int': self.config.k_int,
            'compression_enabled': self.config.compression_enabled,
            'compression_rate': self.config.compression_rate,
            'min_porosity': self.config.min_porosity,
            'total_steps': self._total_steps,
            'particles_internalized_total': self._particles_internalized_total,
            'particles_compressed_total': self._particles_compressed_total,
        }
    
    def reset(self) -> None:
        """Reset handler state (for repeated simulations)."""
        self._current_time = 0.0
        self._total_steps = 0
        self._particles_internalized_total = 0
        self._particles_compressed_total = 0


def create_continuous_processes_handler(
    solver,
    **kwargs
) -> ContinuousProcessesHandler:
    """
    Factory function for ContinuousProcessesHandler.
    
    Args:
        solver: Parent solver
        **kwargs: ContinuousProcessesConfig args
    
    Returns:
        Configured ContinuousProcessesHandler
    """
    config = ContinuousProcessesConfig(**kwargs)
    handler = ContinuousProcessesHandler(solver, config)
    return handler
