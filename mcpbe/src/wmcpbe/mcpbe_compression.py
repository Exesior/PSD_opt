"""
Compression Handler for WMCPBE Solver.

Handles porosity reduction under shear using exponential decay model.
Applied after each Monte Carlo event to eligible particles.

Model:
    poro_new = min_poro + (poro_old - min_poro) * exp(-rate * dt)

Reference:
    Iveson & Litster (1998), PhysRevLett.64.2727

Example:
    >>> config = CompressionConfig(enabled=True, rate=0.02, min_porosity=0.3)
    >>> solver.create_compression_handler(**config.__dict__)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompressionConfig:
    """
    Porosity compression configuration.
    
    Attributes:
        enabled: Whether compression is active
        rate: Rate constant [1/s]. Typical: 0.01-0.1 for high-shear mixers.
              Fit from experimental data.
        min_porosity: Minimum asymptotic porosity. Particles won't compress below.
    """
    enabled: bool = False
    rate: float = 0.02  # 1/s
    min_porosity: float = 0.3
    
    def __post_init__(self) -> None:
        if self.enabled:
            if self.rate <= 0.0:
                raise ValueError("rate must be positive when enabled")
            if not (0.0 <= self.min_porosity < 1.0):
                raise ValueError("min_porosity must be in [0, 1)")
            if self.min_porosity >= 0.4:
                # Warning: typical nucleation porosity is 0.4
                pass  # Allow but user should be aware


class CompressionHandler:
    """
    Applies exponential porosity decay after MC events.
    
    Model: poro_new = min_poro + (poro_old - min_poro) * exp(-rate * dt)
    
    Kernel is pluggable for future extensions (size/saturation/shear dependence).
    """
    
    def __init__(self, solver, config: CompressionConfig):
        """Initialize with parent solver and config."""
        self.solver = solver
        self.config = config
        
        # Internal state
        self._current_time = 0.0
        self._fixed_dt = 0.1  # Fixed dt for "compression" process type
        
        # Statistics
        self._total_compression_steps = 0
        self._particles_compressed_total = 0
    
    @property
    def fixed_dt(self) -> float:
        """Fixed time step for compression-only mode."""
        return self._fixed_dt
    
    @fixed_dt.setter
    def fixed_dt(self, value: float) -> None:
        """Set fixed time step for compression-only mode."""
        if value <= 0.0:
            raise ValueError("fixed_dt must be positive")
        self._fixed_dt = value
    
    def step(self, current_time: float, dt_event: float) -> None:
        """
        Apply compression after MC event.
        
        Uses dt_event for agglomeration/breakage modes,
        fixed_dt for compression-only mode.
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
        
        # Apply compression to all eligible particles
        self._apply_compression(dt)
        
        # Update statistics
        self._total_compression_steps += 1
    
    def _apply_compression(self, dt: float) -> None:
        """
        Compress particles with porosity > min_porosity.
        
        Skips Vollkörper (NaN porosity). Updates V_particle_dry to conserve V_solid.
        Externalizes excess liquid when saturation exceeds 1.0.
        """
        a_tot = self.solver.a_tot
        if a_tot < 1:
            return
        
        min_poro = self.config.min_porosity
        particles_compressed = 0
        
        # Iterate over active particles
        for i in range(a_tot):
            current_poro = self.solver.porosity[i]
            
            # Skip particles without assigned porosity (NaN = Vollkörper)
            if np.isnan(current_poro):
                continue
            
            # Only compress if above minimum
            if current_poro > min_poro:
                new_poro = self.compression_kernel(i, dt)
                self.solver.porosity[i] = new_poro
                
                # ==========================================
                # Compression: Conserve V_solid, reduce V_pore
                # ==========================================
                # Physical model:
                #   V_solid = constant (mass conservation!)
                #   V_pore decreases under shear
                #   V_dry = V_solid + V_pore
                #
                # Relations:
                #   porosity = V_pore / V_dry
                #   V_solid = V_dry × (1 - porosity)
                #   V_pore = V_solid × porosity / (1 - porosity)
                #
                # After compression with new_porosity:
                #   V_solid stays constant
                #   V_pore_new = V_solid × poro_new / (1 - poro_new)
                #   V_dry_new = V_solid + V_pore_new = V_solid / (1 - poro_new)
                
                V_dry_old = self.solver.V_flat[-1, i]
                
                # SAFETY: Skip if V_dry is invalid or too small
                if V_dry_old <= 0 or not np.isfinite(V_dry_old):
                    continue
                
                # SAFETY: Ensure (1 - new_poro) is not too close to zero
                one_minus_poro = 1.0 - new_poro
                if one_minus_poro < 1e-10:
                    # Porosity too close to 1.0 - skip compression
                    continue
                
                if (1.0 - current_poro) > 1e-10:
                    # Step 1: Calculate V_solid (CONSERVED - mass conservation!)
                    V_solid = V_dry_old * (1.0 - current_poro)
                    
                    # Step 2: Calculate old and new pore volumes
                    V_pore_old = V_dry_old * current_poro
                    
                    # SAFETY: Avoid division by zero for very low porosity
                    if one_minus_poro > 1e-10:
                        V_pore_new = V_solid * new_poro / one_minus_poro
                    else:
                        V_pore_new = 0.0
                    
                    # Step 3: Calculate new V_dry
                    V_dry_new = V_solid + V_pore_new
                    
                    # SAFETY: Ensure V_dry_new is valid
                    if V_dry_new > 0 and np.isfinite(V_dry_new):
                        # Step 4: Update V_flat
                        # ================================================================
                        # V_flat[-1] = V_dry (changes due to pore collapse)
                        # V_flat[:dim] = V_solid (CONSTANT - do NOT modify!)
                        # ================================================================
                        self.solver.V_flat[-1, i] = V_dry_new
                        # EXPLICITLY DO NOT touch V_flat[:dim] - it represents V_solid!
                        # self.solver.V_flat[:self.solver.dim, i] *= scale_factor  ← WRONG!
                        # Mass is conserved because V_solid remains constant
                    
                    # Step 5: Handle saturation increase and liquid externalization
                    if hasattr(self.solver, 'saturation') and hasattr(self.solver, 'liquid_volume'):
                        current_sat = self.solver.saturation[i]
                        
                        # Current internal liquid: V_liq_int = V_pore × S
                        V_liq_total = self.solver.liquid_volume[i]
                        V_liq_int_old = min(V_liq_total, V_pore_old * current_sat)
                        
                        # New saturation (internal liquid / new pore volume)
                        # As pores shrink, saturation increases
                        if V_pore_new > 0:
                            new_sat = V_liq_int_old / V_pore_new
                            
                            # If saturation exceeds 1, externalize the excess
                            if new_sat > 1.0:
                                new_sat = 1.0
                                # Excess liquid becomes external
                                # V_liq_total stays constant (intensive property)
                            
                            self.solver.saturation[i] = new_sat
                            # Total liquid_volume[i] unchanged (intensive per particle)
                
                particles_compressed += 1
        
        self._particles_compressed_total += particles_compressed
    
    def compression_kernel(self, particle_idx: int, dt: float) -> float:
        """
        Default exponential decay kernel: poro_new = min_poro + (poro_old - min_poro) * exp(-rate * dt).
        
        Overrideable for complex models (size/saturation/shear dependence).
        
        DEPRECATED: Use ContinuousProcessesHandler with kernels instead.
        
        Args:
            particle_idx: Particle index
            dt: Time step [s]
        
        Returns:
            New porosity clamped to [min_porosity, 1.0]
        """
        import warnings
        warnings.warn(
            "CompressionHandler.compression_kernel is deprecated. "
            "Use ContinuousProcessesHandler with kernel manager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        current_poro = self.solver.porosity[particle_idx]
        min_poro = self.config.min_porosity
        rate = self.config.rate
        
        # Exponential decay toward minimum porosity
        # poro(t) = min_poro + (poro_0 - min_poro) * exp(-rate * t)
        new_poro = min_poro + (current_poro - min_poro) * np.exp(-rate * dt)
        
        # Ensure bounds (numerical safety)
        new_poro = max(min_poro, min(1.0, new_poro))
        
        return new_poro
    
    def get_statistics(self) -> dict:
        """Return compression statistics dict."""
        return {
            'current_time': self._current_time,
            'enabled': self.config.enabled,
            'rate': self.config.rate,
            'min_porosity': self.config.min_porosity,
            'total_compression_steps': self._total_compression_steps,
            'particles_compressed_total': self._particles_compressed_total,
        }
    
    def reset(self) -> None:
        """Reset compression state (for repeated simulations)."""
        self._current_time = 0.0
        self._total_compression_steps = 0
        self._particles_compressed_total = 0


def create_compression_handler(solver, **kwargs) -> CompressionHandler:
    """
    Factory function for CompressionHandler.
    
    Args:
        solver: Parent solver
        **kwargs: CompressionConfig args (enabled, rate, min_porosity)
    
    Returns:
        Configured CompressionHandler
    """
    config = CompressionConfig(**kwargs)
    handler = CompressionHandler(solver, config)
    return handler
