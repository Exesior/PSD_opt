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
           - l_intern: Internal liquid volume [m^3]
           - l_ex: External liquid volume [m^3]
           - v_pore: Pore volume [m^3]
           - k: Rate constant [1/(m^3·s)]
    
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
        k_int: Internalization rate constant [1/(m^3·s)]
               Typical: 1e10 - 1e14 depending on material
        compression_enabled: Whether porosity compression is active
        compression_rate: Rate constant [1/s]. Typical: 0.01-0.1
        min_porosity: Minimum asymptotic porosity [0, 1). Typical: 0.25-0.4
    """
    enabled: bool = False
    k_int: float = 1e12  # 1/(m^3·s)
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

        Vectorised over the active particle slice: this runs once per
        Monte-Carlo event, so a Python loop here costs O(n) interpreter
        overhead per event and used to dominate granulation runs.

        Args:
            dt: Time step [s]
        """
        solver = self.solver
        a_tot = solver.a_tot
        if a_tot < 1:
            return

        porosity = solver.porosity[:a_tot]
        v_dry = solver.V_flat[-1, :a_tot]
        v_pore = v_dry * porosity  # NaN porosity propagates to NaN pore volume

        # Vollkoerper (NaN porosity) and degenerate pores take no liquid.
        mask = ~np.isnan(porosity) & (porosity > 0) & (v_pore > 0) & np.isfinite(v_pore)
        n_active = int(np.count_nonzero(mask))
        if n_active == 0:
            return

        saturation = solver.saturation[:a_tot]
        l_total = solver.liquid_volume[:a_tot]

        sat_sel = saturation[mask]
        pore_sel = v_pore[mask]
        liq_sel = l_total[mask]

        if self.internalization_kernel is not None:
            saturation_new = self.internalization_kernel.compute_array(
                saturation=sat_sel, v_pore=pore_sel, l_total=liq_sel, dt=dt
            )
        else:
            saturation_new = self._internalization_fallback_array(
                sat_sel, pore_sel, liq_sel, dt
            )

        saturation[mask] = saturation_new
        self._particles_internalized_total += n_active
    
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

    def _internalization_fallback_array(
        self,
        saturation: np.ndarray,
        v_pore: np.ndarray,
        l_total: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Vectorised counterpart of :meth:`_internalization_fallback`.

        Explicit Euler step of the Braumann equation
        ``dl_intern/dt = k_int * l_ex * (v_pore - l_intern)``.
        """
        k_int = self.config.k_int
        sat = np.clip(saturation, 0.0, 1.0)

        if k_int <= 0:
            return np.where(l_total > 0, sat, saturation)

        # k_int <= 0, v_pore <= 0 or l_total <= 0 -> saturation unchanged.
        # (v_pore > 0 is guaranteed by the caller's mask.)
        unchanged = l_total <= 0
        out = np.where(unchanged, saturation, sat)

        active = ~unchanged & (sat < 1.0)
        if not np.any(active):
            return np.where(unchanged, saturation, np.minimum(sat, 1.0))

        vp = v_pore[active]
        lt = l_total[active]
        l_intern = sat[active] * vp
        l_ex = lt - l_intern
        v_empty = vp - l_intern

        l_new = l_intern + (k_int * l_ex * v_empty) * dt
        l_new = np.maximum(0.0, np.minimum(np.minimum(l_new, lt), vp))

        out[active] = l_new / vp
        return out

    def _apply_compression(self, dt: float) -> None:
        """
        Apply porosity compression to all particles.
        
        Reduces porosity according to compression model.
        Updates V_dry to conserve V_solid.
        Externalizes excess liquid if saturation exceeds 1.0.
        
        Args:
            dt: Time step [s]
        """
        solver = self.solver
        a_tot = solver.a_tot
        if a_tot < 1:
            return

        min_poro = self.config.min_porosity
        # Views onto the active slice; writes go straight into solver state.
        poro_view = solver.porosity[:a_tot]
        v_dry_view = solver.V_flat[-1, :a_tot]
        sat_view = solver.saturation[:a_tot]
        poro_old = poro_view.copy()
        
        # Vollkoerper (NaN) cannot be compressed, and neither can particles that
        # already sit at or below the asymptotic minimum porosity.
        active = ~np.isnan(poro_old) & (poro_old > min_poro)

        # === DEBUG COMP: VOR KOMPRESSION ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_comp', True):
            v_dry = solver.V_flat[-1, :solver.a_tot]
            poro = solver.porosity[:solver.a_tot]
            w = solver.W[:solver.a_tot]
            valid = ~np.isnan(poro)
            v_solid = np.zeros_like(v_dry)
            v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
            v_solid[~valid] = v_dry[~valid]
            
            solid_before = np.sum(v_solid * w)
            print(f"\n[DEBUG COMP] t={getattr(solver, '_elapsed', 0.0):.4f}s, dt={dt:.4f}s")
            print(f"  Active particles: {np.sum(active)}")
            print(f"  poro_range=[{np.nanmin(poro):.4f}, {np.nanmax(poro):.4f}]")
            print(f"  BEFORE: V_solid_total={solid_before:.6e} (MUSST KONSTANT BLEIBEN!)")
# ================================
        if not np.any(active):
            return

        # ------------------------------------------------------------------
        # Step 1: new porosity
        # ------------------------------------------------------------------
        if self.compression_kernel is not None:
            poro_new = self.compression_kernel.compute_array(poro_old, dt)
        else:
            # Fallback: analytical exponential decay towards min_porosity.
            rate = self.config.compression_rate
            poro_new = min_poro + (poro_old - min_poro) * np.exp(-rate * dt)

        poro_view[active] = poro_new[active]

        # ------------------------------------------------------------------
        # Step 2: conserve V_solid, shrink V_pore
        # ------------------------------------------------------------------
        V_dry_old = v_dry_view
        one_minus_new = 1.0 - poro_new

        # Guards mirroring the original per-particle `continue` statements: the
        # porosity update above still applies to these particles, only the
        # volume/saturation bookkeeping is skipped.
        reached = (
            active
            & (V_dry_old > 0)
            & np.isfinite(V_dry_old)
            & (one_minus_new >= 1e-10)
        )
        vol_ok = reached & ((1.0 - poro_old) > 1e-10)

        if np.any(vol_ok):
            V_solid = V_dry_old * (1.0 - poro_old)
            V_pore_old = V_dry_old * poro_old

            V_pore_new = np.zeros_like(V_solid)
            np.divide(V_solid * poro_new, one_minus_new, out=V_pore_new, where=vol_ok)

            V_dry_new = V_solid + V_pore_new
            write_dry = vol_ok & (V_dry_new > 0) & np.isfinite(V_dry_new)
            v_dry_view[write_dry] = V_dry_new[write_dry]

            # Step 3: saturation rises as pores shrink; excess liquid becomes
            # external. `liquid_volume` (total per particle) stays untouched, so
            # total liquid is conserved by construction.
            sat_mask = vol_ok & (V_pore_new > 0)
            if np.any(sat_mask):
                V_liq_int_old = np.minimum(
                    solver.liquid_volume[:a_tot], V_pore_old * sat_view
                )
                new_sat = np.zeros_like(V_pore_new)
                np.divide(V_liq_int_old, V_pore_new, out=new_sat, where=sat_mask)
                np.minimum(new_sat, 1.0, out=new_sat)
                sat_view[sat_mask] = new_sat[sat_mask]

        # === DEBUG COMP: NACH KOMPRESSION ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_comp', True):
            v_dry_after = solver.V_flat[-1, :solver.a_tot]
            poro_after = solver.porosity[:solver.a_tot]
            w_after = solver.W[:solver.a_tot]
            valid_after = ~np.isnan(poro_after)
            v_solid_after = np.zeros_like(v_dry_after)
            v_solid_after[valid_after] = v_dry_after[valid_after] * (1.0 - poro_after[valid_after])
            v_solid_after[~valid_after] = v_dry_after[~valid_after]
            
            solid_after = np.sum(v_solid_after * w_after)
            liq_after = np.sum(solver.liquid_volume[:solver.a_tot] * w_after)
            
            print(f"  AFTER:  V_solid_total={solid_after:.6e}, V_liq_total={liq_after:.6e}")
            print(f"  ΔV_solid={solid_after - solid_before:.6e} (SOLLTE = 0 sein!)")
            print(f"  poro_range=[{np.nanmin(poro_after):.4f}, {np.nanmax(poro_after):.4f}]")
# ================================

        self._particles_compressed_total += int(np.count_nonzero(reached))
    
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
