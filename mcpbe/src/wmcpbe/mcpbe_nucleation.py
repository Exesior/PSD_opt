"""
Nucleation Handler for WMCPBE Solver.

Handles liquid addition via droplet distribution during Monte Carlo simulation.
Droplets are distributed onto particles using weight-based uniform sampling.

Physical Model:
    Droplets collide randomly with particles. All particles have equal 
    probability of being hit (uniform sampling weighted by computational weight W).

Architecture:
    - NucleationConfig: Configuration dataclass
    - NucleationHandler: Main handler class (composition pattern)

Example:
    >>> config = NucleationConfig(
    ...     enabled=True,
    ...     volumetric_flow_rate=1e-9,  # m³/s
    ...     droplet_diameter=1e-6,  # m
    ...     liquid_addition_start=0.0,  # s
    ...     liquid_addition_duration=5.0,  # s
    ... )
    >>> solver.create_nucleation_handler(**config.__dict__)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .fenwick_new import FenwickSampler


class InitializationType(Enum):
    """
    Deprecated: Kept for backward compatibility only.
    
    Modern nucleation uses unified time-based liquid addition. Duration is
    either specified directly or calculated from mass parameters.
    
    Members:
        DURATION: Legacy - use liquid_addition_duration directly
        MASS: Legacy - use target_wt_percent + solid_mass_in_mixer
    """
    DURATION = "duration"
    MASS = "mass"


@dataclass
class NucleationConfig:
    """
    Configuration for time-based liquid addition via droplet distribution.
    
    Physical Model:
        Droplets collide randomly with particles during simulation.
        All particles have equal collision probability (uniform sampling by weight W).
        Each computational droplet event represents multiple physical droplets
        via DSMC scaling: physical_droplets = (Vc_ref/Vc) × W_selected.
    
    Duration specification (choose one):
        Option A: Direct duration via `liquid_addition_duration` [s]
        Option B: Calculated from target_wt_percent + solid_mass_in_mixer + densities
                  Formula: duration = (solid_mass × wt%/100 / rho_liquid) / flow_rate
    
    Consistency Check:
        If both options provided, they must agree within 0.1% tolerance.
    
    Parameters - Required:
        enabled: Activate nucleation
        volumetric_flow_rate: Physical flow rate [m³/s]. Used directly without
                              DSMC scaling. The weight-based sampler accounts
                              for control volume changes automatically.
        droplet_diameter: Physical droplet size [m]
    
    Parameters - Duration (choose one method):
        liquid_addition_duration: Direct duration [s]
        OR (all required for mass-based calculation):
            target_wt_percent: Target liquid/solid weight percent (dry basis) [%]
            solid_mass_in_mixer: Total solid mass in mixer [kg]
            rho_solid: Solid density [kg/m³]
            rho_liquid: Liquid density [kg/m³]
    
    Parameters - Optional:
        liquid_addition_start: Start time [s], default 0.0
        batch_size: Computational weight per droplet distribution event [dimensionless].
                    Default: 25.0. dW = min(batch_size, W_i). Allows fine-grained control
                    over how many physical particles receive a droplet per event.
        consistency_tol: Relative tolerance for consistency check, default 0.001 (0.1%)
    
    Understanding Computational vs Physical Quantities:
        
        Computational Weight (W):
            Definition: W = N_physical / N_computational
            Meaning: Each computational particle represents W physical particles.
            During simulation: W changes as particles agglomerate or break.
            
            Example: 10,000 physical particles simulated with 500 computational
                     particles → Average W = 20
            
            Why it matters for nucleation:
                When a droplet hits a particle with W=50, it represents
                50 physical particles being hit simultaneously.
        
        n_comp vs n_phys:
            n_comp: Number of computational particles in simulation
            n_phys: Number of real physical particles in the system
            
            Relationship: n_phys = Σ(W_i) for all computational particles i
            
            Example: 3 computational particles with W=[10, 20, 30]
                     → n_comp = 3
                     → n_phys = 10 + 20 + 30 = 60
        
        droplet_comp vs droplet_phys:
            droplet_comp: Computational droplets per MC event (always 1)
            droplet_phys: Actual physical droplets added
            
            Formula: droplet_phys = (Vc_ref / Vc) × W_selected
            
            Where:
                Vc_ref: Reference control volume (initial volume at t=0)
                Vc: Current control volume (changes during simulation)
                W_selected: Weight of the particle that was hit
            
            Example scenario:
                Initial state: Vc_ref = 1.0e-6 m³, 500 particles, avg W = 20
                Later state: Vc = 2.0e-6 m³ (doubled due to agglomeration)
                
                One MC event hits particle with W = 25:
                droplet_phys = (1.0e-6 / 2.0e-6) × 25 = 12.5 physical droplets
                
                Why DSMC scaling matters:
                - As Vc grows, fewer physical droplets added per event
                - But physical flow rate (m³/s) stays correct!
                - Sum(W)/Vc remains constant = initial particle density
    
    Deprecated (backward compatibility only):
        initialization_type: Legacy enum, don't use
        n_comp: Legacy parameter, don't use
    
    Example (direct duration):
        >>> config = NucleationConfig(
        ...     enabled=True,
        ...     volumetric_flow_rate=1e-9,      # 1 µL/s
        ...     droplet_diameter=1e-6,          # 1 µm droplets
        ...     liquid_addition_duration=300.0, # Add for 5 minutes
        ... )
    
    Example (calculate from mass):
        >>> config = NucleationConfig(
        ...     enabled=True,
        ...     volumetric_flow_rate=1e-9,      # 1 µL/s
        ...     droplet_diameter=1e-6,          # 1 µm droplets
        ...     target_wt_percent=5.0,          # 5 wt% liquid (dry basis)
        ...     solid_mass_in_mixer=0.1,        # 100g solid material
        ...     rho_solid=2500.0,               # kg/m³ (typical powder)
        ...     rho_liquid=1000.0,              # kg/m³ (water)
        ... )
        # Automatically calculates:
        #   mass_liquid = 0.1 kg × 0.05 = 0.005 kg
        #   volume_liquid = 0.005 kg / 1000 kg/m³ = 5e-6 m³
        #   duration = 5e-6 m³ / 1e-9 m³/s = 5000 seconds
    
    Raises:
        ValueError: If volumetric_flow_rate or droplet_diameter <= 0
        ValueError: If neither duration nor mass parameters provided
        ValueError: If wt% provided without complete mass/density info
        ValueError: If both methods provided but disagree > 0.1%
    """
    enabled: bool = False
    volumetric_flow_rate: float = 0.0  # m³/s
    droplet_diameter: float = 0.0  # m
    liquid_addition_start: float = 0.0  # s
    liquid_addition_duration: Optional[float] = None  # s
    target_wt_percent: Optional[float] = None  # %
    solid_mass_in_mixer: Optional[float] = None  # kg
    rho_solid: Optional[float] = None  # kg/m³
    rho_liquid: Optional[float] = None  # kg/m³
    
    # Deprecated (backward compatibility)
    initialization_type: Optional[InitializationType] = None
    n_comp: Optional[int] = None
    
    # Batch size for droplet distribution (computational weight per event)
    # Required when enabled=True. No default value - must be set explicitly.
    batch_size: Optional[float] = None
    
    # Consistency check tolerance (relative)
    consistency_tol: float = 0.001  # 0.1%
    
    def __post_init__(self) -> None:
        if not self.enabled:
            return
        
        # Basic validation (always required)
        if self.volumetric_flow_rate <= 0.0:
            raise ValueError("volumetric_flow_rate must be positive when enabled")
        if self.droplet_diameter <= 0.0:
            raise ValueError("droplet_diameter must be positive when enabled")
        
        # Validate batch_size - warn if not set, use default value
        if self.batch_size is None:
            import warnings
            warnings.warn(
                "batch_size not specified for nucleation. Using default value of 25.0. "
                "It is recommended to set batch_size explicitly in create_nucleation_handler().",
                UserWarning,
                stacklevel=2
            )
            self.batch_size = 25.0
        elif self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}. "
                "Typical values: 10-50."
            )
        
        # Check parameter combinations
        has_duration = self.liquid_addition_duration is not None and self.liquid_addition_duration > 0.0
        has_wt = self.target_wt_percent is not None and self.target_wt_percent > 0.0
        has_mass = self.solid_mass_in_mixer is not None and self.solid_mass_in_mixer > 0.0
        has_rho_solid = self.rho_solid is not None and self.rho_solid > 0.0
        has_rho_liquid = self.rho_liquid is not None and self.rho_liquid > 0.0
        
        # Case 1: Only duration provided (legacy DURATION mode)
        if has_duration and not has_wt:
            # Valid: simple time-based addition
            pass
        
        # Case 2: wt% + mass + densities provided (calculate duration)
        elif has_wt and has_mass and has_rho_solid and has_rho_liquid:
            # Calculate duration from mass parameters
            mass_liquid = self.solid_mass_in_mixer * (self.target_wt_percent / 100.0)
            volume_liquid = mass_liquid / self.rho_liquid
            duration_calculated = volume_liquid / self.volumetric_flow_rate
            
            if has_duration:
                # Case 2a: Both provided - consistency check
                rel_diff = abs(self.liquid_addition_duration - duration_calculated) / duration_calculated
                if rel_diff > self.consistency_tol:
                    raise ValueError(
                        f"Inconsistent parameters: liquid_addition_duration={self.liquid_addition_duration}s "
                        f"differs from calculated duration={duration_calculated:.6f}s "
                        f"(from wt%={self.target_wt_percent}%, solid_mass={self.solid_mass_in_mixer}kg). "
                        f"Relative difference: {rel_diff*100:.3f}% "
                        f"(tolerance: {self.consistency_tol*100:.1f}%). "
                        f"Please check your input parameters for physical consistency."
                    )
            else:
                # Case 2b: Only mass params - set calculated duration
                self.liquid_addition_duration = duration_calculated
        
        # Case 3: wt% provided WITHOUT complete mass/density info - INVALID
        elif has_wt and not (has_mass and has_rho_solid and has_rho_liquid):
            missing = []
            if not has_mass:
                missing.append("solid_mass_in_mixer")
            if not has_rho_solid:
                missing.append("rho_solid")
            if not has_rho_liquid:
                missing.append("rho_liquid")
            raise ValueError(
                f"target_wt_percent requires solid_mass_in_mixer, rho_solid, and rho_liquid. "
                f"Missing: {', '.join(missing)}"
            )
        
        # Case 4: No duration and no wt% - INVALID
        elif not has_duration and not has_wt:
            raise ValueError(
                "Either liquid_addition_duration OR (target_wt_percent + solid_mass_in_mixer + densities) "
                "must be specified for nucleation."
            )
        
        # Set deprecated initialization_type for backward compatibility
        if self.initialization_type is None:
            if has_wt and has_mass:
                self.initialization_type = InitializationType.MASS
            else:
                self.initialization_type = InitializationType.DURATION
    
    @property
    def liquid_addition_end(self) -> float:
        """End time of liquid addition."""
        return self.liquid_addition_start + self.liquid_addition_duration
    
    @property
    def droplet_volume(self) -> float:
        """Volume of a single spherical droplet [m³]."""
        radius = self.droplet_diameter / 2.0
        return (4.0 / 3.0) * np.pi * radius ** 3


class NucleationHandler:
    """
    Handler for liquid addition via droplet distribution during Monte Carlo simulation.
    
    Physical Model:
        Liquid droplets collide randomly with particles in the mixer.
        All particles have equal collision probability regardless of size,
        implemented via weight-based uniform sampling.
    
    How it works:
        1. Determine number of physical droplets to add based on flow rate and dt
        2. Select target particle uniformly by physical count (using Fenwick sampler)
        3. Add droplet volume to selected particle's liquid content
        4. Update particle properties (saturation, porosity if needed)
    
    DSMC Scaling (critical for correct physics):
        
        Computational Weight (W):
            Each computational particle represents W physical particles.
            W = N_physical / N_computational (varies during simulation)
            
            Example: A particle with W=50 represents 50 identical physical particles.
                     When hit by a droplet, all 50 are considered hit.
        
        Control Volume Scaling:
            DSMC maintains constant particle density: Sum(W) / Vc = constant
            As particles agglomerate, Vc grows to keep density constant.
            
            Physical droplets per event = (Vc_ref / Vc) × W_selected
            
            Where:
                Vc_ref: Initial control volume (at t=0)
                Vc: Current control volume (changes during simulation)
                W_selected: Weight of the hit particle
            
            Why this matters:
                - Early simulation: Vc small → more physical droplets per event
                - Late simulation: Vc large → fewer physical droplets per event
                - Result: Physical flow rate (m³/s) stays constant!
        
        droplet_comp vs droplet_phys:
            droplet_comp: Always 1 (one computational event)
            droplet_phys: Actual physical droplets represented
            
            Formula: droplet_phys = (Vc_ref / Vc) × W_selected
            
            Typical values:
                Start: Vc_ref/Vc ≈ 1.0, W ≈ 10-100 → 10-100 physical droplets
                End: Vc_ref/Vc ≈ 0.3-0.7, W ≈ 50-500 → 15-350 physical droplets
    
    Key Features:
        - Time-based liquid addition (active window: start to start+duration)
        - Weight-based Fenwick sampler for O(log n) uniform physical particle selection
        - Incremental sampler updates when weights change (efficient)
        - Liquid remainder accumulation across time steps (no rounding errors)
        - DSMC-consistent statistics tracking (normalized to reference volume)
    
    Attributes:
        solver: Parent MCPBESolver instance
        config: NucleationConfig with all parameters
        _current_time: Current simulation time [s]
        _Vc_reference: Reference control volume for DSMC statistics [m³]
        _droplets_added_total: Total physical droplets added (weighted sum)
        _liquid_volume_added_total: Total liquid volume added [m³]
    
    Usage:
        >>> handler = NucleationHandler(solver, config)
        >>> # In solve loop, after each MC event:
        >>> handler.step(current_time, dt_event)
    """
    
    def __init__(self, solver, config: NucleationConfig):
        """
        Initialize nucleation handler.
        
        Args:
            solver: Parent MCPBESolver instance
            config: NucleationConfig instance
        
        Attributes:
            solver: Parent solver reference
            config: Configuration object
            _current_time: Current simulation time [s]
            _Vc_reference: Reference control volume for DSMC statistics [m³]
            _droplets_added_total: Total physical droplets added (weighted)
            _liquid_volume_added_total: Total liquid volume added [m³]
            _n_particles_real: Real particle count from solid_mass (if provided)
            _weight_sampler: Fenwick sampler for uniform physical particle selection
            _pending_weight_updates: Accumulated weight changes for incremental update
            _use_incremental_updates: Enable O(log n) incremental sampler updates
        """
        self.solver = solver
        self.config = config
        
        # Internal state
        self._current_time = 0.0
        self._next_nucleation_time = config.liquid_addition_start
        
        # Adaptive wt% checking (for performance)
        self._last_wt_check_time = -1.0
        self._last_wt_value = None
        self._wt_check_interval = 0.01
        
        # Statistics with Vc normalization for DSMC consistency
        self._Vc_reference = solver.Vc
        self._droplets_added_total = 0
        self._liquid_volume_added_total = 0.0
        
        # Debug tracking (from current debugging session)
        self._debug_last_print_time = -1.0
        self._debug_print_interval = 0.1
        
        # Particle count from solid_mass (statistics/validation only)
        self._n_particles_real = None
        self._flow_rate_scaled = config.volumetric_flow_rate
        
        if config.solid_mass_in_mixer is not None:
            self._initialize_mass_mode()
        
        # Window tracking
        self._mc_events_in_window = False
        self._nucleation_triggered_at_window_end = False
        self._was_in_window = False
        self._first_event_after_window = True
        
        # Track last time we were INSIDE the window (for finalize_after_solve)
        self._last_time_in_window = None
        
        # RNG shortcut
        self._rng = solver._rng
        
        # Weight-based sampler for uniform physical particle selection
        self._weight_sampler: Optional[FenwickSampler] = None
        self._weight_array: Optional[np.ndarray] = None
        
        # Sampler state tracking
        self._sampler_generation = -1
        self._last_W_sum = -1.0
        self._last_a_tot = -1
        
        # PERFORMANCE: Incremental sampler updates (O(log n) vs O(n))
        self._pending_weight_updates: list[tuple[int, float]] = []
        self._use_incremental_updates = True
    
    def _initialize_mass_mode(self) -> None:
        """
        Calculate real particle count from solid_mass_in_mixer.
        
        Used for statistics and validation only. Flow rate is NOT scaled -
        DSMC maintains constant represented particle count (Sum(W)/Vc).
        
        Raises:
            ValueError: If required parameters (solid_mass, rho_solid, solver.x) missing
        """
        config = self.config
        solver = self.solver
        
        if config.solid_mass_in_mixer is None:
            return
        
        # Calculate particle volume (monodisperse assumption)
        if not hasattr(solver, 'x') or solver.x is None or len(solver.x) < 1:
            raise ValueError(
                "solid_mass_in_mixer requires solver.x to be set (particle diameter)."
            )
        
        diameter = float(solver.x[0])
        v_particle = (np.pi / 6.0) * diameter ** 3
        
        if config.rho_solid is None:
            raise ValueError(
                "solid_mass_in_mixer requires rho_solid to be set."
            )
        mass_particle = v_particle * config.rho_solid
        
        # Real particle count in experiment
        self._n_particles_real = config.solid_mass_in_mixer / mass_particle
        
        # Verify solver W scaling (warning only, no enforcement)
        n_current = solver.a_tot
        sum_W = np.sum(solver.W[:n_current])
        n_represented = sum_W / solver.Vc
        
        ratio = n_represented / self._n_particles_real if self._n_particles_real > 0 else 1.0
        if ratio < 0.5 or ratio > 2.0:
            import warnings
            warnings.warn(
                f"MASS mode: Represented particle count differs from calculated real count.\n"
                f"  Calculated n_particles_real: {self._n_particles_real:.2e}\n"
                f"  Current n_represented (Sum(W)/Vc): {n_represented:.2e}\n"
                f"  Ratio: {ratio:.2f}\n"
                f"  Expected Sum(W) ≈ n_particles_real × Vc = {self._n_particles_real * solver.Vc:.2e}",
                UserWarning,
                stacklevel=2
            )
        
        self._flow_rate_scaled = config.volumetric_flow_rate
    
    def configure_time_step(self, dt: float) -> None:
        """
        Deprecated: This method has no effect.
        
        Nucleation timing is now fully driven by the MC event loop.
        The effective time step is calculated from the overlap between
        the event interval and the nucleation window.
        
        This method is kept for backward compatibility only.
        
        Args:
            dt: Ignored
        """
        # Deprecated: Nucleation is now event-driven, not time-step driven.
        # The effective dt is calculated in step() from the MC event timing.
        pass
    
    def mark_mc_event_in_window(self) -> None:
        """
        Mark that an MC event occurred during the nucleation window.
        
        Call this from the solver's main loop when an agglomeration/breakage
        event happens. Used to track event rate for diagnostics.
        
        NOTE: Nucleation is now TIME-BASED, not event-based. This flag is only
        used for the fallback trigger at window end if NO events occurred.
        """
        if self.config.enabled and self._in_addition_window(self._current_time):
            self._mc_events_in_window = True
            self._first_event_after_window = False  # Event happened inside window
    
    def check_first_event(self, current_time: float) -> None:
        """
        Check and handle first MC event for nucleation.
        
        Called at count == 0 (first MC event) to handle the case where
        the nucleation window ended before or at the first event.
        
        Args:
            current_time: Time of first MC event [s]
        """
        if not self.config.enabled:
            return
        
        # If first event is after window end, trigger manual nucleation
        # with full window duration
        if current_time > self.config.liquid_addition_end:
            self._trigger_manual_nucleation_at_window_end()
            self._nucleation_triggered_at_window_end = True
    
    def finalize_after_solve(self, final_time: float, event_count: int) -> None:
        """
        Finalize nucleation after MC loop completes.
        
        Handles:
        - No MC events during simulation → manual trigger
        - Remaining liquid in _liquid_remainder → distribute at window end
        - Window ended between last event and window_end → distribute remaining time
        
        Args:
            final_time: Final simulation time [s]
            event_count: Number of MC events that occurred
        """
        if not self.config.enabled:
            return
        
        # If no events occurred AND window has passed, trigger manually
        if event_count == 0 and final_time >= self.config.liquid_addition_end:
            if not self._nucleation_triggered_at_window_end:
                self._trigger_manual_nucleation_at_window_end()
                self._nucleation_triggered_at_window_end = True
        
        # NEW: Handle case where window ended AFTER last event but BEFORE next event
        # This captures the time slice [last_event_time, window_end] that was missed
        if self._was_in_window and not self._nucleation_triggered_at_window_end:
            window_end = self.config.liquid_addition_end
            
            # Use last time we were INSIDE the window, not current time!
            # Current time might be AFTER window_end (first event outside window)
            last_window_time = getattr(self, '_last_time_in_window', self._current_time)
            
            if last_window_time is not None and last_window_time < window_end:
                # There's unprocessed time between last in-window event and window end
                dt_remaining = window_end - last_window_time
                v_liquid_remaining = self._flow_rate_scaled * dt_remaining
                
                print(f"\n[FINALIZE DEBUG] Processing remaining window time:")
                print(f"  Last time IN WINDOW: {last_window_time:.6f}s")
                print(f"  Window end:          {window_end:.6f}s")
                print(f"  dt_remaining:        {dt_remaining:.6f}s")
                print(f"  v_liquid:            {v_liquid_remaining:.6e} m³")
                
                if v_liquid_remaining > 0:
                    self._ensure_samplers()
                    self._distribute_liquid_volume(v_liquid_remaining, is_manual_trigger=False)
        
        # Distribute remaining liquid accumulated during last step
        # This handles the case where _liquid_remainder > 0 after window closes
        if hasattr(self, '_liquid_remainder'):
            print(f"\n[FINALIZE DEBUG] _liquid_remainder before distribution: {self._liquid_remainder:.6e} m³")
            if self._liquid_remainder > 0:
                self._distribute_remaining_liquid(final_time)
            print(f"[FINALIZE DEBUG] _liquid_remainder after distribution: {self._liquid_remainder:.6e} m³")
        
        # Print final debug status
        self.print_debug_status(force=True)
        print("\n[NUCLEATION DEBUG] Simulation complete - Final statistics printed above")
    
    def _distribute_remaining_liquid(self, final_time: float) -> None:
        """
        Distribute remaining liquid accumulated in _liquid_remainder.
        
        Called at simulation end to ensure all accumulated liquid is distributed,
        even if the nucleation window has closed. Uses fractional droplet distribution
        for remainders between 0.5 and 1.0 droplets to avoid systematic under-addition.
        
        Physical Model:
            Instead of discarding sub-droplet remainders (e.g., 0.82 droplets),
            distribute the EXACT remaining volume as a "fractional droplet".
            This avoids systematic bias while maintaining mass conservation.
        
        Args:
            final_time: Final simulation time [s]
        """
        if not hasattr(self, '_liquid_remainder') or self._liquid_remainder <= 0:
            return
        
        v_droplet = self.config.droplet_volume
        n_droplets_exact = self._liquid_remainder / v_droplet
        
        # Discard only truly negligible remainders (< 0.001 droplets)
        # THRESHOLD REDUCED: From 0.1 to 0.001 to minimize systematic liquid loss
        if n_droplets_exact < 0.001:
            print(f"\n[REMAINDER] Keeping {self._liquid_remainder:.3e} m³ "
                  f"({n_droplets_exact:.6f} droplets) as numerical remainder")
            return
        
        self._ensure_samplers()
        vc_scale = self._Vc_reference / self.solver.Vc
        
        # Case 1: Significant remainder (≥ 1 droplet) - distribute integer droplets
        # Case 2: Fractional remainder (0.1-1.0 droplets) - distribute exact volume
        if n_droplets_exact >= 1.0:
            n_droplets_full = int(n_droplets_exact)
            print(f"\n[REMAINDER] Distributing {n_droplets_full} remaining droplets "
                  f"(exact: {n_droplets_exact:.2f}) at t={final_time:.4f}s...")
            
            v_target = n_droplets_full * v_droplet
            v_distributed = 0.0
            max_consecutive_failures = max(1, n_droplets_full // 10 + 1)
            consecutive_failures = 0
            
            while v_distributed < v_target and consecutive_failures < max_consecutive_failures:
                v_remaining = v_target - v_distributed
                n_physical_remaining = v_remaining / v_droplet
                dW = self._distribute_one_droplet_with_dW(v_droplet, max_physical_droplets=n_physical_remaining)
                
                if dW > 0:
                    effective_dW = dW * vc_scale
                    v_event = v_droplet * effective_dW
                    v_distributed += v_event
                    self._droplets_added_total += effective_dW
                    self._liquid_volume_added_total += v_event
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            
            self._liquid_remainder -= v_distributed
        
        else:
            # Fractional droplet: distribute exact remaining volume
            # Note: _liquid_remainder is PHYSICAL volume (already scaled by flow_rate_per_particle).
            # We distribute it as a single computational event with capped dW.
            print(f"\n[REMAINDER] Distributing fractional droplet "
                  f"({n_droplets_exact:.2f} × {v_droplet:.3e} m³) at t={final_time:.4f}s...")
            
            # CRITICAL: Pass ACTUAL droplet volume and cap max_physical_droplets!
            # This ensures dW is capped correctly: dW <= n_droplets_exact / vc_scale
            # Without this cap, dW would be W[i]/2.0 (~46.5) and the particle would
            # receive dW × v_droplet = 46.5 × 5.236e-13 = 2.43e-11 m³ (83x too much!)
            dW = self._distribute_one_droplet_with_dW(
                v_droplet=v_droplet,  # Actual droplet volume for saturation calc
                max_physical_droplets=n_droplets_exact  # Cap: only 0.56 physical droplets!
            )
            
            if dW > 0:
                effective_dW = dW * vc_scale
                # The distributed physical volume is: effective_dW × v_droplet
                # With proper capping: effective_dW ≈ n_droplets_exact = 0.56
                v_physical_distributed = effective_dW * v_droplet
                
                # Statistics: Add PHYSICAL droplet count
                # effective_dW IS the number of physical droplets distributed
                self._droplets_added_total += effective_dW
                self._liquid_volume_added_total += v_physical_distributed
                self._liquid_remainder -= v_physical_distributed
                
                print(f"[REMAINDER] Distributed fractional droplet: "
                      f"{v_physical_distributed:.3e} m³ (dW={dW:.4f}, effective_dW={effective_dW:.4f})")
                
                print(f"[REMAINDER] Distributed fractional droplet: "
                      f"{v_physical_distributed:.3e} m³ (dW={dW:.2f}, effective_dW={effective_dW:.2f})")
            else:
                print(f"[REMAINDER] WARNING: Failed to distribute fractional droplet")
        
        # Sanity check
        if self._liquid_remainder < -1e-15:
            raise RuntimeError(
                f"Nucleation: _liquid_remainder became negative after final distribution "
                f"({self._liquid_remainder:.6e})"
            )
    
    def _reset_mc_event_flag_if_needed(self, current_time: float) -> None:
        """
        Reset the MC event flag when entering a new nucleation window.
        
        Args:
            current_time: Current simulation time [s]
        """
        if not self.config.enabled:
            return
        
        # Check if we just entered the nucleation window
        start = self.config.liquid_addition_start
        end = self.config.liquid_addition_end
        
        # If we're in the window but wasn't in it before, reset the flag
        if start <= current_time < end and not self._was_in_window:
            self._mc_events_in_window = False
    
    def step(self, current_time: float, solver_last_dt: float) -> None:
        """
        Execute one nucleation step.
        
        Called from solver's main loop after each agglomeration/breakage event.
        Calculates effective time overlap between event interval and addition window.
        
        Args:
            current_time: Current simulation time [s]
            solver_last_dt: Time since last MC event [s]
        """
        if not self.config.enabled:
            return
        
        self._current_time = current_time
        
        # Track if we're currently in the window
        currently_in_window = self._in_addition_window(current_time)
        
        # Reset MC event flag when entering a new nucleation window
        self._reset_mc_event_flag_if_needed(current_time)
        
        # Update "was in window" flag AND track last time inside window
        if currently_in_window:
            self._was_in_window = True
            self._last_time_in_window = current_time
        
        # Check if we've passed the end of the nucleation window
        # Trigger manual nucleation if no MC events occurred during the window
        past_window_end = current_time >= self.config.liquid_addition_end
        
        if past_window_end and not self._mc_events_in_window and not self._nucleation_triggered_at_window_end:
            self._trigger_manual_nucleation_at_window_end()
            self._nucleation_triggered_at_window_end = True
            return
        
        if not currently_in_window:
            return
        
        # Check target wt% for early stop (if configured)
        if self.config.target_wt_percent is not None:
            if self._should_check_wt(current_time) and self.has_reached_target_wt():
                return
        
        self._ensure_samplers()
        
        # Calculate effective time: overlap between event interval and nucleation window
        window_start = self.config.liquid_addition_start
        window_end = self.config.liquid_addition_end
        event_start = current_time - solver_last_dt
        event_end = current_time
        
        if self._first_event_after_window and current_time > window_end:
            dt_effective = self.config.liquid_addition_duration
            self._first_event_after_window = False
        else:
            overlap_start = max(window_start, event_start)
            overlap_end = min(window_end, event_end)
            dt_effective = max(0.0, overlap_end - overlap_start)

        # Early return if no overlap (event outside window)
        if dt_effective <= 0.0:
            return
        
        # Calculate liquid volume for this time step
        v_liquid_this_step = self._flow_rate_scaled * dt_effective
        
        # Use common distribution logic (shared with manual trigger)
        self._distribute_liquid_volume(v_liquid_this_step, is_manual_trigger=False)
        
        # Print debug status after distributing droplets
        self.print_debug_status(force=False)
    
    def _trigger_manual_nucleation_at_window_end(self) -> None:
        """
        Manually trigger nucleation at window end if no MC events occurred.
        
        Ensures liquid is added even for systems with very slow agglomeration
        where no MC events happen during the nucleation window.
        
        Uses the same distribution logic as regular step(), just with the
        total volume for the entire window instead of per-step volume.
        """
        import warnings
        warnings.warn(
            f"No MC events occurred during nucleation window "
            f"[{self.config.liquid_addition_start:.2f}s, {self.config.liquid_addition_end:.2f}s]. "
            f"Manually triggering nucleation at window end. "
            f"Consider increasing liquid_addition_duration or CORR_BETA for better results.",
            UserWarning,
            stacklevel=2
        )
        
        print(f"\n[MANUAL TRIGGER] Starting manual nucleation...")
        
        # Calculate total liquid for the entire window
        v_liquid_total = self._flow_rate_scaled * self.config.liquid_addition_duration
        
        if v_liquid_total <= 0.0:
            print(f"  [MANUAL TRIGGER] No liquid to add (v_liquid_total={v_liquid_total})")
            return
        
        v_droplet = self.config.droplet_volume
        n_droplets_total = int(v_liquid_total / v_droplet)
        
        print(f"  Total liquid to add: {v_liquid_total:.6e} m³")
        print(f"  Number of droplets: {n_droplets_total:.0f}")
        print(f"  Current n_comp: {self.solver.a_tot}")
        
        # Use common distribution logic (shared with regular step)
        # Pass is_manual_trigger=True for detailed progress output
        self._distribute_liquid_volume(v_liquid_total, is_manual_trigger=True)
        
        print(f"  [MANUAL TRIGGER] Complete:")
        print(f"    Total droplets distributed: {self._droplets_added_total:.2e}")
        print(f"    Total liquid distributed: {self._liquid_volume_added_total:.6e} m³")
        print(f"  Final n_comp: {self.solver.a_tot}")
        
        # Distribute any remaining liquid for consistency with natural nucleation
        if hasattr(self, '_liquid_remainder') and self._liquid_remainder > 0:
            self._distribute_remaining_liquid(self.solver._elapsed)
        
        # Print debug status AFTER updating statistics
        self.print_debug_status(force=True)
    
    def _in_addition_window(self, current_time: float) -> bool:
        """Check if current time is within liquid addition window."""
        return (
            current_time >= self.config.liquid_addition_start and
            current_time < self.config.liquid_addition_end
        )
    
    def _should_rebuild_sampler(self) -> bool:
        """
        Check if sampler needs rebuild.
        
        With incremental updates: rebuild only on structural changes (a_tot)
        or too many pending updates.
        
        Returns:
            True if sampler should be rebuilt
        """
        a_tot = self.solver.a_tot
        
        # Always rebuild if particle count changed
        a_tot_changed = (a_tot != self._last_a_tot)
        
        if a_tot_changed:
            self._last_a_tot = a_tot
            self._sampler_generation += 1
            return True
        
        # With incremental updates: don't rebuild based on W changes alone
        # The incremental updates handle weight changes efficiently
        if self._use_incremental_updates:
            # Only rebuild if too many pending updates (performance degradation)
            if len(self._pending_weight_updates) > a_tot * 2:
                # Too many incremental updates, full rebuild is more efficient
                self._pending_weight_updates.clear()
                self._sampler_generation += 1
                return True
            return False
        
        # Legacy mode (without incremental updates): rebuild on any W change
        W_sum = np.sum(self.solver.W[:a_tot])
        W_changed = False
        if self._last_W_sum > 0:
            rel_change = abs(W_sum - self._last_W_sum) / self._last_W_sum
            W_changed = rel_change > 1e-10
        else:
            W_changed = (W_sum != 0.0)
        
        if W_changed:
            self._last_W_sum = W_sum
            self._sampler_generation += 1
            return True
        
        return False
    
    def _ensure_samplers(self) -> None:
        """
        Build or rebuild weight-based sampler for uniform physical particle selection.
        
        Physical Model:
            To sample PHYSICAL particles uniformly, sample COMPUTATIONAL particles
            with probability proportional to their weight W:
            
                P(select particle i) = W[i] / Sum(W)
            
            This ensures each physical particle has equal collision probability.
        
        Performance:
            Uses incremental Fenwick tree updates (O(log n)) for weight changes.
            Full rebuild (O(n)) only when particle count (a_tot) changes.
        """
        a_tot = self.solver.a_tot
        if a_tot < 1:
            return
        
        # Check if rebuild is needed
        needs_rebuild = self._should_rebuild_sampler()
        
        # If using incremental updates and no structural change, apply pending updates
        if self._use_incremental_updates and not needs_rebuild and self._weight_sampler is not None:
            if self._pending_weight_updates:
                self._apply_pending_weight_updates()
            return
        
        # Full rebuild needed (initial build or structural change)
        # Build fresh array matching current a_tot
        weight_array_local = np.zeros(a_tot, dtype=float)
        
        # Use weights W for sampling (represents physical particle count)
        W = self.solver.W[:a_tot]
        
        # Filter valid particles (positive, finite weights)
        valid_mask = (W > 0) & np.isfinite(W)
        
        if not np.any(valid_mask):
            # No valid particles - create uniform weights as fallback
            weight_array_local[:] = 1.0 / a_tot
        else:
            # Copy weights using the mask
            weight_array_local[valid_mask] = W[valid_mask]
            
            # Normalize for numerical stability
            weight_sum = np.sum(weight_array_local)
            if weight_sum > 0:
                weight_array_local /= weight_sum
        
        # Store for potential reuse
        self._weight_array = weight_array_local
        
        # Build Fenwick sampler from scratch
        self._weight_sampler = FenwickSampler(weight_array_local.copy())
        
        # Clear pending updates after rebuild
        self._pending_weight_updates.clear()
        
        # Update tracking state
        self._last_a_tot = a_tot
        self._last_W_sum = float(np.sum(W))
    
    def _apply_pending_weight_updates(self) -> None:
        """
        Apply accumulated weight changes to the sampler incrementally.
        
        PERFORMANCE:
        - Each update: O(log n) instead of O(n) for full rebuild
        - For k updates on n particles: O(k log n) vs O(k n)
        - Example: 1000 updates on 500 particles = 1000× faster!
        
        This method is called automatically by _ensure_samplers() when
        incremental updates are enabled.
        """
        if not self._pending_weight_updates or self._weight_sampler is None:
            return
        
        solver = self.solver
        a_tot = solver.a_tot
        W = solver.W[:a_tot]
        
        for idx, new_weight in self._pending_weight_updates:
            # Skip if index is out of range (particle was removed)
            if idx < 0 or idx >= a_tot:
                continue
            
            # Get current weight from solver state (authoritative source)
            current_w = float(W[idx])
            
            # Skip invalid weights
            if current_w <= 0 or not np.isfinite(current_w):
                continue
            
            # Normalize weight for sampler (same normalization as full rebuild)
            # Note: We use the global sum for consistency
            W_sum = self._last_W_sum
            if W_sum <= 0:
                continue
            
            normalized_weight = current_w / W_sum
            
            # Update sampler incrementally (O(log n))
            try:
                self._weight_sampler.update(idx, normalized_weight)
            except (IndexError, ValueError):
                # Sampler update failed, trigger full rebuild
                self._sampler_generation += 1
                self._pending_weight_updates.clear()
                self._ensure_samplers()
                return
        
        # Clear pending updates after successful application
        self._pending_weight_updates.clear()
    
    def _record_weight_change(self, idx: int, new_weight: float) -> None:
        """
        Record a weight change for later incremental update.
        
        Call this whenever a particle's weight changes during nucleation.
        The actual sampler update is deferred until _ensure_samplers() is called.
        
        Args:
            idx: Particle index whose weight changed
            new_weight: New weight value (absolute, not normalized)
        """
        if self._use_incremental_updates:
            self._pending_weight_updates.append((idx, new_weight))
    
    def _select_particle_uniform_physical(self) -> int:
        """
        Select a computational particle such that all PHYSICAL particles
        have equal probability of being chosen.
        
        This is achieved by sampling computational particles with probability
        proportional to their weight W.
        
        Kernel Framework Support:
            If solver has kernel_manager with liquid_dist_kernel, delegates
            to kernel for advanced selection strategies (surface-weighted,
            saturation-preferential, etc.).
        
        Returns:
            Index of selected computational particle, or -1 if no valid particle
        """
        solver = self.solver
        
        # Try kernel framework first
        if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
            liq_kernel = solver.kernel_manager.liquid_dist_kernel
            if liq_kernel is not None:
                # Delegate to kernel
                return solver.kernel_manager.select_liquid_target(
                    solver=solver,
                    v_droplet=self.config.droplet_volume,
                    current_time=getattr(solver, '_elapsed', 0.0)
                )
        
        # Fallback to legacy weight-based sampling
        a_tot = solver.a_tot
        if a_tot < 1:
            return -1
        
        # Ensure sampler is built
        if self._weight_sampler is None:
            self._ensure_samplers()
        
        if self._weight_sampler is None or self._weight_sampler.total() <= 0:
            # Fallback: uniform random over computational particles
            return int(self._rng.integers(0, a_tot))
        
        # Sample weighted by W → uniform over physical particles
        return int(self._weight_sampler.sample(self._rng))
        
    def _find_similar_particle(self, V_solid_target: float, liquid_target: float, 
                                poro_target: Optional[float] = None, 
                                sat_target: Optional[float] = None,
                                tol: float = 0.00001) -> int:
        """
        Find existing particle with similar properties (for merging nucleated particles).
        
        Uses NESTED IF structure for optimal performance:
        - LEVEL 1: V_dry (fastest, most selective) → outermost
        - LEVEL 2: liquid_volume (second fastest) → middle
        - LEVEL 3: porosity (requires NaN check) → inner
        - LEVEL 4: saturation (most expensive) → innermost
        
        Args:
            V_solid_target: Target solid volume
            liquid_target: Target liquid volume
            poro_target: Target porosity (optional)
            sat_target: Target saturation (optional)
            tol: Relative tolerance for comparison
            
        Returns:
            Index of similar particle, or -1 if not found
        """
        solver = self.solver
        n_active = solver.a_tot
        
        # Pre-compute tolerances (avoid repeated multiplication/division)
        V_tol = tol * V_solid_target
        liq_tol = tol * liquid_target
        poro_tol = tol * poro_target if poro_target is not None else 0.0
        sat_tol = tol * sat_target if sat_target is not None else 0.0
        
        for k in range(n_active):
            # ===== LEVEL 1: Dry Volume (fastest check, most selective) =====
            V_dry_k = float(solver.V_flat[-1, k])
            if abs(V_dry_k - V_solid_target) > V_tol:
                continue  # Exit immediately - doesn't match
            
            # ===== LEVEL 2: Liquid Volume (second fastest) =====
            liq_k = solver.liquid_volume[k] if hasattr(solver, 'liquid_volume') else 0.0
            if abs(liq_k - liquid_target) > liq_tol:
                continue  # Exit - liquid doesn't match
            
            # ===== LEVEL 3: Porosity (requires NaN check) =====
            if poro_target is not None:
                poro_k = solver.porosity[k]
                if np.isnan(poro_k):
                    continue  # Skip non-nucleated particles
                if abs(poro_k - poro_target) > poro_tol:
                    continue  # Exit - porosity doesn't match
            
            # ===== LEVEL 4: Saturation (innermost, most expensive) =====
            if sat_target is not None:
                sat_k = solver.saturation[k] if hasattr(solver, 'saturation') else 0.0
                if abs(sat_k - sat_target) > sat_tol:
                    continue  # Exit - saturation doesn't match
            
            # All criteria matched!
            return k
        
        return -1
    
    def _distribute_liquid_volume(self, v_liquid_to_distribute: float, 
                                   is_manual_trigger: bool = False) -> None:
        """
        Core distribution logic used by both step() and manual trigger.
        
        Physical Model:
            Distributes liquid volume as discrete droplets using DSMC weighting.
            Accumulates remainder across calls to avoid systematic loss from
            integer droplet discretization.
        
        Algorithm:
            1. Add new liquid to accumulated remainder
            2. Calculate integer droplets from total remainder
            3. Distribute droplets with volume tracking (not count tracking!)
            4. Distribute any remaining volume as mini-droplet (effective_dW = 1)
        
        Args:
            v_liquid_to_distribute: Physical liquid volume to distribute [m³]
            is_manual_trigger: If True, print detailed progress (for manual trigger)
        """
        # Initialize remainder accumulator if needed
        if not hasattr(self, '_liquid_remainder'):
            self._liquid_remainder = 0.0
        
        # Add new liquid to accumulated remainder
        self._liquid_remainder += v_liquid_to_distribute
        
        if self._liquid_remainder <= 0.0:
            return
        
        # Calculate number of whole droplets from accumulated liquid
        v_droplet = self.config.droplet_volume
        n_droplets_full = int(self._liquid_remainder / v_droplet)
        
        if n_droplets_full <= 0:
            return
        
        # DSMC scale factor for converting computational weight to physical droplets
        vc_scale = self._Vc_reference / self.solver.Vc
        
        # Ensure sampler is built
        self._ensure_samplers()
        
        # DEBUG: Print on first call in regular step (not manual trigger)
        if not is_manual_trigger and not hasattr(self, '_debug_loop_printed'):
            self._debug_loop_printed = True
            v_liquid_expected = self._flow_rate_scaled * 0.1
            print(f"    [LOOP DEBUG] n_droplets_full={n_droplets_full:.1f}, vc_scale={vc_scale:.3f}")
            print(f"    [LOOP DEBUG] _liquid_remainder={self._liquid_remainder:.3e} (expected after 0.1s: {v_liquid_expected:.3e})")
            print(f"    [LOOP DEBUG] W[0:5]={self.solver.W[:min(5,self.solver.a_tot)]}")
        
        # Distribute droplets with DSMC weight handling.
        # Track accumulated liquid VOLUME (not droplet count) to avoid floating-point
        # accumulation errors. Compute target volume from integer droplet count.
        v_target = n_droplets_full * v_droplet
        v_distributed = 0.0
        max_consecutive_failures = 1000
        consecutive_failures = 0
        
        # For manual trigger: track progress
        if is_manual_trigger and n_droplets_full > 100:
            print(f"  [MANUAL TRIGGER] Distributing {n_droplets_full} droplets...")
        
        while v_distributed < v_target and consecutive_failures < max_consecutive_failures:
            # Convert remaining volume to physical droplets for dW capping
            v_remaining = v_target - v_distributed
            n_physical_remaining = v_remaining / v_droplet
            
            dW = self._distribute_one_droplet_with_dW(v_droplet, max_physical_droplets=n_physical_remaining)
            
            if dW > 0:
                effective_dW = dW * vc_scale
                v_event = v_droplet * effective_dW
                v_distributed += v_event
                self._droplets_added_total += effective_dW
                self._liquid_volume_added_total += v_event
                consecutive_failures = 0
                
                # Progress report for manual trigger every 10%
                if is_manual_trigger and n_droplets_full > 100:
                    progress = v_distributed / v_target * 100
                    if int(progress) % 10 == 0 and progress > 0:
                        # Avoid duplicate prints
                        pass  # Progress tracking would need state variable
            else:
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    if is_manual_trigger:
                        print(f"  [MANUAL TRIGGER] Stopping after {consecutive_failures} consecutive failures")
                    break
        
        # Update remainder: subtract ACTUALLY distributed liquid volume
        self._liquid_remainder -= v_distributed
        
        # Sanity check: remainder must never be negative
        if self._liquid_remainder < -1e-15:  # Small tolerance for floating point
            raise RuntimeError(
                f"Nucleation: _liquid_remainder became negative ({self._liquid_remainder:.6e}). "
                f"This indicates a bug in droplet counting. "
                f"n_droplets_full={n_droplets_full}, v_target={v_target:.6e}, "
                f"v_distributed={v_distributed:.6e}, v_droplet={v_droplet:.6e}"
            )
        
        # Distribute remaining volume as exactly ONE physical droplet (effective_dW = 1)
        # This avoids systematic loss from discarding sub-droplet remainders.
        # Physical meaning: One real droplet of volume = remainder lands on a random particle.
        # THRESHOLD REDUCED: From 0.1 to 0.001 to minimize systematic liquid loss
        if self._liquid_remainder > 0 and self._liquid_remainder >= v_droplet * 0.001:
            solver = self.solver
            
            # Select one particle weighted by W (uniform over physical particles)
            i = self._select_particle_uniform_physical()
            
            if i >= 0 and i < solver.a_tot:
                # For effective_dW = 1, we need dW = 1 / vc_scale
                dW_for_one_physical = 1.0 / vc_scale
                
                # Create nucleated particle with the remainder volume
                # This represents exactly ONE physical droplet
                self._create_nucleated_particle_copy(i, dW_for_one_physical, self._liquid_remainder)
                
                # Reduce parent weight by dW
                solver.W[i] -= dW_for_one_physical
                self._record_weight_change(i, solver.W[i])
                
                if solver.W[i] <= 0:
                    solver._remove_particle_column(i)
                
                # Update statistics: count fractional droplets based on actual remainder volume
                # This avoids systematic overshoot when remainder < 1.0 droplet
                n_droplets_fractional = self._liquid_remainder / v_droplet
                self._droplets_added_total += n_droplets_fractional
                self._liquid_volume_added_total += self._liquid_remainder
                self._liquid_remainder = 0.0
        
        # Log any remaining liquid (should be zero or negligible now)
        if not is_manual_trigger and self._liquid_remainder > 1e-20:
            print(f"    [REMAINDER] {self._liquid_remainder:.3e} m³ carried to next step ({self._liquid_remainder/v_droplet:.2f} droplets)")
    
    def _distribute_one_droplet_with_dW(self, v_droplet: float, 
                                         max_physical_droplets: Optional[float] = None) -> float:
        """
        Distribute one computational droplet onto particles.
        
        Physical Model:
            One computational droplet event distributes a single droplet of volume v_droplet
            onto collected particles. The event weight dW determines how many PHYSICAL
            droplets this represents:
            
                n_physical_droplets = effective_dW = dW × (Vc_ref / Vc)
            
            Sampling is weighted by W to ensure uniform distribution over PHYSICAL particles.
        
        Algorithm:
            1. Select particle i weighted by W (uniform over physical particles)
            2. Collect additional particles j (weighted by W) until Sum(V_solid) >= v_droplet
            3. Compute dW from collected particle weight W_i
            4. Cap dW if max_physical_droplets is set: dW <= max_physical_droplets / vc_scale
            5. Create/merge nucleated particle with droplet volume v_droplet
            6. Update parent weights and samplers
        
        Args:
            v_droplet: Droplet volume [m³]
            max_physical_droplets: Limit on physical droplets to distribute. Caps dW such that
                                   effective_dW = dW × vc_scale <= max_physical_droplets
        
        Returns:
            dW: Weight consumed for this distribution event
        """
        solver = self.solver
        a_tot = solver.a_tot
        
        if a_tot < 1:
            return 0.0
        
        # Select first particle weighted by W (uniform over physical particles)
        i = self._select_particle_uniform_physical()
        
        if i < 0 or i >= a_tot:
            return 0.0
        
        # Collect particles until Sum(V_solid) >= v_droplet
        V_solid_sum = self._get_V_solid_single(i)
        max_attempts = a_tot * 3
        attempts = 0
        
        while V_solid_sum < v_droplet and attempts < max_attempts:
            attempts += 1
            j = self._select_particle_uniform_physical()
            
            if j < 0 or j >= a_tot:
                break
            
            child_idx = self._perform_nucleation_agglomeration(i, j)
            
            if child_idx < 0 or child_idx >= solver.a_tot:
                break
            
            i = child_idx
            V_solid_sum = self._get_V_solid_single(i)
        
        if V_solid_sum < v_droplet or i >= solver.a_tot:
            return 0.0
        
        # Compute dW from collected particle weight
        W_i = float(solver.W[i])
        if W_i <= 0:
            return 0.0
        
        dW = min(self.config.batch_size, W_i)
        
        # Cap dW to limit physical droplets distributed
        if max_physical_droplets is not None and max_physical_droplets > 0:
            vc_scale = self._Vc_reference / self.solver.Vc
            max_dW_allowed = max_physical_droplets / vc_scale
            dW = min(dW, max_dW_allowed)
        
        # ==========================================
        # PHASE 3: Add droplet to collected particles
        # ==========================================
        
        # Check if already nucleated
        poro_old = solver.porosity[i]
        already_nucleated = not np.isnan(poro_old)
        
        if already_nucleated:
            # Already nucleated: distribute droplet to subset of physical particles.
            # 
            # DSMC Logic:
            #   Parent n_comp represents W physical particles. When dW of them receive
            #   a droplet, we create a NEW n_comp (child) with:
            #     - W_child = dW (represents the particles that GOT the droplet)
            #     - Updated liquid, saturation properties
            #   Parent keeps its ORIGINAL properties and now represents only
            #     (W_parent - dW) physical particles that did NOT get a droplet.
            # 
            # Mass Conservation:
            #   - Parent: V_dry unchanged, poro unchanged, liquid unchanged ✓
            #   - Child:  V_dry copied from parent, liquid increased ✓
            #   - Total:  Σ(V_solid × W) conserved because V_solid doesn't change here
            
            current_liquid = solver.liquid_volume[i] if hasattr(solver, 'liquid_volume') else 0.0
            new_liquid = current_liquid + v_droplet
            V_solid_i = self._get_V_solid_single(i)
            poro_i = solver.porosity[i]
            sat_i = solver.saturation[i] if hasattr(solver, 'saturation') else 0.0
            
            # Calculate expected saturation after adding droplet
            V_pore_i = V_solid_i * poro_i / (1.0 - poro_i) if poro_i < 1.0 else 0.0
            V_liq_int_current = V_pore_i * sat_i if V_pore_i > 0 else 0.0
            V_liq_int_new = V_liq_int_current + v_droplet
            sat_new = V_liq_int_new / V_pore_i if V_pore_i > 0 else 0.0
            
            # Find similar particle with ALL criteria
            existing = self._find_similar_particle(
                V_solid_target=V_solid_i,
                liquid_target=new_liquid,
                poro_target=poro_i,
                sat_target=sat_new,
                tol=0.00001
            )
            
            if existing >= 0 and existing != i:
                # Found similar particle: transfer dW weight to it
                solver.W[existing] += dW
                solver.W[i] -= dW
                
                # Record weight changes for incremental sampler update
                self._record_weight_change(existing, solver.W[existing])
                self._record_weight_change(i, solver.W[i])
                
                if solver.W[i] <= 0:
                    solver._remove_particle_column(i)
                
                return dW
            
            # No similar particle: create new one with updated liquid
            self._create_nucleated_particle_copy(i, dW, new_liquid)
            
            # Reduce parent weight
            solver.W[i] -= dW
            
            # Record weight changes for incremental sampler update
            self._record_weight_change(i, solver.W[i])
            # New particle weight recorded in _create_nucleated_particle_copy
            
            if solver.W[i] <= 0:
                solver._remove_particle_column(i)
            
            return dW
        
        else:
            # First-time nucleation: convert Vollkörper → porous particle.
            # 
            # DSMC Logic:
            #   Parent (Vollkörper) splits into two n_comp:
            #     - Child:  porous (poro=0.4), has liquid, V_dry = V_solid/(1-poro)
            #     - Parent: remains Vollkörper, NO liquid, V_dry = V_solid (unchanged!)
            # 
            # Volume Transformation:
            #   Vollkörper: V_dry = V_solid (no pores)
            #   Porous:     V_dry = V_solid / (1 - poro) > V_solid (has pores!)
            # 
            # CRITICAL: Parent V_dry must be set to V_solid (not old V_dry)!
            #   Because parent represents the fraction that stayed Vollkörper.
            #   Its V_dry must equal its V_solid since it has no pores.
            
            V_solid = V_solid_sum  # Total solid volume from collected particles
            
            new_poro = 0.4
            V_particle_dry_new = V_solid / (1.0 - new_poro)
            V_pore = V_particle_dry_new * new_poro
            
            new_total_liquid = v_droplet  # First droplet
            
            # Create new particle (nucleated fraction) - gets ALL the liquid
            self._create_nucleated_particle_copy(i, dW, new_total_liquid)
            
            # IMPORTANT: Parent stays VOLLLKÖRPER (no porosity, no liquid, no saturation)!
            # Parent represents particles that did NOT receive a droplet.
            # Update V_flat[:dim] to solid volume
            if solver.dim == 1:
                solver.V_flat[0, i] = V_solid
            else:
                # For multi-component: scale proportionally
                scale = V_solid / V_solid_sum if V_solid_sum > 0 else 1.0
                solver.V_flat[:solver.dim, i] *= scale
            
            # V_flat[-1] = V_solid for Vollkörper (no pores!)
            solver.V_flat[-1, i] = V_solid
            
            # Reduce parent weight
            solver.W[i] -= dW
            
            # Record weight changes for incremental sampler update
            self._record_weight_change(i, solver.W[i])
            # New particle weight recorded in _create_nucleated_particle_copy
            
            if solver.W[i] <= 0:
                solver._remove_particle_column(i)
            
            return dW
    
    def _get_V_solid_single(self, idx: int) -> float:
        """
        Get solid volume for a single particle.
        
        Volume Semantics:
            - V_flat[-1,:] stores V_dry (= V_solid + V_pore), NOT V_solid!
            - For Vollkörper (NaN porosity): V_solid = V_dry
            - For porous particles: V_solid = V_dry × (1 - porosity)
        
        Mass Conservation:
            - Σ(V_solid × W) is conserved (mass conservation law)
            - Σ(V_dry × W) is NOT conserved (changes when porosity changes)
        
        Args:
            idx: Particle index
            
        Returns:
            Solid volume in m³
        """
        solver = self.solver
        V_dry = float(solver.V_flat[-1, idx])
        poro = solver.porosity[idx] if hasattr(solver, 'porosity') else np.nan
        
        if np.isnan(poro):
            return V_dry  # Vollkörper
        else:
            return V_dry * (1.0 - poro)
    
    def _create_nucleated_particle_copy(self, src_idx: int, dW: float, liquid: float):
        """
        Create a new nucleated particle as a copy of source with specified weight and liquid.
        
        IMPORTANT: Uses PorosityGrowthKernel for consistent physics!
        Default: volume_mixing with 0.4 default porosity (enables porosity formation!)
        
        Args:
            src_idx: Index of source particle to copy
            dW: Weight for new particle
            liquid: Liquid volume for new particle
        """
        solver = self.solver
        
        # Copy solid volumes
        V_solid_src = solver.V_flat[:solver.dim, src_idx].copy()
        V_solid_total = float(np.sum(V_solid_src))
        
        # Get porosity growth kernel (create default if not exists)
        porosity_kernel = None
        if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
            porosity_kernel = solver.kernel_manager.porosity_growth_kernel
        
        if porosity_kernel is None:
            # Create default volume_mixing kernel
            from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
            porosity_kernel = get_porosity_growth_kernel('volume_mixing')
        
        # Compute nucleation porosity via kernel
        # Kernel decides its own default (volume_mixing: 0.4, incomplete_mixing: configurable)
        v_dry_new, poro_new = porosity_kernel.compute_nucleation_porosity(
            v_solid=V_solid_total,
            v_liquid=liquid,
            nucleation_params=None,  # Kernel uses its own default
            solver=solver
        )
        
        # Create new particle
        solver._append_particle_column(V_solid_src)
        new_idx = solver.a_tot - 1
        
        # Set properties with CORRECT V_dry from kernel
        solver.V_flat[-1, new_idx] = v_dry_new
        solver.W[new_idx] = dW
        
        if hasattr(solver, 'liquid_volume'):
            solver.liquid_volume[new_idx] = liquid
        
        if hasattr(solver, 'porosity'):
            solver.porosity[new_idx] = poro_new
        
        if hasattr(solver, 'saturation'):
            # Determine initial liquid distribution (internal vs external)
            # Check if liquid_internalization kernel is active
            has_internalization_kernel = (
                hasattr(solver, 'kernel_manager') and 
                solver.kernel_manager is not None and
                solver.kernel_manager.liquid_internalization_kernel is not None
            )
            
            V_pore = v_dry_new * poro_new if not np.isnan(poro_new) else 0.0
            if V_pore > 0:
                if has_internalization_kernel:
                    # Kernel is active: All liquid starts as external (saturation = 0)
                    # Internalization will occur over time via the continuous process kernel
                    # This avoids double-counting or conflicting physics
                    solver.saturation[new_idx] = 0.0
                else:
                    # Kernel NOT active: Use empirical 40/60 split (legacy behavior)
                    # 40% of liquid goes internal, 60% remains external
                    # Internal fraction is capped at pore capacity
                    # Option B2: V_liq_int = min(0.4 × liquid, V_pore)
                    V_liq_int_target = 0.4 * liquid
                    V_liq_int = min(V_liq_int_target, V_pore)  # Cap at pore capacity
                    solver.saturation[new_idx] = V_liq_int / V_pore  # S ∈ [0, 1]
            else:
                solver.saturation[new_idx] = 0.0
        
        # Record new particle weight for incremental sampler update
        # Note: a_tot changed, so sampler will be rebuilt anyway
        # But we still record for consistency
        self._record_weight_change(new_idx, dW)
    
    def _perform_nucleation_agglomeration(self, i: int, j: int) -> int:
        """
        Perform agglomeration between particles i and j for nucleation purposes.
        
        This calls the solver's real agglomeration logic to maintain
        physical consistency (liquid volume transfer, weight updates, etc.).
        
        Args:
            i: Index of first particle
            j: Index of second particle
            
        Returns:
            Index of the child particle after agglomeration
        """
        # Save current state
        solver = self.solver
        
        if i >= solver.a_tot or j >= solver.a_tot:
            return -1
        
        if i == j:
            # Self-agglomeration is now allowed (physically valid)
            pass
        
        # Mark that we're in nucleation mode (optional flag for logging)
        old_in_nucleation = getattr(solver, '_in_nucleation', False)
        solver._in_nucleation = True
        
        try:
            # Call the manual agglomeration method and return child index
            return self._manual_agglomerate_particles(i, j)
            
        finally:
            solver._in_nucleation = old_in_nucleation
    
    def _manual_agglomerate_particles(self, i: int, j: int) -> int:
        """
        Manually agglomerate two particles (bypassing standard propensity logic).
        
        This creates a merged particle from i and j, preserving liquid volume
        and maintaining mass conservation.
        
        IMPORTANT: Returns the actual index of the child particle, which may
        differ from (a_tot - 1) if a parent was removed via swap-remove!
        
        VOLUME SEMANTICS:
        - V_flat[-1, :] stores V_dry (= V_solid + V_pore) - the total dry particle volume
        - V_flat[:dim, :] stores V_solid components (for multi-component systems)
        - For dim=1 (monodisperse): V_flat[:dim, :] = V_solid (redundant with V_flat[-1]*(1-poro))
        - V_solid is ALWAYS computed as: V_dry * (1 - porosity)
        
        Args:
            i: Index of first particle
            j: Index of second particle
            
        Returns:
            Index of the child particle after agglomeration
        """
        solver = self.solver
        
        if i >= solver.a_tot or j >= solver.a_tot:
            return -1
        
        # Get weights
        Wi = float(solver.W[i])
        Wj = float(solver.W[j])
        
        # Get liquid volumes
        liq_i = float(solver.liquid_volume[i]) if hasattr(solver, 'liquid_volume') else 0.0
        liq_j = float(solver.liquid_volume[j]) if hasattr(solver, 'liquid_volume') else 0.0
        
        # Get dry volumes from V_flat[-1] (this is the PRIMARY storage location)
        V_dry_i = float(solver.V_flat[-1, i])
        V_dry_j = float(solver.V_flat[-1, j])
        
        # Get porosities
        poro_i = solver.porosity[i] if hasattr(solver, 'porosity') else np.nan
        poro_j = solver.porosity[j] if hasattr(solver, 'porosity') else np.nan
        
        # Compute solid volumes from V_dry and porosity
        # For Vollkörper (NaN porosity): V_solid = V_dry
        # For porous particles: V_solid = V_dry * (1 - poro)
        V_solid_i = V_dry_i * (1.0 - poro_i) if not np.isnan(poro_i) else V_dry_i
        V_solid_j = V_dry_j * (1.0 - poro_j) if not np.isnan(poro_j) else V_dry_j
        
        # DSMC-compliant weight handling:
        # dW = amount that actually merges (not sum of both weights!)
        # For i==j: we merge the particle with itself, consuming 2*dW
        dW = min(Wi, Wj)
        
        # Get porosity growth kernel (create default if not exists)
        porosity_kernel = None
        if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
            porosity_kernel = solver.kernel_manager.porosity_growth_kernel
        
        if porosity_kernel is None:
            # Create default volume_mixing kernel
            from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
            porosity_kernel = get_porosity_growth_kernel('volume_mixing')
        
        # Estimate collision energy for kernel (simplified)
        rho = 1000.0  # kg/m³
        d_eff = float(solver.X[i] + solver.X[j]) * 0.5
        g = float(getattr(solver, 'G', 1000.0))
        v_rel = g * d_eff
        v_particle = float(V_dry_i + V_dry_j)
        m = rho * v_particle
        E_coll = 0.5 * m * v_rel ** 2
        
        # Compute merged porosity via kernel (CONSISTENT PHYSICS!)
        V_dry_merged, poro_merged = porosity_kernel.compute_merged_porosity(
            v_dry1=V_dry_i, poro1=poro_i,
            v_dry2=V_dry_j, poro2=poro_j,
            v_liq1=liq_i, v_liq2=liq_j,
            sat1=solver.saturation[i] if not np.isnan(poro_i) else 0.0,
            sat2=solver.saturation[j] if not np.isnan(poro_j) else 0.0,
            collision_energy=E_coll,
            solver=solver
        )
        
        # Compute V_solid_merged from V_dry_merged and poro_merged
        if np.isnan(poro_merged):
            V_solid_merged = V_dry_merged
        else:
            V_solid_merged = V_dry_merged * (1.0 - poro_merged)
        
        # ==========================================
        # Compute pore volumes for liquid handling
        # ==========================================
        V_pore_i = V_dry_i * poro_i if not np.isnan(poro_i) else 0.0
        V_pore_j = V_dry_j * poro_j if not np.isnan(poro_j) else 0.0
        V_pore_merged = V_dry_merged * poro_merged if not np.isnan(poro_merged) else 0.0
        
        # Prepare solid volumes for _append_particle_column
        # For dim=1: just use the scalar V_solid_merged
        # For dim>1: would need to scale proportionally (not implemented here)
        if solver.dim == 1:
            V_merged_solid = np.array([V_solid_merged])
        else:
            # Multi-component: copy from V_flat[:dim] and scale
            V_merged_solid = solver.V_flat[:solver.dim, i].copy() + solver.V_flat[:solver.dim, j].copy()
        
        sat_merged = None
        
        # Compute merged saturation via internal/external liquid handling
        if hasattr(solver, 'saturation') and poro_merged is not None and not np.isnan(poro_merged):
            sat_i = solver.saturation[i] if not np.isnan(poro_i) else 0.0
            sat_j = solver.saturation[j] if not np.isnan(poro_j) else 0.0
            
            # Internal liquid of parents (limited by pore capacity * saturation)
            V_liq_int_i = min(liq_i, V_pore_i * sat_i) if V_pore_i > 0 else 0.0
            V_liq_int_j = min(liq_j, V_pore_j * sat_j) if V_pore_j > 0 else 0.0
            
            # External liquid of parents (total - internal)
            V_liq_ext_i = liq_i - V_liq_int_i
            V_liq_ext_j = liq_j - V_liq_int_j
            
            # Merge: add internal liquids
            V_liq_int_merged = V_liq_int_i + V_liq_int_j
            V_liq_ext_merged = V_liq_ext_i + V_liq_ext_j
            
            # Compute new saturation from merged internal liquid
            if V_pore_merged > 0:
                sat_merged = V_liq_int_merged / V_pore_merged
            else:
                sat_merged = 0.0
            
            # If saturation exceeds 1.0, externalize the excess
            if sat_merged > 1.0:
                excess_internal = V_pore_merged * (sat_merged - 1.0)
                V_liq_int_merged = V_pore_merged  # Fully saturated
                V_liq_ext_merged += excess_internal  # Add excess to external
                sat_merged = 1.0  # Cap at 1.0
            
            # Total merged liquid
            liq_merged = V_liq_int_merged + V_liq_ext_merged
        else:
            liq_merged = liq_i + liq_j
        
        # Append new particle (pass only solid volumes)
        solver._append_particle_column(V_merged_solid)
        new_idx = solver.a_tot - 1
        
        # Overwrite V_flat[-1] with dry volume (includes pores)
        solver.V_flat[-1, new_idx] = V_dry_merged
        solver.W[new_idx] = dW  # DSMC: child gets dW, not Wi+Wj!
        
        if hasattr(solver, 'liquid_volume'):
            solver.liquid_volume[new_idx] = liq_merged
        
        if poro_merged is not None and hasattr(solver, 'porosity'):
            solver.porosity[new_idx] = poro_merged
        
        if sat_merged is not None and hasattr(solver, 'saturation'):
            solver.saturation[new_idx] = sat_merged
        
        # Record new particle weight for incremental sampler update
        self._record_weight_change(new_idx, dW)
        
        # CRITICAL: Track child index through parent removals!
        # When _remove_particle_column swaps the child with another index,
        # we need to update new_idx accordingly.
        child_current_idx = new_idx
        
        # Reduce parent weights (DSMC-compliant: don't delete unless W <= 0)
        if i == j:
            # Self-agglomeration: consume 2*dW from same particle
            # This represents the particle merging with itself (weight reduction)
            solver.W[i] -= 2.0 * dW
            
            # Record weight change for incremental sampler update
            self._record_weight_change(i, solver.W[i])
            
            if solver.W[i] <= 0:
                # If removing i which is also the child location, we have a problem
                if i == child_current_idx:
                    # Child would be removed - this shouldn't happen in normal operation
                    # Return -1 to signal error
                    return -1
                # CRITICAL: Save 'last' BEFORE remove, because a_tot changes!
                last_before_remove = solver.a_tot - 1
                solver._remove_particle_column(i)
                # After swap-remove: if child was at 'last' position, it moved to 'i'
                if child_current_idx == last_before_remove:
                    child_current_idx = i
        else:
            # Regular agglomeration: consume dW from each parent
            solver.W[i] -= dW
            solver.W[j] -= dW
            
            # Record weight changes for incremental sampler update
            self._record_weight_change(i, solver.W[i])
            self._record_weight_change(j, solver.W[j])
            
            # Remove parents only if weight exhausted (order: larger index first)
            # CRITICAL: Track child index through each removal!
            for idx in sorted([i, j], reverse=True):
                if idx >= solver.a_tot:
                    continue
                if solver.W[idx] <= 0:
                    child_was_at = child_current_idx
                    # CRITICAL: Save 'last' BEFORE remove, because a_tot changes!
                    last_before_remove = solver.a_tot - 1
                    solver._remove_particle_column(idx)
                    # After swap-remove: if child was at 'last' position, it moved to 'idx'
                    if child_was_at == last_before_remove:
                        child_current_idx = idx
                    elif child_was_at == idx:
                        return -1  # Child got removed - error
        
        # IMPORTANT: Do NOT rebuild main agglomeration propensities (_r_agg) here!
        # The main MC loop will rebuild _r_agg and _agg_sampler at the next MC event.
        # Rebuilding here would be O(n²) and is called for EVERY droplet distribution,
        # which makes manual triggering extremely slow for many droplets.
        #
        # The nucleation weight-based sampler ONLY depends on W values,
        # not on agglomeration propensities. It needs to be rebuilt because W changed.
        # PERFORMANCE: With incremental updates, this is now O(k log n) instead of O(k n)
        # where k = number of droplets distributed
        self._ensure_samplers()
        
        return child_current_idx  # Return actual child index!
    
    def get_current_wt_percent(self) -> float:
        """
        Calculate current liquid/solid weight percent (dry basis).
        
        wt% = (mass_liquid / mass_solid) × 100
            = (V_liquid × rho_liquid) / (V_solid × rho_solid) × 100
        
        Returns:
            Current wt% value, or 0.0 if no particles exist or densities not available
        """
        solver = self.solver
        n_active = solver.a_tot
        
        if n_active <= 0:
            return 0.0
        
        # Check if densities are available (required for wt% calculation)
        if self.config.rho_solid is None or self.config.rho_liquid is None:
            return 0.0
        
        # Calculate weighted total volumes (DSMC-consistent)
        # IMPORTANT: liquid_volume is INTENSIVE (per physical particle).
        # Total liquid = Sum(liquid_volume[k] × W[k]) - NO division by Vc!
        # Same for solid volume: V_solid is per particle, total = Sum(V_solid[k] × W[k])
        V_solid_total = 0.0
        V_liquid_total = 0.0
        
        for k in range(n_active):
            W_k = float(solver.W[k])
            V_solid_k = float(np.sum(solver.V_flat[:solver.dim, k]))
            V_liquid_k = float(solver.liquid_volume[k]) if hasattr(solver, 'liquid_volume') else 0.0
            
            # NO division by Vc - these are extensive quantities (total volume of ensemble)
            V_solid_total += V_solid_k * W_k
            V_liquid_total += V_liquid_k * W_k
        
        # Convert to masses
        mass_solid = V_solid_total * self.config.rho_solid
        mass_liquid = V_liquid_total * self.config.rho_liquid
        
        if mass_solid <= 0:
            return 0.0
        
        # Dry basis: liquid / solid
        wt_percent = (mass_liquid / mass_solid) * 100.0
        return wt_percent
    
    def _should_check_wt(self, current_time: float) -> bool:
        """
        Determine if wt% should be checked now using adaptive timing.
        
        Strategy:
        - First check: immediately
        - If no liquid added yet: check frequently (every 10ms)
        - If approaching target: extrapolate with safety factor (50%)
        - Caps: min 10ms, max 500ms between checks
        
        Args:
            current_time: Current simulation time [s]
            
        Returns:
            True if wt% should be checked now
        """
        # First check: always
        if self._last_wt_check_time < 0:
            return True
        
        # Time since last check
        dt_since_last = current_time - self._last_wt_check_time
        
        # If no previous wt% value, use fixed interval
        if self._last_wt_value is None:
            return dt_since_last >= self._wt_check_interval
        
        # Calculate rate of change
        current_wt = self.get_current_wt_percent()
        dw = current_wt - self._last_wt_value
        
        if dw <= 0 or dt_since_last <= 0:
            # No progress: check soon
            return dt_since_last >= 0.01
        
        # Rate of wt% increase
        rate = dw / dt_since_last  # wt% per second
        
        # Time to reach target at current rate
        remaining = self.config.target_wt_percent - current_wt
        
        if remaining <= 0:
            # Already at or above target: check immediately
            return True
        
        time_to_target = remaining / rate
        
        # SAFETY FACTOR: Check at 50% of predicted time (conservative!)
        # Better to check too early than overshoot!
        safety_factor = 0.5
        next_check_delay = time_to_target * safety_factor
        
        # Cap intervals
        max_interval = 0.5  # Max 500ms
        min_interval = 0.01  # Min 10ms
        
        next_check_delay = np.clip(next_check_delay, min_interval, max_interval)
        
        return dt_since_last >= next_check_delay
    
    def _update_wt_check(self, current_time: float, current_wt: float) -> None:
        """
        Update internal state after wt% check.
        
        Call this after has_reached_target_wt() to update prediction.
        
        Args:
            current_time: Current simulation time [s]
            current_wt: Current wt% value
        """
        self._last_wt_check_time = current_time
        self._last_wt_value = current_wt
        
        # Adjust interval based on proximity to target
        if self.config.target_wt_percent is not None:
            fraction = current_wt / self.config.target_wt_percent
            if fraction > 0.8:
                # Close to target: check more often
                self._wt_check_interval = 0.01
            elif fraction > 0.5:
                self._wt_check_interval = 0.05
            else:
                # Far from target: can wait longer
                self._wt_check_interval = 0.1
    
    def has_reached_target_wt(self) -> bool:
        """
        Check if target wt% has been reached (if configured).
        
        Returns:
            True if target_wt_percent is set and reached, False otherwise
        """
        if self.config.target_wt_percent is None:
            return False
        
        current_wt = self.get_current_wt_percent()
        
        # Update internal state for adaptive checking
        self._update_wt_check(self._current_time, current_wt)
        
        return current_wt >= self.config.target_wt_percent
    
    def get_statistics(self) -> dict:
        """
        Get nucleation statistics.
        
        Returns:
            Dictionary with nucleation statistics
        """
        stats = {
            'current_time': self._current_time,
            'next_nucleation_time': self._next_nucleation_time,
            'droplets_added_total': self._droplets_added_total,
            'liquid_volume_added_total': self._liquid_volume_added_total,
            'current_wt_percent': self.get_current_wt_percent(),
            'target_wt_percent': self.config.target_wt_percent,
            'target_reached': self.has_reached_target_wt(),
            'liquid_addition_start': self.config.liquid_addition_start,
            'liquid_addition_duration': self.config.liquid_addition_duration,
            'liquid_addition_end': self.config.liquid_addition_end,
            'in_addition_window': self._in_addition_window(self._current_time),
            'initialization_type': self.config.initialization_type.value if self.config.initialization_type else None,
        }
        
        # Optional mass-based parameters (if configured)
        if self.config.solid_mass_in_mixer is not None:
            stats['solid_mass_in_mixer'] = self.config.solid_mass_in_mixer
            stats['n_particles_real'] = self._n_particles_real
            stats['volumetric_flow_rate_scaled'] = self._flow_rate_scaled
        
        return stats
    
    def reset(self) -> None:
        """Reset nucleation state for repeated simulations."""
        self._current_time = 0.0
        self._next_nucleation_time = self.config.liquid_addition_start
        self._Vc_reference = self.solver.Vc
        self._droplets_added_total = 0
        self._liquid_volume_added_total = 0.0
        
        # Reset window tracking
        self._mc_events_in_window = False
        self._nucleation_triggered_at_window_end = False
        self._was_in_window = False
        self._first_event_after_window = True
        
        # Reset adaptive wt% checking
        self._last_wt_check_time = -1.0
        self._last_wt_value = None
        self._wt_check_interval = 0.01
        
        # Reset sampler
        self._weight_sampler = None
        self._weight_array = None
        
        # Reset debug tracking
        self._debug_last_print_time = -1.0
    
    def print_debug_status(self, force: bool = False) -> None:
        """
        Print debug status of nucleation process.
        
        Args:
            force: If True, print regardless of time interval
        """
        if not self.config.enabled:
            return
        
        current_time = self._current_time
        
        # Check if we should print (based on interval or force)
        if not force:
            if current_time - self._debug_last_print_time < self._debug_print_interval:
                return
        
        self._debug_last_print_time = current_time
        
        # Gather statistics
        solver = self.solver
        config = self.config
        
        # Time elapsed
        t_elapsed = current_time
        
        # Percentage of t_total
        t_total = float(getattr(solver, 't_vec', [1.0])[-1])
        t_percent = (t_elapsed / t_total * 100) if t_total > 0 else 0.0
        
        # Real time elapsed (wall clock)
        real_time_elapsed = getattr(solver, 'MACHINE_TIME', 0.0) if hasattr(solver, 'MACHINE_TIME') else 0.0
        
        # Droplets added
        n_droplets = self._droplets_added_total
        
        # Percentage of expected amount
        if config.volumetric_flow_rate > 0 and t_elapsed >= config.liquid_addition_start:
            t_in_window = min(max(0.0, t_elapsed - config.liquid_addition_start), config.liquid_addition_duration)
            v_expected = config.volumetric_flow_rate * t_in_window
            v_actual = self._liquid_volume_added_total
            percent_of_expected = (v_actual / v_expected * 100) if v_expected > 0 else 0.0
        else:
            percent_of_expected = 0.0
        
        # Computational and physical particle counts
        n_comp = solver.a_tot
        sum_W = np.sum(solver.W[:n_comp]) if n_comp > 0 else 0.0
        n_phys = sum_W / solver.Vc
        
        # Print formatted status
        print(
            f"\n[NUCLEATION DEBUG] t={t_elapsed:.4f}s ({t_percent:.1f}% of total) | "
            f"Real time: {real_time_elapsed:.2f}s | "
            f"Droplets: {n_droplets:.2e} ({percent_of_expected:.1f}% of expected) | "
            f"Particles: n_comp={n_comp}, n_phys={n_phys:.2e}"
        )


# Convenience function for creating nucleation handler
def create_nucleation_handler(solver, **kwargs) -> NucleationHandler:
    """
    Create and configure a nucleation handler.
    
    Args:
        solver: MCPBESolver instance
        **kwargs: Arguments passed to NucleationConfig
        
    Returns:
        Configured NucleationHandler instance
    """
    config = NucleationConfig(**kwargs)
    handler = NucleationHandler(solver, config)
    return handler
