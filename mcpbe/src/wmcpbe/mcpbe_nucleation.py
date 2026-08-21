"""Liquid addition: distributing binder droplets onto particles.

Physical model
--------------
Droplets hit particles at random. Every *physical* particle is equally likely
to be hit, which means sampling the *computational* particles weighted by
``W``.

DSMC scaling: one computational droplet event stands for
``physical_droplets = (Vc_ref / Vc) * W_selected`` physical droplets, so the
physical flow rate stays correct even as ``Vc`` doubles during a run.

Structure
---------
``NucleationConfig``
    Configuration dataclass; validates itself in ``__post_init__``.
``NucleationHandler``
    The handler itself. Composed onto the solver, not mixed in -- it runs on
    its own time scale via ``.step(t, dt)`` after each MC event, rather than as
    a drawn event.

Usage
-----
    >>> handler = solver.create_nucleation_handler(
    ...     enabled=True,
    ...     volumetric_flow_rate=1e-9,      # m^3/s
    ...     droplet_diameter=1e-6,          # m
    ...     liquid_addition_duration=300.0, # s
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .fenwick_new import FenwickSampler


class InitializationType(Enum):
    """Deprecated marker for how the addition window was specified.

    Nucleation is time-based either way; this only records which of the two
    input forms the user chose. Nothing in the solver branches on it -- it is
    set in ``NucleationConfig.__post_init__`` and read back out only in
    ``NucleationHandler.get_statistics()``.
    """
    DURATION = "duration"
    MASS = "mass"


@dataclass
class NucleationConfig:
    """Configuration for time-based liquid addition.

    The addition window can be given in two mutually exclusive ways:

    A. directly, as ``liquid_addition_duration`` [s]
    B. derived from ``target_wt_percent`` together with
       ``solid_mass_in_mixer``, ``rho_solid`` and ``rho_liquid``

    Giving both is allowed only if they agree within ``consistency_tol``.

    Parameters
    ----------
    enabled : bool
        Turn nucleation on.
    volumetric_flow_rate : float
        Physical flow rate [m^3/s], used as such -- the DSMC correction for a
        changing ``Vc`` happens in the handler.
    droplet_diameter : float
        Physical droplet diameter [m].
    liquid_addition_start : float, optional
        Start time [s], default 0.0.
    liquid_addition_duration : float, optional
        Window length [s] (option A).
    target_wt_percent : float, optional
        Target liquid fraction on a dry basis [%] (option B).
    solid_mass_in_mixer : float, optional
        Total solid mass in the mixer [kg] (option B).
    rho_solid, rho_liquid : float, optional
        Densities [kg/m^3] (option B).
    batch_size : float, optional
        Computational weight per droplet event. ``dW = min(batch_size, W_i)``
        sets how coarsely the liquid is distributed. Required when enabled.
    consistency_tol : float
        Relative tolerance of the A-vs-B check, default 0.001 (0.1 %).
    similarity_tol : float
        Relative tolerance for reusing an existing particle, default 1e-5.
    liquid_match_scale : str
        What the liquid part of that tolerance is measured against --
        ``'droplet'`` (default) or ``'total'``. See the comment at the field.
    debug : bool
        Print diagnostics, default False.

    Raises
    ------
    ValueError
        If ``volumetric_flow_rate`` or ``droplet_diameter`` is <= 0, if neither
        duration nor mass parameters are given, or if both forms are given and
        disagree by more than ``consistency_tol``.

    See Also
    --------
    NucleationHandler : the handler that consumes this configuration

    Examples
    --------
    >>> config = NucleationConfig(
    ...     enabled=True,
    ...     volumetric_flow_rate=1e-9,
    ...     droplet_diameter=1e-6,
    ...     liquid_addition_duration=300.0,
    ... )
    >>> handler = solver.create_nucleation_handler(**config.__dict__)
    """
    enabled: bool = False
    volumetric_flow_rate: float = 0.0  # m^3/s
    droplet_diameter: float = 0.0  # m
    liquid_addition_start: float = 0.0  # s
    liquid_addition_duration: Optional[float] = None  # s
    target_wt_percent: Optional[float] = None  # %
    solid_mass_in_mixer: Optional[float] = None  # kg
    rho_solid: Optional[float] = None  # kg/m^3
    rho_liquid: Optional[float] = None  # kg/m^3
    
    # Deprecated (backward compatibility)
    initialization_type: Optional[InitializationType] = None
    n_comp: Optional[int] = None
    
    # Batch size for droplet distribution (computational weight per event)
    # Required when enabled=True. No default value - must be set explicitly.
    batch_size: Optional[float] = None
    
    # Consistency check tolerance (relative)
    consistency_tol: float = 0.001  # 0.1%

    # Relative tolerance used when looking for an existing particle with the
    # same post-droplet state. Passed to ParticleMerger.find_or_create() as
    # tol_rel_override -- nucleation, agglomeration and breakage all share the
    # same merger, they only differ in how tight they want the match.
    similarity_tol: float = 0.00001

    # What the liquid part of that tolerance is measured against.
    #
    #   "total"   - relative to the particle's own liquid, i.e. the plain rule
    #               |target - candidate| / target <= similarity_tol.
    #   "droplet" (default) - the same rule, but additionally capped at
    #               similarity_tol * v_droplet. The cap can only tighten the
    #               tolerance, never loosen it.
    #
    # Why the cap: under "total" the accepted absolute liquid difference grows
    # with how much liquid a particle already carries, so late merges accept
    # ever larger differences while the statistics still book a whole droplet.
    # The mechanism is plausible; its magnitude has NOT been measured here.
    # "droplet" is therefore the conservative default. To trade that for a
    # higher merge hit rate, switch to "total" and compare
    # `python -m tests.test_conservation`.
    liquid_match_scale: str = "droplet"

    # Which lookup path the ParticleMerger uses for nucleation.
    #   True (default) - vectorised linear scan over all active particles.
    #           Also deterministic: it returns the lowest matching index,
    #           whereas the hash path iterates a set.
    #   False - hash index.
    #
    # Why not the hash, despite being "O(1)": its key bins V_dry and liquid to
    # 8 log10 digits and porosity/saturation to 4 decimals. In nucleation all
    # particles start monodisperse and dry, so they all land in ONE bucket.
    # _find_via_hash walks that bucket in a PYTHON loop, calling _matches_exact
    # per candidate -- with ~n entries per bucket and a hit rate of a few
    # percent, almost every lookup interprets the whole bucket. The linear scan
    # runs the same test vectorised in NumPy and is orders of magnitude faster
    # despite the identical formal complexity. Measured: with the hash a 20 s
    # run stalled around t~4; with the linear scan it completes in ~50 s.
    merge_linear_scan: bool = True

    # Diagnostic output. The handler used to print progress unconditionally,
    # which polluted stdout of every production run (and cost measurable time
    # in tight loops). Set debug=True to restore that output.
    debug: bool = False
    
    def __post_init__(self) -> None:
        if self.liquid_match_scale not in ("total", "droplet"):
            raise ValueError(
                "liquid_match_scale must be 'total' or 'droplet', "
                f"got {self.liquid_match_scale!r}"
            )
        if self.similarity_tol < 0.0:
            raise ValueError("similarity_tol must be non-negative")

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
        """Volume of a single spherical droplet [m^3]."""
        radius = self.droplet_diameter / 2.0
        return (4.0 / 3.0) * np.pi * radius ** 3


class NucleationHandler:
    """Adds binder liquid to the population as discrete droplets.

    Composed onto the solver rather than mixed in: ``solve()`` calls
    ``.step(t, dt)`` after every MC event, so nucleation runs on its own,
    continuous time scale instead of being a drawn event.

    Per step
    --------
    1. Intersect the event interval with the addition window.
    2. Liquid to add = ``flow_rate * dt_overlap``.
    3. Pick a particle weighted by ``W`` (Fenwick sampler, O(log n)) -- which
       samples *physical* particles uniformly.
    4. Add the droplet, update porosity and saturation through the kernels.
    5. Carry the leftover volume over to the next step, so integer
       discretisation of the droplet count cannot lose liquid.

    DSMC scaling: one computational droplet event stands for
    ``physical_droplets = (Vc_ref / Vc) * W_selected``, which keeps the
    physical flow rate [m^3/s] constant across control-volume doublings.

    Attributes
    ----------
    solver : MCPBESolver
        The owning solver.
    config : NucleationConfig
        Configuration.
    _current_time : float
        Current simulated time [s].
    _Vc_reference : float
        Control volume at start, the reference for the DSMC scaling [m^3].
    _droplets_added_total : float
        Physical droplets added so far (weighted sum).
    _liquid_volume_added_total : float
        Liquid volume added so far [m^3].
    _liquid_remainder : float
        Carried-over volume that has not yet filled a whole droplet [m^3].

    See Also
    --------
    NucleationConfig : configuration dataclass
    fenwick_new.FenwickSampler : weighted sampling

    Examples
    --------
    >>> handler = solver.create_nucleation_handler(
    ...     enabled=True, volumetric_flow_rate=1e-9,
    ...     droplet_diameter=1e-6, liquid_addition_duration=300.0)
    >>> solver.solve()
    """
    
    def __init__(self, solver, config: NucleationConfig) -> None:
        """Attach the handler to a solver.

        Parameters
        ----------
        solver : MCPBESolver
            The owning solver.
        config : NucleationConfig
            Flow rate, droplet size and time window.

        Raises
        ------
        ValueError
            If required solver attributes (``x``, ``Vc``) are not set.

        See Also
        --------
        NucleationConfig : the configuration
        MCPBESolver.create_nucleation_handler : factory that calls this
        """
        self.solver = solver
        self.config = config
        
        # Internal state
        self._current_time = 0.0
        
        # Statistics with Vc normalization for DSMC consistency
        self._Vc_reference = solver.Vc
        self._droplets_added_total = 0
        self._liquid_volume_added_total = 0.0
        
        # Debug tracking (from current debugging session)
        self._debug_last_print_time = -1.0
        self._debug_print_interval = 0.1
        
        # Particle count from solid_mass (statistics/validation only)
        self._n_particles_real = None
        # Note: Use config.volumetric_flow_rate directly, no scaling needed
        # DSMC scaling happens automatically via weight-based sampler
        
        if config.solid_mass_in_mixer is not None:
            self._initialize_mass_mode()
        
        # MINIMAL STATE for window tracking (5 variables - added _next_nucleation_time for stats)
        self._last_event_time = 0.0           # Time of last MC event (for finalize)
        self._had_events_in_window = False    # Track if any event had overlap with window
        self._manual_trigger_done = False     # Prevent double manual trigger
        self._liquid_remainder = 0.0          # Accumulated remainder for mass conservation
        self._next_nucleation_time = self.config.liquid_addition_start  # For statistics/reset()
        
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

        # Signals to the solve() loop that this handler changed the population
        # (new particle appended and/or weights moved) and the AGGLOMERATION /
        # BREAKAGE propensities are therefore stale. The sampler maintained by
        # this class (`_weight_sampler`) only serves the droplet target draw --
        # it says nothing about `_r_agg` / `_break_rate`.
        self._population_changed = False
    
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
    
    def mark_event_for_statistics(self, current_time: float) -> None:
        """Record the time of the last MC event, for :meth:`finalize_after_solve`.

        Called by the solver after every event. The ``_had_events_in_window``
        flag is not set here but in :meth:`step`, once ``dt_overlap > 0``.

        Parameters
        ----------
        current_time : float
            Current simulated time [s].
        """
        self._last_event_time = current_time
    
    def check_first_event(self, current_time: float) -> None:
        """Catch an addition window that closed before the first MC event.

        Called on the first event (``count == 0``). If the whole window already
        lies in the past, no ``step()`` will ever see it, so the liquid is
        added in one go instead.

        Parameters
        ----------
        current_time : float
            Time of the first MC event [s].

        See Also
        --------
        _trigger_manual_full_window : the catch-up path
        """
        if not self.config.enabled:
            return
        
        # If first event is after window end, trigger manual nucleation
        if current_time >= self.config.liquid_addition_end:
            self._trigger_manual_full_window()
            self._manual_trigger_done = True
    
    def finalize_after_solve(self, final_time: float, event_count: int) -> None:
        """Add whatever liquid the event loop did not get to.

        Three cases:

        1. No MC events at all -- add the full window in one go.
        2. The window closed after the last event -- add the missing span.
        3. Liquid left in ``_liquid_remainder`` -- always placed.

        Parameters
        ----------
        final_time : float
            Final simulated time [s].
        event_count : int
            Number of MC events that occurred.

        See Also
        --------
        _trigger_manual_full_window : case 1
        _distribute_remaining_liquid : case 3
        """
        if not self.config.enabled:
            return
        
        window_end = self.config.liquid_addition_end
        
        # Case 1: No events occurred AND window has passed -> Manual Trigger
        if event_count == 0 and final_time >= window_end:
            if not self._manual_trigger_done:
                self._trigger_manual_full_window()
                self._manual_trigger_done = True
            # After manual trigger, still distribute remainder below
        
        # Case 2: Window ended AFTER last event but we had events in window
        # This captures the time slice [_last_event_time, window_end] that was missed
        elif (self._had_events_in_window and 
              not self._manual_trigger_done and
              self._last_event_time < window_end <= final_time):
            
            dt_remaining = window_end - self._last_event_time
            v_remaining = self.config.volumetric_flow_rate * dt_remaining
            
            self._log(f"Processing remaining window time: last_event={self._last_event_time:.6f}s, "
                      f"window_end={window_end:.6f}s, dt={dt_remaining:.6f}s, v={v_remaining:.6e} m^3", "DEBUG")
            
            if v_remaining > 0:
                self._ensure_samplers()
                self._distribute_liquid_volume(v_remaining, is_manual_trigger=False)
        
        # Case 3: Always distribute accumulated remainder
        if self._liquid_remainder > 0:
            self._log(f"Distributing remainder: {self._liquid_remainder:.6e} m^3", "DEBUG")
            self._distribute_remaining_liquid(final_time)
            self._log(f"Remainder after distribution: {self._liquid_remainder:.6e} m^3", "DEBUG")
        
        # Print final statistics
        self.print_debug_status(force=True)
        self._log("Simulation complete", "INFO")
    
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
            self._log(f"Keeping {self._liquid_remainder:.3e} m^3 ({n_droplets_exact:.6f} droplets) as numerical remainder", "DEBUG")
            return
        
        self._ensure_samplers()
        vc_scale = self._Vc_reference / self.solver.Vc
        
        # Case 1: Significant remainder (≥ 1 droplet) - distribute integer droplets
        # Case 2: Fractional remainder (0.1-1.0 droplets) - distribute exact volume
        if n_droplets_exact >= 1.0:
            n_droplets_full = int(n_droplets_exact)
            self._log(f"Distributing {n_droplets_full} remaining droplets (exact: {n_droplets_exact:.2f}) at t={final_time:.4f}s", "DEBUG")
            
            v_target = n_droplets_full * v_droplet
            v_distributed = 0.0
            max_consecutive_failures = max(1, n_droplets_full // 10 + 1)
            consecutive_failures = 0
            
            while v_distributed < v_target and consecutive_failures < max_consecutive_failures:
                v_remaining = v_target - v_distributed
                n_physical_remaining = v_remaining / v_droplet
                dW, v_eff = self._distribute_one_droplet_with_dW(
                    v_droplet, max_physical_droplets=n_physical_remaining)

                # Refresh the sampler per droplet; see the reasoning in
                # _distribute_liquid_volume.
                self._ensure_samplers()

                if dW > 0:
                    effective_dW = dW * vc_scale
                    # v_eff, not v_droplet: on a capped event each physical
                    # particle receives less than a full droplet, so booking
                    # v_droplet here would charge more than was delivered.
                    v_event = v_eff * effective_dW
                    v_distributed += v_event
                    self._droplets_added_total += v_event / v_droplet
                    self._liquid_volume_added_total += v_event
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

            self._liquid_remainder -= v_distributed
        
        else:
            # Fractional droplet: distribute exact remaining volume
            # Note: _liquid_remainder is PHYSICAL volume (already scaled by flow_rate_per_particle).
            # We distribute it as a single computational event with capped dW.
            self._log(f"Distributing fractional droplet ({n_droplets_exact:.2f} × {v_droplet:.3e} m^3) at t={final_time:.4f}s", "DEBUG")
            
            # CRITICAL: Pass ACTUAL droplet volume and cap max_physical_droplets!
            # This ensures dW is capped correctly: dW <= n_droplets_exact / vc_scale
            # Without this cap, dW would be W[i]/2.0 (~46.5) and the particle would
            # receive dW × v_droplet = 46.5 × 5.236e-13 = 2.43e-11 m^3 (83x too much!)
            dW, v_eff = self._distribute_one_droplet_with_dW(
                v_droplet=v_droplet,  # Actual droplet volume for saturation calc
                max_physical_droplets=n_droplets_exact  # Cap: only 0.56 physical droplets!
            )

            if dW > 0:
                effective_dW = dW * vc_scale
                # Capping shrinks the amount PER PARTICLE (v_eff), not dW.
                # The liquid actually delivered is therefore
                # effective_dW * v_eff -- booking the nominal v_droplet
                # instead drives _liquid_remainder negative.
                v_physical_distributed = effective_dW * v_eff

                # Statistics count NOMINAL droplets, consistent with the volume.
                self._droplets_added_total += v_physical_distributed / v_droplet
                self._liquid_volume_added_total += v_physical_distributed
                self._liquid_remainder -= v_physical_distributed
                
                self._log(f"Distributed fractional droplet: {v_physical_distributed:.3e} m^3 (dW={dW:.4f}, effective_dW={effective_dW:.4f})", "DEBUG")
            else:
                self._log("WARNING: Failed to distribute fractional droplet", "WARNING")
        
        # Sanity check
        if self._liquid_remainder < -1e-15:
            raise RuntimeError(
                f"Nucleation: _liquid_remainder became negative after final distribution "
                f"({self._liquid_remainder:.6e})"
            )
    
    def step(self, current_time: float, solver_last_dt: float) -> None:
        """Add the liquid belonging to one MC event.

        Called after every event. Intersects the event interval
        ``[t - dt, t]`` with the addition window ``[start, end]`` and adds
        ``flow_rate * dt_overlap``. An event entirely outside the window adds
        nothing.

        Parameters
        ----------
        current_time : float
            Simulated time AFTER this MC event [s].
        solver_last_dt : float
            Time since the previous event, i.e. the interval length [s].

        See Also
        --------
        _distribute_liquid_volume : places the computed volume
        """
        if not self.config.enabled:
            return
        
        window_start = self.config.liquid_addition_start
        window_end = self.config.liquid_addition_end
        
        # Retroactive overlap calculation:
        # Event interval: [event_start, event_end]
        event_start = current_time - solver_last_dt
        event_end = current_time
        
        # Intersection with window
        overlap_start = max(event_start, window_start)
        overlap_end = min(event_end, window_end)
        dt_overlap = max(0.0, overlap_end - overlap_start)
        
        # Track that we had events in/near window (for Manual Trigger suppression)
        if dt_overlap > 0:
            self._had_events_in_window = True
        
        # Update last event time for finalize()
        self._last_event_time = current_time
        
        # Early return if no overlap
        if dt_overlap <= 0.0:
            return
        
        # Ensure sampler is ready
        self._ensure_samplers()
        
        # Calculate liquid volume for overlap duration
        # Note: Use config.volumetric_flow_rate directly (no scaling needed)
        v_liquid = self.config.volumetric_flow_rate * dt_overlap
        
        # Distribute liquid volume
        self._distribute_liquid_volume(v_liquid, is_manual_trigger=False)
        
        # Print debug status
        self.print_debug_status(force=False)
    
    def _trigger_manual_full_window(self) -> None:
        """Add the entire window's liquid in one pass.

        Used when the event loop cannot cover the window: either the first
        event already lies past its end (see :meth:`check_first_event`), or the
        simulation produced no events at all (see
        :meth:`finalize_after_solve`).

        Raises
        ------
        UserWarning
            Always -- a missed window means the event rate was far lower than
            the addition rate, which is worth knowing about.

        See Also
        --------
        check_first_event, finalize_after_solve : the two callers
        """
        import warnings
        warnings.warn(
            f"Nucleation window [{self.config.liquid_addition_start:.2f}s, {self.config.liquid_addition_end:.2f}s] "
            f"passed without MC events. Triggering manual nucleation with full duration. "
            f"Consider increasing liquid_addition_duration or CORR_BETA.",
            UserWarning,
            stacklevel=2
        )
        
        self._log(f"\n[MANUAL TRIGGER] Full window nucleation...")
        
        # Calculate total liquid for entire window duration
        v_total = self.config.volumetric_flow_rate * self.config.liquid_addition_duration
        
        if v_total <= 0.0:
            self._log(f"  [MANUAL TRIGGER] No liquid to add (v_total={v_total})")
            return
        
        self._log(f"  Duration: {self.config.liquid_addition_duration:.2f}s")
        self._log(f"  Flow rate: {self.config.volumetric_flow_rate:.6e} m^3/s")
        self._log(f"  Total volume: {v_total:.6e} m^3")
        self._log(f"  Current n_comp: {self.solver.a_tot}")
        
        # Distribute using standard logic
        self._distribute_liquid_volume(v_total, is_manual_trigger=True)
        
        self._log(f"  [MANUAL TRIGGER] Complete:")
        self._log(f"    Droplets added: {self._droplets_added_total:.2e}")
        self._log(f"    Volume added: {self._liquid_volume_added_total:.6e} m^3")
        
        # Distribute remainder immediately
        if self._liquid_remainder > 0:
            self._distribute_remaining_liquid(self.solver._elapsed)
        
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
        """Build or refresh the weighted sampler used to pick a target particle.

        To hit *physical* particles uniformly, *computational* particles have
        to be drawn proportional to their weight::

            P(particle i) = W[i] / sum(W)

        Weight changes are applied incrementally in O(log n); only a change in
        particle count forces a full O(n) rebuild.

        See Also
        --------
        fenwick_new.FenwickSampler : the sampler
        _apply_pending_weight_updates : the incremental path
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
        """Flush the recorded weight changes into the sampler.

        One update costs O(log n) instead of the O(n) of a full rebuild, so k
        updates over n particles cost O(k log n) rather than O(k n).

        Called automatically from :meth:`_ensure_samplers` when incremental
        updates are enabled.

        See Also
        --------
        _record_weight_change : records the changes this applies
        fenwick_new.FenwickSampler.update : the underlying operation
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
        """Note that a particle's weight changed, without touching the sampler yet.

        Called on every weight change during nucleation. The sampler update is
        deferred until the next :meth:`_ensure_samplers` call, so a droplet
        that changes several weights pays for one flush rather than several.

        Parameters
        ----------
        idx : int
            Index of the particle whose weight changed.
        new_weight : float
            The new absolute weight.

        See Also
        --------
        _apply_pending_weight_updates : applies what this records
        """
        # Unconditional: this is the single hook every weight change of this
        # handler passes through (droplet transfer, parent depletion, forced
        # agglomeration child). It is therefore the right place to mark the
        # agglomeration/breakage propensities as stale -- independently of
        # whether the handler's OWN sampler is updated incrementally or rebuilt.
        self._population_changed = True

        if self._use_incremental_updates:
            self._pending_weight_updates.append((idx, new_weight))

    def consume_population_changed(self) -> bool:
        """Report and reset whether the population changed since the last call.

        Used by :meth:`MCPBEBase.solve` to refresh the agglomeration and
        breakage propensities only when this handler actually moved something,
        instead of paying an O(n) rebuild on every Monte-Carlo event.
        """
        changed = self._population_changed
        self._population_changed = False
        return changed


    def _select_particle_uniform_physical(self) -> int:
        """Draw a computational particle so that all PHYSICAL ones are equally likely.

        Sampling proportional to ``W`` is what makes the distribution uniform
        over physical particles rather than over array slots.

        If a ``liquid_dist_kernel`` is configured, it takes over and can
        implement other strategies (surface-weighted, saturation-preferential).

        Returns
        -------
        int
            Index of the chosen particle, or ``-1`` if none is eligible.

        See Also
        --------
        kernels.liquid_distribution : the pluggable selection strategies
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
        
        # Sample weighted by W -> uniform over physical particles
        return int(self._weight_sampler.sample(self._rng))
        
    def _distribute_liquid_volume(self, v_liquid_to_distribute: float, 
                                   is_manual_trigger: bool = False) -> None:
        """Turn a liquid volume into droplets and place them on particles.

        The volume is added to ``_liquid_remainder`` first, and only whole
        droplets are placed; what is left stays in the remainder for the next
        call. Tracking *volume* rather than a droplet *count* is what keeps
        this mass-conserving -- rounding a count down each time would lose
        liquid systematically.

        Whatever remains at the very end is placed as one final partial
        droplet with an effective weight of 1.

        Parameters
        ----------
        v_liquid_to_distribute : float
            Physical liquid volume to add [m^3].
        is_manual_trigger : bool
            Print detailed progress (used by the manual trigger path).

        See Also
        --------
        _distribute_one_droplet_with_dW : places a single droplet
        _distribute_remaining_liquid : places the remainder at end of run
        """
        # Initialize remainder accumulator if needed
        if not hasattr(self, '_liquid_remainder'):
            self._liquid_remainder = 0.0
        
        # === DEBUG NUC: VOR VERTEILUNG ===
        if getattr(self.solver, 'mcpbe_debug_mass', False) and getattr(self.solver, 'mcpbe_debug_nuc', True):
            solver = self.solver
            v_dry = solver.V_flat[-1, :solver.a_tot]
            poro = solver.porosity[:solver.a_tot]
            w = solver.W[:solver.a_tot]
            valid = ~np.isnan(poro)
            v_solid = np.zeros_like(v_dry)
            v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
            v_solid[~valid] = v_dry[~valid]
            
            solid_before = np.sum(v_solid * w)
            liq_before = np.sum(solver.liquid_volume[:solver.a_tot] * w)
            n_phys_before = np.sum(w) / solver.Vc
            v_dry_total = np.sum(v_dry * w)

            print(f"\n[DEBUG NUC] t={self._current_time:.4f}s")
            print(f"  Distributing droplets: volume={v_liquid_to_distribute:.6e} m^3")
            print(f"  BEFORE: V_solid_total={solid_before:.6e}, V_liq_total={liq_before:.6e}")
            print(f"  n_phys={n_phys_before:.3e}, a_tot={solver.a_tot}")
# ============================
        
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
        
        # Distribute droplets with DSMC weight handling.
        # Track accumulated liquid VOLUME (not droplet count) to avoid floating-point
        # accumulation errors. Compute target volume from integer droplet count.
        v_target = n_droplets_full * v_droplet
        v_distributed = 0.0
        max_consecutive_failures = 1000
        consecutive_failures = 0
        
        # For manual trigger: track progress
        if is_manual_trigger and n_droplets_full > 100:
            self._log(f"Manual trigger: distributing {n_droplets_full} droplets", "INFO")
        
        while v_distributed < v_target and consecutive_failures < max_consecutive_failures:
            # Convert remaining volume to physical droplets for dW capping
            v_remaining = v_target - v_distributed
            n_physical_remaining = v_remaining / v_droplet
            
            dW, v_eff = self._distribute_one_droplet_with_dW(
                v_droplet, max_physical_droplets=n_physical_remaining)

            # Refresh the sampler after EVERY droplet. The droplet just took
            # weight off the particle it hit, possibly created a new particle
            # and possibly forced an agglomeration -- the next droplet has to
            # draw from the state AFTER all of that.
            #
            # Doing this once before the loop instead would leave the sampler
            # on stale weights from the second droplet onwards, and particles
            # created within this step would be unreachable for the rest of it.
            # Weighted drawing over ALL available particles is precisely what
            # `batch_size` is for: it is the droplet's W, analogous to a
            # particle's W. At batch_size = 1 one event is exactly one physical
            # droplet landing on one particle.
            self._ensure_samplers()

            if dW > 0:
                effective_dW = dW * vc_scale
                # v_eff rather than v_droplet -- see _distribute_remaining_liquid:
                # capped events deliver less than a full droplet.
                v_event = v_eff * effective_dW
                v_distributed += v_event
                self._droplets_added_total += v_event / v_droplet
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
                        self._log(f"Stopping after {consecutive_failures} consecutive failures", "WARNING")
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
        #
        # UNIFIED PATH: Same logic as _distribute_one_droplet_with_dW() for consistency!
        if self._liquid_remainder > 0 and self._liquid_remainder >= v_droplet * 0.001:
            solver = self.solver
            
            # Select one particle weighted by W (uniform over physical particles)
            i = self._select_particle_uniform_physical()
            
            if i >= 0 and i < solver.a_tot:
                # For effective_dW = 1, we need dW = 1 / vc_scale
                dW_for_one_physical = 1.0 / vc_scale
                
                # ==========================================
                # UNIFIED LOGIC: Same as _distribute_one_droplet_with_dW()
                # ==========================================
                current_liquid = float(solver.liquid_volume[i]) if hasattr(solver, 'liquid_volume') else 0.0
                new_liquid = current_liquid + self._liquid_remainder

                # Check if first contact (for Volume Mixing special case)
                is_first_contact = (current_liquid == 0.0)

                # Geometrie nach dem Tropfen: bestehende Porenstruktur bleibt
                # erhalten, nur echte Neu-Nukleation geht in den Kernel.
                v_dry_new, poro_new = self._resolve_nucleation_geometry(
                    i, new_liquid, is_first_contact
                )

                # Carry saturation forward (on every droplet). The remainder
                # `_liquid_remainder` arrives as purely external liquid.
                sat_new = self._compute_saturation_for_liquid(
                    src_idx=i,
                    v_dry=v_dry_new,
                    poro=poro_new,
                    v_liquid=new_liquid,
                    v_added=self._liquid_remainder,
                    is_first_contact=is_first_contact
                )
                
                # Zielzustand einbuchen (mergen oder neu anlegen) -- gleicher
                # Merger-Pfad wie im regulaeren Tropfenpfad. Frueher legte
                # dieser Restpfad IMMER ein neues Partikel an, ohne Dedup.
                self._place_nucleated_state(
                    src_idx=i,
                    dW=dW_for_one_physical,
                    v_dry=v_dry_new,
                    poro=poro_new,
                    liquid=new_liquid,
                    saturation=sat_new,
                    v_droplet=v_droplet if self.config.liquid_match_scale == "droplet" else None,
                )

                # Reduce parent weight by dW
                solver.W[i] -= dW_for_one_physical
                self._record_weight_change(i, solver.W[i])

                if solver.W[i] <= 0.0:
                    self._remove_particle_tracked(i)
                
                # Update statistics: count fractional droplets based on actual remainder volume
                # This avoids systematic overshoot when remainder < 1.0 droplet
                n_droplets_fractional = self._liquid_remainder / v_droplet
                self._droplets_added_total += n_droplets_fractional
                self._liquid_volume_added_total += self._liquid_remainder
                self._liquid_remainder = 0.0
        
        # === DEBUG NUC: NACH VERTEILUNG ===
        if getattr(self.solver, 'mcpbe_debug_mass', False) and getattr(self.solver, 'mcpbe_debug_nuc', True):
            solver = self.solver
            v_dry = solver.V_flat[-1, :solver.a_tot]
            poro = solver.porosity[:solver.a_tot]
            w = solver.W[:solver.a_tot]
            valid = ~np.isnan(poro)
            v_solid = np.zeros_like(v_dry)
            v_solid[valid] = v_dry[valid] * (1.0 - poro[valid])
            v_solid[~valid] = v_dry[~valid]
            
            solid_after = np.sum(v_solid * w)
            liq_after = np.sum(solver.liquid_volume[:solver.a_tot] * w)
            n_phys_after = np.sum(w) / solver.Vc
            v_dry_total_after = np.sum(v_dry * w)
            
            print(f"  AFTER:  V_solid={solid_after:.6e}, V_dry={v_dry_total_after:.6e}, V_liq={liq_after:.6e}")
            print(f"  ΔV_solid={solid_after - solid_before:.6e} (SOLLTE = 0 sein!)")
            print(f"  ΔV_dry={v_dry_total_after - v_dry_total:.6e} (SOLLTE = 0 sein!)")
            print(f"  ΔV_liq={liq_after - liq_before:.6e} (SOLLTE ≈ added volume)")
            print(f"  n_phys={n_phys_after:.3e}, a_tot={solver.a_tot}")
            print(f"  Statistics: droplets_added={self._droplets_added_total:.2e}, volume_added={self._liquid_volume_added_total:.6e}")
            
            # KRITISCHE WARNUNG wenn sich V_solid geaendert hat!
            if abs(solid_after - solid_before) > 1e-20:
                print(f"  *** FEHLER: V_SOLID HAT SICH GEAENDERT UM {solid_after - solid_before:.6e}! ***")
# ============================

        # Log any remaining liquid (should be zero or negligible now)
        if not is_manual_trigger and self._liquid_remainder > 1e-20:
            self._log(f"    [REMAINDER] {self._liquid_remainder:.3e} m^3 carried to next step ({self._liquid_remainder/v_droplet:.2f} droplets)")
    
    def _distribute_one_droplet_with_dW(self, v_droplet: float, 
                                         max_physical_droplets: Optional[float] = None) -> float:
        """Place one computational droplet, agglomerating first if necessary.

        The single path for every droplet -- the first and all later ones.

        One computational droplet event distributes a droplet of volume
        ``v_droplet``. The event weight ``dW`` decides how many *physical*
        droplets that represents::

            n_physical_droplets = effective_dW = dW * (Vc_ref / Vc)

        The target particle is drawn weighted by ``W``, which samples physical
        particles uniformly.

        Capacity criterion
        ------------------
        A particle can only carry the liquid if ``V_dry >= v_liquid``. Note
        that this uses **V_dry**, not ``V_solid``: a highly porous particle can
        hold more liquid, because ``V_dry = V_solid + V_pore`` includes the
        pore space.

        If the drawn particle cannot take the droplet, it is agglomerated with
        further particles until enough ``V_dry`` has been collected.

        Steps
        -----
        1. Draw particle ``i`` weighted by ``W``.
        2. ``new_liquid = current_liquid + v_droplet``.
        3. While ``V_dry < new_liquid``: agglomerate with a partner ``j`` and
           update ``V_dry`` and ``new_liquid`` from the merged particle.
        4. Derive ``dW`` from the collected weight ``W_i``.
        5. Porosity from the porosity-growth kernel.
        6. Saturation (updated on every droplet, never recomputed from
           scratch -- see :meth:`_compute_saturation_for_liquid`).
        7. Merge into a matching existing particle, or create a new one.
        8. Update parent weights and the samplers.

        Kernel-dependent behaviour: ``volume_mixing`` starts a first contact at
        ``poro=0.4`` (otherwise no porosity could ever form), ``cone_model``
        starts at ``poro=0.0`` and grows porosity geometrically on later
        agglomeration.

        Parameters
        ----------
        v_droplet : float
            Droplet volume [m^3].
        max_physical_droplets : float, optional
            Upper limit on physical droplets, capping ``dW`` such that
            ``effective_dW = dW * vc_scale <= max_physical_droplets``.

        Returns
        -------
        tuple[float, float]
            ``(dW, v_droplet_effective)`` -- the weight this event consumed and
            the droplet size it ACTUALLY delivered.

            When ``max_physical_droplets`` binds, it is not ``dW`` that shrinks
            but the amount per particle. The caller must therefore book the
            delivered liquid as ``dW * vc_scale * v_droplet_effective``; using
            the nominal ``v_droplet`` subtracts too much from
            ``_liquid_remainder`` and drives it negative. Without capping,
            ``v_droplet_effective == v_droplet``.

        See Also
        --------
        _perform_nucleation_agglomeration : forced agglomeration to gain V_dry
        _compute_saturation_for_liquid : saturation update
        """
        solver = self.solver
        a_tot = solver.a_tot

        if a_tot < 1:
            return 0.0, 0.0

        # ==========================================
        # SCHRITT 1: Partikel auswaehlen
        # ==========================================
        i = self._select_particle_uniform_physical()

        if i < 0 or i >= a_tot:
            return 0.0, 0.0
        
        # ==========================================
        # SCHRITT 2: V_dry sammeln bis >= v_liquid
        # ==========================================
        V_dry_sum = float(solver.V_flat[-1, i])
        current_liquid = float(solver.liquid_volume[i]) if hasattr(solver, 'liquid_volume') else 0.0
        new_liquid = current_liquid + v_droplet
        
        # Agglomeration Loop: Solange V_dry < new_liquid
        max_attempts = a_tot * 3
        attempts = 0
        
        while V_dry_sum < new_liquid and attempts < max_attempts:
            j = self._select_particle_uniform_physical()
            # solver.a_tot, NOT the a_tot frozen above: this loop agglomerates
            # and therefore changes the particle count itself (parents removed,
            # children created). Checking against the stale value made the
            # bound either too strict (valid new particles rejected, the
            # collection loop aborting early) or too loose (invalid index).
            # See mcpbe/docs/historical/Audit_2026-08-17.md, N-01.
            if j < 0 or j >= solver.a_tot:
                break
            
            # Manuelles Agglomerieren (mit PorosityGrowthKernel!)
            child_idx = self._perform_nucleation_agglomeration(i, j)
            if child_idx < 0 or child_idx >= solver.a_tot:
                break
            
            i = child_idx
            V_dry_sum = float(solver.V_flat[-1, i])
            # The merged particle brings its own pre-droplet liquid; the
            # pending droplet v_droplet still has to be added on top, otherwise
            # it is lost from the target value.
            current_liquid = float(solver.liquid_volume[i])
            new_liquid = current_liquid + v_droplet

            attempts += 1
        
        # Check ob erfolgreich
        if V_dry_sum < new_liquid or i >= solver.a_tot:
            return 0.0, 0.0  # Nicht genug Feststoff gefunden

        # ==========================================
        # SCHRITT 3: dW berechnen (DSMC-compliant)
        # ==========================================
        W_i = float(solver.W[i])
        if W_i <= 0:
            return 0.0, 0.0
        
        # Effective batch size (paper Eq. 33): the event consumes EXACTLY dW,
        # so the final event drains a particle to precisely 0.0 and no residual
        # weights can arise.
        dW = min(self.config.batch_size, W_i)

        # Volume capping.
        #
        # The cap is a VOLUME condition, not a weight condition: in total
        # max_physical_droplets * v_droplet of physical liquid should be
        # delivered. So instead of shrinking the packet, the amount PER
        # PARTICLE is shrunk -- dW stays on the delta grid, and
        #
        #     dW * vc_scale * v_eff  ==  max_physical_droplets * v_droplet
        #
        # holds exactly. That is also the physically correct reading: a
        # particle receives less liquid, rather than "a fraction of a particle
        # receives a full droplet". Shrinking dW to the continuous quotient
        # max_physical_droplets/vc_scale instead is what produced fractional
        # residual weights.
        if max_physical_droplets is not None and max_physical_droplets > 0:
            vc_scale = self._Vc_reference / self.solver.Vc
            max_dW_allowed = max_physical_droplets / vc_scale
            if dW > max_dW_allowed:
                v_eff = v_droplet * max_dW_allowed / dW
                if v_eff <= 0.0:
                    return 0.0, 0.0
                # Rebuild new_liquid from the smaller amount. Capacity was
                # checked above against the LARGER v_droplet, so it stays
                # conservatively valid.
                v_droplet = v_eff
                new_liquid = current_liquid + v_droplet


        # ==========================================
        # SCHRITT 4: Porositaet bestimmen (Kernel-spezifisch!)
        # ==========================================
        # Kernel-spezifische Behandlung fuer first contact
        is_first_contact = (current_liquid == 0.0)

        # Geometrie nach dem Tropfen: bestehende Porenstruktur bleibt erhalten,
        # nur echte Neu-Nukleation geht in den Porositaets-Kernel.
        v_dry_new, poro_new = self._resolve_nucleation_geometry(
            i, new_liquid, is_first_contact
        )

        # ==========================================
        # SCHRITT 5: Saturation berechnen (jeder Tropfen!)
        # ==========================================
        sat_new = self._compute_saturation_for_liquid(
            src_idx=i,
            v_dry=v_dry_new,
            poro=poro_new,
            v_liquid=new_liquid,
            v_added=v_droplet,
            is_first_contact=is_first_contact
        )
        
        # ==========================================
        # STEP 6: book the target state (merge or create)
        # ==========================================
        # Goes through the shared ParticleMerger, so matching rules and hash
        # index are the same as for agglomeration and breakage.
        self._place_nucleated_state(
            src_idx=i,
            dW=dW,
            v_dry=v_dry_new,
            poro=poro_new,
            liquid=new_liquid,
            saturation=sat_new,
            v_droplet=v_droplet if self.config.liquid_match_scale == "droplet" else None,
        )

        # Parent weight reduzieren
        solver.W[i] -= dW
        self._record_weight_change(i, solver.W[i])

        if solver.W[i] <= 0.0:
            self._remove_particle_tracked(i)

        # From here on `v_droplet` is the amount ACTUALLY delivered per
        # physical droplet -- possibly reduced by the capping above.
        return dW, v_droplet

    def _resolve_nucleation_geometry(self, i: int, new_liquid: float,
                                      is_first_contact: bool) -> tuple:
        """Determine ``(V_dry, porosity)`` of a particle after a droplet lands.

        Two physically different cases:

        1. **The particle already has pore structure** (``poro_old > 0``).
           Wetting adds liquid; it does not compact the solid skeleton.
           ``V_solid`` and ``V_pore`` stay, so ``V_dry`` and the porosity stay
           too. The liquid goes into ``liquid_volume``, the internalisation
           kernel draws it into the pores over time, and ``saturation`` tracks
           how full they are. Porosity is geometric void, saturation is fill
           level -- two separate quantities.

        2. **The particle is poreless** (``poro_old == 0`` or NaN).
           A genuine new nucleus, so the kernel sets the starting geometry.
           ``volume_mixing`` seeds 0.4 (without it, pure pore addition could
           never create porosity), ``cone_model`` starts at 0.0 and lets
           porosity grow through later agglomeration.

        Case 1 exists because ``compute_nucleation_porosity`` only receives
        ``v_solid``: routing an already-porous particle through it would return
        ``(v_solid, 0.0)`` and erase its entire pore space.

        Parameters
        ----------
        i : int
            Index of the particle that was hit.
        new_liquid : float
            The particle's total liquid after the droplet [m^3].
        is_first_contact : bool
            True if the particle was dry before (only used by volume_mixing).

        Returns
        -------
        (v_dry_new, poro_new) : tuple[float, float]
        """
        solver = self.solver

        V_dry_i = float(solver.V_flat[-1, i])
        poro_old = float(solver.porosity[i]) if hasattr(solver, 'porosity') else np.nan

        # Fall 1: bestehende Porenstruktur -> Geometrie unveraendert lassen.
        if not np.isnan(poro_old) and poro_old > 0.0:
            return V_dry_i, poro_old

        # Case 2: poreless particle -> the kernel sets the starting geometry.
        porosity_kernel = self._get_porosity_kernel()

        if porosity_kernel.name == 'volume_mixing' and is_first_contact:
            # Volume Mixing BRAUCHT die initiale 0.4 (reine Porenaddition
            # koennte aus 0 + 0 nie Porositaet erzeugen).
            nucleation_params = {'default_porosity': 0.4}
        else:
            nucleation_params = None

        V_solid_i = V_dry_i if np.isnan(poro_old) else V_dry_i * (1.0 - poro_old)

        return porosity_kernel.compute_nucleation_porosity(
            v_solid=V_solid_i,
            v_liquid=new_liquid,
            nucleation_params=nucleation_params,
            solver=solver
        )

    def _get_porosity_kernel(self):
        """
        Get PorosityGrowthKernel from solver or create default.

        Returns:
            PorosityGrowthKernel instance (volume_mixing as default)
        """
        solver = self.solver
        
        if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
            kernel = solver.kernel_manager.porosity_growth_kernel
            if kernel is not None:
                return kernel
        
        # Create default volume_mixing kernel
        from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
        return get_porosity_growth_kernel('volume_mixing')
    
    def _compute_saturation_for_liquid(self, src_idx: int, v_dry: float, poro: float,
                                        v_liquid: float, v_added: float,
                                        is_first_contact: bool) -> float:
        """Saturation of the target state after a droplet lands.

        Called for EVERY droplet, so this updates the previous state rather
        than recomputing it from scratch.

        Physics
        -------
        The internal/external split is a state variable **with memory**:
        internal liquid arises from capillary suction over time (the
        internalisation kernel), not as a fixed fraction of the total. A
        droplet that has just landed has penetrated nothing and is therefore
        **entirely external**; liquid already imbibed stays where it is.

        Hence::

            V_int_new = V_int_old                        (internalisation kernel active)
            V_int_new = V_int_old + split * v_added      (no kernel: empirical split,
                                                          applied to the INCREMENT only)
            V_int_new = min(V_int_new, V_pore_new, v_liquid)  -> excess becomes external
            S         = V_int_new / V_pore_new

        Recomputing the split from the total liquid on every droplet would reset
        a fully saturated particle (S = 1) back to S = 0 with the next droplet,
        leaving internalisation to work against a constant reset and breaking
        the chain wetting -> strength -> breakage.

        Excess liquid from a shrinking pore space is externalised, not
        discarded: the liquid balance is as inviolable as the mass balance
        (same rule as in ``_apply_compression`` and ``mcpbe_agg._merge_pair``).

        Parameters
        ----------
        src_idx : int
            Index of the particle that was hit -- the source of the previous
            internal/external state. ``None`` or invalid means no prior state.
        v_dry, poro : float
            Geometry of the target state.
        v_liquid : float
            Total liquid of the target state (previous + droplet).
        v_added : float
            Volume of the droplet just landing [m^3].
        is_first_contact : bool
            True if the particle was dry before.

        Returns
        -------
        float
            Saturation S in [0, 1].
        """
        solver = self.solver

        # Handle NaN or zero porosity
        if np.isnan(poro) or poro <= 0.0:
            return 0.0

        V_pore_new = v_dry * poro
        if V_pore_new <= 0:
            return 0.0

        # ------------------------------------------------------------------
        # Bisherige interne Fluessigkeit des getroffenen Partikels
        # ------------------------------------------------------------------
        V_int_old = 0.0
        if src_idx is not None and 0 <= src_idx < solver.a_tot:
            poro_old = float(solver.porosity[src_idx])
            if np.isfinite(poro_old) and poro_old > 0.0:
                V_pore_old = float(solver.V_flat[-1, src_idx]) * poro_old
                sat_old = float(solver.saturation[src_idx])
                if not np.isfinite(sat_old):
                    sat_old = 0.0
                sat_old = min(max(sat_old, 0.0), 1.0)
                liq_old = float(solver.liquid_volume[src_idx])
                # Cap at the liquid actually present: poro, saturation and
                # liquid_volume are set independently elsewhere and are not
                # guaranteed to be mutually consistent.
                V_int_old = max(0.0, min(liq_old, V_pore_old * sat_old))

        # Check if internalization kernel is active
        has_internalization_kernel = (
            hasattr(solver, 'kernel_manager') and
            solver.kernel_manager is not None and
            solver.kernel_manager.liquid_internalization_kernel is not None
        )

        if has_internalization_kernel:
            # The kernel draws liquid inward over time; the droplet itself
            # lands entirely on the outside.
            V_int_new = V_int_old
        else:
            # No kernel: the empirical split is then the only mechanism that
            # moves liquid inward at all, so it acts on the INCREMENT rather
            # than on the total.
            split_ratio = self._get_liquid_split_ratio(poro)
            V_int_new = V_int_old + split_ratio * max(0.0, v_added)

        # Pore space may have shrunk -> excess becomes external, not discarded.
        V_int_new = min(V_int_new, V_pore_new, max(0.0, v_liquid))

        return V_int_new / V_pore_new
    
    def _get_liquid_split_ratio(self, poro: float) -> float:
        """Fraction of newly arriving liquid that counts as internal.

        ``split_ratio = V_liq_int / V_liq_total``; the rest is the external
        surface film. Only used when no internalisation kernel is configured --
        with a kernel, internal liquid is produced by capillary suction over
        time instead (see :meth:`_compute_saturation_for_liquid`).

        Kernel-dependent values: ``volume_mixing`` uses a hardcoded empirical
        0.4, ``cone_model`` reads it from its parameters (default 0.4).

        Parameters
        ----------
        poro : float
            Current porosity. Not used at present; accepted so a
            porosity-dependent split can be added without changing callers.

        Returns
        -------
        float
            Split ratio in [0, 1].

        See Also
        --------
        _compute_saturation_for_liquid : the only caller
        """
        porosity_kernel = self._get_porosity_kernel()
        
        if porosity_kernel.name == 'volume_mixing':
            # Volume Mixing: 0.4 hardcoded (legacy behavior)
            return 0.4
        elif porosity_kernel.name == 'cone_model':
            # NOTE: the 0.4 default is questionable for cone_model. That
            # kernel creates pore space geometrically on contact and makes no
            # claim about how much liquid goes inward on impact -- 0.0 would be
            # the consistent choice. The path is only reachable when NO
            # liquid_internalization kernel is active. Left unchanged because
            # it shifts results; decide deliberately and record the decision
            # here.
            return porosity_kernel.params.get('liquid_split_ratio', 0.4)
        else:
            # Other kernels: Default 0.4
            return 0.4
    
    def _get_merger(self):
        """Return the solver's ParticleMerger, or ``None`` if there is none.

        Nucleation deliberately shares the merger with agglomeration and
        breakage, so all three use one dedup rule and one hash index. A
        separate implementation here would create particles the other two
        cannot find, and they would then duplicate them.
        """
        return getattr(self.solver, '_particle_merger', None)

    def _remove_particle_tracked(self, idx: int) -> None:
        """Remove a particle, keeping the merger's hash index consistent.

        ``_remove_particle_column`` works by swap-with-last, so the removed
        particle's hash entry has to go BEFORE the swap -- otherwise it points
        at a slot that a different particle is about to occupy. The swap itself
        is covered by ``notify_index_swap``. Agglomeration and breakage do the
        same at their removal sites.
        """
        merger = self._get_merger()
        if merger is not None:
            merger.remove_from_hash_index(idx)
        self.solver._remove_particle_column(idx)

    def _place_nucleated_state(self, src_idx: int, dW: float, v_dry: float,
                               poro: float, liquid: float, saturation: float,
                               v_droplet: Optional[float] = None) -> tuple:
        """Book the post-droplet state: merge into an existing particle or create one.

        Goes through ``ParticleMerger.find_or_create``, so nucleation shares
        the dedup logic and the hash index with agglomeration and breakage.

        Parameters
        ----------
        src_idx : int
            Source particle. Supplies the solid components and is excluded from
            matching against itself.
        dW : float
            Weight credited to the target particle.
        v_dry, poro, liquid, saturation : float
            The target state after the droplet.
        v_droplet : float, optional
            Droplet volume, used as the reference scale for the liquid
            tolerance (only when ``liquid_match_scale == "droplet"``).

        Returns
        -------
        (idx, was_merged) : tuple[int, bool]
        """
        solver = self.solver
        V_solid_src = solver.V_flat[:solver.dim, src_idx].copy()

        merger = self._get_merger()
        if merger is None:
            # Kein Merger konfiguriert -> direkt anlegen (Alt-Verhalten).
            self._create_nucleated_particle_direct(
                V_solid_src, dW, v_dry, poro, liquid, saturation, src_idx
            )
            return solver.a_tot - 1, False

        idx, merged = merger.find_or_create(
            V_solid_target=V_solid_src,
            V_dry_target=v_dry,
            liquid_target=liquid,
            poro_target=poro,
            sat_target=saturation,
            weight_to_add=dW,
            component_sum=V_solid_src,
            tol_rel_override=self.config.similarity_tol,
            liquid_scale_ref=v_droplet,
            exclude_idx=src_idx,
            force_linear_scan=self.config.merge_linear_scan,
        )
        self._record_weight_change(idx, solver.W[idx])
        return idx, merged

    def _create_nucleated_particle_direct(self, V_solid_src, dW: float,
                                          v_dry: float, poro: float,
                                          liquid: float, saturation: float,
                                          src_idx: int) -> None:
        """Create a particle directly, bypassing the merger.

        Fallback for the case where no merger exists; the normal path is
        :meth:`_place_nucleated_state`.

        Parameters
        ----------
        V_solid_src : np.ndarray
            Solid components of the source particle.
        dW : float
            Weight for the new particle.
        v_dry : float
            Dry volume [m^3].
        poro : float
            Porosity.
        liquid : float
            Total liquid volume [m^3].
        saturation : float
            Saturation, already computed.
        src_idx : int
            Source particle index, for debug output only.
        """
        solver = self.solver

        # Create new particle
        solver._append_particle_column(V_solid_src)
        new_idx = solver.a_tot - 1

        # Set properties. V_dry from the kernel -- and X, the collision
        # diameter derived from it (see mcpbe_base.set_particle_dry_volume).
        solver.set_particle_dry_volume(new_idx, v_dry)
        solver.W[new_idx] = dW

        if hasattr(solver, 'liquid_volume'):
            solver.liquid_volume[new_idx] = liquid

        if hasattr(solver, 'porosity'):
            solver.porosity[new_idx] = poro

        if hasattr(solver, 'saturation'):
            solver.saturation[new_idx] = saturation

        # === DEBUG NUC: CREATE/UPDATE PARTICLE ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_nuc', True):
            # V_solid vom Parent (Source)
            V_solid_src = solver.V_flat[:solver.dim, src_idx].copy()
            V_solid_src_total = float(np.sum(V_solid_src))
            
            # V_solid vom neuen Partikel
            V_solid_new = solver.V_flat[-1, new_idx] * (1.0 - solver.porosity[new_idx]) if not np.isnan(solver.porosity[new_idx]) else solver.V_flat[-1, new_idx]
            
            print(f"\n[DEBUG NUC-PARTICLE] Created particle at idx={new_idx}")
            print(f"  Source: idx={src_idx}, dW={dW:.2f}")
            print(f"  V_solid_src (from parent)={V_solid_src_total:.6e}")
            print(f"  V_dry_new (from kernel)={v_dry:.6e}")
            print(f"  poro_new={solver.porosity[new_idx]:.4f}")
            print(f"  V_solid_new = V_dry_new*(1-poro)={V_solid_new:.6e}")
            print(f"  DELTA_V_SOLID = V_solid_new - V_solid_src = {V_solid_new - V_solid_src_total:.6e} (MUSST = 0 sein!)")
            print(f"  liquid={solver.liquid_volume[new_idx]:.6e}, sat={saturation:.4f}")
# ========================================
        
        # Record weight change
        self._record_weight_change(new_idx, dW)
    
    def _perform_nucleation_agglomeration(self, i: int, j: int) -> int:
        """Force an agglomeration of ``i`` and ``j`` to gain dry volume.

        Used when the drawn particle cannot hold the droplet
        (``V_dry < v_liquid``). Delegates to the solver's own agglomeration
        logic so liquid transfer, weight bookkeeping and porosity stay
        consistent with a normally drawn event.

        Parameters
        ----------
        i, j : int
            Indices of the two particles.

        Returns
        -------
        int
            Index of the resulting child particle.

        See Also
        --------
        mcpbe_agg.MCPBEAgg._do_one_agg : the drawn-event counterpart
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
        """Merge particles ``i`` and ``j`` without touching the propensities.

        The propensity rebuild is O(n); running it once per droplet inside the
        collection loop would dominate the whole simulation. It is skipped here
        and made up for once per MC event in ``solve()``.

        Returns the child's ACTUAL index, which need not be ``a_tot - 1``: if a
        parent was removed by swap-with-last, the child may have been moved
        into that slot.

        Volume semantics
        ----------------
        ``V_flat[-1, :]`` holds ``V_dry = V_solid + V_pore``; ``V_flat[:dim, :]``
        holds the solid components (a single one for ``dim == 1``). Solid volume
        is always derived as ``V_dry * (1 - porosity)``, never stored twice
        independently.

        Parameters
        ----------
        i, j : int
            Indices of the two particles.

        Returns
        -------
        int
            Index of the child particle.

        See Also
        --------
        _perform_nucleation_agglomeration : wrapper with the nucleation-specific
            bookkeeping
        """
        solver = self.solver
        
        if i >= solver.a_tot or j >= solver.a_tot:
            return -1
        
        # === DEBUG NUC-AGG: VOR MANUELLER AGG ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_nuc', True):
            # Globale Masse VOR der Agg berechnen
            v_dry_gl = solver.V_flat[-1, :solver.a_tot]
            poro_gl = solver.porosity[:solver.a_tot]
            w_gl = solver.W[:solver.a_tot]
            valid_gl = ~np.isnan(poro_gl)
            v_solid_gl = np.zeros_like(v_dry_gl)
            v_solid_gl[valid_gl] = v_dry_gl[valid_gl] * (1.0 - poro_gl[valid_gl])
            v_solid_gl[~valid_gl] = v_dry_gl[~valid_gl]
            solid_global_before = np.sum(v_solid_gl * w_gl)
            
            # V_solid von beiden Partikeln
            V_dry_i_dbg = float(solver.V_flat[-1, i])
            V_dry_j_dbg = float(solver.V_flat[-1, j])
            poro_i_dbg = solver.porosity[i] if hasattr(solver, 'porosity') else np.nan
            poro_j_dbg = solver.porosity[j] if hasattr(solver, 'porosity') else np.nan
            
            V_solid_i_dbg = V_dry_i_dbg * (1.0 - poro_i_dbg) if not np.isnan(poro_i_dbg) else V_dry_i_dbg
            V_solid_j_dbg = V_dry_j_dbg * (1.0 - poro_j_dbg) if not np.isnan(poro_j_dbg) else V_dry_j_dbg
            
            liq_i_dbg = float(solver.liquid_volume[i]) if hasattr(solver, 'liquid_volume') else 0.0
            liq_j_dbg = float(solver.liquid_volume[j]) if hasattr(solver, 'liquid_volume') else 0.0
            
            print(f"\n[DEBUG NUC-AGG] Manual agglomeration")
            print(f"  Particles: i={i}, j={j}")
            print(f"  W[i]={solver.W[i]:.2f}, W[j]={solver.W[j]:.2f}")
            print(f"  V_solid[i]={V_solid_i_dbg:.6e}, V_solid[j]={V_solid_j_dbg:.6e}")
            print(f"  V_solid_SUM={V_solid_i_dbg + V_solid_j_dbg:.6e} (MUSST = Child V_solid sein)")
            print(f"  liq[i]={liq_i_dbg:.6e}, liq[j]={liq_j_dbg:.6e}, SUM={liq_i_dbg + liq_j_dbg:.6e}")
            print(f"  GLOBAL BEFORE: V_solid_total={solid_global_before:.6e}")
# =========================================
        
        # Get weights
        Wi = float(solver.W[i])
        Wj = float(solver.W[j])

        if Wi <= 0.0 or Wj <= 0.0:
            return -1

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
        # For Vollkoerper (NaN porosity): V_solid = V_dry
        # For porous particles: V_solid = V_dry * (1 - poro)
        V_solid_i = V_dry_i * (1.0 - poro_i) if not np.isnan(poro_i) else V_dry_i
        V_solid_j = V_dry_j * (1.0 - poro_j) if not np.isnan(poro_j) else V_dry_j

        # DSMC-compliant weight handling, identical to mcpbe_agg.py::_compute_agg_dW:
        # the consumed batch is exactly the effective batch size (paper Eq. 33), so
        # the final event on a particle drains it to exactly 0 and no residual
        # weight can survive. For i == j one event consumes 2*dW from the SAME
        # packet, hence the W_i/2 cap (2*(0.5*W_i) == W_i is exact in IEEE-754).
        dW_const = float(
            getattr(solver, "_agg_dW_const", None) or solver._prepare_agg_delta_config()
        )
        delta_i = min(dW_const, Wi)
        delta_j = min(dW_const, Wj)
        if i == j:
            dW = min(delta_i, 0.5 * Wi)
        else:
            dW = min(delta_i, delta_j)
        if dW <= 0.0:
            return -1
        
        # Get porosity growth kernel (create default if not exists)
        porosity_kernel = None
        if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
            porosity_kernel = solver.kernel_manager.porosity_growth_kernel
        
        if porosity_kernel is None:
            # Create default volume_mixing kernel
            from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
            porosity_kernel = get_porosity_growth_kernel('volume_mixing')
        
        # === FIX: MASS CONSERVATION ===
        # Compute V_solid_merged FIRST by summing parent solid volumes (EXACT conservation!)
        # This follows the pattern in mcpbe_agg.py::_merge_pair()
        V_solid_i = V_dry_i * (1.0 - poro_i) if not np.isnan(poro_i) else V_dry_i
        V_solid_j = V_dry_j * (1.0 - poro_j) if not np.isnan(poro_j) else V_dry_j
        V_solid_merged = V_solid_i + V_solid_j  # ← EXAKTE MASSEERHALTUNG!
        
        # Compute merged porosity via kernel (for V_dry and poro_merged only)
        V_dry_merged, poro_merged = porosity_kernel.compute_merged_porosity(
            v_dry1=V_dry_i, poro1=poro_i,
            v_dry2=V_dry_j, poro2=poro_j,
            v_liq1=liq_i, v_liq2=liq_j,
            sat1=solver.saturation[i],
            sat2=solver.saturation[j],
            solver=solver
        )
        # Note: V_dry_merged is used for solver.V_flat[-1], but V_solid_merged
        # is used for _append_particle_column() to ensure mass conservation.
        
        # ==========================================
        # Compute pore volumes for liquid handling
        # ==========================================
        # Universal formula: V_pore = V_dry * poro
        V_pore_i = V_dry_i * poro_i
        V_pore_j = V_dry_j * poro_j
        V_pore_merged = V_dry_merged * poro_merged
        
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

            # At the newly formed contact, additional liquid is drawn from the
            # surface films into the fresh bridge. This is a separate,
            # event-based process at collision -- not the same as continuous
            # internalisation over time. `mcpbe_agg._merge_pair` calls this
            # kernel too; nucleation-driven agglomeration must not skip it.
            l_e_to_i = 0.0
            if getattr(solver, 'kernel_manager', None) is not None:
                l_e_to_i = solver.kernel_manager.compute_liquid_internalization_agglomeration(
                    v_dry1=float(V_dry_i),
                    v_dry2=float(V_dry_j),
                    v_liq_ext1=float(V_liq_ext_i),
                    v_liq_ext2=float(V_liq_ext_j),
                    particle1_idx=i,
                    particle2_idx=j,
                    solver=solver,
                )
            V_liq_int_merged += l_e_to_i
            V_liq_ext_merged -= l_e_to_i
            if V_liq_ext_merged < 0.0:
                # Kernel wollte mehr internalisieren als aussen vorhanden war:
                # alles nach innen, nichts erfinden (wie in _merge_pair).
                V_liq_int_merged = V_liq_int_i + V_liq_int_j + V_liq_ext_i + V_liq_ext_j
                V_liq_ext_merged = 0.0

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
        
        # Overwrite V_flat[-1] with dry volume (includes pores). X is the
        # collision diameter derived from V_dry and has to follow -- see
        # mcpbe_base.set_particle_dry_volume.
        solver.set_particle_dry_volume(new_idx, V_dry_merged)
        solver.W[new_idx] = dW  # DSMC: child gets dW, not Wi+Wj!
        
        if hasattr(solver, 'liquid_volume'):
            solver.liquid_volume[new_idx] = liq_merged
        
        if poro_merged is not None and hasattr(solver, 'porosity'):
            solver.porosity[new_idx] = poro_merged
        
        if sat_merged is not None and hasattr(solver, 'saturation'):
            solver.saturation[new_idx] = sat_merged

        # Register in the merger hash index. The child is created directly
        # rather than through find_or_create, because the code below has to
        # track its index across several parent removals (child_current_idx).
        # Without registration it would be invisible to every later lookup and
        # agglomeration/breakage would create duplicates.
        merger = self._get_merger()
        if merger is not None:
            merger.register_particle(new_idx)

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
            
            if solver.W[i] <= 0.0:
                # If removing i which is also the child location, we have a problem
                if i == child_current_idx:
                    # Child would be removed - this shouldn't happen in normal operation
                    # Return -1 to signal error
                    return -1
                # CRITICAL: Save 'last' BEFORE remove, because a_tot changes!
                last_before_remove = solver.a_tot - 1
                self._remove_particle_tracked(i)
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
                if solver.W[idx] <= 0.0:
                    child_was_at = child_current_idx
                    # CRITICAL: Save 'last' BEFORE remove, because a_tot changes!
                    last_before_remove = solver.a_tot - 1
                    self._remove_particle_tracked(idx)
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
        
        # === DEBUG NUC-AGG: NACH MANUELLER AGG ===
        if getattr(solver, 'mcpbe_debug_mass', False) and getattr(solver, 'mcpbe_debug_nuc', True):
            # Globale Masse NACH der Agg berechnen
            v_dry_gl_after = solver.V_flat[-1, :solver.a_tot]
            poro_gl_after = solver.porosity[:solver.a_tot]
            w_gl_after = solver.W[:solver.a_tot]
            valid_gl_after = ~np.isnan(poro_gl_after)
            v_solid_gl_after = np.zeros_like(v_dry_gl_after)
            v_solid_gl_after[valid_gl_after] = v_dry_gl_after[valid_gl_after] * (1.0 - poro_gl_after[valid_gl_after])
            v_solid_gl_after[~valid_gl_after] = v_dry_gl_after[~valid_gl_after]
            solid_global_after = np.sum(v_solid_gl_after * w_gl_after)
            
            V_dry_child = float(solver.V_flat[-1, child_current_idx])
            poro_child = solver.porosity[child_current_idx] if hasattr(solver, 'porosity') else np.nan
            V_solid_child = V_dry_child * (1.0 - poro_child) if not np.isnan(poro_child) else V_dry_child
            
            liq_child = float(solver.liquid_volume[child_current_idx]) if hasattr(solver, 'liquid_volume') else 0.0
            
            print(f"  Child: idx={child_current_idx}")
            print(f"  W[child]={solver.W[child_current_idx]:.2f}")
            print(f"  V_solid[child]={V_solid_child:.6e}")
            print(f"  ΔV_solid={V_solid_child - (V_solid_i_dbg + V_solid_j_dbg):.6e} (MUSST = 0 sein)")
            print(f"  liq[child]={liq_child:.6e}")
            print(f"  a_tot changed: {solver.a_tot}")
            print(f"  GLOBAL AFTER: V_solid_total={solid_global_after:.6e}")
            print(f"  GLOBAL ΔV_solid={solid_global_after - solid_global_before:.6e} (MUSST = 0 sein!)")
# =========================================
        
        return child_current_idx  # Return actual child index!
    
    # The wt% runtime monitoring has been removed here
    # (get_current_wt_percent / _should_check_wt / _update_wt_check /
    # has_reached_target_wt). It computed the current liquid fraction during
    # the run to stop addition early once target_wt_percent was reached, but
    # the "adaptive throttling" in front of it called the very check it was
    # meant to avoid -- an O(n) Python loop over all particles -- and
    # has_reached_target_wt() then ran it a second time. Measured at 1.93
    # passes per throttle check, i.e. doubled rather than saved work.
    # See mcpbe/docs/historical/Audit_2026-08-17.md, N-03.
    #
    # NOT removed: `target_wt_percent` as a CONFIGURATION parameter. It still
    # converts wt% + solid_mass_in_mixer + densities into an addition duration
    # (NucleationConfig.__post_init__, option B) -- turning "add 53 wt% water"
    # into a window length, which is the form the process specification
    # actually comes in.

    def get_statistics(self) -> dict:
        """Current nucleation statistics.

        Returns
        -------
        dict
            ``current_time`` [s], ``droplets_added_total`` (physical droplets),
            ``liquid_volume_added_total`` [m^3], ``target_wt_percent`` from the
            configuration (or ``None``), and ``in_addition_window``.

        See Also
        --------
        print_debug_status : formats these for the console
        """
        stats = {
            'current_time': self._current_time,
            'next_nucleation_time': self._next_nucleation_time,
            'droplets_added_total': self._droplets_added_total,
            'liquid_volume_added_total': self._liquid_volume_added_total,
            'target_wt_percent': self.config.target_wt_percent,
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
            stats['volumetric_flow_rate'] = self.config.volumetric_flow_rate
        
        return stats
    
    def reset(self) -> None:
        """Clear all nucleation state so the solver can be run again.

        Resets statistics, timers and samplers. Called from
        ``MCPBESolver._reset_state``.
        """
        self._current_time = 0.0
        self._next_nucleation_time = self.config.liquid_addition_start
        self._Vc_reference = self.solver.Vc
        self._droplets_added_total = 0
        self._liquid_volume_added_total = 0.0
        
        # Reset MINIMAL window tracking (5 variables)
        self._last_event_time = 0.0
        self._had_events_in_window = False
        self._manual_trigger_done = False
        self._liquid_remainder = 0.0
        
        # Reset sampler
        self._weight_sampler = None
        self._weight_array = None
        
        # Reset debug tracking
        self._debug_last_print_time = -1.0
    
    def _log(self, message: str, level: str = "DEBUG") -> None:
        """
        Emit diagnostic output via solver logger.
        
        Args:
            message: Log message
            level: Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        
        Logs when config.debug OR solver.mcpbe_debug is set.
        """
        if getattr(self.config, "debug", False) or getattr(self.solver, "mcpbe_debug", False):
            logger = getattr(self.solver, 'logger', None)
            if logger is not None:
                log_method = getattr(logger, level.lower(), logger.debug)
                log_method(f"[Nucleation] {message}")
            else:
                # Fallback to print if logger not available
                print(f"[NUCLEATION {level}] {message}")

    def print_debug_status(self, force: bool = False) -> None:
        """Print a progress line for the nucleation process.

        Covers simulated time (absolute and as a fraction of the total), wall
        clock time and the resulting speed-up, droplet count against
        expectation, and particle counts (computational and physical).

        Parameters
        ----------
        force : bool
            Print regardless of the interval throttle.

        Notes
        -----
        Prints only when ``config.debug`` or ``solver.mcpbe_debug`` is set, and
        at most every ``_debug_print_interval`` seconds (default 0.1).
        """
        import time as time_module
        
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
        
        # Calculate speedup (simulated time / real time)
        speedup = t_elapsed / real_time_elapsed if real_time_elapsed > 0 else 0.0
        
        # Print formatted status with timing
        self._log(
            f"t={t_elapsed:.4f}s ({t_percent:.1f}% of total) | "
            f"Real time: {real_time_elapsed:.2f}s (speedup: {speedup:.1f}x) | "
            f"Droplets: {n_droplets:.2e} ({percent_of_expected:.1f}% of expected) | "
            f"Particles: n_comp={n_comp}, n_phys={n_phys:.2e}",
            "INFO"
        )


# Convenience function for creating nucleation handler
def create_nucleation_handler(solver, **kwargs) -> NucleationHandler:
    """Create a nucleation handler for ``solver``.

    Equivalent to ``solver.create_nucleation_handler(**kwargs)``, kept as a
    free function for callers that build handlers without going through the
    solver.

    Parameters
    ----------
    solver : MCPBESolver
        The owning solver.
    **kwargs
        Passed to :class:`NucleationConfig`.

    Returns
    -------
    NucleationHandler

    Examples
    --------
    >>> handler = create_nucleation_handler(
    ...     solver,
    ...     enabled=True,
    ...     volumetric_flow_rate=1e-9,
    ...     droplet_diameter=1e-6,
    ...     liquid_addition_duration=300.0
    ... )
    """
    config = NucleationConfig(**kwargs)
    handler = NucleationHandler(solver, config)
    return handler
