"""
Nucleation Handler fuer Fluessigkeitszugabe via Tropfenverteilung.

Physikalisches Modell:
    Tropfen kollidieren zufaellig mit Partikeln. Alle Partikel haben gleiche
    Kollisionswahrscheinlichkeit (uniform Sampling gewichtet nach W).
    
    DSMC-Skalierung: Ein computational droplet event repraesentiert
    physical_droplets = (Vc_ref/Vc) × W_selected physikalische Tropfen.

Komponenten:
    NucleationConfig: Konfigurations-Dataclass mit Validierungslogik
    NucleationHandler: Hauptklasse (Composition Pattern, kein Mixin)

Verwendung:
    >>> config = NucleationConfig(
    ...     enabled=True,
    ...     volumetric_flow_rate=1e-9,      # m^3/s
    ...     droplet_diameter=1e-6,          # m
    ...     liquid_addition_duration=300.0, # s
    ... )
    >>> handler = solver.create_nucleation_handler(**config.__dict__)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .fenwick_new import FenwickSampler


class InitializationType(Enum):
    """
    Veraltet: Nur fuer Abwaertskompatibilitaet.
    
    Moderne Nukleation verwendet einheitliche zeitbasierte Fluessigkeitszugabe.
    Diese Enum wird nicht mehr verwendet.
    """
    DURATION = "duration"
    MASS = "mass"


@dataclass
class NucleationConfig:
    """
    Konfiguration fuer zeitbasierte Fluessigkeitszugabe via Tropfenverteilung.
    
    Physikalisches Modell:
        Tropfen kollidieren zufaellig mit Partikeln (uniform Sampling nach W).
        Ein computational droplet event repraesentiert mehrere physikalische Tropfen:
        physical_droplets = (Vc_ref/Vc) × W_selected
    
    Dauer-Spezifikation (mutually exclusive):
        Option A: Direkt via `liquid_addition_duration` [s]
        Option B: Berechnet aus target_wt_percent + solid_mass_in_mixer + Dichten
    
    Parameter
    ---------
    enabled : bool
        Aktiviert Nukleation
    volumetric_flow_rate : float
        Physikalische Flussrate [m^3/s]. Wird direkt verwendet (keine DSMC-Skalierung).
    droplet_diameter : float
        Physikalischer Tropfendurchmesser [m]
    liquid_addition_start : float, optional
        Startzeit [s], Default: 0.0
    liquid_addition_duration : float, optional
        Direkte Dauer [s]
    target_wt_percent : float, optional
        Ziel Fluessigkeitsanteil (dry basis) [%]
    solid_mass_in_mixer : float, optional
        Gesamte Feststoffmasse im Mixer [kg]
    rho_solid : float, optional
        Dichte Feststoff [kg/m^3]
    rho_liquid : float, optional
        Dichte Fluessigkeit [kg/m^3]
    batch_size : float, optional
        Computational weight pro Tropfen-Event. Default: 25.0.
        dW = min(batch_size, W_i) steuert Granularitaet der Verteilung.
    consistency_tol : float
        Relative Toleranz fuer Konsistenzpruefung, Default: 0.001 (0.1%)
    similarity_tol : float
        Relative Toleranz fuer Partikel-Merging, Default: 1e-5
    liquid_match_scale : str
        Referenz fuer liquid tolerance: 'droplet' (Default, massenerhaltend)
        oder 'total' (veraltet, fuehrt zu systematischem Verlust)
    debug : bool
        Diagnoseausgaben aktivieren, Default: False
    
    Raises
    ------
    ValueError
        Wenn volumetric_flow_rate oder droplet_diameter ≤ 0
        Wenn weder Dauer noch Massen-Parameter angegeben
        Wenn beide Methoden disagreed > 0.1%
    
    See Also
    --------
    NucleationHandler : Hauptklasse die diese Konfiguration verwendet
    
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
    #               |ist - Kandidat| / ist <= similarity_tol.
    #   "droplet" (default) - the same rule, but additionally capped at
    #               similarity_tol * v_droplet. The cap can only tighten the
    #               tolerance, never loosen it.
    #
    # Why the cap: under "total" the accepted absolute liquid difference grows
    # with how much liquid a particle already carries, so late merges accept
    # ever larger differences while the statistics still book a whole droplet.
    # That mechanism is plausible but its magnitude has NOT been measured here.
    #
    # CORRECTION (2026-08): earlier revisions of this comment justified the
    # default with "0.58 % liquid drift, see docs/REFACTORING_FINDINGS.md
    # (F-03)". That attribution was wrong on two counts: F-03 describes the
    # nucleation REMAINDER path (the child was created with only
    # _liquid_remainder instead of parent_liquid + remainder) and has long been
    # fixed, and the file now lives at docs/old/REFACTORING_FINDINGS.md. The
    # 0.58 % belongs to that unrelated bug and says nothing about this
    # tolerance. "droplet" is kept as the conservative default; switch to
    # "total" and compare `python -m tests.test_conservation` if you want the
    # higher merge hit rate.
    liquid_match_scale: str = "droplet"

    # Suchweg im ParticleMerger.
    #   True (Default) - vektorisierter linearer Scan ueber alle aktiven
    #           Partikel. Zusaetzlich deterministisch: liefert den niedrigsten
    #           passenden Index, waehrend der Hash-Pfad ueber ein set iteriert.
    #   False - Hash-Index.
    #
    # Warum NICHT der Hash, obwohl er "O(1)" ist: der Hash-Key binnt V_dry und
    # liquid auf log10/8 Stellen und poro/sat auf 4 Nachkommastellen. In der
    # Nucleation starten aber alle Partikel monodispers und trocken, sitzen also
    # im selben Bucket. _find_via_hash iteriert dieses Bucket in einer
    # PYTHON-Schleife und ruft _matches_exact je Kandidat einzeln auf -- bei
    # einem Bucket mit ~n Eintraegen und einer Trefferquote von wenigen Prozent
    # heisst das: fast jeder Lookup laeuft das ganze Bucket interpretiert durch.
    # Der lineare Scan macht dieselbe Pruefung vektorisiert in NumPy und ist
    # dadurch um Groessenordnungen schneller, obwohl er formal dieselbe
    # Komplexitaet hat. Gemessen: mit Hash blieb ein 20-s-Lauf nach t~4 haengen,
    # mit linearem Scan laeuft er in ~50 s durch.
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
    """
    Handler fuer Fluessigkeitszugabe via Tropfenverteilung im MC-Simulator.
    
    Physikalisches Modell:
        Fluessigkeit wird als diskrete Tropfen zugegeben. Alle Partikel haben
        gleiche Kollisionswahrscheinlichkeit (uniform Sampling nach W).
    
    DSMC-Skalierung:
        Ein computational droplet event repraesentiert:
        physical_droplets = (Vc_ref / Vc) × W_selected
        
        Dadurch bleibt die physikalische Flussrate (m^3/s) konstant,
        unabhaengig von Control-Volume-Änderungen.
    
    Funktionsweise:
        1. Überlappung Event-Intervall mit Nukleations-Fenster berechnen
        2. Fluessigkeitsvolumen = flow_rate × dt_overlap
        3. Partikel gewichtet nach W auswaehlen (FenwickSampler, O(log n))
        4. Tropfen hinzufuegen, Porositaet/Saettigung via Kernel aktualisieren
        5. Rest-Volumen akkumulieren (massenerhaltend)
    
    Attributes
    ----------
    solver : MCPBESolver
        Eltern-Solver Instanz
    config : NucleationConfig
        Konfigurationsobjekt
    _current_time : float
        Aktuelle Simulationszeit [s]
    _Vc_reference : float
        Referenz-Control-Volume fuer DSMC-Statistiken [m^3]
    _droplets_added_total : float
        Gesamtzahl physikalischer Tropfen (gewichtete Summe)
    _liquid_volume_added_total : float
        Gesamtes Fluessigkeitsvolumen [m^3]
    _liquid_remainder : float
        Akkumuliertes Restvolumen fuer Massenerhaltung [m^3]
    
    See Also
    --------
    NucleationConfig : Konfigurations-Dataclass
    fenwick_new.FenwickSampler : Gewichtetes Sampling
    
    Examples
    --------
    >>> config = NucleationConfig(enabled=True, volumetric_flow_rate=1e-9,
    ...                           droplet_diameter=1e-6, liquid_addition_duration=300.0)
    >>> handler = solver.create_nucleation_handler(**config.__dict__)
    >>> solver.solve()
    """
    
    def __init__(self, solver, config: NucleationConfig) -> None:
        """
        Initialisiert Nukleations-Handler fuer Fluessigkeitszugabe.
        
        Parameter
        ---------
        solver : MCPBESolver
            Eltern-Solver Instanz
        config : NucleationConfig
            Konfigurationsobjekt mit Flussrate, Tropfengroesse, Zeitfenster
        
        Raises
        ------
        ValueError
            Wenn required Solver-Attribute (x, Vc) nicht gesetzt
        
        See Also
        --------
        NucleationConfig : Konfigurationsparameter
        MCPBESolver.create_nucleation_handler : Factory-Methode
        """
        self.solver = solver
        self.config = config
        
        # Internal state
        self._current_time = 0.0
        
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
        """
        Markiert MC-Event fuer Statistik (veraltet, nur Kompatibilitaet).
        
        Wird vom Solver nach jedem Event aufgerufen. Aktualisiert
        _last_event_time fuer finalize(). Das _had_events_in_window Flag
        wird in step() gesetzt wenn dt_overlap > 0.
        
        Parameter
        ---------
        current_time : float
            Aktuelle Simulationszeit [s]
        """
        self._last_event_time = current_time
    
    def check_first_event(self, current_time: float) -> None:
        """
        Prueft ob Nukleationsfenster vor erstem MC-Event abgeschlossen wurde.
        
        Wird bei count==0 (erstes MC-Event) aufgerufen. Falls das Fenster
        vor dem ersten Event endete, wird manuelle Nukleation ausgeloest.
        
        Parameter
        ---------
        current_time : float
            Zeitpunkt des ersten MC-Events [s]
        
        See Also
        --------
        _trigger_manual_full_window : Manuelle Ausloesung bei verpasstem Fenster
        """
        if not self.config.enabled:
            return
        
        # If first event is after window end, trigger manual nucleation
        if current_time >= self.config.liquid_addition_end:
            self._trigger_manual_full_window()
            self._manual_trigger_done = True
    
    def finalize_after_solve(self, final_time: float, event_count: int) -> None:
        """
        Finalisiert Nukleation nach Abschluss des MC-Loops.
        
        Behandelt drei Faelle:
        1. Keine MC-Events waehrend Simulation -> manueller Trigger mit voller Dauer
        2. Fenster endete nach letztem Event -> verbleibende Zeit verteilen
        3. Verbleibendes Volumen in _liquid_remainder -> immer verteilen
        
        Parameter
        ---------
        final_time : float
            Finale Simulationszeit [s]
        event_count : int
            Anzahl aufgetretener MC-Events
        
        See Also
        --------
        _trigger_manual_full_window : Manuelle Ausloesung
        _distribute_remaining_liquid : Restvolumen-Verteilung
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
            self._log(f"Distributing fractional droplet ({n_droplets_exact:.2f} × {v_droplet:.3e} m^3) at t={final_time:.4f}s", "DEBUG")
            
            # CRITICAL: Pass ACTUAL droplet volume and cap max_physical_droplets!
            # This ensures dW is capped correctly: dW <= n_droplets_exact / vc_scale
            # Without this cap, dW would be W[i]/2.0 (~46.5) and the particle would
            # receive dW × v_droplet = 46.5 × 5.236e-13 = 2.43e-11 m^3 (83x too much!)
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
                
                self._log(f"Distributed fractional droplet: {v_physical_distributed:.3e} m^3 (dW={dW:.4f}, effective_dW={effective_dW:.4f})", "DEBUG")
                
                self._log(f"Distributed fractional droplet: {v_physical_distributed:.3e} m^3 (dW={dW:.2f}, effective_dW={effective_dW:.2f})", "DEBUG")
            else:
                self._log("WARNING: Failed to distribute fractional droplet", "WARNING")
        
        # Sanity check
        if self._liquid_remainder < -1e-15:
            raise RuntimeError(
                f"Nucleation: _liquid_remainder became negative after final distribution "
                f"({self._liquid_remainder:.6e})"
            )
    
    def step(self, current_time: float, solver_last_dt: float) -> None:
        """
        Fuehrt einen Nukleationsschritt via Überlappungsberechnung aus.
        
        Wird nach jedem MC-Event aufgerufen. Berechnet das Überlappungsintervall
        zwischen Event-Intervall [t-dt, t] und Nukleationsfenster [start, end].
        
        Physikalisches Modell:
            Fluessigkeitsvolumen = flow_rate × dt_overlap
            wobei dt_overlap = Schnittlaenge der beiden Intervalle
        
        Parameter
        ---------
        current_time : float
            Aktuelle Simulationszeit NACH diesem MC-Event [s]
        solver_last_dt : float
            Zeit seit vorherigem MC-Event (= Intervall-Laenge) [s]
        
        See Also
        --------
        _distribute_liquid_volume : Verteilt berechnetes Volumen
        has_reached_target_wt : Prueft Ziel-wt% fuer Early Stop
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
        
        # Check target wt% for early stop (if configured)
        if self.config.target_wt_percent is not None:
            if self._should_check_wt(current_time) and self.has_reached_target_wt():
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
        """
        Loest manuelle Nukleation mit voller Fensterdauer aus.
        
        Aufgerufen wenn:
        - Erstes MC-Event nach Fensterende (Case C in check_first_event)
        - Keine MC-Events waehrend Simulation (Case 1 in finalize_after_solve)
        
        Verteilt das GESAMTE Fluessigkeitsvolumen fuer die gesamte Fensterdauer
        in einem Durchgang.
        
        Raises
        ------
        UserWarning
            Warnung ueber verpasstes Zeitfenster
        
        See Also
        --------
        check_first_event : Prueft verpasstes Fenster beim ersten Event
        finalize_after_solve : Finalisierung nach MC-Loop
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
        """
        Erstellt oder aktualisiert gewichteten Sampler fuer uniformes Partikel-Sampling.
        
        Physikalisches Modell:
            Um PHYSISCHE Partikel uniform zu samplen, werden COMPUTATIONALE Partikel
            proportional zu ihrem Gewicht W gesampelt:
            P(Partikel i) = W[i] / Sum(W)
        
        Performance:
            - Inkrementelle Updates: O(log n) bei Gewichtsaenderungen
            - Vollstaendiger Rebuild: O(n) nur bei Partikelanzahl-Änderung
        
        See Also
        --------
        fenwick_new.FenwickSampler : Implementierung des gewichteten Samplers
        _apply_pending_weight_updates : Inkrementelle Gewichtsaktualisierung
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
        Wendet akkumulierte Gewichtsaenderungen inkrementell am Sampler an.
        
        Performance:
            - Einzelnes Update: O(log n) statt O(n) fuer vollstaendigen Rebuild
            - Bei k Updates auf n Partikeln: O(k log n) vs O(k n)
        
        Wird automatisch von _ensure_samplers() aufgerufen wenn
        inkrementelle Updates aktiviert sind.
        
        See Also
        --------
        _record_weight_change : Zeichnet Gewichtsaenderung auf
        fenwick_new.FenwickSampler.update : Inkrementelle Fenwick-Update-Methode
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
        Zeichnet Gewichtsaenderung fuer spaetere inkrementelle Aktualisierung auf.
        
        Wird bei jeder Gewichtsaenderung waehrend Nukleation aufgerufen.
        Das eigentliche Sampler-Update wird bis zum naechsten Aufruf von
        _ensure_samplers() verzoegert (Batch-Verarbeitung).
        
        Parameter
        ---------
        idx : int
            Partikelindex dessen Gewicht sich geaendert hat
        new_weight : float
            Neuer absoluter Gewichtswert
        
        See Also
        --------
        _apply_pending_weight_updates : Wendet aufgezeichnete Änderungen an
        """
        if self._use_incremental_updates:
            self._pending_weight_updates.append((idx, new_weight))
    
    def _select_particle_uniform_physical(self) -> int:
        """
        Waehlt computationales Partikel sodass alle PHYSISCHEN gleiche Wahrscheinlichkeit haben.
        
        Physikalisches Modell:
            Sampling proportional zu W gewaehrleistet uniforme Verteilung
            ueber physikalische Partikel.
        
        Kernel-Framework:
            Bei vorhandenem liquid_dist_kernel wird dieser fuer erweiterte
            Strategien verwendet (surface-weighted, saturation-preferential).
        
        Returns
        -------
        int
            Index des gewaehlten Partikels, oder -1 falls kein gueltiges Partikel
        
        See Also
        --------
        kernels.liquid_distribution : Kernel fuer Partikelauswahl
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
        """
        Kern-Logik zur Verteilung von Fluessigkeitsvolumen als Tropfen.
        
        Physikalische Grundlagen:
            - Verteilt Volumen als diskrete Tropfen via DSMC-Sampling
            - Akkumuliert Restvolumen ueber Aufrufe hinweg (Massenerhaltung)
            - Vermeidet systematischen Verlust durch Integer-Diskretisierung
        
        Algorithmus:
            1. Neues Liquid zu _liquid_remainder addieren
            2. Anzahl ganzer Tropfen berechnen
            3. Tropfen mit Volumen-Tracking verteilen (nicht Count-Tracking!)
            4. Restliches Volumen als Mini-Tropfen verteilen (effektive_dW=1)
        
        Parameter
        ---------
        v_liquid_to_distribute : float
            Physikalisches Fluessigkeitsvolumen [m^3]
        is_manual_trigger : bool
            Wenn True, detaillierte Fortschrittsausgaben (fuer manuellen Trigger)
        
        See Also
        --------
        _distribute_one_droplet_with_dW : Verteilt einzelnen Tropfen
        _distribute_remaining_liquid : Verteilt Restvolumen am Simulationende
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

                # Compute saturation (every droplet!)
                sat_new = self._compute_saturation_for_liquid(
                    v_dry=v_dry_new,
                    poro=poro_new,
                    v_liquid=new_liquid,
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
        """
        Verteilt einen computionalen Tropfen auf Partikel.
        
        UNIFIED PATH fuer ALLE Tropfen (1., 2., 3., ...).
        
        Physikalisches Modell:
            Ein computational droplet Event verteilt einen einzelnen Tropfen des Volumens
            v_droplet auf gesammelte Partikel. Das Event-Gewicht dW bestimmt wie viele
            PHYSISCHEN Tropfen dies repraesentiert:
            
                n_physical_droplets = effective_dW = dW × (Vc_ref / Vc)
            
            Sampling erfolgt gewichtet nach W fuer uniforme Verteilung ueber physikalische
            Partikel.
        
        Stabilitaetskriterium (KRITISCH!):
            V_dry >= v_liquid  ->  Partikel kann Fluessigkeit tragen
            V_dry < v_liquid   ->  Zu viel Fluessigkeit! Agglomeration fuer mehr V_dry
            
            Warum V_dry (nicht V_solid)? Hochporoese Partikel koennen MEHR Fluessigkeit
            aufnehmen! V_dry = V_solid + V_pore beruecksichtigt Porenkapazitaet.
        
        Algorithmus:
            1. Partikel i gewichtet nach W auswaehlen (uniform ueber physikalische Partikel)
            2. Tropfen addieren: new_liquid = current_liquid + v_droplet
            3. SOLANGE V_dry < new_liquid:
                 - Mit Partikel j agglomerieren um V_dry zu erhoehen
                 - V_dry und new_liquid vom gemergten Partikel aktualisieren
            4. dW aus gesammeltem Partikelgewicht W_i berechnen
            5. Porositaet via PorosityGrowthKernel bestimmen (kernel-spezifisch!)
            6. Saettigung berechnen (jeder Tropfen!)
            7. Versuchen mit aehnlichem Partikel zu mergen (Optimierung)
            8. Neues Partikel erstellen oder Gewicht transferieren
            9. Eltern-Gewichte und Sampler aktualisieren
        
        Kernel-spezifisches Verhalten:
            - volume_mixing: Erster Kontakt bekommt poro=0.4 (erforderlich fuer Porositaet!)
            - cone_model: Startet mit poro=0.0, Porositaet waechst via ΔV bei Agglomeration
        
        Parameter
        ---------
        v_droplet : float
            Tropfenvolumen [m^3]
        max_physical_droplets : float, optional
            Limit fuer physikalische Tropfen. Cappt dW sodass:
            effective_dW = dW × vc_scale <= max_physical_droplets
        
        Returns
        -------
        float
            Fuer dieses Event verbrauchtes Gewicht dW
        
        See Also
        --------
        _perform_nucleation_agglomeration : Manuelle Agglomeration fuer V_dry-Sammlung
        _get_porosity_kernel : Holt Porositaets-Kernel
        _compute_saturation_for_liquid : Berechnet Saettigung
        """
        solver = self.solver
        a_tot = solver.a_tot
        
        if a_tot < 1:
            return 0.0
        
        # ==========================================
        # SCHRITT 1: Partikel auswaehlen
        # ==========================================
        i = self._select_particle_uniform_physical()
        
        if i < 0 or i >= a_tot:
            return 0.0
        
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
            if j < 0 or j >= a_tot:
                break
            
            # Manuelles Agglomerieren (mit PorosityGrowthKernel!)
            child_idx = self._perform_nucleation_agglomeration(i, j)
            if child_idx < 0 or child_idx >= solver.a_tot:
                break
            
            i = child_idx
            V_dry_sum = float(solver.V_flat[-1, i])
            # Das gemergte Partikel bringt seine eigene (Vor-Tropfen-)Fluessigkeit mit;
            # der anstehende Tropfen v_droplet muss weiterhin oben draufgerechnet werden,
            # sonst geht er aus dem Zielwert verloren (siehe Fundamentals.md Abschnitt 7).
            current_liquid = float(solver.liquid_volume[i])
            new_liquid = current_liquid + v_droplet

            attempts += 1
        
        # Check ob erfolgreich
        if V_dry_sum < new_liquid or i >= solver.a_tot:
            return 0.0  # Nicht genug Feststoff gefunden
        
        # ==========================================
        # SCHRITT 3: dW berechnen (DSMC-compliant)
        # ==========================================
        W_i = float(solver.W[i])
        if W_i <= 0:
            return 0.0
        
        # Effektive Batch-Groesse (Paper Gl. 33): der Event verbraucht GENAU dW,
        # deshalb raeumt der letzte Event ein Partikel exakt auf 0.0 leer und es
        # koennen keine Restgewichte entstehen.
        dW = min(self.config.batch_size, W_i)

        # Volumen-Deckelung. Frueher wurde hier dW auf den kontinuierlichen Quotienten
        # max_physical_droplets/vc_scale heruntergesetzt -- das war die Hauptquelle
        # fraktionaler Restgewichte.
        #
        # Der Deckel ist aber eine VOLUMEN-Bedingung, keine Gewichts-Bedingung: es soll
        # insgesamt max_physical_droplets * v_droplet an physikalischer Fluessigkeit
        # abgegeben werden. Statt das Paket zu verkleinern, wird deshalb die Menge PRO
        # Partikel verkleinert -- dW bleibt auf dem delta-Raster, und
        #
        #     dW * vc_scale * v_eff  ==  max_physical_droplets * v_droplet
        #
        # gilt exakt. Physikalisch ist das auch die richtige Lesart: ein Partikel
        # bekommt weniger Fluessigkeit, nicht "ein Bruchteil eines Partikels bekommt
        # einen vollen Tropfen".
        if max_physical_droplets is not None and max_physical_droplets > 0:
            vc_scale = self._Vc_reference / self.solver.Vc
            max_dW_allowed = max_physical_droplets / vc_scale
            if dW > max_dW_allowed:
                v_eff = v_droplet * max_dW_allowed / dW
                if v_eff <= 0.0:
                    return 0.0
                # new_liquid mit der kleineren Menge neu bilden. Die
                # Aufnahmefaehigkeit wurde oben mit dem GROESSEREN v_droplet geprueft,
                # bleibt also konservativ gueltig.
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
            v_dry=v_dry_new,
            poro=poro_new,
            v_liquid=new_liquid,
            is_first_contact=is_first_contact
        )
        
        # ==========================================
        # SCHRITT 6: Zielzustand einbuchen (mergen oder neu anlegen)
        # ==========================================
        # Geht ueber den gemeinsamen ParticleMerger -- Matching-Regeln und
        # Hash-Index sind damit dieselben wie bei Agglomeration und Breakage.
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

        return dW
    
    def _resolve_nucleation_geometry(self, i: int, new_liquid: float,
                                      is_first_contact: bool) -> tuple:
        """
        Bestimmt (V_dry, Porositaet) eines Partikels nach Tropfenauftrag.

        Zwei Faelle, physikalisch verschieden:

        1. **Partikel hat bereits Porenstruktur** (``poro_old > 0``):
           Benetzen fuegt Fluessigkeit hinzu, es verdichtet das Feststoffgeruest
           nicht. V_solid und V_pore bleiben, also bleibt auch V_dry und die
           Porositaet unveraendert. Die Fluessigkeit landet in ``liquid_volume``;
           der Internalisierungs-Kernel zieht sie ueber die Zeit in die Poren und
           ``saturation`` bildet den Fuellgrad ab. Porositaet = geometrischer
           Hohlraum, Saettigung = Fuellgrad -- zwei getrennte Groessen.

        2. **Partikel ist porenlos** (``poro_old == 0`` oder NaN):
           Echte Neu-Nukleation, der Kernel bestimmt die Startgeometrie.
           ``volume_mixing`` setzt seine 0.4-Saat (ohne die koennte durch reine
           Porenaddition nie Porositaet entstehen), ``cone_model`` startet bei
           0.0 und laesst Porositaet erst durch Agglomeration entstehen.

        Vorher lief JEDER Tropfen durch Fall 2. Da an
        ``compute_nucleation_porosity`` nur ``v_solid`` uebergeben wird, kam bei
        ``cone_model`` ``(v_solid, 0.0)`` zurueck -- ein Tropfen auf ein Partikel
        mit eps=0.8 loeschte dessen kompletten Porenraum und liess V_dry auf
        V_solid zusammenfallen. V_solid blieb erhalten (kein Massenfehler), aber
        Durchmesser, Rumpf-Festigkeit, Kompression und Internalisierung wurden
        dadurch entwertet.

        Parameter
        ---------
        i : int
            Index des getroffenen Partikels
        new_liquid : float
            Gesamtfluessigkeit des Partikels nach dem Tropfen [m^3]
        is_first_contact : bool
            True, wenn das Partikel vorher trocken war (nur fuer volume_mixing)

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

        # Fall 2: porenloses Partikel -> Kernel bestimmt die Startgeometrie.
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
    
    def _compute_saturation_for_liquid(self, v_dry: float, poro: float, 
                                        v_liquid: float, is_first_contact: bool) -> float:
        """
        Compute saturation for given liquid state.
        
        IMPORTANT: Called for EVERY droplet (not just first contact!).
        
        Physics:
            - If liquid_internalization_kernel active -> S=0 (all external)
            - Otherwise: Empirical split (split_ratio × liquid, capped at V_pore)
        
        Args:
            v_dry: Dry volume [m^3]
            poro: Porosity (may be NaN for legacy Vollkoerper)
            v_liquid: Total liquid volume [m^3]
            is_first_contact: True if this is the first droplet (not used currently,
                            but available for future kernel-specific logic)
        
        Returns:
            Saturation S ∈ [0, 1]
        """
        # Handle NaN or zero porosity
        if np.isnan(poro) or poro <= 0.0:
            return 0.0
        
        V_pore = v_dry * poro
        
        if V_pore <= 0:
            return 0.0
        
        # Check if internalization kernel is active
        has_internalization_kernel = (
            hasattr(self.solver, 'kernel_manager') and
            self.solver.kernel_manager is not None and
            self.solver.kernel_manager.liquid_internalization_kernel is not None
        )
        
        if has_internalization_kernel:
            # Kernel macht Internalization ueber Zeit -> hier alles external
            return 0.0
        
        # Kein Kernel: Empirischer Split (parameterized!)
        split_ratio = self._get_liquid_split_ratio(poro)
        
        V_liq_int_target = split_ratio * v_liquid
        V_liq_int = min(V_liq_int_target, V_pore)  # Cap at pore capacity!
        
        # S ∈ [0, 1]
        saturation = V_liq_int / V_pore
        
        return saturation
    
    def _get_liquid_split_ratio(self, poro: float) -> float:
        """
        Ermittelt Liquid-Split-Ratio aus Porositaets-Kernel.
        
        Bestimmt wie Fluessigkeit auf intern/extern aufgeteilt wird:
        - split_ratio = V_liq_int / V_liq_total
        - (1 - split_ratio) = externer Anteil
        
        Kernel-spezifische Werte:
            - volume_mixing: 0.4 (hardcodiert, empirisch)
            - cone_model: Aus Kernel-Params (Default: 0.4)
        
        Parameter
        ---------
        poro : float
            Aktuelle Porositaet (derzeit nicht verwendet, fuer Zukunft vorbehalten)
        
        Returns
        -------
        float
            Split-Ratio im Bereich [0, 1]
        
        See Also
        --------
        _compute_saturation_for_liquid : Verwendet Split-Ratio fuer Saettigung
        """
        porosity_kernel = self._get_porosity_kernel()
        
        if porosity_kernel.name == 'volume_mixing':
            # Volume Mixing: 0.4 hardcoded (legacy behavior)
            return 0.4
        elif porosity_kernel.name == 'cone_model':
            # HINWEIS: Der Default 0.4 ist fuer cone_model fragwuerdig. Der
            # Kernel erzeugt Porenraum geometrisch bei Kontakt und trifft
            # bewusst keine Aussage darueber, wie viel Fluessigkeit beim
            # Auftreffen sofort nach innen geht -- 0.0 waere die konsistente
            # Wahl. Frueher war der Wert ohnehin unerreichbar, weil
            # compute_nucleation_porosity fuer cone_model immer poro=0 lieferte
            # und _compute_saturation_for_liquid dann am poro<=0-Guard abbrach.
            # Seit die Porositaet bei Nucleation erhalten bleibt, ist der Pfad
            # erreichbar, sobald KEIN liquid_internalization-Kernel aktiv ist.
            # Nicht geaendert, weil das Ergebnisse verschiebt -- bewusst
            # entscheiden und dann hier festhalten.
            return porosity_kernel.params.get('liquid_split_ratio', 0.4)
        else:
            # Other kernels: Default 0.4
            return 0.4
    
    def _get_merger(self):
        """
        Liefert den ParticleMerger des Solvers, oder None.

        Die Nucleation nutzt bewusst denselben Merger wie Agglomeration und
        Breakage. Frueher hatte sie eine eigene ``_find_similar_particle``, die
        (a) das Feststoffvolumen gegen ``V_flat[-1]`` = V_dry verglich, also
        gegen die falsche Groesse, und (b) den Hash-Index weder fuellte noch
        aufraeumte -- Agglomeration und Breakage konnten die so erzeugten
        Partikel deshalb nie wiederfinden und legten Duplikate an.
        """
        return getattr(self.solver, '_particle_merger', None)

    def _remove_particle_tracked(self, idx: int) -> None:
        """
        Entfernt ein Partikel und haelt den Merger-Hash-Index konsistent.

        ``_remove_particle_column`` arbeitet mit Swap-with-last. Der Eintrag des
        entfernten Partikels muss VORHER aus dem Hash-Index, sonst zeigt er auf
        einen Slot, der danach von einem fremden Partikel belegt wird (der Swap
        selbst wird von ``notify_index_swap`` abgedeckt). Agglomeration und
        Breakage machen das an ihren Entfernstellen genauso.
        """
        merger = self._get_merger()
        if merger is not None:
            merger.remove_from_hash_index(idx)
        self.solver._remove_particle_column(idx)

    def _place_nucleated_state(self, src_idx: int, dW: float, v_dry: float,
                               poro: float, liquid: float, saturation: float,
                               v_droplet: Optional[float] = None) -> tuple:
        """
        Bucht den Zustand nach dem Tropfen ein: mergen oder neu anlegen.

        Geht ueber ``ParticleMerger.find_or_create``, damit Nucleation,
        Agglomeration und Breakage dieselbe Dedup-Logik und denselben
        Hash-Index benutzen.

        Parameter
        ---------
        src_idx : int
            Quellpartikel (liefert die Feststoffkomponenten, wird vom Matching
            ausgeschlossen)
        dW : float
            Gewicht, das dem Zielpartikel zugeschlagen wird
        v_dry, poro, liquid, saturation : float
            Zielzustand nach dem Tropfen
        v_droplet : float, optional
            Tropfenvolumen als Referenzmassstab fuer die Liquid-Toleranz
            (nur bei ``liquid_match_scale == "droplet"``)

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
        """
        Legt ein Partikel ohne Merger an (Fallback, wenn kein Merger existiert).

        Parameter
        ---------
        V_solid_src : np.ndarray
            Feststoffkomponenten des Quellpartikels
        dW : float
            Gewicht fuer das neue Partikel
        v_dry : float
            Trockenvolumen [m^3]
        poro : float
            Porositaet
        liquid : float
            Gesamtes Fluessigkeitsvolumen [m^3]
        saturation : float
            Berechnete Saettigung
        src_idx : int
            Quell-Partikelindex (nur fuer Debug-Ausgaben)
        """
        solver = self.solver

        # Create new particle
        solver._append_particle_column(V_solid_src)
        new_idx = solver.a_tot - 1

        # Set properties
        solver.V_flat[-1, new_idx] = v_dry  # ← V_dry from kernel!
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
        """
        Fuehrt Agglomeration zwischen Partikeln i und j fuer Nukleation durch.
        
        Ruft die echte Agglomerationslogik des Solvers auf um physikalische
        Konsistenz zu wahren (Fluessigkeitstransfer, Gewichts-Updates, etc.).
        
        Parameter
        ---------
        i : int
            Index erstes Partikel
        j : int
            Index zweites Partikel
        
        Returns
        -------
        int
            Index des Kind-Partikels nach Agglomeration
        
        See Also
        --------
        mcpbe_agg.MCPBEAgg._do_one_agg : Basis-Agglomerationsmethode
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
        Fuehrt manuelle Agglomeration zweier Partikel durch (ohne Propensity-Logik).
        
        Erstellt gemergtes Partikel aus i und j unter Erhaltung von Fluessigkeits-
        volumen und Massenerhaltung.
        
        WICHTIG: Returniert den tatsaechlichen Index des Kind-Partikels, der sich
        von (a_tot - 1) unterscheiden kann wenn ein Elternteil via swap-remove
        entfernt wurde!
        
        Volumen-Semantik:
            - V_flat[-1, :]: Speichert V_dry (= V_solid + V_pore)
            - V_flat[:dim, :]: Speichert V_solid-Komponenten (Multi-Component)
            - Fuer dim=1 (monodispers): V_flat[:dim, :] = V_solid
            - V_solid wird IMMER berechnet als: V_dry × (1 - Porositaet)
        
        Parameter
        ---------
        i : int
            Index erstes Partikel
        j : int
            Index zweites Partikel
        
        Returns
        -------
        int
            Index des Kind-Partikels nach Agglomeration
        
        See Also
        --------
        _perform_nucleation_agglomeration : Wrapper mit Nukleations-spezifischer Logik
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
        
        # Estimate collision energy for kernel (simplified)
        rho = 1000.0  # kg/m^3
        d_eff = float(solver.X[i] + solver.X[j]) * 0.5
        g = float(getattr(solver, 'G', 1000.0))
        v_rel = g * d_eff
        v_particle = float(V_dry_i + V_dry_j)
        m = rho * v_particle
        E_coll = 0.5 * m * v_rel ** 2
        
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
            collision_energy=E_coll,
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

        # Im Merger-Hash-Index registrieren. Das Kind wird hier bewusst direkt
        # angelegt statt ueber find_or_create, weil der folgende Code seinen
        # Index durch mehrere Eltern-Entfernungen hindurch nachverfolgen muss
        # (child_current_idx). Ohne Registrierung waere es fuer jede spaetere
        # Suche unsichtbar und Agglomeration/Breakage wuerden Duplikate anlegen.
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
        Aktualisiert internen Zustand nach wt%-Pruefung.
        
        Wird nach has_reached_target_wt() aufgerufen um Vorhersage zu aktualisieren.
        Passt Pruefintervall basierend auf Zielnaehe an:
        - >80% Ziel: Alle 10ms pruefen (konservativ)
        - >50% Ziel: Alle 50ms pruefen
        - <50% Ziel: Alle 100ms pruefen
        
        Parameter
        ---------
        current_time : float
            Aktuelle Simulationszeit [s]
        current_wt : float
            Aktueller wt%-Wert
        
        See Also
        --------
        has_reached_target_wt : Prueft Zielerreichung
        _should_check_wt : Bestimmt naechstes Pruefintervall
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
        Prueft ob Ziel-wt% erreicht wurde (falls konfiguriert).
        
        Returns
        -------
        bool
            True wenn target_wt_percent gesetzt und erreicht, sonst False
        
        See Also
        --------
        get_current_wt_percent : Berechnet aktuellen wt%-Wert
        _update_wt_check : Aktualisiert Praediktion nach Pruefung
        """
        if self.config.target_wt_percent is None:
            return False
        
        current_wt = self.get_current_wt_percent()
        
        # Update internal state for adaptive checking
        self._update_wt_check(self._current_time, current_wt)
        
        return current_wt >= self.config.target_wt_percent
    
    def get_statistics(self) -> dict:
        """
        Liefert Nukleations-Statistiken.
        
        Returns
        -------
        dict
            Dictionary mit Schluesselwerten:
            - current_time: Aktuelle Simulationszeit [s]
            - droplets_added_total: Gesamtzahl physikalischer Tropfen
            - liquid_volume_added_total: Gesamtes Fluessigkeitsvolumen [m^3]
            - current_wt_percent: Aktueller Fluessigkeitsanteil [%]
            - target_wt_percent: Ziel-wt% (oder None)
            - target_reached: Boolean ob Ziel erreicht
            - in_addition_window: Boolean ob im Zeitfenster
        
        See Also
        --------
        print_debug_status : Gibt Statistiken formatiert aus
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
            stats['volumetric_flow_rate'] = self.config.volumetric_flow_rate
        
        return stats
    
    def reset(self) -> None:
        """
        Setzt Nukleations-Zustand fuer wiederholte Simulationen.
        
        Wird nach solve() aufgerufen um Solver fuer naechsten Run vorzubereiten.
        Setzt alle Statistiken, Timer und Sampler zurueck.
        
        See Also
        --------
        MCPBESolver._reset_state : Ruft diese Methode auf
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
        
        # Reset adaptive wt% checking
        self._last_wt_check_time = -1.0
        self._last_wt_value = None
        self._wt_check_interval = 0.01
        
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
        """
        Gibt Debug-Status des Nukleationsprozesses aus.
        
        Ausgabe enthaelt:
        - Simulationszeit (absolut und % von total)
        - Echtzeit (Wall Clock) und Speedup-Faktor
        - Anzahl Tropfen (% vom Erwartungswert)
        - Partikelanzahl (computational und physikalisch)
        
        Parameter
        ---------
        force : bool
            Wenn True, Ausgabe unabhaengig vom Zeitintervall
        
        Notes
        -----
        Ausgabe erfolgt nur wenn config.debug=True oder solver.mcpbe_debug=True.
        Intervall zwischen Ausgaben: _debug_print_interval (Default: 0.1s)
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
    """
    Factory-Funktion zum Erstellen eines Nukleations-Handlers.
    
    Parameter
    ---------
    solver : MCPBESolver
        Solver-Instanz
    **kwargs : dict
        Argumente fuer NucleationConfig (enabled, volumetric_flow_rate, etc.)
    
    Returns
    -------
    NucleationHandler
        Konfigurierter Handler
    
    Example
    -------
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
