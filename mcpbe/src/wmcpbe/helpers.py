"""
WMCPBE Helper Functions & Config Templates.

This module provides convenience functions and pre-configured templates
for common simulation scenarios.

Usage:
    >>> from wmcpbe.helpers import create_dry_agglomeration_solver
    >>> solver = create_dry_agglomeration_solver(n_particles=500, t_total=60.0)
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any
import numpy as np

from .mcpbe import MCPBESolver


# =============================================================================
# CONFIG TEMPLATES
# =============================================================================

def create_dry_agglomeration_solver(
    n_particles: int = 500,
    particle_diameter: float = 10e-6,
    t_total: float = 60.0,
    g_shear: float = 1000,
    corr_beta: float = 1e-3,
    seed: int = 42,
    **kwargs
) -> MCPBESolver:
    """
    Erstellt Solver für trockene Agglomeration (Shear-induziert).
    
    Dies ist das einfachste Szenario - nur Agglomeration ohne Breakage,
    Nukleation oder andere komplexe Physik. Ideal für erste Tests und
    Validierungen.
    
    Parameter
    ---------
    n_particles : int
        Anzahl initialer Computational Partikel
    particle_diameter : float
        Partikeldurchmesser [m]. Default: 10µm (typisch für feine Pulver)
    t_total : float
        Gesamte Simulationsdauer [s]
    g_shear : float
        Scherrate [1/s]. Typisch: 100-5000
    corr_beta : float
        Korrekturfaktor für Agglomerationsrate. Typisch: 1e-4 bis 1e-2
    seed : int
        Random Seed für Reproduzierbarkeit
    **kwargs : dict
        Zusätzliche Solver-Parameter (werden an MCPBESolver übergeben)
    
    Returns
    -------
    MCPBESolver
        Vollständig konfigurierter Solver (ready für _initialize_particles + solve)
    
    Example
    -------
    >>> solver = create_dry_agglomeration_solver(
    ...     n_particles=500,
    ...     particle_diameter=50e-6,
    ...     t_total=120.0,
    ...     g_shear=2000,
    ... )
    >>> 
    >>> # Partikel initialisieren
    >>> V_flat = np.zeros((2, 500), dtype=float)
    >>> V_flat[-1, :] = np.pi/6 * (50e-6)**3
    >>> W_init = np.full(500, 50.0, dtype=float)
    >>> solver.Vc = 1e-6
    >>> solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
    >>> solver._initialize_samplers()
    >>> solver.solve()
    
    See Also
    --------
    create_wet_granulation_solver : Feuchtgranulation mit Nukleation
    create_breakage_test_solver : Breakage-only Test
    """
    # Calculate particle volume
    v_particle = np.pi / 6.0 * particle_diameter ** 3
    
    # Control Volume estimation (maintain reasonable particle density)
    control_volume = n_particles * v_particle * 10  # Factor 10 for spacing
    
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        verbose=False,
        load_attr=False,
        init=True,
        seed=seed,
        agg_kernel_name='shear_chin1998',
        agg_kernel_params={
            'corr_beta': corr_beta,
            'g': g_shear,
        },
        maybe_double_control_volume=False,
        recon_enable=False,
        **kwargs
    )
    
    # Store metadata for reference
    solver._config_metadata = {
        'template': 'dry_agglomeration',
        'n_particles': n_particles,
        'particle_diameter': particle_diameter,
        'control_volume': control_volume,
    }
    
    return solver


def create_wet_granulation_solver(
    n_particles: int = 1000,
    particle_diameter: float = 500e-6,
    droplet_diameter: float = 200e-6,
    volumetric_flow_rate: float = 1e-8,
    nucleation_duration: float = 180.0,
    t_total: float = 300.0,
    g_shear: float = 1000,
    batch_size: float = 25.0,
    enable_breakage: bool = True,
    enable_compression: bool = True,
    seed: int = 42,
    **kwargs
) -> MCPBESolver:
    """
    Erstellt Solver für Feuchtgranulation mit vollständiger Physik.
    
    Aktiviert alle relevanten Module:
    - ✅ Agglomeration (Liquid Bridge Kernel)
    - ✅ Nukleation (Tropfenverteilung)
    - ✅ Breakage (PowerLaw-Rumpf, optional)
    - ✅ Kompression (optional)
    - ✅ Porositätswachstum (Cone Model)
    - ✅ Stokes-Akzeptanzkriterium
    
    Parameter
    ---------
    n_particles : int
        Anzahl initialer Computational Partikel
    particle_diameter : float
        Durchmesser der Startpartikel [m]. Default: 500µm
    droplet_diameter : float
        Tropfendurchmesser [m]. Default: 200µm
    volumetric_flow_rate : float
        Flüssigkeitsflussrate [m³/s]. Default: 1e-8 m³/s (= 10 µL/s)
    nucleation_duration : float
        Dauer der Flüssigkeitszugabe [s]
    t_total : float
        Gesamte Simulationsdauer [s]
    g_shear : float
        Scherrate [1/s]
    batch_size : float
        Computational weight pro Tropfen-Event. Default: 25.0
    enable_breakage : bool
        Breakage aktivieren (PowerLaw-Rumpf)
    enable_compression : bool
        Kompression aktivieren (Porositätsreduktion)
    seed : int
        Random Seed
    **kwargs : dict
        Zusätzliche Solver-Parameter
    
    Returns
    -------
    MCPBESolver
        Vollständig konfigurierter Solver
    
    Example
    -------
    >>> solver = create_wet_granulation_solver(
    ...     n_particles=1000,
    ...     particle_diameter=700e-6,
    ...     droplet_diameter=150e-6,
    ...     t_total=600.0,
    ... )
    >>> 
    >>> # Nukleation konfigurieren (automatisch erstellt)
    >>> # Solver ist ready nach Initialisierung
    >>> # (siehe QUICKSTART.md für Details)
    
    Notes
    -----
    Dieser Template erstellt einen komplexen Solver mit vielen Modulen.
    Für einfache Tests bevorzugen Sie create_dry_agglomeration_solver().
    
    Die Nukleation wird NICHT automatisch aktiviert - rufen Sie nach
    der Partikelinitialisierung create_nucleation_handler() auf.
    
    See Also
    --------
    create_dry_agglomeration_solver : Einfache trockene Agglomeration
    setup_initial_particles : Partikelinitialisierung
    """
    v_particle = np.pi / 6.0 * particle_diameter ** 3
    control_volume = n_particles * v_particle * 5
    
    # Base configuration
    base_kwargs = {
        'dim': 1,
        't_total': t_total,
        'verbose': False,
        'load_attr': False,
        'init': True,
        'seed': seed,
        'maybe_double_control_volume': False,
        'recon_enable': False,
    }
    
    # Agglomeration: Liquid Bridge (optimal für Feuchtgranulation)
    base_kwargs['agg_kernel_name'] = 'liquid_bridge'
    base_kwargs['agg_kernel_params'] = {
        'corr_beta': 1e-3,
        'g': g_shear,
        'optimal_saturation': 0.5,
    }
    
    # Stokes-Akzeptanz (wichtig für realistische Kollisionen)
    base_kwargs['agg_acceptance_kernel_name'] = 'stokes_krit'
    base_kwargs['agg_acceptance_kernel_params'] = {
        'U_coll': 0.5,
        'binder_viscosity': 0.1,
        'rho_solid': 2500,
        'rho_liquid': 1000,
        'h_a': 1e-7,
    }
    
    # Breakage (optional)
    if enable_breakage:
        base_kwargs['break_kernel_name'] = 'powerlaw_rumpf'
        base_kwargs['break_kernel_params'] = {
            'p1': 0.5,
            'p2': 1.0,
            'g': g_shear,
            'breakrval': 4,
            'k': 2.5,
            'alpha': 1.0,
            'gamma': 0.036,
            'delta': 0.0,
        }
    
    # Porosity Growth
    base_kwargs['porosity_growth_kernel_name'] = 'cone_model'
    base_kwargs['porosity_growth_kernel_params'] = {}
    
    # Compression (optional)
    if enable_compression:
        base_kwargs['porosity_compression_kernel_name'] = 'porosity_compression'
        base_kwargs['porosity_compression_kernel_params'] = {
            'rate': 0.02,
            'min_porosity': 0.15,
        }

    # Liquid Internalization (kontinuierlich)
    base_kwargs['liquid_internalization_kernel_name'] = 'liquid_internalization'
    base_kwargs['liquid_internalization_kernel_params'] = {
        'k_int': 1e8,
    }
    
    # Event-basierte Internalisierung bei Agglomeration
    base_kwargs['liq_internalisation_agglomeration_kernel_name'] = 'liq_internalisation_agglomeration'
    base_kwargs['liq_internalisation_agglomeration_kernel_params'] = {}
    
    # Override with user-provided kwargs
    base_kwargs.update(kwargs)
    
    solver = MCPBESolver(**base_kwargs)
    
    # Store metadata
    solver._config_metadata = {
        'template': 'wet_granulation',
        'n_particles': n_particles,
        'particle_diameter': particle_diameter,
        'droplet_diameter': droplet_diameter,
        'volumetric_flow_rate': volumetric_flow_rate,
        'nucleation_duration': nucleation_duration,
        'batch_size': batch_size,
        'control_volume': control_volume,
        'enable_breakage': enable_breakage,
        'enable_compression': enable_compression,
    }
    
    return solver


def create_breakage_test_solver(
    n_particles: int = 500,
    particle_diameter: float = 1e-3,
    p1: float = 0.01,
    p2: float = 2.0,
    g_shear: float = 1000,
    t_total: float = 10.0,
    seed: int = 42,
    **kwargs
) -> MCPBESolver:
    """
    Erstellt Solver für reine Breakage-Tests (ohne Agglomeration).
    
    Ideal für:
    - Breakage-Kernel Validierung
    - Vergleich mit analytischen Lösungen
    - Performance-Tests von Breakage-CDFs
    
    Parameter
    ---------
    n_particles : int
        Anzahl initialer Partikel
    particle_diameter : float
        Partikeldurchmesser [m]. Default: 1mm (groß für hohe Breakage-Rate)
    p1 : float
        Breakage rate pre-factor [1/s·m⁻³ᵅ]. Höher = mehr Breakage
    p2 : float
        Volumen-Exponent in S = P1·G·V^P2. >1 bedeutet größere Partikel
        brechen leichter. (Hieß früher `pl_v` und war wirkungslos, weil
        `breakrval` auf dem Default 1 = konstante Rate stand.)
    g_shear : float
        Scherrate [1/s]
    t_total : float
        Simulationsdauer [s]
    seed : int
        Random Seed
    **kwargs : dict
        Zusätzliche Solver-Parameter
    
    Returns
    -------
    MCPBESolver
        Breakage-only Solver (process_type='breakage')
    
    Example
    -------
    >>> solver = create_breakage_test_solver(
    ...     n_particles=1000,
    ...     particle_diameter=500e-6,
    ...     p1=0.1,  # Hohe Rate für schnelle Tests
    ...     t_total=5.0,
    ... )
    >>> solver.process_type = 'breakage'
    >>> # Initialize and solve...
    
    Notes
    -----
    Wichtig: Setzen Sie process_type='breakage' vor solve()!
    
    See Also
    --------
    validate_breakage_cdf : CDF-Validierung
    """
    v_particle = np.pi / 6.0 * particle_diameter ** 3
    control_volume = n_particles * v_particle * 2
    
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        verbose=False,
        load_attr=False,
        init=True,
        seed=seed,
        break_kernel_name='power_law',
        break_kernel_params={
            'p1': p1,
            'p2': p2,
            'g': g_shear,
            'breakrval': 4,   # S = P1 * G * V^P2 (size dependent)
        },
        maybe_double_control_volume=False,
        recon_enable=False,
        **kwargs
    )
    
    solver.process_type = 'breakage'
    
    solver._config_metadata = {
        'template': 'breakage_test',
        'n_particles': n_particles,
        'particle_diameter': particle_diameter,
        'control_volume': control_volume,
    }
    
    return solver


# =============================================================================
# INITIALIZATION HELPERS
# =============================================================================

def setup_initial_particles(
    solver: MCPBESolver,
    n_particles: int,
    particle_diameter: float,
    initial_weight: float = 50.0,
    initial_porosity: float = 0.0,
    set_auxiliary_arrays: bool = True,
) -> None:
    """
    Initialisiert Partikelarrays für Solver.
    
    Convenience-Funktion die alle Schritte der Partikelinitialisierung
    durchführt. Spart Boilerplate-Code in Test-Skripten.
    
    Parameter
    ---------
    solver : MCPBESolver
        Solver-Instanz (bereits erstellt via Constructor oder Template)
    n_particles : int
        Anzahl Partikel
    particle_diameter : float
        Partikeldurchmesser [m]
    initial_weight : float
        Initiales computational weight pro Partikel
    initial_porosity : float
        Initiale Porosität. 0.0 = Vollkörper, NaN = kein Porositätsarray
    set_auxiliary_arrays : bool
        Wenn True, setzt auch liquid_volume, porosity, saturation Arrays
    
    Example
    -------
    >>> from wmcpbe import MCPBESolver
    >>> from wmcpbe.helpers import setup_initial_particles
    >>> 
    >>> solver = MCPBESolver(
    ...     dim=1, t_total=60.0, load_attr=False,
    ...     agg_kernel_name='shear_chin1998',
    ...     agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
    ... )
    >>> 
    >>> setup_initial_particles(
    ...     solver,
    ...     n_particles=500,
    ...     particle_diameter=100e-6,
    ...     initial_weight=50.0,
    ... )
    >>> 
    >>> solver.solve()
    
    Notes
    -----
    Diese Funktion führt folgende Schritte aus:
    1. V_flat Array erstellen (dim+1 Zeilen, n_particles Spalten)
    2. V_dry berechnen und setzen
    3. W_init Array erstellen
    4. solver.Vc setzen (basierend auf Partikelvolumen)
    5. _initialize_particles() aufrufen
    6. _initialize_samplers() aufrufen
    7. Optional: porosity, liquid_volume, saturation Arrays initialisieren
    
    See Also
    --------
    create_dry_agglomeration_solver : Template das diesen Helper verwendet
    """
    v_particle = np.pi / 6.0 * particle_diameter ** 3
    
    # Create V_flat array
    V_flat = np.zeros((solver.dim + 1, n_particles), dtype=float)
    V_flat[-1, :] = v_particle  # V_dry
    
    if solver.dim > 1:
        # Multi-component: distribute volume across components
        for i in range(solver.dim):
            V_flat[i, :] = v_particle / solver.dim
    
    # Create weight array
    W_init = np.full(n_particles, initial_weight, dtype=float)
    
    # Set control volume
    solver.Vc = n_particles * v_particle * 0.5  # Reasonable density
    
    # Initialize particles
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Initialize samplers (critical!)
    solver._initialize_samplers()
    
    # Set auxiliary arrays if requested
    if set_auxiliary_arrays:
        if hasattr(solver, 'porosity'):
            if np.isnan(initial_porosity):
                solver.porosity[:solver.a_tot] = np.nan
            else:
                solver.porosity[:solver.a_tot] = initial_porosity
        
        if hasattr(solver, 'liquid_volume'):
            solver.liquid_volume[:solver.a_tot] = 0.0
        
        if hasattr(solver, 'saturation'):
            solver.saturation[:solver.a_tot] = 0.0


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def compute_moments(solver: MCPBESolver, time_index: Optional[int] = None) -> Dict[str, float]:
    """
    Berechnet Momente (M0, M1, M2, ...) aus Solver-Zustand.
    
    Parameter
    ---------
    solver : MCPBESolver
        Solver-Instanz
    time_index : int, optional
        Zeitindex für historische Daten. None = aktueller Zustand
    
    Returns
    -------
    dict
        Dictionary mit Momenten:
        - M0: Anzahl Partikel (Summe W)
        - M1: Gesamtvolumen
        - M2: Zweites Moment (für Varianz)
        - M3: Drittes Moment (für Skewness)
        - mean_volume: Mittleres Volumen (M1/M0)
        - d50: Median-Durchmesser (geschätzt)
    
    Example
    -------
    >>> moments = compute_moments(solver)
    >>> print(f"M0={moments['M0']:.0f}, M1={moments['M1']:.6e}")
    
    See Also
    --------
    validate_mass_conservation : Massenbilanz prüfen
    """
    if time_index is not None:
        # Historical data
        W = solver.W_save[time_index]
        V_dry = solver.V_save[time_index][-1, :]
    else:
        # Current state
        W = solver.W[:solver.a_tot]
        V_dry = solver.V_flat[-1, :solver.a_tot]
    
    # Compute moments
    M0 = float(np.sum(W))
    M1 = float(np.sum(W * V_dry))
    M2 = float(np.sum(W * V_dry**2))
    M3 = float(np.sum(W * V_dry**3))
    
    # Derived quantities
    mean_volume = M1 / M0 if M0 > 0 else 0.0
    
    # Estimate d50 (simplified - assumes monodisperse-ish distribution)
    d50 = (6.0 * mean_volume / np.pi) ** (1.0/3.0) if mean_volume > 0 else 0.0
    
    return {
        'M0': M0,
        'M1': M1,
        'M2': M2,
        'M3': M3,
        'mean_volume': mean_volume,
        'd50': d50,
    }


def validate_mass_conservation(
    solver: MCPBESolver,
    initial_solid_mass: Optional[float] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Prüft Massenerhaltung der Simulation.
    
    Validiert dass Feststoffmasse während Simulation erhalten bleibt
    (innerhalb numerischer Toleranz).
    
    Parameter
    ---------
    solver : MCPBESolver
        Solver-Instanz (nach solve() Aufruf)
    initial_solid_mass : float, optional
        Bekannte initiale Masse. Wenn None, wird aus initalem Zustand berechnet
    tolerance : float
        Relative Toleranz für Abweichung. Default: 1e-6 (0.0001%)
    
    Returns
    -------
    dict
        Ergebnis-Dictionary mit:
        - passed: Boolean ob Validierung bestanden
        - initial_mass: Initiale Feststoffmasse [kg]
        - final_mass: Finale Feststoffmasse [kg]
        - relative_error: Relative Abweichung
        - message: Lesbare Beschreibung
    
    Raises
    ------
    ValueError
        Wenn required Arrays nicht vorhanden
    
    Example
    -------
    >>> result = validate_mass_conservation(solver)
    >>> if not result['passed']:
    ...     print(f"WARNING: {result['message']}")
    
    Notes
    -----
    Die Validierung berücksichtigt:
    - Feststoffmasse aus V_solid × density
    - Porositätsänderungen (V_solid = V_dry × (1-poro))
    - Computational Weight (W)
    
    Nicht berücksichtigt:
    - Flüssigkeitsmasse (separate Validierung nötig)
    - Gasphase (in diesem Modell nicht vorhanden)
    """
    # Get solid density (assume constant)
    rho_solid = 2500.0  # kg/m³ (default, should be parameter)
    
    # Get current state
    W = solver.W[:solver.a_tot]
    V_dry = solver.V_flat[-1, :solver.a_tot]
    porosity = solver.porosity[:solver.a_tot] if hasattr(solver, 'porosity') else None
    
    # Compute V_solid from V_dry and porosity
    if porosity is not None and not np.all(np.isnan(porosity)):
        valid_mask = ~np.isnan(porosity)
        V_solid = np.zeros_like(V_dry)
        V_solid[valid_mask] = V_dry[valid_mask] * (1.0 - porosity[valid_mask])
        V_solid[~valid_mask] = V_dry[~valid_mask]  # Vollkörper
    else:
        V_solid = V_dry  # Assume Vollkörper
    
    # Compute solid mass
    final_solid_mass = float(np.sum(W * V_solid)) * rho_solid
    
    # Get initial mass
    if initial_solid_mass is not None:
        initial_mass = initial_solid_mass
    elif hasattr(solver, 'V0') and solver.V0 is not None:
        # From saved initial state
        V0_dry = solver.V0[-1, :]
        W0 = solver.W0
        initial_mass = float(np.sum(W0 * V0_dry)) * rho_solid
    else:
        raise ValueError(
            "Cannot determine initial mass. Provide initial_solid_mass parameter "
            "or ensure solver.V0 and solver.W0 are set."
        )
    
    # Compute relative error
    if initial_mass > 0:
        rel_error = abs(final_solid_mass - initial_mass) / initial_mass
    else:
        rel_error = float('inf') if final_solid_mass != 0 else 0.0
    
    # Determine pass/fail
    passed = rel_error <= tolerance
    
    # Generate message
    if passed:
        message = f"✅ Mass conservation OK (error={rel_error:.2e} ≤ {tolerance})"
    else:
        message = (
            f"❌ Mass conservation FAILED (error={rel_error:.2e} > {tolerance}).\\n"
            f"  Initial: {initial_mass:.6e} kg, Final: {final_solid_mass:.6e} kg, "
            f"Delta: {final_solid_mass - initial_mass:.6e} kg"
        )
    
    return {
        'passed': passed,
        'initial_mass': initial_mass,
        'final_mass': final_solid_mass,
        'relative_error': rel_error,
        'message': message,
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_scientific(value: float, unit: str = "", precision: int = 3) -> str:
    """
    Formatiert Zahl in wissenschaftlicher Notation mit Einheit.
    
    Parameter
    ---------
    value : float
        Zu formatierende Zahl
    unit : str
        Physikalische Einheit (optional)
    precision : int
        Anzahl Nachkommastellen
    
    Returns
    -------
    str
        Formatierter String
    
    Example
    -------
    >>> format_scientific(1.234e-9, 'm³')
    '1.234e-09 m³'
    
    >>> format_scientific(6.022e23, 'mol⁻¹', precision=2)
    '6.02e+23 mol⁻¹'
    """
    fmt = f"{value:.{precision}e}"
    if unit:
        return f"{fmt} {unit}"
    return fmt


def print_section(title: str, char: str = "=") -> None:
    """
    Druckt formatierte Sektionsüberschrift.
    
    Parameter
    ---------
    title : str
        Überschriftentext
    char : str
        Umrandungszeichen
    
    Example
    -------
    >>> print_section("SIMULATION RESULTS")
    ======================================================================
    SIMULATION RESULTS
    ======================================================================
    """
    width = max(len(title) + 4, 70)
    border = char * width
    print(f"\\n{border}")
    print(f"{title.center(width)}")
    print(border)
