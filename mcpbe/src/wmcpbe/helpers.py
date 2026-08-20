"""
WMCPBE Helper Functions & Config Templates.

This module provides convenience functions and pre-configured templates
for common simulation scenarios.

Usage:
    >>> from wmcpbe.helpers import setup_initial_particles

Note:
    The three `create_*_solver` config templates were removed. They wired a
    fixed set of kernels together for a fixed kind of case, and the cases in
    this project differ too much for that to have been useful - nothing ever
    called them. Build solvers explicitly, or via `framework.builder`.
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any
import numpy as np

from .mcpbe import MCPBESolver


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
