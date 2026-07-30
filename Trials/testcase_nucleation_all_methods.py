"""
===============================================================================
Test Case: Nucleation - Alle Methoden im Vergleich
===============================================================================

Dieses Skript testet ALLE Nukleations-Methoden:
1. STANDARD: Sampler rebuild bei JEDEM Tropfen (korrekt, langsam)
2. CACHED: Sampler cached, rebuild nur nach Agglomeration (~100-1000x schneller)
3. DUAL-TIMESTEP: 2 Sampler/Zeitschritt + Exclusion-Tracking (am schnellsten!)

Verwendung:
-----------
    %run testcase_nucleation_all_methods.py
    
    Oder mit Argumenten:
    python testcase_nucleation_all_methods.py --method standard
    python testcase_nucleation_all_methods.py --method cached
    python testcase_nucleation_all_methods.py --method timestep
    python testcase_nucleation_all_methods.py --method all  # Alle vergleichen (default)

===============================================================================
"""

from __future__ import annotations

import sys
#import os
import time
from pathlib import Path

import numpy as np

# =============================================================================
# Import Setup - Handle different execution contexts
# =============================================================================

def setup_imports():
    """Setup Python path for importing mcpbe package."""
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()
    mcpbe_src = project_root / "mcpbe" / "src"
    
    paths_added = []
    for path in [str(mcpbe_src), str(project_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)
            paths_added.append(path)
    
    return paths_added

_paths_added = setup_imports()

# =============================================================================
# Solver Selection via Command Line
# =============================================================================

_METHOD_MAP = {
    "standard": ("MCPBESolver", None),
    "cached": ("MCPBESolverOpti", None),
    "timestep": ("MCPBESolverTimestep", None),
    "all": (None, None),
}

def get_solver_class(method_name: str):
    """Returns solver class for given method name."""
    try:
        from mcpbe import MCPBESolver, MCPBESolverOpti, MCPBESolverTimestep
        
        if method_name == "standard":
            return MCPBESolver
        elif method_name == "cached":
            if MCPBESolverOpti is None:
                print("WARNING: MCPBESolverOpti not available, falling back to MCPBESolver")
                return MCPBESolver
            return MCPBESolverOpti
        elif method_name == "timestep":
            if MCPBESolverTimestep is None:
                print("WARNING: MCPBESolverTimestep not available, falling back to MCPBESolver")
                return MCPBESolver
            return MCPBESolverTimestep
        else:
            return MCPBESolver
    except ImportError as e:
        print(f"ERROR: Could not import solver classes")
        print(f"Details: {e}")
        raise


# =============================================================================
# Test Configuration
# =============================================================================

class NucleationTestConfig:
    """Configuration for a single nucleation test case."""
    
    def __init__(
        self,
        name: str,
        process_type: str = "nucleation",
        dim: int = 1,
        t_total: float = 5.0,
        t_write: float = 0.5,
        
        # Initial particles
        initial_particles: int = 5000,
        particle_size: float = 27e-5,  # [m]
        
        # Nucleation parameters
        liquid_flow_rate: float = 0.000001,  # [L/s] = 1 µL/s
        droplet_diameter: float = 0.0005,  # [m] = 500 µm
        nucleation_enable: bool = True,
        
        # Seed for reproducibility
        seed: int = 42,
    ):
        self.name = name
        self.process_type = process_type
        self.dim = dim
        self.t_total = t_total
        self.t_write = t_write
        self.initial_particles = initial_particles
        self.particle_size = particle_size
        self.liquid_flow_rate = liquid_flow_rate
        self.droplet_diameter = droplet_diameter
        self.nucleation_enable = nucleation_enable
        self.seed = seed
    
    def create_solver(self, solver_class) -> object:
        """Create and configure a solver instance."""
        t_vec = np.arange(0.0, self.t_total + 0.5*self.t_write, self.t_write, dtype=float)
        
        solver = solver_class(
            dim=self.dim,
            t_vec=t_vec,
            verbose=False,
            load_attr=False,
            init=True,
            seed=self.seed,
        )
        
        solver.a0 = float(self.initial_particles)
        solver.x = np.full(solver.dim, self.particle_size)
        solver.process_type = self.process_type
        solver.liquid_flow_rate = self.liquid_flow_rate
        solver.droplet_diameter = self.droplet_diameter
        # IMPORTANT: Set nucleation_enable BEFORE _initialize_particles!
        solver.nucleation_enable = self.nucleation_enable
        
        solver._initialize_particles()
        
        # Initialize nucleation explicitly after particle init
        if self.nucleation_enable:
            solver.initialize_nucleation()
        
        return solver


# =============================================================================
# Test Runner
# =============================================================================

def run_single_test(method_name: str, config: NucleationTestConfig) -> dict:
    """Run a single test with specified method."""
    
    solver_class = get_solver_class(method_name)
    
    # Print method description
    descriptions = {
        "standard": "Standard: Sampler rebuild bei jedem Tropfen",
        "cached": "Cached: Sampler cached, rebuild nur nach Agglomeration",
        "timestep": "Dual-Timestep: 2 Sampler/Zeitschritt + Exclusion-Tracking",
    }
    
    print(f"\n{'='*70}")
    print(f"METHOD: {method_name.upper()}")
    print(f"Solver Class: {solver_class.__name__}")
    print(f"Strategie: {descriptions.get(method_name, 'Unknown')}")
    print(f"{'='*70}")
    
    solver = config.create_solver(solver_class)
    
    print(f"Initial particles: {solver.a_tot}")
    
    # Berechne erwartete Werte
    droplet_vol = (np.pi / 6.0) * (config.droplet_diameter ** 3)
    liquid_flow_m3 = config.liquid_flow_rate * 1e-3
    droplet_rate = liquid_flow_m3 / droplet_vol
    expected_droplets = int(droplet_rate * config.t_total)
    expected_total_liquid = droplet_rate * config.t_total * droplet_vol
    
    print(f"Erwartete Tropfen: ~{expected_droplets}")
    print(f"Erwartete Gesamtflüssigkeit: {expected_total_liquid*1e9:.4f} nL")
    print()
    
    # Start simulation
    start_time = time.perf_counter()
    solver.solve(maxiter=int(1e7))
    elapsed = time.perf_counter() - start_time
    
    # Collect results
    final_particles = solver.a_tot
    total_liquid = np.sum(solver.liquid_volume[:final_particles])
    
    # NEW: Get detailed nucleation stats
    total_droplets_distributed = getattr(solver, '_total_droplets_distributed', 0)
    accumulated_time = getattr(solver, '_accumulated_time', 0.0)
    droplet_rate = getattr(solver, '_droplet_rate', 0.0)
    droplet_vol = getattr(solver, '_droplet_volume', 0.0)
    
    results = {
        'method': method_name,
        'solver_class': solver_class.__name__,
        'time': elapsed,
        'final_particles': final_particles,
        'total_liquid': total_liquid,
        'total_droplets_distributed': total_droplets_distributed,
        'accumulated_time': accumulated_time,
        'droplet_rate': droplet_rate,
        'droplet_volume': droplet_vol,
        'expected_droplets': expected_droplets,
        'machine_time': getattr(solver, 'MACHINE_TIME', 0),
    }
    
    # Print results
    print(f"Ergebnisse:")
    print(f"  Rechenzeit: {elapsed:.3f} s")
    print(f"  Finale Partikel: {final_particles}")
    print(f"  Gesamt-Liquid: {total_liquid*1e9:.4f} nL")
    
    # Massenbilanz prüfen
    liq_error = abs(total_liquid - expected_total_liquid) / expected_total_liquid * 100 if expected_total_liquid > 0 else 0
    print(f"  Massenbilanz-Fehler: {liq_error:.4f}%")
    
    # DEBUG: Detailed nucleation statistics
    print(f"\n  [DEBUG] Nukleation-Statistiken:")
    print(f"    - Erwartete Tropfen: {expected_droplets}")
    print(f"    - Tatsächlich verteilt: {total_droplets_distributed}")
    print(f"    - Differenz: {total_droplets_distributed - expected_droplets:+d} ({(total_droplets_distributed/expected_droplets-1)*100:+.1f}%)")
    print(f"    - _accumulated_time (Rest): {accumulated_time:.9f} s")
    print(f"    - _droplet_rate: {droplet_rate:.2f} Tropfen/s")
    print(f"    - _droplet_volume: {droplet_vol:.3e} m³")
    print(f"    - Flüssigkeit aus Tropfen: {total_droplets_distributed * droplet_vol * 1e9:.4f} nL")
    
    return results


def run_comparison(method: str = "all"):
    """Führt Vergleichstest durch."""
    
    print("="*70)
    print("NUCLEATION METHODS COMPARISON")
    print("="*70)
    
    # Test-Konfiguration
    config = NucleationTestConfig(
        name="Nucleation Performance Test",
        process_type="nucleation",
        dim=1,
        t_total=5.0,
        t_write=0.5,
        initial_particles=5000,
        particle_size=27e-5,
        liquid_flow_rate=0.00001,
        droplet_diameter=0.0005,
        nucleation_enable=True,
        seed=42,
    )
    
    print(f"\nTest-Parameter:")
    print(f"  Dimension: {config.dim}")
    print(f"  Simulationszeit: {config.t_total} s")
    print(f"  Initiale Partikel: {config.initial_particles}")
    print(f"  Partikelgröße: {config.particle_size*1e6:.0f} µm")
    print(f"  Flüssigkeitsrate: {config.liquid_flow_rate*1e6:.1f} µL/s")
    print(f"  Tropfendurchmesser: {config.droplet_diameter*1e3:.1f} mm")
    print()
    
    results = {}
    
    if method == "all":
        methods = ["standard", "cached", "timestep"]
    else:
        methods = [method]
    
    for method_name in methods:
        try:
            results[method_name] = run_single_test(method_name, config)
        except Exception as e:
            print(f"\nFEHLER bei {method_name}: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {'error': str(e)}
    
    # =====================================================================
    # VERGLEICH
    # =====================================================================
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)
    
    if len(results) > 1:
        print(f"\n{'Methode':<20} | {'Zeit (s)':>10} | {'Partikel':>10} | {'Liquid (nL)':>12} | {'Fehler (%)':>10}")
        print("-"*70)
        
        # Referenz (standard) für Speedup-Berechnung
        ref_time = results.get('standard', {}).get('time', 1.0)
        
        for method_name in ["standard", "cached", "timestep"]:
            if method_name not in results or 'error' in results[method_name]:
                continue
            
            r = results[method_name]
            speedup = ref_time / r['time'] if r['time'] > 0 else 1.0
            
            # Massenbilanz-Fehler
            droplet_vol = (np.pi / 6.0) * (config.droplet_diameter ** 3)
            liquid_flow_m3 = config.liquid_flow_rate * 1e-3
            droplet_rate = liquid_flow_m3 / droplet_vol
            expected_liquid = droplet_rate * config.t_total * droplet_vol
            liq_error = abs(r['total_liquid'] - expected_liquid) / expected_liquid * 100 if expected_liquid > 0 else 0
            
            print(f"{method_name.upper():<20} | {r['time']:>10.3f} | {r['final_particles']:>10} | {r['total_liquid']*1e9:>12.4f} | {liq_error:>10.4f}")
        
        print()
        print("Speedup (relativ zu standard):")
        for method_name in ["cached", "timestep"]:
            if method_name not in results or 'error' in results[method_name]:
                continue
            r = results[method_name]
            speedup = ref_time / r['time'] if r['time'] > 0 else 1.0
            desc = "(2 Sampler/Zeitschritt)" if method_name == "timestep" else ""
            print(f"  {method_name.upper():>15}: {speedup:>8.1f}x schneller {desc}")
    else:
        # Nur eine Methode getestet
        method_name = list(results.keys())[0]
        r = results[method_name]
        if 'error' not in r:
            print(f"\n{method_name.upper()} Ergebnisse:")
            print(f"  Rechenzeit: {r['time']:.3f} s")
            print(f"  Finale Partikel: {r['final_particles']}")
            print(f"  Gesamt-Liquid: {r['total_liquid']*1e9:.4f} nL")
    
    print("\n" + "="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nucleation Methods Comparison")
    parser.add_argument(
        "--method",
        choices=["standard", "cached", "timestep", "all"],
        default="all",
        help="Welche Methode(n) testen"
    )
    
    args = parser.parse_args()
    run_comparison(method=args.method)
