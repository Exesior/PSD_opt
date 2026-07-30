"""
===============================================================================
Test: Cached-Sampler vs Dual-Timestep-Sampler Performance (Nucleation Only)
===============================================================================

Vergleicht die Performance der beiden optimierten Nukleations-Versionen:
- MCPBESolverOpti: Single-Sampler mit Caching (nach Agglomeration neu bauen)
- MCPBESolverTimestep: Dual-Timestep-Sampler (2 Sampler pro Zeitschritt, ohne Zurücklegen)

Parameter:
----------
dim=1, t_total=5.0s, initial_particles=5000, particle_size=27e-5 m
liquid_flow_rate=0.000001 L/s (1 µL/s), droplet_diameter=0.0005 m (500µm)

Verwendung:
-----------
    %run testcase_smplopt_vs_smpldual.py
    
Oder mit Kommandozeilenargumenten:
    python testcase_smplopt_vs_smpldual.py --mode opti      # Nur Opti
    python testcase_smplopt_vs_smpldual.py --mode timestep  # Nur Timestep
    python testcase_smplopt_vs_smpldual.py --mode both      # Beide vergleichen (default)

===============================================================================
"""

from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path

import numpy as np

# Setup imports
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
mcpbe_src = project_root / "mcpbe" / "src"

for path in [str(mcpbe_src), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from mcpbe import MCPBESolverOpti, MCPBESolverTimestep


def run_comparison(mode="both"):
    """Führt Vergleichstest durch.
    
    Parameters
    ----------
    mode : str
        "opti" - nur MCPBESolverOpti
        "timestep" - nur MCPBESolverTimestep  
        "both" - beide vergleichen (default)
    """
    
    print("="*70)
    print("Performance-Vergleich: Cached-Sampler vs Dual-Timestep-Sampler")
    print("="*70)
    
    # Test-Parameter
    params = {
        'dim': 1,
        't_total': 5.0,
        't_write': 0.5,
        'initial_particles': 5000,
        'particle_size': 27e-5,
        'liquid_flow_rate': 0.000001,
        'droplet_diameter': 0.0005,
        'nucleation_enable': True,
        'seed': 42,
    }
    
    t_vec = np.arange(0.0, params['t_total'] + 0.5*params['t_write'], 
                      params['t_write'], dtype=float)
    
    print(f"\nTest-Parameter:")
    print(f"  Dimension: {params['dim']}")
    print(f"  Simulationszeit: {params['t_total']} s")
    print(f"  Initiale Partikel: {params['initial_particles']}")
    print(f"  Partikelgröße: {params['particle_size']*1e6:.0f} µm")
    print(f"  Flüssigkeitsrate: {params['liquid_flow_rate']*1e6:.1f} µL/s")
    print(f"  Tropfendurchmesser: {params['droplet_diameter']*1e3:.1f} mm")
    print()
    
    results = {}
    
    # Berechne erwartete Werte (für beide gleich)
    droplet_vol = (np.pi / 6.0) * (params['droplet_diameter'] ** 3)
    liquid_flow_m3 = params['liquid_flow_rate'] * 1e-3
    droplet_rate = liquid_flow_m3 / droplet_vol
    expected_droplets = int(droplet_rate * params['t_total'])
    expected_total_liquid = droplet_rate * params['t_total'] * droplet_vol
    
    print(f"Erwartete Tropfen: ~{expected_droplets}")
    print(f"Erwartete Gesamtflüssigkeit: {expected_total_liquid*1e9:.4f} nL")
    print()
    
    # =====================================================================
    # TEST 1: MCPBESolverOpti (Cached-Sampler)
    # =====================================================================
    if mode in ["opti", "both"]:
        print("-"*70)
        print("TEST 1: MCPBESolverOpti (Cached-Sampler)")
        print("-"*70)
        
        solver_opti = MCPBESolverOpti(
            dim=params['dim'],
            t_vec=t_vec,
            verbose=False,
            load_attr=False,
            init=True,
            seed=params['seed'],
        )
        
        solver_opti.a0 = float(params['initial_particles'])
        solver_opti.x = np.full(params['dim'], params['particle_size'])
        solver_opti.process_type = "nucleation"
        solver_opti.liquid_flow_rate = params['liquid_flow_rate']
        solver_opti.droplet_diameter = params['droplet_diameter']
        solver_opti.nucleation_enable = params['nucleation_enable']
        
        solver_opti._initialize_particles()
        
        print(f"Initial: {solver_opti.a_tot} Partikel")
        print()
        
        start_time = time.perf_counter()
        solver_opti.solve(maxiter=int(1e7))
        elapsed_opti = time.perf_counter() - start_time
        
        results['opti'] = {
            'time': elapsed_opti,
            'final_particles': solver_opti.a_tot,
            'total_liquid': np.sum(solver_opti.liquid_volume[:solver_opti.a_tot]),
            'machine_time': getattr(solver_opti, 'MACHINE_TIME', 0),
        }
        
        print(f"Ergebnisse:")
        print(f"  Rechenzeit: {elapsed_opti:.3f} s")
        print(f"  Finale Partikel: {solver_opti.a_tot}")
        print(f"  Gesamt-Liquid: {results['opti']['total_liquid']*1e9:.2f} nL")
        
        # DEBUG: Prüfe Massenbilanz (OPTI)
        opti_accumulated = solver_opti._accumulated_time * droplet_rate
        opti_distributed = int(opti_accumulated)
        liq_error_opti = abs(results['opti']['total_liquid'] - expected_total_liquid) / expected_total_liquid * 100 if expected_total_liquid > 0 else 0
        
        print(f"\n  [DEBUG OPTI] Accumulated time: {solver_opti._accumulated_time:.6f} s")
        print(f"  [DEBUG OPTI] Calculated droplets (time*rate): {opti_accumulated:.2f}")
        print(f"  [DEBUG OPTI] Distributed droplets: {opti_distributed}")
        print(f"  [DEBUG OPTI] Expected liquid: {expected_total_liquid*1e9:.4f} nL")
        print(f"  [DEBUG OPTI] Actual liquid:   {results['opti']['total_liquid']*1e9:.4f} nL")
        print(f"  [DEBUG OPTI] Liquid diff:     {(results['opti']['total_liquid'] - expected_total_liquid)*1e9:.4f} nL ({liq_error_opti:.4f}%)")
        print()
    else:
        print("-"*70)
        print("TEST 1: MCPBESolverOpti (SKIPPED)")
        print("-"*70)
        print()
    
    # =====================================================================
    # TEST 2: MCPBESolverTimestep (Dual-Timestep-Sampler)
    # =====================================================================
    if mode in ["timestep", "both"]:
        print("-"*70)
        print("TEST 2: MCPBESolverTimestep (Dual-Timestep-Sampler)")
        print("-"*70)
        print("  Strategie:")
        print("  - 2 Sampler pro Zeitschritt (i=groß bevorzugt, j=klein bevorzugt)")
        print("  - Exclusion-Tracking verhindert doppelte Auswahl")
        print()
        
        solver_timestep = MCPBESolverTimestep(
            dim=params['dim'],
            t_vec=t_vec,
            verbose=False,
            load_attr=False,
            init=True,
            seed=params['seed'],
        )
        
        solver_timestep.a0 = float(params['initial_particles'])
        solver_timestep.x = np.full(params['dim'], params['particle_size'])
        solver_timestep.process_type = "nucleation"
        solver_timestep.liquid_flow_rate = params['liquid_flow_rate']
        solver_timestep.droplet_diameter = params['droplet_diameter']
        solver_timestep.nucleation_enable = params['nucleation_enable']
        
        solver_timestep._initialize_particles()
        
        print(f"Initial: {solver_timestep.a_tot} Partikel")
        print()
        
        start_time = time.perf_counter()
        solver_timestep.solve(maxiter=int(1e7))
        elapsed_timestep = time.perf_counter() - start_time
        
        results['timestep'] = {
            'time': elapsed_timestep,
            'final_particles': solver_timestep.a_tot,
            'total_liquid': np.sum(solver_timestep.liquid_volume[:solver_timestep.a_tot]),
            'machine_time': getattr(solver_timestep, 'MACHINE_TIME', 0),
        }
        
        print(f"Ergebnisse:")
        print(f"  Rechenzeit: {elapsed_timestep:.3f} s")
        print(f"  Finale Partikel: {solver_timestep.a_tot}")
        print(f"  Gesamt-Liquid: {results['timestep']['total_liquid']*1e9:.2f} nL")
        
        # DEBUG: Prüfe Massenbilanz (TIMESTEP)
        timestep_accumulated = solver_timestep._accumulated_time * droplet_rate
        timestep_distributed = int(timestep_accumulated)
        liq_error_timestep = abs(results['timestep']['total_liquid'] - expected_total_liquid) / expected_total_liquid * 100 if expected_total_liquid > 0 else 0
        
        print(f"\n  [DEBUG TIMESTEP] Accumulated time: {solver_timestep._accumulated_time:.6f} s")
        print(f"  [DEBUG TIMESTEP] Calculated droplets (time*rate): {timestep_accumulated:.2f}")
        print(f"  [DEBUG TIMESTEP] Distributed droplets: {timestep_distributed}")
        print(f"  [DEBUG TIMESTEP] Expected liquid: {expected_total_liquid*1e9:.4f} nL")
        print(f"  [DEBUG TIMESTEP] Actual liquid:   {results['timestep']['total_liquid']*1e9:.4f} nL")
        print(f"  [DEBUG TIMESTEP] Liquid diff:     {(results['timestep']['total_liquid'] - expected_total_liquid)*1e9:.4f} nL ({liq_error_timestep:.4f}%)")
        print()
    else:
        print("-"*70)
        print("TEST 2: MCPBESolverTimestep (SKIPPED)")
        print("-"*70)
        print()
    
    # =====================================================================
    # VERGLEICH
    # =====================================================================
    print("="*70)
    print("VERGLEICH")
    print("="*70)
    
    if mode == "both" and 'opti' in results and 'timestep' in results:
        elapsed_opti = results['opti']['time']
        elapsed_timestep = results['timestep']['time']
        
        speedup = elapsed_opti / elapsed_timestep if elapsed_timestep > 0 else float('inf')
        
        print(f"Rechenzeit Opti:       {elapsed_opti:.3f} s")
        print(f"Rechenzeit Timestep:   {elapsed_timestep:.3f} s")
        print(f"Speedup Timestep:      {speedup:.2f}x {'schneller' if speedup > 1 else 'langsamer'}")
        print()
        
        # Massenbilanz prüfen
        liq_diff = abs(results['opti']['total_liquid'] - results['timestep']['total_liquid'])
        liq_avg = (results['opti']['total_liquid'] + results['timestep']['total_liquid']) / 2
        rel_diff = (liq_diff / liq_avg * 100) if liq_avg > 0 else 0
        
        liq_error_opti = abs(results['opti']['total_liquid'] - expected_total_liquid) / expected_total_liquid * 100 if expected_total_liquid > 0 else 0
        liq_error_timestep = abs(results['timestep']['total_liquid'] - expected_total_liquid) / expected_total_liquid * 100 if expected_total_liquid > 0 else 0
        
        print(f"Massenbilanz (vs. erwartete Gesamtmenge {expected_total_liquid*1e9:.4f} nL):")
        print(f"  Liquid Opti:       {results['opti']['total_liquid']*1e9:.4f} nL ({liq_error_opti:.4f}% Fehler)")
        print(f"  Liquid Timestep:   {results['timestep']['total_liquid']*1e9:.4f} nL ({liq_error_timestep:.4f}% Fehler)")
        print(f"  Differenz Opti-Timestep: {rel_diff:.4f}%")
        print()
        
        if rel_diff < 1.0:
            print("✓ Massenbilanz konsistent (< 1% Abweichung zwischen Methoden)")
        else:
            print("⚠ Massenbilanz zeigt Abweichungen (> 1% zwischen Methoden)")
        
        # Finale Partikel
        part_diff = abs(results['opti']['final_particles'] - results['timestep']['final_particles'])
        print(f"\nFinale Partikel:")
        print(f"  Opti:       {results['opti']['final_particles']}")
        print(f"  Timestep:   {results['timestep']['final_particles']}")
        print(f"  Differenz:  {part_diff}")
        
    elif mode == "opti":
        print("Nur Opti wurde getestet (use --mode both für Vergleich)")
    elif mode == "timestep":
        print("Nur Timestep wurde getestet (use --mode both für Vergleich)")
    
    print()
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vergleich: Cached-Sampler vs Dual-Timestep-Sampler")
    parser.add_argument(
        "--mode",
        choices=["opti", "timestep", "both"],
        default="both",
        help="Welche Version(en) testen"
    )
    
    args = parser.parse_args()
    run_comparison(mode=args.mode)
