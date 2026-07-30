"""
Debug Script: Understanding DSMC Nucleation Physics

CRITICAL QUESTION: What does liquid_volume represent in DSMC?

Option 1: liquid_volume = liquid PER COMPUTATIONAL particle
    Physical liquid = Σ(liquid_volume[i] × W[i]) / Vc
    
Option 2: liquid_volume = liquid PER PHYSICAL particle (represented by comp. particle)
    Physical liquid = Σ(liquid_volume[i] × W[i]) / Vc
    (same formula, different interpretation)

Current Implementation Analysis:
================================

When nucleation creates a child particle:
    solver.liquid_volume[child] = v_droplet  # Full droplet volume
    solver.W[child] = dW                      # Weight ~0.5

Physical liquid contribution:
    = v_droplet × dW / Vc

If dW = 0.5 and Vc = 1.0:
    = v_droplet × 0.5  ← Only HALF a droplet physically!

But we intended to add ONE droplet!

PROBLEM IDENTIFIED:
===================
The current code assigns liquid_volume = v_droplet to the child,
but the child only represents dW physical particles.

For dW < 1, this means:
- Child represents < 1 physical particles
- But carries liquid for 1 full droplet
- Physical liquid = v_droplet × dW (NOT v_droplet!)

CORRECT IMPLEMENTATION:
=======================

Method A: Scale liquid_volume by dW
    solver.liquid_volume[child] = v_droplet × dW
    → Physical liquid = (v_droplet × dW) × dW / Vc = v_droplet × dW² / Vc
    ← Still wrong!

Method B: Interpret dW as NUMBER of physical particles
    If dW = 0.5 means "0.5 physical particles nucleate"
    Then: liquid_volume = v_droplet (per physical particle)
    Physical liquid = v_droplet × dW / Vc
    ← This is what we have, but gives only partial droplet!

Method C: One droplet is distributed among dW/Vc particles
    Each particle gets: v_droplet / (dW/Vc) = v_droplet × Vc / dW
    liquid_volume[child] = v_droplet × Vc / dW
    Physical liquid = (v_droplet × Vc / dW) × dW / Vc = v_droplet ✓
    ← CORRECT but liquid_volume >> v_droplet for small dW!

Method D: Child represents ALL particles that received droplets
    dW/Vc physical particles each get one droplet
    Total liquid needed: (dW/Vc) × v_droplet
    liquid_volume[child] = (dW/Vc) × v_droplet × (Vc/dW) = v_droplet
    ← Wait, this is circular...

ACTUAL DSMC INTERPRETATION:
===========================

In DSMC, when we split a particle:
    Parent: W_parent → W_parent - dW (stays dry)
    Child:  W_child = dW (gets the droplet)

The child represents dW/Vc PHYSICAL particles.
These dW/Vc particles ALL receive droplets.
Total physical liquid: (dW/Vc) × v_droplet

For the weighted sum to give this:
    liquid_volume[child] × W[child] / Vc = (dW/Vc) × v_droplet
    liquid_volume[child] × dW / Vc = (dW/Vc) × v_droplet
    liquid_volume[child] = v_droplet ✓

SO THE CURRENT CODE IS CORRECT!

Then why the 8% error?

Possible causes:
1. Remainder not distributed (v_remainder in step())
2. Agglomeration changing weights incorrectly  
3. Multiple nucleations on same particle
4. Statistics counting wrong

DEBUG STEPS:
============
1. Print n_comp vs n_phys at each step
2. Track Σ(liquid_volume × W) manually
3. Compare to statistics
4. Check if agglomeration conserves liquid
"""

import numpy as np
import sys
sys.path.insert(0, '../')

from wmcpbe.mcpbe import MCPBESolver

def debug_nucleation():
    """Run nucleation with detailed debugging."""
    
    print(__doc__)
    
    solver = MCPBESolver(
        dim=1, t_total=1.0, t_write=10, verbose=False,
        load_attr=False, init=False, seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([5e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 100
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    # Initialize liquid
    solver.liquid_volume[:solver.a_tot] = 0.0
    
    print("\nInitial State:")
    print(f"  a_tot: {solver.a_tot}")
    print(f"  Sum(W): {np.sum(solver.W[:solver.a_tot]):.4f}")
    print(f"  Vc: {solver.Vc:.6e}")
    print(f"  n_phys: {np.sum(solver.W[:solver.a_tot]) / solver.Vc:.2f}")
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=1e-15,
        tropfen_durchmesser=1e-6,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=1.0,
    )
    solver.nucleation.configure_time_step(0.1)
    
    v_droplet = solver.nucleation.config.tropfen_volumen
    print(f"\nDroplet volume: {v_droplet:.6e} m³")
    
    # Manual stepping with debug output
    dt = 0.1
    time = 0.0
    
    print("\nManual Stepping:")
    print("-" * 70)
    
    cumulative_liquid_manual = 0.0
    step_count = 0
    
    while time < 1.0:
        # Trigger nucleation manually
        solver.nucleation.step(time, dt)
        
        n_comp = solver.a_tot
        sum_W = np.sum(solver.W[:n_comp])
        n_phys = sum_W / solver.Vc
        
        # Calculate physical liquid manually
        liquid_weighted = np.sum(solver.liquid_volume[:n_comp] * solver.W[:n_comp])
        liquid_physical = liquid_weighted / solver.Vc
        
        # Get statistics
        stats = solver.nucleation.get_statistics()
        liquid_stat = stats['liquid_volume_added_total']
        droplets_stat = stats['droplets_added_total']
        
        if step_count > 0:
            print(f"t={time:.1f}s: n_comp={n_comp:+4d} Δ | n_phys={n_phys:7.2f} | "
                  f"Liq(manual)={liquid_physical:8.6e} | Liq(stat)={liquid_stat:8.6e} | "
                  f"ΔLiq={liquid_physical-cumulative_liquid_manual:+8.6e}")
            cumulative_liquid_manual = liquid_physical
        
        step_count += 1
        time += dt
    
    print("-" * 70)
    
    # Final comparison
    n_final = solver.a_tot
    liquid_final = np.sum(solver.liquid_volume[:n_final] * solver.W[:n_final]) / solver.Vc
    stats_final = solver.nucleation.get_statistics()
    
    print(f"\nFinal Comparison:")
    print(f"  Liquid in system (weighted sum / Vc): {liquid_final:.6e} m³")
    print(f"  Liquid from statistics:               {stats_final['liquid_volume_added_total']:.6e} m³")
    print(f"  Difference:                           {liquid_final - stats_final['liquid_volume_added_total']:.6e} m³")
    print(f"  Relative error:                       {abs(liquid_final - stats_final['liquid_volume_added_total']) / max(liquid_final, 1e-30) * 100:.2f}%")
    
    # Check droplet count
    expected_droplets = 1e-15 / v_droplet
    print(f"\n  Expected droplets: {expected_droplets:.1f}")
    print(f"  Statistic droplets (effective dW): {stats_final['droplets_added_total']:.2f}")
    
    return liquid_final, stats_final['liquid_volume_added_total']

if __name__ == "__main__":
    debug_nucleation()
