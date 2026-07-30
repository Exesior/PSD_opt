"""
Debug Mix Test - Show why only agglomeration happens.

This test prints detailed propensity information to understand
why breakage events might not be selected in mix mode.
"""

from __future__ import annotations

import numpy as np
import sys


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_mix_debug():
    """
    Mix test with detailed propensity tracking.
    """
    print_separator("MIX DEBUG: Why only Agglomeration?")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=3,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([3e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 200
    solver.process_type = "mix"
    
    # Agglomeration parameters
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-9
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Breakage parameters
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2
    solver.pl_v = 1.0
    solver.pl_q = 1.0
    solver.pl_P1 = 0.2  # Breakage rate
    solver.pl_P2 = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Assign liquid
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Process type: {solver.process_type}")
    
    # Check propensities BEFORE any events
    print(f"\n--- INITIAL PROPENSITIES ---")
    
    agg_prop = float(solver._agg_sampler.total()) if solver._agg_sampler else 0.0
    break_prop = float(solver._break_sampler.total()) if solver._break_sampler else 0.0
    total_prop = agg_prop + break_prop
    
    print(f"  Agg propensity:     {agg_prop:.6e}")
    print(f"  Break propensity:   {break_prop:.6e}")
    print(f"  Total propensity:   {total_prop:.6e}")
    
    if total_prop > 0:
        agg_prob = agg_prop / total_prop
        break_prob = break_prop / total_prop
        print(f"  P(Agg event):     {agg_prob*100:.2f}%")
        print(f"  P(Break event):   {break_prob*100:.2f}%")
    
    # Check breakage rates per particle
    print(f"\n--- BREAKAGE RATES PER PARTICLE ---")
    if hasattr(solver, '_break_rate') and solver._break_rate is not None:
        for i in range(min(5, n_init)):
            print(f"  Particle {i}: rate={solver._break_rate[i]:.6e}, W={solver.W[i]:.2f}")
        if n_init > 5:
            print(f"  ... ({n_init - 5} more particles)")
    
    # Run events with detailed logging
    max_events = 100
    print(f"\n--- RUNNING {max_events} EVENTS ---")
    
    agg_count = 0
    break_count = 0
    
    for evt in range(max_events):
        if solver.a_tot < 1:
            print(f"  Stopped: no particles left")
            break
        
        # Re-check propensities
        agg_prop = float(solver._agg_sampler.total()) if solver._agg_sampler else 0.0
        break_prop = float(solver._break_sampler.total()) if solver._break_sampler else 0.0
        total_prop = agg_prop + break_prop
        
        if total_prop <= 0:
            print(f"  Event {evt}: No propensities > 0, stopping")
            break
        
        agg_prob = agg_prop / total_prop
        rand_val = solver._rng.random()
        
        # Decide event type
        if rand_val < agg_prob:
            event_type = "AGG"
            agg_count += 1
            solver._do_one_agg()
        else:
            event_type = "BREAK"
            break_count += 1
            solver._do_one_break()
        
        # Log every event
        print(f"  Event {evt:2d}: {event_type:5s} | rand={rand_val:.3f}, "
              f"P(agg)={agg_prob*100:5.1f}% | particles={solver.a_tot:3d}")
    
    print(f"\n--- SUMMARY ---")
    print(f"  Total events:  {agg_count + break_count}")
    print(f"  Agg events:    {agg_count} ({agg_count/(agg_count+break_count)*100:.1f}%)")
    print(f"  Break events:  {break_count} ({break_count/(agg_count+break_count)*100:.1f}%)")
    
    # Analysis
    print(f"\n--- ANALYSIS ---")
    if break_count == 0:
        print("  ❌ NO BREAKAGE EVENTS occurred!")
        print("\n  Possible reasons:")
        print("    1. Breakage propensity too low compared to agglomeration")
        print("    2. BREAKRVAL=1 uses size-independent rate (may be ~0)")
        print("    3. pl_P1=0.2 is too small for these particle sizes")
        print("\n  Suggested fixes:")
        print("    - Increase pl_P1 (e.g., 1.0 or higher)")
        print("    - Use BREAKRVAL=2 for size-dependent breakage")
        print("    - Decrease CORR_BETA to reduce agglomeration dominance")
    else:
        print("  ✅ Breakage events occurred as expected")
    
    return True


def run_mix_fixed():
    """
    Mix test with better balanced parameters.
    """
    print_separator("MIX FIXED: Better Balanced Parameters")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=3,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([3e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 25
    solver.process_type = "mix"
    
    # REDUCED agglomeration
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-12  # Much smaller!
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # INCREASED breakage
    solver.BREAKRVAL = 2  # Size-dependent
    solver.BREAKFVAL = 2
    solver.pl_v = 1.0
    solver.pl_q = 1.0
    solver.pl_P1 = 2.0  # Higher breakage rate!
    solver.pl_P2 = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    
    # Initial propensities
    agg_prop = float(solver._agg_sampler.total()) if solver._agg_sampler else 0.0
    break_prop = float(solver._break_sampler.total()) if solver._break_sampler else 0.0
    total_prop = agg_prop + break_prop
    
    print(f"\n  Agg propensity:   {agg_prop:.6e}")
    print(f"  Break propensity: {break_prop:.6e}")
    if total_prop > 0:
        print(f"  P(Agg):  {agg_prop/total_prop*100:.1f}%")
        print(f"  P(Break):{break_prop/total_prop*100:.1f}%")
    
    # Run events
    max_events = 100
    print(f"\nRunning {max_events} events...")
    
    agg_count = 0
    break_count = 0
    
    for evt in range(max_events):
        if solver.a_tot < 1:
            break
        
        agg_prop = float(solver._agg_sampler.total()) if solver._agg_sampler else 0.0
        break_prop = float(solver._break_sampler.total()) if solver._break_sampler else 0.0
        total_prop = agg_prop + break_prop
        
        if total_prop <= 0:
            break
        
        if solver._rng.random() < (agg_prop / total_prop):
            agg_count += 1
            solver._do_one_agg()
        else:
            break_count += 1
            solver._do_one_break()
        
        if evt % 5 == 0:
            liq = np.sum(solver.liquid_volume[:solver.a_tot] * solver.W[:solver.a_tot])
            print(f"  Event {evt:2d}: particles={solver.a_tot:3d}, liquid={liq:.6e}")
    
    print(f"\n--- RESULTS ---")
    print(f"  Agg events:   {agg_count} ({agg_count/(agg_count+break_count)*100:.1f}%)")
    print(f"  Break events: {break_count} ({break_count/(agg_count+break_count)*100:.1f}%)")
    
    # Mass conservation check
    n_final = solver.a_tot
    initial_liquid = 0.1 * np.sum(solver.V_flat[-1, :n_init] * solver.W[:n_init])
    final_liquid = np.sum(solver.liquid_volume[:n_final] * solver.W[:n_final])
    liq_error = abs(final_liquid - initial_liquid) / max(abs(initial_liquid), 1e-300)
    
    print(f"\n  Liquid error: {liq_error:.2e} {'✅' if liq_error < 1e-10 else '❌'}")
    
    return break_count > 0 and liq_error < 1e-10


def main():
    print("\n" + "#" * 70)
    print("# MIX TEST DEBUG: Why only Agglomeration?")
    print("#"*70)
    
    # Part 1: Debug original parameters
    run_mix_debug()
    
    # Part 2: Try with fixed parameters
    try:
        success = run_mix_fixed()
        print(f"\n{'='*70}")
        if success:
            print("FIXED VERSION: ✅ Both event types + mass conservation!")
        else:
            print("FIXED VERSION: ⚠️ Still issues")
        print("="*70)
    except Exception as e:
        print(f"\n❌ Fixed version EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
