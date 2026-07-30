"""
SIMPLE BREAKAGE TEST V2 - Test different BREAKRVAL values

BREAKRVAL=1: S = P1 (constant) - uses BREAKFVAL table
BREAKRVAL=4: S = P1 × G^P2 × V^(pl_v/3) - uses BREAKFVAL table  
BREAKRVAL=5: S = P1 × G^P2 × (V/V_ref)^pl_q - analytic fragments
"""
import numpy as np


def test_breakage_with_breakrval(breakrval, label=""):
    """Test breakage with specific BREAKRVAL."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print(f"\n{'='*70}")
    print(f"BREAKRVAL={breakrval} ({label})")
    print(f"{'='*70}")
    
    solver = MCPBESolver(
        dim=1, 
        t_total=0.001,
        t_write=10, 
        init=False,
        load_attr=False,
        seed=42
    )
    solver.COLEVAL = 1
    solver.BREAKRVAL = breakrval
    solver.G = 1000
    solver.pl_P1 = 100.0
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    # Check break config
    if hasattr(solver, '_break_config'):
        solver._prepare_break_config()
        print(f"Break config:")
        print(f"  _break_BREAKFVAL: {solver._break_BREAKFVAL}")
        print(f"  frag_num: {getattr(solver, 'frag_num', 'N/A')}")
    
    a_tot_initial = solver.a_tot
    print(f"\nInitial state:")
    print(f"  a_tot: {a_tot_initial}")
    print(f"  V[:3]: {solver.V_flat[-1, :3]}")
    
    # Check rates
    if hasattr(solver, '_break_rate'):
        total_rate = np.sum(solver._break_rate[:a_tot_initial])
        print(f"  Total rate: {total_rate:.2e} events/s")
    
    # Solve
    solver.solve()
    
    a_tot_final = solver.a_tot
    print(f"\nFinal state:")
    print(f"  a_tot: {a_tot_final}")
    print(f"  Δ particles: {a_tot_final - a_tot_initial}")
    print(f"  real_break_events: {solver.real_break_events}")
    
    # Check fragment volumes
    if a_tot_final > a_tot_initial:
        print(f"\n✓ SUCCESS: {a_tot_final - a_tot_initial} new particles!")
        print(f"  New particle volumes:")
        for i in range(a_tot_initial, min(a_tot_initial + 5, a_tot_final)):
            print(f"    Particle {i}: V={solver.V_flat[-1, i]:.6e}")
    else:
        print(f"\n✗ No new particles created!")
        print(f"  → Fragments likely filtered out (zero volume)")
    
    return solver


# Test different BREAKRVAL values
print("="*70)
print("BREAKRVAL COMPARISON")
print("="*70)

# BREAKRVAL=1: Constant rate, table-based fragments
solver1 = test_breakage_with_breakrval(1, "Constant rate, table fragments")

# BREAKRVAL=4: Volume power law, table-based fragments
solver4 = test_breakage_with_breakrval(4, "Volume power law, table fragments")

# BREAKRVAL=5: Modified power law, ANALYTIC fragments (no table!)
solver5 = test_breakage_with_breakrval(5, "Modified power law, ANALYTIC fragments")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"\nBREAKRVAL=1: {solver1.a_tot - 1000:+d} particles")
print(f"BREAKRVAL=4: {solver4.a_tot - 1000:+d} particles")
print(f"BREAKRVAL=5: {solver5.a_tot - 1000:+d} particles")

if solver5.a_tot > 1000:
    print(f"\n✓ BREAKRVAL=5 works! Uses analytic fragments (no NaN table).")
else:
    print(f"\n✗ None work! Need to investigate fragment generation.")
