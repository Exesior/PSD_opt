"""
SIMPLE BREAKAGE TEST - Force breakage events to happen

This test uses VERY HIGH breakage rate to ensure events occur.
"""
import numpy as np


def test_legacy_breakage():
    """Test breakage with legacy implementation."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print("="*70)
    print("LEGACY BREAKAGE TEST")
    print("="*70)
    
    # Use VERY HIGH pl_P1 to force events
    solver = MCPBESolver(
        dim=1, 
        t_total=0.001,  # Very short time
        t_write=10, 
        init=False,  # Manual init
        load_attr=False,  # Don't load config
        seed=42
    )
    solver.COLEVAL = 1
    solver.BREAKRVAL = 1
    solver.G = 1000
    solver.pl_P1 = 100.0  # VERY HIGH! (vs 3e-4 original)
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    # Manual init sequence
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    a_tot_initial = solver.a_tot
    print(f"\nInitial state:")
    print(f"  a_tot: {a_tot_initial}")
    print(f"  pl_P1: {solver.pl_P1}")
    print(f"  Expected total rate: {a_tot_initial * solver.pl_P1:.2e} events/s")
    print(f"  Expected events in {solver.t_total}s: ~{a_tot_initial * solver.pl_P1 * solver.t_total:.1f}")
    
    # Solve
    solver.solve()
    
    a_tot_final = solver.a_tot
    print(f"\nFinal state:")
    print(f"  a_tot: {a_tot_final}")
    print(f"  Δ particles: {a_tot_final - a_tot_initial}")
    print(f"  real_break_events: {solver.real_break_events}")
    
    if a_tot_final > a_tot_initial:
        print(f"\n✓ SUCCESS: {a_tot_final - a_tot_initial} new particles created!")
        print(f"  Breakage events occurred!")
    else:
        print(f"\n✗ FAILURE: No breakage events!")
    
    return solver


def test_new_breakage():
    """Test breakage with new kernel framework."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print("\n" + "="*70)
    print("NEW KERNEL FRAMEWORK BREAKAGE TEST")
    print("="*70)
    
    solver = MCPBESolver(
        dim=1, 
        t_total=0.001,  # Very short time
        t_write=10, 
        init=False,  # Manual init
        load_attr=False,
        seed=42
    )
    solver.break_kernel_name = 'power_law'
    solver.break_kernel_params = {
        'p1': 100.0,  # VERY HIGH!
        'p2': 1.0,
        'g': 1000,
        'breakrval': 1
    }
    solver.porosity_growth_kernel_name = 'volume_mixing'
    solver.compression_kernel_name = 'exponential_decay'
    solver.compression_kernel_params = {'rate': 0.02, 'min_porosity': 0.3}
    solver.liquid_dist_kernel_name = 'uniform_weighted'
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    # Manual init sequence
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    a_tot_initial = solver.a_tot
    print(f"\nInitial state:")
    print(f"  a_tot: {a_tot_initial}")
    print(f"  pl_P1: {solver.pl_P1}")
    
    # Check rates
    if hasattr(solver, '_break_rate'):
        total_rate = np.sum(solver._break_rate[:a_tot_initial])
        print(f"  Total breakage rate: {total_rate:.2e} events/s")
        print(f"  Expected events in {solver.t_total}s: ~{total_rate * solver.t_total:.1f}")
    
    # Solve
    solver.solve()
    
    a_tot_final = solver.a_tot
    print(f"\nFinal state:")
    print(f"  a_tot: {a_tot_final}")
    print(f"  Δ particles: {a_tot_final - a_tot_initial}")
    print(f"  real_break_events: {solver.real_break_events}")
    
    if a_tot_final > a_tot_initial:
        print(f"\n✓ SUCCESS: {a_tot_final - a_tot_initial} new particles created!")
        print(f"  Breakage events occurred!")
    else:
        print(f"\n✗ FAILURE: No breakage events!")
    
    return solver


def test_manual_step_by_step():
    """Manual step-by-step breakage to see what happens."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print("\n" + "="*70)
    print("MANUAL STEP-BY-STEP BREAKAGE")
    print("="*70)
    
    solver = MCPBESolver(
        dim=1, 
        t_total=0.001, 
        t_write=10, 
        init=False, 
        load_attr=False,
        seed=42
    )
    solver.COLEVAL = 1
    solver.BREAKRVAL = 1
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
    
    print(f"\nInitial:")
    print(f"  a_tot: {solver.a_tot}")
    print(f"  _break_sampler.total(): {solver._break_sampler.total():.2e}")
    
    # Manually trigger one event
    print(f"\nManually triggering events...")
    for i in range(5):
        tau = solver._get_time_to_next_event()
        print(f"\nEvent {i+1}:")
        print(f"  Tau (time to next): {tau:.6f}s")
        
        if tau > solver.t_total:
            print(f"  → Tau exceeds t_total, stopping")
            break
        
        # Determine event type
        event_type = solver._get_event_type()
        print(f"  Event type: {event_type}")
        
        if event_type == 'break':
            k = solver._break_sampler.sample(solver._rng)
            print(f"  Selected particle: k={k}")
            print(f"  W[k] before: {solver.W[k]}")
            
            # Trigger breakage
            solver._do_one_break()
            
            print(f"  a_tot after: {solver.a_tot}")
            print(f"  W[k] after: {solver.W[k]}")
            print(f"  real_break_events: {solver.real_break_events}")
        else:
            print(f"  → Not a breakage event!")
            break
    
    return solver


# Run all tests
if __name__ == "__main__":
    legacy_solver = test_legacy_breakage()
    new_solver = test_new_breakage()
    manual_solver = test_manual_step_by_step()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nLegacy: {legacy_solver.a_tot} final particles")
    print(f"New:    {new_solver.a_tot} final particles")
    print(f"Manual: {manual_solver.a_tot} final particles")
    
    if legacy_solver.a_tot > 1000 or new_solver.a_tot > 1000:
        print(f"\n✓ Breakage IS working with high pl_P1!")
        print(f"  → Original test needs higher pl_P1 or longer t_total")
    else:
        print(f"\n✗ Breakage still not working!")
        print(f"  → Need to investigate further")
