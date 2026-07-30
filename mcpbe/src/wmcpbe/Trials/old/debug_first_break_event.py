"""
DEBUG FIRST BREAKAGE EVENT - Compare exactly what happens during solve()

This logs:
1. Which particle is selected by the sampler
2. What break rate it has
3. How many fragments are created
4. What the fragment volumes are
"""
import numpy as np


def run_and_log_first_event(seed=42, label=""):
    """Run solver and log first breakage event."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print(f"\n{'='*70}")
    print(f"{label} CASE")
    print(f"{'='*70}")
    
    solver = MCPBESolver(dim=1, t_total=0.05, t_write=10, init=False, load_attr=False, seed=seed)
    solver.COLEVAL = 1
    solver.BREAKRVAL = 1
    solver.G = 1000
    solver.pl_P1 = 3e-4
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    # Store initial state
    a_tot_before = solver.a_tot
    W_before = solver.W[:a_tot_before].copy()
    V_before = solver.V_flat[-1, :a_tot_before].copy()
    
    print(f"Initial state:")
    print(f"  a_tot: {a_tot_before}")
    print(f"  W[:3]: {W_before[:3]}")
    print(f"  V[:3]: {V_before[:3]}")
    print(f"  Break sampler total: {solver._break_sampler.total():.6e}")
    
    # Monkey-patch _do_one_break to log details
    original_do_one_break = solver._do_one_break
    
    event_count = [0]
    first_event_info = {}
    
    def logged_do_one_break():
        if event_count[0] == 0:
            # Before event
            k_sampled = None
            
            # Get sampled particle index (before calling original)
            if hasattr(solver, '_break_sampler') and solver._break_sampler:
                # Sample without consuming
                rng_state_before = solver._rng.__getstate__()
                k_sampled = solver._break_sampler.sample(solver._rng)
                solver._rng.__setstate__(rng_state_before)  # Restore state
            
            # Call original
            result = original_do_one_break()
            
            # After event
            event_count[0] += 1
            first_event_info['k'] = k_sampled
            first_event_info['a_tot_after'] = solver.a_tot
            first_event_info['W_change'] = solver.W[k_sampled] - W_before[k_sampled] if k_sampled is not None else None
            first_event_info['real_break_events'] = solver.real_break_events
            
            print(f"\nFirst Breakage Event:")
            print(f"  Selected particle k={k_sampled}")
            print(f"  W[{k_sampled}] before: {W_before[k_sampled] if k_sampled is not None else 'N/A'}")
            print(f"  W[{k_sampled}] after: {solver.W[k_sampled] if k_sampled is not None else 'N/A'}")
            print(f"  W change: {first_event_info['W_change']}")
            print(f"  a_tot before: {a_tot_before}")
            print(f"  a_tot after: {solver.a_tot}")
            print(f"  New particles created: {solver.a_tot - a_tot_before}")
            print(f"  real_break_events: {solver.real_break_events}")
            
            return result
        else:
            return original_do_one_break()
    
    solver._do_one_break = logged_do_one_break
    
    # Run solve (will stop after first event due to our logging)
    try:
        solver.solve(maxiter=10)  # Limit iterations
    except Exception as e:
        print(f"Error during solve: {e}")
    
    print(f"\nFinal state:")
    print(f"  a_tot: {solver.a_tot}")
    print(f"  W[:5]: {solver.W[:min(5, solver.a_tot)]}")
    print(f"  V[:5]: {solver.V_flat[-1, :min(5, solver.a_tot)]}")
    
    return solver, first_event_info


# Run both cases
legacy_solver, legacy_info = run_and_log_first_event(seed=42, label="LEGACY")

# For new case, use kernel framework
from wmcpbe.mcpbe import MCPBESolver

print(f"\n{'='*70}")
print(f"NEW CASE")
print(f"{'='*70}")

new = MCPBESolver(dim=1, t_total=0.05, t_write=10, init=False, load_attr=False, seed=42)
new.break_kernel_name = 'power_law'
new.break_kernel_params = {'p1': 3e-4, 'p2': 1.0, 'g': 1000, 'breakrval': 1}
new.porosity_growth_kernel_name = 'volume_mixing'
new.compression_kernel_name = 'exponential_decay'
new.compression_kernel_params = {'rate': 0.02, 'min_porosity': 0.3}
new.liquid_dist_kernel_name = 'uniform_weighted'
new.alpha_prim = 1.0
new.process_type = 'breakage'

new._initialize_kernels()
new._initialize_particles()
new._initialize_samplers()

print(f"Initial state:")
print(f"  a_tot: {new.a_tot}")
print(f"  W[:3]: {new.W[:3]}")
print(f"  V[:3]: {new.V_flat[-1, :3]}")
print(f"  Break sampler total: {new._break_sampler.total():.6e}")

# Monkey-patch for logging
original_do_one_break_new = new._do_one_break
event_count_new = [0]

def logged_do_one_break_new():
    if event_count_new[0] == 0:
        k_sampled = None
        if hasattr(new, '_break_sampler') and new._break_sampler:
            rng_state_before = new._rng.__getstate__()
            k_sampled = new._break_sampler.sample(new._rng)
            new._rng.__setstate__(rng_state_before)
        
        result = original_do_one_break_new()
        
        event_count_new[0] += 1
        print(f"\nFirst Breakage Event:")
        print(f"  Selected particle k={k_sampled}")
        print(f"  a_tot before: {new.a_tot}")
        print(f"  a_tot after: {new.a_tot}")
        print(f"  New particles created: {new.a_tot - new.a_tot}")
        print(f"  real_break_events: {new.real_break_events}")
        
        return result
    else:
        return original_do_one_break_new()

new._do_one_break = logged_do_one_break_new

try:
    new.solve(maxiter=10)
except Exception as e:
    print(f"Error during solve: {e}")

print(f"\nFinal state:")
print(f"  a_tot: {new.a_tot}")
print(f"  W[:5]: {new.W[:min(5, new.a_tot)]}")
print(f"  V[:5]: {new.V_flat[-1, :min(5, new.a_tot)]}")

# Comparison
print(f"\n{'='*70}")
print(f"COMPARISON")
print(f"{'='*70}")
print(f"\nSelected particle:")
print(f"  Legacy: k={legacy_info.get('k', 'N/A')}")
print(f"  New:    k={event_count_new[0]}")  # Will be wrong, need to fix

print(f"\na_tot after first event:")
print(f"  Legacy: {legacy_solver.a_tot}")
print(f"  New:    {new.a_tot}")
