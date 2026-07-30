"""
BREAKAGE PARAMETER DEBUG SCRIPT
Fast verification (<30s) that Legacy and New use IDENTICAL parameters and RNG.

This helps diagnose why Breakage results differ (1000 vs 1001 particles).
"""
import numpy as np
import time

print_header = lambda title: print("\n" + "=" * 70 + f"\n {title}\n" + "=" * 70)


def compute_moments(X, W):
    """Compute Q0, Q3, d_mean from diameter/weight arrays."""
    Q0 = np.sum(W)
    V_tot = np.sum(W * (np.pi / 6.0) * X**3)
    Q3 = V_tot
    d_mean = (6.0 * Q3 / (np.pi * Q0))**(1.0 / 3.0) if Q0 > 0 else 0.0
    return {'Q0': Q0, 'Q3': Q3, 'd_mean': d_mean}


def run_legacy_case(
    t_total=0.1,
    seed=42,
    breakrval=1,
    pl_p1=3e-4,
    pl_p2=1.0,
    g=1000,
    verbose=False,
) -> dict:
    """Run simulation with LEGACY COLEVAL/BREAKRVAL approach."""
    from wmcpbe.mcpbe import MCPBESolver
    
    start_time = time.time()
    
    # Create solver WITHOUT auto-init
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=10,
        init=False,
        load_attr=False,
        seed=seed,
    )
    
    # Set ALL parameters BEFORE initializing
    solver.COLEVAL = 1
    solver.BREAKRVAL = breakrval
    solver.G = g
    solver.pl_P1 = pl_p1
    solver.pl_P2 = pl_p2
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    # Controlled initialization order
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    # Debug: Show kernel params
    if verbose:
        print("\n  Legacy Kernel Configuration:")
        km = solver.kernel_manager
        if km and km.break_kernel:
            print(f"    Break Kernel: {type(km.break_kernel).__name__}")
            print(f"    p1: {getattr(km.break_kernel, 'p1', 'N/A')}")
            print(f"    p2: {getattr(km.break_kernel, 'p2', 'N/A')}")
            print(f"    g: {getattr(km.break_kernel, 'g', 'N/A')}")
            print(f"    breakrval: {getattr(km.break_kernel, 'breakrval', 'N/A')}")
            print(f"    pl_v: {getattr(km.break_kernel, 'pl_v', 'N/A')}")
            print(f"    pl_q: {getattr(km.break_kernel, 'pl_q', 'N/A')}")
    
    # Debug: Show initial sampler state
    if verbose:
        print(f"\n  Legacy Initial State:")
        print(f"    a_tot: {solver.a_tot}")
        if hasattr(solver, '_break_sampler') and solver._break_sampler:
            print(f"    Break sampler total: {solver._break_sampler.total()}")
        print(f"    pl_P1 (solver attr): {getattr(solver, 'pl_P1', 'N/A')}")
        print(f"    pl_v (solver attr): {getattr(solver, 'pl_v', 'N/A')}")
        print(f"    pl_q (solver attr): {getattr(solver, 'pl_q', 'N/A')}")
    
    try:
        solver.solve(max_particles=50000)
    except Exception as e:
        print(f"\n  [ERROR] Legacy solve failed: {e}")
        return {'X': np.array([]), 'W': np.array([]), 'moments': {'Q0': 0, 'Q3': 0, 'd_mean': 0}, 'time': 0}
    
    elapsed = time.time() - start_time
    
    a_tot = solver.a_tot
    if a_tot < 1:
        return {'X': np.array([]), 'W': np.array([]), 'moments': {'Q0': 0, 'Q3': 0, 'd_mean': 0}, 'time': elapsed}
    
    V_ges = solver.V_flat[-1, :a_tot].copy()
    X_final = (6.0 * V_ges / np.pi) ** (1.0/3.0)
    W_final = solver.W[:a_tot].copy()
    
    moments = compute_moments(X_final, W_final)
    moments['time'] = elapsed
    moments['a_tot'] = a_tot
    moments['events'] = getattr(solver, 'real_break_events', 0)
    
    return {
        'X': X_final,
        'W': W_final,
        'moments': moments,
        'time': elapsed,
        'a_tot': a_tot,
        'events': moments['events'],
    }


def run_new_case(
    t_total=0.1,
    seed=42,
    break_kernel_name='power_law',
    breakrval=1,
    pl_p1=3e-4,
    pl_p2=1.0,
    g=1000,
    verbose=False,
) -> dict:
    """Run simulation with NEW kernel framework."""
    from wmcpbe.mcpbe import MCPBESolver
    
    start_time = time.time()
    
    # Create solver WITHOUT auto-init
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=10,
        init=False,
        load_attr=False,
        seed=seed,
    )
    
    # Set kernel parameters BEFORE initialization
    if break_kernel_name:
        solver.break_kernel_name = break_kernel_name
        solver.break_kernel_params = {
            'p1': pl_p1,
            'p2': pl_p2,
            'g': g,
            'breakrval': breakrval,
        }
    
    # Always set these (Legacy always has them)
    solver.porosity_growth_kernel_name = 'volume_mixing'
    solver.compression_kernel_name = 'exponential_decay'
    solver.compression_kernel_params = {'rate': 0.02, 'min_porosity': 0.3}
    solver.liquid_dist_kernel_name = 'uniform_weighted'
    
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    # Controlled initialization order
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    # Debug: Show kernel params
    if verbose:
        print("\n  New Kernel Configuration:")
        km = solver.kernel_manager
        if km and km.break_kernel:
            print(f"    Break Kernel: {type(km.break_kernel).__name__}")
            print(f"    p1: {getattr(km.break_kernel, 'p1', 'N/A')}")
            print(f"    p2: {getattr(km.break_kernel, 'p2', 'N/A')}")
            print(f"    g: {getattr(km.break_kernel, 'g', 'N/A')}")
            print(f"    breakrval: {getattr(km.break_kernel, 'breakrval', 'N/A')}")
            print(f"    pl_v: {getattr(km.break_kernel, 'pl_v', 'N/A')}")
            print(f"    pl_q: {getattr(km.break_kernel, 'pl_q', 'N/A')}")
    
    # Debug: Show initial sampler state
    if verbose:
        print(f"\n  New Initial State:")
        print(f"    a_tot: {solver.a_tot}")
        if hasattr(solver, '_break_sampler') and solver._break_sampler:
            print(f"    Break sampler total: {solver._break_sampler.total()}")
        print(f"    pl_P1 (solver attr): {getattr(solver, 'pl_P1', 'N/A')}")
        print(f"    pl_v (solver attr): {getattr(solver, 'pl_v', 'N/A')}")
        print(f"    pl_q (solver attr): {getattr(solver, 'pl_q', 'N/A')}")
    
    try:
        solver.solve(max_particles=50000)
    except Exception as e:
        print(f"\n  [ERROR] New solve failed: {e}")
        return {'X': np.array([]), 'W': np.array([]), 'moments': {'Q0': 0, 'Q3': 0, 'd_mean': 0}, 'time': 0}
    
    elapsed = time.time() - start_time
    
    a_tot = solver.a_tot
    if a_tot < 1:
        return {'X': np.array([]), 'W': np.array([]), 'moments': {'Q0': 0, 'Q3': 0, 'd_mean': 0}, 'time': elapsed}
    
    V_ges = solver.V_flat[-1, :a_tot].copy()
    X_final = (6.0 * V_ges / np.pi) ** (1.0/3.0)
    W_final = solver.W[:a_tot].copy()
    
    moments = compute_moments(X_final, W_final)
    moments['time'] = elapsed
    moments['a_tot'] = a_tot
    moments['events'] = getattr(solver, 'real_break_events', 0)
    
    return {
        'X': X_final,
        'W': W_final,
        'moments': moments,
        'time': elapsed,
        'a_tot': a_tot,
        'events': moments['events'],
    }


# ======================================================================
# MAIN DEBUG RUN
# ======================================================================

print_header("BREAKAGE PARAMETER DEBUG")
print("Comparing Legacy vs New Breakage Implementation\n")

# Test parameters
t_total = 0.1
seed = 42
breakrval = 1
pl_p1 = 3e-4
pl_p2 = 1.0
g = 1000

print(f"Parameters:")
print(f"  t_total={t_total}, seed={seed}, BREAKRVAL={breakrval}")
print(f"  pl_P1={pl_p1}, pl_P2={pl_p2}, G={g}")
print(f"  pl_v=2.0, pl_q=0.5 (legacy defaults)\n")

# Run both cases with verbose output
print("=" * 70)
print("LEGACY CASE")
print("=" * 70)
legacy = run_legacy_case(
    t_total=t_total, seed=seed, breakrval=breakrval,
    pl_p1=pl_p1, pl_p2=pl_p2, g=g, verbose=True  # Set to True for detailed debug
)

print("\n" + "=" * 70)
print("NEW CASE")
print("=" * 70)
new = run_new_case(
    t_total=t_total, seed=seed, break_kernel_name='power_law', breakrval=breakrval,
    pl_p1=pl_p1, pl_p2=pl_p2, g=g, verbose=True  # Set to True for detailed debug
)

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(f"\n  Particle Count:")
print(f"    Legacy: {legacy['a_tot']}")
print(f"    New:    {new['a_tot']}")
if legacy['a_tot'] != new['a_tot']:
    print(f"    ⚠️  DIFFERENT! Difference: {abs(new['a_tot'] - legacy['a_tot'])}")
else:
    print(f"    ✓  SAME!")

print(f"\n  Events:")
print(f"    Legacy: {legacy['events']:.0f}")
print(f"    New:    {new['events']:.0f}")
if abs(new['events'] - legacy['events']) > 1:
    print(f"    ⚠️  DIFFERENT! Difference: {abs(new['events'] - legacy['events'])}")
else:
    print(f"    ✓  SAME (or within 1 event)")

print(f"\n  Moments:")
for key in ['Q0', 'Q3', 'd_mean']:
    val_legacy = legacy['moments'][key]
    val_new = new['moments'][key]
    
    if val_legacy != 0:
        rel_error = abs(val_new - val_legacy) / abs(val_legacy) * 100
    else:
        rel_error = abs(val_new - val_legacy) * 100
    
    status = "✓" if rel_error < 0.1 else "⚠️"
    print(f"    {status} {key}: Legacy={val_legacy:.6e}, New={val_new:.6e}, Error={rel_error:.4f}%")

print(f"\n  Performance:")
print(f"    Legacy: {legacy['time']:.2f}s")
print(f"    New:    {new['time']:.2f}s")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if legacy['a_tot'] == new['a_tot'] and abs(new['events'] - legacy['events']) <= 1:
    print("  ✓ Particles and events match!")
    print("  If results still differ, check algorithm implementation.")
else:
    print("  ⚠️  MISMATCH detected! Possible causes:")
    print("     1. Different propensity calculations → Check kernel params (pl_v, pl_q)")
    print("     2. Different RNG consumption → Sampler initialization differs")
    print("     3. Different fragment distribution → _do_one_break() differs")
