"""Debug test to find where V_solid is lost."""

import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver
import numpy as np

print("=" * 80)
print("DEBUG: WHERE IS V_SOLID LOST?")
print("=" * 80)

# Setup solver with SAME parameters as main test
solver = MCPBESolver(
    dim=1, t_total=2.0, t_write=10, verbose=True,
    load_attr=False, init=False, seed=42,
)

solver.c = np.array([1e-3])
solver.x = np.array([5e-6])
solver.PGV = np.array(['mono'])
solver.SIG = np.array([0.0])
solver.a0 = 100
solver.process_type = 'agglomeration'
solver.COLEVAL = 3
solver.CORR_BETA = 1e-15

# Initialize particles and samplers
solver._initialize_particles()
solver._initialize_samplers()

# Create nucleation handler
solver.create_nucleation_handler(
    enabled=True, volumenstrom=1e-15, tropfen_durchmesser=1e-6,
    wasserzugabe_start=0.0, wasserzugabe_dauer=3.0,
)
solver.nucleation.configure_time_step(0.1)

print(f"\n=== Initial State ===")
n_initial = solver.a_tot
V_solid_init = np.sum(solver.get_V_solid() * solver.W[:n_initial])
print(f"  Particles: {n_initial}")
print(f"  V_solid (weighted): {V_solid_init:.6e} m³")

# Manually check a few particles
print(f"\n  Sample particles (first 5):")
for i in range(min(5, n_initial)):
    v_solid_i = np.sum(solver.V_flat[:solver.dim, i])
    v_dry_i = solver.V_flat[-1, i]
    w_i = solver.W[i]
    print(f"    [{i}] V_solid={v_solid_i:.6e}, V_dry={v_dry_i:.6e}, W={w_i:.2f}")

# Run simulation with tracing
print(f"\n=== Running solve (tracing first 5 events) ===")

# Monkey-patch the nucleation method to trace
original_distribute = solver.nucleation._distribute_one_droplet_with_dW

def traced_distribute(v_droplet):
    # Log state before
    n_before = solver.a_tot
    V_solid_before = np.sum(solver.get_V_solid() * solver.W[:n_before])
    
    result = original_distribute(v_droplet)
    
    # Log state after
    n_after = solver.a_tot
    V_solid_after = np.sum(solver.get_V_solid() * solver.W[:n_after])
    
    if hasattr(solver, '_trace_count'):
        solver._trace_count += 1
    else:
        solver._trace_count = 1
    
    if solver._trace_count <= 5:
        print(f"\n  [Event {solver._trace_count}] dW={result:.3f}")
        print(f"    Before: n={n_before}, V_solid={V_solid_before:.6e}")
        print(f"    After:  n={n_after}, V_solid={V_solid_after:.6e}")
        print(f"    ΔV_solid={V_solid_after - V_solid_before:.6e}")
        
        # Show new/modified particles
        if n_after > n_before:
            print(f"    New particles created: {n_after - n_before}")
            for i in range(n_before, n_after):
                v_solid_i = np.sum(solver.V_flat[:solver.dim, i])
                w_i = solver.W[i]
                print(f"      [{i}] V_solid={v_solid_i:.6e}, W={w_i:.3f}")
    
    return result

solver.nucleation._distribute_one_droplet_with_dW = traced_distribute

solver.solve(maxiter=5000)

print(f"\n=== Final State ===")
n_final = solver.a_tot
V_solid_final = np.sum(solver.get_V_solid() * solver.W[:n_final])
print(f"  Particles: {n_final}")
print(f"  V_solid (weighted): {V_solid_final:.6e} m³")
print(f"  V_solid error: {abs(V_solid_final - V_solid_init) / V_solid_init * 100:.2f}%")

# Check all particles for consistency
print(f"\n=== Particle Consistency Check ===")
total_weighted_solid = 0.0
for i in range(n_final):
    v_solid_i = np.sum(solver.V_flat[:solver.dim, i])
    w_i = solver.W[i]
    v_dry_i = solver.V_flat[-1, i]
    poro_i = solver.porosity[i]
    
    weighted_solid = v_solid_i * w_i
    total_weighted_solid += weighted_solid
    
    # Check V_dry vs V_solid relationship
    if not np.isnan(poro_i):
        expected_v_dry = v_solid_i / (1.0 - poro_i)
        v_dry_error = abs(v_dry_i - expected_v_dry) / expected_v_dry
        if v_dry_error > 1e-6:
            print(f"  [WARN] Particle {i}: V_dry inconsistent!")
            print(f"         V_solid={v_solid_i:.6e}, poro={poro_i:.3f}")
            print(f"         V_dry actual={v_dry_i:.6e}, expected={expected_v_dry:.6e}")

print(f"\n  Total weighted V_solid (manual sum): {total_weighted_solid:.6e} m³")
print(f"  Via get_V_solid():                   {V_solid_final:.6e} m³")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
