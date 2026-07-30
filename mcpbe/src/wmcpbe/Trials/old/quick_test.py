"""Quick test for nucleation physics."""
import numpy as np
import sys
sys.path.insert(0, '../')

from wmcpbe.mcpbe import MCPBESolver

solver = MCPBESolver(
    dim=1, t_total=1.0, t_write=10, verbose=False,
    load_attr=False, init=False, seed=42,
)

solver.c = np.array([1e-3])
solver.x = np.array([5e-6])
solver.PGV = np.array(["mono"])
solver.SIG = np.array([0.0])
solver.a0 = 1000
solver.process_type = "agglomeration"
solver.COLEVAL = 3
solver.CORR_BETA = 1e-15
solver.G = 1.0
solver.alpha_prim = 1.0

solver._initialize_particles()
solver._initialize_samplers()
solver.liquid_volume[:solver.a_tot] = 0.0

print("Initial:")
n_comp = solver.a_tot
sum_W = np.sum(solver.W[:n_comp])
n_phys = sum_W / solver.Vc
print(f"  n_comp={n_comp}, n_phys={n_phys:.2f}, Sum(W)={sum_W:.2f}, Vc={solver.Vc}")

volumenstrom = 1e-12
duration = 1.0
v_droplet = (4.0/3.0) * np.pi * (0.5e-6)**3
expected_liquid = volumenstrom * duration
expected_droplets = expected_liquid / v_droplet

print(f"\nExpected:")
print(f"  Liquid: {expected_liquid:.6e} m³")
print(f"  Droplets: {expected_droplets:.1f}")

solver.create_nucleation_handler(
    enabled=True, volumenstrom=volumenstrom,
    tropfen_durchmesser=1e-6, wasserzugabe_start=0.0,
    wasserzugabe_dauer=duration,
)
solver.nucleation.configure_time_step(0.1)

solver.solve(maxiter=10000)

n_final = solver.a_tot
liquid_final = np.sum(solver.liquid_volume[:n_final] * solver.W[:n_final]) / solver.Vc
stats = solver.nucleation.get_statistics()

print(f"\nFinal:")
print(f"  n_comp={n_final} (Δ={n_final-solver.a0:+d})")
sum_W_final = np.sum(solver.W[:n_final])
n_phys_final = sum_W_final / solver.Vc
print(f"  n_phys={n_phys_final:.2f} (Δ={n_phys_final-n_phys:+.2f})")
print(f"  Liquid in system: {liquid_final:.6e} m³")
print(f"  Liquid in stats:  {stats['liquid_volume_added_total']:.6e} m³")
print(f"  Error: {abs(liquid_final - stats['liquid_volume_added_total']) / expected_liquid * 100:.2f}%")
print(f"  Droplets stat: {stats['droplets_added_total']:.2f}")
