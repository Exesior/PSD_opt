"""Test what _append_particle_column does."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
import numpy as np

print("=" * 80)
print("TEST _append_particle_column BEHAVIOR")
print("=" * 80)

# Create minimal solver
solver = MCPBESolver(
    dim=1,
    t_vec=np.linspace(0, 1, 11),
    verbose=False,
    load_attr=False,
    init=True,
)

solver.process_type = "agglomeration"
solver.recon_enable = False

# Initialize with single particle
particle_volume = 1.0e-10
V_flat = np.zeros((2, 1), dtype=float)
V_flat[0, 0] = particle_volume
V_flat[1, 0] = particle_volume  # V_dry = V_solid for Vollkörper
W_init = np.array([100.0], dtype=float)
solver.Vc = 1.0

solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver.porosity[:solver.a_tot] = np.nan
solver._initialize_samplers()

print(f"\nInitial state:")
print(f"  a_tot: {solver.a_tot}")
print(f"  V_flat shape: {solver.V_flat.shape}")
print(f"  V_flat[:, 0]: {solver.V_flat[:, 0]}")
print(f"  V_flat[-1, 0] (V_dry): {solver.V_flat[-1, 0]:.6e}")
print(f"  V_flat[0, 0] (V_solid): {solver.V_flat[0, 0]:.6e}")

# Now append a new particle with different V_solid
V_solid_new = np.array([2.0e-10])
print(f"\nCalling _append_particle_column([{V_solid_new[0]:.6e}])...")
solver._append_particle_column(V_solid_new)

print(f"\nAfter append:")
print(f"  a_tot: {solver.a_tot}")
print(f"  V_flat[:, 1]: {solver.V_flat[:, 1]}")
print(f"  V_flat[-1, 1] (V_dry): {solver.V_flat[-1, 1]:.6e}")
print(f"  V_flat[0, 1] (V_solid): {solver.V_flat[0, 1]:.6e}")

print(f"\nAnalysis:")
print(f"  V_flat[0, 1] == V_solid_new[0]? {solver.V_flat[0, 1] == V_solid_new[0]}")
print(f"  V_flat[-1, 1] == V_solid_new[0]? {solver.V_flat[-1, 1] == V_solid_new[0]}")
print(f"  V_flat[-1, 1] == sum(V_flat[:1, 1])? {solver.V_flat[-1, 1] == np.sum(solver.V_flat[:1, 1])}")

if solver.V_flat[-1, 1] == solver.V_flat[0, 1]:
    print("\n⚠️  WARNING: _append_particle_column sets V_flat[-1] = V_flat[0] for dim=1!")
    print("   This means we MUST overwrite V_flat[-1] manually after appending!")
else:
    print("\n✓ V_flat[-1] is set correctly (different from V_flat[0])")

print("\n" + "=" * 80)
