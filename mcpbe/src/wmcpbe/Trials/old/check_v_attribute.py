"""Check what solver.v returns."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
import numpy as np

# Create minimal solver
solver = MCPBESolver(dim=1, t_vec=[0, 1], init=True, verbose=False)

# Initialize with test particles
V_flat = np.zeros((2, 10), dtype=float)
V_flat[0, :] = 1e-12  # Volume
V_flat[1, :] = 1e-12  # V_dry (same for Vollkörper)
W_init = np.ones(10, dtype=float) * 100

solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver.porosity[:solver.a_tot] = np.nan  # Vollkörper
solver.liquid_volume[:solver.a_tot] = 0.0

print("=== Checking solver.v attribute ===")
print(f"Has .v: {hasattr(solver, 'v')}")
print(f"Type of .v: {type(getattr(solver, 'v', None))}")

if hasattr(solver, 'v'):
    print(f"\nsolver.v shape: {solver.v.shape}")
    print(f"solver.v[:3]: {solver.v[:3]}")
    print(f"solver.v is view of V_flat[-1]: {np.shares_memory(solver.v, solver.V_flat)}")

print(f"\nV_flat shape: {solver.V_flat.shape}")
print(f"V_flat[-1, :3]: {solver.V_flat[-1, :3]}")
print(f"V_flat[0, :3]: {solver.V_flat[0, :3]}")

# Check if there's a @property decorator for v
import inspect
print("\n=== Searching for 'def v' in mcpbe_base.py ===")
with open(os.path.join(os.path.dirname(__file__), '..', 'mcpbe_base.py'), 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def v(' in line.lower() or '@property' in line:
            # Print context
            start = max(0, i-1)
            end = min(len(lines), i+5)
            print(f"\nLine {start+1}-{end}:")
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"{marker} {j+1}: {lines[j]}")
