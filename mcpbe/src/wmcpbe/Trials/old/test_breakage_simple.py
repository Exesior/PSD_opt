"""
Simple Breakage Rate Test.

Directly compare the kernel output vs JIT function.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wmcpbe.kernels.breakage.power_law import PowerLawBreakageKernel
from pbe_core.func.jit_kernel_break import calc_break_rate_1d as jit_break_rate

import numpy as np

print("=" * 70)
print(" Direct Kernel vs JIT Comparison")
print("=" * 70)

# Parameters matching validate_old_vs_new.py
pl_P1 = 3e-4
pl_P2 = 1.0
G = 1000
BREAKRVAL = 1

print(f"\nParameters:")
print(f"  pl_P1 = {pl_P1}")
print(f"  pl_P2 = {pl_P2}")
print(f"  G = {G}")
print(f"  BREAKRVAL = {BREAKRVAL}")

# Create kernel
kernel = PowerLawBreakageKernel(
    p1=pl_P1,
    p2=pl_P2,
    g=G,
    breakrval=BREAKRVAL,
    pl_v=2.0,
    pl_q=0.5
)

print(f"\nKernel initialized:")
print(f"  p1 = {kernel.p1}")
print(f"  p2 = {kernel.p2}")
print(f"  g = {kernel.g}")
print(f"  breakrval = {kernel.breakrval}")
print(f"  pl_v = {kernel.pl_v}")
print(f"  pl_q = {kernel.pl_q}")

# Test volumes
volumes = [1e-18, 5e-18, 1e-17, 5e-17, 1e-16]

print(f"\n{'Volume':>12} | {'JIT Rate':>14} | {'Kernel Rate':>14} | {'Rel Error':>10}")
print(f"{'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")

V_array = np.array(volumes, dtype=np.float64)

max_error = 0.0

for i, v in enumerate(volumes):
    # JIT function expects array
    jit_rate = float(jit_break_rate(V_array, pl_P1, pl_P2, G, BREAKRVAL, i))
    
    # Kernel function
    kernel_rate = kernel.compute_rate(v, particle_idx=i)
    
    if jit_rate > 0:
        rel_error = abs(kernel_rate - jit_rate) / jit_rate
    else:
        rel_error = abs(kernel_rate - jit_rate)
    
    max_error = max(max_error, rel_error)
    
    status = "✓" if rel_error < 1e-10 else "✗"
    print(f"{v:>12.3e} | {jit_rate:>14.6e} | {kernel_rate:>14.6e} | {rel_error*100:>9.4f}% {status}")

print(f"\nMax Relative Error: {max_error*100:.10f}%")

if max_error < 1e-10:
    print("\n✓ PASS: Kernel matches JIT exactly!")
else:
    print(f"\n✗ FAIL: Error exceeds machine precision!")

# Also test what BREAKRVAL=1 should return theoretically
print(f"\nTheoretical expectation for BREAKRVAL=1:")
print(f"  Formula: S = P1 (constant)")
print(f"  Expected: {pl_P1:.6e} for all volumes")
