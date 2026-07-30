"""Quick import and basic functionality test."""
import sys
import numpy as np

# Test imports
try:
    from porosity_compression import PorosityCompressionKernel
    from liquid_internalization import LiquidInternalizationKernel
    print("✓ Direct imports successful")
except ImportError as e:
    print(f"✗ Direct import failed: {e}")
    sys.exit(1)

# Test Porosity Compression Kernel
print("\n--- Porosity Compression Kernel ---")
kernel_pc = PorosityCompressionKernel(rate=0.02, min_porosity=0.3)
print(f"Created: {kernel_pc.name}")
print(f"Parameters: {kernel_pc.params}")

# Test compute method
poro_initial = 0.5
dt = 1.0
poro_new = kernel_pc.compute(poro_initial, dt)
expected = 0.3 + (0.5 - 0.3) * np.exp(-0.02 * 1.0)
error = abs(poro_new - expected)
print(f"compute({poro_initial}, {dt}) = {poro_new:.6f}")
print(f"Expected (analytical): {expected:.6f}")
print(f"Error: {error:.2e}")
if error < 1e-10:
    print("✓ Porosity compression test PASSED")
else:
    print("✗ Porosity compression test FAILED")

# Test Liquid Internalization Kernel
print("\n--- Liquid Internalization Kernel ---")
kernel_li = LiquidInternalizationKernel(k_int=1e12)
print(f"Created: {kernel_li.name}")
print(f"Parameters: {kernel_li.params}")

# Test compute method
S_0 = 0.0
v_pore = 1e-18
l_total = 0.8e-18
dt = 0.01
S_new = kernel_li.compute(S_0, v_pore, l_total, dt)
print(f"compute(S={S_0}, v_pore={v_pore:.2e}, l_total={l_total:.2e}, dt={dt}) = {S_new:.6f}")
print(f"Saturation increased from {S_0} to {S_new:.6f}")
if S_new > S_0 and S_new <= 1.0:
    print("✓ Liquid internalization test PASSED")
else:
    print("✗ Liquid internalization test FAILED")

# Test factory function
print("\n--- Factory Function Test ---")
try:
    from __init__ import get_continuous_kernel, list_continuous_kernels
    available = list_continuous_kernels()
    print(f"Available kernels: {available}")
    
    k1 = get_continuous_kernel('porosity_compression', rate=0.01)
    print(f"✓ Created via factory: {k1.name}")
    
    k2 = get_continuous_kernel('liquid_internalization', k_int=1e10)
    print(f"✓ Created via factory: {k2.name}")
except Exception as e:
    print(f"✗ Factory test failed: {e}")

print("\n" + "=" * 60)
print("QUICK TEST COMPLETE")
print("=" * 60)
