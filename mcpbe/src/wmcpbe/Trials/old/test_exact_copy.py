"""EXACT copy of debug_simple_rng.py to reproduce bug"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class SimpleRNGCounter:
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.count = 0
    def random(self, size=None):
        self.count += 1
        return self._rng.random(size)
    def choice(self, a, size=None, replace=True, p=None):
        self.count += 1
        return self._rng.choice(a, size, replace, p)
    @property
    def rng(self): return self._rng

from wmcpbe.mcpbe import MCPBESolver

seed = 42
pl_p1 = 3e-4
n_particles = 100

print("Creating LEGACY...")
legacy_solver = MCPBESolver(dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None)
legacy_solver.a0 = n_particles
legacy_solver.COLEVAL = 1
legacy_solver.BREAKRVAL = 1
legacy_solver.CORR_BETA = 1e-2
legacy_solver.G = 1000
legacy_solver.pl_P1 = pl_p1
legacy_solver.pl_P2 = 1.0
legacy_solver.pl_v = 2.0
legacy_solver.pl_q = 0.5
legacy_solver.alpha_prim = 1.0
legacy_solver.process_type = 'breakage'
legacy_solver._initialize_particles()
legacy_solver._init_lmc()
legacy_solver._initialize_kernels()
legacy_solver._initialize_samplers()

legacy_counter = SimpleRNGCounter(seed)
legacy_solver._rng = legacy_counter

print("Creating NEW...")
new_solver = MCPBESolver(
    dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    break_kernel_name='power_law',
    break_kernel_params={'p1': pl_p1, 'p2': 1.0, 'g': 1000, 'breakrval': 1, 'pl_v': 2.0, 'pl_q': 0.5},
    porosity_growth_kernel_name='volume_mixing',
    compression_kernel_name='exponential_decay',
    compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
    liquid_dist_kernel_name='uniform_weighted',
)
new_solver.a0 = n_particles
new_solver.alpha_prim = 1.0
new_solver.process_type = 'breakage'
new_solver._initialize_particles()
new_solver._init_lmc()
new_solver._initialize_samplers()

new_counter = SimpleRNGCounter(seed)
new_solver._rng = new_counter

print(f"\nLEGACY before: {legacy_counter.count} calls, {legacy_solver.a_tot} particles")
print(f"NEW before:    {new_counter.count} calls, {new_solver.a_tot} particles")

print("\nRunning LEGACY _do_one_break()...")
legacy_solver._do_one_break()
print(f"LEGACY after:  {legacy_counter.count} calls, {legacy_solver.a_tot} particles")

print("\nRunning NEW _do_one_break()...")
new_solver._do_one_break()
print(f"NEW after:     {new_counter.count} calls, {new_solver.a_tot} particles")

print(f"\n{'='*60}")
print(f"LEGACY total: {legacy_counter.count} calls")
print(f"NEW total:    {new_counter.count} calls")
print(f"Difference:   {legacy_counter.count - new_counter.count} calls")
