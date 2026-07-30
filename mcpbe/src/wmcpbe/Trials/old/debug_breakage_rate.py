"""
DEBUG BREAKAGE RATES - Why are no events happening?
"""
import numpy as np


def debug_rates(seed=42, label="", pl_p1=1e-2):
    """Debug breakage rates and sampling."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print(f"\n{'='*70}")
    print(f"{label} CASE (pl_P1={pl_p1:.2e})")
    print(f"{'='*70}")
    
    solver = MCPBESolver(dim=1, t_total=0.01, t_write=10, init=False, load_attr=False, seed=seed)
    solver.COLEVAL = 1
    solver.BREAKRVAL = 1
    solver.G = 1000
    solver.pl_P1 = pl_p1
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    a_tot = solver.a_tot
    
    print(f"Initial state:")
    print(f"  a_tot: {a_tot}")
    print(f"  V[:3]: {solver.V_flat[-1, :3]}")
    print(f"  W[:3]: {solver.W[:3]}")
    
    # Check break rates
    if hasattr(solver, '_break_rate'):
        print(f"\nBreakage rates:")
        print(f"  _break_rate[:5]: {solver._break_rate[:5]}")
        print(f"  Sum of rates: {np.sum(solver._break_rate[:a_tot]):.6e}")
        
        # Expected rate for BREAKRVAL=1: S = P1 (constant)
        expected_rate = pl_p1  # Should be constant
        print(f"  Expected rate per particle: {expected_rate:.2e}")
        print(f"  Expected total rate: {a_tot * expected_rate:.6e}")
    
    # Check sampler
    if hasattr(solver, '_break_sampler') and solver._break_sampler:
        sampler_total = solver._break_sampler.total()
        print(f"\nSampler:")
        print(f"  Total propensity: {sampler_total:.6e}")
        print(f"  Expected time to next event: {1.0/sampler_total:.6f}s")
        
        # Simulate tau-leaping
        if sampler_total > 0:
            tau = solver._rng.exponential(1.0 / sampler_total)
            print(f"  Sampled tau (time to next event): {tau:.6f}s")
            print(f"  Simulation time: 0.01s")
            print(f"  Expected events in 0.01s: ~{sampler_total * 0.01:.2f}")
    else:
        print(f"\n[ERROR] No break sampler!")
    
    # Try one manual sample
    if hasattr(solver, '_break_sampler') and solver._break_sampler:
        print(f"\nManual sampling test:")
        k_sampled = solver._break_sampler.sample(solver._rng)
        print(f"  Sampled particle: k={k_sampled}")
        print(f"  Rate at k: {solver._break_rate[k_sampled]:.6e}")
    
    return solver


# Test with different rates
print("="*70)
print("BREAKAGE RATE ANALYSIS")
print("="*70)

# Original low rate
solver1 = debug_rates(seed=42, label="LOW RATE", pl_p1=3e-4)

# Increased rate
solver2 = debug_rates(seed=42, label="MEDIUM RATE", pl_p1=1e-2)

# Very high rate
solver3 = debug_rates(seed=42, label="HIGH RATE", pl_p1=1.0)

# Analysis
print(f"\n{'='*70}")
print(f"ANALYSIS")
print(f"{'='*70}")
print(f"\nWith pl_P1=3e-4:")
print(f"  Total rate: ~{1000 * 3e-4:.2e} events/s")
print(f"  In 0.01s: ~{1000 * 3e-4 * 0.01:.4f} events")
print(f"\nWith pl_P1=1e-2:")
print(f"  Total rate: ~{1000 * 1e-2:.2e} events/s")
print(f"  In 0.01s: ~{1000 * 1e-2 * 0.01:.2f} events")
print(f"\nWith pl_P1=1.0:")
print(f"  Total rate: ~{1000 * 1.0:.2e} events/s")
print(f"  In 0.01s: ~{1000 * 1.0 * 0.01:.1f} events")

print(f"\n⚠️  Even with pl_P1=1e-2, erwarten wir nur ~0.1 Events in 0.01s!")
print(f"   → Entweder t_total erhöhen ODER pl_P1 weiter erhöhen!")
