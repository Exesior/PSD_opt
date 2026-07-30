"""
Reproducibility Diagnosis Test for MCPBE Nucleation + Agglomeration.

Purpose:
    Identify sources of non-determinism in MC-PBE simulations.
    
    Two runs with identical seed=42 should produce IDENTICAL results.
    If they don't, this test will help identify where randomness diverges.

What this test tracks:
    1. RNG state before/after each MC event
    2. All calls to np.random.* (global vs seeded)
    3. Partner selection indices and times
    4. Nucleation droplet distribution events
    5. Control volume doubling events
    
Usage:
    Run twice with memory cleared between runs:
    >>> %runfile test_reproducibility_diagnosis.py --wdir
    
    Compare the output logs. Any differences indicate non-determinism source.
"""
import sys
import numpy as np
from wmcpbe import MCPBESolver


class RNGTracker:
    """
    Track all random number generator usage to identify non-determinism.
    
    This wraps an np.random.Generator and logs all random number calls.
    It provides the same interface as np.random.Generator for drop-in replacement.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.call_log = []
        self.global_rng_calls = []
        self._original_np_random = np.random.random
        self._original_np_random_normal = np.random.normal
        self._patch_global_rng()
    
    # =========================================================================
    # Drop-in replacement for np.random.Generator methods
    # =========================================================================
    
    def random(self, size=None):
        """Return random float in [0, 1)."""
        return self.rng.random(size=size)
    
    def choice(self, a, size=None, replace=True, p=None):
        """Random choice from array."""
        return self.rng.choice(a, size=size, replace=replace, p=p)
    
    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        """Return random integers."""
        return self.rng.integers(low, high=high, size=size, dtype=dtype, endpoint=endpoint)
    
    def standard_normal(self, size=None):
        """Return standard normal samples."""
        return self.rng.standard_normal(size=size)
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        """Return normal samples."""
        return self.rng.normal(loc=loc, scale=scale, size=size)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """Return uniform samples."""
        return self.rng.uniform(low=low, high=high, size=size)
    
    def shuffle(self, x, axis=0):
        """Shuffle array in place."""
        return self.rng.shuffle(x, axis=axis)
    
    def permutation(self, x, axis=None):
        """Random permutation."""
        return self.rng.permutation(x, axis=axis)
    
    @property
    def bit_generator(self):
        """Expose underlying bit generator."""
        return self.rng.bit_generator
    
    def _patch_global_rng(self):
        """Detect any usage of global np.random (should not happen!)."""
        def tracked_random(*args, **kwargs):
            import traceback
            frame = traceback.extract_stack()[-2]
            self.global_rng_calls.append({
                'location': f"{frame.filename}:{frame.lineno} in {frame.name}",
                'result': self._original_np_random(*args, **kwargs)
            })
            return self.global_rng_calls[-1]['result']
        
        np.random.random = tracked_random
    
    def log_event(self, event_type: str, details: dict):
        """Log an RNG-related event."""
        # Get current RNG state (PCG64 state has 'state' and 'inc' fields)
        try:
            state = self.rng.bit_generator.state
            if isinstance(state, dict) and 'state' in state:
                # PCG64: state is a hex string
                rng_state_str = state.get('state', 'unknown')[:16]
            else:
                rng_state_str = str(state)[:16]
        except Exception:
            rng_state_str = 'error'
        
        self.call_log.append({
            'event_type': event_type,
            'details': details,
            'rng_state': rng_state_str,
        })
    
    def sample(self):
        """Wrapper for rng.random() that logs the call."""
        value = self.rng.random()
        return value
    
    def report(self):
        """Print diagnostic report."""
        print("\n" + "=" * 80)
        print("RNG TRACKING REPORT")
        print("=" * 80)
        
        print(f"\nTotal RNG events logged: {len(self.call_log)}")
        print(f"Global np.random calls detected: {len(self.global_rng_calls)}")
        
        if self.global_rng_calls:
            print("\n⚠️ WARNING: Global np.random was used! This breaks reproducibility!")
            print("Locations:")
            for i, call in enumerate(self.global_rng_calls[:10]):
                print(f"  {i+1}. {call['location']}")
            if len(self.global_rng_calls) > 10:
                print(f"  ... and {len(self.global_rng_calls) - 10} more")
        
        # Show first few and last few events
        print("\nFirst 5 RNG events:")
        for i, event in enumerate(self.call_log[:5]):
            print(f"  {i+1}. {event['event_type']}: {event['details']}")
        
        print("\nLast 5 RNG events:")
        for i, event in enumerate(self.call_log[-5:]):
            idx = len(self.call_log) - 5 + i
            print(f"  {idx+1}. {event['event_type']}: {event['details']}")
        
        # Check for divergence patterns
        partner_selections = [e for e in self.call_log if e['event_type'] == 'partner_selection']
        nucleation_events = [e for e in self.call_log if e['event_type'] == 'nucleation_sample']
        
        print(f"\nPartner selections: {len(partner_selections)}")
        print(f"Nucleation samples: {len(nucleation_events)}")
        
        return self.call_log, self.global_rng_calls


def run_diagnostic_test(run_id: int = 1, log_file: str = None):
    """
    Run diagnostic simulation with detailed RNG tracking.
    
    Args:
        run_id: Identifier for this run (1 or 2)
        log_file: Optional file to write detailed log
    """
    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC TEST RUN #{run_id}")
    print("=" * 80)
    
    # Fixed parameters
    seed = 42
    particle_diameter = 50e-6
    droplet_diameter = 10e-6
    volumetric_flow_rate = 3.0e-12
    agg_coefficient = 1.0e-6
    t_total = 4.0
    t_write = 0.1
    
    # Derived quantities
    particle_volume = (np.pi / 6.0) * particle_diameter ** 3
    droplet_volume = (np.pi / 6.0) * droplet_diameter ** 3
    expected_liquid_volume = volumetric_flow_rate * 2.0  # 2 second window
    
    print(f"\nSeed: {seed}")
    print(f"Particle diameter: {particle_diameter*1e6:.1f} µm")
    print(f"Droplet diameter: {droplet_diameter*1e6:.1f} µm")
    print(f"Flow rate: {volumetric_flow_rate*1e12:.3f} pL/s")
    print(f"Agg coefficient: {agg_coefficient*1e6:.1f} µm³/s")
    
    # Create RNG with fixed seed
    rng = np.random.default_rng(seed)
    tracker = RNGTracker(rng)
    
    # Time vector
    t_vec = np.linspace(0.0, t_total, int(t_total / t_write) + 1)
    
    print(f"\nCreating solver with kernel configuration (seed={seed})...")
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=True,
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': agg_coefficient},
        rng=rng,
    )
    
    # Replace solver's RNG with our tracked version AFTER initialization
    # This preserves the seeded RNG used during init
    solver._rng = tracker
    solver._rng_tracker = tracker  # Store reference for debugging
    
    solver.process_type = "agglomeration"
    solver.recon_enable = False
    
    # Custom initialization
    n_particles_initial = 1000
    V_flat = np.zeros((2, n_particles_initial), dtype=float)
    V_flat[0, :] = particle_volume
    V_flat[1, :] = particle_volume
    W_init = np.full(n_particles_initial, 100.0, dtype=float)
    
    solver.Vc = 1.0
    solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
    solver._initialize_samplers()
    
    # Configure nucleation
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=volumetric_flow_rate,
        droplet_diameter=droplet_diameter,
        liquid_addition_start=0.0,
        liquid_addition_duration=2.0,
    )
    
    # Patch nucleation step to log RNG usage
    original_nuc_step = solver.nucleation.step
    original_do_agg = solver._do_one_agg
    original_pick_partner = solver._pick_partner_kernel if hasattr(solver, '_pick_partner_kernel') else None
    
    mc_event_count = [0]
    cv_doublings = []
    
    def logged_nuc_step(current_time, solver_last_dt):
        result = original_nuc_step(current_time, solver_last_dt)
        if hasattr(solver.nucleation, '_last_droplet_added') and solver.nucleation._last_droplet_added:
            tracker.log_event('nucleation_sample', {
                'time': current_time,
                'particle_idx': solver.nucleation._last_hit_particle_idx,
                'droplet_vol': solver.nucleation._last_droplet_volume,
            })
        return result
    
    def logged_do_agg():
        try:
            before_state = str(tracker.rng.bit_generator.state)[:16]
        except Exception:
            before_state = 'unknown'
        
        result = original_do_agg()
        
        try:
            after_state = str(tracker.rng.bit_generator.state)[:16]
        except Exception:
            after_state = 'unknown'
        
        if solver._last_agg_dW > 0:
            mc_event_count[0] += 1
            # Get current time from solver (try multiple attributes)
            current_time = getattr(solver, 't_current', None)
            if current_time is None:
                current_time = getattr(solver, '_current_time', None)
            if current_time is None:
                current_time = getattr(solver, '_time', 0.0)
            
            tracker.log_event('agglomeration_event', {
                'event_num': mc_event_count[0],
                'time': float(current_time) if current_time is not None else None,
                'dW': float(solver._last_agg_dW),
                'a_tot': int(solver.a_tot),
                'rng_state_before': before_state,
                'rng_state_after': after_state,
            })
        return result
    
    solver.nucleation.step = logged_nuc_step
    solver._do_one_agg = logged_do_agg
    
    # Track CV doublings
    original_double_cv = solver._maybe_double_control_volume
    def logged_double_cv(current_time, count):
        before_a_tot = solver.a_tot
        try:
            rng_state_before = str(tracker.rng.bit_generator.state)[:16]
        except Exception:
            rng_state_before = 'unknown'
        
        result = original_double_cv(current_time, count)
        
        if solver.a_tot > before_a_tot:
            try:
                rng_state_after = str(tracker.rng.bit_generator.state)[:16]
            except Exception:
                rng_state_after = 'unknown'
            
            cv_doublings.append({
                'time': float(current_time),
                'count': count,
                'a_tot_before': before_a_tot,
                'a_tot_after': int(solver.a_tot),
            })
            tracker.log_event('cv_doubling', {
                'time': float(current_time),
                'a_tot_before': before_a_tot,
                'a_tot_after': int(solver.a_tot),
                'rng_state_before': rng_state_before,
                'rng_state_after': rng_state_after,
            })
        return result
    
    solver._maybe_double_control_volume = logged_double_cv
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve()
    
    # Collect results
    nuc_stats = solver.nucleation.get_statistics()
    actual_liquid = nuc_stats['liquid_volume_added_total']
    error = (actual_liquid - expected_liquid_volume) / expected_liquid_volume
    
    # Get final time from various possible attributes
    final_time = getattr(solver, 't_current', None)
    if final_time is None:
        final_time = getattr(solver, '_elapsed', None)
    if final_time is None:
        # Fallback: use last entry in t_save or t_right
        if hasattr(solver, 't_right') and solver.t_right:
            final_time = solver.t_right[-1]
        elif hasattr(solver, 't_save') and solver.t_save:
            final_time = solver.t_save[-1]
        else:
            final_time = t_total  # Use planned total time
    
    print(f"\nResults:")
    print(f"  MC events: {mc_event_count[0]}")
    print(f"  Final time: {final_time:.6f}s")
    print(f"  Final a_tot: {solver.a_tot}")
    print(f"  Liquid added: {actual_liquid:.6e} m³")
    print(f"  Error: {error*100:+.4f}%")
    print(f"  CV doublings: {len(cv_doublings)}")
    
    # Generate RNG report
    call_log, global_calls = tracker.report()
    
    # Write detailed log if requested
    if log_file:
        with open(log_file, 'w') as f:
            f.write(f"RUN ID: {run_id}\n")
            f.write(f"SEED: {seed}\n")
            f.write(f"MC EVENTS: {mc_event_count[0]}\n")
            f.write(f"FINAL TIME: {final_time:.10f}\n")
            f.write(f"LIQUID ERROR: {error*100:+.10f}%\n")
            f.write(f"CV DOUBLINGS: {len(cv_doublings)}\n\n")
            
            f.write("DETAILED EVENT LOG:\n")
            for i, event in enumerate(call_log):
                f.write(f"{i+1}. {event['event_type']}: {event['details']}\n")
            
            if global_calls:
                f.write(f"\nGLOBAL RNG CALLS ({len(global_calls)}):\n")
                for i, call in enumerate(global_calls):
                    f.write(f"{i+1}. {call['location']}\n")
    
    return {
        'run_id': run_id,
        'mc_events': mc_event_count[0],
        'final_time': float(final_time),
        'final_a_tot': int(solver.a_tot),
        'liquid_error': error,
        'cv_doublings': len(cv_doublings),
        'rng_events': len(call_log),
        'global_rng_calls': len(global_calls),
        'call_log': call_log,
    }


def compare_runs(run1: dict, run2: dict):
    """Compare two diagnostic runs and identify differences."""
    print("\n" + "=" * 80)
    print("COMPARISON OF TWO RUNS")
    print("=" * 80)
    
    fields = ['mc_events', 'final_time', 'final_a_tot', 'liquid_error', 
              'cv_doublings', 'rng_events', 'global_rng_calls']
    
    print(f"\n{'Field':<20} {'Run 1':<20} {'Run 2':<20} {'Match':<10}")
    print("-" * 70)
    
    all_match = True
    for field in fields:
        val1 = run1[field]
        val2 = run2[field]
        
        if isinstance(val1, float):
            match = abs(val1 - val2) < 1e-10
            val1_str = f"{val1:.10f}"
            val2_str = f"{val2:.10f}"
        else:
            match = val1 == val2
            val1_str = str(val1)
            val2_str = str(val2)
        
        match_str = "✓ YES" if match else "✗ NO"
        if not match:
            all_match = False
        
        print(f"{field:<20} {val1_str:<20} {val2_str:<20} {match_str:<10}")
    
    print("\n" + "=" * 70)
    if all_match:
        print("✓ ALL FIELDS MATCH - Reproducibility confirmed!")
    else:
        print("✗ FIELDS DIVERGE - Non-determinism detected!")
        print("\nNext steps:")
        print("  1. Check 'GLOBAL RNG CALLS' in individual reports")
        print("  2. Compare detailed event logs for first divergence point")
        print("  3. Look for parallel/threading operations")
    
    # Detailed event comparison
    log1 = run1['call_log']
    log2 = run2['call_log']
    
    if len(log1) != len(log2):
        print(f"\n⚠️ Event count differs: {len(log1)} vs {len(log2)}")
        min_len = min(len(log1), len(log2))
    else:
        min_len = len(log1)
    
    first_divergence = None
    for i in range(min_len):
        e1 = log1[i]
        e2 = log2[i]
        
        if e1['event_type'] != e2['event_type']:
            first_divergence = i
            print(f"\n✗ First divergence at event {i+1}:")
            print(f"  Run 1: {e1['event_type']} - {e1['details']}")
            print(f"  Run 2: {e2['event_type']} - {e2['details']}")
            break
        
        if e1['details'] != e2['details']:
            first_divergence = i
            print(f"\n✗ First divergence at event {i+1}:")
            print(f"  Run 1: {e1['event_type']} - {e1['details']}")
            print(f"  Run 2: {e2['event_type']} - {e2['details']}")
            break
    
    if first_divergence is None and len(log1) == len(log2):
        print("\n✓ Event logs are identical!")
    
    return all_match


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MCPBE REPRODUCIBILITY DIAGNOSIS")
    print("=" * 80)
    print("""
This test will:
  1. Run two simulations with identical seed=42
  2. Track all RNG usage in both runs
  3. Compare results to identify non-determinism sources
  
Expected: Both runs produce IDENTICAL results
If not: The comparison will show where they diverge
""")
    
    # Run first simulation
    run1 = run_diagnostic_test(run_id=1, log_file='diagnosis_run1.txt')
    
    print("\n\n>>> Clearing memory and running second simulation...\n")
    
    # Run second simulation
    run2 = run_diagnostic_test(run_id=2, log_file='diagnosis_run2.txt')
    
    # Compare results
    compare_runs(run1, run2)
    
    print("\n" + "=" * 80)
    print("Diagnosis complete. Check diagnosis_run1.txt and diagnosis_run2.txt for details.")
    print("=" * 80 + "\n")
    
    sys.exit(0)
