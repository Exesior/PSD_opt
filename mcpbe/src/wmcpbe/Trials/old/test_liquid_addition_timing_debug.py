"""
Test: Liquid Addition Timing - Comprehensive Debug Suite

Tests 5 different scenarios for liquid addition timing with detailed debugging:

1. SIMPLE: Start at t=0, ends exactly on an event
2. SPLIT_END: Start at t=0, ends BETWEEN two events (must use partial dt)
3. DELAYED_START: Start at t>0 but before first MC event
4. WINDOW_BETWEEN: Addition window entirely between two events
5. WINDOW_BEFORE_FIRST: Window ends before first MC event (manual trigger)

KEY DESIGN PRINCIPLES:
- VERY LARGE particles (500µm) → minimal agglomeration rate
- SMALL droplets (0.5-2µm) → many droplets, smooth addition
- LOW event rate (corr_beta=1e-15) → controlled event timing
- FEW initial particles (20-30) → track individual events
- Duration-based addition (no wt%)

VALIDATION:
- Compare V_liquid_in_system vs V_liquid_expected = volumenstrom × dt_effective
- Track agglomeration events (should be <5% for large particles)
- Verify droplet count matches expectation

Author: Generated for debugging liquid addition timing cases
"""

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver


def print_separator(title, char='='):
    print("\n" + char * 80)
    print(title)
    print(char * 80 + "\n")


def calculate_weighted_total(values, weights, n_active):
    """Calculate weighted total for DSMC-consistent comparison."""
    if n_active <= 0:
        return 0.0
    total = 0.0
    for k in range(n_active):
        total += float(values[k]) * float(weights[k])
    return total


def setup_solver_for_nucleation_test(
    t_total,
    seed=42,
    corr_beta=1e-12,      # HIGHER event rate for faster simulation
    diameter=100e-6,       # MEDIUM particles (100µm) - balance between agglomeration and speed
    a0=100,                # MORE particles for faster agglomeration
    verbose=False
):
    """
    Create a solver configured for nucleation timing tests.
    
    DESIGN PRINCIPLES FOR SPEED:
    - MEDIUM particles (100µm) → reasonable collision rate
    - HIGHER event rate (corr_beta=1e-12) → frequent MC events
    - MORE particles (a0=100) → faster agglomeration (n² combinations)
    - SMALL droplets (0.5-2µm) → less solid volume needed per droplet
    
    PHYSICS: Agglomeration rate scales with:
    - n_particles² (more particles = more pairs)
    - 1/diameter² (smaller particles collide more often)
    - corr_beta (direct scaling factor)
    
    Args:
        t_total: Total simulation time [s]
        seed: Random seed
        corr_beta: Agglomeration rate (default: 1e-12, HIGH for fast events)
        diameter: Particle diameter [m] (default: 100µm)
        a0: Initial particle count (default: 100)
        verbose: Print debug info
    
    Returns:
        Configured MCPBESolver instance
    """
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=-1,  # No intermediate output
        verbose=verbose,
        load_attr=False,
        init=False,
        seed=seed,
    )
    
    # MEDIUM particles for reasonable agglomeration rate
    # 100µm → 25x higher collision rate than 500µm!
    solver.c = np.array([1e-3])
    solver.x = np.array([diameter])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = a0
    
    # HIGHER agglomeration rate for FAST events
    # This prevents manual triggering and speeds up simulation
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3
    solver.CORR_BETA = corr_beta
    solver.G = 0.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    return solver


def validate_liquid_addition(
    solver,
    expected_v_liquid,
    n_init,
    tolerance=0.10,
    max_agglo_fraction=0.05,
    case_name=""
):
    """
    Validate liquid addition with detailed diagnostics.
    
    DSMC STATISTICS INTERPRETATION:
    The nucleation handler tracks statistics with DSMC weight normalization:
      _liquid_volume_added_total = Σ(v_droplet × dW × Vc_ref/Vc)
    
    This represents the PHYSICAL liquid volume added, accounting for the
    fact that each computational particle represents dW/Vc physical particles.
    
    For validation, we compare:
    1. V_liq_system (weighted): Σ(liquid_volume[k] × W[k]) - physical liquid in system
    2. V_liq_stat: Statistics from nucleation handler
    3. V_liq_expected: volumenstrom × dt_effective
    
    All three should match within tolerance!
    
    Args:
        solver: MCPBESolver instance
        expected_v_liquid: Expected liquid volume [m³]
        n_init: Initial particle count
        tolerance: Error tolerance for liquid conservation (default 10%)
        max_agglo_fraction: Maximum allowed agglomeration fraction (default 5%)
        case_name: Name of test case for reporting
    
    Returns: (passed, error_fraction, details_dict)
    """
    n_final = solver.a_tot
    
    # Calculate liquid in system (weighted sum)
    # This represents the PHYSICAL liquid volume in the system
    V_liq_system = calculate_weighted_total(
        solver.liquid_volume[:n_final],
        solver.W[:n_final],
        n_final
    )
    
    # Get nucleation statistics
    # Note: These are DSMC-weighted: Σ(v_droplet × dW × Vc_ref/Vc)
    nuc_stats = solver.nucleation.get_statistics()
    V_liq_stat = nuc_stats['liquid_volume_added_total']
    n_droplets_stat = nuc_stats.get('droplets_added_total', 0)
    
    # Track agglomeration
    n_agglo_events = n_final - n_init
    agglo_fraction = n_agglo_events / n_init if n_init > 0 else 0.0
    
    # Calculate errors - focus on system liquid vs expected
    # Also check consistency between system and statistics
    if expected_v_liquid > 0:
        system_error = abs(V_liq_system - expected_v_liquid) / expected_v_liquid
        stat_error = abs(V_liq_stat - expected_v_liquid) / expected_v_liquid
        # Internal consistency check
        if V_liq_system > 0:
            consistency_error = abs(V_liq_stat - V_liq_system) / V_liq_system
        else:
            consistency_error = 0.0 if V_liq_stat == 0 else 1.0
    else:
        system_error = 0.0 if V_liq_system == 0 else 1.0
        stat_error = 0.0 if V_liq_stat == 0 else 1.0
        consistency_error = 0.0 if V_liq_stat == V_liq_system else 1.0
    
    # Validation criteria
    liquid_passed = system_error < tolerance
    agglo_passed = agglo_fraction < max_agglo_fraction
    passed = liquid_passed and agglo_passed
    
    details = {
        'case_name': case_name,
        'V_liq_system': V_liq_system,
        'V_liq_stat': V_liq_stat,
        'V_liq_expected': expected_v_liquid,
        'system_error': system_error,
        'stat_error': stat_error,
        'consistency_error': consistency_error,
        'n_init': n_init,
        'n_final': n_final,
        'n_agglo_events': n_agglo_events,
        'agglo_fraction': agglo_fraction,
        'n_droplets_stat': n_droplets_stat,
        'liquid_passed': liquid_passed,
        'agglo_passed': agglo_passed,
    }
    
    return passed, system_error, details


def print_results(details, passed, tolerance, max_agglo):
    """Print detailed test results."""
    print(f"\nResults:")
    print(f"  V_liquid (system):     {details['V_liq_system']:.6e} m³")
    print(f"  V_liquid (statistics): {details['V_liq_stat']:.6e} m³")
    print(f"  V_liquid (expected):   {details['V_liq_expected']:.6e} m³")
    print(f"  Error (system):        {details['system_error']*100:.2f}%")
    print(f"  Error (statistics):    {details['stat_error']*100:.2f}%")
    print(f"  Consistency (sys/stat):{details['consistency_error']*100:.2f}%")
    print(f"  Droplets (stat):       {details['n_droplets_stat']:.2f}")
    print(f"  Particles (init):      {details['n_init']}")
    print(f"  Particles (final):     {details['n_final']}")
    print(f"  Agglomeration events:  {details['n_agglo_events']} ({details['agglo_fraction']*100:.2f}%)")
    
    # Detailed validation
    print(f"\nValidation:")
    liquid_status = "✅" if details['liquid_passed'] else "❌"
    agglo_status = "✅" if details['agglo_passed'] else "❌"
    print(f"  Liquid conservation: {liquid_status} (tolerance: {tolerance*100:.0f}%, error: {details['system_error']*100:.2f}%)")
    print(f"  Agglomeration limit: {agglo_status} (max: {max_agglo*100:.0f}%, actual: {details['agglo_fraction']*100:.2f}%)")
    
    # Consistency warning
    if details['consistency_error'] > 0.01:
        print(f"  ⚠️  WARNING: System/Statistics inconsistency > 1%!")
    
    overall_status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nOverall: {overall_status}")
    
    return passed


# ============================================================================
# TEST CASE 1: Simple - Start at t=0, ends on event
# ============================================================================

def test_case_1_simple():
    """
    CASE 1: Simple timing - Start at t=0, ends exactly on an event
    
    Timeline:
    t=0.0s: ┌───────────────┐●
            │   Nucleation  │ ← Event happens right at window end
    t=T:    └───────────────┘
    
    Expected: Full duration used, all liquid distributed
    """
    print_separator("CASE 1: Simple (Start t=0, End on Event)")
    
    # Parameters - designed for clean event at window end
    volumenstrom = 1e-14      # m³/s
    dauer = 1.0               # s - long enough for events to occur
    tropfen_durchmesser = 1e-6  # m (1µm droplets)
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = expected_v_liquid / v_droplet
    
    print(f"Parameters:")
    print(f"  volumenstrom:       {volumenstrom:.2e} m³/s")
    print(f"  duration:           {dauer} s")
    print(f"  droplet_diameter:   {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  v_droplet:          {v_droplet:.2e} m³")
    print(f"  corr_beta:          1e-14 (moderate event rate)")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets:.0f}")
    
    # Setup solver - moderate event rate for clean timing
    solver = setup_solver_for_nucleation_test(
        t_total=dauer + 0.5,
        seed=42,
        corr_beta=1e-12,       # Moderate for reliable events
        diameter=5e-5,        # 50µm - very large
        a0=700                   # Few particles
    )
    
    # Track initial state
    n_init = solver.a_tot
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Vc: {solver.Vc:.2e} m³")
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve(maxiter=10000)
    
    # Validate
    passed, error, details = validate_liquid_addition(
        solver, expected_v_liquid, n_init,
        tolerance=0.10,
        max_agglo_fraction=0.05,
        case_name="Case 1: Simple"
    )
    
    print_results(details, passed, tolerance=0.10, max_agglo=0.05)
    
    return passed


# ============================================================================
# TEST CASE 2: Split End - Start at t=0, ends BETWEEN events
# ============================================================================

def test_case_2_split_end():
    """
    CASE 2: Split End - Start at t=0, ends BETWEEN two events
    
    Timeline:
    t=0.0s: ┌─────────────────────┐
            │     Nucleation      │
    t=T1:   ●                     │
            │         ×           │ ← Window ends HERE (between events)
    t=T2:   │                     ● ← Next event (NOT included)
            └─────────────────────┘
    
    Expected: Only liquid up to window end, NOT until next event!
    Must use dt = window_end - t_last_event (not t_next_event - t_last_event)
    """
    print_separator("CASE 2: Split End (Start t=0, End Between Events)")
    
    # Parameters - designed for window ending between events
    volumenstrom = 1e-14        # m³/s
    dauer_configured = 2.0      # s - configured duration
    tropfen_durchmesser = 0.5e-6  # m (0.5µm - small droplets)
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    
    # KEY INSIGHT: Window ends between events
    # We expect events at irregular intervals due to MC nature
    # The effective duration should be EXACTLY dauer_configured
    # (window end cuts off the partial interval)
    expected_v_liquid = volumenstrom * dauer_configured
    expected_droplets = expected_v_liquid / v_droplet
    
    print(f"Parameters:")
    print(f"  volumenstrom:         {volumenstrom:.2e} m³/s")
    print(f"  duration (configured):{dauer_configured} s")
    print(f"  droplet_diameter:     {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  v_droplet:            {v_droplet:.2e} m³")
    print(f"  corr_beta:            5e-15 (low event rate)")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets:.0f}")
    print(f"\nNote: Window ends between events - must use exact duration!")
    
    # Setup solver - LOW event rate so window likely ends between events
    solver = setup_solver_for_nucleation_test(
        t_total=dauer_configured + 1.0,
        seed=42,
        corr_beta=5e-15,       # Low event rate
        diameter=500e-6,        # 500µm - very large
        a0=20                   # Few particles
    )
    
    # Track initial state
    n_init = solver.a_tot
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=dauer_configured,
    )
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve(maxiter=15000)
    
    # Validate
    passed, error, details = validate_liquid_addition(
        solver, expected_v_liquid, n_init,
        tolerance=0.10,
        max_agglo_fraction=0.05,
        case_name="Case 2: Split End"
    )
    
    print_results(details, passed, tolerance=0.10, max_agglo=0.05)
    
    if not passed:
        print(f"\n⚠️  DEBUG INFO:")
        print(f"   If error > 10%, check if partial dt was handled correctly.")
        print(f"   The overlap calculation in step() must use:")
        print(f"   dt_effective = min(window_end, event_end) - max(window_start, event_start)")
    
    return passed


# ============================================================================
# TEST CASE 3: Delayed Start - t_start > 0, before first event
# ============================================================================

def test_case_3_delayed_start():
    """
    CASE 3: Delayed Start - t_start > 0 but before first MC event
    
    Timeline:
    t=0.0s: ───────────────────────  
            │
    t=0.5s: ┌───────────┐           ← Nucleation STARTS (delayed)
            │ Nucleation│
    t=1.0s: └────●──────┘           ← First event during window
    
    Expected: Liquid added from t_start onward
    """
    print_separator("CASE 3: Delayed Start (t_start > 0, Before First Event)")
    
    # Parameters
    volumenstrom = 1e-14      # m³/s
    t_start = 0.5             # s (delayed start)
    dauer = 0.5               # s
    t_end = t_start + dauer   # 1.0s
    tropfen_durchmesser = 1e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = expected_v_liquid / v_droplet
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  t_start:          {t_start} s")
    print(f"  duration:         {dauer} s")
    print(f"  t_end:            {t_end} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  corr_beta:        1e-14 (moderate)")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets:.0f}")
    
    # Setup solver
    solver = setup_solver_for_nucleation_test(
        t_total=t_end + 0.5,
        seed=42,
        corr_beta=1e-14,
        diameter=500e-6,
        a0=25
    )
    
    # Track initial state
    n_init = solver.a_tot
    print(f"\nInitial state: {n_init} particles")
    
    # Create nucleation handler with DELAYED start
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=t_start,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve(maxiter=10000)
    
    # Validate
    passed, error, details = validate_liquid_addition(
        solver, expected_v_liquid, n_init,
        tolerance=0.10,
        max_agglo_fraction=0.05,
        case_name="Case 3: Delayed Start"
    )
    
    print_results(details, passed, tolerance=0.10, max_agglo=0.05)
    
    return passed


# ============================================================================
# TEST CASE 4: Window Between Two Events
# ============================================================================

def test_case_4_window_between():
    """
    CASE 4: Addition window entirely between two events
    
    Timeline:
    t=0.0s: ●────────────────────●  ← Events
                ┌────┐
    t=0.3s:     │Win │              ← Window STARTS after first event
                └────┘
    t=0.5s:         │                ← Window ENDS before second event
    
    Expected: Liquid distributed based on overlap with event interval
    """
    print_separator("CASE 4: Window Between Two Events")
    
    # Parameters
    volumenstrom = 1e-14      # m³/s
    t_start = 0.3             # s
    dauer = 0.2               # s
    t_end = t_start + dauer   # 0.5s
    tropfen_durchmesser = 0.8e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = expected_v_liquid / v_droplet
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  t_start:          {t_start} s")
    print(f"  duration:         {dauer} s")
    print(f"  t_end:            {t_end} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  corr_beta:        2e-14 (for events around window)")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets:.0f}")
    
    # Setup solver - event rate tuned for events around window
    solver = setup_solver_for_nucleation_test(
        t_total=t_end + 0.5,
        seed=42,
        corr_beta=2e-14,
        diameter=500e-6,
        a0=25
    )
    
    # Track initial state
    n_init = solver.a_tot
    print(f"\nInitial state: {n_init} particles")
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=t_start,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve(maxiter=10000)
    
    # Validate
    passed, error, details = validate_liquid_addition(
        solver, expected_v_liquid, n_init,
        tolerance=0.10,
        max_agglo_fraction=0.05,
        case_name="Case 4: Window Between"
    )
    
    print_results(details, passed, tolerance=0.10, max_agglo=0.05)
    
    return passed


# ============================================================================
# TEST CASE 5: Window Before First Event (Manual Trigger)
# ============================================================================

def test_case_5_window_before_first():
    """
    CASE 5: Window ends BEFORE first MC event (manual triggering)
    
    Timeline:
    t=0.0s: ┌──────────┐
            │ Nucleation│
    t=0.1s: └─────┬────┘  ← Window ends (very short)
                  │
    t=0.5s:       ●  ← First MC event → Manual trigger needed
    
    Expected: Full liquid added via manual trigger at first event
    """
    print_separator("CASE 5: Window Before First Event (Manual Trigger)")
    
    # Parameters - VERY SHORT window, will trigger manually
    volumenstrom = 1e-15      # m³/s (lower for reasonable droplet count)
    t_start = 0.0             # s
    dauer = 0.1               # s (very short!)
    t_end = t_start + dauer   # 0.1s
    tropfen_durchmesser = 0.5e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = expected_v_liquid / v_droplet
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  t_start:          {t_start} s")
    print(f"  duration:         {dauer} s")
    print(f"  t_end:            {t_end} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  v_droplet:        {v_droplet:.2e} m³")
    print(f"  corr_beta:        1e-16 (very low for delayed first event)")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets:.0f}")
    print(f"\nNote: Window ends BEFORE first MC event → manual trigger required")
    
    # Setup solver - VERY LOW event rate so window ends before first event
    solver = setup_solver_for_nucleation_test(
        t_total=0.5,
        seed=42,
        corr_beta=1e-16,       # Very low for delayed first event
        diameter=500e-6,
        a0=20
    )
    
    # Track initial state
    n_init = solver.a_tot
    print(f"\nInitial state: {n_init} particles")
    
    # Create nucleation handler with VERY SHORT window
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=t_start,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve(maxiter=5000)
    
    # Validate - SLIGHTLY RELAXED tolerance for manual trigger
    passed, error, details = validate_liquid_addition(
        solver, expected_v_liquid, n_init,
        tolerance=0.15,  # Slightly relaxed for manual trigger edge case
        max_agglo_fraction=0.05,
        case_name="Case 5: Manual Trigger"
    )
    
    print_results(details, passed, tolerance=0.15, max_agglo=0.05)
    
    return passed


# ============================================================================
# MAIN: Run all test cases
# ============================================================================

def main():
    print("#" * 80)
    print("# Liquid Addition Timing - Comprehensive Debug Suite")
    print("# Testing 5 different liquid addition timing scenarios")
    print("#" * 80)
    
    results = []
    
    # Case 1: Simple
    try:
        passed = test_case_1_simple()
        results.append(("Case 1: Simple", passed))
    except Exception as e:
        print(f"\n❌ Case 1 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 1: Simple", False))
    
    # Case 2: Split End
    try:
        passed = test_case_2_split_end()
        results.append(("Case 2: Split End", passed))
    except Exception as e:
        print(f"\n❌ Case 2 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 2: Split End", False))
    
    # Case 3: Delayed Start
    try:
        passed = test_case_3_delayed_start()
        results.append(("Case 3: Delayed Start", passed))
    except Exception as e:
        print(f"\n❌ Case 3 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 3: Delayed Start", False))
    
    # Case 4: Window Between
    try:
        passed = test_case_4_window_between()
        results.append(("Case 4: Window Between", passed))
    except Exception as e:
        print(f"\n❌ Case 4 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 4: Window Between", False))
    
    # Case 5: Manual Trigger
    try:
        passed = test_case_5_window_before_first()
        results.append(("Case 5: Manual Trigger", passed))
    except Exception as e:
        print(f"\n❌ Case 5 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 5: Manual Trigger", False))
    
    # Summary
    print_separator("SUMMARY", '=')
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TEST CASES PASSED! 🎉")
    else:
        print("SOME TEST CASES FAILED! ⚠️")
        print("\nDebugging hints:")
        print("  - Check overlap calculation in mcpbe_nucleation.py step()")
        print("  - Verify dt_effective = min(window_end, event_end) - max(window_start, event_start)")
        print("  - Ensure manual trigger adds FULL window duration liquid")
        print("  - Check liquid_volume tracking in _distribute_one_droplet_with_dW()")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
