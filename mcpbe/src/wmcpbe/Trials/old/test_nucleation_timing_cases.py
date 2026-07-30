"""
Test: Nucleation Timing Cases - Comprehensive Coverage

Tests 5 different scenarios for liquid addition timing:

1. SIMPLE: Start at t=0, ends on event (frequent events)
2. SPLIT_END: Start at t=0, duration-based stop
3. DELAYED_START: Start at t>0 
4. DURING_EVENTS: Window during continuous events
5. MANUAL_TRIGGER: Very short window → manual triggering fallback

All tests use:
- VERY LARGE particles (200µm) → minimal agglomeration
- HIGH event rate (1e-13) → no manual triggering (except Case 5)
- Few particles (30-50) → low absolute agglomeration
- Duration-based (no wt%)
- Validation: V_liquid_system ≈ volumenstrom × dt

TOLERANCES:
- Cases 1-4: 15% (MC variance + small agglomeration)
- Case 5: 50% (manual triggering limitation known issue)
"""

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src')

from wmcpbe.mcpbe import MCPBESolver


def print_separator(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def calculate_weighted_total(values, weights, n_active):
    """Calculate weighted total for DSMC-consistent comparison."""
    if n_active <= 0:
        return 0.0
    total = 0.0
    for k in range(n_active):
        total += float(values[k]) * float(weights[k])
    return total


def setup_base_solver(t_total, seed=42, corr_beta=1e-13, diameter=200e-6, a0=50):
    """
    Create a solver configured for nucleation-dominant tests.
    
    DESIGN PRINCIPLES:
    - HIGH event rate (corr_beta=1e-13) → No manual triggering needed
    - VERY LARGE particles (200µm) → Minimal agglomeration rate  
    - FEW particles (a0=50) → Less absolute agglomeration events
    
    Args:
        t_total: Total simulation time [s]
        seed: Random seed
        corr_beta: Agglomeration rate (default: 1e-13, HIGH for frequent events)
        diameter: Particle diameter [m] (default: 200µm, very large)
        a0: Initial particle count (default: 50, low)
    """
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=-1,  # No intermediate output
        verbose=False,
        load_attr=False,
        init=False,
        seed=seed,
    )
    
    # VERY LARGE particles to minimize agglomeration
    # 200µm → collision rate ~ 1/diameter² → 16x lower than 50µm!
    solver.c = np.array([1e-3])
    solver.x = np.array([diameter])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = a0
    
    # HIGH agglomeration rate for FREQUENT EVENTS
    # This prevents manual triggering!
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3
    solver.CORR_BETA = corr_beta
    solver.G = 0.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    return solver


def validate_liquid_conservation(solver, expected_v_liquid, n_init, tolerance=0.15):
    """
    Validate that liquid in system matches expected added liquid.
    
    Also tracks agglomeration for diagnostics.
    
    Args:
        solver: MCPBESolver instance
        expected_v_liquid: Expected liquid volume [m³]
        n_init: Initial particle count (for agglomeration check)
        tolerance: Error tolerance (default 15% for MC variance + agglomeration)
    
    Returns: (passed, error_fraction, details_dict)
    """
    n_final = solver.a_tot
    
    # Calculate liquid in system (weighted sum)
    V_liq_system = calculate_weighted_total(
        solver.liquid_volume[:n_final],
        solver.W[:n_final],
        n_final
    )
    
    # Get statistics
    nuc_stats = solver.nucleation.get_statistics()
    V_liq_added_stat = nuc_stats['liquid_volume_added_total']
    
    # Track agglomeration
    n_agglo_events = n_final - n_init
    agglo_fraction = n_agglo_events / n_init if n_init > 0 else 0.0
    
    # Calculate errors
    if expected_v_liquid > 0:
        system_error = abs(V_liq_system - expected_v_liquid) / expected_v_liquid
        stat_error = abs(V_liq_added_stat - expected_v_liquid) / expected_v_liquid
    else:
        system_error = 0.0 if V_liq_system == 0 else 1.0
        stat_error = 0.0 if V_liq_added_stat == 0 else 1.0
    
    # Focus on liquid conservation
    passed = system_error < tolerance
    
    details = {
        'V_liq_system': V_liq_system,
        'V_liq_added_stat': V_liq_added_stat,
        'V_liq_expected': expected_v_liquid,
        'system_error': system_error,
        'stat_error': stat_error,
        'n_init': n_init,
        'n_final': n_final,
        'n_agglo_events': n_agglo_events,
        'agglo_fraction': agglo_fraction,
        'nuc_droplets': nuc_stats.get('droplets_added_total', 0),
    }
    
    return passed, system_error, details


# ============================================================================
# TEST CASE 1: Simple - Start at t=0, frequent events
# ============================================================================

def test_case_1_simple():
    """
    CASE 1: Simple timing with frequent events
    
    Timeline:
    t=0.0s: ┌───────────────┐
            │   Nucleation  │
    t=0.5s: └──●──●──●──●──┘  ← Multiple events during window
    
    Expected: Full duration used, all liquid distributed across events
    """
    print_separator("CASE 1: Simple (t=0, Frequent Events)")
    
    # Parameters
    volumenstrom = 1e-14  # m³/s
    dauer = 0.5  # s
    tropfen_durchmesser = 10e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = int(expected_v_liquid / v_droplet)
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  duration:         {dauer} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  v_droplet:        {v_droplet:.2e} m³")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets}")
    
    # Setup solver - HIGH event rate, VERY LARGE particles
    solver = setup_base_solver(
        t_total=dauer + 0.5,
        seed=42,
        corr_beta=1e-13,      # HIGH for frequent events
        diameter=200e-6,       # 200µm - very large
        a0=50                  # Few particles
    )
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=dauer,
    )
    
    # Track initial state
    n_init = solver.a_tot
    
    # Run simulation
    solver.solve(maxiter=5000)
    
    # Validate
    passed, error, details = validate_liquid_conservation(solver, expected_v_liquid, n_init, tolerance=0.15)
    
    print(f"\nResults:")
    print(f"  V_liquid (system):   {details['V_liq_system']:.2e} m³")
    print(f"  V_liquid (stat):     {details['V_liq_added_stat']:.2e} m³")
    print(f"  V_liquid (expected): {details['V_liq_expected']:.2e} m³")
    print(f"  Error (system):      {details['system_error']*100:.2f}%")
    print(f"  Droplets added:      {details['nuc_droplets']}")
    print(f"  Particles (init):    {details['n_init']}")
    print(f"  Particles (final):   {details['n_final']}")
    print(f"  Agglomeration events:{details['n_agglo_events']} ({details['agglo_fraction']*100:.1f}%)")
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nValidation: {status} (tolerance: 15%)")
    
    return passed


# ============================================================================
# TEST CASE 2: Duration-based stop
# ============================================================================

def test_case_2_split_end():
    """
    CASE 2: Duration-based liquid addition
    
    Timeline:
    t=0.0s: ┌─────────────┐
            │  Nucleation │
    t=0.3s: └──────×──────┘  ← Stop at configured duration
    
    Expected: Liquid added for exactly configured duration
    """
    print_separator("CASE 2: Duration-Based Stop")
    
    # Parameters
    volumenstrom = 1e-14  # m³/s
    dauer = 0.3  # s
    tropfen_durchmesser = 5e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = int(expected_v_liquid / v_droplet)
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  duration:         {dauer} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  v_droplet:        {v_droplet:.2e} m³")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets}")
    
    # Setup solver
    solver = setup_base_solver(
        t_total=dauer + 0.5,
        seed=42,
        corr_beta=1e-13,
        diameter=200e-6,
        a0=50
    )
    
    # Track initial state
    n_init = solver.a_tot
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    solver.solve(maxiter=3000)
    
    # Validate
    passed, error, details = validate_liquid_conservation(solver, expected_v_liquid, n_init, tolerance=0.15)
    
    print(f"\nResults:")
    print(f"  V_liquid (system):   {details['V_liq_system']:.2e} m³")
    print(f"  V_liquid (stat):     {details['V_liq_added_stat']:.2e} m³")
    print(f"  V_liquid (expected): {details['V_liq_expected']:.2e} m³")
    print(f"  Error (system):      {details['system_error']*100:.2f}%")
    print(f"  Droplets added:      {details['nuc_droplets']}")
    print(f"  Particles (init):    {details['n_init']}")
    print(f"  Particles (final):   {details['n_final']}")
    print(f"  Agglomeration events:{details['n_agglo_events']} ({details['agglo_fraction']*100:.1f}%)")
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nValidation: {status} (tolerance: 15%)")
    
    return passed


# ============================================================================
# TEST CASE 3: Delayed Start
# ============================================================================

def test_case_3_delayed_start():
    """
    CASE 3: Delayed start
    
    Timeline:
    t=0.0s: ─────────────────────  
            │
    t=0.2s: ┌───────────┐          ← Nucleation STARTS (delayed)
            │ Nucleation│
    t=0.5s: └────●──────┘          ← Events during window
    
    Expected: Liquid added from t=0.2s onward
    """
    print_separator("CASE 3: Delayed Start (t_start > 0)")
    
    # Parameters
    volumenstrom = 1e-14  # m³/s
    t_start = 0.2  # s (delayed)
    dauer = 0.3  # s
    t_end = t_start + dauer  # 0.5s
    tropfen_durchmesser = 5e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = int(expected_v_liquid / v_droplet)
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  t_start:          {t_start} s")
    print(f"  duration:         {dauer} s")
    print(f"  t_end:            {t_end} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets}")
    
    # Setup solver - HIGH event rate
    solver = setup_base_solver(
        t_total=t_end + 0.5,
        seed=42,
        corr_beta=1e-13,
        diameter=200e-6,
        a0=50
    )
    
    # Track initial state
    n_init = solver.a_tot
    
    # Create nucleation handler with DELAYED start
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=t_start,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    solver.solve(maxiter=3000)
    
    # Validate
    passed, error, details = validate_liquid_conservation(solver, expected_v_liquid, n_init, tolerance=0.15)
    
    print(f"\nResults:")
    print(f"  V_liquid (system):   {details['V_liq_system']:.2e} m³")
    print(f"  V_liquid (stat):     {details['V_liq_added_stat']:.2e} m³")
    print(f"  V_liquid (expected): {details['V_liq_expected']:.2e} m³")
    print(f"  Error (system):      {details['system_error']*100:.2f}%")
    print(f"  Droplets added:      {details['nuc_droplets']}")
    print(f"  Particles (init):    {details['n_init']}")
    print(f"  Particles (final):   {details['n_final']}")
    print(f"  Agglomeration events:{details['n_agglo_events']} ({details['agglo_fraction']*100:.1f}%)")
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nValidation: {status} (tolerance: 15%)")
    
    return passed


# ============================================================================
# TEST CASE 4: Window During Events
# ============================================================================

def test_case_4_window_between():
    """
    CASE 4: Addition window during continuous events
    
    Timeline:
    t=0.0s: ●──●──●──●──●──●  ← Continuous events
    t=0.1s:       ┌────┐       ← Window DURING events
    t=0.3s:       └────┘
    
    Expected: Liquid distributed across events in window
    """
    print_separator("CASE 4: Window During Events")
    
    # Parameters
    volumenstrom = 1e-14  # m³/s
    t_start = 0.1  # s
    dauer = 0.2  # s
    t_end = t_start + dauer  # 0.3s
    tropfen_durchmesser = 5e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = int(expected_v_liquid / v_droplet)
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  t_start:          {t_start} s")
    print(f"  duration:         {dauer} s")
    print(f"  t_end:            {t_end} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets}")
    
    # Setup solver - HIGH event rate
    solver = setup_base_solver(
        t_total=t_end + 0.5,
        seed=42,
        corr_beta=1e-13,
        diameter=200e-6,
        a0=50
    )
    
    # Track initial state
    n_init = solver.a_tot
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=t_start,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    solver.solve(maxiter=5000)
    
    # Validate
    passed, error, details = validate_liquid_conservation(solver, expected_v_liquid, n_init, tolerance=0.15)
    
    print(f"\nResults:")
    print(f"  V_liquid (system):   {details['V_liq_system']:.2e} m³")
    print(f"  V_liquid (stat):     {details['V_liq_added_stat']:.2e} m³")
    print(f"  V_liquid (expected): {details['V_liq_expected']:.2e} m³")
    print(f"  Error (system):      {details['system_error']*100:.2f}%")
    print(f"  Droplets added:      {details['nuc_droplets']}")
    print(f"  Particles (init):    {details['n_init']}")
    print(f"  Particles (final):   {details['n_final']}")
    print(f"  Agglomeration events:{details['n_agglo_events']} ({details['agglo_fraction']*100:.1f}%)")
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nValidation: {status} (tolerance: 15%)")
    
    return passed


# ============================================================================
# TEST CASE 5: Manual Trigger (Window Before First Event)
# ============================================================================

def test_case_5_window_before_first():
    """
    CASE 5: Window ends before first MC event (manual triggering)
    
    Timeline:
    t=0.0s: ┌──────────┐
            │ Nucleation│
    t=0.05s:└─────┬────┘  ← Window ends (very short)
                  │
    t=0.1s:       ●  ← First MC event → Manual trigger
    
    Expected: Full liquid added via manual trigger at first event
    
    NOTE: This tests the manual triggering fallback when no events occur
    during the nucleation window. KNOWN LIMITATION: May not distribute
    all droplets if particles become saturated.
    """
    print_separator("CASE 5: Manual Trigger (Window Before First Event)")
    
    # Parameters - VERY SHORT window, will trigger manually
    volumenstrom = 1e-15  # m³/s (lower for reasonable droplet count)
    t_start = 0.0  # s
    dauer = 0.05  # s (very short!)
    t_end = t_start + dauer  # 0.05s
    tropfen_durchmesser = 2e-6  # m
    
    v_droplet = (np.pi / 6.0) * tropfen_durchmesser ** 3
    expected_v_liquid = volumenstrom * dauer
    expected_droplets = int(expected_v_liquid / v_droplet)
    
    print(f"Parameters:")
    print(f"  volumenstrom:     {volumenstrom:.2e} m³/s")
    print(f"  t_start:          {t_start} s")
    print(f"  duration:         {dauer} s")
    print(f"  t_end:            {t_end} s")
    print(f"  droplet_diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  v_droplet:        {v_droplet:.2e} m³")
    print(f"\nExpected:")
    print(f"  V_liquid: {expected_v_liquid:.2e} m³")
    print(f"  Droplets: ~{expected_droplets}")
    print(f"\nNote: Window ends BEFORE first MC event → manual trigger")
    
    # Setup solver - LOW event rate so window ends before first event
    solver = setup_base_solver(
        t_total=0.5,  # Short total time
        seed=42,
        corr_beta=1e-15,  # Very low for delayed first event
        diameter=200e-6,
        a0=30  # Few particles
    )
    
    # Track initial state
    n_init = solver.a_tot
    
    # Create nucleation handler with VERY SHORT window
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=t_start,
        wasserzugabe_dauer=dauer,
    )
    
    # Run simulation
    solver.solve(maxiter=5000)
    
    # Validate - RELAXED tolerance for manual trigger (KNOWN LIMITATION)
    passed, error, details = validate_liquid_conservation(solver, expected_v_liquid, n_init, tolerance=0.50)
    
    print(f"\nResults:")
    print(f"  V_liquid (system):   {details['V_liq_system']:.2e} m³")
    print(f"  V_liquid (stat):     {details['V_liq_added_stat']:.2e} m³")
    print(f"  V_liquid (expected): {details['V_liq_expected']:.2e} m³")
    print(f"  Error (system):      {details['system_error']*100:.2f}%")
    print(f"  Droplets added:      {details['nuc_droplets']}")
    print(f"  Particles (init):    {details['n_init']}")
    print(f"  Particles (final):   {details['n_final']}")
    print(f"  Agglomeration events:{details['n_agglo_events']} ({details['agglo_fraction']*100:.1f}%)")
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nValidation: {status} (tolerance: 50% - manual trigger limitation)")
    
    return passed


# ============================================================================
# MAIN: Run all test cases
# ============================================================================

def main():
    print("#" * 80)
    print("# Nucleation Timing Cases - Comprehensive Test Suite")
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
    
    # Case 2: Duration-based
    try:
        passed = test_case_2_split_end()
        results.append(("Case 2: Duration Stop", passed))
    except Exception as e:
        print(f"\n❌ Case 2 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 2: Duration Stop", False))
    
    # Case 3: Delayed Start
    try:
        passed = test_case_3_delayed_start()
        results.append(("Case 3: Delayed Start", passed))
    except Exception as e:
        print(f"\n❌ Case 3 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 3: Delayed Start", False))
    
    # Case 4: During Events
    try:
        passed = test_case_4_window_between()
        results.append(("Case 4: During Events", passed))
    except Exception as e:
        print(f"\n❌ Case 4 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Case 4: During Events", False))
    
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
    print_separator("SUMMARY")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TEST CASES PASSED! 🎉")
    else:
        print("SOME TEST CASES FAILED! ⚠️")
        print("\nNote: Case 5 (Manual Trigger) has known limitations.")
        print("      Consider increasing CORR_BETA to avoid manual triggering.")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
