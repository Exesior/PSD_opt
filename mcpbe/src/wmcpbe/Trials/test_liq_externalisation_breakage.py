"""
Validation Test for Liquid Externalization during Breakage.

Reverse process of the Braumann et al. 2007 internalization model
(see test_liq_internalisation_agglomeration.py): when a particle
fragments, pore volume lost to new fracture surfaces pushes internal
liquid out to become external liquid.

    Delta_V_pore = V_pore_parent - V_pore_fragments_total
    V_liq_int_to_ext = S_parent * Delta_V_pore

Part 1 tests the kernel formula (LiquidInternalisationAgglomerationKernel
.compute_externalization) in isolation against a manually computed
expected value.

Part 2 runs ONE REAL breakage event through the actual production code
path (MCPBEBreak._break_apply_and_maintain, the method touched by this
feature) with hand-picked, exactly-reproducible input numbers, and
compares the resulting fragment particles (liquid_volume, porosity,
saturation, mass conservation) against independently computed expected
values -- i.e. the same formulas, re-implemented from scratch in this
test file, NOT by calling the kernel/production functions under test.

The porosity growth kernel is replaced by a small deterministic fake
(fixed per-fragment porosities, not proportional to fragment size) so
that the pore-volume loss is an exact, arbitrary, hand-checkable number
that isolates the code under test from the (separate, pre-existing)
cone_model geometry kernel.

Usage:
    python -m wmcpbe.Trials.test_liq_externalisation_breakage
"""

import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from wmcpbe import MCPBESolver
from wmcpbe.kernels.continuous_processes.liq_internalisation_agglomeration import (
    LiquidInternalisationAgglomerationKernel,
)


# =============================================================================
# Part 1: Kernel formula, isolated, hand-calculated
# =============================================================================

def test_kernel_formula_hand_calculated():
    """Test 1: compute_externalization() vs. manual formula evaluation."""
    print("=" * 70)
    print("TEST 1: compute_externalization() Formula (Hand-Calculated)")
    print("=" * 70)

    kernel = LiquidInternalisationAgglomerationKernel()

    cases = [
        # (v_pore_parent, v_pore_fragments_total, saturation_parent)
        (1.0e-18, 8.0e-19, 0.5),
        (5.0e-18, 5.0e-18, 0.7),   # no pore loss -> 0
        (2.0e-18, 0.0, 1.0),       # all pore volume lost, fully saturated
        (3.0e-18, 1.0e-18, 0.0),   # dry parent -> 0
    ]

    for v_pore_parent, v_pore_frag_total, s_p in cases:
        delta_v_pore = v_pore_parent - v_pore_frag_total
        expected = s_p * delta_v_pore if delta_v_pore > 0.0 else 0.0
        expected = min(expected, s_p * v_pore_parent)  # clamp, mirrors kernel

        result = kernel.compute_externalization(
            v_pore_parent=v_pore_parent,
            v_pore_fragments_total=v_pore_frag_total,
            saturation_parent=s_p,
        )

        diff = abs(result - expected)
        status = "OK" if diff < 1e-30 else "MISMATCH"
        print(f"  V_pore_p={v_pore_parent:.3e}  V_pore_frag={v_pore_frag_total:.3e}  "
              f"S_p={s_p:.2f}  ->  expected={expected:.6e}  got={result:.6e}  [{status}]")
        assert diff < 1e-30, f"Kernel result {result} != manual calc {expected}"

    print("[OK] Formula matches manual calculation for all cases\n")


# =============================================================================
# Part 2: Real breakage event through _break_apply_and_maintain,
#         hand-calculated expected fragment state
# =============================================================================

class _FakeExternalizingPorosityKernel:
    """
    Deterministic stand-in for the porosity growth kernel.

    Returns FIXED per-fragment porosities that are NOT proportional to
    fragment size (unlike cone_model, which distributes pore volume
    proportional to V_solid). This isolates the liquid-bookkeeping code
    under test (mcpbe_break.py) from the separate, pre-existing
    cone_model geometry, and additionally validates that the aggregate
    conservation law (V_liq_int_to_ext = S_p * Delta_V_pore) holds even
    when pore volume is NOT distributed proportionally to fragment
    liquid share -- the general case.
    """

    def __init__(self, fixed_porosities):
        self.fixed_porosities = list(fixed_porosities)

    def _compute_fragment_porosity_multi(self, parent_porosity, fragment_volumes,
                                          parent_volume, breakage_energy=None, solver=None):
        assert len(fragment_volumes) == len(self.fixed_porosities)
        return list(self.fixed_porosities)

    def compute_fragment_porosity(self, parent_porosity, fragment_volume,
                                   parent_volume, breakage_energy=None, solver=None):
        # Not expected to be called (2 fragments -> deferred multi path),
        # kept only for interface completeness.
        return self.fixed_porosities[0]


def _build_minimal_solver():
    """One-particle solver, just enough machinery to call
    _break_apply_and_maintain directly (bypasses event sampling)."""
    solver = MCPBESolver(
        dim=1,
        t_vec=np.array([0.0, 1.0]),
        verbose=False,
        load_attr=False,
        init=True,
        seed=42,
        maybe_double_control_volume=False,
        recon_enable=False,
        agg_kernel_name='constant',
        agg_kernel_params={'beta0': 0.0},
        break_kernel_name='power_law',
        break_kernel_params={'p1': 1.0, 'p2': 1.0, 'g': 1000.0, 'breakrval': 4},
        porosity_growth_kernel_name='volume_mixing',
        porosity_growth_kernel_params={},
        liq_internalisation_agglomeration_kernel_name='liq_internalisation_agglomeration',
        liq_internalisation_agglomeration_kernel_params={},
    )
    solver.process_type = 'breakage'
    solver.Vc = 1.0

    V_flat = np.zeros((2, 1), dtype=float)
    V_solid_p = 8.0e-18
    poro_p = 0.5
    V_dry_p = V_solid_p / (1.0 - poro_p)
    V_flat[0, 0] = V_solid_p
    V_flat[1, 0] = V_dry_p

    W_init = np.array([100.0], dtype=float)

    solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)

    solver.porosity[:solver.a_tot] = poro_p
    S_p = 0.5
    lv_parent = 6.0e-18
    solver.liquid_volume[:solver.a_tot] = lv_parent
    solver.saturation[:solver.a_tot] = S_p

    # Swap in the deterministic fake porosity kernel (fragment porosities
    # 0.4 / 0.3, chosen arbitrarily, NOT proportional to fragment size).
    solver.kernel_manager.porosity_growth_kernel = _FakeExternalizingPorosityKernel([0.4, 0.3])

    return solver, V_solid_p, V_dry_p, poro_p, S_p, lv_parent


def test_breakage_event_hand_calculated():
    """Test 2: One real breakage event vs. independently computed expectation."""
    print("=" * 70)
    print("TEST 2: Real Breakage Event (_break_apply_and_maintain)")
    print("=" * 70)

    solver, V_solid_p, V_dry_p, poro_p, S_p, lv_parent = _build_minimal_solver()

    W_parent = float(solver.W[0])
    dW = 20.0

    # Two fragments, unequal split: 5e-18 / 3e-18 solid volume (sum = V_solid_p)
    V_solid_1 = 5.0e-18
    V_solid_2 = 3.0e-18
    frags = [np.array([V_solid_1]), np.array([V_solid_2])]
    Vrem_k = np.array([V_solid_p])

    print(f"\nInput (parent particle k=0, before breakage):")
    print(f"  V_solid_p   = {V_solid_p:.6e} m^3")
    print(f"  V_dry_p     = {V_dry_p:.6e} m^3")
    print(f"  porosity_p  = {poro_p:.4f}")
    print(f"  liquid_p    = {lv_parent:.6e} m^3")
    print(f"  saturation_p= {S_p:.4f}")
    print(f"  W[0]        = {W_parent:.2f}, dW = {dW:.2f}")
    print(f"  Fragments (solid volume): {V_solid_1:.3e}, {V_solid_2:.3e} m^3")
    print(f"  Fragment porosities (fake kernel, fixed): 0.4, 0.3")

    # -------------------------------------------------------------------
    # Hand-calculated expectation (independent re-implementation of the
    # formulas -- does NOT call any of the code under test).
    # -------------------------------------------------------------------
    vol_fraction_1 = V_solid_1 / (V_solid_1 + V_solid_2)
    vol_fraction_2 = V_solid_2 / (V_solid_1 + V_solid_2)

    frag_poro_1, frag_poro_2 = 0.4, 0.3
    V_dry_frag_1 = V_solid_1 / (1.0 - frag_poro_1)
    V_dry_frag_2 = V_solid_2 / (1.0 - frag_poro_2)
    V_pore_frag_1 = frag_poro_1 * V_dry_frag_1
    V_pore_frag_2 = frag_poro_2 * V_dry_frag_2
    V_pore_fragments_total = V_pore_frag_1 + V_pore_frag_2

    V_pore_p = V_dry_p * poro_p
    V_intern_p = S_p * V_pore_p
    V_extern_p = lv_parent - V_intern_p

    delta_v_pore = V_pore_p - V_pore_fragments_total
    V_liq_i2e_expected = S_p * delta_v_pore  # no clamping needed here (checked below)
    assert V_liq_i2e_expected <= V_intern_p, "test setup invalid: would need clamping"

    V_intern_frag_1 = V_intern_p * vol_fraction_1 - V_liq_i2e_expected * vol_fraction_1
    V_extern_frag_1 = V_extern_p * vol_fraction_1 + V_liq_i2e_expected * vol_fraction_1
    V_intern_frag_2 = V_intern_p * vol_fraction_2 - V_liq_i2e_expected * vol_fraction_2
    V_extern_frag_2 = V_extern_p * vol_fraction_2 + V_liq_i2e_expected * vol_fraction_2

    liq_frag_1_expected = V_intern_frag_1 + V_extern_frag_1
    liq_frag_2_expected = V_intern_frag_2 + V_extern_frag_2
    sat_frag_1_expected = V_intern_frag_1 / V_pore_frag_1
    sat_frag_2_expected = V_intern_frag_2 / V_pore_frag_2

    print(f"\nHand-calculated expectation:")
    print(f"  V_pore_p              = {V_pore_p:.6e} m^3")
    print(f"  V_intern_p            = {V_intern_p:.6e} m^3")
    print(f"  V_extern_p            = {V_extern_p:.6e} m^3")
    print(f"  V_pore_fragments_total= {V_pore_fragments_total:.6e} m^3")
    print(f"  Delta_V_pore          = {delta_v_pore:.6e} m^3")
    print(f"  V_liq_int->ext (S_p*DeltaV_pore) = {V_liq_i2e_expected:.6e} m^3")
    print(f"  Fragment 1: liquid={liq_frag_1_expected:.6e} m^3, saturation={sat_frag_1_expected:.6f}")
    print(f"  Fragment 2: liquid={liq_frag_2_expected:.6e} m^3, saturation={sat_frag_2_expected:.6f}")
    print(f"  Sum(liquid) = {liq_frag_1_expected + liq_frag_2_expected:.6e} m^3 "
          f"(parent liquid = {lv_parent:.6e} m^3)")

    # -------------------------------------------------------------------
    # Run the REAL production code path.
    # -------------------------------------------------------------------
    solver._break_apply_and_maintain(k=0, frags=frags, dW=dW, Vrem_k=Vrem_k)

    assert solver.a_tot == 3, f"Expected 3 particles after breakage (parent + 2 fragments), got {solver.a_tot}"

    idx1, idx2 = 1, 2
    liq_frag_1_actual = float(solver.liquid_volume[idx1])
    liq_frag_2_actual = float(solver.liquid_volume[idx2])
    sat_frag_1_actual = float(solver.saturation[idx1])
    sat_frag_2_actual = float(solver.saturation[idx2])
    poro_frag_1_actual = float(solver.porosity[idx1])
    poro_frag_2_actual = float(solver.porosity[idx2])

    print(f"\nActual result from _break_apply_and_maintain (production code):")
    print(f"  Fragment 1 (idx={idx1}): liquid={liq_frag_1_actual:.6e} m^3, "
          f"porosity={poro_frag_1_actual:.4f}, saturation={sat_frag_1_actual:.6f}")
    print(f"  Fragment 2 (idx={idx2}): liquid={liq_frag_2_actual:.6e} m^3, "
          f"porosity={poro_frag_2_actual:.4f}, saturation={sat_frag_2_actual:.6f}")
    print(f"  Sum(liquid) = {liq_frag_1_actual + liq_frag_2_actual:.6e} m^3")
    print(f"  Parent liquid (unchanged, idx=0) = {float(solver.liquid_volume[0]):.6e} m^3")
    print(f"  Parent W after breakage (idx=0)  = {float(solver.W[0]):.2f} "
          f"(expected {W_parent - dW:.2f})")

    tol = 1e-27  # absolute tolerance, volumes are ~1e-18 m^3
    print(f"\nComparison (tolerance = {tol:.0e} m^3 absolute):")

    checks = [
        ("liquid_volume[frag1]", liq_frag_1_actual, liq_frag_1_expected),
        ("liquid_volume[frag2]", liq_frag_2_actual, liq_frag_2_expected),
        ("saturation[frag1]", sat_frag_1_actual, sat_frag_1_expected),
        ("saturation[frag2]", sat_frag_2_actual, sat_frag_2_expected),
        ("porosity[frag1]", poro_frag_1_actual, frag_poro_1),
        ("porosity[frag2]", poro_frag_2_actual, frag_poro_2),
    ]
    all_ok = True
    for label, actual, expected in checks:
        diff = abs(actual - expected)
        ok = diff < 1e-6 if "saturation" in label or "porosity" in label else diff < tol
        status = "OK" if ok else "MISMATCH"
        all_ok = all_ok and ok
        print(f"  {label:<24} actual={actual:.10e}  expected={expected:.10e}  "
              f"diff={diff:.3e}  [{status}]")

    total_liquid_after = liq_frag_1_actual + liq_frag_2_actual
    liquid_conserved = abs(total_liquid_after - lv_parent) < tol
    print(f"  {'liquid conservation':<24} sum(frags)={total_liquid_after:.10e}  "
          f"parent={lv_parent:.10e}  diff={abs(total_liquid_after - lv_parent):.3e}  "
          f"[{'OK' if liquid_conserved else 'MISMATCH'}]")

    no_nan = not (np.isnan(poro_frag_1_actual) or np.isnan(poro_frag_2_actual)
                  or np.isnan(sat_frag_1_actual) or np.isnan(sat_frag_2_actual))
    print(f"  {'no NaN in poro/sat':<24} {'[OK]' if no_nan else '[MISMATCH]'}")

    assert all_ok, "Fragment properties do not match hand-calculated expectation"
    assert liquid_conserved, "Total liquid not conserved across breakage"
    assert no_nan, "NaN found in fragment porosity/saturation"

    print("\n[OK] Real breakage event matches hand-calculated expectation exactly\n")


def run_all_tests():
    print("\n")
    print("*" * 70)
    print("* LIQUID EXTERNALIZATION DURING BREAKAGE - VALIDATION TEST SUITE")
    print("* Reverse Braumann process: Delta_V_pore * S = V_liq_int->ext")
    print("*" * 70)
    print("\n")

    try:
        test_kernel_formula_hand_calculated()
        test_breakage_event_hand_calculated()

        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
        print()

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}\n")
        raise


if __name__ == '__main__':
    run_all_tests()
