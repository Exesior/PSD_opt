"""Liquid volume mass conservation test for WMCPBE.

This test verifies that liquid volume is conserved during:
1. Agglomeration events
2. Breakage events
3. Mixed agglomeration + breakage

Run with: python test_liquid_conservation.py
"""

from __future__ import annotations

import numpy as np
from validation import (
    CaseConfig,
    LiquidConservationValidationConfig,
    LiquidConservationRunner,
    WMCPBEVariantConfig,
    check_mass_conservation,
)


def run_agglomeration_test() -> bool:
    """Test liquid conservation during pure agglomeration."""
    print("\n" + "=" * 70)
    print("TEST 1: Agglomeration - Liquid Volume Conservation")
    print("=" * 70)

    case = CaseConfig(
        dim=1,
        kernel="const",
        process="agglomeration",
        t_vec=np.linspace(0.0, 50.0, 11, dtype=float),
        x=2e-6,
        beta0=1e-9,
        initial_number_density=1e8,
        initial_total_weight=100000.0,
        initial_liquid_fraction=0.1,  # 10% of particle volume is liquid
    )

    wmcpbe_variants = [
        WMCPBEVariantConfig(
            name="WMCPBE Agg Baseline",
            repeats=1,
            base_seed=42,
            attrs={
                "break_dW_max": 1.0,
                "agg_dW_max": 5.0,
            },
        ),
    ]

    config = LiquidConservationValidationConfig(
        case=case,
        wmcpbe_variants=wmcpbe_variants,
        verbose=False,
    )

    result = LiquidConservationRunner(config).run()
    
    # Print results
    for name, method in result.methods.items():
        initial_liq = method.meta["initial_liquid_total"]
        final_liq = method.meta["final_liquid_total"]
        rel_error = method.meta["relative_error"]
        
        print(f"\n{name}:")
        print(f"  Initial liquid volume: {initial_liq:.6e} m³")
        print(f"  Final liquid volume:   {final_liq:.6e} m³")
        print(f"  Relative error:        {rel_error:.6e} ({rel_error*100:.4f}%)")
    
    passed = check_mass_conservation(result, tolerance=1e-10)
    all_passed = all(passed.values())
    
    print(f"\nResult: {'✅ PASS' if all_passed else '❌ FAIL'}")
    return all_passed


def run_breakage_test() -> bool:
    """Test liquid conservation during pure breakage."""
    print("\n" + "=" * 70)
    print("TEST 2: Breakage - Liquid Volume Conservation")
    print("=" * 70)

    case = CaseConfig(
        dim=1,
        kernel="const",
        process="breakage",
        t_vec=np.linspace(0.0, 50.0, 11, dtype=float),
        x=10e-6,  # Larger particles for breakage
        beta0=1e-9,
        p1=0.5,  # Breakage rate
        initial_number_density=1e8,
        initial_total_weight=100000.0,
        initial_liquid_fraction=0.1,  # 10% of particle volume is liquid
    )

    wmcpbe_variants = [
        WMCPBEVariantConfig(
            name="WMCPBE Break Baseline",
            repeats=1,
            base_seed=42,
            attrs={
                "break_dW_max": 10.0,
                "agg_dW_max": 1.0,
            },
        ),
    ]

    config = LiquidConservationValidationConfig(
        case=case,
        wmcpbe_variants=wmcpbe_variants,
        verbose=False,
    )

    result = LiquidConservationRunner(config).run()
    
    # Print results
    for name, method in result.methods.items():
        initial_liq = method.meta["initial_liquid_total"]
        final_liq = method.meta["final_liquid_total"]
        rel_error = method.meta["relative_error"]
        
        print(f"\n{name}:")
        print(f"  Initial liquid volume: {initial_liq:.6e} m³")
        print(f"  Final liquid volume:   {final_liq:.6e} m³")
        print(f"  Relative error:        {rel_error:.6e} ({rel_error*100:.4f}%)")
    
    passed = check_mass_conservation(result, tolerance=1e-10)
    all_passed = all(passed.values())
    
    print(f"\nResult: {'✅ PASS' if all_passed else '❌ FAIL'}")
    return all_passed


def run_mix_test() -> bool:
    """Test liquid conservation during mixed agglomeration + breakage."""
    print("\n" + "=" * 70)
    print("TEST 3: Mix (Agg + Break) - Liquid Volume Conservation")
    print("=" * 70)

    case = CaseConfig(
        dim=1,
        kernel="const",
        process="mix",
        t_vec=np.linspace(0.0, 50.0, 11, dtype=float),
        x=5e-6,
        beta0=1e-9,
        p1=0.3,  # Breakage rate
        initial_number_density=1e8,
        initial_total_weight=100000.0,
        initial_liquid_fraction=0.1,  # 10% of particle volume is liquid
    )

    wmcpbe_variants = [
        WMCPBEVariantConfig(
            name="WMCPBE Mix Baseline",
            repeats=1,
            base_seed=42,
            attrs={
                "break_dW_max": 10.0,
                "agg_dW_max": 5.0,
            },
        ),
    ]

    config = LiquidConservationValidationConfig(
        case=case,
        wmcpbe_variants=wmcpbe_variants,
        verbose=False,
    )

    result = LiquidConservationRunner(config).run()
    
    # Print results
    for name, method in result.methods.items():
        initial_liq = method.meta["initial_liquid_total"]
        final_liq = method.meta["final_liquid_total"]
        rel_error = method.meta["relative_error"]
        
        print(f"\n{name}:")
        print(f"  Initial liquid volume: {initial_liq:.6e} m³")
        print(f"  Final liquid volume:   {final_liq:.6e} m³")
        print(f"  Relative error:        {rel_error:.6e} ({rel_error*100:.4f}%)")
    
    passed = check_mass_conservation(result, tolerance=1e-10)
    all_passed = all(passed.values())
    
    print(f"\nResult: {'✅ PASS' if all_passed else '❌ FAIL'}")
    return all_passed


def main():
    """Run all liquid conservation tests."""
    print("\n" + "#" * 70)
    print("# Liquid Volume Mass Conservation Tests")
    print("#"*70)
    
    results = []
    
    # Test 1: Agglomeration
    try:
        results.append(("Agglomeration", run_agglomeration_test()))
    except Exception as e:
        print(f"\n❌ Agglomeration test FAILED with exception: {e}")
        results.append(("Agglomeration", False))
    
    # Test 2: Breakage
    try:
        results.append(("Breakage", run_breakage_test()))
    except Exception as e:
        print(f"\n❌ Breakage test FAILED with exception: {e}")
        results.append(("Breakage", False))
    
    # Test 3: Mix
    try:
        results.append(("Mix", run_mix_test()))
    except Exception as e:
        print(f"\n❌ Mix test FAILED with exception: {e}")
        results.append(("Mix", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! 🎉")
    else:
        print("SOME TESTS FAILED! ⚠️")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
