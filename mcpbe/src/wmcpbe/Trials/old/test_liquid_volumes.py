"""
Test for Liquid Volume Utilities (Phase 2).

This test validates the new liquid volume calculation methods:
- get_V_pore(): Pore volume = V_ges * porosity
- get_V_liquid_internal(): Internal liquid = V_ges * porosity * saturation
- get_V_liquid_external(): External liquid = V_liquid_total - V_liquid_internal

These utilities are needed for advanced breakage and agglomeration models.

Usage:
    %runfile C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src/wmcpbe/Trials/test_liquid_volumes.py --wdir
"""

import numpy as np
import sys
import os

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_separator(title: str) -> None:
    """Print a formatted section separator."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_liquid_volume_utilities():
    """
    Test liquid volume calculation utilities.
    
    Creates particles with known properties and verifies:
    1. V_pore = V_ges * porosity
    2. V_liq_int = V_ges * porosity * saturation
    3. V_liq_ext = V_liq_total - V_liq_int
    4. Mass conservation relationships
    """
    from wmcpbe.mcpbe import MCPBESolver
    
    print_separator("Liquid Volume Utilities Test")
    
    # Create solver with simple monodisperse initial condition
    solver = MCPBESolver(
        dim=1,
        t_total=0.0,      # No simulation needed
        t_write=1.0,
        verbose=False,
        init=True,
        seed=42,
        load_attr=False,
    )
    
    n_particles = solver.a_tot
    
    # Set up test case: all particles have porosity=0.4 and saturation=0.5
    test_porosity = 0.4
    test_saturation = 0.5
    
    solver.porosity[:n_particles] = test_porosity
    solver.saturation[:n_particles] = test_saturation
    
    # Add known liquid volume
    # For testing: set liquid_volume such that saturation = 0.5
    # V_liq_total = V_ges * poro * sat
    V_ges = solver.get_V_total()
    V_liq_target = V_ges * test_porosity * test_saturation
    solver.liquid_volume[:n_particles] = V_liq_target
    
    print(f"\nTest Setup:")
    print(f"  Particles: {n_particles}")
    print(f"  Target porosity: {test_porosity}")
    print(f"  Target saturation: {test_saturation}")
    print(f"  Mean V_ges: {np.mean(V_ges):.6e} m³")
    
    # Calculate volumes using new utilities
    V_pore = solver.get_V_pore()
    V_liq_int = solver.get_V_liquid_internal()
    V_liq_ext = solver.get_V_liquid_external()
    V_solid = solver.get_V_solid()
    
    print(f"\nCalculated Volumes (mean values):")
    print(f"  V_ges (total):     {np.mean(V_ges):.6e} m³")
    print(f"  V_solid:           {np.mean(V_solid):.6e} m³")
    print(f"  V_pore:            {np.mean(V_pore):.6e} m³")
    print(f"  V_liq_internal:    {np.mean(V_liq_int):.6e} m³")
    print(f"  V_liq_external:    {np.mean(V_liq_ext):.6e} m³")
    
    # Validation 1: V_ges = V_solid + V_pore
    V_reconstructed = V_solid + V_pore
    error_Vges = np.abs(V_reconstructed - V_ges) / V_ges
    print(f"\nValidation 1: V_ges = V_solid + V_pore")
    print(f"  Max relative error: {np.max(error_Vges)*100:.6f}%")
    
    # Validation 2: V_pore = V_ges * porosity
    V_pore_expected = V_ges * test_porosity
    error_Vpore = np.abs(V_pore - V_pore_expected) / V_pore_expected
    print(f"\nValidation 2: V_pore = V_ges * porosity")
    print(f"  Max relative error: {np.max(error_Vpore)*100:.6f}%")
    
    # Validation 3: V_liq_int = V_ges * porosity * saturation
    V_liq_int_expected = V_ges * test_porosity * test_saturation
    error_Vliqint = np.abs(V_liq_int - V_liq_int_expected) / V_liq_int_expected
    print(f"\nValidation 3: V_liq_int = V_ges * porosity * saturation")
    print(f"  Max relative error: {np.max(error_Vliqint)*100:.6f}%")
    
    # Validation 4: V_liq_ext = V_liq_total - V_liq_int
    V_liq_total = solver.liquid_volume[:n_particles]
    V_liq_ext_expected = V_liq_total - V_liq_int
    error_Vliqext = np.abs(V_liq_ext - V_liq_ext_expected)
    print(f"\nValidation 4: V_liq_ext = V_liq_total - V_liq_int")
    print(f"  Max absolute error: {np.max(error_Vliqext):.6e} m³")
    
    # Validation 5: Saturation consistency
    # sat = V_liq_int / V_pore (where V_pore > 0)
    valid_pores = V_pore > 0
    sat_calculated = np.zeros_like(V_pore)
    sat_calculated[valid_pores] = V_liq_int[valid_pores] / V_pore[valid_pores]
    error_sat = np.abs(sat_calculated[valid_pores] - test_saturation) / test_saturation
    print(f"\nValidation 5: Saturation = V_liq_int / V_pore")
    print(f"  Max relative error: {np.max(error_sat)*100:.6f}%")
    
    # Overall validation
    all_errors = [
        np.max(error_Vges),
        np.max(error_Vpore),
        np.max(error_Vliqint),
        np.max(error_Vliqext),
        np.max(error_sat),
    ]
    max_error = np.max(all_errors)
    
    print(f"\n{'='*70}")
    print(f"Maximum error across all validations: {max_error*100:.6f}%")
    
    assert max_error < 1e-10, f"Error too large: {max_error*100:.6f}%"
    print(f"✅ Test PASSED: All liquid volume calculations correct!")
    
    return {
        'V_ges': V_ges,
        'V_solid': V_solid,
        'V_pore': V_pore,
        'V_liq_int': V_liq_int,
        'V_liq_ext': V_liq_ext,
        'errors': {
            'Vges': error_Vges,
            'Vpore': error_Vpore,
            'Vliqint': error_Vliqint,
            'Vliqext': error_Vliqext,
            'sat': error_sat,
        }
    }


def test_vollkemer_case():
    """
    Test liquid volume utilities for Vollkörper (no porosity, NaN).
    
    For particles without porosity (NaN):
    - V_pore = 0
    - V_liq_int = 0
    - V_liq_ext = V_liq_total (all liquid is external)
    """
    from wmcpbe.mcpbe import MCPBESolver
    
    print_separator("Vollkörper Case (NaN Porosity)")
    
    solver = MCPBESolver(
        dim=1,
        t_total=0.0,
        t_write=1.0,
        verbose=False,
        init=True,
        seed=42,
        load_attr=False,
    )
    
    n_particles = solver.a_tot
    
    # Porosity is already NaN from initialization (Vollkörper)
    # Add some liquid volume (all must be external)
    test_liquid = 1e-20  # Small liquid volume
    solver.liquid_volume[:n_particles] = test_liquid
    
    print(f"\nTest Setup:")
    print(f"  Particles: {n_particles}")
    print(f"  Porosity: NaN (Vollkörper)")
    print(f"  Liquid volume (total): {test_liquid:.6e} m³")
    
    # Calculate volumes
    V_pore = solver.get_V_pore()
    V_liq_int = solver.get_V_liquid_internal()
    V_liq_ext = solver.get_V_liquid_external()
    
    print(f"\nResults:")
    print(f"  V_pore:          {np.mean(V_pore):.6e} m³ (should be 0)")
    print(f"  V_liq_internal:  {np.mean(V_liq_int):.6e} m³ (should be 0)")
    print(f"  V_liq_external:  {np.mean(V_liq_ext):.6e} m³ (should equal total)")
    
    # Validations
    assert np.all(V_pore == 0.0), "V_pore should be 0 for Vollkörper"
    assert np.all(V_liq_int == 0.0), "V_liq_int should be 0 for Vollkörper"
    assert np.allclose(V_liq_ext, test_liquid), "V_liq_ext should equal total liquid"
    
    print(f"\n✅ Vollkörper test PASSED!")
    
    return {'V_pore': V_pore, 'V_liq_int': V_liq_int, 'V_liq_ext': V_liq_ext}


def test_mixed_population():
    """
    Test with mixed population: some Vollkörper, some porous.
    """
    from wmcpbe.mcpbe import MCPBESolver
    
    print_separator("Mixed Population Test")
    
    solver = MCPBESolver(
        dim=1,
        t_total=0.0,
        t_write=1.0,
        verbose=False,
        init=True,
        seed=42,
        load_attr=False,
    )
    
    n_particles = solver.a_tot
    
    # First half: Vollkörper (NaN porosity)
    # Second half: Porous (poro=0.4, sat=0.6)
    mid = n_particles // 2
    
    solver.porosity[mid:] = 0.4
    solver.saturation[mid:] = 0.6
    
    # Add liquid only to porous particles
    V_ges = solver.get_V_total()
    V_liq_target = np.zeros(n_particles)
    V_liq_target[mid:] = V_ges[mid:] * 0.4 * 0.6
    solver.liquid_volume[:n_particles] = V_liq_target
    
    print(f"\nTest Setup:")
    print(f"  Total particles: {n_particles}")
    print(f"  Vollkörper (0-{mid}): porosity=NaN")
    print(f"  Porous ({mid}-{n_particles}): porosity=0.4, saturation=0.6")
    
    # Calculate volumes
    V_pore = solver.get_V_pore()
    V_liq_int = solver.get_V_liquid_internal()
    V_liq_ext = solver.get_V_liquid_external()
    
    print(f"\nVollköper subset (0-{mid}):")
    print(f"  V_pore:         {np.mean(V_pore[:mid]):.6e} m³ (should be 0)")
    print(f"  V_liq_int:      {np.mean(V_liq_int[:mid]):.6e} m³ (should be 0)")
    print(f"  V_liq_ext:      {np.mean(V_liq_ext[:mid]):.6e} m³ (should be 0)")
    
    print(f"\nPorous subset ({mid}-{n_particles}):")
    print(f"  V_pore:         {np.mean(V_pore[mid:]):.6e} m³")
    print(f"  V_liq_int:      {np.mean(V_liq_int[mid:]):.6e} m³")
    print(f"  V_liq_ext:      {np.mean(V_liq_ext[mid:]):.6e} m³ (should be ~0)")
    
    # Validations
    assert np.all(V_pore[:mid] == 0.0), "Vollkörper should have no pores"
    assert np.all(V_liq_int[:mid] == 0.0), "Vollkörper should have no internal liquid"
    assert np.all(V_liq_ext[:mid] == 0.0), "Vollkörper should have no external liquid (none added)"
    
    assert np.all(V_pore[mid:] > 0.0), "Porous particles should have pores"
    assert np.all(V_liq_int[mid:] > 0.0), "Porous particles should have internal liquid"
    
    # Check saturation for porous particles
    sat_calc = V_liq_int[mid:] / V_pore[mid:]
    sat_error = np.abs(sat_calc - 0.6) / 0.6
    assert np.max(sat_error) < 1e-10, f"Saturation error too large: {np.max(sat_error)*100:.6f}%"
    
    print(f"\n✅ Mixed population test PASSED!")
    
    return {
        'V_pore': V_pore,
        'V_liq_int': V_liq_int,
        'V_liq_ext': V_liq_ext,
    }


def main():
    """Run all liquid volume utility tests."""
    print("\n" + "#" * 70)
    print("# Liquid Volume Utilities Tests - Phase 2")
    print("#" * 70)
    
    results = []
    
    # Test 1: Basic liquid volume calculations
    try:
        test_liquid_volume_utilities()
        results.append(("Basic Utilities", True))
    except Exception as e:
        print(f"\n❌ Basic Utilities EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Basic Utilities", False))
    
    # Test 2: Vollkörper case
    try:
        test_vollkemer_case()
        results.append(("Vollkörper Case", True))
    except Exception as e:
        print(f"\n❌ Vollkörper Case EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Vollkörper Case", False))
    
    # Test 3: Mixed population
    try:
        test_mixed_population()
        results.append(("Mixed Population", True))
    except Exception as e:
        print(f"\n❌ Mixed Population EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Mixed Population", False))
    
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
    sys.exit(main())
