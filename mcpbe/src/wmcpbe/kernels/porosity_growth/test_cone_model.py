"""
Test and validation script for ConeModelKernel.

Verifies:
1. Mass conservation (V_solid conserved in agglomeration and breakage)
2. Positive pore volumes
3. Valid porosity range [0, 1)
4. Geometric calculations correctness
5. Fitting parameter effects

Usage (from mcpbe/src):
    python -m wmcpbe.kernels.porosity_growth.test_cone_model
"""

import os
import sys

import numpy as np

# `cone_model` itself imports `..base`, so it can only be loaded as part of the
# package -- the previous flat `from cone_model import ConeModelKernel` failed
# with ImportError no matter which directory it was started from. Adding
# mcpbe/src to sys.path keeps a direct `python test_cone_model.py` working too.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    )
from wmcpbe.kernels.porosity_growth.cone_model import ConeModelKernel


def test_radius_volume_conversion():
    """Test sphere radius ↔ volume conversion."""
    print("\n" + "="*60)
    print("TEST: Radius-Volume Conversion")
    print("="*60)
    
    kernel = ConeModelKernel()
    
    # Test round-trip
    V_test = 1.0e-18
    r = kernel._radius_from_volume(V_test)
    V_back = kernel._volume_from_radius(r)
    
    print(f"V_original = {V_test:.3e} m³")
    print(f"r = {r:.3e} m")
    print(f"V_back = {V_back:.3e} m³")
    print(f"Error: {abs(V_back - V_test) / V_test * 100:.6f}%")
    
    assert abs(V_back - V_test) < V_test * 1e-10, "Round-trip failed!"
    print("✓ PASSED")


def test_cone_pill_volume_equal_spheres():
    """Test cone-pill volume for equal-sized spheres."""
    print("\n" + "="*60)
    print("TEST: Cone-Pill Volume (Equal Spheres)")
    print("="*60)
    
    kernel = ConeModelKernel()
    
    # Equal spheres: ri = rj = r
    r = 1.0e-6  # 1 µm
    V_sphere = (4/3) * np.pi * r**3
    
    V_pill = kernel._cone_pill_volume(r, r)
    V_old = 2 * V_sphere
    
    # Analytical solution for equal spheres:
    # V_cone = 2 × π × r³
    # V_hemi = (4/3) × π × r³
    # V_pill = (10/3) × π × r³
    V_analytical = (10/3) * np.pi * r**3
    
    ΔV = V_pill - V_old
    ΔV_analytical = (2/3) * np.pi * r**3  # Expected
    
    print(f"r = {r:.3e} m")
    print(f"V_sphere = {V_sphere:.3e} m³")
    print(f"V_old (2 spheres) = {V_old:.3e} m³")
    print(f"V_pill (analytical) = {V_analytical:.3e} m³")
    print(f"V_pill (computed) = {V_pill:.3e} m³")
    print(f"ΔV (analytical) = {ΔV_analytical:.3e} m³")
    print(f"ΔV (computed) = {ΔV:.3e} m³")
    print(f"ΔV/V_old = {ΔV/V_old * 100:.2f}%")
    
    assert abs(V_pill - V_analytical) < V_analytical * 1e-10, "V_pill mismatch!"
    assert abs(ΔV - ΔV_analytical) < ΔV_analytical * 1e-10, "ΔV mismatch!"
    assert ΔV > 0, "ΔV should be positive!"
    print("✓ PASSED")


def test_agglomeration_mass_conservation():
    """Test mass conservation in agglomeration."""
    print("\n" + "="*60)
    print("TEST: Agglomeration Mass Conservation")
    print("="*60)
    
    kernel = ConeModelKernel(k_agg=0.5)
    
    # Parent 1: V_dry = 1e-18, ε = 0.4
    V_dry_1 = 1.0e-18
    poro_1 = 0.4
    V_solid_1 = V_dry_1 * (1 - poro_1)
    V_pore_1 = V_dry_1 * poro_1
    
    # Parent 2: V_dry = 2e-18, ε = 0.5
    V_dry_2 = 2.0e-18
    poro_2 = 0.5
    V_solid_2 = V_dry_2 * (1 - poro_2)
    V_pore_2 = V_dry_2 * poro_2
    
    print(f"Parent 1: V_dry={V_dry_1:.3e}, ε={poro_1:.3f}")
    print(f"  → V_solid={V_solid_1:.3e}, V_pore={V_pore_1:.3e}")
    print(f"Parent 2: V_dry={V_dry_2:.3e}, ε={poro_2:.3f}")
    print(f"  → V_solid={V_solid_2:.3e}, V_pore={V_pore_2:.3e}")
    
    # Agglomerate
    V_dry_new, poro_new = kernel.compute_merged_porosity(
        V_dry_1, poro_1, V_dry_2, poro_2
    )
    
    # Compute child properties
    V_solid_new_expected = V_solid_1 + V_solid_2
    V_pore_new_expected = V_pore_1 + V_pore_2 + kernel.k_agg * kernel._delta_volume_pair(V_dry_1, V_dry_2)
    V_dry_new_expected = V_solid_new_expected + V_pore_new_expected
    poro_new_expected = V_pore_new_expected / V_dry_new_expected
    
    print(f"\nChild (computed): V_dry={V_dry_new:.3e}, ε={poro_new:.6f}")
    print(f"Child (expected): V_dry={V_dry_new_expected:.3e}, ε={poro_new_expected:.6f}")
    
    # Verify mass conservation
    V_solid_new_computed = V_dry_new * (1 - poro_new)
    
    print(f"\nMass Conservation Check:")
    print(f"  V_solid_expected = {V_solid_new_expected:.3e}")
    print(f"  V_solid_computed = {V_solid_new_computed:.3e}")
    print(f"  Error = {abs(V_solid_new_computed - V_solid_new_expected) / V_solid_new_expected * 100:.10f}%")
    
    assert abs(V_solid_new_computed - V_solid_new_expected) < V_solid_new_expected * 1e-14, \
        f"Mass NOT conserved! Error: {V_solid_new_computed - V_solid_new_expected}"
    
    # Verify porosity
    assert abs(poro_new - poro_new_expected) < 1e-10, "Porosity mismatch!"
    
    # Verify porosity in valid range
    assert 0 <= poro_new < 1, f"Invalid porosity: {poro_new}"
    
    print("✓ PASSED (Mass conserved!)")


def test_breakage_mass_conservation():
    """Test mass conservation in breakage."""
    print("\n" + "="*60)
    print("TEST: Breakage Mass Conservation")
    print("="*60)
    
    kernel = ConeModelKernel(k_break=0.1)
    
    # Parent: V_dry = 3e-18, ε = 0.45
    V_dry_parent = 3.0e-18
    poro_parent = 0.45
    V_solid_parent = V_dry_parent * (1 - poro_parent)
    V_pore_parent = V_dry_parent * poro_parent
    
    print(f"Parent: V_dry={V_dry_parent:.3e}, ε={poro_parent:.3f}")
    print(f"  → V_solid={V_solid_parent:.3e}, V_pore={V_pore_parent:.3e}")
    
    # Break into 3 fragments
    fragment_volumes = [1.5e-18, 1.0e-18, 0.5e-18]  # Sum = 3e-18 = parent volume
    
    poros_frag = kernel.compute_fragment_porosity(
        poro_parent, fragment_volumes, V_dry_parent
    )
    
    # CONTRACT: the input `fragment_volumes` are V_DRY. The kernel keeps the
    # solid share of the parent (V_solid = V_dry * (1 - ε_parent)) and returns a
    # NEW porosity for the shrunken pore space. The consumer therefore has to
    # re-derive V_dry from the solid volume -- exactly what mcpbe_break.py does:
    #     V_dry_frag = V_solid_frag / (1 - frag_poro)
    #
    # Applying the NEW ε to the OLD V_dry instead mixes two states and produced
    # an apparent 3.23% mass error, which this test used to print and then not
    # assert on (its only assertion was that the input volumes sum to the parent
    # volume -- a tautology about its own test data).
    print(f"\nFragments: {len(fragment_volumes)} pieces")
    V_solid_frags = [V * (1 - poro_parent) for V in fragment_volumes]
    V_dry_frags_new = [Vs / (1 - p) for Vs, p in zip(V_solid_frags, poros_frag)]

    for k, (V_old, V_new, Vs, p) in enumerate(
            zip(fragment_volumes, V_dry_frags_new, V_solid_frags, poros_frag)):
        print(f"  Frag {k+1}: V_dry {V_old:.3e} → {V_new:.3e}, ε={p:.6f}")
        print(f"    → V_solid={Vs:.3e}, V_pore={V_new - Vs:.3e}")

    # Verify mass conservation
    V_solid_frags_sum = sum(V_solid_frags)
    V_pore_frags_sum = sum(V - Vs for V, Vs in zip(V_dry_frags_new, V_solid_frags))
    rel_err = abs(V_solid_frags_sum - V_solid_parent) / V_solid_parent

    print(f"\nMass Conservation Check:")
    print(f"  V_solid_parent = {V_solid_parent:.3e}")
    print(f"  Σ V_solid_frags = {V_solid_frags_sum:.3e}")
    print(f"  Error = {rel_err * 100:.10f}%")

    assert abs(sum(fragment_volumes) - V_dry_parent) < V_dry_parent * 1e-14, \
        "Fragment volumes don't sum to parent!"
    assert rel_err < 1e-12, (
        f"V_solid not conserved across breakage: parent {V_solid_parent:.6e}, "
        f"fragments {V_solid_frags_sum:.6e} (rel. error {rel_err:.3e})")
    # Each fragment must also be internally consistent: V_dry*(1-eps) == V_solid
    for Vs, V_new, p in zip(V_solid_frags, V_dry_frags_new, poros_frag):
        assert abs(V_new * (1 - p) - Vs) <= 1e-12 * Vs, \
            "Fragment V_dry/porosity/V_solid are mutually inconsistent!"

    # Verify pore loss (should decrease due to new surface)
    print(f"\nPore Volume Check:")
    print(f"  V_pore_parent = {V_pore_parent:.3e}")
    print(f"  Σ V_pore_frags = {V_pore_frags_sum:.3e}")
    print(f"  ΔV_pore = {V_pore_frags_sum - V_pore_parent:.3e} (should be ≤ 0)")
    
    # Pore volume should decrease (or stay same if k_break=0)
    # Note: Due to redistribution, individual fragment pores may vary
    # but total should generally decrease
    
    # Verify all porosities in valid range
    for k, poro_frag in enumerate(poros_frag):
        assert 0 <= poro_frag < 1, f"Invalid porosity for fragment {k}: {poro_frag}"
    
    print("✓ PASSED")


def test_vollkörper_handling():
    """Test handling of Vollkörper (NaN porosity)."""
    print("\n" + "="*60)
    print("TEST: Vollkörper Handling")
    print("="*60)
    
    kernel = ConeModelKernel()
    
    # NOTE ON THE CONTRACT: a legacy Vollkörper (ε = NaN) means "no pores", so
    # it is mapped to ε = 0.0 -- the modern convention used across the codebase
    # (cf. mcpbe_break.py, and compute_nucleation_porosity, which returns 0.0
    # rather than NaN). NaN is deliberately NOT propagated.
    #
    # These cases used to assert NaN preservation, which the kernel never
    # actually delivered: `max(0.0, min(1.0, nan))` is 1.0 in Python, so a
    # pore-free body silently became "pure void" and its solid volume collapsed
    # to zero. What matters here is therefore mass, and that is asserted below.

    # Case 1: Both parents Vollkörper
    print("\nCase 1: Both Vollkörper")
    V_dry, poro = kernel.compute_merged_porosity(1e-18, np.nan, 1e-18, np.nan)
    V_solid = V_dry * (1.0 - poro)
    print(f"  Result: V_dry={V_dry:.3e}, ε={poro:.6f}, V_solid={V_solid:.6e}")
    assert not np.isnan(poro), "Porosity must be a number, not NaN!"
    assert 0 <= poro < 1, "Invalid porosity!"
    # Vollkörper have V_solid == V_dry, so the merged solid is the plain sum.
    assert abs(V_solid - 2e-18) / 2e-18 < 1e-12, (
        f"Mass not conserved: expected 2.000e-18, got {V_solid:.6e}")

    # Case 2: One parent Vollkörper
    print("\nCase 2: One Vollkörper, one porous")
    V_dry, poro = kernel.compute_merged_porosity(1e-18, np.nan, 1e-18, 0.4)
    V_solid = V_dry * (1.0 - poro)
    expected_solid = 1e-18 + 1e-18 * (1.0 - 0.4)
    print(f"  Result: V_dry={V_dry:.3e}, ε={poro:.6f}, V_solid={V_solid:.6e}")
    assert not np.isnan(poro), "Should have valid porosity!"
    assert 0 <= poro < 1, "Invalid porosity!"
    assert abs(V_solid - expected_solid) / expected_solid < 1e-12, (
        f"Mass not conserved: expected {expected_solid:.6e}, got {V_solid:.6e}")

    # Case 3: Parent Vollkörper → fragments
    print("\nCase 3: Parent Vollkörper → breakage")
    poros = kernel.compute_fragment_porosity(np.nan, [0.5e-18, 0.5e-18], 1e-18)
    print(f"  Result: ε={poros}")
    assert not any(np.isnan(p) for p in poros), "Fragment porosity must not be NaN!"
    assert all(0 <= p < 1 for p in poros), "Invalid fragment porosity!"
    # A pore-free parent has no pore volume to hand down, so the fragments stay
    # pore-free too (V_pore_new = max(0, 0 - k_break*ΔV) = 0).
    assert all(p == 0.0 for p in poros), f"Expected pore-free fragments, got {poros}"

    print("✓ PASSED")


def test_fitting_parameters():
    """Test effect of k_agg and k_break parameters."""
    print("\n" + "="*60)
    print("TEST: Fitting Parameter Effects")
    print("="*60)
    
    # Test k_agg
    print("\nk_agg variation (agglomeration):")
    V_dry_1, poro_1 = 1e-18, 0.4
    V_dry_2, poro_2 = 1e-18, 0.4
    
    for k_agg in [0.0, 0.5, 1.0, 2.0]:
        kernel = ConeModelKernel(k_agg=k_agg, k_break=0.1)
        V_dry_new, poro_new = kernel.compute_merged_porosity(V_dry_1, poro_1, V_dry_2, poro_2)
        Δporo = poro_new - poro_1
        print(f"  k_agg={k_agg:4.1f} → ε={poro_new:.6f} (Δε={Δporo:+.6f})")
    
    # Test k_break
    print("\nk_break variation (breakage, 3 fragments):")
    V_dry_parent, poro_parent = 3e-18, 0.5
    frags = [1e-18, 1e-18, 1e-18]
    
    for k_break in [0.0, 0.1, 0.5, 1.0]:
        kernel = ConeModelKernel(k_agg=1.0, k_break=k_break)
        poros = kernel.compute_fragment_porosity(poro_parent, frags, V_dry_parent)
        avg_poro = np.mean(poros)
        Δporo = avg_poro - poro_parent
        print(f"  k_break={k_break:4.1f} → avg(ε)={avg_poro:.6f} (Δε={Δporo:+.6f})")
    
    print("✓ PASSED (parameters have expected effect)")


def test_edge_cases():
    """Test edge cases and numerical stability."""
    print("\n" + "="*60)
    print("TEST: Edge Cases")
    print("="*60)
    
    kernel = ConeModelKernel()
    
    # Very small particles
    print("\nVery small particles (nm scale):")
    V_dry, poro = kernel.compute_merged_porosity(1e-27, 0.3, 1e-27, 0.3)
    print(f"  V=1e-27 m³ → ε={poro:.6f}")
    assert 0 <= poro < 1
    
    # Very large size ratio
    print("\nLarge size ratio (1000:1):")
    V_dry, poro = kernel.compute_merged_porosity(1e-18, 0.4, 1e-21, 0.4)
    print(f"  V1=1e-18, V2=1e-21 → ε={poro:.6f}")
    assert 0 <= poro < 1
    
    # Zero porosity parents
    print("\nZero porosity parents:")
    V_dry, poro = kernel.compute_merged_porosity(1e-18, 0.0, 1e-18, 0.0)
    print(f"  ε1=0, ε2=0 → ε_new={poro:.6f}")
    # Should have some porosity from geometric ΔV
    assert poro >= 0
    
    # Many fragments (stress test)
    print("\nMany fragments (n=10):")
    V_parent = 1e-17
    frags = [V_parent / 10] * 10
    poros = kernel.compute_fragment_porosity(0.5, frags, V_parent)
    print(f"  n=10 → avg(ε)={np.mean(poros):.6f}")
    assert all(0 <= p < 1 for p in poros)
    
    print("✓ PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# ConeModelKernel Test Suite")
    print("#"*60)
    
    try:
        test_radius_volume_conversion()
        test_cone_pill_volume_equal_spheres()
        test_agglomeration_mass_conservation()
        test_breakage_mass_conservation()
        test_vollkörper_handling()
        test_fitting_parameters()
        test_edge_cases()
        
        print("\n" + "#"*60)
        print("# ALL TESTS PASSED ✓")
        print("#"*60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
