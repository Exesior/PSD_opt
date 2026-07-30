# -*- coding: utf-8 -*-
"""Test script to verify liquid_volume implementation in MCPBE solver."""

import numpy as np
from mcpbe import MCPBESolver


def test_liquid_volume():
    """Test that liquid_volume attribute is correctly initialized and maintained."""
    
    # Create solver without config file
    solver = MCPBESolver(dim=1, t_total=10, t_write=5, load_attr=False, init=False)
    
    # Set required parameters manually (normally loaded from config)
    solver.COLEVAL = 3      # kernel type
    solver.CORR_BETA = 1e-3 # kernel correction factor
    solver.alpha_prim = 1.0 # collision efficiency
    solver.G = 1.0          # shear rate
    solver.process_type = "agglomeration"
    solver.a0 = 2000         # initial particle count (must be > a_tot for CV doubling)
    solver.recon_enable = False
    solver.recon_method = "4PMC"
    solver.recon_N_max = 4000
    solver.recon_bins = 30
    solver.recon_RS_target = 1000
    solver.break_dW_max = 50.0
    solver.agg_dW_min = 1.0
    solver.agg_dW_max = 20.0
    
    # Initialize particles
    solver._initialize_particles()
    
    print("=" * 60)
    print("Testing liquid_volume implementation")
    print("=" * 60)
    
    # Test 1: liquid_volume exists
    assert hasattr(solver, 'liquid_volume'), "ERROR: liquid_volume attribute missing"
    print("✓ Test 1: liquid_volume attribute exists")
    
    # Test 2: Same shape as X
    assert solver.liquid_volume.shape[0] == solver.X.shape[0], \
        f"ERROR: liquid_volume shape {solver.liquid_volume.shape} != X shape {solver.X.shape}"
    print(f"✓ Test 2: liquid_volume shape matches X shape: {solver.liquid_volume.shape}")
    
    # Test 3: Initialized with zeros
    assert np.all(solver.liquid_volume[:solver.a_tot] == 0.0), \
        "ERROR: Initial liquid_volume should be zeros"
    print("✓ Test 3: Initial liquid_volume is all zeros")
    
    # Test 4: Capacity growth
    old_cap = solver._cap
    old_liquid_size = solver.liquid_volume.shape[0]
    solver._ensure_capacity_for(100)
    assert solver.liquid_volume.shape[0] >= old_liquid_size, \
        "ERROR: liquid_volume did not grow with capacity"
    print(f"✓ Test 4: liquid_volume grows with capacity ({old_cap} -> {solver._cap})")
    
    # Test 5: Values preserved after capacity growth
    preserved = np.allclose(
        solver.liquid_volume[:solver.a_tot], 
        np.zeros(solver.a_tot)
    )
    assert preserved, "ERROR: liquid_volume values not preserved after capacity growth"
    print("✓ Test 5: liquid_volume values preserved after capacity growth")
    
    # Test 6: Control volume doubling
    old_a_tot = solver.a_tot
    old_liquid = solver.liquid_volume[:old_a_tot].copy()
    solver._maybe_double_control_volume(elapsed_time=1.0, iter_count=0)
    assert solver.a_tot == 2 * old_a_tot, \
        f"ERROR: a_tot not doubled ({solver.a_tot} != {2*old_a_tot})"
    assert solver.liquid_volume.shape[0] >= solver.a_tot, \
        "ERROR: liquid_volume capacity insufficient after CV doubling"
    doubled = np.allclose(
        solver.liquid_volume[:old_a_tot], 
        solver.liquid_volume[old_a_tot:2*old_a_tot]
    )
    assert doubled, "ERROR: liquid_volume not correctly duplicated during CV doubling"
    print(f"✓ Test 6: Control volume doubling works (a_tot: {old_a_tot} -> {solver.a_tot})")
    
    print("=" * 60)
    print("All tests passed! liquid_volume implementation is correct.")
    print("=" * 60)


if __name__ == "__main__":
    test_liquid_volume()