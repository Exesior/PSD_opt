# ================================================================================
# AGGLOMERATION ACCEPTANCE KERNEL BLUEPRINT
# ================================================================================
# 
# This is a TEMPLATE for implementing new agglomeration acceptance kernels.
# Copy this file and modify it to create your own kernel.
#
# Purpose:
#   Agglomeration acceptance kernels determine if a collision between two
#   particles results in agglomeration or if they bounce off. They are called
#   AFTER partner selection, BEFORE agglomeration execution.
#
# Physical Models:
#   - Stokes criterion (viscous dissipation vs. inertia)
#   - Liquid bridge rupture
#   - Viscoelastic rebound
#   - Surface roughness effects
#   - Custom phenomenological models
#
# Implementation Steps:
#   1. Copy this file to a new name (e.g., my_custom_kernel.py)
#   2. Rename the class (e.g., MyCustomKernel)
#   3. Update the 'name' property to match your filename
#   4. Define your parameters in get_default_params()
#   5. Implement accept_collision() with your physics
#   6. Add your kernel to __init__.py registry
#
# Testing:
#   Run directly: python blueprint.py (syntax check only)
#   Run in solver: Use agg_acceptance_kernel_name='my_custom_kernel'
#
# ================================================================================

from typing import Any, Dict, Optional
import numpy as np

# Import base class (with fallback for direct execution)
try:
    from ..base import AggAcceptanceKernel
except ImportError:
    # Mock for direct execution
    class AggAcceptanceKernel:
        def __init__(self, **params):
            self.params = dict(params)
        @property
        def name(self): return 'example'
        @property
        def category(self): return 'agglomeration_acceptance'


class ExampleAggAcceptanceKernel(AggAcceptanceKernel):
    """
    Example Agglomeration Acceptance Kernel.
    
    REPLACE THIS DOCSTRING with your kernel's description:
    
    Physical Model:
        Describe the physics behind your acceptance criterion.
        What determines if particles stick or bounce?
        
    Parameters:
        param1: Description, units, typical range
        param2: Description, units, typical range
        
    Usage:
        >>> kernel = ExampleAggAcceptanceKernel(param1=0.1, param2=1000.0)
        >>> accepted = kernel.accept_collision(r1, r2, v_dry1, v_dry2, solver=solver)
    
    References:
        [1] Author et al., "Title", Journal, Year.
    """
    
    # --- REQUIRED: Kernel Name ---
    @property
    def name(self) -> str:
        """Return kernel name (must match registry key in __init__.py)."""
        return 'example_kernel'
    
    # --- REQUIRED: Default Parameters ---
    def get_default_params(self) -> Dict[str, Any]:
        """
        Define default parameters with units and typical ranges.
        
        All parameters should have sensible defaults for testing.
        """
        return {
            'param1': 1.0,      # [unit] Description, typical range: [min, max]
            'param2': 0.5,      # [dimensionless] Description, range: [0, 1]
            # Add your parameters here
        }
    
    # --- REQUIRED: Constructor ---
    def __init__(self, **params):
        """
        Initialize kernel with parameters.
        
        Args:
            **params: Keyword arguments overriding default parameters
        """
        defaults = self.get_default_params()
        defaults.update(params)
        self.params = self.validate_params(defaults)
        
        # Cache frequently used values for performance
        self.param1 = float(self.params['param1'])
        self.param2 = float(self.params['param2'])
    
    # --- REQUIRED: Parameter Validation ---
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Validated parameter dictionary
            
        Raises:
            ValueError: If parameters are invalid
        """
        if params['param1'] <= 0:
            raise ValueError(f"param1 must be positive, got {params['param1']}")
        if not (0.0 <= params['param2'] <= 1.0):
            raise ValueError(f"param2 must be in [0, 1], got {params['param2']}")
        return params
    
    # --- REQUIRED: Core Acceptance Method ---
    def accept_collision(self,
                         r1: float, r2: float,
                         v_dry1: float, v_dry2: float,
                         particle1_idx: Optional[int] = None,
                         particle2_idx: Optional[int] = None,
                         solver: Optional[Any] = None
                         ) -> bool:
        """
        Determine if collision results in agglomeration.
        
        This is the MAIN METHOD that implements your acceptance criterion.
        
        Args:
            r1: Radius of particle 1 [m]
            r2: Radius of particle 2 [m]
            v_dry1: Dry volume of particle 1 [m³]
            v_dry2: Dry volume of particle 2 [m³]
            particle1_idx: Index of particle 1 in solver arrays
            particle2_idx: Index of particle 2 in solver arrays
            solver: Reference to solver for accessing state arrays
        
        Returns:
            accepted: True if agglomeration proceeds, False if rejected
        
        Note:
            - Solver may be None (handle gracefully!)
            - Indices may be None (handle gracefully!)
            - Access solver state like: solver.liquid_volume[particle_idx]
            - Use solver._rng for reproducible random numbers
        
        Example Implementation (Stokes-like):
            # STEP 1: Get additional properties from solver
            if solver is not None and particle1_idx is not None:
                liq1 = solver.liquid_volume[particle1_idx]
                liq2 = solver.liquid_volume[particle2_idx]
            else:
                return False  # Can't compute without data
            
            # STEP 2: Compute your criterion
            criterion_value = self.param1 * (r1 + r2) / (liq1 + liq2 + 1e-30)
            
            # STEP 3: Compare against threshold
            threshold = self.param2
            return criterion_value < threshold
        """
        # TODO: IMPLEMENT YOUR PHYSICS HERE!
        
        # Example: Always accept (replace with your logic)
        return True
        
        # Example: Reject if no liquid available
        # if solver is None:
        #     return False
        # liq1 = solver.liquid_volume.get(particle1_idx, 0.0)
        # liq2 = solver.liquid_volume.get(particle2_idx, 0.0)
        # if liq1 + liq2 <= 0:
        #     return False
        # return True
        
        # Example: Probabilistic acceptance
        # rng = solver._rng if solver is not None else np.random.default_rng()
        # return rng.random() < self.param2


# ================================================================================
# TESTING SECTION
# ================================================================================

if __name__ == '__main__':
    """
    Quick test of the kernel implementation.
    
    This runs basic sanity checks. For full testing, use in actual solver.
    """
    print("Testing AggAcceptanceKernel implementation...")
    
    # Test 1: Creation with defaults
    print("\n1. Creating kernel with default parameters...")
    kernel = ExampleAggAcceptanceKernel()
    print(f"   ✓ Created: {kernel}")
    print(f"   ✓ Parameters: {kernel.params}")
    
    # Test 2: Creation with custom parameters
    print("\n2. Creating kernel with custom parameters...")
    kernel = ExampleAggAcceptanceKernel(param1=0.5, param2=0.8)
    print(f"   ✓ Created: {kernel}")
    
    # Test 3: Parameter validation (should raise error)
    print("\n3. Testing parameter validation...")
    try:
        kernel = ExampleAggAcceptanceKernel(param1=-1.0)
        print("   ✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    
    # Test 4: Acceptance test (mock solver)
    print("\n4. Testing accept_collision()...")
    
    class MockSolver:
        _rng = np.random.default_rng(42)
        liquid_volume = {0: 1e-15, 1: 2e-15}
    
    mock_solver = MockSolver()
    
    accepted = kernel.accept_collision(
        r1=1e-6,
        r2=2e-6,
        v_dry1=1e-18,
        v_dry2=8e-18,
        particle1_idx=0,
        particle2_idx=1,
        solver=mock_solver
    )
    print(f"   ✓ accept_collision() returned: {accepted}")
    
    # Test 5: Edge cases
    print("\n5. Testing edge cases...")
    
    # No solver
    accepted_no_solver = kernel.accept_collision(
        r1=1e-6, r2=2e-6,
        v_dry1=1e-18, v_dry2=8e-18,
        solver=None
    )
    print(f"   ✓ No solver: {accepted_no_solver}")
    
    # No indices
    accepted_no_indices = kernel.accept_collision(
        r1=1e-6, r2=2e-6,
        v_dry1=1e-18, v_dry2=8e-18,
        particle1_idx=None,
        particle2_idx=None,
        solver=mock_solver
    )
    print(f"   ✓ No indices: {accepted_no_indices}")
    
    print("\n" + "="*70)
    print("✓ All tests completed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Replace ExampleAggAcceptanceKernel with your implementation")
    print("  2. Update 'name' property to match your filename")
    print("  3. Add to __init__.py registry:")
    print("     from .your_file import YourKernel")
    print("     AGG_ACCEPTANCE_KERNELS['your_kernel'] = YourKernel")
    print("  4. Test in actual MCPBE solver")
