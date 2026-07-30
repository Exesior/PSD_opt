"""
================================================================================
LIQUID DISTRIBUTION KERNEL BLUEPRINT
================================================================================

This file serves as a TEMPLATE for implementing new liquid distribution kernels.
Copy this file and modify it to create your own kernel implementation.

Liquid distribution kernels SELECT TARGET PARTICLES for liquid droplet addition
during nucleation events. They determine WHICH particle receives the next droplet.

Unlike other kernels (which compute rates/frequencies), this kernel performs
WEIGHTED RANDOM SAMPLING from the particle population.

================================================================================
PHYSICAL BACKGROUND
================================================================================

Nucleation Process:
    During wet granulation, liquid is sprayed into the system as droplets.
    Each droplet must be assigned to a target particle based on physical
    criteria:
    
    1. Collision probability: Larger particles have higher collision cross-section
    2. Surface availability: Unsaturated particles can accept more liquid
    3. Wettability: Material properties affect droplet capture

Selection Probability Models:
    1. Uniform weighted:     P(i) ∝ W[i]
       - All particles equally likely per unit weight
       - Default, simplest model
    
    2. Surface weighted:     P(i) ∝ r[i]² × W[i]
       - Larger surface area → higher collision probability
       - Physical for spray processes
    
    3. Saturation preferential: P(i) ∝ (1 - S[i])^bias × W[i]
       - Unsaturated particles preferred
       - Models capillary-driven uptake

Key Distinction: Selection vs. Addition
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │ Step                │ Liquid Distribution  │ Liquid Addition      │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ What                │ SELECT target        │ ADD liquid to target │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Output              │ Particle index       │ Updated saturation   │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Physics             │ Collision probability│ Pore filling         │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Called by           │ NucleationHandler    │ NucleationHandler    │
    └─────────────────────┴──────────────────────┴──────────────────────┘

================================================================================
IMPLEMENTATION GUIDE
================================================================================

Step 1: Copy this file
    cp blueprint.py my_custom_kernel.py

Step 2: Rename the class
    class MyCustomLiquidDistributionKernel(LiquidDistributionKernel):

Step 3: Implement required methods
    - name (property)
    - get_default_params()
    - __init__()
    - validate_params()
    - select_target_particle()

Step 4: Register in __init__.py
    from .my_custom_kernel import MyCustomLiquidDistributionKernel
    LIQUID_DISTRIBUTION_KERNELS['my_custom'] = MyCustomLiquidDistributionKernel

Step 5: Test your kernel
    kernel = get_liquid_distribution_kernel('my_custom', param1=0.1)
    idx = kernel.select_target_particle(solver, v_droplet, current_time)

================================================================================
"""

import numpy as np
from typing import Optional, Any

# Handle both package import and direct execution
try:
    # Package context (imported by other modules)
    from ..base import LiquidDistributionKernel
except ImportError:
    # Direct execution (python blueprint.py)
    # Mock the base class for testing purposes
    class LiquidDistributionKernel:
        """Mock base class for standalone testing."""
        pass


# =============================================================================
# KERNEL CLASS DEFINITION
# =============================================================================

class ExampleLiquidDistributionKernel(LiquidDistributionKernel):
    """
    EXAMPLE: Liquid distribution kernel with [YOUR PHYSICS HERE].
    
    Replace this docstring with a clear description of your kernel's physics:
    
    Physical Model:
        Describe the physical mechanism your kernel models. For example:
        "This kernel selects target particles based on their available
        surface area and saturation state. Particles with large surface
        area and low saturation are preferred, modeling both collision
        probability and capillary-driven uptake."
    
    Key Parameters:
        - param1: Description, units, typical range
        - param2: Description, units, typical range
    
    When to Use:
        Describe the application scenarios where this kernel is appropriate.
        For example: "Use this kernel for spray granulation processes where
        droplet-particle collisions dominate over random mixing."
    
    References:
        [1] Author, "Title", Journal, Year
        [2] Author, "Title", Journal, Year
    
    Example:
        >>> kernel = ExampleLiquidDistributionKernel(saturation_bias=2.0)
        >>> idx = kernel.select_target_particle(solver, v_droplet=1e-12, t=1.0)
        >>> print(f"Selected particle {idx}")
    """
    
    # -------------------------------------------------------------------------
    # REQUIRED: Kernel Name
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """
        Return the kernel name (must match registry key in __init__.py).
        
        This name is used to select the kernel via factory function:
            kernel = get_liquid_distribution_kernel('example_kernel')
        
        Naming convention: lowercase_with_underscores
        """
        return 'example_kernel'
    
    # -------------------------------------------------------------------------
    # REQUIRED: Default Parameters
    # -------------------------------------------------------------------------
    
    def get_default_params(self) -> dict:
        """
        Define default parameters for this kernel.
        
        Returns:
            Dictionary with parameter names and default values.
            
        Guidelines:
            - Use physically meaningful defaults
            - Include ALL parameters needed by the kernel
            - Document units and typical ranges in comments
        
        Example structure:
            return {
                'param1': 1.0,              # Dimensionless, bias factor
                'param2': 0.5,              # Threshold saturation
                'flag_option': True,        # Boolean flag for feature X
            }
        """
        return {
            # Your custom parameter 1
            # Description, units, range
            # Default: value (justification)
            'your_param_1': 1.0,
            
            # Your custom parameter 2
            # Description, units, range
            # Default: value (justification)
            'your_param_2': 0.5,
        }
    
    # -------------------------------------------------------------------------
    # REQUIRED: Constructor
    # -------------------------------------------------------------------------
    
    def __init__(self, **params):
        """
        Initialize the kernel with parameters.
        
        Args:
            **params: Keyword arguments overriding default parameters.
                     Only specify parameters you want to change.
        
        Example:
            kernel = ExampleLiquidDistributionKernel(
                your_param_1=2.0,    # Override default
                your_param_2=0.3     # Override default
            )
        
        Implementation pattern:
            1. Get default parameters
            2. Update with user-provided parameters
            3. Validate all parameters
            4. Cache frequently used values as instance attributes
        """
        # Step 1: Get default parameters
        defaults = self.get_default_params()
        
        # Step 2: Override with user-provided parameters
        defaults.update(params)
        
        # Step 3: Validate parameters (raises ValueError if invalid)
        self.params = self.validate_params(defaults)
        
        # Step 4: Cache frequently used values for performance
        self.your_param_1 = float(self.params['your_param_1'])
        self.your_param_2 = float(self.params['your_param_2'])
    
    # -------------------------------------------------------------------------
    # REQUIRED: Parameter Validation
    # -------------------------------------------------------------------------
    
    def validate_params(self, params: dict) -> dict:
        """
        Validate and sanitize input parameters.
        
        Args:
            params: Dictionary of parameters to validate.
        
        Returns:
            Validated parameter dictionary (may be modified).
        
        Raises:
            ValueError: If any parameter is out of valid range or invalid.
        
        Best practices:
            - Check physical bounds (e.g., positive values, probabilities in [0,1])
            - Provide clear error messages with expected ranges
            - Convert types explicitly (e.g., float(), bool())
        
        IMPORTANT: Use 'params' argument, NOT 'self.params'
                   (self.params doesn't exist yet - set AFTER validation!)
        
        Example validation rules:
            if params['param1'] <= 0:
                raise ValueError("param1 must be positive")
            if not (0 <= params['param2'] <= 1):
                raise ValueError("param2 must be in [0, 1]")
        """
        # Validate custom parameters
        if params['your_param_1'] < 0:
            raise ValueError(
                f"your_param_1 must be non-negative, got {params['your_param_1']}"
            )
        
        if not (0 <= params['your_param_2'] <= 1):
            raise ValueError(
                f"your_param_2 must be in [0, 1], got {params['your_param_2']}"
            )
        
        return params
    
    # -------------------------------------------------------------------------
    # REQUIRED: Core Computation Method
    # -------------------------------------------------------------------------
    
    def select_target_particle(
        self,
        solver: Any,
        v_droplet: float,
        current_time: float,
        rng: Optional[np.random.Generator] = None
    ) -> int:
        """
        Select a target particle for liquid droplet addition.
        
        This is the MAIN METHOD called by the NucleationHandler during
        nucleation events. It performs WEIGHTED RANDOM SAMPLING to select
        which particle receives the next droplet.
        
        Args:
            solver: MCPBE solver instance (REQUIRED)
                Type: MCPBESolver
                Purpose: Access particle state (W, V, porosity, saturation, etc.)
                Note: This kernel ALWAYS needs solver access!
            
            v_droplet: Volume of droplet to add [m³]
                Type: float
                Range: > 0
                Purpose: May affect selection (large droplets prefer larger pores)
            
            current_time: Current simulation time [s]
                Type: float
                Range: >= 0
                Purpose: For time-dependent selection models
            
            rng: Random number generator (optional)
                Type: numpy.random.Generator or None
                Purpose: For reproducible random sampling
                Note: If None, use solver's RNG
        
        Returns:
            target_idx: Index of selected particle
                Type: int
                Range: [0, solver.a_tot)
                Meaning: This particle will receive the droplet
        
        Important Notes:
            1. MUST return a valid index (0 <= idx < solver.a_tot)
            2. Should handle edge cases (empty population, all saturated)
            3. Should be computationally efficient (called every droplet event)
            4. Must use WEIGHTED sampling (not uniform!)
        
        Weighted Sampling Pattern:
            The general approach is:
            
            1. Compute weight w[i] for each particle i
            2. Normalize weights: p[i] = w[i] / sum(w)
            3. Sample from categorical distribution
            
            Example:
                # Compute weights
                weights = np.ones(solver.a_tot)  # Start with uniform
                
                # Apply your physics model
                # weights *= some_function_of_particle_properties
                
                # Normalize
                probs = weights / np.sum(weights)
                
                # Sample
                target_idx = rng.choice(solver.a_tot, p=probs)
                
                return target_idx
        
        Accessing Solver State:
            Common solver arrays you may need:
            
            - solver.W[:solver.a_tot]: Computational weights
            - solver.V_flat[-1, :solver.a_tot]: Dry volumes (V_dry)
            - solver.porosity[:solver.a_tot]: Porosities
            - solver.saturation[:solver.a_tot]: Saturations (if tracked)
            - solver.liquid_volume[:solver.a_tot]: Liquid volumes (if tracked)
            - solver.X[:solver.a_tot]: Particle sizes (radii)
            
            Always slice to solver.a_tot (active particles only)!
        
        Example Implementation (Uniform Weighted):
            # Get active particle count
            n_particles = solver.a_tot
            
            if n_particles == 0:
                raise ValueError("No particles available for nucleation")
            
            # Weights: just use computational weight W
            weights = solver.W[:n_particles].copy()
            
            # Handle zero weights
            weights = np.maximum(weights, 1e-30)
            
            # Normalize to probabilities
            probs = weights / np.sum(weights)
            
            # Sample
            if rng is None:
                rng = solver._rng
            
            target_idx = rng.choice(n_particles, p=probs)
            
            return target_idx
        
        Example (Surface Weighted):
            n_particles = solver.a_tot
            
            # Base weight: computational weight
            weights = solver.W[:n_particles].copy()
            
            # Size-dependent enhancement: larger particles have more surface
            # Surface area ~ r^2 ~ V^(2/3)
            v_dry = solver.V_flat[-1, :n_particles]
            surface_factor = v_dry ** (2.0 / 3.0)
            
            weights *= surface_factor
            
            # Normalize and sample
            weights = np.maximum(weights, 1e-30)
            probs = weights / np.sum(weights)
            
            if rng is None:
                rng = solver._rng
            
            target_idx = rng.choice(n_particles, p=probs)
            
            return target_idx
        
        Example (Saturation Preferential):
            n_particles = solver.a_tot
            
            # Base weight
            weights = solver.W[:n_particles].copy()
            
            # Check if saturation is tracked
            if hasattr(solver, 'saturation'):
                sat = solver.saturation[:n_particles]
                
                # Prefer unsaturated particles
                # Availability = 1 - saturation
                availability = 1.0 - np.nan_to_num(sat, nan=0.0)
                
                # Bias toward low saturation (higher power = stronger preference)
                bias = self.saturation_bias  # e.g., 2.0
                weights *= availability ** bias
            
            # Normalize and sample
            weights = np.maximum(weights, 1e-30)
            probs = weights / np.sum(weights)
            
            if rng is None:
                rng = solver._rng
            
            target_idx = rng.choice(n_particles, p=probs)
            
            return target_idx
        """
        # =====================================================================
        # STEP 1: Get active particle count
        # =====================================================================
        
        n_particles = solver.a_tot
        
        if n_particles == 0:
            raise ValueError("No particles available for nucleation")
        
        # =====================================================================
        # STEP 2: Compute selection weights
        # =====================================================================
        
        # Start with base weight: computational weight W
        weights = solver.W[:n_particles].copy()
        
        # TODO: Apply YOUR physics model here!
        # Examples:
        
        # Example A: Size-dependent (surface-weighted)
        # v_dry = solver.V_flat[-1, :n_particles]
        # surface_area = v_dry ** (2.0 / 3.0)
        # weights *= surface_area
        
        # Example B: Saturation-preferential
        # if hasattr(solver, 'saturation'):
        #     sat = solver.saturation[:n_particles]
        #     availability = 1.0 - np.nan_to_num(sat, nan=0.0)
        #     weights *= availability ** self.your_param_1
        
        # Example C: Pore-size dependent (large droplets prefer large pores)
        # if hasattr(solver, 'porosity'):
        #     poro = solver.porosity[:n_particles]
        #     v_pore = solver.V_flat[-1, :n_particles] * np.nan_to_num(poro, nan=0.0)
        #     
        #     # Large droplets need large pores
        #     pore_size_factor = v_pore / v_droplet
        #     weights *= np.clip(pore_size_factor, 0.1, 10.0)
        
        # Example D: Time-dependent (early vs. late stage)
        # if current_time < self.your_param_2:
        #     # Early stage: prefer small particles
        #     v_dry = solver.V_flat[-1, :n_particles]
        #     weights /= v_dry ** self.your_param_1
        # else:
        #     # Late stage: prefer large particles
        #     v_dry = solver.V_flat[-1, :n_particles]
        #     weights *= v_dry ** self.your_param_1
        
        # =====================================================================
        # STEP 3: Handle edge cases
        # =====================================================================
        
        # Ensure all weights are positive (avoid division by zero)
        min_weight = 1e-30
        weights = np.maximum(weights, min_weight)
        
        # Handle NaN weights
        weights = np.nan_to_num(weights, nan=min_weight)
        
        # =====================================================================
        # STEP 4: Normalize to probabilities
        # =====================================================================
        
        total_weight = np.sum(weights)
        
        if total_weight <= 0:
            # Fallback: uniform sampling
            probs = np.ones(n_particles) / n_particles
        else:
            probs = weights / total_weight
        
        # =====================================================================
        # STEP 5: Sample from categorical distribution
        # =====================================================================
        
        if rng is None:
            rng = solver._rng
        
        target_idx = rng.choice(n_particles, p=probs)
        
        return target_idx
    
    # -------------------------------------------------------------------------
    # OPTIONAL: Helper Methods
    # -------------------------------------------------------------------------
    
    def _compute_surface_area(
        self,
        v_dry: np.ndarray,
        shape_factor: float = 1.0
    ) -> np.ndarray:
        """
        Compute estimated surface area from dry volume (helper method).
        
        Assumes spherical particles: A = π × d² = π × (6V/π)^(2/3)
        
        Args:
            v_dry: Dry volumes [m³] (array)
            shape_factor: Correction for non-spherical particles (default: 1.0)
        
        Returns:
            Surface areas [m²] (array)
        """
        # Sphere: A = π^(1/3) × (6V)^(2/3)
        area = (np.pi ** (1/3)) * (6 * v_dry) ** (2/3)
        
        # Shape correction
        area *= shape_factor
        
        return area
    
    def _compute_pore_capacity(
        self,
        v_dry: np.ndarray,
        porosity: np.ndarray
    ) -> np.ndarray:
        """
        Compute pore volume (capacity for internal liquid) (helper method).
        
        Args:
            v_dry: Dry volumes [m³] (array)
            porosity: Porosities (array, may contain NaN)
        
        Returns:
            Pore volumes [m³] (array)
        """
        # Handle Vollkörper (NaN porosity)
        poro_valid = np.nan_to_num(porosity, nan=0.0)
        
        v_pore = v_dry * poro_valid
        
        return v_pore
    
    def _compute_available_capacity(
        self,
        v_pore: np.ndarray,
        saturation: np.ndarray
    ) -> np.ndarray:
        """
        Compute remaining pore capacity (helper method).
        
        Args:
            v_pore: Total pore volumes [m³] (array)
            saturation: Current saturations [0, 1] (array)
        
        Returns:
            Available capacity [m³] (array)
        """
        # Handle NaN saturation
        sat_valid = np.nan_to_num(saturation, nan=0.0)
        
        # Available = total × (1 - saturation)
        available = v_pore * (1.0 - sat_valid)
        
        return available


# =============================================================================
# TESTING SECTION (run this file directly to test)
# =============================================================================

if __name__ == '__main__':
    """
    Quick test of the kernel implementation.
    
    Note: Full testing requires a solver instance.
    This test demonstrates the API only.
    
    Run: python blueprint.py
    """
    print("=" * 80)
    print("LIQUID DISTRIBUTION KERNEL TEST")
    print("=" * 80)
    
    # Create kernel with default parameters
    print("\n1. Creating kernel with default parameters...")
    kernel = ExampleLiquidDistributionKernel()
    print(f"   ✓ Kernel created: {kernel.name}")
    print(f"   ✓ Parameters: {kernel.params}")
    
    # Demonstrate API (without actual solver)
    print("\n2. Demonstrating API...")
    print("   Note: Full testing requires MCPBESolver instance")
    print("   Typical usage:")
    print("   ```")
    print("   solver = MCPBESolver(...)")
    print("   kernel = get_liquid_distribution_kernel('example_kernel')")
    print("   ")
    print("   # During nucleation:")
    print("   for _ in range(n_droplets):")
    print("       target_idx = kernel.select_target_particle(")
    print("           solver=solver,")
    print("           v_droplet=1e-12,")
    print("           current_time=solver.t_current,")
    print("           rng=solver._rng")
    print("       )")
    print("       # Add liquid to solver.liquid_volume[target_idx]")
    print("   ```")
    
    # Test helper methods
    print("\n3. Testing helper methods...")
    
    # Surface area
    v_test = np.array([1e-18, 8e-18, 27e-18])  # 1, 2, 3 µm spheres
    areas = kernel._compute_surface_area(v_test)
    print(f"   Surface areas: {areas*1e12:.2f} µm²")
    
    # Pore capacity
    poro_test = np.array([0.3, 0.5, np.nan])
    v_pore = kernel._compute_pore_capacity(v_test, poro_test)
    print(f"   Pore volumes: {v_pore*1e18:.2f} µm³")
    
    # Available capacity
    sat_test = np.array([0.2, 0.8, 0.0])
    available = kernel._compute_available_capacity(v_pore, sat_test)
    print(f"   Available capacity: {available*1e18:.2f} µm³")
    
    print("\n" + "=" * 80)
    print("✓ API demonstration complete!")
    print("=" * 80)
