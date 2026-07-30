"""
================================================================================
AGGREGATION KERNEL BLUEPRINT
================================================================================

This file serves as a TEMPLATE for implementing new agglomeration kernels.
Copy this file and modify it to create your own kernel implementation.

Aggregation kernels compute the COLLISION FREQUENCY β(i,j) between two particles.
Units: [m³/s] (volume per time)

The collision frequency determines how likely two particles are to collide
and potentially merge during the simulation.

================================================================================
PHYSICAL BACKGROUND
================================================================================

General form of aggregation kernel:

    β(i,j) = β_hydrodynamic × f_liquid × f_capillary × f_other
    
where:
    - β_hydrodynamic: Base collision frequency from fluid mechanics
                      (shear, Brownian motion, differential settling)
    - f_liquid: Enhancement/reduction due to liquid content
    - f_capillary: Capillary forces from liquid bridges
    - f_other: Additional physics (e.g., electrostatic, van der Waals)

Common hydrodynamic models:
    1. Shear-induced (Smoluchowski): β = (4/3) × G × (r_i + r_j)³
    2. Brownian diffusion:           β = 2kT(r_i + r_j)² / (3μ r_i r_j)
    3. Differential settling:        β = π(r_i + r_j)² × |v_i - v_j|

================================================================================
IMPLEMENTATION GUIDE
================================================================================

Step 1: Copy this file
    cp blueprint.py my_custom_kernel.py

Step 2: Rename the class
    class MyCustomKernel(AggregationKernel):

Step 3: Implement required methods
    - name (property)
    - get_default_params()
    - __init__()
    - validate_params()
    - compute_beta()

Step 4: Register in __init__.py
    from .my_custom_kernel import MyCustomKernel
    AGGREGATION_KERNELS['my_custom'] = MyCustomKernel

Step 5: Test your kernel
    kernel = get_aggregation_kernel('my_custom', param1=0.1)
    beta = kernel.compute_beta(r1=1e-6, r2=2e-6)

================================================================================
"""

import numpy as np
from typing import Optional, Any

# Handle both package import and direct execution
try:
    # Package context (imported by other modules)
    from ..base import AggregationKernel
except ImportError:
    # Direct execution (python blueprint.py)
    # Mock the base class for testing purposes
    class AggregationKernel:
        """Mock base class for standalone testing."""
        pass


# =============================================================================
# KERNEL CLASS DEFINITION
# =============================================================================

class ExampleAggregationKernel(AggregationKernel):
    """
    EXAMPLE: Aggregation kernel with [YOUR PHYSICS HERE].
    
    Replace this docstring with a clear description of your kernel's physics:
    
    Physical Model:
        Describe the physical mechanism your kernel models. For example:
        "This kernel models shear-induced agglomeration with liquid bridge
        enhancement. The collision frequency is increased when particles
        have optimal liquid saturation for bridge formation."
    
    Key Parameters:
        - param1: Description, units, typical range
        - param2: Description, units, typical range
    
    When to Use:
        Describe the application scenarios where this kernel is appropriate.
        For example: "Use this kernel for wet granulation processes with
        moderate shear rates (100-1000 1/s) and liquid saturations 0.3-0.7."
    
    References:
        [1] Author, "Title", Journal, Year
        [2] Author, "Title", Journal, Year
    
    Example:
        >>> kernel = ExampleAggregationKernel(param1=0.1, param2=1000)
        >>> beta = kernel.compute_beta(r1=1e-6, r2=2e-6)
        >>> print(f"Collision frequency: {beta:.3e} m³/s")
    """
    
    # -------------------------------------------------------------------------
    # REQUIRED: Kernel Name
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """
        Return the kernel name (must match registry key in __init__.py).
        
        This name is used to select the kernel via factory function:
            kernel = get_aggregation_kernel('example_kernel')
        
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
                'param1': 0.1,              # Dimensionless, range: [0, 1]
                'param2': 1000.0,           # [1/s], typical: 100-10000
                'param3': 0.072,            # [N/m], surface tension of water
                'flag_option': True,        # Boolean flag for feature X
            }
        """
        return {
            # Correction factor for collision efficiency
            # Dimensionless, typical range: 1e-5 to 1.0
            # Default: 1e-3 (moderate efficiency)
            'corr_beta': 1e-3,
            
            # Shear rate [1/s]
            # Typical range: 100 - 10000 1/s
            # Default: 1000 1/s (high-shear mixer)
            'g': 1000.0,
            
            # Example additional parameter (replace with your own)
            # Description, units, range
            'your_param_here': 0.5,
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
            kernel = ExampleAggregationKernel(
                corr_beta=1e-4,    # Override default
                g=500,             # Override default
                your_param_here=0.8  # Override default
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
        # This avoids repeated dictionary lookups during compute_beta()
        self.corr_beta = float(self.params['corr_beta'])
        self.g = float(self.params['g'])
        
        # Cache your custom parameters
        # self.your_param = float(self.params['your_param_here'])
    
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
            - Warn about suspicious values (optional)
        
        Example validation rules:
            if params['param1'] <= 0:
                raise ValueError("param1 must be positive")
            if not (0 <= params['param2'] <= 1):
                raise ValueError("param2 must be in [0, 1]")
        """
        # IMPORTANT: Use 'params' argument, NOT 'self.params'
        # (self.params doesn't exist yet - it's set AFTER validation!)
        
        # Validate standard parameters
        if params['corr_beta'] <= 0:
            raise ValueError(
                f"corr_beta must be positive, got {params['corr_beta']}"
            )
        
        if params['g'] < 0:
            raise ValueError(
                f"g must be non-negative, got {params['g']}"
            )
        
        # Add your custom validation here
        # Example:
        # if not (0 <= params['your_param_here'] <= 1):
        #     raise ValueError(
        #         f"your_param_here must be in [0, 1], "
        #         f"got {params['your_param_here']}"
        #     )
        
        return params
    
    # -------------------------------------------------------------------------
    # REQUIRED: Core Computation Method
    # -------------------------------------------------------------------------
    
    def compute_beta(
        self,
        r1: float,
        r2: float,
        particle1_idx: Optional[int] = None,
        particle2_idx: Optional[int] = None,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute collision frequency β between two particles.
        
        This is the MAIN METHOD called by the solver during agglomeration events.
        It must be fast and numerically stable.
        
        Args:
            r1: Radius of particle 1 [m]
                Type: float (scalar)
                Range: > 0
                Example: 1e-6 (1 µm particle)
            
            r2: Radius of particle 2 [m]
                Type: float (scalar)
                Range: > 0
                Example: 2e-6 (2 µm particle)
            
            particle1_idx: Index of particle 1 in solver arrays
                Type: int or None
                Purpose: Access solver state (porosity, saturation, etc.)
                Note: May be None if solver doesn't provide indices
            
            particle2_idx: Index of particle 2 in solver arrays
                Type: int or None
                Purpose: Access solver state (porosity, saturation, etc.)
                Note: May be None if solver doesn't provide indices
            
            solver: Reference to MCPBE solver instance
                Type: MCPBESolver or None
                Purpose: Access global state (temperature, viscosity, etc.)
                Note: May be None if kernel doesn't need solver state
        
        Returns:
            beta: Collision frequency [m³/s]
                Type: float
                Range: >= 0
                Typical values: 1e-15 to 1e-9 m³/s
                Physical meaning: Volume swept per unit time
        
        Important Notes:
            1. MUST return a non-negative float
            2. Should handle edge cases (NaN, inf, zero radii)
            3. Should be computationally efficient (called millions of times)
            4. Can access solver state via particle indices (optional)
        
        Accessing Solver State (Advanced):
            If you need particle-specific properties (porosity, saturation, etc.),
            use the particle indices to access solver arrays:
            
            if solver is not None and particle1_idx is not None:
                poro1 = solver.porosity[particle1_idx]
                sat1 = solver.saturation[particle1_idx]
                liq1 = solver.liquid_volume[particle1_idx]
            
            Always check for None and NaN values!
        
        Example Implementation (Shear Kernel):
            # Classic Smoluchowski shear kernel
            beta_shear = (4.0 / 3.0) * self.g * (r1 + r2)**3
            
            # Apply correction factor
            beta = self.corr_beta * beta_shear
            
            return beta
        
        Example with Solver State (Liquid Bridge):
            # Base shear collision frequency
            beta_shear = (4.0 / 3.0) * self.g * (r1 + r2)**3
            
            # Access saturation if available
            if solver is not None and particle1_idx is not None:
                sat1 = solver.saturation[particle1_idx]
                sat2 = solver.saturation[particle2_idx]
                
                # Liquid bridge enhancement
                if not np.isnan(sat1) and not np.isnan(sat2):
                    s_eff = (sat1 + sat2) / 2.0
                    bridge_factor = np.exp(-((s_eff - 0.5)**2) / 0.02)
                    beta *= bridge_factor
            
            return beta
        """
        # =====================================================================
        # STEP 1: Compute base collision frequency (hydrodynamic part)
        # =====================================================================
        
        # Choose ONE of these common models (or implement your own):
        
        # Option A: Shear-induced (Smoluchowski)
        # Best for: High-shear mixers, stirred tanks
        beta_base = (4.0 / 3.0) * self.g * (r1 + r2)**3
        
        # Option B: Brownian diffusion
        # Best for: Sub-micron particles, quiescent fluids
        # k_B = 1.38e-23  # Boltzmann constant [J/K]
        # T = 298.0       # Temperature [K]
        # mu = 1e-3       # Dynamic viscosity [Pa·s]
        # beta_base = 2 * k_B * T * (r1 + r2)**2 / (3 * mu * r1 * r2)
        
        # Option C: Differential settling
        # Best for: Large particles with different densities
        # rho_p = 2500.0  # Particle density [kg/m³]
        # rho_f = 1000.0  # Fluid density [kg/m³]
        # mu = 1e-3       # Viscosity [Pa·s]
        # g_grav = 9.81   # Gravity [m/s²]
        # v1 = 2/9 * (rho_p - rho_f) * g_grav * r1**2 / mu
        # v2 = 2/9 * (rho_p - rho_f) * g_grav * r2**2 / mu
        # beta_base = np.pi * (r1 + r2)**2 * abs(v1 - v2)
        
        # Option D: Your custom model
        # beta_base = ...
        
        # =====================================================================
        # STEP 2: Apply corrections/enhancements (optional)
        # =====================================================================
        
        # Start with base value
        beta = self.corr_beta * beta_base
        
        # Example: Add liquid-dependent enhancement
        # if solver is not None and particle1_idx is not None:
        #     sat1 = solver.saturation[particle1_idx]
        #     sat2 = solver.saturation[particle2_idx]
        #     
        #     if not np.isnan(sat1) and not np.isnan(sat2):
        #         # Your enhancement logic here
        #         enhancement_factor = 1.0 + 0.5 * (sat1 + sat2)
        #         beta *= enhancement_factor
        
        # Example: Add capillary force effects
        # if solver is not None:
        #     gamma = getattr(solver, 'surface_tension', 0.072)  # N/m
        #     theta = getattr(solver, 'contact_angle', 0.0)      # rad
        #     
        #     # Capillary enhancement factor
        #     f_cap = 1.0 + 2 * gamma * np.cos(theta) / (rho * g * r1 * r2)
        #     beta *= f_cap
        
        # =====================================================================
        # STEP 3: Handle edge cases
        # =====================================================================
        
        # Ensure non-negative result
        if beta < 0:
            beta = 0.0
        
        # Handle NaN/inf inputs gracefully
        if not np.isfinite(beta):
            beta = 0.0
        
        return beta
    
    # -------------------------------------------------------------------------
    # OPTIONAL: Helper Methods
    # -------------------------------------------------------------------------
    
    def _compute_collision_energy(
        self,
        r1: float,
        r2: float,
        solver: Optional[Any] = None
    ) -> float:
        """
        Estimate collision energy (helper method, optional).
        
        Useful for energy-dependent kernels (e.g., breakage, coalescence efficiency).
        
        Formula: E_coll ≈ 0.5 × m × v²
        
        Args:
            r1, r2: Particle radii [m]
            solver: Solver instance (for shear rate G)
        
        Returns:
            Collision energy [J]
        """
        # Effective radius
        r_eff = (r1 + r2) / 2.0
        
        # Assume particle density
        rho = 1000.0  # kg/m³
        
        # Particle mass
        m = (4.0 / 3.0) * np.pi * r_eff**3 * rho
        
        # Relative velocity (from shear)
        g = self.g if solver is None else getattr(solver, 'G', 1000.0)
        v_rel = g * (2 * r_eff)
        
        # Collision energy
        e_coll = 0.5 * m * v_rel**2
        
        return e_coll
    
    def _compute_stokes_number(
        self,
        r: float,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute Stokes number for particle (helper method, optional).
        
        Stokes number indicates particle inertia relative to fluid drag.
        
        Formula: St = (ρ_p × d² × G) / (18 × μ)
        
        Args:
            r: Particle radius [m]
            solver: Solver instance (for viscosity)
        
        Returns:
            Stokes number (dimensionless)
        """
        rho_p = 2500.0  # Particle density [kg/m³]
        mu = 1e-3       # Fluid viscosity [Pa·s]
        g = self.g if solver is None else getattr(solver, 'G', 1000.0)
        
        st = (rho_p * (2*r)**2 * g) / (18 * mu)
        
        return st


# =============================================================================
# TESTING SECTION (run this file directly to test)
# =============================================================================

if __name__ == '__main__':
    """
    Quick test of the kernel implementation.
    
    Run: python blueprint.py
    """
    print("=" * 80)
    print("AGGREGATION KERNEL TEST")
    print("=" * 80)
    
    # Create kernel with default parameters
    print("\n1. Creating kernel with default parameters...")
    kernel = ExampleAggregationKernel()
    print(f"   ✓ Kernel created: {kernel.name}")
    print(f"   ✓ Parameters: {kernel.params}")
    
    # Test basic computation
    print("\n2. Testing compute_beta()...")
    r1 = 1e-6   # 1 µm
    r2 = 2e-6   # 2 µm
    beta = kernel.compute_beta(r1, r2)
    print(f"   β({r1*1e6:.1f} µm, {r2*1e6:.1f} µm) = {beta:.3e} m³/s")
    
    # Test with custom parameters
    print("\n3. Testing with custom parameters...")
    kernel_custom = ExampleAggregationKernel(corr_beta=1e-4, g=500)
    beta_custom = kernel_custom.compute_beta(r1, r2)
    print(f"   β(custom) = {beta_custom:.3e} m³/s")
    
    # Test edge cases
    print("\n4. Testing edge cases...")
    
    # Very small particles
    beta_small = kernel.compute_beta(1e-9, 1e-9)
    print(f"   β(nano particles) = {beta_small:.3e} m³/s")
    
    # Very large particles
    beta_large = kernel.compute_beta(1e-3, 1e-3)
    print(f"   β(milli particles) = {beta_large:.3e} m³/s")
    
    # Size ratio
    beta_ratio = kernel.compute_beta(1e-6, 100e-6)
    print(f"   β(size ratio 1:100) = {beta_ratio:.3e} m³/s")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
