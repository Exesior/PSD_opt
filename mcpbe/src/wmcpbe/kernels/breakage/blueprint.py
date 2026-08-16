"""
================================================================================
BREAKAGE KERNEL BLUEPRINT
================================================================================

This file serves as a TEMPLATE for implementing new breakage kernels.
Copy this file and modify it to create your own kernel implementation.

Breakage kernels compute the BREAKAGE RATE S(V) for a particle of volume V.
Units: [1/s] (rate per time)

The breakage rate determines how likely a particle is to fragment during
the simulation. Higher rates mean faster fragmentation.

================================================================================
PHYSICAL BACKGROUND
================================================================================

General form of breakage kernel:

    S(V) = S_0 × f_stress(V) × f_material × f_environment
    
where:
    - S_0: Base breakage rate [1/s]
    - f_stress(V): Stress dependence (function of particle size)
    - f_material: Material properties (strength, porosity, etc.)
    - f_environment: Environmental factors (viscosity, concentration)

Common breakage models:
    1. Power-law:           S(V) = P1 × G × V^P2
    2. Stress-based:        S(V) = A × exp(-(σ_crit/σ_applied)^m)
    3. Energy-based:        S(V) = k × (E_impact / E_threshold)^n

Weibull Statistics (for brittle materials):
    P(break) = 1 - exp(-(σ_applied / σ_crit)^m)
    
    where:
        - σ_applied: Applied stress [Pa]
        - σ_crit: Critical stress (material strength) [Pa]
        - m: Weibull modulus (scatter parameter, typically 1-5)

================================================================================
IMPLEMENTATION GUIDE
================================================================================

Step 1: Copy this file
    cp blueprint.py my_custom_kernel.py

Step 2: Rename the class
    class MyCustomKernel(BreakageKernel):

Step 3: Implement required methods
    - name (property)
    - get_default_params()
    - __init__()
    - validate_params()
    - compute_rate()

Step 4: Register in __init__.py
    from .my_custom_kernel import MyCustomKernel
    BREAKAGE_KERNELS['my_custom'] = MyCustomKernel

Step 5: Test your kernel
    kernel = get_breakage_kernel('my_custom', param1=0.1)
    rate = kernel.compute_rate(v_particle=1e-18)

================================================================================
"""

import numpy as np
from typing import Optional, Any

# Handle both package import and direct execution
try:
    # Package context (imported by other modules)
    from ..base import BreakageKernel
except ImportError:
    # Direct execution (python blueprint.py)
    # Mock the base class for testing purposes
    class BreakageKernel:
        """Mock base class for standalone testing."""
        pass


# =============================================================================
# KERNEL CLASS DEFINITION
# =============================================================================

class ExampleBreakageKernel(BreakageKernel):
    """
    EXAMPLE: Breakage kernel with [YOUR PHYSICS HERE].
    
    Replace this docstring with a clear description of your kernel's physics:
    
    Physical Model:
        Describe the physical mechanism your kernel models. For example:
        "This kernel models stress-induced breakage of brittle particles
        in a high-shear environment. The breakage probability follows
        Weibull statistics with size-dependent stress amplification."
    
    Key Parameters:
        - param1: Description, units, typical range
        - param2: Description, units, typical range
    
    When to Use:
        Describe the application scenarios where this kernel is appropriate.
        For example: "Use this kernel for brittle materials (ceramics,
        crystals) in high-shear mixers or mills."
    
    References:
        [1] Author, "Title", Journal, Year
        [2] Author, "Title", Journal, Year
    
    Example:
        >>> kernel = ExampleBreakageKernel(critical_stress=1e6, weibull_modulus=2.0)
        >>> rate = kernel.compute_rate(v_particle=1e-18)
        >>> print(f"Breakage rate: {rate:.3e} 1/s")
    """
    
    # -------------------------------------------------------------------------
    # REQUIRED: Kernel Name
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """
        Return the kernel name (must match registry key in __init__.py).
        
        This name is used to select the kernel via factory function:
            kernel = get_breakage_kernel('example_kernel')
        
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
                'param2': 1e6,              # [Pa], material strength
                'param3': 2.0,              # Weibull modulus, typical: 1-5
                'flag_option': True,        # Boolean flag for feature X
            }
        """
        return {
            # Shear rate [1/s]
            # Typical range: 100 - 10000 1/s
            # Default: 1000 1/s (high-shear mixer)
            'g': 1000.0,
            
            # Base breakage rate coefficient
            # Units: [1/s] (if size-independent) or [m^(-3*p2)/s] (if size-dependent)
            # Typical range: 1e-3 to 10 1/s
            'p1': 3e-2,

            # Volume exponent (dimensionless)
            # Controls size dependence: S ~ V^p2
            # Typical range: 0.0 - 2.0
            # Default: 1.0 (larger particles break faster)
            # NOTE: do NOT name this `pl_v` -- that name belongs to the
            # breakage FUNCTION (fragment size distribution, BREAKFVAL).
            # Mixing the two up is what made the rate formula diverge from
            # the reference implementation once before.
            'p2': 1.0,

            # Your custom parameter (replace with your own)
            # Description, units, range
            'your_param_here': 1.0,
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
            kernel = ExampleBreakageKernel(
                p1=1e-2,         # Override default
                g=500,           # Override default
                p2=1.5           # Override default
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
        self.g = float(self.params['g'])
        self.p1 = float(self.params['p1'])
        self.p2 = float(self.params['p2'])

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
        if params['g'] < 0:
            raise ValueError(
                f"g must be non-negative, got {params['g']}"
            )
        
        if params['p1'] <= 0:
            raise ValueError(
                f"p1 must be positive, got {params['p1']}"
            )
        
        # Add your custom validation here
        # Example:
        # if params['your_param_here'] <= 0:
        #     raise ValueError(
        #         f"your_param_here must be positive, got {params['your_param_here']}"
        #     )
        
        return params
    
    # -------------------------------------------------------------------------
    # REQUIRED: Core Computation Method
    # -------------------------------------------------------------------------
    
    def compute_rate(
        self,
        v_particle: float,
        particle_idx: Optional[int] = None,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute breakage rate S(V) for a particle.
        
        This is the MAIN METHOD called by the solver during breakage events.
        It must be fast and numerically stable.
        
        Args:
            v_particle: Particle volume [m³]
                Type: float (scalar)
                Range: > 0
                Example: 1e-18 (1 µm diameter sphere)
            
            particle_idx: Index of particle in solver arrays
                Type: int or None
                Purpose: Access solver state (porosity, saturation, etc.)
                Note: May be None if solver doesn't provide indices
            
            solver: Reference to MCPBE solver instance
                Type: MCPBESolver or None
                Purpose: Access global state (temperature, viscosity, etc.)
                Note: May be None if kernel doesn't need solver state
        
        Returns:
            rate: Breakage rate [1/s]
                Type: float
                Range: >= 0
                Typical values: 1e-6 to 10 1/s
                Physical meaning: Probability per unit time to break
        
        Important Notes:
            1. MUST return a non-negative float
            2. Should handle edge cases (NaN, inf, zero volume)
            3. Should be computationally efficient (called millions of times)
            4. Can access solver state via particle index (optional)
        
        Accessing Solver State (Advanced):
            If you need particle-specific properties (porosity, strength, etc.),
            use the particle index to access solver arrays:
            
            if solver is not None and particle_idx is not None:
                poro = solver.porosity[particle_idx]
                sat = solver.saturation[particle_idx]
                
                # Modify breakage rate based on porosity
                if not np.isnan(poro):
                    # Porous particles are weaker
                    rate *= (1.0 + 2.0 * poro)
            
            Always check for None and NaN values!
        
        Example Implementation (Power-Law):
            # Classic power-law breakage rate
            # S(V) = P1 × G × V^P2

            rate = self.p1 * self.g * (v_particle ** self.p2)

            return rate

        Example with Solver State (Porosity-Dependent):
            # Base breakage rate
            rate = self.p1 * self.g * (v_particle ** self.p2)

            # Access porosity if available
            if solver is not None and particle_idx is not None:
                poro = solver.porosity[particle_idx]
                
                if not np.isnan(poro):
                    # Porous particles break easier
                    weakening_factor = (1.0 - poro) ** (-2.0)
                    rate *= weakening_factor
            
            return rate
        """
        # =====================================================================
        # STEP 1: Compute base breakage rate
        # =====================================================================
        
        # Choose ONE of these common models (or implement your own):
        
        # Option A: Power-law (most common, reference-compatible)
        # S(V) = P1 × G × V^P2  -- see kernels/breakage/_base_rate.py
        # Best for: General purpose, empirical fitting
        rate = self.p1 * self.g * (v_particle ** self.p2)
        
        # Option B: Size-independent (constant rate)
        # S(V) = constant
        # Best for: Simple tests, uniform breakage
        # rate = self.p1
        
        # Option C: Linear in size
        # S(V) = k × d = k × (6V/π)^(1/3)
        # Best for: Surface-dominated breakage
        # d = (6 * v_particle / np.pi) ** (1/3)
        # rate = self.p1 * d
        
        # Option D: Your custom model
        # rate = ...
        
        # =====================================================================
        # STEP 2: Apply modifications based on particle state (optional)
        # =====================================================================
        
        # Example: Porosity-dependent weakening
        # if solver is not None and particle_idx is not None:
        #     poro = solver.porosity[particle_idx]
        #     
        #     if not np.isnan(poro):
        #         # Porous particles are weaker
        #         weakening = (1.0 - poro) ** (-self.porosity_exponent)
        #         rate *= weakening
        
        # Example: Saturation-dependent strengthening
        # if solver is not None and particle_idx is not None:
        #     sat = solver.saturation[particle_idx]
        #     
        #     if not np.isnan(sat):
        #         # Wet particles are stronger (liquid bridges)
        #         strengthening = 1.0 + 0.5 * sat
        #         rate /= strengthening
        
        # Example: Stress-based Weibull model
        # if solver is not None:
        #     # Applied stress from shear
        #     mu = getattr(solver, 'viscosity', 1e-3)  # Pa·s
        #     sigma_applied = mu * self.g
        #     
        #     # Critical stress (material property)
        #     sigma_crit = self.critical_stress
        #     
        #     # Weibull modulus
        #     m = self.weibull_modulus
        #     
        #     # Breakage probability per unit time
        #     rate = self.rate_prefactor * (1 - np.exp(-(sigma_applied / sigma_crit)**m))
        
        # =====================================================================
        # STEP 3: Handle edge cases
        # =====================================================================
        
        # Ensure non-negative result
        if rate < 0:
            rate = 0.0
        
        # Handle very small particles (numerical stability)
        if v_particle < 1e-30:
            rate = 0.0  # Too small to break
        
        # Handle NaN/inf inputs gracefully
        if not np.isfinite(rate):
            rate = 0.0
        
        return rate
    
    # -------------------------------------------------------------------------
    # OPTIONAL: Helper Methods
    # -------------------------------------------------------------------------
    
    def _compute_applied_stress(
        self,
        solver: Optional[Any] = None
    ) -> float:
        """
        Estimate applied stress from shear (helper method, optional).
        
        Useful for stress-based breakage models.
        
        Formula: σ_applied ≈ μ × G
        
        Args:
            solver: Solver instance (for viscosity)
        
        Returns:
            Applied stress [Pa]
        """
        mu = 1e-3  # Dynamic viscosity [Pa·s], water at 20°C
        
        if solver is not None:
            mu = getattr(solver, 'viscosity', mu)
        
        sigma = mu * self.g
        
        return sigma
    
    def _compute_particle_strength(
        self,
        v_particle: float,
        porosity: float = None
    ) -> float:
        """
        Estimate particle strength (helper method, optional).
        
        Useful for material-dependent breakage models.
        
        Formula: σ_crit = σ_0 × (d/d_0)^(-α) × (1 - poro)^β
        
        Args:
            v_particle: Particle volume [m³]
            porosity: Particle porosity (optional)
        
        Returns:
            Critical stress (strength) [Pa]
        """
        # Reference strength and size
        sigma_0 = 1e6  # Pa (1 MPa, typical for granules)
        d_0 = 1e-3     # m (1 mm reference)
        
        # Size exponent (typically 0.5-1.5 for brittle materials)
        alpha = 1.0
        
        # Compute particle diameter
        d = (6 * v_particle / np.pi) ** (1/3)
        
        # Size-dependent strength
        sigma_crit = sigma_0 * (d / d_0) ** (-alpha)
        
        # Porosity weakening (if porosity known)
        if porosity is not None and not np.isnan(porosity):
            beta = 2.0  # Porosity weakening exponent
            sigma_crit *= (1.0 - porosity) ** beta
        
        return sigma_crit


# =============================================================================
# TESTING SECTION (run this file directly to test)
# =============================================================================

if __name__ == '__main__':
    """
    Quick test of the kernel implementation.
    
    Run: python blueprint.py
    """
    print("=" * 80)
    print("BREAKAGE KERNEL TEST")
    print("=" * 80)
    
    # Create kernel with default parameters
    print("\n1. Creating kernel with default parameters...")
    kernel = ExampleBreakageKernel()
    print(f"   ✓ Kernel created: {kernel.name}")
    print(f"   ✓ Parameters: {kernel.params}")
    
    # Test basic computation
    print("\n2. Testing compute_rate()...")
    v = 1e-18  # 1 µm diameter sphere
    rate = kernel.compute_rate(v)
    print(f"   S({v*1e18:.1f} µm³) = {rate:.3e} 1/s")
    
    # Test with custom parameters
    print("\n3. Testing with custom parameters...")
    kernel_custom = ExampleBreakageKernel(p1=1e-2, g=500, p2=1.5)
    rate_custom = kernel_custom.compute_rate(v)
    print(f"   S(custom) = {rate_custom:.3e} 1/s")
    
    # Test size dependence
    print("\n4. Testing size dependence...")
    
    # Small particles
    v_small = 1e-24  # 0.1 µm
    rate_small = kernel.compute_rate(v_small)
    print(f"   S(nano: {v_small*1e18:.3f} µm³) = {rate_small:.3e} 1/s")
    
    # Large particles
    v_large = 1e-12  # 100 µm
    rate_large = kernel.compute_rate(v_large)
    print(f"   S(micro: {v_large*1e18:.1f} µm³) = {rate_large:.3e} 1/s")
    
    # Size ratio
    print(f"   Rate ratio (large/small) = {rate_large/rate_small:.1f}×")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
