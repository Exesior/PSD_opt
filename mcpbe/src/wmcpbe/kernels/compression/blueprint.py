"""
================================================================================
COMPRESSION KERNEL BLUEPRINT
================================================================================

This file serves as a TEMPLATE for implementing new compression kernels.
Copy this file and modify it to create your own kernel implementation.

Compression kernels compute the POROSITY REDUCTION over TIME (continuous process).
Units: [1/s] (rate) or dimensionless (new porosity)

Unlike porosity_growth kernels (event-based), compression is TIME-BASED and
operates continuously during the simulation via operator splitting.

================================================================================
PHYSICAL BACKGROUND
================================================================================

Compression Mechanism:
    During high-shear processing, particles experience mechanical stress that
    reduces their porosity over time. This is modeled as:
    
    dp/dt = -f(p, σ, material_properties)
    
    where:
        - p: Current porosity
        - σ: Applied stress (from shear, collisions, etc.)
        - material_properties: Yield strength, compressibility, etc.

Common Compression Models:
    1. Exponential decay:     p(t) = p_min + (p₀ - p_min) × exp(-k×t)
    2. Stress-compaction:     dp/dt = -k × (σ_applied - σ_yield)
    3. Consolidation theory:  dp/dt = -C_c × log(σ/σ₀)

Key Distinction: Event-Based vs. Time-Based
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │ Aspect              │ Porosity Growth      │ Compression          │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Trigger             │ Discrete events      │ Continuous time      │
    │                     │ (agg/break/nuc)      │ (dt integration)     │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ When called         │ During MC events     │ Every time step      │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Physics             │ Pore creation/change │ Pore reduction       │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Typical effect      │ Porosity increases   │ Porosity decreases   │
    └─────────────────────┴──────────────────────┴──────────────────────┘

Both mechanisms can be active simultaneously!

================================================================================
IMPLEMENTATION GUIDE
================================================================================

Step 1: Copy this file
    cp blueprint.py my_custom_kernel.py

Step 2: Rename the class
    class MyCustomCompressionKernel(CompressionKernel):

Step 3: Implement required methods
    - name (property)
    - get_default_params()
    - __init__()
    - validate_params()
    - compute_porosity_decay()

Step 4: Register in __init__.py
    from .my_custom_kernel import MyCustomCompressionKernel
    COMPRESSION_KERNELS['my_custom'] = MyCustomCompressionKernel

Step 5: Test your kernel
    kernel = get_compression_kernel('my_custom', rate=0.01)
    poro_new = kernel.compute_porosity_decay(poro=0.5, dt=0.1, ...)

================================================================================
"""

import numpy as np
from typing import Optional, Any

# Handle both package import and direct execution
try:
    # Package context (imported by other modules)
    from ..base import CompressionKernel
except ImportError:
    # Direct execution (python blueprint.py)
    # Mock the base class for testing purposes
    class CompressionKernel:
        """Mock base class for standalone testing."""
        pass


# =============================================================================
# KERNEL CLASS DEFINITION
# =============================================================================

class ExampleCompressionKernel(CompressionKernel):
    """
    EXAMPLE: Compression kernel with [YOUR PHYSICS HERE].
    
    Replace this docstring with a clear description of your kernel's physics:
    
    Physical Model:
        Describe the physical mechanism your kernel models. For example:
        "This kernel models time-dependent porosity reduction due to
        viscoelastic creep under constant stress. The compression rate
        depends on both current porosity and applied shear stress."
    
    Key Parameters:
        - param1: Description, units, typical range
        - param2: Description, units, typical range
    
    When to Use:
        Describe the application scenarios where this kernel is appropriate.
        For example: "Use this kernel for processes with sustained mechanical
        loading, such as tablet compaction or roller pressing."
    
    References:
        [1] Author, "Title", Journal, Year
        [2] Author, "Title", Journal, Year
    
    Example:
        >>> kernel = ExampleCompressionKernel(rate=0.02, min_porosity=0.2)
        >>> poro_new = kernel.compute_porosity_decay(
        ...     porosity=0.5, dt=1.0, v_particle=1e-18
        ... )
        >>> print(f"New porosity: {poro_new:.3f}")
    """
    
    # -------------------------------------------------------------------------
    # REQUIRED: Kernel Name
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """
        Return the kernel name (must match registry key in __init__.py).
        
        This name is used to select the kernel via factory function:
            kernel = get_compression_kernel('example_kernel')
        
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
                'param1': 0.01,             # [1/s], compression rate
                'param2': 0.2,              # Dimensionless, min porosity
                'param3': 1e6,              # [Pa], yield stress
            }
        """
        return {
            # Compression rate coefficient
            # Units: [1/s]
            # Typical range: 0.001 - 0.1 1/s
            # Default: 0.02 1/s (moderate compression)
            'rate': 0.02,
            
            # Minimum achievable porosity
            # Dimensionless, range: [0, 1)
            # Default: 0.3 (30% residual porosity)
            # Physical meaning: Densest packing achievable
            'min_porosity': 0.3,
            
            # Your custom parameter (replace with your own)
            # Description, units, range
            # Default: value (justification)
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
            kernel = ExampleCompressionKernel(
                rate=0.05,           # Override default (faster compression)
                min_porosity=0.2     # Override default (denser packing)
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
        self.rate = float(self.params['rate'])
        self.min_porosity = float(self.params['min_porosity'])
        
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
            - Check physical bounds (e.g., porosity in [0, 1])
            - Ensure rate is positive
            - Provide clear error messages with expected ranges
        
        IMPORTANT: Use 'params' argument, NOT 'self.params'
                   (self.params doesn't exist yet - set AFTER validation!)
        
        Example validation rules:
            if params['rate'] <= 0:
                raise ValueError("rate must be positive")
            if not (0 <= params['min_porosity'] < 1):
                raise ValueError("min_porosity must be in [0, 1)")
        """
        # Validate standard parameters
        if params['rate'] <= 0:
            raise ValueError(
                f"rate must be positive, got {params['rate']}"
            )
        
        if not (0 <= params['min_porosity'] < 1):
            raise ValueError(
                f"min_porosity must be in [0, 1), got {params['min_porosity']}"
            )
        
        if params['min_porosity'] >= params.get('rate', 1):
            # Sanity check: min_porosity should be reasonable
            pass  # Add warning if needed
        
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
    
    def compute_porosity_decay(
        self,
        porosity: float,
        dt: float,
        v_particle: float = None,
        local_stress: float = None,
        saturation: float = None,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute new porosity after compression over time dt.
        
        This is the MAIN METHOD called by the solver during the compression
        step (operator splitting). It computes the porosity reduction due
        to mechanical stress over a time interval.
        
        Args:
            porosity: Current particle porosity
                Type: float or np.nan
                Range: [0, 1] or np.nan (Vollkörper)
                Note: np.nan means solid sphere (no pores to compress!)
            
            dt: Time step [s]
                Type: float
                Range: > 0
                Typical: 1e-6 to 1.0 s (depends on simulation)
            
            v_particle: Particle volume [m³] (optional)
                Type: float or None
                Purpose: For size-dependent compression models
            
            local_stress: Local mechanical stress [Pa] (optional)
                Type: float or None
                Purpose: For stress-dependent compression models
                Note: May be estimated from shear rate
            
            saturation: Particle saturation [0, 1] (optional)
                Type: float or None
                Purpose: For liquid-assisted compression models
                Note: Saturated particles may compress differently
            
            solver: Solver instance (optional)
                Type: MCPBESolver or None
                Purpose: Access global state (shear rate, viscosity, etc.)
        
        Returns:
            new_porosity: Porosity after compression
                Type: float or np.nan
                Range: [0, 1] or np.nan (unchanged if Vollkörper)
        
        Important Notes:
            1. MUST return a value in [0, 1] or np.nan
            2. Should handle edge cases (NaN, inf, zero dt)
            3. Should be computationally efficient (called every timestep)
            4. Compression can only REDUCE porosity (or stay same)
        
        Accessing Solver State (Advanced):
            If you need global properties (shear rate, viscosity, etc.),
            access them via the solver:
            
            if solver is not None:
                g = getattr(solver, 'G', 1000.0)  # Shear rate [1/s]
                mu = getattr(solver, 'viscosity', 1e-3)  # Viscosity [Pa·s]
                
                # Estimate stress
                stress = mu * g
            
            Always check for None before accessing!
        
        Example Implementation (Exponential Decay):
            # Guard: Vollkörper cannot be compressed
            if np.isnan(porosity):
                return np.nan
            
            # Clamp input porosity to valid range
            porosity = max(0.0, min(1.0, porosity))
            
            # Exponential decay toward min_porosity
            # p(t) = p_min + (p0 - p_min) × exp(-rate × t)
            delta_p = (porosity - self.min_porosity) * (1 - np.exp(-self.rate * dt))
            new_porosity = porosity - delta_p
            
            # Ensure we don't go below minimum
            new_porosity = max(new_porosity, self.min_porosity)
            
            return new_porosity
        
        Example with Stress Dependence:
            # Estimate applied stress if not provided
            if local_stress is None and solver is not None:
                mu = getattr(solver, 'viscosity', 1e-3)  # Pa·s
                g = getattr(solver, 'G', 1000.0)  # 1/s
                local_stress = mu * g
            
            # Only compress if stress exceeds yield
            if local_stress is not None and local_stress > self.yield_stress:
                excess_stress = local_stress - self.yield_stress
                rate_effective = self.rate * (excess_stress / self.yield_stress)
            else:
                rate_effective = 0.0  # No compression below yield
            
            # Exponential decay with effective rate
            if np.isnan(porosity):
                return np.nan
            
            delta_p = (porosity - self.min_porosity) * (1 - np.exp(-rate_effective * dt))
            new_porosity = max(porosity - delta_p, self.min_porosity)
            
            return new_porosity
        """
        # =====================================================================
        # STEP 1: Handle Vollkörper case
        # =====================================================================
        
        if np.isnan(porosity):
            # Solid spheres cannot be compressed (no pores)
            return np.nan
        
        # =====================================================================
        # STEP 2: Clamp input porosity to valid range
        # =====================================================================
        
        porosity = max(0.0, min(1.0, porosity))
        
        # Early exit: already at minimum
        if porosity <= self.min_porosity:
            return porosity
        
        # =====================================================================
        # STEP 3: Apply YOUR compression model
        # =====================================================================
        
        # Choose ONE of these common models (or implement your own):
        
        # Option A: Exponential decay (most common)
        # p(t) = p_min + (p0 - p_min) × exp(-rate × t)
        # Best for: Time-dependent consolidation, viscoelastic creep
        delta_p = (porosity - self.min_porosity) * (1 - np.exp(-self.rate * dt))
        new_porosity = porosity - delta_p
        
        # Option B: Linear decay (simple, less physical)
        # p(t) = p0 - rate × t × (p0 - p_min)
        # Best for: Small time steps, approximate models
        # delta_p = self.rate * dt * (porosity - self.min_porosity)
        # new_porosity = porosity - delta_p
        
        # Option C: Power-law consolidation
        # p(t) = p0 × (1 + rate × t)^(-n)
        # Best for: Soil mechanics, granular materials
        # n = 0.5  # Consolidation exponent
        # new_porosity = porosity * (1 + self.rate * dt) ** (-n)
        # new_porosity = max(new_porosity, self.min_porosity)
        
        # Option D: Stress-dependent (requires local_stress)
        # if local_stress is not None:
        #     yield_stress = getattr(self, 'yield_stress', 1e5)  # Pa
        #     
        #     if local_stress > yield_stress:
        #         excess = (local_stress - yield_stress) / yield_stress
        #         rate_eff = self.rate * (1 + excess)
        #         delta_p = (porosity - self.min_porosity) * (1 - np.exp(-rate_eff * dt))
        #         new_porosity = porosity - delta_p
        #     else:
        #         new_porosity = porosity  # No compression below yield
        # else:
        #     # Fallback to exponential
        #     delta_p = (porosity - self.min_porosity) * (1 - np.exp(-self.rate * dt))
        #     new_porosity = porosity - delta_p
        
        # Option E: Your custom model
        # new_porosity = ...
        
        # =====================================================================
        # STEP 4: Enforce minimum porosity
        # =====================================================================
        
        new_porosity = max(new_porosity, self.min_porosity)
        
        # =====================================================================
        # STEP 5: Ensure valid output
        # =====================================================================
        
        # Clamp to valid range
        new_porosity = max(0.0, min(1.0, new_porosity))
        
        # Handle NaN/inf gracefully
        if not np.isfinite(new_porosity):
            new_porosity = self.min_porosity
        
        return new_porosity
    
    # -------------------------------------------------------------------------
    # OPTIONAL: Helper Methods
    # -------------------------------------------------------------------------
    
    def _estimate_stress_from_shear(
        self,
        solver: Optional[Any] = None
    ) -> float:
        """
        Estimate applied stress from shear rate (helper method, optional).
        
        Useful for stress-dependent compression models.
        
        Formula: σ ≈ μ × G
        
        Args:
            solver: Solver instance (for viscosity and shear rate)
        
        Returns:
            Estimated stress [Pa]
        """
        # Default values
        mu = 1e-3   # Dynamic viscosity [Pa·s], water at 20°C
        g = 1000.0  # Shear rate [1/s]
        
        if solver is not None:
            mu = getattr(solver, 'viscosity', mu)
            g = getattr(solver, 'G', g)
        
        sigma = mu * g
        
        return sigma
    
    def _compute_compression_time_scale(
        self,
        target_porosity: float,
        initial_porosity: float = 0.5
    ) -> float:
        """
        Estimate time to reach target porosity (helper method, optional).
        
        Useful for planning simulations or setting rate parameters.
        
        From exponential decay model:
            t = -ln((p_target - p_min) / (p_initial - p_min)) / rate
        
        Args:
            target_porosity: Desired final porosity
            initial_porosity: Starting porosity (default: 0.5)
        
        Returns:
            Estimated time [s] (inf if impossible)
        """
        if target_porosity < self.min_porosity:
            return float('inf')  # Impossible
        
        if initial_porosity <= self.min_porosity:
            return 0.0  # Already there
        
        ratio = (target_porosity - self.min_porosity) / (initial_porosity - self.min_porosity)
        
        if ratio <= 0:
            return float('inf')
        
        t = -np.log(ratio) / self.rate
        
        return t


# =============================================================================
# TESTING SECTION (run this file directly to test)
# =============================================================================

if __name__ == '__main__':
    """
    Quick test of the kernel implementation.
    
    Run: python blueprint.py
    """
    print("=" * 80)
    print("COMPRESSION KERNEL TEST")
    print("=" * 80)
    
    # Create kernel with default parameters
    print("\n1. Creating kernel with default parameters...")
    kernel = ExampleCompressionKernel()
    print(f"   ✓ Kernel created: {kernel.name}")
    print(f"   ✓ Parameters: {kernel.params}")
    
    # Test basic computation
    print("\n2. Testing compute_porosity_decay()...")
    poro_initial = 0.5
    dt = 1.0  # 1 second
    poro_new = kernel.compute_porosity_decay(poro_initial, dt)
    print(f"   p({poro_initial:.2f}) after {dt}s → {poro_new:.3f}")
    
    # Test with custom parameters
    print("\n3. Testing with custom parameters...")
    kernel_fast = ExampleCompressionKernel(rate=0.1, min_porosity=0.15)
    poro_fast = kernel_fast.compute_porosity_decay(poro_initial, dt)
    print(f"   Fast compression: p({poro_initial:.2f}) → {poro_fast:.3f}")
    
    # Test time evolution
    print("\n4. Testing time evolution...")
    poro = 0.5
    times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"   {'Time [s]':<12} {'Porosity':<12}")
    print(f"   {'-'*12} {'-'*12}")
    
    for t in times:
        print(f"   {t:<12.2f} {poro:<12.4f}")
        if t < max(times):
            dt_step = 0.1  # Small steps
            poro = kernel.compute_porosity_decay(poro, dt_step)
    
    # Test edge cases
    print("\n5. Testing edge cases...")
    
    # Vollkörper
    poro_vk = kernel.compute_porosity_decay(np.nan, dt)
    print(f"   Vollkörper: {poro_vk} (np.nan expected)")
    
    # Already at minimum
    poro_min = kernel.compute_porosity_decay(kernel.min_porosity, dt)
    print(f"   At min ({kernel.min_porosity:.2f}): {poro_min:.3f}")
    
    # Very small dt
    poro_small = kernel.compute_porosity_decay(0.5, 1e-9)
    print(f"   Tiny dt (1 ns): Δp = {0.5 - poro_small:.6f}")
    
    # Large dt
    poro_large = kernel.compute_porosity_decay(0.5, 100.0)
    print(f"   Large dt (100 s): {poro_large:.4f} (should approach {kernel.min_porosity})")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
