"""
================================================================================
POROSITY GROWTH KERNEL BLUEPRINT
================================================================================

This file serves as a TEMPLATE for implementing new porosity growth kernels.
Copy this file and modify it to create your own kernel implementation.

Porosity growth kernels compute the POROSITY evolution during DISCRETE EVENTS:
1. Agglomeration (two particles merge)
2. Breakage (particle fragments)
3. Nucleation (new particle forms)

Unlike compression kernels (time-based), these are EVENT-BASED.

================================================================================
PHYSICAL BACKGROUND
================================================================================

Porosity Definition:
    porosity = V_pore / V_dry
    
    where:
        - V_dry = V_solid + V_pore (total dry volume)
        - V_solid = V_dry × (1 - porosity) (solid material volume)
        - V_pore = V_dry × porosity (void volume)

Key Distinction: Vollkörper vs. Porous
    - Vollkörper (solid sphere): porosity = np.nan
    - Porous particle: porosity ∈ [0, 1]
    
    IMPORTANT: Always check for NaN before calculations!
        if np.isnan(porosity):
            # Vollkörper: V_solid = V_dry
        else:
            # Porous: V_solid = V_dry × (1 - porosity)

V_flat Structure (dim=1):
    V_flat[0, :] = V_solid  (solid volume, conserved during compression!)
    V_flat[1, :] = V_dry    (total volume = V_solid + V_pore)
    
    CRITICAL: Mass conservation requires V_flat[0,:] = V_solid (NOT V_dry!)

Three Methods, One Physics:
    The SAME physical model must be applied consistently across all three
    operations. For example:
    
    VolumeMixing Model:
        - Agg: Volumes add linearly (V_pore_new = V_pore₁ + V_pore₂)
        - Break: Fragments inherit parent porosity (equal splitting)
        - Nuc: Default porosity needed to create pores from Vollkörper
    
    IncompleteMixing Model:
        - Agg: Additional trapped pores from incomplete coalescence
        - Break: Pore collapse from mechanical stress
        - Nuc: Default porosity (configurable)
    
    The key insight: If agglomeration CREATES pores, breakage should
    DESTROY them (and vice versa). This ensures physical consistency.

================================================================================
IMPLEMENTATION GUIDE
================================================================================

Step 1: Copy this file
    cp blueprint.py my_custom_kernel.py

Step 2: Rename the class
    class MyCustomPorosityKernel(PorosityGrowthKernel):

Step 3: Implement required methods
    - name (property)
    - get_default_params()
    - __init__()
    - validate_params()
    - compute_merged_porosity()      [AGGLOMERATION]
    - compute_fragment_porosity()    [BREAKAGE]
    - compute_nucleation_porosity()  [NUCLEATION]

Step 4: Register in __init__.py
    from .my_custom_kernel import MyCustomPorosityKernel
    POROSITY_GROWTH_KERNELS['my_custom'] = MyCustomPorosityKernel

Step 5: Test your kernel
    kernel = get_porosity_growth_kernel('my_custom', param1=0.1)
    
    # Agglomeration
    v_dry, poro = kernel.compute_merged_porosity(v1, p1, v2, p2)
    
    # Breakage
    poro_frag = kernel.compute_fragment_porosity(poro_parent, v_frag, v_parent)
    
    # Nucleation
    v_dry, poro = kernel.compute_nucleation_porosity(v_solid, v_liquid)

================================================================================
"""

import numpy as np
from typing import Optional, Any, Tuple

# Handle both package import and direct execution
try:
    # Package context (imported by other modules)
    from ..base import PorosityGrowthKernel
except ImportError:
    # Direct execution (python blueprint.py)
    # Mock the base class for testing purposes
    class PorosityGrowthKernel:
        """Mock base class for standalone testing."""
        pass


# =============================================================================
# KERNEL CLASS DEFINITION
# =============================================================================

class ExamplePorosityKernel(PorosityGrowthKernel):
    """
    EXAMPLE: Porosity growth kernel with [YOUR PHYSICS HERE].
    
    Replace this docstring with a clear description of your kernel's physics:
    
    Physical Model:
        Describe the physical mechanism your kernel models. For example:
        "This kernel models porosity evolution based on energy-dependent
        pore creation and collapse. High-energy events create more pores
        during agglomeration but also cause more collapse during breakage."
    
    Key Parameters:
        - param1: Description, units, typical range
        - param2: Description, units, typical range
    
    When to Use:
        Describe the application scenarios where this kernel is appropriate.
        For example: "Use this kernel for processes with varying energy
        input, such as fluidized beds or high-shear granulators."
    
    References:
        [1] Author, "Title", Journal, Year
        [2] Author, "Title", Journal, Year
    
    Example:
        >>> kernel = ExamplePorosityKernel(energy_factor=0.1)
        
        # Agglomeration
        v_dry, poro = kernel.compute_merged_porosity(
            v_dry1=1e-18, poro1=0.4,
            v_dry2=2e-18, poro2=0.5,
        )
        
        # Breakage
        poro_frag = kernel.compute_fragment_porosity(
            parent_porosity=0.4,
            fragment_volume=5e-19,
            parent_volume=1e-18,
        )
        
        # Nucleation
        v_dry, poro = kernel.compute_nucleation_porosity(
            v_solid=1e-18,
            v_liquid=1e-19
        )
    """
    
    # -------------------------------------------------------------------------
    # REQUIRED: Kernel Name
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """
        Return the kernel name (must match registry key in __init__.py).
        
        This name is used to select the kernel via factory function:
            kernel = get_porosity_growth_kernel('example_kernel')
        
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
            - Include ALL parameters needed by ALL THREE methods
            - Document units and typical ranges in comments
        
        Example structure:
            return {
                'param1': 0.1,              # Dimensionless, range: [0, 1]
                'param2': 1e-12,            # [J], energy threshold
                'nucleation_porosity': 0.4, # Default for nucleation
            }
        """
        return {
            # Your custom parameter 1
            # Description, units, range
            # Default: value (justification)
            'your_param_1': 0.1,
            
            # Your custom parameter 2
            # Description, units, range
            # Default: value (justification)
            'your_param_2': 1e-12,
            
            # Default porosity for nucleation (REQUIRED)
            # This enables porosity formation from Vollkörper
            # Range: [0, 0.999], typical: 0.3-0.5
            # Default: 0.4 (empirical value for granules)
            'nucleation_porosity': 0.4,
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
            kernel = ExamplePorosityKernel(
                your_param_1=0.2,      # Override default
                nucleation_porosity=0.35  # Override default
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
        self.nucleation_porosity = float(self.params['nucleation_porosity'])
    
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
            - Provide clear error messages with expected ranges
            - Convert types explicitly (e.g., float(), bool())
            - Validate nucleation_porosity separately (special case)
        
        IMPORTANT: Use 'params' argument, NOT 'self.params'
                   (self.params doesn't exist yet - set AFTER validation!)
        
        Example validation rules:
            if not (0 <= params['param1'] <= 1):
                raise ValueError("param1 must be in [0, 1]")
            if params['nucleation_porosity'] < 0 or params['nucleation_porosity'] > 0.999:
                raise ValueError("nucleation_porosity must be in [0, 0.999]")
        """
        # Validate custom parameters
        if not (0 <= params['your_param_1'] <= 1):
            raise ValueError(
                f"your_param_1 must be in [0, 1], got {params['your_param_1']}"
            )
        
        if params['your_param_2'] < 0:
            raise ValueError(
                f"your_param_2 must be non-negative, got {params['your_param_2']}"
            )
        
        # CRITICAL: Validate nucleation_porosity
        if not (0 <= params['nucleation_porosity'] <= 0.999):
            raise ValueError(
                f"nucleation_porosity must be in [0, 0.999], "
                f"got {params['nucleation_porosity']}"
            )
        
        return params
    
    # -------------------------------------------------------------------------
    # REQUIRED METHOD 1: Agglomeration
    # -------------------------------------------------------------------------
    
    def compute_merged_porosity(
        self,
        v_dry1: float,
        poro1: float,
        v_dry2: float,
        poro2: float,
        v_liq1: float = None,
        v_liq2: float = None,
        sat1: float = None,
        sat2: float = None,
        solver: Optional[Any] = None
    ) -> Tuple[float, float]:
        """
        Compute porosity after two particles merge (agglomeration).
        
        This method is called whenever two particles agglomerate. It computes
        the NEW porosity of the merged particle based on the parent porosities
        and your physical model.
        
        Args:
            v_dry1: Dry volume of particle 1 [m³]
                Type: float
                Range: > 0
                Note: Includes solid + pores
            
            poro1: Porosity of particle 1
                Type: float or np.nan
                Range: [0, 1] or np.nan (Vollkörper)
                Note: np.nan means solid sphere (no pores)
            
            v_dry2: Dry volume of particle 2 [m³]
                Type: float
                Range: > 0
            
            poro2: Porosity of particle 2
                Type: float or np.nan
                Range: [0, 1] or np.nan (Vollkörper)
            
            v_liq1: Liquid volume of particle 1 [m³] (optional)
                Type: float or None
                Purpose: For liquid-dependent porosity models
            
            v_liq2: Liquid volume of particle 2 [m³] (optional)
                Type: float or None
            
            sat1: Saturation of particle 1 (optional)
                Type: float or None
                Range: [0, 1] or np.nan
                Note: sat = V_liq_int / V_pore
            
            sat2: Saturation of particle 2 (optional)
                Type: float or None
            
            solver: Solver instance (optional)
                Type: MCPBESolver or None
                Purpose: Access additional state variables
        
        Returns:
            Tuple of (v_dry_merged, poro_merged):
                - v_dry_merged: New dry volume [m³] (V_solid + V_pore)
                - poro_merged: New porosity (dimensionless, [0, 1] or np.nan)
        
        CRITICAL: V_flat Consistency
            The returned v_dry_merged will be stored in V_flat[-1, new_idx].
            V_flat[:dim, new_idx] contains V_solid_merged (computed by solver).
            
            You MUST ensure: v_dry_merged >= V_solid_merged
            Otherwise: porosity would be negative (invalid!)
        
        Physical Model Template:
            # Step 1: Compute solid volumes from parents
            V_solid_1 = v_dry1 * (1.0 - poro1) if not np.isnan(poro1) else v_dry1
            V_solid_2 = v_dry2 * (1.0 - poro2) if not np.isnan(poro2) else v_dry2
            V_solid_merged = V_solid_1 + V_solid_2
            
            # Step 2: Compute pore volumes from parents
            V_pore_1 = v_dry1 * poro1 if not np.isnan(poro1) else 0.0
            V_pore_2 = v_dry2 * poro2 if not np.isnan(poro2) else 0.0
            
            # Step 3: Apply YOUR physics model
            # Example: Add trapped pores
            V_pore_trapped = self.your_param_1 * min(v_dry1, v_dry2)
            V_pore_merged = V_pore_1 + V_pore_2 + V_pore_trapped
            
            # Step 4: Compute merged dry volume and porosity
            v_dry_merged = V_solid_merged + V_pore_merged
            poro_merged = V_pore_merged / v_dry_merged if v_dry_merged > 0 else np.nan
            
            return v_dry_merged, poro_merged
        
        Example (Volume Mixing - Simple Addition):
            # Solid volumes
            V_solid_1 = v_dry1 * (1.0 - poro1) if not np.isnan(poro1) else v_dry1
            V_solid_2 = v_dry2 * (1.0 - poro2) if not np.isnan(poro2) else v_dry2
            V_solid_merged = V_solid_1 + V_solid_2
            
            # Pore volumes (simple addition)
            V_pore_1 = v_dry1 * poro1 if not np.isnan(poro1) else 0.0
            V_pore_2 = v_dry2 * poro2 if not np.isnan(poro2) else 0.0
            V_pore_merged = V_pore_1 + V_pore_2
            
            # Result
            v_dry_merged = V_solid_merged + V_pore_merged
            poro_merged = V_pore_merged / v_dry_merged if v_dry_merged > 0 else np.nan
            
            return v_dry_merged, poro_merged
        
        """
        # =====================================================================
        # STEP 1: Extract solid and pore volumes from parents
        # =====================================================================
        
        # Particle 1
        if np.isnan(poro1):
            # Vollkörper: all volume is solid
            V_solid_1 = v_dry1
            V_pore_1 = 0.0
        else:
            # Porous: split into solid and pore
            V_solid_1 = v_dry1 * (1.0 - poro1)
            V_pore_1 = v_dry1 * poro1
        
        # Particle 2
        if np.isnan(poro2):
            V_solid_2 = v_dry2
            V_pore_2 = 0.0
        else:
            V_solid_2 = v_dry2 * (1.0 - poro2)
            V_pore_2 = v_dry2 * poro2
        
        # Merged solid volume (always conserved)
        V_solid_merged = V_solid_1 + V_solid_2
        
        # =====================================================================
        # STEP 2: Apply YOUR porosity model
        # =====================================================================
        
        # Start with base pore volume (simple addition)
        V_pore_merged = V_pore_1 + V_pore_2
        
        # TODO: Add your physics here!
        # Examples:
        
        # Example A: Trapped pores (incomplete mixing)
        # V_pore_trapped = self.your_param_1 * min(v_dry1, v_dry2)
        # V_pore_merged += V_pore_trapped
        
        # Example B: Liquid-assisted pore collapse
        # if sat1 is not None and sat2 is not None:
        #     sat_avg = (sat1 + sat2) / 2.0
        #     collapse_factor = 1.0 - self.your_param_1 * sat_avg
        #     V_pore_merged *= collapse_factor
        
        # =====================================================================
        # STEP 3: Compute merged dry volume and porosity
        # =====================================================================
        
        v_dry_merged = V_solid_merged + V_pore_merged
        
        if v_dry_merged <= 0:
            # Edge case: no volume
            return 0.0, np.nan
        
        if V_pore_merged <= 0:
            # No pores → Vollkörper
            poro_merged = np.nan
        else:
            poro_merged = V_pore_merged / v_dry_merged
        
        # Clamp porosity to valid range
        poro_merged = max(0.0, min(1.0, poro_merged))
        
        return v_dry_merged, poro_merged
    
    # -------------------------------------------------------------------------
    # REQUIRED METHOD 2: Breakage
    # -------------------------------------------------------------------------
    
    def compute_fragment_porosity(
        self,
        parent_porosity: float,
        fragment_volume: float,
        parent_volume: float,
        solver: Optional[Any] = None
    ) -> float:
        """
        Compute porosity for a fragment after breakage.
        
        This method is called whenever a particle breaks. It computes the
        porosity of the resulting fragment(s) based on the parent porosity
        and your physical model.
        
        Args:
            parent_porosity: Porosity of parent particle
                Type: float or np.nan
                Range: [0, 1] or np.nan (Vollkörper)
                Note: np.nan means solid sphere (no pores)
            
            fragment_volume: Solid volume of fragment [m³]
                Type: float
                Range: > 0
                Note: This is V_solid (not V_dry!)
            
            parent_volume: Solid volume of parent [m³]
                Type: float
                Range: > 0
                Note: Total solid volume before breakage
            
            solver: Solver instance (optional)
                Type: MCPBESolver or None
        
        Returns:
            fragment_porosity: Porosity of fragment
                Type: float or np.nan
                Range: [0, 1] or np.nan (if parent was Vollkörper)
        
        Physical Model Template:
            # Vollkörper stays Vollkörper
            if np.isnan(parent_porosity):
                return np.nan
            
            # Base model: inherit parent porosity
            frag_porosity = parent_porosity
            
            # TODO: Add your physics here!
            # Examples:
            
            # Example A: Size-dependent porosity (smaller = denser)
            # size_ratio = fragment_volume / parent_volume
            # frag_porosity = parent_porosity * size_ratio ** self.your_param_1
            
            return frag_porosity
        
        Example (Volume Mixing - Inherit Parent):
            # Fragments inherit parent porosity (equal splitting of solid and pores)
            return parent_porosity
        
        """
        # =====================================================================
        # STEP 1: Handle Vollkörper case
        # =====================================================================
        
        if np.isnan(parent_porosity):
            # Solid spheres produce solid fragments
            return np.nan
        
        # =====================================================================
        # STEP 2: Apply YOUR porosity model
        # =====================================================================
        
        # Start with base model (inherit parent porosity)
        frag_porosity = parent_porosity
        
        # TODO: Add your physics here!
        # Examples:
        
        # Example A: Pore collapse (incomplete mixing)
        # collapse_factor = self.your_param_1  # e.g., 0.1 = 10% collapse
        # frag_porosity = parent_porosity * (1.0 - collapse_factor)
        
        # Example B: Size-dependent (smaller fragments are denser)
        # size_ratio = fragment_volume / parent_volume
        # frag_porosity = parent_porosity * (size_ratio ** self.your_param_1)
        
        # =====================================================================
        # STEP 3: Ensure valid porosity
        # =====================================================================
        
        frag_porosity = max(0.0, min(1.0, frag_porosity))
        
        return frag_porosity
    
    # -------------------------------------------------------------------------
    # REQUIRED METHOD 3: Nucleation
    # -------------------------------------------------------------------------
    
    def compute_nucleation_porosity(
        self,
        v_solid: float,
        v_liquid: float,
        nucleation_params: dict = None,
        solver: Optional[Any] = None
    ) -> Tuple[float, float]:
        """
        Compute porosity for a newly nucleated particle.
        
        This method is called whenever a new particle forms via nucleation
        (liquid droplet addition). It computes the INITIAL porosity of the
        new particle.
        
        CRITICAL: Why is a default porosity needed?
            With simple volume mixing, porosity is ONLY additive:
                V_pore_new = V_pore₁ + V_pore₂
            
            If both parents are Vollkörper (poro=NaN), the child would
            ALWAYS be Vollkörper. Porosity could NEVER form!
            
            Solution: Nucleation creates particles with DEFAULT porosity,
            enabling porosity formation from initially solid particles.
        
        Args:
            v_solid: Solid volume of nucleus [m³]
                Type: float
                Range: > 0
                Note: From added droplet + captured solids
            
            v_liquid: Liquid volume associated with nucleus [m³]
                Type: float
                Range: >= 0
                Note: May affect porosity (solvent trapping)
            
            nucleation_params: Additional parameters (optional)
                Type: dict or None
                Keys:
                    - 'default_porosity': Override kernel default (rarely needed)
                Note: Usually None (kernel uses its own default)
            
            solver: Solver instance (optional)
                Type: MCPBESolver or None
        
        Returns:
            Tuple of (v_dry, porosity):
                - v_dry: Dry volume [m³] (V_solid + V_pore)
                - porosity: Initial porosity (dimensionless, [0, 1] or np.nan)
        
        Physical Model Template:
            # Get default porosity from kernel config
            default_porosity = self.nucleation_porosity
            
            # Allow override via params (rarely needed)
            if nucleation_params and 'default_porosity' in nucleation_params:
                default_porosity = nucleation_params['default_porosity']
            
            # Option A: Create porous nucleus
            if default_porosity > 0:
                v_dry = v_solid / (1.0 - default_porosity)
                return v_dry, default_porosity
            
            # Option B: Create Vollkörper (slower porosity formation)
            else:
                return v_solid, np.nan
        
        Example (Default Porosity 0.4):
            porosity = self.nucleation_porosity  # 0.4
            v_dry = v_solid / (1.0 - porosity)
            return v_dry, porosity
        
        Example (Liquid-Dependent Porosity):
            # More liquid → higher initial porosity (trapped solvent)
            liq_ratio = v_liquid / v_solid if v_solid > 0 else 0.0
            porosity = min(0.6, self.nucleation_porosity + 0.2 * liq_ratio)
            v_dry = v_solid / (1.0 - porosity)
            return v_dry, porosity
        """
        # =====================================================================
        # STEP 1: Get default porosity
        # =====================================================================
        
        # Use kernel's configured default
        porosity = self.nucleation_porosity
        
        # Allow override via nucleation_params (advanced use)
        if nucleation_params is not None and 'default_porosity' in nucleation_params:
            porosity = float(nucleation_params['default_porosity'])
        
        # =====================================================================
        # STEP 2: Handle Vollkörper option
        # =====================================================================
        
        if porosity <= 0:
            # Create solid nucleus (Vollkörper)
            # Note: Porosity can only form via agglomeration later
            return float(v_solid), np.nan
        
        # =====================================================================
        # STEP 3: Create porous nucleus
        # =====================================================================
        
        # Clamp porosity to valid range
        porosity = max(0.0, min(0.999, porosity))
        
        # Compute dry volume from solid volume and porosity
        # Formula: V_dry = V_solid / (1 - porosity)
        v_dry = float(v_solid) / (1.0 - porosity)
        
        return v_dry, porosity
    
    # -------------------------------------------------------------------------
    # OPTIONAL: Helper Methods
    # -------------------------------------------------------------------------
    
    def _compute_pore_volume(
        self,
        v_dry: float,
        porosity: float
    ) -> float:
        """
        Compute pore volume from dry volume and porosity (helper).
        
        Args:
            v_dry: Dry volume [m³]
            porosity: Porosity (dimensionless)
        
        Returns:
            Pore volume [m³] (0.0 if Vollkörper)
        """
        if np.isnan(porosity):
            return 0.0
        return v_dry * porosity
    
    def _compute_solid_volume(
        self,
        v_dry: float,
        porosity: float
    ) -> float:
        """
        Compute solid volume from dry volume and porosity (helper).
        
        Args:
            v_dry: Dry volume [m³]
            porosity: Porosity (dimensionless)
        
        Returns:
            Solid volume [m³]
        """
        if np.isnan(porosity):
            return v_dry
        return v_dry * (1.0 - porosity)
    


# =============================================================================
# TESTING SECTION (run this file directly to test)
# =============================================================================

if __name__ == '__main__':
    """
    Quick test of the kernel implementation.
    
    Run: python blueprint.py
    """
    print("=" * 80)
    print("POROSITY GROWTH KERNEL TEST")
    print("=" * 80)
    
    # Create kernel with default parameters
    print("\n1. Creating kernel with default parameters...")
    kernel = ExamplePorosityKernel()
    print(f"   ✓ Kernel created: {kernel.name}")
    print(f"   ✓ Parameters: {kernel.params}")
    
    # Test agglomeration
    print("\n2. Testing compute_merged_porosity()...")
    v_dry, poro = kernel.compute_merged_porosity(
        v_dry1=1e-18, poro1=0.4,
        v_dry2=2e-18, poro2=0.5
    )
    print(f"   Agg: V_dry={v_dry*1e18:.2f} µm³, poro={poro:.3f}")
    
    # Test breakage
    print("\n3. Testing compute_fragment_porosity()...")
    poro_frag = kernel.compute_fragment_porosity(
        parent_porosity=0.4,
        fragment_volume=5e-19,
        parent_volume=1e-18
    )
    print(f"   Break: poro_frag={poro_frag:.3f}")
    
    # Test nucleation
    print("\n4. Testing compute_nucleation_porosity()...")
    v_dry_nuc, poro_nuc = kernel.compute_nucleation_porosity(
        v_solid=1e-18,
        v_liquid=1e-19
    )
    print(f"   Nuc: V_dry={v_dry_nuc*1e18:.2f} µm³, poro={poro_nuc:.3f}")
    
    # Test Vollkörper handling
    print("\n5. Testing Vollkörper handling...")
    
    # Vollkörper + porous
    v_dry_mix, poro_mix = kernel.compute_merged_porosity(
        v_dry1=1e-18, poro1=np.nan,  # Vollkörper
        v_dry2=1e-18, poro2=0.4
    )
    print(f"   Vollkörper + porous: poro={poro_mix:.3f}")
    
    # Vollkörper breakage
    poro_vk = kernel.compute_fragment_porosity(
        parent_porosity=np.nan,
        fragment_volume=5e-19,
        parent_volume=1e-18
    )
    print(f"   Vollkörper breakage: poro={poro_vk} (np.nan expected)")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
