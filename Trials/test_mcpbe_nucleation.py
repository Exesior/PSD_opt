"""Test script for WMCPBE with nucleation support.

This script follows the same structure as simple_validation.py but extends
it to support nucleation processes using the MCPBENucleation mixin.

Nucleation process (from mcpbe_nucleation.py):
- Liquid droplets are distributed onto particles
- Distribution criterion: solid_volume > liquid_volume
- Droplet rate calculated from liquid_flow_rate and droplet_diameter
"""

from __future__ import annotations

import numpy as np

# Import validation framework classes
try:
    from scripts.pbe_validation.validation import (
        CaseConfig,
        GranulationValidationConfig,
        GranulationValidationRunner,
        ValidationPlotter,
        WMCPBEVariantConfig,
    )
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from scripts.pbe_validation.validation import (
        CaseConfig,
        GranulationValidationConfig,
        GranulationValidationRunner,
        ValidationPlotter,
        WMCPBEVariantConfig,
    )


# =============================================================================
# Configuration Classes
# =============================================================================

class NucleationConfig:
    """Configuration for nucleation processes.
    
    Parameters match those in MCPBENucleation (mcpbe_nucleation.py):
    
    Attributes:
        liquid_flow_rate: Liquid flow rate in L/s
        droplet_diameter: Diameter of individual droplets in meters
        enable: Whether nucleation is enabled
    """
    
    def __init__(
        self,
        liquid_flow_rate: float = 0.0,
        droplet_diameter: float = 0.0,
        enable: bool = True,
    ):
        self.liquid_flow_rate = liquid_flow_rate
        self.droplet_diameter = droplet_diameter
        self.enable = enable
    
    def to_dict(self) -> dict:
        """Convert to dictionary for solver attribute assignment."""
        if not self.enable:
            return {"nucleation_enable": False}
        return {
            "nucleation_enable": True,
            "liquid_flow_rate": self.liquid_flow_rate,
            "droplet_diameter": self.droplet_diameter,
        }
    
    @property
    def droplet_volume(self) -> float:
        """Calculate droplet volume from diameter (sphere formula)."""
        if self.droplet_diameter <= 0:
            return 0.0
        return (np.pi / 6.0) * (self.droplet_diameter ** 3)
    
    @property
    def droplet_rate(self) -> float:
        """Calculate droplets per second."""
        if not self.enable or self.liquid_flow_rate <= 0:
            return 0.0
        vol_m3 = self.droplet_volume
        if vol_m3 <= 0:
            return 0.0
        # Convert L/s to m³/s
        flow_m3_per_s = self.liquid_flow_rate * 1e-3
        return flow_m3_per_s / vol_m3


class WMCPBEVariantConfigWithNucleation(WMCPBEVariantConfig):
    """WMCPBE variant configuration extended for nucleation.
    
    Extends WMCPBEVariantConfig with nucleation-specific parameters.
    """
    
    def __init__(
        self,
        name: str,
        repeats: int = 1,
        base_seed: int = 42,
        attrs: dict | None = None,
        nucleation_config: NucleationConfig | None = None,
    ):
        super().__init__(name=name, repeats=repeats, base_seed=base_seed, attrs=attrs or {})
        self.nucleation_config = nucleation_config
    
    def get_all_attrs(self) -> dict:
        """Get combined attributes including nucleation parameters."""
        all_attrs = dict(self.attrs)
        if self.nucleation_config:
            all_attrs.update(self.nucleation_config.to_dict())
        return all_attrs


# =============================================================================
# Factory Functions
# =============================================================================

def create_base_case(
    dim: int = 2,
    kernel: str = "const",
    process: str = "breakage",
    t_end: float = 10.0,
    dt: float = 2.0,
    x: float = 2e-1,
    beta0: float = 1e-3,
    p1: float = 3e-2,
    p2: float = 1.0,
    initial_number_density: float = 100000.0,
    initial_total_weight: float = 100000.0,
) -> CaseConfig:
    """Create a base case configuration."""
    return CaseConfig(
        dim=dim,
        kernel=kernel,
        process=process,
        t_vec=np.arange(0.0, t_end, dt, dtype=float),
        x=x,
        beta0=beta0,
        p1=p1,
        p2=p2,
        initial_number_density=initial_number_density,
        initial_total_weight=initial_total_weight,
    )


def create_wmcpbe_variants_granulation() -> list[WMCPBEVariantConfig]:
    """Create WMCPBE variant configurations for granulation only."""
    return [
        WMCPBEVariantConfig(
            name="WMCPBE baseline",
            repeats=1,
            base_seed=42,
            attrs={
                "recon_enable": False,
                "break_dW_max": 1.0,
                "agg_dW_max": 1.0,
            },
        ),
        WMCPBEVariantConfig(
            name="WMCPBE reconstructed",
            repeats=1,
            base_seed=43,
            attrs={
                "recon_enable": True,
                "recon_method": "4PMC",
                "recon_N_max": 4000,
                "recon_bins": 30,
                "recon_RS_target": 1000,
                "break_dW_max": 10.0,
                "agg_dW_max": 1.0,
            },
        ),
    ]


def create_wmcpbe_variants_with_nucleation() -> list[WMCPBEVariantConfigWithNucleation]:
    """Create WMCPBE variants comparing different nucleation scenarios.
    
    Scenarios:
    1. No nucleation (baseline)
    2. Low nucleation rate
    3. High nucleation rate
    4. Different droplet sizes
    """
    return [
        # Baseline: No nucleation
        WMCPBEVariantConfigWithNucleation(
            name="No nucleation (baseline)",
            repeats=1,
            base_seed=42,
            attrs={
                "recon_enable": False,
                "break_dW_max": 1.0,
                "agg_dW_max": 1.0,
            },
            nucleation_config=NucleationConfig(
                enable=False,
            ),
        ),
        
        # Low nucleation rate
        WMCPBEVariantConfigWithNucleation(
            name="Low nucleation (0.1 mL/s, 50µm)",
            repeats=1,
            base_seed=43,
            attrs={
                "recon_enable": True,
                "recon_method": "4PMC",
                "recon_N_max": 4000,
                "recon_bins": 30,
                "recon_RS_target": 1000,
                "break_dW_max": 10.0,
                "agg_dW_max": 1.0,
            },
            nucleation_config=NucleationConfig(
                enable=True,
                liquid_flow_rate=0.1e-3,  # 0.1 mL/s = 0.0001 L/s
                droplet_diameter=50e-6,   # 50 micrometers
            ),
        ),
        
        # High nucleation rate
        WMCPBEVariantConfigWithNucleation(
            name="High nucleation (1.0 mL/s, 50µm)",
            repeats=1,
            base_seed=44,
            attrs={
                "recon_enable": True,
                "recon_method": "4PMC",
                "recon_N_max": 4000,
                "recon_bins": 30,
                "recon_RS_target": 1000,
                "break_dW_max": 10.0,
                "agg_dW_max": 1.0,
            },
            nucleation_config=NucleationConfig(
                enable=True,
                liquid_flow_rate=1.0e-3,  # 1.0 mL/s = 0.001 L/s
                droplet_diameter=50e-6,   # 50 micrometers
            ),
        ),
        
        # Small droplets (same flow rate, more droplets)
        WMCPBEVariantConfigWithNucleation(
            name="Small droplets (1.0 mL/s, 10µm)",
            repeats=1,
            base_seed=45,
            attrs={
                "recon_enable": True,
                "recon_method": "4PMC",
                "recon_N_max": 4000,
                "recon_bins": 30,
                "recon_RS_target": 1000,
                "break_dW_max": 10.0,
                "agg_dW_max": 1.0,
            },
            nucleation_config=NucleationConfig(
                enable=True,
                liquid_flow_rate=1.0e-3,  # 1.0 mL/s
                droplet_diameter=10e-6,   # 10 micrometers (smaller = more droplets)
            ),
        ),
    ]


# =============================================================================
# Validation Runners
# =============================================================================

def run_granulation_validation(verbose: bool = True) -> None:
    """Run granulation validation (breakage/agglomeration only)."""
    print("=" * 70)
    print("WMCPBE Granulation Validation")
    print("=" * 70)
    
    case = create_base_case(
        dim=2,
        kernel="const",
        process="breakage",
        t_end=10.0,
        dt=2.0,
        x=2e-1,
        beta0=1e-3,
        p1=3e-2,
        p2=1.0,
        initial_number_density=100000.0,
        initial_total_weight=100000.0,
    )
    
    wmcpbe_variants = create_wmcpbe_variants_granulation()
    
    config = GranulationValidationConfig(
        case=case,
        wmcpbe_variants=wmcpbe_variants,
        verbose=verbose,
    )
    
    result = GranulationValidationRunner(config).run()
    plotter = ValidationPlotter(result)
    plotter.plot_all_moments(relative=True, include_total_volume=False)
    plotter.show()


def run_nucleation_validation(verbose: bool = True) -> None:
    """Run nucleation validation.
    
    Compares different nucleation scenarios:
    - No nucleation (baseline)
    - Low nucleation rate
    - High nucleation rate  
    - Different droplet sizes
    
    Note: Requires nucleation support in MCPBE solver.
    """
    print("=" * 70)
    print("WMCPBE Nucleation Validation")
    print("=" * 70)
    
    # Nucleation works best with agglomeration or mix process
    # since new nuclei need to agglomerate with existing particles
    case = create_base_case(
        dim=2,
        kernel="const",
        process="agglomeration",  # or "mix" for breakage + agglomeration + nucleation
        t_end=10.0,
        dt=1.0,
        x=2e-1,
        beta0=1e-3,
        p1=3e-2,
        p2=1.0,
        initial_number_density=100000.0,
        initial_total_weight=100000.0,
    )
    
    wmcpbe_variants = create_wmcpbe_variants_with_nucleation()
    
    # Print nucleation configuration summary
    print("\nNucleation Configuration Summary:")
    print("-" * 70)
    for variant in wmcpbe_variants:
        nc = variant.nucleation_config
        if nc and nc.enable:
            print(f"  {variant.name}:")
            print(f"    Flow rate: {nc.liquid_flow_rate*1e6:.1f} µL/s")
            print(f"    Droplet size: {nc.droplet_diameter*1e6:.1f} µm")
            print(f"    Droplet rate: {nc.droplet_rate:.2e} droplets/s")
        else:
            print(f"  {variant.name}: Nucleation disabled")
    print("-" * 70)
    
    config = GranulationValidationConfig(
        case=case,
        wmcpbe_variants=wmcpbe_variants,
        verbose=verbose,
    )
    
    result = GranulationValidationRunner(config).run()
    plotter = ValidationPlotter(result)
    plotter.plot_all_moments(relative=True, include_total_volume=True)
    plotter.show()


def demonstrate_nucleation_calculations() -> None:
    """Demonstrate nucleation parameter calculations."""
    print("=" * 70)
    print("Nucleation Parameter Calculations")
    print("=" * 70)
    
    examples = [
        {"flow_mL_s": 0.1, "diameter_um": 50},
        {"flow_mL_s": 1.0, "diameter_um": 50},
        {"flow_mL_s": 1.0, "diameter_um": 10},
        {"flow_mL_s": 0.5, "diameter_um": 20},
    ]
    
    print(f"\n{'Flow (mL/s)':>12} | {'Diameter (µm)':>14} | {'Vol (fL)':>12} | {'Rate (1/s)':>14}")
    print("-" * 70)
    
    for ex in examples:
        flow_L_s = ex["flow_mL_s"] * 1e-3  # mL/s -> L/s
        diam_m = ex["diameter_um"] * 1e-6  # µm -> m
        
        nc = NucleationConfig(
            liquid_flow_rate=flow_L_s,
            droplet_diameter=diam_m,
            enable=True,
        )
        
        vol_fL = nc.droplet_volume * 1e15  # m³ -> fL
        rate = nc.droplet_rate
        
        print(f"{ex['flow_mL_s']:>12.1f} | {ex['diameter_um']:>14.1f} | {vol_fL:>12.2f} | {rate:>14.2e}")
    
    print("-" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="WMCPBE validation test script with nucleation support"
    )
    parser.add_argument(
        "--mode",
        choices=["granulation", "nucleation", "demo"],
        default="granulation",
        help="Validation mode:"
             " 'granulation' (breakage/agglo only),"
             " 'nucleation' (with liquid addition),"
             " 'demo' (show calculations)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    if args.mode == "granulation":
        run_granulation_validation(verbose=not args.quiet)
    
    elif args.mode == "nucleation":
        run_nucleation_validation(verbose=not args.quiet)
    
    elif args.mode == "demo":
        demonstrate_nucleation_calculations()
        print("\nTo run nucleation validation:")
        print("  python test_mcpbe_nucleation.py --mode nucleation")
        print("\nTo run granulation validation:")
        print("  python test_mcpbe_nucleation.py --mode granulation")
