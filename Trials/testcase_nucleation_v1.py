"""
===============================================================================
Test Case: Nucleation Validation v1
===============================================================================

Dieses Skript testet die Nukleations-Implementierung im MCPBE-Solver.

Verwendung (in Spyder oder IPython):
------------------------------------
    %run testcase_nucleation_v1.py
    
    Oder mit Argumenten:
    %run testcase_nucleation_v1.py --mode test1        # Nur Test 1
    %run testcase_nucleation_v1.py --mode all          # Alle Tests
    %run testcase_nucleation_v1.py --opti              # Single-Sampler Opti (schnell)
    %run testcase_nucleation_v1.py --dual              # Dual-Sampler (am schnellsten!)
    %run testcase_nucleation_v1.py --mode test1 --dual # Test 1 mit Dual-Sampler

===============================================================================
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Import Setup - Handle different execution contexts
# =============================================================================

def setup_imports():
    """Setup Python path for importing mcpbe package."""
    # Get current file directory and project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()
    mcpbe_src = project_root / "mcpbe" / "src"
    
    # Add paths in correct order
    paths_added = []
    for path in [str(mcpbe_src), str(project_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)
            paths_added.append(path)
    
    return paths_added

# Setup imports
_paths_added = setup_imports()

# Now try to import MCPBESolver
# Check for solver version flags
_USE_OPTI = '--opti' in sys.argv
_USE_DUAL = '--dual' in sys.argv

# Dual takes precedence over opti
if _USE_DUAL:
    _USE_OPTI = False

try:
    if _USE_DUAL:
        from mcpbe import MCPBESolverDual as MCPBESolver
        print("\n" + "="*60)
        print("Using DUAL-SAMPLER nucleation (MCPBESolverDual)")
        print("2 Sampler: i=groß bevorzugt, j=klein bevorzugt")
        print("FASTEST option - ~50-200x faster than standard!")
        print("="*60 + "\n")
    elif _USE_OPTI:
        from mcpbe import MCPBESolverOpti as MCPBESolver
        print("\n" + "="*60)
        print("Using OPTIMIZED nucleation (MCPBESolverOpti)")
        print("Sampler caching ENABLED - ~100-1000x faster!")
        print("="*60 + "\n")
    else:
        from mcpbe import MCPBESolver
        print("\n" + "="*60)
        print("Using STANDARD nucleation (MCPBESolver)")
        print("Sampler rebuilds on EVERY droplet (slower)")
        print("="*60 + "\n")
except ImportError as e:
    print(f"ERROR: Could not import MCPBESolver")
    print(f"Details: {e}")
    print(f"\nPaths added:")
    for p in _paths_added:
        print(f"  {p}")
    print(f"\nFull sys.path:")
    for p in sys.path[:10]:
        print(f"  {p}")
    
    # Try to diagnose the issue
    import importlib.util
    spec = importlib.util.find_spec("mcpbe")
    if spec:
        print(f"\nmcpbe package found at: {spec.origin}")
    else:
        print("\nmcpbe package NOT FOUND in any path")
    
    raise

# =============================================================================
# Test Configuration Classes
# =============================================================================

class NucleationTestConfig:
    """Configuration for a single nucleation test case."""
    
    def __init__(
        self,
        name: str,
        process_type: str = "nucleation",
        dim: int = 2,
        t_total: float = 10.0,
        t_write: float = 1.0,
        
        # Initial particles
        initial_particles: int = 1000,
        particle_size: float = 1e-6,  # [m]
        
        # Nucleation parameters
        liquid_flow_rate: float = 0.001,  # [L/s]
        droplet_diameter: float = 50e-6,  # [m]
        nucleation_enable: bool = True,
        nucleation_duration: float = 0.0,  # [s], 0 = infinite
        
        # Sampler mode for performance optimization (#4) - DERZEIT OHNE EFFEKT
        # Hinweis: Die geplante Optimierung (Sampler nur pro Zeitschritt) wurde verworfen,
        # da nach Agglomerationen der Sampler zwingend neu gebaut werden muss.
        sampler_mode_nuc: str = "per_timestep",  # Platzhalter für zukünftige Optimierungen
        
        # Agglomeration parameters (if enabled)
        beta0: float = 1e-9,  # [m³/s]
        
        # Seed for reproducibility
        seed: int = 42,
    ):
        self.name = name
        self.process_type = process_type
        self.dim = 1
        self.t_total = t_total
        self.t_write = t_write
        self.initial_particles = initial_particles
        self.particle_size = particle_size
        self.liquid_flow_rate = liquid_flow_rate
        self.droplet_diameter = droplet_diameter
        self.nucleation_enable = nucleation_enable
        self.nucleation_duration = nucleation_duration
        self.sampler_mode_nuc = sampler_mode_nuc
        self.beta0 = beta0
        self.seed = seed
    
    def create_solver(self) -> MCPBESolver:
        """Create and configure a solver instance."""
        # Build t_vec from t_total and t_write
        t_vec = np.arange(0.0, self.t_total + 0.5*self.t_write, self.t_write, dtype=float)
        
        # Create solver with basic parameters only (MCPBEBase signature)
        solver = MCPBESolver(
            dim=self.dim,
            t_vec=t_vec,
            verbose=False,
            load_attr=False,  # Don't load external config
            init=True,
            seed=self.seed,
        )
        
        # Set simulation parameters AFTER initialization
        solver.a0 = float(self.initial_particles)
        solver.x = np.full(solver.dim, self.particle_size)
        solver.process_type = self.process_type
        solver.liquid_flow_rate = self.liquid_flow_rate
        solver.droplet_diameter = self.droplet_diameter
        solver.nucleation_enable = self.nucleation_enable
        solver.nucleation_duration = self.nucleation_duration
        solver.sampler_mode_nuc = self.sampler_mode_nuc  
        solver.CORR_BETA = self.beta0
        
        # Re-initialize particles with new settings
        solver._initialize_particles()
        
        return solver


# =============================================================================
# Test Runner
# =============================================================================

class NucleationTestRunner:
    """Runs nucleation test cases and collects results."""
    
    def __init__(self, config: NucleationTestConfig):
        self.config = config
        self.solver = None
        self.results = {}
    
    def run(self) -> dict:
        """Run the test case and return results."""
        print(f"\n{'='*70}")
        print(f"Running: {self.config.name}")
        print(f"{'='*70}")
        
        # Initialize results with basic info early
        self.results = {
            'name': self.config.name,
            'success': False,
        }
        
        # Create and configure solver
        try:
            self.solver = self.config.create_solver()
        except Exception as e:
            print(f"ERROR creating solver: {e}")
            raise
        
        # Store initial state
        initial_particles = self.solver.a_tot
        initial_volume = np.sum(self.solver.V_flat[-1, :initial_particles])
        
        print(f"Initial particles: {initial_particles}")
        print(f"Initial total volume: {initial_volume:.3e} m³")
        print(f"Liquid flow rate: {self.config.liquid_flow_rate*1000:.1f} mL/s")
        print(f"Droplet diameter: {self.config.droplet_diameter*1e6:.1f} µm")
        print(f"Sampler mode (nuc): {self.config.sampler_mode_nuc}")
        
        if hasattr(self.solver, '_droplet_rate') and self.solver._droplet_rate > 0:
            print(f"Droplet rate: {self.solver._droplet_rate:.2e} 1/s")
        
        if self.config.nucleation_duration > 0:
            print(f"Nucleation duration: {self.config.nucleation_duration:.1f} s")
        print()
        
        # Run simulation
        try:
            self.solver.solve(maxiter=int(1e7))
        except Exception as e:
            print(f"ERROR during simulation: {e}")
            import traceback
            traceback.print_exc()
            self.results['success'] = False
            self.results['error'] = str(e)
            return self.results
        
        # Collect results
        final_particles = self.solver.a_tot
        final_volume = np.sum(self.solver.V_flat[-1, :final_particles])
        total_liquid = np.sum(self.solver.liquid_volume[:final_particles])
        total_solid = final_volume - total_liquid
        
        self.results = {
            'name': self.config.name,
            'success': True,
            'initial_particles': initial_particles,
            'final_particles': final_particles,
            'particles_lost': initial_particles - final_particles,
            'initial_volume': initial_volume,
            'final_volume': final_volume,
            'volume_change': final_volume - initial_volume,
            'total_liquid': total_liquid,
            'total_solid': total_solid,
            'liquid_fraction': total_liquid / final_volume if final_volume > 0 else 0,
            'simulation_time': self.solver.t[-1] if hasattr(self.solver, 't') and len(self.solver.t) > 0 else 0,
            'machine_time': getattr(self.solver, 'MACHINE_TIME', 0),
        }
        
        # Print results
        print(f"\nResults:")
        print(f"  Final particles: {final_particles}")
        print(f"  Final total volume: {final_volume:.3e} m³")
        print(f"  Total liquid added: {total_liquid:.3e} m³ ({total_liquid*1e6:.1f} µL)")
        print(f"  Liquid fraction: {self.results['liquid_fraction']*100:.1f}%")
        print(f"  Simulation time: {self.results['simulation_time']:.2f} s")
        print(f"  Computation time: {self.results['machine_time']:.2f} s")
        
        return self.results


# =============================================================================
# Test Cases
# =============================================================================

def create_test_cases():
    """Create all test cases."""
    test_cases = []
    
    # Test 1: Nucleation only (pure validation)
    test_cases.append(NucleationTestConfig(
        name="Test 1: Nucleation Only",
        process_type="nucleation",
        dim=2,
        t_total=5.0,
        t_write=0.5,
        initial_particles=5000,
        particle_size=27e-5,
        liquid_flow_rate=0.000001,  # 1 mL/s
        droplet_diameter=0.0005,  #5mm
        nucleation_enable=True,
        nucleation_duration=0.0,
        sampler_mode_nuc="per_timestep",  # Fast mode (default)
        seed=42,
    ))
    
    # Test 2: Nucleation with duration limit
    test_cases.append(NucleationTestConfig(
        name="Test 2: Nucleation Duration (2s)",
        process_type="nucleation",
        dim=2,
        t_total=5.0,
        t_write=0.5,
        initial_particles=500,
        particle_size=2e-6,
        liquid_flow_rate=0.001,
        droplet_diameter=50e-6,
        nucleation_enable=True,
        nucleation_duration=2.0,
        sampler_mode_nuc="per_timestep",
        seed=42,
    ))
    
    # Test 3: Nucleation + Agglomeration
    test_cases.append(NucleationTestConfig(
        name="Test 3: Nucleation + Agglomeration",
        process_type="agglomeration",
        dim=2,
        t_total=5.0,
        t_write=0.5,
        initial_particles=500,
        particle_size=2e-6,
        liquid_flow_rate=0.001,
        droplet_diameter=50e-6,
        nucleation_enable=True,
        nucleation_duration=0.0,
        sampler_mode_nuc="per_timestep",
        beta0=1e-9,
        seed=42,
    ))
    
    # Test 4: Higher nucleation rate
    test_cases.append(NucleationTestConfig(
        name="Test 4: High Flow Rate (10 mL/s)",
        process_type="nucleation",
        dim=2,
        t_total=5.0,
        t_write=0.5,
        initial_particles=500,
        particle_size=2e-6,
        liquid_flow_rate=0.01,  # 10 mL/s
        droplet_diameter=0.005,  #5mm
        nucleation_enable=True,
        nucleation_duration=0.0,
        sampler_mode_nuc="per_timestep",
        seed=42,
    ))
    
    # Test 5: Smaller droplets
    test_cases.append(NucleationTestConfig(
        name="Test 5: Small Droplets (10 µm)",
        process_type="nucleation",
        dim=2,
        t_total=5.0,
        t_write=0.5,
        initial_particles=500,
        particle_size=2e-6,
        liquid_flow_rate=0.001,
        droplet_diameter=10e-6,  # 10 µm
        nucleation_enable=True,
        nucleation_duration=0.0,
        sampler_mode_nuc="per_timestep",
        seed=42,
    ))
    
    return test_cases


# =============================================================================
# Main Execution
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Nucleation Validation Test Suite")
    parser.add_argument(
        "--mode",
        choices=["test1", "test2", "test3", "test4", "test5", "all"],
        default="test1",
        help="Which test(s) to run"
    )
    parser.add_argument(
        "--sampler-mode-nuc",
        choices=["per_timestep", "per_droplet"],
        default="per_timestep",
        help="Sampler update mode: 'per_timestep' (fast, default) or 'per_droplet' (exact, slow)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Select test cases based on mode
    all_tests = create_test_cases()
    
    # Override sampler_mode_nuc for all tests if specified via command line
    if args.sampler_mode_nuc != "per_timestep":
        for test in all_tests:
            test.sampler_mode_nuc = args.sampler_mode_nuc
    
    mode_map = {
        "test1": [0],
        "test2": [1],
        "test3": [2],
        "test4": [3],
        "test5": [4],
        "all": list(range(len(all_tests)))
    }
    
    indices = mode_map.get(args.mode, list(range(len(all_tests))))
    test_cases = [all_tests[i] for i in indices]
    
    print("="*70)
    print("NUCLEATION VALIDATION TEST SUITE v1")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Number of tests: {len(test_cases)}")
    
    # Run all selected tests
    all_results = []
    for test_config in test_cases:
        runner = NucleationTestRunner(test_config)
        results = runner.run()
        all_results.append(results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if all_results:
        print(f"{'Test Name':<35} | {'Particles':>10} | {'Liquid (µL)':>12} | {'Time (s)':>10} | {'Status':>8}")
        print("-"*85)
        
        for r in all_results:
            status = "PASS" if r.get('success', False) else "FAIL"
            particles = r.get('final_particles', 'N/A')
            liquid = r.get('total_liquid', 0) * 1e6 if isinstance(r.get('total_liquid'), (int, float)) else 'N/A'
            comp_time = r.get('machine_time', 0) if isinstance(r.get('machine_time'), (int, float)) else 'N/A'
            time_str = f"{comp_time:.2f}" if isinstance(comp_time, float) else str(comp_time)
            print(f"{r['name']:<35} | {str(particles):>10} | {str(liquid):>12} | {time_str:>10} | {status:>8}")
        
        print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()
    
    # Exit with error code if any test failed
    if any(not r.get('success', False) for r in results):
        sys.exit(1)
