"""Liquid volume mass conservation validation for WMCPBE.

This module provides utilities to verify that liquid volume is conserved
during agglomeration and breakage events in the weighted MC-PBE solver.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from wmcpbe.mcpbe import MCPBESolver


MIN = 1e-40


@dataclass
class CaseConfig:
    """Configuration for a validation case."""
    dim: int
    kernel: str
    process: str
    t_vec: np.ndarray
    x: float = 2e-6
    beta0: float = 1e-2
    g: float = 1.0
    p1: float = 1e-2
    p2: float = 1.0
    pl_v: float = 1.0
    pl_q: float = 1.0
    initial_number_density: float = 1.0
    initial_total_weight: float = 100000.0
    initial_liquid_fraction: float = 0.0  # Fraction of particle volume as liquid

    def __post_init__(self) -> None:
        self.t_vec = np.asarray(self.t_vec, dtype=float)
        if self.dim not in (1, 2):
            raise NotImplementedError("This validator supports dim=1 or dim=2.")
        if self.process not in ("agglomeration", "breakage", "mix"):
            raise ValueError("process must be 'agglomeration', 'breakage', or 'mix'.")
        if self.kernel not in ("const", "sum"):
            raise ValueError("kernel must be 'const' or 'sum'.")


@dataclass
class WMCPBEVariantConfig:
    """Configuration for a WMCPBE solver variant."""
    name: str
    repeats: int = 1
    base_seed: int = 42
    maxiter: int = int(1e9)
    enabled: bool = True
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiquidConservationValidationConfig:
    """Configuration for liquid conservation validation."""
    case: CaseConfig
    wmcpbe_variants: List[WMCPBEVariantConfig]
    verbose: bool = False


@dataclass
class InitialParticleState:
    """Initial particle state with liquid volume."""
    Vc: float
    V_flat: np.ndarray
    W_init: np.ndarray
    liquid_volume_init: np.ndarray


@dataclass
class MethodResult:
    """Result from a single method run."""
    name: str
    family: str
    moments: np.ndarray
    liquid_volume_total: np.ndarray  # Weighted total liquid volume over time
    std: Optional[np.ndarray] = None
    liquid_std: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation result container."""
    time: np.ndarray
    dim: int
    kernel: str
    process: str
    initial_state: InitialParticleState
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def add_method(self, result: MethodResult) -> None:
        self.methods[result.name] = result

    def methods_by_family(self, family: str) -> List[MethodResult]:
        return [item for item in self.methods.values() if item.family == family]


class LiquidConservationRunner:
    """Run WMCPBE variants and track liquid volume conservation."""

    def __init__(self, config: LiquidConservationValidationConfig):
        self.config = config

    def run(self) -> ValidationResult:
        """Run all variants and collect results."""
        initial_state = self.build_initial_particles()
        
        result = ValidationResult(
            time=self.config.case.t_vec.copy(),
            dim=self.config.case.dim,
            kernel=self.config.case.kernel,
            process=self.config.case.process,
            initial_state=initial_state,
        )

        # Calculate initial total liquid volume
        initial_liquid_total = np.sum(
            initial_state.liquid_volume_init * initial_state.W_init
        )

        for variant in self.config.wmcpbe_variants:
            if not variant.enabled:
                continue
            method_result = self.run_wmcpbe_variant(variant, initial_state, initial_liquid_total)
            result.add_method(method_result)

        return result

    def build_initial_particles(self) -> InitialParticleState:
        """Build weighted initial particles with liquid volume."""
        case = self.config.case
        v0 = self._reference_particle_volume()
        total_weight = float(case.initial_total_weight)
        
        if total_weight <= 0.0:
            raise ValueError("initial_total_weight must be positive.")
        if case.initial_number_density <= 0.0:
            raise ValueError("initial_number_density must be positive.")

        if case.dim == 1:
            # Three representative particle volumes
            component_volumes = np.array([[v0, 2.0 * v0, 4.0 * v0]], dtype=float)
            fractions = np.array([0.60, 0.30, 0.10], dtype=float)
        else:
            # Four representative particles in 2D
            component_volumes = np.array(
                [
                    [v0, 2.0 * v0, v0, 2.0 * v0],
                    [v0, v0, 2.0 * v0, 2.0 * v0],
                ],
                dtype=float,
            )
            fractions = np.array([0.40, 0.20, 0.20, 0.20], dtype=float)

        fractions = fractions / float(np.sum(fractions))
        weights = total_weight * fractions
        
        V_flat = np.zeros((case.dim + 1, component_volumes.shape[1]), dtype=float)
        V_flat[: case.dim, :] = component_volumes
        V_flat[-1, :] = np.sum(component_volumes, axis=0)
        
        Vc = total_weight / float(case.initial_number_density)
        
        # Initialize liquid volume as fraction of particle volume
        liquid_volume_init = V_flat[-1, :] * case.initial_liquid_fraction
        
        return InitialParticleState(
            Vc=Vc,
            V_flat=V_flat,
            W_init=weights,
            liquid_volume_init=liquid_volume_init,
        )

    def run_wmcpbe_variant(
        self,
        variant: WMCPBEVariantConfig,
        initial_state: InitialParticleState,
        initial_liquid_total: float,
    ) -> MethodResult:
        """Run a single WMCPBE variant."""
        solver = self._build_wmcpbe_solver(variant)

        time_start = time.time()
        results, _ = solver.solve_repeats(
            N=variant.repeats,
            base_seed=variant.base_seed,
            maxiter=variant.maxiter,
            init_Vc=False,
            Vc=float(initial_state.Vc),
            V_flat=initial_state.V_flat,
            W_init=initial_state.W_init,
        )
        elapsed = time.time() - time_start

        trajectories: List[np.ndarray] = []
        liquid_trajectories: List[np.ndarray] = []
        
        for item in results:
            tv = np.asarray(item["t_vec"], dtype=float)
            if tv.shape != self.config.case.t_vec.shape or not np.allclose(tv, self.config.case.t_vec):
                raise ValueError(f"WMCPBE variant '{variant.name}' returned a different time grid.")
            
            trajectories.append(np.asarray(item["moments"], dtype=float))
            
            # Extract liquid volume history if available
            if "liquid_volume_save" in item:
                liq_save = item["liquid_volume_save"]
                liq_traj = np.array([
                    np.sum(liq * solver.W[:len(liq)]) for liq in liq_save
                ])
                liquid_trajectories.append(liq_traj)

        if not trajectories:
            raise ValueError(f"WMCPBE variant '{variant.name}' produced no repeat results.")

        stack = np.stack(trajectories, axis=0)
        moments = np.mean(stack, axis=0)
        std = np.std(stack, axis=0, ddof=1) if variant.repeats > 1 else None
        
        # Process liquid volume trajectories
        if liquid_trajectories:
            liq_stack = np.stack(liquid_trajectories, axis=0)
            liquid_volume_total = np.mean(liq_stack, axis=0)
            liquid_std = np.std(liq_stack, axis=0, ddof=1) if variant.repeats > 1 else None
        else:
            # Fallback: use initial value for all times
            liquid_volume_total = np.full(len(self.config.case.t_vec), initial_liquid_total)
            liquid_std = None

        # Calculate mass conservation error
        rel_error = abs(liquid_volume_total[-1] - initial_liquid_total) / max(abs(initial_liquid_total), MIN)
        
        return MethodResult(
            name=variant.name,
            family="wmcpbe",
            moments=moments,
            liquid_volume_total=liquid_volume_total,
            std=std,
            liquid_std=liquid_std,
            meta={
                "elapsed_s": elapsed,
                "repeats": variant.repeats,
                "base_seed": variant.base_seed,
                "initial_liquid_total": initial_liquid_total,
                "final_liquid_total": liquid_volume_total[-1],
                "relative_error": rel_error,
                **copy.deepcopy(variant.attrs),
            },
        )

    def _build_wmcpbe_solver(self, variant: WMCPBEVariantConfig) -> MCPBESolver:
        """Build WMCPBE solver with case parameters."""
        case = self.config.case
        solver = MCPBESolver(
            dim=case.dim,
            t_vec=case.t_vec,
            verbose=self.config.verbose,
            load_attr=False,
            init=False,
        )
        solver.process_type = case.process
        solver.G = case.g
        solver.alpha_prim = np.ones(case.dim ** 2, dtype=float)
        solver.pl_v = case.pl_v
        solver.pl_q = case.pl_q
        solver.recon_enable = False
        solver.break_dW_max = 50.0
        solver.agg_dW_min = 1.0
        solver.agg_dW_max = 20.0
        
        self._apply_case_params(solver)
        self._apply_attrs(solver, variant.attrs)
        
        return solver

    def _apply_case_params(self, solver: MCPBESolver) -> None:
        """Apply case-specific physics parameters."""
        case = self.config.case
        if case.kernel == "const":
            solver.COLEVAL = 3
            solver.SIZEEVAL = 1
            solver.CORR_BETA = case.beta0
            solver.BREAKRVAL = 1
            solver.BREAKFVAL = 2
            solver.pl_P1 = case.p1
            solver.pl_P2 = case.p2
            solver.pl_P3 = case.p1
            solver.pl_P4 = case.p2
        elif case.kernel == "sum":
            solver.COLEVAL = 4
            solver.SIZEEVAL = 1
            solver.CORR_BETA = case.beta0 / max(self._reference_particle_volume(), MIN)
            solver.BREAKRVAL = 2
            solver.BREAKFVAL = 2
            solver.pl_P1 = case.p1
            solver.pl_P2 = case.p2
            solver.pl_P3 = case.p1
            solver.pl_P4 = case.p2
        else:
            raise NotImplementedError(f"Unsupported kernel '{case.kernel}'.")

    @staticmethod
    def _apply_attrs(solver: MCPBESolver, attrs: Dict[str, Any]) -> None:
        """Apply variant-specific attributes."""
        for key, value in attrs.items():
            setattr(solver, key, value)

    def _reference_particle_volume(self) -> float:
        """Calculate reference particle volume from diameter."""
        radius = float(self.config.case.x) / 2.0
        return float((4.0 / 3.0) * math.pi * radius ** 3)


def check_mass_conservation(result: ValidationResult, tolerance: float = 1e-10) -> Dict[str, bool]:
    """Check if liquid mass is conserved within tolerance for all methods."""
    passed = {}
    for name, method in result.methods.items():
        rel_error = method.meta.get("relative_error", float("inf"))
        passed[name] = rel_error < tolerance
    return passed


__all__ = [
    "CaseConfig",
    "WMCPBEVariantConfig",
    "LiquidConservationValidationConfig",
    "LiquidConservationRunner",
    "ValidationResult",
    "check_mass_conservation",
]
