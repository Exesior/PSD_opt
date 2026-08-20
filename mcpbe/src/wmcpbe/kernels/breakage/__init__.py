"""
Breakage Kernels for WMCPBE Solver.

Available kernels:
    - power_law: Power-law breakage rate (BREAKRVAL=1,2,3,4)
    - powerlaw_rumpf: Rumpf theory with porosity/saturation strength
    - powerlaw_rumpf_dynamic: same, but driven by the mixer speed instead of a
                              shear rate (gas continuum)

Usage:
    >>> from wmcpbe.kernels.breakage import get_breakage_kernel
    >>> kernel = get_breakage_kernel('power_law', p1=3e-2, p2=1.0, g=1000)
    >>> rate = kernel.compute_rate(v_particle=1e-18)
"""

from typing import Type
from ..base import BreakageKernel, reject_unknown_params

# Import all kernel implementations
from .power_law import PowerLawBreakageKernel
from .powerlaw_rumpf import PowerLawRumpfBreakageKernel
from .powerlaw_rumpf_dynamic import PowerLawRumpfDynamicBreakageKernel

# Registry of available kernels
BREAK_KERNELS: dict[str, Type[BreakageKernel]] = {
    'power_law': PowerLawBreakageKernel,
    'powerlaw_rumpf': PowerLawRumpfBreakageKernel,
    'powerlaw_rumpf_dynamic': PowerLawRumpfDynamicBreakageKernel,
}


def get_breakage_kernel(name: str, **params) -> BreakageKernel:
    """
    Factory function to create breakage kernel by name.
    
    Args:
        name: Kernel name (see list_breakage_kernels())
        **params: Kernel-specific parameters
    
    Returns:
        Instantiated BreakageKernel
    
    Raises:
        ValueError: If the kernel name is not found, or if `params` contains a
                    name this kernel does not read (see `reject_unknown_params`)
    """
    if name not in BREAK_KERNELS:
        available = list(BREAK_KERNELS.keys())
        raise ValueError(
            f"Unknown breakage kernel '{name}'. "
            f"Available kernels: {available}"
        )
    kernel = BREAK_KERNELS[name](**params)
    reject_unknown_params(kernel, params)
    return kernel


def list_breakage_kernels() -> list[str]:
    """List available breakage kernels."""
    return list(BREAK_KERNELS.keys())


__all__ = [
    'get_breakage_kernel',
    'list_breakage_kernels',
    'PowerLawBreakageKernel',
    'PowerLawRumpfBreakageKernel',
    'PowerLawRumpfDynamicBreakageKernel',
]
