"""
Continuous Processes Kernels for WMCPBE Solver.

These kernels model time-continuous physical processes that occur
during the simulation via operator splitting (not tied to discrete events).

Available kernels:
    - porosity_compression: Exponential porosity decay under shear
    - porosity_compression_dynamik: as above, rate driven by the mixer speed
      (k = rate * n_mixer ** c_mixer)
    - porosity_compression_dynamik_rumpf: as above, additionally divided by the
      Rumpf granule strength sigma(eps, S)
    - liquid_internalization: Capillary-driven liquid internalization

Usage:
    >>> from wmcpbe.kernels.continuous_processes import get_continuous_kernel
    >>> kernel = get_continuous_kernel('porosity_compression', rate=0.02, min_porosity=0.3)
    >>> poro_new = kernel.compute(poro=0.5, dt=0.1)
    
    >>> kernel = get_continuous_kernel('liquid_internalization', k_int=1e12)
    >>> s_new = kernel.compute(saturation=0.3, v_pore=1e-18, l_intern=0, dt=0.1)
"""

from typing import Type
from ..base import (
    CompressionKernel,
    LiquidInternalizationKernel as InternalizationKernelBase,
    reject_unknown_params,
)

# Import all kernel implementations
from .porosity_compression import PorosityCompressionKernel
from .porosity_compression_dynamik import PorosityCompressionDynamikKernel
from .porosity_compression_dynamik_rumpf import (
    PorosityCompressionDynamikRumpfKernel,
)
from .liquid_internalization import LiquidInternalizationKernel
from .liq_internalisation_agglomeration import LiquidInternalisationAgglomerationKernel

# Registry of available kernels
CONTINUOUS_KERNELS: dict[str, Type] = {
    'porosity_compression': PorosityCompressionKernel,
    'porosity_compression_dynamik': PorosityCompressionDynamikKernel,
    'porosity_compression_dynamik_rumpf': PorosityCompressionDynamikRumpfKernel,
    'liquid_internalization': LiquidInternalizationKernel,
    'liq_internalisation_agglomeration': LiquidInternalisationAgglomerationKernel,
}


def get_continuous_kernel(name: str, **params):
    """
    Factory function to create continuous process kernel by name.
    
    Args:
        name: Kernel name (see list_continuous_kernels())
        **params: Kernel-specific parameters
    
    Returns:
        Instantiated ContinuousProcessKernel
    
    Raises:
        ValueError: If the kernel name is not found, or if `params` contains a
                    name this kernel does not read (see `reject_unknown_params`)
    """
    if name not in CONTINUOUS_KERNELS:
        available = list(CONTINUOUS_KERNELS.keys())
        raise ValueError(
            f"Unknown continuous process kernel '{name}'. "
            f"Available kernels: {available}"
        )
    kernel = CONTINUOUS_KERNELS[name](**params)
    reject_unknown_params(kernel, params)
    return kernel


def list_continuous_kernels() -> list[str]:
    """List available continuous process kernels."""
    return list(CONTINUOUS_KERNELS.keys())


__all__ = [
    'get_continuous_kernel',
    'list_continuous_kernels',
    'PorosityCompressionKernel',
    'PorosityCompressionDynamikKernel',
    'PorosityCompressionDynamikRumpfKernel',
    'LiquidInternalizationKernel',
]
