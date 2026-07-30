"""
Compression Kernels for WMCPBE Solver.

DEPRECATED: This module is deprecated in favor of continuous_processes.
Use wmcpbe.kernels.continuous_processes instead.

These kernels model porosity reduction over time due to mechanical compaction.
They are applied continuously by the CompressionHandler after each MC event.

Available kernels:
    - exponential_decay: Exponential porosity decay (existing behavior)
    - stress_compaction: Stress-dependent compaction (NEW)

Usage:
    >>> from wmcpbe.kernels.compression import get_compression_kernel
    >>> kernel = get_compression_kernel('exponential_decay', rate=0.02, min_porosity=0.3)
    >>> poro_new = kernel.compute_porosity_decay(poro=0.5, dt=0.1)
"""

import warnings
from typing import Type
from ..base import CompressionKernel

# Import all kernel implementations
from .exponential_decay import ExponentialDecayKernel
from .stress_compaction import StressCompactionKernel

# Registry of available kernels
COMPRESSION_KERNELS: dict[str, Type[CompressionKernel]] = {
    'exponential_decay': ExponentialDecayKernel,
    'stress_compaction': StressCompactionKernel,
}

# Deprecation warning
warnings.warn(
    "wmcpbe.kernels.compression is deprecated. "
    "Use wmcpbe.kernels.continuous_processes instead.",
    DeprecationWarning,
    stacklevel=2
)


def get_compression_kernel(name: str, **params) -> CompressionKernel:
    """
    Factory function to create compression kernel by name.
    
    Args:
        name: Kernel name (see list_compression_kernels())
        **params: Kernel-specific parameters
    
    Returns:
        Instantiated CompressionKernel
    
    Raises:
        ValueError: If kernel name is not found
    """
    if name not in COMPRESSION_KERNELS:
        available = list(COMPRESSION_KERNELS.keys())
        raise ValueError(
            f"Unknown compression kernel '{name}'. "
            f"Available kernels: {available}"
        )
    return COMPRESSION_KERNELS[name](**params)


def list_compression_kernels() -> list[str]:
    """List available compression kernels."""
    return list(COMPRESSION_KERNELS.keys())


__all__ = [
    'get_compression_kernel',
    'list_compression_kernels',
    'ExponentialDecayKernel',
    'StressCompactionKernel',
]
