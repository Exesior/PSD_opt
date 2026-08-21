# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:42:58 2024

@author: Administrator

MC-PBE Package Exports:
-----------------------
- MCPBESolver: Standard solver with basic nucleation (slowest)
- MCPBESolverOpti: Optimized solver with cached sampler (~100-1000x faster)
- MCPBESolverTimestep: Dual-timestep sampler (FASTEST! Combines dual + timestep)

Note: MCPBESolverDual is DEPRECATED. Use MCPBESolverTimestep instead.
"""

from .mcpbe import (
    MCPBESolver,
    MCPBESolverOpti,
    MCPBESolverTimestep,
)

# Optional imports - warn but don't fail if not available
if MCPBESolverOpti is None:
    import warnings
    warnings.warn(
        "MCPBESolverOpti is not available. "
        "Falling back to standard nucleation. "
        "Check that mcpbe_nucleation_sampleopti.py exists.",
        ImportWarning,
    )

if MCPBESolverTimestep is None:
    import warnings
    warnings.warn(
        "MCPBESolverTimestep is not available. "
        "Falling back to standard nucleation. "
        "Check that mcpbe_nucleation_timestep.py exists.",
        ImportWarning,
    )

# Deprecated: MCPBESolverDual has been merged into MCPBESolverTimestep
MCPBESolverDual = None

# from .mcpbe_jit import MCPBESolver
