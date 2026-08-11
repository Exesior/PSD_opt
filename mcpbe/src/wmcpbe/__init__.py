# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:42:58 2024

@author: Administrator
"""

from .mcpbe import MCPBESolver

# Alias for backward compatibility
WMCPBE = MCPBESolver

__all__ = ["MCPBESolver", "WMCPBE"]

# Note: CompressionHandler and CompressionConfig have been removed.
# Use ContinuousProcessesHandler and ContinuousProcessesConfig instead.
