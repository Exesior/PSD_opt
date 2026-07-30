"""Check Python paths and pbe_core availability."""

import sys
import os

print("=" * 70)
print("PYTHON PATHS")
print("=" * 70)
for path in sys.path:
    print(f"  {path}")

print("\n" + "=" * 70)
print("TRYING TO IMPORT pbe_core")
print("=" * 70)

try:
    import pbe_core
    print(f"  SUCCESS: pbe_core found at: {pbe_core.__file__}")
except ImportError as e:
    print(f"  FAILED: {e}")
    
print("\n" + "=" * 70)
print("TRYING TO IMPORT wmcpbe")
print("=" * 70)

# Add src to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"  Added to path: {src_dir}")

try:
    from wmcpbe.mcpbe import MCPBESolver
    print(f"  SUCCESS: wmcpbe.mcpbe imported")
except ImportError as e:
    print(f"  FAILED: {e}")
