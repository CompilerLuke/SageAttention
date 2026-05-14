"""
Compatibility import alias for the Blackwell SageAttention 4 package.

The package built by this directory is named ``sageattn4``.  Some callers use
``sageattention4`` by analogy with the top-level SageAttention package, so keep
that import spelling working as a thin re-export.
"""

from sageattn4 import sageattn4_blackwell

__all__ = ["sageattn4_blackwell"]
