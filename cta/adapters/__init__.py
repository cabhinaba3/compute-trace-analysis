"""Trace format adapters.

Importing this package registers every built-in adapter with the registry.
"""

from cta.adapters import chakra as _chakra  # noqa: F401
from cta.adapters import goal as _goal  # noqa: F401
from cta.adapters import jax as _jax  # noqa: F401
from cta.adapters import jtop as _jtop  # noqa: F401
from cta.adapters import kineto as _kineto  # noqa: F401
from cta.adapters import ncu as _ncu  # noqa: F401
from cta.adapters import nsys as _nsys  # noqa: F401
from cta.adapters import tensorrt as _tensorrt  # noqa: F401
