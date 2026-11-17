"""Trace format adapters.

Importing this package registers every built-in adapter with the registry.
"""

from cta.adapters import chakra as chakra
from cta.adapters import goal as goal
from cta.adapters import jax as jax
from cta.adapters import jtop as jtop
from cta.adapters import kineto as kineto
from cta.adapters import ncu as ncu
from cta.adapters import nsys as nsys
from cta.adapters import trtexec as trtexec
