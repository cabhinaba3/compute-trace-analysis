"""Single-device comparison aspects.

Importing this package registers every aspect it contains. Reports list
aspects in registration order, so the imports below keep the original
order (latency, kernels, timeline, OS runtime, roofline) instead of the
alphabetical one.
"""

# isort: off
from cta.aspects.device import latency  # noqa: F401
from cta.aspects.device import kernels  # noqa: F401
from cta.aspects.device import timeline  # noqa: F401
from cta.aspects.device import os_runtime  # noqa: F401
from cta.aspects.device import roofline  # noqa: F401
# isort: on
