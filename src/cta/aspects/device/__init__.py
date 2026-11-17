"""Single-device comparison aspects.

Importing this package registers every aspect it contains. Reports list
aspects in registration order, so the imports below keep the original
order (latency, kernels, timeline, OS runtime, roofline) instead of the
alphabetical one.
"""

# isort: off
from cta.aspects.device import latency as latency
from cta.aspects.device import kernels as kernels
from cta.aspects.device import timeline as timeline
from cta.aspects.device import os_runtime as os_runtime
from cta.aspects.device import roofline as roofline
# isort: on
