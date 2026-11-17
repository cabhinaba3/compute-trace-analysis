"""Deterministic synthetic trace generators.

Each module writes valid files in one vendor format. The writers are C++ (the
``cta._native`` extension, built with the package), so they also scale to
multi-gigabyte traces. The test suite, ``cta synth`` and the SoftwareX
quickstart examples use them.
"""
