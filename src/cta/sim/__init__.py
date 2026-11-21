"""Discrete-event simulation of GOAL and Chakra execution graphs.

Both formats run on one engine (``native/src/sim``; ``engine.hpp`` states its
semantics), so they can be compared with each other op by op: ``config``
(settings), ``graph`` (traces to graphs), ``engine`` (runs), ``result``,
``compare`` and ``alignment`` (comparisons), ``collectives``, ``sweep``,
``trace_export`` and ``goal_export``.
"""
