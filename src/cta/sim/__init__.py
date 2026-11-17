"""Discrete-event simulation of GOAL and Chakra execution graphs.

The simulator replays a trace's graph (every rank's compute, sends, receives
and collectives with their dependencies) on a model of the machine and reports
when each op ran, where each rank's time went and what the critical path was
made of. GOAL and Chakra graphs run on the same engine, so two schedules of
either format, or a Chakra trace and the GOAL schedule converted from it, can
be compared op by op.

Modules:
    config: SimConfig and how files, presets and overrides build one.
    presets: Named machine descriptions (LogGOPSim's defaults, InfiniBand,
        RoCE, NVLink clusters).
    graph: Builds a SimGraph from ingested GOAL or Chakra tables.
    compute: Turns recorded compute durations into simulated ones (roofline,
        scaling, stragglers, jitter).
    engine: Runs a graph: ``simulate`` and ``run_graph``.
    result: SimResult and its tables.
    alignment: Matches the compute ops of two runs and cuts them into phases.
    compare: Compares runs: overview, ranks, phases, ops, structure, traffic,
        and an exact attribution of makespan gaps to critical-path categories.
    sweep: Parameter sweeps and Monte Carlo noise studies.
    collectives: Collective algorithms: schedules and analytical costs.
    trace_export: Simulated timelines as Chrome/Perfetto trace JSON.

The engine is C++ (``native/src/sim``); see its ``engine.hpp`` for the exact
semantics of every network model.
"""
