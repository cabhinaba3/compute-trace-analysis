# native/

The C++ behind `cta._native`. `pip install .` builds it (a hatch hook in
`hatch_build.py` runs CMake with nanobind) and puts `src/cta/_native*.so` next
to the Python code. Linux x86_64 and macOS arm64; a C++20 compiler plus zlib, SQLite,
Protobuf and simdjson development packages are needed.

Every `#include` names its file from `src/`, for example
`#include "util/sink.hpp"`. Each directory under `src/` lists its sources in
its own `CMakeLists.txt` (there is no globbing), so a new `.cpp` file is added
there. `cmake/ChakraProto.cmake` generates the C++ for Chakra's protobuf
schema from the submodule.

```
CMakeLists.txt            dependencies and the one module target
cmake/ChakraProto.cmake   protobuf, and C++ generated from Chakra's schema
src/
  module.cpp              the NB_MODULE; calls the four bind_* functions
  bindings/               what Python sees: names, arguments and defaults
    readers.cpp             trace readers
    synth.cpp               synthetic trace generators
    analysis.cpp            the DAG solver and interval helpers
    sim.cpp                 the simulator: simulate, lower_collectives, schedules
    convert.hpp             column structs to dicts of NumPy arrays, no copies
  util/                   header-only helpers shared by the areas below
    sink.hpp                buffered file or gzip output
    format.hpp              numbers and strings formatted like json.dumps
    rng.hpp                 deterministic random numbers
    json_value.hpp          a small JSON value for the writers
    cpus.hpp                usable_cpus(): reads the affinity mask
    interner.hpp            distinct strings numbered by first appearance
    launch_map.hpp          correlation id -> launching CUDA runtime call
    sqlite.hpp              RAII wrappers for reading with SQLite
    ordered_pool.hpp        worker threads whose results are merged in task order
  analysis/               namespace cta::analysis: dag (critical path and
                          slack), dag_edges (the graph's edge list), intervals
                          (nesting_parents), whatif (in-memory DagPlan)
  streaming/              namespace cta::streaming: bounded-RAM streaming graph
                          sources, quantile sketches, .cta_graph block cache,
                          Scalasca delay attribution, and pluggable DAG/what-if
                          engines (see below)
  readers/<format>/       namespace cta::readers: one reader per format, with
                          reader.hpp declaring read_<format>() and its columns
  sim/                    namespace cta::sim: the simulator. graph (ops, edges,
                          groups), collectives (algorithm schedules and their
                          contention-free cost), lowering, topology and flow
                          (max-min fair link sharing), noise, engine (the event
                          loop and critical path); engine.hpp states the
                          semantics of every network model
  synth/                  namespace cta::synth: one generator per format
                          (kineto, goal, nsys, chakra, ncu, jtop, trtexec) and
                          the 3D-parallel training graphs
```

The helpers in `util/` stay in namespace `cta`, so code in the nested
namespaces uses them unqualified.

A reader returns a struct of columns. The struct lists its columns once, in
`for_each_column(visit)`, under the names Python sees; `bindings/convert.hpp`
turns any such struct into a dict (vectors become NumPy arrays without a copy,
string tables lists, nested structs dicts). Adding a column means adding the
member and one `visit` line.

Changing a Python-visible name or default means changing `src/cta/_native.pyi`,
the `cta.synth` wrappers and the tests too.

## readers/chakra

One `.et` file is one task. `read_chakra` gives each file to an
`OrderedPool` worker, which streams the length-prefixed protobuf messages
(`ParseDelimitedFromZeroCopyStream`, one `Node` object reused with `Clear()`)
into columns, and the merge step appends the files in the order given. Strings
(`name`, `comm_type`) are numbered per file and renumbered at the merge, so
the codes follow first appearance and do not depend on the thread count.

Attribute handling follows what the Python reader did: a later attribute with
the same name wins, a present `comm_size` hides `tensor_size`, an integer
`comm_type` names its collective (`COLL_<n>` if unknown), and a send or receive
defaults its own side to the file's rank. The rank of each file is worked out
in Python from the file name and passed in.

## readers/goal

`syntax.*` holds the rules for one line of GOAL text: where a line ends
(`\n`, `\r\n` or a lone `\r`), how comments come off, and which lines are rank
headers, dependencies or ops. The rank-boundary scan and the parser both call
it, so they agree on every line. `reader.cpp` maps the file, walks it once
to find rank headers (a header reached while a `/* */` comment is open is not
one), cuts it into pieces of about 32 MiB at headers, and parses the pieces on
an `OrderedPool`. A piece starts in the state a rank header leaves behind, which
is why pieces do not depend on each other. Op names are returned as one buffer
of bytes with end offsets, because labels are unique per op; Python builds the
String column from that through Arrow.

## readers/nsys

`read_nsys` opens the export once per part, on separate read-only
connections, and reads the GPU activities (kernels, then memcpys, then
memsets), the CUDA runtime launches, the NVTX ranges, the GPU metrics and the
two API summaries at the same time. It builds the same SQL as the Python reader
did, from the columns the export has, and resolves each activity's launch (the
first runtime row of a correlation id) before returning. Names and NVTX texts
come back dictionary-coded. The step and scope of each activity are worked out
in Python from these columns (`cta.adapters.nsys.reader`), with NumPy and the
interval helpers rather than a loop over events.

## readers/kineto

A trace is read in three stages that run at the same time (the worker pool is
`util/ordered_pool.hpp`, shared with the Chakra and GOAL readers):

```
Source -> Splitter -> OrderedPool { ChunkParser x N threads } -> Merger
file/gzip   cuts the event      parse chunks in parallel     merge in file order,
blocks      array into chunks                                match launches to GPU
                                                             activities
```

| File | Role |
| --- | --- |
| `reader.hpp/.cpp` | the public entry point, `read_kineto` |
| `source.*` | reads a plain or gzip file in blocks |
| `splitter.*` | finds event boundaries while reading; keeps `distributedInfo` |
| `chunk.hpp` | what a chunk and its parsed columns look like |
| `chunk_parser.*` | simdjson parsing of one chunk; the only file that includes simdjson besides the CMake link |
| `merger.*` | renumbers names and threads in file order; resolves CUDA launches (`util/launch_map.hpp`) |

Two rules keep the output independent of thread count, chunk size and block
size: chunks are merged in file order, and ids are assigned at the merge, not
while parsing. `tests/unit/native/test_kineto_reader.py` checks this by cutting
the same file at many sizes, down to one byte.

## streaming/

Chunked, bounded-memory execution-graph streaming and out-of-core solvers for GOAL (`.bin`/`.goal`) and Chakra-ET (`.et`/host `.json`):

```
streaming/
  batch.hpp, scratch.hpp          canonical SoA NodeBatch/EdgeBatch, StreamProvenance, RAII ScratchDir
  sources/                        chunked GraphSource readers (goal_bin, goal_text, chakra_et, chakra_host)
  sketches/                       pluggable Hybrid Exact-First -> DDSketch, KLL, Greenwald-Khanna quantiles
  cache/graph_cache.{hpp,cpp}     .cta_graph 2D GridGraph topological block cache (64B TopoVertexMeta POD)
  attribution/scalasca_delay.hpp  Scalasca backward wait-state & delay root-cause accumulator (IPDPS '12)
  engines/
    engine.hpp, engine_registry.cpp  pluggable GraphEngine interface & factory
    online_agg/                      Engine A: 1-pass O(P^2) aggregates + sketches
    frontier/                        Engine B: 1-pass Kahn frontier + Certified Eviction-Epoch intervals
    gridgraph/                       Engine C default: GridGraph (USENIX ATC '15) 2-pass exact DAG + Scalasca
    graphchi/                        pluggable GraphChi PSW (OSDI '12) out-of-core DAG solver
    xstream/                         pluggable X-Stream Scatter-Gather (SOSP '13) out-of-core DAG solver
    whatif/                          Engine D: block-streaming multi-lane what-if + 1-byte winner log
```

All streaming DAG, cache, and what-if translation units compile with `-ffp-contract=off` (applied via `TARGET_DIRECTORY _native`) and use Neumaier compensated summation (`util/compensated_sum.hpp`) for cross-CPU determinism and bit-exact parity with `analysis/dag.cpp` and `analysis/whatif.cpp`.

The splitter's inner loop (`Splitter::feed` and `skip_event`) and the event
parser (`add_event` and the `read_*` helpers) each stay in one source file, so
the compiler can inline them.

## sim

`simulate()` runs one graph of all ranks with an event queue ordered by
(time, event type, id), so a run depends only on its ops and parameters.
CPUs and NICs are resources with a free-at time and a queue of waiting
activities (an op starting, or a receiver processing a message); the one that
has waited longest gets a freed resource. Messages match per (receiver,
sender, tag) in FIFO order, as in LogGOPSim. `SimParams` lists its settings
once, under the dotted names of `cta.sim.config`, and the binding parses
them generically.

`build_schedule()` gives one member's actions of a collective. Lowering
inlines them between an entry and an exit NOP that carry the collective's
edges; `evaluate_collective()` runs all members' schedules as a
contention-free longest path for the analytical level.

Topologies only declare their links; routes are shortest paths, with ties
broken by a hash of (src, dst). `FlowNetwork` re-shares bandwidth by
progressive filling among the flows connected to the one that started or
ended. The engine records each message's split points as it runs, and the
critical path walks back from the last op through them, cutting time into
Segment categories that tile [0, makespan].
