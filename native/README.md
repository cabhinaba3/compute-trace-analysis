# native/

The C++ behind `cta._native`. `pip install .` builds it (a hatch hook in
`hatch_build.py` runs CMake with nanobind) and puts `cta/_native*.so` next to
the Python code. Linux x86_64 only; a C++20 compiler plus zlib, SQLite,
Protobuf and simdjson development packages are needed.

Every `#include` names its file from `src/`, for example
`#include "common/sink.hpp"`. Sources are listed by area in `CMakeLists.txt`;
there is no globbing, so a new `.cpp` file has to be added there.

```
src/
  module.cpp              the NB_MODULE; calls the three bind_* functions
  bindings/               what Python sees: names, arguments and defaults
    synth.cpp               synthetic trace generators
    readers.cpp             trace readers
    core.cpp                the DAG solver and interval helpers
    numpy.hpp               hands std::vector to NumPy without a copy
  common/                 header-only helpers shared by the areas below
    sink.hpp                buffered file or gzip output
    format.hpp              numbers and strings formatted like json.dumps
    rng.hpp                 deterministic random numbers
    json_value.hpp          a small JSON value for the writers
    cpus.hpp                usable_cpus(): reads the affinity mask
    interner.hpp            distinct strings numbered by first appearance
    launch_map.hpp          correlation id -> launching CUDA runtime call
    sqlite.hpp              RAII wrappers for reading with SQLite
    ordered_pool.hpp        worker threads whose results are merged in task order
  synth/                  synthetic trace generators (kineto, goal, nsys,
                          chakra, training, profilers = ncu/jtop/trtexec)
  readers/chakra/         the Chakra .et reader (see below)
  readers/goal/           the GOAL text reader (see below)
  readers/nsys/           the Nsight Systems SQLite reader (see below)
  readers/kineto/         the Kineto trace reader (see below)
  core/                   dag (critical path and slack), dag_edges (the graph's
                          edge list), intervals (nesting_parents)
```

Changing a Python-visible name or default means changing `cta/_native.pyi`,
the `cta.synth` wrappers and the tests too.

## readers/chakra

One `.et` file is one task. `read_chakra_files` gives each file to an
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

`goal_text.*` holds the rules for one line of GOAL text: where a line ends
(`\n`, `\r\n` or a lone `\r`), how comments come off, and which lines are rank
headers, dependencies or ops. The rank-boundary scan and the parser both call
it, so they agree on every line. `goal_read.cpp` maps the file, walks it once
to find rank headers (a header reached while a `/* */` comment is open is not
one), cuts it into pieces of about 32 MiB at headers, and parses the pieces on
an `OrderedPool`. A piece starts in the state a rank header leaves behind, which
is why pieces do not depend on each other. Op names are returned as one buffer
of bytes with end offsets, because labels are unique per op; Python builds the
String column from that through Arrow.

## readers/nsys

`read_nsys_sqlite` opens the export once per part, on separate read-only
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
`common/ordered_pool.hpp`, shared with the Chakra and GOAL readers):

```
Source -> Splitter -> OrderedPool { ChunkParser x N threads } -> Merger
file/gzip   cuts the event      parse chunks in parallel     merge in file order,
blocks      array into chunks                                match launches to GPU
                                                             activities
```

| File | Role |
| --- | --- |
| `kineto_read.hpp/.cpp` | the public entry point, `read_kineto_columns` |
| `source.*` | reads a plain or gzip file in blocks |
| `splitter.*` | finds event boundaries while reading; keeps `distributedInfo` |
| `chunk.hpp` | what a chunk and its parsed columns look like |
| `chunk_parser.*` | simdjson parsing of one chunk; the only file that includes simdjson besides the CMake link |
| `merger.*` | renumbers names and threads in file order; resolves CUDA launches (`common/launch_map.hpp`) |

Two rules keep the output independent of thread count, chunk size and block
size: chunks are merged in file order, and ids are assigned at the merge, not
while parsing. `tests/unit/test_kineto_native_reader.py` checks this by cutting
the same file at many sizes, down to one byte.

The splitter's inner loop (`Splitter::feed` and `skip_event`) and the event
parser (`add_event` and the `read_*` helpers) each stay in one source file, so
the compiler can inline them.
