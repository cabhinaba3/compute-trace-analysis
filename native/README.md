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
    core.cpp                interval helpers
    numpy.hpp               hands std::vector to NumPy without a copy
  common/                 header-only helpers shared by the areas below
    sink.hpp                buffered file or gzip output
    format.hpp              numbers and strings formatted like json.dumps
    rng.hpp                 deterministic random numbers
    json_value.hpp          a small JSON value for the writers
    cpus.hpp                usable_cpus(): reads the affinity mask
  synth/                  synthetic trace generators (kineto, goal, nsys,
                          chakra, training, profilers = ncu/jtop/trtexec)
  readers/kineto/         the Kineto trace reader (see below)
  core/                   interval helpers (nesting_parents)
```

Changing a Python-visible name or default means changing `cta/_native.pyi`,
the `cta.synth` wrappers and the tests too.

## readers/kineto

A trace is read in three stages that run at the same time:

```
Source -> Splitter -> Pipeline { ChunkParser x N threads -> Merger }
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
| `pipeline.*` | worker threads, bounded queue, ordered merge thread |
| `merger.*`, `launch_map.hpp` | renumbers names and threads in file order; resolves CUDA launches |
| `interner.hpp` | distinct strings numbered by first appearance |

Two rules keep the output independent of thread count, chunk size and block
size: chunks are merged in file order, and ids are assigned at the merge, not
while parsing. `tests/unit/test_kineto_native_reader.py` checks this by cutting
the same file at many sizes, down to one byte.

The splitter's inner loop (`Splitter::feed` and `skip_event`) and the event
parser (`add_event` and the `read_*` helpers) each stay in one source file, so
the compiler can inline them.
