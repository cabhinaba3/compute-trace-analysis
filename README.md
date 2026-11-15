# CTA: compute-trace-analysis

CTA analyses and compares performance traces from distributed AI/HPC runs, GPU profilers, inference engines and edge power telemetry. It reads the trace formats below, caches them as Parquet, and compares two or more runs (latency, kernels, compute/communication overlap, critical path, network APIs, energy).

Supported inputs: Chakra `.et`, GOAL `.goal`/`.bin`, Nsight Systems (`.sqlite`, `.nsys-rep`), PyTorch Kineto (`.json`, `.json.gz`), Nsight Compute (`.csv`, `.ncu-rep`), TensorRT `trtexec` JSON, Jetson `jtop` JSON, and JAX/XProf. Run `cta list` to see every registered format and comparison aspect.

## Install

Linux (x86_64) and Python 3.12 or newer. The install compiles a C++ extension, so it needs a C++20 compiler and the zlib, SQLite, Protobuf (with `protoc`) and simdjson development libraries.

```bash
git clone --recurse-submodules https://github.com/cabhinaba3/compute-trace-analysis.git
cd compute-trace-analysis
./configure.sh        # checks this machine and lists the easiest ways to get what is missing
pip install -e .
```

`configure.sh` changes nothing by itself. Without root, `conda env create -f environment.yml` (or mamba, micromamba) provides the compiler and every library; then `conda activate cta` and `pip install --no-build-isolation -e .`. If you cloned without `--recurse-submodules`, run `git submodule update --init` first.

The default install keeps to the packages the command line needs. Add what you use: `.[sql]` for SQL plots and `where` filters, `.[viz]` for interactive HTML plots, `.[chakra]` to write Chakra `.et` files from Python, `.[pandas]`, `.[all]` for these together, and `.[dev]` for contributing. A feature whose extra is missing says which one to install.

The build reads optional environment variables: `CTA_BUILD_JOBS`, `CTA_PREFIX_PATH` (libraries in a non-standard place), `CTA_GENERATOR`, `CTA_BUILD_TYPE`, `CTA_BUILD_DIR`, `CTA_CMAKE_ARGS` and `CTA_FRESH`. They are listed at the top of `hatch_build.py`.

## Use

Command line:

```bash
cta synth demo_traces                                  # write small synthetic traces to try things on
cta inspect demo_traces/gpu_bundle_fp16                # show what CTA found in a trace or directory
cta analyze demo_traces/gpu_bundle_fp16 --warmup-steps 2
cta compare baseline.json.gz candidate.json.gz --labels Baseline,Candidate --warmup-steps 1
cta compare a b --aspects network_api                  # pick aspects; `cta list` shows them
cta compare a b --gate "mean_step_ms:+5%"              # exit code 2 if a metric regresses by more than 5%
cta profile nsys --analyze -- python train.py          # run a vendor profiler with the flags CTA needs
```

Add `--out-dir DIR --export markdown,latex,csv,json,pdf` to write reports, and `--format json` to print one. `cta --help` and `cta <command> --help` list everything else.

Python:

```python
import cta

report = cta.compare("baseline.json.gz", "candidate.json.gz",
                     labels=["Baseline", "Candidate"], warmup_steps=1)
df = report.to_polars("kernel_diff", table="by_family")
report.plot(kind="tornado", path="kernel_tornado.pdf")
```

`cta.analyze(path)` takes a single trace. Plugins add formats and aspects with the decorators in `cta/core/registry.py`, or through the `cta.plugins` entry-point group.

## Contributing

Issues and pull requests are welcome at <https://github.com/cabhinaba3/compute-trace-analysis>.

Before opening a pull request, run the same checks as CI:

```bash
ruff check cta tests examples
mypy cta tests examples
pytest
```

Add a test for a change in behaviour and a line under "Unreleased" in `CHANGELOG.md`. Code under `native/` is built by `pip install`; rebuild with `pip install -e .` after changing it.

## License

GNU General Public License v2.0. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
