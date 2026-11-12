# CTA v1.0 — Multi-Trace & Multi-Aspect Performance Comparison Framework

**CTA (`compute-trace-analysis`)** is a cross-platform (macOS + Linux) performance comparison framework for distributed AI/HPC execution graphs, GPU profilers, inference engines, and edge power telemetry.

Originally developed for GPU/Jetson inference characterization (ISPASS'25), CTA v1.0 provides a unified **Content-Addressed Parquet Cache + Vectorized Polars/DuckDB/NumPy Engine** that compares **2-way (`Baseline` vs `Candidate`)** or **N-way** trace runs across latency, kernel shifts, compute/communication overlap, DAG critical paths, roofline efficiency, layer fusion, and system energy.

---

## Supported Trace Ecosystems

| Ecosystem | Supported Inputs | Key Extracted Signals |
| :--- | :--- | :--- |
| **MLCommons Chakra-ET** | Binary `.et` (`et_def.proto`), `chakra_host_trace.json` | Operator DAG, rank compute/comm nodes, collective types & byte volumes |
| **SPCL GOAL** | Text `.goal`, `txt2bin` binary `.bin` (`MAGIC_COOKIE = 4223` via `mmap`) | Per-rank `calc`/`send`/`recv` schedules, `requires`/`irequires` DAG edges |
| **NVIDIA Nsight Systems** | `.sqlite` (offline native SQL) and `.nsys-rep` (auto-exported if `nsys` present) | CUDA kernels, memcpy/memset, NVTX ranges, `OSRT_API`, modern & legacy `GPU_METRICS` |
| **PyTorch Kineto** | Chrome Trace `.json` and `.json.gz` | `ProfilerStep#*` slicing, CPU `aten::*` to GPU kernel correlation linking |
| **NVIDIA Nsight Compute** | Raw/Details `.csv` and `.ncu-rep` (auto-imported if `ncu` present) | SM/Tensor/DRAM utilization, FLOP/Byte arithmetic intensity, warp stall shares |
| **TensorRT (`trtexec`)** | `--exportTimes`, `--exportProfile`, `--exportLayerInfo` JSONs | $O(N \log N)$ H2D/Compute/D2H sweep-line decomposition, precision & layer fusion |
| **Jetson `jtop`** | `jtop` telemetry JSON | Multi-core CPU, GPU/EMC utilization, temperatures, multi-rail trapezoidal energy (J) |

---

## Installation

Requires **Python >= 3.11** on macOS or Linux.

```bash
git clone --recurse-submodules https://github.com/cabhinaba3/compute-trace-analysis.git
cd compute-trace-analysis
pip install -e ".[dev]"   # builds the pinned MLCommons Chakra submodule in third_party/chakra
```

If you cloned without `--recurse-submodules`, run `git submodule update --init` first.

---

## CLI Quickstart

### 1. Generate synthetic traces & analyze a single trace / bundle
```bash
cta synth demo_traces
cta inspect demo_traces/gpu_bundle_fp16

# Full single-trace analysis (critical path, overlap, hotspots, bottleneck regime, energy)
cta analyze demo_traces/gpu_bundle_fp16 --warmup-steps 2 --out-dir reports/single_fp16 --export markdown,latex,csv,pdf

# Group kernels by the outermost op/range (e.g. a record_function region) instead of the innermost op
cta analyze trace.json.gz --scope-depth 1

# Single-trace visualization (breakdown, topk hotspots, traffic, skew, roofline)
cta plot demo_traces/gpu_bundle_fp16 --kind breakdown,topk --format pdf --out-dir figures/single/
```

### 2. Compare two traces, `RunBundle` directories, or N-way sweeps
```bash
# Distributed Chakra-ET or GOAL schedule comparison
cta compare demo_traces/chakra_ring demo_traces/chakra_skewed --labels Balanced,Skewed

# Multi-artifact RunBundle directory comparison (Nsys + TensorRT + Jtop) with CI gates
cta compare demo_traces/gpu_bundle_fp16 demo_traces/gpu_bundle_int8 \
  --warmup-steps 2 \
  --gate "mean_step_ms:+5%,energy_per_query_mj:+5%" \
  --out-dir reports/jetson_int8 \
  --export markdown,latex,csv,json,pdf

# Declarative YAML/TOML sweep manifest
cta compare --manifest demo_traces/sweep.yaml
```

Gates fail the run (exit code 2) when a metric's `delta_pct` exceeds the threshold, and also when a gate names a metric the report does not contain (use `--allow-missing-gates` to ignore). Add `--format json` to `compare`/`analyze` to print the report as JSON on stdout (export status goes to stderr), e.g. `cta compare a b --format json | jq .violations`.

### 3. Verify Chakra-ET $\leftrightarrow$ GOAL schedule equivalence
```bash
cta equiv demo_traces/chakra_ring demo_traces/goal/ring.bin --collective-algo ring --rel-tol 0.02
```

### 4. Render publication-ready vector figures (Type-42 PDF / SVG / PNG)
```bash
cta plot demo_traces/chakra_ring demo_traces/chakra_skewed \
  --kind breakdown,skew,traffic_delta \
  --style paper_single_col \
  --format pdf \
  --out-dir figures/
```

Supported plot kinds: `breakdown`, `topk` (alias `tornado`), `traffic_delta`, `skew`, `roofline`, `scaling`.

### 5. Collect profiles with the vendor tools (`cta profile`)
```bash
cta profile nsys    --analyze -- python train.py                 # Nsight Systems: .nsys-rep + .sqlite export
cta profile ncu     --launch-count 50 -- python train.py         # Nsight Compute (needs counter access): .ncu-rep
cta profile kineto  --analyze -- python train.py                 # torch.profiler Chrome trace; each optimizer.step() = a ProfilerStep
cta profile trtexec -- --loadEngine=model.plan --iterations=200  # TensorRT: --exportTimes/Profile/LayerInfo are added
cta profile jtop    -- python infer.py                           # Jetson power & utilisation sampled while it runs
```
Each command writes `profiles/<name>/` (override with `-o`/`--name`), which `cta analyze profiles/<name>` reads. The tools do the profiling; CTA only runs them with the flags its adapters need (`nsys`/`ncu` must be on `PATH`, or set `NSYS_PATH`/`NCU_PATH`; `trtexec` via `TRTEXEC_PATH`; `jtop` needs `sudo pip install jetson-stats`). `nsight-python` (the `[profile]` extra, Linux only) is for annotating kernels inside your own Python code; `cta profile ncu` reuses its `ncu` flag defaults (flush caches, do not lock clocks) but works on any command.

---

## Python SDK

```python
import cta

# 0. Analyze & plot a single trace or RunBundle directory
single = cta.analyze("demo_traces/kineto/eager.json.gz", label="Eager_FP16", warmup_steps=1)
single.plot(kind="topk", path="figures/eager_hotspots.pdf")

# 1. Compare traces, globs, or multi-artifact RunBundle directories
report = cta.compare(
    "demo_traces/kineto/eager.json.gz",
    "demo_traces/kineto/compiled_fp8.json.gz",
    labels=["Eager_FP16", "Compiled_FP8"],
    warmup_steps=1,
    gates={"mean_step_ms": 5.0},
)

# 2. Access Polars / Pandas tables & automated bottleneck-shift diagnosis
df_kernels = report.to_polars("kernel_diff", table="by_family")
for shift in report.diagnoses:
    print(shift.summary)

# 3. Export LaTeX booktabs tables, Markdown, CSV/JSON, and vector plots
report.to_latex("tables/kineto_diff.tex")
report.plot(kind="tornado", path="figures/kernel_tornado.pdf")

# 4. Validate Chakra-ET to GOAL lowering fidelity
verdict = cta.equiv("demo_traces/chakra_ring", "demo_traces/goal/ring.goal", collective_algo="ring")
assert verdict.equivalent
```

---

## Extending CTA with Custom Plugins

Third-party formats and comparison aspects register via decorators in [`cta/core/registry.py`](cta/core/registry.py), either in your own module or through the `cta.plugins` entry-point group. Run `cta list` to see what is registered.

```python
from pathlib import Path

import polars as pl
from cta import AspectResult, CompareConfig, TraceArtifact, TrialGroup, register_adapter, register_aspect


@register_adapter("custom_log")
class CustomAdapter:
    def detect(self, path: Path) -> bool:
        return path.suffix == ".clog"

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        # Write each normalized table as Parquet and return {table_name: path}.
        dst = out_dir / "steps.parquet"
        pl.read_csv(artifact.paths[0]).write_parquet(dst)
        return {"steps": dst}


@register_aspect("custom_metric", title="Custom Metric", required_formats=("custom_log",))
class CustomMetric:
    def run(
        self, groups: tuple[TrialGroup, ...], tables: dict[str, list[dict[str, Path]]], cfg: CompareConfig
    ) -> AspectResult:
        # tables[group_id][trial_index]["custom_log.steps"] -> Parquet path
        ...
```

---

## Repository Layout

```text
cta/
  core/          Canonical schemas, BLAKE2b Parquet cache, CSR DAG & sweep-line engines, stats, diagnosis
  adapters/      Format readers: chakra/, goal/, nsys/, kineto/, ncu/, tensorrt/, jtop/
  aspects/       Comparison aspects: distributed.py, device.py, edge.py
  exporters/     Rich terminal, Markdown, LaTeX booktabs, CSV/JSON bundle exporters
  viz/           Okabe-Ito Type-42 Matplotlib publication figure suite
  synth/         Deterministic synthetic trace generators for all 7 ecosystems
  profile/       Thin runners for nsys, ncu, torch.profiler, trtexec and jtop (`cta profile`)
  cli/           Typer CLI (`cta compare`, `analyze`, `profile`, `equiv`, `plot`, `inspect`, `list`, `synth`, `cache`)
examples/        Runnable SDK quickstarts and declarative sweep YAML manifest
tests/           Unit and end-to-end integration test suites
```

## License

GNU General Public License v2.0 (GPL-2.0). See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
