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

Requires **Python >= 3.10** on macOS or Linux.

```bash
git clone https://github.com/cabhinaba3/compute-trace-analysis.git
cd compute-trace-analysis
pip install -e ".[dev]"
```

---

## CLI Quickstart

### 1. Generate synthetic traces across all 7 ecosystems
```bash
cta synth demo_traces
cta inspect demo_traces/gpu_bundle_fp16
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

Supported plot kinds: `breakdown`, `tornado`, `traffic_delta`, `skew`, `roofline`, `scaling`.

---

## Python SDK

```python
import cta

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

Third-party formats and comparison aspects register cleanly via decorators in [`cta/core/registry.py`](cta/core/registry.py):

```python
import polars as pl
from cta import AspectResult, CompareConfig, TraceArtifact, TrialGroup, register_adapter, register_aspect


@register_adapter("custom_log", extensions=(".clog",))
class CustomAdapter:
    name = "custom_log"

    def can_sniffer(self, path):
        return path.suffix == ".clog"

    def ingest(self, artifact: TraceArtifact) -> dict[str, pl.DataFrame]:
        return {"steps": pl.read_csv(artifact.paths[0])}


@register_aspect("custom_metric", required_tables=("steps",))
def compute_custom_metric(groups: tuple[TrialGroup, ...], tables, cfg: CompareConfig) -> AspectResult:
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
  cli/           Typer CLI (`cta compare`, `equiv`, `plot`, `inspect`, `synth`, `cache`)
examples/        Runnable SDK quickstarts and declarative sweep YAML manifest
tests/           Unit and end-to-end integration test suites
```

## License

GNU General Public License v2.0 (GPL-2.0). See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
