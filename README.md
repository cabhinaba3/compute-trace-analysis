# CTA v1.0 — Multi-Trace & Multi-Aspect Performance Comparison Framework

**CTA (`compute-trace-analysis`)** is a Linux performance comparison framework for distributed AI/HPC execution graphs, GPU profilers, inference engines, and edge power telemetry.

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

Requires **Python >= 3.12** on **Linux** (x86_64). macOS is no longer supported. The package builds a C++ extension (`cta._native`, the synthetic trace generators) as part of `pip install`, so it needs a C++20 compiler plus the zlib, SQLite and Protobuf development packages (Debian/Ubuntu: `apt install g++ zlib1g-dev libsqlite3-dev libprotobuf-dev protobuf-compiler`). CMake, Ninja and nanobind are fetched by pip.

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
cta verify-equivalence demo_traces/chakra_ring demo_traces/goal/ring.bin --collective-algorithm ring --rel-tol 0.02
```

### 4. Plots: static figures, interactive HTML, one dashboard
```bash
cta plot --list-kinds                                   # every kind, what data it needs, its options
cta plot a.json b.json --kind breakdown,topk -f pdf     # Type-42 PDF / SVG / PNG for papers
cta plot a.json b.json --html report.html               # ONE interactive dashboard (needs `pip install cta[viz]`)
cta plot run/ --kind step_trace,kernel_drift -O steps=1:20 -O rolling=7
cta plot run/ --kind all -f png                         # every plot the data supports
cta plot --kind history --history 'reports/*/report.json' --metric mean_step_ms   # a metric across saved runs
```
**Kinds, by use case.** Overview: `breakdown`, `family_share`. Timeline: `timeline` (GPU Gantt), `overlap_timeline`, `stream_util`, `pareto`. Steps and drift: `step_trace` (rolling median, Theil-Sen trend, outliers), `step_dist`, `kernel_drift`, `history` (control chart over saved `report.json` files). A/B: `topk` (alias `tornado`), `waterfall`, `kernel_scatter`, `step_compare`, `scorecard`, `gates`. Sweeps: `scaling`, `sweep`. Distributed: `rank_heatmap`, `straggler`, `skew`, `traffic_delta`. Kernels: `roofline`, `occupancy`, `ncu_util`, `stalls`. Edge/energy: `power_trace`, `gpu_metrics_trace`, `energy_pareto`. Custom: `custom`. A kind that does not apply to your data says why instead of drawing an empty chart.

**Dashboard filters.** `cta plot TRACE --serve` opens the dashboard *live* on a loopback server (stdlib only; `--port`, `--no-open`): each change of group, rank, GPU device or step window re-runs the analysis for that slice and redraws every plot, so all aggregates are exact (a device filter halves a 2-GPU breakdown, a step window changes `total_steps`). A saved `--html` file is static and offline, so its filters only hide or restyle what is drawn: group (also bars/heatmap columns on a category axis), rank, and a step-range zoom; use `--serve` or `--devices/--ranks/--warmup-steps/--max-steps` when you need recomputed numbers. There is no cross-plot linking yet.

The same slicing is available for any analysis: `cta analyze run/ --devices 0` and `--ranks 0,1` (manifest: `ranks:`/`devices:` under `config`, SDK: `devices=[0]`).

**Your own plots.** Raw tables (`kernels`, `steps`, `gpu_metrics`, `runtime`, `ncu_kernels`, `ncu_findings`, `jtop`, `osrt`; `cta plot TRACE --list-sources` shows columns) are available besides the report tables. Times are in ms relative to each rank's first step, and `--warmup-steps/--max-steps` apply exactly as in the analysis. Draw them with a few fields or with DuckDB SQL:
```bash
cta plot run/ --sql "SELECT idx, rank, sum(duration_ms) AS ms FROM kernels WHERE category='comm' GROUP BY ALL" \
  --mark line --x idx --y ms --color rank
```
or declaratively, in a file (`cta plot run/ --spec plots.yaml`) or a manifest `plots:` block:
```yaml
plots:
  - {kind: step_trace, rolling: 7, steps: "2:50", theme: dark}
  - {kind: custom, name: comm_by_step, source: kernels, where: "category = 'comm'", mark: bar, x: idx, y: duration_ms, agg: sum}
```
Every kind declares typed options (shown by `--list-kinds`); unknown options are an error with a suggestion. Common ones: `title`, `theme` (light/dark), `palette`, `width`/`height`, `labels`, `rank`, `device`, `steps`, `where`. In Python: `report.plot("step_trace", path="x.png", rolling=7)`, `report.dashboard("report.html")`, and `cta.viz.register_plot` adds a kind that both backends and the dashboard pick up.

### 5. Collect profiles with the vendor tools (`cta profile`)
```bash
cta profile nsys    --analyze -- python train.py                 # Nsight Systems: .nsys-rep + .sqlite export
cta profile ncu     --launch-count 50 -- python train.py         # Nsight Compute (needs counter access): .ncu-rep
cta profile kineto  --analyze -- python train.py                 # torch.profiler Chrome trace; each optimizer.step() = a ProfilerStep
cta profile jax     --analyze -- python train.py                 # jax.profiler (XProf) log dir; mark steps with StepTraceAnnotation
cta profile trtexec -- --loadEngine=model.plan --iterations=200  # TensorRT: --exportTimes/Profile/LayerInfo are added
cta profile jtop    -- python infer.py                           # Jetson power & utilisation sampled while it runs
```
Each command writes `profiles/<name>/` (override with `-o`/`--name`), which `cta analyze profiles/<name>` reads. The tools do the profiling; CTA only runs them with the flags its adapters need (`nsys`/`ncu` must be on `PATH`, or set `NSYS_PATH`/`NCU_PATH`; `trtexec` via `TRTEXEC_PATH`; `jtop` needs `sudo pip install jetson-stats`). JAX `*.xplane.pb` files are converted with `xprof` (`pip install cta[jax]`); the Chrome-format `trace.json.gz` next to them needs nothing extra. `nsight-python` (the `[profile]` extra, Linux only) is for annotating kernels inside your own Python code; `cta profile ncu` reuses its `ncu` flag defaults (flush caches, do not lock clocks) but works on any command.

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
verdict = cta.verify_chakra_goal_equivalence(
    "demo_traces/chakra_ring",
    "demo_traces/goal/ring.goal",
    collective_algorithm="ring",
)
assert verdict.equivalent
```

---

## Extending CTA with Custom Plugins

Third-party formats and comparison aspects register via decorators in [`cta/core/registry.py`](cta/core/registry.py), either in your own module or through the `cta.plugins` entry-point group. Run `cta list` to see what is registered.

```python
import pathlib

import polars as pl

import cta


@cta.register_adapter("custom_log")
class CustomAdapter:
    def detect(self, path: pathlib.Path) -> bool:
        return path.suffix == ".clog"

    def ingest(
        self, artifact: cta.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        # Write each normalized table as Parquet and return {table_name: path}.
        steps_path = out_dir / "steps.parquet"
        pl.read_csv(artifact.paths[0]).write_parquet(steps_path)
        return {"steps": steps_path}


@cta.register_aspect(
    "custom_metric", title="Custom Metric", required_formats=("custom_log",)
)
class CustomMetric:
    def run(
        self,
        groups: tuple[cta.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: cta.CompareConfig,
    ) -> cta.AspectResult:
        # tables[group_id][trial_index]["custom_log.steps"] -> Parquet path
        ...
```

---

## Repository Layout

```text
cta/
  core/          Canonical schemas, BLAKE2b Parquet cache, CSR DAG & sweep-line engines, stats, diagnosis
  adapters/      Format readers: chakra/, goal/, nsys/, kineto/, jax/, ncu/, tensorrt/, jtop/
  aspects/       Comparison aspects: device/ (latency, kernels, timeline, os_runtime, roofline), distributed.py, edge.py, hta.py
  exporters/     Rich terminal, Markdown, LaTeX booktabs, CSV/JSON bundle exporters
  viz/           Plot kinds (registry + typed options), raw-table/SQL data layer, Matplotlib and Plotly renderers, dashboard
  synth/         Deterministic synthetic trace generators, one module per ecosystem (chakra, goal, nsys, kineto, ncu, tensorrt, jtop)
  profile/       Thin runners for nsys, ncu, torch.profiler, jax.profiler, trtexec and jtop (`cta profile`)
  cli/           Typer CLI: app.py wires the command modules (`compare`, `analyze`, `verify-equivalence`, `plot`, `synth`, `inspect`, `list`, `profile`, `cache`)
examples/        Runnable SDK quickstarts and declarative sweep YAML manifest
tests/           Unit and end-to-end integration test suites
```

## License

GNU General Public License v2.0 (GPL-2.0). See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
