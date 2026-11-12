# Changelog

## Unreleased

### Fixed
- `--gate` with a metric missing from the report now fails (exit 2) instead of passing silently; opt out with `--allow-missing-gates`. Malformed gate specs give a CLI error rather than a traceback.
- `--export pdf|svg|png` writes every applicable plot (`breakdown`, `topk`, `traffic_delta`, `skew`, `roofline`, `scaling`); keys in the returned dict are now `"<ext>:<kind>"`.
- `report.json` now also contains `details` and `scalars`.
- `compare --manifest` honors `--warmup-steps`, `--max-steps`, `--collective-algo`, `--alpha` and `--cache-dir` as overrides, and warns when positional traces are ignored.
- Inapplicable plot kinds no longer leak open Matplotlib figures.
- README plugin example matches the real registry API.

### Changed
- **Breaking:** requires Python >=3.11 and protobuf >=6.31.1. Chakra-ET parsing now uses the official `mlc-chakra` schema from the pinned `third_party/chakra` git submodule (PyPI 1.0.0 predates storage/data-op nodes) instead of the vendored copy. Clone with `--recurse-submodules`. A path dependency means this package cannot be uploaded to PyPI as is.

### Added
- `--scope-depth N` (`compare`, `analyze`; `scope_depth=` in the SDK and manifest) groups the `by_scope` kernel table by the N-th enclosing op/range counted from the outermost (`1` = outermost, e.g. a `record_function` region), clamped to the innermost; the default stays the innermost. Backed by a new `scope_path` kernel column (`outer > ... > inner`) written by the Kineto (v2.2) and Nsight Systems (v2.1) readers, so cached results rebuild once.

### Visualisation (rewritten)
- Plots are now built once as a backend-neutral chart (`cta.viz.ir`) and drawn by Matplotlib (pdf/svg/png, unchanged styling) or Plotly (`.html`, optional `cta[viz]` extra). `cta plot --html FILE` / `report.dashboard()` writes one offline dashboard with tabs per use case and group/rank/step-range filters; `--cdn` makes it small. `cta plot ... --serve` runs it live on a loopback server: every filter (group, rank, GPU device, step window) re-runs the analysis for that slice, so aggregates are exact (verified on 2xH200: a device filter halves the breakdown, a step window changes `total_steps`). The saved static file only hides/restyles what is drawn (group incl. category-axis bars, rank, step-range zoom). Not implemented: links between plots.
- New analysis filters `--ranks` / `--devices` for `analyze` and `compare` (manifest `ranks:`/`devices:`, SDK `ranks=`/`devices=`), applied before warm-up/max-step slicing, so every aspect and plot sees the same slice.
- 30 plot kinds, each with declared, typed options (`cta plot --list-kinds`; unknown options are an error with a suggestion): the original six plus timeline Gantt, overlap timeline, stream utilisation, Pareto, family share, step trace (rolling median, Theil-Sen drift, outliers), step distribution/comparison, kernel drift heatmap, waterfall, kernel scatter, scorecard, gates, sweep, rank heatmap, straggler, occupancy, throughput, stalls, power and GPU-metric traces, latency/energy Pareto, `history` (control chart over saved `report.json` files) and `custom`.
- Plots can use raw tables, not just report aggregates: `kernels`, `steps`, `gpu_metrics`, `runtime`, `ncu_kernels`, `ncu_findings`, `jtop`, `osrt`, sliced with the same warm-up/max-step logic as the aspects and with per-rank relative time. `custom` and `--sql` accept DuckDB SQL over them; `--spec FILE` and a manifest `plots:` block declare plots in YAML.
- `report.json` now records `created_at`, the cta version and the git commit (used to order `history`; old reports fall back to file mtime). `--export html` writes the dashboard.
- Backwards compatible: `render_plot`/`report.plot` signature, the six original kinds and the `tornado` alias, `--export pdf|svg|png` keys, and ValueError for inapplicable plots (now the `NotApplicable` subclass).
- Fixed: `compare` failed with a polars SchemaError when one capture had no OS-runtime rows (empty frame without a key dtype).

### JAX profiler adapter (new, v1.0)
- Reads `jax.profiler.trace` log directories: the XProf `*.trace.json.gz`, jax's `perfetto_trace.json.gz`, or a bare `*.xplane.pb` (converted with the optional `xprof` package, `pip install cta[jax]`). The three files of a session describe one profile, so one is read per host. Device ops come from the GPU stream lines (XProf's derived lines are skipped); the `hlo_op`/name stack gives `scope` and `scope_path` (`jit(train_step) > jvp() > dot_general`), NCCL/all-reduce ops are `comm`, and memcpy/memset are `memcpy`.
- Steps come from `jax.profiler.StepTraceAnnotation(name, step_num=i)`; without markers each host-side execution of the busiest jitted module is a step (with a warning; the first one includes compilation, use `--warmup-steps 1`). Ops are assigned to the step whose host code launched them (`correlation_id`), and work outside every step (compilation, autotuning) is dropped. Host CUDA API calls made inside a step fill the runtime table, so `cuda_runtime_ms` is real (0.75 ms on the H200 run) instead of 0.
- `cta profile jax -- python train.py [--analyze]` writes the log directory (`cta.profile.jax_runner`). Validated on 2x H200: 8 annotated steps x 16 ops/device, one NCCL all-reduce per step and device, identical results from the trace JSON and from the `xplane.pb`.

### Kineto adapter
- **Scope assignment (v2.1):** a kernel's scope is now the innermost CPU op that encloses its launch *on the launching thread* (runtime correlation id -> thread and time). The old lookup took the latest-starting op in the whole trace, so a finished sibling hid the enclosing op: on real H200 traces 15% (single GPU) and 56% (2-rank DDP) of kernels came out with no scope; now none do. The interval lookup is shared with the Nsight Systems reader (`cta.core.intervals`).
- **Multi-rank traces (v2.0, cache rebuilds automatically):** every shard keeps its own clock and `rank` (from the trace's `distributedInfo.rank`, else a rank in the file name, else its position) and each GPU event its `device` (`args.device`). Step ranges are matched per rank and a kernel is assigned to the step whose CPU range *launched* it (via the runtime correlation id) instead of the step it runs in. Previously all shards were merged into one pile and step lookups mixed clocks. Verified on a 2-rank DDP run on two H200s. `cta.profile.kineto_runner --out dir/trace_rank{rank}.json.gz` writes one trace per rank under `torchrun`.
- `ProfilerStep#N` is emitted by PyTorch both as a CPU `user_annotation` and as a `gpu_user_annotation`; the reader counted both, so a trace with N steps reported N+1 and its cold-start step was counted twice (inflating `total_steps`, the mean and the tail percentiles). Only the CPU annotation is used now. Found on a real H200 trace.

### Nsight Systems adapter (v2.0, cache rebuilds automatically)
- Reads **every** report of a bundle (one per rank); rows carry new `rank` and `device` columns. Previously only the first file was read.
- Reads `deviceId`, so multi-GPU reports are analysed per device instead of one merged timeline; timeline overlap is computed per (rank, device) inside each rank's own step window.
- Steps and scopes are assigned from the **launching** thread: kernel/memcpy/memset -> `correlationId` -> runtime call (`globalTid`, launch time) -> innermost enclosing NVTX range. Launches are asynchronous, so the old GPU-start-time lookup attributed kernels to the wrong step. Unmatched activity falls back to a cross-thread lookup.
- Scope lookup returns the innermost *enclosing* range even when a later-starting sibling has already ended.
- `CUPTI_ACTIVITY_KIND_MEMSET` is ingested; memcpy kinds include HtoH.
- Zero-valued `GPU_METRICS` samples are kept (they were dropped, inflating SM-active medians).
- Traces without NVTX step markers warn that `--warmup-steps`/`--max-steps` have no effect; the warmup slicing also warns when a rank has too few steps.
- Throughput is computed per rank and averaged (ranks have independent clocks); `total_steps` is steps per rank.
- Kineto and TensorRT tables gain `rank`/`device` (0) so all timeline formats share one schema.

### Nsight Compute adapter (v2.0, cache rebuilds automatically)
- **Correctness (numbers change):** on real `--set full` reports the old reader returned `0` FLOPs and `0` DRAM bytes for every kernel (its metric names do not exist in such reports). FLOPs are now the achieved counts `smsp__sass_thread_inst_executed_op_{f,h,d}{add,mul,fma}_pred_on.sum.per_cycle_elapsed` x `smsp__cycles_elapsed.avg` (FMA = 2, packed half add/mul = 2, half FMA = 4) plus tensor-core `sm__ops_path_tensor_src_*.sum`. Verified on an H200 report: an FP32 1024x512x512 GEMM measures 5.370e8 FLOPs vs 2MNK = 5.369e8. The `derived__..._x2/_x4` metrics are per-device *peak* constants and are not used as counts.
- DRAM bytes are `dram__bytes_read.sum + dram__bytes_write.sum`; DRAM utilisation is `gpu__dram_throughput` (it used the overall-memory maximum `gpu__compute_memory_throughput`, now a separate `mem_throughput_pct` column).
- Reads `.ncu-rep` and `.ncu-repz` directly through NVIDIA's `ncu_report` (new optional extra `[ncu]`; macOS/Linux/Windows wheels, no `ncu` CLI needed), falling back to `ncu --import --csv`, then a sibling `.csv`.
- CSV units row is honoured for every column (`us`/`ms`/`s`, `Kbyte`/`Mbyte`/`Gbyte`); the report and CSV paths agree on real data.
- Missing metrics are `null` instead of `0`; duration-weighted means use only kernels that report the metric.
- Roofline points use sum(FLOPs)/sum(bytes) and sum(FLOPs)/sum(time) per kernel, not a mean of per-call ratios.
- Stall columns are shares of all issue-stall cycles (memory: long scoreboard, LG/TEX throttle, membar; compute: math pipe throttle, GMMA).
- New `ncu_findings` table: NCU's own guided-analysis rules (title, message, estimated speedup) per kernel, summarised in `roofline_diff` details as `findings_<label>`.
- New columns: `kernel_id`, `mem_throughput_pct`, `flops_tensor`, `grid_size`, `block_size`, `registers_per_thread`.
- Optional `[profile]` extra installs the `nsight-python` submodule (Linux only; v1.1.1 imports need `cuda-core`/`cuda-bindings`, which have no macOS wheels, and conflict with PyTorch's `cuda-bindings<13` pin).

### Added
- `cta profile {nsys,ncu,kineto,trtexec,jtop} -- <command>`: runs the vendor profiler with the flags the adapters need and writes a RunBundle directory (`--analyze` runs `cta analyze` on it). nsys, ncu and kineto were verified end-to-end on an H200; trtexec and jtop are unit-tested only (they need TensorRT / Jetson hardware).
- Chakra `STORAGE_NODE` and `DATA_OP_NODE` nodes are ingested as `op_class` `storage` / `data_op` (storage size taken from the typed fs/page/block detail).
- `cta --version`, `python -m cta`, `cta list`, `--format json|rich` and `--alpha` on `compare`/`analyze`, `--warmup-steps`/`--max-steps` on `plot`.
- `py.typed` marker, Python 3.13 in CI and classifiers.
- Version is single-sourced from `cta/__init__.py` (hatch dynamic version).
