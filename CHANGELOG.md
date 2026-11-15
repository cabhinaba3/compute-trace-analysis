# Changelog

## Unreleased

### Network API analysis (`network_api`)
- New `network_ops` table from the `kineto` and `nsys` adapters, one row per communication call: API (`nccl`, `mpi`, `ucx`, `p2p`), logical operation, algorithm and protocol, dtype, process group, group size, message size, launch/start/end times and a `posted` flag. Both adapter versions went up, so cached ingests are rebuilt.
- NCCL: the Kineto reader takes sizes, dtype and group from the args PyTorch puts on NCCL kernels. nsys has no per-call sizes for NCCL, so those rows have timing and algorithm but a size of 0. Kernel names from NCCL 2.32 (`ncclDevKernel_AllReduce_f32_RING`) parse too.
- MPI and UCX come from the `nsys` adapter (`MPI_COLLECTIVES_EVENTS`, `MPI_P2P_EVENTS`, `UCP_SUBMIT_EVENTS`, `MPI_COMMUNICATORS`). MPI names map to the NCCL operation names, and GPU peer copies become `p2p` rows. Calls that only post a transfer (`MPI_Isend`, UCX submits) are counted but left out of bandwidth and skew.
- MPI sizes are converted to the nccl-tests size per operation. I measured how nsys records them with 2, 3 and 4 Open MPI ranks; other MPI implementations were not checked. An in-place allreduce, allgather or alltoall cannot be told from a plain one and reads low by (n-1)/n.
- New aspect `network_api`: bandwidth (algbw, busbw), time, call counts, cross-rank wait skew and launch-to-start gap, per API, operation and size bucket. The `by_operation` table pools the APIs so a run that used NCCL lines up with one that used MPI.
- `network_ops` has a `peer` column: the remote rank of an MPI send or receive from nsys, -1 where the trace does not record one (collectives, NCCL, UCX). The `nsys` and `kineto` adapters are version 2.4, so cached tables are rebuilt.
- New plot source `network_ops`, usable in `custom` and `--sql`, and four plot kinds on it: `network_bandwidth` (bus or algorithm bandwidth by operation and size), `network_skew` (cross-rank wait skew or launch gap), `network_size_sweep` (bandwidth against message size, one panel per operation) and `network_matrix` (send volume between rank pairs; MPI point-to-point only).

### Trace analysis plots
- New plot kinds `metric_trace`, `metric_dist`, `category_compare`, `execution_heatmap` and `metric_correlation` for time series, distributions, category comparisons, rank-by-time heatmaps and metric correlations over the raw tables.
- New `violin` mark and kinds `violin`, `boxplot` and `line_plot`; `metric_dist` has a `violin` mode.
- New `queue_depth` kind (aliases `concurrency`, `active_kernels`): how many kernels run at once on each GPU, with the peak of every time bin.
- Plots now warn when the data does not suit the kind (a violin of a few samples, a line through categories, a cross-rank plot of a one-rank trace). `cta plot` prints the warnings, and the dashboard shows them with buttons for the suggested kinds. Each dashboard card has a kind selector; switching works in the live dashboard (`--serve`) only. `cta.viz.dashboard.figure_json` returns the warnings as a fourth value.

### More plot kinds
- New `runtime_calls` table from the `nsys` and `kineto` adapters (both now version 2.5): every CUDA runtime call with its name, thread, correlation id, step and time. It is also a plot source, usable in `custom` and `--sql`.
- Statistics: `forest` (change of every metric with its combined 95% interval) and `volcano` (size of the change against -log10 p).
- Memory: `memcpy_bandwidth`, `memcpy_sizes` and `memcpy_overlap` (copy time hidden behind compute, per device).
- Hierarchy and idle time: `scope_icicle` (time by nested scope), `idle_gaps` (gaps from the union of each device's kernels) and `critical_path` (compute, exposed communication and slack from the `dag_critical_path` summary; the solver returns no path, so the path itself is not drawn).

### Native synthetic generators; Linux only; Python 3.12+ (breaking)
- Python 3.11 is no longer supported. CI, the classifiers, ruff and mypy moved to 3.12.
- The generators in `cta.synth` (Kineto, GOAL, Chakra, nsys SQLite, ncu, jtop, trtexec) are now C++ in `native/`, built into the package as `cta._native` by `pip install`. The Python generators are gone. Output matches the old writers byte for byte (nsys SQLite: same rows).
- New `cta.synth.training` writes 3D-parallel training graphs (data, pipeline, tensor, optional MoE) as GOAL text, GOAL binary and Chakra shards, at 512 ranks and beyond.
- New `write_kineto_trace_large` (any size, about 400 MB/s) and `write_goal_allreduce_bin`.
- macOS is no longer supported. The package needs Linux, a C++20 compiler, and the zlib, SQLite, Protobuf and simdjson development packages.

### Native Nsight Systems reader
- The `nsys` adapter reads the SQLite export in C++ (`cta._native.read_nsys_sqlite`) instead of looping over rows in Python. A 326 MB export (5.0M kernels) takes 7.1 s instead of 30.8 s, with about 3 GB of RSS instead of 4 GB.
- Activities that start together keep their table order; before, their order was undefined. A NULL `bytes`, stream or device reads as 0.
- `cta.core.intervals.CodedNames` is public.

### Native GOAL text reader
- `parse_goal_text` reads `.goal` text in C++ (`cta._native.read_goal_text`), in parallel, and keeps the old parser's rules and quirks. A 3.0M-node schedule (209 MB) takes 1.4 s instead of 20.8 s.
- Text that is not UTF-8, or a number too big for its column, now raises `ValueError` from the reader.
- The `.bin` path and the `traffic` table are unchanged.

### Native DAG edge build
- `cta.core.dag._global_edges` builds the edge list in C++ (`cta._native.build_dag_edges`) instead of with Polars joins. On the 512-rank, 10.9M-node training graph the edge build takes 2.4 s instead of 5.8 s.

### Native DAG solver
- `solve_distributed_dag` keeps its signature, but the graph passes (Kahn, slack, critical chain) run in C++ (`cta._native.solve_dag`). 10.9M nodes take 4.8 s in total instead of 48.8 s.
- **Behaviour change:** when predecessors tie on finish time and depth, the one with the lowest vertex id is now the critical-path parent. Before, it depended on a row order Polars does not define. `cp_compute_ns`, `cp_exposed_comm_ns` and `cp_comm_bytes` can differ from older runs on graphs with such ties; the same graph now gives the same answer every time.
- Floating-point sums use extended precision and agree with NumPy's to within 1e-9 relative.

### Native Chakra reader
- The `chakra_et` adapter reads `.et` files in C++ (`cta._native.read_chakra_et`) with the generated protobuf classes. A 10M-node, 512-rank graph takes 5.4 s instead of 295 s, with 3.7 GB of RSS instead of 11 GB. The Python per-node loop and varint reader are removed.
- The `traffic` table is now sorted by (src_rank, dst_rank); its row order used to be arbitrary. Values are the same.
- `chakra_host_et` is unchanged.

### Native Kineto reader
- The Kineto adapter reads traces in C++ (`cta._native.read_kineto_trace`, simdjson) with a thread pool; the tables do not depend on the thread count. A 1 GB trace takes 3.8 s instead of 39.8 s, with 1.06 GB of RSS instead of 5.04 GB.
- `cta.core.intervals`: the nesting-parent loop is in C++ (`cta._native.nesting_parents`), `build_thread_ranges_arrays` takes NumPy arrays, and `range_paths` no longer loops over every range in Python.

### Google Python style refactor (breaking)
- The code now follows the Google Python Style Guide: 80-column lines, Google docstrings, one import per line, and modules imported instead of names (`from cta.core import models`, then `models.TraceArtifact`). Ruff's `D` (google convention) and `N` rules are on; `tests/unit/test_code_style.py` checks the import rule, which ruff cannot.
- Renamed Parquet columns: `tag` is `message_tag` (nodes), `msg_count` is `message_count` (traffic), `arith_intensity` is `arithmetic_intensity` (ncu kernels, and the `roofline_diff` metric is `overall_arithmetic_intensity`), `power_tot_w` is `power_total_w` and `cum_energy_j` is `cumulative_energy_j` (jtop samples), `pct_time` is `time_share_pct` (trtexec layer profile). The chakra_et, chakra_host_et and goal adapters are now v1.1, jtop v1.2, ncu v2.1 and trtexec_profile v1.1, so cached tables are rebuilt once.
- Renamed CLI: `cta equiv` is `cta verify-equivalence`, and `--collective-algo` is `--collective-algorithm`. The manifest key and `CompareConfig` field are `collective_algorithm` too.
- Renamed SDK names: `cta.equiv` is `cta.verify_chakra_goal_equivalence`, the `schedule_equiv` aspect is `schedule_equivalence` (class `ScheduleEquivalenceAspect`), and `cta.viz.spec.NotApplicable` is `NotApplicableError`.
- Renamed keyword arguments and fields: `write_goal_allreduce(algo=)` is `algorithm=`; `compare_entity_tables(group_dfs, key_col=, val_col=, count_col=)` is `(group_tables, key_column=, value_column=, count_column=)`; `slice_steps_and_kernels(steps, kernels, config)`, `gpu_metrics_in_steps(gpu_metrics, steps)`, `pick_timeline_tables(table_paths)` and `with_scope_level(kernels, depth)` lost their `_df`/abbreviated names; `kineto_command(trace_path=)`, `jax_command(log_dir=)`, `trtexec_command(bundle_dir=)`; `enrich_ops_df(ops, name_column=)`, `op_family_expr(column=)`, `CacheStore.write_parquet(frame, path)`; in `cta.core.intervals`, `ThreadRanges.gids` is `range_ids` and the range helpers take `thread_ranges`, `query_ns`, `range_ids`, `gpu_start_ns`, `launch_ns`, `gpu_ns`, `compute_starts`/`compute_ends`; in `cta.viz`, `Opt.type`/`Opt.help` are `value_type`/`help_text`, `Card.id` is `index`, `render(colors=)` is `render(palette=)`, the `opts` argument of `to_matplotlib`/`to_plotly`/`render_chart` is `options`, and `dashboard.page(js=)` is `plotly_js=`. `cta.adapters.goal.reader.parse_goal_text` is now public.
- Split modules: `cta.aspects.device` is a package (`latency`, `kernels`, `timeline`, `os_runtime`, `roofline`, and `step_slicing` for the shared step-window helpers). `cta.synth.generators` is split into `cta.synth.chakra`, `goal`, `nsys`, `kineto`, `ncu`, `tensorrt` and `jtop`. `cta/cli/app.py` only registers commands; they live in `compare_commands`, `plot_command`, `trace_commands`, `profile_commands` and `cache_commands`, with shared console output in `cta.cli.output`. `cta.viz.kinds._util` is `cta.viz.kinds.chart_helpers`.
- Test modules are named after what they test (`test_core_wave0.py` is `test_core.py`, `test_wave1_chakra_goal.py` is `test_chakra_goal_adapters.py`, `test_review_fixes.py` is `test_cli_regressions.py`, and so on).

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
