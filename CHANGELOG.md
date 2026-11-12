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

### Kineto adapter
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
