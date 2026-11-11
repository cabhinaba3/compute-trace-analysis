# compute-trace-analysis

Python helpers for finding where time goes in DNN inference on NVIDIA GPUs and Jetson boards. It reads three kinds of trace:

- trtexec timing exports (`trtexec --exportTimes=times.json`)
- jtop logs from Jetson devices, saved as JSON
- Nsight Systems reports (`.nsys-rep` or the exported `.sqlite`)

It was written for the measurements in our ISPASS'25 paper. It's research code, so expect rough edges.

## Install

Tested on Ubuntu 20.04 with Python 3.10. There's no PyPI package.

```
git clone https://github.com/cabhinaba3/compute-trace-analysis.git
cd compute-trace-analysis
pip install -r requirements.txt
pip install -e .
```

The Nsight Systems parts call `nsys export`, so `nsys` has to be on your `PATH`.

## Usage

### trtexec timings

`cta.time.Time` takes the per-inference timestamps from `--exportTimes` and splits the run into host-to-device copy, compute, device-to-host copy and idle time.

```python
from cta.time import Time, throughput

t = Time("times.json", timing_info=True)
t.get_utilization_stats
# {'communication_overlap_time': ..., 'gpu_idle_time': ...,
#  'communication_time': ..., 'compute_time': ...}

throughput("times.json")  # inferences per second
```

The values are fractions of the total run time. `communication_overlap_time` is the share of copy time that overlaps with compute.

### jtop

`cta.jtop.Jtop` takes a JSON list of jtop samples and collects CPU and GPU utilisation, RAM, swap, GPU memory, GPU temperature and energy (from the `Power TOT` rail).

```python
from cta.jtop import Jtop

j = Jtop("jtop.json", cpu_core=6, jtop_interval=1.0)
j.get_gpu_utilization       # list, one value per sample
j.get_energy_consumption    # total energy in J
j.get_power_cosnumption     # average power in W
j.get_peak_gpu_memory_usage
```

Set `cpu_core` to the number of CPU cores on the board (default 6).

### Nsight Systems

`cta.nsys.multi_report` takes a list of reports, so you can compare runs (batch sizes, precisions, models) in one call. A `.nsys-rep` is exported to `.sqlite` next to it the first time; after that the `.sqlite` is reused.

GPU metrics come from the `GPU_METRICS` table, so profile with `--gpu-metrics-device`:

```python
from cta.nsys import multi_report as rpt

reports = ["bs1.nsys-rep", "bs8.nsys-rep"]

rpt.sm_active(reports, smoothing=None, plotting_criteria="median")
rpt.tensor_active(reports, smoothing="moving_average")
rpt.sm_issue(reports, smoothing=None)
rpt.compute_warps(reports, smoothing=None, all_data=True)
```

Each call returns `{"raw_stats": [...], "smoothing_stats": {...}}` with one entry per report. `plotting_criteria` can be `median`, `mean` or `max`; pass `all_data=True` to get every non-zero sample. Smoothing options are `moving_average`, `exponential_moving_average` and `savitzky_golay_filter`.

For API and kernel timings:

```python
rpt.kernel_stats(reports)  # CUDA runtime calls (CUPTI_ACTIVITY_KIND_RUNTIME)
rpt.nvtx_stats(reports)    # NVTX ranges, e.g. TensorRT layers
rpt.os_stats(reports)      # OS runtime calls (OSRT_API)
```

These return one dict per report with the mean, median, count and sum of durations (in ns), grouped by name.

### Performance models

`cta.perf_model` has small curve models (linear, exponential, linear plus exponential) for fitting latency or energy against things like batch size, and `GatherData`, which pulls the kernel and NVTX stats for a set of reports in one go.

## Layout

```
cta/
  time.py            trtexec --exportTimes parser
  jtop.py            jtop log parser
  util.py            path and JSON helpers
  nsys/
    db.py            nsys-rep -> sqlite export and queries
    nsys.py          readers for GPU metrics, CUDA, NVTX and OS runtime tables
    multi_report.py  the same, over a list of reports
  perf_model/
    math_model.py    curve models
    perf.py          GatherData
```

The plotting scripts for the paper and the characterisation experiments are not in this repo.

## Known issues

- The Nsight table names follow recent `nsys` versions. Older reports use `GENERIC_EVENTS` instead of `GPU_METRICS` and won't load.
- `LinExponentialModel_2D.model` is unfinished.
- Only tested on Linux.

Bug reports and pull requests are welcome.
