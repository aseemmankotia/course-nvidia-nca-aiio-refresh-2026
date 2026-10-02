# Chapter 7 Lab: GPU Monitoring, Troubleshooting, and Optimization

**Estimated time:** 16 minutes

## Exercise

Monitoring and Troubleshooting Lab: Analyze provided nvidia-smi output samples showing: (A) thermal throttling (temp 87°C, clock reduced), (B) OOM error (memory 100%, process killed), (C) low utilization (15% util, pipeline bottleneck). Diagnose root cause and propose fix. Write nvidia-smi query command to export GPU memory usage as CSV every 10 seconds. Review DCGM sample config for alerting on temp >85°C.

## Key takeaway

nvidia-smi for ad-hoc queries (memory, temp, utilization), DCGM for production monitoring/alerting—low GPU utilization usually means data pipeline bottleneck (CPU), not GPU problem.
