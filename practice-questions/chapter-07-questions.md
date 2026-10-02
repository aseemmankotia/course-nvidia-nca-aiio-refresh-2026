## Chapter 7: GPU Monitoring, Troubleshooting, and Optimization — Practice Questions

**Q1.** A data scientist reports that their PyTorch training job is running slower than expected on an A100 GPU. You run nvidia-smi and see GPU utilization at 45% and memory utilization at 30%. Temperature is 65°C and power draw is 180W (limit 400W). What is the MOST likely bottleneck?

A) Thermal throttling is limiting GPU performance due to high temperature
B) Out of memory errors are causing frequent garbage collection
C) Driver version mismatch is preventing full GPU utilization
D) Data pipeline bottleneck - GPU is waiting for data preprocessing on CPU

<details>
<summary>Answer</summary>

**Correct: D**

Low GPU utilization (45%) with low memory usage (30%), normal temperature (65°C is well below 80°C throttling threshold), and low power draw (180W vs 400W limit) indicates the GPU is idle waiting for data. This is classic data pipeline bottleneck where CPU preprocessing can't keep up with GPU consumption.

**Why the others are wrong:**
- Thermal throttling occurs above 80-85°C. At 65°C the GPU is running cool with plenty of thermal headroom.
- OOM errors would show memory at 95-100% utilization, not 30%. Also would see errors in application logs.
- Driver mismatch typically prevents GPU detection entirely or causes immediate errors, not gradual low utilization. nvidia-smi working confirms driver is functional.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** You need to continuously monitor GPU temperature and memory usage across 50 DGX servers and trigger alerts when temperature exceeds 82°C or memory exceeds 90%. Which solution BEST meets these requirements?

A) Write a bash script using nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv --loop=1 and parse output to trigger alerts
B) Use Nsight Systems to profile each GPU and configure temperature thresholds in the GUI
C) Deploy DCGM agents on each node, configure health checks with temperature and memory policies, and integrate with alerting system
D) Enable NVIDIA GPU Operator in Kubernetes and use kubectl to query GPU metrics

<details>
<summary>Answer</summary>

**Correct: C**

DCGM (Data Center GPU Manager) is purpose-built for production GPU monitoring at scale. It runs lightweight agents on each node, supports policy-based health checks, efficient group-level monitoring, and integrates with enterprise alerting systems (Prometheus, Grafana, etc.). This is the enterprise-grade solution for 50+ servers.

**Why the others are wrong:**
- nvidia-smi polling scripts work for 1-2 servers but don't scale to 50 servers. Creates high overhead, no centralized management, brittle alert logic, and difficult to maintain across fleet.
- Nsight Systems is a profiling tool for development/debugging individual applications, not a production monitoring solution. Cannot continuously monitor 50 servers or trigger automated alerts.
- GPU Operator helps deploy GPU drivers/toolkit in Kubernetes but doesn't provide built-in alerting or health policies. You'd still need DCGM or similar monitoring layer on top.

</details>

---

**Q3.** A training job fails with 'CUDA out of memory' error on an A100 80GB GPU. You check nvidia-smi and see memory used is 72GB. The model has 70B parameters. Which action will MOST effectively allow the job to run?

A) Increase GPU temperature limit to allow higher clock speeds and faster memory clearing
B) Upgrade CUDA driver to latest version to optimize memory management
C) Switch from A100 80GB to H100 96GB GPU for more memory capacity
D) Reduce batch size from 32 to 16 and enable gradient checkpointing

<details>
<summary>Answer</summary>

**Correct: D**

OOM with 72GB used on 80GB GPU indicates memory fragmentation or activation spikes during training. Reducing batch size directly reduces activation memory (linear reduction). Gradient checkpointing trades compute for memory by recomputing activations instead of storing them, saving 40-60% memory. This combination typically solves OOM without hardware changes.

**Why the others are wrong:**
- Temperature limit affects clock speeds and performance, not memory capacity. Higher temperatures won't free memory and could cause thermal throttling.
- CUDA driver updates provide bug fixes and features but don't significantly change memory capacity. The 72GB→80GB gap won't be solved by driver updates alone.
- While H100 96GB provides more memory, it's expensive overkill when software optimizations (batch size, checkpointing, mixed precision) can solve the problem on existing hardware.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** After updating CUDA toolkit from 12.1 to 12.4, nvidia-smi shows the GPUs but training jobs fail with 'no CUDA-capable device detected'. Driver version is 525.85.12. What is the problem?

A) GPU firmware needs to be updated to support CUDA 12.4
B) CUDA 12.4 requires driver version 530.30.02 or higher - driver must be upgraded
C) DCGM agents need to be restarted after CUDA toolkit upgrades
D) PyTorch needs to be reinstalled to link against new CUDA libraries

<details>
<summary>Answer</summary>

**Correct: B**

CUDA 12.4 requires minimum driver version 530.30.02. Driver 525.85.12 is too old and only supports up to CUDA 12.1. nvidia-smi works because it uses driver API, but CUDA runtime requires driver/CUDA version compatibility. This is the most common driver mismatch issue and heavily tested on exam.

**Why the others are wrong:**
- GPU firmware is separate from driver/CUDA stack and rarely needs updates. Firmware issues would prevent nvidia-smi from detecting GPUs at all.
- DCGM is a monitoring tool and doesn't affect CUDA runtime availability. Restarting DCGM won't fix driver/CUDA version mismatch.
- PyTorch may need reinstall for optimal performance but the error 'no CUDA-capable device detected' is a driver/CUDA compatibility issue, not a PyTorch linking issue. PyTorch would give different error if just linking was wrong.

</details>

---

**Q5.** You need to identify which CUDA kernel is consuming the most GPU time in a TensorFlow training script. Which tool provides kernel-level GPU profiling with timeline visualization?

A) Nsight Compute to profile individual kernel launches with detailed metrics
B) nvidia-smi with --query-compute-apps flag to show per-process kernel usage
C) DCGM profiling metrics with dcgmi profile --start
D) TensorBoard profiler plugin to capture GPU timeline and kernel traces

<details>
<summary>Answer</summary>

**Correct: A**

Nsight Compute is NVIDIA's kernel-level profiling tool that provides detailed analysis of individual CUDA kernels: execution time, memory bandwidth, occupancy, warp efficiency, etc. It shows exactly which kernels are bottlenecks with timeline visualization and optimization suggestions. This is the definitive tool for kernel-level profiling.

**Why the others are wrong:**
- nvidia-smi --query-compute-apps shows process-level GPU usage (PID, memory, GPU %) but cannot break down individual kernel performance. Too coarse-grained for kernel profiling.
- DCGM profiling provides GPU-level metrics (utilization, memory, power) for monitoring but doesn't capture kernel-level details or timelines. It's for fleet monitoring, not kernel debugging.
- TensorBoard profiler shows TensorFlow op-level performance and GPU timeline, which is helpful but provides framework-level view. Nsight Compute gives lower-level CUDA kernel details needed to optimize GPU code directly.

</details>

---

**Q6.** A production inference server with 8x A100 GPUs shows nvidia-smi reporting GPU 3 at 0% utilization while GPUs 0-2 and 4-7 are at 95% utilization. The application uses CUDA_VISIBLE_DEVICES to distribute work. What is the MOST likely cause?

A) GPU 3 has entered thermal throttling mode and shut down to cool
B) Application's GPU assignment logic is skipping GPU index 3 due to off-by-one error or incorrect device enumeration
C) GPU 3 has a driver error and needs nvidia-smi -r to reset
D) GPU 3 is running at lower clock speed due to power limit configuration

<details>
<summary>Answer</summary>

**Correct: B**

When one GPU in a multi-GPU system shows 0% utilization while all others are heavily utilized, it indicates the application is not assigning work to that GPU. Common causes: off-by-one errors in device indexing (e.g., using range(8) but skipping index 3), hardcoded device lists missing GPU 3, or incorrect CUDA_VISIBLE_DEVICES mapping. Hardware/driver issues would show errors or prevent detection.

**Why the others are wrong:**
- Thermal throttling reduces clock speeds and performance but doesn't drop utilization to 0%. GPU would still show activity, just slower. Also throttling at 0% while others at 95% suggests different workload, not thermal issue.
- Driver errors typically prevent GPU detection entirely (nvidia-smi wouldn't show GPU 3) or cause visible errors/crashes. A functioning but unused GPU points to application logic, not driver failure.
- Power/clock limits affect performance (lower clocks = slower work) but don't prevent work assignment. GPU 3 would still show utilization percentage even at reduced clocks.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** You want to monitor GPU utilization percentage, memory used in MB, and power draw in watts for 4 GPUs every 2 seconds and save to CSV file. Which nvidia-smi command achieves this?

A) nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv --loop=2 > gpu_metrics.csv
B) nvidia-smi dmon -s umpt -c 2 -d 4 > gpu_metrics.csv
C) nvidia-smi --query-gpu=gpu_utilization,memory_usage,power --format=csv --interval=2000 > gpu_metrics.csv
D) nvidia-smi stats -d 2 -c 4 --csv > gpu_metrics.csv

<details>
<summary>Answer</summary>

**Correct: A**

--query-gpu with exact metric names (utilization.gpu, memory.used, power.draw), --format=csv for CSV output, and --loop=2 for 2-second intervals is the correct syntax. This is heavily tested on exam - exact flag and metric names matter.

**Why the others are wrong:**
- dmon is a valid nvidia-smi subcommand but -c flag specifies count (number of samples), not interval duration. Also dmon uses different metric names (sm for utilization, not same as --query-gpu).
- Metric names are wrong: must be utilization.gpu (not gpu_utilization), memory.used (not memory_usage). Also --interval flag doesn't exist, nvidia-smi uses --loop for interval in seconds.
- nvidia-smi stats command exists but doesn't support the flags shown. -d and -c flags work differently in stats mode, and --csv flag syntax is incorrect.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A deep learning training job runs on V100 32GB GPU. You want to maximize GPU utilization by increasing batch size but keep encountering OOM errors. Current batch size is 64 using FP32 precision. Which combination will MOST likely allow largest batch size increase without OOM?

A) Enable CUDA memory caching and increase batch size to 96
B) Switch to mixed precision (FP16) training with automatic loss scaling and increase batch size to 128
C) Reduce model parameter count by 25% and keep batch size at 64
D) Enable gradient accumulation with 4 steps and keep batch size at 64

<details>
<summary>Answer</summary>

**Correct: B**

Mixed precision (FP16) reduces memory usage by ~50% for activations and gradients (16-bit vs 32-bit). Automatic loss scaling prevents underflow. This memory saving allows doubling batch size from 64→128. Mixed precision is the single most effective technique for increasing batch size on existing hardware and heavily tested on exam.

**Why the others are wrong:**
- CUDA memory caching is enabled by default and doesn't free memory - it reuses allocated memory for efficiency. Won't allow OOM at 64 to suddenly work at 96.
- Reducing model parameters helps but requires retraining and architectural changes. Question asks to maximize batch size for existing model, not redesign model. Also 25% parameter reduction doesn't translate to 25% memory saving due to activations.
- Gradient accumulation simulates larger batch size by accumulating gradients over multiple forward passes, but doesn't reduce memory usage per batch. Still OOM at batch 64. It helps when you want effective batch 256 but can only fit batch 64.

</details>

---

