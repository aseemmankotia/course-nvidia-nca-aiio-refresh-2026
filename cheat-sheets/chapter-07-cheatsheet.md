# GPU Monitoring, Troubleshooting, and Optimization Cheat Sheet

## nvidia-smi Critical Commands

| Command | Purpose | Exam Note |
|---------|---------|----------|
| `nvidia-smi` | Basic GPU status | Shows utilization %, memory, temp, power |
| `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` | Query specific metrics | Exact metric names required |
| `nvidia-smi --loop=5` | Poll every 5 seconds | --loop=N where N is seconds |
| `nvidia-smi -i 0` | Query GPU 0 only | -i flag for specific GPU |
| `nvidia-smi -r` | Reset GPU | Use for stuck GPU, not normal operation |
| `nvidia-smi -l 1` | Legacy loop (1 sec) | -l same as --loop |

## Key Metrics and Thresholds

| Metric | Normal Range | Alert Threshold | Exam Tested |
|--------|--------------|----------------|-------------|
| GPU Utilization | 70-100% training | <50% = bottleneck | Heavily |
| Memory Usage | 80-95% training | >95% = risk OOM | Heavily |
| Temperature | 60-75°C | >80°C = throttling | Yes |
| Power Draw | Near limit when training | Sustained at limit = check cooling | Yes |
| GPU Clocks | Max during training | Below max = throttling/idle | Sometimes |

## DCGM Architecture

```
Host Engine (nv-hostengine)
  └── Agents (per node)
       └── Groups (logical GPU collections)
            └── Policies (health checks/alerts)
```

**DCGM Key Commands:**
- `nv-hostengine` - Start DCGM daemon
- `dcgmi discovery -l` - List GPUs
- `dcgmi group -c mygroup` - Create group
- `dcgmi health -g 1 -c` - Check group health
- `dcgmi profile --start --group 1` - Start profiling
- `dcgmi stats -g 1 -e` - View stats

**Exam tip:** DCGM is enterprise solution for multi-node GPU monitoring. Supports Prometheus integration, health policies, automated alerts.

## Driver vs CUDA Compatibility Matrix

| CUDA Version | Minimum Driver | Release |
|--------------|----------------|----------|
| 12.4 | 530.30.02 | Q1 2024 |
| 12.3 | 525.60.13 | Q4 2023 |
| 12.2 | 525.60.13 | Q3 2023 |
| 12.1 | 525.60.13 | Q2 2023 |
| 12.0 | 525.60.13 | Q4 2022 |
| 11.8 | 520.61.05 | Q3 2022 |

**Exam critical:** nvidia-smi showing GPUs ≠ CUDA working. Check driver >= CUDA minimum.

## Common GPU Issues and Solutions

| Issue | Symptom | Solution | Exam Tested |
|-------|---------|----------|-------------|
| OOM Error | CUDA out of memory | Reduce batch size, gradient checkpointing, FP16 | Heavily |
| Low GPU Util | <50% util, low memory | Data pipeline bottleneck - increase workers | Heavily |
| Thermal Throttling | High temp (>80°C), reduced clocks | Check cooling, reduce power limit | Yes |
| Driver Mismatch | nvidia-smi works, CUDA fails | Upgrade driver to match CUDA version | Heavily |
| GPU Not Detected | nvidia-smi shows nothing | Check driver install, lspci \| grep NVIDIA | Yes |
| One GPU Idle | GPU X at 0%, others 95% | Application logic skipping GPU - check CUDA_VISIBLE_DEVICES | Sometimes |

## Memory Optimization Techniques

| Technique | Memory Savings | Trade-off | Exam Weight |
|-----------|---------------|-----------|-------------|
| Mixed Precision (FP16) | ~50% (2x batch size) | Minimal (loss scaling needed) | Heavily tested |
| Gradient Checkpointing | 40-60% | 20-30% slower (recompute activations) | Heavily tested |
| Reduce Batch Size | Linear | Slower training, different convergence | Always an option |
| Gradient Accumulation | 0% (simulates larger batch) | More steps per update | Moderate |
| torch.cuda.empty_cache() | Fixes fragmentation | Doesn't increase capacity | Light |

**Exam tip:** FP16 + gradient checkpointing can allow 4x batch size increase.

## Profiling Tools Comparison

| Tool | Use Case | Output | When to Use |
|------|----------|--------|-------------|
| nvidia-smi | Quick status check | Terminal metrics | Real-time monitoring |
| DCGM | Production monitoring | Prometheus metrics | Multi-node fleets |
| Nsight Systems | System-wide performance | Timeline GUI | Find bottlenecks (CPU/GPU interaction) |
| Nsight Compute | Kernel optimization | Detailed kernel metrics | Optimize specific CUDA kernels |
| TensorBoard Profiler | Framework-level | TensorFlow/PyTorch op timeline | Optimize training loop |

**Exam rule:** Systems for big picture, Compute for kernel details, DCGM for production.

## Batch Size Tuning Formula

```
Max Batch Size ≈ (GPU Memory - Model Weights - Overhead) / Activation Memory per Sample

Activation Memory (FP32) ≈ Batch Size × Sequence Length × Hidden Size × Layers × 4 bytes
Activation Memory (FP16) ≈ Batch Size × Sequence Length × Hidden Size × Layers × 2 bytes
```

**Exam scenario:** Given GPU memory, model size, and precision → calculate max batch size.

## Quick Diagnostics Checklist

1. **GPU detected?** → `nvidia-smi` (if fails: driver issue)
2. **CUDA working?** → Run CUDA code (if fails: driver/CUDA version mismatch)
3. **Utilization low?** → Data pipeline bottleneck, increase num_workers
4. **Memory high?** → Reduce batch size or enable optimizations
5. **Temperature high?** → Check cooling, ambient temp, power limits
6. **One GPU idle?** → Application logic, check device assignment

**Exam weight:** GPU monitoring and troubleshooting is 22% of exam (AI Operations domain). Heavily tested: nvidia-smi query syntax, DCGM architecture, OOM solutions, driver compatibility.
