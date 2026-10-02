# Chapter 5: GPU Virtualization, Multi-Tenancy, and Capacity Planning

## MIG vs vGPU Decision Matrix

| Factor | MIG | vGPU |
|--------|-----|------|
| **Workload Type** | Containers, bare-metal | VMs (ESXi, KVM, Hyper-V) |
| **Isolation Level** | Hardware partitioning | Software sharing |
| **QoS Guarantee** | Yes (dedicated memory/compute) | No (dynamic allocation) |
| **Max Instances** | 7 per GPU (A100/H100) | Up to 24 per GPU (A30) |
| **Live Migration** | No | Yes (vMotion supported) |
| **Use Case** | Multi-tenant training, batch inference | VDI, data science VMs, dev/test |
| **Management** | nvidia-smi, Kubernetes device plugin | vGPU Manager, vCenter |
| **GPU Models** | A100, A30, H100 | A2, A10, A16, A30, A40 |

**Exam Trap:** MIG requires containers/bare-metal. If question mentions VMs or live migration → vGPU.

---

## GPU Power Consumption (TDP)

| GPU Model | TDP | Typical Use Case |
|-----------|-----|------------------|
| A100 | 350W | Training, HPC |
| H100 | 700W | Large model training, HPC |
| A30 | 165W | Multi-instance inference |
| A10 | 150W | Mixed AI + graphics |
| L40 | 300W | Training + rendering |

**Rack Power Calculation:**
- 8× A100 = 2.8 kW (GPUs only)
- 8× H100 = 5.6 kW (GPUs only)
- Add system overhead: +40-50% (CPUs, storage, networking, fans)
- Total rack with 8× H100 ≈ 8-10 kW
- Account for PSU efficiency: Divide by 0.8 (80% efficient PSU)
- **Final draw:** 10 kW ÷ 0.8 = **12.5 kW at the PDU**

---

## Cooling Capacity Calculations

**Key Conversions:**
- 1 kW = 3,412 BTU/hr
- All electrical power becomes heat (conservation of energy)

**Example:**
- Rack with 10 kW power draw
- Cooling needed: 10 × 3,412 = **34,120 BTU/hr**

**Cooling Technology Thresholds:**
- <15 kW/rack → Air cooling sufficient
- 15-25 kW/rack → Hot-aisle containment + high-efficiency air
- >25 kW/rack → Liquid cooling (rear-door heat exchangers or direct-to-chip)

**Air vs Liquid Comparison:**
| Factor | Air Cooling | Liquid Cooling |
|--------|-------------|----------------|
| Max Density | ~15 kW/rack | 50+ kW/rack |
| Efficiency | 1.3-1.5 PUE | 1.1-1.2 PUE |
| Cost | Lower CapEx | Higher CapEx, lower OpEx |
| Maintenance | Simple | Complex, leak risk |

---

## Capacity Planning Formulas

### Training Workloads
```
training_time = dataset_size / (GPU_count × throughput_per_GPU)

GPU_count = dataset_size / (target_time × throughput_per_GPU)
```

**Example:**
- Dataset: 500 TB
- Per-GPU throughput: 2 TB/hr
- Target time: 10 hours
- GPUs needed: 500 / (10 × 2) = **25 GPUs**

### Inference Workloads
```
concurrent_streams = requests_per_sec / (1 / latency_per_request)

GPU_count = target_RPS / current_RPS_per_GPU
```

**Example 1 (Throughput Scaling):**
- Current: 1 GPU handles 1,000 req/s
- Target: 5,000 req/s
- GPUs needed: 5,000 / 1,000 = **5 GPUs**

**Example 2 (Latency to Streams):**
- Target: 1,000 req/s
- Latency: 50 ms (0.05s) per request
- Streams: 1,000 / (1 / 0.05) = 1,000 / 20 = **50 concurrent streams**

---

## MIG Configuration Profiles

**A100 (80GB) MIG Profiles:**
- 1× 7g.80gb (full GPU)
- 2× 3g.40gb
- 3× 2g.20gb
- 7× 1g.10gb

**Command to create MIG instance:**
```bash
sudo nvidia-smi mig -cgi 9,9,9 -C  # Create 3× 2g.20gb instances
```

**Exam Note:** Max 7 instances only with 1g profile. Mixing profiles reduces total count.

---

## Common Exam Traps

1. **MIG + VMs:** MIG does NOT support VMs. If question has VMs + isolation → vGPU.
2. **Power ≠ Cooling:** Use INPUT power (at PDU) for cooling, not GPU TDP alone.
3. **Linear Scaling Assumption:** Exam often states "linear scaling"—don't add overhead unless specified.
4. **BTU Conversion:** Always use 3,412 factor, not 1,000 or other invented numbers.
5. **PSU Efficiency:** 80% efficient means 20% becomes heat. Cooling needs to remove ALL input power.
6. **vGPU Instance Limits:** A30 supports 24× vGPU VMs, not unlimited. Know this limit.
