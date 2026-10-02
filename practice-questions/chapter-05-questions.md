## Chapter 5: GPU Virtualization, Multi-Tenancy, and Capacity Planning — Practice Questions

**Q1.** A research team needs to run 7 independent AI training jobs simultaneously on a single GPU, each requiring strict QoS isolation and dedicated GPU memory. The jobs are containerized using Docker. Which solution BEST meets these requirements?

A) Deploy NVIDIA vGPU with 7 virtual machines, each running one container
B) Configure NVIDIA MIG on an A100 GPU with 7 instances, one per container
C) Use time-slicing to share a single GPU across 7 containers without partitioning
D) Deploy 7 separate A30 GPUs in passthrough mode, one per container

<details>
<summary>Answer</summary>

**Correct: B**

MIG provides hardware-level partitioning of A100/H100 into up to 7 isolated instances with guaranteed memory and compute resources. Perfect for containerized workloads requiring QoS isolation without VM overhead.

**Why the others are wrong:**
- vGPU requires VMs (hypervisor layer), adding overhead. Designed for VDI/virtualized workloads, not optimal for containers needing hardware isolation.
- Time-slicing provides scheduling but NO memory isolation or QoS guarantees. Jobs compete for resources and can interfere with each other.
- Massively overprovisioned and expensive. MIG allows single GPU to serve 7 workloads with hardware isolation, eliminating need for 7 physical GPUs.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A VDI deployment needs to provide GPU acceleration to 40 knowledge workers running CAD applications. Each user needs 2GB GPU memory and occasional compute bursts. Which solution is MOST cost-effective?

A) Configure MIG on 6 A100 GPUs to create 40 isolated instances
B) Assign 40 T4 GPUs in passthrough mode, one per user VM
C) Use time-slicing on 4 A100 GPUs without virtualization
D) Deploy NVIDIA vGPU with 2 A30 GPUs, sharing across 40 VMs using vGPU profiles

<details>
<summary>Answer</summary>

**Correct: D**

vGPU is designed specifically for VDI workloads with VM-based users. A30 supports up to 24 vGPU instances per GPU, so 2 GPUs handle 40 users efficiently. Provides dynamic resource allocation for bursty workloads.

**Why the others are wrong:**
- MIG requires containers or bare-metal, NOT VMs. Designed for AI training isolation, not VDI flexibility. A100 is overkill for CAD workloads and more expensive than A30.
- 40 physical GPUs for light VDI usage is 20x more expensive than needed. Passthrough prevents sharing, wasting GPU resources during idle periods.
- Time-slicing without virtualization doesn't provide VM isolation or management features needed for VDI. No support for live migration or vMotion.

</details>

---

**Q3.** A data center is planning a rack with 8 H100 GPUs for AI training. Assuming 700W TDP per H100 GPU and 80% power supply efficiency, what is the MINIMUM power capacity required at the rack level?

A) 5.6 kW (8 GPUs × 700W)
B) 7.0 kW (8 GPUs × 700W ÷ 0.8)
C) 10.5 kW (includes CPU, networking, storage overhead)
D) 14.0 kW (2× safety factor for peak loads)

<details>
<summary>Answer</summary>

**Correct: C**

H100 draw: 8 × 700W = 5.6kW. With 80% PSU efficiency: 5.6kW ÷ 0.8 = 7.0kW. But exam tests TOTAL rack power including dual CPUs (~500W), networking (~200W), storage (~300W), fans (~200W). Realistic minimum: ~10.5kW. Always plan for system-level power, not GPU-only.

**Why the others are wrong:**
- Only accounts for GPU TDP at 100% efficiency. Ignores PSU losses (typically 15-20%) and all other system components.
- Accounts for PSU efficiency but ignores CPU, memory, storage, networking, and cooling fans. Exam expects full-system calculations.
- Excessive overprovisioning. While safety margins are good practice, 2× is wasteful. Industry standard is 1.2-1.3× for peak loads, already included in the 10.5kW estimate.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** An AI training cluster generates 42 kW of heat per rack. The facility uses air cooling with hot-aisle containment. What is the approximate cooling capacity required in BTU/hr for a single rack?

A) 42,000 BTU/hr (1 kW = 1,000 BTU/hr)
B) 143,304 BTU/hr (42 kW × 3,412 BTU/hr per kW)
C) 85,982 BTU/hr (using 50% cooling efficiency factor)
D) 200,000 BTU/hr (2× safety factor for peak summer loads)

<details>
<summary>Answer</summary>

**Correct: B**

Standard conversion: 1 kW = 3,412 BTU/hr. For 42 kW: 42 × 3,412 = 143,304 BTU/hr. This is the heat removal requirement. Exam tests this exact conversion formula for cooling capacity planning.

**Why the others are wrong:**
- Incorrect conversion factor. 1 kW ≠ 1,000 BTU/hr. This is off by factor of 3.4×, leading to severe under-cooling and thermal throttling.
- No such thing as 'cooling efficiency factor' reducing heat output. All electrical power becomes heat (conservation of energy). 42 kW in = 42 kW heat out = 143,304 BTU/hr.
- While safety margins are prudent, exam expects exact calculation. Overprovisioning by 2× wastes HVAC capacity and energy. Typical safety factor is 1.1-1.2×.

</details>

---

**Q5.** A deep learning team needs to train a model on a 500TB dataset. Each A100 GPU processes 2TB/hour during training. The team wants to complete training in 10 hours. How many A100 GPUs are required?

A) 50 GPUs (500TB ÷ 10hr ÷ 1TB/hr with scaling overhead)
B) 100 GPUs (accounting for 50% distributed training efficiency)
C) 25 GPUs (500TB ÷ 2TB/hr ÷ 10hr)
D) 250 GPUs (500TB ÷ 2TB/hr total throughput needed)

<details>
<summary>Answer</summary>

**Correct: C**

Formula: GPU_count = dataset_size / (throughput_per_gpu × target_time). Calculation: 500TB / (2TB/hr × 10hr) = 500TB / 20TB = 25 GPUs. This assumes near-linear scaling, reasonable for data-parallel training on modern frameworks.

**Why the others are wrong:**
- Incorrectly reduces per-GPU throughput to 1TB/hr. Problem states 2TB/hr per GPU. Also invented 'scaling overhead' not mentioned in scenario.
- Assumes only 50% distributed training efficiency (2× more GPUs). Modern frameworks (PyTorch FSDP, DeepSpeed) achieve 90%+ efficiency. Massive overprovisioning.
- Confuses total throughput needed (500TB / 10hr = 50TB/hr) with GPU count. Forgot to divide by per-GPU throughput (2TB/hr). Math error leads to 10× overprovisioning.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** An inference service receives 1,000 requests/second with 50ms average latency per request on a single GPU. The service needs to scale to 5,000 requests/second. Assuming linear scaling, how many GPUs are required?

A) 10 GPUs (accounting for 50% overhead in load balancing)
B) 4 GPUs (1 / 0.05s = 20 concurrent streams per GPU, 5,000 / 1,000 = 5, but 4 GPUs × 1.25× = 5×)
C) 5 GPUs (5,000 / 1,000 requests handled by 1 GPU)
D) 2 GPUs (50ms latency = 20 requests/sec per stream, 1,000 / 20 = 50 streams, double for safety)

<details>
<summary>Answer</summary>

**Correct: C**

Current capacity: 1 GPU handles 1,000 req/s. Target: 5,000 req/s. Simple scaling: 5,000 / 1,000 = 5 GPUs needed. Problem states 'linear scaling', so no complex calculations needed. Exam tests straightforward proportional scaling.

**Why the others are wrong:**
- Invents 50% overhead not mentioned in problem. With linear scaling assumption, 5 GPUs is exact answer. Adding arbitrary safety factors contradicts the scenario.
- Overcomplicated calculation mixing concurrent streams formula with simple scaling. Result doesn't even match the math shown. Confused approach leads to under-provisioning.
- Completely wrong approach. Confuses latency with throughput. 1 GPU already handles 1,000 req/s, so 2 GPUs would only handle 2,000 req/s, not 5,000 req/s needed.

</details>

---

**Q7.** A company is choosing between MIG and vGPU for a new AI platform. They need to support both containerized training jobs requiring QoS isolation AND VM-based data science workloads with live migration support. Which solution is BEST?

A) Use MIG exclusively for both containerized and VM workloads
B) Use vGPU exclusively for both containerized and VM workloads
C) Deploy hybrid setup: MIG for containers, vGPU for VMs on separate GPU pools
D) Use passthrough mode for containers, time-slicing for VMs

<details>
<summary>Answer</summary>

**Correct: C**

MIG provides hardware isolation ideal for containers but doesn't support VMs or live migration. vGPU supports VMs and vMotion but lacks hardware-level QoS for training. Hybrid deployment uses each technology for its strengths. This is a real-world best practice for mixed workloads.

**Why the others are wrong:**
- MIG cannot be used with VMs in most configurations and doesn't support live migration. Would fail VM workload requirements.
- vGPU provides software-level sharing, not hardware isolation. Training jobs would lack guaranteed QoS. Can work but suboptimal for training workloads that need predictable performance.
- Passthrough prevents sharing and live migration. Time-slicing lacks QoS isolation. Neither technology addresses the stated requirements properly.

</details>

---

**Q8.** A rack contains 10 servers, each with 4 A100 GPUs (350W TDP each). The facility is considering liquid cooling vs air cooling. Assuming 100% GPU utilization, what is the total heat load from GPUs only?

A) 14.0 kW (10 servers × 4 GPUs × 350W)
B) 3.5 kW (10 servers × 350W)
C) 17.5 kW (14.0 kW × 1.25 PSU inefficiency)
D) 47.7 kBTU/hr (14.0 kW × 3,412 BTU/hr per kW)

<details>
<summary>Answer</summary>

**Correct: A**

Heat from GPUs = total GPU power draw. Calculation: 10 servers × 4 A100s × 350W = 14,000W = 14.0 kW. Question asks for 'heat load from GPUs only', so we report power draw, not BTU or total system power. All electrical power becomes heat.

**Why the others are wrong:**
- Only counts 1 GPU per server (10 × 350W). Missed the '4 GPUs' specification. Math error leads to 4× under-estimation.
- Adds PSU inefficiency to GPU heat output. PSU losses ARE additional heat but question specifically asks for 'GPUs only'. Also, PSU inefficiency doesn't multiply GPU heat—it adds to it.
- Converts to BTU/hr (47.7k BTU/hr is correct conversion of 14 kW), but question asks for heat load without specifying units. When not specified and comparing cooling technologies, kW is standard. Not wrong but not the expected answer format.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

