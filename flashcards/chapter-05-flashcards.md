## Chapter 5: GPU Virtualization, Multi-Tenancy, and Capacity Planning — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | NVIDIA MIG maximum number of instances per GPU? | 7 instances (on A100/H100). Exam tip: MIG partitions ONE physical GPU into up to 7 isolated instances with dedicated memory and compute. Used for multi-tenant AI training, not VDI. |
| 2 | MIG vs vGPU: when to use MIG? | Use MIG when: containers (not VMs), need hardware QoS isolation, multi-tenant AI training, fixed resource allocation. Exam trap: MIG doesn't support VMs or live migration—that's vGPU territory. |
| 3 | MIG vs vGPU: when to use vGPU? | Use vGPU when: VM-based workloads, need live migration/vMotion, VDI deployments, dynamic resource allocation. Exam trap: vGPU provides software sharing, NOT hardware isolation like MIG. |
| 4 | NVIDIA A100 GPU TDP (power consumption)? | 350W. Exam tip: H100 is 700W (2× A100). Use for rack power planning: 8× A100 = 2.8kW GPU power. Always add system overhead (CPU, storage, networking). |
| 5 | NVIDIA H100 GPU TDP (power consumption)? | 700W. Exam tip: Double the A100 (350W). Critical for power/cooling calculations. 8× H100 = 5.6kW just for GPUs. Total rack can exceed 10kW with all components. |
| 6 | kW to BTU/hr conversion formula? | 1 kW = 3,412 BTU/hr. Exam formula: cooling_BTU = power_kW × 3,412. Example: 10 kW rack = 34,120 BTU/hr cooling needed. Used for HVAC sizing. |
| 7 | Training time capacity planning formula? | training_time = dataset_size / (GPU_count × throughput_per_GPU). Solve for GPU_count: dataset / (target_time × throughput). Exam gives 3 variables, you solve for 4th. |
| 8 | Inference capacity planning: GPU count formula? | GPU_count = target_requests_per_sec / current_requests_per_sec_per_GPU. If 1 GPU handles 1,000 req/s and you need 5,000 req/s → 5 GPUs. Assumes linear scaling. |
| 9 | Concurrent streams formula for inference sizing? | concurrent_streams = requests_per_second / (1 / latency_per_request). Example: 1,000 req/s ÷ (1/0.05s) = 50 concurrent streams needed. Exam tests this with different latencies. |
| 10 | Air cooling vs liquid cooling decision criteria? | Air: <15kW/rack, lower cost, simpler maintenance. Liquid: >15kW/rack, 30-50% better efficiency, required for dense GPU racks (8+ H100s). Exam trap: liquid not always better—depends on density. |
| 11 | Hot-aisle containment purpose in GPU data centers? | Isolates hot exhaust air from equipment intake. Prevents hot air recirculation, allows higher intake temps, improves cooling efficiency. Exam: know this is standard for high-density AI racks (>10kW). |
| 12 | Power supply efficiency impact on cooling? | 80% efficient PSU: 20% energy becomes heat (not delivered to components). 10kW load = 12.5kW drawn, 12.5kW cooling needed. Exam: always use input power (drawn) for cooling calculations, not output power. |
