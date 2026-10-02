# Answer Key — Practice Test 1

**Q1. Correct: D**

Recall (True Positives / (True Positives + False Negatives)) directly measures the proportion of actual defects the model successfully detects. With severe class imbalance (100 defects vs 10,000 normal), a model can achieve 95% accuracy by predicting 'normal' for everything while missing all defects. Class-specific recall reveals this failure mode. For rare-event detection, recall on the minority class is the critical metric.

---

**Q2. Correct: C**

NVIDIA NGC (NVIDIA GPU Cloud) catalog is the official source for production-ready, GPU-optimized containers including TensorRT. NGC containers are pre-configured with correct CUDA, cuDNN, and TensorRT versions, tested for compatibility with NVIDIA GPUs, security-scanned, and include performance optimizations. For enterprise deployments, NGC is the recommended and supported path.

---

**Q3. Correct: D**

GPUs contain thousands of smaller, energy-efficient cores designed for massive parallelism, whereas CPUs have fewer (tens) powerful cores optimized for sequential tasks. Training neural networks involves matrix operations (multiplications, additions) performed on millions/billions of parameters simultaneously. GPUs excel at these embarrassingly parallel workloads. For a 175B parameter model, the ability to process thousands of matrix operations in parallel across tensor cores provides 10-100x speedup over CPUs.

---

**Q4. Correct: D**

TensorRT is NVIDIA's inference optimization engine that takes trained models (from TensorFlow, PyTorch, etc.) and optimizes them for production deployment. It performs layer fusion, precision calibration (FP32→FP16→INT8), kernel auto-tuning, and graph optimization to minimize latency and maximize throughput on NVIDIA GPUs. For a 200ms latency requirement in production inference, TensorRT is the correct tool—it can reduce latency by 2-10x compared to unoptimized models.

---

**Q5. Correct: A**

Healthcare deployments require data locality (HIPAA/patient privacy), low-latency inference (radiologists need real-time feedback), explainability (medical liability and regulatory approval), and reliability (24/7 availability). Edge deployment on hospital servers with NVIDIA-certified models addresses all requirements: data stays on-premises (compliance), GPUs provide real-time inference, medical-grade models include explainability features (saliency maps, attention visualization), and local deployment ensures availability independent of internet connectivity.

---

**Q6. Correct: C**

With severe class imbalance (0.5% fraud), precision-recall curves are more informative than ROC curves because they focus on the minority (positive) class performance. ROC curves can be overly optimistic for imbalanced data because true negative rate (specificity) is high by default when negatives dominate. A higher PR-AUC (0.72 vs 0.68) indicates Model A achieves better precision-recall trade-offs for actually detecting fraud. In fraud detection with 0.5% base rate, the model that better identifies actual fraud cases (higher PR-AUC) is more valuable despite slightly lower ROC-AUC.

---

**Q7. Correct: A**

TensorRT is NVIDIA's inference optimizer that takes trained models and applies graph optimizations (layer fusion, kernel auto-tuning), precision calibration (FP16/INT8), and generates optimized runtime engines specifically for low-latency inference workloads. The <50ms latency requirement is an inference optimization problem, which is TensorRT's core purpose.

---

**Q8. Correct: A**

Data parallelism requires all-reduce operations to synchronize gradients across GPUs after each batch. With 8 GPUs, the communication overhead becomes the dominant factor if network bandwidth (interconnect like NVLink or InfiniBand) cannot sustain the gradient transfer volume. The drop from 85% to 23% per-GPU utilization is characteristic of communication bottlenecks in distributed training, where GPUs spend more time waiting for gradient synchronization than computing.

---

**Q9. Correct: B**

Recall (sensitivity) measures True Positives / (True Positives + False Negatives), which directly quantifies how many actual fraud cases are caught. A Recall of 0.94 means the model identifies 94% of real fraud cases, minimizing false negatives as required. This is the metric that directly aligns with the business requirement of 'minimizing missed fraud cases.'

---

**Q10. Correct: C**

NVIDIA vGPU software is the virtualization platform that enables GPU sharing in virtualized environments like VMware vSphere. It provides time-slicing (for older GPUs) or MIG (Multi-Instance GPU) for A100/H100 to partition physical GPUs into isolated instances that VMs can access. This is a core component of NVIDIA AI Enterprise for virtualized infrastructure.

---

**Q11. Correct: C**

A PR curve where precision drops sharply as recall increases indicates that the model makes confident, accurate predictions for clear cases (high precision at low recall), but as the threshold is lowered to catch more defects (increase recall), it starts making less confident predictions with more false positives. This pattern suggests the model handles obvious defects well but struggles with borderline cases, indicating need for better decision boundaries through retraining or threshold optimization.

---

**Q12. Correct: B**

The fundamental architectural difference is that GPUs have thousands of simpler cores designed for massive parallelism (SIMT: Single Instruction, Multiple Threads), while CPUs have fewer, more complex cores. Deep learning operations like matrix multiplications and convolutions are highly parallelizable - the same operation (multiply-accumulate) is performed on millions of data elements. GPUs excel at this data-parallel workload, achieving 50-100x speedup by executing thousands of operations simultaneously.

---

**Q13. Correct: A**

cuDNN (CUDA Deep Neural Network library) provides highly optimized implementations of standard deep learning primitives like convolutions, pooling, normalization, and activation layers. Without cuDNN, PyTorch falls back to its own generic CUDA kernels or CPU implementations for these operations, which are significantly slower. CNNs are particularly impacted because convolution operations are their core building block. The A100 GPU can still execute the workload but at a fraction of optimal performance.

---

**Q14. Correct: D**

TensorRT is NVIDIA's inference optimization SDK that specifically targets latency reduction through layer fusion, precision calibration (FP16/INT8), kernel auto-tuning, and dynamic tensor memory management. For inference workloads requiring sub-50ms latency, TensorRT can achieve 2-10x speedup over native framework execution by optimizing the model graph and generating highly efficient runtime engines. It is purpose-built for production inference scenarios where latency matters most.

---

**Q15. Correct: B**

Precision of 60% means 40% of flagged items are false positives. While each false positive costs only $100, the 98% recall means 2% false negatives - each costing $10,000. In this asymmetric cost scenario, the high precision problem (excessive false positives) indicates the model flags too many non-defects. However, the question asks which metric BEST indicates improvement is needed: the 60% precision directly shows the business impact ($100 per false alarm) and should be weighed against the cost structure. Given the 100x cost difference, improving precision to reduce false positives might seem counterintuitive, but the metric itself (60% precision) clearly shows significant room for improvement in positive predictive value.

---

**Q16. Correct: A**

Modern NVIDIA GPUs (Volta, Turing, Ampere, Hopper architectures) contain thousands of CUDA cores plus specialized Tensor Cores specifically designed for mixed-precision matrix operations. An A100 GPU has 6912 CUDA cores and 432 third-generation Tensor Cores. Matrix multiplication is embarrassingly parallel - an 8192x8192 multiplication requires billions of independent multiply-accumulate operations that can execute simultaneously. Tensor Cores can perform matrix operations at 312 TFLOPS (FP16) on A100 versus typical CPUs at 1-2 TFLOPS. The massive parallelism, not clock speed or cache, drives the 100x+ performance advantage for this workload.

---

**Q17. Correct: D**

NVIDIA AI Enterprise includes access to the NGC (NVIDIA GPU Cloud) enterprise registry, which provides curated, security-scanned, and validated container images specifically certified for AI Enterprise. These containers include pre-integrated frameworks (PyTorch, TensorFlow, TensorRT), libraries (CUDA, cuDNN), and undergo regular security scanning and updates. The NGC enterprise registry is separate from the public NGC catalog and includes additional validation, long-term support, and service-level agreements. This is a key value proposition of AI Enterprise versus using public container images.

---

**Q18. Correct: D**

DGX H100 systems use NVLink 4.0 with fourth-generation NVSwitch, providing 900 GB/s of bidirectional bandwidth per GPU. This creates a fully connected topology where every GPU can communicate with every other GPU at maximum bandwidth, essential for model parallelism in large model training.

---

**Q19. Correct: B**

MIG on A100 provides hardware-level isolation with guaranteed memory and compute resources. A100 80GB supports configurations like 1x MIG 3g.20gb instance and 2x MIG 2g.10gb instances, providing complete isolation with dedicated SM, memory, and cache partitions for each tenant—critical for multi-tenancy security and QoS.

---

**Q20. Correct: B**

InfiniBand NDR (400 Gb/s) with RDMA and GPUDirect RDMA enables direct GPU-to-GPU transfers across nodes, bypassing CPU and reducing latency for collective operations like all-reduce. For 256-GPU training with frequent synchronization, the lower latency (sub-microsecond) and higher efficiency of InfiniBand provides 30-40% faster training compared to Ethernet alternatives.

---

**Q21. Correct: A**

Parallel file systems like Lustre or GPFS with NVMe SSDs provide the aggregate bandwidth (50GB/s+), low latency, and concurrent access patterns required for distributed training. Multiple storage servers with NVMe deliver sustained high throughput, and parallel architecture prevents bottlenecks when 100+ jobs read simultaneously. This is the standard for large-scale AI training infrastructure.

---

**Q22. Correct: A**

16 DGX H100 systems × 10.2 kW = 163.2 kW maximum power. AI training workloads frequently hit 100% GPU utilization simultaneously across all systems. Power delivery must support maximum load (not average), and cooling must match because nearly 100% of power converts to heat. Undersizing causes thermal throttling, reducing training performance by 20-40%.

---

**Q23. Correct: B**

DGX H100 provides maximum training performance with NVLink/NVSwitch for multi-GPU workloads. L40S GPUs are optimized for inference with lower power (350W vs 700W), PCIe connectivity (sufficient for inference), and 50-60% lower cost while delivering excellent inference throughput. This workload-specific deployment optimizes $/performance for each use case—the standard enterprise pattern.

---

**Q24. Correct: D**

MIG can be selectively enabled per GPU in the system. This approach correctly allocates 2 full GPUs without MIG for Team A (maximum performance), enables MIG on multiple GPUs to create fractional instances for Team B's inference needs, and reserves a dedicated GPU with namespace/container isolation for Team C's compliance requirements. DGX H100 has 8 GPUs total, making this distribution feasible. MIG provides hardware-level isolation when enabled.

---

**Q25. Correct: D**

NVIDIA Quantum-2 InfiniBand with 400Gb/s NDR (Next Data Rate) provides the required 400GB/s bidirectional bandwidth per adapter. Using 8 adapters per node in a rail-optimized fat-tree topology ensures non-blocking bandwidth and minimizes latency. GPUDirect RDMA allows direct memory access between GPUs across nodes, bypassing CPU and achieving the ultra-low latency critical for distributed training with 256 GPUs. This is the standard configuration for large-scale NVIDIA AI clusters.

---

**Q26. Correct: C**

Total maximum DGX power is 16 × 10.2kW = 163.2kW. With 95% PDU efficiency, the utility must provide 163.2 ÷ 0.95 = 172kW to deliver this to the DGX systems. N+1 redundancy for DGX systems means supporting one additional system failure/addition, adding 10.2kW capacity. Total minimum: 172 + 10.2 = 182.2kW. You must plan for maximum power draw, not average utilization, as training jobs can spike to full power. The question asks for utility feed capacity, which must account for PDU losses but not cooling (that's separate infrastructure).

---

**Q27. Correct: B**

NVIDIA AI Enterprise with vGPU integrates natively with VMware vSphere, allowing use of existing management tools (vCenter, vMotion, DRS). Q-series vGPU profiles are specifically designed for inference workloads with optimized graphics/compute balance and lower latency than compute-only profiles. vGPU allows multiple VMs per physical GPU, enabling cost-effective scaling from 10 to 1000 requests by adjusting VM count. This meets the sub-20ms latency requirement while leveraging existing VMware infrastructure investments.

---

**Q28. Correct: C**

A parallel file system like Lustre or GPFS is designed for high-concurrency random access from many GPU nodes simultaneously. NVMe SSDs provide the IOPS needed for random data access patterns during training (HDDs would bottleneck). 2PB capacity handles the 500TB dataset plus checkpoints with room for growth. Most critically, 400Gb/s InfiniBand RDMA connectivity matches the GPU cluster's low-latency interconnect, preventing storage from becoming a bottleneck. This architecture is standard for large-scale training clusters.

---

**Q29. Correct: A**

The hybrid model addresses all requirements optimally: 32 on-premises GPUs handle continuous baseline workloads without the high capital cost of 256 GPUs, avoiding CapEx concerns while providing owned infrastructure for predictable workloads. Cloud bursting provides flexibility to scale to 256 GPUs when needed without long-term capacity planning or sitting idle. For 2-4 week training jobs, reserved or on-demand cloud instances are cost-effective for temporary scale. This balances cost, flexibility, and avoids over-provisioning on-premises infrastructure.

---

**Q30. Correct: A**

DGX H100 systems feature 4th generation NVLink with 900 GB/s bidirectional bandwidth per GPU (18 links × 25 GB/s per direction). This is critical for distributed training performance as it determines how quickly gradients and activations can be exchanged between GPUs during backpropagation.

---

**Q31. Correct: A**

InfiniBand HDR with SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) provides the lowest latency (~1 microsecond), highest throughput (200 Gb/s), and hardware-accelerated collective operations critical for large-scale synchronous training. SHARP offloads all-reduce operations to the network switches, reducing GPU and CPU overhead by up to 5x for communication-intensive workloads.

---

**Q32. Correct: B**

The 2 TB/s aggregate throughput requirement is the critical constraint. NVMe SSDs in a parallel filesystem (like BeeGFS or Lustre with NVMe OSTs) can deliver the sustained sequential read bandwidth needed to keep GPUs fed. GPU starvation occurs when storage cannot deliver data fast enough, causing GPUs to idle waiting for batches—the most expensive bottleneck in AI training.

---

**Q33. Correct: C**

MIG on H100 allows partitioning each GPU into up to 7 instances with dedicated SM slices, memory, and memory bandwidth. This provides hardware-level isolation and guaranteed QoS—critical for Team A's inference latency SLAs—while allowing efficient multi-tenancy. Each MIG instance appears as a separate GPU to the OS, eliminating noisy neighbor issues that affect latency-sensitive workloads.

---

**Q34. Correct: D**

Total peak load is 8 × 10.2 kW = 81.6 kW. At 208V three-phase, power = √3 × V × I, so 81.6 kW requires minimum 226A. Circuit breakers should operate at 80% max continuous load (NEC), so minimum breaker rating is 226A / 0.8 = 283A. Four 50A three-phase circuits provide 4 × 50A = 200A total capacity × 0.8 = 160A continuous, which is insufficient. However, the question states "one per two DGX units" meaning each 50A circuit handles 2 × 10.2 kW = 20.4 kW, requiring 56.5A at 208V 3-phase, which is within the 80% rule for a 50A breaker (40A continuous max). This provides proper distribution and redundancy.

---

**Q35. Correct: C**

A dedicated 10 Gbps link provides 10x bandwidth improvement, reducing transfer time from 6+ hours to approximately 36 minutes for a 250 GB model (theoretical: 250 GB × 8 / 10 Gbps = 200 seconds, real-world ~30-40 min with overhead). Direct Connect/ExpressRoute also provides consistent latency, bypasses public internet congestion, and establishes private encrypted connectivity without VPN overhead, meeting security requirements while dramatically improving transfer performance.

---

**Q36. Correct: B**

Temperature cycling between 45°C and 83°C with performance drops at 80°C indicates thermal throttling caused by inadequate cooling. DGX systems require proper datacenter cooling (18-27°C ambient) and unobstructed airflow. Thermal throttling is the GPU's protective mechanism when cooling is insufficient—fixing the root cause (cooling infrastructure) is the correct operational approach.

---

**Q37. Correct: C**

DCGM is NVIDIA's purpose-built GPU monitoring solution for datacenters. The dcgm-exporter component provides native Prometheus integration, collects comprehensive metrics (utilization, memory, power, temperature, health), and is designed for low overhead on production systems. This is the standard solution for production GPU monitoring.

---

**Q38. Correct: B**

For Slurm to recognize `--gres=gpu:X` requests, GresTypes must be configured in slurm.conf (e.g., `GresTypes=gpu`). Without this, Slurm doesn't understand the GPU resource type and cannot allocate GPUs, causing jobs to pend indefinitely even when GPUs are physically available. This is a common configuration oversight.

---

**Q39. Correct: D**

The model requires 24GB but memory usage reaches 78GB—more than 3x expected. This indicates a memory leak, most commonly from gradient accumulation implementation errors where activations aren't properly freed between accumulation steps. Checking for this specific bug addresses the root cause: why memory consumption far exceeds model requirements.

---

**Q40. Correct: A**

Fair-share scheduling guarantees each team 50% of resources when competing, but allows teams to use idle resources beyond their share when available (bursting to 100%). With preemption disabled, running jobs aren't interrupted when other teams submit work—new jobs wait for resources to free naturally. This matches the requirement exactly.

---

**Q41. Correct: B**

GPUs show as allocatable but pods can't use them—this indicates the device plugin detects GPUs but containers can't access them. The nvidia-container-runtime must be configured as the container runtime to expose GPUs to containers. Without this, even though Kubernetes sees GPUs, containers cannot access them, causing scheduling failures.

---

**Q42. Correct: D**

Clock throttling at low temperatures with low utilization is typically caused by power limit constraints, not thermal issues. The power cap may be set below the default TDP, causing performance throttling even when thermal headroom exists. Checking nvidia-smi -q -d POWER reveals current vs. default power limits and whether power capping is the root cause.

---

**Q43. Correct: C**

PriorityClasses with preemption allow interactive pods (high priority) to evict lower-priority batch jobs when GPU resources are needed, ensuring immediate access. Batch jobs still run when resources are available. Resource quotas prevent any workload type from monopolizing the cluster. This is the standard Kubernetes pattern for mixed-criticality workloads.

---

**Q44. Correct: C**

In distributed training with DDP, if dataset sharding is not perfectly balanced, some ranks may receive more samples per iteration than others. This causes those specific GPUs to require more memory for activations and gradients, leading to OOM on affected nodes while others with smaller shard sizes run normally. The fact that only 3 specific nodes fail (not random) indicates a data distribution issue.

---

**Q45. Correct: A**

This configuration addresses all requirements: model versioning with policies enables safe weekly updates; liveness/readiness probes ensure Kubernetes routes traffic only to healthy pods (uptime SLA); DCGM exporter provides comprehensive GPU metrics; rolling updates with maxUnavailable=0 ensures zero downtime during deployments. This is production-grade MLOps for inference services.

---

