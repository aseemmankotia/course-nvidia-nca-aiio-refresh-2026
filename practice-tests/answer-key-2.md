# Answer Key — Practice Test 2

**Q1. Correct: C**

TensorRT is specifically designed for inference optimization. It converts trained models (from PyTorch, TensorFlow, etc.) into optimized inference engines using layer fusion, precision calibration (FP16/INT8), and kernel auto-tuning to maximize throughput and minimize latency. For real-time inference with strict latency requirements, TensorRT is the critical component.

---

**Q2. Correct: C**

When business costs of false positives are high, precision is the key metric. The PR curve plots precision vs recall at different classification thresholds. By adjusting the threshold (typically increasing it above 0.5 for binary classification), you can increase precision at the cost of recall. The PR curve helps visualize this trade-off and select the threshold that meets business requirements for acceptable precision while maximizing recall.

---

**Q3. Correct: D**

The fundamental architectural advantage of GPUs for AI workloads is massive parallelism. Modern NVIDIA GPUs have thousands of CUDA cores (e.g., A100 has 6912 CUDA cores) designed to execute the same instruction on multiple data points simultaneously (SIMD). Transformer training involves massive matrix multiplications and element-wise operations that are perfectly suited for this parallel architecture. The 8x larger batch size (256 vs 32) combined with parallel execution explains the performance gain.

---

**Q4. Correct: D**

NVIDIA AI Enterprise is the commercial software suite that includes enterprise-grade support, security updates, and validated configurations. NGC containers tagged with AI Enterprise have undergone additional security scanning and are covered by NVIDIA enterprise support SLAs. The security scan results provide CVE information and vulnerability assessments critical for enterprise compliance. This combination ensures both support and security requirements are met.

---

**Q5. Correct: D**

Training memory requirements are significantly higher than inference. During training, GPU memory must hold: (1) model parameters, (2) gradients for each parameter, (3) optimizer states (momentum, variance for Adam), and (4) intermediate activations for backpropagation, plus (5) the training data batch. Inference only requires model parameters and the inference batch (much smaller than training batches). This is why a model trainable on 80GB A100 might serve inference on 16GB T4.

---

**Q6. Correct: A**

In medical contexts where missing a diagnosis (false negative) has severe consequences, high recall is critical and the 0.92 is acceptable for 'minimizing missed diagnoses'. However, 0.68 precision means 32% of positive predictions are false positives, creating alert fatigue and workflow inefficiency. Raising the classification threshold (e.g., from 0.5 to 0.7) will decrease false positives (improving precision) at a small cost to recall. This balances regulatory safety requirements with clinical workflow efficiency. The PR curve helps find this optimal threshold.

---

**Q7. Correct: B**

TensorRT is NVIDIA's inference optimizer that provides maximum throughput through layer fusion, precision calibration, and kernel auto-tuning. For deterministic results required by regulatory compliance, TensorRT's strict type constraints prevent automatic precision conversion that could cause non-deterministic behavior. FP16 mode provides 2-4x throughput over FP32 while maintaining accuracy for financial workloads. Builder optimization level 5 enables most aggressive optimizations while strict types ensure reproducibility.

---

**Q8. Correct: D**

Large gap between ROC-AUC (0.93) and PR-AUC (0.81) is the signature pattern of class imbalance. ROC curves can appear optimistic on imbalanced datasets because they use true negative rate (which dominates when negatives are abundant). PR curves focus on precision and recall—both measured against positives only—making them sensitive to imbalance. The 76% recall vs 89% precision suggests the model is conservative (fewer false positives than false negatives), typical when optimizing for accuracy on imbalanced data overfits to predicting the majority class. Medical imaging often has this pattern (rare diseases).

---

**Q9. Correct: B**

Total compute required: 12 cameras × 30 fps × 18.4 TFLOPS = 6,624 TFLOPS/sec (6.6 PFLOPS/sec) in FP32. This is impossible for cost-effective deployment. Modern NVIDIA GPUs (Ampere/Hopper onwards) deliver 8-20x more throughput in FP16/INT8 via Tensor Cores compared to FP32 CUDA cores. CNN inference typically tolerates FP16 with <1% accuracy loss. Using Tensor Cores in FP16 mode reduces required throughput to ~800-1000 TFLOPS/sec, achievable on a single A100/H100. Tensor Cores are specifically designed for matrix operations in CNNs. This is why GPUs accelerate AI: specialized hardware (Tensor Cores) for common AI operations.

---

**Q10. Correct: D**

NVIDIA AI Enterprise is the commercial software suite providing: (1) Enterprise support with SLAs, (2) Security updates and CVE patching, (3) Version-locked, tested container stacks ensuring consistency, (4) Compatibility certification across on-prem/cloud (vSphere, bare metal, Kubernetes, cloud VMs), (5) NGC Private Registry for controlled deployment and air-gapped environments. This matches all requirements: consistent versions (locked containers), pre-configured frameworks (containers include PyTorch/TF/TensorRT), security patching (enterprise updates), cross-environment consistency (certified stacks), and support (subscription includes it). MLOps teams need this vs. free tools for production stability.

---

**Q11. Correct: D**

The fundamental reason GPUs accelerate AI is specialized hardware for dense matrix operations that dominate deep learning. GPUs have thousands of CUDA cores with FP32/FP16 ALUs plus Tensor Cores specifically designed for matrix multiply-accumulate operations. ResNet-152 training is dominated by convolutions (matrix operations) and backpropagation gradients (more matrix ops). Modern GPUs deliver 20-300 TFLOPS for these operations versus CPUs at 1-3 TFLOPS. This 50-100x compute advantage directly translates to training speedup. 18 days × 24 hours = 432 hours; reducing to <8 hours is 54x speedup—precisely what GPU specialized compute provides for neural network workloads.

---

**Q12. Correct: B**

Training vs inference workloads have fundamentally different optimization goals and constraints. Training: maximize accuracy, throughput (samples/sec), uses FP32/mixed precision, batch sizes of 32-256, runs in data centers with power/cooling. Inference: minimize latency per prediction, maximize throughput/watt, runs on edge devices (vehicles) with power/thermal limits, batch size typically 1-8 for real-time. The critical software shift is from training frameworks (PyTorch/TF using cuDNN) to TensorRT inference optimizer. TensorRT performs: layer fusion, kernel auto-tuning, precision calibration (FP16/INT8), reducing model size and compute 3-10x while maintaining accuracy. INT8 quantization cuts memory 4x and compute 4x versus FP32, critical for edge deployment. This is the essence of the training→inference transition.

---

**Q13. Correct: B**

Recall measures the proportion of actual positive cases correctly identified (true positives / [true positives + false negatives]). A recall of 0.78 means 22% of actual fraud cases are being missed (false negatives). For fraud detection where missing fraud is costly, recall must be prioritized even at the expense of precision. The business requirement explicitly states minimizing false negatives is critical.

---

**Q14. Correct: B**

TensorRT is NVIDIA's inference optimization SDK that provides layer fusion, kernel auto-tuning, dynamic tensor memory, and reduced precision (FP16/INT8) specifically designed to maximize inference throughput and minimize latency. Triton Inference Server provides production-grade features like dynamic batching (combining multiple requests to maximize GPU utilization), concurrent model execution, and model versioning. This combination directly addresses the low-latency inference requirement for LLM serving.

---

**Q15. Correct: A**

The fundamental architectural difference is that modern GPUs contain thousands of CUDA cores (e.g., NVIDIA A100 has 6,912 CUDA cores) designed for SIMT (Single Instruction Multiple Thread) execution, allowing massive parallelism for data-parallel workloads like neural network inference where the same operation is applied to many data points simultaneously. CPUs typically have 8-64 cores optimized for complex sequential logic with branch prediction and out-of-order execution. Medical image segmentation with 50M operations and high data parallelism is ideal for GPU's many-core architecture.

---

**Q16. Correct: A**

NVIDIA AI Enterprise is a software platform that provides enterprise-grade support, including guaranteed SLAs, CVE security patches, long-term support (LTS) versions, certified configurations with platforms like VMware vSphere and Red Hat OpenShift, and validated reference architectures. While it includes optimized versions of the same open-source frameworks available in NGC, the primary value proposition is the enterprise support, certification, and lifecycle management — critical for production deployments in regulated industries.

---

**Q17. Correct: D**

For highly imbalanced datasets (95:5 ratio), Precision-Recall curves are superior to ROC curves because they focus on the minority class performance. PR curves plot precision vs recall at different thresholds, directly showing the tradeoff between false positives (precision) and false negatives (recall) for the positive class. Average Precision (area under PR curve) provides a single metric that emphasizes minority class performance. A naive model predicting all negatives would achieve 94% accuracy and high ROC-AUC but would have terrible PR-AUC, revealing the true poor performance.

---

**Q18. Correct: A**

With 120kW available and 10.2kW per system, raw math allows ~11 systems. However, N+1 redundancy requires planning for one power supply failure per system, meaning each system needs dual feeds from separate PDUs. With redundancy overhead (typically 40-50% for N+1), safe deployment is 5 systems per rack (51kW base + redundancy = ~102kW total).

---

**Q19. Correct: C**

InfiniBand with GPU Direct RDMA eliminates CPU involvement in GPU-to-GPU communication, providing deterministic low-latency communication essential for multi-node training. HDR InfiniBand (200Gbps) with GPU Direct RDMA reduces gradient synchronization overhead by 60-80% compared to RoCEv2 on Ethernet. This directly addresses the collective communication bottleneck during all-reduce operations.

---

**Q20. Correct: C**

MIG (Multi-Instance GPU) provides hardware-level isolation with guaranteed QoS—each instance has dedicated memory and SM resources. 1g.10gb instances (1/7 GPU, 10GB) are optimal for BERT inference (small models, low latency needs), while 3g.40gb instances (3/7 GPU, 40GB) suit Stable Diffusion workloads. MIG ensures no noisy neighbor issues and deterministic latency, critical for SLA compliance.

---

**Q21. Correct: A**

Parallel filesystems like WEKA or BeeGFS stripe data across multiple NVMe-backed storage servers, enabling aggregate throughput to scale linearly with nodes. Combined with 200GbE RDMA (GPU Direct Storage capable), this architecture can deliver 800GB/s+ aggregate throughput. Random access patterns require NVMe (not HDD), and distributed architecture prevents single-point bottlenecks.

---

**Q22. Correct: A**

NVIDIA Container Toolkit enables GPU access in Docker containers on-prem and in cloud VMs identically. Kubernetes provides consistent orchestration (pod specs, resource requests, scheduling policies work identically). NGC registry stores GPU-optimized containers that run unchanged across environments. This stack is NVIDIA's recommended approach for hybrid AI infrastructure.

---

**Q23. Correct: B**

8 DGX A100 nodes with 8 GPUs each = 64 GPUs total. During all-reduce, each GPU must exchange gradients with all others. Ring all-reduce requires bandwidth proportional to (N-1)/N × data × GPUs. With 200GbE (25GB/s) shared across 8 nodes, per-node bandwidth is ~3.1GB/s, but each DGX can generate 600GB/s+ intra-node via NVSwitch. The network is a 95x bottleneck, explaining 45% iteration time in NCCL.

---

**Q24. Correct: A**

DGX H100 uses 4th generation NVLink with NVSwitch fabric providing 900 GB/s bidirectional bandwidth per GPU to every other GPU, enabling full all-to-all connectivity at maximum speed. This is critical for large gradient synchronization in distributed training where all GPUs need to exchange data simultaneously.

---

**Q25. Correct: D**

With 64 nodes at 400 Gbps (50 GB/s) each, aggregate network bandwidth is 3.2 TB/s. Storage must sustain roughly 25% of network bandwidth for data loading (800 GB/s read) to prevent GPU starvation. The 2 TB/hour checkpoint requirement translates to ~570 MB/s sustained write, but bursts need 200 GB/s to minimize training interruption. NVMe-oF (NVMe over Fabrics) is essential for low-latency remote storage access.

---

**Q26. Correct: B**

MIG on A100 80GB supports up to 7 instances with hardware-level isolation of memory, SMs, and cache. A 3g.20gb instance provides 3 SLices (20GB memory), 1g.10gb provides 1 slice (10GB), and 4g.40gb provides 4 slices (40GB), totaling exactly 7 slices (80GB). MIG provides true hardware isolation for compute, memory, and L2 cache, meeting strict isolation requirements for multi-tenant inference.

---

**Q27. Correct: C**

With InfiniBand (not RoCE), adaptive routing is critical for distributed training workloads. All-reduce operations create many-to-many traffic patterns that cause severe congestion when all traffic follows static shortest paths. 25% packet loss is characteristic of buffer overflow when adaptive routing is disabled. Quantum-2 switches support adaptive routing that dynamically load-balances traffic across multiple paths to prevent hotspots.

---

**Q28. Correct: C**

With 2 PB existing data and 500 TB monthly growth, data gravity is the primary constraint. A unified namespace with object storage gateway allows training jobs to access data via consistent S3 API whether running on-prem (hitting local parallel filesystem) or in cloud (hitting cloud object storage), with optional tiering and caching. This solves data accessibility without requiring full dataset migration. Exam emphasizes storage architecture as the foundation for hybrid AI workflows.

---

**Q29. Correct: B**

Each DGX H100 draws 10.2 kW. With 1.15 PUE: 10.2 × 1.15 = 11.73 kW total per system. Adding 10% transient headroom: 11.73 × 1.10 = 12.9 kW. Converting to kVA assuming 0.95 power factor: 12.9 / 0.95 = 13.6 kVA. For N+1 redundancy with dual-feed configuration, each feed must support full load: 13.6 × 2 = 27.2 kVA, rounded to 28 kVA per rack. This ensures either power feed can handle full rack load if one feed fails.

---

**Q30. Correct: A**

NDR 400Gb/s InfiniBand provides the lowest latency (sub-microsecond) for multi-node GPU communication. SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) offloads all-reduce operations to the network switches, significantly reducing GPU cycles and network traffic for collective operations. This is the recommended NVIDIA solution for large-scale training clusters requiring maximum performance.

---

**Q31. Correct: C**

Multi-Instance GPU (MIG) on A100 provides hardware-level isolation with dedicated memory, compute resources, and QoS. The 1g.20gb profile provides 1 GPU slice with 20GB memory (fits Customer A's needs), and 2g.40gb provides 2 GPU slices with 40GB memory (fits Customer B's needs). MIG ensures fault isolation - if one instance crashes, others are unaffected. This is the only solution that provides true hardware-level multi-tenancy with guaranteed resources.

---

**Q32. Correct: B**

32 DGX H100 systems × 8 GPUs × 700W = 179,200W (179.2kW) for GPUs alone. Adding system overhead (CPUs, memory, NVSwitch, networking, storage, fans) typically adds 40-50%, bringing total to approximately 250-270kW per rack of 4 DGX systems, or ~900-1080kW for all 32 systems. This fits within the 1.2MW budget. H100 SXM5 at 700W TDP with liquid cooling is the standard high-performance configuration for large-scale training.

---

**Q33. Correct: B**

Parallel distributed file systems like BeeGFS or Lustre are designed specifically for high-throughput, high-concurrency random I/O workloads. They stripe data across multiple storage targets (OSS in Lustre, storage servers in BeeGFS) and use multiple metadata servers to handle concurrent file operations. RDMA networking (InfiniBand or RoCE) reduces CPU overhead and latency. This architecture can scale to hundreds of GB/s for large training clusters and handles 500+ parallel streams efficiently.

---

**Q34. Correct: D**

S3-compatible object storage with cross-region replication provides automated, incremental sync of model checkpoints. Modern object storage supports intelligent delta transfers (only changed blocks), multi-part uploads for large files, and automatic retry/resume. Setting up replication is a one-time configuration with minimal ongoing overhead. 10Gbps = ~1.25GB/s theoretical, so 200GB would take ~160 seconds with overhead. Object storage APIs are cloud-native and integrate seamlessly with cloud inference serving platforms.

---

**Q35. Correct: A**

For latency-sensitive applications with predictable load (100 concurrent users), on-prem CAPEX often wins over cloud OPEX on a 3-year TCO basis. 2x L40S (~$16-20k) versus 4x A10G cloud instances running 24/7 for 3 years (~$30-40k+ depending on region/commitment). More importantly, on-prem eliminates network variability and noisy-neighbor effects in cloud environments, providing consistent P99 latency. While cloud offers elasticity, a fixed 100-user load doesn't benefit from that. MIG on L40S can create 4x 1g.12gb instances (2 per GPU), matching the instance count.

---

**Q36. Correct: C**

NCCL timeouts with zero GPU utilization during multi-node training typically indicate communication path issues rather than compute problems. Verifying NVLink/NVSwitch topology ensures GPU-to-GPU communication paths are correctly established. Misconfigured or degraded NVLink connections cause NCCL collective operations to fail or fall back to slower paths, triggering timeouts. This is the most direct diagnostic for inter-GPU communication issues in DGX systems.

---

**Q37. Correct: C**

Kubernetes resource quotas with preemption provide the optimal balance: guaranteeing capacity for high-priority workloads while allowing low-priority jobs to use idle resources. Preemption enables immediate eviction of P1 pods when P0 workloads arrive, ensuring zero wait time for critical requests. This approach maximizes utilization by allowing P1 to consume unused capacity rather than leaving GPUs idle. Base Command Platform supports this Kubernetes-native priority and preemption model.

---

**Q38. Correct: B**

High SM utilization with low memory bandwidth utilization indicates the GPUs are compute-bound but not effectively using memory subsystem capacity. This pattern suggests small batch sizes or inefficient memory access patterns. Increasing batch size improves memory bandwidth utilization by creating larger, more efficient memory transfers and better amortizing fixed costs of memory operations. With memory bandwidth underutilized at 60%, there is headroom to increase batch size without hitting memory limits.

---

**Q39. Correct: C**

Fair-share scheduling in Slurm uses PriorityDecayHalfLife to decay historical usage over time, ensuring teams that have consumed significant resources see their priority decrease while underutilized teams gain priority. Setting a 7-day half-life with priority/multifactor enables the scheduler to consider Team A's week-long GPU consumption, lower their priority, and boost Team B's priority based on their underutilization relative to their 30% fair-share target. This implements true fair-share principles dynamically.

---

**Q40. Correct: D**

A 3x increase in p99 latency with low GPU utilization suggests the model is not executing efficiently on the GPU, possibly due to misconfiguration. Triton server logs would reveal if the model backend is mismatched (e.g., ONNX model being run on PyTorch backend), causing fallback to CPU execution or inefficient GPU code paths. Model format mismatches are common deployment errors that cause severe performance degradation while showing low GPU utilization because the GPU is waiting or the work is being done on CPU. This is the most direct diagnostic for sudden post-deployment latency spikes.

---

**Q41. Correct: A**

Correctable ECC errors at 5-10/hour indicate degrading memory that will likely progress to uncorrectable errors. While correctable errors are automatically fixed by hardware, their increasing frequency suggests the memory cells are failing. The correlation with NaN losses indicates some errors are affecting computation before correction. Best practice is to schedule maintenance replacement (to avoid sudden failures) while immediately preventing new critical workloads from being scheduled (using taints). This balances operational continuity with proactive hardware management, preventing data corruption and job failures.

---

**Q42. Correct: A**

Illegal memory access CUDA errors with otherwise healthy GPU metrics typically indicate hardware-level memory errors (single/double-bit ECC errors) or specific GPU exceptions tracked by XIDs (GPU error codes). DCGM collects XID events and ECC counters. Checking volatile ECC errors and XID codes (like XID 48 for double-bit ECC, XID 43 for GPU page retirement) is the correct diagnostic path for this failure pattern, as these errors don't always trigger thermal or utilization alerts but cause job crashes.

---

**Q43. Correct: D**

Volcano scheduler is purpose-built for batch/HPC workloads on Kubernetes and provides: (1) gang scheduling to ensure all GPUs for a distributed job start together or not at all, (2) queue-based fair-share with configurable weights per team/queue, (3) DRF algorithm for multi-resource fairness, (4) preemption support with priority-based reclaim. This combination directly addresses all requirements: dynamic team priorities (queue weights), preemption, and gang scheduling—features not available in default kube-scheduler.

---

**Q44. Correct: C**

nvidia-smi's 'utilization.gpu' metric measures the percentage of time ANY kernel was executing or memory copies were active during the sample period—it can show high values even if actual compute (SMs) is underutilized. DCGM provides more granular metrics including SM occupancy, which reflects actual compute efficiency. For AI inference workloads, Tensor Core utilization (available via DCGM's PROF metrics in modern GPUs like A100/H100) is critical since inference often uses INT8/FP16 Tensor Core operations. This combination reveals true computational efficiency vs. just 'GPU busy' status.

---

**Q45. Correct: A**

NCCL (NVIDIA Collective Communications Library) for multi-node communication requires explicit network interface specification when multiple network interfaces are present. If NCCL_SOCKET_IFNAME is not set in multi-NIC environments, NCCL may auto-select slower Ethernet interfaces instead of high-speed InfiniBand/RoCE. The 'connection timed out' pattern affecting specific GPU pairs across nodes (not within nodes) points to inter-node communication issues. Setting NCCL_SOCKET_IFNAME=ib0 and NCCL_IB_HCA=mlx5_0 (or appropriate IB device) forces NCCL to use InfiniBand, which is the standard solution for this failure pattern in NCCL troubleshooting.

---

