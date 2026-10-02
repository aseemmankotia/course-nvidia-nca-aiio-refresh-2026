## Chapter 4: AI Cluster Networking and Storage Infrastructure — Practice Questions

**Q1.** A research lab is building a 128-GPU cluster for training large language models with 100B+ parameters. Training runs require all-reduce operations every 50ms with sub-5μs latency between GPUs. The cluster will perform distributed training with model parallelism across all nodes. Which networking solution BEST meets these requirements?

A) 400GbE Ethernet with RoCEv2 RDMA and adaptive routing
B) InfiniBand NDR 400 Gb/s with GPUDirect RDMA in fat-tree topology
C) 100GbE Ethernet with TCP offload engines and ECMP load balancing
D) InfiniBand HDR 200 Gb/s with lossless Ethernet gateway for storage

<details>
<summary>Answer</summary>

**Correct: B**

InfiniBand NDR provides 400 Gb/s bandwidth with guaranteed sub-microsecond latency (<700ns typical). GPUDirect RDMA enables zero-copy GPU-to-GPU communication bypassing CPU and system memory, critical for frequent all-reduce operations in large-scale distributed training. Fat-tree topology provides full bisection bandwidth ensuring no bottlenecks during collective operations.

**Why the others are wrong:**
- RoCEv2 over Ethernet introduces 2-10μs latency due to Ethernet switching and PFC (Priority Flow Control) handling, exceeding the 5μs requirement. Adaptive routing helps but cannot match InfiniBand's hardware-level congestion management.
- 100GbE lacks sufficient bandwidth (need 400 Gb/s for 128 GPUs) and TCP adds 20-50μs latency overhead. TCP offload engines reduce CPU load but don't provide the sub-5μs latency needed for tight training loops.
- HDR 200 Gb/s provides only half the bandwidth of NDR. While latency is acceptable, insufficient bandwidth creates bottlenecks during gradient synchronization across 128 GPUs. Lossless Ethernet gateway adds latency and complexity without benefit for this GPU-only workload.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An AI infrastructure team needs to checkpoint a 175B parameter model (350GB state) every 30 minutes during training. The checkpoint must complete in under 2 minutes to minimize training disruption. Which storage configuration provides the MINIMUM required IOPS and bandwidth?

A) NFS over 100GbE providing 12.5 GB/s bandwidth and 50K IOPS
B) Object storage (S3-compatible) with 10 Gb/s bandwidth and 100K IOPS
C) Parallel Lustre file system with 40 OSTs providing 3 GB/s aggregate bandwidth and 500K IOPS
D) WekaFS parallel file system with 8 NVMe OSDs providing 20 GB/s and 2M IOPS

<details>
<summary>Answer</summary>

**Correct: C**

350GB in 2 minutes requires 350GB / 120s = 2.92 GB/s sustained write bandwidth. Lustre with 40 OSTs easily exceeds this (3 GB/s) while providing 500K IOPS for metadata operations (creating checkpoint files, updating directory structures). Parallel architecture distributes load across multiple storage targets, avoiding bottlenecks.

**Why the others are wrong:**
- While 12.5 GB/s bandwidth is sufficient, NFS is a single-node bottleneck. Metadata operations (creating 350GB worth of checkpoint files) overwhelm single NFS server. 50K IOPS insufficient for parallel GPU writes from multiple nodes.
- Object storage has high GET latency (10-100ms) and optimized for throughput not IOPS. 10 Gb/s = 1.25 GB/s is below the 2.92 GB/s requirement. S3-compatible storage inappropriate for frequent checkpoint writes due to eventual consistency and PUT rate limits.
- WekaFS provides excessive resources (20 GB/s, 2M IOPS) at higher cost. While it works, Lustre meets the MINIMUM requirement more cost-effectively. WekaFS better suited for workloads needing >5 GB/s or >1M IOPS.

</details>

---

**Q3.** A company is deploying BlueField-3 DPUs in their AI training cluster. The cluster experiences CPU bottlenecks during RDMA operations and storage protocol processing. Which tasks should be offloaded to the BlueField-3 DPU to maximize GPU utilization?

A) Model gradient calculation and parameter updates only
B) Python interpreter execution and data preprocessing pipelines
C) RDMA transport, NVMe-oF storage protocol, and overlay network encapsulation
D) GPU memory allocation and CUDA kernel scheduling

<details>
<summary>Answer</summary>

**Correct: C**

BlueField-3 DPU offloads networking (RDMA transport, RoCE, overlay networks) and storage (NVMe-oF, NVMe/TCP) from host CPU. This frees CPU cycles for data preprocessing and frees GPU for compute. DPU handles packet processing, encapsulation/decapsulation, storage protocol parsing at line rate without CPU intervention. Typical result: 30-40% reduction in CPU utilization for network/storage tasks.

**Why the others are wrong:**
- Gradient calculation and parameter updates are compute-intensive operations performed on GPU using CUDA cores. DPU has ARM cores unsuitable for ML math operations. This would severely slow training.
- Python interpreter and data preprocessing require general-purpose CPU with large memory. DPU has limited ARM cores (16 A78 cores in BF-3) and memory (16-32GB). Moving these to DPU creates new bottleneck.
- GPU memory allocation and CUDA scheduling are handled by GPU driver and CUDA runtime on host CPU. DPU cannot manage GPU PCIe devices. This is architecturally impossible.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** An MLOps team is comparing InfiniBand HDR versus 400GbE with RoCEv2 for a new 64-GPU training cluster. Training jobs perform frequent all-reduce operations (every 100ms) and the budget allows for either solution. Which factor is the PRIMARY technical advantage of InfiniBand HDR over RoCEv2 for this workload?

A) Higher bandwidth: 200 Gb/s HDR versus 400 Gb/s Ethernet
B) Lower cost per port for switches and NICs
C) Native support for TCP/IP applications and protocols
D) Hardware-based congestion control and in-network computing capabilities

<details>
<summary>Answer</summary>

**Correct: D**

InfiniBand provides hardware-based congestion control (credit-based flow control) guaranteeing lossless transmission without PFC (Priority Flow Control) pauses that plague RoCEv2. Additionally, InfiniBand switches support SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) performing in-network all-reduce operations, reducing collective operation latency by 5-10x. This is critical for frequent all-reduce every 100ms.

**Why the others are wrong:**
- This is factually backwards: 400GbE has HIGHER bandwidth (400 Gb/s) than InfiniBand HDR (200 Gb/s). InfiniBand NDR provides 400 Gb/s, not HDR. This wrong answer tests if candidate confuses HDR/NDR specifications.
- InfiniBand is MORE expensive than Ethernet. HDR switches cost $2000-3000 per port versus $800-1200 for 400GbE ports. Cost is a disadvantage, not advantage. Exam tests knowing real-world costs.
- InfiniBand does NOT natively support TCP/IP. Requires IP-over-InfiniBand (IPoIB) gateway adding latency and complexity. This is actually a disadvantage for mixed workloads, though irrelevant for pure RDMA training.

</details>

---

**Q5.** A training pipeline loads 50TB of image data from storage at the start of each epoch. Data is accessed sequentially with 4MB read sizes. The cluster has 32 nodes each with 8 GPUs. Current NFS storage provides 2 GB/s read bandwidth causing 3-hour data loading time. Which storage upgrade provides the BEST price/performance improvement?

A) Upgrade to 10x NFS servers with load balancing for 20 GB/s aggregate bandwidth
B) Deploy GPFS Spectrum Scale with 20 NSD servers providing 15 GB/s and 200K IOPS
C) Migrate to S3 object storage with 100 Gb/s Internet egress bandwidth
D) Deploy WekaFS with NVMe SSDs providing 40 GB/s and 5M IOPS

<details>
<summary>Answer</summary>

**Correct: B**

Sequential 4MB reads are bandwidth-bound, not IOPS-bound. Need 50TB / 3600s = 13.9 GB/s to achieve 1-hour load time. GPFS with 15 GB/s meets requirement while 200K IOPS handles metadata operations (directory listings). GPFS provides parallel file system benefits (distributed metadata, multi-path I/O) at mid-tier cost. Load balancing 10x NFS servers adds complexity without true parallelism.

**Why the others are wrong:**
- While 20 GB/s bandwidth is sufficient, managing 10 NFS servers with load balancing adds operational complexity. NFS lacks parallel file system features (distributed metadata, simultaneous multi-node writes). Each server still a single point of failure requiring complex HA setup.
- S3 object storage has 10-100ms GET latency per object. For 50TB with 4MB objects = 12.8M objects × 50ms avg latency = 640K seconds = 178 hours. S3 optimized for throughput over many small requests, not low-latency large sequential reads. Wrong architecture for this access pattern.
- WekaFS with 40 GB/s and 5M IOPS is overprovisioned (costs 2-3x GPFS). 5M IOPS wasted on sequential 4MB reads. Better suited for random access workloads. Meets requirement but fails BEST price/performance criterion.

</details>

---

**Q6.** A distributed training job using 256 GPUs across 32 nodes experiences inconsistent performance. Monitoring shows some all-reduce operations complete in 2ms while others take 40ms. Network utilization is only 30% of InfiniBand HDR capacity. Which configuration issue is the MOST likely cause?

A) Insufficient CPU cores allocated for RDMA verb processing on each node
B) Missing GPUDirect RDMA configuration causing GPU-to-GPU traffic to traverse system memory
C) InfiniBand switches configured with static routing instead of adaptive routing
D) Storage checkpointing operations competing for NIC bandwidth during training

<details>
<summary>Answer</summary>

**Correct: C**

Static routing in InfiniBand can cause persistent congestion on specific links while overall fabric utilization remains low (30%). Some all-reduce paths experience congestion (40ms) while others are uncongested (2ms). Adaptive routing (AR) dynamically selects least-congested paths avoiding hotspots. This exact symptom (10-20x latency variance with low utilization) is classic static routing problem in fat-tree topologies.

**Why the others are wrong:**
- CPU cores not involved in RDMA verb processing when using kernel bypass. RDMA verbs library (libibverbs) posts work requests directly to NIC using user-space memory mapping. Insufficient CPU would show 100% CPU utilization, not variable network latency.
- Missing GPUDirect RDMA would show CONSISTENT poor performance (always 40ms+) not variable latency. GPU-to-GPU via system memory adds 20-30μs per transfer consistently. Also would show high CPU and memory bandwidth utilization.
- Storage checkpointing is infrequent (every 30-60 minutes) and wouldn't cause millisecond-scale variability during training. Bandwidth competition would show during checkpoint windows only, not random variation during all-reduce operations.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** An AI team is selecting between RoCEv2 over 400GbE and InfiniBand NDR for a new cluster. The workload includes distributed training (70% of runtime), checkpoint storage operations (20%), and infrequent model inference requests via REST API (10%). The cluster must integrate with existing Ethernet-based monitoring and management systems. Which solution BEST balances performance and operational requirements?

A) InfiniBand NDR for all traffic with IPoIB gateway for management
B) 400GbE with RoCEv2 for training and storage, native Ethernet for management
C) Dual-fabric: InfiniBand NDR for training, separate 100GbE for storage and management
D) 400GbE with TCP for all traffic and NIC-based RDMA offload for training

<details>
<summary>Answer</summary>

**Correct: B**

RoCEv2 on 400GbE provides near-InfiniBand performance (2-5μs latency vs <1μs IB) while maintaining Ethernet compatibility for management, monitoring, and REST API traffic. Single fabric reduces complexity and cost versus dual-fabric. Native Ethernet integration with existing systems eliminates IPoIB gateway overhead and complexity. Performance difference minimal for 70% training workload that can tolerate 2-5μs latency.

**Why the others are wrong:**
- IPoIB gateway adds 10-50μs latency and 30-40% throughput overhead for management traffic. Separate management network required anyway (InfiniBand can't run native Ethernet protocols). Increases complexity without benefit since training workload is only 70% of runtime.
- Dual-fabric doubles NIC costs (2x NICs per node), switch costs (2 fabrics), and operational complexity (2 networks to manage). Overkill when RoCEv2 performance sufficient for stated workload. Better for 100% training workloads with extreme latency requirements.
- TCP for training eliminates RDMA benefits (zero-copy, kernel bypass). TCP adds 20-50μs latency and 30-40% CPU overhead destroying training performance. NIC-based TCP offload (TOE) reduces CPU but still incurs TCP protocol overhead. Wrong architecture for RDMA-dependent collective operations.

</details>

---

**Q8.** A model training job writes checkpoints totaling 2TB every 20 minutes to a Lustre file system with 60 OSTs (Object Storage Targets). The job runs on 40 nodes with 8 GPUs each. Checkpoint writes are slow (8 minutes) despite Lustre bandwidth capacity. Monitoring shows only 15 OSTs receiving writes. What is the MOST likely configuration issue?

A) Lustre stripe count set to 1 instead of distributed across multiple OSTs
B) Insufficient Lustre MDT (metadata target) IOPS for file creation operations
C) GPU memory bandwidth limiting transfer rate to storage
D) Network MTU size mismatch between compute nodes and Lustre storage servers

<details>
<summary>Answer</summary>

**Correct: A**

Lustre stripe count determines how many OSTs a file is distributed across. Default stripe count of 1 means each checkpoint file written to single OST, explaining why only 15 OSTs active (one per checkpoint file/model shard). Setting stripe count to 4-8 distributes each file across multiple OSTs enabling parallel writes. Correct command: 'lfs setstripe -c 8 /lustre/checkpoints'. This is a common misconfiguration in AI workloads.

**Why the others are wrong:**
- MDT bottleneck would show slow file creation (seconds per file) not slow writes. 2TB checkpoint typically 10-20 large files, well within MDT capacity (50K+ creates/sec). MDT IOPS irrelevant for bulk data writes after file creation.
- GPU memory bandwidth (900 GB/s for A100) far exceeds storage bandwidth (10-20 GB/s). GPU-to-storage path limited by PCIe (64 GB/s), network (25 GB/s for 200GbE), or storage (variable). GPU memory not bottleneck here.
- MTU mismatch causes packet fragmentation reducing throughput 10-30% but wouldn't explain why only 15 OSTs active. Would show reduced bandwidth across ALL active OSTs, not selective OST usage. Also would affect all I/O patterns not just checkpoints.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

