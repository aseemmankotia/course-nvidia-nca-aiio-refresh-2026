# Practice Test 1: NCA-AIIO: NVIDIA AI Infrastructure & Operations Prep

- **Questions:** 45
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-1.md`. Track results in `score-tracker.md`.

---

**Q1.** A data science team is building an object detection model and reports 95% accuracy on their test set, but stakeholders notice the model rarely detects rare defects in manufacturing. The dataset contains 10,000 normal samples and 100 defect samples. Which metric would BEST reveal the model's actual performance on detecting defects?

A) Increase the test set size to 20,000 samples to improve statistical significance
B) Use ROC-AUC score since it handles class imbalance automatically
C) Retrain with 50/50 class balance and re-measure accuracy
D) Calculate recall (sensitivity) specifically for the defect class to measure detection rate

---

**Q2.** An AI infrastructure engineer needs to deploy a container for running TensorRT inference workloads on NVIDIA GPUs in a production environment. Which source should they use to obtain the optimized, validated container image?

A) Docker Hub public repository for community TensorRT images
B) Build a custom container from scratch using the official TensorRT SDK download
C) NVIDIA NGC catalog with GPU-optimized and security-scanned containers
D) Use the default TensorFlow container and install TensorRT as a plugin

---

**Q3.** A company is evaluating whether to use CPUs or GPUs for training a large transformer model with 175 billion parameters. The training dataset requires processing 500TB of text data through multiple epochs. Which architectural characteristic of GPUs provides the PRIMARY advantage for this workload?

A) Higher clock speeds allowing faster sequential instruction execution per core
B) Larger L3 cache sizes that reduce memory access latency for model parameters
C) Superior branch prediction capabilities that optimize conditional logic in neural network layers
D) Thousands of parallel cores optimized for executing identical operations across massive datasets simultaneously

---

**Q4.** An organization is deploying an AI-powered customer service chatbot that must respond to user queries within 200ms. The ML team has a trained model but needs to optimize it for production deployment on NVIDIA GPUs. Which component of the NVIDIA software stack is specifically designed for this use case?

A) CUDA toolkit for writing custom GPU-accelerated kernels
B) cuDNN library for accelerating training of convolutional neural networks
C) NVIDIA AI Enterprise for enterprise support and managed services
D) TensorRT for optimizing trained models to maximize inference throughput and minimize latency

---

**Q5.** A healthcare provider wants to deploy an AI radiology assistant to analyze chest X-rays for pneumonia detection across multiple hospitals. Regulatory requirements mandate understanding model decisions for medical liability. The IT team is evaluating different AI adoption approaches. Which deployment pattern BEST fits these requirements?

A) Edge deployment on hospital servers running NVIDIA-certified medical imaging models with explainability features and local data retention
B) Centralized cloud inference with X-rays uploaded to a public cloud provider's pre-trained model API
C) Hybrid approach using GPU-accelerated batch processing overnight with next-day results delivery
D) Mobile deployment on radiologists' tablets using quantized models for offline analysis

---

**Q6.** A machine learning engineer is comparing two binary classification models for fraud detection using precision-recall (PR) curves and ROC curves. Model A has PR-AUC of 0.72 and ROC-AUC of 0.94. Model B has PR-AUC of 0.68 and ROC-AUC of 0.96. The fraud rate in production is 0.5% (highly imbalanced). Which statement is correct?

A) Model B is superior because ROC-AUC is higher and ROC curves are always the definitive metric for classification
B) The models are equivalent in real-world performance since both have ROC-AUC above 0.90
C) Model A is likely better for this use case because PR curves are more informative for severely imbalanced datasets and Model A has higher PR-AUC
D) More information is needed because precision-recall curves cannot be compared across different models

---

**Q7.** A healthcare imaging company needs to deploy a real-time diagnostic model that processes X-ray images with <50ms latency. The model was trained on NVIDIA A100 GPUs. Which NVIDIA software component is MOST critical for optimizing the model for production inference deployment?

A) TensorRT to optimize the trained model with layer fusion and precision calibration
B) cuDNN to accelerate convolutional operations during inference
C) CUDA Toolkit to recompile the model with custom kernels
D) NGC Catalog to download a pre-optimized diagnostic model

---

**Q8.** An AI infrastructure team observes that their CNN training job achieves 85% GPU utilization on a single NVIDIA A100 but only 23% per-GPU utilization when scaled to 8 GPUs. The model uses data parallelism with gradient synchronization. What is the MOST likely bottleneck?

A) Network bandwidth limitation during all-reduce gradient synchronization operations
B) CPU preprocessing pipeline cannot feed data fast enough to 8 GPUs
C) Insufficient GPU memory causing frequent host-device transfers
D) CUDA kernel launch overhead scales linearly with GPU count

---

**Q9.** A data science team evaluates a binary fraud detection model and observes: Precision=0.72, Recall=0.94, F1=0.82. The business requires minimizing false negatives (missed fraud cases) even at the cost of more false positives. Which metric indicates the model BEST meets this business requirement?

A) The F1 score of 0.82 shows balanced performance for the fraud use case
B) The Recall of 0.94 shows the model catches 94% of actual fraud cases
C) The Precision of 0.72 shows acceptable false positive rate for fraud detection
D) The model should be re-evaluated using ROC-AUC instead of these metrics

---

**Q10.** An enterprise is deploying NVIDIA AI Enterprise on VMware vSphere. Which component from the NVIDIA AI software stack provides the GPU virtualization technology that enables multiple VMs to share physical GPU resources?

A) CUDA Multi-Process Service (MPS) for VM GPU sharing
B) NGC containers with built-in virtualization support
C) NVIDIA vGPU software with time-sliced or MIG partitioning
D) TensorRT runtime with virtual GPU allocation

---

**Q11.** A manufacturing company uses computer vision models for defect detection. During evaluation, the PR curve shows high precision at low recall values, but precision drops sharply as recall increases beyond 0.6. What does this indicate about the model's behavior?

A) The model performs well across all confidence thresholds with consistent accuracy
B) The model has high false positive rate and should use ROC curve analysis instead
C) The model is highly confident about obvious defects but struggles with subtle defects, requiring threshold tuning or additional training data
D) The model's F1 score is maximized at recall=0.6 and should operate at that threshold

---

**Q12.** An AI infrastructure engineer needs to understand why GPUs outperform CPUs for deep learning training by 50-100x. Which architectural difference is the PRIMARY reason for this performance gap?

A) GPUs have higher clock speeds (GHz) than CPUs, enabling faster sequential computation
B) GPUs have thousands of smaller cores optimized for parallel execution of identical operations on different data
C) GPUs have larger L3 cache sizes that reduce memory access latency during training
D) GPUs support lower precision arithmetic (FP16/INT8) while CPUs only support FP32/FP64

---

**Q13.** A data scientist reports that their PyTorch training job on an NVIDIA A100 GPU is running slower than expected. They are using CUDA 11.8 but have not installed cuDNN. What is the PRIMARY performance impact of this configuration?

A) Convolutional neural network operations will fall back to unoptimized implementations, significantly reducing training speed
B) The training will fail completely because PyTorch requires cuDNN for GPU operations
C) Only inference operations will be affected; training performance remains unchanged
D) Memory allocation will be inefficient but compute performance is unaffected

---

**Q14.** An organization needs to deploy a pre-trained large language model for real-time inference with sub-50ms latency requirements. Which NVIDIA software component would provide the MOST significant latency reduction compared to running the native PyTorch model?

A) CUDA Toolkit for improved GPU kernel execution
B) cuDNN for optimized neural network primitives
C) NGC catalog for accessing optimized container images
D) TensorRT for model optimization and runtime inference acceleration

---

**Q15.** A manufacturing company is evaluating their defect detection model which classifies products as defective or non-defective. The model has 95% accuracy, but only 60% of products flagged as defective are actually defective, while 98% of actual defects are correctly identified. Production costs $100 per false positive (unnecessary inspection) and $10,000 per false negative (defective product shipped). Which metric BEST indicates the model needs improvement before production deployment?

A) The F1 score is too low because precision and recall are imbalanced
B) The 60% precision indicates excessive false positives, costing $100 each versus $10,000 for missed defects
C) The 95% accuracy is insufficient for production quality requirements
D) The 98% recall is too low and should be 99.9% to prevent defective shipments

---

**Q16.** An AI infrastructure engineer is comparing GPU and CPU architectures for a deep learning workload. The workload involves matrix multiplications on tensors sized 8192x8192 with FP16 precision. Why do NVIDIA GPUs provide superior performance for this specific operation compared to high-core-count CPUs?

A) GPUs have thousands of parallel cores optimized for simultaneous arithmetic operations with specialized tensor cores for matrix math
B) GPUs have higher clock speeds, allowing faster sequential execution of multiplication operations
C) GPUs have larger L3 cache sizes that can hold entire matrices in cache memory
D) GPUs use lower precision FP16 arithmetic while CPUs are limited to FP32 operations

---

**Q17.** A healthcare organization is deploying an AI radiology assistant using NVIDIA AI Enterprise. The application requires GPU acceleration, enterprise support, security patches, and validated container images. Which component of the NVIDIA AI Enterprise software suite provides the curated, security-scanned container images with pre-integrated AI frameworks?

A) CUDA Toolkit containers from the official NVIDIA repository
B) TensorRT runtime images built from source with security hardening
C) NVIDIA driver packages with bundled framework containers
D) NGC catalog enterprise registry with AI Enterprise-certified containers

---

**Q18.** A research team is deploying a DGX H100 system for training large language models with 80B parameters. They need to maximize GPU-to-GPU communication bandwidth while minimizing latency for model parallelism across all 8 GPUs. Which interconnect technology provides the highest bandwidth in this configuration?

A) PCIe Gen5 x16 providing 128 GB/s bidirectional bandwidth
B) InfiniBand HDR200 providing 400 Gb/s per port
C) 100GbE RoCE v2 providing 100 Gb/s per NIC
D) NVLink 4.0 with NVSwitch providing 900 GB/s per GPU bidirectional bandwidth

---

**Q19.** An AI infrastructure team needs to support multiple concurrent inference workloads on a single A100 GPU, each requiring isolation for different customer tenants with guaranteed resources. The workloads vary: one needs 20GB memory, another needs 10GB, and a third needs 10GB. Which NVIDIA technology should they implement?

A) NVIDIA vGPU with virtual machine partitioning
B) NVIDIA Multi-Instance GPU (MIG) with 1x 20GB and 2x 10GB instances
C) Time-sliced GPU sharing with CUDA MPS
D) Docker containers with GPU passthrough and memory limits

---

**Q20.** A company is building a 256-GPU training cluster for computer vision models using DGX A100 systems. They need to decide between InfiniBand and Ethernet for the cluster network. Training involves frequent all-reduce operations with 50GB model gradients synchronized across all GPUs. Which statement BEST describes the networking choice?

A) 200GbE with RoCE v2 is sufficient because Ethernet has lower cost and simpler management than InfiniBand
B) InfiniBand NDR400 is required because RDMA with GPUDirect enables kernel bypass and significantly reduces collective operation latency
C) 100GbE is adequate because NVLink handles all GPU communication within nodes and network only transfers checkpoints
D) Either technology works equally well because the bottleneck is GPU computation time, not network bandwidth

---

**Q21.** An enterprise AI team is designing storage for a training pipeline that processes 500TB of raw image data, requires 50GB/s aggregate read throughput during training, and needs to support 100+ concurrent preprocessing jobs. Which storage architecture BEST meets these requirements?

A) Parallel distributed file system (e.g., Lustre, GPFS) with NVMe SSDs across multiple storage nodes
B) Single NFS server with 100GbE network connection and RAID-6 HDDs
C) Object storage (S3-compatible) with data cached locally on each training node's NVMe
D) Direct-attached NVMe SSDs on each DGX system with data replicated to each node

---

**Q22.** A data center architect is planning power and cooling for a new AI cluster with 16 DGX H100 systems. Each DGX H100 has a maximum power draw of 10.2 kW. The data center uses hot-aisle/cold-aisle containment with in-row cooling. Which power and cooling specification is MINIMUM required?

A) 163 kW power delivery with 163 kW cooling capacity to handle maximum simultaneous load
B) 120 kW power delivery with 100 kW cooling capacity assuming 80% average utilization
C) 200 kW power delivery with 140 kW cooling capacity accounting for 70% power-to-heat conversion
D) 180 kW power delivery with 120 kW cooling capacity assuming GPU power throttling during peaks

---

**Q23.** An organization needs to deploy AI infrastructure for both training (requiring maximum GPU compute) and inference (requiring low latency and high throughput). They have budget constraints and want to optimize for workload characteristics. Which deployment strategy is MOST cost-effective?

A) Deploy all workloads on DGX H100 systems using time-sharing—training during off-peak hours, inference during business hours
B) Use DGX H100 systems for training and NVIDIA L40S GPU servers for inference workloads
C) Deploy everything on cloud GPU instances with auto-scaling based on demand
D) Use HGX H100 8-GPU systems for both training and inference with MIG partitioning for isolation

---

**Q24.** A research lab is configuring an NVIDIA DGX H100 system for multi-tenant workloads. They need to allocate GPU resources to three teams: Team A requires 2 full GPUs, Team B needs fractional GPU access for inference workloads, and Team C requires isolation for compliance reasons. Which MIG (Multi-Instance GPU) configuration approach BEST meets these requirements?

A) Disable MIG entirely and use time-slicing to allocate GPU access among teams, as MIG cannot provide both full GPU and fractional GPU access simultaneously
B) Enable MIG across all 8 GPUs and create 7g.40gb instances for Team A, 1g.5gb instances for Team B, and 3g.20gb instances for Team C to ensure equal resource distribution
C) Use vGPU instead of MIG because MIG does not support compliance-level isolation between tenants and cannot mix instance sizes on the same system
D) Enable MIG on 6 GPUs, create 2 single-instance profiles for Team A, multiple smaller instances on 3 GPUs for Team B, and dedicate 1 non-MIG GPU for Team C with namespace isolation

---

**Q25.** An AI infrastructure team is designing a 256-GPU training cluster for large language models. Network latency between nodes must be minimized, and they need 400GB/s bidirectional bandwidth per node. Which interconnect topology and configuration BEST meets these requirements?

A) Deploy 32 DGX H100 nodes with NVLink Switch System providing 900GB/s NVLink fabric within each node, connected via 8x 400GbE Ethernet NICs per node using RoCEv2 for inter-node communication
B) Implement a spine-leaf Ethernet architecture with 400GbE switches and 16x 25GbE connections per node, aggregated using LACP, with TCP/IP for reliable delivery across the cluster
C) Connect nodes using NVLink bridges across DGX systems to create a unified 256-GPU NVLink domain, supplemented by 100GbE management network for control plane traffic
D) Use NVIDIA Quantum-2 InfiniBand with 400Gb/s NDR adapters in a rail-optimized fat-tree topology, 8 adapters per node, with GPU-direct RDMA enabled to bypass CPU for inter-node GPU communication

---

**Q26.** A company is planning power infrastructure for a new AI training facility housing 16 DGX H100 systems. Each DGX H100 has a maximum power draw of 10.2kW. The facility will operate at 85% average utilization. Accounting for power distribution unit (PDU) efficiency of 95% and N+1 redundancy, what is the MINIMUM total power capacity required at the utility feed level?

A) 163.2kW (total DGX power) ÷ 0.95 (PDU efficiency) = 172kW with N+1 requiring 2x capacity = 344kW minimum
B) 163.2kW × 0.85 (utilization) = 138.7kW ÷ 0.95 (PDU efficiency) = 146kW, with N+1 adding one additional PDU circuit at 10.2kW = 156.2kW minimum
C) 163.2kW ÷ 0.95 (PDU efficiency) = 172kW, with N+1 requiring capacity for one additional DGX (10.2kW) = 182.2kW minimum
D) 163.2kW × 0.85 (utilization) ÷ 0.95 (PDU efficiency) × 1.15 (cooling overhead) = 169kW, no additional N+1 capacity needed as redundancy is in PDU count not power capacity

---

**Q27.** An enterprise needs to deploy AI inference services that can scale from 10 to 1000 concurrent requests with sub-20ms latency. They have existing VMware infrastructure and want to leverage their current management tools. Which NVIDIA virtualization approach BEST meets these requirements?

A) Use MIG-backed virtual machines on bare-metal DGX systems, as MIG provides better isolation than vGPU and can be managed through VMware vCenter plugins
B) Deploy NVIDIA AI Enterprise with vGPU on VMware vSphere, using vGPU profiles optimized for inference (Q-series) with GPU partitioning to support multiple inference VM instances per physical GPU
C) Implement NVIDIA GPU Operator on Kubernetes running on VMware Tanzu, using time-slicing to multiplex GPU access across inference pods without virtualization overhead
D) Deploy bare-metal servers with GPU passthrough to VMware VMs, dedicating one full GPU per VM to ensure consistent sub-20ms latency for inference workloads

---

**Q28.** A financial services company is sizing storage for a 128-GPU cluster training a trillion-parameter model. The training dataset is 500TB and will be accessed randomly during training. Checkpointing occurs every 2 hours and generates 800GB snapshots. Which storage architecture BEST supports this workload?

A) Deploy NFS file server with 1PB HDD capacity and 25GbE network connectivity, using read caching on compute nodes to reduce storage bandwidth requirements during training epochs
B) Use local NVMe storage on each DGX node (30TB per node, 3.84TB total cluster) with dataset sharding across nodes and checkpoint aggregation to shared NAS storage over 100GbE
C) Implement parallel file system (Lustre or GPFS) with NVMe SSD object storage targets providing 2PB capacity, connected via 400Gb/s InfiniBand RDMA to match the GPU cluster interconnect
D) Configure Amazon S3 buckets with intelligent tiering for the dataset and checkpoints, accessed via S3 API from on-premises GPU cluster over dedicated 100Gb/s AWS Direct Connect

---

**Q29.** An organization is evaluating deployment options for a new computer vision training platform. They need GPU acceleration but have concerns about capital expenditure, want to avoid long-term capacity planning, and require the ability to scale from 8 GPUs to 256 GPUs based on project demands. Training jobs run 2-4 weeks continuously. Which deployment model BEST addresses these requirements?

A) Hybrid deployment with 32-GPU on-premises DGX base capacity for continuous workloads, bursting to cloud-based GPU instances (AWS p5 or Azure ND-series) for peak demand periods
B) On-premises DGX SuperPOD deployment with 256 GPUs and dynamic resource allocation, amortizing capital cost over 5 years while maintaining full control and avoiding cloud egress fees
C) Fully cloud-based deployment using reserved GPU instances for baseline 8-GPU capacity and on-demand instances for scaling to 256 GPUs, with checkpointing to resume jobs if using spot instances
D) Colocation deployment with leased NVIDIA HGX H100 servers, paying monthly per-GPU fees to a managed service provider who handles infrastructure maintenance and capacity scaling

---

**Q30.** A research lab is deploying a DGX H100 system for large language model training. The team needs to understand the GPU-to-GPU interconnect topology to optimize their distributed training framework. What is the maximum aggregate NVLink bandwidth per GPU in a DGX H100 system?

A) 900 GB/s bidirectional
B) 600 GB/s bidirectional
C) 450 GB/s bidirectional
D) 300 GB/s bidirectional

---

**Q31.** An AI infrastructure team is designing a 256-GPU training cluster and must choose between InfiniBand HDR and RoCE v2 over 200GbE Ethernet for the compute fabric. The workload involves frequent all-reduce operations for synchronous distributed training with models exceeding 100B parameters. Which networking approach BEST meets these requirements?

A) InfiniBand HDR (200 Gb/s) with SHARP in-network computing
B) RoCE v2 over 200GbE Ethernet with lossless DCB configuration
C) Standard TCP/IP over 100GbE Ethernet with NCCL optimizations
D) Multiple 25GbE Ethernet links aggregated per GPU node

---

**Q32.** A company is sizing storage for a multi-petabyte dataset used to train computer vision models on a 128-GPU cluster. Training jobs perform sequential reads of 10-50MB image batches, with peak aggregate throughput demands of 2 TB/s across all GPUs. Which storage architecture component is MOST critical to avoid GPU starvation?

A) Storage capacity of at least 5 PB to hold the full dataset plus checkpoints
B) NVMe SSDs configured in a parallel file system delivering sustained 2+ TB/s aggregate read bandwidth
C) 10 Gbps network connectivity from storage servers to compute nodes
D) Hard disk arrays with high IOPS for random access patterns

---

**Q33.** An enterprise is planning an on-premises AI infrastructure supporting multiple teams with varying workload requirements: Team A runs 24/7 inference services requiring consistent latency, Team B runs multi-day training jobs, and Team C runs short experimental jobs. The organization has purchased DGX H100 systems and wants to maximize utilization while providing workload isolation. Which GPU virtualization approach BEST meets these requirements?

A) NVIDIA vGPU with time-sliced scheduling for all workloads
B) Kubernetes with GPU device plugin for simple whole-GPU allocation
C) Multi-Instance GPU (MIG) partitioning H100s into smaller instances with guaranteed resources
D) NVIDIA vComputeServer with virtual GPU sharing across all teams

---

**Q34.** A data center architect is planning power and cooling for a deployment of eight DGX H100 systems in a single rack. Each DGX H100 has a maximum power draw of 10.2 kW. The facility has 208V three-phase power distribution. What is the MINIMUM circuit breaker rating required to safely support this configuration with appropriate headroom?

A) Two 60A three-phase circuits (one per four DGX units)
B) Three 30A single-phase circuits per DGX system
C) One 100A three-phase circuit for the entire rack
D) Four 50A three-phase circuits (one per two DGX units)

---

**Q35.** An organization is deploying a hybrid AI infrastructure with on-premises DGX systems for training and cloud instances for inference serving. Models are trained on-prem using multi-node jobs, then exported and deployed to cloud endpoints. The team reports that model transfer to cloud takes 6+ hours for a 250 GB model checkpoint over their current 1 Gbps internet link. Which architectural improvement would MOST effectively reduce transfer time while maintaining security?

A) Implement model compression using FP8 quantization before transfer
B) Use cloud object storage lifecycle policies to pre-stage model artifacts
C) Establish a dedicated 10 Gbps Direct Connect/ExpressRoute link between on-prem datacenter and cloud region
D) Split the model into smaller shards and transfer in parallel over multiple 1 Gbps connections

---

**Q36.** A production AI training cluster running on NVIDIA DGX systems shows inconsistent job completion times. The admin runs `nvidia-smi` and observes GPU temperatures cycling between 45°C and 83°C every few minutes across multiple nodes. GPU utilization drops from 95% to 40% when temperatures exceed 80°C. Which action will BEST resolve this issue?

A) Increase the Kubernetes resource request limits for GPU memory to prevent thermal throttling
B) Verify datacenter HVAC is maintaining recommended ambient temperature (18-27°C) and check for blocked airflow in server racks
C) Lower the GPU power limit using `nvidia-smi -pl 250` to reduce heat generation
D) Update the NVIDIA driver to the latest version to fix thermal management bugs

---

**Q37.** An MLOps team needs to monitor GPU utilization, memory usage, power consumption, and temperature across a 64-GPU cluster running continuous training jobs. The monitoring solution must integrate with Prometheus for alerting and require minimal performance overhead. Which tool should they deploy?

A) Run `nvidia-smi dmon` on each node and parse output with custom scripts
B) NVIDIA Nsight Systems profiler configured to export metrics every 5 seconds
C) NVIDIA DCGM (Data Center GPU Manager) with dcgm-exporter for Prometheus integration
D) Kubernetes metrics-server with GPU device plugin annotations

---

**Q38.** A Slurm-managed AI cluster has 8 nodes with 8x A100 GPUs each. A researcher submits a distributed training job requiring 16 GPUs with `#SBATCH --gres=gpu:8` and `#SBATCH --nodes=2`. The job remains in PENDING state. `sinfo` shows all nodes are idle with all GPUs available. What is the MOST likely cause?

A) The job script is missing the `--ntasks-per-node=8` parameter to match GPU count
B) Slurm's GresTypes configuration doesn't include 'gpu' in slurm.conf
C) The partition doesn't have sufficient GPU resources allocated in partition configuration
D) The user has exceeded their GPU hour quota in Slurm accounting

---

**Q39.** During a PyTorch distributed training job across 4 nodes with 8 GPUs each (32 GPUs total), the job crashes with 'CUDA out of memory' errors despite each GPU having 80GB memory and the model requiring only 24GB per GPU. Monitoring shows GPU memory usage at 78GB per GPU before the crash. What is the BEST troubleshooting step?

A) Reduce the batch size to decrease per-GPU memory consumption
B) Enable PyTorch gradient checkpointing to reduce activation memory
C) Increase the number of GPUs to 64 to distribute memory load further
D) Check for gradient accumulation bugs causing activation tensors to accumulate without being freed

---

**Q40.** An NVIDIA Base Command cluster administrator needs to ensure fair resource allocation where each research team can use up to 50% of cluster GPUs simultaneously, but individual teams can burst to 100% when others are idle. Which scheduling policy achieves this requirement?

A) Configure fair-share scheduling with 50% guaranteed allocation per team and preemption disabled
B) Set hard resource quotas at 50% for each team with no burst allowance
C) Implement priority-based scheduling with equal priorities and backfill enabled
D) Use time-slicing to allocate GPUs in 50% increments to each team

---

**Q41.** A Kubernetes cluster running GPU workloads shows that pods with GPU requests are stuck in Pending state with the error 'Insufficient nvidia.com/gpu'. Running `kubectl describe node` shows 'Allocatable: nvidia.com/gpu: 8' but 'Allocated: nvidia.com/gpu: 0'. All nodes have the NVIDIA device plugin daemonset running. What should the administrator verify FIRST?

A) The NVIDIA GPU Operator is installed and all components are in Running state
B) The container runtime (containerd/Docker) is configured with the nvidia-container-runtime
C) GPU driver version matches the CUDA version required by the workload containers
D) Node labels include 'accelerator=nvidia-gpu' for proper pod scheduling

---

**Q42.** A production AI training cluster is experiencing inconsistent GPU performance across nodes. When running nvidia-smi on affected nodes, you observe GPU clocks throttling to 50% of base frequency despite low utilization (30%). Temperature readings show 65°C, well below thermal limits. DCGM metrics show no ECC errors. Which action should you take FIRST to resolve this issue?

A) Restart the NVIDIA driver on affected nodes and clear the persistence daemon cache
B) Reduce batch size in training jobs to lower GPU memory utilization
C) Upgrade to the latest CUDA toolkit version to fix potential driver bugs
D) Check power capping settings and current power limit configuration using nvidia-smi -q -d POWER

---

**Q43.** Your organization runs mixed workloads on a Kubernetes cluster with 8 NVIDIA A100 GPUs per node. Interactive development pods require immediate GPU access, while batch training jobs can tolerate delays. You need to implement a scheduling strategy that prioritizes interactive workloads without starving batch jobs. Which Kubernetes configuration BEST meets these requirements?

A) Deploy NVIDIA GPU Operator with time-slicing enabled for all GPUs and set equal resource limits for all pods
B) Configure all pods with guaranteed QoS class and use pod anti-affinity rules to separate workload types
C) Use PriorityClasses with preemption enabled for interactive pods, set lower priority for batch jobs, and configure resource quotas per namespace
D) Implement CPU-based pod scheduling with node selectors and disable GPU device plugin for batch workloads

---

**Q44.** During a large-scale training job on a Slurm-managed cluster, you receive alerts that 3 out of 16 allocated nodes show "CUDA out of memory" errors, while the remaining 13 nodes complete their work successfully. All nodes have identical hardware (8x A100 80GB GPUs) and run the same CUDA 12.3 driver version. The training script uses PyTorch DDP with automatic gradient accumulation. What is the MOST likely cause of the OOM errors on specific nodes?

A) Memory fragmentation from previous jobs that wasn't cleared by Slurm's epilog script
B) ECC memory errors reducing available GPU memory on affected nodes
C) Uneven data distribution in the dataset sharding logic causing certain ranks to process larger batches
D) Network latency causing gradient synchronization delays and memory buildup

---

**Q45.** You are implementing MLOps best practices for a production inference service running NVIDIA Triton Inference Server on Kubernetes. The service must maintain 99.9% uptime SLA, serve models updated weekly, and provide metrics for monitoring. Which combination of configurations BEST supports these requirements?

A) Enable Triton model versioning with explicit version policies, configure liveness and readiness probes, integrate DCGM exporter for GPU metrics, and use rolling updates with maxUnavailable=0
B) Deploy Triton with model control mode POLL, disable health checks to reduce overhead, export metrics via custom sidecar container, and use Recreate deployment strategy
C) Configure Triton in ensemble mode only, implement custom health endpoints in application code, use Prometheus node exporter for all metrics, and deploy with maxSurge=3
D) Use explicit model loading with static configuration files, rely on Kubernetes default probes, enable NVML queries in the application, and perform blue-green deployments manually

---

