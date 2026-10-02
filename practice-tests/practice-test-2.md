# Practice Test 2: NCA-AIIO: NVIDIA AI Infrastructure & Operations Prep

- **Questions:** 45
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-2.md`. Track results in `score-tracker.md`.

---

**Q1.** A data science team is deploying a computer vision model for real-time quality inspection on a manufacturing line. The model was trained using PyTorch and needs to process 60 frames per second with sub-10ms latency. Which NVIDIA software component is MOST critical for achieving this inference performance requirement?

A) cuDNN to accelerate the convolutional layer operations during inference
B) CUDA Toolkit to compile custom kernels for the inference pipeline
C) TensorRT to optimize and convert the PyTorch model for high-throughput inference
D) NGC PyTorch container to ensure framework compatibility with the GPU

---

**Q2.** An MLOps engineer notices that their binary classification model achieves 95% accuracy but only 40% precision on the positive class in production. The business cost of false positives is significantly higher than false negatives. Which evaluation approach would BEST help tune the model for this business requirement?

A) Retrain the model with balanced class weights to improve the ROC AUC score
B) Increase the training epochs until the F1 score exceeds 90%
C) Adjust the classification threshold while monitoring the Precision-Recall curve to find the optimal precision-recall trade-off point
D) Use cross-validation to reduce variance and improve overall accuracy to 98%

---

**Q3.** A research team is comparing GPU vs CPU performance for training a large transformer model. They observe that batch size 32 on CPU takes 450 seconds per epoch, while batch size 256 on GPU takes 28 seconds per epoch. Which architectural characteristic of GPUs PRIMARILY explains this 16x performance improvement?

A) GPUs have higher clock speeds (GHz) than CPUs, allowing faster sequential instruction execution
B) GPUs have larger L3 cache sizes that reduce memory access latency during training
C) GPUs support dynamic voltage and frequency scaling that optimizes power efficiency during compute-intensive tasks
D) GPUs have thousands of cores optimized for parallel execution of identical operations on large data arrays

---

**Q4.** A company is selecting pre-trained models from the NGC catalog for a natural language processing project. They need to ensure the models meet enterprise security and support requirements. Which combination of NGC features would BEST address these enterprise needs?

A) Download only models marked as 'community contributed' to ensure broad peer review
B) Use models with the highest download count to ensure production readiness
C) Filter for models that support CUDA 11.x to ensure GPU compatibility
D) Select models with NVIDIA AI Enterprise license AND review the security scan results in the container manifest

---

**Q5.** An AI infrastructure team is provisioning resources for two distinct workloads: (1) monthly retraining of a recommendation model using 500GB of user interaction data, and (2) serving 10,000 inference requests per second for the same model. How should GPU memory requirements differ between these workloads?

A) Both workloads require identical GPU memory since they use the same model architecture
B) Inference requires higher GPU memory due to concurrent request processing and caching of results
C) Training requires lower GPU memory because data is streamed from disk; inference loads the entire dataset into GPU memory
D) Training requires higher GPU memory to hold model weights, optimizer states, gradients, and data batches; inference only needs to hold model weights and small inference batches

---

**Q6.** A healthcare AI team is analyzing model performance across different patient demographics. Their model shows 0.92 recall but only 0.68 precision for detecting a rare condition (2% prevalence). Regulatory requirements mandate minimizing missed diagnoses while maintaining clinical workflow efficiency. Which metric interpretation would BEST guide their threshold adjustment decision?

A) The high recall is acceptable for regulatory compliance; focus on increasing precision by raising the classification threshold to reduce false positive alerts that burden clinicians
B) The low precision indicates the model is unreliable; retrain with more data until precision and recall are both above 0.90
C) Calculate the F1 score and adjust the threshold until F1 is maximized, ensuring balanced performance
D) The 2% prevalence suggests using accuracy as the primary metric since the dataset is imbalanced

---

**Q7.** A financial services company is deploying a fraud detection model that processes 50,000 transactions per second. The model uses mixed precision (FP16/FP32) training and requires deterministic inference results for regulatory compliance. Which NVIDIA software component should be prioritized to maximize inference throughput while maintaining bit-exact reproducibility across deployments?

A) cuDNN with default settings, as it automatically handles mixed precision and provides sufficient performance
B) TensorRT with FP16 precision mode and builder optimization level 5 with strict type constraints enabled
C) CUDA Toolkit with custom kernel implementations using __half2 intrinsics for all operations
D) NGC-hosted pre-optimized container with RAPIDS libraries for accelerated preprocessing

---

**Q8.** An AI research team is evaluating a binary classifier for medical imaging. The model achieves 94% accuracy, 89% precision, 76% recall, and an F1 score of 0.82. The team observes that the PR curve shows AUC of 0.81 while the ROC curve shows AUC of 0.93. What does this metric pattern most likely indicate about the dataset and model behavior?

A) The model has high false negative rate relative to false positive rate, indicating conservative classification behavior
B) The dataset is balanced but the decision threshold is suboptimal, requiring calibration to improve F1 score
C) The model exhibits high variance across different patient populations, requiring stratified validation
D) The dataset has severe class imbalance with positive cases being rare, and the model is overfitting to the majority class

---

**Q9.** A manufacturing company needs to deploy real-time defect detection on 12 assembly line cameras, each producing 4K video at 30 fps. The CNN model requires 18.4 TFLOPS of FP32 compute per frame and must process all cameras with less than 100ms latency. Which GPU architecture characteristic is most critical for meeting these requirements cost-effectively?

A) Memory bandwidth (GB/s) to handle simultaneous 4K frame ingestion from 12 camera streams without bottlenecking
B) Tensor Core count and FP16 throughput to leverage mixed precision and reduce compute time per frame
C) CUDA core count and base clock frequency to deliver sustained 18.4 TFLOPS across parallel inference streams
D) NVLink interconnect bandwidth for multi-GPU scaling when single-GPU compute is insufficient

---

**Q10.** An MLOps team is standardizing their AI infrastructure across on-premises and cloud deployments. They need consistent CUDA, cuDNN, and TensorRT versions, pre-configured frameworks (PyTorch, TensorFlow), and security patching. Which combination of NVIDIA ecosystem components best addresses these requirements with enterprise support?

A) Public NGC catalog containers with Kubernetes DaemonSet automatic updates enabled
B) CUDA Toolkit direct installation on bare metal with manual framework compilation from source
C) NVIDIA GPU Cloud (NGC) individual accounts with shared container image repository
D) NVIDIA AI Enterprise subscription with NGC Private Registry and version-locked container tags

---

**Q11.** A healthcare AI startup is comparing infrastructure approaches for their medical imaging training workload. The model trains on 400,000 images with batch size 128, performs extensive data augmentation (rotation, scaling, color jittering), and uses a ResNet-152 architecture. Initial CPU-based training takes 18 days per epoch. Which architectural difference between GPUs and CPUs most directly explains why GPU training will reduce this to under 8 hours per epoch?

A) GPUs have thousands of ALUs operating in SIMD fashion, enabling parallel execution of identical augmentation operations across all 128 batch images simultaneously
B) GPU memory architecture uses HBM2/HBM3 with 900+ GB/s bandwidth versus CPU DDR5 at 80 GB/s, eliminating data transfer bottlenecks during backpropagation
C) GPU schedulers use context-free thread switching with zero-overhead execution, while CPU context switching incurs 1-5 microsecond penalties per thread
D) GPUs implement hardware-accelerated floating point units that compute matrix multiplications and convolutions 50-100x faster than CPU SIMD instructions

---

**Q12.** An autonomous vehicle company is moving from research (ad-hoc GPU clusters) to production inference deployment (1,000 vehicles × 8 cameras × real-time object detection). They need to transition from training-focused infrastructure to inference-optimized deployment while maintaining model accuracy. Which shift in infrastructure approach and NVIDIA software stack usage represents the most critical change for this transition?

A) From multi-GPU training with NCCL collective communications to single-GPU inference with CUDA MPS for multi-process isolation
B) From PyTorch/TensorFlow with cuDNN-accelerated training to TensorRT-optimized inference engines with INT8 post-training quantization and calibration
C) From NGC catalog research containers to NVIDIA AI Enterprise containers with production support and CVE patching
D) From data-parallel distributed training with NVLink to model-parallel inference with NVSwitch-connected DGX systems

---

**Q13.** A financial services company is deploying a fraud detection model that must minimize false negatives (missed fraud cases) even if it increases false positives. The model achieves 0.92 precision, 0.78 recall, and 0.84 F1 score. Which metric should the ML team prioritize when tuning hyperparameters, and what does the current recall value indicate?

A) Precision — the current 0.92 value means 92% of predicted fraud cases are actually fraud, which is excellent for reducing customer complaints
B) Recall — the current 0.78 value means the model is missing 22% of actual fraud cases, which is unacceptable for this use case
C) F1 score — the balanced 0.84 value provides the best tradeoff and should be maximized to satisfy both precision and recall equally
D) Area Under ROC Curve (AUC-ROC) — this metric is always superior to precision/recall for imbalanced classification problems

---

**Q14.** An AI infrastructure team needs to accelerate transformer-based large language model inference for a customer service chatbot requiring sub-100ms response latency. The deployment uses NVIDIA A100 GPUs. Which combination of NVIDIA software components provides the BEST optimization for this workload?

A) CUDA Toolkit for GPU kernels + cuDNN for optimized primitives + NVIDIA Triton Inference Server for model serving
B) TensorRT for inference optimization + FP16/INT8 quantization + NVIDIA Triton Inference Server for dynamic batching and multi-model serving
C) NGC containers for pre-built environments + cuDNN for neural network operations + NCCL for multi-GPU scaling
D) NVIDIA AI Enterprise with vGPU for virtualization + RAPIDS cuDF for data preprocessing + TensorRT for inference

---

**Q15.** A healthcare AI team is comparing GPU and CPU architectures for deploying a medical image segmentation model. The model performs 50 million floating-point operations per image with high data parallelism. Which architectural difference MOST explains why GPUs provide 10-50x speedup over CPUs for this workload?

A) GPUs have thousands of smaller cores optimized for parallel execution of identical operations, while CPUs have fewer larger cores optimized for sequential complex logic
B) GPUs have higher clock speeds (5+ GHz) compared to CPUs (2-3 GHz), allowing faster execution of individual operations
C) GPUs have larger L3 cache sizes (128+ MB) compared to CPUs, reducing memory access latency for neural network weights
D) GPUs support ECC memory which prevents bit flips during training, while CPUs lack hardware error correction capabilities

---

**Q16.** An enterprise is evaluating NVIDIA AI Enterprise for deploying production AI workloads across their VMware vSphere infrastructure. Which capability is a PRIMARY differentiator of NVIDIA AI Enterprise compared to using open-source NGC catalog containers alone?

A) Enterprise support with SLAs, security updates, extended lifecycle support, and certified compatibility with enterprise platforms like VMware and Red Hat
B) Access to TensorRT and Triton Inference Server, which are exclusive to AI Enterprise subscribers
C) GPU virtualization (vGPU) technology that enables multiple VMs to share a single physical GPU with hardware-level isolation
D) Pre-trained foundation models optimized for inference that are not available in the public NGC catalog

---

**Q17.** A manufacturing company is implementing predictive maintenance using sensor data from 10,000 machines. The data science team reports class imbalance: 95% normal operation, 5% failure cases. When evaluating model performance, they observe high accuracy (94%) but poor failure detection. Which visualization technique and metric combination would BEST reveal the model's true performance on the minority class?

A) ROC curve with AUC metric — this shows true positive rate vs false positive rate across all thresholds and is the standard for binary classification
B) Confusion matrix with overall accuracy — this provides the complete picture of all prediction outcomes in a single view
C) Feature importance plot with F1 score — this identifies which sensor readings contribute most to predictions and balances precision and recall
D) Precision-Recall curve with Average Precision (AP) metric — this focuses on performance for the positive (failure) class and is more informative for imbalanced datasets

---

**Q18.** A financial services company is deploying a 64-GPU cluster for LLM training with strict power budget constraints. The data center can provide 120kW per rack. Each DGX H100 system consumes approximately 10.2kW at peak load. The infrastructure team needs to determine rack density while maintaining N+1 redundancy for power distribution. What is the MAXIMUM number of DGX H100 systems that can safely be deployed per rack?

A) 5 systems per rack to allow for power redundancy overhead
B) 8 systems per rack with dual power supplies
C) 10 systems per rack using dynamic power capping
D) 12 systems per rack with liquid cooling optimization

---

**Q19.** An AI research lab running multi-node training jobs across 16 DGX A100 systems is experiencing inconsistent training performance. Network monitoring shows periodic 40% throughput drops during gradient synchronization phases. The cluster uses 200GbE Ethernet with RoCEv2. Which configuration change would BEST address this performance degradation?

A) Increase TCP buffer sizes and enable jumbo frames to 9000 MTU
B) Configure PFC (Priority Flow Control) and ECN on the Ethernet fabric
C) Replace Ethernet infrastructure with HDR InfiniBand and enable GPU Direct RDMA
D) Add additional NICs per node and implement network bonding with LACP

---

**Q20.** A cloud service provider needs to offer AI inference services with strict SLA guarantees for latency-sensitive applications while maximizing GPU utilization across diverse workloads. Some customers run BERT inference (small batch, low latency required), others run Stable Diffusion (larger batch, moderate latency acceptable). The infrastructure uses A100 80GB GPUs. Which multi-tenancy strategy provides the BEST balance of isolation and utilization?

A) Use vGPU with time-sliced scheduling and quality-of-service policies per customer
B) Assign full dedicated GPUs per customer to guarantee SLA compliance
C) Deploy MIG with 3g.40gb instances for Stable Diffusion and 1g.10gb instances for BERT workloads
D) Implement CUDA MPS with per-client memory limits and compute priority settings

---

**Q21.** A pharmaceutical company is sizing storage infrastructure for a drug discovery pipeline that will train foundation models on 500TB of molecular structure data. Training jobs perform random access to 15% of the dataset per epoch with 8 DGX H100 nodes reading concurrently. The storage system must sustain 800GB/s aggregate read throughput during training. Which storage architecture BEST meets these requirements?

A) Parallel distributed filesystem (WEKA, BeeGFS) with NVMe backend and 200GbE RDMA fabric
B) Single all-flash NVMe array with 25GbE connectivity to compute nodes
C) Object storage (S3-compatible) with local NVMe caching on compute nodes
D) NFS over RDMA with HDD-based tiered storage and SSD read cache

---

**Q22.** An enterprise ML platform team is evaluating GPU deployment options for a hybrid infrastructure supporting both on-premise DGX systems and cloud-based training. They need to run identical containerized training workloads across both environments with minimal portability friction. Training jobs use NCCL for multi-GPU communication and require consistent performance profiling. Which combination of technologies provides the MOST seamless hybrid deployment experience?

A) Docker containers with NVIDIA Container Toolkit, Kubernetes for orchestration, and NGC container registry for image management
B) VMware vSphere with vGPU, proprietary hypervisor networking, and cloud VM snapshots for migration
C) Singularity containers with native GPU passthrough and manual job scheduling scripts
D) Cloud-native container services (EKS, GKE) with Terraform for infrastructure provisioning only

---

**Q23.** A university research cluster with 8 DGX A100 nodes connected via NVSwitch is experiencing suboptimal scaling efficiency. Single-node training achieves 95% GPU utilization, but 8-node distributed training drops to 68% efficiency during all-reduce operations. Network topology uses a single 200GbE switch with NCCL configured for ring algorithm. Profiling shows 45% of iteration time spent in NCCL calls. What is the PRIMARY bottleneck?

A) NVSwitch bandwidth saturation due to intra-node communication conflicts
B) Insufficient inter-node network bandwidth—NCCL all-reduce requires full bisection bandwidth
C) NCCL ring algorithm inefficiency—tree algorithm would reduce communication rounds
D) CPU bottleneck in gradient computation preprocessing before NCCL synchronization

---

**Q24.** A financial services company is deploying a DGX H100 system for fraud detection model training. The workload requires maximum GPU-to-GPU bandwidth for large gradient synchronization during distributed training across all 8 GPUs. The system architect must choose between PCIe Gen5 switches and NVLink configurations. Which interconnect configuration provides the BEST performance for this use case?

A) NVSwitch-based NVLink fabric providing 900 GB/s bidirectional bandwidth per GPU across all 8 GPUs
B) PCIe Gen5 x16 switches connecting all GPUs with 128 GB/s bidirectional bandwidth per GPU
C) Direct NVLink connections between adjacent GPU pairs with 600 GB/s per link
D) Hybrid configuration using NVLink for local pairs and PCIe Gen5 for cross-pair communication

---

**Q25.** An AI infrastructure team is sizing storage for a large language model training cluster with 64 DGX nodes. The training dataset is 500 TB with random read patterns during data loading, and checkpoints generate 2 TB per hour that must be written with minimal training interruption. Network bandwidth is 400 Gbps InfiniBand per node. What is the MINIMUM storage performance requirement to avoid training bottlenecks?

A) Aggregate 50 GB/s read throughput and 10 GB/s write throughput across the cluster
B) Aggregate 3.2 TB/s read throughput and 500 GB/s write throughput across the cluster
C) Aggregate 25 TB/s read throughput and 5 TB/s write throughput using DDN EXAScaler
D) Aggregate 800 GB/s read throughput and 200 GB/s write throughput with NVMe-oF support

---

**Q26.** A cloud service provider is implementing multi-tenancy on NVIDIA A100 80GB GPUs to serve multiple inference workloads with strict isolation requirements. Tenant A needs 20GB GPU memory with full SM access, Tenant B needs 10GB with compute isolation, and Tenant C needs 40GB with memory isolation. Which NVIDIA technology configuration BEST meets these requirements?

A) vGPU with three virtual GPU profiles assigned to separate VMs with memory limits enforced by hypervisor
B) MIG (Multi-Instance GPU) with one 3g.20gb instance, one 1g.10gb instance, and one 4g.40gb instance on a single A100
C) Time-slicing scheduler with three containers sharing the full GPU with memory quotas
D) MPS (Multi-Process Service) with three processes sharing compute resources with memory partitioning

---

**Q27.** An on-premises AI training cluster with 32 DGX A100 nodes connected via NVIDIA Quantum-2 InfiniBand switches experiences job completion times 40% longer than expected. Network monitoring shows 25% packet loss during all-reduce operations. Physical layer tests show no cable faults. What is the MOST likely root cause?

A) RDMA over Converged Ethernet (RoCE) is misconfigured with insufficient PFC (Priority Flow Control) buffers
B) NCCL topology detection failed and is using ring algorithm instead of tree algorithm for all-reduce
C) InfiniBand adaptive routing is disabled causing congestion on default paths during collective operations
D) GPU Direct RDMA is not enabled forcing data to copy through CPU memory during network transfers

---

**Q28.** A research institution is planning a hybrid AI infrastructure with on-premises DGX SuperPOD for training and cloud bursting to NVIDIA DGX Cloud for overflow capacity. Training jobs must seamlessly migrate between environments. Dataset storage is 2 PB and growing 500 TB monthly. Which architecture component is MOST critical for enabling this hybrid workflow?

A) Direct Connect or ExpressRoute link with minimum 100 Gbps bandwidth between on-prem datacenter and cloud region
B) Kubernetes cluster federation with shared etcd database synchronizing workload state across environments
C) Unified namespace storage with object storage gateway providing S3-compatible API to both on-prem parallel filesystem and cloud object storage
D) Common container registry mirrored across both environments with identical CUDA toolkit and cuDNN library versions

---

**Q29.** An AI datacenter operator is evaluating power infrastructure for a 16-rack DGX H100 SuperPOD deployment. Each DGX H100 system draws 10.2 kW at full load. The facility has 3-phase 480V distribution with N+1 redundancy requirement. Assuming 15% power overhead for cooling (PUE 1.15) and 10% headroom for transients, what is the MINIMUM required UPS capacity in kVA per rack?

A) 12 kVA per rack with single-feed configuration
B) 28 kVA per rack with dual-feed redundant configuration
C) 20 kVA per rack assuming 0.95 power factor and including cooling overhead
D) 14 kVA per rack for compute load only, with separate cooling circuits

---

**Q30.** A research lab is deploying a multi-node AI training cluster with 8 DGX H100 systems. Each system has 8 H100 GPUs. The workload requires all-reduce operations across all 64 GPUs with minimal latency. The network engineer proposes using 200Gb/s Ethernet with RoCE v2. Which configuration BEST meets the low-latency requirement?

A) Deploy NDR 400Gb/s InfiniBand with SHARP in-network computing to offload collective operations from GPUs
B) Use the proposed 200Gb/s Ethernet with RDMA over Converged Ethernet and enable PFC on all switches
C) Upgrade to 400Gb/s Ethernet and implement TCP/IP with socket-direct for NCCL communication
D) Connect all DGX systems using NVLink Switch Systems to create a single shared memory domain

---

**Q31.** An AI service provider needs to host multiple customer workloads on shared DGX A100 infrastructure while ensuring strict isolation and QoS guarantees. Customer A requires 3 A100 GPU instances with 20GB memory each. Customer B needs 2 instances with 40GB each. The DGX A100 has 8x 80GB A100 GPUs. Which solution provides the required isolation?

A) Configure time-slicing on the GPUs to allocate dedicated time windows to each customer workload
B) Use vGPU with A100-10-20C profile for Customer A and A100-20-40C profile for Customer B, deployed on VMware vSphere
C) Deploy NVIDIA MIG with 3 instances of 1g.20gb profile for Customer A and 2 instances of 2g.40gb profile for Customer B
D) Use Kubernetes with NVIDIA GPU Operator and node affinity to dedicate specific GPUs to each customer namespace

---

**Q32.** A financial services company is designing an AI data center with 256 NVIDIA H100 GPUs for training large language models. The power budget is 1.2MW for the entire GPU infrastructure. Cooling is provided by liquid cooling with a design ambient temperature of 27°C. Which configuration is feasible within the power constraints?

A) Deploy 256 H100 PCIe GPUs at 350W TDP in 4U servers, requiring approximately 90kW total GPU power
B) 32 DGX H100 systems (8 GPUs each) with H100 SXM5 GPUs at 700W TDP, requiring 896kW total GPU power plus overhead
C) 32 HGX H100 8-GPU baseboards with H100 SXM5 at 350W TDP, requiring 448kW total GPU power plus system overhead
D) Use 64 systems with 4x H100 SXM5 GPUs each at 700W TDP, totaling 179kW for GPUs with minimal overhead

---

**Q33.** An AI training job processes 500TB of data from distributed storage. The training script performs random reads of 1-10MB chunks with high concurrency (500+ parallel streams). Current setup uses NFS over 100GbE, achieving 6GB/s aggregate read throughput, creating a bottleneck. Storage team proposes four solutions. Which BEST addresses the random-read throughput requirement?

A) Deploy NVIDIA GPUDirect Storage with NVMe-oF to enable direct GPU memory access, bypassing CPU
B) Migrate to parallel distributed file system (BeeGFS or Lustre) with multiple metadata servers and OSS nodes over RDMA
C) Upgrade NFS to pNFS (parallel NFS) with multiple data servers and 200GbE network links
D) Cache the working dataset in node-local NVMe SSDs and use a data loader with prefetching to GPU memory

---

**Q34.** A company operates a hybrid AI infrastructure with on-premises DGX systems for training and cloud instances for inference serving. They need a solution to move trained model checkpoints (50-200GB per model) from on-prem to cloud multiple times daily with minimal operational overhead. Network bandwidth is 10Gbps dedicated. Which solution minimizes transfer time and operational complexity?

A) Use NVIDIA NGC Private Registry on-premises, replicate to NGC cloud instance, and pull models from cloud NGC
B) Set up a site-to-site VPN and use rsync with compression to transfer checkpoints directly to cloud VMs
C) Deploy NVIDIA Base Command Manager on-prem and cloud, using its built-in model versioning and sync capabilities
D) Implement object storage (S3-compatible) on-premises with cross-region replication to cloud object storage

---

**Q35.** An AI team is comparing two deployment options for a latency-sensitive inference application serving 100 concurrent users with 50ms P99 latency SLA. Option A: Cloud with 4x A10G instances (24GB each). Option B: On-prem with 2x L40S GPUs (48GB each) using MIG. Model size is 12GB, with 8GB working memory per instance. Which factors make Option B the better choice?

A) Cost savings from capital expenditure versus ongoing cloud operational expense over 3-year period and predictable performance
B) Lower per-query latency due to MIG's dedicated compute slices and elimination of cloud network round-trip time
C) Higher GPU memory bandwidth on L40S (864 GB/s vs A10G's 600 GB/s) enabling faster inference throughput
D) L40S supports FP8 Transformer Engine while A10G does not, providing 2x throughput for transformer-based models

---

**Q36.** An AI training cluster running multi-node PyTorch distributed training jobs on DGX systems experiences intermittent NCCL timeout errors. DCGM shows GPU utilization drops to 0% on some nodes during these failures, while nvidia-smi reports no GPU errors. Network bandwidth monitoring shows normal IB traffic. Which troubleshooting step should be performed FIRST?

A) Increase NCCL_TIMEOUT environment variable from default 600s to 1800s and restart jobs
B) Review container orchestrator logs for OOMKilled events and increase memory limits in pod specifications
C) Check NVLink topology with nvidia-smi topo -m and verify PCIe switch connectivity matches expected configuration
D) Verify GPU clocks are not being throttled due to power limits using nvidia-smi -q -d PERFORMANCE

---

**Q37.** A production inference cluster managed by NVIDIA Base Command Platform needs to support both high-priority real-time inference requests (P0) and lower-priority batch scoring jobs (P1). The cluster has 8 A100 GPUs. Which resource allocation strategy BEST ensures P0 workloads always have immediate GPU access while maximizing overall cluster utilization?

A) Use NVIDIA MIG to partition each A100 into 7 instances, dedicating 28 instances to P0 and 28 to P1 with strict limits
B) Reserve 6 GPUs exclusively for P0 workloads using node taints and tolerations, allowing P1 to use only 2 GPUs
C) Configure Kubernetes resource quotas to guarantee 4 GPUs for P0 namespace and allow P1 to use remaining capacity with preemption enabled
D) Implement time-slicing on all 8 GPUs with 100ms context switch intervals, prioritizing P0 requests through CUDA stream priorities

---

**Q38.** DCGM metrics show that GPUs in a training cluster have consistent 95%+ SM (streaming multiprocessor) utilization but only 60% memory bandwidth utilization. Training throughput is 40% below expected performance. Which optimization should be implemented?

A) Enable tensor cores by using mixed precision training (FP16 or BF16) to increase compute throughput
B) Increase batch size to improve memory bandwidth utilization and better amortize data transfer costs
C) Reduce batch size and increase gradient accumulation steps to fit more data in L2 cache
D) Enable GPU peer-to-peer memory access using cudaDeviceEnablePeerAccess for faster multi-GPU communication

---

**Q39.** A Slurm cluster running AI training workloads needs to enforce fair-share scheduling across three research teams with different priorities: Team A (weight 50), Team B (weight 30), Team C (weight 20). Team A has submitted jobs consuming all available GPUs for the past week. Team B now submits urgent jobs but they remain queued. Which Slurm configuration change will allow Team B jobs to start while maintaining fair-share principles?

A) Configure OverSubscribe=FORCE on GPU resources to allow Team B jobs to time-share GPUs with Team A jobs
B) Enable PreemptMode=REQUEUE with PreemptType=preempt/partition_prio to suspend Team A jobs and allocate GPUs to Team B
C) Set PriorityDecayHalfLife=7-0 and PriorityType=priority/multifactor to decay Team A's accumulated usage and boost Team B priority
D) Modify Team B's QoS to include Flags=OverPartQOS to bypass partition limits and preempt lower-priority jobs

---

**Q40.** An MLOps pipeline deploys inference models to production using NVIDIA Triton Inference Server on Kubernetes. After deploying a new model version, p99 latency increases from 15ms to 45ms despite GPU utilization remaining at 30%. DCGM shows no thermal throttling and nvidia-smi reports normal clock speeds. Which diagnostic step will MOST likely identify the root cause?

A) Compare model file sizes between versions and check if dynamic batching configuration max_batch_size needs tuning for the new model
B) Check if the new model version uses a different precision (FP32 vs FP16) and optimize instance_group settings for the precision
C) Verify GPU memory allocation using nvidia-smi and check if the new model causes memory fragmentation requiring more frequent allocations
D) Review Triton server logs for model loading errors and verify the model format (ONNX, TensorRT, PyTorch) matches the backend configuration

---

**Q41.** A GPU cluster experiences recurring ECC (Error Correction Code) memory errors on specific A100 GPUs. DCGM reports 5-10 correctable ECC errors per hour on 2 out of 16 GPUs, with no uncorrectable errors. Training jobs complete successfully but occasionally show NaN losses on these GPUs. What is the BEST operational response?

A) Schedule the affected GPUs for maintenance window replacement while using node taints to prevent new training jobs from being scheduled on them
B) Continue monitoring since correctable ECC errors are normal and self-healing; only take action if uncorrectable errors occur
C) Immediately drain the affected nodes, remove the GPUs from the cluster, and RMA them as correctable errors indicate failing hardware
D) Reduce GPU clock speeds on affected GPUs using nvidia-smi -lgc to decrease error rate while maintaining the GPUs in production

---

**Q42.** A production AI cluster running distributed training workloads experiences intermittent job failures with CUDA error 'an illegal memory access was encountered'. DCGM shows all GPUs healthy, nvidia-smi reports normal temperatures, and GPU utilization is at 95%. The cluster uses Kubernetes with NVIDIA GPU Operator 24.x. Which troubleshooting step will MOST likely identify the root cause?

A) Check for GPU ECC errors using nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv and review DCGM XID error codes in logs
B) Reduce GPU clock speeds using nvidia-smi -lgc to prevent power throttling during peak utilization
C) Restart the NVIDIA Device Plugin daemonset to refresh GPU allocation mappings in Kubernetes
D) Increase the Kubernetes pod memory limits to prevent CPU RAM overflow affecting GPU operations

---

**Q43.** An AI operations team needs to implement fair-share scheduling for multiple research teams submitting training jobs to a 128-GPU cluster. Team priorities change monthly, some jobs are preemptible, and the system must support gang scheduling for multi-GPU jobs. Which scheduling solution BEST meets these requirements?

A) Kubernetes with default kube-scheduler and PriorityClasses, using resource quotas per namespace for team allocation
B) Slurm with fair-share configuration, partition priorities, and --exclusive flag for multi-GPU jobs
C) NVIDIA Base Command Manager with job queues, configurable team weights, and multi-node job support
D) Kubernetes with Volcano scheduler using queue priorities, gang scheduling plugin, and DRF (Dominant Resource Fairness) algorithm

---

**Q44.** GPU utilization metrics from DCGM show an average of 45% across a cluster running inference workloads, while nvidia-smi reports GPU utilization at 92%. Memory utilization is at 78%. The DevOps team wants to understand true GPU efficiency to right-size instance types. What explains this discrepancy and what metric should be prioritized?

A) nvidia-smi polling interval is too slow (default 1 second) causing sampling bias; DCGM uses faster polling and is more accurate—switch all monitoring to DCGM only
B) The discrepancy indicates thermal throttling reducing effective compute; check gpu_temp and power_limit in DCGM to identify throttling events and increase cooling capacity
C) DCGM measures active SM (streaming multiprocessor) utilization while nvidia-smi measures any GPU activity including memory copies; prioritize DCGM metrics for compute efficiency and add Tensor Core utilization metrics for AI workloads
D) DCGM aggregates utilization across all GPUs while nvidia-smi shows per-GPU peaks; use the average of both metrics (68%) for capacity planning and instance selection

---

**Q45.** A distributed training job across 16 nodes (8 GPUs per node) fails after 6 hours with 'NCCL operation ncclAllReduce failed' and timeout errors. NCCL_DEBUG logs show 'Connection timed out' for specific GPU pairs across nodes. Network bandwidth tests show 100 Gbps InfiniBand connectivity is functional. What is the MOST likely cause and solution?

A) NCCL_SOCKET_IFNAME environment variable is not set, causing NCCL to use Ethernet instead of InfiniBand; set NCCL_SOCKET_IFNAME=ib0 and NCCL_IB_HCA to specify InfiniBand adapters
B) GPU peer-to-peer (P2P) access is disabled in the BIOS; enable P2P in system settings and verify with nvidia-smi topo -p2p w command
C) Firewall rules are blocking NCCL's dynamic port range; open TCP ports 1024-65535 between all nodes in the cluster
D) InfiniBand subnet manager is misconfigured causing partial connectivity; restart the opensm service on the master node and verify all ports are in Active state using ibstat

---

