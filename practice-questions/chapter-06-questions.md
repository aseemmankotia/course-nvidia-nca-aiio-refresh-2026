## Chapter 6: Kubernetes and Slurm Orchestration for AI Workloads — Practice Questions

**Q1.** A financial services company runs containerized fraud detection inference services that must scale horizontally based on API request load and maintain 99.9% uptime. They also run weekly deep learning model retraining jobs on historical transaction data using multi-node GPU clusters. Which orchestration solution BEST meets these requirements?

A) Use Kubernetes for both inference services and training jobs with StatefulSets for training workloads
B) Use Kubernetes for inference services with HPA (Horizontal Pod Autoscaler) and Slurm for batch training jobs
C) Use Slurm for both inference and training with GPU gres allocation for all workloads
D) Use NVIDIA Base Command Platform exclusively for unified management of all AI workloads

<details>
<summary>Answer</summary>

**Correct: B**

Kubernetes excels at containerized microservices with auto-scaling (HPA) and high availability for inference workloads. Slurm is purpose-built for HPC batch jobs with sophisticated GPU scheduling via gres (generic resources). This hybrid approach uses each tool for its strength: K8s for always-on services, Slurm for batch training.

**Why the others are wrong:**
- StatefulSets in Kubernetes are designed for stateful apps with persistent identity (databases), not optimal for distributed training which needs gang scheduling and MPI coordination that Slurm provides
- Slurm is a batch scheduler, not designed for long-running services or auto-scaling based on real-time metrics like API load - lacks native HTTP routing and service discovery
- Base Command Platform is a management layer that can work WITH Kubernetes/Slurm but isn't a replacement orchestrator - it provides job submission interface and monitoring for underlying schedulers

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A machine learning engineer needs to deploy a GPU-accelerated PyTorch inference service on Kubernetes. The container requires exactly 2 NVIDIA A100 GPUs and must not be scheduled on nodes with fewer GPUs available. Which Kubernetes manifest configuration is correct?

A) resources: requests: gpu: 2 limits: gpu: 4
B) resources: requests: nvidia.com/gpu: 2 limits: nvidia.com/gpu: 2
C) resources: requests: nvidia.com/gpu: 2 limits: nvidia.com/gpu: 4
D) nodeSelector: accelerator: nvidia-tesla-a100 with resources: requests: gpu: 2

<details>
<summary>Answer</summary>

**Correct: B**

nvidia.com/gpu is the extended resource exposed by NVIDIA GPU device plugin. For GPUs, requests MUST equal limits (GPUs are not overcommittable). Setting requests=limits=2 ensures the pod gets exactly 2 GPUs and won't start unless 2 are available on a node.

**Why the others are wrong:**
- gpu is not a valid resource name - must use nvidia.com/gpu as the extended resource registered by the device plugin
- Setting limits higher than requests (2 vs 4) is invalid for GPUs - Kubernetes treats GPUs as whole-unit resources that cannot be overcommitted, so requests must equal limits
- nodeSelector alone doesn't request GPU resources - it only selects node labels. Without resources.requests.nvidia.com/gpu the device plugin won't allocate GPUs to the container

</details>

---

**Q3.** A research team submits a Slurm job requiring 8 GPUs across 4 nodes (2 GPUs per node) for distributed training. The cluster has nodes with 4 GPUs each. Which Slurm directive combination correctly requests this configuration?

A) #SBATCH --nodes=4 --ntasks-per-node=1 --gres=gpu:2
B) #SBATCH --nodes=4 --gpus=8 --gpus-per-node=2
C) #SBATCH --nodes=4 --ntasks=4 --gres=gpu:8
D) #SBATCH --nodes=4 --cpus-per-task=2 --gres=gpu:a100:2

<details>
<summary>Answer</summary>

**Correct: A**

--nodes=4 allocates 4 nodes. --gres=gpu:2 requests 2 GPUs per node via generic resources. --ntasks-per-node=1 ensures one task per node (common for distributed training frameworks that spawn processes internally). Total: 4 nodes × 2 GPUs = 8 GPUs.

**Why the others are wrong:**
- --gpus=8 --gpus-per-node=2 is contradictory: --gpus is total GPUs across job, --gpus-per-node specifies per-node count. Slurm would interpret this as 8 total with 2 per node = only 4 nodes max, but --nodes=4 might cause conflict depending on version
- --gres=gpu:8 requests 8 GPUs per node (not total), which would try to allocate 4 nodes × 8 GPUs = 32 GPUs total - exceeds requirement and likely unavailable
- --gres=gpu:a100:2 specifies GPU type (a100) which is valid syntax but --cpus-per-task=2 only requests 2 CPUs per task, doesn't relate to GPU count - missing the gpu allocation

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** An AI inference service deployment must ensure that no two replica pods run on the same GPU node to maximize fault tolerance. The cluster has nodes labeled with gpu-type: a100. Which Kubernetes configuration achieves this?

A) Use nodeSelector: gpu-type: a100 with requiredDuringSchedulingIgnoredDuringExecution
B) Use podAffinity with topologyKey: gpu-type and preferredDuringSchedulingIgnoredDuringExecution
C) Use podAntiAffinity with topologyKey: kubernetes.io/hostname and labelSelector matching the pod labels
D) Use nodeAffinity with requiredDuringSchedulingIgnoredDuringExecution matching gpu-type: a100

<details>
<summary>Answer</summary>

**Correct: C**

podAntiAffinity with topologyKey: kubernetes.io/hostname ensures pods are spread across different nodes (hostname uniquely identifies nodes). labelSelector matches the deployment's pod labels to apply anti-affinity to replicas of the same service. This guarantees no two replicas on same node.

**Why the others are wrong:**
- nodeSelector only selects nodes with specific labels but doesn't prevent multiple pods on same node - all replicas could land on one a100 node if it has capacity
- podAffinity attracts pods together (opposite of requirement). preferredDuringScheduling is soft preference not guarantee. topologyKey: gpu-type is invalid - must be a node label key like hostname or zone
- nodeAffinity selects which nodes pods can run on (node selection) but doesn't control pod-to-pod placement - all replicas could still schedule on the same qualifying node

</details>

---

**Q5.** A company wants to manage their on-premises DGX cluster with centralized user management, job scheduling, and monitoring. They need integration with Active Directory for authentication and want a web UI for job submission. Which NVIDIA solution is designed for this?

A) NVIDIA NGC Catalog
B) Kubernetes with NVIDIA GPU Operator
C) NVIDIA Triton Inference Server
D) NVIDIA Base Command Manager (BCM)

<details>
<summary>Answer</summary>

**Correct: D**

Base Command Manager (BCM) is NVIDIA's on-prem cluster management software specifically for DGX systems. Provides web UI, job scheduling (works with Slurm underneath), user/group management with LDAP/AD integration, monitoring dashboards, and resource allocation policies.

**Why the others are wrong:**
- NGC Catalog is a repository of GPU-optimized containers, models, and Helm charts - not a cluster management platform, just a software distribution mechanism
- GPU Operator automates GPU driver/toolkit deployment on Kubernetes but doesn't provide centralized cluster management, AD integration, or job scheduling UI for HPC workloads
- Triton Inference Server is for serving ML models (inference) with multi-framework support - not a cluster management or job scheduling system

</details>

---

**Q6.** A data scientist submits a training job to a Slurm cluster and wants to ensure it runs on nodes with NVIDIA A100 80GB GPUs specifically, not A100 40GB. The cluster has gres configured with GPU types. Which Slurm directive is correct?

A) #SBATCH --gres=gpu:4 --constraint=a100_80gb
B) #SBATCH --gpus=a100:4 --mem-per-gpu=80G
C) #SBATCH --gres=gpu:a100-80gb:4
D) #SBATCH --partition=a100_partition --gres=gpu:4

<details>
<summary>Answer</summary>

**Correct: C**

--gres=gpu:<type>:<count> is the correct syntax to request specific GPU types via generic resources. If gres.conf defines gpu types like a100-80gb, this directive requests exactly 4 of those GPUs. Slurm will only schedule on nodes advertising that specific GPU type.

**Why the others are wrong:**
- --constraint is for general node features in Slurm but --gres doesn't natively parse constraint in this form - you'd use --gres=gpu:a100-80gb:4 if GPU types are defined in gres.conf, or separate --constraint for node features
- --gpus=a100:4 syntax is valid in newer Slurm but --mem-per-gpu requests memory per GPU (system RAM) not GPU memory type - doesn't differentiate between 40GB and 80GB A100 variants
- --partition selects a job queue but doesn't guarantee GPU type unless partition is exclusively configured with a100-80gb nodes - less precise than gres type specification which works across partitions

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** A Kubernetes cluster runs mixed workloads: inference services needing 1 GPU each and training jobs needing 8 GPUs. Training jobs should only run when 8 GPUs are available on a single node. Which approach prevents training pods from fragmenting GPU resources?

A) Use PodDisruptionBudget with minAvailable: 8 for training pods
B) Set resources.requests.nvidia.com/gpu: 8 and use nodeSelector to choose 8-GPU nodes
C) Configure nodeAffinity with requiredDuringSchedulingIgnoredDuringExecution for 8-GPU nodes only
D) Use ResourceQuota to limit inference pods to 7 GPUs total per namespace

<details>
<summary>Answer</summary>

**Correct: B**

resources.requests.nvidia.com/gpu: 8 ensures pod won't schedule unless 8 GPUs are available. nodeSelector targeting nodes with label like nvidia.com/gpu.count=8 ensures scheduling only on nodes that have 8 GPUs total. Combined, this prevents scheduling on nodes with fewer GPUs or partial availability.

**Why the others are wrong:**
- PodDisruptionBudget controls how many pods can be voluntarily disrupted during drains/evictions - doesn't affect initial scheduling or GPU allocation, only availability during updates
- nodeAffinity alone selects eligible nodes but if an 8-GPU node already has 5 GPUs allocated to inference pods, training pod could get scheduled and wait (Pending) - doesn't guarantee atomic 8-GPU availability
- ResourceQuota limits total resources in a namespace but doesn't control per-node allocation - inference pods could still fragment GPUs across nodes, and quota doesn't guarantee 8 GPUs on one node for training

</details>

---

**Q8.** NVIDIA Base Command Platform (BCP) provides job scheduling and data management for AI workloads. Which statement accurately describes BCP's architecture?

A) BCP replaces Slurm entirely with a proprietary GPU scheduler optimized for transformer models
B) BCP is a SaaS platform that requires migrating workloads to NVIDIA cloud infrastructure
C) BCP provides a unified interface for job submission and monitoring while using Slurm or Kubernetes as the underlying scheduler
D) BCP only works with DGX SuperPOD systems and cannot integrate with commodity GPU servers

<details>
<summary>Answer</summary>

**Correct: C**

Base Command Platform is a management layer/portal that sits above schedulers like Slurm or Kubernetes. Users submit jobs through BCP web UI or CLI, and BCP translates to underlying scheduler (Slurm for HPC, K8s for containers). Provides unified monitoring, dataset management, and job tracking across different execution backends.

**Why the others are wrong:**
- BCP works WITH Slurm, not replacing it - uses Slurm for batch scheduling underneath while providing enhanced UI, data management, and multi-tenancy features on top
- BCP can be deployed on-premises or in cloud - it's software that manages your infrastructure, not a SaaS that hosts workloads. NVIDIA offers cloud services separately but BCP itself runs in your environment
- BCP supports DGX systems but also works with certified servers from partners (Dell, HPE, Lenovo) running NVIDIA GPUs - not exclusive to SuperPOD, can manage heterogeneous GPU clusters

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

