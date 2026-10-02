# Kubernetes and Slurm Orchestration for AI Workloads - Exam Cheat Sheet

## When to Use What (Heavily Tested)

| Use Case | Orchestrator | Why |
|----------|-------------|-----|
| Real-time inference API | Kubernetes | Auto-scaling, load balancing, service discovery, rolling updates |
| Batch model training | Slurm | Gang scheduling, MPI support, GPU affinity, backfill scheduling |
| MLOps pipeline (containers) | Kubernetes | CI/CD integration, Helm charts, pod lifecycle hooks |
| Multi-node HPC simulation | Slurm | InfiniBand support, tightly-coupled jobs, resource reservation |
| Microservices architecture | Kubernetes | Service mesh, ingress, ConfigMaps, secrets management |

**Exam Rule:** Kubernetes = long-running services with dynamic scaling. Slurm = batch jobs with fixed resource allocation.

## Kubernetes GPU Configuration

### GPU Resource Requests (MUST KNOW)
```yaml
resources:
  requests:
    nvidia.com/gpu: 2  # requests = limits (no overcommit)
  limits:
    nvidia.com/gpu: 2
```
- Extended resource: `nvidia.com/gpu`
- Integer values only (whole GPUs)
- Requests MUST equal limits
- Requires NVIDIA GPU device plugin DaemonSet

### Pod Placement for GPUs

**Node Selector (simple):**
```yaml
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
```

**Pod Anti-Affinity (spread replicas):**
```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - topologyKey: kubernetes.io/hostname
      labelSelector:
        matchLabels:
          app: inference-service
```
**Exam Tip:** `kubernetes.io/hostname` = one pod per node. `topology.kubernetes.io/zone` = one pod per availability zone.

## Slurm GPU Scheduling

### Gres (Generic Resources) Syntax
```bash
# 4 nodes, 2 GPUs per node = 8 total
sbatch --nodes=4 --gres=gpu:2 train.sh

# Specific GPU type
sbatch --gres=gpu:a100:4 train.sh

# GPU type with 80GB memory variant
sbatch --gres=gpu:a100-80gb:2 train.sh
```

### Common Slurm Directives for AI
| Directive | Purpose | Example |
|-----------|---------|----------|
| `--nodes=N` | Number of nodes | `--nodes=4` |
| `--gres=gpu:N` | GPUs per node | `--gres=gpu:2` |
| `--ntasks-per-node=N` | Tasks per node (usually 1 for DL) | `--ntasks-per-node=1` |
| `--cpus-per-task=N` | CPUs per task | `--cpus-per-task=16` |
| `--mem=XG` | Memory per node | `--mem=256G` |
| `--time=HH:MM:SS` | Max runtime | `--time=48:00:00` |

**Exam Gotcha:** `--gres=gpu:4` is 4 GPUs **per node**. `--gpus=4` is 4 GPUs **total across job**.

## NVIDIA Base Command Platform

### Base Command Manager (BCM)
- **What:** On-prem cluster management for DGX systems
- **Features:** Web UI, LDAP/AD integration, job submission, monitoring, resource policies
- **Backend:** Works with Slurm scheduler underneath
- **Target:** Enterprise on-prem AI infrastructure

### Base Command Platform (BCP)
- **What:** Unified job scheduling and data management platform
- **Deployment:** On-prem or cloud
- **Schedulers:** Integrates with Slurm (batch) or Kubernetes (containers)
- **Features:** Dataset versioning, job templates, multi-tenancy, usage tracking
- **Exam Note:** BCP is management layer, not replacement for Slurm/K8s

## Key Architecture Patterns

### Hybrid Orchestration (Common Exam Scenario)
- **Inference:** Kubernetes with HPA, 1-2 GPUs per pod
- **Training:** Slurm batch jobs, 8+ GPUs multi-node
- **Data Prep:** Kubernetes Jobs or Slurm array jobs
- **Why:** Each tool optimized for its workload type

### GPU Device Plugin Architecture
1. Device plugin DaemonSet discovers GPUs on nodes
2. Registers `nvidia.com/gpu` extended resource with kubelet
3. Kubelet advertises GPU capacity to scheduler
4. Scheduler places pods based on `nvidia.com/gpu` requests
5. Device plugin allocates specific GPU IDs to containers

## Common Exam Traps

1. **GPU overcommit:** Kubernetes does NOT allow GPU overcommit. `requests < limits` is invalid.
2. **Gres vs gpus:** `--gres=gpu:4` = 4 per node. `--gpus=4` = 4 total.
3. **Pod anti-affinity:** Use `kubernetes.io/hostname` not custom labels for node spread.
4. **BCP role:** It's a management layer, not a scheduler replacement.
5. **Node selector alone:** Doesn't guarantee resources - multiple pods can land on same node without anti-affinity.
