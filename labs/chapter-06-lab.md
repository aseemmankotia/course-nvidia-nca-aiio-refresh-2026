# Chapter 6 Lab: Kubernetes and Slurm Orchestration for AI Workloads

**Estimated time:** 18 minutes

## Exercise

Orchestration Configuration Lab: Write Kubernetes deployment YAML requesting 2 GPUs with MIG profile. Write Slurm batch script with #SBATCH --gres=gpu:a100:4 requesting 4 A100 GPUs. Compare job submission workflows. Explore Base Command Manager UI screenshots for job queue, GPU utilization, and user quotas.

## Key takeaway

Slurm for batch training jobs (queuing, checkpointing, multi-day runs), Kubernetes for inference services (scaling, load balancing, short-lived containers)—Base Command provides unified management layer.
