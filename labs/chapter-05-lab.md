# Chapter 5 Lab: GPU Virtualization, Multi-Tenancy, and Capacity Planning

**Estimated time:** 20 minutes

## Exercise

Capacity Planning Worksheet Lab: Given requirements (train 50M-parameter model, 10TB dataset, 2-week deadline), calculate GPU count needed using provided throughput benchmarks. Configure MIG partitions on simulated A100 (7 instances: 1g.5gb, 2g.10gb, etc.). Calculate power/cooling for 10-rack AI cluster (20 DGX H100 per rack).

## Key takeaway

MIG for multi-tenant container/bare-metal QoS isolation, vGPU for VM workloads—always calculate power as GPU_watts + system_overhead, then multiply by 1.3-1.5x for PSU and cooling.
