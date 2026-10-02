# Chapter 4 Lab: AI Cluster Networking and Storage Infrastructure

**Estimated time:** 15 minutes

## Exercise

Networking Decision Matrix Lab: Complete capacity-planning worksheet comparing InfiniBand NDR 400G ($8K/port) vs Ethernet 400G ($2K/port) for 64-GPU cluster. Calculate total bandwidth, latency impact on training time (provided formulas), ROI break-even. Analyze BlueField-3 DPU spec sheet: identify offloaded functions (encryption, compression, NVMe-oF).

## Key takeaway

InfiniBand for large training clusters (>64 GPUs, low latency), Ethernet for inference and small clusters (cost-effective)—RDMA (InfiniBand or RoCE) enables GPUDirect for faster GPU communication.
