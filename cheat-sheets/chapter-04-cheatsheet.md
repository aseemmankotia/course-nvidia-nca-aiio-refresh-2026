# AI Cluster Networking and Storage Cheat Sheet

## InfiniBand vs Ethernet Comparison

| Feature | InfiniBand NDR | InfiniBand HDR | 400GbE RoCEv2 | 100GbE RoCEv2 |
|---------|---------------|----------------|---------------|---------------|
| Bandwidth | 400 Gb/s (50 GB/s) | 200 Gb/s (25 GB/s) | 400 Gb/s (50 GB/s) | 100 Gb/s (12.5 GB/s) |
| Latency | <700ns | <1μs | 2-5μs | 3-8μs |
| Congestion Control | Credit-based (hardware) | Credit-based (hardware) | PFC (Priority Flow Control) | PFC |
| In-network Compute | SHARP (all-reduce) | SHARP (all-reduce) | No | No |
| Cost per port | $3000-4000 | $2000-3000 | $800-1200 | $400-600 |
| Use Case | >128 GPUs, <1μs needed | 32-128 GPUs, cost-sensitive | Mixed workloads, existing Ethernet | <32 GPUs, dev/test |

**Exam Key**: InfiniBand = lowest latency + SHARP. RoCEv2 = Ethernet compatibility + cost. Choose IB when latency <1μs required.

## RDMA Technologies

**GPUDirect RDMA Benefits**:
- Zero-copy: GPU memory → Network → GPU memory (no system memory)
- Kernel bypass: User-space application posts RDMA verbs directly to NIC
- CPU offload: CPU not involved in data movement
- Latency: 2-5μs vs 20-30μs without GPUDirect

**RoCEv2 (RDMA over Converged Ethernet)**:
- Requires: Lossless Ethernet (PFC enabled), flow control, ECN (Explicit Congestion Notification)
- Protocol: RDMA verbs over UDP/IP
- Compatibility: Works with existing Ethernet switches (with PFC support)
- Limitation: PFC can cause head-of-line blocking affecting latency consistency

**Exam Trap**: GPUDirect requires RDMA-capable network (IB or RoCE). Does NOT work over standard TCP/IP.

## BlueField DPU Architecture

**BlueField-3 Specifications**:
- ARM cores: 16x Arm Cortex-A78 @ 3.0 GHz
- Network: 400 Gb/s or 2x200 Gb/s
- Memory: 16-32 GB DDR5
- Storage: PCIe Gen5 NVMe support

**Offload Capabilities**:
1. **Networking**: RDMA transport (RoCEv2, InfiniBand), overlay networks (VxLAN, GENEVE), IPsec
2. **Storage**: NVMe-oF (NVMe over Fabrics), NVMe/TCP, virtio-blk
3. **Security**: Encryption, firewall, DDoS protection
4. **Result**: 30-40% CPU utilization reduction for network/storage tasks

**Exam Key**: DPU offloads network/storage protocol processing, NOT ML compute. Frees CPU/GPU for training.

## Storage Performance Requirements

**Checkpoint Bandwidth Formula**:
```
Required Bandwidth (GB/s) = Model Size (GB) / Checkpoint Time (seconds)
Example: 350GB model in 2 min = 350 / 120 = 2.92 GB/s
Add 20% headroom: 2.92 × 1.2 = 3.5 GB/s minimum
```

**IOPS vs Bandwidth**:
- **Checkpointing**: Bandwidth-bound (large sequential writes 4MB+). Need 3-20 GB/s, IOPS irrelevant.
- **Data Loading**: Bandwidth-bound (sequential reads). Need 10-50 GB/s depending on dataset size.
- **Metadata Operations**: IOPS-bound (file creation, directory listings). Need 100K-500K IOPS.

## Parallel File Systems Comparison

| System | Architecture | Typical Bandwidth | IOPS | Use Case |
|--------|--------------|-------------------|------|----------|
| Lustre | MDT + OSTs | 3-30 GB/s | 200K-1M | General-purpose, proven, large scale |
| GPFS/Spectrum Scale | NSD servers | 5-40 GB/s | 500K-2M | Enterprise, high reliability |
| WekaFS | NVMe-based OSDs | 20-100 GB/s | 2M-10M | Ultra-high performance, cost-premium |
| NFS | Single server | 0.5-2 GB/s | 20K-50K | Small clusters (<16 nodes), simple |

**Lustre Stripe Count**:
```bash
# Set stripe count for directory (files inherit)
lfs setstripe -c 8 /lustre/checkpoints

# Check current stripe settings
lfs getstripe /lustre/checkpoints

# Optimal stripe count: 4-8 for checkpoints, 1-2 for logs
```

**Exam Key**: Low stripe count (1-2) = common bottleneck. Only few OSTs active despite high total capacity.

## Storage Selection Decision Tree

**Checkpoint Storage**:
- **Need**: High bandwidth (>3 GB/s), low latency (<10ms), parallel writes
- **Solution**: Lustre/GPFS with 20+ OSTs, stripe count 4-8
- **Never**: S3 object storage (high latency, eventual consistency)

**Training Data (initial load)**:
- **Small dataset (<1TB)**: NFS acceptable
- **Medium (1-50TB)**: Lustre/GPFS with high stripe count
- **Large (>50TB)**: Parallel FS or pre-stage from S3 to local NVMe

**Archive/Backup**:
- **Solution**: S3 object storage (cost-effective, durable)
- **Pattern**: Train from parallel FS, archive to S3 post-training

## Common Exam Scenarios

**Scenario 1**: "Sub-5μs latency required" → InfiniBand (NDR or HDR)

**Scenario 2**: "Integrate with existing Ethernet monitoring" → RoCEv2 over 400GbE

**Scenario 3**: "Checkpoint 500GB in <3 minutes" → Need 500/180 = 2.78 GB/s → Lustre with 30+ OSTs

**Scenario 4**: "Variable latency despite low utilization" → Static routing issue, enable adaptive routing

**Scenario 5**: "High CPU during RDMA operations" → Missing GPUDirect RDMA or kernel bypass

**Scenario 6**: "Only 10 OSTs active out of 60" → Stripe count set to 1, increase to 4-8
