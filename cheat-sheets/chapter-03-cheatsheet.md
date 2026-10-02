# GPU Server Architectures Cheat Sheet

## PCIe Bandwidth by Generation (x16 slot)
| Generation | Bandwidth per GPU | Use Case |
|------------|------------------|----------|
| PCIe Gen3 | ~16 GB/s | Legacy servers, low inter-GPU communication |
| PCIe Gen4 | ~32 GB/s | Current standard, data streaming workloads |
| PCIe Gen5 | ~64 GB/s | Latest servers, 2x Gen4, still far below NVLink |

**Exam Key**: PCIe doubles every generation. Even Gen5 is 10x+ slower than NVLink.

## NVLink Bandwidth Comparison
| Platform | Bandwidth per GPU | Technology |
|----------|------------------|------------|
| A100 NVLink | 600 GB/s bidirectional | 12 NVLink lanes |
| H100 NVLink | 900 GB/s bidirectional | 18 NVLink lanes |

**Exam Key**: NVLink = 18-28x faster than PCIe Gen4. Required for distributed training.

## DGX System Specifications
| System | GPUs | NVSwitch Chips | Bisection BW | Per-GPU BW |
|--------|------|----------------|--------------|------------|
| DGX A100 | 8x A100 | 6 | 4.8 TB/s | 600 GB/s |
| DGX H100 | 8x H100 | 18 | 7.2 TB/s | 900 GB/s |

**Exam Key**: DGX = complete NVIDIA system. Turnkey solution. H100 has 3x more switches than A100.

## HGX Platform vs Standalone GPUs
| Configuration | Interconnect | Bandwidth | Best For |
|---------------|--------------|-----------|----------|
| HGX 4-GPU | NVLink mesh | 600 GB/s (A100) | 4-GPU training, OEM integration |
| HGX 8-GPU | NVSwitch fabric | 600-900 GB/s | 8-GPU training, OEM servers |
| PCIe GPUs | PCIe only | 32-64 GB/s | Inference, independent workloads |

**Exam Key**: HGX = baseboard for OEMs. Includes NVSwitch. Standalone PCIe = no NVLink.

## nvidia-smi topo -m Output Codes
| Code | Meaning | Bandwidth | Path Type |
|------|---------|-----------|----------|
| NV# | # NVLink connections | 600-900 GB/s | Direct GPU-to-GPU, high-speed |
| PHB | PCIe Host Bridge | ~32 GB/s | Through CPU/chipset, slow |
| SYS | System connection | Varies | Multi-hop, slowest |

**Command**: `nvidia-smi topo -m` shows GPU interconnect matrix

**Exam Key**: NV## = fast NVLink. Higher number = more lanes. PHB = PCIe bottleneck.

## Architecture Decision Matrix
| Requirement | Recommended Platform | Why |
|-------------|---------------------|-----|
| >400 GB/s GPU-to-GPU | HGX or DGX with NVLink | PCIe cannot provide >64 GB/s |
| 8-GPU distributed training | DGX or HGX 8-GPU | NVSwitch provides all-to-all fabric |
| Multi-tenant inference | HGX 4-GPU or PCIe GPUs | Isolated workloads, don't need NVLink |
| 4-GPU training | HGX 4-GPU baseboard | NVLink mesh, lower cost than DGX |
| Custom server build | HGX baseboard | OEM integration, flexibility |
| Turnkey solution | DGX system | Validated, optimized, support |

## Common Exam Traps
1. **PCIe Gen5 ≠ NVLink replacement**: Even Gen5 x16 (~64 GB/s) is 9x slower than A100 NVLink (600 GB/s)
2. **Individual H100 GPUs ≠ NVLink capable**: Need HGX baseboard with NVSwitch for NVLink
3. **DGX A100 bandwidth**: 4.8 TB/s, NOT 7.2 TB/s (that's H100)
4. **HGX vs DGX**: HGX = baseboard component, DGX = complete system
5. **nvidia-smi topo**: NV12 = 12 NVLink lanes (good), PHB = PCIe bridge (slow)

## Quick Formulas
- **PCIe Bandwidth**: Gen# × 2 GB/s × lanes (e.g., Gen4 x16 = 4 × 2 × 16 = ~32 GB/s)
- **DGX Bisection BW**: GPUs × Per-GPU BW (e.g., 8 × 600 GB/s = 4.8 TB/s for A100)
- **Bandwidth Ratio**: NVLink / PCIe Gen4 ≈ 600/32 ≈ 18-20x faster
