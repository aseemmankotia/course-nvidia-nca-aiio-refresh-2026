# Chapter 3 Lab: GPU Server Architectures: DGX, HGX, and Multi-GPU Interconnects

**Estimated time:** 18 minutes

## Exercise

Topology Analysis Lab: Review provided nvidia-smi topo -m output samples from DGX A100 vs 8-GPU PCIe server. Identify NVLink connections (NV12, NV18 links), PCIe switches, NUMA nodes. Calculate effective bandwidth for all-reduce operation. Compare DGX NVSwitch fabric (all-to-all) vs tree-based PCIe topology.

## Key takeaway

NVLink connects GPU pairs, NVSwitch creates all-to-all fabric, PCIe creates bottlenecks—DGX uses NVSwitch for maximum training bandwidth (600 GB/s A100, 900 GB/s H100).
