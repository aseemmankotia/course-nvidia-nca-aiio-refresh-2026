## Chapter 3: GPU Server Architectures: DGX, HGX, and Multi-GPU Interconnects — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | PCIe Gen4 x16 bandwidth per GPU? | ~32 GB/s bidirectional. Exam tip: Gen3 = ~16 GB/s, Gen4 = ~32 GB/s, Gen5 = ~64 GB/s. Each generation doubles bandwidth. Much slower than NVLink (600+ GB/s). |
| 2 | NVLink bandwidth per GPU in A100? | 600 GB/s bidirectional. Exam tip: 18-20x faster than PCIe Gen4. A100 has 12 NVLink lanes per GPU. This is for point-to-point or when using NVSwitch fabric. |
| 3 | NVSwitch bandwidth per GPU in H100? | 900 GB/s bidirectional per GPU. Exam tip: H100 improved from A100's 600 GB/s. DGX H100 uses 18 NVSwitch chips (vs 6 in DGX A100) for this bandwidth. |
| 4 | DGX A100 total bisection bandwidth? | 4.8 TB/s. Exam tip: 8x A100 GPUs × 600 GB/s = 4.8 TB/s. Uses 6 NVSwitch chips. Compare to DGX H100's 7.2 TB/s. |
| 5 | DGX H100 total bisection bandwidth? | 7.2 TB/s. Exam tip: 8x H100 GPUs × 900 GB/s = 7.2 TB/s. Uses 18 NVSwitch chips (3x more than A100's 6). 50% more bandwidth than DGX A100. |
| 6 | When to use HGX platform vs standalone PCIe GPUs? | HGX when workload needs GPU-to-GPU bandwidth >200 GB/s (NVLink required). Standalone PCIe when GPUs work independently (inference, isolated training). Exam tip: HGX includes NVSwitch fabric; PCIe GPUs limited to ~32-64 GB/s. |
| 7 | What does 'NV12' mean in nvidia-smi topo output? | 12 NVLink connections between two GPUs (direct high-speed link). Exam tip: Higher NV number = more NVLink lanes = higher bandwidth. Typical A100: NV12. Compare to 'PHB' (PCIe host bridge, much slower). |
| 8 | What does 'PHB' mean in nvidia-smi topo output? | PCIe Host Bridge - GPUs on different PCIe root complexes, communicate through CPU/chipset at PCIe speeds (~32 GB/s). Exam tip: PHB = slow path. NV## = fast NVLink path. |
| 9 | Command to check GPU interconnect topology? | nvidia-smi topo -m. Exam tip: Shows matrix of GPU connections. NV## = NVLink lanes. PHB = PCIe bridge. SYS = multi-hop. Critical for diagnosing performance issues. |
| 10 | DGX A100: how many GPUs and NVSwitch chips? | 8x A100 GPUs, 6x NVSwitch chips. Exam tip: Creates full NVLink fabric with 600 GB/s per GPU. All-to-all connectivity. Compare to DGX H100: 8 GPUs but 18 NVSwitch chips. |
| 11 | DGX H100: how many GPUs and NVSwitch chips? | 8x H100 GPUs, 18x NVSwitch chips (3x more than A100). Exam tip: More switches = higher bandwidth (900 GB/s per GPU vs 600 GB/s). Total: 7.2 TB/s bisection bandwidth. |
| 12 | HGX baseboard vs DGX system - key difference? | HGX = GPU+NVSwitch baseboard for OEMs to integrate into servers. DGX = complete turnkey system from NVIDIA. Exam tip: HGX gives flexibility, DGX is validated/optimized end-to-end. Both use same GPU+NVSwitch tech. |
