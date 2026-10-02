## Chapter 3: GPU Server Architectures: DGX, HGX, and Multi-GPU Interconnects — Practice Questions

**Q1.** A research team is training a 175B parameter language model across 8x A100 GPUs in a DGX A100 system. They observe that gradient synchronization is taking 40% of their training time. Which architectural feature of the DGX A100 should they leverage to reduce this overhead?

A) Increase PCIe Gen4 lanes from x16 to x32 to double GPU-to-GPU bandwidth
B) Enable GPU Direct RDMA to allow GPUs to communicate through the network card
C) Use PCIe peer-to-peer transfers with BAR1 memory mapping
D) Configure the 6x NVSwitch fabric to enable direct all-to-all GPU communication bypassing CPU

<details>
<summary>Answer</summary>

**Correct: D**

DGX A100 has 6x NVSwitch chips creating a full NVLink fabric that provides 600 GB/s bidirectional bandwidth per GPU with direct all-to-all connectivity. This eliminates PCIe bottlenecks and CPU involvement for gradient synchronization, drastically reducing collective operation latency.

**Why the others are wrong:**
- PCIe lanes are fixed at x16 per slot in standard configurations and cannot be increased to x32; even if possible, PCIe Gen4 x16 provides only ~32 GB/s vs NVLink's 600 GB/s
- GPU Direct RDMA is for GPU-to-network communication, not inter-GPU communication within a node; it doesn't help with local gradient synchronization
- PCIe peer-to-peer is limited to ~32 GB/s and requires CPU root complex involvement; NVSwitch provides 18x higher bandwidth with direct GPU-GPU paths

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An MLOps engineer runs 'nvidia-smi topo -m' on a server and sees 'NV12' between GPU0 and GPU1, but 'PHB' between GPU0 and GPU7. What does this topology indicate about the server architecture?

A) The server has 12 NVLink connections total with GPUs 0-1 directly connected but 0 and 7 on different PCIe host bridges
B) GPU0 and GPU1 have NVLink Gen12 connections, while GPU0 and GPU7 communicate through PCIe host bridge
C) The system has 12 NVSwitch chips connecting all GPUs, with PHB indicating phase bridge connections
D) GPU pairs 0-1 share NVLink, while 0 and 7 require going through 12 PCIe hops

<details>
<summary>Answer</summary>

**Correct: A**

NV12 indicates 12 NVLink connections between GPU0 and GPU1 (direct high-speed link). PHB means the GPUs are on different PCIe Host Bridges, requiring communication through CPU/chipset at much lower bandwidth (~32 GB/s vs 600 GB/s for NVLink).

**Why the others are wrong:**
- NV12 refers to 12 NVLink lanes, not 'Gen12'; there is no such thing as NVLink Gen12
- NVSwitch systems show 'NV##' between all GPU pairs; PHB indicates PCIe topology without NVSwitch fabric
- The number after NV is the count of NVLink connections, not PCIe hops; PHB is a single bridge, not 12 hops

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A cloud provider is designing a server for multi-tenant AI inference workloads where each tenant gets 1-2 GPUs. Which solution provides the best cost-performance ratio?

A) HGX H100 4-GPU baseboard in a standard 2U server with PCIe Gen5 connectivity
B) DGX H100 system with 8x H100 GPUs and NVSwitch for maximum tenant density
C) Eight separate 1U servers each with a single H100 GPU connected via Ethernet
D) DGX A100 system partitioned using MIG to create 56 GPU instances

<details>
<summary>Answer</summary>

**Correct: A**

For isolated inference workloads not requiring GPU-to-GPU communication, HGX baseboard in standard servers avoids the premium cost of NVSwitch fabric (DGX) while providing PCIe Gen5 bandwidth (~128 GB/s) sufficient for inference. More cost-effective than DGX and easier to manage than separate servers.

**Why the others are wrong:**
- DGX H100's 7.2 TB/s NVSwitch fabric is unnecessary overhead for isolated inference tenants; paying for interconnect bandwidth that won't be used
- Eight 1U servers have 8x the overhead cost (power supplies, NICs, management), higher failure rates, and increased operational complexity
- MIG creates instances on same physical GPUs with shared memory/compute, not true isolation; inference workloads need dedicated GPU resources for predictable latency

</details>

---

**Q4.** A financial services company needs to train fraud detection models on 4 GPUs with 450 GB/s aggregate GPU-to-GPU bandwidth for their distributed training pipeline. The data science team is evaluating options. Which configuration meets the minimum bandwidth requirement at lowest cost?

A) 4x A100 GPUs in a PCIe Gen4 server (32 GB/s per GPU = 128 GB/s total)
B) DGX A100 using only 4 of the 8 GPUs (600 GB/s per GPU with NVSwitch)
C) 4x H100 GPUs in PCIe Gen5 server (64 GB/s per GPU = 256 GB/s total)
D) HGX A100 4-GPU baseboard with full NVLink mesh (600 GB/s per GPU bidirectional)

<details>
<summary>Answer</summary>

**Correct: D**

HGX A100 4-GPU provides full NVLink mesh with 600 GB/s bidirectional per GPU, exceeding the 450 GB/s requirement. It's purpose-built for 4-GPU configurations and costs less than DGX while meeting bandwidth needs. Exam tip: HGX = OEM platform, DGX = complete NVIDIA system.

**Why the others are wrong:**
- PCIe Gen4 x16 provides only ~32 GB/s per GPU, total system bandwidth ~128 GB/s falls far short of 450 GB/s requirement
- Using only 4 GPUs of DGX A100 wastes the cost of 4 unused GPUs and full 6-switch fabric; HGX 4-GPU is designed for this use case
- PCIe Gen5 ~64 GB/s per GPU totals ~256 GB/s, still below 450 GB/s requirement; PCIe cannot match NVLink bandwidth

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** During model training on a DGX H100, a researcher notices GPU7 is 30% slower than other GPUs at receiving gradient updates. Using 'nvidia-smi topo -m', all GPU pairs show 'NV18'. What is the most likely cause?

A) GPU7 has a hardware defect in its NVLink controllers
B) One of the 18 NVSwitch chips has failed, creating an asymmetric fabric
C) The system is running in PCIe fallback mode due to SBIOS misconfiguration
D) GPU7's NVLink connections are running at NVLink 3.0 instead of 4.0

<details>
<summary>Answer</summary>

**Correct: B**

DGX H100 has 18 NVSwitch chips providing full bisection bandwidth. If one switch fails, the fabric remains functional but becomes asymmetric—some GPU-to-GPU paths have reduced bandwidth. 'nvidia-smi topo -m' may still show NV18 for direct connections, but effective bandwidth to GPU7 drops if its primary switch paths are down. Exam tip: NVSwitch redundancy allows partial failures.

**Why the others are wrong:**
- If GPU7 NVLink controllers were defective, topology would show PHB or SYS for connections to GPU7, not NV18
- PCIe fallback would show PHB or SYS in topology matrix for all GPUs, not NV18; all GPUs would be affected equally
- All H100 GPUs use NVLink 4.0; there's no mixed-version scenario; topology would reflect degraded link speed with lower NV number

</details>

---

**Q6.** A hardware vendor is building a custom AI server and needs to choose between integrating an HGX H100 8-GPU baseboard or assembling 8 individual H100 GPUs with PCIe connections. The workload requires 5 TB/s of total GPU-to-GPU bandwidth. Which option meets the requirement?

A) 8x individual H100 GPUs with PCIe Gen5 switches providing 512 GB/s total bandwidth
B) Either option meets 5 TB/s requirement as both support NVLink connectivity
C) Neither option meets 5 TB/s; requires dual HGX baseboards with NVLink domain bridging
D) HGX H100 8-GPU baseboard with NVLink fabric providing 7.2 TB/s bisection bandwidth

<details>
<summary>Answer</summary>

**Correct: D**

HGX H100 8-GPU includes the NVSwitch fabric providing 900 GB/s per GPU and 7.2 TB/s bisection bandwidth, exceeding the 5 TB/s requirement. This is the platform OEMs integrate into servers. Exam tip: HGX includes NVSwitch; individual GPUs use only PCIe.

**Why the others are wrong:**
- Individual PCIe H100 GPUs cannot use NVLink without the HGX baseboard's NVSwitch fabric; PCIe Gen5 x16 provides ~64 GB/s per GPU, total ~512 GB/s falls far short of 5 TB/s
- Individual H100 GPUs in PCIe slots do NOT support NVLink connectivity; NVLink requires the HGX baseboard with NVSwitch chips
- Single HGX H100 8-GPU baseboard provides 7.2 TB/s, which exceeds 5 TB/s requirement; no dual baseboard needed

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** An infrastructure architect needs to decide between PCIe Gen4 and Gen5 for a server with 8x H100 GPUs that will NOT use NVLink. The workload streams training data from NVMe SSDs to GPUs. What is the bandwidth difference per GPU slot?

A) PCIe Gen4 x16: ~32 GB/s vs Gen5 x16: ~64 GB/s per GPU (2x improvement)
B) PCIe Gen4 x16: ~16 GB/s vs Gen5 x16: ~32 GB/s per GPU (2x improvement)
C) PCIe Gen4 x16: ~64 GB/s vs Gen5 x16: ~128 GB/s per GPU (2x improvement)
D) No difference; both Gen4 and Gen5 are limited to 32 GB/s by GPU memory controllers

<details>
<summary>Answer</summary>

**Correct: A**

PCIe Gen4 x16 provides ~32 GB/s bidirectional; Gen5 x16 provides ~64 GB/s bidirectional (doubles bandwidth). For data streaming workloads without GPU-to-GPU communication, Gen5 can be beneficial. Exam tip: Each PCIe generation doubles bandwidth; Gen4 x16 = ~32 GB/s, Gen5 x16 = ~64 GB/s.

**Why the others are wrong:**
- PCIe Gen4 x16 is ~32 GB/s, not ~16 GB/s; the numbers are incorrect
- PCIe Gen4 x16 is ~32 GB/s, not ~64 GB/s; these values are doubled incorrectly
- GPU memory controllers support much higher bandwidth than PCIe; HBM3 memory provides >3 TB/s, not limited by PCIe specs

</details>

---

**Q8.** A data center operator receives a DGX A100 and runs diagnostics. The system reports 4.8 TB/s total fabric bandwidth instead of the expected value. All 8 GPUs and 6 NVSwitches pass hardware tests. What is the most likely explanation?

A) Two NVSwitch chips are running in degraded mode, reducing bandwidth by 33%
B) 4.8 TB/s is correct for DGX A100; the expected value was for DGX H100 which has 7.2 TB/s
C) The system is configured for 4-GPU mode, utilizing only half the fabric
D) NVLink is running at 2.0 speeds instead of 3.0 due to firmware version mismatch

<details>
<summary>Answer</summary>

**Correct: B**

DGX A100 with 8x A100 GPUs and 6x NVSwitch provides 600 GB/s per GPU with 4.8 TB/s bisection bandwidth. DGX H100 with 8x H100 GPUs and 18x NVSwitch provides 900 GB/s per GPU with 7.2 TB/s bisection bandwidth. 4.8 TB/s is correct for A100. Exam tip: A100 = 4.8 TB/s, H100 = 7.2 TB/s.

**Why the others are wrong:**
- If 2 of 6 switches were degraded, bandwidth would be more severely impacted than 33%; all switches passed hardware tests
- DGX A100 doesn't have a '4-GPU mode' configuration; all 8 GPUs are part of unified fabric
- A100 uses NVLink 3.0; there's no fallback to 2.0; firmware mismatch would cause link failures, not speed reduction

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

