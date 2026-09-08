---
id: 32-hardware-and-kernels/collectives-heterogeneous-topology
title: "Optimal Collective Algorithms for Heterogeneous Interconnect Topologies"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Collective Algorithms for Heterogeneous Interconnect Topologies

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/collectives-heterogeneous-topology` · **Status:** open

## 1. Problem Statement

Given a concrete cluster interconnect — NVLink/NVSwitch inside a node, PCIe to the NICs, InfiniBand or Ethernet with a fat-tree or rail-optimized fabric between nodes — find the collective communication schedule (all-reduce, all-gather, reduce-scatter, all-to-all) that minimizes completion time for a given message size, and prove it optimal or bound its gap to optimal.

Three variants, of very different difficulty:

- **Theory variant.** For a capacitated graph with per-link latency and bandwidth, is the minimum-makespan all-reduce schedule computable in polynomial time? Minimum broadcast time in a general graph is already NP-complete (Garey & Johnson, ND49), so the interesting question is which *structured* topology families admit exact polynomial algorithms and what the approximation ratio is elsewhere.
- **Method variant.** Given a topology description, synthesize a schedule that beats the hand-written vendor library (NCCL/RCCL) on real hardware. This is what SCCL, TACCL and Blink attack.
- **Measurement variant.** Decide whether a schedule is optimal *on the machine*, not in the cost model. This requires a link-level cost model whose parameters are stable enough that "optimal under the model" predicts "fastest in wall-clock".

Solving it means: an algorithm that takes a topology plus a message size and emits a schedule with a proven constant-factor guarantee against the true machine optimum, verified by measurement across at least two vendor fabrics.

## 2. Formal Setting

Interconnect as a directed multigraph $G=(V,E)$. $V$ = accelerators plus switch nodes. Each edge $e$ has:

- $\beta_e$ — achieved unidirectional bandwidth in bytes/s. **Measured** as the asymptotic slope of a two-endpoint `ib_write_bw` / `p2pBandwidthLatencyTest` sweep, not the nameplate figure.
- $\alpha_e$ — per-message latency in seconds. **Measured** as the intercept of the same sweep at message size $\to 0$.

A transfer of $b$ bytes on $e$ costs $\alpha_e + b/\beta_e$ (Hockney model). Node-level constraints: for GPU $v$, $\sum_{e \in \text{out}(v)} r_e \le \Beta_v^{\text{out}}$ and likewise inbound, where $r_e$ is instantaneous rate — this captures the NIC and PCIe root-complex bottleneck that per-link capacities alone miss.

All-reduce input: each of $p$ GPUs holds a vector $x_i \in \mathbb{R}^{S/4}$ ($S$ bytes, fp32). Output: every GPU holds $\sum_i x_i$. A **schedule** is a set of chunk-transfer events $(c, e, t)$ — chunk $c$ crosses edge $e$ starting at $t$ — respecting data dependence (a reduction of chunk $c$ cannot be sent before its operands arrive) and the capacity constraints. Objective:

$$T^\star(G,S,p) \;=\; \min_{\text{schedules}} \; \max_{(c,e,t)} \big(t + \alpha_e + |c|/\beta_e\big).$$

Bandwidth lower bound via cuts: for any bipartition $(A,\bar A)$ with cut capacity $B_{A}=\sum_{e: A\to\bar A}\beta_e$, all-reduce requires at least $\tfrac{|A|\wedge|\bar A|}{p}\,S$ bytes... more usefully, for $p$-GPU ring/rabenseifner on a homogeneous ring, the classical bound is

$$T \;\ge\; 2\frac{p-1}{p}\cdot\frac{S}{\beta} \quad\text{(bandwidth term, Patarasuk \& Yuan 2009)} \qquad T \;\ge\; 2\log_2 p \cdot \alpha \quad\text{(latency term)}.$$

**Assumptions, and which break.**

- *Linear cost, no congestion.* Violated: shared fat-tree uplinks under ECMP hash collisions give 30–60% bandwidth loss on some flow placements.
- *Static $\beta_e$.* Violated: NVLink SHARP in-switch reduction changes effective $\beta$ by operation type; power/thermal capping shifts sustained bandwidth run to run.
- *Full-duplex, independent links.* Violated: PCIe and NIC share a root complex; concurrent send/recv on 8 NICs rarely reaches $8\times$ single-NIC.
- *Compute is free.* Violated: reduction kernels consume SM cycles that the overlapping compute also wants.

## 3. State of the Art

**Theory SOTA.** Optimal algorithms are known only for regular topologies. Bandwidth-optimal all-reduce on a ring (Patarasuk & Yuan, *JPDC* 2009); recursive halving/doubling with Rabenseifner's reduce-scatter + all-gather for power-of-two $p$ (Thakur, Rabenseifner & Gropp, *IJHPCA* 2005); two-tree broadcast achieving latency and bandwidth optimality simultaneously (Sanders, Speck & Träff, *Parallel Computing* 2009). No approximation guarantee exists for the general heterogeneous case. *Established.*

**Synthesis SOTA.** SCCL (Cai, Rashidi, et al., *PPoPP 2021*) encodes the schedule as an SMT instance and produces Pareto-optimal latency/bandwidth algorithms — but only up to about 8 GPUs; solve time explodes beyond that. TACCL (Shah et al., *NSDI 2023*) replaces exact synthesis with a MILP over a human-supplied "communication sketch", reporting up to $2\times$ speedup over NCCL on multi-node A100 clusters. MSCCLang (Cowan et al., *ASPLOS 2023*) is the DSL/compiler that makes such schedules executable. *Established that synthesized schedules beat NCCL on the reported configurations; not established that they are near-optimal* — none of these papers reports a lower bound on the same hardware, so the residual gap is unknown.

**Topology-aware systems SOTA.** Blink (Wang et al., *MLSys 2020*) packs spanning trees over the available link set, reporting up to $8\times$ over NCCL on irregular intra-node link sets and 40% end-to-end training speedup. BlueConnect (Cho et al., *MLSys 2019*) decomposes all-reduce into hierarchical reduce-scatter/all-gather stages matched to link tiers. TopoOpt (Wang et al., *NSDI 2023*) co-optimizes the *physical* topology with the schedule using optical circuit switches. Themis (Rashidi et al., *ISCA 2022*) schedules collectives across multi-dimensional fabrics.

**Claimed but unablated.** Vendor speedup numbers (NCCL vs MSCCL vs RCCL) are almost always single-cluster benchmark numbers with no held-out topology. NCCL's PAT (parallel aggregated trees) path for all-gather/reduce-scatter, added in the 2.2x series, is reported as reducing the step count to logarithmic; public independent ablation across fabrics is thin *(frontier — verify)*.

## 4. What Is Known

- Ring all-reduce moves $2\frac{p-1}{p}S$ bytes per link and is bandwidth-optimal on a homogeneous ring; latency scales $O(p)$, which is why NCCL switches to trees below roughly 1 MB messages. Measured at $p=8$–$256$ across many published benchmarks.
- Hardware ratio driving the problem: H100 SXM gives ~450 GB/s unidirectional NVLink per GPU; a 400 Gb/s NDR InfiniBand NIC gives ~50 GB/s, ~45 GB/s achieved. Intra:inter ratio ≈ 9–10:1 per GPU with 1 NIC/GPU, worse with fewer NICs. This ratio is the entire reason flat ring is wrong.
- SCCL: exact Pareto-optimal schedules for $p \le 8$ on DGX-1's irregular NVLink graph; up to $2.4\times$ lower latency than NCCL for small buffers at that scale.
- TACCL: reported up to $2\times$ all-reduce/all-to-all improvement and ~17% end-to-end transformer training gain on 2–8 node A100 clusters.
- Blink: up to $8\times$ collective speedup on DGX-1 configurations where NCCL's ring cannot use all links.
- Minimum broadcast time and minimum gossip time in arbitrary graphs are NP-complete (Garey & Johnson 1979).

## 5. What Is Not Known

- **Theoretically open.** No approximation algorithm with a proven ratio for makespan-minimal all-reduce on a capacitated heterogeneous graph with node-level port constraints. Even hardness *of approximation* for this specific formulation is unproven — the NP-completeness result covers broadcast, not all-reduce with chunking, where fractional splitting may restore tractability. Whether chunk-splittable all-reduce on two-level (intra/inter) hierarchies is in P is open.
- **Empirically open.** Nobody has published, at $p \ge 1024$, the measured gap between a synthesized schedule and the cut-based lower bound on the same machine. The experiment is runnable; the clusters exist; the number is unreported.
- **Methodologically blocked.** "Optimal" is not well defined while $\beta_e$ moves under congestion and thermal state. Two schedules within 10% of each other cannot currently be ranked reproducibly across runs, because run-to-run variance on shared fabrics is of the same order.

## 6. Why It Is Hard

**The obstruction is confounded measurement compounded by a search space that is exponential in $p$.**

1. *Cost-model parameters are not identifiable from endpoint benchmarks.* $\alpha_e,\beta_e$ measured in isolation do not predict behavior under concurrent load: switch buffering, ECMP hashing and PCIe contention are shared state the model has no variable for. A schedule optimal under the fitted model can lose to a "worse" one on hardware, and the post-hoc explanation is unfalsifiable.
2. *Synthesis does not scale.* SMT-exact synthesis stalls near $p=8$; MILP relaxations need a human sketch that already encodes most of the answer, so the reported win partly measures the sketch author.
3. *No lower bound is computed alongside the measurement.* Papers report speedup over NCCL, which bounds nothing. Without a machine-specific lower bound, "2× over NCCL" is compatible with being 3× off optimal.

## 7. Current Research (as of 2026)

- **Microsoft Research / MSCCL line** — MSCCL++, a GPU-initiated communication runtime giving finer-grained schedules than the NCCL collective API; continued TACCL-style MILP synthesis *(frontier — verify current status)*.
- **Georgia Tech (Krishna group) + Meta/Intel** — ASTRA-sim 2.0 (*ISPASS 2023*) as the simulation substrate for multi-dimensional network/collective co-design; Themis follow-ons.
- **MIT/Meta (TopoOpt line)** — optical-circuit-switch reconfiguration co-designed with the collective schedule.
- **Vendor** — NVIDIA NCCL algorithm additions (PAT trees, NVLS/SHARP in-network reduction), AMD RCCL topology detection. In-switch reduction changes the cost model qualitatively: it removes the $2\times$ factor of reduce-scatter + all-gather across the switch, and no synthesis tool yet models it as a first-class primitive *(frontier — verify)*.
- **Interconnect for MoE** — all-to-all with expert-parallel skew is the emerging hard case; loads are data-dependent, so the schedule must be chosen online.

## 8. Concrete Next Experiment

**Question:** how far from the machine lower bound is the best available schedule, at realistic scale?

- **Scale.** 64 nodes × 8 H100 = 512 GPUs, NDR 400 Gb/s rail-optimized fabric, 8 NICs/node. All-reduce at $S \in \{1\ \text{MB}, 64\ \text{MB}, 1\ \text{GB}\}$, fp32 and bf16.
- **Arms.** (a) NCCL default (control); (b) NCCL with tuner forced to ring / tree / NVLS separately; (c) a TACCL/MSCCL-synthesized schedule from a hierarchical sketch; (d) the **cut lower bound**, computed from *measured* $\beta_e$ under concurrent load — not nameplate — by solving the max-flow/min-cut LP over the rail topology.
- **Procedure.** Measure $\beta_e$ twice: idle, and with a background all-to-all on unrelated rails. Report both lower bounds. Run 200 iterations per arm, report median and interquartile range.
- **Deciding number.** $\rho = T_{\text{best schedule}} / T_{\text{lower bound, loaded}}$ at $S = 1$ GB. If $\rho \le 1.15$ with IQR under 5%, the method variant is effectively closed at this scale and effort should move to all-to-all and to the theory variant. If $\rho \ge 1.5$, there is a factor of 1.5 sitting unclaimed in every large training run, and synthesis at $p=512$ becomes the priority. If the IQR exceeds the gap between arms, the *measurement* variant is the blocker and no schedule comparison at this scale is meaningful — publish that result, it is the more important one.

## 9. Key References

- **[Foundational]** M. Garey, D. Johnson. *Computers and Intractability: A Guide to the Theory of NP-Completeness.* W. H. Freeman, 1979. (Minimum broadcast time, problem ND49.)
- **[Foundational]** R. Thakur, R. Rabenseifner, W. Gropp. *Optimization of Collective Communication Operations in MPICH.* International Journal of High Performance Computing Applications, 2005.
- **[Foundational]** P. Patarasuk, X. Yuan. *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations.* Journal of Parallel and Distributed Computing, 2009.
- **[Foundational]** P. Sanders, J. Speck, J. L. Träff. *Two-tree Algorithms for Full Bandwidth Broadcast, Reduction and Scan.* Parallel Computing, 2009.
- **[Foundational]** E. Chan, M. Heimlich, A. Purkayastha, R. van de Geijn. *Collective Communication: Theory, Practice, and Experience.* Concurrency and Computation: Practice and Experience, 2007.
- **[Foundational]** D. Culler et al. *LogP: Towards a Realistic Model of Parallel Computation.* PPoPP, 1993.
- **[SOTA]** Z. Cai, Z. Liu, S. Maleki, M. Musuvathi, T. Mytkowicz, J. Nelson, O. Saarikivi. *Synthesizing Optimal Collective Algorithms (SCCL).* PPoPP, 2021.
- **[SOTA]** A. Shah, V. Chidambaram, M. Cowan, S. Maleki, M. Musuvathi, T. Mytkowicz, J. Nelson, O. Saarikivi, R. Singh. *TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches.* NSDI, 2023.
- **[SOTA]** M. Cowan, S. Maleki, M. Musuvathi, O. Saarikivi, Y. Xiong. *MSCCLang: Microsoft Collective Communication Language.* ASPLOS, 2023.
- **[SOTA]** G. Wang, S. Venkataraman, A. Phanishayee, N. Devanur, J. Thelin, I. Stoica. *Blink: Fast and Generic Collectives for Distributed ML.* MLSys, 2020.
- **[SOTA]** M. Cho, U. Finkler, M. Serrano, D. Kung, H. Hunter. *BlueConnect: Decomposing All-Reduce for Deep Learning on Heterogeneous Network Hierarchy.* MLSys, 2019.
- **[SOTA]** W. Wang, M. Khazraee, Z. Zhong, M. Ghobadi, Z. Jia, D. Mudigere, Y. Zhang, A. Kewitsch. *TopoOpt: Co-optimizing Network Topology and Parallelization Strategy for Distributed Training Jobs.* NSDI, 2023.
- **[SOTA]** S. Rashidi, W. Won, S. Srinivasan, S. Sridharan, T. Krishna. *Themis: A Network Bandwidth-Aware Collective Scheduling Policy for Distributed Training of DL Models.* ISCA, 2022.
- **[Survey]** W. Won, T. Heo, S. Rashidi, S. Sridharan, S. Srinivasan, T. Krishna. *ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale.* ISPASS, 2023.

## 10. Worked Example

Two DGX-class nodes, 8 GPUs each ($p=16$), fp32 all-reduce of $S = 1$ GB. Intra-node NVLink: 450 GB/s per GPU unidirectional. Inter-node: one 400 Gb/s NIC per node, 45 GB/s achieved. Ignore latency ($S$ is large).

**Arm A — flat 16-GPU ring.** Every link carries $2\frac{p-1}{p}S = 2 \cdot \frac{15}{16} \cdot 1\ \text{GB} = 1.875$ GB. Two ring hops cross the NIC, but the ring is serialized, so makespan is set by the slowest link:

$$T_A = 1.875\ \text{GB} / 45\ \text{GB/s} = 41.7\ \text{ms}.$$

**Arm B — hierarchical (BlueConnect-style).** Intra-node reduce-scatter: $2\cdot\frac{7}{8}\cdot 1\ \text{GB}/450 = 3.9$ ms, leaving a 128 MB shard per GPU. Inter-node all-reduce across the 2-node cut: each node sends $8 \times 128\ \text{MB} = 1$ GB through its single NIC, once for reduce-scatter and once for all-gather at half size each — net 1 GB across the NIC: $1/45 = 22.2$ ms. Intra-node all-gather: 3.9 ms.

$$T_B = 3.9 + 22.2 + 3.9 = 30.0\ \text{ms}.$$

**Lower bound.** The cut has capacity 45 GB/s; each node must export at least its reduced 1 GB and import the complement, so $T^\star \ge 1\ \text{GB}/45\ \text{GB/s} = 22.2$ ms.

**Where the obstruction shows.** Arm B is $30.0/22.2 = 1.35\times$ off the bound. The whole 7.8 ms gap is the intra-node phases running *serially* with the NIC transfer; a pipelined schedule that chunks into 16 pieces and overlaps stages should recover nearly all of it. But when you run it, the NIC no longer sustains 45 GB/s — 8 GPUs now DMA concurrently through the same PCIe root complex during the intra-node phase, and measured $\beta$ drops to 38–43 GB/s depending on chunk size and thermal state. The predicted 23 ms lands at 26–29 ms, indistinguishable from Arm B at typical run-to-run variance. The cost model has no variable for the thing that decides the outcome, so "is the pipelined schedule optimal?" is not answerable on this machine as instrumented. That is the blocker: not the search, the ruler.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*