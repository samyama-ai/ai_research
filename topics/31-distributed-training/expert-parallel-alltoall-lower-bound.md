---
id: 31-distributed-training/expert-parallel-alltoall-lower-bound
title: "All-to-All Latency Lower Bounds for Expert Parallelism"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# All-to-All Latency Lower Bounds for Expert Parallelism

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/expert-parallel-alltoall-lower-bound` · **Status:** open

## 1. Problem Statement

Expert-parallel Mixture-of-Experts (MoE) training places $E$ experts across $p$ accelerators and moves every token to the devices holding its top-$k$ experts (dispatch), then moves the outputs back (combine). Both are irregular all-to-all exchanges whose message sizes are decided at runtime by the router.

The problem: **given the realized routing of a batch, what is the minimum achievable makespan of the dispatch/combine pair on a given interconnect, and how close do production kernels get?**

Three variants, different difficulty:

- **Theory.** Prove a lower bound on makespan for irregular all-to-all under a two-level (intra-node / inter-node) bandwidth-and-latency model with a fanout constraint (each token visits at most $M$ nodes), and exhibit a schedule matching it to a constant factor. Open.
- **Measurement.** Define the quantity being bounded so it is measurable *in situ* — i.e. while the collective overlaps with GEMMs and shares streaming multiprocessors with them. Currently ill-posed.
- **Method.** Build a scheduler that attains the bound on real routing traces. Open, and blocked on the first two.

Solving it means: an instance-conditioned lower bound $L^*(R)$ computable from a routing trace $R$, plus a measured ratio $L^{\text{kernel}}/L^*(R)$ with a proof that no schedule beats $L^*$ by more than a constant.

## 2. Formal Setting

**Topology.** $p$ GPUs on $N$ nodes, $g = p/N$ per node. Per-GPU inter-node injection bandwidth $B_{\text{out}}$ (bytes/s, measured as achieved RDMA throughput on a saturating 2-node pairwise test, not the link nominal). Per-GPU intra-node bandwidth $B_{\text{in}}$. Per-message startup $\alpha$ (measured as half the 0-byte round-trip time).

**Workload.** Per-GPU token batch $T$, hidden width $d$, element size $b$ bytes, top-$k$ routing over $E$ experts, $e = E/p$ experts per GPU. Routing is the binary matrix $R \in \{0,1\}^{pT \times E}$ with row sums $k$.

**Traffic matrix.** With per-destination deduplication (a token crossing to node $n$ is sent once and fanned out on NVLink), the node-level demand in bytes is
$$C_{mn} \;=\; b\,d\,\bigl|\{\,t \in \mathcal{B}_m \;:\; \exists\, j \in \text{experts}(n),\ R_{tj}=1\,\}\bigr|.$$

**Lower bounds currently in use.** Node egress/ingress:
$$L_{\text{bw}} \;=\; \max\Bigl(\max_m \tfrac{\sum_{n\neq m} C_{mn}}{g B_{\text{out}}},\ \max_n \tfrac{\sum_{m\neq n} C_{mn}}{g B_{\text{out}}}\Bigr),$$
latency: $L_{\text{lat}} = \alpha\lceil \log_2 p\rceil$ in the single-port store-and-forward model. The working bound is $L^{\dagger} = \max(L_{\text{bw}}, L_{\text{lat}})$; the true $L^*$ is unknown.

**Decision predicate.** Is $\sup_R \; L^{\text{ALG}}(R)/L^*(R)$ bounded by a constant independent of $p$, $N$ and the skew of $R$?

**Assumptions, and which are violated.**

| Assumption | Status in practice |
|---|---|
| Balanced routing, $C_{mn}$ near-uniform | **Violated.** Expert load is heavy-tailed and drifts across training even with auxiliary balance losses. |
| Single-port, one link active per GPU | **Violated.** NVLink and IB are driven concurrently; DeepSeek-V3's kernels overlap them by design. |
| Linear $\alpha + m\beta$ cost, no congestion | **Violated.** Incast at receivers and rail-optimized fat-tree path asymmetry make effective $\beta$ demand-dependent. |
| Communication is free of compute resources | **Violated.** DeepSeek-V3 reserves 20 of 132 SMs for comm kernels; the collective and the GEMM contend. |
| Message sizes known before the collective | **Violated.** Sizes are router outputs; a metadata exchange precedes dispatch. |

## 3. State of the Art

**Theory SOTA (established).** For *regular* all-to-all with equal $m$-byte messages, the classical results stand: direct exchange costs $(p-1)(\alpha + m\beta)$; Bruck's index algorithm costs $\lceil\log_2 p\rceil\,\alpha + \tfrac{p}{2}\log_2 p\cdot m\beta$ (Bruck et al., IEEE TPDS 1997), and the $\log_2 p$ round count is optimal in the single-port model. The bandwidth term $(p-1)m\beta$ is optimal by a counting argument. **No matching latency–bandwidth tradeoff lower bound is known for the irregular, fanout-limited case**, which is exactly the MoE case.

**Systems SOTA (established by ablation).** Tutel (Hwang et al., MLSys 2023) shows adaptive parallelism switching plus 2D hierarchical all-to-all; FasterMoE (He et al., PPoPP 2022) shows shadowing of hot experts and pipelined dispatch; MegaBlocks (Gale et al., MLSys 2023) removes the capacity-factor token drop via block-sparse kernels. Each ablates its own contribution.

**Claimed but unablated / benchmark-only.** DeepEP's reported per-GPU throughput (roughly 150+ GB/s NVLink intranode, low-40s GB/s RDMA internode on H800) exists as repository benchmark numbers on a single hardware configuration, not as a peer-reviewed ablation, and is not accompanied by an instance-conditioned optimality claim. COMET (ByteDance, 2025) reports fine-grained compute/comm overlap speedups; the reported end-to-end gains are not decomposed into "better schedule" versus "better overlap with compute". No published system reports $L^{\text{kernel}}/L^{\dagger}$ over a distribution of *real* routing traces.

## 4. What Is Known

- **Bruck bound is tight for regular exchange.** $\lceil\log_2 p\rceil$ rounds is a proven lower bound in the single-port model; the algorithm attains it (TPDS 1997). Verified in MPICH at $p \le 1024$ (Thakur, Rabenseifner, Gropp, IJHPCA 2005).
- **Fanout limiting works.** DeepSeek-V3 (2048 H800 GPUs, $d = 7168$, 256 routed experts, $k=8$) caps each token at $M = 4$ nodes, chosen from the measured NVLink:IB bandwidth ratio of $160{:}50 = 3.2$. This makes inter-node volume $O(M)$ rather than $O(k)$ and is reported to make dispatch cost near-independent of $k$ at fixed $M$.
- **Communication is a first-order cost.** DeepSpeed-MoE (Rajbhandari et al., ICML 2022) and GShard (Lepikhin et al., ICLR 2021) both report all-to-all as the dominant non-GEMM term at $p \ge 128$; published MoE-layer breakdowns commonly put dispatch+combine at 20–40% of layer time before overlap.
- **Skew is real and large.** With auxiliary-loss balancing, per-expert load in trained MoEs still routinely shows max/mean ratios in the 1.5–3× range at the batch level; capacity factors of 1.25–2.0 are standard precisely to absorb it (GShard, Switch Transformer, JMLR 2022).
- **Bisection is not the binding constraint at current scale.** On rail-optimized 8-GPU-per-node clusters with per-GPU 400 Gb/s NICs, the per-GPU injection term dominates the fabric bisection term for $N \lesssim 512$; measured egress bound exceeds measured bisection bound in that range.

## 5. What Is Not Known

- **Theoretically open.** No lower bound for irregular, fanout-limited all-to-all on a two-level topology with concurrent multi-port links. Specifically: is there an instance where every schedule exceeds $\max(L_{\text{bw}}, L_{\text{lat}})$ by $\omega(1)$? Nobody has exhibited such an instance or proved none exists. The classical latency/bandwidth tradeoff lower bound does not transfer, because deduplication makes the demand matrix a function of the schedule's grouping choices.
- **Empirically open.** The ratio $L^{\text{kernel}}/L^{\dagger}$ on *recorded* routing traces at $p \ge 512$ has not been published. The experiment is runnable today on any 64-node cluster; it needs a trace dump and an offline optimal-schedule solver, not new hardware.
- **Methodologically blocked.** "All-to-all latency" is not well defined in situ. Isolated kernel timing gives one number; the same kernel co-resident with GEMMs on shared SMs gives another; the marginal contribution to step time gives a third. These differ by more than the gains being claimed, and papers do not state which they report.

## 6. Why It Is Hard

**Confounded measurement, compounded by an instance-dependent baseline.** Two distinct confounds stack:

1. *Resource contention.* The collective and the expert GEMMs share SMs, L2 and memory bandwidth. Measuring the collective alone changes the thing measured. There is no accepted decomposition of step time into "comm" and "compute" when they are deliberately fused.
2. *The denominator moves.* $L^{\dagger}$ depends on the realized $C_{mn}$, which depends on the router, which depends on the checkpoint. A 16% headroom against a uniform-traffic bound can be 1% headroom against the instance's own bound. Because papers report the uniform bound (or no bound), reported "efficiency" numbers are not comparable across systems, models, or even training steps.

Add non-identifiability: a slower kernel that produces less SM contention can yield a faster step. Optimizing the named quantity is not optimizing the thing that matters.

## 7. Current Research (as of 2026)

- **Fanout-limited routing as a systems primitive** — device- and node-limited routing from DeepSeek-V2/V3; being adopted as a default in open MoE recipes. Established.
- **Compute–communication fusion** — COMET (ByteDance), DeepEP (DeepSeek), and fused dispatch-GEMM kernels. The direction is real; per-kernel optimality claims are benchmark-only. *(frontier — verify)*
- **Topology-aware expert placement** — reassigning experts to devices to reduce $\max_m \sum_n C_{mn}$ given observed routing statistics; explored by SmartMoE (USENIX ATC 2023) and Lina (USENIX ATC 2023). Whether periodic re-placement beats static placement at $p>1000$ under drifting routers is unsettled. *(frontier — verify)*
- **Communication lower-bound theory carried over from linear algebra** — the Ballard–Demmel–Holtz–Schwartz program (SIMAX 2011) and red-blue pebbling give tight bounds for dense kernels; extending them to data-dependent sparse routing is, to our knowledge, unattempted. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** 64 nodes × 8 H100 (512 GPUs), 400 Gb/s per-GPU IB, NVLink intra-node. A 30–60B-parameter MoE, $d = 6144$, $E = 128$, $k = 8$, $T = 4096$ tokens/GPU. Dump 1000 real routing traces $R$ from steps spread across a full training run (early, mid, late — routing skew changes).

**Arms.**
- *Treatment:* the production kernel (DeepEP or Tutel hierarchical all-to-all), timed two ways: isolated, and as marginal step-time contribution (step time with the collective minus step time with the collective replaced by a same-shape no-op memcpy).
- *Control:* an offline near-optimal schedule for the same node-level demand matrix $C_{mn}$, computed by Birkhoff–von Neumann / bipartite edge-coloring decomposition into permutation rounds. In a non-blocking multi-port model this attains a makespan equal to the maximum line sum, so it is a *constructive* upper bound sitting on top of $L_{\text{bw}}$.
- *Second control:* uniform-traffic $L^{\dagger}$, i.e. the bound everyone currently reports.

**The deciding number.** The median over 1000 traces of
$$\rho \;=\; L^{\text{kernel}}_{\text{isolated}} \big/ L^{\text{BvN}}(C).$$
If $\rho \le 1.15$, production kernels are already near-optimal for the realized traffic and further effort belongs in routing and placement, not scheduling. If $\rho \ge 1.5$, scheduling is the gap and a solver-in-the-loop dispatcher is warranted. The spread between $\rho$ and the uniform-bound ratio also quantifies, for the first time, how much of the reported headroom in the literature is an artifact of the wrong denominator.

## 9. Key References

- **[Foundational]** J. Bruck, C.-T. Ho, S. Kipnis, E. Upfal, D. Weathersby. *Efficient Algorithms for All-to-All Communications in Multiport Message-Passing Systems.* IEEE Transactions on Parallel and Distributed Systems, 1997.
- **[Foundational]** D. Culler, R. Karp, D. Patterson, A. Sahay, K. Schauser, E. Santos, R. Subramonian, T. von Eicken. *LogP: Towards a Realistic Model of Parallel Computation.* PPoPP, 1993.
- **[Foundational]** R. Thakur, R. Rabenseifner, W. Gropp. *Optimization of Collective Communication Operations in MPICH.* International Journal of High Performance Computing Applications, 2005.
- **[Foundational]** G. Ballard, J. Demmel, O. Holtz, O. Schwartz. *Minimizing Communication in Numerical Linear Algebra.* SIAM J. Matrix Anal. Appl., 2011.
- **[Foundational]** N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, J. Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, Z. Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[SOTA]** C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram, J. Chau, P. Cheng, F. Yang, M. Yang, Y. Xiong. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys, 2023. — arXiv:2206.03382
- **[SOTA]** J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, Q. Li. *FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models.* PPoPP, 2022.
- **[SOTA]** T. Gale, D. Narayanan, C. Young, M. Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[SOTA]** S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, Y. He. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[Survey]** W. Cai, J. Jiang, F. Wang, J. Tang, S. Kim, J. Huang. *A Survey on Mixture of Experts.* 2024. — arXiv:2407.06204
- **[Survey]** W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961

## 10. Worked Example

**Instance.** 2048 GPUs, 256 nodes × 8. $d = 7168$, $k = 8$ routed experts, node-limited fanout $M = 4$, $T = 4096$ tokens/GPU, fp8 dispatch ($b = 1$). Per-GPU nominal IB injection $B_{\text{out}} = 50$ GB/s.

**Dispatch volume per GPU (deduplicated per destination node).**
$$V = T \cdot M \cdot d \cdot b = 4096 \times 4 \times 7168 \times 1 = 1.174 \times 10^{8}\ \text{B} = 117.4\ \text{MB}.$$

**Uniform-traffic bound.** $L_{\text{bw}} = 117.4\,\text{MB} / 50\,\text{GB/s} = 2.35$ ms. Latency term $\alpha\lceil\log_2 2048\rceil \approx 2\,\mu\text{s} \times 11 = 22\ \mu$s — negligible. So $L^\dagger = 2.35$ ms.

**Measured.** At a reported achieved internode rate of 43 GB/s, the kernel takes $117.4 / 43 = 2.73$ ms. Ratio against the uniform bound: $2.73/2.35 = 1.16$ — "16% headroom left".

**Now condition on the instance.** Suppose the realized routing gives the busiest node an egress 15% above the mean, $\max_m \sum_n C_{mn} = 1.15\,\bar{V}$. The instance bound is $L^*(R) \ge 2.35 \times 1.15 = 2.70$ ms. The same measurement now shows $2.73/2.70 = 1.01$ — **1% headroom**.

**The obstruction, visible.** One measurement, two conclusions differing by a factor of 16 in the size of the remaining opportunity, and the difference is entirely in a denominator nobody reports. A team optimizing against the uniform bound will spend engineer-months chasing 380 $\mu$s that the routing skew already consumed. Worse, the two bounds cannot be distinguished after the fact: the traces needed to compute $\max_m \sum_n C_{mn}$ are discarded at the end of every step. Until instance-conditioned bounds are computed and reported alongside kernel times, "we reached 87% of the achievable all-to-all bandwidth" is not a falsifiable claim.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*