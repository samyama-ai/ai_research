---
id: 15-mixture-of-experts/all-to-all-lower-bound-moe
title: "All-to-All Communication Lower Bound for MoE"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# All-to-All Communication Lower Bound for MoE

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/all-to-all-lower-bound-moe` · **Status:** open

## 1. Problem Statement

Expert-parallel MoE layers move every token to the device holding its selected expert and back. That is two personalized all-to-all exchanges per MoE layer per forward pass, four per training step counting the backward pass. The question: **what is the minimum communication any correct expert-parallel MoE implementation must perform, and is the gap between that bound and deployed systems constant-factor or asymptotic?**

Three variants, commonly conflated:

- **Measurement variant.** What fraction of MoE step time is *irreducible* communication — bytes that no overlap, fusion, or scheduling can remove — as opposed to exposed latency that better overlap would hide? Deployed profiles report "all-to-all is X% of step time" without separating the two.
- **Method variant.** Given a routing function fixed by the model, find the schedule (token movement, expert-weight movement, replication, hierarchical aggregation) minimizing max-per-device bytes. This is an optimization problem with a known feasible region and no known optimum.
- **Theory variant.** Prove a lower bound $\Omega(\cdot)$ on per-device words communicated for any distributed algorithm computing an MoE layer, in a model where routing decisions are *inputs* (fixed) or *free variables* (co-designed with placement). No such bound exists for either case.

A solution to the theory variant is a matching pair: a lower bound on words moved per device per MoE layer, and an algorithm attaining it to within a constant.

## 2. Formal Setting

Let $P$ devices hold $E$ experts, $E \geq P$, with $E/P$ experts per device. Batch $B$ tokens per device, model width $d$, top-$k$ routing, element size $s$ bytes ($s=2$ for bf16). Expert FFN has hidden width $d_{\text{ff}}$, so weights per expert are $w = 2 d\,d_{\text{ff}} s$ bytes (gate-up plus down, ignoring SwiGLU's third matrix).

Routing is a matrix $R \in \{0,1\}^{BP \times E}$ with $\sum_j R_{ij} = k$. Define the **dispatch volume** for device $p$:

$$V_p^{\text{send}} = s\,d \sum_{i \in \mathcal{B}_p} \sum_{j : \text{dev}(j) \neq p} R_{ij}$$

measured as bytes counted on the NIC/NVLink counters, not as tensor sizes in the program. The metric of interest is the makespan-relevant maximum:

$$V^\star = \min_{\text{schedule}} \; \max_p \left( V_p^{\text{send}} + V_p^{\text{recv}} \right).$$

Under **uniform routing** ($R$'s columns balanced, expert choice independent of device) the naive bound is
$$\mathbb{E}[V_p^{\text{send}}] = s\,d\,B\,k\,(1 - 1/P),$$
doubled for the combine step, doubled again for backward. For DeepSeek-V3-shaped numbers ($d=7168$, $k=8$, $s=2$, $B=4096$ tokens/device, $P=64$) that is $\approx 462$ MB per device per layer per direction.

Time uses the $\alpha$–$\beta$ model: a message of $n$ bytes costs $\alpha + n\beta$, with $\alpha$ the per-message latency and $\beta$ the inverse bandwidth, both measured by an isolated NCCL/RCCL benchmark on the same fabric, not from vendor peak. On a two-tier fabric write $\beta_{\text{intra}}$ (NVLink) and $\beta_{\text{inter}}$ (IB/RoCE), typically $\beta_{\text{inter}} / \beta_{\text{intra}} \approx 4$–$10$.

The competing primitive is **expert movement**: broadcasting $w$ bytes of weights instead of routing tokens. Token movement wins when $s d B k \ll w$, i.e. when the local token count per expert is below the arithmetic break-even $B k / E \ll d_{\text{ff}}$.

**Assumptions known to be violated in practice:**

- *Uniform routing.* Real routers are heavy-tailed; max-to-mean expert load ratios of 1.5–4× are routine even with auxiliary load-balancing loss, and load shifts across training.
- *Static placement.* Optimal placement depends on $R$, which depends on the trained router; co-adaptation makes the offline optimum unattainable online.
- *Congestion-free $\beta$.* All-to-all saturates bisection bandwidth; effective $\beta$ under full-mesh traffic is worse than pairwise-benchmark $\beta$, often by 1.3–2×.
- *Independent layers.* Overlap across layers/microbatches couples the schedules, so per-layer bounds do not compose additively.

## 3. State of the Art

**Theory SOTA.** There is no MoE-specific lower bound. The nearest established results are: (i) latency/bandwidth trade-offs for personalized all-to-all in the multi-port message-passing model — Bruck et al. (IEEE TPDS 1997) give $\Theta(\log P)$-step algorithms with $\Theta(n P \log P)$ bandwidth versus direct exchange at $P-1$ messages and $n(P-1)$ bandwidth, with matching lower bounds *within that model*; (ii) I/O lower bounds for dense matmul via the red–blue pebble game (Hong & Kung, STOC 1981) and its distributed-memory form (Irony, Toledo, Tiskin, JPDC 2004; Ballard, Demmel, Holtz, Schwartz, SIMAX 2011), giving $\Omega(n^3/(P\sqrt{M}))$ words. Neither covers the MoE case, where the communication pattern is *data-dependent* and the algorithm may choose to move weights instead of data. **Established**: the all-to-all primitive bounds. **Not established**: any bound on the MoE layer as a whole.

**Systems SOTA.** DeepSpeed-MoE (Rajbhandari et al., ICML 2022) introduced hierarchical all-to-all, cutting inter-node messages by aggregating within a node; reports up to 7.3× better inference latency/cost than dense equivalents — a benchmark number, not an ablation isolating the communication term. Tutel (Hwang et al., MLSys 2023) adds adaptive parallelism switching and reports 4.96×/5.75× speedup for 1.1T-parameter MoE on 2048 A100s. FasterMoE (He et al., PPoPP 2022) adds a performance model plus "shadowing" (replicating hot experts) and dynamic shadowing. Lina (Li et al., USENIX ATC 2023) prioritizes all-to-all over overlapping all-reduce and reports large tail-latency reductions. MegaBlocks (Gale et al., MLSys 2023) removes token dropping via block-sparse kernels — a compute, not a communication, fix. DeepSeek-V3 (2024) uses node-limited routing (each token's experts confined to $\leq 4$ nodes) plus DualPipe overlap, and reports near-zero exposed all-to-all cost at 2048 H800s. Fine-grained overlap kernels that fuse the all-to-all into the expert GEMM (the COMET line, ByteDance, 2025) claim further gains *(frontier — verify)*.

**Claimed but unablated:** almost every "we eliminated all-to-all overhead" claim measures *exposed* time under a specific overlap schedule, not *bytes moved*. Bytes are rarely reported.

## 4. What Is Known

- Naive per-layer dispatch volume scales as $\Theta(B k d)$ per device, independent of $E$ and nearly independent of $P$ (factor $1-1/P$). Confirmed by counter-level profiling in DeepSpeed-MoE and Tutel at 128–2048 GPUs.
- Hierarchical aggregation reduces *inter-node* bytes by up to the intra-node group size $g$ (typically 8) when multiple local tokens target the same remote node; the realized reduction is data-dependent and is far below $g$ under near-uniform routing.
- Node-limited routing is a genuine byte reduction, not a scheduling trick: restricting each token to $M$ nodes bounds inter-node fan-out at $M$ rather than $\min(k, \text{nodes})$. DeepSeek-V3 uses $M=4$, $k=8$, 256 routed experts, 37B active of 671B total, trained on 2048 H800s.
- Reported all-to-all share of MoE training step time spans roughly 20–60% across published profiles, depending on fabric, expert-parallel degree, and overlap quality (DeepSpeed-MoE, FasterMoE, Lina, at 64–512 GPUs). The spread itself is the finding: the fraction is a property of the deployment, not of MoE.
- Load imbalance dominates the tail. FasterMoE and Lina both show the slowest device sets the step time; expert-load skew is the mechanism.
- Granularity trades against communication: finer experts improve loss at fixed FLOPs (Krajewski et al., *Scaling Laws for Fine-Grained Mixture of Experts*, 2024) but increase $k$ and therefore dispatch volume linearly.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $V^\star$ for MoE under any realistic machine model. Specifically unproven: (a) that $\Omega(Bkd)$ bytes per device is necessary when the algorithm may replicate or migrate experts; (b) whether hierarchical/node-limited schemes are constant-factor optimal or asymptotically improvable; (c) whether a communication–memory trade-off of the Hong–Kung form ($\text{words} \geq f(\text{replication factor})$) exists for data-dependent routing.
- **Empirically open.** Nobody has published the byte-level ratio $V^{\text{measured}}/V^{\text{oracle}}$, where the oracle is an offline optimal placement computed from the recorded routing matrix, at $\geq 512$ GPUs. The measurement is runnable today: log $R$, solve the placement ILP or an LP relaxation offline, compare.
- **Methodologically blocked.** "Communication cost of MoE" is not a well-defined quantity as currently reported: bytes moved, exposed time, and step-time share are three different numbers, and papers use whichever is favorable. There is no agreed protocol separating irreducible bytes from unhidden latency.

## 6. Why It Is Hard

The obstruction is **non-identifiability between the algorithm and the routing distribution**. A lower bound must quantify over algorithms for a *fixed* problem, but in MoE the routing matrix $R$ is a learned object that co-adapts with placement: change the placement policy, retrain, and $R$ changes. A bound proved for worst-case $R$ is vacuous (adversarial routing forces the naive volume); a bound proved for uniform $R$ is inapplicable (real routers are skewed and can be regularized toward locality). Choosing the right restricted class of $R$ — "routings a trained model would actually produce, at no loss penalty" — is exactly the missing formal object.

Secondary: the measurement is confounded. Step-time share mixes bytes, fabric congestion, kernel launch overhead, and overlap quality; two systems with identical bytes can report a 3× difference in "all-to-all cost."

## 7. Current Research (as of 2026)

- **Locality-aware routing as a regularizer** — penalizing cross-node dispatch in the router loss, generalizing DeepSeek's node-limited routing to a soft constraint. Question: what is the loss cost per byte saved? *(frontier — verify)*
- **Fused overlap kernels** — decomposing the MoE layer so communication and expert GEMM interleave at tile granularity (COMET, ByteDance; FlashDMoE-style single-kernel designs, 2025). These reduce exposed time, not bytes. *(frontier — verify)*
- **Expert migration vs token movement** — dynamic shadowing (FasterMoE) and data-centric variants; the crossover condition $sdBk/E$ vs $w$ is understood arithmetically but not exploited adaptively at scale.
- **Inference-side disaggregation** — prefill/decode split with expert-parallel decode at very large $P$ (DeepSeek's inference stack); decode has tiny $B$, so latency $\alpha$ dominates over bandwidth $\beta$ and the bound question changes character entirely.

## 8. Concrete Next Experiment

**Question:** is deployed MoE all-to-all within a constant factor of the offline optimum, or asymptotically off?

**Scale.** One 20B-active / 200B-total MoE ($d=6144$, $E=128$, $k=8$, 32 layers) trained for 5B tokens on 256 H100s, expert-parallel degree 64, 8 GPUs/node.

**Instrumentation.** Log the full routing matrix $R$ for 200 sampled steps. Record NIC and NVLink byte counters per device per MoE layer.

**Arms.**
1. *Control:* standard expert parallelism, static round-robin expert placement, NCCL all-to-all.
2. Hierarchical all-to-all (DeepSpeed-MoE style), same $R$.
3. *Oracle:* offline, given logged $R$, solve min-max per-device inter-node bytes over expert-to-device assignments (LP relaxation + rounding is sufficient; the gap is reportable). No retraining — this is a bound on the *placement* degree of freedom only.

**Deciding number.** $\rho = V^{\text{control}}_{\text{inter-node}} / V^{\text{oracle}}_{\text{inter-node}}$, max over devices, averaged over the 200 steps. If $\rho < 1.3$, placement is exhausted and the remaining gap must come from routing co-design or expert movement — the theory question narrows to those. If $\rho > 2$, a large purely-schedulable gap exists and the empirical problem is open before the theory one is. Secondary readout: does $\rho$ grow with $P$ (run at $P=16,64,256$)? Growth is evidence the gap is asymptotic, not constant.

## 9. Key References

- **[Foundational]** Jia-Wei Hong, H. T. Kung. *I/O Complexity: The Red-Blue Pebble Game.* STOC, 1981.
- **[Foundational]** Dror Irony, Sivan Toledo, Alexander Tiskin. *Communication lower bounds for distributed-memory matrix multiplication.* Journal of Parallel and Distributed Computing, 2004.
- **[Foundational]** Grey Ballard, James Demmel, Olga Holtz, Oded Schwartz. *Minimizing Communication in Numerical Linear Algebra.* SIAM J. Matrix Anal. Appl., 2011.
- **[Foundational]** Jehoshua Bruck, Ching-Tien Ho, Shlomo Kipnis, Eli Upfal, Derrick Weathersby. *Efficient Algorithms for All-to-All Communications in Multiport Message-Passing Systems.* IEEE Trans. Parallel and Distributed Systems, 1997.
- **[Foundational]** Dmitry Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[SOTA]** Samyam Rajbhandari et al. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[SOTA]** Jiaao He et al. *FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models.* PPoPP, 2022.
- **[SOTA]** Chang Hwang et al. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys, 2023. — arXiv:2206.03382
- **[SOTA]** Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA]** Jiamin Li et al. *Accelerating Distributed MoE Training and Inference with Lina.* USENIX ATC, 2023.
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Survey]** Jakub Krajewski et al. *Scaling Laws for Fine-Grained Mixture of Experts.* 2024. — arXiv:2402.07871
- **[Survey]** Ernie Chan, Marcel Heimlich, Avi Purkayastha, Robert van de Geijn. *Collective communication: theory, practice, and experience.* Concurrency and Computation: Practice and Experience, 2007.

## 10. Worked Example

Take $d = 7168$, $k = 8$, $s = 2$ bytes, $B = 4096$ tokens/device, $P = 64$ devices across 8 nodes of 8 GPUs.

Dispatch bytes per device per layer, one direction:
$$V^{\text{send}} = 2 \times 7168 \times 4096 \times 8 \times (1 - 1/64) \approx 4.62 \times 10^{8}\ \text{B} = 462\ \text{MB}.$$

Fraction crossing the node boundary under uniform routing: $1 - 8/64 = 0.875$, so 404 MB inter-node. At an achieved inter-node bandwidth of 50 GB/s per GPU, that is 8.1 ms — per layer, per direction. Four such exchanges per layer per training step (dispatch/combine, forward/backward) gives 32 ms/layer. Over 32 MoE layers: **1.04 s of communication per step**.

Now the expert-movement alternative. With $d_{\text{ff}} = 2048$ per fine-grained expert, one expert's weights are $w = 2 \times 7168 \times 2048 \times 2 \approx 58.7$ MB. Moving all 128 experts costs 7.5 GB — 16× the token volume. Token movement wins decisively here, as expected when $Bk/E = 4096 \times 8/128 = 256 \gg$ nothing; the arithmetic-intensity crossover is at $Bk/E \approx d_{\text{ff}}=2048$, i.e. an 8× larger batch per device.

Apply node-limited routing with $M=4$: inter-node fan-out per token drops from up to 7 destination nodes to 4, but the *bytes* only drop if tokens coalesce. With hierarchical aggregation the token is sent once per destination node, so bytes scale as (distinct nodes hit) $\times sd$ rather than $k \times sd$. Under uniform routing over 8 nodes with $k=8$, expected distinct nodes hit is $8(1 - (7/8)^8) \approx 5.25$; capping at 4 gives a 1.31× byte reduction. Measured reductions in this family are typically larger than 1.31× because real routing is correlated.

**Where the obstruction shows.** The 1.31× is computed under the uniform-$R$ assumption. The real reduction depends on how correlated $R$ is — and that correlation is itself a function of whether the router was trained under the node-limit constraint. So the same intervention has three different "savings" depending on which $R$ you assume, and none of the three is a lower bound on anything: an adversarial $R$ forces the full 462 MB, and a locality-trained $R$ could in principle push inter-node bytes toward zero at some unknown loss cost. Until that loss-versus-bytes frontier is characterized, there is no fixed problem instance to prove a bound about.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*