---
id: 11-inference-and-serving/moe-serving-load-imbalance
title: "Mixture-of-Experts Routing Load Imbalance at Serving Time"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Mixture-of-Experts Routing Load Imbalance at Serving Time

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/moe-serving-load-imbalance` · **Status:** open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) layer routes each token to $k$ of $E$ experts. When experts are sharded across devices (expert parallelism, EP), the layer's latency is set by the **busiest** device, not the average one. Training fixes this with an auxiliary balancing loss over large batches. Serving cannot: the decode batch is small, the token mix is whatever arrived, and no gradient is available to reshape the router.

- **Input:** a trained MoE model with frozen router weights; a live request stream; a device topology with $D$ devices holding $E$ experts.
- **Output:** an expert-to-device placement and a token dispatch schedule.
- **Objective:** minimise p99 time-per-output-token (TPOT) subject to exact preservation of the model's output distribution (no re-routing that changes which experts a token sees), or with a stated, bounded quality delta.
- **Solved** would mean: a serving policy that keeps the max-to-mean device load ratio near its stochastic floor across shifting workloads, with a proof or reproducible measurement that the residual gap is irreducible.

Three variants, different difficulty:
- **Measurement:** how much of observed MoE-layer latency is routing skew versus multinomial noise, all-to-all communication, or kernel inefficiency? Currently confounded.
- **Method:** placement/replication/scheduling policies. Empirically active, weakly ablated.
- **Theory:** is there a placement achieving $O(1)$ competitive max load against an adversarial, drifting token stream with bounded expert-replication budget? Open.

## 2. Formal Setting

Let a MoE layer have experts $\{1,\dots,E\}$, top-$k$ routing, and a device map $\pi: \{1,\dots,E\} \to 2^{\{1,\dots,D\}}$ (a set, allowing replication). For a decode step with $B$ tokens, let $A_{te}\in\{0,1\}$ indicate token $t$ is routed to expert $e$, so $\sum_e A_{te}=k$.

**Expert load** (measured by instrumenting the dispatch buffer, per step, not averaged):
$$n_e = \sum_{t=1}^{B} A_{te}, \qquad \sum_e n_e = kB.$$

**Device load** under a dispatch $\sigma$ splitting each expert's tokens over its replicas:
$$L_d = \sum_{e:\, d\in\pi(e)} \sigma_{ed}\, n_e .$$

**Imbalance ratio**, the quantity that actually multiplies latency:
$$\rho = \frac{\max_d L_d}{\frac{1}{D}\sum_d L_d}.$$
Measured per step and reported as a distribution (p50/p99 over steps), never as a mean — the mean of $\rho$ understates tail latency.

**Stochastic floor.** Under i.i.d. uniform routing, $n_e\sim\text{Multinomial}(kB, 1/E)$, so even a perfect router gives $\rho_0 = \mathbb{E}[\max_d L_d]\,D/(kB) > 1$. The scientifically meaningful quantity is the **excess imbalance** $\rho/\rho_0$, not $\rho$.

**Layer time model** (each term separately measurable with CUDA events):
$$T_{\text{layer}} = \underbrace{T_{a2a}(\max_d L_d)}_{\text{dispatch}} + \underbrace{c\cdot \max_d L_d}_{\text{GEMM}} + T_{a2a}^{\text{comb}} + T_{\text{sync}},$$
where $c$ is per-token expert FLOP time at the achieved arithmetic intensity. $c$ is **not** constant: at low $n_e$ the GEMM is memory-bound (weight loading dominates), so halving load does not halve time. This is the single most common measurement error in the literature.

**Capacity factor** $C$: each device processes at most $\lceil C\,kB/E\rceil$ tokens; overflow tokens are dropped or deferred. Drop rate $\delta = \frac{1}{kB}\sum_e \max(0, n_e - \text{cap})$.

Assumptions and their status:
- *Tokens route independently* — **violated**. Routing is strongly correlated within a sequence and across layers; consecutive decode tokens from one request hit similar experts.
- *Expert popularity is stationary* — **violated**. Popularity shifts with language, domain, and prompt template.
- *Per-token expert cost is uniform* — holds for homogeneous experts, violated for shared-expert or heterogeneous-width designs.
- *Replication is free* — violated; each replica costs HBM that would otherwise hold KV cache.

## 3. State of the Art

**Established (ablated, reproduced):**
- **Capacity-factor dropping** (GShard, Lepikhin et al., ICLR 2021; Switch Transformer, Fedus et al., JMLR 2022) bounds $\max_d L_d$ by construction and is the baseline every system compares to. Its cost — token drops — is measured and real.
- **Grouped/blocked kernels** (MegaBlocks, Gale et al., MLSys 2023) remove the need for dropping by expressing MoE as block-sparse matmul; the speedups over dropping baselines were ablated against Tutel and dMoE variants.
- **Tutel** (Hwang et al., MLSys 2023) — adaptive parallelism switching per layer per step, with measured end-to-end gains on 2048-GPU runs.
- **Auxiliary-loss-free balancing** (Wang et al., 2024; deployed in DeepSeek-V3, arXiv:2412.19437) adds a per-expert bias to router logits, updated by observed load. Ablated against auxiliary-loss baselines during training.

**Claimed but weakly ablated:**
- **Redundant-expert replication + dynamic rebalancing** (DeepSeek's EPLB, released 2025) replicates hot experts across the EP group. The reported serving throughput is a *system benchmark number* on one model and one cluster, not an ablation isolating the contribution of replication from that of the surrounding kernel and scheduling work.
- **Offload/prefetch systems** — MoE-Infinity, Fiddler, Pre-gated MoE (Hwang et al., ISCA 2024) — target the memory-constrained single-GPU case. Their speedups are benchmark numbers under specific batch sizes; sensitivity to batch size and workload mix is largely unreported.
- **Expert Choice routing** (Zhou et al., NeurIPS 2022) makes load balance exact by construction (each expert picks its top tokens) but breaks causal autoregressive decoding without modification, so it is a training-time, not serving-time, answer.

**Theory SOTA** is generic: balls-in-bins and "power of two choices" (Azar et al., SICOMP 1999; Mitzenmacher) bound max load for *uniform* placement. No bound is known for the adversarial, correlated, replication-budgeted MoE serving case.

## 4. What Is Known

- **Routing is skewed but not pathologically so at training scale.** Switch Transformer and ST-MoE (Zoph et al., 2022) report that with a balancing loss, expert loads stay within a small factor of uniform over large batches ($10^5$–$10^6$ tokens per step).
- **Skew is severe at decode batch sizes.** The gap is arithmetic: with $B=32$, $k=2$, $E=8$, only 64 assignments fall into 8 bins. Even under perfectly uniform routing, $\sigma=\sqrt{64\cdot\frac18\cdot\frac78}=2.65$ against a mean of 8, giving $\rho_0\approx 1.45$. Small-batch imbalance is dominated by counting noise, not router pathology.
- **Mixtral 8x7B** (Jiang et al., 2024) reported no strong topic-level expert specialisation, but did report **consecutive-token routing repetition well above chance** — the temporal correlation that breaks the independence assumption in §2.
- **DeepSeek-V3** uses 256 routed experts with top-8 and **node-limited routing** (each token's experts confined to at most 4 nodes) plus redundant experts specifically because unconstrained all-to-all imbalance was the deployment bottleneck (arXiv:2412.19437; ISCA 2025 system paper).
- **Balancing losses cost quality.** ST-MoE and DeepSeekMoE (Dai et al., ACL 2024) both report that strengthening the auxiliary loss coefficient degrades loss/downstream accuracy — the loss-free bias method exists precisely because this trade-off was measured.
- **All-to-all can exceed expert compute.** DeepSpeed-MoE (Rajbhandari et al., ICML 2022) and Lina (Li et al., USENIX ATC 2023) both report communication as the dominant MoE inference cost at serving batch sizes on multi-node clusters.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed decomposition of MoE-layer latency into (i) stochastic imbalance floor $\rho_0$, (ii) router-induced excess skew, (iii) all-to-all straggler effects, (iv) memory-bound GEMM inefficiency at low occupancy. Papers report end-to-end throughput; the terms are confounded. Until $\rho/\rho_0$ is reported, "our method improves load balance" is not falsifiable.
- **Empirically open.** No public study measures excess imbalance $\rho/\rho_0$ as a function of decode batch size, sequence position, and workload mix (code vs. multilingual vs. chat) on a frontier-scale MoE. The experiment is runnable today on any open MoE with hooks; nobody has published it at the right scale.
- **Theoretically open.** No competitive-ratio bound for online expert placement with replication budget $R$ against a drifting adversarial popularity distribution. It is unknown whether $O(\log E/\log\log E)$-style balls-in-bins guarantees survive expert-level correlation, or whether the correlated case admits a constant-factor bound with $R = O(E)$.
- Unknown whether routing skew at serving time is **request-identifiable** — i.e., predictable from the prompt before the first forward pass, which is what any admission-control or batch-composition policy would need.

## 6. Why It Is Hard

**Confounded measurement, plus a null model nobody subtracts.** The reported quantity ($\rho$, or throughput) mixes an irreducible combinatorial floor with the effect being claimed. At $B=32$ the floor is already $\approx1.45$; a paper reporting "$\rho$ reduced from 1.9 to 1.5" may have achieved nothing beyond what uniform random routing gives for free, and there is no way to tell from the published number.

Compounding it: **the latency-load relation is non-linear**. Below the roofline knee, expert GEMMs are weight-load-bound, so shaving the max load does not shave time proportionally. A perfectly balanced layer at $B=32$ can be *slower* per token than an imbalanced one at $B=256$, so throughput comparisons across batch sizes silently change the mechanism under test.

And **the ground truth is absent**: the optimal placement for a given trace requires solving an online assignment problem whose offline optimum is itself NP-hard (it is scheduling on identical machines with replication), so no system reports its gap to optimum — only to the previous system.

## 7. Current Research (as of 2026)

- **Replication-and-rebalance at inference.** DeepSeek's EPLB and its derivatives in vLLM and SGLang expert-parallel backends; redundant hot experts, periodic replacement from observed counters. *(frontier — verify current upstream state.)*
- **Disaggregated prefill/decode with different EP degrees**, on the premise that prefill (large $B$, near the balanced regime) and decode (small $B$, noise-dominated) want different placements. Pursued in industrial serving stacks. *(frontier — verify.)*
- **Communication-aware routing constraints** — node-limited routing as in DeepSeek-V3, and hierarchical all-to-all in Tutel/DeepSpeed lineages.
- **Predictive prefetch** from cross-layer routing correlation (Pre-gated MoE, ISCA 2024; MoE-Infinity line), targeting the offload rather than the multi-GPU case.
- **Router post-hoc editing** — bias terms or temperature adjusted at serving time to flatten load. Quality impact is the open question; ablations are thin.

## 8. Concrete Next Experiment

**Question:** how much serving imbalance is excess skew rather than counting noise, and does replication remove the excess?

**Scale:** one open frontier-scale MoE (e.g. a 100B+-parameter, 128+-expert model) on 8×H100 with EP=8, plus one small model (Mixtral 8x7B, $E=8$, $k=2$) for cheap replication. Trace: 50k real requests spanning chat, code, and non-English, replayed at fixed arrival rate. Decode batch sizes $B\in\{8,32,128,512\}$.

**Instrumentation:** per-layer, per-step $n_e$ from the dispatch buffer; CUDA-event timings split into dispatch / GEMM / combine.

**Control arm (the part usually missing):** for every measured step, resample $kB$ assignments i.i.d. uniform over experts and compute $\rho_0$ empirically from the same $(B,k,E)$. Second control: a *load-oracle* run where tokens are re-routed to force $\rho=1$, giving the achievable latency floor including the memory-bound-GEMM effect.

**Deciding number:** the p99 of **excess imbalance $\rho/\rho_0$ at $B=32$**, before and after redundant-expert replication with a 25% HBM budget.
- If p99 $\rho/\rho_0 < 1.15$ without replication, serving imbalance is a batch-size problem, not a routing problem, and the entire replication literature is optimising noise.
- If p99 $\rho/\rho_0 > 1.5$ and replication cuts it below 1.2 while p99 TPOT drops by a matching factor, replication is confirmed and the mechanism is identified.
- If $\rho/\rho_0$ falls but TPOT does not, the bottleneck is all-to-all or GEMM occupancy, and load balancing is the wrong lever.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[SOTA]** Gale, Narayanan, Young, Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA]** Hwang, Cui, Xiong, Yang, Liu, Zhu, et al. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys, 2023. — arXiv:2206.03382
- **[SOTA]** Rajbhandari, Li, Yao, Zhang, Aminabadi, Awan, Rasley, He. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[SOTA]** Wang, Chen, Xie, Zhao, et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[SOTA]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[SOTA]** Li, Jiang, Zhang, Chen, Xu. *Accelerating Distributed MoE Training and Inference with Lina.* USENIX ATC, 2023.
- **[SOTA]** Hwang, Wei, Kim, Kim, Mutlu, Rhu. *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.* ISCA, 2024. — arXiv:2308.12066
- **[Foundational, theory]** Azar, Broder, Karlin, Upfal. *Balanced Allocations.* SIAM Journal on Computing, 1999.
- **[Survey]** Cai, Jiang, Wang, Tang, Kim, Huang. *A Survey on Mixture of Experts in Large Language Models.* IEEE TKDE / arXiv, 2024. — arXiv:2407.06204

## 10. Worked Example

Mixtral 8x7B, $E=8$, $k=2$, EP=8 (one expert per GPU), decode batch $B=256$.

**Assignments:** $kB = 512$ tokens spread over 8 experts. Mean load $\bar n = 64$.

**Stochastic floor.** Under uniform routing, $n_e\sim\text{Bin}(512,\tfrac18)$, $\sigma=\sqrt{512\cdot\tfrac18\cdot\tfrac78}=7.48$. The expected maximum of 8 such (negatively correlated) counts sits near $\bar n + 1.4\sigma \approx 74.5$, so
$$\rho_0 \approx 74.5/64 = 1.16.$$

**Measured.** Suppose the instrumented p99 step gives $\max_e n_e = 118$:
$$\rho = 118/64 = 1.84, \qquad \rho/\rho_0 = 1.84/1.16 = 1.59.$$
So 16 percentage points of the 84% overhead is free noise; the genuine router skew is 59%.

**Now shrink the batch.** Same model, $B=32$: $kB=64$, $\bar n=8$, $\sigma=2.65$, $\rho_0\approx 1.45$. A measured $\rho=1.70$ is now $\rho/\rho_0 = 1.17$ — *less* excess skew than the $B=256$ case, despite the larger-looking headline number.

**The obstruction, visible.** A system that reports "we reduced imbalance from 1.84 to 1.45" has, at $B=32$, reduced it to the floor and cannot do better; at $B=256$ it has left 25% of the excess on the table. Identical headline numbers, opposite conclusions. Worse, moving from $B=32$ to $B=256$ raises per-expert GEMM occupancy 8-fold, so $c$ in §2 falls — TPOT improves for a reason that has nothing to do with balance. Without publishing $\rho_0$ and a load-oracle control, no reader can separate the three effects, and no two papers in this literature are comparable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*