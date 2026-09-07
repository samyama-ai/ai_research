---
id: 15-mixture-of-experts/bandwidth-optimal-expert-placement
title: "Memory-Bandwidth-Optimal Expert Placement"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memory-Bandwidth-Optimal Expert Placement

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/bandwidth-optimal-expert-placement` · **Status:** open

## 1. Problem Statement

A sparse MoE model holds $E$ experts per layer but activates $k \ll E$ per token. Total parameters scale with $E$; FLOPs scale with $k$. The binding constraint at inference is therefore not arithmetic but **bytes moved**: expert weights travel from wherever they are stored (HBM, host DRAM, NVMe, a peer GPU) to wherever the token is being processed.

**Input.** A trained MoE checkpoint (layer count $L$, experts per layer $E$, top-$k$ routing), a device topology (per-device memory capacity, per-link bandwidth and latency), and a request distribution over token sequences.

**Output.** A placement — an assignment of each expert to a device or memory tier, possibly with replication — plus a prefetch/eviction policy.

**Objective.** Minimize expected per-token decode latency, or equivalently the makespan of the bytes that must cross the slowest link per decode step, subject to per-device capacity.

Three variants, with different difficulty:

- **Measurement.** Given a placement and a workload, what fraction of decode latency is attributable to expert movement rather than attention, all-to-all, or kernel launch overhead? Not cleanly separable on current stacks.
- **Method.** Find a good placement heuristically. Many published systems; none compared against a common optimum.
- **Theory.** Is the optimal placement computable, and how far from optimal are the heuristics? The static version is a capacitated quadratic assignment problem; no approximation guarantee for the MoE case is known.

A solution would be: an algorithm producing placements within a stated factor of a computable lower bound, on a workload whose routing statistics are published.

## 2. Formal Setting

Experts $\mathcal{E} = \{(\ell, e)\}$, $|\mathcal{E}| = LE$. Devices/tiers $\mathcal{D}$, device $d$ with capacity $M_d$ bytes. Expert $(\ell,e)$ has size $s_{\ell e}$ bytes — **measured** as the serialized tensor bytes at the deployment dtype, including quantization scales, not the nominal parameter count.

Placement $\pi: \mathcal{E} \to 2^{\mathcal{D}}$ (a set, to allow replication), feasible iff
$$\sum_{(\ell,e):\, d \in \pi(\ell,e)} s_{\ell e} \le M_d \quad \forall d \in \mathcal{D}.$$

**Routing statistics.** For a workload $\mathcal{W}$, let $p_{\ell e} = \Pr[\text{token routes to expert } e \text{ at layer } \ell]$, measured by instrumenting the gate's top-$k$ indices over a held-out sample of $\ge 10^6$ tokens. Let $C_{(\ell e),(\ell' e')}$ be the **co-activation** probability within one sequence — the quantity that decides whether two experts should be co-located, and the one most systems never measure.

**Cost.** Link $d \to d'$ has bandwidth $B_{dd'}$ (measured achieved, not nominal — PCIe 4.0 x16 delivers ~24–26 GB/s of a 32 GB/s peak) and latency $\lambda_{dd'}$. Per decode step with batch $b$, the bytes device $d$ must pull for layer $\ell$ are
$$V_d^{(\ell)} = \sum_{e} s_{\ell e} \cdot \mathbf{1}[\text{expert } (\ell,e) \text{ used by some token in the batch}] \cdot \mathbf{1}[d \notin \pi(\ell,e)\text{-resident cache}],$$
and step latency lower-bounds as
$$T \;\ge\; \sum_{\ell=1}^{L} \max_{d \in \mathcal{D}} \left( \lambda_{d} + \frac{V_d^{(\ell)}}{B_d} \right),$$
the roofline form (Williams, Waterman & Patterson, CACM 2009). The optimization is $\min_\pi \mathbb{E}_{\mathcal{W}}[T]$.

**Assumptions, and which are violated.**

1. *Routing is stationary.* Violated: $p_{\ell e}$ shifts with domain and with position in the sequence; MoE gates show measurable drift between prefill and decode.
2. *Co-activations are independent across layers.* Violated: expert choices correlate across adjacent layers, which is exactly what makes prefetch feasible and also what makes independent per-layer placement suboptimal.
3. *Transfer overlaps compute perfectly.* Violated at batch size 1, where there is not enough arithmetic to hide a 350 MB fetch.
4. *Expert sizes are uniform.* Holds for most released models (Mixtral, DeepSeek-V3), fails under heterogeneous quantization.

## 3. State of the Art

**Systems/empirical SOTA.**

- **Training-time placement.** GShard (Lepikhin et al., ICLR 2021) and Switch Transformer (Fedus, Zoph & Shazeer, JMLR 2022) place one expert per device and pay all-to-all. FasterMoE (He et al., PPoPP 2022) adds shadow-expert replication for hot experts; Tutel (Hwang et al., MLSys 2023) adds adaptive parallelism switching; Lina (Li et al., USENIX ATC 2023) prioritizes all-to-all over allreduce and dynamically rebalances. All report end-to-end speedups (Tutel: up to $2.75\times$ single-layer over Fairseq MoE at 2048 A100s) — *established as benchmark numbers on their own hardware, not as distance-from-optimal.*
- **Inference offloading.** Mixtral-offloading (Eliseev & Mazur, 2023) uses LRU caching plus speculative expert prefetch. Pre-gated MoE (Hwang et al., ISCA 2024) modifies the architecture so layer $\ell{+}1$'s experts are known at layer $\ell$, making prefetch exact. Fiddler (Kamahori et al., 2024) runs cold experts on the CPU rather than moving weights. MoE-Infinity (Xue et al., 2024) exploits sequence-level expert-activation locality for cache admission.
- **Deployment-scale.** DeepSeek-V3 (2024) uses node-limited routing (each token's experts confined to $\le 4$ nodes) and redundant hot-expert replication during inference. This is the largest-scale *deliberate* placement decision publicly documented.

**Theory SOTA.** Essentially absent. The static problem reduces to capacitated graph partitioning with a quadratic objective — NP-hard by reduction from minimum bisection (Garey, Johnson & Stockmeyer, 1976). No MoE-specific approximation ratio, and no published lower bound that any of the above systems measure themselves against.

**Claimed but unablated.** Prefetch-hit-rate gains are almost always reported jointly with a quantization or kernel change; the placement contribution alone is rarely isolated.

## 4. What Is Known

- **Bytes dominate at low batch.** Mixtral 8x7B: 46.7B total, 12.9B active parameters (Jiang et al., 2024). At bf16, one expert's three matrices ($4096 \times 14336$) are 352 MB; all 8 experts × 32 layers are 90.2 GB — more than an 80 GB A100.
- **Routing is not uniform but not degenerate.** Mixtral's own analysis reports no strong domain-topic specialization on The Pile, but does report **positional/temporal locality**: consecutive tokens repeat the same expert at rates well above the $1/8$ chance baseline (reported at layers 15 and 31).
- **Hot experts exist at scale.** Switch Transformer and GShard both required an auxiliary load-balancing loss precisely because unbalanced routing otherwise emerges; DeepSeek-V3 documents replicating high-load experts in production, implying measured imbalance.
- **Architectural prefetch works.** Pre-gated MoE (ISCA 2024) reports order-of-magnitude memory-footprint reduction versus keeping all experts resident, with accuracy roughly preserved — established, but requires retraining/fine-tuning the gate, so it does not apply to existing checkpoints.
- **Expert-choice routing changes the statistics.** Zhou et al. (NeurIPS 2022) show capacity-balanced routing by construction, which removes the hot-expert problem and replaces it with a variable-tokens-per-expert problem.

## 5. What Is Not Known

- **Theoretically open.** No approximation guarantee for capacitated expert placement under a co-activation objective. Also open: whether the co-activation matrix $C$ of trained MoEs has structure (low rank, block-diagonal, bounded treewidth) that admits a PTAS.
- **Empirically open.** Nobody has published, for a single frontier-scale MoE, (a) the measured co-activation matrix, (b) an ILP/branch-and-bound optimum or a valid lower bound on a reduced instance, and (c) the gap to the deployed heuristic. All three are runnable today.
- **Methodologically blocked.** Attributing decode latency to expert movement. On fused stacks, the transfer overlaps attention and the all-to-all; the standard measurement — ablate the transfer by pinning everything in HBM — changes batch size, kernel selection, and page behaviour simultaneously. There is no accepted counterfactual for "the same run without the byte movement."

## 6. Why It Is Hard

The obstruction is **confounded measurement plus an absent optimum**.

- Every reported placement improvement is measured end-to-end on a stack where the placement change also shifts kernel selection, cache residency, and effective batch. The number that gets reported ("$1.9\times$ decode speedup") is not the placement's contribution.
- There is no computable reference point. Without a lower bound on $\min_\pi \mathbb{E}[T]$, "better than LRU" is the only available claim, and LRU is a weak baseline in a domain with known cross-layer correlation.
- The routing statistics that define the objective are workload-dependent and unpublished for the models people actually deploy. A placement tuned on The Pile is not evaluated on code or multilingual traffic, where $p_{\ell e}$ differs.

## 7. Current Research (as of 2026)

- **Locality-aware expert caching** for consumer and single-GPU inference — the MoE-Infinity / Fiddler / Mixtral-offloading line. Active, incremental, dominated by cache-policy engineering.
- **Architectural co-design** so placement becomes trivially schedulable: pre-gating, shared-expert designs (DeepSeek), and node-limited routing. Strongest practical direction; requires touching training.
- **Prediction of next-layer routing** from hidden states, to convert a demand fetch into a prefetch. *(frontier — verify)* Reported hit rates of 70–90% appear in several 2024–2025 preprints; independent replication at $>50$B scale is thin.
- **Formalization as QAP/partitioning with published lower bounds.** Largely absent from the ML literature; the HPC task-mapping community has the tooling but not the MoE traces.

## 8. Concrete Next Experiment

**Scale.** Mixtral 8x7B (46.7B, $E{=}8$, $k{=}2$, $L{=}32$) on one A100 80GB with host DRAM over PCIe 4.0 — a regime where the model does not fit and placement is forced to matter. $10^6$ decode tokens sampled from three workloads (web text, code, multilingual).

**Procedure.**
1. Instrument the gate; record the full expert-activation trace. Publish it.
2. From the trace build the co-activation matrix $C$ over all 256 experts.
3. Solve the static placement ILP — which experts are HBM-resident — to optimality or to a certified gap, using a commercial MIP solver on the offline trace. This is the **clairvoyant lower bound**: it sees the future and is not achievable online.
4. Run three online arms on the same trace and the same kernels: **(control) LRU cache**, **(arm A) frequency-static placement from $p_{\ell e}$**, **(arm B) co-activation-aware placement + one-layer-lookahead prefetch**.

**Deciding number.** The **bytes-over-PCIe per decoded token**, normalized by the clairvoyant ILP optimum. Report $\rho = V_{\text{arm}} / V_{\text{ILP}}$ for each arm. If $\rho_{\text{LRU}} < 1.15$, placement optimization is a solved non-problem for this class and effort should move to architecture. If $\rho_{\text{LRU}} > 2$ while $\rho_B < 1.3$, co-activation structure is exploitable and the field has been leaving a factor on the table. Bytes, not seconds, is the metric — it is the one quantity not confounded by kernel and batch effects.

## 9. Key References

- **[Foundational]** Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23(120), 2022. — arXiv:2101.03961
- **[SOTA]** Hwang, Cui, Xiong, Yang, Liu, Xu, Wang, Wang, Zhou, Zhu, Chen, Yang. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys, 2023. — arXiv:2206.03382
- **[SOTA]** He, Zhai, Antunes, Wang, Luo, Shi, Li. *FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models.* PPoPP, 2022.
- **[SOTA]** Li, Jiang, Zhu, Che, Shi, Chen, Ma, Wang, Xu, Zhang, Zhu. *Accelerating Distributed MoE Training and Inference with Lina.* USENIX ATC, 2023.
- **[SOTA]** Hwang, Wei, Sanjabi, Kim, Yun, et al. *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.* ISCA, 2024. — arXiv:2308.12066
- **[SOTA]** Gale, Narayanan, Young, Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA]** Rajbhandari, Li, Yao, Zhang, Aminabadi, Awan, Rasley, He. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[Empirical]** Jiang, Sablayrolles, Roux, et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Empirical]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Empirical]** Eliseev, Mazur. *Fast Inference of Mixture-of-Experts Language Models with Offloading.* 2023. — arXiv:2312.17238
- **[Empirical]** Kamahori, Gu, Zhu, Kasikci. *Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.* 2024. — arXiv:2402.07033
- **[Method]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[Theory]** Garey, Johnson, Stockmeyer. *Some Simplified NP-Complete Graph Problems.* Theoretical Computer Science 1(3), 1976.
- **[Model]** Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4), 2009.
- **[Survey]** Cai, Jiang, Wang, Tang, Kim, Huang. *A Survey on Mixture of Experts.* 2024. — arXiv:2407.06204

## 10. Worked Example

**Mixtral 8x7B, batch 1, A100 80GB + host DRAM.**

Expert size: $3 \times 4096 \times 14336 \times 2\ \text{B} = 352.3$ MB. Per layer: $8 \times 352.3\ \text{MB} = 2.82$ GB. All layers: $32 \times 2.82 = 90.2$ GB. Non-expert weights: $\approx 1.6$B params $= 3.2$ GB.

Usable HBM after KV cache and activations: $\approx 74$ GB, of which $\approx 71$ GB for experts — **79% of the expert set fits**. 21% (about 54 of 256 experts) live in DRAM.

Per decode token, top-2 of 8 per layer: $64$ expert activations. Under a placement that is *oblivious* to routing, expected misses $= 64 \times 0.21 = 13.4$, i.e. $13.4 \times 352.3\ \text{MB} = 4.72$ GB fetched per token. At an achieved 25 GB/s over PCIe: **189 ms/token**, or 5.3 tok/s. Compute alone (12.9B active params at bf16, 25.8 GB read from HBM at 1.6 TB/s achieved) is **16 ms**. Transfer is $12\times$ compute.

Now the obstruction. Suppose the 54 DRAM-resident experts are chosen as the 54 *least frequent* by $p_{\ell e}$. If routing were uniform, misses stay at 13.4. If the top 79% of experts by frequency carry 90% of activations, misses drop to $64 \times 0.10 = 6.4$, giving 90 ms/token — a $2.1\times$ win from one line of code. **Which of these two worlds we are in is not published for Mixtral.** The activation-frequency distribution over $10^6$ tokens has never been released, so the $2.1\times$ is neither claimed nor refuted.

Worse, the co-activation term is invisible to frequency ranking. If two experts at layers $\ell$ and $\ell{+}1$ co-fire with probability 0.6, evicting them as a pair costs 704 MB in one 28 ms window with no overlap opportunity, whereas evicting two anti-correlated experts of identical frequency spreads the cost across steps. Frequency-optimal and bandwidth-optimal placements differ, and no published system measures $C$ to tell them apart. That is the open problem: the objective depends on a matrix nobody has measured, evaluated against an optimum nobody has computed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*