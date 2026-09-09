---
id: 11-inference-and-serving/expert-offloading-latency
title: "Expert Offloading Latency Under Sparse Activation"
topic: 11-inference-and-serving
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expert Offloading Latency Under Sparse Activation

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/expert-offloading-latency` · **Status:** empirically-open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) model activates a small fraction of its parameters per token — DeepSeek-V3 activates 37B of 671B — but the *unactivated* parameters must still be reachable. When total weights exceed accelerator memory, experts live in host DRAM or NVMe and are fetched over PCIe on demand. The question: **how much of the arithmetic saving from sparse activation survives once the parameter movement it implies is paid for?**

Three variants, which are usually conflated:

- **Measurement.** Define a decode-latency metric for offloaded MoE that is not gameable by cache warm-up, prompt domain, or batch size. What is the right reporting unit — tokens/s at fixed batch, or bytes-moved-per-token at fixed hit rate?
- **Method.** Given a memory budget $M$ on device, choose which experts to resident-cache and which to prefetch, so as to minimize p50/p99 inter-token latency. Solving this means beating an LRU baseline by a stated margin *at a batch size that matters for serving*, not only at batch 1.
- **Theory.** Expert fetching is online caching with predictions. What is the achievable competitive ratio against Belady-optimal offline, given a router whose next-layer decision is itself a function of the current layer's output?

Solved would mean: a policy with a proven robustness guarantee and a measured $\geq 2\times$ p99 decode improvement over LRU on a public routing trace at batch $\geq 8$, on hardware where the model does not fit.

## 2. Formal Setting

Model with $L$ MoE layers, $E$ routed experts per layer, top-$k$ routing. Let $x_t^{(\ell)}$ be the hidden state of token $t$ at layer $\ell$, and the router select

$$S_t^{(\ell)} = \operatorname{top-}k\left( W_g^{(\ell)} x_t^{(\ell)} \right) \subseteq \{1,\dots,E\}, \qquad |S_t^{(\ell)}| = k.$$

**Quantities as measured.**

- $b$ — bytes per expert. Measured, not derived: `sizeof(dtype) × param_count`, including quantization scales and any padding. Mixtral-8x7B fp16: $b = 3 \cdot 4096 \cdot 14336 \cdot 2 \approx 352$ MB.
- $\beta$ — *effective* host-to-device bandwidth, measured by a pinned-memory copy benchmark **at transfer size $b$**, concurrently with the decode kernel stream. Not the vendor's peak number: PCIe 5.0 x16 peaks at 63 GB/s and delivers 40–50 GB/s pinned, less under KV-cache contention.
- $h$ — hit rate, $h = 1 - \mathbb{E}_t\!\left[\frac{1}{Lk}\sum_\ell |S_t^{(\ell)} \setminus C^{(\ell)}|\right]$, where $C^{(\ell)}$ is the resident set. Measured over a token stream from a *held-out domain mixture*, after discarding the first 512 tokens (warm-up).
- $T_\ell^{\text{comp}}$ — layer compute time, measured with all experts resident.
- $T_\ell^{\text{fetch}} = |S_t^{(\ell)} \setminus C^{(\ell)}| \cdot b / \beta$.

Per-token decode latency under perfect prefetch overlap:

$$T_{\text{tok}} = \sum_{\ell=1}^{L} \max\left(T_\ell^{\text{comp}},\, T_\ell^{\text{fetch}}\right),$$

and without overlap, the $\max$ becomes a sum. The **arithmetic intensity of an expert fetch** is the diagnostic quantity: at batch $B$, one fetched expert does $O(B)$ FLOPs of work per byte moved, so the regime is bandwidth-bound until $B$ is large.

At batch $B$, the fetch set is the union $\bigcup_{t=1}^{B} S_t^{(\ell)}$. Under the (false) assumption of independent uniform routing,

$$\mathbb{E}\left|\bigcup_{t=1}^{B} S_t^{(\ell)}\right| = E\left(1 - \left(1 - \tfrac{k}{E}\right)^{B}\right).$$

**Assumptions known to be violated.** (i) *Uniform, independent routing* — real routers are imbalanced and correlated across layers and adjacent tokens; the union above is therefore a pessimistic bound, by an unmeasured margin. (ii) *Stationary routing* — expert popularity shifts with domain and language, so a hit rate fit on WikiText does not transfer. (iii) *Perfect compute/fetch overlap* — prefetch depth is bounded by the router's dependence on the previous layer's output. (iv) *Uniform expert size* — shared/dense experts (DeepSeek-V3, Qwen-MoE) break it.

## 3. State of the Art

**Systems/empirical SOTA.**

- *Fast Inference of MoE Language Models with Offloading* (Eliseev & Mazur, 2023, arXiv:2312.17238) — LRU expert cache plus speculative prefetch from the *previous* layer's hidden state; 2–3 tokens/s for Mixtral-8x7B on a T4/RTX 3060 with mixed quantization. **Established**: the LRU + speculative-prefetch combination beats naive on-demand fetch. **Benchmark number only**: the tokens/s figure is batch-1, single-GPU, and not decomposed into hit rate versus quantization gain.
- *Fiddler* (Kamahori et al., ICLR 2025, arXiv:2402.07033) — runs expert FFNs *on the CPU* rather than moving weights, when the activation is smaller than the weight. Established, and the crossover condition is analytic. Batch-1 regime.
- *MoE-Infinity* (Xue et al., arXiv:2401.14361) — request-level expert activation tracing to drive prefetch and caching; claims large latency reductions over DeepSpeed-Inference/Mixtral-Offloading. The trace-locality claim is **claimed but not independently ablated** against a strong LRU-with-lookahead control.
- *Pre-gated MoE* (Hwang et al., ISCA 2024, arXiv:2308.12066) — changes the architecture so layer $\ell$'s gate is computed at layer $\ell-1$, making prefetch exact. Established as a co-design; costs a fine-tuning step and a small quality delta.
- *MoE-Lightning* (Cao et al., ASPLOS 2025, arXiv:2411.11217) — CPU-GPU pipelining with a hierarchical roofline model; targets throughput, not p99 latency.
- Production baselines: DeepSpeed-MoE (Rajbhandari et al., ICML 2022) and FlexGen (Sheng et al., ICML 2023) define the offloading-without-MoE-awareness control.

**Theory SOTA.** Expert caching is weighted paging. Deterministic online paging is $k$-competitive and no better (Sleator & Tarjan, 1985); randomized achieves $\Theta(\log k)$. With an imperfect predictor, Lykouris & Vassilvitskii (ICML 2018) give a caching algorithm whose competitive ratio degrades gracefully with prediction error, improved by Rohatgi (SODA 2020). **No published work instantiates these bounds with the MoE router as the predictor**, which is where the problem sits.

## 4. What Is Known

- **Sparse activation does not reduce bytes moved.** Mixtral-8x7B, top-2 of 8, 32 layers: a cold token needs $2 \times 32 = 64$ expert fetches = 22.5 GB at fp16. At a measured 25 GB/s that is 0.9 s/token — 1.1 tokens/s. Reproduced in spirit by every offloading paper.
- **Routing has real, exploitable temporal locality.** Eliseev & Mazur report that experts active for token $t$ are more likely than chance to be active for $t+1$; this is what makes LRU work at all. Measured on Mixtral-8x7B, batch 1.
- **Router imbalance is persistent.** Switch Transformer (Fedus et al., JMLR 2022) and GShard (Lepikhin et al., ICLR 2021) both needed explicit load-balancing losses because unregularized routers collapse onto few experts — an imbalance that *helps* caching and *hurts* expert-parallel throughput.
- **Quantization is the largest single lever.** Dropping 352 MB/expert to 88 MB at 4 bits cuts the cold-token bound to 5.6 GB and 4.4 tokens/s — a $4\times$ gain no scheduling policy has matched.
- **Batching destroys sparsity.** At $E=8$, $k=2$, $B=8$, expected distinct experts per layer is $8(1-0.75^8) = 7.2$ of 8. Confirmed as a design constraint by expert-parallel serving systems, which assume near-dense expert use at serving batch sizes.

## 5. What Is Not Known

- **Empirically open.** No published measurement of hit rate as a function of cache budget, batch size, and *domain shift*, on a modern large-$E$ model (DeepSeek-V3: $E=256$, $k=8$, 58 MoE layers). Every result above is Mixtral-class, $E=8$. The runnable experiment: instrument a router, dump the trace, sweep. Nobody has published the sweep.
- **Empirically open.** The batch-size crossover $B^\ast$ at which offloading loses to expert-parallel sharding is not measured for any model. It is the number that decides whether offloading is a serving technique or a hobbyist technique.
- **Methodologically blocked.** There is no standard offloaded-MoE benchmark. Reported tokens/s figures vary with prompt domain, warm-up length, and whether the cache is flushed between requests — none of which are reported. Two papers' numbers are not comparable.
- **Theoretically open.** No bound relating router prediction error $\epsilon$ (measured as top-$k$ set overlap between the speculative gate and the true gate) to competitive ratio for the layer-chained case, where a prefetch error at layer $\ell$ corrupts the prediction input at $\ell+1$. Standard caching-with-advice results assume predictions are exogenous; here they are not.

## 6. Why It Is Hard

**The obstruction is confounded measurement, and it is specific.** A reported "3.2× speedup from expert prefetching" mixes at least four effects that no published ablation separates: (1) quantization of the offloaded weights, (2) resident-cache capacity, (3) prefetch accuracy, (4) whether the evaluation prompt's domain matches the trace the policy was tuned on. Effects (1) and (2) are large and policy-independent; (3) is the actual contribution. Because papers report end-to-end tokens/s rather than hit rate at fixed budget and fixed dtype, the contribution of the policy is not identifiable from the published numbers.

The second obstruction is a **dependency, not a cost**: the router at layer $\ell$ consumes the output of layer $\ell-1$, so exact prefetch lookahead is one layer, and $T_\ell^{\text{comp}}$ for one layer at batch 1 is ~1 ms against a ~14 ms fetch for one fp16 Mixtral expert. Overlap cannot hide the transfer. Pre-gated MoE escapes by changing the architecture — which concedes that the scheduling problem, as posed, has no solution at batch 1.

## 7. Current Research (as of 2026)

- **Trace-driven prefetch.** MoE-Infinity (Edinburgh/Xue et al.) and AdapMoE (arXiv:2408.10284) build activation histories per request and adapt the number of experts fetched to per-token sensitivity.
- **Compute-instead-of-move.** Fiddler's CPU-expert execution generalizes to heterogeneous placement; the crossover is bandwidth-vs-CPU-FLOPs and moves with each hardware generation. *(frontier — verify)* Several groups are extending this to NPU-equipped clients.
- **Architectural co-design.** Shared-expert designs (DeepSeek-V2/V3) pin a dense expert on device and route only the residual, structurally raising the hit rate. Whether this was motivated by offloading or by training stability is not stated in the technical reports.
- **Caching with learned advice, applied.** *(frontier — verify)* Bringing Lykouris–Vassilvitskii-style consistency/robustness guarantees to expert caching is an obvious and, as far as we can establish, unclaimed target.

## 8. Concrete Next Experiment

**The hit-rate/batch-size sweep on a large-$E$ model.**

- **Scale.** DeepSeek-V3-class (671B total, $E=256$, $k=8$, 58 MoE layers) in fp8, one H100 80GB, 1 TB host DRAM, PCIe 5.0. Router traces dumped for 200k decode tokens over five domains (code, English web, Chinese web, math, multi-turn chat).
- **Arms.** (a) LRU cache, budget $M \in \{8, 16, 32, 64\}$ GB. (b) LRU + one-layer speculative prefetch. (c) Belady-optimal offline oracle on the recorded trace — the ceiling. (d) **Control arm:** no offloading, tensor-parallel across enough GPUs to fit, same dtype, same batch.
- **Sweep.** Batch $B \in \{1, 2, 4, 8, 16, 32\}$. Domain matched vs. shifted between the tuning trace and the eval trace.
- **The deciding number.** $B^\ast$ — the largest batch size at which arm (b) at $M = 32$ GB achieves p99 inter-token latency within $2\times$ of arm (d) per-GPU-normalized. If $B^\ast \geq 8$, offloading is a serving technique and the scheduling problem is worth solving. If $B^\ast \leq 2$, it is a single-user technique and the field should stop reporting throughput.
- **Secondary, and cheap:** the gap between arm (b) and arm (c) at each $M$. It upper-bounds everything any future policy can win, and it costs one replay over an already-recorded trace.

## 9. Key References

- **[Foundational]** Sleator, D. & Tarjan, R. *Amortized Efficiency of List Update and Paging Rules.* Communications of the ACM, 1985.
- **[Foundational]** Belady, L. A. *A Study of Replacement Algorithms for a Virtual-Storage Computer.* IBM Systems Journal, 1966.
- **[Foundational]** Fedus, W., Zoph, B. & Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[Foundational]** Lepikhin, D. et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Theory]** Lykouris, T. & Vassilvitskii, S. *Competitive Caching with Machine Learned Advice.* ICML, 2018. — arXiv:1802.05399
- **[Theory]** Rohatgi, D. *Near-Optimal Bounds for Online Caching with Machine Learned Advice.* SODA, 2020.
- **[SOTA]** Eliseev, A. & Mazur, D. *Fast Inference of Mixture-of-Experts Language Models with Offloading.* Preprint, 2023. — arXiv:2312.17238
- **[SOTA]** Kamahori, K., Gu, Y., Zhu, K. & Kasikci, B. *Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.* ICLR, 2025. — arXiv:2402.07033
- **[SOTA]** Hwang, R. et al. *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.* ISCA, 2024. — arXiv:2308.12066
- **[SOTA]** Xue, L. et al. *MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache.* Preprint, 2024. — arXiv:2401.14361
- **[SOTA]** Cao, S. et al. *MoE-Lightning: High-Throughput MoE Inference on Memory-Constrained GPUs.* ASPLOS, 2025. — arXiv:2411.11217
- **[Systems]** Rajbhandari, S. et al. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[Systems]** Sheng, Y. et al. *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU.* ICML, 2023. — arXiv:2303.06865
- **[Model]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Model]** Jiang, A. Q. et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Survey]** Cai, W. et al. *A Survey on Mixture of Experts in Large Language Models.* 2024. — arXiv:2407.06204

## 10. Worked Example

DeepSeek-V3, fp8, one H100 80GB, PCIe 5.0 measured at $\beta = 40$ GB/s pinned under load.

Expert size: intermediate 2048, hidden 7168, three matrices → $3 \cdot 7168 \cdot 2048 \approx 44$M params → $b = 44$ MB at fp8. Routed activations per token: $k \cdot L = 8 \cdot 58 = 464$ experts → **20.4 GB per token if nothing is cached**.

$$T_{\text{tok}}^{\text{cold}} = \frac{20.4\ \text{GB}}{40\ \text{GB/s}} = 0.51\ \text{s} \;\Rightarrow\; 2.0\ \text{tokens/s}.$$

Now put the 80 GB device to work. Attention, embeddings, and shared experts take ~25 GB, leaving $M = 55$ GB ≈ 1250 cached experts of 14,848 routed experts — **8.4% resident**. Under uniform routing the hit rate would be 0.084 and

$$T_{\text{tok}} = \frac{464 \cdot (1-0.084) \cdot 44\ \text{MB}}{40\ \text{GB/s}} = 0.47\ \text{s},$$

a 9% improvement. To get to 20 tokens/s you need $h = 0.90$ — a hit rate $10.7\times$ above the uniform baseline, achieved from a cache holding 8.4% of the experts. **The entire feasibility of offloading rests on routing skew of that magnitude, and nobody has published the measurement of whether it exists at $E=256$.**

Then batch it. At $B = 8$, if routing were independent-uniform, distinct experts per layer would be $256(1 - (1-8/256)^8) = 256 \cdot 0.225 = 57.5$ — versus 8 for a single token. Bytes per *batch step* rise $7.2\times$ while tokens produced rise $8\times$: a marginal 11% per-token win, far below the $8\times$ a dense-resident model gets from the same batching. Correlated routing makes the true union smaller than 57.5, which is exactly the unmeasured quantity.

The obstruction is now visible in one line: **the cold-token bound (0.51 s) and the target (0.05 s) differ by a factor that only the hit rate can supply, and the hit rate at serving batch sizes has never been measured.** A published "3× speedup" that does not report $h$, $M$, dtype, and $B$ tells you nothing about which of those four it came from.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*