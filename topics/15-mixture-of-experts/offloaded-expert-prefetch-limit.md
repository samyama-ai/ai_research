---
id: 15-mixture-of-experts/offloaded-expert-prefetch-limit
title: "Offloaded-Expert Prefetch Accuracy Limit"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Offloaded-Expert Prefetch Accuracy Limit

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/offloaded-expert-prefetch-limit` · **Status:** open

## 1. Problem Statement

When a sparse MoE model does not fit in accelerator memory, expert weights live in host DRAM or NVMe and are copied to the GPU on demand. A **prefetcher** guesses which experts layer $\ell$ will route to, $d$ layers before the router at $\ell$ actually runs, and starts the copy early. The question is how accurate such a guess can be.

- **Measurement variant.** Define prefetch accuracy so that it predicts end-to-end serving latency. Reported "cache hit rate" does not: it mixes the trivially-cached hot experts with the ones that actually stall the pipeline, and it is reported at batch size 1 on perplexity corpora.
- **Method variant.** Build a predictor $\hat{S}_\ell$ of the routed expert set from state available at layer $\ell-d$, under a compute budget small enough that running it does not itself cost more than the stall it avoids.
- **Theory variant.** Give a nontrivial upper bound on achievable accuracy at depth $d$ under a predictor-cost budget $c$. Note the naive information-theoretic bound is vacuous: routing is a deterministic function of the hidden state, so an unbounded predictor achieves 100%. Any real limit is **resource-bounded**, and no framework currently produces one.

Solving it means: a bound $A^*(d,c)$ plus a predictor that meets it within a stated gap, on traffic with realistic batch size.

## 2. Formal Setting

Model with $L$ MoE layers, $E$ experts per layer, top-$k$ routing. Expert $(\ell,e)$ has weight tensor of $P$ parameters at $b$ bytes each; transfer size $W = Pb$ bytes.

Router at layer $\ell$ for token $t$ with hidden state $h_\ell(t)\in\mathbb{R}^{D}$:
$$S_\ell(t) = \operatorname*{top-}k\big(W_\ell^{g}\,h_\ell(t)\big) \subseteq [E].$$

GPU holds a cache $C_\ell \subseteq [E]$, $|C_\ell| = m$. A prefetcher $\pi$ emits $\hat{S}_\ell(t) \subseteq [E]$, $|\hat S_\ell| = k'\ge k$, from the state at layer $\ell-d$: $\pi\big(h_{\ell-d}(t), \text{history}\big)$. Measured quantities:

- **Depth-$d$ recall** — the number that matters, because a single miss stalls:
$$R_d = \Pr\big[S_\ell(t) \subseteq \hat S_\ell(t)\cup C_\ell\big].$$
Measured per (layer, decode step) over a trace, not averaged over experts.
- **Non-trivial recall** — recall restricted to steps where a static frequency cache of the same size $m$ misses. This strips the free hits.
- **Lead time** $\tau_d$ = wall-clock between issuing the prefetch and the first use, measured with CUDA events, equal to the compute time of the $d$ intervening layers at the serving batch size.
- **Transfer time** $T = W/\beta_{\text{eff}}$, $\beta_{\text{eff}}$ = measured host-to-device bandwidth under concurrent compute (not the PCIe nameplate).
- **Stall fraction** $\sigma = \mathbb{E}[\max(0, T-\tau_d)\cdot \mathbb{1}[\text{miss}]] / \text{step time}$.
- **Predictor budget** $c$ = FLOPs plus bytes touched by $\pi$ per prediction; must satisfy $c \ll$ cost of layers $\ell-d..\ell$, else prediction is just early execution.

Batch size $B$: the set to fetch is $\bigcup_{t=1}^{B} S_\ell(t)$. Under uniform routing its expected size is $E\big(1-(1-k/E)^B\big)$ — for $E=8,k=2,B=32$ this is $7.9$ of $8$ experts, so caching and prefetching both become vacuous.

**Assumptions, and which are violated.** (i) *Routing is stationary across the trace* — violated: routing shifts with prompt domain and with position in the sequence. (ii) *Expert popularity is heavy-tailed* — holds for many checkpoints but is a training artifact, and load-balancing losses (Shazeer et al. 2017; Fedus et al. 2022) are explicitly designed to flatten it. (iii) *Consecutive tokens route similarly* — partially holds at first and last layers, weak in the middle. (iv) *Bandwidth is constant* — violated: $\beta_{\text{eff}}$ drops under concurrent kernel execution and PCIe contention.

## 3. State of the Art

**Systems/empirical SOTA.** Established and reproduced: offloading with an LRU expert cache plus one-layer-ahead speculative prefetch runs Mixtral-8x7B at interactive-ish rates on a single consumer GPU (Eliseev & Mazur, 2023). Fiddler (Kamahori et al., 2024) instead runs missing experts *on the CPU* rather than moving weights, reporting >3 tokens/s for Mixtral-8x7B on one 24 GB GPU and roughly an order-of-magnitude gain over weight-moving offload baselines — this is a benchmark number on a fixed setup, not an ablation of prefetch accuracy. Pre-gated MoE (Hwang et al., ISCA 2024) sidesteps prediction entirely by *retraining* the router at layer $\ell-1$ to emit layer $\ell$'s selection, reporting ~1.9$\times$ latency improvement with reduced GPU memory; the cost is a modified, retrained model. MoE-Infinity (Xue et al., 2024) uses per-sequence activation tracing for prefetch and cache admission and reports large cost reductions over vLLM/DeepSpeed offloading paths.

**Theory SOTA.** None. There is no published upper bound on $R_d$ as a function of $d$ and predictor budget $c$. The literature reports achieved accuracies, never a ceiling.

**Claimed but unablated.** Nearly every prefetcher paper reports end-to-end tokens/s and, sometimes, a hit rate — but almost none reports the lift over a *static top-$m$ frequency cache at matched VRAM*, which is the only control that separates "the predictor works" from "expert usage is skewed". Depth is also almost never swept: results are reported at $d=1$ and the accuracy-vs-depth curve is missing.

## 4. What Is Known

- **The transfer is the bottleneck, quantitatively.** Mixtral-8x7B: $D{=}4096$, intermediate $14336$, 3 matrices per expert $\Rightarrow P = 1.76\times10^8$ params/expert; fp16 $W = 352$ MB, 4-bit $\approx 88$ MB. 32 layers $\times$ 8 experts $= 256$ experts, 46.7 B params total, 12.9 B active per token (Jiang et al., 2024). Measured PCIe 4.0 x16 host-to-device is ~$20$–$25$ GB/s against a 32 GB/s nameplate.
- **Temporal locality is real but weak and layer-dependent.** Mixtral's own analysis finds consecutive-token expert repetition above the random baseline, most pronounced at the first and last layers; it also finds no clean topical specialization by domain. So the signal a prefetcher exploits is positional/sequential, not semantic.
- **Skew exists in trained routers.** Load-balanced training (GShard, Lepikhin et al., ICLR 2021) flattens but does not eliminate the popularity tail; a static cache therefore already captures a large fraction of accesses, which is exactly what inflates naive hit-rate numbers.
- **Retraining the router removes the prediction problem.** Pre-gated MoE (ISCA 2024) demonstrates that if you are willing to change the model, depth-1 accuracy becomes 100% by construction. This bounds the *value* of any pure-inference predictor: it can at best match a retrained gate.
- **Batch destroys the premise.** The coupon-collector arithmetic in §2 is not empirical but arithmetic: at $E=8$, $k=2$, $B\ge 32$ essentially every expert is needed each layer. All reported prefetch wins are small-batch or single-stream results.

## 5. What Is Not Known

- **Theoretically open.** No bound on $R_d$ under predictor budget $c$. The right formalism is resource-bounded prediction (the target is computable exactly, just expensively), and no one has instantiated it for router logits.
- **Empirically open.** The accuracy-versus-depth curve $R_d$ for $d = 1,2,4,8,16$ on a public checkpoint, on real serving traces, with a static-frequency control. Runnable today on one node; nobody has published it.
- **Methodologically blocked.** "Prefetch accuracy" has no agreed definition. Hit rate, recall, per-expert precision and stall fraction are used interchangeably, and they rank methods differently — a policy with lower miss *rate* but misses on cold, large experts can be slower.

## 6. Why It Is Hard

The obstruction is a **lead-time deficit combined with a confounded metric**, not compute cost.

- At batch 1, decode is memory-bound and one MoE layer takes on the order of $10^{-4}$–$10^{-3}$ s, while $T = 352\,\text{MB}/22\,\text{GB/s} \approx 16$ ms. Hiding one fp16 expert transfer needs $\tau_d \ge T$, i.e. tens of layers of lead — more depth than the model has. So the predictor must forecast across a horizon where hidden states have not been computed, using a budget too small to compute them.
- The metric confound: because expert popularity is skewed, a do-nothing static cache scores high hit rates. Papers reporting hit-rate improvements without that control are measuring skew, not prediction. This is an evaluation that does not measure the thing it names.
- Ground truth is trace-dependent: the "correct" prefetch depends on serving traffic that is not public, so cross-paper comparisons run on WikiText/C4 decoding, which understates domain shift.

## 7. Current Research (as of 2026)

- **Co-design over prediction** — pushing the gate earlier (Pre-gated MoE) or splitting compute across CPU and GPU (Fiddler) rather than predicting. This is the direction with the strongest reproduced results.
- **Activation-trace prefetchers** — per-sequence expert-activation histories driving cache admission (MoE-Infinity, Edinburgh); reported as system speedups, not as accuracy-vs-depth curves.
- **Edge/quantized offload** — EdgeMoE (Yi et al., 2023) and successors quantize experts to shrink $W$, converting the prediction problem into a bandwidth problem. Cutting fp16→4-bit divides $T$ by ~4 and is the single most reliable lever.
- *(frontier — verify)* Learned prefetchers trained with a router-distillation objective, and prefetch-aware routing regularizers that trade a small perplexity increase for expert-sequence predictability. Individual preprints exist; no independent reproduction of the accuracy claims.

## 8. Concrete Next Experiment

**Scale.** Mixtral-8x7B (46.7 B params, 32 layers, 8 experts, top-2), one A100-40GB or RTX 4090, experts in host DRAM, fp16 and 4-bit arms. 10k decode steps from 500 prompts spanning $\ge$4 domains (code, chat, math, retrieval). Batch sizes $B \in \{1, 8, 32\}$. One node, under a day.

**Arms.**
1. *Control:* static top-$m$ frequency cache, $m$ chosen to fill available VRAM, no prediction.
2. LRU + depth-1 speculative prefetch (Eliseev & Mazur configuration).
3. Learned prefetcher taking $h_{\ell-d}$ at $d\in\{1,2,4,8,16\}$, budget $c \le 1\%$ of one layer's FLOPs.
4. *Oracle ceiling:* true $S_\ell$ revealed at $\ell-d$ — isolates prediction error from bandwidth.

**Deciding number.** **Non-trivial recall lift at matched VRAM**: $\Delta R_d = R_d^{\text{arm 3}} - R_d^{\text{arm 1}}$, restricted to steps where arm 1 misses, reported as a curve over $d$. If $\Delta R_{d} < 0.05$ for $d\ge 4$ at $B=1$, pure-inference prefetching is dead and the field should commit to co-design or quantization. If $\Delta R_8 > 0.3$, a depth-8 predictor exists and the theory question (is there a ceiling?) becomes the live one. Secondary readout: stall fraction $\sigma$, which should track $\Delta R_d$ — if it does not, the metric definition (§5) is the blocker.

## 9. Key References

- **[Foundational]** Noam Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Dmitry Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[SOTA]** Albert Q. Jiang et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[SOTA]** Ranggi Hwang et al. *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.* ISCA, 2024. — arXiv:2308.12066
- **[SOTA]** Artyom Eliseev, Denis Mazur. *Fast Inference of Mixture-of-Experts Language Models with Offloading.* 2023. — arXiv:2312.17238
- **[SOTA]** Keisuke Kamahori et al. *Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.* 2024. — arXiv:2402.07033
- **[Systems]** Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, Mahesh Marina. *MoE-Infinity: offloading-efficient MoE model serving.* 2024. (arXiv preprint; identifier omitted — verify before citing.)
- **[Systems]** Samyam Rajbhandari et al. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[Systems]** Ying Sheng et al. *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU.* ICML, 2023. — arXiv:2303.06865
- **[Systems]** Rongjie Yi et al. *EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models.* 2023. — arXiv:2308.14352
- **[Survey]** Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, Jiayi Huang. *A Survey on Mixture of Experts.* 2024. (arXiv preprint; identifier omitted — verify before citing.)

## 10. Worked Example

Mixtral-8x7B, fp16, single stream, RTX 4090 (24 GB), experts in host DRAM, measured $\beta_{\text{eff}} = 22$ GB/s.

**Cold cost per token.** Top-2 over 32 layers = 64 expert loads $\times$ 352 MB = **22.5 GB** moved per token. At 22 GB/s that is **1.02 s/token** — under 1 token/s, before any compute. Even a perfect predictor cannot beat this: the bound is bandwidth, not accuracy.

**What caching buys.** 24 GB VRAM minus ~2 GB attention/embeddings/KV leaves 22 GB $\approx$ 62 of 256 experts resident (24% of the model). If routing were uniform, hit rate = 24% and 49 of 64 loads still miss: 17.2 GB, 0.78 s/token. Skew helps; suppose the static frequency cache achieves 60% hits. Then 25.6 misses $\times$ 352 MB = 9.0 GB = **0.41 s/token**.

**What prefetching can buy on top.** Prefetching does not remove a transfer, it only *hides* it behind compute. Available lead at depth $d$: one Mixtral layer at batch 1 reads ~0.75 GB of resident weights, so $\tau_1 \approx 0.75/1000 \approx 0.8$ ms on a 1 TB/s-class card. Hiding $T = 16$ ms needs $d \approx 20$ layers of lead. **Twenty layers ahead, in a 32-layer model.** The predictor would have to forecast layer 30's routing from layer 10's hidden state, using under 1% of a layer's FLOPs.

**The obstruction, made visible.** Even if the predictor were perfect, the remaining 25.6 missed experts per token still cost 9.0 GB of traffic; prefetch only overlaps at most $32 \times 0.8\,\text{ms} = 26$ ms of the 410 ms — **6%**. The realistic wins in the literature therefore come from shrinking $W$ (4-bit: $T$ drops to 4 ms, misses cost 2.3 GB $\to$ 0.10 s/token) or from not moving weights at all (Fiddler's CPU execution). Prefetch accuracy is being optimized in the regime where it has the least leverage, which is why a $\Delta R_d$ measurement against a static-cache control (§8) is the experiment that matters: it would tell us whether the accuracy knob is worth turning at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*