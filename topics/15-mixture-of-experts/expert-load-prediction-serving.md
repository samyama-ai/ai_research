---
id: 15-mixture-of-experts/expert-load-prediction-serving
title: "Expert Capacity Load Prediction for Serving"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expert Capacity Load Prediction for Serving

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/expert-load-prediction-serving` · **Status:** empirically-open

## 1. Problem Statement

A sparse MoE layer routes each token to $k$ of $E$ experts. At serving time the experts are spread over devices, and some are offloaded to CPU or NVMe. The per-step cost is set by the *most loaded* device and by which expert weights are resident when needed. So the serving system wants, **before** the layer runs, a prediction of the load vector $n = (n_1,\dots,n_E)$ — tokens per expert — for the current batch.

Three variants, of very different difficulty:

- **Measurement.** Define a prediction target and an error metric whose improvement provably reduces tail latency, not just routing entropy. Which functional of $n$ actually controls the step time?
- **Method.** Build a predictor $\hat n$ (from prompt features, previous-layer hidden states, or previous-token routing) that is cheap enough to run in the critical path and accurate enough to drive prefetch, expert placement, or capacity allocation.
- **Theory.** Characterize when $n$ is predictable at all: is the routing map from residual stream to expert index Lipschitz/low-entropy enough that a $O(d)$-cost probe can beat the marginal frequency prior?

**Solved** = a serving system that, at fixed model and fixed quality (bit-exact or within noise on a held-out benchmark), reduces p99 decode latency by a stated margin over a strong static-placement baseline, with the gain attributed by ablation to prediction accuracy rather than to extra memory or a better scheduler.

## 2. Formal Setting

Layer $\ell$, hidden states $H\in\mathbb{R}^{T\times d}$ for $T$ tokens in flight. Router $g_\ell(h)=\mathrm{TopK}(W_\ell h)$, $W_\ell\in\mathbb{R}^{E\times d}$. Assignment indicator $a_{t,e}\in\{0,1\}$, load

$$n_e \;=\; \sum_{t=1}^{T} a_{t,e},\qquad \sum_e n_e = kT .$$

*Measured as:* a counter incremented in the dispatch kernel, dumped per layer per step. This is the ground truth and it is cheap — the difficulty is not measuring $n$, it is measuring what $n$ costs.

Devices $\mathcal{D}$, placement $\pi:[E]\to\mathcal{D}$. Device load $L_d=\sum_{e:\pi(e)=d} n_e$. Step time under all-to-all dispatch:

$$\tau \;\approx\; \underbrace{\max_{d} \frac{L_d\,c_{\text{gemm}}}{\rho_d}}_{\text{compute skew}} \;+\; \underbrace{\max_{d}\frac{B_{\text{in}}(d)+B_{\text{out}}(d)}{\beta}}_{\text{all-to-all}} \;+\; \underbrace{\sum_{e\in \mathcal{M}(t)} \frac{|\theta_e|}{\beta_{\text{pcie}}}}_{\text{miss stalls}}$$

with $\rho_d$ device throughput (tokens/s, measured by a per-shape microbenchmark), $\beta$ interconnect bandwidth, $\mathcal{M}(t)$ the set of experts needed but not resident, $|\theta_e|$ expert bytes. *Measured as:* CUDA events around dispatch / expert GEMM / combine, plus a resident-set trace from the offload manager.

Two decision problems consume $\hat n$:

1. **Capacity / drop.** With capacity factor $f$, capacity $C=f\,kT/E$; tokens beyond $C$ are dropped or overflow to a slow path. Drop rate $\delta=\frac{1}{kT}\sum_e (n_e-C)^+$.
2. **Prefetch.** Predict the support $S=\{e:n_e>0\}$ (or top-$m$ by load) $\Delta$ layers ahead. Hit rate $\mathrm{HR}=\Pr[e\in \hat S \mid e \in S]$ weighted by $n_e$.

Useful error metrics, in increasing relevance:

$$\mathrm{MAE}=\tfrac{1}{E}\|\hat n - n\|_1,\qquad \mathrm{TailErr}=\big|\max_d \hat L_d - \max_d L_d\big|,\qquad \mathrm{Regret}=\tau(\pi(\hat n))-\min_\pi \tau(\pi(n)).$$

Only $\mathrm{Regret}$ is the objective; MAE is the one usually reported.

Skew is summarized by the standard MoE imbalance statistic $\mathrm{CV}=\mathrm{std}(n)/\mathrm{mean}(n)$, or the GShard/Switch balance loss $E\sum_e \frac{n_e}{kT}\bar p_e$ with $\bar p_e$ the mean router probability.

**Assumptions, and which are violated:**
- *Tokens i.i.d. given the prompt* — violated: routing is strongly autocorrelated across consecutive decode tokens and across layers.
- *Stationary routing distribution* — violated: load shifts with domain, language, and sequence position; DeepSeek-V3-class models add bias terms that move load online.
- *$\rho_d$ homogeneous* — violated in mixed fleets and under thermal throttling.
- *Prediction is free* — violated: a predictor in the critical path costs microseconds that the saved stall must exceed.
- *Prefill and decode share a load distribution* — violated: prefill has $T\!\sim\!10^3$ tokens per step (loads concentrate near uniform), decode has $T\!\sim\!$ batch size (loads are sparse and lumpy).

## 3. State of the Art

**Established (ablated, reproduced):**
- **Balance-by-training.** GShard auxiliary loss (Lepikhin et al., ICLR 2021) and Switch Transformer (Fedus et al., JMLR 2022) reduce skew but do not remove it; capacity factor remains a tuned knob.
- **Balance-by-construction.** Expert Choice routing (Zhou et al., NeurIPS 2022) makes $n_e$ exactly $C$ by letting experts pick tokens — the load becomes *known a priori*. It is not causal, so it is not usable for autoregressive decode as-is.
- **Balance-by-assignment.** BASE Layers (Lewis et al., ICML 2021) solves a linear assignment to force $n_e=T/E$.
- **Drop-free kernels.** MegaBlocks (Gale et al., MLSys 2023) reformulates the expert MLP as block-sparse matmul, removing token dropping and thus removing capacity prediction from the *quality* path — but not from the *latency* path.
- **Load-aware serving.** DeepSpeed-MoE (Rajbhandari et al., ICML 2022) and Tutel (Hwang et al., MLSys 2023) show adaptive parallelism/capacity at runtime beats static configuration.

**Claimed but unablated / benchmark-only:**
- Prefetch and offload systems — Pre-gated MoE (Hwang et al., ISCA 2024), SiDA-MoE (MLSys 2024), MoE-Infinity (Xue et al.) — report large speedups for offloaded serving, but the reported number bundles predictor accuracy, cache policy, and I/O scheduling. Almost none report *Regret* against an oracle-$n$ arm, which is the only way to isolate prediction quality.
- Expert-activation "locality" claims (skewed expert popularity, temporal reuse across decode steps) are consistently observed but the measured quantity differs per paper (top-1 hit rate, unique experts per sequence, cache miss rate), so the numbers are not comparable across systems.

## 4. What Is Known

- **Skew is real and layer-dependent.** Switch Transformer and ST-MoE (Zoph et al., 2022) report that balance losses are needed at every scale; residual imbalance persists. Reported capacity factors in production-scale training sit around $f\in[1.0,2.0]$ — i.e. systems provision 1–2× the mean to absorb skew.
- **Auxiliary-loss-free balancing works.** DeepSeek's bias-adjusted routing (Wang et al., 2024; deployed in DeepSeek-V3, 2024) achieves balance comparable to aux-loss training without the quality tax, and DeepSeek-V3 serving uses *redundant experts* — duplicating high-load experts across devices — with the duplication set from observed load statistics. This is prediction by moving average, and it is the strongest deployed baseline. Scale: 671B total / 37B active parameters, 256 routed experts per layer, 8 active.
- **Mixtral 8×7B** (Jiang et al., 2024) reports expert assignment is far from uniform and shows temporal locality: consecutive tokens frequently reuse the same expert, with repeat rates well above the $1/8$ chance level at higher layers. This is the empirical basis for every prefetcher.
- **Prefill vs decode differ.** With $T=4096$ prefill tokens and $E=64$, the mean per-expert load is 64·k tokens and relative fluctuation is small; with decode batch 8 and $k=2$, at most 16 of 64 experts are touched — the support prediction problem is the binding one at decode, the balance problem at prefill.

## 5. What Is Not Known

- **Methodologically blocked:** there is no agreed target. Papers optimize MAE, top-$m$ hit rate, or cache miss rate; none reports $\mathrm{Regret}$ against an oracle. Two predictors with identical MAE can differ by 2× in p99 latency because the cost is a max over devices, not a mean over experts. Until the reported metric is a latency regret against an oracle-$n$ arm, cross-paper comparison is meaningless.
- **Empirically open:** whether a cheap causal predictor beats the *moving-average prior* (per-expert historical frequency, the DeepSeek-V3 redundancy rule) by enough to pay for itself. Nobody has published the head-to-head at production scale with the prior as the control arm.
- **Theoretically open:** no bound relating router margin (the gap between top-$k$ and $(k{+}1)$-th logit) to predictability of $n$ from layer $\ell-\Delta$ activations. Equivalently: no lower bound saying $\Delta$-layer-ahead prediction must fail beyond some $\Delta$.
- **Open:** how predictability degrades under distribution shift at serving — multilingual, code, long-context, and adversarial prompts designed to concentrate load on one expert (a denial-of-service surface nobody has quantified).

## 6. Why It Is Hard

Three specific obstructions.

1. **The objective is a max, the metric is a mean.** $\tau$ depends on $\max_d L_d$. Squared or absolute error over experts is nearly uninformative about the max under skewed loads: reducing MAE by improving the prediction on 60 lightly-loaded experts changes $\tau$ by zero. The evaluation does not measure what it names.
2. **Confounded measurement.** Every published prefetch speedup mixes predictor quality, cache capacity, and overlap scheduling. Without an oracle-$n$ arm (which is trivially runnable — run the router twice) the attribution is unidentifiable.
3. **Prediction must beat a very strong trivial prior at near-zero cost.** Expert popularity is heavy-tailed and stable over minutes. A per-expert exponential moving average costs $O(E)$ and captures most of the exploitable structure. A learned predictor must beat it by more than its own latency, inside a decode step that is 5–20 ms end-to-end. The margin available is small and is itself unmeasured.

## 7. Current Research (as of 2026)

- **Deployed load-driven placement.** DeepSeek-style redundant-expert duplication with periodic rebalancing from live counters is the current production answer; refinement is incremental (rebalance period, duplication budget).
- **Ahead-of-layer probes.** Predicting layer $\ell$'s routing from layer $\ell-1$ or $\ell-2$ hidden states with a tiny linear head, to hide PCIe transfer — the Pre-gated / SiDA line, now being pushed to NVMe-tier offload. *(frontier — verify: current accuracy numbers at $E\ge 128$.)*
- **Request-level scheduling.** Batching requests whose predicted expert supports overlap, so a device serves one resident set per batch. Reported in academic serving systems; no published production ablation. *(frontier — verify.)*
- **Making prediction unnecessary.** Expert-Choice-style or assignment-based routing at inference, block-sparse drop-free kernels (MegaBlocks lineage), and grouped/shared-expert designs that bound per-device load by construction. This is the strongest competitor to the whole problem: the best predictor may be a routing scheme that makes $n$ deterministic.

## 8. Concrete Next Experiment

**Scale.** One open MoE with $E\ge 64$ (Mixtral 8×7B for a small arm; a DeepSeek-V3-class or OLMoE checkpoint for the large arm), served on 8 GPUs with 25–50% of expert parameters offloaded to host memory. Decode batch sizes $\{1,8,32\}$, 10k requests mixing chat, code, and non-English prompts.

**Arms** (identical kernels, cache size, and scheduler; only the load estimate differs):
- **A. Oracle.** Run the router for layer $\ell$ before the layer executes; perfect $n$. Upper bound.
- **B. Control — EMA prior.** Per-expert exponential moving average of $n_e$ over the last 1000 steps. This is the arm to beat.
- **C. Learned probe.** Linear head on layer $\ell-1$ hidden states predicting layer $\ell$ top-$m$ support.
- **D. Static uniform.** No prediction.

**Deciding number.** The **fraction of the oracle's p99 decode-latency gain that arm C captures beyond arm B**:

$$\mathrm{Gain} = \frac{\tau^{p99}_{B}-\tau^{p99}_{C}}{\tau^{p99}_{B}-\tau^{p99}_{A}}.$$

Report with the predictor's own latency included. $\mathrm{Gain} > 0.5$ at batch 8 means learned prediction is worth building. $\mathrm{Gain} < 0.1$ means the EMA prior already extracts the available signal and the field should redirect to routing schemes that make $n$ deterministic. Also report MAE per arm, to document whether MAE ranks the arms the same way regret does — the answer to that is itself publishable.

## 9. Key References

- **[Foundational]** Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[Foundational]** Lewis, Bhosale, Dettmers, Goyal, Zettlemoyer. *BASE Layers: Simplifying Training of Large, Sparse Models.* ICML, 2021. — arXiv:2103.16716
- **[SOTA]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[SOTA]** Rajbhandari, Li, Yao, Zhang, Aminabadi, Awan, Rasley, He. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[SOTA]** Hwang, Cui, Xiong, Yang, Liu, Xu, Yang, Zhang, Zhou et al. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys, 2023. — arXiv:2206.03382
- **[SOTA]** Gale, Narayanan, Young, Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA]** Wang, Zhu, Dai, Gao, Sun, Wang. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437 (redundant-expert deployment)
- **[Systems]** Hwang, Wei, Kim, Kwon, Kim, Ahn, Kim, Rhu. *Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.* ISCA, 2024. — arXiv:2308.12066
- **[Systems]** Du, Li, Zhu et al. *SiDA-MoE: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models.* MLSys, 2024.
- **[Systems]** Li, Jiang, Zhu, Chen et al. *Accelerating Distributed MoE Training and Inference with Lina.* USENIX ATC, 2023.
- **[Empirical]** Jiang et al. (Mistral AI). *Mixtral of Experts.* 2024. — arXiv:2401.04088 (expert-assignment and temporal-locality analysis)
- **[Empirical]** Zoph, Bello, Kumar, Du, Huang, Dean, Shazeer, Fedus. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906

## 10. Worked Example

Mixtral-style layer: $E=8$, $k=2$, decode batch $T=32$, so $kT=64$ token-expert assignments, mean load 8. Two experts offloaded to host; PCIe Gen4 ×16 at $\approx 20$ GB/s effective, expert weights $\approx 340$ MB in bf16 → 17 ms to fetch one expert. Resident-expert step time $\approx 6$ ms.

Observed load on one step: $n=(21,14,9,7,5,4,3,1)$. Mean 8, $\mathrm{CV}=0.79$.

**Predictor P1** predicts $\hat n=(19,15,9,8,5,4,3,1)$. MAE $=\frac{1}{8}(2+1+0+1+0+0+0+0)=0.5$.
**Predictor P2** predicts $\hat n=(8,8,8,8,8,8,8,8)$ — the uniform prior. MAE $=\frac{1}{8}(13+6+1+1+3+4+5+7)=5.0$.

P1 looks 10× better. Now the decision. Suppose expert 8 (load 1) is the offloaded one and both predictors correctly rank it last, so both prefetch the same set and both stall 0 ms. Regret is identical: $0$ ms. MAE separated the predictors by 10×; latency separated them by nothing.

Flip it. Suppose the offloaded expert is the one that actually took load 5 (rank 5 of 8), and the cache holds 6 experts. P1 ranks it 5th → prefetched → 0 ms stall. P2's uniform prediction gives an arbitrary tie-break; with probability $2/8$ it evicts that expert → expected stall $0.25\times 17 = 4.25$ ms on a 6 ms step, a 1.7× slowdown. Same MAE gap as before; now it costs 71%.

The obstruction is visible: MAE moved by 10× in both cases while regret moved between 0 ms and 4.25 ms, depending entirely on whether the error landed on the ranking boundary at the cache size. The quantity the field reports is not monotone in the quantity the system pays. Until the oracle arm of §8 is run and $\mathrm{Regret}$ is the reported number, "better expert load prediction" is not a well-posed claim.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*