---
id: 15-mixture-of-experts/continuous-routing-relaxations
title: "Continuous Relaxations That Match Discrete Routing Quality"
topic: 15-mixture-of-experts
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continuous Relaxations That Match Discrete Routing Quality

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/continuous-routing-relaxations` · **Status:** solved-but-impractical

## 1. Problem Statement

Sparse MoE routing is a discrete assignment: a top-$k$ argmax over expert logits. The argmax is non-differentiable, so the router receives gradient only through the retained gate values — never through the counterfactual of an expert that was not chosen. This is the standard explanation for MoE training pathologies: router collapse, load-imbalance auxiliary losses, discontinuous loss surfaces, and sensitivity to initialization.

The problem: **construct a continuous, fully differentiable routing rule that matches or beats top-$k$ discrete routing on quality at equal training FLOPs and equal inference FLOPs, in the causal autoregressive setting.**

Three variants, with different difficulty:

- **Measurement.** Is the router-gradient bias actually the binding constraint on sparse MoE quality, or is it dominated by capacity/balance effects? Requires an estimator that separates the two.
- **Method.** Build a relaxation (soft assignment, Gumbel-softmax, Sinkhorn, expert-parameter merging) that is compute-matched to a sparse baseline and wins. Partly done — see §3 — but every existing win breaks either causality or the FLOP match.
- **Theory.** Prove or refute that a relaxation whose sparsification error vanishes still admits gradients informative enough to escape the collapse basin that top-$k$ falls into. Open.

"Solving it" means: a routing rule $r_\theta$, differentiable everywhere, with per-token inference cost $\le$ that of top-2 routing, causal-safe, that beats top-2 on validation loss at $\ge 10^{21}$ training FLOPs.

## 2. Formal Setting

Layer input token $x \in \mathbb{R}^d$; experts $\{f_e\}_{e=1}^{E}$, $f_e:\mathbb{R}^d\to\mathbb{R}^d$, parameters $\phi_e$. Router logits $z = W_r x \in \mathbb{R}^{E}$, $p = \mathrm{softmax}(z)$.

Discrete (top-$k$) layer output, with $\mathcal{S}_k(z)$ the index set of the $k$ largest logits:
$$y_{\text{hard}} = \sum_{e \in \mathcal{S}_k(z)} \frac{p_e}{\sum_{j\in\mathcal{S}_k}p_j} f_e(x).$$

Continuous relaxations, three families:

1. **Output mixing:** $y_{\text{soft}} = \sum_{e=1}^{E} p_e f_e(x)$ — exact gradient, but $E/k$ times the FLOPs.
2. **Parameter merging** (SMEAR): $y = f_{\bar\phi}(x)$, $\bar\phi = \sum_e p_e \phi_e$ — one expert's FLOPs plus an $O(E \cdot |\phi|)$ merge.
3. **Slot mixing** (Soft MoE): slots $\tilde{x}_s = \sum_{t} D_{ts} x_t$ over a sequence of $T$ tokens, dispatch weights $D = \mathrm{softmax}_{\text{tokens}}(X\Phi)$, and combine weights $C=\mathrm{softmax}_{\text{slots}}(X\Phi)$.

**Quantities as measured.**

- *Sparsification gap* $\Delta = \mathcal{L}(y_{\text{hard}}) - \mathcal{L}(y_{\text{soft}})$: measured by evaluating one trained checkpoint under both readouts on a held-out set, not by comparing two training runs.
- *Router gradient bias* $b = \mathbb{E}\!\left[\nabla_{W_r}\mathcal{L}_{\text{ST}} - \nabla_{W_r}\mathcal{L}_{\text{soft}}\right]$, estimated by running the dense-mixing backward pass on a subsample of batches during a sparse run. Cost: $E/k\times$ on those batches only.
- *Compute match:* count as inference FLOPs the expert matmuls **plus** the merge or slot-mixing cost. Merging $E$ experts of $|\phi|$ params costs $E|\phi|$ multiply-adds per token, which for $|\phi| \gg d$ dominates the expert forward pass — this is the term papers most often omit.
- *Load balance:* $\mathrm{CV} = \mathrm{std}_e(n_e)/\mathrm{mean}_e(n_e)$ over tokens per expert per batch.

**Assumptions, and which are violated.**

- *A1: the relaxation's optimum is near the discrete optimum.* Violated — annealing a Gumbel-softmax temperature $\tau\to 0$ moves the solution; Nie et al.'s dense-to-sparse gate shows the endpoint depends on the schedule.
- *A2: soft mixing is causal-safe.* Violated outright for Soft MoE: slots average over all tokens in the sequence, so a slot at position $t$ sees $t' > t$. This is why Soft MoE is a vision result.
- *A3: expert parameters are mergeable (loss is locally linear in $\phi$).* Violated at scale — merging is only well-behaved when experts stay in one linearly-connected basin, which is not enforced.
- *A4: equal FLOPs implies equal wall-clock.* Violated; merged-expert MoE has no expert parallelism, so it is memory-bandwidth-bound differently from dispatch-based MoE.

## 3. State of the Art

**Established (compute-matched, ablated).**

- **Soft MoE** (Puigcerver et al., ICLR 2024) beats sparse MoE and dense ViT on JFT-4B pretraining and ImageNet transfer at matched training cost, and Soft MoE L/16 exceeds ViT H/14 quality at far lower inference cost. Ablations cover slot count, expert count, and the sparse baselines (Top-$k$, Expert Choice, BASE). **Encoder-only; not causal.**
- **DSelect-k** (Hazimeh et al., NeurIPS 2021) gives a smooth, exactly-$k$-sparse selector via a binary-encoding relaxation; wins on multi-task recommendation benchmarks at small $E$. Cost grows as $E$ grows; not demonstrated at LM pretraining scale.
- **SMEAR** (Muqeeth, Liu, Raffel, TMLR 2024) merges expert parameters with router probabilities and beats top-1 discrete routing and several gradient-estimator baselines on T5-scale GLUE-style adaptation and ResNet/DomainNet — with adapter-sized experts, where the merge is cheap.

**Claimed but under-ablated.**

- **Lory** (Zhong et al., COLM 2024) makes causal LM merging tractable via a causal segment-level routing scheme plus similarity-based data batching; reports perplexity and downstream gains over dense at 0.3B/1.5B active params on 150B tokens. The FLOP accounting for the merge and the comparison against a *strong* top-2 MoE at matched total params is the weak point.
- **SparseMixer / GRIN** (Liu et al., 2023/2024) improve the router gradient estimator (a midpoint/Heun-style correction to straight-through) rather than relaxing the routing; GRIN reports strong 6.6B-active MoE results, but the estimator's contribution is entangled with architecture and data changes.

**Benchmark-number-only.** Sinkhorn/soft-balanced routers appear in Clark et al. (ICML 2022) scaling-law fits (S-BASE); the reported advantage is a fit over a scaling family, not a controlled head-to-head at fixed size.

## 4. What Is Known

- **Top-$k$ discrete routing works at frontier scale.** Switch Transformer (JMLR 2022) top-1 at 1.6T params; DeepSeekMoE-16B (2024) matches a 7B dense model at ~40% of its compute. Whatever the gradient bias costs, it does not prevent competitive models.
- **The router's learned assignment carries less signal than assumed.** Hash Layers (Roller et al., NeurIPS 2021) replace the learned router with a fixed hash and remain competitive with Switch on LM perplexity at ~1B scale. THOR (Zuo et al., ICLR 2022) routes to random experts and beats Switch on machine translation. This caps how much a better relaxation can buy.
- **Soft MoE's win is real in vision.** Matched-training-cost gains over Top-$k$ MoE and Expert Choice on JFT-4B/ImageNet at ViT-S through ViT-H scale.
- **Discrete routing collapses without an explicit fix.** Auxiliary balance loss (Shazeer 2017), capacity factors (Fedus 2022), or bias-based loss-free balancing (Wang et al., 2024) are required; the last removes the auxiliary-loss/quality tradeoff at DeepSeek-V2-class scale.
- **Router choice matters less than expected in vision.** Liu et al. (TMLR 2024) find most router variants within noise on ViT MoE once balancing is controlled.

## 5. What Is Not Known

- **Empirically open.** No published, compute-matched comparison of a causal continuous relaxation against a well-tuned top-2 sparse MoE at $\ge 10^{22}$ training FLOPs with total params, active params, data, and the merge/mixing FLOPs all held constant. Lory is the nearest approach and stops short at 1.5B active / 150B tokens. Runnable today for roughly $10^5$ GPU-hours.
- **Theoretically open.** Whether the loss surface of the relaxed objective has fewer bad stationary points than the straight-through objective; no separation theorem in either direction. Also open: whether output-mixing MoE and parameter-merging MoE have the same expressive class for nonlinear $f_e$ (they do not in general, but no bound on the gap).
- **Methodologically blocked.** Attributing a quality difference to *gradient bias* rather than to *effective capacity* or *balance*. Soft/merged routing changes all three at once. The bias estimator $b$ of §2 is definable but its variance at realistic $E$ and batch size has not been characterized, so the attribution is currently not measurable.

## 6. Why It Is Hard

The specific obstruction is **confounded compute accounting under a hard causality constraint**, not compute cost alone.

Every relaxation that has beaten discrete routing pays for the gradient in a currency the benchmark does not price:

| Family | Extra cost | Why it isn't in the headline number |
|---|---|---|
| Output mixing | $E/k\times$ expert FLOPs | Usually run only at small $E$ or as an analysis arm |
| Parameter merging | $E\cdot|\phi|$ MACs/token + no expert parallelism | Reported as "one expert's FLOPs" |
| Slot mixing | $O(T\cdot S\cdot d)$ and non-causal | Vision-only; the causality violation is architectural, not a tuning issue |

Second obstruction: **non-identifiability of the mechanism.** A soft router that wins may be winning because it is implicitly perfectly balanced, because it is effectively an ensemble, or because its gradients are unbiased. These are three different claims with three different implications for scaling, and no current experiment separates them.

Third: the Hash/THOR results mean the ceiling is low. If a fixed random router is within ~1% of a learned one, the maximum prize from fixing router gradients is small — and smaller than the noise floor of most pretraining comparisons.

## 7. Current Research (as of 2026)

- **Better estimators over better relaxations.** SparseMixer-style corrections and GRIN (Microsoft) keep hard routing and fix the gradient. This is the direction with frontier-scale evidence. *(frontier — verify current deployment status.)*
- **Causal segment-level merging.** Lory-line work (Princeton/Meta authors) on making parameter merging causal; the open question is whether it survives contact with a tuned top-$k$ baseline at 10B+ total params. *(frontier — verify.)*
- **Loss-free balancing** (DeepSeek): if bias-adjusted routing removes the balance confound, it becomes the correct control arm for any relaxation study.
- **Soft MoE in decoders.** Attempts to restore causality via causal-masked slot construction; no compute-matched LM result published that we can verify. *(frontier — verify.)*
- **Representation collapse** (X-MoE, Chi et al. 2022) as a competing explanation for router failure — dimension collapse of routed representations rather than gradient bias.

## 8. Concrete Next Experiment

**Question:** at fixed training and inference FLOPs, does a causal continuous relaxation beat top-2 discrete routing on LM validation loss?

- **Scale.** 1.3B active / ~8B total params, $E=32$ experts, 100B tokens of a fixed corpus (~$8\times10^{21}$ FLOPs). Three seeds per arm. Roughly 4k H100-days total.
- **Arms.**
  1. *Control:* top-2 MoE with loss-free bias balancing (Wang et al., 2024), tuned capacity factor.
  2. *Relaxation A:* segment-level causal parameter merging (Lory-style), with the $E|\phi|$ merge FLOPs **charged to the budget** — i.e. shrink $d_{\text{ff}}$ until total FLOPs match arm 1.
  3. *Relaxation B:* Gumbel-softmax top-2 with $\tau$ annealed $2.0\to0.1$, identical FLOPs to arm 1.
  4. *Ceiling:* full dense mixing over all 32 experts, $16\times$ the FLOPs — not a fair arm, it bounds $\Delta$.
- **Measurement.** Also log $b$ (§2) on 1% of batches in arms 1–3, and $\mathrm{CV}$ throughout.
- **The deciding number.** $\delta = \mathcal{L}_{\text{val}}(\text{arm 1}) - \min(\mathcal{L}_{\text{val}}(\text{arm 2}), \mathcal{L}_{\text{val}}(\text{arm 3}))$ in nats/token at 100B tokens. Seed std at this scale is ~0.004 nats. **$\delta > 0.01$ nats decides for relaxations; $|\delta| < 0.005$ closes the method variant as negative** and redirects effort to estimators. If arm 4 also shows $\delta_4 < 0.02$, the entire premise — that discreteness costs quality — is falsified at this scale.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** Jang, Gu, Poole. *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017. — arXiv:1611.01144
- **[Foundational]** Maddison, Mnih, Teh. *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* ICLR 2017. — arXiv:1611.00712
- **[SOTA]** Puigcerver, Riquelme, Mustafa, Houlsby. *From Sparse to Soft Mixtures of Experts.* ICLR 2024. — arXiv:2308.00951
- **[SOTA]** Muqeeth, Liu, Raffel. *Soft Merging of Experts with Adaptive Routing.* TMLR 2024. — arXiv:2306.03745
- **[SOTA]** Zhong, Meng, Chen, Danqi Chen et al. *Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training.* COLM 2024. — arXiv:2405.03133
- **[SOTA]** Hazimeh, Zhao, Chowdhery, Sathiamoorthy, Chen, Mazumder, Hong, Chi. *DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning.* NeurIPS 2021. — arXiv:2106.03760
- **[Baseline]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. — arXiv:2101.03961
- **[Baseline]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022. — arXiv:2202.09368
- **[Ceiling]** Roller, Sukhbaatar, Szlam, Weston. *Hash Layers For Large Sparse Models.* NeurIPS 2021. — arXiv:2106.04426
- **[Ceiling]** Zuo, Liu, Jiao, Kim, Hassan, Zhang, Zhao, Gao. *Taming Sparsely Activated Transformer with Stochastic Experts.* ICLR 2022. — arXiv:2110.04260
- **[Control arm]** Wang, Chen, Xie, Zhao, Dai et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[Estimator]** Liu, Gao, Chen. *Sparse Backpropagation for MoE Training.* 2023. — arXiv:2310.00811
- **[Survey]** Clark, de las Casas, Guy, Mensch et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[Diagnosis]** Chi, Dong, Huang, Dai, Ma, Patra, Singhal, Bajaj, Song, Wei. *On the Representation Collapse of Sparse Mixture of Experts.* NeurIPS 2022. — arXiv:2204.09179

## 10. Worked Example

Take one MoE FFN layer: $d = 2048$, $d_{\text{ff}} = 8192$ per expert, $E = 32$, top-2. Expert params $|\phi| = 2 d\, d_{\text{ff}} = 3.36\times10^{7}$.

**Discrete, per token:** 2 experts $\times$ $2d\,d_{\text{ff}}$ MACs $= 6.7\times10^{7}$.

**Parameter merging, per token:** one expert forward, $3.36\times10^{7}$ MACs — plus the merge, $E\,|\phi| = 32 \times 3.36\times10^{7} = 1.07\times10^{9}$ MACs. Total $1.10\times10^{9}$, which is **16.4× the discrete cost**, not 0.5×.

This is the obstruction, made numeric. The merge is only cheap when it is amortized over a segment of tokens sharing one merged expert. Amortizing over a segment of length $L$ gives per-token cost $3.36\times10^{7} + 1.07\times10^{9}/L$. Setting that equal to the discrete $6.7\times10^{7}$ requires
$$L \ge \frac{1.07\times10^{9}}{6.7\times10^{7} - 3.36\times10^{7}} \approx 32.$$

So parameter merging is FLOP-competitive only if the routing decision is held fixed for $\ge 32$ consecutive tokens. That is exactly Lory's segment-level scheme — and it means the relaxation buys differentiability by **throwing away per-token routing granularity**, the property that motivated MoE in the first place. A 32-token segment router makes roughly $1/32$ as many decisions per sequence as top-2.

The obstruction is therefore a conservation law, not an engineering gap: at fixed FLOPs, differentiable routing and per-token routing trade against each other at a rate set by $E|\phi|/(k \cdot 2d\,d_{\text{ff}}) = E/(2k) = 8$ here. Any claimed win must state which side of that trade it paid on. Most do not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*