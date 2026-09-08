---
id: 10-scaling-laws/depth-versus-width-allocation
title: "Depth Versus Width Allocation Law"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Depth Versus Width Allocation Law

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/depth-versus-width-allocation` · **Status:** open

## 1. Problem Statement

Chinchilla-style scaling laws allocate a compute budget between parameter count $N$ and token count $D$. They say nothing about how to *spend* $N$ — how many layers $L$, at what residual width $d$. Given $N$ fixed, the space of shapes $(L, d, d_{\text{ffn}}, n_{\text{head}})$ satisfying $N \approx 12 L d^2$ is a one-parameter family, and the question is which point on it minimises loss.

Three variants, of different difficulty:

- **Measurement.** For a fixed $N$ and token budget $D$, does loss vary enough across shapes to matter, and by how much, once learning rate and batch size are re-tuned per shape? Runnable today.
- **Method.** Is there a fitted law $L^\star(N)$ — or $L^\star(C)$ under a compute budget $C$ — that predicts the loss-minimising depth from below, and extrapolates across two orders of magnitude of $N$? Partially attempted.
- **Theory.** Is there a derivation of the exponent in $L^\star \propto N^{\alpha}$ from expressivity or optimisation, rather than a curve fit? Open.

A solution is a law $L^\star(N, D)$ plus a stated loss penalty $\Delta \mathcal{L}(L/L^\star)$ for being off-optimum, validated by held-out extrapolation to a scale $\ge 10\times$ beyond the fitting range.

## 2. Formal Setting

A decoder-only transformer is specified by $\theta = (L, d, d_{\text{ffn}}, n_{\text{head}}, d_{\text{head}}, V, T)$. Non-embedding parameters, as counted in code (not asymptotically):

$$N = L\left(4d^2 + 2 d\, d_{\text{ffn}}\right) \;\approx\; 12 L d^2 \quad \text{when } d_{\text{ffn}} = 4d.$$

**Aspect ratio** $\rho = d / L$; **shape exponent** $\alpha$ defined by the fitted relation $L^\star(N) \propto N^{\alpha}$. Constant $\rho$ corresponds to $\alpha = 1/3$ (since $N \propto d^3$ when $L \propto d$); depth growing logarithmically in $N$ corresponds to $\alpha \to 0$.

**Objective.** For budget $C$ (FLOPs) with the standard estimate $C \approx 6ND$:

$$L^\star(C) = \arg\min_{L} \; \min_{\eta, B, \text{init}} \; \mathcal{L}\big(\theta(L, N(C)), D(C); \eta, B\big),$$

where $\mathcal{L}$ is held-out cross-entropy in nats/token on a fixed validation set drawn from the training distribution. The inner minimisation over learning rate $\eta$, batch size $B$, and initialisation scale is the part that is usually skipped and is the reason most reported comparisons are uninterpretable.

**Measured quantities.**
- $\mathcal{L}$: mean negative log-likelihood over $\ge 10^7$ held-out tokens; run-to-run seed noise at 1B scale is roughly $\pm 0.003$–$0.01$ nats, which sets the resolution floor.
- $C$: counted, not estimated, via per-layer FLOP accounting including attention $\mathcal{O}(L T^2 d)$ terms — the $6ND$ approximation drops these and biases long-context comparisons toward wide models.
- Wall-clock throughput $\tau$ (tokens/s) at fixed hardware and parallelism plan, reported separately from $C$.
- Inference latency $\ell$ per decoded token, which scales with $L$ (serial) but only weakly with $d$ under tensor parallelism.

**Assumptions, with the violated ones flagged.**
1. $N \approx 12Ld^2$ ignores embeddings — **violated** below ~1B params, where $2Vd$ is a large fraction of $N$ and biases the optimum toward small $d$.
2. Optimal $(\eta, B)$ transfer across shapes — **violated**: $\mu$P transfers across width but the depth direction needs the separate $1/\sqrt{L}$ residual-branch scaling of Depth-$\mu$P (Yang et al., 2024), and even that is proven only for the residual-block class.
3. FLOPs are the cost — **violated** in every deployment: depth buys latency and pipeline bubbles that FLOPs do not price.
4. Validation loss ranks models the same way downstream tasks do — **violated**; Tay et al. (2022) show shape changes reorder pretraining and downstream rankings.

## 3. State of the Art

**Established (ablated, reproduced).**
- Kaplan et al. (2020) held $N$ fixed and swept shape: loss depends only weakly on aspect ratio. Across aspect ratios spanning more than an order of magnitude at $10^7$–$10^8$ non-embedding parameters, loss moves by a few percent — a flat basin, not a sharp optimum.
- Alabdulmohsin et al. (NeurIPS 2023) fit per-dimension shape laws for ViT and derived SoViT-400m/14, which reaches 90.3% ImageNet top-1 after fine-tuning — matching a ViT-g/14 roughly 4$\times$ its size. Key ablated finding: the dimensions saturate at *different* rates, with MLP width the first to scale, then depth, then width.
- Depth-$\mu$P (Yang, Yu, Zhu, Hayou, ICLR 2024): scaling residual branches by $1/\sqrt{L}$ gives hyperparameter transfer across depth for residual networks. This makes iso-hyperparameter depth comparisons well posed for the first time.

**Claimed but unablated, or benchmark-only.**
- Levine et al. (NeurIPS 2020) derive a depth–width interplay for self-attention in which useful depth grows *logarithmically* with width, implying $\alpha \approx 0$. The theory is for a simplified self-attention model; the language-modelling validation is at $\le 10^8$ parameters and has not been re-run at $10^{10}$.
- MobileLLM (Liu et al., ICML 2024) reports that deep-and-thin beats wide-and-shallow at 125M/350M, with 2.7%/4.3% average zero-shot accuracy gains over prior SOTA. This is a benchmark number on a package of changes; the depth contribution is not isolated with re-tuned hyperparameters.
- Tay et al. (ICLR 2022) "DeepNarrow" reports matching or beating T5-Base with ~50% fewer parameters and faster training. Real, but on the T5 encoder-decoder family with fixed hyperparameters, so the depth effect and the LR-mistuning effect are confounded.

**No theory SOTA.** There is no derivation of $\alpha$ for realistic transformers.

## 4. What Is Known

- **Flat basin.** At fixed $N$, loss is within a few percent over aspect ratios varying by more than $10\times$ (Kaplan et al., 2020, at $10^7$–$10^8$ non-embedding params). At 1B scale, published sweeps put the spread across $L \in [12, 48]$ at order $10^{-2}$ nats — only a few times seed noise.
- **Deployed shapes do not agree on $\alpha$.** GPT-2 1.5B ($L{=}48$, $d{=}1600$, $\rho{=}33$) to GPT-3 175B ($L{=}96$, $d{=}12288$, $\rho{=}128$) implies $\alpha = \ln 2 / \ln 117 \approx 0.15$. Llama-3 8B ($L{=}32$, $d{=}4096$) to Llama-3 70B ($L{=}80$, $d{=}8192$) implies $\alpha = \ln 2.5/\ln 8.75 \approx 0.42$. Both families were tuned by competent labs; the implied exponents differ by 3$\times$.
- **Depth is expressivity-relevant.** Constant-depth log-precision transformers are contained in uniform $\mathrm{TC}^0$ (Merrill & Sabharwal, TACL 2023); serial-composition tasks need depth that width cannot buy. Classical depth-separation results (Telgarsky, COLT 2016; Eldan & Shamir, COLT 2016) show exponential width costs for constant-depth emulation of deeper nets.
- **Depth helps compositional generalisation at fixed $N$, with sharply diminishing returns** (Petty et al., NAACL 2024) — most of the gain arrives in the first few layers.
- **Depth is not free at inference.** Decode latency is serial in $L$; width is parallelised by tensor parallelism. Two iso-$N$, iso-loss models can differ by 20–30% in tokens/s.

## 5. What Is Not Known

- **Theoretically open.** The value of $\alpha$, or even whether $L^\star$ grows polynomially or logarithmically in $N$. Levine et al. argue log; deployed practice looks polynomial. No proof either way for real transformer training dynamics.
- **Empirically open.** Whether the flat basin *stays* flat past $10^{10}$ parameters, or whether the loss-vs-depth curve sharpens. The experiment is a shape sweep with per-shape hyperparameter re-tuning at 10B; nobody has published one. Cost is the only blocker.
- **Methodologically blocked.** The objective itself. "Optimal shape" is undefined until the cost model is fixed: iso-parameter, iso-FLOP, iso-training-wall-clock, and iso-inference-latency optima are four different points, and papers routinely report one while claiming another. There is no agreed cost functional $\mathcal{J}(L, d)$ that prices serial depth.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement inside a signal that is smaller than the confounds**. The shape effect at fixed $N$ is order $10^{-2}$ nats. Three nuisance effects are the same size or larger:

1. **Hyperparameter mis-tuning.** Optimal $\eta$ moves with $L$; without Depth-$\mu$P or a per-shape sweep, an iso-$\eta$ comparison measures LR-transfer failure and labels it depth.
2. **Embedding accounting.** Below ~1B, holding $N$ fixed with or without $2Vd$ embeddings flips which shape looks best.
3. **Seed noise.** $\pm 0.003$–$0.01$ nats at 1B — so a 3-seed comparison can resolve the effect only with several seeds per shape, multiplying an already expensive sweep.

Correcting all three costs $\ge 5\times$ the compute of a naive sweep at every scale, and the sweep must be repeated at several $N$ to fit an exponent. That is why $\alpha$ is estimated from deployed model families instead — and those are a non-identifiable source, because lab shape choices are jointly determined by loss, hardware topology, and serving latency, which the published number cannot separate.

## 7. Current Research (as of 2026)

- **Shape-aware scaling laws.** Google DeepMind's ViT shape-law line (Alabdulmohsin, Zhai, Kolesnikov, Beyer) extended to decoder-only LMs *(frontier — verify)*.
- **Depth parameterisation.** Depth-$\mu$P and successors (Yang, Hayou and collaborators); the open question is whether $1/\sqrt{L}$ branch scaling holds with normalisation layers and attention, not just plain residual blocks.
- **Inference-aware allocation.** Sardana et al. (ICML 2024) put inference cost into the $N$-vs-$D$ law; the natural next step is putting serial-depth latency into the shape law. Not yet published as such *(frontier — verify)*.
- **Layer-sharing and recurrent-depth models** (e.g. weight-tied "looped" transformers) decouple $L$ from $N$ and turn the shape question into a compute-per-token question.
- **Mechanistic depth studies** measuring per-layer marginal contribution via layer pruning, which consistently find deep-middle layers highly redundant — evidence that the effective depth is well below $L$.

## 8. Concrete Next Experiment

**Question.** Is $\alpha$ in $L^\star \propto N^{\alpha}$ distinguishable from $1/3$ (constant aspect ratio) and from $0$ (log depth)?

**Scale.** Four parameter budgets: $N \in \{0.3, 1.2, 4.8, 19\}$B non-embedding, Chinchilla-matched tokens ($D = 20N$). At each $N$, five shapes at iso-$N$ spanning $\rho \in \{16, 32, 64, 128, 256\}$. Total ~20 runs; the 19B tier dominates at roughly $2.3 \times 10^{22}$ FLOPs per run.

**Control arm.** Every shape trained under Depth-$\mu$P ($1/\sqrt{L}$ residual branch scaling) with $\eta$ transferred from a width-64, depth-8 proxy sweep. Control: the *same* grid trained with a single fixed $\eta$ tuned only at $\rho = 128$. The gap between arms is the size of the mis-tuning confound and must be reported. Three seeds at the two smallest tiers to pin seed noise; one seed above.

**Deciding number.** Fit $\log L^\star = \alpha \log N + c$ by parabolic interpolation of $\mathcal{L}$ vs $\log \rho$ at each tier, with bootstrap CIs. The experiment resolves the question iff the 95% CI on $\alpha$ has width $< 0.15$ and excludes at least one of $\{0, 1/3\}$. Report alongside it the loss penalty at $2\times$ off-optimum depth: if $\Delta\mathcal{L} < 0.01$ nats at 19B, the correct conclusion is that shape should be chosen by serving latency, and the loss-optimal law is not worth having.

## 9. Key References

- **[Foundational]** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Yoav Levine, Noam Wies, Or Sharir, Hofit Bata, Amnon Shashua. *Limits to Depth Efficiencies of Self-Attention.* NeurIPS, 2020.
- **[SOTA]** Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, Lucas Beyer. *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design.* NeurIPS, 2023. — arXiv:2305.13035
- **[SOTA]** Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou. *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks.* ICLR, 2024. — arXiv:2310.02244
- **[Empirical]** Yi Tay, Mostafa Dehghani, Jinfeng Rao, et al. *Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers.* ICLR, 2022. — arXiv:2109.10686
- **[Empirical]** Zechun Liu, Changsheng Zhao, Forrest Iandola, et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML, 2024. — arXiv:2402.14905
- **[Theory]** William Merrill, Ashish Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023.
- **[Theory]** Matus Telgarsky. *Benefits of Depth in Neural Networks.* COLT, 2016. — arXiv:1602.04485
- **[Empirical]** Jackson Petty, Sjoerd van Steenkiste, Ishita Dasgupta, Fei Sha, Dan Garrette, Tal Linzen. *The Impact of Depth on Compositional Generalization in Transformer Language Models.* NAACL, 2024.
- **[Related]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024.

## 10. Worked Example

Fix $N = 1.21$B non-embedding, i.e. $L d^2 = 1.006\times10^{8}$. Three iso-$N$ shapes:

| shape | $L$ | $d$ | $\rho = d/L$ | serial matmuls/token | est. decode latency (rel.) |
|---|---|---|---|---|---|
| wide | 12 | 2896 | 241 | 12 | 1.00 |
| middle | 24 | 2048 | 85 | 24 | 1.55 |
| deep | 48 | 1448 | 30 | 48 | 2.40 |

All three have identical $N$ and identical training FLOPs under $C \approx 6ND$. Suppose a clean Depth-$\mu$P sweep returns validation losses $2.184$ / $2.176$ / $2.181$ nats (deep-middle optimum, spread $0.008$ nats). Seed noise at this scale is $\pm 0.005$ nats, so the wide-vs-middle gap is about $1.6\sigma$ — the sweep does not separate them at three seeds.

Now convert to the quantity a deployer cares about. Chinchilla's loss exponent implies roughly $\mathcal{L} \propto N^{-0.34}$, so recovering $0.008$ nats by adding parameters needs $\Delta N/N \approx e^{0.008/2.18 \cdot (1/0.34)} - 1 \approx 1.1\%$. Choosing the "optimal" depth is worth about 1% of model size. Choosing the wide shape instead is worth 55% of decode latency.

The obstruction is now visible. The measurement is not merely expensive — it is *the wrong measurement*. Even executed perfectly, it resolves a $0.008$-nat effect that is dominated by a $1.55\times$ latency effect the objective never priced. Until the cost functional $\mathcal{J}(L,d)$ includes serial depth, the loss-optimal $\alpha$ can be estimated to arbitrary precision and still not tell anyone what shape to build.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*