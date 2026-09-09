---
id: 02-attention/head-count-vs-head-dimension
title: "Optimal Head Count versus Head Dimension Tradeoff"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Head Count versus Head Dimension Tradeoff

> **Topic:** Attention Mechanisms · **ID:** `02-attention/head-count-vs-head-dimension` · **Status:** empirically-open

## 1. Problem Statement

A multi-head attention layer of width $d$ is conventionally split into $h$ heads of dimension $d_h$ with $h \cdot d_h = d$. The split is a free hyperparameter: it changes no parameter count and almost no FLOPs. Yet practice has drifted to a narrow band ($d_h \in \{64, 128\}$) with no published derivation.

**Input.** A width $d$, depth $L$, context length $n$, token budget $T$, and a hardware target.
**Output.** The pair $(h, d_h)$ — and, separately, the key/value head count $h_{kv} \le h$.
**Objective.** Minimise validation loss at fixed training FLOPs and fixed inference memory bandwidth.

Three variants, with different difficulty:

- **Measurement.** Does $\partial \mathcal{L} / \partial \log d_h$ at fixed $d$ exceed the noise floor from seed and learning-rate variation? Nobody has published this with error bars above 1B parameters.
- **Method.** Is there a rule $d_h^\star = f(d, L, n, T)$ that transfers across scale, in the sense that $\mu$P transfers learning rate?
- **Theory.** Is the loss gap between shapes an expressivity gap (a function class that narrow heads cannot represent), an optimisation gap, or neither?

Solving it means: a rule that predicts the loss-minimising $d_h$ within the measured noise band, validated by extrapolation to a scale it was not fit on.

## 2. Formal Setting

For head $i$ at layer $\ell$, with input $X \in \mathbb{R}^{n \times d}$ and projections $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d \times d_h}$, $W_O^i \in \mathbb{R}^{d_h \times d}$:

$$A^i = \mathrm{softmax}\!\left(\frac{X W_Q^i (X W_K^i)^\top}{\sqrt{d_h}} + M\right), \qquad \mathrm{MHA}(X) = \sum_{i=1}^{h} A^i X W_V^i W_O^i .$$

**Quantities as measured.**

- *Parameters.* $P_{\text{attn}} = 4 d^2 L$ when $h d_h = d$ and $h_{kv} = h$. Invariant in $h$. With grouped queries, $P_{\text{attn}} = 2d^2 L (1 + h_{kv}/h)$.
- *Training FLOPs.* $C \approx 6 P T + 12 L n d T$ (Kaplan et al., 2020). The second term is the attention score/output matmul: $2n^2 d$ per layer forward, independent of the split. Only the softmax and the $1/\sqrt{d_h}$ scale differ.
- *Decode memory traffic.* Bytes of KV cache per token $= 2 L h_{kv} d_h b$, $b$ = bytes/element. Under $h_{kv} = h$ and $h d_h = d$ this is $2Ldb$ — **also invariant in the split**. It varies only through $h_{kv}$.
- *Logit rank.* Per head, $\mathrm{rank}(X W_Q^i W_K^{i\top} X^\top) \le \min(n, d_h)$. This is the object the low-rank bottleneck argument acts on.
- *Loss noise floor.* $\sigma_{\text{seed}}$, the standard deviation of final validation loss over $\ge 3$ seeds at fixed shape, plus $\sigma_{\text{lr}}$, the loss increase from a half-octave learning-rate misspecification. A shape difference is real only if it exceeds $\sqrt{\sigma_{\text{seed}}^2 + \sigma_{\text{lr}}^2}$.

**Assumptions, and how they break.**

1. *$h d_h = d$.* Violated by design in T5 (relative-position variants use $d_h$ decoupled from $d/h$) and by the fixed-$d_h$ prescription of Bhojanapalli et al. Once decoupled, parameter count changes and the comparison is no longer FLOP-matched.
2. *Learning rate is shape-independent.* False. The $1/\sqrt{d_h}$ scale and the fan-in of $W_O$ both change with the split, so the optimal LR moves. Most published head-count ablations reuse one LR — this is the dominant confound.
3. *Kernels are shape-neutral.* False. FlashAttention's tiling is tuned for $d_h \in \{64,128\}$; $d_h=32$ wastes tensor-core tiles and $d_h=256$ pressures SRAM. Measured wall-clock differences are partly kernel artifacts, not model properties.
4. *Heads are independent.* False — Cordonnier et al. (2020) show learned key/query subspaces overlap heavily across heads.

## 3. State of the Art

**Established (ablated, reproduced).**

- Vaswani et al. (NeurIPS 2017), Table 3: at fixed $d=512$, both extremes are worse than the middle. $h=1,d_h=512$: dev PPL 5.29 / BLEU 24.9. $h=8,d_h=64$: 4.92 / 25.8. $h=16,d_h=32$: 4.91 / 25.8. $h=32,d_h=16$: 5.01 / 25.4. The interior is flat: the 8-vs-16 gap (0.01 PPL) is within seed noise, never reported with error bars.
- Bhojanapalli et al. (ICML 2020) prove a representational limit: when $d_h < n$, there exist context matrices no head can produce, and fixing $d_h$ independent of $h$ measurably improves BERT-scale models.
- Michel et al. (NeurIPS 2019): most heads in a *trained* 16-head WMT/BERT model can be pruned at test time with little BLEU/accuracy loss; many layers tolerate a single head. This is a post-hoc redundancy result, not a training-time shape result — the frequent conflation of the two is a category error.
- Ainslie et al. (EMNLP 2023, GQA) and Shazeer (2019, MQA): reducing $h_{kv}$ cuts KV cache linearly. GQA-8 uptrained from T5-XXL recovers near-MHA quality at close to MQA decode speed.

**Claimed but unablated.** That $d_h = 128$ is optimal for large models. Llama-2/3, Mistral and most open models use it; none publishes a matched sweep at their scale. It is a convention inherited from GPT-3 ($d_h = 128$) and hardware convenience.

**Benchmark-number-only.** DeepSeek-V2's MLA (2024) reports strong quality at ~5–13% of MHA KV cache, but the comparison bundles latent compression, decoupled RoPE, and shape changes; the head-geometry contribution alone is not isolated.

**Theory SOTA is separate and weaker.** Sanford, Hsu & Telgarsky (NeurIPS 2023) exhibit tasks (sparse averaging) where multi-head width provably matters, but the separations are in $d$ and $L$, not in the $h$–$d_h$ split at fixed $d$.

## 4. What Is Known

- The split is free in parameters, near-free in FLOPs, and exactly free in KV bytes when $h_{kv}=h$. Scale: any $h d_h = d$ transformer.
- Loss is a shallow U in $\log d_h$ with a wide flat floor. Measured at $d=512$, 65M params, WMT14 (Vaswani 2017): total spread across $h \in \{1,32\}$ is 0.38 PPL, but only 0.10 PPL across $h \in \{4,16\}$.
- Kaplan et al. (2020) found loss depends *weakly* on shape: varying aspect ratio over a wide range moves loss by a few percent, at 768–1.5B params on WebText2. Head dimension was not swept independently.
- Tay et al. (EMNLP Findings 2022) show architecture rankings are not scale-invariant: a shape that wins at 10M can lose at 1B. This directly undermines transferring small-scale head sweeps.
- Rank collapse (Dong, Cordonnier & Loukas, ICML 2021): pure attention converges doubly-exponentially to rank-1 with depth; residuals and MLPs counteract it. Head width interacts with this but the interaction is not quantified.
- $d_h$ must satisfy $d_h \ge$ RoPE's paired-dimension requirement and is bounded below in practice by numerical stability of $1/\sqrt{d_h}$ scaling at low precision.

## 5. What Is Not Known

- **Empirically open (primary).** No published sweep of $d_h \in \{32,64,128,256\}$ at fixed $d$, ≥7B parameters, ≥200B tokens, with per-shape LR tuning and ≥3 seeds. The experiment is entirely runnable — it is roughly $4 \times 3 \times$ one 7B pretraining run — and nobody has published it.
- **Empirically open (secondary).** Whether $d_h^\star$ grows with context length $n$, as the $d_h < n$ bottleneck argument predicts. Long-context models did not increase $d_h$ when $n$ went from 2K to 128K; either the theory is loose or the models are leaving loss on the table.
- **Theoretically open.** Whether there is a separation in *learnable* function class between $(h, d_h)$ and $(2h, d_h/2)$ at equal $d$ and equal parameters. Existing separations are about $d$ and $L$.
- **Methodologically blocked.** "Head redundancy" has no agreed measure. Prunability, attention-map similarity, and subspace overlap disagree with each other, so "16 heads are redundant" is not a well-posed claim.

## 6. Why It Is Hard

**Confounded measurement, with an effect size below the confound.** The quantity of interest is a loss difference of order 0.005–0.02 nats. Three confounds are each at least that large:

1. Optimal learning rate shifts with the split (the $1/\sqrt{d_h}$ scale and $W_O$ fan-in both move). Reusing one LR across shapes produces a difference that is a tuning artifact.
2. Kernel efficiency differs by shape, so any "iso-wall-clock" comparison compares model *and* kernel.
3. Seed variance at 1B+ is comparable to the effect.

Add non-transferability (Tay et al.): the cheap version of the experiment answers a different question than the expensive one. So the honest experiment costs several 7B-scale pretraining runs to resolve a sub-1% loss difference — and that cost, not conceptual difficulty, is why it is unrun.

## 7. Current Research (as of 2026)

- **KV-geometry, not head-geometry.** The field has largely redirected the question: GQA (Google), MLA (DeepSeek), and cross-layer KV sharing optimise $h_{kv}$ and the cache latent, treating $d_h$ as fixed at 128. This is a rational response to the fact that $h_{kv}$ has a large measurable effect and the split does not.
- **Scaling-law-shaped architecture search.** Fitting loss surfaces over shape rather than picking one shape; the natural home for this problem, still mostly applied to depth/width. *(frontier — verify)*
- **Head-dimension decoupling in long context.** Whether $d_h$ should grow with $n$ is being revisited as contexts pass 1M tokens. *(frontier — verify)*
- **Mechanistic accounts.** Circuit-level work (induction heads, QK/OV decomposition) suggests some circuits need a minimum per-head rank, which would give the U-curve a mechanism rather than a fit. Not yet connected to pretraining loss.

## 8. Concrete Next Experiment

**Scale.** $d = 4096$, $L = 32$ (≈7B params), 200B tokens, $n = 8192$, identical data order.

**Arms.** $d_h \in \{32, 64, 128, 256\}$ with $h = d/d_h \in \{128, 64, 32, 16\}$, so parameters and score-matmul FLOPs are identical across arms. Keep $h_{kv} = h$ to hold KV bytes fixed at $2Ldb = 512$ KiB/token — this isolates the split from the cache question.

**Control arm.** $d_h = 128$ (the convention), run with **3 seeds** and a **5-point LR sweep** (half-octave spacing). Every other arm gets the same 5-point LR sweep; report each arm at *its own* LR minimum. Without per-arm LR tuning the experiment answers nothing.

**Deciding number.** $\Delta \mathcal{L} = \min_{\text{lr}} \mathcal{L}(d_h) - \min_{\text{lr}} \mathcal{L}(128)$, in nats, against the control's seed standard deviation $\sigma_{\text{seed}}$. **If $|\Delta \mathcal{L}| < 2\sigma_{\text{seed}}$ for all $d_h \in \{64,128,256\}$, the split is loss-irrelevant over an octave in each direction and should be chosen purely by kernel throughput.** If instead $\Delta \mathcal{L}(256) < -2\sigma_{\text{seed}}$, the field is under-sizing heads at long context and the bottleneck argument is live.

Cost: ~20 runs, but 16 are the LR sweep and can be truncated at 20B tokens (LR ranking is stable early), so ≈ 4 full + 16 short runs.

## 9. Key References

- **[Foundational]** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. *Attention Is All You Need.* NeurIPS 2017. — arXiv:1706.03762
- **[Foundational]** Bhojanapalli, Yun, Rawat, Reddi, Kumar. *Low-Rank Bottleneck in Multi-head Attention Models.* ICML 2020. — arXiv:2002.07028
- **[SOTA]** Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[SOTA]** Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019. — arXiv:1905.10650
- Voita, Talbot, Moiseev, Sennrich, Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL 2019. — arXiv:1905.09418
- Cordonnier, Loukas, Jaggi. *Multi-Head Attention: Collaborate Instead of Concatenate.* 2020. — arXiv:2006.16362
- Dong, Cordonnier, Loukas. *Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.* ICML 2021. — arXiv:2103.03404
- Sanford, Hsu, Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023. — arXiv:2306.02896
- Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Survey]** Tay, Dehghani, Abnar, Chung, Fedus, Rao, Narang, Tran, Yogatama, Metzler. *Scaling Laws vs Model Architectures: How Does Inductive Bias Influence Scaling?* Findings of EMNLP 2023. — arXiv:2207.10551
- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024. — arXiv:2405.04434

## 10. Worked Example

Take $d = 4096$, $L = 32$, $n = 8192$, bf16 ($b=2$).

**Costs are identical across the split.**

```
h=32, d_h=128:  attn params 4d²L = 2.15e9   KV/token = 2·32·32·128·2 = 512 KiB
h=64, d_h=64 :  attn params 4d²L = 2.15e9   KV/token = 2·32·64· 64·2 = 512 KiB
h=16, d_h=256:  attn params 4d²L = 2.15e9   KV/token = 2·32·16·256·2 = 512 KiB
score matmul FLOPs/layer/token fwd = 2·n·d = 6.7e7   (all three arms)
```

**What theory predicts.** Per head, the attention logit matrix has rank $\le \min(n, d_h)$. With $n = 8192$: $d_h = 64$ caps each head at rank 64, i.e. $64/8192 = 0.8\%$ of the available rank; $d_h = 256$ gives 3.1%. Bhojanapalli's bottleneck condition $d_h < n$ is violated by a factor of 32–128 in *every* arm. The theorem therefore predicts a deficiency for all three and orders them $256 \succ 128 \succ 64$ — but gives no magnitude.

**What measurement can resolve.** At 7B/200B tokens, validation loss is ≈1.9 nats and seed standard deviation is ≈0.004 nats (derived from published multi-seed pretraining spreads; assumed). A half-octave LR miss costs ≈0.01–0.02 nats — **two to five times the seed noise**. So an untuned sweep can manufacture a 0.02-nat "head-dimension effect" that is entirely learning rate.

**The obstruction, made visible.** Everything that is easy to measure — parameters, FLOPs, KV bytes — is exactly invariant to the split. The only thing that varies is a loss difference plausibly of order 0.005 nats, sitting underneath a 0.02-nat tuning confound. The experiment is not hard to design; it is hard to make the signal exceed the noise without spending several 7B-scale runs on learning-rate control alone. That is why the field standardised on $d_h = 128$ by convention and moved its optimisation effort to $h_{kv}$, where the effect on KV bytes is a factor of 8, not a fraction of a percent.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*