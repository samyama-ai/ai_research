---
id: 02-attention/cross-attention-necessity-multimodal
title: "Cross-Attention Necessity in Multimodal Fusion"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Attention Necessity in Multimodal Fusion

> **Topic:** Attention Mechanisms · **ID:** `02-attention/cross-attention-necessity-multimodal` · **Status:** empirically-open

## 1. Problem Statement

Multimodal models fuse a vision (or audio) stream into a language backbone in one of two families:

- **Cross-attention fusion (X):** language tokens attend to a separate, frozen-or-trained encoder's outputs through dedicated cross-attention blocks inserted into the backbone (Flamingo, BLIP-2 Q-Former, Llama-3.2-Vision).
- **Self-attention / early fusion (S):** the encoder's outputs are projected into the token embedding space, concatenated with text tokens, and processed by the ordinary decoder stack (LLaVA, Idefics2-FA, Chameleon).

The question: **is cross-attention ever necessary — i.e. does it buy accuracy, sample efficiency, or compute efficiency that self-attention over the concatenated sequence cannot match at equal budget?**

Three variants, of very different difficulty:

- **Theory variant.** Is there a task family where the X-parameterization is strictly more expressive or more learnable (better optimization geometry, lower sample complexity) than S at matched parameters? *Answer for pure expressivity is known and negative* (§4); the learnability question is open.
- **Method variant.** Given a compute budget $C$, a modality-token count $N_v$, and a backbone update policy (frozen / LoRA / full), which family wins? Currently answered only by scattered, single-seed ablations.
- **Measurement variant.** What is "equal budget" when X adds parameters inside the backbone and S adds sequence length? Without a matched-FLOPs-and-matched-parameters protocol, published comparisons are not comparable.

**Solved** would mean: a scaling-law-backed decision rule mapping $(C, N_v, \text{update policy}, \text{data mix})$ to the winning family, with the crossover located to within a factor of 2 in compute.

## 2. Formal Setting

Text tokens $x_{1:N_t} \in \mathbb{R}^{N_t \times d}$; modality tokens $z_{1:N_v} \in \mathbb{R}^{N_v \times d_v}$ from encoder $E_\phi$.

**Self-attention arm (S).** Project $\tilde z = W z$, $W \in \mathbb{R}^{d \times d_v}$, and run the backbone on $u = [\tilde z; x] \in \mathbb{R}^{(N_v+N_t)\times d}$:
$$\mathrm{Attn}(u) = \mathrm{softmax}\!\left(\frac{(uW_Q)(uW_K)^\top}{\sqrt{d_h}} + M\right) uW_V .$$

**Cross-attention arm (X).** Backbone runs on $x$ only; every $k$-th layer inserts
$$x \leftarrow x + \tanh(\alpha)\cdot \mathrm{softmax}\!\left(\frac{(xW_Q^c)(zW_K^c)^\top}{\sqrt{d_h}}\right) zW_V^c ,$$
the gated form of Flamingo ($\alpha$ initialized to 0).

**Measured quantities.**

- Parameters: $P_X = P_{\text{lm}} + \lceil L/k\rceil \cdot 4d^2$ (plus FFN if the block carries one); $P_S = P_{\text{lm}} + d\,d_v$. Match by shrinking $d$ or $L$ in the S arm, or by adding an equal-size adapter — state which.
- Compute: per-example forward FLOPs, counted, not estimated. $F_S \approx 2P_{\text{lm}}(N_v+N_t) + 4Ld(N_v+N_t)^2$; $F_X \approx 2P_{\text{lm}}N_t + 4LdN_t^2 + \lceil L/k\rceil\,4d\,N_tN_v$. X is subquadratic in $N_v$; S is quadratic.
- Performance: held-out log-likelihood on a fixed multimodal validation mix (nats/token), **not** downstream benchmark accuracy, plus a benchmark suite as secondary.
- Sample efficiency: tokens to reach a fixed loss $\ell^\*$.

**Decision predicate.** Fit $L(C) = A C^{-\beta} + L_\infty$ per arm. X is *necessary* at budget $C$ iff $L_X(C) < L_S(C) - \varepsilon$ with $\varepsilon$ above seed noise (measure it: $\ge 3$ seeds).

**Assumptions, and which are violated.**
1. *Equal-budget matching is possible* — violated in practice: nearly all published pairs differ in encoder, data, resolution, and tuning effort simultaneously.
2. *Loss is the right target* — violated: instruction-tuned benchmark accuracy and validation loss decorrelate after SFT.
3. *$N_v$ fixed* — violated: X arms routinely resample to 64 latents (Perceiver Resampler, Q-Former) while S arms carry 576–2880 tokens, so "architecture" and "token budget" are confounded.
4. *Backbone update policy held fixed* — the single largest known effect modifier (§4), and frequently not held fixed.

## 3. State of the Art

**Established (ablated within one codebase, matched data):**

- **Idefics2 / "What matters when building vision-language models?"** (Laurençon et al., NeurIPS D&B 2024) is the only published head-to-head with data, encoder, and LM held fixed. It reports that with **frozen** backbones cross-attention wins by a large margin (~13 points on their 4-benchmark average), and with backbones trained via LoRA the **fully-autoregressive (self-attention) arm reverses the result and wins by ~9 points**. Scale: 9B-class LM, single seed.
- **Attention Bottlenecks for Multimodal Fusion (MBT)** (Nagrani et al., NeurIPS 2021) establishes that restricting cross-modal flow to a handful of bottleneck tokens beats full pairwise attention on audio-visual classification at lower FLOPs (AudioSet mAP in the low-to-mid 40s; matched-backbone ablation).

**Claimed but unablated:**

- Flamingo (Alayrac et al., NeurIPS 2022) attributes its few-shot ability to gated cross-attention over a frozen LM, but never runs a matched self-attention arm at the same scale.
- BLIP-2 (Li et al., ICML 2023) claims Q-Former compute efficiency; the comparison is to differently-trained baselines, not to a matched linear-projection arm.
- LLaVA-1.5 (Liu et al., CVPR 2024) shows a two-layer MLP projector plus self-attention reaching then-SOTA on 11 benchmarks with 1.2M examples — a benchmark number, not an architecture ablation.

**Frontier, less settled:** native early-fusion models (Chameleon, FAIR 2024) and multimodal scaling-law work (Shukor et al., 2025) argue early fusion matches or beats late/cross fusion at scale, with a compute-optimal preference shifting toward early fusion as $C$ grows *(frontier — verify)*.

## 4. What Is Known

- **Expressivity containment (theorem-level, easy direction).** Self-attention over $[\tilde z; x]$ with a block mask that zeroes $z\!\to\!z$ and $x\!\to\!x$ contributions reproduces a cross-attention block exactly, given $d_v \le d$ and $W$ full rank. So X $\subseteq$ S as function classes at matched width; **any advantage of X is optimization- or efficiency-based, not representational.**
- **The frozen/unfrozen interaction is real and large.** ~13 points for X frozen, ~9 points for S with LoRA, same codebase (Idefics2, 9B).
- **Token-count reduction, not the attention pattern, carries much of X's efficiency.** MBT's bottleneck of 4 tokens recovers full-fusion accuracy; Perceiver/Perceiver IO (Jaegle et al., ICML 2021 / ICLR 2022) show 64–256 latents suffice for high-dimensional inputs.
- **Quadratic cost is the operative constraint.** At $N_t=512$, $N_v=2880$ (LLaVA-NeXT-style tiling), the S arm's attention FLOPs are $(3392/512)^2 \approx 44\times$ the text-only cost; the X arm's cross term grows linearly in $N_v$.
- **Cross-attention degrades more gracefully under missing modalities** in classification settings (Ma et al., "Are Multimodal Transformers Robust to Missing Modality?", CVPR 2022) — established at ViLT scale (~100M), not at LM scale.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published $L(C)$ scaling curve for X vs S with parameters, data, encoder, $N_v$, and update policy all matched, across $\ge 3$ compute decades. The experiment is runnable at $\le 10^{22}$ FLOPs; nobody has published it.
- **Empirically open.** Whether X's frozen-backbone advantage survives modern LoRA ranks and longer training, or is a low-data-regime artifact.
- **Theoretically open.** Whether the block-masked (X) parameterization has provably better conditioning or lower sample complexity than the dense (S) one for a nontrivial task family. No separation result either way.
- **Methodologically blocked.** "Equal budget" itself: X adds parameters, S adds sequence. There is no agreed matching protocol, so no two papers' numbers compose.

## 6. Why It Is Hard

**Confounded measurement, primarily.** In every published pair, at least three of {encoder, resolution, $N_v$, projector capacity, LM update policy, data mix, tuning budget} vary together with the fusion type. The Idefics2 result shows the sign of the effect flips with one of these confounds (frozen vs LoRA) — so any comparison that does not sweep it reports an artifact of its chosen policy, not a property of cross-attention.

**Second: the evaluation does not measure what it names.** VQA-style benchmark averages are dominated by instruction-tuning data and answer formatting; they move several points from SFT-mix changes alone, which is the same magnitude as the architecture effect being measured.

**Third: compute.** Locating a crossover needs $\ge 3$ decades of budget $\times$ 2 arms $\times$ 3 seeds $\times$ 2 update policies = 36 runs, each with a full multimodal data pipeline. That is a frontier-lab-scale ablation for a negative-result-shaped question.

## 7. Current Research (as of 2026)

- **Native early fusion at scale.** FAIR (Chameleon), and follow-on token-interleaved models; the implicit bet is that S wins asymptotically.
- **Multimodal scaling laws.** Apple and academic groups fitting separate $L(C)$ per fusion type; early reports favor early fusion at compute-optimal allocation *(frontier — verify)*.
- **Cross-attention revival for long context.** Llama-3.2-Vision and video models retain cross-attention specifically because $N_v$ grows with frames — the efficiency argument, not the accuracy argument.
- **Token compression as the real variable.** Q-Former, Perceiver Resampler, and pruning methods (visual-token dropping) suggest the fusion-type question partly dissolves into "how many modality tokens does the backbone need to see?"

## 8. Concrete Next Experiment

**Scale.** Two arms, LM widths giving 160M / 700M / 3B non-embedding parameters, trained on an identical 30B-token interleaved image-text mix, ViT-L/14 encoder shared and identical across arms.

**Matching protocol.** Fix $N_v = 256$ in *both* arms (resample the S arm's tokens with the same Perceiver Resampler the X arm uses, so token count is not a confound). Add to the S arm parallel adapters with exactly $P_X - P_S$ parameters, so total parameters match to $<1\%$. Report FLOPs measured, not estimated.

**Control arms.** (a) Text-only backbone, same budget — establishes the modality-free floor. (b) X arm with $\alpha$ frozen at 0 (cross-attention disabled) — establishes that the cross-attention path is actually used. (c) Both arms under two policies: encoder+LM frozen, and LoRA-$r$=64.

**Deciding number.** $\Delta = L_S(C) - L_X(C)$ in nats/token on a held-out interleaved validation set, at each of the three scales, under each policy, with $\ge 3$ seeds to get $\sigma$. The question resolves if $\mathrm{sign}(\Delta)$ is stable and $|\Delta| > 3\sigma$ at the largest scale under the LoRA policy — and, more informative, if $d\Delta/d\log C$ has a consistent sign, which locates or excludes a crossover. Cost estimate: ~$4\times10^{21}$ FLOPs total, i.e. a few thousand A100-days.

## 9. Key References

- **[Foundational]** Jaegle, Gimpel, Brock, Zisserman, Vinyals, Carreira. *Perceiver: General Perception with Iterative Attention.* ICML 2021. — arXiv:2103.03206
- **[Foundational]** Nagrani, Yang, Arnab, Jansen, Schmid, Sun. *Attention Bottlenecks for Multimodal Fusion.* NeurIPS 2021. — arXiv:2107.00135
- **[Foundational]** Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS 2022. — arXiv:2204.14198
- **[SOTA / key ablation]** Laurençon, Tronchon, Cord, Sanh. *What matters when building vision-language models?* NeurIPS Datasets & Benchmarks 2024. — arXiv:2405.02246
- **[SOTA]** Li, Li, Savarese, Hoi. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* ICML 2023. — arXiv:2301.12597
- **[SOTA]** Liu, Li, Li, Lee. *Improved Baselines with Visual Instruction Tuning.* CVPR 2024. — arXiv:2310.03744
- **[SOTA]** Chameleon Team (FAIR). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* 2024. — arXiv:2405.09818
- **[Robustness]** Ma, Xu, Wang, Wang, Zhang, Sun. *Are Multimodal Transformers Robust to Missing Modality?* CVPR 2022.
- **[Survey]** Xu, Zhu, Clifton. *Multimodal Learning with Transformers: A Survey.* IEEE TPAMI, 2023. — arXiv:2206.06488

## 10. Worked Example

Take a 7B decoder, $d = 4096$, $L = 32$, $N_t = 512$, and a CLIP ViT-L/14@336 encoder emitting $N_v = 576$ tokens.

**X arm.** Cross-attention every $k = 4$ layers: 8 blocks $\times\ 4d^2 = 8 \times 6.7\text{e}7 = 5.4\text{e}8$ added parameters (+7.7%). Cross term FLOPs: $8 \times 4 \times 4096 \times 512 \times 576 \approx 3.9\text{e}10$ per example.

**S arm.** Linear projector: $d\,d_v = 4096 \times 1024 \approx 4.2\text{e}6$ parameters (+0.06%). Backbone now runs on 1088 tokens instead of 512 — dense FLOPs rise by $2 \times 7\text{e}9 \times 576 \approx 8.1\text{e}12$, plus attention $4Ld(1088^2 - 512^2) \approx 4.7\text{e}11$.

So S costs about **8.6e12 extra FLOPs per example against X's 3.9e10 — a $220\times$ gap in marginal fusion cost** — while X costs 130× more added parameters.

**Where the obstruction shows.** Suppose you now train both to the same 30B tokens. If you match *parameters*, the S arm gets far more FLOPs per token and should win on loss — and the literature would call that "self-attention is better". If you match *FLOPs*, the S arm must shrink $N_v$ to ~64 or shrink $d$, and X wins — and the literature would call that "cross-attention is necessary". Both papers exist in spirit; neither is wrong; they answer different questions. Add the Idefics2 finding that the sign flips again between frozen and LoRA backbones, and the observed effect size (~9–13 points) is entirely inside the space spanned by the matching choices. **That is the obstruction: the outcome is a function of the normalization, and no published protocol fixes the normalization.** §8's design exists precisely to nail $N_v$, parameters, and FLOPs simultaneously, which is why it needs the resampler in *both* arms.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*