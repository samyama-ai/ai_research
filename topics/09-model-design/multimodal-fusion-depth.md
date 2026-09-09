---
id: 09-model-design/multimodal-fusion-depth
title: "Modality Fusion Depth in Multimodal Backbones"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Modality Fusion Depth in Multimodal Backbones

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/multimodal-fusion-depth` · **Status:** empirically-open

## 1. Problem Statement

A multimodal backbone must decide **where in depth** tokens from different modalities start to attend to each other. Options run from *early fusion* (concatenate at layer 0; one shared stack) through *mid fusion* (unimodal stacks for $\ell < \ell^\*$, joint attention after) to *late fusion* (separate towers, one dot product at the end, as in CLIP), with cross-attention adapters (Flamingo, BLIP-2 Q-Former) as a parameterized middle.

Three distinct questions get called "the fusion depth problem":

- **Measurement variant.** Given a trained model, at what depth does cross-modal information *actually* get used, as opposed to where the architecture permits it? A model with layer-0 concatenation may not mix modalities until layer 14.
- **Method variant.** Given a compute budget $C$, a token mixture, and a target task distribution, choose $\ell^\*$ (or a fusion schedule) that minimizes loss. Solving it means a predictive rule — $\ell^\*/L$ as a function of $C$, modality entropy, and data mixture — that transfers across scales without a sweep.
- **Theory variant.** Prove that a depth-$L$ transformer with fusion at $\ell^\*$ separates from one fusing at $\ell^\* \pm \Delta$ on some task family, i.e. an expressivity or sample-complexity gap, not a benchmark delta.

**Solved** means: the method variant has a scaling rule validated by held-out extrapolation to a scale not used to fit it, and the measurement variant has an estimator of realized fusion depth that is invariant to reparameterizations of the backbone.

## 2. Formal Setting

Let $x_a \in \mathcal{X}_a$ and $x_b \in \mathcal{X}_b$ be inputs from two modalities, tokenized to $n_a$ and $n_b$ tokens of width $d$. A backbone of $L$ blocks produces hidden states $h^{(\ell)} \in \mathbb{R}^{(n_a+n_b)\times d}$. A **fusion mask** $M^{(\ell)} \in \{0,1\}^{(n_a+n_b)^2}$ gates attention; the architectural fusion depth is

$$\ell^\* = \min\{\ell : M^{(\ell)}_{ij} = 1 \text{ for some } i \in a,\; j \in b\}, \qquad \rho = \ell^\*/L \in [0,1].$$

**Realized fusion depth** is the quantity that must be measured, not read off the config. Define per-layer cross-modal attention mass

$$A^{(\ell)} = \frac{1}{H n_a}\sum_{h=1}^{H}\sum_{i \in a}\sum_{j \in b} \alpha^{(\ell,h)}_{ij},$$

with $\alpha$ the post-softmax attention weights. $A^{(\ell)}$ is cheap but unreliable: attention weight is not attribution. The causal estimator is a **cross-modal ablation curve**. Let $\mathcal{L}$ be the task loss and $\mathcal{L}_{\neg b}^{(\ell)}$ the loss when modality-$b$ keys/values are replaced by a modality-marginal baseline (mean over a held-out batch) at layers $\ge \ell$ only. Then

$$\delta(\ell) = \mathcal{L}_{\neg b}^{(\ell)} - \mathcal{L}, \qquad \hat{\ell}_\varepsilon = \min\{\ell : \delta(\ell) < \varepsilon\},$$

the shallowest layer above which modality $b$ can be discarded at cost under $\varepsilon$ nats. $\hat{\ell}_\varepsilon$ is the *usage* depth; $\ell^\*$ is the *permission* depth. $\hat{\ell}_\varepsilon \ge \ell^\*$ always, and the gap is the object of interest.

Compute: a fused layer over $n_a+n_b$ tokens costs $\Theta((n_a+n_b)^2 d + (n_a+n_b)d^2)$ against $\Theta((n_a^2+n_b^2)d + (n_a+n_b)d^2)$ split. Total training FLOPs $C \approx 6ND$ with $N$ parameters, $D$ tokens; late fusion buys back the $2n_an_b d$ cross term per layer, so an $\ell^\*$ sweep is **not** compute-matched by default. Any honest comparison fixes $C$, not $L$ or $N$.

**Assumptions, and where they break.**

1. *Both modalities are equally informative for the objective.* Violated: on most VQA-style data, text alone recovers a large fraction of the loss, so $\delta(\ell)$ for the image branch is small at every $\ell$.
2. *Tokenizers are quality-matched.* Violated: a frozen CLIP ViT encoder has already done 24 layers of unimodal work before "layer 0" of the backbone, so $\rho$ is not comparable across systems.
3. *The mean-token ablation baseline is off-manifold-safe.* Violated in practice; mean-ablation and resample-ablation give different $\hat{\ell}_\varepsilon$.
4. *One $\ell^\*$ per model.* Violated by mixture-of-modality-experts and by Flamingo's every-$k$-blocks interleaving, where fusion is a schedule, not a scalar.

## 3. State of the Art

**Empirical SOTA.** Two contradictory regimes coexist, and both are established as benchmark numbers rather than as mechanisms.

- *Mid fusion wins for audio–video.* Nagrani et al., **Attention Bottlenecks for Multimodal Fusion** (NeurIPS 2021), sweep $\ell^\*$ over a 12-layer ViT-Base on AudioSet, Epic-Kitchens and VGGSound, and find an interior optimum near $\ell^\*=8$, beating both $\ell^\*=0$ and $\ell^\*=12$. This is the cleanest published sweep of the actual variable. It is one architecture family, one token budget, one scale.
- *Early fusion wins at compute-optimal for native multimodal LMs.* Shukor et al., **Scaling Laws for Native Multimodal Models** (2025), fit scaling laws for early- versus late-fusion models trained from scratch and report early fusion equal or better at matched compute, with late fusion requiring a larger parameter-to-data ratio. Established as fitted scaling laws; the mechanism is not ablated.
- *Adapter depth is data-regime dependent.* Laurençon et al., **What matters when building vision-language models?** (NeurIPS 2024, Idefics2), compare Flamingo-style cross-attention against fully-autoregressive early fusion under matched pretrained backbones, and report a **reversal**: cross-attention is better with frozen backbones, fully-autoregressive is better once the LM is unfrozen. Single-seed, one scale — the direction is credible, the magnitude is not.

**Established but often mis-stated.** Bugliarello et al., **Multimodal Pretraining Unmasked** (TACL 2021), re-implement single-stream and dual-stream V&L BERTs in one codebase and find the published architecture gaps largely dissolve under matched data, hyperparameters and embeddings. The correct reading: **most reported fusion-depth effects are confounded**, not that fusion depth is inert.

**Theory SOTA.** There is no separation theorem in $\ell^\*$. The nearest formal results are generic transformer depth-separation results and the Platonic Representation Hypothesis (Huh et al., ICML 2024), which argues unimodal representations converge with scale — an argument that fusion depth should *matter less* as models grow, not a proof.

## 4. What Is Known

- **An interior optimum exists in at least one setting.** MBT's layer sweep on AudioSet-500k with ViT-Base ($L=12$) puts the best $\ell^\*$ at 8, i.e. $\rho \approx 0.67$, with a gain of roughly one to two mAP points over both extremes. Scale: ~86M-parameter backbone, ~500k clips.
- **Bottlenecked fusion is nearly free.** Restricting cross-modal flow to 4 latent bottleneck tokens matches or beats full pairwise cross-attention on AudioSet while cutting fusion FLOPs; measured at ViT-Base/ViT-Large.
- **Cross-modal influence is asymmetric.** Frank et al., *Vision-and-Language or Vision-for-Language?* (EMNLP 2021), ablate one modality's input and find text representations are far more perturbed by removing images than the converse, across several V&L transformers at BERT-base scale. This is direct evidence that $\hat{\ell}_\varepsilon$ differs by direction.
- **Architecture effects are dominated by data and optimization at small scale** (Bugliarello 2021; Hendricks et al., TACL 2021), measured at ~110M parameters and ~3–10M image–text pairs.
- **Early fusion is trainable at scale.** Chameleon (Meta, 2024) trains a fully token-level early-fusion mixed-modal model at 7B/34B, and reports needing norm-reordering and QK-norm to stay stable — evidence that $\ell^\*=0$ has an optimization cost, not just an accuracy profile.

## 5. What Is Not Known

- **Empirically open (the main gap).** No compute-matched sweep of $\rho$ across two or more orders of magnitude of $C$ exists in public. The single-scale sweeps (MBT, $\sim$0.1B) and the scaling-law studies (early vs late only, no interior $\ell^\*$) do not intersect. The experiment is runnable today for under a few hundred GPU-days; nobody has published it.
- **Methodologically blocked.** "Realized fusion depth" has no agreed estimator. Attention mass, ablation curves, and probing accuracy rank layers differently, and none is invariant to frozen-encoder depth or to residual-stream rescaling. Until $\hat{\ell}_\varepsilon$ is standardized, cross-paper claims about "where fusion happens" are not comparable.
- **Theoretically open.** Whether there is a task family with a provable sample-complexity or expressivity separation between fusion at $\ell$ and at $\ell'$ under matched parameter count. No proof either way.
- **Open with a stated hypothesis.** Whether $\rho^\*$ shrinks toward 0 with scale (the Platonic/native-multimodal prediction) or stays interior. Both are consistent with current data.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an unmatched compute axis**. Changing $\ell^\*$ changes, simultaneously: (i) attention FLOPs, because the cross term $2n_an_bd$ appears only in fused layers; (ii) effective parameter sharing, because unimodal layers below $\ell^\*$ are typically duplicated per modality, so late fusion at fixed $L$ has more parameters; (iii) optimization conditioning, since early fusion mixes two token distributions with different norm statistics into one residual stream and destabilizes training. A naive sweep therefore varies four things and attributes the result to one.

Second obstruction: **the evaluation does not measure the thing it names.** VQA and captioning benchmarks are answerable at high accuracy from language priors alone, so a fusion-depth delta of 1–2 points sits inside the range that unimodal shortcut exploitation covers. A benchmark where $\delta(\ell)$ for the *visual* branch is large by construction does not exist at scale.

## 7. Current Research (as of 2026)

- **Native mixed-modal pretraining** — Meta (Chameleon line), Apple (native multimodal scaling laws), and the open Idefics/SmolVLM line at Hugging Face are the groups with matched-compute infrastructure. Direction: push $\ell^\* \to 0$ and absorb the instability with normalization tricks.
- **Learned fusion schedules** — routing or gating that makes $\ell^\*$ a per-token, per-layer decision rather than a hyperparameter, typically via modality-specific experts *(frontier — verify)*.
- **Mechanistic localization of fusion** — applying activation patching to find the layer band where visual information enters the text stream in LLaVA-class models *(frontier — verify)*; several 2025 preprints report a narrow mid-stack band, but with different patching baselines and no shared protocol.
- **Bottleneck/latent fusion at long context** — Perceiver-style latent arrays as the fusion interface, motivated by video token counts rather than by accuracy.

## 8. Concrete Next Experiment

**Question:** does the compute-optimal fusion fraction $\rho^\*$ move with compute?

- **Scale.** Train from scratch a decoder-only backbone at four compute budgets $C \in \{3\times10^{19}, 3\times10^{20}, 3\times10^{21}, 3\times10^{22}\}$ FLOPs (roughly 150M to 3B parameters at Chinchilla-optimal tokens), on a fixed interleaved image–text corpus with a *fixed, from-scratch* patch embedder (no pretrained CLIP tower — assumption 2 above must not be violated). At each budget, sweep $\rho \in \{0, 0.25, 0.5, 0.75, 1.0\}$.
- **Compute matching.** Hold total training FLOPs constant per budget, not $L$ or $N$: shrink $d$ or $D$ for the fused-heavy arms to pay for the extra cross-attention term. Report realized FLOPs per arm.
- **Control arm.** $\rho = 1.0$ (pure late fusion, contrastive head only) at identical $C$, plus a **text-only** arm at identical $C$ to bound the language-prior shortcut. Any fusion-depth effect smaller than the text-only-to-$\rho{=}1.0$ gap is not interpretable.
- **Deciding number.** $\rho^\*(C)$ — the argmin of held-out multimodal loss — fitted as $\rho^\*(C) = a + b\log_{10} C$. **The decision is $b$.** If $|b| < 0.05$ per decade with a bootstrap CI excluding 0.1, fusion depth is scale-invariant and can be set once. If $b \le -0.1$ per decade, $\rho^\* \to 0$ and early fusion is the correct asymptotic default. Report $\hat{\ell}_{0.01}$ (resample-ablation) alongside, to test whether realized depth tracks architectural depth.

Cost estimate: 20 runs, dominated by the top budget; roughly 200–400 A100-days.

## 9. Key References

- **[Foundational]** Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun. *Attention Bottlenecks for Multimodal Fusion.* NeurIPS, 2021. — arXiv:2107.00135
- **[Foundational]** Jean-Baptiste Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS, 2022. — arXiv:2204.14198
- **[Foundational]** Alec Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. — arXiv:2103.00020
- **[Methodology]** Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki, Desmond Elliott. *Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs.* TACL, 2021. — arXiv:2011.15124
- **[Methodology]** Stella Frank, Emanuele Bugliarello, Desmond Elliott. *Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers.* EMNLP, 2021. — arXiv:2109.04448
- **[Methodology]** Lisa Anne Hendricks, John Mellor, Rosalia Schneider, Jean-Baptiste Alayrac, Aida Nematzadeh. *Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers.* TACL, 2021.
- **[SOTA]** Mustafa Shukor, Enrico Fini, Victor Guilherme Turrisi da Costa, Matthieu Cord, Joshua Susskind, Alaaeldin El-Nouby. *Scaling Laws for Native Multimodal Models.* 2025. — arXiv:2504.07951
- **[SOTA]** Hugo Laurençon, Léo Tronchon, Matthieu Cord, Victor Sanh. *What matters when building vision-language models?* NeurIPS, 2024. — arXiv:2405.02246
- **[SOTA]** Chameleon Team (Meta AI). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* 2024. — arXiv:2405.09818
- **[SOTA]** Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* ICML, 2023. — arXiv:2301.12597
- **[Theory-adjacent]** Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola. *The Platonic Representation Hypothesis.* ICML, 2024. — arXiv:2405.07987
- **[Survey]** Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency. *Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions.* ACM Computing Surveys, 2024.
- **[Survey]** Tadas Baltrušaitis, Chaitanya Ahuja, Louis-Philippe Morency. *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE TPAMI, 2019.

## 10. Worked Example

Take a $L=24$, $d=1024$ backbone, $n_a = 576$ image tokens (a $24\times24$ patch grid) and $n_b = 512$ text tokens.

Per-layer attention FLOPs scale with the number of query–key pairs. Split: $576^2 + 512^2 = 594{,}$ thousand pairs (0.594M). Fused: $(576+512)^2 = 1.184$M pairs. So each *fused* layer costs about **2.0×** the attention of a split layer. With $\rho = 0$ (fuse everywhere) versus $\rho = 0.5$, the extra cost is 12 layers $\times$ 0.59M pairs $\times$ $d$ $\times$ 2 (QK and AV) $\approx$ 14.5 GFLOP per forward token-batch — roughly a 15–20% total training-FLOP increase at this shape, once the $\Theta(nd^2)$ projection terms (identical in both arms) are included.

Now the trap. A typical paper runs both arms for the same number of steps at the same $N$ and reports early fusion ahead by 1.4 points on a VQA average. Three things are true at once:

1. The early-fusion arm consumed ~18% more FLOPs. Give the late-fusion arm the same FLOPs — 18% more tokens — and Chinchilla-style loss scaling ($L \propto D^{-0.28}$ empirically for text) predicts a loss reduction that maps to roughly 0.5–1.0 points on the same average. Half the gap or more is compute, not architecture.
2. Run the text-only control on the same benchmark. On VQAv2-style data, language-prior-only models reach well above chance; if the text-only arm is within 3 points of both fusion arms, a 1.4-point gap is inside the shortcut band.
3. Measure $\hat{\ell}_{0.01}$ on the $\rho=0$ arm by resample-ablating image keys/values from layer $\ell$ upward. If the curve shows $\delta(\ell) < 0.01$ nats for all $\ell \ge 14$, then the model that was *permitted* to fuse at layer 0 in fact stops using vision above layer 14 — the architectural knob and the realized behaviour disagree by 14 layers.

The obstruction is visible in the arithmetic: the effect size being argued over (1.4 points) is smaller than the compute confound (~0.5–1.0 points) plus the shortcut band (~3 points), and the mechanism claim ("early fusion mixes modalities earlier") is contradicted by the ablation curve on the model's own weights. Until the sweep in §8 is run compute-matched with a text-only floor, the literature cannot distinguish "early fusion is better" from "early fusion got more FLOPs and the benchmark did not need vision."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*