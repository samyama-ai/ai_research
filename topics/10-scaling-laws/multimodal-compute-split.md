---
id: 10-scaling-laws/multimodal-compute-split
title: "Multimodal Compute Split"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multimodal Compute Split

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/multimodal-compute-split` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training budget $C$ (FLOPs) for a model that must handle several modalities — text, images, audio, video — how should $C$ be divided?

Three distinct splits are in play, and papers routinely conflate them:

1. **Data split.** What fraction of tokens/samples comes from each modality and from each pairing (interleaved, captioned, unimodal)?
2. **Parameter split.** How many parameters go to a vision/audio encoder versus the shared backbone, and at what resolution / token count per image?
3. **Modality-conditional shape.** Given the split, what is the compute-optimal $(N, D)$ per component?

The problem has three variants of very different difficulty:

- **Measurement variant.** Define a scalar objective under which "optimal split" is even well posed. Cross-modal losses are in incomparable units (nats per BPE token vs nats per VQ image token), so the objective is a choice, not a measurement. *This is the blocking variant.*
- **Method variant.** Given a fixed scalarization, find $\alpha^\star(C)$ cheaply — from small-scale runs, without a full sweep at target scale.
- **Theory variant.** Prove that the multimodal loss surface admits a Chinchilla-style closed-form optimum, i.e. that $\alpha^\star(C)$ has a limit or a power-law trajectory rather than oscillating with scale.

Solving it means: a rule that, from runs costing $\le 10^{-2} C$, predicts the allocation minimizing a stated multimodal objective at $C$, within the noise floor of a held-out full-budget run.

## 2. Formal Setting

Modalities $m \in \mathcal{M}$, $|\mathcal{M}| = M$. Allocation vector $\alpha \in \Delta^{M-1}$ over the data budget and $\beta \in \Delta^{K-1}$ over $K$ architectural components (encoders, backbone, adapters).

**Compute.** Measured, not estimated: $C = 6 N_{\text{act}} D$ for dense transformers, where $N_{\text{act}}$ is active (non-embedding) parameters and $D$ is total tokens *after* modality tokenization. In practice, take $C$ from device-seconds × achieved FLOP/s (MFU-corrected), because encoder and backbone have different arithmetic intensity and the $6ND$ proxy drifts by 10–30% for vision towers with high-resolution patching.

**Tokens.** $D = \sum_m \alpha_m D$, where an image contributes $T_m$ tokens under the chosen tokenizer/patcher — $T_{\text{img}} \in [64, 4096]$ depending on resolution and pooling. $T_m$ is a *design choice inside the allocation*, which is why $\alpha$ in samples and $\alpha$ in tokens are not the same variable.

**Loss.** Per modality, measured as held-out negative log-likelihood in nats *per unit of the modality's own token grid*:
$$L_m(N, D, \alpha) = \mathbb{E}_{x \sim \mathcal{D}_m}\left[-\tfrac{1}{|x|}\log p_\theta(x)\right].$$

**Objective.** The decision predicate needs a scalarization $w \in \mathbb{R}^M_{\ge 0}$:
$$\alpha^\star(C) = \arg\min_{\alpha \in \Delta^{M-1}} \sum_m w_m\, L_m\big(N^\star(C,\alpha), D^\star(C,\alpha), \alpha\big) \quad \text{s.t.}\quad 6ND \le C.$$

**Interaction.** The multimodal analogue of the Chinchilla form adds a cross-modality term:
$$L_m(N, D, \alpha) = E_m + \frac{A_m}{N^{a_m}} + \frac{B_m}{(\alpha_m D)^{b_m}} + \underbrace{\textstyle\sum_{m' \ne m} \Gamma_{m m'}(N, D)\, \alpha_{m'}}_{\text{competition } (>0) \,/\, \text{synergy } (<0)}.$$
$\Gamma$ is what makes this problem not just $M$ independent scaling laws.

**Assumptions, with the violated ones flagged:**
- *Separable power law in $N$ and $D$ per modality* — approximately holds within a modality (Kaplan 2020; Hoffmann 2022); **violated** across modalities, since $\Gamma \ne 0$ is the empirical finding of Aghajanyan et al. (2023).
- *Losses comparable across modalities* — **violated**. Image NLL under a VQ tokenizer and text NLL under BPE differ by an arbitrary scale set by the tokenizer, so $w$ is unidentifiable from data.
- *Loss is monotone in downstream capability* — **violated** for vision-language: better image reconstruction NLL routinely does not improve VQA or zero-shot classification.
- *Single-epoch, non-repeated data* — **violated** for high-quality interleaved corpora, where repetition returns diminish (Muennighoff et al. 2023).

## 3. State of the Art

**Established (ablated, reproduced):**
- Compute-optimal *unimodal* $(N,D)$ scaling: $N \propto C^{0.5}$, $D \propto C^{0.5}$, ~20 tokens/param (Hoffmann et al., NeurIPS 2022). Reproduced widely.
- Compute-optimal *shape* for vision encoders: SoViT-400m/14 matches ViT-g/14 quality at roughly half the compute, from a scaling-law-derived width/depth/MLP schedule (Alabdulmohsin et al., NeurIPS 2023). Ablated.
- Contrastive image-text scaling exponents are **task- and dataset-dependent**: OpenCLIP on LAION-2B and OpenAI WIT-400M cross over depending on whether the metric is ImageNet zero-shot or retrieval (Cherti et al., CVPR 2023). Fully reproduced with open models and code.
- At fixed total parameters, moving parameters from the vision encoder to the language backbone helps more than the reverse (Laurençon et al., NeurIPS 2024, Idefics2 ablations).

**Claimed but not fully ablated:**
- Mixed-modal scaling laws with an explicit competition/synergy term, fitted across text, image, speech and code up to ~30B parameters (Aghajanyan et al., ICML 2023). The functional form is fitted, not derived; the transferability of the fitted $\Gamma$ to other tokenizers or architectures has not been independently reproduced.
- MM1's reported pre-training mixture (roughly equal weight on interleaved and caption data with a smaller pure-text share) and the finding that image resolution matters more than visual-token count (McKinzie et al., ECCV 2024). These are ablations at one scale family, not a scaling law.
- Native/early-fusion multimodal scaling laws claiming early fusion matches late fusion at matched compute and favours parameters over data (Shukor, Fini, El-Nouby et al., 2025). Single-lab result.

**Benchmark-number-only:** most VLM allocation claims (Chameleon 2024, Transfusion 2024) are reported as end-task scores at one budget, with no held-out prediction of the optimum at a larger budget.

## 4. What Is Known

- Multimodal generative loss follows power laws in compute across image, video, math and image↔text, with **domain-specific exponents and non-zero irreducible terms** (Henighan et al., 2020; models $10^5$–$10^{10}$ parameters).
- Competition between modalities is **scale-dependent**: at small $N$, adding a second modality raises per-modality loss; the penalty shrinks as $N$ grows (Aghajanyan et al., ICML 2023; 8M–30B parameters, up to ~100B tokens). This is the single most important known fact — it means small-scale sweeps systematically over-estimate the cost of mixing.
- Vision-encoder shape can be optimized by scaling law and transferred: ~2× compute saving at ImageNet-scale evaluation (Alabdulmohsin et al., 2023, ViT 400M–2B).
- Locking the image tower and training only text (LiT, Zhai et al., CVPR 2022) beats joint training on zero-shot classification at matched budget — a large, reproduced effect, and an existence proof that the *parameter* split interacts with which parts receive gradient.
- Data repetition: up to ~4 epochs is near-lossless relative to fresh tokens; beyond ~16 epochs returns approach zero (Muennighoff et al., NeurIPS 2023; up to 9B parameters, 900B tokens). Directly constrains how far scarce interleaved multimodal data can be stretched.

## 5. What Is Not Known

- **Methodologically blocked.** There is no principled $w$. Without one, "the optimal split" is a family of answers indexed by an unstated preference. No paper states $w$ explicitly; each implicitly fixes it via a downstream benchmark suite, and different suites give different $\alpha^\star$.
- **Empirically open.** Whether $\alpha^\star(C)$ converges, drifts monotonically, or is non-monotone in $C$. Aghajanyan's scale-dependent competition implies drift, but nobody has run a 3-point-in-$C$, 5-point-in-$\alpha$ grid spanning two decades of compute with a fixed tokenizer and fixed evaluation. The experiment is runnable today for well under $10^{23}$ FLOPs.
- **Empirically open.** Whether the vision-encoder/backbone parameter split $\beta^\star$ is invariant to $C$ (the Idefics2 result is one budget).
- **Theoretically open.** No proof that the interaction term $\Gamma_{mm'}$ admits a form making $\alpha^\star$ closed-form, nor any bound on how far a small-scale-fitted $\alpha^\star$ can be from the large-scale one. No multimodal analogue of the Chinchilla derivation exists.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the objective compounded by tokenizer-dependent units**. Per-token NLL in different modalities is measured on grids whose granularity is a free design parameter, so the aggregate loss can be made to favour any modality by re-tokenizing — no data can pin $w$ down. Two secondary obstructions:

- **Confounded measurement.** Changing $\alpha$ changes $D$ per modality, effective epochs, the batch composition, and often the learning-rate schedule shape. Reported "mixture effects" bundle all four.
- **Absent ground truth for transfer.** The quantity practitioners care about — downstream multimodal capability — is measured by benchmark suites that are known to be weakly coupled to generative loss, so the evaluation does not measure the thing the objective names.

Compute cost is real but secondary: a decisive grid is $\sim 10^{21}$–$10^{22}$ FLOPs, within reach of a mid-size academic cluster.

## 7. Current Research (as of 2026)

- **Native/early-fusion scaling laws.** Apple (Shukor, Fini, El-Nouby and colleagues) fitting joint laws for from-scratch multimodal models, including MoE variants where experts specialize by modality — which reframes the split as a routing question rather than a budget question. *(frontier — verify)*
- **Mixture-optimization transfer.** Data-mixing laws (Ye et al., 2024) and DoReMi-style reweighting (Xie et al., NeurIPS 2023) predict optimal mixtures from proxy runs; extending them across modalities rather than across text domains is the obvious open port. *(frontier — verify)*
- **Open reproduction.** LAION / open-CLIP-lineage groups continue to publish reproducible contrastive scaling curves, which are the only multimodal scaling results with fully open data and code.
- **Token-budget-aware VLMs.** Work on adaptive visual token counts moves $T_{\text{img}}$ from a fixed constant into the allocation variable, making $\alpha$ and $\beta$ non-separable. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does the compute-optimal text/image data split $\alpha^\star$ move with compute?

- **Scale.** Three compute budgets, one decade apart: $C \in \{3\times10^{19},\ 3\times10^{20},\ 3\times10^{21}\}$ FLOPs. At each, choose $(N,D)$ Chinchilla-optimally ($N \approx 0.4$B, 1.2B, 4B; $D \approx$ 12B, 42B, 130B tokens). Early fusion, one frozen VQ image tokenizer at fixed $T_{\text{img}} = 256$ across all runs — this holds the units constant, which is the point.
- **Grid.** $\alpha_{\text{img}} \in \{0.05, 0.15, 0.30, 0.50, 0.70\}$. 15 runs, 3 seeds at $\alpha=0.30$ for noise. Total $\approx 1.8\times10^{22}$ FLOPs — roughly 2–3 weeks on 256 H100s.
- **Control arm.** Two unimodal reference runs per budget ($\alpha=0$ and $\alpha=1$), giving $L_{\text{text}}^{\text{solo}}$ and $L_{\text{img}}^{\text{solo}}$. The competition term is then measured directly as $\Gamma$-per-modality $= L_m(\alpha) - L_m^{\text{solo}}$, and every mixed run is scored as *degradation relative to its own solo control*, which removes the unit problem.
- **The deciding number.** $\Delta = \alpha^\star(3{\times}10^{21}) - \alpha^\star(3{\times}10^{19})$, where $\alpha^\star$ minimizes the **unit-free** objective $\sum_m [L_m(\alpha)/L_m^{\text{solo}}]$. If $|\Delta| < 0.05$ (within seed noise, expected $\sigma \approx 0.02$), the split is scale-invariant and can be tuned once at small scale. If $|\Delta| > 0.10$ with consistent sign, small-scale mixture tuning is invalid and every published VLM mixture is fitted at the wrong budget.

## 9. Key References

- **[Foundational]** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Tom Henighan, Jared Kaplan, Mor Katz, et al. *Scaling Laws for Autoregressive Generative Modeling.* 2020. — arXiv:2010.14701
- **[Foundational]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Armen Aghajanyan, Lili Yu, Alexis Conneau, et al. *Scaling Laws for Generative Mixed-Modal Language Models.* ICML, 2023. — arXiv:2301.03728
- **[SOTA]** Mehdi Cherti, Romain Beaumont, Ross Wightman, et al. *Reproducible Scaling Laws for Contrastive Language-Image Learning.* CVPR, 2023. — arXiv:2212.07143
- **[SOTA]** Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, Lucas Beyer. *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design.* NeurIPS, 2023. — arXiv:2305.13035
- **[SOTA]** Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, et al. *MM1: Methods, Analysis and Insights from Multimodal LLM Pre-training.* ECCV, 2024. — arXiv:2403.09611
- **[SOTA]** Hugo Laurençon, Léo Tronchon, Matthieu Cord, Victor Sanh. *What Matters When Building Vision-Language Models?* NeurIPS, 2024. — arXiv:2405.02246
- **[SOTA]** Mustafa Shukor, Enrico Fini, Victor Guilherme Turrisi da Costa, Matthieu Cord, Joshua Susskind, Alaaeldin El-Nouby. *Scaling Laws for Native Multimodal Models.* Apple, 2025. (identifier omitted — verify before citing)
- **[Related]** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Related]** Sang Michael Xie, Hieu Pham, Xuanyi Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023. — arXiv:2305.10429
- **[Related]** Xiaohua Zhai, Xiao Wang, Basil Mustafa, et al. *LiT: Zero-Shot Transfer with Locked-image Text Tuning.* CVPR, 2022. — arXiv:2111.07991
- **[Survey]** Chameleon Team (FAIR at Meta). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* 2024. — arXiv:2405.09818

## 10. Worked Example

Budget $C = 10^{22}$ FLOPs. Chinchilla-optimal dense model: $N = 3$B, $D = 6\times10^{21}/(6\times3\times10^9) \approx 555$B tokens.

Split text/image with $\alpha_{\text{img}}$. Take realistic held-out losses: text at 2.5 nats/BPE-token, images at 4.5 nats/VQ-token with $T_{\text{img}} = 1024$, captions ~12 BPE tokens.

Naive aggregate objective, $w = (1,1)$, summing per-*sample* NLL:

| | text doc (512 tok) | image (1024 tok) |
|---|---|---|
| NLL per sample | $512 \times 2.5 = 1280$ nats | $1024 \times 4.5 = 4608$ nats |

A 1% relative improvement on images buys 46 nats/sample; a 1% improvement on text buys 12.8. The summed objective therefore pushes $\alpha^\star_{\text{img}} \to 1$ — not because images matter more, but because the VQ codebook emits 1024 tokens per image.

Now change nothing except the image tokenizer: pool to $T_{\text{img}} = 256$ at 5.2 nats/token. Image NLL per sample becomes $256 \times 5.2 = 1331$ nats, comparable to text. The same optimizer now returns $\alpha^\star_{\text{img}} \approx 0.4$ on the same data, the same model, the same compute.

**The obstruction, visible:** $\alpha^\star$ moved from ~1.0 to ~0.4 under a re-tokenization that changed no information content. The optimum is a function of an arbitrary design choice inside the measurement, so no amount of additional training resolves it. Section 8's fix — normalizing each modality's loss by its own unimodal control, $L_m(\alpha)/L_m^{\text{solo}}$ — cancels the tokenizer scale factor to first order and is the minimum requirement for the question to have an answer at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*