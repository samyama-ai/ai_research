---
id: 24-multimodal/frozen-versus-tuned-vision-encoder
title: "Frozen versus Unfrozen Vision Encoders"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Frozen versus Unfrozen Vision Encoders

> **Topic:** Multimodal Models · **ID:** `24-multimodal/frozen-versus-tuned-vision-encoder` · **Status:** empirically-open

## 1. Problem Statement

A vision-language model (VLM) is usually assembled from three parts: a pretrained image encoder $f_\theta$ (CLIP/SigLIP-style ViT), a connector $g_\phi$ (linear projection, MLP, or resampler), and a pretrained language model $h_\psi$. During multimodal training the builder chooses which parameter blocks receive gradients. The recurring choice is whether $\theta$ is **frozen** (gradients blocked at the connector boundary) or **unfrozen** (fully or partially trained).

- **Measurement variant.** Given a fixed data mixture, fixed compute budget, and fixed evaluation suite, does unfreezing $\theta$ change downstream accuracy, and by how much, per unit of extra compute? This is the variant most papers claim to answer and mostly do not: the arms differ in learning rate, schedule, and data as well as in the freeze flag.
- **Method variant.** If unfreezing helps, what is the cheapest intervention that captures the gain — LoRA on $\theta$, unfreezing the last $k$ blocks, a lower encoder learning rate, weight-space interpolation back toward $\theta_0$, or simply a larger/higher-resolution frozen encoder?
- **Theory variant.** Under what conditions does gradient flow into a contrastively pretrained encoder improve the in-distribution objective while destroying features that the evaluation suite (or a later distribution shift) depends on? The linear-probing-then-fine-tuning analysis of Kumar et al. (ICLR 2022) gives one such condition; no VLM-specific version exists.

**Solved** means: a scaling law $\Delta(\text{unfreeze})$ as a function of encoder size, resolution, multimodal token count, and data composition, with an identified crossover point, reproduced by an independent group.

## 2. Formal Setting

Let $\mathcal{D} = \{(x_i, c_i, y_i)\}$ be multimodal training data (image $x$, context $c$, target token sequence $y$). The VLM defines
$$p_{\theta,\phi,\psi}(y \mid x, c) = \prod_t h_\psi\!\left(y_t \mid y_{<t}, c, g_\phi(f_\theta(x))\right),$$
trained with token cross-entropy $\mathcal{L} = -\mathbb{E}_{\mathcal{D}} \log p(y \mid x, c)$.

Define two arms sharing $(\mathcal{D}, \psi_0, \phi_0, \theta_0)$ and a token budget $N$:
- **Frozen:** $\arg\min_{\phi,\psi} \mathcal{L}$, with $\theta \equiv \theta_0$.
- **Unfrozen:** $\arg\min_{\theta,\phi,\psi} \mathcal{L}$, with encoder learning rate $\eta_v = \alpha \eta$, $\alpha \in (0,1]$.

**Quantities as measured.**
- *Gain:* $\Delta = A_{\text{unfrozen}} - A_{\text{frozen}}$, where $A$ is mean accuracy over a named suite, each benchmark scored by its own official metric. $\Delta$ is only interpretable if the suite is declared before the run; retro-selected suites are the main source of contradictory published $\Delta$.
- *Compute:* FLOPs $C \approx C_{\text{fwd}} + \beta C_{\text{bwd}}$; unfreezing adds encoder backward and optimizer state. For a ViT-L/14 at 336px inside a 7B LLM, the encoder is roughly $0.3$B of $7.3$B parameters but a large share of activation memory at high resolution. Report $\Delta$ per matched $C$, not per matched epoch.
- *Feature drift:* $d(\theta_0,\theta) = 1 - \mathbb{E}_x \cos\!\big(f_\theta(x), f_{\theta_0}(x)\big)$, and representational drift via CKA between penultimate features on a held-out image set.
- *Retained encoder skill:* zero-shot ImageNet top-1 of the perturbed tower re-paired with its original text tower, plus mean top-1 over {ImageNet-A, -R, -Sketch, ObjectNet}. This separates "the encoder got better for this LLM" from "the encoder got better".

**Assumptions, and which fail.** (i) *Same optimum reachable* — false; the arms differ in effective loss landscape, so a single shared LR is not a fair control. (ii) *Benchmarks measure perception* — false in part; MMMU, MME, and many VQA sets are substantially answerable from language priors, so $\Delta$ absorbs LLM effects. (iii) *Encoder pretraining data disjoint from evaluation* — false; CLIP-family pretraining corpora overlap common benchmarks. (iv) *Freezing preserves features* — true for $\theta$, false for the composed system: $g_\phi$ can rotate away useful subspaces.

## 3. State of the Art

**Established (controlled, ablated).**
- **LiT** (Zhai et al., CVPR 2022): locking a pretrained image tower and tuning only the text tower beats tuning both for zero-shot transfer, at ImageNet-scale contrastive training — the cleanest single-factor ablation in the literature, though for a contrastive, not generative, objective.
- **Prismatic VLMs** (Karamcheti et al., ICML 2024): a matched-budget sweep over VLM design axes at 7B scale reports that fine-tuning the vision backbone *degrades* performance across their suite, with the connector and data mixture mattering more.
- **MM1** (McKinzie et al., ECCV 2024): ablations at up to 30B rank image resolution and encoder pretraining data above connector architecture; encoder capacity and resolution dominate.

**Claimed but unablated, or benchmark-number-only.**
- **Cambrian-1** (Tong et al., NeurIPS 2024) reports that unlocking the vision encoder during instruction tuning helps, especially on vision-centric benchmarks and with sufficient vision-centric data. The confound is that the unlocked arm co-varies with data mixture and encoder ensemble.
- **Idefics2** (Laurençon et al., NeurIPS 2024) trains backbones with LoRA rather than full fine-tuning and reports full fine-tuning is unstable; the comparison is not run at matched compute across all axes.
- **VILA** (Lin et al., CVPR 2024) reports that unfreezing the *LLM* during interleaved pretraining is the decisive factor; encoder freezing is a secondary, lightly ablated axis.
- Frontier proprietary systems (GPT-4V/o-series, Gemini, Claude) publish no encoder-freezing ablations. Any claim about their choice is inference, not evidence.

**Theory SOTA.** Kumar et al. (ICLR 2022) prove, in an overparameterized linear setting, that full fine-tuning can distort pretrained features and underperform linear probing out-of-distribution; LP-FT (probe first, then fine-tune) is the mitigation. Wortsman et al. (CVPR 2022, WiSE-FT) show weight-space interpolation between $\theta_0$ and fine-tuned $\theta$ recovers robustness. Neither has been extended to a generative VLM with a frozen LLM downstream.

## 4. What Is Known

- Frozen encoders suffice for strong instruction-following VLMs: LLaVA-1.5 (Liu et al., CVPR 2024) reaches competitive results on 11 benchmarks with a frozen CLIP ViT-L/336px and a two-layer MLP connector, trained on ~1.2M examples on 8×A100 in about a day.
- Resolution and encoder quality move the number more than the freeze flag in reported sweeps: LLaVA-1.5's 224→336px change and the CLIP→SigLIP swap each produce multi-point gains on several suites; published freeze/unfreeze deltas are typically 1–3 points and sign-inconsistent across suites.
- Fine-tuning distorts features in the unimodal case, measurably: WiSE-FT reports OOD accuracy gains of up to ~8.7 points over standard fine-tuning on ImageNet distribution shifts while matching in-distribution accuracy, at ViT-L/CLIP scale. LP-FT reports ~10-point average OOD improvement over full fine-tuning across a suite of shift benchmarks.
- Frozen CLIP-family encoders have specific blind spots that a frozen pipeline cannot fix: on MMVP (Tong et al., CVPR 2024), many strong open VLMs score below the 25% random-chance floor on paired questions where CLIP embeddings are near-identical but images differ, against ~95% human accuracy. This is the strongest positive argument for unfreezing — or for encoder ensembling.
- Frozen-LLM designs (Frozen, Tsimpoukelli et al., NeurIPS 2021; Flamingo, Alayrac et al., NeurIPS 2022; BLIP-2, Li et al., ICML 2023) establish that a frozen backbone plus a trained bridge transfers at all — BLIP-2 matched Flamingo-80B zero-shot VQAv2 with ~54× fewer trainable parameters.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published study varies only the freeze flag at matched tokens, matched compute, and a pre-registered suite, across at least three encoder scales and two resolutions. The experiment is runnable today for well under 10k GPU-hours; nobody has published it.
- **Empirically open.** Whether the MMVP-style perception gap closes by unfreezing at all, or only by changing the encoder's pretraining objective/data. Current evidence conflates the two.
- **Methodologically blocked.** "Perception ability" of a VLM has no measurement isolated from language priors. Until a benchmark is validated as language-prior-free (blind-LLM baseline at chance), $\Delta$ cannot be attributed to the visual pathway.
- **Theoretically open.** No condition, even in a linear surrogate, that predicts when gradients through a connector into a contrastive encoder improve versus destroy the composed system. The LP-FT result covers a shared task head, not a frozen 7B decoder.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**, not compute.

- The freeze flag is never varied alone. Unfreezing forces a different learning rate, often a different warmup and precision setting, and in practice a different data mixture (teams add vision-centric data when they unlock the encoder). The published $\Delta$ is the sum of four interventions.
- $\Delta$ is small relative to seed and mixture variance. Instruction-tuning benchmarks move 0.5–1.5 points on seed alone; typical claimed deltas sit inside that band, and almost no paper reports multi-seed error bars.
- Non-identifiability: an unfrozen encoder and a higher-capacity connector can absorb the same degree of freedom. A gain attributed to unfreezing may be reachable by a deeper $g_\phi$ at a fraction of the cost — untested.
- The evaluation does not measure what it names. Score changes on MME/MMMU can come entirely from the LLM's answer formatting or priors while the visual pathway is unchanged; feature drift $d(\theta_0,\theta)$ is never reported alongside.

## 7. Current Research (as of 2026)

- **Partial unfreezing and low-rank encoder adaptation.** LoRA-on-encoder is the pragmatic default in open builds (Idefics2 line, several Qwen-VL/InternVL derivatives) — stability motivated, not ablated.
- **Encoder ensembling instead of tuning.** Cambrian-1 and Eagle-style spatial fusion of multiple frozen towers (CLIP + DINOv2 + SAM-family) target the same blind spots without gradient flow.
- **Native-resolution and encoder-free designs.** Qwen2-VL's dynamic resolution and encoder-free lines (Fuyu-style, early-fusion Chameleon) dissolve the question: with no separate tower there is nothing to freeze. Whether early-fusion models match encoder-based ones at fixed data scale is itself open. *(frontier — verify)*
- **Robustness-preserving fine-tuning transplanted to VLMs** — WiSE-FT-style interpolation of the tuned tower back toward $\theta_0$, evaluated on both VLM benchmarks and retained zero-shot accuracy. Rare in published VLM work. *(frontier — verify)*
- Groups most active on controlled VLM ablations: Stanford/TRI (Prismatic), NYU (Cambrian, MMVP), HuggingFace (Idefics), Apple (MM1), NVIDIA (VILA, Eagle).

## 8. Concrete Next Experiment

**Scale.** Fixed LLM (Qwen2.5-7B or Llama-3.1-8B, identical checkpoint), fixed connector (2-layer MLP), fixed 2-stage recipe: 1M alignment pairs, then 1.5M instruction examples. Encoders: SigLIP-SO400M at 384px and CLIP ViT-L/14 at 336px. Total ≈8 arms × 3 seeds ≈ 6–9k A100-hours.

**Arms.** (1) Frozen. (2) Unfrozen, $\alpha \in \{0.02, 0.1, 1.0\}$. (3) LoRA $r{=}64$ on encoder. (4) Last-6-blocks unfrozen. (5) WiSE-FT: arm (2) at $\alpha{=}0.1$, then $\theta \leftarrow (1-\lambda)\theta_0 + \lambda\theta$, $\lambda \in \{0.25,0.5,0.75\}$.

**Control arm.** Frozen encoder given the *same extra FLOPs* the unfrozen arms consume — spent on a deeper connector and additional instruction tokens. This is the arm that makes the result identifiable; its absence is why the current literature is inconclusive.

**Deciding number.** $\Delta$ = mean accuracy over a pre-registered suite (MMVP, BLINK, RealWorldQA, CV-Bench, TextVQA, MMStar), averaged over 3 seeds, versus the compute-matched frozen control. **Decision rule:** unfreezing wins iff $\Delta > 1.5$ points with the 95% seed interval excluding zero, *and* the retained zero-shot ImageNet top-1 of the perturbed tower drops by less than 5 points. Report $d(\theta_0,\theta)$ for every arm; a large $\Delta$ with $d < 0.02$ falsifies the causal story and indicates an optimizer artifact.

## 9. Key References

- **[Foundational]** M. Tsimpoukelli, J. Menick, S. Cabi, S. M. A. Eslami, O. Vinyals, F. Hill. *Multimodal Few-Shot Learning with Frozen Language Models.* NeurIPS, 2021. — arXiv:2106.13884
- **[Foundational]** X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, L. Beyer. *LiT: Zero-Shot Transfer with Locked-image text Tuning.* CVPR, 2022. — arXiv:2111.07991
- **[Foundational]** A. Kumar, A. Raghunathan, R. Jones, T. Ma, P. Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR, 2022. — arXiv:2202.10054
- **[Foundational]** M. Wortsman, G. Ilharco, J. W. Kim, M. Li, S. Kornblith, R. Roelofs, R. G. Lopes, H. Hajishirzi, A. Farhadi, H. Namkoong, L. Schmidt. *Robust fine-tuning of zero-shot models.* CVPR, 2022. — arXiv:2109.01903
- **[Foundational]** J.-B. Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS, 2022. — arXiv:2204.14198
- **[Foundational]** J. Li, D. Li, S. Savarese, S. Hoi. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* ICML, 2023. — arXiv:2301.12597
- **[SOTA]** S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, D. Sadigh. *Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models.* ICML, 2024. — arXiv:2402.07865
- **[SOTA]** S. Tong, E. Brown, P. Wu, S. Woo, et al. *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.* NeurIPS, 2024. — arXiv:2406.16860
- **[SOTA]** B. McKinzie et al. *MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training.* ECCV, 2024. — arXiv:2403.09611
- **[SOTA]** H. Laurençon, L. Tronchon, M. Cord, V. Sanh. *What matters when building vision-language models?* NeurIPS, 2024. — arXiv:2405.02246
- **[SOTA]** H. Liu, C. Li, Y. Li, Y. J. Lee. *Improved Baselines with Visual Instruction Tuning.* CVPR, 2024. — arXiv:2310.03744
- **[SOTA]** J. Lin, H. Yin, W. Ping, Y. Lu, P. Molchanov, A. Tao, H. Mao, J. Kautz, M. Shoeybi, S. Han. *VILA: On Pre-training for Visual Language Models.* CVPR, 2024. — arXiv:2312.07533
- **[Survey/Diagnostic]** S. Tong, Z. Liu, Y. Zhai, Y. Ma, Y. LeCun, S. Xie. *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs.* CVPR, 2024. — arXiv:2401.06209

## 10. Worked Example

Take LLaVA-1.5-7B: CLIP ViT-L/14-336 ($\approx0.30$B params) + MLP + Vicuna-7B. Instruction tuning is ~665k examples, ~576 image tokens each.

Frozen-arm cost per step is dominated by the LLM. Unfreezing the tower adds encoder backward (~2× its forward FLOPs) plus AdamW state for 0.30B params ($\approx 2.4$ GB in fp32 moments). Empirically this is roughly a **10–15% step-time increase** and a few GB of optimizer memory — cheap. So the naive read is "just unfreeze".

Now price the control. That same 10–15% buys ~90k additional instruction examples in the frozen arm, or a connector deepened from 2 to 6 layers. Published instruction-data scaling in this regime moves the mean suite score by roughly 0.5–1.5 points for a 15% data increase. Published freeze/unfreeze deltas are also 1–3 points, sign-varying by benchmark: Prismatic reports a *negative* delta on its suite; Cambrian-1 reports a positive one on a vision-centric suite. Both cannot be read as measuring the same quantity, because the suites and mixtures differ.

The obstruction becomes visible when you decompose one reported +2.0-point unfrozen gain:

| Component | Plausible contribution |
|---|---|
| Extra effective capacity (encoder params now trainable) | +0.5 to +1.5 |
| Co-varying vision-centric data added with the unlock | +0.5 to +1.5 |
| Retuned encoder LR acting as implicit regularization | −0.5 to +0.5 |
| Seed variance (single seed reported) | ±0.7 |
| Compute the frozen arm never received | +0.5 to +1.5 (owed to control) |

Every row is within the size of the headline effect, and no published run separates them. Add the second axis: the same unfrozen model's tower, re-paired with its original text encoder, typically loses zero-shot ImageNet accuracy — a cost that appears in no VLM benchmark table. A +2.0 VLM gain bought with a −10 zero-shot drop is a different trade than +2.0 for free, and current papers do not let you tell which one you got.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*