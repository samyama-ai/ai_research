---
id: 07-embeddings/modality-gap-vision-language
title: "The Modality Gap in Joint Vision-Language Embedding Spaces"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# The Modality Gap in Joint Vision-Language Embedding Spaces

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/modality-gap-vision-language` · **Status:** open

## 1. Problem Statement

Contrastively trained vision-language models (CLIP and successors) are advertised as producing a *shared* embedding space. They do not. Image embeddings and text embeddings occupy two disjoint, nearly linearly separable regions of the hypersphere, separated by a near-constant offset vector. A linear probe distinguishes modality from the embedding alone at essentially 100% accuracy, even for a matched image-caption pair whose cosine similarity is the highest in the batch.

Three variants, with different difficulty:

- **Measurement.** Define a gap statistic that is invariant to the symmetries of the InfoNCE optimum (global rotation, per-modality scaling under temperature reparameterization) and that predicts downstream failure. Open: current statistics are not invariant and do not predict.
- **Method.** Train an encoder pair whose embeddings are modality-indistinguishable *without* degrading retrieval and zero-shot transfer. Partially achieved; no method has both closed the gap and matched a compute-matched CLIP baseline at $\geq$400M pairs.
- **Theory.** Characterize the set of global minimizers of the multimodal InfoNCE objective at finite temperature $\tau$ and finite $d$, and state whether a nonzero gap is (a) a global optimum, (b) an optimization artifact of initialization, or (c) forced by an information asymmetry between modalities. No proof either way at realistic $d$, $\tau$, and data.

Solving it means: a theorem separating (a)/(b)/(c), plus a measurement that changes when the gap changes and moves in step with a downstream number.

## 2. Formal Setting

Encoders $f_\theta:\mathcal{X}\to\mathbb{R}^d$ (image), $g_\phi:\mathcal{T}\to\mathbb{R}^d$ (text), each followed by $\ell_2$ normalization onto $S^{d-1}$. Write $u_i=f_\theta(x_i)$, $v_i=g_\phi(t_i)$ for paired data $(x_i,t_i)\sim\mathcal{D}$, batch size $B$, temperature $\tau$. The training objective is symmetric InfoNCE:

$$\mathcal{L}=-\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{e^{u_i^\top v_i/\tau}}{\sum_j e^{u_i^\top v_j/\tau}}+\log\frac{e^{u_i^\top v_i/\tau}}{\sum_j e^{u_j^\top v_i/\tau}}\right].$$

**Quantities, as measured.**

- **Gap vector.** $\Delta=\bar u-\bar v$ with $\bar u=\frac{1}{N}\sum_i u_i$, $\bar v=\frac{1}{N}\sum_i v_i$, computed on a held-out set (MS-COCO 5k val is the de facto standard). **Gap magnitude** $\gamma=\|\Delta\|_2$. Note $\gamma$ is *not* scale-free: it shrinks when either modality's embeddings spread out, so it conflates offset with intra-modal variance.
- **Normalized gap.** $\tilde\gamma=\gamma/\sqrt{\tfrac12(\sigma_I^2+\sigma_T^2)}$ where $\sigma_I^2=\frac1N\sum_i\|u_i-\bar u\|_2^2$. This is a two-sample effect size; it is the statistic most papers should report and most do not.
- **Modality separability.** $\mathrm{AUC}_{\text{mod}}$ of a logistic probe trained to predict modality from the raw embedding, and $\mathrm{AUC}^{\perp}_{\text{mod}}$ of the same probe after projecting out $\hat\Delta$. The difference isolates how much of the gap is a rank-1 offset.
- **Similarity gap.** $s_{\text{cross}}=\mathbb{E}_i[u_i^\top v_i]$ versus $s_{\text{intra}}=\mathbb{E}_{i\neq j}[u_i^\top u_j]$. In CLIP, $s_{\text{intra}}$ typically *exceeds* $s_{\text{cross}}$ — a matched caption is farther from its image than a random other image is.
- **Alignment/uniformity** (Wang & Isola, ICML 2020): $\mathcal{L}_{\text{align}}=\mathbb{E}\|u_i-v_i\|^2$, $\mathcal{L}_{\text{unif}}=\log\mathbb{E}_{i\neq j}e^{-2\|z_i-z_j\|^2}$ over the pooled set.

**Assumptions, and which are violated.**

1. *Pairs are one-to-one and semantically complete.* Violated: a caption carries far less information than its image; many images share a caption. This asymmetry is the leading candidate cause of the gap.
2. *The batch negatives approximate the marginal.* Violated at $B=32{,}768$ with $\sim$400M–5B pair corpora; false negatives (semantically identical, labeled negative) are common.
3. *The gap is a rank-1 offset.* Approximately true, not exactly: removing $\hat\Delta$ leaves residual modality-separable structure.
4. *$\tau$ is a nuisance parameter.* Violated: $\tau$ is learned, and the geometry of the optimum depends on it; comparing $\gamma$ across models with different learned $\tau$ compares different objectives.

## 3. State of the Art

**Established (reproduced independently).**
- The gap exists at random initialization, before any training, as a consequence of the *cone effect* — deep nonlinear encoders map inputs into a narrow cone whose axis depends on the random seed (Liang et al., NeurIPS 2022). Two independently initialized encoders yield two different cones.
- Contrastive training with a small $\tau$ preserves rather than closes the gap; the model reaches a local basin in which the offset persists (Liang et al., 2022).
- The gap is dominated by a single direction. Zhang, Sui & Yeung-Levy (ICLR 2024, "Connect, Collapse, Corrupt") show the geometry is well modeled as a constant offset plus modality-specific noise, and exploit it: subtract the mean offset, add Gaussian noise, and train cross-modal decoders on *text only* that work on images at test time. This is the strongest evidence the offset is largely a nuisance affine term.

**Claimed but not fully ablated.**
- **Information imbalance as the cause.** Schrodi et al. (ICLR 2025) argue the gap and the well-known object bias share one trigger: images carry more information than captions, and the contrastive objective cannot equalize them. The causal claim rests on synthetic caption-richness manipulations, not on a compute-matched large-scale rerun.
- **The gap is a contrastive-loss artifact, not a modality artifact.** Fahim, Murphy & Fyshe (2024) show that adding explicit uniformity terms closes the gap and reports downstream gains; scale is well below CLIP-400M.
- **Mitigations.** Geodesic multimodal mixup (Oh et al., NeurIPS 2023), latent modality structures (Jiang et al., CVPR 2023), AlignCLIP-style shared-encoder variants (Eslami & de Melo, 2024/2025), Gramian multi-modal alignment (Cicchetti et al., ICLR 2025). Each reports gap reduction plus retrieval gains.

**Benchmark-number-only.** Nearly all reported "closing the gap improves X" results are single-seed, single-scale (typically CC3M/CC12M, ViT-B) retrieval or ImageNet zero-shot deltas of 1–3 points. No mitigation has been shown to hold at LAION-2B scale against a compute-matched baseline. Treat those as benchmark numbers, not established effects.

## 4. What Is Known

- **Magnitude.** For pretrained CLIP ViT-B/32 on MS-COCO, the Euclidean distance between modality centroids is $\gamma\approx0.8$ on the unit sphere (Liang et al., NeurIPS 2022) — an appreciable fraction of the maximum $2$. The same paper found a nonzero gap in every one of 3 pretrained multimodal models and 100+ randomly initialized encoder pairs it tested.
- **Manually shifting embeddings along $\Delta$** changes zero-shot accuracy and measured fairness metrics non-monotonically: small shifts can *improve* both, so $\gamma=0$ is not the optimum of any downstream metric (Liang et al., 2022). Measured on CLIP ViT-B/32, ImageNet and CelebA-scale evaluations.
- **Modality is linearly decodable.** A linear probe separates image from text embeddings at $\approx$100% on standard CLIP checkpoints; embeddings are visibly two clusters under PCA.
- **Offset removal transfers.** C3 (ICLR 2024) reaches competitive image-captioning and audio-captioning by training on text alone after mean-subtraction plus noise injection — evidence the residual after removing $\Delta$ is usable cross-modally.
- **Temperature matters.** Low learned $\tau$ (CLIP converges near $\tau\approx0.01$) sharpens the loss and, empirically, preserves the gap; SigLIP's sigmoid loss (Zhai et al., ICCV 2023) changes the negative-sampling geometry but was not proposed as, and has not been shown to be, a gap fix.

## 5. What Is Not Known

- **Theoretically open.** Whether a nonzero gap is a global minimizer of symmetric InfoNCE at finite $d$, finite $\tau$, and non-degenerate data. Existing analyses either take $d\to\infty$, assume perfectly aligned pair distributions, or analyze the loss only at initialization. No proof either way.
- **Theoretically open.** Whether an information asymmetry $H(X)\gg H(T\mid X)$ *forces* $\gamma>0$ at the optimum, or merely makes it optimization-favored.
- **Empirically open.** Whether any gap-closing method survives at $\geq$400M pairs against a compute-matched baseline. Every mitigation to date is $\leq$CC12M, ViT-B, often one seed. The experiment is runnable — it costs a CLIP pretraining run — and has not been run.
- **Methodologically blocked.** The measurement itself. $\gamma$ is not invariant to learned $\tau$ or to intra-modal variance, so cross-model comparisons of "gap size" in the literature are not comparing the same quantity. No agreed statistic has been shown to correlate with any downstream metric across models.

## 6. Why It Is Hard

**Confounded measurement compounded by non-identifiability.** The gap statistic $\gamma$ moves for at least three unrelated reasons: a genuine centroid offset, a change in intra-modal spread, and a change in learned $\tau$. Any intervention that closes the gap (mixup, uniformity regularizers, shared encoders) also changes spread and temperature. So a reported $\gamma$ reduction is never attributable to the offset alone, and the downstream delta is never attributable to the gap.

Underneath that: the InfoNCE optimum is invariant to global rotation and, jointly with a $\tau$ rescaling, to a family of geometric deformations. Two networks with the same loss can have very different $\gamma$. The quantity is not identified by the objective.

The blocker is *not* compute in the first instance — it is that if you spend the compute, you still cannot say what you measured.

## 7. Current Research (as of 2026)

- **Causal-mechanism line.** Schrodi et al. (Brox group, Freiburg) — information imbalance as the shared trigger for the gap and object bias; follow-on work manipulating caption density. *(frontier — verify current status.)*
- **Exploit-don't-close line.** Yeung-Levy's group (Stanford) — treat the gap as a known affine nuisance (C3) and use uni-modal data for cross-modal tasks. Fastest-moving direction because it does not require retraining.
- **Objective redesign.** Gramian/volume-based alignment for $n>2$ modalities (Cicchetti et al., ICLR 2025); sigmoid and pairwise losses following SigLIP. Whether these close the gap is largely unreported.
- **Relative/absolute representation.** Moschella et al. (ICLR 2023) relative representations — encode by similarity to shared anchors, which is invariant to the offset by construction. Under-tested at scale for VLMs. *(frontier — verify.)*
- **Interpretability angle.** Whether the gap direction is a usable steering axis (e.g. for retrieval calibration). *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Is the gap causally responsible for any downstream loss, or is it an affine nuisance?

**Scale.** Four OpenCLIP ViT-B/16 runs on LAION-400M, 12.8B samples seen, $B=32{,}768$ — roughly 400 A100-days each. Fix seed across arms.

**Arms.**
1. **Control:** standard CLIP, learned $\tau$.
2. **Gap-closed:** add penalty $\lambda\|\bar u_B-\bar v_B\|_2^2$ on batch centroids, $\lambda$ tuned so $\tilde\gamma<0.1$ at convergence.
3. **Post-hoc offset removal:** control checkpoint, subtract $\hat\Delta$ estimated on 100k held-out pairs, renormalize. No retraining.
4. **Placebo:** subtract a random unit vector of the same norm as $\hat\Delta$. This arm controls for the renormalization itself.

**Deciding number.** COCO 5k text$\to$image Recall@1, arm 3 minus arm 1. If post-hoc offset removal changes R@1 by $<0.5$ points (and arm 4 matches arm 3 within noise) while $\tilde\gamma$ drops by $>5\times$, the gap is an affine nuisance and the mitigation literature is optimizing an uninformative statistic. If arm 2 beats arm 1 by $>1.0$ R@1 with three seeds and non-overlapping intervals, the gap is causal and worth training against. Report $\tilde\gamma$, $\mathrm{AUC}^{\perp}_{\text{mod}}$, and learned $\tau$ for every arm — without $\tau$ the comparison is void.

## 9. Key References

- **[Foundational]** Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Zou. *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS, 2022. — arXiv:2203.02053
- **[Foundational]** Alec Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. — arXiv:2103.00020
- **[SOTA]** Yuhui Zhang, Elaine Sui, Serena Yeung-Levy. *Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data.* ICLR, 2024.
- **[SOTA]** Simon Schrodi, David T. Hoffmann, Max Argus, Volker Fischer, Thomas Brox. *Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models.* ICLR, 2025.
- **[Theory]** Tongzhou Wang, Phillip Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML, 2020.
- **[Method]** Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. *Sigmoid Loss for Language Image Pre-Training.* ICCV, 2023.
- **[Method]** Luca Moschella et al. *Relative Representations Enable Zero-Shot Latent Space Communication.* ICLR, 2023.
- **[Method]** Qian Jiang, Changyou Chen, Han Zhao, Liqun Chen, Qing Ping, Son Dinh Tran, Yi Xu, Belinda Zeng, Trishul Chilimbi. *Understanding and Constructing Latent Modality Structures in Multi-modal Representation Learning.* CVPR, 2023.
- **[Related]** Peter Fahim, Brian Murphy, Alona Fyshe. *It's Not a Modality Gap: Characterizing and Addressing the Contrastive Gap.* Preprint, 2024. (Identifier uncertain — cite by title.)

## 10. Worked Example

Take CLIP ViT-B/32 and 5,000 MS-COCO validation pairs. Measured values (order of magnitude, consistent with Liang et al., 2022):

```
gamma  = ||mean(u) - mean(v)||_2        ~ 0.82
sigma_I = mean ||u_i - mean(u)||        ~ 0.42
sigma_T = mean ||v_i - mean(v)||        ~ 0.53
tilde_gamma = 0.82 / sqrt((0.42^2+0.53^2)/2)  ~ 1.72
s_cross = E[u_i . v_i]                  ~ 0.31
s_intra = E[u_i . u_j], i != j          ~ 0.53
```

Read the last two lines. A caption and its own image sit at cosine $0.31$; two *unrelated* images sit at $0.53$. If you ranked all 10,000 embeddings by similarity to one image, every other image comes before the correct caption. Retrieval still works only because it compares within a single modality's column of the similarity matrix — the offset is common to all candidates and cancels in the argmax.

Now the obstruction. Subtract $\hat\Delta$ from every image embedding and renormalize. $\gamma$ collapses to $\approx 0.05$; $\tilde\gamma$ to $\approx 0.10$ — a $17\times$ reduction on the headline statistic. Recompute COCO text$\to$image R@1: it moves by well under a point, because the softmax over a fixed text query is invariant to any constant vector added to every image embedding, up to the second-order distortion from renormalization.

So the statistic dropped $17\times$ and the task number did not move. Every paper reporting "our method reduces the modality gap and improves retrieval" must rule out that the two facts are unrelated — that the retrieval gain came from the regularizer's effect on spread or on learned $\tau$, not from the offset. None currently does. That is the blocked measurement, and it is why arm 4 of §8 — the placebo direction — is the load-bearing control, not a formality.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*