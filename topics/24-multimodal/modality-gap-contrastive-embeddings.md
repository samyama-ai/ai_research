---
id: 24-multimodal/modality-gap-contrastive-embeddings
title: "Modality Gap in Contrastive Embedding Spaces"
topic: 24-multimodal
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Modality Gap in Contrastive Embedding Spaces

> **Topic:** Multimodal Models · **ID:** `24-multimodal/modality-gap-contrastive-embeddings` · **Status:** partially-solved

## 1. Problem Statement

Train an image encoder and a text encoder to a shared $d$-dimensional sphere with a contrastive loss (CLIP-style). The two modalities do not interleave. Image embeddings occupy one narrow cone, text embeddings another, and a linear classifier separates them at near-100% accuracy — even though the loss only ever rewards *relative* similarity of matched pairs. This is the modality gap.

Three distinct questions get called "the modality gap":

- **Measurement.** What scalar functional of the joint embedding distribution is "the gap", and does it track anything downstream? The default — Euclidean distance between modality centroids — is one choice among many, and different choices disagree on whether a given intervention helped.
- **Method.** Can we train or post-process encoders so that the gap shrinks *and* retrieval / zero-shot / cross-modal-generation quality does not fall? Partially yes; the pareto frontier is not established.
- **Theory.** Is the gap a defect (an optimization artifact that a better objective removes), or a *feature* — a geometrically necessary consequence of contrastive learning on many-to-one, information-imbalanced pairs? These predict opposite research programs.

Solving it means: a gap measure that is invariant to the transformations the loss is invariant to, plus a proof or refutation that zero gap under that measure is compatible with optimal contrastive loss.

## 2. Formal Setting

Encoders $f_\theta:\mathcal{X}\to\mathbb{R}^d$, $g_\phi:\mathcal{T}\to\mathbb{R}^d$, outputs $L^2$-normalized to $S^{d-1}$. Paired data $(x_i,t_i)\sim\mathcal{D}$, batch size $B$, learned temperature $\tau$. The loss is symmetric InfoNCE:

$$\mathcal{L}=-\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{e^{\langle f_i,g_i\rangle/\tau}}{\sum_j e^{\langle f_i,g_j\rangle/\tau}}+\log\frac{e^{\langle f_i,g_i\rangle/\tau}}{\sum_j e^{\langle f_j,g_i\rangle/\tau}}\right].$$

**Gap vector**, as measured: encode a held-out set of $n$ pairs, $L^2$-normalize each embedding, then

$$\Delta=\frac{1}{n}\sum_i f(x_i)-\frac{1}{n}\sum_i g(t_i),\qquad \text{gap}=\|\Delta\|_2\in[0,2].$$

**Cone width**, as measured: mean pairwise cosine within a modality, $\rho_f=\binom{n}{2}^{-1}\sum_{i<j}\langle f_i,f_j\rangle$. Note $\|\frac{1}{n}\sum_i f_i\|^2\approx\rho_f$ for large $n$, so a large $\|\Delta\|$ *requires* a narrow cone; the two statistics are not independent.

**Separability**, as measured: 5-fold cross-validated linear SVM accuracy on labels $\{$image, text$\}$. Report this alongside $\|\Delta\|$ — they dissociate.

**Alignment / uniformity** (Wang & Isola, ICML 2020): $\mathcal{A}=\mathbb{E}\|f_i-g_i\|^2$ and $\mathcal{U}=\log\mathbb{E}_{i\neq j}e^{-2\|z_i-z_j\|^2}$ over the pooled cloud.

**Invariance that breaks the measure.** InfoNCE is invariant to any orthogonal $Q$ applied to *both* encoders, and — critically — to no translation, because of normalization. But it *is* invariant to adding any constant vector $c$ to $f$ pre-normalization only in the limit $\|c\|\to0$. In practice, $\|\Delta\|$ can be driven to near-zero by a rigid rotation of one modality's cone onto the other with the loss essentially unchanged (Liang et al., 2022) — so $\|\Delta\|$ is *not* a function of the loss surface alone.

**Assumptions known to be violated.**
1. *Uniformity on $S^{d-1}$.* False: participation ratio of the embedding covariance is typically tens, not $d=512$ — the cone effect plus dimensional collapse (Jing et al., ICLR 2022).
2. *One-to-one pairing.* False: a caption maps to a vast set of compatible images. Mutual information between $x$ and $t$ is far below $H(x)$; Schrodi et al. (ICLR 2025) make this *information imbalance* the causal driver.
3. *In-batch negatives are true negatives.* False at scale — false negatives are common in web corpora.
4. *$\tau$ fixed.* False: it is learned, and CLIP's converges near $0.01$, which sharpens the loss and stiffens the gap.

## 3. State of the Art

**Established (reproduced, ablated).**
- The gap exists at initialization, before any training. Random encoders already place the two modalities in disjoint narrow cones (Liang et al., NeurIPS 2022). This is architectural, not learned.
- Contrastive training *preserves* rather than creates the gap: the trajectory from gapped to non-gapped configurations passes through higher loss, so it is a local optimum under the standard objective and temperature (Liang et al., 2022).
- Post-hoc translation along $\Delta$ changes downstream metrics — both up and down, dataset-dependent — establishing the gap is not inert (Liang et al., 2022).
- Cross-modal transfer works *despite* the gap: text-only training of a classifier that is then applied to image embeddings recovers most accuracy (Zhang et al., ICLR 2023). The gap is largely a near-constant offset plus noise.

**Claimed but unablated / benchmark-only.**
- That closing the gap improves retrieval. Fahim et al. (2024) report gains from adding an explicit uniformity + alignment term ("contrastive gap" framing) on MS-COCO/Flickr retrieval, but the comparison is confounded with extra loss terms and tuning budget, and has not been reproduced at CLIP-scale pretraining.
- Geodesic multi-modal mixup (Oh et al., NeurIPS 2023) reports robustness and retrieval gains from mixing across modalities on the hypersphere; ablations are at fine-tuning scale, not pretraining scale.
- SigLIP's sigmoid loss (Zhai et al., ICCV 2023) removes the batch-wise softmax and beats CLIP at matched compute, but *no published ablation isolates its effect on $\|\Delta\|$* — the gap claim is folklore.

**Theory SOTA** is Liang et al.'s local-optimum argument plus Schrodi et al.'s information-imbalance account. Neither is a theorem about the optimum of population InfoNCE with a general encoder class.

## 4. What Is Known

- CLIP ViT-B/32 on MS-COCO validation: $\|\Delta\|\approx0.82$ on unit-norm embeddings; the modality label is linearly separable at ~100% (Liang et al., NeurIPS 2022, $n\approx5{,}000$ pairs).
- The same paper measures gaps across 3 architectures and 6 checkpoints; every one is nonzero, and random-initialized (untrained) encoders show the same qualitative cone separation.
- Shifting all embeddings by $\lambda\Delta$ over $\lambda\in[-1,1]$ moves zero-shot accuracy by roughly $\pm 1$–$2$ points on CIFAR-10/CIFAR-100-scale evaluations, with the optimum often at $\lambda\neq0$ — i.e. neither zero gap nor the trained gap is optimal.
- C3 (Zhang, Sui, Yeung-Levy, ICLR 2024): after subtracting the mean gap and adding Gaussian noise, a *text-only*-trained captioning decoder driven by image embeddings reaches within a few points of paired-supervised captioning on MS-COCO — evidence the residual gap is dominated by a constant translation.
- Schrodi et al. (ICLR 2025): reducing information imbalance between the paired views shrinks the gap; object bias and gap move together, both traced to the same trigger. Measured at ViT-B scale on LAION-subset training.
- Temperature is the strongest single knob: larger $\tau$ yields smaller $\|\Delta\|$ and worse zero-shot accuracy in the standard regime — a confound in every "we closed the gap" claim.

## 5. What Is Not Known

- **Theoretically open.** Whether the population InfoNCE optimum over an unconstrained encoder class admits a zero-gap solution at all, for many-to-one pairings with $I(X;T)<H(X)$. No proof either way. The conjecture that nonzero gap is *necessary* under information imbalance is stated but unproven.
- **Theoretically open.** Whether $\|\Delta\|>0$ costs anything in the achievable loss, or is a free direction (a flat manifold) that gradient descent lands on arbitrarily.
- **Empirically open.** Does a gap-penalizing pretraining run at $\geq400$M pairs and $\geq10^{21}$ FLOPs beat matched-compute CLIP on ImageNet zero-shot? Every existing gap-closing result is fine-tuning or $\leq15$M-pair scale. Runnable today for roughly $10$–$50$k USD; nobody has published it.
- **Methodologically blocked.** There is no agreed gap measure invariant to the symmetries of the loss. $\|\Delta\|$, SVM separability, Wasserstein-2 between modality clouds, and CKA give different orderings of the same checkpoints. Until one is fixed, "we reduced the gap by 40%" is not a comparable claim.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability.** Every intervention that shrinks $\|\Delta\|$ also changes cone width, temperature, or effective dimension — and $\|\Delta\|^2\approx\rho_f+\rho_g-2\langle\bar f,\bar g\rangle$ makes it *algebraically* coupled to cone width. So a "gap reduction" is often a cone-widening in disguise, which independently changes retrieval by changing the score distribution's dynamic range. Separating the two requires holding $\rho_f,\rho_g,\tau$ fixed while varying $\|\Delta\|$, which no published study does.

Second, the gap direction is not identified: the loss is blind to a rigid rotation aligning the cones, so the trained value of $\|\Delta\|$ carries seed-dependent variance of the same order as reported treatment effects. Third, the deciding experiment is a pretraining run, not a probe — 3+ orders of magnitude more compute than the fine-tuning studies that produced the current evidence.

## 7. Current Research (as of 2026)

- **Information-imbalance accounts** (Schrodi, Brox et al., Freiburg) — treating gap and object bias as one phenomenon; extending to video/audio pairs *(frontier — verify)*.
- **Gap-as-feature** work (Yeung-Levy group, Stanford; C3 line) — exploiting the gap as a known constant offset to do cross-modal tasks with uni-modal data, rather than removing it.
- **Loss redesign** — sigmoid (SigLIP), and multi-positive / captioner-relabeled objectives that reduce false negatives; gap effects reported anecdotally, not ablated.
- **Platonic-representation framing** (Huh et al., ICML 2024) — if modality-independent encoders converge to a shared statistical structure, the gap is a coordinate artifact and should vanish under representational-similarity measures even when $\|\Delta\|$ is large *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is nonzero $\|\Delta\|$ causally harmful, or merely correlated with things that are?

**Scale.** Pretrain ViT-B/16 + 12-layer text encoder on DataComp-medium (128M pairs), 32 epochs-equivalent, batch 32k — one run is ~$3$k USD on 32×A100 for ~3 days.

**Arms** (3 seeds each, identical data order):
1. **Control:** standard CLIP, learned $\tau$.
2. **Gap-penalized:** add $\beta\|\Delta_{\text{batch}}\|^2$, $\beta$ tuned so final $\|\Delta\|\leq0.1$.
3. **Cone-matched control:** the decisive arm — penalize cone width ($\rho_f,\rho_g$) to match arm 2's measured cone widths, *without* penalizing $\Delta$. This isolates the gap from its algebraic correlate.
4. Freeze $\tau$ at the control's converged value in all arms.

**Deciding number.** ImageNet zero-shot top-1, arm 2 minus arm 3, with seed-level 95% CI. If $|{\Delta\text{acc}}|<0.5$ points, the gap is not causally relevant and the field should stop reporting $\|\Delta\|$ as a quality metric. If arm 2 exceeds arm 3 by $>1.5$ points, gap-closing is a real objective and the pretraining recipe should change.

## 9. Key References

- **[Foundational]** Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Zou. *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS, 2022. — arXiv:2203.02053
- **[Foundational]** Tongzhou Wang, Phillip Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML, 2020. — arXiv:2005.10242
- **[Foundational]** Alec Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. — arXiv:2103.00020
- **[SOTA]** Simon Schrodi, Julian Hoffmann, Max Argus, Volker Fischer, Thomas Brox. *Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models.* ICLR, 2025.
- **[SOTA]** Yuhui Zhang, Elaine Sui, Serena Yeung-Levy. *Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data.* ICLR, 2024.
- **[SOTA]** Yuhui Zhang, Jeff Z. HaoChen, Shih-Cheng Huang, Kuan-Chieh Wang, James Zou, Serena Yeung. *Diagnosing and Rectifying Vision Models using Language.* ICLR, 2023.
- **[SOTA]** Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. *Sigmoid Loss for Language Image Pre-Training.* ICCV, 2023. — arXiv:2303.15343
- **[Method]** Changdae Oh et al. *Geodesic Multi-Modal Mixup for Robust Fine-Tuning.* NeurIPS, 2023.
- **[Method]** Abrar Fahim, Alex Murphy, Alona Fyshe. *It's Not a Modality Gap: Characterizing and Addressing the Contrastive Gap.* 2024.
- **[Context]** Li Jing, Pascal Vincent, Yann LeCun, Yuandong Tian. *Understanding Dimensional Collapse in Contrastive Self-supervised Learning.* ICLR, 2022. — arXiv:2110.09348
- **[Context]** Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola. *The Platonic Representation Hypothesis.* ICML, 2024. — arXiv:2405.07987

## 10. Worked Example

Take CLIP ViT-B/32 and 5,000 MS-COCO val pairs. Measured: $\|\Delta\|\approx0.82$, $\rho_f\approx0.55$, $\rho_g\approx0.45$, SVM separability $\approx100\%$.

Check the algebra. With $\|\bar f\|^2\approx\rho_f$ and $\|\bar g\|^2\approx\rho_g$:

$$\|\Delta\|^2=\|\bar f\|^2+\|\bar g\|^2-2\langle\bar f,\bar g\rangle\approx0.55+0.45-2\langle\bar f,\bar g\rangle=0.67\ \Rightarrow\ \langle\bar f,\bar g\rangle\approx0.165.$$

Now run the "gap closing" intervention everyone runs: subtract $\frac{1}{2}\Delta$ from every image embedding, add it to every text embedding, renormalize. The new gap is $\approx0.0$. Retrieval is unchanged to within noise — because InfoNCE ranks by $\langle f_i,g_j\rangle$ and the shift adds a term that is nearly constant across $j$.

Here is the obstruction. Suppose instead you *widen the cones* — say by an isotropic-noise or uniformity penalty that drops $\rho_f$ to $0.30$ and $\rho_g$ to $0.25$, leaving $\langle\bar f,\bar g\rangle$ at $0.165$. Then

$$\|\Delta\|^2\approx0.30+0.25-0.33=0.22,\qquad\|\Delta\|\approx0.47.$$

The gap fell 43% and you touched nothing about cross-modal alignment. Any retrieval change came from the cone, not the gap — but the paper reports "$\|\Delta\|$: 0.82 → 0.47, R@1 +1.2". Without arm 3 of §8, that number cannot distinguish the two mechanisms. This is why the status here is *partially-solved*: the phenomenon is well characterized and reliably reproduced, and the measure used to score every proposed fix is confounded with a quantity nobody controls for.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*