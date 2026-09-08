---
id: 30-synthetic-data/generator-independent-fidelity-metrics
title: "Fidelity Metrics Decoupled from the Generator's Own Likelihood"
topic: 30-synthetic-data
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fidelity Metrics Decoupled from the Generator's Own Likelihood

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/generator-independent-fidelity-metrics` · **Status:** methodologically-blocked

## 1. Problem Statement

Synthetic data is filtered, ranked, and accepted using *fidelity metrics*. Most of the usable ones are contaminated by the generator itself, in one of three ways:

1. **Explicit** — the metric evaluates $q_\theta(x)$ (per-token log-likelihood, perplexity, ELBO, diffusion NLL bound). Only the generator can compute it, and the generator was trained to maximise it.
2. **Implicit** — the metric fits an auxiliary density over generated samples (kernel density in feature space, $k$-NN radii) and so inherits the generator's support.
3. **Representational** — the metric compares in an embedding (Inception-V3, DINOv2, CLIP) whose training distribution overlaps the generator's, so the metric is blind in exactly the directions the generator is blind.

**Measurement variant.** Define $m(\mathcal{D}_r, \mathcal{D}_g)$ computable from finite sample sets alone, with no access to $q_\theta$, no auxiliary density fit on $\mathcal{D}_g$, and a stated invariance class. *This is the blocked one*: there is no agreed definition of what "generator-independent" means operationally.

**Method variant.** Given such a definition, build an estimator that is sample-efficient, unbiased or bias-corrected, and detects the failure modes likelihood does not: memorisation, mode-dropping, and support inflation.

**Theory variant.** Prove or refute: any metric estimable from $n$ samples of each of $P$ and $Q$, invariant to a class $\mathcal{G}$ of feature reparametrisations, and consistent for a divergence, must have sample complexity exponential in intrinsic dimension — the estimation-theoretic obstruction behind the whole family.

A solution is a metric that (i) ranks generators the same way a held-out downstream task does, (ii) assigns a *worse* score to a training-set copier than to a genuine generator, and (iii) does not change rank order when the embedding is swapped.

## 2. Formal Setting

Real distribution $P$ on $\mathcal{X}$; generator $q_\theta$ with induced law $Q_\theta$. Measured objects:

- $\mathcal{D}_r = \{x_i\}_{i=1}^{n} \sim P$, split into $\mathcal{D}_r^{\text{tr}}$ (the generator's training set) and $\mathcal{D}_r^{\text{te}}$ (held out, never seen). The split is the only source of ground truth about memorisation.
- $\mathcal{D}_g = \{ \tilde x_j \}_{j=1}^{m} \sim Q_\theta$, sampled at a fixed decoding configuration (temperature, guidance scale, sampler steps) — *reported, because the metric is a function of it*.
- Embedding $\phi: \mathcal{X} \to \mathbb{R}^d$, $d = 2048$ (Inception pool3) or $d = 768/1024$ (DINOv2 ViT-B/L).

Fréchet distance, as actually computed:
$$\widehat{\mathrm{FID}} = \|\hat\mu_r - \hat\mu_g\|_2^2 + \operatorname{tr}\!\big(\hat\Sigma_r + \hat\Sigma_g - 2(\hat\Sigma_r\hat\Sigma_g)^{1/2}\big),$$
with $\hat\mu,\hat\Sigma$ the empirical mean and covariance of $\phi$. Note $\widehat{\mathrm{FID}}$ is a biased estimator of its population value and the bias is $O(1/m)$ with a generator-dependent constant, so cross-paper numbers at different $m$ are not comparable.

Precision/recall (Kynkäänniemi et al., 2019): with $r_k(x)$ the distance to the $k$-th nearest neighbour within its own set,
$$\widehat{\mathrm{prec}} = \tfrac{1}{m}\sum_j \mathbf{1}\!\left[\exists\, i: \|\phi(\tilde x_j)-\phi(x_i)\| \le r_k(x_i)\right].$$
The $r_k$ radii *are* a density estimate; this is contamination class 2.

Generator-independence, the definition the field lacks. Candidate: $m$ is $\mathcal{G}$-independent if for all $\theta$ and all $g \in \mathcal{G}$ acting on the embedding,
$$m_{\phi}(\mathcal{D}_r,\mathcal{D}_g) \;\overset{\text{rank}}{=}\; m_{g\circ\phi}(\mathcal{D}_r,\mathcal{D}_g),$$
i.e. rank order over generators is invariant. No published metric states its $\mathcal{G}$.

Assumptions, with violation status:
- $\mathcal{D}_r \perp \phi$ — **violated**: Inception and DINOv2 are trained on ImageNet/LVD-142M, which overlap generator training corpora.
- Gaussianity of $\phi(\cdot)$ under $P$ and $Q$ — **violated**; FID's closed form is exact only for Gaussians.
- I.i.d. sampling of $\mathcal{D}_g$ — **violated** under guidance, rejection sampling, or best-of-$n$ selection, which are standard in synthetic-data pipelines.
- $\mathcal{D}_r^{\text{te}}$ disjoint from training — **often violated** by web-scale duplication.

## 3. State of the Art

**Established.**
- FID (Heusel et al., NeurIPS 2017) and KID (Bińkowski et al., ICLR 2018) are the operational standard; KID's MMD estimator is unbiased, FID's is not.
- Chong & Forsyth (CVPR 2020) show FID and IS are biased in $m$ and give an extrapolated $\mathrm{FID}_\infty$; this is a reproduced, non-controversial correction.
- Stein et al. (NeurIPS 2023) ran a large human-evaluation study across diffusion, GAN, and autoregressive models and found feature choice, not metric form, dominates: DINOv2 features align with human judgements substantially better than Inception, and FID rank order flips between the two on several model pairs.
- Kynkäänniemi et al. (ICLR 2024) show FID can be lowered by shaping the ImageNet-class histogram of generated samples with no perceptual improvement — a direct demonstration that the metric measures embedding statistics rather than fidelity.

**Claimed but unablated.**
- CMMD (Jayasumana et al., CVPR 2024) replaces the Gaussian assumption with CLIP+MMD; the human-alignment claim rests on the authors' own study, not independent replication.
- $\alpha$-precision / $\beta$-recall / authenticity (Alaa et al., ICML 2022) is the most direct attempt at a memorisation-aware, three-way decomposition; the authenticity term uses nearest-neighbour radii, so it is contamination class 2, and it has not been ablated against embedding swap.
- Feature Likelihood Divergence (Jiralerspong et al., NeurIPS 2023) explicitly targets memorisation but does so by fitting a Gaussian-kernel density on generated samples — it fixes contamination class 1 by adopting class 2.

**Benchmark-number-only.** Tabular fidelity scores (TabSynDex, SDMetrics suites) exist as leaderboard values with no established relationship to downstream task transfer.

## 4. What Is Known

- FID's estimator bias is large at practical $m$: reported FID at $m=10{,}000$ differs from $m=50{,}000$ by roughly a factor that reverses model rankings in Chong & Forsyth's CIFAR-10 and ImageNet experiments.
- A model that emits training images verbatim scores near-optimally on FID, precision, and recall against a held-out reference of the same distribution. This is arithmetic, not conjecture: $Q = \hat P_{\text{tr}}$ minimises the empirical objective.
- Memorisation is real at scale: Carlini et al. (USENIX Security 2023) extracted 109 near-duplicate training images from ~175 million Stable Diffusion generations seeded from 350,000 highly duplicated captions. No standard fidelity metric flagged those models.
- Likelihood does not track sample quality. Theis, van den Oord & Bethge (ICLR 2016) show log-likelihood and perceptual quality can be made arbitrarily discordant in high dimension. Nalisnick et al. (ICLR 2019) show deep generative models assign *higher* likelihood to OOD data (CIFAR-10-trained models on SVHN).
- Training on generator-scored synthetic data degrades: Shumailov et al. (Nature, 2024) show recursive training on model output collapses tails within single-digit generations.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no accepted operational test for "this metric is independent of the generator." No paper states an invariance class $\mathcal{G}$ and shows its metric is invariant under it. Without that, "generator-independent" is a slogan.
- **Methodologically blocked.** No agreed ground-truth target. Human preference, downstream task transfer, and divergence-to-$P$ disagree with each other and none is canonical.
- **Empirically open.** Whether any existing metric's ranking of generators survives a full embedding swap (Inception → DINOv2 → CLIP → SigLIP) across ≥20 modern generators. Stein et al. did part of this; the complete rank-correlation matrix over metric × embedding × modality is runnable and unrun.
- **Theoretically open.** Whether a sample-only metric can be simultaneously consistent for a divergence, memorisation-sensitive, and polynomially sample-efficient in intrinsic dimension. Suspected impossible; no proof either way.

## 6. Why It Is Hard

**Absent ground truth compounded by non-identifiability.** The quantity being estimated — divergence between $P$ and $Q_\theta$ in a perceptually meaningful geometry — requires choosing that geometry, and every available choice is a network trained on data that overlaps the generator's. The metric and the object it judges share a prior. Two generators can be indistinguishable under every embedding available and still differ on the axis that matters downstream: the copier and the honest model are *identical* in distribution and differ only in their relation to $\mathcal{D}_r^{\text{tr}}$ — information not present in $(\mathcal{D}_r, \mathcal{D}_g)$ as unlabelled sets. That is a genuine identifiability failure, not a compute limitation. The escape route — bring in $q_\theta$ or a surrogate density — is exactly what the problem forbids.

Secondary: divergence estimation in $\mathbb{R}^{768}$ has sample complexity exponential in intrinsic dimension, so all deployed metrics are low-order summaries (mean, covariance, $k$-NN radii) with unquantified blind spots.

## 7. Current Research (as of 2026)

- **Embedding-robust evaluation.** Post-Stein consensus is drifting to DINOv2-based FD as default, with CMMD as an alternative. Rank-stability studies across embeddings are the active thread *(frontier — verify current results)*.
- **Memorisation-aware fidelity.** Successors to authenticity ($\alpha$-precision line, Alaa/van der Schaar) and FLD (Mila) attempt three-way quality/diversity/novelty decompositions. All currently use a density surrogate.
- **Two-sample tests as metrics.** Classifier two-sample testing (Lopez-Paz & Oquab, ICLR 2017) and MAUVE (Pillutla et al., NeurIPS 2021) are being revisited for synthetic-data acceptance because the critic is trained on both sample sets, not on the generator's likelihood — a partial fix to class 1 that leaves class 3 open.
- **Downstream-transfer as the target.** Train-on-synthetic/test-on-real (TSTR) is increasingly used as the arbiter for tabular and instruction data. It is generator-independent by construction but expensive and task-specific.

## 8. Concrete Next Experiment

**Question:** does any published fidelity metric rank generators the same way under an embedding swap, and does any of them penalise a copier?

**Scale.** CIFAR-10 and ImageNet-256. Twelve generators spanning FID 1.8–25 (DDPM, EDM, LDM, StyleGAN-XL, VQGAN, plus three deliberately degraded variants). $m = 50{,}000$ samples each, fixed decoding config, logged.

**Control arm — the decisive addition.** A thirteenth "generator" $Q_{\text{copy}}$ that returns a uniformly sampled training image with additive Gaussian noise $\sigma = 0.01$. It has zero generalisation and near-zero divergence from $\hat P_{\text{tr}}$.

**Metrics.** FID, $\mathrm{FID}_\infty$, KID, precision/recall, density/coverage, $\alpha$-precision/authenticity, FLD, CMMD, C2ST — each computed under four embeddings (Inception-V3, DINOv2 ViT-L, CLIP ViT-L, SigLIP), against $\mathcal{D}_r^{\text{te}}$ (10k held-out).

**The deciding number.** For each metric, report
$$\rho^{\min} = \min_{\phi \neq \phi'} \operatorname{Kendall}\tau\big(\text{rank}_\phi, \text{rank}_{\phi'}\big) \quad\text{over the 12 honest generators},$$
plus the copier's percentile rank. A metric passes only if $\rho^{\min} \ge 0.8$ **and** $Q_{\text{copy}}$ places in the bottom quartile. Prediction: no current metric passes both; most place the copier first. Cost: roughly 600 sample sets × embedding forward passes, order 2–3 GPU-weeks — cheap enough that the absence of this table is a gap in the field, not a budget problem.

## 9. Key References

- **[Foundational]** L. Theis, A. van den Oord, M. Bethge. *A Note on the Evaluation of Generative Models.* ICLR 2016. — arXiv:1511.01844
- **[Foundational]** M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, S. Hochreiter. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS 2017. — arXiv:1706.08500
- **[Foundational]** E. Nalisnick, A. Matsukawa, Y. W. Teh, D. Gorur, B. Lakshminarayanan. *Do Deep Generative Models Know What They Don't Know?* ICLR 2019. — arXiv:1810.09136
- **[SOTA]** G. Stein et al. *Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models.* NeurIPS 2023. — arXiv:2306.04675
- **[SOTA]** T. Kynkäänniemi, T. Karras, M. Aittala, T. Aila, J. Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR 2023.
- **[SOTA]** S. Jayasumana, S. Ramalingam, A. Veit, D. Glasner, A. Chakrabarti, S. Kumar. *Rethinking FID: Towards a Better Evaluation Metric for Image Generation.* CVPR 2024. — arXiv:2401.09603
- **[SOTA]** A. Alaa, B. van Breugel, E. Saveliev, M. van der Schaar. *How Faithful Is Your Synthetic Data? Sample-Level Metrics for Evaluating and Auditing Generative Models.* ICML 2022. — arXiv:2102.08921
- **[SOTA]** M. Jiralerspong, A. Bose, I. Gemp, C. Qin, Y. Bachrach, G. Gidel. *Feature Likelihood Divergence: Evaluating the Generalization of Generative Models Using Samples.* NeurIPS 2023.
- **[Method]** T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, T. Aila. *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS 2019. — arXiv:1904.06991
- **[Method]** M. F. Naeem, S. J. Oh, Y. Uh, Y. Choi, J. Yoo. *Reliable Fidelity and Diversity Metrics for Generative Models.* ICML 2020. — arXiv:2002.09797
- **[Method]** M. Bińkowski, D. J. Sutherland, M. Arbel, A. Gretton. *Demystifying MMD GANs.* ICLR 2018. — arXiv:1801.01401
- **[Method]** M. Chong, D. Forsyth. *Effectively Unbiased FID and Inception Score and Where to Find Them.* CVPR 2020. — arXiv:1911.07023
- **[Method]** D. Lopez-Paz, M. Oquab. *Revisiting Classifier Two-Sample Tests.* ICLR 2017. — arXiv:1610.06545
- **[Method]** K. Pillutla et al. *MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers.* NeurIPS 2021. — arXiv:2102.01454
- **[Evidence]** N. Carlini et al. *Extracting Training Data from Diffusion Models.* USENIX Security 2023. — arXiv:2301.13188
- **[Evidence]** I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. *AI Models Collapse When Trained on Recursively Generated Data.* Nature 631, 2024.

## 10. Worked Example

CIFAR-10, 50,000 train / 10,000 test. Three "generators", reference set $\mathcal{D}_r^{\text{te}}$:

| Generator | What it does | FID (Inception, $m{=}50$k) | Precision | Recall | New information |
|---|---|---|---|---|---|
| $Q_A$ | EDM diffusion model | ~2.0 | ~0.68 | ~0.60 | yes |
| $Q_{\text{copy}}$ | training image + $\mathcal{N}(0,0.01^2)$ | $\approx$ FID(train, test) $\approx 3$ | $\approx 0.99$ | $\approx 0.99$ | **none** |
| $Q_C$ | 500 training images, resampled | $\approx$ 3 + mode-drop penalty | $\approx 0.99$ | low | none |

The arithmetic that makes the obstruction visible: $\mathrm{FID}(\mathcal{D}_r^{\text{tr}}, \mathcal{D}_r^{\text{te}})$ is not zero — it is the finite-sample floor, a small positive number (order 3 at these sizes; recompute rather than cite). $Q_{\text{copy}}$ sits *at that floor*. It cannot be distinguished from a perfect generator by any statistic of $(\mathcal{D}_r^{\text{te}}, \mathcal{D}_g)$, because in distribution it is one — $\hat P_{\text{tr}}$ and $P$ are indistinguishable at 50k samples in a 2048-dimensional feature space, and any metric consistent for a divergence *must* score it well. Its precision and recall are near 1 for the same reason.

Only $Q_C$ is caught, and only by recall, because mode-dropping *is* a distributional difference.

Now the trap. The natural fix is to bring in $\mathcal{D}_r^{\text{tr}}$ and measure novelty: nearest-neighbour distance from each $\tilde x_j$ to the training set, or FLD's held-out kernel likelihood. Both work — $Q_{\text{copy}}$ has median $\ell_2$ novelty distance $\approx 0.01\sqrt{3072} \approx 0.55$ pixels-units versus a genuine model's tens — and both reintroduce exactly what the problem forbids: a density estimate whose bandwidth, feature space, and support are set by the generated samples themselves. Halve the noise to $\sigma=0.005$ and the threshold moves; swap Inception for DINOv2 and the ordering of borderline models moves with it.

The obstruction is therefore not that we lack a good metric. It is that the information distinguishing a fidelity-preserving generator from a copier is not in the sample sets a generator-independent metric is allowed to see, and every method that recovers it does so by re-coupling the metric to a density model.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*