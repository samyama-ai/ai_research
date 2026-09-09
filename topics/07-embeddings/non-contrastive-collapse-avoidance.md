---
id: 07-embeddings/non-contrastive-collapse-avoidance
title: "Why Non-Contrastive Methods Avoid Collapse"
topic: 07-embeddings
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Why Non-Contrastive Methods Avoid Collapse

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/non-contrastive-collapse-avoidance` · **Status:** partially-solved

## 1. Problem Statement

BYOL, SimSiam, DINO and their descendants minimize a loss whose global minimum is a constant function: map every input to the same vector and the loss is optimal. They do not use negative pairs. Yet trained networks do not collapse, and their features reach ImageNet linear-probe accuracy comparable to contrastive methods. The question is *why*.

Three variants, with different difficulty:

- **Theory.** Given the architecture (predictor head, stop-gradient, EMA target, weight decay, normalization) and an optimizer, prove that the collapsed solution is not reached from standard initialization — either because it is a saddle/unstable point of the gradient flow, or because the trajectory never enters its basin. Solving means: a theorem for a non-linear network at realistic width, with the necessary and sufficient set of components identified.
- **Method.** Which components are *causally* required? An ablation lattice over {predictor, stop-grad, EMA, weight decay, BatchNorm/LayerNorm, centering+sharpening} that identifies a minimal sufficient subset, and shows the remaining components are redundant rather than merely helpful.
- **Measurement.** Define "collapse" so it is falsifiable. Complete collapse (rank 1) and dimensional collapse (rank $\ll d$) are different failures; the standard diagnostic used in practice detects only the first.

Status is *partially-solved*: the linear/two-layer case is understood, the deep non-linear case is not, and the ablations are confounded.

## 2. Formal Setting

Encoder $f_\theta:\mathcal{X}\to\mathbb{R}^d$, predictor $q_\phi:\mathbb{R}^d\to\mathbb{R}^d$, target encoder $f_\xi$ with EMA update $\xi \leftarrow \tau\xi + (1-\tau)\theta$, $\tau\in[0,1]$ ($\tau=0$ recovers SimSiam's shared weights + stop-gradient). Augmentations $t,t'\sim\mathcal{T}$. The BYOL/SimSiam loss on input $x$:

$$\mathcal{L}(\theta,\phi) \;=\; \mathbb{E}_{x,t,t'}\Big\|\,\overline{q_\phi(f_\theta(t(x)))} - \overline{\mathrm{sg}[f_\xi(t'(x))]}\,\Big\|_2^2,$$

with $\bar{u}=u/\|u\|_2$ and $\mathrm{sg}$ the stop-gradient. Note $\mathcal{L}=0$ whenever $f$ is constant.

Measured quantities:

- **Embedding second moment.** $C = \mathbb{E}_x[\bar z \bar z^\top]$, $\bar z = \overline{f_\theta(x)}$, estimated over a held-out set of $N \ge 10^4$ images (not the training batch; batch statistics are themselves a candidate mechanism).
- **Per-coordinate std** (the SimSiam diagnostic): $s = \frac{1}{d}\sum_i \mathrm{std}_x(\bar z_i)$. Healthy reference value $1/\sqrt d$; collapse gives $s\to 0$.
- **Effective rank** (Roy & Vetterli 2007), with $p_k = \sigma_k/\sum_j\sigma_j$ over singular values of the $N\times d$ embedding matrix: $\mathrm{RankMe} = \exp\!\big(-\sum_k p_k\log p_k\big)$. Reported as a number in $[1,d]$.
- **Alignment/uniformity** (Wang & Isola 2020) as an auxiliary geometry probe.
- **Downstream signal**: ImageNet-1k linear probe top-1, fixed protocol, 90 epochs.

Assumptions the theory rests on, and their status:

| Assumption | Used by | Violated in practice? |
|---|---|---|
| Linear or two-layer encoder | Tian et al. 2021; Simon et al. 2023 | Yes — ResNet-50/ViT are deep and non-linear |
| Augmentations act as an isotropic-noise / known covariance operator | most closed-form results | Yes — crop+color-jitter has strongly anisotropic, image-dependent covariance |
| Gradient flow (continuous time), no momentum | all dynamical proofs | Yes — SGD+momentum or LARS/AdamW with cosine schedule |
| Infinite data / population loss | all | Partially — 1.28M images, heavy augmentation |
| $\ell_2$-normalized output, no BatchNorm interaction | some | Yes — projector BN is standard |

## 3. State of the Art

**Theory SOTA (established).** Tian, Chen & Ganguli (*Understanding Self-Supervised Learning Dynamics without Contrastive Pairs*, ICML 2021) give the reference result: in a linear two-layer setting they derive coupled dynamics for the predictor $W_p$ and the feature correlation matrix, show the eigenspaces align, and show that weight decay plus EMA produce a repulsive term on small eigenvalues. Ziyin, Li & Meng (*What shapes the loss landscape of self-supervised learning?*, ICLR 2023) prove that in a linear model the collapsed solution is a **saddle point, not a minimum**, once weight decay is present — collapse is escapable, not merely avoided. Halvagal, Laborieux & Zenke (*Implicit variance regularization in non-contrastive SSL*, NeurIPS 2023) show the predictor+EMA pair induces an *implicit* variance penalty: eigenmodes with small variance get an effectively larger learning rate, which is the same effect VICReg imposes explicitly.

**Established as prediction, not just description:** DirectPred (Tian et al. 2021) sets $W_p$ analytically from the eigendecomposition of the feature correlation matrix — no predictor gradient at all — and still trains, reaching within ~1 point of the learned predictor on ImageNet at 300 epochs ResNet-50. A theory that makes a working algorithm is stronger evidence than a post-hoc explanation.

**Claimed but unablated.** That the mechanism in the linear analysis is *the* mechanism at ResNet-50 scale. No paper measures the predicted eigenspace-alignment quantity inside a trained ResNet-50 and shows it tracks the theory's prediction; the transfer is asserted by analogy.

**Benchmark-number-only results.** "BYOL works fine without BatchNorm" rests on Richemond et al. (2020) reporting 73.9% top-1 with GroupNorm + weight standardization vs 74.3% with BN. That is one number on one architecture; it refutes the strong claim that BN provides implicit contrast, and nothing more.

**Empirical SOTA.** DINOv2 (Oquab et al., TMLR 2024) is the largest system relying on non-contrastive objectives (centering/sharpening + Sinkhorn-Knopp + KoLeo) and does not collapse at 1B parameters — but it stacks several anti-collapse devices, so it is evidence of *practice*, not of *mechanism*.

## 4. What Is Known

- **Stop-gradient is necessary in SimSiam.** Removing it makes the loss fall to $-1$ within tens of iterations and linear probe drops to 0.1% (chance) — ResNet-50, ImageNet-1k, 100 epochs (Chen & He, CVPR 2021).
- **The predictor is necessary in SimSiam** (0.1% top-1 without it), but *not* if it is replaced by a fixed random head with a suitable lr schedule; SimSiam reaches 68.1% top-1 at 100 epochs with predictor lr held constant.
- **EMA is not necessary.** SimSiam ($\tau=0$) works. So any theory that requires EMA is not the general explanation.
- **BatchNorm is not necessary.** 73.9% vs 74.3% (Richemond et al. 2020), ResNet-50, 1000 epochs.
- **Redundancy-reduction methods reach the same place by explicit means.** Barlow Twins 73.2% and VICReg 73.2% top-1 (ResNet-50, 1000 epochs) with an explicit off-diagonal/variance penalty and no predictor and no stop-gradient.
- **Duality.** Garrido et al. (ICLR 2023) show contrastive and dimension-contrastive (non-contrastive) criteria are related by a transpose of the Gram/covariance matrix, and empirically match within ~1 point when tuned — so the two families are not mechanistically as far apart as the naming suggests.
- **Learning is stepwise.** Simon et al. (ICML 2023) show representations are learned one eigendirection at a time, with exact solutions in a linearized model; rank grows in discrete jumps.
- **Collapse is a real failure mode, not hypothetical.** Li, Efros & Pathak (ECCV 2022) document partial collapse in Siamese training and show it is preceded by measurable rank loss.

## 5. What Is Not Known

- **Theoretically open.** No proof that collapse is avoided for a deep non-linear encoder under SGD with momentum and realistic augmentation covariance. All existing proofs are linear/two-layer or gradient-flow. Also open: whether the saddle-point result of Ziyin et al. survives without weight decay.
- **Theoretically open.** Whether the anti-collapse mechanisms of SimSiam ($\tau=0$, predictor, stop-grad) and BYOL ($\tau\approx0.996$, EMA) are the *same* mechanism or two different ones with a common outcome. Currently there are at least three sufficient mechanisms and no proof of which is operative when several are present.
- **Empirically open.** No paper has run the full ablation lattice (predictor × stop-grad × EMA × weight decay × projector-BN, $2^5=32$ cells) at a single scale with a fixed protocol and RankMe reported for each cell. Each cell is roughly 100 ResNet-50 GPU-days at 100 epochs; the full lattice is feasible for a well-resourced lab and has not been run.
- **Methodologically blocked.** Dimensional collapse has no agreed threshold. RankMe is a continuous number with no calibrated "collapsed" cutoff, and it is not invariant to output normalization choices. Without that, "did it collapse?" is not a decidable predicate.

## 6. Why It Is Hard

**Non-identifiability of the mechanism.** The obstruction is not compute and not importance — it is that the standard recipe contains four or five devices, each *individually sufficient* to prevent complete collapse. Remove EMA and stop-grad+predictor carries it; remove the predictor and EMA plus weight decay carries it; remove both and add an explicit variance term and Barlow Twins/VICReg carries it. A single-factor ablation therefore cannot attribute causation: every removal is masked by the survivors. The clean design is a factorial ablation, and nobody has paid for it.

**Compounding this: an evaluation that does not measure what it names.** The per-coordinate std diagnostic $s\approx 1/\sqrt d$ is invariant to the rank of the embedding (Section 10). Runs pass the standard collapse check while sitting at rank $\ll d$, so the literature's "no collapse" claims are partly claims about a metric that cannot see the failure mode people actually care about.

## 7. Current Research (as of 2026)

- **Spectral/eigendynamics accounts.** Following Tian et al. and Simon et al. — Stanford (Ganguli), Meta AI/FAIR (Balestriero, LeCun, Garrido). Direction: closed-form predictors and rank-growth schedules.
- **Implicit-regularization accounts.** Zenke's group (FMI Basel) on implicit variance regularization; connects non-contrastive SSL to biologically plausible learning rules.
- **Landscape accounts.** Ziyin and collaborators (MIT/NTT) on saddle structure and symmetry-induced collapse; the general claim is that collapse points are symmetric solutions whose stability is controlled by weight decay.
- **Rank as a first-class objective.** RankMe (Garrido et al., ICML 2023) used as an unsupervised model-selection criterion; KoLeo in DINOv2 as an explicit rank/uniformity term. *(frontier — verify)* Whether explicit rank terms are now standard because they are needed, or because they cheaply insure against a rare failure, is not settled in print.
- **Scaling.** *(frontier — verify)* Whether collapse pressure grows or shrinks with model size at 1B+ parameters — anecdotally instabilities increase, but no controlled study exists.

## 8. Concrete Next Experiment

**Question decided:** is there a *single* necessary component, or is anti-collapse a redundant multi-mechanism property?

- **Scale.** ResNet-50, ImageNet-1k, 100 epochs, batch 512, LARS, fixed augmentation recipe. ~100 GPU-days per cell on A100s.
- **Design.** Full $2^4 = 16$ factorial over {predictor on/off, stop-gradient on/off, EMA $\tau\in\{0, 0.996\}$, weight decay $\in\{0, 1.5\times10^{-6}\}$}, each cell run at 3 seeds. 48 runs. Projector BN held fixed on.
- **Control arm.** (a) SimSiam default cell (predictor on, sg on, $\tau=0$, wd on) — must reproduce 68.1% ±0.5. (b) A deliberately collapsed arm (predictor off, sg off) — must give 0.1%.
- **Deciding number.** For each cell, RankMe on 50k held-out ImageNet validation embeddings, $d=2048$. The predicate: **is there any single factor whose "off" setting drives RankMe below 32 in every cell where the other three are off, and above 256 otherwise?** If yes, that factor is necessary and the theory has one target. If instead $\ge 2$ distinct single-factor-on cells hold RankMe $>256$, redundancy is established and any single-mechanism theory is refuted at this scale. Secondary readout: linear-probe top-1, to check that RankMe $>256$ actually corresponds to useful features.

Cost: ~4,800 GPU-hours, about $15k of spot compute. This is the cheapest experiment that resolves the identifiability problem in Section 6.

## 9. Key References

- **[Foundational]** Jean-Bastien Grill, Florian Strub, Florent Altché, et al. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* NeurIPS 2020. — arXiv:2006.07733
- **[Foundational]** Xinlei Chen, Kaiming He. *Exploring Simple Siamese Representation Learning.* CVPR 2021. — arXiv:2011.10566
- **[SOTA — theory]** Yuandong Tian, Xinlei Chen, Surya Ganguli. *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs.* ICML 2021. — arXiv:2102.06810
- **[SOTA — theory]** Liu Ziyin, Ekdeep Singh Lubana, Masahito Ueda, Hidenori Tanaka. *What shapes the loss landscape of self-supervised learning?* ICLR 2023. — arXiv:2210.00638
- **[SOTA — theory]** Manu Srinath Halvagal, Axel Laborieux, Friedemann Zenke. *Implicit variance regularization in non-contrastive SSL.* NeurIPS 2023. — arXiv:2212.04858
- **[SOTA — theory]** James B. Simon, Maksis Knutins, Liu Ziyin, Daniel Geisz, Abraham J. Fetterman, Joshua Albrecht. *On the Stepwise Nature of Self-Supervised Learning.* ICML 2023. — arXiv:2303.15438
- **[Method]** Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny. *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* ICML 2021. — arXiv:2103.03230
- **[Method]** Adrien Bardes, Jean Ponce, Yann LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022. — arXiv:2105.04906
- **[Method]** Mathilde Caron, Hugo Touvron, Ishan Misra, et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021. — arXiv:2104.14294
- **[Measurement]** Quentin Garrido, Randall Balestriero, Laurent Najman, Yann LeCun. *RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank.* ICML 2023. — arXiv:2210.02885
- **[Analysis]** Quentin Garrido, Yubei Chen, Adrien Bardes, Laurent Najman, Yann LeCun. *On the Duality between Contrastive and Non-Contrastive Self-Supervised Learning.* ICLR 2023. — arXiv:2206.02574
- **[Analysis]** Pierre H. Richemond, Jean-Bastien Grill, Florent Altché, et al. *BYOL works even without batch statistics.* NeurIPS 2020 SSL Workshop. — arXiv:2010.10241
- **[Analysis]** Alexander C. Li, Alexei A. Efros, Deepak Pathak. *Understanding Collapse in Non-Contrastive Siamese Representation Learning.* ECCV 2022. — arXiv:2209.15007
- **[Analysis]** Randall Balestriero, Yann LeCun. *Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods.* NeurIPS 2022. — arXiv:2205.11508
- **[Survey]** Randall Balestriero, Mark Ibrahim, Vlad Sobal, et al. *A Cookbook of Self-Supervised Learning.* 2023. — arXiv:2304.12210

## 10. Worked Example

**The diagnostic is blind to the failure it is used to rule out.**

Take $d=2048$, $\ell_2$-normalized embeddings, $N=50{,}000$ held-out images.

*Healthy target.* If $\bar z$ is isotropic on the unit sphere $S^{2047}$, then $\mathbb{E}[\bar z_i^2] = 1/d$ and

$$\mathrm{std}(\bar z_i) \approx \sqrt{1/2048} = 0.0221.$$

This is the number SimSiam-style training logs converge to, and it is the standard "no collapse" check.

*Dimensionally collapsed run.* Now suppose training has driven the embedding into a **random 64-dimensional subspace** $V\subset\mathbb{R}^{2048}$, isotropic within $V$. Rank is 64 out of 2048 — a 32× loss of capacity. Take a random orthonormal basis of $V$ with columns $v_1..v_{64}$. For coordinate $i$, $\bar z_i = \sum_{k}c_k (v_k)_i$ with $\sum_k c_k^2 = 1$, and by rotational symmetry of a *randomly oriented* subspace, $\mathbb{E}[(v_k)_i^2]=1/2048$. Hence

$$\mathbb{E}[\bar z_i^2] = \sum_k \mathbb{E}[c_k^2]\,\mathbb{E}[(v_k)_i^2] = \frac{1}{2048}, \qquad \mathrm{std}(\bar z_i)\approx 0.0221.$$

Identical to the healthy case, to within sampling noise of order $1/\sqrt{N}$.

*What the rank metric says.* Singular values: 64 equal non-zero values, 1984 zeros. $p_k = 1/64$ for $k\le 64$, so

$$\mathrm{RankMe} = \exp\!\Big(-\sum_{k=1}^{64}\tfrac{1}{64}\log\tfrac{1}{64}\Big) = 64,$$

against 2048 for the healthy run.

**The obstruction made visible.** Two runs, one with 32× more usable representational capacity than the other, produce *the same value* of the diagnostic the field uses to certify that non-contrastive training "did not collapse." Every ablation that concludes "component X is not needed, the std stayed at $1/\sqrt d$" is therefore uninformative about dimensional collapse. Until the factorial ablation in Section 8 is run with RankMe as the readout, claims about which component prevents collapse are claims about rank-1 collapse only — and rank-1 collapse is the easy case that four separate devices each already prevent.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*