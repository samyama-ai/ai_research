---
id: 07-embeddings/dimensional-collapse-prediction
title: "Dimensional Collapse Prediction in Self-Supervised Learning"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Dimensional Collapse Prediction in Self-Supervised Learning

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/dimensional-collapse-prediction` · **Status:** open

## 1. Problem Statement

Joint-embedding self-supervised learning (SSL) can converge to representations that occupy a strict subspace of the embedding space: the covariance spectrum has many near-zero eigenvalues, and downstream linear probes lose accuracy. This is **dimensional collapse** — distinct from *complete collapse*, where all embeddings become one point.

The problem: **given a training run's early-time state, predict its final effective dimensionality and the downstream accuracy loss attributable to collapse — before the compute is spent.**

Three variants, of different difficulty:

- **Measurement.** Given a trained encoder and an unlabeled probe set, output a scalar that ranks checkpoints by downstream linear-probe accuracy without labels. Partially solved (RankMe, LiDAR, $\alpha$-ReQ).
- **Method (forecasting).** Given the run at $t = \epsilon T$ (say the first 5% of steps), predict the spectrum at $t = T$. Essentially unaddressed as a stated task.
- **Theory.** Given architecture, augmentation distribution, loss, learning rate, and weight decay, decide whether the SSL objective's reachable minimum is rank-deficient. Solved only for linear/deep-linear models.

Solving it means: a predictor that, at $\le 5\%$ of pretraining compute, ranks hyperparameter configurations by final linear-probe accuracy with Spearman $\rho$ close to that of the final-checkpoint measurement, and beats the trivial baseline of extrapolating the current rank.

## 2. Formal Setting

Encoder $f_\theta: \mathcal{X} \to \mathbb{R}^{d}$, projector $g_\phi: \mathbb{R}^d \to \mathbb{R}^{k}$, augmentation distribution $\mathcal{A}$. Embeddings $z = g_\phi(f_\theta(a(x)))$, $a \sim \mathcal{A}$.

**Empirical covariance.** On a probe set of $n$ unlabeled images (measured: $n = 25{,}600$ is the RankMe convention; $n \ge 10k$ is where the estimate stabilises for $k = 2048$),
$$\hat{\Sigma} = \tfrac{1}{n}\sum_{i=1}^n (z_i - \bar z)(z_i - \bar z)^\top, \qquad \lambda_1 \ge \dots \ge \lambda_k \ge 0 .$$

**Effective rank** (Roy & Vetterli 2007), the measured collapse statistic. With $p_i = \sigma_i / \sum_j \sigma_j$ over singular values $\sigma_i$ of the centred embedding matrix,
$$\mathrm{RankMe} = \exp\!\Big(-\sum_{i=1}^{k} p_i \log p_i\Big) \in [1, k].$$
Reported in practice with $\epsilon = 10^{-7}$ added to $p_i$ for numerical stability.

**Spectral decay exponent** ($\alpha$-ReQ). Fit $\lambda_i \propto i^{-\alpha}$ by least squares on $\log \lambda_i$ vs $\log i$ over $i \in [i_{\min}, i_{\max}]$ (typically the first few hundred indices; the fit range is a free choice and changes $\alpha$ materially).

**Collapse fraction.** $c_\tau = \tfrac{1}{k}\,\\#\{i : \lambda_i < \tau \lambda_1\}$, $\tau = 10^{-4}$ a common but arbitrary cut.

**Target.** $\mathrm{Acc}(T)$ = ImageNet-1k linear probe top-1 at the end of pretraining. The forecasting task is a map $\Psi: \text{state}(\epsilon T) \mapsto \widehat{\mathrm{Acc}}(T)$ scored by rank correlation over a config grid.

**Assumptions, and which fail.**
- *Collapse is a property of the projector output.* Violated: encoder and projector spectra differ substantially; the projector absorbs much of the collapse, which is why probes are read off $f_\theta$, not $g_\phi$.
- *Effective rank is monotone in downstream accuracy.* Violated at the extremes — whitening-based methods (Barlow Twins, VICReg, W-MSE) push rank toward $k$ by construction, so rank saturates and stops discriminating.
- *A single scalar summarises the spectrum.* Violated: two runs with equal RankMe can have different $\alpha$ and different transfer.
- *Training dynamics are smooth in $t$.* Violated: eigendirections are learned in discrete steps (Simon et al., ICML 2023), so extrapolation from a plateau is systematically wrong.

## 3. State of the Art

**Established (measurement).**
- **RankMe** (Garrido, Balestriero, Najman, LeCun, ICML 2023) — label-free effective rank on 25.6k unlabeled images tracks downstream accuracy across hyperparameter sweeps of SimCLR, VICReg, DINO, and selects configurations close to the oracle on ImageNet and on OOD transfer sets. Independently reused since as a standard diagnostic.
- **LiDAR** (Thilak, Huang, Saremi, Dinh, Goh, Nakkiran, Susskind, Littwin, ICLR 2024) — replaces the raw covariance with the LDA-style matrix built from augmentation-induced classes, reporting higher rank correlation with linear probing than RankMe, notably for I-JEPA and DINO. Claimed improvement is a benchmark number over the authors' own sweep; independent replication is thin.
- **$\alpha$-ReQ** (Agrawal, Raju, Pitkow, NeurIPS 2022) — power-law exponent $\alpha \approx 1$ correlates with best transfer; $\alpha \gg 1$ (fast decay, effectively collapsed) and $\alpha \ll 1$ both hurt.

**Established (mechanism).** Jing, Vondrick, Tian, LeCun (ICLR 2022) showed contrastive SSL still collapses along a subspace, attributed it to implicit regularisation plus augmentation-induced covariance, and showed the projector protects the encoder; DirectCLR (fixed subvector, no trainable projector) reported ImageNet linear-probe top-1 of roughly 62–63% at 100 epochs, about one point above their SimCLR-with-linear-projector baseline. Single seed, single setting.

**Theory SOTA.** Ziyin, Li, Meng (ICLR 2023) characterise, for linear models, when collapsed solutions are global minima of the SSL loss as a function of weight decay and prediction-head structure. Tian, Chen, Ganguli (ICML 2021) give eigenspace dynamics for BYOL-type predictors, yielding DirectPred. Balestriero & LeCun (NeurIPS 2022) identify contrastive and non-contrastive objectives with spectral embedding methods.

**Claimed but unablated.** That collapse *causes* the downstream drop, rather than co-occurring with it. Every published correlation is over hyperparameter grids where learning rate and weight decay move both quantities at once. No intervention that fixes rank while holding everything else constant has been reported at ImageNet scale.

**Forecasting SOTA.** None. There is no published method that takes a checkpoint at 5% of training and predicts the final spectrum.

## 4. What Is Known

- Contrastive losses do not prevent dimensional collapse: SimCLR/ResNet-50/ImageNet-1k, $k = 2048$ projector output, shows a large block of singular values many orders of magnitude below $\sigma_1$ (Jing et al., ICLR 2022).
- Removing the trainable projector and using a fixed subvector ($d_0 = 360$ of 2048) matches or slightly beats the projector baseline at 100 epochs, ImageNet-1k (Jing et al.).
- Effective rank measured on 25.6k images ranks checkpoints across SSL methods and hyperparameters, at ResNet-50/ImageNet-1k and ViT scale (Garrido et al., ICML 2023).
- Explicit decorrelation works: Barlow Twins (Zbontar et al., ICML 2021) reaches 73.2% ImageNet top-1 linear probe (ResNet-50, 1000 epochs) with a redundancy-reduction term; VICReg (Bardes, Ponce, LeCun, ICLR 2022) reaches 73.2% with an explicit variance hinge; both use very wide projectors ($k = 8192$).
- Rank falls with depth in ordinary supervised networks too — "rank diminishing" (Feng et al., NeurIPS 2022) — so low rank is not SSL-specific.
- Eigendirections emerge sequentially, in discrete steps, in the small-learning-rate limit (Simon, Knutins, Ziyin, Geisz, Fetterman, Albrecht, ICML 2023).
- Neural collapse in supervised training (Papyan, Han, Donoho, PNAS 2020) is a *different* phenomenon: rank $\to C-1$ for $C$ classes, driven by labels, not augmentations.

## 5. What Is Not Known

- **Theoretically open.** No characterisation of the collapsed set for nonlinear encoders with realistic augmentations. Ziyin et al. cover linear models; the deep ReLU/ViT case has no proof either way. Also open: whether a threshold on augmentation strength separating collapsing from non-collapsing regimes exists in any nonlinear model.
- **Empirically open.** Does an early-time signal predict late-time collapse? Runnable today — a 60-config grid of 100-epoch ImageNet pretrainings is roughly 1k–3k GPU-hours per method on A100s — but unrun as a stated forecasting benchmark.
- **Empirically open.** Is collapse causal? An intervention arm (e.g. rank-preserving reparameterisation at matched loss, matched LR, matched weight decay) has not been run.
- **Methodologically blocked.** There is no agreed definition of "collapsed". $c_\tau$ depends on $\tau$; $\alpha$ depends on the fit window; RankMe saturates for whitening losses. Different papers report different collapse counts for the same checkpoint class.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth**.

- *Confounding.* Rank and accuracy are both monotone functions of learning rate and weight decay over the usable range. Sweeping either moves both. The reported correlations are therefore consistent with rank being a proxy for "was this run tuned well" rather than for collapse itself.
- *Absent ground truth.* There is no label for "this run collapsed". The only endpoint is the downstream probe, which is what you wanted to predict — so any evaluation of a collapse predictor is circular unless a separate intervention pins the causal arrow.
- *Non-stationary dynamics.* Stepwise eigenvalue emergence means the spectrum at 5% of training can sit on a plateau immediately preceding the emergence of the directions that matter. Linear extrapolation is not merely noisy; it is biased low.
- *Scale.* Deciding the forecasting question needs $O(10^2)$ full pretraining runs. That is the reason it has not been done, not the difficulty of the idea.

## 7. Current Research (as of 2026)

- **Label-free model selection.** Successors to RankMe/LiDAR, aiming at scalars stable under whitening losses. Meta FAIR (Garrido, Balestriero, LeCun) and Apple ML Research (Littwin, Nakkiran, Thilak) are the identifiable groups.
- **Spectral theory of joint embedding.** Duality between contrastive and non-contrastive objectives (Garrido et al., ICLR 2023); rank differential accounts of non-contrastive learning (Zhuo et al., ICLR 2023).
- **Collapse in masked/predictive SSL.** I-JEPA-style models where the augmentation-covariance argument does not directly apply; measurement transfers poorly *(frontier — verify)*.
- **Collapse diagnostics for multimodal encoders.** Effective-rank arguments applied to CLIP-family embedding spaces and to the modality gap *(frontier — verify)*.
- **Scaling-law framing.** Treating effective rank as the quantity that saturates first as data scales *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does the state at 5% of pretraining predict final downstream accuracy better than the trivial current-value baseline?

**Scale.** ResNet-50, ImageNet-1k, 100-epoch pretraining. Grid of 48 configurations: 4 learning rates $\times$ 3 weight decays $\times$ 2 projector widths $\{2048, 8192\}$ $\times$ 2 methods $\{$SimCLR, VICReg$\}$. Two seeds on a 12-config subset for noise floor. Cost: about 2.5k A100-hours.

**Instrumentation.** At epochs $\{1,2,3,4,5,10,25,50,100\}$ record: full projector and encoder covariance spectra on a fixed 25.6k-image probe set, RankMe, $\alpha$ with a pre-registered fit window $i \in [1, 256]$, loss, and gradient-noise scale.

**Predictor.** Ridge regression from epoch-5 features (log-spectrum quantiles, RankMe, $\alpha$, their epoch 1→5 slopes) to final top-1, evaluated leave-one-learning-rate-out so the model must extrapolate across the confounder.

**Control arm.** (a) epoch-5 RankMe alone, as the trivial predictor; (b) epoch-5 loss alone; (c) final-checkpoint RankMe, as the upper bound.

**Deciding number.** Held-out Spearman $\rho$ between predicted and actual final top-1. The forecasting claim is supported if $\rho \ge 0.90$ while control (a) gives $\rho \le 0.60$ on the same splits. If both land in $[0.6, 0.9]$, the early signal adds nothing beyond current rank and the method variant should be declared negative.

## 9. Key References

- **[Foundational]** Li Jing, Pascal Vondrick? — *see corrected:* Li Jing, Pascal Vincent, Yann LeCun, Yuandong Tian. *Understanding Dimensional Collapse in Contrastive Self-Supervised Learning.* ICLR, 2022. — arXiv:2110.09348
- **[Foundational]** Olivier Roy, Martin Vetterli. *The Effective Rank: A Measure of Effective Dimensionality.* EUSIPCO, 2007.
- **[SOTA]** Quentin Garrido, Randall Balestriero, Laurent Najman, Yann LeCun. *RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank.* ICML, 2023. — arXiv:2210.02885
- **[SOTA]** Vimal Thilak, Chen Huang, Omid Saremi, Laurent Dinh, Hanlin Goh, Preetum Nakkiran, Joshua Susskind, Etai Littwin. *LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures.* ICLR, 2024.
- **[Theory]** Liu Ziyin, Ekdeep Singh Lubana, Masahito Ueda, Hidenori Tanaka. *What shapes the loss landscape of self-supervised learning?* ICLR, 2023.
- **[Theory]** Yuandong Tian, Xinlei Chen, Surya Ganguli. *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs.* ICML, 2021.
- **[Theory]** James B. Simon, Maksis Knutins, Liu Ziyin, Daniel Geisz, Abraham J. Fetterman, Joshua Albrecht. *On the Stepwise Nature of Self-Supervised Learning.* ICML, 2023.
- **[Method]** Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny. *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* ICML, 2021.
- **[Method]** Adrien Bardes, Jean Ponce, Yann LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR, 2022.
- **[Related]** Vardan Papyan, X. Y. Han, David L. Donoho. *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training.* PNAS, 2020.
- **[Related]** Tongyao Feng et al. *Rank Diminishing in Deep Neural Networks.* NeurIPS, 2022.
- **[Survey]** Randall Balestriero et al. *A Cookbook of Self-Supervised Learning.* Technical report, 2023. — arXiv:2304.12210

## 10. Worked Example

Take two SimCLR runs, ResNet-50, ImageNet-1k, $k = 2048$, 100 epochs, differing only in weight decay: $\mathrm{wd} = 10^{-6}$ (run A) and $\mathrm{wd} = 10^{-4}$ (run B).

Suppose the measured endpoints are:

| | $\mathrm{RankMe}$ (proj.) | $c_{10^{-4}}$ | $\alpha$ | top-1 |
|---|---|---|---|---|
| A | 410 | 0.31 | 0.9 | 66.1 |
| B | 190 | 0.62 | 1.4 | 63.4 |

Read naively: rank halves, accuracy drops 2.7 points, so collapse costs accuracy.

Now compute $c_\tau$ at $\tau = 10^{-6}$ instead of $10^{-4}$: the counts become 0.08 and 0.19. The *ratio* survives, but the absolute "number of collapsed dimensions" changed by a factor of four with a threshold nobody has justified. Compute $\alpha$ over $i \in [1, 64]$ rather than $[1, 256]$: for power-law-plus-noise spectra the fitted exponent typically shifts by tens of percent, enough to flip the "$\alpha \approx 1$ is best" ordering between two nearby runs.

Then the causal problem. Weight decay is the only difference, and weight decay independently changes the encoder's effective capacity. Nothing in this pair separates "B is worse because its embeddings are rank-190" from "B is worse because it is over-regularised, and rank-190 is a symptom". To break it you need a third arm: run B's weight decay with a rank-restoring intervention (whitening the projector output, or VICReg's covariance term at matched invariance weight) and check whether accuracy returns to 66. That arm is what nobody has published at this scale.

The obstruction is visible in the table: every column moves together, one knob moved them, and the measurement that defines the phenomenon has two free parameters ($\tau$, the fit window) with no principled setting.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*