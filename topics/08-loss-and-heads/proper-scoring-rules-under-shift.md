---
id: 08-loss-and-heads/proper-scoring-rules-under-shift
title: "Proper Scoring Rules That Also Calibrate Under Distribution Shift"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Proper Scoring Rules That Also Calibrate Under Distribution Shift

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/proper-scoring-rules-under-shift` · **Status:** open

## 1. Problem Statement

A proper scoring rule is a loss whose unique minimiser is the true conditional distribution. Log-loss and Brier are proper. Properness is a statement about one distribution: minimise the population risk **under $P$** and you recover $P(y\mid x)$, which is calibrated under $P$. It says nothing about $Q \neq P$. In practice models trained with log-loss lose calibration sharply under shift while accuracy degrades more gracefully.

Three variants, different difficulty:

- **Theory.** Does there exist a loss $\ell$, proper under $P$, whose empirical minimiser over a hypothesis class $\mathcal{H}$ has calibration error bounded on a shift family $\mathcal{Q}$ by a quantity that does not scale with the worst-case density ratio? Open.
- **Method.** Build a training objective (loss term, head parameterisation, or post-hoc map) that empirically holds calibration error flat across corruption severity without paying accuracy. Partially addressed; no method holds ECE flat.
- **Measurement.** Estimate calibration error on the target domain when target labels are scarce or absent, with an estimator whose bias is smaller than the effect being measured. Currently the binding constraint.

Solving it means: a loss, a shift family, a theorem bounding target calibration, and an experiment where the bound is not vacuous at a severity where the baseline's ECE has tripled.

## 2. Formal Setting

Inputs $x \in \mathcal{X}$, labels $y \in \mathcal{Y} = \{1,\dots,K\}$. A model outputs $f(x) \in \Delta^{K-1}$ — measured as the post-softmax vector, at the final checkpoint, in inference mode (dropout off, BN in eval).

A scoring rule $S(f(x), y)$ is **proper** for $P$ if $\mathbb{E}_{y\sim P(\cdot\mid x)} S(P(\cdot\mid x), y) \le \mathbb{E}_{y\sim P(\cdot\mid x)} S(q, y)$ for all $q$, strictly for $q \neq P(\cdot\mid x)$. Log-loss: $S = -\log f_y(x)$. Brier: $S = \|f(x) - e_y\|_2^2$.

**Confidence calibration error.** With $c(x) = \max_k f_k(x)$ and $\hat y(x) = \arg\max_k f_k(x)$,
$$\mathrm{CE}_Q = \mathbb{E}_{Q}\big|\Pr[\hat y = y \mid c(x)] - c(x)\big|.$$
Measured as binned ECE with $M$ equal-mass bins over $n$ target points:
$$\widehat{\mathrm{ECE}} = \sum_{m=1}^{M} \tfrac{|B_m|}{n}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.$$
This estimator is biased downward and the bias grows with $M/n$ (Kumar et al. 2019; Roelofs et al. 2022) — the first violated assumption: "ECE" as reported is a plug-in for a quantity it does not consistently estimate.

**Shift family.** $\mathcal{Q} = \{Q : D(Q\,\|\,P) \le \rho\}$ for a divergence $D$, or a covariate-shift family $Q(x,y) = w(x)P(x)P(y\mid x)$ with $\|w\|_\infty \le B$. Measured in practice as a *fixed corruption suite* (CIFAR-10-C, ImageNet-C, WILDS splits), not a divergence ball — the second violated assumption: no reported $\rho$ or $B$ corresponds to the benchmark used.

**Objective.** Find $\ell$ and $\hat f = \arg\min_{f\in\mathcal H}\hat{\mathbb E}_P \ell(f(x),y)$ with
$$\sup_{Q \in \mathcal{Q}} \mathrm{CE}_Q(\hat f) \le \varepsilon(\rho, n, \mathcal{H}),$$
$\varepsilon$ sublinear in $\rho$ and not of the form $B \cdot \mathrm{CE}_P$ (that bound is trivial and vacuous at $B \gtrsim 10$).

**Assumptions known violated:** covariate shift ($P(y\mid x)$ constant) fails for ImageNet-C at high severity where human labels also degrade; exchangeability fails for temporal shift; the shift family is unknown at train time.

## 3. State of the Art

**Theory.** Błasiok, Gopalan, Hu & Nakkiran, *When Does Optimizing a Proper Loss Yield Calibration?* (NeurIPS 2023): a predictor that cannot be improved by more than $\epsilon^2$ under any Lipschitz post-processing has distance-to-calibration $O(\epsilon)$ — properness plus *local optimality* gives calibration, on the training distribution. Their earlier *A Unifying Theory of Distance from Calibration* (STOC 2023) gives a consistent, estimable calibration distance replacing binned ECE. Neither extends to $Q \neq P$. Multicalibration (Hébert-Johnson, Kim, Reingold & Rothblum, ICML 2018) gives calibration on every set in a family $\mathcal{C}$ with sample complexity polynomial in $|\mathcal{C}|$ and $1/\alpha$; if $Q$'s reweighting lies in the span of $\mathcal{C}$ this transfers, which is the only clean shift-transfer theorem in the area. Omniprediction (Gopalan et al., ITCS 2022) links multicalibration to simultaneous optimality across all convex losses.

**Empirical.** Deep ensembles remain the strongest across the corruption suites (Ovadia et al., NeurIPS 2019). Wald et al., *On Calibration and Out-of-Domain Generalization* (NeurIPS 2021), show multi-domain calibration implies invariance and add a calibration penalty (CLOvE); the gains are reported on a small number of domains and the penalty's contribution is not ablated against simply tuning temperature per domain. Yu, Bates, Ma & Jordan, *Robust Calibration with Multi-Domain Temperature Scaling* (NeurIPS 2022), fit a shift-aware temperature — established for the affine-recalibration family, unestablished beyond it. Minderer et al. (NeurIPS 2021) report ViT/MLP-Mixer better calibrated in and out of distribution than convnets: a benchmark number, with architecture, data scale and augmentation confounded.

**Claimed but unablated:** focal loss "improves calibration" (Mukhoti et al., NeurIPS 2020) — the effect is partly a confidence-shrinkage that a single temperature reproduces; on-shift gains are not separated from that.

## 4. What Is Known

- **Decomposition.** Every proper score decomposes as calibration $+$ refinement (Bröcker, *Reliability, sufficiency, and the decomposition of proper scores*, QJRMS 2009). So a proper loss penalises miscalibration on $P$ only, and can trade calibration for sharpness under any reweighting.
- **Post-hoc recalibration does not transfer.** Ovadia et al. (2019): on CIFAR-10-C, temperature scaling fitted in-distribution leaves ECE rising from ~0.02–0.03 at severity 0 to ~0.10–0.15 at severity 5 for ResNet-20/VGG; deep ensembles (5 members, ResNet-20) roughly halve that. On ImageNet-C, ECE reaches ~0.15–0.25 at severity 5. Scale: CIFAR-10 (50k train) and ImageNet-1k.
- **Modern nets are overconfident in-distribution.** Guo et al. (ICML 2017): ResNet-110 on CIFAR-100 has ECE ≈ 0.16 (15 bins); one temperature reduces it below 0.03. Temperature scaling is a 1-parameter fix for a 1-dimensional failure.
- **Binned ECE is biased.** Kumar, Liang & Ma (NeurIPS 2019) show binning underestimates true calibration error and give a debiased estimator; Roelofs et al. (AISTATS 2022) show bin count changes the ranking of methods on ImageNet-scale comparisons.
- **Conformal transfers under known shift.** Tibshirani et al. (NeurIPS 2019) give exact coverage under covariate shift with *known* likelihood ratio $w$; Gibbs & Candès (NeurIPS 2021) give online coverage with no shift assumption but no conditional validity. Coverage is weaker than calibration.

## 5. What Is Not Known

- **Theoretically open.** Whether any loss proper on $P$ can bound $\sup_{Q\in\mathcal{Q}}\mathrm{CE}_Q$ non-trivially without access to $\mathcal{Q}$. No impossibility theorem exists either; the obvious construction (adversarial reweighting inside a $\chi^2$ ball) has not been pushed to a matching lower bound. Also open: whether the Błasiok et al. local-optimality-implies-calibration result has a multi-domain analogue.
- **Empirically open.** Whether the multicalibration sample complexity is affordable at ImageNet scale with $\mathcal{C}$ = the shift-relevant group family. Runnable today; unrun.
- **Methodologically blocked.** Target-domain calibration error with few or no target labels. Every reported "OOD ECE" uses labelled target data, which is precisely the resource a deployed model lacks.

## 6. Why It Is Hard

**Non-identifiability plus a confounded measurement.** Under covariate shift the target calibration error is a $w$-reweighted functional of the same predictor; without observing $w$ or target labels it is not identified from source data — two predictors indistinguishable on $P$ can have $\mathrm{CE}_Q$ differing by $\Theta(1)$. Layered on top: the reported quantity, binned ECE, has bias of the same order as the differences between competing methods (Roelofs et al. 2022), so method rankings under shift are not stable to the estimator's bin count. The evaluation does not measure the thing it names. Compute is not the obstruction — CIFAR-10-C sweeps are cheap.

## 7. Current Research (as of 2026)

- **Calibration distance as the target quantity**, replacing binned ECE — Błasiok/Gopalan/Nakkiran and follow-ups; the open piece is a shift-robust version.
- **Multicalibration as shift insurance** — Kim, Rothblum, Reingold and collaborators; groups defined by learned features rather than protected attributes *(frontier — verify)*.
- **Multi-domain / group temperature scaling** — Jordan and Bates groups (Berkeley), extending Yu et al. 2022 to non-affine maps.
- **LLM verbalised-confidence calibration under domain shift**, where the "head" is a token distribution and shift is prompt-distribution shift; largely benchmark-driven, weak theory *(frontier — verify)*.
- **Distributionally robust proper losses** — minimising worst-case score over a $\chi^2$ or Wasserstein ball; known to improve tail risk, unclear whether it improves calibration rather than uniformly shrinking confidence.

## 8. Concrete Next Experiment

**Question:** does any training-time loss beat *in-distribution temperature scaling plus a single global confidence shrink* on target calibration, once the estimator bias is removed?

**Scale.** CIFAR-10 and CIFAR-100, WRN-28-10, 5 seeds each; evaluate on CIFAR-10-C/100-C, 19 corruptions × 5 severities. Total ≈ 40 GPU-hours on one A100. Then replicate the winner on ImageNet-1k → ImageNet-C with ResNet-50, 3 seeds.

**Arms.** (1) cross-entropy, (2) cross-entropy + temperature scaling on a held-out in-distribution split, (3) focal loss $\gamma=3$, (4) CLOvE penalty (Wald et al. 2021), (5) multi-domain temperature scaling (Yu et al. 2022) using severity-1 corruptions as extra source domains, (6) 5-member deep ensemble.

**Control arm.** Arm 2 with one extra scalar: temperature refit to minimise ECE at severity 3, then applied at all severities. Any method that does not beat this 2-parameter control is producing confidence shrinkage, not shift-robust calibration.

**Deciding number.** $\Delta = \mathrm{dCE}(\text{method}, \text{sev }5) - \mathrm{dCE}(\text{control}, \text{sev }5)$, where dCE is the debiased / distance-to-calibration estimator (Kumar et al. 2019; Błasiok et al. 2023), not binned ECE, averaged over 19 corruptions with seed-level bootstrap CIs, at matched top-1 accuracy within 0.5 pp. **Decision rule: a method counts as progress only if $\Delta \le -0.02$ with the 95% CI excluding zero.** Current expectation from published curves: every training-time loss lands in $\Delta \in [-0.01, +0.01]$ and only the ensemble clears the bar — which would localise the problem to *ensembling*, not *loss design*.

## 9. Key References

- **[Foundational]** Tilmann Gneiting, Adrian Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102(477), 2007.
- **[Foundational]** Jochen Bröcker. *Reliability, sufficiency, and the decomposition of proper scores.* Quarterly Journal of the Royal Meteorological Society 135, 2009.
- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. — arXiv:1706.04599
- **[SOTA-empirical]** Yaniv Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS 2019. — arXiv:1906.02530
- **[SOTA-theory]** Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran. *When Does Optimizing a Proper Loss Yield Calibration?* NeurIPS 2023.
- **[SOTA-theory]** Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC 2023.
- **[Foundational]** Úrsula Hébert-Johnson, Michael Kim, Omer Reingold, Guy Rothblum. *Multicalibration: Calibration for the (Computationally-Identifiable) Masses.* ICML 2018.
- **[SOTA]** Yoav Wald, Amir Feder, Daniel Greenfeld, Uri Shalit. *On Calibration and Out-of-Domain Generalization.* NeurIPS 2021. — arXiv:2102.10395
- **[SOTA]** Yachong Yu, Stephen Bates, Yi Ma, Michael I. Jordan. *Robust Calibration with Multi-Domain Temperature Scaling.* NeurIPS 2022.
- **[Method]** Ananya Kumar, Percy Liang, Tengyu Ma. *Verified Uncertainty Calibration.* NeurIPS 2019. — arXiv:1909.10155
- **[Method]** Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, Michael Mozer. *Mitigating Bias in Calibration Error Estimation.* AISTATS 2022.
- **[Related]** Ryan Tibshirani, Rina Foygel Barber, Emmanuel Candès, Aaditya Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019. — arXiv:1904.06019
- **[Survey]** Matthias Minderer et al. *Revisiting the Calibration of Modern Neural Networks.* NeurIPS 2021. — arXiv:2106.07998

## 10. Worked Example

Binary problem, one feature $x\in\{a,b\}$. Source $P(x{=}a)=0.9$. True $P(y{=}1\mid a)=0.9$, $P(y{=}1\mid b)=0.5$.

Two predictors:
- $f^\star$: outputs $0.9$ on $a$, $0.5$ on $b$. Perfect.
- $\tilde f$: outputs $0.9$ on $a$, $0.7$ on $b$.

Brier risk under $P$: $f^\star$ scores $0.9(0.09) + 0.1(0.25) = 0.1060$. $\tilde f$ scores $0.9(0.09) + 0.1\big(0.5(0.09)+0.5(0.49)\big) = 0.081 + 0.029 = 0.1100$. Gap: $0.0040$.

With $n = 2{,}000$ source samples the per-sample Brier standard deviation is about $0.20$, so the standard error of the risk gap is $\approx 0.20/\sqrt{2000} \approx 0.0045$. **The gap is inside one standard error.** Properness identifies $f^\star$ in the population; at realistic $n$ it does not separate the two.

Now shift to $Q(x{=}b) = 0.9$ (covariate shift, $w(b)=9$). Target calibration error: $f^\star$ has $\mathrm{CE}_Q = 0$; $\tilde f$ has $\mathrm{CE}_Q = 0.9 \times |0.7-0.5| = 0.18$.

A $0.004$ source-risk difference — statistically invisible at $n{=}2{,}000$ — becomes a $0.18$ target calibration error. Increasing $n$ helps here, but in the real case $b$ is a *rare region of a high-dimensional input space*: the effective sample count in the region that dominates $Q$ stays at hundreds regardless of total $n$, and the source risk gap stays inside its own noise. That is the obstruction in miniature: the source-proper loss has no gradient signal where the target puts its mass, and no amount of properness supplies one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*