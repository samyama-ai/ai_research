---
id: 19-evaluation/binning-robust-calibration-metrics
title: "Calibration Metrics Robust to Binning Choices"
topic: 19-evaluation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Metrics Robust to Binning Choices

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/binning-robust-calibration-metrics` · **Status:** partially-solved

## 1. Problem Statement

Expected Calibration Error (ECE) is the default number reported for probabilistic forecast quality. It is not a property of the model alone: it is a property of the model *and* a binning scheme (bin count, equal-width vs equal-mass, top-label vs full-vector) *and* the sample size. Changing the bin count from 5 to 100 on the same predictions can change the reported error by a factor of 4 and can reverse the ranking of two models.

Three variants, of different difficulty:

- **Measurement.** Given a held-out sample of $n$ prediction/label pairs, output a scalar that estimates the model's true distance from calibration, with no free binning hyperparameter, and with a stated bias and variance. *Largely solved as of 2023 — see §3.*
- **Method.** Make that scalar cheap, differentiable, defined for $K$-class vector-valued (not just top-label) predictions, and reportable with a confidence interval that practitioners actually publish. *Partly open.*
- **Theory.** Characterise which calibration measures are *consistent* — polynomially related to the true distance from calibration — and establish minimax estimation/testing rates without smoothness assumptions on the miscalibration curve. *Partly open.*

Solving it means: a metric $M$ such that $M(\hat f, S)$ depends only on $(\hat f, S)$, converges to a quantity with a distribution-free operational meaning, and produces the same model ranking under any reasonable analyst.

## 2. Formal Setting

Let $(X, Y) \sim \mathcal{D}$ with $Y \in \{0,1\}$ (binary case) and a predictor $f: \mathcal{X} \to [0,1]$. Write $Z = f(X)$. Perfect calibration:

$$\mathbb{E}[Y \mid Z = z] = z \quad \text{for } \mathcal{D}\text{-almost every } z.$$

**True calibration error** (the estimand, not measurable directly):
$$\mathrm{CE}_p(f) = \big(\mathbb{E}_Z\big|\mathbb{E}[Y\mid Z] - Z\big|^p\big)^{1/p}, \quad p \in \{1,2\}.$$

**Plugin binned estimator** (what is actually computed). Partition $[0,1]$ into bins $B_1,\dots,B_B$; let $n_b = |\{i : z_i \in B_b\}|$, $\bar z_b = n_b^{-1}\sum_{i \in B_b} z_i$, $\bar y_b = n_b^{-1}\sum_{i \in B_b} y_i$:
$$\widehat{\mathrm{ECE}}_B = \sum_{b=1}^{B} \frac{n_b}{n}\,\big|\bar y_b - \bar z_b\big|.$$

Free choices: $B$; equal-width ($B_b = [\tfrac{b-1}{B}, \tfrac{b}{B})$) vs equal-mass ($n_b \approx n/B$); for $K$ classes, whether $Z = \max_k \hat p_k$ with $Y = \mathbb{1}[\hat y = y]$ (top-label / confidence calibration), per-class one-vs-rest, or the full simplex (canonical calibration).

**Distance to calibration** (Błasiok et al., STOC 2023), the assumption-free estimand:
$$\underline{\mathrm{dCE}}(f) = \inf_{g \text{ perfectly calibrated}} \mathbb{E}\,|f(X) - g(X)|.$$

**Smooth ECE**: convolve the reliability curve with a kernel of bandwidth $\sigma$ and pick $\sigma$ by a fixed-point rule, $\mathrm{smECE}(f) = \mathrm{smECE}_{\sigma^*}$ with $\sigma^*$ the largest $\sigma$ satisfying $\mathrm{smECE}_\sigma \ge \sigma$.

**Assumptions and their violations.**
- *i.i.d. held-out sample.* Violated under distribution shift and under repeated test-set reuse — the calibration set is usually the same one used to fit temperature.
- *Continuous, non-atomic $Z$.* Violated: softmax confidences on ImageNet pile up near $1$, so equal-width bins are near-empty in $[0.1, 0.9]$ and one bin holds most mass.
- *$B$ fixed independent of data.* Violated whenever $B$ is tuned, including implicitly by reporting the prettiest reliability diagram.
- *Top-label calibration is the target.* Almost never the decision-relevant quantity; it is reported because it is $1$-dimensional.

## 3. State of the Art

**Established.**
- Plugin $\widehat{\mathrm{ECE}}_B$ is *downward*-biased for $\mathrm{CE}_1$ when bins are coarse (within-bin miscalibration cancels) and *upward*-biased when bins are fine (finite-sample noise in $\bar y_b$ does not cancel under $|\cdot|$). Kumar, Liang, Ma (NeurIPS 2019) formalise this and give a debiased estimator plus a variance-reduced *scaling-binning* recalibrator with sample-complexity guarantees.
- Vaicenavicius et al. (AISTATS 2019) prove that any fixed binning yields a *lower* bound on calibration error and propose consistency-resampling hypothesis tests.
- Widmann, Lindsten, Zachariah (NeurIPS 2019) give kernel-based calibration errors (MMD-type) for the full $K$-simplex with unbiased estimators and asymptotic tests — no binning at all.
- Błasiok, Gopalan, Hu, Nakkiran (STOC 2023) prove ECE is *not* a consistent calibration measure: predictors with $\mathrm{ECE} = \Omega(1)$ can be $O(\epsilon)$-close to perfectly calibrated in $\ell_1$. They give a class of consistent measures polynomially sandwiched around $\underline{\mathrm{dCE}}$.
- Błasiok & Nakkiran (NeurIPS 2023) show $\mathrm{smECE}$ is consistent, hyperparameter-free, and computable in near-linear time; it also yields a principled reliability diagram.
- Lee, Huang, Hassani, Dobriban (T-Cal, JMLR 2023) give a minimax-optimal test for calibration under Hölder-$\alpha$ smoothness of the miscalibration curve, with detection boundary $n^{-2\alpha/(4\alpha+1)}$ (so $n^{-2/5}$ for Lipschitz).
- Gruber & Buettner (NeurIPS 2022) show proper-score decompositions (Brier/log-loss into calibration + refinement) give lower estimation error for the calibration term than direct ECE binning.

**Claimed but unablated / benchmark-only.**
- Adaptive-binning ACE (Nixon et al., CVPR-W 2019) and spline calibration error (Gupta et al., ICLR 2021) are reported to change model rankings versus 15-bin ECE, but the *correct* ranking is not established on those benchmarks — they show disagreement, not accuracy.
- $\mathrm{ECE}^{\mathrm{sweep}}$ (Roelofs et al., AISTATS 2022) — pick the largest $B$ for which the reliability curve stays monotone — reduces bias empirically across thousands of trained models; its bias is not bounded theoretically.
- KDE-ECE (Zhang, Kailkhura, Han, ICML 2020) reduces bias on synthetic ground truth but reintroduces a bandwidth hyperparameter; the claim that a fixed bandwidth rule transfers across datasets is not ablated.

## 4. What Is Known

- **Guo et al. (ICML 2017)** set the de facto standard at $B=15$ equal-width bins, top-label. DenseNet-161 on ImageNet: ECE $\approx 5.7\%$ pre-temperature, $\approx 2\%$ post; the *reported* improvement is partly the estimator's bias floor.
- **Kumar, Liang, Ma (NeurIPS 2019):** on CIFAR-10 and ImageNet with Platt-scaled recalibrators ($n \approx 1{,}000$–$10{,}000$ calibration points), the true $\ell_2$ calibration error was measured at roughly $2\times$ the plugin binned estimate; plugin ECE was reported as "near zero" for models with materially non-zero error.
- **Roelofs et al. (AISTATS 2022):** across $\sim$6 image/text datasets and thousands of model checkpoints, equal-width 15-bin ECE bias is comparable in magnitude to the calibration differences between competing models; bin-count choice alone changes pairwise model ordering in a non-trivial fraction of pairs.
- **Noise floor, closed form.** For a *perfectly calibrated* model with $n$ points spread over $B$ bins, $\mathbb{E}[\widehat{\mathrm{ECE}}_B] \approx \sqrt{2\bar p(1-\bar p)B/(\pi n)}$ — it grows as $\sqrt{B}$ and never reaches $0$.
- **Minderer et al. (NeurIPS 2021):** modern architectures (ViT, MLP-Mixer) are better calibrated than ResNets at matched accuracy on ImageNet — a finding that survives across binning schemes, which is why it is credible.

## 5. What Is Not Known

- **Theoretically open.** Minimax rates for *estimating* (not testing) $\underline{\mathrm{dCE}}$ without smoothness assumptions. T-Cal's optimality is for testing under Hölder classes; the estimation counterpart, and whether smECE attains it, is unproven. Also open: the tight exponent in the sandwich $\underline{\mathrm{dCE}} \lesssim \mathrm{smECE} \lesssim \sqrt{\underline{\mathrm{dCE}}}$ — is the square root necessary?
- **Theoretically open.** A consistent, computable analogue of $\underline{\mathrm{dCE}}$ for *canonical* $K$-class calibration with sample complexity sub-exponential in $K$. Kernel tests exist; a distance-to-calibration for the simplex does not.
- **Empirically open.** Whether adopting smECE would change published conclusions. Nobody has re-scored a large public model zoo (say, 500+ ImageNet checkpoints) under {15-bin ECE, equal-mass ECE, debiased ECE, smECE, KDE-ECE} and reported rank correlations. The data and code exist; the run has not been done at that scale.
- **Methodologically blocked.** *Which* calibration a benchmark should report. Top-label ECE is not the decision-relevant quantity for selective prediction, abstention, or downstream expected-cost decisions, yet it is what leaderboards carry. Until the target is fixed, "robust to binning" is under-specified.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth combined with a bias/variance trade-off that has no interior optimum under the $\ell_1$ contrast**. $\mathbb{E}[Y \mid Z=z]$ is never observed; every estimator must smooth, and smoothing is exactly the binning choice. Coarse bins average away real miscalibration (bias down); fine bins let Bernoulli noise pass through the absolute value (bias up, floor $\propto \sqrt{B/n}$). There is no unbiased plugin estimator of $\mathrm{CE}_1$ from binary labels, because $|\cdot|$ is not a polynomial in the bin mean.

Two consequences. First, on real prediction distributions — mass concentrated near $1$ — the bias magnitude is on the order of the between-model differences being reported, so the metric can rank noise. Second, because there is no ground truth, "which estimator is right" cannot be settled on real data at all; validation is confined to synthetic distributions where $\mathbb{E}[Y\mid Z]$ is known by construction, and those distributions look nothing like a softmax.

## 7. Current Research (as of 2026)

- **Consistent calibration measures.** Błasiok, Gopalan, Hu, Kim, Nakkiran and collaborators — extending distance-to-calibration to multiclass, regression, and to *calibration for downstream decisions* (decision-calibration / omniprediction links). Active and well-founded.
- **smECE adoption.** `relplot` (Błasiok & Nakkiran) is the reference implementation; uptake in LLM-evaluation stacks is partial. *(frontier — verify)* whether major leaderboards have switched.
- **LLM confidence calibration.** Verbalised confidence and token-probability calibration for language models re-import the 10/15-bin habit wholesale; the binning-bias literature has largely not propagated there. *(frontier — verify)*
- **Proper-score decompositions.** Gruber, Buettner, and the forecasting/meteorology community (Bröcker-style CORP/isotonic decompositions, Dimitriadis–Gneiting–Jordan, PNAS 2021) — isotonic regression as a binning-free reliability estimator with automatic bin selection.
- **Sequential/online calibration** under distribution shift, where the i.i.d. estimand itself is wrong.

## 8. Concrete Next Experiment

**Question:** does the binning choice change published model rankings, and does smECE remove that dependence?

**Scale.** 500 publicly available ImageNet-1k checkpoints (timm) plus 100 CIFAR-100 checkpoints. Score each on the 50k validation set with a fixed 25k/25k calibration/evaluation split, $n = 25{,}000$.

**Arms.** (1) 15-bin equal-width top-label ECE — *the control*, the currently published number; (2) equal-mass 15-bin; (3) $B \in \{5,10,50,100\}$ equal-width; (4) debiased ECE (Kumar et al. 2019); (5) $\mathrm{ECE}^{\mathrm{sweep}}$; (6) smECE.

**Ground-truth arm.** Re-run all six on a synthetic mirror: resample labels from a *known* $\mathbb{E}[Y\mid Z]$ fitted by isotonic regression to each real model's confidences, so true $\mathrm{CE}_1$ and $\underline{\mathrm{dCE}}$ are known exactly.

**Deciding number.** Kendall's $\tau$ between each arm's model ranking and the synthetic-mirror ground-truth ranking, plus the *spread* $\tau_{\min}$ over arms 1–3 (bin-choice sensitivity). Decision rule: if $\tau_{\min} < 0.8$ for arms 1–3 while smECE achieves $\tau \ge 0.95$, the binning-dependence is material and the field should switch metric. If all arms exceed $\tau = 0.95$, binning choice is a non-issue at $n = 25{,}000$ and the concern is confined to small calibration sets. Cost: single GPU-day for logits, minutes for scoring.

## 9. Key References

- **[Foundational]** Naeini, Cooper, Hauskrecht. *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI, 2015.
- **[Foundational]** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** Vaicenavicius, Widmann, Andersson, Lindsten, Roll, Schön. *Evaluating Model Calibration in Classification.* AISTATS, 2019. — arXiv:1902.06977
- **[SOTA]** Kumar, Liang, Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[SOTA]** Widmann, Lindsten, Zachariah. *Calibration Tests in Multi-class Classification: A Unifying Framework.* NeurIPS, 2019. — arXiv:1910.11385
- **[SOTA]** Błasiok, Gopalan, Hu, Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC, 2023. — arXiv:2211.16886
- **[SOTA]** Błasiok, Nakkiran. *Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing.* NeurIPS / ICLR-track, 2023. — arXiv:2309.12236
- **[SOTA]** Lee, Huang, Hassani, Dobriban. *T-Cal: An Optimal Test for the Calibration of Predictive Models.* JMLR, 2023.
- **[SOTA]** Roelofs, Cain, Shlens, Mozer. *Mitigating Bias in Calibration Error Estimation.* AISTATS, 2022. — arXiv:2012.08668
- **[SOTA]** Zhang, Kailkhura, Han. *Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning.* ICML, 2020.
- **[SOTA]** Gruber, Buettner. *Better Uncertainty Calibration via Proper Scores for Classification and Beyond.* NeurIPS, 2022.
- **[Survey]** Nixon, Dusenberry, Zhang, Jerfel, Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops, 2019. — arXiv:1904.01685
- **[Survey]** Minderer, Djolonga, Romijnders, Hubis, Zhai, Houlsby, Tran, Lucic. *Revisiting the Calibration of Modern Neural Networks.* NeurIPS, 2021. — arXiv:2106.07998
- **[Related]** Dimitriadis, Gneiting, Jordan. *Stable Reliability Diagrams for Probabilistic Classifiers.* PNAS, 2021.

## 10. Worked Example

Take $n = 1{,}000$ held-out points. **Model A** is *perfectly calibrated*, with confidences spread uniformly on $[0.5, 1]$ (mean $\bar p = 0.75$, so $\bar p(1-\bar p) = 0.1875$). Its true $\mathrm{CE}_1 = 0$.

With $B$ equal-mass bins, each holds $m = 1000/B$ points, and $\bar y_b - \bar z_b$ is approximately $\mathcal{N}(0, 0.1875/m)$. Since $\mathbb{E}|N(0,\sigma^2)| = \sigma\sqrt{2/\pi}$:

| $B$ | $m$ | $\sigma$ | $\mathbb{E}[\widehat{\mathrm{ECE}}_B]$ |
|---|---|---|---|
| 5 | 200 | 0.0306 | **2.4%** |
| 15 | 67 | 0.0529 | **4.2%** |
| 100 | 10 | 0.1369 | **10.9%** |

A model with *zero* calibration error is reported at anywhere from 2.4% to 10.9% depending on a choice the analyst makes freely.

Now **Model B**, genuinely miscalibrated: $\mathbb{E}[Y\mid Z] = Z + 0.03$ everywhere, so $\mathrm{CE}_1 = 3\%$. At the standard $B=15$, $\sigma = 0.0529$ and
$$\mathbb{E}\big|N(0.03, 0.0529^2)\big| = \sigma\sqrt{\tfrac{2}{\pi}}e^{-\mu^2/2\sigma^2} + \mu\big(1-2\Phi(-\mu/\sigma)\big) = 0.0359 + 0.0129 = \mathbf{4.9\%}.$$

So at the field-standard setting: perfectly calibrated Model A scores 4.2%, 3%-miscalibrated Model B scores 4.9%. The gap is $0.7$ points where the true gap is $3.0$ — and the per-run standard deviation of $\widehat{\mathrm{ECE}}_{15}$ at $n=1000$ is itself around $0.4$ points, so a single evaluation cannot separate them. Worse, if A is evaluated at $B=100$ and B at $B=5$ (both defensible), A reports 10.9% and B reports 3.7%: **the ranking inverts.**

The obstruction is visible here: the reported number is dominated by an estimator artifact whose size, $\sqrt{2\bar p(1-\bar p)B/(\pi n)}$, is set by the analyst rather than the model. smECE removes the free parameter by fixing the bandwidth via the $\sigma^* = \mathrm{smECE}_{\sigma^*}$ rule, and its consistency guarantee bounds the artifact in terms of $\underline{\mathrm{dCE}}$ — but the residual $\sqrt{\cdot}$ slack in that sandwich means it still cannot certify "A is better than B by 3 points" from $n=1000$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*