---
id: 33-uncertainty-calibration/calibration-error-finite-sample-estimation
title: "Calibration Error Estimation from Finite Samples"
topic: 33-uncertainty-calibration
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Error Estimation from Finite Samples

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/calibration-error-finite-sample-estimation` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a held-out sample $(x_i, y_i)_{i=1}^n$ and a probabilistic predictor $f$, report a number that says how miscalibrated $f$ is — with a confidence interval, and with the property that the number is a function of $f$ and the data distribution, not of the estimator's hyperparameters.

Three variants, with different difficulty:

- **Measurement.** Which population functional should be reported? Expected Calibration Error (ECE) is the default, but it is discontinuous in $f$: two predictors that differ by $10^{-6}$ in sup-norm can have ECE $0$ and ECE $1/2$. So "estimate ECE accurately" is a well-posed statistical problem whose answer is not a useful quantity.
- **Method.** Given a fixed functional, build an estimator with vanishing bias and a valid finite-sample interval. Binned plugin ECE fails: its bias at typical $n$ is the same size as the effect being measured.
- **Theory.** Establish minimax rates for estimating and for *testing* calibration, and the sample complexity of distinguishing "calibrated" from "$\varepsilon$-far from calibrated" without smoothness assumptions on the true calibration curve.

Solved would mean: a functional that is Lipschitz in $f$, an estimator whose bias is provably $o(\varepsilon)$ at the $n$ practitioners actually have ($n \in [10^3, 10^5]$), and a distribution-free interval. None of the three is currently in hand simultaneously.

## 2. Formal Setting

Labels $y \in \{0,1\}$ (binary case; multiclass reduces via top-label or class-wise projections), features $x \sim \mathcal{D}$, predictor $f: \mathcal{X} \to [0,1]$. Write $Z = f(X)$ and the **calibration curve**

$$c(v) = \mathbb{E}[Y \mid Z = v].$$

$f$ is perfectly calibrated iff $c(v) = v$ almost surely. The $\ell_p$ calibration error is

$$\mathrm{CE}_p(f) = \left(\mathbb{E}_Z \big| c(Z) - Z \big|^p\right)^{1/p}, \qquad \mathrm{ECE} = \mathrm{CE}_1 .$$

**As actually measured.** $c$ is never observed. The reported statistic partitions $[0,1]$ into bins $B_1,\dots,B_B$ (equal-width, or equal-mass with $n/B$ points each) and computes

$$\widehat{\mathrm{ECE}}_B = \sum_{b=1}^{B} \frac{n_b}{n} \left| \bar{y}_b - \bar{z}_b \right|, \quad \bar y_b = \tfrac{1}{n_b}\!\!\sum_{i: z_i \in B_b}\!\! y_i, \ \ \bar z_b = \tfrac{1}{n_b}\!\!\sum_{i: z_i \in B_b}\!\! z_i .$$

Two errors are conflated in $\widehat{\mathrm{ECE}}_B$: **discretization bias** (binning replaces $c$ by its bin average, which is downward), and **estimation bias** (the absolute value of a noisy mean is upward, $\mathbb{E}|\hat\mu| > |\mu|$, of order $\sqrt{n_b^{-1} p_b(1-p_b)}$). They have opposite signs and neither vanishes at fixed $B$.

Bin-free alternatives replace the partition with a smoothing kernel: **smECE** $= \inf_\sigma$-style kernel-smoothed residual (Błasiok–Nakkiran), or the **kernel calibration error** $\mathrm{KCE}$ (Widmann et al.), a Maximum Mean Discrepancy between the joint law of $(Z,Y)$ and its calibrated projection, estimable by a U-statistic with $O(n^{-1/2})$ bias-free convergence but a bandwidth $h$ that plays the role $B$ played before. The **distance to calibration**

$$\mathrm{dCE}(f) = \inf_{g \text{ calibrated}} \mathbb{E}|f(X) - g(X)|$$

is the Lipschitz functional the others approximate; $\mathrm{dCE} \le \mathrm{ECE}$ always, with the gap unbounded.

**Assumptions, and which are violated.** (i) $c$ is smooth or Hölder-$\alpha$ — assumed by every rate result, unverifiable, and false for models with discrete score plateaus. (ii) $Z$ has a density — violated by softmax outputs of over-confident networks, which pile up mass at $z \approx 1$; equal-width binning then puts $>90\%$ of ImageNet test points in one bin. (iii) i.i.d. held-out data — violated when the calibration set is reused for temperature selection and reporting. (iv) The bin grid is chosen independently of the data — violated by every paper that sweeps $B \in \{10,15,20\}$ and reports the sweep.

## 3. State of the Art

**Established.**
- *Bias of the plugin estimator.* Vaicenavicius et al. (AISTATS 2019) showed $\widehat{\mathrm{ECE}}_B$ is biased upward for calibrated models and gave a resampling test; the bias is confirmed independently.
- *Debiased estimation.* Kumar, Liang & Ma (NeurIPS 2019) gave a variance-subtraction debiased $\mathrm{CE}_2$ estimator with bootstrap intervals, plus scaling-binning recalibration with $O(\sqrt{B/n})$ guarantees.
- *ECE is not a consistent calibration measure.* Błasiok, Gopalan, Hu & Nakkiran (STOC 2023) formalized *consistent calibration measures* — polynomially related to $\mathrm{dCE}$ — and proved ECE is not one: there are $f$ with $\mathrm{dCE}(f) = O(\varepsilon)$ and $\mathrm{ECE}(f) = \Omega(1)$. This is a theorem, not a benchmark observation, and it is the reason the page's status is *methodologically blocked*.
- *Testing rates.* Lee, Huang, Hassani & Dobriban (T-Cal, JMLR 2023) gave a minimax-optimal test for calibration under Hölder-$\alpha$ smoothness; the detection boundary for $\ell_2$ calibration error scales as $n^{-2\alpha/(4\alpha+1)}$, i.e. $n^{-2/5}$ at $\alpha=1$ — slower than $n^{-1/2}$, so calibration testing is strictly harder than mean estimation.

**Claimed but unablated / benchmark-only.**
- Adaptive (equal-mass) binning is "less biased". Roelofs et al. (AISTATS 2022) measured bias across a large model sweep and found equal-mass binning plus a bias-aware bin count reduces but does not remove it; the recommendation of $\gtrsim 100$ samples per bin is empirical, not a bound.
- Kernel/KDE estimators (Zhang, Kailkhura & Han, ICML 2020; Popordanoska et al., NeurIPS 2022) report lower error on synthetic curves with *known* $c$. Transfer of that advantage to real networks, where $c$ is unknown, is asserted, not measured.
- Reported ECE improvements from recalibration methods in the $1$–$2$ percentage-point range on $n = 10^4$ test sets are, in most papers, within the estimator's own bias band. This is a benchmark number, not a measured improvement.

## 4. What Is Known

- Guo et al. (ICML 2017), $15$ equal-width bins, CIFAR-100 ResNet-110: $\widehat{\mathrm{ECE}} \approx 16.5\%$ pre-temperature-scaling, $\approx 1\%$ post. The pre-scaling number is large enough to survive any bias correction; the post-scaling number is not.
- Nixon et al. (CVPR Workshops, 2019): on ImageNet-scale models the reported ECE moves by a factor of $2$–$3$ purely as a function of bin count and equal-width vs. equal-mass choice, at fixed $n = 50{,}000$.
- Kumar, Liang & Ma (2019): for recalibrated ImageNet classifiers, the plugin estimator *underestimates* $\mathrm{CE}_2$ by a multiplicative factor; their debiased estimator plus bootstrap gives intervals at $n \approx 1000$ calibration points where the plugin estimate lies outside the interval.
- Roelofs et al. (2022), sweep over hundreds of trained image models: plugin ECE bias is on the order of the ECE itself once true ECE drops below $\approx 1\%$ at $n \le 10^4$.
- Gupta & Ramdas (ICML 2021; ICLR 2022): histogram binning with $B$ bins and $k$ points per bin gives distribution-free *conditional* coverage guarantees without sample splitting — the only distribution-free positive result in this area, and it bounds the calibration of the *binned* predictor, not of $f$.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no consensus population functional. ECE is discontinuous and provably not consistent with $\mathrm{dCE}$; $\mathrm{dCE}$ is consistent but has no accepted finite-sample estimator with intervals at $n \le 10^4$; smECE and KCE are consistent-ish but bandwidth-parameterized, so "the calibration error of model M" remains a number that depends on the analyst. Until the functional is fixed, "estimate it better" is not well posed.
- **Theoretically open.** Minimax rates for estimating $\mathrm{dCE}$ (as opposed to testing $\mathrm{CE}_2$ under smoothness) without any smoothness assumption. Whether a distribution-free two-sided confidence interval for any consistent calibration measure exists at width $o(1)$.
- **Empirically open.** Whether the published ranking of recalibration methods (temperature scaling vs. vector scaling vs. Dirichlet vs. conformal-style binning) survives being re-scored under a consistent measure at fixed $n$. The experiment is a re-scoring run over existing checkpoints; nobody has published it at scale.

## 6. Why It Is Hard

**Absent ground truth compounded by non-identifiability of the functional.** $c(v)$ is a conditional expectation on a continuum of conditioning events; from $n$ samples, each level set contains at most one point, so the unbinned plugin estimate of ECE for a continuous-output model is $\frac1n\sum_i |y_i - z_i| \approx 0.5$ for *any* predictor, calibrated or not. Every estimator therefore imposes a resolution — bins, bandwidth, or smoothness class — and the answer is a function of that choice. The second obstruction is that the standard target is the wrong object: because ECE is discontinuous, an estimator that converges to it faster is converging harder onto a quantity that can be $\Omega(1)$ for a predictor $\varepsilon$-close to calibrated. This is *an evaluation that does not measure the thing it names*, in the precise sense of the STOC 2023 separation.

## 7. Current Research (as of 2026)

- **Consistent calibration measures.** Błasiok, Gopalan, Hu, Nakkiran and collaborators (Apple/Harvard/Stanford orbit): smECE as a practical consistent measure with reliability diagrams that are actually functions of $f$; follow-ups comparing measures by their polynomial relationships to $\mathrm{dCE}$.
- **Algorithmic calibration testing.** Hu, Jambulapati, Tian, Yang and related work on testing calibration in nearly-linear time, and on the sample complexity of calibration testing in the standard query model *(frontier — verify exact venue)*.
- **Proper-score decompositions.** Gruber & Buettner (NeurIPS 2022) estimate calibration via the calibration term of a proper-score decomposition, inheriting the score's estimability; Bröcker's (2009) decomposition and Ferro–Fricker (2012) bias correction are the classical ancestors.
- **LLM-specific calibration.** Verbalized-confidence and token-probability calibration for language models, where $Z$ is heavily atomic and equal-width binning is degenerate. Reported ECEs here are the least trustworthy in the literature *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does the published ranking of recalibration methods survive a change of calibration measure?

**Scale.** $50$ ImageNet checkpoints (mix of ResNet/ViT/ConvNeXt, public weights), each recalibrated by $4$ methods (identity, temperature scaling, vector scaling, scaling-binning) on a $5{,}000$-image split, evaluated on the remaining $45{,}000$. $200$ (model, method) pairs. Cost: inference only, $\approx$ a few GPU-days.

**Measures.** $\widehat{\mathrm{ECE}}_{15}$ equal-width (the incumbent); equal-mass with $100$ points/bin; Kumar debiased $\mathrm{CE}_2$; smECE. Subsample to $n \in \{1000, 5000, 20000, 45000\}$, $50$ bootstrap replicates each.

**Control arm.** Synthetic recalibration of each checkpoint to *known* ground truth: fit $c$ on the full $45{,}000$, then generate labels from the fitted $c$, so true $\mathrm{ECE}$ and true $\mathrm{dCE}$ are known by construction. This is the only arm where estimator bias is directly observable.

**Deciding number.** Kendall's $\tau$ between the method ranking under $\widehat{\mathrm{ECE}}_{15}$ at $n=5000$ and under smECE at $n=45000$, averaged over the $50$ checkpoints. $\tau \ge 0.9$: the incumbent measure is a usable proxy and the field's published rankings stand. $\tau \le 0.5$: a substantial fraction of the recalibration literature is ranking on estimator artifacts, and re-scoring is mandatory.

## 9. Key References

- **[Foundational]** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. — arXiv:1706.04599
- **[Foundational]** Vaicenavicius, Widmann, Andersson, Lindsten, Roll, Schön. *Evaluating Model Calibration in Classification.* AISTATS 2019. — arXiv:1902.06977
- **[Foundational]** Kumar, Liang, Ma. *Verified Uncertainty Calibration.* NeurIPS 2019. — arXiv:1909.10155
- **[SOTA/theory]** Błasiok, Gopalan, Hu, Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC 2023. — arXiv:2211.16886
- **[SOTA/practice]** Błasiok, Nakkiran. *Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing.* ICLR 2024. — arXiv:2309.12236
- **[SOTA/testing]** Lee, Huang, Hassani, Dobriban. *T-Cal: An Optimal Test for the Calibration of Predictive Models.* JMLR 2023. — arXiv:2203.01850
- **[Empirical]** Roelofs, Cain, Shlens, Mozer. *Mitigating Bias in Calibration Error Estimation.* AISTATS 2022. — arXiv:2012.08668
- **[Empirical]** Nixon, Dusenberry, Zhang, Jerfel, Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops 2019. — arXiv:1904.01685
- **[Method]** Widmann, Lindsten, Zachariah. *Calibration Tests in Multi-class Classification: A Unifying Framework.* NeurIPS 2019. — arXiv:1910.11385
- **[Method]** Gupta, Ramdas. *Distribution-free Calibration Guarantees for Histogram Binning without Sample Splitting.* ICML 2021. — arXiv:2105.04656
- **[Method]** Popordanoska, Sayer, Blaschko. *A Consistent and Differentiable $L_p$ Canonical Calibration Error Estimator.* NeurIPS 2022. — arXiv:2210.07810
- **[Survey/decomposition]** Bröcker. *Reliability, Sufficiency, and the Decomposition of Proper Scores.* Quarterly Journal of the Royal Meteorological Society, 2009.

## 10. Worked Example

Take a **perfectly calibrated** predictor: $Z \sim \mathrm{Unif}[0,1]$, $Y \mid Z \sim \mathrm{Bernoulli}(Z)$. True $\mathrm{ECE} = 0$, true $\mathrm{dCE} = 0$. Draw $n = 1000$ points and report the standard statistic.

**Arm A — $B = 15$ equal-width bins.** Each bin holds $n_b \approx 66.7$ points. Within bin $b$ at level $p_b$, $\bar y_b - \bar z_b$ has mean $\approx 0$ and standard deviation $\approx \sqrt{p_b(1-p_b)/66.7}$. The absolute value of a mean-zero Gaussian has expectation $\sqrt{2/\pi}\,\sigma \approx 0.798\,\sigma$. Averaging over $p_b \sim \mathrm{Unif}[0,1]$, with $\mathbb{E}\sqrt{p(1-p)} = \pi/8 = 0.3927$:

$$\mathbb{E}\big[\widehat{\mathrm{ECE}}_{15}\big] \approx 0.798 \times \frac{0.3927}{\sqrt{66.7}} = 0.798 \times 0.0481 \approx \mathbf{0.038}.$$

A perfectly calibrated model reports $3.8\%$ ECE.

**Arm B — $B = 50$ bins.** $n_b = 20$, giving $0.798 \times 0.3927/\sqrt{20} \approx \mathbf{0.070}$.

**Arm C — no binning** (each distinct $z_i$ its own level set). Then $\widehat{\mathrm{ECE}} = \frac1n\sum_i|y_i - z_i| = \mathbb{E}|Y-Z| = \mathbb{E}[2Z(1-Z)] = 1/3 \approx \mathbf{0.333}$.

**Arm D — $B = 1$.** $\bar y \approx \bar z \approx 0.5$, so $\widehat{\mathrm{ECE}} \approx 0.016$ — and it would also be $\approx 0.016$ for a badly miscalibrated model whose errors cancel across the range.

Same model, same data, four answers: $0.016$, $0.038$, $0.070$, $0.333$, against a truth of $0$. The spread is not noise — it is the estimator's resolution parameter, and it is larger than the $1$–$2$ percentage-point differences that recalibration papers report as improvements. The debiased $\mathrm{CE}_2$ estimator removes the $\sqrt{1/n_b}$ term in Arms A–B, but the $B=1$ and $B=n$ pathologies survive it: they are properties of the functional, not of the estimator. That is the obstruction.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*