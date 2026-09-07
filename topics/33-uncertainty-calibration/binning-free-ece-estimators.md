---
id: 33-uncertainty-calibration/binning-free-ece-estimators
title: "Binning-Free ECE Estimators with Provable Rates"
topic: 33-uncertainty-calibration
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Binning-Free ECE Estimators with Provable Rates

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/binning-free-ece-estimators` · **Status:** partially-solved

## 1. Problem Statement

Given a held-out sample $(f(X_i), Y_i)_{i=1}^n$ from a probabilistic classifier, report a scalar miscalibration number with (a) no binning hyperparameter, (b) a finite-sample error bound, and (c) a rate that does not silently depend on the smoothness of the unknown conditional $\mathbb{E}[Y \mid f(X)]$.

Three variants, with different difficulty:

- **Measurement.** Which functional should be reported? The population ECE is a discontinuous functional of the joint law of $(f(X), Y)$ — two distributions arbitrarily close in Wasserstein distance can have ECE $0$ and ECE $\approx 1/2$. So "estimate ECE" is ill-posed before it is hard.
- **Method.** Given a chosen functional (smooth calibration error, kernel calibration error, smoothed ECE, distance to calibration), produce an estimator with no free bin count and a computable confidence interval.
- **Theory.** Prove minimax rates: for which functional and which assumption class is the achievable $|\hat{E}_n - E|$ of order $n^{-1/2}$, and where is $n^{-1/2}$ provably unattainable?

Solving it means: a single estimator, no tuned bin count, with a two-sided bound holding for all distributions in a stated class, plus a matching lower bound.

## 2. Formal Setting

Let $X \in \mathcal{X}$, $Y \in \{0,1\}$ (or $\{1,\dots,K\}$), and $f : \mathcal{X} \to [0,1]$ the model's reported confidence. Write $Z = f(X)$ and the calibration curve
$$r(z) \;=\; \mathbb{E}[Y \mid Z = z].$$
**Measured as:** $r$ is never observed. Only $n$ i.i.d. pairs $(Z_i, Y_i)$ are, with $Y_i \in \{0,1\}$ a single Bernoulli draw per point — this is the crux: there is one label per confidence value and $Z$ is continuous, so no $z$ repeats.

The $\ell_p$ calibration error is
$$\mathrm{CE}_p(f) \;=\; \big(\mathbb{E}_Z\,|r(Z) - Z|^p\big)^{1/p}, \qquad \mathrm{ECE} = \mathrm{CE}_1 .$$

**Plugin (binned) estimator.** Partition $[0,1]$ into $B$ bins $I_1,\dots,I_B$; with $n_b = |\{i : Z_i \in I_b\}|$,
$$\widehat{\mathrm{ECE}}_B \;=\; \sum_{b=1}^{B} \frac{n_b}{n}\left| \frac{1}{n_b}\sum_{i \in I_b} Y_i \;-\; \frac{1}{n_b}\sum_{i \in I_b} Z_i \right| .$$
Two errors, opposite in sign: **discretisation bias** (downward — averaging over a bin cancels miscalibration of opposite signs inside it, $\widehat{\mathrm{ECE}}_B \le \mathrm{ECE}$ in expectation of the binned functional) and **estimation bias** (upward — $\mathbb{E}|\hat{\mu}_b - \mu_b| \approx \sqrt{\mathrm{Var}}\;>0$ even when $r(z)=z$ exactly, so a perfectly calibrated model measures $\widehat{\mathrm{ECE}}_B \approx \sqrt{B/(2\pi n)}$).

**Kernel / smoothing alternatives.** For a bandwidth $\sigma$ and kernel $K_\sigma$,
$$\mathrm{smECE}_\sigma(f) = \int \Big| \big(K_\sigma * \textstyle\sum_i \delta_{Z_i}(Y_i - Z_i)\big)(z) \Big| \, dz \Big/ n,$$
a bin-free reliability functional (Błasiok–Nakkiran). The kernel calibration error (Widmann et al.) uses an RKHS $\mathcal{H}$ with kernel $k$:
$$\mathrm{KCE}^2 = \mathbb{E}\big[(Y - Z)(Y' - Z')\,k(Z,Z')\big],$$
estimated by a U-statistic — unbiased, $n^{-1/2}$ CLT, but it is a *different* functional from ECE and is zero only under the RKHS's separating condition.

**Distance to calibration.** $\mathrm{dCE}(f) = \inf_{g \text{ calibrated}} \mathbb{E}|f(X) - g(X)|$ — the Wasserstein-style repair cost, which unlike ECE is Lipschitz in the data distribution.

**Assumptions and their status.**
- *i.i.d. held-out data*: routinely violated — validation sets are reused across model selection, so the reported number is post-selection.
- *$r$ Hölder-$\beta$ smooth*: assumed by every rate result; unverifiable from data, and empirically false at the endpoints, where modern networks pile mass at $Z \approx 1$ (atoms in the law of $Z$ break the density assumptions outright).
- *Binary reduction*: multiclass ECE is usually top-label ECE, which is not the same functional as full-vector (canonical) calibration; a top-label-calibrated model can be arbitrarily miscalibrated canonically.

## 3. State of the Art

**Theory SOTA (established).**
- *Debiased plugin.* Kumar, Liang, Ma (NeurIPS 2019) give a variance-corrected $\ell_2$ estimator and prove it reaches accuracy $\epsilon$ with $O(\sqrt{B}/\epsilon^2)$ samples versus $O(B/\epsilon^2)$ for the plugin — still binned, but the bin count enters at half the exponent.
- *Impossibility.* Gupta, Podkopaev, Ramdas (NeurIPS 2020): distribution-free calibration guarantees are impossible for continuous-output predictors; any such guarantee forces discretisation. This is a theorem about the *guarantee*, and it is why "binning-free with distribution-free coverage" is not simply an engineering gap.
- *Well-posed replacement.* Błasiok, Gopalan, Hu, Nakkiran (STOC 2023) prove that smooth calibration error, LDTC, and several others are all polynomially equivalent to $\mathrm{dCE}$, and that ECE is *not* in this class (no constant-factor relation). This is the strongest result on the page: it says the measurement variant is solved by changing the functional.
- *Consistent bin-free estimator.* Błasiok, Nakkiran (ICLR 2024) define $\mathrm{smECE}$, prove it is consistent, monotone in $\sigma$, and admits a self-consistent bandwidth choice; the empirical-to-population gap scales as roughly $\tilde{O}(1/(\sigma\sqrt{n}))$ for fixed $\sigma$.
- *Testing rate.* Lee, Huang, Hassani, Dobriban (T-Cal, JMLR 2023) give a minimax-optimal test for $\mathrm{CE}_2$; under $\beta$-Hölder smoothness the detection boundary is of order $n^{-2\beta/(4\beta+1)}$, i.e. strictly slower than $n^{-1/2}$, and they show no consistent ECE estimator exists without a smoothness assumption.

**Empirical SOTA (claimed, partly unablated).**
- Kernel-density ECE (Zhang, Kailkhura, Han, ICML 2020) and spline-based KS calibration error (Gupta et al., ICLR 2021) are reported as lower-bias than 15-bin ECE on CIFAR/ImageNet, but the comparisons are benchmark numbers against an unknown truth — no ground-truth $r$, so "lower" is not "more accurate."
- Equal-mass ("adaptive") binning (Nixon et al., CVPRW 2019) is widely adopted on the strength of benchmark tables; Roelofs et al. (AISTATS 2022) show it reduces but does not remove bias.

## 4. What Is Known

- **Plugin ECE is upward-biased at zero miscalibration.** For a perfectly calibrated model with $B$ equal-mass bins, $\mathbb{E}[\widehat{\mathrm{ECE}}_B] \approx \sqrt{B/(2\pi n)}$: with $B=15$, $n=10{,}000$, that is $\approx 0.0155$ — the same order as the ECE of a well-tuned ImageNet model.
- **Bin count changes the ranking, not just the value.** Roelofs et al. (2022), sweeping thousands of trained models on CIFAR-10/100 and ImageNet, find the sign of the difference between two models' ECE flips with $B$ for a non-trivial fraction of pairs.
- **Debiasing moves the number materially.** Kumar et al. (2019) report that on ImageNet-scale validation sets ($n \approx 10^4$) with 15 bins, plugin and debiased $\ell_2$ estimates differ by a factor near two for temperature-scaled models — the reported ECE of $\sim 0.02$ is dominated by estimator noise.
- **Non-identifiability is formal, not folkloric.** ECE is discontinuous in the data distribution (STOC 2023); $\mathrm{dCE}$ is $1$-Lipschitz. Smooth calibration error is within a quadratic factor of $\mathrm{dCE}$.
- **$n^{-1/2}$ is unattainable in general.** T-Cal's lower bound rules out root-$n$ estimation of $\mathrm{CE}_2$ uniformly over $\beta$-Hölder classes.

## 5. What Is Not Known

- **Theoretically open.** Minimax rates for estimating (not merely testing) $\mathrm{dCE}$ or $\mathrm{smECE}$ — is there a matching lower bound for $\mathrm{dCE}$ estimation, and is $\mathrm{dCE}$ estimable at $n^{-1/2}$ without smoothness assumptions? Also open: the exact polynomial relating smooth calibration error to $\mathrm{dCE}$ (current bounds are quadratic on one side and linear on the other; the truth is unknown).
- **Theoretically open.** Multiclass. All rate results are effectively one-dimensional (binary or top-label). Canonical $K$-class calibration is a $(K-1)$-dimensional regression problem; no rate is known that avoids a $n^{-1/(K+\cdot)}$ curse.
- **Empirically open.** Whether bin-free estimators change *decisions*. Nobody has run a large model-selection study — thousands of checkpoints — asking whether ranking by $\mathrm{smECE}$ versus $\widehat{\mathrm{ECE}}_{15}$ selects different models, and which ranking better predicts downstream decision loss.
- **Methodologically blocked.** Estimator accuracy cannot be measured on real data because $r$ is unknown. Every "our estimator is less biased" claim on CIFAR/ImageNet is unfalsifiable without synthetic ground truth.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by non-identifiability of the target**. With one Bernoulli label per continuous $Z_i$, $r(z)$ is a regression function seen through a single-sample-per-point channel; any estimate of $|r(Z)-Z|$ requires pooling, and pooling *is* smoothing, which is the bias the method claims to remove. Binning is not an implementation choice one can delete — it is the only distribution-free way to obtain a repeated observation (Gupta et al. 2020). Bandwidth replaces bin count: the hyperparameter is relocated, not eliminated. And because the population ECE is discontinuous, there is no consistent estimator to converge to, so "less biased" is a claim about a quantity that no experiment on real data can score.

## 7. Current Research (as of 2026)

- **Distance-to-calibration program** (Błasiok, Gopalan, Hu, Nakkiran, and the algorithmic-fairness/multicalibration community): tighten the polynomial equivalences, extend to multiclass and to regression/quantile settings. *(frontier — verify)* Work on online and sequential calibration distance is active.
- **smECE as default reporting** — adoption in reliability-diagram tooling, replacing 15-bin plots. Uptake is partial; most 2025–26 LLM calibration papers still report $\widehat{\mathrm{ECE}}_{15}$.
- **Testing over estimation** (Dobriban group and others): report a calibration *p*-value with optimal detection rate rather than a point estimate.
- **LLM-specific**: token-level and free-form-answer calibration, where $Z$ has heavy atoms at $1$ and the smoothness assumptions fail hardest. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does any bin-free estimator beat a debiased 15-bin plugin when the truth is known?

**Scale.** Build a synthetic benchmark with *known* $r$: draw $Z \sim \mathrm{Beta}$ mixtures fitted to the confidence histograms of 5 real models (ResNet-50/ImageNet, ViT-B/16, a temperature-scaled CIFAR-100 WRN, a 7B LLM's MC-answer confidences, a logistic-regression tabular model), then set $Y_i \sim \mathrm{Bernoulli}(r(Z_i))$ for perturbations $r(z) = z + a\,\phi_\omega(z)$ with amplitude $a \in \{0, 0.01, 0.02, 0.05\}$ and frequency $\omega \in \{1,4,16\}$ (the high-$\omega$ case is exactly what binning cancels). Sample sizes $n \in \{10^3, 10^4, 10^5\}$, 500 replicates per cell. Total cost: CPU-hours, no GPU.

**Control arm.** Debiased $\ell_2$ plugin with equal-mass $B = 15$ (Kumar et al. 2019) — the current default with a rate proof.

**Deciding number.** Root-mean-square relative error $\mathrm{RMSRE} = \big(\mathbb{E}[(\hat{E} - E)^2]\big)^{1/2} / E$ at $n = 10^4$, $a = 0.02$, $\omega = 16$, where $E$ is the true $\mathrm{CE}_2$ of the generator. Verdict: a bin-free estimator supersedes the control iff it achieves RMSRE below the control's *and* below $0.25$ in every $(\omega, a>0)$ cell. If no estimator clears $0.25$ at high $\omega$, the honest conclusion is that ECE at $n=10^4$ carries $\pm 25\%$ and single-decimal ECE comparisons in the literature are noise.

## 9. Key References

- **[Foundational]** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** Kumar, Liang, Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[SOTA]** Błasiok, Gopalan, Hu, Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC, 2023.
- **[SOTA]** Błasiok, Nakkiran. *Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing.* ICLR, 2024.
- **[SOTA]** Lee, Huang, Hassani, Dobriban. *T-Cal: An Optimal Test for the Calibration of Predictive Models.* JMLR, 2023.
- **[Theory]** Gupta, Podkopaev, Ramdas. *Distribution-free binary classification: prediction sets, confidence intervals and calibration.* NeurIPS, 2020.
- **[Method]** Widmann, Lindsten, Zachariah. *Calibration tests in multi-class classification: a unifying framework.* NeurIPS, 2019.
- **[Method]** Zhang, Kailkhura, Han. *Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning.* ICML, 2020.
- **[Method]** Gupta, Rahimi, Ajanthan, Mensink, Sminchisescu, Hartley. *Calibration of Neural Networks using Splines.* ICLR, 2021.
- **[Empirical]** Roelofs, Cain, Shlens, Mozer. *Mitigating bias in calibration error estimation.* AISTATS, 2022.
- **[Empirical]** Nixon, Dusenberry, Zhang, Jerfel, Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops, 2019.
- **[Survey]** Vaicenavicius, Widmann, Andersson, Lindsten, Roll, Schön. *Evaluating model calibration in classification.* AISTATS, 2019.

## 10. Worked Example

Take a model that is **exactly calibrated**: $Z \sim \mathrm{Uniform}[0,1]$, $Y \mid Z \sim \mathrm{Bernoulli}(Z)$, so $\mathrm{ECE} = 0$. Draw $n = 10{,}000$.

Equal-mass binning, $B = 15$: each bin holds $n_b \approx 667$ points with mean confidence $\bar{z}_b$. The empirical accuracy $\hat{\mu}_b$ has standard deviation $\sqrt{\bar{z}_b(1-\bar{z}_b)/n_b}$; averaged over bins with $\bar z_b$ spread uniformly, $\mathbb{E}[\bar z(1-\bar z)] = 1/6$, so $\mathrm{sd} \approx \sqrt{0.167/667} = 0.0158$. For a mean-zero Gaussian, $\mathbb{E}|\cdot| = \sqrt{2/\pi}\,\mathrm{sd} = 0.0126$. So
$$\mathbb{E}\big[\widehat{\mathrm{ECE}}_{15}\big] \approx 0.0126 \quad \text{when the true ECE is } 0.$$
Published ImageNet ECEs after temperature scaling sit around $0.01$–$0.03$. The floor is the same size as the signal.

Now perturb: $r(z) = z + 0.02\sin(32\pi z)$, true $\mathrm{ECE} = 0.02 \cdot \frac{2}{\pi} = 0.0127$. With 15 bins, each bin spans $1/15$ of $[0,1]$ and contains $\approx 1.07$ full periods of the sine, so the within-bin average of $r(z)-z$ is $\approx 0$. The binned estimator returns $\approx 0.0126$ — **numerically identical to the perfectly calibrated case**. The measurement cannot distinguish ECE $=0$ from ECE $=0.0127$.

Raising $B$ to $200$ resolves the oscillation but lifts the noise floor to $\sqrt{200/(2\pi \cdot 10^4)} \approx 0.056$, four times the signal. $\mathrm{smECE}$ with the self-consistent bandwidth faces the same trade in continuous form: any $\sigma \gtrsim 1/32$ averages the perturbation away, and $\sigma \ll 1/32$ makes the estimate noise. The obstruction is not the choice of bins — it is that at $n = 10^4$ the data does not contain the information, and no estimator, binned or not, can manufacture it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*