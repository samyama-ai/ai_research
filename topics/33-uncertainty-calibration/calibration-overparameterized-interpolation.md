---
id: 33-uncertainty-calibration/calibration-overparameterized-interpolation
title: "Calibration in the Overparameterized Interpolation Regime"
topic: 33-uncertainty-calibration
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration in the Overparameterized Interpolation Regime

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/calibration-overparameterized-interpolation` · **Status:** open

## 1. Problem Statement

A model that interpolates its training set assigns probability $\approx 1$ to every training label, including noisy ones. Test accuracy can still be excellent (benign overfitting). The question is what happens to the *probabilities* rather than the argmax.

- **Measurement variant.** Given an interpolating classifier $\hat{f}$ and a finite test sample, estimate its distance from calibration with a bounded, consistent estimator. Plugin binned ECE is neither, so this variant is not merely an engineering detail.
- **Method variant.** Find a training procedure that reaches interpolation-regime test accuracy *and* calibration error at the level of a well-tuned underparameterized model, without a held-out recalibration split. Post-hoc temperature scaling is the incumbent and it needs that split.
- **Theory variant.** For a stated overparameterized model class (random-features, linear high-dimensional logistic, NTK-regime networks), characterize $\lim_{n\to\infty}$ calibration error as a function of the parameter/sample ratio $\kappa = p/n$ and label noise $\eta$, and prove whether the map $\kappa \mapsto \text{miscalibration}$ is monotone, peaked at the interpolation threshold like the test-error double-descent curve, or decoupled from it.

Solving it means: a theorem for the theory variant in at least one non-trivial nonlinear class, plus a reproduced empirical law relating $\kappa$, $\eta$ and calibration on real data.

## 2. Formal Setting

Data $(X, Y) \sim \mathcal{D}$ on $\mathcal{X} \times \{1,\dots,K\}$; training set $S = \{(x_i, y_i)\}_{i=1}^n$ i.i.d. A predictor $\hat{f}: \mathcal{X} \to \Delta^{K-1}$ has confidence $\hat{c}(x) = \max_k \hat{f}_k(x)$ and prediction $\hat{y}(x) = \arg\max_k \hat{f}_k(x)$.

**Interpolation.** $\hat{f}$ interpolates $S$ if $\hat{y}(x_i) = y_i$ for all $i$; *confidently* interpolates if additionally $\hat{c}(x_i) \ge 1-\epsilon$ (measured: mean train confidence, typically $>0.999$ after cross-entropy training to zero loss).

**Perfect calibration.** $\Pr[Y = \hat{y}(X) \mid \hat{c}(X) = c] = c$ for a.e. $c$. Expected calibration error:
$$\mathrm{ECE}(\hat{f}) = \mathbb{E}_X\big|\Pr[Y=\hat{y}(X)\mid \hat{c}(X)] - \hat{c}(X)\big|.$$

**As actually measured.** With $M$ bins $B_1,\dots,B_M$ over $[0,1]$ and $N$ test points,
$$\widehat{\mathrm{ECE}}_M = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.$$
This is a *biased-downward* estimator of $\mathrm{ECE}$: binning can only cancel errors, never create them, and the bias grows as $M$ shrinks (Kumar, Liang, Ma 2019). Reported settings are usually $M=15$ equal-width bins, $N=10{,}000$ (ImageNet val) or $N=10{,}000$ (CIFAR test) — small enough that per-bin accuracy noise is $\pm 1$–$3$ points in the top bin.

**Overparameterization ratio.** $\kappa = p/n$ with $p$ = trainable parameters. For linear high-dimensional logistic regression, $\kappa$ is the aspect ratio and separability occurs above the Candès–Sur phase-transition curve $h(\kappa,\gamma)$, where $\gamma^2 = \mathrm{Var}(x^\top\beta)$ is the signal strength.

**Label noise.** $\eta = \Pr[Y \ne \arg\max_k \Pr(Y{=}k\mid X)]$; measured only by construction (synthetic flips), not observable on real datasets.

**Assumptions and their violations.**
1. *Test distribution = train distribution.* Violated under shift; calibration degrades far faster than accuracy (Ovadia et al. 2019).
2. *$\hat{c}$ has a density.* Violated at interpolation: confidences pile up at $1-10^{-6}$, so equal-width bins put $>90\%$ of ImageNet test mass in one bin.
3. *Top-label calibration is the target.* The full-distribution ($K$-class) criterion is strictly stronger and is what a downstream decision rule needs.
4. *$p$ measures capacity.* False across architectures; $\kappa$ is comparable only within a width- or depth-scaled family.

## 3. State of the Art

**Established.**
- Temperature scaling (Guo et al., ICML 2017): one scalar $T$ fit by NLL on a validation split; reduces ImageNet/CIFAR ECE by roughly an order of magnitude and leaves accuracy exactly unchanged. Still the strongest cost/benefit baseline nine years later.
- Sur & Candès (PNAS 2019): in the separable/high-dimensional logistic regime the MLE is *provably* overconfident — coefficients inflate by a deterministic factor $\alpha_\star(\kappa,\gamma) > 1$, so miscalibration in this class is a bias of the estimator, not an optimization artifact.
- Kumar, Liang, Ma (NeurIPS 2019): scaling-then-binning gives calibration error estimates with finite-sample guarantees; plugin binned ECE does not.
- Błasiok, Gopalan, Hu, Nakkiran (STOC 2023): a consistent-calibration-measure framework showing which "ECE variants" are polynomially related to the true distance from calibration and which (including plugin binned ECE) are not.

**Claimed but unablated.**
- "Modern networks are miscalibrated because they are overparameterized" (the popular reading of Guo 2017). Minderer et al. (NeurIPS 2021) contradicts it: the largest, most overparameterized models of that generation (ViT, MLP-Mixer, BiT) were among the *best* calibrated in-distribution and under shift. Neither paper isolates $\kappa$ from architecture, data augmentation, and pretraining scale.
- Focal loss (Mukhoti et al., NeurIPS 2020) and label smoothing as calibration fixes: reported as benchmark ECE numbers, with the confound that both prevent confident interpolation outright, so they change the regime rather than fix it.

**Benchmark-number-only.** Most reported gains under 1 ECE point on CIFAR-100 are within the estimator's binning-choice variation (§10) and should not be read as ordering methods.

## 4. What Is Known

- **Interpolation of noisy labels is achievable and not fatal to accuracy.** Zhang et al. (ICLR 2017) fit random labels on CIFAR-10 to zero training error; Bartlett, Long, Lugosi, Tsigler (PNAS 2020) give exact conditions for benign overfitting in linear regression — the excess risk vanishes when the covariance spectrum has heavy enough an effective-rank tail.
- **Test-error double descent is real at scale.** Nakkiran et al. (ICLR 2020) show model-wise and epoch-wise double descent on CIFAR-10/100 ResNet-18 with 15% label noise; the peak sits at the interpolation threshold and disappears at $\eta=0$.
- **Overparameterization alone does not explain overconfidence.** Bai, Mei, Wang, Xiong (ICML 2021) show in high-dimensional binary classification that the calibration behavior depends on the loss and the ratio, and that logistic loss is overconfident where a suitably modified objective is not.
- **Scale-era numbers.** Guo et al. (2017) report DenseNet/ResNet CIFAR-100 ECE of roughly 15–20% pre-scaling, falling to ~1–2% after temperature scaling ($N=10^4$, $M=15$). Minderer et al. (2021) report ImageNet ECE of a few percent for the best 2021 models at $N=5\times10^4$. The two are not directly comparable: different architectures, pretraining data, and bin counts.
- **Proper-loss minimization does not imply calibration in general, but near-optimal proper loss does imply near-calibration under a stated condition** (Błasiok, Gopalan, Hu, Kalai, Nakkiran, NeurIPS 2023).

## 5. What Is Not Known

- **Theoretically open.** Whether calibration error, defined by any consistent measure, exhibits double descent in $\kappa$ for a nonlinear class (random features or NTK). No proof either way. Sur–Candès covers only the linear separable logistic case; the random-features replica analyses (Clarté, Loureiro, Krzakala, Zdeborová) cover uncertainty in a restricted setting.
- **Theoretically open.** Whether there exists an interpolating estimator that is asymptotically calibrated without a held-out split in the presence of $\eta>0$ label noise, or whether split-free calibration and confident interpolation are provably incompatible.
- **Empirically open.** The clean $\kappa$-sweep on real data: width-scaled ResNets with fixed everything else, $\eta \in \{0, 0.1, 0.2\}$, measuring calibration with a consistent estimator across the interpolation threshold. Cheap enough to run ($10^2$–$10^3$ GPU-hours), never run at publication quality.
- **Methodologically blocked.** Calibration *of an interpolating model on its training distribution's hard region*. When >90% of test mass sits in the top confidence bin, ECE degenerates to "1 − accuracy in that bin", which is accuracy, not calibration. No accepted measure resolves the top bin at $10^4$ test points.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus estimator degeneracy at the boundary**.

1. *Degeneracy.* Confident interpolation pushes the confidence distribution to a point mass near 1. Every binned estimator loses resolution exactly where the model lives, so the estimator's bias is largest in the regime being studied. Increasing $M$ trades bias for variance with $\le 10^4$ test points and cannot be escaped by tuning $M$.
2. *Confounding.* $\kappa$ cannot be varied in real training without also varying optimization trajectory length, implicit regularization strength, and effective augmentation. The Guo-vs-Minderer disagreement is entirely inside this confound.
3. *Absent ground truth.* $\eta$ and the Bayes-optimal $\Pr(Y\mid X)$ are unobservable on CIFAR/ImageNet, so "the model should have said 0.7 here" has no referent. Synthetic-noise experiments restore ground truth but change the problem.
4. *Non-identifiability.* Many probability functions induce the same ECE; low ECE is necessary, not sufficient, and a model can be perfectly ECE-calibrated while being useless per-instance (predict the base rate everywhere).

## 7. Current Research (as of 2026)

- **Consistent calibration measures.** Follow-ups to the Błasiok–Gopalan–Hu–Nakkiran line, replacing ECE with smooth/Lipschitz-dual distances that behave at the confidence boundary *(frontier — verify)*.
- **High-dimensional asymptotics of uncertainty.** EPFL/IdePHICS-adjacent replica analyses of uncertainty in random-features and generalized linear models, extending Sur–Candès beyond the linear case.
- **Calibration of LLM verbalized and token-level confidence**, where the interpolation framing recurs: fine-tuning to near-zero loss on preference data reliably sharpens output distributions and degrades calibration *(frontier — verify the causal claim; most evidence is observational)*.
- **Training-time calibration objectives** (soft-binning/soft-calibration losses, Karandikar et al. NeurIPS 2021) as the split-free alternative to temperature scaling.

## 8. Concrete Next Experiment

**Question.** Does calibration error double-descend in $\kappa$, or is it monotone?

- **Scale.** CIFAR-10, ResNet-18 with width multiplier $w \in \{1,2,4,6,8,10,12,16,24,32,64\}$ (spanning $\kappa = p/n$ from $\approx 0.02$ to $\approx 15$ at $n=50{,}000$), label noise $\eta \in \{0, 0.10, 0.20\}$ applied by symmetric flips so the Bayes probabilities are *known*, 5 seeds each. 165 runs, roughly 400–800 A100-hours.
- **Measurement.** Not plugin ECE. Report (a) the scaling-binning estimator of Kumar et al. with its confidence interval, and (b) since $\eta$ is known by construction, the true $L_1$ calibration gap $\mathbb{E}|\hat{c}(X) - \Pr(Y=\hat{y}(X)\mid X)|$ computed against the clean-label oracle on a held-out 10,000-point clean set.
- **Control arm.** Same sweep with temperature scaling fit on 5,000 held-out points. If a peak exists pre-scaling and vanishes post-scaling, the phenomenon is a single global scalar and is not interesting. A second control: fixed-width $w=16$ with training-set size $n$ varied to sweep $\kappa$ the other way — separates capacity from optimization horizon.
- **Deciding number.** $\Delta = \max_{\kappa} \mathrm{CalGap}(\kappa) - \mathrm{CalGap}(\kappa_{\max})$ at $\eta = 0.2$, with the max taken near the interpolation threshold. A non-monotone peak with $\Delta > 0.03$ (3 points), 95% CI excluding zero over 5 seeds, establishes calibration double descent. $\Delta < 0.01$ with a monotone fit falsifies it.

## 9. Key References

- **[Foundational]** C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals. *Understanding Deep Learning Requires Rethinking Generalization.* ICLR, 2017. — arXiv:1611.03530
- **[Foundational]** M. Belkin, D. Hsu, S. Ma, S. Mandal. *Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-off.* PNAS 116(32), 2019.
- **[Foundational]** P. L. Bartlett, P. M. Long, G. Lugosi, A. Tsigler. *Benign Overfitting in Linear Regression.* PNAS 117(48), 2020.
- **[Theory SOTA]** P. Sur, E. J. Candès. *A Modern Maximum-Likelihood Theory for High-Dimensional Logistic Regression.* PNAS 116(29), 2019.
- **[Theory SOTA]** J. Błasiok, P. Gopalan, L. Hu, P. Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC, 2023. — arXiv:2211.16886
- **[Theory SOTA]** J. Błasiok, P. Gopalan, L. Hu, A. T. Kalai, P. Nakkiran. *When Does Optimizing a Proper Loss Yield Calibration?* NeurIPS, 2023.
- **[SOTA]** A. Kumar, P. Liang, T. Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[SOTA]** M. Minderer, J. Djolonga, R. Romijnders, F. Hubis, X. Zhai, N. Houlsby, D. Tran, M. Lucic. *Revisiting the Calibration of Modern Neural Networks.* NeurIPS, 2021. — arXiv:2106.07998
- **[SOTA]** Y. Bai, S. Mei, H. Wang, C. Xiong. *Don't Just Blame Over-parametrization for Over-confidence: Theoretical Analysis of Calibration in Binary Classification.* ICML, 2021.
- **[Empirical]** P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, I. Sutskever. *Deep Double Descent: Where Bigger Models and More Data Hurt.* ICLR, 2020. — arXiv:1912.02292
- **[Empirical]** Y. Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Survey/Method]** J. Nixon, M. Dusenberry, L. Zhang, G. Jerfel, D. Tran. *Measuring Calibration in Deep Learning.* CVPR Workshops, 2019.

## 10. Worked Example

**Part A — the theory side has a closed form.** Take high-dimensional logistic regression with $\kappa = p/n = 0.2$ and signal strength $\gamma = \sqrt{5}$ (the worked case in Sur & Candès 2019). The data are separable-adjacent; the MLE exists but its coordinates converge to $\alpha_\star \beta_j$ with $\alpha_\star > 1$ (their reported value for this configuration is close to $1.17$). So with true $x^\top\beta = 1.0$, the model outputs $\sigma(1.17) = 0.763$ where the truth is $\sigma(1.0) = 0.731$ — a $3.2$-point overconfidence that is *deterministic*, present at $n = 1000$, and not reduced by more data at fixed $\kappa$. Miscalibration here is a property of the estimator, not of SGD or of architecture.

**Part B — the measurement side breaks.** Now the same phenomenon on a real interpolating model. A ResNet-18 trained to zero training error on CIFAR-10 has test accuracy $\approx 0.95$ and a confidence histogram with about $88\%$ of the 10,000 test points in $[0.99, 1.0]$.

| Binning | $M$ | $\widehat{\mathrm{ECE}}$ |
|---|---|---|
| Equal-width | 15 | $\approx 0.030$ |
| Equal-width | 100 | $\approx 0.038$ |
| Equal-mass | 15 | $\approx 0.021$ |
| Equal-mass | 100 | high-variance; top bins hold $<100$ points each |

The spread across defensible binning choices is $\approx 0.017$ — larger than most published improvements on this benchmark. And the top equal-width bin contains 8,800 points with accuracy $\approx 0.97$ and mean confidence $\approx 0.997$; its contribution, $0.88 \times 0.027 = 0.024$, is 80% of the total and is arithmetically just $0.88\times(1-\text{accuracy in that bin})$ up to the $0.003$ confidence slack.

**The obstruction made visible.** In Part A the calibration gap is known exactly because $\Pr(Y\mid X)$ is known. In Part B, the number the community reports is dominated by a single bin in which the estimator cannot distinguish miscalibration from error rate, and its value moves by more than the effect size when the binning changes. That gap between (A) and (B) — not compute, not model scale — is what keeps this problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*