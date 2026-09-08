---
id: 08-loss-and-heads/heteroscedastic-loss-variance-collapse
title: "Loss Functions for Heteroscedastic Uncertainty Without Variance Collapse"
topic: 08-loss-and-heads
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Functions for Heteroscedastic Uncertainty Without Variance Collapse

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/heteroscedastic-loss-variance-collapse` · **Status:** partially-solved

## 1. Problem Statement

A network with two output heads predicts a conditional mean $\mu_\theta(x)$ and a conditional variance $\sigma^2_\theta(x)$, trained by Gaussian negative log-likelihood (NLL). The failure mode is well documented: on some subset of inputs the variance head collapses toward zero early in training, the NLL gradient on the mean head is scaled by $1/\sigma^2$, and those points come to dominate the mean fit while high-noise points are effectively discarded. The resulting model is *both* worse in RMSE than a plain MSE-trained network *and* badly calibrated. The mirror failure — variance inflation, where $\sigma^2$ absorbs mean-model misfit and the mean head underfits — occurs in the same setup with different initialization.

Three variants, different difficulty:

- **Measurement.** Define a metric that separates "the variance head is correct" from "the variance head is small". Aleatoric ground truth is unobserved; test NLL rewards both good calibration and good mean fit, and cannot tell them apart.
- **Method.** Find a training objective whose optimum is the true conditional Gaussian *and* whose optimization trajectory does not pass through a mean-fit-destroying regime. Partially solved: $\beta$-NLL (Seitzer et al., ICLR 2022) and stop-gradient decoupling (Stirn et al., AISTATS 2023) both fix the mean-fit degradation. Neither is known to be optimal.
- **Theory.** Characterize when the joint $(\mu,\sigma^2)$ optimization has spurious stationary points or diverges, as a function of model capacity, noise level, and $\sigma^2$ parameterization. Open.

Solved would mean: an objective with (a) a proof of properness (its population minimizer is the true conditional distribution), (b) mean-head accuracy no worse than MSE training on the same architecture at every noise level, and (c) calibration no worse than post-hoc-recalibrated MSE + residual fitting, all without a tuned hyperparameter.

## 2. Formal Setting

Data $\{(x_i,y_i)\}_{i=1}^n$, $x_i \in \mathcal{X}$, $y_i \in \mathbb{R}$, generated as
$$y = f^\star(x) + \varepsilon(x), \qquad \varepsilon(x) \sim \mathcal{N}(0, s^\star(x)),$$
with $f^\star$ the true regression function and $s^\star(x) = \mathrm{Var}[y\mid x]$ the true aleatoric variance. The model outputs $\mu_\theta(x)$ and $\sigma^2_\theta(x) = g(v_\theta(x))$ with link $g$ — typically $\exp(\cdot)$ (predict $\log\sigma^2$) or $\mathrm{softplus}(\cdot)+\epsilon_{\min}$. The floor $\epsilon_{\min}$ (commonly $10^{-6}$) is itself a confound: it sets the maximum gradient amplification.

Gaussian NLL, up to constants:
$$\mathcal{L}(\theta) = \frac{1}{n}\sum_i \left[ \frac{(y_i - \mu_\theta(x_i))^2}{2\sigma^2_\theta(x_i)} + \tfrac{1}{2}\log \sigma^2_\theta(x_i)\right].$$

The mechanism of collapse is one line:
$$\frac{\partial \mathcal{L}_i}{\partial \mu} = \frac{\mu_\theta(x_i) - y_i}{\sigma^2_\theta(x_i)}.$$
Per-example mean gradients are inversely weighted by predicted variance. Before $\sigma^2_\theta$ is accurate, this weighting is arbitrary, and it is self-reinforcing: a point that happens to be fit well early gets small $\sigma^2$, which raises its gradient weight, which fits it better.

**$\beta$-NLL** rescales each term by a stop-gradiented power of the predicted variance:
$$\mathcal{L}^\beta = \frac{1}{n}\sum_i \lfloor \sigma^{2\beta}_\theta(x_i)\rfloor_{\text{sg}} \left[\frac{(y_i-\mu_\theta(x_i))^2}{2\sigma^2_\theta(x_i)} + \tfrac12\log\sigma^2_\theta(x_i)\right],$$
so $\beta=0$ recovers NLL and $\beta=1$ gives mean gradients identical to MSE.

**Quantities as measured.**
- Mean quality: test RMSE against $y$, *not* against $f^\star$ (unavailable outside synthetic data).
- Calibration: expected calibration error over quantile levels, $\mathrm{ECE}_q = \frac{1}{K}\sum_k |\hat{F}(q_k) - q_k|$ where $\hat F(q_k)$ is the empirical fraction of test points below the model's $q_k$-quantile. Marginal, not conditional — this is the measurement weakness.
- Variance fidelity: only on synthetic data, $\frac1m\sum_j (\log\sigma^2_\theta(x_j) - \log s^\star(x_j))^2$.
- Collapse: $\Pr_{x\sim\text{test}}[\sigma^2_\theta(x) < \alpha \cdot \min_x s^\star(x)]$, reported for a stated $\alpha$; without synthetic ground truth, the fraction of points hitting the numerical floor.

**Assumptions and their violations.** Gaussianity of $\varepsilon$ — violated for most real regression targets (skew, heavy tails, censoring). Well-specified mean class ($f^\star$ representable) — violated at finite capacity, and the violation *causes* variance inflation, since $\sigma^2$ then correctly absorbs bias. Homogeneous i.i.d. sampling — violated for spatial and time series data. Aleatoric/epistemic separability — not identifiable from a single dataset at all: given finite data there is no test distinguishing irreducible noise from unlearned structure.

## 3. State of the Art

**Established (independently reproduced).**
- $\beta$-NLL (Seitzer, Tangemann, Bauer, Schölkopf, ICLR 2022) with $\beta=0.5$ recovers MSE-level mean accuracy while keeping usable variance estimates on UCI regression and on model-based RL dynamics learning. The gradient-weighting diagnosis in that paper is the accepted explanation of the failure.
- Warm-up / mean-first training (fit $\mu$ under MSE, then unfreeze $\sigma^2$) removes most collapse. Used in practice long before it was written up; formalized by Sluijterman, Cator, Heskes (*Optimal Training of Mean Variance Estimation Neural Networks*, Neurocomputing 2024), who also show that fitting mean and variance on split data avoids the self-reinforcing loop.
- **Faithful heteroscedastic regression** (Stirn, Barac, Pe'er, Knowles, AISTATS 2023): stop-gradient on the $1/\sigma^2$ factor in the mean gradient makes the mean head's optimum provably identical to MSE training — "faithfulness" — with no hyperparameter. This is the cleanest current result: a stated property with a proof, not a benchmark number.
- Deep ensembles (Lakshminarayanan, Pritzel, Blundell, NeurIPS 2017) use the same NLL head; their reported calibration gains come mostly from ensembling, and the underlying single-model collapse is not fixed by ensembling.

**Claimed but unablated.** That $\beta=0.5$ is a good default across domains — the ICLR 2022 evidence is UCI + RL, not vision or LLM-scale. That Student-$t$ or evidential heads fix the problem: evidential regression (Amini et al., NeurIPS 2020) was later shown by Meinert, Gawlikowski, Lavin (*The Unreasonable Effectiveness of Deep Evidential Regression*, AAAI 2023) to produce an uncertainty that is not a proper posterior quantity and is sensitive to a regularizer weight. Marginal-calibration-only benchmark numbers ($\mathrm{ECE}_q$ tables) are common and do not establish conditional calibration.

**Theory SOTA.** Immer, Palumbo, Marx, Vogt (*Effective Bayesian Heteroscedastic Regression with Deep Neural Networks*, NeurIPS 2023) give a natural-gradient/Laplace treatment where the variance is marginalized rather than point-estimated, removing the collapse at its source for the models where the approximation holds. Wong-Toi, Boyd, Fortuin, Mandt (*Understanding Pathologies of Deep Heteroskedastic Regression*, AISTATS 2024) characterize the pathology as a phase transition in regularization strength: below a critical value the fit collapses to interpolation with vanishing variance, above it to a homoscedastic fit.

## 4. What Is Known

- The $1/\sigma^2$ mean-gradient weighting is the mechanism, not a symptom. Established analytically and by ablation.
- $\beta$-NLL at $\beta \in [0.5, 1]$ recovers MSE-level RMSE on standard UCI sets ($n \approx 500$–$45{,}000$, $d \le 90$, MLPs of ~50–100 hidden units) while NLL training is markedly worse on several of them; on the RL benchmarks in the same paper the gap is large enough to change task success rates.
- Faithful heteroscedastic regression: the stop-gradient estimator's mean converges to the MSE solution *by construction*, verified on UCI and on single-cell genomics data at $d\sim10^4$ (AISTATS 2023).
- Variance networks trained without care yield overconfident extrapolation; Skafte, Jørgensen, Hauberg (NeurIPS 2019) showed locality-aware variance (fitting variance with a mean-reverting inductive bias outside the data) fixes the extrapolation half at small MLP scale.
- Pathology is regularization-controlled: below a critical weight decay / prior precision, the AISTATS 2024 analysis shows the solution collapses; there is a narrow band of correct behavior. The critical value is dataset-dependent and not predictable a priori.
- Scale: essentially all of the above is MLPs with $10^4$–$10^6$ parameters on tabular or low-dimensional data. Nothing here has been reproduced at $10^9$ parameters.

## 5. What Is Not Known

- **Methodologically blocked.** Whether a given variance head is *right* rather than merely *small*. Conditional calibration — $\Pr[y \le \hat F^{-1}_x(q) \mid x] = q$ for all $x$ — cannot be estimated from one sample per $x$ without smoothing assumptions. Every reported number is a marginal proxy. This blocks all of the comparisons below from being decisive.
- **Theoretically open.** Whether an objective exists that is (i) strictly proper, (ii) faithful in the Stirn et al. sense, and (iii) hyperparameter-free. $\beta$-NLL sacrifices properness for $\beta>0$; the stop-gradient estimator is not a gradient of any scalar loss. No impossibility proof is known either.
- **Theoretically open.** The location of the collapse phase boundary as a function of $(n, \text{capacity}, s^\star, \epsilon_{\min})$. AISTATS 2024 shows it exists; no closed form.
- **Empirically open.** Whether heteroscedastic heads matter at LLM/foundation-model scale, e.g. per-token predicted variance in a regression-head reward model or a diffusion $\sigma$ head. Runnable at $\sim10^9$ parameters for maybe $10^4$ GPU-hours; nobody has published the controlled comparison.
- **Empirically open.** Whether $\beta$-NLL, stop-gradient, warm-up, and Laplace-marginalized variance rank consistently across domains, or whether the ranking is dataset-permuted.

## 6. Why It Is Hard

The primary obstruction is **absent ground truth combined with a confounded metric**. $s^\star(x)$ is never observed. The default evaluation, test NLL, is a sum of a mean-fit term and a variance-fit term; a model can improve NLL by getting the mean right and the variance systematically wrong, or vice versa, and the aggregate hides which. So the field's headline metric does not measure the thing its name implies.

Second: **non-identifiability of aleatoric versus epistemic variance** at finite $n$. Excess residual variance from an under-capacity mean model is indistinguishable, in-sample, from genuine noise. This means "variance inflation" and "correct variance under misspecification" are the same event, and no loss function can separate them without an out-of-sample assumption.

Third: the pathology is an **optimization-trajectory** property, not a property of the optimum. The population NLL minimizer is correct; strict properness of the loss buys nothing. Any fix therefore has to reason about the training path, which is where theory is weakest.

## 7. Current Research (as of 2026)

- **Marginalized-variance Bayesian treatments.** Immer et al. (ETH Zürich / Univ. Basel) extending Laplace and natural-gradient variational inference to variance heads; the direction is to remove the point estimate entirely.
- **Faithfulness as a design constraint.** Stirn/Knowles (Columbia) line: define the property you want of the mean head, then construct the estimator to satisfy it. Extensions to non-Gaussian heads are the obvious next step *(frontier — verify)*.
- **Distribution-free alternatives.** Conformal prediction with locally adaptive scores (Romano, Patterson, Candès, NeurIPS 2019, CQR) sidesteps the loss problem by producing intervals with finite-sample marginal coverage. Growing view that heteroscedastic NLL heads should be trained for the mean and conformalized for the interval. This is a reframing, not a solution — conformal gives marginal, not conditional, coverage.
- **Diffusion and flow-matching variance heads.** Learned $\sigma$ schedules exhibit the same collapse dynamics; largely unstudied as an instance of this problem *(frontier — verify)*.
- **Regularization phase-boundary analysis.** Mandt group (UC Irvine) and collaborators, following the AISTATS 2024 characterization.

## 8. Concrete Next Experiment

**Question.** Does any current fix ($\beta$-NLL $\beta{=}0.5$, stop-gradient faithful, MSE warm-up, Laplace-marginalized) produce a variance head that is *conditionally* correct, or only marginally calibrated?

**Design.** Use a semi-synthetic construction that supplies ground-truth $s^\star(x)$ while keeping realistic $x$: take a real tabular or image-regression dataset, fit a large mean model, then *resample* targets as $y' = \hat f(x) + \mathcal{N}(0, s^\star(x))$ with $s^\star(x) = \exp(a^\top \phi(x) + b)$ for a fixed random projection $\phi$, spanning three decades of variance. Now $s^\star$ is known exactly.

**Scale.** $n = 200{,}000$ train, $50{,}000$ test, $d = 128$; ResNet-MLP with $5\times10^6$ parameters; 5 seeds per arm; ~200 GPU-hours total on one A100 — deliberately small enough to run in a week.

**Control arm.** Two-stage oracle-free baseline: train the mean with MSE alone, then fit a second network to $\log(y-\hat\mu)^2$ on held-out data. This is the strongest method that structurally cannot suffer the coupling pathology; every proposed loss must beat it or is not worth the complexity.

**Deciding number.** Conditional log-variance error
$$\Delta = \sqrt{\tfrac{1}{m}\sum_j \big(\log\sigma^2_\theta(x_j) - \log s^\star(x_j)\big)^2}$$
on the test set, reported per decile of $s^\star$. A method wins only if $\Delta$ beats the two-stage baseline in **every** decile, including the highest-noise one where collapse-driven methods discard data. Secondary gate: test RMSE within 1% of MSE-only training. Predicted outcome, worth stating in advance: $\beta$-NLL and stop-gradient pass the RMSE gate and fail the top-decile $\Delta$ gate — meaning the field has fixed the mean-fit symptom, not the variance estimate.

## 9. Key References

- **[Foundational]** D. A. Nix, A. S. Weigend. *Estimating the mean and variance of the target probability distribution.* IEEE ICNN, 1994.
- **[Foundational]** T. Gneiting, A. E. Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102(477), 2007.
- **[Foundational]** A. Kendall, Y. Gal. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS, 2017. — arXiv:1703.04977
- **[Foundational]** B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS, 2017. — arXiv:1612.01474
- **[SOTA]** M. Seitzer, A. Tangemann, S. Bauer, B. Schölkopf. *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks.* ICLR, 2022.
- **[SOTA]** A. Stirn, H. Barac, D. Pe'er, D. A. Knowles. *Faithful Heteroscedastic Regression with Neural Networks.* AISTATS, 2023.
- **[SOTA]** A. Immer, E. Palumbo, A. Marx, J. E. Vogt. *Effective Bayesian Heteroscedastic Regression with Deep Neural Networks.* NeurIPS, 2023.
- **[Theory]** E. Wong-Toi, A. Boyd, V. Fortuin, S. Mandt. *Understanding Pathologies of Deep Heteroskedastic Regression.* AISTATS, 2024.
- N. Skafte (Detlefsen), M. Jørgensen, S. Hauberg. *Reliable Training and Estimation of Variance Networks.* NeurIPS, 2019.
- L. Sluijterman, E. Cator, T. Heskes. *Optimal Training of Mean Variance Estimation Neural Networks.* Neurocomputing, 2024.
- A. Amini, W. Schwarting, A. Soleimany, D. Rus. *Deep Evidential Regression.* NeurIPS, 2020. — and the critique: N. Meinert, J. Gawlikowski, A. Lavin. *The Unreasonable Effectiveness of Deep Evidential Regression.* AAAI, 2023.
- Y. Romano, E. Patterson, E. Candès. *Conformalized Quantile Regression.* NeurIPS, 2019.
- **[Survey]** J. Gawlikowski et al. *A Survey of Uncertainty in Deep Neural Networks.* Artificial Intelligence Review, 2023.

## 10. Worked Example

Two-cluster synthetic problem. $x$ uniform on $[-1,1]$; $f^\star(x)=\sin(3x)$; noise $s^\star(x)=10^{-4}$ for $x<0$ (10,000 points) and $s^\star(x)=10^{-1}$ for $x\ge0$ (10,000 points) — a 1000:1 variance ratio. Model: MLP, 2 hidden layers of 64, two heads, $\log\sigma^2$ parameterization, Adam at $10^{-3}$.

Take the state after 200 steps, before either head is converged. Suppose $\sigma^2_\theta \approx 10^{-3}$ on the low-noise side (already shrinking) and $\sigma^2_\theta\approx 10^{-2}$ on the high-noise side, with equal mean residual $|y-\mu| \approx 0.3$ everywhere. Mean-gradient magnitudes:

```
low-noise  point:  0.3 / 1e-3  = 300
high-noise point:  0.3 / 1e-2  =  30
ratio                          =  10x
```

The low-noise half now contributes 10× the mean gradient per example. Its residuals fall, its $\sigma^2$ falls further, and the ratio grows. By convergence, with $\epsilon_{\min}=10^{-6}$, the low-noise side can reach a gradient weight of $10^6$ — five orders of magnitude above the high-noise side, which is then fit no better than the constant predictor. Final numbers in this regime: RMSE on $x\ge0$ roughly $\sqrt{s^\star + \text{bias}^2} \approx 0.42$ versus $0.32$ ($=\sqrt{0.1}$, the irreducible floor) for MSE training — a 30% excess that is pure optimization pathology.

Now the obstruction. Test NLL for the collapsed model on the low-noise half is about $-\tfrac12\log(2\pi\cdot10^{-4}) \approx 4.1$ nats *better* per point than a correctly-fit model would be if it slightly overestimated variance there. Averaged over the test set, the collapsed model can post a *lower* total NLL than the faithful one, because the pathologically confident half pays no penalty when it happens to be right. And $\mathrm{ECE}_q$, computed marginally over all 20,000 points, mixes an over-confident half with an under-confident half; the two errors cancel and the reported ECE is small.

So the standard metrics score the broken model well. Only $\Delta$ per decile of $s^\star$ — which needs ground truth we do not have outside constructions like this one — exposes it. That is the reason this problem is only partially solved: the mean-fit symptom has three working fixes, and the variance estimate itself still has no field-usable test.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*