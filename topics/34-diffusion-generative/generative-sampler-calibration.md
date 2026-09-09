---
id: 34-diffusion-generative/generative-sampler-calibration
title: "Calibrated Uncertainty From Generative Samplers"
topic: 34-diffusion-generative
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibrated Uncertainty From Generative Samplers

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/generative-sampler-calibration` · **Status:** methodologically-blocked

## 1. Problem Statement

A conditional generative model (diffusion, flow-matching, autoregressive) is increasingly used not to produce *one* output but to produce an *ensemble* that is read as a posterior: 50 weather trajectories, 32 MRI reconstructions, $k$ protein backbones. The claim implicit in that use is that the sampler's output distribution $q(\cdot\mid y)$ is a calibrated stand-in for the true posterior $p(x\mid y)$.

Three variants, with very different difficulty:

- **Measurement.** Given samples from $q(\cdot\mid y)$ and *one* observed $x^\star$ per condition $y$, decide whether $q$ is calibrated — and for *which* functionals. This is the blocked variant: no agreed definition of joint calibration in high dimension that is both estimable from one draw per condition and not satisfiable by a wrong model.
- **Method.** Post-hoc recalibrate a sampler so a target functional is calibrated, without destroying sample quality. Solved for scalar functionals (conformal, isotonic), open for joint/semantic ones.
- **Theory.** Bound the calibration error of the *sampler* (finite steps, learned score, guidance) in terms of score error $\epsilon_{\text{score}}$ and step count $N$. Partially solved in $\mathrm{TV}$/$W_2$; unsolved for calibration functionals, which are not Lipschitz in these metrics.

Solving it means: a statistic $\widehat{C}$, estimable from $\{(y_i, x_i^\star)\}_{i=1}^n$ with one ground-truth draw each, that is (a) zero iff $q=p$ up to the decision-relevant $\sigma$-algebra, (b) has known finite-sample bias, and (c) is not trivially passed by a model that gets marginals right and dependence wrong.

## 2. Formal Setting

Condition $y \in \mathcal{Y}$, target $x \in \mathbb{R}^d$, true conditional $p(x \mid y)$, sampler law $q_\theta^{(N)}(x\mid y)$ where $N$ is the number of function evaluations (NFE). Data: $(y_i, x_i^\star) \sim p(y)p(x\mid y)$, $i=1..n$; per condition we draw $m$ samples $x_i^{(1..m)} \sim q_\theta^{(N)}$.

**Functional calibration.** For $\phi:\mathbb{R}^d\to\mathbb{R}$ let $F_{i,\phi}(t) = \Pr_{x\sim q}[\phi(x)\le t \mid y_i]$, estimated by the empirical CDF over $m$ samples. The PIT (probability integral transform) value is $u_{i,\phi}=F_{i,\phi}(\phi(x_i^\star))$. $q$ is **$\phi$-calibrated** iff $u_\phi \sim \mathrm{Unif}[0,1]$. Measured as
$$\widehat{\mathrm{CE}}_\phi \;=\; \sup_{t\in[0,1]}\Big|\tfrac{1}{n}\textstyle\sum_i \mathbf{1}\{u_{i,\phi}\le t\} - t\Big|,$$
a Kolmogorov–Smirnov statistic with the $m$-sample discretization bias $O(1/m)$ and sampling error $O(n^{-1/2})$.

**Coverage.** For a set map $y\mapsto \mathcal{C}_\alpha(y)$ built from the $m$ samples, marginal coverage is $\Pr[x^\star\in\mathcal{C}_\alpha(y)]$; measured as the empirical rate over $n$ conditions. Conditional coverage $\Pr[x^\star\in\mathcal{C}_\alpha(y)\mid y]$ is **not** estimable from one draw per $y$ without smoothness assumptions.

**Sharpness.** $\mathbb{E}|\mathcal{C}_\alpha(y)|$ or per-variable spread. Calibration without sharpness is free: the climatological ensemble is perfectly calibrated for every marginal.

**Proper scores.** $\mathrm{CRPS}(F,x^\star)=\int (F(t)-\mathbf{1}\{t\ge x^\star\})^2 dt$, estimated as $\frac{1}{m}\sum_j |x^{(j)}-x^\star| - \frac{1}{2m^2}\sum_{j,k}|x^{(j)}-x^{(k)}|$, with the fair/unbiased $m(m-1)$ correction. Energy score generalizes to $\mathbb{R}^d$: $\mathrm{ES}=\mathbb{E}\|X-x^\star\|-\tfrac12\mathbb{E}\|X-X'\|$.

**Assumptions, and which break.**
1. *$\phi$ is fixed before seeing data.* Violated: $\phi$ is chosen after inspecting failures, so $\widehat{\mathrm{CE}}_\phi$ is a post-selection statistic with no valid $p$-value.
2. *$(y_i,x_i^\star)$ i.i.d.* Violated in weather (spatiotemporal correlation shrinks effective $n$ by 1–2 orders) and in medical imaging (patient-level clustering).
3. *$x^\star$ is a draw from $p(\cdot\mid y)$.* Violated wherever the "ground truth" is itself a reconstruction (fully-sampled MRI is a low-noise estimate, not a posterior draw) or an analysis product (ERA5 is a model-data hybrid).
4. *$m\to\infty$ negligible.* Violated: $m\in\{8,32,50\}$ in practice; PIT with $m=50$ has resolution $1/51$, and rank histograms at small $m$ confound sampler bias with binning.
5. *Decision-relevant $\sigma$-algebra is the full Borel one.* Violated by construction — nobody needs calibration on all of $\mathbb{R}^{10^6}$.

## 3. State of the Art

**Established.**
- Score-based samplers have polynomial convergence guarantees under an $L^2$-accurate score: Chen, Chewi, Li, Li, Salim, Zhang (ICLR 2023) give $\mathrm{TV}$ bounds with iteration complexity polynomial in $d$ and $1/\epsilon$, needing no log-concavity. This bounds distribution error, not calibration error.
- Distribution-free coverage: conformal prediction (Vovk et al. 2005) and RCPS applied to image-to-image models (Angelopoulos, Bates, Fisch, Lei, Schuster, ICML 2022) give *marginal, per-pixel* risk control at a chosen $\alpha$, with finite-sample validity. Established and reproduced.
- Scalar recalibration: Kuleshov, Fenner, Ermon (ICML 2018) isotonic recalibration of regression CDFs; Guo et al. (ICML 2017) temperature scaling. Both established for 1-D outputs.
- Ensemble diffusion beats physics ensembles on *marginal* scores: GenCast (Price et al., *Nature* 637, 2025), 0.25°, 50-member, 15-day, better CRPS than ECMWF ENS on 97.2% of 1320 variable/lead-time targets. This is an established benchmark number for per-variable CRPS, not evidence of joint calibration.

**Claimed but unablated.**
- That posterior-sampling algorithms for inverse problems (DPS — Chung, Kim, McCann, Klasky, Ye, ICLR 2023) approximate the true posterior. DPS uses an uncontrolled Jacobian approximation; its samples are diverse, but the diversity has not been shown to match posterior width. SMC-based correctors (Twisted Diffusion Sampler, Wu, Trippe, Naesseth, Blei, Cunningham, NeurIPS 2023; MCGDiff, Cardoso et al., ICLR 2024) are asymptotically exact for linear-Gaussian likelihoods and are the correct control arm — rarely used as one.
- That guidance scale is an uncertainty knob. Classifier-free guidance $w>1$ provably samples a tilted distribution $\propto p(x\mid y)^w p(x)^{1-w}$, not $p(x\mid y)$; any calibration claim at $w>1$ is a claim about the wrong target.

## 4. What Is Known

- **Binned calibration error is biased downward.** Kumar, Liang, Ma (NeurIPS 2019) show binned ECE systematically understates true calibration error; for Platt-scaled models on CIFAR-10/ImageNet the true error was measured at roughly $1.5$–$3\times$ the binned estimate. Vaicenavicius et al. (AISTATS 2019) give the same result independently. Scale: standard CIFAR/ImageNet classifiers, $n\approx 10^4$ test points.
- **Modern nets are miscalibrated by default.** Guo et al. (2017): ResNet-110 on CIFAR-100 reaches ECE $\approx 16\%$; temperature scaling drops it below $1\%$ without changing accuracy. Scale: $10^4$ images, single scalar parameter.
- **Distribution shift destroys calibration and ensembles help but do not fix it.** Ovadia et al. (NeurIPS 2019), ImageNet-C/ CIFAR-C, across 5 method families.
- **Sampler hyperparameters move calibration without moving quality metrics.** Stochastic churn and step count in EDM (Karras, Aittala, Aila, Laine, NeurIPS 2022) alter the sample distribution at fixed FID; FID is a 2-moment Inception-space statistic (Theis, van den Oord, Bethge, ICLR 2016 argued the general point) and is near-blind to variance changes of the size that flip coverage.
- **Diffusion models can generate off-manifold compositions.** Aithal, Maini, Lipton, Kolter (NeurIPS 2024) show "mode interpolation": samples between training modes with high model likelihood — mass placed where $p$ has none.
- **Two diffusion models trained on disjoint data converge to nearly the same denoiser** (Kadkhodaie, Guth, Simoncelli, Mallat, ICLR 2024). Implication for this problem: sampler-to-sampler agreement is *not* evidence of posterior correctness — an ensemble-of-models diagnostic will agree while both are wrong.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted definition of joint calibration in $\mathbb{R}^d$ that is estimable from one ground-truth draw per condition and not passed by a model with correct marginals and wrong dependence. Rank histograms, per-variable CRPS, per-pixel coverage, and PIT are all marginal-family statistics. The multivariate rank histogram and energy score are joint but have low power in $d\gg 1$; the energy score's insensitivity to misspecified dependence is a known deficiency in the forecast-verification literature.
- **Theoretically open.** No bound of the form $\widehat{\mathrm{CE}}_\phi \le g(\epsilon_{\text{score}}, N, L_\phi)$ for non-Lipschitz decision functionals. Small $\mathrm{TV}$ does not control tail-event probability ratios, which is what calibration on rare events needs.
- **Empirically open.** Nobody has run the ablation "hold FID/CRPS fixed, sweep NFE and stochasticity, report coverage" at frontier scale with an asymptotically exact control arm. It is runnable today on any linear inverse problem.

## 6. Why It Is Hard

The specific obstruction is **the evaluation does not measure the thing it names**, compounded by **absent ground truth**. Every deployed metric — per-variable CRPS, rank histogram, per-pixel interval coverage — is a functional of one-dimensional marginals. A sampler can match all $d$ marginals exactly while getting the joint arbitrarily wrong (Section 10 gives an exact instance). Because we observe one $x^\star$ per $y$, the only cheap consistency checks are marginal ones. The joint alternatives (energy score, multivariate ranks) trade the blindness for power loss: their discriminating ability degrades with $d$, so at $d=10^6$ they fail to reject models that are visibly wrong. Second, ground truth is contaminated: assumption 3 above means the residual $x^\star - \mathbb{E}_q[x]$ mixes sampler error with reference-product error, and the two are not separable without a synthetic problem where $p(x\mid y)$ is known in closed form.

## 7. Current Research (as of 2026)

- **Exact conditional sampling as a yardstick.** SMC-based diffusion posterior samplers (Wu/Trippe/Cunningham, Columbia; Cardoso, Le Corff, Moulines, Olsson, Institut Polytechnique de Paris) give consistent baselines for linear-Gaussian likelihoods. The obvious use — as a control arm for measuring DPS/DDRM bias — is still under-exploited. *(frontier — verify)*
- **Conformal on generative outputs.** Extending RCPS/conformal risk control to semantic functionals of generated images rather than pixels (Angelopoulos, Bates, Jordan, Malik and collaborators, Berkeley).
- **Ensemble weather as the honest testbed.** GenCast, and follow-on work on spatially-coherent verification, because meteorology already has 40 years of proper-scoring practice and $n\approx 10^4$ independent forecast cases per year. Groups: Google DeepMind; ECMWF (AIFS-ENS); NVIDIA (StormCast/CorrDiff). *(frontier — verify for 2026 releases)*
- **Calibration of LLM sample sets** via semantic-equivalence clustering (Kuhn, Gal, Farquhar, ICLR 2023) — the same measurement problem in a discrete space, with the same marginal-vs-joint failure.

## 8. Concrete Next Experiment

**Scale.** Linear inverse problem with an *analytically known* posterior: pretrained EDM/DDPM on CIFAR-10 or a 2-component GMM prior in $d=64$; Gaussian likelihood $y = Ax+\eta$, $A$ a random $32\times d$ projection, $\eta\sim\mathcal{N}(0,\sigma^2 I)$ with $\sigma\in\{0.05,0.2\}$. $n=2000$ conditions, $m=64$ samples each. Single A100-day.

**Control arm.** Two, both required: (i) the closed-form posterior for the GMM prior; (ii) MCGDiff / Twisted Diffusion Sampler with 4096 particles, which is consistent for this likelihood. Test arms: DPS, DDRM, $\Pi$GDM, at NFE $\in\{20, 50, 250, 1000\}$ and stochasticity $S_{\text{churn}}\in\{0, 40\}$.

**The deciding number.** Report, at matched sample quality (FID within $\pm 0.3$ of the NFE $=1000$ arm), the **coverage of the 90% highest-density region of the true posterior**, $\kappa = \Pr[x^\star \in \mathcal{C}_{0.90}^{q}]$, alongside per-coordinate PIT KS-statistic $\widehat{\mathrm{CE}}$. The question is settled in the "metrics are blind" direction if any two arms differ by $|\Delta\kappa| \ge 0.10$ while $|\Delta\widehat{\mathrm{CE}}| \le 0.01$ and FID matches. If instead $\kappa$ tracks the marginal statistics to within 0.02 across every arm, the marginal metrics are adequate and the problem downgrades from methodologically blocked to empirically open.

## 9. Key References

- **[Foundational]** A. P. Dawid. *The Well-Calibrated Bayesian.* Journal of the American Statistical Association, 1982.
- **[Foundational]** T. Gneiting, F. Balabdaoui, A. Raftery. *Probabilistic Forecasts, Calibration and Sharpness.* JRSS-B, 2007.
- **[Foundational]** T. Gneiting, A. Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA, 2007.
- **[Foundational]** V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[Foundational]** L. Theis, A. van den Oord, M. Bethge. *A Note on the Evaluation of Generative Models.* ICLR, 2016. — arXiv:1511.01844
- **[SOTA]** I. Price, A. Sanchez-Gonzalez, F. Alet, T. Ewalds, et al. *Probabilistic Weather Forecasting with Machine Learning.* Nature 637, 2025.
- **[SOTA]** S. Chen, S. Chewi, J. Li, Y. Li, A. Salim, A. R. Zhang. *Sampling Is as Easy as Learning the Score.* ICLR, 2023. — arXiv:2209.11215
- **[SOTA]** L. Wu, B. Trippe, C. Naesseth, D. Blei, J. Cunningham. *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models.* NeurIPS, 2023.
- **[SOTA]** G. Cardoso, Y. J. El Idrissi, S. Le Corff, E. Moulines. *Monte Carlo Guided Diffusion for Bayesian Linear Inverse Problems.* ICLR, 2024.
- **[SOTA]** H. Chung, J. Kim, M. McCann, M. Klasky, J. C. Ye. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* ICLR, 2023. — arXiv:2209.14687
- **[SOTA]** A. Angelopoulos, A. Bates, A. Fisch, L. Lei, T. Schuster. *Image-to-Image Regression with Distribution-Free Uncertainty Quantification.* ICML, 2022. — arXiv:2202.05265
- **[Method]** V. Kuleshov, N. Fenner, S. Ermon. *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML, 2018. — arXiv:1807.00263
- **[Method]** A. Kumar, P. Liang, T. Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019.
- **[Method]** J. Vaicenavicius, D. Widmann, C. Andersson, F. Lindsten, J. Roll, T. Schön. *Evaluating Model Calibration in Classification.* AISTATS, 2019.
- **[Empirical]** C. Guo, G. Pleiss, Y. Sun, K. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Empirical]** Y. Ovadia, E. Fertig, J. Ren, Z. Nado, et al. *Can You Trust Your Model's Uncertainty?* NeurIPS, 2019.
- **[Empirical]** S. Aithal, P. Maini, Z. Lipton, J. Z. Kolter. *Understanding Hallucinations in Diffusion Models through Mode Interpolation.* NeurIPS, 2024.
- **[Empirical]** Z. Kadkhodaie, F. Guth, E. Simoncelli, S. Mallat. *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations.* ICLR, 2024.
- **[Survey]** A. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021. — arXiv:2107.07511

## 10. Worked Example

Two dimensions, closed form, no training needed.

True posterior: $p = \tfrac12\mathcal{N}\!\big((-2,-2),I\big) + \tfrac12\mathcal{N}\!\big((2,2),I\big)$.
Sampler output: $q = \tfrac12\mathcal{N}\!\big((-2,2),I\big) + \tfrac12\mathcal{N}\!\big((2,-2),I\big)$.

**Every marginal statistic is exactly equal.** Under both $p$ and $q$, each coordinate has law $\tfrac12\mathcal{N}(-2,1)+\tfrac12\mathcal{N}(2,1)$. So:

| statistic | under $p$ | under $q$ |
|---|---|---|
| marginal mean, each coord | $0$ | $0$ |
| marginal variance $=1+4$ | $5$ | $5$ |
| per-coordinate PIT | $\mathrm{Unif}[0,1]$ | $\mathrm{Unif}[0,1]$ |
| per-coordinate $\widehat{\mathrm{CE}}$ (population) | $0$ | $0$ |
| per-coordinate CRPS | identical | identical |
| 90% marginal interval coverage | $0.90$ | $0.90$ |
| $\mathrm{Corr}(x_1,x_2) = \pm4/5$ | $+0.80$ | $-0.80$ |

**The decision-relevant event.** Take $\phi(x)=\mathbf{1}\{x_1>0 \wedge x_2>0\}$ — "both variables exceed threshold", the shape of a compound flood warning or a joint lesion-presence call.

$$\Pr_p[\phi=1] = \tfrac12\Phi(2)^2 + \tfrac12\Phi(-2)^2 = \tfrac12(0.9772)^2 + \tfrac12(0.02275)^2 = 0.4779$$
$$\Pr_q[\phi=1] = \Phi(2)\Phi(-2) = 0.9772 \times 0.02275 = 0.0222$$

**The obstruction, visible.** The sampler reports a $2.2\%$ chance of an event whose true probability is $47.8\%$ — off by $21\times$, a Brier-score gap of about $0.21$ on this single functional. Every metric a diffusion-ensemble paper actually reports (per-variable CRPS, rank histogram, per-pixel 90% coverage, spread-skill ratio) is **exactly** zero-error here. Rejecting $q$ requires either knowing $\phi$ in advance — which reintroduces assumption 1, no valid post-hoc $p$-value — or a joint test whose power at $d=2$ is fine and at $d=10^6$ is not. That gap between "estimable" and "informative" is the blockage, and it is why the status is methodologically blocked rather than empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*