---
id: 33-uncertainty-calibration/diffusion-model-likelihood-calibration
title: "Calibration of Diffusion Model Likelihoods"
topic: 33-uncertainty-calibration
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration of Diffusion Model Likelihoods

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/diffusion-model-likelihood-calibration` · **Status:** open

## 1. Problem Statement

A diffusion model trained on $p_{\text{data}}$ can report a number $\log p_\theta(x)$ for any input $x$ — either an ELBO, or an exact probability-flow ODE log-density. The problem: **decide whether that number is a calibrated statement about probability, and if not, correct it.**

Three variants, routinely conflated:

- **Measurement.** Given a trained model and a held-out set, produce a statistic that is zero if and only if $p_\theta = p_{\text{data}}$ in the relevant sense, and whose estimator bias is bounded. Bits/dim is *not* such a statistic: it is a one-number cross-entropy estimate that cannot distinguish a model that is over-dispersed from one that is under-dispersed.
- **Method.** Post-hoc recalibration. For classifiers, temperature scaling fixes most miscalibration with one parameter. The density analogue — a map $T$ with $\log \tilde p = T(\log p_\theta, x)$ that is both normalized and better calibrated — has no accepted form.
- **Theory.** Does score-matching training with the standard weighting induce any calibration guarantee, or only an $L^2$ score bound that is compatible with arbitrarily bad tail densities?

Solving it means: a decision procedure that takes $(\theta, x)$ and returns a probability statement whose empirical coverage matches its nominal level, with a test that would detect failure.

## 2. Formal Setting

Data $x_0 \in \{0,\dots,255\}^D$, $D = 3072$ for CIFAR-10, dequantized to $\mathbb{R}^D$. Forward SDE (Song et al., 2021):

$$dx = f(x,t)\,dt + g(t)\,dw, \qquad t \in [0,T].$$

The probability-flow ODE has drift $\tilde f_\theta(x,t) = f(x,t) - \tfrac{1}{2}g(t)^2 s_\theta(x,t)$ and gives an exact log-density by instantaneous change of variables:

$$\log p_\theta(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot \tilde f_\theta(x_t,t)\,dt.$$

**As actually measured**, the divergence is a Skilling–Hutchinson estimate $\hat{\nabla\cdot} = \epsilon^\top (\partial_x \tilde f_\theta) \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$, with one or a few probes, and the integral is an adaptive RK45 quadrature with tolerance $\texttt{atol}=\texttt{rtol}\in[10^{-5},10^{-3}]$. Reported bits/dim for $x\in[0,1]^D$:

$$\text{bpd}(x) = -\frac{\log p_\theta(x)}{D \ln 2} + 8.$$

Candidate calibration criteria, each measurable:

1. **Typicality / self-consistency.** Let $F_\theta$ be the law of $\log p_\theta(X)$ for $X\sim p_\theta$ (model samples) and $G_\theta$ the law for $X\sim p_{\text{data}}$ (held-out). If $p_\theta=p_{\text{data}}$ then $F_\theta = G_\theta$. Test statistic: $\Delta = \mathbb{E}_{p_{\text{data}}}[\log p_\theta] - \mathbb{E}_{p_\theta}[\log p_\theta]$, or a two-sample KS distance.
2. **Highest-density-region coverage.** $R_\alpha = \{x : \log p_\theta(x) \ge \tau_\alpha\}$ with $\tau_\alpha$ the $(1-\alpha)$-quantile under $p_\theta$; calibrated iff $\Pr_{p_{\text{data}}}[X \in R_\alpha] = \alpha$ for all $\alpha$.
3. **Conditional PIT.** For a scalar coordinate or projection $u$, the probability integral transform of $p_\theta(u \mid x_{\setminus u})$ should be $\mathrm{Unif}[0,1]$.

Assumptions and their status:

- *Absolute continuity of $p_{\text{data}}$ w.r.t. Lebesgue measure.* **Violated.** Natural image data is widely modelled as concentrated near a low-dimensional manifold; then no finite Lebesgue density exists and the target of calibration is undefined without a dequantization convention.
- *Uniform $L^2$ score accuracy.* **Violated in the tails** — training draws $x_t$ from noised data, so low-density regions get near-zero training signal, yet ODE likelihoods integrate through them.
- *Exact ODE integration and unbiased divergence.* $\hat{\nabla\cdot}$ is unbiased for the divergence but $\log p_\theta$ is a nonlinear functional of the trajectory; the reported bpd is **not** an unbiased estimate of $\mathbb{E}[\log p_\theta]$ at finite probe count.
- *The likelihood model equals the deployed sampler.* **Violated.** The ODE density is not the density of the discrete DDPM/DDIM/EDM sampler actually run.

## 3. State of the Art

**Established.** Exact ODE likelihoods (Song et al., ICLR 2021); the diffusion ELBO with likelihood weighting $g(t)^2$ is a valid bound (Song et al., NeurIPS 2021); VDM makes the bound continuous-time and variance-reduced (Kingma et al., NeurIPS 2021). Kingma & Gao (NeurIPS 2023) established that essentially all common weighted diffusion objectives are ELBOs under Gaussian noise data augmentation when the weighting is monotonic — this is a proof, not a benchmark. Best density numbers: VDM ≈ 2.65 bpd on CIFAR-10 and ≈ 3.40 bpd on ImageNet-64; Zheng et al. (ICML 2023, "Improved Techniques for MLE for Diffusion ODEs") ≈ 2.56 bpd on CIFAR-10. These are **benchmark numbers only** — they report cross-entropy, not calibration.

**Claimed but unablated.** That better bpd implies better-calibrated density. No paper reports coverage curves or PIT diagnostics for a diffusion likelihood at scale. OOD-detection papers using diffusion reconstruction error (Graham et al., CVPR-W 2023) report AUROC, which conflates a monotone re-scoring with a calibration fix.

**Theory SOTA.** Chen et al. (ICLR 2023) give polynomial-time TV/KL convergence for the sampler under an $L^2$-accurate score and mild data assumptions. This bounds the *distributional* error of samples; it gives no pointwise control of $\log p_\theta(x)$ at a specific $x$. Le Lan & Dinh (*Entropy*, 2021) prove that even a perfect density model gives no guarantee for likelihood-thresholded anomaly detection, because the decision is not invariant to reparameterization.

## 4. What Is Known

- **Likelihood tracks complexity, not membership.** Glow trained on CIFAR-10 assigns SVHN *higher* likelihood (≈ 2.4 bpd vs ≈ 3.46 bpd on CIFAR-10 test) — Nalisnick et al., ICLR 2019, at 3072 dimensions. The same inversion reproduces for diffusion likelihoods.
- **A compression correction largely removes it.** Serrà et al. (ICLR 2020) show $S(x) = -\log p_\theta(x) - L(x)$, with $L$ a PNG/FLIF codelength, restores AUROC near 0.9+ on CIFAR-10 vs SVHN — evidence the raw likelihood is miscalibrated by an input-complexity term.
- **Typicality, not density, is the right region.** Nalisnick et al. (2019, typicality test) show high-dimensional data does not concentrate at the density mode; single-sample thresholding is the wrong test.
- **Likelihood and sample quality are decoupled** (Theis et al., ICLR 2016) — a model can be near-optimal in bpd and produce poor samples, and vice versa, at any $D$.
- **The gap between ELBO and ODE likelihood is real and model-dependent**, typically a few hundredths to a few tenths of a bpd on CIFAR-10; the two disagree about *which* samples are unlikely.

## 5. What Is Not Known

- **Theoretically open.** Whether any bound of the form "$L^2$ score error $\le \varepsilon$ $\Rightarrow$ $|\log p_\theta(x) - \log p_{\text{data}}(x)| \le h(\varepsilon)$ on a set of $p_{\text{data}}$-measure $1-\delta$" holds. No proof either way; the manifold case suggests the LHS may be vacuous.
- **Methodologically blocked.** The target itself. Under the manifold hypothesis $p_{\text{data}}$ has no Lebesgue density, so "calibrated likelihood" has no ground truth. Dequantization choice shifts the answer; the $x \mapsto x/256$ rescaling shifts bpd by exactly 8 (Section 10). Until a coordinate-free criterion is fixed, calibration is not well defined.
- **Empirically open.** Coverage-curve calibration (criterion 2 above) is runnable today on any released CIFAR-10/ImageNet-64 diffusion checkpoint and has not been published at that scale. Likewise the ODE-vs-sampler density mismatch.

## 6. Why It Is Hard

Three named obstructions, in order of severity.

1. **Absent ground truth.** There is no reference $\log p_{\text{data}}(x)$ for real images. Every reported result compares models to each other. Fixing this requires a synthetic ground-truth distribution, which then may not be representative.
2. **Non-identifiability under reparameterization.** Density is a coordinate-dependent object; likelihood-thresholded decisions can be inverted by a smooth bijection of input space (Le Lan & Dinh, 2021). Any calibration claim is a claim about a chosen chart, not about the model.
3. **Confounded measurement.** The reported bpd bundles four error sources — score error, ODE discretization, Hutchinson probe variance, and dequantization — with no published decomposition. A 0.05 bpd difference between two methods is not attributable.

Compute is a secondary cost, not the obstruction: a tight-tolerance ODE likelihood is $\sim$200–1000 network evaluations per image, so 10k images is $\sim$10 GPU-hours on a CIFAR-scale model.

## 7. Current Research (as of 2026)

- **ELBO-as-objective unification** (Kingma & Gao line, Google DeepMind) — established; the open follow-up is whether the ELBO gap is uniform over $x$.
- **Manifold-aware likelihood** — Loaiza-Ganem, Cresswell, Caterini and collaborators (Layer 6 AI / Vector) on manifold overfitting and density on submanifolds; the most direct attack on the ill-posed target. *(frontier — verify current results.)*
- **Score-based OOD scoring** using reconstruction error, noise-conditioned scores, or likelihood ratios against a background model — active but evaluated by AUROC, which does not test calibration.
- **Sampler-consistent densities** — computing the exact density of the discrete sampler rather than the continuous ODE. *(frontier — verify.)*
- **Conformal wrappers** around generative scores, which sidestep calibration of $\log p_\theta$ by calibrating only a decision threshold on held-out data. This works and is under-used, but yields set-valued guarantees, not a calibrated density.

## 8. Concrete Next Experiment

**Question.** Is the ODE likelihood of a SOTA diffusion model self-consistent — is $\Delta = \mathbb{E}_{p_{\text{data}}}[\log p_\theta] - \mathbb{E}_{p_\theta}[\log p_\theta]$ zero?

**Scale.** One released CIFAR-10 checkpoint (VDM- or EDM-class, $\le$ 100M params). $n = 10{,}000$ held-out test images and $n = 10{,}000$ model samples drawn with a high-NFE ODE sampler. Exact likelihood with $\texttt{atol}=\texttt{rtol}=10^{-5}$, 8 Hutchinson probes per image. Cost $\approx$ 20–40 A100-hours.

**Control arm.** The *same architecture and training recipe* fitted to a synthetic 3072-dimensional distribution with an analytically known density — a 50-component Gaussian mixture with full-rank covariances. Here true $\log p$ is computable, so the same pipeline yields the estimator's bias and the $\Delta$ that a *correctly specified* model produces under finite-sample and finite-tolerance error. Second control: repeat CIFAR-10 with tolerance $10^{-3}$ and 1 probe to size the numerics term.

**Deciding number.** $\Delta$ in bpd with a bootstrap 95% CI, compared to the control's $|\Delta_{\text{ctrl}}|$. If $|\Delta| < 3\,|\Delta_{\text{ctrl}}|$ the likelihood is self-consistent at this resolution and miscalibration must be sought in higher moments (report the KS statistic between $F_\theta$ and $G_\theta$ as the secondary readout). If $|\Delta| > 0.10$ bpd with $|\Delta_{\text{ctrl}}| < 0.01$ bpd, the model's own samples sit in a different density band than real data — direct evidence that bpd rankings are miscalibrated, and the first published instance of it.

$\Delta = 0$ is necessary, not sufficient: it is a difference of cross-entropies and can vanish by cancellation. Publish the full coverage curve $\alpha \mapsto \Pr_{p_{\text{data}}}[X \in R_\alpha]$ alongside it.

## 9. Key References

- **[Foundational]** Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. — arXiv:2011.13456
- **[Foundational]** Song, Durkan, Murray, Ermon. *Maximum Likelihood Training of Score-Based Diffusion Models.* NeurIPS 2021. — arXiv:2101.09258
- **[Foundational]** Theis, van den Oord, Bethge. *A Note on the Evaluation of Generative Models.* ICLR 2016. — arXiv:1511.01844
- **[SOTA]** Kingma, Salimans, Poole, Ho. *Variational Diffusion Models.* NeurIPS 2021. — arXiv:2107.00630
- **[SOTA]** Kingma, Gao. *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.* NeurIPS 2023.
- **[SOTA]** Zheng, Lu, Bao, Chen, Li, Zhu. *Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs.* ICML 2023.
- **[Key negative result]** Nalisnick, Matsukawa, Teh, Görür, Lakshminarayanan. *Do Deep Generative Models Know What They Don't Know?* ICLR 2019. — arXiv:1810.09136
- **[Key negative result]** Nalisnick, Matsukawa, Teh, Lakshminarayanan. *Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality.* 2019. — arXiv:1906.02994
- **[Theory]** Le Lan, Dinh. *Perfect Density Models Cannot Guarantee Anomaly Detection.* Entropy, 2021.
- **[Theory]** Chen, Chewi, Li, Li, Salim, Zhang. *Sampling is as Easy as Learning the Score.* ICLR 2023.
- **[Correction method]** Serrà, Álvarez, Gómez, Slizovskaia, Núñez, Luque. *Input Complexity and Out-of-Distribution Detection with Likelihood-Based Generative Models.* ICLR 2020.
- **[Survey]** Loaiza-Ganem, Ross, Hosseinzadeh, Caterini, Cresswell. *Deep Generative Models through the Lens of the Manifold Hypothesis: A Survey and New Connections.* TMLR, 2024.
- **[Background]** Gneiting, Balabdaoui, Raftery. *Probabilistic Forecasts, Calibration and Sharpness.* JRSS-B, 2007.

## 10. Worked Example

Take a CIFAR-10 model reporting 2.65 bpd on the test set. $D = 3072$, so the total log-density is

$$-\log_2 p_\theta(x) = 2.65 \times 3072 \approx 8{,}141 \text{ bits}.$$

Now apply the same model's density under a different chart. Rescale inputs from $[0,1]^D$ to $[0,255]^D$. The Jacobian of $x \mapsto 256x$ contributes $\log_2 256 = 8$ bits per dimension, so the density becomes $2.65 + 8 = 10.65$ bpd — 32,717 bits. Nothing about the model changed. Any statement of the form "$x$ is unlikely because $\log p_\theta(x) < \tau$" is a statement about the chart, and a smooth non-affine reparameterization can reorder two inputs' likelihoods arbitrarily (Le Lan & Dinh, 2021). **Obstruction 2 made concrete.**

Now the ordering failure. Score the SVHN test set with a CIFAR-10 model. The Glow measurement (Nalisnick et al., 2019) gives roughly 2.4 bpd for SVHN against 3.46 bpd for CIFAR-10 test — SVHN is judged about $1.06 \times 3072 \approx 3{,}260$ bits *more likely*, a likelihood ratio of $2^{3260}$ in favour of the data the model never saw. The complexity correction explains it: PNG codelengths for SVHN run roughly 1 bpd below CIFAR-10, and subtracting $L(x)$ restores separation. So the raw likelihood is close to a measure of image complexity plus a model term, and the complexity term dominates.

**What this makes visible.** Both diagnostics are unavailable on real data without an external reference: the first needs a privileged chart, the second needs a compressor standing in for the unknown $p_{\text{data}}$. Neither is a calibration measurement — they are sanity checks that the number is not what it names. That is why Section 8's control arm is the load-bearing part: only with an analytic ground-truth density can $\Delta$ be attributed to the model rather than to the estimator.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*