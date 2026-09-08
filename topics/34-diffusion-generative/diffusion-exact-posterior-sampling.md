---
id: 34-diffusion-generative/diffusion-exact-posterior-sampling
title: "Inverse Problem Solving With Exact Posterior Sampling"
topic: 34-diffusion-generative
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Inverse Problem Solving With Exact Posterior Sampling

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-exact-posterior-sampling` · **Status:** solved-but-impractical

## 1. Problem Statement

Given a pretrained diffusion model that samples from a prior $p(x)$ over signals $x \in \mathbb{R}^d$, and a measurement $y = \mathcal{A}(x) + n$ with known forward operator $\mathcal{A}$ and known noise law, draw samples from the **true posterior** $p(x \mid y)$ — not from a heuristic that merely produces plausible reconstructions consistent with $y$.

Three variants, with different difficulty:

- **Method.** Build an algorithm whose output law converges to $p(x\mid y)$ with a controllable error, at a cost comparable to unconditional sampling (typically $50$–$1000$ network function evaluations, NFEs).
- **Measurement.** Certify that a given sampler's output law is close to $p(x\mid y)$ on a real problem, where $p(x\mid y)$ is not available in closed form. This is currently the weakest link.
- **Theory.** Characterise which $(\text{prior}, \mathcal{A}, \sigma_y)$ triples admit polynomial-time posterior sampling given oracle access to the prior's score.

Solving it means: an algorithm with a bound $\mathrm{TV}(\hat{p}(\cdot\mid y),\, p(\cdot\mid y)) \le \epsilon$ at cost polynomial in $d$, $1/\epsilon$, and score error — or a proof that no such algorithm exists for the class in question. The status is **solved-but-impractical**: asymptotically exact algorithms exist; their cost is not bounded, and a hardness result says it cannot be in general.

## 2. Formal Setting

**Prior via diffusion.** Forward SDE $\mathrm{d}x_t = f(t)x_t\,\mathrm{d}t + g(t)\,\mathrm{d}w_t$, with the VP convention $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$. The trained network $s_\theta(x_t,t) \approx \nabla_{x_t}\log p_t(x_t)$; measured score error is
$$\mathcal{E}_{\text{score}}^2 = \mathbb{E}_{t\sim\mathcal{U},\,x_t\sim p_t}\big\|s_\theta(x_t,t) - \nabla\log p_t(x_t)\big\|^2,$$
which in practice is *not* measured — the training loss is an upper bound with an unknown additive constant, so $\mathcal{E}_{\text{score}}$ is an assumed, not observed, quantity.

**Likelihood.** $y = \mathcal{A}(x_0) + n$, $n\sim\mathcal{N}(0,\sigma_y^2 I_m)$. Linear case $\mathcal{A}(x)=Ax$, $A\in\mathbb{R}^{m\times d}$.

**The intractable object.** Exact conditional sampling requires the *noisy-measurement likelihood*
$$p_t(y\mid x_t) = \int p(y\mid x_0)\,p(x_0\mid x_t)\,\mathrm{d}x_0,$$
whose conditional $p(x_0\mid x_t)$ is a complicated multimodal law. Every fast method replaces it with a surrogate. DPS uses the Dirac approximation $p(x_0\mid x_t)\approx\delta(\hat{x}_0(x_t))$ with $\hat{x}_0 = \mathbb{E}[x_0\mid x_t]$ (Tweedie); $\Pi$GDM uses a Gaussian $\mathcal{N}(\hat x_0, r_t^2 I)$.

**Cost, as measured.** NFEs $=$ number of forward passes of $s_\theta$ per returned sample. Sequential Monte Carlo (SMC) methods with $N$ particles and $T$ steps cost $NT$ NFEs per sample (or per batch, depending on resampling scheme) — report which.

**Accuracy, as measured.** On problems with closed-form posteriors (Gaussian mixture priors, linear $\mathcal{A}$): sliced Wasserstein-2 between $n$ sampler draws and $n$ exact draws. On images: no ground-truth posterior exists, so PSNR/LPIPS/FID are used — none of which is a divergence to $p(x\mid y)$.

**Assumptions known to be violated.** (i) $s_\theta$ equals the true score — false; error is unbounded near $t\to 0$. (ii) $\mathcal{A}$ and $\sigma_y$ known exactly — false in real imaging (blind blur, calibration error). (iii) The image prior is the model's prior — the target posterior is the *model's* posterior, not nature's; the two are conflated in every image benchmark. (iv) Latent-space methods assume the encoder is invertible — it is not.

## 3. State of the Art

**Theory SOTA (established).** Gupta, Jalal, Parulekar, Price & Xun, *Diffusion Posterior Sampling is Computationally Intractable* (ICML 2024): under standard cryptographic assumptions (one-way functions exist), there are priors for which unconditional sampling is easy and posterior sampling for a simple linear observation is hard for all polynomial-time algorithms. This closes the "just do it faster" route in the worst case. Complementing it, Chen, Chewi, Li, Li, Salim & Zhang (ICLR 2023) show unconditional sampling is polynomial given an $L^2$-accurate score — so the hardness is specific to conditioning.

**Asymptotically exact methods (established convergence, no finite-$N$ bounds).**
- Twisted Diffusion Sampler, Wu, Trippe, Naesseth, Blei & Cunningham (NeurIPS 2023): SMC with twisting functions; consistent as $N\to\infty$.
- MCGDiff, Cardoso, El Idrissi, Le Corff & Moulines (ICLR 2024): SMC for *linear* inverse problems, proven to target the exact posterior in the particle limit.
- FPS, Dou & Song (ICLR 2024): filtering formulation, also asymptotically exact for linear-Gaussian observations.
- Split Gibbs / PnP-DM, Sun, Wu, Chen, Feng & Bouman (IEEE TCI 2024) and Xu & Chi (NeurIPS 2024): Markov-chain schemes with stationarity or robustness guarantees under regularity conditions.

**Fast heuristics (benchmark numbers only, no posterior-fidelity claim).** DDRM (Kawar et al., NeurIPS 2022), DPS (Chung et al., ICLR 2023), $\Pi$GDM (Song et al., ICLR 2023), DDNM (Wang et al., ICLR 2023), RED-diff (Mardani et al., ICLR 2024). All report PSNR/FID on FFHQ/ImageNet $256\times256$ at $100$–$1000$ NFEs. **Claimed but unablated:** that these produce posterior *samples*. RED-diff is explicitly a variational mode-seeker and does not claim it; DPS's approximation error is bounded only by a Jensen gap that is not evaluated on images.

## 4. What Is Known

- **Hardness is real, not folklore.** Gupta et al. (2024) construct the hard instance; the separation is between unconditional (poly) and conditional (super-poly under OWF) sampling.
- **SMC methods beat gradient guidance on problems with known answers.** On Gaussian-mixture priors with linear observations in $d$ up to $\sim 800$, MCGDiff and twisted SMC report sliced-Wasserstein errors an order of magnitude or more below DPS/DDRM, which show visible mode-weight bias at $d=8$ already (Cardoso et al., ICLR 2024).
- **Cost of exactness.** Reported SMC configurations use $N \approx 64$–$256$ particles over $T \approx 300$–$1000$ steps, i.e. $\sim 2\times10^4$–$2.5\times10^5$ NFEs per posterior sample, versus $10^3$ for DPS — a $20$–$250\times$ gap at $256\times256$ resolution.
- **Image benchmarks do not separate the methods.** On FFHQ $256\times256$ inpainting and $4\times$ super-resolution, DPS-family PSNR sits in the $\sim 24$–$28$ dB band; exact SMC samplers do not dominate on PSNR, because PSNR rewards the posterior *mean*, and a correct posterior sample is by construction noisier than the mean.
- **Guidance strength is a free parameter.** DPS's step size $\zeta_t$ is tuned per task; changing it moves reconstructions between "prior-like" and "data-fit-like" with no principled setting — evidence the sampler is not targeting a fixed distribution.

## 5. What Is Not Known

- **Theoretically open.** Finite-$N$, finite-$T$ TV bounds for twisted SMC / MCGDiff under $L^2$ score error. Which structural assumptions (log-concavity of the prior? bounded $\chi^2$ between prior and posterior? small $m$?) buy polynomial-time exact posterior sampling. Whether the Gupta et al. hardness survives for *natural-image-like* priors or is confined to cryptographic constructions.
- **Empirically open.** Whether any published fast method is within a stated TV of the posterior at image scale. The experiment is runnable: build a high-dimensional prior with a tractable posterior (mixture-of-Gaussians fit to image patches, or a diffusion model trained on a synthetic prior) and measure. Nobody has run it at $d \gtrsim 10^4$.
- **Methodologically blocked.** Posterior fidelity on real images. There is no ground-truth $p(x\mid y)$ and no consistent estimator of a divergence to it from samples in $10^5$ dimensions. Proposed proxies — posterior-predictive calibration, simulation-based calibration ranks, coverage of credible intervals — are necessary but not sufficient: a sampler can be perfectly calibrated on every 1-D projection and still be wrong jointly.

## 6. Why It Is Hard

Two obstructions, both specific.

**Non-identifiability of the target under evaluation.** On the benchmarks that decide publication, the ground truth is a single image $x^\star$, and the score is a distortion metric. That metric is minimised by $\mathbb{E}[x\mid y]$, which is *not* a posterior sample. So the evaluation systematically rewards the wrong distribution, and no amount of benchmark improvement is evidence of posterior correctness. This is the "evaluation does not measure what it names" failure in a clean form.

**Weight degeneracy, quantified.** All exact methods are importance-sampling-based; their cost scales with $\chi^2\!\left(p(x\mid y)\,\|\,q\right)$ for the proposal $q$. As $\sigma_y \to 0$ or $m$ grows, this diverges exponentially in $m$ — Section 10 gives $\mathrm{ESS}/N \approx 3\times10^{-19}$ for $m=10$, $\sigma_y=0.01$ under the untwisted proposal. Twisting reduces but does not remove the exponent, and the hardness result says no proposal can remove it for all priors.

## 7. Current Research (as of 2026)

- **Finite-particle analysis of twisted SMC** — Columbia (Cunningham, Blei), Institut Polytechnique de Paris (Moulines, Le Corff). Goal: replace "consistent as $N\to\infty$" with a bound in $N$, $T$, $\mathcal{E}_{\text{score}}$. *(frontier — verify)*
- **Split-Gibbs / plug-and-play Monte Carlo at scale** — Caltech (Bouman), CMU (Chi). Trades exactness for a stationary distribution that is provably close under regularity.
- **Amortised conditional models** — train a conditional diffusion on $(y,x)$ pairs; exact by construction for the training operator, useless for a new $\mathcal{A}$. The open question is how far operator-conditioned amortisation generalises.
- **Calibration-based auditing** — importing simulation-based calibration from the SBI community as the missing measurement. Currently the most promising route around the methodological block. *(frontier — verify)*
- **Latent-space posterior sampling** for Stable-Diffusion-class priors, where the encoder makes the target posterior itself ill-defined. Actively published, weakly justified.

## 8. Concrete Next Experiment

**Question.** How large is the posterior error of DPS, $\Pi$GDM and DDNM at image-relevant dimension, in a setting where the exact posterior is computable?

**Setup.** Prior: a $K=100$-component Gaussian mixture in $d=4096$ ($64\times64$ pixels), fit to CelebA downsamples — for a GMM prior with linear $A$ and Gaussian noise, the posterior is exactly a $K$-component GMM with closed-form weights and covariances. Train a diffusion model on samples from this GMM to $\mathcal{E}_{\text{score}}$ measured against the *analytic* mixture score (this is the point of a GMM prior: the true score is known, so score error is observed, not assumed). Tasks: inpainting ($m=2048$) and $4\times$ super-resolution, at $\sigma_y \in \{0.05, 0.01\}$.

**Arms.** (1) DPS, 1000 NFEs, $\zeta$ tuned. (2) $\Pi$GDM, 100 NFEs. (3) DDNM, 100 NFEs. (4) **Control arm:** MCGDiff with $N=1024$, $T=1000$ ($10^6$ NFEs/sample). (5) **Reference arm:** $10^4$ exact analytic posterior draws.

**Deciding number.** Sliced Wasserstein-2 (1000 random projections, $10^4$ samples per arm) to the analytic posterior, reported as a ratio $R = \mathrm{SW}_2(\text{method})/\mathrm{SW}_2(\text{control})$, with the sampling-noise floor established by splitting the reference arm in half. If any fast method achieves $R < 2$ at $\sigma_y = 0.01$, the practical gap is smaller than the theory suggests and cheap approximate posterior sampling is defensible. If $R > 10$ — the outcome the $d=8$ mixture results predict — then every image-domain "posterior sampling" claim built on these methods is unsupported, and the field should stop using the word.

**Cost.** Roughly $10^6$ NFEs $\times\,10^4$ control samples is prohibitive; run the control at $10^3$ samples and widen the confidence interval accordingly. Estimated: $\sim 2{,}000$ A100-hours.

## 9. Key References

- **[Foundational]** Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR, 2021. — arXiv:2011.13456
- **[Foundational]** B. Kawar, M. Elad, S. Ermon, J. Song. *Denoising Diffusion Restoration Models.* NeurIPS, 2022. — arXiv:2201.11793
- **[SOTA-heuristic]** H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye. *Diffusion Posterior Sampling for General Noisy Inverse Problems.* ICLR, 2023. — arXiv:2209.14687
- **[SOTA-heuristic]** J. Song, A. Vahdat, M. Mardani, J. Kautz. *Pseudoinverse-Guided Diffusion Models for Inverse Problems.* ICLR, 2023.
- **[SOTA-exact]** L. Wu, B. L. Trippe, C. A. Naesseth, D. M. Blei, J. P. Cunningham. *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models.* NeurIPS, 2023. — arXiv:2306.17775
- **[SOTA-exact]** G. Cardoso, Y. J. El Idrissi, S. Le Corff, E. Moulines. *Monte Carlo Guided Diffusion for Bayesian Linear Inverse Problems.* ICLR, 2024. — arXiv:2308.07983
- **[SOTA-exact]** Z. Dou, Y. Song. *Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective.* ICLR, 2024.
- **[Hardness]** S. Gupta, A. Jalal, A. Parulekar, E. Price, Z. Xun. *Diffusion Posterior Sampling is Computationally Intractable.* ICML, 2024. — arXiv:2402.12727
- **[Theory]** S. Chen, S. Chewi, J. Li, Y. Li, A. Salim, A. R. Zhang. *Sampling is as Easy as Learning the Score: Theory for Diffusion Models with Minimal Data Assumptions.* ICLR, 2023. — arXiv:2209.11215
- **[Theory]** X. Xu, Y. Chi. *Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction.* NeurIPS, 2024. — arXiv:2403.17042
- **[Theory]** Y. Sun, Z. Wu, Y. Chen, B. T. Feng, K. L. Bouman. *Provable Probabilistic Imaging Using Score-Based Generative Priors.* IEEE Transactions on Computational Imaging, 2024. — arXiv:2310.10835
- **[Survey]** G. Daras, H. Chung, C.-H. Lai, Y. Mitsufuji, J. C. Ye, P. Milanfar, A. G. Dimakis, M. Delbracio. *A Survey on Diffusion Models for Inverse Problems.* 2024. — arXiv:2410.00083

## 10. Worked Example

**Setup.** Prior on a single coordinate $x_1 \sim \mathcal{N}(0,1)$; measurement $y_1 = x_1 + n$, $n\sim\mathcal{N}(0,\sigma_y^2)$, $\sigma_y = 0.01$, observed $y_1 = 0$. Use the prior as the SMC proposal (the untwisted case, which is what guidance-free importance sampling reduces to). Weight $w(x) = \mathcal{N}(y_1; x_1, \sigma_y^2)$.

Effective sample size fraction:
$$\frac{\mathrm{ESS}}{N} = \frac{\mathbb{E}[w]^2}{\mathbb{E}[w^2]} = \frac{\mathcal{N}(y;0,1+\sigma_y^2)^2 \cdot 2\sigma_y\sqrt{\pi}}{\mathcal{N}(y;0,1+\sigma_y^2/2)}.$$
At $y=0$: $\mathcal{N}(0;0,1)=0.3989$, so $\mathrm{ESS}/N = 0.3989 \times 2(0.01)(1.7725) = 0.0141$.

**Scaling to $m$ measured coordinates.** With $m$ independent rows the weights multiply, so $\mathrm{ESS}/N = (0.0141)^m$:

| $m$ | $\mathrm{ESS}/N$ | particles for $\mathrm{ESS}=100$ |
|---|---|---|
| 1 | $1.4\times10^{-2}$ | $7\times10^{3}$ |
| 3 | $2.8\times10^{-6}$ | $3.6\times10^{7}$ |
| 10 | $3.1\times10^{-19}$ | $3.2\times10^{20}$ |

**What this makes visible.** At $m=10$ — ten observed pixels — an untwisted particle sampler needs $\sim 3\times10^{20}$ particles, each carrying $T\approx 10^3$ network evaluations. Real inpainting has $m \approx 10^4$. Twisting (using $\hat{x}_0(x_t)$ to build the proposal) is what makes SMC work at all: it cuts the per-coordinate degeneracy factor, but the failure mode is the *same exponential in $m$* with a smaller base, which is why reported SMC results stop at a few hundred dimensions.

The complementary half of the obstruction: DPS on this instance returns essentially $\hat{x}_1 = \mathbb{E}[x_1\mid y] \approx 0$ with variance far below the true posterior variance $\sigma_y^2/(1+\sigma_y^2) \approx 10^{-4}$ — but at $\sigma_y=0.01$ the posterior is so tight that DPS's PSNR against the true $x^\star$ is indistinguishable from the exact sampler's. The benchmark cannot see the difference the theory says is there. That is the block.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*