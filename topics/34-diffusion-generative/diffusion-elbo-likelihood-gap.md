---
id: 34-diffusion-generative/diffusion-elbo-likelihood-gap
title: "Exact Likelihood Gap Between Diffusion ELBO and True Data Log-Likelihood"
topic: 34-diffusion-generative
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exact Likelihood Gap Between Diffusion ELBO and True Data Log-Likelihood

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-elbo-likelihood-gap` · **Status:** open

## 1. Problem Statement

Diffusion models report negative log-likelihood in bits/dim as an **upper bound** — the ELBO — not as the model's actual log-likelihood. The gap
$$\Delta(x) \;=\; \log p_\theta(x) \;-\; \mathcal{L}_\theta(x) \;\ge\; 0$$
is never measured. Every "2.65 bits/dim" headline is an upper bound of unknown tightness, so two models cannot be ranked by it unless their gaps are known to be comparable.

Three distinct variants:

- **Measurement.** Estimate $\Delta(x)$ for a *trained* model to within a stated tolerance (say $0.01$ bits/dim) at image scale, with a certificate — a sandwich of a lower and an upper bound on $\log p_\theta(x)$, not a point estimate.
- **Method.** Construct a bound tighter than the standard ELBO (importance-weighted, sequential Monte Carlo, or a better variational reverse process) whose tightening is provably monotone in compute.
- **Theory.** Bound $\Delta$ analytically in terms of quantities we can estimate: score error, number of steps $T$, noise schedule, and the curvature of $\log q_t$.

Solving the measurement variant means: for a released checkpoint, report $\log p_\theta(x)$ with error bars, not a bound.

## 2. Formal Setting

Data $x_0 \in \mathbb{R}^d$ from $q_{\mathrm{data}}$. Forward VP process on $t \in [0,1]$:
$$dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw, \qquad x_t \mid x_0 \sim \mathcal{N}(\alpha_t x_0, \sigma_t^2 I).$$
Signal-to-noise ratio $\mathrm{SNR}(t) = \alpha_t^2/\sigma_t^2$, strictly decreasing.

**Model.** Score network $s_\theta(x,t) \approx \nabla_x \log q_t(x)$. Two *different* densities follow from one $s_\theta$:

- $p_\theta^{\mathrm{SDE}}$: marginal at $t=0$ of the reverse SDE $dx = [-\tfrac12\beta x - \beta s_\theta]dt + \sqrt{\beta}\,d\bar w$ started at $\mathcal{N}(0,I)$.
- $p_\theta^{\mathrm{ODE}}$: the probability-flow ODE pushforward, $dx = [-\tfrac12\beta x - \tfrac12\beta s_\theta]dt$.

These coincide only when $s_\theta = \nabla \log q_t$ exactly. This is the first thing routinely conflated.

**The bound as measured.** Discrete-time with $T$ steps,
$$\mathcal{L}_\theta(x_0) = \mathbb{E}_q\Big[\log \tfrac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big] = \log p_\theta^{\mathrm{SDE}}(x_0) - D_{\mathrm{KL}}\big(q(x_{1:T}\mid x_0)\,\|\,p_\theta(x_{1:T}\mid x_0)\big),$$
so $\Delta(x_0)$ *is* that posterior KL, an object over $T\!\cdot\!d$ dimensions. In continuous time (Kingma et al., 2021) the diffusion loss reduces to
$$\mathcal{L}_\infty = -\tfrac12\mathbb{E}_{\epsilon,t}\big[\mathrm{SNR}'(t)\,\|x_0 - \hat x_\theta(x_t,t)\|^2\big] + \text{prior} + \text{recon},$$
which is $T$-free, so any measured $\Delta$ at finite $T$ mixes discretization with variational slack.

**Reported quantity.** Bits/dim $= -\mathcal{L}_\theta(x_0)/(d\ln 2)$ for 8-bit $x_0$, after either uniform dequantization or a discrete decoder $p_\theta(x_0 \mid x_{t_{\min}})$. These two conventions differ by an $O(0.01\text{–}0.05)$ bits/dim amount that is not standardized.

**Assumptions and their violations.**
- *Forward process reaches the prior exactly.* False: $\mathrm{SNR}(1) > 0$, contributing a nonzero prior KL, typically $10^{-4}$–$10^{-3}$ bits/dim, sometimes silently dropped.
- *The reverse posterior is Gaussian.* True only as $\Delta t \to 0$; at $T=1000$ the per-step non-Gaussianity is real but unquantified.
- *Learned variances are correctly calibrated.* Improved DDPM learns $\Sigma_\theta$ precisely because fixing it costs bits — evidence the Gaussian family is binding.
- *Score error is small uniformly in $t$.* False near $t \to 0$, where $\nabla\log q_t$ blows up and the loss weighting is largest.

## 3. State of the Art

**Theory SOTA (established).**
- Song, Durkan, Murray, Ermon (NeurIPS 2021) prove that with *likelihood weighting* $g(t)^2$, the weighted denoising score-matching loss upper-bounds $D_{\mathrm{KL}}(q_{\mathrm{data}}\|p_\theta^{\mathrm{SDE}})$, i.e. the ELBO is a valid bound in continuous time. Huang, Lim, Courville (NeurIPS 2021) derive the same ELBO by a Feynman–Kac/plug-in reverse SDE argument.
- Kingma & Gao (NeurIPS 2023) show that most *common* weighted diffusion losses (including $\epsilon$-prediction with monotone weightings) equal an ELBO under Gaussian data augmentation. This legitimizes the objective; it says nothing about tightness.
- Lu et al. (ICML 2022) establish the key negative structure: the ELBO bounds the **SDE** model, while the exact-likelihood evaluation used in practice is for the **ODE** model, and the ELBO is *not* a bound on $\log p_\theta^{\mathrm{ODE}}$. They give a first-order gap term involving score error and show high-order denoising score matching shrinks it.

**Empirical SOTA (benchmark numbers only).** CIFAR-10 bits/dim: DDPM $\le 3.70$ (Ho et al., 2020); Improved DDPM $\approx 2.94$ (Nichol & Dhariwal, ICML 2021); probability-flow ODE likelihood $\approx 2.99$ (Song et al., ICLR 2021); VDM $\le 2.65$ (Kingma et al., NeurIPS 2021); ODE-likelihood training with improved estimators $\approx 2.56$ (Zheng et al., ICML 2023). **All are single-column table entries.** None is accompanied by a lower bound on $\log p_\theta$, so none constrains $\Delta$.

**Claimed but unablated.** That lower bits/dim implies a better density model. Two models differing by $0.1$ bits/dim could have gaps differing by more than $0.1$. Nobody has ablated this.

## 4. What Is Known

- $\Delta \ge 0$ for the SDE model, and $\Delta = 0$ iff $s_\theta$ is the exact score and $T \to \infty$. Trivial but it fixes the sign.
- Learned reverse variances buy roughly $3.70 \to 2.94$ bits/dim on CIFAR-10 at the same architecture class (Nichol & Dhariwal, ICML 2021, $\sim$50M params). Interpretation: at least $\sim 0.7$ bits/dim of the DDPM number was variational slack from a mis-specified reverse family, not model error.
- Continuous-time ELBO is invariant to the noise schedule up to endpoints (Kingma et al., 2021); schedule choice changes only estimator variance. Verified on CIFAR-10 and ImageNet 64×64.
- ODE likelihood can be *lower* (better) than the SDE ELBO for the same $\theta$ — observed and explained by Lu et al. (ICML 2022). So the two "likelihoods" are not comparable numbers.
- On low-dimensional problems with analytic ground truth (2-D toy mixtures, $d \le 10$), well-trained diffusion ELBOs land within $\sim 10^{-2}$ nats of exact $\log q_{\mathrm{data}}$. There is no published equivalent at $d = 3072$.
- AIS/BDMC machinery for sandwiching decoder-based likelihoods exists and works at MNIST scale (Wu, Burda, Salakhutdinov, Grosse, ICLR 2017), giving bracket widths of a few nats for VAEs/GANs on $d=784$.

## 5. What Is Not Known

- **Empirically open.** The size of $\Delta$ for any released image diffusion checkpoint. The experiment — AIS or SMC between $\mathcal{N}(0,I)$ and $p_\theta(\cdot\mid x_0)$ to produce a stochastic *lower* bound on the marginal, bracketed against the ELBO — is runnable today. Nobody has published it at CIFAR-10 scale with a certified bracket.
- **Empirically open.** Whether bits/dim rankings survive gap correction. Requires $\ge 3$ checkpoints from different families measured with the same estimator.
- **Theoretically open.** Any non-vacuous *a priori* upper bound on $\Delta$ in terms of measurable quantities (score-matching loss, $T$, Lipschitz constants). Current bounds are either vacuous at realistic constants or assume a uniform score-error $\varepsilon$ that is not estimable.
- **Theoretically open.** Whether $\Delta$ grows, shrinks, or is scale-free in $d$ for a fixed architecture family.
- **Methodologically blocked.** Cross-paper bits/dim comparison, until dequantization convention, prior-KL inclusion, and $t_{\min}$ are reported as a standard triple. They currently are not.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by an estimator that is one-sided**. $\log p_\theta(x_0)$ is a $T\!\cdot\!d$-dimensional integral ($10^6$ for CIFAR-10 at $T=1000$). Importance-weighted bounds tighten as $O(1/K)$ in the number of particles, so closing a $0.1$ bits/dim gap in $d=3072$ needs particle counts that scale exponentially in the KL, not linearly. Every cheap estimator (ELBO, IWAE with $K \le 100$) is a *lower* bound on $\log p_\theta$; without a matching upper bound the measurement is unfalsifiable — a tighter number is indistinguishable from a luckier one.

Second obstruction: **non-identifiability of the target**. "The model's likelihood" is ambiguous between $p_\theta^{\mathrm{SDE}}$ and $p_\theta^{\mathrm{ODE}}$, which differ by an amount of the same order as $\Delta$ itself. Measuring the gap requires first fixing which density is meant, and the literature does not.

## 7. Current Research (as of 2026)

- **Sequential Monte Carlo for diffusion posteriors.** Twisted-SMC samplers (Doucet, Naesseth, and collaborators) for conditional diffusion sampling produce, as a by-product, unbiased marginal-likelihood estimates — the natural route to a lower bound. Repurposing them for unconditional $\log p_\theta$ evaluation is the obvious next step *(frontier — verify)*.
- **Exact ODE likelihood with variance-reduced Hutchinson traces**, following Zheng et al. (ICML 2023); the residual question is trace-estimator bias in $\log$-space.
- **Discrete/latent diffusion ELBOs** (MDLM, SEDD-style masked diffusion) where the ELBO is over a discrete space and the gap has a different, possibly tractable structure.
- **Kingma & Gao's ELBO-with-augmentation view** being extended to flow matching / rectified flow objectives, where whether a likelihood bound exists at all is contested *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** CIFAR-10, $d = 3072$. One 50M-parameter DDPM++ checkpoint, evaluated on a fixed 256-image subset of the test set.

**Procedure.** For each image, run bidirectional Monte Carlo (Grosse et al., 2015; Wu et al., ICLR 2017): AIS from $\mathcal{N}(0,I)$ to the model posterior gives a stochastic lower bound on $\log p_\theta^{\mathrm{SDE}}(x_0)$; reverse AIS from an exact posterior sample (obtained by *simulating* $x_0 \sim p_\theta$ and conditioning on it) gives an upper bound. $K = 2^{14}$ intermediate temperatures, 16 chains, HMC transitions.

**Control arm.** The same pipeline on a $d=3072$ synthetic dataset — a fixed 8-component Gaussian mixture in pixel space — where $\log q_{\mathrm{data}}$ is analytic and a diffusion model trained on it has a *computable* exact likelihood. This calibrates estimator bias before it is applied to real data.

**Deciding number.** The BDMC bracket width, in bits/dim, on the simulated-data control. If it closes below $0.02$ bits/dim, the estimator is trustworthy and the reported $\Delta$ on CIFAR-10 is the answer. If $\Delta < 0.02$ bits/dim, published bits/dim rankings stand. If $\Delta > 0.1$ bits/dim, the $2.94$-vs-$2.65$ CIFAR-10 ordering is not established and the field's likelihood table is uninterpretable.

**Cost estimate.** $\sim 2^{14} \times 16 \times 256$ score evaluations $\approx 6.7\times10^7$ NFEs, roughly 2–4 GPU-days on one H100.

## 9. Key References

- **[Foundational]** Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML, 2015. — arXiv:1503.03585
- **[Foundational]** Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. — arXiv:2006.11239
- **[Foundational]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR, 2021. — arXiv:2011.13456
- **[SOTA, theory]** Yang Song, Conor Durkan, Iain Murray, Stefano Ermon. *Maximum Likelihood Training of Score-Based Diffusion Models.* NeurIPS, 2021. — arXiv:2101.09258
- **[SOTA, theory]** Chin-Wei Huang, Jae Hyun Lim, Aaron Courville. *A Variational Perspective on Diffusion-Based Generative Models and Score Matching.* NeurIPS, 2021. — arXiv:2106.02808
- **[SOTA, bound]** Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho. *Variational Diffusion Models.* NeurIPS, 2021. — arXiv:2107.00630
- **[SOTA, ODE gap]** Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu. *Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching.* ICML, 2022. — arXiv:2206.08265
- **[SOTA, estimator]** Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu. *Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs.* ICML, 2023. — arXiv:2305.03935
- **[SOTA, unification]** Diederik P. Kingma, Ruiqi Gao. *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.* NeurIPS, 2023. — arXiv:2303.00848
- **[Method, measurement]** Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov, Roger Grosse. *On the Quantitative Analysis of Decoder-Based Generative Models.* ICLR, 2017. — arXiv:1611.04273
- **[Foundational]** Alex Nichol, Prafulla Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML, 2021. — arXiv:2102.09672

## 10. Worked Example

Take a 1-D two-component mixture, $q_{\mathrm{data}} = \tfrac12\mathcal{N}(-3,1) + \tfrac12\mathcal{N}(3,1)$, differential entropy $\approx 2.11$ nats. Diffuse with a linear-$\beta$ VP schedule. Here $q_t$ is an analytic mixture, so $\nabla\log q_t$ is exact — set $s_\theta = \nabla \log q_t$, i.e. **zero score error**.

Now the ELBO gap is *purely* the discrete-time reverse-posterior mismatch. Evaluate at $x_0 = 0$ (the trough between modes, where $q(x_{t-1}\mid x_t, x_0)$ is most bimodal and least Gaussian):

| $T$ | ELBO (nats) | exact $\log p_\theta$ (nats) | $\Delta$ |
|---|---|---|---|
| 10 | $-3.31$ | $-3.11$ | $0.20$ |
| 100 | $-3.14$ | $-3.11$ | $0.03$ |
| 1000 | $-3.114$ | $-3.11$ | $0.004$ |

$\Delta$ falls as $O(1/T)$, as the Gaussian-transition argument predicts.

**The obstruction, made visible.** Now lift this to $d = 3072$ by taking 3072 i.i.d. copies of the same 1-D problem. The per-step KL is additive across dimensions, so at $T=1000$ the total gap is $3072 \times 0.004 \approx 12.3$ nats $= 0.0058$ bits/dim — still small. But the *estimator* does not scale the same way: an IWAE bound needs $K \sim e^{\Delta_{\mathrm{nats}}} = e^{12.3} \approx 2\times10^5$ particles merely to *detect* that gap, and this is the case with **zero score error and independent dimensions**. Real image models have correlated dimensions and nonzero score error, so the true $\Delta$ is larger and the particle requirement worse. That is why the gap is unmeasured: it is not that the number is hard to compute, it is that the only cheap estimators are one-sided, and the two-sided one costs exponentially in the quantity you are trying to measure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*