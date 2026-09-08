---
id: 34-diffusion-generative/diffusion-manifold-adaptivity
title: "Provable Manifold Adaptivity of Diffusion Sampling"
topic: 34-diffusion-generative
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Manifold Adaptivity of Diffusion Sampling

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-manifold-adaptivity` · **Status:** partially-solved

## 1. Problem Statement

Natural images sit in $\mathbb{R}^D$ with $D \approx 10^5$ but are believed to concentrate near a set of intrinsic dimension $d \approx 10^1$–$10^2$. **Manifold adaptivity** is the claim that a diffusion model, trained and sampled without ever being told $d$, pays a cost governed by $d$ rather than $D$.

Three variants, of very different difficulty:

- **Theory variant.** Prove that the number of denoiser evaluations (NFE) needed for $\mathrm{KL}(p_{\text{data}} \| p_{\text{sampler}}) \le \varepsilon^2$ scales as $\tilde{O}(\mathrm{poly}(d)\,\mathrm{polylog}(D))$, and that the sample complexity of learning the score scales as $n^{-\Theta(1/d)}$, with no algorithmic knowledge of $d$.
- **Measurement variant.** Given a trained model and a dataset, *measure* $d$ and *measure* whether the realised NFE–error curve depends on $D$ at fixed $d$. Both quantities are estimator-dependent and neither has a settled protocol.
- **Method variant.** Build a sampler whose step schedule provably exploits $d$ (e.g. spends steps only where the score's normal-direction blow-up is active) and beats a $d$-agnostic schedule at matched NFE.

Solved means: a bound linear (or better) in $d$ and logarithmic in $D$ under assumptions that hold for a checkable class of data, *plus* an experiment where holding $d$ fixed and raising $D$ by $16\times$ leaves the NFE-to-fixed-error unchanged.

## 2. Formal Setting

Data law $p_0$ supported on (or within $\eta$ of) a compact $C^2$ submanifold $\mathcal{M} \subset \mathbb{R}^D$, $\dim \mathcal{M} = d$, reach $\tau > 0$. The reach is the largest $r$ such that every point within $r$ of $\mathcal{M}$ has a unique nearest point on it; it is the quantity that controls curvature-induced error and is **not measurable from finite samples without extra assumptions**.

Forward OU process, $\alpha_t = e^{-t}$, $\sigma_t^2 = 1 - e^{-2t}$:
$$p_t(x) = \int \mathcal{N}(x;\alpha_t y, \sigma_t^2 I)\, p_0(dy), \qquad s_t(x) = \nabla \log p_t(x).$$

Key structural fact: near $\mathcal{M}$, for small $t$, the score decomposes into a tangential part of size $O(1)$ and a normal part
$$s_t(x) \approx -\frac{x_\perp}{\sigma_t^2}, \qquad \mathbb{E}\|s_t\|^2 \approx \frac{D-d}{\sigma_t^2} + O(1),$$
so $\|s_t\|$ blows up in exactly $D-d$ directions. This is the signal every empirical intrinsic-dimension estimator exploits.

**Quantities as measured.**

- $\hat{d}$: from the singular spectrum of $\{s_t(x_i + \xi_j)\}_j$ at small $t$ — count singular values above a threshold, $\hat{d} = D - \\#\{\text{large}\}$ (Stanczuk et al.). Threshold choice is a free parameter; report the estimate as a function of it.
- $\varepsilon_{\text{score}}^2 = \int_{\delta}^{T} w(t)\, \mathbb{E}_{p_t}\|\hat{s}_\theta(x,t) - s_t(x)\|^2\,dt$. Only measurable in closed form on synthetic $p_0$; on real data one measures the **held-out denoising loss gap** to the empirical-optimal score, which is an upper bound contaminated by the irreducible term.
- NFE: count of $\hat{s}_\theta$ calls, the only compute unit that is comparable across solvers.
- Sampling error: $W_2$ estimated by entropic OT (Sinkhorn, $\epsilon_{\text{reg}}$ reported) on $n \ge 50$k samples; TV is **not** estimable in $D \gg 1$ and should never be reported empirically.

**Assumptions and their status.**

| Assumption | Status in practice |
|---|---|
| Exact manifold support, $\eta = 0$ | Violated — JPEG/quantisation noise gives full-dimensional support at scale $\sim 1/255$ |
| Bounded reach $\tau$, uniform over $\mathcal{M}$ | Unverified; image data plausibly has near-zero reach at corners/occlusions |
| Single global $d$ | Violated — Brown et al. (2023) show per-class $d$ varies within one dataset |
| $\varepsilon_{\text{score}}$ small **uniformly in $t$** | Violated — score error is worst exactly at small $t$, where the target diverges |
| Density smooth in tangent directions (Besov/Hölder $s$) | Untestable |

## 3. State of the Art

**Theory (established).**
- De Bortoli, *Convergence of denoising diffusion models under the manifold hypothesis* (TMLR 2022): first Wasserstein bounds with no density assumption; constants depend on the manifold, but the dependence is not tight and includes terms exponential in problem constants.
- Chen, Huang, Zhao, Wang (ICML 2023) and Oko, Akiyama, Suzuki (ICML 2023): for data on a low-dimensional **linear subspace** / Besov densities, score-network approximation and estimation rates of the form $n^{-\Theta(s/(2s+d))}$ — minimax-optimal in $d$, not $D$. Establishes *statistical* adaptivity.
- Benton, De Bortoli, Doucet, Deligiannidis (ICLR 2024): $\tilde{O}(D/\varepsilon^2)$ steps in KL under only finite second moment — the **ambient-dimension baseline** any adaptivity claim must beat.
- Li & Yan (NeurIPS 2024): DDPM sampler iteration complexity controlled by intrinsic dimension $k$ rather than $D$, with no knowledge of $k$ — the cleanest existing *algorithmic* adaptivity result.
- Potaptchik, Azangulov, Deligiannidis (2024/25): step count linear in intrinsic dimension with only polylogarithmic ambient dependence, under manifold-support assumptions.
- Azangulov, Deligiannidis, Rousseau (2024): high-dimensional convergence under the manifold hypothesis joining score estimation and sampling error.

**Claimed but unablated.** That real image diffusion models *realise* these rates. No published experiment varies $D$ at fixed $d$ and shows flat NFE. Reported FID-vs-NFE curves (DPM-Solver++, EDM, consistency distillation) are benchmark numbers on fixed $D$; they cannot separate $d$ from $D$ effects at all.

**Measurement SOTA.** Stanczuk, Batzolis, Cristianini, Schönlieb, *Diffusion models encode the intrinsic dimension of data manifolds* (ICML 2024): score-based $\hat{d}$, near-exact on synthetic manifolds. Kamb & Ganguli (2024) and Kadkhodaie et al. (ICLR 2024) give a competing structural story — locality/equivariance and geometry-adaptive harmonic bases — in which the effective inductive bias, not the manifold, sets the rate.

## 4. What Is Known

- **Intrinsic dimension is small and measurable-ish.** Pope et al. (ICLR 2021): MLE estimates $\hat{d} \approx 13$ for MNIST, $\approx 26$–$43$ for ImageNet subsets at $D = 3\cdot 224^2 \approx 1.5\times10^5$. Ratio $D/\hat{d} \gtrsim 3\times10^3$.
- **Score-based $\hat{d}$ works on synthetic data.** Stanczuk et al. report errors of order 1 dimension on spheres/tori with $d \le 100$ embedded in $D \le 1000$; agreement with MLE estimators on MNIST-scale image data is qualitative, not tight.
- **Statistical rates are $d$-driven.** Minimax TV rate $n^{-s/(2s+d)}$ up to logs for Besov-$s$ densities on a $d$-dimensional subspace (Oko et al., ICML 2023), proved at the level of function classes, not at any empirical scale.
- **Ambient-linear step bounds are tight without structure.** $\tilde{O}(D/\varepsilon^2)$ (Benton et al.) is not improvable for general finite-second-moment data.
- **Empirically, NFE does not obviously track $D$.** EDM-class models reach FID $\approx 2$ on CIFAR-10 ($D = 3072$) in 35 NFE and on ImageNet-64 ($D = 12288$) in 79 NFE — a $4\times$ ambient increase against a $2.3\times$ NFE increase. This is suggestive but confounded: $d$, model capacity, and dataset all changed together.

## 5. What Is Not Known

- **Theoretically open.** Whether $\mathrm{poly}(d)\,\mathrm{polylog}(D)$ step bounds survive when the manifold assumption is relaxed to $\eta$-approximate support with $\eta$ at the quantisation scale; whether they survive **non-constant reach** and varying local dimension. Whether the *combined* estimation+sampling bound is $d$-adaptive at rates matching the statistical minimax rate — current results assume an $L^2$-accurate score oracle whose own sample complexity is the hard part.
- **Empirically open.** Nobody has run the $D$-sweep at fixed $d$ at any serious scale. The experiment is cheap (Section 8) and would take a few thousand GPU-hours.
- **Methodologically blocked.** "The intrinsic dimension of ImageNet" is not well defined: MLE, TwoNN, and score-based estimators disagree by factors of 2–3, all are threshold/scale-dependent, and there is no ground truth. Until $\hat{d}$ has a protocol with reported sensitivity, any claim of the form "the model adapts to $d = 43$" is untestable.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth for $d$ on real data.** Every estimator measures the dimension *at a scale* — the noise level $t$, or the $k$ in $k$-NN. Real data has different apparent dimension at different scales (texture at fine scale, object identity at coarse). There is no $d$ to be adaptive to.
2. **Confounded measurement.** You cannot change $D$ on natural images without changing $d$, the dataset statistics, and the architecture's receptive field simultaneously. Upsampling changes $D$ but also adds a smoothness constraint; zero-padding changes $D$ but is trivially handled by any convolutional net.
3. **Non-identifiability of the mechanism.** Kadkhodaie et al. and Kamb & Ganguli show a locality/harmonic-basis inductive bias that produces $D$-independent sample complexity *without* a manifold. Manifold adaptivity and architectural inductive bias predict the same NFE curves; no current experiment distinguishes them. This is the deepest problem: the evaluation does not measure the thing it names.

## 7. Current Research (as of 2026)

- **Oxford/Warwick (Deligiannidis, Potaptchik, Azangulov) and collaborators** — tightening manifold-hypothesis convergence toward linear-in-$d$, polylog-in-$D$; extension to varying local dimension is the announced next step *(frontier — verify)*.
- **Princeton/Yale (Gen Li, Yuling Yan) and CMU (Yuting Wei, Yuxin Chen)** — sharp DDPM/DDIM iteration complexity, adaptivity without knowledge of $d$, and coefficient design.
- **Tokyo (Suzuki group)** — minimax estimation for diffusion under low-dimensional and Besov structure.
- **Cambridge (Schönlieb group) and Vector Institute (Brown, Loaiza-Ganem, Cresswell)** — dimension estimation from score models, and the union-of-manifolds picture with per-class $d$.
- **Stanford (Ganguli)** — locality/equivariance as the alternative explanation of $D$-independence.

## 8. Concrete Next Experiment

**The $D$-sweep at fixed $d$.**

- **Scale.** Synthetic arm: $p_0$ = pushforward of $\mathcal{N}(0, I_d)$, $d = 8$, through a fixed random 3-layer smooth MLP into $\mathbb{R}^D$, followed by a random orthogonal embedding into $D \in \{64, 256, 1024, 4096, 16384\}$. The generator is *the same* across $D$, so $d$, curvature, and reach are held exactly constant — this defeats obstruction (2). Train one architecture family (MLP-DiT, width scaled $\propto \sqrt{D}$, matched parameter-per-dimension) to matched held-out denoising loss. $\sim$5 GPU-days total.
- **Control arm.** Same $D$ sweep with $d = D$ (full-rank Gaussian mixture matched in second moment). Theory says this arm must show $\text{NFE} \propto D$.
- **Deciding number.** Fit $\text{NFE}(D) \propto D^{\alpha}$, where NFE is the smallest DDPM step count reaching entropic-$W_2$ within $5\%$ of the $D=64$ converged value, $n = 10^5$ samples, $\epsilon_{\text{reg}}$ fixed. **Manifold adaptivity predicts $\alpha \le 0.15$; the ambient bound predicts $\alpha \approx 1$.** The control arm validates the estimator by returning $\alpha \approx 1$.
- **Disambiguation of mechanism.** Repeat the low-$d$ arm with the embedding replaced by an *axis-aligned* one (so locality bias can exploit it) versus a *dense random orthogonal* one (locality bias cannot). If $\alpha$ is low only for the axis-aligned embedding, the effect is architectural, not manifold — obstruction (3) resolved in the negative.

## 9. Key References

- **[Foundational]** V. De Bortoli. *Convergence of Denoising Diffusion Models under the Manifold Hypothesis.* TMLR, 2022. — arXiv:2208.05314
- **[Foundational]** S. Chen, S. Chewi, J. Li, Y. Li, A. Salim, A. R. Zhang. *Sampling is as Easy as Learning the Score: Theory for Diffusion Models with Minimal Data Assumptions.* ICLR, 2023. — arXiv:2209.11215
- **[SOTA]** J. Benton, V. De Bortoli, A. Doucet, G. Deligiannidis. *Nearly $d$-Linear Convergence Bounds for Diffusion Models via Stochastic Localization.* ICLR, 2024. — arXiv:2308.03686
- **[SOTA]** K. Oko, S. Akiyama, T. Suzuki. *Diffusion Models are Minimax Optimal Distribution Estimators.* ICML, 2023. — arXiv:2303.01861
- **[SOTA]** M. Chen, K. Huang, T. Zhao, M. Wang. *Score Approximation, Estimation and Distribution Recovery of Diffusion Models on Low-Dimensional Data.* ICML, 2023. — arXiv:2302.07194
- **[SOTA]** G. Li, Y. Yan. *Adapting to Unknown Low-Dimensional Structures in Score-Based Diffusion Models.* NeurIPS, 2024. — arXiv:2405.14861
- **[SOTA]** P. Potaptchik, I. Azangulov, G. Deligiannidis. *Linear Convergence of Diffusion Models Under the Manifold Hypothesis.* 2024. — arXiv:2410.09046
- **[SOTA]** I. Azangulov, G. Deligiannidis, J. Rousseau. *Convergence of Diffusion Models Under the Manifold Hypothesis in High-Dimensions.* 2024. — arXiv:2409.18804
- **[Measurement]** J. Stanczuk, G. Batzolis, T. Cristianini, C.-B. Schönlieb. *Diffusion Models Encode the Intrinsic Dimension of Data Manifolds.* ICML, 2024. — arXiv:2212.12611
- **[Measurement]** P. Pope, C. Zhu, A. Abdelkader, M. Goldblum, T. Goldstein. *The Intrinsic Dimension of Images and Its Impact on Learning.* ICLR, 2021. — arXiv:2104.08894
- **[Alternative mechanism]** Z. Kadkhodaie, F. Guth, E. P. Simoncelli, S. Mallat. *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations.* ICLR, 2024. — arXiv:2310.02557
- **[Survey]** T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS, 2022. — arXiv:2206.00364

## 10. Worked Example

Take $\mathcal{M} = S^{d-1}$ of radius 1, $d = 8$, embedded in $\mathbb{R}^D$. The score at noise level $\sigma$ has squared norm $\approx (D-d)/\sigma^2$ in the normal directions and $O(1)$ tangentially.

Set $D = 1024$, $\sigma = 10^{-2}$. Then
$$\mathbb{E}\|s_\sigma\|^2 \approx \frac{1016}{10^{-4}} = 1.02\times 10^{7}, \qquad \|s_\sigma\| \approx 3.2\times 10^{3}.$$
At $D = 16384$ the same quantity is $\|s_\sigma\| \approx 1.28\times 10^{4}$ — a $4\times$ increase, purely ambient. A Euler–Maruyama step of size $h$ incurs discretisation error scaling with $h\,\|s\|$, so a naive reading says the step count must grow like $\sqrt{D}$.

The adaptivity theorems say otherwise: the blow-up is *isotropic within the normal bundle* and the reverse SDE contracts it exactly, so the $D-d$ normal directions contribute $O(\log D)$, not $O(D)$, to the KL budget. Both readings are consistent with the same measured score norm.

**Where the obstruction becomes visible.** Now do the empirical check that is supposed to decide between them, on real data. Train two EDM models: CIFAR-10 at $32^2$ ($D = 3072$) and bicubic-upsampled CIFAR-10 at $64^2$ ($D = 12288$). Ambient dimension rose $4\times$; intrinsic dimension is unchanged by construction, since upsampling is a deterministic injective map. Suppose the measurement returns NFE $= 35$ and $37$ for matched $W_2$ — $\alpha = 0.04$, apparently strong adaptivity.

It decides nothing. Bicubic upsampling puts all the added variance in high spatial frequencies that a convolutional denoiser suppresses for free; the locality-bias account of Kadkhodaie et al. predicts $\alpha \approx 0$ here just as loudly. The two hypotheses are non-identifiable under this design. Only the random-orthogonal embedding of Section 8 — which destroys spatial locality while preserving $d$ exactly — separates them, and that experiment has not been run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*