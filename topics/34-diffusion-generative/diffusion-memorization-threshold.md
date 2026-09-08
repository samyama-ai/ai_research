---
id: 34-diffusion-generative/diffusion-memorization-threshold
title: "Memorization Threshold in Diffusion Models as a Function of Dataset Size"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memorization Threshold in Diffusion Models as a Function of Dataset Size

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-memorization-threshold` · **Status:** empirically-open

## 1. Problem Statement

A diffusion model trained on $N$ samples either reproduces training examples at generation time (memorization) or produces novel samples from a smooth density (generalization). Empirically the switch is sharp in $N$. The problem: **characterize the threshold $N^*$ at which memorization collapses, as a function of dataset size, intrinsic data dimension, model capacity, and training budget.**

Three variants, of very different difficulty:

- **Measurement.** Given a trained model and its training set, produce a memorization rate that is a property of the model, not of the extraction attack or the similarity threshold used to score it. Currently not solved.
- **Method.** Given a target memorization rate $\epsilon$, predict $N^*(\epsilon)$ before training — a scaling law of the form $\log N^* = f(d, P, T)$ for intrinsic dimension $d$, parameters $P$, training steps $T$.
- **Theory.** Prove whether $N^*$ grows exponentially in $d$ (the statistical-physics prediction, since exact score matching on $N$ points always memorizes) or polynomially (the inductive-bias prediction, where architecture, not sample count, buys generalization).

Solved means: a fitted law that predicts, within a factor of 2, the $N$ at which measured memorization drops below $\epsilon$ for a held-out (architecture, dataset) pair.

## 2. Formal Setting

Data $\mathcal{D}_N=\{x_i\}_{i=1}^N \stackrel{iid}{\sim} p_{\mathrm{data}}$ on $\mathbb{R}^D$, supported near a manifold of intrinsic dimension $d \ll D$. Forward process $x_t=\alpha_t x_0+\sigma_t\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$. Denoiser $D_\theta$ trained by denoising score matching:

$$\mathcal{L}(\theta)=\mathbb{E}_{t,x_0,\epsilon}\big[\lambda(t)\,\|D_\theta(x_t,t)-x_0\|_2^2\big].$$

**The central fact that defines the problem.** The unconstrained minimizer over all measurable $D$ is the posterior mean under the *empirical* distribution,

$$D^\star(x_t,t)=\frac{\sum_i x_i\,\mathcal{N}(x_t;\alpha_t x_i,\sigma_t^2 I)}{\sum_j \mathcal{N}(x_t;\alpha_t x_j,\sigma_t^2 I)},$$

whose reverse process converges to $\frac1N\sum_i\delta_{x_i}$. Perfect optimization is perfect memorization. Generalization is therefore a *controlled failure* to reach the optimum, and $N^*$ is the point at which that failure becomes automatic.

**Measured quantities.**

- Similarity $s(x,x_i)$: SSCD embedding cosine (Somepalli et al. 2023), or Carlini's calibrated $\ell_2$: $d(x,x_i)$ divided by the mean distance from $x$ to its other near neighbours, which removes the background-brightness confound.
- Memorization rate at threshold $\tau$: $M_\tau(N)=\Pr_{x\sim p_\theta}\!\big[\max_i s(x,x_i)>\tau\big]$, estimated from $K$ samples; standard error $\sqrt{M(1-M)/K}$, so detecting $M=10^{-6}$ needs $K\gtrsim 10^8$.
- Threshold: $N^*(\epsilon)=\min\{N: M_\tau(N)<\epsilon\}$.
- Counterfactual memorization (adapted from Feldman & Zhang 2020): $\mathrm{cm}(x_i)=\mathbb{E}_{\mathcal{D}\ni x_i}[\,s(\hat x, x_i)\,]-\mathbb{E}_{\mathcal{D}\not\ni x_i}[\,s(\hat x,x_i)\,]$. This is the only definition that separates memorization from dataset redundancy, and it costs $\ge 2$ full training runs per estimate.

**Assumptions, and where they break.** (i) *iid data* — false for LAION, which has near-duplicate clusters with multiplicity $>10^2$; duplication, not $N$, drives most observed replication. (ii) *Fixed compute as $N$ varies* — most published sweeps hold epochs fixed, so $N$ and gradient steps are confounded. (iii) *$\tau$ is a property of the data, not a knob* — false; $M_\tau$ moves by orders of magnitude over plausible $\tau$. (iv) *Unconditional generation* — text conditioning changes the problem: a memorized caption is a near-unique key into one training image.

## 3. State of the Art

**Established (reproduced independently).**

- Extraction works. Carlini et al. (USENIX Security 2023) recovered 94 exact and 13 near-copies of training images from Stable Diffusion v1.4 by generating 500 images for each of the 350,000 most-duplicated captions ($1.75\times10^8$ generations), and extracted 1,280 training images from a CIFAR-10 diffusion model.
- Replication tracks duplication. Somepalli et al. (CVPR 2023; NeurIPS 2023) showed the retrieved-nearest-neighbour rate for SD outputs is on the order of a few percent at their SSCD threshold, and that de-duplicating or randomizing captions sharply reduces it.
- A memorization→generalization transition in $N$ exists for unconditional image diffusion (Yoon et al., ICML 2023 SPIGM workshop; Gu et al. 2023; Kadkhodaie et al., ICLR 2024).

**Claimed but unablated.**

- That the transition is *sharp* (a phase transition) rather than a smooth crossover made to look sharp by a hard similarity threshold. No paper has shown the sharpness survives varying $\tau$.
- Statistical-physics predictions of transition times/sizes (Biroli et al., *Nature Communications* 2024; Bonnaire et al. 2025) are derived for Gaussian-mixture or random-feature data and verified qualitatively on images. The quantitative exponents are theory, not measurement.

**Benchmark-number-only.** Reported "memorization rates" for Stable Diffusion variants (e.g. SD2 trained on de-duplicated data replicating less than SD1.4) are single numbers under one attack, one prompt set and one $\tau$. They do not transfer across attacks.

## 4. What Is Known

- **Numbers, with scale.** SD v1.4, $1.75\times10^8$ generations from 350k duplicated captions → 109 memorized images, a rate of $6.2\times10^{-7}$ per generation *conditioned on the most-duplicated prompts* (Carlini et al. 2023). Webster (2023) recovered a substantially larger set from the same model class with a cheaper detector, showing the 109 is a lower bound set by attack budget.
- **Duplication dominates.** Every memorized SD image in Carlini et al. had duplicates in LAION; images duplicated $\gtrsim 100$ times are far more likely to be extractable than singletons at the same $N$.
- **Two models, one denoiser.** Kadkhodaie et al. (ICLR 2024) trained BF-CNN denoisers on *disjoint* subsets of a face dataset and found near-identical generated samples once $N$ is large (their sweep spans $N=1$ to $\sim10^5$ on small grayscale patches); at small $N$ the two models each reproduce their own training set. This is the cleanest existing operationalization of $N^*$: the $N$ at which two disjoint-data models agree.
- **Capacity matters at fixed $N$.** Gu et al. (2023) define "effective model memorization" and report that memorization on CIFAR-10-scale data increases with model size and training time and decreases with $N$, with conditioning (especially unique per-image conditioning) sharply increasing it.
- **Theory of the optimum.** Score-based models fit to $N$ points converge to the empirical measure; the learned score detects the data manifold (Pidstrigach, NeurIPS 2022). So any generalization observed is an optimization/architecture effect.

## 5. What Is Not Known

- **Theoretically open.** Whether $N^*$ scales as $\exp(\Theta(d))$ or $\mathrm{poly}(d)$ for realistic architectures. Exact-score arguments give a curse-of-dimensionality answer; convolutional/equivariant inductive-bias arguments (Kadkhodaie et al. 2024; Kamb & Ganguli 2025) suggest generalization from far fewer samples. No proof either way for a U-Net.
- **Empirically open.** No published sweep varies $N$ over $\ge 3$ decades at **fixed gradient steps and fixed architecture** with $\ge 3$ seeds and reports $M_\tau(N)$ as a function of $\tau$. The experiment is runnable on 8 GPUs; nobody has run it as a clean control.
- **Methodologically blocked.** A memorization rate that is invariant to the extraction attack. Present numbers are attack-dependent lower bounds. Counterfactual memorization is attack-free but costs $O(N)$ retrainings for per-example estimates.

## 6. Why It Is Hard

Three named obstructions.

1. **Non-identifiability of the rate.** $M_\tau$ is a function of (model, attack, prompt distribution, $\tau$). Carlini's $6.2\times10^{-7}$ and Somepalli's few-percent describe the same model family; they differ by five orders of magnitude because the conditioning distributions differ. Without fixing the attack, $N^*$ is not a number.
2. **Confounded sweep.** Shrinking $N$ at fixed epochs shrinks total gradient steps; shrinking $N$ at fixed steps raises epochs. Both are confounds, in opposite directions, and published sweeps rarely say which they used.
3. **Statistical cost of small $\epsilon$.** Resolving $M=10^{-6}$ needs $\sim10^8$ generations per grid point. At 20 NFE and 512² resolution that is thousands of GPU-hours *per point on the curve*, before seeds.

## 7. Current Research (as of 2026)

- **Statistical physics of the transition.** Biroli, Mézard, de Bortoli and collaborators (ENS/Bocconi/DeepMind): speciation and collapse times, and the claim that memorization onset is governed by a time-scale separation between a generalization time and a memorization time that grows with $N$ — implying early stopping is an implicit regularizer *(frontier — verify the exponents)*.
- **Geometric detection.** Ross, Cresswell and colleagues (Layer 6 AI): local intrinsic dimension of the learned score as a per-sample memorization detector, ICLR 2025.
- **Detection and mitigation at deployment.** Wen et al. (ICLR 2024): magnitude of the text-conditional noise prediction as a memorization trigger detector; prompt perturbation as mitigation.
- **Inductive-bias theory.** Simoncelli/Mallat (NYU/ENS) on geometry-adaptive harmonic bases; Kamb & Ganguli (Stanford) on locality and equivariance predicting diffusion outputs analytically — both argue $N^*$ is set by architecture, not by $d$ alone *(frontier — verify)*.
- Differential privacy and de-duplication pipelines as engineering answers that sidestep, rather than measure, $N^*$.

## 8. Concrete Next Experiment

**Goal.** Decide whether $\log N^*$ grows linearly in intrinsic dimension $d$ (exponential curse) or is flat (architecture-limited).

**Scale.** Procedurally generated 64×64 image datasets from a renderer with *known* intrinsic dimension $d\in\{4,8,16,32\}$ (e.g. sprite scenes with $d$ continuous latent factors), sampled without duplicates. For each $d$, train an EDM/DDPM++ U-Net (~55M params, fixed across all runs) on $N\in\{2^8,2^{10},\dots,2^{18}\}$, 3 seeds. **Fixed budget: 200M images seen** — i.e. fixed gradient steps, not fixed epochs. 108 runs; ~8 GPU-hours each on A100s at this resolution, ≈900 GPU-hours plus sampling.

**Control arm.** For each $(d,N)$, a paired run on a *disjoint* dataset of the same size and same $d$. Memorization is scored two ways: (a) $M_\tau(N)$ against the training set at $\tau\in\{0.5,0.6,0.7,0.8\}$ SSCD, and (b) the attack-free Kadkhodaie criterion — mean SSCD between samples of the two disjoint-data models at matched noise seeds. A second control fixes epochs instead of steps, on the $d=8$ row only, to quantify the confound.

**Deciding number.** Fit $\log_2 N^*(\epsilon{=}10^{-2}) = a + b\,d$ across the four $d$ values, with $N^*$ from the attack-free criterion. Report $b$ with a bootstrap 95% CI over seeds and $\tau$. $b \ge 0.5$ bits per latent dimension, with CI excluding 0, supports the exponential/curse picture. $b$ indistinguishable from 0 supports the architecture-limited picture. One number, one CI.

## 9. Key References

- **[Foundational]** N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramèr, B. Balle, D. Ippolito, E. Wallace. *Extracting Training Data from Diffusion Models.* USENIX Security, 2023. — arXiv:2301.13188
- **[Foundational]** G. Somepalli, V. Singla, M. Goldblum, J. Geiping, T. Goldstein. *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models.* CVPR, 2023. — arXiv:2212.03860
- **[SOTA]** Z. Kadkhodaie, F. Guth, E. P. Simoncelli, S. Mallat. *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations.* ICLR, 2024. — arXiv:2310.02557
- **[SOTA]** X. Gu, C. Du, T. Pang, C. Li, M. Lin, Y. Wang. *On Memorization in Diffusion Models.* 2023. — arXiv:2310.02664
- **[SOTA]** G. Biroli, T. Bonnaire, V. de Bortoli, M. Mézard. *Dynamical Regimes of Diffusion Models.* Nature Communications, 2024. — arXiv:2402.18491
- **[SOTA]** Y. Wen, Y. Liu, C. Chen, L. Lyu. *Detecting, Explaining, and Mitigating Memorization in Diffusion Models.* ICLR, 2024.
- **[Related]** G. Somepalli, V. Singla, M. Goldblum, J. Geiping, T. Goldstein. *Understanding and Mitigating Copying in Diffusion Models.* NeurIPS, 2023. — arXiv:2305.20086
- **[Related]** J. Pidstrigach. *Score-Based Generative Models Detect Manifolds.* NeurIPS, 2022.
- **[Related]** T. Yoon, J. Y. Choi, S. Kwon, E. K. Ryu. *Diffusion Probabilistic Models Generalize when They Fail to Memorize.* ICML 2023 Workshop on Structured Probabilistic Inference and Generative Modeling.
- **[Related]** V. Feldman, C. Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS, 2020. — arXiv:2008.03703
- **[Survey]** C. Meehan, K. Chaudhuri, S. Jegelka. *A Non-Parametric Test to Detect Data-Copying in Generative Models.* AISTATS, 2020.

## 10. Worked Example

Take Carlini et al.'s Stable Diffusion v1.4 result and try to read $N^*$ off it.

Measured: 109 memorized images from $K=1.75\times10^8$ generations, so

$$\hat M = \frac{109}{1.75\times10^8} = 6.2\times10^{-7}.$$

Training set: $N\approx1.6\times10^8$ LAION images (the SD1.4 aesthetic subset). If $M_\tau(N)$ were a function of $N$ alone, this single point would pin the curve near $N^*(10^{-6})\approx1.6\times10^8$.

It does not, for three reasons visible in the arithmetic.

1. **The prompts were not sampled from the training distribution.** The 350k captions used are the most-duplicated ones — the top $0.2\%$ of $1.6\times10^8$. Sampling captions uniformly instead would lower $\hat M$ by at least the duplication-enrichment factor, plausibly $10^2$–$10^3$. Same model, same $N$, $\hat M$ moves to $\sim10^{-9}$.
2. **The rate is an attack lower bound.** Webster's cheaper detector on the same model family surfaced far more replications than 109. Raising attack compute raises $\hat M$ without touching the model. So $\hat M$ is monotone in a variable that is not part of the model.
3. **The effective $N$ is not $1.6\times10^8$.** After near-duplicate collapse, the memorized images sit in clusters of multiplicity $\ge10^2$; the model's exposure to those examples is that of a dataset of size $\sim10^6$ unique concepts, not $10^8$.

So the same experiment supports $\hat M$ anywhere in $[10^{-9},10^{-5}]$ and $N_{\mathrm{eff}}$ anywhere in $[10^6,10^8]$ — a point with error bars spanning four decades on one axis and two on the other. That is the obstruction: not that the number is hard to compute, but that the estimator is a function of the attack and the prompt distribution, so no number of extra generations narrows it. The §8 design removes all three by using a fixed, duplicate-free generator with known $d$, unconditional sampling, and an attack-free disjoint-model criterion.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*