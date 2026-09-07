---
id: 08-loss-and-heads/diffusion-loss-weighting-likelihood-quality
title: "Diffusion Loss Weighting and Likelihood Versus Sample Quality"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Diffusion Loss Weighting and Likelihood Versus Sample Quality

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/diffusion-loss-weighting-likelihood-quality` · **Status:** open

## 1. Problem Statement

A diffusion model is trained by denoising score matching at many noise levels. The only free choice in the objective — once the network, the noise schedule and the parameterization are fixed — is the **weighting function** $w(\sigma)$ over noise levels, together with the **sampling density** $p(\sigma)$ from which training noise levels are drawn. One specific weighting makes the objective an exact bound on negative log-likelihood (NLL); the weightings that produce the best-looking samples are not that one. They up-weight middle noise levels and discard the high-frequency detail levels that dominate bits-per-dimension (bpd).

The problem: **characterize the map from $w(\sigma)$ to the (likelihood, perceptual quality) pair, and decide whether the observed trade-off is intrinsic or an artifact of finite capacity, of the perceptual metric, or of the sampler.**

Three variants, different difficulty:

- **Measurement.** Is the trade-off real, or does it vanish under a perceptual metric that is not FID-with-InceptionV3? Blocked partly by metric validity, not by compute.
- **Method.** Is there a $w(\sigma)$ that is Pareto-dominant — matching likelihood-optimal bpd *and* quality-optimal FID at fixed architecture, compute and sampler? Empirically open.
- **Theory.** For a model class of finite capacity $\mathcal{F}$, prove or refute the existence of an unavoidable frontier: a lower bound on $\mathrm{FID}$ achievable by any $f \in \mathcal{F}$ whose bpd is within $\epsilon$ of the class optimum. No such theorem exists.

Solving it means: a stated rule for choosing $w$ from a declared downstream objective, plus evidence that no other $w$ beats it on that objective at matched compute.

## 2. Formal Setting

Data $x_0 \sim p_{\mathrm{data}}$ on $\mathbb{R}^d$. Forward process $x_\sigma = x_0 + \sigma \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$ (EDM convention; the variance-preserving convention rescales by $\alpha_t$). Denoiser $D_\theta(x_\sigma;\sigma)$ trained with

$$\mathcal{L}(\theta) = \mathbb{E}_{\sigma \sim p(\sigma)}\, \mathbb{E}_{x_0,\varepsilon}\left[\frac{w(\sigma)}{p(\sigma)}\,\lVert D_\theta(x_0+\sigma\varepsilon;\sigma) - x_0\rVert_2^2\right].$$

Only the product $\lambda(\sigma) := w(\sigma)/p(\sigma)$ affects the population minimizer; $p(\sigma)$ alone controls gradient variance. This factorization is why "weighting" comparisons in the literature are frequently non-comparable: two papers can report different $w$ and identical $\lambda$.

**Signal-to-noise ratio.** $\mathrm{SNR}(\sigma) = 1/\sigma^2$ (VP: $\alpha_t^2/\sigma_t^2$). Write $\ell = \log \mathrm{SNR}$.

**Likelihood weighting.** Kingma et al. (VDM, 2021) show the continuous-time ELBO is, up to constants, the diffusion loss with $\lambda_{\mathrm{ELBO}}(\ell) = 1$ uniformly in $\log\mathrm{SNR}$ — equivalently $w(\sigma)\propto 1/\sigma^2$ in $\epsilon$-prediction terms. Song et al. (2021) give the SDE form: with $w(t) = g(t)^2$ the objective upper-bounds NLL.

**Measured quantities.**
- **bpd** $= -\frac{1}{d\ln 2}\log p_\theta(x_0)$ on a held-out set, computed either by the probability-flow ODE with Hutchinson trace estimation (report the number of Hutchinson samples and the ODE tolerance — both shift the third decimal) or as the ELBO plus a discrete dequantization term. **These two estimators are not interchangeable**; ODE bpd is a stochastic estimate of the true likelihood, ELBO bpd is an upper bound.
- **FID** $= \lVert\mu_r-\mu_g\rVert^2 + \mathrm{tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ over InceptionV3 pool3 features, 50k samples. Depends on sampler, step count, guidance scale, and the reference batch.

**Assumptions, and which are violated.**
1. *$\lambda$ determines the population optimum, so weighting only matters through capacity allocation.* True at infinite capacity; the entire phenomenon lives in the violation.
2. *ELBO bpd is tight.* Violated: the gap is unmeasured for large models.
3. *FID measures perceptual quality.* Violated — Kynkäänniemi et al. (ICLR 2023) show FID is steered by ImageNet class content; Stein et al. (NeurIPS 2023) show ranking disagreement with human judgment.
4. *Loss weighting and noise schedule are separable.* Violated in practice: changing $p(\sigma)$ changes gradient noise and hence the effective optimization, so a weighting ablation at fixed step count is confounded with an optimization ablation.

## 3. State of the Art

**Theory SOTA (established).** Song, Durkan, Murray, Ermon, *Maximum Likelihood Training of Score-Based Diffusion Models* (NeurIPS 2021): $w(t)=g(t)^2$ makes the score-matching objective a bound on NLL. Kingma, Salimans, Poole, Ho, *Variational Diffusion Models* (NeurIPS 2021): the ELBO is invariant to the noise schedule except through endpoints and gradient variance — the schedule is not a modeling choice, only the weighting is. Kingma & Gao, *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation* (NeurIPS 2023): **any monotone-increasing weighting in $\ell$ equals an ELBO under Gaussian-noise data augmentation.** This is the sharpest result available and it partially dissolves the dichotomy — but only for monotone weightings; the popular bell-shaped ones (EDM, Min-SNR) are non-monotone and are not covered.

**Empirical SOTA (established, reproduced).** EDM (Karras et al., NeurIPS 2022) — log-normal $p(\sigma)$ with $P_{\mathrm{mean}}=-1.2$, $P_{\mathrm{std}}=1.2$ plus preconditioning; CIFAR-10 conditional FID 1.79. EDM2 (Karras et al., CVPR 2024) adds a *learned* per-noise-level uncertainty that adaptively normalizes the loss, ImageNet-512 FID 1.81 with guidance. Min-SNR-$\gamma$ (Hang et al., ICCV 2023): $\lambda = \min(\mathrm{SNR},\gamma)$, $\gamma=5$, reported ~3.4× faster convergence than baselines for ViT diffusion, ImageNet-256 FID 2.06. Stable Diffusion 3 (Esser et al., ICML 2024) uses logit-normal timestep sampling for rectified flow, selected by sweep.

**Claimed but unablated.** That these weightings are *near-optimal* rather than merely better than uniform. Almost all are single-axis sweeps at one scale, one architecture, one sampler. No published Pareto frontier sweeps $w$ while holding architecture, compute, sampler and guidance fixed and reports both bpd and FID for each arm. The recurring claim "likelihood-weighted training gives worse samples" rests on a small number of comparisons where sampler and capacity were not matched.

## 4. What Is Known

- **The simplified loss beats the ELBO loss on FID.** DDPM (Ho et al., NeurIPS 2020): $L_{\mathrm{simple}}$ ($\lambda \propto \mathrm{SNR}^{-1}\cdot\mathrm{SNR}=$ constant in $\epsilon$-space) gave CIFAR-10 FID 3.17 versus a worse FID with the full variational objective, at 35.7M parameters.
- **Likelihood-optimized diffusion reaches strong bpd.** VDM: CIFAR-10 **2.65 bpd**, ImageNet-32 3.72 bpd (~100M params) — best-in-class at the time; its samples were not competitive with contemporaneous FID leaders. ScoreFlow (Song et al. 2021): CIFAR-10 2.83 bpd with likelihood weighting, and the paper reports FID degradation relative to the same model trained with the sample-quality weighting.
- **Hybrid objectives partially reconcile.** Improved DDPM (Nichol & Dhariwal, ICML 2021): $L_{\mathrm{hybrid}} = L_{\mathrm{simple}} + 0.001\,L_{\mathrm{vlb}}$ with learned $\Sigma$ improves bpd substantially while keeping FID close — evidence that part of the observed trade-off is a *variance-head* problem, not a weighting problem.
- **Monotone weightings are ELBOs.** Kingma & Gao (2023): the widely used $v$-prediction + cosine and EDM-adjacent monotone weightings correspond to ELBOs under augmentation, so "likelihood vs quality" is not a clean dichotomy for that family. They report competitive ImageNet FID from a weighting derived this way.
- **The perceptual metric is unreliable at the resolution the question needs.** Stein et al. (NeurIPS 2023) find FID rankings of diffusion vs GAN models disagree with human evaluation; FID differences below roughly 0.5 on ImageNet carry little signal.

## 5. What Is Not Known

- **Theoretically open.** No lower-bound theorem of the form: for capacity class $\mathcal{F}$ and data $p_{\mathrm{data}}$, achieving bpd within $\epsilon$ of $\min_{\mathcal{F}}$ forces a distributional error (Wasserstein, or FID-analogue) at least $\Phi(\epsilon)$. Nothing rules out a Pareto-dominant weighting. Also open: whether non-monotone weightings (EDM, Min-SNR) admit *any* ELBO interpretation, extending Kingma & Gao.
- **Empirically open.** The full 2-D frontier — a $\ge 10$-arm weighting sweep, fixed architecture and token budget, reporting (bpd, FID, human preference) per arm — has not been published at $\ge$1B parameters. Also open: whether the trade-off shrinks with scale (capacity conflict weakens) or persists.
- **Methodologically blocked.** "Sample quality" has no metric with established construct validity at the scale of differences weighting induces ($\Delta$FID $\sim 0.3$–$2$). Until a metric with human-agreement error bars exists, a Pareto claim cannot be falsified.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement (the dominant one).** Changing $w$ changes $p(\sigma)$ in most implementations, which changes gradient variance, which changes the effective learning rate. A weighting ablation at fixed steps is simultaneously an optimization ablation. Isolating this needs importance-sampled $p(\sigma)$ decoupled from $w$ — rarely done.
2. **An evaluation that does not measure what it names.** FID names "sample quality" and measures ImageNet-feature moment mismatch. bpd names "how well the model fits the data" and is dominated by the lowest noise levels, i.e. by imperceptible high-frequency pixel noise — on natural images most of the bits are in detail humans cannot see. The two metrics are literally weighted views of the same $\sigma$-axis, so their disagreement may be a restatement of the weighting choice rather than a fact about models.
3. **Compute cost of the decisive design.** A clean frontier requires $\ge 10$ arms trained to convergence at a scale where FID differences exceed metric noise. At 1B parameters on ImageNet-512, one arm is $\mathcal{O}(10^3)$ A100-hours; the sweep is $\mathcal{O}(10^4)$ — plus a human-preference study per arm.

## 7. Current Research (as of 2026)

- **Adaptive/learned weightings.** EDM2-style uncertainty weighting (NVIDIA) is the strongest deployed instance: learn $\log$-variance per $\sigma$ and normalize, so weighting is set by observed loss scale rather than by hand.
- **Flow matching and rectified flow** (Meta AI; Stability): weighting reappears as the timestep-sampling density; SD3's logit-normal choice is a rediscovery of the EDM log-normal in a different parameterization. Whether the flow-matching objective has a distinct likelihood/quality structure is unsettled. *(frontier — verify)*
- **Weighting-as-ELBO extensions.** Follow-ups to Kingma & Gao attempting to cover non-monotone weightings via augmentation distributions beyond isotropic Gaussian. *(frontier — verify)*
- **Perceptual-space objectives.** Training in latent/perceptual spaces where the $\sigma$-axis is closer to perceptual scale, making the trade-off geometry different. Active in the video/image generation groups. *(frontier — verify)*
- **Metric replacement.** DINOv2-feature FID and human-preference models (from the Stein et al. line) as the quality axis — a prerequisite for settling the measurement variant.

## 8. Concrete Next Experiment

**The frontier sweep with $p(\sigma)$ decoupled from $w(\sigma)$.**

- **Scale.** 400M-parameter DiT-XL/2-class latent diffusion on ImageNet-256, 400k steps, batch 256 — approximately 200 A100-days total for the whole sweep. Large enough that FID differences of 0.5 are above run-to-run noise (measure that noise: 3 seeds on the control arm).
- **Arms (8).** $\lambda_{\mathrm{ELBO}}$ (uniform in $\log$SNR); EDM log-normal; Min-SNR-$\gamma$ with $\gamma\in\{1,5,20\}$; cosine/$v$; sigmoid; and one interpolation family $\lambda_\beta = \lambda_{\mathrm{ELBO}}^{1-\beta}\lambda_{\mathrm{EDM}}^{\beta}$ at $\beta=0.5$. **Critical control:** hold $p(\sigma)$ identical across all arms (EDM log-normal) and implement $w$ purely as an importance ratio $\lambda = w/p$ in the loss. This removes obstruction 1.
- **Control arm.** $\lambda_{\mathrm{ELBO}}$, same $p(\sigma)$, same seed, same sampler (Heun, 50 steps, no guidance), same 3 seeds.
- **Reported per arm.** ODE bpd (Hutchinson $n=16$, tolerance $10^{-5}$), FID-50k (Inception *and* DINOv2), and 2AFC human preference vs the control on 1,000 pairs.
- **The deciding number.** $\Delta = \mathrm{FID}_{\mathrm{ELBO}} - \min_{\text{arms}} \mathrm{FID}$, restricted to arms whose bpd is within 0.02 of $\mathrm{bpd}_{\mathrm{ELBO}}$. If $\Delta > 0.5$ with non-overlapping 3-seed intervals, the trade-off is not intrinsic and a Pareto-dominant weighting exists at this scale. If no arm satisfies the bpd constraint while improving FID by more than seed noise, the frontier is real at 400M and the question becomes whether it survives to 10B.

## 9. Key References

- **[Foundational]** Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. — arXiv:2006.11239
- **[Foundational]** Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. — arXiv:2011.13456
- **[Theory]** Song, Durkan, Murray, Ermon. *Maximum Likelihood Training of Score-Based Diffusion Models.* NeurIPS 2021. — arXiv:2101.09258
- **[Theory]** Kingma, Salimans, Poole, Ho. *Variational Diffusion Models.* NeurIPS 2021. — arXiv:2107.00630
- **[SOTA / theory]** Kingma, Gao. *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.* NeurIPS 2023. — arXiv:2303.00848
- **[SOTA]** Karras, Aittala, Aila, Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. — arXiv:2206.00364
- **[SOTA]** Karras, Aittala, Lehtinen, Hellsten, Aila, Laine. *Analyzing and Improving the Training Dynamics of Diffusion Models.* CVPR 2024. — arXiv:2312.02696
- **[SOTA]** Hang, Gu, Li, Chen, et al. *Efficient Diffusion Training via Min-SNR Weighting Strategy.* ICCV 2023. — arXiv:2303.09556
- **[Method]** Nichol, Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. — arXiv:2102.09672
- **[Method]** Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022. — arXiv:2202.00512
- **[Method]** Esser, Kulal, Blattmann, et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024. — arXiv:2403.03206
- **[Evaluation]** Theis, van den Oord, Bethge. *A Note on the Evaluation of Generative Models.* ICLR 2016. — arXiv:1511.01844
- **[Evaluation]** Kynkäänniemi, Karras, Aittala, Aila, Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR 2023. — arXiv:2203.06026
- **[Evaluation]** Stein, Cresswell, Hosseinzadeh, et al. *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.* NeurIPS 2023. — arXiv:2306.04675

## 10. Worked Example

Take CIFAR-10, $d = 3072$. Consider the two published endpoints: VDM at **2.65 bpd**, and DDPM-family models trained with $L_{\mathrm{simple}}$ at roughly **3.7 bpd** but FID 3.17 (DDPM) down to 1.97 (EDM, unconditional).

Convert the bpd gap to a total: $(3.7-2.65)\times 3072 \approx 3{,}220$ bits per image. That is a *large* likelihood gap — a factor of $2^{3220}$ in probability mass. Now ask where those bits live. Under the Gaussian channel decomposition, the bpd integrand is $\tfrac{1}{2}\mathbb{E}\lVert D_\theta - x_0\rVert^2$ integrated uniformly in $\log\mathrm{SNR}$. For 8-bit images the discretization floor sits near $\sigma \approx 1/255 \approx 0.004$, i.e. $\ell = \log\mathrm{SNR} \approx 11$. EDM's log-normal $p(\sigma)$ places $P_{\mathrm{mean}}=-1.2$, $P_{\mathrm{std}}=1.2$ in $\ln\sigma$, so $\sigma < 0.004$ receives $\Phi\!\left(\frac{\ln 0.004 + 1.2}{1.2}\right) = \Phi(-3.6) \approx 1.6\times 10^{-4}$ of training weight. Under $\lambda_{\mathrm{ELBO}}$, that same band — if the schedule runs to $\sigma_{\min}=10^{-3}$ — carries roughly $\frac{\ln(0.004/0.001)}{\ln(80/0.001)} \approx 12\%$ of the weight.

So the ELBO arm spends order $10^{-1}$ of capacity, and the EDM arm order $10^{-4}$, on the noise band that produces almost all of the 3,220-bit gap — and that band encodes sub-quantization-step pixel detail, which no human rater and no Inception feature can see.

**The obstruction, visible.** The two metrics are integrals of the *same* per-$\sigma$ error curve under near-disjoint weights: bpd is dominated by $\ell \gtrsim 8$, FID by $\ell \in [-2, 4]$. A model can be optimal on one and mediocre on the other with no contradiction and no shared capacity conflict — the "trade-off" as usually reported may be pure metric non-overlap rather than a Pareto frontier. Distinguishing the two requires measuring the per-$\sigma$ error curve for both arms and checking whether the ELBO arm is *also* worse in the mid-$\sigma$ band (real capacity conflict) or only worse where FID does not look (metric artifact). That curve is cheap to compute and, at 400M-parameter scale, has not been published for matched arms.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*