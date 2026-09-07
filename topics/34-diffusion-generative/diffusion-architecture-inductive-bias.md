---
id: 34-diffusion-generative/diffusion-architecture-inductive-bias
title: "Architecture Inductive Bias Beyond UNet and Transformer Parity"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Architecture Inductive Bias Beyond UNet and Transformer Parity

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-architecture-inductive-bias` · **Status:** empirically-open

## 1. Problem Statement

Diffusion models were built on convolutional UNets with multi-scale skips; they now train equally well or better as plain transformers over patch tokens. The field reads this as "architecture does not matter, scale does." That reading is not established.

The problem: **does the denoiser's architecture contribute inductive bias that survives scale, or is UNet-vs-transformer parity an artifact of measuring only FID at one resolution on one dataset?**

Three variants, different difficulty:

- **Measurement.** Define a quantity that separates *architectural prior* from *effective capacity* and *optimization ease*, and that is not saturated by FID on ImageNet-256. Currently undefined; this is the binding constraint.
- **Method.** Given a compute budget $C$, data budget $N$, and resolution $R$, predict which family (conv-UNet, isotropic DiT, hierarchical transformer, SSM) minimizes held-out score-matching loss. Solving means a fitted rule that transfers to an unseen $(C, N, R)$ point.
- **Theory.** Prove a separation: exhibit a data distribution class where one architecture attains $\varepsilon$ score error with $\mathrm{poly}$ parameters and the other requires $\exp$. No such separation exists for the diffusion setting.

## 2. Formal Setting

Data $x_0 \sim p_{\mathrm{data}}$ on $\mathbb{R}^{d}$, $d = 3R^2$ in pixel space or $d = c h w$ in a latent space. Forward process $x_\sigma = x_0 + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. Denoiser $D_\theta(x_\sigma; \sigma)$ from architecture family $\mathcal{A}$. Training objective (EDM parameterization):

$$\mathcal{L}(\theta) = \mathbb{E}_{\sigma \sim p(\sigma)} \, \mathbb{E}_{x_0, \epsilon} \left[ \lambda(\sigma) \, \| D_\theta(x_0 + \sigma\epsilon; \sigma) - x_0 \|_2^2 \right].$$

**Quantities as measured.**

- **Compute** $C$: training FLOPs, $C \approx 6 N_{\mathrm{param}} \cdot N_{\mathrm{tok}}$ per pass for transformers; for UNets, measured directly with a profiler, since the $6ND$ rule is wrong under resolution-dependent channel widths. Report *both* training FLOPs and wall-clock on fixed hardware — the two rank families differently.
- **Held-out denoising loss** $\mathcal{L}^\star(\mathcal{A}, C)$: $\mathcal{L}$ evaluated on a held-out split with the *same* $\lambda(\sigma)$, $p(\sigma)$, and preconditioning across arms. This is the only architecture-comparable scalar; FID is not, because it moves under sampler and guidance changes that are orthogonal to $\theta$.
- **Per-noise-level loss curve** $\ell(\sigma) = \mathbb{E}\left[\lambda(\sigma)\|D_\theta - x_0\|^2\right]$. Architectural bias, if it exists, should show as a *shape* difference in $\ell(\sigma)$ — conv priors helping at low $\sigma$ (local texture) and mattering less at high $\sigma$ (global layout).
- **Sample quality**: FID, FD-DINOv2, precision/recall, at a *fixed* sampler (same solver, same NFE, guidance swept and the minimum reported per arm).

**Assumptions, and which are violated.**

1. *Arms are compute-matched.* Violated in nearly all published comparisons: UNet and DiT baselines differ in training steps, batch size, EMA schedule, and augmentation.
2. *FID ranks models the way held-out likelihood does.* Violated — guidance scale changes FID by more than most architecture gaps.
3. *The VAE latent space is architecture-neutral.* Violated: the standard $8\times$ KL autoencoder was tuned alongside conv UNets, and its latents carry a convolutional smoothness prior that a patchifying transformer inherits for free.
4. *One resolution generalizes.* Violated: isotropic transformers cost $O(R^4)$ in attention at fixed patch size, so parity at $256^2$ says nothing at $1024^2$.

## 3. State of the Art

**Established (compute-matched or close, with ablations).**

- **DiT** (Peebles & Xie, ICCV 2023) showed FID falls monotonically with transformer Gflops across DiT-S/B/L/XL and patch sizes 8/4/2, and that DiT-XL/2 reaches FID 2.27 on class-conditional ImageNet-256 with guidance, beating the LDM and ADM UNet baselines. The scaling trend within the DiT family is well ablated. The *cross-family* claim rests on comparison to numbers reported by other papers under different training budgets.
- **U-ViT** (Bao et al., CVPR 2023) showed the long skip connections, not the convolutional stem or downsampling, are the load-bearing part of the UNet; removing skips degrades sharply, removing convolutions does not.
- **EDM2** (Karras et al., CVPR 2024) held the architecture family fixed (a UNet) and improved ImageNet-512 FID to about 1.81 purely through magnitude-preserving normalization, weight-decay-free training and post-hoc EMA. This is the strongest evidence that *training-recipe* variance dominates the reported architecture gaps.

**Claimed but unablated.**

- Hierarchical transformers (HDiT, Crowson et al., ICML 2024) claim pixel-space megapixel training without the usual multi-scale hacks; the reported ImageNet-256 FID (~2.1 class-conditional) is competitive, but the comparison to a compute-matched UNet at the same resolution is not run.
- SSM/Mamba denoisers (DiffuSSM, Yan et al., CVPR 2024; Diffusion Mamba variants, 2024) claim better FLOP scaling at high resolution. Evidence exists mostly as single benchmark numbers at $256^2$–$512^2$, not as a fitted compute-vs-loss curve.
- **REPA** (Yu et al., ICLR 2025) shows that aligning DiT/SiT internal features to a pretrained DINOv2 representation cuts training time by more than an order of magnitude and reaches FID ≈ 1.4 on ImageNet-256. This is *representation* prior injected externally, which is direct evidence that the transformer's own inductive bias is weak — but it has not been run on a UNet arm to test whether the gain is architecture-specific.

## 4. What Is Known

- **Within-family scaling is clean.** DiT: FID improves monotonically with Gflops from ~10 to ~120 Gflops per forward pass at ImageNet-256; the S→XL and patch-8→patch-2 axes trade off almost interchangeably at equal Gflops (ICCV 2023).
- **Recipe beats architecture at fixed family.** EDM2 improved ImageNet-512 FID from ~2.4-class UNet baselines to ~1.81 with no change of family (CVPR 2024). The magnitude of that gain exceeds most published UNet-vs-DiT gaps.
- **Skips matter, convolutions less so.** U-ViT ablations at ImageNet-256 and CIFAR-10 scale (CVPR 2023).
- **Interpolant/objective choice is worth ~0.2–0.5 FID at DiT-XL scale.** SiT (Ma et al., ECCV 2024) reached ~2.06 on ImageNet-256 by changing only the diffusion-to-flow formulation, holding the DiT backbone fixed.
- **Resolution changes the ranking of costs, not (yet) of quality.** simple diffusion (Hoogeboom et al., ICML 2023) showed a single-stage UNet with most capacity at $16\times16$ handles $512^2$/$1024^2$ pixel space; the analogous compute allocation in transformers is patch size, and no paper has measured both under one budget at $1024^2$.
- **No published cross-family compute-matched scaling law.** Searching for a plot of $\mathcal{L}^\star$ vs $C$ with UNet and DiT curves fitted under one recipe returns nothing at ImageNet scale.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted scalar that isolates architectural prior. $\mathcal{L}^\star$ is comparable only if preconditioning and $\lambda(\sigma)$ are shared, which forces every arm into the EDM parameterization and thereby removes one of the things being compared. FID is confounded by guidance and sampler.
- **Empirically open.** The compute-matched two-family scaling law — fit $\mathcal{L}^\star(C) = a C^{-\alpha} + \mathcal{L}_\infty$ for conv-UNet and for isotropic transformer, under one recipe, over $\ge 2$ decades of $C$ — is runnable today for roughly $10^{21}$–$10^{22}$ FLOPs. Nobody has published it. Whether the curves cross, are parallel with an offset, or have different $\alpha$ is unknown.
- **Empirically open.** Whether parity holds at $1024^2$ and above, and whether it holds off natural images (medical volumes, molecular grids, audio spectrograms) where the translation-equivariance prior may be worth more or less.
- **Theoretically open.** No separation theorem. There is no known distribution family where a UNet score approximator is provably polynomially sized and any bounded-depth transformer is provably exponentially sized, or vice versa, in the $L^2(p_\sigma)$ score-approximation metric.

## 6. Why It Is Hard

The obstruction is **confounded measurement under a compute floor**, in two parts.

1. *Non-identifiability of prior vs. capacity vs. optimizability.* A family that wins at fixed $C$ may win because it encodes the right symmetry, because its FLOPs buy more effective parameters, or because Adam works better on it. These three are not separable by any single held-out number. EDM2's result shows term three alone can be worth more than the whole measured gap, so any comparison in which the recipe is not co-tuned per arm measures recipe, not architecture.
2. *The decisive regime is expensive.* Inductive bias should matter most in the data-limited and high-resolution corners: small $N$, large $R$. Two families $\times$ 5 compute points $\times$ 2 resolutions, at the scale where the curves separate, is on the order of $10^{22}$ FLOPs — thousands of H100-days. The cheap version ($64^2$, CIFAR) sits in the regime where both families are underfit and the curves are near-parallel, so it cannot decide the question.

Add: the standard latent space is not neutral (assumption 3 above), so a pixel-space arm is required, which raises cost again.

## 7. Current Research (as of 2026)

- **Recipe-normalized scaling** — NVIDIA's EDM line (Karras et al.) continues to show that normalization and EMA choices dominate; the natural extension is applying the EDM2 recipe unchanged to a transformer backbone. *(frontier — verify whether a published compute-matched version exists.)*
- **Representation-aligned training** — REPA and successors (Yu, Xie et al.) treat external representation priors as a substitute for architectural ones. If a DINOv2-aligned DiT and a DINOv2-aligned UNet converge to the same $\mathcal{L}^\star$, that is strong evidence that architecture is a proxy for representation quality only.
- **Efficient long-sequence denoisers** — linear-attention and SSM backbones for $\ge 1024^2$; claimed FLOP advantages, mostly unbenchmarked as scaling curves. *(frontier — verify.)*
- **Hierarchical transformers** — HDiT-style neighborhood attention pyramids, which are UNets with attention blocks; these blur the dichotomy and suggest the real axis is *multi-scale token allocation*, not conv-vs-attention. *(frontier — verify.)*

## 8. Concrete Next Experiment

**The compute-matched two-family scaling law, in pixel space, at two resolutions.**

- **Scale.** ImageNet at $R = 256$ and $R = 64$, pixel space (no VAE, to kill assumption 3). Five compute points per arm, log-spaced from $2\times10^{19}$ to $2\times10^{21}$ training FLOPs. Two arms: EDM2-style conv UNet, and an isotropic DiT with patch size chosen so token count matches UNet's bottleneck cost. Total ≈ $10^{22}$ FLOPs, roughly 1,500–3,000 H100-days.
- **Control arm.** Identical everywhere else: EDM preconditioning, identical $p(\sigma)$ and $\lambda(\sigma)$, identical augmentation, identical post-hoc EMA sweep, and a *per-arm* learning-rate/weight-decay sweep at the smallest compute point extrapolated by the same rule. Report FLOPs measured by profiler, not estimated.
- **The deciding number.** Fit $\mathcal{L}^\star(C) = a C^{-\alpha} + \mathcal{L}_\infty$ per arm and report $\Delta\alpha = \alpha_{\mathrm{UNet}} - \alpha_{\mathrm{DiT}}$ with a bootstrap CI over seeds.
  - $|\Delta\alpha| < 0.01$ with overlapping CI at both resolutions → parity is real; architecture is an efficiency constant, not a prior. Problem closes toward "solved, negative."
  - $|\Delta\alpha| > 0.03$, or a sign flip between $R=64$ and $R=256$ → architectural bias survives scale and is resolution-dependent. Problem stays open with a sharpened target.
- **Free secondary readout.** $\ell(\sigma)$ curves per arm. If the families differ only at $\sigma < 0.5$, the prior is a texture prior and is removable by better local parameterization.

## 9. Key References

- **[Foundational]** Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. — arXiv:2006.11239
- **[Foundational]** Prafulla Dhariwal, Alex Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS, 2021. — arXiv:2105.05233
- **[Foundational]** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS, 2022. — arXiv:2206.00364
- **[SOTA]** William Peebles, Saining Xie. *Scalable Diffusion Models with Transformers.* ICCV, 2023. — arXiv:2212.09748
- **[SOTA]** Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, Jun Zhu. *All are Worth Words: A ViT Backbone for Diffusion Models.* CVPR, 2023. — arXiv:2209.12152
- **[SOTA]** Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, Samuli Laine. *Analyzing and Improving the Training Dynamics of Diffusion Models.* CVPR, 2024. — arXiv:2312.02696
- **[SOTA]** Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, Saining Xie. *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers.* ECCV, 2024. — arXiv:2401.08740
- **[SOTA]** Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie. *Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think.* ICLR, 2025. — arXiv:2410.06940
- **[Method]** Emiel Hoogeboom, Jonathan Heek, Tim Salimans. *simple diffusion: End-to-end diffusion for high resolution images.* ICML, 2023. — arXiv:2301.11093
- **[Method]** Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z. Kaplan, Enrico Shippole. *Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers.* ICML, 2024. — arXiv:2401.11605
- **[Method]** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR, 2022. — arXiv:2112.10752
- **[Survey]** Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, Ming-Hsuan Yang. *Diffusion Models: A Comprehensive Survey of Methods and Applications.* ACM Computing Surveys, 2023. — arXiv:2209.00796

## 10. Worked Example

Take the two headline ImageNet-256 numbers usually cited as evidence that transformers won: ADM-U at FID ≈ 3.94 and DiT-XL/2 at FID 2.27. The apparent gap is 1.67 FID.

Now account for the confounds, in order:

| Source of variance | Measured magnitude | Where from |
|---|---|---|
| Guidance scale sweep on one fixed model | > 2 FID between cfg 1.0 and the optimum | standard cfg curves, DiT paper |
| Training recipe at fixed family (EDM2) | ~0.6 FID at ImageNet-512, from normalization + post-hoc EMA alone | Karras et al., CVPR 2024 |
| Objective/interpolant at fixed backbone (SiT) | ~0.2 FID at DiT-XL scale | Ma et al., ECCV 2024 |
| Claimed architecture gap | 1.67 FID | ADM vs DiT reported numbers |

Two of the three nuisance terms are individually within a factor of ~3 of the effect, and one exceeds it. The arms also differ in training duration (DiT-XL/2 was trained for ~7M steps at batch 256), in whether a VAE is used at all (ADM is pixel-space, DiT is latent), and in EMA handling. **The architecture effect is not identifiable from these two numbers.**

Push it further. Suppose you run the Section 8 experiment and get $\alpha_{\mathrm{UNet}} = 0.081$, $\alpha_{\mathrm{DiT}} = 0.079$, with offsets such that DiT's curve sits 0.4% lower in $\mathcal{L}^\star$ at every $C$. Extrapolating a 0.002 difference in exponent across the two decades you measured predicts a crossover at $C \approx 10^{26}$ FLOPs — three decades beyond your data and well inside the bootstrap CI of "never." The obstruction is visible here: at the compute you can afford, the two families are separated by a *constant factor*, and constant factors are exactly what recipe tuning moves. To distinguish "architecture is a 0.4% efficiency constant" from "architecture is a prior with a different exponent," you need the curve over enough decades that $\Delta\alpha$ is resolvable — and that is the experiment nobody has funded.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*