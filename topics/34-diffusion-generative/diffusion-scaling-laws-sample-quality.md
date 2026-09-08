---
id: 34-diffusion-generative/diffusion-scaling-laws-sample-quality
title: "Scaling Laws for Diffusion Model Sample Quality"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Diffusion Model Sample Quality

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-scaling-laws-sample-quality` · **Status:** empirically-open

## 1. Problem Statement

For autoregressive language models, a single power law in parameters $N$ and tokens $D$ predicts held-out loss, and loss transfers monotonically to downstream quality. Diffusion models have no such chain. The denoising loss is a weighted ELBO whose weighting is chosen by the practitioner; sample quality is measured by FID, which is not a monotone function of that loss; and quality also depends on sampler steps and guidance scale, neither of which appears in training-time scaling laws.

Three variants, of different difficulty:

- **Measurement.** Does there exist a scalar sample-quality functional $Q$, computable at the scales people train at, that is (a) a deterministic function of the model distribution rather than of a particular feature extractor's blind spots, and (b) monotone in the training objective? Currently blocked: FID fails (a).
- **Method.** Given a compute budget $C$ (training FLOPs) and an inference budget $B$ (FLOPs per sample), what is the compute-optimal allocation $(N^\star, D^\star, \text{NFE}^\star, w^\star)$ — parameters, training samples-seen, sampler function evaluations, guidance scale — and does it follow a power law? Empirically open.
- **Theory.** Is there a bound of the form $W_2(p_\theta, p_{\text{data}}) \le f(N, D, \text{NFE})$ with matching lower bound for realistic architectures? Theoretically open; minimax rates exist only under smoothness assumptions nobody claims hold for images.

**Solved** means: a fitted law that predicts, out of sample and across at least one order of magnitude of extrapolation in $C$, the quality of a held-out configuration to within the seed-to-seed noise of the metric — and that gets the *ranking* of allocations right, not just the level.

## 2. Formal Setting

Data $x_0 \sim p_{\text{data}}$ on $\mathbb{R}^d$ (or on a VAE latent, $d = 4\times64\times64$ for SD-style latents at $512^2$). Forward process $x_t = \alpha_t x_0 + \sigma_t \epsilon$, $\epsilon\sim\mathcal N(0,I)$. Model $\epsilon_\theta$ or $D_\theta$ with $N$ parameters. Training objective, **as actually computed**:

$$\mathcal{L}(\theta) = \mathbb{E}_{t\sim q,\, x_0,\, \epsilon}\big[\lambda(t)\,\|D_\theta(x_t,t) - x_0\|_2^2\big].$$

Measured quantities:

- $N$ — non-embedding parameters, model only, excluding the frozen VAE and text encoder. Papers differ here; comparisons across papers are unreliable unless the convention is stated.
- $D$ — images seen (samples $\times$ epochs), not unique images. Diffusion training is heavily repeated-epoch; the LM notion of "tokens" has no clean analogue.
- $C \approx 6ND$ for a transformer backbone, which double-counts nothing but ignores attention $O(L^2)$ terms that matter at high resolution.
- $\mathcal{L}_{\text{val}}$ — the same weighted loss on held-out data, with $\lambda(t)$ and the $t$-sampling distribution $q$ **fixed across all compared runs**. This is the single largest source of incomparability in the literature: EDM's $\lambda$, $v$-prediction, and rectified-flow logit-normal $q$ give different numbers for the same model.
- $\text{NFE}$ — network evaluations per sample; with classifier-free guidance at scale $w$, cost is $2\times\text{NFE}$.
- $\mathrm{FID} = \|\mu_r-\mu_g\|^2 + \mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ over Inception-V3 pool3 features, typically $50{,}000$ generated samples against the full training set.

The hoped-for law, by analogy to Chinchilla:

$$\mathcal{L}_{\text{val}}(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \qquad Q(N,D,\text{NFE},w) = g\big(\mathcal{L}_{\text{val}}\big) + \text{(sampler terms)}.$$

**Assumptions known to be violated.**
1. *$Q$ monotone in $\mathcal{L}_{\text{val}}$* — false at fixed guidance: $w$ trades FID against ELBO, and the FID-optimal $w>1$ strictly worsens likelihood.
2. *FID measures distributional distance* — false as used; the estimator is biased at $N_{\text{gen}}=50\text{k}$ and dominated by ImageNet-class content (Chong & Forsyth 2020; Kynkäänniemi et al. 2023).
3. *Single power law over the whole range* — the sampler contributes a term that saturates at large NFE, so any law with unbounded improvement in NFE is wrong by construction.
4. *Fixed data distribution* — most frontier results change data mixture as they scale, confounding $N$ with data quality.

## 3. State of the Art

**Established (ablated, reproduced).**
- **Loss scales smoothly with $N$ and $C$.** DiT (Peebles & Xie, ICCV 2023) showed FID decreases monotonically with backbone Gflops across S/B/L/XL at ImageNet-256, holding data and sampler fixed — the cleanest controlled scaling ablation in the field. DiT-XL/2: FID 2.27 with guidance, 9.62 without.
- **Loss $\to$ preference transfer at fixed sampler.** Esser et al. (ICML 2024, SD3) fit validation loss against model size to 8B parameters and report that lower validation loss tracks GenEval and human preference *within their sweep*, with no sign of saturation.
- **Bigger is not always better under an inference budget.** Mei et al. (TMLR 2024) trained latent diffusion models from 39M to 5B parameters and found that under a *fixed sampling cost*, smaller models frequently match or beat larger ones, because larger models are only better per-step, not per-FLOP.

**Claimed but unablated.**
- Explicit power-law fits for diffusion transformers (e.g. Liang et al., *Scaling Laws For Diffusion Transformers*, 2024) fit $\mathcal{L}(C)$ over roughly $10^{17}$–$6\times10^{18}$ FLOPs and extrapolate. The fit is to loss; the extrapolation to *sample quality* rests on an assumed monotone map that is not tested at the extrapolated point.
- MoE diffusion scaling (DiT-MoE to 16.5B params) reports competitive FID but does not isolate the scaling exponent from architecture changes.

**Benchmark-number-only.** ImageNet-512 FID 1.81 (EDM2-XXL, guided; Karras et al., CVPR 2024) and FID 1.25 with autoguidance (Karras et al., NeurIPS 2024) are single-configuration records, not points on a controlled scaling curve.

## 4. What Is Known

- **Score-estimation error controls sample error.** Chen et al. (ICLR 2023) prove that with an $L^2$-accurate score, $\epsilon_{\text{score}}$, DDPM-style samplers converge in TV/$W_2$ with polynomial iteration complexity under only finite second moment — no log-concavity. This makes "better denoiser $\Rightarrow$ better samples" a theorem, but with constants far too loose to predict FID.
- **Minimax optimality.** Oko, Akiyama & Suzuki (ICML 2023) show diffusion models attain nearly minimax-optimal rates for Besov-class densities, with the rate governed by intrinsic dimension. Predicts $\alpha$ depends on data manifold dimension, unmeasured for real images.
- **Sampler cost floor.** EDM (Karras et al., NeurIPS 2022) reaches FID 1.79 on CIFAR-10 at 35 NFE; second-order solvers buy roughly a $5$–$10\times$ NFE reduction versus DDPM's 1000 steps at matched FID.
- **Guidance breaks the loss$\to$FID link.** Dhariwal & Nichol (NeurIPS 2021) showed guidance improves FID and IS while reducing diversity; the FID-optimal $w$ is typically $1.5$–$3.0$ at ImageNet-256 and worsens the ELBO.
- **FID's own failure modes are quantified.** Kynkäänniemi et al. (ICLR 2023) show FID can be moved substantially by matching ImageNet-class histograms without any perceptual change; Stein et al. (NeurIPS 2023) find FID rankings disagree with human judgment on diffusion vs GAN comparisons at ImageNet scale.

## 5. What Is Not Known

- **Methodologically blocked:** the target of the law. There is no agreed $Q$ that is metric-stable, sample-efficient, and monotone in model quality. Until $Q$ is fixed, "scaling law for sample quality" names a curve whose $y$-axis is undefined. This is the binding constraint.
- **Empirically open:** the compute-optimal frontier. Nobody has run an IsoFLOP sweep — $\ge 5$ compute budgets $\times \ge 4$ $(N,D)$ splits each, fixed data, fixed $\lambda(t)$, fixed sampler — and reported $N^\star \propto C^{a}$ for diffusion. The Chinchilla analogue ($a \approx 0.5$) is simply unmeasured. Runnable today at $\sim 10^{21}$ FLOPs total.
- **Empirically open:** train/inference joint optimality. Whether the optimal $(N,\text{NFE})$ trade at fixed total serving cost follows a clean exponent, and whether distillation shifts it.
- **Theoretically open:** any bound relating $N$ (finite network capacity) to $\epsilon_{\text{score}}$ for realistic architectures on image data. All current theory assumes the score is learned to accuracy $\epsilon$ and says nothing about how many parameters that costs.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by an evaluation that does not measure what it names**, not compute.

FID is a two-moment Gaussian distance in a feature space chosen in 2015 for ImageNet classification. It is (i) biased at finite sample count, so a $50$k-sample FID at model A and model B differs by an amount that depends on their entropy, not only their fidelity; (ii) sensitive to ImageNet-class frequency, so a model that shifts content composition scores differently at identical perceptual quality; (iii) not decomposable into fidelity and diversity, so guidance — which trades one for the other — moves it non-monotonically. A scaling law fit to FID therefore fits a curve whose residuals are structured, not noise, and extrapolation amplifies exactly that structure.

Second obstruction: **non-identifiability of the loss axis.** $\lambda(t)$ is a free function. Two papers reporting "validation loss vs. $N$" with different $\lambda$ are fitting different functionals; the exponents are not comparable, and no paper reports the same runs under two weightings.

## 7. Current Research (as of 2026)

- **IsoFLOP-style diffusion sweeps.** Follow-on work to Liang et al. and Mei et al. is extending fits to text-to-image at $>10^{19}$ FLOPs with fixed data mixtures *(frontier — verify)*.
- **Inference-time scaling.** "Search over noise" / verifier-guided sampling (Ma et al., 2025, Google/NYU) shows FID and preference improve with sampling compute beyond added denoising steps — adds a third axis to the law *(frontier — verify)*.
- **Metric replacement.** DINOv2-feature FD, CMMD (Jayasumana et al., CVPR 2024, CLIP-MMD), and precision/recall pairs are being adopted as FID substitutes; no consensus, and no paper has refit a scaling law under two metrics to show the exponent is metric-invariant.
- **Post-training-aware laws.** NVIDIA (EDM line) and Stability continue to report that guidance/autoguidance changes the optimal model size at fixed FID, which if true means the law must be stated jointly over $(N, w)$ *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is the fitted scaling exponent an artifact of the metric?

**Scale.** ImageNet-512 latents, DiT/EDM2-style backbone. Five model sizes $N \in \{80\text{M}, 160\text{M}, 320\text{M}, 640\text{M}, 1.3\text{B}\}$, each trained at four token budgets on an IsoFLOP grid spanning $3\times10^{19}$–$3\times10^{21}$ FLOPs. Fixed data (no mixture changes), fixed EDM $\lambda(t)$, fixed 2nd-order Heun sampler at NFE $=63$, guidance swept $w \in \{1.0, 1.5, 2.0, 3.0\}$ at eval only. Roughly 20 runs, $\sim 3\times10^{21}$ FLOPs total — order 10k H100-hours.

**Control arm.** The same checkpoints scored under three metrics computed from identical sample sets: FID (Inception-V3, 50k), $\text{FD}_{\text{DINOv2}}$, and CMMD. Plus a seed-variance arm: 5 independent sample sets per checkpoint to establish metric noise.

**The deciding number.** Fit $Q(C) = E + A C^{-\gamma}$ separately per metric at the FID-optimal $w$ for each. Report $\gamma_{\text{FID}}$, $\gamma_{\text{DINO}}$, $\gamma_{\text{CMMD}}$ with bootstrap CIs. **If the three $\gamma$ values agree within overlapping 95% CIs**, the exponent is metric-invariant and the scaling law is a property of the model, not the metric — the problem moves from methodologically blocked to empirically open. **If they differ by more than 20% relative**, the published diffusion exponents describe Inception-V3, not sample quality, and every extrapolation built on them is void.

## 9. Key References

- **[Foundational]** Ho, Jain & Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. — arXiv:2006.11239
- **[Foundational]** Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV, 2023. — arXiv:2212.09748
- **[SOTA]** Karras, Aittala, Lehtinen, Hellsten, Aila & Laine. *Analyzing and Improving the Training Dynamics of Diffusion Models.* CVPR, 2024. — arXiv:2312.02696
- **[SOTA]** Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML, 2024. — arXiv:2403.03206
- **[SOTA]** Mei et al. *Bigger is not Always Better: Scaling Properties of Latent Diffusion Models.* TMLR, 2024. — arXiv:2404.01367
- **[Theory]** Chen, Chewi, Li, Li, Salim & Zhang. *Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions.* ICLR, 2023. — arXiv:2209.11215
- **[Theory]** Oko, Akiyama & Suzuki. *Diffusion Models are Minimax Optimal Distribution Estimators.* ICML, 2023. — arXiv:2303.01861
- **[Measurement]** Chong & Forsyth. *Effectively Unbiased FID and Inception Score and where to find them.* CVPR, 2020.
- **[Measurement]** Kynkäänniemi, Karras, Aittala, Aila & Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR, 2023. — arXiv:2203.06026
- **[Measurement]** Stein et al. *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.* NeurIPS, 2023.
- **[Measurement]** Jayasumana et al. *Rethinking FID: Towards a Better Evaluation Metric for Image Generation.* CVPR, 2024.
- **[Context]** Dhariwal & Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS, 2021. — arXiv:2105.05233
- **[Context]** Karras, Aittala, Aila & Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS, 2022. — arXiv:2206.00364

## 10. Worked Example

Take DiT at ImageNet-256, the one place with a clean published size ladder. Reported guided FID: DiT-S/2 $\approx 68$ (unguided), DiT-B/2 $\approx 43$, DiT-L/2 $\approx 23$, DiT-XL/2 $= 9.62$ unguided and $2.27$ at $w=1.5$. Backbone cost: S/2 $6$ Gflops, B/2 $23$, L/2 $80$, XL/2 $119$ per forward pass.

Fit $\mathrm{FID} = A\,C^{-\gamma}$ on the *unguided* points from B/2 to XL/2. Using $(23, 43)$ and $(119, 9.62)$:

$$\gamma = \frac{\log(43/9.62)}{\log(119/23)} = \frac{1.497}{1.643} \approx 0.91.$$

Extrapolating one order of magnitude, $C = 1190$ Gflops predicts $\mathrm{FID} \approx 9.62 \times 10^{-0.91} \approx 1.19$ — below the best unguided number anyone has ever reported at this resolution, and below the FID of *real held-out ImageNet images* against the training set at 50k samples, which sits near $1.5$–$2$ depending on the split. The fit predicts a value the metric cannot express.

Now the same points at $w=1.5$. XL/2 moves $9.62 \to 2.27$, a $4.2\times$ improvement from a single eval-time scalar — larger than the entire gain from L/2 to XL/2 ($23 \to 9.62$, $2.4\times$), and obtained at zero training FLOPs. The $x$-axis of the scaling law is training compute; the largest single lever on the $y$-axis is not on the $x$-axis at all.

That is the obstruction in one figure: an exponent estimated from four points, extrapolating past the metric's own floor, with a free parameter outside the law that dominates it. Nothing here is fixed by more GPUs. It is fixed by defining $Q$ first, then re-fitting.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*