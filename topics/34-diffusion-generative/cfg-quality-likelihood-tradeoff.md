---
id: 34-diffusion-generative/cfg-quality-likelihood-tradeoff
title: "Why Classifier-Free Guidance Improves Sample Quality but Hurts Likelihood"
topic: 34-diffusion-generative
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Why Classifier-Free Guidance Improves Sample Quality but Hurts Likelihood

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/cfg-quality-likelihood-tradeoff` · **Status:** open

## 1. Problem Statement

Classifier-free guidance (CFG) replaces a diffusion model's conditional score with an extrapolation away from the unconditional score. Empirically it is the single largest lever on perceptual sample quality in conditional image and video diffusion, and it is used in essentially every deployed text-to-image system. Yet the guided sampler no longer targets the learned data distribution $p_\theta(x\mid y)$: held-out likelihood degrades monotonically in guidance weight, distribution-level coverage falls, and the "distribution" being sampled has no closed form.

Three variants, of very different difficulty:

- **Theory.** Identify the distribution $p_w$ that the CFG sampler actually samples from, and prove why $\mathrm{FID}(p_w)$ can beat $\mathrm{FID}(p_{\theta})$ while $\mathbb{E}_{x\sim p_{\mathrm{data}}}[-\log p_w(x)]$ is strictly worse. Solving it means a theorem relating the two, not a heuristic.
- **Measurement.** Decide whether the quality gain is a real reduction in distributional distance or an artifact of Inception-feature metrics that reward mode-concentration. Solving it means a metric-independent demonstration either way.
- **Method.** Find a guidance-like operator that captures the quality gain *without* the likelihood loss — i.e. improves FID and precision at fixed or improved bits/dim.

## 2. Formal Setting

Data $x_0 \sim p_{\mathrm{data}}(\cdot \mid y)$, forward noising $x_t = \alpha_t x_0 + \sigma_t \epsilon$, $\epsilon\sim\mathcal N(0,I)$. A network estimates $\epsilon_\theta(x_t,t,y)$, equivalently the score $s_\theta(x_t,t,y) \approx \nabla_{x_t}\log p_t(x_t\mid y)$. CFG with weight $w \ge 0$ uses

$$\tilde s_w(x_t,t,y) \;=\; (1+w)\,s_\theta(x_t,t,y) \;-\; w\,s_\theta(x_t,t,\varnothing).$$

**Quantities as measured.**

- **Sample quality.** $\mathrm{FID}$ between 50 000 samples and the reference set, Inception-V3 pool3 features. Also precision/recall (Kynkäänniemi et al., 2019) at $k=3$ on the same features, and FD$_{\text{DINOv2}}$ as a backbone control.
- **Likelihood.** Not the training ELBO. The measurable quantity is the probability-flow ODE density: integrate $\dot x = f(x,t) - \tfrac12 g(t)^2 \tilde s_w(x,t,y)$ backwards with the instantaneous change-of-variables, giving
$$\log p_w(x_0\mid y) = \log p_T(x_T\mid y) + \int_0^T \nabla\!\cdot\! v_w(x_t,t)\,dt,$$
reported in bits/dim on held-out data with a fixed solver (e.g. RK45, tolerance $10^{-5}$) and Skilling–Hutchinson trace estimation with a stated number of probes. This is well defined for any $w$ because a pushforward of a smooth ODE always has a density — even when $\tilde s_w$ is **not** a gradient field.
- **Target of the naive story.** The folk claim is $p_w \propto p(x\mid y)\,\big(p(y\mid x)\big)^{w}$, i.e. a sharpened posterior. That object is a *static* tilt of the clean-data distribution; CFG applies the tilt at every noise level to the *noisy* marginals.

**Assumptions, and which fail.** (i) $s_\theta$ is exact — violated; the unconditional branch is typically trained on 10–20 % dropout of $y$ and is the weaker of the two heads. (ii) $\nabla\!\log p_t(x\mid y) - \nabla\!\log p_t(x)$ is the score of a valid likelihood term at every $t$ — violated: the diffused product $\int p_t(x\mid x_0)p(x_0)p(y\mid x_0)^w dx_0$ is not the tilt of the diffused marginal, so the two orderings of "tilt" and "diffuse" do not commute. (iii) $\tilde s_w$ is conservative — violated whenever the two heads have inconsistent curl, which they generically do at finite training.

## 3. State of the Art

**Established.**
- CFG (Ho & Salimans, NeurIPS 2021 workshop / arXiv:2207.12598) reproduces the fidelity/diversity trade-off of classifier guidance (Dhariwal & Nichol, NeurIPS 2021) without an auxiliary classifier. Reproduced by essentially every subsequent system.
- **Autoguidance** (Karras et al., NeurIPS 2024): guide with a *smaller, less-trained copy of the same model* instead of the unconditional head. ImageNet-512 FID 1.25 and ImageNet-64 FID 1.01, both records at publication, and, importantly, without the diversity collapse of CFG. This is the strongest evidence that CFG's gain comes from *cancelling shared model error*, not from posterior sharpening.
- **Interval guidance** (Kynkäänniemi et al., NeurIPS 2024): applying guidance only in a middle band of noise levels improves FID and FD$_{\text{DINOv2}}$ over guidance-everywhere at the same peak weight (e.g. EDM2-XXL ImageNet-512 FID $\approx 1.8 \to 1.4$).
- **CFG is not the tilted distribution.** Bradley & Nakkiran (2024) show CFG's update is a predictor–corrector scheme whose stationary target is *not* $p(x\mid y)p(y\mid x)^w$ except in the Gaussian case.
- Chidambaram et al. (NeurIPS 2024) give a fine-grained analysis in mixtures of Gaussians: guidance provably shrinks variance and moves mass toward class means, and can move samples *outside* the support region of high-density data for large $w$.

**Claimed but unablated.** That guidance "corrects for an imperfect score model" is stated widely; autoguidance supports it but does not isolate which error component (bias vs. variance vs. limited capacity) is cancelled. That CFG's FID U-curve reflects a genuine distributional optimum, rather than an Inception-feature artifact, is asserted routinely and tested rarely.

**Benchmark-number-only.** Nearly all reported "best" guidance weights ($w \approx 1.5$–$3$ for text-to-image, $\approx 0.3$–$1.0$ for EDM2-class models) exist only as FID minima on one dataset with one feature extractor. No paper reports a full bits/dim-vs-$w$ curve under the PF-ODE at a modern scale.

## 4. What Is Known

- **Fidelity up, coverage down.** Dhariwal & Nichol (ADM, ImageNet 256×256, 512 TPU-scale training): unguided FID 10.94 → guided 4.59; precision 0.69 → 0.82; recall 0.63 → 0.52. The recall drop is the coverage cost, measured on 50 k samples.
- **FID is U-shaped in $w$**; IS is monotone increasing. Consistently reproduced from 64×64 class-conditional up to 1024×1024 text-to-image.
- **Oversaturation at large $w$** is a systematic, reproducible artifact (Sadat et al., ICLR 2025, adaptive projected guidance): the guidance term's component parallel to the current denoised estimate inflates contrast; projecting it out removes the artifact at fixed $w$.
- **Where the gain lives in $t$.** Kynkäänniemi et al.: guidance at very high noise levels destroys diversity, at very low noise levels does nothing measurable; the useful band is intermediate. Measured on ImageNet-512 with EDM2.
- **Non-conservativity is measurable.** The Jacobian of $\tilde s_w$ is not symmetric for trained models; the asymmetry grows with $w$ since it scales the difference of two independently-parameterized fields.
- **Likelihood training and sample quality are already in tension without guidance:** Nichol & Dhariwal (ICML 2021) and Kingma & Gao (NeurIPS 2023) show that reweighting the diffusion loss toward the true ELBO improves bits/dim and worsens FID. CFG is the same trade-off pushed to the sampler.

## 5. What Is Not Known

- **Theoretically open.** No characterization of $p_w$ for non-Gaussian $p_{\mathrm{data}}$ with imperfect scores. No theorem stating conditions under which $D(p_{\mathrm{data}} \| p_w) < D(p_{\mathrm{data}}\|p_\theta)$ for any $f$-divergence — i.e. whether guidance ever genuinely reduces distributional distance, or only trades mass in a way FID rewards. No decomposition of $\tilde s_w$ into conservative + solenoidal parts with a bound on how much the solenoidal part contributes to the FID gain.
- **Empirically open.** The bits/dim-vs-$w$ curve under the PF-ODE, jointly with FID and FD$_{\text{DINOv2}}$, on a single modern model at ImageNet-512 scale. Runnable today; the compute is a few thousand GPU-hours. Nobody has published it.
- **Methodologically blocked.** "Sample quality" is operationalized by FID, which is known to reward ImageNet-class-consistent, low-diversity outputs (Kynkäänniemi et al., ICLR 2023; Stein et al., NeurIPS 2023). Until quality is measured by something that is not a Gaussian fit in a classifier's feature space, "CFG improves quality" and "CFG concentrates mass onto features FID likes" are not distinguishable statements.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**, not compute.

1. The two axes are measured in incomparable spaces: FID is a Gaussian distance in a supervised classifier's 2048-d features; bits/dim is an exact density on 786 k pixel dimensions dominated by high-frequency detail. A model can win one and lose the other for reasons having nothing to do with the phenomenon being studied.
2. Three distinct mechanisms predict the same observed FID gain: (a) posterior sharpening, (b) cancellation of shared score error, (c) a non-conservative drift that concentrates samples in high-Inception-density regions. Autoguidance argues for (b), Chidambaram et al. for (a), and nothing distinguishes (c) — the mechanisms are not identifiable from the FID-vs-$w$ curve alone.
3. There is no ground truth. For real image data the true $\log p_{\mathrm{data}}$ is unavailable, so "the guided model is worse" can only be stated relative to a held-out set under a specific solver, and solver/discretization error at $w>0$ is itself untabulated.

## 7. Current Research (as of 2026)

- **Bad-teacher and self-guidance variants.** Following autoguidance (NVIDIA, Karras/Aittala/Lehtinen), work on guiding with degraded, earlier-checkpoint, or intentionally-noised copies of the same network. Direction: make guidance an error-cancellation operator with a stated error model. *(frontier — verify specific 2026 follow-ups.)*
- **Schedule and geometry of the guidance term.** Interval guidance, adaptive projected guidance, and per-timestep learned weights; the shared claim is that scalar-$w$-everywhere is strictly suboptimal.
- **Sampler-theoretic accounts.** Predictor–corrector and annealed-Langevin framings (Bradley & Nakkiran; Nakkiran and collaborators) that describe CFG as a sampling dynamic rather than a change of target.
- **Guidance-free distillation.** Training a single network to reproduce guided outputs, removing the two-forward-pass cost; open question is whether the distilled student inherits the likelihood penalty or merely the samples. *(frontier — verify.)*
- **Metric replacement.** DINOv2- and CLIP-feature FDs, and human-preference evaluations, applied to the guidance sweep specifically.

## 8. Concrete Next Experiment

**The guidance Pareto sweep.**

- **Scale.** One publicly released EDM2-class model, ImageNet-512, XS and XXL sizes (so capacity is a controlled variable). $w \in \{0, 0.2, 0.5, 0.8, 1.2, 2.0, 4.0\}$.
- **Measure per $w$:** (i) FID and FD$_{\text{DINOv2}}$ on 50 k samples; (ii) precision/recall at $k=3$; (iii) **held-out bits/dim** on 10 k ImageNet validation images via the PF-ODE with the *guided* drift, RK45 tol $10^{-5}$, 20 Hutchinson probes, with the $w=0$ bits/dim reproduced against the published value as a solver sanity check.
- **Control arms.** (a) Autoguidance at matched FID — same quality, different mechanism. (b) A *temperature* control: unguided sampling with initial-noise scaled by $\tau<1$, which sharpens without any score extrapolation. (c) A **conservative projection** arm: replace $\tilde s_w$ with the nearest gradient field (least-squares fit of a scalar potential's gradient over sampled points, or symmetrized-Jacobian correction) — this isolates mechanism (c).
- **The deciding number.** $\Delta = \mathrm{BPD}(w^\star) - \mathrm{BPD}(0)$ at the FID-optimal $w^\star$, reported alongside $\Delta_{\mathrm{auto}}$ for autoguidance at matched FID. If $\Delta > 0$ (likelihood worse) while $\Delta_{\mathrm{auto}} \le 0$, the trade-off is an artifact of the *unconditional-head choice*, not intrinsic to guidance, and the method variant is solved. If both are $>0$ and of similar magnitude, the trade-off is intrinsic to the operator and the theory variant is the live one. Secondary: if the conservative-projection arm keeps most of the FID gain, mechanism (c) is refuted.

## 9. Key References

- **[Foundational]** Jonathan Ho, Tim Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021 Workshop on Deep Generative Models; arXiv:2207.12598.
- **[Foundational]** Prafulla Dhariwal, Alex Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021 — arXiv:2105.05233.
- **[SOTA]** Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, Samuli Laine. *Guiding a Diffusion Model with a Bad Version of Itself.* NeurIPS 2024 — arXiv:2406.02507.
- **[SOTA]** Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, Jaakko Lehtinen. *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models.* NeurIPS 2024 — arXiv:2404.07724.
- **[Theory]** Arwen Bradley, Preetum Nakkiran. *Classifier-Free Guidance is a Predictor-Corrector.* 2024 — arXiv:2408.09000.
- **[Theory]** Muthu Chidambaram, Khashayar Gatmiry, Sitan Chen, Holden Lee, Jianfeng Lu. *What does guidance do? A fine-grained analysis in a simple setting.* NeurIPS 2024.
- **[Method]** Seyedmorteza Sadat, Otmar Hilliges, Romann M. Weber. *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models.* ICLR 2025.
- **[Likelihood]** Yang Song, Conor Durkan, Iain Murray, Stefano Ermon. *Maximum Likelihood Training of Score-Based Diffusion Models.* NeurIPS 2021.
- **[Likelihood]** Diederik P. Kingma, Ruiqi Gao. *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.* NeurIPS 2023.
- **[Metric critique]** Tuomas Kynkäänniemi, Tero Karras, Miika Aittala, Timo Aila, Jaakko Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR 2023.
- **[Metric critique]** George Stein et al. *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.* NeurIPS 2023.

## 10. Worked Example

Take $p(x\mid y)$ a 1-D two-component Gaussian mixture, class $y=1$: $\tfrac12\mathcal N(-2,1) + \tfrac12\mathcal N(2,1)$, and the unconditional $p(x) = \mathcal N(0, 4)$ (variance 4, matching the conditional's total variance). At $t=0$:

$$\tilde s_w(x) = (1+w)\,\frac{d}{dx}\log p(x\mid 1) - w\,\frac{d}{dx}\log p(x) = (1+w)\frac{d}{dx}\log p(x\mid 1) + \frac{w x}{4}.$$

The added term $+wx/4$ is an *outward* push. At $w=1$ the stationary distribution of the corresponding Langevin dynamic has log-density $2\log p(x\mid 1) - \log \mathcal N(x;0,4) + C$. Its modes move outward from $\pm 2.00$ to about $\pm 2.4$, and the mass between the modes drops by roughly an order of magnitude. Two consequences, both real:

1. **A "quality" gain.** If a downstream feature detector simply asks "is $|x| > 1$" — the 1-D analogue of a classifier being confident — the guided distribution scores better: about 0.95 versus 0.89 of mass satisfies it.
2. **A likelihood loss.** Evaluate the guided density on samples from the *true* mixture. Points near $x=0$, which the true mixture emits with density $\approx 0.05$, are emitted by the guided density with density $\approx 0.006$. Averaged over the true mixture, the cross-entropy rises by roughly $0.1$ nat per sample. The guided model is strictly worse as a model of the data while looking better under the detector.

The obstruction is visible here in one line: the modes **moved**, from $\pm2.00$ to $\approx\pm2.4$. Guidance did not merely reweight the mixture components — it relocated mass to a place the data never occupies. In 1-D that is a 0.4-unit error one can plot. In 786 432 dimensions, the same relocation is invisible to FID, because the Inception features of a slightly-oversaturated, slightly-off-manifold image are *closer* to the class-conditional feature mean than a correct sample is. Any experiment that measures quality with FID alone will score the relocation as an improvement. That is why the deciding number in §8 must be bits/dim, evaluated under the guided drift, and not a feature-space distance.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*