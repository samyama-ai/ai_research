---
id: 34-diffusion-generative/guidance-weight-semantics
title: "Correct Probabilistic Interpretation of Guidance Weights Above One"
topic: 34-diffusion-generative
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Correct Probabilistic Interpretation of Guidance Weights Above One

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/guidance-weight-semantics` · **Status:** methodologically-blocked

## 1. Problem Statement

Classifier-free guidance (CFG) replaces the learned noise prediction with an extrapolation between conditional and unconditional predictions, controlled by a scalar $w$. Every practical text-to-image and text-to-video system ships with $w$ well above the value at which the extrapolation is a no-op. The folk justification is that CFG samples from the *tilted* density $p_w(x\mid c) \propto p(x)\,p(c\mid x)^{1+w}$. That justification is known to be wrong. No replacement exists.

Three variants, of very different difficulty:

- **Theory variant.** For a given sampler (DDIM, ancestral SDE, DPM-Solver) and guidance weight $w$, characterize the distribution $q_w$ the sampler actually produces. Solved would mean: a closed form or a variational characterization, plus a bound $D(q_w \,\|\, \text{target})$ that is tight in $w$.
- **Measurement variant.** Given only samples from a deployed model, estimate *how far* $q_w$ is from any coherent probabilistic target, and decompose the deviation into (a) mode sharpening within the data manifold and (b) mass placed off it. Currently blocked: there is no estimator of "distance from the tilted target" when the tilted target is not normalizable.
- **Method variant.** Design a sampler with a knob that has the semantics practitioners *believe* $w$ has — monotone control of fidelity-vs-diversity with a stated probabilistic target — at CFG's cost (two network evaluations per step, no MCMC inner loop).

## 2. Formal Setting

Data $x_0 \sim p_{\text{data}}(\cdot\mid c)$, condition $c$. The forward process gives $x_t = \alpha_t x_0 + \sigma_t \epsilon$, and $p_t$ denotes the marginal at time $t$. A network $\epsilon_\theta(x_t, t, c)$ is trained on $\mathbb{E}\|\epsilon_\theta - \epsilon\|^2$ with the condition dropped to a null token $\emptyset$ with probability $p_{\text{drop}}$ (typically $0.1$). At the optimum $\epsilon_\theta(x_t,t,c) = -\sigma_t \nabla_{x_t}\log p_t(x_t\mid c)$.

**Guidance.** Two conventions circulate and are routinely conflated:
$$\tilde\epsilon = (1+w)\,\epsilon_\theta(x_t,t,c) - w\,\epsilon_\theta(x_t,t,\emptyset), \qquad \tilde\epsilon = \epsilon_\theta(x_t,t,\emptyset) + s\,[\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\emptyset)]$$
with $s = 1+w$. "Guidance weight above one" means $s>1$, i.e. $w>0$; the harmful regime in practice is $s \gtrsim 5$. Stable Diffusion ships $s=7.5$; Imagen used $s$ up to $30$ with dynamic thresholding.

**The claimed target.** Writing $\nabla\log p_t(c\mid x_t) = \nabla\log p_t(x_t\mid c) - \nabla\log p_t(x_t)$, the guided score is $\nabla\log p_t(x_t) + s\,\nabla\log p_t(c\mid x_t)$, which is the score of
$$p_t^{(s)}(x) \;\propto\; p_t(x)\,p_t(c\mid x)^{s} \;=\; p_t(x\mid c)^{s}\,p_t(x)^{1-s}.$$

**Assumptions, and which fail.**
1. *$p_t^{(s)}$ is a probability density.* Fails whenever $p_t(x\mid c)^s p_t(x)^{1-s}$ is non-integrable — generic for $s>1$ wherever the conditional is locally flatter than the marginal (§10).
2. *The score of the time-$t$ marginal of the target equals the tilted score.* False: $\{p_t^{(s)}\}_t$ is **not** the diffusion path of $p_0^{(s)}$. Tilting and noising do not commute. This is the core error, made precise by Bradley & Nakkiran (2024).
3. *$\epsilon_\theta(\cdot,\emptyset)$ is the true unconditional score.* Fails in practice: the null branch is trained on $10\%$ of gradient steps and is systematically undertrained. What is measured as "$w$" therefore multiplies a model-error direction as well as a likelihood-ratio direction.
4. *Measured quantities.* FID, precision/recall (Kynkäänniemi et al., NeurIPS 2019), and CLIPScore are all computed on $50$k samples in a fixed embedding space; none of them estimates $D_{\mathrm{KL}}(q_w\|p^{(s)})$, and none is invariant to the mean-shift artifact guidance induces.

## 3. State of the Art

**Theory (established).** Bradley & Nakkiran (2024) prove CFG-DDIM does not sample $p^{(s)}$, and reinterpret CFG as a *predictor–corrector* scheme: a DDIM predictor on the conditional plus a Langevin-flavored corrector against a sharpened target, with the two steps annealing on mismatched schedules. Chidambaram et al. (NeurIPS 2024) analyze mixtures exactly and show guidance does not interpolate between conditional and tilted distributions; it drives mass toward the boundary of the conditional support and, in the $w\to\infty$ limit for their 1D two-point setting, to point masses. Wu et al. (ICML 2024) give Gaussian-mixture bounds: guidance monotonically reduces per-mode variance while improving classification accuracy of the generated sample, i.e. diversity loss is a theorem, not an artifact.

**Theory (constructive).** Feynman–Kac correctors (Skreta, Bradley, Nakkiran et al., 2025) give a sequential-Monte-Carlo weighting that provably targets the annealed/tilted family at finite compute — the first sampler with a *stated* target for $s>1$. Du et al. (ICML 2023) established the analogous point for compositional products: the summed score is not the product's score without MCMC correction.

**Empirical (established by ablation).** Restricting guidance to a middle noise interval improves FID with no loss in prompt adherence (Kynkäänniemi et al., NeurIPS 2024). Autoguidance — guiding with a smaller, undertrained copy of the same model rather than an unconditional one (Karras et al., NeurIPS 2024) — improves both FID and recall, which the tilted-distribution story cannot explain at all.

**Claimed but unablated.** That $s$ trades "diversity for quality" along a single axis; that oversaturation at high $s$ is a decoder/VAE artifact rather than a sampler artifact; that CFG++ (Chung et al., 2024) and adaptive projected guidance (Sadat et al., ICLR 2025) fix the *distributional* defect rather than its most visible symptom. These are supported by benchmark numbers and side-by-side grids, not by a distance to a defined target.

## 4. What Is Known

- Guidance improves FID at small $s$ and degrades it at large $s$, with a sharp optimum. Dhariwal & Nichol (NeurIPS 2021) report ImageNet-256 FID $4.59$ with classifier guidance at scale $1.0$; ADM-G quality collapses as scale grows.
- EDM2-XXL, ImageNet-512, $50$k samples: FID $1.81$ with standard CFG, $1.40$ with guidance restricted to a limited noise interval (Kynkäänniemi et al., 2024). Autoguidance reports record FIDs of $1.01$ (ImageNet-64) and $1.25$ (ImageNet-512) at the same scale.
- Precision rises and recall falls monotonically in $s$ over the useful range, measured at $50$k samples on ImageNet and COCO across many papers. The recall loss is the diversity theorem of Wu et al. showing up in a metric.
- CFG at $s\approx 3$ beat CLIP guidance on human preference in GLIDE (Nichol et al., ICML 2022) — the empirical result that made large $s$ standard, with no accompanying probabilistic account.
- Oversaturation and contrast blow-up at high $s$ are reproducible across architectures, samplers, and latent/pixel spaces; they are not VAE-specific.

## 5. What Is Not Known

- **Theoretically open.** A closed-form or variational characterization of $q_w$ for CFG-DDIM at $s>1$ in any setting richer than Gaussian mixtures. Whether *any* fixed target family $\{P_s\}$ exists for which CFG is a consistent sampler as step count $\to\infty$.
- **Theoretically open.** Whether the diversity loss at fixed prompt-fidelity is information-theoretically necessary for two-evaluation-per-step samplers, or an avoidable defect of the extrapolation form.
- **Methodologically blocked.** Measuring "how wrong CFG is" on real image models. When $p^{(s)}$ is non-normalizable the natural divergence is undefined; when it is normalizable it is uncomputable at image scale. There is no accepted surrogate that is not itself a fidelity metric with the same bias as the sampler.
- **Empirically open.** Whether the observed benefit of $s>1$ survives replacing the dropout-trained null branch with a separately trained, converged unconditional model. Runnable at ImageNet-512 for roughly the cost of one extra model train; not reported at that scale.

## 6. Why It Is Hard

The obstruction is **non-identifiability plus an evaluation that does not measure what it names**, compounding.

Non-identifiability: the guided score adds $s\,[\epsilon_c - \epsilon_\emptyset]$, and that vector is a sum of the true log-likelihood-ratio gradient and the difference of two model errors. Autoguidance is direct evidence the second term does useful work — deliberately degrading the guiding branch improves FID. So no experiment on a trained model can attribute the effect of $s$ to the density-tilting story without an oracle score, which does not exist for images.

Evaluation: FID is a Fréchet distance between Gaussian fits in Inception space. A sampler that shifts the mean toward the class centroid and shrinks covariance can lower FID while placing samples in regions of near-zero data density. The metric used to justify $s>1$ is exactly the metric that a mean-shift artifact games. Precision/recall splits the two directions but is not a divergence and has no calibrated zero.

## 7. Current Research (as of 2026)

- **Corrected samplers with stated targets.** SMC/Feynman–Kac weighting for guidance, annealing and products of experts (Skreta, Bradley, Nakkiran, and collaborators at Toronto/Vector and Apple). Open question is variance of the weights at image scale. *(frontier — verify)*
- **Guidance without an unconditional branch.** Autoguidance and its descendants (NVIDIA, Karras and colleagues); guidance from an earlier training checkpoint, a smaller model, or a noisier copy.
- **Schedule engineering.** Interval-limited guidance, $s$ ramps over $t$, and per-frequency guidance in video models. Widely deployed, largely justified by FID.
- **Fine-grained theory in tractable settings.** Mixtures, low-dimensional manifolds, and linear-Gaussian analyses (Chidambaram, Wu, and coauthors) mapping exactly which geometric feature guidance amplifies.
- **Distillation.** Guidance-distilled and reward-tuned few-step models fold $s$ into weights, which removes the knob and with it any hope of reading its semantics off the sampler. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** A synthetic ladder with exact ground truth: $d \in \{2, 8, 32\}$, data a $K=10$-component Gaussian mixture with per-component covariance chosen so that $p_0(x\mid c)$ is locally flatter than $p_0(x)$ on a known region. Here $p_t$, $\nabla\log p_t$, and $p_t^{(s)}$ are all closed-form, and normalizability of $p_0^{(s)}$ is checkable analytically. Train an MLP score net to within $1\%$ relative score error (measurable against the closed form). Sample $10^6$ points per configuration for $s \in \{1, 1.5, 2, 4, 8\}$, under CFG-DDIM and CFG-SDE.

**Control arm.** The Feynman–Kac / SMC-corrected sampler with the same network, same step count, matched to equal total network evaluations (fewer particles, more steps, or vice versa). This arm has a proved target, so its residual divergence measures estimator noise, not method error.

**Deciding number.** $D_{\mathrm{KL}}(q_s \,\|\, p^{(s)})$ in nats, estimated by importance sampling against the closed-form $p^{(s)}$ (valid only where $p^{(s)}$ is proper — run the ladder in the proper regime, and report the divergence of the normalizing constant separately in the improper regime). **The single number: $D_{\mathrm{KL}}(q_{s=4}\|p^{(4)})$ for CFG-DDIM divided by the same quantity for the corrected sampler at matched compute, in $d=32$.** A ratio below $2$ means the tilted-distribution reading is a usable approximation and the field's intuition is salvageable. A ratio above $10$, growing in $s$, means CFG at $s>1$ has no distributional semantics and the knob must be redefined operationally.

## 9. Key References

- **[Foundational]** Jonathan Ho, Tim Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications. — arXiv:2207.12598
- **[Foundational]** Prafulla Dhariwal, Alex Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021. — arXiv:2105.05233
- **[SOTA / theory]** Arwen Bradley, Preetum Nakkiran. *Classifier-Free Guidance is a Predictor-Corrector.* 2024. — arXiv:2408.09000
- **[SOTA / theory]** Muthu Chidambaram, Khashayar Gatmiry, Sitan Chen, Holden Lee, Jianfeng Lu. *What does guidance do? A fine-grained analysis in a simple setting.* NeurIPS 2024. — arXiv:2409.13074
- **[Theory]** Yuchen Wu, Minshuo Chen, Zihao Li, Mengdi Wang, Yuting Wei. *Theoretical Insights for Diffusion Guidance: A Case Study for Gaussian Mixture Models.* ICML 2024. — arXiv:2403.01639
- **[SOTA / empirical]** Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, Samuli Laine. *Guiding a Diffusion Model with a Bad Version of Itself.* NeurIPS 2024. — arXiv:2406.02507
- **[SOTA / empirical]** Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, Jaakko Lehtinen. *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models.* NeurIPS 2024. — arXiv:2404.07724
- **[Method]** Marta Skreta, Tara Akhound-Sadegh, Arwen Bradley, Preetum Nakkiran, et al. *Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts.* 2025. — arXiv:2503.02819
- **[Method]** Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, Will Grathwohl. *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC.* ICML 2023. — arXiv:2302.11552
- **[Method]** Seyedmorteza Sadat, Otmar Hilliges, Romann M. Weber. *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models.* ICLR 2025. — arXiv:2410.02416
- **[Measurement]** Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila. *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS 2019. — arXiv:1904.06991

## 10. Worked Example

One dimension, everything exact. Let the conditional be $p(x\mid c) = \mathcal{N}(1, 2)$ and the marginal $p(x) = \mathcal{N}(0, 1)$ — the conditional is *broader*, which happens whenever the null branch is undertrained and over-confident, and happens locally in real models wherever a caption admits more variation than the model's null prediction.

The tilted target has log-density $(1+w)\log p(x\mid c) - w \log p(x)$, with quadratic coefficient
$$-\tfrac12\left[\frac{s}{\sigma_c^2} - \frac{s-1}{\sigma_0^2}\right] = -\tfrac12\left[\frac{s}{2} - (s-1)\right] = -\tfrac12\left[1 - \tfrac{s}{2}\right].$$
For $s < 2$ this is negative and $p^{(s)}$ is a proper Gaussian. At $s = 2$ it is zero: the target is flat on $\mathbb{R}$, mass infinite. For $s > 2$ it is positive: the "density" grows without bound in both tails and $p^{(s)}$ **does not exist**.

The sampler does not notice. At $s=4$ the guided score is a perfectly well-behaved affine function of $x$, the DDIM update runs, and out comes a finite sample. Every image system operates in this regime — Stable Diffusion at $s=7.5$ — and the object those samples are supposedly drawn from is not a distribution.

Take the well-posed case for a magnitude check: equal variances $\sigma_c^2=\sigma_0^2=1$, $\mu_c = 1$, $\mu_0 = 0$. Then $p^{(s)} = \mathcal{N}(s, 1)$ — precision is unchanged, the mean is extrapolated. At $s=5$ the target sits at $x=5$, where the true conditional density is $\phi(4) = 1.34\times10^{-4}$ and $P_{p(\cdot\mid c)}(X > 4.5) = P(Z > 2.47) \approx 6.8\times10^{-3}$. So the sampler concentrates essentially all of its mass in a region holding under $1\%$ of the true conditional probability — and a Fréchet-style metric that compares only first and second moments will score this pathology as a modest mean shift, not as a support failure.

That is the obstruction in miniature: for $s>1$ the intended target either does not exist or lies off the data, the sampler is indifferent either way, and the standard metrics cannot tell the two apart.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*