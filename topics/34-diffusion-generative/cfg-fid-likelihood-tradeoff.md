---
id: 34-diffusion-generative/cfg-fid-likelihood-tradeoff
title: "Why Classifier-Free Guidance Improves FID While Hurting Likelihood"
topic: 34-diffusion-generative
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Why Classifier-Free Guidance Improves FID While Hurting Likelihood

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/cfg-fid-likelihood-tradeoff` · **Status:** open

## 1. Problem Statement

Classifier-free guidance (CFG) replaces the learned conditional score with an extrapolation away from the unconditional score. Every large image and video diffusion model ships with it, at weights $w \in [1, 8]$. It reliably improves FID and Inception Score, and it reliably makes the sampled distribution a *worse* model of the data: less diverse, and — where anyone has measured it — lower held-out likelihood.

Three distinct variants are usually conflated:

- **Measurement.** Does guidance actually reduce the held-out log-likelihood of the sampling distribution, and by how much, as a function of $w$? The guided sampler is a valid ODE flow, so this number exists and is computable. It is almost never reported.
- **Method.** Can one obtain the FID gain without the density loss — i.e. is the gain attributable to a *correctable* model defect rather than to distribution-tilting? Autoguidance (Karras et al., 2024) is evidence that at least part of it is.
- **Theory.** What functional is guidance optimising? CFG does not sample from $p(x\mid c)^{1+w}p(x)^{-w}$ except in degenerate cases; there is no known variational or divergence-minimisation characterisation of the distribution it does sample from.

Solving the page means: a closed-form or provably-tight characterisation of the guided marginal, plus a decomposition of the FID gain into (i) correction of model error and (ii) mode-sharpening that a perfect model would not need.

## 2. Formal Setting

Forward process $x_t = \alpha_t x_0 + \sigma_t \varepsilon$, $\varepsilon\sim\mathcal N(0,I)$, with marginals $p_t$. A network $s_\theta(x_t,t,c) \approx \nabla_{x}\log p_t(x\mid c)$ is trained with label dropout so that $c=\varnothing$ gives $\nabla_x \log p_t(x)$. The guided field is

$$\tilde s_w(x,t,c) = (1+w)\, s_\theta(x,t,c) - w\, s_\theta(x,t,\varnothing).$$

**Measured quantities.**

- **Guided density $p^w_\theta$.** Run the probability-flow ODE $\dot x = f(x,t) - \tfrac12 g(t)^2 \tilde s_w(x,t,c)$ backwards. Any ODE flow induces a normalised density by instantaneous change of variables, so
  $$\log p^w_\theta(x_0\mid c) = \log p_T(x_T) + \int_0^T \nabla\!\cdot\! v_w(x_t,t)\,dt,$$
  divergence estimated by Hutchinson trace, reported in bits/dim on held-out data. **This holds even though $\tilde s_w$ is not a gradient field** — non-conservativity blocks the interpretation of $\tilde s_w$ as a score, not the likelihood computation.
- **FID.** $\|\mu_r-\mu_g\|^2 + \mathrm{tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ on Inception-V3 pool3 features, 50k samples.
- **Precision / recall** (Kynkäänniemi et al., 2019) on the same features, $k=3$.

**Assumptions, and which are violated.**

1. *Tilting identity:* $\tilde s_w = \nabla\log\big(p_t(x\mid c)p_t(c\mid x)^w\big)$ — **holds at each $t$ pointwise, but does not imply the sampled terminal marginal equals the $t{=}0$ tilted density**, because the tilted marginals do not form a consistent diffusion family. Violated in practice; this is the central error in the folk explanation.
2. *Learned scores are exact.* Violated — and autoguidance's success suggests the violation is where the FID gain lives.
3. *Inception features measure image quality.* Violated: FID is strongly driven by ImageNet-class alignment (Kynkäänniemi et al., ICLR 2023).

## 3. State of the Art

**Established.** Ho & Salimans (NeurIPS 2021 workshop; arXiv:2207.12598) introduced CFG and showed the IS/FID *tradeoff curve* in $w$: small $w$ minimises FID, large $w$ maximises IS, and diversity falls monotonically. Dhariwal & Nichol (NeurIPS 2021) showed the same for classifier guidance, with explicit precision/recall numbers. Bradley & Nakkiran (arXiv:2408.09000) prove CFG is *not* annealed sampling from the tilted density and reinterpret it as a predictor–corrector scheme: DDIM prediction on $p(x\mid c)$ plus Langevin correction on $p(x\mid c)^{\gamma}$. Chidambaram et al. (NeurIPS 2024) and Wu et al. (ICML 2024) give exact analyses in Gaussian-mixture settings: guidance shrinks conditional variance and can *move* the mode away from the true conditional mean.

**Claimed but unablated.** The statement "CFG trades likelihood for sample quality" is near-universal in the literature and rests on almost no direct measurement — most guided models report no NLL at all, because guidance is applied at sampling time and the training NLL bound is unaffected. The quantity in §2 is well defined but is essentially unreported at ImageNet scale.

**Benchmark-number-only results.** Karras et al., *Guiding a Diffusion Model with a Bad Version of Itself* (NeurIPS 2024): guiding with a smaller/undertrained copy of the same model instead of the unconditional model gives ImageNet FID 1.01 at $64\times64$ and 1.25 at $512\times512$, with diversity preserved. Kynkäänniemi et al. (NeurIPS 2024): applying guidance only on a middle noise interval improves EDM2 ImageNet-512 FID from ≈1.8 to ≈1.4. Both are FID/recall numbers; neither reports guided likelihood.

## 4. What Is Known

- **Magnitude of the FID gain.** ADM on ImageNet $256\times256$: FID 10.94 unguided → 4.59 with classifier guidance (Dhariwal & Nichol, NeurIPS 2021), a 58% reduction, at 400M+ params.
- **It is not monotone in $w$.** FID is U-shaped: minimised near $w\approx 0.1$–$0.3$ in the original CFG ImageNet-64/128 experiments (best reported FID 1.55 at $64\times64$), while IS keeps rising past $w=4$.
- **Diversity cost is real and measured.** Guidance raises precision and lowers recall in every reported ablation from 2021 onward; the guided distribution demonstrably drops modes.
- **The tilted-density story is false.** Bradley & Nakkiran, 2024, and the Gaussian analyses of Chidambaram et al., 2024 — closed-form, not empirical.
- **The gain is partly a model-error correction.** Autoguidance (Karras et al., 2024) obtains a *larger* FID gain with *less* diversity loss by removing the class-tilt entirely, at ImageNet-512, EDM2-XXL scale.
- **FID and likelihood are near-independent in high dimension** (Theis, van den Oord & Bethge, ICLR 2016). A model can be excellent on one and poor on the other; the sign of the CFG effect on each is therefore not automatically informative.

## 5. What Is Not Known

- **Empirically open.** The $w \mapsto (\text{FID}, \text{recall}, \text{bits/dim of } p^w_\theta)$ curve on ImageNet-512 with EDM2. Runnable today: Hutchinson-trace likelihood on the guided ODE costs ~$10^2$ NFE per image over ~2k held-out images. Nobody has published it.
- **Empirically open.** Does the FID gain vanish as the base model approaches the true score? Requires a family of models along a compute axis with matched samplers. Autoguidance is suggestive, not decisive.
- **Theoretically open.** Is there any divergence $D$ such that guided sampling decreases $D(p_{\text{data}}\,\|\,\cdot)$ relative to unguided? No proof either way. Known negative results only rule out the tilted-density characterisation.
- **Methodologically blocked.** "Sample quality" as distinct from "class-alignment" has no agreed measurement. FID moves with ImageNet-class prototypicality (Kynkäänniemi et al., ICLR 2023); CMMD (Jayasumana et al., CVPR 2024) and the metric audit of Stein et al. (NeurIPS 2023) show FID's ranking is unstable. Until quality has a metric that guidance does not directly game, "improves quality, hurts likelihood" is not falsifiable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**. FID's largest sensitivity is to the alignment of samples with Inception's ImageNet classes — exactly the axis CFG pushes on. So the observed gain has at least three inseparable sources: real removal of off-manifold samples, deliberate mode-sharpening, and metric-gaming of the feature extractor. No held-out ground truth distinguishes them, because we do not have samples from a "correctly-modelled but sharper" reference to compare against. Second, the guided field is non-conservative: there is no $\log \tilde p$ whose gradient it is, so the usual toolkit (ELBO, KL decomposition, score matching identities) does not apply, and one is forced into ODE-flow likelihoods that cost ~100× a normal sample to estimate.

## 7. Current Research (as of 2026)

- **Guidance without class-tilt.** Autoguidance and its descendants (NVIDIA, Karras/Aittala/Laine). The claim under test: the useful component of CFG is a self-correction, not a conditional sharpening. *(frontier — verify: replications outside NVIDIA at ImageNet-512 scale.)*
- **Interval and schedule guidance.** Kynkäänniemi et al. 2024; Wang et al., *Analysis of Classifier-Free Guidance Weight Schedulers* (TMLR 2024). Established that $w$ should be time-varying; the *reason* is still empirical.
- **Sampler-theoretic accounts.** Bradley & Nakkiran's predictor–corrector view; ongoing work on when the Langevin correction converges to a well-defined stationary distribution.
- **Training-free / degraded-model guidance.** Sadat et al., *No Training, No Problem* (2024), independent condition-free guidance variants.
- **Metric replacement.** CMMD, DINOv2-feature FD (Stein et al.), and human-preference studies as the arbiter of whether the "quality gain" survives changing the feature space. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Scale.** EDM2-S on ImageNet-512 (public weights, ~280M params), one training run reused for all arms.

**Arms.** Sweep $w \in \{0, 0.3, 1, 2, 4\}$ for (a) standard CFG, (b) autoguidance with a 4×-undertrained copy of the same net, (c) **control arm: temperature sampling** — scale the terminal ODE noise / apply a matched-precision truncation with no guidance, tuned to hit the *same* precision as each CFG arm. The control isolates "sharpening" from "correction".

**Measurements per arm.** FID-50k; recall@$k{=}3$; and $\mathrm{NLL}_w = -\mathbb E_{x\sim\text{val}}\log p^w_\theta(x\mid c)$ in bits/dim, via the guided probability-flow ODE with Hutchinson trace, 2,000 held-out images, 5 noise seeds (report the standard error — it must be below 0.005 bits/dim to be usable).

**Deciding number.** $\Delta\mathrm{FID}$ per bit/dim of likelihood lost, $\rho = \frac{\mathrm{FID}_0 - \mathrm{FID}_w}{\mathrm{NLL}_w - \mathrm{NLL}_0}$, compared between CFG and the precision-matched control. If $\rho_{\text{CFG}} \approx \rho_{\text{control}}$, guidance is pure sharpening and the page collapses to "FID rewards low-entropy samplers". If $\rho_{\text{CFG}} \gg \rho_{\text{control}}$ (say $>2\times$), guidance is doing something a temperature knob cannot, and the correction hypothesis survives.

## 9. Key References

- **[Foundational]** Jonathan Ho, Tim Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021 Workshop on Deep Generative Models; arXiv:2207.12598.
- **[Foundational]** Prafulla Dhariwal, Alex Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021 — arXiv:2105.05233.
- **[Foundational]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021 — arXiv:2011.13456.
- **[Theory SOTA]** Arwen Bradley, Preetum Nakkiran. *Classifier-Free Guidance is a Predictor-Corrector.* 2024 — arXiv:2408.09000.
- **[Theory SOTA]** Muthu Chidambaram, Khashayar Gatmiry, Sitan Chen, Holden Lee, Jianfeng Lu. *What does guidance do? A fine-grained analysis in a simple setting.* NeurIPS 2024.
- **[Theory]** Yuchen Wu, Minshuo Chen, Zihao Li, Mengdi Wang, Yuting Wei. *Theoretical Insights for Diffusion Guidance: A Case Study for Gaussian Mixture Models.* ICML 2024.
- **[SOTA]** Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, Samuli Laine. *Guiding a Diffusion Model with a Bad Version of Itself.* NeurIPS 2024 — arXiv:2406.02507.
- **[SOTA]** Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, Jaakko Lehtinen. *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models.* NeurIPS 2024 — arXiv:2404.07724.
- **[Measurement]** Lucas Theis, Aäron van den Oord, Matthias Bethge. *A note on the evaluation of generative models.* ICLR 2016 — arXiv:1511.01844.
- **[Measurement]** Tuomas Kynkäänniemi, Tero Karras, Miika Aittala, Timo Aila, Jaakko Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR 2023 — arXiv:2203.06026.
- **[Survey/Audit]** George Stein et al. *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.* NeurIPS 2023.

## 10. Worked Example

Take a 1-D binary-class model where every quantity is exact. $p(x\mid c{=}{+}1)=\mathcal N(1,1)$, $p(x\mid c{=}{-}1)=\mathcal N(-1,1)$, balanced prior, so $\nabla\log p(x) = -x+\tanh x$. At $t=0$ the guided field is

$$\tilde s_w(x) = -(1+w)(x-1) + w(x - \tanh x) = -x + (1+w) - w\tanh x,$$

which here *is* a gradient, so the guided density is exactly

$$\tilde p_w(x) \propto \exp\!\big(-\tfrac{x^2}{2} + (1+w)x\big)\,(\cosh x)^{-w}.$$

At $w=0$ this is the true $\mathcal N(1,1)$; held-out NLL is the Gaussian entropy $1.4189$ nats. At $w=1$, numerical quadrature (step $0.5$ on $[-2,5]$) gives $Z=6.406$, $\log Z = 1.857$, and $\mathbb E_{x\sim\mathcal N(1,1)}[\log \cosh x] = 0.634$, hence

$$\mathrm{NLL}_{w=1} = 1.857 - (-1 + 2 - 0.634) = 1.491 \text{ nats} = +0.104 \text{ bits worse}.$$

Meanwhile the "wrong-side" mass — the natural quality/classifier proxy, the analogue of Inception Score — falls from $P(x<0)=0.159$ to $0.066$: a 58% reduction, the same order as ADM-G's 58% FID reduction on ImageNet-256.

**Where the obstruction becomes visible.** The model here is *exact*: the score is the true score, so there is no model error to correct, and yet the quality proxy improves by 58%. Every nat of that improvement is bought by moving away from $p_{\text{data}}$. If a real ImageNet model shows the same 58%/0.1-bit exchange rate, guidance is doing nothing but sharpening — and FID is simply rewarding a lower-entropy sampler. Only a precision-matched control arm can tell the two cases apart, and that number has never been published.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*