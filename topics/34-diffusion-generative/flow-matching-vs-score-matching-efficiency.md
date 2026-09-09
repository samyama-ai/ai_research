---
id: 34-diffusion-generative/flow-matching-vs-score-matching-efficiency
title: "Flow Matching versus Score Matching Sample Efficiency"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Flow Matching versus Score Matching Sample Efficiency

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/flow-matching-vs-score-matching-efficiency` · **Status:** empirically-open

## 1. Problem Statement

Flow matching (FM) and denoising score matching (DSM) are trained with the same architecture, the same Gaussian corruption family, and the same data. Practitioners report that FM-style objectives — conditional flow matching, rectified flow, stochastic interpolants — reach a given sample quality with fewer gradient steps and fewer sampling steps. The question is whether that advantage is **intrinsic to the objective** or an artifact of three confounds bundled with it: the interpolation path, the loss weighting over noise levels, and the network output parameterization.

Three variants, different difficulty:

- **Measurement.** Given a fixed architecture, fixed data budget $N$, and fixed compute $C$, does FM attain lower FID / higher likelihood than DSM once the path, weighting, and parameterization are matched? Runnable today; the matching is what nobody does carefully.
- **Method.** Does there exist a weighting/parameterization of DSM that recovers *all* of FM's empirical advantage? If yes, "flow matching is better" is a statement about defaults, not about objectives.
- **Theory.** Do the two estimators have different variance or different sample complexity for the same target regression function, under any nontrivial data assumption?

Solving it means: a statement of the form "at scale $S$, matched on $(\text{path}, w, \text{param})$, the FM–DSM gap is $\Delta \pm \epsilon$ with $\epsilon < \Delta/3$", plus an identification of which of the three factors carries the gap.

## 2. Formal Setting

Data $x_0 \sim q$ on $\mathbb{R}^d$. Gaussian probability path indexed by $t \in [0,1]$ with schedule $(\alpha_t, \sigma_t)$:

$$x_t = \alpha_t x_0 + \sigma_t \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).$$

The marginal $p_t$ has score $s_t(x) = \nabla_x \log p_t(x)$ and marginal velocity $u_t(x) = \mathbb{E}[\dot\alpha_t x_0 + \dot\sigma_t \epsilon \mid x_t = x]$. These are **affinely related, pointwise**:

$$u_t(x) = \frac{\dot\alpha_t}{\alpha_t} x + \left(\dot\sigma_t - \frac{\dot\alpha_t \sigma_t}{\alpha_t}\right)\sigma_t\, s_t(x).$$

So FM and DSM regress the *same* conditional expectation up to a known affine map. Both losses are instances of

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \pi}\, \mathbb{E}_{x_0, \epsilon} \left[ w(t)\, \big\| F_\theta(x_t, t) - \tau_t(x_0, \epsilon) \big\|^2 \right],$$

with three free choices: the **path** $(\alpha_t,\sigma_t)$ (VP: $\alpha_t^2+\sigma_t^2=1$; linear/rectified: $\alpha_t = 1-t$, $\sigma_t = t$), the **weighting** $w(t)$ together with the timestep density $\pi$, and the **target/parameterization** $\tau_t \in \{\epsilon,\; x_0,\; v = \dot\alpha_t x_0 + \dot\sigma_t\epsilon\}$. "FM vs DSM" as usually run varies all three at once.

Measured quantities, as they would actually be logged:

- **Sample efficiency:** FID-50K as a function of training samples seen $N = B \cdot T_{\text{steps}}$, and of PF-ODE function evaluations (NFE) at sampling. Report the pair $(N, \text{NFE})$, never one alone.
- **Likelihood:** bits/dim via the instantaneous-change-of-variables integral for the ODE, or the diffusion ELBO. Kingma & Gao (NeurIPS 2023) show every $w(t)$ that is monotonic in log-SNR $\lambda_t = 2\log(\alpha_t/\sigma_t)$ gives an ELBO on a noise-augmented density — so likelihood and FID rank objectives differently by construction.
- **Estimator variance:** $\mathrm{Var}[\nabla_\theta \ell]$ per minibatch at fixed $\theta$, the only quantity that could make one objective intrinsically more sample-efficient at equal target.

Assumptions and their violations. (i) *Both models are trained to the same optimization quality* — violated: learning-rate and EMA schedules are typically tuned for one arm. (ii) *FID measures sample quality* — violated: FID is Inception-feature-Gaussian and sensitive to sampler discretization, so it partly scores the ODE solver, not the learned field. (iii) *Same effective timestep density* — violated: rectified-flow training with logit-normal $t$-sampling (SD3) implies a $\pi$ that has no counterpart in default DDPM uniform-$t$ training. (iv) *Same network capacity across noise levels* — violated: the $\epsilon$- and $v$-parameterizations rescale the target by $\sigma_t$-dependent factors, changing the effective per-$t$ loss scale by orders of magnitude at the endpoints.

## 3. State of the Art

**Theory SOTA (established).** The affine identity above is exact; Albergo, Boffi & Vanden-Eijnden's stochastic-interpolants framework (ICLR 2023 / 2023 monograph) and Lipman et al.'s *Flow Matching Guide and Code* (2024) both state FM and DSM as one family. Kingma & Gao (NeurIPS 2023) unify diffusion objectives as weighted ELBOs, reducing "which objective" to "which $w(\lambda)$". Convergence theory for the samplers (Chen et al., ICLR 2023; Benton et al., ICLR 2024, $\tilde O(d)$ iteration complexity) is stated in score error and transfers to velocity error through the same affine map. **No theory distinguishes the two on estimator variance or sample complexity.**

**Empirical SOTA (established, ablated).** SiT (Ma et al., ECCV 2024) is the cleanest controlled study: identical DiT backbone, ImageNet 256×256, stepping DDPM → interpolant/FM one factor at a time. SiT-XL/2 reaches FID-50K 2.06 with classifier-free guidance against DiT-XL/2's 2.27 at matched architecture and training length. The paper attributes the gain to the *choice of interpolant and to decoupling the training path from the sampling diffusion coefficient* — not to "flow matching" as such.

**Claimed but unablated.** Scaling Rectified Flow Transformers (Esser et al., ICML 2024, SD3) reports rectified flow with logit-normal timestep sampling beating 61 alternative formulations at text-to-image scale — but the comparison ships path, weighting and $t$-density jointly, and the ranking is a benchmark table, not a factorized ablation. Widespread claims that FM needs fewer sampling steps are largely benchmark numbers under different solvers; EDM (Karras et al., NeurIPS 2022) reaches CIFAR-10 conditional FID 1.79 at 35 NFE with a *score*-based model and a tuned $\sigma$-schedule, which is not worse than typical FM few-step results at that scale.

## 4. What Is Known

- **Exact reducibility.** Any FM model on a Gaussian path yields a score, and vice versa, with no retraining. Established analytically.
- **Weighting dominates.** Kingma & Gao show $\epsilon$-, $x_0$- and $v$-prediction losses differ only by $w(\lambda)$; their monotonic-weighting family recovers or exceeds prior CIFAR-10 and ImageNet-64 results without changing the "objective family". Measured at ImageNet 64×64 and CIFAR-10 scale.
- **Path matters more than name.** SiT's ablation ladder moves FID-50K on ImageNet 256 from DiT-XL/2's 9.62 (no guidance) to 8.61 for SiT-XL/2 — a ~10% relative change, at 7M training iterations, batch 256. That is the size of the honest matched-architecture gap at this scale, and it is small relative to the 2× swings from guidance or sampler choice.
- **Tuned score models remain competitive.** EDM: CIFAR-10 FID 1.79 (conditional), 1.97 (unconditional), 35 NFE — a preconditioned DSM model, not FM.
- **Straightness is trainable, not automatic.** Rectified flow (Liu, Gong & Liu, ICLR 2023) achieves near-one-step sampling only after *reflow* on self-generated pairs; the base FM model is not straight. Lee et al. (NeurIPS 2024) show reflow quality is bottlenecked by the teacher's ODE error.

## 5. What Is Not Known

- **Theoretically open.** Whether the FM and DSM gradient estimators differ in variance at equal parameterization and weighting. No proof either way; the affine identity constrains the targets but not the estimator second moments once $w$ and $\pi$ differ.
- **Theoretically open.** Whether any $w(\lambda)$ for DSM can match the *best* FM configuration for FID (as opposed to likelihood, where the ELBO view says yes for monotonic $w$).
- **Empirically open.** The 3-factor factorial — path × weighting × parameterization, all other things fixed, at ≥1B-parameter text-to-image scale. Runnable; nobody has published it. SiT does it at 675M on class-conditional ImageNet only.
- **Methodologically blocked.** "Sample efficiency" has no agreed definition here. Samples-to-target-FID, compute-to-target-FID, and NFE-to-target-FID rank the methods differently, and FID itself confounds solver discretization with field quality.

## 6. Why It Is Hard

The obstruction is **non-identifiability under a confounded measurement**, not compute. Because $u_t$ and $s_t$ are affinely equivalent, any performance difference must come from $(\text{path}, w, \pi, \text{param})$ — a 4-dimensional design space in which the two communities' defaults sit at different, individually-tuned points. Comparing "FM vs DSM" therefore compares two *tuning traditions*. Worse, the readout is FID under a sampler: change ODE solver or step count and the ranking moves by more than the objective effect (~10% relative at ImageNet 256). A factorial that isolates the objective needs per-cell LR/EMA retuning, or the result measures optimizer luck. That is what makes the experiment expensive — not the FLOPs of one run.

## 7. Current Research (as of 2026)

- **Unification and pedagogy.** Lipman, Chen, Ben-Hamu and collaborators (Meta AI) continue the flow-matching framework, extending to non-Gaussian and discrete paths where the affine reduction to a score does *not* hold — the one regime where the objectives are genuinely distinct.
- **Weighting/schedule search.** Kingma-style weighted-ELBO analysis and SD3's logit-normal $t$-sampling are converging on "the schedule is the hyperparameter"; resolution-dependent timestep shifting is now standard in large text-to-image and video models. *(frontier — verify)* Several 2025–2026 systems reports treat shift as the primary knob and the FM/DSM label as incidental.
- **Few-step distillation.** Consistency-style and reflow-style distillation increasingly dominate the NFE axis, which weakens the "FM samples in fewer steps" claim as a discriminator between objectives. *(frontier — verify)*
- **Interpolant theory.** Albergo/Vanden-Eijnden (NYU/Flatiron) on interpolant design and the SDE-vs-ODE sampling trade-off.

## 8. Concrete Next Experiment

**Scale.** DiT-B/2 (≈130M params), ImageNet 256×256 latents (SD-VAE), batch 256, 400K steps per cell — roughly 8 A100-days per cell.

**Design.** Full $2\times2\times2$ factorial, 8 cells: path $\in$ {VP cosine, linear/rectified} × weighting $\in$ {uniform-$t$ $\epsilon$-loss, logit-normal $t$ with FM weight} × target $\in$ {$\epsilon$, $v$}. Every cell gets an identical 3-point learning-rate sweep; report the best per cell so the result is not optimizer luck. All cells evaluated with the *same* Heun ODE solver at NFE $\in$ {10, 25, 50, 250}, converting each model to a common score field via the affine identity before sampling — so the sampler is byte-identical across arms.

**Control arm.** The DiT-B/2 DDPM cell (VP cosine, uniform-$t$, $\epsilon$) at the published recipe; its FID-50K must reproduce the literature value within 5% or the harness is wrong.

**Deciding number.** $\Delta_{\text{obj}}$ = the main effect of the *path* factor on FID-50K at NFE 250, marginalized over weighting and target, with a 3-seed bootstrap CI. If $|\Delta_{\text{obj}}| < 0.3$ FID while the weighting main effect exceeds 1.0 FID, the FM advantage is a weighting/schedule effect and the objective distinction is empty at this scale. If $\Delta_{\text{obj}} > 1.0$ FID with the CI excluding zero, the path is doing real work and the question becomes *why*.

## 9. Key References

- **[Foundational]** Yang Song, Stefano Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS, 2019. — arXiv:1907.05600
- **[Foundational]** Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. — arXiv:2006.11239
- **[Foundational]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR, 2021. — arXiv:2011.13456
- **[Foundational]** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. *Flow Matching for Generative Modeling.* ICLR, 2023. — arXiv:2210.02747
- **[Foundational]** Xingchao Liu, Chengyue Gong, Qiang Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR, 2023. — arXiv:2209.03003
- **[Foundational]** Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden. *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.* 2023. — arXiv:2303.08797
- **[SOTA]** Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, Saining Xie. *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers.* ECCV, 2024. — arXiv:2401.08740
- **[SOTA]** Patrick Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML, 2024. — arXiv:2403.03206
- **[SOTA]** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS, 2022. — arXiv:2206.00364
- **[Analysis]** Diederik P. Kingma, Ruiqi Gao. *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.* NeurIPS, 2023.
- **[Theory]** Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, Anru R. Zhang. *Sampling is as Easy as Learning the Score.* ICLR, 2023. — arXiv:2209.11215
- **[Theory]** Joe Benton, Valentin De Bortoli, Arnaud Doucet, George Deligiannidis. *Nearly $d$-Linear Convergence Bounds for Diffusion Models via Stochastic Localization.* ICLR, 2024.
- **[Survey]** Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, Itai Gat. *Flow Matching Guide and Code.* 2024. — arXiv:2412.06264

## 10. Worked Example

Take a 1-D Gaussian data model, $q = \mathcal{N}(0, 1)$, and the linear path $\alpha_t = 1-t$, $\sigma_t = t$. Then $p_t = \mathcal{N}(0, (1-t)^2 + t^2)$, so with $\rho_t^2 = (1-t)^2+t^2$:

$$s_t(x) = -x/\rho_t^2, \qquad u_t(x) = \frac{(2t-1)}{\rho_t^2}\,x.$$

Both are linear in $x$ with a single scalar coefficient. Fit each by regression from $N$ samples at a fixed $t$. The per-sample regression target for DSM is $\epsilon$ (variance 1 at every $t$); for FM it is $v = x_0(-1) + \epsilon$ — wait, $v = \dot\alpha_t x_0 + \dot\sigma_t \epsilon = \epsilon - x_0$, variance 2, constant in $t$. The FM target has **twice the marginal variance** of the DSM target, uniformly. Naively this says FM is the *worse* estimator.

Now include the weighting. Converting the fitted FM coefficient back to a score costs a factor $1/(\sigma_t(\dot\sigma_t - \dot\alpha_t\sigma_t/\alpha_t)) = (1-t)/t$ at this path. At $t = 0.1$: the conversion factor is 9, so the FM estimate's error in score units is amplified 9×, but the FM loss at that $t$ was also implicitly weighted 81× less. At $t = 0.9$ the factor is $1/9$ and the weighting flips. Net over the timestep distribution, the two estimators' score-error curves cross — around $t = 0.5$ here.

That crossing is the whole problem in miniature. Neither objective is uniformly better; each concentrates its statistical accuracy at a different band of noise levels, and *which band matters* is set by the sampler's step allocation and by which frequencies the FID feature extractor is sensitive to. At $d = 1$ with a Gaussian, this is a two-line calculation. At $d = 4{\times}32{\times}32$ latents with a real image distribution, "which band matters" has no closed form and no measurement that is independent of the sampler — which is exactly why Section 8 fixes the solver before it varies the objective.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*