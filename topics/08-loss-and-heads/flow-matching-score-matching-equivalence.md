---
id: 08-loss-and-heads/flow-matching-score-matching-equivalence
title: "Flow Matching Versus Score Matching Objective Equivalence"
topic: 08-loss-and-heads
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Flow Matching Versus Score Matching Objective Equivalence

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/flow-matching-score-matching-equivalence` · **Status:** partially-solved

## 1. Problem Statement

Conditional flow matching (CFM) regresses a velocity field along an interpolation between noise and data; denoising score matching (DSM) regresses the score of a noised data distribution. Both train a single network with a squared loss on a Gaussian-corrupted sample. The question is whether they are the *same* objective.

Three variants, with different difficulty:

- **Theory variant.** For Gaussian conditional paths, is the CFM loss an instance of the DSM loss family, i.e. does there exist a reparameterization plus a weighting $w(t)$ and timestep density $p(t)$ mapping one to the other exactly? *Answered yes.* The remaining theory question is which path families break the correspondence, and whether the induced weightings are ever outside the ELBO-admissible (monotone) class.
- **Measurement variant.** Given two trained models, is there a metric that certifies "same objective up to weighting" rather than "similar FID"? Currently there is no such certificate.
- **Method variant.** If the objectives are equivalent as population losses, are they equivalent as *optimization problems* at finite width, finite data and finite steps? Reported FID gaps between flow-matching and diffusion training runs are 5–15% at matched architecture. Solving the problem means attributing that gap entirely to $(w, p(t), \text{parameterization}, \text{sampler})$ — or exhibiting a residual that none of these explain.

## 2. Formal Setting

Data $x_1 \sim q$ on $\mathbb{R}^d$, noise $x_0 \sim \mathcal{N}(0, I)$. Affine Gaussian path with schedule $(\alpha_t, \sigma_t)$, $t \in [0,1]$, $\alpha_0 = 0, \sigma_0 = 1, \alpha_1 = 1, \sigma_1 \approx 0$:

$$x_t = \alpha_t x_1 + \sigma_t x_0, \qquad p_t(x) = \int \mathcal{N}(x; \alpha_t x_1, \sigma_t^2 I)\, q(dx_1).$$

Define log-SNR $\lambda_t = 2\log(\alpha_t/\sigma_t)$ — measured directly from the schedule, not estimated. The marginal velocity and score are

$$u_t(x) = \mathbb{E}[\dot\alpha_t x_1 + \dot\sigma_t x_0 \mid x_t = x], \qquad s_t(x) = \nabla_x \log p_t(x) = -\frac{x - \alpha_t\,\mathbb{E}[x_1|x_t=x]}{\sigma_t^2}.$$

Eliminating $\mathbb{E}[x_1|x_t]$ gives the exact affine link:

$$u_t(x) = \frac{\dot\alpha_t}{\alpha_t}\,x + \sigma_t\!\left(\frac{\dot\alpha_t \sigma_t}{\alpha_t} - \dot\sigma_t\right) s_t(x).$$

Both objectives are then instances of one weighted denoising loss. With $\hat x_\theta$ the implied clean-data prediction,

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim p(t)}\,\mathbb{E}_{x_1, x_0}\left[\,w(t)\,\|\hat x_\theta(x_t, t) - x_1\|^2\,\right].$$

CFM with $\|v_\theta - (\dot\alpha_t x_1 + \dot\sigma_t x_0)\|^2$ corresponds to $w_{\mathrm{FM}}(t) \propto (\dot\alpha_t - \alpha_t\dot\sigma_t/\sigma_t)^2$; $\epsilon$-prediction DSM corresponds to $w_{\epsilon}(t) \propto \alpha_t^2/\sigma_t^2$. Under the change of variables to $\lambda$, both become $\int w(\lambda)\,\mathbb{E}\|\hat x_\theta - x_1\|^2 d\lambda$ (Kingma & Gao, NeurIPS 2023).

**Measured quantities.** $w(t)$ and $p(t)$ are read off the code, not fit. Per-$t$ loss is measured as a Monte-Carlo mean over $\geq 10^4$ $(x_1, x_0)$ pairs per bin. Gradient noise is measured as $\mathrm{tr}\,\mathrm{Cov}(g)/\|\mathbb{E} g\|^2$ over a fixed batch stream. Sample quality is FID-50K against the reference statistics of the dataset.

**Assumptions, and where they fail.**
1. *Gaussian conditional path with independent coupling.* Violated by minibatch-OT couplings (Tong et al., TMLR 2024) and by reflow: $x_0 \not\perp x_1$, so $p_t$ is no longer a Gaussian mixture with known score, and the affine link above does not hold.
2. *$\sigma_1 = 0$ exactly.* Violated — flow matching terminates at $\sigma_1 = 0$ where $w_\epsilon$ diverges; diffusion uses $\sigma_1 > 0$. The two losses therefore integrate over different $\lambda$ supports, and the endpoint is where the mapping is singular.
3. *Optimal network.* The equivalence is a statement about population minimizers. Finite-capacity networks with different output heads ($v$, $\epsilon$, $\hat x$) do not share a minimizer.
4. *Same sampler.* Violated in nearly every published comparison.

## 3. State of the Art

**Theory (established).** Vincent (2011) reduces score matching on Gaussian-smoothed data to denoising regression. Song et al. (ICLR 2021) give the probability-flow ODE, whose drift is exactly the $u_t$ above. Lipman et al. (ICLR 2023), Liu et al. (ICLR 2023) and Albergo & Vanden-Eijnden (ICLR 2023) independently derive the CFM identity: the conditional and marginal regression losses have the same gradient. Kingma & Gao (NeurIPS 2023) close the loop: every common diffusion objective — $\epsilon$, $v$, $\hat x$, EDM, and flow matching — is $\int w(\lambda) \mathcal{L}_\lambda d\lambda$, and is an ELBO under Gaussian data augmentation iff $w(\lambda)$ is monotone non-increasing. This is the strongest form of the equivalence currently proven.

**Empirical (claimed, partially ablated).** SiT (Ma et al., ECCV 2024) is the closest thing to a controlled study: one DiT backbone, stepwise interchange of discrete→continuous time, score→velocity prediction, path, and sampler. FID improves at each step. This is the only published ablation that varies one axis at a time; it is at a single architecture family and one dataset.

**Benchmark-only.** Stable Diffusion 3 (Esser et al., ICML 2024) sweeps 61 formulation/schedule combinations and reports rectified flow with logit-normal timestep sampling as best. The comparison is by validation loss and human preference at fixed step count; the weighting and timestep distribution are *not* held fixed across arms, so it does not separate objective from weighting. Treat as a benchmark number, not an ablation.

## 4. What Is Known

- **Exact population equivalence for Gaussian paths.** Theorem (Lipman et al. 2023, Thm. 2): $\nabla_\theta \mathcal{L}_{\mathrm{CFM}} = \nabla_\theta \mathcal{L}_{\mathrm{FM}}$. Combined with the affine link, a flow-matching model and a score model on the same schedule are deterministic reparameterizations of each other at the optimum.
- **Weighting, not objective, carries the empirical differences.** Kingma & Gao show the FM weighting in $\lambda$-space is close to the EDM monotone weighting; on ImageNet 64×64 with a ~400M-parameter U-ViT, swapping weightings moves FID by more than swapping parameterizations.
- **Measured gaps at matched architecture.** SiT-XL/2 vs DiT-XL/2, ImageNet 256×256, identical backbone (675M params): FID-50K 2.06 vs 2.27 with classifier-free guidance; at 400K training iterations without guidance, roughly 17.2 vs 19.5. The gap is ~10%, and the SiT ablation attributes most of it to the continuous-time interpolant and sampler choice rather than to velocity-vs-score prediction alone.
- **Non-equivalence is also established** for non-Gaussian and non-independent couplings: reflow (Liu et al. 2023) changes the marginal path, so the resulting model is not a reparameterized score model of the original $p_t$.
- **Preconditioning matters more than either loss name.** EDM (Karras et al., NeurIPS 2022) improved ImageNet 64×64 FID from ~2.07 to 1.36 by changing preconditioning, weighting and sampler while keeping DSM.

## 5. What Is Not Known

- **Theoretically open.** Whether the *optimization trajectories* under $w_{\mathrm{FM}}$ and $w_\epsilon$ converge to the same function class neighborhood at finite width. No result bounds $\|\hat x_{\mathrm{FM}} - \hat x_{\mathrm{DSM}}\|$ after $T$ SGD steps as a function of $w$. Also open: whether any commonly used flow-matching weighting is non-monotone in $\lambda$ and therefore not an ELBO — Kingma & Gao settle this for the schedules they list, not in general.
- **Empirically open.** Nobody has run the fully-crossed 2×2×2 (parameterization × weighting × sampler) at $\geq 1$B parameters with everything else frozen. The ablations that exist are single-axis at $\leq 700$M. Runnable today for roughly $10^4$–$10^5$ A100-hours.
- **Methodologically blocked.** There is no metric that certifies two trained models implement the same $\hat x$ up to the affine link. FID differences of 0.2 are within seed noise ($\pm 0.1$–$0.2$ at FID-50K), so FID cannot resolve the residual. A direct functional distance $\mathbb{E}_{t,x_t}\|\hat x_A - \hat x_B\|^2$ is measurable but has no calibrated scale — nobody has published the seed-to-seed baseline for it.

## 6. Why It Is Hard

**Confounded measurement.** The "flow matching vs score matching" comparison bundles at least five variables: prediction target, loss weighting, timestep density, time discretization, and sampler (ODE vs SDE, step count, guidance scale). Published comparisons change three or more at once. A flow-matching arm nearly always ships with a uniform-$t$ or logit-normal $p(t)$ and an ODE sampler; a diffusion arm ships with a cosine schedule and an ancestral SDE sampler. The reported FID gap is a gap between *configurations*, not between objectives, and the theory says the objective axis alone should contribute exactly zero at the population optimum.

Second obstruction: **non-identifiability of the residual**. Because the population minimizers coincide, any measured difference is a finite-sample/finite-capacity effect, whose size is the same order as seed variance in the standard metric. Detecting it requires a metric more sensitive than FID, and defining that metric is the blocked step.

## 7. Current Research (as of 2026)

- **Unified frameworks.** Stochastic interpolants (Albergo, Boffi & Vanden-Eijnden, NYU/Flatiron) continue to be the reference formalism; the *Flow Matching Guide and Code* (Lipman et al., Meta FAIR, 2024) is the current pedagogical unification.
- **Weighting-first design.** DeepMind's *Diffusion Meets Flow Matching* exposition (Gao, Hoogeboom, Heek, De Bortoli, Murphy, Salimans, 2024) argues the two differ only by schedule and weighting in the Gaussian case — the position this catalog page treats as established for population losses.
- **Genuinely non-equivalent extensions.** Discrete/Markov-jump flow matching, Riemannian flow matching, and OT-coupled flows are where the correspondence provably breaks; work here is about building score analogues at all *(frontier — verify)*.
- **Distillation.** Whether few-step distillation is easier from a velocity head than a score head is actively claimed by several groups and, as far as we know, untested with matched weighting *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** after equalizing weighting, timestep density and sampler, does any FID gap between velocity-head and $\epsilon$-head training survive?

- **Scale.** DiT-B/2 (130M params) and DiT-XL/2 (675M), ImageNet 256×256 latents, 400K iterations, batch 256. Three seeds per arm. About 3K–8K A100-hours total.
- **Arms.** 2×2 crossed: head $\in \{v, \epsilon\}$ × weighting $\in \{w_{\mathrm{FM}}(\lambda), w_{\epsilon}(\lambda)\}$. Timestep density fixed to uniform in $\lambda$ on $[\lambda_{\min}, \lambda_{\max}] = [-12, 6]$ for all four. Identical schedule, identical data order, identical Heun ODE sampler at 50 NFE, no guidance.
- **Control arm.** Same head and weighting, three different seeds — this gives the noise floor.
- **Deciding number.** $\Delta = |\mathrm{FID}_{v} - \mathrm{FID}_{\epsilon}|$ at matched weighting, compared against the seed-to-seed standard deviation $s$. If $\Delta < 2s$ (expect $s \approx 0.15$ at FID-50K, so $\Delta < 0.3$), the head choice is empirically inert and the equivalence is complete in practice. If $\Delta > 2s$ at XL and grows with scale, a residual exists that the weighting account does not explain.
- **Secondary readout.** $\mathbb{E}_{t,x_t}\|\hat x_v - \hat x_\epsilon\|^2$ per $\lambda$-bin, normalized by the same quantity between two seeds of one arm. A ratio near 1 is the functional certificate section 5 says is missing; publishing the seed baseline is itself the contribution.

## 9. Key References

- **[Foundational]** Pascal Vincent. *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation 23(7), 2011.
- **[Foundational]** Aapo Hyvärinen. *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR 6, 2005.
- **[Foundational]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. — arXiv:2011.13456
- **[Foundational]** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. *Flow Matching for Generative Modeling.* ICLR 2023. — arXiv:2210.02747
- **[Foundational]** Xingchao Liu, Chengyue Gong, Qiang Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. — arXiv:2209.03003
- **[Foundational]** Michael S. Albergo, Eric Vanden-Eijnden. *Building Normalizing Flows with Stochastic Interpolants.* ICLR 2023. — arXiv:2209.15571
- **[SOTA — theory]** Diederik P. Kingma, Ruiqi Gao. *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.* NeurIPS 2023. — arXiv:2303.00848
- **[SOTA — empirical]** Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, Saining Xie. *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers.* ECCV 2024. — arXiv:2401.08740
- **[SOTA — systems]** Patrick Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024. — arXiv:2403.03206
- **[SOTA — design space]** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. — arXiv:2206.00364
- **[Related]** Alexander Tong et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR 2024. — arXiv:2302.00482
- **[Survey]** Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, Itai Gat. *Flow Matching Guide and Code.* 2024. — arXiv:2412.06264
- **[Survey]** Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin P. Murphy, Tim Salimans. *Diffusion Meets Flow Matching: Two Sides of the Same Coin.* Technical exposition, Google DeepMind, 2024.

## 10. Worked Example

Take the linear (rectified-flow) path $\alpha_t = t$, $\sigma_t = 1-t$. Then $\dot\alpha_t = 1$, $\dot\sigma_t = -1$, and the affine link gives

$$u_t(x) = \frac{1}{t}x + (1-t)\left(\frac{1-t}{t} + 1\right) s_t(x) = \frac{x}{t} + \frac{1-t}{t}\,s_t(x).$$

Convert both losses to the common $\hat x$ form. Flow matching: $\|v_\theta - (x_1 - x_0)\|^2$. Since $x_1 - x_0 = (x_1 - x_t)/(1-t)$, the implied weighting is $w_{\mathrm{FM}}(t) = 1/(1-t)^2$. Epsilon prediction: $w_\epsilon(t) = t^2/(1-t)^2$. The ratio is

$$\frac{w_{\mathrm{FM}}(t)}{w_\epsilon(t)} = \frac{1}{t^2}.$$

At $t=0.5$ the ratio is 4; at $t = 0.05$ (nearly pure noise) it is 400. So with uniform $t$, flow matching puts ~400× more relative weight on the high-noise end than $\epsilon$-prediction does. Nothing about "velocity versus score" is involved — this is entirely the weighting.

Now the obstruction. Suppose you run the two configurations as shipped and get FID 2.06 and 2.27 on ImageNet 256×256 — a 0.21 gap, roughly the SiT/DiT number. Seed noise at FID-50K is about $\pm 0.15$. The 400× weighting difference at $t=0.05$ is enough to explain a gap of that size several times over, and so is the sampler difference. The measurement has one number and at least three candidate causes, two of which the theory says should matter and one of which the theory says should not. Until the weighting is equalized in $\lambda$-space and the sampler is frozen, the 0.21 is uninterpretable — and once it is equalized, the predicted gap is smaller than the noise floor of the metric used to look for it. That is why the problem is partially solved: the theory is closed, and the experiment that would confirm the theory is closed empirically has not been run with an instrument sharp enough to read the answer.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*