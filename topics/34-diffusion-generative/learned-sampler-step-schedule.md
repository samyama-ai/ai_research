---
id: 34-diffusion-generative/learned-sampler-step-schedule
title: "Optimal Sampler Step Schedule Learned per Model"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Sampler Step Schedule Learned per Model

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/learned-sampler-step-schedule` · **Status:** empirically-open

## 1. Problem Statement

A diffusion or flow model is sampled by integrating a reverse-time ODE/SDE at a finite set of noise levels. Which noise levels, and how many, is a free choice made *after* training. Practice uses hand-designed families: time-uniform (DDIM), log-SNR-uniform, or the EDM power law $\sigma_i \propto (\cdot)^{\rho}$ with $\rho=7$.

**The problem:** given a *fixed* trained model $s_\theta$, a fixed solver, and a budget of $N$ network evaluations (NFE), find the step schedule that minimizes distributional error — and determine whether the optimum is genuinely **per-model** (a property of that checkpoint's score error profile) or whether one schedule per architecture family is within noise of the per-model optimum.

Three variants, with different difficulty:

- **Measurement.** Can we tell two schedules apart at $N=10$? The claimed gains (0.3–1.0 FID) sit near the resolution of the metric, and metric choice reorders schedules.
- **Method.** Search $\{\sigma_i\}$ efficiently. Discrete, non-differentiable through a sample-quality metric, combinatorially large.
- **Theory.** Prove a schedule optimal for a stated divergence under stated score-error assumptions. Existing convergence proofs prescribe schedule *shapes* for KL/TV, not optima for perceptual metrics.

Solving it means: a procedure that, given a checkpoint and $N$, returns a schedule whose advantage over the best hand-designed schedule exceeds seed noise, is reproduced under a second metric, and does **not** transfer to a sibling checkpoint (which is what makes it per-model rather than per-family).

## 2. Formal Setting

Forward process $x_t = \alpha_t x_0 + \sigma_t \epsilon$, $\epsilon \sim \mathcal N(0,I)$. In EDM parameterization $\alpha_t=1$ and the probability-flow ODE is

$$\frac{dx}{d\sigma} = \frac{x - D_\theta(x;\sigma)}{\sigma},$$

with $D_\theta$ the learned denoiser. A **schedule** is a strictly decreasing sequence $\mathcal S = (\sigma_0 = \sigma_{\max} > \sigma_1 > \cdots > \sigma_N = 0)$; NFE is $N$ for Euler, $2N-1$ for Heun.

Quantities, as measured:

- **Sample-quality error** $\mathcal E(\mathcal S)$ — in practice FID between $50{,}000$ generated samples and the reference statistics (Inception-V3 pool3, 2048-d), or FD$_\text{DINOv2}$ on the same samples. Both are plug-in Fréchet distances between Gaussians fitted to finite samples, so both are biased at fixed $n$ (Chong & Forsyth, CVPR 2020).
- **Seed noise** $\hat\sigma_{\text{FID}}$ — standard deviation of FID over independent sampling seeds at fixed $\mathcal S$, $n=50$k. On CIFAR-10 this is $\approx 0.03$–$0.10$; it must be measured, not assumed, because it is the decision threshold.
- **Local truncation error** $\ell_i = \|\Phi_{\sigma_i \to \sigma_{i+1}}(x) - \phi_{\sigma_i \to \sigma_{i+1}}(x)\|$, discrete step vs. exact flow, estimated by comparing against a $\ge 1000$-step reference trajectory from the *same* seed.
- **Score error** $\varepsilon(\sigma)^2 = \mathbb E_{x_\sigma}\|s_\theta(x_\sigma,\sigma) - \nabla\log p_\sigma(x_\sigma)\|^2$ — **not directly measurable**: the true score is unknown. Proxies: denoising loss residual above the irreducible floor, or a held-out ensemble disagreement.
- **KLUB objective** (Align Your Steps, ICML 2024): an upper bound on $\mathrm{KL}(p_{\text{true}} \| p_{\mathcal S})$ decomposed per interval and estimated by Monte Carlo along model trajectories.

Assumptions and their status:

| Assumption | Status |
|---|---|
| Score error is uniform in $\sigma$ | **Violated** — error concentrates at low and mid $\sigma$; this is the whole reason schedules matter |
| Truncation error dominates score error | **Violated at low NFE**, where the two are comparable and interact |
| FID is a monotone proxy for sample quality | **Violated** — FID rankings disagree with human judgment and with FD$_\text{DINOv2}$ (Stein et al., NeurIPS 2023) |
| Schedule optimum is independent of prompt/class conditioning | Untested at scale; likely violated for text-to-image |

## 3. State of the Art

**Established (ablated, reproduced):**

- **EDM** (Karras et al., NeurIPS 2022) ablated $\rho$ directly and showed a broad optimum near $\rho\in[5,10]$ for CIFAR-10 and ImageNet-64, with $\rho=7$ adopted. This is the strongest evidence that schedule shape matters *and* that the optimum is flat.
- **Higher-order solvers dominate schedule tuning at moderate NFE.** DPM-Solver (Lu et al., NeurIPS 2022) reaches CIFAR-10 FID $\approx 4.7$ at 10 NFE versus DDIM's $13.36$ at 10 steps (Song et al., ICLR 2021) — a $\sim 8.6$ FID gap from the solver, against $\lesssim 1$ FID typically attributed to schedule.
- **ELBO-optimal $\neq$ FID-optimal.** Watson et al. (2021) solved the schedule choice exactly by dynamic programming under the ELBO and found the resulting schedules did not improve, and sometimes hurt, FID. This is the cleanest known negative result and directly motivates the measurement variant.

**Claimed, partially ablated:**

- **Align Your Steps** (Sabour, Fidler, Kreis, ICML 2024) optimizes schedules by minimizing a KL upper bound, releasing per-model schedules for SD1.5, SDXL, DeepFloyd-IF and SVD. Gains at 10 steps are reported on FID/CLIP plus human preference. The *per-model* claim (schedules differ meaningfully between checkpoints) is asserted from released tables, not from a controlled cross-application experiment.
- **GITS** (Chen et al., ICML 2024, "On the Trajectory Regularity of ODE-based Diffusion Sampling") uses dynamic programming over a discretized $\sigma$ grid with a geometric trajectory cost.
- **AutoDiffusion** (Li et al., ICCV 2023) evolutionary-searches timesteps *and* subnetworks jointly — the two factors are not separated, so the schedule-only contribution is unidentified.
- **DDSS** (Watson et al., ICLR 2022) differentiates through a perceptual sample-quality loss (KID) to learn schedules and solver coefficients — benchmark numbers only; no independent reproduction at ImageNet scale.

**Theory SOTA** is disjoint from all of the above: Chen et al. (ICLR 2023) and Benton et al. (ICLR 2024) give convergence bounds under $L^2$-accurate scores, with the latter reaching $\tilde O(d)$ iteration complexity in KL using exponentially-decreasing-then-uniform step sizes. These prescribe a schedule *family*, not a per-model optimum, and target KL, not FID.

## 4. What Is Known

- CIFAR-10, 32×32, unconditional: DDIM $\eta=0$ gives FID $13.36$ / $6.84$ / $4.67$ / $4.16$ at $10/20/50/100$ steps (Song et al., ICLR 2021, 50k samples).
- CIFAR-10: EDM Heun, $\rho=7$, 35 NFE reaches FID $1.97$ unconditional / $1.79$ conditional (Karras et al., NeurIPS 2022, 50k samples). ImageNet-64: FID $2.44$ at 511 NFE.
- The $\rho$ ablation surface is **flat**: FID changes by well under $1$ across $\rho\in[5,10]$ at 35 NFE on CIFAR-10. Most of the achievable schedule gain is already captured by any reasonable power law.
- Exact optimization under the ELBO is tractable — Watson et al.'s DP is $O(T^2)$ over a $T$-point grid — and does not improve FID. The optimization is not the bottleneck; the objective is.
- Metric disagreement is measured, not speculative: Stein et al. (NeurIPS 2023) show FID and FD$_\text{DINOv2}$ rank generative models differently, and that FID's ranking diverges from human evaluation on ImageNet-scale models.
- At $N\le 4$, distillation (progressive distillation, Salimans & Ho ICLR 2022; consistency models, Song et al. ICML 2023) beats any schedule choice on a non-distilled model by a wide margin. Schedule search is a $5\le N\le 20$ phenomenon.

## 5. What Is Not Known

- **Empirically open** — *the core gap*. Nobody has published a controlled cross-application matrix: search a schedule per checkpoint on $\ge 4$ sibling checkpoints, then apply every schedule to every checkpoint, with seed-noise error bars. Without it, "per-model" is unsupported; the observed gains are consistent with "one good schedule per resolution and solver."
- **Empirically open.** Whether searched-schedule gains survive a metric swap (FID → FD$_\text{DINOv2}$ → human preference) at fixed samples. Runnable today; not run.
- **Methodologically blocked.** There is no accepted, low-variance objective for "distributional error at 10 NFE" that is cheap enough to sit inside a search loop. KLUB is cheap but bounds a divergence nobody validates against; FID is the target but is biased and seed-noisy.
- **Theoretically open.** No characterization of the optimal $\{\sigma_i\}$ for a solver of order $p$ under a *non-uniform* score-error profile $\varepsilon(\sigma)$. Existing bounds assume a uniform $L^2$ bound and so cannot express the effect the search exploits.

## 6. Why It Is Hard

**Primary obstruction: the evaluation does not measure the thing it names, and the effect size is inside the noise band.** Claimed schedule gains at $N=10$ are $0.3$–$1.0$ FID. Seed-to-seed $\hat\sigma_{\text{FID}}$ at $n=50$k on CIFAR-10 is $0.03$–$0.10$, but FID's finite-sample bias at $n=50$k is an order of magnitude larger than that and is *not* constant across schedules — schedules that reduce sample diversity reduce FID for reasons unrelated to fidelity (Kynkäänniemi et al., ICLR 2023, on FID's sensitivity to the Inception class histogram). So the quantity being optimized is partly an artifact of the estimator.

**Secondary: non-identifiability.** Schedule, solver order, and guidance scale trade off against one another. AutoDiffusion's joint search over timesteps and architecture is the explicit case; more common is an implicit one, where a searched schedule is compared against a default schedule at a guidance scale tuned for the default.

**Tertiary: search cost.** Choosing 10 of 1000 grid points is $\binom{1000}{10}\approx 2.6\times 10^{23}$. DP reduces this to $O(T^2 N)$ only under an additive-per-interval cost — an assumption that FID violates, since FID is not a sum of per-step terms.

## 7. Current Research (as of 2026)

- **NVIDIA Toronto (Sabour, Fidler, Kreis)** — KLUB-optimized schedules, now shipped as presets in ComfyUI and Diffusers; the practical bar any new method must clear.
- **Bespoke solvers** (Shaul, Lipman et al., ICLR 2024; ICML 2024) — learn a small per-model reparameterization of the sampler, of which the schedule is a special case. This reframes the question: schedule is one coordinate of a low-dimensional per-model sampler fit, and the interesting number is how much of the gain the schedule coordinate alone contributes.
- **DP-based stepsize optimization** — 2025 work optimizing step sizes directly by dynamic programming with a trajectory-matching cost, reporting large gains on text-to-image at $\le 10$ NFE *(frontier — verify; check the specific paper's ablation for whether the control arm was retuned)*.
- **Flow-matching models with straighter trajectories** (SD3, Flux-class) reduce the schedule's leverage: as trajectories straighten, the optimal schedule approaches uniform in $t$. *(frontier — verify)*
- **Metric reform** — FD$_\text{DINOv2}$ and $\text{FID}_\infty$ adoption is the precondition for settling this problem, not a side quest.

## 8. Concrete Next Experiment

**Question decided:** is the optimal schedule per-*model* or per-*family*?

**Scale.** Four CIFAR-10 EDM checkpoints (same architecture, seeds/data-order varied) plus two ImageNet-64 EDM checkpoints. Heun solver, $N=10$ fixed. Search space: monotone 10-point subsets of a 200-point log-$\sigma$ grid, searched by DP with a trajectory-matching cost, then top-32 candidates re-ranked by full FD$_\text{DINOv2}$ at 50k samples.

**Control arms (two, both required).** (a) EDM $\rho=7$ default. (b) A *single global* schedule fitted by the same search on checkpoint 1 and applied unchanged to checkpoints 2–4. Arm (b) is the one usually omitted, and it is the one that carries the claim.

**Measurements.** For each (search-checkpoint, eval-checkpoint) pair, FD$_\text{DINOv2}$ over 5 independent 50k sample sets; report $\hat\sigma$ per cell. Repeat the whole matrix under FID to check rank stability.

**Deciding number.** Let $\Delta_{\text{own}}$ be the mean improvement of a checkpoint's own searched schedule over control (b), the transferred schedule. **Per-model is supported iff $\Delta_{\text{own}} > 3\hat\sigma$ and the same sign holds under both metrics.** Rough prior: expect $\Delta_{\text{own}} \approx 0.05$–$0.15$ FD units against $\hat\sigma \approx 0.05$ — i.e. the honest expected outcome is *inconclusive at this sample size*, which is itself publishable and is why the experiment must pre-register the sample count needed for $3\hat\sigma$.

**Cost.** One 50k-sample eval at 10 NFE is $5\times10^5$ denoiser calls; on the 56M-parameter CIFAR-10 EDM net, $\approx 4$ A100-minutes. The full matrix ($6$ checkpoints $\times$ $7$ schedules $\times$ $5$ seeds $\times$ 2 metrics) is $\approx 30$ A100-hours — cheap. The reason it has not been run is not compute.

## 9. Key References

- **[Foundational]** Jiaming Song, Chenlin Meng, Stefano Ermon. *Denoising Diffusion Implicit Models.* ICLR, 2021. — arXiv:2010.02502
- **[Foundational]** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS, 2022. — arXiv:2206.00364
- **[Foundational]** Daniel Watson, Jonathan Ho, Mohammad Norouzi, William Chan. *Learning to Efficiently Sample from Diffusion Probabilistic Models.* 2021. — arXiv:2106.03802
- **[SOTA]** Amirmojtaba Sabour, Sanja Fidler, Karsten Kreis. *Align Your Steps: Optimizing Sampling Schedules in Diffusion Models.* ICML, 2024. — arXiv:2404.14507
- **[SOTA]** Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS, 2022. — arXiv:2206.00927
- **[SOTA]** Daniel Watson, William Chan, Jonathan Ho, Mohammad Norouzi. *Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality.* ICLR, 2022. — arXiv:2202.05830
- **[SOTA]** Neta Shaul, Juan Perez, Ricky T. Q. Chen, Ali Thabet, Albert Pumarola, Yaron Lipman. *Bespoke Solvers for Generative Flow Models.* ICLR, 2024.
- **[SOTA]** Defang Chen, Zhenyu Zhou, Can Wang, Chunhua Shen, Siwei Lyu. *On the Trajectory Regularity of ODE-based Diffusion Sampling.* ICML, 2024.
- **[SOTA]** Lijiang Li et al. *AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration.* ICCV, 2023.
- **[Theory]** Joe Benton, Valentin De Bortoli, Arnaud Doucet, George Deligiannidis. *Nearly $d$-Linear Convergence Bounds for Diffusion Models via Stochastic Localization.* ICLR, 2024.
- **[Theory]** Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, Anru R. Zhang. *Sampling Is as Easy as Learning the Score: Theory for Diffusion Models with Minimal Data Assumptions.* ICLR, 2023.
- **[Evaluation]** George Stein et al. *Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models.* NeurIPS, 2023.
- **[Evaluation]** Min Jin Chong, David Forsyth. *Effectively Unbiased FID and Inception Score and Where to Find Them.* CVPR, 2020.
- **[Evaluation]** Tuomas Kynkäänniemi, Tero Karras, Miika Aittala, Timo Aila, Jaakko Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR, 2023.

## 10. Worked Example

Take the CIFAR-10 unconditional EDM checkpoint, Heun, $N=10$ (19 NFE). Two schedules on $\sigma\in[0.002, 80]$:

- $\mathcal S_A$: $\rho=7$ power law — $\sigma = (80^{1/7} + \tfrac{i}{10}(0.002^{1/7}-80^{1/7}))^7$, giving $\sigma_1 \approx 24.4$, $\sigma_5 \approx 1.62$, $\sigma_9 \approx 0.036$.
- $\mathcal S_B$: $\rho=9$, which pushes steps toward low noise — $\sigma_5 \approx 1.13$, $\sigma_9 \approx 0.020$.

Run each with 5 seeds $\times$ 50k samples. Representative outcome: FID $\;3.41 \pm 0.06$ for $\mathcal S_A$, $3.29 \pm 0.07$ for $\mathcal S_B$. A $0.12$ difference against a pooled $\hat\sigma\approx 0.065$ is $t\approx 1.8$ — a result that would be reported as a win in most papers and is not significant at $p<0.05$ with 5 seeds.

Now the part that makes the obstruction visible. Re-score the *same* image sets with FD$_\text{DINOv2}$. Because DINOv2 features weight fine texture differently from Inception's ImageNet-class-driven pool3 features, and $\mathcal S_B$ spends more steps at low $\sigma$ (sharpening, slightly reducing diversity), the ordering can invert: $\mathcal S_B$ can score *worse* on FD$_\text{DINOv2}$ while scoring better on FID. Nothing about the samples changed — only the estimator did.

Two consequences follow directly:

1. To reach $3\hat\sigma$ separation for a $0.12$ FID gap you need $\hat\sigma \le 0.04$, i.e. roughly $5\times$ more seeds ($\approx 25$ runs of 50k) per schedule. A 1000-candidate search under that standard is $\sim 1700$ A100-hours — and the search would be optimizing a partly-artifactual objective.
2. Any per-model schedule search that reports a single FID number per schedule, with one seed and one metric, cannot distinguish "this checkpoint prefers this schedule" from estimator noise. That is the state of most published results, and it is why the status here is **empirically open** rather than partially solved.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*