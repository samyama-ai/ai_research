---
id: 03-training-dynamics/optimal-learning-rate-schedule-shape
title: "Optimal Learning Rate Schedule Shape"
topic: 03-training-dynamics
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Learning Rate Schedule Shape

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/optimal-learning-rate-schedule-shape` · **Status:** partially-solved

## 1. Problem Statement

Given a fixed compute budget, what function of step should the learning rate be?

- **Input:** a model family, a data distribution, an optimizer (in practice AdamW), a total step count $T$, a batch size, and a weight-decay setting.
- **Output:** a sequence $\eta_1,\dots,\eta_T$.
- **Objective:** minimize the final validation loss $L(\theta_T)$ — not the average loss, not the best intermediate loss.

Three variants that are usually conflated:

- **Theory variant.** Prove that a schedule shape is optimal (or within a constant) for the *last iterate* under an assumption class that includes deep-network training. Convex non-smooth theory already answers this; the assumption class does not match.
- **Measurement variant.** Establish that one shape beats another by more than seed and hyperparameter noise, at a scale where the answer transfers. Requires separating *shape* from *peak LR*, which are not independently identified.
- **Method variant.** Produce a schedule that is compute-anytime (does not need $T$ in advance) and loses nothing against a $T$-tuned schedule. Warmup-Stable-Decay and Schedule-Free both attack this.

Solved would mean: a shape family with a parameterization whose optimum is *predictable* from $(N, D)$ without running the sweep, plus evidence that no other family beats it by more than $\sim 0.005$ nats at matched compute.

## 2. Formal Setting

Parameters $\theta_t \in \mathbb{R}^N$, loss $L(\theta) = \mathbb{E}_{x\sim\mathcal{D}}[\ell(\theta,x)]$. AdamW update with peak LR $\eta_{\max}$, schedule $s: \{1..T\}\to[0,1]$, $\eta_t = \eta_{\max}\, s(t)$:

$$\theta_{t+1} = \theta_t - \eta_t\big(\hat m_t/(\sqrt{\hat v_t}+\epsilon) + \lambda\theta_t\big).$$

**Measured quantities.**

- $L_{\text{final}}$: cross-entropy in nats/token on a held-out shard of the *same* distribution, $\ge 10^8$ tokens so the standard error is $\lesssim 10^{-3}$ nats.
- Compute $C = 6ND$ tokens-FLOPs (forward+backward, dense transformer), $D = T\cdot B\cdot L_{\text{seq}}$.
- Seed noise $\sigma_{\text{seed}}$: std of $L_{\text{final}}$ over $\ge 3$ data-order + init seeds. At 100M–1B scale this is $2\text{–}5\times10^{-3}$ nats. **Any claimed shape effect below $2\sigma_{\text{seed}}$ is unmeasured.**
- $\bar\eta = \frac{1}{T}\sum_t \eta_t$ (the "area" of the schedule) and the decay fraction $\rho$ = share of steps in the final decay.

**Common shapes.** Cosine to $\alpha\eta_{\max}$: $s(t)=\alpha+(1-\alpha)\tfrac12(1+\cos(\pi t/T))$; linear-to-zero (D2Z): $s(t)=1-t/T$; WSD: $s=1$ for $t<(1-\rho)T$ then a decay (linear, $1/t$, or $\sqrt{}$-shaped) to $\approx 0$.

**Assumptions, and which are violated.**

1. *Convexity / bounded subgradients* — used by every last-iterate proof. **Violated.**
2. *$L$ is stationary in $t$* — repeated-epoch or curriculum data breaks it. Usually holds for single-epoch LLM pretraining.
3. *Peak LR $\eta_{\max}$ is held fixed across shapes.* **Routinely violated in published comparisons**, which is the core identification problem (§6).
4. *Weight decay $\lambda$ independent of $\eta$.* Violated: AdamW's effective decay is $\eta\lambda$, so changing shape changes the decay trajectory. Power Lines (Bergsma et al., 2025) shows optimal $\lambda$ scales with $1/D$.
5. *Loss is a function of the schedule alone, not of the schedule–batch-size interaction.* Violated near the critical batch size.

## 3. State of the Art

**Theory SOTA (established).** For convex, $G$-Lipschitz $f$ over a $D$-diameter set, the *last iterate* of subgradient descent with linearly decaying step size attains the optimal $O(GD/\sqrt{T})$ rate; Zamani & Glineur (2023) give the exact tight constant, and Defazio et al. (2023) showed the resulting schedule is linear-decay-to-zero. Schaipp et al. (ICML 2025) then showed the convex bound, evaluated on measured gradient norms, *predicts* the shape of real LLM loss curves — including the sharp WSD cooldown drop — to within qualitative agreement. This is the strongest link from theory to practice currently in existence, and it is an agreement of shape, not a guarantee.

**Empirical SOTA (established).** WSD / trapezoidal schedules match cosine at matched compute while removing the need to fix $T$ in advance (Hu et al., MiniCPM 2024; Hägele et al., NeurIPS 2024). Linear-decay-to-zero beats cosine-to-$0.1\eta_{\max}$ at high tokens-per-parameter (Bergsma et al., ICLR 2025).

**Claimed but under-ablated.**

- The specific *cooldown curve* (linear vs. $1-\sqrt{}$ vs. $1/t$): several papers report small wins for non-linear cooldowns; the deltas are near seed noise and have not been reproduced independently across model families.
- Multi-power-law schedule optimization (Luo et al., ICML 2025) fits a loss-curve law over schedules and optimizes the schedule directly; the reported gains over cosine at ~400M–1B are real benchmark numbers but exist mainly in the authors' own setup.
- Schedule-Free AdamW (Defazio et al., NeurIPS 2024) won the 2024 AlgoPerf self-tuning track — a benchmark number on a non-LLM-dominated suite. Whether it matches a tuned WSD run at LLM pretraining scale is not settled.

## 4. What Is Known

- **Cycle-length mismatch is costly.** Hoffmann et al. (Chinchilla, 2022) report that setting the cosine cycle to $\sim\!10\times$ the actual training length raises final loss materially; the cycle must end at $T$. This is the single most reproduced schedule result.
- **The decay phase does most of the work.** In WSD, loss is roughly flat during the stable phase and drops sharply during cooldown. Hägele et al. (2024), at 210M and 400M params, find $\rho \in [0.1, 0.2]$ suffices; shorter cooldowns underperform, longer ones give nothing.
- **Decay to zero beats decay to a floor at high TPP.** Bergsma et al. (2025), models 0.1B–1.2B, 20–320 tokens/param: the advantage of D2Z over cosine-to-10% *grows* with tokens-per-parameter, and at ~600M params and ~80 TPP the D2Z run reaches the cosine run's loss with substantially less compute (authors report on the order of a 40–60% saving; the exact figure is setup-specific).
- **Peak LR and shape trade off.** Lower-area schedules tolerate (and want) higher $\eta_{\max}$. This is why shape comparisons at fixed $\eta_{\max}$ are misleading.
- **Warmup matters less than believed.** Kosson et al. (2024) show most of warmup's benefit is reproducible by controlling the update-to-weight ratio, i.e. warmup is a proxy for a normalization effect, not a schedule primitive.
- **Mechanism sketch.** Wen et al. (2024) explain WSD via a "river valley" landscape: high LR moves fast along the river (bias reduction) while bouncing across it; cooldown collapses the cross-river variance. Explanatory, not a theorem about optimality.

## 5. What Is Not Known

- **Theoretically open.** No last-iterate optimality result for any schedule under assumptions that admit deep networks. No proof that a constant-then-decay shape is optimal for *any* non-convex class of interest. The convex bound's agreement with practice (Schaipp et al.) is unexplained.
- **Empirically open.** Whether the residual gap between the best hand-designed shape and the true optimum exceeds seed noise, at $\ge$ 7B params. Nobody has run a proper shape sweep at that scale with $\eta_{\max}$ re-tuned per shape and $\ge 3$ seeds, because the cost is a full pretraining run per cell.
- **Methodologically blocked.** "Optimal shape" is not well posed without specifying whether $\eta_{\max}$, $\lambda$, and $B$ are re-optimized jointly. Under joint re-optimization, several distinct shapes may be *the same point* in the induced dynamics; there is no accepted invariant (candidate: $\bar\eta$, or $\sum_t \eta_t^2$) under which schedules can be compared shape-to-shape.

## 6. Why It Is Hard

**Non-identifiability of shape from peak.** The final loss depends on the schedule mostly through low-order summaries — total area $\sum_t\eta_t$, the decay tail, and $\eta\lambda$ integrated over training. Two schedules with different shapes but matched summaries are often within seed noise of each other. So a shape comparison is really a comparison of *shape families under their own optimal $\eta_{\max}$*, which requires an inner sweep per arm. Cost per honest cell: (number of $\eta_{\max}$ values) $\times$ (number of seeds) $\times$ one full run. At 7B/140B tokens that is $\sim 5\times3 = 15$ runs $\approx 1.8\times10^{22}$ FLOPs per shape family.

**Compounding this:** the effect size is small. Published shape wins at matched compute are $0.005$–$0.02$ nats, against $\sigma_{\text{seed}}\approx 0.003$. And the quantity that matters downstream — benchmark accuracy — is a noisier function of loss than loss is of the schedule, so "the evaluation does not measure the thing it names" applies to any shape claim validated on task scores.

## 7. Current Research (as of 2026)

- **Loss-curve laws over schedules.** Multi-power-law fitting (Luo, Lyu et al.) to predict $L(t)$ for *unseen* schedules, then optimize the schedule inside the fitted law. Extending these to $>$1B and to re-tuned $\eta_{\max}$ is active *(frontier — verify)*.
- **Convex-theory transfer.** Schaipp, Defazio and collaborators, using measured gradient-norm trajectories to instantiate convex bounds as predictive curves.
- **Schedule-free and anytime methods.** Meta/FAIR (Defazio, Mishchenko, Cutkosky) — averaging-based methods that remove $T$ from the design.
- **Joint hyperparameter scaling laws.** Cerebras (Bergsma et al.), and $\mu$P-adjacent work, fitting $(\eta_{\max}, \lambda, B)$ jointly against $(N,D)$ so shape is not compared at a mis-tuned peak.
- **Cooldown for continual/branched training.** Reusing a stable-phase checkpoint to branch multiple cooldowns for different data mixes — increasingly the practical reason WSD is chosen over cosine.

## 8. Concrete Next Experiment

**Question:** does schedule shape matter beyond total LR area, once $\eta_{\max}$ is re-tuned per shape?

- **Scale:** 1.3B-param decoder, 26B tokens (20 TPP) and 104B tokens (80 TPP) — two TPP points, because the D2Z effect is known to grow with TPP.
- **Arms (4 shapes):** cosine-to-10%; linear-to-zero; WSD with $\rho=0.2$ linear cooldown; WSD with $\rho=0.2$ and a $1-\sqrt{t}$ cooldown. Warmup fixed at 2% for all.
- **Inner sweep:** 4 values of $\eta_{\max}$ per arm on a log grid spanning $4\times$; 3 seeds at the arm's argmin only. Weight decay set by the $\eta\lambda D$ rule, not held fixed.
- **Control arm:** a *shape-scrambled* schedule — the linear-to-zero values permuted into a random monotone-free order with identical $\sum_t\eta_t$ and identical final 5% of steps. If shape carries information beyond area and tail, this arm must lose.
- **Cost:** $4\times4 + 4\times2 = 24$ runs per TPP point, $\approx 2\times10^{21}$ FLOPs total for both points.

**The deciding number:** $\Delta = L_{\text{best shape}} - L_{\text{worst shape}}$, each at its own tuned $\eta_{\max}$, compared to $2\sigma_{\text{seed}}$ measured in the same run set. If $\Delta < 2\sigma_{\text{seed}}$ at both TPP points, shape is a red herring at fixed area and the field should tune area and cooldown fraction only. If $\Delta > 0.01$ nats at 80 TPP but not at 20 TPP, shape optimality is a high-TPP phenomenon and every sub-Chinchilla ablation in the literature is uninformative about it.

## 9. Key References

- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Loshchilov & Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. — arXiv:1608.03983
- **[Theory]** Zamani & Glineur. *Exact convergence rate of the last iterate in subgradient methods.* 2023. — arXiv:2307.11134
- **[Theory]** Defazio, Cutkosky, Mehta, Mishchenko. *When, Why and How Much? Adaptive Learning Rate Scheduling by Refinement.* 2023. — arXiv:2310.07831
- **[SOTA]** Hägele, Bakouch, Kosson, Allal, von Werra, Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS 2024. — arXiv:2405.18392
- **[SOTA]** Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* COLM 2024. — arXiv:2404.06395
- **[SOTA]** Bergsma, Dey, Gosal, Gray, Soboleva, Hestness. *Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs.* ICLR 2025. — arXiv:2502.15938
- **[SOTA]** Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky. *The Road Less Scheduled.* NeurIPS 2024. — arXiv:2405.15682
- **[SOTA]** Luo, Wen, Hu, Sun, Liu, Sun, Lyu, Chen. *A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules.* ICML 2025. — arXiv:2503.12811
- **[Analysis]** Wen, Li, Liu, He, Xie. *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective.* 2024. — arXiv:2410.05192
- **[Analysis]** Schaipp, Hägele, Taylor, Simsekli, Bach. *The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training.* ICML 2025. — arXiv:2501.18965
- **[Analysis]** Kosson, Messmer, Jaggi. *Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training.* NeurIPS 2024. — arXiv:2410.23922
- **[Related]** Bergsma, Dey, et al. *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training.* 2025. — arXiv:2505.13738

## 10. Worked Example

Take $T = 100{,}000$ steps, $\eta_{\max} = 3\times10^{-3}$, and compare cosine-to-10% against linear-to-zero at *the same* $\eta_{\max}$.

Areas:

$$\bar\eta_{\cos} = \eta_{\max}\big(0.1 + 0.9\cdot\tfrac12\big) = 0.55\,\eta_{\max}, \qquad \bar\eta_{\text{lin}} = 0.5\,\eta_{\max}.$$

So the two arms differ by 10% in total LR area — before any question of shape. With AdamW at $\lambda = 0.1$, the integrated decay $\sum_t \eta_t\lambda$ differs by the same 10%, and the cosine arm spends its last 10,000 steps at $3\times10^{-4}$ while the linear arm ends near $3\times10^{-8}$.

Now the measurement. Suppose the experiment reports $L_{\text{lin}} = 2.812$, $L_{\cos} = 2.826$ — a 0.014-nat win for linear. Is this shape?

- Seed std at this scale: $\sigma_{\text{seed}} \approx 0.003$, so $2\sigma = 0.006$. The gap clears noise.
- But re-tune $\eta_{\max}$ per arm. Cosine's larger area means its optimum sits at a *lower* peak; on a log grid the cosine arm's best cell typically moves to $\eta_{\max} = 2.4\times10^{-3}$, recovering perhaps 0.008 nats. The residual is 0.006 — exactly at the noise threshold.
- Then match areas explicitly by scaling cosine's peak by $0.5/0.55$. Most of what remains is the *tail*: cosine's $0.1\eta_{\max}$ floor never lets the cross-valley variance collapse.

The obstruction is visible in that sequence. The headline 0.014 decomposes into an area effect (tunable away), a peak-LR mis-tuning effect (an artifact of the protocol), and a tail effect (probably real). Only the third is "shape", and it is the smallest of the three — roughly the size of seed noise at 1B params. Any experiment that does not run the inner $\eta_{\max}$ sweep is measuring the first two and calling it the third.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*