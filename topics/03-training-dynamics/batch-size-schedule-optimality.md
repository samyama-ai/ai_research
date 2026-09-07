---
id: 03-training-dynamics/batch-size-schedule-optimality
title: "Batch Size Schedule Optimality"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Batch Size Schedule Optimality

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/batch-size-schedule-optimality` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed compute budget, is the optimal batch size a *constant*, and if not, what is the optimal *schedule* $B(t)$?

Almost all large-scale training runs hold batch size fixed (or ramp it once, early, by hand). Theory since Byrd et al. (2012) and McCandlish et al. (2018) says the noise-efficient batch size grows as the gradient signal-to-noise ratio falls — i.e. as training progresses — implying a monotonically increasing $B(t)$. Nobody has established, at frontier scale, how much loss a tuned $B(t)$ buys over the best constant $B$ once the learning-rate schedule is re-tuned for each arm.

Three variants, of very different difficulty:

- **Measurement.** Estimate the critical batch size $B_{\text{crit}}(t)$ — the batch size beyond which doubling $B$ buys less than half the step reduction — *during* a run, from quantities available online. Cost: cheap in principle, confounded in practice (§6).
- **Method.** Produce a schedule rule $B(t)$ (open-loop or feedback) that beats the best constant-$B$ baseline in wall-clock or total FLOPs at matched final loss, with the learning rate re-tuned per arm.
- **Theory.** Prove, for a non-convex objective under a stated noise model, that a time-varying $B(t)$ strictly dominates every constant $B$ at equal gradient-evaluation budget, and characterize the optimum.

Solving it means: a rule computable from a run's own telemetry that gives a measured reduction in FLOPs-to-target-loss over the tuned constant baseline, reproduced at two model scales an order of magnitude apart.

## 2. Formal Setting

Loss $L(\theta) = \mathbb{E}_{x\sim\mathcal{D}}[\ell(\theta,x)]$, $\theta\in\mathbb{R}^d$. At step $t$ a batch $B_t$ of examples (in LLM practice: tokens) gives $g_t = \frac{1}{B_t}\sum_{i} \nabla\ell(\theta_t,x_i)$, with $\mathbb{E}[g_t]=\nabla L$ and $\operatorname{Cov}(g_t) = \Sigma(\theta_t)/B_t$.

**Budgets.** Sequential steps $T=\sum_t 1$; total examples $E=\sum_t B_t$. Compute $C \propto E$ for a fixed model; wall-clock $\approx T\cdot(\text{step time})$ where step time is roughly flat in $B$ until the accelerators saturate, then linear. The objective is a two-resource optimum, not one:

$$\min_{B(\cdot),\,\eta(\cdot)} \; \lambda_{\text{seq}} T + \lambda_{\text{data}} E \quad \text{s.t.}\quad L(\theta_T)\le L^\star .$$

**Noise scale (measured).** McCandlish et al. (2018) define
$$\mathcal{B}_{\text{noise}} = \frac{\operatorname{tr}(\Sigma)}{\nabla L^\top H \nabla L}, \qquad \mathcal{B}_{\text{simple}} = \frac{\operatorname{tr}(\Sigma)}{\|\nabla L\|^2},$$
with $H$ the Hessian. $\mathcal{B}_{\text{simple}}$ is measured without $H$: run two batch sizes $B_{\text{small}}, B_{\text{big}}$ (in practice, per-worker versus all-reduced gradient), form unbiased estimates
$$\widehat{|G|^2} = \frac{B_{\text{big}}\|g_{B_{\text{big}}}\|^2 - B_{\text{small}}\|g_{B_{\text{small}}}\|^2}{B_{\text{big}}-B_{\text{small}}},\qquad \widehat{S} = \frac{\|g_{B_{\text{small}}}\|^2-\|g_{B_{\text{big}}}\|^2}{1/B_{\text{small}}-1/B_{\text{big}}},$$
and take $\widehat{\mathcal{B}}_{\text{simple}} = \widehat{S}/\widehat{|G|^2}$, exponentially averaged (the ratio of noisy estimates is badly biased at single-step granularity).

**Critical batch size (measured).** Empirically, $B_{\text{crit}}$ is fit from a *steps-to-target* curve: train to fixed loss $L^\star$ at several $B$, fit the Pareto hyperbola
$$\left(\frac{T}{T_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1,$$
which gives $B_{\text{crit}} = E_{\min}/T_{\min}$. Every arm needs its own tuned $\eta$; otherwise $B_{\text{crit}}$ measures tuning effort, not dynamics.

**Assumptions, and which are violated.**
1. *Fixed-noise / stationary $\Sigma$* — violated; $\operatorname{tr}(\Sigma)/\|\nabla L\|^2$ rises by 1–2 orders of magnitude over a run.
2. *SDE approximation of SGD* is valid — holds only under small enough $\eta$ (Li, Malladi & Arora, NeurIPS 2021); frontier runs sit near the edge-of-stability boundary where it fails.
3. *Linear scaling rule* $\eta\propto B$ — holds for SGD in a mid range only (Goyal et al. 2017); for Adam the correct rule is $\eta\propto\sqrt{B}$ under an SDE analysis (Malladi et al., NeurIPS 2022), and neither holds at the extremes.
4. *IID sampling* — violated by curriculum, data ordering, and single-epoch pretraining where $B$ also changes what data is seen when.
5. *Step time flat in $B$* — violated once gradient accumulation is required.

## 3. State of the Art

**Established (independently reproduced).**
- The two-regime picture: below $B_{\text{crit}}$, doubling $B$ nearly halves steps; above it, steps barely fall. Shallue et al. (JMLR 2019) measured this across 6 datasets, 7 architectures, batch sizes to $2^{17}$, with per-arm tuning; McCandlish et al. (2018) fit the same hyperbola.
- $\eta$-$B$ coupling is real and rule-dependent (Goyal et al. 2017 linear scaling + warmup, ImageNet at $B=8192$ matching $B=256$ top-1 to within ~0.1%).
- Increasing $B$ can substitute for decaying $\eta$: Smith et al. (ICLR 2018) matched a decaying-$\eta$ ResNet baseline with a fixed $\eta$ and a ramped $B$, cutting parameter updates roughly 10x (ImageNet-scale, ResNet-50 at $B$ up to 65k).

**Claimed but unablated.**
- That $\mathcal{B}_{\text{simple}}$ is a usable *online controller*. McCandlish et al. show correlation with measured $B_{\text{crit}}$ across tasks; they do not demonstrate a closed-loop schedule beating a tuned constant at matched budget.
- That frontier LLM runs' hand-designed batch ramps are near-optimal. These are engineering choices reported in model cards, not ablated against a constant-$B$ control.

**Benchmark-number-only.** Published LLM batch-size ramps (e.g. GPT-3-style ramps from small to full batch over early tokens) appear as single training configurations with no control arm. Treat as anecdote.

**Theory SOTA** is separate and weaker: convergence-rate analyses give $B$-dependent step counts under bounded-variance assumptions (Bottou, Curtis & Nocedal, *SIAM Review* 2018), and adaptive-sampling methods with norm tests (Byrd et al., *Math. Prog.* 2012) prove convergence, not optimality of the resulting schedule for deep nets.

## 4. What Is Known

- **$B_{\text{crit}}$ grows during training.** Kaplan et al. (2020) fit $B_{\text{crit}}(L)\approx B_\star/L^{1/\alpha_B}$ with $B_\star\approx 2\times 10^8$ tokens and $\alpha_B\approx 0.21$ on autoregressive transformers — a power law in loss, not in parameters. Since $L$ falls monotonically, this predicts a rising $B(t)$.
- **$B_{\text{crit}}$ tracks data, not parameters.** Zhang et al. (2024) hold data fixed and vary model size (85M–1.2B) and vice versa: critical batch size scales with data size and is nearly invariant to model size once data is matched. This contradicts the common practice of setting $B$ from parameter count.
- **Optimizer changes where the wall is.** Zhang et al. (NeurIPS 2019) show momentum and preconditioning move $B_{\text{crit}}$ substantially; comparisons at a single $B$ are not informative about which optimizer is better.
- **Large $B$ costs total compute.** Golmant et al. (2018) report roughly linear growth in total examples-to-target above the knee across CIFAR/ImageNet-scale models — large batch buys latency, not FLOPs.

## 5. What Is Not Known

- **Empirically open (the main gap).** Whether any $B(t)$ beats the best *tuned constant* $B$ on FLOPs-to-target-loss at $\ge$1B-parameter, $\ge$20B-token scale. The experiment is runnable today at maybe $10^4$ GPU-hours per replicated arm. Nobody has published a matched-budget, per-arm-$\eta$-tuned comparison.
- **Empirically open.** Whether $\widehat{\mathcal{B}}_{\text{simple}}$ predicts the *fitted* $B_{\text{crit}}$ within a run well enough to control on, or only correlates across tasks.
- **Theoretically open.** No proof that a time-varying $B$ strictly dominates all constants for non-convex objectives under realistic (non-stationary, anisotropic) $\Sigma$; nor a proof of the converse.
- **Methodologically blocked.** The joint $(B,\eta,$ warmup, weight decay, $\beta_2)$ optimum is not separable, and "the optimal batch size" is not well defined without fixing that tuning protocol. Reported $B_{\text{crit}}$ values are protocol-relative.

## 6. Why It Is Hard

**The obstruction is confounding with the learning-rate schedule, on top of a cost that forbids the ablation.**

- Any $B(t)$ change silently changes effective step size: gradient noise acts as implicit regularization of magnitude $\propto \eta/B$. A schedule that raises $B$ *is* an $\eta/B$ decay. So "increasing batch size helped" and "the learning-rate decay was mistuned in the baseline" produce identical evidence. Distinguishing them requires re-tuning $\eta(\cdot)$ inside each arm — turning a 2-arm comparison into a 2$\times k$-arm sweep.
- Non-identifiability: Smith et al. (2018) is direct evidence that $B$↑ and $\eta$↓ are partly interchangeable, so the schedule is at best identified up to that equivalence.
- Cost: the deciding measurement is a Pareto curve, and each point on it needs a full run to target loss with its own tuned $\eta$. At 1B parameters, 20B tokens, 5 batch sizes, 3 $\eta$ each, 2 seeds — 30+ full runs.
- Proxy-scale results do not transfer: $B_{\text{crit}}$ depends on loss level, so a small-model run never reaches the loss regime where the large run's schedule matters.

## 7. Current Research (as of 2026)

- **Scaling-law-grounded batch/LR prescriptions.** DeepSeek's scaling-law work (2024) fits $B$ and $\eta$ jointly as power laws in compute budget $C$ rather than in $N$ — the closest thing to a principled open-loop schedule in production use.
- **Critical batch size as a function of data.** Kakade-group work (Zhang, Morwani, Vyas et al., 2024) on how $B_{\text{crit}}$ scales in pretraining; the natural sequel is the within-run schedule. *(frontier — verify)*
- **Adaptive-batch optimizers.** AdaScale SGD (Johnson et al., ICML 2020) and gradient-diversity-based rules (Yin et al., AISTATS 2018) adapt $\eta$ to observed noise rather than $B$ — the dual formulation of the same control problem.
- **Feedback controllers using $\widehat{\mathcal{B}}_{\text{simple}}$ in LLM pretraining stacks.** Reported informally by several labs; no matched-budget ablation published. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** 1.4B-parameter decoder-only transformer, 28B tokens (Chinchilla-ratio), one epoch of a public corpus. Sequence length 4096, AdamW, cosine schedule with 1% warmup.

**Arms.**
- *Control:* constant $B\in\{0.5, 1, 2, 4\}$M tokens, each with $\eta\in\{0.5,1,2\}\times$ its scaling-rule value ($\eta\propto\sqrt{B}$ for Adam). 12 runs; take the best.
- *Treatment A (open-loop):* $B(t)$ following Kaplan's $B_{\text{crit}}(L)=B_\star/L^{1/\alpha_B}$ using the run's own smoothed loss, clipped to $[0.5, 8]$M tokens, with $\eta(t)\propto\sqrt{B(t)}$; 3 $\eta$ multipliers.
- *Treatment B (feedback):* $B(t) = \kappa\,\widehat{\mathcal{B}}_{\text{simple}}(t)$, EMA horizon 500 steps, $\kappa\in\{0.5,1,2\}$.
- 2 seeds on the winner of each family.

**Deciding number.** Total training FLOPs to reach held-out loss $L^\star = $ (best control arm's final loss), expressed as ratio $R = C_{\text{sched}}/C_{\text{const}}$. Decision rule: $R < 0.95$ with non-overlapping seed ranges ⇒ schedules win and the problem moves to *partially-solved*. $R \in [0.95, 1.05]$ ⇒ constant-$B$ is adequate at this scale, and the field should stop hand-tuning ramps. Report sequential steps $T$ alongside, since $R>1$ with $T$ down is still a wall-clock win.

**Cost estimate.** $\approx 30$ runs $\times\ 2.4\times10^{21}$ FLOPs $\approx 7\times10^{22}$ FLOPs; order $10^4$ H100-hours. Affordable to one mid-sized lab.

## 9. Key References

- **[Foundational]** L. Bottou, F. Curtis, J. Nocedal. *Optimization Methods for Large-Scale Machine Learning.* SIAM Review 60(2), 2018. — arXiv:1606.04838
- **[Foundational]** R. Byrd, G. Chin, J. Nocedal, Y. Wu. *Sample Size Selection in Optimization Methods for Machine Learning.* Mathematical Programming 134(1), 2012.
- **[Foundational]** S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* 2018. — arXiv:1812.06162
- **[SOTA]** C. Shallue, J. Lee, J. Antognini, J. Sohl-Dickstein, R. Frostig, G. Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[SOTA]** H. Zhang, D. Morwani, N. Vyas, J. Wu, D. Zou, U. Ghai, D. Foster, S. Kakade. *How Does Critical Batch Size Scale in Pre-training?* 2024. — arXiv:2410.21676
- **[SOTA]** J. Kaplan, S. McCandlish, T. Henighan, T. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, D. Amodei. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- P. Goyal, P. Dollár, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, K. He. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. — arXiv:1706.02677
- S. Smith, P.-J. Kindermans, C. Ying, Q. Le. *Don't Decay the Learning Rate, Increase the Batch Size.* ICLR 2018. — arXiv:1711.00489
- G. Zhang, L. Li, Z. Nado, J. Martens, S. Sachdeva, G. Dahl, C. Shallue, R. Grosse. *Which Algorithmic Choices Matter at Which Batch Sizes? Insights from a Noisy Quadratic Model.* NeurIPS 2019. — arXiv:1907.04164
- S. Malladi, K. Lyu, A. Panigrahi, S. Arora. *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* NeurIPS 2022. — arXiv:2205.10287
- Z. Li, S. Malladi, S. Arora. *On the Validity of Modeling SGD with Stochastic Differential Equations.* NeurIPS 2021. — arXiv:2102.12470
- T. Johnson, P. Agrawal, H. Gu, C. Guestrin. *AdaScale SGD: A User-Friendly Algorithm for Distributed Training.* ICML 2020. — arXiv:2007.05105
- N. Golmant, N. Vemuri, Z. Yao, V. Feinberg, A. Gholami, K. Rothauge, M. Mahoney, J. Gonzalez. *On the Computational Inefficiency of Large Batch Sizes for Stochastic Gradient Descent.* 2018. — arXiv:1811.12941
- **[Survey]** DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* 2024. — arXiv:2401.02954

## 10. Worked Example

Take Kaplan's fit at face value on the 1.4B/28B run above. Loss falls from $L\approx 4.0$ early to $L\approx 2.2$ at the end. With $B_\star = 2\times 10^8$ tokens and $\alpha_B = 0.21$:

$$B_{\text{crit}}(4.0) = \frac{2\times10^8}{4.0^{1/0.21}} = \frac{2\times10^8}{4.0^{4.76}} \approx \frac{2\times10^8}{659} \approx 3.0\times10^5 \text{ tokens},$$
$$B_{\text{crit}}(2.2) = \frac{2\times10^8}{2.2^{4.76}} \approx \frac{2\times10^8}{35.6} \approx 5.6\times10^6 \text{ tokens}.$$

So the prescription says $B$ should rise about **19x** over the run — from 0.3M to 5.6M tokens. A constant $B=1$M is above critical for the first stretch (wasting data) and below it for the last (wasting steps). That is a large enough mismatch that the schedule should be visibly better.

Now the obstruction. Ramping $B$ by 19x while holding $\eta$ fixed cuts the noise term $\eta/B$ by the same 19x. A cosine schedule over the same run cuts $\eta$ from peak to $0.1\times$ peak, so $\eta/B$ falls 10x from the LR alone. Under the $\eta\propto\sqrt{B}$ Adam rule the treatment arm's $\eta$ *rises* $\sqrt{19}\approx 4.4$x while cosine pulls it down 10x — net $\eta$ trajectory down 2.3x instead of 10x. The treatment arm is therefore not "the same run with a batch schedule"; it is a run with a materially different effective-LR trajectory.

Consequence: if Treatment A reaches $L^\star$ in $0.9\times$ the FLOPs, the honest reading is *either* "the batch schedule tracked $B_{\text{crit}}$" *or* "cosine decays $\eta$ too aggressively for this run and the flatter effective schedule was better." Only the 12-run control sweep — which contains a constant-$B$ arm with a $2\times$ $\eta$ multiplier and hence a comparable effective-LR trajectory — separates them. If the best control arm is the flat-LR one, the batch schedule explained nothing. This is why the experiment in §8 sweeps $\eta$ *inside* every arm rather than fixing a single "standard" recipe, and why the single deciding number must be FLOPs-to-$L^\star$ against the sweep's *best* control, not against a default configuration.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*