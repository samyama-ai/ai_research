---
id: 03-training-dynamics/lr-rewarming-after-decay
title: "Learning Rate Rewarming After Decay"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning Rate Rewarming After Decay

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/lr-rewarming-after-decay` · **Status:** empirically-open

## 1. Problem Statement

A pretraining run ends by decaying the learning rate to near zero. Later, someone wants to train the same checkpoint further — more tokens, a new domain, a new language. Doing so requires raising the learning rate again ("rewarming"). Every observed rewarm produces a **loss spike** followed by a recovery, and the open question is whether the run ever fully recovers.

- **Input:** a checkpoint $\theta_T$ produced by a schedule that decayed to $\eta_{\min}\approx 0$, plus a token budget $B$ of new data from distribution $\mathcal{D}_2$.
- **Output:** a schedule $\eta(t)$ on $[T, T+B]$, plus optional replay of $\mathcal{D}_1$.
- **Decision predicate:** does there exist a rewarm schedule whose final loss on $\mathcal{D}_1\cup\mathcal{D}_2$ matches the loss of a single from-scratch run of length $T+B$ under one schedule, to within the seed noise floor?

Three variants, of very different difficulty:

- **Measurement:** is the post-rewarm deficit *persistent* or merely *slow*? Distinguishing "never recovers" from "recovers after $10\times$ the budget anyone ran" needs a defined extrapolation, and there is none.
- **Method:** given the deficit exists, which knob removes it — max LR, rewarm length, optimizer-state reset, replay fraction, or skipping the decay entirely (WSD-style branching)?
- **Theory:** does a decayed-then-rewarmed trajectory reach a different basin than a monotone one, or the same one later? No proof either way for non-convex deep networks.

## 2. Formal Setting

Loss $\mathcal{L}_i(\theta)=\mathbb{E}_{x\sim\mathcal{D}_i}[-\log p_\theta(x)]$, measured as mean next-token cross-entropy in nats on a held-out shard of $\ge 10^7$ tokens (so the standard error is $\lesssim 10^{-3}$ nats and below the seed spread).

Baseline schedule over $T$ steps, cosine form:
$$\eta_{\text{base}}(t)=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right).$$

Rewarm schedule on $t\in[T,T+B]$, parameterized by peak $\eta_2$, warmup length $w$, and final floor:
$$\eta_{\text{re}}(t)=\begin{cases}\eta_{\min}+\frac{t-T}{w}(\eta_2-\eta_{\min}) & t-T\le w\\[2pt] \text{decay}(\eta_2,\,t-T-w,\,B-w) & \text{otherwise.}\end{cases}$$

Quantities as measured:

- **Spike depth** $S=\max_{t\in[T,T+B]}\big(\mathcal{L}_1(\theta_t)-\mathcal{L}_1(\theta_T)\big)$, in nats, on $\mathcal{D}_1$ validation. This is *forgetting during rewarm*, not the transfer target.
- **Recovery time** $\tau=\min\{t: \mathcal{L}_1(\theta_t)\le \mathcal{L}_1(\theta_T)\}$, in tokens; $\tau=\infty$ if not reached within $B$.
- **Terminal deficit** $\Delta=\mathcal{L}(\theta_{T+B}^{\text{rewarm}})-\mathcal{L}(\theta_{T+B}^{\text{single}})$ against a from-scratch control of identical total tokens, identical data order, identical seed budget.
- **Seed floor** $\sigma$: standard deviation of final loss across $\ge 3$ seeds of the control. A claim of "$\Delta>0$" is only meaningful at $|\Delta|>2\sigma$; typical $\sigma$ at 1B scale is $\sim 3\times10^{-3}$ nats.

Assumptions, and which break:

- *Stationary data.* Violated by construction — the whole point is $\mathcal{D}_2\neq\mathcal{D}_1$.
- *Optimizer state carried across the boundary.* Often violated: many continual runs reset Adam's $m,v$ and step counter, which changes the effective LR by the bias-correction factor $\sqrt{1-\beta_2^t}/(1-\beta_1^t)$ and confounds the rewarm with a warm restart of the preconditioner.
- *Fixed batch size and sequence length.* Frequently violated in practice; batch changes rescale the noise temperature $\eta/|\mathcal{B}|$ and are not separated from the LR effect in most reports.
- *Loss is the objective.* Downstream task accuracy is what practitioners buy, and it is not a monotone function of validation loss across distribution shifts.

## 3. State of the Art

**Established (ablated, multi-scale):**

- Gupta et al. (2023), *Continual Pre-Training of Large Language Models: How to (re)warm your model?* — rewarming to a high peak causes a transient rise in $\mathcal{D}_1$ loss, and the peak LR trades off $\mathcal{D}_1$ forgetting against $\mathcal{D}_2$ adaptation. Pythia-scale (410M, 2.8B), Pile → SlimPajama.
- Ibrahim et al. (2024, TMLR), *Simple and Scalable Strategies to Continually Pre-train LLMs* — rewarm + re-decay combined with a small replay fraction of $\mathcal{D}_1$ closes most of the gap to a full retrain at 405M and 10B parameters, on both a weak shift (Pile → SlimPajama) and a strong one (Pile → German). This is the strongest positive result in the literature.
- Hägele et al. (2024), *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* — a constant-LR trunk plus a short cooldown matches cosine at matched compute, and the trunk checkpoint can be branched repeatedly. Ablated at 210M–1B on SlimPajama.
- Hu et al. (2024), *MiniCPM* — WSD schedules; the decay phase produces a sharp loss drop, and the stable-phase checkpoint is the natural continuation point. Deployed at 1.2B/2.4B.

**Claimed but unablated:** that rewarming from a *fully decayed* checkpoint is strictly worse than branching from a *stable-phase* checkpoint at the same token count. This is widely assumed and consistent with WSD practice, but no published run holds tokens, data order, and optimizer state fixed while varying only the branch point.

**Benchmark-number-only:** most industrial continual-pretraining reports (domain-adapted code and math models) give downstream scores after rewarming with no from-scratch control at matched tokens. They establish the recipe is usable, not that $\Delta=0$.

## 4. What Is Known

- Spikes are real and reproducible. At 405M–10B, rewarming to the original $\eta_{\max}$ raises $\mathcal{D}_1$ loss by roughly $0.1$–$0.3$ nats within the first few hundred million tokens, with the larger spikes at the larger peak LR (Gupta et al. 2023; Ibrahim et al. 2024).
- Recovery is usual, not universal-in-budget. With $\sim 5\%$ replay of $\mathcal{D}_1$, $\mathcal{D}_1$ loss returns to baseline and $\mathcal{D}_2$ loss approaches the retrained model within a few hundredths of a nat, at 10B on a 100B-token continuation (Ibrahim et al. 2024).
- Warmup length matters less than peak LR. Gupta et al. report the peak dominates; varying warmup duration over the plausible range moves final loss by a small fraction of the peak effect.
- The decay phase carries a large, real gain. In WSD runs the cooldown drops loss discontinuously relative to the stable-phase trace (Hu et al. 2024; Hägele et al. 2024), so a decayed checkpoint sits in a measurably better place than its stable-phase sibling — which is exactly why rewarming out of it feels like a loss.
- Schedule shape is partly explained by convex theory. Schaipp et al. (ICML 2025) show a convex non-smooth suboptimality bound predicts the observed shape of cosine and WSD curves, including the cooldown drop, at 124M–210M scale.
- Warm restarts are not new. SGDR (Loshchilov & Hutter, ICLR 2017) showed cyclic restarts help image classifiers; that result has never transferred cleanly to single-epoch LLM pretraining.

## 5. What Is Not Known

- **Empirically open (the core gap):** whether $\Delta > 2\sigma$ persists at $\ge 7$B parameters with $B \ge 300$B continuation tokens, matched data order, and a from-scratch control. Every published control is at $\le 10$B params with $B \lesssim 100$B tokens. The experiment is runnable today; it costs money, not new ideas.
- **Empirically open:** the branch-point question — decayed checkpoint vs. stable-phase checkpoint at identical total tokens.
- **Methodologically blocked:** "irreversible" is undefined. Nobody has specified the extrapolation that separates a permanent deficit from one that closes at $10\times$ budget, so "does not recover" is currently a statement about a budget, not a model.
- **Methodologically blocked:** the optimizer-state confound. Rewarming and resetting Adam moments are almost always changed together; the two effects are not identified in any published ablation.
- **Theoretically open:** whether a decay–rewarm cycle changes basin. The river-valley picture (Wen et al. 2024) predicts the decay moves the iterate off the river onto a valley floor and rewarming must climb back, but this is a mechanism sketch, not a theorem; no non-convex result bounds $\Delta$ from below or above.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a control that costs as much as the treatment**. To claim $\Delta>0$ you need a from-scratch run of $T+B$ tokens, which is the full cost of the thing continual pretraining exists to avoid. So the one arm that would settle the question is the arm nobody has budget to run at frontier scale, and its absence is not an oversight — it is the economics of the method.

Secondary: at least four variables move together at the boundary (peak LR, warmup length, optimizer-state reset, data mixture), and the reported effect sizes ($10^{-2}$ nats) are within a small multiple of the seed floor ($3\times10^{-3}$ nats) at the scales where controls exist. Underpowered comparisons across a four-dimensional confound produce exactly the literature we have: consistent directions, unreliable magnitudes.

## 7. Current Research (as of 2026)

- **WSD/branchable trunks as the standard answer.** Keep a constant-LR trunk, cool down copies. Sidesteps rewarming rather than solving it; now default in several open recipes (MiniCPM, OLMo-2 style two-stage runs).
- **Schedule-free and cooldown-free optimizers.** Defazio et al., *The Road Less Scheduled* (NeurIPS 2024) — if there is no decay, there is nothing to rewarm from. Matched or near-matched cosine at moderate scale; frontier-scale parity is *(frontier — verify)*.
- **Annealing-aware scaling laws.** Fitting loss as a function of the LR-schedule area/momentum so a continuation's final loss is predictable without running it (Tissue et al. 2024; Schaipp et al. 2025 for the convex link).
- **Loss-landscape mechanism work.** River-valley and warmup-mechanism analyses (Wen et al. 2024; Kalra & Barkeshli, NeurIPS 2024) aiming to predict spike depth from curvature at $\theta_T$ — sharpness $\lambda_{\max}$ at the decayed checkpoint as a spike predictor is *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale:** 1.4B params, Llama-style, batch $2^{21}$ tokens, $T=100$B tokens on a fixed mix, then $B=100$B tokens on a shifted mix (e.g. 80% new domain, 20% replay). Three seeds per arm. Roughly $6\times10^{21}$ FLOPs per arm; four arms plus control $\approx$ 5 arms $\times$ 3 seeds.

**Arms (everything else held fixed — data order, batch, sequence length):**
1. **Control:** single from-scratch run, 200B tokens, one cosine schedule.
2. **Rewarm-from-decayed:** decay to $\eta_{\min}=0.1\eta_{\max}$ at 100B, rewarm to $\eta_2=\eta_{\max}$ over 1B tokens, re-decay. Adam state **preserved**.
3. **Same as (2), Adam state reset.** Isolates the optimizer confound.
4. **Branch-from-stable:** WSD trunk, branch at 100B without cooling down, cool down at 200B.
5. **Rewarm to $\eta_2=0.3\eta_{\max}$**, state preserved. Isolates peak LR.

**The deciding number:** $\Delta = \mathcal{L}_{\text{mix}}(\theta_{200\text{B}}^{\text{arm}}) - \mathcal{L}_{\text{mix}}(\theta_{200\text{B}}^{\text{control}})$ in nats on a 20M-token held-out mixture shard, with the seed floor $\sigma$ estimated from the control's three seeds. Decision rule: if arm 2 has $\Delta > 2\sigma$ and arm 4 has $\Delta \le 2\sigma$, decay-then-rewarm carries a real, avoidable cost and trunk-branching is the fix. If both are within $2\sigma$, the spike is transient and the problem is closed at this scale — then repeat once at 7B to check scale-dependence.

**Also report:** spike depth $S$ and recovery time $\tau$ per arm, plus $\lambda_{\max}$ (top Hessian eigenvalue, power iteration on a 4M-token subsample) at the branch point, to test whether curvature predicts $S$.

## 9. Key References

- **[Foundational]** Ilya Loshchilov, Frank Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR, 2017. — arXiv:1608.03983
- **[Foundational]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, Irina Rish. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[SOTA]** Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L. Richter, Quentin Anthony, Eugene Belilovsky, Irina Rish, Timothée Lesort. *Continual Pre-Training of Large Language Models: How to (re)warm your model?* Workshop paper / arXiv preprint, 2023.
- **[SOTA]** Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro von Werra, Martin Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS, 2024. — arXiv:2405.18392
- **[SOTA]** Shengding Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* COLM, 2024. — arXiv:2404.06395
- **[SOTA]** Aaron Defazio, Xingyu (Alice) Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky. *The Road Less Scheduled.* NeurIPS, 2024. — arXiv:2405.15682
- **[Theory]** Fabian Schaipp, Alexander Hägele, Adrien Taylor, Umut Şimşekli, Francis Bach. *The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training.* ICML, 2025.
- **[Theory]** Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, Tengyu Ma. *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective.* Preprint, 2024.
- **[Theory]** Dayal Singh Kalra, Maissam Barkeshli. *Why Warmup the Learning Rate? Underlying Mechanisms and Improvements.* NeurIPS, 2024.

## 10. Worked Example

Take a 1.4B model, $\eta_{\max}=3\times10^{-4}$, cosine to $\eta_{\min}=3\times10^{-5}$ at 100B tokens, ending at $\mathcal{L}_1=2.410$ nats. Continue on a code-heavy mix for 100B tokens, rewarming to $\eta_2=3\times10^{-4}$ over 1B tokens.

Typical trace, using effect sizes from the 405M/10B literature:

```
tokens (B)   L_D1     L_D2
100.0        2.410    2.980   <- branch point
101.0        2.585    2.845   spike S = 0.175 nats
104.0        2.470    2.760
112.0        2.408    2.690   recovery tau ~ 12B tokens
200.0        2.402    2.531   rewarm arm, final
200.0        2.389    2.518   from-scratch control, final
```

$\Delta_{\mathcal{D}_1}=0.013$ nats, $\Delta_{\mathcal{D}_2}=0.013$ nats. With three control seeds giving $\sigma=0.003$, $\Delta=4.3\sigma$ — nominally significant.

Now the obstruction. That $0.013$ nats is about $0.5\%$ of the loss, and the *same* $0.013$ can be produced by: (a) a genuine basin difference; (b) resetting Adam's step counter, which at $\beta_2=0.95$ inflates the effective step for the first $\sim 60$ updates; (c) the control seeing the code mix interleaved for all 200B tokens instead of the last 100B, which is a data-order difference, not a schedule difference. Arms 2 and 3 of §8 separate (a) from (b); nothing separates (a) from (c) without also running a from-scratch control on the *sequential* mixture, which doubles the control cost.

So the measured deficit is real and reproducible, and its cause is not identified. That, not the magnitude, is what keeps the problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*