---
id: 30-synthetic-data/sim-to-real-gap-prediction
title: "Sim-to-Real Gap Prediction Without Real Rollouts"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sim-to-Real Gap Prediction Without Real Rollouts

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/sim-to-real-gap-prediction` · **Status:** open

## 1. Problem Statement

Given a policy $\pi$ trained on synthetic data from a simulator, predict how much worse it will perform on the physical system — **before executing it there**.

- **Input:** a simulator $\mathcal{S}$ (possibly a randomization distribution over simulators), a policy $\pi$, and optionally a fixed offline real dataset $D$ collected under *other* policies (logs, teleoperation, video). No online real rollouts of $\pi$.
- **Output:** an estimate $\hat{\Delta}(\pi)$ of the real-minus-sim performance difference, or a lower confidence bound on real return.
- **Solved** means: over a held-out family of policies and tasks, $\hat{\Delta}$ is calibrated (bounded absolute error) or at least correctly *ranks* policies by real performance, with the guarantee holding for policies not used to fit the predictor.

Three variants, of very different difficulty:

- **Measurement:** define a gap statistic that is stable, comparable across labs, and not dominated by which real trials happened to be run. Largely unsolved; there is no agreed statistic.
- **Method:** build a predictor. Runnable today; nobody has evaluated one at the scale that would settle it.
- **Theory:** prove a bound on $|\hat\Delta - \Delta|$ from sim-side quantities plus assumptions on $D$. Known to be impossible without assumptions that are false in practice (§4).

The *ranking* variant is strictly easier than the *calibration* variant and is what most deployments actually need — pick the best of $k$ candidate policies without burning real trials.

## 2. Formal Setting

Real system: MDP $M^\star=(\mathcal{S},\mathcal{A},P^\star,r^\star,\gamma,\rho_0^\star)$. Simulator family: $\{M_\xi\}$ indexed by $\xi\sim p_\phi$ (masses, frictions, latencies, sensor noise, textures). Return $J_M(\pi)=\mathbb{E}\big[\sum_t \gamma^t r(s_t,a_t)\big]$.

**Gap.**
$$\Delta(\pi) \;=\; J_{M^\star}(\pi)\;-\;\mathbb{E}_{\xi\sim p_\phi}\big[J_{M_\xi}(\pi)\big].$$

*Measured as:* $\hat J_{M^\star}$ from $n$ real episodes (typically $n=10$–$50$ per policy, binary success $\Rightarrow$ standard error $\le 0.5/\sqrt{n}$, i.e. $\ge 0.07$ at $n=50$); $\hat J_{\text{sim}}$ from $m \gg n$ simulated episodes with $\xi$ resampled per episode, so its Monte-Carlo error is negligible and *all* the noise in $\hat\Delta$ is real-side.

**Predictor.** $\hat\Delta = f(\mathcal{S},\pi,D)$, with $D$ containing zero on-policy real transitions of $\pi$. Target guarantee:
$$\Pr\big[\,|\hat\Delta(\pi)-\Delta(\pi)|\le\epsilon\,\big]\ge 1-\delta \quad \text{for all } \pi\in\Pi.$$

**Ranking target.** For candidates $\pi_1..\pi_k$, Spearman $\rho$ between $\{\hat J_{\text{sim}}+\hat\Delta\}$ and $\{\hat J_{M^\star}\}$. Kadian et al. call this the Sim-vs-Real Correlation Coefficient (SRCC).

**Simulation lemma (the only clean handle).** If $\sup_{s,a}\|P^\star(\cdot|s,a)-P_\xi(\cdot|s,a)\|_{TV}\le\varepsilon_P$ and $\|r^\star-r_\xi\|_\infty\le\varepsilon_r$, then
$$|J_{M^\star}(\pi)-J_{M_\xi}(\pi)| \;\le\; \frac{\varepsilon_r}{1-\gamma}+\frac{\gamma R_{\max}\,\varepsilon_P}{(1-\gamma)^2}.$$
The $(1-\gamma)^{-2}$ makes this vacuous at realistic horizons: $\gamma=0.99$, $R_{\max}=1$, $\varepsilon_P=10^{-2}$ gives a bound of $99$ on a return scale of $100$.

**Assumptions, and their status:**

| Assumption | Status |
|---|---|
| $M^\star$ lies in $\operatorname{supp}(p_\phi)$ (realizability) | **Violated.** Contact, cable dynamics, deformables and lighting are structurally absent, not mis-parameterized. |
| $\varepsilon_P$ uniform over $(s,a)$ | **Violated.** Error concentrates on contact/impact events, which are exactly where policies dwell. |
| $D$ covers $\pi$'s occupancy measure | **Violated by construction** — if it did, this would be ordinary off-policy evaluation. |
| $M^\star$ stationary across the eval window | Violated over days (wear, calibration drift, lighting). |

## 3. State of the Art

**Theory SOTA.** Domain-adaptation bounds — Ben-David et al. (*Machine Learning*, 2010) $\mathcal{H}\Delta\mathcal{H}$-divergence; Mansour–Mohri–Rostamizadeh (COLT 2009) discrepancy distance — bound target risk by source risk plus a divergence plus an *unmeasurable* joint-optimal term $\lambda$. Ben-David et al. (AISTATS 2010) give matching impossibility results: with unlabeled target data alone, no adaptation guarantee exists under covariate shift + small-$\lambda$ alone. These are supervised-learning bounds; the sequential-control analogue with compounding error is the simulation lemma above. **Established.**

Muratore, Gienger & Peters (*IEEE TPAMI*, 2021) define the **Simulation Optimization Bias**, prove it is non-negative (optimizing on a finite set of sampled domains overstates expected return), and give SPOTA, which estimates an *optimality gap* using only simulation. **Established for within-$p_\phi$ transfer; gives no bound on transfer outside $\operatorname{supp}(p_\phi)$** — which is the case that matters.

**Empirical SOTA.** Kadian et al. (*RA-L*, 2020) is the reference measurement: they show simulator rank order can be near-useless for real PointGoal navigation and that tuning the simulator (actuation noise, collision response) lifts SRCC to $\approx 0.84$. Chebotar et al. (ICRA 2019) and Ramos et al. (BayesSim, RSS 2019) infer $p_\phi$ from real trajectories — strong results, but they *consume* real rollouts, so they solve a different problem. OpenAI's Automatic Domain Randomization (2019) trained without real rollouts and transferred, but published no gap *prediction*.

**Claimed but unablated:** that DR-ensemble return variance $\sigma_\xi$ predicts the gap; that a learned "reality score" discriminator predicts it; that sim-and-real co-training reduces it *predictably*. These appear as single-task demonstrations without policy-held-out evaluation *(frontier — verify)*. Where a number exists (SRCC $0.84$), it exists as one benchmark number on one robot, one task, one lab.

## 4. What Is Known

- **The gap is large and task-dependent, not a constant offset.** Acosta, Yang & Posa (*RA-L*, 2022) benchmarked Drake, MuJoCo, Bullet and Dart against 500+ real planar-impact trajectories; no simulator's tuned parameters generalized across impact conditions, and post-impact velocity error stayed comparable to the between-simulator spread. Scale: single object, thousands of trials.
- **Contact models disagree with reality at the level a policy can exploit.** Fazeli et al. (ICRA 2017) show common planar-impact models mispredict outcomes on real pushes of a single rigid object.
- **Simulator rank order can invert.** Kadian et al. (2020): default-sim ranking of navigation policies correlated poorly with real; SRCC $\approx 0.84$ only after deliberate simulator tuning. Scale: ~6 policies, LoCoBot, one apartment.
- **Randomization reduces the gap but widens the variance.** Peng et al. (ICRA 2018), Tobin et al. (IROS 2017): dynamics/visual randomization enables zero-shot transfer, at a documented cost in mean sim return.
- **Optimizing on sampled domains is optimistic in expectation** (Muratore et al., TPAMI 2021) — the sign of the bias is proven, the magnitude is not bounded.
- **Off-policy evaluation gives high-confidence bounds only under coverage** (Thomas & Brunskill, AAAI 2015). Importance weights blow up exactly when $\pi$ leaves $D$'s support.

## 5. What Is Not Known

- **Theoretically open:** whether any nontrivial bound on $\Delta(\pi)$ is obtainable from sim-side statistics plus *off-policy* real data under a structural assumption weaker than realizability (e.g. bounded model error on a *policy-independent* state cover). No proof either way. The known impossibility results rule out the assumption-free case, not this one.
- **Empirically open:** whether cheap sim-side predictors (DR return variance, ensemble disagreement, contact-event density, observation-space discriminator score) rank policies by real performance across $\ge 20$ policies $\times$ $\ge 5$ tasks. Every ingredient exists; the experiment has not been run at that scale by anyone.
- **Methodologically blocked:** the gap statistic itself. $\Delta$ depends on $p_\phi$, on the real trial protocol, on reset distribution, and on $n$. Two labs reporting "a 30-point sim-to-real gap" are not reporting the same quantity, and no benchmark fixes the protocol.

## 6. Why It Is Hard

**Non-identifiability of the on-policy real occupancy measure.** $\Delta(\pi)$ is a functional of $d^\pi_{M^\star}$ — the state distribution $\pi$ induces on the real system. Offline data $D$ from other policies pins down $P^\star$ only where those policies went. A sim-trained policy's failure modes are precisely the states it reaches *because* the simulator said they were safe, which are by construction under-covered in $D$. No amount of $D$ identifies $\Delta$ without an extrapolation assumption, and the extrapolation assumption is the thing in question.

Compounding this: **absent ground truth at usable precision.** Establishing $\Delta$ to $\pm 0.05$ success rate needs $\sim 100$ real trials per policy; a 20-policy study is 2,000 robot trials — weeks of hardware time with drift over the window. That cost is why the empirically-open question stays unrun, and why published gap numbers carry error bars wider than the effects being compared.

## 7. Current Research (as of 2026)

- **Real-to-sim system identification** (BayesSim lineage; Ramos, Fox, NVIDIA) — inverts the problem, but needs real rollouts. Recent differentiable-simulation variants push toward using only passive video *(frontier — verify)*.
- **Sim-and-real co-training** for vision-based manipulation (Stanford/Toyota Research Institute lineage, 2025) — reports that small real fractions recover most of the gap; whether the *residual* gap is predictable is untested *(frontier — verify)*.
- **Conformal and distribution-free bounds on transfer**, e.g. the Sim-to-Lab-to-Real line (Hsu et al., *Artificial Intelligence*, 2023), giving probabilistic safety bounds via an intermediate "lab" domain rather than pure zero-real prediction.
- **Simulator validation benchmarks** (Posa group, Penn; Fazeli/MIT) — measuring $\varepsilon_P$ directly on contact-rich data instead of inferring it from policy return.
- **Robust/adversarial training** (EPOpt, RARL lineage) — reduces worst-case gap rather than predicting it.

## 8. Concrete Next Experiment

**Question:** does any zero-real-rollout predictor beat "trust the simulator" at ranking policies?

- **Scale.** $k=24$ policies (4 algorithms $\times$ 3 seeds $\times$ 2 randomization widths) on $T=4$ tasks (peg insertion, cloth flattening, PointGoal navigation, non-prehensile push). $n=100$ real trials per policy per task $= 9{,}600$ real trials. Two robots, two labs, trials interleaved across policies to absorb drift.
- **Predictors (sim-only):** (a) DR return standard deviation $\sigma_\xi$; (b) ensemble-of-simulators disagreement (MuJoCo/Bullet/Drake); (c) contact-event density along $\pi$'s sim rollouts; (d) discriminator score of sim observations vs. an offline real image corpus.
- **Control arm.** $\hat\Delta \equiv 0$ — rank by mean sim return alone. This is the arm every paper implicitly uses and almost none reports.
- **Deciding number.** Leave-one-task-out Spearman $\rho$ between predicted and real rank, pooled over $k=24$. **A predictor wins if $\rho - \rho_{\text{control}} \ge 0.25$ with a 95% bootstrap CI excluding 0 on a task it was not fit on.** Secondary: mean absolute calibration error $|\hat\Delta-\Delta|$; report it, and expect it to be $\ge 0.15$ even when ranking succeeds.

Cost estimate: ~3 robot-months. That is the entire reason this has not been run.

## 9. Key References

- **[Foundational]** S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, J. W. Vaughan. *A Theory of Learning from Different Domains.* Machine Learning 79(1–2), 2010.
- **[Foundational]** S. Ben-David, T. Lu, T. Luu, D. Pál. *Impossibility Theorems for Domain Adaptation.* AISTATS, 2010.
- **[Foundational]** Y. Mansour, M. Mohri, A. Rostamizadeh. *Domain Adaptation: Learning Bounds and Algorithms.* COLT, 2009. — arXiv:0902.3430
- **[Foundational]** J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS, 2017. — arXiv:1703.06907
- **[SOTA]** A. Kadian, J. Truong, A. Gokaslan, A. Clegg, E. Wijmans, S. Lee, M. Savva, S. Chernova, D. Batra. *Sim2Real Predictivity: Does Evaluation in Simulation Predict Real-World Performance?* IEEE RA-L, 2020. — arXiv:1912.06321
- **[SOTA]** F. Muratore, M. Gienger, J. Peters. *Assessing Transferability from Simulation to Reality for Reinforcement Learning.* IEEE TPAMI 43(4), 2021.
- **[SOTA]** B. Acosta, W. Yang, M. Posa. *Validating Robotics Simulators on Real-World Impacts.* IEEE RA-L, 2022.
- **[SOTA]** Y. Chebotar, A. Handa, V. Makoviychuk, M. Macklin, J. Issac, N. Ratliff, D. Fox. *Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience.* ICRA, 2019.
- **[SOTA]** F. Ramos, R. Carvalhaes Possas, D. Fox. *BayesSim: Adaptive Domain Randomization via Probabilistic Inference for Robotics Simulators.* RSS, 2019.
- **[Related]** X. B. Peng, M. Andrychowicz, W. Zaremba, P. Abbeel. *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.* ICRA, 2018.
- **[Related]** P. S. Thomas, E. Brunskill. *High-Confidence Off-Policy Evaluation.* AAAI, 2015.
- **[Survey]** F. Muratore, F. Ramos, G. Turk, W. Yu, M. Gienger, J. Peters. *Robot Learning from Randomized Simulations: A Review.* Frontiers in Robotics and AI, 2022.

## 10. Worked Example

Six policies on a peg-insertion task. Sim success is the DR-average over 1,000 domains; $\sigma_\xi$ is the across-domain standard deviation; the predictor is $\hat\Delta=-\sigma_\xi$. Real success is over $n=20$ trials each.

| Policy | Sim succ. | $\sigma_\xi$ | Predicted real | Real succ. |
|---|---|---|---|---|
| A | 0.95 | 0.21 | 0.74 | 0.30 |
| B | 0.92 | 0.06 | 0.86 | 0.75 |
| C | 0.90 | 0.18 | 0.72 | 0.25 |
| D | 0.88 | 0.05 | 0.83 | 0.80 |
| E | 0.85 | 0.09 | 0.76 | 0.55 |
| F | 0.80 | 0.04 | 0.76 | 0.70 |

Spearman against real: sim-only ranking gives $\rho = 1 - \frac{6(44)}{6(35)} = -0.26$ — **anticorrelated**; the best-in-sim policy is the second-worst in reality. The $\sigma_\xi$ predictor gives $\sum d^2 = 2.5$, $\rho = 1 - \frac{15}{210} = +0.93$.

Now the obstruction, in two lines:

1. **Ranking succeeds, calibration fails.** Policy A's predicted gap is $0.21$; its actual gap is $0.95-0.30=0.65$, off by $3.1\times$. Fitting the scale $k$ in $\hat\Delta=-k\sigma_\xi$ requires real rollouts — the resource the problem forbids. Without $k$, you cannot answer "is A good enough to ship?", only "is A better than C?".
2. **The evidence is thinner than it looks.** At $n=20$, each real number has standard error $\approx 0.11$. With $k=6$ policies, the standard error on $\rho$ is roughly $1/\sqrt{k-1}=0.45$; a bootstrap CI on $\rho=0.93$ plausibly spans $[0.3, 1.0]$. The $-0.26$ and the $+0.93$ are not reliably distinguishable at this scale.

This is why §8 specifies $k=24$ and $n=100$: the effect is measurable, but not at the sample sizes robotics papers currently report.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*