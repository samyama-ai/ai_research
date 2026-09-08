---
id: 35-world-models/credit-assignment-differentiable-rollouts
title: "Credit Assignment Through Differentiable Model Rollouts"
topic: 35-world-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Credit Assignment Through Differentiable Model Rollouts

> **Topic:** World Models & Planning · **ID:** `35-world-models/credit-assignment-differentiable-rollouts` · **Status:** partially-solved

## 1. Problem Statement

A learned world model is differentiable. So you can, in principle, unroll a policy inside it for $H$ steps, sum the predicted rewards, and backpropagate to the policy parameters — getting an *analytic* (pathwise) policy gradient instead of a score-function estimate. The promise is a variance reduction of orders of magnitude: the gradient tells you which direction to move each action, not just whether the return was good.

In practice the estimator degrades as $H$ grows. Gradients through long rollouts explode or vanish, and — worse — a low-variance gradient of a *wrong* model is a confidently wrong direction.

Three variants, of different difficulty:

- **Measurement.** Given a learned model and a policy, decide whether the analytic gradient through $H$ steps is a *better descent direction for true environment return* than a zeroth-order estimator at matched compute. Requires a definition of "better" that survives the fact that the true gradient is unavailable.
- **Method.** Build an estimator that keeps the variance advantage of the pathwise gradient while bounding the bias contributed by model error and by chaotic sensitivity. Solving it means: a rule for choosing $H$ (or a mixture) that is computable from quantities the learner already has, and that beats hand-tuned $H$ across ≥3 task families.
- **Theory.** Prove a bias–variance decomposition of the $H$-step analytic gradient in terms of (i) model error and (ii) the Jacobian spectrum of the closed-loop dynamics, tight enough to predict the empirically observed optimal $H$.

## 2. Formal Setting

MDP $(\mathcal{S},\mathcal{A},p,r,\gamma)$; policy $\pi_\theta$; learned model $\hat p_\phi$ with reparameterized transition $s_{t+1} = f_\phi(s_t, a_t, \epsilon_t)$, $\epsilon_t \sim \mathcal{N}(0,I)$, and $a_t = \pi_\theta(s_t, \eta_t)$. The $H$-step surrogate objective, bootstrapped by a learned value $V_\psi$:

$$\hat J_H(\theta) = \mathbb{E}\Big[\sum_{t=0}^{H-1}\gamma^t \hat r_\phi(s_t,a_t) + \gamma^H V_\psi(s_H)\Big].$$

The analytic gradient factors through a product of Jacobians:

$$\nabla_\theta \hat J_H = \sum_{t}\gamma^t \Big(\tfrac{\partial \hat r}{\partial s_t}\Big)^\top \prod_{k<t} A_k \, \tfrac{\partial s_1}{\partial \theta}, \qquad A_k = \tfrac{\partial f_\phi}{\partial s_k} + \tfrac{\partial f_\phi}{\partial a_k}\tfrac{\partial \pi_\theta}{\partial s_k}.$$

Measured quantities:

- **Closed-loop Lyapunov proxy** $\lambda_H = \frac{1}{H}\log \|\prod_{k<H} A_k\|_2$, estimated by power iteration on a batch of rollouts (no explicit Jacobian materialization needed; 10 Hessian-vector-style products suffice for $\|\cdot\|_2$ to 5%).
- **Empirical gradient variance** $\mathrm{Var}[\hat g] = \frac{1}{B-1}\sum_i \|\hat g_i - \bar g\|^2$ over $B$ independent minibatches.
- **Bias** $b_H = \|\mathbb{E}[\hat g_H] - g^\star\|$ where $g^\star$ is the true-environment policy gradient. $g^\star$ is *not observable*; it is approximated by a high-$B$ REINFORCE/ES estimate in the real environment ($B \ge 10^4$ rollouts), which is itself noisy. This substitution is the load-bearing assumption of every empirical claim in this problem.
- **Cosine alignment** $\rho_H = \cos(\hat g_H, g^\star)$ — the decision statistic used throughout, since gradient *scale* is absorbed by Adam.

Assumptions and their status:

| Assumption | Status |
|---|---|
| $f_\phi$ differentiable a.e. with useful gradients | **violated** at contact, and where the model interpolates smoothly through a true discontinuity |
| $\hat p_\phi \approx p$ on the policy's own state distribution | **violated** — the policy optimizes into model error (objective mismatch) |
| $\|A_k\|$ bounded s.t. products don't explode | **violated** for stiff/chaotic systems; $\lambda_H > 0$ is routine |
| $V_\psi$ accurate at $s_H$ | approximately holds on-distribution, degrades exactly where the rollout drifts |

## 3. State of the Art

**Established (independently reproduced).**
- **SVG($H$)** (Heess et al., NeurIPS 2015) — the reparameterized value-gradient family; SVG(1) with a real-data replay buffer was the robust member, SVG($\infty$) the fragile one.
- **Short-horizon truncation with a value bootstrap** is the operational fix. **SHAC** (Xu et al., RSS 2022) uses $H \approx 32$ in a differentiable simulator with a terminal critic and reports order-of-magnitude wall-clock wins over PPO on Ant/Humanoid-class tasks. **Dreamer/DreamerV3** (Hafner et al., 2020/2023; Nature 2025) backprops through latent rollouts of $H = 15$ with a $\lambda$-return bootstrap — the $H=15$ choice has survived four years and 150+ tasks without being beaten by a longer horizon.
- **Chaos is the binding constraint, not model error alone.** Metz et al. (2021) show pathwise gradient variance through unrolled dynamics grows exponentially in $H$ when $\lambda > 0$, and that the *mean* of the reparameterization gradient can be arbitrarily far from the smoothed objective's gradient.

**Claimed but unablated.**
- That analytic model gradients beat zeroth-order estimators *at matched compute* on stochastic, contact-rich tasks. Suh et al. (ICML 2022) show the first-order estimator can have higher error than the zeroth-order one under contact and stochasticity — the general claim does not hold.
- Adaptive-$H$ schemes: no scheme selecting $H$ online has been shown to beat a per-task tuned constant across a benchmark suite.

**Benchmark-number-only.** SHAC/DiffRL-family results are reported on a single differentiable-simulator stack (Warp/DiffTaichi lineage) with its own contact smoothing; cross-simulator replication is thin. Treat the speedups as stack-conditional.

## 4. What Is Known

- **Variance grows as $e^{2\lambda H}$.** Metz et al. (2021) demonstrate this on a 2-parameter linear dynamical system and on RL/meta-learning unrolls; at $\lambda>0$ the pathwise estimator's variance exceeds the score-function estimator's beyond a crossover $H$ typically in the tens.
- **Bias and variance trade against $H$ with an interior optimum.** Every published system that unrolls a *learned* model lands in $H \in [5,50]$: Dreamer $H=15$ (DMC, Atari, Minecraft), SHAC $H\approx 32$ (differentiable sim, $\le 10^3$-dim state), TD-MPC2 planning horizon $H=3$ with MPPI plus a learned value (Hansen et al., ICLR 2024, 104 tasks, up to 317M params). No published system uses $H > 100$ productively.
- **Mixing estimators helps.** Parmas et al. (PIPPS, ICML 2018) show total-propagation — inverse-variance mixing of pathwise and likelihood-ratio gradients — restores learning where pure pathwise fails on PILCO-style cart-pole with 3+ chaotic swing-ups.
- **Real-data anchoring beats pure imagination.** SVG(1) with replay and Dreamer's replay-conditioned latents both work; open-loop long-horizon imagination without replay does not.
- **Contact smoothing changes the gradient's meaning.** Suh et al. (2022) formalize this: the differentiable simulator returns the gradient of a *smoothed* problem, and that gradient can be a worse descent direction than a finite-difference estimate of the unsmoothed one.

## 5. What Is Not Known

- **Theoretically open.** No bound on $\|\mathbb{E}[\hat g_H] - g^\star\|$ that jointly accounts for model error $\varepsilon_{\text{model}}$ and closed-loop $\lambda$ and is tight enough to *predict* the optimal $H$. Existing results bound one term or the other. A conjecture worth killing: $H^\star \approx \min\{1/(2\lambda),\ \log(1/\varepsilon_{\text{model}})/\log(1/\gamma)\}$.
- **Empirically open.** Whether analytic gradients through a *learned* model beat well-tuned zeroth-order methods at matched FLOPs, at scale, on stochastic contact-rich tasks. Runnable today; nobody has published the matched-compute sweep across $H \in \{1,\dots,100\}$ with the same model and the same critic.
- **Methodologically blocked.** Measuring bias requires $g^\star$, which requires $\ge 10^4$ real rollouts per measurement point and is itself variance-limited. Consequence: reported "bias" curves are ratios of two noisy estimates, and the community has no agreed reference estimator. This is why $H$ is tuned rather than derived.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the failure cause under a single observable**. When training with $H=40$ degrades, three mechanisms produce the same symptom — a diverging policy:

1. exploding Jacobian products ($\lambda>0$) — a property of the *true* system the model faithfully reproduced;
2. model exploitation — the policy walked off-distribution into a region where $\hat r_\phi$ is fictitiously high;
3. smoothed-contact bias — the gradient is correct for a problem that isn't the one being solved.

All three are fixed by shortening $H$, which is why short horizons are universal and why nobody has isolated which mechanism dominates. Separating them needs per-mechanism instrumentation ($\lambda_H$, model-error-on-visited-states, and a smoothing parameter sweep) held simultaneously — and the ground truth for (2) requires real-environment rollouts at exactly the states imagination visited, which by construction were never visited for real.

Secondary cost: a matched-FLOPs comparison must count the backward pass through $H$ model steps (roughly $2\times$ to $3\times$ the forward cost), so an $H=40$ analytic update is ~$80$ model-network evaluations against a REINFORCE update's ~$40$. Papers routinely compare at matched *environment steps* instead, which flatters the model-based arm.

## 7. Current Research (as of 2026)

- **Differentiable-simulator policy learning** (NVIDIA Warp / Georgia Tech / MIT lineage): SHAC descendants, contact smoothing with learned relaxation schedules. *(frontier — verify)* — current work is on annealing the smoothing during training rather than fixing it.
- **Hybrid estimators**: gradient-informed PPO (Son et al., NeurIPS 2023) interpolates first-order simulator gradients with PPO's score-function gradient using an $\alpha$-policy-mixing rule; the Parmas total-propagation line continues in the variance-adaptive direction.
- **Latent-space world models with unified objectives** (Ghugare et al., ICLR 2023, ALM): make the representation, model and policy share one objective so that the analytic gradient is a gradient of the thing you care about. This is the most direct attack on failure mechanism (2).
- **Scaling latent imagination**: DreamerV3/TD-MPC2 lines, both keeping $H$ small and investing in the critic instead. The implicit bet is that credit assignment beyond ~15 steps is better handled by a value function than by a Jacobian chain — no one has tested this bet directly.

## 8. Concrete Next Experiment

**Question.** Does an interior optimum in $H$ exist for learned models, and is it predicted by $\lambda_H$?

**Scale.** 3 environments spanning $\lambda$: DMC `cheetah-run` ($\lambda \approx 0$), `humanoid-walk` (contact), and a chaotic `acrobot-swingup-sparse`. One shared DreamerV3-class world model (~20M params), trained to fixed quality (model MSE plateau) and then **frozen**. 5 seeds. Sweep $H \in \{1,2,5,10,15,25,40,70,100\}$. Cost: 135 policy-training runs at ~2 GPU-hours each ≈ 270 GPU-hours — a single 8-GPU node for ~1.5 days.

**Control arm.** At each $H$, a score-function (REINFORCE with the same $V_\psi$ baseline) update through the *same frozen model*, budgeted at **matched model-network FLOPs**, not matched imagined steps.

**The deciding number.** $\rho_H = \cos(\hat g_H,\ g^\star)$, with $g^\star$ estimated from $2\times10^4$ real-environment rollouts at 20 checkpoints along training. Report $H^\star = \arg\max_H \rho_H$ per environment and regress it on the measured $\lambda_H$.

- If $H^\star \cdot \lambda_H \approx \text{const}$ (within a factor of 2 across all three environments), the chaos-limited theory holds and $H$ becomes computable rather than tuned.
- If $H^\star$ is flat near 15 regardless of $\lambda$, the binding constraint is model error, and the field's fixed short horizon is right for the wrong reason.
- If $\rho_H \le \rho^{\text{REINFORCE}}$ at every $H$ on `humanoid-walk`, the analytic gradient through learned models has no advantage under contact — a negative result worth more than the positive one.

## 9. Key References

- **[Foundational]** N. Heess, G. Wayne, D. Silver, T. Lillicrap, Y. Tassa, T. Erez. *Learning Continuous Control Policies by Stochastic Value Gradients.* NeurIPS, 2015. — arXiv:1510.09142
- **[Foundational]** M. Deisenroth, C. Rasmussen. *PILCO: A Model-Based and Data-Efficient Approach to Policy Search.* ICML, 2011.
- **[Key negative result]** L. Metz, C. D. Freeman, S. S. Schoenholz, T. Kachman. *Gradients are Not All You Need.* arXiv, 2021. — arXiv:2111.05803
- **[Key negative result]** H. J. T. Suh, M. Simchowitz, K. Zhang, R. Tedrake. *Do Differentiable Simulators Give Better Policy Gradients?* ICML, 2022. — arXiv:2202.00817
- **[SOTA]** J. Xu, V. Makoviychuk, Y. Narang, F. Ramos, W. Matusik, A. Garg, M. Macklin. *Accelerated Policy Learning with Parallel Differentiable Simulation.* ICLR, 2022. — arXiv:2204.07137
- **[SOTA]** D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering Diverse Control Tasks through World Models.* Nature, 2025 (preprint 2023). — arXiv:2301.04104
- **[SOTA]** N. Hansen, H. Su, X. Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR, 2024. — arXiv:2310.16828
- **[Method]** P. Parmas, C. E. Rasmussen, J. Peters, K. Doya. *PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos.* ICML, 2018.
- **[Method]** S. H. Son, L. Zheng, R. Sullivan, Y.-L. Qiao, M. Lin. *Gradient Informed Proximal Policy Optimization.* NeurIPS, 2023.
- **[Method]** R. Ghugare, H. Bharadhwaj, B. Eysenbach, S. Levine, R. Salakhutdinov. *Simplifying Model-Based RL: Learning Representations, Latent-Space Models, and Policies with One Objective.* ICLR, 2023. — arXiv:2209.08466

## 10. Worked Example

Take a linearized cart-pole around the upright, closed loop under a mediocre policy, with dominant Jacobian eigenvalue $\|A\| = 1.12$ per control step at 50 Hz ($\lambda = \log 1.12 = 0.113$ per step). Suppose the learned model has one-step state error $\varepsilon = 10^{-3}$ (relative), which is a good model.

**Variance term.** Gradient magnitude through $H$ steps scales as $e^{\lambda H}$, so relative gradient standard deviation grows as $\sigma_H \propto e^{0.113 H}$:

| $H$ | $e^{\lambda H}$ | model error amplified: $\varepsilon e^{\lambda H}$ |
|---|---|---|
| 10 | 3.1 | 0.003 |
| 25 | 16.8 | 0.017 |
| 40 | 91 | 0.091 |
| 70 | 2 700 | 2.7 |
| 100 | 82 000 | 82 |

At $H=40$ the accumulated model error is 9% of the state scale — still arguably usable. At $H=70$ the imagined trajectory has diverged from any real one, and the analytic gradient is a precise derivative of a fiction. The *same* factor $e^{\lambda H} = 2700$ multiplies the gradient variance, so to hold the gradient's standard error fixed relative to $H=10$ you need $B$ to scale as $(2700/3.1)^2 \approx 7.6\times10^5$ times more rollouts.

**The obstruction, made visible.** Both columns blow up with the *same* exponent $\lambda$. So a practitioner who observes training collapse at $H=70$ cannot tell from the collapse whether to (a) collect more real data to shrink $\varepsilon$, or (b) accept that the system is chaotic and no amount of data will help — the fix for (a) is 10× more environment interaction, and for (b) it is to shorten $H$ and let $V_\psi$ carry the credit. The two prescriptions differ by an order of magnitude in cost, and the observable that would distinguish them ($\lambda$ measured on the *true* system versus on the model) is essentially never reported. That is the gap Section 8's regression of $H^\star$ on $\lambda_H$ is designed to close.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*