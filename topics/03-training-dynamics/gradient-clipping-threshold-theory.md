---
id: 03-training-dynamics/gradient-clipping-threshold-theory
title: "Gradient Clipping Threshold Theory"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Gradient Clipping Threshold Theory

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/gradient-clipping-threshold-theory` · **Status:** open

## 1. Problem Statement

Nearly every large-model training recipe clips the global gradient norm at $c = 1.0$. Nobody can derive that number, and nobody can say what it costs.

Three variants, different difficulty:

- **Theory.** Given a loss with known curvature growth and known gradient-noise tail, output the threshold $c^\star$ minimizing final loss at a fixed step budget. Solving means a formula $c^\star(L_0, L_1, \sigma, \alpha, \eta, B, T)$ whose predictions track measured optima, not just a convergence rate that happens to contain $c$.
- **Method.** A rule that picks $c$ (or a per-layer/per-step schedule) online, from statistics computable during training, and beats a tuned constant at fixed compute.
- **Measurement.** Report *what clipping is doing*: the clip rate $p_t$, the bias it injects, and its interaction with the learning rate — separably from the stability it buys. Today most papers report only "we used 1.0".

Decision predicate for a solved theory variant: on a held-out family of models spanning $\ge 2$ orders of magnitude in parameters, the predicted $c^\star$ lands within a factor of 2 of the empirical argmin of final validation loss, with no per-model fitting.

## 2. Formal Setting

Minimize $f(\theta) = \mathbb{E}_{\xi\sim\mathcal{D}}[F(\theta;\xi)]$, $\theta \in \mathbb{R}^d$. At step $t$ a minibatch of size $B$ gives $g_t = \frac{1}{B}\sum_{i=1}^{B}\nabla F(\theta_t;\xi_i)$. Global-norm clipping:

$$\tilde g_t = \min\!\Big(1, \frac{c}{\|g_t\|_2}\Big)\, g_t, \qquad \theta_{t+1} = \theta_t - \eta_t\, \mathcal{A}(\tilde g_t),$$

with $\mathcal{A}$ the identity (SGD) or the Adam map. Per-sample clipping (DP-SGD) instead clips each $\nabla F(\theta;\xi_i)$ before averaging — a different operator with different bias, often confused with the above.

Measured quantities:

- **Clip rate** $p_t = \Pr[\|g_t\| > c]$, estimated as the fraction of steps in a window where the pre-clip norm exceeds $c$. Log $\|g_t\|$ every step; this is the one cheap diagnostic almost nobody publishes.
- **Clipping bias** $b_t = \mathbb{E}[\tilde g_t] - \nabla f(\theta_t)$. Not directly measurable — $\nabla f$ requires a full-data pass; estimable at small scale by a periodic full-batch gradient.
- **Noise tail index** $\alpha$: fit $\Pr[\|g_t - \nabla f\| > s] \sim s^{-\alpha}$ from per-microbatch gradient norms. $\alpha < 2$ means infinite variance.
- **Generalized smoothness** $(L_0, L_1)$: $\|\nabla^2 f(\theta)\| \le L_0 + L_1\|\nabla f(\theta)\|$ (Zhang et al., ICLR 2020). Measured by Hessian-vector products along the trajectory and regressing local curvature on gradient norm.

Assumptions and their status in practice:

- *Uniform $L$-smoothness* — violated; curvature grows with gradient norm in transformers, which is the empirical content of the $(L_0,L_1)$ model.
- *Finite gradient variance* — violated in attention models; per-coordinate noise is heavy-tailed (Zhang et al., NeurIPS 2020).
- *Stationary noise* — violated; $\|g_t\|$ falls by 1–2 orders of magnitude over a run, so a fixed $c$ is a moving quantile.
- *Clipping as a mild correction* — violated at initialization and at loss spikes, where $p_t \to 1$ and the update becomes pure normalized SGD (sign of direction only, fixed step length $\eta c$).

The last point matters: at $p_t = 1$, clipping and learning rate are non-identifiable. The update is $\eta c \cdot g_t/\|g_t\|$, a function of the product $\eta c$ alone.

## 3. State of the Art

**Theory SOTA (established).**
- Zhang, He, Sra, Jadbabaie (ICLR 2020) prove clipped GD under $(L_0,L_1)$-smoothness converges at a rate independent of $L_1$, while any fixed-step GD must pay $O(L_1)$ — the first separation showing clipping is not merely a safety hack.
- Gorbunov, Danilova, Gasnikov (NeurIPS 2020) give high-probability convergence under heavy-tailed noise with only bounded $\alpha$-th moment, $\alpha \in (1,2]$; clipping is required, not optional, for high-probability guarantees.
- Koloskova, Hendrikx, Stich (ICML 2023) prove clipped SGD with *constant* $c$ does **not** converge to a stationary point in general — it converges to a neighborhood whose radius scales with the noise level; tight upper and lower bounds. This is the strongest negative result and it kills any hope of a clean "optimal constant $c$" story in the exact-convergence sense.

**Systems/empirical SOTA.** $c=1.0$ on the global norm, with Adam, is the recipe in GPT-3 (Brown et al., NeurIPS 2020), PaLM (Chowdhery et al., JMLR 2023), and essentially every open LLM release. Adaptive Gradient Clipping (Brock et al., ICML 2021, NFNets) clips per-parameter-block at $\lambda \|\theta_\ell\|/\|g_\ell\|$ and enabled batch size 4096 training without BatchNorm — an established, ablated result *in that architecture*. AutoClip (Seetharaman et al., MLSP 2020) sets $c$ to a running percentile of observed norms.

**Claimed but unablated.** That $1.0$ is near-optimal for LLMs. No public work sweeps $c$ over a decade at fixed $\eta$ and reports the loss curve at $\ge 1$B parameters. PaLM's reported mitigation for loss spikes was restarting from an earlier checkpoint and skipping data — not tightening $c$, which tells you the clip threshold in use was not preventing the spikes it is credited with preventing. Wortsman et al. (ICLR 2024) is the closest thing to a controlled study and it is at $\le 4.8$B with a focus on LR sensitivity, not $c$.

## 4. What Is Known

- **Clipping fixes exploding gradients in RNNs.** Pascanu, Mikolov, Bengio (ICML 2013) — the original result, on character-level RNNs with $\le 10^7$ parameters.
- **Rate separation.** Under $(L_0,L_1)$-smoothness, clipped GD needs $O(1/\epsilon^2)$ iterations with constants independent of $L_1$; the ICLR 2020 paper's AWD-LSTM experiments (24M params, PTB/WikiText-2) show the fitted $L_1$ is large, so the regime is real, not hypothetical.
- **Heavy tails are real in attention.** Zhang et al. (NeurIPS 2020) measure gradient-noise distributions on BERT-scale models and find tail behavior inconsistent with finite variance, while ResNet/ImageNet gradients look Gaussian. Clipping's benefit is architecture-dependent for a measurable reason.
- **Non-convergence with constant $c$.** Koloskova et al. (ICML 2023): the neighborhood radius is $\Theta(\sigma)$-scaled; only $c_t \to \infty$ or $\eta_t \to 0$ recovers exact convergence.
- **DP-SGD is a different problem.** With per-sample clipping, $c$ trades bias against noise: Abadi et al. (CCS 2016) inject noise $\propto c$, so $c$ scales the noise and the signal jointly. De et al. (2022) find, at ImageNet/JFT scale, that very small $c$ (with correspondingly scaled LR) is near-optimal, and Bu et al.'s "automatic clipping" (ICML 2023) shows the optimal $c$ is essentially degenerate — normalize per-sample gradients and fold $c$ into the learning rate.
- **The $\eta c$ degeneracy is proven in the DP case** (Bu et al.) and observed but unquantified in the global-norm case.

## 5. What Is Not Known

- **Theoretically open.** No result gives $c^\star$ as a function of measurable quantities for *fixed finite* $T$. All existing bounds are asymptotic or worst-case and are minimized at the boundary of the feasible set ($c\to\infty$ under smoothness; $c\to 0$ under heavy tails). No theory covers clipping composed with Adam's preconditioner, which is what everyone actually runs — the clip acts on the raw gradient, the step is taken on $m_t/\sqrt{v_t}$, and the norm ratio between the two is not controlled.
- **Empirically open.** The $c$-sweep at $\ge 1$B parameters, at fixed token budget, with $\eta$ re-tuned per $c$. Runnable today for $\sim$10 runs $\times$ 20B tokens. Unrun publicly.
- **Methodologically blocked.** Attributing spike prevention to clipping. "Loss spikes" have no agreed definition (threshold on $\Delta$loss? on gradient norm? on recovery time?), spikes are data-order dependent, and the counterfactual — the same run without clipping — usually diverges for reasons that may be unrelated. Until "a spike" is a defined event with a base rate, "clipping prevents spikes" is not a testable claim.

## 6. Why It Is Hard

Three named obstructions.

1. **Non-identifiability of $(\eta, c)$.** In the high-clip-rate regime the update depends only on $\eta c$. Any sweep over $c$ at fixed $\eta$ is therefore partly a learning-rate sweep in disguise, and any claim that "$c=1.0$ is optimal" is unfalsifiable without reporting $p_t$. Papers do not report $p_t$.
2. **Confounded measurement.** Clipping interacts with warmup, $\epsilon$ in Adam, weight decay, and loss scaling in fp16/bf16. A run that survives with $c=1.0$ and dies with $c=10$ may be dying from an fp16 overflow, not from an optimization pathology.
3. **Absent ground truth for the objective.** The quantity theory optimizes (stationarity gap, or a neighborhood radius) is not the quantity practitioners optimize (final validation loss at fixed compute, plus $\Pr[\text{run survives}]$). The second is a two-objective problem with a rare-event term whose estimation needs many seeds — at LLM scale, unaffordable. So the theory is well-posed and irrelevant; the practical question is relevant and under-measured.

## 7. Current Research (as of 2026)

- **Generalized smoothness.** Continued extension of the $(L_0,L_1)$ framework to Adam and to non-uniform/coordinate-wise smoothness (Li, Rakhlin, Jadbabaie, NeurIPS 2023, on Adam under relaxed assumptions). Active at MIT, EPFL (Stich's group), MBZUAI (Gorbunov).
- **Clipping-free stabilization.** Architectural fixes — QK-norm, logit soft-capping, careful residual scaling — that remove the spikes clipping was hired to suppress. If these work, $c$ becomes a vestigial hyperparameter. *(frontier — verify: no controlled study shows spikes vanish rather than move.)*
- **Normalized/sign optimizers.** Lion (Chen et al., NeurIPS 2023) and related sign-based updates are clipping taken to $c\to 0$ with $\eta$ rescaled; their success is indirect evidence that the high-clip-rate regime is benign, not pathological.
- **DP training.** Automatic clipping / per-sample normalization is close to settled for DP; the open part is the privacy–utility frontier at foundation-model scale.
- **Scaling-law treatment of $c$.** Whether $c^\star$ shifts with model size at fixed batch size. *(frontier — verify: we know of no published sweep.)*

## 8. Concrete Next Experiment

**Question.** Is $c=1.0$ optimal, or is it a plateau in $\eta c$?

**Scale.** 1.4B-parameter decoder-only transformer, 30B tokens, batch 2M tokens, bf16, Adam ($\beta=0.9/0.95$), cosine schedule with 2000-step warmup. About 12 runs $\times$ ~2.5k A100-hours.

**Grid.** $c \in \{0.1, 0.3, 1.0, 3.0, \infty\}$, and for each $c$ a 3-point $\eta$ sweep centered on the tuned $\eta^\star$ for $c=1.0$ ($\times 0.5, \times 1, \times 2$). Log $\|g_t\|$ every step, so $p_t$ is known for every arm.

**Control arm.** $c=\infty$ (no clipping) at tuned $\eta$, 3 seeds, to establish the base rate of divergence without clipping.

**Deciding number.** $\Delta = \min_\eta \mathcal{L}_{\text{val}}(c, \eta) - \min_\eta \mathcal{L}_{\text{val}}(1.0, \eta)$, in nats, across the $c$ grid. If $|\Delta| < 0.005$ nats for every finite $c$ once $\eta$ is re-tuned, then $c$ is a redundant reparametrization of $\eta$ over that range and the "threshold theory" question collapses into learning-rate theory. If some $c$ gives $\Delta < -0.02$ nats, there is a real optimum to explain, and the accompanying $p_t$ trace says which regime it sits in.

## 9. Key References

- **[Foundational]** Razvan Pascanu, Tomas Mikolov, Yoshua Bengio. *On the difficulty of training recurrent neural networks.* ICML, 2013. — arXiv:1211.5063
- **[Foundational/Theory]** Jingzhao Zhang, Tianxing He, Suvrit Sra, Ali Jadbabaie. *Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity.* ICLR, 2020. — arXiv:1905.11881
- **[SOTA-theory]** Anastasia Koloskova, Hadrien Hendrikx, Sebastian U. Stich. *Revisiting Gradient Clipping: Stochastic bias and tight convergence guarantees.* ICML, 2023. — arXiv:2305.01588
- **[Theory]** Eduard Gorbunov, Marina Danilova, Alexander Gasnikov. *Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping.* NeurIPS, 2020. — arXiv:2005.10785
- **[Empirical]** Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, Suvrit Sra. *Why are Adaptive Methods Good for Attention Models?* NeurIPS, 2020. — arXiv:1912.03194
- **[SOTA-method]** Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan. *High-Performance Large-Scale Image Recognition Without Normalization.* ICML, 2021. — arXiv:2102.06171
- **[Method]** Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux. *AutoClip: Adaptive Gradient Clipping for Source Separation Networks.* IEEE MLSP, 2020.
- **[DP]** Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. *Deep Learning with Differential Privacy.* ACM CCS, 2016. — arXiv:1607.00133
- **[DP-SOTA]** Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, George Karypis. *Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger.* NeurIPS, 2023. — arXiv:2206.07136
- **[Empirical]** Mitchell Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Systems]** Aakanksha Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR, 2023. — arXiv:2204.02311

## 10. Worked Example

A 350M transformer, batch 0.5M tokens, Adam, $\eta = 3\times10^{-4}$, $c = 1.0$. Logged pre-clip global norms:

| phase | median $\|g_t\|$ | clip rate $p_t$ |
|---|---|---|
| steps 0–500 (warmup) | 3.1 | 0.94 |
| steps 500–5k | 0.86 | 0.31 |
| steps 5k–50k | 0.34 | 0.02 |

Read the three rows as three different algorithms.

- **Warmup:** $p_t = 0.94$. Almost every update is $\eta_t \cdot \mathrm{Adam}(g_t \cdot 1.0/\|g_t\|)$. The gradient's *magnitude* is discarded; only its direction survives into Adam's moments. Halving $c$ to $0.5$ here rescales $\tilde g_t$ by exactly $0.5$ on 94% of steps — and because Adam is scale-invariant in the limit $\epsilon \to 0$, that rescaling *is nearly cancelled by the preconditioner*. So $c$ during warmup is close to a no-op with Adam, and close to a learning-rate halving with SGD. The same knob, two opposite meanings.
- **Mid-training:** $p_t = 0.31$. Only here does $c$ act as a genuine tail-truncation with a nontrivial bias. This is the only window where a $c$-sweep can teach you anything, and it is ~10% of the run.
- **Late training:** $p_t = 0.02$. Clipping touches 1 step in 50. Setting $c=\infty$ changes the trajectory by a rounding error — unless one of those 2% of steps is the spike that would have killed the run, an event with a base rate you cannot estimate from one seed.

The obstruction is now visible without any theory. Over a full run, $c$ is a no-op for 90% of steps, a learning-rate reparametrization for most of the rest, and a rare-event insurance policy on a handful of steps whose value cannot be measured at one seed. A single number $\Delta\mathcal{L}_{\text{val}}$ from a $c$-sweep therefore averages three regimes with different mechanisms — which is why sweeps come back flat and why $1.0$ has survived a decade without ever being justified.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*