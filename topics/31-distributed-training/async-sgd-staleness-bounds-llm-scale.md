---
id: 31-distributed-training/async-sgd-staleness-bounds-llm-scale
title: "Asynchronous SGD Staleness Bounds at LLM Scale"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Asynchronous SGD Staleness Bounds at LLM Scale

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/async-sgd-staleness-bounds-llm-scale` · **Status:** empirically-open

## 1. Problem Statement

Asynchronous SGD applies a gradient computed at parameters $\theta_{t-\tau}$ to the current parameters $\theta_t$. The gradient is *stale* by $\tau$ steps. Theory says convergence survives bounded staleness; practice at frontier scale mostly avoids async entirely and pays synchronization cost instead.

The problem: **determine the critical staleness $\tau_c(N, B, D)$ — the largest delay a language model of $N$ parameters, batch size $B$, token budget $D$ can tolerate before it needs more tokens than synchronous SGD to reach the same loss — and determine how $\tau_c$ scales with $N$.**

Three variants, different difficulty:

- **Measurement.** Given a training run, report staleness as a distribution, not a max, and report the loss penalty attributable to staleness rather than to the confounded change in effective batch size. Currently done inconsistently; this is the weakest link.
- **Method.** Build a staleness-compensation rule (delay-adaptive step size, momentum correction, gradient-difference extrapolation) that raises $\tau_c$ at $N \geq 10^{10}$ without a per-step cost proportional to $\tau$.
- **Theory.** Prove a rate whose delay-dependent term is tight for the loss landscape transformers actually present — non-convex, heavy-tailed gradient noise, non-uniform smoothness — rather than for the bounded-variance $L$-smooth model.

A solution to the measurement variant is a published curve $\tau_c$ vs $N$ over at least a decade of $N$ with a synchronous control at matched tokens. A solution to the theory variant is a bound that predicts that curve within a constant.

## 2. Formal Setting

$M$ workers, one logical parameter store. Worker $m$ pulls $\theta_{t_m}$, computes a stochastic gradient on a microbatch $\xi$, and pushes it back at server step $t$. The update is

$$\theta_{t+1} = \theta_t - \eta_t \, g(\theta_{t - \tau_t}; \xi_t), \qquad \mathbb{E}[g(\theta;\xi)] = \nabla f(\theta).$$

**Quantities as measured.**

- $\tau_t$: *server steps* elapsed between the pull that produced $g_t$ and its application. Measured by tagging every pull with a monotonic step counter; not by wall-clock, which conflates staleness with straggling.
- $\bar\tau = \frac{1}{T}\sum_t \tau_t$, $\tau_{\max} = \max_t \tau_t$, and the full empirical CDF $\hat F_\tau$. Report all three; a single number hides the tail that actually causes divergence.
- $\sigma^2$: per-microbatch gradient variance, measured as $\frac{1}{K-1}\sum_k \|g_k - \bar g\|^2$ over $K$ concurrent microbatches at fixed $\theta$ — the same estimator used for critical batch size in McCandlish et al. (2018).
- $B_{\text{eff}}$: tokens per applied update. In async with $M$ workers this drifts; it must be logged, because a staleness ablation that changes $B_{\text{eff}}$ measures batch size, not staleness.
- $D_\tau(\ell^\ast)$: tokens consumed to first reach validation loss $\ell^\ast$ under staleness regime $\tau$. Define the **token-inflation ratio** $R(\tau) = D_\tau(\ell^\ast)/D_0(\ell^\ast)$ and

$$\tau_c(N,B,D) = \max\{\tau : R(\tau) \leq 1.05\}.$$

The 5% threshold is a convention; state it, do not hide it.

**Assumptions, and which are violated.**

| Assumption | Status in LLM training |
|---|---|
| $L$-smooth $f$ | Violated locally; loss spikes and attention-logit blowups are non-smooth events |
| Bounded variance $\mathbb{E}\|g-\nabla f\|^2 \le \sigma^2$ | Approximately holds per-step, but $\sigma^2$ falls by 1–2 orders of magnitude over a run, so a fixed-$\sigma$ bound is loose early and vacuous late |
| $\tau_t$ independent of $\xi_t$ | Violated: slow workers hold specific data shards and specific hardware, so delay correlates with gradient content |
| $\tau_t \le \tau_{\max}$ bounded | Violated by preemption and network partitions; the tail is what matters |
| Single stochastic gradient per step | Violated: Adam second moments, gradient clipping, and weight decay all interact with stale gradients non-linearly, and no delay bound covers Adam with clipping |

That last row is the crux: essentially all ASGD theory is for plain SGD, and every LLM is trained with AdamW plus global-norm clipping.

## 3. State of the Art

**Theory (established).**
- Agarwal & Duchi (NeurIPS 2011) and Lian et al. (NeurIPS 2015): async SGD attains the $O(1/\sqrt{MT})$ linear-speedup rate provided $\tau_{\max} = O(\sqrt{T/M})$-ish; delay enters as a constraint on step size.
- Arjevani, Shamir & Srebro (ALT 2020): tight analysis for delayed SGD on quadratics — delay costs a transient of order $\tau$ but does not degrade the asymptotic $\sigma^2/\epsilon^2$ term.
- Mishchenko, Bach, Even & Woodworth (NeurIPS 2022): ASGD beats minibatch SGD under *arbitrary, unbounded, adversarial* delays, with no bounded-delay assumption. Koloskova, Stich & Jaggi (NeurIPS 2022) independently show the rate depends on delay through a quantity closer to $\bar\tau$ than $\tau_{\max}$, via delay-adaptive step sizes.
- Cohen, Daniely, Drori, Koren & Schain (NeurIPS 2021): robustness to arbitrary delays via a picky-worker scheme.

Established: these are proved for smooth non-convex or convex $f$ with plain SGD. **Not established**: any of it for Adam, for clipped gradients, or for the transformer loss landscape.

**Systems / empirical (mixed).**
- Ho et al. (NIPS 2013), Stale Synchronous Parallel: bounded-staleness parameter server, $s$-step slack, validated on topic models and small nets.
- Chen, Monga, Bengio & Jozefowicz (ICLR 2016 workshop, arXiv:1604.00981): with ~100 workers on Inception/ImageNet, async degrades final quality and *synchronous SGD with backup workers wins*. This result is the reason frontier labs are synchronous, and it has never been re-run at LLM scale.
- Zheng et al. (ICML 2017), delay compensation via a Taylor correction using the Hessian-vector approximation $\lambda g \odot g$: reported gains on CIFAR/ImageNet; **unablated at scale** and unreplicated for transformers.
- Douillard et al., DiLoCo (2023, arXiv:2311.08105) and Asynchronous Local-SGD (2024, arXiv:2401.09135): local-SGD-style outer optimization tolerates hundreds of local steps; the async variant identifies the outer-optimizer/staleness interaction as the binding constraint. Measured at ~150M parameters.
- Scaling Laws for DiLoCo (Charles et al., 2025, arXiv:2503.09799): extends the local-update family to ~10B parameters — the closest thing to a staleness-vs-$N$ curve, but for *local steps*, not for gradient staleness in the ASGD sense.
- INTELLECT-1 (Jaghouar et al., 2024, arXiv:2412.01152) and OpenDiLoCo (arXiv:2407.07852): 10B-parameter decentralized runs. These are **benchmark numbers only** — a single configuration, no staleness sweep, no matched synchronous control at the same token budget.

## 4. What Is Known

- Hogwild! (Recht et al., NeurIPS 2011): lock-free async attains near-linear speedup for *sparse* problems; sparsity is the reason it works, and LLM gradients are dense.
- Chen et al. (2016): at ~100 workers, async Inception-v3 reaches materially worse ImageNet top-1 than sync with backup workers; the sync-with-backups arm matched or beat async on both time-to-quality and final quality. Scale: ~$10^7$ parameters, 2016 hardware.
- Mishchenko et al. (2022): for $M$ workers, the delay term in the ASGD complexity is additive, not multiplicative — degradation is $O(\bar\tau/\epsilon)$-type, not $O(\tau_{\max})$ slowdown. Proved for smooth non-convex SGD, no experiments above CIFAR scale.
- DiLoCo (2023): 150M-parameter transformer on C4, $H = 500$ local steps, 8 workers, matches the synchronous baseline while communicating ~500× less. This bounds a *different* delay (outer-step delay), and it holds only with an outer Nesterov momentum optimizer — plain outer averaging is worse.
- Asynchronous Local-SGD (2024): at 20M–150M, naive async local SGD *underperforms* its sync counterpart; the fix required delay-dependent momentum handling (DyLU / delayed Nesterov). So even in the regime where async works, it needed an algorithmic correction, at 150M.
- PipeDream (Narayanan et al., SOSP 2019): staleness of exactly one pipeline flush, handled by weight stashing, with no measurable accuracy loss. This is $\tau = 1$–$4$, not $\tau = 100$.

The largest $N$ at which a *staleness sweep with a matched synchronous control* has been published is on the order of $10^8$ parameters.

## 5. What Is Not Known

- **Empirically open (primary).** Whether $\tau_c$ grows, shrinks, or is constant in $N$. Two plausible stories, both untested above ~1B: (a) $\tau_c$ *grows* because larger models take smaller relative steps per update, so $\|\theta_t - \theta_{t-\tau}\|/\|\theta_t\|$ shrinks and the stale gradient stays valid longer; (b) $\tau_c$ *shrinks* because larger models are trained at larger $B$, where $\sigma^2$ is already suppressed and staleness bias — not noise — dominates. The experiment is runnable today on 256–1024 accelerators. Nobody has published it.
- **Theoretically open.** No convergence bound for delayed **Adam with global-norm clipping**. Clipping makes the update non-linear in the gradient, so the standard "delay = perturbation of order $\eta\tau L$" argument does not close.
- **Methodologically blocked.** "Staleness tolerance" is routinely reported at fixed wall-clock or fixed step count, where async also changes $B_{\text{eff}}$ and the learning-rate schedule position. Without holding tokens and $B_{\text{eff}}$ fixed, the measured penalty is not attributable to staleness. There is no agreed protocol; $R(\tau)$ above is a proposal, not a standard.

## 6. Why It Is Hard

**Confounded measurement, compounded by compute cost.** Staleness cannot be varied in isolation on real hardware: making workers slower changes $\tau$, throughput, and the token order simultaneously. The clean design is *simulated* staleness — buffer gradients for exactly $\tau$ steps on a synchronous cluster — but that costs the same GPU-hours as a real run while producing none of the speedup that motivates async, so nobody funds it at frontier scale.

Second obstruction: **the effect is small and the noise is large**. A 3% token-inflation penalty at 1B parameters sits inside the seed-to-seed spread of a single-seed run. Deciding $\tau_c$ at the 5% threshold needs either multiple seeds or a loss-curve fit, multiplying an already large budget.

Third: **the theory measures a different object than the practice names.** Bounds are stated in gradient-norm-$\epsilon$ for plain SGD; practitioners care about validation loss under AdamW with clipping and a cosine schedule. A bound can be tight and still say nothing about $\tau_c$.

## 7. Current Research (as of 2026)

- **Google DeepMind** — DiLoCo line: streaming/overlapped-communication variants (arXiv:2501.18512) and DiLoCo scaling laws to ~10B (arXiv:2503.09799). Direction: push the communication-delay tolerance, characterize it as a function of $N$. *(frontier — verify current status)*
- **Prime Intellect** — OpenDiLoCo, INTELLECT-1/2: decentralized and async-RL training over the open internet, where staleness is imposed by the network rather than chosen. *(frontier — verify)*
- **EPFL (Jaggi/Koloskova lineage) and Inria (Bach, Even)** — delay-adaptive step sizes and arbitrary-delay analysis; the open thread is extending these to adaptive optimizers.
- **Federated learning** — FedBuff (Nguyen et al., AISTATS 2022) buffered async aggregation; staleness weighting $1/\sqrt{1+\tau}$ is standard practice with weak theory.
- **Async RL post-training** — off-policy staleness in generation/learning splits is now the highest-stakes async setting at frontier scale, and its staleness bounds are governed by policy-lag/importance-weight arguments rather than by ASGD theory. *(frontier — verify)*

## 8. Concrete Next Experiment

**Simulated-staleness sweep with a matched-token synchronous control.**

- **Scale.** Three model sizes: 150M, 1.3B, 8B decoder-only transformers, Chinchilla-ratio token budgets ($\approx 20N$ tokens), identical data order, AdamW with global-norm clip 1.0, cosine schedule. ~$3\times10^{22}$ FLOPs total for the 8B arm's five conditions; roughly 40–60k H100-hours.
- **Manipulation.** On a synchronous cluster, hold each computed gradient in a FIFO of depth $\tau \in \{0, 4, 16, 64, 256\}$ before applying it. This fixes $B_{\text{eff}}$, token order, and schedule position exactly. Staleness is then the only variable — this is the arm that removes the confound.
- **Control arm.** $\tau = 0$, same seed, same data order, same $B_{\text{eff}}$. Three seeds at 150M to estimate seed variance; one seed at 1.3B and 8B, with the 150M seed spread used as the error bar.
- **Deciding number.** $\tau_c$ at $R(\tau) \le 1.05$, read off per model size, where $\ell^\ast$ is the $\tau=0$ final validation loss. The question is settled by the **sign of $d\log \tau_c / d\log N$**: positive means async gets *easier* with scale and decentralized training is a live path to frontier models; negative or zero means the Chen et al. (2016) verdict extends to LLMs and synchronous training stays correct.
- **Second output, nearly free.** Log $\|\theta_t - \theta_{t-\tau}\|/\|\theta_t\|$ and the cosine between $g(\theta_{t-\tau})$ and $g(\theta_t)$. If $\tau_c$ tracks the $\tau$ at which that cosine crosses a fixed value (say 0.5) across all three $N$, that cosine becomes a cheap surrogate predictor — testable at 150M, usable at 100B.

## 9. Key References

- **[Foundational]** B. Recht, C. Ré, S. Wright, F. Niu. *Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.* NeurIPS, 2011.
- **[Foundational]** A. Agarwal, J. Duchi. *Distributed Delayed Stochastic Optimization.* NeurIPS, 2011.
- **[Foundational]** J. Dean et al. *Large Scale Distributed Deep Networks.* NIPS, 2012.
- **[Foundational]** Q. Ho, J. Cipar, H. Cui, S. Lee, J. K. Kim, P. Gibbons, G. Gibson, G. Ganger, E. Xing. *More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server.* NIPS, 2013.
- **[Foundational]** X. Lian, Y. Huang, Y. Li, J. Liu. *Asynchronous Parallel Stochastic Gradient for Nonconvex Optimization.* NeurIPS, 2015.
- **[Established/negative]** J. Chen, R. Monga, S. Bengio, R. Jozefowicz. *Revisiting Distributed Synchronous SGD.* ICLR Workshop, 2016. — arXiv:1604.00981
- **[Theory SOTA]** K. Mishchenko, F. Bach, M. Even, B. Woodworth. *Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays.* NeurIPS, 2022. — arXiv:2206.07638
- **[Theory SOTA]** A. Koloskova, S. U. Stich, M. Jaggi. *Sharper Convergence Guarantees for Asynchronous SGD for Distributed and Federated Learning.* NeurIPS, 2022. — arXiv:2206.08307
- **[Theory]** Y. Arjevani, O. Shamir, N. Srebro. *A Tight Convergence Analysis for Stochastic Gradient Descent with Delayed Updates.* ALT, 2020.
- **[Theory]** A. Cohen, A. Daniely, Y. Drori, T. Koren, M. Schain. *Asynchronous Stochastic Optimization Robust to Arbitrary Delays.* NeurIPS, 2021.
- **[Method]** S. Zheng, Q. Meng, T. Wang, W. Chen, N. Yu, Z.-M. Ma, T.-Y. Liu. *Asynchronous Stochastic Gradient Descent with Delay Compensation.* ICML, 2017.
- **[Systems SOTA]** A. Douillard, Q. Feng, A. A. Rusu, R. Chhaparia, Y. Donchev, A. Kuncoro, M. Ranzato, A. Szlam, J. Shen. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105
- **[Systems SOTA]** B. Liu, R. Chhaparia, A. Douillard, S. Kale, A. A. Rusu, J. Shen, A. Szlam, M. Ranzato. *Asynchronous Local-SGD Training for Language Modeling.* 2024. — arXiv:2401.09135
- **[Scale]** Z. Charles, G. Teston, L. Dery, K. Rush, N. Fallen, Z. Garrett, A. Szlam, A. Douillard. *Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo.* 2025. — arXiv:2503.09799
- **[Systems]** D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. Devanur, G. Ganger, P. Gibbons, M. Zaharia. *PipeDream: Generalized Pipeline Parallelism for DNN Training.* SOSP, 2019.
- **[Survey]** M. Assran, A. Aytekin, H. R. Feyzmahdavian, M. Johansson, M. Rabbat. *Advances in Asynchronous Parallel and Distributed Optimization.* Proceedings of the IEEE, 2020.
- **[Survey]** P. Kairouz et al. *Advances and Open Problems in Federated Learning.* Foundations and Trends in Machine Learning, 2021.

## 10. Worked Example

Take a 1.3B-parameter model, $B = 2$M tokens/step, $\eta = 3\times10^{-4}$ peak with AdamW. At mid-training the per-step parameter change has relative norm roughly $\|\Delta\theta_t\|/\|\theta_t\| \approx 10^{-4}$ (Adam's normalized update times $\eta$, divided by a weight RMS of order $10^{-2}$).

The first-order staleness error in the applied gradient is bounded by

$$\|\nabla f(\theta_{t-\tau}) - \nabla f(\theta_t)\| \le L\,\|\theta_t - \theta_{t-\tau}\| \approx L \cdot \tau \cdot 10^{-4}\|\theta\|.$$

With $\tau = 64$ this is a $\sim 0.6\%$ relative parameter displacement. Compare that to the *stochastic* error: at $B = 2$M tokens the gradient noise-to-signal ratio is still order 1 early in training. Naive conclusion: staleness of 64 is invisible, buried under noise. This is the argument that gets made informally, and it predicts $\tau_c \gg 100$.

Now make the obstruction visible. Run the argument in the other direction, on the same numbers. Adam's update is $\hat m/(\sqrt{\hat v}+\epsilon)$, with $\beta_2 = 0.95$, so $\hat v$ has an effective window of ~20 steps. A gradient stale by $\tau = 64$ is applied against a second-moment estimate built from a *disjoint* window of the trajectory. The stale gradient is not a small perturbation of the current gradient in the preconditioned metric — the preconditioner itself has turned over three times. The $L$-smoothness bound above says nothing about this, because it bounds $\|\nabla f(\theta_{t-\tau}) - \nabla f(\theta_t)\|$ in the Euclidean norm and the update is taken in a different, time-varying one.

So the two available first-principles estimates disagree in kind: the smoothness argument predicts $\tau_c$ in the hundreds and *increasing* with $N$ (since relative step size falls as models grow); the preconditioner-turnover argument predicts $\tau_c \approx 1/(1-\beta_2) \approx 20$, roughly independent of $N$. No published bound covers delayed Adam, and no published experiment above ~150M distinguishes the two. That gap — a factor of ten in $\tau_c$, decidable by one sweep, undecided by every theorem on the books — is the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*