---
id: 31-distributed-training/async-sgd-adaptive-optimizer-convergence
title: "Asynchronous SGD Convergence Under Adaptive Optimizers"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Asynchronous SGD Convergence Under Adaptive Optimizers

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/async-sgd-adaptive-optimizer-convergence` · **Status:** open

## 1. Problem Statement

Asynchronous SGD — workers compute gradients against whatever parameter version they last pulled, and the server applies them without waiting — has convergence guarantees under plain SGD that hold for *arbitrary, unbounded* delays. Every large model actually trained today uses an adaptive optimizer (Adam/AdamW, Adafactor, Lion, Shampoo), and for those the guarantees do not transfer.

Three distinct variants, routinely conflated:

- **Theory variant.** Does asynchronous Adam converge to a stationary point of a non-convex $f$ under a delay sequence $\{\tau_t\}$ with only $\mathbb{E}[\tau_t] < \infty$ (no uniform bound), and at what rate in the delay statistics? Solving it means a theorem with an explicit dependence on the delay distribution and on $\beta_2$, or a counterexample construction showing divergence at some finite average delay.
- **Method variant.** Is there a modification — staleness-scaled steps, delay-compensated preconditioner, buffered aggregation, second-moment freezing — that recovers the *synchronous* Adam loss curve at equal step count while retaining async throughput? Solving it means a token-matched loss match, not a wall-clock win.
- **Measurement variant.** What is the right control? An async run is faster per step and worse per step; nobody agrees whether the comparison arm is fixed steps, fixed tokens, fixed wall-clock, or fixed hardware-hours, and the four give opposite verdicts.

The theory variant is open. The method variant is empirically open at frontier scale. The measurement variant is methodologically blocked.

## 2. Formal Setting

Minimize $f(x) = \mathbb{E}_{\xi\sim\mathcal{D}}[F(x;\xi)]$, $x\in\mathbb{R}^d$, $f$ $L$-smooth and bounded below by $f^\star$. $n$ workers, one server, iteration counter $t$.

**Delay.** Worker $i$ pulls version $x_{t-\tau_t}$ and pushes $g_t = \nabla F(x_{t-\tau_t};\xi_t)$. Measured as a *version counter difference*: the server stamps every parameter broadcast with an integer, and $\tau_t$ is (stamp at apply time) − (stamp on the gradient). This is the only honest measurement — wall-clock latency is not the quantity in the bounds. Report the full empirical distribution: $\tau_{\text{avg}} = \frac{1}{T}\sum_t \tau_t$, $\tau_{\max}$, and $\tau_{p99}$; $\tau_{\max}$ alone is dominated by preemptions and is the wrong summary.

**Async Adam.** With $m_0=v_0=0$,
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2},\quad x_{t+1} = x_t - \eta\, \hat m_t \oslash (\sqrt{\hat v_t}+\epsilon).$$
The staleness enters **twice**: in the first moment (a delayed gradient) and in the second moment (a delayed *curvature estimate*). The second entry is what breaks the standard analysis.

**The measurable obstruction.** Define preconditioner drift over the delay window
$$\Delta_t \;=\; \Big\| \tfrac{1}{\sqrt{\hat v_t}+\epsilon} \oslash \tfrac{1}{\sqrt{\hat v_{t-\tau_t}}+\epsilon} - \mathbf{1} \Big\|_\infty ,$$
the per-coordinate ratio between the scaling the worker implicitly assumed and the one applied. $\Delta_t$ is directly loggable. Async SGD is the special case $\Delta_t \equiv 0$.

**Assumptions, and which are violated.**

| Assumption | Status in practice |
|---|---|
| $\tau_t \le \tau_{\max}$ bounded | **Violated.** Straggler/preemption tails are heavy; $\tau_{\max}/\tau_{\text{avg}} > 10$ is normal. |
| $\tau_t \perp \xi_t$ (delay independent of data) | **Violated.** Slow shards correlate with long sequences and with datacenter locality. |
| $\|\nabla F\|\le G$ (bounded gradient) — required by nearly every Adam proof | **Violated** at LLM scale: loss spikes are gradient-norm excursions of $10$–$100\times$. |
| Bounded variance $\mathbb{E}\|\nabla F - \nabla f\|^2 \le \sigma^2$ | Approximately holds; $\sigma^2$ is batch-size dependent and rarely measured. |
| $v_t$ stationary enough that $\hat v_t \approx \hat v_{t-\tau}$ | **Violated** exactly in the regime that matters — warmup, LR decay, spikes. |

## 3. State of the Art

**Theory SOTA — async SGD (established).** Mishchenko, Bach, Even, Woodworth, *Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays* (NeurIPS 2022) removes the bounded-delay assumption entirely for plain SGD. Koloskova, Stich, Jaggi, *Sharper Convergence Guarantees for Asynchronous SGD* (NeurIPS 2022) gives a rate governed by $\tau_{\text{avg}}$ rather than $\tau_{\max}$, plus a matching lower bound. Both are proved; neither covers any adaptive preconditioner.

**Theory SOTA — adaptive, synchronous (established).** Défossez, Bottou, Bach, Usunier, *A Simple Convergence Proof of Adam and Adagrad* (TMLR 2022) gives $O(\log T/\sqrt T)$ to a stationary point under bounded gradients. Reddi, Kale, Kumar (ICLR 2018) show Adam can diverge even synchronously without $\beta_2$ conditions.

**Theory SOTA — adaptive + delay (thin).** Sra, Yu, Li, Smola, *AdaDelay* (AISTATS 2016) handles delay-adaptive stepsizes but for **convex** objectives and AdaGrad-style scaling, not Adam's exponential second moment. Federated results (Reddi et al., *Adaptive Federated Optimization*, ICLR 2021) put the adaptivity in the **server/outer** step with synchronous rounds — a different problem. No published theorem covers non-convex Adam with a stale second moment and unbounded delay. This is the gap.

**Systems/empirical SOTA.** FedBuff (Nguyen et al., AISTATS 2022) — buffered async aggregation, $K$ updates flushed together; works, but the reported gains are federated benchmark numbers, not LLM pretraining. PipeMare (Yang, Lipton, Zaharia et al., MLSys 2021) — async pipeline parallelism with discrepancy correction and learning-rate rescheduling; the ablation isolating *which* fix carries the gain is partial. Liu et al., *Asynchronous Local-SGD Training for Language Modeling* (DeepMind, 2024, arXiv:2401.09135) is the closest thing to a direct attack: inner AdamW, async outer updates, ~20M–150M params on C4; they report that naive async local SGD underperforms sync at matched steps and propose delayed Nesterov outer updates plus dynamic local-step counts to close it. **Claimed but unablated:** that the fix generalizes past 150M parameters, and that the failure mechanism is the outer momentum rather than the inner second moment.

## 4. What Is Known

- **Async SGD is provably robust to arbitrary delay.** Rate $O(\sigma^2/(n\varepsilon^2) + \tau_{\text{avg}}/\varepsilon)$ class results, with a lower bound showing $\tau_{\text{avg}}$ (not $\tau_{\max}$) is the right parameter (Koloskova et al. 2022; Mishchenko et al. 2022). Theory, not scale-limited.
- **Async quality degrades with worker count in deep nets.** Chen, Monga, Bengio, Józefowicz, *Revisiting Distributed Synchronous SGD* (ICLR 2016 workshop): on Inception/ImageNet with up to 100 workers, sync-with-backup-workers reached higher test accuracy at equal wall-clock than async; async accuracy fell as workers grew. Measured at ~100 workers, CNN scale.
- **Staleness scaling helps in the CNN regime.** Zhang et al., *Staleness-aware Async-SGD* (IJCAI 2016): scaling the step by $1/\tau_t$ recovered much of the sync accuracy on CIFAR/ImageNet-scale models with tens of learners.
- **Delay compensation via a Hessian-vector approximation helps.** Zheng et al., *Asynchronous SGD with Delay Compensation* (ICML 2017): DC-ASGD narrowed the async-sync gap on CIFAR-10/ImageNet at 4–16 workers. Both of these were validated with **momentum SGD, not Adam**.
- **Adam is not optional at LLM scale.** Zhang et al. (NeurIPS 2020) tie Adam's advantage on attention models to heavy-tailed gradient noise; SGD does not substitute. So "just use async SGD, it has proofs" is not available.
- **Async local SGD at 150M params underperforms sync at matched outer steps** unless the outer update is modified (Liu et al. 2024). This is the only frontier-adjacent measurement, and 150M is two to three orders of magnitude below where the question is asked.

## 5. What Is Not Known

- **Theoretically open.** Whether async Adam converges under unbounded delay in non-convex $f$. No proof; no divergence counterexample either. Even the bounded-delay non-convex case lacks a rate with explicit $\beta_2$–$\tau$ coupling. Conjecture worth attacking: convergence requires $\tau_{\text{avg}} \cdot (1-\beta_2) \lesssim 1$, i.e. the second moment must not turn over within the delay window.
- **Empirically open.** Whether async AdamW loses token-matched loss at $\ge 7$B parameters and $\tau_{\text{avg}} \in [1,8]$. Runnable today on a few hundred accelerators; not run publicly.
- **Empirically open.** Whether the damage comes from the stale first moment or the stale second moment. Decidable by a cheap arm (freeze $v$ during the delay window) that nobody reports.
- **Methodologically blocked.** The comparison arm. Async wins wall-clock and loses per-step; without a community-agreed accounting unit — we suggest *accelerator-seconds to fixed validation loss* — every paper can pick the axis that flatters it.

## 6. Why It Is Hard

The specific obstruction is that **the standard proof device is unavailable, and the empirical substitute is confounded.**

Async SGD proofs use the perturbed-iterate framework (Mania et al., *Perturbed Iterate Analysis for Asynchronous Stochastic Optimization*, SIOPT 2017): a stale gradient is a gradient at a nearby point, and $L$-smoothness charges the error at $O(\eta L \sum_{j} \|x_t - x_{t-j}\|)$, which the step size can absorb. Adam's update $\hat m_t \oslash \sqrt{\hat v_t}$ is **not the gradient of any function**, and $v_t$ is optimizer state that drifts independently of the parameters. A stale gradient is therefore not a perturbed gradient — it is a correctly-computed gradient divided by the *wrong* preconditioner. The smoothness argument has nothing to charge $\Delta_t$ against.

Second: non-identifiability in the experiment. Async changes effective batch composition, effective learning rate (via $\Delta_t$), and gradient noise simultaneously. An async run that loses 0.05 nats may be suffering from staleness or may simply be at the wrong effective LR — and the two are indistinguishable without an LR sweep per delay level, which multiplies the cost by 5–7×.

## 7. Current Research (as of 2026)

- **Low-communication distributed training.** DiLoCo (Douillard et al., 2023) and Streaming DiLoCo (Google DeepMind, 2025) push toward infrequent sync; the async variants inherit exactly this problem in the outer optimizer *(frontier — verify current scale claims)*.
- **Decentralized/volunteer training runs** (Prime Intellect INTELLECT series, Nous Research DisTrO/DeMo) operate at 10B+ params over heterogeneous links, where async is a throughput necessity; published artifacts are run reports, not controlled ablations *(frontier — verify)*.
- **Federated adaptive optimization theory** (CMU, Google Research) continues extending server-side adaptivity to partial participation and staleness; results remain convex or bounded-gradient.
- **Delay-adaptive stepsize theory** (EPFL/MLO, Inria SIERRA) — the natural home for the first async-Adam theorem.

## 8. Concrete Next Experiment

**Scale.** 1.3B-parameter decoder-only transformer, 30B tokens, 64 accelerators, AdamW ($\beta_1{=}0.9$, $\beta_2{=}0.95$), global batch 1M tokens. About 3k accelerator-hours per arm; 6 arms ≈ 18k accelerator-hours. Deliberately *simulated* delay: inject staleness in software so $\tau$ is exact and reproducible, rather than inheriting a datacenter's tail.

**Arms.**
1. **Control:** synchronous AdamW, $\tau=0$.
2. Async, $\tau_{\text{avg}}=4$ (geometric tail, $\tau_{p99}=16$), unmodified Adam.
3. Arm 2 + staleness-scaled step $\eta_t = \eta/(1+\tau_t)$.
4. Arm 2 + **frozen second moment**: $v$ updated only from gradients with $\tau_t \le 1$; stale gradients update $m$ only. *This is the mechanism isolator.*
5. Arm 2 + $\beta_2 = 0.999$ (slow $v$, so $\Delta_t \to 0$ by construction).
6. Control at $\eta \times 0.7$ — the LR-confound arm, so a loss gap cannot be re-explained as mistuning.

**Deciding number.** Token-matched validation loss gap at 30B tokens, $\delta = \mathcal{L}_{\text{async}} - \mathcal{L}_{\text{sync}}$, in nats. Decision rule: if arm 2 has $\delta > 0.02$ nats (≈ 2% perplexity, well above the ~0.005-nat seed noise at this scale) **and** arms 4 or 5 cut $\delta$ by more than half, the second moment is the culprit and the method variant has a fix. If arm 3 alone closes it, this is ordinary step-size mistuning, not an adaptivity problem. Log $\Delta_t$ throughout; the prediction is $\delta$ tracks $\mathbb{E}[\Delta_t]$, not $\tau_{\text{avg}}$.

## 9. Key References

- **[Foundational]** Recht, Ré, Wright, Niu. *Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.* NIPS, 2011.
- **[Foundational]** Dean et al. *Large Scale Distributed Deep Networks.* NIPS, 2012.
- **[Foundational]** Mania, Pan, Papailiopoulos, Recht, Ramchandran, Jordan. *Perturbed Iterate Analysis for Asynchronous Stochastic Optimization.* SIAM Journal on Optimization, 2017.
- **[SOTA — theory]** Mishchenko, Bach, Even, Woodworth. *Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays.* NeurIPS, 2022. — arXiv:2206.07638
- **[SOTA — theory]** Koloskova, Stich, Jaggi. *Sharper Convergence Guarantees for Asynchronous SGD for Distributed and Federated Learning.* NeurIPS, 2022. — arXiv:2206.08307
- **[SOTA — adaptive]** Défossez, Bottou, Bach, Usunier. *A Simple Convergence Proof of Adam and Adagrad.* TMLR, 2022.
- **[SOTA — empirical]** Liu, Douillard, Sharma et al. *Asynchronous Local-SGD Training for Language Modeling.* 2024. — arXiv:2401.09135
- **[Method]** Zheng, Meng, Wang, Chen, Yu, Ma, Liu. *Asynchronous Stochastic Gradient Descent with Delay Compensation.* ICML, 2017.
- **[Method]** Nguyen, Malik, Zhan, Yousefpour, Rabbat, Malek, Huba. *Federated Learning with Buffered Asynchronous Aggregation.* AISTATS, 2022.
- **[Method]** Sra, Yu, Li, Smola. *AdaDelay: Delay Adaptive Distributed Stochastic Optimization.* AISTATS, 2016.
- **[Systems]** Yang, Zhang, Ré, Aberger, De Sa. *PipeMare: Asynchronous Pipeline Parallel DNN Training.* MLSys, 2021.
- **[Systems]** Chen, Pan, Monga, Bengio, Józefowicz. *Revisiting Distributed Synchronous SGD.* ICLR Workshop, 2016.
- **[Context]** Reddi, Charles, Zaheer, Garrett, Rush, Konečný, Kumar, McMahan. *Adaptive Federated Optimization.* ICLR, 2021.
- **[Context]** Douillard, Feng, Rusu et al. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105

## 10. Worked Example

One coordinate $j$, one delay event. Take $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, and a coordinate entering a gradient-norm spike — an event that occurs several times in any multi-billion-token run.

At step $t-\tau$ the worker pulls parameters when $\hat v_{t-\tau,j} = 10^{-6}$, so it *implicitly* expects its gradient to be scaled by $1/\sqrt{10^{-6}} = 10^{3}$.

During the delay window ($\tau = 8$ steps), the spike hits: eight gradients of magnitude $10^{-1}$ arrive. The second moment updates as
$$\hat v_{t,j} \approx \beta_2^{8}\cdot 10^{-6} + (1-\beta_2^{8})\cdot 10^{-2} = 0.663\cdot 10^{-6} + 0.337\cdot 10^{-2} \approx 3.4\times 10^{-3}.$$
The applied scaling is $1/\sqrt{3.4\times10^{-3}} \approx 17.2$. The drift is
$$\Delta_t = \left|\frac{10^{3}}{17.2} - 1\right| \approx 57.$$

The stale gradient is applied with a preconditioner **58× smaller** than the one under which it was computed. Note the direction: the gradient is *under*-applied, so this event does not blow up. Now reverse it — the coordinate was hot ($\hat v = 10^{-2}$) when pulled, and quiets during the window ($\hat v \to 10^{-4}$). Then the applied scaling is $10\times$ larger than assumed, and a stale, now-wrong-direction gradient is amplified an order of magnitude. Under plain async SGD the same event contributes exactly $\eta \cdot 10^{-1}$, bounded, and $L$-smoothness absorbs it.

That asymmetry is the obstruction made concrete. The staleness error under Adam is not $O(\eta \tau)$; it is $O(\eta \tau \Delta_t)$, and $\Delta_t$ is a *ratio of optimizer state at two times*, unbounded above by anything the smoothness constant controls. A convergence proof must either bound $\Delta_t$ — which requires assuming the curvature estimate is stationary over the delay window, precisely false during warmup, LR decay, and loss spikes — or find a Lyapunov function that tracks $(x_t, v_t)$ jointly. Neither exists. And empirically, a 1.3B run that loses 0.03 nats to async cannot be attributed to this mechanism without the frozen-$v$ arm, because an unswept learning rate produces an identical-looking gap.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*