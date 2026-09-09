---
id: 16-state-space-models/learning-rate-timescale-coupling
title: "Learning Rate and Timescale Coupling in SSM Training"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning Rate and Timescale Coupling in SSM Training

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/learning-rate-timescale-coupling` · **Status:** open

## 1. Problem Statement

Every deep SSM (S4, S4D, S5, LRU, Mamba, Mamba-2, RG-LRU) carries a set of parameters that control *how long* each channel remembers: the step size $\Delta$ and the state matrix eigenvalues $\lambda$. Every published recipe treats these parameters specially — a separate, smaller learning rate (typically $10^{-3}$) and zero weight decay — and every recipe justifies this with a footnote rather than an analysis.

The problem: **given a target memory horizon distribution and a compute budget, predict the learning rate on the timescale parameters that lets training reach it — and show that the resulting horizons are learned rather than inherited from initialization.**

Three variants, of very different difficulty:

- **Measurement.** Define and measure the *effective memory horizon* $H$ of a trained channel and its *drift* from initialization. Currently no standard estimator; papers report $\Delta$ histograms, which is not the same object.
- **Method.** Find a parameterization or per-group LR rule under which the optimal timescale LR is invariant to width, depth, sequence length and batch size — the SSM analogue of $\mu$P.
- **Theory.** Prove, for a linear recurrence with $\bar\lambda = \exp(\Delta\lambda)$ trained by Adam, a bound on the number of steps needed to move $\log H$ by a given amount, and show whether the loss landscape in $\log H$ has curvature that scales with sequence length $L$.

Solving it means: a stated rule, tested across $\geq$ two orders of magnitude of width and sequence length, that removes the hand-tuned $10^{-3}$.

## 2. Formal Setting

A diagonal SSM layer with state size $N$, channels $d$, input $u_t \in \mathbb{R}^d$:

$$x_t = \bar{A}\,x_{t-1} + \bar{B}\,u_t, \qquad y_t = C x_t + D u_t, \qquad \bar{A} = \exp(\Delta A),$$

with $A = \mathrm{diag}(\lambda_1,\dots,\lambda_N)$, $\mathrm{Re}\,\lambda_n < 0$, and $\Delta > 0$ per channel. Parameterizations as actually implemented: $\Delta = \mathrm{softplus}(\delta)$, $\lambda = -\exp(a) + i\,\omega$ (S4D, Mamba) or $\lambda = -\exp(-\exp(a))$ (LRU).

**Effective memory horizon** — measured, per state $n$, as the geometric decay time in *tokens*:

$$H_n \;=\; \frac{-1}{\log|\bar\lambda_n|} \;=\; \frac{1}{\Delta\,|\mathrm{Re}\,\lambda_n|}.$$

Equivalently the impulse response $|h_n(k)| = |\bar\lambda_n|^k$ falls to $e^{-1}$ at $k = H_n$. Measure it by reading $\Delta$ and $\lambda$ off the checkpoint; measure it *empirically* by the input-perturbation lag $\hat H = \min\{k : \|\partial y_{t}/\partial u_{t-k}\| < e^{-1}\|\partial y_t/\partial u_t\|\}$, which for input-dependent $\Delta$ (Mamba) is the only honest estimator.

**Timescale transport.** The quantity that couples to LR is $\log H$, not $H$. Since $\log H = -\log\Delta - \log|\mathrm{Re}\,\lambda|$ and $\partial\Delta/\partial\delta = \sigma(\delta)$,

$$\frac{\partial \log H}{\partial \delta} = -\frac{\sigma(\delta)}{\Delta} \;\xrightarrow[\Delta \to 0]{}\; -1 ,$$

because $\mathrm{softplus}$ is $\approx \exp$ in its left tail. So in the small-$\Delta$ regime the timescale parameters are *already* log-parameterized: an Adam step of size $\eta$ moves $\log H$ by at most $\eta$. Over $T$ steps the reachable set is

$$|\Delta \log H| \;\le\; \eta\,T\,\rho, \qquad \rho = \Big|\tfrac{1}{T}\textstyle\sum_t \mathrm{sign}(g_t)\Big| \in [0,1],$$

with $\rho$ the **sign-consistency** of the gradient sequence — measurable directly from the optimizer state. This is the budget constraint the whole problem turns on.

**Assumptions, and where they break.** (i) Adam's update is unit-magnitude — violated during warmup and whenever $\hat v$ is stale. (ii) $\Delta$ is a free parameter — violated in Mamba, where $\Delta_t = \mathrm{softplus}(W_\Delta u_t + b_\Delta)$, so the horizon is a function of a *projection*, and the LR that matters is the one on $W_\Delta$. (iii) Layers are independent — violated: stacked SSMs compose horizons, and the network-level horizon is not the max of layer horizons. (iv) $|\mathrm{Re}\,\lambda|$ is fixed at $\approx 1/2$ — violated whenever $A$ is trained.

## 3. State of the Art

**Established (ablated, reproduced).**
- Separate optimizer group for $(A, \Delta, B)$ with LR $10^{-3}$ and weight decay $0$ is required, not cosmetic. Gu, Gupta, Goel & Ré (*On the Parameterization and Initialization of Diagonal State Space Models*, NeurIPS 2022) ablate this: applying the global LR or nonzero weight decay to $A$ and $\Delta$ degrades Long Range Arena results substantially, on Path-X to chance.
- The $\Delta$ **initialization range** must track sequence length. The S4/S4D line initializes $\Delta$ log-uniform on $[\Delta_{\min}, \Delta_{\max}] = [10^{-3}, 10^{-1}]$ and reports that Path-X ($L = 16{,}384$) needs the range shifted down by roughly $L$'s ratio; with the default range the task does not leave chance.
- $\exp$/log-space parameterization of the decay beats direct parameterization for stability. Orvieto et al. (*Resurrecting Recurrent Neural Networks for Long Sequences*, ICML 2023) ablate exponential vs. direct $\lambda$ and show the exponential form is what makes long-horizon training stable.

**Claimed but unablated.**
- That $10^{-3}$ is *the* right value rather than an artifact of the $\approx$100M-parameter, $\approx$100k-step regime the recipes were tuned in. No published sweep of the timescale LR across width at fixed data.
- That $\mu$P (Yang et al., *Tensor Programs V*, 2022) transfers to SSMs. $\mu$P is derived for weight matrices; $\Delta$ and $A$ are not matrices and have no width-scaling argument. Mamba-2 (Dao & Gu, ICML 2024) reports $\mu$P-style LR scaling for the model at large, while leaving the SSM group's LR fixed — the combination is unjustified in the paper.

**Benchmark-number-only.** Griffin/Hawk (De et al., 2024) report scaling curves to 14B with the RG-LRU gate, but the timescale LR is folded into an unreported hyperparameter search; the numbers do not isolate it.

## 4. What Is Known

- **Scale $\approx$ 0.1–1M params, LRA.** S4 reaches 86.1% LRA average and $\approx$96% on Path-X; the same architecture with $A,\Delta$ under global LR and weight decay collapses on Path-X to 50%. (Gu et al., ICLR 2022; S4D paper, NeurIPS 2022.)
- **Scale 130M–2.8B, The Pile.** Mamba initializes $\Delta$ log-uniform on $[10^{-3}, 10^{-1}]$ — a 100$\times$ spread of horizons at step 0 — and excludes $A_{\log}$ and $\Delta$ bias from weight decay. Perplexity matches or beats a Transformer++ at equal params. (Gu & Dao, COLM 2024.)
- **Stability theory.** Wang & Li (*StableSSM*, ICML 2024) prove that the reparameterization used for $\lambda$ determines whether the "curse of memory" — the approximation gap for long-memory targets — is removed, and that stable reparameterizations change the *gradient scale*, not just the feasible set. This is the closest existing link between parameterization and optimization.
- **LR-sensitivity methodology.** Wortsman et al. (*Small-scale proxies for large-scale Transformer training instabilities*, ICLR 2024) establish that LR-sensitivity curves measured at 100M+ params predict instabilities at larger scale — the measurement protocol exists, it has just not been applied to the SSM parameter group.

## 5. What Is Not Known

- **Theoretically open.** No bound on steps-to-horizon. Nobody has shown whether the loss as a function of $\log H$ has curvature growing with $L$ (which would force $\eta \propto 1/L$) or bounded curvature (which would make $\eta$ length-invariant). No $\mu$P-style derivation for $\Delta$ or $A$ under width or depth scaling.
- **Empirically open.** The 2-D sweep $(\eta_{\text{global}}, \eta_{\text{ssm}})$ at 3–4 widths, on a task with a *known* required horizon, is runnable today on a few hundred GPU-hours. Nobody has published it. Likewise: does the final horizon distribution depend on $\eta_{\text{ssm}}$ at all, or is it pinned by initialization?
- **Methodologically blocked.** For selective SSMs, "the model's memory horizon" has no agreed definition — $\Delta$ is input-dependent, so $H$ is a random variable over the data distribution, and the reported $\Delta$ histograms conflate the parameter with the induced horizon. The perturbation-lag estimator $\hat H$ above is one proposal, not a standard.

## 6. Why It Is Hard

**Non-identifiability, compounded by transport limits.** Two hypotheses produce the same training curve: (a) the model *found* the horizons it needs; (b) the model *could not move* its horizons and the loss adapted around the initialization. Both give a converged run and a plausible $\Delta$ histogram. Separating them requires knowing the reachable set, which is $\eta T \rho$ — and $\rho$ is never reported.

The second obstruction is that $H$ and the readout $C$ trade off exactly: a channel can emulate a long horizon by chaining two short-horizon layers, so per-layer $H$ is not identifiable from behavior. Any evaluation that scores "long-context ability" and infers "the timescales were learned" is measuring the composition, not the parameter.

Third: the confound with weight decay. Timescale params are excluded from decay, so the $\eta_{\text{ssm}}$ ablation and the $\lambda_{\text{wd}}$ ablation have never been run orthogonally at scale.

## 7. Current Research (as of 2026)

- **$\mu$P for recurrent/state parameters.** Extending Tensor Programs to non-matrix parameters with their own scaling exponent. Groups: Microsoft Research (Yang and collaborators), Cerebras. *(frontier — verify)*
- **Input-dependent timescale analysis.** Measuring the induced horizon distribution in Mamba-2 and gated linear attention (Yang, Kim et al. line of work) rather than the parameter distribution. *(frontier — verify)*
- **Reparameterization-for-optimization.** StableSSM's line (Wang & Li) extended from approximation to trainability bounds.
- **Length-generalization work** repeatedly rediscovers the $\Delta$-range/length coupling; the connection to LR is stated informally but not tested.

## 8. Concrete Next Experiment

**Scale.** Mamba-2, four widths $d \in \{256, 512, 1024, 2048\}$ (≈15M to ≈700M params), 20B tokens each, $L = 8192$. Plus a diagnostic task with a *known* required horizon: selective copy with delay $k \in \{64, 1024, 8192\}$.

**Sweep.** $\eta_{\text{ssm}} \in \{10^{-4}, 3\!\times\!10^{-4}, 10^{-3}, 3\!\times\!10^{-3}, 10^{-2}\}$ crossed with the $\mu$P-scaled global LR at its own optimum. Log $\rho$ (gradient sign-consistency on $\delta$) and $\hat H$ every 1000 steps.

**Control arm.** Timescale parameters **frozen** at initialization, everything else trained, at each width. This is the arm that distinguishes hypothesis (a) from (b) and it is absent from the literature.

**The deciding number.** $\eta^\star_{\text{ssm}}(d)$, the argmin of validation loss, plotted against $d$. If $\eta^\star_{\text{ssm}}$ is flat in $d$ within a factor of 2, the fixed $10^{-3}$ is justified and the problem's method variant is closed. If it moves by $\geq 4\times$ across the 8$\times$ width range, the recipe is a scale-specific artifact. Secondary decider: if the frozen-timescale arm matches the trained arm within 0.02 nats at every width, timescales are not being learned at all and the LR question is moot.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR, 2022. — arXiv:2111.00396
- **[Foundational]** Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Ré. *HiPPO: Recurrent Memory with Optimal Polynomial Projections.* NeurIPS, 2020. — arXiv:2008.07669
- **[SOTA / recipe]** Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré. *On the Parameterization and Initialization of Diagonal State Space Models.* NeurIPS, 2022. — arXiv:2206.11893
- **[SOTA]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[Parameterization]** Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De. *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML, 2023. — arXiv:2303.06349
- **[Theory]** Shida Wang, Qianxiao Li. *StableSSM: Alleviating the Curse of Memory in State-Space Models through Stable Reparameterization.* ICML, 2024. — arXiv:2311.14495
- **[Method]** Jimmy Ba, Greg Yang, Edward J. Hu et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* 2022. — arXiv:2203.03466
- **[Method]** Mitchell Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Related]** Jimmy T. H. Smith, Andrew Warrington, Scott W. Linderman. *Simplified State Space Layers for Sequence Modeling.* ICLR, 2023. — arXiv:2208.04933
- **[Related]** Soham De, Samuel L. Smith, Anushan Fernando et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427

## 10. Worked Example

Take a Mamba-style channel at initialization with $\Delta = 10^{-1}$ and $|\mathrm{Re}\,\lambda| = 1$. Its horizon is $H = 1/(\Delta|\mathrm{Re}\lambda|) = 10$ tokens. The task needs $H = 1000$ tokens. Required travel:

$$\Delta \log H = \log(1000/10) = 4.61 \text{ nats}.$$

Adam with $\eta_{\text{ssm}} = 10^{-3}$ moves $\delta$ by $\approx 10^{-3}$ per step, and in the small-$\Delta$ regime $|\partial \log H/\partial\delta| \to 1$, so:

- **Perfect sign consistency** ($\rho = 1$): $4.61 / 10^{-3} = 4{,}610$ steps. Fine.
- **Realistic sign consistency.** SSM timescale gradients are dominated by a noisy long-range credit-assignment term; a plausible $\rho \approx 0.05$ gives $4.61/(10^{-3}\cdot 0.05) = 92{,}200$ steps.

A 130M Mamba run on ~10B tokens at batch 0.5M tokens is $\approx 20{,}000$ steps. **The channel cannot arrive.** It does not diverge, it does not spike, it does not show up in the loss curve as anything but a slightly worse constant — it simply stays near $H \approx 25$ and the network routes around it by composing two layers.

Now the obstruction. The published fix for this task would be to shift $\Delta_{\min}$ down at initialization — and that works, which is exactly why the LR question stays open: **initialization and learning rate are substitutes here.** Both papers' evidence is compatible with "$\eta_{\text{ssm}} = 10^{-3}$ is correct" and with "$\eta_{\text{ssm}}$ is irrelevant because $\eta T \rho \ll$ the needed travel and the init range is doing all the work." The frozen-timescale control arm in §8 costs one extra run per width and settles it; nobody has published it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*