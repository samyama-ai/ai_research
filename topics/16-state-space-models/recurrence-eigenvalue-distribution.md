---
id: 16-state-space-models/recurrence-eigenvalue-distribution
title: "Optimal Eigenvalue Distribution on the Unit Disk"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Eigenvalue Distribution on the Unit Disk

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/recurrence-eigenvalue-distribution` · **Status:** open

## 1. Problem Statement

Diagonal linear recurrences — S4D, S5, LRU, Mamba, GLA, DeltaNet — all reduce to a per-channel scalar update $h_t = \lambda h_{t-1} + b x_t$ with $\lambda \in \mathbb{C}$, $|\lambda| < 1$. A layer with $N$ channels is therefore fully described, up to input/output mixing, by a multiset of $N$ points in the open unit disk $\mathbb{D}$. The question: **which distribution of those points should a model be initialized with, and which one should training converge to?**

Three variants, of very different difficulty:

- **Measurement.** Given a trained model, what *is* the empirical spectral distribution, and does it depend on task, depth, or scale? Complicated by input-dependent gating, which makes the spectrum a function of the input rather than of the weights.
- **Method.** Does any initialization law $\mu$ on $\mathbb{D}$ beat the incumbents (S4D-Lin, LRU ring, Mamba's $\exp(\Delta A)$) at matched compute, by more than seed noise?
- **Theory.** For a stated task class, is there a provably optimal $\mu$? Here the answer is partly known and mostly negative: expressivity results say the *support* matters (real negative eigenvalues buy state tracking), while approximation theory says exponential decay caps memory regardless of $\mu$.

Solving it means: a law $\mu^\star$, a task class for which it is optimal, and a demonstration that deviating from it costs measurable loss.

## 2. Formal Setting

A diagonal SSM layer maps $x_{1:T} \in \mathbb{R}^{T \times d}$ to $y_{1:T}$ via

$$h_t = \Lambda h_{t-1} + B x_t, \qquad y_t = \Re\big(C h_t\big) + D x_t, \qquad \Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_N).$$

Unrolled, $y_t = \sum_{k\ge 0} \big(\sum_{j} c_j b_j \lambda_j^k\big) x_{t-k}$, so the layer is a convolution whose kernel is a sum of $N$ complex exponentials. The **empirical spectral measure** is $\hat\mu_N = \frac1N\sum_j \delta_{\lambda_j}$, measured by reading $\Lambda$ off the checkpoint (no eigendecomposition needed — the parameterization is already diagonal).

Derived quantities, as actually measured:

- **Memory horizon** $\tau_j = -1/\log|\lambda_j|$ (steps to decay by $1/e$). For $|\lambda|$ near 1, $\tau \approx (1-|\lambda|)^{-1}$.
- **Oscillation period** $P_j = 2\pi/|\arg \lambda_j|$ (steps per rotation).
- **Effective rank of memory** at lag $k$: $r(k) = \big(\sum_j |c_jb_j\lambda_j^k|\big)^2 / \sum_j |c_jb_j\lambda_j^k|^2$ — how many channels still carry signal $k$ steps back.

For gated models (Mamba, GLA) the recurrence is $h_t = \bar\Lambda_t h_{t-1} + \dots$ with $\bar\lambda_{j,t} = \exp(\Delta_t(x_t)\,a_j)$, $\Delta_t > 0$. The spectrum is then a *random measure* driven by the data, and the reported quantity must be a token-averaged $\mathbb{E}_{x}[\hat\mu_N]$ over a held-out corpus, not a weight statistic.

Assumptions, with the ones known to be violated flagged:

1. *Diagonalizability with well-conditioned eigenvectors.* Holds by construction. **But** the parameterization is non-identifiable: $(\Lambda, B, C) \mapsto (\Lambda, \alpha B, C/\alpha)$ leaves the layer invariant, so eigenvalue *positions* are identifiable while their *importance* is not — $\hat\mu_N$ alone is uninformative without the $|c_jb_j|$ weights.
2. *Linearity of the recurrence.* Holds within a layer; **violated across layers** — stacking with MLPs and normalization means the composed system's memory is not the union of per-layer spectra.
3. *Stationarity.* $\Lambda$ fixed across time. **Violated** in every selective model since Mamba.
4. *Stability* $|\lambda_j| < 1$. Enforced by parameterizations like $\lambda = \exp(-\exp(\nu) + i\theta)$.

## 3. State of the Art

**Established (ablated, reproduced):**

- **S4D-Lin / S4D-Inv** (Gu, Gupta, Goel, Ré, NeurIPS 2022) — continuous-time poles $a_n = -\tfrac12 + i\pi n$ (Lin) or $-\tfrac12 + i\frac{N}{\pi}(\frac{N}{2n+1}-1)$ (Inv), discretized. Ablated against HiPPO-LegS and random init; the paper's own ablations show initialization choice dominates on Long Range Arena.
- **LRU ring initialization** (Orvieto et al., ICML 2023) — $|\lambda| \sim \mathcal{U}[r_{\min}, r_{\max}]$ (uniform in modulus, via the $\nu$ reparameterization), $\arg\lambda \sim \mathcal{U}[0,\theta_{\max}]$. The paper ablates modulus range, phase range, and normalization separately; the modulus range is the single most load-bearing choice.
- **Negative real eigenvalues enable state tracking** (Grazzi et al., ICLR 2025). Extending the range from $[0,1]$ to $[-1,1]$ in DeltaNet/Mamba-style recurrences lets the model express parity and modular counting; models restricted to $\lambda \in [0,1]$ provably cannot. Ablated on synthetic state-tracking suites and carried to language-model pretraining.

**Claimed but unablated, or benchmark-only:**

- That the LRA average of ~86% for S4/S5/LRU reflects spectral quality. It is a benchmark number. Amos, Berant & Gupta (ICLR 2024) showed the same architectures' LRA gaps largely close under self-supervised pretraining of the *data*, not respecification of the spectrum — so LRA rankings do not cleanly attribute credit to $\mu$.
- That Mamba's $\Delta$-gating implements an "optimal" adaptive spectrum. Asserted in the architecture's motivation, never ablated against a fixed spectrum with matched parameter count at scale.
- Reports of "the learned spectrum concentrates near the unit circle after training." Observed in individual checkpoints; not reproduced across scales with the $|c_jb_j|$ weighting that would make it meaningful.

## 4. What Is Known

- **Initialization changes LRA outcomes by tens of points at $N=64$–$256$ per layer, $\sim$100k–300k params/layer.** S4D ablations report Path-X (length 16384) succeeding with structured init and failing outright (~50%, chance) with naive random init.
- **The modulus range must be tuned to sequence length.** Orvieto et al. report the LRU needs $r_{\max}$ pushed to $\approx 0.999$ for Path-X; with the default ring reaching $r_{\max}=1$ but uniform in modulus, the expected number of channels with $\tau > 16384$ is under $0.1$ across the whole 6-layer, 256-channel model (calculation in §10).
- **Spectral shape is largely dispensable at scale.** Mamba-2 (Dao & Gu, ICML 2024) restricts $A$ to a scalar times identity per head — a *single* eigenvalue per head, i.e. a degenerate $\hat\mu$ — and matches or beats Mamba at 2.7B parameters / 300B Pile tokens on perplexity and zero-shot averages. This is the strongest negative evidence that a rich $\mu$ is necessary.
- **Support matters where shape does not.** $\lambda \in [0,1]$ diagonal SSMs are in uniform $\mathrm{TC}^0$ and cannot do $\mathrm{NC}^1$-hard state tracking (Merrill, Petty & Sabharwal, ICML 2024); formal-language characterizations (Sarrof, Veitsman & Hahn, NeurIPS 2024) place non-gated SSMs at star-free regular languages. Adding negative real eigenvalues lifts parity.
- **Exponential memory decay is a hard ceiling.** Inverse approximation results (Li & collaborators, ICLR 2024; Wang & Li, ICML 2024) show that stably parameterized linear recurrences approximate only targets with exponentially decaying memory — no choice of $\mu$ on the *open* disk escapes it.

## 5. What Is Not Known

- **Empirically open.** Whether any $\mu$ beats the incumbents at language-model scale ($\ge$1B params, $\ge$100B tokens). Every published spectral ablation is at LRA scale or on synthetics. The experiment is runnable today; nobody has run a matched-compute sweep over spectral families at 1B.
- **Empirically open.** Whether the *trained* spectrum differs materially from the initialized one at scale, once weighted by $|c_jb_j|$. Checkpoints exist; the measurement has not been published across a scale ladder.
- **Theoretically open.** Given a task with autocorrelation $\rho(k)$ and a budget of $N$ poles, no proof of the loss-minimizing $\mu^\star$ exists — the problem is a nonlinear rational approximation with an $\ell_2$-optimality criterion, and even the $N=2$ case has no closed form for general $\rho$.
- **Methodologically blocked.** For gated models, "the eigenvalue distribution" is not yet a well-defined object: it is input-dependent, and no accepted estimator of $\mathbb{E}_x[\hat\mu_N]$ with importance weighting has been standardized. Papers report incomparable quantities.

## 6. Why It Is Hard

The obstruction is **non-identifiability compounded by confounded measurement**. The layer's behavior depends on the products $c_jb_j\lambda_j^k$, not on $\lambda_j$ alone; the $(\alpha B, C/\alpha)$ symmetry means a channel with $|\lambda| = 0.9999$ and $|c_jb_j| = 10^{-6}$ is spectrally long-memory and functionally absent. Every published "learned spectra concentrate near the circle" plot ignores this weighting, so the plotted measure and the model's actual memory are different objects.

Second: the standard evaluation does not measure what it names. LRA Path-X is the benchmark used to justify long-memory initializations, but it is a single binary task whose signal is dominated by whether *any* channel has $\tau \gtrsim T$ — a threshold, not a distribution. Passing it certifies the support's edge, not the law's shape. Language-model perplexity, where shape could matter, is dominated by short-range statistics; the Mamba-2 scalar-$A$ result shows a degenerate spectrum costs nothing there. So the two available evaluations bracket the question without answering it: one is a threshold test, the other is insensitive.

## 7. Current Research (as of 2026)

- **Negative and complex eigenvalue ranges for state tracking** — Grazzi, Siems, Franke, Zela, Hutter, Pontil and collaborators (Freiburg/IIT). Established for DeltaNet-family; extension to gated linear attention at scale is *(frontier — verify)*.
- **Oscillatory / rotation-constrained recurrences.** Placing eigenvalues on or near a fixed radius with learnable phase, motivated by harmonic-oscillator dynamics (Rusch & Rus, ICLR 2025, LinOSS). Claimed gains on long time-series; scale-up to LM pretraining unpublished *(frontier — verify)*.
- **Theory of selective SSMs** — Cirone, Orvieto, Salvi, Lyons and collaborators, treating gated recurrences via rough-path/signature expansions; gives expressivity in terms of the gating law rather than a static spectrum.
- **Stable reparameterizations and the curse of memory** — Li, Wang and collaborators (NUS); Zucchet & Orvieto on gradient pathologies near $|\lambda|=1$.
- **Spectral probing of open checkpoints** (Mamba, Falcon-Mamba, Jamba, RWKV-7). Reported informally; no standardized importance-weighted estimator *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does the spectral *law*, holding support and parameter count fixed, change language-model loss by more than seed noise?

**Scale.** 370M-parameter Mamba-2-style model, $d_{\text{model}}=1024$, 48 layers, state size $N=128$ per head, 8 heads; 15B tokens of SlimPajama; ~$2\times10^{20}$ FLOPs per run. Five spectral arms × 3 seeds = 15 runs, roughly 1,100 A100-hours total.

**Arms** (all with $|\lambda|$ supported on $[r_{\min},1)$ with the same $r_{\min} = 1 - 10^{-4}$ upper reach, so *support* is identical and only the *law* changes):
1. **Control:** S4D-Real / Mamba-2 default, $a_j = -(j+1)$, scalar per head.
2. Uniform-in-modulus ring (LRU).
3. Uniform-in-$\log\tau$ (log-spaced horizons, $\tau_j$ geometric from $1$ to $10^4$).
4. Marchenko–Pastur-like bulk: $|\lambda|$ concentrated near $0.9$ with a thin tail to $1$.
5. Half the channels at $\lambda < 0$ (negative real), rest as arm 3.

**The deciding number.** Validation loss on held-out SlimPajama, in nats/token. Seed noise at this scale is $\approx 0.004$ nats. **Decision rule: if $\max_{\text{arm}} - \min_{\text{arm}} < 0.02$ nats (5× seed noise), the spectral law is confirmed irrelevant at fixed support and the problem collapses to a support question.** If any arm gains $\ge 0.02$ nats, report the winning law and re-run at 1.4B to test whether the gap survives scale.

**Secondary readout (cheap, same checkpoints):** importance-weighted spectral drift $W_1(\hat\mu^{\text{init}}_w, \hat\mu^{\text{final}}_w)$ where $w_j \propto |c_jb_j|$. If drift is $<0.01$ in $W_1$, training does not move the spectrum and initialization is the whole story.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Ankit Gupta, Albert Gu, Jonathan Berant. *Diagonal State Spaces are as Effective as Structured State Spaces.* NeurIPS 2022. — arXiv:2203.14343
- **[Foundational]** Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré. *On the Parameterization and Initialization of Diagonal State Space Models.* NeurIPS 2022. — arXiv:2206.11893
- **[SOTA]** Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De. *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML 2023. — arXiv:2303.06349
- **[SOTA]** Jimmy T. H. Smith, Andrew Warrington, Scott W. Linderman. *Simplified State Space Layers for Sequence Modeling.* ICLR 2023. — arXiv:2208.04933
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Riccardo Grazzi, Julien Siems, Jörg K. H. Franke, Arber Zela, Frank Hutter, Massimiliano Pontil. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025.
- **[Theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Theory]** Yash Sarrof, Yana Veitsman, Michael Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS 2024.
- **[Theory]** Shida Wang, Qianxiao Li. *StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization.* ICML 2024.
- **[Critique]** Ido Amos, Jonathan Berant, Ankit Gupta. *Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors.* ICLR 2024.
- **[Survey]** Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler. *Long Range Arena: A Benchmark for Efficient Transformers.* ICLR 2021. — arXiv:2011.04006

## 10. Worked Example

**Setting.** LRU on Path-X: $T = 16{,}384$, 6 layers, $N = 256$ complex channels per layer, so 1,536 channels total. Default ring init: $|\lambda|$ uniform on $[r_{\min}, r_{\max}] = [0, 1]$.

**Requirement.** For a channel to still carry input from step 1 at step $T$, need $|\lambda|^{T} \gtrsim e^{-1}$, i.e.

$$|\lambda| \ge \exp(-1/16384) = 0.9999390.$$

**Count under the default law.** $P(|\lambda| \ge 0.999939) = 6.10\times10^{-5}$. Expected number of qualifying channels in the whole model:

$$1536 \times 6.10\times10^{-5} = 0.094.$$

So a randomly initialized LRU has, in expectation, **fewer than one-tenth of one long-memory channel in the entire network**. $P(\text{at least one}) = 1 - (1-6.1\times10^{-5})^{1536} = 0.089$. Nine runs in ten start with no channel that can see the beginning of the sequence.

**With the tuned ring** $[r_{\min},r_{\max}] = [0.999, 0.9999]$: every channel has $\tau \in [1000, 10000]$, and $\tau \ge T/4$ for the top decile. Path-X becomes learnable — matching the reported need to change the ring for this task specifically.

**Where the obstruction shows up.** Now weight by importance. Suppose training drives $|c_jb_j|$ for the longest-$\tau$ channels down by $10^{-3}$ relative to the bulk while leaving $\Lambda$ untouched — entirely possible, since the $(\alpha B, C/\alpha)$ symmetry means the optimizer can silence a channel without moving its eigenvalue. The unweighted $\hat\mu_N$ still shows a mass at $|\lambda|\approx0.9999$ and one would report "the model learned long memory." The weighted measure $\hat\mu_w$ shows nothing there. Both plots come from the same checkpoint. Until the reported statistic is fixed to the weighted one, the measurement cannot distinguish a model that uses long memory from one that merely has the eigenvalues for it — which is precisely why the Mamba-2 scalar-$A$ result (one eigenvalue per head, no loss at 2.7B) is not yet decisive evidence either way.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*