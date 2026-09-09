---
id: 16-state-space-models/recurrent-depth-width-tradeoff
title: "Recurrent Depth Versus Width Tradeoff for Sequence Reasoning"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Recurrent Depth Versus Width Tradeoff for Sequence Reasoning

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/recurrent-depth-width-tradeoff` · **Status:** open

## 1. Problem Statement

A recurrent or state-space sequence model spends its budget on two axes that are usually confounded:

- **Serial depth** — how many dependent nonlinear steps run per token: layers $L$, times per-token recurrence iterations $r$ (looping, latent-recurrent depth, deep-equilibrium fixed-point steps).
- **Width / state capacity** — how many bits the model carries forward: model dimension $d$, SSM state dimension $N$, head count.

**The question.** At a fixed compute and parameter budget, what is the exchange rate between serial depth and state width for *sequence reasoning* — tasks whose solution is a composition of $k$ dependent steps over a length-$T$ input (multi-hop retrieval, permutation composition, iterated state tracking, arithmetic with carries)?

Three variants, different difficulties:

- **Theory.** Is there a task family where accuracy is achievable with depth $D$ and state $S$ iff $D \ge f(k)$ *and* $S \ge g(T)$ (a hard corner, no substitution), versus families where $D$ and $S$ substitute along a smooth frontier $D^{\alpha} S^{1-\alpha} \ge c$? **Open.**
- **Method.** Given a FLOP budget, what $(L, r, d, N)$ maximises reasoning accuracy? Currently set by folklore and small sweeps.
- **Measurement.** Can depth and state capacity be varied independently in a real architecture, and is any existing reasoning benchmark sensitive to depth rather than to memorised parametric knowledge? Currently **no** on both counts — see §6.

**Solved** would mean: a scaling law $\mathcal{L}(D, S, C)$ with a measured exponent pair, validated by extrapolation to a held-out $(D,S)$ point at ≥1B parameters, plus a matching separation theorem.

## 2. Formal Setting

Model $M_\theta$ processes $x_{1:T} \in \Sigma^T$. Layer $\ell$ maintains state $h_t^{(\ell)} \in \mathbb{R}^{P}$ updated by a selective linear recurrence (Mamba form):

$$h_t^{(\ell)} = A_t^{(\ell)} \odot h_{t-1}^{(\ell)} + B_t^{(\ell)} x_t^{(\ell)}, \qquad y_t^{(\ell)} = C_t^{(\ell)\top} h_t^{(\ell)},$$

with $h_t^{(\ell)} \in \mathbb{R}^{d_{\mathrm{in}} \times N}$, $d_{\mathrm{in}} = e\,d$ (expansion $e$, typically 2).

Measured quantities:

- **Serial depth** $D = r \cdot L$ — the number of dependent matrix-nonlinearity stages on the token-$t$ computation path, $r$ = recurrence iterations per token. Measured by counting blocks in the unrolled graph, not by wall-clock.
- **State capacity** $S = L \cdot e\, d \cdot N \cdot b$ bits, $b$ = numerical precision of the recurrent carry (bf16 $\Rightarrow b=16$). This is the *only* channel from prefix to suffix; it upper-bounds any information the model transports across the sequence.
- **Compute** $C \approx 2 \cdot r \cdot L \cdot \kappa d^2 \cdot T$ FLOPs, $\kappa \approx 12$ for a Mamba block. Measured by profiler, not analytic count.
- **Parameters** $\Theta$ — with looping, $\Theta$ is invariant in $r$; this is the point of loops.
- **Task difficulty** $k$ — the depth of the minimal circuit or the number of dependent hops; specified by construction, not inferred.

Objective: for task family $\mathcal{T}_{k,T}$, characterise the achievable region
$$\mathcal{R}(\epsilon) = \{(D, S) : \exists \theta,\; \Pr[M_\theta(x) \ne \mathcal{T}_{k,T}(x)] \le \epsilon\}.$$

**Assumptions, and which break.**

1. *Trainability equals expressivity* — assumed; **violated**. Log-depth shortcut constructions exist for automata but are not reliably found by SGD (Liu et al., 2023).
2. *Precision is $O(\log T)$* — assumed in all TC$^0$ arguments; **violated** in practice: bf16 is fixed at 8 mantissa bits, and Mamba's selective scan runs in fp32, so the theory's precision regime is neither the practical one nor conservative.
3. *State is fully usable* — assumed; **violated**. Measured effective rank of Mamba states is well below $e\,d\,N$; the bit bound is loose by an unmeasured factor.
4. *Loop iterations are homogeneous* — assumed by weight-tied recurrence; **violated** by Mixture-of-Recursions-style per-token adaptive depth.

## 3. State of the Art

**Theory SOTA (established).** Fixed-depth SSMs and transformers with log-precision both lie in uniform $\mathrm{TC}^0$ (Merrill & Sabharwal, TACL 2023; Merrill, Petty & Sabharwal, ICML 2024). Hence no amount of *width* buys $\mathrm{NC}^1$-hard state tracking at constant depth, assuming $\mathrm{TC}^0 \ne \mathrm{NC}^1$. Serial steps do buy it: $t$ chain-of-thought steps at constant depth simulate size-$t$ circuits; polynomial CoT reaches $\mathrm{P}$ (Li, Liu, Zhou & Ma, ICLR 2024; Merrill & Sabharwal, ICLR 2024). Depth $\Theta(\log k)$ is necessary and sufficient for $k$-hop induction in transformers (Sanford, Hsu & Telgarsky, ICML 2024). These are asymptotic separations, not exchange rates.

**Empirical SOTA (claimed, partly unablated).** Geiping et al. (2025) trained a 3.5B-parameter latent-recurrent-depth model on 800B tokens and report reasoning gains from raising test-time $r$ up to ~32, saturating thereafter. The comparison to a non-recurrent baseline is at matched parameters, **not** at matched state capacity or matched FLOPs, so the depth attribution is unablated. Saunshi et al. (ICLR 2025) report that an $L$-layer model looped $k$ times recovers much of a $kL$-layer model's reasoning accuracy while lagging badly on memorisation-heavy tasks — the cleanest evidence for "depth for reasoning, parameters for memorisation", but at ≤1B scale and on a fixed benchmark suite.

**Benchmark-number-only results.** Every published GSM8K/ARC delta for looped or recurrent-depth models is a single benchmark figure without a depth-matched, state-matched control arm. Treat as suggestive, not established.

## 4. What Is Known

- **State size is a hard information bound.** Copying a length-$n$ string requires $\Omega(n)$ carried bits; transformers trained on strings ≤50 tokens generalise to ~1000, while matched GSSMs/Mamba fail (Jelassi et al., ICML 2024, at 160M parameters and at 2.8B Pythia-vs-Mamba scale). Depth does not substitute.
- **Recall degrades with state, smoothly.** In Zoology (Arora et al., ICLR 2024), multi-query associative-recall accuracy for gated-convolution models is a monotone function of recurrent state size; the gap to attention closes as state grows, at 355M parameters. This is a *width* frontier, measured.
- **Depth buys state tracking, log-cheaply — in principle.** Depth $O(\log T)$ suffices to simulate $T$ steps of a solvable semiautomaton (Liu et al., ICLR 2023, at ≤12 layers, synthetic).
- **SSMs cannot do $S_5$ word problems at fixed depth**, confirmed empirically for S4 and Mamba (Merrill et al., ICML 2024).
- **Weight tying costs little on reasoning, much on knowledge.** ALBERT (Lan et al., ICLR 2020) and Universal Transformers (Dehghani et al., ICLR 2019) both show cross-layer sharing preserves compositional generalisation while reducing capacity-bound performance.

## 5. What Is Not Known

- **Theoretically open.** No separation theorem quantifies the *exchange rate*. There is no known family $\mathcal{T}_{k,T}$ with a proven frontier of the form $D \cdot \log S \ge \Omega(k)$ for recurrent models. Depth-vs-width tradeoffs are known for threshold circuits in the constant-vs-log regime only; the intermediate regime is untouched.
- **Empirically open.** Nobody has run a two-dimensional sweep over $(D, S)$ at fixed FLOPs at ≥1B parameters on reasoning tasks with controlled $k$. The experiment is runnable on ~10$^{21}$ FLOPs — roughly a 64-point grid of 400M-parameter models — and has not been run.
- **Methodologically blocked.** "Reasoning depth" of a natural benchmark is undefined. GSM8K has no measured $k$; its items mix retrieval, arithmetic and 2–8 dependent steps. Without a per-item $k$ label, a depth scaling law cannot be fit against a real benchmark at all.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement: in every shipped architecture, depth and state capacity covary by construction.** $S = L \cdot e d N b$ is linear in $L$. Doubling layers at fixed $d$ doubles the carried bits. So any observed gain from "more depth" in a layer sweep is jointly a gain from more state, and the published comparisons do not separate them.

Three secondary obstructions:

- **Loops break the covariance in the wrong direction.** Raising $r$ at fixed $L$ raises depth without raising the *number of distinct* recurrent states, but it does raise total recurrent updates — again not a clean depth knob.
- **Absent ground truth for $k$.** See §5.
- **Trainability confound.** Deep-narrow models are harder to optimise; a null result at depth 48 may be an optimiser result, not a capacity result. Distinguishing needs a matched hyperparameter search per grid point, multiplying cost.

## 7. Current Research (as of 2026)

- **Latent recurrent depth at scale** — Geiping and collaborators (ELLIS Tübingen / Maryland), continuing test-time depth scaling. *(frontier — verify whether a state-matched baseline has since been published.)*
- **Looped transformers for reasoning** — Saunshi, Dikkala and colleagues (Google Research), extending loop-count scaling laws.
- **Adaptive per-token depth** — Mixture-of-Recursions and successors (KAIST / Google DeepMind), routing tokens to different loop counts. *(frontier — verify.)*
- **State expansion** — Mamba-2 / SSD (Dao & Gu, ICML 2024) makes $N$ cheap to scale to 64–256, which for the first time makes an orthogonal $(L, N)$ grid affordable.
- **Circuit-complexity characterisation of recurrence** — Merrill, Sabharwal and collaborators (NYU / AI2).

## 8. Concrete Next Experiment

**The orthogonalised grid.**

- **Scale.** 400M-parameter Mamba-2 models, 20B tokens each, 25 grid points: $L \in \{6,12,24,48\}$ crossed with $N$ chosen per $L$ so that total state bits $S$ takes 4 fixed values (e.g. $N = 128, 64, 32, 16$ for $L = 6, 12, 24, 48$ holds $S$ constant), plus a looped arm ($L=6$, $r \in \{2,4,8\}$, weights tied) that raises $D$ at *exactly* constant $S$ and $\Theta$. FLOPs equalised per point by adjusting token count within ±5%. Total ≈ $3\times10^{20}$ FLOPs; a few thousand H100-hours.
- **Task.** Synthetic with a *labelled* $k$: composition of $k$ permutations from $S_5$ interleaved into a 4k-token distractor stream, $k \in \{2,4,8,16,32\}$, plus a $K$-pair associative-recall task with $K \in \{16, \dots, 1024\}$ to pin the width axis.
- **Control arm.** The constant-$S$ diagonal. Depth varies 8× along it while carried bits are identical. Any accuracy change on the diagonal is depth, and nothing else.
- **The deciding number.** $k^{*}(D)$ — the largest $k$ solved at ≥90% accuracy, as a function of $D$, along the constant-$S$ diagonal. If $k^{*}$ grows like $2^{cD}$ (exponential in depth, matching the shortcut construction), depth is the binding constraint and width is nearly free for reasoning. If $k^{*}$ grows linearly or saturates, SGD does not find the log-depth shortcut and the practical frontier is set by optimisation, not expressivity. One curve, one exponent, decides it.

## 9. Key References

- **[Foundational]** Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser. *Universal Transformers.* ICLR 2019. — arXiv:1807.03819
- **[Foundational]** Bai, Kolter, Koltun. *Deep Equilibrium Models.* NeurIPS 2019. — arXiv:1909.01377
- **[Foundational]** Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[Theory SOTA]** Merrill, Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL 2023.
- **[Theory SOTA]** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Theory SOTA]** Li, Liu, Zhou, Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR 2024. — arXiv:2402.12875
- **[Theory SOTA]** Sanford, Hsu, Telgarsky. *Transformers, Parallel Computation, and Logarithmic Depth.* ICML 2024. — arXiv:2402.09268
- **[Theory]** Liu, Ash, Goel, Krishnamurthy, Zhang. *Transformers Learn Shortcuts to Automata.* ICLR 2023. — arXiv:2210.10749
- **[Empirical SOTA]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Empirical SOTA]** Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024. — arXiv:2312.04927
- **[SOTA]** Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025. — arXiv:2502.05171
- **[SOTA]** Saunshi, Dikkala, Li, Kumar, Reddi. *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* ICLR 2025.
- **[SOTA]** Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[Related]** Wen, Dang, Lyu. *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval.* 2024.
- **[Survey]** Wang, Tsepa, Ma, Zheng, et al. *State Space Model for New-Generation Network Alternative to Transformers: A Survey.* 2024. (arXiv preprint; identifier omitted.)

## 10. Worked Example

Two Mamba-2 models, matched on parameters and FLOPs, at $e=2$, $N=16$, bf16 carry.

| | Arm A (deep–narrow) | Arm B (shallow–wide) |
|---|---|---|
| $L$ | 48 | 12 |
| $d$ | 1024 | 2048 |
| Params $\approx 12Ld^2$ | 604M | 604M |
| FLOPs/token | equal | equal |
| Serial depth $D$ | 48 | 12 |
| State values $L\!\cdot\!e d\!\cdot\!N$ | $48 \times 2048 \times 16 = 1{,}572{,}864$ | $12 \times 4096 \times 16 = 786{,}432$ |
| State bits $S$ | 25.2 Mb | 12.6 Mb |

Run a 16-hop composition task. Suppose A scores 71% and B scores 44%. The standard reading — "depth helps reasoning, 4× depth is worth 27 points" — **is not supported**, because A also carries exactly 2× the bits. The 27 points could be entirely a recall effect: Zoology's measured curves show associative-recall accuracy moving by tens of points across a 2× state change at this scale.

The fix is one line of the config. Set Arm B's $N = 32$: state values become $12 \times 4096 \times 32 = 1{,}572{,}864$, identical to A. Parameters move by <1% ($B$ and $C$ projections scale with $N$, not $d^2$); FLOPs move by <2%. Now the two arms differ only in $D$: 48 versus 12.

That control has not been reported in any published looped- or recurrent-depth comparison. The obstruction is not that the experiment is expensive — it is that the default architecture ties $S$ to $L$, so the natural sweep is uninterpretable, and everyone has run the natural sweep.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*