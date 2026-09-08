---
id: 16-state-space-models/gating-forgetting-spectrum
title: "Gating Functions and the Forgetting Spectrum"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Gating Functions and the Forgetting Spectrum

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/gating-forgetting-spectrum` · **Status:** open

## 1. Problem Statement

Every modern linear-recurrent language model — Mamba, Mamba-2, HGRN2, GLA, RWKV-6/7, Griffin's RG-LRU, Gated DeltaNet — carries a per-channel, per-token gate $\alpha_t \in (0,1)$ that decides how much of the recurrent state survives one step. The set of realized decay rates across channels, layers and inputs is the model's **forgetting spectrum**. The problem is to characterize the map from *gate parameterization* to *spectrum* to *downstream capability*.

Three variants, of different difficulty:

- **Measurement.** Given a trained checkpoint and a data distribution, define and estimate the spectrum — the distribution of effective memory horizons the model actually uses, not the range its parameterization permits. Currently there is no agreed estimator.
- **Method.** Given a target task mixture, choose a gate family (scalar vs. diagonal vs. matrix-valued, input-dependent vs. static, bounded vs. unbounded timescale) that dominates alternatives at matched parameters and matched training tokens. Open empirically.
- **Theory.** Prove which capability classes are gated by which spectral property. Concretely: is there a separation showing that a model whose spectrum is supported on horizons $\le H$ cannot solve a retrieval task at range $\gg H$ with $o(\text{state size} \cdot \text{range})$ bits, while a model with heavy-tailed spectrum can?

Solving it means: an estimator that is stable across seeds, a predictive law from spectrum to long-context performance, and at least one separation theorem.

## 2. Formal Setting

A gated linear recurrence over $T$ tokens with channel index $i \in [d]$ and state dimension $n$:

$$h_t^{(i)} = \alpha_t^{(i)} \odot h_{t-1}^{(i)} + b_t^{(i)} x_t^{(i)}, \qquad y_t^{(i)} = \langle c_t^{(i)}, h_t^{(i)} \rangle$$

with $\alpha_t^{(i)} \in (0,1)^n$. In Mamba, $\alpha_t = \exp(\Delta_t A)$ with $\Delta_t = \mathrm{softplus}(W_\Delta x_t)$ and $A$ diagonal negative; in HGRN2/GLA, $\alpha_t = \sigma(W x_t)$ possibly with a learned lower bound; in Mamba-2 the gate is scalar per head.

**Cumulative retention.** For $s < t$, $\quad R^{(i)}_{s\to t} = \prod_{u=s+1}^{t} \alpha_u^{(i)}$.

**Effective horizon** at tolerance $\epsilon$ (measured, per channel, per position):

$$H^{(i)}_t(\epsilon) = \max\{\,k : R^{(i)}_{t-k \to t} \ge \epsilon \,\}, \qquad \epsilon = 10^{-2}\ \text{by convention.}$$

Under a constant gate this reduces to $H(\epsilon) = \log \epsilon / \log \alpha \approx \tau \log(1/\epsilon)$ with time constant $\tau = -1/\log\alpha$.

**Forgetting spectrum.** The empirical measure over $\log_{10} H$ induced by sampling channels and positions on a held-out corpus $\mathcal{D}$:

$$\mathcal{S}_\ell(\cdot) = \mathbb{E}_{x \sim \mathcal{D}}\ \frac{1}{dT}\sum_{i,t} \delta_{\log_{10} H^{(i)}_t(\epsilon)}(\cdot)$$

per layer $\ell$. Summary statistics as actually reported: median $\tilde H$, 99th percentile $H_{99}$, and tail index $\hat\xi$ from a Hill estimator on the top decile.

**Sensitivity ground truth.** The spectrum is a proxy for the true influence
$$J^{(i)}_{s\to t} = \left\| \partial y_t / \partial x_s^{(i)} \right\|,$$
measurable directly by backprop but at $O(T)$ cost per position pair.

Assumptions, with those known violated in practice flagged:

1. *Gates are input-dependent, so $R$ is a random variable, not a model constant.* Any spectrum reported without naming $\mathcal{D}$ is undefined. **Routinely violated in reporting.**
2. *Channels are independent.* False: the output projection mixes channels, and depth composes recurrences, so a shallow-fast/deep-slow stack can hold information longer than any single $H$. **Violated.**
3. *Retention $\Rightarrow$ readability.* $R$ large does not imply the information is linearly decodable — state capacity $n$ bounds what can be stored regardless of decay. **Violated whenever the state saturates.**
4. *Diagonal, real gates.* RWKV-7 and DeltaNet-family models use non-diagonal transition matrices (identity minus rank-one), for which "the spectrum" needs eigenvalues of a product of non-commuting matrices. **Violated by the current SOTA family.**

## 3. State of the Art

**Established (ablated, reproduced).**
- Input-dependent gating beats static decay on recall-heavy tasks. Mamba's selection mechanism ablates cleanly against a non-selective S4 baseline on selective copying and induction heads (Gu & Dao, 2023).
- Forget gates transplanted into softmax attention help: FoX (Forgetting Transformer, Lin et al., ICLR 2025) trains 760M-parameter models on 48B tokens and improves long-context perplexity over a matched RoPE Transformer, with ablations over the gate form.
- Bounding the gate away from zero matters. HGRN (Qin et al., NeurIPS 2023) introduces monotonically increasing lower bounds on the forget gate by layer, and ablates that removing them costs accuracy.

**Claimed but weakly ablated.**
- That state expansion (HGRN2, Mamba-2) and gating are separable contributions. Reported gains confound state size $n$, gate family, and training recipe.
- That "learned timescales match data timescales." Asserted in several papers from $\Delta$ histograms; no controlled test where data timescale is manipulated and the spectrum is shown to move.

**Benchmark-number-only.** Nearly all long-context claims for gated recurrent LMs — RULER, BABILong, LongBench scores for Mamba-2, Griffin, Gated DeltaNet — exist as single-table numbers without a spectrum measurement, so they cannot distinguish "the gate learned a long horizon" from "the state got bigger" from "the tokenizer/recipe changed."

## 4. What Is Known

- **Timescale initialization is causal.** Tallec & Ollivier (ICLR 2018) show chrono-initialization of LSTM forget-gate biases to span $[1,T_{\max}]$ is what makes long-range tasks learnable; the same trick is why Mamba initializes $\Delta$ log-uniformly in $[0.001, 0.1]$ — i.e. $\tau$ spanning roughly $10$ to $1000$ steps at $d_{\text{model}}$-scale $\ge 768$.
- **Gates fix an optimization problem, not only a memory one.** Zucchet & Orvieto (NeurIPS 2024) show that for linear RNNs the difficulty at long horizons is a curse of memory in the *loss landscape*: eigenvalues near $1$ make gradients ill-conditioned, and gating/reparameterization is what makes them trainable.
- **Recurrent models are provably weaker on exact copying.** Jelassi et al. (ICML 2024) prove a Transformer can copy strings of length $n$ with $O(\log n)$-width constructions while any fixed-state recurrent model needs state $\Omega(n)$ bits; empirically, Mamba-370M degrades on copying beyond its trained length while a matched Transformer generalizes further.
- **State-space models are in $\mathrm{TC}^0$ under standard uniformity assumptions** (Merrill, Petty & Sabharwal, ICLR 2024) — so no gate choice buys sequential-state expressivity; gating changes *which* $\mathrm{TC}^0$ functions are learnable, not the class.
- **Scale at which the above holds:** 130M–2.8B (Mamba suite, 300B tokens), 1B–14B (Griffin/Hawk, 300B tokens), 8B (Waleffe et al., 2024, 3.5T tokens — where pure Mamba-2 trails a hybrid on in-context retrieval by wide margins on standard 5-shot suites). No spectrum measurement exists at any of these scales.

## 5. What Is Not Known

- **Methodologically blocked.** The spectrum itself. $H(\epsilon)$ depends on an arbitrary $\epsilon$, ignores channel mixing and depth composition, and is not comparable across gate families (Mamba's $\exp(\Delta A)$ vs. GLA's $\sigma(\cdot)$ vs. DeltaNet's non-diagonal transitions). No estimator has been shown seed-stable.
- **Empirically open.** Whether the realized spectrum, rather than state size $n$ or training-token count, predicts long-context accuracy. The experiment is a matched-parameter sweep at 1–3B; it is affordable and unrun.
- **Empirically open.** Whether spectra adapt to data. Train identical architectures on corpora with deliberately different dependency lengths and test whether $H_{99}$ shifts.
- **Theoretically open.** A separation of the form: for retrieval at range $r$, any model with $\mathbb{P}_{\mathcal S}[H > r] \le \delta$ and state budget $B$ has error $\ge f(\delta, B, r)$. Nothing of this shape is proved; existing lower bounds are state-size bounds indifferent to the gate.
- **Theoretically open.** Whether heavy-tailed (power-law) spectra are optimal for natural language, whose token-dependency statistics are themselves power-law.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability plus a confound**. Retention $R_{s\to t}$ and readout $c_t$ are only identified jointly: rescaling $\alpha$ toward $1$ while shrinking the input gate $b$ and rescaling $c$ leaves $y$ almost unchanged over the trained range, so two checkpoints with very different measured spectra can be the same function. There is no canonical gauge in which "how long this model remembers" is a property of the weights.

Compounding this, the standard comparison — Mamba-2 vs. GLA vs. Gated DeltaNet — changes gate family, state size, head structure and training recipe together. Long-context benchmarks then supply the second obstruction: RULER and needle-style tests measure *retrieval given a prompt format*, which for recurrent models is dominated by state capacity $n \cdot d_{\text{head}}$ in bits, not by decay. An evaluation that names "long-range memory" is largely reporting state capacity.

## 7. Current Research (as of 2026)

- **Non-diagonal gating.** Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) and RWKV-7 combine a decay gate with a delta-rule update; the transition is identity-minus-rank-one, so the effective spectrum is no longer read off a diagonal. Extending spectrum estimation to this family is active *(frontier — verify)*.
- **Gates in attention.** FoX and follow-on gated-attention work at Meta/academic labs test whether the forgetting inductive bias is architecture-independent.
- **Hybrids as an implicit answer.** Nvidia, AI21 (Jamba) and Mistral ship interleaved attention+SSM stacks, effectively conceding that the gate cannot supply exact long-range recall; the open question is the minimal attention fraction, empirically around 1-in-6 to 1-in-8 layers *(frontier — verify)*.
- **Theory of learnable timescales.** Orvieto/Zucchet-line work on landscape conditioning and on what gating buys over reparameterization.

## 8. Concrete Next Experiment

**Question:** does the realized forgetting spectrum predict long-context accuracy after controlling for state capacity?

- **Scale.** 8 models at $\approx$1.3B parameters, 100B tokens each of a fixed corpus, 4K training context. Roughly $8 \times 10^{21}$ FLOPs total — days on 64 H100s.
- **Grid.** $2 \times 2 \times 2$: gate family (Mamba-2 scalar-per-head vs. GLA diagonal) $\times$ state bits (matched at $2^{16}$ and $2^{18}$ per layer) $\times$ gate lower bound ($\alpha \ge 0$ vs. $\alpha \ge 1 - 2^{-6}$ in the top half of layers).
- **Control arm.** A matched-parameter, matched-token Transformer with sliding-window attention of width equal to the median measured $\tilde H$ of the recurrent models. This is the arm that separates "the gate learned a horizon" from "the model has $n$ bits of state."
- **Measurement.** Estimate $\mathcal{S}_\ell$ on 512 held-out documents; report $\tilde H$, $H_{99}$, Hill $\hat\xi$; verify against direct Jacobian influence $J_{s\to t}$ on 64 position pairs.
- **Deciding number.** Partial $R^2$ of $\log H_{99}$ on RULER-style retrieval accuracy at 32K, after regressing out $\log(\text{state bits})$ and $\log(\text{tokens})$. **Threshold: partial $R^2 \ge 0.5$ across the eight points supports "spectrum predicts capability"; $\le 0.1$ says the spectrum is a bookkeeping artifact and state capacity is the whole story.** Seed variance must be estimated from 3 seeds on one cell; if seed spread in $H_{99}$ exceeds 0.3 dex, the measurement is not yet well posed and the methodological blocker in §5 stands.

## 9. Key References

- **[Foundational]** F. Gers, J. Schmidhuber, F. Cummins. *Learning to Forget: Continual Prediction with LSTM.* Neural Computation 12(10), 2000.
- **[Foundational]** C. Tallec, Y. Ollivier. *Can Recurrent Neural Networks Warp Time?* ICLR 2018. — arXiv:1804.11188
- **[Foundational]** A. Gu, K. Goel, C. Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, S. De. *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML 2023. — arXiv:2303.06349
- **[SOTA]** A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024 (arXiv 2023). — arXiv:2312.00752
- **[SOTA]** T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** S. Yang, B. Wang, Y. Shen, R. Panda, Y. Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. — arXiv:2312.06635
- **[SOTA]** Z. Qin, S. Yang, Y. Zhong. *Hierarchically Gated Recurrent Neural Network for Sequence Modeling.* NeurIPS 2023. — arXiv:2311.04823
- **[SOTA]** S. De, S. L. Smith, A. Fernando, A. Botev, et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427
- **[SOTA]** S. Yang, J. Kautz, A. Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. — arXiv:2412.06464
- **[SOTA]** Z. Lin, E. Nikishin, X. He, A. Courville. *Forgetting Transformer: Softmax Attention with a Forget Gate.* ICLR 2025. — arXiv:2503.02130
- **[Theory]** W. Merrill, J. Petty, A. Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Theory]** S. Jelassi, D. Brandfonbrener, S. Kakade, E. Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Theory]** N. Zucchet, A. Orvieto. *Recurrent Neural Networks: Vanishing and Exploding Gradients Are Not the End of the Story.* NeurIPS 2024. — arXiv:2405.21064
- **[Survey/Benchmark]** C.-P. Hsieh, S. Sun, S. Kriman, et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. — arXiv:2404.06654
- **[Empirical]** R. Waleffe, W. Byeon, D. Riach, et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887

## 10. Worked Example

Take Mamba's default initialization: $\Delta \sim \mathrm{LogUniform}[0.001, 0.1]$, $A_{jj} = -j$ for $j = 1..16$. Then $\alpha = \exp(-\Delta j)$ and $\tau = 1/(\Delta j)$, so per-channel time constants span

$$\tau \in [\,1/(0.1 \cdot 16),\ 1/(0.001 \cdot 1)\,] = [0.63,\ 1000] \text{ tokens},$$

i.e. $H(10^{-2}) = \tau \ln 100 \in [2.9,\ 4600]$. At initialization the spectrum is log-uniform over about 3.2 decades — by construction, not by learning.

Now the obstruction. Suppose training drives one head to $\Delta_t = 0.002$ on average, giving $H_{99} \approx 2300$ tokens, and RULER-4K accuracy is high. Is the gate responsible? Apply the gauge transformation: set $\alpha' = \alpha^{1/2}$ (halving $\tau$, so $H_{99} \to 1150$), $b' = b$, and rescale the readout $c'_t$ to re-fit the same outputs. Over the range where the recurrence is dominated by a single recently-written value — which is the regime of needle retrieval, where one token's contribution is much larger than the accumulated background — the readout rescaling absorbs most of the change, and measured 4K retrieval moves little. The measured $H_{99}$ halved; the capability did not.

Concretely: with $n = 16$ state slots per head, $d_{\text{head}} = 64$ and 4-bit effective precision, the head holds $\approx 16 \cdot 64 \cdot 4 = 4096$ bits. A 4K-token needle task needs to store one $\sim$40-bit key–value pair. Capacity is $100\times$ oversupplied, so retrieval succeeds for any $H_{99} \gtrsim$ the needle distance and fails below it, roughly as a step function — which is exactly why a regression of accuracy on $\log H_{99}$ across a handful of checkpoints can look either perfectly predictive or perfectly flat depending on where the needles sit relative to the step. That is the confound §8 is designed to break: sweep state bits and horizon on separate axes, and require the sliding-window control to place the step.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*