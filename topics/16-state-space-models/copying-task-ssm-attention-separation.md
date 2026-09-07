---
id: 16-state-space-models/copying-task-ssm-attention-separation
title: "Copying Task Separation Between SSMs and Attention"
topic: 16-state-space-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Copying Task Separation Between SSMs and Attention

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/copying-task-ssm-attention-separation` · **Status:** partially-solved

## 1. Problem Statement

A model reads a string $x \in V^L$ followed by a separator, and must emit $x$ verbatim. Attention can do this by construction: a query at output position $i$ retrieves the key at input position $i$. A fixed-size recurrent state cannot store an arbitrary string of unbounded length. The question is where exactly the line falls, and whether it matters for language models.

Three variants, routinely conflated:

- **Theory.** For a state-space model (SSM) with recurrent state of $S$ bits, what is the largest $L$ for which exact copying is representable? Is the bound tight, and does it survive if the model may use chain-of-thought or multiple passes?
- **Measurement.** Copying is a proxy for *in-context retrieval*: induction heads, associative recall, few-shot format-following, long-document QA. Does the copying gap predict the downstream gap, and at what conversion rate?
- **Method.** How much attention must be added to a recurrent backbone to close the gap, and does that addition cost the linear-time inference property that motivated the SSM?

Solved: the representational separation, up to constants. Open: whether the empirical failures of trained SSMs are caused by that separation or by optimization, and how much attention is *necessary* rather than merely sufficient.

## 2. Formal Setting

Vocabulary $V$, $|V| = v$. Input $x_{1:L} \sim \mathrm{Unif}(V^L)$, prompt $x_{1:L} \Vert \texttt{[COPY]}$, target $x_{1:L}$.

**State size.** For a recurrent layer with hidden state $h_t \in \mathbb{R}^{d_s}$ stored at $b$ bits per entry, over $\ell$ layers:
$$S \;=\; \ell \cdot d_s \cdot b \quad \text{bits}.$$
For Mamba-2 with model width $d$, expansion $e$, head state dimension $N$: $d_s = e\,d\,N$ per layer. This is the quantity that actually gets measured — allocate the KV cache of a transformer the same way and it grows as $\Theta(L)$, which is the entire asymmetry.

**Objective.** String-exact accuracy
$$\mathrm{Acc}(L) \;=\; \Pr_{x \sim \mathrm{Unif}(V^L)}\!\left[\hat{y}_{1:L} = x_{1:L}\right],$$
under greedy decoding. Report also per-token accuracy $\frac1L\sum_i \Pr[\hat y_i = x_i]$; the two diverge sharply and papers are not consistent about which they plot.

**Length generalization.** Train on $L \le L_{\mathrm{tr}}$, evaluate at $L > L_{\mathrm{tr}}$. Define the breaking length
$$L^\star(\varepsilon) \;=\; \max\{L : \mathrm{Acc}(L) \ge 1-\varepsilon\},\qquad \varepsilon = 0.1 .$$

**Information-theoretic bound.** Any model whose entire dependence on $x_{1:L}$ passes through $S$ bits satisfies, by Fano,
$$\mathrm{Acc}(L) \;\le\; 2^{\,S - L\log_2 v} ,$$
so exact copying requires $S \ge L\log_2 v$. This is the counting argument behind the SSM lower bound.

**Assumptions, and where they break.**
1. *Uniform tokens.* Real text is compressible; the $\log_2 v$ factor is an overestimate for natural strings. Violated in every downstream use.
2. *State is the only channel.* Convolutional/short-window paths (Mamba's depthwise conv, sliding-window layers in hybrids) carry information outside $h_t$. Violated in all deployed hybrids.
3. *Single forward pass, no scratchpad.* With chain-of-thought the model can re-emit and re-read; the bound applies per pass, not to the composite.
4. *Full precision arithmetic.* Bounds assume $b$ usable bits; in practice bf16 selective-scan states are noisy well before $b=16$ bits of capacity are used.

## 3. State of the Art

**Theory (established).** Jelassi, Brandfonbrener, Kakade, Malach, *Repeat After Me: Transformers are Better than State Space Models at Copying* (ICML 2024, arXiv:2402.01032). Two results: (i) a two-layer transformer using hash-based $n$-gram matching copies strings of length exponential in its parameter count, with a construction independent of $L$ up to positional-encoding precision; (ii) any generalized state-space model with $S$-bit state fails to copy strings longer than $\Theta(S)$ bits. The separation is exponential in parameters and is a proof, not a benchmark.

**Theory (complementary).** Merrill, Petty, Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024, arXiv:2404.08819): SSMs, like transformers, are in uniform $\mathsf{TC}^0$. Depth/circuit class does *not* separate the families — memory does. Wen, Dang, Lyu, *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval* (arXiv:2402.18510) show the gap closes with chain-of-thought for some tasks but not for index-lookup-style retrieval unless memory grows.

**Empirical (established).** Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models* (ICLR 2024, arXiv:2312.04927) attribute most of the perplexity gap between gated-convolution models and attention to associative-recall tokens. *Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff* ("Based", ICML 2024, arXiv:2402.18668) establishes recall accuracy as an increasing function of recurrent state size at fixed training, and builds the Pareto frontier.

**Empirical (claimed but under-ablated).** Hybrid architectures — Jamba (Lieber et al. 2024, arXiv:2403.19887), Samba (Ren et al. 2024, arXiv:2406.07522), Griffin (De et al. 2024, arXiv:2402.19427), Zamba — report that a small fraction of attention layers recovers retrieval. Almost none ablate *which* layers, *how many*, or whether sliding-window attention suffices for a target context length. The claim "a few attention layers are enough" is a benchmark number, not a characterized boundary.

## 4. What Is Known

- **Trained-model gap.** Jelassi et al. (2024): at ~360M–410M parameters, Pythia transformers reach near-perfect string copying at input lengths where equally sized Mamba models collapse; the transformer advantage grows with $L$, and transformers with hard-ALiBi positional encoding generalize far beyond the training length while Mamba does not. Measured on synthetic uniform strings, models under 1B parameters.
- **Retrieval in pretrained models.** Same paper: on a phone-book lookup task, Pythia-2.8B (pretrained on the Pile) beats Mamba-2.8B (same data, same token count) once the phone book exceeds a few hundred entries. Both degrade; the SSM degrades earlier.
- **At 8B scale.** Waleffe et al., *An Empirical Study of Mamba-based Language Models* (arXiv:2406.07887, NVIDIA): 8B-parameter Mamba, Mamba-2, Mamba-2-Hybrid and transformer, each trained on 3.5T tokens. The hybrid (Mamba-2 layers plus a small number of self-attention and MLP layers) exceeds the pure transformer by **+2.65 points** averaged over 12 standard tasks and matches or beats it on 23 long-context tasks at 16K–32K; pure Mamba-2 lags on the retrieval-heavy subset, notably five-shot MMLU and phone-book lookup. This is the strongest apples-to-apples evidence and it says the gap is real, mild, and cheaply fixable.
- **State size is the knob.** Based/Zoology: recall accuracy rises monotonically with recurrent state size across two orders of magnitude, matching the $S \gtrsim L\log v$ shape rather than any depth- or parameter-count explanation.
- **In-context learning.** Park et al., *Can Mamba Learn How to Learn?* (ICML 2024, arXiv:2402.04248): Mamba matches transformers on regression-style ICL but trails on retrieval-style ICL; a hybrid ("MambaFormer") matches on both.

## 5. What Is Not Known

- **Empirically open.** No controlled study isolates *how much* attention is necessary. The runnable experiment — sweep the number and placement of attention layers at fixed total parameters and fixed tokens, and measure $L^\star$ — has not been run at $\ge$7B with a common data recipe. Cost is the only barrier.
- **Empirically open.** Whether the copying gap *causes* the downstream gap. The correlation is established; no intervention experiment (fix copying, observe downstream) exists.
- **Theoretically open.** Whether trained SSMs achieve their $\Theta(S)$ capacity. The lower bound says they cannot exceed it; nothing says they approach it. Observed $L^\star$ is orders of magnitude below the counting bound (§10) and no theory explains the deficit.
- **Theoretically open.** Tight bounds for input-dependent (selective) state transitions with $C$ chain-of-thought steps: is copying possible with $S = \tilde O(\log L)$ and $C = O(L)$?
- **Methodologically blocked.** "Retrieval ability" has no agreed measurement. Needle-in-a-haystack, phone book, MAD synthetic suites (Poli et al., *Mechanistic Design and Scaling of Hybrid Architectures*, arXiv:2403.17844), and RULER give different orderings of the same models. There is no calibration mapping any of them to $S$.

## 6. Why It Is Hard

**Confounded measurement, in a specific way.** Copying accuracy at a given $L$ mixes three independent factors: state capacity $S$, positional encoding / length-generalization mechanism, and optimization. The transformer advantage in Jelassi et al. is partly a positional-encoding result — hard-ALiBi versus NoPE changes transformer $L^\star$ by an order of magnitude on the same architecture. SSMs have no positional encoding to swap, so an architecture comparison silently compares two different length-generalization schemes and reports the sum as a family separation.

**Non-identifiability of the fix.** A hybrid that works does not tell you why: the attention layers may be supplying retrieval, or supplying softmax sharpness, or merely widening the effective residual stream. Ablating attention layers out of a trained hybrid does not answer this — retraining is required for each configuration, at $\sim10^{22}$ FLOPs per arm at 7B/1T tokens.

**Absent ground truth downstream.** There is no labelled set of "tokens whose prediction requires in-context retrieval" in natural text. Zoology's AR-token attribution is a heuristic (repeated bigrams), and it is the best available.

## 7. Current Research (as of 2026)

- **Hybrid ratio search.** NVIDIA, AI21 (Jamba line), Microsoft (Samba), Google DeepMind (Griffin/Hawk lineage) continue to publish hybrids with hand-chosen ratios near 1 attention layer per 6–8 recurrent layers. The ratio is empirical, not derived. *(frontier — verify current ratios per release.)*
- **Larger and better-conditioned state.** DeltaNet-style and gated delta-rule updates (Yang, Kautz, Hatamizadeh and collaborators) target higher effective capacity per state bit rather than more bits.
- **Test-time / data-order remedies.** Arora et al., *Just Read Twice: Closing the Recall Gap for Recurrent Language Models* (2024): a second pass over context recovers a large part of the recall gap without architecture change — evidence the deficit is partly a *scheduling* problem, not only a capacity one.
- **Theory of selectivity.** Extensions of the $\mathsf{TC}^0$ and communication-complexity analyses to input-dependent transitions and to chain-of-thought-augmented recurrence. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question:** is the SSM copying deficit capacity-bound or optimization-bound?

**Scale.** Six models at 1.3B parameters, 100B tokens each (~$10^{21}$ FLOPs per arm, roughly 2k A100-days total). Arms: pure Mamba-2 at head-state $N \in \{16, 64, 256\}$ (state size $S$ varies $16\times$ at nearly constant parameter count), pure transformer, and Mamba-2 ($N{=}16$) hybrids with 1 and 3 interleaved attention layers. Identical data order, tokenizer, and optimizer.

**Control arm.** The $N{=}16$ pure Mamba-2 model, plus a *capacity oracle*: for each arm compute the counting-bound length $L_{\max} = S/\log_2 v$.

**Measured number.** The ratio
$$\rho \;=\; \frac{L^\star(0.1)}{L_{\max}}$$
on uniform-string copying after identical copy-task fine-tuning (1B tokens).

**Decision rule.** If $\rho$ is roughly constant across $N \in \{16,64,256\}$, the deficit is capacity-bound and scaling state size is the fix — the theory transfers. If $\rho$ falls as $N$ grows (large states left unused), the deficit is optimization-bound, the lower bound is not the operative constraint at deployed sizes, and hybridization is a workaround for a training problem. Secondary read-out: does the 1-attention-layer hybrid reach transformer $L^\star$? If yes, "how much attention is necessary" has the answer *one layer*, and the architecture question narrows to placement.

## 9. Key References

- **[Foundational]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Foundational]** Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Theory]** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[Theory]** Wen, Dang, Lyu. *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval.* 2024. — arXiv:2402.18510
- **[SOTA, empirical]** Waleffe et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887
- **[SOTA, measurement]** Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[SOTA, method]** Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[Related]** Park, Park, Xiong, Lee, Cho, Oymak, Lee, Papailiopoulos. *Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks.* ICML, 2024. — arXiv:2402.04248
- **[Survey/design]** Poli et al. *Mechanistic Design and Scaling of Hybrid Architectures.* 2024. — arXiv:2403.17844
- **[Systems]** Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* 2024. — arXiv:2403.19887

## 10. Worked Example

Take a Mamba-2 backbone with $d = 2048$, expansion $e = 2$, head state dimension $N = 128$, $\ell = 24$ layers, bf16 states.

Per-layer state entries: $e\,d\,N = 2 \times 2048 \times 128 = 524{,}288$. Over 24 layers: $1.26 \times 10^7$ entries. At $b = 16$ bits:
$$S \;=\; 1.26 \times 10^7 \times 16 \;\approx\; 2.0 \times 10^8 \text{ bits}.$$

Copying budget with $v = 50{,}257$ ($\log_2 v = 15.6$ bits/token):
$$L_{\max} \;=\; \frac{2.0\times 10^8}{15.6} \;\approx\; 1.3 \times 10^7 \text{ tokens}.$$

The counting bound permits copying 13 million random tokens. Trained Mamba models of this size break well below $10^3$ tokens on uniform-string copying — a shortfall of four orders of magnitude, $\rho \approx 10^{-4}$.

Even discounting brutally — assume only 4 usable bits per bf16 entry and that a single layer carries the copy — $L_{\max}$ is still $\approx 1.3\times10^5$ tokens, two orders above what is observed.

**The obstruction is visible here.** The theorem that everyone cites as the explanation for the empirical gap is not binding at any size anyone trains. The lower bound is correct and the empirical gap is real, but at deployed state sizes they are not the same phenomenon: the proof constrains $L \sim 10^7$, the models fail at $L \sim 10^2$. Whatever stops a trained Mamba from copying 500 tokens is an optimization or representation failure that no current theory names — and it is why "SSMs can't copy because their state is finite" is a satisfying sentence that does not predict the measurement. §8 exists to separate the two.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*