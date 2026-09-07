---
id: 09-model-design/attention-ssm-expressivity-separation
title: "Expressivity Separation Between Attention and State-Space Models"
topic: 09-model-design
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expressivity Separation Between Attention and State-Space Models

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/attention-ssm-expressivity-separation` · **Status:** partially-solved

## 1. Problem Statement

Softmax attention keeps a cache that grows with sequence length; a state-space model (SSM) keeps a fixed-size recurrent state. The question is what this buys and what it costs.

Three variants, routinely conflated:

- **Theory variant.** Is there a function family separating the two model classes — computable by a transformer of size polynomial in $n$ but not by any SSM of that size, or vice versa? Solving it means a theorem with matched parameter counts, matched precision, and a stated uniformity condition.
- **Measurement variant.** Given two trained models, is an observed task gap caused by *capacity* (the SSM's state cannot hold the required information) or by *optimization* (it could, but gradient descent did not find the circuit)? Solving it means a test that discriminates the two causes.
- **Method variant.** Given a fixed state budget $S$ bits and a fixed FLOP budget, what is the best achievable recall/throughput frontier, and does a hybrid attain it? Solving it means a Pareto curve, not a leaderboard row.

The theory variant is largely settled for the coarse question (circuit class) and settled for the sharp question (fixed-state copying). The measurement variant is open and is where the field is stuck.

## 2. Formal Setting

Input: a sequence $x_{1:n} \in \Sigma^n$ over a finite vocabulary, $|\Sigma| = V$. Output: $y_{1:n}$, or a decision $f(x_{1:n}) \in \{0,1\}$.

**SSM layer.** With input projections $u_t \in \mathbb{R}^{d}$ and per-channel state $h_t \in \mathbb{R}^{d \times N}$:
$$h_t = A_t \odot h_{t-1} + B_t u_t^\top, \qquad y_t = C_t^\top h_t,$$
where $A_t, B_t, C_t$ may depend on $x_t$ (selective / input-dependent, as in Mamba) or not (S4, linear time-invariant).

**Measured state budget.** The quantity that matters is not $N$ but bits:
$$S \;=\; L \cdot d \cdot N \cdot p,$$
$L$ layers, $p$ bits per stored scalar (bf16 $\Rightarrow p=16$). Measure it by reading tensor shapes and dtypes at inference, not from the paper's headline $N$.

**Measured attention budget.** KV cache bits $S_{\text{attn}}(n) = 2 L H d_h n p$ — linear in $n$, which is exactly the asymmetry under test.

**Task information content.** For a task requiring verbatim retrieval of a span of $\ell$ tokens drawn uniformly from $\Sigma$, the required carried information is $I = \ell \log_2 V$ bits. The capacity claim is: any streaming model with $S < I$ fails, by a one-way communication-complexity argument (Jelassi et al., 2024).

**The decisive ratio.** Define state-utilization efficiency
$$\eta \;=\; \frac{\ell^{*} \log_2 V}{S},$$
where $\ell^{*}$ is the longest span the trained model retrieves at $\ge 90\%$ exact-match. $\eta = 1$ means the model saturates its information-theoretic bound; $\eta \ll 1$ means the observed gap is *not* a capacity separation.

**Assumptions, and which are violated.**
- *Finite precision, $p$ fixed.* Holds in deployment; violated in most expressivity proofs, which assume $O(\log n)$ or unbounded precision. Under infinite precision a single recurrent state is Turing-complete, so all separations vanish.
- *Uniformity.* Circuit-class results need log-space-uniform families; a per-$n$ handcrafted construction proves nothing about a trained model.
- *Trained by gradient descent from random init.* Never assumed in the theory, always true of the artifact being measured. This is the gap in §5.
- *Matched data.* Empirical comparisons assume identical tokenizer, data order and token count. Violated in most published head-to-heads.

## 3. State of the Art

**Theory SOTA — established.**
- Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): S4- and Mamba-style SSMs with finite precision are simulable in uniform $\mathsf{TC}^0$. They cannot solve $\mathsf{NC}^1$-hard problems such as the $S_5$ word problem unless $\mathsf{TC}^0 = \mathsf{NC}^1$. Transformers sit in the same class (Merrill & Sabharwal, TACL 2023). **At the level of circuit classes there is no separation.**
- Jelassi et al., *Repeat After Me: Transformers are Better than State Space Models at Copying* (ICML 2024): a fixed-state model with $S$ bits cannot copy strings carrying more than $S$ bits; a transformer with $O(\log n)$-width hash-based construction copies strings of length exponential in its parameter count. This is a genuine, proved separation — but it is a separation in *memory*, parameterized by $S$, not in computational class.
- Bhattamishra, Hahn, Blunsom & Kanade, *Separations in the Representational Capabilities of Transformers and Recurrent Architectures* (NeurIPS 2024): index lookup and nearest-neighbour need $\Theta(\log n)$ transformer width but $\Omega(n)$ recurrent state.
- Sarrof, Veitsman & Hahn, *The Expressive Capacity of State Space Models: A Formal Language Perspective* (NeurIPS 2024): finite-precision SSMs capture the star-free regular languages, matching soft-attention transformers, with a learnability barrier beyond.
- Dao & Gu, *Transformers are SSMs* (ICML 2024): linear attention and semiseparable-matrix SSMs are the same object. Any separation must therefore come from the softmax, not from the recurrence.

**Empirical SOTA — established but narrow.** Waleffe et al. (NVIDIA, 2024) trained Mamba-2, Transformer and a Mamba-2-Hybrid at 8B parameters on 3.5T tokens with identical data. The hybrid beat the transformer by 2.65 points averaged over 12 standard tasks; pure Mamba-2 lagged specifically on 5-shot MMLU and on phonebook-style lookup. This is the cleanest matched-data comparison published.

**Claimed but unablated.** That hybrid architectures (Jamba, Griffin, RecurrentGemma) "close the gap" is a benchmark number, not a mechanism: none isolate *which* attention layers carry the retrieval and none report $\eta$. Claims that state expansion alone fixes recall rest on ≤1.4B-scale sweeps.

## 4. What Is Known

- **No circuit-class separation.** Both families are in uniform $\mathsf{TC}^0$ under finite precision (ICML 2024, TACL 2023). Neither does $S_5$ composition in a bounded number of layers.
- **Copying separates, with numbers.** Jelassi et al. compared Pythia-2.8B and Mamba-2.8B trained on the same Pile data: transformer copy accuracy stayed high hundreds of tokens past the training length; Mamba's collapsed near it. Measured at 2.8B parameters.
- **Recall–throughput is a real frontier.** Arora et al., *Zoology* and *Based* (ICML 2024), showed at 360M parameters that gated-convolution models' associative-recall error tracks state size, and that adding a small sliding-window attention window recovers most of the gap at a fraction of the KV cache.
- **Hybrids beat both pure arms at 8B/3.5T** (Waleffe et al., 2024), with ~8× faster generation than the transformer control.
- **Chain of thought moves the boundary.** Merrill & Sabharwal (ICLR 2024) and Wen et al. (2024) show intermediate tokens let bounded-memory recurrent models simulate retrieval they cannot do in one pass — so any separation claim must fix the decoding protocol.

## 5. What Is Not Known

- **Theoretically open.** Whether a separation exists for *matched total memory*: given a transformer restricted to a KV cache of $S$ bits (sliding window or compressed) and an SSM with $S$ bits of state, is there a function family separating them? Every proved separation to date exploits the transformer's unbounded cache.
- **Theoretically open.** Whether the star-free characterization survives input-dependent (selective) $A_t$ with realistic precision, and whether SSM depth trades off against state size the way transformer depth trades off against width.
- **Empirically open.** Whether the 8B/3.5T hybrid advantage is a capacity effect or a data-efficiency effect: no matched-$S$, matched-FLOP, matched-token run exists above 8B.
- **Methodologically blocked.** There is no accepted measurement that separates "the state cannot hold it" from "training did not learn to use it." $\eta$ is proposed here; nobody publishes it. Without it, "expressivity gap" in the empirical literature names something the experiment does not measure.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement**, and it is quantitative, not rhetorical.

The information-theoretic bound is loose by four to five orders of magnitude on real models (see §10). Every empirically observed copy/recall failure therefore occurs deep inside the capacity-feasible region. The observed gap is consistent with *both* an unknown tighter capacity bound *and* an optimization failure, and no published experiment discriminates them.

Secondary obstructions: (a) cost — a matched 8B/3.5T triple is roughly $10^{23}$ FLOPs per arm, so few groups can run the control; (b) the theory's precision assumption ($O(\log n)$ bits) does not match bf16 inference, and results flip sign across that boundary; (c) benchmark contamination means MMLU-style gaps confound retrieval with memorized knowledge.

## 7. Current Research (as of 2026)

- **Hybrid ratio search.** AI21 (Jamba), Google DeepMind (Griffin/RecurrentGemma), NVIDIA (Mamba-2-Hybrid) converged empirically on roughly 1 attention layer in 6–8. No published theory predicts the ratio. *(frontier — verify the current best ratio at ≥8B.)*
- **Formal-language probing of selective SSMs** — Hahn's group (Saarland) and Merrill/Sabharwal (AI2) extending star-free and $\mathsf{TC}^0$ results to gated linear recurrences and to test-time-training layers.
- **Delta-rule and state-expansion variants** (DeltaNet, Gated DeltaNet lines) aiming to raise effective $\eta$ without linear cache growth. *(frontier — verify.)*
- **State-bit-matched comparisons** against compressed KV caches. Rare; the natural control arm and the one most missing.

## 8. Concrete Next Experiment

**Question:** is the transformer–SSM retrieval gap capacity or optimization?

- **Scale.** 1.3B parameters, 100B tokens, identical data order and tokenizer. Four arms, each ~$2\times10^{21}$ FLOPs; ~2 000 H100-hours total.
- **Arms.**
  1. Mamba-2, $N = 128$.
  2. Mamba-2, $N = 512$ (4× state bits, same parameter count to within 2%).
  3. **Control arm:** transformer with sliding-window attention, window $w$ chosen so the KV cache in bits equals arm 1's $S$ exactly. This is the bit-matched control the literature lacks.
  4. Full-attention transformer (upper reference).
- **Protocol.** Single-pass decoding, no chain of thought. Task: verbatim copy of a uniform random string over $|\Sigma| = 32$, evaluated at lengths $\ell \in \{32, 64, \dots, 8192\}$, with copying present in pretraining data at $\ell \le 256$ only.
- **The deciding number.** $\eta = \ell^{*}\log_2 V / S$ for arms 1, 2, 3, where $\ell^{*}$ is the longest length with $\ge 90\%$ exact match.
  - If $\eta_{\text{SSM}} \approx \eta_{\text{window}}$ and both $\ll 1$: the gap is **not** capacity; it is shared optimization failure, and the separation literature is measuring the wrong thing.
  - If $\eta_{\text{window}} \gg \eta_{\text{SSM}}$ at equal $S$: a genuine architectural separation at matched memory — currently unproved and unmeasured.
  - If $\eta_2 \approx 4\,\eta_1$ with $\ell^*$ scaling linearly in $S$: capacity is binding and state expansion is the fix.

## 9. Key References

- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[SOTA — theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[SOTA — theory]** Samy Jelassi, David Brandfonbrener, Sham Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[SOTA — theory]** Satwik Bhattamishra, Michael Hahn, Phil Blunsom, Varun Kanade. *Separations in the Representational Capabilities of Transformers and Recurrent Architectures.* NeurIPS 2024.
- **[SOTA — theory]** Yash Sarrof, Yana Veitsman, Michael Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS 2024.
- **[SOTA — unification]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA — empirical]** Roger Waleffe et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887
- **[SOTA — empirical]** Simran Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML 2024. — arXiv:2402.18668
- **[Foundational]** William Merrill, Ashish Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL 2023.
- **[Survey]** Soham De et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427

## 10. Worked Example

Take Mamba-2.8B: $d_{\text{model}} = 2560$, expansion 2 $\Rightarrow d = 5120$ channels, $N = 16$, $L = 64$ layers, bf16.

$$S = 64 \times 5120 \times 16 \times 16 \approx 8.4 \times 10^{7} \text{ bits} \;=\; 10.5\ \text{MB}.$$

For copying a uniform string over $|\Sigma| = 32$ ($\log_2 V = 5$ bits/token), the communication bound permits
$$\ell_{\max} = \frac{8.4\times10^{7}}{5} \approx 1.7 \times 10^{7} \text{ tokens}.$$

Measured behaviour (Jelassi et al., 2024, same model): exact-match copying collapses within a few hundred tokens of the training length — call it $\ell^{*} \approx 300$.

$$\eta = \frac{300 \times 5}{8.4\times10^{7}} \approx 1.8 \times 10^{-5}.$$

The model uses about **one part in 55 000** of its provable capacity. The obstruction is now visible: the theorem that "SSMs cannot copy" is true, but it bites at 17 million tokens while the artifact fails at 300. The proved separation and the observed separation are five orders of magnitude apart, so the observed gap is not evidence for the theorem — it is evidence about optimization, tokenizer, or training distribution, none of which the theorem mentions. Any paper that cites the copying bound to explain a benchmark gap at $n = 4096$ is asserting a link that this number does not support. Reporting $\eta$ alongside every claimed separation would make that visible by default.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*