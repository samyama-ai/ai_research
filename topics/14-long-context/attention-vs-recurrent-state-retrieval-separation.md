---
id: 14-long-context/attention-vs-recurrent-state-retrieval-separation
title: "Provable Separation Between Softmax Attention and Recurrent State for Long-Range Retrieval"
topic: 14-long-context
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Separation Between Softmax Attention and Recurrent State for Long-Range Retrieval

> **Topic:** Long Context · **ID:** `14-long-context/attention-vs-recurrent-state-retrieval-separation` · **Status:** partially-solved

## 1. Problem Statement

Does softmax attention hold a *real* advantage over fixed-size recurrent state on long-range retrieval, or only a memory-budget advantage that vanishes once both sides are charged for the bytes they carry?

Three variants, routinely conflated:

- **Theory variant.** Exhibit a task family $\{T_L\}$ and prove: (a) a transformer of depth $O(1)$ and width $\mathrm{polylog}(L)$ solves $T_L$; (b) *every* recurrent model whose per-token state is $S$ bits fails $T_L$ unless $S = \Omega(L)$. Status: **solved** for copying and index-style retrieval. The open form fixes $S$ equal to the transformer's own inference-time cache and asks whether a separation survives.
- **Method variant.** Find a recurrent or hybrid architecture that matches attention's retrieval accuracy at state size $S \ll L$, or prove none exists in a defined class (gated linear attention, delta rule, low-rank SSM).
- **Measurement variant.** Define a retrieval benchmark whose score is a function of the architecture's memory capacity and not of tokenizer, position encoding, training-length extrapolation, or prompt format. Currently not well defined.

Solving it means: a theorem separating the classes at *matched inference state*, plus an experiment whose measured state-size exponent confirms or refutes it.

## 2. Formal Setting

Sequence $x_{1:L} \in \Sigma^L$, $|\Sigma| = V$. A causal sequence model computes $y_t = g(h_t, x_t)$ with $h_t = f(h_{t-1}, x_t)$.

**State size (measured, not nominal).** $S(L)$ is the number of bits the model must carry across the position-$t$ boundary to emit tokens $>t$:
$$S(L) \;=\; \max_t \; \log_2 \big|\{\,\text{distinguishable } h_t \text{ over inputs } x_{1:t}\,\}\big|.$$
Measured in practice as bytes of runtime cache per sequence at batch 1, fp16, excluding weights: KV cache $2 L d_{\text{kv}} n_{\text{layer}} \cdot 2$ bytes for attention; $d_{\text{state}} d_{\text{model}} n_{\text{layer}} \cdot 2$ bytes for Mamba-style SSMs; window $\times$ heads for sliding-window attention.

**Retrieval task (MQAR).** Multi-query associative recall: $N$ key–value pairs $(k_i, v_i)$ placed in a length-$L$ context, then $Q$ queries. Accuracy $\mathrm{Acc}(L, N, Q)$ = fraction of queries whose value token is argmax-decoded correctly.

**Copy task.** Input $s \in \Sigma^n$ followed by a separator; output $s$. Exact-match accuracy.

**The quantity of interest.** For target accuracy $1-\epsilon$, the minimum state
$$S^\star(L, \epsilon) \;=\; \min\{\, S : \exists \text{ model in class } \mathcal{C} \text{ with state } S,\ \mathrm{Acc} \ge 1-\epsilon \,\},$$
and its scaling exponent $\alpha$ in $S^\star(L,\epsilon) \asymp L^{\alpha}$. **A separation exists iff $\alpha_{\text{attn}} < \alpha_{\text{rec}}$ at fixed $\epsilon$.** Note $\alpha_{\text{attn}} = 1$ for vanilla attention — its cache is linear in $L$ too. This is the crux the literature usually skips.

**Assumptions and their violations.**
- *Finite precision.* Lower bounds assume $p$-bit state; real SSMs use bf16, and the "infinite-precision analog memory" loophole is closed in practice but not always in the proofs. Violated in the permissive direction for theory.
- *Single pass, no chain of thought.* Wen et al. (2024) show CoT collapses the gap; deployed models use CoT. **Known violated.**
- *Trained to optimality.* Lower bounds are representational; upper bounds are constructions, not gradient-descent outcomes. Learned models sit strictly below both. **Known violated.**
- *Uniform key distribution.* Real retrieval keys are Zipfian and compressible, so worst-case $\Omega(L)$ bounds do not bind. **Known violated.**

## 3. State of the Art

**Theory SOTA (established).**
- Jelassi, Brandfonbrener, Kakade, Malach, *Repeat After Me: Transformers are Better than State Space Models at Copying* (ICML 2024): a 2-layer transformer with $O(\log n)$-width hash-based $n$-gram induction copies length-$n$ strings; any state-space model with $S$ bits of state fails to copy strings longer than $\Theta(S)$. Clean $\Theta(L)$ vs $O(\log L)$ separation (in width, not cache).
- Sanford, Hsu, Telgarsky, *Representational Strengths and Limitations of Transformers* (NeurIPS 2023): one-layer attention solves sparse averaging with $\tilde O(1)$ width; recurrent/fully-connected models need $\Omega(L)$ — via one-way communication complexity of `INDEX`.
- Wen, Dang, Lyu, *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval* (2024): RNNs with $o(L)$ memory cannot do in-context retrieval in one pass; CoT plus explicit retrieval closes the gap.
- Merrill, Petty, Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): SSMs, like transformers, are in uniform $\mathrm{TC}^0$. So the separation is *memory*, not circuit depth — a negative result that rules out a whole family of proof strategies.

**Empirical SOTA.**
- Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models* (ICLR 2024) and *Simple Linear Attention Balances the Recall-Throughput Tradeoff* (Based, ICML 2024): established a recall/state-size Pareto frontier; gated convolutions need state growing with sequence length to match attention on MQAR.
- Hybrids (Jamba, Samba, and Nvidia/Mistral hybrid stacks) reach near-attention recall with a small number of full-attention layers. **Claimed but unablated:** most hybrid papers report aggregate long-context scores without a matched-state control, so the reported win confounds "attention layers help" with "the hybrid has a larger cache".
- **Benchmark-number-only:** needle-in-a-haystack passes and RULER (Hsieh et al., COLM 2024) scores for Mamba-2/RWKV-7 class models. These are single numbers at one context length with no state-size sweep; they cannot identify $\alpha$.

## 4. What Is Known

- **Copying, measured.** Jelassi et al.: at ~160M parameters, trained on strings up to length 300, transformers generalize to roughly $2\times$ training length; Mamba and comparable SSMs degrade sharply past training length, with exact-match falling toward 0 by ~2× at matched parameter count. Pretrained checkpoints (Pythia 2.8B vs Mamba 2.8B) show the same ordering on phone-book lookup.
- **MQAR, measured.** Zoology: at $d_{\text{model}} \in \{64,\dots,512\}$ and $L$ up to 8192, attention reaches ~100% MQAR while gated-convolution models at equal $d_{\text{model}}$ drop to well under 50% as the number of key–value pairs grows; raising state size recovers accuracy, tracing an explicit frontier. Recall gap accounts for a large majority of the associative-recall-slice perplexity gap between attention and gated convolutions at 360M.
- **Circuit class.** No separation available from depth: both families are $\mathrm{TC}^0$ under log precision (Merrill & Sabharwal 2023; Merrill, Petty & Sabharwal 2024).
- **State expressivity within recurrence.** DeltaNet-style delta-rule updates (Yang, Wang, Zhang, Kim, Shen, et al., NeurIPS 2024) beat pure linear attention on recall at equal state, showing $S$ alone does not determine accuracy — the update rule matters.

## 5. What Is Not Known

- **Theoretically open.** Whether a separation survives at *matched* state: a transformer restricted to $S$ bits of cache (any eviction/compression policy) versus the best recurrent model with $S$ bits. No lower bound for constrained-cache attention, and no matching recurrent upper bound. Also open: separation for *learned* models, i.e. a gradient-descent-reachability lower bound rather than a representational one.
- **Empirically open.** The exponent $\alpha_{\text{rec}}(\epsilon)$ for modern gated recurrences (Mamba-2, RWKV-7, DeltaNet, Gated DeltaNet) has never been fit over a $L \times S$ grid at $\ge 1$B parameters. Runnable today; unrun at scale.
- **Methodologically blocked.** "Long-range retrieval ability" has no measurement invariant to prompt format, distractor statistics and length extrapolation. NIAH-style scores move by tens of points under paraphrase of the needle, so they do not isolate memory capacity.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus a missing control arm**. Every published comparison varies at least three things at once: architecture, cache bytes at inference, and training length. The theorems that do exist compare *unbounded* attention cache ($\Theta(Ld)$) against *bounded* recurrent state ($\Theta(d^2)$) — so they prove that $\Theta(L)$ memory beats $\Theta(1)$ memory, which is information theory, not architecture. The interesting claim — attention *uses* its bytes better — has no isolating experiment, because building a transformer with a genuinely fixed byte budget requires committing to an eviction policy, and the policy then becomes the object of study rather than attention itself. Secondary: the worst-case bounds are driven by incompressible uniform keys, which never occur in natural text, so a proved separation need not predict any measured one.

## 7. Current Research (as of 2026)

- Hardware-efficient delta-rule and gated-delta recurrences (MIT/Songlin Yang and collaborators; Nvidia) — pushing recall at fixed state.
- Hybrid attention/SSM stacks with explicit KV budgets (AI21 Jamba line, Microsoft Samba, Nvidia Nemotron-H). *(frontier — verify current ratios.)*
- KV-cache compression (quantization, eviction, latent attention as in DeepSeek MLA) — accidentally supplies the missing matched-state control arm.
- Communication-complexity lower bounds for streaming attention variants (Columbia/Sanford, Stanford/Hazy Research adjacent). *(frontier — verify.)*
- Test-time-training and memory-module recurrences (Titans-style) claiming learned, larger effective state. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Fit $\alpha$ for both families on a matched-state grid.**

- **Scale.** 1.3B-parameter models, ~50B tokens each, identical data and tokenizer. Grid: $L \in \{2048, 8192, 32768, 131072\}$ × state budget $S \in \{0.25, 1, 4, 16\}$ MB per sequence. 16 cells per arm.
- **Arms.** (1) Mamba-2 / Gated DeltaNet with $d_{\text{state}}$ chosen to hit each $S$. (2) **Control: full softmax attention with a hard cache cap at the same $S$ bytes**, using three eviction policies (sliding window, H2O-style heavy-hitter, learned MLA-style latent compression); report the best. (3) Reference ceiling: uncapped attention.
- **Task.** MQAR with $N$ pairs scaled so $N \cdot \log_2 V \in \{0.1, 0.5, 1, 2\} \times S$, plus RULER as an external check.
- **Deciding number.** Fit $S^\star(L) \propto L^{\alpha}$ at $\epsilon = 0.05$ for each arm. **If $\alpha_{\text{attn-capped}} - \alpha_{\text{rec}} \le 0.1$, there is no architectural separation at matched state** and the entire observed advantage is cache size. If $\alpha_{\text{rec}} \approx 1$ while $\alpha_{\text{attn-capped}} \le 0.6$, the separation is real and architectural.

Cost estimate: ~32 pretraining runs at 1.3B/50B ≈ $4\times10^{20}$ FLOPs total, feasible on ~256 H100s for a few weeks.

## 9. Key References

- **[Foundational]** Sanford, Hsu, Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023.
- **[SOTA — theory]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024.
- **[SOTA — theory]** Wen, Dang, Lyu. *RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval.* 2024.
- **[SOTA — empirical]** Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024.
- **[SOTA — empirical]** Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra, Ré. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML 2024.
- **[Negative result]** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024.
- **[Foundational]** Merrill, Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL 2023.
- **[Architecture]** Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024.
- **[Architecture]** Yang, Wang, Zhang, Kim, Shen, et al. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024.
- **[Benchmark]** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024.

## 10. Worked Example

Take Mamba-2 at 2.7B: $n_{\text{layer}}=64$, $d_{\text{model}}=2560$, $d_{\text{state}}=128$. Recurrent state ≈ $64 \times 2560 \times 128 \times 2$ bytes ≈ **42 MB**, constant in $L$.

A 2.7B transformer with $n_{\text{layer}}=32$, $d_{\text{kv}}=2560$: KV cache = $2 \times L \times 2560 \times 32 \times 2$ bytes = $0.33 L$ MB. The two are equal at $L \approx 128{,}000$ tokens.

Now the copy task at $L = 8192$, $V = 32000$ ($\log_2 V \approx 15$ bits/token). Information to reproduce the string: $8192 \times 15 \approx 123$ kbit ≈ **15 kB**. Mamba's 42 MB dwarfs it; the Jelassi bound is nowhere near binding. Yet measured exact-match copying at 8192 collapses for the SSM and holds for attention, whose cache at that length is 2.7 MB — **16× smaller than the SSM's**.

That is the obstruction in one line: **the architecture with less memory wins, so the proved memory lower bound does not explain the observed failure.** The failure is about *addressability* — attention can index an arbitrary past position by content; a fixed linear recurrence must have pre-committed the write. No current theorem separates "bits held" from "bits addressable", and no benchmark measures the latter. The experiment in §8 is designed to make that distinction measurable by holding bits constant and reading off $\alpha$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*