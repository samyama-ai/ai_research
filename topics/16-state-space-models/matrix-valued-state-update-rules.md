---
id: 16-state-space-models/matrix-valued-state-update-rules
title: "Higher-Order and Matrix-Valued State Update Rules"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Higher-Order and Matrix-Valued State Update Rules

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/matrix-valued-state-update-rules` · **Status:** open

## 1. Problem Statement

Modern linear recurrent models carry a **matrix** state $S_t$ and update it with a structured transition. The design space has three axes: the **order** of the state (vector, matrix, tensor, or a small network), the **structure** of the transition operator (scalar, diagonal, identity-plus-rank-one, product of Householders, dense), and the **degree** of the update in the current input (linear vs. bilinear vs. higher).

The problem: **does a richer transition operator buy anything that cannot be bought more cheaply by simply enlarging a diagonal state?**

Three variants, of different difficulty:

- **Theory.** Characterize, per transition family, the class of functions computable in $L$ layers at fixed precision. Partly answered for diagonal and Householder families; open for tensor-valued and learned-optimizer states.
- **Method.** Find the transition family with the best loss per unit of *state bytes* and per unit of *wall-clock*, not per parameter.
- **Measurement.** Define a benchmark on which a state-tracking gain shows up in language-model loss. Currently no such benchmark is agreed on, and this is the binding constraint.

Solving it means: a rule that, at matched state bytes and matched throughput, strictly dominates diagonal gating on both perplexity and state tracking, with the mechanism ablated rather than asserted.

## 2. Formal Setting

Sequence $x_{1:T}$, per head keys/values/queries $k_t\in\mathbb{R}^{d_k}$, $v_t\in\mathbb{R}^{d_v}$, $q_t\in\mathbb{R}^{d_k}$. State $S_t\in\mathbb{R}^{d_v\times d_k}$, output $o_t = S_t q_t$. The general two-sided (bilinear) recurrence:

$$S_t \;=\; A_t\, S_{t-1}\, B_t \;+\; v_t k_t^\top, \qquad A_t\in\mathbb{R}^{d_v\times d_v},\; B_t\in\mathbb{R}^{d_k\times d_k}.$$

Instances, all with $A_t = I$ unless noted:

| Rule | $B_t$ | Params/step |
|---|---|---|
| Linear attention | $I$ | 0 |
| Mamba-2 / GLA | $\alpha_t I$ or $\mathrm{diag}(\alpha_t)$ | $1$ to $d_k$ |
| DeltaNet | $I-\beta_t k_tk_t^\top$, $\|k_t\|=1$ | $1$ |
| Gated DeltaNet | $\alpha_t(I-\beta_tk_tk_t^\top)$ | $2$ |
| DeltaProduct$_{n}$ | $\prod_{i=1}^{n}(I-\beta_t^{(i)}k_t^{(i)}k_t^{(i)\top})$ | $n(d_k{+}1)$ |
| TTT / Titans | $S_t$ = weights of an MLP after a gradient step | — |

**Measured quantities.**

- **State bytes** $M = L\cdot H\cdot d_v d_k \cdot b/8$ ($b$ = bits, $L$ layers, $H$ heads). This, not parameter count, is the thing to hold fixed. Mamba-2 at $d{=}2560$, $H{=}40$, $d_k{=}128$, $d_v{=}64$: $327{,}680$ floats/layer.
- **Training FLOPs** counted on the chunkwise kernel actually run, chunk size $C$, not on the recurrence.
- **State-tracking accuracy**: exact-match on the word problem of $S_5$ (permutation composition) at length $T$, and on parity, with train/test length extrapolation $T_{\text{test}} > T_{\text{train}}$.
- **Loss**: nats/token on a held-out split of the *same* corpus used for training, not a benchmark average.

**Assumptions and where they break.** (i) Fixed-precision arithmetic — the expressivity separations are stated for $O(\log T)$-bit precision; bf16 kernels violate this and the empirical consequences are unmeasured. (ii) The chunkwise form is assumed numerically equal to the recurrence — for DeltaNet-style rules it requires inverting a $C\times C$ triangular matrix, which is where bf16 error concentrates. (iii) Papers compare at matched *parameters*; matched state bytes is the economically correct control and is almost never reported.

## 3. State of the Art

**Theory (established).** Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024): S4/Mamba-class diagonal SSMs with log-precision fall in uniform $\mathrm{TC}^0$, so they cannot solve $S_5$ word problems at any depth unless $\mathrm{TC}^0 = \mathrm{NC}^1$. Sarrof, Veitsman & Hahn (NeurIPS 2024) give the matching positive result: diagonal SSMs capture exactly the star-free regular languages. Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (ICLR 2025): allowing $\beta_t\in[0,2]$ (eigenvalues in $[-1,1]$) makes DeltaNet recognize parity with one layer and, with two layers, richer non-star-free classes. Siems, Grazzi et al., *DeltaProduct* (2025): $n$ Householders per step give a monotone expressivity ladder toward dense transitions.

**Empirical (established).** Chunkwise-parallel training exists for all of the above: DeltaNet (Yang et al., NeurIPS 2024) via the WY/UT representation; Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025).

**Claimed but unablated.** (i) That the delta rule's LM gains come from its *associative-memory* transition rather than from its implicit key normalization and extra per-step scalar — no paper isolates these. (ii) Titans/ATLAS (Behrouz et al., 2024–2025) report that a deep (MLP) memory beats a matrix memory, but at unmatched state bytes and unmatched optimizer steps. (iii) RWKV-7 (Peng et al., 2025) claims regular-language recognition beyond $\mathrm{TC}^0$ under standard assumptions; the language-modeling results are benchmark averages on a bespoke corpus, not matched-token comparisons.

**Benchmark-number-only results.** Every reported "+X points on LM-eval-harness average" for a new transition rule. These aggregate 6–8 zero-shot tasks whose per-task noise at 1.3B scale is comparable to the reported gap.

## 4. What Is Known

- **Scale 2.7B / 300B Pile tokens:** Mamba-2 with state $d_k{=}128$ (8× Mamba's $N{=}16$) matches or exceeds Mamba-2.8B and Pythia-2.8B on standard zero-shot suites while training faster (Dao & Gu, ICML 2024). Enlarging the *diagonal* state is cheap and works.
- **Scale 1.3B / 100B tokens (SlimPajama):** DeltaNet beats Mamba and GLA on LM perplexity and on recall-intensive tasks (MQAR-style), with gains largest where recall is stressed (Yang et al., NeurIPS 2024).
- **Scale 1.3B / 100B tokens (FineWeb-Edu):** Gated DeltaNet reports lower perplexity than Mamba2 and DeltaNet, i.e. gating and the delta rule are complementary rather than redundant (Yang et al., ICLR 2025).
- **Synthetic, <10M params:** the parity / $S_5$ separations are clean and reproduced. Diagonal-gated models sit at chance on $S_5$ regardless of width; negative-eigenvalue DeltaNet and DeltaProduct$_{n\ge2}$ reach high accuracy with length extrapolation.
- **Cost:** the delta-rule chunkwise kernel does strictly more matmul work per chunk than a diagonal one (see §10), and the gap is a small constant, not asymptotic.

## 5. What Is Not Known

- **Theoretically open.** The expressivity class of tensor-valued ($p\ge3$) states and of TTT/Titans-style states whose transition is a gradient step on a nonlinear inner loss. No upper bound in $\mathrm{TC}^0$ and no separation from it has been proved. Also open: whether the DeltaProduct ladder collapses at some finite $n < d_k$.
- **Empirically open.** Whether any of the state-tracking-capable rules improves natural-language loss *at matched state bytes and matched wall-clock*. The experiment is runnable at 1.3B/100B tokens today; nobody has published the matched-state-bytes control. This is the central gap.
- **Methodologically blocked.** There is no natural-language evaluation known to depend on non-star-free state tracking. Until one exists, "does higher-order state help real modeling?" has no measurement. Long-context retrieval benchmarks measure recall capacity — a function of state bytes — not transition structure.

## 6. Why It Is Hard

**Confounded measurement, in a specific way: the two axes trade off against each other under a fixed compute budget.** A richer $B_t$ costs FLOPs per token; at fixed wall-clock, buying it means shrinking $d_k$ or $L$. Since LM loss at these scales is strongly monotone in state bytes and only weakly (perhaps not at all) sensitive to transition rank, the published comparisons — matched on parameters, unmatched on state bytes and throughput — cannot distinguish "the delta rule helps" from "this configuration happened to carry more state".

Second obstruction: **the evaluation does not measure what it names.** The theory separations are about non-star-free regular languages; the empirical claims are made on zero-shot benchmark averages. Nothing connects them. A model can gain 15 points on $S_5$ and 0.00 nats on FineWeb-Edu, and both papers can be correct.

## 7. Current Research (as of 2026)

- **Householder-product ladders.** Freiburg/Hutter's group and collaborators (Grazzi, Siems, Zela, Pontil) pushing DeltaProduct to larger $n$ and to LM scale.
- **Test-time-training states.** Sun, Guestrin, Hashimoto et al. (TTT layers); Behrouz, Mirrokni et al. (Titans, ATLAS); von Oswald et al. (MesaNet, locally optimal test-time regression). The unifying frame — the recurrence *is* an online learner and the transition is its optimizer — is now standard. Whether second-order inner optimizers pay off at scale is unresolved *(frontier — verify)*.
- **Kernel engineering.** FLA-library chunkwise kernels for generalized-Householder transitions; the practical question is whether the UT-transform inverse can stay in bf16 at $C{=}64$.
- **Hybrids.** Most production-scale releases interleave a few full-attention layers with linear-recurrent ones, which quietly removes the state-tracking motivation for the recurrent layers.

## 8. Concrete Next Experiment

**Matched-state-bytes ladder.**

- **Scale.** 340M and 1.3B params, 100B tokens of FineWeb-Edu, identical data order, tokenizer, LR schedule, and 3 seeds each. Six arms per scale.
- **Control arm.** Mamba-2/GLA (diagonal $B_t$) with $d_k$ chosen so that **state bytes are identical** to each treatment arm, and with training FLOPs equalized by adjusting token count downward for the cheaper arm — report both the equal-token and equal-FLOP points.
- **Treatments.** DeltaNet ($\beta\in[0,1]$), DeltaNet ($\beta\in[0,2]$), Gated DeltaNet, DeltaProduct$_2$, DeltaProduct$_4$.
- **Deciding number.** Held-out FineWeb-Edu loss gap, in nats/token, between the best non-diagonal arm and the state-byte-matched diagonal control at equal FLOPs. **If $|\Delta| < 0.01$ nats with seed std $<0.005$, the transition structure buys nothing for language at 1.3B and the field should stop reporting parameter-matched wins.** If $\Delta < -0.02$ nats, it buys something real and the next question is what.
- **Secondary, same runs.** $S_5$ accuracy at $T_{\text{test}}{=}512$ after fine-tuning on $T_{\text{train}}{=}128$. Reporting both numbers on the *same checkpoints* is what closes the theory–practice gap; no published run does this.

## 9. Key References

- **[Foundational]** Gu, Goel & Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Sutskever, Martens & Hinton. *Generating Text with Recurrent Neural Networks.* ICML 2011. (multiplicative/second-order RNN)
- **[Foundational]** Barrington. *Bounded-width polynomial-size branching programs recognize exactly those languages in $NC^1$.* STOC 1986 / JCSS 1989.
- **[SOTA]** Dao & Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Yang, Wang, Zhang & Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. — arXiv:2406.06484
- **[SOTA]** Yang, Kautz & Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. — arXiv:2412.06464
- **[Theory]** Merrill, Petty & Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Theory]** Sarrof, Veitsman & Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective.* NeurIPS 2024.
- **[Theory]** Grazzi, Siems, Franke, Zela, Hutter & Pontil. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025. — arXiv:2411.12537
- **[Frontier]** Siems, Grazzi et al. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.* 2025. — arXiv:2502.10297
- **[Related]** Sun, Li, Dalal, Xu, Vikram, Zhang, Guestrin, Wang, Hashimoto et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* 2024. — arXiv:2407.04620
- **[Related]** Behrouz, Zhong & Mirrokni. *Titans: Learning to Memorize at Test Time.* 2024. — arXiv:2501.00663
- **[Survey]** Yang, Wang, Shen, Panda & Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. — arXiv:2312.06635

## 10. Worked Example

One head, $d_k = d_v = 128$, chunk $C = 64$, one chunk of tokens.

**Diagonal (Mamba-2) chunk cost**, in multiply-accumulates:
- intra-chunk $QK^\top$ and $(\cdot)V$: $2C^2d = 2\cdot 64^2\cdot 128 = 1.05\text{M}$
- state read/write $QS$ and $K^\top V$: $2Cd^2 = 2\cdot 64\cdot 128^2 = 2.10\text{M}$
- **total $\approx 3.15$M MAC**

**DeltaNet chunk cost** adds the UT transform $T = (I + \mathrm{tril}(\mathrm{diag}(\beta)KK^\top))^{-1}$:
- $KK^\top$: $C^2 d = 0.52\text{M}$
- triangular inverse by forward substitution: $C^3/6 \approx 0.044\text{M}$
- two applications of $T$ (to $V$ and to $K$-side terms): $2C^2 d = 1.05\text{M}$
- **total $\approx 4.76$M MAC, i.e. $1.51\times$ the diagonal rule.**

**The obstruction, made concrete.** Hold wall-clock fixed. A $1.51\times$ cost per token means the DeltaNet arm must shrink somewhere; the cheapest place is $d_k$. Dropping $d_k$ from 128 to 96 recovers roughly the budget ($0.75^2$ on the $d^2$ terms) — and cuts state bytes per head from $128\cdot128 = 16{,}384$ to $128\cdot96 = 12{,}288$, a **25% loss of memory capacity**. Recall-heavy loss is empirically sensitive to state bytes at this scale; state-tracking loss is not measured by the LM objective at all.

So the published parameter-matched comparison reports a $\sim0.02$-nat DeltaNet win at *equal* $d_k$ — which is a win bought partly with 51% more compute. The compute-matched comparison, which is the one that decides whether higher-order transitions are worth building kernels for, has not been published at 1.3B. That single missing number is the whole problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*