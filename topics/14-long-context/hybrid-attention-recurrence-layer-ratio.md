---
id: 14-long-context/hybrid-attention-recurrence-layer-ratio
title: "Hybrid Attention-Recurrence Layer Ratio"
topic: 14-long-context
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hybrid Attention-Recurrence Layer Ratio

> **Topic:** Long Context · **ID:** `14-long-context/hybrid-attention-recurrence-layer-ratio` · **Status:** empirically-open

## 1. Problem Statement

Hybrid language models interleave two kinds of sequence-mixing layer: **softmax attention**, whose KV cache grows linearly in context length $T$ and which supports exact retrieval, and **fixed-state recurrence** (Mamba-2, gated DeltaNet, RWKV, linear attention, or sliding-window attention), whose state is $O(1)$ in $T$. The design knob is the **ratio** $\rho$ — what fraction of sequence-mixing layers are full attention — plus the **placement** of those layers in depth.

The problem: given a parameter budget $N$, token budget $D$, and a target context length $T$, predict the ratio and placement that minimise loss subject to a decode-time memory constraint. Three variants, of very different difficulty:

- **Measurement.** Does a benchmark exist whose score is monotone in the capability that attention layers uniquely provide (exact long-range retrieval), and that is not saturated? Needle-in-a-haystack is saturated; RULER and BABILong are not, but confound retrieval with reasoning and instruction-following.
- **Method.** Find $\rho^\star(N, D, T)$ empirically — a scaling law in three variables where each point costs a pretraining run.
- **Theory.** Prove a lower bound on the number of full-attention layers needed for a task class (e.g. $k$-hop retrieval over a $T$-token context) given per-layer recurrent state of $s$ bits.

A solution to the method variant is a fitted rule $\rho^\star(N,D,T)$ that predicts held-out hybrid runs' loss and retrieval accuracy better than the current practice of copying a published ratio.

## 2. Formal Setting

A model has $L$ sequence-mixing layers, indexed $\ell = 1..L$, each typed $\tau_\ell \in \{\mathrm{attn}, \mathrm{rec}\}$. Define

$$\rho = \frac{1}{L}\sum_{\ell=1}^{L} \mathbb{1}[\tau_\ell = \mathrm{attn}], \qquad \pi = (\tau_1, \dots, \tau_L).$$

Two models with equal $\rho$ can differ in $\pi$; the literature almost always reports $\rho$ and holds $\pi$ to a periodic pattern, so placement effects are folded into $\rho$'s error bars.

**Decode memory**, measured as peak bytes of per-sequence cache at generation step $T$, with $b$ bytes per element, $H$ KV heads, head dim $d_h$, recurrent state dim $s$ per layer:

$$M(T) = b\left[\underbrace{2 H d_h T \cdot \rho L}_{\text{KV cache}} + \underbrace{s \cdot (1-\rho) L}_{\text{recurrent state}}\right].$$

Measure $M$ by reading the allocator's peak reserved bytes during a fixed-length generation, not by formula — kernel workspaces and page-granular KV allocators add 5–20%.

**Objective.** For validation loss $\mathcal{L}$ and retrieval score $R$ at length $T$:

$$\rho^\star(N,D,T) = \arg\min_{\rho} \ \mathcal{L}(N,D,\rho) \quad \text{s.t.} \quad M(T) \le M_{\max}, \ R(T,\rho) \ge R_{\min}.$$

$\mathcal{L}$ is measured on a held-out corpus at sequence length $T$ with the same tokenizer across arms. $R$ is measured as exact-match on a synthetic multi-key retrieval probe with a controlled number of distractors, reported per depth (fraction of context) rather than as a single average.

**Assumptions, and which are violated.**
1. *Parameter-matched arms.* Attention and recurrent blocks have different parameter counts per layer (Mamba-2 blocks are typically wider), so matching $N$ forces unequal $L$. Violated in most published comparisons, which match $L$ or match wall-clock instead.
2. *$\rho$ separable from $\pi$.* Assumed; unverified — the one systematic placement sweep in the open literature is small-scale.
3. *Ratio transfers across scale.* Assumed by every deployed model. No published scaling law tests it; ratios chosen at 1–3B are reused at 80B+.
4. *Training-length equals eval-length.* Violated: hybrids are trained at 4k–32k and evaluated at 128k–1M, so measured $R(T)$ conflates ratio with length extrapolation of the attention layers' positional scheme.

## 3. State of the Art

**Established (ablated within a paper, at one scale).**
- **Jamba** (Lieber et al., AI21, 2024): 52 blocks, 1 attention layer per 8 (attention $\approx 12.5\%$ of sequence-mixing layers), MoE, 256K context. The paper ablates 1:3 vs 1:7 attention-to-Mamba at ~1.3B scale and reports the ratios are close in loss, with pure Mamba failing an in-context-learning format-following behaviour that a single attention layer restores.
- **Mamba-2-Hybrid 8B** (Waleffe et al., NVIDIA, 2024): 24 Mamba-2, 4 attention, 24 MLP layers — 4/28 $\approx 14\%$ of sequence-mixing layers. Trained on 3.5T tokens, the hybrid exceeds a matched 8B Transformer on average across 12 standard tasks and beats pure Mamba-2, which collapses on tasks requiring in-context copying. This is the cleanest large-scale controlled comparison in the open literature, but it compares *hybrid vs pure*, not $\rho$ against $\rho'$ at 8B.
- **Samba** (Ren et al., Microsoft, 2024): 1:1 Mamba / sliding-window attention (window 2048), 3.8B on 3.2T tokens; extrapolates from 4K training to 256K with retained perplexity.
- **Griffin / Hawk** (De et al., DeepMind, 2024): 2 recurrent : 1 local-attention block, up to 14B, matching Llama-2-scale performance on held-out loss with fewer training tokens.

**Claimed but unablated.** Production ratios — MiniMax-01's 7:1 lightning-to-softmax, Qwen3-Next's 3:1 gated-DeltaNet-to-attention, Granite 4.0's roughly 9:1 Mamba-2 hybrid *(frontier — verify)* — are reported as design choices with benchmark tables, not as sweeps. No public artifact shows the loss curve as a function of $\rho$ at those scales.

**Benchmark-number-only.** Nearly all long-context claims for hybrids are RULER or needle scores at a single $\rho$. RULER (Hsieh et al., 2024) shows that most models with claimed 128K+ support fall below their short-context baseline well before their advertised length; it is not designed to attribute that failure to $\rho$.

## 4. What Is Known

- **A little attention is enough for most benchmarks, at 8B / 3.5T tokens.** 4 of 28 sequence-mixing layers ($\approx 14\%$) suffices for the NVIDIA hybrid to match or beat a Transformer on 12 standard tasks. At 12.5% (Jamba) the model is competitive at 7B-active/52B-total.
- **Zero attention is not enough.** Pure Mamba/Mamba-2 degrades sharply on tasks requiring verbatim copying or few-shot format induction; Waleffe et al. isolate this, and Jelassi et al. (ICML 2024) show it holds across model families.
- **The failure has an information-theoretic floor.** A recurrent model with $s$ bits of state cannot copy an input string carrying more than $s$ bits (Jelassi et al., 2024) — a counting argument, not an empirical claim. Two-layer transformers copy strings of length exponential in width.
- **Recall trades against state size, not against depth alone.** Arora et al. (ICML 2024) fit an explicit recall-throughput frontier: associative-recall accuracy for linear-attention variants scales with recurrent state size, and their Based architecture moves the frontier by combining short sliding-window attention with large-state linear attention at 1.3B scale.
- **Sliding window counts as recurrence for memory but not for retrieval.** Gemma 3 (2025) uses 5 local : 1 global attention with a 1024-token window and reports large KV-cache savings; global layers remain load-bearing for long-range retrieval.
- **Distillation transfers the ratio question.** "The Mamba in the Llama" (Wang et al., NeurIPS 2024) converts a Llama into a hybrid keeping roughly 25–50% of attention layers, showing that a trained model's attention layers are partly redundant — but at a ratio chosen by fiat.

## 5. What Is Not Known

- **Empirically open.** Does $\rho^\star$ depend on $N$ and $D$? Every ingredient — pretraining code, evaluation harnesses, compute — exists. Nobody has published a $\rho$-sweep at two or more scales with parameter-matched arms. This is the central gap.
- **Empirically open.** Placement: is $\pi$ = "attention early", "attention late", or "attention uniform" better at fixed $\rho$? Small-scale results are contradictory and no run above ~1.5B is public.
- **Methodologically blocked.** There is no accepted retrieval metric that is (i) unsaturated, (ii) monotone in attention capacity, and (iii) independent of instruction-following ability. RULER and BABILong mix all three, so a drop when $\rho$ falls cannot be attributed.
- **Theoretically open.** No lower bound of the form "$k$-hop retrieval over $T$ tokens requires $\ge f(k,T,s)$ full-attention layers." The copying bound (Jelassi et al.) covers $\rho = 0$ only; it says nothing about how quickly capability is restored as $\rho$ grows from $0$ to $1/L$.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by per-point cost**. One $\rho$ point at 8B / 1T tokens is roughly $6ND \approx 5\times10^{22}$ FLOPs, or order $10^4$ H100-hours; a 5-point sweep at two scales is $10^5$ H100-hours before evaluation. That alone would be tolerable if the measurement were clean. It is not: changing $\rho$ at fixed $N$ changes $L$, per-layer width, the positional scheme's effective load, and the optimal learning rate simultaneously. A loss difference of 0.01 nats between $\rho = 1/8$ and $\rho = 1/4$ is within the seed-and-LR noise band of most published training setups, so the sweep must be repeated across seeds — multiplying the cost by 3–5 — or the result is not distinguishable from noise. Secondarily, ground truth is absent: there is no task set known to be *exactly* the set of things attention layers do and recurrence does not.

## 7. Current Research (as of 2026)

- **Production hybrids at scale.** AI21 (Jamba family), NVIDIA (Nemotron-H), IBM (Granite 4.0), Alibaba (Qwen3-Next), MiniMax, and TII (Falcon-H1) all ship hybrids; ratios cluster in $\rho \in [1/12, 1/4]$ *(frontier — verify individual figures)*. Convergence of ratios across independent labs is suggestive but is not evidence — labs copy each other's ratios.
- **Attention-free-with-large-state.** Gated DeltaNet (Yang et al., 2025) and successors push $s$ up so that fewer attention layers are needed; the open question becomes whether $\rho^\star \to 0$ as $s$ grows.
- **Post-hoc hybridisation.** Distilling attention layers into recurrence (Wang et al., 2024; MOHAWK, Bick et al., 2024) offers a cheap proxy sweep: measure how many attention layers can be removed from a trained model before capability breaks. Proxy validity is unestablished.
- **Measurement work.** RULER, BABILong, and HELMET (Yen et al., 2025) are converging on decomposed long-context suites; none yet isolates the attention-specific axis.

## 8. Concrete Next Experiment

**Scale.** Two model sizes, 1.4B and 8B active parameters, each trained on 300B tokens at sequence length 8192 — roughly $2.5\times10^{21}$ and $1.4\times10^{22}$ FLOPs per run.

**Arms.** $\rho \in \{0, 1/16, 1/8, 1/4, 1/2, 1\}$ at each scale, with $L$ adjusted so total non-embedding parameters match within 1%. Recurrent layers: Mamba-2 with fixed state dim. Attention layers: full, RoPE, identical head config across arms. Placement fixed to a uniform periodic $\pi$. Two seeds per point at 1.4B, one at 8B (24 runs total, ~$10^5$ H100-hours).

**Control arm.** $\rho = 1$ (pure Transformer) at both scales, same data order and LR schedule. This anchors the loss axis and gives the seed-noise band.

**The deciding number.** Fit $\hat\rho^\star$ at each scale as the smallest $\rho$ whose validation loss is within the two-seed noise band of the $\rho = 1$ control, evaluated at 8192 tokens. The question is decided by the ratio

$$\Delta = \hat\rho^\star(8\mathrm{B}) / \hat\rho^\star(1.4\mathrm{B}).$$

If $\Delta \in [0.8, 1.25]$, ratio transfer across scale is supported and the field's copy-the-published-ratio practice is justified. If $\Delta < 0.8$ (larger models need proportionally less attention) or $> 1.25$, every published ratio chosen at small scale is mis-set at deployment scale, and $\rho$ must enter the scaling law. Report alongside: multi-key retrieval exact-match at 4 depths and measured peak decode bytes $M(131072)$ per arm.

## 9. Key References

- **[Foundational]** Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Dao, T., Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[SOTA]** Lieber, O., Lenz, B., Bata, H., et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* 2024. — arXiv:2403.19887
- **[SOTA]** Waleffe, R., Byeon, W., Riach, D., et al. *An Empirical Study of Mamba-based Language Models.* 2024. — arXiv:2406.07887
- **[SOTA]** Ren, L., Liu, Y., Lu, Y., et al. *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* ICLR, 2025. — arXiv:2406.07522
- **[SOTA]** De, S., Smith, S. L., Fernando, A., et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427
- **[Theory]** Jelassi, S., Brandfonbrener, D., Kakade, S., Malach, E. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Theory/Empirical]** Arora, S., Eyuboglu, S., Zhang, M., et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[Method]** Wang, J., Paliotta, D., May, A., Rush, A. M., Dao, T. *The Mamba in the Llama: Distilling and Accelerating Hybrid Models.* NeurIPS, 2024. — arXiv:2408.15237
- **[Evaluation]** Hsieh, C.-P., Sun, S., Kriman, S., et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Evaluation]** Yen, H., Gao, T., Hou, M., et al. *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.* ICLR, 2025. — arXiv:2410.02694

## 10. Worked Example

Take an 8B hybrid with $L = 32$ sequence-mixing layers, $H = 8$ KV heads, $d_h = 128$, fp16 ($b = 2$), Mamba-2 state $\approx 128\text{K}$ elements per layer, decoding at $T = 131072$.

Per attention layer, KV cache is $2 \times 8 \times 128 \times 131072 \times 2 = 537$ MB. Per recurrent layer, state is $\approx 0.26$ MB — a factor of ~2000.

| $\rho$ | attn layers | cache (GB) | copy/retrieval expectation |
|---|---|---|---|
| $1$ | 32 | 17.2 | full |
| $1/4$ | 8 | 4.3 | full |
| $1/8$ | 4 | 2.15 | full (Jamba, NVIDIA hybrid regime) |
| $1/16$ | 2 | 1.07 | unknown |
| $0$ | 0 | 0.008 | fails copying (Jelassi et al.) |

The memory axis is settled: going $\rho = 1 \to 1/8$ buys an 8× cache reduction, which is why every lab ships a hybrid. The obstruction appears in the last column. Between $\rho = 1/8$ and $\rho = 0$ lies a factor-of-270 further memory saving, and the only two data points bracketing it are "1/8 works at 8B" and "0 fails at 8B." The single intermediate point, $\rho = 1/16$ (2 attention layers), is unmeasured at any scale above ~1.5B.

Now make the measurement problem visible. Suppose a 2-layer-attention arm scores 71% on RULER-128K against 78% for the 4-layer arm. The 7-point gap is not attributable: RULER's aggregate mixes needle retrieval, variable tracking, and aggregation subtasks, and the arms differ in $L$ (parameter-matched), so the recurrent layers also changed count. A 7-point drop is also within the range that a 2× learning-rate misspecification produces in an 8B run. Deciding whether $\rho = 1/16$ is viable therefore needs seed replication and a decomposed metric — which is why the answer, despite costing under $10^5$ H100-hours to obtain, is still not in the literature.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*