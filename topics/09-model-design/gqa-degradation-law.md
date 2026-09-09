---
id: 09-model-design/gqa-degradation-law
title: "Grouped-Query Attention Quality Degradation Law"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Grouped-Query Attention Quality Degradation Law

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/gqa-degradation-law` · **Status:** empirically-open

## 1. Problem Statement

Grouped-query attention (GQA) replaces the $H$ independent key/value projections of multi-head attention (MHA) with $G \le H$ shared KV heads, cutting the KV cache by $H/G$. Every production decoder since 2023 uses it, and essentially all of them pick $G = 8$. Nobody has published the function that $G$ is being chosen against.

The problem: **give the loss penalty of KV-head sharing as a function of the group ratio, the parameter count, the token budget, and the context length.** Find $\Delta(G)$ such that

$$L(N, D, S, G) \;=\; L_\infty(N, D) \;+\; \Delta(N, D, S, G),$$

and characterise its shape — in particular whether the compute-optimal $G$ grows, shrinks, or stays fixed as $N$ and $S$ grow.

Three variants, of different difficulty:

- **Measurement.** Is the GQA penalty even resolvable above seed noise at fixed $(N,D)$? At 7B scale the reported gaps are a few tenths of a benchmark point on single runs. Isolating a signal that small needs multi-seed replication that nobody has paid for.
- **Method.** Given a KV-cache byte budget, choose $(G, d_h, n_{\text{layer}})$ jointly to minimise loss. Current practice fixes $G=8$ and varies nothing else.
- **Theory.** Prove a separation: exhibit a task family where any $G$-group model needs $\Omega(f(H/G))$ more parameters or layers than MHA to reach the same loss. Open.

Solved means: a fitted law with held-out predictive error smaller than the design decisions it is used to make (say, $<0.005$ nats on next-token loss), validated across at least one order of magnitude in $N$ and $S$.

## 2. Formal Setting

Decoder layer, $H$ query heads of dimension $d_h$, model width $d = H d_h$, $G$ KV heads, sharing ratio $r = H/G$. Query head $i$ attends with KV head $\lceil i G / H \rceil$:

$$\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_{g(i)}^\top}{\sqrt{d_h}}\right) V_{g(i)}, \qquad g(i) = \lceil iG/H \rceil .$$

$G = H$ is MHA; $G = 1$ is multi-query attention (MQA).

**Quantities as measured.**

- $N$: non-embedding parameters, counted directly. Note GQA *reduces* $N$ by $2(H-G)d\,d_h$ per layer — any comparison at "the same model size" is either not matched on $N$ or not matched on width. State which.
- $D$: training tokens seen, not unique tokens.
- $L$: mean next-token cross-entropy in nats on a held-out corpus disjoint from training, reported per position bucket $[2^k, 2^{k+1})$ so long-context effects are visible.
- $S$: evaluation context length in tokens.
- KV cache bytes at batch $b$, precision $p$ bytes: $M = 2\,n_{\text{layer}}\,G\,d_h\,S\,b\,p$. This, not $N$, is the resource GQA buys.
- $\Delta(G)$: $L$ of the $G$-group run minus $L$ of the $G=H$ control, both trained from scratch on identical data order, with the seed-to-seed standard deviation of $L$ reported.

**Assumptions, and which are violated.**

1. *Uniform grouping is optimal.* Violated — QCQA and key-driven grouping report that non-uniform, similarity-based assignments beat contiguous grouping at the same $G$.
2. *$\Delta$ is separable from $L_\infty$.* Assumed, untested. If sharing interacts with data quality or repetition, $\Delta$ is not additive.
3. *Uptrained GQA $\approx$ from-scratch GQA.* Violated in an unknown direction. The original GQA result is a conversion of an MHA checkpoint with $\sim$5% extra pretraining compute; from-scratch models may have a different penalty.
4. *Benchmark averages track $L$.* Violated routinely — a 0.3-point MMLU shift can correspond to no measurable loss change.
5. *Attention head count is the binding constraint.* Violated when RoPE, sliding windows, or long-context extension change what heads are doing.

## 3. State of the Art

**Empirical SOTA (established).** Ainslie et al. (EMNLP 2023) introduced GQA, converting T5-XXL (11B) by mean-pooling KV projections and uptraining. GQA-8 recovers nearly all of MHA quality at close to MQA decode speed; MQA loses noticeably more. Established: the *ordering* MHA $\ge$ GQA $>$ MQA on that checkpoint.

**Claimed but unablated.** The specific magnitudes — a fraction of a point on summarization/QA averages — are single-run benchmark numbers with no seeds, no from-scratch control, and no scale sweep. $G=8$ was not shown optimal; it was shown adequate for one 11B encoder-decoder. Its propagation into Llama 2 70B, the entire Llama 3 family, Mistral 7B, Qwen, and Gemma is transfer by convention, not by re-derivation.

**Competing point on the frontier.** DeepSeek-V2/V3 multi-head latent attention (MLA) compresses KV into a low-rank latent instead of sharing heads, and reports both a smaller cache than GQA-8 *and* better loss. If reproduced independently at matched $N$ and $D$, that is evidence GQA sits off the Pareto frontier — but the comparison is internal to one lab's training stack.

**Theory SOTA.** No degradation law. The closest results are structural: Bhojanapalli et al. (ICML 2020) prove a low-rank bottleneck when $d_h < S$, tying head dimension to representable attention matrices; Michel et al. (NeurIPS 2019) show most heads can be pruned post hoc with small loss. Meng et al. (2025, TransMLA) argue MLA is strictly more expressive than GQA at equal cache and give a construction converting GQA to MLA — an expressivity statement, not a loss bound.

## 4. What Is Known

- **The cache saving is exact and large.** Llama 2 70B: $H=64$, $G=8$, so an 8× KV reduction. At 4k context, batch 1, fp16, this moves the cache from $\sim$2.5 GB to $\sim$320 MB. Measured, not estimated.
- **Decode is memory-bandwidth bound**, so cache reduction translates near-linearly into throughput at long context (Pope et al., MLSys 2023).
- **$G=8$ is near-universal at 7B–70B.** Mistral 7B: $H=32, G=8$. Llama 3 8B and 70B: $G=8$. Llama 2 used MHA at 7B/13B and GQA-8 only at 34B/70B — the one public hint that the tradeoff is scale-dependent, and it was never ablated.
- **MQA ($G=1$) is measurably worse** than GQA-8 on the T5-XXL uptraining setup and shows training instability in several reports; PaLM-540B used MQA and later families did not.
- **Architecture rankings do not transfer across scale.** Tay et al. (2023) show inductive-bias changes alter scaling *exponents*, so a $G$ chosen at 1B need not hold at 100B. This is the strongest available reason to distrust the constant $G=8$.
- **Grouping assignment matters at fixed $G$.** QCQA (2024) and key-similarity grouping report gains over uniform contiguous groups on 7B-class models — small, single-seed.

## 5. What Is Not Known

- **Empirically open.** The primary gap. The full grid — from-scratch training at matched $N$ and $D$ across $G \in \{1,2,4,8,16,H\}$ and at least three model scales, multi-seed, with loss reported per position bucket — is entirely runnable on a few hundred GPU-days. Nobody has published it. Whether optimal $G$ scales with $H$, with $\sqrt{H}$, or is constant is unresolved.
- **Empirically open.** Whether the penalty grows with context length $S$. Every intuition says shared KV heads should hurt most where retrieval is hardest, i.e. at long range; there is no position-resolved measurement.
- **Methodologically blocked.** "Quality" in the deployed comparisons is a benchmark average. At the effect sizes involved (tenths of a point) that instrument is below its own noise floor. Until the field reports $\Delta L$ with seed variance, the measurement is not defined well enough to fit a law to.
- **Theoretically open.** No lower bound separating $G$-group attention from MHA on any natural task family. No proof that the penalty is monotone in $r = H/G$, though it is universally assumed.

## 6. Why It Is Hard

**Confounded measurement, then compute cost — in that order.**

The confound: GQA changes parameter count, width-per-KV-head, and cache size simultaneously. Hold $N$ fixed and you must add width or layers elsewhere, which changes the thing you are measuring. Hold width fixed and the GQA model is smaller, so part of any observed penalty is just fewer parameters. Published comparisons pick one convention silently, so their numbers are not commensurable across papers. This is non-identifiability, not sloppiness: with one loss number and three changed variables, $\Delta(G)$ is not recoverable from existing runs.

The cost: the effect is small. Resolving a 0.003-nat difference needs seed replication, and a 3-scale × 5-$G$ × 3-seed grid is 45 pretraining runs. At 1B parameters and Chinchilla-optimal 20B tokens that is affordable; at the scale where the answer actually matters (70B+), it is not — and the scale-transfer result in §4 says the cheap version may not answer the expensive question.

Third: the evaluation does not measure what it names. Benchmarks aggregate over positions, so a penalty concentrated at long range is averaged away by short-context examples.

## 7. Current Research (as of 2026)

- **Latent/compressed KV over head sharing.** DeepSeek's MLA line and follow-ups (TransMLA, Meng et al. 2025) argue the whole GQA design axis is dominated. Active; the conversion direction (GQA checkpoint → MLA) is the practical hook. *(frontier — verify)*
- **Learned grouping.** QCQA, key-driven grouping, and merge-based post-training methods that pick which heads share rather than assuming contiguity. Mostly small-scale academic work.
- **Cross-layer KV sharing.** MLKV and YOCO share KV across layers instead of within a layer, a different point on the same budget curve; a joint law over (within-layer, across-layer) sharing does not exist.
- **Hybrid attention stacks.** Mixing full-attention layers with linear/sliding-window layers changes what the KV budget is spent on and makes a single scalar $G$ the wrong parameterisation. *(frontier — verify)*
- **Architecture scaling laws generally.** Groups fitting laws with architecture as an input rather than a fixed choice; GQA is a clean test case because it has exactly one integer knob.

## 8. Concrete Next Experiment

**Scale.** Three model sizes: 160M, 600M, 1.4B non-embedding parameters, $H$ = 12/16/24, each trained from scratch on 20 tokens/parameter of an open corpus (e.g. FineWeb-Edu), fixed data order, 4k context. Sweep $G \in \{1, 2, 4, 8, H\}$ subject to $G \mid H$. Three seeds for the 160M and 600M grids, one seed at 1.4B. Roughly 60 runs, $\sim$400 A100-days.

**Control arm.** Two controls, both required. (a) $G = H$ (MHA) at identical width — isolates sharing, unmatched on $N$. (b) $G$-group model with $n_{\text{layer}}$ or $d_{\text{ffn}}$ increased to restore $N$ to the MHA value — isolates sharing at matched parameters. Reporting only (a) is what makes existing numbers uninterpretable.

**Deciding number.** The fitted exponent $\alpha$ in

$$\Delta(N, r) = c \, N^{-\beta} r^{\alpha}, \qquad r = H/G,$$

estimated on the 160M/600M grids and tested against the held-out 1.4B runs. Decision rule: if $\beta > 0$ with the 1.4B point inside the 95% predictive interval, the penalty shrinks with scale and $G=8$ is conservative — larger $r$ is free at frontier scale. If $\beta \le 0$, the penalty persists or grows, and every 70B+ model shipped with $G=8$ is paying an unmeasured tax. Secondary readout: $\Delta$ restricted to positions in $[2048, 4096)$ versus $[0, 512)$ — a ratio above 1.5 establishes the long-context claim in §5 that currently has no evidence either way.

## 9. Key References

- **[Foundational]** Noam Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- **[Foundational / SOTA]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[SOTA]** DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024. — arXiv:2405.04434
- **[SOTA]** Reid Pope, Sholto Douglas, Aakanksha Chowdhery, et al. *Efficiently Scaling Transformer Inference.* MLSys 2023. — arXiv:2211.05102
- **[Theory]** Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank Reddi, Sanjiv Kumar. *Low-Rank Bottleneck in Multi-head Attention Models.* ICML 2020. — arXiv:2002.07028
- **[Theory]** Paul Michel, Omer Levy, Graham Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019. — arXiv:1905.10650
- **[Scaling]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Scaling]** Yi Tay, Mostafa Dehghani, Samira Abnar, et al. *Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?* Findings of EMNLP 2023. — arXiv:2207.10551
- **[Systems]** Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR 2023. — arXiv:2204.02311
- **[Systems]** Llama Team, AI @ Meta. *The Llama 3 Herd of Models.* 2024. — arXiv:2407.21783
- **[Frontier]** Fanxu Meng, Zengwei Yao, Muhan Zhang. *TransMLA: Multi-Head Latent Attention Is All You Need.* 2025. — arXiv:2502.07864
- **[Frontier]** Vinay Joshi, Prashant Laddha, Shambhavi Sinha, Om Ji Omer, Sreenivas Subramoney. *QCQA: Quality and Capacity-aware grouped Query Attention.* 2024.
- **[Frontier]** Zayd M. K. Zuhri, Erland Hilman Fuadi, Alham Fikri Aji, et al. *MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding.* 2024.

## 10. Worked Example

Take Llama 3 8B: $n_{\text{layer}} = 32$, $d = 4096$, $H = 32$, $d_h = 128$, $G = 8$.

KV cache at $S = 8192$, $b = 1$, fp16 ($p = 2$):

$$M = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 8192 \cdot 1 \cdot 2 \;=\; 1.07 \times 10^{9}\ \text{bytes} \approx 1.0\ \text{GiB}.$$

The MHA counterfactual ($G = 32$) is 4.0 GiB. On an 80 GB accelerator holding 16 GB of weights, the servable batch at 8k context goes from about 16 to about 64 — a 4× throughput swing in the memory-bound decode regime.

Now the parameter side. Dropping from $G=32$ to $G=8$ removes $2(H-G) d\, d_h = 2 \cdot 24 \cdot 4096 \cdot 128 \approx 25.2$M parameters per layer, or **806M over 32 layers** — 10% of the model. So the GQA-8 model is not "Llama 3 8B minus a little cache"; it is a 10% smaller model.

Here is the obstruction, in numbers. Chinchilla-style fits put the loss sensitivity near $L \propto N^{-0.34}$, so a 10% parameter cut alone costs roughly $1 - 0.9^{0.34} \approx 3.6\%$ in the reducible loss term — order 0.01 nats at this scale. The *reported* GQA-vs-MHA quality gap is smaller than that: a few tenths of a benchmark point, plausibly under 0.005 nats.

The measured gap is therefore **smaller than the confound**. Two readings fit the same data equally well:

| Hypothesis | Predicted $\Delta L$ | Consistent with published numbers? |
|---|---|---|
| Sharing is free; all loss is the 806M missing parameters | $\approx 0.010$ | yes |
| Sharing hurts, but 806M fewer KV parameters were partly dead weight | $\approx 0.010$ | yes |

Nothing published separates them, because no run holds $N$ fixed while varying $G$. That is precisely why §8's control arm (b) — restore the 806M parameters as extra layers or FFN width — is not optional bookkeeping. It is the only way $\Delta(G)$ becomes identifiable at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*