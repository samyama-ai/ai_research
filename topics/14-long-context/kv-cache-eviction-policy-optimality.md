---
id: 14-long-context/kv-cache-eviction-policy-optimality
title: "KV Cache Eviction Policy Optimality"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# KV Cache Eviction Policy Optimality

> **Topic:** Long Context · **ID:** `14-long-context/kv-cache-eviction-policy-optimality` · **Status:** open

## 1. Problem Statement

An autoregressive transformer serving a long prompt stores one key/value pair per token per layer per KV head. Memory grows linearly in context length; at 128K tokens it dominates weights. **Eviction** discards a subset of those pairs irrevocably and answers subsequent queries from what remains.

- **Input:** a token stream $x_{1:n}$, a per-head cache budget $B \ll n$, and a model $f_\theta$.
- **Output:** an online policy $\pi$ that, at each step, chooses which cached entry to drop.
- **Objective:** minimize downstream task loss subject to the budget.

Three variants, routinely conflated:

- **Measurement:** what *is* the loss induced by evicting a given set? Per-token output error, perplexity, and task accuracy disagree, and no accepted scalar exists.
- **Method:** find a policy with better loss-at-budget than the incumbents (H2O, SnapKV, PyramidKV, StreamingLLM).
- **Theory:** bound the competitive ratio of any online policy against the hindsight-optimal subset. Nothing here is proved for transformers.

Solving it means: a policy with a proved worst-case ratio against offline optimum, plus an empirical demonstration that the ratio is achieved at 128K on retrieval-heavy tasks. Currently neither half exists.

## 2. Formal Setting

Fix layer $\ell$ and KV head $h$. Let $k_j, v_j \in \mathbb{R}^{d_h}$ be the key/value at position $j$. Full attention at step $t$:

$$o_t \;=\; \sum_{j \le t} a_t(j)\, v_j, \qquad a_t(j) \;=\; \frac{\exp(q_t^\top k_j / \sqrt{d_h})}{\sum_{j' \le t} \exp(q_t^\top k_{j'} / \sqrt{d_h})}.$$

A policy maintains $\mathcal{C}_t \subseteq \{1,\dots,t\}$, $|\mathcal{C}_t| \le B$, with the **irrevocability constraint** $\mathcal{C}_{t+1} \subseteq \mathcal{C}_t \cup \{t+1\}$. The restricted output renormalizes over the survivors:

$$\tilde o_t = \sum_{j \in \mathcal{C}_t} \tilde a_t(j) v_j, \qquad \tilde a_t(j) = \frac{\exp(q_t^\top k_j/\sqrt{d_h})}{\sum_{j' \in \mathcal{C}_t} \exp(q_t^\top k_{j'}/\sqrt{d_h})}.$$

**Measured quantities.**

- *Budget in bytes*, the only quantity a serving system cares about: $M = 2 L H_{kv} d_h B \cdot b/8$, with $b$ bits per element. Report this, not "compression ratio" — ratios hide the sink/recent-window overhead policies add back.
- *Per-step attention loss* $\delta_t = \sum_{j \notin \mathcal{C}_t} a_t(j) \in [0,1]$: the probability mass thrown away. Directly measurable from a full-cache reference run.
- *Output error* $\varepsilon_t = \lVert \tilde o_t - o_t \rVert_2 / \lVert o_t \rVert_2$, measured layer-by-layer against a full-cache teacher-forced run on the same inputs.
- *Task loss* $\mathcal{L}(\pi)$: exact-match or F1 on RULER / LongBench / $\infty$Bench, decoded greedily.
- *Offline optimum* $\mathrm{OPT}(B)$: minimum $\mathcal{L}$ over all budget-$B$ subset schedules with hindsight over the full generation. Competitive ratio $\rho(\pi) = \sup \mathcal{L}(\pi)/\mathcal{L}(\mathrm{OPT})$.

**Assumptions, and which are violated.**

1. *Attention score is a sufficient importance statistic.* Violated: keys with small attention but large value norm matter; $\lVert v_j \rVert$ varies by ~an order of magnitude across positions, and value-norm-weighted variants change the ranking.
2. *Accumulated attention $s_j = \sum_{t} a_t(j)$ is unbiased across positions.* Violated by construction — token $j$ can only accumulate over $n-j$ steps, so the statistic favors early tokens.
3. *Submodularity of the accumulated-score set function*, assumed by H2O's greedy guarantee. Violated: softmax renormalization over $\mathcal{C}_t$ makes each token's contribution depend on what else survives, so the set function is not submodular in general.
4. *Persistence of importance* (Scissorhands): tokens important now stay important. Violated whenever the disambiguating query arrives after eviction — the retrieval case.
5. *Per-layer independence.* Violated: errors compose across 32+ layers; $\varepsilon$ at layer 1 changes the queries at layer 2.

## 3. State of the Art

**Empirical/systems SOTA (established, reproduced).**

- **StreamingLLM** (Xiao et al., ICLR 2024): keep the first few "attention sink" tokens plus a recent window. Reproduced widely; the sink effect is now a standard architectural fact.
- **SnapKV** (Li et al., NeurIPS 2024): score prefill tokens by the attention they receive from an observation window at the end of the prompt, pool, then keep top-$B$. Prompt-compression only — it does not evict during generation. Independently reproduced in NVIDIA's KVPress.
- **PyramidKV** (Cai et al., 2024): allocate larger budgets to lower layers. The non-uniform-across-layers finding replicates; the specific pyramid schedule is one point in a family.
- **Ada-KV** (Feng et al., 2024): head-wise budget reallocation with an $L_1$ output-error bound for the *single-step, un-renormalized* case. The bound is real but does not cover multi-step renormalized decoding.
- **DuoAttention** (Xiao et al., ICLR 2025): learn which heads are "retrieval heads" and give only those full cache.

**Claimed but unablated.** H2O's greedy submodular guarantee is stated under an assumption the softmax violates; the paper's accuracy results do not isolate how much of the gain comes from the recent-token window versus the heavy-hitter selection. Throughput figures (H2O: up to 29× over baseline systems at 20% cache) are systems numbers against specific baselines, not policy-quality evidence. Most "X% compression with no loss" claims are **benchmark numbers only**, measured on LongBench, whose tasks are answerable from local context and so under-penalize eviction.

**Theory SOTA.** There is none for this problem. The closest transferable results are classical paging — Belady's offline MIN (1966) and Sleator–Tarjan (CACM 1985), which shows every deterministic online paging algorithm is at best $k$-competitive — and representational lower bounds showing $\Omega(n)$ memory is needed for exact copying/retrieval (Jelassi et al., ICML 2024). Neither maps cleanly: attention has no re-fetch, and its loss is a continuous function of the discarded mass rather than a miss count.

## 4. What Is Known

- **Sinks are load-bearing.** Dropping the first ~4 tokens raises Llama-2-7B perplexity from single digits to $>10^3$ (StreamingLLM, ICLR 2024, 7B scale). Every competitive policy now protects them.
- **Attention is empirically sparse.** Over 90% of attention mass concentrates on a small token subset in most heads at 7B–13B (H2O, Scissorhands, NeurIPS 2023) — but sparsity is measured *post hoc*, per query.
- **Uniform-across-layers budgets are suboptimal.** Pyramid-shaped allocation beats uniform at equal total bytes on LongBench at 7B–8B (PyramidKV, 2024); the direction reproduces in KVPress.
- **Cheap statistics are competitive.** Keys with low $L_2$ norm carry most attention; an $L_2$-only policy matches attention-score policies at high compression on some models (Devoto et al., EMNLP 2024, 7B–8B).
- **Compression is not task-uniform.** Yuan et al. (EMNLP Findings 2024) show methods that look lossless on summarization degrade sharply on arithmetic and multi-step reasoning at the same budget — the first systematic evidence that the headline benchmarks mismeasure.
- **Prompt-eviction ≠ multi-turn eviction.** SCBench (Li et al., ICLR 2025) shows sub-$O(n)$ cache methods degrade across turns when the cache is reused, a regime LongBench never tests.

## 5. What Is Not Known

- **Theoretically open.** No competitive-ratio bound, upper or lower, for online KV eviction under the irrevocability constraint with softmax renormalization. It is not even known whether a constant-competitive policy exists for attention-mass loss $\sum_t \delta_t$ at budget $B$, nor whether the offline problem is NP-hard.
- **Empirically open.** The hindsight-oracle gap. Nobody has measured, at 128K on a frontier-scale model, how far the best online policy sits from an offline oracle that sees the query. The experiment is runnable today on 8 GPUs.
- **Methodologically blocked.** There is no agreed loss functional. $\delta_t$, $\varepsilon_t$, perplexity, and task EM rank policies differently, and no paper reports the rank correlation between them. Until one is fixed, "optimal" has no referent.

## 6. Why It Is Hard

**The deciding statistic is computed before the information that determines relevance exists.** Any online policy scores token $j$ using queries $q_{t \le t_{\mathrm{evict}}}$. In retrieval workloads the query that makes $j$ relevant arrives strictly later. This is not a compute problem — it is an information-theoretic one, and it means the achievable competitive ratio may be unboundedly bad on adversarial inputs while excellent on the average benchmark.

Compounding it: **the evaluation does not measure what it names.** LongBench scores are dominated by tasks solvable from the last few thousand tokens, so a recency-only policy scores near full cache. Reporting "95% of full performance at 12% cache" on such a suite is consistent with a policy that fails every needle retrieval.

Third: **absent ground truth for OPT.** Computing the true offline optimum requires searching $\binom{n}{B}$ schedules under a loss that depends on the surviving set through renormalization — non-decomposable, so greedy hindsight gives a bound, not the optimum.

## 7. Current Research (as of 2026)

- **Selection over eviction.** Quest (Tang et al., ICML 2024) and successors keep everything, offloaded, and retrieve top-$B$ pages per query — sidestepping irrevocability at the cost of bandwidth. This is the strongest practical direction and increasingly the default in production stacks. *(frontier — verify)*
- **Learned/architectural allocation.** DuoAttention, RazorAttention (ICLR 2025), and Dynamic Memory Compression (Nawrot et al., ICML 2024) move the decision into training rather than inference.
- **Reproducible benchmarking.** NVIDIA's KVPress and SCBench are consolidating the field's numbers; expect several published gains to shrink under uniform byte accounting. *(frontier — verify)*
- **Hybrid quantize-then-evict.** KIVI/KVQuant-style 2-bit caches compose with eviction; the joint frontier is under-explored.

## 8. Concrete Next Experiment

**Measure the online–oracle gap.**

- **Scale:** Llama-3.1-8B-Instruct, RULER at 128K (13 subtasks), plus $\infty$Bench En.MC. Budgets $B \in \{256, 512, 1024, 2048, 4096\}$ per head. ~8 A100/H100-80GB, under 500 GPU-hours.
- **Arms:** (a) full cache — upper reference; (b) recency-only + 4 sinks — lower control, the arm most papers beat only marginally; (c) H2O, SnapKV, PyramidKV, Ada-KV at matched *bytes*, not matched ratio; (d) **hindsight oracle**: run the full cache, record $a_t(j)$ for all generated $t$, keep the top-$B$ positions by $\max_t a_t(j)$, then re-decode with only those. Feasible-set, so it lower-bounds $\mathrm{OPT}$'s loss weakly and is computable.
- **Deciding number:** $G = \mathrm{EM}(\text{oracle}) - \mathrm{EM}(\text{best online})$ at $B = 512$ on RULER-128K multi-key NIAH.
  - $G \le 2$ points: online eviction is already near the information ceiling; further policy work is wasted, and gains must come from selection/offload.
  - $G \ge 20$ points: the ceiling is far away and the blocker is the statistic, not the budget — justifying query-aware or learned schedulers.
- **Required secondary output:** Spearman correlation between policy rankings under $\sum_t \delta_t$, mean $\varepsilon_t$, perplexity, and EM. If below ~0.7, the field's loss functional is unresolved and Section 5's methodological block is confirmed empirically.

## 9. Key References

- **[Foundational]** L. A. Belady. *A study of replacement algorithms for a virtual-storage computer.* IBM Systems Journal, 1966.
- **[Foundational]** D. Sleator, R. Tarjan. *Amortized efficiency of list update and paging rules.* Communications of the ACM, 1985.
- **[Foundational]** Z. Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS, 2023. — arXiv:2306.14048
- **[Foundational]** G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[SOTA]** Y. Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS, 2024. — arXiv:2404.14469
- **[SOTA]** Z. Cai et al. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* 2024. — arXiv:2406.02069
- **[SOTA]** J. Tang et al. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML, 2024. — arXiv:2406.10774
- **[SOTA]** G. Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* ICLR, 2025. — arXiv:2410.10819
- **[Evaluation]** C.-P. Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Evaluation]** J. Yuan et al. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* Findings of EMNLP, 2024. — arXiv:2407.01527
- **[Related]** S. Jelassi et al. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Related]** A. Devoto et al. *A Simple and Effective $L_2$ Norm-Based Strategy for KV Cache Compression.* EMNLP, 2024. — arXiv:2406.11430

## 10. Worked Example

Llama-3.1-8B-Instruct: $L=32$, $H_{kv}=8$, $d_h=128$, fp16. Per token:

$$2 \times 32 \times 8 \times 128 \times 2\ \text{bytes} = 131{,}072\ \text{B} = 128\ \text{KiB}.$$

At $n = 128{,}000$: **15.6 GiB** of cache against 15.0 GiB of weights. Setting $B = 1024$ gives 128 MiB — a 125× reduction, and exactly the regime where policies are advertised as lossless.

Now place a needle at position $p = 40{,}000$ ("the passcode is 7431") and the question at $t^\star = 127{,}900$. Under accumulated-attention scoring, the needle's score during prefill is $s_p = \sum_{t=p}^{n} a_t(p)$. Intervening tokens are unrelated filler, so $a_t(p) \approx 10^{-5}$ over ~88,000 steps, giving $s_p \approx 0.9$. A high-frequency function word appearing 400 times, each receiving $a \approx 0.01$ from nearby tokens, scores $\approx 4$. The needle ranks **below** thousands of syntactically salient tokens and is evicted long before $t^\star$.

At $t^\star$ the true attention puts $a_{t^\star}(p) = 0.62$ on the needle. With $p$ evicted, that mass renormalizes across survivors: $\delta_{t^\star} = 0.62$, and

$$\varepsilon_{t^\star} = \frac{\lVert \tilde o - o\rVert}{\lVert o \rVert} \approx \frac{0.62 \,\lVert v_p - \bar v_{\mathcal{C}}\rVert}{\lVert o \rVert} \sim 0.5$$

for value vectors of comparable norm — a half-magnitude error in one head at one step, which is enough to flip the decoded digit.

The obstruction is visible in one line: $s_p \approx 0.9$ versus $a_{t^\star}(p) = 0.62$. The statistic that decided the eviction is **five orders of magnitude smaller** than the relevance it was standing in for, because it was averaged over 88,000 queries that had no reason to look at the needle. No amount of tuning the scoring function fixes this — the query that makes $p$ important did not exist when $p$ was scored. Doubling $B$ to 2048 does not help either; the needle is not near the top of the ranking, it is near the middle. Only a query-aware method (retrieve rather than evict) or an oracle recovers it — which is precisely the gap Section 8 proposes to quantify.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*