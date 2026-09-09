---
id: 11-inference-and-serving/optimal-kv-cache-eviction-budget
title: "Optimal KV Cache Eviction Under Fixed Memory Budget"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal KV Cache Eviction Under Fixed Memory Budget

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/optimal-kv-cache-eviction-budget` · **Status:** open

## 1. Problem Statement

An autoregressive transformer stores one key/value pair per layer, per head, per token. At batch 32, 128K context, 70B-class models, this cache is tens of gigabytes — larger than the weights. A serving system therefore fixes a budget $B$ of retained entries and must decide, online, which to drop.

**The decision predicate.** Given a model $f$, a budget $B$, and a task distribution $\mathcal{D}$, does there exist an eviction policy $\pi$ that keeps end-task quality within $\epsilon$ of the full cache? And how far is any deployable $\pi$ from the best possible policy at that same $B$?

Three variants, of very different difficulty:

- **Measurement.** Define quality loss from eviction so that it is not dominated by tasks the model can answer without the evicted tokens. Most reported "lossless at 20% cache" numbers fail this.
- **Method.** Build $\pi$ that is causal (decides at write time or on a bounded lookback), cheap ($O(1)$ amortized per token, no full-cache scan), and hardware-friendly (contiguous pages, no per-head ragged layouts).
- **Theory.** Bound the competitive ratio of any online eviction policy against the offline optimum, and prove a memory lower bound $B^*(\text{task}, n)$ below which no policy can succeed.

Solving it means: an eviction rule with a proof or a robust empirical guarantee that quality degrades gracefully and *predictably* in $B$, across retrieval, aggregation, and multi-turn reuse — not just summarization.

## 2. Formal Setting

Prompt $x_{1:n}$, generation $x_{n+1:n+m}$. For layer $\ell$, head $h$, token $j$: $k_j^{(\ell,h)}, v_j^{(\ell,h)} \in \mathbb{R}^{d_h}$. Full cache size in bytes, as actually measured on device:

$$M_{\text{full}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot (n+m) \cdot b \cdot \text{batch}$$

with $b$ bytes per element (2 for fp16/bf16), $H_{kv}$ the number of *KV* heads under grouped-query attention. Budget $B$ is a byte figure; the *token budget* per head is $B_{tok} = B / (2 L H_{kv} d_h b \cdot \text{batch})$. Compression ratio $\rho = B / M_{\text{full}}$.

A policy $\pi$ maintains retained sets $S_t^{(\ell,h)} \subseteq \{1,\dots,t\}$ with $|S_t| \le B_{tok}$, and is **irrevocable**: $j \notin S_t \Rightarrow j \notin S_{t'}$ for all $t' > t$. Attention is then computed over $S_t$ only:

$$\tilde{a}_t(j) = \frac{\exp(q_t^\top k_j / \sqrt{d_h})}{\sum_{i \in S_t} \exp(q_t^\top k_i / \sqrt{d_h})}, \quad j \in S_t$$

**Measured quantities.**

- *Attention mass lost*: $\mathcal{L}_t = \sum_{j \notin S_t} a_t(j)$, where $a_t$ is the full-cache distribution. Computable only by running the uncompressed model in parallel — it is a diagnostic, not a deployable signal.
- *Output drift*: $\|\,\tilde{o}_t - o_t\|_2 / \|o_t\|_2$ at the attention output, per layer. This is what propagates; $\mathcal{L}_t$ is only a proxy for it, since a dropped token with small $a_t(j)$ but large $\|v_j\|$ can dominate.
- *Task loss*: $\Delta = \mathbb{E}_{\mathcal{D}}[\,\text{score}(f_{\text{full}}) - \text{score}(f_\pi)\,]$, per task family, with $\mathcal{D}$ stratified by whether the answer depends on evicted spans.
- *Offline optimum*: $\text{OPT}(B) = \min_{|S|\le B_{tok}} \Delta$, computed post hoc with the query sequence known. Approximable only by search on tiny instances.

**Assumptions and their status.**

1. *Persistence of importance* — a token with low accumulated attention will stay unimportant (Scissorhands' "persistence of importance hypothesis"). **Violated**: multi-turn and multi-query reuse changes which tokens matter after eviction is already committed (SCBench, ICLR 2025).
2. *Attention weight ranks value contribution*. **Violated in part**: value norms vary by an order of magnitude; $a_t(j)\|v_j\|$ is the correct ordering statistic and is not what most policies use.
3. *Uniform per-head budget is near-optimal*. **Violated**: retrieval heads carry most long-range recall (DuoAttention, RazorAttention), so uniform allocation over-serves local heads.
4. *Eviction composes with quantization and paging*. Untested at scale; the two interact through the same byte budget.

## 3. State of the Art

**Systems/empirical SOTA (established).**

- **StreamingLLM** (Xiao et al., ICLR 2024, arXiv:2309.17453): keep 4 "sink" tokens + a sliding window. Stable perplexity to 4M tokens. Established that the sink is an artifact of softmax mass dumping, reproduced widely. It does *not* retain long-range content — it is a stability result, not a compression result.
- **H2O** (Zhang et al., NeurIPS 2023, arXiv:2306.14048): greedy eviction on accumulated attention scores ("heavy hitters"). Reported near-full quality at 20% budget on OPT-6.7B/LLaMA-7B, with a submodular-style guarantee that holds only under a stated assumption on the attention matrix.
- **SnapKV** (Li et al., NeurIPS 2024, arXiv:2404.14469): compress the *prompt* cache using an observation window of the last ~32 queries plus pooling. 3.6× decode speedup, 8.2× memory cut at 16K on LLaMA-2-7B-chat variants. Currently the strongest prompt-side baseline.
- **Quest** (Tang et al., ICML 2024, arXiv:2406.10774) and page-selection methods: do not evict — keep everything in a lower memory tier, select pages per query. Sidesteps irrevocability at the cost of retaining $M_{\text{full}}$ somewhere.
- **Per-head budget allocation**: PyramidKV (arXiv:2406.02069), Ada-KV, DuoAttention (arXiv:2410.10819). Consistent gains over uniform allocation.

**Claimed but unablated.** Most "lossless at 10–20%" claims are benchmark numbers on LongBench-style suites where a compressed model can score well from the prompt tail alone. Yuan et al. (*KV Cache Compression, But What Must We Give in Return?*, EMNLP Findings 2024, arXiv:2407.01527) showed the same methods lose substantially on retrieval and aggregation at budgets where summarization looks unaffected. Head-budget allocators are rarely ablated against a matched-bytes quantization arm.

**Theory SOTA.** Thin. The relevant hard result is a *memory* lower bound rather than an eviction bound: Jelassi et al. (*Repeat After Me*, ICML 2024, arXiv:2402.01032) show bounded-state models fail at copying length-$n$ strings unless state grows with $n$ — any $B = o(n)$ policy fails on exact recall in the worst case. No competitive-ratio bound for online KV eviction against $\text{OPT}(B)$ is known.

## 4. What Is Known

- **The sink is real and cheap.** 4 tokens of budget; removing them collapses perplexity by orders of magnitude on LLaMA-2-7B (StreamingLLM, ICLR 2024).
- **Attention is sparse per query.** Scissorhands (NeurIPS 2023, arXiv:2305.17118) reports ~5× cache reduction with a few pivotal tokens dominating each query's mass on OPT-6.7B/30B/66B.
- **The sparsity pattern is not stable across queries.** The union of per-query top-$k$ sets over a long generation is far larger than any single top-$k$; this is the mechanism behind eviction's retrieval failures.
- **Budget-quality curves are non-monotone across tasks.** At $\rho = 0.2$, summarization ROUGE moves by ~1 point while needle-style exact retrieval can drop from near-100% to near-0% on the same 7B model (Yuan et al., 2024).
- **Layers differ.** Lower layers attend broadly, upper layers concentrate; pyramid-shaped budgets beat uniform at fixed total bytes (PyramidKV, 2024, 7B–8B scale).
- **Quantization is a competitive alternative at matched bytes.** KIVI (ICML 2024, arXiv:2402.02750) reports 2-bit asymmetric KV quantization with small quality loss, i.e. $\rho \approx 0.125$ *without eviction* — a control arm that eviction papers frequently omit.
- **Multi-turn reuse breaks single-shot compression.** SCBench (arXiv:2412.10319) shows methods tuned for one query degrade across turns sharing a prefix.

## 5. What Is Not Known

- **Theoretically open.** No competitive ratio for online, irrevocable KV eviction against the offline optimum $\text{OPT}(B)$ — not even for a single attention head with a stochastic query model. No known reduction to weighted caching that preserves the *softmax-normalized, query-dependent* value of a retained entry. The task-dependent threshold $B^*$ below which quality collapses is unproven outside the copying construction.
- **Empirically open.** No study reports the frontier of eviction vs. quantization vs. page-offload vs. attention-head sparsification at a *matched byte budget*, on the same model, at ≥70B, with ≥128K context, over a task mix stratified by evicted-span dependence. Every ingredient is runnable today; nobody has run the full grid.
- **Methodologically blocked.** "Quality loss from eviction" is not a well-defined measurement while benchmarks mix tasks that need the evicted tokens with tasks that do not. Until the stratification in §2 is standard, a headline number is uninterpretable.

## 6. Why It Is Hard

The specific obstruction is **irrevocability under an unknown future query distribution**, compounded by a **confounded evaluation**.

Eviction is an online decision made before the queries that would reveal a token's value. Unlike Belady's optimal cache replacement, the "next reference" is not a discrete access but a continuous, query-dependent softmax weight, and the cost of a mistake is not a miss but a silent perturbation of the output distribution. There is no ground-truth label for "this token was needed" short of running the uncompressed model.

The second obstruction: aggregate benchmark scores do not measure what they name. A method that keeps the prompt tail scores well on summarization at $\rho=0.1$ while having destroyed the model's ability to answer anything about the prompt head. The measurement rewards the failure mode.

## 7. Current Research (as of 2026)

- **Head-specialized budgets.** Retrieval-head identification and per-head allocation (DuoAttention, RazorAttention, Ada-KV lines) — MIT HAN Lab, Tsinghua, Microsoft Research. Established as a gain; the identification procedure's transfer across model families is *(frontier — verify)*.
- **Selection instead of eviction.** Quest/InfLLM/MInference-style page retrieval, plus disaggregated prefix caching in production stacks (vLLM, SGLang). Trades DRAM/SSD for HBM; changes the problem from "what to drop" to "what to fetch".
- **Hybrid attention architectures.** Sliding-window/full interleaving and linear-attention hybrids that bound the cache by construction (Nawrot et al., *Dynamic Memory Compression*, ICML 2024). *(frontier — verify)* whether trained-in compression dominates post-hoc eviction at matched bytes.
- **Merging rather than dropping.** Value-averaging evicted tokens into retained slots; reported gains are small and rarely ablated against simply enlarging $B$ by the same bytes.
- **Benchmark reform.** SCBench and needle-variant suites stratified by dependence on evicted spans.

## 8. Concrete Next Experiment

**The matched-byte frontier, run once, properly.**

- **Scale.** One 70B-class open model (e.g. Llama-3.1-70B-Instruct), 128K context, batch 8, single 8×H100 node. Six byte budgets: $\rho \in \{1.0, 0.5, 0.25, 0.125, 0.0625, 0.03\}$.
- **Arms, all at identical measured cache bytes:** (a) H2O, (b) SnapKV, (c) PyramidKV/Ada-KV allocation, (d) StreamingLLM window+sinks, (e) **control: KIVI-style 4-bit and 2-bit quantization with no eviction**, (f) **control: random eviction with sinks retained**, (g) full cache.
- **Evaluation, stratified.** Split every instance by whether the gold answer's supporting span falls inside the retained set under arm (d) at that $\rho$ — the "evicted-dependent" and "evicted-independent" strata — and report them separately. Include RULER (multi-key/multi-hop), a 5-turn shared-prefix multi-turn set, and one aggregation task.
- **The deciding number.** $\Delta_{\text{dep}}(\rho)$: accuracy drop on the *evicted-dependent* stratum, per arm, at $\rho = 0.125$. If no eviction arm beats the 2-bit quantization control on $\Delta_{\text{dep}}$ at equal bytes, the field's premise — that *which* tokens you keep matters more than *how precisely* you keep all of them — is falsified at that budget. If some arm beats it by >10 accuracy points, that arm's selection signal is the thing to formalize.
- **Cost estimate.** ~2,000 GPU-hours; 7 arms × 6 budgets × ~3 task suites.

## 9. Key References

- **[Foundational]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[Foundational]** Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett, Wang, Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023. — arXiv:2306.14048
- **[Foundational]** Liu, Desai, Liao, Wang, Xie, Xu, Kyrillidis, Shrivastava. *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* NeurIPS 2023. — arXiv:2305.17118
- **[SOTA]** Li, Huang, Yang, Hu, Zhang, Wang, Chen, Chen, et al. *SnapKV: LLM Knows What You Are Looking for Before Generation.* NeurIPS 2024. — arXiv:2404.14469
- **[SOTA]** Tang, Zhao, Zhu, Cai, Han, Dai, Chen. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024. — arXiv:2406.10774
- **[SOTA]** Xiao, Tang, Zhu, Han, Chen, Han. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* ICLR 2025. — arXiv:2410.10819
- **[Theory]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Control arm]** Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. — arXiv:2402.02750
- **[Evaluation]** Yuan, Liu, Zhong, Chuang, Li, Wang, Yin, Hu. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* Findings of EMNLP 2024. — arXiv:2407.01527
- **[Evaluation]** Li, Zhang, Jiang, Qianhui Wu, Xufang Luo, et al. *SCBench: A KV Cache-Centric Analysis of Long-Context Methods.* ICLR 2025. — arXiv:2412.10319
- **[Related]** Oren, Hassid, Adi, Schwartz. *Transformers are Multi-State RNNs.* EMNLP 2024 (TOVA policy). — arXiv:2401.06104
- **[Related]** Ge, Zhang, Liu, Zhang, Han, Gao. *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs.* ICLR 2024. — arXiv:2310.01801
- **[Survey]** Shi, Zhang, Yang, et al. *Keep the Cost Down: A Review on Methods to Optimize LLM's KV-Cache Consumption.* COLM 2024.

## 10. Worked Example

Llama-3-8B-Instruct: $L=32$, $H_{kv}=8$ (GQA), $d_h=128$, fp16. Per token, per layer: $2 \times 8 \times 128 \times 2 = 4{,}096$ bytes. Across 32 layers: **131 KB/token**. At $n = 128{,}000$: $\approx 16.8$ GB for one sequence — more than the 16 GB of weights, and at batch 4 it exceeds an 80 GB H100's remaining capacity.

Set $\rho = 0.125$, so $B_{tok} = 16{,}000$ per head.

Now the instance. Place a needle at position 2,000 of a 128K-token haystack; the question arrives at position 128,000.

- **StreamingLLM** (4 sinks + 15,996-token window): the retained window covers positions 112,004–128,000. Token 2,000 is gone. Recall = 0.
- **H2O**: token 2,000 is written at step 2,000, when the only queries that have scored it are from tokens 2,001 onward — generic continuation queries with no reason to attend to it. Its accumulated score sits near the median. It is evicted long before the question is asked. Recall ≈ 0, and *no online policy without the query can do better than chance here*: at step 2,000 the needle is information-theoretically indistinguishable from the 126,000 tokens that will turn out to be irrelevant.
- **SnapKV**: compresses using the last 32 prompt queries — which *include* the question. It retains the needle. Recall ≈ 1. But this only works because the query was in the prompt; in multi-turn serving the second turn's question was not available when the cache was compressed for the first.
- **KIVI 2-bit, no eviction**: retains all 128K tokens at 2.1 GB — the same byte budget as $\rho=0.125$ eviction. Recall ≈ 1, degraded only by quantization noise.

The obstruction is visible in the contrast between rows 2 and 4. At *identical bytes*, the eviction arm scores 0 and the non-eviction arm scores ~1 — because eviction spends its budget on an irreversible guess about the future, and quantization does not. And it is visible again in the contrast between rows 2 and 3: SnapKV's advantage is not a better importance signal, it is access to the query. Strip that access — the multi-turn case — and the gap closes. Any claim that a policy "solves" eviction at $\rho = 0.125$ has to survive both comparisons; the published ones are typically run against neither.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*