---
id: 12-quantization-compression/kv-cache-compression-lower-bound
title: "KV Cache Compression Information-Theoretic Lower Bound"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# KV Cache Compression Information-Theoretic Lower Bound

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/kv-cache-compression-lower-bound` · **Status:** open

## 1. Problem Statement

How many bits of state must an autoregressive transformer carry forward per context token to answer an arbitrary downstream query at a fixed compute budget?

Every deployed KV cache compression method — quantization, eviction, low-rank projection, head merging — reports a compression ratio and a task score. None is measured against a floor. Without a floor there is no way to tell whether 2-bit KV quantization is near-optimal or leaving 100x on the table.

Three variants, with different difficulty:

- **Theory.** Prove a lower bound $B^\star(n, \varepsilon, C)$ on the bits of per-token state any algorithm must retain to answer queries with distortion $\le \varepsilon$ under generation compute budget $C$. Unconditional bounds are the goal; bounds under a restricted computational model (streaming, one-pass, attention-only readout) are the realistic near-term target.
- **Measurement.** Define the distortion. "Perplexity within 0.1" and "LongBench score within 2 points" are not the same predicate and do not order methods the same way.
- **Method.** Construct a compressor that provably meets the bound, or show a constant-factor gap to the best known one.

Solving it means: a bound that is (a) non-vacuous at production context lengths, (b) tight to within a stated constant against a matching construction, and (c) stated as a function of the compute budget, not memory alone.

## 2. Formal Setting

Let $x_{1:n}$ be a context of $n$ tokens over vocabulary $\mathcal{V}$. A transformer with $L$ layers, $H_{kv}$ key/value heads, head dimension $d_h$, and precision $b$ bits produces the exact cache

$$\mathcal{K} = \{(k_\ell^{(h)}(x_{1:t}), v_\ell^{(h)}(x_{1:t}))\}_{\ell \le L,\, h \le H_{kv},\, t \le n}, \qquad |\mathcal{K}| = 2 n L H_{kv} d_h b \ \text{bits}.$$

Measured as: allocated tensor bytes, not theoretical entropy. Report per token: $\beta = 2 L H_{kv} d_h b$ bits/token.

A **compressor** is a pair $(\mathrm{Enc}, \mathrm{Dec})$ with $\mathrm{Enc}: x_{1:n} \mapsto s \in \{0,1\}^{S}$ computed causally (one pass, no lookahead) and a decoder that, given a query $q$ arriving *after* $\mathrm{Enc}$ has run, emits $\hat{y} = \mathrm{Dec}(s, q)$ using at most $C$ FLOPs. Define the per-token rate $R = S/n$ bits.

**Distortion.** Three candidate measures, which do not agree:

$$D_{\mathrm{KL}} = \mathbb{E}_{q}\big[ \mathrm{KL}(p(\cdot \mid x_{1:n}, q) \,\|\, \hat{p}(\cdot \mid s, q)) \big], \quad D_{\mathrm{attn}} = \mathbb{E}\|\mathrm{Attn}(q,\mathcal{K}) - \widehat{\mathrm{Attn}}(q,s)\|_2, \quad D_{\mathrm{task}} = \Pr[\hat{y} \ne y^\star].$$

$D_{\mathrm{attn}}$ is what most methods bound; $D_{\mathrm{task}}$ is what users care about; $D_{\mathrm{KL}}$ is the only one that composes across the $m$ tokens of a generated answer, where errors compound as roughly $m \cdot D_{\mathrm{KL}}$ by Pinsker plus a union bound.

The object sought is the rate–distortion–compute surface

$$R^\star(\varepsilon, C) = \inf \{ R : \exists (\mathrm{Enc},\mathrm{Dec}) \text{ with } D \le \varepsilon \text{ and decode cost} \le C \}.$$

**Assumptions, and which are violated.**
- *Query independent of the encoder* — violated by prompt-aware methods (SnapKV, Quest) that see the query before or during compression; their numbers are not comparable to query-agnostic ones.
- *Distortion at the attention output predicts task error* — violated: attention error concentrates on a few heads whose failure is catastrophic, so the mean is the wrong statistic.
- *Tokens i.i.d. under a stationary source* — violated; natural context has long-range structure that a bound assuming independence will under-count.
- *Fixed compute budget* — violated by every recomputation-based scheme, which converts memory into FLOPs and so lives on a different point of the surface.

## 3. State of the Art

**Theory SOTA.** There is no published unconditional lower bound on KV cache size for transformer inference at a fixed compute budget. The nearest established results:

- One-way communication complexity of INDEX is $\Omega(n)$ bits (Kremer, Nisan, Ron, STOC 1995), which gives an $\Omega(n)$ bit floor for *any* streaming summary that must answer an adversarially chosen retrieval query. Established, but it bounds bits per *context*, not bits per *token times model width*, so it is vacuous against the $\beta \approx 10^6$ bits/token that real caches use.
- Upper bounds under distributional assumptions: SubGen (Zandieh, Han, Mirfakhraei, Karbasi, 2024) achieves sublinear $\tilde{O}(\varepsilon^{-2})$ memory for attention approximation under a bounded key-clusterability assumption. Established as a theorem; the assumption is not verified on production models.
- Rate–distortion framing for the adjacent problem of *prompt* compression: Nagle, Girish, Bondaschi, Gastpar, Makkuva, Kim, "Fundamental Limits of Prompt Compression: A Rate-Distortion Framework" (NeurIPS 2024). Establishes an optimal-rate LP for token-level prompt compression and shows existing heuristics sit well below the frontier. The KV-cache analogue has not been written.

**Systems/empirical SOTA.** Established, independently reproduced: 4-bit and 3-bit KV quantization with per-channel key / per-token value grouping (KIVI, Liu et al., ICML 2024; KVQuant, Hooper et al., NeurIPS 2024) holds perplexity degradation under ~0.1 on Llama-2-7B. Architectural reductions — GQA (Ainslie et al., EMNLP 2023) and MLA (DeepSeek-V2, 2024) — are training-time changes, and the 93.3% cache reduction MLA reports is a design fact, not a compression bound.

**Claimed but unablated.** Eviction methods (H2O, Zhang et al., NeurIPS 2023; SnapKV, Li et al., NeurIPS 2024; PyramidKV, Cai et al., 2024) report 10–20% cache budgets at near-full quality. These numbers exist only as benchmark scores on LongBench-style suites, whose retrieval demands are weak; Yuan et al., "KV Cache Compression, but What Must We Give in Return?" (EMNLP Findings 2024) and SCBench (Li et al., ICLR 2025) show the same methods degrade sharply on multi-turn and exact-retrieval workloads. The claimed ratios are not budget-invariant.

## 4. What Is Known

- Cache size per token is exactly computable: Llama-3-8B (32 layers, 8 KV heads, $d_h=128$, fp16) uses 128 KiB/token; 128K context = 16 GiB. Llama-2-7B (MHA, 32 KV heads) uses 512 KiB/token.
- 4-bit KV quantization is safe at 7B–70B scale: <0.1 perplexity change, and KVQuant reports serving 1M context on 8×A100 at 3 bits (NeurIPS 2024).
- 2-bit is the empirical cliff: KIVI reports 2-bit works with fp16 residual for the most recent ~32–128 tokens and per-channel key grouping; without those, 2-bit collapses. Measured at 7B–13B.
- Attention is empirically sparse: ~5% of positions carry most of the mass for most heads (H2O, Scissorhands, NeurIPS 2023), and attention-sink tokens are load-bearing (StreamingLLM, Xiao et al., ICLR 2024 — 4 sink tokens plus a sliding window stabilizes generation to 4M tokens).
- Sparsity does not imply compressibility to the same ratio: RULER (Hsieh et al., COLM 2024) shows models with nominal 128K windows have effective retrieval lengths far shorter, so headline compression ratios measured on weak benchmarks overstate.
- Lower-bound-side: nothing tight. The $\Omega(n)$ INDEX bound and $\Omega(n^{2-o(1)})$ fine-grained *time* bounds for exact attention under SETH (Duman Keles, Wijewardena, Hegde, ALT 2023) are the established formal anchors, and neither constrains bits/token.

## 5. What Is Not Known

- **Theoretically open.** Is there a lower bound of the form $R^\star(\varepsilon, C) = \Omega(f(d, \varepsilon))$ per token for query-agnostic caches, and does it separate from the $O(\log n)$-bit regime achievable when the query is known in advance? No proof either way. Whether recomputation-augmented decoders (unbounded $C$) collapse $R^\star$ to the entropy of the raw text is also unproven.
- **Methodologically blocked.** The distortion measure. Until the field agrees whether $\varepsilon$ is $D_{\mathrm{KL}}$ per generated token, worst-case over heads, or task accuracy, "optimal rate" is not a well-posed quantity, and a bound proved for one measure will not bind for another.
- **Empirically open.** Nobody has measured the achievable rate–distortion curve with a strong, compute-unconstrained encoder (e.g. a learned autoencoder over the cache trained per model) at 70B scale and 128K context. That experiment is runnable today and would give the first credible upper envelope to compare heuristics against.

## 6. Why It Is Hard

The specific obstruction is **the query arrives after the encoder commits**, combined with **absent ground truth for the distortion target**.

The first makes the problem adversarial rather than average-case: the encoder must be simultaneously good for every query in a set whose size grows with $n$, so any bound derived from average attention sparsity is measuring the wrong quantity. The second makes it unfalsifiable in practice: there is no reference "correct" long-context output to compare against, so methods are scored on benchmarks whose retrieval difficulty is far below worst case. A method that passes LongBench at 10% budget and fails a 5-needle retrieval at 50% budget is not caught by the evaluation that certified it — the evaluation does not measure the thing it names.

Compute cost is secondary but real: sweeping rate against distortion at 70B/128K requires a few thousand GPU-hours per method, which is why nobody publishes full curves.

## 7. Current Research (as of 2026)

- **Rate–distortion formalisms for LLM state.** The EPFL/UT-Austin line (Gastpar, Kim, Makkuva) that produced the prompt-compression limits is the natural source of a KV-cache analogue *(frontier — verify)*.
- **Architectural sidesteps.** MLA-style latent caches (DeepSeek) and hybrid state-space/attention stacks push the bits/token down by construction rather than by proving a floor. Industrially dominant; theoretically silent.
- **Query-aware and multi-turn-safe compression.** DuoAttention (Xiao et al., ICLR 2025), Quest (Tang et al., ICML 2024), and the KV-Press ecosystem (NVIDIA) target the reusability failure that SCBench exposed.
- **Adversarial evaluation.** RULER, ∞Bench, and SCBench successors are converging on retrieval-hard suites; this is the fastest path to making the measurement well defined.

## 8. Concrete Next Experiment

**Question:** how far is the best deployed KV compressor from the best *achievable* rate at fixed distortion?

- **Scale.** Llama-3-8B, 32K context, 200 held-out contexts. Small enough for a full sweep, large enough that GQA and sinks are present.
- **Arm A (empirical floor).** Train a per-model, query-agnostic autoencoder over the layer-wise cache: encoder sees the full cache offline (compute-unconstrained, so it is an *upper envelope on achievable rate*, not a deployable method), decoder reconstructs $(k,v)$ under a rate constraint. Sweep $R \in \{2, 4, 8, 16, 32, 64\}$ bits/token/head-pair.
- **Control arm.** KIVI 2-bit and 4-bit, H2O at matched *bits/token* (not matched token budget — this is the comparison the literature omits), and uniform random eviction at the same rate.
- **Deciding number.** $\Delta R$ = the ratio of the control's rate to Arm A's rate at the same $D_{\mathrm{KL}} = 0.05$ nats/token, measured on a 5-needle retrieval task at 32K. If $\Delta R \le 2$, deployed methods are near the achievable frontier and the open problem is theoretical only. If $\Delta R \ge 10$, there is an order of magnitude on the table and the problem is a method problem.

Cost estimate: ~600 A100-hours.

## 9. Key References

- **[Foundational]** Kremer, Nisan, Ron. *On randomized one-round communication complexity.* STOC, 1995.
- **[Foundational]** Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- **[Foundational]** Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023.
- **[Theory]** Nagle, Girish, Bondaschi, Gastpar, Makkuva, Kim. *Fundamental Limits of Prompt Compression: A Rate-Distortion Framework.* NeurIPS, 2024.
- **[Theory]** Duman Keles, Wijewardena, Hegde. *On the Computational Complexity of Self-Attention.* ALT, 2023.
- **[Theory]** Zandieh, Han, Mirfakhraei, Karbasi. *SubGen: Token Generation in Sublinear Memory.* 2024.
- **[SOTA]** Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML, 2024.
- **[SOTA]** Hooper, Kim, Mohammadzadeh, Mahoney, Shao, Keutzer, Gholami. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS, 2024.
- **[SOTA]** Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett, Wang, Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS, 2023.
- **[SOTA]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024.
- **[Evaluation]** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024.
- **[Evaluation]** Yuan, Liu, Zhong, Chen, Ling, Guo, Zhang, Chen, Hu, et al. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* EMNLP Findings, 2024.
- **[Survey]** Li, Jiang, Zhang, Qiu, et al. *SCBench: A KV Cache-Centric Analysis of Long-Context Methods.* ICLR, 2025.

## 10. Worked Example

Llama-3-8B, 128K-token context.

**Cache size.** $\beta = 2 \times 32 \times 8 \times 128 \times 16 = 1{,}048{,}576$ bits/token $= 128$ KiB. Total: $131072 \times 128\,\text{KiB} = 16$ GiB $= 1.37 \times 10^{11}$ bits.

**Information content of the source.** The same model assigns the context roughly 2 bits/token under its own likelihood (typical for English at ~4 chars/token). So the *text itself*, arithmetic-coded by the model, is $131072 \times 2 = 2.6 \times 10^5$ bits $\approx 32$ KiB.

**The gap.** $1.37 \times 10^{11} / 2.6 \times 10^5 \approx 5 \times 10^5$. The cache is half a million times larger than the information it is derived from, and no information is added by the forward pass — the cache is a deterministic function of the text. A pure information bound therefore says $R^\star \le 2$ bits/token, achieved trivially: store the compressed text, decompress, re-prefill.

**Where it breaks.** Re-prefilling costs $\approx 2 \times 8\times10^9 \times 131072 = 2.1 \times 10^{15}$ FLOPs, about 5 s on one H100 at realistic utilization, against ~10 ms for a cached decode step. The 2 bits/token bound is real and useless, because it is stated at $C = \infty$.

**What this shows.** Any lower bound that ignores the compute budget is vacuous, and any compression number that ignores it is incomparable. The live quantity is $R^\star(\varepsilon, C)$ at $C$ equal to a few hundred FLOPs per cached token — the regime where the decoder may do local arithmetic on the summary but cannot re-run the network. Nobody has proven anything about that regime. Meanwhile the best deployed methods sit at $R \approx 2\text{–}4$ bits per stored value, i.e. $\beta \approx 2.6 \times 10^5$ bits/token — still five orders of magnitude above the unconstrained floor, and with no theory saying whether that distance is forced by the compute budget or by the crudeness of the method.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*