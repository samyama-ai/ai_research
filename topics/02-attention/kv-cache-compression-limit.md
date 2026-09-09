---
id: 02-attention/kv-cache-compression-limit
title: "KV Cache Compression Information Limit"
topic: 02-attention
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# KV Cache Compression Information Limit

> **Topic:** Attention Mechanisms · **ID:** `02-attention/kv-cache-compression-limit` · **Status:** open

## 1. Problem Statement

A decoder-only transformer serving a prompt of $n$ tokens stores $2 n L H d_h$ scalars of key/value state. Every deployed compression method — eviction, quantization, low-rank projection, head sharing — reduces that state and reports "no quality loss" on some benchmark. The question this page catalogs: **how many bits of KV state are actually necessary, as a function of context length $n$ and the task the model is asked to perform, before some query is answered wrong?**

Three variants, with different difficulty:

- **Theory.** For a fixed model $f_\theta$ and a query distribution $\mathcal{Q}$, is there a lower bound $B^\star(n, \mathcal{Q}, \epsilon)$ on the bits of prompt-derived state any decoder must retain to keep expected loss within $\epsilon$ of the uncompressed model? Open in general; solved for narrow task families (exact copying, associative recall).
- **Method.** Construct an encoder that provably attains $B^\star$, or comes within a constant factor. No method has been analyzed against a lower bound; all are heuristics validated by benchmark.
- **Measurement.** Given a compressor, *report* the bit budget at which it breaks, in a way that transfers to a task the reporter did not choose. This is the variant that is currently blocking the other two: near-universal practice is to tune the budget on the same benchmark used to claim losslessness.

Solving it means: a lower bound as a function of $n$ and a task-complexity parameter, a matching construction, and a benchmark whose failure point predicts failure points on held-out task families.

## 2. Formal Setting

Let $f_\theta$ be a transformer with $L$ layers, $H$ KV heads, head dimension $d_h$. Prompt $x_{1:n}$ induces cache $C(x_{1:n}) = \{(k_i^{(l,h)}, v_i^{(l,h)})\}$.

**Bit budget, as measured.** Not the compression ratio — the *resident bytes*:
$$B = \big|\,\mathrm{enc}(C(x_{1:n}))\,\big|_{\text{bits}} + \big|\,\text{side state}\,\big|_{\text{bits}}$$
Side state is where reported numbers go wrong: per-group scale/zero-point in quantization (KIVI at group size 32 adds $\approx 1$ bit/element at fp16 scales), retained full-precision "sink" and recent-window tokens, and page-level metadata in query-aware selection. A "2-bit cache" is routinely 2.5–3 effective bits. Any comparison must fix $B$ in bytes on the device, not in nominal precision.

**Distortion, as measured.** For a query $q$ drawn from $\mathcal{Q}$ and decoding rule $g$:
$$D(\mathrm{enc}) = \mathbb{E}_{x,q}\Big[\ell\big(g(\mathrm{enc}(C(x)), q),\, g(C(x), q)\big)\Big]$$
with $\ell$ measured against the **uncompressed same model**, not against ground truth. Measuring against ground truth confounds compression damage with the model's own error and lets a compressor look lossless on tasks the base model already fails.

**Rate–distortion object.** $B^\star(n,\mathcal{Q},\epsilon) = \min\{B : \exists\,\mathrm{enc},\ D(\mathrm{enc})\le\epsilon\}$. Two regimes must be separated:

- **Query-oblivious** ($\mathrm{enc}$ runs before $q$ is known — prefill-time eviction, KV quantization): $B^\star$ is a one-way communication complexity quantity and can be $\Omega(n)$.
- **Query-aware** ($\mathrm{enc}$ may read $q$ — SnapKV, Quest, sparse selection): $B^\star$ collapses toward the bits of the answer, $O(1)$ for a single lookup. Multi-turn reuse of one cache across many queries pushes it back to the oblivious regime.

**Assumptions known to be violated in practice.** (i) *Stationary importance* — that a token unimportant at step $t$ stays unimportant (the Scissorhands "persistence of importance" hypothesis); violated whenever a later query targets an evicted span. (ii) *Attention mass $\approx$ information* — low attention weight is treated as low value, but attention sinks carry near-zero information at very high mass, and induction-head keys carry high information at low mass. (iii) *Single query per cache* — most benchmarks; most production traffic is multi-turn on a shared prefix. (iv) *Uniform per-layer budget* — falsified by pyramid-shaped allocations.

## 3. State of the Art

**Theory SOTA.** Jelassi et al. (ICML 2024) prove that copying an $n$-token string requires state growing with $n$: any constant-state recurrent model fails, while a transformer with logarithmic-precision cache succeeds. Arora et al. (ICLR 2024, "Zoology") establish an empirical and partly analytic recall–state-size Pareto frontier for multi-query associative recall. Haris & Onak (2025) give communication-complexity barriers for autoregressive attention, implying no query-oblivious encoder achieves sublinear-in-$n$ state for exact attention over arbitrary inputs. These bound *task families*, not *deployed compressors*. **No lower bound is known for $B^\star$ of a trained LLM on natural-language distributions.**

**Systems SOTA (established, independently reproduced).** Grouped-query attention (Ainslie et al., EMNLP 2023) and multi-head latent attention (DeepSeek-V2, 2024; MLA reports 93.3% KV reduction vs. its MHA baseline) are *architectural* and are trained-in — these are the only reductions that are unambiguously not lossy post-hoc. StreamingLLM (Xiao et al., ICLR 2024) establishes the attention-sink phenomenon and stable perplexity to 4M tokens with a fixed window.

**Claimed but unablated.** H2O (NeurIPS 2023) reports quality retained at 20% cache; SnapKV (NeurIPS 2024) reports 8.2× memory reduction with "comparable" performance; PyramidKV, Ada-KV, and their descendants report near-lossless LongBench scores at 1–2% budget. These are **benchmark numbers, not ablations of the information limit**: LongBench-style tasks are largely summarization and QA over redundant text, where the answer is recoverable from a small fraction of tokens. Yuan et al. (EMNLP Findings 2024) run the cross-method control and find the claims do not survive task shift. Nawrot et al. (2025, "The Sparse Frontier") find the sustainable sparsity is strongly task- and size-dependent, with no single budget safe across tasks.

## 4. What Is Known

- **Sinks are real and cheap.** Evicting the first 4 tokens of a 4-token-sink + 1020-token window collapses perplexity by orders of magnitude on Llama-2-7B; retaining them restores it (Xiao et al., ICLR 2024, measured to 4M tokens).
- **2-bit KV quantization is near-free on generation-heavy tasks, and not free elsewhere.** KIVI (ICML 2024) reports $\approx 2.6\times$ peak-memory reduction and 2.35–3.47× throughput at 2-bit on Llama-2-7B/13B, Falcon-7B, Mistral-7B, with keys quantized per-channel and values per-token — the asymmetry is itself a reproduced finding.
- **Aggressive eviction fails needle-style retrieval specifically.** Under the Yuan et al. benchmark, methods that hold LongBench scores at ~20% budget degrade sharply on exact-retrieval and multi-hop subtasks at the same budget. The failure is task-selective, not uniform.
- **Effective context $\ll$ claimed context even uncompressed.** RULER (Hsieh et al., COLM 2024) shows most models claiming 32K support fall below their 4K baseline well before 32K. Any compression claim measured against a model already past its effective length is uninformative.
- **Budget allocation matters more than the scoring rule.** Pyramid-shaped per-layer budgets beat uniform allocation at fixed total $B$ across several scorers — reproduced across PyramidKV and Ada-KV lines.
- **Query-aware selection dominates query-oblivious eviction at equal $B$.** Quest (ICML 2024) reports 7.03× self-attention speedup at high accuracy by selecting pages per query — consistent with the oblivious/aware separation in §2.

## 5. What Is Not Known

- **Theoretically open.** Whether $B^\star(n,\mathcal{Q},\epsilon)$ is sublinear in $n$ for natural-language $\mathcal{Q}$ at any nonzero $\epsilon$. Copying and MQAR lower bounds are worst-case over adversarial strings; natural text is highly redundant and the bounds say nothing about it. No upper bound better than "store everything" is proven for a trained model.
- **Theoretically open.** Whether attention-score-based scoring is ever within a constant factor of optimal. No approximation guarantee exists for H2O-family scorers.
- **Empirically open.** The scaling of the break point in $(n, B, \text{model size})$. Nobody has run a single compressor across $\{7\text{B}, 70\text{B}, 400\text{B}\}\times\{8\text{K},128\text{K},1\text{M}\}$ with matched byte budgets and a fixed task suite. Cost, not difficulty, is the reason.
- **Empirically open.** Multi-turn cache reuse. Essentially all published numbers are single-query; the regime where one compressed cache must serve $k$ unknown future queries is unmeasured.
- **Methodologically blocked.** There is no accepted measurement of "bits of information in a KV cache." Reported ratios mix nominal precision, side state, and retained-token counts; degradation is scored against ground truth rather than the uncompressed model; and budgets are tuned on the reporting benchmark. Until $B$ and $D$ are standardized as in §2, the empirical questions cannot be answered even in principle.

## 6. Why It Is Hard

The obstruction is **the evaluation does not measure what it names, compounded by non-identifiability of the query distribution**.

Concretely: $B^\star$ is defined relative to $\mathcal{Q}$, and the compressor's designer chooses $\mathcal{Q}$ by choosing the benchmark. A summarization-heavy suite makes almost any eviction rule look lossless, because the target output is invariant to which of many redundant tokens survive. There is no distribution-free statement to fall back on, because the distribution-free answer is already known and useless: worst-case, you must keep $\Omega(n)$ bits (Haris & Onak; Jelassi et al.). So every interesting claim lives inside a choice of $\mathcal{Q}$ that is never specified, and two papers reporting "1% budget, lossless" may differ by 100× in true bit requirement.

Second obstruction: **absent ground truth for token importance.** There is no oracle labeling which cached entries a future query will need. Attention weight is used as a proxy and is known to be a biased one (sinks, induction heads). Without the oracle, no scorer can be shown suboptimal except by exhibiting a task where it fails — which is a lower bound on nothing.

## 7. Current Research (as of 2026)

- **Trained-in latent caches.** MLA (DeepSeek) and successors move compression into pretraining, where the encoder is optimized jointly with the model rather than bolted on. This sidesteps the post-hoc rate–distortion question entirely and is currently the strongest practical answer.
- **Learned compression policies.** Dynamic Memory Compression (Nawrot et al., ICML 2024) trains the model to decide online whether to append or merge into the cache. Duo Attention (Xiao et al., ICLR 2025) learns a per-head split into full-cache "retrieval heads" and window-only "streaming heads" — an empirical claim that $B^\star$ is concentrated in a small head subset.
- **Task-conditional budget theory.** Attempts to parameterize $B^\star$ by an explicit task descriptor (number of distinct retrievals, dependency span) rather than by benchmark identity. *(frontier — verify)*
- **Multi-turn / shared-prefix benchmarks.** SCBench (Li et al., ICLR 2025) evaluates the full cache lifecycle rather than single-query prefill, and reports that sub-$O(n)$ methods that look fine single-turn degrade under reuse.
- **Communication-complexity framing.** Haris & Onak and follow-ups pushing toward bounds that hold for approximate, rather than exact, attention. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question decided:** does a compressor's break point transfer across task families at fixed measured bytes, or is it a property of the benchmark?

- **Scale.** One model family, three sizes (8B, 70B), context 128K. Four compressors: StreamingLLM (window), H2O (oblivious eviction), SnapKV (query-aware), KIVI-2bit (quantization). Byte budgets swept at $B/B_{\text{full}} \in \{1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64\}$, **measured as resident bytes including all side state and retained fp16 tokens**, not nominal ratio. ~2 GPU-weeks on 8×H100.
- **Task suite.** Six families, held out from any tuning: single-needle retrieval, multi-needle (5), variable tracking, exact copying of a 2K random-token span, multi-hop QA, and summarization. Plus a **multi-turn arm**: one compressed cache, 8 sequential unrelated queries.
- **Control arm.** The identical model at $B = B_{\text{full}}$, with $D$ scored as agreement with *that* run's outputs — not against dataset ground truth. Second control: random eviction at the same byte budget, which is the floor any scorer must beat.
- **The deciding number.** For each (compressor, task family), the **break budget** $B_{50}$: the largest $B$ at which $D \ge 0.05$ (5% disagreement with the uncompressed control). The single decisive statistic is the **spread ratio** $\max_{\text{family}} B_{50} / \min_{\text{family}} B_{50}$ per compressor. If the spread is $< 4\times$, a single benchmark-derived budget is defensible and the field's reporting practice is sound. If it exceeds $\sim 16\times$ — which the Yuan et al. and Nawrot et al. results predict — then every published "X% lossless" number is a statement about the chosen benchmark and not about the compressor, and the measurement variant in §1 must be fixed before the method variant can progress.

## 9. Key References

- **[Foundational]** Noam Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- **[Foundational]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[Foundational]** Zhenyu Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023. — arXiv:2306.14048
- **[Foundational]** Zichang Liu et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* NeurIPS 2023. — arXiv:2305.17118
- **[Foundational]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[Theory]** Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Theory]** Simran Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024. — arXiv:2312.04927
- **[Theory]** Themistoklis Haris, Krzysztof Onak. *Compression Barriers for Autoregressive Transformers.* 2025. (identifier omitted — verify before citing)
- **[SOTA]** Zirui Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. — arXiv:2402.02750
- **[SOTA]** Yuhong Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS 2024. — arXiv:2404.14469
- **[SOTA]** Jiaming Tang et al. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024. — arXiv:2406.10774
- **[SOTA]** DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024. — arXiv:2405.04434
- **[SOTA]** Piotr Nawrot, Adrian Łańcucki, Marcin Chochowski, David Tarjan, Edoardo M. Ponti. *Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference.* ICML 2024. — arXiv:2403.09636
- **[Benchmark]** Jiayi Yuan et al. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* Findings of EMNLP 2024. — arXiv:2407.01527
- **[Benchmark]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. — arXiv:2404.06654
- **[Survey]** Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio, Edoardo M. Ponti. *The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs.* 2025.

## 10. Worked Example

**Setup.** Llama-3-8B, 32 layers, 8 KV heads, $d_h = 128$, fp16. Per token per layer: $2 \times 8 \times 128 \times 2\text{ B} = 4096$ B. Over 32 layers: $131{,}072$ B/token $= 128$ KiB/token. A 128K-token prompt holds
$$131072 \times 131072 \;=\; 1.72\times10^{10}\ \text{B} \;\approx\; 16\ \text{GiB}.$$

**The claim.** A method reports "lossless at 1.5% budget" — 245 MiB, ~1960 retained tokens.

**Where the accounting leaks.** Suppose the method keeps 4 sink tokens + a 512-token recent window in fp16, selects 1444 more, and stores per-token 8-bit indices. Retained state is $1960 \times 128\text{ KiB} = 245$ MiB, plus index side state $1960 \times 32 \times 8 \times 3\text{ B} \approx 1.5$ MiB. The leak is small here — but layer the same claim on top of 2-bit quantization and the "1.5% of 2-bit" budget silently excludes a fp16 window that is itself 64 MiB, a 26% understatement.

**Where the information limit bites.** Ask the model to reproduce a 2000-token random hexadecimal string embedded at position 60,000. The answer requires $2000 \times \log_2 16 = 8000$ bits of *irreducible* content, and — by the copying lower bound — no query-oblivious encoder can produce it without having retained state that determines those tokens. The compressor kept 1960 tokens chosen by attention score at prefill, before the copy instruction was seen. Expected overlap with the required span under a redundancy-driven scorer is near chance: $1960 \times (2000/131072) \approx 30$ tokens. Exact-match accuracy $\approx 0$.

**Same budget, same model, LongBench-style summarization of the same 128K prompt:** ROUGE-L within 1 point of uncompressed, because the summary is invariant to which redundant sentences survive.

**The obstruction, made visible.** One compressor, one byte budget, one model. $B_{50}$ differs between these two tasks by more than the full sweep range — the copy task needs essentially the uncompressed cache, the summarization task needs 1.5%. The number "1.5% lossless" is therefore not a property of the compressor. It is a property of which of these two tasks the author put in the table.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*