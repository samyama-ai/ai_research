---
id: 09-model-design/kv-cache-compression-limits
title: "KV Cache Compression Limits"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# KV Cache Compression Limits

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/kv-cache-compression-limits` · **Status:** open

## 1. Problem Statement

An autoregressive transformer decoding token $n+1$ must retain per-layer key and value vectors for all $n$ prior tokens. That cache grows linearly in context and dominates inference memory and bandwidth past ~32K tokens. The question: **how few bytes per token can a decoder retain and still reproduce the full-cache output distribution on arbitrary downstream queries?**

Three variants, routinely conflated:

- **Theory.** Is there a $\Omega(n)$ lower bound on cache bytes for exact (or $\epsilon$-close) next-token distributions under adversarial context? Or is sublinear memory achievable under a stated assumption about natural text?
- **Method.** Given a fixed byte budget $B$, what compressor minimizes downstream task loss? Eviction, quantization, low-rank projection, head sharing, and learned recurrent summarization are the candidate families.
- **Measurement.** What benchmark distinguishes a compressor that *preserves information* from one that *predicts the query distribution well*? This is where most published gains live and where the field is weakest.

Solving it means: a compressor with a proven distortion bound, plus an evaluation on which query-agnostic compression at budget $B$ matches an oracle that saw the queries before compressing.

## 2. Formal Setting

A decoder with $L$ layers, $H$ key/value heads, head dimension $d$, context length $n$, element width $w$ bytes. The uncompressed cache is
$$\mathcal{C}_n = \{(K^{(\ell)}, V^{(\ell)})\}_{\ell=1}^{L}, \quad K^{(\ell)}, V^{(\ell)} \in \mathbb{R}^{n \times H \times d},$$
with **measured size** $|\mathcal{C}_n| = 2\,n\,L\,H\,d\,w$ bytes — the number to report, not "tokens kept."

Per-token budget: $b = |\hat{\mathcal{C}}_n| / n$ bytes/token. Compression ratio $\rho = |\hat{\mathcal{C}}_n| / |\mathcal{C}_n|$.

A compressor is a map $\phi: \mathcal{C}_n \mapsto \hat{\mathcal{C}}_n$ applied **online** (streaming, one pass, no lookahead at the query). **Distortion** is measured at the output, not the cache:
$$D(\phi) = \mathbb{E}_{x \sim \mathcal{D}_{\text{query}}} \big[ \mathrm{KL}\big(p(\cdot \mid x, \mathcal{C}_n) \,\|\, p(\cdot \mid x, \hat{\mathcal{C}}_n)\big) \big].$$
Report KL in nats over the full vocabulary at each generated position — perplexity on a continuation is a strictly weaker probe and hides retrieval failure.

Task fidelity: $\Delta_{\text{task}} = \mathrm{Acc}(\mathcal{C}_n) - \mathrm{Acc}(\hat{\mathcal{C}}_n)$ at matched decoding temperature and matched wall-clock, on tasks with a known information requirement (RULER multi-key, multi-query needle retrieval), not on summarization.

The **information-theoretic floor**: if the context encodes $m$ facts of $k$ bits each and queries can address any one of them, the one-pass INDEX lower bound gives $|\hat{\mathcal{C}}_n| \ge mk$ bits (Kremer, Nisan, Ron 1999), independent of $n$.

Assumptions the method literature rests on, and their status:

| Assumption | Status |
|---|---|
| Attention is approximately sparse; a small heavy-hitter set persists over decoding | **Violated** for retrieval heads (Wu et al. 2024) and for query-shifted multi-turn use |
| Importance measured on the prompt transfers to future queries | **Violated** by construction under adversarial or unseen queries; SnapKV explicitly conditions on an observation window |
| Key/value entries are approximately Gaussian per channel | **Violated** — keys have persistent outlier channels; KIVI/KVQuant quantize keys per-channel for exactly this reason |
| Compression error at layer $\ell$ does not compound through $L$ layers | Unproven; no published bound on cross-layer error propagation |

## 3. State of the Art

**Architectural (established, ablated at scale).** MQA (Shazeer 2019) and GQA (Ainslie et al., EMNLP 2023) reduce $H$ directly; GQA-8 is standard in Llama-3. Multi-head Latent Attention (DeepSeek-V2, 2024) caches a shared low-rank latent plus a decoupled RoPE component — reported 93.3% cache reduction versus a same-family MHA 67B baseline, with training-time ablations. These are the only families where the reduction is architectural and the model was trained under it, so no post-hoc distribution shift exists.

**Quantization (established).** KIVI (Liu et al., ICML 2024): 2-bit, per-channel keys / per-token values, 2.6× peak memory reduction, 2.35–3.47× throughput. KVQuant (Hooper et al., NeurIPS 2024): 3-bit with <0.1 perplexity degradation on Llama-7B; 4-bit is essentially lossless across the board. Quantization is the one family that is uniformly reproducible because the error is unconditional on the query.

**Eviction/sparsity (claimed, incompletely ablated).** H2O (Zhang et al., NeurIPS 2023), Scissorhands (Liu et al., NeurIPS 2023), FastGen (Ge et al., ICLR 2024), SnapKV (Li et al., NeurIPS 2024), PyramidKV (Cai et al. 2024), Quest (Tang et al., ICML 2024), DuoAttention (Xiao et al. 2024). Headline numbers — H2O at a 20% budget with "negligible" quality loss, SnapKV 8.2× memory efficiency at 16K — are benchmark numbers on LongBench-style suites whose queries are visible in the prompt. They are not evidence about arbitrary later queries.

**Learned compression.** DMC (Nawrot et al., ICML 2024) retrofits Llama-2 with learned per-head merging at up to 8× on 4B extra training tokens. YOCO (Sun et al., NeurIPS 2024) caches once and shares across layers.

**Theory SOTA.** No tight bound for transformer decoding exists. The closest results are negative: fixed-state models cannot copy strings longer than their state (Jelassi et al., ICML 2024); associative recall accuracy is governed by recurrent state size (Arora et al., ICLR 2024); representational separations for attention (Sanford et al., NeurIPS 2023). SubGen (Zandieh et al. 2024) gives sublinear memory *under a key-clusterability assumption* that is not verified on production models.

## 4. What Is Known

- **Cache size is exactly computable.** Llama-3.1-8B: $L{=}32$, $H{=}8$, $d{=}128$, fp16 → $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ B/token = **128 KB/token**; 16 GB at 128K context, exceeding the 16 GB of weights.
- **4-bit KV is near-free.** Sub-0.1 perplexity change at 7B scale (KVQuant, NeurIPS 2024); reproduced independently across KIVI, KVQuant, and vLLM's FP8 path. This is the only compression claim with cross-lab replication.
- **Below 4 bits, error is task-dependent, not perplexity-dependent.** 2-bit KIVI holds perplexity but degrades on long-context retrieval.
- **Aggressive eviction fails on retrieval, passes on summarization.** Yuan et al. (EMNLP Findings 2024) benchmarked seven compression families and found retrieval-style and code tasks break at budgets where summarization is intact — the same method, same budget, opposite verdicts.
- **Attention sinks are load-bearing.** Keeping the first few tokens plus a sliding window restores streaming stability (Xiao et al., ICLR 2024), at 4M+ tokens on Llama-2 — but the model cannot use evicted content.
- **Retrieval capability is head-localized.** A small set of "retrieval heads" (Wu et al. 2024) accounts for factual recall; masking them destroys needle-in-a-haystack accuracy. DuoAttention exploits this: full cache on ~5–25% of heads, streaming cache elsewhere.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on bytes/token for $\epsilon$-approximate next-token distributions under a *realistic* text distribution. INDEX gives an adversarial $\Omega(mk)$; nobody has shown a matching upper bound for natural language, nor an assumption on $\mathcal{D}_{\text{query}}$ under which $o(n)$ provably suffices for transformer decoding.
- **Theoretically open.** Whether compression error compounds multiplicatively or additively across $L$ layers. No published bound.
- **Empirically open.** The budget–accuracy frontier against a *query-oracle control* at 128K on an 8B model. Runnable today on 2× H100; nobody has published it.
- **Empirically open.** Whether trained-in compression (MLA, DMC) dominates post-hoc compression at equal bytes/token when both are measured at matched pretraining compute. No head-to-head exists.
- **Methodologically blocked.** "Compression quality" has no query-independent definition. Every eviction method is scored on benchmarks whose queries are in the prompt, so a method that predicts the query well scores identically to one that preserves information. The measurement cannot currently separate the two.

## 6. Why It Is Hard

**The obstruction is confounded measurement, not compute.** Prompt-conditional eviction (SnapKV, H2O with a prompt-derived score) reads the query before deciding what to drop. LongBench and most LongBench-v2 tasks put the query in the prompt. So the reported metric conflates *information retention* with *query prediction*, and the benchmark does not measure what its name claims. A compressor that keeps 10% of tokens and scores 99% of full-cache accuracy may be retaining nothing beyond the answer span — which is worthless the moment a second, different question arrives over the same cache, exactly the multi-turn setting deployment cares about.

Secondary: **absent ground truth** for the floor. Without knowing the true information content of a 128K context relative to a given model's parameters, there is no reference against which a bytes/token number is good or bad — the reported ratios are relative to fp16 MHA, an arbitrary and inflated baseline.

## 7. Current Research (as of 2026)

- **Trained-in latent caches.** MLA-style low-rank caches are now the default in several open frontier models; the open question is whether the latent bottleneck costs capability at fixed training compute. *(frontier — verify)*
- **Head-role–aware allocation.** DuoAttention, Ada-KV, PyramidKV: non-uniform budgets across heads and layers. Strong empirically; no theory for why the allocation is optimal.
- **Query-aware sparse attention over a full cache** (Quest, and page-level selection in vLLM/SGLang-style serving stacks). This sidesteps the problem — memory stays $O(n)$, only bandwidth drops. It is the honest engineering answer and is winning in production.
- **Hybrid attention–recurrent stacks** interleaving a few full-attention layers with linear-attention layers, targeting the recall–memory frontier of Arora et al.
- **Benchmarks built to break query-conditional compression**: multi-query and post-hoc-query variants of RULER. Underdeveloped; this is the highest-leverage gap.

## 8. Concrete Next Experiment

**Question.** Does query-agnostic KV compression buy anything beyond quantization, once the query is hidden at compression time?

**Scale.** Llama-3.1-8B-Instruct, 128K context, 2× H100 80GB. 500 documents × 32K–128K tokens, each containing $m \in \{1, 4, 16, 64\}$ planted key–value facts of ~32 bits. After the cache is built and compressed, issue $q = m$ independent retrieval queries against the *same* compressed cache. Total ≈ 3 GPU-days.

**Arms.**
1. **Control (upper bound):** query-oracle compression — the compressor sees all $q$ queries first. Any prompt-conditional method run this way.
2. **Control (baseline):** uniform 4-bit and 2-bit quantization of the *full* cache, byte-matched to each eviction budget.
3. **Treatment:** H2O, SnapKV, PyramidKV, DuoAttention, run strictly before queries are revealed, at $b \in \{64, 32, 16, 8, 4\}$ KB/token.

**Deciding number.** $b^\ast(q)$ — the smallest bytes/token at which the treatment stays within 5 accuracy points of the full-cache arm on all $q$ queries.

- If $b^\ast(q)$ grows **linearly in $q$** while byte-matched quantization is flat, query-agnostic eviction is doing no semantic compression; it is deferring the INDEX bound, and the entire eviction literature's gains are query prediction.
- If $b^\ast(q)$ **saturates** below the quantization baseline, structured eviction retains information independent of the query and the frontier is real.

Single scalar to report: $\partial b^\ast / \partial q$ at $q{=}16$, in KB/token per query.

## 9. Key References

- **[Foundational]** Noam Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- **[Foundational]** Joshua Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[Foundational]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[SOTA]** Zhenyu Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023. — arXiv:2306.14048
- **[SOTA]** Yuhong Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS 2024. — arXiv:2404.14469
- **[SOTA]** Zirui Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. — arXiv:2402.02750
- **[SOTA]** Coleman Hooper et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS 2024. — arXiv:2401.18079
- **[SOTA]** DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024. — arXiv:2405.04434
- **[SOTA]** Piotr Nawrot et al. *Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference.* ICML 2024. — arXiv:2403.09636
- **[SOTA]** Guangxuan Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* 2024. — arXiv:2410.10819
- **[Theory]** Samy Jelassi et al. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Theory]** Simran Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024. — arXiv:2312.04927
- **[Theory]** Clayton Sanford, Daniel Hsu, Matus Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS 2023. — arXiv:2306.02896
- **[Evaluation]** Jiayi Yuan et al. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* Findings of EMNLP 2024. — arXiv:2407.01527
- **[Evaluation]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. — arXiv:2404.06654
- **[Systems]** Woosuk Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. — arXiv:2309.06180
- **[Survey]** Haoyang Li et al. *A Survey on Large Language Model Acceleration based on KV Cache Management.* 2024. — arXiv:2412.19442

## 10. Worked Example

Llama-3.1-8B, one 128K-token document containing $m = 64$ planted facts of 32 bits each.

**Cache accounting.**

```
full fp16 : 131,072 B/tok × 131,072 tok = 16.0 GB
4-bit KV  :  32,768 B/tok               =  4.0 GB   (near-lossless)
2-bit KV  :  16,384 B/tok               =  2.0 GB
H2O @ 20% :  26,214 B/tok effective     =  3.2 GB
H2O @ 10% :  13,107 B/tok effective     =  1.6 GB
```

**Information floor.** The queries can address any of the 64 facts, so a one-pass compressor needs at least $64 \times 32 = 2048$ bits $= 256$ bytes for the *whole* cache — $0.002$ B/token at 128K.

**The gap.** The best-reported operating point (H2O at 10%, 1.6 GB) sits $6.7 \times 10^6$ times above the information floor. That gap is not evidence of a hard limit; it is evidence that no method compresses *content*. Eviction keeps whole 128 KB token-slots or drops them; quantization shrinks every slot uniformly. Neither can express "store these 64 facts."

**Where the obstruction becomes visible.** Run H2O at 10% with the query in the prompt: SnapKV/H2O-style prompt-conditional scoring concentrates the retained budget on the ~200 tokens surrounding the queried fact, and accuracy stays near full-cache — the published result. Now hide the query until after compression and ask all 64. The retained 10% is selected by prompt-time attention mass, which for a uniform-density document is close to uninformative about which 64 spans matter; expected coverage of the 64 fact spans is roughly the retention rate, so ~6 of 64 facts survive intact and accuracy collapses toward $\approx 10\%$. Byte-matched 2-bit quantization, at 2.0 GB, keeps every token and loses only precision — it degrades gracefully across all 64.

The measured quantity that separates them is not perplexity (both fine) and not LongBench (both fine). It is $b^\ast(q)$ from §8. Until that number is published, "10× KV compression" is a claim about benchmark query distributions, not about caches.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*