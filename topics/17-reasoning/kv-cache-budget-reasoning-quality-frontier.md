---
id: 17-reasoning/kv-cache-budget-reasoning-quality-frontier
title: "KV-Cache Budget Versus Reasoning Quality Frontier"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# KV-Cache Budget Versus Reasoning Quality Frontier

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/kv-cache-budget-reasoning-quality-frontier` · **Status:** empirically-open

## 1. Problem Statement

A reasoning model spends its test-time compute emitting a long chain of thought. The dominant memory cost of that chain is the KV cache, which grows linearly in generated tokens. The question: **for a fixed per-sequence memory budget, what is the highest achievable reasoning accuracy, and which allocation of that budget attains it?**

Three variants, usually conflated:

- **Measurement.** Define the frontier $Q^*(B)$ — best accuracy at KV budget $B$ bytes — so that it is comparable across eviction, quantization, architectural sharing, and simply generating fewer tokens. Today's papers do not measure a frontier; they measure one policy against an unconstrained baseline.
- **Method.** Find a policy that attains $Q^*(B)$ at small $B$. Solving it means: given $B$, output a decode procedure whose accuracy dominates every iso-memory alternative, including the trivial one of truncating the CoT.
- **Theory.** Characterize the class of reasoning tasks solvable with $o(n)$ cache state over $n$ CoT tokens. Solving it means a separation theorem: a task family provably requiring $\Omega(n)$ KV bytes, and a nontrivial family requiring $O(\mathrm{polylog}\,n)$.

## 2. Formal Setting

Model $f_\theta$, $L$ layers, $H_{kv}$ key/value heads, head dim $d_h$, precision $b$ bits. Uncompressed cache after $n$ tokens:

$$M(n) = n \cdot L \cdot H_{kv} \cdot d_h \cdot 2 \cdot \frac{b}{8} \ \text{bytes}.$$

**Budget** $B$ is *measured* as peak resident KV bytes for one sequence during decode — allocator pages included (PagedAttention block granularity), auxiliary state (importance scores, quantization scales, retained sink tokens) included. Reporting "keep 20% of tokens" is not a budget; scale factors and index structures are real bytes.

A **policy** $\pi$ is a map from history to a cache state $S_t$ with $|S_t| \le B$ for all $t$, plus a decode rule. It covers eviction, quantization, low-rank/latent compression, head sharing, offload-with-recompute, and CoT-length control as special cases.

**Quality.** For task distribution $\mathcal{D}$ and verifier $v$,

$$Q(\pi, B) = \mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim \pi(\cdot\mid x)}\big[v(x,y)\big],$$

estimated as pass@1 over $k \ge 16$ seeds at fixed temperature, with a stated $\pm$ from the seed variance. The **frontier** is $Q^*(B) = \sup_{\pi:\,|S|\le B} Q(\pi,B)$; the object of interest is its shape — where it is flat, where it has a knee, and where its slope $\partial Q/\partial \log B$ becomes steep.

The deployment-relevant constraint is joint: with device memory $M_{\text{dev}}$ and weights $W$, concurrency is $C = \lfloor (M_{\text{dev}}-W)/B \rfloor$, so tokens/second scales with $C$ and the decision is over $(B, n, C)$ jointly, not $B$ alone.

**Assumptions, and which fail.**
1. *$Q$ is monotone nondecreasing in $B$.* Violated: compression sometimes truncates degenerate self-doubt loops and raises accuracy at moderate $B$.
2. *Attention scores are a sufficient statistic for future utility* (the H2O/SnapKV premise). Violated at reasoning phase boundaries — a token unattended during derivation becomes critical when the model backtracks and verifies.
3. *Eviction decisions are separable across layers/heads.* Violated; retrieval heads are sparse and non-uniformly distributed (DuoAttention, MInference).
4. *Prefill-time importance transfers to decode.* Holds for long-input summarization, fails for short-input/long-output reasoning, which is the regime here.

## 3. State of the Art

**Systems/empirical SOTA.** *Established and independently reproduced:* attention-sink retention plus recent-window (StreamingLLM, Xiao et al., ICLR 2024) keeps perplexity stable to millions of tokens; heavy-hitter eviction (H2O, Zhang et al., NeurIPS 2023; Scissorhands, Liu et al., NeurIPS 2023) and prompt-side selection (SnapKV, Li et al., NeurIPS 2024) preserve long-input task scores at ~20% cache; 2-bit per-channel/per-token quantization (KIVI, Liu et al., ICML 2024) is near-lossless on perplexity. Architectural budget reduction is the only fully solved part: GQA (Ainslie et al., EMNLP 2023) and MLA (DeepSeek-V2, 2024, reported ~93% KV reduction vs. its MHA counterpart) buy large constant factors with retraining.

*Claimed but unablated for reasoning:* that these transfer to long-CoT decoding. Reasoning-specific methods (e.g. R-KV, 2025, reporting high retention of math accuracy at ~10–34% cache on R1-distilled models) exist mostly as benchmark tables against a full-cache arm, with no iso-memory control and few seeds — on AIME-scale sets (30 items) the seed noise is comparable to the reported gaps.

*Benchmark-number-only:* nearly all headline "X% cache, no quality loss" claims. SCBench (Li et al., ICLR 2025) and RULER (Hsieh et al., COLM 2024) show the loss appears once you evaluate multi-turn reuse and true retrieval rather than perplexity.

**Theory SOTA.** Merrill & Sabharwal (ICLR 2024) tie CoT length to expressive power (log steps → $\mathrm{L}$-ish, poly steps → $\mathrm{P}$) — but assume full state access. Jelassi et al., *Repeat After Me* (ICML 2024) give $\Omega(n)$ memory lower bounds for copying, separating fixed-state models from attention. No result bridges these into a lower bound on *compressed* KV bytes for a reasoning task family.

## 4. What Is Known

- Cache size, not FLOPs, binds test-time scaling at deployment concurrency (Kinetics, Sadhukhan et al., 2025): under memory-bandwidth accounting the optimal $(B,n,C)$ point differs sharply from the FLOP-optimal one.
- Retention of ~20% of KV entries preserves LongBench-style scores within ~1 point for 7B–13B models on long *input* tasks (H2O, SnapKV, PyramidKV, 2023–2024).
- The same policies degrade sharply on RULER-style exact retrieval: of ten models advertising $\ge$32K context, RULER (2024) found roughly half fail to hold performance at their claimed length even *uncompressed*.
- Query-aware sparse attention gives ~7× self-attention speedup at 4K-token budgets on 7B models with negligible perplexity change (Quest, Tang et al., ICML 2024).
- Retrieval capability is concentrated: a minority of heads need full cache; the rest tolerate a streaming window (DuoAttention, Xiao et al., ICLR 2025), measured on 7B–8B Llama-class models.
- Not known at any scale: a single published curve of pass@1 versus *bytes* with an iso-memory shorter-CoT control on a competition-math or code benchmark.

## 5. What Is Not Known

- **Empirically open.** The frontier itself. Runnable today on 8×H100 for a 32B reasoning model across five budgets, six policies, three benchmarks, 16 seeds — roughly $10^4$ GPU-hours. Nobody has published it. In particular: whether *any* compression policy beats simply generating a shorter CoT at equal bytes.
- **Empirically open.** Whether compression error compounds along a CoT. Single-step logit KL is small; the effect on a 20K-token derivation with one arithmetic slip is unmeasured.
- **Methodologically blocked.** Comparability. "20% cache" is not bytes; policies with different auxiliary state, page granularity, and precision are routinely plotted on the same axis. Until budgets are reported in bytes/token including overheads, the frontier cannot be assembled from existing papers.
- **Theoretically open.** No proof either way that some reasoning task family requires $\Omega(n)$ KV bytes under an arbitrary adaptive compression policy. The copying lower bounds do not apply, because a compression policy is allowed unbounded compute over the retained state.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an absent control arm**. Every published comparison is *compressed-$n$-tokens vs. full-cache-$n$-tokens*. The decision a deployer actually faces is *compressed-$n$ vs. full-cache-$n'$* where $n' \approx n \cdot B/M(n)$ — the same memory spent on a shorter chain, or on more parallel samples. Because reasoning accuracy rises steeply with CoT length and with sample count, the iso-memory alternative is strong, and no reported number rules it out.

Second obstruction: **statistical power against compute cost**. Frontier knees are 1–3 point effects; AIME-sized sets need $k\ge16$ seeds for $\pm$2-point CIs, and each seed is a 20K-token generation. Power and cost push in opposite directions, which is why papers report one seed on 30 problems.

## 7. Current Research (as of 2026)

- Reasoning-aware eviction that scores redundancy rather than attention mass, targeting repeated verification passes in R1-style traces *(frontier — verify)*.
- Latent/low-rank KV (MLA lineage; DeepSeek, and cross-layer sharing variants) as the retrain-once alternative to inference-time eviction.
- Memory-aware test-time scaling laws following Kinetics (CMU/Infini-AI lineage), optimizing $(B,n,C)$ jointly *(frontier — verify)*.
- Serving-side budget enforcement: paged, quantized, offloaded caches with SLO-aware admission (vLLM/SGLang ecosystems).
- Benchmarks that stress decode-side rather than prefill-side compression; SCBench is the closest existing instrument, still input-heavy.

## 8. Concrete Next Experiment

**Scale.** Two models: DeepSeek-R1-Distill-Qwen-7B and -32B. Three benchmarks: AIME 2024+2025 (60 items), GPQA-Diamond (198), LiveCodeBench-v5 subset (200). $k=16$ seeds, temperature 0.6, max 32K thinking tokens.

**Budget grid.** $B/M \in \{1, 0.5, 0.25, 0.125, 0.0625\}$, measured in bytes/token including all auxiliary state.

**Arms.** Compression: H2O, SnapKV, PyramidKV, StreamingLLM-window, KIVI-2bit, DuoAttention. Random-eviction floor (same byte count). **Control arm (the point of the experiment):** full-precision full cache with the token limit cut to $n' = 32\text{K}\cdot B/M$, and a second control spending the same bytes on $C$ parallel shorter samples with majority vote.

**Deciding number.** The **crossover budget** $B_\times$: the largest $B$ at which the best compression arm's pass@1 fails to exceed the best iso-memory control by more than the 95% CI half-width. If $B_\times \ge 0.25\,M$ — compression loses at every budget a deployer would use, and the field's premise for reasoning workloads is wrong. If $B_\times \le 0.0625\,M$, compression is a genuine frontier-mover and the curve's knee locates the right operating point.

## 9. Worked Example

Qwen2.5-32B geometry: $L=64$, $H_{kv}=8$, $d_h=128$, fp16.

$$M/\text{token} = 64\cdot 8\cdot 128\cdot 2\cdot 2 = 262{,}144\ \text{B} = 256\ \text{KiB}.$$

A 32K-token chain of thought holds $32768 \times 256\,\text{KiB} = 8\ \text{GiB}$ of KV for **one** sequence. On an 80 GB H100 with ~64 GB of weights (32B in fp16), a single sequence at full length nearly exhausts the remaining 16 GB: concurrency $C=2$.

Now take the standard claim, "25% cache, no quality loss." That is 64 KiB/token, 2 GiB per sequence, $C=8$. The paper's comparison is *2 GiB compressed 32K-token chain* vs. *8 GiB full 32K-token chain*, and reports a 0.5-point drop on AIME — one seed, 30 problems, where one item is 3.3 points.

The control nobody ran: 2 GiB also buys a **full-precision 8K-token chain**, or four 8K chains with majority vote. On R1-distilled 32B, published length-ablation curves put 8K-token AIME pass@1 several points below 32K — but majority-vote-over-4 typically recovers more than that. The two effects are the same size as the claimed compression loss, and they point in opposite directions. So the reported 0.5-point drop decides nothing: the obstruction is not that compression is lossy, it is that the loss was never compared against what the same bytes buy otherwise.

## 10. Key References

- **[Foundational]** Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023.
- **[Foundational]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024.
- **[Foundational]** Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023.
- **[SOTA]** Li et al. *SnapKV: LLM Knows What You Are Looking for Before Generation.* NeurIPS 2024.
- **[SOTA]** Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
- **[SOTA]** Tang et al. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024.
- **[SOTA]** Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* ICLR 2025.
- **[SOTA]** DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* Technical report, 2024.
- **[Theory]** Merrill, Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR 2024.
- **[Theory]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024.
- **[Benchmark]** Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024.
- **[Benchmark]** Li et al. *SCBench: A KV Cache-Centric Analysis of Long-Context Methods.* ICLR 2025.
- **[Systems]** Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
- **[Frontier]** Sadhukhan et al. *Kinetics: Rethinking Test-Time Scaling Laws.* Preprint, 2025.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*