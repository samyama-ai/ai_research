---
id: 14-long-context/kv-cache-compression-exact-recall-bound
title: "Optimal KV Cache Compression Rate Under Exact-Recall Constraints"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal KV Cache Compression Rate Under Exact-Recall Constraints

> **Topic:** Long Context · **ID:** `14-long-context/kv-cache-compression-exact-recall-bound` · **Status:** open

## 1. Problem Statement

A decoder-only transformer serving a context of $n$ tokens stores a key-value cache whose size grows linearly in $n$. Compression methods (eviction, quantization, low-rank projection, head sparsification) shrink it. The question is how far they can shrink it **without losing exact recall** — the ability to reproduce, verbatim, an arbitrary span of the context on demand.

- **Input:** a fixed pretrained model $f_\theta$, a context $x_{1:n}$, a compression budget $B$ bits.
- **Output:** a compressed cache $\tilde{C}$ with $|\tilde{C}| \le B$ and a decoding rule using $\tilde{C}$ in place of the full cache.
- **Predicate:** for every query in a family $\mathcal{Q}$ that asks to reproduce a span $x_{i:j}$, the decoded output equals $x_{i:j}$ exactly.
- **Solved** means: a tight characterization of the minimal $B^\star(n, \mathcal{Q}, f_\theta)$, plus a method attaining it.

Three variants, of very different difficulty:

- **Theory:** prove a lower bound on $B^\star$ as a function of $n$ and the query family, and an achievability result matching it. Partially settled for *copying*; open for query-conditional recall.
- **Method:** build a compressor that hits the bound on a real model. Open.
- **Measurement:** define "exact recall" so a benchmark number is a bound and not an artifact of the probe distribution. This is where most of the confusion lives.

## 2. Formal Setting

Model with $L$ layers, $H$ KV heads, head dimension $d_h$, elements at $b$ bits. The uncompressed cache after $n$ tokens is

$$ |C_n| \;=\; 2\,L\,H\,d_h\,b\,n \quad \text{bits}. $$

Measured, not estimated: for Llama-3.1-8B ($L=32$, $H=8$ GQA groups, $d_h=128$, fp16) this is $2\cdot32\cdot8\cdot128\cdot16 = 1.05\times10^6$ bits $= 128$ KiB per token, i.e. 16 GiB at $n=131{,}072$.

**Compression rate.** $\rho = B / |C_n| \in (0,1]$. Report $\rho$ at a fixed $n$; "4× compression" without $n$ is not a measurable claim, because eviction methods with a fixed token budget $k$ have $\rho = k/n$ that falls as context grows.

**Exact-recall risk.** Fix a probe distribution $\mathcal{D}$ over (context, query, gold-span) triples. With greedy decoding,

$$ R_{\text{exact}}(\tilde{C}) \;=\; \Pr_{(x,q,y)\sim\mathcal{D}}\big[\,\hat{y}(\tilde C, q) \neq y\,\big], $$

measured as exact string match after whitespace normalization, over $\ge 500$ probes per (context length, budget) cell so a 1% difference is resolvable.

**The constrained problem.**

$$ B^\star(\epsilon) \;=\; \min\{\,B : \exists\,\tilde{C},\ |\tilde{C}|\le B,\ R_{\text{exact}}(\tilde{C}) \le \epsilon \,\}. $$

The interesting regime is $\epsilon \to 0$, where near-misses (right paragraph, wrong number) count as failures, unlike perplexity or ROUGE.

**Information-theoretic floor.** If the query is revealed *after* compression and can address any of $n$ positions, the compressor cannot know which token to keep. Reduction to the one-way communication complexity of `INDEX` gives $B^\star(0) = \Omega(n \log |V|)$ for a query family that can demand any single token — the cache must retain the context's entropy, not the model's summary of it. If the query is known *before* compression, $B^\star(0)$ collapses to $O(|y|\log|V|)$.

**Assumptions, with the ones known to be violated flagged:**

1. *The compressor is query-agnostic.* **Holds** for prefill-time eviction (H2O, SnapKV); **violated** by query-aware selection (Quest, MInference), which moves the problem to a different complexity class — those methods keep the full cache in HBM or on disk and sparsify *reads*, so their $\rho$ is a bandwidth ratio, not a memory ratio. Conflating the two is the single most common error in the literature.
2. *Attention scores predict future importance.* **Violated:** importance is non-stationary across generation steps (Scissorhands' "persistence of importance" is a statistical, not universal, claim).
3. *Errors are independent across probes.* **Violated:** evicting a shared prefix token fails all probes touching it, so naive binomial confidence intervals understate variance.
4. *Greedy decoding.* Adopted for determinism; sampling makes $R_{\text{exact}}$ a property of the sampler too.

## 3. State of the Art

**Theory SOTA.** Jelassi et al., *Repeat After Me: Transformers Are Better than State Space Models at Copying* (ICML 2024), prove that any fixed-state-size sequence model needs state growing with the length of the string it can copy, while a transformer with full cache copies with $O(\log n)$ width — a lower bound for the *copy* query family. Nagle et al., *Fundamental Limits of Prompt Compression: A Rate-Distortion Framework* (NeurIPS 2024), give the rate-distortion optimum for token-level prompt compression via linear programming and show existing prompt compressors sit far from it. Neither yields $B^\star(\epsilon)$ for query-conditional recall on a fixed pretrained model. **Established.**

**Systems/empirical SOTA.**
- **StreamingLLM** (Xiao et al., ICLR 2024): 4 attention-sink tokens + local window streams to 4M tokens at constant memory. Established that sinks are load-bearing; also established that it *cannot* recall outside the window — the authors say so.
- **H2O** (Zhang et al., NeurIPS 2023): accumulated-attention eviction, 20% budget, perplexity preserved. **Established** for LM loss; the retrieval claim is weaker.
- **SnapKV** (Li et al., NeurIPS 2024): observation-window attention pooling; 1024-token budget on 128k contexts with high needle-retrieval scores.
- **Quest** (Tang et al., ICML 2024) and **MInference** (Jiang et al., NeurIPS 2024): query-aware page/pattern selection. Reported as bandwidth reduction with full cache resident — **not** a memory-compression result, though widely cited as one.
- **KIVI** (Liu et al., ICML 2024): 2-bit asymmetric quantization, per-channel keys, per-token values; ~2.6× memory reduction at near-baseline quality. Quantization is the only family with a clean $\rho$ that does not depend on $n$.
- **DuoAttention** (Xiao et al., ICLR 2025): splits heads into retrieval vs. streaming, full cache only for retrieval heads.

**Claimed but unablated:** near-lossless claims at $\rho \le 0.1$ generally rest on needle-in-a-haystack, whose gold answer is a single injected sentence dissimilar from the distractor context — an easy case for attention-score eviction. Yuan et al., *KV Cache Compression, but What Must We Give in Return?* (EMNLP Findings 2024) is the main ablation showing these gains do not transfer to multi-step retrieval and arithmetic-over-context tasks. **Benchmark-number-only:** most reported $\rho$–quality curves come from one model family at one context length.

## 4. What Is Known

- Attention is empirically sparse: >95% of attention mass on a small token subset for most heads at 4k–32k contexts (H2O, NeurIPS 2023, OPT/Llama-7B–30B).
- A small number of **retrieval heads** (~3–6% of heads) drive copy behavior; masking them collapses needle retrieval while perplexity barely moves (Wu et al., *Retrieval Head Mechanistically Explains Long-Context Factuality*, 2024, arXiv:2404.15574; measured on Llama-2/Mistral-scale models).
- Recall degrades with state size in a smooth, measurable way: Arora et al., *Zoology* (ICLR 2024) map a recall–memory Pareto frontier for gated-convolution and attention hybrids at 355M scale, and show recall gaps persist at matched perplexity.
- **Perplexity is not a proxy.** RULER (Hsieh et al., 2024, arXiv:2404.06654) shows models with claimed 32k+ contexts fall below their 4k-baseline accuracy well before the claimed length, at full cache.
- 2-bit KV quantization is近 lossless on generation-quality benchmarks (KIVI, ICML 2024, Llama-2-7B/13B, Falcon-7B) — a real $\rho \approx 0.125$ on the value path with no $n$ dependence.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $B^\star(\epsilon)$ for $\epsilon>0$ under a realistic probe distribution. The `INDEX` argument gives $\Omega(n)$ for worst-case queries; natural-language contexts are compressible, and nobody has proved the rate for a context distribution with entropy rate $h$ bits/token — the plausible conjecture $B^\star(0) \approx h\,n$ (independent of $L,H,d_h$) is unproven and would imply current caches are ~$10^5\times$ redundant.
- **Empirically open.** The $\rho$–$R_{\text{exact}}$ curve at $n \ge 128$k, on $\ge 70$B models, with multi-span and adversarially-similar distractor probes. Runnable today; the compute is a few thousand GPU-hours, not a research program.
- **Methodologically blocked.** "Exact recall" has no agreed probe distribution. Needle-in-a-haystack and RULER measure different things and neither bounds worst-case recall; there is no accepted way to certify a compressed cache rather than sample-test it.

## 6. Why It Is Hard

**Confounded measurement plus absent worst case.** Every published $\rho$ is a *sample* estimate against a probe distribution the compressor was tuned on, while the quantity of interest is a worst case over queries the compressor never saw. Eviction is an irreversible decision made before the query exists, so any benchmark whose queries are predictable from the context (all current ones — the needle is lexically distinct, the RULER key is a UUID) systematically over-reports.

Second obstruction: **non-identifiability of the budget**. Query-aware sparse-attention methods and eviction methods report the same headline number ("16× compression") for different physical quantities (bytes read per step vs. bytes resident). Without separating them, the field's $\rho$–quality frontier is not a frontier over one axis.

## 7. Current Research (as of 2026)

- Head-differentiated budgets: DuoAttention (MIT Han Lab), Ada-KV, PyramidKV — allocate cache per head/layer rather than uniformly.
- Hybrid quantize-then-evict stacks; the interaction term is largely unmeasured *(frontier — verify)*.
- Offloaded exact caches with learned retrieval over cache pages, treating recall as an ANN-search problem rather than a compression problem.
- Rate-distortion formalization extended from prompt text to cache tensors, following Nagle et al. *(frontier — verify)*.
- Benchmark work moving toward shared-context, multi-turn settings (SCBench, Li et al., ICLR 2025), which stresses reuse of a compressed cache across queries — the setting where query-agnostic eviction is worst.

## 8. Concrete Next Experiment

**Question:** does exact-recall risk have a sharp threshold in $\rho$, and is that threshold set by context entropy or by cache geometry?

- **Scale:** Llama-3.1-8B and Llama-3.1-70B, contexts $n \in \{8\text{k}, 32\text{k}, 128\text{k}\}$, budgets $\rho \in \{1, \tfrac12, \tfrac14, \tfrac18, \tfrac1{16}, \tfrac1{32}, \tfrac1{64}\}$.
- **Probes:** 1,000 per cell, each asking for verbatim reproduction of a 32-token span drawn uniformly from the context, with distractors constructed by resampling near-duplicate spans (edit distance $\le 8$) so lexical salience cannot be exploited. Report exact match.
- **Arms:** (a) SnapKV eviction; (b) KIVI-style quantization at matched $\rho$; (c) **control — random uniform token eviction at matched $\rho$**; (d) **oracle control — query-aware eviction that keeps the gold span**, which upper-bounds any compressor.
- **Deciding number:** $\rho_{1\%}$, the largest compression at which $R_{\text{exact}} \le 0.01$, for each arm at $n=128$k. If SnapKV's $\rho_{1\%}$ is within $2\times$ of random eviction's, attention-score eviction contributes almost nothing to exact recall and the published gains are probe artifacts. If the oracle arm reaches $\rho_{1\%} \le 1/64$ while all query-agnostic arms stall near $1/2$, the gap is information-theoretic, not algorithmic, and effort should move to offload-and-retrieve.

Cost estimate: ~$7\times10^4$ probe generations, dominated by 70B prefill at 128k; order 2,000 A100-hours.

## 9. Key References

- **[Foundational]** Zhang, Sheng, Zhou, Chen, et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS, 2023. — arXiv:2306.14048
- **[Foundational]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Theory]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Theory]** Nagle, Girish, Bondaschi, Gastpar, Makkuva, Jaggi. *Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models.* NeurIPS, 2024. — arXiv:2407.15504
- **[SOTA]** Li, Huang, Yang, Shrivastava, et al. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS, 2024. — arXiv:2404.14469
- **[SOTA]** Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML, 2024. — arXiv:2402.02750
- **[SOTA]** Tang, Zhao, Zhu, Cai, Wang, Han. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML, 2024. — arXiv:2406.10774
- **[SOTA]** Xiao, Tang, Zhu, et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* ICLR, 2025. — arXiv:2410.10819
- **[Evaluation]** Hsieh, Sun, Kriman, et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Evaluation]** Yuan, Liu, Zhong, et al. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* Findings of EMNLP, 2024. — arXiv:2407.01527
- **[Mechanism]** Wu, Wang, Xiao, Wang, Yan, Han, Xie. *Retrieval Head Mechanistically Explains Long-Context Factuality.* 2024. — arXiv:2404.15574
- **[Survey]** Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927

## 10. Worked Example

Llama-3.1-8B, $n = 32{,}768$ tokens of English prose. Full cache: $32768 \times 128$ KiB $= 4.0$ GiB.

Information floor: English at ~1.0 bit/token of *irreducible* entropy under a strong LM is optimistic; take a conservative 2 bits/token. Storing the context losslessly costs $32768 \times 2 = 65{,}536$ bits $= 8$ KiB. Against 4.0 GiB, that is

$$ \rho_{\text{floor}} \approx \frac{8\ \text{KiB}}{4.0\ \text{GiB}} \approx 1.9\times10^{-6}. $$

Even raw UTF-8 text is ~130 KiB, $\rho \approx 3\times10^{-5}$. Published "aggressive" compression is $\rho = 1/64 \approx 1.6\times10^{-2}$ — **three orders of magnitude above the text itself**.

Now the obstruction. Take SnapKV with a 512-token budget ($\rho = 1/64$). Ask for the 32-token span starting at position 14{,}207. If the observation window's attention did not rank those tokens in the top 512, they are gone; $R_{\text{exact}} = 1$ for that probe, with no partial credit and no recovery. Averaged over uniformly drawn spans, expected recall is bounded above by the fraction of the context retained, ~1.6%, unless the query was visible at compression time.

Yet the same configuration scores >95% on needle-in-a-haystack. The gap is entirely the probe distribution: the needle is a lexical outlier that attention scores select for. So a method sitting $1000\times$ above the entropy floor reports near-lossless recall, and the benchmark cannot distinguish that from a method sitting $10\times$ above it. That is the obstruction — not that compression is hard, but that the number reported as "recall preserved" is measuring salience, not recall.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*