---
id: 11-inference-and-serving/kv-compression-task-independent
title: "KV Cache Compression Without Task-Dependent Degradation"
topic: 11-inference-and-serving
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# KV Cache Compression Without Task-Dependent Degradation

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/kv-compression-task-independent` · **Status:** methodologically-blocked

## 1. Problem Statement

A decoder-only transformer serving a prompt of $n$ tokens stores $2 n L H d_h$ scalars of key/value state. KV cache compression replaces that state with a smaller object and decodes from it. The problem is not "compress the cache" — many methods do that. The problem is:

**Find a compression operator whose quality loss is bounded uniformly over the downstream task, not on average over a benchmark.**

- **Input:** a frozen model $f_\theta$, a prompt $x_{1:n}$, a memory budget $B < 2nLHd_h$.
- **Output:** a compressed state $\tilde C$ of size $\le B$ plus a decode rule, producing $\tilde y$.
- **Decision predicate:** does there exist a compressor such that for *every* task distribution $\mathcal{T}$ in a stated admissible class, the quality drop is $\le \epsilon$?

Three variants, with sharply different difficulty:

- **Measurement variant (the blocked one).** Define "task-dependent degradation" so it can be estimated from finite evaluations. Current benchmark averages hide the failure mode: a method at 4× compression can be lossless on summarization and catastrophic on multi-hop retrieval, and the reported mean is "1.2 points down".
- **Method variant (empirically open).** Build a compressor that is simultaneously good on retrieval, aggregation, in-context learning, multi-turn reuse, and reasoning-with-long-CoT. No method has been shown to be.
- **Theory variant (partially settled, negatively).** Query-independent eviction cannot be uniformly safe: the future query is not a function of the past. What is open is the *achievable* rate — the smallest $B$ admitting $\epsilon$-uniform quality under realistic (not adversarial) query distributions.

Solving it means: a compressor plus a certificate — an a-priori bound, computable before the query arrives, on the output deviation it can induce.

## 2. Formal Setting

**Objects.** Layer $\ell$, head $h$. Cache $C = \{(k_i, v_i)\}_{i=1}^n$, $k_i, v_i \in \mathbb{R}^{d_h}$. At decode step $t$ with query $q_t$, exact attention output is

$$o_t = \sum_{i=1}^{n} \alpha_i v_i, \qquad \alpha_i = \frac{\exp(q_t^\top k_i / \sqrt{d_h})}{\sum_{j\le n} \exp(q_t^\top k_j/\sqrt{d_h})}.$$

A compressor is a map $\mathcal{A}: C \mapsto \tilde C$, $|\tilde C| \le B$, chosen **before** $q_t$ is observed (prompt-time eviction/quantization) or **jointly with** $q_t$ (query-aware selection, e.g. Quest). This distinction is the whole problem.

**Compression ratio** $\rho = |\tilde C|_{\text{bytes}} / |C|_{\text{bytes}}$, measured as peak allocator bytes for the cache tensors, not as retained token count — quantization methods carry per-group scales and zero-points that a token count hides (KIVI: group size 32, fp16 scale + zero → ~1 extra byte per 16 values).

**Per-step attention error**, the quantity actually measurable without any task:
$$\delta_t = \big\| o_t - \tilde o_t \big\|_2 \big/ \big\| o_t \big\|_2 .$$

**Task degradation**, the quantity we care about:
$$\Delta(\mathcal{T}, \rho) = \mathbb{E}_{(x,y)\sim\mathcal{T}}\big[ s(y, f_\theta(x)) - s(y, f_\theta^{\mathcal{A},\rho}(x)) \big],$$
with $s$ the task score (EM, F1, pass@1). The object of interest is the **worst-case over tasks**:
$$\Delta^\star(\rho) = \sup_{\mathcal{T}\in\mathfrak{T}} \Delta(\mathcal{T},\rho),$$
and the target is $\Delta^\star(\rho) \le \epsilon$ — not $\mathbb{E}_{\mathcal{T}}[\Delta]\le\epsilon$, which is what every published table reports.

**Assumptions, and which are violated.**
1. *Attention scores observed during prefill predict future attention* — the H2O/Scissorhands premise. **Violated:** Ren & Zhu (2024) show persistence of "heavy hitters" decays with distance and with query shift; a question appended after eviction re-ranks the keys.
2. *$\delta_t$ small $\Rightarrow$ $\Delta$ small.* **Violated:** the map from attention output to token argmax is not Lipschitz at decision boundaries; a single evicted key can flip one token of a 40-digit needle and zero the EM score while $\delta_t \approx 10^{-2}$.
3. *Compression composes across turns.* **Violated:** SCBench shows methods evaluated single-turn degrade further when the compressed cache is *reused* across turns; errors accumulate.
4. *Budget is uniform across heads/layers.* **Violated:** DuoAttention, PyramidKV, Ada-KV all report strongly non-uniform per-head sensitivity.

## 3. State of the Art

**Empirical/systems SOTA.**
- *Eviction:* H2O (Zhang et al., NeurIPS 2023), Scissorhands (Liu et al., NeurIPS 2023), SnapKV (Li et al., NeurIPS 2024), PyramidKV (Cai et al., 2024), Ada-KV (Feng et al., 2024). **Established:** large throughput/memory wins at 10–20% retained budget on LongBench-style suites. **Claimed but unablated:** that these hold when the query is unknown at prefill. SnapKV and Quest explicitly use the observation window / query, so their LongBench numbers do not transfer to prefix-caching deployments where the prompt is compressed before the question exists.
- *Quantization:* KIVI (Liu et al., ICML 2024) — per-channel keys, per-token values, 2-bit; KVQuant (Hooper et al., NeurIPS 2024) — pre-RoPE key quantization + outlier isolation, 3-bit and below. Quantization is the more robust family: it degrades all positions slightly rather than deleting some entirely.
- *Structural:* StreamingLLM (Xiao et al., ICLR 2024) — attention sinks + local window; DuoAttention (Xiao et al., ICLR 2025) — split heads into retrieval vs streaming, full cache only for retrieval heads.
- *Query-aware sparsity:* Quest (Tang et al., ICML 2024) — keeps full cache, selects pages per query. This sidesteps the problem rather than solving it: memory is still $O(n)$.

**Theory SOTA.** Sublinear-memory attention approximation exists under distributional assumptions — SubGen (Zandieh et al., 2024) gives $\tilde O(1)$-per-token memory with $\epsilon$-additive attention error when key embeddings are clusterable; HyperAttention (Han et al., ICLR 2024) gives near-linear-time attention under bounded stable rank. Neither yields a task-level bound, and both assume conditions that are checkable only after the fact. Hardness results for subquadratic attention alternatives (Alman & Yu, 2025) indicate no uniformly accurate subquadratic surrogate exists under fine-grained complexity assumptions.

**Benchmark-number-only results.** Nearly all "lossless at 4×" claims. They are single-turn, single-benchmark, and use the query at compression time.

## 4. What Is Known

- **Attention sinks are real and cheap.** Keeping 4 initial tokens plus a local window restores perplexity for streams to 4M tokens (Llama-2-7B/13B, MPT-7B; Xiao et al., ICLR 2024). It does **not** extend usable context — retrieval beyond the window is lost by construction.
- **Compression families are not interchangeable.** Yuan et al. (EMNLP Findings 2024), benchmarking across seven long-context capabilities on 7B–13B models, found quantization broadly preserves capability at 4× while eviction collapses on retrieval and multi-hop reasoning at the same ratio.
- **Reuse breaks eviction.** SCBench (Li et al., ICLR 2025), evaluating 12 methods across 8B–70B models in multi-turn shared-context settings, found $O(n)$-memory-with-sparse-decode methods robust, while sub-$O(n)$ eviction degrades progressively across turns.
- **Sensitivity is concentrated.** DuoAttention reports roughly a quarter of heads are retrieval heads carrying long-range dependence, with ~2.5× MHA memory reduction when only those keep full cache; Wu et al. (ICLR 2025) independently identify a sparse, stable set of retrieval heads that mechanistically account for long-context factuality.
- **Benchmark context length overstates effective length.** RULER (Hsieh et al., COLM 2024) shows models advertising 32K contexts hold quality to far shorter effective lengths — so an uncompressed baseline is itself already degrading, confounding compression measurements.
- **2-bit KV is practical.** KIVI reports ~2.6× peak-memory reduction and 2.3–3.5× throughput at negligible loss on standard suites at 7B scale.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition or estimator of $\Delta^\star(\rho)$. "Admissible task class" $\mathfrak{T}$ is undefined; every benchmark is a finite, biased sample, and worst-case over an unspecified class is not estimable. No published method reports a *worst-task* number.
- **Methodologically blocked (secondary).** No validated proxy linking $\delta_t$ to $\Delta$. Papers report perplexity or benchmark score, never a calibrated curve $\Delta = g(\delta)$.
- **Empirically open.** Whether *any* query-independent compressor is lossless at $\rho = 0.25$ across retrieval + aggregation + ICL + multi-turn simultaneously, at $\ge$70B scale with a strong uncompressed control. The experiment is runnable; the matrix has not been run.
- **Theoretically open.** The achievable rate–distortion frontier for KV state under a realistic query prior. A communication-complexity argument (reduction from INDEX: an adversarial query asks for one of $n$ positions) forces $\Omega(n)$ bits for exact recovery, but the constant and the behaviour under non-adversarial query priors are unproven.

## 6. Why It Is Hard

**The evaluation does not measure what it names.** "Lossless compression" is reported as a mean over a benchmark whose task mix determines the answer. Eviction is near-free on tasks where the answer depends on a diffuse summary and catastrophic where it depends on one token; benchmark means average these into a number that predicts neither.

Compounding:
- **Non-identifiability of the query.** Prompt-time compression must commit before knowing what is asked. Any fixed compressor has an adversarial query; the only defence is a prior over queries that nobody has specified.
- **Absent ground truth for "importance".** There is no oracle labelling which of $n$ KV entries the eventual answer needs, so eviction policies are validated only by their own downstream scores.
- **Confounded baseline.** RULER shows the uncompressed model already fails at long context; a compressed model matching it may be matching a failure.
- **Compute.** Filling task $\times$ ratio $\times$ method $\times$ scale at 70B with 128K contexts is thousands of GPU-hours, so the worst-case cell is the one routinely skipped.

## 7. Current Research (as of 2026)

- **Head-differentiated budgets.** DuoAttention, Ada-KV, PyramidKV; MIT Han Lab and collaborators. Direction: allocate by measured head function rather than uniformly. *(frontier — verify)* extension to learned, prompt-conditional budget allocators.
- **Quantization over eviction.** KIVI/KVQuant lineage (Berkeley BAIR, Rice); low-bit with outlier handling as the safer default.
- **Reuse-aware evaluation.** SCBench (Microsoft) reframes the metric around multi-turn shared prefixes — the setting real serving stacks (vLLM prefix caching, SGLang RadixAttention) actually run.
- **Certified compression.** *(frontier — verify)* per-query error bounds computed from retained-key norms, so a server can refuse to compress when the bound is loose. No production system does this.
- **Learned latent caches.** MLA-style architectural compression (DeepSeek-V2/V3) moves the problem into pretraining, avoiding post-hoc eviction entirely — arguably the strongest practical answer, at the cost of requiring a new model.

## 8. Concrete Next Experiment

**Question:** what is the worst-task degradation, not the mean, at fixed $\rho$?

- **Scale:** one 8B and one 70B open model, 128K context. Five methods: H2O, SnapKV, PyramidKV, KIVI-2bit, DuoAttention. Three ratios $\rho \in \{0.5, 0.25, 0.125\}$.
- **Task grid (12 cells):** RULER NIAH single/multi-key/multi-value, variable tracking, common-word extraction; LongBench multi-hop QA; summarization; many-shot ICL; code completion; two SCBench multi-turn reuse tasks.
- **Critical protocol constraint:** compression must run **before** the question is appended. Prefix-only compression is the deployment reality and the condition under which query-aware methods lose their advantage.
- **Control arm:** the *same* model, same prompts, uncompressed, at the same context length — establishing the RULER-style baseline ceiling. Second control: random eviction at matched $\rho$, which no paper reports and which bounds how much of the gain is policy versus slack.
- **Deciding number:** $\Delta^{\max}(\rho=0.25) = \max_{\text{cell}} \big(\text{score}_{\text{control}} - \text{score}_{\text{compressed}}\big)$, per method, in absolute score points. Publish the max, not the mean. A method with $\Delta^{\max} \le 2$ points at $\rho=0.25$ under prefix-only compression would be the first evidence of task-independent compression. Current expectation, from Yuan et al. and SCBench: eviction methods land at $\Delta^{\max} > 30$ points on multi-key NIAH while their means read $< 3$.

## 9. Key References

- **[Foundational]** Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett, Wang, Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS, 2023. — arXiv:2306.14048
- **[Foundational]** Liu, Desai, Liao, Wang, Xie, Wang, Zhao, Shrivastava. *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* NeurIPS, 2023. — arXiv:2305.17118
- **[Foundational]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[SOTA]** Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML, 2024. — arXiv:2402.02750
- **[SOTA]** Hooper, Kim, Mohammadzadeh, Mahoney, Shao, Keutzer, Gholami. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS, 2024. — arXiv:2401.18079
- **[SOTA]** Li, Huang, Yang, Bai, Hu, Zhao, Han, Yang. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS, 2024. — arXiv:2404.14469
- **[SOTA]** Xiao, Tang, Zhu, Zhang, Han, Chen, Han. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* ICLR, 2025. — arXiv:2410.10819
- **[SOTA]** Tang, Zhao, Zhu, Cai, Wang, Han. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML, 2024. — arXiv:2406.10774
- **[Evaluation]** Yuan, Liu, Zhang, Zhao, et al. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* Findings of EMNLP, 2024. — arXiv:2407.01527
- **[Evaluation]** Li, Jiang, Zhang, Han, Chen, Abdi, Qiu, et al. *SCBench: A KV Cache-Centric Analysis of Long-Context Methods.* ICLR, 2025. — arXiv:2412.10319
- **[Evaluation]** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Analysis]** Ren, Zhu. *On the Efficacy of Eviction Policy for Key-Value Constrained Generative Language Model Inference.* 2024. — arXiv:2402.06262
- **[Analysis]** Wu, Wang, Xiao, Wang, Yan, Fu, Xiao. *Retrieval Head Mechanistically Explains Long-Context Factuality.* ICLR, 2025. — arXiv:2404.15574
- **[Theory]** Zandieh, Han, Mirrokni, Karbasi. *SubGen: Token Generation in Sublinear Time and Memory.* 2024. — arXiv:2402.06082
- **[Theory]** Alman, Yu. *Fundamental Limitations on Subquadratic Alternatives to Transformers.* 2025.

## 10. Worked Example

**Setup.** Llama-3-8B-Instruct, 32K-token prompt, 32 layers, 8 KV heads (GQA), $d_h=128$, fp16.

Cache size:
$$2 \times 32768 \times 32 \times 8 \times 128 \times 2\ \text{B} = 4.29\ \text{GB}.$$
At $\rho=0.125$ (H2O, 12.5% retained): 0.54 GB. On an 80 GB H100 with ~16 GB weights, concurrency goes from ~14 to ~110 sequences. The systems win is real and large. Now the obstruction.

**Two tasks, same cache, same budget.**

*Task A — summarize the 32K document.* The answer depends on a diffuse average over positions. Evicting 87.5% of entries chosen by accumulated attention mass shifts $o_t$ by roughly $\delta_t \approx 0.03$; ROUGE-L moves ~1 point. Reported as lossless.

*Task B — the question "what is the passcode mentioned near line 8,400?" is appended after compression.* The passcode occupies 3 tokens. During prefill nothing attended to them: accumulated attention mass for those keys is at the ~15th percentile, below the retention threshold. They are gone. $\delta_t$ for the decode step that should emit the passcode is again small — about 0.04, because 4093 other keys still dominate the softmax — but the argmax now emits a plausible wrong number. Exact match: 100 → 0.

**The visible obstruction.** Both tasks show $\delta_t \approx 0.03$–$0.04$. One loses 1 point, the other loses 100. The per-step attention error, the only quantity measurable without committing to a task, carries no information about which case you are in. And a benchmark that samples 90% summarization-like tasks reports $\bar\Delta \approx 10.9$ points — while the honest number, $\Delta^{\max} = 100$, appears nowhere in the table.

Deciding the question requires reporting the max over the task grid under prefix-only compression. That number is currently unpublished for every method in Section 3.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*