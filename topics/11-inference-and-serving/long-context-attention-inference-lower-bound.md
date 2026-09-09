---
id: 11-inference-and-serving/long-context-attention-inference-lower-bound
title: "Long-Context Attention Cost Lower Bound at Inference"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Attention Cost Lower Bound at Inference

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/long-context-attention-inference-lower-bound` · **Status:** open

## 1. Problem Statement

Serving a transformer over a context of $n$ tokens costs $\Theta(n^2)$ arithmetic at prefill and $\Theta(n)$ memory traffic per decoded token. Every deployed long-context system pays this. The question is whether it *must*.

**Decision predicate.** Fix a model family and a task distribution. Does there exist an inference procedure that, for context length $n$, uses $o(n)$ bits of per-request state and $o(n)$ memory traffic per generated token, while matching the full-attention model's accuracy to within $\epsilon$ on that distribution?

Three variants, with very different difficulty:

- **Theory.** Prove an unconditional or fine-grained-conditional lower bound on time/space for *approximating* attention output or for solving the downstream retrieval task. Partially answered for exact and for high-entry-magnitude attention; open for the approximation regimes that matter.
- **Method.** Build a sparse/compressed/recurrent decoder with $o(n)$ state that loses nothing measurable. Many claims; none survives a query-adversarial control.
- **Measurement.** Define "loses nothing." Current long-context benchmarks are dominated by tasks solvable from a fixed-size summary, so a method with $O(1)$ state can score well while destroying information the deployment actually needs. This variant is the binding constraint.

Solving it means either (a) a theorem that any procedure achieving accuracy $\ge 1-\epsilon$ on a stated task family needs $\Omega(n)$ state, or (b) a $o(n)$-state system that holds up against that task family.

## 2. Formal Setting

**Objects.** Context $x_{1:n}$, vocabulary $V$. Model $M$ with $L$ layers, $H$ query heads, $H_{kv}$ key/value heads, head dimension $d_h$, model width $d = H d_h$. Per layer and head, with $Q, K, V \in \mathbb{R}^{n \times d_h}$ and causal mask $\mathcal{M}$:

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + \mathcal{M}\right)V .$$

**Measured quantities.**

- *Cache size* $S(n)$ — bits resident per request. Measured, not derived: allocator-reported bytes for the KV blocks (e.g. vLLM's block manager), including quantization scales and page fragmentation. Baseline $S_{\text{full}}(n) = 2 L H_{kv} d_h n b$ bits at $b$ bits/element.
- *Decode traffic* $T(n)$ — bytes read from HBM per generated token, from hardware counters (`dram__bytes_read` via Nsight, or DCGM), not from a FLOP model.
- *Prefill work* $C(n)$ — achieved FLOPs, wall-clock $\times$ measured utilization.
- *Quality* $Q(M', \mathcal{D}) = \mathbb{E}_{(x,q,y)\sim\mathcal{D}}[\,\mathbb{1}\{M'(x,q) = y\}\,]$, where $q$ is a query issued **after** the context is compressed.
- *Fidelity* $\delta = \max_i \|\hat{o}_i - o_i\|_\infty / \|o_i\|_\infty$, attention-output error against the exact FP32 reference.

**The predicate, formally.** Method $A$ *beats the linear barrier* on $\mathcal{D}$ at tolerance $\epsilon$ if $S_A(n) = o(n)$, $T_A(n) = o(n)$, and $Q(A) \ge Q(\text{full}) - \epsilon$ for all $n$ in the tested range.

**Assumptions, and which are violated.**

1. *Bounded entries.* Theory results assume $\|Q\|_\infty,\|K\|_\infty \le B$ with $B = o(\sqrt{\log n})$. **Violated:** production models have massive activations and attention sinks; entry magnitudes grow with depth and are not $o(\sqrt{\log n})$ in any measured sense.
2. *Query independent of the compression.* **Violated by construction** in query-aware methods (SnapKV, Quest), which observe part of the query before evicting — this is a different, easier problem and is routinely compared against query-agnostic baselines.
3. *Softmax attention concentrates.* Sparsity methods assume the attention matrix is near-low-rank or near-sparse. **Partially violated:** measured sparsity is head-dependent and layer-dependent; retrieval heads are dense over the exact tokens that matter.
4. *Single-pass, single-query.* Real serving is multi-turn: a cache compressed for turn 1 must answer turn 7. Almost no published evaluation does this.

## 3. State of the Art

**Theory SOTA (established).**

- Keles, Wijewardena & Hegde (ALT 2023): computing self-attention exactly requires $n^{2-o(1)}$ time under SETH. Established, for exact computation.
- Alman & Song (NeurIPS 2023), *Fast Attention Requires Bounded Entries*: with entries bounded by $B = o(\sqrt{\log n})$ there is an $n^{1+o(1)}$ algorithm for $1/\mathrm{poly}(n)$-approximate attention via the polynomial method; at $B = \Theta(\sqrt{\log n})$ no truly subquadratic algorithm exists unless SETH fails. This is the sharpest known threshold and it is a *phase transition*, not a blanket barrier.
- Alman & Yu (ICLR 2025), *Fundamental Limitations on Subquadratic Alternatives to Transformers*: any subquadratic-time architecture fails document-similarity-style tasks that full attention solves, under fine-grained hypotheses.
- Sanford, Hsu & Telgarsky (NeurIPS 2023): communication-complexity separation — one attention head solves sparse averaging at size $O(\log n)$; bounded-state alternatives need polynomially more.
- Jelassi et al. (ICML 2024): transformers copy length-$n$ strings with $O(\log n)$-size construction; state-space models need state growing with $n$. Reproduced empirically.

**Systems SOTA (established).** FlashAttention-2 (Dao, ICLR 2024) and FlashAttention-3 (Shah et al., NeurIPS 2024) remove the $O(n^2)$ *memory* term but not the $O(n^2)$ *FLOP* term. PagedAttention/vLLM (Kwon et al., SOSP 2023) removes fragmentation, not asymptotics. GQA (Ainslie et al., EMNLP 2023) and MLA (DeepSeek-V2, 2024) cut the constant on $S(n)$ by $8\text{–}30\times$; the $n$ dependence is untouched.

**Claimed but unablated.** H2O (NeurIPS 2023), StreamingLLM (ICLR 2024), SnapKV (NeurIPS 2024), MInference (NeurIPS 2024), Quest (ICML 2024) all report large speedups at near-parity accuracy — MInference reports up to $10\times$ prefill latency reduction at 1M tokens on A100; StreamingLLM $22\times$ decode speedup. These are **benchmark numbers on query-known-in-advance or summarization-heavy suites**. None has been ablated against an adversarial-query control where the query arrives after eviction, and StreamingLLM's authors state plainly that it extends streaming, not effective context.

## 4. What Is Known

- **Cache is the binding cost at long context.** Llama-3.1-70B: $L=80$, $H_{kv}=8$, $d_h=128$, bf16 → $2\cdot80\cdot8\cdot128\cdot2 = 327{,}680$ B = **320 KiB per token**; 128K tokens = **40 GiB**, versus 140 GB of weights. Batch 4 at 128K exceeds the weights.
- **Attention FLOPs overtake the FFN at ~100K.** Same model, $n=131{,}072$: causal attention $\approx 2 n^2 d L = 2.2\times10^{16}$ FLOP versus $2 N_{\text{params}} n = 1.8\times10^{16}$ FLOP for everything else. Measured at 128K, single sequence.
- **Attention sinks are real and reproduced.** Removing the first few tokens collapses perplexity; keeping 4 sink tokens plus a sliding window restores it (Xiao et al., ICLR 2024, Llama-2-7B/13B, up to 4M streamed tokens).
- **Recall degrades with state budget in a measurable law.** Arora et al. (ICML 2024, *Based*) show a recall–state-size tradeoff on MQAR, with a communication-complexity argument that associative recall over $N$ pairs needs state growing with $N$; verified at 355M–1.3B scale.
- **Fixed-budget eviction fails multi-query.** Reported across follow-ups: methods holding $\sim$20% of the cache retain single-needle accuracy but lose multi-needle and multi-turn accuracy, dropping tens of points on RULER-style suites at 64K–128K.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on *state* for approximate autoregressive decoding under a realistic error metric. The Alman–Song threshold governs time for one attention call with bounded entries; nothing forbids an $O(\sqrt{n})$-state decoder from matching full attention to $\delta = 10^{-2}$ on natural inputs. Whether the $B = \Theta(\sqrt{\log n})$ hard regime is *reached* by trained models is also unproven either way. *(A 2025 line on compression barriers for autoregressive transformers — Haris & Onak — targets exactly this; frontier — verify.)*
- **Empirically open.** No one has run a query-adversarial, multi-turn evaluation of the leading $o(n)$-state methods at $\ge 128$K on a $\ge 70$B model. The experiment costs $\sim$$10^4$ GPU-hours; it is runnable today.
- **Methodologically blocked.** "Same accuracy" has no accepted definition. Needle-in-a-haystack is saturated and is solvable from an $O(1)$ summary if the needle is salient. Perplexity is insensitive to losing 1 fact in 100K tokens. Without a task family whose information-theoretic state requirement is known, no method claim is falsifiable.

## 6. Why It Is Hard

**Confounded measurement, not compute.** The obstruction is that the standard long-context benchmarks do not measure the quantity in the problem statement. A KV-eviction method is nominally tested on "can the model use 128K tokens," but the tasks are dominated by items answerable from a few hundred salient tokens — so the benchmark score is largely invariant to how much of the context survives compression. Two methods with $S(n) = \Theta(n)$ and $S(n) = O(1)$ can land within noise. That makes the empirical question unfalsifiable and, in turn, gives theory no target: nobody can state the task distribution the bound should be proven against.

Second obstruction, secondary: **non-identifiability of the query.** A lower bound on state only holds if the compressor is query-agnostic. Any method that peeks at the query converts an $\Omega(n)$ streaming problem into an $O(1)$ filtering problem, and the two are compared as if equivalent.

## 7. Current Research (as of 2026)

- **Fine-grained complexity of attention variants.** Alman and Song, and Alman and Yu, extending polynomial-method upper bounds and SETH lower bounds to RoPE attention, gradients, and multi-layer composition *(frontier — verify the 2025–26 entries)*.
- **Hybrid architectures.** Mamba-2/Jamba-style interleaving of a few full-attention layers with linear-recurrent layers, on the hypothesis that $O(1)$ layers of $\Theta(n)$ state suffice. Empirically strong; the question of *how many* attention layers are necessary is unanswered.
- **Query-aware sparse decode.** Quest, DuoAttention, and successors: per-head classification into retrieval versus streaming heads, then $\Theta(n)$ cache only for retrieval heads. Reduces the constant, keeps the asymptotic.
- **Benchmark repair.** RULER and successors add synthetic multi-hop and multi-needle tasks with controllable information content. This is the direction most likely to unblock the measurement variant.

## 8. Concrete Next Experiment

**The state–accuracy frontier, measured against a query-adversarial control.**

- **Scale.** One 70B-class GQA model, contexts at $n \in \{32\text{K}, 128\text{K}, 512\text{K}\}$, 8×H100 node. $\sim$2,000 GPU-hours.
- **Task.** $k$-needle retrieval with **calibrated information content**: plant $k \in \{1, 8, 64, 512\}$ independent 64-bit facts uniformly in the context; ask for $m=8$ of them, chosen uniformly *after* the cache is built. The context carries $64k$ bits that a query-agnostic compressor cannot know it will need — an explicit, computable state floor.
- **Arms.** (i) Full attention. (ii) Each of StreamingLLM, H2O, SnapKV, Quest at cache budgets $\{50\%, 25\%, 12.5\%, 6.25\%, 3\%\}$. (iii) **Control arm:** the identical methods with the query prepended before compression. The gap between (ii) and (iii) isolates how much of the reported accuracy comes from query leakage rather than compression quality.
- **The deciding number.** $k^\star(S)$ — the largest $k$ at which a method holds within 2 points of full attention, as a function of measured cache bits $S$. If $k^\star(S)$ scales linearly in $S$ with slope near $1/64$ bits per fact, the $\Omega(n)$ barrier is empirically real and the theory target is fixed. If any query-agnostic method achieves $k^\star$ growing while $S = o(n)$, the barrier is false and that method is the counterexample.

Report $S$ from the allocator and $T$ from DRAM counters, not from the nominal budget.

## 9. Key References

- **[Foundational]** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[Theory SOTA]** Keles, Wijewardena, Hegde. *On the Computational Complexity of Self-Attention.* ALT, 2023. — arXiv:2209.04881
- **[Theory SOTA]** Alman, Song. *Fast Attention Requires Bounded Entries.* NeurIPS, 2023. — arXiv:2302.13214
- **[Theory SOTA]** Alman, Yu. *Fundamental Limitations on Subquadratic Alternatives to Transformers.* ICLR, 2025. — arXiv:2410.04271
- **[Theory]** Sanford, Hsu, Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS, 2023. — arXiv:2306.02896
- **[Theory]** Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[Systems SOTA]** Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[Systems SOTA]** Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* NeurIPS, 2024. — arXiv:2407.08608
- **[Systems SOTA]** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Systems]** Pope, Douglas, Chowdhery, Devlin, Bradbury, Levskaya, Heek, Xiao, Agrawal, Dean. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102
- **[Method]** Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023. — arXiv:2305.13245
- **[Method]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Method]** Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett, Wang, Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS, 2023. — arXiv:2306.14048
- **[Method]** Tang, Zhao, Zhu, Cai, Wang, Han. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML, 2024. — arXiv:2406.10774
- **[Method]** Jiang, Li, Zhang, Luo, et al. *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention.* NeurIPS, 2024. — arXiv:2407.02490
- **[Empirical/Survey]** Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra, Ré. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[Benchmark]** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654

## 10. Worked Example

Llama-3.1-70B, 8×H100 (80 GB, 3.35 TB/s HBM each), $n = 128$K.

**Full attention.** KV cache $= 320$ KiB/token $\times$ 131,072 $= 40$ GiB per request. Weights in bf16 $= 140$ GB. Per decoded token the node reads $140 + 40 = 180$ GB; at 26.8 TB/s aggregate that is a **6.7 ms floor**, of which 1.5 ms is KV. At batch 8 the KV term becomes $8 \times 40 = 320$ GiB — it no longer fits alongside the weights in 640 GB, and the KV read alone costs 12 ms/token. Attention is now 80% of the traffic.

**A 3% eviction method.** $S = 1.2$ GiB per request, KV read drops to 0.045 ms/token. Batch 64 fits. Reported speedup: $\sim6\times$ end-to-end. On needle-in-a-haystack it scores 99%.

**Where the obstruction becomes visible.** Run the calibrated task instead. At 3% budget, $S = 1.2$ GiB $\approx 10^{10}$ bits — apparently vast. But the retained bits are *chosen by attention scores computed without the query*. Plant $k = 512$ facts of 64 bits each: 32,768 bits of query-relevant information, spread over 131,072 positions with no local salience. The eviction policy keeps the 3% of tokens with the highest historical attention, which correlates with sinks, delimiters, and recent tokens — not with the planted facts, which are individually unremarkable. Measured outcome in this regime: full attention answers $\ge 95\%$ of the 8 sampled queries; the 3% method answers close to the rate at which a needle happens to fall inside the retained window, $\approx 3\%$ plus recency effects.

Now the control arm. Prepend the query before compression. The same method jumps back to $>90\%$, because it is no longer compressing — it is filtering with the answer in hand. **The $6\times$ speedup and the 99% needle score are both real; neither is evidence about the state lower bound.** That is the whole difficulty: the published number and the number the problem asks for come apart, and only the adversarial control separates them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*