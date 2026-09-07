---
id: 11-inference-and-serving/batch-size-invariance
title: "Batch-Size Invariance of Model Outputs"
topic: 11-inference-and-serving
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Batch-Size Invariance of Model Outputs

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/batch-size-invariance` · **Status:** partially-solved

## 1. Problem Statement

A served language model is supposed to be a function of its prompt. It is not. The same prompt, same weights, same seed, same temperature-0 decoding rule, submitted twice to the same server, can return different token sequences — because the *other* requests sharing the batch changed. Batch composition is a hidden input.

Three variants, distinct in difficulty:

- **Measurement.** Define and estimate the deviation between the output distribution at batch size $B$ and at $B=1$, for a fixed request. Cheap per-run, but the right functional (logit distance? token-level agreement? task accuracy?) is contested.
- **Method.** Build kernels whose per-request output is bitwise independent of batch composition, at acceptable throughput. This is the part that is **solved in principle and costly in practice** — hence the page status.
- **Theory.** Bound the downstream divergence of greedy or sampled decoding given a per-step logit perturbation of size $\varepsilon$. Open: the sequence map is discontinuous at argmax ties, so worst-case bounds are vacuous and average-case bounds need an unavailable model of the tie-margin distribution.

Solved means: a serving stack that returns bitwise-identical logits for a request regardless of what else is in flight, with a measured throughput cost, plus a quantitative account of what the *non*-invariant stack costs in eval reliability.

## 2. Formal Setting

Let $f_\theta$ be the model in exact real arithmetic and $\tilde f_\theta$ its floating-point realization. A serving step processes a batch $\mathcal{B} = (x_1,\dots,x_B)$ of requests with KV states. Write the realized logits for request $i$ as

$$\ell_i^{(\mathcal{B})} = \tilde f_\theta(x_i \mid \mathcal{B}) \in \mathbb{R}^{|V|}.$$

In exact arithmetic $\ell_i^{(\mathcal{B})} = \ell_i^{(\{x_i\})}$ for all $\mathcal{B}$, since the model has no cross-request term. The deviation is pure implementation.

**Measured quantities.**

- *Logit deviation*: $\delta_i(\mathcal{B}) = \lVert \ell_i^{(\mathcal{B})} - \ell_i^{(\{x_i\})} \rVert_\infty$, measured by running the request alone under an otherwise identical stack and differencing in fp32 after upcast.
- *Bitwise invariance*: the predicate $\mathbb{1}[\,\ell_i^{(\mathcal{B})} \equiv \ell_i^{(\{x_i\})}\,]$ over raw bit patterns. Binary, no tolerance, and the only definition that composes over $T$ decode steps.
- *Tie margin*: $m_i = \ell_i^{(1)} - \ell_i^{(2)}$, the gap between top-1 and top-2 logits. Greedy decoding flips at step $t$ iff $\delta_i > m_i/2$ roughly; measured by histogramming $m$ over a corpus.
- *Completion divergence*: $D = \Pr[\text{two runs of the same prompt differ in any of the first } T \text{ tokens}]$, estimated from $N$ repeats. Also report $t^\ast$, the index of first divergence.

**Where the non-invariance comes from.** Floating-point addition is non-associative: $(a+b)+c \neq a+(b+c)$, with relative error up to $O(n u)$ for an $n$-term sum at unit roundoff $u$ (Higham, 1993). GPU kernels choose reduction order by *shape*: a GEMM at $M=1$ picks a split-K schedule, at $M=512$ a different tile; attention picks its split over KV length; RMSNorm picks a row-per-block or block-per-row strategy. Change $B$, change the schedule, change the sum, change the last bits of $\ell$.

**Assumptions, and which are violated.**

- *Weights fixed across requests.* Holds, unless LoRA-swapping or expert offload rewrites buffers.
- *No cross-request attention.* Holds architecturally; **violated in effect** by shared-prefix KV reuse, where prefix blocks may be recomputed under a different chunk split than they were cached with.
- *Deterministic kernels given fixed shape.* Mostly holds for cuBLAS/cuDNN with fixed algorithm selection; **violated** by `atomicAdd`-based reductions and by autotuners that re-select on cache miss.
- *Batch size is the only varying axis.* **Violated**: chunked prefill splits a prompt at boundaries that depend on the concurrent load, so the same prompt is reduced differently even at equal $B$ (Sarathi-Serve, OSDI 2024).
- *MoE routing is per-token.* **Violated in practice** by capacity-factor dropping, which makes expert assignment an explicit function of batch composition — a first-order, not last-bit, dependence.

## 3. State of the Art

**Established.** He and colleagues at Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025), identified batch-size-dependent reduction order — not GPU atomics or concurrency races — as the dominant cause of temperature-0 nondeterminism in vLLM, and released `batch-invariant-ops`: RMSNorm, matmul, and attention kernels with fixed reduction strategy across shapes. With them, 1000 temperature-0 completions of the same prompt on Qwen3-235B-A22B were reported identical; without, the same run produced on the order of 80 distinct completions, with the first divergence around token 100. This is the reference result for the method variant.

**Claimed but unablated.** The reported throughput cost (roughly a 1.5–2$\times$ slowdown on an unoptimized path in a single measured configuration) comes from a blog post, one model, one hardware generation, no error bars, no sweep over batch size or sequence length. Treat it as an existence proof of cost, not an estimate of it.

**Benchmark-number-only.** Claims that batch-invariant inference improves RL training stability by removing train/inference sampler mismatch rest on single reported runs; the mechanism is plausible and the ablation against a matched non-invariant control at equal compute has not been published.

**Systems SOTA elsewhere.** vLLM (SOSP 2023) and SGLang (NeurIPS 2024) both default to non-invariant kernels; continuous batching and PagedAttention are the throughput wins that create the problem. FlashAttention-2 (ICLR 2024) selects its KV split by occupancy, which is load-dependent.

**Theory SOTA.** Reproducible summation with deterministic error bounds independent of order exists — Demmel and Nguyen, *Fast Reproducible Floating-Point Summation* (ARITH 2013) — at roughly a small constant-factor cost for BLAS-1. It has not been carried into fused transformer kernels at production throughput.

## 4. What Is Known

- Non-associativity magnitude: at bf16 ($u \approx 2^{-8}$) with fp32 accumulate, a $d=8192$ reduction has worst-case relative error $O(du)$; observed $\delta_\infty$ on logits from reduction-order change alone is typically $10^{-3}$–$10^{-2}$ in absolute logit units for 7B-class models — small relative to typical margins, decisive for small ones.
- Divergence is a rare-event amplifier, not a broad shift: in the Qwen3-235B run above, ~100 tokens matched before any run diverged. One flipped argmax then rewrites the whole suffix.
- Greedy decoding is not deterministic in deployed stacks. Atil et al., *LLM Stability: A Detailed Analysis with Some Surprises* (2024/2025), measured repeated greedy runs across several models and tasks and found accuracy spreads of several points across identical repeats — single-digit percentage points at 7B–70B scale on standard benchmarks.
- Activation outliers make it worse. Sun et al., *Massive Activations in Large Language Models* (COLM 2024), document activations $10^3$–$10^4\times$ the median in specific channels of Llama-2-7B/13B; these dominate reduction error and concentrate it in a few coordinates.
- Fixing the reduction order is sufficient for bitwise invariance in dense transformers — no other source needed to be eliminated in the reported vLLM experiment.

## 5. What Is Not Known

- **Empirically open.** The cost curve. Nobody has published throughput-vs-invariance for batch-invariant kernels across $B \in \{1,\dots,256\}$, sequence lengths, and at least two GPU generations. Runnable today on 8 GPUs for under a week.
- **Empirically open.** Whether non-invariance measurably biases benchmark *rankings*, as opposed to adding variance. Distinguishing these requires paired invariant/non-invariant evaluation of many models on the same harness; not done.
- **Theoretically open.** A useful bound on completion divergence $D(T,\varepsilon)$ given per-step logit perturbation $\varepsilon$. Worst case is trivially $1$ (ties exist); the average case needs the tie-margin density near $0$, which is model- and corpus-specific and unmodeled.
- **Methodologically blocked.** Invariance for MoE under capacity-factor token dropping. Here batch composition changes *which expert computes a token* — a semantic, not numerical, dependence. There is no agreed definition of "the batch-size-1 answer" for a model whose routing is defined only relative to a batch.

## 6. Why It Is Hard

The specific obstruction is **an unavoidable trade between reduction-order freedom and arithmetic intensity**. Throughput on modern accelerators comes from choosing the tiling and split to fit the shape: at $B=1$ a GEMM is memory-bound and wants split-K across many SMs; at $B=256$ it is compute-bound and wants a single-K sweep per tile. Forcing one reduction order across all shapes means accepting the wrong schedule at one end of the range. This is a real cost, not a missing engineering effort.

Second obstruction: **confounded measurement**. Batch size in a live server co-varies with chunked-prefill boundaries, KV-cache hit patterns, autotuner cache state, and — for MoE — expert load. An observed output change at $B=64$ versus $B=1$ cannot be attributed to reduction order without holding all of these fixed, which requires instrumenting the scheduler, not just setting a flag.

## 7. Current Research (as of 2026)

- Thinking Machines Lab: batch-invariant kernels, and the claim that they remove on-policy/off-policy mismatch in RL fine-tuning *(frontier — verify)*.
- vLLM and SGLang maintainers: opt-in deterministic execution modes; integration status and coverage of MoE paths varies by release *(frontier — verify)*.
- Numerical-reproducibility groups (descendants of the ReproBLAS line, Demmel et al.): reproducible reductions with bounded cost; not yet fused into attention kernels.
- Evaluation-reliability work quantifying run-to-run variance in LLM benchmarks and calling for repeat-count reporting; largely descriptive so far.

## 8. Concrete Next Experiment

**Question.** How much throughput does bitwise batch invariance cost, and does it change any benchmark conclusion?

**Scale.** One 8$\times$H100 node. Two models: Llama-3.1-8B (dense) and a 30B-class MoE. Two arms of the same server build:

- **Control arm:** stock vLLM, default kernels, continuous batching, chunked prefill on.
- **Treatment arm:** identical build with `batch-invariant-ops` kernels, fixed chunk boundaries, autotuner cache pre-warmed and frozen.

**Protocol.** (a) Sweep $B \in \{1,2,4,8,16,32,64,128,256\}$ at input 2k / output 512, record tokens/s and p50/p99 latency, 5 repeats. (b) For 500 fixed prompts, run $N=100$ temperature-0 repeats per arm under randomized concurrent load; record $D$ and $t^\ast$. (c) Run MMLU, GSM8K, and one long-form agentic benchmark, 5 full repeats per arm.

**The deciding number.** The **throughput ratio at the p50 production batch size**, $R = \text{tok/s}_{\text{invariant}} / \text{tok/s}_{\text{control}}$. If $R \ge 0.95$, invariance is free enough to become the serving default and the problem moves to closed for dense models. If $R \le 0.7$, the problem stays open as a kernel-design question and the field needs the tie-margin theory instead. Secondary decider: benchmark standard deviation across the 5 repeats — control arm expected $>0$, treatment arm must be exactly $0$.

## 9. Key References

- **[SOTA]** Horace He and Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab: Connectionism, September 2025.
- **[Foundational]** Nicholas J. Higham. *The Accuracy of Floating Point Summation.* SIAM Journal on Scientific Computing, 14(4), 1993.
- **[Foundational]** James Demmel and Hong Diep Nguyen. *Fast Reproducible Floating-Point Summation.* IEEE Symposium on Computer Arithmetic (ARITH), 2013.
- **[Foundational]** Nathan Whitehead and Alex Fit-Florea. *Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs.* NVIDIA technical whitepaper, 2011.
- **[Systems]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Systems]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024.
- **[Systems]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691
- **[Systems]** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024.
- **[Empirical]** Berk Atil, Alexa Chittams, Liseng Fu, Ferhan Ture, Lixinyu Xu, Breck Baldwin. *LLM Stability: A Detailed Analysis with Some Surprises.* 2024.
- **[Empirical]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762

## 10. Worked Example

Take a single decode step of an 8B model, $d_{\text{model}}=4096$, bf16 weights, fp32 accumulate. The final unembedding row for token $v$ is a 4096-term dot product. Under split-K with $K_{\text{split}}=8$ (the $B=1$ schedule) the sum is eight partial sums of 512 terms, each rounded, then combined. Under the $B=64$ schedule, $K_{\text{split}}=1$: one 4096-term sequential accumulation.

Empirically, the two orders differ in the logit by roughly $10^{-3}$ in absolute units for typical activations — call it $\delta = 2\times10^{-3}$.

Now the margins. At a confident step, $m = \ell^{(1)} - \ell^{(2)} \approx 4.0$; $\delta/m = 5\times10^{-4}$, no flip, ever. But margins are heavy near zero. Suppose $\Pr[m < 2\delta] = 10^{-4}$ per step — one step in ten thousand is inside the noise floor. Over a $T=512$-token completion, the chance of at least one flip is

$$1 - (1 - 10^{-4})^{512} \approx 5\%.$$

Run 1000 completions and about 50 diverge; each divergence rewrites its suffix, so you observe dozens of distinct outputs at temperature 0 — the order of magnitude actually reported for Qwen3-235B.

**Where the obstruction becomes visible.** The per-step error is a millionth of the logit scale and could be dismissed as irrelevant. It is not, because decoding applies $\arg\max$, a discontinuous map, 512 times in series, and the tail of the margin distribution — not the size of $\delta$ — sets the divergence rate. Halving $\delta$ with higher-precision accumulation halves $\Pr[m<2\delta]$ only linearly, so you would need roughly $10^{4}\times$ error reduction to make divergence negligible by precision alone. Bitwise invariance sets $\delta = 0$ exactly and ends the argument; there is no cheaper approximate route, and that is why the trade is against throughput rather than against precision.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*