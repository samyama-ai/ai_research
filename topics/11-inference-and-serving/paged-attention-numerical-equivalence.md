---
id: 11-inference-and-serving/paged-attention-numerical-equivalence
title: "Exact Equivalence of Paged and Contiguous Attention"
topic: 11-inference-and-serving
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exact Equivalence of Paged and Contiguous Attention

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/paged-attention-numerical-equivalence` · **Status:** partially-solved

## 1. Problem Statement

PagedAttention stores the KV cache in fixed-size non-contiguous blocks and computes attention by a blocked, split-reduction kernel. The contiguous reference computes attention over one flat KV tensor. Both implement the same mathematical function. Neither is claimed to produce the same *bits*.

The problem: **state and achieve a well-defined equivalence relation between a paged attention implementation and its contiguous reference, under which the relation provably holds.**

Three variants, of very different difficulty:

- **Measurement.** What is "equivalent"? Bitwise identity of output tensors; bounded deviation in ULPs; identical greedy token sequences; equality in distribution of sampled outputs. These are not interchangeable — bounded output error does **not** imply identical token sequences, because $\arg\max$ is discontinuous.
- **Method.** Build a paged kernel whose reduction order does not depend on block count, batch composition, chunk boundaries, or prefix-cache hits, at acceptable throughput cost.
- **Theory.** Prove a forward-error bound on the paged/contiguous gap, and a bound on the resulting divergence probability over $T$ decode steps.

A solution: a serving stack where, for a fixed model, fixed weights, fixed prompt and fixed seed, the emitted token sequence is invariant to block size, batch composition, prefix-cache state, and chunked-prefill boundaries — with a stated throughput cost and a proof or exhaustive test that the invariance holds.

## 2. Formal Setting

One head, query $q \in \mathbb{R}^d$, keys/values $K, V \in \mathbb{R}^{n \times d}$, scores $s_i = q^\top k_i / \sqrt d$. Exact attention:

$$O = \sum_{i=1}^{n} \frac{e^{s_i}}{\sum_{j} e^{s_j}} v_i .$$

**Paged evaluation.** $K,V$ are partitioned into $B = \lceil n/b \rceil$ blocks of size $b$ (vLLM default $b = 16$). Each block $c$ yields a local max $m_c = \max_{i \in c} s_i$, a local weight sum $\ell_c = \sum_{i\in c} e^{s_i - m_c}$, and a partial output $O_c = \sum_{i\in c} e^{s_i-m_c} v_i$. Partials are merged by log-sum-exp:

$$m = \max_c m_c, \quad \ell = \sum_c e^{m_c-m}\ell_c, \quad O_{\text{page}} = \frac{1}{\ell}\sum_c e^{m_c - m} O_c .$$

The merge is mathematically exact and floating-point *inexact*: it is a different association of the same sum.

**Measured quantities.**

- **Output deviation.** $\delta = \|O_{\text{page}} - O_{\text{ref}}\|_\infty / \|O_{\text{ref}}\|_\infty$, measured against an fp64 contiguous reference, per layer, per head, per token.
- **Logit deviation.** $\Delta = \|z_{\text{page}} - z_{\text{ref}}\|_\infty$ over the vocabulary logits $z \in \mathbb{R}^{|\mathcal V|}$ at the final layer.
- **Decision margin.** $g_t = z_t^{(1)} - z_t^{(2)}$, gap between top-1 and top-2 logit at step $t$. Greedy output flips at $t$ when $\Delta_t > g_t/2$ (sufficient condition for safety: $\Delta_t < g_t/2$).
- **Divergence rate.** $D = \Pr[\exists\, t \le T : \hat y_t \ne y_t]$, over prompts, measured as the fraction of prompts whose greedy continuation differs from the reference within $T$ tokens.

**Standard error bound.** For floating-point summation of $n$ terms with unit roundoff $u$, sequential summation gives $|\hat S - S| \le \gamma_{n-1}\sum_i |x_i|$ with $\gamma_k = ku/(1-ku)$; blocked summation with block $b$ gives $\gamma_{b + B - 2}$ (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002). Paged is often *better* conditioned than sequential — and still different.

**Assumptions, and which are violated.**

- *Same arithmetic precision on both paths.* Violated: reference implementations (PyTorch SDPA, HF eager) may run fp32 accumulation with different intermediate rounding; paged kernels use tensor-core fp32 accumulation with fp16/bf16 inputs and, on Hopper/Blackwell, FP8 variants.
- *Reduction order depends only on $n$.* Violated: split-K attention chooses the split count from occupancy heuristics, so it depends on batch size and available SMs.
- *Prefix cache is a pure memoization.* Violated: a cache hit changes chunk boundaries, so the recomputed prefix is not bit-identical to the cached one.
- *Attention is the only non-associative stage.* Violated: RMSNorm reductions, MoE expert routing under different token grouping, and the LM-head GEMM tiling are all batch-size dependent.

## 3. State of the Art

**Established.**

- PagedAttention (Kwon et al., SOSP 2023) established the memory model — 4× or better serving throughput at matched latency versus contiguous-allocation baselines — and does **not** claim bit-equivalence with a contiguous reference.
- FlashAttention-2 (Dao, ICLR 2024) and FlashAttention-3 (Shah et al., NeurIPS 2024) establish that the online-softmax tiling is exact in real arithmetic; FA-3 reports FP8 attention with ~2.6× lower RMS error than a baseline FP8 attention, which is a *statement about error magnitude*, not about equivalence.
- Batch-invariant kernels (Thinking Machines Lab / He, *Defeating Nondeterminism in LLM Inference*, 2025) established the diagnosis: the dominant source of run-to-run nondeterminism at temperature 0 is **lack of batch invariance** in RMSNorm, matmul and attention reductions, not atomics. They released batch-invariant RMSNorm/matmul/attention kernels and a vLLM integration that produces bitwise-identical completions across batch sizes.

**Claimed but unablated.**

- The claim that fixed-split-size attention removes *all* paged/contiguous divergence has been shown for specific model/kernel pairs and demonstrated as a run, not proven across block sizes, prefix-cache states, chunked-prefill boundaries, MoE routing, and speculative-decoding acceptance paths jointly.
- Throughput cost of batch-invariance is reported in the low tens of percent for one configuration. That is a benchmark number from a single setup, not a characterized cost curve over model size, sequence length, and hardware generation.

**Benchmark-only.** vLLM and SGLang both expose flags/modes intended to make outputs reproducible. Their coverage is documented by tests, not by an equivalence argument; no published matrix reports divergence rate as a function of block size $\times$ batch $\times$ cache state.

## 4. What Is Known

- **Temperature-0 decoding is not deterministic by default.** He et al. (2025) report 1000 completions of a single prompt with Qwen3-235B-A22B-Instruct-2507 in vLLM producing **80 distinct completions**, with the earliest divergence around token 103 — same prompt, same seed, same weights, greedy sampling. With batch-invariant kernels, the same run yields 1 distinct completion out of 1000.
- **Divergence is driven by batch composition, not by the paging itself.** The relevant variable is what else is in the batch, which changes reduction shapes.
- **Error magnitudes are small; consequences are not.** bf16 has 8 explicit mantissa bits, $u = 2^{-9} \approx 2.0\times 10^{-3}$; fp32 accumulate has $u = 2^{-24} \approx 6.0\times10^{-8}$. Attention output deviations $\delta$ between paged and contiguous at 8B scale sit in the $10^{-7}$–$10^{-6}$ range with fp32 accumulation — several orders below any weight-quantization error, and still enough to flip argmax.
- **Downstream variance is measurable at benchmark scale.** Atil et al., *LLM Stability: A Detailed Analysis with Some Surprises* (2024), report that repeated identical runs at temperature 0 across models and tasks rarely reproduce exact outputs, with accuracy varying run to run on standard benchmarks.
- **Blocked summation is not worse than sequential.** By the $\gamma_{b+B-2}$ bound, paged attention with $b=16$, $n=4096$ has worst-case constant $\approx 270u$ against $\approx 4095u$ sequential. Paged is, in the worst case, ~15× tighter — the problem is difference, not degradation.

## 5. What Is Not Known

- **Theoretically open.** No published bound relates per-step logit deviation $\Delta$ to sequence divergence rate $D$ over $T$ steps. This requires a model of the decision-margin distribution $g_t$, which nobody has characterized. Also open: whether any paged layout admits bitwise equality with a contiguous reference at fixed precision without forcing the contiguous path to adopt the paged reduction order (i.e. whether "equivalence" can be anything other than "make both sides use the same tree").
- **Empirically open.** The full grid — block size $\{16,32,64\}$ × batch $\{1,\dots,256\}$ × prefix-cache hit/miss × chunked-prefill on/off — has not been run at any scale with divergence rate reported. It is runnable today on a single 8×H100 node.
- **Methodologically blocked.** There is no agreed equivalence predicate. Papers report RMS error against fp64; serving systems report reproducibility across reruns; users care about identical text. No standard names which of these is being claimed, so "deterministic inference" claims are not comparable across systems.

## 6. Why It Is Hard

The obstruction is **discontinuity of the decoding map under a continuous perturbation**, compounded by **absent ground truth for the decision margin**.

A forward-error bound gives $\Delta_t \le \varepsilon$. Greedy decoding is safe only if $\varepsilon < g_t/2$. But $g_t$ is a property of the model's uncertainty at step $t$ and is arbitrarily small at any step where two continuations are near-tied — which is exactly where natural language is ambiguous. So no per-step bound converts into a sequence-level guarantee without a distributional assumption about $g_t$ that nobody has measured at scale. Worse, once a flip occurs, the two rollouts see different contexts and the error is no longer small: the perturbation is amplified from $10^{-6}$ to $O(1)$ in one step.

Second obstruction: **confounded measurement**. The paged/contiguous difference cannot be isolated by A/B-ing engines, because changing engines also changes GEMM tiling, norm reductions, RoPE implementation, and dtype casts. Attributing a divergence to paging requires holding all of those fixed, which in practice means writing both arms inside one kernel library.

## 7. Current Research (as of 2026)

- **Batch-invariant kernel libraries.** Thinking Machines Lab's `batch_invariant_ops` and its vLLM integration is the reference implementation. Extension to FP8 attention and to MoE routing under batch-invariance is active *(frontier — verify)*.
- **Deterministic modes in serving engines.** vLLM and SGLang both track reproducibility as a first-class issue; the direction is fixed split counts and fixed reduction trees rather than post-hoc seeding.
- **Attention kernel generators.** FlashInfer (Ye et al., MLSys 2025) makes reduction schedules explicit and parameterizable, which is the precondition for pinning them. Whether a pinned schedule can be exposed as a supported contract is not settled.
- **RL/training-inference mismatch.** The strongest current motivation: on-policy RL requires the sampler's log-probs to match the trainer's. Divergence here is a silent off-policy bias. Several groups report that batch-invariant sampling removes a measured KL gap between sampler and trainer *(frontier — verify)*.
- **Verification.** Nothing published applies formal FP verification (interval arithmetic, or an SMT-backed ULP bound) to an attention kernel. This is the open lane.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B-Instruct, bf16 weights, fp32 attention accumulation, single H100 80GB. 2,000 prompts from a mixed pool (1,000 short-context $\le$ 512 tokens, 1,000 long-context 8k–32k), greedy decoding, $T = 512$ tokens each.

**Arms.** One engine (vLLM), one kernel library, varying exactly one axis at a time from a pinned baseline of (block 16, batch 1, prefix cache off, chunked prefill off):
1. block size $\in \{16, 32, 64\}$
2. batch size $\in \{1, 8, 64, 256\}$ (identical prompt padded into a batch of decoys)
3. prefix cache hit vs. miss on an identical prompt
4. chunked prefill on/off with chunk 512 vs 2048

**Control arm.** Contiguous fp64 reference in PyTorch, single request, no paging, no batching — the numerically-exact ground truth for $g_t$ and $\Delta_t$. Second control: the same grid with batch-invariant kernels enabled.

**Instrumentation.** Log per-step $\Delta_t$ and $g_t$ against the fp64 control; log the step index of first divergence.

**The deciding number.** The **greedy-divergence rate $D$ across the 13-cell grid**, reported as (prompts diverged) / 2000 per cell. Exact equivalence is claimed only if every cell reports $0/2000$. Any nonzero cell names the axis that breaks it. The secondary number that makes it explanatory: the empirical fraction $\Pr[g_t < 2\Delta_t]$ — the per-step flip probability — which converts $D$ into a predictive model instead of an observation. Cost: roughly 200 GPU-hours.

## 9. Key References

- **[Foundational]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Foundational]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[Foundational]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — arXiv:2307.08691
- **[SOTA]** Horace He and Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab: Connectionism, 2025.
- **[SOTA]** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* NeurIPS, 2024. — arXiv:2407.08608
- **[SOTA]** Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, et al. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* MLSys, 2025. — arXiv:2501.01005
- **[Empirical]** Berk Atil, Alexa Chittams, Liseng Fu, Ferhan Ture, Lixinyu Xu, Breck Baldwin. *LLM Stability: A Detailed Analysis with Some Surprises.* 2024. — arXiv:2408.04667
- **[Survey]** Nicholas J. Higham. *Accuracy and Stability of Numerical Algorithms.* SIAM, 2nd ed., 2002. — Ch. 4, summation error bounds.
- **[Related]** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024. — arXiv:2312.07104

## 10. Worked Example

One decode step, one head, $n = 4096$ KV entries, $d = 128$, fp32 accumulation ($u = 5.96\times10^{-8}$), bf16 inputs.

**Reduction trees.** Contiguous FlashAttention with tile 128: 32 tiles, worst-case summation constant $\gamma_{128 + 32 - 2} = 158u \approx 9.4\times10^{-6}$. Paged with $b = 16$: 256 blocks, $\gamma_{16+256-2} = 270u \approx 1.6\times10^{-5}$. Split-K with 4 splits reassociates again. All three differ; none is wrong.

**Realized deviation.** In practice the bound is loose and the observed attention-output deviation is $\delta \approx 10^{-7}$. Propagated through 32 layers and the LM head over $|\mathcal V| = 128{,}256$, the logit deviation lands around $\Delta \approx 3\times 10^{-5}$ in absolute logit units.

**Where it bites.** Take a step where the model is deciding between " the" and " a": top-1 logit 12.4013, top-2 logit 12.4012, so $g_t = 1.0\times10^{-4}$. Then $g_t/2 = 5\times10^{-5} > \Delta$ — safe, barely. Shift the batch from 1 to 64 and $\Delta$ grows to $1.2\times10^{-4}$: the flip condition $\Delta > g_t/2$ is met, and the token changes.

**Why a small error is not a small effect.** Suppose the per-step flip probability is $p = \Pr[g_t < 2\Delta_t] = 10^{-3}$ — a plausible value when a few tokens per thousand are near-tied. Over $T = 512$ steps:

$$D = 1 - (1-p)^{T} = 1 - (0.999)^{512} \approx 0.40 .$$

**Forty percent of 512-token completions change**, from a numerical perturbation eight orders of magnitude below bf16 resolution. And the flip is not a local edit: after the first divergent token the two rollouts condition on different prefixes, so the perturbation is $O(1)$ from step $t+1$ onward.

This is the obstruction in one line: the error bound is tight and small, the decision margin $g_t$ is unbounded below, and no bound on $\delta$ constrains $D$ without a measured distribution of $g_t$ — which is precisely the quantity nobody has published.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*