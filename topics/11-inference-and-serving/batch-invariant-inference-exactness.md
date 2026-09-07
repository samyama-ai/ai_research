---
id: 11-inference-and-serving/batch-invariant-inference-exactness
title: "Exactness of Batched Nondeterministic Inference"
topic: 11-inference-and-serving
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exactness of Batched Nondeterministic Inference

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/batch-invariant-inference-exactness` · **Status:** partially-solved

## 1. Problem Statement

A production LLM server returns different outputs for the same request, same weights, same sampling seed, and temperature $0$. The cause is not sampling. It is that the request was batched with different neighbours on each call, and floating-point reduction order inside GEMM, attention, and MoE kernels depends on batch shape. The server is a deterministic function of $(\text{request}, \text{batch})$, but the client only controls $\text{request}$.

Three variants, of very different difficulty:

- **Measurement.** Given a serving stack, quantify how far it is from batch-invariant: how often, how early, and how far outputs diverge as a function of load. Requires an agreed metric and an adversarial load generator; neither is standard.
- **Method.** Build kernels whose per-element output is bitwise independent of batch composition, at acceptable throughput cost. Largely solved for dense transformers on a single node; open for MoE, speculative decoding, multi-node tensor/expert parallelism, and cross-hardware portability.
- **Theory.** Characterise the cost of batch invariance. Is there an asymptotic penalty, or only a constant-factor one from giving up split-$K$ and dynamic tiling? No lower bound is known.

**Solved** means: for a fixed weight file and fixed server binary, every request returns bitwise-identical logits regardless of what else is in flight, with a documented throughput penalty and a test that fails when the property breaks.

## 2. Formal Setting

Let $\theta$ be the weight tensors, $x$ a request (prompt plus sampling parameters plus seed), and $B = \{x, x_1, \dots, x_{k-1}\}$ the multiset of requests co-resident in the forward pass. A serving stack realises
$$\hat f_\theta : (x, B, \sigma) \mapsto y \in \mathcal{V}^*,$$
where $\sigma$ is the *execution context*: kernel-selection autotuning results, tensor-parallel degree $p$, NCCL algorithm choice, chunked-prefill boundaries, prefix-cache hit pattern, GPU clocks.

**Batch invariance** is the predicate
$$\forall B, B' \ni x:\quad \hat f_\theta(x, B, \sigma) \;=\; \hat f_\theta(x, B', \sigma) \quad\text{(bitwise)}.$$
This is strictly weaker than reproducibility across $\sigma$ (different $p$, different GPU) and strictly stronger than run-to-run determinism at fixed batch (which cuBLAS/cuDNN already give with deterministic algorithm flags).

**Measured quantities.** Issue $N$ identical copies of $x$ against a server under a background load process $\Lambda$, collecting $y^{(1)},\dots,y^{(N)}$ and per-step logits $\ell^{(i)}_t \in \mathbb{R}^{|\mathcal{V}|}$.

- Unique-completion count $U = |\{y^{(i)}\}|$; the exactness rate is $R = \max_v |\{i: y^{(i)}=v\}| / N$.
- First-divergence index $D = \min\{t : \exists i,j,\ y^{(i)}_t \neq y^{(j)}_t\}$, $D=\infty$ if none.
- Logit drift $\epsilon_\infty(t) = \max_{i,j} \|\ell^{(i)}_t - \ell^{(j)}_t\|_\infty$, reported in units of the bf16 ULP at the logit magnitude.
- Throughput ratio $\rho = \text{tok/s}_{\text{invariant}} / \text{tok/s}_{\text{default}}$ at matched p50 latency.

The source of nonzero $\epsilon_\infty$ is non-associativity: for a reduction of length $n$ split into $s$ chunks, $\mathrm{fl}(\sum) $ depends on $s$, and $s$ is chosen at runtime from the batch dimension by split-$K$ GEMM and by FlashDecoding-style attention splits.

**Assumptions, and which are violated in practice.**
1. *The batch is the only hidden input.* Violated: prefix caching changes prefill chunk boundaries, so cache state is a second hidden input.
2. *Reduction order is the only nondeterminism.* Violated where kernels use `atomicAdd` (some backward and MoE scatter paths), which is order-nondeterministic even at fixed shape.
3. *Requests are independent given the batch.* Violated by MoE expert-capacity dropping — a token's routing can depend on other tokens in the batch, which is a *semantic*, not numerical, coupling.
4. *One node.* Violated at $p>1$: NCCL selects ring vs tree by message size, changing allreduce summation order with batch shape.

## 3. State of the Art

**Systems/empirical SOTA — established.** He and collaborators at Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025), identified batch-size-dependent reduction order as the dominant cause, and shipped batch-invariant RMSNorm, matmul, and attention kernels (`thinking-machines-lab/batch-invariant-ops`) integrated with vLLM. The claim that the resulting stack returns identical completions across load is supported by a released repro, and equivalent batch-invariant modes have since appeared as opt-in flags in mainstream serving stacks *(frontier — verify current flag names and coverage)*.

**Established but narrower.** Deterministic-algorithm modes in cuDNN/cuBLAS and `torch.use_deterministic_algorithms(True)` guarantee run-to-run determinism at *fixed* shapes; they do not give batch invariance and were never claimed to.

**Theory SOTA.** Reproducible reduction is solved in the numerical-linear-algebra sense: Demmel and Nguyen's pre-rounding/indexed-summation gives order-independent sums with a small constant-factor cost (*Parallel Reproducible Summation*, IEEE Trans. Computers, 2015; ReproBLAS). No one has shown this is or is not the cheapest route for transformer inference on tensor cores — the LLM systems work instead fixes the split, which is cheaper but constrains parallelism.

**Claimed but unablated.** (i) That the throughput penalty is small at production batch sizes — public numbers are single-model, single-node microbenchmarks. (ii) That batch invariance is what fixes RL trainer/sampler mismatch — the RL demo is a single run, not an ablation isolating bitwise equality from lower-variance importance weights. (iii) MoE batch invariance — largely a benchmark-free claim.

## 4. What Is Known

Numbers, with the scale they were measured at:

- **Qwen3-235B-A22B, temperature 0, 1000 identical completions of one prompt, 1000 tokens each:** 80 distinct completions; the most common appeared 78 times; all 1000 shared a prefix and first diverged at token 103 (Thinking Machines Lab, 2025). With batch-invariant kernels: 1000/1000 identical, i.e. $R = 1.000$, $D = \infty$.
- **Throughput cost, Qwen3-8B, one H100 node, 1000 sequences × 1000 tokens:** ~26 s default vLLM, ~55 s unoptimised batch-invariant, ~42 s after attention-kernel improvement — $\rho \approx 0.47$ then $\approx 0.62$ (same source; single configuration, not a sweep).
- **Downstream effect:** in on-policy RL, bitwise-identical sampler and trainer drove the sampler–trainer KL to exactly $0$; without it, the reported run showed reward collapse mid-training. One run.
- **Independent evidence that "deterministic settings" are not deterministic:** Atil et al., *LLM Stability: A Detailed Analysis with Some Surprises* (2024), report accuracy spreads of several points across repeated greedy-decoding runs of the same model on the same benchmark — measured on open 7B–70B models via hosted APIs.
- **Non-associativity as root cause** is textbook and was quantified for GPUs well before LLMs (Villa et al., CUG 2009).
- MoE routing adds a second, non-numerical channel: batch-dependent expert capacity can change which expert a token reaches (Chann, 2023, on GPT-4 nondeterminism — a blog analysis, not a controlled study).

## 5. What Is Not Known

- **Empirically open.** The throughput cost of batch invariance across the real configuration space — $p \in \{1,2,4,8\}$, MoE vs dense, 8B to 500B+, prefill-heavy vs decode-heavy, with chunked prefill and prefix caching on. One published data point exists. Also open: whether $R=1.000$ survives adversarial load (preemption, ragged batches, spec-decode rejection) rather than benign load.
- **Theoretically open.** Whether batch invariance costs more than a constant factor. No lower bound relating achievable occupancy to a fixed reduction tree exists for tensor-core GEMM; conversely no proof that a fixed split is optimal among order-independent schemes.
- **Methodologically blocked.** What exactness is *worth*. There is no accepted metric mapping $\epsilon_\infty$ or $D$ to a downstream harm (eval variance, RL instability, reproducibility of a safety audit). Papers report accuracy spread, which confounds nondeterminism with benchmark sensitivity. Also blocked: MoE capacity-dropping — batch invariance may be unachievable without changing the model's semantics to drop-free routing, and no one has defined which of those two objects is the thing to make exact.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a semantic ambiguity**, not compute.

Confounding: to attribute a divergence to batch composition you must hold $\sigma$ fixed, but production servers vary $\sigma$ continuously — autotuner cache state, clock throttling, prefix-cache hits, and NCCL algorithm switches all move with load. A naive experiment that varies load varies six things at once, so a nonzero $U$ is uninformative about which fix would help.

Ambiguity: for MoE with capacity limits, the model's mathematical output *is* a function of the batch. Batch invariance then requires either drop-free routing (a different model) or per-sequence capacity (a throughput loss). There is no ground truth for "the correct output of this token" independent of the batch, so exactness is not merely unimplemented — it is under-specified.

## 7. Current Research (as of 2026)

- Thinking Machines Lab: batch-invariant kernel library, RL applications *(frontier — verify)*.
- vLLM and SGLang maintainers: opt-in deterministic/batch-invariant execution modes, and CI tests asserting bitwise equality across batch sizes *(frontier — verify coverage for MoE and $p>1$).*
- Numerical-reproducibility community (ReproBLAS lineage, Exablas): order-independent summation on accelerators; not yet fused into attention.
- RL-for-LLM groups: exploiting bitwise sampler/trainer equality to eliminate importance-correction terms, i.e. truly on-policy updates *(frontier — verify)*.
- Evaluation/reproducibility groups: seed- and load-controlled re-runs of standard benchmarks to separate nondeterminism from sensitivity.

## 8. Concrete Next Experiment

**Question.** Does batch invariance hold under adversarial load, and what does it cost?

**Scale.** Qwen3-8B (dense) and one MoE of comparable active parameters, on one 8×H100 node, tensor-parallel $p=8$, vLLM with chunked prefill and prefix caching **enabled**. Probe set: 200 distinct prompts × 100 repeats = 20,000 probe requests, 512 output tokens, temperature 0 *and* temperature 1 with a fixed per-request seed. Background load: Poisson arrivals driving in-flight batch size over $\{1,\dots,256\}$ with forced preemptions, so each probe sees a different neighbour multiset.

**Control arm.** Same binary, same load trace, batch-invariant kernels **off**. Second control: batch size pinned to 1 (serial), giving the reference output $y^\star$ per prompt.

**Deciding number.** $R^\star = \tfrac{1}{20000}\left|\{i : y^{(i)} = y^\star_{\text{prompt}(i)} \text{ bitwise}\}\right|$.
- Invariant arm must give $R^\star = 1.0000$ (any single mismatch falsifies the claim under realistic load).
- Report $\rho$ at matched p50 latency; the practical threshold is $\rho \ge 0.80$.
- If $R^\star < 1$, report $D$ and $\epsilon_\infty$ per subsystem by re-running with prefix caching off, then chunked prefill off, then $p=1$ — a 4-cell ablation that localises the residual.

Cost: roughly 40 GPU-hours per arm. Nothing about this is out of reach; it has not been run.

## 9. Key References

- **[SOTA]** Horace He and Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab: Connectionism, 2025. — accompanying code: `thinking-machines-lab/batch-invariant-ops`
- **[Foundational]** James Demmel and Hong Diep Nguyen. *Parallel Reproducible Summation.* IEEE Transactions on Computers, 2015.
- **[Foundational]** Oreste Villa, Daniel Chavarría-Miranda, Vidhya Gurumoorthi, Andrés Márquez, Sriram Krishnamoorthy. *Effects of Floating-Point Non-Associativity on Numerical Computations on Massively Multithreaded Systems.* Cray User Group (CUG), 2009.
- **[Empirical]** Berk Atil et al. *LLM Stability: A Detailed Analysis with Some Surprises.* 2024 (preprint).
- **[Systems]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023.
- **[Systems]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR, 2024. — the split-$K$/split-KV decomposition that makes attention batch-dependent.
- **[Systems]** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS, 2024.
- **[Context]** Sherman Chann. *Non-determinism in GPT-4 is caused by Sparse MoE.* Blog post, 2023. — hypothesis, not a controlled study.

## 10. Worked Example

Take one attention head, head dimension $d=128$, context length $n=1024$, bf16 accumulation into fp32. FlashDecoding splits the KV axis into $s$ chunks and combines partial softmax results. The scheduler picks $s$ to fill the GPU: at batch 1 it needs many splits ($s=16$), at batch 128 the batch already fills the machine ($s=1$).

Both compute $\sum_{j=1}^{1024} p_j v_j$, but with different reduction trees. For fp32 accumulation of $n$ terms the worst-case relative error grows like $n u$ with sequential order and $\log_2(n)\,u$ with a balanced tree, $u = 2^{-24}$. The *difference* between the two orders is what matters here, and empirically it is a few ULP: at a logit magnitude of $\sim 20$, one bf16 ULP is $2^{-8}\cdot 16 = 0.0625$; a fp32 reduction discrepancy of $\sim 10^{-5}$ is far below that.

So the per-step logit difference is invisible — until two candidate tokens are near-tied. Take $\ell_{\text{A}} = 12.34071$, $\ell_{\text{B}} = 12.34069$ at some step $t$. A $2\times10^{-5}$ reduction-order difference flips the argmax. Greedy decoding then emits a different token, and every subsequent step conditions on a different prefix.

This is the whole obstruction in one number. The measured Qwen3-235B run showed exactly this shape: all 1000 completions identical through token 102, then a fork at token 103, then 80 distinct trajectories by token 1000. The numerical error is $10^{-5}$ and never grows; the *output* error is unbounded because argmax is a discontinuous function of the logits and autoregression amplifies a single flip. Averaging, tolerance thresholds, and "close enough in float" all fail as fixes, because the quantity that must be preserved is a discrete decision, not a real number. That is why the only known solution is bitwise equality — and why the cost of bitwise equality (fixing $s$ regardless of batch, $\rho \approx 0.6$ in the one public measurement) is the real open question.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*