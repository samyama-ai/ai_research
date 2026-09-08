---
id: 11-inference-and-serving/cross-hardware-output-reproducibility
title: "Reproducibility of Served Model Outputs Across Hardware"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Reproducibility of Served Model Outputs Across Hardware

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/cross-hardware-output-reproducibility` · **Status:** open

## 1. Problem Statement

Fix a model checkpoint $\theta$, a prompt $x$, a decoding rule (greedy, or sampling with a fixed seed), and everything a user can control through an API. Run it on two different serving stacks — different GPU model, different kernel library version, different tensor-parallel degree, different batch composition. Do you get the same token sequence?

Today, usually not. Three variants, with very different difficulty:

- **Measurement.** Define a distance on served outputs that separates "bitwise different" from "behaviourally different", and estimate it with bounded sample cost. Currently ad hoc.
- **Method.** Build a serving stack whose output is invariant to hardware, batch composition, and scheduling, at acceptable throughput cost. Partially solved for single-hardware, batch-invariant cases; unsolved across device families.
- **Theory.** Given a transformer's arithmetic and a decoding rule, bound the probability that two IEEE-754-conformant but differently-ordered evaluations diverge within $T$ tokens. Open.

A solution to the method variant: a serving configuration $\mathcal{S}$ such that for all $x$ in a held-out prompt set, $\mathcal{S}$ on hardware $H_1$ and $H_2$ emits identical token IDs, with measured throughput loss stated.

## 2. Formal Setting

Let $f_\theta: \mathcal{V}^{*} \to \mathbb{R}^{|\mathcal{V}|}$ be the exact real-arithmetic logit map. A serving stack is a triple $c = (H, K, B)$: hardware $H$, kernel/library configuration $K$ (cuBLAS/CUTLASS version, attention implementation, TP/PP degree, dtype), and batch context $B$ (the other requests co-resident in the step, plus KV-cache chunking and scheduler state). The realised map is $\hat f_\theta^{\,c}$.

**Numerical deviation, as measured.** For prompt $x$ at decode step $t$, run both stacks on the *same* prefix (teacher-forced, so divergence does not compound) and record

$$\Delta_t(x; c_1, c_2) \;=\; \bigl\lVert \hat f_\theta^{\,c_1}(x_{<t}) - \hat f_\theta^{\,c_2}(x_{<t}) \bigr\rVert_\infty .$$

**Decision margin.** With $z = \hat f^{\,c_1}_\theta(x_{<t})$ sorted descending, the greedy margin is $\delta_t = z_{(1)} - z_{(2)}$. A greedy flip at step $t$ occurs iff $\delta_t < $ the realised logit perturbation on the argmax pair; a sufficient condition for *no* flip is $\delta_t > 2\Delta_t$.

**Free-running divergence.** Without teacher forcing, define $T^{*}(x) = \min\{t : y_t^{c_1} \neq y_t^{c_2}\}$, and the reproducibility rate over a prompt set $\mathcal{X}$ at horizon $T$:

$$R_T = \frac{1}{|\mathcal{X}|}\sum_{x\in\mathcal{X}} \mathbf{1}\!\left[T^{*}(x) > T\right].$$

**Behavioural equivalence** is separate: $|\mathbb{E}[\text{score}(y^{c_1})] - \mathbb{E}[\text{score}(y^{c_2})]|$ on a benchmark, with CIs from bootstrap over prompts. $R_T = 0$ with zero score gap is common and is exactly why bitwise metrics alone mislead.

**Unit roundoff.** For accumulation dtype with unit roundoff $u$ (fp32: $u = 2^{-24} \approx 5.96\times10^{-8}$; fp16: $2^{-11}$; bf16: $2^{-9} \approx 1.95\times10^{-3}$), recursive summation of $n$ terms carries the classical bound $\gamma_n = nu/(1-nu)$ (Higham 2002); pairwise/tree summation gives $O(u\log_2 n)$. Different reduction *trees* — which is what changes when the split-K factor, warp tiling, or TP degree changes — give different results at exactly this magnitude.

**Assumptions, and which are violated.**
1. *Both stacks are IEEE-754 conformant with correctly-rounded FMA.* Mostly holds; violated by TF32 tensor-core paths (10-bit input mantissa) and by fast-math/approximate transcendental units used in softmax and GELU.
2. *Addition is associative.* False. This is the root cause, not an edge case.
3. *The batch context $B$ does not enter the per-request function.* False for batched GEMM and split-attention: the reduction order depends on batch shape.
4. *Weights are bit-identical across stacks.* Violated whenever quantization is done per-deployment (calibration data, ordering in GPTQ/AWQ) rather than shipped as a fixed artifact.
5. *Expert routing is request-local.* False for MoE with cross-request capacity limits — a co-resident request can evict another's token from its expert.

## 3. State of the Art

**Established.**
- *Batch invariance eliminates single-hardware nondeterminism.* Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (2025), identifies batch-size-dependent reduction order — not GPU atomics — as the dominant cause of temperature-0 nonreproducibility in production serving, and ships batch-invariant RMSNorm/matmul/attention kernels for vLLM. Reported: 1000 temperature-0 completions of the same prompt on Qwen3-235B gave 80 distinct outputs by default, and 1000/1000 identical with batch-invariant kernels. This is a same-hardware result; cross-hardware is not claimed.
- *Reproducible reduction is a solved numerical problem in isolation.* Demmel & Nguyen, "Fast Reproducible Floating-Point Summation" (ARITH 2013) and Collange et al. (Parallel Computing, 2015) give order-independent summation via pre-rounding / long accumulators, at roughly constant-factor cost on memory-bound reductions.
- *Deterministic-algorithm modes exist and are documented as per-device only.* cuBLAS guarantees run-to-run bitwise reproducibility only for a fixed architecture, library version, and identical shapes/handle configuration; PyTorch `use_deterministic_algorithms(True)` carries the same scope caveat.

**Claimed but unablated.**
- That MoE routing is *the* cause of API nondeterminism (Chann, 2023, blog) — a plausible mechanism, but it was never separated from batch-dependent reduction order, which the 2025 work shows is sufficient on its own for dense models.
- Vendor "deterministic inference" marketing: no public artifact demonstrates bitwise equality across, say, H100 and MI300X for the same checkpoint.

**Benchmark-number-only.** OpenAI's `seed` + `system_fingerprint` (API, 2023) offers "mostly deterministic" outputs with no bound and an explicit fingerprint-change escape hatch. There is no published $R_T$ for it.

## 4. What Is Known

- **Reduction order is the dominant lever.** Same GPU, same weights, batch size 1 vs. 64 changes the split-K/tile decomposition and thus the summation tree; the resulting logit perturbation is at the $\gamma_n u$ scale. For $n = 8192$ (Llama-3-70B hidden width) in fp32 accumulate, $nu \approx 4.9\times10^{-4}$ — comparable to the smallest greedy margins seen in practice.
- **Scale of the flip.** Reported temperature-0 divergence in the Qwen3-235B experiment above began as early as token ~100 in individual runs, then compounded.
- **Training-side analogue is measured.** Zhuang, Zhang, Song & Hooker, "Randomness in Neural Network Training: Characterizing the Impact of Tooling" (MLSys 2022) and Pham et al. (ASE 2020) show that tooling-induced nondeterminism alone produces accuracy spreads on the order of a benchmark's own noise floor at ImageNet/CIFAR scale — i.e. large enough to flip a model-selection decision.
- **Compression moves the tail, not the mean.** Hooker et al., "What Do Compressed Deep Neural Networks Forget?" (2019): pruning/quantization at matched top-1 accuracy changes predictions disproportionately on a small, identifiable exemplar subset. The same structure is expected for cross-hardware kernel differences and explains why aggregate scores hide divergence.
- **Hardware differences are not just rounding.** TF32 (19-bit inputs), differing FMA fusion decisions by the compiler, and vendor-specific transcendental approximations for $\exp$ in softmax put cross-vendor deviation well above the fp32 $u$ floor.

## 5. What Is Not Known

- **Theoretically open.** No bound on $\Pr[T^{*} < T]$ as a function of $\Delta$, the margin distribution of $\delta_t$, and $T$. The needed ingredient — the tail of $\delta_t$ near zero for a real LLM — has no analytic model, and the process is not i.i.d. across $t$ (divergence is self-reinforcing).
- **Empirically open.** No published cross-vendor study reporting $R_T$ for a fixed checkpoint on NVIDIA vs. AMD vs. TPU vs. CPU, holding dtype, TP degree, and kernel semantics fixed. This is runnable today for under $2{,}000$ of GPU-hours; nobody has published it.
- **Empirically open.** The throughput cost of *cross-hardware* reproducibility. The batch-invariant single-hardware cost is reported as substantial but recoverable; the cost of order-independent reduction plus a common transcendental library across vendors is unmeasured.
- **Methodologically blocked.** "Behaviourally equivalent" has no accepted definition. Token-identity is too strict (paraphrases score identically); benchmark-mean equality is too loose (it passes stacks that disagree on 30% of individual answers). Nobody has fixed a divergence measure with a stated decision use.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth**. There is no reference output: the exact real-arithmetic $f_\theta$ is not computable, so neither stack is "right", and every comparison is between two equally-arbitrary roundings. Worse, in a production serving stack you cannot hold $B$ fixed — batch composition depends on other users' traffic — so the treatment variable is not under the experimenter's control unless the experiment is run offline, which changes the system being measured. Layered on top: a single flipped token at step $t$ makes all subsequent tokens incomparable, so free-running divergence measures *one* event with unbounded downstream amplification, while teacher-forced $\Delta_t$ measures the arithmetic cleanly but cannot predict $R_T$ without the unknown margin tail.

## 7. Current Research (as of 2026)

- Batch-invariant kernel work in vLLM and SGLang, seeded by the Thinking Machines batch-invariant ops release; the extension to attention with paged/chunked KV is the active edge *(frontier — verify)*.
- Reproducible-RL motivation: on-policy RL becomes off-policy when the sampler and trainer disagree numerically; several labs cite bitwise sampler/trainer agreement as a training-stability requirement *(frontier — verify)*.
- Compliance pressure. EU AI Act model-documentation duties and financial/medical audit requirements push toward "same input, same output, re-runnable years later", which cross-hardware drift breaks *(frontier — verify)*.
- Numerical-analysis side: reproducible BLAS (ReproBLAS lineage) has not been ported to transformer-shaped GEMMs at production throughput.

## 8. Concrete Next Experiment

**Question.** How much of cross-hardware divergence is reduction order versus everything else?

**Scale.** Llama-3.1-8B-Instruct, bf16 weights, fp32 accumulate, TP=1, 2000 prompts from a mixed set (500 each: GSM8K, HumanEval, MMLU stems, long-form open-ended), max 512 new tokens, greedy.

**Arms.**
1. *Control:* H100, vLLM default, batch size 1, run twice. Establishes the noise floor; should give $R_{512} = 1.0$.
2. H100, default kernels, batch size 64 with fixed co-resident filler.
3. A100, default kernels, batch size 1.
4. MI300X (ROCm), default kernels, batch size 1.
5. Arms 3 and 4 with batch-invariant + order-independent reduction kernels and a shared software `exp`/`erf` implementation.

**Measurements.** $R_{512}$ per arm; teacher-forced $\Delta_t$ percentiles; the empirical CDF of $\delta_t$ near zero; benchmark score deltas with bootstrap CIs; tokens/s.

**The deciding number.** $R_{512}$ for arm 5 (cross-vendor, hardened kernels). If $R_{512} \geq 0.99$ at under a 30% throughput loss, cross-hardware reproducibility is an engineering matter and the problem downgrades to *partially-solved*. If $R_{512} < 0.5$, non-associativity is not the whole story and transcendental/FMA-fusion differences must be attacked separately — the problem stays open, with a sharper target.

## 9. Key References

- **[Foundational]** Nicholas J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002. — summation error bounds $\gamma_n$, reduction-order sensitivity.
- **[Foundational]** David Goldberg. *What Every Computer Scientist Should Know About Floating-Point Arithmetic.* ACM Computing Surveys, 1991.
- **[SOTA]** Horace He and Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference.* Thinking Machines Lab: Connectionism, 2025. — batch-invariant kernels; Qwen3-235B temperature-0 experiment.
- **[SOTA]** James Demmel and Hong Diep Nguyen. *Fast Reproducible Floating-Point Summation.* IEEE Symposium on Computer Arithmetic (ARITH), 2013.
- **[SOTA]** Sylvain Collange, David Defour, Stef Graillat, Roman Iakymchuk. *Numerical reproducibility for the parallel reduction on multi- and many-core architectures.* Parallel Computing, 2015.
- **[Empirical]** Donglin Zhuang, Xingyao Zhang, Shuaiwen Leon Song, Sara Hooker. *Randomness in Neural Network Training: Characterizing the Impact of Tooling.* MLSys, 2022.
- **[Empirical]** Hung Viet Pham et al. *Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance.* ASE, 2020.
- **[Empirical]** Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin, Andrea Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- **[Context]** Nathan Whitehead and Alex Fit-Florea. *Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs.* NVIDIA technical whitepaper, 2011.
- **[Survey]** Odd Erik Gundersen and Sigbjørn Kjensmo. *State of the Art: Reproducibility in Artificial Intelligence.* AAAI, 2018.

## 10. Worked Example

Llama-3.1-8B, greedy, hidden width $d = 4096$, vocabulary $|\mathcal{V}| = 128{,}256$.

Take one logit. It is an inner product of length 4096 in bf16 inputs with fp32 accumulation. Two stacks split that reduction differently (split-K = 1 vs. split-K = 4). The bound on the difference between the two summation trees is roughly $2\gamma_{4096}\sum|a_ib_i|$ with $\gamma_{4096} \approx 4096 \times 5.96\times 10^{-8} = 2.4\times 10^{-4}$. With final-layer activations giving $\sum|a_ib_i| \approx 40$ for a typical logit of magnitude ~15, the deviation is $\Delta \approx 10^{-2}$ in the worst case and $\sim 10^{-4}$ in the RMS case. Take $\Delta = 3\times 10^{-4}$.

Now the margin. A greedy flip needs $\delta_t < 2\Delta = 6\times10^{-4}$. On a code or math prompt, most steps are decided: $\delta_t > 1$. But at every genuine branch point — a coin-flip between "The" and "A", a tie between two equally good variable names — $\delta_t$ lands in a near-zero band. Suppose the fraction of steps with $\delta_t < 6\times10^{-4}$ is $p = 10^{-3}$ (one step in a thousand).

Over $T = 512$ tokens:

$$\Pr[\text{at least one flip}] = 1 - (1-10^{-3})^{512} = 0.40 .$$

So $R_{512} \approx 0.60$ — 40% of 512-token completions diverge, from an arithmetic difference at the eighth significant digit. And the *benchmark score* of both stacks will be statistically indistinguishable, because the flip usually occurs at a step where both continuations are fine.

That is the obstruction in one line: the quantity that changes is $10^{-4}$, the quantity that decides is $\delta_t$, and nobody has measured the distribution of $\delta_t$ near zero — so $p$, and therefore $R_T$, is currently a guess.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*