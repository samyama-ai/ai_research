---
id: 12-quantization-compression/hardware-faithful-compression-benchmarking
title: "Hardware-Faithful Compression Benchmarking"
topic: 12-quantization-compression
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hardware-Faithful Compression Benchmarking

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/hardware-faithful-compression-benchmarking` · **Status:** methodologically-blocked

## 1. Problem Statement

Compression papers report a *proxy* cost (bits per weight, parameter count, sparsity ratio, FLOPs) and a *proxy* quality (WikiText perplexity, a few zero-shot accuracies). Deployment cares about a *realized* cost (latency, throughput, energy, memory at a given batch size and sequence length on a named accelerator) and a *task* quality. The mapping between the two is neither monotone nor even sign-preserving: methods that cut bits can raise latency, and methods with identical perplexity can differ on downstream tasks.

**The problem.** Define a benchmarking protocol $B$ that takes a compressed model and returns a cost–quality pair such that ranking by $B$ predicts the ranking a deploying engineer would obtain, across at least two hardware generations and two serving regimes (latency-bound decode, throughput-bound batch).

Three variants, of different difficulty:

- **Measurement (the blocked one).** What is the right cost coordinate? It is a function of hardware, kernel maturity, batch size, and sequence length — a 4-tuple that papers usually leave implicit and unreported. Until the coordinate is pinned, "2× compression" is not a comparable quantity.
- **Method.** Given a fixed, faithful protocol, find the compression method on the Pareto frontier. This is ordinary empirical work and is not blocked.
- **Theory.** Prove bounds relating a hardware-independent complexity measure of a compressed network to achievable latency on a class of accelerators. Essentially untouched.

Solving it means: a published protocol where rerunning a paper's ranking on a different GPU, with equally mature kernels, reproduces the original ordering — and where the quality axis is a task metric that does not collapse the failure modes compression actually introduces.

## 2. Formal Setting

Let $f_\theta$ be a dense model, $C$ a compression operator, $\hat\theta = C(\theta)$ the compressed parameters, and $K$ a *kernel implementation* mapping $\hat\theta$ to executable code on device $h$.

**Cost, as measured.** Fix a workload $w = (B, L_{\text{in}}, L_{\text{out}})$ — batch size, prompt length, generation length. Realized latency is

$$T(\hat\theta, K, h, w) = \text{median}_{r=1}^{R} \; t_r,$$

with $R \geq 20$ timed runs after $\geq 5$ warmup runs, CUDA-graph-free, clocks locked, measured with device-side events (not wall clock around an async launch). Report the interquartile range; on unlocked clocks a thermally throttled A100 drifts 5–15%. Energy $E$ is the integral of board power over the same window from the on-device sensor (NVML `nvmlDeviceGetPowerUsage`), sampled at $\geq 100$ Hz — it omits host CPU and DRAM outside the package, so $E$ is a lower bound on system energy.

**The speedup that papers report.** Almost always

$$S_{\text{proxy}} = \frac{b_{\text{dense}}}{b_{\text{compressed}}} \quad\text{or}\quad S_{\text{FLOP}} = \frac{\text{FLOPs}(\theta)}{\text{FLOPs}(\hat\theta)},$$

whereas the deployable quantity is $S_{\text{real}} = T(\theta, K_{\text{dense}}, h, w) / T(\hat\theta, K, h, w)$.

**Arithmetic intensity decides which one is relevant.** For a GEMM of shape $(B \times d) \times (d \times d)$ at weight precision $b$ bits,

$$I = \frac{2Bd^2}{\;b d^2/8 + 2(Bd + Bd)\;} \approx \frac{16B}{b} \ \text{FLOP/byte at large } d.$$

The roofline crossover on an A100 (312 TFLOP/s BF16, 2.0 TB/s HBM) is $I^\star \approx 156$. So decode at $B{=}1$, $b{=}16$ gives $I \approx 1$: bandwidth-bound, and $S_{\text{real}} \to S_{\text{proxy}}$. At $B{=}256$, $b{=}4$, $I \approx 1024$: compute-bound, dequantization is pure overhead, and $S_{\text{real}}$ can fall below 1.

**Quality, as measured.** $Q = \mathbb{E}_{x \sim \mathcal{D}}[\ell(f_{\hat\theta}(x), y)]$. Perplexity uses $\ell = -\log p$ on WikiText-2 or C4; task metrics use exact-match or pass@1. The two disagree (§4).

**Assumptions known to be violated.**
1. *Kernel maturity is equal across arms.* False. A method with a hand-tuned CUDA kernel beats an equally good method with a PyTorch reference by 3–10×, and the paper reports this as a method difference.
2. *Latency is workload-independent.* False, by the roofline argument above.
3. *Perplexity is a sufficient statistic for quality.* False (§4).
4. *Compression is composable with serving-stack features.* False: paged KV cache, continuous batching, speculative decoding, and tensor parallelism each change which term dominates $T$.

## 3. State of the Art

**Established.**
- **MLPerf Inference** (Reddi et al., ISCA 2020) is the only benchmark with enforced accuracy floors — a submission must stay within 1% or 0.1% of the FP32 reference — plus fixed query distributions and audited runs. It is *hardware-faithful by construction* but covers a handful of frozen models and does not accept arbitrary research methods.
- **Kernel-aware quantization**: GPTQ (Frantar et al., ICLR 2023) with the MARLIN kernel (Frantar, Castro, Chen, Alistarh, PPoPP 2025) reports near-ideal ~4× INT4 speedup sustained up to batch ~16–32 on A100 — the clearest existing demonstration that the batch-size axis, not the bit-width, decides realized gain. AWQ/TinyChat (Lin et al., MLSys 2024) reports ~3× over FP16 on A100 and desktop/edge GPUs.
- **Confounder documented**: "The Efficiency Misnomer" (Dehghani et al., ICLR 2022) shows FLOPs, parameter count, and throughput induce *different* model rankings, and that reporting one is not evidence about the others.
- **Benchmarking failure documented in pruning**: Blalock et al., "What is the State of Neural Network Pruning?" (MLSys 2020) surveyed 81 papers and found only a small minority compared against each other on a common setup; they released ShrinkBench in response.

**Claimed but unablated.**
- Most "X% speedup" claims in quantization papers are single-GPU, single-batch-size, against an FP16 baseline that is not the fastest available FP16 path. Whether the ordering survives a kernel-parity control has not been ablated.
- Energy claims are almost always derived from bit-width, not measured. Cross-vendor energy comparisons (NVIDIA vs. AMD vs. TPU vs. NPU) exist mainly as vendor benchmark numbers.

**Benchmark-number-only.** Aggregate quantization leaderboards (e.g. LLMC, Gong et al., EMNLP 2024 industry track) standardize *method* configuration but not *kernel* or *workload*; their throughput columns are not parity-controlled.

## 4. What Is Known

- **4-bit is Pareto-optimal per bit for zero-shot accuracy.** Dettmers & Zettlemoyer (ICML 2023), ~35,000 runs over 19M–176B parameters: at fixed total bits, 4-bit maximizes accuracy. This is a *bit-count* result, not a latency result.
- **Bit reduction can cost latency.** LLM.int8() (Dettmers et al., NeurIPS 2022) reports the mixed-precision decomposition running *slower* than FP16 for small models; the win appears only at 6.7B+ where memory dominates.
- **Structured sparsity underdelivers against its ratio.** 2:4 sparsity halves weight bytes but NVIDIA's own results (Mishra et al., 2021) report layer-level speedups typically ~1.3–1.8× on A100 Tensor Cores, not 2×, and end-to-end gains smaller still.
- **Perplexity hides task damage.** "Accuracy is Not All You Need" (Dutta et al., NeurIPS 2024) shows compressed models matching baseline aggregate accuracy while flipping a large fraction of individual answers, i.e. per-example disagreement is far higher than the accuracy delta suggests. Perplexity deltas under 0.1 coexist with multi-point drops on reasoning and code tasks.
- **Calibration set choice moves results.** GPTQ-family results shift measurably with the 128-sample calibration corpus; papers rarely report variance over calibration draws.

## 5. What Is Not Known

**Methodologically blocked (the core gap).**
- No accepted definition of *kernel parity*. Comparing method A with an optimized kernel to method B without one is the field's default, and there is no agreed control (a shared reference kernel? a compute-budget cap on kernel engineering? a roofline-normalized upper bound?).
- No agreed cost coordinate. Latency at which $(B, L_{\text{in}}, L_{\text{out}})$? A single scalar cannot summarize a surface; nobody has defined the standard slice.
- No quality metric that captures the *behavioral* shift compression induces. Aggregate accuracy provably does not (Dutta et al.); per-example flip rate is a candidate but has no accepted threshold.

**Empirically open.** Whether published method rankings survive a kernel-parity, multi-hardware rerun. The experiment is runnable today (§8) and has not been run.

**Theoretically open.** Any bound of the form "a network with structure $s$ cannot achieve latency below $L(s, h)$ on accelerator class $h$." Roofline gives a loose necessary condition; nothing tighter and method-general exists.

## 6. Why It Is Hard

**Confounded measurement, with the confounder being engineering effort.** The realized speedup of a compression method is a product of two factors that the literature never separates:

$$S_{\text{real}} = \underbrace{S_{\text{struct}}}_{\text{what the format allows}} \times \underbrace{\eta_K}_{\text{kernel efficiency, } 0 < \eta_K \le 1}.$$

$\eta_K$ is a function of person-months of CUDA work, not of the method. GPTQ's structure did not change between 2023 and MARLIN; $\eta_K$ did, and end-to-end speedup roughly tripled. No experiment currently distinguishes "better format" from "better-funded kernel," and $S_{\text{struct}}$ is not separately identifiable from a single end-to-end timing.

Secondary: the cost surface is 4-dimensional (hardware × batch × sequence × serving stack) and the sign of $\partial S_{\text{real}} / \partial B$ flips at the roofline crossover, so any scalar summary is a choice of slice that authors make in their own favor.

## 7. Current Research (as of 2026)

- **Kernel-format co-design** — IST Austria (Alistarh) on MARLIN/Sparse-MARLIN, MIT HAN Lab (Han) on AWQ/TinyChat/QServe; these treat the kernel as part of the method, which is honest but makes cross-method comparison harder, not easier.
- **Serving-stack-integrated evaluation** — vLLM (Kwon et al., SOSP 2023) and SGLang now ship quantized-kernel paths, making parity-controlled end-to-end measurement newly feasible inside one runtime. *(frontier — verify)*
- **Behavioral divergence metrics** — flip-rate and KL-to-baseline as replacements for aggregate accuracy, following Dutta et al. Not yet standardized.
- **MLPerf Inference** continues adding LLM workloads with accuracy floors; the open question is whether the research community will adopt its discipline outside of vendor submissions. *(frontier — verify)*

## 8. Concrete Next Experiment

**The kernel-parity ablation.**

- **Scale.** Two models (Llama-3.1-8B, Llama-3.1-70B), three methods at nominal 4 bits (GPTQ, AWQ, and round-to-nearest with group size 128), two GPUs of different generations (A100-80GB, H100-80GB), inside one runtime (vLLM), across a batch grid $B \in \{1, 4, 16, 64, 256\}$ at $L_{\text{in}}{=}1024$, $L_{\text{out}}{=}128$.
- **Control arm (the point of the experiment).** Every method is run twice: (a) with its own author-supplied kernel, and (b) with one *shared* dequant-GEMM kernel that all three formats can use, so $\eta_K$ is held constant across methods. Baseline is the fastest available FP16 path in the same runtime, not a naive one.
- **The deciding number.** Kendall's $\tau$ between the method ranking under arm (a) and arm (b), averaged over the 10 (GPU, batch) cells. $\tau \geq 0.8$ means published rankings are about the format and the field's protocol is roughly sound. $\tau \leq 0.4$ means published rankings are mostly measuring kernel engineering, and the benchmarking protocol must be rewritten to report $S_{\text{struct}}$ and $\eta_K$ separately.
- **Cost.** ~200 GPU-hours. This is small; it has not been run because no venue rewards it.

## 9. Key References

- **[Foundational]** V. J. Reddi, C. Cheng, D. Kanter, et al. *MLPerf Inference Benchmark.* ISCA, 2020. — arXiv:1911.02549
- **[Foundational]** D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, J. Guttag. *What is the State of Neural Network Pruning?* MLSys, 2020. — arXiv:2003.03033
- **[Foundational]** M. Dehghani, A. Arnab, L. Beyer, A. Vaswani, Y. Tay. *The Efficiency Misnomer.* ICLR, 2022. — arXiv:2110.12894
- **[SOTA]** E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** E. Frantar, R. L. Castro, J. Chen, T. Hoefler, D. Alistarh. *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models.* PPoPP, 2025.
- **[SOTA]** J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, S. Han. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** T. Dettmers, L. Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Evaluation]** A. Dutta, S. Krishnan, N. Kwatra, R. Ramjee. *Accuracy is Not All You Need.* NeurIPS, 2024. — arXiv:2407.09141
- **[Systems]** W. Kwon, Z. Li, S. Zhuang, et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Survey]** T. Hoefler, D. Alistarh, T. Ben-Nun, N. Dryden, A. Peste. *Sparsity in Deep Learning: Pruning and Growth for Efficient Inference and Training in Neural Networks.* JMLR 22(241), 2021. — arXiv:2102.00554
- **[Survey]** A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, K. Keutzer. *A Survey of Quantization Methods for Efficient Neural Network Inference.* 2021. — arXiv:2103.13630

## 10. Worked Example

Llama-3.1-8B, INT4 weights with FP16 activations, A100-80GB (2.0 TB/s HBM, 312 TFLOP/s BF16). Weights: 16 GB at FP16, 4 GB at INT4.

**Decode, $B{=}1$.** Time is bandwidth-bound. Per token:

$$T_{16} \approx \frac{16\ \text{GB}}{2.0\ \text{TB/s}} = 8.0\ \text{ms}, \qquad T_{4} \approx \frac{4\ \text{GB}}{2.0\ \text{TB/s}} = 2.0\ \text{ms}.$$

Predicted $S_{\text{real}} = 4.0$ — matching $S_{\text{proxy}} = 16/4$. The paper's headline is honest *here*.

**Prefill / large batch, $B{=}256$.** Arithmetic intensity $I \approx 16 \cdot 256 / 4 = 1024 \gg I^\star = 156$: compute-bound. Weight bytes no longer gate anything. INT4 weights must be dequantized to FP16 before the Tensor Core GEMM, so the *only* effect of compression is added dequant work. With a naive kernel this costs 20–60% extra time; $S_{\text{real}} \approx 0.6$–$0.8$. With MARLIN-class asynchronous dequantization overlapped with the GEMM, $S_{\text{real}} \approx 1.0$.

**Where the obstruction becomes visible.** Two papers report the same method. Paper A benchmarks at $B{=}1$ and reports "4× faster." Paper B benchmarks at $B{=}256$ and reports "1.3× slower." Both are correct measurements of the same $\hat\theta$. Now add a second method with $S_{\text{proxy}} = 16/3$ (3-bit) but only a reference PyTorch kernel: at $B{=}1$ it measures 1.4× *slower* than the 4-bit method despite strictly fewer bytes, because $\eta_K \approx 0.25$ against $\eta_K \approx 0.95$.

From the end-to-end timings alone, $S_{\text{struct}}$ and $\eta_K$ are not separately recoverable — a low measured speedup is consistent with a bad format *and* with a good format plus an unoptimized kernel. That is the non-identifiability, and it is why the protocol, not the methods, is what needs fixing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*