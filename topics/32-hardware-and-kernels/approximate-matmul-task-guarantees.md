---
id: 32-hardware-and-kernels/approximate-matmul-task-guarantees
title: "Approximate Matrix Multiply With Provable End-Task Guarantees"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Approximate Matrix Multiply With Provable End-Task Guarantees

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/approximate-matmul-task-guarantees` · **Status:** open

## 1. Problem Statement

Approximate matrix multiplication (AMM) — sampling, sketching, quantization, structured sparsity, product-quantized lookup — buys throughput by returning $\widehat{AB} \neq AB$. Every deployed method is validated by *benchmark degradation*: run the model, look at perplexity or accuracy, ship if the drop is small. No deployed method carries a guarantee of the form "for any input in distribution $\mathcal{D}$, the end-task metric degrades by at most $\delta$."

The problem: **construct an AMM scheme plus an accompanying certificate that bounds end-task loss, not kernel-level residual norm, at a speedup that beats exact hardware matmul.**

Three variants, of sharply different difficulty:

- **Measurement.** Define a per-layer error functional whose value predicts end-task degradation across inputs. Currently open — Frobenius residual correlates weakly with task loss.
- **Method.** Build an approximator that is *fast on real tensor-core hardware* and whose error is controlled in the quantity the measurement variant identifies.
- **Theory.** Prove a composition theorem: per-layer error $\epsilon_\ell$ across $L$ layers implies end-task degradation $\delta(\epsilon_1,\dots,\epsilon_L)$ that is non-vacuous for a trained transformer.

Conflating these is why "approximate matmul" has 20 years of tight theory and zero end-task guarantees.

## 2. Formal Setting

Let $f_\theta: \mathcal{X} \to \mathcal{Y}$ be a network with $L$ matmul layers, $\theta = \{W_\ell\}$, $W_\ell \in \mathbb{R}^{d_\ell \times d_{\ell+1}}$. Task loss $\mathcal{L}(f) = \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x),y)]$, measured as mean cross-entropy over a held-out set of $N$ tokens; report the standard error $\hat\sigma/\sqrt{N}$, since claimed degradations are often smaller than it.

An AMM operator $\mathcal{A}_\ell$ replaces $XW_\ell$ with $\mathcal{A}_\ell(X, W_\ell)$. Write $\tilde f_\theta$ for the perturbed network. The quantity to bound:

$$\delta = \mathcal{L}(\tilde f_\theta) - \mathcal{L}(f_\theta).$$

**Kernel-level error**, as reported in the AMM literature:

$$E_F(\ell) = \frac{\|\mathcal{A}_\ell(X,W_\ell) - XW_\ell\|_F}{\|X\|_F\|W_\ell\|_F}.$$

**Cost**, measured not in FLOPs but in achieved wall-clock: $T$ = median kernel latency over 100 runs at fixed clocks, on a named device, against the vendor library (cuBLAS/CUTLASS) at the same shape. Arithmetic-op counts are not a proxy — a method with $4\times$ fewer ops that abandons tensor cores is typically *slower*.

**Sensitivity.** A composition bound requires layer-wise Lipschitz constants $\mathrm{Lip}(g_\ell)$ of the sub-network above layer $\ell$, giving the naive chain

$$\delta \;\le\; \mathrm{Lip}(\ell(\cdot,y)) \sum_{\ell=1}^{L} \Big(\prod_{k>\ell}\mathrm{Lip}(g_k)\Big)\,\|\Delta_\ell\|.$$

Assumptions and their status:

- *Errors are independent across layers.* Violated: quantization error correlates with activation outliers, which persist across depth (Dettmers et al., 2022).
- *Products of Lipschitz constants are informative.* Violated: exact Lipschitz computation is NP-hard (Virmaux & Scaman, 2018); certified upper bounds for transformers exceed $10^{10}$, making the chain vacuous.
- *Input distribution is fixed.* Violated at deployment: calibration sets used to fit quantizers are typically $10^2$–$10^3$ sequences of web text, and errors grow on out-of-distribution prompts.
- *The residual is zero-mean and unbiased.* Violated by rounding-based quantizers, which are biased; bias accumulates linearly in depth while variance accumulates as $\sqrt{L}$.

## 3. State of the Art

**Theory SOTA (established).** Drineas, Kannan & Mahoney (SICOMP 2006) give column-sampling $\widehat{AB}=CR$ with
$$\mathbb{E}\|AB - CR\|_F \le \|A\|_F\|B\|_F/\sqrt{c}$$
for $c$ sampled columns under optimal (norm-proportional) probabilities. Sketching via JL/subsampled-Hadamard transforms (Ailon & Chazelle 2009; Clarkson & Woodruff, STOC 2013) achieves subspace embeddings in input-sparsity time. Martinsson & Tropp (*Acta Numerica* 2020) is the consolidated account. All of these are *relative-to-$\|A\|_F\|B\|_F$* additive bounds — none is a relative-error bound per entry, and none is composed through a nonlinearity.

**Systems SOTA (established by ablation).** INT8 weight+activation with outlier handling (LLM.int8(), NeurIPS 2022) at 175B; GPTQ 3–4 bit post-training quantization (ICLR 2023); AWQ (MLSys 2024, best-paper); FP8 training formats (Micikevicius et al., 2022) now native on H100/B200. NVIDIA 2:4 structured sparsity (Mishra et al., 2021) gives up to $2\times$ math throughput with retraining. These have accuracy tables, hardware kernels, and independent reproductions — but no certificates.

**Claimed but unablated / benchmark-only.** MADDNESS (Blalock & Guttag, ICML 2021) reports up to $100\times$ over exact GEMM and $10\times$ over prior AMM on CPU, but is evaluated on single layers and small classifiers, not composed end-to-end in a large model; its speedups do not transfer to tensor-core GPUs. Monarch matrices (Dao et al., ICML 2022) have structural approximation theory but end-task results are benchmark numbers on specific architectures. Sparse-attention approximations (BigBird, NeurIPS 2020) carry a universal-approximation theorem that says nothing about the *trained* model's loss under substitution.

## 4. What Is Known

- **Sampling error is $\Theta(1/\sqrt{c})$ and dimension-independent** for $c$ samples in Frobenius norm (Drineas et al. 2006). Tight; no method beats it without structure.
- **8-bit inference is empirically free at scale.** LLM.int8() matches FP16 perplexity on OPT-175B; measured degradation below evaluation noise on a few hundred million tokens.
- **4-bit is nearly free with second-order correction.** GPTQ on OPT-175B reports perplexity increases of roughly $0.1$–$0.3$ nats-equivalent at 3–4 bits versus FP16, on WikiText-2/C4, at 175B scale, in a few GPU-hours.
- **Outliers dominate.** In models above ~6.7B parameters, a small set of feature dimensions carry activations $\sim 20\times$ larger than the median and account for most quantization damage (Dettmers et al., 2022) — a scale-dependent phenomenon absent below ~2.7B.
- **2:4 sparsity recovers accuracy only with retraining**, verified across dozens of vision and language networks (Mishra et al., 2021).
- **Exact Lipschitz estimation is NP-hard**; LipSDP (Fazlyab et al., NeurIPS 2019) gives tractable bounds for small MLPs, not transformers at depth 32+.

## 5. What Is Not Known

- **Theoretically open.** Whether any non-vacuous end-task bound exists for a trained transformer under per-layer $\epsilon$ perturbation. No proof either way; the only candidate route, Lipschitz composition, is known to be loose by many orders of magnitude but has no matching lower-bound construction showing looseness is necessary.
- **Methodologically blocked.** The right per-layer error functional. $E_F$ demonstrably fails: layers with equal $E_F$ produce end-task deltas differing by more than $10\times$. No accepted alternative (per-token relative error? logit-gap-relative error? curvature-weighted error?) has been validated as *predictive* rather than merely correlated.
- **Empirically open.** Whether sublinear-sample AMM (sampling/sketching, not quantization) can beat a tensor-core FP8 GEMM in wall-clock at transformer shapes ($M{\times}K{\times}N$ with $K \le 8192$) with bounded end-task loss. Runnable today on one node; nobody has published the head-to-head at fixed clocks.
- **Empirically open.** Worst-case rather than average-case degradation: all published deltas are averages over benchmark sets. Adversarial-prompt degradation under 4-bit quantization is unmeasured at frontier scale.

## 6. Why It Is Hard

The obstruction is **norm-blindness**: every classical AMM guarantee bounds error relative to $\|A\|_F\|B\|_F$, while the task-relevant signal lives in *gaps between near-tied scalars* — attention logits, softmax margins, argmax over the vocabulary. Those gaps are $O(10^{-2})$ of the norms involved. A bound that is tight in Frobenius norm is therefore vacuous for the decision the layer feeds, and the sample count required to control the gap exceeds the inner dimension $K$ itself (see §10) — at which point exact multiplication is cheaper.

Second obstruction: **evaluation does not measure what it names.** "No degradation" typically means a perplexity delta smaller than the standard error of the eval set, over $10^5$–$10^6$ tokens. That is consistent with a catastrophic failure rate of $10^{-4}$ per token, which is unacceptable for agentic use but invisible to the metric.

## 7. Current Research (as of 2026)

- **Quantization with error compensation** — rotation/incoherence preprocessing (QuIP, NeurIPS 2023; QuaRot and successors) is the most active line; it makes weight matrices incoherent so that uniform rounding bounds become tight. This is the closest existing thing to a *provable* bridge, because incoherence gives per-entry, not per-norm, control. *(frontier — verify whether any 2025–26 paper closes the loop to end-task loss.)*
- **Hardware-native microscaling formats** (MXFP4/MXFP6, OCP spec; Blackwell FP4 tensor cores) shift the question from "can we approximate" to "which approximation the silicon already implements" — sketching methods must now beat FP4, not FP16.
- **Certified inference** — small groups apply randomized smoothing and interval bound propagation to quantized models; scale is limited to sub-1B models. *(frontier — verify.)*
- **Sublinear-time attention** with data-dependent sampling (KDEformer-style estimators, Hyperattention) offers $\epsilon$-guarantees on the attention *matrix* — again the wrong object. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question:** does any per-layer error functional predict end-task degradation well enough to serve as a certificate?

- **Scale.** One 7B decoder-only model (e.g. Llama-3-8B class), 32 layers, on a single H100. Evaluation: 4M held-out tokens of C4 plus 3 downstream tasks. Cost: ~200 GPU-hours.
- **Design.** Generate 400 perturbation configurations by applying, per layer independently, one of {column sampling at $c/K \in \{0.25,0.5,0.75\}$, INT4 RTN, INT4 GPTQ, 2:4 mask, FP8-E4M3} at random. For each config record: $E_F$ per layer, three candidate functionals (per-token relative $\ell_2$; logit-gap-normalized error at the attention layer; Fisher-weighted error $\Delta^\top \hat F \Delta$ using a diagonal empirical Fisher), and the measured $\delta$.
- **Control arm.** Exact FP16 forward, plus a *random-direction* perturbation of matched $E_F$ per layer — this separates "error magnitude matters" from "error *direction* matters."
- **Deciding number.** Spearman $\rho$ between predicted and measured $\delta$ across the 400 configs, and — the harder bar — the **maximum ratio of measured $\delta$ to predicted bound**. A functional with $\rho > 0.9$ *and* bound-to-truth ratio within $10\times$ across all 400 configs would be the first usable certificate. If the best functional stays above $100\times$, the measurement variant is confirmed blocked and Lipschitz-style composition should be abandoned for distributional arguments.

## 9. Key References

- **[Foundational]** P. Drineas, R. Kannan, M. W. Mahoney. *Fast Monte Carlo Algorithms for Matrices I: Approximating Matrix Multiplication.* SIAM Journal on Computing, 2006.
- **[Foundational]** N. Ailon, B. Chazelle. *The Fast Johnson–Lindenstrauss Transform and Approximate Nearest Neighbors.* SIAM Journal on Computing, 2009.
- **[Foundational]** K. Clarkson, D. Woodruff. *Low Rank Approximation and Regression in Input Sparsity Time.* STOC, 2013.
- **[Survey]** P.-G. Martinsson, J. A. Tropp. *Randomized Numerical Linear Algebra: Foundations and Algorithms.* Acta Numerica, 2020. — arXiv:2002.01387
- **[Survey]** D. Woodruff. *Sketching as a Tool for Numerical Linear Algebra.* Foundations and Trends in Theoretical Computer Science, 2014.
- **[SOTA]** T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[SOTA]** E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, S. Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** J. Chee, Y. Cai, V. Kuleshov, C. De Sa. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees.* NeurIPS, 2023. — arXiv:2307.13304
- **[Method]** D. Blalock, J. Guttag. *Multiplying Matrices Without Multiplying.* ICML, 2021. — arXiv:2106.10860
- **[Method]** A. Mishra et al. *Accelerating Sparse Deep Neural Networks.* Technical report, NVIDIA, 2021. — arXiv:2104.08378
- **[Method]** T. Dao, B. Chen, N. Sohoni, A. Desai, M. Poli, J. Grogan, A. Liu, A. Rao, A. Rudra, C. Ré. *Monarch: Expressive Structured Matrices for Efficient and Accurate Training.* ICML, 2022. — arXiv:2204.00595
- **[Theory]** M. Fazlyab, A. Robey, H. Hassani, M. Morari, G. Pappas. *Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks.* NeurIPS, 2019. — arXiv:1906.04893
- **[Theory]** A. Virmaux, K. Scaman. *Lipschitz Regularity of Deep Neural Networks: Analysis and Efficient Estimation.* NeurIPS, 2018.
- **[Context]** P. Micikevicius et al. *FP8 Formats for Deep Learning.* Technical report, 2022. — arXiv:2209.05433

## 10. Worked Example

One attention head. Head dimension $K = 128$, sequence length $n = 2048$. Query $q \in \mathbb{R}^{128}$ and keys $k_j$ all unit-norm. We approximate $qK^\top$ by sampling $c$ of the $128$ inner-dimension coordinates.

True top-two scores: $q\cdot k_{(1)} = 0.300$, $q\cdot k_{(2)} = 0.280$. Gap $g = 0.020$.

Sampling estimator error per entry has standard deviation about $\|q\|\|k_j\|/\sqrt{c} = 1/\sqrt{c}$.

| $c$ | per-entry RMS error | error / gap |
|---|---|---|
| 16 | 0.250 | 12.5× |
| 32 | 0.177 | 8.8× |
| 64 | 0.125 | 6.3× |
| 128 (= exact cost) | 0.088 | 4.4× |

To keep the argmax stable across $n=2048$ candidates you need the error well under $g$ — roughly $1/\sqrt{c} \le 0.02/3$, i.e. $c \ge 22{,}500$, which is $176\times$ the inner dimension itself. Sampling is strictly worse than exact multiplication here, by two orders of magnitude.

Meanwhile the Drineas bound is *satisfied and tight* the whole time: at $c=64$, $\mathbb{E}\|qK^\top - \widehat{qK^\top}\|_F \le \|q\|_F\|K\|_F/\sqrt{c} = 1 \cdot 45.3/8 = 5.66$, and the realized error matches. The theory is correct. It is simply bounding the wrong thing: $5.66$ spread over 2048 entries is a per-entry perturbation 6× larger than the decision margin.

That is the obstruction in one table. Frobenius-norm guarantees are tight and useless; quantization works in practice precisely because it is *deterministic and per-entry bounded* ($\pm\Delta/2$ per weight, $\Delta \approx 0.005$ at INT8 on unit-norm rows) rather than sampled — and no one has yet turned that per-entry control into a statement about $\delta$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*