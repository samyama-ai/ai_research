---
id: 31-distributed-training/low-precision-allreduce-convergence
title: "Low-Precision Collective Reductions and Convergence"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Low-Precision Collective Reductions and Convergence

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/low-precision-allreduce-convergence` · **Status:** empirically-open

## 1. Problem Statement

Data-parallel training averages per-worker gradients with an all-reduce. The arithmetic inside that collective — the format of the wire payload and the format of the accumulator — is a free design choice. Practice has converged on a folk rule: send in BF16 or FP8, but accumulate in FP32. Nobody has shown the rule is necessary at scale.

The question: **for a fixed token budget, how far can the reduction format be lowered before the final loss degrades by more than run-to-run seed noise, and how does that threshold scale with world size $N$?**

Three variants, different difficulty:

- **Measurement.** Define a scalar that predicts loss damage from a reduction's numerics. Candidates (relative error of the averaged gradient, injected gradient-noise variance, accumulated bias norm) are not known to be interchangeable, and none has been shown to be the right summary statistic.
- **Method.** Build a collective that hits a target end-to-end quality at lower cost than BF16-payload/FP32-accumulate. Error feedback, stochastic rounding, and block-wise scaling are the tools; the open part is whether any of them beats the baseline *end to end* once the extra kernel launches and buffers are counted.
- **Theory.** Give a convergence rate for non-convex SGD/Adam under a reduction operator whose error is *multiplicative in the gradient's condition number for summation* and grows with $N$. Existing compression theory assumes bounded relative or bounded absolute error; floating-point reduction error is neither.

Solved means: a curve of maximum safe $(\text{payload bits}, \text{accumulator bits})$ as a function of $N$ and model scale, validated by a run that goes to a real token budget.

## 2. Formal Setting

$N$ workers, parameters $\theta \in \mathbb{R}^d$, per-worker minibatch gradient $g_i \in \mathbb{R}^d$. Exact target: $\bar g = \frac{1}{N}\sum_{i=1}^N g_i$. The collective returns $\hat g = \mathcal{R}(g_1,\dots,g_N)$.

**Floating-point model.** A format with $p$ mantissa bits has unit roundoff $u = 2^{-(p+1)}$. Measured, not assumed: BF16 $u = 2^{-9} \approx 1.95\times10^{-3}$; FP16 $u = 2^{-11} \approx 4.9\times 10^{-4}$; FP8 E4M3 $u = 2^{-4} = 6.25\times10^{-2}$; FP32 $u = 2^{-24} \approx 6.0\times10^{-8}$.

**Reduction tree.** Ring all-reduce performs $N-1$ sequential additions per element in reduce-scatter, then $N-1$ broadcasts; a tree performs $\lceil \log_2 N\rceil$. Let $m$ = number of dependent roundings on the critical path, measured by reading the NCCL algorithm actually selected (`NCCL_ALGO`), not by assuming ring.

**Error.** Per coordinate $j$, with round-to-nearest,
$$\left|\hat g_j - \bar g_j\right| \le \gamma_m \cdot \tfrac{1}{N}\sum_i |g_{ij}|, \qquad \gamma_m = \frac{mu}{1-mu}.$$
With stochastic rounding the bound is probabilistic and $O(\sqrt{m}\,u)$ (Higham & Mary 2019; Connolly, Higham & Mary 2021). Define the **summation condition number**
$$\kappa_j = \frac{\sum_i |g_{ij}|}{\left|\sum_i g_{ij}\right|},$$
measured directly by running one FP64 reference reduction on a captured gradient snapshot. The observable relative error is $\varepsilon_j \le \gamma_m \kappa_j$.

**Convergence surrogate.** Model $\hat g = \bar g + b + \xi$ with bias $b = \mathbb{E}[\hat g - \bar g]$ and zero-mean $\xi$, $\Sigma = \mathrm{Cov}(\xi)$. Both are measured by repeating the same reduction on the same inputs across dithers/orderings and comparing to FP64. The decision quantity is not $\|b\|$ but the loss gap $\Delta L = L_T^{\text{low}} - L_T^{\text{fp32}}$ against seed variance $\sigma_{\text{seed}}$, with $T$ = full token budget.

**Assumptions, and which are violated.**
- *Gradients are zero-mean across workers.* Approximately true, and it is the problem: it makes $\kappa \gg 1$.
- *Rounding errors are independent across ranks.* Violated — round-to-nearest is deterministic and correlated with input sign, which is why $b \neq 0$.
- *Bounded relative compression error* (the standard $\|\mathcal{C}(x)-x\| \le (1-\delta)\|x\|$ assumption of QSGD/error-feedback theory). Violated: floating-point reduction error scales with $\sum_i |g_{ij}|$, not with $\|\bar g\|$.
- *Determinism.* Violated: NCCL chunking and algorithm selection vary with buffer size and topology, so $\hat g$ is not reproducible across cluster shapes.

## 3. State of the Art

**Established.**
- Mixed-precision training with FP32 master weights and FP32 loss-scaled accumulation matches FP32 quality (Micikevicius et al., ICLR 2018). BF16 with FP32 accumulate likewise (Kalamkar et al., 2019).
- Error feedback restores convergence for biased compressors including sign-based ones (Karimireddy et al., ICML 2019); Stich et al. (NeurIPS 2018) give the memory-based analysis. These cover *compression* operators, not floating-point reduction error.
- Gradient compression frequently fails to pay off end to end on modern interconnects once encode/decode cost is counted (Agarwal et al., MLSys 2022). This is a reproduced negative result and the main reason the field defaults to plain BF16 collectives.

**Claimed but unablated.**
- FP8-LM (Peng et al., 2023) reports FP8 gradients and FP8 all-reduce with automatic pre/post scaling, 39% memory reduction and up to 75% speedup vs BF16 for GPT-175B. The loss-parity evidence is at 7B/13B; the 175B number is a systems throughput benchmark, not a converged quality comparison.
- DeepSeek-V3 (2024) trains at FP8 over 14.8T tokens but explicitly *keeps the MoE combine step in BF16* and promotes tensor-core accumulation to FP32 every 128 elements. This is the strongest existing evidence that naive low-precision reduction is unsafe — but it is a design decision reported without the ablation that would show what breaks.
- SwitchML (Sapio et al., NSDI 2021) does in-network aggregation in 32-bit integers with per-block scaling; quality parity is reported on ResNet/BERT-scale workloads only.
- THC (Li et al., NSDI 2024) makes quantized values directly aggregatable, avoiding decompress-aggregate-recompress; benchmarked at cluster scale, not at frontier LLM scale.

**Theory SOTA** is QSDP (Markov et al., ICML 2023): convergence guarantees for quantized fully-sharded training. It bounds a quantizer, not a reduction tree, and has no $N$-dependence of the kind $\gamma_m\kappa$ implies.

## 4. What Is Known

- Worst-case summation error grows as $mu$; stochastic rounding gives $O(\sqrt{m}u)$ with high probability (Higham & Mary, *SIAM J. Sci. Comput.* 2019).
- 16-bit fixed point with stochastic rounding trains MNIST/CIFAR CNNs to FP32 parity; round-to-nearest at the same width does not (Gupta et al., ICML 2015). Scale: ~1M-parameter models.
- FP8 with chunk-based accumulation in FP16 trains ResNet-50/AlexNet to baseline accuracy (Wang et al., NeurIPS 2018); naive FP8 accumulation does not. Scale: ImageNet.
- DeepSeek-V3 measures ~2% relative error from H800 FP8 tensor-core accumulation on a 4096-length inner product before FP32 promotion. Scale: production, 671B parameters.
- 1-bit SGD with error feedback trains production speech DNNs with no accuracy loss (Seide et al., INTERSPEECH 2014). Scale: ~50M parameters.
- Precision affects effective parameter count in a fittable way (Kumar et al., "Scaling Laws for Precision", 2024) — for weights and activations. No analogous law exists for the reduction.

## 5. What Is Not Known

- **Empirically open (the core gap).** Whether a pure-BF16 reduction (BF16 accumulator, not just BF16 payload) degrades final loss at $N \ge 1024$ and $\ge 10^{12}$ tokens. Runnable today; nobody publishes the control arm because the FP32-accumulate baseline is cheap enough that the ablation has no product value.
- **Empirically open.** Whether stochastic rounding inside NCCL's reduce kernel removes the safety margin FP32 accumulation buys. No production collective implements it.
- **Theoretically open.** A non-convex convergence rate under error of the form $\gamma_m \kappa$, where $\kappa$ itself depends on the gradient-noise-to-signal ratio and therefore on training phase. Existing bounds do not apply.
- **Methodologically blocked.** Which scalar predicts $\Delta L$. Cosine similarity to the FP64 gradient, $\|b\|/\|\bar g\|$, and injected variance are all reported in the literature; none has been shown to be monotone in end-of-run loss.

## 6. Why It Is Hard

The specific obstruction is **condition-number blowup under cancellation, compounded by confounded measurement**.

Averaged gradients across workers are close to zero-mean per coordinate, so $\kappa_j = \sum_i|g_{ij}| / |\sum_i g_{ij}|$ grows roughly as $\sqrt{N}$ times the per-rank noise-to-signal ratio. The quantity being computed is exactly the small residual left after near-total cancellation — the classic worst case for floating-point summation. Bounds that look harmless in isolation ($mu \approx 0.002$) become vacuous once multiplied by $\kappa \sim 30$–$300$.

The second obstruction: the error is *not* obviously harmful. SGD already injects gradient noise, and a zero-mean reduction error may be absorbed. So the sign of the effect cannot be read off the numerics — it has to be measured at the end of a full-budget run. That makes the deciding experiment cost a frontier pretraining run, and every cheap proxy (short runs, small $N$) has the wrong $\kappa$ and the wrong signal-to-noise, so it does not transfer.

## 7. Current Research (as of 2026)

- **FP8 and below in production collectives.** NVIDIA (Transformer Engine), DeepSeek, and Microsoft (FP8-LM lineage) all ship partial-FP8 pipelines that deliberately exempt the reduction. *(frontier — verify)* whether any has published an ablation isolating the collective.
- **In-network aggregation.** Follow-ons to SwitchML/ATP/THC push aggregation into switches and NICs, where the accumulator width is a hardware constant, making this problem load-bearing rather than academic.
- **Stochastic rounding in hardware.** Available on Graphcore IPUs and in some AMD/NVIDIA instruction paths; not yet exposed inside vendor collective libraries. *(frontier — verify)*
- **Precision scaling laws.** Extending Kumar et al. from weights/activations to communication is an obvious next step and, to our knowledge, unpublished.

## 8. Concrete Next Experiment

**Scale.** 1.4B-parameter decoder, 100B tokens (Chinchilla-ish), $N = 512$ data-parallel ranks, ring all-reduce, 3 seeds per arm. ~2–3k GPU-days total for all arms — expensive but not frontier-expensive.

**Arms.**
- **Control:** BF16 payload, FP32 accumulator (current default).
- **A:** BF16 payload, BF16 accumulator, round-to-nearest.
- **B:** BF16 payload, BF16 accumulator, stochastic rounding.
- **C:** FP8 E4M3 payload with per-128-element block scaling, FP32 accumulator.

Instrument every arm: at 20 checkpoints, replay the captured per-rank gradients through an FP64 reference reduction and log $\kappa$, $\|b\|/\|\bar g\|$, and cosine similarity.

**The deciding number.** $\Delta L = L_{100\text{B}}^{\text{arm}} - L_{100\text{B}}^{\text{control}}$ in nats, against $\sigma_{\text{seed}}$ (typically ~0.003 nats at this scale). **Decision predicate: arm A fails if $\Delta L > 3\sigma_{\text{seed}}$.** If A fails and B passes, stochastic rounding is the fix and the result is a vendor feature request. If A passes, the FP32 accumulator is dead weight at $N=512$ and the question moves to $N=4096$.

Secondary output: the regression of $\Delta L$ on $\langle \kappa \rangle$ across arms — the first evidence for or against $\kappa$ as the predictive scalar, which is what unblocks Section 5's methodological gap.

## 9. Key References

- **[Foundational]** N. J. Higham, T. Mary. *A New Approach to Probabilistic Rounding Error Analysis.* SIAM J. Sci. Comput., 2019.
- **[Foundational]** M. Connolly, N. J. Higham, T. Mary. *Stochastic Rounding and Its Probabilistic Backward Error Analysis.* SIAM J. Sci. Comput., 2021.
- **[Foundational]** S. Gupta, A. Agrawal, K. Gopalakrishnan, P. Narayanan. *Deep Learning with Limited Numerical Precision.* ICML, 2015. — arXiv:1502.02551
- **[Foundational]** F. Seide, H. Fu, J. Droppo, G. Li, D. Yu. *1-Bit Stochastic Gradient Descent and Its Application to Data-Parallel Distributed Training of Speech DNNs.* INTERSPEECH, 2014.
- **[Foundational]** P. Micikevicius et al. *Mixed Precision Training.* ICLR, 2018. — arXiv:1710.03740
- **[Theory]** D. Alistarh, D. Grubic, J. Li, R. Tomioka, M. Vojnovic. *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding.* NeurIPS, 2017. — arXiv:1610.02132
- **[Theory]** S. P. Karimireddy, Q. Rebjock, S. Stich, M. Jaggi. *Error Feedback Fixes SignSGD and other Gradient Compression Schemes.* ICML, 2019. — arXiv:1901.09847
- **[Theory]** I. Markov, A. Vladu, Q. Guo, D. Alistarh. *Quantized Distributed Training of Large Models with Convergence Guarantees.* ICML, 2023. — arXiv:2302.02390
- **[SOTA]** N. Wang, J. Choi, D. Brand, C.-Y. Chen, K. Gopalakrishnan. *Training Deep Neural Networks with 8-bit Floating Point Numbers.* NeurIPS, 2018.
- **[SOTA]** H. Peng et al. *FP8-LM: Training FP8 Large Language Models.* Microsoft Research, 2023. — arXiv:2310.18313
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[SOTA]** A. Sapio et al. *Scaling Distributed Machine Learning with In-Network Aggregation.* NSDI, 2021.
- **[SOTA]** M. Li et al. *THC: Accelerating Distributed Deep Learning Using Tensor Homomorphic Compression.* NSDI, 2024.
- **[Negative result]** S. Agarwal, H. Wang, S. Venkataraman, D. Papailiopoulos. *On the Utility of Gradient Compression in Distributed Training Systems.* MLSys, 2022.
- **[Survey]** Z. Tang, S. Shi, W. Wang, B. Li, X. Chu. *Communication-Efficient Distributed Deep Learning: A Comprehensive Survey.* 2020. — arXiv:2003.06307
- **[Related]** T. Kumar et al. *Scaling Laws for Precision.* 2024. — arXiv:2411.04330

## 10. Worked Example

One coordinate $j$ of a 1.4B-parameter model, $N = 1024$ ranks, ring all-reduce ($m = N-1 = 1023$ dependent additions per element in reduce-scatter).

Suppose the per-rank gradient at that coordinate has mean $\mu = 10^{-4}$ and per-rank standard deviation $\sigma = 10^{-3}$ (noise-to-signal $r = 10$; typical mid-training for a well-tuned batch size). Then

$$\sum_i |g_{ij}| \approx N \cdot \sigma\sqrt{2/\pi} \approx 1024 \times 8.0\times10^{-4} = 0.82, \qquad \left|\sum_i g_{ij}\right| \approx N\mu = 0.102,$$

so $\kappa_j \approx 8.0$.

**BF16 accumulator, round-to-nearest.** $u = 1.95\times10^{-3}$, $\gamma_m = mu/(1-mu)$ — but $mu = 1023 \times 1.95\times10^{-3} = 2.0 > 1$. **The deterministic bound is vacuous.** Nothing can be concluded; the sequential ring has more rounding steps than BF16 has precision to absorb.

**BF16 accumulator, stochastic rounding.** The probabilistic bound gives $\approx \sqrt{m}\,u\,\kappa = 32 \times 1.95\times10^{-3} \times 8.0 \approx 0.50$ — a **50% relative error on the averaged gradient**, unbiased.

**FP32 accumulator.** $\sqrt{m}\,u\,\kappa = 32 \times 6.0\times10^{-8}\times 8.0 = 1.5\times10^{-5}$. Negligible.

**Where the obstruction becomes visible.** The BF16-accumulate error is 50% of the gradient magnitude — an order of magnitude larger than anything the compression literature tolerates. Yet with stochastic rounding it is *zero-mean*, and the gradient already carries per-rank noise $\sigma/\mu = 10$ that the average reduces to $\sigma/(\mu\sqrt N) \approx 0.31$. So the injected error (0.50) and the intrinsic minibatch noise (0.31) are the *same order*. Doubling the gradient-noise scale is equivalent, to first order, to halving the batch size — which for a fixed token budget is a real but modest loss penalty, not a divergence.

That is the whole difficulty in one line: **the numerics say the error is enormous, the optimization theory says it may be free, and the two arguments have never been adjudicated by a run at real scale.** The Section 8 experiment is designed to force that adjudication with one number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*