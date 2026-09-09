---
id: 32-hardware-and-kernels/low-precision-accumulation-error-bounds
title: "Low-Precision Accumulation Error Bounds for Long Reductions"
topic: 32-hardware-and-kernels
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Low-Precision Accumulation Error Bounds for Long Reductions

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/low-precision-accumulation-error-bounds` · **Status:** partially-solved

## 1. Problem Statement

A modern LLM kernel reduces over thousands of terms: a GEMM inner dimension $K \in [2^{10}, 2^{14}]$, an attention softmax denominator over the KV length, an all-reduce over $10^9$ parameters, an optimizer moment over a token batch. The multiplies happen in FP8/FP6/FP4; the accumulator is FP16, BF16, FP22-ish tensor-core internal formats, or FP32. The question is what error the reduction actually incurs, and at what $K$ the accumulator precision — not the input precision — becomes the binding constraint.

Three variants, with different difficulty:

- **Theory.** Give a bound on $|\hat s_K - s_K|$ that is (i) valid for the *actual* arithmetic of a tensor core — block FMA, non-round-to-nearest, flushed subnormals — and (ii) non-vacuous at the $K$ and unit roundoff $u$ used in production. Classical bounds go vacuous at $Ku \approx 1$: $K \approx 2048$ for FP16, $K \approx 256$ for BF16.
- **Method.** Choose the cheapest accumulation scheme (chunk size, promotion frequency, split/compensated variant, stochastic rounding) that keeps end-to-end training loss indistinguishable from an FP32-accumulate control.
- **Measurement.** Define the error functional that predicts downstream damage. Per-GEMM relative error does not; the same $10^{-3}$ error is harmless in an MLP and destabilizing in an attention logit that feeds $\exp$.

Solved means: a bound, tight to within a small constant against measurement, that tells a kernel author the minimum accumulator width for a given $(K, u, \text{data distribution})$ and a target end-to-end degradation.

## 2. Formal Setting

Inputs $x_1,\dots,x_K \in \mathbb{R}$, exact sum $s_K = \sum_i x_i$. A precision has unit roundoff $u$ (half the gap at 1): FP32 $u = 2^{-24}$, TF32 $2^{-11}$, FP16 $2^{-11}$, BF16 $2^{-8}$, FP8-E4M3 $2^{-4}$.

**Standard model.** $fl(a \circ b) = (a \circ b)(1+\delta)$, $|\delta| \le u$. Recursive summation then satisfies (Higham 1993, 2002)

$$|\hat s_K - s_K| \le \gamma_{K-1} \sum_{i=1}^K |x_i|, \qquad \gamma_n = \frac{nu}{1-nu}.$$

Pairwise (tree) summation replaces $\gamma_{K-1}$ with $\gamma_{\lceil \log_2 K\rceil}$. Blocked accumulation with chunk length $b$ and a wide accumulator of roundoff $u_{\text{hi}}$ gives $\gamma_{b-1}(u) + \gamma_{K/b}(u_{\text{hi}})$ to first order.

**Probabilistic model.** Treat $\delta_i$ as mean-zero independent random variables. Higham & Mary (2019) and Connolly, Higham & Mary (2021) replace $\gamma_n$ with $\tilde\gamma_n(\lambda) = \exp(\lambda\sqrt{n}u + O(u^2)) - 1$, holding with probability $\ge 1 - 2\exp(-\lambda^2/2)$ — the $n \to \sqrt{n}$ rule. Under **stochastic rounding** the $\delta_i$ are mean-independent by construction, so the bound is unconditional, and the error is *unbiased*: $\mathbb{E}[\hat s_K] = s_K$.

**Measured quantities.**
- Forward relative error $E_K = |\hat s_K - s_K| / |s_K|$, with $s_K$ computed in FP64 or exactly (Ogita–Rump–Oishi / integer superaccumulator). FP64 reference is only valid when $K u_{64} \kappa \ll E_K$.
- Condition number $\kappa = \sum_i |x_i| / |\sum_i x_i|$. This is the quantity that actually varies across kernels: $\kappa = 1$ for the softmax denominator (all terms positive), $\kappa \gg 1$ for logits and gradient reductions.
- Effective mantissa bits of the accumulator, $b_{\text{eff}} = -\log_2 \hat u$, estimated by fitting $E_K$ against $K$ on synthetic data with known $\kappa$ — the only way to probe undocumented tensor-core internals.
- End-to-end target: $\Delta\mathcal{L}$, validation loss gap against an FP32-accumulate control at matched tokens and seed.

**Assumptions known to be violated.**
1. *Round-to-nearest.* Fasi, Higham, Mikaitis & Pranesh (2021) measured NVIDIA V100/T4 tensor cores: the five-term accumulation rounds **toward zero**, not to nearest, and products are held exactly before summation. The standard model's symmetric $|\delta|\le u$ is wrong; RTZ error is one-sided and accumulates linearly with no cancellation.
2. *Independence of rounding errors.* Fails exactly where it matters — softmax denominators (all $x_i > 0$, all RTZ errors same sign), and repeated/quantized values that produce identical $\delta_i$.
3. *No subnormals / no flush.* FP8 dynamic range is ~$2^{-9}$ to $448$ in E4M3; per-tensor scaling puts real gradient tails into flush-to-zero.
4. *Data independence.* Bounds scale with $\sum|x_i|$ and are worst-case in sign pattern; real activations post-LayerNorm are near-Gaussian, making the worst case unreachable but the *tail over $10^{12}$ reductions per training run* reachable.

## 3. State of the Art

**Theory SOTA — established.**
- Deterministic $\gamma_{K-1}$ (recursive) and $\gamma_{\log K}$ (pairwise) bounds: Higham (1993); *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Ch. 4. Tight in the worst case.
- $\sqrt{K}u$ probabilistic bounds with explicit failure probability: Higham & Mary, SIAM J. Sci. Comput. 2019; sharpened for random data, 2020.
- Stochastic rounding gives unconditional $\sqrt{K}u$ and unbiasedness: Connolly, Higham & Mary, SIMAX 2021.
- Mixed-precision **block FMA** model covering tensor cores (exact products, wide internal accumulator, block size $b$): Blanchard, Higham, Lopez, Mary & Pranesh, SIAM J. Sci. Comput. 2020. This is the closest thing to a hardware-faithful bound.
- FABsum (fast-and-accurate blocked summation): Blanchard, Higham & Mary, SIAM J. Sci. Comput. 2020 — error $\gamma_b + \gamma_{K/b}^{\text{hi}}$ at near-recursive cost.
- Compensated summation: Kahan (CACM 1965), Neumaier (1974), Ogita–Rump–Oishi (SISC 2005) achieve $u + O(\kappa u^2)$ — effectively doubled precision.

**Systems SOTA — established.**
- FP16 multiply / FP32 accumulate as the default safe point: Micikevicius et al., *Mixed Precision Training*, ICLR 2018.
- Chunk-based accumulation as the enabling trick for 8-bit training: Wang et al., NeurIPS 2018 — FP8 multiply into FP16 accumulator with chunk size 64 and stochastic rounding, matching FP32 on ResNet-50/ImageNet.
- DeepSeek-V3 (2024): on Hopper, FP8 tensor-core accumulation retains roughly 14 mantissa bits; they promote partial sums to FP32 CUDA cores every $N_C = 128$ elements of $K$.

**Claimed but unablated.**
- That $\sqrt{K}u$ is the operative rate *for LLM data*. It is proved under independence, and reported to hold on synthetic and small dense-linear-algebra workloads. There is no published fit of $E_K$ vs $K$ across LLM GEMMs at $K = 2^{14}$ separating the $K$, $\sqrt{K}$ and $\log K$ regimes.
- That $N_C = 128$ is near-optimal rather than merely sufficient. It appears as an engineering choice, not a swept ablation with loss curves at each chunk size.
- Golden et al. (*Is Flash Attention Stable?*, 2024, arXiv:2405.02803) report Flash Attention shows an order of magnitude more numeric deviation than baseline attention in BF16 — a benchmark number on one model family, with the link to training instability argued but not causally demonstrated.

## 4. What Is Known

- **Vacuity thresholds are exact and small.** $\gamma_{K-1}$ exceeds 1 at $K = 1 + 1/u$: $K = 2049$ (FP16/TF32), $K = 257$ (BF16), $K = 17$ (FP8-E4M3). Production GEMMs run $K = 4096$–$16384$.
- **The $\sqrt{K}$ rate is real for random data.** Higham & Mary (2019) report measured errors tracking $\sqrt{K}u$ over $K$ up to $10^6$ in FP32/FP64 summation of random vectors, with the deterministic bound overshooting by $10^2$–$10^3\times$.
- **Tensor cores are not IEEE.** Fasi et al. (PeerJ CS 2021) established on V100/T4: products exact, accumulation order fixed, rounding toward zero on the 5-term sum, and monotonicity preserved. This makes RTZ drift, not RN cancellation, the dominant mechanism for one-signed reductions.
- **Chunking works at 8 bits.** Wang et al. (2018): FP8 with FP16 chunked accumulation matched FP32 baselines on ResNet-18/50, AlexNet at ImageNet scale; naive FP16 accumulation without chunking lost several points of top-1.
- **The concrete FP8 number.** DeepSeek-V3's report: a $K = 4096$ FP8 GEMM accumulated entirely on tensor cores reaches a maximum relative error near $2\%$; FP32 promotion every 128 elements removes it, at a reported cost well under 10% of GEMM throughput on Hopper.
- **Bit-reproducibility is solvable at a price.** Demmel & Nguyen (IEEE Trans. Computers 2015) and ReproBLAS give order-independent reproducible summation for roughly $1.2$–$2\times$ the cost of a plain reduction.

## 5. What Is Not Known

- **Theoretically open.** A non-vacuous, hardware-faithful bound for *round-toward-zero block accumulation with correlated one-signed data* — the softmax denominator case. The probabilistic $\sqrt{K}u$ machinery assumes mean-zero errors; RTZ violates this by construction and the correct rate there is plausibly $\Theta(Ku)$, but no matching lower bound for the tensor-core block-FMA model has been published.
- **Theoretically open.** Whether stochastic rounding's unbiasedness survives composition through a nonlinearity: $\mathbb{E}[\hat s] = s$ does not give $\mathbb{E}[f(\hat s)] = f(s)$, and no bound exists on the induced bias in $\exp$/softmax over a full forward pass.
- **Empirically open.** The chunk-size sweep. Run the same LLM pretrain at $N_C \in \{32, 64, 128, 256, 512, \infty\}$ and read off where $\Delta\mathcal{L}$ departs from zero. Runnable today on 8–64 GPUs at 1B scale; nobody has published it.
- **Methodologically blocked.** The error functional that predicts $\Delta\mathcal{L}$. Per-GEMM relative error, max absolute deviation and Wasserstein distance between activation histograms are all in use; none has been shown to rank-order interventions the same way end-to-end loss does.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth at the hardware level combined with a confounded end-to-end signal**.

- The accumulator internals are undocumented. $b_{\text{eff}}$ must be inferred from black-box probing (the Fasi et al. method), which is per-architecture, breaks silently on each new generation, and cannot be run for arithmetic the vendor exposes only through a fused kernel.
- The end-to-end number is confounded. Changing accumulation changes the loss curve, but so does the reduction order, which changes with tile shape, which changes with the tuner's kernel choice, which changes when you change accumulation. An A/B on chunk size is not an A/B on numerics unless the schedule is pinned — and pinning it costs the performance the low precision was for.
- Worst-case theory and average-case measurement disagree by $10^2$–$10^3\times$ (Section 4), so neither alone sizes the accumulator. The binding event is a tail: one destabilizing reduction among $\sim 10^{12}$ per run. Bounds with a $2e^{-\lambda^2/2}$ failure probability need $\lambda \approx 7.4$ to survive a union bound at $10^{12}$ trials, which returns roughly $7.4\sqrt{K}u$ — within an order of magnitude of the deterministic bound it was meant to replace.

## 7. Current Research (as of 2026)

- **Manchester numerical linear algebra group** (Higham†/Mary/Mikaitis lineage, now distributed across Manchester, Sorbonne and Leeds): probabilistic and stochastic-rounding error analysis, hardware probing of reduced-precision units. The most reliable source of theorems here.
- **Vendor-side**: NVIDIA's FP8/FP4 format work (Micikevicius et al., arXiv:2209.05433) and the Blackwell-generation microscaling formats (MX-FP8/FP6/FP4, OCP standard) push block-scaled arithmetic where the *scale block* size and the *accumulation chunk* size interact — largely unanalysed jointly. *(frontier — verify)*
- **Emulation via integer arithmetic**: Ozaki-scheme splitting (Ozaki et al., Numerical Algorithms 2012) revived by Ootomo & Yokota (IJHPCA 2022) to recover FP32-accurate GEMM from tensor cores, and more recently to emulate FP64 on INT8 units. Gives error-free transformations with quantifiable bounds — the strongest current bridge between theory and kernels.
- **Training-stability forensics**: Meta's Flash Attention numeric-deviation work (Golden et al. 2024) and follow-ons trying to connect per-kernel deviation to loss spikes. Causal link still open. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** At what FP32-promotion interval $N_C$ does FP8 accumulation start to cost loss, and does the $\sqrt{K}u$ or the $Ku$ rate predict it?

**Scale.** A 1.3B-parameter decoder, $d_{\text{model}} = 2048$, $K = 2048$–$8192$ per GEMM, 100B tokens, 64×H100 — about 3 GPU-days per arm.

**Arms.** $N_C \in \{32, 64, 128, 512, \infty\}$ (where $\infty$ = pure tensor-core FP8 accumulation), 2 seeds each.

**Control arm.** Identical run with FP8 multiplies and full FP32 accumulation, *with the tile schedule pinned identically across all arms* (fixed tile shape, fixed split-K, autotuner disabled). This is the step that de-confounds numerics from reduction order — without it the experiment measures the tuner.

**Instrumentation.** On a held-out set of 200 GEMM invocations per arm, record $E_K$ against an exact integer-superaccumulator reference, and $\kappa$ per reduction. Fit $\log E_K = \alpha \log K + c$.

**The deciding number.** $\Delta\mathcal{L}$, validation loss gap to the control at 100B tokens, with the decision threshold at $0.01$ nats (roughly $2\times$ the seed-to-seed spread at this scale). Report the largest $N_C$ with $\Delta\mathcal{L} < 0.01$. The secondary number is the fitted exponent $\alpha$: $\alpha \approx 0.5$ confirms the probabilistic regime, $\alpha \approx 1.0$ says RTZ drift dominates and the probabilistic bounds do not apply to this hardware.

## 9. Key References

- **[Foundational]** Nicholas J. Higham. *The Accuracy of Floating Point Summation.* SIAM J. Sci. Comput. 14(4), 1993.
- **[Foundational]** Nicholas J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002. (Ch. 4, summation.)
- **[Foundational]** William Kahan. *Further Remarks on Reducing Truncation Errors.* Communications of the ACM 8(1), 1965.
- **[Foundational]** Takeshi Ogita, Siegfried M. Rump, Shin'ichi Oishi. *Accurate Sum and Dot Product.* SIAM J. Sci. Comput. 26(6), 2005.
- **[SOTA — theory]** Nicholas J. Higham, Théo Mary. *A New Approach to Probabilistic Rounding Error Analysis.* SIAM J. Sci. Comput. 41(5), 2019.
- **[SOTA — theory]** Michael P. Connolly, Nicholas J. Higham, Théo Mary. *Stochastic Rounding and Its Probabilistic Backward Error Analysis.* SIAM J. Matrix Anal. Appl. 42(3), 2021.
- **[SOTA — theory]** Pierre Blanchard, Nicholas J. Higham, Théo Mary. *A Class of Fast and Accurate Summation Algorithms.* SIAM J. Sci. Comput. 42(3), 2020.
- **[SOTA — hardware model]** Pierre Blanchard, Nicholas J. Higham, Florent Lopez, Théo Mary, Srikara Pranesh. *Mixed Precision Block Fused Multiply-Add: Error Analysis and Application to GPU Tensor Cores.* SIAM J. Sci. Comput. 42(3), 2020.
- **[SOTA — measurement]** Massimiliano Fasi, Nicholas J. Higham, Mantas Mikaitis, Srikara Pranesh. *Numerical Behavior of NVIDIA Tensor Cores.* PeerJ Computer Science 7:e330, 2021.
- **[SOTA — systems]** Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, Kailash Gopalakrishnan. *Training Deep Neural Networks with 8-bit Floating Point Numbers.* NeurIPS 2018.
- **[SOTA — systems]** Paulius Micikevicius et al. *Mixed Precision Training.* ICLR 2018 — arXiv:1710.03740.
- **[SOTA — systems]** Paulius Micikevicius et al. *FP8 Formats for Deep Learning.* 2022 — arXiv:2209.05433.
- **[SOTA — systems]** Hiroyuki Ootomo, Rio Yokota. *Recovering Single Precision Accuracy from Tensor Cores While Surpassing the FP32 Theoretical Peak Performance.* Int. J. High Performance Computing Applications 36(4), 2022.
- **[Related]** James Demmel, Hong Diep Nguyen. *Parallel Reproducible Summation.* IEEE Transactions on Computers 64(7), 2015.
- **[Related]** Alicia Golden et al. *Is Flash Attention Stable?* 2024 — arXiv:2405.02803.
- **[Survey]** Ahmad Abdelfattah et al. *A Survey of Numerical Linear Algebra Methods Utilizing Mixed-Precision Arithmetic.* Int. J. High Performance Computing Applications 35(4), 2021.

## 10. Worked Example

**Setting.** One attention softmax denominator, $Z = \sum_{i=1}^{K} e_i$, $K = 4096$, $e_i > 0$. All terms positive, so $\kappa = 1$ — the *best possible* conditioning. Accumulate in BF16, $u = 2^{-8} = 3.91\times10^{-3}$.

**The three bounds.**

| Scheme | Bound | Value at $K=4096$ |
|---|---|---|
| Recursive, deterministic $\gamma_{K-1}$ | $(K{-}1)u/(1-(K{-}1)u)$ | $(K{-}1)u = 16.0 \Rightarrow$ **vacuous** |
| Recursive, probabilistic $\lambda\sqrt{K}u$, $\lambda=2$ | $2\cdot 64 \cdot u$ | $0.50$ — still useless |
| Pairwise, deterministic $\gamma_{\log_2 K}$ | $12u$ | $4.7\times10^{-2}$ |
| Chunked, $b=128$ into FP32 | $\gamma_{127}(u) + \gamma_{32}(2^{-24})$ | $0.50 + 1.9\times10^{-6}$ — the *chunk*, not the tree, dominates |

The first row is the standard textbook bound and it says nothing: a 16.0 relative-error bound on a quantity that is provably positive. The second row, the modern probabilistic bound, is 30× better and still says nothing.

**Now the hardware.** Because tensor-core accumulation rounds toward zero (Fasi et al. 2021) and every $e_i > 0$, each of the $K-1$ additions loses up to $2u$ of magnitude *in the same direction*. There is no cancellation. The expected error is not $\sqrt{K}u$; it is $\approx K u \cdot \mathbb{E}[\text{fractional truncation}] \approx 0.5 \cdot 4096 \cdot 2^{-8} \approx 8$, i.e. $Z$ is systematically underestimated by an amount comparable to $Z$ itself. In practice the reduction saturates: once the running sum exceeds $2^{8}$ times a typical $e_i$, adding $e_i$ returns the running sum unchanged and the reduction silently stalls. Empirically this is the stagnation point at $K_{\text{stall}} \approx 1/u = 256$ for BF16 — matching DeepSeek-V3's need to promote every 128 elements, one binary order of magnitude inside it.

**The obstruction, made visible.** All four bounds above are for round-to-nearest. Every one of them is *not applicable* to the arithmetic that actually ran, and the applicable behaviour — one-signed RTZ stagnation — is not a $\sqrt{K}$ or a $\log K$ phenomenon at all; it is a hard cutoff at $K \approx 1/u$ that no published bound for tensor-core reductions states. The chunk size 128 in production kernels was found by measurement, and the theory that would have predicted it does not yet exist.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*