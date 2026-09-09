---
id: 12-quantization-compression/optimal-codebook-learning-weights
title: "Optimal Codebook Learning for Vector Quantized Weights"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Codebook Learning for Vector Quantized Weights

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/optimal-codebook-learning-weights` · **Status:** open

## 1. Problem Statement

Given a trained network's weights, vector quantization (VQ) replaces each $d$-dimensional group of weights with an index into a learned codebook. The problem: **choose the codebook — its dimension, size, structure, and entries — to minimize end-task loss at a fixed total bit budget, where the budget includes the codebook itself.**

Three variants, routinely conflated:

- **Measurement.** What is the right distortion to minimize? Layerwise output error is a proxy for task loss. No one has shown the proxy's minimizer is the task loss minimizer at 2 bits/weight.
- **Method.** Given a fixed objective, find the codebook. This is $k$-means with a non-Euclidean (Hessian-weighted) metric and a rate constraint — NP-hard in general, solved by heuristics.
- **Theory.** How much can VQ beat scalar quantization on weights at 2–4 bits/weight, and is that gain reachable under hardware decode constraints? At high rate the answer is capped (§6). At 2 bits it is open.

Solving it means: a procedure that, given $(W, \text{calibration data}, \text{bit budget})$, returns a codebook provably or reproducibly within a stated margin of the best achievable at that budget — with the codebook storage counted.

## 2. Formal Setting

Let $W \in \mathbb{R}^{m \times n}$ be one linear layer's weights and $X \in \mathbb{R}^{n \times N}$ the calibration activations. Partition each row into $n/d$ contiguous groups of dimension $d$, giving $M = mn/d$ source vectors $\{w_i\}$.

A codebook is $C = \{c_1,\dots,c_K\} \subset \mathbb{R}^d$ with assignment $a: [M] \to [K]$, reconstruction $\hat w_i = c_{a(i)}$.

**Layerwise objective (what is actually optimized):**
$$\mathcal{L}_{\text{layer}}(C,a) = \|(W - \hat W)X\|_F^2 = \sum_{r=1}^{m} (W_r - \hat W_r)\, H \,(W_r - \hat W_r)^\top, \quad H = XX^\top .$$
Measured as: run $N \approx 128$–$4096$ sequences of length 2048 from a calibration corpus, accumulate $H$ in fp32, add damping $\lambda \operatorname{tr}(H)/n$ with $\lambda \approx 10^{-2}$ for invertibility.

**Rate (what must be counted, and often isn't):**
$$R = \underbrace{\frac{\log_2 K}{d}}_{\text{index}} + \underbrace{\frac{K d\, b_c}{n_{\text{shared}}}}_{\text{codebook}} + \underbrace{\frac{b_s}{g}}_{\text{scales}} \quad \text{bits/weight},$$
with $b_c$ the codebook entry precision (typically 16), $n_{\text{shared}}$ the number of weights sharing that codebook, $b_s$ scale precision and $g$ the scaling group size. Measured as: total serialized checkpoint bytes divided by parameter count. Any paper reporting only $\log_2 K / d$ is reporting a lower bound, not the rate.

**End objective (what is claimed):** $\Delta\mathcal{L} = \mathcal{L}_{\text{task}}(\hat\theta) - \mathcal{L}_{\text{task}}(\theta)$, measured as WikiText-2 perplexity at seqlen 4096 plus zero-shot accuracy on a fixed suite.

**Assumptions, and their status:**

| Assumption | Status |
|---|---|
| Layerwise error is a faithful proxy for $\Delta\mathcal{L}_{\text{task}}$ | Violated — errors compound across depth; the map is monotone in practice but not calibrated |
| $H$ from calibration generalizes to deployment distribution | Violated — $H$ is data-dependent; outlier channels shift across domains |
| Weight groups are i.i.d. samples from a fixed density | Violated — strong cross-column structure, which is exactly what $d>1$ exploits |
| Cross-layer independence (each layer quantized separately) | Violated by construction; sequential methods (GPTQ-style) partly compensate |
| Codebook storage is negligible | Violated at $d \le 8$, $K \ge 2^{16}$ (see §10) |

## 3. State of the Art

**Empirical SOTA (post-training, LLMs).** At $\approx 2$ bits/weight on Llama-2-70B, three families are within noise of each other on WikiText-2:

- **AQLM** (Egiazarian et al., ICML 2024) — additive quantization, sum of $M$ codes from separate codebooks, plus block-wise fine-tuning of codebooks and non-quantized parameters. Reported ~3.9 ppl at 2-bit vs ~3.1 fp16.
- **QuIP#** (Tseng et al., ICML 2024) — randomized Hadamard incoherence processing plus a fixed $E_8$-lattice codebook ($E_8P$), so the codebook is *not* learned at all; only the rotation and scales are. Reported ~4.0 ppl at 2-bit.
- **QTIP** (Tseng et al., NeurIPS 2024) — trellis-coded quantization: an implicit, compute-generated codebook of effectively unbounded dimension, dominating both above at matched rate.
- **VPTQ** (Liu et al., EMNLP 2024) — second-order-guided centroid initialization with residual codebooks.

**Established:** incoherence processing (random rotation) before quantization reliably improves 2-bit results, and is ablated in QuIP/QuIP#. Fine-tuning codebook entries after assignment recovers a large fraction of the gap, ablated in AQLM.

**Claimed but unablated:** that the *learned* codebook is what buys the gain. QuIP#'s hand-designed lattice matching learned AQLM codebooks is direct evidence against it, but no paper runs the clean cross-ablation (learned vs. lattice vs. Gaussian-random codebook, same rotation, same fine-tuning, same rate accounting).

**Benchmark-number-only results:** nearly all 2-bit LLM perplexities. They are single-seed, single-calibration-set numbers on one corpus; run-to-run variance from calibration sampling is rarely reported.

**Theory SOTA.** Zador's high-resolution theorem, the Gish–Pierce bound on entropy-constrained scalar quantization, and the $1.53$ dB asymptotic space-filling limit (Gray & Neuhoff, *Quantization*, IEEE Trans. Inf. Theory 1998). QuIP (Chee et al., NeurIPS 2023) gives the only proxy-error guarantee tied to a practical LLM method, via incoherence.

## 4. What Is Known

- **$k$-means is NP-hard**, even in the plane for general $k$ (Mahajan et al., *The planar $k$-means problem is NP-hard*, TCS 2012) and in general dimension for $k=2$ (Aloise et al., *Machine Learning* 2009). Lloyd's algorithm converges only to a local optimum.
- **Asymptotic space-filling gain is $\le 1.53$ dB $= 0.2546$ bits/dimension**, as $d\to\infty$, over entropy-constrained scalar quantization on a smooth source. This is a hard cap on the *dimension* benefit at high rate.
- **Dimension helps, with diminishing returns, at LLM scale.** GPTVQ (van Baalen et al., 2024) measured $d = 1 \to 2 \to 4$ on Llama-family models: most of the gain appears at $d=2$, and $d=4$ adds materially less.
- **Learned codebooks are not required.** QuIP#'s $E_8P$ codebook is fixed a priori and matches learned-codebook methods at 2 bits on Llama-2-7B/13B/70B.
- **Codebook collapse is real and quantifiable.** In VQ-VAE-style *trained* quantizers, active-code fraction is commonly well under 50%; FSQ (Mentzer et al., ICLR 2024) reports near-100% utilization by removing the learned codebook entirely. Straight-through estimator pathologies are documented by Huh et al. (ICML 2023).
- **Rate–quality scaling.** Dettmers & Zettlemoyer (ICML 2023) found 4-bit to be near the accuracy-per-bit optimum for scalar PTQ across 19M–176M–66B parameter models; Kumar et al. (ICLR 2025) give precision-aware scaling laws in which post-training-quantization damage *grows* with pretraining tokens.

## 5. What Is Not Known

- **Theoretically open.** No finite-rate (non-asymptotic) bound on the VQ-over-scalar gain for a Hessian-weighted distortion at $R \approx 2$ bits/weight. High-resolution theory does not apply when $K^{1/d}$ is $O(1)$ per dimension. No approximation guarantee for Hessian-weighted $k$-means with a rate constraint.
- **Empirically open.** The cross-ablation isolating the codebook's contribution — learned vs. lattice vs. random, holding rotation, fine-tuning, group size, and *true serialized rate* fixed — has never been run at 70B. It is runnable today on 8×H100 in days.
- **Empirically open.** Whether any 2-bit VQ gain survives when the codebook budget is instead spent on scalar quantization at a higher rate. Nobody reports the iso-bytes control.
- **Methodologically blocked.** "Optimal" has no agreed measurement. WikiText-2 perplexity at 2 bits differentiates methods by $<0.2$ ppl while calibration-seed variance is unreported; and the proxy $\mathcal{L}_{\text{layer}}$ has no established transfer function to $\Delta\mathcal{L}_{\text{task}}$.

## 6. Why It Is Hard

The specific obstruction is **an accounting trap plus a capped prize**.

The theoretical gain from raising VQ dimension is bounded by $0.2546$ bits/weight asymptotically. The codebook needed to approach that bound costs $Kd b_c / n_{\text{shared}}$ bits/weight, which at practically useful $(K,d)$ is *the same order as the prize* (§10). Methods therefore share codebooks across large weight blocks, which reintroduces exactly the distribution mismatch the codebook was learned to remove. So the design space is squeezed from both sides, and the residual differences between methods are smaller than the measurement's resolution.

Secondary: **non-identifiability**. Codebook, assignment, per-group scale, and the incoherence rotation are jointly degenerate — a rotation absorbed into $C$ leaves $\hat W$ unchanged. Ablating "the codebook" alone is therefore ill-posed unless the rotation and scales are pinned, which no published ablation does.

## 7. Current Research (as of 2026)

- **Implicit / compute-generated codebooks.** QTIP's trellis line (Cornell, De Sa group) removes the storage term entirely by generating codewords from a hash of the state. This is the most direct attack on the accounting trap. Extensions to larger trellises and hardware kernels are active *(frontier — verify)*.
- **Lattice and structured codebooks.** $E_8$, Barnes–Wall, and residual-lattice constructions; zero learned storage, decode by table-free arithmetic.
- **Quantization-aware pretraining with VQ.** Whether codebooks learned during training beat post-hoc ones at matched rate is being explored; results so far are at $\le 7$B *(frontier — verify)*.
- **Groups:** IST Austria (Alistarh), Yandex Research, Cornell (De Sa), Qualcomm AI Research, Microsoft Research (VPTQ), Berkeley (SqueezeLLM line).

## 8. Concrete Next Experiment

**Question:** at 2 bits/weight, does *learning* the codebook beat a fixed lattice once serialized bytes are matched?

**Scale.** Llama-3.1-8B and Llama-3.1-70B. Fixed calibration: 256 sequences × 8192 tokens from RedPajama; 5 calibration seeds.

**Arms** (identical randomized Hadamard rotation, identical block fine-tuning schedule, identical scale group size, rate measured as serialized checkpoint bytes / parameter count, target $2.00 \pm 0.01$ bits/weight):

1. Learned codebook, $d=8$, $K=2^{16}$, Hessian-weighted $k$-means + fine-tune (AQLM-style).
2. **Control:** fixed $E_8P$ lattice codebook, zero learned entries, same rate — codebook bytes reallocated to a finer scale grid.
3. **Control:** codebook entries drawn i.i.d. Gaussian, frozen, same rate.
4. **Control (iso-bytes, cross-family):** scalar quantization at whatever bit width consumes the same total bytes ($\approx 2.19$ bits/weight for the 8B arm).

**Deciding number.** Mean WikiText-2 perplexity at seqlen 8192 across 5 calibration seeds, reported with its standard deviation. **If arm 1 does not beat arm 2 by more than $2\sigma$ of the seed variance, codebook learning is not the mechanism** and the field's 2-bit gains are attributable to rotation, fine-tuning, and scale allocation. Secondary readout: arm 3 vs arm 2 separates "any dense $d=8$ code" from "good lattice geometry".

Cost estimate: ~4 GPU-days per arm at 8B, ~40 at 70B; ~$10^3$ GPU-hours total.

## 9. Key References

- **[Foundational]** Lloyd, S. *Least Squares Quantization in PCM.* IEEE Transactions on Information Theory, 1982.
- **[Foundational]** Gray, R. M., Neuhoff, D. L. *Quantization.* IEEE Transactions on Information Theory, 44(6), 1998. — source of the $1.53$ dB space-filling limit.
- **[Foundational]** Jégou, H., Douze, M., Schmid, C. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI, 2011.
- **[Foundational]** Babenko, A., Lempitsky, V. *Additive Quantization for Extreme Vector Compression.* CVPR, 2014.
- **[Foundational]** Han, S., Mao, H., Dally, W. *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR, 2016. — arXiv:1510.00149
- **[Foundational]** van den Oord, A., Vinyals, O., Kavukcuoglu, K. *Neural Discrete Representation Learning.* NeurIPS, 2017. — arXiv:1711.00937
- **[SOTA]** Egiazarian, V., Panferov, A., Kuznedelev, D., Frantar, E., Babenko, A., Alistarh, D. *Extreme Compression of Large Language Models via Additive Quantization.* ICML, 2024. — arXiv:2401.06118
- **[SOTA]** Tseng, A., Chee, J., Sun, Q., Kuleshov, V., De Sa, C. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024. — arXiv:2402.04396
- **[SOTA]** Tseng, A., Sun, Q., Hou, D., De Sa, C. *QTIP: Quantization with Trellises and Incoherence Processing.* NeurIPS, 2024. — arXiv:2406.11235
- **[SOTA]** van Baalen, M., et al. *GPTVQ: The Blessing of Dimensionality for LLM Quantization.* Qualcomm AI Research, 2024. — arXiv:2402.15319
- **[SOTA]** Liu, Y., et al. *VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models.* EMNLP, 2024. — arXiv:2409.17066
- **[Theory]** Chee, J., Cai, Y., Kuleshov, V., De Sa, C. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees.* NeurIPS, 2023. — arXiv:2307.13304
- **[Theory]** Mahajan, M., Nimbhorkar, P., Varadarajan, K. *The Planar $k$-Means Problem is NP-Hard.* Theoretical Computer Science, 2012.
- **[Related]** Mentzer, F., Minnen, D., Agustsson, E., Tschannen, M. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR, 2024. — arXiv:2309.15505
- **[Survey]** Dettmers, T., Zettlemoyer, L. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Survey]** Kumar, T., et al. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330

## 10. Worked Example

Take `down_proj` of Llama-2-7B: $W \in \mathbb{R}^{11008 \times 4096}$, $mn = 45{,}088{,}768$ weights.

**Budget at 2.00 bits/weight:** $90{,}177{,}536$ bits $= 11.27$ MB.

**A $d=8$, $K=2^{16}$ codebook** gives index rate $\log_2 K / d = 16/8 = 2.00$ bits/weight — the whole budget, before storing the codebook. The codebook itself is $65536 \times 8 \times 16 = 8{,}388{,}608$ bits $= 1.05$ MB. Shared across the entire layer, that is
$$\frac{8{,}388{,}608}{45{,}088{,}768} = 0.186 \text{ bits/weight}.$$

Now the cap. For a smooth source, entropy-constrained *scalar* quantization is within $0.2546$ bits/sample of the rate–distortion function at high rate (Gish–Pierce), and unbounded-dimension VQ closes at most that gap. So:

- Prize available to VQ over good scalar coding: **$0.2546$ bits/weight**.
- Price paid to store this one codebook: **$0.186$ bits/weight** — 73% of the prize.

Concretely on a unit Gaussian: Lloyd–Max scalar at 2 bits gives $D = 0.1175$ (9.30 dB); entropy-constrained scalar at rate 2 gives $D \approx 2^{-2(2-0.2546)} = 0.089$ (10.5 dB); the Shannon bound is $D = 2^{-4} = 0.0625$ (12.04 dB). The entire remaining VQ headroom is that last 1.53 dB — and the codebook eats three quarters of it.

**What this makes visible.** Papers reporting "2-bit" while excluding codebook bytes are comparing at $2.19$ effective bits against a $2.00$-bit scalar baseline, inside a regime where the honest maximum advantage is $0.25$ bits. That is why QTIP's storage-free trellis codebook wins, and why the learned-vs-lattice question in §8 is not yet answered: the reported gains are the same size as the unreported accounting error.

**Caveat that keeps the problem open:** the $0.2546$-bit cap is a *high-resolution* result. At $K^{1/d} = 2^{16/8} = 4$ points per dimension, the asymptotics are not tight, and low-rate VQ can exceed it. Nobody has computed the finite-rate bound for the Hessian-weighted objective at $R=2$. That computation, or the experiment in §8, is the next move.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*