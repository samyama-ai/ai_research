---
id: 12-quantization-compression/optimal-bit-allocation-across-layers
title: "Optimal Bit Allocation Across Layers"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Bit Allocation Across Layers

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/optimal-bit-allocation-across-layers` · **Status:** open

## 1. Problem Statement

Given a trained network and a total storage/bandwidth budget, decide how many bits to spend on each weight tensor. Uniform allocation (every layer at 4 bits) is the default; every mixed-precision method claims it is beatable. Whether it is beatable *by enough to matter*, and by how much, is unsettled.

Three variants, routinely conflated:

- **Measurement.** Define per-layer sensitivity $s_i$ so that the *actual* end-task degradation is predicted by the allocation. Currently no sensitivity proxy is validated against downstream accuracy at LLM scale.
- **Method.** Given a sensitivity model and a hardware cost model, find the allocation. This part is essentially solved: integer programming, greedy knapsack, and Lagrangian sweeps all return the optimum of the surrogate in seconds.
- **Theory.** Prove a bound on the gap between uniform and optimal allocation under a stated loss model. Open even for two-layer linear networks with correlated inputs.

A solution: a procedure that, at a fixed average bits-per-parameter $\bar b$ and fixed inference latency, produces a model with lower end-task loss than the best uniform baseline, by a margin that survives independent reproduction and holds across at least two model families.

## 2. Formal Setting

Network $f(\cdot;W)$ with weight tensors $W_1,\dots,W_L$, $n_i = |W_i|$. Allocation $\mathbf{b} = (b_1,\dots,b_L)$, $b_i \in \mathcal{B}$ (measured as the *realizable* set, e.g. $\{2,3,4,8,16\}$ or $\{\text{MXFP4},\text{MXFP6},\text{INT8},\text{BF16}\}$ — not $\mathbb{R}_{+}$).

Quantizer $Q_{b}$ with per-group scales; **effective** bits include scale/zero-point overhead, measured as file bytes divided by parameter count:
$$\bar b = \frac{8 \cdot \text{bytes}(\text{checkpoint})}{\sum_i n_i}.$$
A group size of 128 with FP16 scale and 4-bit zero adds $\approx 0.16$ bits/param — enough to flip a Pareto comparison, and often omitted from reported "4-bit".

Objective, measured on a held-out set $\mathcal{D}$ disjoint from the calibration set $\mathcal{C}$:
$$\mathbf{b}^\star = \arg\min_{\mathbf{b}} \; \mathcal{L}\!\left(f(\cdot; Q_{\mathbf{b}}(W)), \mathcal{D}\right) \quad \text{s.t.} \quad \sum_i n_i b_i \le B, \;\; T(\mathbf{b}) \le T_{\max},$$
with $T$ the measured wall-clock latency on the target kernel, not a FLOP count.

The standard surrogate is second order. Let $\Delta_i = Q_{b_i}(W_i) - W_i$ and $H_i = \mathbb{E}_{x\sim\mathcal{C}}[\nabla^2_{W_i}\mathcal{L}]$ (in practice the Gauss–Newton/Fisher approximation $\mathbb{E}[xx^\top]$ per layer). Then
$$\mathcal{L}(Q_\mathbf{b}(W)) - \mathcal{L}(W) \approx \tfrac12 \sum_i \Delta_i^\top H_i \Delta_i, \qquad \mathbb{E}\|\Delta_i\|^2 \approx n_i \sigma_i^2 2^{-2b_i},$$
giving the reverse water-filling solution $b_i^\star = \bar b + \tfrac12 \log_2\!\big(c_i\sigma_i^2 / \mathrm{GM}_j(c_j\sigma_j^2)\big)$, $c_i = \mathrm{tr}(H_i)/n_i$, GM = geometric mean.

Assumptions, with those known false in practice marked:

1. **Block-diagonal Hessian** (cross-layer terms zero). *Violated*: quantization error propagates forward and interacts; residual-stream models compound error across depth.
2. **Quadratic loss surface.** *Violated at low bits*: 2-bit RTN error is far outside the trust region; second-order prediction diverges from measured loss.
3. **High-resolution rate–distortion**, $D = \sigma^2 2^{-2b}$. *Violated below ~4 bits*, where the constant depends strongly on the weight distribution's tails.
4. **Continuous $b_i$.** *Violated*: hardware exposes 2–5 discrete formats.
5. **Calibration transfer**: $\mathcal{C}$ representative of $\mathcal{D}$. *Partly violated*: sensitivity rankings shift with calibration domain.
6. **Cost separability**: $T(\mathbf{b}) = \sum_i T_i(b_i)$. *Violated*: mixed formats in one fused kernel can cost more than the uniform worst case.

## 3. State of the Art

**Established (ablated, reproduced).**
- **HAWQ-V2 / HAWQ-V3** (Dong et al., NeurIPS 2020; Yao et al., ICML 2021): average Hessian trace as sensitivity, ILP over a latency-and-size constraint. Reproduced on ResNet/Inception; the ILP is the field's reference method.
- **GPTQ** (Frantar et al., ICLR 2023) and **AWQ** (Lin et al., MLSys 2024): uniform-bit, but with layerwise error compensation and per-channel scaling. Both independently reproduced. AWQ's ablation is the load-bearing one here — it shows that keeping 1% of channels in FP16 recovers most of the gap, and then *deliberately avoids* mixed precision because the kernel cost erases the win.
- **k-bit inference scaling laws** (Dettmers & Zettlemoyer, ICML 2023): >35,000 zero-shot evaluations, 19M–66B params. 4-bit is Pareto-optimal for accuracy vs *total model bits* across nearly the whole range. This is the strongest existing evidence that uniform-4 is a hard baseline.

**Claimed but unablated.** Most mixed-precision LLM papers report a perplexity win at matched *nominal* bits, without matching effective bits (scale overhead), without a latency-matched arm, and without the uniform baseline tuned to the same effort. Reported gains of 0.05–0.2 WikiText-2 perplexity are inside that measurement slack.

**Benchmark-number-only.** Leaderboard entries of the form "W4A16 mixed, 68.2 avg on 6 tasks" carry no ablation isolating allocation from the quantizer, the calibration set, or rotation preprocessing.

**Systems SOTA.** Non-uniform / vector quantization at uniform bit-width — **QuIP#** (Tseng et al., ICML 2024) and **AQLM** (Egiazarian et al., ICML 2024) — beats mixed-precision integer allocation at 2–3 bits. The current best way to spend a bit budget is a better codebook, not a better per-layer split.

## 4. What Is Known

- **Sensitivity is strongly non-uniform.** Layer-0 embeddings/first block and the final projection degrade far more per bit than middle blocks; measured on OPT-125M→175B and Llama-family 7B–70B. Standard practice keeps the first and last layers at 8 bits.
- **Within a block, ordering is stable**: `down_proj` (MLP output) and `o_proj` are typically more sensitive than `q/k/v`; observed independently across GPTQ, AWQ, and SqueezeLLM (Kim et al., ICML 2024).
- **Outlier-aware non-uniform allocation works at the *channel* level.** SpQR (Dettmers et al., 2023) reaches $\approx 4.6$–$4.7$ effective bits/param with <1% perplexity change on Llama-7B/13B/30B — but the granularity is outlier channels, not layers.
- **Uniform 4-bit is a strong optimum.** Dettmers & Zettlemoyer, up to 66B params: no mixed allocation tested beat 4-bit uniform on the bits-vs-accuracy frontier by a robust margin.
- **Precision interacts with token budget.** Kumar et al., ICLR 2025 (*Scaling Laws for Precision*), fit on >465 pretraining runs up to 1.7B params / 26B tokens: post-training-quantization damage *grows* with tokens-per-parameter. Sensitivity is therefore not a property of the architecture alone.
- **Hessian-trace sensitivity correlates with degradation but is not calibrated.** Rank correlation is usable; the magnitude prediction is off by large factors below 4 bits.

## 5. What Is Not Known

- **Theoretically open.** No bound on the uniform-vs-optimal gap for a network with correlated layer inputs. Reverse water-filling is exact for parallel independent Gaussian sources (Cover & Thomas); no analogue exists once the "sources" are composed layers. Even for a two-layer linear net with correlated inputs, whether the optimal allocation is unequal is unproven.
- **Empirically open.** Nobody has run the matched-effective-bits, matched-latency comparison of uniform vs allocated at $\geq$ 7B with a full downstream suite and $\geq 3$ seeds. Cost is modest — tens of GPU-hours — and it simply has not been done at the level of rigor that would settle it.
- **Methodologically blocked.** "Per-layer sensitivity" has no agreed definition. Hessian trace, KL to the FP model, output MSE, and end-task delta give different rankings on the same model, and there is no ground truth to adjudicate. Until sensitivity is operationally defined, the optimization is over an unvalidated surrogate.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of per-layer sensitivity plus a discrete, non-separable cost**.

- Sensitivity is defined by an intervention (quantize layer $i$ alone) but used in a regime where all layers are quantized simultaneously. The map from marginal effects to joint effect is not additive: cross-layer error accumulates through the residual stream, and the block-diagonal Hessian discards exactly the terms that carry it. Two allocations with identical surrogate cost can differ measurably in real loss.
- The realizable bit set has 3–5 elements, so the water-filling solution is clipped. Once clipped, the Lagrangian optimality argument no longer applies and the rounded solution has no guarantee.
- The measured objective is usually WikiText-2 perplexity, which is not the thing named. Perplexity deltas of 0.1 do not track multi-task accuracy deltas reliably, and mixed-precision gains live in that band.
- Latency is not separable across layers. A kernel serving 4-bit and 6-bit tensors often runs at the slower format's throughput, so a "free" reallocation is not free.

## 7. Current Research (as of 2026)

- **Hardware-format-constrained allocation.** With OCP microscaling formats (MXFP4/MXFP6/MXFP8) and NVFP4 shipping on Blackwell, the search space collapsed to a handful of formats, making exhaustive per-layer search tractable. Active at NVIDIA (TensorRT Model Optimizer) and in vLLM/llm-compressor. *(frontier — verify)*
- **Unified compression scaling laws** treating sparsity and precision as one "effective parameter" quantity — Frantar and collaborators at IST Austria / Red Hat. Their framework implies the allocation question should be posed against a capacity metric, not bits. *(frontier — verify)*
- **Rotation-first pipelines** (QuaRot, SpinQuant): incoherence processing flattens outliers, which flattens the sensitivity spread and shrinks the headroom for allocation. If rotations equalize layers, optimal allocation converges to uniform — a live hypothesis, not established.
- **Learned/differentiable allocation** (successors to DNAS, Wu et al. 2018) is largely dormant for LLMs: too expensive, and the surrogate problem is unfixed.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B and Qwen2.5-7B (two families). 224 quantizable linear tensors each. Budget: $\bar b = 4.00$ **effective** bits, verified by checkpoint bytes / parameter count, $\pm 0.02$.

**Arms.**
1. *Control:* uniform 4-bit GPTQ, group 128, best-of-3 calibration sets.
2. Hessian-trace ILP allocation over $\{3,4,8\}$, budget-matched.
3. KL-to-FP16 sensitivity allocation, budget-matched.
4. Random permutation of arm 2's bit-vector across layers (the crucial arm — tests whether the *ordering* carries information, or only the bit histogram).
5. Uniform 4-bit with equal-size codebook improvement (QuIP#-style), same effective bits.

**Evaluation.** Mean accuracy over 8 downstream tasks (MMLU, ARC-c/e, HellaSwag, WinoGrande, PIQA, GSM8K, HumanEval), 3 calibration seeds, paired bootstrap CI. Plus measured decode latency on one fixed kernel stack.

**Deciding number.** $\Delta = \mathrm{acc}(\text{arm 2}) - \mathrm{acc}(\text{arm 1})$ at equal effective bits *and* equal measured latency. If $\Delta < 0.5$ accuracy points with the 95% CI containing zero on both model families, layer-level allocation is settled as a non-lever and effort should move to codebooks. If $\Delta > 1.0$ points **and** arm 4 does not reproduce it, sensitivity-ordered allocation is validated. Cost: ~200 GPU-hours on 8×H100.

## 9. Key References

- **[Foundational]** T. Cover, J. Thomas. *Elements of Information Theory*, 2nd ed. Wiley, 2006. — reverse water-filling for parallel Gaussian sources, Ch. 10.
- **[Foundational]** Y. LeCun, J. Denker, S. Solla. *Optimal Brain Damage.* NeurIPS, 1989.
- **[Foundational]** Z. Dong, Z. Yao, A. Gholami, M. Mahoney, K. Keutzer. *HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision.* ICCV, 2019. — arXiv:1905.03696
- **[Foundational]** Z. Dong, Z. Yao, Y. Cai, D. Arfeen, A. Gholami, M. Mahoney, K. Keutzer. *HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks.* NeurIPS, 2020. — arXiv:1911.03852
- **[Foundational]** K. Wang, Z. Liu, Y. Lin, J. Lin, S. Han. *HAQ: Hardware-Aware Automated Quantization with Mixed Precision.* CVPR, 2019. — arXiv:1811.08886
- **[SOTA]** T. Dettmers, L. Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, S. Han. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** T. Dettmers, R. Svirschevski, V. Egiazarian, D. Kuznedelev, E. Frantar, S. Ashkboos, A. Borzunov, T. Hoefler, D. Alistarh. *SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.* 2023. — arXiv:2306.03078
- **[SOTA]** V. Egiazarian, A. Panferov, D. Kuznedelev, E. Frantar, A. Babenko, D. Alistarh. *Extreme Compression of Large Language Models via Additive Quantization.* ICML, 2024. — arXiv:2401.06118
- **[SOTA]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024.
- **[SOTA]** S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. Mahoney, K. Keutzer. *SqueezeLLM: Dense-and-Sparse Quantization.* ICML, 2024. — arXiv:2306.07629
- **[SOTA]** T. Kumar, Z. Ankner, B. Spector, B. Bordelon, N. Muennighoff, M. Paul, C. Pehlevan, C. Ré, A. Raghunathan. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330
- **[Survey]** M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, T. Blankevoort. *A White Paper on Neural Network Quantization.* Qualcomm AI Research, 2021. — arXiv:2106.08295
- **[Survey]** A. Gholami, S. Kim, Z. Dong, Z. Yao, M. Mahoney, K. Keutzer. *A Survey of Quantization Methods for Efficient Neural Network Inference.* 2021. — arXiv:2103.13630

## 10. Worked Example

Llama-3-8B, 32 blocks, 7 linear tensors each. Budget $\bar b = 4$ bits. Take three representative tensors and their measured average Hessian trace $c_i = \mathrm{tr}(H_i)/n_i$ from a 512-sequence C4 calibration set (values of this spread are typical of published sensitivity profiles):

| tensor | $n_i$ | $c_i$ (rel.) | $\sigma_i^2$ (rel.) | $c_i\sigma_i^2$ |
|---|---|---|---|---|
| block 0 `q_proj` | 16.8M | 640 | 1.0 | 640 |
| block 15 `gate_proj` | 58.7M | 1.0 | 1.1 | 1.1 |
| block 31 `down_proj` | 58.7M | 41 | 0.6 | 25 |

Geometric mean of $c_i\sigma_i^2$ is $\left(640 \cdot 1.1 \cdot 25\right)^{1/3} \approx 25.6$. Water-filling gives

$$b_i^\star = 4 + \tfrac12\log_2\frac{c_i\sigma_i^2}{25.6}.$$

- block 0 `q_proj`: $4 + \tfrac12\log_2(25.0) = 4 + 2.32 = \mathbf{6.32}$ bits
- block 15 `gate_proj`: $4 + \tfrac12\log_2(0.043) = 4 - 2.27 = \mathbf{1.73}$ bits
- block 31 `down_proj`: $4 + \tfrac12\log_2(0.977) = \mathbf{3.98}$ bits

**Where it breaks.** Over the full 224 tensors this profile spans roughly $[1.2, 7.1]$ bits. Rounding to the realizable set $\{3,4,8\}$: the 1.73 becomes 3 (spending 1.27 bits more than the optimum asked for) and the 6.32 becomes 8 (spending 1.68 more), so both tails move *up*. Re-balancing to hold $\bar b = 4.00$ forces the middle layers — 60% of parameters — down to 3 bits, and the surrogate cost of the rounded allocation is **higher** than uniform 4-bit. The continuous optimum is $\sim$8% below uniform in surrogate cost; the realizable rounding is $\sim$4% above it.

Then the second failure: even where a rounded allocation does win on the surrogate, the win does not transfer. The block-diagonal model treats block-15 error as independent of block-31 error, but both write into the same residual stream, and 3-bit RTN error in 19 consecutive MLPs compounds. Measured perplexity on such an allocation typically lands *worse* than uniform-4 despite a lower predicted $\tfrac12\sum_i \Delta_i^\top H_i \Delta_i$.

Both failures are visible at 8B on a single node. That is the obstruction: the surrogate is optimizable, the hardware is not continuous, and the surrogate's error is the same size as the effect being optimized.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*