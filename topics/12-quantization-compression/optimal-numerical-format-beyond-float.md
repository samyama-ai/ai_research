---
id: 12-quantization-compression/optimal-numerical-format-beyond-float
title: "Optimal Numerical Format Beyond Floating Point"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Numerical Format Beyond Floating Point

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/optimal-numerical-format-beyond-float` · **Status:** open

## 1. Problem Statement

IEEE-754 floating point was designed in 1985 for scientific computing: wide dynamic range, graceful underflow, tight per-operation error bounds. Neural network training and inference need none of those properties in the form FP provides them. The question is what *should* replace it.

Three variants, with different difficulty:

- **Measurement variant.** Given a fixed bit budget $b$ per stored value and a fixed hardware cost model, which encoding minimizes end-task loss? Currently unresolved because no agreed cost model exists that spans silicon area, memory bandwidth, and accuracy simultaneously.
- **Method variant.** Construct a format — codebook shape, block-scaling granularity, exponent/mantissa split, or a non-scalar (lattice / trellis) code — that Pareto-dominates INT, FP, and microscaling formats at matched $b$ and matched hardware cost. Empirically runnable today.
- **Theory variant.** Is there a characterization of the loss-optimal $2^b$-point codebook for the weight and activation distributions that actually arise in trained transformers, as opposed to for an assumed Gaussian source under squared error? Theoretically open, and entangled with an open lattice-quantization problem (Gersho's conjecture) in the vector case.

Solving it means: a format $F^\star$ with a proof or a reproduced multi-scale ablation showing it dominates FP8/INT8/MXFP4/NVFP4 at equal effective bits *and* equal multiply-accumulate area.

## 2. Formal Setting

A **format** at $b$ bits is a pair $(Q, D)$ with encoder $Q:\mathbb{R}\to\{0,\dots,2^b-1\}$ and decoder $D$ inducing a codebook $\mathcal{C}=\{c_0,\dots,c_{2^b-1}\}\subset\mathbb{R}$. Real deployments add **block scaling**: a tensor is partitioned into groups of size $g$, each with scale $s_j$ stored in $b_s$ bits. Measured storage is

$$b_{\text{eff}} = b + \frac{b_s}{g}.$$

This is the quantity to hold fixed when comparing formats; comparing "FP4 vs INT4" at nominal $b=4$ while one uses $g=16$ and the other $g=128$ is not a comparison.

Let $w\in\mathbb{R}^n$ be a weight tensor, $\hat w = D(Q(w))$. Two objectives, routinely conflated:

$$\text{(proxy)}\quad \mathcal{D} = \tfrac1n\|w-\hat w\|_2^2, \qquad \text{(target)}\quad \Delta\mathcal{L} = \mathcal{L}(\hat\theta) - \mathcal{L}(\theta)$$

with $\mathcal{L}$ the held-out cross-entropy in nats/token, measured on $\ge 10^7$ tokens of a corpus disjoint from calibration data. To second order, $\Delta\mathcal{L}\approx\frac12(\hat\theta-\theta)^\top H(\hat\theta-\theta)$, so $\mathcal{D}$ predicts $\Delta\mathcal{L}$ only when $H\propto I$ — false for transformers, where Hessian eigenspectra span several orders of magnitude.

Hardware cost is measured, not assumed: MAC-array area in $\mu m^2$ at a named process node, and energy in pJ/op from synthesis, plus HBM bytes moved per token. The **decision predicate**: $F_1 \succ F_2$ iff $\Delta\mathcal{L}(F_1)<\Delta\mathcal{L}(F_2)$ at equal $b_{\text{eff}}$ and equal area *and* equal energy.

Assumptions the literature rests on, and their status:

| Assumption | Status |
|---|---|
| Weights are i.i.d. Gaussian within a block | Violated — per-channel outliers, heavy tails in activations |
| Squared error is the right distortion | Violated — Hessian-weighted error is a strictly better predictor |
| Scalar (per-element) quantization is optimal | Violated — QuIP\#/QTIP beat scalar codes at 2–3 bits |
| Hardware cost $\propto$ bit width | Violated — FP MACs cost more area than INT MACs at equal width |

## 3. State of the Art

**Established.**
- FP8 (E4M3 for weights/activations, E5M2 for gradients) trains models to parity with BF16; codified in the OCP FP8 spec (Micikevicius et al., 2022) and used end to end in DeepSeek-V3 (671B params, 14.8T tokens, 2024). Reproduced across vendors.
- Microscaling (MX) formats — block size $g=32$, shared E8M0 power-of-two scale — are an OCP standard; MXFP4/MXFP6 inference results (Rouhani et al., 2023) are reproduced in shipping silicon.
- Non-scalar codes win at low rate. QuIP\# (Tseng et al., ICML 2024) uses an $E_8$-lattice codebook; QTIP (Tseng et al., NeurIPS 2024) uses trellis-coded quantization and improves on it at 2–4 bits. This is the clearest demonstration that *the format itself*, not just the rounding algorithm, carries accuracy.

**Claimed but unablated.**
- NF4 ("4-bit NormalFloat", Dettmers et al., QLoRA, NeurIPS 2023) is described as information-theoretically optimal for normally distributed weights. The optimality claim is for a quantile construction under a Gaussian assumption, not a proof of loss-optimality; the head-to-head against FP4 is a benchmark number on a fixed finetuning suite, not an area-matched ablation.
- Posit arithmetic (Gustafson & Yonemoto, 2017) claims to beat float "at its own game". For DL workloads the claim has never been supported by an area-matched, scale-matched training comparison; the critique in Dinechin et al. (CoNGA 2019) remains unanswered for ML.
- BitNet b1.58 (Ma et al., 2024) reports ternary $\{-1,0,1\}$ weights matching FP16 LLaMA perplexity at 3B params / 100B tokens. Trained from scratch, so not comparable to PTQ formats, and not independently reproduced at $\ge$7B with matched token budgets.

**Benchmark-number-only.** NVFP4 ($g=16$, E4M3 scale) vs MXFP4 ($g=32$, E8M0 scale) accuracy comparisons are vendor-reported; the confound between block size and scale format has not been separated in public work.

## 4. What Is Known

- **Scalar rate-distortion for a Gaussian source is closed.** Lloyd–Max optimal MSE for unit-variance Gaussian: $0.1175$ at 2 bits, $0.03454$ at 3, $0.009497$ at 4, $0.002499$ at 5 (Max, 1960). Slope $\approx 6.1$ dB/bit.
- **4 bits is the inference Pareto point for uniform formats.** Dettmers & Zettlemoyer (ICML 2023) swept 35,000+ zero-shot evaluations across OPT/BLOOM/Pythia/LLaMA from 19M to 176B params: 4-bit maximizes accuracy at fixed total model bits; 3-bit loses more than the size saves.
- **INT8 is not worse than FP8 for PTQ inference.** van Baalen et al. (2023) found INT8 matches or beats FP8-E4M3 on most networks post-training, and that FP8 MAC hardware is materially larger than INT8 at equal width. The format advantage attributed to FP8 is largely a dynamic-range advantage that per-channel INT scaling also buys.
- **Precision interacts with token budget.** Kumar et al. (ICLR 2025), 465 pretraining runs up to 1.7B params / 26B tokens: post-training quantization damage grows with tokens-per-parameter, so a format ranked at Chinchilla ratios can invert at 100× overtraining.
- **Lattices beat scalars at low rate.** QuIP\#/QTIP recover usable 2-bit Llama-2-70B where scalar 2-bit collapses.

## 5. What Is Not Known

- **Theoretically open.** The optimal $d$-dimensional lattice quantizer is unknown for $d\ge 4$ (Gersho's conjecture; settled only for $d=2$). So the achievable low-rate ceiling for vector formats is unproven. Also open: any characterization of the loss-optimal codebook under a non-isotropic Hessian, rather than under MSE.
- **Empirically open.** No public study varies codebook shape, block size $g$, and scale format $b_s$ orthogonally at fixed $b_{\text{eff}}$, across $\ge 3$ model scales, with pretraining from scratch. Every ingredient is runnable now; the factorial design has not been run.
- **Methodologically blocked.** "Hardware cost" has no shared definition. Without a published area/energy model per format at a fixed node, "Pareto-optimal format" is not a well-posed measurement, and vendor claims are unfalsifiable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**. A format ships as a bundle: codebook shape + block size + scale encoding + rounding algorithm + which tensors are excluded. Published comparisons vary several at once, so the accuracy delta is not attributable to the codebook. Worse, the effects are near-degenerate: as Section 10 shows, the entire MSE spread between the best and worst 4-bit scalar codebook on Gaussian data is smaller than the effect of changing $g$ from 64 to 32. Two very different formats can be made to swap rankings by a block-size choice, so the codebook is not identifiable from the reported end-task number.

Second obstruction: **cost of the deciding experiment**. Deciding the training-format question requires pretraining from scratch, and Kumar et al. show the answer depends on tokens/parameter — so a single Chinchilla-ratio run does not settle it. A credible sweep is $\ge 10^{21}$ FLOPs.

## 7. Current Research (as of 2026)

- **Microscaling refinement.** OCP MX working group (Microsoft, NVIDIA, AMD, Arm, Intel, Meta, Qualcomm) on second-generation block formats; NVFP4-style small blocks with FP scales are the current direction in Blackwell-class silicon.
- **Trellis and lattice codes for weights.** Cornell (Tseng, De Sa) — QTIP line, pushing structured codes with hardware-cheap decode.
- **Hessian-aware codebook design.** Descendants of GPTQ (IST Austria, Alistarh group) and AWQ (MIT HAN Lab) increasingly optimize the codebook jointly with the rounding, blurring "format" and "algorithm".
- **Sub-4-bit native training.** Follow-ons to BitNet and to FP8 training at frontier scale; MXFP4/NVFP4 *training* recipes are the live frontier *(frontier — verify: public, reproduced MXFP4 pretraining at $\ge$70B is not established)*.
- **Log-domain and hybrid arithmetic.** LNS-style multiplier-free designs remain a niche with strong energy claims and weak accuracy ablations.

## 8. Concrete Next Experiment

**Question:** does codebook shape matter at all, once block scaling is held fixed?

- **Scale.** Pretrain from scratch at 1.4B params on 300B tokens (≈215 tokens/param, deliberately overtrained so the Kumar et al. effect is visible). Repeat at 350M / 75B tokens for a scaling arm. ~5 runs of each.
- **Arms**, all at $b_{\text{eff}}=4.25$ exactly ($b=4$, $b_s=16$, $g=64$), weights-only, same rounding (RTN), same excluded tensors (embeddings, lm_head, layernorms in BF16):
  1. INT4 (uniform) — **control arm**.
  2. FP4-E2M1.
  3. NF4 (quantile codebook).
  4. Lloyd–Max-4 codebook fit per model on the actual weight histogram.
  5. Hessian-weighted Lloyd codebook (diagonal Fisher from 128 calibration sequences).
- **Cross arm.** INT4 at $g=32,\ b_s=8$ (E8M0), i.e. $b_{\text{eff}}=4.25$ again — isolates block-scaling against codebook shape at identical storage.
- **Deciding number.** $\Delta\mathcal{L}$ in nats/token on 10M held-out tokens, mean over 3 seeds. **If the spread across arms 1–5 is below the spread between arm 1 and the cross arm, codebook shape is not the lever and the field should stop optimizing it.** Predicted from Section 10: arms 1–5 span $\le 0.010$ nats; the cross arm differs by more.

Cost: roughly 6 GPU-months on H100-class hardware. This is affordable and has not been run.

## 9. Key References

- **[Foundational]** J. Max. *Quantizing for minimum distortion.* IRE Transactions on Information Theory, 1960.
- **[Foundational]** S. P. Lloyd. *Least squares quantization in PCM.* IEEE Transactions on Information Theory, 1982.
- **[Foundational]** R. M. Gray, D. L. Neuhoff. *Quantization.* IEEE Transactions on Information Theory, 44(6), 1998.
- **[Foundational]** J. H. Conway, N. J. A. Sloane. *Sphere Packings, Lattices and Groups.* Springer, 3rd ed., 1999.
- **[Foundational]** P. Micikevicius et al. *Mixed Precision Training.* ICLR 2018 — arXiv:1710.03740
- **[SOTA]** P. Micikevicius et al. *FP8 Formats for Deep Learning.* 2022 — arXiv:2209.05433
- **[SOTA]** B. Darvish Rouhani et al. *Microscaling Data Formats for Deep Learning.* 2023 — arXiv:2310.10537
- **[SOTA]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024 — arXiv:2402.04396
- **[SOTA]** A. Tseng, Q. Sun, D. Hou, C. De Sa. *QTIP: Quantization with Trellises and Incoherence Processing.* NeurIPS 2024 — arXiv:2406.11235
- **[SOTA]** T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023 — arXiv:2305.14314
- **[Key empirical]** T. Dettmers, L. Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML 2023 — arXiv:2212.09720
- **[Key empirical]** T. Kumar et al. *Scaling Laws for Precision.* ICLR 2025 — arXiv:2411.04330
- **[Key empirical]** M. van Baalen et al. *FP8 versus INT8 for efficient deep learning inference.* 2023 — arXiv:2303.17951
- **[Alternative format]** J. Gustafson, I. Yonemoto. *Beating Floating Point at its Own Game: Posit Arithmetic.* Supercomputing Frontiers and Innovations, 4(2), 2017.
- **[Critique]** F. de Dinechin, L. Forget, J.-M. Muller, Y. Uguen. *Posits: the good, the bad and the ugly.* CoNGA 2019.
- **[Survey]** A. Gholami, S. Kim, Z. Dong, Z. Yao, M. Mahoney, K. Keutzer. *A Survey of Quantization Methods for Efficient Neural Network Inference.* In *Low-Power Computer Vision*, Chapman & Hall/CRC, 2022 — arXiv:2103.13630

## 10. Worked Example

Take one block of $g=64$ weights drawn i.i.d. $\mathcal{N}(0,1)$, absmax-scaled. For $n=64$, $\mathbb{E}[\max|z|]\approx 2.75$.

*Derived (nearest-neighbour MSE against the standard normal, unit variance):*

| 4-bit codebook | MSE | SQNR |
|---|---|---|
| FP4-E2M1, levels $\{0,\pm.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}/6\times 2.75$ | 0.0121 | 19.2 dB |
| INT4 uniform, step $2\times2.75/15=0.367$ | 0.0112 | 19.5 dB |
| NF4 (quantile) | $\approx$0.0100 | 20.0 dB |
| Lloyd–Max-4 (optimal scalar) | 0.009497 | 20.2 dB |

The entire span from the worst shipping 4-bit format to the *provably optimal* scalar code is **1.05 dB**, or **0.17 equivalent bits**.

Now price the bookkeeping. Going from $g=64$ to $g=32$ with an FP16 scale changes $b_{\text{eff}}$ from $4.25$ to $4.50$. At the Lloyd slope of 6.1 dB/bit, that 0.25 bit is worth **1.5 dB** — more than the total spread across every codebook in the table.

**The obstruction, made visible:** any published comparison of FP4 against INT4 or NF4 that does not hold $g$ and $b_s$ fixed is measuring block-scaling granularity and reporting it as codebook shape. The signal being argued over (1.05 dB) is smaller than the confound (1.5 dB). That is why fifteen years of format proposals have not converged, and why the Section 8 cross arm — not another codebook — is the experiment that matters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*