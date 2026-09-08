---
id: 32-hardware-and-kernels/compute-optimal-format-below-fp4
title: "Compute-Optimal Arithmetic Format Below FP4"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Arithmetic Format Below FP4

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/compute-optimal-format-below-fp4` · **Status:** empirically-open

## 1. Problem Statement

FP4 (4 bits per element, block-scaled: MXFP4, NVFP4) is the narrowest format with shipping tensor-core support. The question is whether anything **narrower** — 3-bit, 2-bit, ternary, binary, or a mixed/split format at an average of $<4$ bits per element — is *compute-optimal* for training, meaning: for a fixed silicon area and energy budget, does the extra throughput bought by the narrower multiplier outweigh the loss degradation it causes?

Three variants, with different difficulty:

- **Measurement.** Given a format $f$, measure its *effective parameter multiplier* and *effective data multiplier* — how much a model trained in $f$ behaves like a smaller/less-trained BF16 model. Runnable today; contested because it requires fitting a scaling law per format.
- **Method.** Find the best training recipe (rounding, block scale, Hadamard/rotation transform, error feedback) at each bit-width. Open and active.
- **Theory.** Prove a bound relating quantization noise variance at $b$ bits to the irreducible term in the loss, i.e. predict the crossover without running the sweep. Open; nobody has a non-vacuous version.

**A solution** is a curve $b^\star(\text{hardware efficiency exponent})$ with confidence intervals, showing where the optimum sits and whether it is below 4.

## 2. Formal Setting

A format is $f = (m, e, b_{\text{blk}}, s)$: $m$ mantissa bits, $e$ exponent bits, block size $b_{\text{blk}}$ elements sharing one scale, scale encoded in $s$ bits. Measured bits per element:

$$B(f) \;=\; 1 + e + m + \frac{s}{b_{\text{blk}}}.$$

MXFP4 is $(m{=}1, e{=}2, b_{\text{blk}}{=}32, s{=}8)$ giving $B = 4.25$. NVFP4 is $(1,2,16,8)$ plus an FP32 per-tensor scale, $B = 4.5$. Ternary $\{-1,0,+1\}$ with a per-32 FP8 scale gives $B = \log_2 3 + 0.25 \approx 1.83$.

**Hardware efficiency** $S(f)$: sustained dense matmul throughput of format $f$ divided by that of BF16, on fixed area and power. Measured as achieved TFLOP/s on a large square GEMM, not from a spec sheet — a format with no native multiplier that is emulated by shifts must be scored at its emulated rate. Multiplier area scales roughly as $\Theta(m^2)$, so a mantissa-bit reduction gives superlinear area savings, but accumulator, scale-application, and memory-movement costs do not shrink, so $S(f)$ saturates.

**Loss model.** Following the effective-parameter parametrization used by Quartet (2025) and the precision scaling laws of Kumar et al. (ICLR 2025):

$$L(N, D, f) \;=\; \frac{A}{\big(\mathrm{ef}_N(f)\, N\big)^{\alpha}} \;+\; \frac{B}{\big(\mathrm{ef}_D(f)\, D\big)^{\beta}} \;+\; E,$$

with $\mathrm{ef}_N, \mathrm{ef}_D \in (0,1]$ fitted per format from a 2-D sweep over $N$ (parameters) and $D$ (tokens). $\mathrm{ef}_N(\text{BF16}) = \mathrm{ef}_D(\text{BF16}) = 1$ by construction.

**Compute-optimal format.** With budget $C$ in BF16-equivalent FLOPs, format $f$ buys $6ND = S(f)\cdot C$ raw FLOPs. Then

$$f^\star(C) \;=\; \arg\min_{f}\ \min_{6ND = S(f)C} L(N, D, f).$$

**Assumptions, and which are violated.**
1. *Separable degradation* — quantization enters only through $\mathrm{ef}_N, \mathrm{ef}_D$ and never through $E$ or the exponents $\alpha,\beta$. **Violated**: at very low precision the loss curve bends, i.e. $\alpha$ itself shifts.
2. *Format applied uniformly to all three GEMMs* (forward, dgrad, wgrad). **Violated in every published FP4 training run**: master weights, optimizer state, and often the backward pass sit at higher precision, so $S(f)$ measured end-to-end is far below the tensor-core ratio.
3. *Gaussian-ish, outlier-free block statistics.* **Violated**: activation outliers concentrate in a few channels; this is exactly what rotations are for.
4. *$S(f)$ is measurable for a format with no silicon.* **Violated below 4 bits** — this is the core obstruction (§6).

## 3. State of the Art

**Established (ablated, multiple groups).**
- MXFP4/NVFP4 forward-pass training with stochastic rounding and a random Hadamard transform reaches near-BF16 loss. Quartet (Castro, Panferov, et al., 2025) trains Llama-style models to 7B in native FP4 for both forward and backward GEMMs and reports FP4 as compute-optimal *under the Blackwell $2\times$ FP8 speed ratio*.
- Tseng, Yu, Park et al., *Training LLMs with MXFP4* (2025): stochastic rounding plus a Hadamard transform on the backward pass makes MXFP4 gradient GEMMs unbiased; they recover near-BF16 quality at roughly half the backward-GEMM cost.
- Post-training, sub-4-bit works: QuIP# and AQLM reach 2-bit weight-only inference with real but non-trivial loss increase.

**Claimed but unablated.**
- Vendor "FP4 training is $2$–$3\times$ faster" numbers are peak tensor-core ratios, not end-to-end token/s. Reported end-to-end speedups for FP4 pretraining are typically in the $1.2$–$1.6\times$ range once master weights, norms, attention softmax, and communication are excluded from the low-precision path. Where a paper reports a speedup, check whether it is a microbenchmark.
- BitNet b1.58 (Ma et al., 2024) claims ternary weights match FP16 at 3B. This is **weights-only, activations at 8 bits**, and the reported match is at a fixed $N$, not on an iso-compute frontier — so it is not evidence about $f^\star$.
- Every claimed sub-4-bit *training* result to date exists as a benchmark number on a specific model/token count, without the $N \times D$ sweep needed to fit $\mathrm{ef}_N, \mathrm{ef}_D$.

**Theory SOTA.** Kumar et al. (ICLR 2025) give a unified precision scaling law: post-training quantization degradation *grows* with the training token/parameter ratio, and compute-optimal *training* precision is roughly bit-width-independent near 7–8 bits under their hardware cost model, dropping when the cost model rewards narrow formats more steeply. Dettmers & Zettlemoyer (ICML 2023) show 4-bit is the zero-shot-accuracy-per-bit optimum for *inference* across 35k evaluations, 19M–176B parameters.

## 4. What Is Known

- **4-bit is the inference optimum.** Dettmers & Zettlemoyer: across OPT/BLOOM/Pythia/LLaMA at 19M–176B, 4-bit weights maximize zero-shot accuracy per total model bit; 3-bit is worse at every size tested.
- **MXFP4 numerics.** $B = 4.25$ bits/element; E2M1 has 15 representable values with max magnitude 6. Round-to-nearest MXFP4 gradients are biased; stochastic rounding + Hadamard removes the bias (Tseng et al., 2025).
- **Hardware ratio.** Blackwell-class GPUs advertise dense FP4 at $\approx 2\times$ dense FP8 and $\approx 4\times$ BF16 tensor-core throughput. There is no $2$-bit or $3$-bit tensor core in any shipping accelerator.
- **Multiplier energy.** Horowitz (ISSCC 2014, 45 nm): FP32 multiply 3.7 pJ, FP16 multiply 1.1 pJ, INT8 multiply 0.2 pJ; a 32-bit DRAM read is 640 pJ. Below ~4 bits the multiplier is already a rounding error against data movement — the saving must come from *memory and interconnect*, not arithmetic.
- **Scaling-law fits exist for FP4.** Quartet reports $\mathrm{ef}_N,\mathrm{ef}_D$ fits for FP4 vs FP8 vs BF16 up to 7B parameters. No comparable fit is published for any format below 4 bits with quantized activations *and* gradients.

## 5. What Is Not Known

- **Empirically open (primary).** No published $N \times D$ scaling sweep for FP3, FP2, or ternary training with all three GEMMs quantized. The experiment is runnable — a 30M–1B sweep costs on the order of $10^{21}$ FLOPs — and nobody has run it with a matched recipe.
- **Empirically open.** Whether $\mathrm{ef}_D$ (data efficiency) collapses faster than $\mathrm{ef}_N$ below 4 bits. Kumar et al. imply it should; this predicts sub-4-bit training gets *worse* as token budgets grow, which is the regime everyone is in.
- **Methodologically blocked.** $S(f)$ for a format with no silicon. Simulated formats run *slower* than BF16, so the compute-optimality claim depends entirely on an assumed, unvalidated $S(f)$ — a free parameter that determines the answer.
- **Theoretically open.** No bound of the form "quantization noise of variance $\sigma^2$ per weight raises the irreducible loss $E$ by at least $g(\sigma^2)$". Without it, the crossover cannot be predicted, only measured.
- **Methodologically blocked.** Fair accounting of the block scale. At $b_{\text{blk}}=16$ and $s=8$, the scale is 0.5 bits/element — 25% of a 2-bit format's budget. Papers routinely quote "2-bit" and mean 2.5.

## 6. Why It Is Hard

The specific obstruction is **the hardware efficiency term is unmeasurable and dominates the answer**. $f^\star$ is decided by $S(f)$, but $S(f)$ for a sub-4-bit format can only be estimated from an area/energy model of hypothetical silicon, and reasonable models disagree by $2\times$. Multiplier area falls as $\Theta(m^2)$ but accumulators stay FP32, block-scale application is a per-block multiply that does not shrink, and below ~4 bits the GEMM becomes bandwidth- and accumulator-bound, so $S$ saturates near $1.2$–$1.5\times$ over FP4 rather than $2\times$. Since the loss penalty *keeps growing* below 4 bits while the speedup saturates, the sign of the optimum depends on an unmeasurable quantity.

Secondary: **confounded measurement.** A sub-4-bit run that matches BF16 almost always keeps some path at higher precision. Unless the paper reports the fraction of total FLOPs actually executed in $f$, its speedup claim and its quality claim are about two different systems.

## 7. Current Research (as of 2026)

- **ISTA / Red Hat (Alistarh group)** — Quartet and successors: native FP4 training, effective-parameter scaling laws, and the "is FP4 optimal?" framing itself. Extensions to 3-bit are the natural next step *(frontier — verify)*.
- **Cornell / AWS (Tseng, De Sa)** — unbiased low-precision GEMMs via rotations and stochastic rounding; QuIP#/QTIP lineage on the inference side.
- **NVIDIA** — NVFP4 pretraining at multi-billion scale; the $b_{\text{blk}}=16$ + two-level scaling design point is the current production answer.
- **Microsoft** — microscaling (MX) format standardization via OCP; BitNet ternary line.
- **Harvard / CMU** — nanoscaling variants (NxFP-style: adaptive microexponents, shared-mantissa tricks) targeting the 3-bit regime *(frontier — verify)*.
- Open question being actively argued: whether the right sub-4-bit move is *uniform 3-bit* or *mixed 4/2-bit per-layer*, since the average bits are the same but the hardware is not.

## 8. Concrete Next Experiment

**Scale.** Train a Llama-style dense decoder at $N \in \{50\text{M}, 150\text{M}, 400\text{M}, 1.2\text{B}\}$ non-embedding parameters, each at $D \in \{4, 12, 40\} \times N$ tokens (12 runs per format) on a fixed corpus. Total: about $4\times10^{21}$ FLOPs per format, roughly 2k H100-days for all four arms.

**Arms.** Four formats, all with forward, dgrad, and wgrad GEMMs quantized, identical recipe (random Hadamard transform, stochastic rounding on backward, $b_{\text{blk}}=32$, FP8 E4M3 scales): BF16 (**control**), MXFP4 ($B=4.25$), MXFP3 = E2M0 ($B=3.25$), ternary + FP8 scale ($B\approx1.83$). Report the FLOP fraction executed in-format for each arm; it must exceed 0.9.

**Deciding number.** Fit $\mathrm{ef}_N(f)$ and $\mathrm{ef}_D(f)$ per arm. The single decisive quantity is the **bit-normalized efficiency ratio**

$$R(f) \;=\; \frac{\mathrm{ef}_N(f)^{\alpha}\,\mathrm{ef}_D(f)^{\beta}}{\ \mathrm{ef}_N(\text{MXFP4})^{\alpha}\,\mathrm{ef}_D(\text{MXFP4})^{\beta}}\ \Big/\ \frac{B(\text{MXFP4})}{B(f)}.$$

If $R(\text{MXFP3}) > 1$, a 3-bit format beats FP4 under a bandwidth-proportional speedup model and the sub-4-bit regime is live. If $R(\text{MXFP3}) < 1$ — the degradation outruns the bit saving even before accounting for saturating $S(f)$ — the question is settled negatively and no silicon investment below FP4 is warranted for training. Publish $R$ with bootstrap CIs; a result whose CI straddles 1 at this scale tells you the sweep must extend to 7B.

## 9. Key References

- **[Foundational]** T. Dettmers, L. Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Foundational]** T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[SOTA — theory]** T. Kumar, Z. Ankner, B. F. Spector, B. Bordelon, N. Muennighoff, M. Paul, C. Pehlevan, C. Ré, A. Raghunathan. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330
- **[SOTA — systems]** R. L. Castro, A. Panferov, S. Tabesh, O. Sieberling, J. Chen, M. Nikdan, S. Ashkboos, D. Alistarh. *Quartet: Native FP4 Training Can Be Optimal for Large Language Models.* 2025. — arXiv:2505.14669
- **[SOTA — method]** A. Tseng, T. Yu, Y. Park. *Training LLMs with MXFP4.* 2025. — arXiv:2502.20586
- **[Format spec]** B. D. Rouhani et al. *Microscaling Data Formats for Deep Learning.* 2023. — arXiv:2310.10537
- **[Format spec]** B. D. Rouhani et al. *With Shared Microexponents, A Little Shifting Goes a Long Way.* ISCA, 2023.
- **[Foundational]** P. Micikevicius et al. *FP8 Formats for Deep Learning.* 2022. — arXiv:2209.05433
- **[Related]** S. Ma, H. Wang, L. Ma, L. Wang, W. Wang, S. Huang, L. Dong, R. Wang, J. Xue, F. Wei. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* 2024. — arXiv:2402.17764
- **[Related]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024. — arXiv:2402.04396
- **[Hardware cost model]** M. Horowitz. *Computing's Energy Problem (and what we can do about it).* ISSCC, 2014.
- **[Standard]** Open Compute Project. *OCP Microscaling Formats (MX) Specification v1.0.* 2023.

## 10. Worked Example

Take a $1.2$B-parameter model at $D = 24$B tokens, and compare MXFP4 against a hypothetical MXFP3 (E2M0, one sign + two exponent bits, no mantissa; $B = 3.25$ with an FP8 scale per 32 elements).

**Step 1 — bits saved.** $4.25 \to 3.25$ is a $23.5\%$ reduction in element storage. Weight+activation traffic falls by the same factor, so a *bandwidth-bound* GEMM gets $S = 1.31\times$.

**Step 2 — arithmetic saved.** E2M1 has a 1-bit mantissa; the multiplier is already a 2×2 table lookup. E2M0 removes the mantissa entirely, making the multiply a pure exponent add. Multiplier area drops maybe $40\%$, but at 4 bits the multiplier is under $15\%$ of tensor-core area (accumulator + scale path + register file dominate). Net area saving: $\approx 6\%$. So **the arithmetic contributes almost nothing**; the speedup is essentially the $1.31\times$ bandwidth term, and that is an upper bound since accumulators do not shrink. Call $S = 1.2\times$ realistically.

**Step 3 — quality cost.** E2M0 represents only $\{\pm 1, \pm 2, \pm 4, \pm 8\}$ (8 values vs 15) — one bit of dynamic range but *zero* relative resolution, giving $100\%$ worst-case relative error between adjacent codes versus $33\%$ for E2M1. Under the standard fit ($\alpha \approx 0.34$, $\beta \approx 0.28$), a modest $\mathrm{ef}_N = 0.75$, $\mathrm{ef}_D = 0.80$ for MXFP3 relative to MXFP4 gives an effective compute penalty of $\mathrm{ef}_N^{\alpha}\mathrm{ef}_D^{\beta} = 0.75^{0.34}\cdot 0.80^{0.28} = 0.906 \cdot 0.939 = 0.851$.

**Step 4 — the crossover.** MXFP3 wins iff $S \cdot 0.851 > 1$, i.e. $S > 1.18$. The realistic estimate is $S \approx 1.2$.

**The obstruction, made visible.** The decision turns on whether $S$ is $1.18$ or $1.20$ — a 2-point gap inside the error bar of any area model for silicon that does not exist. And the $0.851$ penalty is an assumption, not a measurement: nobody has fitted $\mathrm{ef}_N,\mathrm{ef}_D$ for a fully-quantized 3-bit training run. Both sides of the inequality are guesses of similar magnitude, which is precisely why the problem is empirically open rather than merely unresolved.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*