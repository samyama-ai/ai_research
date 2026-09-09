---
id: 12-quantization-compression/microscaling-block-size-optimality
title: "Microscaling Block Size Optimality"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Microscaling Block Size Optimality

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/microscaling-block-size-optimality` · **Status:** empirically-open

## 1. Problem Statement

Microscaling (MX) formats quantize a tensor in contiguous blocks of $k$ elements. Each block carries one shared scale plus $k$ low-precision elements. The block size $k$ is a free parameter that trades two things against each other: a smaller $k$ amortizes the scale over fewer elements (more bits per parameter) but tracks local dynamic range better (less clipping and less step-size waste).

Deployed choices disagree. OCP MX v1.0 fixes $k=32$ with an 8-bit power-of-two scale (E8M0). NVIDIA's NVFP4 on Blackwell uses $k=16$ with an FP8 E4M3 scale and a second per-tensor FP32 scale. QLoRA's NF4 uses $k=64$ with a quantized FP8 scale. GPTQ/AWQ deployments commonly use $k=128$ with an FP16 scale.

**The question:** for a fixed bits-per-parameter budget $b$, which $k$ minimizes end-task loss, and does the optimum depend on model scale, layer, or training-vs-inference use?

Three variants, different difficulty:

- **Measurement.** Does an isolated change in $k$, holding element format, scale format, scale count, rounding mode, and rotation fixed, change downstream loss by more than seed noise? *No published experiment does this cleanly.*
- **Method.** Given a hardware budget, pick $k$ and the scale encoding jointly to maximize quality per bit and per unit of tile-level dequantization cost.
- **Theory.** Characterize $k^\star(b)$ under a stated weight/activation distribution. Solvable in closed form under Gaussian iid; the closed form disagrees with practice, which is the interesting part.

Solved means: a reproduced curve of loss versus $k$ at matched $b$, with the scale format held fixed, at $\geq 7$B parameters, plus a stated mechanism for the shape.

## 2. Formal Setting

Let $w \in \mathbb{R}^n$ be a row of a weight matrix (or a slice of activations along the reduction axis), partitioned into $n/k$ contiguous blocks. For block $B = (w_1,\dots,w_k)$:

$$X = \mathcal{Q}_s\!\left(\frac{\max_i |w_i|}{\alpha}\right), \qquad \hat{w}_i = X \cdot \mathcal{Q}_e\!\left(\frac{w_i}{X}\right)$$

- $\mathcal{Q}_e$: element quantizer, an $b_e$-bit grid (FP4 E2M1, FP6 E2M3/E3M2, INT8). $\alpha$ is the largest representable element magnitude ($\alpha=6$ for E2M1, $7$ for symmetric INT4).
- $\mathcal{Q}_s$: scale quantizer with $b_s$ bits. E8M0 restricts $X$ to powers of two; E4M3 does not.

**Bits per parameter, as actually measured** (count stored bits, not nominal element width):

$$b(k) = b_e + \frac{b_s}{k} \;+\; \frac{b_t}{n}$$

with $b_t$ the per-tensor scale bits (0 for MX, 32 for NVFP4). MXFP4: $4 + 8/32 = 4.25$. NVFP4: $4 + 8/16 = 4.50$. NF4 with double quantization: $4 + 8/64 + 32/256 \approx 4.127$.

**Distortion.** Per-block MSE $D(k) = \frac{1}{k}\sum_i (w_i - \hat w_i)^2$; report SQNR $= 10\log_{10}(\mathbb{E}[w^2]/\mathbb{E}[D])$ in dB. End metric is validation cross-entropy $\mathcal{L}$ in nats/token on a held-out corpus, plus zero-shot accuracy; both must be reported with $\geq 3$ seeds.

**Objective:**

$$k^\star = \arg\min_k \; \mathcal{L}\big(\hat{\theta}_k\big) \quad \text{s.t.} \quad b(k) \leq b, \; \text{scale format fixed}$$

**Assumptions, and which fail.**

1. *Elements within a block are iid from a smooth unimodal density.* **Violated.** Transformer activations have persistent outlier channels (Dettmers et al., 2022) and attention-sink dimensions; weights inherit structure from these.
2. *Block boundaries are exchangeable with the data layout.* **Violated by construction** — MX blocks run along the reduction axis, so an outlier channel lands in the same lane of every block, not randomly.
3. *Quantization error is additive white noise independent of $w$.* **Violated** at $b_e = 4$; error is strongly correlated with the block max.
4. *Bits-per-parameter is the binding cost.* **Often false.** Blackwell's tensor cores fix $k$ in hardware; $k$ is not a free knob at inference time, only at format-design time.

## 3. State of the Art

**Established (ablated, multi-site).**

- The OCP Microscaling Formats (MX) Specification v1.0 (2023) standardizes $k=32$, E8M0. The design rationale — 32 amortizes the scale to 0.25 bits/element while fitting a hardware tile — is documented but the value 32 is not derived from a loss-versus-$k$ sweep in the public record.
- Rouhani et al., *Microscaling Data Formats for Deep Learning* (2023, arXiv:2310.10537), report direct-cast (no retraining) inference on GPT-class models up to ~7B: MXINT8 is near-lossless, MXFP6 within noise of FP32, MXFP4 needs quantization-aware training or fine-tuning to close the gap. All at $k=32$ only.
- Dettmers & Zettlemoyer, *The case for 4-bit precision* (ICML 2023), sweep bit-width across 19k runs, 125M–176B: 4-bit is the quality-per-bit optimum for inference, and small block size is one of the two interventions (with data types) that make 4-bit work. Their block sizes are 64/128 with FP16 scales, not MX-style E8M0.

**Claimed but unablated.**

- "NVFP4 beats MXFP4 at 4-bit." Reported in NVIDIA Blackwell technical material and echoed in follow-on work. Three variables move together: $k$ ($16$ vs $32$), scale format (E4M3 vs E8M0), and the extra per-tensor FP32 scale. Nothing published isolates $k$.
- "Block size 32 is the sweet spot." Asserted in format documentation; not backed by a published sweep at matched bits with fixed scale precision.

**Benchmark-number-only.** MXFP4 pretraining results (Tseng, Yu, Park, *Training LLMs with MXFP4*, 2025) show that stochastic rounding plus a random Hadamard transform on the backward pass recovers most of the BF16 loss curve at ~1B scale. This is a loss number at one $k$; it does not measure whether $k=16$ would have needed the rotation at all.

## 4. What Is Known

- **Scale overhead is exactly $b_s/k$.** Halving $k$ from 32 to 16 with an 8-bit scale costs 0.25 bits/param — 5.9% of a 4.25-bit budget.
- **Direct-cast MXINT8 ($k=32$) matches FP32** on generative inference for models to ~7B (Rouhani et al., 2023).
- **MXFP4 direct-cast degrades materially**; recovering it needs QAT or rotation-based smoothing. Measured at 1B–7B.
- **Rotations substitute partially for small blocks.** QuaRot (Ashkboos et al., NeurIPS 2024) shows a random Hadamard transform makes activations near-Gaussian and removes outliers, enabling 4-bit weights *and* activations on Llama-2 70B with a few-point WikiText-2 perplexity gap. Rotation is a competing fix for the same failure mode that small $k$ addresses — so the two are not independent knobs.
- **Group size effects saturate.** In GPTQ/AWQ-style pipelines at 3–4 bits, moving group size 128 → 64 typically buys a small perplexity gain and 64 → 32 buys less; the published curves use FP16 scales, so they do not transfer to E8M0.
- **Under Gaussian iid, larger $k$ wins at matched bits.** See §10 — the closed form says the loss from a bigger block grows like $\ln k$ while the bit saving is $b_s/k$; the crossover sits well above 32.

## 5. What Is Not Known

- **Empirically open.** The clean sweep — loss versus $k \in \{8,16,32,64,128\}$, matched bits-per-parameter, *one* scale format, one rounding mode, no rotation, $\geq 7$B parameters, $\geq 3$ seeds — has not been published. Runnable today on commodity clusters; the arithmetic can be emulated in BF16.
- **Empirically open.** Whether $k^\star$ drifts with model scale. Outlier magnitude grows with scale (Dettmers et al., 2022), which predicts $k^\star$ shrinks; nobody has measured the trend.
- **Empirically open.** Whether $k^\star$ differs between forward-pass inference and backward-pass gradients, where the distribution is heavier-tailed and stochastic rounding changes the error model.
- **Theoretically open.** No characterization of $k^\star$ under a realistic heavy-tailed, spatially structured weight model. The Gaussian answer is known and wrong.
- **Methodologically blocked.** "Quality per bit" is ill-defined once the scale is not the only overhead — NVFP4's per-tensor FP32 scale, double quantization, and layout padding all shift the denominator, and papers count them inconsistently.

## 6. Why It Is Hard

**Confounded measurement, at exactly the magnitude of the effect.** Every published $k$ comparison also changes the scale encoding. Under the Gaussian model, moving $k$ from 32 to 16 buys ~1.0 dB of SQNR. Replacing a power-of-two E8M0 scale with an E4M3 scale buys ~3.3 dB, because rounding the block max up to a power of two inflates the quantizer step by a factor uniform in $[1,2)$, costing $10\log_{10}\!\big(\int_0^1 4^u du\big) = 10\log_{10} 2.164 = 3.35$ dB on average. The confound is three times the size of the thing being measured. Any experiment that varies both attributes the scale-format win to block size.

Second obstruction: **the knob is not free at test time.** $k$ is baked into tensor-core datapaths. Sweeping it requires emulation, which costs 3–10× BF16 throughput and puts a 7B pretraining sweep at real money — enough that vendors publish the configuration they shipped rather than the sweep that justified it.

Third: **non-identifiability against rotation.** Small $k$ and Hadamard rotation both suppress the same outlier-driven range inflation. A sweep run with rotations on will report a flat $k$ curve; run with rotations off, a steep one. Neither is "the" answer without stating the pipeline.

## 7. Current Research (as of 2026)

- **Format co-design at vendors.** Microsoft (MX lineage, from MSFP through shared microexponents), NVIDIA (NVFP4, two-level scaling), AMD and Qualcomm as OCP MX signatories. Public output is specifications and benchmark tables, not sweeps.
- **FP4 training.** Quartet (ISTA/Red Hat, 2025) and MXFP4 pretraining with stochastic rounding and Hadamard backward passes — both aim at native 4-bit training; block size is held fixed at the hardware value. *(frontier — verify current scale claims.)*
- **Two-level and hierarchical scaling.** Per-block plus per-tensor scales, and micro-exponent variants that split the exponent across nested granularities. *(frontier — verify.)*
- **Precision scaling laws.** Kumar et al., *Scaling Laws for Precision* (2024, arXiv:2411.04330) fit loss as a function of bit-width and token budget; extending the same functional form to block size is the obvious open extension and is not yet done.

## 8. Concrete Next Experiment

**Scale.** Llama-3-8B-class dense transformer, 8B parameters, evaluated as (a) post-training direct-cast quantization and (b) 20B-token QAT fine-tune. Emulate MX arithmetic in BF16 — no Blackwell hardware needed.

**Arms.** Element format fixed at FP4 E2M1. Rotation off. Round-to-nearest-even. Scale format **fixed at E4M3 in all arms** (this is the control that kills the confound). Sweep $k \in \{8,16,32,64,128\}$, giving $b(k) \in \{5.0, 4.5, 4.25, 4.125, 4.0625\}$. Add a *bit-matched* row: $k=32$ with FP6 E2M3 elements at $b=6.25$ as an upper anchor, and $k=128$ with E8M0 as the single arm that varies the scale format, to size the confound directly.

**Control arm.** $k=32$/E8M0 — the shipped OCP configuration — plus BF16.

**The deciding number.** $\Delta\mathcal{L}(16 \to 32)$: the validation cross-entropy gap in nats/token between $k=16$ and $k=32$ at fixed E4M3 scales, over 3 seeds. If $|\Delta\mathcal{L}| < 0.005$ nats (roughly the seed spread at this scale), block size is not the reason NVFP4 outperforms MXFP4, and the field should stop attributing it there. If $\Delta\mathcal{L} > 0.02$ nats, $k=16$ is genuinely worth its extra 0.25 bits and OCP's $k=32$ is mis-set for 4-bit elements. Cost estimate: ~5 GPU-days for the direct-cast sweep, ~600 GPU-days for the QAT arm on H100s.

## 9. Key References

- **[Foundational]** Bita Darvish Rouhani et al. *Pushing the Limits of Narrow Precision Inferencing at Cloud Scale with Microsoft Floating Point.* NeurIPS, 2020.
- **[Foundational]** Bita Darvish Rouhani et al. *With Shared Microexponents, A Little Shifting Goes a Long Way.* ISCA, 2023. — arXiv:2302.08007
- **[SOTA / Spec]** Open Compute Project. *OCP Microscaling Formats (MX) Specification, Version 1.0.* 2023.
- **[SOTA]** Bita Darvish Rouhani et al. *Microscaling Data Formats for Deep Learning.* Technical report, 2023. — arXiv:2310.10537
- **[SOTA]** Tim Dettmers, Luke Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** Saleh Ashkboos et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[SOTA]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- **[SOTA]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** Ji Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[Related]** Tanishq Kumar et al. *Scaling Laws for Precision.* 2024. — arXiv:2411.04330
- **[Related]** Andrey Kuzmin et al. *FP8 Quantization: The Power of the Exponent.* NeurIPS, 2022. — arXiv:2208.09225
- **[Related]** Albert Tseng, Tao Yu, Youngsuk Park. *Training LLMs with MXFP4.* 2025. (identifier not verified here)

## 10. Worked Example

Take one block of $k$ weights, iid $\mathcal{N}(0,\sigma^2)$, symmetric INT4 elements ($\alpha = 7$), and an exact real-valued scale $X = \max_i|w_i|/7$. Step size $\Delta = X$, so MSE $\approx \Delta^2/12$. The expected block max is $\mathbb{E}[\max_i |w_i|] \approx \sigma\sqrt{2\ln k}$, giving

$$\mathrm{SQNR}(k) \;\approx\; \frac{\sigma^2}{\sigma^2 \cdot 2\ln k / (12 \cdot 49)} \;=\; \frac{294}{2\ln k}$$

| $k$ | bits/param | SQNR (dB) | $\Delta$ vs $k{=}16$ |
|---|---|---|---|
| 16 | 4.500 | 20.3 | — |
| 32 | 4.250 | 19.3 | $-1.0$ |
| 64 | 4.125 | 18.5 | $-1.8$ |
| 128 | 4.0625 | 17.8 | $-2.5$ |

Now spend the saved bits on elements instead. A scalar quantizer buys about 6 dB per bit. Going $16 \to 32$ frees 0.25 bits/param, worth ~1.5 dB of extra element precision, against a 1.0 dB block-max loss. **The Gaussian model says $k=32$ strictly beats $k=16$**, and by the same argument $k=64$ beats $k=32$ (0.125 bits ≈ 0.75 dB versus 0.8 dB lost — a near tie, so the model's optimum sits near $k \approx 64$).

Practice reports the opposite ordering for MXFP4 versus NVFP4. Two things explain the discrepancy, and they are not the same thing:

1. **The scale format, not $k$.** MXFP4's E8M0 scale rounds the block max up to a power of two, inflating $\Delta$ by a factor $2^u$, $u \sim U[0,1)$. Average penalty $10\log_{10}\!\big(3/(2\ln 2)\big) = 3.35$ dB. That alone reverses the table: $k{=}32$/E8M0 sits at $19.3 - 3.35 = 15.9$ dB, well below $k{=}16$/E4M3 at ~20.3 dB. The measured "block size effect" is 3× larger than the block size can account for.
2. **Non-Gaussian tails.** With outlier channels, $\mathbb{E}[\max]$ grows faster than $\sqrt{2\ln k}$ — polynomially for a heavy-tailed body — which steepens the $k$ curve. But this is exactly the effect a Hadamard rotation removes, so measuring it requires declaring rotation off.

The obstruction is visible in one line: every published MXFP4-vs-NVFP4 number confounds a ~1 dB effect with a ~3.3 dB one, and the field has been reading the sum as the first term.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*