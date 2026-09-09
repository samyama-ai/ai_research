---
id: 32-hardware-and-kernels/nonuniform-quantization-kernel-design
title: "Quantization-Aware Kernel Design for Non-Uniform Formats"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Quantization-Aware Kernel Design for Non-Uniform Formats

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/nonuniform-quantization-kernel-design` · **Status:** open

## 1. Problem Statement

Non-uniform quantization formats — NF4, k-means codebooks, lattice codebooks (E8), additive/vector codebooks, trellis codes — reach lower perplexity per bit than uniform integer grids. Uniform formats reach higher throughput, because dequantization is an affine map that costs two or three ALU ops and because integer tensor-core paths exist in silicon. The problem is whether the accuracy advantage survives contact with the kernel.

Three variants, different difficulty:

- **Measurement.** Given a format $Q$, a GPU, a shape, and a batch size, report the *achieved* tokens/s and the *achieved* accuracy under one fixed serving stack. Most published comparisons vary format, kernel author, and stack together, so the reported deltas are not attributable. Not yet standard practice.
- **Method.** Design a codebook $\mathcal{C}$ and a matching kernel that is simultaneously (a) within $\epsilon$ of the memory-bandwidth roofline at decode batch sizes and (b) strictly Pareto-better than group-wise int4 on the (bits/weight, perplexity) plane. Open at $\le 3$ bits.
- **Theory.** Characterize which codebook families admit a dequantization map computable in $O(1)$ register-resident instructions per weight — i.e., which non-uniform grids are *kernel-cheap* — and prove a lower bound on distortion for that restricted family. No such characterization exists.

Solving it means: a format whose kernel hits $\ge 90\%$ of roofline at batch 1–32 *and* beats int4-g128 on perplexity at equal bits/weight, on the same stack, with the source released.

## 2. Formal Setting

Let $W \in \mathbb{R}^{m \times n}$ be a weight matrix and $X \in \mathbb{R}^{n \times b}$ activations at batch $b$. A format is a triple $(\mathcal{C}, \pi, s)$: a codebook $\mathcal{C} \subset \mathbb{R}^d$ of size $K$ over vectors of dimension $d$ ($d=1$ is scalar quantization), an index map $\pi: \mathbb{R}^d \to [K]$, and per-group scales $s$.

**Bits per weight**, as actually stored:
$$ \beta = \frac{\log_2 K}{d} + \frac{b_s}{g} $$
with $b_s$ scale bits and group size $g$. NF4 with fp16 scales at $g=64$: $\beta = 4 + 16/64 = 4.25$. QuIP\# E8P at 2 bits with fp16 group scales at $g=256$ is $\approx 2.06$, plus Hadamard-factor storage.

**Arithmetic intensity** at the L2/HBM boundary:
$$ I(b) = \frac{2mnb}{\beta mn/8 + 2nb + 2mb} \;\xrightarrow{b=1}\; \frac{16}{\beta}\ \text{FLOP/byte} $$
so int4 decode GEMV sits at $\approx 3.8$ FLOP/byte against an A100 ridge point of $\approx 160$ — deeply memory-bound. The roofline time is $T_{\text{mem}} = \beta mn/(8 \cdot \text{BW}_{\text{eff}})$, where $\text{BW}_{\text{eff}}$ is *measured* streaming bandwidth (A100-80GB: $\approx 1.55$–$1.65$ TB/s against 1.94 peak), not the datasheet number.

**Dequantization cost** is the quantity the field routinely omits. Let $c_{\text{alu}}$ be issued ALU instructions per weight and $c_{\text{lds}}$ bytes of shared-memory traffic per weight. Dequant time is
$$ T_{\text{deq}} = mn \cdot \max\!\left( \frac{c_{\text{alu}}}{R_{\text{alu}}},\ \frac{c_{\text{lds}} \cdot \phi}{R_{\text{lds}}} \right) $$
with $R_{\text{alu}}$ the INT32 issue rate ($\approx 9.7 \times 10^{12}$/s on A100), $R_{\text{lds}}$ shared-memory bandwidth ($\approx 19.5$ TB/s aggregate), and $\phi \ge 1$ the mean bank-conflict factor for the access pattern — $\phi$ depends on the *data*, since codebook indices are the addresses. The **roofline efficiency** to report is $\eta = T_{\text{mem}} / T_{\text{measured}}$.

**Accuracy** is $\Delta = \mathrm{PPL}_Q - \mathrm{PPL}_{\text{fp16}}$ on a fixed corpus (WikiText-2, 4096-token context) plus a zero-shot suite, both at a fixed calibration set.

Assumptions and their violations:
1. *Weights are the only traffic.* Violated once KV cache dominates: at 8k context, batch 32, Llama-2-7B, KV traffic exceeds weight traffic and $\eta$ stops predicting tokens/s.
2. *$\phi = 1$.* Violated for $K \ge 256$ scalar LUTs; conflicts are input-dependent and vary across layers.
3. *Dequant overlaps memory.* Violated when the LUT occupies shared memory that the kernel needs for tiling, cutting occupancy.
4. *Perplexity ranks methods.* Weakly violated: at 2 bits, PPL gaps of $<0.15$ do not reliably order zero-shot accuracy.

## 3. State of the Art

**Empirical/systems SOTA.** Marlin (Frantar et al., PPoPP 2025) holds close to $4\times$ over fp16 for *uniform* int4-g128 up to batch $\approx 32$ — the strongest published evidence that uniform formats can be run at near-roofline efficiency across the batch range, not just at batch 1. This is established: the kernel is open, ablated against vLLM/ExLlama baselines, and independently rebuilt in serving stacks.

For non-uniform formats, LUT-GEMM (Park et al., ICLR 2024) and FLUTE (Guo et al., EMNLP Findings 2024) are the reference kernels; FLUTE reports $2$–$4\times$ over fp16 GEMM and $\approx 1.5$–$2\times$ over prior LUT kernels, with an end-to-end Gemma-2 result. SqueezeLLM (Kim et al., ICML 2024) reports $\approx 2.3\times$ over fp16 for 3-bit non-uniform on an A6000. T-MAC (EuroSys 2025) does the analogous thing on CPU with table lookup.

**Accuracy SOTA.** QuIP\# (Tseng et al., ICML 2024) and QTIP (Tseng et al., NeurIPS 2024) with incoherence processing plus lattice/trellis codebooks, and AQLM (Egiazarian et al., ICML 2024) with additive codebooks, define the 2-bit frontier. These are genuine, reproduced accuracy wins over GPTQ/AWQ at 2 bits.

**Claimed but unablated.** Speedups for the high-accuracy 2-bit methods are usually reported as end-to-end tokens/s against an fp16 baseline, not as $\eta$ against the format's own roofline. AQLM's and QuIP\#'s reported $\approx 2$–$3\times$ decode speedups are benchmark numbers at batch 1; there is no published ablation isolating $T_{\text{deq}}$, and no head-to-head at batch $>8$ against Marlin int4 in one stack. Whether the 2-bit accuracy win converts into a serving win is, as of 2026, unshown.

## 4. What Is Known

- **NF4 beats int4 on accuracy at matched bits** for 4-bit weight-only, measured on LLaMA 7B–65B (Dettmers et al., NeurIPS 2023). The gap is small — fractions of a perplexity point — and shrinks with group-wise int4 scaling.
- **Incoherence processing plus vector codebooks is worth $\ge 1$ PPL at 2 bits.** QuIP\# reports Llama-2-70B 2-bit WikiText-2 PPL near $\approx 4$ versus fp16 $\approx 3.1$, where round-to-nearest int2 collapses entirely; QTIP improves further. Scale: 7B–70B.
- **Uniform int4 is essentially free on accuracy above 4 bits.** GPTQ (ICLR 2023) and AWQ (MLSys 2024) hold within $\approx 0.1$–$0.3$ PPL of fp16 at 4 bits, 7B–70B.
- **The 4-bit decode roofline is reachable.** Marlin sustains near-$4\times$ to batch 32; the accepted lesson is that the win comes from asynchronous global-to-shared copy, double buffering, and a dequant path of $\approx 2$–$3$ ALU ops, not from the format.
- **Hardware has moved toward block-scaled, near-uniform formats.** OCP MX (Rouhani et al., 2023) and the FP4/FP6/FP8 tensor-core paths in Blackwell make *some* non-uniformity (the FP mantissa/exponent grid) free while leaving arbitrary codebooks unsupported.

## 5. What Is Not Known

- **Theoretically open.** No characterization of *kernel-cheap* codebooks: which $\mathcal{C}$ admit dequantization in $O(1)$ register ops with no shared-memory indirection, and what the minimum achievable distortion is within that class. The E8P construction is an existence proof for one point, not a theory. Also open: a distortion–rate lower bound for codebooks constrained to be expressible as a fixed low-degree polynomial or a `prmt`/`lop3`-realizable byte permutation.
- **Empirically open.** Nobody has published a single-stack, fixed-hardware comparison of {int4-g128 Marlin, NF4, SqueezeLLM-3bit, AQLM-2bit, QTIP-2bit} sweeping batch $1 \to 128$ and reporting $\eta$ per format. The experiment is runnable in GPU-weeks. It is unrun mainly because each format ships its own kernel of differing maturity.
- **Methodologically blocked.** "Cost of a format" has no accepted definition. Reported speedups mix kernel engineering effort, baseline choice, and hardware generation. Until $\eta$ (roofline efficiency) and $T_{\text{deq}}$ are reported separately, format comparisons are not measurements of formats.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**: a format's throughput is inseparable from the engineering effort spent on its kernel. Marlin represents person-months of Ampere-specific tuning; a research codebook gets a weekend. A 2× gap between them identifies neither the format nor the kernel. There is no "format-neutral compiler" that would let you hold kernel quality fixed, so the confound cannot be removed by design, only bounded by reporting $\eta$ against the format's own roofline — which almost nobody does.

Second obstruction: **the LUT cliff is a hard resource boundary, not a tuning parameter.** A $K \le 16$ scalar codebook lives in two 32-bit registers and is read by byte-permute instructions. $K \ge 256$, or any $d \ge 2$ vector codebook, must live in shared memory, where addresses are data-dependent and bank conflicts ($\phi$) are input-dependent and unpredictable at compile time. Crossing that boundary changes the cost model discontinuously — exactly at the bit-widths (2–3) where non-uniform formats have their accuracy advantage.

Third: **tensor cores do not accept codebook indices.** Every non-uniform format must materialize fp16/bf16 before the MMA, so the dequant path is pure overhead that uniform block-scaled formats (MXFP4, NVFP4) increasingly avoid in hardware.

## 7. Current Research (as of 2026)

- **Trellis and lattice codes with cheap decoders** — Cornell (De Sa group, QuIP\#/QTIP lineage): the design pressure is explicitly toward codebooks whose decoder is a few ALU ops rather than a table.
- **Vector/additive quantization at scale** — IST Austria and Yandex Research (AQLM, and the GPTQ/Marlin line at IST Austria) — pushing 2-bit accuracy and, separately, kernel maturity.
- **Hardware-native block formats** — Microsoft and the OCP MX working group, plus NVIDIA's NVFP4 path; the bet is that hardware-supported near-uniform formats make software codebooks obsolete for serving. *(frontier — verify: the crossover batch size at which NVFP4 dominates 2-bit codebooks is not published.)*
- **LUT-first compilers** — Microsoft T-MAC on CPU; tensor-transformation compilers such as Ladder (OSDI 2024) that lower arbitrary low-precision types to supported ones.
- *(frontier — verify)* Codebook learning with an explicit instruction-count penalty in the objective — discussed, no strong published result.

## 8. Concrete Next Experiment

**Question.** At what batch size does the 2-bit non-uniform accuracy advantage stop paying for itself in throughput?

**Scale.** Llama-2-7B and Llama-2-70B, one A100-80GB and one H100-80GB, one serving stack (vLLM), all kernels compiled into it. Formats: int4-g128 (Marlin), NF4-g64, SqueezeLLM 3-bit, AQLM 2-bit, QTIP 2-bit. Sweep $b \in \{1,2,4,8,16,32,64,128\}$, 2048-token prompt, 256-token decode. Cost: $\approx 200$ GPU-hours plus integration.

**Control arm.** Two controls, both essential. (1) fp16 cuBLAS. (2) A *dequant-elided* variant of each kernel: identical memory traffic and tiling, with the codebook lookup replaced by a constant — this isolates $T_{\text{deq}}$ from bandwidth and from kernel-engineering quality, and is what makes the comparison identifiable.

**Deciding number.** Per format, per batch: $\eta = T_{\text{mem}} / T_{\text{measured}}$, with $T_{\text{mem}}$ computed from *measured* streaming bandwidth on that device. The single decisive figure is $\eta_{\text{QTIP-2bit}}(b)$ versus $\eta_{\text{Marlin-int4}}(b)$. If $\eta_{\text{2bit}} \ge 0.85$ at $b=32$, non-uniform codebooks are serving-viable and the accuracy win is real end to end. If $\eta_{\text{2bit}} \le 0.5$ at $b=32$ while the elided control shows $\ge 0.85$, the loss is provably dequantization, not bandwidth — and the research target shifts from better codebooks to cheaper decoders.

## 9. Key References

- **[Foundational]** Dettmers, Pagnoni, Holtzman, Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. — arXiv:2305.14314
- **[Foundational]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. — arXiv:2210.17323
- **[SOTA, accuracy]** Tseng, Chee, Sun, Kuleshov, De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. — arXiv:2402.04396
- **[SOTA, accuracy]** Tseng, Sun, Hou, De Sa. *QTIP: Quantization with Trellises and Incoherence Processing.* NeurIPS 2024. — arXiv:2406.11235
- **[SOTA, accuracy]** Egiazarian, Panferov, Kuznedelev, Frantar, Babenko, Alistarh. *Extreme Compression of Large Language Models via Additive Quantization.* ICML 2024. — arXiv:2401.06118
- **[SOTA, kernels]** Frantar, Castro, Chen, Hoefler, Alistarh. *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models.* PPoPP 2025. — arXiv:2408.11743
- **[SOTA, kernels]** Guo, Brandon, Cholakov, Ragan-Kelley, Xing, Kim. *Fast Matrix Multiplications for Lookup Table-Quantized LLMs (FLUTE).* Findings of EMNLP 2024. — arXiv:2407.10960
- **[Kernels]** Park, Park, Kwon, Kim, Lee, Lee. *LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models.* ICLR 2024. — arXiv:2206.09557
- **[Kernels]** Kim, Hooper, Gholami, Dong, Li, Shen, Mahoney, Keutzer. *SqueezeLLM: Dense-and-Sparse Quantization.* ICML 2024. — arXiv:2306.07629
- **[Kernels, CPU]** Wei et al. *T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge.* EuroSys 2025. — arXiv:2407.00088
- **[Formats]** Rouhani et al. *Microscaling Data Formats for Deep Learning.* Technical report / OCP MX specification, 2023. — arXiv:2310.10537
- **[Compiler]** Wang et al. *Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation.* OSDI 2024.
- **[Survey]** Lin, Tang, Tang, Yang, Dang, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024 (Best Paper). — arXiv:2306.00978

## 10. Worked Example

One Llama-2-7B projection, $W \in \mathbb{R}^{4096 \times 4096}$, $mn = 16.78$M weights, A100-80GB, batch 1 decode.

**Roofline.** At $\beta = 4.25$: bytes $= 16.78\text{M} \times 4.25/8 = 8.9$ MB. At measured $\text{BW}_{\text{eff}} = 1.6$ TB/s, $T_{\text{mem}} = 5.6\ \mu$s. At $\beta = 2.06$: 4.3 MB, $T_{\text{mem}} = 2.7\ \mu$s. The 2-bit format's *ideal* is $2.1\times$ faster.

**Case A — $K=16$ register LUT (NF4).** Sixteen fp16 entries pack into eight 32-bit registers; a nibble index resolves to a value in $\approx 2$ issued instructions via byte-permute plus select. $c_{\text{alu}} = 2$, $\phi$ irrelevant.
$$T_{\text{deq}} = \frac{16.78\times10^6 \times 2}{9.7\times10^{12}} = 3.5\ \mu\text{s} < 5.6\ \mu\text{s}.$$
Dequant hides under memory. $\eta \approx 0.9$ is achievable. Consistent with observed near-$4\times$ 4-bit kernels.

**Case B — $K=256$ shared-memory LUT (2-bit, $d=2$ vector code).** The table no longer fits registers. Each weight costs one 4-byte `LDS` at a data-dependent address. Assume a modest mean bank-conflict factor $\phi = 4$ (32 banks, indices drawn from a near-Gaussian codebook occupancy — empirically $\phi$ ranges 2–8 across layers):
$$T_{\text{deq}} = \frac{16.78\times10^6 \times 4\,\text{B} \times 4}{19.5\times10^{12}\,\text{B/s}} = 13.8\ \mu\text{s}.$$

**The obstruction, in one line.** $13.8\ \mu\text{s}$ against a $2.7\ \mu\text{s}$ roofline: the 2-bit format is $5.1\times$ *slower* than its own bandwidth limit and $2.5\times$ slower than 4-bit NF4 in wall clock, despite moving half the bytes. Halving the bits doubled the time. And note what the standard reporting would say: "2-bit achieves $1.9\times$ over fp16 ($26\ \mu$s)" — a true statement that conceals a $5\times$ efficiency deficit, because the baseline is fp16 rather than the format's own roofline. That is the confound in Section 6, made numeric: only the dequant-elided control arm of Section 8 separates "this format is expensive" from "this kernel is immature."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*