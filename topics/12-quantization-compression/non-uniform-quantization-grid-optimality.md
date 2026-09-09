---
id: 12-quantization-compression/non-uniform-quantization-grid-optimality
title: "Non-Uniform Quantization Grid Optimality"
topic: 12-quantization-compression
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Non-Uniform Quantization Grid Optimality

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/non-uniform-quantization-grid-optimality` · **Status:** partially-solved

## 1. Problem Statement

Given a trained network, weight quantization replaces each parameter with an index into a finite codebook. The **grid** is the set of reproduction levels. Uniform grids (INT4/INT8) space levels evenly; non-uniform grids (NF4, k-means codebooks, APoT, floating-point formats, lattice and trellis codebooks) do not.

**Input.** A weight tensor $W \in \mathbb{R}^{m\times n}$, a bit budget $b$, a block size $g$, a calibration set $\mathcal{D}$, and a hardware cost model.
**Output.** A codebook $\mathcal{C} = \{c_1,\dots,c_{2^b}\}$ and an assignment map.
**Decision predicate.** Does there exist a non-uniform $\mathcal{C}$ that beats the best uniform grid at *equal deployed cost* — equal bits per weight including scales, equal or lower decode latency?

Three variants, routinely conflated:

- **Theory.** Characterize the optimal grid for the actual objective (end-task loss, not weight MSE) under a fixed-rate constraint. Largely settled for MSE under high-resolution asymptotics; open for the loss objective.
- **Method.** Construct the grid. Mostly heuristic: k-means, quantiles, hand-designed exponent/mantissa splits.
- **Measurement.** Compare grids at matched cost. Blocked in practice, because published comparisons vary block size, outlier handling, rounding algorithm, and kernel simultaneously.

## 2. Formal Setting

Let $w$ be drawn from within-block density $f$. A scalar quantizer $Q$ with $N=2^b$ levels induces distortion
$$D(Q) = \mathbb{E}_{w\sim f}\big[(w - Q(w))^2\big].$$

**High-resolution model.** Write $Q$ as a compander with point density $\lambda(x)\ge 0$, $\int\lambda = 1$; the level nearest $x$ has spacing $\approx 1/(N\lambda(x))$. Bennett's integral gives
$$D \approx \frac{1}{12N^2}\int \frac{f(x)}{\lambda(x)^2}\,dx.$$
Minimizing over $\lambda$ (Panter–Dite, 1951) yields
$$\lambda^\star(x) \propto f(x)^{1/3}, \qquad D^\star \approx \frac{1}{12N^{2}}\Big(\int f(x)^{1/3}dx\Big)^{3}.$$
This is the *fixed-rate* optimum: every index costs exactly $b$ bits.

**Entropy-constrained variant.** If indices are entropy-coded to rate $R = H(Q(w))$, the optimal $\lambda$ is *uniform* (Gish–Pierce, 1968), and
$$D_{\mathrm{ECSQ}}(R) \le D_{\text{rate-distortion}}(R)\cdot 2^{2\times 0.2546} \quad\text{as } R\to\infty,$$
i.e. within $0.2546$ bit/sample of the Shannon bound.

**Measured quantities.**
- **Bits per weight:** $\bar b = b + (b_s + b_z)/g$ — payload plus scale and zero-point amortized over the block. Reported "4-bit" methods with $g=64$, FP16 scale and FP16 zero are $4.5$ bpw, not $4$.
- **Grid quality:** relative weight error $\|W-\hat W\|_F^2/\|W\|_F^2$, and proxy loss $\mathbb{E}_{x\sim\mathcal{D}}\|(W-\hat W)x\|^2$ — the GPTQ/AWQ objective, which is $\mathrm{tr}\big((W-\hat W)H(W-\hat W)^\top\big)$ with $H = \mathbb{E}[xx^\top]$.
- **End metric:** WikiText-2 perplexity at sequence length 2048, plus 0-shot accuracy on ARC/HellaSwag/PIQA/WinoGrande.
- **Cost:** tokens/s at batch 1 and decode arithmetic intensity, not just bpw.

**Assumptions and their violations.**
1. *Weights are i.i.d. Gaussian per block.* Violated: block statistics are heavy-tailed and the block is normalized by its own absmax, so the conditional density is a Gaussian truncated at its own maximum — not Gaussian.
2. *High resolution ($N\to\infty$).* Violated hard: $b\in\{2,3,4\}$ means $N \le 16$, where Bennett's integral is not accurate.
3. *MSE is the objective.* Violated: layer sensitivity varies by orders of magnitude, and Hessian-weighted error correlates with loss far better than plain MSE.
4. *Independent scalar coding.* Violated by design in QuIP#/AQLM/QTIP, which quantize vectors.

## 3. State of the Art

**Theory SOTA (established).** Panter–Dite/Bennett for fixed-rate scalar; Gish–Pierce for entropy-constrained; Zador's theorem for $d$-dimensional fixed-rate VQ, with space-filling gain bounded by $1.53$ dB ($0.2546$ bit) as $d\to\infty$. These are theorems, not claims. They say: **non-uniform scalar grids are worth little once entropy coding is available, and vector grids can recover at most 0.25 bit.**

**Empirical SOTA (established).** Vector/lattice/trellis codebooks dominate scalar grids at 2–3 bits. QuIP# (Tseng et al., ICML 2024) uses incoherence processing by random Hadamard rotation plus an $E_8$-lattice codebook; QTIP (Tseng et al., NeurIPS 2024) replaces the codebook with trellis-coded quantization and reports further gains at 2 bits. AQLM (Egiazarian et al., ICML 2024) uses learned additive codebooks. All three beat GPTQ (Frantar et al., ICLR 2023) and AWQ (Lin et al., MLSys 2024) at 2-bit on Llama-2 by large perplexity margins.

**Claimed but unablated.** NF4 (Dettmers et al., QLoRA, NeurIPS 2023) is described as "information-theoretically optimal for normally distributed weights." Yoshida (2023) shows the derivation is not an optimality proof — NF4's levels are not the Lloyd–Max levels and not the quantile levels of the relevant conditional distribution — and that NF4 nonetheless works well empirically. The optimality claim is false as stated; the empirical result stands.

**Benchmark-number-only results.** Most head-to-head "non-uniform beats uniform" comparisons (NF4 vs FP4 vs INT4) exist as single perplexity columns at one block size, without matched-$\bar b$ accounting or seed variance. Treat them as benchmark numbers, not ablations.

## 4. What Is Known

- **Gaussian, $b=4$, $N=16$, fixed rate.** Lloyd–Max non-uniform: $D/\sigma^2 = 0.009497$ (20.22 dB). Best uniform: $D/\sigma^2 = 0.011539$ (19.38 dB), optimal step $0.3352\sigma$ (Max, IRE Trans. IT, 1960). Gap: **0.84 dB, i.e. 0.14 bit.**
- **Entropy-coded uniform at the same rate** reaches $\approx 0.00555\sigma^2$ — a **0.39 bit** gain, nearly 3× the non-uniform-grid gain.
- **Vector quantization ceiling:** $0.2546$ bit/dimension of space-filling gain, plus whatever the source's memory/shape gives. QuIP#/QTIP operationalize the first term.
- **k-bit scaling laws.** Dettmers & Zettlemoyer (ICML 2023), ~35,000 zero-shot runs over models from 19M to 176B parameters: 4-bit is Pareto-optimal for bit-level scaling; quantile/data-aware types beat plain integer at 4 bits, but **the advantage shrinks as block size falls**, i.e. non-uniformity substitutes for finer scaling granularity.
- **Sensitivity beats grid shape.** SqueezeLLM (Kim et al., ICML 2024) shows Hessian-weighted k-means plus a small dense outlier set (0.05% of weights) gives larger 3-bit gains than any uniform-vs-non-uniform grid choice at the same bpw.
- **Rotation beats grid shape.** Random Hadamard/orthogonal rotation before quantization (QuIP#, QuaRot, SpinQuant) removes outliers and shrinks the uniform/non-uniform gap, because it makes the block distribution closer to the isotropic Gaussian a uniform grid handles well.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the loss-optimal grid at low resolution ($N \le 16$) under a Hessian-weighted objective with a *shared* codebook across many blocks. Panter–Dite is asymptotic and per-source; the deployed problem is finite-$N$, tied-codebook, non-MSE. No lower bound exists for it.
- **Empirically open.** The matched-cost sweep has not been run: uniform vs NF4 vs Lloyd–Max vs learned-codebook, over $b\in\{2,3,4\}$ × $g\in\{16,32,64,128\}$ × {rotation on/off} × {GPTQ rounding on/off}, at $\ge$ 7B and $\ge$ 70B, with equal $\bar b$ and reported seed variance. Everything needed is available; nobody has published the full factorial.
- **Methodologically blocked.** "Equal cost" is not well defined. A non-uniform codebook costs a LUT lookup per weight in the decode kernel; a uniform grid costs a multiply-add. Whether that is free depends on whether the kernel is memory-bound, which depends on batch size, sequence length, and GPU. There is no accepted normalization, so cost-matched claims are not comparable across papers.

## 6. Why It Is Hard

**The obstruction is confounded measurement, not compute.** The reported gain of any grid is entangled with four other knobs — block size, rotation, rounding algorithm (RTN vs GPTQ vs sign-flip search), and outlier carve-out — each of which moves perplexity by more than the grid does. Section 4's arithmetic is the reason: the entire fixed-rate uniform-vs-optimal-scalar gap is **0.14 bit**, while dropping block size from 128 to 32 changes $\bar b$ by 0.375 bit and changes error by more. A grid comparison that does not hold $\bar b$ fixed to three decimal places is measuring block size.

Second obstruction: **absent ground truth for the source**. There is no agreed model of the within-block weight density after absmax normalization, so "optimal for the true distribution" has no referent. NF4's optimality claim failed exactly here.

## 7. Current Research (as of 2026)

- **Trellis and lattice codebooks.** Cornell (De Sa, Kuleshov, Tseng) — QTIP line; the active question is decode throughput of trellis decoding at batch 1. *(frontier — verify current numbers.)*
- **Rotation-first pipelines.** QuaRot (Ashkboos et al.) and SpinQuant (Meta) treat the rotation as the learned object and leave the grid uniform. If this wins, the non-uniform grid question becomes moot for weights.
- **Microscaling (MX) formats.** OCP MX spec (Rouhani et al., 2023) fixes a shared power-of-two scale per 32-element block with FP4/FP6/FP8 elements — a hardware-blessed *mildly* non-uniform grid, now in Blackwell-class silicon. This makes the practical comparison "MXFP4 vs INT4 at matched hardware", not "arbitrary codebook vs uniform".
- **Loss-aware codebook learning.** Learning $\mathcal{C}$ by straight-through gradient on a calibration objective rather than k-means on weights. Reported gains are small and not independently reproduced. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** At matched $\bar b$ and matched rounding, how much perplexity does grid shape buy, and does rotation eliminate it?

**Scale.** Llama-3.1-8B and Llama-3.1-70B, WikiText-2 perplexity at 2048 tokens, plus 5-task 0-shot average. Calibration: 128 sequences of 2048 tokens from C4.

**Design.** $2\times2\times3$: grid $\in$ {uniform INT, NF4-style quantile, per-tensor Lloyd–Max fit to the empirical post-normalization distribution} × rotation $\in$ {none, random Hadamard} × $b \in \{2,3,4\}$. Hold $\bar b$ constant per cell by adjusting $g$ and scale precision (e.g. INT4/$g$=64/FP16 scale = 4.25 bpw; match the non-uniform arms to 4.25 ± 0.01). All arms use identical GPTQ rounding and identical outlier policy (none). Three seeds for the calibration draw.

**Control arm.** Uniform INT, no rotation, GPTQ, matched $\bar b$ — the standard baseline.

**Deciding number.** $\Delta\mathrm{PPL}_{\text{grid}}$ = perplexity(best non-uniform) − perplexity(uniform) **within the rotation-on cells at 4 bits**, compared against the seed standard deviation. If $|\Delta\mathrm{PPL}_{\text{grid}}| < 2\sigma_{\text{seed}}$ (expected $\sigma_{\text{seed}}\approx 0.02$ PPL at 8B), scalar grid shape is dead for rotated 4-bit weights and effort should move to vector codebooks and rotations. If $\Delta > 0.1$ PPL, non-uniform scalar grids retain independent value and the theory gap in §5 becomes worth attacking.

## 9. Key References

- **[Foundational]** J. Max. *Quantizing for Minimum Distortion.* IRE Transactions on Information Theory, 1960.
- **[Foundational]** S. P. Lloyd. *Least Squares Quantization in PCM.* IEEE Transactions on Information Theory, 1982 (written 1957).
- **[Foundational]** W. R. Bennett. *Spectra of Quantized Signals.* Bell System Technical Journal, 1948.
- **[Foundational]** H. Gish, J. N. Pierce. *Asymptotically Efficient Quantizing.* IEEE Transactions on Information Theory, 1968.
- **[Foundational]** P. Zador. *Asymptotic Quantization Error of Continuous Signals and the Quantization Dimension.* IEEE Transactions on Information Theory, 1982.
- **[Survey]** A. Gersho, R. M. Gray. *Vector Quantization and Signal Compression.* Kluwer, 1992.
- **[Foundational]** S. Han, H. Mao, W. J. Dally. *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR, 2016. — arXiv:1510.00149
- **[SOTA]** Y. Li, X. Dong, W. Wang. *Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks.* ICLR, 2020. — arXiv:1909.13144
- **[SOTA]** T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- **[SOTA]** T. Dettmers, L. Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** D. Yoshida. *NF4 Isn't Information Theoretically Optimal (and that's Good).* 2023. — arXiv:2306.06965
- **[SOTA]** E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** S. Kim, C. Hooper, A. Gholami, et al. *SqueezeLLM: Dense-and-Sparse Quantization.* ICML, 2024. — arXiv:2306.07629
- **[SOTA]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024. — arXiv:2402.04396
- **[SOTA]** A. Tseng, Q. Sun, D. Hou, C. De Sa. *QTIP: Quantization with Trellises and Incoherence Processing.* NeurIPS, 2024. — arXiv:2406.11235
- **[SOTA]** V. Egiazarian, A. Panferov, D. Kuznedelev, et al. *Extreme Compression of Large Language Models via Additive Quantization.* ICML, 2024. — arXiv:2401.06118
- **[SOTA]** J. Lin, J. Tang, H. Tang, et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** S. Ashkboos, A. Mohtashami, M. Croci, et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456

## 10. Worked Example

Take one block of $g=64$ weights from a Llama MLP down-projection, absmax-normalized. Model the source as $\mathcal{N}(0,\sigma^2)$ and budget $b=4$, $N=16$.

| Scheme | $D/\sigma^2$ | SNR | Gain over uniform |
|---|---|---|---|
| Uniform, optimal step $0.3352\sigma$ | $0.011539$ | 19.38 dB | — |
| Lloyd–Max (optimal non-uniform, fixed rate) | $0.009497$ | 20.22 dB | 0.84 dB = **0.14 bit** |
| Uniform + ideal entropy coding at $R=4$ | $\approx 0.00555$ | 22.56 dB | 3.18 dB = **0.39 bit** |
| Shannon bound $\sigma^2 2^{-8}$ | $0.003906$ | 24.08 dB | 4.70 dB = 0.59 bit |

Now price the block. INT4 payload is $64\times 4 = 256$ bits; an FP16 absmax scale adds 16, giving $\bar b = 4.25$. A Lloyd–Max codebook shared across the tensor adds $\approx 0$ amortized. So the non-uniform grid is genuinely free in bits — and buys 0.14 bit of equivalent rate.

Compare to a knob nobody controls for: moving $g$ from 64 to 32 changes $\bar b$ from 4.25 to 4.50, a **0.25 bit** change — nearly double the entire theoretical grid gain. Any paper comparing NF4 at $g=64$ against INT4 at $g=128$ ($\bar b = 4.125$) has handed the non-uniform arm 0.125 bit of extra budget, which alone can exceed the effect being measured.

**The obstruction, visible.** The largest available scalar gain (0.39 bit) comes from entropy coding a *uniform* grid, and is unusable because variable-length codes destroy the aligned, random-access reads that a decode kernel needs. The field therefore optimizes the smaller term (0.14 bit) inside a measurement setup whose uncontrolled confounds are larger (0.25 bit). That is why fifteen years of non-uniform grid proposals produce inconsistent orderings: the signal is below the noise floor of the experimental design, not below the noise floor of the models.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*