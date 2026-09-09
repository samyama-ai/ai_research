---
id: 12-quantization-compression/lattice-versus-learned-quantizers
title: "Lattice versus Learned Quantizers at Extreme Compression"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Lattice versus Learned Quantizers at Extreme Compression

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/lattice-versus-learned-quantizers` · **Status:** empirically-open

## 1. Problem Statement

At 2 bits per weight and below, two families of vector quantizers compete for the same job: **fixed algebraic lattices** (E8, Leech, trellis/bitshift codes) whose codebooks are chosen once from packing theory and never see data, and **learned codebooks** (residual/additive quantization, VQ-VAE-style, AQLM) fitted to the empirical weight or activation distribution. Both are paired with the same preprocessing — incoherence processing by random rotation — and both report similar end-task numbers. Nobody has established which family is actually better, or under what conditions.

Three variants, with different difficulty:

- **Measurement.** Given a fixed model, bit budget $b$, and calibration set, which family attains lower end-task loss at equal *deployed* bits (codebook storage, scales and rotations included) and equal decode throughput? Runnable today; not run cleanly.
- **Method.** Is there a hybrid — lattice for the shape, learned for the shell/scale — that strictly dominates both? Open constructively.
- **Theory.** Post-rotation weight blocks are near-Gaussian but not exactly Gaussian, and not i.i.d. across a block. For that source class, does a data-fitted codebook have an asymptotic rate–distortion advantage over the best lattice, or is the lattice within the $0.2546$-bit universal gap and hence essentially optimal? Open.

Solving it means: a stated regime boundary in $(b, d, \text{model scale})$ where one family wins, with the crossing point measured, not asserted.

## 2. Formal Setting

Let $W \in \mathbb{R}^{m \times n}$ be a linear layer's weights and $H = \mathbb{E}[xx^\top]$ the calibration second-moment matrix over inputs $x$. Incoherence processing applies random orthogonal $U, V$ (Hadamard-based, so $O(n\log n)$):

$$\tilde{W} = U W V^\top, \qquad \mu(\tilde W) = \max_{ij} |\tilde W_{ij}| \, \sqrt{mn} \, / \, \|\tilde W\|_F .$$

$\mu$ is the **incoherence**, measured directly from $\tilde W$; the point of $U,V$ is to drive $\mu$ toward $O(\sqrt{\log mn})$ and make the empirical distribution of entries close to $\mathcal{N}(0, \|\tilde W\|_F^2/mn)$.

Split each row into blocks $z \in \mathbb{R}^d$. A quantizer is a map $Q: \mathbb{R}^d \to \mathcal{C}$, $|\mathcal{C}| = 2^{bd}$.

- **Lattice arm:** $\mathcal{C} = (\Lambda \cap S) \cdot s$ for a fixed lattice $\Lambda$ and shaping region $S$; $s$ is a per-group scale, stored.
- **Learned arm:** $\mathcal{C} = \{\sum_{k=1}^{K} C_k[i_k]\}$ with codebooks $C_k \in \mathbb{R}^{2^{c} \times d}$ fitted by alternating least squares on the calibration data.

**Rate, as actually measured.** Not $b$, but deployed bits per weight:

$$b_{\text{eff}} = \frac{bmn \;+\; \underbrace{|\mathcal{C}|_{\text{stored}} \cdot d \cdot 16}_{\text{codebook}} \;+\; \underbrace{\text{scales} + \text{rotation seeds}}_{\text{overhead}}}{mn}.$$

For a lattice arm the codebook term is $\approx 0$ (generated on the fly); for a learned arm with $K=2$, $c=12$, $d=8$ it is $2\cdot 4096\cdot 8\cdot 16 = 1.05$ Mbit per shared codebook, which is negligible if shared model-wide and material if per-layer.

**Distortion.** The proxy objective is the layerwise proxy loss

$$\mathcal{L}_{\text{proxy}} = \mathbb{E}_x \big\| (W - \hat W)x \big\|_2^2 = \operatorname{tr}\!\big[(W-\hat W) H (W-\hat W)^\top\big],$$

and the target is end-task loss $\mathcal{L}_{\text{task}}$ (WikiText2/C4 perplexity, or zero-shot accuracy).

**High-rate theory.** For a lattice $\Lambda$ with normalized second moment $G(\Lambda)$ and cell volume $V$, per-dimension MSE is $D = G(\Lambda) V^{2/d}$, and the excess rate over the Shannon rate–distortion function of a Gaussian source is

$$\Delta R = \tfrac{1}{2}\log_2\!\big(2\pi e\, G(\Lambda)\big) \ \text{bits/dim}.$$

Scalar: $G = 1/12$, $\Delta R = 0.2546$ bits ($1.53$ dB). $G(E_8) = 0.0717 \Rightarrow \Delta R = 0.1466$; $G(\Lambda_{24}) = 0.0658 \Rightarrow \Delta R = 0.0847$.

**Assumptions, and which are violated.**
1. *High rate* ($b \gg 1$). Violated: the regime of interest is $b \in [1,3]$, where $\Delta R$ formulas are not tight and Zador's asymptotics do not apply.
2. *Gaussian, i.i.d. source.* Approximately true after Hadamard rotation for the marginal; violated for the joint — outlier channels and heavy-tailed rows survive rotation partially, and $H$ is far from isotropic.
3. *Proxy loss tracks task loss.* Violated at low $b$: layerwise proxy improvements stop predicting perplexity once error compounds across depth.
4. *Bits are the cost.* Violated on real hardware: decode latency and cache pressure, not bits, set throughput.

## 3. State of the Art

**Systems/empirical SOTA (LLM weight-only, $b \approx 2$).**

| Method | Family | Codebook | Venue |
|---|---|---|---|
| QuIP (Chee et al.) | scalar + incoherence | none | NeurIPS 2023 |
| QuIP# (Tseng et al.) | lattice | $E_8$ (E8P), fixed | ICML 2024 |
| QTIP (Tseng et al.) | trellis-coded, fixed | bitshift trellis, compute-generated | NeurIPS 2024 |
| AQLM (Egiazarian et al.) | learned | additive, $K$ codebooks fitted | ICML 2024 |
| PV-Tuning (Malinovskii et al.) | orthogonal | — (fine-tunes any of the above) | NeurIPS 2024 |

**Established:** incoherence processing (random Hadamard rotation) is the single largest contributor at 2 bits and is required by *both* families; QuIP#, QTIP, AQLM all use it, and ablating it costs more than switching quantizer family. Llama-2-70B FP16 WikiText2 perplexity is $3.12$; all three 2-bit methods land in roughly the $3.5$–$4.0$ band, i.e. within a few tenths of each other and far above the pre-incoherence baselines (GPTQ at 2 bits diverges).

**Claimed but unablated:** that the lattice/trellis structure *itself* is responsible for QTIP's edge over QuIP#. QTIP changes the codebook dimension, the code structure, and the decode kernel at once. Likewise, AQLM's advantage over QuIP# is reported with a different calibration budget and a different fine-tuning recipe; the two arms differ in more than the codebook.

**Benchmark-number-only results:** every published head-to-head between families is a table of perplexities from separate papers, each with its own calibration set size, sequence length, block layout, and whether end-to-end fine-tuning was applied. No paper holds $b_{\text{eff}}$, $d$, $H$, and fine-tuning fixed and varies only $\mathcal{C}$.

**Theory SOTA.** Zamir–Feder (1996): dithered lattice quantization with entropy coding is universally within $0.2546$ bits/dim of $R(D)$ for *any* source, and the gap vanishes as $d\to\infty$ for good lattices. Agrell & Allen (IEEE Trans. IT, 2023) give the best known lattice quantizers up to dimension ~16 and confirm no lattice beats the $0.0847$-bit $\Lambda_{24}$ figure at $d=24$.

## 4. What Is Known

- **The maximum prize is small.** The total space-filling gain available to *any* quantizer over scalar quantization is $0.2546$ bits/dim ($1.53$ dB), Zador/Gersho, high-rate. $E_8$ captures $58\%$ of it ($0.108$ of $0.2546$ bits); the Leech lattice captures $67\%$. So a perfect learned codebook can beat $E_8$ by at most $\approx 0.11$ bits/dim in the high-rate limit — under $6\%$ of a 2-bit budget.
- **Gersho's conjecture** (optimal high-rate quantizer cells are congruent to a single polytope) is proved for $d=1,2$ and open for $d\ge 3$. So even "is the best quantizer a lattice?" is theoretically unresolved above $d=2$.
- **Sphere packing optimality** in $d=8$ (Viazovska, Annals 2017) and $d=24$ (Cohn–Kumar–Miller–Radchenko–Viazovska, Annals 2017) is proved; **quantization** optimality (minimum $G$) in those dimensions is not.
- **Incoherence processing works and is measurable.** QuIP reports proxy-loss guarantees under incoherence; QuIP#/QuaRot show Hadamard rotation is cheap ($O(n\log n)$) and removes the outlier-channel problem that 2-bit scalar quantization cannot survive.
- **Fine-tuning dominates codebook choice at 2 bits.** PV-Tuning (Llama-2 family, 7B–70B) recovers more perplexity than the reported gap between any two 2-bit quantizer families, which means uncontrolled fine-tuning invalidates cross-paper comparisons.
- **Learned codebooks win when the source is structured.** In ANN search, OPQ (CVPR 2013) and LSQ++ (ECCV 2018) beat plain product quantization on SIFT1M/Deep1B by learning rotations and additive codebooks — but that is *pre-rotation* data with strong anisotropy, the exact structure incoherence processing deliberately destroys.

## 5. What Is Not Known

- **Theoretically open.** Whether, for the post-Hadamard weight source (near-Gaussian marginal, correlated joint, anisotropic $H$), any codebook can beat a good lattice by more than $o(1)$ bits at $b\in[1,3]$. No low-rate analogue of Zador's theorem covers this. Gersho's conjecture for $d\ge3$ remains open, so the lattice arm has no optimality certificate either.
- **Empirically open (the core gap).** The controlled experiment — same model, same $H$, same $b_{\text{eff}}$, same $d$, same fine-tuning, only $\mathcal{C}$ swapped — is entirely runnable on 8×A100 in days and has not been published. Everything current is cross-paper.
- **Methodologically blocked.** $b_{\text{eff}}$ is not reported consistently: papers variously exclude codebook storage, per-group FP16 scales, or the Hadamard seed. Until a single accounting convention exists, "2-bit" names two different rates that differ by up to $0.15$ bits/weight — comparable to the entire effect being measured.

## 6. Why It Is Hard

**The obstruction is confounded measurement against a small effect.** The theoretical ceiling on the whole question is $0.11$ bits/dim, and at 2 bits the observed perplexity differences between families are $\sim 0.1$–$0.4$ on WikiText2. That is the same magnitude as the noise from: calibration-set choice (RedPajama vs C4, 256 vs 4096 sequences), sequence length (2048 vs 8192), whether end-to-end fine-tuning ran, and the unreported bits in $b_{\text{eff}}$. Every published comparison varies at least three of these simultaneously. A second obstruction is **non-identifiability of the credit**: a learned codebook and a lattice codebook of the same size induce different decode kernels, so a throughput-matched comparison forces different $d$, which changes the space-filling gain — the thing under test — as a side effect.

## 7. Current Research (as of 2026)

- **Cornell (De Sa group)** — trellis and bitshift codes, the QuIP#→QTIP line; direction is toward compute-generated codebooks with zero storage, which sidesteps the $b_{\text{eff}}$ accounting problem entirely.
- **IST Austria / Yandex-lineage groups (Alistarh, Egiazarian, Malinovskii)** — additive/learned quantization plus PV-tuning; direction is that the codebook matters less than the post-quantization optimization.
- **ETH/Google image-compression line (Ballé, Agustsson, Theis)** — universal/dithered quantization inside learned transforms; LVQAC (Zhang & Wu, CVPR 2023) puts a lattice VQ inside a learned codec with adaptive companding, which is the clean hybrid statement of the problem.
- **Coding-theory side (Agrell, Allen)** — better lattice quantizers in moderate dimensions; slowly closing the $G(\Lambda)$ tables.
- *(frontier — verify)* KV-cache and activation quantization is where lattices should help most, because activations are dynamic and a fitted codebook cannot be re-fitted per token; early reports suggest the lattice arm's advantage is larger there than for weights.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B and Llama-3.1-70B (two scales, to test whether the gap shrinks with model size).

**Design.** Fix everything but $\mathcal{C}$:
- same Hadamard rotations and seeds; same $H$ from 4096 sequences of RedPajama at length 8192;
- same block dimension $d=8$; same $2^{16}$-entry effective codebook size;
- **arm A (lattice):** $E_8$-based codebook, no stored entries;
- **arm B (learned):** additive codebook of identical size, fitted by ALS on the same $H$, its storage *charged* to $b_{\text{eff}}$;
- **control arm:** scalar round-to-nearest under the same rotation and same $b_{\text{eff}}$ — this calibrates how much of any gap is vector quantization at all;
- both arms then get the *identical* PV-tuning recipe, and are also evaluated without it.

**Rate matching.** Tune per-group scale granularity until $b_{\text{eff}}$ matches to within $0.005$ bits/weight across all arms at three targets: $b_{\text{eff}} \in \{1.5, 2.0, 2.5\}$.

**The deciding number.** $\Delta = \text{ppl}_B - \text{ppl}_A$ on WikiText2 at $b_{\text{eff}} = 2.0$, with a bootstrap CI over 5 calibration seeds. **If $|\Delta| < 0.05$ with the CI excluding $0.15$, the families are equivalent for weights and codebook research should stop; the budget belongs to fine-tuning and rate accounting.** If $\Delta < -0.15$ at 8B but $\to 0$ at 70B, the learned advantage is a small-model artifact.

## 9. Key References

- **[Foundational]** A. Gersho. *Asymptotically optimal block quantization.* IEEE Transactions on Information Theory, 1979.
- **[Foundational]** P. Zador. *Asymptotic quantization error of continuous signals and the quantization dimension.* IEEE Transactions on Information Theory, 1982.
- **[Foundational]** R. Zamir, M. Feder. *On lattice quantization noise.* IEEE Transactions on Information Theory, 1996.
- **[Foundational]** J. H. Conway, N. J. A. Sloane. *Sphere Packings, Lattices and Groups.* Springer, 3rd ed., 1999.
- **[Foundational]** R. Zamir. *Lattice Coding for Signals and Networks.* Cambridge University Press, 2014.
- **[Theory]** M. Viazovska. *The sphere packing problem in dimension 8.* Annals of Mathematics, 2017.
- **[Theory]** H. Cohn, A. Kumar, S. D. Miller, D. Radchenko, M. Viazovska. *The sphere packing problem in dimension 24.* Annals of Mathematics, 2017.
- **[Theory]** E. Agrell, B. Allen. *On the best lattice quantizers.* IEEE Transactions on Information Theory, 2023.
- **[SOTA]** J. Chee, Y. Cai, V. Kuleshov, C. De Sa. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees.* NeurIPS, 2023.
- **[SOTA]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024.
- **[SOTA]** A. Tseng, Q. Sun, D. Hou, C. De Sa. *QTIP: Quantization with Trellises and Incoherence Processing.* NeurIPS, 2024.
- **[SOTA]** V. Egiazarian, A. Panferov, D. Kuznedelev, E. Frantar, A. Babenko, D. Alistarh. *Extreme Compression of Large Language Models via Additive Quantization.* ICML, 2024.
- **[SOTA]** V. Malinovskii, D. Mazur, I. Ilin, D. Kuznedelev, K. Buredenkov, K. Yi, D. Alistarh, P. Richtarik. *PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression.* NeurIPS, 2024.
- **[Related]** E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023.
- **[Related]** S. Ashkboos, A. Mohtashami, M. Croci, B. Li, M. Jaggi, D. Alistarh, T. Hoefler, J. Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024.
- **[Related]** H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI, 2011.
- **[Related]** T. Ge, K. He, Q. Ke, J. Sun. *Optimized Product Quantization.* CVPR, 2013.
- **[Related]** J. Ballé, V. Laparra, E. P. Simoncelli. *End-to-end Optimized Image Compression.* ICLR, 2017.
- **[Related]** E. Agustsson, L. Theis. *Universally Quantized Neural Compression.* NeurIPS, 2020.
- **[Related]** X. Zhang, X. Wu. *LVQAC: Lattice Vector Quantization Coupled with Spatially Adaptive Companding for Efficient Learned Image Compression.* CVPR, 2023.
- **[Survey]** J. Ballé, P. A. Chou, D. Minnen, S. Singh, N. Johnston, E. Agustsson, S. J. Hwang, G. Toderici. *Nonlinear Transform Coding.* IEEE Journal of Selected Topics in Signal Processing, 2021.

## 10. Worked Example

Take one Llama-2-7B layer: `model.layers.10.mlp.down_proj`, $W \in \mathbb{R}^{4096 \times 11008}$, $45.1$M weights. Target $b=2$, block dimension $d=8$, so $2^{16}$ codewords per block.

**Step 1 — the theoretical prize.** After Hadamard rotation, treat $\tilde W$ entries as $\mathcal{N}(0,\sigma^2)$. The $E_8$ arm sits $0.1466$ bits/dim above $R(D)$; a hypothetical perfect $d=8$ codebook cannot do better than the $d\to\infty$ limit of $0$, but at $d=8$ no quantizer is known to beat $G(E_8)=0.0717$. The learned arm's *entire* upside over $E_8$ at $d=8$ is therefore bounded by whatever the true optimal $d=8$ quantizer's $G$ is — and since Gersho's conjecture is open at $d=8$, that bound is not even computable. Assume generously it is $0.02$ bits/dim: **$1\%$ of the 2-bit budget.**

**Step 2 — the cost side.** The learned arm needs its codebook stored. Additive, $K=2$, $2^{12}$ entries each, $d=8$, FP16:

$$2 \times 4096 \times 8 \times 16 = 1{,}048{,}576 \text{ bits}.$$

If shared across all 224 linear layers of the 7B model ($6.5$B weights), that is $0.00016$ bits/weight — free. If fitted **per layer**, over this layer's $45.1$M weights it is $0.023$ bits/weight.

**Step 3 — the collision.** The per-layer learned arm buys at most $0.02$ bits/dim of coding gain and spends $0.023$ bits/weight on storing the codebook. **The ledger is negative before a single perplexity is measured.** The shared-codebook variant is free but is fitting one codebook to 224 layers with different $H$, which is exactly the regime where a fixed lattice already suffices.

**Step 4 — what actually shows up in the table.** Published 2-bit Llama-2-70B numbers from the lattice and learned families differ by roughly $0.1$–$0.4$ WikiText2 perplexity against an FP16 anchor of $3.12$. But between those papers the calibration corpora differ, the sequence lengths differ (2048 vs 8192), and one arm was fine-tuned end-to-end while the other was not. Re-running a single arm with a different calibration seed moves perplexity by a comparable amount.

**The obstruction, made visible:** the effect being measured ($\le 0.02$ bits/dim of coding gain) is smaller than the codebook-storage term that is routinely omitted from the rate ($0.023$ bits/weight), which is in turn smaller than the calibration-seed noise in the metric. Three quantities of the same size, only one of them reported. That is why the question is empirically open rather than merely unsettled.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*