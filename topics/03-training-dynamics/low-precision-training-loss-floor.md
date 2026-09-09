---
id: 03-training-dynamics/low-precision-training-loss-floor
title: "Low-Precision Training Loss Floor"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Low-Precision Training Loss Floor

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/low-precision-training-loss-floor` · **Status:** empirically-open

## 1. Problem Statement

When a language model is pretrained with quantized matrix multiplications (FP8, FP6, FP4, INT4), the loss curve sits above the high-precision curve. Two mechanisms produce that gap and they have opposite long-run consequences:

- **Compute multiplier.** Low precision costs a constant factor of effective parameters or tokens. The curve is *shifted*; more data closes the gap. Cheap precision then always wins on loss-per-FLOP once the hardware speedup exceeds the multiplier.
- **Loss floor.** Low precision adds an irreducible term $E(P) > E(\infty)$. The curves *cross and never re-converge*; past some token count, extra data buys nothing and the low-precision run is permanently worse.

**The decision predicate.** For precision $P$ bits, is the irreducible term in the fitted scaling law precision-dependent?

$$L(N, D, P) = \underbrace{E(P)}_{\text{floor}} + \frac{A}{N_{\text{eff}}(P)^{\alpha}} + \frac{B}{D_{\text{eff}}(P)^{\beta}}$$

Solving the problem means deciding, with the floor and the multiplier separately identified, whether $E(P) - E(\infty) > 0$ at $P = 4, 6, 8$ — and if so, at what $(N, D)$ it becomes the dominant term.

Three variants, different difficulty:

- **Measurement.** Separate $E(P)$ from $N_{\text{eff}}(P)$ using finite runs. Hard: the two are near-degenerate over any single scan.
- **Method.** Build a quantized training recipe whose loss gap is *provably* multiplicative — stochastic rounding, per-block scaling, high-precision residual paths.
- **Theory.** Prove that quantized SGD on a non-convex objective with gradient noise $\sigma^2$ and rounding noise $\delta^2$ converges to a neighborhood of radius $\Theta(\delta)$ that does not shrink with $D$.

## 2. Formal Setting

Model with $N$ non-embedding parameters, trained on $D$ tokens. Loss $L$ is cross-entropy in nats/token on a held-out set drawn from the *training* distribution — not a downstream benchmark, and not perplexity on a shifted corpus.

**Precision as measured.** $P$ is not a single number. A run is specified by a triple $(P_w, P_a, P_g)$ — bits for weights, activations, gradients in the GEMM operands — plus:

- the **block size** $b$ over which a shared scale is computed (per-tensor $b=\infty$, per-channel, MXFP4 $b=32$, NVFP4 $b=16$);
- the **accumulate precision** $P_{\text{acc}}$ (near-universally FP32 or FP22, and frequently the dominant hidden variable);
- the **master weight** precision (FP32 optimizer state is standard even in "FP8 training");
- the set of **excluded layers** (embedding, unembedding, LayerNorm, softmax, often the first and last block).

Reporting only "FP8 training" underspecifies all five. A quantizer is $Q_b(x) = s \cdot \text{round}(x/s)$ with $s = \max_{i \in \text{block}} |x_i| / q_{\max}$. Rounding error per element:

$$\varepsilon = Q_b(x) - x, \qquad \mathbb{E}[\varepsilon^2] \approx \frac{s^2}{12} \ \ \text{(round-to-nearest, non-saturating)}$$

**Effective parameters.** Kumar et al. (ICLR 2025) model precision as parameter degradation, $N_{\text{eff}}(P) = N \cdot (1 - e^{-P/\gamma})$ with $\gamma$ fitted per operand type. Under this model $E$ is precision-*independent* — the floor hypothesis is exactly the claim that this functional form is wrong.

**Assumptions, and which are violated:**

| Assumption | Status |
|---|---|
| Rounding error is zero-mean, i.i.d. across elements | **Violated.** Round-to-nearest is deterministic and correlated with weight sign; errors correlate across a block sharing one scale. |
| Activation distributions are stationary in $D$ | **Violated.** Fishman et al. (2024) report SwiGLU-driven outlier amplification appearing only after ~200B tokens, invisible in short runs. |
| $E$ is estimable from finite $(N,D)$ scans | **Doubtful.** $E$, $A$, $\alpha$ trade off; Chinchilla-style fits give $E$ wide confidence intervals. |
| Scaling law form holds unchanged under quantization | **Untested** at $P \le 4$. |

## 3. State of the Art

**Established (reproduced, ablated).**
- FP16/BF16 mixed precision with FP32 master weights and loss scaling matches FP32 loss to within noise (Micikevicius et al., ICLR 2018). No floor at 16 bits.
- FP8 E4M3/E5M2 with per-tensor scaling matches BF16 to within ~0.1–0.3% relative loss on runs up to hundreds of billions of tokens (Micikevicius et al., 2022; DeepSeek-V3, 2024). Deployed at production scale.

**Claimed but unablated.**
- "FP4 pretraining matches BF16." NVFP4 and MXFP4 results (NVIDIA, 2025; Microsoft, 2025) hold FP8 or BF16 for a nontrivial fraction of layers and keep FP32 accumulation and master weights. The fraction of GEMM FLOPs actually in 4 bits is often unreported, and no published FP4 run is long enough to test crossing.
- "Compute-optimal precision is 7–8 bits" (Kumar et al., ICLR 2025). A fit, not a measurement of $E(P)$; assumes the no-floor functional form it is used to justify.
- BitNet b1.58 (Ma et al., 2024) reports parity with FP16 LLaMA at 3B/100B tokens — but trains *quantization-aware* with full-precision latent weights, so it does not bear on quantized-arithmetic training at all.

**Benchmark-number-only.** Nearly all FP4 claims are single downstream-eval tables (MMLU, HellaSwag) at one $(N,D)$ point. A benchmark table cannot distinguish a shift from a floor.

## 4. What Is Known

- **16-bit is free.** BF16 vs FP32, up to 175B params, no measurable loss gap. Universal industry practice.
- **FP8 gap at moderate scale.** DeepSeek-V3 reports FP8 vs BF16 relative loss error below 0.25% on a 16B-param validation run; the 671B production model trained FP8 end-to-end (arXiv:2412.19437). The gap is small and *did not visibly grow* over 14.8T tokens — the strongest existing evidence against a large FP8 floor, though it is a single uncontrolled run.
- **Late-training FP8 divergence exists.** Fishman et al. (2024) observed FP8 instability emerging after ~200B tokens in a 7B model, traced to SwiGLU outlier growth, fixed by Smooth-SwiGLU. Short-run FP8 stability does not imply long-run stability.
- **Stochastic rounding matters below 8 bits.** Wang et al. (NeurIPS 2018) showed 8-bit FP training needs stochastic rounding in accumulation; Chmiel et al. (ICLR 2024) achieved 4-bit-format matmuls at near-baseline accuracy with stochastic-rounding variants on ResNet/BERT scale.
- **4-bit integer training degrades.** Xi et al. (NeurIPS 2023) trained transformers with INT4 forward and backward: fine-tuning and small pretraining within ~1% accuracy, but explicit accuracy loss on harder tasks.
- **Inference bit-optimality is 4 bits** across 19M–176B params (Dettmers & Zettlemoyer, ICML 2023). This is *post-training* quantization and is often mis-cited as a training result.
- **Precision interacts with token/parameter ratio.** Kumar et al., 465 runs, $N \le 1.7$B, $D \le 26$B tokens: PTQ damage grows with $D/N$, so over-trained models are more fragile — consistent with a floor *or* with a shift; the scan does not separate them.

## 5. What Is Not Known

- **Empirically open.** Does any low-precision curve *cross* the BF16 curve — i.e. is there a $D^\*$ past which low-precision loss stops decreasing while BF16 continues? Nobody has run matched $P \in \{4,8,16\}$ arms to $D/N > 1000$ at fixed $N$. The experiment is runnable today for $\sim$10$^{21}$ FLOP; it has not been run because production runs never spend compute on a control arm.
- **Methodologically blocked.** $E(P)$ is not identifiable from short runs. With finite $(N,D)$ grids, $E$ and $\alpha$ are strongly anti-correlated in the fit; a nonzero $\hat{E}(P)$ can always be absorbed into a smaller $\hat{N}_{\text{eff}}$. No published protocol fixes the identifiability.
- **Theoretically open.** No convergence theorem for quantized-arithmetic SGD on non-convex objectives with *deterministic* round-to-nearest and block-shared scales. Stochastic-rounding results give $O(\delta)$ neighborhoods that shrink with step size but not with $D$; whether that neighborhood corresponds to a loss floor of size $\Theta(\delta^2)$ in the cross-entropy is unproven.
- **Unknown mechanism.** If a floor exists, is it from gradient underflow (signal below the quantization grid), forward-pass information destruction, or optimizer-state staleness? Each implies a different fix.

## 6. Why It Is Hard

**The specific obstruction is non-identifiability compounded by compute cost.** A floor $E(P) > 0$ and an effective-parameter shift $N_{\text{eff}} < N$ produce curves that are numerically indistinguishable over any $D$ range within about $10\times$. They diverge only in the asymptote — the region that costs the most to reach. Concretely: a floor of $0.01$ nats is below the seed-to-seed variance of a 1B-param run until $D/N \approx 10^3$, at which point the run costs $\sim 10^{22}$ FLOP *per arm*.

Two aggravating factors:

- **The measurement is confounded by the recipe.** Every FP8/FP4 result includes hand-tuned exclusions and scaling. If low precision underperforms, the paper improves the recipe and reruns; papers therefore report the best-case gap, not the distribution. The literature is systematically censored toward "no floor".
- **Downstream evals do not measure the named quantity.** MMLU parity at 5B tokens tells you nothing about $E(P)$; benchmark noise ($\pm 1$–2 points) exceeds a 0.01-nat loss difference by an order of magnitude in effect size.

## 7. Current Research (as of 2026)

- **NVIDIA / hardware vendors.** NVFP4 (block 16, FP8 scales, two-level scaling) pretraining at 10B+ scale on Blackwell; the open question they are pursuing is the fraction of layers that can leave BF16 *(frontier — verify)*.
- **Microsoft Research (Asia).** MXFP4 training with differentiable-quantization gradient estimators; BitNet line continues on quantization-aware, full-precision-latent training.
- **DeepSeek, Meta, Anthropic-scale labs.** FP8 as production default; fine-grained (per-128-block) scaling is now standard. Public evidence remains single-arm.
- **Habana/Intel and academic groups.** Long-horizon FP8 stability, outlier dynamics in SwiGLU/attention, Smooth-SwiGLU-style reparameterizations.
- **Scaling-law theory.** Extensions of Kumar et al. to separate forward and backward precision; work on whether $\gamma$ is architecture-invariant *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does the FP8 (and FP4) loss curve cross the BF16 curve, or stay parallel?

**Scale.** $N = 1.0$B non-embedding params, fixed. Train to $D = 500$B tokens ($D/N = 500$, $\approx 3\times10^{21}$ FLOP per arm), checkpointing loss every 1B tokens. Three arms plus control:

| Arm | GEMM operands | Accumulate | Master weights |
|---|---|---|---|
| Control | BF16 | FP32 | FP32 |
| A | FP8 E4M3, per-128-block | FP32 | FP32 |
| B | NVFP4, block 16 | FP32 | FP32 |
| C | FP8 **all** layers incl. embed/unembed | FP32 | FP32 |

Two seeds per arm to bound run-to-run variance (expect $\sigma \approx 0.003$ nats at this scale). Identical data order, identical hyperparameters — no per-arm LR retuning, which is the usual source of censoring.

**The deciding number.** Fit $L(D) = E + B D^{-\beta}$ separately per arm on $D \in [100\text{B}, 500\text{B}]$, and report

$$\Delta E = \hat{E}_{\text{arm}} - \hat{E}_{\text{BF16}}$$

with a bootstrap CI over checkpoints and seeds. **Decision:** $\Delta E > 0.01$ nats with the CI excluding zero $\Rightarrow$ a floor exists at that precision. $\Delta E$ CI containing zero while the *level* gap $L_{\text{arm}} - L_{\text{BF16}}$ stays constant at $\ge 0.01$ nats $\Rightarrow$ shift, not floor. A pre-registered fit window is essential; choosing the window after seeing the curves reintroduces the identifiability problem.

Total cost: $\approx 8 \times 3\times10^{21}$ FLOP, roughly 3,000 H100-days. Within reach of one academic consortium or a day of a frontier lab's cluster.

## 9. Key References

- **[Foundational]** Micikevicius, Narang, Alben, et al. *Mixed Precision Training.* ICLR 2018. — arXiv:1710.03740
- **[Foundational]** Wang, Choi, Brand, Chen, Gopalakrishnan. *Training Deep Neural Networks with 8-bit Floating Point Numbers.* NeurIPS 2018. — arXiv:1812.08011
- **[Foundational]** Micikevicius, Stosic, Burgess, et al. *FP8 Formats for Deep Learning.* 2022. — arXiv:2209.05433
- **[SOTA]** Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan. *Scaling Laws for Precision.* ICLR 2025. — arXiv:2411.04330
- **[SOTA]** Dettmers, Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML 2023. — arXiv:2212.09720
- **[SOTA]** Xi, Li, Chen, Zhu. *Training Transformers with 4-bit Integers.* NeurIPS 2023. — arXiv:2306.11987
- **[SOTA]** Chmiel, Banner, Hoffer, Ben-Yaacov, Soudry. *Accurate Neural Training with 4-bit Matrix Multiplications at Standard Formats.* ICLR 2024.
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Empirical]** Fishman, Chmiel, Banner, Soudry. *Scaling FP8 Training to Trillion-Token LLMs.* ICLR 2025.
- **[Empirical]** Ma, Wang, Ma, et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* 2024. — arXiv:2402.17764
- **[Context]** Wortsman, Dettmers, Zettlemoyer, Morcos, Farhadi, Schmidt. *Stable and Low-Precision Training for Large-Scale Vision-Language Models.* NeurIPS 2023. — arXiv:2304.13013
- **[Context]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556

## 10. Worked Example

Take a 1B-param model, BF16 baseline fit $L(D) = 1.69 + 410\,D^{-0.28}$ (Chinchilla-form constants, $D$ in tokens). At $D = 100$B: $L = 1.69 + 410 \cdot (10^{11})^{-0.28} = 1.69 + 0.0295 = 1.7195$ nats.

Now suppose FP8 costs a 10% effective-parameter penalty (a *shift*, $E$ unchanged), which at $\alpha=0.34$ raises the $N$-term by $\approx 3.5\%$ — call it $+0.008$ nats, flat in $D$. Alternatively suppose FP8 adds a *floor* $\Delta E = 0.008$ nats. Compare:

| $D$ | BF16 | FP8 shift | FP8 floor | Gap difference |
|---|---|---|---|---|
| 100B | 1.7195 | 1.7275 | 1.7275 | 0.0000 |
| 500B | 1.7016 | 1.7096 | 1.7096 | 0.0000 |
| 5T | 1.6832 | 1.6912 | 1.6912 | 0.0000 |

The two hypotheses are **numerically identical** in the constant-offset case — which is the point. They separate only when the floor is *not* a constant offset but an asymptote the curve approaches from above, i.e. when $L_{\text{FP8}}(D) = 1.698 + 410 D^{-0.28}$ vs $L_{\text{BF16}} = 1.69 + 410 D^{-0.28}$ *is itself* the floor model, and the alternative is $L_{\text{FP8}} = 1.69 + 424 D^{-0.28}$ (shift). Those differ by:

$$L_{\text{floor}} - L_{\text{shift}} = 0.008 - 14 D^{-0.28}$$

which is zero at $D^\* = (14/0.008)^{1/0.28} \approx 5.6 \times 10^{11}$ tokens, and reaches $+0.004$ nats only near $D \approx 10^{13}$.

**The obstruction, made visible:** the two hypotheses differ by less than the $\pm 0.003$-nat seed variance until $D \approx 2$T tokens at $N=1$B. Detecting the difference requires either $6\times$ more compute than the experiment in §8, or a variance-reduction trick (shared data order, paired seeds, difference-of-curves fitting) that has not been standardized. Every published FP8 comparison operates in the region where the curves are indistinguishable — which is why the loss floor is empirically open despite hundreds of low-precision training papers.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*