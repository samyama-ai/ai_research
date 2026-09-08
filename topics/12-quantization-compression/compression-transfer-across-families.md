---
id: 12-quantization-compression/compression-transfer-across-families
title: "Compression Transfer Across Model Families"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compression Transfer Across Model Families

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/compression-transfer-across-families` · **Status:** empirically-open

## 1. Problem Statement

A compression recipe — method, bit-width, group size, calibration set, per-layer bit allocation, rotation/scaling hyperparameters — is almost always tuned on one model family (usually Llama) and then shipped as a default for every other. The problem is whether that transfer is valid, and if not, what predicts the failure.

Three variants, with different difficulty:

- **Measurement.** Given a recipe $R$ tuned on donor family $\mathcal{A}$ and applied to recipient $\mathcal{B}$, define and measure the *transfer gap* — the excess degradation on $\mathcal{B}$ relative to the recipe re-tuned on $\mathcal{B}$. Open because perplexity, the usual metric, does not track the downstream failures that matter.
- **Method.** Build a recipe selector that, given cheap statistics of $\mathcal{B}$ (activation kurtosis, Hessian spectra, weight outlier mass), picks bit allocation and hyperparameters within $\epsilon$ of the oracle re-tuned recipe, at a cost $\ll$ one full re-tune sweep.
- **Theory.** Prove or refute that compressibility at a given bit-width is a function of architecture-independent quantities (parameter count, tokens-per-parameter, effective rank) rather than of family-specific idiosyncrasies (normalization placement, activation function, attention variant, MoE routing).

**Solved** would mean: a published predictor $\hat{\Delta}$ that estimates the transfer gap for an unseen family to within a stated tolerance, validated across $\geq 5$ architecturally distinct families, with the sign of the error known.

## 2. Formal Setting

Let $\theta \in \mathbb{R}^d$ be a dense model, $\mathcal{D}_{\text{cal}}$ a calibration set of $n$ sequences, and $R$ a recipe mapping $(\theta, \mathcal{D}_{\text{cal}}) \mapsto \tilde{\theta}$ at effective bit-width $b$ (bits per weight, *including* scales and zero-points amortized over group size $g$: $b = b_w + (b_s + b_z)/g$ — the number that must be reported, since group size 32 at 4 bits is really $\approx 4.5$).

Quality is measured by a task vector $\mathcal{E}$, not a scalar. Define per-task degradation

$$\delta_t(\theta, R) = \mathcal{E}_t(\theta) - \mathcal{E}_t(R(\theta)),$$

and the **transfer gap** of donor recipe $R_\mathcal{A}$ on recipient $\theta_\mathcal{B}$:

$$\Delta_t(\mathcal{A} \to \mathcal{B}) = \delta_t(\theta_\mathcal{B}, R_\mathcal{A}) - \min_{R' \in \mathcal{R}_b} \delta_t(\theta_\mathcal{B}, R'),$$

where $\mathcal{R}_b$ is the hyperparameter family at fixed $b$. The second term is the **oracle arm** and is what makes the quantity expensive: it requires a full sweep on $\mathcal{B}$.

Candidate architecture-independent predictors, each measured on $\mathcal{D}_{\text{cal}}$:

- Activation excess kurtosis per layer $\ell$: $\kappa_\ell = \mathbb{E}[(x-\mu)^4]/\sigma^4 - 3$, the standard outlier proxy (Bondarenko et al., 2023).
- Outlier mass: fraction of activation coordinates with $|x| > \tau\sigma$, $\tau = 6$ following the LLM.int8() threshold.
- Layer Hessian $H_\ell = \mathbb{E}[x x^\top]$, its stable rank $\|H\|_F^2/\|H\|_2^2$, and the GPTQ objective $\|W X - \tilde W X\|_F^2$.
- Token-to-parameter ratio $\rho = D/N$, the variable that governs quantization sensitivity in Kumar et al. (2024).

Assumptions, with the ones known to be violated marked:

1. $\mathcal{D}_{\text{cal}}$ is drawn from the pretraining distribution. **Violated** — calibration is typically C4 or WikiText for models trained on undisclosed, heavily filtered mixtures.
2. Degradation is monotone in $b$. **Approximately true** within a method, but non-monotone across methods and group sizes.
3. Perplexity ordering implies task ordering. **Violated** — Dutta et al. (2024) show perplexity-matched compressed models diverge on downstream tasks.
4. Layers are independent under compression. **Violated** — sequential error propagation is the whole reason GPTQ/SparseGPT beat round-to-nearest.

## 3. State of the Art

**Established (ablated, independently reproduced):**
- One-shot data-aware weight quantization: GPTQ (Frantar et al., ICLR 2023) and AWQ (Lin et al., MLSys 2024) reliably beat round-to-nearest at 4 bits across many families.
- Rotation removes the outlier problem: QuaRot (Ashkboos et al., NeurIPS 2024) applies computational-invariance Hadamard rotations to make W4A4KV4 feasible; SpinQuant follows with learned rotations. The mechanism — outliers are basis-dependent — is confirmed by SliceGPT (Ashkboos et al., ICLR 2024).
- Incoherence processing plus lattice codebooks: QuIP# (Tseng et al., ICML 2024) holds near-lossless quality near 2–3 bits.

**Claimed but unablated across families:** almost every default. Group size 128, 128 calibration sequences of length 2048, "activation-aware" scaling exponent $\alpha = 0.5$ in SmoothQuant (Xiao et al., ICML 2023) — these were selected on OPT/Llama/BLOOM and are carried unchanged onto Qwen, Mistral, Gemma, Phi, DeepSeek, MoE models, and hybrid-attention models. No paper sweeps them per family and reports the gap.

**Benchmark-number-only results:** most cross-family quantization tables in model cards and toolkit READMEs (LLMC, llm-compressor) report WikiText perplexity and a handful of lm-eval-harness accuracies at one seed, one calibration set, no oracle arm. These establish that a recipe *runs* on a family, not that it is *right* for it.

**Theory SOTA** is separate and thinner: k-bit inference scaling laws (Dettmers & Zettlemoyer, ICML 2023) and precision scaling laws (Kumar et al., 2024) are fit within families and do not carry an architecture term.

## 4. What Is Known

- **4 bits is the compute-optimal point for zero-shot accuracy per bit**, measured across 35k+ experiments on OPT, BLOOM, Pythia and Llama-1, 19M–176B parameters (Dettmers & Zettlemoyer, ICML 2023). This is the most cross-family evidence that exists, and it predates Llama-3-era training regimes.
- **More training tokens make a model harder to quantize post-hoc.** Kumar et al. (2024) fit degradation growing roughly as a power law in $D/N$ over 465 pretraining runs up to 1.7B parameters and 26B tokens. Independent reports of Llama-3-8B (15T tokens) degrading more under 4-bit PTQ than Llama-2-7B (2T tokens) are consistent with this — that is a family-confounded observation, not a controlled one.
- **Outliers are localized and small in number.** LLM.int8() (Dettmers et al., NeurIPS 2022) found emergent outlier features in $\approx 6$ of 4096 hidden dimensions at 6.7B; massive activations concentrate in very few dimensions and tokens (Sun et al., 2024). Their *count and location differ by family*, which is the mechanism by which a transferred recipe can fail.
- **Multilingual damage exceeds English damage.** Marchisio et al. (EMNLP Findings 2024) report automatic metrics understating human-judged quantization degradation, with non-Latin-script languages worst hit — measured on models up to 103B.
- **Perplexity is an insufficient proxy.** Dutta et al. (2024) show compressed variants within ~1% perplexity of the baseline differ by tens of points on individual tasks; "flips" (per-example answer changes) are far more common than net accuracy change suggests.
- **Depth pruning transfers poorly across families.** Gromov et al. (2024) found the fraction of removable deep layers varies substantially by family at fixed size.

## 5. What Is Not Known

- **Empirically open (dominant).** Nobody has published the full factorial — $\geq 5$ families $\times$ $\geq 3$ methods $\times$ $\geq 3$ bit-widths $\times$ per-family hyperparameter sweep — that would give even one measured value of $\Delta_t(\mathcal{A}\to\mathcal{B})$ with an oracle arm. The compute is ordinary; the experiment is unrun.
- **Empirically open.** Whether $\rho = D/N$ alone explains cross-family variance once architecture is held fixed. No open family discloses $D$ reliably, so the regressor's key input is often unavailable.
- **Methodologically blocked.** The recipient-side quality metric. Until "degradation" has a definition that is stable across tasks, languages and seeds, $\Delta_t$ is metric-dependent and different papers can honestly report opposite signs.
- **Theoretically open.** Whether any architecture-independent statistic computable from $O(1)$ forward passes upper-bounds the transfer gap. No lower-bound result exists saying it cannot.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by an absent oracle arm**. Families differ simultaneously in parameter count, token budget, tokenizer, normalization placement, attention variant, instruction-tuning recipe and data mixture. When Llama-3-8B degrades more than Llama-2-7B under identical 4-bit GPTQ, at least six variables moved. Isolating the architecture term requires pretraining matched models — which is the one part of the experiment that is genuinely expensive ($10^{22}$–$10^{23}$ FLOPs for a credible sweep).

Second obstruction: **the oracle term is rarely computed**. Reporting $\delta_t(\theta_\mathcal{B}, R_\mathcal{A})$ without $\min_{R'} \delta_t(\theta_\mathcal{B}, R')$ makes "family $\mathcal{B}$ is hard to quantize" indistinguishable from "the hyperparameters were wrong". Most published cross-family claims are of this form.

Third: **the evaluation does not measure what it names**. WikiText-2 perplexity names "quality" and measures next-token likelihood on 1990s encyclopedic English — insensitive to exactly the long-context, multilingual and tool-use failures that quantization induces.

## 7. Current Research (as of 2026)

- **Rotation-based methods as a family-agnostic normalizer.** QuaRot/SpinQuant lineage (ETH Zurich, Microsoft, Meta). The implicit claim — that rotation removes the family-specific part of the problem — is testable and largely untested. *(frontier — verify)*
- **Precision-aware scaling laws** extended with architecture terms (Harvard/Stanford/Databricks lineage of Kumar et al.). *(frontier — verify)*
- **Standardized compression harnesses** (LLMC, Gong et al. 2024; llm-compressor from Neural Magic/Red Hat) making the factorial cheap to run. The infrastructure now exists; the sweep does not.
- **MoE and hybrid-attention compression**, where expert-level outliers and state-space components break recipe assumptions outright. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Is the transfer gap explained by token-to-parameter ratio $\rho$, or by architecture family?

**Scale.** Pretrain 6 models yourself at 1.4B parameters: 3 architectures (Llama-style pre-LN + SwiGLU + GQA; a QK-norm + post-LN variant; a Gemma-style variant with logit soft-capping and large vocab) $\times$ 2 token budgets ($\rho = 20$ and $\rho = 400$, i.e. 28B and 560B tokens). Cost $\approx 5\times10^{21}$ FLOPs total — a few thousand H100-hours. Identical data, tokenizer held fixed across arms.

**Compression grid.** GPTQ, AWQ, QuaRot at $b \in \{4.25, 3.25, 2.5\}$ effective bits, group sizes $\{32,64,128\}$, calibration $\in$ {C4, pretraining-mixture sample}, $n \in \{64,256\}$ — 108 cells per model.

**Control arm.** For each recipient model, the *oracle*: best cell on that model. Donor recipe = best cell on the Llama-style, $\rho{=}20$ model. Transfer gap is the difference.

**Deciding number.** The variance decomposition of $\Delta$ across the 6 models. If architecture explains $<20\%$ of variance in $\Delta$ once $\rho$ is regressed out, recipe transfer is safe and a $\rho$-only predictor suffices. If architecture explains $>50\%$, per-family re-tuning is mandatory and every shipped default is suspect. Metric: mean over 12 tasks of accuracy drop, plus flip rate, reported with seed variance — not perplexity.

## 9. Key References

- **[Foundational]** Dettmers, Lewis, Belkada, Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. — arXiv:2208.07339
- **[Foundational]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. — arXiv:2210.17323
- **[Foundational]** Dettmers, Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML 2023. — arXiv:2212.09720
- **[SOTA]** Lin, Tang, Tang, Yang, Dang, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024 (Best Paper). — arXiv:2306.00978
- **[SOTA]** Ashkboos, Mohtashami, Croci, Li, Jaggi, Alistarh, Hoefler, Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS 2024. — arXiv:2404.00456
- **[SOTA]** Tseng, Chee, Sun, Kuleshov, De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. — arXiv:2402.04396
- **[Theory]** Kumar, Ankner, Blakeney, Sharma, Dey, Frankle, Kakade, et al. *Scaling Laws for Precision.* 2024. — arXiv:2411.04330
- **[Mechanism]** Bondarenko, Nagel, Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS 2023. — arXiv:2306.12929
- **[Mechanism]** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models.* COLM 2024. — arXiv:2402.17762
- **[Measurement]** Dutta, Krishna, Gupta, et al. *Accuracy is Not All You Need.* NeurIPS 2024. — arXiv:2407.09141
- **[Measurement]** Marchisio, Dash, Chen, Aumiller, Üstün, Hooker, Ruder. *How Does Quantization Affect Multilingual LLMs?* Findings of EMNLP 2024. — arXiv:2407.03211
- **[Related]** Ashkboos, Croci, Nascimento, Hoefler, Hensman. *SliceGPT: Compress Large Language Models by Deleting Rows and Columns.* ICLR 2024. — arXiv:2401.15024
- **[Related]** Frantar, Riquelme, Houlsby, Alistarh, Evci. *Scaling Laws for Sparsely-Connected Foundation Models.* ICLR 2024. — arXiv:2309.08520

## 10. Worked Example

Take AWQ's default: group size $g=128$, 4-bit weights, FP16 scale and 4-bit zero-point. Effective bit-width:

$$b = 4 + \frac{16 + 4}{128} = 4.156\ \text{bits/weight}.$$

Now consider two recipients at nominally the same setting.

- **Model P** (Llama-style, 7B, $d_{\text{model}}=4096$). Measured activation kurtosis in the down-projection input: $\kappa \approx 200$–$1000$ in mid-layers; outlier coordinates $\approx 6$/4096. AWQ's per-channel scaling search, run over $\alpha \in [0,1]$ on 128 C4 sequences, picks $\alpha^\star \approx 0.5$, and the default is the oracle. Transfer gap $\approx 0$.
- **Model Q** (large-vocab, QK-normed, trained to $\rho \approx 2000$). Suppose $\kappa$ peaks two orders of magnitude higher in a single layer, and the outlier coordinates fall unevenly across 128-wide groups — 3 of 32 groups in one row carry all the range. Under $g=128$, those groups' scales are set by one coordinate, so the other 127 weights quantize into 2–3 usable levels.

The naive report: "Model Q loses 1.9 points average accuracy under 4-bit AWQ; Model P loses 0.4. Model Q is harder to quantize."

The obstruction becomes visible when the oracle arm is added. Re-running the same grid on Q with $g=32$ ($b = 4 + 20/32 = 4.625$ bits) and pretraining-matched calibration might recover 1.5 of the 1.9 points. Then $\Delta \approx 0.4$, not $1.9$ — and the residual is confounded with a $0.47$-bit budget difference that the headline "4-bit" hid entirely. Without the oracle, and without reporting $b$ inclusive of scales, the two competing explanations — *Q is intrinsically less compressible* versus *the donor hyperparameters were wrong for Q* — are not separable from published numbers. That inseparability, not the compute, is why the problem is open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*