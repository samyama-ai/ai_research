---
id: 09-model-design/architecture-quantization-robustness
title: "Architectural Determinants of Quantization Robustness"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Architectural Determinants of Quantization Robustness

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/architecture-quantization-robustness` · **Status:** open

## 1. Problem Statement

Two models with the same parameter count, data, and validation loss can differ by an order of magnitude in how much quality they lose when weights and activations are cast to 4 bits. The question is **which architectural choices cause that difference, and whether the causal direction can be established rather than correlated**.

Three variants, of very different difficulty:

- **Measurement.** Define a quantization-robustness score for an architecture that is not confounded by the quantizer, the calibration set, or the evaluation suite. Currently the same model ranks differently under GPTQ, AWQ, and rotation-based methods.
- **Method.** Find architectural interventions — normalization placement, attention gating, activation function, head dimension, MLP expansion ratio, positional scheme — that reduce post-training quantization (PTQ) loss at fixed pretraining loss and fixed inference FLOPs. The constraint "at fixed pretraining loss" is what makes this hard; almost any intervention that suppresses outliers also costs some full-precision quality.
- **Theory.** Predict, from the architecture and training hyperparameters alone, the bit width $b^\*$ below which degradation becomes superlinear. No such predictor exists.

A solution to the method variant would be a recipe: change $X$, lose $\le \epsilon$ full-precision loss, gain $\ge \delta$ at 4 bits, reproduced across at least two labs and two scales.

## 2. Formal Setting

Let $f_\theta$ be a model with parameters $\theta$, and let $Q_b$ be a quantizer at $b$ bits, defined by a calibration set $C$ and a rounding rule. Write $\hat\theta = Q_b(\theta; C)$.

**Degradation** is measured, not assumed, as the gap in cross-entropy on a held-out corpus $D$:

$$\Delta_b(f, Q, D) = \mathcal{L}\big(f_{Q_b(\theta;C)}, D\big) - \mathcal{L}(f_\theta, D)$$

with $\mathcal{L}$ in nats/token on a fixed tokenization. Perplexity ratios are *not* interchangeable across tokenizers; comparisons of architectures with different vocabularies are invalid unless renormalized per byte.

**Outlier magnitude.** For layer $\ell$ and hidden dimension $j$, with activations $a^{(\ell)}_{t,j}$ over tokens $t$ in a calibration batch, the per-channel kurtosis-free measure actually used in practice is the massive-activation ratio

$$\rho^{(\ell)} = \frac{\max_{t,j} |a^{(\ell)}_{t,j}|}{\mathrm{median}_{t,j} |a^{(\ell)}_{t,j}|}.$$

Values $\rho > 10^3$ are what break per-tensor int8. Measured on one calibration batch, $\rho$ has high variance; report over $\ge 128$ sequences.

**Quantization difficulty of a weight matrix** $W$ under round-to-nearest with per-channel scales is bounded through the incoherence constant used by QUIP: $W$ is $\mu$-incoherent if $\max_{i,j}|W_{ij}| \le \mu \|W\|_F / \sqrt{mn}$. Rotation-based methods reduce $\mu$; the bound is on proxy error $\|(W - \hat W)x\|$, not on $\Delta_b$.

**Fair-comparison constraint.** Architectures $A_1, A_2$ are comparable only at matched $\mathcal{L}(f_\theta, D)$ and matched inference cost. In practice this is enforced by training to equal validation loss, which changes token counts — and token/parameter ratio is itself a strong determinant of $\Delta_b$ (§4). This assumption is *known to be violated* in nearly every published architecture-vs-quantization comparison.

Other assumptions routinely violated: (i) the calibration set is drawn from $D$ — usually it is C4 while $D$ is WikiText or a task suite; (ii) $\Delta_b$ on language modeling tracks $\Delta_b$ on downstream tasks — it does not for reasoning-heavy evaluations; (iii) the quantizer is architecture-neutral — GPTQ's Hessian estimate depends on activation covariance, which is exactly what the architecture changes.

## 3. State of the Art

**Established (ablated, reproduced).**
- Outlier features in transformer activations are systematic and concentrate in a few hidden dimensions; LLM.int8() (Dettmers et al., NeurIPS 2022) showed emergence around 6.7B parameters and mixed-precision decomposition restores full-precision quality.
- Rotation removes most of the outlier problem. QuaRot (Ashkboos et al., NeurIPS 2024) applies Hadamard rotations that leave the network function unchanged and reports W4A4KV4 Llama-2-70B within 0.47 WikiText-2 perplexity of FP16. SpinQuant (Liu et al., ICLR 2025) learns the rotations and narrows the W4A4KV4 zero-shot gap on Llama-2-7B to ~2.9 points.
- Architectural attention fixes work at small scale. Bondarenko et al. (NeurIPS 2023) traced outliers to attention heads trying to emit a no-op, and showed clipped softmax and gated attention remove them, giving int8-quantizable BERT and OPT-family models up to 350M–1.5B with no full-precision loss.

**Claimed but not fully ablated.**
- That outliers are *inevitable at scale*. Ahmadian et al. (NeurIPS 2023) is the counterexample: optimizer and regularization choices (weight decay, gradient clipping, dropout, fp16 vs bf16 training) control outlier emergence in models up to 52B. This reframes "architectural determinant" as partly an *optimization* determinant, and the two are not disentangled anywhere.
- That specific modern components (QK-norm, gated attention, no-positional-encoding, sink tokens) improve quantizability. Reported in model cards and system reports; no controlled pair at matched loss.

**Benchmark-number-only results.** Most "model X quantizes better than Y" claims — Llama-3 vs Llama-2, MoE vs dense, Mamba vs transformer — exist only as perplexity/accuracy tables at one scale, one quantizer, one calibration set. They do not isolate architecture from token budget or training recipe.

## 4. What Is Known

- **Outlier onset.** Emergent outlier features appear across all transformer layers at ~6.7B parameters and grow smoothly before that (Dettmers et al., 2022, OPT/BLOOM families up to 175B).
- **Massive activations.** Llama-2-7B has ~3 activation coordinates exceeding the median magnitude by ~$10^4$, concentrated at specific token positions (delimiters, BOS) and fixed dimensions; they act as attention bias (Sun et al., COLM 2024; measured at 7B–70B).
- **4-bit is the accuracy-per-bit optimum** for weight-only PTQ across 35,000+ zero-shot runs, 19M–176B parameters (Dettmers & Zettlemoyer, ICML 2023). Below 4 bits the frontier bends unless the quantizer changes.
- **Token/parameter ratio drives PTQ fragility.** Kumar et al. (ICLR 2025), 465 pretraining runs up to 1.7B parameters and 26B tokens: models trained on more data per parameter degrade *more* under PTQ, with degradation growing roughly as a power law in the data/parameter ratio. This is the strongest known non-architectural confound.
- **LayerNorm scaling amplifies outliers**; folding $\gamma$ into subsequent weights reduces activation range (Wei et al., NeurIPS 2022; BERT/BART scale).
- **Rotation invariance is exact**, not heuristic: orthogonal $R$ inserted as $W R$ / $R^\top x$ preserves the function while reducing incoherence $\mu$ (QuIP, NeurIPS 2023; QuaRot 2024).

## 5. What Is Not Known

- **Theoretically open.** No theorem links an architectural property (normalization placement, gating, head dimension) to a bound on $\Delta_b$. Existing bounds (QuIP-style) are on layer-wise proxy error under incoherence assumptions and do not compose into an end-to-end loss bound for a deep network.
- **Empirically open.** The matched-loss, matched-FLOPs, multi-architecture sweep at $\ge$7B has not been run. Every ingredient exists; nobody has paid for the ~$10^5$ GPU-hour grid. Also open: whether attention-level fixes that work at 350M still work at 70B, and whether MoE experts quantize differently from dense FFNs at equal active parameters.
- **Methodologically blocked.** "Quantization robustness of an architecture" is not yet a well-defined quantity: it depends on the quantizer, and the best quantizer differs by architecture. Any score is a max over methods, and the method set is open-ended. Reported $\rho$ and kurtosis are also calibration-set dependent with no standard.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**. Architecture, optimizer, and token budget jointly determine outlier structure. Ahmadian et al. showed the optimizer alone can flip the outcome; Kumar et al. showed the token budget alone can. So a two-model comparison has at least three free variables and one observation. Controlling them requires retraining, which is the compute cost: an honest 2-architecture × 3-scale × 2-token-budget grid at 7B is on the order of $10^5$ A100-hours.

Second obstruction: **the evaluation does not measure what it names.** Perplexity gap under-reports quantization damage on multi-step tasks — a model can lose <1% perplexity and a large fraction of chain-of-thought accuracy, because errors compound over generated tokens. Robustness rankings therefore invert between LM-loss and agentic evaluations.

## 7. Current Research (as of 2026)

- **Rotation and incoherence processing** as the default preprocessing layer (QuaRot, SpinQuant, QuIP#/QTIP lines; ETH Zürich, Meta, Cornell). Direction: fusing learned rotations into pretraining so the deployed model is natively rotated. *(frontier — verify)*
- **Quantization-aware architecture search**, where PTQ loss is an objective alongside pretraining loss. Mostly industrial, reported in model reports rather than papers. *(frontier — verify)*
- **Attention-sink engineering** — explicit sink tokens and gated/clipped softmax adopted in several 2025 open-weight releases specifically to reduce activation range. *(frontier — verify)*
- **Native low-precision training** (BitNet b1.58 line, Microsoft Research; fp4/fp8 pretraining at NVIDIA), which sidesteps PTQ entirely and changes what "robustness" means.
- **Precision-aware scaling laws** extending Kumar et al. to architecture terms.

## 8. Concrete Next Experiment

**Question.** Does attention gating (Bondarenko-style) reduce W4A4 degradation at matched full-precision loss, at a scale where outliers are known to exist?

- **Scale.** Two 7B-parameter decoder models, identical data order, identical optimizer, 300B tokens (~43 tokens/param, inside the regime where Kumar et al. predicts measurable PTQ fragility). Arm A: standard softmax attention. Arm B: gated attention (per-head sigmoid gate on the attention output), $<0.2\%$ extra FLOPs. Cost estimate: ~2 × 8×10$^{22}$ FLOPs, roughly 70k H100-hours total.
- **Control arm.** A third run, Arm A′, identical to A but stopped at the token count that matches Arm B's *final validation loss*. This is the arm that removes the loss-vs-token-budget confound and is the one usually missing.
- **Quantizers.** RTN, GPTQ, and QuaRot, each with the same 128-sequence C4 calibration set, evaluated at W4A4KV4.
- **Deciding number.** $\Delta_4$ in nats/token on held-out C4, best-over-quantizers, for Arm B minus Arm A′. A gap $\le -0.02$ nats/token (≈2% perplexity) with non-overlapping seed intervals establishes gating as a genuine architectural determinant; a gap within $\pm 0.005$ falsifies it and shifts the explanation to the optimizer/token budget. Report $\rho^{(\ell)}$ per layer as the mechanistic secondary.

## 9. Key References

- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Foundational]** Tim Dettmers, Luke Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Foundational]** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023. — arXiv:2306.12929
- **[SOTA]** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[SOTA]** Zechun Liu, Changsheng Zhao, Igor Fedorov, et al. *SpinQuant: LLM Quantization with Learned Rotations.* ICLR, 2025. — arXiv:2405.16406
- **[SOTA]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** Ji Lin, Jiaming Tang, Haotian Tang, et al. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[Theory]** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees.* NeurIPS, 2023. — arXiv:2307.13304
- **[Key result]** Arash Ahmadian, Saurabh Dash, Hongyu Chen, et al. *Intriguing Properties of Quantization at Scale.* NeurIPS, 2023. — arXiv:2305.19268
- **[Key result]** Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, et al. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330
- **[Mechanism]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[Related]** Guangxuan Xiao, Ji Lin, Mickael Seznec, et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438

## 10. Worked Example

Take Llama-2-7B. Measured massive activations: roughly 3 coordinates in the residual stream reach $|a| \approx 2\times10^3$ against a median $|a| \approx 0.2$, so $\rho \approx 10^4$ (Sun et al., 2024).

Per-tensor symmetric int4 uses step $s = \max|a| / 7$. With $\max|a| = 2000$, $s \approx 286$. A typical activation of magnitude $0.2$ rounds to $0$. Effectively the whole tensor collapses: the signal-carrying coordinates get $\lfloor 0.2/286 \rceil = 0$ and only the outliers survive. This is why naive W4A4 destroys the model.

Now rotate. A random Hadamard $R$ of size $4096$ spreads a single spike of magnitude $M$ across all coordinates, each of size $\approx M/\sqrt{4096} = M/64$. With $M = 2000$, the new max is $\approx 31$ plus the pre-existing bulk, so $s \approx 4.5$ and the bulk at $0.2$ still rounds to $0$ — one rotation is not enough by itself; QuaRot's gain comes from doing this per-block with online Hadamards *and* keeping per-channel scales, which brings the effective $\rho$ into single digits.

**Where the obstruction becomes visible.** Suppose a new architecture halves $\rho$ from $10^4$ to $5\times10^3$. Post-rotation this changes $s$ by a factor of 2 in one layer and moves WikiText perplexity by perhaps 0.05 — inside seed noise for a single 7B run. To claim the architecture caused it you must also show the two models had the same full-precision loss; but the gated variant in the literature trades ~0.3% perplexity for outlier removal, which at 43 tokens/param is worth roughly the same 0.05. **The effect size and the confound are the same size.** That is the problem: the measurement is not blocked by instrumentation, it is blocked by the fact that no published comparison holds the loss fixed, and holding it fixed costs an extra pretraining run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*