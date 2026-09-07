---
id: 09-model-design/architecture-determined-quantizability
title: "Architecture Choices That Determine Quantizability"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Architecture Choices That Determine Quantizability

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/architecture-determined-quantizability` · **Status:** empirically-open

## 1. Problem Statement

Two models trained to the same loss on the same data can differ by several bits in how far they can be quantized before accuracy collapses. The difference is attributed to architecture: normalization placement, attention softmax form, gated vs. plain MLPs, head count, residual scaling, embedding tying, vocabulary size. The problem is to turn that attribution into a predictive rule.

- **Measurement variant.** Define a scalar *quantizability* of a trained checkpoint that is (i) independent of the specific post-training quantization (PTQ) algorithm used and (ii) predicts end-task degradation. No accepted definition exists.
- **Method variant.** Find architectural modifications that raise quantizability at fixed full-precision loss and fixed training FLOPs. Several candidates exist (gated attention, clipped softmax, rotation-friendly norms); none has been shown to hold at $\geq$70B scale.
- **Theory variant.** Prove that some architectural family provably admits low-precision representation with bounded loss increase — i.e., a bound on $\Delta L$ in terms of architectural constants rather than measured activation statistics.

Solving it means: given an architecture spec and a token budget, predict the lowest weight/activation bit-width at which perplexity degradation stays under a threshold, *before* training.

## 2. Formal Setting

Let $f_\theta$ be a transformer with parameters $\theta \in \mathbb{R}^P$ trained on $D$ tokens, with test loss $L(\theta)$. A quantization scheme $Q_{b_w,b_a}$ maps weights to $b_w$ bits and activations to $b_a$ bits under a chosen granularity (per-tensor, per-channel, per-group of $g$ elements).

**Degradation.** The measured quantity is
$$\Delta L(b_w,b_a) = L\big(Q_{b_w,b_a}(\theta)\big) - L(\theta),$$
evaluated as WikiText-2 or C4 next-token cross-entropy on a held-out split of $\geq 10^6$ tokens (perplexity differences below ~0.05 are within seed noise at 7B scale).

**Quantizability.** For a degradation budget $\epsilon$ (typically $\epsilon = 0.1$ nats or 1 point of zero-shot accuracy),
$$b^\star(\epsilon; \mathcal{A}) = \min\{ b : \Delta L(b,b) \leq \epsilon \},$$
where $\mathcal{A}$ is the architecture. $b^\star$ depends on the PTQ algorithm; the algorithm-independent version replaces $Q$ with the best known method at the time of measurement, which makes $b^\star$ a moving target — this is the core measurement defect.

**Outlier statistics.** The standard architectural proxy is the per-channel activation kurtosis or the infinity-to-mean ratio at layer $\ell$:
$$\rho_\ell = \frac{\max_{i,j} |X^{(\ell)}_{ij}|}{\frac{1}{nd}\sum_{i,j} |X^{(\ell)}_{ij}|},$$
measured over a calibration set of 128–512 sequences of length 2048. For uniform min–max quantization to $b$ bits, per-tensor step size is $\Delta = \frac{\max|X| - \min|X|}{2^b - 1}$, so effective bits on a typical channel scale as $b - \log_2 \rho_\ell$: an outlier ratio of $\rho = 64$ costs 6 bits.

**Assumptions and where they break.**
- *Uniform quantization noise* ($\mathbb{E}[\eta^2] = \Delta^2/12$, independent across coordinates) — violated; rounding error correlates with weight magnitude, which is why GPTQ's error-feedback helps.
- *Loss is locally quadratic*, $\Delta L \approx \tfrac12 \eta^\top H \eta$ — violated at $b \leq 3$ where $\|\eta\|$ leaves the quadratic basin.
- *Architecture and data are separable* — violated: Kumar et al. show $\Delta L$ grows with $D/P$, so the same architecture is less quantizable when trained longer.
- *Calibration set is representative* — violated for long-context and code, where outlier channels differ.

## 3. State of the Art

**Established (reproduced independently):**
- Outlier features emerge as a phase transition with scale: LLM.int8() (Dettmers et al., NeurIPS 2022) reports systematic outlier dimensions appearing across all layers at ~6.7B parameters, in <0.1% of feature dimensions, with magnitudes ~20$\times$ typical. Reproduced by many groups.
- Rotation removes outliers. QuaRot (Ashkboos et al., NeurIPS 2024) and SpinQuant (Liu et al., 2024) apply computational-invariance Hadamard/learned rotations to make activations near-Gaussian; QuaRot reports W4A4KV4 Llama-2-70B within roughly 0.5 WikiText-2 perplexity of FP16. This is now standard and widely replicated — it is an *inference-time* architectural change (fold rotations into weights), and it is the strongest evidence that quantizability is a property of the basis, not the function.
- Massive activations are a small, fixed set of coordinates acting as learned biases. Sun et al. (*Massive Activations in Large Language Models*, COLM 2024) find them in a handful of dimensions in Llama-2/Mixtral/Phi, tied to attention sinks (Xiao et al., ICLR 2024).

**Claimed but unablated at scale:**
- Training-time architectural fixes. Bondarenko et al. (*Quantizable Transformers*, NeurIPS 2023) show clipped softmax and gated attention nearly eliminate outliers in BERT and OPT up to 1.3B with no perplexity cost. Never replicated at $\geq$7B, never combined with rotation-based PTQ.
- Normalization choice (RMSNorm vs. LayerNorm, pre- vs. post-norm, QK-norm) is repeatedly asserted to matter for quantizability; there is no matched-FLOP ablation isolating it.
- Ternary/1.58-bit training (BitNet b1.58, Ma et al. 2024) claims parity with FP16 baselines at 3B. The comparison is against a reproduced LLaMA recipe, not a jointly tuned control; treat the parity claim as a benchmark number.

**Theory SOTA** is weaker: k-bit inference scaling laws (Dettmers & Zettlemoyer, ICML 2023) are empirical fits, not bounds. No theorem connects an architectural constant to $b^\star$.

## 4. What Is Known

- **4 bits is bit-optimal for weight-only PTQ.** Across 35k zero-shot evaluations spanning 19M–176B parameters (OPT, BLOOM, Pythia, GPT-2), 4-bit weights maximize zero-shot accuracy per total model bit; 3-bit is worse at fixed bits (Dettmers & Zettlemoyer, ICML 2023).
- **More training data makes a model less quantizable.** Kumar et al. (*Scaling Laws for Precision*, 2024, arXiv:2411.04330) fit >465 pretraining runs up to 1.7B parameters / 26B tokens and find PTQ degradation increasing as a power law in $D/P$ — for large enough $D$, additional pretraining tokens *reduce* post-quantization accuracy.
- **Outlier magnitude, not count, drives failure.** In OPT-175B a few hundred coordinates carry infinity norms two orders of magnitude above the mean; zeroing them destroys the model, while zeroing an equal number of random coordinates does not (Sun et al., COLM 2024).
- **Attention softmax forces the outlier.** Softmax cannot output zero attention, so heads that should "do nothing" push value/residual coordinates to extremes; adding a no-op escape hatch removes the outliers (Bondarenko et al., EMNLP 2021 and NeurIPS 2023; BERT-base and OPT up to 1.3B).
- **GLU/SwiGLU MLPs concentrate outliers at the down-projection input.** Widely observed in Llama-family models and the reason AWQ (Lin et al., MLSys 2024) and SmoothQuant (Xiao et al., ICML 2023) apply per-channel scale migration precisely there.

## 5. What Is Not Known

- **Empirically open.** Whether any training-time architectural fix (gated attention, clipped softmax, QK-norm, tanh-bounded residuals) still yields a quantizability gain *after* rotation-based PTQ is applied, at $\geq$7B and $\geq$1T tokens. Both arms are runnable today; nobody has published the paired run. This is the load-bearing gap.
- **Empirically open.** Whether the $D/P$ degradation law of Kumar et al. has an architecture-dependent coefficient, or only an architecture-independent exponent.
- **Methodologically blocked.** $b^\star$ is defined relative to the best available PTQ algorithm. When QuaRot cut W4A4 loss by ~1 perplexity point, every prior architecture's $b^\star$ changed retroactively. There is no PTQ-invariant quantizability measure.
- **Theoretically open.** No bound of the form $\Delta L \leq g(\mathcal{A}) \cdot 2^{-2b}$ with $g$ computable from the architecture. Even for a single linear layer with correlated inputs, the tight rate is unknown outside the Gaussian case.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a moving reference**. Quantizability is measured as a difference $\Delta L$ between two models, but changing architecture changes both terms: an architecture with slightly worse full-precision loss can show smaller $\Delta L$ simply because it sits in a flatter basin. Matching full-precision loss requires retuning learning rate and token budget per arm, which costs a full pretraining run per architectural variant — at 7B/1T tokens that is roughly $10^{22}$ FLOPs per arm, and a 4-arm ablation with 2 seeds is a several-hundred-thousand-dollar experiment. Meanwhile the quantizer improves faster than the ablations complete, so any $b^\star$ measured is stale by publication. Small-scale proxies do not substitute: the outlier phenomenon the experiment is about *does not exist* below ~6.7B parameters.

## 7. Current Research (as of 2026)

- **Rotation and incoherence processing** — QuaRot/SpinQuant/QuIP# lineage (ETH Zürich, Meta, Cornell). Direction: fold rotations into the architecture at pretraining time so no inference-time overhead remains. *(frontier — verify)*
- **Quantization-aware pretraining at the format level** — MXFP4/NVFP4 microscaling formats, driven by hardware vendors; the open question is whether block-scaled formats make architectural outlier fixes unnecessary. *(frontier — verify)*
- **Precision-aware scaling laws** — extending Kumar et al. to architecture-conditioned coefficients (CMU/Harvard/Databricks lines of work). *(frontier — verify)*
- **Attention-sink-aware design** — explicit sink tokens, off-by-one softmax, gated attention variants appearing in open frontier model releases. Public model cards note the choices; ablations are rarely published.

## 8. Concrete Next Experiment

**Question.** Does a training-time outlier fix still buy bits once rotation-based PTQ is used?

**Scale.** Four pretraining runs at 7B parameters, 300B tokens (Llama-3-style tokenizer, identical data order, 2 seeds on the control). ~$3\times10^{22}$ FLOPs total.

**Arms.**
1. **Control:** standard pre-RMSNorm / SwiGLU / softmax attention.
2. Gated attention (Bondarenko et al., NeurIPS 2023) — softmax with a learned per-head gate.
3. QK-norm + tanh-clipped residual writes.
4. Hadamard rotations folded into the architecture from step 0.

**Protocol.** Match FP16 validation loss to within 0.01 nats by tuning LR only. Then quantize every arm with (a) round-to-nearest, (b) GPTQ, (c) QuaRot, at W4A4KV4 and W3A8.

**Deciding number.** $\Delta L_{\text{QuaRot,W4A4}}(\text{arm}) - \Delta L_{\text{QuaRot,W4A4}}(\text{control})$ in nats on C4. A gain $\leq 0.02$ nats (below seed noise) for arms 2–4 means architectural outlier fixes are subsumed by rotation and the design question is closed in favor of "train normally, rotate at inference". A gain $\geq 0.05$ nats for any arm means architecture carries quantizability independently, and the sub-question becomes which component.

## 9. Key References

- **[Foundational]** Dettmers, Lewis, Belkada, Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Foundational]** Bondarenko, Nagel, Blankevoort. *Understanding and Overcoming the Challenges of Efficient Transformer Quantization.* EMNLP, 2021. — arXiv:2109.12948
- **[SOTA]** Bondarenko, Nagel, Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023. — arXiv:2306.12929
- **[SOTA]** Ashkboos, Mohtashami, Croci, Li, Jaggi, Alistarh, Hoefler, Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[SOTA]** Liu, Zhao, Fedorov, Soran, Choudhary, Krishnamoorthi, Chandra, Tian, Blankevoort. *SpinQuant: LLM Quantization with Learned Rotations.* 2024. — arXiv:2405.16406
- **[SOTA]** Lin, Tang, Tang, Yang, Dang, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[Key result]** Dettmers, Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Key result]** Kumar, Ankner, Blumofe, Frankle, et al. *Scaling Laws for Precision.* 2024. — arXiv:2411.04330
- **[Key result]** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[Related]** Xiao, Lin, Seznec, Wu, Demouth, Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438
- **[Related]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453

## 10. Worked Example

Take one down-projection input in a 7B SwiGLU MLP: $d = 11008$, per-tensor symmetric INT4 activation quantization, calibration over 128 sequences of length 2048.

Measured (typical of Llama-2-7B middle layers): mean $|x| \approx 0.35$, $\max |x| \approx 42$, so $\rho \approx 120$.

Per-tensor step: $\Delta = \frac{2 \times 42}{2^4 - 1} = 5.6$. A typical coordinate of magnitude 0.35 falls entirely inside the first bin — it quantizes to zero. Effective bits on the typical coordinate: $4 - \log_2 120 \approx -2.9$, i.e. the layer carries no usable signal. Squared error per coordinate under the uniform model is $\Delta^2/12 = 2.6$, against a signal power of about $0.35^2 = 0.12$ — SNR of $-13$ dB.

Now the obstruction. Three interventions all fix this number, and the metric cannot separate them:
- **Per-group scaling** ($g=128$): outliers are confined to a few groups, so most groups see $\max|x| \approx 1.2$, $\Delta = 0.16$, SNR $\approx +10$ dB. Architecture unchanged.
- **Rotation** (QuaRot): a random Hadamard on the 11008-dim basis spreads the outlier, giving $\max|x| \approx 42/\sqrt{11008} \cdot c \approx 1.5$ under near-Gaussian mixing. Architecture unchanged at train time.
- **Gated attention** upstream: the outlier is never created, $\rho \approx 8$.

All three land $\Delta L$ within noise of each other at W4. So the observed quantizability of the *checkpoint* is not a property of the checkpoint — it is a property of the (basis, granularity, algorithm) triple applied afterwards. Any claim that architecture $\mathcal{A}$ is "more quantizable" that is not conditioned on the strongest available PTQ is measuring the quantizer, not the architecture. That is exactly why the Section 8 experiment puts QuaRot on *every* arm.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*