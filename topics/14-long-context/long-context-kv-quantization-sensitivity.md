---
id: 14-long-context/long-context-kv-quantization-sensitivity
title: "Quantization Sensitivity of Long-Context Keys and Values"
topic: 14-long-context
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Quantization Sensitivity of Long-Context Keys and Values

> **Topic:** Long Context · **ID:** `14-long-context/long-context-kv-quantization-sensitivity` · **Status:** empirically-open

## 1. Problem Statement

KV cache memory is linear in context length, so long-context inference is quantized in practice: 4-bit or 2-bit keys and values are standard in serving stacks. The question is whether the *acceptable* bit-width is itself a function of context length $L$.

Three variants, with different difficulty:

- **Measurement.** Define a degradation functional $\Delta_b(L)$ = task accuracy at fp16 minus accuracy at $b$ bits, at context length $L$, holding everything else fixed. Is $\Delta_b(L)$ increasing in $L$? Solving the measurement variant means producing an estimator of $\Delta_b(L)$ that is not confounded by the task getting harder with $L$ on its own.
- **Method.** Find a quantizer $Q_b$ whose $\Delta_b(L)$ is flat in $L$ up to 1M tokens at $b \le 4$, or show that any flat-$\Delta$ quantizer must spend $\Omega(\log L)$ bits per entry.
- **Theory.** Bound the attention-output perturbation induced by per-entry KV error as a function of $L$, the score distribution, and the quantization grid — tightly enough to predict, not merely bound, observed degradation.

Solving it means: given a model, a target task, and $L$, predict the minimum $b$ within $\pm 0.5$ bits without running the long-context evaluation.

## 2. Formal Setting

For one attention head with head dimension $d_h$, the cache after $L$ tokens is $K, V \in \mathbb{R}^{L \times d_h}$. A query $q \in \mathbb{R}^{d_h}$ produces scores $s = Kq/\sqrt{d_h}$, weights $p = \mathrm{softmax}(s)$, output $o = V^\top p$.

**Quantizer.** Group-wise affine quantization over groups of size $g$: for group $G$,
$$\hat{x} = \sigma \cdot \mathrm{clip}\big(\lfloor x/\sigma \rceil + z, 0, 2^b-1\big) - \sigma z, \qquad \sigma = \frac{\max_G x - \min_G x}{2^b - 1}.$$
Grouping is **per-channel** ($G$ = a column of $K$, spanning tokens) or **per-token** ($G$ = a row). The effective rate, which is the number that must be reported and often is not, includes metadata:
$$b_{\text{eff}} = b + \frac{b_\sigma + b_z}{g}.$$
At $b=2$, $g=32$, fp16 scale and zero: $b_{\text{eff}} = 3.0$, a 50% overhead on the headline number.

**Error propagation.** Let $\Delta K = \hat K - K$, $\Delta V = \hat V - V$. The softmax Jacobian has spectral norm $\le 1/2$ (Gao & Pavel, 2017), so
$$\|\hat p - p\|_2 \;\le\; \tfrac{1}{2}\|\Delta s\|_2 \;\le\; \frac{\|q\|_2\,\sqrt{L}\,\max_j\|\Delta k_j\|_2}{2\sqrt{d_h}},$$
$$\|\hat o - o\|_2 \;\le\; \underbrace{\|\hat p - p\|_1 \max_j \|v_j\|_2}_{\text{key error, } L\text{-dependent}} \;+\; \underbrace{\max_j \|\Delta v_j\|_2}_{\text{value error, } L\text{-free}}.$$
The $\sqrt{L}$ factor is the formal statement of the problem: the key-error term has an explicit length dependence, the value-error term does not. This is a **worst-case bound and is loose** — it assumes adversarial alignment of per-token errors. Under i.i.d. errors and a peaked $p$, the realized growth could be $O(1)$.

**Measured quantities.** $\Delta_b(L)$ is measured on a task whose difficulty is held constant in $L$ — RULER-style synthetic retrieval with fixed needle count and fixed distractor density, varying only filler length. Perplexity on long documents is *not* a valid estimator of $\Delta_b(L)$: mean next-token loss is dominated by short-range tokens whose count grows with $L$, diluting exactly the long-range failures being measured.

**Assumptions, and which are violated.**
1. *Errors are unstructured.* Violated. Key channels carry massive, persistent outliers (Sun et al., COLM 2024); a handful of channels have magnitudes $10^2$–$10^3\times$ the median, which per-token grouping smears across the whole row.
2. *Attention is diffuse.* Violated. Attention sinks put large mass on the first few tokens (Xiao et al., ICLR 2024), so $\|p\|_2 \approx 1$ and the bound above is far from tight.
3. *RoPE commutes with quantization.* False. Rotary embedding mixes channel pairs, so pre-RoPE and post-RoPE key outliers live in different bases; KVQuant quantizes pre-RoPE for this reason.
4. *The evaluation task saturates the context.* Usually violated — most long-context benchmarks are solvable from a short span.

## 3. State of the Art

**Empirical/systems SOTA.**
- **KIVI** (Liu et al., ICML 2024): per-channel keys, per-token values, 2-bit, with a small fp16 residual window. Reports 2.6× cache reduction and 2.35–3.47× throughput at near-baseline accuracy. *Established* for the reported tasks; the length-scaling of its gap is **not** ablated — evaluations sit mostly under 8K.
- **KVQuant** (Hooper et al., NeurIPS 2024): pre-RoPE per-channel keys, non-uniform (sensitivity-weighted) grids, dense-and-sparse outlier split. Reports $<0.1$ perplexity degradation at 3-bit on Llama-2, and 10M-token context on a single 8-GPU node. The 10M figure is a *memory-feasibility* result, not an accuracy result at 10M.
- **QuaRot** (Ashkboos et al., NeurIPS 2024) and **SpinQuant** (Liu et al., 2024): Hadamard/learned rotations flatten outlier channels, enabling 4-bit KV. Established as a mechanism; long-context ablation absent.
- **Coupled Quantization** (Zhang et al., NeurIPS 2024): exploits inter-channel dependence to reach ~1 bit/channel on reported benchmarks. Claimed, single-group result.
- **Evaluation SOTA.** **RULER** (Hsieh et al., COLM 2024) is the standard length-controlled probe. **SCBench** (Li et al., ICLR 2025) evaluates KV compression across multi-turn reuse and is the closest existing instrument to the question here.

**Theory SOTA.** No length-dependent bound tight enough to predict $b$. The nearest formal anchor is **precision-aware scaling laws** (Kumar et al., ICLR 2025), which model post-training-quantization degradation as growing with tokens seen during *training* — a different axis, but the same functional shape.

**Claimed but unablated across the board:** that a bit-width validated at 4K transfers to 128K. Nearly every KV-quantization paper implicitly assumes this and none tests it as its primary axis.

## 4. What Is Known

- **Keys are harder than values.** Per-channel key grouping plus per-token value grouping is the reproduced recipe (KIVI, KVQuant, independently). Swapping the two costs several perplexity points at 2 bits on Llama-2-7B.
- **Key outliers are channel-structured and persistent** across tokens and inputs; measured on Llama-2-7B/13B, Mistral-7B, at 7B–13B scale (Sun et al., COLM 2024).
- **3-bit is nearly free at short context.** $<0.1$ perplexity delta on Llama-2-7B/13B, WikiText-2, sequences of 2K–4K (KVQuant).
- **2-bit is not free without structure.** Naive 2-bit round-to-nearest collapses; KIVI recovers it only with a fp16 residual window of 32–128 recent tokens.
- **Compression methods degrade unevenly by task.** KV-compression benchmarking (Yuan et al., EMNLP Findings 2024) shows retrieval and arithmetic degrade well before summarization at matched compression — measured at 7B scale.
- **Claimed context $\ne$ usable context even at fp16.** RULER shows most models with 32K+ claimed windows fall off well below the claim, at 7B–70B scale. Any $\Delta_b(L)$ measurement must be taken against this already-declining fp16 baseline.

## 5. What Is Not Known

- **Empirically open (the core gap).** Whether $\Delta_b(L)$ grows with $L$ at fixed task difficulty. The experiment is entirely runnable — it needs a length-controlled benchmark, four bit-widths, and about 10<sup>3</sup> GPU-hours — and has not been run as a primary result at 128K+ on a frontier-scale model.
- **Empirically open.** Whether attention sinks *protect* against key quantization error (mass concentrated on few tokens $\Rightarrow$ few error terms) or *amplify* it (sink tokens are exactly the massive-outlier tokens, so their quantization error is largest).
- **Theoretically open.** Whether an $L$-dependent lower bound on bits/entry exists for maintaining top-$k$ retrieval fidelity under attention. No proof either way. A plausible target: exact argmax-preservation over $L$ keys with i.i.d. sub-Gaussian scores requires $b = \Omega(\log\log L)$.
- **Methodologically blocked.** Attributing degradation to *quantization* versus *the model's own length limits*. There is no accepted way to separate "the 4-bit model failed at 128K" from "the fp16 model was already failing at 128K and 4 bits moved it over threshold." Ratio, difference, and length-of-equal-accuracy estimators disagree in sign on real data.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a floor effect**. $\Delta_b(L)$ is a difference of two quantities that both degrade in $L$ for unrelated reasons. On RULER-style tasks, fp16 accuracy often falls from ~95% at 4K to ~60% at 128K. A quantized arm scoring 55% at 128K yields a 5-point gap — smaller than its 8-point gap at 4K where the baseline was near ceiling — so the naive difference estimator reports that quantization *helps* at long context. The ratio estimator reports the opposite. Neither is wrong; the metric is not defined on a scale where differences are comparable across $L$.

Second obstruction: **cost asymmetry**. A 128K-token evaluation is 32× the prefill FLOPs of a 4K one per sample, and variance across needle placements requires many samples, so the length axis is the most expensive axis to sweep — which is precisely why every paper sweeps bit-width and method instead.

## 7. Current Research (as of 2026)

- **Rotation-based outlier removal extended to long context** — QuaRot/SpinQuant lineage, ETH Zürich and Meta. Open question is whether a single rotation suffices when the outlier basis drifts with position after RoPE. *(frontier — verify)*
- **Mixed-precision-by-position**: fp16 for sink tokens and a recent window, 2-bit for the middle. Present in KIVI and StreamingLLM-descended systems; a principled allocation rule is not established.
- **Layer- and head-adaptive bit allocation** driven by measured sensitivity (KVTuner and related 2025 work). *(frontier — verify)*
- **Quantization interacting with sparse attention/eviction** — SCBench-style evaluation of compression stacks, Microsoft Research and academic groups.
- **Cache quantization in serving systems** (vLLM, TensorRT-LLM, SGLang) where fp8 KV is now default and int4 optional; deployment data on long-context regressions exists inside providers and is not published.

## 8. Concrete Next Experiment

**Question.** Does the quantization gap grow with context length at fixed task difficulty?

**Scale.** Two models spanning a size decade — Llama-3.1-8B-Instruct and Qwen2.5-72B-Instruct (or equivalents with verified 128K windows). Context lengths $L \in \{4\text{K}, 16\text{K}, 64\text{K}, 128\text{K}\}$. Task: RULER multi-key NIAH with **needle count, distractor count, and query type held constant** across $L$ — only filler tokens vary. 500 samples per cell, needle positions stratified uniformly over depth. Cost estimate: ~800 A100-hours.

**Arms.**
- **Control 1 (accuracy ceiling):** fp16 KV cache.
- **Control 2 (matched-memory):** fp16 cache with SnapKV/H2O eviction to the same byte count as the 4-bit arm. This separates "fewer bits per entry" from "fewer bytes" — the single most-omitted control in this literature.
- **Treatment:** KIVI-2, KVQuant-3, RTN-4, QuaRot-4, all reported at $b_{\text{eff}}$, not nominal $b$.

**Confound handling.** Report accuracy on a **difficulty-matched, length-normalized** scale: for each arm, fit the length $L^*_{\text{arm}}$ at which accuracy crosses 80%, and report $\log_2(L^*_{\text{fp16}} / L^*_{\text{arm}})$ — an *effective context halving factor*, which is comparable across $L$ where raw differences are not.

**The deciding number.** The effective context halving factor of the 4-bit arm. If it is $\le 0.15$ (under ~11% shrinkage of usable context), bit-width chosen at 4K transfers and the problem is closed in the negative. If it is $\ge 0.5$ (usable context cut by $\ge 30\%$), quantization sensitivity is length-dependent and bit allocation must be a function of $L$. The interval between is the honest "unresolved" verdict and calls for the 72B replication.

## 9. Key References

- **[Foundational]** Sheng et al. *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU.* ICML, 2023. — arXiv:2303.06865
- **[Foundational]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[SOTA]** Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML, 2024. — arXiv:2402.02750
- **[SOTA]** Hooper, Kim, Mohammadzadeh, Mahoney, Shao, Keutzer, Gholami. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS, 2024. — arXiv:2401.18079
- **[SOTA]** Ashkboos, Mohtashami, Croci, Li, Jaggi, Alistarh, Hoefler, Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[Mechanism]** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[Evaluation]** Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Evaluation]** Yuan et al. *KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches.* Findings of EMNLP, 2024.
- **[Evaluation]** Li et al. *SCBench: A KV Cache-Centric Analysis of Long-Context Methods.* ICLR, 2025.
- **[Theory]** Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330
- **[Theory]** Gao, Pavel. *On the Properties of the Softmax Function with Application in Game Theory and Reinforcement Learning.* 2017. — arXiv:1704.00805
- **[Eviction baseline]** Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS, 2023. — arXiv:2306.14048

## 10. Worked Example

Llama-3-8B: $d_h = 128$, 8 KV heads (GQA), 32 layers. Per token, the cache holds $2 \times 32 \times 8 \times 128 = 65{,}536$ values. At fp16 that is 128 KB/token, so 128K tokens is **16 GB** — more than the weights. At nominal 2 bits with $g=32$ and fp16 metadata, $b_{\text{eff}}=3.0$, giving 24 KB/token and 3.0 GB. The advertised "8× saving" is 5.3×.

Now the obstruction. Take one head at $L=128\text{K}$ with a single correct key $k^\star$ and score margin $s^\star - \max_{j \ne \star} s_j = 1.2$ (a typical retrieval margin in nats). Per-token 4-bit quantization on a key row whose largest channel is $|k_{\max}| = 40$ and whose median is $0.4$ gives $\sigma = 80/15 \approx 5.3$, so per-channel error is uniform on $\pm 2.7$ — **six times the median channel magnitude**. Score error is $\approx \|q\|_2 \cdot 2.7 \sqrt{d_h/3} / \sqrt{d_h} = 1.56\,\|q\|_2$ per key in standard deviation. With $\|q\|_2 \approx 1$ and $L = 128\text{K}$ distractors, the maximum of $128{,}000$ such perturbations is roughly $1.56\sqrt{2\ln(1.28\times10^5)} \approx 7.6$ — far above the 1.2 margin. At $L = 4\text{K}$ the same calculation gives $\approx 6.3$: also above the margin.

So the extreme-value term grows only as $\sqrt{\log L}$ — from 6.3 to 7.6, a 20% increase over a 32× length increase. The theory predicts a *weak* length effect, and the practical difference between 4K and 128K is dominated instead by whether per-channel grouping was used at all (which removes the outlier entirely, since the outlier channel gets its own scale). **This is the obstruction made visible:** the length dependence that the problem is named for is real but second-order and buried under a first-order grouping choice and a declining fp16 baseline. Any experiment that does not fix the grouping scheme and normalize against the fp16 length curve will measure the wrong effect and report it with confidence.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*