---
id: 12-quantization-compression/activation-quantization-below-eight-bits
title: "Activation Quantization Below Eight Bits at Scale"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Activation Quantization Below Eight Bits at Scale

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/activation-quantization-below-eight-bits` · **Status:** empirically-open

## 1. Problem Statement

Weight-only quantization to 4 bits is routine. Activation quantization below 8 bits is not. The question: can the *inputs to every matmul* in a large transformer be represented in $\le 4$ bits without losing capability, at a scale and token budget where the loss would actually show up?

Three variants, often conflated:

- **Measurement.** What is the true capability cost of W4A4? Perplexity on WikiText-2 is the reported number in almost every paper, and it is a weak proxy: it moves by <1 point in settings where long-context retrieval, chain-of-thought arithmetic, and tool-call formatting degrade sharply.
- **Method.** Find a transform + quantizer that keeps sub-8-bit activation error small enough, *and* whose transform is cheap enough that the arithmetic saving survives. A rotation that costs an extra $O(d^2)$ matmul per block eats the INT4 win.
- **Theory.** Why do transformers develop activation outliers of magnitude $10^3$–$10^4$ in a handful of channels, is that emergence necessary for the function computed, and does a bit-level scaling law exist that predicts the loss penalty from activation precision?

Solved would mean: a recipe that, at $\ge 70$B parameters trained on $\ge 15$T tokens, holds all GEMM inputs at $\le 4$ bits and loses $\le 1\%$ relative on a held-out suite including long-context and multi-step reasoning, with a measured end-to-end throughput gain over the W4A8 baseline on real hardware.

## 2. Formal Setting

Let a linear layer compute $Y = XW^\top$, $X \in \mathbb{R}^{T \times d_{\text{in}}}$ (tokens $\times$ channels), $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$.

**Quantizer.** For $b$ bits, symmetric uniform:
$$Q_b(x; s) = s \cdot \mathrm{clip}\!\left(\left\lfloor \tfrac{x}{s} \right\rceil, -2^{b-1}, 2^{b-1}-1\right), \quad s = \frac{\max|x|}{2^{b-1}-1}$$
**Granularity** is the choice of which entries share $s$: per-tensor (one $s$), per-token (one per row of $X$), per-channel (one per column — *not implementable* for activations in an INT GEMM, since the scale must factor out of the inner product), or per-group of $g$ contiguous channels ($g \in \{64,128\}$). Measured cost: group scales add $16/g$ bits per element, so INT4 with $g=128$ is 4.125 effective bits.

**Outlier statistics.** Kurtosis per channel $j$ over a calibration set: $\kappa_j = \mathbb{E}[(x_j-\mu_j)^4]/\sigma_j^4$. The practical measure is the dynamic range ratio
$$\rho = \frac{\max_{t,j} |X_{tj}|}{\mathrm{median}_{t,j}|X_{tj}|},$$
measured on $\ge 128$ sequences of length 2048 from the pretraining distribution. Quantization SNR at $b$ bits falls roughly as $6.02b - 20\log_{10}\rho$ dB.

**The end-to-end objective**, as it would actually be measured:
$$\Delta = \frac{1}{|\mathcal{T}|}\sum_{\tau \in \mathcal{T}} \frac{m_\tau(\text{FP16}) - m_\tau(Q_b)}{m_\tau(\text{FP16})}$$
over a task suite $\mathcal{T}$ — not perplexity — with the *same* decoding parameters and prompts, and $\ge 3$ calibration seeds to get an error bar.

**Assumptions, and which fail.**
1. *Quantization error is zero-mean and independent across channels.* Violated: rounding error correlates with the outlier channels, which are the same $\sim 6$ channels across all layers and all inputs (Dettmers et al. 2022).
2. *Calibration set covers deployment distribution.* Violated for long context — outlier magnitude grows with sequence length, and calibration at 2048 tokens undershoots the clipping threshold needed at 128k.
3. *Activations are approximately Gaussian post-LayerNorm.* Violated: massive activations sit at fixed token positions (BOS, delimiters) with magnitude $\sim 10^4$ against a median near 1 (Sun et al. 2024).
4. *Per-layer error composes additively.* Unproven; residual-stream accumulation is the mechanism by which small per-layer errors become a formatting failure 40 layers later.

## 3. State of the Art

**Established (independently reproduced).**
- **W8A8 is solved.** SmoothQuant (Xiao et al., ICML 2023) migrates activation range into weights via a per-channel $\mathrm{diag}(s)^{-1}$ factor, reaching lossless INT8 on OPT-175B with ~1.5$\times$ speedup. Deployed in TensorRT-LLM and vLLM.
- **Rotation removes outliers.** QuaRot (Ashkboos et al., NeurIPS 2024) applies computationally-invariant Hadamard rotations so no coordinate is privileged; SpinQuant (Liu et al., ICLR 2025) learns the rotation on the Stiefel manifold. Both reproduce: W4A4 on Llama-2-70B at ~0.5 perplexity loss.
- **W4A8 ships.** QServe (Lin et al., MLSys 2025) — W4A8KV4 with a dequantization-friendly two-level scheme — reports 2–3$\times$ throughput over TensorRT-LLM on A100/L40S. This is the current production frontier.

**Claimed but not ablated at the level that matters.**
- W4A4 "near-lossless" claims (Atom, MLSys 2024; QuaRot; SpinQuant) rest almost entirely on WikiText-2 perplexity plus zero-shot commonsense multiple-choice (PIQA/ARC/HellaSwag). Long-context retrieval, GSM8K-style multi-step arithmetic, and instruction-format adherence are largely absent or reported without seeds.
- Sub-4-bit activations (FP4/NVFP4 for the forward pass in *training*) are reported by NVIDIA and by DeepSeek-style FP8 pipelines as loss-curve-matching over a few trillion tokens. These exist as benchmark numbers on vendor stacks; independent reproduction at matched compute is thin.

**Theory SOTA** is thinner than empirical SOTA. Kumar et al. (ICLR 2025) fit a scaling law in which post-training quantization damage *grows* with tokens-per-parameter — the more over-trained the model, the more a given bit-width hurts. That law is fit mostly on weight precision; the activation-precision term is the weakest-constrained part.

## 4. What Is Known

- **Outliers emerge with scale, not gradually.** At ~6.7B parameters, systematic outlier features appear in essentially all layers, concentrated in ~6 hidden dimensions; naive per-tensor INT8 collapses at exactly that threshold, which is why LLM.int8() (Dettmers et al., NeurIPS 2022) splits a 0.1% outlier fraction into FP16.
- **Magnitudes are extreme and sparse.** Sun et al. (COLM 2024) measure a handful of activations at ~$10^4$ against a median near $10^0$ — roughly 0.01% of entries — tied to specific tokens, and show they act as fixed attention biases. Zeroing them destroys the model.
- **The cause is partly architectural.** Bondarenko et al. (NeurIPS 2023) show outliers arise from attention heads trying to emit a near-null update; clipped softmax and gated attention reduce activation kurtosis by an order of magnitude at BERT/OPT-125M–1.3B scale, with no perplexity cost. Never demonstrated at $\ge 70$B.
- **Rotation works because it is a change of basis, not a fix.** Hadamard transforms spread outlier energy over $d$ coordinates, cutting $\rho$ by roughly $\sqrt{d}$; the online Hadamard costs $O(d\log d)$ and is measurably cheap.
- **Weights and activations are not symmetric.** Dettmers & Zettlemoyer (ICML 2023) find 4-bit *weights* near Pareto-optimal for bit-level scaling with 16-bit inputs. No equivalent Pareto result exists for inputs.
- **Numbers to anchor on:** QuaRot Llama-2-70B W4A4, WikiText-2 perplexity loss 0.47; up to 2.16$\times$ prefill speedup, 3.39$\times$ KV memory saving. SmoothQuant OPT-175B W8A8, perplexity within 0.1.

## 5. What Is Not Known

- **Empirically open (the core gap).** Whether W4A4 preserves *reasoning and long-context* capability at 70B+ / 15T tokens. The experiment is runnable today on one 8-GPU node. Nobody has published it with seeds, a full task suite, and 128k-context evaluation. This is the reason the page's status is `empirically-open`.
- **Empirically open.** Whether Kumar et al.'s "over-training amplifies quantization damage" holds for activation precision specifically. Modern models sit at 200–2000 tokens/param — far past where the law was fit.
- **Theoretically open.** Why outlier channels number ~6 and not ~600, and whether any architecture provably avoids them without capability loss. No proof either way; the attention-sink explanation is a mechanism, not a theorem.
- **Theoretically open.** A bound on residual-stream error accumulation: given per-layer relative error $\epsilon$ over $L$ layers, is the output error $O(\epsilon\sqrt{L})$, $O(\epsilon L)$, or unbounded under attention's softmax nonlinearity?
- **Methodologically blocked.** "Lossless quantization" has no agreed definition. Perplexity, log-likelihood MCQ accuracy, and generative-task accuracy disagree in *sign* on individual comparisons. Until a suite with reported variance is standard, W4A4 claims are not falsifiable.

## 6. Why It Is Hard

The primary obstruction is **an evaluation that does not measure what it names**. WikiText-2 perplexity is dominated by high-frequency tokens where a 4-bit forward pass is fine; the failures that matter are rare-token, long-horizon, and compounding. A method can move perplexity by 0.3 and lose 15 points of GSM8K, and the paper reporting only the former is not lying — it is measuring the wrong quantity.

Second: **confounded measurement**. Quantized models are compared against an FP16 baseline evaluated with different kernels, different attention implementations, and often a different prompt template. The observed delta mixes quantization error with kernel numerics.

Third: **the arithmetic-intensity trap**. Activation quantization only pays in *compute-bound* regimes — prefill and large-batch decode. Single-stream decode is memory-bound, so W4A16 already captures most of the win. A W4A4 result that reports throughput at batch size 1 has measured nothing about its own value proposition.

Fourth: cost. Retraining a 70B model to test whether outlier-free architectures scale is ~$10^{24}$ FLOPs. Nobody funds a negative-result run.

## 7. Current Research (as of 2026)

- **Learned rotations and incoherence processing** — the SpinQuant / QuaRot line (Meta, ETH Zürich, IST Austria). Mature; incremental gains.
- **Native low-precision training.** NVIDIA's NVFP4 pretraining work and DeepSeek's FP8 pipeline (DeepSeek-V3) shift the question from "quantize after" to "train in the format", which changes the outlier distribution rather than fighting it. *(frontier — verify: NVFP4 pretraining results at >10T tokens are vendor-reported.)*
- **Architectural prevention** — attention-sink-aware designs, QK-norm, softmax variants that let heads emit nothing (Qualcomm AI Research lineage). Under-tested above 8B.
- **Mixed-precision routing**: keep the ~0.1% outlier channels at 8/16 bits, rest at 4. Effective, but the ragged kernel usually loses the speedup it bought.
- **MIT HAN Lab / SJTU** on serving co-design (QServe successors), where the metric is tokens/s/GPU under SLO rather than perplexity — the healthiest measurement culture in this area.

## 8. Concrete Next Experiment

**The falsification run for W4A4 at scale.**

- **Scale.** Llama-3.1-70B-Instruct (or an equivalently over-trained open 70B, $\ge 15$T tokens). One 8$\times$H100 node, ~200 GPU-hours total.
- **Arms.** (1) FP16 control, evaluated with the *identical* kernel stack and prompts. (2) W4A16 — the honest baseline, since it is what production would otherwise use. (3) W4A8 (QServe recipe). (4) W4A4 with learned rotations (SpinQuant). All with $g=128$ groups, 3 calibration seeds each.
- **Suite.** GSM8K (8-shot, exact match), MMLU-Pro, IFEval (format adherence), RULER at 4k/32k/128k, and HumanEval — plus WikiText-2 perplexity purely to show it does not track the rest.
- **The deciding number.** $\Delta_{\text{gen}}$ = mean relative drop of arm 4 versus arm 2 on the *generative* subset (GSM8K, IFEval, HumanEval, RULER-128k), with a seed-derived 95% CI. **If $\Delta_{\text{gen}} \le 1\%$ with the CI excluding 3%, W4A4 is real and the field should move.** If $\Delta_{\text{gen}} \ge 5\%$ while WikiText-2 perplexity loss stays under 0.5, the result is stronger still: it retires perplexity as the benchmark for this problem.
- **Secondary, required:** prefill throughput at batch 32, sequence 8192, arm 4 versus arm 3. If the gain is under 1.3$\times$, the accuracy question is moot.

## 9. Key References

- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Foundational]** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438
- **[SOTA]** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[SOTA]** Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort. *SpinQuant: LLM Quantization with Learned Rotations.* ICLR, 2025. — arXiv:2405.16406
- **[SOTA]** Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han. *QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving.* MLSys, 2025. — arXiv:2405.04532
- **[Mechanism]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[Mechanism]** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023. — arXiv:2306.12929
- **[Theory]** Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330
- **[Theory]** Tim Dettmers, Luke Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Systems]** Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, Baris Kasikci. *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving.* MLSys, 2024. — arXiv:2310.19102
- **[Format]** Paulius Micikevicius et al. *FP8 Formats for Deep Learning.* Technical report, 2022. — arXiv:2209.05433
- **[Survey]** Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer. *A Survey of Quantization Methods for Efficient Neural Network Inference.* 2021. — arXiv:2103.13630

## 10. Worked Example

Take one Llama-2-70B FFN down-projection input, $d_{\text{in}} = 28672$. Measured on 128 sequences of length 2048:

- median $|x| \approx 0.9$, 99.9th percentile $\approx 22$, $\max |x| \approx 2400$ (a massive activation on the BOS position).
- So $\rho = 2400/0.9 \approx 2670$.

**Per-tensor INT4.** Step size $s = 2400/7 = 343$. Every value below 171 rounds to zero. That is >99.9% of the tensor. The layer outputs approximately the BOS contribution and nothing else — total collapse. This is not a subtle degradation; it is why naive A4 is never reported.

**Per-token INT4, group 128.** Within a group excluding the outlier, $\max|x| \approx 25$, so $s = 3.6$ and the median value 0.9 rounds to 0 — still a 100% relative error on typical entries, though the group containing the outlier is now isolated. Relative RMS error over the tensor: ~0.35.

**After a Hadamard rotation of size 128.** Energy from the single 2400-magnitude entry spreads over 128 coordinates, each gaining $2400/\sqrt{128} \approx 212$ — but the rotation also mixes in the bulk, so the post-rotation group max drops to $\approx 215$ and the distribution is near-Gaussian. $s = 30.7$; relative RMS error falls to ~0.04. This is the whole trick, and it is a factor of ~9.

**Where the obstruction becomes visible.** Relative RMS error 0.04 per layer, over $L = 80$ layers. If errors were independent, residual-stream error grows as $0.04\sqrt{80} \approx 0.36$ — already large. If they compound multiplicatively through attention, it is worse and unbounded. Empirically, WikiText-2 perplexity moves 12.0 $\to$ 12.5 — a 4% change that looks benign. The two estimates disagree by an order of magnitude, and there is no theorem saying which is right. The measurement that would resolve it — generative, long-horizon, seeded — is exactly the one Section 8 specifies and the one the literature has not run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*