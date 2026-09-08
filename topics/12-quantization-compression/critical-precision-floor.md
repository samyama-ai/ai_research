---
id: 12-quantization-compression/critical-precision-floor
title: "Existence of a Critical Precision Floor"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Existence of a Critical Precision Floor

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/critical-precision-floor` · **Status:** open

## 1. Problem Statement

Is there a bit-width $P^\star > 0$ below which a neural language model cannot reach a given loss *at any parameter count*, or does every precision above 1 bit buy back capability by adding parameters?

Three variants, routinely conflated:

- **Measurement variant.** Given a family of models trained or quantized at precision $P$, estimate the loss-vs-bits Pareto frontier and test whether it has a vertical asymptote in $P$. Requires a definition of "same model at different precision" that is not confounded by the quantization algorithm.
- **Method variant.** Does *some* training or quantization procedure attain FP16-equivalent loss at $P$ bits per weight, for each $P$? A negative result here is a statement about the methods tried, not about $P$.
- **Theory variant.** For a fixed architecture class and precision-$P$ weight alphabet, is there a task family (or a loss target under a fixed data distribution) that is unreachable at $P$ bits regardless of width and depth? Here $P^\star$ would be a genuine representational threshold.

A solution to the theory variant is a proof: either a separation theorem showing a target loss is unattainable below $P^\star$ at any $N$, or a universality theorem showing 1-bit (or ternary) weights suffice up to a constant-factor blowup in $N$. A solution to the measurement variant is a fitted frontier with error bars that excludes or confirms divergence as $P \downarrow P^\star$.

## 2. Formal Setting

Model $f_\theta$, $\theta \in \mathbb{R}^N$. A precision-$P$ weight representation is a codebook $\mathcal{C}_P \subset \mathbb{R}$ with $|\mathcal{C}_P| \le 2^P$ per weight (per-group scales excluded from the count, then accounted separately), plus a quantizer $Q_P: \mathbb{R}^N \to \mathcal{C}_P^N$.

**Measured quantities.**

- **Loss** $L = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})$, in nats/token, on a held-out corpus fixed across all precisions. Downstream accuracy is a secondary readout; it saturates and is not a substitute.
- **Total bits** $B = N \cdot P + N/g \cdot P_s$, where $g$ is the group size and $P_s$ the scale precision. Reporting "4-bit" while carrying FP16 scales at $g=64$ is really $4.25$ bits; the floor question is decided at the third decimal place of $B/N$, so this bookkeeping is load-bearing.
- **Effective parameters.** Kumar et al. (2024) fit
$$L(N, D, P) \approx A\,[N_{\text{eff}}(P)]^{-\alpha} + B D^{-\beta} + E, \qquad N_{\text{eff}}(P) = N\left(1 - e^{-P/\gamma}\right),$$
with $\gamma$ a fitted decay constant of order one bit. Under this form, $N_{\text{eff}} \to 0$ only as $P \to 0$: **the fitted family assumes no floor**. Testing for $P^\star$ means fitting a strictly more general family, e.g. $N_{\text{eff}}(P) = N\,(1 - e^{-(P - P^\star)_+/\gamma})$, and asking whether $\hat{P}^\star$ is significantly $> 0$.

**Assumptions, and which are violated.**

1. *A single scalar $P$ describes the model.* Violated everywhere: activations, KV cache, and optimizer state carry separate precisions, and outlier channels are near-universally kept in higher precision (LLM.int8(), 2022).
2. *Loss is a smooth function of $P$.* Violated by outlier-driven cliffs at 8-bit activations in models above ~6.7B parameters.
3. *The quantizer is optimal.* Never true. Every empirical floor is an upper bound on the true floor, confounded with quantizer quality.
4. *Precision is independent of data budget.* Known false: post-training quantization damage grows with tokens-per-parameter $D/N$ (Kumar et al., 2024).

## 3. State of the Art

**Empirical SOTA (established).** Post-training quantization to 4 bits with second-order or activation-aware correction is near-lossless at 7B–70B: GPTQ (Frantar et al., ICLR 2023), AWQ (Lin et al., MLSys 2024), SmoothQuant (Xiao et al., ICML 2023) for W8A8. Below 4 bits, vector/lattice quantization dominates scalar rounding: QuIP# (Tseng et al., ICML 2024) and AQLM (Egiazarian et al., ICML 2024) give usable 2-bit Llama-2-70B — this is the strongest evidence *against* a floor near 3 bits.

**Quantization-aware training SOTA.** BitNet b1.58 (Ma et al., 2024) trains ternary weights $\{-1,0,+1\}$ ($\approx 1.58$ bits) from scratch and reports parity with an FP16 baseline in perplexity and downstream accuracy at 3B parameters and 100B tokens. The Spectra suite (Kaushal et al., 2024) trains ternary models to 3.9B parameters over 300B tokens and reports the same qualitative result.

**Claimed but unablated.** BitNet parity is a benchmark comparison against a *separately trained* FP16 model at matched $N$, not a matched-compute or matched-total-bits control, and the token budget (~33 tokens/param) is far below the $D/N$ regime where quantization damage is known to grow. Reports of "1-bit LLMs" are ternary plus FP16 activations, layer-norms, and embeddings; the end-to-end bits-per-weight is materially above 1.58.

**Theory SOTA.** No separation theorem for bit-width in transformers. The nearest results are classical: any continuous function is approximable by networks with weights from a finite set given enough width, and the strong lottery-ticket line (Malach et al., ICML 2020; Pensia et al., NeurIPS 2020) shows random $\pm 1$-style subnetworks approximate arbitrary targets with polylogarithmic width overhead. These *suggest* no floor above 1 bit, but say nothing about trainability or about loss on a fixed data distribution at fixed compute.

## 4. What Is Known

- **4 bits is Pareto-optimal for inference bits.** Dettmers & Zettlemoyer (ICML 2023) swept 35k+ zero-shot evaluations across 19M–176B parameters and found 4-bit weights maximize zero-shot accuracy per total model bit; 3-bit was worse at every scale tested. This used round-to-nearest-style PTQ, so it dates the method, not the floor — QuIP#/AQLM later moved the 2-bit point substantially.
- **Damage scales with data.** Kumar et al. (2024), fitting over 465 pretraining runs up to 1.7B parameters and 26B tokens, find PTQ degradation increasing in $D/N$, to the point of being *worse* for more-trained models. Their fits put compute-optimal *training* precision around 7–8 bits, with both 16-bit and sub-4-bit training predicted suboptimal.
- **Outliers, not bit-width alone, drive 8-bit failure.** Emergent activation outliers appear around 6.7B parameters and break naive INT8 inference; isolating $\sim 0.1\%$ of dimensions in FP16 restores full accuracy (Dettmers et al., NeurIPS 2022).
- **Ternary QAT reaches FP16 parity at 3B/100B tokens** (BitNet b1.58), replicated in spirit by Spectra at 3.9B/300B.
- **2-bit PTQ is no longer catastrophic.** QuIP# and AQLM report Llama-2-70B at ~2 bits within a small perplexity gap of FP16 — a regime that round-to-nearest destroys.

## 5. What Is Not Known

- **Theoretically open.** No theorem states, for any $P \ge 1$, that a target cross-entropy on a fixed distribution is unreachable at precision $P$ for all $N$. Equally, no theorem rules it out for transformers under gradient training. Both directions are open.
- **Empirically open.** Whether ternary parity survives at $D/N \gtrsim 1000$ tokens/param and $N \gtrsim 30$B. Every ternary parity claim lives at $D/N \le 100$; every scaling-law fit predicting low-precision failure lives at $N \le 1.7$B. The two literatures do not overlap in the plane where the answer lives. The experiment is runnable — it costs a frontier-lab pretraining budget.
- **Methodologically blocked.** "Precision of a model" is not a well-defined scalar while embeddings, activations, layer-norms, KV cache, and per-group scales are excluded from the count by convention. Until $B$ is reported as a single audited total-bits number, floor estimates from different papers are not comparable.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability between the floor and the quantizer**. Any observation of the form "loss diverges below $P$ bits" is jointly a claim about the representation and about the search procedure that found the weights. The 2-bit regime demonstrates the failure mode concretely: RTN said 2 bits was unusable in 2022; QuIP# said it was usable in 2024, with the same models and the same bit budget. No experiment can separate "no $P$-bit network achieves $L$" from "our optimizer did not find one", because the search space at $P=2$ over $N=7\times10^9$ weights has $4^{7\times10^9}$ points and no known certificate of optimality.

Second obstruction: **compute cost of the decisive regime**. The floor, if it exists, is a statement about the $N \to \infty$ limit at fixed $L$. Distinguishing $N_{\text{eff}} = N(1-e^{-P/\gamma})$ from $N(1-e^{-(P-P^\star)_+/\gamma})$ with $\hat{P}^\star \approx 1.2$ requires precise loss measurements at $P \in \{1, 1.58, 2, 3, 4\}$ across at least a decade of $N$ — a grid of full pretraining runs, not fine-tunes.

## 7. Current Research (as of 2026)

- **Precision-aware scaling laws.** Extending Kumar et al. to larger $N$ and to QAT rather than PTQ; the open question is whether $\gamma$ is universal or architecture-dependent *(frontier — verify)*.
- **Native low-precision pretraining.** Microsoft Research's BitNet line (bitnet.cpp kernels, ternary attention), and academic ternary suites (Spectra, Nolano/Mila-adjacent groups).
- **Lattice and incoherence-processing PTQ.** Cornell (Tseng, De Sa) on QuIP#/QTIP-style trellis codes; Yandex Research/IST Austria on AQLM/PV-tuning — pushing 2 bits toward losslessness and thereby lowering any empirical floor estimate.
- **Hardware co-design.** NVFP4/MXFP4 microscaling block formats in Blackwell-class hardware make 4-bit *training* economically testable for the first time, which is what converts the empirical variant from unrunnable to expensive *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Pretrain a fixed architecture at $N \in \{0.3, 1, 3, 8\}$B, each at $D = 1$T tokens (so $D/N$ spans 125 to 3300), at weight precisions $P \in \{1.58, 2, 3, 4, 8\}$ via QAT with identical data order and hyperparameter sweeps per cell. 20 runs. Activations fixed at BF16 in *all* arms so the single manipulated variable is weight bit-width; report audited total bits $B$ including scales and embeddings.

**Control arm.** BF16 weights at each $N$, same data, same tuning budget — plus a *parameter-matched-bits* control, i.e. a BF16 model with $N' = N \cdot P/16$ so the comparison is at equal $B$, not equal $N$.

**Deciding number.** Fit $L(N,D,P) = A[N(1-e^{-(P-P^\star)_+/\gamma})]^{-\alpha} + BD^{-\beta} + E$ and report the bootstrap 95% confidence interval on $\hat{P}^\star$. If the interval excludes 0, a floor exists and its location is measured. If it contains 0 and its upper end is below 1.0 bits, the floor hypothesis is dead above ternary. Secondary readout: the sign of $\partial(\Delta L_{P\text{ vs BF16}})/\partial \log(D/N)$ at $P=1.58$ — if positive and growing at $D/N = 3300$, ternary parity is an artifact of undertraining.

## 9. Key References

- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[Foundational]** Tim Dettmers, Luke Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330
- **[SOTA]** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* 2024. — arXiv:2402.17764
- **[SOTA]** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024. — arXiv:2402.04396
- **[SOTA]** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh. *Extreme Compression of Large Language Models via Additive Quantization.* ICML, 2024. — arXiv:2401.06118
- **[Method]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[Method]** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[Theory]** Eran Malach, Gilad Yehudai, Shai Shalev-Shwartz, Ohad Shamir. *Proving the Lottery Ticket Hypothesis: Pruning is All You Need.* ICML, 2020. — arXiv:2002.00585
- **[Survey]** Ayush Kaushal, Tejas Vaidhya, et al. *Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language Models.* 2024. — arXiv:2407.12327

## 10. Worked Example

Take Llama-2-70B, $N = 69$B weights, FP16 WikiText-2 perplexity $3.12$.

| Arm | Bits/weight (audited) | Total weight bits | Reported PPL |
|---|---|---|---|
| FP16 | 16 | 1.10 Tb | 3.12 |
| GPTQ 4-bit, $g{=}128$ | 4.13 | 0.285 Tb | ~3.2 |
| RTN 2-bit | 2.13 | 0.147 Tb | diverges (>10) |
| QuIP#/AQLM ~2-bit | ~2.1–2.3 | ~0.15 Tb | ~3.6–4.2 |

Read the RTN row alone and the conclusion is "the floor is between 2 and 3 bits". Add the QuIP# row — same architecture, same weights, same bit budget, different quantizer — and the loss gap shrinks by roughly an order of magnitude in excess perplexity. Nothing about the *representation* changed between rows 3 and 4. Only the search did.

Now do the counterfactual the floor question actually asks. At equal total bits, the 2-bit arm's 0.15 Tb also buys a BF16 model of $N' = 9.2$B parameters. Chinchilla-style fits put a well-trained 9B model near a WikiText-2 perplexity in the mid-3s — the *same band* as 2-bit-70B. So the equal-bits comparison at 2 bits is close to a tie, and the sign of the tie is within the error bar of the quantizer's quality.

That is the obstruction in one table: the empirical floor moves whenever someone builds a better quantizer, and the equal-bits control lands inside the interval the quantizer can move. No finite experiment settles a claim about all $P$-bit networks; it only reports the best one found so far.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*