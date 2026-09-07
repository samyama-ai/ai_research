---
id: 12-quantization-compression/calibration-set-theory
title: "Calibration Set Size and Composition Theory"
topic: 12-quantization-compression
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Set Size and Composition Theory

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/calibration-set-theory` · **Status:** methodologically-blocked

## 1. Problem Statement

One-shot post-training compression methods (GPTQ, AWQ, SparseGPT, Wanda, SmoothQuant, QuIP#) fit their quantization or pruning decisions to a small **calibration set**: typically 128 sequences of 2048 tokens scraped from C4, WikiText-2, or the Pile. That choice is folklore. No published method derives it.

The problem has three variants with different difficulty:

- **Measurement.** Given a compression method $\mathcal{A}$, a target model, and a downstream evaluation suite, how much of the observed post-compression quality variance is attributable to the calibration set (size $n$, source distribution $\mathcal{D}_c$, sequence length $L$, sampling seed)? Solving this means a variance decomposition that separates calibration effects from quantization noise, evaluation noise, and kernel/dtype nondeterminism.
- **Method.** Given a compute budget of $n$ forward passes, choose $\mathcal{D}_c$ and the sample selection rule that minimizes downstream loss on a *deployment* distribution $\mathcal{D}_t$ that is not the calibration distribution. Solving this means a selection algorithm that beats uniform-random C4 sampling by a margin larger than seed variance, across model families.
- **Theory.** Prove a bound of the form: with $n$ calibration sequences drawn from $\mathcal{D}_c$, the excess loss of the compressed model on $\mathcal{D}_t$ is at most $f(n, d, \kappa, \mathrm{TV}(\mathcal{D}_c,\mathcal{D}_t))$ with high probability. Solving this means a non-vacuous sample-complexity theorem for layer-wise compression, currently absent.

## 2. Formal Setting

Let $W_\ell \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be layer $\ell$'s weights and $X_\ell \in \mathbb{R}^{d_{\text{in}} \times m}$ the layer's inputs over $m = nL$ calibration tokens. Layer-wise compression minimizes the **local reconstruction error**

$$\widehat{W}_\ell = \arg\min_{W \in \mathcal{C}} \; \frac{1}{m}\lVert W X_\ell - W_\ell X_\ell \rVert_F^2 ,$$

with $\mathcal{C}$ the quantization grid or sparsity set. The solution depends on $X_\ell$ only through the empirical second-moment matrix

$$\widehat{H}_\ell = \tfrac{1}{m} X_\ell X_\ell^\top \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}},$$

measured in practice as a running sum accumulated in fp32 over calibration forward passes, plus a damping term $\lambda \,\mathrm{tr}(\widehat H_\ell)/d_{\text{in}} \cdot I$ (GPTQ default $\lambda = 0.01$) added because $\widehat H_\ell$ is ill-conditioned or singular.

Quantities as measured:

- **$n$**: number of calibration sequences (GPTQ/AWQ/SparseGPT/Wanda default $n = 128$).
- **$L$**: sequence length, default 2048 tokens; $m = nL \approx 2.6\times10^5$ tokens.
- **Effective rank** $r_{\text{eff}}(\widehat H_\ell) = \mathrm{tr}(\widehat H_\ell)/\lVert \widehat H_\ell\rVert_2$ — the number of activation directions the estimate actually resolves.
- **Deployment gap**: the target is $\mathbb{E}_{x\sim\mathcal{D}_t}[\mathcal{L}(\widehat\theta, x)] - \mathbb{E}_{x\sim\mathcal{D}_t}[\mathcal{L}(\theta, x)]$, measured as a perplexity or task-accuracy delta, not as $\lVert \widehat W X - W X\rVert_F$.
- **Seed variance** $\sigma^2_{\text{seed}}$: variance of the deployment gap over $K$ independent draws of the calibration set at fixed $n$. This is the noise floor against which any composition claim must be judged.

Assumptions the standard analysis rests on, and their status:

1. *i.i.d. token activations.* Violated — tokens within a sequence are strongly autocorrelated, so the effective sample size is far below $m$.
2. *Sub-Gaussian activations.* Violated — LLM activations have massive outlier channels (Dettmers et al., 2022; Sun et al., 2024, "massive activations"), so covariance concentration rates for sub-Gaussian vectors do not apply.
3. *Layer-wise error decomposition.* Assumed additive across layers; the true objective is end-to-end and errors compound nonlinearly.
4. *$\mathcal{D}_c \approx \mathcal{D}_t$.* Violated by construction — models are calibrated on web text and deployed on chat, code, and reasoning traffic.

## 3. State of the Art

**Established.** GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023) and SparseGPT (Frantar & Alistarh, ICML 2023) show layer-wise Hessian-based compression works at 175B scale with 128 sequences; the *value* 128 is stated as a setting, not derived or ablated against a criterion. AWQ (Lin et al., MLSys 2024) reports robustness to calibration source, showing smaller degradation than GPTQ when switching Pile→other corpora — an ablation, at 7B–70B, on perplexity. Wanda (Sun, Liu, Bansal, Kolter, ICLR 2024) shows an activation-norm pruning criterion is nearly insensitive to $n$ down to a handful of sequences, which is real evidence that *criterion choice* dominates *calibration size*.

**Claimed but under-ablated.** Williams & Aletras (ACL 2024) report that calibration source materially changes downstream task scores for pruning and quantization at 7B scale, but the study covers few seeds, so how much of the reported spread is $\sigma_{\text{seed}}$ is not separated. Papers arguing for "calibration data similar to pretraining data" (e.g. Ji et al., 2024, on pruning) report gains that are within a few tenths of perplexity — the same order as seed noise in independently reported runs.

**Benchmark-number-only results.** Nearly all published calibration comparisons are single WikiText-2 perplexity numbers at one seed. Perplexity on WikiText-2 with a WikiText-2-adjacent calibration set is a leakage-shaped measurement, and the field's headline table is largely this.

**Theory SOTA.** Optimal Brain Compression (Frantar & Alistarh, NeurIPS 2022) gives exact solutions of the layer-wise problem given the *true* $H$. No published result bounds the loss from using $\widehat H$ instead of $H$ under realistic heavy-tailed activations.

## 4. What Is Known

- 128 sequences $\times$ 2048 tokens $= 262{,}144$ token samples for a $d_{\text{in}} = 4096$ covariance (Llama-2-7B). Naive sub-Gaussian concentration would need $n \gtrsim d/\epsilon^2$; $262{,}144 \gg 4096$, so under the textbook assumption the Hessian is over-determined by ~64$\times$. Outcomes nonetheless vary with the calibration draw. The assumption, not the sample count, is the failure.
- GPTQ damping $\lambda=0.01$ is required for numerical stability at 7B–70B; without it Cholesky factorization of $\widehat H$ fails on some layers. This is direct evidence that $\widehat H$ is rank-deficient in practice despite $m \gg d$ — measured $r_{\text{eff}}$ in attention projections is typically orders of magnitude below $d_{\text{in}}$.
- Calibration sensitivity scales with compression aggressiveness: at 4-bit weight-only, reported cross-corpus perplexity spreads on Llama-2-7B are ~0.05–0.2 ppl; at 3-bit and at 2:4 sparsity the same swaps produce ~0.5–2+ ppl. Sensitivity is a function of the bit budget, not a constant of the method.
- Activation-aware methods that use only per-channel magnitude statistics (AWQ, Wanda) are measurably less calibration-sensitive than second-order methods (GPTQ, SparseGPT), at 7B–13B.
- Jaiswal et al. (ICLR 2024) show perplexity-preserving compression can lose substantial accuracy on reasoning and generation tasks — so any calibration study measured only in perplexity is measuring the wrong endpoint.

## 5. What Is Not Known

- **Theoretically open.** No sample-complexity bound for layer-wise compression under heavy-tailed, autocorrelated activations. No result connecting $\mathrm{TV}(\mathcal{D}_c,\mathcal{D}_t)$ or $r_{\text{eff}}(\widehat H)$ to the end-to-end deployment gap. Whether the layer-wise proxy is even a consistent surrogate for end-to-end loss as $n\to\infty$ is unproven.
- **Empirically open.** The seed-variance floor $\sigma_{\text{seed}}$ at fixed $n$ has never been published at $K \geq 20$ for any frontier-scale model. Without it, no composition claim in the literature is falsifiable.
- **Methodologically blocked** (the dominant blocker). "Calibration set quality" has no measurement that is independent of the evaluation set. Perplexity on a corpus related to the calibration corpus conflates fit with leakage; downstream benchmarks have per-task noise of 1–2 accuracy points at 7B, larger than most reported calibration effects. Until a calibration-quality metric is defined that is (a) computable before compression and (b) predictive of a held-out deployment gap, the field cannot rank calibration sets at all.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**.

Confounding: the observed quantity is $\Delta = g(\text{calibration}) + \epsilon_{\text{eval}} + \epsilon_{\text{seed}} + \epsilon_{\text{kernel}}$. Reported calibration effects at 4-bit (~0.1 ppl) are smaller than $\epsilon_{\text{eval}}$ on typical few-thousand-token evaluation splits and comparable to $\epsilon_{\text{seed}}$. Papers report $\Delta$ at $K=1$ and attribute it entirely to $g$.

Non-identifiability: many calibration sets induce nearly identical $\widehat H$ in the top-$r_{\text{eff}}$ subspace and differ only in tail directions with eigenvalues below the damping floor $\lambda\,\mathrm{tr}(\widehat H)/d$. Those differences are erased by the algorithm. Two calibration sets that differ in the erased subspace are behaviourally identical; two that differ in an outlier channel's scale are not. No published statistic distinguishes these cases in advance.

Compute is a secondary obstruction: a properly powered study is $K$ seeds $\times$ $|\mathcal{D}_c|$ sources $\times$ $|n|$ sizes $\times$ model scales $\times$ full evaluation — hundreds of compression runs and thousands of evaluation runs per model family.

## 7. Current Research (as of 2026)

- Calibration-source ablations as a standard section in quantization papers, following Williams & Aletras (ACL 2024) and the "beware of calibration data" line for pruning. Mostly at 7B, mostly perplexity.
- Rotation-based methods (QuIP#, Tseng et al., ICML 2024; QuaRot, Ashkboos et al., NeurIPS 2024) reduce calibration dependence by making activations more isotropic before quantization. This is the most credible practical answer so far: change the problem so $\widehat H$ matters less. *(frontier — verify)* Whether incoherence processing removes calibration sensitivity or merely shifts it to the rotation-fitting stage is untested at $K\geq 10$ seeds.
- Task- and domain-matched calibration for deployment-specific quantization (code models, multilingual). *(frontier — verify)* Vendor reports claim gains; independent replications at controlled seed counts are not published.
- Data-free and synthetic calibration, descending from Nagel et al. (ICCV 2019). Revived for LLMs using model-generated text. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Is the calibration *source* effect larger than the calibration *seed* effect?

**Scale.** Llama-3-8B and Qwen2.5-14B, GPTQ 4-bit and 3-bit group-128, plus SparseGPT 2:4. Fixed $n=128$, $L=2048$.

**Arms.**
- *Control (the arm that is always missing):* $K=25$ independent uniform-random draws of 128 sequences from a **single** corpus (C4), same source, different seeds. This yields $\hat\sigma_{\text{seed}}$.
- *Treatment:* $K=25$ draws each from 4 other sources — Pile, GitHub code, a chat/instruct corpus, and Wikipedia. 125 compression runs per model per method.

**Evaluation.** A held-out deployment mixture never used for calibration: 5 tasks (GSM8K, HumanEval, MMLU, ARC-Challenge, a long-form generation scored by exact-match subtasks), plus perplexity on a held-out corpus disjoint from all five calibration sources.

**Deciding number.** The variance ratio
$$F = \frac{\hat\sigma^2_{\text{between-source}}}{\hat\sigma^2_{\text{seed}}}.$$
If $F < 2$ at 3-bit, calibration composition is folklore and the field should stop reporting single-seed source comparisons. If $F > 5$, composition is real and a selection theory is worth building; report the effect size in accuracy points alongside it. Budget: ~250 compression runs (minutes each) and ~1{,}250 evaluation runs — roughly 2{,}000 A100-hours, within a single academic cluster's reach.

## 9. Key References

- **[Foundational]** Elias Frantar, Sidak Pal Singh, Dan Alistarh. *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning.* NeurIPS 2022. — arXiv:2208.11580
- **[Foundational]** Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort. *Up or Down? Adaptive Rounding for Post-Training Quantization.* ICML 2020. — arXiv:2004.10568
- **[SOTA]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. — arXiv:2210.17323
- **[SOTA]** Elias Frantar, Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML 2023. — arXiv:2301.00774
- **[SOTA]** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys 2024. — arXiv:2306.00978
- **[SOTA]** Mingjie Sun, Zhuang Liu, Anna Bansal, J. Zico Kolter. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR 2024. — arXiv:2306.11695
- **[SOTA]** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. — arXiv:2402.04396
- **[Directly on-problem]** Miles Williams, Nikolaos Aletras. *On the Impact of Calibration Data in Post-training Quantization and Pruning.* ACL 2024. — arXiv:2311.09755
- **[Evaluation critique]** Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, Yinfei Yang. *Compressing LLMs: The Truth is Rarely Pure and Never Simple.* ICLR 2024. — arXiv:2310.01382
- **[Outliers]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. — arXiv:2208.07339
- **[Survey]** Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang. *A Survey on Model Compression for Large Language Models.* TACL, 2024. — arXiv:2308.07633

## 10. Worked Example

Take Llama-2-7B, layer 10, the `q_proj` matrix, $d_{\text{in}} = 4096$. Run GPTQ's calibration pass with $n = 128$, $L = 2048$, so $m = 262{,}144$ token vectors.

**Step 1 — nominal sufficiency.** For sub-Gaussian $x$, $\lVert \widehat H - H\rVert_2 / \lVert H \rVert_2 \lesssim \sqrt{d/m} = \sqrt{4096/262{,}144} = 0.125$. A 12% relative spectral error looks acceptable, and this is the implicit justification for $n=128$.

**Step 2 — where it breaks.** Measured LLM activations are not sub-Gaussian. A handful of channels carry magnitudes 20–100$\times$ the median (Dettmers et al., 2022). The correct rate for heavy tails scales with the fourth-moment ratio, not $d/m$; a single channel with kurtosis $\sim 10^3$ inflates the required $m$ by the same factor. That pushes the requirement from $2.6\times10^5$ tokens to $\sim\!10^8$ — three orders of magnitude past the default. Simultaneously, $r_{\text{eff}} = \mathrm{tr}(\widehat H)/\lVert\widehat H\rVert_2$ in such layers is typically in the tens-to-low-hundreds, not 4096: most directions carry almost no energy.

**Step 3 — the algorithm hides both facts.** GPTQ adds $\lambda\,\mathrm{tr}(\widehat H)/d \cdot I$ with $\lambda = 0.01$. Every eigendirection with $\mu_i < 0.01\,\bar\mu$ is dominated by damping and its quantization order and error-compensation term become calibration-independent. So of the 4096 directions, a few hundred are estimated from an effective sample far smaller than $m$ (under-determined, calibration-sensitive), and the rest are erased by damping (calibration-irrelevant).

**Step 4 — the obstruction, made visible.** Draw two calibration sets, $A$ from C4 and $B$ from GitHub code. Quantize to 4-bit. Suppose WikiText-2 perplexity differs by 0.08. That number cannot be interpreted: the seed-to-seed spread at fixed source is unpublished and plausibly of the same magnitude, WikiText-2 is closer in distribution to $A$ than to $B$, and the layer-wise error $\lVert\widehat W X - WX\rVert_F$ may be *lower* for $B$ while perplexity is worse — the proxy and the endpoint are not monotonically related. Three distinct confounds, one number, no way to separate them. That is what "methodologically blocked" means here: the experiment in Section 8 is cheap, but until the control arm exists, every calibration result in the literature is an unlabelled mixture of signal and seed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*