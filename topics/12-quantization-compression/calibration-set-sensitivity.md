---
id: 12-quantization-compression/calibration-set-sensitivity
title: "Calibration Set Size and Distribution Sensitivity"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Set Size and Distribution Sensitivity

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/calibration-set-sensitivity` · **Status:** empirically-open

## 1. Problem Statement

One-shot post-training quantization (PTQ) and post-training pruning methods — GPTQ, AWQ, SmoothQuant, SparseGPT, Wanda, QuaRot/SpinQuant rotations, activation-aware scaling — all fit their compression parameters against a small **calibration set**: typically 128 sequences of 2048 tokens, drawn from C4, WikiText-2, or the Pile. That is roughly $2.6\times10^5$ tokens used to decide the fate of $10^{10}$ parameters.

The problem: **how much does the compressed model depend on which tokens were in that set, and how many are needed?**

Three variants, of very different difficulty:

- **Measurement.** Define and estimate the *sensitivity* of a compression pipeline to calibration draw — the variance of downstream quality across independent calibration samples, and its decay with set size $m$. Blocked mainly by evaluation noise, not by compute.
- **Method.** Build a compression procedure whose output quality is provably or empirically insensitive to the calibration draw, or a selection rule that picks a calibration set better than a random C4 draw for a *stated* deployment distribution.
- **Theory.** Give a generalization bound: for a layerwise reconstruction objective fit on $m$ samples, bound the excess end-to-end loss on a deployment distribution $\mathcal{D}_{\text{test}}$ as a function of $m$, the layer's activation covariance spectrum, and the shift between calibration and deployment distributions.

A solution to the measurement variant is a sensitivity curve with error bars. A solution to the theory variant is a bound that is non-vacuous at $m=128$.

## 2. Formal Setting

Let $W_\ell \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ be layer $\ell$'s weights and $X_\ell \in \mathbb{R}^{d_{\text{in}}\times N}$ the stacked layer inputs produced by running the calibration set through the model. GPTQ/SparseGPT-style methods solve, per layer,

$$\hat{W}_\ell = \arg\min_{W \in \mathcal{Q}} \; \big\| W X_\ell - W_\ell X_\ell \big\|_F^2 ,$$

with $\mathcal{Q}$ the quantization grid (or sparsity pattern). The solution depends on $X_\ell$ only through the second moment $H_\ell = X_\ell X_\ell^\top + \lambda I$, the layerwise Hessian of the reconstruction objective. **Measured as:** accumulate $H_\ell$ in fp32 over $m$ sequences of length $T$ ($N = mT$ columns), with damping $\lambda = 0.01 \cdot \frac{1}{d_{\text{in}}}\mathrm{tr}(H_\ell)$ — the GPTQ default. AWQ instead measures a per-channel activation magnitude $s_j = \frac{1}{N}\sum_i |X_{\ell,ji}|$ and searches a scaling exponent; SmoothQuant uses $\max_i |X_{\ell,ji}|$, a max statistic whose estimator variance does not shrink like $1/\sqrt{N}$.

Let $C \sim \mathcal{D}_{\text{cal}}^m$ be a calibration draw and $A(C)$ the compressed model. Define the **calibration sensitivity** of a metric $\mu$ (perplexity on a held-out corpus, or accuracy on task $t$) at size $m$:

$$\sigma^2_\mu(m) \;=\; \operatorname{Var}_{C \sim \mathcal{D}_{\text{cal}}^m}\big[\mu(A(C))\big], \qquad \Delta_\mu(\mathcal{D}_1,\mathcal{D}_2) \;=\; \mathbb{E}_{C\sim\mathcal{D}_1^m}[\mu(A(C))] - \mathbb{E}_{C\sim\mathcal{D}_2^m}[\mu(A(C))].$$

$\sigma_\mu$ is the **within-source** term (same corpus, different seed); $\Delta_\mu$ is the **cross-source** term (C4 vs WikiText vs code vs chat transcripts). **Measured as:** $K \geq 10$ independent draws per condition, sample variance of $\mu$, with a separate estimate of the metric's own evaluation noise $\sigma^2_{\text{eval}}$ (finite test set, decoding seed) subtracted — otherwise $\sigma_\mu^2$ is upward-biased.

Assumptions the standard pipeline rests on, and their status:

1. **Layerwise surrogate.** Minimizing $\|\Delta W X_\ell\|_F^2$ per layer minimizes end-to-end loss. *Violated*: errors compound across layers, and BRECQ (Li et al., ICLR 2021) showed block-wise reconstruction beats layer-wise precisely because the assumption fails.
2. **$H_\ell$ is well-estimated at $N = mT$.** With $d_{\text{in}} = 4096$ and $m=128$, $T=2048$, $N/d_{\text{in}} \approx 64$ — adequate for a bulk spectrum estimate, marginal for the tail eigendirections that quantization error is projected onto. Damping $\lambda$ silently regularizes the deficit.
3. **Tokens within a sequence are i.i.d. samples.** *Violated*: intra-document correlation means the effective sample size is far below $N$; nobody reports an effective $N$.
4. **$\mathcal{D}_{\text{cal}} \approx \mathcal{D}_{\text{test}}$.** *Violated by construction* for instruction-tuned models calibrated on C4 web text and deployed on chat, code, or long-context inputs.

## 3. State of the Art

**Established (ablated, reproduced):**

- GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023) fixed 128 C4 sequences as the field default and showed the method works at 4-bit for OPT-175B/BLOOM-176B. The *choice* of 128 was not derived; it was a compute-convenient default.
- Williams & Aletras (NAACL 2024, arXiv:2311.09755) is the reference study on this exact question: varying the calibration source across C4, WikiText, and others changes downstream task accuracy for GPTQ and SparseGPT, with effects that are larger for pruning than for quantization and larger at more aggressive rates.
- Wanda (Sun, Liu, Bansal, Kolter, ICLR 2024) reports that its magnitude$\times$activation-norm criterion is stable down to very few calibration sequences — a robustness *claim* that is well-supported for that specific statistic, and is evidence that sensitivity is method-dependent, not intrinsic to compression.

**Claimed but not fully ablated:**

- AWQ (Lin et al., MLSys 2024) claims lower calibration-distribution sensitivity than GPTQ because it fits one scale per channel rather than solving a Hessian system. Reported as perplexity deltas under a swapped calibration corpus; not reported as a variance over repeated draws, and not on instruction-following or code tasks.
- Rotation-based methods (QuaRot, Ashkboos et al., NeurIPS 2024; SpinQuant, Liu et al., 2024) reduce outlier-driven error and are argued to reduce calibration dependence. SpinQuant learns rotations on data, which reintroduces it. No published $\sigma_\mu(m)$ curve for either.
- Domain-matched calibration ("calibrate on your deployment data") is folklore in deployment guides. Existing measurements are single-seed benchmark numbers.

## 4. What Is Known

- **The default is 128 sequences $\times$ 2048 tokens $\approx 2.6\times10^5$ tokens**, from GPTQ; inherited by SparseGPT, AWQ, LLM Compressor, AutoGPTQ, and most published PTQ baselines. Scale: 7B–175B.
- **Perplexity understates compression damage.** Jaiswal et al., *Compressing LLMs: The Truth Is Rarely Pure and Never Simple* (ICLR 2024), show compressed models with near-identical WikiText perplexity diverge sharply on reasoning and knowledge-intensive tasks at 7B–13B. Any sensitivity measured only in perplexity is measured with the wrong instrument.
- **Sensitivity grows with aggressiveness.** Across GPTQ/SparseGPT studies, 4-bit weight-only quantization of a 7B model is far more forgiving of calibration choice than 2–3 bit or 50–60% unstructured sparsity, where calibration-source effects on task accuracy become several accuracy points rather than fractions of one.
- **Statistic type matters.** Methods whose fitted quantity is a *max* over activations (SmoothQuant-style outlier scaling) inherit an estimator whose variance does not shrink at the $1/\sqrt{N}$ rate of a mean or covariance; methods using second moments (GPTQ) or norms (Wanda) concentrate faster.
- **Format ranking is stable even where calibration is not.** Kurtic et al., *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization* (2025), find W8A8 and W4A16 recover near-baseline accuracy across a large evaluation suite at 8B–405B, which bounds how large calibration effects can be *at 4-bit and above*.

## 5. What Is Not Known

- **Empirically open.** No published $\sigma_\mu(m)$ curve: nobody has run $K\ge10$ independent calibration draws at each of $m \in \{8,32,128,512,2048\}$ for a fixed model and method, and reported variance on a task suite with evaluation noise subtracted. The experiment costs GPU-hours, not GPU-years. This is the central gap.
- **Empirically open.** Whether domain-matched calibration beats generic web text for a *stated* deployment distribution, measured as a mean shift larger than the within-source standard deviation. Everyone assumes yes; the effect has not been shown to exceed its own noise floor.
- **Methodologically blocked.** There is no agreed sensitivity metric. "Perplexity delta" is the wrong instrument (§4), and task-suite averages hide per-task swings that cancel. Until the field fixes a metric, cross-paper claims about robustness are not comparable.
- **Theoretically open.** No non-vacuous finite-sample bound linking $m$, the spectrum of $\mathbb{E}[X_\ell X_\ell^\top]$, and end-to-end loss under distribution shift. Classical OBS theory (Hassibi & Stork, NIPS 1992) assumes the exact Hessian; the finite-sample version is unwritten.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement against an unmeasured noise floor**. Compression quality is reported as a single number from a single calibration draw, and the standard task suites have their own sampling noise of roughly 1 accuracy point on 1000-item benchmarks ($\sqrt{p(1-p)/n} \approx 1.5\%$ at $p=0.6$, $n=1000$). Calibration effects at 4-bit sit at or below that floor, so most published "robust to calibration data" claims and most "domain-matched calibration helps" claims are both consistent with the same data. Separating them requires $K$ repeats per condition, which multiplies cost by $K$ for a result that reads as a null — a publication-incentive problem on top of a statistical one.

Secondary: **absent ground truth**. There is no defined deployment distribution for a general-purpose model, so "the right calibration set" has no referent. The problem is only well-posed once a deployment distribution is fixed, and papers that fix one lose generality.

## 7. Current Research (as of 2026)

- **Calibration-free and rotation-first quantization.** QuaRot/SpinQuant descendants aim to remove outliers structurally so that the data-fitted component matters less (ETH Zürich / IST Austria / Meta lines). *(frontier — verify)* claims of full calibration independence.
- **Calibration set selection and curation.** Selecting sequences by activation coverage or perplexity rather than uniform sampling, mostly in the pruning literature. Reported gains are typically sub-point and single-seed.
- **Standardized PTQ benchmarking.** LLMC / LLM-QBench (Gong et al., 2024) provides one harness across many PTQ methods, which is the infrastructure the missing variance study needs.
- **Long-context and multimodal calibration.** Calibrating with 2048-token sequences for a 128K-context model is an obvious distribution mismatch; the sensitivity has not been quantified. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B, one method family (GPTQ W4A16 and GPTQ W2A16 — the second to place the effect above the noise floor).

**Design.** $m \in \{8, 32, 128, 512\}$ sequences of 2048 tokens; $K = 10$ independent draws per $m$ from each of three sources: C4, a chat-transcript corpus, and a code corpus. That is $4 \times 10 \times 3 \times 2 = 240$ quantization runs; GPTQ on 8B is minutes on one A100, so under 100 GPU-hours including evaluation.

**Control arm.** The same evaluation run $K=10$ times on the *uncompressed* model with different evaluation seeds/subsamples, to estimate $\sigma_{\text{eval}}$. Without this arm the result is uninterpretable.

**Deciding number.** The ratio

$$R(m) \;=\; \frac{\sigma_\mu(m)}{\sigma_{\text{eval}}}$$

on a fixed 5-task suite (MMLU, GSM8K, HumanEval, ARC-C, HellaSwag), reported per task, not averaged. If $R(128) < 1$ at W4A16, the field's default is vindicated and calibration choice is noise at 4-bit. If $R(128) > 2$ at W2A16 while $R(512) < 1$, the answer is "128 is too small below 4 bits, and 512 fixes it" — a directly actionable change to every quantization toolchain default.

## 9. Key References

- **[Foundational]** Hassibi, B., Stork, D. *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon.* NIPS, 1992.
- **[Foundational]** Nagel, M., Amjad, R. A., van Baalen, M., Louizos, C., Blankevoort, T. *Up or Down? Adaptive Rounding for Post-Training Quantization.* ICML, 2020. — arXiv:2004.10568
- **[Foundational]** Li, Y., Gong, R., Tan, X., Yang, Y., Hu, P., Zhang, Q., Yu, F., Wang, W., Gu, S. *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction.* ICLR, 2021. — arXiv:2102.05426
- **[SOTA]** Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., Han, S. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438
- **[SOTA]** Sun, M., Liu, Z., Bansal, A., Kolter, J. Z. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR, 2024. — arXiv:2306.11695
- **[SOTA]** Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Jaggi, M., Alistarh, D., Hoefler, T., Hensman, J. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[Direct]** Williams, M., Aletras, N. *On the Impact of Calibration Data in Post-training Quantization and Pruning.* ACL, 2024. — arXiv:2311.09755
- **[Direct]** Jaiswal, A., Gan, Z., Du, X., Zhang, B., Wang, Z., Yang, Y. *Compressing LLMs: The Truth Is Rarely Pure and Never Simple.* ICLR, 2024. — arXiv:2310.01382
- **[Survey]** Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., Blankevoort, T. *A White Paper on Neural Network Quantization.* Qualcomm AI Research, 2021. — arXiv:2106.08295
- **[Benchmark]** Kurtic, E., Marques, A., Pandit, S., Kurtz, M., Alistarh, D. *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization.* 2025. — arXiv:2411.02355

## 10. Worked Example

Take one attention projection in a 7B model: $d_{\text{in}} = 4096$. The GPTQ default gives $N = 128 \times 2048 = 262{,}144$ activation columns, so $N/d_{\text{in}} = 64$ — the sample covariance $H_\ell$ has 64 samples per dimension. For a bulk eigenvalue that is fine: the relative error on an eigenvalue is $O(\sqrt{d_{\text{in}}/N}) = O(1/8) \approx 12\%$.

Now the part that matters. GPTQ's error-compensation update is
$$\delta_{F} = -\frac{w_q - \mathrm{quant}(w_q)}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{q,F},$$
which depends on the **inverse** Hessian. Inversion amplifies error in the small eigenvalues, exactly where a 64-samples-per-dimension estimate is worst — and transformer activation covariances are strongly anisotropic, with a heavy tail of small eigenvalues from rare tokens. The damping term $\lambda = 0.01\,\mathrm{tr}(H)/d_{\text{in}}$ exists to keep the inversion stable. It does that by *overwriting* the poorly-estimated tail with a constant.

The obstruction is now visible: the default damping makes the pipeline look insensitive to $m$, because it discards the part of the statistic that $m$ controls. Increasing $m$ from 128 to 512 buys a better tail estimate that $\lambda$ then flattens. Any experiment that varies $m$ with $\lambda$ fixed at the default measures a *floored* sensitivity curve, and will report a reassuring null.

A correct experiment must therefore sweep $m$ and $\lambda$ jointly — the $2\times$ design in §8 becomes $2\times$ three damping levels ($\lambda/10$, $\lambda$, $10\lambda$), 720 runs, still under 300 GPU-hours. The decision number is unchanged: $R(m) = \sigma_\mu(m)/\sigma_{\text{eval}}$, minimized over $\lambda$ at each $m$. Reporting $R$ at the default $\lambda$ alone would answer a different question than the one the title asks.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*