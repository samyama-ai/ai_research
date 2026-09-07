---
id: 12-quantization-compression/alignment-degradation-compression
title: "Safety Alignment Degradation Under Compression"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Safety Alignment Degradation Under Compression

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/alignment-degradation-compression` · **Status:** empirically-open

## 1. Problem Statement

Post-training compression — weight quantization (GPTQ, AWQ, bitsandbytes NF4), unstructured/semi-structured pruning (SparseGPT, Wanda), low-rank factorization, distillation — is applied to an *already aligned* chat model and evaluated on perplexity and task accuracy. The question: **does compression damage the refusal behaviour installed by RLHF/DPO more than it damages general capability, and if so, by how much and through what mechanism?**

Three variants, of very different difficulty:

- **Measurement.** Define a scalar $\Delta$ that separates "the compressed model is worse at everything" from "the compressed model is selectively worse at refusing". Requires a capability-matched control; without it, raw attack-success rate (ASR) comparisons are uninterpretable.
- **Method.** Build a compression pipeline with a certificate: for a fixed attack budget, $\Delta \le \epsilon$. No current method offers this.
- **Theory.** Given a weight perturbation of bounded norm $\|\hat\theta-\theta\|$, bound the change in refusal probability on a harmful-prompt distribution. Open even for linear probes of the refusal direction.

Solved would mean: a compression recipe whose *capability-matched* safety gap is statistically indistinguishable from zero at $n$ large enough to detect 2 percentage points, replicated across model families and calibration seeds.

## 2. Formal Setting

Aligned model $\pi_\theta(y\mid x)$, compression operator $C$ with hyperparameters $h$ (bit width $b$, group size $g$, sparsity $s$) and calibration set $D_{\text{cal}}$ drawn with seed $\sigma$: $\hat\theta = C(\theta; h, D_{\text{cal}}, \sigma)$. $C$ is **stochastic in $\sigma$** — this is routinely ignored and is a main source of contradictory published results.

**Harm rate.** Harmful-request distribution $\mathcal{H}$ (HarmBench, AdvBench, StrongREJECT). Attack $a$ from budget class $\mathcal{A}_B$ (direct request; system-prompt strip; prefill; GCG at $B$ steps). Binary judge $J(x,y)\in\{0,1\}$, 1 = substantive compliance.

$$\mathrm{ASR}_B(\theta)=\mathbb{E}_{x\sim\mathcal{H}}\Big[\max_{a\in\mathcal{A}_B}\ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid a(x))} J(a(x),y)\Big]$$

Measured as: $n$ prompts $\times$ $k$ samples at $T=0$ and $T=1$, judged by a held-fixed classifier (HarmBench Llama-2-13B cls, or StrongREJECT rubric), with judge false-positive rate $\rho$ estimated on a human-labelled subset.

**Capability.** $U(\theta)$ = a fixed basket (MMLU, GSM8K, IFEval, MT-Bench) mapped to a scalar by z-scoring against the fp16 reference.

**Over-refusal.** $R(\theta)$ = refusal rate on benign look-alike prompts (XSTest), needed because a model that refuses everything scores perfectly on $\mathrm{ASR}$.

**The quantity of interest** is the capability-matched gap. Let $\theta_{U}$ be an uncompressed reference matched to $\hat\theta$ on $U$ (a smaller fp16 sibling, or $\theta$ degraded by an alignment-neutral perturbation such as isotropic weight noise calibrated to equal $U(\hat\theta)$):

$$\Delta_B \;=\; \mathrm{ASR}_B(\hat\theta) \;-\; \mathrm{ASR}_B(\theta_U), \qquad U(\hat\theta) = U(\theta_U) \pm \tau$$

$\Delta_B>0$ means compression removes safety *specifically*. Report with $\mathbb{E}_\sigma$ and $\mathrm{Var}_\sigma$ over $\ge 5$ calibration seeds.

**Assumptions, and which are violated.**
1. *$J$ is accurate and compression-independent.* Violated: judges over-fire on truncated/degenerate text, which low-bit models produce more often — the judge's error correlates with the treatment.
2. *$U$ is a sufficient statistic for capability.* Violated: instruction-following degrades before MMLU does; a model can match on MMLU and be much worse at following a refusal policy.
3. *$\mathcal{H}$ is fixed and non-adaptive.* Violated when GCG is re-optimised per model — attack strength then differs across arms.
4. *Compression is deterministic.* Violated: seed-to-seed $\mathrm{ASR}$ spread is often comparable to the reported effect.

## 3. State of the Art

**Established (ablated, multi-model).**
- *Decoding Compressed Trust* (Hong, Duan, Zhang, et al., ICML 2024, arXiv:2403.15447) benchmarks GPTQ/AWQ at 3–8 bit and magnitude/SparseGPT/Wanda at 50% sparsity across LLaMA-2-13B-chat, Vicuna-13B and others on eight trustworthiness axes. Finding: 4-bit quantization is roughly trust-neutral or mildly beneficial; 3-bit is high-variance and can collapse individual axes while perplexity looks acceptable.
- *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications* (Wei, Huang, Huang, Xie, Qi, Xia, Mittal, Wang, Henderson, ICML 2024, arXiv:2402.05162) localises safety-critical neurons/ranks and shows that removing a very small, identifiable subset destroys refusal while leaving utility largely intact — safety occupies a sparse, prunable substructure.

**Claimed but unablated.**
- Reports that quantization "increases jailbreak vulnerability" (e.g. Kumar et al., *Increased LLM Vulnerabilities from Fine-tuning and Quantization*, 2024, arXiv:2404.04392) mostly compare $\hat\theta$ against fp16 $\theta$ with no capability-matched arm and no calibration-seed replication. The direction is plausible; the magnitude is not established.
- *Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression* (Xu, Hong, et al., 2024) reports axis-specific degradation invisible to perplexity. Benchmark numbers, single seed.

**Adversarial SOTA (a distinct, sharper result).** *Exploiting LLM Quantization* (Egashira, Vero, Staab, He, Vechev, NeurIPS 2024, arXiv:2405.18137) constructs weights that are benign in fp16 and malicious after standard int4 quantization — the quantization step is the trigger. This establishes that the fp16 checkpoint is **not** a sound safety certificate for its quantized deployment.

## 4. What Is Known

- **Compression error is concentrated, not uniform.** GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023, arXiv:2210.17323) and SparseGPT (ICML 2023, arXiv:2301.00774) minimise layerwise reconstruction on ~128 calibration sequences; error lands on rare activation directions. Hooker et al. (*What Do Compressed Deep Neural Networks Forget?*, 2019, arXiv:1911.05248) showed on CIFAR/ImageNet-scale CNNs that pruning at constant top-1 accuracy shifts predictions disproportionately on a small "compression-identified exemplar" set — the aggregate metric hides the damage.
- **Safety is shallow.** Qi et al. (*Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, ICLR 2025, arXiv:2406.05946) show refusal is carried largely by the first few generated tokens; prefilling a non-refusal prefix collapses it. A perturbation need only shift early-token logits.
- **Alignment is fragile to small weight changes generally.** Qi et al. (ICLR 2024, arXiv:2310.03693) removed Llama-2-7B-Chat's refusal with 100 fine-tuning examples, ASR rising from near 0% to >80%. Compression is a weight change of comparable or larger norm.
- **4-bit is the practical accuracy sweet spot.** Dettmers & Zettlemoyer (*The case for 4-bit precision*, ICML 2023) — measured on OPT/BLOOM/Pythia up to 176B, zero-shot accuracy per bit peaks at 4-bit. Safety was not measured.
- **Not known to be established:** any *reproduced, capability-matched* estimate of $\Delta_B$ at 4-bit for a frontier-class model.

## 5. What Is Not Known

- **Empirically open (the main gap).** The sign and magnitude of $\Delta_B$ at 4-bit under a realistic attack budget, with a capability-matched control and $\ge 5$ calibration seeds, across $\ge 3$ model families and $\ge 2$ sizes. Every ingredient exists; the factorial has not been run at adequate $n$. Estimated cost: low thousands of GPU-hours.
- **Methodologically blocked.** There is no agreed capability-matched control. "Smaller fp16 sibling" differs in training data; "isotropic noise" is not distributionally like quantization error. Until this is settled, $\Delta_B$ is not well defined and published disagreements are unresolvable.
- **Theoretically open.** No bound of the form $|\mathrm{ASR}(\hat\theta)-\mathrm{ASR}(\theta)| \le f(\|\hat\theta-\theta\|_\ast, L)$ with a usable Lipschitz-type constant $L$ for the refusal decision. Also open: whether a quantization objective regularised on a harmful-prompt calibration set can *certify* $\Delta_B\le\epsilon$, or only reduce it on the calibration distribution.

## 6. Why It Is Hard

**The measurement is confounded and the evaluation does not measure what it names.** $\mathrm{ASR}$ falls when a model gets dumber — an incoherent 3-bit model refuses "successfully" by being unable to produce a usable answer. So a raw fp16-vs-int4 comparison conflates *safety preservation* with *capability loss*, in the direction that flatters compression. Removing the confound needs a control arm nobody agrees on (Section 5).

Three compounding obstructions:
1. **Statistical power.** Base rates are ~1–5%. Detecting a 2-point shift needs $n\approx 1{,}100$ prompts per arm (Section 10); standard suites ship 200–400.
2. **Judge noise correlates with treatment.** A judge FP rate of 3–5% is the same order as the effect, and its errors are not independent of bit width.
3. **Seed variance.** $\hat\theta$ depends on 128 calibration sequences; single-seed results are not reproducible measurements.

## 7. Current Research (as of 2026)

- **Safety-aware calibration** — adding refusal/harmful transcripts to $D_{\text{cal}}$ so reconstruction error avoids safety-critical directions. Natural extension of Wei et al.'s localisation; effectiveness beyond the calibration distribution unestablished *(frontier — verify)*.
- **Quantization-triggered backdoors and defences**, following Egashira et al. (ETH Zurich SRI Lab); defences based on quantization-interval noise are proposed but not standardised *(frontier — verify)*.
- **Trustworthiness benchmarking of compressed models** — Hong/Kailkhura/Wang (UT Austin, LLNL) line of work extending *Decoding Compressed Trust* to newer families.
- **Mechanistic refusal work** (refusal-direction ablation, Arditi et al. 2024) applied to quantization error: does quantization noise project onto the refusal direction more than onto random directions? A clean, cheap, largely unrun test *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** 3 families × 2 sizes (Llama-3.1-8B/70B-Instruct, Qwen2.5-7B/72B-Instruct, Mistral/Mixtral-Instruct) × 4 operators (GPTQ-W4A16-g128, AWQ-W4A16-g128, GPTQ-W3, Wanda 2:4) × 5 calibration seeds = 120 compressed checkpoints.

**Prompts.** $n=1{,}200$ harmful prompts (HarmBench + StrongREJECT), $k=4$ samples at $T=1$, plus 450 XSTest benign prompts for $R$. Attack budget fixed *once* on fp16 and replayed verbatim: direct request, system-prompt strip, 3-token prefill. Judge: HarmBench classifier, with 400 responses human-labelled per arm to estimate $\rho$.

**Control arms (both required).** (a) fp16 sibling matched to $\hat\theta$ within $\tau=1$ point of composite $U$; (b) fp16 base with Gaussian weight noise scaled per layer to match $\hat\theta$'s layerwise reconstruction error, tuned to equal $U(\hat\theta)$.

**The deciding number.** $\Delta_B = \mathrm{ASR}_B(\hat\theta) - \mathrm{ASR}_B(\theta_U)$, reported per operator as a mean over 5 seeds with a 95% CI (seed-clustered bootstrap). **Decision rule:** 4-bit compression is safety-neutral iff the CI for $\Delta_B$ lies within $\pm 2$ percentage points against *both* controls, at matched $R$ (within 3 points). If $\Delta_B > 2$ points for GPTQ-W4 on any family, "4-bit is trust-neutral" fails and safety-aware calibration becomes mandatory rather than optional. Estimated cost: ~3,000 A100-hours.

## 9. Key References

- **[Foundational]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[Foundational]** Frantar, Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML, 2023. — arXiv:2301.00774
- **[Foundational]** Hooker, Courville, Clark, Dauphin, Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- **[SOTA]** Hong, Duan, Zhang, et al. *Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression.* ICML, 2024. — arXiv:2403.15447
- **[SOTA]** Wei, Huang, Huang, Xie, Qi, Xia, Mittal, Wang, Henderson. *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.* ICML, 2024. — arXiv:2402.05162
- **[SOTA]** Egashira, Vero, Staab, He, Vechev. *Exploiting LLM Quantization.* NeurIPS, 2024. — arXiv:2405.18137
- **[SOTA]** Qi, Panda, Wang, et al. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR, 2025. — arXiv:2406.05946
- **[Method]** Lin, Tang, Tang, et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[Method]** Sun, Liu, Bair, Kolter. *A Simple and Effective Pruning Approach for Large Language Models (Wanda).* ICLR, 2024. — arXiv:2306.11695
- **[Evaluation]** Mazeika, Phan, Yin, et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML, 2024. — arXiv:2402.04249
- **[Evaluation]** Souly, Lu, Bowen, et al. *A StrongREJECT for Empty Jailbreaks.* NeurIPS, 2024. — arXiv:2402.10260
- **[Context]** Qi, Zeng, Xie, et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024. — arXiv:2310.03693
- **[Survey]** Zhu, Li, Liu, Ma, Wang. *A Survey on Model Compression for Large Language Models.* TACL, 2024.

## 10. Worked Example

Take Llama-3.1-8B-Instruct, GPTQ W4A16 with group size 128 and 128 C4 calibration sequences. A typical single-seed evaluation on 400 AdvBench-style direct requests, judged by the HarmBench classifier:

| Arm | Compliances / 400 | ASR | MMLU |
|---|---|---|---|
| fp16 | 8 | 2.0% | 68.1 |
| GPTQ-W4 (seed 0) | 16 | 4.0% | 67.4 |

The headline reads "quantization doubles ASR". Now do the arithmetic.

**Power.** Two-proportion test, $p_1=0.02$, $p_2=0.04$, $\alpha=0.05$, power 0.8:

$$n = \frac{\left(z_{0.975}\sqrt{2\bar p\bar q} + z_{0.8}\sqrt{p_1q_1+p_2q_2}\right)^2}{(p_2-p_1)^2} = \frac{(1.96\cdot 0.241 + 0.842\cdot 0.241)^2}{(0.02)^2} \approx 1{,}140$$

At $n=400$ this comparison is underpowered by ~2.9×; Fisher's exact on 8 vs 16 gives $p\approx 0.15$. The result is not evidence.

**Judge noise.** With judge FP rate $\rho=0.03$ measured on human labels, expected false compliances alone are $0.03\times 400 = 12$. The entire 8-vs-16 difference sits inside judge error, and low-bit models emit more degenerate text, which biases $\rho$ upward *in the treatment arm only*.

**Seed variance.** Re-running GPTQ with seeds 1–4 typically yields compliance counts spanning roughly 10–20 out of 400 — a spread as large as the claimed effect.

**Confound.** MMLU dropped 0.7 points. The capability-matched control asks: does an fp16 model 0.7 MMLU points weaker also sit at 4.0%? Nobody ran that arm, so the 2-point gap cannot be attributed to quantization rather than to generic capability loss.

Every published disagreement about this problem is reproduced in this one table: an effect smaller than the judge's error, measured at a third of the required $n$, with one seed and no control. That is the obstruction — not the compute to quantize the model, but the design of the comparison.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*