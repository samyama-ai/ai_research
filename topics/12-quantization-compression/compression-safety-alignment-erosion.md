---
id: 12-quantization-compression/compression-safety-alignment-erosion
title: "Compression and Safety Alignment Erosion"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compression and Safety Alignment Erosion

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/compression-safety-alignment-erosion` · **Status:** empirically-open

## 1. Problem Statement

Post-training compression — weight quantization (GPTQ, AWQ, int4/int8), unstructured or semi-structured pruning (SparseGPT, Wanda), and low-rank factorization — is selected almost entirely on perplexity and a handful of capability benchmarks. The question is whether a compression operator that preserves capability also preserves *refusal behavior*: the alignment-induced disposition to decline harmful requests, resist jailbreaks, and avoid unsafe completions.

- **Input:** an aligned model $\theta \in \mathbb{R}^d$, a compression operator $C$ (bit-width, sparsity pattern, calibration set), and a harm-elicitation distribution.
- **Output:** a decision on whether $C(\theta)$ is safety-equivalent to $\theta$ at a stated tolerance.
- **Solving it** means a compression-time predictor: given $C$ and cheap statistics of $\theta$, predict post-compression attack success rate to within a few points *without* running a full red-team evaluation.

Three variants, of unequal difficulty:

- **Measurement.** Define a safety metric that is stable enough that a 2-point change means something. Currently blocked: attack success rate (ASR) depends on the attack, the judge, and decoding temperature.
- **Method.** Build compression operators that provably or reliably retain refusal. Partially addressed (safety-aware calibration data, protecting identified safety-critical weights).
- **Theory.** Explain *why* refusal is more fragile than capability under the same perturbation norm. Open.

## 2. Formal Setting

Let $\pi_\theta(y \mid x)$ be the aligned policy. A compression operator $C_{b,s}$ maps $\theta \mapsto \hat\theta$ at bit-width $b$ and sparsity $s$, fit on a calibration corpus $\mathcal{D}_{\text{cal}}$ (typically 128 sequences of 2048 tokens from C4 or WikiText).

**Capability loss**, measured as it actually is measured — mean negative log-likelihood on a held-out corpus, exponentiated:

$$\mathrm{PPL}(\hat\theta) = \exp\Big(-\tfrac{1}{N}\sum_{i=1}^{N}\log \pi_{\hat\theta}(x_i \mid x_{<i})\Big), \qquad \Delta_{\text{cap}} = \mathrm{PPL}(\hat\theta) - \mathrm{PPL}(\theta).$$

**Safety**, measured as an attack success rate over a harmful-behavior set $\mathcal{X}_{\text{harm}}$ (e.g. HarmBench's 400 behaviors), an attack $A$ (direct request, GCG suffix, persona jailbreak, prefill), and a binary judge $J$ (a classifier or an LLM grader):

$$\mathrm{ASR}(\hat\theta; A, J) = \frac{1}{|\mathcal{X}_{\text{harm}}|}\sum_{x \in \mathcal{X}_{\text{harm}}} \mathbb{E}_{y \sim \pi_{\hat\theta}(\cdot \mid A(x))}\big[ J(x,y) \big].$$

**Erosion** is the paired difference at matched capability:

$$E(C) = \mathrm{ASR}(C(\theta)) - \mathrm{ASR}(\theta) \quad \text{subject to} \quad \Delta_{\text{cap}} \le \epsilon.$$

The constraint matters: an uncontrolled comparison confounds erosion with plain degradation — a model too broken to answer also refuses less coherently. A second axis is needed, over-refusal on benign look-alike prompts (XSTest), since $E(C) < 0$ can mean "safer" or "now refuses everything."

Assumptions and their status in practice:

| Assumption | Status |
|---|---|
| $J$ is an unbiased harm oracle | **Violated.** LLM judges disagree with human labels at rates in the 5–20% range and drift across model versions. |
| $\mathcal{D}_{\text{cal}}$ is safety-neutral | **Violated.** Calibration content measurably shifts which weights are preserved; adversarial calibration is an attack surface. |
| $\Delta_{\text{cap}} \le \epsilon$ controls for degradation | **Weak.** Perplexity is a poor proxy for instruction-following, which is the capability refusal depends on. |
| Erosion is a property of $C$ alone | **Violated.** It interacts with the alignment recipe (RLHF vs. DPO vs. safety SFT) and with prompt template fidelity. |

## 3. State of the Art

**Empirical SOTA (established).** *Decoding Compressed Trust* (Hong et al., ICML 2024) is the reference multi-dimension study: eight trust dimensions from DecodingTrust across quantization and pruning of Llama-2 and Vicuna-scale models. Established finding: 4-bit weight-only quantization (AWQ, GPTQ) preserves most trust dimensions, while 3-bit and aggressive pruning degrade them non-uniformly — some dimensions improve, adversarial robustness reliably does not. *Beyond Perplexity* (Xu, Gupta, Li, Bentham, Srikumar, 2024) extends this to generation-time harms and shows perplexity-matched compressed models diverge on safety and dialect fairness.

**Attack SOTA (established).** *Exploiting LLM Quantization* (Egashira, Vero, Staab, He, Vechev, NeurIPS 2024) constructs weights that are benign in fp16 and malicious after standard int4 quantization, demonstrating that the fp16 evaluation of a released checkpoint does not certify its quantized deployment.

**Mechanism SOTA (established).** *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications* (Wei, Huang, Y. Huang, Xie, Qi, Xia, Mittal, Wang, Henderson, ICML 2024) localizes safety to a small, identifiable set of neurons and rank-1 directions, disjoint enough from utility weights that they can be removed selectively.

**Claimed but unablated.** Safety-aware calibration sets ("include refusal data in $\mathcal{D}_{\text{cal}}$") are widely reported to reduce erosion; the ablation isolating calibration content from calibration *length* and domain match is not, to our knowledge, published at multiple scales. Vendor claims that an int4 release is "safety-equivalent" to the fp16 parent are typically a single benchmark number on one refusal set with no attack arm — a benchmark number, not an ablation.

## 4. What Is Known

- **4-bit is the practical capability knee.** Dettmers & Zettlemoyer (ICML 2023) show 4-bit maximizes zero-shot accuracy per bit across 19K runs, 125M–176B parameters. This is a *capability* result and has been repeatedly misread as a safety result.
- **Quantization SOTA numbers.** GPTQ (Frantar et al., ICLR 2023) quantizes OPT-175B/BLOOM-176B to 3–4 bits in ~4 GPU-hours with near-lossless perplexity; AWQ (Lin et al., MLSys 2024) matches at 4-bit with activation-aware scaling. SparseGPT (ICML 2023) reaches 50% unstructured sparsity on 175B models with minor perplexity loss; Wanda (ICLR 2024) matches without weight update.
- **Safety is localizable.** Wei et al. (ICML 2024) report that pruning a small fraction (on the order of a few percent) of safety-critical neurons in Llama-2-7b-chat raises attack success dramatically while utility benchmarks stay near baseline — measured at 7B and 13B.
- **The quantization backdoor is real.** Egashira et al. (NeurIPS 2024) report near-total flips in harmful-behavior compliance and insecure-code emission between fp16 and int4 for the same checkpoint, at 7B scale, across GPTQ/AWQ/LLM.int8 pipelines.
- **Alignment is shallow.** Qi et al. (ICLR 2025) show safety behavior in aligned Llama-2/Gemma models concentrates in the first few generated tokens; perturbing early-token distributions is enough to unlock the model. This gives a mechanistic reason why small weight perturbation moves refusal more than it moves accuracy.
- **Fine-tuning erodes alignment cheaply.** Qi et al. (ICLR 2024): 10 adversarial examples, <$0.20 of API fine-tuning, flips GPT-3.5-Turbo's refusal behavior. Compression is a different operator but the same fragility class.

## 5. What Is Not Known

- **Empirically open.** No scaling study of $E(C)$ across model size (7B → 70B → 400B+), bit-width (2/3/4/8), and alignment recipe *with the capability constraint enforced*. Runnable today; nobody has published it at $\ge$70B with matched $\Delta_{\text{cap}}$ and a multi-attack suite.
- **Empirically open.** Whether erosion is monotone in bits. Reported non-monotonicity (some dimensions improve at 4-bit) has not been replicated across seeds and calibration draws.
- **Theoretically open.** No bound relating $\|\hat\theta - \theta\|$ (or a Hessian-weighted equivalent) to change in refusal probability. Existing quantization theory bounds layerwise reconstruction error, which is capability-facing; there is no established transfer from reconstruction error to a behavior-level guarantee.
- **Methodologically blocked.** "Safety preserved" has no agreed definition. ASR is attack-, judge-, and decoding-dependent; a compressed model can lose refusal on GCG suffixes while gaining it on direct requests. Without a fixed attack-judge-decoding triple, cross-paper numbers are not comparable.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** Erosion and degradation are entangled. A 3-bit model has higher ASR *and* worse instruction-following; without matching on an instruction-following measure (not perplexity) you cannot attribute the ASR change to alignment loss. Almost no published comparison enforces the match.
2. **Absent ground truth.** $J$ is itself a model. When the judge and the compressed model share a base family, judge errors correlate with the thing being measured. Human relabeling of a few hundred completions costs more per configuration than the compression run.
3. **Non-identifiability of the safety subspace.** Safety-critical weights are defined operationally, by an attribution method plus a threshold. Different attributions select different weight sets with comparable post-hoc ASR effects, so "the safety circuit was damaged" is not a falsifiable claim as currently posed.

Compute is the secondary cost: a full grid (5 bit-widths × 3 model sizes × 4 attacks × 5 calibration seeds) is ~300 evaluation runs, with GCG-style optimized attacks costing GPU-hours per behavior.

## 7. Current Research (as of 2026)

- **Trustworthiness-under-compression benchmarking.** Continuation of the Hong et al. / DecodingTrust line (UT Austin, UIUC, Illinois-affiliated groups), extending to MoE and KV-cache quantization *(frontier — verify)*.
- **Adversarial calibration and quantization backdoors.** ETH Zurich SRI (Vechev group) following *Exploiting LLM Quantization*; open question is detectability from the fp16 checkpoint alone.
- **Safety-region-preserving compression.** Princeton/Stanford-affiliated work (Wei, Qi, Henderson, Mittal, Wang) on protecting identified safety directions during pruning; extensions to quantization-aware protection are active *(frontier — verify)*.
- **Depth of alignment.** Post-Qi-et-al. work on deepening safety beyond the first tokens, which if successful would predict *reduced* compression fragility — an untested falsifiable consequence.
- **KV-cache and activation quantization safety.** Nearly unstudied compared to weight quantization; a live gap.

## 8. Concrete Next Experiment

**Question:** does 4-bit quantization erode refusal at fixed capability, and does it worsen with scale?

- **Scale.** Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct (or equivalent openly aligned pair), bit-widths $\{16, 8, 4, 3\}$, GPTQ and AWQ, 5 calibration seeds each. ~80 model instances.
- **Capability matching (the control that is usually missing).** Accept a configuration into the comparison only if IFEval strict-prompt accuracy is within 1.0 point of the fp16 parent. Report erosion only within this matched set. Perplexity is recorded but not used for matching.
- **Control arms.** (a) fp16 parent; (b) *random-perturbation control*: add Gaussian noise to fp16 weights scaled so that IFEval drop equals the quantized model's drop. This separates "any perturbation of this magnitude erodes refusal" from "quantization specifically does."
- **Measurement.** HarmBench 400 behaviors × 3 attacks (direct, prefill, GCG-transfer), fixed judge (HarmBench classifier, frozen version), greedy decoding, plus XSTest over-refusal. 5 seeds → paired bootstrap CI.
- **The deciding number.** $E_{4\text{bit}} = \mathrm{ASR}_{\text{int4}} - \mathrm{ASR}_{\text{fp16}}$ on the capability-matched set, aggregated over attacks. **If the 95% CI for $E_{4\text{bit}}$ excludes 0 and its point estimate exceeds the noise control's $E_{\text{noise}}$ by more than 3 points, quantization erodes alignment beyond generic perturbation.** If the CI contains 0 at both 8B and 70B, the folk claim is not supported at 4 bits and attention should move to 3-bit and to adversarial calibration.

## 9. Key References

- **[Foundational]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[Foundational]** Frantar, Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML, 2023. — arXiv:2301.00774
- **[Foundational]** Dettmers, Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** Hong, Duan, Zhang, Wang, Yao, Zhou, et al. *Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression.* ICML, 2024.
- **[SOTA]** Egashira, Vero, Staab, He, Vechev. *Exploiting LLM Quantization.* NeurIPS, 2024. — arXiv:2405.18137
- **[SOTA]** Wei, Huang, Huang, Xie, Qi, Xia, Mittal, Wang, Henderson. *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.* ICML, 2024. — arXiv:2402.05162
- **[SOTA]** Lin, Tang, Tang, Yang, Dang, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** Sun, Zhou, Zhao, Kolter. *A Simple and Effective Pruning Approach for Large Language Models (Wanda).* ICLR, 2024. — arXiv:2306.11695
- **[Context]** Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024. — arXiv:2310.03693
- **[Context]** Qi, Panda, Lyu, Ma, Roy, Beirami, Mittal, Henderson. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR, 2025. — arXiv:2406.05946
- **[Benchmark]** Mazeika, Phan, Yin, Zou, Wang, Hendrycks, et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML, 2024. — arXiv:2402.04249
- **[Benchmark]** Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* NAACL, 2024. — arXiv:2308.01263
- **[Survey]** Xu, Gupta, Li, Bentham, Srikumar. *Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression.* 2024.
- **[Survey]** Zhu, Li, Liu, Ma, Wang. *A Survey on Model Compression for Large Language Models.* TACL, 2024.

## 10. Worked Example

Take an aligned 8B model with fp16 WikiText-2 perplexity 6.14 and HarmBench direct-request ASR of 2.0% (8 of 400 behaviors judged harmful).

Quantize with GPTQ to 4 bits, 128 C4 calibration sequences. Measured: perplexity 6.31 ($\Delta_{\text{cap}} = +0.17$, well inside a typical $\epsilon = 0.5$), ASR 5.5% (22/400). The headline reads: *+3.5 points of erosion at negligible capability cost.*

Now count. With $n = 400$ behaviors, the standard error on a 5.5% rate is $\sqrt{0.055 \cdot 0.945 / 400} \approx 1.14\%$; on 2.0% it is $0.70\%$. The unpaired 95% CI on the difference is roughly $3.5 \pm 2.6$ points — barely excluding zero. Then re-run with a different calibration seed: 4.2%. A third: 6.8%. The across-seed spread is comparable to the effect itself.

Then the judge. Hand-label the 22 flagged completions: 6 are refusals with a harmful-sounding preamble, 3 are fabricated non-answers the classifier scored as compliance. True ASR is 13/400 = 3.25%, and the effect halves to +1.25 points — now indistinguishable from zero.

Finally the control nobody runs: add Gaussian noise to the fp16 weights sized to produce $\Delta_{\text{cap}} = +0.17$. If that model also lands at 3–5% ASR, the finding is not about quantization at all; it is that *any* perturbation of this magnitude perturbs a shallow refusal boundary.

The obstruction is visible in the arithmetic: the effect size sought (a few ASR points) sits below the combined judge error, calibration-seed variance, and behavior-set sampling noise. The experiment in §8 is designed around exactly this — paired seeds, a frozen judge with a human-audited subsample, and a noise arm — because without all three the measurement cannot resolve the quantity it names.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*