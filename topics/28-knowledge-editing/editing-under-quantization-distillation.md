---
id: 28-knowledge-editing/editing-under-quantization-distillation
title: "Editing Under Quantization and Distillation"
topic: 28-knowledge-editing
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Editing Under Quantization and Distillation

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editing-under-quantization-distillation` · **Status:** empirically-open

## 1. Problem Statement

A model is edited, then compressed. Or compressed, then edited. Does the edit survive?

Deployment pipelines almost never ship the weights that were edited. A fact is patched into an fp16 checkpoint with ROME/MEMIT/LoRA, and the artifact that reaches users is a 4-bit GPTQ or AWQ build, or a distilled student one tenth the size. The edit is a low-norm, highly structured perturbation; quantization is a comparatively large, near-isotropic perturbation; distillation regenerates the weights entirely from a soft-label objective.

Three variants, different difficulty:

- **Measurement.** Define *edit survival* so that it is not confounded by the compression's own effect on the same probe. Requires a base-rate arm; most reported numbers lack one.
- **Method.** Build an editor whose output is invariant under a declared compression operator — i.e. edit in a way that is quantization-aware or distillation-transferable.
- **Theory.** Give conditions on the edit update $\Delta W$ and the quantization grid under which the edited behaviour is provably preserved (or provably not).

Solving it means: a stated compression operator $\mathcal{C}$, an editor $\mathcal{E}$, and a bound of the form "for edits with margin $\geq m$, $\mathcal{C}\circ\mathcal{E}$ preserves efficacy to within $\epsilon$", verified empirically at 8B+ scale. The safety-relevant dual also matters: an *unwanted* deletion (unlearning, safety edit) that is undone by quantization is a live failure mode, not a hypothetical.

## 2. Formal Setting

Base model $f_\theta$, $\theta \in \mathbb{R}^d$. Edit request $e = (s, r, o^*)$ — subject, relation, target object — with pre-edit answer $o^c$.

**Editor.** $\mathcal{E}: (\theta, \{e_i\}_{i=1}^n) \mapsto \theta'$. For locate-and-edit methods the update is low-rank on one MLP matrix: $\Delta W = v k^\top / (k^\top C^{-1} k)$ for ROME, with $k$ the key activation at the subject's last token and $C = \mathbb{E}[kk^\top]$ the second-moment matrix of the layer's inputs.

**Compressor.** Two families.

- Quantization $Q_b$: per-group affine rounding, group size $g$, scale $s_G = \max_{j\in G}|w_j| / (2^{b-1}-1)$, so $Q_b(w) = s_G \cdot \mathrm{round}(w/s_G)$. Measured rounding error per element: $\sigma_q \approx s_G/\sqrt{12}$ for RTN; GPTQ and AWQ reduce the *output* error, not $\sigma_q$ itself.
- Distillation $D$: student $\phi$ trained by $\min_\phi \mathbb{E}_{x\sim\mathcal{D}}\, \mathrm{KL}(f_{\theta'}(\cdot|x)\,\|\,f_\phi(\cdot|x))$ over a transfer set $\mathcal{D}$ — measured as the actual token corpus used, since whether $\mathcal{D}$ contains the edited subject $s$ is the whole question.

**Quantities, as measured.**

- Efficacy: $\mathrm{ES}(\theta) = \Pr_e[\,p_\theta(o^*|s,r) > p_\theta(o^c|s,r)\,]$ over the edit set.
- Margin: $m_\theta(e) = \log p_\theta(o^*|s,r) - \log p_\theta(o^c|s,r)$.
- Generalization: same predicate on held-out paraphrases of $(s,r)$.
- Locality: same predicate on neighbourhood prompts sharing $r$ but not $s$; the score is *unchanged* prediction, not correct prediction.
- **Commutation defect**, the object of interest:
$$\Delta_{\mathrm{ES}} = \mathrm{ES}\big(\mathcal{C}(\mathcal{E}(\theta))\big) - \mathrm{ES}\big(\mathcal{E}(\mathcal{C}(\theta))\big)$$
- **Survival**, base-rate-corrected:
$$\mathrm{Surv} = \frac{\mathrm{ES}(\mathcal{C}(\mathcal{E}(\theta))) - \mathrm{ES}(\mathcal{C}(\theta))}{\mathrm{ES}(\mathcal{E}(\theta)) - \mathrm{ES}(\theta)}$$
The denominator is the edit's fp16 lift; the numerator subtracts what compression alone does to the same probes. Reported "post-quantization edit success" numbers almost never subtract the base rate.

**Assumptions, and which fail.**

1. *Rounding error is isotropic and independent of the edit.* False for GPTQ/AWQ: both use calibration activations, and error compensation is explicitly correlated across a row.
2. *The edit is small relative to the weight matrix.* Holds for single ROME edits ($\|\Delta W\|_F/\|W\|_F \sim 10^{-3}$) and fails after $10^3$–$10^4$ sequential MEMIT edits.
3. *Compression preserves the layer decomposition.* False for distillation, which has no weight correspondence at all — $\Delta_{\mathrm{ES}}$ is only definable behaviourally there.
4. *Edit and compression calibration sets are exchangeable.* False whenever the calibration corpus (typically WikiText or C4 slices) does not contain the edited entity — the common case for the rare entities editing benchmarks favour.

## 3. State of the Art

**Established.**

- Editing SOTA on the weight side: ROME (Meng et al., NeurIPS 2022), MEMIT (Meng et al., ICLR 2023) for batched edits, GRACE (Hartvigsen et al., NeurIPS 2023) for lifelong edits via a discrete key-value adaptor. Quantization SOTA: GPTQ (Frantar et al., ICLR 2023), AWQ (Lin et al., MLSys 2024), LLM.int8() (Dettmers et al., NeurIPS 2022).
- **Quantization can undo an intended forgetting.** Zhang et al., *Catastrophic Failure of LLM Unlearning via Quantization* (ICLR 2025) show unlearned models that retain little of the forget set in full precision recover most of it after 4-bit quantization. This is the strongest existing evidence that a weight-space intervention is not compression-invariant.
- **Quantization is an attackable channel.** Egashira et al., *Exploiting LLM Quantization* (NeurIPS 2024) construct models benign in fp16 whose int4 quantization is malicious — a constructive proof that behaviour is not preserved across $Q_b$, and that the defect can be placed deliberately.
- **Compression erases the tail first.** Hooker et al., *What Do Compressed Deep Neural Networks Forget?* (2019) — aggregate accuracy moves ~1% while a small "compression-identified exemplar" subset flips heavily. Edited facts are, by construction, tail facts.

**Claimed but unablated.**

- That PEFT/adapter edits are quantization-robust because QLoRA trains adapters over a frozen 4-bit base (Dettmers et al., NeurIPS 2023). QLoRA shows *training* works at 4-bit; it does not show an fp16-trained edit survives post-hoc quantization. LoftQ (Li et al., ICLR 2024) is direct evidence the mismatch is real enough to need correcting.
- That MEMIT edits are "robust to deployment" — asserted in deployment write-ups, not measured with a base-rate arm.

**Benchmark-number-only.** Any post-quantization edit score reported without $\mathrm{ES}(\mathcal{C}(\theta))$ alongside it. Nearly all of them.

## 4. What Is Known

- ROME/MEMIT reach ~99% efficacy and ~95%+ paraphrase generalization on CounterFact at GPT-J 6B and GPT-2 XL 1.5B (Meng et al., ICLR 2023) — in fp32/fp16, uncompressed.
- Sequential editing degrades badly before any compression: Gupta et al. (ACL Findings 2024) show gradual then catastrophic forgetting past ~10³ edits on GPT-2 XL and Llama-2 7B; Gu et al. (EMNLP 2024) show general-ability collapse from repeated edits.
- 4-bit weight-only quantization costs roughly 0.1–0.5 perplexity on WikiText2 for 7B–70B models with group size 128 (GPTQ, AWQ) — small in aggregate, which is exactly why per-fact effects go unnoticed.
- Precision has a measurable capacity cost: Kumar et al., *Scaling Laws for Precision* (ICLR 2025), fit effective-parameter-count loss from bit-width, at up to 1.7B params / 26B tokens.
- Editing efficacy does not track internal localization (Hase et al., NeurIPS 2023): an edit's success at layer $\ell$ says little about where the fact is stored. So there is no principled reason to expect the edited subspace to be the one quantization preserves.
- Ripple effects are already weak in fp16 (Cohen et al., TACL 2024): consequences of an edit largely fail to propagate. Any compression-induced loss stacks on top of an already-fragile signal.

## 5. What Is Not Known

- **Empirically open (the main gap).** Nobody has published a base-rate-corrected $\mathrm{Surv}$ table across {ROME, MEMIT, AlphaEdit, LoRA, GRACE} × {INT8, GPTQ-INT4, AWQ-INT4, INT3} × {8B, 70B}. The experiment is a few hundred GPU-hours. It has not been run at the right scale with the right control.
- **Empirically open.** Whether distillation transfers edits at all when the transfer corpus omits the edited subject — and whether on-policy distillation (GKD, Agarwal et al., ICLR 2024) transfers them better than off-policy, since on-policy sampling can visit the edited region.
- **Theoretically open.** No bound relating edit margin $m$, update norm $\|\Delta W\|$, key norm $\|k\|$, and quantization step $s_G$ to survival probability. The rank-1 signal versus isotropic-noise calculation in §10 is heuristic, not a theorem.
- **Methodologically blocked.** Edit identity under distillation. With no weight correspondence, "the same edit" in a student is only definable behaviourally, and behavioural equality on a finite probe set does not distinguish a transferred edit from a student that never knew the pre-edit fact.

## 6. Why It Is Hard

**Confounded measurement, primarily.** Compression changes the model's answer on the probe for reasons unrelated to the edit. Without the $\mathcal{C}(\theta)$ arm, a post-quantization efficacy of 0.72 is uninterpretable: it could be a 28% edit loss, or compression damaging facts the edit never touched, or the two cancelling.

**Non-identifiability, second.** The edit direction and the quantization error live in the same weight space with no orthogonality guarantee, and the calibration data that defines the quantization error is not the data that defines the edit. Two pipelines with identical fp16 behaviour can have different int4 behaviour — Egashira et al. build such pairs on purpose.

**Absent ground truth for distillation.** There is no answer to "did the edit transfer?" that is independent of the probe. The student's fp16 teacher-matched behaviour on the edit prompt is the *only* observable, and it conflates transfer with the student's own prior.

Compute is a minor obstruction: the grid is small. The blocker is that the control arm is cheap and still routinely omitted.

## 7. Current Research (as of 2026)

- **Compression-conditional behaviour as a security problem.** Egashira et al. (ETH Zurich / SRI Lab) and follow-on work on quantization-triggered backdoors. Active and growing *(frontier — verify current state)*.
- **Unlearning robustness.** Zhang et al.'s quantization result has made "robustness of forgetting to post-hoc transforms" a standard axis in unlearning evaluation; the same axis has not migrated to knowledge editing *(frontier — verify)*.
- **Editing toolchains.** EasyEdit (Zhang et al., ZJU) is the de facto harness; adding a compression stage to it is the obvious low-cost path to the missing table.
- **Null-space and preservation-constrained editors** (AlphaEdit and successors) aim at locality under sequential edits; whether the null-space constraint also buys quantization robustness is untested *(frontier — verify)*.
- **Quantization-aware adapter initialization** (LoftQ lineage) is the closest existing method-side answer, applied to fine-tuning rather than editing.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B-Instruct and Qwen2.5-14B. 3,000 CounterFact edits per condition. Editors: MEMIT, AlphaEdit, LoRA rank-16, GRACE. Compressors: INT8 RTN, GPTQ-INT4 g128, AWQ-INT4 g128, GPTQ-INT3 g128. Cost estimate: ~200 A100-hours.

**Arms.**
1. Treatment: $\mathcal{C}(\mathcal{E}(\theta))$.
2. **Control arm (the one usually missing):** $\mathcal{C}(\theta)$ — compressed but unedited, evaluated on the identical probe set, giving the base rate $\mathrm{ES}(\mathcal{C}(\theta))$.
3. Order control: $\mathcal{E}(\mathcal{C}(\theta))$ — edit the already-compressed model, for $\Delta_{\mathrm{ES}}$.

**Deciding number.** $\mathrm{Surv}$ at GPTQ-INT4, as defined in §2, on the efficacy probe. One threshold: **if $\mathrm{Surv} < 0.80$ for any weight-space editor while locality is unchanged, edits are compression-fragile and every deployed edit pipeline needs a post-quantization re-verification step.** If $\mathrm{Surv} > 0.95$ across all four editors, the concern is closed for quantization and the open problem reduces to distillation alone.

**Distillation extension.** Distil the edited 8B teacher into a 1B student with GKD over 200M tokens, in two conditions: transfer corpus containing the edited subjects, and corpus with those subjects filtered out. Report $\mathrm{Surv}$ for each. The gap between them is the transfer-set dependence, and it is currently unmeasured.

## 9. Key References

- **[Foundational]** Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Meng, Sharma, Andonian, Belinkov, Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[SOTA]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. — arXiv:2210.17323
- **[SOTA]** Lin, Tang, Tang, Yang, Dang, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. — arXiv:2306.00978
- **[SOTA]** Zhang, Wang, Ren, et al. *Catastrophic Failure of LLM Unlearning via Quantization.* ICLR 2025. — arXiv:2410.16454
- **[SOTA]** Egashira, Vero, Staab, He, Vechev. *Exploiting LLM Quantization.* NeurIPS 2024. — arXiv:2405.18137
- **[Foundational]** Hooker, Courville, Clark, Dauphin, Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- **[SOTA]** Hartvigsen, Sankaranarayanan, Palangi, Kim, Ghassemi. *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.* NeurIPS 2023. — arXiv:2211.11031
- **[SOTA]** Dettmers, Pagnoni, Holtzman, Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. — arXiv:2305.14314
- **[SOTA]** Agarwal, Vieillard, Zhou, et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR 2024. — arXiv:2306.13649
- **[SOTA]** Kumar, Ankner, Blumberg, et al. *Scaling Laws for Precision.* ICLR 2025. — arXiv:2411.04330
- **[Analysis]** Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Analysis]** Cohen, Biran, Yoran, Globerson, Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. — arXiv:2307.12976
- **[Analysis]** Gupta, Rao, Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024. — arXiv:2401.07453
- **[Survey]** Yao, Wang, Tian, et al. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172
- **[Survey]** Wang, Zhu, Liu, et al. *Knowledge Editing for Large Language Models: A Survey.* ACM Computing Surveys, 2024. — arXiv:2310.16218

## 10. Worked Example

One ROME edit on a 6B-class MLP down-projection, $W \in \mathbb{R}^{4096 \times 16384}$, weight std $\approx 0.02$.

**Quantization noise.** INT4, group size 128, absmax scale. Take a group max of $|w|_{\max} \approx 0.10$ (≈5σ, typical for a 128-element Gaussian group). Step $s_G = 2 \times 0.10 / 15 \approx 0.0133$. RMS rounding error:
$$\sigma_q \approx s_G/\sqrt{12} \approx 3.8\times10^{-3}$$
That is **19% of the weight standard deviation** — a large per-element perturbation that nonetheless costs ~0.3 perplexity, because it is near-isotropic and the network is robust to isotropic noise.

**Edit signal.** ROME's update is rank-1 and satisfies $\Delta W\,k = v$ exactly at the edit key. Its relative Frobenius norm is small, $\|\Delta W\|_F / \|W\|_F \approx 5\times10^{-3}$, giving a per-element RMS of $\approx 10^{-4}$ — **38× below $\sigma_q$**. Element-wise, the edit is invisible inside the rounding error.

**Why that is not yet the answer.** The edit is coherent; the noise is not. At the key direction, signal is $\|v\|$ and the rounding contribution is $E k$ with
$$\|Ek\| \approx \sigma_q \|k\| \sqrt{d_{\text{out}}} = 3.8\times10^{-3} \times \|k\| \times 64 \approx 0.24\,\|k\|$$
So the edit dominates the rounding perturbation at its own key only if
$$\frac{\|v\|}{\|k\|} \gtrsim 0.24$$

**The obstruction, made visible.** That ratio decides the outcome, and **no editing paper reports it.** ROME's $v$ is solved for a target logit, not a target norm; $\|k\|$ is whatever the layer-5 activation happens to be, and in GPT-J-class models MLP key norms vary by more than an order of magnitude across subjects. So the survival condition is satisfied for some edits and violated for others, within the same edit batch, and the aggregate efficacy number cannot tell you which. Two edits with identical fp16 margin $m = 4.2$ nats can differ by 10× in $\|v\|/\|k\|$ and therefore land on opposite sides of the int4 threshold — one survives, one reverts silently to $o^c$.

The fix is not a new method. It is logging $\|v\|/\|k\|$ per edit and reporting $\mathrm{Surv}$ against the $\mathcal{C}(\theta)$ base rate. Neither is currently standard.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*