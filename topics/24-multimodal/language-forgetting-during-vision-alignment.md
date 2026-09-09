---
id: 24-multimodal/language-forgetting-during-vision-alignment
title: "Catastrophic Forgetting of Language During Vision Alignment"
topic: 24-multimodal
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting of Language During Vision Alignment

> **Topic:** Multimodal Models · **ID:** `24-multimodal/language-forgetting-during-vision-alignment` · **Status:** partially-solved

## 1. Problem Statement

Take a pretrained text-only language model and continue training it on image–text data so it can see. The resulting vision-language model (VLM) is usually worse at text-only tasks than the model it started from. The problem is to characterise, predict, and eliminate that loss.

Three variants, with different difficulty:

- **Measurement.** Given base model $\theta_0$ and aligned model $\theta_V$, produce a number that is the *language capability lost*, separated from changes in output format, chat template, decoding, and evaluation harness. Currently the hardest of the three.
- **Method.** Produce a training recipe that reaches a target visual score $V^\star$ with text-only degradation $\le \epsilon$ at no more than $c\times$ the compute of the naive recipe. Partially solved: text-replay and frozen/expanded-parameter architectures reduce measured drops to near noise on standard suites.
- **Theory.** Explain *why* gradient updates on image-conditioned next-token prediction damage text-only behaviour — is it representational overwriting, task inference (the model infers "I am in image mode"), or output-distribution shift? Open.

A solution to the method variant is not a solution to the theory variant, and the field has largely stopped distinguishing them.

## 2. Formal Setting

Let $\theta_0 \in \mathbb{R}^d$ be the base LM, pretrained on text distribution $\mathcal{D}_T$. Vision alignment trains $\theta_V = A(\theta_0, \mathcal{D}_M, \phi)$ where $\mathcal{D}_M$ is a multimodal corpus, $\phi$ is a vision encoder plus connector, and $A$ is the recipe (which parameters are unfrozen, LR, data mixture, number of stages).

**Raw forgetting**, as measured:
$$\Delta_{\text{raw}} = \mathbb{E}_{b \sim \mathcal{B}}\big[ s_b(\theta_0) - s_b(\theta_V) \big]$$
where $\mathcal{B}$ is a text-only benchmark suite (MMLU, GSM8K, HellaSwag, HumanEval, IFEval) and $s_b$ is accuracy under a *fixed* harness. In practice $s_b(\theta_0)$ is measured few-shot on a base model and $s_b(\theta_V)$ zero-shot inside a chat template — so $\Delta_{\text{raw}}$ conflates capability loss with protocol change. This is the central measurement defect.

**Format-controlled forgetting.** Let $\Pi$ be a set of prompt formats. Define
$$\Delta_{\text{cap}} = \max_{\pi \in \Pi} s_b(\theta_0, \pi) - \max_{\pi \in \Pi} s_b(\theta_V, \pi).$$
Taking the max over formats absorbs template shift. $\Delta_{\text{raw}} - \Delta_{\text{cap}} \ge 0$ is the *elicitation gap*: capability present but not surfaced by the default prompt.

**Recoverable forgetting.** Let $R_k(\theta_V)$ be the score after $k$ text-only gradient steps at the alignment LR. Define
$$\Delta_{\text{perm}} = \lim_{k \to k_{\max}} \big[s_b(\theta_0) - R_k(\theta_V)\big], \qquad \Delta_{\text{rec}} = \Delta_{\text{cap}} - \Delta_{\text{perm}}.$$
$\Delta_{\text{perm}}$ is destroyed information; $\Delta_{\text{rec}}$ is suppressed information. Only $\Delta_{\text{perm}}$ deserves the word "catastrophic".

**Drift.** $\delta = \|\theta_V - \theta_0\|_2 / \|\theta_0\|_2$, per-layer $\delta_\ell$. For LoRA (Hu et al., ICLR 2022) with rank $r$, $\delta$ is bounded by construction; for full fine-tuning it is not.

**Assumptions, and where they break.**
1. *$\mathcal{B}$ is uncontaminated with respect to $\mathcal{D}_M$.* Violated — instruction-tuning mixtures such as LLaVA-Instruct contain GPT-4-generated text overlapping benchmark style, so some measured *gains* are contamination.
2. *$\theta_0$ is fixed and available.* Violated for most open VLMs whose base checkpoint was itself an instruct model already subject to prior fine-tuning.
3. *Scores are comparable across harnesses.* Violated — reported MMLU for the same checkpoint varies by several points across `lm-evaluation-harness` versions and answer-extraction rules.
4. *The vision encoder is inert.* Violated when $\phi$ is unfrozen; gradients then flow through a jointly-changing input distribution.

## 3. State of the Art

**Established (ablated, reproduced across labs).**
- *Freeze the LM entirely.* Flamingo (Alayrac et al., NeurIPS 2022) keeps LM weights frozen and inserts gated cross-attention; text-only behaviour is preserved by construction, $\Delta_{\text{perm}} = 0$ exactly. Cost: lower visual ceiling than fully-unfrozen recipes at matched scale.
- *Mix text-only data into every stage.* VILA (Lin et al., CVPR 2024) ablates this directly: training on image–text pairs alone degrades text-only accuracy, and blending text-only corpora plus interleaved image-text documents restores it while improving in-context visual ability. MM1 (McKinzie et al., ECCV 2024) reports the same at pretraining scale for SFT mixtures.
- *Interleaved > caption-only data.* Both VILA and OBELICS/Idefics2 (Laurençon et al., NeurIPS 2023/2024) find caption-only corpora, whose text is short and distributionally narrow, cause the largest degradation.

**Claimed but under-ablated.**
- *Parameter-isolation methods.* Model Tailor (Zhu et al., ICML 2024) and Wings (Zhang et al., NeurIPS 2024) report retained text-only performance via sparse patching / parallel modality-shifted learners. Reported on their own suites; no independent replication at $\ge 13$B.
- *Block expansion.* LLaMA Pro (Wu et al., ACL 2024) adds new zero-initialised blocks and freezes the rest — attractive theoretically, but the multimodal version is a benchmark number, not an ablation of $\Delta_{\text{perm}}$.
- *Weight interpolation.* WiSE-FT (Wortsman et al., CVPR 2022) and task arithmetic (Ilharco et al., ICLR 2023) give a one-parameter dial $\theta_\lambda = \theta_0 + \lambda(\theta_V - \theta_0)$ that trades vision for text. Widely used informally in VLM training; rarely reported.

**Benchmark-number-only.** Most frontier VLM reports (Qwen2-VL, InternVL, Molmo) state that text-only benchmarks are "maintained", citing a table with no matched-harness base-model row and no format control. Those tables do not establish $\Delta_{\text{cap}}$.

## 4. What Is Known

- **Forgetting is real and recipe-dependent, not architecture-dependent.** VILA's ablations (7B, Llama-2 base) show multi-point text-only degradation from image-text-pair-only training that is recovered by data blending — same architecture, different mixture.
- **Scale reduces it.** Ramasesh, Lewkowycz & Dyer (ICLR 2022) show forgetting in sequential fine-tuning shrinks monotonically with pretrained model scale, measured from 100M to ~10B parameters. VLM practice is consistent: 70B-class VLMs report smaller text-only drops than 7B-class ones at matched mixtures.
- **Much apparent forgetting is task inference, not erasure.** Kotha, Springer & Raghunathan (ICLR 2024) show fine-tuned LMs recover "forgotten" abilities when the prompt makes the original task explicit — evidence that $\Delta_{\text{rec}} > 0$ and often dominates. Luo et al. (2023, arXiv:2308.08747) find continual instruction fine-tuning degrades reasoning and knowledge in 1B–7B LMs and that degradation *grows* with model scale in their setup — the opposite trend to Ramasesh et al., and the two have never been reconciled under one protocol.
- **The vision side forgets too.** Zhai et al. (arXiv:2309.10313) show LLaVA-style tuning degrades the CLIP encoder's own classification accuracy, i.e. alignment is bidirectionally lossy.
- **Anchors.** Llama-2-13B scores 54.8 on 5-shot MMLU (Touvron et al., 2023). Derived VLM MMLU numbers are typically reported in the low-to-mid 50s under a chat template — a comparison that is *not* like-for-like, and is the reason the literature's "small drop" claims are weak evidence.

## 5. What Is Not Known

- **Methodologically blocked.** No standard protocol computes $\Delta_{\text{cap}}$ or $\Delta_{\text{perm}}$. Every published VLM "text retention" table is $\Delta_{\text{raw}}$ under a changed protocol. Until format-controlled and relearning-controlled numbers exist, the field cannot say whether *any* information is destroyed.
- **Empirically open.** The scaling law of forgetting: is $\Delta_{\text{perm}}(N, T_M, \rho)$ — parameters $N$, multimodal tokens $T_M$, text replay fraction $\rho$ — a clean power law? The runs are affordable (1B–8B, $\rho \in \{0, 0.05, 0.15, 0.4\}$) and nobody has published the grid with a matched-harness control.
- **Empirically open.** Whether the minimum $\rho$ that keeps $\Delta_{\text{cap}} \le 1$ point falls with $N$, which would make forgetting a small-model problem only.
- **Theoretically open.** No proof relating $\delta_\ell$, the multimodal gradient covariance, and $\Delta_{\text{perm}}$. Fisher-based accounts (EWC, Kirkpatrick et al., PNAS 2017) predict which weights matter, but there is no theorem giving a bound on capability loss from a bounded parameter move in a transformer.

## 6. Why It Is Hard

**Confounded measurement, and it is the binding constraint.** Vision alignment changes four things at once: the weights, the input distribution (image tokens now occupy context), the output style (chat template, verbosity, refusal behaviour), and the tokenizer/prompt scaffold. A drop on MMLU is attributable to any of them. The default protocol — base model few-shot versus VLM zero-shot-in-template — guarantees the four are entangled. Nothing about the standard reporting distinguishes "the model no longer knows the Krebs cycle" from "the model now answers in prose and the regex scores it wrong".

Secondary: **non-identifiability.** $\Delta_{\text{rec}}$ and $\Delta_{\text{perm}}$ cannot be separated without relearning experiments, and the relearning budget $k$ is a free parameter with no principled setting — a large enough $k$ retrains the capability outright.

## 7. Current Research (as of 2026)

- **Data-mixture engineering** is the production answer: every major open VLM release blends text-only tokens throughout. Direction is settled; the *optimal* $\rho$ as a function of scale is not published *(frontier — verify)*.
- **Modality-parallel parameters** — Wings-style side learners, block expansion, MoE routing where text tokens keep the original expert path. Active at academic groups (Alibaba/Tongyi, CUHK/Shanghai AI Lab lineage) *(frontier — verify)*.
- **Merging as a post-hoc dial.** Task-arithmetic interpolation between $\theta_0$ and $\theta_V$ is increasingly used to recover text scores after alignment; reported anecdotally, rarely ablated.
- **Mechanistic accounts.** Extending the implicit-inference view (Kotha et al.) to the multimodal case: does the presence of image tokens act as a latent task variable that shifts the whole conditional? *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Of the measured text-only drop in a standard LLaVA-style recipe, what fraction is permanent?

**Scale.** One base model, 8B class, fully open (e.g. Llama-3.1-8B base), single seed, two data conditions. Vision stage: 600M image-text tokens, SigLIP encoder, MLP connector, LM unfrozen. Roughly 2k A100-hours total including all arms — affordable for an academic lab.

**Arms.**
1. **Control (protocol-matched base).** $\theta_0$ evaluated under the *exact* VLM chat template with a blank-image-free text prompt, plus the same few-shot format, taking the max over 5 fixed formats. This is the control the literature omits.
2. **Naive.** Image-text pairs only, $\rho = 0$.
3. **Replay.** $\rho = 0.15$ text-only tokens from the base pretraining distribution.
4. **Relearn probe.** Take arm 2's checkpoint, run $k = 500$ steps on 50M text-only tokens (0.08% of the vision-stage budget), re-evaluate. Report $R_k$.

**Deciding number.** $\Delta_{\text{perm}} / \Delta_{\text{raw}}$ on MMLU + GSM8K + HumanEval, averaged.

- If $\Delta_{\text{perm}}/\Delta_{\text{raw}} < 0.2$, "catastrophic forgetting" in VLMs is mostly an elicitation and suppression artifact, and the correct fix is cheap text relearning or a merge — the method variant is solved and the framing should change.
- If $> 0.5$, weights are genuinely being overwritten, and parameter-isolation architectures are justified over data mixing.

No published VLM paper reports this ratio.

## 9. Key References

- **[Foundational]** McCloskey, M. & Cohen, N. *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 1989.
- **[Foundational]** French, R. *Catastrophic forgetting in connectionist networks.* Trends in Cognitive Sciences, 1999.
- **[Foundational]** Kirkpatrick, J. et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017. — arXiv:1612.00796
- **[Foundational]** Alayrac, J.-B. et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS, 2022. — arXiv:2204.14198
- **[SOTA]** Lin, J. et al. *VILA: On Pre-training for Visual Language Models.* CVPR, 2024. — arXiv:2312.07533
- **[SOTA]** McKinzie, B. et al. *MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training.* ECCV, 2024. — arXiv:2403.09611
- **[SOTA]** Zhang, Y. et al. *Wings: Learning Multimodal LLMs without Text-only Forgetting.* NeurIPS, 2024. — arXiv:2406.03496
- **[SOTA]** Zhu, D. et al. *Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models.* ICML, 2024. — arXiv:2402.12048
- **[Analysis]** Kotha, S., Springer, J. M. & Raghunathan, A. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference.* ICLR, 2024. — arXiv:2309.10105
- **[Analysis]** Ramasesh, V. V., Lewkowycz, A. & Dyer, E. *Effect of scale on catastrophic forgetting in neural networks.* ICLR, 2022.
- **[Analysis]** Luo, Y. et al. *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.* 2023. — arXiv:2308.08747
- **[Analysis]** Zhai, Y. et al. *Investigating the Catastrophic Forgetting in Multimodal Large Language Models.* 2023. — arXiv:2309.10313
- **[Method]** Wortsman, M. et al. *Robust fine-tuning of zero-shot models.* CVPR, 2022. — arXiv:2109.01903
- **[Method]** Ilharco, G. et al. *Editing Models with Task Arithmetic.* ICLR, 2023. — arXiv:2212.04089
- **[Method]** Wu, C. et al. *LLaMA Pro: Progressive LLaMA with Block Expansion.* ACL, 2024. — arXiv:2401.02415
- **[Survey/Context]** Laurençon, H. et al. *What matters when building vision-language models?* NeurIPS, 2024. — arXiv:2405.02246
- **[Survey/Context]** Liu, H. et al. *Visual Instruction Tuning.* NeurIPS, 2023. — arXiv:2304.08485

## 10. Worked Example

Take a 13B VLM built on Llama-2-13B. The published comparison looks like:

```
Llama-2-13B base, MMLU 5-shot, lm-eval-harness   54.8
VLM-13B,          MMLU 0-shot, chat template     51.2   (illustrative)
Reported "forgetting"                            -3.6 points
```

Now decompose it under the definitions in §2.

1. **Protocol control.** Re-run the base model *inside the VLM chat template*, zero-shot, scored by the same answer extractor. Base models routinely lose several points when moved from 5-shot letter-completion to zero-shot chat, purely from answer-format mismatch. Suppose the controlled base scores 52.4. Then $\Delta_{\text{raw}} = 3.6$ but $\Delta_{\text{cap}} \le 1.2$ — two-thirds of the headline was protocol.
2. **Format max.** Evaluate both models over five formats and take the max, per $\Delta_{\text{cap}}$. If the VLM's best format reaches 52.0 and the base's best reaches 54.8, $\Delta_{\text{cap}} = 2.8$ — a *larger* number than step 1 gave. The two controls disagree, and the literature specifies neither.
3. **Relearning probe.** 500 text-only steps on 50M tokens returns the VLM to 54.1. Then $\Delta_{\text{perm}} = 0.7$ and $\Delta_{\text{rec}} = 2.1$.

The obstruction is visible in the arithmetic: the same checkpoint pair yields "3.6 points forgotten", "1.2", "2.8", or "0.7" depending only on which control you pick, and every published VLM table picks the first. The quantity the field names — catastrophic forgetting — is not the quantity it measures. Until $\Delta_{\text{perm}}$ is reported, claims that a method "prevents forgetting" are claims about 0.7 points hidden inside a 3.6-point measurement artifact.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*