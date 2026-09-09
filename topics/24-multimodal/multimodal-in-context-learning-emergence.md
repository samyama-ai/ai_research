---
id: 24-multimodal/multimodal-in-context-learning-emergence
title: "Emergence Threshold for Multimodal In-Context Learning"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Emergence Threshold for Multimodal In-Context Learning

> **Topic:** Multimodal Models · **ID:** `24-multimodal/multimodal-in-context-learning-emergence` · **Status:** empirically-open

## 1. Problem Statement

A vision-language model (VLM) given $n$ image-text demonstrations in its prompt sometimes improves on a held-out query and sometimes does not. The question: **is there a threshold — in parameters, tokens, interleaved-data fraction, or context length — above which a VLM acquires genuine cross-modal in-context learning (ICL), meaning it infers an image→label mapping it was not trained on, from the demonstrations alone?**

Three variants, with different difficulty:

- **Measurement.** Define a statistic that separates *cross-modal* ICL (the mapping from visual features to labels is read off the demonstrations) from *text-only* ICL (the demonstrations narrow the label set, prime a format, or recall a pretrained association) and from *task recognition* (the demonstrations merely name a task the model already knows). Currently unsettled — see §6.
- **Method.** Train a family of VLMs across scales and interleaved-data mixtures, and locate the transition in that statistic. Runnable, unrun at the right scale.
- **Theory.** Prove that under some data-distribution condition (burstiness, Zipfian label frequency, modality-alignment noise) cross-modal ICL is or is not achievable below a given model/data budget. Open.

**Solved** would mean: a published scaling law $f(N, D, \rho_{\text{interleaved}})$ predicting the cross-modal ICL statistic on held-out synthetic tasks, validated by out-of-sample prediction at a scale not used to fit it, with a stated transition sharpness.

## 2. Formal Setting

A model $M_\theta$ with $N$ non-embedding parameters, trained on $D$ tokens, of which a fraction $\rho \in [0,1]$ come from interleaved image-text documents (as opposed to single image-caption pairs). A **task** $\tau$ is a distribution over $(x, y)$ with $x$ an image and $y$ a label string. A prompt is
$$
P_n^\tau = \big[(x_1,y_1),\dots,(x_n,y_n), x_{\text{query}}\big],
$$
demonstrations drawn i.i.d. from $\tau$. Performance is $A_n(\tau) = \mathbb{E}\big[\mathbb{1}\{M_\theta(P_n^\tau) = y_{\text{query}}\}\big]$, estimated over $\geq 500$ queries and $\geq 8$ demonstration orderings (order variance in ICL is large; Lu et al., ACL 2022).

Three ablated prompt families, each measured the same way:

| Arm | Construction | Isolates |
|---|---|---|
| $A_n$ | true pairs | full ICL |
| $A_n^{\text{shuf}}$ | labels permuted within the demo set | label-mapping use |
| $A_n^{\text{noimg}}$ | demo images removed, labels kept | text-only priors |

Define the **cross-modal ICL gain**
$$
\Gamma_n(\tau) \;=\; A_n(\tau) \;-\; \max\!\big(A_n^{\text{shuf}}(\tau),\; A_n^{\text{noimg}}(\tau)\big),
$$
which is positive only if the model is using the *pairing* between demo images and demo labels. The emergence threshold is
$$
N^\star(\rho,\epsilon) \;=\; \inf\{\,N : \mathbb{E}_{\tau \sim \mathcal{T}}\,\Gamma_{n_{\max}}(\tau) > \epsilon\,\},
$$
with $\mathcal{T}$ a family of **novel** mappings (e.g. random assignments of shapes/textures to nonce words) so that $\tau$ cannot be recalled from pretraining.

Assumptions, and which fail in practice:

- *Demonstrations are i.i.d. from $\tau$.* Holds for synthetic $\mathcal{T}$; violated on natural benchmarks where retrieval-based demo selection is standard.
- *$\tau$ is unseen in pretraining.* Not verifiable for web-scale corpora; only enforceable with synthetic images or held-out label vocabularies. **Known violated** on VQAv2/COCO-style evaluations.
- *Accuracy is a smooth functional of capability.* Violated by construction — exact-match accuracy is discontinuous, which is the core of the mirage critique (Schaeffer et al., NeurIPS 2023). Use per-token log-likelihood of $y_{\text{query}}$ alongside accuracy.
- *Vision encoder resolution is not the binding constraint.* Violated when demo images are downsampled to fit context.

## 3. State of the Art

**Empirical SOTA (established).** Flamingo (Alayrac et al., NeurIPS 2022) is the first system with a clean monotone shot curve: Flamingo-80B on VQAv2 goes 56.3 (0-shot) → 63.1 (4-shot) → 67.6 (32-shot), and COCO CIDEr 84.3 → 113.8 at 32 shots. Open reproductions — OpenFlamingo (Awadalla et al., 2023), IDEFICS (Laurençon et al., NeurIPS D&B 2023, the OBELICS corpus) — recover the qualitative shape at 9B with smaller slopes. Emu2 (Sun et al., CVPR 2024) reports strong 37B few-shot results.

**Established negative result.** VL-ICL Bench (Zong, Bohdal, Hospedales, ICLR 2025) shows that on tasks requiring a *new* image→label mapping, many frontier VLMs are flat or decreasing in $n$, while the same models improve on tasks recoverable from text.

**Claimed but weakly ablated.** "Many-Shot In-Context Learning in Multimodal Foundation Models" (Jiang, Gao, Zou et al., 2024) reports large gains for GPT-4o and Gemini 1.5 Pro out to ~1000 demonstrations across ~10 datasets. The image-ablated arm is not run per-dataset, so the split between $\Gamma_n$ and text-only gain is unmeasured. This is **a benchmark number, not a mechanism claim**.

**Theory SOTA.** No VLM-specific theory. The closest is unimodal: Chan et al. (NeurIPS 2022) show ICL emerges only when training data is *bursty* and label distribution Zipfian; Reddy (ICLR 2024) gives a mechanistic account of the abrupt induction-head phase transition; Olsson et al. (2022) tie ICL onset to induction-head formation. None has been shown to transfer across a modality boundary.

## 4. What Is Known

- **Cross-modal gain is small where measured.** Baldassini et al. (CVPR-W 2024), studying IDEFICS-9B/80B and OpenFlamingo, find most of the many-shot benefit comes from text demonstrations and output-format priming; replacing demo images with blanks costs far less than replacing demo text. Scale: 9B–80B, VQA/captioning.
- **Label-shuffling barely hurts.** The unimodal finding (Min et al., EMNLP 2022 — random labels cost little at 0.8B–175B) reproduces for VLM classification prompts in VL-ICL Bench, which is exactly the signature of task recognition rather than mapping inference.
- **Interleaved pretraining data is necessary, not just helpful.** Flamingo's own ablation: removing the M3W interleaved corpus costs about a third of overall few-shot score. OBELICS-trained IDEFICS reproduces the direction.
- **Demo selection dominates demo count** at small $n$: retrieval-based selection can exceed random selection by more than 4→32 shot scaling on captioning (Yang et al., NeurIPS 2023).
- **Sharpness is metric-dependent.** Schaeffer et al. (NeurIPS 2023) show apparent discontinuities on exact-match collapse to smooth curves under token-level metrics. No published VLM emergence claim has been re-run under a continuous metric.

## 5. What Is Not Known

- **Empirically open.** Whether $\Gamma_n > 0$ at all for genuinely novel mappings above some $N$. No lab has trained a matched VLM family across $\ge 4$ scales with $\rho$ varied and run the shuffled/image-ablated control arms on synthetic tasks. The experiment is affordable (§8); it has not been run.
- **Empirically open.** Whether the threshold is in $N$, in $\rho$, or in context length. All three co-vary in every public model.
- **Theoretically open.** No result stating when a transformer trained on paired-then-interleaved data forms *cross-modal* induction heads, i.e. copy circuits whose key is a visual embedding and whose value is a text token. Chan et al.'s burstiness condition has no proven multimodal analogue.
- **Methodologically blocked.** "Novel mapping" has no operational test at web scale. Contamination can be excluded only by constructing images that cannot exist in the corpus, which changes the distribution the encoder was trained on — the control and the confound move together.

## 6. Why It Is Hard

**Confounded measurement, plus non-identifiability.** A VLM's few-shot gain has at least four sources: format priming, label-set restriction, pretrained image→label recall, and true mapping inference. Only the fourth is the object of interest, and standard benchmarks sum all four into one accuracy number. Worse, the two natural controls are not independent: removing demo images also removes the visual-token budget and shifts sequence statistics, so $A_n^{\text{noimg}}$ is not a clean counterfactual.

Second obstruction: **vision-token context cost.** At 256–729 tokens per image, 32 shots consume 8k–23k tokens before any text. Studying $n \to 10^3$ requires either aggressive token reduction (which changes the signal) or 1M-context models (which exist only as closed APIs, where $\rho$ and $N$ are unknown). So the many-shot regime is measurable only where the independent variables are hidden.

## 7. Current Research (as of 2026)

- **Controlled synthetic ICL for VLMs** — extending Chan/Reddy-style burstiness studies to two-modality streams; small academic groups, DeepMind alumni networks. *(frontier — verify)*
- **VL-ICL Bench and successors** (Edinburgh/Hospedales) as the standard novel-mapping harness; adoption is growing but ablation arms are still optional.
- **Token-compression for many-shot** (perceiver-resampler variants, visual token pruning) to make $n>100$ tractable on open 7B–34B models. *(frontier — verify)*
- **Mechanistic search for cross-modal induction heads** in open interleaved models (IDEFICS-2, Qwen2-VL, InternVL-2 class). No published positive identification known to this catalog. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Train 5 VLMs at $N \in \{160\text{M}, 410\text{M}, 1.4\text{B}, 2.8\text{B}, 6.9\text{B}\}$ (Pythia-class LM backbones, frozen SigLIP encoder, Flamingo-style cross-attention), each on 100B tokens, at two interleaved fractions $\rho \in \{0.1, 0.5\}$. 10 runs; roughly 3–8k A100-hours total.

**Evaluation.** 200 synthetic tasks $\tau$: random bijections from 8 procedurally generated shape/texture classes to 8 nonce words, images rendered at test time so they cannot be in pretraining. Sweep $n \in \{0,2,4,8,16,32\}$.

**Control arms.** (a) label-shuffled demos; (b) demo images blanked; (c) an $N$-matched *text-only* LM given ground-truth attribute descriptions instead of images, which upper-bounds what the text pathway alone can do.

**Deciding number.** $\Gamma_{32} = A_{32} - \max(A_{32}^{\text{shuf}}, A_{32}^{\text{noimg}})$, reported per scale with 95% bootstrap CIs over tasks and orderings. **The question is settled if $\Gamma_{32}$ crosses $+0.10$ absolute at some $N$ and stays below $+0.02$ at the next scale down, at both $\rho$.** If $\Gamma_{32} \le 0.02$ at 6.9B for both $\rho$, the claim "current-scale VLMs do cross-modal ICL" is falsified in the controlled regime and the threshold, if any, lies above 7B.

## 9. Key References

- **[Foundational]** Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS 2022. — arXiv:2204.14198
- **[Foundational]** Tsimpoukelli et al. *Multimodal Few-Shot Learning with Frozen Language Models.* NeurIPS 2021. — arXiv:2106.13884
- **[Foundational]** Chan et al. *Data Distributional Properties Drive Emergent In-Context Learning in Transformers.* NeurIPS 2022. — arXiv:2205.05055
- **[Foundational]** Wei et al. *Emergent Abilities of Large Language Models.* TMLR 2022. — arXiv:2206.07682
- **[Critique]** Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. — arXiv:2304.15004
- **[SOTA]** Zong, Bohdal, Hospedales. *VL-ICL Bench: The Devil in the Details of Multimodal In-Context Learning.* ICLR 2025. — arXiv:2403.13164
- **[SOTA]** Baldassini et al. *What Makes Multimodal In-Context Learning Work?* CVPR Workshops 2024. — arXiv:2404.15736
- **[SOTA]** Jiang, Gao, Zou et al. *Many-Shot In-Context Learning in Multimodal Foundation Models.* 2024. — arXiv:2405.09798
- **[Mechanism]** Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits, 2022. — arXiv:2209.11895
- **[Mechanism]** Reddy. *The Mechanistic Basis of Data Dependence and Abrupt Learning in an In-Context Classification Task.* ICLR 2024. — arXiv:2312.03002
- **[Data]** Laurençon et al. *OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents.* NeurIPS D&B 2023. — arXiv:2306.16527
- **[Related]** Min et al. *Rethinking the Role of Demonstrations.* EMNLP 2022. — arXiv:2202.12837

## 10. Worked Example

Task: 4-way classification of rendered textures — `{stippled, woven, cracked, spiralled}` → nonce labels `{blicket, dax, fep, wug}`, bijection resampled per task. Chance is 25%. Suppose a 9B interleaved VLM at $n=16$ measures:

```
A_16       = 0.61  (± 0.04)
A_16^shuf  = 0.55  (± 0.04)   labels permuted
A_16^noimg = 0.52  (± 0.05)   demo images blanked
A_0        = 0.27              zero-shot, labels listed
```

The headline reads as "+34 points over zero-shot — strong multimodal ICL." The decomposition says otherwise:

$$\Gamma_{16} = 0.61 - \max(0.55, 0.52) = 0.06 \pm 0.06.$$

Of the 34-point gain, ~25 points come from learning that only four nonce strings are legal outputs (visible in $A_{16}^{\text{noimg}}$ climbing to 0.52 with **no demo images at all**), and only ~6 points, within noise, from using the image→label pairing. The model is doing label-set restriction and format priming, not cross-modal mapping inference.

The obstruction is now visible as a sample-size problem. To resolve $\Gamma = 0.06$ against $\pm 0.06$ CIs you need roughly $16\times$ the evaluation budget — 200 tasks × 500 queries × 8 orderings ≈ $8\times10^5$ forward passes **per model per shot count**, each pass carrying 16 × 729 ≈ 11.7k vision tokens. That is why the number that decides the question has not been published: the expensive part is not training the five models, it is the variance reduction on the control arms, and every paper that skips the control arms reports 0.34 instead of 0.06.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*