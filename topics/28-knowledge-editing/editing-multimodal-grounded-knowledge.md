---
id: 28-knowledge-editing/editing-multimodal-grounded-knowledge
title: "Editing Multimodal Grounded Knowledge"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Editing Multimodal Grounded Knowledge

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editing-multimodal-grounded-knowledge` · **Status:** open

## 1. Problem Statement

A vision-language model (VLM) holds a fact that is *grounded*: it is retrievable both from a name ("Lionel Messi plays for …") and from a picture of the entity, with no name in the prompt. The edit problem: change the fact once, and have the change hold through **every** access path — text prompt, image prompt, image+text, a different photo of the same entity, a paraphrase, a downstream multi-hop query — while leaving unrelated facts and general perception intact.

Three variants, different difficulty:

- **Measurement.** Define a success predicate that separates "the model changed the fact" from "the model learned a shortcut from the edit prompt's pixels." No accepted definition exists. This is the binding variant.
- **Method.** Given a working predicate, find an edit operator that satisfies it. Current operators edit either the LLM decoder or the vision encoder/projector, and generalization across that boundary is poor.
- **Theory.** Characterize when a fact is *localizable* to a parameter subspace shared by both modalities, and prove whether a rank-1 update can be modality-invariant. Open.

Solving it means: an edit applied once yields the new answer for held-out images of the entity at rates within a few points of held-out text paraphrases, with no drop on unrelated multimodal benchmarks.

## 2. Formal Setting

Let $f_\theta: \mathcal{I} \times \mathcal{T} \to \Delta(\mathcal{V})$ be a VLM with parameters $\theta = (\theta_v, \theta_p, \theta_\ell)$ — vision encoder, projector, language decoder. A fact is a triple $(s, r, o)$. Its **access set** is
$$A(s,r) = \{(i, t) : i \in I_s \cup \{\varnothing\},\ t \in T_{s,r}\},$$
where $I_s$ is the set of images depicting $s$ and $T_{s,r}$ the set of textual phrasings of the query; $\varnothing$ is the no-image case. An edit request is $(s,r,o \to o^*)$ applied at a single prompt $(i_0, t_0) \in A(s,r)$, producing $\theta'$.

Measured quantities, each an empirical mean over a finite sampled set:

- **Reliability** $\mathrm{Rel} = \mathbb{1}[\arg\max f_{\theta'}(i_0,t_0) = o^*]$ — exact-match on the edit prompt itself.
- **Text generality** $\mathrm{G}_T = \mathbb{E}_{t \sim T_{s,r}\setminus t_0}\,\mathbb{1}[f_{\theta'}(\varnothing,t) = o^*]$.
- **Image generality** $\mathrm{G}_I = \mathbb{E}_{i \sim I_s\setminus i_0}\,\mathbb{1}[f_{\theta'}(i,t_0) = o^*]$ — the quantity of interest.
- **Modality gap** $\Delta = \mathrm{G}_T - \mathrm{G}_I$. The problem is open exactly because $\Delta \gg 0$ in every published system.
- **Text locality** $\mathrm{L}_T$, **image locality** $\mathrm{L}_I$: KL or top-1 agreement between $f_\theta$ and $f_{\theta'}$ on out-of-scope prompts, in-modality and cross-modality respectively.
- **Portability** $\mathrm{P}$: accuracy on queries whose answer *entails* $o^*$ through one further hop (Cohen et al., TACL 2024, ported to VLMs by VLKEB).

Assumptions, with those known violated marked:

1. $I_s$ is sampleable and the model recognizes $s$ from each $i \in I_s$ **pre-edit**. *Violated* — VLM entity recognition on long-tail entities is often below 50%, so $\mathrm{G}_I$ mixes editing failure with recognition failure.
2. Locality sets are independent of the edit. *Violated* — nearest-neighbor entities in CLIP space are the ones that break, and they are usually excluded from locality sets by construction.
3. Exact match is a valid success predicate. *Violated* for open-ended captioning; MMEdit-style benchmarks score generation with token overlap, which passes strings that do not assert $o^*$.
4. $\theta_v$ is frozen (most editors touch only $\theta_\ell$). *Assumed, not established* — this presupposes the fact is stored decoder-side, which Section 4 shows is only partly true.

## 3. State of the Art

**Benchmarks (established as artifacts, not as valid measurements).**
- **MMEdit** (Cheng, Tian, Zhang et al., *Can We Edit Multimodal Large Language Models?*, EMNLP 2023) — first multimodal editing suite; VQA and image-captioning subtasks over BLIP-2 OPT and MiniGPT-4; introduces T-Locality vs M-Locality.
- **VLKEB** (Huang et al., NeurIPS 2024 Datasets & Benchmarks) — built from a multimodal knowledge graph, adds **portability** (multi-hop) to multimodal editing; evaluates BLIP-2, MiniGPT-4, LLaVA-1.5.
- **MIKE** (Li et al., 2024) — fine-grained *entity* editing, few-shot images per entity.
- **MC-MKE** (2024) — decomposes errors into misreading (visual) vs misrecognition (knowledge).
- **MMKE-Bench** (ICLR 2025) — 2,940 knowledge items over 8,363 images and 33 categories, three edit types: visual entity, visual semantic, user-specific.

**Methods.** Ported from text: **ROME**/**MEMIT** (Meng et al., NeurIPS 2022; ICLR 2023), **MEND** (Mitchell et al., ICLR 2022), **SERAC** (Mitchell et al., ICML 2022), **IKE** in-context editing, and the **EasyEdit** toolkit implementations. Multimodal-native: **UniKE** (Pan et al., NeurIPS 2024) unifies parametric and in-context editing in one semantic space and reports the best joint reliability/generality/locality on MMEdit-style splits.

**Established:** ports run and produce non-trivial reliability; SERAC-style external-memory editors dominate on locality because they do not touch $\theta$.
**Claimed but unablated:** that parametric editors "insert multimodal knowledge." No published ablation isolates whether $\mathrm{G}_I > 0$ comes from the edit or from the model re-deriving $o^*$ via an entity name it reads off the image. **Benchmark-number-only:** essentially every reported $\mathrm{G}_I$; none has an independent reproduction across all three of BLIP-2, LLaVA, and Qwen-VL class models.

## 4. What Is Known

- **Reliability is easy, cross-modal generality is not.** Across MMEdit and VLKEB, single-edit reliability for ROME/MEND/IKE on BLIP-2 and MiniGPT-4 is routinely $>90\%$, while image-generality is tens of points lower — a $\Delta$ of roughly 20–40 points at the 7B-decoder scale. The ordering (Rel > $\mathrm{G}_T$ > $\mathrm{G}_I$ > $\mathrm{P}$) is stable across every editor tested.
- **Portability collapses.** VLKEB reports multi-hop portability far below in-scope generality for all editors at 7B–13B; no editor exceeds chance-adjusted competence on 2-hop multimodal queries.
- **Vision-side editing damages perception.** MMEdit's finding that editing the vision module gives worse generality *and* worse locality than editing the last decoder layers has held up qualitatively in later work.
- **Localization does not predict editability.** Hase et al. (NeurIPS 2023) showed for text LLMs that causal-tracing localization is uncorrelated with where an edit succeeds. Nothing contradicts this in the multimodal case; it is the reason "edit the visual pathway" is not obviously right.
- **Sequential edits degrade.** Gupta et al. (ACL Findings 2024) document gradual then catastrophic forgetting under long text edit sequences; multimodal sequential editing at $n>1000$ is largely unreported.
- **Scale.** All of the above is measured at 7B–13B decoders with ViT-g/CLIP-L vision towers and $\le 5{,}000$ edits per benchmark. Nothing is measured at 70B+ or with native-resolution/interleaved architectures.

## 5. What Is Not Known

- **Methodologically blocked.** Whether $\mathrm{G}_I$ measures editing at all. If a VLM answers a held-out image query by first inferring the entity name and then querying decoder-side text knowledge, then $\mathrm{G}_I$ is downstream of OCR/recognition and an edit that "works" is just a text edit reached through a caption. No published benchmark controls for this. Until it is controlled, $\Delta$ is uninterpretable.
- **Empirically open.** Whether editing the projector $\theta_p$ (a small module, $\sim 20$M params in LLaVA-style models) gives better $\mathrm{G}_I$ per unit locality damage than editing $\theta_\ell$. Runnable today on 8× A100 in days; unrun as a clean ablation.
- **Empirically open.** Whether $\Delta$ shrinks with decoder scale or with training-data multimodality. Requires 7B/34B/72B sweep on one architecture family.
- **Theoretically open.** Whether a rank-1 update to a decoder MLP can be modality-invariant. The linear-associative-memory model behind ROME assumes a single key $k_s$ per subject; images of $s$ induce a *distribution* of keys $\{k_s^{(i)}\}$ with non-trivial spread. No result bounds $\mathrm{G}_I$ in terms of that spread, and no proof that a low-rank edit covering the key distribution must also hit off-subject keys.

## 6. Why It Is Hard

**Confounded measurement, specifically.** $\mathrm{G}_I$ is a product of two events — the model recognizes $s$ in the held-out image, and the edited association fires — and benchmarks report only the product. A 30-point $\Delta$ is consistent with a perfect editor on a 70%-recognition model, or a 70%-effective editor on a perfect recognizer. These have opposite engineering implications and no current benchmark separates them.

Compounding: **non-identifiability of the visual key.** ROME solves $W' k_s = v_{o^*}$ for one key. For images there is no one key; the edit must cover a set whose diameter in $\mathbb{R}^d$ is unknown and image-dependent, and enlarging the covered ball is exactly what destroys $\mathrm{L}_I$. The trade-off is a geometric constraint, not an engineering shortfall.

Third: **absent ground truth for locality.** "Unrelated image" is undefined. Sampling unrelated images uniformly makes $\mathrm{L}_I$ near 1.0 trivially; sampling CLIP-nearest-neighbors makes it collapse. Reported locality numbers are a function of the sampler, and samplers are not standardized across MMEdit, VLKEB, and MMKE-Bench.

## 7. Current Research (as of 2026)

- **Unified parametric + non-parametric editing** — UniKE line (USTC/Zhejiang groups) treating in-context and weight edits in one space. Active.
- **Benchmark decomposition** — MC-MKE's misreading/misrecognition split is the closest existing attempt at de-confounding $\mathrm{G}_I$; it separates error *types* but not the recognition/association product. *(frontier — verify)* Extensions adding pre-edit recognition gating are reported in 2025–2026 workshop work.
- **Lifelong multimodal editing** — sequential edits at $n \ge 10^3$ with memory-based routers; Zhejiang NLP (EasyEdit maintainers) is the main source of tooling. *(frontier — verify)*
- **Editing in unified any-to-any models** (image generation and understanding in one backbone) — whether an edit propagates to generated images. Essentially unstudied. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** is the multimodal generality gap $\Delta$ an editing failure or a recognition failure?

**Scale.** One architecture family, LLaVA-1.5-7B and -13B (or Qwen2-VL-7B). 500 entities from VLKEB/MMKE-Bench, each with $\ge 5$ held-out images. 500 single edits with ROME, MEMIT, MEND, and IKE — 4 editors × 2 models × 500 edits ≈ 4,000 edit runs, under 500 GPU-hours on A100s.

**Gating step.** Before editing, ask the unedited model an entity-identification question on each held-out image ("Who/what is shown?"). Partition images into $R^+$ (recognized) and $R^-$ (not).

**Control arm.** The same 500 edits applied and evaluated **text-only** on entity names, giving $\mathrm{G}_T$. Second control: held-out images of $s$ with the entity name *also present in the prompt*, which removes the recognition requirement entirely.

**The deciding number.** $\Delta^{+} = \mathrm{G}_T - \mathrm{G}_I \mid R^+$ — the modality gap restricted to images the model provably recognizes.

- If $\Delta^{+} < 5$ points, the multimodal editing problem is a *recognition* problem; the field should stop building editors and start gating benchmarks on recognition.
- If $\Delta^{+} > 20$ points, there is a genuine cross-modal insertion failure, and the projector-editing ablation in Section 5 becomes the priority.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning. *Fast Model Editing at Scale.* ICLR 2022. — arXiv:2110.11309
- **[Foundational]** Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn. *Memory-Based Model Editing at Scale.* ICML 2022.
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[SOTA / benchmark]** Siyuan Cheng, Bozhong Tian, Qingbin Liu, Xi Chen, Yongheng Wang, Huajun Chen, Ningyu Zhang. *Can We Edit Multimodal Large Language Models?* EMNLP 2023. — arXiv:2310.08475
- **[SOTA / benchmark]** Han Huang et al. *VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark.* NeurIPS 2024 Datasets and Benchmarks Track.
- **[SOTA / benchmark]** *MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge.* ICLR 2025.
- **[SOTA]** Kaihang Pan et al. *Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration.* NeurIPS 2024.
- **[Analysis]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Analysis]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. — arXiv:2307.12976
- **[Analysis]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale Leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024.
- **[Survey]** Ningyu Zhang et al. *A Comprehensive Study of Knowledge Editing for Large Language Models.* 2024. — arXiv:2401.01286

## 10. Worked Example

Edit: *(Lionel Messi, plays-for, Inter Miami $\to$ Newell's Old Boys)* on LLaVA-1.5-7B, applied with ROME at layer 5 of the decoder MLP, prompt = one photo of Messi + "Which club does this player play for?".

Post-edit measurements on 10 held-out photos and 10 text paraphrases (illustrative but typical magnitudes):

| Probe | $n$ | Answer $=o^*$ |
|---|---|---|
| Edit prompt (Rel) | 1 | 1/1 |
| Text paraphrases, name given ($\mathrm{G}_T$) | 10 | 9/10 |
| Held-out photos, no name ($\mathrm{G}_I$) | 10 | 4/10 |
| Held-out photos, name in prompt | 10 | 9/10 |
| CLIP-nearest other footballers ($\mathrm{L}_I$) | 10 | 3/10 flipped to $o^*$ |

Raw $\Delta = 90 - 40 = 50$ points — the headline "multimodal editing fails" number. Now gate: the unedited model identifies "Messi" from only 5 of the 10 photos. Restricting to $R^+$, $\mathrm{G}_I \mid R^+ = 4/5 = 80\%$, so $\Delta^{+} = 10$ points. Four-fifths of the apparent gap was recognition, not editing.

The obstruction is visible in two places. First, the same run supports two incompatible conclusions depending on a gating step no benchmark performs. Second, $\mathrm{L}_I$: 3 of 10 visually similar footballers now also "play for Newell's Old Boys" — the rank-1 update covered a ball in key space large enough to reach four Messi photos and, unavoidably, three other players. Widening coverage to fix the fifth photo would flip more neighbors. That trade is geometric, and no current benchmark reports both sides of it on the same sampler.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*