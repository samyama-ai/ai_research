---
id: 07-embeddings/binding-problem-vision-language-embeddings
title: "Binding Problem in Vision-Language Embeddings"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Binding Problem in Vision-Language Embeddings

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/binding-problem-vision-language-embeddings` · **Status:** open

## 1. Problem Statement

A dual-encoder vision-language model maps an image and a caption to a shared vector space and scores them by inner product. The binding problem is that this score is largely insensitive to *which attribute belongs to which object* and *which object fills which role of a relation*. "A red cube left of a blue sphere" and "a blue cube left of a red sphere" have nearly identical text embeddings; the same image scores nearly the same against both.

Three variants, with different difficulty:

- **Measurement.** Define a quantity that isolates binding failure from lexical, syntactic, plausibility, and visual-recognition confounds. Not yet settled — most published "compositionality" scores are partly measuring caption naturalness.
- **Method.** Train or fine-tune an encoder pair whose score function separates a scene description from its role-swapped variant, without degrading retrieval. Partially achieved by hard-negative fine-tuning, at a cost in generality.
- **Theory.** Determine whether contrastive image-text training with a fixed-width joint embedding *can* identify binding structure from natural caption distributions, or whether the swap direction is non-identifiable under that objective. Open.

Solving it means: an encoder whose binding accuracy on artifact-controlled swap tests approaches its own object-recognition accuracy, with the gap closing rather than the benchmark being fit.

## 2. Formal Setting

Let $f_\theta: \mathcal{X} \to \mathbb{S}^{d-1}$ and $g_\phi: \mathcal{T} \to \mathbb{S}^{d-1}$ be image and text encoders onto the unit sphere in $\mathbb{R}^d$ ($d = 512$ for CLIP ViT-B/32, $768$ for ViT-L/14). The score is
$$s(x,t) = \tau^{-1}\langle f_\theta(x), g_\phi(t)\rangle,$$
with learned temperature $\tau$ ($\approx 0.01$ at convergence for CLIP). Training minimises the symmetric InfoNCE loss over batches $B$ of size $|B|$ (32{,}768 for CLIP, 98{,}304 for some OpenCLIP runs):
$$\mathcal{L} = -\frac{1}{2|B|}\sum_{i\in B}\left[\log \frac{e^{s(x_i,t_i)}}{\sum_{j\in B} e^{s(x_i,t_j)}} + \log \frac{e^{s(x_i,t_i)}}{\sum_{j\in B} e^{s(x_j,t_i)}}\right].$$

**Scene as a role-filler set.** A scene is $S = \{(o_k, a_k)\}_{k=1}^{n}$ with objects $o_k \in \mathcal{O}$ and attributes/roles $a_k \in \mathcal{A}$. A *swap* $\sigma$ permutes attributes across objects, giving $S^\sigma$; the caption renderer $c(\cdot)$ produces $t = c(S)$ and $t^\sigma = c(S^\sigma)$. Crucially $t$ and $t^\sigma$ are **bag-of-words identical** — same multiset of tokens — when $\sigma$ is an attribute transposition.

**Binding margin.** For image $x$ depicting $S$:
$$\Delta(x,S,\sigma) = s(x, c(S)) - s(x, c(S^\sigma)), \qquad \mathrm{BA} = \Pr_{x,S,\sigma}[\Delta > 0].$$
Chance is $0.5$. $\mathrm{BA}$ is measured by evaluating both captions against a held-out image set with verified ground-truth annotations (e.g. synthetic renderers, or Visual Genome region graphs).

**Confound-adjusted binding accuracy.** Let $p_{\mathrm{LM}}(t)$ be the log-likelihood under a caption-only language model. Define the blind baseline $\mathrm{BA}_{\text{blind}} = \Pr[p_{\mathrm{LM}}(t) > p_{\mathrm{LM}}(t^\sigma)]$. The reportable quantity is the excess
$$\mathrm{BA}^{\ast} = \mathrm{BA} - \mathrm{BA}_{\text{blind}},$$
or, better, $\mathrm{BA}$ restricted to the stratum where $|p_{\mathrm{LM}}(t) - p_{\mathrm{LM}}(t^\sigma)| < \epsilon$. Most published numbers are $\mathrm{BA}$, not $\mathrm{BA}^\ast$.

**Bag-of-words null model.** If $g_\phi(t) \propto \sum_{w \in t} e_w$ for token embeddings $e_w$, then $g_\phi(t) = g_\phi(t^\sigma)$ exactly and $\Delta \equiv 0$, so $\mathrm{BA} = 0.5$. Any $\mathrm{BA}^\ast > 0$ measures departure from this null.

**Assumptions, and which fail.**
1. *Captions describe the image faithfully and completely.* Violated: web alt-text is partial and often non-descriptive.
2. *The swapped caption is equally plausible a priori.* Violated in most benchmarks — "the grass is green, the sky is blue" vs. its swap differ in prior likelihood, which is what SugarCrepe exposed.
3. *Ground-truth bindings are available.* Violated for natural images; Visual Genome attribute annotations are incomplete and noisy.
4. *A single global vector suffices.* This is the object of study, not an assumption to keep.

## 3. State of the Art

**Established.**
- *Winoground* (Thrush et al., CVPR 2022): 400 hand-curated pairs where two images and two captions share a word multiset. All tested VLMs score near or below chance on the group metric; humans 85.5%.
- *ARO* (Yuksekgonul et al., ICLR 2023) showed CLIP-family models behave like bags of words on relation and attribution swaps, and that a hard-negative fine-tune (NegCLIP) raises those scores substantially.
- *SugarCrepe* (Hsieh et al., NeurIPS 2023) established that ARO/CREPE/VALSE hard negatives are separable by **text-only** models: a blind grammar/plausibility baseline reaches high accuracy on the original sets, and much of the reported VLM gain is artifact exploitation. After de-biasing negatives with an LLM plus adversarial refinement, model scores drop.
- *Diwan et al.* (EMNLP 2022) reannotated Winoground and found only a minority of items isolate compositional reasoning; the rest require unusual images, commonsense, or fine visual detail.

**Claimed but unablated.** Papers reporting that structured or negation-aware objectives (NegCLIP-style data augmentation, SVLC-style caption rewriting, DAC-style dense caption distillation) "fix compositionality" generally report gains on ARO/VALSE — the benchmarks SugarCrepe showed to be hackable — and rarely report the blind-baseline delta on the same split. Treat these as benchmark numbers, not as demonstrated binding.

**Systems SOTA vs. theory SOTA.** Systems: generative/autoregressive VLMs and cross-attention rerankers beat dual encoders on swap tests, but they are not embeddings and do not give a metric space. Theory: there is no identifiability theorem for binding under InfoNCE with natural caption distributions.

## 4. What Is Known

- Winoground group score: CLIP ViT-B/32 ≈ 8%, best models at publication in single digits to low teens, against 16.67% random and 85.5% human (400 items, CVPR 2022).
- ARO: CLIP ViT-B/32 ≈ 59% on VG-Relation and ≈ 63% on VG-Attribution against 50% chance, at ~24k and ~29k test items (ICLR 2023).
- Blind text-only baselines reach roughly 80–90% on several ARO/CREPE splits — above the VLMs they were meant to test (SugarCrepe, NeurIPS 2023). Scale: full ARO/CREPE test sets.
- Order insensitivity: shuffling caption word order changes CLIP retrieval performance only marginally on COCO/Flickr30k (ICLR 2023), the direct empirical signature of the bag-of-words null.
- Accuracy on binding tasks degrades with object count $n$, with a serial-search-like signature analogous to human visual search under conjunction (Campbell et al., NeurIPS 2024), measured on synthetic scenes with small $n$ (roughly 2–8).
- Capacity is *not* the limit: $d = 512$ dimensions support superposed role-filler bindings for $n \lesssim 10$ at cross-talk variance $\approx (n-1)/d$ (tensor-product / vector-symbolic analysis, Smolensky 1990).

## 5. What Is Not Known

- **Theoretically open.** Whether the swap direction is identifiable from image-text contrastive learning at all. No theorem states conditions on the caption distribution and batch construction under which the minimiser of $\mathcal{L}$ must satisfy $\Delta > 0$. Existing compositional-generalisation identifiability results (Wiedemer et al., NeurIPS 2023) assume a compositional decoder and do not cover this setting.
- **Empirically open.** Whether $\mathrm{BA}^\ast$ improves with scale. No published scaling curve of an artifact-controlled binding metric against parameters, data, and batch size for a single training recipe. The experiment is runnable — it is a matter of nobody having run the sweep with a clean metric.
- **Methodologically blocked.** A binding metric on natural images with verified per-object attribute ground truth at scale. Visual Genome annotations are incomplete; synthetic renderers give clean ground truth but shift the distribution. Until this is fixed, $\mathrm{BA}$ and $\mathrm{BA}^\ast$ disagree by amounts comparable to the effect.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under the training objective, compounded by confounded measurement**.

Non-identifiability: in an InfoNCE batch, the in-batch negatives are captions of *other* images. The probability that some negative $t_j$ is a role-swap of $t_i$ — same tokens, different binding — is effectively zero in web corpora. So the loss surface is flat along the direction that would separate $g_\phi(c(S))$ from $g_\phi(c(S^\sigma))$: no gradient ever pushes them apart. Two parameter settings, one binding-sensitive and one bag-of-words, attain the same loss. Scale does not fix a flat direction.

Confounded measurement: the natural fix — build swap negatives — makes the negatives less plausible as English, and models then win by plausibility rather than by looking at the image. SugarCrepe quantified this: text-only baselines outscored VLMs. So progress reported on binding is not distinguishable from progress on caption likelihood without the blind control, which most papers omit.

## 7. Current Research (as of 2026)

- Artifact-controlled benchmark construction: LLM-generated, adversarially filtered negatives with published blind baselines (SugarCrepe line, U. Washington / AI2).
- Hard-negative and dense-caption objectives (NegCLIP, SVLC, DAC lines; Stanford, IBM/Weizmann, Tel Aviv).
- Slot- and object-centric encoders that give per-object vectors instead of one global vector, then aggregate for retrieval *(frontier — verify)*.
- Cognitive-science framing: serial binding, capacity limits, and search-like scaling in VLMs (Princeton / Campbell, Griffiths and collaborators).
- Mechanistic work locating binding-relevant directions in transformer residual streams — "binding IDs" in language models — being ported to vision-language models *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does artifact-controlled binding accuracy improve with contrastive scale, or is it flat?

**Scale.** Train OpenCLIP ViT-B/32 on DataComp-1B subsets at $\{128\text{M}, 400\text{M}, 1.28\text{B}\}$ samples-seen, three model widths (B/32, B/16, L/14) — nine runs, roughly $10^{3}$ A100-days total. Evaluate on a synthetic swap set: 20{,}000 rendered scenes, $n \in \{2,3,4,5\}$ objects, attributes drawn from 8 colours × 6 shapes × 4 spatial relations, with a matched caption pair $(t, t^\sigma)$ per scene, filtered so $|p_{\mathrm{LM}}(t) - p_{\mathrm{LM}}(t^\sigma)| < 0.1$ nats under Llama-3-8B.

**Control arms.** (a) The same nine runs evaluated on *object-presence* accuracy — "is there a red cube?" — to confirm recognition improves with scale. (b) A blind text-only scorer on the identical pairs, which must sit at $0.50 \pm 0.01$ by construction. (c) One run with 5% of batches augmented with swap negatives, to show the metric is movable.

**Deciding number.** The slope of $\mathrm{BA}^\ast$ at $n=3$ against $\log(\text{samples seen} \times \text{params})$. If the slope is below $0.01$ per decade of compute while object-presence accuracy gains $>0.05$ per decade, contrastive scale does not buy binding, and the flat-direction argument in §6 is empirically supported. A slope above $0.03$ per decade refutes it.

## 9. Key References

- **[Foundational]** Paul Smolensky. *Tensor product variable binding and the representation of symbolic structures in connectionist systems.* Artificial Intelligence, 1990.
- **[Foundational]** Anne Treisman, Garry Gelade. *A feature-integration theory of attention.* Cognitive Psychology, 1980.
- **[Foundational]** Klaus Greff, Sjoerd van Steenkiste, Jürgen Schmidhuber. *On the binding problem in artificial neural networks.* Preprint, 2020. — arXiv:2012.05208
- **[Foundational]** Alec Radford et al. *Learning transferable visual models from natural language supervision.* ICML, 2021. — arXiv:2103.00020
- **[SOTA]** Tristan Thrush et al. *Winoground: Probing vision and language models for visio-linguistic compositionality.* CVPR, 2022. — arXiv:2204.03162
- **[SOTA]** Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, James Zou. *When and why vision-language models behave like bags-of-words, and what to do about it?* ICLR, 2023. — arXiv:2210.01936
- **[SOTA]** Cheng-Yu Hsieh et al. *SugarCrepe: Fixing hackable benchmarks for vision-language compositionality.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.14610
- **[Analysis]** Anuj Diwan, Layne Berry, Eunsol Choi, David Harwath, Kyle Mahowald. *Why is Winoground hard? Investigating failures in visuolinguistic compositionality.* EMNLP, 2022. — arXiv:2211.00768
- **[Analysis]** Declan Campbell, Sunayana Rane, Tyler Giallanza, et al. *Understanding the limits of vision language models through the lens of the binding problem.* NeurIPS, 2024.
- **[Analysis]** Martha Lewis, Nihal V. Nayak, Peilin Yu, Qinan Yu, Jack Merullo, Stephen H. Bach, Ellie Pavlick. *Does CLIP bind concepts? Probing compositionality in large image models.* EACL Findings, 2024.
- **[Theory]** Thaddäus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, Wieland Brendel. *Compositional generalization from first principles.* NeurIPS, 2023.
- **[Survey]** Ranjay Krishna et al. *Visual Genome: Connecting language and vision using crowdsourced dense image annotations.* IJCV, 2017. — the annotation source most binding benchmarks are built from.

## 10. Worked Example

Take one scene: a red cube and a blue sphere, side by side.

- $t$ = "a red cube next to a blue sphere"
- $t^\sigma$ = "a blue cube next to a red sphere"

Token multisets are identical. Under the bag-of-words null, $g_\phi(t) = g_\phi(t^\sigma)$ and $\Delta = 0$ exactly, so $\mathrm{BA} = 0.5$.

**Capacity check.** Suppose the text encoder used tensor-product binding with random unit role and filler vectors in $d = 512$. Decoding a filler by unbinding gives signal $1$ and cross-talk with standard deviation $\sqrt{(n-1)/d}$. At $n = 2$ that is $\sqrt{1/512} \approx 0.044$ — a signal-to-noise ratio of about $23$. Even at $n = 8$, SNR $\approx 8.6$. **Capacity is not the obstruction.** A 512-dimensional embedding could carry the binding twenty times over.

**Gradient check.** CLIP-scale training sees $\sim 1.28 \times 10^{10}$ image-text pairs in batches of $3.28\times10^4$. For the swap direction to be constrained, some in-batch negative must be a token-multiset-preserving reordering of the positive. In LAION-style alt-text, captions that share a full token multiset with a different binding occur at a rate that is empirically indistinguishable from zero — in a $3.28\times10^4$ batch, the expected count is far below one. So across the whole run the model receives approximately **zero** gradient signal separating $t$ from $t^\sigma$.

**What the benchmarks then measure.** Evaluated on ARO-style pairs, CLIP nonetheless scores 59–63% rather than 50%. That excess is not evidence of binding: a text-only scorer on the same pairs reaches 80–90%, because "a blue cube next to a red sphere" is not the swap the annotator would naturally have written. Subtracting the blind baseline gives $\mathrm{BA}^\ast < 0$ — the model is worse than plausibility alone.

The obstruction is visible in three numbers: capacity SNR $\approx 23$ (ample), expected swap-negative gradient events $\approx 0$ (none), blind baseline $\approx 0.85$ vs. model $\approx 0.61$ (the metric is measuring the wrong thing). The representation could hold the binding; the objective never asks for it; and the standard evaluation cannot tell whether it did.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*