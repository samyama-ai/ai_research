---
id: 24-multimodal/attribute-object-binding
title: "Binding Problem for Attributes and Objects"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Binding Problem for Attributes and Objects

> **Topic:** Multimodal Models · **ID:** `24-multimodal/attribute-object-binding` · **Status:** open

## 1. Problem Statement

A scene contains several objects, each carrying several attributes (color, shape, material, size, position). A model *binds* correctly when it represents *which* attribute belongs to *which* object, not merely that both are present. Failure mode: an **illusory conjunction** — the system reports "a red cube and a blue sphere" for an image of a blue cube and a red sphere.

Three variants, routinely conflated:

- **Measurement.** Input: an image $x$ and two captions $c, c'$ that differ only by a permutation of attribute-to-object assignment. Output: a decision as to which caption matches. Solving the measurement variant means having a benchmark whose failures are attributable to binding and not to language priors, negative-caption implausibility, or object detection. This is the variant currently blocked.
- **Method.** Build a discriminative or generative multimodal model whose binding accuracy stays above a fixed threshold as the number of objects $n$ grows, at constant single-object accuracy.
- **Theory.** Characterise what a fixed-width, permutation-equivariant contrastive embedding can represent. Does a single vector of dimension $d$ trained with an InfoNCE objective on natural captions have the capacity — and the training signal — to encode $n$ bindings, or is the failure architectural rather than data-driven?

Solved would mean: binding accuracy flat in $n$ over $n \in [1,8]$ on controlled stimuli, and the mechanism identified (a named circuit or representational format), not just a benchmark number.

## 2. Formal Setting

A scene is a set of $n$ objects, $S = \{(o_i, a_i)\}_{i=1}^n$, where $o_i \in \mathcal{O}$ is an object identity and $a_i \in \mathcal{A}$ an attribute. A renderer $R$ maps $S \mapsto x$; a caption function $C$ maps $S \mapsto c$.

**Swap negative.** For a transposition $\pi$ acting on attributes only, $S^\pi = \{(o_i, a_{\pi(i)})\}$. The negative caption is $c^\pi = C(S^\pi)$. By construction the *bag* of tokens is identical: $\text{multiset}(c) = \text{multiset}(c^\pi)$. Any score computable from the unordered token set is uninformative here — this is the property that makes swaps the right probe.

**Binding accuracy** for a scorer $s(x,c)$ (CLIP-style cosine similarity, an ITM head logit, or a VLM's log-likelihood of the caption):
$$
\mathrm{BA}(n) \;=\; \mathbb{E}_{S,\pi}\big[\mathbb{1}\{s(R(S), C(S)) > s(R(S), C(S^\pi))\}\big], \qquad \text{chance} = 0.5 .
$$

**Feature accuracy** (the control): same scene, negative replaces an attribute with one absent from the scene, $a_j \notin \{a_i\}$. Call it $\mathrm{FA}(n)$. Binding is only implicated when $\mathrm{FA}(n)$ is high and $\mathrm{BA}(n)$ is not.

**Binding capacity.** $n^\star = \max\{n : \mathrm{BA}(n) \ge \tau\}$ for a fixed $\tau$ (e.g. $0.9$), measured with $\mathrm{FA}(n) \ge \tau$ held as a precondition. This is the single scalar the field lacks for any deployed model.

**Generative side.** For a text-to-image model $G$, sample $K$ images from $c$, extract $\hat S$ with an oracle (renderer ground truth is unavailable, so a VQA judge or detector is used), and report
$$
\mathrm{BAcc}_{\text{gen}} = \tfrac{1}{K}\sum_k \mathbb{1}\{\hat S_k = S\},
$$
which is *lower-bounded in error* by the judge's own binding error — the measurement is circular whenever the judge is itself a VLM.

**Assumptions, and where they break.**
1. *Swap negatives are equally plausible a priori.* Violated: "a red apple and a green leaf" vs the swap is decidable from language statistics alone. SugarCrepe (Hsieh et al., NeurIPS 2023) showed text-only models beat multimodal ones on the earlier ARO negatives.
2. *The judge is unbiased.* Violated: VLM judges fail binding themselves.
3. *Object detection is solved at the relevant $n$.* Violated for $n \ge 6$ with occlusion.
4. *Attributes are independent of identity.* Violated: shape and material co-vary in natural images, so $\mathrm{BA}$ on natural photos partly measures world knowledge.

## 3. State of the Art

**Established (ablated, reproduced).**
- Contrastive dual encoders behave partly like bags-of-words: Yuksekgonul et al., *When and why vision-language models behave like bags-of-words, and what to do about it?* (ICLR 2023) introduced ARO and NegCLIP; the NegCLIP fine-tuning gain replicates.
- The ARO/CREPE gains were substantially an artefact of negative-generation bias: Hsieh et al., *SugarCrepe* (NeurIPS 2023) — a text-only grammar/plausibility model scores far above chance on ARO-style negatives, so those numbers do not isolate binding.
- Winoground (Thrush et al., CVPR 2022) is hard for reasons beyond compositionality: Diwan et al. (EMNLP 2022) decomposed the set and found large fractions requiring unusual images, commonsense, or fine-grained localisation.
- Set-size dependence: Campbell et al., *Understanding the limits of vision language models through the lens of the binding problem* (NeurIPS 2024) reported human-like serial degradation with object count in frontier VLMs.

**Claimed but unablated.**
- That caption re-captioning (DALL·E 3, Betker et al. 2023 technical report) fixes binding. It raises T2I-CompBench-style attribute scores; no controlled $n$-sweep isolates binding from prompt coverage.
- That cross-attention interventions at inference (Attend-and-Excite, Chefer et al., SIGGRAPH 2023; StructureDiffusion, Feng et al., ICLR 2023) fix leakage. Reported as benchmark deltas on colour/shape/texture splits, not as capacity curves.

**Benchmark-number-only results.** Most T2I-CompBench (Huang et al., NeurIPS 2023) attribute-binding scores: a single BLIP-VQA-derived number, judge-limited, no per-$n$ breakdown.

## 4. What Is Known

- **Winoground, CVPR 2022 scale (400 items, ~10 models incl. CLIP ViT-B/32, UNITER-large):** best group score $10.5\%$ vs chance $16.67\%$; human $85.5\%$. Models are *below chance* on the joint criterion.
- **ARO, ICLR 2023 (~50k relation/attribution items):** CLIP near $50$–$60\%$ on VG-Relation, i.e. near chance on a binary task; NegCLIP fine-tuning lifts it above $80\%$.
- **SugarCrepe, NeurIPS 2023 (~7.5k items):** on ARO-Attribution a text-only plausibility model exceeds $80\%$ without seeing the image; on the debiased SugarCrepe SWAP splits, strong CLIP variants sit roughly $60$–$70\%$ — much lower than the ARO numbers implied.
- **Set size (NeurIPS 2024):** VLM accuracy on conjunction search degrades monotonically with the number of distractor objects, with near-ceiling performance at $n \le 2$.
- **Probing (Lewis et al., EACL 2024, *Does CLIP bind concepts?*):** on a controlled single-object colour/object grid, CLIP's representation of a conjunction is close to a linear combination of the parts — consistent with weak or absent binding structure.

Scales: all above are $10^2$–$10^4$ evaluation items on public checkpoints (CLIP ViT-B/32 through ViT-G, BLIP-2, LLaVA-class, GPT-4V-class). No published $n^\star$ for any model.

## 5. What Is Not Known

- **Methodologically blocked.** A binding benchmark on *natural* images with a bias-free negative distribution. SugarCrepe removed the worst artefacts, but the residual plausibility gap is unmeasured, and generative-side scoring uses VLM judges that fail the same test.
- **Empirically open.** The capacity curve $\mathrm{BA}(n)$ for $n = 1..8$ with $\mathrm{FA}(n)$ matched, on synthetic stimuli, for one open-weight model family across scale. Runnable today for a few hundred GPU-hours; unrun.
- **Empirically open.** Whether autoregressive VLMs with per-patch tokens bind better than pooled dual encoders *at matched training data*. Every existing comparison confounds architecture with data.
- **Theoretically open.** Whether InfoNCE on natural caption distributions has any gradient signal for binding — the swap negative essentially never appears in a natural batch, so the objective may be indifferent to $\pi$. No separation theorem either way.
- **Theoretically open.** Capacity bound: minimum $d$ for a fixed-width embedding to linearly decode $n$ bindings over $|\mathcal{O}| \times |\mathcal{A}|$ with error $\epsilon$. Tensor-product / superposition arguments (Greff et al. 2020) frame it; no tight result.

## 6. Why It Is Hard

**Primary obstruction: the evaluation does not measure what it names.** Constructing a swap negative that is (a) grammatical, (b) equally plausible as text alone, and (c) false of the image requires suppressing a prior the caption distribution encodes everywhere — objects and attributes are correlated in the world. The SugarCrepe result is the demonstration: an entire benchmark's headline gap was recoverable without images.

**Secondary: absent ground truth on the generative side.** For $G$'s outputs there is no renderer state, so binding is scored by a judge that has the same defect. The measurement error is correlated with the quantity measured, so a scoring improvement and a generator improvement are indistinguishable.

**Tertiary: non-identifiability of the cause.** $\mathrm{BA}$ can drop from vision-side crowding, from text-encoder order-insensitivity, or from the pooling operation. Without $\mathrm{FA}$ matched and both towers ablated, any single number is uninterpretable.

## 7. Current Research (as of 2026)

- **Debiased compositional benchmarks** — successors to SugarCrepe with adversarially-filtered, plausibility-matched negatives; the Washington/AI2 line around Hsieh, Krishna and collaborators.
- **Cognitive-science framing** — Princeton (Griffiths, Cohen, Campbell, Rane): serial-binding accounts, set-size manipulations, human baselines on the same stimuli.
- **Object-centric representations** — slot-based encoders (the Slot Attention lineage, Locatello et al., NeurIPS 2020) grafted onto multimodal encoders to give binding an architectural home. *(frontier — verify current results at scale.)*
- **Mechanistic localisation** — attention-head-level circuits for attribute routing in VLM decoders; claims of identified "binding heads" exist but reproduction across model families is thin *(frontier — verify)*.
- **Generation-side control** — layout-conditioned and attention-regularised diffusion; strong on 2-object prompts, unevaluated as a capacity curve.

## 8. Concrete Next Experiment

**Question.** Is attribute binding a capacity limit that scales away, or a representational-format failure invariant to scale?

**Stimuli.** Blender/CLEVR-style renders. $n \in \{1,2,3,4,6,8\}$, attributes = 8 colours $\times$ 3 shapes $\times$ 2 materials, uniform and *decorrelated by construction* (so language priors carry zero information). 2,000 scenes per $n$ = 12,000 images. Two negatives per scene: swap (binding) and replace (feature).

**Arms.** Discriminative: CLIP ViT-B/32, L/14, H/14, G/14 (same LAION data recipe, four scales — isolates scale). Generative-decoder: an open LLaVA-class model at 7B and 34B. **Control arm:** $n=1$ crops of the identical objects, each scored alone — establishes that every attribute is individually legible, i.e. $\mathrm{FA}$ ceiling. Second control: text-only scorer on the caption pair; must sit at $50.0 \pm 1\%$ or the stimulus set is broken.

**Cost.** Rendering + inference only, no training: ~200 A100-hours.

**Decisive number.** The slope $\beta$ in $\mathrm{BA}(n) = \alpha + \beta n$ fit over $n \in [2,8]$, on scenes where $\mathrm{FA}(n) \ge 0.95$, compared across the four CLIP scales.
- If $\beta$ moves toward $0$ monotonically with model scale (e.g. $-0.06$ at ViT-B/32 to $-0.01$ at ViT-G/14), binding is a capacity limit and the problem is a data/scale problem.
- If $\beta$ is statistically indistinguishable across an 18$\times$ parameter range ($\Delta\beta$ within $\pm 0.01$, $n{=}2000$ per cell gives SE $\approx 0.011$ on each $\mathrm{BA}$ point), the failure is the representational format, and scaling is ruled out as the fix.

## 9. Key References

- **[Foundational]** Anne Treisman, Garry Gelade. *A feature-integration theory of attention.* Cognitive Psychology, 1980.
- **[Foundational]** Klaus Greff, Sjoerd van Steenkiste, Jürgen Schmidhuber. *On the binding problem in artificial neural networks.* 2020. — arXiv:2012.05208
- **[Foundational]** Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, Candace Ross. *Winoground: Probing vision and language models for visio-linguistic compositionality.* CVPR, 2022.
- **[SOTA]** Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, James Zou. *When and why vision-language models behave like bags-of-words, and what to do about it?* ICLR, 2023.
- **[SOTA]** Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, Ranjay Krishna. *SugarCrepe: Fixing hackable benchmarks for vision-language compositionality.* NeurIPS Datasets and Benchmarks, 2023.
- **[SOTA]** Declan Campbell, Sunayana Rane, Tyler Giallanza, Nicolò De Sabbata, Kia Ghods, Amogh Joshi, Alexander Ku, Steven Frankland, Thomas Griffiths, Jonathan Cohen, Taylor Webb. *Understanding the limits of vision language models through the lens of the binding problem.* NeurIPS, 2024.
- **[Analysis]** Anuj Diwan, Layne Berry, Eunsol Choi, David Harwath, Kyle Mahowald. *Why is Winoground hard? Investigating failures in visuolinguistic compositionality.* EMNLP, 2022.
- **[Analysis]** Martha Lewis, Nihal Nayak, Peilin Yu, Qinan Yu, Jack Merullo, Stephen Bach, Ellie Pavlick. *Does CLIP bind concepts? Probing compositionality in large image models.* EACL Findings, 2024.
- **[Method]** Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, Daniel Cohen-Or. *Attend-and-Excite: Attention-based semantic guidance for text-to-image diffusion models.* SIGGRAPH / ACM TOG, 2023.
- **[Benchmark]** Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, Xihui Liu. *T2I-CompBench: A comprehensive benchmark for open-world compositional text-to-image generation.* NeurIPS, 2023.
- **[Architecture]** Francesco Locatello et al. *Object-centric learning with slot attention.* NeurIPS, 2020. — arXiv:2006.15055

## 10. Worked Example

Scene: a **blue cube** and a **red sphere**. Caption $c$ = "a blue cube and a red sphere"; swap $c^\pi$ = "a red cube and a blue sphere".

Take a CLIP-style scorer whose image embedding pools patch features and whose text embedding is close to additive over content words. Let $u_{\text{blue}}, u_{\text{red}}, u_{\text{cube}}, u_{\text{sphere}}$ be unit content-word directions and approximate
$$
t(c) \approx \tfrac{1}{4}\big(u_{\text{blue}} + u_{\text{cube}} + u_{\text{red}} + u_{\text{sphere}}\big).
$$
The same four terms appear in $t(c^\pi)$. Under strict additivity, $t(c) = t(c^\pi)$ exactly, so $s(x,c) - s(x,c^\pi) = 0$ and $\mathrm{BA} = 0.5$ by coin flip, for *any* image. The observed empirical value is not $0.5$ but roughly $0.6$–$0.7$ on debiased swap splits — the residual comes from word-order effects in the transformer text encoder, not from a verified image-side binding. That residual is the entire signal the field is arguing about.

Now the obstruction. Run the same pair on a natural photo: "a yellow banana and a brown table" vs "a brown banana and a yellow table". A text-only language model assigns higher likelihood to the first with no image at all. Whatever accuracy the multimodal model shows is upper-bounded in informativeness by that prior — which is exactly the leak SugarCrepe measured on ARO. Decorrelating the stimulus set removes the leak, but then the multimodal score collapses toward the additive prediction of $0.5$. The two conditions bracket the problem: where binding is measurable it is confounded, and where it is unconfounded the model does not do it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*