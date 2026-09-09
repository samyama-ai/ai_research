---
id: 24-multimodal/image-usage-attribution-measurement
title: "Do VLMs Actually Use the Image?"
topic: 24-multimodal
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Do VLMs Actually Use the Image?

> **Topic:** Multimodal Models · **ID:** `24-multimodal/image-usage-attribution-measurement` · **Status:** methodologically-blocked

## 1. Problem Statement

A vision-language model (VLM) takes an image $I$ and a text prompt $q$ and emits an answer $a$. It scores well on VQA, MMMU, ChartQA. The question: **how much of that score is caused by the image, and how much by language priors, question form, and answer-option statistics?**

Three variants, of very different difficulty:

- **Measurement.** Define a scalar $\rho(M, \mathcal{D}) \in [0,1]$ — the fraction of a model's task performance that is causally attributable to visual input — that is invariant to how the image is removed. *This is the blocked variant.* Every current estimator (blind baseline, noise image, black image, shuffled image) reports a different number for the same model and dataset, and there is no principle picking one.
- **Method.** Train a VLM whose $\rho$ is provably higher at fixed accuracy. Tractable, and partly addressed by counterfactual-balanced data.
- **Theory.** Prove that $\rho$ is identifiable from behavioural queries alone, or prove it is not. Open, and probably negative under weak assumptions.

Solving it means: a procedure that, given black-box query access to $M$ and a dataset, returns $\rho$ with stated error bars, and where two independent labs implementing the definition agree to within a few points.

## 2. Formal Setting

Let $M$ be a model inducing $p_M(a \mid I, q)$, evaluated on $\mathcal{D} = \{(I_i, q_i, a_i^\star)\}_{i=1}^N$ with metric $s(\hat a, a^\star) \in [0,1]$. Full performance:

$$S_{\text{full}} = \frac{1}{N}\sum_i \mathbb{E}_{\hat a \sim p_M(\cdot \mid I_i, q_i)}\, s(\hat a, a_i^\star).$$

**Ablation estimator.** Fix a replacement operator $T$ (black image, Gaussian noise, image dropped from the context, an image sampled from $\mathcal{D}$ at random). Then

$$\rho_T = \frac{S_{\text{full}} - S_{T}}{S_{\text{full}} - S_{\text{chance}}}, \qquad S_T = \frac{1}{N}\sum_i \mathbb{E}\, s(\hat a, a_i^\star) \text{ with } I_i \leftarrow T(I_i).$$

Measured by running the eval twice with greedy decoding, $N \geq 2{,}000$ for $\pm 2$ point binomial error. $S_{\text{chance}}$ is the uniform-over-options score for multiple choice; for open-ended generation it is undefined, which is itself part of the block.

**Counterfactual estimator.** For paired items $(I, I')$ with the *same* $q$ and different gold answers $a^\star \neq a'^\star$ (the VQA v2 construction, Goyal et al. 2017):

$$\rho_{\text{cf}} = \Pr\big[\hat a(I,q) = a^\star \wedge \hat a(I',q) = a'^\star\big],$$

measured as consistent-pair accuracy. This needs no off-manifold image and is the strongest available estimator, but requires human-authored pairs.

**Information-theoretic target.** The quantity people mean is the conditional mutual information $I(A; \Phi(I) \mid Q)$ under the model's own predictive distribution, where $\Phi$ is a visual encoder. It is not estimable at these dimensions without a variational bound whose looseness is unquantified.

**Mediation form.** Writing $h = \Phi(I)$ as mediator, the natural indirect effect $\mathrm{NIE} = \mathbb{E}[s(\hat a(h(I'), \text{ctx}(I))) - s(\hat a(h(I),\text{ctx}(I)))]$ requires patching visual token activations, which is white-box only.

**Assumptions, and which fail.**

1. *$T(I)$ stays on-manifold.* **Violated.** A black image is out-of-distribution for the connector; the drop can reflect distribution shift, not lost information.
2. *$q$ is uninformative about $a^\star$ alone.* **Violated** on nearly every benchmark: "Is there a clock?" is answered "yes" far above chance without an image.
3. *Answers are image-determined.* **Violated** for ambiguous or annotator-subjective items.
4. *Decoding is deterministic.* Holds under greedy decoding; breaks for sampled or reasoning-trace decoding, where per-item variance can exceed the ablation gap.

## 3. State of the Art

**Established (ablated, reproduced).**
- VQA v2 (Goyal et al., CVPR 2017): complementary-pair balancing cut the language-only baseline substantially and is the field's one reproduced counterfactual protocol.
- EMAP (Hessel & Lee, EMNLP 2020): project a multimodal model onto its best additive $f(I) + g(q)$ surrogate. On several benchmarks the projection loses under a point, meaning the reported score contains almost no cross-modal *interaction*. This is a measurement, not a claim, and it is the most rigorous existing tool.
- Cross-modal influence asymmetry (Frank, Bugliarello & Elliott, EMNLP 2021): in LXMERT/UNITER/ViLBERT, ablating text hurts image-side prediction far more than ablating image hurts text-side prediction.
- MMBench CircularEval (Liu et al., ECCV 2024): rotating answer-option order drops scores sharply, showing much of "vision" accuracy is option-position exploitation.

**Claimed but unablated.** Attention-map and Grad-CAM-style "the model looked at the right region" evidence. Sanity Checks (Adebayo et al., NeurIPS 2018) and ROAR (Hooker et al., NeurIPS 2019) already showed such maps can be insensitive to randomized weights and can overstate feature importance; no VLM paper has re-run those controls on visual-token attributions.

**Benchmark-number-only.** MMStar (Chen et al., NeurIPS D&B 2024) identifies "visual content unnecessary" items across six popular benchmarks by scoring text-only LLMs; the leakage estimate is a benchmark artifact count, not a per-model $\rho$. MMVP (Tong et al., CVPR 2024) reports failure rates on CLIP-blind pairs but does not decompose them into "didn't see" vs "saw and reasoned wrong".

## 4. What Is Known

- **Blind baselines are strong.** On the original VQA, language-only models exceeded 48% accuracy; balancing to VQA v2 was introduced precisely because of that (Goyal et al., 2017; ~1.1M QA pairs).
- **Leakage persists in modern suites.** MMStar's audit of six benchmarks (2024, evaluating 16 LLMs / 16 LVLMs) finds a large block of samples answerable text-only; MMMU's own paper reports text-only LLM baselines well above its 22.1% random-chance line, with GPT-4V at 55.7% validation.
- **Vision is often shallow.** MMVP (150 CLIP-blind pairs, 9 models): GPT-4V scores about 38.7% pair accuracy against ~95% human; several open 7B–13B models fall below the 25% random-pair line. "Vision Language Models Are Blind" (Rahmanzadehgervi et al., ACCV 2024, 7 primitive tasks) shows frontier models near chance on line-intersection and circle-overlap.
- **Object hallucination is systematic.** POPE (Li et al., EMNLP 2023, 3,000 items ×3 splits): instruction-tuned LVLMs answer "yes" to absent objects at high rates, with a strong yes-bias — evidence that answers are generated from priors under a nominal image condition.
- **Visual information is localized.** Basu et al. (NeurIPS 2024) causal-tracing LLaVA-style models finds visual information routed through a narrow band of early-mid layers; Neo et al. (2025) find visual tokens carry object-level content readable by the language head. Both are white-box, single-family results.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No agreed definition of $\rho$. $\rho_{\text{black}}$, $\rho_{\text{noise}}$, $\rho_{\text{shuffle}}$ and $\rho_{\text{no-image}}$ are not calibrated against each other, and no published study reports all four on the same model-dataset pair with the spread quantified. Until that spread is known, every "the model doesn't use the image" claim is estimator-dependent.
- **Theoretically open.** Whether $\rho$ is identifiable from black-box behaviour alone. A model that reads the image and a model that reads the question and confabulates a consistent image-shaped rationale can be behaviourally equal on any finite unpaired set. No impossibility theorem exists; neither does an identifiability result.
- **Empirically open.** Whether $\rho$ rises, falls, or is flat with scale. Runnable today across a 2B→90B open family on one fixed protocol; nobody has published the curve.

## 6. Why It Is Hard

**Confounded measurement with no ground truth.** The ablation $T$ mixes two effects that no experiment separates: (i) information the model genuinely used, and (ii) distribution shift from feeding the connector an input it never saw in training. A black image can *lower* score below blind-LLM performance — the model does worse than having no image at all — which makes $\rho_T$ exceed 1 and exposes the estimator as measuring shift, not usage.

Second: **non-identifiability**. Correct answers produced from priors and from vision are observationally identical per item. Distinguishing them needs counterfactual pairs, which are human-authored, expensive (VQA v2 cost a full annotation campaign), and exist for no reasoning-heavy benchmark.

Third: **the metric names the wrong thing**. "Accuracy on a visual benchmark" is reported as visual competence; MMStar and CircularEval show a substantial fraction is option statistics and text leakage.

## 7. Current Research (as of 2026)

- **Benchmark decontamination.** MMStar-style text-only screening is becoming standard pre-release filtering *(frontier — verify adoption breadth)*. Screening removes leaked items; it does not measure $\rho$ for a model.
- **Mechanistic localization.** Basu et al. (Maryland/Adobe), Neo et al., and BLIP causal-tracing work (Palit et al., ICCVW 2023) are converging on layer-resolved visual information flow. White-box, so inapplicable to API models.
- **Counterfactual image generation.** Using diffusion edits to synthesize $I'$ that flips the gold answer, replacing human pairing. Blocked on verifying that the edit changed only the queried attribute *(frontier — verify)*.
- **Vision-grounded RL / process rewards** claimed to raise image reliance in reasoning VLMs; the reliance claim is asserted from accuracy gains, not from any $\rho$ measurement *(frontier — verify)*.

## 8. Concrete Next Experiment

**The estimator-spread study.** Nobody has measured how much the answer depends on the question.

- **Scale.** One open model family at three sizes (e.g. 2B / 11B / 90B), on 3 datasets × 2,000 items: MMMU-val (leak-prone), VQA v2 complementary pairs (counterfactual ground truth available), MMVP (vision-critical). 18 full eval passes × 5 conditions ≈ 90 runs; under 500 GPU-hours on 8×H100 with greedy decoding.
- **Conditions.** Full image; black; Gaussian noise; image from a *different* item in the same dataset (on-manifold); image token sequence removed entirely.
- **Control arm.** The counterfactual score $\rho_{\text{cf}}$ on the VQA v2 pairs — the one estimator that never leaves the image manifold. It is the reference against which the four ablation estimators are graded.
- **Deciding number.** $\;\Delta_{\text{spread}} = \max_T \rho_T - \min_T \rho_T$ at fixed model and dataset. **If $\Delta_{\text{spread}} < 5$ points, the problem is not methodologically blocked** — pick any $T$, publish it as the standard, and the field moves to the empirical question of $\rho$ vs scale. **If $\Delta_{\text{spread}} > 15$ points**, ablation-based image-usage claims in the literature are uninterpretable, and only counterfactual-pair protocols should be accepted. Secondary readout: the rank correlation between each $\rho_T$ and $\rho_{\text{cf}}$ across the three model sizes — an estimator that does not even preserve ordering is unusable for model comparison.

## 9. Key References

- **[Foundational]** Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, Devi Parikh. *Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering.* CVPR, 2017. — arXiv:1612.00837
- **[Foundational]** Aishwarya Agrawal, Dhruv Batra, Devi Parikh, Aniruddha Kembhavi. *Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering.* CVPR, 2018. — arXiv:1712.00377
- **[Method]** Jack Hessel, Lillian Lee. *Does My Multimodal Model Learn Cross-Modal Interactions? It's Harder to Tell Than You Might Think!* EMNLP, 2020. — arXiv:2010.06572
- **[Method]** Stella Frank, Emanuele Bugliarello, Desmond Elliott. *Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers.* EMNLP, 2021. — arXiv:2109.04448
- **[SOTA]** Lin Chen et al. *Are We on the Right Way for Evaluating Large Vision-Language Models?* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2403.20330
- **[SOTA]** Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, Saining Xie. *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs.* CVPR, 2024. — arXiv:2401.06209
- **[SOTA]** Yifan Li et al. *Evaluating Object Hallucination in Large Vision-Language Models.* EMNLP, 2023. — arXiv:2305.10355
- **[SOTA]** Yuan Liu et al. *MMBench: Is Your Multi-modal Model an All-around Player?* ECCV, 2024. — arXiv:2307.06281
- **[Mechanistic]** Samyadeep Basu et al. *Understanding Information Storage and Transfer in Multi-modal Large Language Models.* NeurIPS, 2024. — arXiv:2406.04236
- **[Methodology]** Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, Been Kim. *Sanity Checks for Saliency Maps.* NeurIPS, 2018. — arXiv:1810.03292
- **[Methodology]** Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim. *A Benchmark for Interpretability Methods in Deep Neural Networks.* NeurIPS, 2019. — arXiv:1806.10758
- **[Benchmark]** Xiang Yue et al. *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.* CVPR, 2024. — arXiv:2311.16502
- **[Survey]** Zechen Bai et al. *Hallucination of Multimodal Large Language Models: A Survey.* 2024. — arXiv:2404.18930

## 10. Worked Example

Take a hypothetical but arithmetically realistic MMMU-val run, $N = 900$, 4-option items, $S_{\text{chance}} = 0.25$.

| Condition | Score | $\rho_T$ |
|---|---|---|
| Full image | 0.52 | — |
| No image in context | 0.39 | $(0.52-0.39)/(0.52-0.25) = 0.48$ |
| Black image | 0.31 | $0.78$ |
| Gaussian noise | 0.28 | $0.89$ |
| Random other image | 0.36 | $0.59$ |

$\Delta_{\text{spread}} = 0.89 - 0.48 = 0.41$ — **41 points**. The same model on the same data is "48% visually driven" or "89% visually driven" depending on a choice nobody has justified.

The obstruction is visible in the black-image row. Score 0.31 is *below* the no-image score 0.39. The model given a black image does worse than a model given no image at all. That gap of 8 points cannot be lost visual information — there was none to lose in either arm. It is the connector emitting an out-of-distribution embedding that actively corrupts the language prior. So $\rho_{\text{black}} = 0.78$ contains at least 8/27 = 30 points of pure distribution shift, and after correcting for it the estimate lands near $\rho_{\text{no-image}}$.

Now the part that cannot be fixed by arithmetic: of the 0.39 answered correctly with no image, some are option-statistics artifacts and some are genuine world knowledge that a sighted human would also use. Nothing in the table separates them. Only paired items with a flipped gold answer do — and MMMU has none. That is the block.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*