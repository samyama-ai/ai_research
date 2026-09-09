---
id: 24-multimodal/faithful-multimodal-chain-of-thought
title: "Multimodal Chain-of-Thought That Is Faithful"
topic: 24-multimodal
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multimodal Chain-of-Thought That Is Faithful

> **Topic:** Multimodal Models · **ID:** `24-multimodal/faithful-multimodal-chain-of-thought` · **Status:** methodologically-blocked

## 1. Problem Statement

A vision-language model (VLM) takes an image $x_v$ and a question $x_t$, emits a natural-language reasoning trace $c$, then an answer $a$. The trace typically makes **visual claims** ("the left gauge reads 40", "the red block is on top"). The problem: guarantee, or at least measure, that those claims describe the computation that actually produced $a$.

Three variants, of very different difficulty:

- **Measurement.** Define a statistic $F(M, x)$ that is high exactly when the trace's visual claims are causal for the answer, and that cannot be gamed by a model that emits plausible-sounding visual claims post hoc. *This is the blocked one.*
- **Method.** Train or decode so that $F$ is high without paying an accuracy cost. Contingent on the measurement.
- **Theory.** Identify conditions on architecture and training under which the trace is provably an execution trace rather than a commentary. Essentially untouched.

Solving it means: an intervention on the image region the trace cites changes the answer in the direction the trace implies, and an intervention on regions it does not cite does not — at rates separated well beyond a text-only baseline that never looked at the pixels.

## 2. Formal Setting

Model $M_\theta: (x_v, x_t) \mapsto (c, a)$, autoregressive over a joint token sequence, image encoded as $n_v$ visual tokens.

**Visual claim extraction.** Parse $c$ into claims $\{(r_i, \phi_i)\}_{i=1}^{k}$, where $r_i \subseteq x_v$ is a spatial region and $\phi_i$ a predicate. Measured by an auxiliary grounder (open-vocabulary detector or a second VLM), so $r_i$ carries annotation noise; report inter-annotator agreement on a human-labelled subset.

**Counterfactual faithfulness.** Let $\mathrm{do}(r \to r')$ be an image edit replacing region $r$ with content making $\phi_i$ false. Define

$$F_{\mathrm{cf}}(M,x) = \frac{1}{k}\sum_{i=1}^{k} \Big[ \mathbb{1}\big(a(\mathrm{do}(r_i)) \neq a\big) - \mathbb{1}\big(a(\mathrm{do}(\tilde r_i)) \neq a\big) \Big],$$

with $\tilde r_i$ a control region matched on area, saliency and edit magnitude but not cited. $F_{\mathrm{cf}} \in [-1,1]$; the subtraction is what separates faithfulness from generic edit sensitivity.

**Trace-causality.** Following the perturbation family of Lanham et al. (2023): truncate $c$ at step $j$, or corrupt claim $j$ to $\neg\phi_j$, and measure $P(a \mid c_{<j}, \neg\phi_j)$. Faithful traces should switch answers when a load-bearing step is corrupted.

**Modality attribution.** $\mathrm{MM\text{-}SHAP}(x)$ = fraction of total Shapley value over input tokens assigned to the $n_v$ visual tokens (Parcalabescu & Frank, ACL 2023). Measured by sampling $\sim10^3$ coalitions per instance; cost is the binding constraint.

**Assumptions, and which are violated.**

1. *Edits are minimal and on-manifold.* Violated: inpainting shifts low-level statistics, and VLMs are sensitive to those shifts independently of semantics — this inflates both terms of $F_{\mathrm{cf}}$ unequally.
2. *A unique region grounds each claim.* Violated for relational, counting, and chart claims, where the support is diffuse.
3. *The trace is the computation.* Violated by construction: $a$ can be computed in the forward pass before $c$ is emitted, and $c$ conditioned on it.
4. *Answer changes are attributable to the edit.* Violated by VLM answer instability under resampling; needs a same-image, no-edit resample control.

## 3. State of the Art

**Empirical SOTA — established.**
- Text-only CoT unfaithfulness is a reproduced finding. Turpin et al. (NeurIPS 2023) show accuracy drops of up to ~36 points on BIG-Bench Hard tasks when biasing features are inserted, with the bias essentially never mentioned in the trace (GPT-3.5, Claude 1.0).
- Lanham et al. (2023) show faithfulness under truncation/corruption is **non-monotonic in scale**: the largest models in their sweep were *less* faithful, holding the answer fixed as the trace is mutilated.
- Parcalabescu & Frank (ACL 2024) argue that most published "faithfulness" tests measure **self-consistency**, not faithfulness — the same intervention family scores a model that is consistently wrong about its own process. This is the direct source of the *methodologically-blocked* status.

**Empirical SOTA — claimed but unablated.**
- Multimodal-CoT (Zhang et al., TMLR 2024) reports 91.68% on ScienceQA with a <1B model, above GPT-3.5's 75.17%. This is an *accuracy* result; no counterfactual test shows the generated rationale is causal.
- Visual CoT (Shao et al., NeurIPS 2024 D&B) supervises bounding-box "attention" steps. Gains are benchmark numbers; the boxes are not shown to be load-bearing under region ablation.
- RL-trained multimodal reasoners (2025–2026, R1-style) report large MathVista/MMMU gains. Faithfulness is not part of the reward, and long traces give more surface for post-hoc rationalisation, not less. *(frontier — verify)*

**Theory SOTA.** There is none specific to multimodality. The definitional frame is Jacovi & Goldberg (ACL 2020): faithfulness is not binary and should be evaluated on a graded scale — still the strongest available statement.

## 4. What Is Known

- **Saliency is not an explanation.** Adebayo et al. (NeurIPS 2018) show several saliency methods are invariant to randomising model weights. Attention maps and Grad-CAM overlays cannot certify a trace.
- **VLMs answer without looking.** POPE (Li et al., EMNLP 2023) finds object-hallucination rates where models affirm absent objects at high rates under adversarial sampling; language priors dominate. HallusionBench (Guan et al., CVPR 2024) reports GPT-4V question-pair accuracy around 31%, far below its single-question accuracy — the model's stated visual reasoning does not survive a paired counterfactual image.
- **Low-level vision is weak.** Rahmanzadehgervi et al. (ACCV 2024) report frontier VLMs near 73% average on tasks (counting line intersections, overlapping circles) humans do at ~100% — so traces asserting such percepts are frequently asserting something the model cannot compute.
- **Bias suppression is trainable.** Chua et al. (2024) show bias-augmented consistency training reduces biased reasoning substantially on text CoT, and generalises across held-out biases — evidence the method variant is tractable *once* the measurement is fixed.
- **Caption-level unfaithfulness has a long record.** Hendricks et al. (ECCV 2018) show captioners predicting gender from context rather than the person region.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No accepted operational definition separates (a) a trace that describes the computation from (b) a trace that is a consistent post-hoc story. Every deployed test — truncation, corruption, paraphrase, early-answering — is passed by a model with a stable but non-explanatory narrator. Parcalabescu & Frank name this directly.
- **Methodologically blocked (secondary).** Region grounding of visual claims has no ground truth for relational, temporal, or chart-derived claims, so $r_i$ is undefined for a large slice of real traces.
- **Empirically open.** Whether $F_{\mathrm{cf}}$ improves, degrades, or is flat from 7B to 200B+ under matched training has not been measured at scale; the text-only analogue (Lanham) suggests degradation. Runnable today on open weights.
- **Empirically open.** Whether outcome-reward RL on multimodal reasoning changes $F_{\mathrm{cf}}$ at fixed accuracy. No published controlled comparison.
- **Theoretically open.** No non-identifiability theorem, and no proof that one is impossible: is there any observational or interventional protocol, without weight access, that distinguishes an execution trace from a rationalisation for a fixed model class?

## 6. Why It Is Hard

The obstruction is **non-identifiability under behavioural testing, compounded by absent ground truth for the visual referent.**

Two models — one that computes $a$ from region $r$ then reports it, one that computes $a$ from a language prior then generates a plausible $r$-claim — can be made behaviourally identical on every input-output test, because the second model's claim-generator can be conditioned on the same features that drive $a$. Interventions on the *image* break the tie only if the edit is semantically minimal, and minimal image edits do not exist: inpainting changes texture statistics, and VLMs respond to those. So the control term $\mathbb{1}(a(\mathrm{do}(\tilde r_i)) \neq a)$ is not a clean control, and $F_{\mathrm{cf}}$ inherits an unquantified bias.

Second obstruction: the evaluation does not measure what it names. ScienceQA-style rationale scoring compares generated rationales to reference rationales by ROUGE/BLEU or LLM judge. A trace can score high on similarity to a correct human explanation while being causally inert in the model that emitted it. Higher rationale-similarity is currently *rewarded*, which selects for fluent rationalisation.

## 7. Current Research (as of 2026)

- **Interventional benchmarks.** Paired-image designs (HallusionBench; VALSE-style foils) are being extended to trace-level rather than answer-level checks. *(frontier — verify)*
- **Mechanistic grounding.** Activation patching from visual-token positions into the trace-generation positions, to test whether cited-region tokens are causally read at the step where the claim is emitted. Small-scale, LLaVA-family only. Most promising route past behavioural non-identifiability.
- **Reward-side work.** Adding process rewards over visual grounding to RLVR pipelines; risk is reward hacking of the grounder. *(frontier — verify)*
- **Metric work.** MM-SHAP-style modality attribution as a faithfulness prior — a trace claiming heavy visual reliance while MM-SHAP shows <10% visual contribution is prima facie unfaithful. Parcalabescu & Frank (Heidelberg/Mainz) remain the main group on definitional critique; Anthropic's alignment-stress-testing line owns the perturbation-test family.

## 8. Concrete Next Experiment

**Question.** Does the cited-region edit flip answers more than a matched control edit, at rates above the model's own text-only prior?

**Scale.** $N = 2{,}000$ items from a *programmatically generated* image set (rendered scenes and synthetic charts) where the causal visual variable is known by construction, and every edit is a re-render — eliminating the inpainting artefact confound. Five models: Qwen2.5-VL-7B/72B, InternVL-8B, LLaVA-OneVision-7B, plus one frontier API model. Three seeds. Estimated cost: ~10 GPU-days plus <$500 API.

**Arms.**
1. **Treatment:** re-render the cited region $r_i$ to falsify $\phi_i$.
2. **Control A (matched non-cited):** re-render $\tilde r_i$, matched on pixel area and rendering delta.
3. **Control B (blind text-only):** same model, image replaced by grey canvas, trace and answer regenerated — measures the language prior's answer distribution.
4. **Control C (no-edit resample):** answer instability floor at temperature 0.7.

**Deciding number.** $F_{\mathrm{cf}}$ with bootstrap 95% CI, after subtracting the Control C floor. **$F_{\mathrm{cf}} > 0.50$ with CI excluding 0.30** for a model whose Control B accuracy is at chance would be the first positive evidence of causally load-bearing multimodal CoT. $F_{\mathrm{cf}} < 0.15$ across all five models — the outcome the text-only literature predicts — establishes that current multimodal traces are decorative, and makes the method variant the priority.

## 9. Key References

- **[Foundational]** Alon Jacovi, Yoav Goldberg. *Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?* ACL, 2020. — arXiv:2004.03685
- **[Foundational]** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS, 2023. — arXiv:2305.04388
- **[Foundational]** Tamera Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic technical report, 2023. — arXiv:2307.13702
- **[SOTA]** Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, Alex Smola. *Multimodal Chain-of-Thought Reasoning in Language Models.* TMLR, 2024. — arXiv:2302.00923
- **[SOTA]** Hao Shao et al. *Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning.* NeurIPS Datasets & Benchmarks, 2024.
- **[SOTA]** Letitia Parcalabescu, Anette Frank. *MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks.* ACL, 2023. — arXiv:2212.08158
- **[Critique]** Letitia Parcalabescu, Anette Frank. *On Measuring Faithfulness or Self-consistency of Natural Language Explanations.* ACL, 2024. — arXiv:2311.07466
- **[Benchmark]** Pan Lu et al. *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering.* NeurIPS, 2022. — arXiv:2209.09513
- **[Benchmark]** Tianrui Guan et al. *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models.* CVPR, 2024.
- **[Benchmark]** Yifan Li et al. *Evaluating Object Hallucination in Large Vision-Language Models.* EMNLP, 2023. — arXiv:2305.10355
- **[Evidence]** Pooyan Rahmanzadehgervi et al. *Vision Language Models Are Blind.* ACCV, 2024. — arXiv:2407.06581
- **[Evidence]** Julius Adebayo et al. *Sanity Checks for Saliency Maps.* NeurIPS, 2018. — arXiv:1810.03292
- **[Method]** James Chua et al. *Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought.* 2024. — arXiv:2403.05518

## 10. Worked Example

A rendered bar chart, four bars, values $[40, 65, 30, 65]$. Question: *"Which bar is tallest?"* Answer key: tie between B and D.

The model's trace: *"Bar B rises to about 65 on the y-axis, higher than A at 40, C at 30, and D at roughly 60. So the answer is B."*

Every number is close. The trace looks grounded. Now run the arms on 500 such charts:

| Arm | Answer changes |
|---|---|
| Treatment: re-render bar B to 20 | 61% |
| Control A: re-render bar C (never load-bearing, cited only in passing) to 20 | 44% |
| Control B: grey canvas, no image | answers "B" 78% of the time |
| Control C: no edit, resample | 12% |

$F_{\mathrm{cf}} = 0.61 - 0.44 = 0.17$; minus the 0.12 instability floor, $\approx 0.05$.

The obstruction is visible in two places. First, Control B: with **no image at all**, the model says "B" 78% of the time — leftmost-tall is a rendering-prior artefact, so the trace's confident reading of B's height is largely predicted before any pixel is consulted. Second, Control A at 44%: editing a bar the trace treats as irrelevant flips the answer almost as often as editing the bar it named. The trace's citation structure carries about 5 points of causal signal out of a 61-point response.

And the trace's one *error* — "D at roughly 60", which converts a tie into a unique answer — is the step that determines the output. A rationale-similarity metric scores this trace highly: correct format, correct axis reading for three of four bars, correct-looking arithmetic. Accuracy on the benchmark is graded against a key that itself often lists "B". Nothing in the standard pipeline detects that the deciding claim is fabricated. That is the methodological block, not a modelling failure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*