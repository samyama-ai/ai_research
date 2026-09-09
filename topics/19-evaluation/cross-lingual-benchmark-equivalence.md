---
id: 19-evaluation/cross-lingual-benchmark-equivalence
title: "Cross-Lingual Benchmark Equivalence After Translation"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Benchmark Equivalence After Translation

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/cross-lingual-benchmark-equivalence` · **Status:** open

## 1. Problem Statement

Almost every multilingual LLM benchmark is an English benchmark that was translated. MMLU, ARC, HellaSwag, GSM8K, MMLU-Pro and their derivatives are shipped in 40–200 languages by machine or professional translation. Scores are then compared across languages and read as a measure of the model's ability in each language.

The problem: **translation does not preserve what a test measures.** A score gap between English and Swahili can come from the model, from the translation, from the item, or from the cultural presupposition of the item — and the standard pipeline cannot tell these apart.

Three variants, of very different difficulty:

- **Measurement variant** (the one that is blocked). Given a source benchmark $B_{\text{en}}$ and its translation $B_\ell$, decide whether they measure the same latent construct on the same scale, so that a score difference is attributable to the model. This is *measurement invariance*, not translation quality.
- **Method variant.** Build a translation/adaptation procedure whose output is invariant by construction, or a post-hoc estimator that corrects a measured gap for non-invariance.
- **Theory variant.** Determine whether cross-lingual ability is identifiable at all from translated items, given that item difficulty and model ability enter the observed accuracy only through their difference.

Solving it means: a procedure that, given $B_{\text{en}}$, $B_\ell$ and a model set, returns a gap estimate with a stated bound on the contribution of non-equivalence — and that bound must be validated against a benchmark independently authored in $\ell$.

## 2. Formal Setting

Let $\mathcal{I}$ be a set of items $i=1..n$, and $\mathcal{L}$ a set of languages with English $e$ as source. A translation map $T_\ell: \mathcal{I}_e \to \mathcal{I}_\ell$ produces the target-language form.

**Observed quantity.** For model $m$, item $i$, language $\ell$, the graded response
$$Y_{m i \ell} \in \{0,1\}, \qquad \hat{p}_{m\ell} = \frac{1}{n}\sum_{i=1}^{n} Y_{mi\ell}$$
is what a leaderboard reports. Everything below is unobserved and must be estimated from $\{Y_{mi\ell}\}$.

**Latent model.** Assume a 2PL item-response model per language:
$$\Pr(Y_{mi\ell}=1) = c_i + (1-c_i)\,\sigma\!\big(a_{i\ell}(\theta_{m\ell} - b_{i\ell})\big)$$
with $\theta_{m\ell}$ the model's ability in $\ell$, $b_{i\ell}$ item difficulty, $a_{i\ell}$ discrimination, $c_i$ the guessing floor ($c_i = 1/4$ for four-way multiple choice — non-negligible: it compresses the informative range of $\theta$).

**Equivalence, stated exactly.** $B_\ell$ is *scalar-invariant* with respect to $B_e$ iff
$$a_{i\ell} = a_{ie} \ \text{ and } \ b_{i\ell} = b_{ie} \quad \forall i,$$
in which case $\hat{p}_{m e} - \hat{p}_{m\ell}$ is a monotone function of $\theta_{me}-\theta_{m\ell}$ alone. Weaker rungs (Meredith, *Psychometrika* 1993): *configural* (same items load on one factor), *metric* ($a$ equal, $b$ free), *scalar* (both equal). Only scalar invariance licenses mean comparison — the operation every multilingual leaderboard performs.

**Item-level non-equivalence** is differential item functioning:
$$\mathrm{DIF}_i = b_{i\ell} - b_{ie},$$
estimated by Mantel–Haenszel or by IRT with an *anchor set* $A \subseteq \mathcal{I}$ assumed invariant.

**Decomposition of the reported gap.** Writing $\Delta = \hat p_{me} - \hat p_{m\ell}$,
$$\Delta \;=\; \underbrace{g(\theta_{me}-\theta_{m\ell})}_{\text{ability}} \;+\; \underbrace{\tfrac{1}{n}\sum_i \partial_b \sigma \cdot \mathrm{DIF}_i}_{\text{translation + adaptation}} \;+\; \underbrace{\varepsilon}_{\text{sampling}}.$$
The catalog question is whether terms 1 and 2 are separable.

**Assumptions, with the violated ones flagged:**

- *Unidimensionality* of $\theta$. **Violated.** Translated MMLU loads on at least language competence, world knowledge, and script/tokenization handling.
- *Existence of an invariant anchor set.* **Unverifiable, and probably false.** No item is known a priori to translate cleanly.
- *Translation preserves the answer key.* **Violated.** Distractors collapse under translation when the source distinction is lexical in English (e.g. `affect`/`effect`) and absent in the target.
- *Cultural neutrality of the item.* **Violated.** Global-MMLU finds a large culturally-sensitive subset in MMLU.
- *Model responses are i.i.d. across items.* **Violated** by prompt-format and few-shot ordering effects, which themselves differ by language.

## 3. State of the Art

**Established.**

- *Translationese is a measurable, direction-dependent artifact.* Zhang and Toral, "The Effect of Translationese in Machine Translation Test Sets" (WMT 2019), and Graham, Haddow, Koehn, "Translationese in Machine Translation Evaluation" (EMNLP 2020), show system rankings change when the test-set translation direction changes. Reproduced across WMT years.
- *Translation artifacts inflate cross-lingual transfer numbers.* Artetxe, Labaka, Agirre, "Translation Artifacts in Cross-lingual Transfer Learning" (EMNLP 2020), show that translate-train vs. translate-test differences on XNLI are partly artifacts of the translation process rather than transfer ability. Ablated.
- *Independently-authored multilingual benchmarks exist and behave differently from translated ones.* TyDi QA (Clark et al., TACL 2020), XCOPA (Ponti et al., EMNLP 2020, human-adapted not literally translated), INCLUDE (Romanou et al., ICLR 2025, built from local exams).

**Claimed but unablated.**

- That professional translation removes the problem. Global-MMLU (Singh et al., 2024/2025) uses professional translation and still reports substantial cultural bias in the underlying items — translation quality and construct equivalence are distinct, and no study has ablated them against each other.
- That translate-test is a valid fallback for low-resource languages. MEGA (Ahuja et al., EMNLP 2023) reports translate-test beating in-language prompting on many low-resource tasks; this is a benchmark number, not evidence of construct equivalence, and the same artifact result of Artetxe et al. applies.

**Benchmark-number-only.** Nearly all published per-language MMLU tables. They report $\hat p_{m\ell}$ with no invariance check whatsoever. There is no widely used multilingual benchmark that ships DIF statistics.

## 4. What Is Known

- **MMLU's own items are noisy.** MMLU-Redux (Gema et al., 2024) hand-annotated 3,000 questions across 30 subjects and found roughly 6.5% erroneous overall, with error concentrated in specific subjects (Virology ~57%). Any translation inherits this floor. Scale: 3,000 items, 30 subjects.
- **A large fraction of MMLU is culturally local.** Global-MMLU (Singh et al.) annotates MMLU items for cultural sensitivity and reports that a substantial minority (~1 in 4 by their annotation) requires Western-specific knowledge; rankings of models shift between the culturally-sensitive and culturally-agnostic splits. Scale: 42 languages, thousands of annotated items.
- **Parallel, quality-controlled multilingual test sets are achievable.** Belebele (Bandarkar et al., ACL 2024): 900 reading-comprehension items in 122 language variants, built on FLORES-200 passages. FLORES-200 (NLLB Team, 2022): 204 languages, sentence-aligned. These fix *passage* parallelism; they do not certify *item* invariance.
- **Test-set translation direction changes system rankings** at WMT scale (thousands of segments, dozens of systems) — the strongest existing evidence that translated evaluation data is not neutral.
- **Anchor-set contamination biases DIF estimates.** Classic psychometrics: if anchor items themselves carry DIF, the estimated ability difference absorbs it. Documented since Holland and Thayer (1988) and the subsequent purification literature.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted operational definition of "the same benchmark in another language" for LLMs. Scalar invariance is the right formal target, but LLM benchmarks are administered to a handful of models, not thousands of examinees, so the $\theta$ estimates that DIF methods need are computed from a sample of models with correlated training data — the psychometric machinery is applied outside its sampling assumptions. Nobody has fixed this.
- **Theoretically open.** Whether $\theta_{m\ell}$ and $b_{i\ell}$ are jointly identifiable from translated items without an externally certified anchor set. The IRT indeterminacy $(\theta - b)$ is standard and resolved by fixing a scale within a language; across languages, translation moves both terms at once, and no identification result is known.
- **Empirically open.** The direct experiment — score $N$ models on a translated benchmark and on an independently authored benchmark of matched construct in the same language, and measure how much of the English-$\ell$ gap survives — is runnable today with INCLUDE, TyDi QA and local exam corpora. It has not been run as a controlled equivalence study at scale.
- **Empirically open.** Whether back-translation round-trip agreement, COMET score, or human adequacy rating predicts DIF at the item level. Plausible, untested.

## 6. Why It Is Hard

**Non-identifiability plus absent ground truth, compounding.**

The observable is $\sigma(a(\theta_{m\ell} - b_{i\ell}))$. A model that scores 12 points lower in Swahili is indistinguishable from a benchmark whose Swahili items are 12 points harder. Breaking the tie needs items known to be invariant — but "known to be invariant" is exactly what has no ground truth. Human-expert bilingual judgment of "same difficulty" has low reliability and, more importantly, judges *translation adequacy*, which is a different construct from *item difficulty for a language model*: a perfectly adequate translation can be far easier or harder for a model because tokenization, script frequency in pretraining, and answer-string surface form all shift.

The usual escape in psychometrics — pool thousands of examinees and purify the anchor set iteratively — fails here because the "examinee" population is a few dozen LLMs whose abilities are highly correlated (shared pretraining corpora, shared architectures), so the ability variance needed to estimate $a_{i\ell}$ is small and structured.

Secondary obstruction: **the evaluation does not measure what it names.** A multilingual MMLU score is reported as "knowledge in language $\ell$" while the items encode US-curriculum knowledge; a low score can be a correct report that the question is culturally foreign, not a deficit in $\ell$.

## 7. Current Research (as of 2026)

- **Locally-authored benchmarks as the escape route.** INCLUDE (EPFL and collaborators, ICLR 2025) from national exams; IrokoBench / AfriMMLU (Adelani and collaborators, Masakhane) for African languages; regional efforts for Indic, Arabic and Southeast Asian languages. Direction: avoid translation instead of certifying it.
- **Cultural-sensitivity annotation of existing benchmarks.** Global-MMLU (Cohere Labs and community annotators). Extends to splitting reported scores into culturally-agnostic and culturally-sensitive subsets.
- **IRT applied to LLM evaluation.** Growing use of item-response models for benchmark efficiency and item selection; cross-lingual DIF is the natural next step and is only beginning to appear *(frontier — verify)*.
- **Translation-quality-aware evaluation**, using COMET/xCOMET (Rei et al., Unbabel) to filter or weight items *(frontier — verify: filtering by MT quality does not target invariance and may not help)*.

## 8. Concrete Next Experiment

**Question.** How much of a reported English-vs-$\ell$ benchmark gap is non-equivalence rather than ability?

**Scale.** Two languages with strong independently-authored exam corpora — say Turkish and Indonesian (both present in INCLUDE). 500 items per language from the local corpus; 500 MMLU items translated professionally into the same language; 500 English MMLU items. $M = 30$ models spanning ability (7B to frontier, at least 5 distinct pretraining families, to break ability correlation). Cost: $30 \times 2500$ items $\approx$ 75k multiple-choice queries, under \$500 at 2026 API prices. This is a cheap experiment.

**Control arm.** The English source items scored by the same 30 models, plus a *placebo translation* arm: English items paraphrased into English by the same pipeline (translate to $\ell$ and back, or LLM-paraphrase) with no language change. The placebo isolates "damage done by rewriting" from "damage done by changing language."

**Deciding number.** Fit a 2PL model per arm, link scales with an iteratively purified anchor set, and report
$$\rho = \frac{\mathrm{Var}_i(\widehat{\mathrm{DIF}}_i)}{\mathrm{Var}_i(\hat b_{ie})},$$
the share of item-difficulty variance introduced by translation relative to native item-difficulty spread, together with the correlation $r$ between model rankings on translated-MMLU-$\ell$ and locally-authored-$\ell$.

**Decision rule.** If $r \geq 0.95$ and $\rho \leq 0.1$ after subtracting the placebo arm's $\rho$, translated benchmarks are usable for ranking and the problem downgrades to "solved for ranking, open for absolute scores." If $r < 0.9$, per-language leaderboard tables on translated benchmarks should not be published without DIF statistics.

## 9. Key References

- **[Foundational]** William Meredith. *Measurement invariance, factor analysis and factorial invariance.* Psychometrika, 1993.
- **[Foundational]** Paul W. Holland and Dorothy T. Thayer. *Differential item performance and the Mantel-Haenszel procedure.* In *Test Validity*, Erlbaum, 1988.
- **[Foundational]** Mikel Artetxe, Gorka Labaka, Eneko Agirre. *Translation Artifacts in Cross-lingual Transfer Learning.* EMNLP, 2020.
- **[Foundational]** Mike Zhang and Antonio Toral. *The Effect of Translationese in Machine Translation Test Sets.* WMT, 2019.
- **[Foundational]** Yvette Graham, Barry Haddow, Philipp Koehn. *Translationese in Machine Translation Evaluation.* EMNLP, 2020.
- **[SOTA]** Lucas Bandarkar et al. *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.* ACL, 2024.
- **[SOTA]** Angelika Romanou et al. *INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledge.* ICLR, 2025.
- **[SOTA]** Shivalika Singh et al. *Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation.* Cohere Labs / community, 2024–2025.
- **[SOTA]** Aryaman Gema et al. *Are We Done with MMLU?* (MMLU-Redux), 2024.
- **[SOTA]** NLLB Team. *No Language Left Behind: Scaling Human-Centered Machine Translation.* Meta AI, 2022. (FLORES-200)
- **[Survey]** Kabir Ahuja et al. *MEGA: Multilingual Evaluation of Generative AI.* EMNLP, 2023.
- **[Survey]** Jonathan H. Clark et al. *TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages.* TACL, 2020.
- **[Survey]** Edoardo Maria Ponti et al. *XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning.* EMNLP, 2020.
- **[Survey]** Alexis Conneau et al. *XNLI: Evaluating Cross-lingual Sentence Representations.* EMNLP, 2018.
- **[Survey]** Ricardo Rei et al. *COMET: A Neural Framework for MT Evaluation.* EMNLP, 2020.

## 10. Worked Example

Take a 200-item MMLU slice, English and a professionally translated Swahili version, scored by 30 models.

Suppose the headline numbers are $\hat p_{e} = 0.78$ and $\hat p_{\mathrm{sw}} = 0.60$ averaged over models — an 18-point gap, the kind routinely reported as "the model is much weaker in Swahili."

Now decompose. Fit a 2PL per language and estimate DIF. Two item types drive it.

**Item A — distractor collapse.** An English item asks which term describes a policy's *effect* on inflation, with `affect` among the distractors. Swahili has no lexical analogue; a faithful translation renders both as forms of *athari*, and the item now has three viable options instead of four. Guessing floor moves from $c=0.25$ to $c=0.33$. Held ability fixed at $\theta=0$ and $b=0$, expected accuracy rises from $0.25 + 0.75(0.5) = 0.625$ to $0.33 + 0.67(0.5) = 0.665$. The item got *easier* in Swahili — $\mathrm{DIF}_A \approx -0.2$ logits.

**Item B — cultural presupposition.** A US-high-school civics item about the Electoral College. English $b = -0.4$ (easy, heavily represented in pretraining); Swahili $b = +1.6$, because the Swahili-language pretraining corpus barely discusses it. $\mathrm{DIF}_B \approx +2.0$ logits. This is a *real* deficit in Swahili-language knowledge — or a *fake* one, if you meant to measure reasoning rather than familiarity with US institutions. The statistic cannot say which.

Say 26 of 200 items look like B and 9 like A. Drop all 35 flagged items and rescore: the gap falls from 18 points to, say, 11.

**Here is the obstruction, made visible.** That 11 is not a corrected estimate. It was produced by an anchor set — the 165 unflagged items — that was *defined by the DIF procedure itself*, using ability estimates $\hat\theta_{m,\mathrm{sw}}$ computed from those same items. If the unflagged 165 carry a uniform $+0.3$ logit of translation difficulty (entirely plausible: every item is longer in Swahili, tokenizes into more subwords, and sits in a thinner slice of pretraining), the procedure absorbs that constant into $\hat\theta_{m,\mathrm{sw}}$ and reports zero DIF for all 165. The 11-point residual and a true 4-point ability gap plus a 7-point uniform translation penalty are **the same likelihood**. No amount of additional model scoring on this item set distinguishes them — only an external, non-translated Swahili instrument does. That is why Section 8's control arm is the whole experiment.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*