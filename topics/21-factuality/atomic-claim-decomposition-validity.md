---
id: 21-factuality/atomic-claim-decomposition-validity
title: "Atomic Claim Decomposition Validity"
topic: 21-factuality
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Atomic Claim Decomposition Validity

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/atomic-claim-decomposition-validity` · **Status:** methodologically-blocked

## 1. Problem Statement

Nearly every long-form factuality metric — FActScore, SAFE, VeriScore, FacTool — first breaks a model response into "atomic facts," verifies each against evidence, and reports the fraction supported. The decomposition step is treated as a preprocessing detail. It is not: it defines the denominator of the score.

**Input.** A response $y$ (a paragraph to a few thousand tokens) and a decomposition operator $D$, usually an LLM prompted with a handful of demonstrations.

**Output.** A multiset of claims $D(y) = \{c_1,\dots,c_n\}$, each intended to be *atomic* (one verifiable proposition), *decontextualized* (checkable without $y$), and *faithful* (entailed by $y$).

**Objective.** Determine whether the induced factuality score is a property of $y$ or a property of $D$.

Three variants, different difficulty:

- **Measurement (the blocked one).** Define validity for $D$ without circularity. There is no gold decomposition — no annotator produces the same claim set twice, and no theory says how many atoms a sentence contains.
- **Method.** Build a $D$ whose downstream score is stable under paraphrase of $y$ and under swapping $D$ for a different valid $D'$. Runnable today; largely unrun at scale.
- **Theory.** Characterize the class of aggregation functions over $D(y)$ that are invariant to admissible re-decompositions. Open; likely a short impossibility result for the mean-of-atoms family.

Solving it means: a stated validity criterion, a measurement of it that does not depend on the extractor being measured, and a demonstration that two independently-built valid extractors rank models the same way.

## 2. Formal Setting

Let $y$ be a response, $\mathcal{E}$ an evidence corpus, and $V:\mathcal{C}\times\mathcal{E}\to\{0,1\}$ a verifier (in practice an LLM plus retrieval; $V$ has its own 5–15% error rate on entailment, measured on WICE-style data). The standard score is

$$ F(y;D,V) = \frac{1}{|D(y)|}\sum_{c\in D(y)} V(c,\mathcal{E}). $$

Four properties, each as it would actually be measured:

- **Faithfulness.** $y \models c$ for all $c\in D(y)$. Measured by a human or NLI judge over sampled $(y,c)$ pairs; report the rate $\phi = \Pr[y\models c]$.
- **Coverage.** $\bigwedge_i c_i \models y$. Measured by asking annotators to reconstruct $y$'s informational content from $D(y)$ and marking dropped propositions; report $\kappa$ = fraction of source propositions recovered. Note the circularity: "source propositions" is itself a decomposition.
- **Decontextualization.** $V(c,\mathcal{E})$ is invariant to whether the judge sees $y$. Measured as $\Pr[V(c\mid y) = V(c)]$ over a sample.
- **Granularity / non-redundancy.** No formal handle. Two extractors may return $n=8$ and $n=21$ for the same paragraph and both satisfy the three properties above.

The failure is here. Let $D'$ refine $D$ by splitting some claim $c$ into $k$ entailed sub-claims. If $c$ was unsupported and $j$ of the $k$ pieces are supported, the score moves by
$$ \Delta F = \frac{S+j}{n+k-1} - \frac{S}{n}, \qquad S=\textstyle\sum_i V(c_i), $$
which for $n=10$, $S=7$, $k=3$, $j=2$ gives $F: 0.70 \to 0.75$. Both decompositions are faithful and both cover $y$. The score changed by 5 points because of a formatting choice.

**Assumptions known to be violated in practice.** (i) Claim independence — atoms from one sentence share presuppositions, so $V$ outcomes are correlated, not i.i.d.; confidence intervals reported by these metrics are therefore too narrow. (ii) Equal weight per atom — a birth year and a Nobel Prize count the same. (iii) Verifier soundness — $V$ is an LLM, so $F$ compounds decomposition error with verification error and the two are not separately identified from the final number. (iv) Atomicity is well defined — it is not; see §6.

## 3. State of the Art

**Established.**
- FActScore (Min et al., EMNLP 2023) fixed the pipeline: InstructGPT-based extraction, retrieval against Wikipedia, mean-of-atoms. Its estimator was validated against human atoms on biography generation and reported error under ~2% for the best automated configuration — on that domain, with that extractor.
- SAFE (Wei et al., NeurIPS 2024) replaced retrieval with Google Search and added an explicit revision step for decontextualization; reported ~72% agreement with crowdworkers on a 100-item sample and higher agreement with researcher-adjudicated labels, at ~20× lower cost.
- VeriScore (Song, Kim & Iyyer, EMNLP 2024) restricted extraction to *verifiable* claims, showing that generic extractors emit large numbers of unverifiable or trivially-true atoms that inflate scores on non-biographical text.
- Claimify (Metropolitansky & Larson, 2025) made ambiguity handling explicit: the extractor declines to resolve referents it cannot disambiguate from context, rather than guessing.

**Claimed but unablated.** Every one of these reports its own extractor's quality using its own criteria and its own judge. No paper holds the verifier fixed, swaps only the extractor across all four systems, and reports the resulting model *ranking* changes. Absolute FActScore/VeriScore numbers quoted across papers are therefore benchmark numbers, not comparable measurements.

**Directly on the question.** Wanner et al. (*SEM 2024) compared decomposition strategies and showed FActScore-style results shift with the decomposition method; Gunjal & Durrett (EMNLP Findings 2024) argued for "molecular" rather than atomic facts, because full decontextualization forces the extractor to add entity-disambiguating content that was never in $y$ — trading faithfulness for checkability. Jiang et al. (CORE, 2024) added a redundancy filter over extracted claims and showed precision estimates move once duplicates are removed.

## 4. What Is Known

- **Extractor choice moves scores by several points.** Wanner et al. report FActScore differences of roughly 5–10 points on the same generations, on the order of the gaps between model checkpoints those scores are used to rank. Scale: hundreds of biography-style generations.
- **Claim counts differ by ~2–3× across extractors** on the same paragraph, and cost scales linearly in $n$ (SAFE: one search-augmented verification call per atom, ~$0.19$ per response class of magnitude at 2024 API prices).
- **Human decomposition is not reproducible.** Annotator-vs-annotator claim-set overlap is well below entailment-judgment agreement; papers in this line report needing adjudication rather than raw agreement as the reference standard.
- **Decontextualization injects content.** Gunjal & Durrett show that atomizing "he won it in 1979" into a standalone claim requires resolving "he" and "it," which the extractor does from $y$ *and from its own parametric knowledge* — the resolved entity is sometimes wrong.
- **Unverifiable atoms are common outside Wikipedia-biography text.** VeriScore's central empirical claim, measured across several long-form domains.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no non-circular definition of the correct granularity of a decomposition. Faithfulness and coverage are both checkable, and both are satisfied by decompositions differing 3× in $n$. Until granularity is pinned by something external — a downstream decision, a user study, an axiom — "validity" of $D$ cannot be measured, only asserted.
- **Theoretically open.** Whether any aggregation over $D(y)$ that is (a) monotone in per-claim support and (b) invariant to admissible refinement exists at all. Conjecture: mean-of-atoms cannot be; a coverage-weighted or minimum-description-length weighting might be. No proof either way.
- **Empirically open.** The full extractor × verifier × model-ranking crossing has never been run. Nobody has published: fix $V$, vary $D$ over $\geq 4$ extractors, and report Kendall's $\tau$ between the induced rankings of $\geq 10$ models.
- **Empirically open.** Whether decomposition helps verification accuracy at all, versus verifying sentences directly. Evidence is mixed and small-sample.

## 6. Why It Is Hard

**Absent ground truth compounded by non-identifiability.** A sentence does not have a determinate number of atomic propositions — this is the old problem of individuating facts, and it has no linguistic answer. So the reference object the metric estimates does not exist independently of the estimator.

That converts into a concrete measurement failure: $F(y;D,V)$ is a single scalar produced by two error-prone components, and the decomposition error and verification error are not separately recoverable from it. A model that scores 0.68 under extractor A and 0.61 under extractor B has not been measured twice; it has been measured once by two different instruments with no shared calibration standard.

Second obstruction: the metric is a *ratio*, so it is sensitive to redundancy, which no extractor controls. Repeating a supported fact three times raises the score. This is directly gameable by a generator and is not detected by any faithfulness or coverage check.

## 7. Current Research (as of 2026)

- **Verifiability-gated extraction** — VeriScore (UMass, Iyyer group) and successors, filtering atoms that no evidence source could settle.
- **Granularity-aware decomposition** — molecular facts (Gunjal & Durrett, UT Austin), Claimify (Microsoft Research), CORE-style redundancy filtering (JHU/CLSP-adjacent).
- **Decomposition-free verification** — sentence- or span-level attribution that skips atomization; motivated partly by the instability documented above *(frontier — verify)*.
- **Weighted aggregation by claim salience or checkworthiness**, borrowing from fact-checking triage *(frontier — verify)*.

## 8. Concrete Next Experiment

**The instrument-swap ranking experiment.**

- **Scale.** 12 models (spanning ~7B to frontier), 200 long-form prompts each (LongFact-style, mixed domains, not only biographies) = 2,400 responses. Four extractors: FActScore-style, SAFE, VeriScore, Claimify. One fixed verifier and one fixed evidence corpus for all arms. Cost: ~4 × 2,400 × mean $n$ verification calls; at $n\approx 25$ that is ~240k calls, low five figures in API spend.
- **Control arm.** A decomposition-free arm: the same verifier scoring each *sentence* of $y$ as supported/unsupported, aggregated as fraction of sentences. Plus a redundancy stress arm: each response duplicated-with-paraphrase in one supported sentence, to measure ratio-gaming sensitivity.
- **The deciding number.** Mean pairwise Kendall's $\tau$ between the four extractor-induced model rankings. $\tau \geq 0.9$ across all six pairs: decomposition choice is a nuisance parameter and the field can keep using these metrics. $\tau \leq 0.7$: published long-form factuality rankings are extractor artifacts, and every cross-paper comparison of FActScore numbers is invalid. Secondary number: score inflation under the duplication arm, in points; anything above ~2 points shows the ratio is gameable.

## 9. Key References

- **[Foundational]** Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. — arXiv:2305.14251
- **[SOTA]** Wei, Yang, Song, Lu, Hu, Zhou, Tran, Peng, Liu, Huang, Zhou, Le. *Long-form Factuality in Large Language Models.* NeurIPS 2024. — arXiv:2403.18802
- **[SOTA]** Song, Kim, Iyyer. *VeriScore: Evaluating the Factuality of Verifiable Claims in Long-Form Text Generation.* EMNLP 2024. — arXiv:2406.19276
- **[Core]** Wanner, Ebner, Jiang, Dredze, Van Durme. *A Closer Look at Claim Decomposition.* *SEM 2024. — arXiv:2403.11903
- **[Core]** Gunjal, Durrett. *Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification.* Findings of EMNLP 2024. — arXiv:2406.20079
- **[Core]** Metropolitansky, Larson. *Towards Effective Extraction and Evaluation of Factual Claims.* Microsoft Research, 2025 (Claimify). — arXiv:2502.10855
- **[Related]** Jiang, Wanner, Van Durme, et al. *CORE: Robust Factual Precision with Fine-Grained Claim Extraction.* 2024 preprint.
- **[Related]** Gao, Dai, Pasupat, Chen, Chaganty, Fan, Zhao, Lao, Lee, Juan, Guu. *RARR: Researching and Revising What Language Models Say, Using Language Models.* ACL 2023. — arXiv:2210.08726
- **[Related]** Kamoi, Goyal, Rodriguez, Durrett. *WiCE: Real-World Entailment for Claims in Wikipedia.* EMNLP 2023. — arXiv:2303.01432
- **[Survey]** Augenstein, Baldwin, Bontcheva, et al. *Factuality Challenges in the Era of Large Language Models and Opportunities for Fact-Checking.* Nature Machine Intelligence, 2024.

## 10. Worked Example

Response sentence: *"Marie Curie, a Polish-born physicist working in Paris, won the Nobel Prize in Physics in 1903 with her husband Pierre and Henri Becquerel."*

**Extractor A (coarse, 4 atoms):**
1. Marie Curie was Polish-born. ✓
2. Marie Curie worked in Paris. ✓
3. Marie Curie won the 1903 Nobel Prize in Physics. ✓
4. She shared it with Pierre Curie and Henri Becquerel. ✓

$F_A = 4/4 = 1.00$.

**Extractor B (fine, 8 atoms):** splits (1) into "was born in Poland" / "was a physicist"; (3) into "won a Nobel Prize" / "the prize was in Physics" / "the year was 1903"; (4) into "shared with Pierre Curie" / "shared with Henri Becquerel" / "Pierre Curie was her husband". All ✓ except "was a physicist," which the retriever returns chemistry-heavy evidence for and $V$ marks unsupported.

$F_B = 7/8 = 0.875$.

Same sentence, same evidence, same verifier. **A 12.5-point gap from granularity alone.** Both decompositions are faithful ($y$ entails every atom) and both cover $y$. No validity criterion in the literature rejects either.

Now the redundancy leg: append *"Curie was awarded the Nobel Prize in Physics in 1903."* to $y$. Extractor A now yields 5 atoms, 5 supported, $F_A = 1.00$ (unchanged, already saturated); Extractor B yields 11 atoms, 10 supported, $F_B = 0.909$ — *up* 3.4 points for adding no new information. A generator that pads with restated supported facts raises its factuality score monotonically.

The obstruction is visible in these two numbers: the metric's value depends on a choice ($n$) that no stated criterion constrains, and it rewards a behavior (repetition) that no stated criterion penalizes. Fixing the verifier does not help; fixing the extractor makes the score comparable only within one paper.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*