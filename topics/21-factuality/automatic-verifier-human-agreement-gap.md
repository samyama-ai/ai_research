---
id: 21-factuality/automatic-verifier-human-agreement-gap
title: "Automatic Fact Verifiers Disagree with Expert Human Judgment"
topic: 21-factuality
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Automatic Fact Verifiers Disagree with Expert Human Judgment

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/automatic-verifier-human-agreement-gap` · **Status:** empirically-open

## 1. Problem Statement

Automatic factuality metrics (FActScore, SAFE, VeriScore, MiniCheck, NLI-based entailment scorers, LLM-as-judge pipelines) assign a support label to each atomic claim extracted from a model generation. These labels are treated as ground truth in papers, leaderboards, and RLHF reward models. The problem: **we do not know how far these labels are from the judgment of a qualified domain expert, nor whether the residual disagreement is random noise or a systematic bias that inflates measured factuality.**

Three variants, of very different difficulty:

- **Measurement variant.** Given a verifier $V$ and an expert panel, estimate the chance-corrected agreement and the *signed* bias $\mathbb{E}[V] - \mathbb{E}[H]$ on a domain-stratified sample. Runnable today; expensive.
- **Method variant.** Build a verifier whose disagreement with experts is no larger than expert–expert disagreement, at a cost that permits per-example use in training loops.
- **Theory variant.** Given that experts themselves disagree, characterise the conditions under which a latent "true" support label is identifiable at all from a finite panel of noisy raters, and what a verifier can be held to when it is not.

Solved means: a published, replicated estimate of $\Delta$ (defined below) across $\geq 4$ knowledge domains, with the sign of the bias established, plus a verifier that closes $\Delta$ to within panel noise.

## 2. Formal Setting

A generation $y$ is decomposed by an extractor $E$ into atomic claims $c_1,\dots,c_n = E(y)$. Each claim carries a label in $\mathcal{L} = \{\texttt{S}, \texttt{NS}, \texttt{IR}\}$ (supported / not supported / irrelevant-or-undecidable).

- **Verifier.** $V: (c, \mathcal{K}) \to \mathcal{L}$, where $\mathcal{K}$ is the evidence store (a Wikipedia dump, a live search API, or the verifier's parametric memory). $\mathcal{K}$ is *not* fixed across papers, which is the first confound.
- **Expert label.** Rater $j$ from a qualified pool gives $H_j(c) \in \mathcal{L}$. The aggregate $H(c)$ is the majority over $m$ raters; ties resolve to $\texttt{IR}$.
- **Raw agreement.** $A_o = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[V(c_i) = H(c_i)]$, measured over $N$ claims sampled from a stated prompt distribution.
- **Chance-corrected agreement.** With marginals $p_V(\ell), p_H(\ell)$, $A_e = \sum_{\ell} p_V(\ell)p_H(\ell)$ and $\kappa = (A_o - A_e)/(1 - A_e)$. Reporting $A_o$ alone is uninformative when $p_H(\texttt{S}) \approx 0.85$, as it is on biography-style benchmarks.
- **Human ceiling.** $\kappa_{HH}$, the same statistic computed between two disjoint expert panels on the same claims. This is the only defensible upper bound.
- **The gap.** $$\Delta = \kappa_{HH} - \kappa_{VH}.$$ $\Delta \approx 0$ means the verifier is as good as an expert; $\Delta > 0$ quantifies the shortfall in units the field can compare across domains.
- **Signed bias.** $b = \Pr[V = \texttt{S}] - \Pr[H = \texttt{S}]$. $b > 0$ means published factuality scores are optimistic. $b$, not $\kappa$, is what corrupts a leaderboard.
- **Score-level error.** Metric $F(y) = \frac{1}{n}\sum_i \mathbf{1}[V(c_i)=\texttt{S}]$; the quantity that matters downstream is $|\mathbb{E}[F_V] - \mathbb{E}[F_H]|$ *and* the rank correlation $\tau$ between model orderings induced by $F_V$ and $F_H$.

**Assumptions, and their status:**

| Assumption | Status |
|---|---|
| Claim decomposition $E$ is label-invariant | **Violated.** Different decompositions of the same $y$ yield different $n$ and different per-claim difficulty; $E$ is usually the same LLM family as $V$. |
| A single latent label exists per claim | **Violated** for context-dependent, temporally indexed, and partially-true claims. |
| Expert raters are conditionally independent given the claim | **Violated** by shared guidelines, shared evidence retrieval, and anchoring on the model's phrasing. |
| $\mathcal{K}$ contains the evidence needed | **Violated** for long-tail entities; verifiers systematically map "absent from $\mathcal{K}$" to $\texttt{NS}$. |
| Crowdworkers approximate experts | **Unestablished** — this is the crux of the problem. |

## 3. State of the Art

**Established (with ablations).**
- **FActScore** (Min et al., EMNLP 2023) validated its retrieval+LM estimator against crowdworker labels on biographies and reported estimation error under 2% on the aggregate score, with human annotation costing roughly $4 per generation. Established at the *aggregate* level only; per-claim agreement is weaker than the aggregate error suggests.
- **MiniCheck** (Tang et al., EMNLP 2024) shows a 400M-parameter checker matching GPT-4-level performance on the LLM-AggreFact collection at ~400× lower cost. Established as a *benchmark* result against existing human labels; it inherits whatever bias those labels carry.
- **AIS** (Rashkin et al., Computational Linguistics 2023) is the one careful protocol paper: it defines attribution operationally and reports the interpretability of the human task itself.

**Claimed but unablated.**
- **SAFE** (Wei et al., 2024) reports 72% agreement with crowdworker labels over ~16k individual facts, and that on a 100-case disagreement sample SAFE was judged correct 76% of the time versus 19% for the human. The "SAFE beats humans" claim rests on a single 100-item adjudication by the same authors, with no expert panel, no chance correction, and no domain stratification. It is the most-cited evidence that verifiers are human-level, and it is the weakest.
- **LLM-as-judge agreement parity** (Zheng et al., NeurIPS 2023 Datasets & Benchmarks) reports GPT-4 agreeing with human preference at ~85%, versus ~81% human–human. This is preference, not factuality, and is routinely over-transferred to the factuality setting.

**Benchmark-number-only.** Every headline factuality score on HaluEval, RAGTruth, FEVER-derived splits and LLM-AggreFact is an agreement rate against a fixed label set, not against a fresh expert panel. No leaderboard reports $\kappa_{HH}$.

## 4. What Is Known

- **FEVER** (Thorne et al., NAACL 2018): five-way inter-annotator Fleiss $\kappa = 0.68$ on 4,000 claims — for *deliberately constructed, short* Wikipedia claims. This is the ceiling in the easiest possible setting.
- **SAFE**: 72% raw agreement, $N \approx 16{,}000$ facts from 496 prompts; no $\kappa$ reported.
- **FActScore**: <2% aggregate estimation error, measured on 500+ biography generations from 6-12 models.
- **WiCE** (Kamoi et al., EMNLP 2023): real Wikipedia claims decompose into sub-claims with mixed support; models that score well on synthetic entailment data degrade sharply on this natural distribution.
- **ExpertQA** (Malaviya et al., NAACL 2024): 484 domain experts evaluated 2,177 questions and found frequent attribution failures that non-expert evaluation had not surfaced — the strongest existing evidence that the crowdworker proxy is loose.
- **Self-preference**: Panickssery et al. (NeurIPS 2024) show LLM evaluators recognise and favour their own generations, a directional bias with $b > 0$ when $V$ and the generator share a family.

## 5. What Is Not Known

- **Empirically open** (the majority of the gap): $\kappa_{VH}$ against *credentialed experts*, on any long-form factuality benchmark, in any domain. Nobody has run the panel. Likewise the sign and magnitude of $b$, and whether $F_V$ and $F_H$ induce the same model ranking — a verifier with $\kappa = 0.5$ can still rank models perfectly if its errors are model-independent, and nobody has tested that.
- **Methodologically blocked**: the label space itself. There is no agreed operationalisation of "supported" for claims that are true-but-misleading, true-at-time-$t$, or true-under-one-reading. Disagreement here is not noise to be averaged away, and $\kappa$ is undefined without a settled $\mathcal{L}$.
- **Theoretically open**: identifiability. Under a Dawid–Skene-style latent-label model with correlated raters, the conditions for identifying the true label — and hence for the gap $\Delta$ to be estimable at all — are unproven for the case where the verifier and some raters share an evidence store.

## 6. Why It Is Hard

**Absent ground truth compounded by circular validation.** Verifiers are validated against crowdworker labels; crowdworkers are validated against nothing. When the verifier and the annotation guideline are both derived from the same LLM family, agreement measures shared prior, not correctness. Second, **an evaluation that does not measure what it names**: raw agreement on a corpus where 85% of atomic claims are supported is dominated by the majority class, so a verifier that answers $\texttt{S}$ unconditionally scores 85% while carrying $\kappa = 0$. Third, **cost**: expert adjudication of a claim with retrieval runs 3-10 minutes; a 5,000-claim, 3-rater, 4-domain panel is roughly 1,000 expert-hours, i.e. $\$75$k-$\$150$k — beyond a typical paper budget and unrewarded by any venue.

## 7. Current Research (as of 2026)

- **Cheaper specialised checkers**: MiniCheck-style distilled entailment models; Bespoke Labs' factuality checkers. Optimising against existing labels, so they cannot close $\Delta$ by construction.
- **Decomposition-aware metrics**: VeriScore (Song, Kim, Iyyer, EMNLP Findings 2024) restricts scoring to verifiable claims, directly attacking the $\texttt{IR}$ class. *(frontier — verify)*
- **Taxonomy-first evaluation**: HALoGEN (Ravichander et al., 2025) separates hallucination types before scoring, which is a prerequisite for a well-defined $\mathcal{L}$.
- **Expert-in-the-loop benchmarks**: follow-ons to ExpertQA at UPenn; medical/legal factuality panels (Stanford CRFM, Google DeepMind health evaluations) are the closest existing thing to the missing experiment. *(frontier — verify)*
- **Rater-model statistics** returning to NLP: Dawid–Skene and item-response models applied to LLM judges. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** 1,500 atomic claims, stratified 375 each across biomedicine, law, software engineering, and history, drawn from long-form generations of three models spanning a capability range. Each claim labelled by **two disjoint expert panels of 3** (6 experts per claim, ~450 expert-hours, ~$60k), under the AIS protocol with a fixed evidence store.

**Arms.** (a) SAFE with search; (b) MiniCheck; (c) GPT-class LLM-as-judge, same family as the generator; (d) **control arm: crowdworkers** on the identical claims and identical guidelines — this arm is what isolates "verifier vs expert" from "crowdworker vs expert".

**Deciding number.** $$\Delta = \kappa_{HH} - \kappa_{VH}, \quad \text{decision threshold } \Delta \leq 0.10.$$ If the best verifier achieves $\Delta \leq 0.10$ with a 95% bootstrap CI excluding 0.20, automatic verification is validated for that domain. If $\Delta > 0.20$ in any domain, every published factuality score in that domain is uncalibrated. Report $b$ alongside: $|b| > 0.05$ means leaderboards are biased, not merely noisy.

## 9. Key References

- **[Foundational]** Thorne, Vlachos, Christodoulopoulos, Mittal. *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* NAACL 2018. — arXiv:1803.05355
- **[Foundational]** Rashkin, Nikolaev, Lamm, Aroyo, Collins, Das, Petrov, Tomar, Turc, Reitter. *Measuring Attribution in Natural Language Generation Models.* Computational Linguistics, 2023. — arXiv:2112.12870
- **[SOTA]** Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. — arXiv:2305.14251
- **[SOTA]** Wei, Yang, Song, Lu, Peng, Chen, Gu, Jiang, Xiong, Xia, Chen, Le, Zhou. *Long-form Factuality in Large Language Models.* 2024. — arXiv:2403.18802
- **[SOTA]** Tang, Laban, Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.* EMNLP 2024. — arXiv:2404.10774
- **[Evidence]** Malaviya, Lee, Chen, Sieber, Yatskar, Roth. *ExpertQA: Expert-Curated Questions and Attributed Answers.* NAACL 2024.
- **[Evidence]** Kamoi, Goyal, Rodriguez, Durrett. *WiCE: Real-World Entailment for Claims in Wikipedia.* EMNLP 2023.
- **[Evidence]** Panickssery, Bowman, Feng. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024.
- **[Survey/Context]** Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks 2023. — arXiv:2306.05685
- **[Survey]** Song, Kim, Iyyer. *VeriScore: Evaluating the Factuality of Verifiable Claims in Long-Form Text Generation.* EMNLP Findings 2024.

## 10. Worked Example

Take SAFE's headline: **72% raw agreement over 16,011 atomic facts** against FActScore crowdworker labels. Reduce to two classes ($\texttt{S}$ vs not).

Marginals are not reported per-class in the paper. Assume (*assumed*, plausible for biography data where most atomic facts are correct) $p_H(\texttt{S}) = 0.85$ and $p_V(\texttt{S}) = 0.88$. Then chance agreement is

$$A_e = 0.88 \times 0.85 + 0.12 \times 0.15 = 0.748 + 0.018 = 0.766,$$

and

$$\kappa = \frac{A_o - A_e}{1 - A_e} = \frac{0.72 - 0.766}{0.234} = -0.20.$$

**A verifier reported as 72%-agreeing is, under these marginals, performing worse than chance on the class that matters.** The constant predictor "always $\texttt{S}$" scores $A_o = 0.85$ — 13 points *above* SAFE — while detecting zero hallucinations.

Now the downstream consequence. A model emits 60 atomic claims per biography, truly 51 supported ($F_H = 0.85$). With $b = +0.03$, the verifier reports $F_V = 0.88$: 1.8 hallucinated claims per generation are scored as supported. Across a 500-generation eval that is ~900 missed errors, and a 3-point inflation applied uniformly — harmless for ranking *if* $b$ is model-independent. It is not: self-preference bias (Panickssery et al. 2024) makes $b$ larger when $V$ shares a family with the generator, so the ranking flips are concentrated exactly where labs compare their own model to a competitor's.

The obstruction is now visible: the field's headline validation number is (i) not chance-corrected, (ii) computed against crowdworkers rather than experts, and (iii) reported without the class marginals needed to recompute it. All three are fixed by the panel in §8, and by nothing cheaper.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*