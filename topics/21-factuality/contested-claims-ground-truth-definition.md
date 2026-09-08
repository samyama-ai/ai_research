---
id: 21-factuality/contested-claims-ground-truth-definition
title: "Ground Truth for Contested and Ambiguous Claims"
topic: 21-factuality
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Ground Truth for Contested and Ambiguous Claims

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/contested-claims-ground-truth-definition` · **Status:** methodologically-blocked

## 1. Problem Statement

Factuality benchmarks assume each claim has a label in $\{\text{supported}, \text{refuted}, \text{not enough info}\}$. For a large fraction of claims that matter — contested empirical questions, normatively loaded descriptions, vague predicates, claims whose truth depends on an unstated referent or time — no such label exists that survives re-annotation by a different, equally competent annotator pool.

Three variants, of very different difficulty:

- **Measurement.** Given a claim $c$ and an evidence corpus $E$, output a target object that an evaluation can score a model against. The open question is *what object*: a single label, a distribution over labels, a set of admissible readings, or a per-subpopulation label vector. Solving this means: a target that is reproducible across independent annotator pools, and that separates *irreducible disagreement* from *annotator noise*.
- **Method.** Given that target, train and evaluate models to predict it, including abstention on claims where the target is high-entropy. Runnable once the measurement exists.
- **Theory.** Under what conditions is the disagreement/noise decomposition identifiable from annotation data at all, and what is the sample complexity in annotators (not items)?

The measurement variant blocks the other two. That is why this page is marked methodologically blocked rather than open.

## 2. Formal Setting

Let $c$ be a claim, $E$ an evidence set, $\mathcal{Y}$ a label space. Let $\mathcal{A}$ be a population of competent annotators with sampling distribution $\pi$. Annotator $a$ on occasion $t$ returns $y_{a,t}(c,E)$.

Two nested distributions:

$$P_a(y \mid c) = \Pr_t[y_{a,t}(c,E)=y], \qquad P(y\mid c) = \mathbb{E}_{a\sim\pi}\big[P_a(y\mid c)\big].$$

$P_a$ is measurable only by re-asking the *same* annotator after a washout interval (test–retest). $P$ is what a standard crowd protocol estimates.

**Disagreement decomposition.** Writing $H$ for Shannon entropy and $a$ as a random annotator,

$$H\big(P(\cdot\mid c)\big) = \underbrace{I(y; a \mid c)}_{\text{contestedness}} + \underbrace{\mathbb{E}_{a\sim\pi}\,H\big(P_a(\cdot\mid c)\big)}_{\text{noise}}.$$

Contestedness $\kappa(c) := I(y;a\mid c)$ is the quantity a catalog of "contested claims" needs. **It is not identifiable from single-pass annotation**: one label per annotator makes $I(y;a)$ and $\mathbb{E}_a H(P_a)$ perfectly confounded. Identification requires $\ge 2$ independent responses per annotator per item.

**Estimator.** With $n$ annotators drawn from a *recruited* pool $\hat\pi$ (Prolific, MTurk, expert panel), $\hat P_n(y\mid c)=\frac1n\sum_i \mathbf{1}[y_i=y]$. Then

$$\|\hat P_n - P\|_1 \le \underbrace{O(\sqrt{|\mathcal{Y}|/n})}_{\text{sampling}} + \underbrace{\|\mathbb{E}_{\hat\pi}P_a - \mathbb{E}_{\pi}P_a\|_1}_{\text{pool bias, does not shrink in } n}.$$

**Model scoring.** A model emits $q(\cdot\mid c)$; score by $\mathrm{JSD}(q\,\|\,\hat P_n)$ or by calibration of $\max_y q$ against $\max_y \hat P_n$. Accuracy against $\arg\max_y \hat P_n$ discards $\kappa$ entirely.

**Assumptions, and which are violated.**
1. *A single population $\pi$ exists and is agreed on.* Violated: for political and medical claims the answer depends on which population is sampled; no principled choice exists.
2. *Annotators are exchangeable.* Violated — annotator identity predicts labels on toxicity and misinformation tasks (Sap et al., NAACL 2022).
3. *Within-annotator responses are i.i.d. over occasions.* Violated by memory and anchoring; washout intervals are rarely enforced.
4. *$E$ is fixed and sufficient.* Violated: real contested claims lack counter-evidence in any retrievable corpus (Glockner et al., EMNLP 2022).
5. *The claim has one reading.* Violated: ambiguity in referent, scope and time is pervasive (AmbigQA: over half of open-domain NQ questions are ambiguous).

## 3. State of the Art

**Established (reproduced, ablated).**
- *Human label variation is signal, not noise.* Pavlick & Kwiatkowski (TACL 2019) collected up to 50 judgments per NLI item on a graded scale and showed the multi-modal shape persists as $n$ grows — it is not sampling error. Independently confirmed by ChaosNLI (Nie et al., EMNLP 2020).
- *Distributional targets are learnable and standard accuracy hides the gap.* ChaosNLI: 100 labels each on 4,645 items (464,500 labels). Models near ceiling on old gold labels have large JSD to the 100-annotator distribution.
- *Fact-checking labels have modest reliability even on curated data.* FEVER (Thorne et al., NAACL 2018) reports Fleiss $\kappa = 0.68$ for label agreement on 185,445 claims — acceptable by convention, but that ceiling is measured on claims deliberately constructed to be checkable against Wikipedia.

**Claimed but unablated / benchmark-number-only.**
- Automatic factuality scorers (FActScore, EMNLP 2023; SAFE/LongFact, Wei et al. 2024) report agreement with human annotators on decomposed atomic facts. These are single-aggregate-label pipelines; their behaviour on high-$\kappa$ atomic facts is a benchmark number with no reported breakdown by contestedness.
- LLM-as-judge "consensus" and bridging-style aggregation (X Community Notes, Wojcik et al. 2022) are deployed at scale but there is no published ablation showing bridging recovers $\kappa$ rather than the majority view of the more numerous faction.
- Collective Constitutional AI (Anthropic, 2023) elicits public input on norms; no evaluation ties the resulting model to a per-claim contestedness estimate. *(frontier — verify)*

## 4. What Is Known

- **Scale of ambiguity.** AmbigQA (Min et al., EMNLP 2020): over 50% of a 14,042-question NQ-Open sample is ambiguous; annotators found on average 2+ distinct valid interpretations for ambiguous questions.
- **Scale of persistent disagreement.** ChaosNLI: 100 annotators × 4,645 items; on a substantial minority of SNLI/MNLI items the majority label of 100 disagrees with the original gold label, and per-item label entropy stays high.
- **Annotator identity predicts labels.** Sap et al. (NAACL 2022, $n\approx$ hundreds of annotators): political identity and racial attitudes significantly predict toxicity ratings of the same texts.
- **Aggregation protocol changes the answer.** Röttger et al. (NAACL 2022): "prescriptive" (rulebook-enforced) versus "descriptive" (belief-eliciting) guidelines yield different datasets from the same items.
- **Cost.** FActScore reports human evaluation of long-form biography factuality costing over \$26,000 for their study — single-pass. Test–retest at $\ge 2$ passes per annotator roughly doubles that before any pool-diversity multiplier.
- **Evidence is missing, not just contested.** Glockner et al. (EMNLP 2022): for real-world misinformation claims, retrievable counter-evidence is largely absent, so NEI labels conflate "contested" with "un-retrievable".

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** No accepted operational definition of "contested" separates $I(y;a\mid c)$ from within-annotator noise. Essentially no factuality dataset collects repeated within-annotator judgments, so $\kappa(c)$ has never been estimated on a factuality corpus.
- **Methodologically blocked.** No principled rule for choosing $\pi$. "General public", "domain experts", and "balanced panel" give different targets for the same claim; no published criterion adjudicates.
- **Empirically open.** Whether a model trained on distributional targets transfers its contestedness estimates to unseen claim domains. Runnable at 7B–70B today; unrun because no multi-domain $\kappa$-labelled corpus exists.
- **Empirically open.** Whether current frontier models' verbalized uncertainty on contested claims correlates with $\hat\kappa$ at all, or only with retrieval sparsity.
- **Theoretically open.** Sample complexity in *annotators* (not items) for estimating $\kappa$ to $\pm\epsilon$ under heterogeneous per-annotator noise; and whether any single-pass protocol plus covariates can identify the decomposition.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under the standard annotation protocol, compounded by absent ground truth for the reference population**.

Single-pass crowd annotation yields exactly one draw from $P_a$ per annotator. Every observable is a function of the mixture $P$. Two worlds — (i) all annotators are individually certain but split 50/50 by ideology, (ii) every annotator flips a fair coin — produce identical data at any $n$. They demand opposite system behaviours: (i) "this is contested, here are the two positions"; (ii) "nobody knows, abstain". No amount of extra annotators distinguishes them.

Second obstruction: **the evaluation does not measure what it names**. Accuracy against $\arg\max \hat P_n$ scores a model as correct for asserting a 51/49 split as fact, and as wrong for reporting the split. The metric name says "factuality"; the quantity is "agreement with the modal view of a convenience sample".

## 7. Current Research (as of 2026)

- **Perspectivist annotation.** Continuation of the line from Basile, Cabitza, Fornaciari, Uma, Poesio and Plank: release per-annotator labels rather than aggregates; the LeWiDi shared tasks (Learning with Disagreements) are the standing venue.
- **Distributional and soft-label training.** Soft-label and calibration losses against human label distributions; ChaosNLI remains the reference dataset.
- **Population-conditioned models.** Santurkar et al. (ICML 2023, OpinionQA) and Durmus et al. (Anthropic, GlobalOpinionQA, 2023) measure whose opinions a model reflects — the closest existing operationalisation of "which $\pi$".
- **Bridging aggregation.** Community Notes' bridging-based ranking is the largest deployed system that explicitly targets cross-faction agreement rather than majority.
- **Contestedness-aware fact-checking.** Extensions of FActScore/SAFE decomposition that route atomic facts to abstention. *(frontier — verify: no reproduced ablation as of 2026-09.)*

## 8. Concrete Next Experiment

**Question.** On real contested claims, how much of the observed label entropy is genuine cross-annotator disagreement versus within-annotator instability?

**Design.** Sample 500 claims stratified into 5 strata by expected contestedness (settled science, disputed science, economic causation, political characterisation, vague-predicate claims). Recruit 150 annotators with recorded covariates (political self-placement, domain expertise, country). Each annotator labels all 500 claims **twice**, sessions separated by $\ge 14$ days, item order re-randomised. Labels: 5-point support scale plus a free-text disambiguating reading. Total judgments: $500 \times 150 \times 2 = 150{,}000$.

- **Scale:** 150k judgments; at \$0.10/judgment ≈ \$15k plus recruitment. Within reach of one lab.
- **Control arm:** the same 500 claims annotated single-pass by 150 *different* annotators from the same pool, i.e. the standard protocol. This is what current datasets would have produced.
- **Deciding number:** the **contestedness fraction** $\rho = \hat I(y;a\mid c) \big/ \hat H(P(\cdot\mid c))$, averaged over the disputed strata, with bootstrap CI over annotators. If $\rho > 0.7$, contestedness is real and separable, and distributional targets are the right object — the method variant unblocks. If $\rho < 0.3$, most apparent contestation is within-annotator instability, and the field should be building abstention and elicitation protocols, not per-population targets. The control arm shows how far the single-pass estimate of $H$ misattributes.

**Secondary readout:** correlation between $\hat\rho$ per claim and frontier-model verbalized uncertainty, on the same 500 claims.

## 9. Key References

- **[Foundational]** Ellie Pavlick, Tom Kwiatkowski. *Inherent Disagreements in Human Textual Inferences.* TACL, 2019.
- **[Foundational]** Lora Aroyo, Chris Welty. *Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation.* AI Magazine 36(1), 2015.
- **[Foundational]** Yixin Nie, Xiang Zhou, Mohit Bansal. *What Can We Learn from Collective Human Opinions on Natural Language Inference Data?* EMNLP, 2020. — arXiv:2010.03532
- **[Foundational]** Barbara Plank. *The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation.* EMNLP, 2022. — arXiv:2211.02570
- **[Survey]** Alexandra Uma, Tommaso Fornaciari, Dirk Hovy, Silviu Paun, Barbara Plank, Massimo Poesio. *Learning from Disagreement: A Survey.* JAIR 72, 2021.
- **[Foundational]** James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal. *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* NAACL, 2018. — arXiv:1803.05355
- **[SOTA]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[SOTA]** Jerry Wei et al. *Long-form Factuality in Large Language Models.* NeurIPS, 2024. — arXiv:2403.18802
- **[SOTA]** Sewon Min, Julian Michael, Hannaneh Hajishirzi, Luke Zettlemoyer. *AmbigQA: Answering Ambiguous Open-domain Questions.* EMNLP, 2020. — arXiv:2004.10645
- **[SOTA]** Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, Tatsunori Hashimoto. *Whose Opinions Do Language Models Reflect?* ICML, 2023. — arXiv:2303.17548
- **[SOTA]** Esin Durmus et al. *Towards Measuring the Representation of Subjective Global Opinions in Language Models.* Anthropic, 2023. — arXiv:2306.16388
- Maarten Sap, Swabha Swayamdipta, Laura Vianna, Xuhui Zhou, Yejin Choi, Noah A. Smith. *Annotators with Attitudes: How Annotator Beliefs And Identities Bias Toxic Language Detection.* NAACL, 2022. — arXiv:2111.07997
- Paul Röttger, Bertie Vidgen, Dirk Hovy, Janet Pierrehumbert. *Two Contrasting Data Annotation Paradigms for Subjective NLP Tasks.* NAACL, 2022. — arXiv:2112.07475
- Max Glockner, Yufang Hou, Iryna Gurevych. *Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation.* EMNLP, 2022. — arXiv:2210.13865
- Tom Hosking, Phil Blunsom, Max Bartolo. *Human Feedback is not Gold Standard.* ICLR, 2024. — arXiv:2309.16349
- Stefan Wojcik et al. *Birdwatch: Crowd Wisdom and Bridging Algorithms can Inform Understanding and Reduce the Spread of Misinformation.* 2022. — arXiv:2210.15723

## 10. Worked Example

**Claim.** "The 2021–22 inflation surge in the United States was caused primarily by fiscal stimulus."

Suppose 100 annotators each give one label in $\{\text{supported},\text{refuted},\text{NEI}\}$ and the counts are $(48, 37, 15)$.

$$\hat P = (0.48, 0.37, 0.15), \quad H(\hat P) = -\textstyle\sum \hat p\log_2 \hat p = 1.44 \text{ bits}.$$

A standard benchmark takes $\arg\max = \text{supported}$ and scores a model 1.0 for asserting it, 0.0 for reporting the dispute. Sampling error on the top proportion is $\sqrt{0.48\cdot0.52/100} = 0.050$, so the 0.48 vs 0.37 gap is only $\approx 1.6$ SE apart — the gold label itself flips with modest probability across replications. That already breaks the benchmark.

Now the identification failure. Two generative stories both produce $\hat P=(0.48,0.37,0.15)$ at any $n$:

| World | Per-annotator $P_a$ | $I(y;a)$ | $\mathbb{E}_a H(P_a)$ | $\rho$ |
|---|---|---|---|---|
| A: partisan certainty | 48 annotators at $(1,0,0)$, 37 at $(0,1,0)$, 15 at $(0,0,1)$ | 1.44 | 0.00 | 1.00 |
| B: uniform confusion | every annotator at $(0.48,0.37,0.15)$ | 0.00 | 1.44 | 0.00 |

Observed totals are identical. Required system behaviour is opposite: World A calls for "economists disagree; the fiscal-stimulus account is held by roughly half, the supply-chain account by roughly a third"; World B calls for "I don't have a reliable basis to attribute a primary cause."

Distinguishing them costs one extra pass: re-ask the same 100 annotators after 14 days. In World A the per-annotator label match rate is near 1.0; in World B it is $\sum_y p_y^2 = 0.48^2+0.37^2+0.15^2 = 0.39$. The gap, 1.00 vs 0.39, is enormous and trivially measurable — and no major factuality dataset has measured it. That is the block: not a hard inference, an uncollected second column.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*