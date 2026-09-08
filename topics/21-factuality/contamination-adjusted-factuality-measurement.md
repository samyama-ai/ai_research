---
id: 21-factuality/contamination-adjusted-factuality-measurement
title: "Benchmark Contamination Inflates Measured Factual Accuracy"
topic: 21-factuality
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Benchmark Contamination Inflates Measured Factual Accuracy

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/contamination-adjusted-factuality-measurement` · **Status:** methodologically-blocked

## 1. Problem Statement

A factuality benchmark (TriviaQA, Natural Questions, TruthfulQA, SimpleQA, MMLU) reports a number. That number is meant to estimate how often a model produces a correct fact about the world. It actually estimates a mixture: how often the model produces a correct fact, plus how often it reproduces a test item it saw during pretraining. Contamination — the presence of benchmark items, or paraphrases of them, in the training corpus — inflates the first quantity by an unknown amount.

Three variants, of very different difficulty:

- **Measurement.** Given a model $M$ and benchmark $\mathcal{D}$, report an estimate of $M$'s factual accuracy that is unbiased with respect to contamination. This is the catalog problem and it is *methodologically blocked*: the estimand itself is not agreed on.
- **Method.** Given black-box access to $M$, decide per-item whether $x \in \mathcal{D}$ was in $M$'s training data. This is a membership-inference problem, empirically open, and current detectors are near chance at the scales that matter.
- **Theory.** Bound the gap between measured and counterfactual accuracy as a function of duplication count, model capacity, and item entropy. Theoretically open; no nontrivial two-sided bound exists.

Solving the measurement variant means: a procedure that, run twice on the same model by two independent labs, returns the same corrected number, and that number predicts held-out accuracy on genuinely novel facts to within a stated interval.

## 2. Formal Setting

Let $M_\theta$ be trained on corpus $C$ (a multiset of documents). Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ be a benchmark with a scorer $s(\hat{y}, y) \in \{0,1\}$. Measured accuracy is the only directly observable quantity:

$$\hat{A}(M,\mathcal{D}) = \frac{1}{n}\sum_{i=1}^n s\big(M(x_i), y_i\big).$$

**Contamination indicator.** $Z_i \in \{0,1\}$ marks whether item $i$ is "in" $C$. This is where the problem starts: $Z_i$ has no canonical definition. Operationalizations in use:

- $Z_i^{\text{ngram}} = \mathbb{1}[\exists\,$ shared $k$-gram between $(x_i,y_i)$ and some $d \in C]$, $k = 8$–$13$ (GPT-3, GPT-4 reports).
- $Z_i^{\text{emb}} = \mathbb{1}[\max_{d \in C} \cos(e(x_i), e(d)) > \tau]$, catching paraphrase but with an arbitrary $\tau$.
- $Z_i^{\text{fact}} = \mathbb{1}[$ the *fact* $y_i$ appears in $C$ in any surface form $]$ — the semantically right notion, and essentially unmeasurable.

The third is the crux. For a factuality benchmark, $Z_i^{\text{fact}} = 1$ for nearly every item by construction: a benchmark asking "who wrote *Ulysses*" tests a fact that must be in the corpus, or the model could not know it. Contamination of the *item* and coverage of the *fact* are not separable by string matching.

**The estimand.** The causally correct target is the counterfactual accuracy under corpus ablation:

$$A^\star(M,\mathcal{D}) = \mathbb{E}\big[\hat{A}(M_{\theta'},\mathcal{D})\big], \quad \theta' \sim \text{Train}(C \setminus \mathcal{D}),$$

where $C \setminus \mathcal{D}$ removes documents containing benchmark *items* but not documents containing the underlying *facts*. Measuring $A^\star$ requires retraining — at frontier scale, $10^{25}$–$10^{26}$ FLOP per arm.

The universal substitute is clean-subset accuracy $\hat{A}_0 = \frac{1}{|\{i: \hat{Z}_i=0\}|}\sum_{i:\hat{Z}_i = 0} s_i$, with inflation $\Delta = \hat{A} - \hat{A}_0$.

**Assumptions, and their status:**

| Assumption | Status |
|---|---|
| $\hat{Z}$ approximates $Z$ | Violated. Rephrased contamination evades $n$-gram detection entirely (Yang et al. 2023). |
| Clean and dirty subsets are exchangeable in difficulty | Violated. Popular, high-frequency facts are both easier and more duplicated; $\hat{A}_0$ is downward-biased. |
| $C$ is inspectable | Violated for every frontier model since GPT-4. |
| One-shot exposure is negligible | Unknown. Memorization scales log-linearly in duplication count (Carlini et al. 2023), but the intercept at count $=1$ is unmeasured for factual QA. |

## 3. State of the Art

**Established.**

- *Provable black-box test-set detection.* Oren et al. (ICLR 2024) exploit exchangeability: if $\mathcal{D}$ was never seen, the model's log-likelihood is invariant to the ordering of examples within the benchmark file. A permutation test gives a $p$-value with valid false-positive control without any corpus access. This is the only contamination method with a proof. It detects contamination of *the dataset*, not of individual items, and needs the canonical ordering.
- *Rephrasing defeats $n$-gram filters.* Yang et al. (2023) trained a 13B model on rephrased and translated test items; it reached near-frontier scores on GSM8K, MMLU and HumanEval while passing standard $n$-gram decontamination.
- *Held-out-clone gaps are real and model-dependent.* GSM1k (Zhang et al., NeurIPS 2024) rebuilt GSM8K from scratch with matched human difficulty; accuracy drops of up to ~13 points appeared in some open-model families, while several frontier models dropped ≈0–2 points.
- *Membership inference on LLM pretraining is near chance.* Duan et al. (COLM 2024) report AUC only marginally above 0.5 across Pythia 160M–12B on MIMIR, once candidate and non-candidate sets are matched for time and topic.

**Claimed but unablated.**

- Min-K% Prob (Shi et al., ICLR 2024) reports strong detection AUC, but the evaluation sets are not distribution-matched; Duan et al. show that most such gains are a domain-shift artifact.
- "Contamination-free" live benchmarks (LiveBench, LiveCodeBench) control item recency, not fact recency, and their claim of contamination-freedom is a design argument, not a measurement.
- Vendor decontamination reports (GPT-4, Llama, Gemini technical reports) are benchmark numbers only: the filter thresholds, the removed-item counts and the pre/post deltas are not independently reproducible.

**Theory SOTA.** Kalai & Vempala (STOC 2024) show calibrated LMs must err at a rate tied to the fraction of facts appearing once in training. This is the closest formal object to "what does exposure count buy you", and it bounds hallucination from below rather than bounding contamination inflation.

## 4. What Is Known

- **Contamination is pervasive.** Sainz et al. (EMNLP Findings 2023) document benchmark leakage across mainstream corpora; Balloccu et al. (EACL 2024) estimate that on the order of 4.7M benchmark samples were leaked to closed models through user-side API evaluation alone, across 255 surveyed papers.
- **Memorization scales.** Carlini et al. (ICLR 2023), GPT-Neo 125M–6B: extractable memorization grows log-linearly in model size, in duplication count, and in prompt-prefix length.
- **Memorization ≠ exploitation.** Magar & Schwartz (ACL 2022) separate the two on BERT-scale models: a model can memorize a contaminated item and still fail to use it at test time; exploitation rate rises with duplication.
- **Benchmarks are themselves wrong often enough to matter.** Gema et al. (NAACL 2025) hand-audited MMLU and found errors in about 6.5% of items overall, and a majority of items in the worst subject. Contamination correction below a few points is inside the label-noise floor.
- **Format leakage is detectable.** Deng et al. (NAACL 2024) show frontier models can complete masked *wrong* answer options in MMLU and HellaSwag far above chance — evidence of item-level exposure independent of the answer.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed estimand. "Contamination-adjusted factual accuracy" is not defined until someone specifies which documents count as leakage versus as the world knowledge the benchmark is supposed to test. For factual QA these overlap by construction, so $\hat{A}_0$ and $A^\star$ are not even estimating the same thing.
- **Empirically open.** The dose–response curve: for a fixed 7B model, how does accuracy on a held-out fact set move as the same items are injected 1, 2, 4, …, 1024 times, at fixed total token budget? Runnable today for under $10^{22}$ FLOP; not published at any careful scale for *factual* (as opposed to arithmetic or code) benchmarks.
- **Empirically open.** Whether clean-subset accuracy is upward- or downward-biased relative to $A^\star$. Both mechanisms exist; the sign has never been measured against a retraining ground truth.
- **Theoretically open.** Any two-sided bound $|\hat{A} - A^\star| \le f(\text{duplications}, \text{capacity}, H(y_i \mid x_i))$. Nothing of this form is proved.

## 6. Why It Is Hard

**Non-identifiability, not compute.** Two models produce the same output on "Who discovered penicillin? → Fleming": one that saw the TriviaQA item verbatim, and one that read a hundred encyclopedias. Their behavior is identical on that item, and the benchmark's whole purpose is to reward the second. There is no function of $(x_i, y_i, M)$ that separates them, because the difference lives in $C$, which is unobservable for every model whose score anyone cares about.

Compounding: the correction is conditioned on a detector $\hat{Z}$ whose error rate is unknown and whose false negatives are adversarially structured (paraphrase, translation, format change). Correcting with a noisy $\hat{Z}$ moves the estimate by an amount whose *sign* depends on the correlation between detection failure and item difficulty — and that correlation is positive (rare facts are both harder and less likely to be duplicated). The correction is therefore an evaluation that does not measure what it names: "accuracy on clean items" is a difficulty-shifted subsample, not a contamination-free accuracy.

## 7. Current Research (as of 2026)

- **Provable detection.** Extensions of the Oren et al. exchangeability test to per-item and per-shard granularity; Stanford (Hashimoto, Ladhak) and follow-ups. *(frontier — verify)*
- **Continuously refreshed benchmarks.** LiveBench (NYU/Abacus/Nvidia), LiveCodeBench (Berkeley), FreshQA (Google). Controls item age; leaves fact age uncontrolled.
- **Matched-clone construction.** GSM1k (Scale AI) is the template: rebuild a benchmark with matched difficulty and never publish it. Extending this to open-domain factuality is under way but blocked by the cost of matched-difficulty fact sampling. *(frontier — verify)*
- **Contamination-aware aggregate metrics.** ConTAM-style work on how detector choice changes model rankings; early result is that rankings are less sensitive than absolute scores.
- **Canary strings and provenance registries.** BIG-bench canary GUIDs, dataset-level opt-out headers. Adoption is voluntary and partial.

## 8. Concrete Next Experiment

**Question:** is clean-subset accuracy $\hat{A}_0$ a biased estimator of the retraining counterfactual $A^\star$, and in which direction?

**Design.** Fully controlled pretraining, small enough to retrain many arms.

- **Scale.** 1.4B-parameter decoder, ~30B tokens, on a fully inspectable corpus (Pile-scale slice or Dolma subset). ~$3\times10^{20}$ FLOP per arm; ~8 arms fits in a few thousand GPU-hours.
- **Benchmark.** 4,000 held-out factual QA items whose supporting *facts* appear in the corpus but whose *item text* does not (verified by exhaustive substring and embedding search).
- **Arms.** Inject a random 2,000-item half at duplication counts $d \in \{0, 1, 2, 8, 64, 512\}$, in both verbatim and rephrased form. Total token budget held fixed by displacing equal-length filler.
- **Control arm.** $d=0$ — the same 2,000 items never injected. Its accuracy on those items *is* $A^\star$, measured, not estimated.
- **Deciding number.** The signed bias $b(d) = \hat{A}_0(d) - A^\star$, where $\hat{A}_0(d)$ is accuracy on the 2,000 non-injected items in arm $d$. If $|b(d)| < 1$ point for all $d \le 64$, clean-subset correction is defensible and the field can standardize on it. If $|b(d)| > 3$ points at $d = 8$ — a duplication level ordinary web crawls produce — every published decontaminated score is uninterpretable at the granularity people compare models at.

Secondary readout: detector recall of $Z$ at each $d$, split verbatim versus rephrased. Expected to collapse toward 0 for rephrased items at $d \le 8$.

## 9. Key References

- **[Foundational]** Tom Brown et al. *Language Models are Few-Shot Learners.* NeurIPS, 2020. — arXiv:2005.14165 (Section 4: the original $n$-gram contamination analysis)
- **[Foundational]** Inbal Magar, Roy Schwartz. *Data Contamination: From Memorization to Exploitation.* ACL, 2022. — arXiv:2203.08242
- **[SOTA]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[SOTA]** Hugh Zhang et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS, 2024. — arXiv:2405.00332
- **[SOTA]** Michael Duan et al. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[SOTA]** Shuo Yang et al. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[SOTA]** Colin White et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR, 2025. — arXiv:2406.19314
- **[Established]** Nicholas Carlini et al. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Established]** Aryo Pradipta Gema et al. *Are We Done with MMLU?* NAACL, 2025. — arXiv:2406.04127
- **[Established]** Chunyuan Deng et al. *Investigating Data Contamination in Modern Benchmarks for Large Language Models.* NAACL, 2024. — arXiv:2311.09783
- **[Theory]** Adam Tauman Kalai, Santosh Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Survey]** Oscar Sainz et al. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark.* Findings of EMNLP, 2023. — arXiv:2310.18018
- **[Survey]** Simone Balloccu et al. *Leak, Cheat, Repeat: Data Contamination and Evaluation Malpractices in Closed-Source LLMs.* EACL, 2024. — arXiv:2402.03927
- **[Survey]** Cheng Xu et al. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244

## 10. Worked Example

Take a 1,000-item factual QA set. A model scores $\hat{A} = 0.780$ (780 correct). A 13-gram filter over an open corpus flags 180 items as contaminated.

- Dirty subset: 168/180 correct → 0.933.
- Clean subset: 612/820 correct → $\hat{A}_0 = 0.746$.
- Reported correction: $\Delta = 0.780 - 0.746 = 3.4$ points. This is the number that appears in papers.

Now apply what is known.

**Detector recall.** Yang et al. show rephrased items pass $n$-gram filters. Suppose true contamination is 300 items and the filter caught 180, so recall is 0.60 and 120 contaminated items sit inside the "clean" 820. If those 120 score at the dirty rate (0.933, i.e. 112 correct), the truly-clean 700 items score $(612-112)/700 = 0.714$. The correction is 6.6 points, not 3.4 — nearly double.

**Difficulty confound, opposite sign.** Duplication tracks entity popularity. Suppose the 300 contaminated items are drawn from the top popularity quartile, where the model would have scored 0.86 with no contamination at all. Then the contamination-attributable gain is $0.933 - 0.86 = 7.3$ points on 30% of items — 2.2 points overall, *less* than the naive 3.4.

**The obstruction, made numeric.** The same 780/1000 supports a correction anywhere from 2.2 to 6.6 points depending on two unmeasured quantities: detector recall, and the counterfactual accuracy of contaminated items. Neither is estimable from $(x_i, y_i, M)$. And the whole 4.4-point spread sits inside MMLU's measured ~6.5% label-error rate. The reported "3.4 points of contamination" is a number with no error bar, produced by a procedure whose free parameters move it by more than its own magnitude. That is why the status is *methodologically blocked* and not *empirically open*: no amount of additional evaluation on the existing benchmark resolves it. Only the retraining control arm in §8 does.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*