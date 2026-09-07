---
id: 21-factuality/calibration-hallucination-rate-lower-bound
title: "Intrinsic Hallucination Rate Lower Bound for Calibrated Generators"
topic: 21-factuality
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Intrinsic Hallucination Rate Lower Bound for Calibrated Generators

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/calibration-hallucination-rate-lower-bound` · **Status:** partially-solved

## 1. Problem Statement

A generator that is *calibrated* — its predicted probabilities match empirical frequencies — cannot also be arbitrarily factual. The question is how large the forced error is, and whether the known bounds bind on real systems.

- **Input:** a fact family $F$ (e.g. "date of birth of person $x$"), a training corpus $C$, a generator $p_\theta$ trained on $C$, a validity oracle $v: \text{string} \to \{0,1\}$.
- **Output:** a lower bound $L(C, F)$ on the rate at which $p_\theta$ emits invalid statements about $F$, valid for every generator satisfying a stated calibration condition.
- **Decision predicate:** does the measured hallucination rate of a deployed model track $L$, and by how much does post-training slack (abstention, refusal, miscalibration) let a model fall below it?

Three variants, different difficulty:

- **Theory.** Prove (or refute) a lower bound of the form $\text{hall}(p_\theta) \ge g(\text{singleton rate}) - \text{miscalibration} - o(1)$. *Partially solved* — Kalai & Vempala (STOC 2024) give exactly this.
- **Measurement.** Estimate the right-hand side on a real corpus, which requires counting how many facts in $F$ appear exactly once in $C$. Blocked for closed models.
- **Method.** Decide whether the intrinsic floor is *escapable* by abstention (the model says "I don't know") without destroying calibration on the answered subset, and at what cost in coverage.

## 2. Formal Setting

Let $F = \{f_1,\dots,f_N\}$ be a set of facts, each expressible as a prompt $q_i$ with a valid answer set $A_i$. A corpus $C$ of $n$ fact-mentions is drawn i.i.d. from a distribution $\mu$ over $F$. Define the **singleton count**

$$n_1 = \big|\{f \in F : \text{count}_C(f) = 1\}\big|, \qquad \hat{M} = n_1/n,$$

the Good–Turing estimator of the **missing mass** $M = \sum_{f: \text{count}_C(f)=0} \mu(f)$ (Good, *Biometrika* 1953). $\hat{M}$ concentrates: $|\hat{M} - M| = O(n^{-1/2})$ with high probability (McAllester & Schapire, COLT 2000). Measured as: a string-match or entity-linked count over the deduplicated training corpus, restricted to mentions of the target relation.

**Hallucination rate.** With $\hat{a}_i \sim p_\theta(\cdot \mid q_i)$,

$$\text{hall}(p_\theta) = \Pr_{i \sim \mu,\ \hat{a}_i}\!\left[\hat{a}_i \notin A_i \ \wedge\ \hat{a}_i \neq \bot\right],$$

where $\bot$ is an explicit abstention. Measured as: sampling one response per prompt at temperature $T$, grading with a judge or exact-match oracle, and separating "wrong" from "not attempted".

**Calibration.** For a partition of prefixes into confidence bins $B_k$,

$$\mathrm{ECE} = \sum_k \frac{|B_k|}{n}\Big|\ \overline{\text{acc}}(B_k) - \overline{\text{conf}}(B_k)\ \Big|.$$

Kalai & Vempala use a stronger, generative form: the model's plausibility ordering over completions must match the corpus distribution, so that mass assigned to the "unseen fact" region equals $M$.

**Core inequality (informal).** For a calibrated generator on a fact family with no learnable structure ("arbitrary facts", $A_i$ independent of $q_i$ given the corpus),

$$\text{hall}(p_\theta) \ \ge\ \hat{M} \;-\; \varepsilon_{\text{cal}} \;-\; O(n^{-1/2}),$$

with $\varepsilon_{\text{cal}}$ a miscalibration term.

**Assumptions, and which break.**

| Assumption | Status in practice |
|---|---|
| Facts i.i.d. from $\mu$ | **Violated.** Web corpora are heavy-tailed and heavily duplicated; deduplication changes $n_1$ by large factors. |
| Facts arbitrary (no shared structure) | **Violated for most families.** Birthdays of famous people correlate with era, name, cohort; a model can partially infer. |
| Model is calibrated | **Violated after RLHF.** The GPT-4 technical report shows MMLU ECE rising from $\approx 0.007$ pre-RLHF to $\approx 0.074$ post-RLHF. |
| Abstention unavailable | **Violated.** Modern models abstain; abstention voids the bound as stated. |
| Validity oracle exists | Approximated by judges with several points of disagreement. |

## 3. State of the Art

**Theory SOTA (established).**
- Kalai & Vempala, *Calibrated Language Models Must Hallucinate* (STOC 2024, arXiv:2311.14648). For arbitrary facts, any calibrated LM hallucinates at a rate at least the singleton (monofact) rate minus miscalibration and $\tilde{O}(n^{-1/2})$. Constructive, distribution-free. This is a genuine theorem, not a benchmark number.
- Kalai, Nachum, Vempala & Zhang, *Why Language Models Hallucinate* (2025, arXiv:2509.04664). Reduces generation to a binary "Is-It-Valid" classification: generative error rate $\ge 2\times$ the IIV misclassification rate. Extends the account to post-training, arguing binary-scored benchmarks reward guessing over abstention.

**Claimed but unablated.**
- Xu, Jain & Kankanhalli, *Hallucination is Inevitable* (arXiv:2401.11817, 2024) — a diagonalization argument over computable functions. Valid as stated but the "hallucination" it forbids is uncomputability, not factual error; no empirical rate follows. Treat the inevitability claim as non-binding for deployed systems.
- The claim that "training on more data removes the floor" is folklore; singleton rate falls slowly with $n$ under Zipfian $\mu$ and no one has measured its decay curve on a fixed fact family across corpus scales.

**Empirical SOTA (benchmark numbers only).**
- SimpleQA (Wei et al., 2024, arXiv:2411.04368): a 4,326-question adversarially filtered short-fact set. Reported GPT-4o accuracy $\approx 38\%$ with $\approx 1\%$ not-attempted — i.e. roughly 60% wrong answers, with near-zero abstention. These are leaderboard numbers, not measurements against a known singleton rate.
- Semantic entropy (Farquhar et al., *Nature* 2024) detects confabulations at AUROC $\approx 0.79$ on several QA sets — a detector, not a bound.

## 4. What Is Known

- **The theorem holds.** Calibration plus arbitrary facts forces hallucination at the missing-mass rate (Kalai & Vempala 2024). No counterexample has been published.
- **Singleton rates are large in real corpora.** Kalai & Vempala's illustrative estimate: if ~20% of birthday facts appear exactly once in the corpus, a calibrated base model must err on $\ge 20\%$ of birthday queries. Scale: Wikipedia-derived reference and biography counts.
- **Abstention moves the error rate a long way.** Kalai et al. (2025) report on SimpleQA-style evaluation a reasoning model abstaining on ~52% of items with ~26% wrong, versus a sibling model abstaining ~1% with ~75% wrong. Same underlying knowledge, ~3× difference in measured hallucination rate. Scale: frontier-model API evaluations, thousands of items.
- **Post-training destroys calibration.** GPT-4 technical report: MMLU ECE $0.007 \to 0.074$ after RLHF. Independently, Guo et al. (ICML 2017) established that modern networks are systematically overconfident and that temperature scaling largely fixes ECE without changing accuracy — so calibration and accuracy are separable knobs.
- **Models have usable internal signals.** Kadavath et al. (2022, arXiv:2207.05221) show P(True) self-evaluation is well calibrated at 52B scale on several tasks — evidence the missing mass is partly *representable* even when not expressed.
- **Good–Turing is estimable.** $\hat{M}$ converges at $O(n^{-1/2})$ (McAllester & Schapire 2000), so the bound's RHS is not statistically exotic.

## 5. What Is Not Known

- **Theoretically open.** No bound covering *structured* fact families, where $A_i$ is partly predictable from $q_i$. The arbitrary-fact assumption is what makes the singleton rate the right quantity; the correct generalization (missing mass of a *residual* after the learnable component is removed) has no proof. Also open: a lower bound for generators permitted to abstain, i.e. a coverage–hallucination frontier $\text{hall} \ge h(\text{coverage}, \hat{M})$.
- **Empirically open.** Nobody has measured $\hat{M}$ and $\text{hall}$ on the *same* fact family with the *same* corpus at any serious scale. The experiment is runnable on open-data models (OLMo, Pythia) today.
- **Methodologically blocked.** For closed frontier models, $\hat{M}$ is unmeasurable — the corpus is not released. Any claim that GPT-class models are near or above their intrinsic floor is currently unfalsifiable. Also blocked: a definition of "calibrated" for open-ended generation that is estimable from finite samples; ECE over token distributions is not the quantity the theorem uses.

## 6. Why It Is Hard

**The primary obstruction is non-identifiability of the two free terms.** A measured hallucination rate below $\hat{M}$ has two indistinguishable explanations: (a) the model is miscalibrated (the bound's premise fails, so no contradiction), or (b) the fact family is not arbitrary, so the effective missing mass is smaller than $\hat{M}$. Both are unobserved; only their combination is. Without an independent estimate of $\varepsilon_{\text{cal}}$ in the theorem's own sense, the bound cannot be falsified by any single measurement.

**Secondary: absent ground truth on the corpus side.** Counting singletons requires entity-resolved fact extraction over trillions of tokens. Near-duplicate documents inflate counts; paraphrase deflates them. A 2× swing in $n_1$ from extraction choices is plausible and would swamp the effect being tested.

**Tertiary: the evaluation does not measure what it names.** SimpleQA-style scoring counts abstention as failure, so it measures *coverage-weighted* error, not hallucination. The 3× spread in Section 4 is largely a scoring artifact.

## 7. Current Research (as of 2026)

- **Theory.** Kalai and Vempala (OpenAI / Georgia Tech) continue the calibration line; the 2025 paper's socio-technical claim — that benchmark scoring must credit abstention — is the active proposal.
- **Behavioral calibration.** Training models to emit a confidence target and abstain below threshold, evaluated as accuracy-at-coverage rather than raw accuracy *(frontier — verify; several labs report internal versions)*.
- **Open-corpus auditing.** AI2's OLMo line ships training data plus membership tooling, which makes the singleton-count side of the bound measurable for the first time at 7B scale.
- **Detection.** Semantic entropy (Oxford, Gal group) and self-consistency methods are being repositioned as estimators of the missing-mass region rather than as generic detectors *(frontier — verify)*.
- **Retrieval as an escape.** RAG changes $n$ at inference time and should lower $\hat{M}$ for the retrieved family; no published work measures the singleton rate of the retrieval index and compares it against post-RAG hallucination.

## 8. Concrete Next Experiment

**Question:** does measured hallucination track the singleton rate, once calibration is enforced and abstention is disallowed?

- **Scale.** Train a 1.4B-parameter decoder on a fixed 30B-token open corpus (Dolma subset) into which 200,000 synthetic biography facts have been injected — "⟨name⟩ was born on ⟨date⟩" — with mention counts drawn so the singleton fraction $\hat{M}$ takes controlled values $\{0.05, 0.10, 0.20, 0.35, 0.50\}$ across five disjoint name cohorts. Cost: one pretraining run (~2–4k A100-hours), no per-arm retraining.
- **Protocol.** Query 5,000 held-out names per cohort. Force an answer (no abstention; constrained date decoding). Grade by exact match. Temperature-scale the model on a held-out split so token-level ECE $< 0.02$ before evaluation.
- **Control arm.** A sixth cohort with every fact duplicated 5× ($\hat{M} \approx 0$) and identical name statistics, plus a second control where dates are *predictable* from the name prefix (structured family) at the same singleton rate — this separates "missing mass" from "unlearnable".
- **Deciding number.** The slack $\Delta = \text{hall}_{\text{measured}} - \hat{M}$, per cohort. If $\Delta \in [0, 0.05]$ across all five singleton rates and $\text{hall} < 0.05$ in the duplicated control, the bound is tight and predictive. If $\Delta < 0$ at any cohort under verified calibration, the theorem's premises do not transfer to trained transformers, and the arbitrary-fact assumption is the first suspect.

## 9. Key References

- **[Foundational]** Kalai, A. T. & Vempala, S. S. *Calibrated Language Models Must Hallucinate.* STOC 2024. — arXiv:2311.14648
- **[Foundational]** Good, I. J. *The Population Frequencies of Species and the Estimation of Population Parameters.* Biometrika, 1953.
- **[Foundational]** McAllester, D. & Schapire, R. E. *On the Convergence Rate of Good-Turing Estimators.* COLT 2000.
- **[SOTA]** Kalai, A. T., Nachum, O., Vempala, S. S. & Zhang, E. *Why Language Models Hallucinate.* 2025. — arXiv:2509.04664
- **[SOTA]** Wei, J. et al. *Measuring Short-Form Factuality in Large Language Models (SimpleQA).* 2024. — arXiv:2411.04368
- **[Method]** Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. *On Calibration of Modern Neural Networks.* ICML 2017. — arXiv:1706.04599
- **[Method]** Kadavath, S. et al. *Language Models (Mostly) Know What They Know.* 2022. — arXiv:2207.05221
- **[Method]** Farquhar, S., Kossen, J., Kuhn, L. & Gal, Y. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature, 2024.
- **[Method]** Min, S. et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. — arXiv:2305.14251
- **[Contrast]** Xu, Z., Jain, S. & Kankanhalli, M. *Hallucination is Inevitable: An Innate Limitation of Large Language Models.* 2024. — arXiv:2401.11817
- **[Context]** OpenAI. *GPT-4 Technical Report.* 2023. — arXiv:2303.08774

## 10. Worked Example

Take one fact family: birth dates of 1,000,000 people mentioned in a corpus.

1. Count mentions. Suppose 200,000 people are mentioned exactly once: $n_1 = 2\times10^5$, $n = 10^6$, so $\hat{M} = 0.20$. Sampling noise: $O(n^{-1/2}) = 10^{-3}$ — negligible.
2. The theorem predicts a calibrated generator errs on $\ge 20\%$ of birth-date queries drawn from $\mu$, minus miscalibration.
3. Now measure. Ask 5,000 such queries. Observed: 12% wrong, 30% "I don't know", 58% correct.

Naively, $0.12 < 0.20$ and the bound looks violated. It is not, for three separately sufficient reasons, and **you cannot tell which applies**:

- Abstention. Restricted to the 70% answered, error is $0.12/0.70 = 17.1\%$ — closer to 20%, but the theorem says nothing about a generator that declines.
- Miscalibration. If post-RLHF ECE is $\approx 0.074$ (the GPT-4-report value), $\varepsilon_{\text{cal}}$ alone can absorb 7 points, leaving a predicted floor of 13% — indistinguishable from 12% within judge noise.
- Structure. Birth *years* correlate with name cohort and era, so the family is not arbitrary; the effective missing mass may be 8%, not 20%.

Each explanation is consistent with the same three numbers. **That is the obstruction:** the bound has two unmeasured slack terms and one unverified structural premise, so a single deployed measurement can never confirm or refute it. Only the Section 8 design — synthetic facts with a known singleton rate, enforced calibration, forced answering, and a structured-family control — closes all three at once.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*