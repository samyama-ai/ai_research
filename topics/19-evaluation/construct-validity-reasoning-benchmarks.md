---
id: 19-evaluation/construct-validity-reasoning-benchmarks
title: "Construct Validity of Reasoning Benchmarks"
topic: 19-evaluation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Construct Validity of Reasoning Benchmarks

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/construct-validity-reasoning-benchmarks` · **Status:** methodologically-blocked

## 1. Problem Statement

A reasoning benchmark reports a scalar: accuracy on GSM8K, MATH, ARC-AGI, HLE. The claim attached to that scalar is that it measures a latent ability ("mathematical reasoning", "abstract reasoning"). Construct validity is the question of whether it does — whether the score tracks the named ability rather than memorization of the test set, sensitivity to prompt surface, or a single undifferentiated "general capability" factor that every benchmark measures equally.

Three variants, with different difficulty:

- **Measurement variant.** Given a model $M$ and benchmark $B$, estimate what fraction of $\mathrm{Var}(\hat S(M,B))$ across models is attributable to the named construct versus construct-irrelevant sources. Runnable today; the estimator is not agreed on.
- **Method variant.** Build a benchmark whose score is invariant to construct-irrelevant transformations and discriminates between distinct reasoning abilities. Partially attacked (functional/procedural benchmarks); no accepted acceptance test.
- **Theory variant.** State conditions under which a latent ability is *identifiable* from behavioral scores alone. Open, and probably negative without interventions on the model or its training data.

Solving it means: a procedure that, given a benchmark and a model population, returns a defensible validity certificate — not a leaderboard.

## 2. Formal Setting

Let $C$ be the target construct with latent value $\theta_C(M) \in \mathbb{R}$ per model $M$. A benchmark is a finite item set $B = \{x_1,\dots,x_n\}$ with scorer $v(\cdot,\cdot) \in \{0,1\}$. The measured score is

$$\hat S(M,B) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{E}_{y \sim M(\cdot\mid \pi(x_i))}\big[v(y, x_i)\big],$$

where $\pi$ is the prompt template (format, ordering, few-shot prefix, decoding parameters). $\pi$ is part of the measurement and is almost never reported as such.

A 2PL item-response model gives the per-item probability

$$p_i(M) = \sigma\!\big(a_i(\theta_C(M) - b_i)\big),$$

with discrimination $a_i$ and difficulty $b_i$. Construct-irrelevant variance enters as nuisance loadings:

$$\mathrm{logit}\,p_i(M) = a_i\theta_C(M) + \sum_k c_{ik}\,\eta_k(M) - a_ib_i,$$

where $\eta_1$ = contamination (item-level memorization), $\eta_2$ = format robustness, $\eta_3$ = instruction-following, $\eta_4$ = long-context retrieval.

Two measurable quantities:

- **Invariance gap** over a set $\mathcal{T}$ of construct-preserving rewrites (renumbering, paraphrase, option permutation, template change): $\Delta_{\mathcal{T}}(M) = \mathbb{E}_{T\sim\mathcal{T}}\big|\hat S(M,T(B)) - \hat S(M,B)\big|$.
- **Campbell–Fiske contrast** on a multitrait–multimethod (MTMM) matrix: monotrait–heteromethod correlation $\rho_{\text{mono}}$ (same construct, different item format) versus heterotrait–monomethod $\rho_{\text{het}}$ (different construct, same format). Validity requires $\rho_{\text{mono}} > \rho_{\text{het}}$ after disattenuation for reliability $r$: $\tilde\rho = \rho/\sqrt{r_1r_2}$.

Assumptions the framework rests on, and their status:

| Assumption | Status |
|---|---|
| Items i.i.d. from the construct domain | Violated — items are scraped/curated, difficulty is non-stationary |
| Local independence given $\theta_C$ | Violated by contamination: $\eta_1$ is item-specific, not a global ability |
| Scorer $v$ is error-free | Violated — 6.49% of sampled MMLU items contain errors (Gema et al., NAACL 2025) |
| Score invariant to $\pi$ | Violated — up to 76 accuracy points spread across plausible formats (Sclar et al., ICLR 2024) |
| Unidimensionality of the model population | Violated — ~3 principal components explain >95% of benchmark variance across 100+ models (Ruan et al., NeurIPS 2024) |

Every assumption needed to read a score as a construct measurement is known-false at the measured scale.

## 3. State of the Art

**Established (replicated, ablated).**
- Format sensitivity is real and large. FormatSpread (Sclar et al., ICLR 2024): LLaMA-2-13B varies by up to 76 accuracy points across semantically equivalent prompt formats. Alzahrani et al. (ACL 2024) independently show leaderboard *rank order* flips under benign perturbations such as answer-option reordering.
- Test-set label error is measurable and non-trivial. Northcutt et al. (NeurIPS D&B 2021): 3.4% average label error across 10 standard test sets. Gema et al. (NAACL 2025): 57% of sampled MMLU Virology items are erroneous.
- Contamination is detectable with a controlled false-positive rate. Oren et al. (ICLR 2024) test exchangeability of item order in the training distribution and give provable guarantees under stated assumptions.

**Claimed but not fully ablated.**
- "Models pattern-match rather than reason." GSM-Symbolic (Mirzadeh et al., ICLR 2025) reports variance across templated instantiations of GSM8K items and drops of up to 65% on GSM-NoOp (irrelevant clause inserted). The confound — instruction-following and distractor-robustness are themselves construct-relevant for some definitions of reasoning — is not separated.
- "Benchmark saturation reflects overfitting." GSM1k (Zhang et al., 2024) found up to ~13% drops for some model families (Mistral, Phi) with near-zero gaps for frontier Gemini/GPT/Claude models — evidence that overfitting is family-specific, not universal.

**Benchmark-number-only results.** ARC-AGI-2 (2025), FrontierMath (Glazer et al., 2024) and Humanity's Last Exam (Phan et al., 2025) headline numbers exist without published validity analysis: no reliability estimate, no invariance gap, no MTMM. They are difficulty claims, not validity claims.

## 4. What Is Known

- **Reliability is usually unreported and often small relative to reported gaps.** GSM8K test has $n=1319$; at $p=0.95$ the binomial standard error is $\sqrt{0.95\cdot0.05/1319} = 0.60$ pp, so a 95% CI is ±1.2 pp. Leaderboard gaps below that are noise (Miller, *Adding Error Bars to Evals*, 2024).
- **Cluster variance dominates sampling variance.** Per-template resampling in GSM-Symbolic produces accuracy distributions with spreads of several points at 1–70B scale — an order of magnitude above the i.i.d. binomial term.
- **Discriminant validity fails at population scale.** Across 100+ open models, downstream benchmark variance is near-rank-3 (Ruan et al., NeurIPS 2024); Ilić & Gignac (*Intelligence*, 2024) report a dominant general factor across LLM benchmark batteries. Distinct "reasoning" benchmarks do not separate distinct abilities.
- **Cleaned benchmarks still expose residual failure.** Platinum benchmarks (Vendrow et al., 2025) revise items to remove ambiguity and label error; frontier models retain non-zero error on revised grade-school arithmetic, showing the failures are not purely annotation artifacts.
- **Frequency effects are systematic.** McCoy et al. (PNAS 2024): GPT-4 accuracy on shift ciphers is far higher at the common rot-13 shift than at rare shifts — a task-frequency effect that a construct-valid "cipher reasoning" score should not show.

## 5. What Is Not Known

- **Theoretically open.** Whether $\theta_C$ is identifiable from behavioral scores alone. With per-item nuisance loadings $c_{i1}\eta_1$ (contamination), the IRT likelihood admits a continuum of $(\theta,\eta)$ decompositions fitting the same response matrix. No non-identifiability theorem has been stated for this setting, and no sufficient condition for identifiability either. Both directions are unproved.
- **Empirically open.** No one has run a full MTMM design — $\geq 3$ reasoning constructs × $\geq 3$ item formats × a large model population — on frontier models. The experiment is affordable (Section 8) and unrun.
- **Methodologically blocked (the dominant blockage).** There is no accepted operational definition of "reasoning" that fixes which transformations are construct-preserving. Is robustness to an irrelevant clause part of reasoning, or a separate instruction-following trait? Until $\mathcal{T}$ is specified, $\Delta_{\mathcal{T}}$ is not a well-defined quantity and any validity estimate is definitional, not empirical.

## 6. Why It Is Hard

**Non-identifiability plus absent ground truth.** There is no external criterion for $\theta_C$ in models. In human psychometrics, criterion validity anchors on outcomes measured outside the test (grades, job performance). For LLMs the only criterion available is another benchmark, so validity arguments are circular: benchmark A is validated by correlation with benchmark B, which shares A's format, its scraping provenance, and often its contamination.

Compounding this: the contamination nuisance $\eta_1$ is *item-specific*, so it is absorbed into item difficulty $b_i$ by any standard IRT fit and cannot be detected by goodness-of-fit. Separating it requires access to the training corpus — which for frontier models is unavailable — or a held-out set built after the training cutoff, which is a one-shot resource that decays the moment it is published.

## 7. Current Research (as of 2026)

- **Functional / procedural benchmarks.** Item generators rather than item lists (GSM-Symbolic, Apple; functional MATH variants, Srivastava et al. 2024). Directly targets $\Delta_{\mathcal{T}}$ for the renumbering subgroup.
- **Post-cutoff and canary-protected sets.** LiveBench-style rolling refresh; the design tradeoff between refresh rate and comparability across time is unresolved *(frontier — verify current maintainers)*.
- **Psychometric imports.** IRT-based leaderboards (Rodriguez et al., ACL 2021; Vania et al., ACL 2021) and factor-analytic capability spaces (Ruan et al., 2024). Ongoing at Stanford CRFM, EleutherAI (lm-evaluation-harness reproducibility work, Biderman et al. 2024), and the measurement-theory line from Jacobs & Wallach (FAccT 2021).
- **Validity-first critique.** Raji et al. (NeurIPS D&B 2021) and Bowman & Dahl (NAACL 2021) argue benchmark generality claims are unsupported by construct. Widely cited, rarely operationalized.

## 8. Concrete Next Experiment

**An MTMM study for reasoning constructs.**

- **Scale.** 3 traits (multi-step arithmetic word problems; deductive/syllogistic entailment; abstract sequence induction) × 3 methods (free-response with exact-match; 4-way multiple choice; verify-a-given-candidate-solution as binary judgment) = 9 cells. 400 items per cell, all *generated procedurally* so that each trait's items exist in all three methods with matched item content. 30 models spanning 3 orders of magnitude of training compute, ≥10 of them frontier-tier. Total: 30 × 3600 = 108k items, ~5 samples each for reliability → ~540k calls, roughly $4–10k in API cost at 2026 prices.
- **Control arm.** A fourth "trait" that is *not* reasoning: recall of low-frequency factual entities, rendered in the same three methods. This gives the heterotrait–monomethod baseline, i.e. how much correlation is explained by shared format alone.
- **Deciding number.** Fit a confirmatory factor model with trait and method factors. Report $\phi = \dfrac{\sigma^2_{\text{trait}}}{\sigma^2_{\text{trait}} + \sigma^2_{\text{method}}}$, the share of reliable score variance loading on trait rather than method. **Pre-register the threshold $\phi \geq 0.7$.** If $\phi < 0.5$ — the outcome the format-sensitivity literature predicts — reasoning benchmark scores are measuring presentation format at least as much as reasoning, and single-number leaderboards are unsupported. Secondary readout: disattenuated $\tilde\rho_{\text{mono}}$ should exceed $\tilde\rho_{\text{het}}$ in all 9 comparisons; any failure localizes which construct is not separable.

## 9. Key References

- **[Foundational]** Cronbach, L. J. & Meehl, P. E. *Construct Validity in Psychological Tests.* Psychological Bulletin, 1955.
- **[Foundational]** Campbell, D. T. & Fiske, D. W. *Convergent and Discriminant Validation by the Multitrait-Multimethod Matrix.* Psychological Bulletin, 1959.
- **[Foundational]** Messick, S. *Validity of Psychological Assessment.* American Psychologist, 1995.
- **[Foundational]** Raji, I. D., Denton, E., Bender, E. M., Hanna, A. & Paullada, A. *AI and the Everything in the Whole Wide World Benchmark.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2111.15366
- **[Foundational]** Jacobs, A. Z. & Wallach, H. *Measurement and Fairness.* ACM FAccT, 2021.
- **[Foundational]** Bowman, S. R. & Dahl, G. E. *What Will it Take to Fix Benchmarking in Natural Language Understanding?* NAACL, 2021. — arXiv:2104.02145
- **[SOTA]** Sclar, M., Choi, Y., Tsvetkov, Y. & Suhr, A. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* ICLR, 2024. — arXiv:2310.11324
- **[SOTA]** Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S. & Farajtabar, M. *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.* ICLR, 2025. — arXiv:2410.05229
- **[SOTA]** Zhang, H. et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS, 2024. — arXiv:2405.00332
- **[SOTA]** Oren, Y., Meister, N., Chatterji, N., Ladhak, F. & Hashimoto, T. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[SOTA]** Ruan, Y., Maddison, C. J. & Hashimoto, T. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[SOTA]** Gema, A. P. et al. *Are We Done with MMLU?* NAACL, 2025. — arXiv:2406.04127
- **[SOTA]** Vendrow, J., Vendrow, E., Beery, S. & Madry, A. *Do Large Language Model Benchmarks Test Reliability?* 2025. — arXiv:2502.03461
- **[SOTA]** Miller, E. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024. — arXiv:2411.00640
- **[SOTA]** McCoy, R. T., Yao, S., Friedman, D., Hardy, M. & Griffiths, T. L. *Embers of Autoregression Show How Large Language Models Are Shaped by the Problem They Are Trained to Solve.* PNAS, 2024.
- **[Survey]** Chollet, F. *On the Measure of Intelligence.* 2019. — arXiv:1911.01547
- **[Survey]** Biderman, S. et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* 2024. — arXiv:2405.14782
- **[Survey]** Liang, P. et al. *Holistic Evaluation of Language Models.* TMLR, 2023. — arXiv:2211.09110

## 10. Worked Example

**Reading a 1.5-point GSM8K gap.**

Model $A$ scores 95.2%, model $B$ scores 93.7% on the GSM8K test set ($n = 1319$). The reported difference is 1.5 pp.

Variance budget, in percentage points:

| Source | Estimate (pp, 1 s.d.) | Basis |
|---|---|---|
| Binomial sampling | 0.60 | $\sqrt{0.95 \cdot 0.05 / 1319}$ |
| Decoding stochasticity | ~0.3 | resampling at $T>0$ |
| Prompt-template choice | 2–5 | FormatSpread spread, conservatively truncated to competent formats |
| Item instantiation (renumbering) | 2–4 | GSM-Symbolic per-template spread |
| Label/scorer error ceiling | ~1–2 | Northcutt 3.4% avg; Platinum-GSM8K revisions |

Combining the three construct-irrelevant terms in quadrature at their low ends: $\sqrt{2^2 + 2^2 + 1^2} \approx 3.0$ pp. The observed gap of 1.5 pp is half of one standard deviation of construct-irrelevant variation, and about 2.5× the only error bar anyone reports (the binomial one).

Now the part that makes the obstruction visible. Suppose you fix the template and the instantiation seed, driving the format and renumbering terms to zero — the gap becomes stable and reproducible. It is still not a construct measurement, because a stable gap is exactly what item-level memorization produces: if $A$ saw 300 more GSM8K items in pretraining than $B$, then $c_{i1}\eta_1$ contributes a deterministic offset of $300/1319 \approx 22.7\%$ of items, more than enough to generate 1.5 pp of advantage on the ~7% of items either model would otherwise miss. From the response matrix alone, "$A$ reasons slightly better" and "$A$ memorized 300 more items" produce identical likelihoods under the IRT fit — the second is absorbed into $b_i$. Distinguishing them requires the training corpus, which is unavailable, or a post-cutoff replica, which is single-use.

The gap is not noisy. It is unidentified.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*