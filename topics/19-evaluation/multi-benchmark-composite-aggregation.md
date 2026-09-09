---
id: 19-evaluation/multi-benchmark-composite-aggregation
title: "Aggregation Rules for Multi-Benchmark Composite Scores"
topic: 19-evaluation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Aggregation Rules for Multi-Benchmark Composite Scores

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/multi-benchmark-composite-aggregation` · **Status:** methodologically-blocked

## 1. Problem Statement

- **Input.** A score matrix $S \in \mathbb{R}^{M \times B}$: $M$ models evaluated on $B$ benchmarks, each benchmark reporting a scalar on its own scale (accuracy, exact match, pass@1, Elo, F1), each estimated from a finite item set.
- **Output.** A composite $C: \mathbb{R}^{B} \to \mathbb{R}$ inducing a total order over models — the thing leaderboards print as one number.
- **Objective.** Pick $C$ such that the induced ranking predicts an external target (downstream task utility, deployment win rate, safety incident rate) better than any single benchmark, and is stable to perturbations that carry no information about that target.
- **Solved would mean:** a rule with a stated loss function it minimizes, a proof or measurement of its invariances, and evidence that its ranking transfers to a held-out target.

Three variants, different difficulty:

| Variant | Question | Difficulty source |
|---|---|---|
| **Measurement** | What is $C$ estimating? | No ground-truth "general capability" scalar exists |
| **Method** | Which rule minimizes rank error under noise? | Runnable, mostly unrun at scale |
| **Theory** | Which axioms can $C$ jointly satisfy? | Arrow-type impossibility applies to the ordinal case |

The measurement variant blocks the other two: without a target, "better aggregation" has no loss.

## 2. Formal Setting

Model $m \in \mathcal{M}$, $|\mathcal{M}| = M$; benchmark $b \in \mathcal{B}$, $|\mathcal{B}| = B$; benchmark $b$ has items $\{x_{b,i}\}_{i=1}^{n_b}$ and a per-item score $\ell_b(m, x) \in [0,1]$.

**As measured**, the benchmark score is a sample mean:
$$\hat s_{mb} = \frac{1}{n_b}\sum_{i=1}^{n_b} \ell_b(m, x_{b,i}), \qquad \widehat{\mathrm{SE}}(\hat s_{mb}) = \sqrt{\frac{\hat s_{mb}(1-\hat s_{mb})}{n_b}}$$
which is the binomial SE and ignores item clustering and prompt-template variance; both are real and both inflate the true SE.

A composite is a normalizer $\phi_b$ plus a pooling rule:
$$C(m) = \Psi\big(\phi_1(\hat s_{m1}), \dots, \phi_B(\hat s_{mB}); w\big)$$

Normalizers in actual use:
- **Identity:** $\phi_b(s) = s$ (GLUE, MMLU macro-average).
- **Random-baseline corrected:** $\phi_b(s) = (s - r_b)/(1 - r_b)$, $r_b$ = chance accuracy (Open LLM Leaderboard v2).
- **Z-score:** $\phi_b(s) = (s - \mu_b)/\sigma_b$, $\mu_b, \sigma_b$ over the *current model pool* — so the score of model $m$ depends on which other models were submitted.
- **Rank / win rate:** $\phi_b(s_{mb}) = \frac{1}{M-1}\sum_{m' \neq m} \mathbb{1}[s_{mb} > s_{m'b}]$ (HELM mean win rate).

Pooling: arithmetic mean $\Psi = \sum_b w_b \phi_b$; geometric mean; Borda count; Kemeny consensus $\arg\min_\pi \sum_b d_{\mathrm{KT}}(\pi, \pi_b)$ with $d_{\mathrm{KT}}$ the Kendall-tau distance.

**Assumptions, and their status:**

| Assumption | Status |
|---|---|
| Items i.i.d. within a benchmark | **Violated** — MMLU, BBH are topic-clustered; SE understated |
| Benchmarks measure distinct constructs | **Violated** — cross-benchmark correlations of $0.7$–$0.95$ among open models |
| $\hat s_{mb}$ is contamination-free | **Violated** — train-set overlap is undisclosed for most released models |
| Score matrix is complete | **Violated** — leaderboards impute or drop missing cells |
| $\phi_b$ is scale-comparable across $b$ | **Undefined**, not merely violated — this is the block |
| Equal weights $w_b = 1/B$ encode a preference | **False** — equal weight on 57 MMLU subjects and 1 coding benchmark is a strong, unstated prior |

## 3. State of the Art

**Established.**
- Arrow's impossibility theorem (Arrow, 1951): with $\geq 3$ alternatives, no ordinal rule satisfies unrestricted domain, Pareto efficiency, independence of irrelevant alternatives, and non-dictatorship simultaneously. Rank-based aggregation (Borda, mean win rate) is exactly in scope; cardinal averaging escapes only by assuming inter-benchmark comparability, which is the disputed step.
- Kemeny consensus is NP-hard for $\geq 4$ voters (Bartholdi, Tovey & Trick, 1989; Dwork, Kumar, Naor & Sivakumar, WWW 2001). Practical instances at $M \approx 100$ are solvable by ILP.
- Colombo, Noiry, Irurozki & Clémençon (NeurIPS 2022) applied Kemeny aggregation to NLP benchmark suites and showed the mean-aggregate ranking differs from the consensus ranking on GLUE-family data.

**Claimed but unablated.**
- HELM's mean win rate (Liang et al., TMLR 2023) is argued to be robust to scale mismatch. Robustness to *scenario selection* — adding or removing scenarios — is not ablated in the paper.
- Open LLM Leaderboard v2 (Hugging Face, 2024) switched to random-baseline normalization to "make the leaderboard steep again." The change is motivated but not validated against any external target.
- Item-response-theory aggregation (Lalor et al., EMNLP 2016; Vania et al., ACL 2021; Polo et al., *tinyBenchmarks*, ICML 2024) gives a latent-ability scalar $\theta$. The 1PL/2PL unidimensionality assumption that makes $\theta$ meaningful is asserted, not tested against a held-out utility measure.

**Benchmark-number-only results.** Every published leaderboard "average" — GLUE, SuperGLUE, MMLU, Open LLM v1/v2, HELM — is a number with no accompanying claim about what it predicts. None has a validated criterion.

## 4. What Is Known

- **Rule choice moves rankings materially.** Alzahrani et al. (ACL 2024) perturbed only surface details (answer-option order, prompt format) on Open LLM Leaderboard benchmarks and observed multi-position rank changes among ~15 top models. Scale: full leaderboard model pool, 2024.
- **Benchmark scores are low-rank.** Ruan, Maddison & Hashimoto (NeurIPS 2024) fit PCA to ~100 open models × standard benchmarks; roughly 3 principal components capture the dominant share of variance. Implication: most aggregation rules are approximating the same first PC, and their disagreement lives in the noise-adjacent residual.
- **Sampling noise is not negligible relative to leaderboard gaps.** Miller (2024, arXiv:2411.00640) shows typical eval suites with $n \sim 10^3$ items give standard errors of $\approx 1$–$1.5$ percentage points; published model gaps are frequently smaller.
- **Small subsets suffice for point estimates.** *tinyBenchmarks* recovers MMLU accuracy to within ~2 points absolute using ~100 items. Aggregate *rankings* are far less stable than aggregate *point estimates* — the same variance that is tolerable for a score flips adjacent ranks.
- **Benchmark selection is itself a degree of freedom.** Dehghani et al., *The Benchmark Lottery* (2021), documents that the choice of which tasks enter a suite determines which method wins.
- **Averaging destroys per-instance information.** Burnell et al. (*Science*, 2023) argue that reporting only aggregates hides subgroup failures that instance-level release would expose.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no agreed target variable $Y(m)$ against which a composite's validity could be scored. Without $Y$, "which aggregation rule is better" has no loss function, and every proposed rule is defended by appeal to axioms rather than fit. This is not a compute problem; it is a construct-validity problem.
- **Empirically open.** Given *any* fixed proxy target — human preference win rate, agentic task success, deployment A/B lift — the rule comparison is a runnable experiment. It has not been run across the full rule family at $M > 50$ with bootstrap controls.
- **Theoretically open.** Whether a cardinal aggregation rule exists that is (i) invariant to monotone per-benchmark reparameterization, (ii) invariant to adding a benchmark perfectly correlated with an existing one, and (iii) non-dictatorial. Conditions (i) and (ii) together are close to Arrow's IIA plus a duplication axiom; no impossibility proof or construction is published.
- **Empirically open.** The contamination-adjusted composite: no rule currently discounts a benchmark by its estimated train-set overlap, and the estimator for that overlap is itself contested.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by non-identifiability**, not compute.

1. **No criterion.** "General capability" is not observed. Any $Y$ you pick (Arena Elo, agentic success) is itself an aggregate with the same problem one level up.
2. **Non-identifiability under low rank.** Because the score matrix is near rank-3, many weight vectors $w$ produce near-identical rankings on the observed pool while diverging sharply on out-of-pool models. The data cannot select among them.
3. **Pool dependence.** Z-score and win-rate normalizers make $C(m)$ a function of the submission set. Adding a weak model changes the ranking of strong ones — a direct IIA violation, present in deployed leaderboards.
4. **Noise amplification.** $\sigma_b$ estimated over a small model pool is the same order as $\widehat{\mathrm{SE}}(\hat s_{mb})$; dividing by it converts measurement error into rank churn (see §10).
5. **Strategic response.** Once $C$ is public, training targets it (Gibbard–Satterthwaite-style manipulability), so validity measured pre-publication does not survive publication.

## 7. Current Research (as of 2026)

- **IRT and latent-ability aggregation.** Continuation of *tinyBenchmarks* (Polo, Weber, Choshen, Sun, Xu, Yurochkin) toward multidimensional IRT, where $\theta \in \mathbb{R}^k$ replaces a scalar. *(frontier — verify)*
- **Observational scaling laws** (Ruan, Maddison, Hashimoto, Stanford) as an aggregation substitute: project benchmarks onto PCs, then predict downstream performance from PC coordinates. This reframes aggregation as regression against a target, which is the right move.
- **Social-choice-theoretic evaluation** (Colombo, Irurozki, Clémençon and collaborators; ENSAE/CentraleSupélec) — Kemeny and two-sided ranking with statistical guarantees.
- **Uncertainty-first leaderboards.** Post-Miller (2024) adoption of error bars and paired-difference tests; LM Evaluation Harness (Biderman et al., 2024) exposes stderr. Adoption on public boards remains partial. *(frontier — verify)*
- **Preference-elicited weights.** Deriving $w$ from stated deployment priorities rather than uniformity. Discussed since Ethayarajh & Jurafsky (EMNLP 2020); no deployed instance known.

## 8. Concrete Next Experiment

**Rule-induced flip rate versus sampling flip rate.**

- **Scale.** $M = 60$ open-weight models spanning 1B–400B parameters, $B = 20$ benchmarks with item-level scores retained (Open LLM v2 suite + HELM Lite + 4 agentic benchmarks). Total cost: one full eval sweep, order $10^4$ GPU-hours — affordable, and much of it already cached in public harness runs.
- **Treatment arm.** Compute the composite under $K = 12$ rules: {identity, random-baseline, z-score, min-max} × {arithmetic mean, geometric mean, Borda}. For every model pair $(m, m')$, record whether the ordering is consistent across all 12 rules. Let $\pi_{\text{rule}}$ = fraction of the $\binom{60}{2} = 1770$ pairs that flip under *some* rule.
- **Control arm.** Fix one rule (random-baseline + arithmetic mean). Bootstrap items within each benchmark, 1000 replicates. Let $\pi_{\text{noise}}$ = fraction of pairs whose ordering flips in $\geq 5\%$ of replicates.
- **The deciding number.** $\rho = \pi_{\text{rule}} / \pi_{\text{noise}}$.
  - $\rho \lesssim 1$: rule choice adds no more disagreement than sampling noise. Aggregation is a non-problem; report any rule with error bars.
  - $\rho \gtrsim 3$: the choice of rule dominates the evidence, and every published single-number leaderboard is reporting an artifact of an unjustified normalizer. Pre-register this threshold.
- **Secondary readout.** Regress each rule's ranking against held-out agentic success rate; report Spearman $\rho_s$ per rule. If the spread in $\rho_s$ across rules exceeds its bootstrap CI width, one rule is genuinely better and the problem moves from *methodologically blocked* to *empirically open*.

## 9. Key References

- **[Foundational]** Kenneth J. Arrow. *Social Choice and Individual Values.* Wiley, 1951.
- **[Foundational]** John Kemeny. *Mathematics without Numbers.* Daedalus, 1959.
- **[Foundational]** John Bartholdi, Craig Tovey & Michael Trick. *Voting schemes for which it can be difficult to tell who won the election.* Social Choice and Welfare, 1989.
- **[Foundational]** Cynthia Dwork, Ravi Kumar, Moni Naor & D. Sivakumar. *Rank Aggregation Methods for the Web.* WWW, 2001.
- **[SOTA]** Percy Liang et al. *Holistic Evaluation of Language Models.* TMLR, 2023. — arXiv:2211.09110
- **[SOTA]** Pierre Colombo, Nathan Noiry, Ekhine Irurozki & Stephan Clémençon. *What are the best systems? New perspectives on NLP Benchmarking.* NeurIPS, 2022. — arXiv:2202.03799
- **[SOTA]** Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu & Mikhail Yurochkin. *tinyBenchmarks: evaluating LLMs with fewer examples.* ICML, 2024. — arXiv:2402.14992
- **[SOTA]** Yangjun Ruan, Chris J. Maddison & Tatsunori Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[SOTA]** Evan Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024. — arXiv:2411.00640
- **[Critique]** Mostafa Dehghani, Yi Tay, Alexey Gritsenko, Zhe Zhao, Neil Houlsby, Fernando Diaz, Donald Metzler & Oriol Vinyals. *The Benchmark Lottery.* 2021. — arXiv:2107.07002
- **[Critique]** Kawin Ethayarajh & Dan Jurafsky. *Utility is in the Eye of the User: A Critique of NLP Leaderboards.* EMNLP, 2020. — arXiv:2009.13888
- **[Critique]** Norah Alzahrani et al. *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards.* ACL, 2024. — arXiv:2402.01781
- **[Critique]** Ryan Burnell et al. *Rethink reporting of evaluation results in AI.* Science, 2023.
- **[Survey]** Stella Biderman et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* 2024. — arXiv:2405.14782

## 10. Worked Example

Three models, three benchmarks. Scores are plausible mid-2024 open-weight values.

| Model | MMLU ($r=0.25$) | GSM8K ($r=0$) | HellaSwag ($r=0.25$) |
|---|---|---|---|
| A | 0.68 | 0.35 | 0.84 |
| B | 0.62 | 0.56 | 0.80 |
| C | 0.65 | 0.44 | 0.86 |

**Rule 1 — raw arithmetic mean.**
$C_A = 0.623$, $C_B = 0.660$, $C_C = 0.650$. Ranking: **B > C > A**.

**Rule 2 — random-baseline normalized mean**, $\phi_b(s) = (s-r_b)/(1-r_b)$.
$C_A = (0.573 + 0.350 + 0.787)/3 = 0.570$;
$C_B = (0.493 + 0.560 + 0.733)/3 = 0.5956$;
$C_C = (0.533 + 0.440 + 0.813)/3 = 0.5956$.
Ranking: **B = C > A** — an exact tie to four decimals.

**Rule 3 — z-score mean** (statistics over this 3-model pool).
Per-benchmark $\sigma$: MMLU $0.0245$, GSM8K $0.0860$, HellaSwag $0.0249$.
$\sum_b z_{Ab} = +1.225 - 1.163 + 0.267 = +0.329$;
$\sum_b z_{Bb} = -1.225 + 1.279 - 1.336 = -1.282$;
$\sum_b z_{Cb} = \phantom{+}0.000 - 0.116 + 1.069 = +0.953$.
Ranking: **C > A > B**.

**Rule 4 — mean win rate.** $C_A = 0.500$, $C_B = 0.333$, $C_C = 0.667$. Ranking: **C > A > B**.

**What the example shows.** Model B is first under the raw mean and last under the z-score mean, with no change to any measurement. The four rules produce three different orderings and one tie. Now add noise: MMLU test has $n = 14{,}042$ items, so $\widehat{\mathrm{SE}} \approx 0.004$; GSM8K has $n = 1{,}319$, so $\widehat{\mathrm{SE}} \approx 0.014$. The MMLU spread across the pool is $\sigma = 0.0245$ — about six SEs, but the *pairwise* MMLU gaps (0.03) are only $\approx 5$ SEs, and GSM8K's $\sigma = 0.086$ is only $6$ SEs. Dividing by a pool $\sigma$ that is within an order of magnitude of the measurement error is what makes Rule 3 flip B from first to last. The obstruction is visible: there is no fact in the table that says which of the four orderings is correct, because nothing in the table is the thing the leaderboard claims to rank.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*