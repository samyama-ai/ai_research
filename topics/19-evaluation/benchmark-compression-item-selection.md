---
id: 19-evaluation/benchmark-compression-item-selection
title: "Optimal Item Selection for Benchmark Compression"
topic: 19-evaluation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Item Selection for Benchmark Compression

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/benchmark-compression-item-selection` · **Status:** empirically-open

## 1. Problem Statement

**Input.** A benchmark item pool $\mathcal{I}$ with $|\mathcal{I}| = N$ items, a historical set of models $\mathcal{M}_{\text{fit}}$ with their full response matrix on $\mathcal{I}$, and a budget $k \ll N$.

**Output.** A subset $S \subseteq \mathcal{I}$, $|S| = k$, plus an estimator $\hat{s}_S: \{0,1\}^k \to [0,1]$ that maps a new model's responses on $S$ to a prediction of its score on the full pool.

**Decision predicate.** Does there exist a $(S, \hat{s}_S)$ pair whose error on *models not seen at selection time* is provably below a target $\epsilon$, and does it beat uniform random sampling of $k$ items by more than the variance of the random baseline?

Three variants, routinely conflated:

- **Measurement.** What is the right loss — absolute score error, rank correlation, or pairwise decision error on the comparisons users actually make? These are not monotone in one another.
- **Method.** Given a loss, find a selection algorithm that generalizes to future models. This is the empirically open core.
- **Theory.** Characterize when a $k$-item subset can be information-equivalent to $N$ items, and give a lower bound on $k$ as a function of the latent dimension of model ability and the item-parameter spread.

## 2. Formal Setting

Let $\mathcal{M}$ be a population of models with distribution $\mathcal{D}$. For model $m$ and item $i$, the response is $Y_{mi} \in \{0,1\}$ (correct/incorrect after the benchmark's own grader; for graded metrics, $Y_{mi} \in [0,1]$). The full-pool score is measured as

$$s(m) = \frac{1}{N}\sum_{i=1}^{N} Y_{mi},$$

i.e. the number the leaderboard prints, including all grader noise.

**Compression objective (expected-loss form).**

$$\min_{|S| = k,\ \hat{s}_S} \ \mathbb{E}_{m \sim \mathcal{D}}\left[\left(\hat{s}_S(Y_{m,S}) - s(m)\right)^2\right].$$

**Ranking objective.** For $m, m' \sim \mathcal{D}$, the pairwise sign-error rate

$$\rho = \Pr\left[\operatorname{sign}(\hat{s}_S(m) - \hat{s}_S(m')) \neq \operatorname{sign}(s(m) - s(m'))\right],$$

optionally restricted to pairs with $|s(m)-s(m')| > \delta$ (below $\delta$, the full benchmark itself is not decisive).

**IRT parameterization.** The dominant model family is 2PL item response theory: model ability $\theta_m \in \mathbb{R}^d$, item discrimination $a_i$ and difficulty $b_i$, with

$$\Pr[Y_{mi} = 1] = \sigma\!\left(a_i^\top \theta_m - b_i\right).$$

Fisher information of item $i$ at ability $\theta$ is $I_i(\theta) = a_i^2 p_i(\theta)(1-p_i(\theta))$ for $d=1$; classical adaptive-testing selection maximizes $\sum_{i \in S} I_i(\theta)$ at the ability region of interest.

**Estimator forms actually used.** (a) Plain subset mean $\frac{1}{k}\sum_{i \in S} Y_{mi}$; (b) IRT ability estimate $\hat\theta_m$ pushed back through fitted item curves to predict the full-pool mean; (c) ridge/linear regression $\hat{s}_S = w^\top Y_{m,S} + b$ fit on $\mathcal{M}_{\text{fit}}$.

**Assumptions, and which are violated.**

| Assumption | Status in practice |
|---|---|
| $\mathcal{M}_{\text{fit}}$ and future models are i.i.d. from one $\mathcal{D}$ | **Violated.** Model families shift yearly; post-training changes the correlation structure of items. |
| Item parameters $(a_i, b_i)$ are model-invariant | **Violated.** An item memorized during pretraining flips its difficulty for contaminated models only. |
| Unidimensional ability $\theta_m$ | **Violated.** Multi-subject benchmarks (MMLU) need $d>1$; how much more is unsettled. |
| Local independence of items given $\theta$ | **Violated** where items share a passage, template, or source document. |
| $s(m)$ is the ground truth | **Approximate.** Label errors in the pool put a floor on any estimator's usefulness. |

## 3. State of the Art

**Established (reproduced, ablated against random-subset controls).**

- **tinyBenchmarks** (Polo et al., ICML 2024). $k = 100$ items per scenario, IRT-based estimators, reported average absolute error $\approx 2\%$ in reconstructing Open LLM Leaderboard and HELM scores. Includes a random-sampling control; the IRT estimator's advantage is largest at small $k$ and shrinks as $k$ grows.
- **Anchor Points** (Vivek, Ethayarajh, Yang, Kiela, EACL 2024 Findings). Clustering models' per-item correctness profiles and choosing cluster representatives predicts *per-class* accuracy better than random subsets at matched $k$.
- **Efficient Benchmarking / Benchmark Agreement Testing** (Perlitz et al., NAACL 2024 and follow-ups). HELM-scale compute can be cut by large factors with small rank perturbation; also documents that *benchmark* choice moves rankings more than *subset* choice does.
- **metabench** (Kipnis, Voudouris, Buckley, Irving, Zhang et al., ICLR 2025). Distills six Open LLM Leaderboard benchmarks (~28k items) to a few hundred items; reconstructs original normalized scores with root-mean-square error of roughly 1–2 points and near-unity correlation with the latent ability factor on the fitting population.

**Claimed but unablated.**

- That IRT-selected subsets beat random *for models drawn from a later generation than the fitting set*. Most reported numbers are cross-validated within one model cohort, not across a time split.
- That a single compressed set serves both absolute-score reporting and top-of-leaderboard ranking. Reported as a benchmark number (mean error, Kendall $\tau$) rather than as a decision-error rate on close pairs.
- Adaptive/CAT-style selection for LLMs (Zhuang et al., 2023) reports large item savings; the saving is measured against the full pool, not against a random subset of the same size under the same estimator.

## 4. What Is Known

- **Items are wildly unequal in information.** Rodriguez et al. (ACL 2021) fit IRT to SQuAD/leaderboard responses and found a large fraction of items with near-zero discrimination — they separate no models at all.
- **~100 items per scenario is enough for ~2% absolute error** across ~5–10 scenarios and dozens of open LLMs (tinyBenchmarks, ICML 2024). This is the most-replicated compression number.
- **Random subsets are a strong baseline.** At $k=100$ the standard error of a random-subset mean is at most $0.5/\sqrt{100} = 5$ points in the worst case and typically 2–4 points for $p\approx 0.5$; sophisticated selection buys roughly a factor of 2 on error at that scale, not an order of magnitude.
- **Underpowered comparisons are common.** Card et al. (EMNLP 2020) showed typical NLP test sets lack power to detect 1-point differences — a property compression inherits and worsens.
- **Label noise floors exist.** Vendrow et al. (2025, PlatinumBench) found that manually cleaning popular benchmarks changes frontier-model error rates substantially; on several sets, most residual "errors" of top models were bad labels.
- **Rank instability under compression is item-set dependent.** Perlitz et al. report rank shifts concentrated among near-tied models, which are exactly the pairs leaderboards are used to decide.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $k$ for a target sign-error rate $\rho$ under a $d$-dimensional IRT model with item-parameter distribution $P(a,b)$. No hardness result either: subset selection under an IRT estimator is not known to be submodular, so the $(1-1/e)$ greedy guarantee does not transfer, and no NP-hardness proof for the exact objective has been published.
- **Empirically open (the core gap).** Nobody has run the clean **temporal-split** experiment: fit item parameters on models released before date $T$, evaluate estimator error and sign-error rate on models released after $T$, with a matched random-subset control. Every needed response matrix already exists in public leaderboard archives. This is runnable today for a few hundred GPU-hours or less.
- **Empirically open.** Whether one compressed set can be reused across many future models without inducing overfitting-by-optimization (labs tuning against the small set once it is public).
- **Methodologically blocked.** The loss itself. "Preserves the benchmark" has no agreed definition — absolute error, Kendall $\tau$, top-1 identification, and close-pair decision error give different optima, and no paper reports all four on the same subsets.
- **Methodologically blocked.** Separating item *difficulty* from item *contamination* using response data alone; the two enter the 2PL likelihood identically for the affected models.

## 6. Why It Is Hard

The specific obstruction is **non-stationarity of the estimand plus absent ground truth for the future**. Item selection is fit on a model cohort, then applied to models that differ in kind — new post-training, new tool use, new refusal behavior. The item parameters $(a_i,b_i)$ that made an item informative for 7B-scale base models can invert for reasoning-trained models: an item at $p\approx0.5$ (maximum Fisher information) moves to $p\approx0.99$ (zero information). There is no way to validate the selection for the target population before that population exists, so the honest test is retrospective and nobody has published it with a matched control.

Second obstruction, compounding: **the evaluation does not measure what it names**. Compression error is reported as mean absolute score error, but the decision the benchmark supports is a pairwise comparison of near-tied models. A 2% mean error is negligible for the first and fatal for the second when the gap is 1%.

## 7. Current Research (as of 2026)

- IRT-based distillation lines continue from the tinyBenchmarks and metabench groups (UMich/MIT-IBM; Cambridge/Google DeepMind adjacent). *(frontier — verify current affiliations.)*
- Dynamic and "lifelong" benchmarks — Prabhu et al., *Lifelong Benchmarks: Efficient Model Evaluation in an Era of Rapid Progress* (2024) — use Sort & Search to rank a growing model pool at sublinear item cost. Directly relevant: it makes the temporal-shift problem explicit.
- Benchmark cleaning as a prerequisite to compression (PlatinumBench line, 2025): if 20% of residual errors are label noise, selecting "hard" items selects noise.
- Arena-style preference evaluation replacing static pools in parts of the field, which changes the compression question to *which prompts to route*. *(frontier — verify.)*
- Contamination-aware item scoring — flagging items whose difficulty is model-family-dependent. Early stage; no standard method. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Do IRT-selected subsets beat random subsets on *future* models, or only within the cohort they were fit on?

**Scale.** Public Open LLM Leaderboard v1/v2 archived response matrices: $\geq 300$ models, item pools MMLU (14,042 test items), ARC-Challenge (1,172), HellaSwag (10,042), GSM8K (1,319). No new inference required if archived per-item responses are used; regenerating responses for 50 post-cutoff models on 4 benchmarks is ~26,500 items × 50 models ≈ 1.3M generations, well under 1,000 A100-hours for models ≤70B.

**Design.** Temporal split at $T$ = 2024-06. Fit item parameters and any regression estimator on models released before $T$. Freeze $S$ at $k \in \{50, 100, 250, 500\}$. Evaluate on models released after $T$.

**Control arm.** Uniform random subsets of the same $k$, same estimator family, 1,000 resamples, reporting the full distribution — not just its mean.

**Deciding number.** The **excess close-pair sign-error rate**: $\rho_{\text{IRT}} - \mathbb{E}[\rho_{\text{random}}]$ at $k=100$, restricted to post-$T$ model pairs whose full-pool scores differ by $1$–$3$ points. If the IRT subset does not reduce this by at least $0.05$ absolute (5 percentage points) and lie outside the 95th percentile of the random-subset distribution, principled selection provides no benefit that survives model-population shift, and the field should compress by sampling more items rather than smarter ones.

## 9. Key References

- **[Foundational]** Frederic M. Lord. *Applications of Item Response Theory to Practical Testing Problems.* Lawrence Erlbaum, 1980.
- **[Foundational]** Pedro Rodriguez, Joe Barrow, Alexander Hoyle, John P. Lalor, Robin Jia, Jordan Boyd-Graber. *Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?* ACL 2021.
- **[SOTA]** Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, Mikhail Yurochkin. *tinyBenchmarks: evaluating LLMs with fewer examples.* ICML 2024. — arXiv:2402.14992
- **[SOTA]** Alex Kipnis, Konstantinos Voudouris, Luca M. Schulze Buchholtz, Jan Zhang et al. *metabench — A Sparse Benchmark of Reasoning and Knowledge in Large Language Models.* ICLR 2025. — arXiv:2407.12844
- **[SOTA]** Rajan Vivek, Kawin Ethayarajh, Diyi Yang, Douwe Kiela. *Anchor Points: Benchmarking Models with Much Fewer Examples.* EACL 2024 (Findings). — arXiv:2309.08638
- **[SOTA]** Yotam Perlitz, Elron Bandel, Ariel Gera, Ofir Arviv, Liat Ein-Dor, Eyal Shnarch, Noam Slonim, Michal Shmueli-Scheuer, Leshem Choshen. *Efficient Benchmarking (of Language Models).* NAACL 2024. — arXiv:2308.11696
- **[SOTA]** Ameya Prabhu, Vishaal Udandarao, Philip Torr, Matthias Bethge, Adel Bibi, Samuel Albanie. *Lifelong Benchmarks: Efficient Model Evaluation in an Era of Rapid Progress.* 2024. — arXiv:2402.19472
- **[Related]** Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky. *With Little Power Comes Great Responsibility.* EMNLP 2020.
- **[Related]** Joshua Vendrow, Edward Vendrow, Sara Beery, Aleksander Mądry. *Do Large Language Model Benchmarks Test Reliability?* 2025 (PlatinumBench).
- **[Related]** Yan Zhuang, Qi Liu, Yuting Ning, Weizhe Huang, Zachary Pardos et al. *Efficiently Measuring the Cognitive Ability of LLMs: An Adaptive Testing Perspective.* 2023.

## 10. Worked Example

Take MMLU, $N = 14{,}042$, and a compression to $k = 100$.

**Step 1 — random baseline.** For a model with $s(m) = 0.70$, a random 100-item subset has standard error $\sqrt{0.7 \cdot 0.3 / 100} = 0.0458$. So a random tiny-MMLU gives a $\pm 9$-point 95% interval on a single model's score. Averaged over models, mean absolute error $\approx 0.8 \times 0.0458 \approx 3.7$ points.

**Step 2 — what selection buys.** tinyBenchmarks-style IRT selection reports ~2 points. So the informed method cuts error roughly in half at $k=100$ — real, but the same reduction comes free from raising $k$ to about 350 with random sampling. Selection is worth a $3.5\times$ item saving, not a $100\times$ one.

**Step 3 — the decision that matters.** Suppose two 2026 models score 0.812 and 0.804 on full MMLU (gap 0.008). With a 2-point estimator error at $k=100$ and independent errors, the sign-error rate is
$$\rho = \Phi\!\left(\frac{-0.008}{\sqrt{2}\cdot 0.02}\right) = \Phi(-0.283) \approx 0.39.$$
The compressed benchmark orders this pair correctly 61% of the time — barely better than a coin flip, despite "2% error" sounding tight.

**Step 4 — the obstruction made visible.** Now add the temporal shift. Items chosen in 2024 were maximally informative near $\theta$ corresponding to $p\approx 0.5$, i.e. MMLU accuracy ~0.55. A 2026 model at 0.81 answers most of those items correctly; suppose 60 of the 100 selected items sit above $p = 0.95$ for both models. Effective information collapses to the remaining 40 items, and the standard error rises to $\sqrt{0.2\cdot0.8/40} = 0.063$ — *worse than the random 100-item subset it was supposed to beat*, because random sampling retains the tail of items still discriminating at high ability. Re-running Step 3 with $\sigma = 0.063$ gives $\rho = \Phi(-0.090) \approx 0.46$: essentially uninformative.

Whether this collapse actually occurs in the archived leaderboard data is exactly the unrun measurement in Section 8. The arithmetic shows it is not a corner case — it follows from Fisher information being peaked at $p=0.5$ combined with a rising model population.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*