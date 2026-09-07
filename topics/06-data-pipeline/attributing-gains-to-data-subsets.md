---
id: 06-data-pipeline/attributing-gains-to-data-subsets
title: "Attributing Downstream Gains to Data Subsets"
topic: 06-data-pipeline
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attributing Downstream Gains to Data Subsets

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/attributing-gains-to-data-subsets` · **Status:** open

## 1. Problem Statement

Given a pretraining corpus partitioned into subsets (domains, sources, quality buckets, clusters, or individual documents) and a downstream evaluation, decide **how much of the measured downstream performance each subset caused**.

- **Input:** corpus $D = \{z_1,\dots,z_N\}$ with a partition or weighting scheme; a training recipe $\mathcal{A}$ (architecture, optimizer, token budget, schedule); an evaluation $\mathcal{E}$ (e.g. MMLU 5-shot, GSM8K, held-out perplexity).
- **Output:** attribution scores $\phi_i$ per subset, plus a stated semantics for what $\phi_i$ predicts.
- **Decision predicate:** for a candidate subset $S$, does removing (or upweighting) $S$ change $\mathcal{E}$ by the predicted amount, at the target scale, under a fresh random seed?

Three variants, routinely conflated:

- **Measurement:** define an attribution target that is well posed under training stochasticity. Retraining without $S$ gives a different answer per seed; the estimand must be an expectation, and the seed variance must be smaller than the effect.
- **Method:** estimate that target without $2^{|S|}$ retrains. This is the influence-function / datamodel / Shapley line.
- **Theory:** prove when a cheap estimator (linear surrogate, gradient inner product) is consistent for the retraining counterfactual in a non-convex, non-converged, single-epoch regime.

Solving it means: an attribution procedure whose predicted $\Delta\mathcal{E}$ from removing a held-out subset matches the actually-retrained $\Delta\mathcal{E}$ within seed noise, at $\geq 1$B parameters and $\geq 100$B tokens, for a benchmark that is not held-out perplexity.

## 2. Formal Setting

Training is a stochastic map $\mathcal{A}: (w, \xi) \mapsto \theta$, where $w \in \Delta^{N-1}$ (or $\{0,1\}^N$) is the data weighting and $\xi$ the seed (init, shuffle, dropout, hardware nondeterminism). Define the **expected evaluation functional**

$$F(w) \;=\; \mathbb{E}_{\xi}\big[\, \mathcal{E}(\mathcal{A}(w,\xi)) \,\big].$$

Measured as: train $K$ models at weighting $w$, report $\hat F(w) = \frac{1}{K}\sum_k \mathcal{E}(\theta_k)$ with standard error $\hat\sigma/\sqrt{K}$. Nothing below is meaningful unless $|\Delta F|$ exceeds $2\hat\sigma/\sqrt{K}$.

**Leave-one-subset-out (LOSO) effect** for subset $S$ with indicator $\mathbf{1}_S$:
$$\Delta_S \;=\; F(\mathbf{1}) - F(\mathbf{1} - \mathbf{1}_S).$$

**Datamodel target** (Ilyas et al., 2022): fit $g_\beta(w) = \beta_0 + \beta^\top w$ to $\{(w^{(m)}, \mathcal{E}(\mathcal{A}(w^{(m)},\xi_m)))\}_{m=1}^M$ with $w^{(m)}$ sampled i.i.d. $\alpha$-subsets. Measured as out-of-sample $R^2$ or Spearman $\rho$ of $g_\beta$ against held-out retrains. $\beta_i$ is an *average* marginal effect at sampling fraction $\alpha$, not $\Delta_{\{i\}}$ at $\alpha = 1$.

**Shapley value:** $\phi_i = \sum_{S \subseteq D\setminus i} \frac{|S|!(N-|S|-1)!}{N!}\big(F(S\cup i) - F(S)\big)$ — the unique attribution with efficiency, symmetry, null-player, linearity. Cost $O(2^N)$ exact; Monte Carlo estimates need $\Omega(1/\epsilon^2)$ retrains per digit.

**Influence-function surrogate:** with $H = \nabla^2_\theta \mathcal{L}(\theta^\star)$,
$$\mathcal{I}(z_i, \mathcal{E}) \;=\; -\nabla_\theta \mathcal{E}(\theta^\star)^\top H^{-1} \nabla_\theta \ell(z_i,\theta^\star).$$
Measured with a preconditioner in place of $H^{-1}$: EK-FAC (Grosse et al., 2023) or the TRAK random projection + generalized-Gauss-Newton form (Park et al., 2023).

**Assumptions, and which are violated:**

| Assumption | Status in LLM pretraining |
| --- | --- |
| $\theta^\star$ is a converged optimum, $H \succ 0$ | Violated. Single-epoch, non-converged, $H$ indefinite. |
| Additivity: $F(w)$ approximately linear in $w$ | Violated for large $|S|$ and for duplicated/near-duplicate data; holds locally near $\alpha \approx 0.5$. |
| Seed variance $\ll$ subset effect | Violated for most single documents; only holds for subsets $\gtrsim 1\%$ of tokens. |
| $\mathcal{E}$ is a smooth function of $\theta$ | Violated. Multiple-choice accuracy is a step function of logit gaps. |
| Recipe held fixed across arms | Often violated: removing data changes token count, so arms differ in epochs *and* data. |

## 3. State of the Art

**Established (ablated, independently reproduced):**
- **Datamodels** (Ilyas, Park, Engstrom, Leclerc, Mądry, ICML 2022): linear surrogates over training-subset indicators predict held-out retrained outputs with high correlation on CIFAR-10/FMoW, using $\sim 3\times10^5$ trained models. The linearity result is real; the cost is the point.
- **TRAK** (Park, Georgiev, Ilyas, Leclerc, Mądry, ICML 2023): matches datamodel-quality attribution with orders of magnitude fewer models by linearizing around a small ensemble. Validated on vision and small LMs.
- **EK-FAC influence at LLM scale** (Grosse et al., Anthropic, 2023): influence estimates for models up to 52B parameters. Established as a *retrieval* tool — surfaces plausibly-related training sequences. Not validated as a counterfactual predictor of benchmark deltas.
- **Proxy-model reweighting transfers:** DoReMi (Xie et al., NeurIPS 2023) tunes domain weights with a 280M proxy and improves an 8B model, reaching baseline average few-shot accuracy in ~2.6× fewer steps. RegMix (Liu et al., 2024) fits a regression from small-run mixtures to loss and picks weights for larger runs.
- **Selection beats scale on fixed compute:** DataComp (Gadre et al., NeurIPS 2023) and DCLM (Li et al., 2024) show filtering choices dominate architecture choices at matched compute; DCLM-baseline 7B trained on 2.6T tokens reports MMLU 5-shot ~64%.

**Claimed but unablated:**
- That influence scores computed on a converged checkpoint predict what happens if you *retrain* without those documents at LLM scale. No published LLM-scale retrain validation of this exists.
- That per-document Shapley/influence rankings aggregate correctly to subset-level effects. Aggregation assumes additivity, which is exactly what is in question.

**Benchmark-number-only results:** most "our filter adds $X$ points on MMLU" claims are a single training run per arm, no seed replication, and the token budget changes with the filter. These are not attributions; they are one paired sample.

## 4. What Is Known

- **Linear surrogates work in-distribution at small scale.** Datamodels on CIFAR-10 (ResNet-9, $\sim$300k models) predict held-out subset outcomes with correlations that are high and stable; the estimator degrades as $\alpha \to 1$.
- **Influence functions are fragile in deep nets.** Basu, Pope, Feizi (ICLR 2021) show influence estimates in deep networks correlate poorly with leave-one-out retraining, with sensitivity to depth, width, and damping. Bae et al. (NeurIPS 2022) show the standard estimator actually approximates the *proximal Bregman response function*, not LOO retraining — a different quantity.
- **Seed noise sets the floor.** For LM benchmarks, seed-to-seed swings of ~1 point on MMLU-scale accuracy at the 1B parameter scale are routine; most single-source ablations report deltas of the same order with $K=1$.
- **Pruning helps up to a point.** Sorscher et al. (NeurIPS 2022) show data pruning can beat power-law scaling, but the optimal pruning fraction depends on the initial dataset size — the sign of a subset's contribution flips with total budget.
- **Deduplication is the one robust, reproduced attribution.** Lee et al. (ACL 2022) show removing near-duplicates improves held-out perplexity and reduces memorized emission, replicated across corpora.
- **Order matters.** Georgiev et al. (2023) show the *trajectory*, not just the data multiset, drives which examples matter — undermining any weighting-only estimand.

## 5. What Is Not Known

- **Methodologically blocked:** what $\phi_i$ *means* when the recipe is compute-matched. Removing 5% of tokens either shortens training or repeats data; the counterfactual is underspecified, and the two choices give different signs for the same subset. No community standard exists.
- **Methodologically blocked:** attribution to a *single document* at pretraining scale. The expected effect is far below seed noise, so the estimand exists only in expectation over an ensemble nobody can afford.
- **Empirically open:** does TRAK/EK-FAC influence predict retrained benchmark deltas at $\geq$1B params? Runnable — 20–40 retrains at 1B×100B tokens — but unrun publicly.
- **Empirically open:** does the proxy-model transfer of mixture weights (DoReMi/RegMix) hold across a $10^3$ compute gap, or only the $10^{1.5}$ gaps tested?
- **Theoretically open:** conditions under which the linear datamodel is consistent for non-convex, single-epoch SGD. No proof either way; existing guarantees assume strong convexity and convergence.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth at the scale that matters.** The only defensible ground truth is retraining. At 7B×2T tokens one run costs $\sim 10^{23}$ FLOPs; a $K=5$, 20-subset LOSO grid is $10^{25}$ FLOPs. Ground truth is therefore only ever collected at scales where the answer may not transfer.
2. **Confounded measurement.** Removing a subset simultaneously changes (a) the data distribution, (b) the token count, (c) the epoch count over the remainder, (d) optimal learning-rate schedule length. Standard practice varies all four and attributes the result to (a).
3. **Non-identifiability under redundancy.** If subsets $A$ and $B$ carry the same signal, $\Delta_A = \Delta_B \approx 0$ while $\Delta_{A\cup B} \gg 0$. Any additive $\phi$ assigns near-zero to both. Web corpora are massively redundant, so this is the typical case, not a corner case.

## 7. Current Research (as of 2026)

- **Mądry lab (MIT):** datamodels → TRAK → DsDm (Engstrom, Feldmann, Mądry, 2024), which selects pretraining data by predicted downstream-task datamodel score rather than by heuristic quality, and reports beating heuristic selection at matched compute in the 125M–1.3B range.
- **Anthropic interpretability:** influence at scale as an interpretability instrument, not a curation instrument.
- **Princeton/UW selection for finetuning:** LESS (Xia, Malladi, Gururangan, Arora, Chen, ICML 2024) — gradient-similarity selection of ~5% of an instruction pool matching or beating full-data finetuning. Finetuning-scale evidence; extrapolation to pretraining is an assumption.
- **Mixture-law fitting:** data mixing laws / RegMix / DoGE — fit $F(w)$ parametrically from many small runs, extrapolate. *(frontier — verify)* whether fitted laws hold beyond the compute range they were fit in.
- **In-run attribution:** In-Run Data Shapley (Wang et al., NeurIPS 2024) computes Shapley-like scores during a single training run, removing the retrain requirement at the cost of changing the estimand to a trajectory-local one.

## 8. Concrete Next Experiment

**Question:** does a cheap attribution estimator predict the *sign and magnitude* of a retrained benchmark delta at 1B scale?

- **Scale:** 1.3B-parameter decoder, 100B tokens, corpus partitioned into 20 source subsets each 2–8% of tokens.
- **Arms:** for each of 6 pre-selected subsets (3 predicted high-value, 3 predicted near-zero by TRAK-style scores computed on a 160M proxy), retrain with that subset removed and the remainder **upsampled to hold total tokens at 100B** — this is the token-matched control that removes confounder (b)/(c). Run $K=3$ seeds per arm. Plus a full-data arm at $K=5$. Total: $6\times3 + 5 = 23$ runs $\approx 2\times10^{22}$ FLOPs.
- **Second control arm:** 6 *random* subsets of matched token mass, $K=3$ each. Without this arm, any correlation may be explained by subset size alone.
- **Deciding number:** Spearman $\rho$ between predicted $\hat\Delta_S$ and measured $\Delta_S$ across the 12 subsets, on an aggregate of MMLU + ARC-c + HellaSwag. $\rho \geq 0.7$ with the random-subset arm's $\rho$ near zero means proxy attribution transfers. $\rho \leq 0.3$ means every published influence-based curation claim rests on an unvalidated step.

## 9. Key References

- **[Foundational]** Pang Wei Koh, Percy Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017. — arXiv:1703.04730
- **[Foundational]** Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Mądry. *Datamodels: Predicting Predictions from Training Data.* ICML 2022. — arXiv:2202.00622
- **[Foundational]** Amirata Ghorbani, James Zou. *Data Shapley: Equitable Valuation of Data for Machine Learning.* ICML 2019. — arXiv:1904.02868
- **[SOTA]** Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Mądry. *TRAK: Attributing Model Behavior at Scale.* ICML 2023. — arXiv:2303.14186
- **[SOTA]** Logan Engstrom, Axel Feldmann, Aleksander Mądry. *DsDm: Model-Aware Dataset Selection with Datamodels.* ICML 2024. — arXiv:2401.12926
- **[SOTA]** Roger Grosse, Juhan Bae, Cem Anil, et al. *Studying Large Language Model Generalization with Influence Functions.* Anthropic, 2023. — arXiv:2308.03296
- **[SOTA]** Sang Michael Xie, Hieu Pham, Xuanyi Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. — arXiv:2305.10429
- **[SOTA]** Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* ICML 2024. — arXiv:2402.04333
- **[Critique]** Samyadeep Basu, Philip Pope, Soheil Feizi. *Influence Functions in Deep Learning Are Fragile.* ICLR 2021. — arXiv:2006.14651
- **[Critique]** Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, Roger Grosse. *If Influence Functions are the Answer, Then What is the Question?* NeurIPS 2022. — arXiv:2209.05364
- **[Empirical]** Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, et al. *DataComp: In Search of the Next Generation of Multimodal Datasets.* NeurIPS 2023 (Datasets & Benchmarks). — arXiv:2304.14108
- **[Empirical]** Jeffrey Li, Alex Fang, Georgios Smyrnis, et al. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models.* 2024. — arXiv:2406.11794
- **[Empirical]** Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al. *Deduplicating Training Data Makes Language Models Better.* ACL 2022. — arXiv:2107.06499
- **[Empirical]** Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari Morcos. *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS 2022. — arXiv:2206.14486
- **[Survey]** Alon Albalak, Yanai Elazar, Sang Michael Xie, et al. *A Survey on Data Selection for Language Models.* TMLR 2024. — arXiv:2402.16827

## 10. Worked Example

Two 1B-parameter runs on 50B tokens. Corpus contains a 1.5B-token math subset $A$ (scraped forum solutions) and a 1.2B-token math subset $B$ (textbook exercises), heavily overlapping in content.

| Arm | GSM8K 8-shot | measured $\Delta$ |
| --- | --- | --- |
| Full corpus ($K=3$) | 12.4% ± 0.9 | — |
| Remove $A$ ($K=3$) | 12.1% ± 0.8 | $-0.3$ |
| Remove $B$ ($K=3$) | 12.6% ± 1.1 | $+0.2$ |
| Remove $A \cup B$ ($K=3$) | 6.9% ± 0.7 | $-5.5$ |

Additive attribution predicts $\Delta_{A\cup B} \approx \Delta_A + \Delta_B = -0.1$. The measured value is $-5.5$ — off by a factor of 55, and the wrong sign for $B$.

The estimator is not merely imprecise; it is **non-identifiable**. Any additive $\phi$ splitting $-5.5$ between $A$ and $B$ fits the joint observation, and the single-removal observations force both marginals to zero. A per-document influence method that ranks and drops "low-influence" math data will drop both, because each is individually redundant.

Note also the noise floor: $\Delta_A = -0.3$ against a seed standard error of $0.9/\sqrt{3} = 0.52$. The single-subset rows carry no information at $K=3$; distinguishing $\Delta_A = 0$ from $\Delta_A = -0.5$ at 95% confidence needs $K \approx 50$ per arm. The obstruction is visible in both directions at once — the estimand is confounded by redundancy, and the measurement that would expose the confounding is priced out.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*