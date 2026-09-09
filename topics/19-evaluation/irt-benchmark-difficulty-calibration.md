---
id: 19-evaluation/irt-benchmark-difficulty-calibration
title: "Benchmark Difficulty Calibration via Item Response Theory at Scale"
topic: 19-evaluation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Benchmark Difficulty Calibration via Item Response Theory at Scale

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/irt-benchmark-difficulty-calibration` · **Status:** empirically-open

## 1. Problem Statement

A benchmark score is a mean over items. It treats a trivially guessable question and a question only two models on Earth answer as equal evidence. Item response theory (IRT) replaces the mean with a latent-trait model: each item $i$ gets a difficulty $b_i$ and discrimination $a_i$, each model $j$ gets an ability $\theta_j$, and the score is a posterior over $\theta$ rather than a sample average.

Three variants, routinely conflated:

- **Measurement variant.** Given a response matrix of models × items, estimate $\{a_i, b_i\}$ and $\{\theta_j\}$. Solved as an estimation exercise; the open question is whether the estimates mean anything outside the fitting population.
- **Method variant.** Use the calibrated item parameters to do something useful: shrink a 15,000-item benchmark to 100 items with bounded error, select items adaptively per model, or flag contaminated/miskeyed items. Partially demonstrated.
- **Theory variant.** State conditions under which $b_i$ is a property of the *item* rather than of the model cohort used to calibrate it. This is the actual open problem. Human IRT inherits parameter invariance from a stable, exchangeable examinee population; LLM cohorts are neither stable nor exchangeable, and no analogue of invariance has been established.

Solving it means: a difficulty scale estimated on cohort $C_1$ predicts item-level responses of a disjoint, later cohort $C_2$ within stated error, with a diagnostic that fires when it will not.

## 2. Formal Setting

Let $\mathcal{I} = \{1,\dots,M\}$ be items and $\mathcal{J} = \{1,\dots,N\}$ be models. The observed object is the binary response matrix $U \in \{0,1\}^{N \times M}$, $u_{ji} = 1$ iff model $j$'s output on item $i$ is scored correct by the benchmark's own grader (exact match, MCQ letter, unit test, or LLM judge — the choice is part of the measurement and changes $U$).

**2PL model.** With ability $\theta_j \in \mathbb{R}$:

$$P(u_{ji}=1 \mid \theta_j, a_i, b_i) = \sigma\!\left(a_i(\theta_j - b_i)\right), \qquad \sigma(z)=\frac{1}{1+e^{-z}}.$$

**3PL** adds a lower asymptote $c_i$ for guessing: $P = c_i + (1-c_i)\sigma(a_i(\theta_j-b_i))$. For 4-option MCQ, $c_i \approx 0.25$ is the prior, but a model that eliminates two distractors has $c_i \approx 0.5$ — $c_i$ is not a property of the item alone.

**Fisher information** of item $i$ at ability $\theta$, under 2PL:

$$I_i(\theta) = a_i^2 \,\sigma(a_i(\theta-b_i))\,\bigl(1-\sigma(a_i(\theta-b_i))\bigr),$$

maximised at $\theta = b_i$ with peak $a_i^2/4$. Test information $I(\theta)=\sum_i I_i(\theta)$ gives the standard error $\mathrm{SE}(\hat\theta) = I(\theta)^{-1/2}$. This is what makes IRT worth the trouble: it says *which* items are worth running for a model of a given strength.

Estimation in practice: marginal maximum likelihood with $\theta_j \sim \mathcal{N}(0,1)$ integrated out (EM), or variational/MCMC Bayesian fits when $N$ is small relative to the number of item parameters ($2M$ or $3M$).

**Assumptions, and how they break:**

| Assumption | Status in LLM evaluation |
|---|---|
| Unidimensionality — one $\theta$ per model | Violated. Code, multilingual, and math items load on separate factors; a single $\theta$ absorbs training-mix differences. |
| Local independence — $u_{ji} \perp u_{ji'} \mid \theta_j$ | Violated. Items sharing a passage, template, or MMLU subject are correlated at fixed $\theta$. |
| Monotonicity in $\theta$ | Violated for inverse-scaling items (sycophancy, memorised-falsehood traps), where $P$ decreases in $\theta$; a 2PL fit returns $a_i<0$ and the item is usually discarded rather than explained. |
| Population exchangeability | Violated hardest. Models share pretraining corpora and distillation ancestry; contamination makes $u_{ji}$ depend on model $j$'s data, not its ability. |
| Scale identifiability | $(\theta,a,b) \to (\alpha\theta+\beta,\ a/\alpha,\ \alpha b + \beta)$ leaves the likelihood unchanged. Fixed by convention ($\theta \sim \mathcal{N}(0,1)$), so $b_i$ is only ever meaningful *relative to the calibrating cohort*. |

## 3. State of the Art

**Established.**
- IRT applied to NLP test sets recovers interpretable difficulty and detects items no model distinguishes on — Lalor, Wu & Yu (EMNLP 2016; EMNLP 2019 with artificial crowds); Martínez-Plumed et al. (*Artificial Intelligence*, 2019) at the instance level for classifiers.
- IRT-weighted leaderboards reorder systems relative to accuracy ranking — Rodriguez et al., ACL 2021, on SQuAD-family leaderboards.
- Compression works: **tinyBenchmarks** (Polo et al., ICML 2024) uses IRT-based performance estimation (p-IRT) to reproduce full-benchmark scores from ~100 items; **metabench** (Kipnis, Voudouris, Schulz et al., 2024/ICLR 2025) distils six benchmarks to a small item bank; **Anchor Points** (Vivek, Ethayarajh, Yang & Kiela, EACL 2024) reaches similar compression with a clustering method that is *not* IRT — an important control arm that is often omitted.
- Adaptive item selection using Fisher information reduces items needed per model — **Fluid Benchmarking** (Hofmann et al., AI2, 2025).

**Claimed but unablated.**
- That fitted $b_i$ is a stable, model-independent property of an item. Papers report within-cohort fit, not out-of-cohort transfer to a later model generation.
- That IRT beats simple baselines *because of the latent-trait structure*. The reported gains over stratified random subsampling and over clustering-based selection are often within overlapping error bars, and the ablation isolating the IRT component is usually missing.
- That negative-discrimination items are "bad items". They may be the informative ones.

**Benchmark-number-only results.** Most compression claims exist as a single table of estimation errors on the model set used for fitting. That is a within-sample number.

## 4. What Is Known

- **Compression magnitude.** tinyBenchmarks reports mean absolute estimation error of roughly 2 percentage points on MMLU using ~100 items in place of ~14,000, measured over the ~100-model Open LLM Leaderboard cohort of early 2024.
- **Item bank reduction.** metabench reduces ~28,000 items across six benchmarks (ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande) to under 3% of the original, reconstructing original scores with a few-percent RMSE and recovering a dominant latent factor explaining the large majority of score variance, on a cohort of ~5,000 models scraped from the leaderboard.
- **Redundancy.** Rodriguez et al. (ACL 2021) find a large fraction of items carry near-zero discrimination — for most benchmarks the median item's Fisher information at the cohort's modal ability is small enough that removing it changes nothing.
- **Guessing floor dominates easy items.** For 4-option MCQ, items with $b_i$ well below the cohort mean have information bounded near zero once $c_i \approx 0.25$ is fitted; on saturated benchmarks this is most of the bank.
- **Classical theory.** Rasch (1960) separability: for the 1PL model, the conditional likelihood given raw score is free of $\theta$, so item parameters are estimable without the ability distribution. This is the only strong invariance result available, and it requires $a_i$ equal across items — false for real benchmarks.

## 5. What Is Not Known

- **Empirically open.** Does an item difficulty scale calibrated on cohort $C_1$ (say, models released before 2024) predict item-level responses of a disjoint later cohort $C_2$? The response matrices exist; the split-cohort transfer experiment has not been run at scale with a pre-registered error target. This is the central gap.
- **Empirically open.** Do IRT-selected subsets beat cluster-selected and stratified-random subsets of the same size when both are evaluated on *held-out* models? The head-to-head with matched budget and out-of-cohort test set is missing.
- **Methodologically blocked.** There is no accepted operational definition of "item difficulty" independent of a model population. Human IRT anchors on a sampled population; there is no sampling frame for "the population of language models," so $b_i$ has no population-level referent.
- **Methodologically blocked.** Contamination is not separable from ability within the model. A memorised item and an easy item produce the same $u_{ji}$ pattern unless an external contamination signal is added.
- **Theoretically open.** No parameter-recovery guarantee under correlated, non-exchangeable respondents. Standard MML consistency assumes i.i.d. examinees; models drawn from a distillation tree are not.

## 6. Why It Is Hard

The obstruction is **non-identifiability against a moving population**, compounded by **confounded measurement**.

$b_i$ is defined only up to the location-scale convention imposed on $\theta$. Fixing $\theta \sim \mathcal{N}(0,1)$ makes $b_i$ a statement about the *calibrating cohort's* distribution. When the cohort shifts — as it does every six months — every $b_i$ shifts with it, and there is no external anchor (no human norming sample, no criterion-referenced cut score) to link the scales. Human testing solves this with anchor items and equating designs that assume a stable construct; LLM capability is not a stable construct across generations, so equating has no fixed point.

Second: differential item functioning (DIF) is the norm, not the exception. If a model has seen an item in pretraining, $P(u_{ji}=1)$ jumps independent of $\theta_j$ — exactly the DIF signature — and the fit absorbs it into $a_i$ (driving discrimination down) or into $\theta_j$ (inflating ability). Without ground truth on contamination, the two are unidentifiable from $U$ alone.

Compute is *not* the obstruction. The response matrices are already public.

## 7. Current Research (as of 2026)

- **Adaptive evaluation.** AI2's Fluid Benchmarking line: IRT-parameterised item selection updated as $\hat\theta$ moves during evaluation, reported to improve variance and validity per item spent, including across training checkpoints.
- **Efficient-evaluation methods.** Polo et al. (tinyBenchmarks) and follow-ons on estimator variance and confidence intervals for compressed benchmarks.
- **Psychometrics-for-AI.** Kipnis/Schulz (Tübingen/MPI) on metabench and on whether the recovered latent factor is a genuine general-ability dimension; Hernández-Orallo and Martínez-Plumed (Valencia) on ability-oriented, item-level evaluation and capability profiles.
- **Contamination-aware calibration.** Joint models with a per-(model, item) memorisation indicator alongside $\theta$ *(frontier — verify)*.
- **Multidimensional IRT for capability profiles.** Replacing scalar $\theta$ with a low-rank factor structure over skills *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Are item difficulties transferable across model generations?

**Scale.** Assemble $U$ over $M \approx 15{,}000$ items (MMLU, ARC-Challenge, GSM8K, HellaSwag, plus a held-out post-2025 set) and $N \approx 400$ open-weight models with public per-item responses. Split the cohort *temporally*, not randomly: $C_1$ = models released before 2024-07 ($\approx 250$), $C_2$ = models released after ($\approx 150$). Cost: no new inference if leaderboard per-item logs are reused; otherwise ~$15$k GPU-hours.

**Procedure.** Fit 2PL on $(C_1, \mathcal{I})$ by MML-EM. Freeze $\{\hat a_i, \hat b_i\}$. For each model in $C_2$, estimate $\hat\theta_j$ from a Fisher-information-selected 100-item subset. Predict held-out responses on the remaining items.

**Control arms (all at the same 100-item budget):**
1. Stratified random subsample.
2. Anchor Points clustering (Vivek et al., 2024) — non-IRT selection.
3. Oracle: 2PL refit on $C_2$ itself (upper bound).

**Deciding number.** Out-of-cohort item-level log-loss on held-out responses of $C_2$, and the gap between the frozen-$b$ arm and the oracle refit arm. If the frozen arm's log-loss is within $0.02$ nats of the oracle and the Spearman correlation $\rho(\hat b^{C_1}, \hat b^{C_2}) \geq 0.90$, difficulty transfers and the method variant is settled. If $\rho \leq 0.7$ or the frozen arm loses to stratified random, IRT difficulty is a cohort statistic, not an item property — and every published compression number is within-sample.

A secondary number worth reporting: the fraction of items whose $\hat b$ moves by more than $1$ logit between cohorts, cross-tabulated against an n-gram contamination flag. That tabulation isolates whether drift is contamination or genuine capability change.

## 9. Key References

- **[Foundational]** Georg Rasch. *Probabilistic Models for Some Intelligence and Attainment Tests.* Danmarks Pædagogiske Institut, 1960.
- **[Foundational]** Allan Birnbaum. "Some latent trait models and their use in inferring an examinee's ability." In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores*, Addison-Wesley, 1968.
- **[Foundational]** John P. Lalor, Hao Wu, Hong Yu. *Building an Evaluation Scale using Item Response Theory.* EMNLP, 2016.
- **[Foundational]** John P. Lalor, Hao Wu, Hong Yu. *Learning Latent Parameters without Human Response Patterns: Item Response Theory with Artificial Crowds.* EMNLP, 2019.
- **[Foundational]** Fernando Martínez-Plumed, Ricardo B. C. Prudêncio, Adolfo Martínez-Usó, José Hernández-Orallo. *Item response theory in AI: Analysing machine learning classifiers at the instance level.* Artificial Intelligence, 2019.
- **[SOTA]** Pedro Rodriguez, Joe Barrow, Alexander Hoyle, John P. Lalor, Robin Jia, Jordan Boyd-Graber. *Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?* ACL, 2021.
- **[SOTA]** Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, Mikhail Yurochkin. *tinyBenchmarks: evaluating LLMs with fewer examples.* ICML, 2024.
- **[SOTA]** Alex Kipnis, Konstantinos Voudouris, Luca M. Schulze Buschoff, Eric Schulz. *metabench — A Sparse Benchmark to Measure General Ability in Large Language Models.* ICLR, 2025.
- **[SOTA]** Rajan Vivek, Kawin Ethayarajh, Diyi Yang, Douwe Kiela. *Anchor Points: Benchmarking Models with Much Fewer Examples.* EACL, 2024.
- **[SOTA]** Valentin Hofmann et al. *Fluid Language Model Benchmarking.* Allen Institute for AI, 2025. (Preprint; identifier omitted.)
- **[Survey]** Clara Vania, Phu Mon Htut, William Huang, Dhara Mungra, Richard Yuanzhe Pang, Jason Phang, Haokun Liu, Kyunghyun Cho, Samuel R. Bowman. *Comparing Test Sets with Item Response Theory.* ACL, 2021.
- **[Survey]** Percy Liang et al. *Holistic Evaluation of Language Models (HELM).* TMLR, 2023.

## 10. Worked Example

Take one MMLU item, `high_school_world_history` Q#412. Fit a 2PL on a 2023 cohort of 120 models: $\hat a = 1.6$, $\hat b = 0.4$. The cohort mean ability is $0$, so at $\theta=0.4$ the item's Fisher information is $a^2/4 = 0.64$ — among the most informative in the bank. Under a 100-item budget selected by Fisher information at $\theta \approx 0$, it is selected.

Now refit on a 2025 cohort of 120 models. The item's raw accuracy has gone from $0.51$ to $0.97$. Two explanations produce identical $U$ columns:

1. **Ability rose.** Mean $\theta$ moved to $+2.0$. Then with $b=0.4$, $a=1.6$: $P = \sigma(1.6 \times 1.6) = 0.93$ — close to observed.
2. **The item leaked.** MMLU is in the pretraining mix of every 2025 model; $P \to 0.97$ regardless of $\theta$.

Because the fit fixes $\theta \sim \mathcal{N}(0,1)$ per cohort, refitting reports $\hat b = -1.1$ and $\hat a = 0.3$ on the 2025 cohort: the item has moved 1.5 logits easier and lost discrimination. Its information at the new cohort mean is $0.3^2/4 = 0.0225$ — a factor of 28 below its 2023 value. It drops out of the selected subset entirely.

The obstruction is visible in that the drift is *the same number* under both explanations. Under (1) the item is still a good item and the scale simply shifted; under (2) the item is dead and should be retired. The response matrix cannot tell them apart, because location shift in $\theta$ and uniform leakage are the same transformation of the likelihood. Any compression pipeline that freezes $\hat b$ from an old cohort therefore silently mixes retired items and rescaled items, and its reported estimation error — the ~2-point figure quoted in Section 4 — is measured on the cohort that produced the parameters. That number does not bound the error on the next generation of models, and no published experiment yet says what does.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*