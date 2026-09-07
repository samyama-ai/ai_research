---
id: 19-evaluation/adaptive-item-selection-model-ranking
title: "Sample-Efficient Model Ranking via Adaptive Item Selection"
topic: 19-evaluation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sample-Efficient Model Ranking via Adaptive Item Selection

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/adaptive-item-selection-model-ranking` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A pool of benchmark items $\mathcal{I}$, $|\mathcal{I}| = N$ (e.g. MMLU's 14,042 test questions), a set of models $\mathcal{M}$, $|\mathcal{M}| = M$, and a query budget $B \ll MN$ evaluations, where one evaluation reveals the graded response $Y_{mi}$ of model $m$ on item $i$.

**Output.** A ranking $\hat\pi$ over $\mathcal{M}$ (or a top-$k$ set), produced by a policy that chooses each next pair $(m,i)$ adaptively from all responses seen so far.

**Decision predicate.** For a target ranking loss $L$ and tolerance $\varepsilon$: does there exist a policy achieving $\mathbb{E}[L(\hat\pi, \pi^\star)] \le \varepsilon$ with $B$ asymptotically smaller than uniform random subsampling, where $\pi^\star$ is the ranking induced by full evaluation on $\mathcal{I}$?

Three variants, routinely conflated:

- **Measurement.** Is "rank on the full pool" the estimand we want, or is it a proxy for a latent ability? Adaptive selection changes which of the two you are estimating.
- **Method.** Build a selection policy that empirically beats uniform subsampling at fixed $B$, with valid uncertainty quantification under the adaptive sampling.
- **Theory.** Characterise the minimax query complexity of ranking $M$ models to accuracy $\varepsilon$ under an item-response model, and prove whether adaptivity helps beyond constants.

## 2. Formal Setting

Response matrix $Y \in \{0,1\}^{M \times N}$; $Y_{mi}$ is the graded correctness of model $m$ on item $i$, measured by the benchmark's own scorer (exact match, MCQ letter match, or an LLM judge — the judge's noise is part of $Y$, not separate from it).

True score, as actually computed by leaderboards:
$$\theta_m \;=\; \frac{1}{N}\sum_{i=1}^{N} Y_{mi}, \qquad \pi^\star = \text{argsort}_{m}(-\theta_m).$$

A policy $\rho$ observes $\mathcal{H}_t = \{(m_s, i_s, Y_{m_s i_s})\}_{s<t}$ and picks $(m_t,i_t)$. Estimator $\hat\theta_m$; ranking loss is either Kendall distance $K(\hat\pi,\pi^\star) = \sum_{m<m'} \mathbb{1}[\text{order flipped}]$, normalised to $\tau \in [-1,1]$, or top-$k$ recall.

**Item-response (2PL) model.** Latent ability $\theta_m \in \mathbb{R}$, item discrimination $a_i > 0$, difficulty $b_i$:
$$\Pr(Y_{mi}=1) = \sigma\!\big(a_i(\theta_m - b_i)\big),\qquad \sigma(z) = (1+e^{-z})^{-1}.$$
Fisher information of item $i$ at ability $\theta$:
$$I_i(\theta) = a_i^2\,\sigma\!\big(a_i(\theta-b_i)\big)\big(1-\sigma(a_i(\theta-b_i))\big),$$
maximised at $b_i = \theta$. Classical computerized adaptive testing (CAT) picks $i_t = \arg\max_i I_i(\hat\theta_{m_t})$.

**Pairwise separation.** For models $m,m'$ define the gap $\Delta_{mm'} = |\theta_m - \theta_{m'}|$ and the disagreement rate $d_{mm'} = \frac{1}{N}\sum_i \mathbb{1}[Y_{mi}\ne Y_{m'i}]$. The paired per-item difference $D_i = Y_{mi}-Y_{m'i}$ has $\mathbb{E}[D_i]=\Delta_{mm'}$ and $\mathrm{Var}(D_i) = d_{mm'} - \Delta_{mm'}^2$ — the quantity that actually governs cost.

**Best-arm lower bound.** Under independent Bernoulli arms, identifying the top model with confidence $1-\delta$ requires $\Omega\big(\sum_{m\ne m^\star} \Delta_{m^\star m}^{-2}\log(1/\delta)\big)$ samples (Mannor & Tsitsiklis, JMLR 2004).

**Assumptions, and where they break.**

| Assumption | Status in practice |
|---|---|
| Unidimensional latent ability | Violated: multi-domain benchmarks load on ≥2 factors |
| Local independence of items | Violated: shared-passage items, near-duplicates, template families |
| Item parameters $(a_i,b_i)$ transfer to unseen models | Violated under contamination and post-training on benchmark-like data |
| Model is a fixed conditional distribution | Violated for APIs: silent version drift, temperature, decoding |
| Binary, noiseless grading | Violated for generative tasks and LLM judges |
| Full-pool score is the estimand of interest | Contested — it is itself a sample from a task distribution |

## 3. State of the Art

**Established (independently reproduced or open-code-and-data).**
- **IRT for NLP evaluation.** Lalor, Wu & Yu (EMNLP 2016) fit IRT scales to NLI data; Rodriguez et al. (ACL 2021) show IRT-based item weighting changes SQuAD/leaderboard orderings and identifies uninformative items. Established: benchmark items differ enormously in discrimination.
- **Fixed anchor subsets work.** `tinyBenchmarks` (Polo et al., ICML 2024) reports ~100-item curated subsets estimating full-benchmark accuracy with roughly 2% average absolute error across Open LLM Leaderboard, HELM and AlpacaEval 2.0. Code and item sets are public and have been re-used by third parties.
- **`metabench`** (Kipnis et al., ICLR 2025) distils six benchmarks (~28.6k items) to under 3% of items and reconstructs original scores with low single-digit-percent RMSE, using IRT fit on ~5,000 models.
- **Efficient Benchmarking / Flash-HELM** (Perlitz et al., NAACL 2024) formalises benchmark "reliability" and reports HELM-rank preservation at orders-of-magnitude less compute by allocating more samples to top-ranked models.
- **Adaptive testing lower bounds for ranking from comparisons.** Jamieson & Nowak (NeurIPS 2011), Heckel et al. (AISTATS 2019) give sample complexities for active ranking; the $\Delta^{-2}$ scaling is a theorem, not a heuristic.

**Claimed but unablated.**
- That *adaptivity per se* (re-selecting items conditioned on partial responses of the model under test) beats a *well-chosen static* subset. Most reported gains come from static anchor selection plus IRT reweighting; the adaptive-vs-static-anchor ablation at matched $B$ is largely missing.
- Chatbot Arena–style adaptive pair sampling (Chiang et al., ICML 2024) reduces votes needed for stable rankings; the reported reduction is a benchmark number on their own vote log, not a controlled comparison against uniform pairing.

**Benchmark-number-only.** Anchor Points (Vivek et al., ICLR 2024) reports ranking correlation from small anchor sets; the numbers are per-benchmark and depend on the model pool used to fit the anchors.

## 4. What Is Known

- Item informativeness is extremely skewed. On several Open LLM Leaderboard tasks, a large fraction of items are answered identically by nearly all models in a pool of hundreds — they carry no ranking information at any budget.
- 100 well-chosen items ≈ full benchmark for *score estimation* to ~2 points (tinyBenchmarks, ICML 2024, measured over hundreds of open models on MMLU/ARC/HellaSwag/GSM8K/TruthfulQA/WinoGrande).
- Under 3% of items suffice to reconstruct six benchmark scores at low single-digit RMSE when IRT is fit on thousands of models (metabench, ICLR 2025).
- Compute-allocation adaptivity across *models* (evaluate weak models less) gives large savings with near-identical top-of-leaderboard rank order (Perlitz et al., NAACL 2024, HELM scale: dozens of models × dozens of scenarios).
- Paired evaluation dominates unpaired when models are correlated: variance $d_{mm'}-\Delta^2$ instead of $\approx 2\bar\theta(1-\bar\theta)$. This is arithmetic, not an empirical claim.
- Adaptive data collection invalidates naive i.i.d. confidence intervals; anytime-valid confidence sequences (Howard, Ramdas, McAuliffe, Sekhon, *Annals of Statistics* 2021) restore coverage at a $\sqrt{\log\log}$ price.

## 5. What Is Not Known

- **Theoretically open.** The minimax query complexity of *full* ranking of $M$ models over a shared item pool with correlated responses. Existing bounds treat models as independent arms; the shared-item structure (which makes pairing work) has no matching lower bound. Whether adaptivity improves the rate — not just the constant — over the best static subset is unproven either way.
- **Empirically open.** Whether CAT-style within-model adaptive item selection beats a fixed IRT-anchor subset at matched $B$ on modern LLMs. The experiment is cheap (public response matrices exist) and has not been run as a clean head-to-head with held-out models.
- **Empirically open.** Robustness of learned item parameters to distribution shift in the model pool: anchors fit on 2024 open models, applied to 2026 reasoning models with long chain-of-thought.
- **Methodologically blocked.** What "the right ranking" is. $\pi^\star$ on the full pool is itself noisy and contaminated; there is no ground-truth capability ordering to validate against, so every result is measured against a proxy the method is also trying to shortcut.

## 6. Why It Is Hard

The specific obstruction is **estimand shift under selection, compounded by absent ground truth**. Selecting the items that best separate models changes what is being measured: a maximally discriminating subset is, by construction, enriched for items on which models disagree, which is not a random sample of the task distribution. The resulting number is a good *ranking statistic* and a bad *accuracy estimate*, and the two are reported interchangeably. Because there is no external capability ground truth, the method is validated against the full-pool ranking — but the full-pool ranking is what the community already distrusts (contamination, format sensitivity, near-duplicate items). Success therefore means faithfully reproducing a measurement whose validity is the actual open question. A secondary obstruction is statistical: adaptive selection breaks the i.i.d. assumption behind reported error bars, so a method can appear to win purely by producing anticonservative intervals.

## 7. Current Research (as of 2026)

- IRT-based benchmark distillation: Polo, Yamada et al. (tinyBenchmarks line); Kipnis, Voudouris, Buckley, Burden, Hernández-Orallo (metabench).
- Efficiency and reliability of large benchmark suites: Perlitz, Bandel, Shmueli-Scheuer et al. (IBM Research, Efficient Benchmarking / Flash-HELM).
- Prediction-powered inference for cheap-judge/expensive-human mixtures: Angelopoulos, Bates, Zrnic, Candès (PPI, *Science* 2023); Chatzi et al., *Prediction-Powered Ranking of Large Language Models* (NeurIPS 2024) — ranks with rank-set guarantees under limited human labels.
- Live arena sampling and rank uncertainty: LMSYS/LMArena (Chiang et al., ICML 2024).
- *(frontier — verify)* Adaptive selection for agentic and long-horizon benchmarks, where a single item costs minutes of tool-calling and per-item cost is heterogeneous — turning this into a knapsack-constrained best-arm problem rather than a fixed-budget one.

## 8. Concrete Next Experiment

**Scale.** Use a fully observed public response matrix: $\ge 300$ open models $\times$ $\ge 20{,}000$ items pooled from MMLU, ARC-Challenge, HellaSwag, GSM8K, WinoGrande (the Open LLM Leaderboard raw responses are downloadable). Split models 80/20 into *fit* and *held-out*. Fit IRT parameters only on the fit pool.

**Arms, all at matched budget $B \in \{200,500,1000,2000\}$ item-evaluations per held-out model:**
1. **Control:** uniform random items, unweighted mean.
2. Static IRT-anchor subset (tinyBenchmarks-style), IRT-weighted estimator.
3. **Treatment:** CAT — greedy maximum-Fisher-information item selection against the running $\hat\theta_m$.
4. Cross-model adaptive allocation (spend more on close pairs), items uniform.

**Deciding number.** Kendall $\tau$ between $\hat\pi$ and full-pool $\pi^\star$ on the 60 held-out models at $B=500$, averaged over 1,000 seeds. Decision rule: CAT is worth its complexity iff $\tau_{\text{CAT}} - \tau_{\text{static-anchor}} \ge 0.05$ with non-overlapping 95% bootstrap intervals. **Mandatory second number (validity gate):** empirical coverage of nominal 95% intervals on $\hat\theta_m$ under each arm; any arm below 90% coverage is disqualified regardless of $\tau$.

## 9. Key References

- **[Foundational]** Frederic M. Lord. *Applications of Item Response Theory to Practical Testing Problems.* Routledge, 1980.
- **[Foundational]** Shie Mannor, John N. Tsitsiklis. *The Sample Complexity of Exploration in the Multi-Armed Bandit Problem.* JMLR, 2004.
- **[Foundational]** John P. Lalor, Hao Wu, Hong Yu. *Building an Evaluation Scale using Item Response Theory.* EMNLP, 2016.
- **[Foundational]** Pedro Rodriguez, Joe Barrow, Alexander Hoyle, John P. Lalor, Robin Jia, Jordan Boyd-Graber. *Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?* ACL, 2021.
- **[SOTA]** Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, Mikhail Yurochkin. *tinyBenchmarks: Evaluating LLMs with Fewer Examples.* ICML, 2024.
- **[SOTA]** Alex Kipnis, Konstantinos Voudouris, Luca M. Schulze Buckley, John Burden, Lucy Cheke, Eric Schulz. *metabench — A Sparse Benchmark of Reasoning and Knowledge in Large Language Models.* ICLR, 2025.
- **[SOTA]** Yotam Perlitz, Elron Bandel, Ariel Gera, Ofir Arviv, Liat Ein-Dor, Eyal Shnarch, Noam Slonim, Michal Shmueli-Scheuer, Leshem Choshen. *Efficient Benchmarking of Language Models.* NAACL, 2024.
- **[SOTA]** Rajan Vivek, Kawin Ethayarajh, Diyi Yang, Douwe Kiela. *Anchor Points: Benchmark Nodes for Efficient Evaluation of Language Models.* ICLR, 2024.
- **[Method]** Kevin Jamieson, Robert Nowak. *Active Ranking using Pairwise Comparisons.* NeurIPS, 2011.
- **[Method]** Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, Jasjeet Sekhon. *Time-uniform, nonparametric, nonasymptotic confidence sequences.* Annals of Statistics, 2021.
- **[Method]** Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, Tijana Zrnic. *Prediction-powered inference.* Science, 2023.
- **[Survey/Systems]** Wei-Lin Chiang et al. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* ICML, 2024.

## 10. Worked Example

Two models on MMLU ($N = 14{,}042$): $\theta_A = 0.700$, $\theta_B = 0.685$, so $\Delta = 0.015$. Suppose they disagree on $d = 0.20$ of items.

Paired variance: $\mathrm{Var}(D_i) = 0.20 - 0.015^2 \approx 0.1998$. Items needed to call the order at 95% confidence with 80% power:
$$n \;\ge\; \frac{(1.96+0.84)^2 \cdot 0.1998}{0.015^2} \;=\; \frac{7.84 \times 0.1998}{2.25\times10^{-4}} \;\approx\; 6{,}960 .$$
Half the benchmark — *with* pairing, which is already the efficient design. A 100-item tiny subset gives standard error $\sqrt{0.1998/100} = 0.045$: three times the gap. The reported "2% average error" of 100-item subsets is therefore fully consistent with being unable to order two models 1.5 points apart.

Now the adaptive fix and its cost. Restrict to the 2,808 disagreement items: there $\mathbb{E}[D_i] = 0.015/0.20 = 0.075$ and $\mathrm{Var}(D_i) \approx 1 - 0.075^2 \approx 0.994$, giving $n \ge 7.84\times0.994/0.075^2 \approx 1{,}385$ — a 5× saving. But the disagreement set is defined by the answers of $A$ and $B$, which you learn only by evaluating them; using a *proxy* disagreement set from earlier models biases the estimator whenever the new models' error structure differs. And the accuracy computed on that subset is $\approx 0.5$, not $0.70$ — the same procedure that buys the 5× destroys the score. That is the obstruction in one instance: the informative items and the representative items are different items, and the field reports one number for both.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*