---
id: 06-data-pipeline/marginal-value-of-data-source
title: "Estimating Marginal Value of an Additional Data Source"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Estimating Marginal Value of an Additional Data Source

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/marginal-value-of-data-source` · **Status:** empirically-open

## 1. Problem Statement

A pretraining corpus is assembled from sources: Common Crawl dumps, GitHub, arXiv, StackExchange, licensed books, a vendor's proprietary chat logs. Someone must decide whether to pay for source $k$, crawl it, clean it, or drop it. That decision needs a number: how much does the final model improve if source $k$ is available, holding compute fixed and re-optimizing everything else?

**Input.** A set of candidate sources $\{S_1,\dots,S_K\}$ (each a token pool with a size and an acquisition cost), a training recipe, a compute budget $C$, and an evaluation $\mathcal{E}$.

**Output.** For each $k$, an estimate $\hat\Delta_k$ of the change in $\mathcal{E}$ from including $S_k$, with a confidence interval, produced at cost $\ll$ the cost of training the model.

Three variants that are routinely conflated:

- **Measurement.** Given unlimited compute, is $\Delta_k$ even well defined? It depends on which other sources are present, on the mixture weights, on $C$, and on $\mathcal{E}$. A single scalar "value of source $k$" does not exist without fixing all four.
- **Method.** Given the definition, estimate $\Delta_k$ cheaply — proxy models, scaling-law extrapolation, influence functions, subset-training surrogates.
- **Theory.** Under what conditions is a cheap estimator consistent for the expensive quantity? No known result gives non-vacuous transfer guarantees from a $10^7$-parameter proxy to a $10^{10}$-parameter target.

Solving it means: predicting the sign and rank of $\Delta_k$ for held-out sources at target scale, validated against actual target-scale training, at $<5\%$ of the target-scale cost.

## 2. Formal Setting

Sources are distributions $P_1,\dots,P_K$ over documents. A **mixture** is a weight vector $w \in \Delta^{K-1}$; training draws tokens i.i.d. from $P_w = \sum_k w_k P_k$. A run is $(w, N, D)$: parameters $N$, tokens $D$, compute $C \approx 6ND$ FLOPs (Kaplan et al. 2020; Hoffmann et al. 2022).

Let $\theta(w,N,D)$ be the trained parameters and

$$L(w,N,D) \;=\; \mathbb{E}_{x\sim Q}\big[-\log p_{\theta(w,N,D)}(x)\big]$$

the loss on an **evaluation distribution** $Q$, measured as mean token negative log-likelihood in nats on a held-out set disjoint from all $P_k$ after cross-source deduplication. Downstream accuracy $A(\cdot)$ substitutes for $L$ when the decision is task-specific.

**Marginal value with re-optimization.** The only definition that is not confounded with mixture tuning:

$$\Delta_k(C) \;=\; \min_{w:\,w_k=0}\; L^\star(w, C) \;-\; \min_{w}\; L^\star(w, C), \qquad L^\star(w,C)=\min_{6ND\le C} L(w,N,D).$$

$\Delta_k \ge 0$ by construction — the inner minimization over a superset can only help. Cost per source pair: two full mixture searches. **Naive ablation**, $L(w^{-k}) - L(w^\star)$ with $w^{-k}$ the renormalized $w^\star$, is what practitioners actually compute; it can be negative and does not equal $\Delta_k$.

**Cooperative-game framing.** With $v(T) = -\min_{w:\,\mathrm{supp}(w)\subseteq T} L^\star(w,C)$ for $T\subseteq[K]$, the Shapley value

$$\phi_k \;=\; \sum_{T\subseteq [K]\setminus\{k\}} \frac{|T|!\,(K-|T|-1)!}{K!}\,\big(v(T\cup\{k\})-v(T)\big)$$

is the unique attribution satisfying efficiency, symmetry, null-player and linearity (Shapley 1953). It costs $2^K$ mixture searches exactly, and Monte Carlo estimation needs $O(\varepsilon^{-2})$ full trainings.

**Assumptions, and which break.**

| Assumption | Status |
|---|---|
| Sources are disjoint | **Violated.** C4, RefinedWeb, FineWeb are all Common Crawl derivatives; arXiv text recurs inside web scrapes. Overlap makes $\phi_k$ split credit between near-duplicates and drives each toward zero. |
| Value is monotone ($\Delta_k\ge0$ in practice) | **Violated for naive ablation**, and even under re-optimization when $Q$ shifts or the fixed token budget forces displacement. |
| Tokens within a source are exchangeable | **Violated.** Within-source quality variance often exceeds between-source variance (DataComp, 2023). |
| $L$ is smooth in $w$ | Approximately holds; the basis of data mixing laws (Ye et al. 2024). |
| Proxy-scale ranking transfers to target scale | **The open question.** Known to fail for filtering aggressiveness (Goyal et al. 2024). |

## 3. State of the Art

**Established.**
- *Scaling-law extrapolation over mixtures.* **Data Mixing Laws** (Ye et al. 2024) fit $L(w)$ as a sum of exponentials in $w$ and predict a better mixture for a 1B/100B-token run from small-scale fits. **RegMix** (Liu et al. 2025) trains 512 models of 1M parameters on random mixtures, fits a regression, and picks a mixture whose 1B-model performance beats human-chosen and DoReMi mixtures; the selected mixture transfers to 7B. Both give *rankings of mixtures*, not calibrated $\Delta_k$ with error bars.
- *Group DRO reweighting.* **DoReMi** (Xie et al. 2023) learns domain weights with a 280M proxy and improves a 8B model's downstream accuracy by 6.5% average few-shot on The Pile, reaching baseline accuracy in 2.6× fewer steps.
- *Data attribution.* **Datamodels** (Ilyas et al. 2022) show subset-training regressions predict held-out behavior linearly and surprisingly well; **TRAK** (Park et al. 2023) reaches comparable attribution quality at orders of magnitude lower cost; **DsDm** (Engstrom et al. 2024) uses datamodels to select LM pretraining data and beats classifier-based selection on target tasks at 1.2B scale.
- *Repetition substitutes for new sources.* **Data-constrained scaling** (Muennighoff et al. 2023) shows up to 4 epochs of repeated data are nearly as valuable as fresh tokens; value decays to near zero by ~40 epochs. This bounds $\Delta_k$ from above whenever existing sources can be repeated.

**Claimed but unablated.**
- That proxy-model mixture rankings transfer across an order of magnitude in $N$. RegMix reports 1M→1B→7B transfer for the *selected* mixture; the *per-source marginal* transfer is not ablated.
- That Shapley-style values computed on fine-tuning sets say anything about pretraining sources. Data Shapley (Ghorbani & Zou 2019), Beta-Shapley (Kwon & Zou 2022), Data Banzhaf (Wang & Jia 2023), LAVA (Just et al. 2023) are validated on $10^3$–$10^5$-example classification, not on $10^{12}$-token corpora.

**Benchmark-number-only.** DataComp (Gadre et al. 2023) and DataComp-LM (Li et al. 2024) rank *filtering pipelines* by a single leaderboard score at fixed scale. That is a ranking of recipes, not a per-source marginal value, and the leaderboard does not report seed variance.

## 4. What Is Known

- **Ordering of sources flips with scale.** Goyal et al. (CVPR 2024) show the optimal data-filtering aggressiveness depends on compute: aggressive quality filtering wins at small compute, hurts at large compute where repeated high-quality data is exhausted. Measured on LAION/DataComp pools, 32M–640M sample budgets.
- **Deduplication changes value more than source identity.** Lee et al. (ACL 2022) find near-duplicate removal cuts memorized-continuation emission by 10×, with equal or better perplexity. A source's marginal value is largely its *non-duplicate* mass.
- **Small curated sources can dominate.** In the Pile ablations and in FineWeb-Edu (Penedo et al. 2024), an educational-quality classifier on Common Crawl yields large MMLU gains at 1.8B scale relative to unfiltered FineWeb — a within-source reweighting outperforming several whole added sources.
- **Pruning has a scaling exponent, not just an offset.** Sorscher et al. (2022) show optimal pruning can beat power-law scaling on ImageNet/CIFAR, i.e. data value is not a fixed additive term.
- **Seed noise is the binding constraint at small $\Delta$.** Reported run-to-run std on downstream few-shot accuracy at 1B scale is on the order of 0.3–1.0 points, comparable to the effect of adding a 2–3% source.

## 5. What Is Not Known

- **Empirically open.** Does a cheap estimator's *per-source* ranking match target-scale $\Delta_k$? Nobody has published a matched study that computes $\hat\Delta_k$ at proxy scale for $K\ge8$ sources and validates each against leave-one-source-out training at $\ge7$B with enough seeds to beat noise. The experiment is runnable today; it costs a few hundred thousand GPU-hours.
- **Theoretically open.** No consistency or transfer bound for proxy-scale data valuation under distribution shift in $N$ and $D$. Influence-function guarantees (Koh & Liang 2017) assume convexity and a converged optimum; Grosse et al. (2023) show LLM influence estimates are usable but give no error bound on aggregate source-level sums.
- **Methodologically blocked.** "Value of a source" has no agreed definition once sources overlap and the mixture is re-optimized. Shapley over overlapping sources assigns near-zero to every member of a duplicate cluster; leave-one-out assigns full value to each. Both are defensible; they disagree by orders of magnitude. Until the catalog fixes a convention, cross-paper numbers are incomparable.

## 6. Why It Is Hard

The obstruction is **effect size below measurement noise, compounded by a $2^K$ counterfactual space**. A 2% source moves held-out loss by $\sim10^{-3}$ nats at 1B scale, while seed-to-seed std is $\sim5\times10^{-3}$ nats. Detecting a difference of $\delta$ against noise $\sigma$ at 80% power needs about $n \approx 16(\sigma/\delta)^2 \approx 400$ runs per arm. Second obstruction: **non-identifiability under overlap** — with two near-duplicate sources, $\Delta_1$, $\Delta_2$ and $\Delta_{\{1,2\}}$ cannot all be recovered from marginal ablations. Third: the quantity is **not scale-invariant**; a measurement at 1B does not name the quantity the buyer needs at 70B.

## 7. Current Research (as of 2026)

- Regression-over-mixtures at proxy scale: RegMix (Sail/Singapore), Data Mixing Laws (Fudan/Shanghai AI Lab), Aioli (Chen et al. 2024, Stanford) which unifies online mixing methods and shows several underperform stratified baselines.
- Target-aware selection: DsDm and LESS (Xia et al. 2024, Princeton) — select data for a *specific* evaluation, sidestepping the "universal value" definition.
- Cheap Shapley: In-Run Data Shapley (Wang et al. 2024) computes Shapley values within a single training run using gradient checkpoints. *(frontier — verify)* Extension from example-level to source-level aggregation at pretraining scale is claimed in follow-ups but not independently reproduced.
- Data markets and pricing under overlap *(frontier — verify)*: extensions of Agarwal, Dahleh & Sarkar (EC 2019) to correlated sellers.

## 8. Concrete Next Experiment

**Question.** Does proxy-scale per-source marginal value predict target-scale marginal value in rank order?

**Scale.** $K=8$ disjointified sources (cross-source MinHash dedup first). Proxy arm: 100 models at 150M/3B tokens on random mixtures ($\approx$ 2.7e18 FLOPs each, $\approx$ 2.2e20 total). Target arm: 9 runs at 3B/60B tokens ($6ND=1.08\times10^{21}$ FLOPs each) — one full mixture plus eight leave-one-source-out, each with mixture weights re-optimized by renormalization *and* by a 3-point local search. Repeat the target arm with 5 seeds: 45 runs, $\approx 4.9\times10^{22}$ FLOPs, roughly 250k H100-hours at 40% MFU.

**Control arm.** Eight *placebo* ablations that remove a random 1/K of tokens sampled proportionally from all sources. These have true $\Delta=0$ up to repetition effects and give the empirical null distribution of measured $\hat\Delta$.

**Deciding number.** Spearman $\rho$ between proxy-predicted $\Delta_k$ and target-measured $\Delta_k$, over the sources whose target-measured $\Delta_k$ exceeds the 95th percentile of the placebo null. **$\rho \ge 0.7$ decides for transfer; $\rho \le 0.3$ decides against.** Report the placebo null width alongside — if fewer than 4 of 8 sources clear it, the answer is that 3B/60B is too small to measure source value at all, which is itself the finding.

## 9. Key References

- **[Foundational]** L. S. Shapley. *A Value for n-Person Games.* Contributions to the Theory of Games II, Princeton University Press, 1953.
- **[Foundational]** A. Ghorbani, J. Zou. *Data Shapley: Equitable Valuation of Data for Machine Learning.* ICML 2019. — arXiv:1904.02868
- **[Foundational]** P. W. Koh, P. Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017. — arXiv:1703.04730
- **[SOTA]** Q. Liu, X. Zheng, N. Muennighoff, et al. *RegMix: Data Mixture as Regression for Language Model Pre-training.* ICLR 2025. — arXiv:2407.01492
- **[SOTA]** J. Ye, P. Liu, Q. Sun, et al. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* 2024. — arXiv:2403.16952
- **[SOTA]** S. M. Xie, H. Pham, X. Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. — arXiv:2305.10429
- **[SOTA]** L. Engstrom, A. Feldmann, A. Mądry. *DsDm: Model-Aware Dataset Selection with Datamodels.* ICML 2024. — arXiv:2401.12926
- **[SOTA]** S. M. Park, K. Georgiev, A. Ilyas, G. Leclerc, A. Mądry. *TRAK: Attributing Model Behavior at Scale.* ICML 2023. — arXiv:2303.14186
- **[SOTA]** A. Ilyas, S. M. Park, L. Engstrom, G. Leclerc, A. Mądry. *Datamodels: Predicting Predictions from Training Data.* ICML 2022. — arXiv:2202.00622
- **[Empirical]** N. Muennighoff, A. M. Rush, B. Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[Empirical]** S. Goyal, P. Maini, Z. C. Lipton, A. Raghunathan, J. Z. Kolter. *Scaling Laws for Data Filtering — Data Curation Cannot Be Compute Agnostic.* CVPR 2024. — arXiv:2404.07177
- **[Empirical]** K. Lee, D. Ippolito, A. Nystrom, et al. *Deduplicating Training Data Makes Language Models Better.* ACL 2022. — arXiv:2107.06499
- **[Empirical]** B. Sorscher, R. Geirhos, S. Shekhar, S. Ganguli, A. S. Morcos. *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS 2022. — arXiv:2206.14486
- **[Benchmark]** S. Y. Gadre, G. Ilharco, A. Fang, et al. *DataComp: In Search of the Next Generation of Multimodal Datasets.* NeurIPS 2023. — arXiv:2304.14108
- **[Benchmark]** J. Li, A. Fang, G. Smyrnis, et al. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models.* NeurIPS 2024. — arXiv:2406.11794
- **[Related]** A. Agarwal, M. Dahleh, T. Sarkar. *A Marketplace for Data: An Algorithmic Solution.* ACM EC 2019. — arXiv:1805.08125
- **[Survey]** Y. Kwon, J. Zou. *Beta Shapley: A Unified and Noise-Reduced Data Valuation Framework for Machine Learning.* AISTATS 2022. — arXiv:2110.14049

## 10. Worked Example

A lab is offered a licensed 40B-token technical-forum corpus for \$1.2M. Existing corpus: 1.2T tokens. Target: a 7B model trained on 300B tokens ($6ND = 1.26\times10^{22}$ FLOPs).

**Step 1 — upper bound from repetition.** The new source would be 40/1240 = 3.2% of the pool, and at sampling weight 3.2% it contributes 9.7B tokens to a 300B run. Muennighoff et al. give near-full value for $\le4$ epochs, so displacing 9.7B tokens of existing web text — repeatable at well under 4 epochs — means the *incremental* effect is bounded by the quality gap between the two, not by token count.

**Step 2 — proxy measurement.** Train a 150M/3B proxy with and without the source, 3 seeds each. Result: held-out loss $2.8412$ vs $2.8395$ nats, so $\hat\Delta = 0.0017$ nats. Seed std within arm: $0.0021$ nats.

**Step 3 — the obstruction, in numbers.** The two-sample $t$ with $n=3$ per arm gives $t = 0.0017/(0.0021\sqrt{2/3}) = 0.99$, $p \approx 0.38$. To reach $p<0.05$ at this effect size needs $n \approx 16(0.0021/0.0017)^2 \approx 25$ seeds per arm — 50 proxy runs, affordable. But the proxy is 150M; the decision is at 7B. Goyal et al. show the *sign* of curation effects flips across a compute decade, so 50 significant proxy runs still do not license the purchase.

**Step 4 — what the number buys.** One 7B/300B run at 40% MFU on 512 H100s is $\approx1.26\times10^{22}/(512\cdot4\times10^{14}) \approx 6.2\times10^{4}$ s $\approx 17$ hours, or 8,800 GPU-hours. Confirming a 0.0017-nat effect at 7B needs $\sim$25 seeds × 2 arms = 440k GPU-hours, roughly \$1.3M at \$3/GPU-hour — **more than the asking price of the data**. The measurement costs more than the decision it informs. That inversion, not the estimator's accuracy, is why the field buys data on judgment and leaderboard proxies rather than on measured marginal value.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*