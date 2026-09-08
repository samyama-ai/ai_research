---
id: 03-training-dynamics/data-constrained-optimal-model-size
title: "Data-Constrained Optimal Model Size"
topic: 03-training-dynamics
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data-Constrained Optimal Model Size

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/data-constrained-optimal-model-size` · **Status:** partially-solved

## 1. Problem Statement

Chinchilla answers: given compute $C$ and unlimited fresh tokens, what model size $N$ minimizes loss? The answer is $N^\star \propto C^{1/2}$, roughly 20 tokens per parameter.

The data-constrained problem replaces "unlimited" with a fixed pool of $U$ unique tokens. Given $(C, U)$, choose $N$, the number of epochs $R$, and any data-manipulation policy (dedup, filtering, mixing, synthetic augmentation) to minimize held-out loss on the target distribution. Three variants, differing sharply in difficulty:

- **Measurement.** Fit $L(N, D, U)$ empirically and read off $\arg\min_N$. Runnable today; run only at $N \le 9$B, $U \le 178$B tokens.
- **Method.** Given a real pool (e.g. all of high-quality English web text, $\sim 10^{13}$ tokens), produce the allocation that actually wins at $10^{26}$ FLOP. Requires extrapolating the fit two to three orders of magnitude in $C$.
- **Theory.** Explain *why* repeated tokens decay in value with a characteristic epoch count, and predict that constant from data properties rather than fitting it. Fully open.

Solving it means: a rule $N^\star(C, U)$ with stated extrapolation error bars, validated by a held-out training run at a scale not used in the fit.

## 2. Formal Setting

Let $\mathcal{D}$ be the target token distribution and $\mathcal{U}$ a pool of $U$ unique tokens sampled from a (generally different) source distribution. A run processes $D$ total tokens; $R = D/U$ is the epoch count. $N$ is non-embedding parameter count. Compute is measured as $C \approx 6ND$ FLOP (forward+backward, ignoring attention's quadratic term — an error of $<10\%$ at sequence length 2048, $N \ge 1$B, but $>25\%$ at 128k context).

The measured objective is cross-entropy on a held-out shard *disjoint from $\mathcal{U}$ at the document level after near-duplicate removal*:
$$L = -\frac{1}{|\mathcal{H}|}\sum_{t \in \mathcal{H}} \log p_\theta(t \mid t_{<t}).$$
Document-level disjointness is the operative definition; token-level overlap is unavoidable and not controlled.

Muennighoff et al. (2023) fit an extension of the Chinchilla form in which repeated tokens and re-used parameters are discounted exponentially:
$$L(N, D) = E + \frac{A}{(N')^{\alpha}} + \frac{B}{(D')^{\beta}},$$
$$D' = U_D + U_D R_D^\star\left(1 - e^{-R_D/R_D^\star}\right), \qquad N' = U_N + U_N R_N^\star\left(1 - e^{-R_N/R_N^\star}\right),$$
where $U_D = \min(U, D)$ is unique tokens seen, $R_D = D/U_D - 1$ is repetitions beyond the first, and $R_D^\star$ is the fitted *half-life in epochs* of a repeated token's value. $N'$ applies the symmetric discount to excess parameters. The compute-optimal size is then
$$N^\star(C, U) = \arg\min_{N} \; L\!\left(N, \tfrac{C}{6N}\right).$$

Assumptions, and which are violated:

- **Power-law-plus-constant form holds over the extrapolated range.** Unverified above $10^{24}$ FLOP; the irreducible term $E$ is not separately identifiable from $A/(N')^\alpha$ at small $N$.
- **$R^\star$ is a constant of the data, independent of $N$ and $C$.** Known to be false in direction: larger models memorize repeated data faster (Hernandez et al., 2022), so $R^\star$ should shrink with $N$. The fit absorbs this into $R_N^\star$.
- **One optimizer/schedule.** Cosine schedule length matched to $D$. Violated whenever practitioners over-train past the schedule or use warmup-stable-decay.
- **$\mathcal{U}$ is i.i.d. and quality-homogeneous.** False for web corpora: the marginal value of epoch 2 on a filtered pool differs from epoch 2 on an unfiltered one.

## 3. State of the Art

**Empirical SOTA (established).** Muennighoff et al., *Scaling Data-Constrained Language Models* (NeurIPS 2023, arXiv:2305.16264). ~400 training runs, up to 9B parameters and 900B total tokens, unique pools from 100M to 178B tokens (C4 and OSCAR). Fitted $R_D^\star \approx 15.4$ and $R_N^\star \approx 5.3$. Two results are ablated and reproduced within the paper across two corpora: (i) up to ~4 epochs, repeated tokens are worth close to fresh ones — loss within a few percent of the unique-data control; (ii) by ~40 epochs, added compute buys essentially nothing.

**Established corollary.** Under a data constraint, the compute-optimal $N$ is *smaller* than Chinchilla's $C^{1/2}$ prescription, and the optimal token-per-parameter ratio *rises* with $R$. This is a fit-derived result, robust across their two corpora but demonstrated only inside the fitted range.

**Claimed but unablated.** The same paper's data-manipulation arms — code mixing (up to ~50% Python without English degradation), perplexity filtering, and deduplication as substitutes for fresh data — are single-corpus, single-scale results. The finding that dedup does *not* help when data is scarce is directionally consistent with Hernandez et al. (2022) but has not been independently reproduced.

**Benchmark-only numbers.** Claims that heavily repeated-data models are "as good" downstream rest on aggregate few-shot suites at $\le 9$B. No study reports a repeat-vs-fresh comparison on a downstream metric with per-task confidence intervals at $\ge 30$B.

**Theory SOTA.** There is none for repetition. The single-epoch Chinchilla exponents have been re-derived and corrected (Besiroglu et al., 2024, arXiv:2404.10102, showing Hoffmann et al.'s Approach 3 fit had understated uncertainty), but no theory predicts $R^\star$ from corpus statistics.

## 4. What Is Known

- **Chinchilla baseline:** at $C \approx 5.76 \times 10^{23}$ FLOP, $N^\star \approx 70$B with $D \approx 1.4$T, beating the 280B Gopher trained on 300B tokens (Hoffmann et al., 2022). Exponents $a \approx b \approx 0.5$ in $N^\star \propto C^a$.
- **Four epochs are nearly free:** at 8.7B unique tokens, models up to 8.7B params trained for 4 epochs land within noise of the fresh-data control (Muennighoff et al., 2023).
- **Decay is exponential, not abrupt:** $R_D^\star = 15.4$ means the *marginal* value of an epoch has fallen to $1/e$ by epoch ~16; total effective tokens saturate at $D'_{\max} \approx U(1 + R_D^\star) \approx 16.4U$.
- **Repetition hurts large models more:** at 800M params, repeating a 0.1% subset many times produced a measurable loss degradation and a documented drop in induction-head strength (Hernandez et al., 2022, arXiv:2205.10487); the effect grows with $N$.
- **Pruning can beat power laws:** with a good example-difficulty ranking, exponential rather than power-law error scaling in dataset size is achievable on ImageNet-scale vision (Sorscher et al., NeurIPS 2022, arXiv:2206.14486). Not replicated for LM pretraining loss.
- **Over-training is predictable:** loss and downstream error remain fittable at $32\times$ Chinchilla token ratios up to 6.9B params (Gadre et al., 2024, arXiv:2403.08540).
- **Knowledge capacity is roughly $2$ bits/parameter** at sufficient exposure count, and drops sharply when facts appear too few times (Allen-Zhu & Li, arXiv:2404.05405) — evidence that $R^\star$ is fact-frequency-dependent, not a single scalar.
- **Public high-quality text is finite:** estimated $\sim 3\times10^{14}$ tokens of indexed web text, with effective exhaustion projected between 2026 and 2032 (Villalobos et al., ICML 2024, arXiv:2211.04325).

## 5. What Is Not Known

- **Empirically open.** Does $N^\star(C,U)$ from the 9B fit hold at $10^{26}$ FLOP with $U = 10^{13}$? The experiment is fully specified and unrun — it costs a frontier pretraining budget and nobody publishes the control arm.
- **Empirically open.** Is $R^\star$ a function of $N$? A fit that lets $R_D^\star$ vary with model size has not been published.
- **Theoretically open.** No derivation of $R^\star$ from corpus properties (duplication rate, entropy, description length). No proof that the $E + AN^{-\alpha} + BD^{-\beta}$ form is the right ansatz under repetition rather than a convenient two-decade interpolant.
- **Methodologically blocked.** Once high-quality synthetic data is in the pool, "unique tokens $U$" stops being well defined — synthetic tokens are neither fresh nor repeats, and no accepted estimator of their *effective* uniqueness exists.
- **Methodologically blocked.** Repeated-data models may match on loss while diverging on downstream capability. There is no agreed metric that separates "compressed the training set" from "learned the skill", so repeat-vs-fresh comparisons are decided on a proxy.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under extrapolation**, compounded by cost.

$L(N,D)$ has seven free parameters ($E, A, B, \alpha, \beta, R_D^\star, R_N^\star$) fitted over roughly two decades of $N$ and $U$. Several are strongly correlated: raising $E$ and lowering $A$ trades off almost exactly at small $N$, and $R_D^\star$ trades against $\beta$ because both control how fast the data term flattens. Besiroglu et al. (2024) showed that even the simpler five-parameter Chinchilla fit had confidence intervals wide enough to admit materially different exponents. Extrapolating three decades in $C$ turns a modest parameter correlation into a wide spread of $N^\star$ predictions — and the argmin is over a shallow basin, so the *loss penalty* for getting $N^\star$ wrong by $2\times$ is small (typically $<1\%$ relative) while the *number* $N^\star$ itself is poorly pinned. That combination is the trap: the fit looks excellent and the decision it supports is under-determined.

Second obstruction: **absent ground truth on the target**. "Data-constrained" in practice means constrained on *good* data, but goodness is defined by the filter, and the filter changes what an epoch is worth. There is no corpus-independent definition of $U$.

## 7. Current Research (as of 2026)

- **Repetition-aware allocation at frontier scale.** Labs training on multi-epoch web corpora are internally re-fitting $R^\star$; no frontier lab has published a data-constrained ablation with a fresh-data control *(frontier — verify)*.
- **Synthetic and rephrased data as a uniqueness substitute.** Rephrasing-the-web and textbook-style generation lines continue; the open question is whether generated tokens contribute to $U$ or to $R$.
- **Quality-aware scaling laws.** Extensions where the data term depends on a filter threshold, following Goyal et al., *Scaling Laws for Data Filtering* (CVPR 2024), which showed filtering-optimal thresholds depend on the compute budget.
- **Inference-aware allocation.** Sardana et al. (ICML 2024, arXiv:2401.00448) shift $N^\star$ down by folding in serving cost; this compounds with the data constraint and the two corrections have not been fitted jointly.
- **Downstream-metric scaling laws** (Gadre et al. line) — the route to replacing loss as the decision variable.

## 8. Concrete Next Experiment

**Question:** does $R_D^\star$ depend on model size?

**Scale.** Fix $U = 10$B unique tokens from a deduplicated, quality-filtered web corpus. Train a $3 \times 4$ grid: $N \in \{0.4\text{B}, 1.4\text{B}, 7\text{B}\}$ × $R \in \{1, 4, 16, 40\}$, cosine schedule matched to each $D$. Total $\approx 1.5 \times 10^{22}$ FLOP — roughly 20k H100-hours, reachable on an academic cluster.

**Control arm.** For each $(N, D)$ cell, an identical run drawing $D$ *fresh* tokens from a $400$B-token reserve of the same corpus, same filter, same shuffling. This is the arm that is routinely omitted and without which nothing is decidable.

**Deciding number.** Fit $R_D^\star$ separately per model size. Report
$$\Delta = \frac{R_D^\star(0.4\text{B}) - R_D^\star(7\text{B})}{R_D^\star(0.4\text{B})}.$$
If $\Delta > 0.20$ with bootstrap CI excluding zero, $R^\star$ is size-dependent, the single-constant fit is misspecified, and every extrapolation of $N^\star(C,U)$ built on it — including the "smaller models under data constraint" corollary — inherits a bias whose sign is now known. If $|\Delta| < 0.05$, the constant-$R^\star$ assumption survives one more decade and the Muennighoff fit can be extrapolated with more confidence.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[SOTA]** Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[SOTA]** Xue, Fu, Zheng, et al. *To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis.* NeurIPS 2023. — arXiv:2305.13230
- Hernandez, Brown, Conerly, et al. *Scaling Laws and Interpretability of Learning from Repeated Data.* Anthropic, 2022. — arXiv:2205.10487
- Sorscher, Geirhos, Shekhar, Ganguli, Morcos. *Beyond neural scaling laws: beating power law scaling via data pruning.* NeurIPS 2022. — arXiv:2206.14486
- Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- Gadre, Smyrnis, Shankar, et al. *Language models scale reliably with over-training and on downstream tasks.* 2024. — arXiv:2403.08540
- Sardana, Portes, Doubov, Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024. — arXiv:2401.00448
- Allen-Zhu, Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Survey]** Villalobos, Ho, Sevilla, Besiroglu, Heim, Hobbhahn. *Will we run out of data? Limits of LLM scaling based on human-generated data.* ICML 2024. — arXiv:2211.04325
- Goyal, Maini, Lipton, Raghunathan, Kolter. *Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic.* CVPR 2024.

## 10. Worked Example

Budget: $C = 10^{22}$ FLOP. Pool: $U = 5$B unique tokens.

**Chinchilla answer.** $D = 20N$ and $C = 6ND$ give $N = \sqrt{C/120} \approx 9.1$B, $D \approx 182$B tokens. But only 5B are unique, so this demands $R = 36$ epochs.

**Data-constrained answer.** Effective tokens saturate at $D' \le U(1 + R_D^\star) = 5\text{B} \times 16.4 = 82$B. At $R = 36$, $R_D = 35$ and
$$D' = 5\left[1 + 15.4\left(1 - e^{-35/15.4}\right)\right] \approx 5 \times 14.4 = 72\text{B}.$$
So 182B tokens of compute buy 72B tokens of value — a 60% waste. Re-optimizing pushes $N$ down and, since $C$ is fixed, $D$ up; the fit's answer lands near $N \approx 4$–5B at $R \approx 70$–90, with an effective $D'$ barely above 78B.

**Where the obstruction becomes visible.** Compare the two allocations under the fitted law: the loss gap between $N = 9.1$B and $N = 4.5$B is on the order of $0.01$–$0.02$ nats — under 1% relative. The basin is nearly flat, so the fit's confidence interval on $N^\star$ spans the whole range $3$–$10$B. Meanwhile the *engineering* consequences diverge completely: a 4.5B model serves at half the cost and needs 90 epochs of data loading; a 9.1B model needs a different parallelism plan. The scaling law is precise about the objective and near-silent about the decision. That, not compute cost alone, is why the problem stays only partially solved.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*