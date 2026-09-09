---
id: 23-privacy-memorization/exact-unlearning-nonconvex-models
title: "Exact Machine Unlearning for Deep Nonconvex Models"
topic: 23-privacy-memorization
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exact Machine Unlearning for Deep Nonconvex Models

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/exact-unlearning-nonconvex-models` · **Status:** solved-but-impractical

## 1. Problem Statement

**Input.** A training set $D$ of $n$ examples, a stochastic learning algorithm $A$ producing a deep nonconvex model $A(D)$, and a stream of deletion requests $z_1, z_2, \dots$

**Output.** After request $z_i$, a model that is *exactly* the model you would have obtained had $z_1,\dots,z_i$ never been in $D$ — in distribution, not pointwise.

**Objective.** Achieve this at cost sublinear in the cost of retraining from scratch, without losing test accuracy.

Three variants, and they differ sharply:

- **Theory variant.** Does an exact unlearning algorithm exist for nonconvex $A$ with amortized deletion cost $o(C_{\text{train}})$ and no accuracy loss? Trivially yes if you allow the sharded construction; the open part is the cost/accuracy frontier.
- **Method variant.** Build one that works at frontier scale (10B+ parameters, web-scale corpora, deduplication, curriculum, adaptive requests). This is where the problem is unsolved in practice.
- **Measurement variant.** Given a deployed model and a deletion claim, verify it. This is **not** currently well posed for a single released model (§5, §6).

The status `solved-but-impractical` is precise: SISA (Bourtoule et al., IEEE S&P 2021) *is* an exact unlearning algorithm for arbitrary nonconvex $A$. Its cost, at realistic deletion rates and realistic scale, is close to retraining.

## 2. Formal Setting

Let $\mathcal{Z}$ be the example space, $D \in \mathcal{Z}^n$, $\Theta$ the parameter space. $A: \mathcal{Z}^* \times \mathcal{R} \to \Theta$ is a randomized learner with random seed $r \sim \mathcal{R}$ (init, shuffling, augmentation, dropout, nondeterministic GPU reductions).

An unlearning operator $U$ takes the trained model, the dataset, and the request. **Exact unlearning** is distributional equality:

$$U\big(A(D,r),\, D,\, z\big) \;\stackrel{d}{=}\; A\big(D \setminus \{z\},\, r'\big) \quad \text{for all } D, z .$$

Approximate $(\varepsilon,\delta)$-unlearning (Guo et al., ICML 2020; Sekhari et al., NeurIPS 2021) relaxes this to a DP-style indistinguishability over measurable $S \subseteq \Theta$:

$$\Pr[U(\cdot) \in S] \le e^{\varepsilon}\Pr[A(D\setminus\{z\}) \in S] + \delta .$$

**Quantities as actually measured.**

- $C_{\text{train}}$: accelerator-hours for one full training run. Measured, not estimated — e.g. Llama 3 405B: $3.084\times 10^7$ H100-hours (Grattafiori et al., 2024).
- $C_{\text{del}}(K)$: accelerator-hours to service $K$ deletions. Measured on the same hardware. The reported figure must be **amortized over the request stream**, not the cost of one isolated deletion.
- Speedup $\rho = K \cdot C_{\text{train}} / C_{\text{del}}(K)$ against the naive retrain-per-request baseline.
- $\Delta_{\text{acc}}$: test accuracy of the unlearning-capable pipeline minus that of monolithic training, same compute, same architecture.
- Verification: exact unlearning is **certified by construction**, not by test. There is no finite-sample test of $\stackrel{d}{=}$ from one model.

**Assumptions, and which break.**

| Assumption | Status in practice |
|---|---|
| Examples i.i.d. and separable across shards | **Violated** — near-duplicates, web crawl repetition; deleting $z$ leaves its copies in other shards |
| Deletion requests non-adaptive (independent of released model) | **Violated** — requests follow model behavior; fixed by Gupta et al. (NeurIPS 2021) at extra cost |
| Hyperparameters, architecture, tokenizer chosen independently of $D$ | **Violated** — HP search and dedup thresholds are functions of all of $D$ |
| Training is a pure function of (shard, seed) | **Violated** — nondeterministic kernel reductions; irrelevant for $\stackrel{d}{=}$ but fatal for bitwise audit |

## 3. State of the Art

**Established, exact.**

- **SISA** — Bourtoule, Chandrasekaran, Choquette-Choo, Jia, Travers, Zhang, Lie, Papernot. *Machine Unlearning.* IEEE S&P 2021. Shard into $S$ disjoint slices, train one constituent model per shard, checkpoint after each of $R$ intra-shard slices, aggregate by label voting. Deletion retrains one shard from one checkpoint. Exact by construction, any $A$.
- **Quantized $k$-means deletion** — Ginart, Guan, Valiant, Zou. *Making AI Forget You.* NeurIPS 2019. Exact, $\sim10^4\times$ amortized speedup — but convex/clustering, not deep.
- **TV-stable exact unlearning** — Ullah, Mai, Rao, Rossi, Arora. *Machine Unlearning via Algorithmic Stability.* COLT 2021. Exact for convex ERM; the analysis does not transfer to nonconvex $A$.

**Established, approximate only.** Descent-to-Delete (Neel, Roth, Sharifi-Malvajerdi, ALT 2021) and certified removal (Guo et al., ICML 2020) give $(\varepsilon,\delta)$ guarantees under convexity/strong-convexity. Applied to deep nets, the guarantee is void — the influence-function correction has no valid bound off a convex loss.

**Claimed but unablated.** Most "unlearning for LLMs" work (gradient ascent on the forget set, preference-optimization variants, task-vector negation) reports forget-set accuracy or MIA AUC and calls it unlearning. Hayes, Shumailov, Triantafillou, Khalifa, Papernot (2024) show these evaluations, run per-example with a LiRA-strength attack (U-LiRA), reverse the ranking of published methods: several methods that look near-perfect under aggregate MIA leak the forget set under per-example attack. Treat any deep unlearning number not audited per-example as a benchmark artifact.

## 4. What Is Known

- **SISA speedups, measured.** $4.63\times$ on Purchase and $2.45\times$ on SVHN for a stream of 8 deletion requests, with negligible accuracy loss at $S=20$ (S&P 2021).
- **SISA accuracy cost scales with task difficulty.** On ImageNet with $S=20$, top-5 accuracy drops by $16.7$ percentage points relative to monolithic training. The mechanism is data starvation per constituent: each learner sees $n/S$ examples.
- **Speedup is capped by $S$.** With $R$ slices and uniform deletion location, expected retrain cost is $(R+1)(2R+1)/6$ slice-units against $S\,R(R+1)/2$ for full retraining, i.e. $\rho \approx 3SR/(2R+1) \to 1.5S$. Buying speedup means buying $\Delta_{\text{acc}}$.
- **Weight-level verification is impossible in general.** Thudi, Jia, Shumailov, Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security 2022. For a given released model, one can exhibit a training run on $D\setminus\{z\}$ that plausibly produced it; a data owner cannot prove deletion from weights alone. Unlearning must be defined on the *algorithm*, not the artifact.
- **Adaptivity costs.** Gupta, Jagielski, Ullman, Waites, Zhang (NeurIPS 2021) restore exactness under adaptive requests, but the guarantee degrades with the number of requests answered before rebuild.
- **Deduplication interacts destructively.** Carlini et al. (ICLR 2023) show memorization grows with duplication count; SISA's shard isolation gives no protection when a near-duplicate of $z$ sits in another shard.

## 5. What Is Not Known

- **Theoretically open.** Whether any exact unlearning scheme for nonconvex $A$ can achieve amortized deletion cost $o(C_{\text{train}}/S)$ *without* the accuracy penalty of data partitioning. No lower bound rules it out; no construction achieves it. Also open: a lower bound tying $\Delta_{\text{acc}}$ to $\rho$ for a fixed architecture and data budget.
- **Empirically open.** SISA has never been run at frontier LLM scale. Nobody has published $\rho$, $\Delta_{\text{acc}}$, and downstream benchmark deltas for a sharded 7B–70B pretraining run against a monolithic control. The experiment is runnable; the compute is the barrier.
- **Empirically open.** Whether ensemble aggregation over $S$ shards can be replaced by a cheap merge (parameter averaging, MoE routing) that preserves exactness *and* recovers monolithic accuracy on generative tasks.
- **Methodologically blocked.** Verification of an exact-unlearning claim by a third party holding only the released model. Thudi et al. show weight-level audit is unavailable; distributional audit needs $\Omega$(many) retrains. Trusted-execution or proof-of-training approaches are early and unmeasured at scale.

## 6. Why It Is Hard

Two named obstructions.

**Obstruction 1 — the speedup collapses under a realistic request stream (compute cost).** SISA's advantage assumes deletions are rare and land late in a slice ordering. Under a stream of $K$ requests spread uniformly, each shard is hit $K/S$ times and the *earliest* hit determines the retrain point. As $K \to S R$, the expected earliest hit index $\to 1$ and cost per rebuild $\to$ full shard training. Speedup decays roughly as $\rho(K) \approx 1.5S \cdot \mathbb{E}[\min]/\!R$, reaching $\approx 1$ at deletion rates well under 1% (§10).

**Obstruction 2 — the guarantee is on the algorithm, but the audit target is the artifact (non-identifiability).** Exactness is a statement about a distribution over seeds. A regulator receives one parameter vector. Thudi et al. (USENIX Sec 2022) prove that vector is consistent with training runs that both did and did not include $z$. So the guarantee is only as strong as the provider's attestation of its own pipeline — the strongest formal property in this literature is precisely the one that cannot be checked from outside.

## 7. Current Research (as of 2026)

- **Better exact partitions.** Learned/clustered sharding and hierarchical aggregation to cut $\Delta_{\text{acc}}$ at fixed $S$; the ImageNet gap is the target number *(frontier — verify)*.
- **Evaluation repair.** Per-example U-LiRA-style audits displacing aggregate MIA as the reporting standard, following Hayes et al. (2024) and the NeurIPS 2023 competition findings (Triantafillou et al.), which reported that the strongest competition entries still leaked under stronger attacks.
- **Datamodel-matched unlearning.** Georgiev et al., *Attribute-to-Delete: Machine Unlearning via Datamodel Matching* (2024) — target the *retrained model's predictions* rather than its weights. Approximate, but the objective is finally the right one.
- **LLM-specific relaxations.** Deletion at the retrieval/adapter layer rather than the base model; exactness is recovered because only the removable component ever touched $z$. Practically deployed, but it moves the problem rather than solving it.
- **Provenance and attestation.** Verifiable training logs and TEE-attested pipelines as a substitute for artifact-level audit *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** At LLM pretraining scale, what is the accuracy price of exact unlearning capability?

**Scale.** 1.4B-parameter decoder, 30B tokens (Chinchilla-ish), fixed architecture and tokenizer, tokenizer and hyperparameters fixed *a priori* from a held-out corpus so the pipeline is honestly exact.

**Arms.**
- **Control:** one monolithic model, 30B tokens.
- **Treatment:** $S \in \{4, 8, 16\}$ shards, $30/S$ B tokens each, $R = 20$ checkpointed slices per shard, logit-averaged ensemble at inference. Total training FLOPs matched to control.
- Deletion stream: $K = 3{,}000$ documents drawn to match observed GDPR-style request patterns (skewed to high-duplication documents).

**Deciding number.** $\Delta_{\text{acc}}$ measured as **validation perplexity gap between treatment ensemble and monolithic control at equal total FLOPs**, reported alongside amortized $\rho$ over the 3,000-request stream.

**Decision rule.** If some $S \ge 8$ gives perplexity within **2%** of control *and* amortized $\rho \ge 4$, exact unlearning is viable for pretraining and the field should stop funding heuristic forgetting. If the gap exceeds 10% at every $S$ with $\rho \ge 4$ — the outcome the ImageNet result predicts — exact unlearning at scale is dead and approximate-with-audit is the only route. Cost estimate: about 4 training runs, roughly $10^4$ H100-hours.

## 9. Key References

- **[Foundational]** Cao, Yang. *Towards Making Systems Forget with Machine Unlearning.* IEEE S&P, 2015.
- **[Foundational]** Ginart, Guan, Valiant, Zou. *Making AI Forget You: Data Deletion in Machine Learning.* NeurIPS, 2019.
- **[SOTA — exact]** Bourtoule, Chandrasekaran, Choquette-Choo, Jia, Travers, Zhang, Lie, Papernot. *Machine Unlearning.* IEEE S&P, 2021.
- **[Theory]** Guo, Goldstein, Hannun, van der Maaten. *Certified Data Removal from Machine Learning Models.* ICML, 2020.
- **[Theory]** Neel, Roth, Sharifi-Malvajerdi. *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning.* ALT, 2021.
- **[Theory]** Ullah, Mai, Rao, Rossi, Arora. *Machine Unlearning via Algorithmic Stability.* COLT, 2021.
- **[Theory]** Sekhari, Acharya, Kamath, Suresh. *Remember What You Want to Forget: Algorithms for Machine Unlearning.* NeurIPS, 2021.
- **[Theory]** Gupta, Jagielski, Ullman, Waites, Zhang. *Adaptive Machine Unlearning.* NeurIPS, 2021.
- **[Limitation]** Thudi, Jia, Shumailov, Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022.
- **[Evaluation]** Hayes, Shumailov, Triantafillou, Khalifa, Papernot. *Inexact Unlearning Needs More Careful Evaluation to Avoid a False Sense of Privacy.* 2024.
- **[Evaluation]** Carlini, Chien, Nasr, Song, Terzis, Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022.
- **[Context]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023.
- **[Benchmark]** Triantafillou et al. *Are We Making Progress in Unlearning? Findings from the First NeurIPS Unlearning Competition.* 2024.
- **[Survey]** Nguyen, Huynh, Nguyen, Liew, Yin, Nguyen. *A Survey of Machine Unlearning.* 2022.
- **[Scale reference]** Grattafiori et al. (Llama Team, AI @ Meta). *The Llama 3 Herd of Models.* 2024.

## 10. Worked Example

**Setup.** $n = 10^6$ documents, $S = 20$ shards, $R = 50$ slices per shard. Unit of cost: one pass over one slice $= n/(SR) = 1{,}000$ documents.

**Full training cost.** Slice $r$ of a shard is trained on top of $r-1$ predecessors, so a shard costs $\sum_{r=1}^{50} r = 1{,}275$ units; all 20 shards cost $25{,}500$ units.

**One deletion.** A point in slice $r$ forces retraining slices $r..R$: cost $\sum_{j=r}^{50} j$. Averaging over uniform $r$:

$$\mathbb{E}[\text{cost}] = \frac{(R+1)(2R+1)}{6} = \frac{51 \cdot 101}{6} = 858.5 \text{ units}.$$

Speedup vs. full retrain: $25{,}500 / 858.5 = 29.7\times$. This is the number the method is sold on.

**Now the request stream.** Take $K = 1{,}000$ deletions — a **0.1%** deletion rate, far below what any consumer service sees. Each shard receives 50 requests. The rebuild point is the minimum slice index among them:

$$\mathbb{E}\big[\min_{1..50} \mathrm{Unif}\{1..50\}\big] \approx \frac{51}{51} \approx 1 .$$

So essentially every shard rebuilds from slice 1: cost $= 20 \times 1{,}275 = 25{,}500$ units — a full retrain. Amortized $\rho = 1{,}000 \times 25{,}500 / 25{,}500 = 1{,}000$ against per-request retraining, but exactly $1.0\times$ against the honest baseline of *batching all deletions into one retrain*.

**The obstruction, visible.** SISA's $29.7\times$ is a single-request figure. At a 0.1% deletion rate it buys nothing that a quarterly batch retrain would not, while the practitioner has already paid $\Delta_{\text{acc}} = 16.7$ top-5 points (ImageNet, $S=20$) for the sharding. And no external party can check any of it: the released ensemble is consistent with a pipeline that never rebuilt at all (Thudi et al., 2022). The method is exact, cheap on paper, and — at deployment rates — neither faster than batch retraining nor auditable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*