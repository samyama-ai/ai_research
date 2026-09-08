---
id: 19-evaluation/compute-matched-test-time-scaling-eval
title: "Compute-Matched Comparison Protocols for Test-Time Scaling"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Matched Comparison Protocols for Test-Time Scaling

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/compute-matched-test-time-scaling-eval` · **Status:** open

## 1. Problem Statement

Test-time scaling (TTS) trades inference compute for accuracy: sampling $k$ solutions and voting, running a verifier-guided search, or emitting a long chain of thought before answering. Claims of the form "method A beats method B" are routinely made without holding inference compute fixed, so the reported gain may be a gain in budget rather than in method.

**Input.** A model family $\{M_\theta\}$, a decoding/search procedure $S$ with knobs (samples $k$, beam width, reasoning-token budget $b$), a benchmark $D$, and a compute budget $C$.

**Output.** A protocol that maps $(M, S, C) \mapsto$ a scalar accuracy estimate, plus an uncertainty interval, such that two systems reported at the same $C$ are actually comparable.

Three variants, of unequal difficulty:

- **Measurement.** What is the right budget unit — FLOPs, tokens, wall-clock latency, dollars, energy? These orderings disagree, and no unit is neutral. *Methodologically blocked.*
- **Method.** Given a unit, how do you allocate budget across problems (uniform vs. adaptive) and report the frontier rather than a point? *Empirically open.*
- **Theory.** Does a compute-optimal allocation exist and is it identifiable from a finite sample of a benchmark? *Theoretically open.*

Solved means: a published protocol under which independent groups, given the same model weights and budget, reproduce the same ordering of methods to within stated error bars — including when a training-compute-heavy baseline is one of the arms.

## 2. Formal Setting

Let $x \sim \mathcal{D}$ be a problem with ground-truth verifier $v(x,y) \in \{0,1\}$. A system is a pair $(M,S)$ producing $\hat y = S(M,x;\rho)$ with random seed $\rho$.

**Accuracy.** $\mathrm{Acc}(M,S) = \mathbb{E}_{x,\rho}[v(x,S(M,x;\rho))]$, estimated on $n$ items with $m$ seeds:
$$\widehat{\mathrm{Acc}} = \frac{1}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m} v(x_i, S(M,x_i;\rho_j)).$$
Measured variance must be decomposed into item variance and seed variance; the usual reported number collapses both and uses $m=1$.

**Budget.** For a dense decoder-only model with $N$ non-embedding parameters, generation cost is measured as
$$C_{\text{FLOP}}(x) \approx 2N\big(T_{\text{out}}(x) + T_{\text{in}}(x)\big) + \text{attention term } O(L\,T^2 d),$$
where $T_{\text{in}}, T_{\text{out}}$ are prompt and generated tokens, counted from the actual log, not from a nominal `max_tokens`. For mixture-of-experts, $N$ is the *active* parameter count — the standard place comparisons silently break. Verifier or reward-model calls add $2N_v T_v$ and are frequently omitted.

Alternative units, all of which have been used as "compute" in published TTS plots:
$$C_{\text{tok}} = \mathbb{E}[T_{\text{out}}], \quad C_{\text{lat}} = \mathbb{E}[\text{wall-clock}], \quad C_{\$} = \text{price} \times \text{tokens}, \quad C_{\text{tr}} = 6ND_{\text{train}}.$$
$C_{\text{lat}}$ and $C_{\text{FLOP}}$ diverge by roughly the batch size: 64 parallel samples cost $64\times$ the FLOPs and, with enough accelerator memory, close to $1\times$ the latency.

**Compute-optimal frontier.** For budget $C$,
$$A^\star(C) = \max_{(M,S)\,:\,\mathbb{E}[C(M,S)] \le C} \mathrm{Acc}(M,S).$$
A method claim is only meaningful as a claim about $A^\star$ at a stated $C$, or about the slope $dA^\star/d\log C$.

**Assumptions known to be violated in practice.**
1. $v$ is a perfect verifier — false for free-form math (string match), code (weak unit tests), and any LLM judge.
2. Per-item cost is independent of the item — false: hard items generate far more tokens, so a fixed nominal $k$ is not a fixed budget.
3. Benchmark items are i.i.d. draws from the deployment distribution — false, and contaminated items have near-zero cost of solution.
4. Training compute is a sunk constant shared by all arms — false whenever a TTS-trained model is compared against a base model.

## 3. State of the Art

**Established.** Snell et al. (2024, *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, arXiv:2408.03314) gave the first explicit FLOP-matched exchange between test-time and pretraining compute on MATH with PaLM 2-S\*, using a compute-optimal per-question allocation based on predicted difficulty. Their qualification is the load-bearing result: test-time compute wins on easy/medium questions at small budgets and loses to a $14\times$ larger model on hard questions or at large budgets. Wu et al. (2024, arXiv:2408.00724) independently derived inference-scaling curves and found simple weighted majority voting competitive with far more complex search.

Brown et al. (2024, *Large Language Monkeys*, arXiv:2407.21787) established coverage scaling: pass@$k$ grows roughly log-linearly in $k$ over four orders of magnitude, so any budget-unmatched pass@$k$ comparison is nearly guaranteed to favor the larger $k$.

**Claimed but unablated.** Most frontier reasoning-model reports ("more thinking tokens → higher accuracy") are benchmark numbers on a single axis of reasoning tokens, without an equal-FLOPs baseline of a larger non-reasoning model or of parallel sampling. OpenAI's o1 blog (2024) and DeepSeek-R1 (Nature, 2025) both plot accuracy against test-time tokens; neither reports the parallel-sampling control at matched FLOPs across the full curve.

**Benchmark-number-only.** Leaderboard entries for "agentic" scaffolds (SWE-bench, ARC-AGI) largely do not publish token counts, so no post-hoc compute matching is possible. ARC Prize's 2024–2025 practice of publishing a cost-per-task axis is the notable exception and should be treated as the current best reporting standard.

## 4. What Is Known

- **Coverage vs. selection gap.** Brown et al. (2024): on SWE-bench Lite, DeepSeek-Coder-V2-Instruct coverage rises from 15.9% (1 sample) to 56% (250 samples), while the best available selector recovers only ~43% — the verifier, not the sampler, is the binding constraint at scale.
- **Verifier imperfection caps the frontier.** Stroebl et al. (2024, arXiv:2411.17501) show that with a false-positive-rate-$\epsilon$ verifier, resampling accuracy is non-monotone in $k$ and plateaus below the perfect-verifier curve; more compute can *reduce* accuracy.
- **Seed variance dominates small benchmarks.** Hochlehnert et al. (2025, *A Sober Look at Progress in Language Model Reasoning*, arXiv:2504.07086) report that on AIME'24 ($n=30$) single-seed differences of several points across runs are common, and that many claimed reasoning gains vanish under standardized decoding and multi-seed evaluation. At $n=30$, one item is 3.3 accuracy points.
- **Budget-shifted equivalence.** Muennighoff et al. (2025, *s1: Simple test-time scaling*, arXiv:2501.19393) show budget forcing on a 32B model produces monotone gains over a controlled token range, then saturates — the saturation point, not the slope, is what a matched comparison needs.
- **Training/inference trade is quantifiable.** Sardana et al. (ICML 2024, arXiv:2401.00448) extend Chinchilla to amortize inference demand: at high query volume the optimal model is smaller and trained longer than Chinchilla-optimal. This is the theory-side statement of the same exchange rate.
- **RL-trained reasoners may not expand coverage.** Yue et al. (2025, arXiv:2504.13837) find RLVR models beat base models at low $k$ but base models match or exceed them at large $k$ on pass@$k$ — an ordering that reverses with budget.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted budget unit. FLOPs ignore memory bandwidth and KV-cache traffic that dominate real decoding; latency depends on batch size and hardware; dollars encode vendor margin. No published normalization makes a serial-reasoning arm and a parallel-sampling arm commensurable across all three at once.
- **Empirically open.** Whether the Snell et al. crossover (TTS wins at small budgets, parameters win at large ones) holds at 2026 frontier scale, with reasoning-trained models and $10^{18}$–$10^{20}$ inference FLOPs per problem. The experiment is runnable by any lab with two model sizes in the same family; it has not been published at that scale.
- **Empirically open.** Whether adaptive per-item allocation beats uniform allocation once the difficulty predictor's own compute is charged to the budget.
- **Theoretically open.** Whether $A^\star(C)$ is estimable from a finite benchmark: the argmax over $(M,S)$ is taken on the same data used to report accuracy, so the frontier is an optimistically biased order statistic, and no correction analogous to selective-inference bounds has been derived for it.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the budget axis combined with selection bias on the frontier**, not cost.

Two systems can be matched on FLOPs and differ by $50\times$ in latency, or matched on latency and differ by $60\times$ in FLOPs. Since accuracy is monotone in compute for both arms, the *choice of unit selects the winner*. There is no unit-free statement available: "method A dominates method B" is only well posed once a unit is fixed, and every candidate unit is defensible.

Second, the frontier $A^\star(C)$ is reported as the max over a family of configurations evaluated on the same benchmark. With $n=30$ (AIME) or $n=500$ (MATH-500) and dozens of configurations, the max is biased upward by several points before any real effect. This is the same failure as tuning a threshold on the test set, but it is invisible because the tuning is presented as "compute-optimal allocation".

Third, ground truth for the selector is absent: verifier false positives mean the measured accuracy of a best-of-$k$ arm is a function of the verifier's error rate as much as the model's, and verifier compute is usually uncounted.

## 7. Current Research (as of 2026)

- **Frontier reporting standards.** ARC Prize publishes cost-per-task alongside accuracy; HELM and Epoch AI have pushed token-count and FLOP disclosure. Adoption outside these is partial. *(frontier — verify)*
- **Statistical rigor for evals.** Miller (2024, *Adding Error Bars to Evals*, arXiv:2411.00640) supplies the CLT/clustered-standard-error machinery; the open work is extending it to a max-over-configurations frontier.
- **Inference-scaling measurement.** Balachandran et al. (Microsoft, 2025, *Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead*, arXiv:2504.00294) report token-cost distributions and show high variance in tokens spent per problem even at equal accuracy.
- **Reproducibility audits.** Hochlehnert et al. (2025) and follow-on work on AIME/MATH variance are the main empirical pressure toward multi-seed, matched-decoding protocols. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** At matched inference FLOPs, does serial reasoning-token scaling beat parallel sampling with a verifier, and does the ordering survive a change of budget unit?

**Scale.** One open model family with two sizes sharing a tokenizer and training recipe (e.g. 8B and 70B class). Benchmarks: MATH-500 ($n=500$) and AIME'24+'25 ($n=60$), plus a held-out contamination-controlled set. Budget grid: $C \in \{10^{14}, 10^{15}, 10^{16}, 10^{17}\}$ FLOPs per problem, four points, log-spaced. $m=8$ seeds per cell. Total ≈ $2\times4\times8\times560 \approx 36$k problem-solves — a few thousand GPU-hours.

**Arms.**
1. Serial: 8B reasoning model, budget forcing to hit target FLOPs.
2. Parallel: 8B, best-of-$k$ with an explicitly FLOP-charged verifier.
3. **Control arm:** 70B, greedy single sample, replicated to the same per-problem FLOPs by drawing $k' = C/(2N_{70}T)$ samples with majority vote — i.e. parameters instead of search.

**Deciding number.** The FLOP budget $C^\times$ at which arm 3 crosses the best of arms 1–2, with a bootstrap CI over items and seeds. Report $C^\times$ three times, once under FLOPs, once under wall-clock at fixed batch 32, once under list price. **If the three $C^\times$ values agree within one grid decade, compute matching is well posed and FLOPs can be standardized. If they disagree by more than a decade, the measurement variant is confirmed blocked and reporting must be a 2-D (accuracy, cost) frontier under all three units.** Prediction: they disagree by $\ge 1.5$ decades.

## 9. Key References

- **[Foundational]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024 — arXiv:2408.03314
- **[Foundational]** Bradley Brown, Jordan Juravsky, Ryan Ehrlich, et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024 — arXiv:2407.21787
- **[SOTA]** Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang. *Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models.* 2024 — arXiv:2408.00724
- **[SOTA]** Niklas Muennighoff, Zitong Yang, Weijia Shi, et al. *s1: Simple test-time scaling.* 2025 — arXiv:2501.19393
- **[Critique]** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan. *Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024 — arXiv:2411.17501
- **[Critique]** Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, et al. *A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility.* 2025 — arXiv:2504.07086
- **[Critique]** Yang Yue, Zhiqi Chen, Rui Lu, et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025 — arXiv:2504.13837
- **[Theory]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024 — arXiv:2401.00448
- **[Method]** Evan Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024 — arXiv:2411.00640
- **[Survey]** Vidhisha Balachandran, Jingya Chen, Lingjiao Chen, et al. *Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead.* Microsoft Research, 2025 — arXiv:2504.00294
- **[Foundational]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022 — arXiv:2203.15556

## 10. Worked Example

Take an 8B model with majority-vote@64 against a 70B model with greedy decoding, 1{,}000 output tokens each, prompt cost ignored.

| Arm | FLOPs/problem | Relative |
|---|---|---|
| 8B, $k=64$ | $2 \times 8\!\times\!10^9 \times 64\,000 = 1.02\times10^{15}$ | $7.3\times$ |
| 70B, $k=1$ | $2 \times 7\!\times\!10^{10} \times 1\,000 = 1.4\times10^{14}$ | $1\times$ |
| 70B, $k=7$ (matched) | $9.8\times10^{14}$ | $7.0\times$ |

The published comparison is almost always the first two rows: "8B with voting matches 70B". The compute-matched comparison is row 1 against row 3, and row 3 is a strictly stronger baseline than row 2 — majority-vote@7 on MATH-class benchmarks typically adds 3–6 points over greedy. A reported 4-point win for the 8B arm can invert.

Now change the unit. Batch the 64 samples on one node: wall-clock is roughly one forward pass, so under $C_{\text{lat}}$ the 8B arm costs about $8\times$ *less* than the 70B arm rather than $7.3\times$ more — a $60\times$ swing in the exchange rate from a unit change alone. Under list pricing, typical per-token prices differ by roughly $5$–$10\times$ between an 8B and a 70B endpoint, so $C_\$$ puts the 8B@64 arm at $6$–$13\times$ the 70B single sample, close to FLOPs but not identical.

Finally, add the verifier. If best-of-64 uses a 7B reward model scoring 64 candidates of 1{,}000 tokens, that is $2\times7\!\times\!10^9\times64\,000 = 9.0\times10^{14}$ FLOPs — nearly doubling the arm's cost, and it is omitted from essentially every published TTS plot. Charge it, and the matched 70B arm gets $k'=14$ instead of $7$.

The obstruction is visible in one table: three defensible budget units place the same two systems on opposite sides of the frontier, and the largest single term in one arm's cost is routinely not counted at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*