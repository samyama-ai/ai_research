---
id: 11-inference-and-serving/output-length-prediction-before-generation
title: "Predicting Output Length Before Generation"
topic: 11-inference-and-serving
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Predicting Output Length Before Generation

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/output-length-prediction-before-generation` · **Status:** empirically-open

## 1. Problem Statement

Given a prompt and a decoding configuration, predict how many tokens the model will emit — **before** emitting any of them. The prediction feeds scheduling (shortest-job-first), KV-cache admission, batch composition, and SLO-aware routing.

Three variants, routinely conflated:

- **Measurement.** How much of the variance in output length is a function of the prompt at all? Output length is a random variable over sampling noise, not a fixed label. Nobody publishes the decomposition, so the ceiling on any predictor is unknown.
- **Method.** Build $\hat{L}(x)$ minimising a loss that the scheduler actually cares about. Fu et al. (NeurIPS 2024) showed the relevant loss is a *ranking* loss, not squared error — a large accuracy gain that came from changing the objective, not the model.
- **Theory.** Given a predictor with known error distribution, what is the competitive ratio of the induced scheduler against clairvoyant SRPT? Partly answered by the learning-augmented-algorithms literature; not instantiated for LLM serving's specific cost structure (preemption costs a KV-cache eviction, not zero).

Solved would mean: a predictor whose induced scheduler closes ≥80% of the gap between FCFS and an oracle-length scheduler on a real production trace, with the residual attributable to a *measured* aleatoric floor rather than to model capacity.

## 2. Formal Setting

Let $x$ be a prompt, $\pi_\theta$ the model, $c = (T, p, \text{max\_tokens}, \text{stop})$ the decoding config, and $\xi$ the sampling seed. The realised length is
$$L(x,\xi,c) = \min\big(\inf\{t : y_t = \texttt{EOS}\},\ \text{max\_tokens}\big),$$
measured as the count of tokens the server bills — including reasoning traces, excluding the prompt.

**Predictor.** $\hat{L}: x \mapsto \mathbb{R}_+$ or a distribution $\hat{F}(\cdot \mid x)$. Cost constraint: prediction latency $\tau_{\text{pred}} \ll$ TTFT (time to first token), typically $\tau_{\text{pred}} < 5$ ms, measured on the scheduler's critical path, not offline.

**Variance decomposition** — the quantity that bounds everything:
$$\underbrace{\mathrm{Var}(L)}_{\text{total}} = \underbrace{\mathbb{E}_x\big[\mathrm{Var}(L \mid x)\big]}_{\text{aleatoric, irreducible}} + \underbrace{\mathrm{Var}_x\big(\mathbb{E}[L \mid x]\big)}_{\text{predictable}}.$$
Define the **prediction ceiling**
$$\rho^2 \;=\; 1 - \frac{\mathbb{E}_x[\mathrm{Var}(L\mid x)]}{\mathrm{Var}(L)},$$
estimated by drawing $k$ independent generations per prompt over $N$ prompts and using the one-way ANOVA estimator; $k \ge 16$ is needed for the within-prompt term to be stable at typical dispersion.

**Scheduling objective.** For jobs $j$ with arrival $a_j$ and service time $s_j \propto L_j$, mean job completion time $\mathrm{JCT} = \frac{1}{n}\sum_j (f_j - a_j)$. Clairvoyant SRPT minimises this on a single server (Schrage, 1968). With predictions $\hat{s}_j$, the useful ranking metric is Kendall's $\tau$ over the queue, not $\mathbb{E}|\hat L - L|$.

**Assumptions, and where they break.**

| Assumption | Status in practice |
|---|---|
| $L$ i.i.d. given $x$ | Violated: prefix caching and batch-dependent numerics make generations weakly correlated. |
| $L$ observed uncleaved | Violated: `max_tokens` right-censors. Fitting uncensored regression to capped data biases $\hat L$ downward. |
| Service time $\propto L$ | Violated under chunked prefill and continuous batching; per-token cost depends on concurrent batch composition. |
| Stationary workload | Violated: prompt mix shifts hourly; a predictor trained last week decays. |
| Fixed $c$ | Violated: reasoning models expose a thinking budget that clients vary per request. |

## 3. State of the Art

**Systems/empirical.**
- *Sequence Scheduling* (Zheng et al., NeurIPS 2023): prompt the LLM itself to perceive its own response length, bucket similar-length requests. Reported up to 86% throughput improvement on Vicuna-style workloads. Established: the self-perception signal is non-trivial. Unablated: how much of the gain is length prediction versus simply batching by bucket.
- *$S^3$* (Jin et al., NeurIPS 2023): a fine-tuned DistilBERT predicts an output-length bucket, used to size KV allocations; reported up to $6.49\times$ throughput over a baseline that reserves worst-case memory. This is a benchmark number against a weak memory baseline — PagedAttention (Kwon et al., SOSP 2023) removes most of the same waste *without* any prediction, so the two gains are not additive.
- *Learning to Rank* (Fu et al., NeurIPS 2024): trains a ranker rather than a regressor; reported $2.8\times$ lower mean latency in chatbot serving and $6.5\times$ higher throughput in synthetic data generation. The ablation that matters — ranking loss versus regression loss with the same backbone — is present, and this is the strongest established result on the page.
- *Proxy-model prediction* (Qiu et al., 2024): a small BERT-class proxy predicts length percentiles for interactive serving; reports large JCT reductions. Benchmark-only; no independent reproduction.
- *TetriInfer* (Hu et al., 2024) and slice-level scheduling (Cheng et al., 2024) both add length predictors to disaggregated serving stacks.

**Theory.** Purohit, Svitkina & Kumar (NeurIPS 2018) give consistency/robustness trade-offs for non-clairvoyant scheduling with a predicted job length; Mitzenmacher (ITCS 2020) analyses the price of misprediction for SRPT-with-predictions. Non-clairvoyant round-robin is $(2-2/(n+1))$-competitive for mean flow time (Motwani, Phillips & Torng, 1994) — the baseline any predictor must beat.

## 4. What Is Known

- **Ranking beats regression.** Same backbone, ranking objective, $2.8\times$ latency reduction versus regression-trained scheduling (Fu et al., 2024; chatbot trace, 70B-class serving).
- **Small encoders carry most of the signal.** DistilBERT-scale (66M) predictors reach bucket accuracies within a few points of much larger predictors ($S^3$, 2023) — the task is not compute-bound on the predictor side.
- **Prediction is unnecessary for memory safety.** PagedAttention (SOSP 2023) achieved 2–4× throughput over prior systems with *zero* length prediction, by paging and preemption. Length prediction's remaining value is scheduling order, not allocation.
- **SRPT is optimal when lengths are known** (Schrage, 1968), so an oracle bound is computable on any trace: replay with true lengths.
- **Reasoning models widen the target.** Output-length distributions for chain-of-thought models are heavy-tailed and multi-modal over difficulty; length variation across seeds for a single hard prompt spans an order of magnitude in reported traces *(frontier — verify at scale)*.

## 5. What Is Not Known

- **Methodologically blocked:** $\rho^2$, the prediction ceiling. No published paper reports the within-prompt variance of $L$ at production temperature over $k\ge16$ resamples. Every reported MAE/accuracy number is therefore uninterpretable — it cannot be compared to the achievable maximum.
- **Empirically open:** whether predictor gains survive on top of a *modern* baseline (continuous batching + paged KV + chunked prefill). Most reported speedups use baselines predating one or more of these.
- **Empirically open:** whether prediction transfers across model families, or whether each served model needs its own predictor retrained on its own traces.
- **Theoretically open:** the competitive ratio for SRPT-with-predictions when preemption has a non-zero, state-dependent cost (KV eviction and recompute). The existing learning-augmented results assume free preemption.
- **Empirically open:** censoring. No published predictor explicitly models the `max_tokens` cap as right-censoring, and no ablation quantifies the resulting bias.

## 6. Why It Is Hard

The core obstruction is **absent ground truth combined with confounded evaluation**. $L$ is not a label; it is one draw from $P(L \mid x)$. Training on a single draw per prompt means the target itself carries aleatoric noise of unknown magnitude, so a predictor that has perfectly learned $\mathbb{E}[L\mid x]$ still shows large MAE and looks like it failed.

Second, the evaluation does not measure what it names. Papers report throughput gains from a pipeline in which length prediction is one of several changes — bucketing, memory sizing, batching policy — and PagedAttention independently removes the memory-waste term that motivated the earliest predictors. A throughput number is not evidence that the predictor is good.

Third, non-identifiability of the utility function: under continuous batching, a job's service time depends on which other jobs are co-resident, so "shortest job" is not well defined independent of the schedule the prediction produces.

## 7. Current Research (as of 2026)

- Ranking-based and embedding-based schedulers: Fu et al. (UCSD/Berkeley line of work); Shahout & Mitzenmacher (Harvard) on embedding-based scheduling and two-stage predict-then-schedule designs.
- Length-controlled decoding as a substitute: if the model can be *instructed* to a length budget reliably, prediction becomes unnecessary (length-following instruction work, Meta AI, 2024). *(frontier — verify)*
- Thinking-budget prediction for reasoning models — routing hard queries to longer budgets. Active in industrial serving stacks; little public ablation. *(frontier — verify)*
- Learning-augmented scheduling theory with imperfect and adversarial predictions (ongoing at ITCS/SODA venues).

## 8. Concrete Next Experiment

**Measure the ceiling, then measure the gap to it.**

- **Scale.** $N = 50{,}000$ prompts sampled from a real conversational trace (LMSYS-Chat-1M) plus 10,000 from a reasoning benchmark mix. For each prompt, $k = 32$ independent generations at production temperature ($T=0.7$, `max_tokens` set to 4× the observed p99 so censoring is negligible). Two models: one 8B instruct, one reasoning model. Cost: $\approx 2\times10^9$ generated tokens — roughly one 8×H100 node-week per model.
- **Control arm.** Replay the trace through vLLM with continuous batching + paged KV + chunked prefill, under three policies: (a) FCFS, (b) oracle SRPT using the *realised* length, (c) the best published predictor (ranking-based).
- **Deciding number.** $\rho^2 = 1 - \mathbb{E}_x[\mathrm{Var}(L\mid x)]/\mathrm{Var}(L)$, with a bootstrap CI. If $\rho^2 < 0.5$ for the reasoning model, prompt-only length prediction is capped near uselessness there and effort should move to length *control*. If $\rho^2 > 0.85$ and policy (c) recovers less than half of the (a)→(b) JCT gap, the problem is a method gap and remains worth attacking.

Both branches are decisive, and the measurement has never been published.

## 9. Key References

- **[Foundational]** L. Schrage. *A Proof of the Optimality of the Shortest Remaining Processing Time Discipline.* Operations Research, 1968.
- **[Foundational]** R. Motwani, S. Phillips, E. Torng. *Non-Clairvoyant Scheduling.* Theoretical Computer Science, 1994.
- **[Foundational]** M. Purohit, Z. Svitkina, R. Kumar. *Improving Online Algorithms via ML Predictions.* NeurIPS, 2018.
- **[Theory]** M. Mitzenmacher. *Scheduling with Predictions and the Price of Misprediction.* ITCS, 2020. — arXiv:1902.00732
- **[SOTA]** Y. Fu, S. Zhu, R. Su, A. Qiao, I. Stoica, H. Zhang. *Efficient LLM Scheduling by Learning to Rank.* NeurIPS, 2024. — arXiv:2408.15792
- **[SOTA]** Z. Zheng, X. Ren, F. Xue, Y. Luo, X. Jiang, Y. You. *Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline.* NeurIPS, 2023. — arXiv:2305.13144
- **[SOTA]** Y. Jin, C.-F. Wu, D. Brooks, G.-Y. Wei. *$S^3$: Increasing GPU Utilization during Generative Inference for Higher Throughput.* NeurIPS, 2023. — arXiv:2306.06000
- **[Systems]** W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, I. Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Systems]** G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, B.-G. Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[Systems]** A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, A. Tumanov, R. Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[Related]** R. Shahout, E. Malach, C. Liu, W. Jiang, M. Yu, M. Mitzenmacher. *Don't Stop Me Now: Embedding Based Scheduling for LLMs.* 2024. — arXiv:2410.01035

## 10. Worked Example

Two prompt classes, equal mass: **short** ($L=100$) and **long** ($L=900$). Both arrive at $t=0$; decode at 100 tok/s, so service times are 1 s and 9 s. Single server, mean JCT.

- Correct order (short first): completions at 1 s and 10 s → **5.5 s**.
- Inverted: 9 s and 10 s → **9.5 s**.
- Random order: **7.5 s**. Oracle: **5.5 s**.

A predictor with 80% bucket accuracy gives $0.8(5.5) + 0.2(9.5) = 6.3$ s, recovering $(7.5-6.3)/(7.5-5.5) = 60\%$ of the oracle gain. This is the arithmetic behind every reported speedup.

Now make the obstruction visible. Resample the *same* long prompt 32 times and suppose realised lengths span 400–1400 tokens (a coefficient of variation of $\approx 0.35$, plausible for open-ended generation). Then for a fraction of draws the "long" job is genuinely shorter than a "short" job drawn at its upper tail, and the label the predictor was trained against was itself an inversion. Decompose: if $\mathbb{E}_x[\mathrm{Var}(L\mid x)] = 0.35^2 \cdot \mathbb{E}[L]^2$ and $\mathrm{Var}(L) = 160{,}000$ (from the 100/900 split), the ceiling is $\rho^2 \approx 1 - 30{,}625/160{,}000 = 0.81$.

So the reachable bucket accuracy is not 100% — it is whatever 0.81 explained variance permits, and the 80%-accurate predictor may already be at the ceiling. Without measuring $\rho^2$, a paper reporting 80% accuracy cannot tell whether it has solved the problem or barely started. That is the gap Section 8 closes.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*