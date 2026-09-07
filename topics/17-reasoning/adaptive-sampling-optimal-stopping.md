---
id: 17-reasoning/adaptive-sampling-optimal-stopping
title: "Optimal Stopping Rule for Adaptive Sampling"
topic: 17-reasoning
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Stopping Rule for Adaptive Sampling

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/adaptive-sampling-optimal-stopping` · **Status:** open

## 1. Problem Statement

A model answers a query by drawing samples one at a time and aggregating them (majority vote, verifier score, best-of-$n$). Each sample costs tokens. **When should it stop?**

- **Input:** a prompt $x$, a sampler $\pi(\cdot\mid x)$, an aggregator $A$, a per-sample cost $c$, and a utility for a correct final answer.
- **Output:** a stopping time $\tau$ (measurable w.r.t. the samples seen so far) and a final answer $\hat{y}_\tau = A(y_{1:\tau})$.
- **Objective:** minimise $\mathbb{E}[\,\ell(\hat y_\tau, y^\star) + c\,\tau\,]$ over the query distribution, not per query.

Three variants that are routinely conflated:

- **Measurement variant.** Given logged rollouts, can we *score* a stopping rule offline without re-sampling? Currently only approximately.
- **Method variant.** Build a rule that beats fixed-$n$ on the accuracy-vs-cost Pareto frontier. Solved in the weak sense (several rules do), unsolved in the strong sense (no rule is known to be near-optimal).
- **Theory variant.** Characterise the Bayes-optimal $\tau$ for majority-vote aggregation with unknown answer distribution, and bound the regret of any practical rule against it. Open.

Solving it means: a rule with a proven regret bound against the oracle stopping time, whose assumptions hold for real LLM samplers, and which dominates fixed-$n$ at every budget on held-out tasks.

## 2. Formal Setting

Fix a query $x$. Samples $y_1, y_2, \dots \overset{\text{iid}}{\sim} \pi(\cdot \mid x)$ take values in a countable answer space $\mathcal{Y}$ after answer extraction. Let

$$p_a = \Pr_{y\sim\pi}[\,y = a \mid x\,], \qquad a^\star = \arg\max_a p_a, \qquad p_{(1)} \ge p_{(2)} \ge \cdots$$

**Measured as:** $p_a$ is estimated by the empirical frequency $\hat p_a^{(n)} = n_a/n$ after string-normalising extracted answers (numeric canonicalisation, `\boxed{}` parsing). The normaliser is part of the definition — two rules using different normalisers are not comparable.

Aggregator (self-consistency): $A(y_{1:n}) = \arg\max_a n_a$. Loss $\ell = \mathbb{1}[\hat y \ne y^\star]$ with $y^\star$ the gold answer.

Stopping time $\tau$ adapted to $\mathcal{F}_n = \sigma(y_{1:n})$, with $\tau \le N_{\max}$. The oracle is

$$\tau^\star \in \arg\min_{\tau} \; \mathbb{E}\!\left[\mathbb{1}[A(y_{1:\tau}) \ne y^\star] + c\,\tau\right],$$

and **regret** of a rule is $R(\tau) = \mathbb{E}[\ell + c\tau] - \mathbb{E}[\ell^\star + c\tau^\star]$, averaged over a query distribution $\mathcal{D}$.

Two distinct targets, often confused:

$$\underbrace{\Pr[\hat y_\tau = a^\star]}_{\text{stability: agrees with }n\to\infty\text{ mode}} \quad \text{vs.} \quad \underbrace{\Pr[\hat y_\tau = y^\star]}_{\text{correctness}}.$$

They coincide only under the **mode-correctness assumption** $a^\star = y^\star$, which is *known to be violated*: on hard MATH/GSM8K items, models place majority mass on a specific wrong answer. Empirically self-consistency saturates well below pass@$k$ coverage, which is the direct evidence of $a^\star \ne y^\star$ on a nontrivial fraction of items.

Other assumptions and their status:
- **i.i.d. samples** — approximately true at fixed temperature and prompt; violated under sequential-revision or KV-cache-shared decoding, and under any scheme conditioning sample $i$ on $y_{1:i-1}$.
- **Fixed cost per sample** — violated: reasoning traces have heavy-tailed lengths, so $c\tau$ should be $\sum_{i\le\tau} c\,|y_i|$ (tokens), and long traces correlate with hard items.
- **Per-query budget separability** — the real constraint is a batch budget $\mathbb{E}_\mathcal{D}[\tau] \le B$; the Lagrangian $c$ is the dual variable and is not known a priori.

## 3. State of the Art

**Established (ablated, reproduced):**
- **Self-consistency** (Wang et al., ICLR 2023) with fixed $n$ is the baseline every adaptive rule is measured against.
- **Adaptive-Consistency** (Aggarwal et al., EMNLP 2023): stop when a Dirichlet/Beta posterior on the answer counts puts mass $>\eta$ on the current mode being the population mode. Reported ~$3.3\times$ fewer samples with average accuracy drop $\approx 0.1$ points across 17 datasets and several models — with ablations over the stopping prior.
- **Early-Stopping Self-Consistency (ESC)** (Li et al., ICLR 2024): sample in windows; stop when a window is unanimous. Reported cost cuts of roughly 30–70% at matched accuracy on GSM8K/MATH; the unanimity criterion is simple enough that the ablation is convincing.

**Claimed but under-ablated:**
- **Compute-optimal test-time scaling** (Snell et al., 2024): allocating budget by difficulty bin gives up to ~$4\times$ efficiency over best-of-$n$. The difficulty bins use oracle or model-estimated difficulty; the oracle-free arm is weaker and the gap is not fully isolated.
- **Confidence-triggered stopping** — Certaindex/Dynasor (Fu et al., 2024), self-assessment mid-generation (Manvi et al., 2024), learned allocators (Damani et al., ICLR 2025). Each reports Pareto improvements on 2–4 benchmarks; none reports regret against an oracle stopping time, and cross-paper comparison is blocked by different answer normalisers and different cost units (samples vs. tokens).

**Theory SOTA, from sequential analysis rather than LLM work:** Wald's SPRT (1945) is optimal for a two-point hypothesis; Chow–Robbins–Siegmund (1971) gives the general optimal-stopping backward-induction characterisation; anytime-valid confidence sequences (Howard, Ramdas, McAuliffe, Sekhon, 2021) give time-uniform bounds on $\hat p_a$; best-arm identification lower bounds (Kaufmann, Cappé, Garivier, JMLR 2016) give $\Omega(\sum_a \Delta_a^{-2}\log(1/\delta))$ sample complexity for identifying $a^\star$. **None of these targets correctness**; they target the mode.

## 4. What Is Known

- Repeated sampling raises coverage steeply: Brown et al. (2024) report Llama-3-8B-Instruct MATH coverage (pass@$k$) rising from ~15.9% at $k=1$ to ~56% at $k=250$; on SWE-bench Lite, coverage rises from 15.9% to 56% with 250 samples. The gap between coverage and majority-vote accuracy is the headroom any stopping rule cannot recover by voting alone.
- Self-consistency accuracy saturates by roughly $n \approx 40$ on GSM8K for strong models (Wang et al., 2023, at 540B PaLM scale and replicated on smaller open models); gains beyond $n=40$ are within noise on 1319 test items ($\pm$~1.3 points at 1 s.e.).
- Verifier-guided selection beats majority vote when a trained verifier exists (Cobbe et al., 2021, GSM8K, 6B/175B GPT-3 class), which changes the aggregator and therefore the optimal $\tau$.
- Adaptive rules give a real 2–4$\times$ mean-sample reduction at $\le 0.5$ point accuracy cost, measured on GSM8K (1319 items), MATH (5000 or 500-item subsets), and coding benchmarks (Aggarwal et al. 2023; Li et al. 2024).
- Beta/Dirichlet stopping certifies **stability**, and the certificate is well calibrated for that event: with counts $(n_A, n_B)$ the posterior $\Pr[p_A > 1/2]$ under $\mathrm{Beta}(1+n_A, 1+n_B)$ is accurate to within Monte-Carlo error.

## 5. What Is Not Known

- **Theoretically open.** No characterisation of the Bayes-optimal $\tau$ for majority vote over an unknown, unbounded-support $\mathcal{Y}$ with a per-token cost, and no regret bound for any deployed rule against it. The special case $|\mathcal{Y}|=2$, known prior, uniform cost reduces to SPRT and is solved; nothing beyond it is.
- **Empirically open.** Whether adaptive stopping still beats fixed-$n$ **at matched token budget** (not matched sample count) on long-CoT models where trace lengths vary $10\times$. Runnable today; not run at scale in a way that controls for length.
- **Empirically open.** Whether the difficulty signal used for allocation transfers across model families, or is re-fit per model.
- **Methodologically blocked.** Scoring a stopping rule offline. A rule that stops at $\tau$ needs the counterfactual continuation; reusing a fixed pool of $N_{\max}$ logged samples without replacement is only valid if the rule never inspects sample content in a way that correlates with pool composition — violated by any confidence-based rule that reads the trace.
- **Methodologically blocked.** "Confidence" has no agreed operationalisation: token logprob, answer-frequency, self-reported score, and verifier score are all called confidence and are not monotone transforms of each other.

## 6. Why It Is Hard

The central obstruction is **an evaluation that does not measure the thing it names**, compounded by **absent ground truth at stopping time**.

Every practical rule certifies $\Pr[\hat y_\tau = a^\star]$ — agreement with the infinite-sample mode. The quantity that matters is $\Pr[\hat y_\tau = y^\star]$. On the items where stopping is actually consequential — hard items, where the model is drawn to a plausible wrong answer — these two diverge maximally and in the harmful direction: high sampling agreement is *evidence of a confident error*, so the rule stops earliest exactly where continuing would help. The stopping rule sees only $y_{1:n}$, and $y_{1:n}$ contains no information about $y^\star$ beyond what $\pi$ contains; a stopping rule cannot manufacture the correctness signal it needs.

Secondary obstructions: (i) the dual variable $c$ is a deployment choice, so Pareto curves, not single points, must be compared, and papers report different slices; (ii) heavy-tailed trace lengths make sample count a misleading cost unit; (iii) counterfactual replay makes offline evaluation biased, so each new rule needs fresh generation — $N_{\max}=250$ samples $\times$ 5000 MATH items $\times$ a 32B reasoning model is on the order of $10^9$ output tokens per arm.

## 7. Current Research (as of 2026)

- **Confidence-sequence stopping.** Porting anytime-valid e-process machinery (Ramdas, Howard, and collaborators) to answer-frequency streams, giving time-uniform stability guarantees without a Bayesian prior. Statistically mature; adoption in LLM serving is partial *(frontier — verify)*.
- **Learned allocators.** Predicting per-query marginal value of one more sample and thresholding it (Damani et al., ICLR 2025 line; MIT/CMU groups). Closest thing to attacking the correctness target directly, because the predictor is trained on correctness labels.
- **Serving-system integration.** Certaindex-style schedulers that treat stopping as an admission-control problem across a batch, optimising the batch budget rather than per query (UCSD/Berkeley systems groups).
- **Budget forcing in long-CoT models.** s1 (Muennighoff et al., 2025) shows stopping can be imposed *inside* a single trace by appending "Final Answer" or "Wait", which changes the problem: the unit is a thinking token, not a sample, and i.i.d. fails outright.
- **Verifier-in-the-loop stopping.** Stop when a process/outcome reward model exceeds a threshold; ties the stopping rule's quality to verifier calibration and shifts the open problem onto the verifier *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does any adaptive stopping rule beat fixed-$n$ at matched *token* budget, once the certificate targets correctness rather than stability?

- **Scale.** MATH-500 and GSM8K (1319 items), two models: an 8B instruct model and a 32B long-CoT reasoning model. Generate $N_{\max}=64$ full traces per item once, logging per-trace token counts. ~$1.5\times10^8$ output tokens total — a few hundred GPU-hours on 8×H100.
- **Arms.** (1) Fixed-$n$ self-consistency swept over $n \in \{1,2,4,8,16,32,64\}$ — **the control**. (2) Adaptive-Consistency (Beta, $\eta \in \{0.7,\dots,0.99\}$). (3) ESC window unanimity ($w=4$). (4) A **correctness-calibrated rule**: a logistic model over $(n, \hat p_{(1)}, \hat p_{(2)}, \text{mean trace length})$ trained on a disjoint item split to predict $\Pr[\hat y_n = y^\star]$, stopping when predicted correctness gain $< c$. (5) **Oracle stopping** — stop at the smallest $n$ whose majority is correct, else $N_{\max}$ — giving the empirical lower envelope.
- **Replay validity.** Rules must consume traces in logged order without inspecting unconsumed traces; report the fraction of items where $\tau = N_{\max}$ so truncation bias is visible.
- **Deciding number.** Plot accuracy against **mean output tokens per item**. The decision statistic is the *area between* each adaptive curve and the fixed-$n$ curve over $[10^3, 10^5]$ tokens/item, in accuracy-points. A rule wins only if that area is $> 0$ by more than 2 standard errors (bootstrap over items, 1000 resamples). Secondary number: regret against arm (5), in accuracy-points at matched tokens. Prediction, stated in advance: arms (2) and (3) win on *samples* and lose or tie on *tokens* for the 32B long-CoT model, because early-stopped items are the short-trace easy ones.

## 9. Key References

- **[Foundational]** Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. — arXiv:2203.11171
- **[Foundational]** Cobbe, Kosaraju, Bavarian, et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[Foundational]** Chow, Robbins, Siegmund. *Great Expectations: The Theory of Optimal Stopping.* Houghton Mifflin, 1971.
- **[Foundational]** Wald. *Sequential Tests of Statistical Hypotheses.* Annals of Mathematical Statistics, 1945.
- **[SOTA]** Aggarwal, Madaan, Yang, Mausam. *Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs.* EMNLP 2023. — arXiv:2305.11860
- **[SOTA]** Li, Yuan, Zhang, et al. *Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning.* ICLR 2024. — arXiv:2401.10480
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Damani, Shenfeld, Peng, Bobu, Andreas. *Learning How Hard to Think: Input-Adaptive Allocation of LM Computation.* ICLR 2025. — arXiv:2410.04707
- **[SOTA]** Muennighoff, Yang, Shi, et al. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[Theory]** Howard, Ramdas, McAuliffe, Sekhon. *Time-uniform, Nonasymptotic Confidence Sequences.* Annals of Statistics, 2021.
- **[Theory]** Kaufmann, Cappé, Garivier. *On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models.* JMLR 17(1), 2016.
- **[Systems]** Fu, Chen, Jha, et al. *Efficiently Serving LLM Reasoning Programs with Certaindex.* 2024. — arXiv:2412.20993

## 10. Worked Example

One MATH item, an 8B model at $T=0.8$, $N_{\max}=40$, Beta stopping with threshold $\eta = 0.85$.

Draws: $y_1{=}A$, $y_2{=}A$, $y_3{=}B$, $y_4{=}A$, $y_5{=}A$. Counts $(n_A,n_B) = (4,1)$. Posterior with a $\mathrm{Beta}(1,1)$ prior is $\mathrm{Beta}(5,2)$, and

$$\Pr[p_A > \tfrac12] = 1 - I_{1/2}(5,2) = 1 - \sum_{k=5}^{6}\binom{6}{k}2^{-6} = 1 - \tfrac{7}{64} = 0.891 > 0.85.$$

Stop at $\tau = 5$. Cost: 5 traces instead of 40, an $8\times$ saving. The certificate is honest about what it says.

Now the population truth for this item: $p_A = 0.55$, $p_B = 0.30$, tail $0.15$ — and **$A$ is wrong**, $B$ is the gold answer. The rule's claim "$A$ is the mode with probability 0.89" is *correct*. Running to $n=40$ would return $A$ too: $\Pr[n_A > n_B]$ at $n=40$ with these parameters is above 0.98. The extra 35 samples buy nothing.

Make the failure quantitative. Set $c = 0.01$ per sample (1 accuracy-point per sample). Costs on this item:

| Rule | $\mathbb{E}[\tau]$ | $\Pr[\text{correct}]$ | $\ell + c\tau$ |
|---|---|---|---|
| Stop at 5 (Beta, $\eta{=}0.85$) | 5 | 0.30 | $0.70 + 0.05 = 0.75$ |
| Fixed $n{=}40$ | 40 | $\approx 0.01$ | $0.99 + 0.40 = 1.39$ |
| Oracle $\tau^\star$ | 1 | — | $0.00 + 0.01 = 0.01$ (returns $B$ from a single lucky draw, then stops) |

The adaptive rule beats fixed-$n$ by 0.64 — and is 0.74 away from the oracle. Every one of those 0.74 points sits in the gap between the mode $a^\star = A$ and the answer $y^\star = B$. No stopping rule reading only $y_{1:n}$ under this aggregator can close it, because voting has already converged to the wrong answer by $n=5$. That is the obstruction: the certificate is sound, the objective it certifies is the wrong one, and the fix lives in the aggregator (a verifier that can prefer $B$ at frequency $0.30$) rather than in $\tau$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*