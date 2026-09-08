---
id: 17-reasoning/difficulty-conditioned-compute-routing
title: "Optimal Difficulty-Conditioned Compute Routing"
topic: 17-reasoning
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Difficulty-Conditioned Compute Routing

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/difficulty-conditioned-compute-routing` · **Status:** open

## 1. Problem Statement

Given a query $x$ and a fixed *average* inference budget, decide **before or during** generation how much compute to spend on $x$: which model, how many samples, how long a chain of thought, which verifier. Uniform allocation is provably wasteful — easy items are solved by one greedy sample, hard items are not solved by 256 — but the allocation must be chosen without knowing the answer.

Three variants, of different difficulty:

- **Measurement.** Define and estimate the *difficulty* of $x$ for policy $\pi$ so that a router can condition on it. Difficulty is not a property of $x$ alone; it is a property of the pair $(x,\pi)$, and the natural estimator (the model's own pass rate) is the quantity the router is trying to exploit.
- **Method.** Learn a router $\rho$ mapping $x$ to an allocation that beats the best uniform allocation at equal expected cost, with the routing overhead charged to the budget.
- **Theory.** Characterize the optimal allocation. Under what conditions on the per-item accuracy-vs-compute curves does the constrained optimum have a closed form, and how much can adaptivity gain over the best uniform policy — is the gap bounded, or unbounded in the number of difficulty strata?

Solving it means: a router that, on held-out tasks with no per-item labels available at inference time, achieves the accuracy of a $k\times$ larger uniform budget with $k$ materially $>1$, and does so under distribution shift in the difficulty mix.

## 2. Formal Setting

Let $x \sim \mathcal{D}$ be a query, $y^\star(x)$ its ground truth, $\pi$ a generator, and $a \in \mathcal{A}$ an **action** — a full allocation, e.g. $a = (\text{model}, n_{\text{samples}}, \text{max thinking tokens}, \text{verifier})$. Cost $c(x,a)$ is measured as accelerator-seconds or, more portably, as generated tokens weighted by model size: $c = \sum_m 2 P_m T_m$ FLOPs, $P_m$ parameters and $T_m$ tokens emitted by model $m$ including all discarded samples.

Success $s(x,a) \in \{0,1\}$ is the verifier-free task metric (exact match, unit tests). Define the **per-item compute–accuracy curve**

$$u_x(b) \;=\; \max_{a:\,c(x,a)\le b} \; \mathbb{E}\big[s(x,a)\big],$$

estimated by $\hat u_x(b)$ from $K$ independent rollouts ($K \ge 32$ for a usable standard error of $\le 0.09$ at $p=0.5$).

The routing problem is a stochastic knapsack:

$$\max_{\rho:\ \mathcal{X}\to\Delta(\mathcal{A})} \ \mathbb{E}_{x,a\sim\rho(x)}\big[s(x,a)\big] \quad \text{s.t.} \quad \mathbb{E}_{x,a\sim\rho(x)}\big[c(x,a)\big] \le B .$$

With a Lagrange multiplier $\lambda \ge 0$ the optimum is pointwise: $\rho^\star(x) \in \arg\max_a \{\mathbb{E}[s(x,a)] - \lambda c(x,a)\}$, and $\lambda$ is tuned so the budget binds. If each $u_x$ is concave in $b$, the water-filling solution equalizes marginal return $u_x'(b_x) = \lambda$ across items; the oracle gain over uniform is then driven entirely by the variance of $u_x'$ across $x$.

**Difficulty** is defined operationally as $d_\pi(x) = 1 - \mathbb{E}[s(x, a_0)]$ at a reference action $a_0$ (one greedy sample), estimated as $1 - \hat p$ over $K$ rollouts. A router uses a *predictor* $\hat d_\phi(x)$ computed from $x$ alone, or from a prefix, at cost charged to $B$.

**Assumptions, and which break.**
1. *$u_x$ is concave.* Violated: many items show a threshold — pass rate near 0 until enough samples or enough thinking tokens, then a jump. Best-of-$n$ coverage curves are roughly linear in $\log n$, which is concave, but *verified* accuracy is not, because verifiers saturate and then decline (Cobbe et al. 2021; Stroebl et al. 2024).
2. *$s$ is Bernoulli with a stable $p$ per item.* Violated by decoding temperature, prompt formatting and KV-cache nondeterminism; the same item's $\hat p$ moves several points across serving stacks.
3. *Cost is separable and known.* Violated: batching and speculative decoding make wall-clock cost of an allocation depend on the *other* items in the batch, so $c(x,a)$ is not a per-item constant.
4. *Difficulty transfers across policies.* Violated: $d_\pi$ is policy-specific, so a router trained against $\pi$ is stale after a checkpoint update.

## 3. State of the Art

**Established (ablated, with controls at equal compute).**
- *Compute-optimal test-time scaling*: Snell et al. (ICLR 2025, arXiv:2408.03314) bin MATH questions into 5 difficulty levels by the model's own pass rate, then select among best-of-$n$, beam search and sequential revision per bin. They report the compute-optimal strategy matching best-of-$N$ accuracy with roughly $4\times$ less test-time compute on PaLM 2-S*. Crucially they report both an *oracle* difficulty bin (labels used) and a *predicted* bin (verifier score used); the predicted variant is meaningfully weaker, which is the honest form of the result.
- *Input-adaptive allocation*: Damani et al. (ICLR 2025, arXiv:2410.04707) train a lightweight predictor of the marginal value of extra compute and route via the Lagrangian above; reported up to ~50% compute reduction at fixed accuracy, or accuracy gains of a few points at fixed budget, on math and code benchmarks with Llama-3-class models.
- *Adaptive self-consistency*: Aggarwal et al. (EMNLP 2023, arXiv:2305.11860) stop sampling when a Beta posterior over the majority answer is confident; ~3× fewer samples at matched accuracy. This is a *sequential* router and is the strongest reproduced result in the family because the stopping rule needs no learned difficulty model.

**Claimed but unablated / benchmark-number-only.**
- Length-control methods — s1 budget forcing (Muennighoff et al., arXiv:2501.19393), L1 (Aggarwal & Welleck, arXiv:2503.04697) — report accuracy-vs-token-budget curves, but the budget is set globally, not per item; the *difficulty-conditioned* claim is not isolated.
- Cross-model cascades — FrugalGPT (Chen et al., arXiv:2305.05176), RouteLLM (Ong et al., arXiv:2406.18665) — report large cost savings, but on preference/chat mixes where a large fraction of queries are trivially easy; the saving is dominated by mix composition, not by routing quality, and no paper in this line reports a difficulty-stratified breakdown.

## 4. What Is Known

- **Coverage scales, verification does not.** Brown et al. (arXiv:2407.21787): with DeepSeek-Coder-V2-Instruct, SWE-bench Lite coverage (any of $n$ samples correct) rises from 15.9% at $n{=}1$ to 56% at $n{=}250$; but without a perfect verifier the selected-answer accuracy is far below coverage. Scale: 8B–236B models, 5 tasks.
- **Verifier quality sets the ceiling.** Lightman et al. (ICLR 2024, arXiv:2305.20050): a process reward model reaches 78.2% on a 500-problem MATH subset with 1860 samples per problem, versus a lower outcome-RM number at identical sample count. Scale: GPT-4-class base.
- **Difficulty ordering is real and cheap to estimate.** IRT-style item difficulty fitted on benchmark response matrices predicts held-out correctness well enough that ~100 anchor items reproduce full-benchmark accuracy within ~2 points (tinyBenchmarks, Polo et al., ICML 2024, arXiv:2402.14992). This is difficulty *across models*, which is the transferable component.
- **Models have some self-knowledge of difficulty.** Kadavath et al. (arXiv:2207.05221) show calibrated P(IK) on 52B models, with calibration degrading off-distribution.
- **Overthinking is measurable.** Chen et al. (arXiv:2412.21187) show o1-like models emit thousands of tokens on arithmetic solvable in tens — the easy tail is where uniform allocation loses most.

## 5. What Is Not Known

- **Theoretically open.** No bound on the adaptivity gap: how large can $\mathbb{E}[u_x(b_x^\star)] - \max_b \mathbb{E}[u_x(b)]$ be as a function of the difficulty distribution and the shape class of $u_x$? Under thresholded (non-concave) $u_x$ the greedy Lagrangian is not optimal and the correct policy may need randomization; no one has characterized the optimum for that class.
- **Empirically open.** Whether *predicted* difficulty (no labels, no oracle) yields more than ~1.5× effective-compute gain at a frontier model scale on a hard, non-saturated benchmark. Every clean multiple ($4\times$, $\sim2\times$) reported so far either uses oracle bins, uses an easy mix, or is measured at $\le$70B scale.
- **Methodologically blocked.** Difficulty itself. $d_\pi(x)$ estimated from $\pi$'s own rollouts is contaminated: the router is trained on the same signal the generator uses, so gains partly reflect the router memorizing which items $\pi$ fails, not a transferable notion of hardness. There is no agreed policy-independent difficulty measure and no standard reporting of routing overhead inside the compute budget.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus absent ground truth in the same loop**. Difficulty must be estimated from the model whose behaviour the router is trying to change; the estimator $\hat d_\pi$ is a noisy function of $K$ Bernoulli draws, and the router's apparent gain scales with how much of $\hat d$ is item signal versus rollout noise. With $K{=}8$ the standard error on $\hat p$ is 0.18 — larger than most reported routing gains. Second, **the evaluation does not measure what it names**: "compute saved" is almost always reported in sampled tokens, excluding the router's own forward passes, the discarded prefixes under budget forcing, and the batching effects that make token counts a poor proxy for accelerator-seconds. Third, **non-identifiability**: routing gain and verifier gain are entangled — a better verifier improves best-of-$n$ selection *and* improves the difficulty predictor, so an A/B against uniform best-of-$n$ credits the router for verifier progress.

## 7. Current Research (as of 2026)

- Difficulty-aware RL for reasoning length: training the policy to emit short chains on easy items and long ones on hard, rather than routing externally (DeepSeek-R1 line, arXiv:2501.12948; L1). *(frontier — verify)* Several groups report that length penalties conditioned on rollout pass rate dominate external routers, which if true dissolves the routing problem into training.
- Mid-generation abort/continue policies from hidden states (Manvi et al., arXiv:2410.02725).
- Serving-side work treating routing as an admission-control problem under batch-level SLOs — this is where assumption 3 above is being taken seriously. *(frontier — verify)*
- Surveys tracking the area: Sui et al., *Stop Overthinking* (arXiv:2503.16419); test-time-scaling survey (arXiv:2503.24235).

## 8. Concrete Next Experiment

**Question.** Does label-free predicted difficulty beat the best uniform budget by more than 1.5× effective compute, once router cost is charged?

**Scale.** One open 32B reasoning model. 1,500 items: 500 MATH-500, 500 AIME-style hard, 500 GSM8K-easy — a *reported*, fixed difficulty mix. Estimate ground-truth $\hat p_x$ with $K{=}64$ rollouts per item at each of 4 budgets $b \in \{2^{10}, 2^{12}, 2^{14}, 2^{16}\}$ thinking tokens. Total: ~$1500 \times 64 \times 4 \approx 3.8\times10^5$ rollouts, ~5k H100-hours.

**Arms.**
1. *Control*: best uniform budget, tuned on held-out data (the honest baseline nobody reports).
2. *Oracle*: allocation from true $\hat p_x$ — the ceiling.
3. *Predicted*: router from $x$ alone (small encoder), router cost added to $B$.
4. *Prefix*: router from the first 256 generated tokens, prefix cost charged.

**Deciding number.** The **effective compute multiplier** $k = B_{\text{control}}/B_{\text{arm}}$ at matched accuracy, with 95% bootstrap CI over items. Arm 3 or 4 achieving $k \ge 1.5$ with the CI excluding 1.0 settles the empirical question affirmatively; $k < 1.2$ while arm 2 shows $k > 3$ localizes the failure in difficulty *prediction*, not in the allocation math. Report the oracle–predicted gap as the headline; it is the quantity the field currently hides.

## 9. Key References

- **[Foundational]** Cobbe, Kosaraju, Bavarian, et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[Foundational]** Wang, Wei, Schuurmans, et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. — arXiv:2203.11171
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* ICLR 2025. — arXiv:2408.03314
- **[SOTA]** Damani, Shenfeld, Peng, Bobu, Andreas. *Learning How Hard to Think: Input-Adaptive Allocation of LM Computation.* ICLR 2025. — arXiv:2410.04707
- **[SOTA]** Aggarwal, Madaan, Yang, Mausam. *Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs.* EMNLP 2023. — arXiv:2305.11860
- Brown, Juravsky, Ehrlich, et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- Lightman, Kosaraju, Burda, et al. *Let's Verify Step by Step.* ICLR 2024. — arXiv:2305.20050
- Chen, Zeng, et al. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* 2023. — arXiv:2305.05176
- Ong, Almahairi, Wu, et al. *RouteLLM: Learning to Route LLMs with Preference Data.* 2024. — arXiv:2406.18665
- Muennighoff, Yang, Shi, et al. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- Polo, Weber, Choshen, et al. *tinyBenchmarks: Evaluating LLMs with Fewer Examples.* ICML 2024. — arXiv:2402.14992
- **[Survey]** Sui, Chuang, Wang, et al. *Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.* 2025. — arXiv:2503.16419

## 10. Worked Example

Two items, budget 100 samples total, verifier is perfect (majority-of-correct selection).

- $x_1$ (easy): $p_1 = 0.9$ per sample. $u_1(n) = 1-0.1^n$.
- $x_2$ (hard): $p_2 = 0.02$. $u_2(n) = 1-0.98^n$.

Uniform ($n{=}50$ each): $u_1 = 1.000$, $u_2 = 1-0.98^{50} = 0.636$. Mean **0.818**.
Oracle: marginal returns equalize at $n_1 = 2$, $n_2 = 98$: $u_1 = 0.99$, $u_2 = 1-0.98^{98} = 0.862$. Mean **0.926**. Adaptivity is worth +10.8 points, and to reach 0.926 uniformly needs $n \approx 140$ each — an effective multiplier of $k = 2.8$.

Now the obstruction. The router does not see $p_2 = 0.02$; it estimates it from $K{=}8$ rollouts. With $p_2 = 0.02$, $\Pr[\hat p_2 = 0] = 0.98^8 = 0.851$. So 85% of the time the hard item is indistinguishable from an item with $p = 0.001$ or $p = 0.05$ — all give $\hat p = 0$. If the router's rule is "spend the budget where $\hat p = 0$", it spends 98 samples on genuinely hopeless items too. Add a third item $x_3$ with $p_3 = 0.0001$: the oracle abandons it ($n_3 = 0$) and keeps mean utility high; the $K{=}8$ estimator cannot separate $x_2$ from $x_3$, splits the hard budget, and yields $u_2 = 1-0.98^{49} = 0.628$ — below the *uniform* baseline's 0.636 on that item, before charging the 8 probe rollouts (16% of the budget) to cost.

Separating $p = 0.02$ from $p = 0.0001$ at 95% confidence needs $K \gtrsim 150$ rollouts — 1.5× the entire budget being routed. That is the problem in one line: the measurement that would make routing optimal costs more than the compute it saves, and every published gain depends on where the paper hid that cost.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*