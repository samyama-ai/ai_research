---
id: 26-code-generation/credit-assignment-long-coding-trajectories
title: "Credit Assignment over Long Agentic Coding Trajectories"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Credit Assignment over Long Agentic Coding Trajectories

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/credit-assignment-long-coding-trajectories` · **Status:** open

## 1. Problem Statement

An agentic coding episode is a sequence of 20–500 tool calls (read file, grep, edit, run tests, read traceback) ending in one binary signal: the hidden test suite passes or it does not. The problem is to attribute that terminal signal to individual actions inside the episode.

Three variants, routinely conflated:

- **Measurement.** Given a completed trajectory $\tau$ and its outcome $R \in \{0,1\}$, produce a per-step credit vector $c_{1:H}$ that is *validated* — i.e. an independent intervention on step $t$ changes $R$ in the direction and magnitude $c_t$ predicts. No accepted validation protocol exists.
- **Method.** Given credit estimates, train a policy that beats outcome-only REINFORCE/GRPO at fixed compute. Solving this does not require solving the measurement variant; it only requires that the estimator reduce gradient variance faster than it adds bias.
- **Theory.** Bound the sample complexity of identifying $c_{1:H}$ under a sparse terminal reward, a non-Markovian observation stream (the agent's own growing context), and a stochastic environment.

Solved = a credit estimator whose per-step attributions survive a counterfactual-resampling audit at $H \ge 50$, *and* which yields a reproducible gain on held-out repositories at matched rollout budget.

## 2. Formal Setting

Episode: $\tau = (s_0, a_0, o_1, s_1, \dots, a_{H-1}, o_H)$, where $a_t$ is a tool call emitted as a token block, $o_t$ the environment observation (file contents, test output), and $s_t$ the agent's context window. Policy $\pi_\theta(a_t \mid s_t)$. Terminal reward
$$R(\tau) = \mathbb{1}[\text{FAIL\_TO\_PASS} \cup \text{PASS\_TO\_PASS tests all pass}] \in \{0,1\}.$$

**Ground-truth credit** is the counterfactual step advantage
$$A_t \;=\; \mathbb{E}_{\pi}\!\left[R \mid s_t, a_t\right] \;-\; \mathbb{E}_{\pi}\!\left[R \mid s_t\right] \;=\; Q^\pi(s_t,a_t) - V^\pi(s_t).$$

**As measured:** fork the container at step $t$, run $K$ continuations under $\pi$ with $a_t$ fixed and $K$ with $a_t$ resampled, and take
$$\hat A_t = \tfrac{1}{K}\sum_{k} R(\tau^{(k)}_{|a_t}) - \tfrac{1}{K}\sum_{k} R(\tau^{(k)}_{\sim \pi}), \qquad \mathrm{Var}(\hat A_t) \le \tfrac{1}{2K}.$$
To resolve $|A_t| = 0.05$ at $2\sigma$ needs $K \approx 800$ rollouts *per step*; a 50-step trajectory costs $\sim 4\times10^4$ episodes. That figure is the whole problem.

Outcome-only policy gradient assigns the same scalar to every token:
$$\nabla_\theta J = \mathbb{E}\Big[\big(R(\tau)-b\big)\textstyle\sum_{t=0}^{H-1}\nabla_\theta \log \pi_\theta(a_t\mid s_t)\Big],$$
whose variance grows as $O(H^2 \sigma_R^2)$ absent a per-step baseline (Kakade, 2003; Schulman et al., GAE, 2016). At $H=100$ tool calls and $\sim 10^4$–$10^5$ tokens per episode, this is the dominant cost driver.

**Assumptions and their status:**

| Assumption | Status in practice |
|---|---|
| Markov state $s_t$ | **Violated** — context truncation and summarization make $s_t$ a lossy function of history; identical file states give different $Q$. |
| Deterministic environment | **Violated** — flaky tests, network installs, timeouts. Empirically 1–5% of container executions in SWE-bench-style harnesses are non-reproducible. |
| $R$ measures task success | **Violated** — hidden tests are an incomplete specification; agents pass by editing tests, special-casing inputs, or exploiting weak assertions. |
| Stationary $\pi$ during audit | Violated by construction once you train on the credit signal. |
| Actions are exchangeable units | Violated — a single "edit" action can carry 300 tokens of which two matter. |

## 3. State of the Art

**Established (ablated, reproduced):**
- Outcome-supervised RL with a group-relative baseline (GRPO; Shao et al., *DeepSeekMath*, 2024) is the working default. It performs credit assignment only across *episodes*, not within them: every token of a successful trajectory gets the same advantage.
- **SWE-RL** (Wei et al., Meta, 2025, arXiv:2502.18449) trains on similarity-to-oracle-patch reward, reaching 41.0% on SWE-bench Verified with a 70B model — established as a benchmark number, and notably it sidesteps intra-trajectory credit entirely by using a dense *outcome* proxy.
- **SWE-Gym** (Pan et al., 2024, arXiv:2412.21139) shows trajectory-level rejection sampling plus a trained verifier lifts a 32B agent to 32.0% on SWE-bench Verified. The gain is from *selecting* trajectories, not from crediting steps within them.
- Process supervision beats outcome supervision in math: **Let's Verify Step by Step** (Lightman et al., ICLR 2024) — a PRM trained on 800K human step labels solves 78.2% of a 500-problem MATH subset under best-of-1860 search vs 72.4% for an outcome RM. Established, but the labels are human and the domain is a 10–20 step chain with a checkable per-step semantics.

**Claimed but unablated:**
- Automatic per-step labeling by Monte-Carlo rollout (**Math-Shepherd**, Wang et al., ACL 2024) transferring to 50+ step agentic coding. Papers report end-task deltas without holding rollout budget fixed against an outcome-only arm at the same total compute.
- Hierarchical / turn-level value functions for LLM agents (**ArCHer**, Zhou et al., ICML 2024) — shown on short text games ($H \lesssim 10$), not on repository-scale trajectories.
- Frontier-lab claims that long-horizon RL yields the SWE-bench Verified gains from ~50% (2024) to ~75%+ (2025–26). The credit-assignment mechanism is undisclosed; the numbers are benchmark numbers only.

## 4. What Is Known

- **Sparse-reward variance scaling.** Policy-gradient variance under terminal-only reward grows quadratically in horizon; per-step baselines/critics are the only known general fix (GAE, Schulman et al., 2016).
- **Return decomposition works when the reward is genuinely decomposable.** RUDDER (Arjona-Medina et al., NeurIPS 2018) gives exponential speedups on delayed-reward tasks with $H \sim 10^2$; hindsight credit assignment (Harutyunyan et al., NeurIPS 2019) gives an unbiased estimator using future-conditioned action probabilities. Neither has been run on LLM tool-use trajectories.
- **Trajectory length distribution.** OpenHands-style SWE agents (Wang et al., ICLR 2025) typically use 20–60 steps on SWE-bench Verified instances, with a long tail to the 100-step cap; token cost $10^5$–$10^6$ per episode.
- **Reward is a leaky proxy.** SWE-bench Verified (OpenAI, 2024) exists precisely because 500 of 2,294 original instances were human-filtered as solvable and well-specified — i.e. ~78% of the original benchmark had reward defects. Contamination and memorization effects on SWE-bench have been reported independently *(frontier — verify)*.
- **Self-reflection is not credit assignment.** Reflexion (Shinn et al., NeurIPS 2023) improves pass@1 by verbal feedback but produces no calibrated per-step attribution; its "credit" is an unvalidated natural-language claim.

## 5. What Is Not Known

- **Theoretically open.** No sample-complexity bound for identifying $A_t$ when $s_t$ is a lossily-compressed history (a POMDP with agent-controlled state abstraction). Whether $c_{1:H}$ is even *identifiable* from outcome data at finite $K$ is unproven; multiple credit vectors are consistent with the same outcome distribution when actions are strongly correlated (a bad `grep` and the bad edit it caused are near-perfectly confounded).
- **Empirically open.** Nobody has published a counterfactual-resampling audit at $H\ge 50$ with $K \ge 100$ on real repositories. It is runnable — cost is roughly $10^5$ container-episodes, order $10^4$–$10^5$ USD — just unrun.
- **Methodologically blocked.** "A step is good" has no agreed definition in coding: good relative to the current policy, to an oracle, or to the ideal patch? Step boundaries are themselves a modeling choice (token / tool call / turn / sub-goal). Until the unit and the reference policy are fixed, PRM accuracy numbers across papers are not comparable.

## 6. Why It Is Hard

Three named obstructions, in order of bite:

1. **Non-identifiability from confounding.** Actions inside a trajectory are near-deterministic functions of prior actions. If action $a_3$ (open the wrong module) makes $a_7$ (wrong edit) almost certain, then $Q(s_3,a_3)$ and $Q(s_7,a_7)$ cannot be separated without interventions that break the dependency — and intervening changes the context distribution, i.e. changes the thing being measured.
2. **Compute cost of the counterfactual.** $O(HK)$ full-episode rollouts per audited trajectory, each an isolated container running a test suite (5–300 s). This is 3–4 orders of magnitude above the cost of the trajectory itself.
3. **The label is not the thing.** $R$ measures "hidden tests pass", not "the bug is fixed". Any credit assigned against $R$ credits reward hacking exactly as much as repair. A PRM trained on such labels learns to reward test-suite-shaped behaviour.

## 7. Current Research (as of 2026)

- **Dense proxies instead of true credit** — similarity-to-oracle-patch (Meta SWE-RL), execution-trace rewards, compile/lint intermediate signals. Cheap; biased in a known direction (rewards resemblance, not correctness).
- **Automatic PRMs for agents** via MC rollout labeling, extending Math-Shepherd to tool use *(frontier — verify)*; groups include the SWE-Gym/OpenHands line (UIUC, CMU, All Hands AI) and Chinese open-weight labs (Qwen, Kimi, DeepSeek) publishing long-horizon agentic RL recipes.
- **Environment scaling as a substitute for credit** — SWE-smith (Yang et al., 2025) synthesizes 50K+ task instances so that more independent episodes replace finer within-episode signal.
- **Turn-level value functions / hierarchical RL** (ArCHer lineage) applied to coding agents.
- **Reward-hacking audits** of SWE-bench-style harnesses — an active and under-published direction.

## 8. Concrete Next Experiment

**Question:** does any current per-step credit estimator agree with the counterfactual ground truth?

- **Scale.** 100 SWE-bench Verified instances × 5 trajectories each from one fixed open-weight agent (e.g. a 32B model in OpenHands, cap 50 steps). For every step of every trajectory, run $K=100$ forked continuations with the step's action held fixed and $K=100$ with it resampled at $T=1.0$. Cost: $100\times5\times40\times200 = 4\times10^6$ container-episodes if exhaustive — so subsample to 8 stratified steps per trajectory (early / mid / late, plus every step whose action type is `edit`), giving $\approx 8\times10^5$ episodes. Budget: one week on ~2,000 cores.
- **Control arm.** Two baselines against the resampling ground truth $\hat A_t$: (i) uniform credit, $c_t = R/H$ — what GRPO effectively uses; (ii) an LLM-judge PRM scoring each step 0–1 from the prefix.
- **Deciding number.** Spearman $\rho$ between predicted credit and $\hat A_t$, pooled within-trajectory. Uniform credit has $\rho = 0$ by construction. **If the LLM-judge PRM does not reach $\rho \ge 0.3$ with a bootstrap CI excluding 0.1, automatic per-step credit in agentic coding is not currently measurable, and dense-proxy rewards are the correct engineering answer.** Secondary readout: fraction of steps with $|\hat A_t| < 0.02$ — if that exceeds ~80%, most steps genuinely carry no credit and the search for a step-level signal is misdirected toward a few pivotal actions.

## 9. Key References

- **[Foundational]** Sutton, R. S. *Temporal Credit Assignment in Reinforcement Learning.* PhD thesis, University of Massachusetts Amherst, 1984.
- **[Foundational]** Arjona-Medina, J. et al. *RUDDER: Return Decomposition for Delayed Rewards.* NeurIPS, 2018. — arXiv:1806.07857
- **[Foundational]** Harutyunyan, A. et al. *Hindsight Credit Assignment.* NeurIPS, 2019. — arXiv:1912.02503
- **[Foundational]** Schulman, J. et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR, 2016. — arXiv:1506.02438
- **[Benchmark]** Jimenez, C. et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[SOTA]** Wei, Y. et al. *SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution.* 2025. — arXiv:2502.18449
- **[SOTA]** Pan, J. et al. *Training Software Engineering Agents and Verifiers with SWE-Gym.* 2024. — arXiv:2412.21139
- **[SOTA]** Lightman, H. et al. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[SOTA]** Wang, P. et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL, 2024. — arXiv:2312.08935
- **[Method]** Shao, Z. et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300
- **[Systems]** Wang, X. et al. *OpenHands: An Open Platform for AI Software Developers as Generalist Agents.* ICLR, 2025. — arXiv:2407.16741
- **[Method]** Zhou, Y. et al. *ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL.* ICML, 2024. — arXiv:2402.19446
- **[Method]** Shinn, N. et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS, 2023. — arXiv:2303.11366
- **[Data]** Yang, J. et al. *SWE-smith: Scaling Data for Software Engineering Agents.* 2025. — arXiv:2504.21798

## 10. Worked Example

Instance: a Django-style bug where the fix is a two-line change in `db/models/fields/related.py`. A 40-step trajectory succeeds ($R=1$). Steps:

| Step | Action | Uniform credit ($R/H$) | Resampled $\hat A_t$, $K=100$ |
|---|---|---|---|
| 1–11 | `grep`, `find`, read 6 files | 0.025 each | $-0.01$ to $+0.02$ (all within noise) |
| 12 | open the *correct* file | 0.025 | $+0.34$ |
| 13–27 | read, reason, 3 discarded edits | 0.025 each | $\approx 0$ |
| 28 | the two-line edit | 0.025 | $+0.41$ |
| 29–40 | run tests, confirm, finish | 0.025 each | $+0.02$ |

Two steps carry 0.75 of the outcome; 38 steps carry the rest. Uniform credit — what outcome-only GRPO applies — over-rewards 38 actions by roughly $16\times$ relative to their true advantage.

**Where the obstruction becomes visible.** Sampling error on each $\hat A_t$ at $K=100$ is $\sigma \le 0.05$, so the $\pm 0.02$ entries are indistinguishable from zero: the audit can find the two pivotal steps but cannot rank the other 38. Worse, steps 1–11 and step 12 are confounded — resampling step 12 from $s_{12}$ preserves the eleven searches that made the right file *findable*, so their contribution is silently folded into $V(s_{12})$ and scores zero. The measured credit is conditional on the prefix, not causal for it. And the whole table is computed against $R$ = "hidden tests pass"; had the agent instead relaxed an assertion in a visible test file at step 28, $\hat A_{28}$ would be $+0.41$ just the same. Total cost of this one table: $40 \times 200 = 8{,}000$ container-episodes, versus 1 to produce the trajectory.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*