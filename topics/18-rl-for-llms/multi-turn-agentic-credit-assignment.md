---
id: 18-rl-for-llms/multi-turn-agentic-credit-assignment
title: "Multi-Turn Agentic Credit Assignment"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Turn Agentic Credit Assignment

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/multi-turn-agentic-credit-assignment` · **Status:** open

## 1. Problem Statement

An LLM agent runs a trajectory of $T$ turns — each turn a block of generated tokens, a tool call, and an environment observation — and receives one scalar reward at the end (tests pass, booking correct, task solved). **Credit assignment** is the problem of converting that single terminal scalar into a per-turn (or per-token) learning signal that is correct in expectation and low enough in variance to train on.

Three variants, with different difficulty:

- **Measurement.** Given a trajectory and a terminal reward, estimate the causal contribution of turn $t$: how much did $a_t$ change the probability of success, holding the policy fixed? This is a counterfactual quantity and there is no ground-truth label for it in any deployed agentic benchmark.
- **Method.** Build an estimator (process reward model, value baseline, return decomposition, hierarchical critic) that improves final task success per unit of compute over the trivial baseline: broadcasting the trajectory-level advantage uniformly to every token, as GRPO does.
- **Theory.** Characterize when the uniform-broadcast estimator is sample-efficient and when it is not, as a function of horizon $T$, reward sparsity, and the fraction of turns that are causally relevant. No such characterization exists for the LLM setting.

**Solved** would mean: a credit-assignment method that, at fixed rollout budget, beats uniform broadcast on long-horizon agentic benchmarks ($T \gtrsim 30$), with the gain shown to come from credit assignment rather than from added data, added compute, or a changed KL penalty.

## 2. Formal Setting

Model the agent as a policy $\pi_\theta$ over a token-level MDP nested inside a turn-level semi-MDP (Sutton, Precup & Singh, 1999). At turn $t$ the state $s_t$ is the full context (system prompt, all prior actions and observations); the action $a_t \in \mathcal{V}^{L_t}$ is a token sequence of length $L_t$; the environment returns observation $o_{t+1}$. The trajectory is $\tau = (s_1, a_1, o_2, \dots, a_T)$ with terminal reward $R(\tau) \in \{0,1\}$ or $[0,1]$.

**Objective.** $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] - \beta\, \mathbb{E}[\mathrm{KL}(\pi_\theta \Vert \pi_{\mathrm{ref}})]$.

**The quantity to be estimated.** The turn-level advantage
$$A^\pi(s_t, a_t) = \mathbb{E}[R \mid s_t, a_t] - \mathbb{E}_{a \sim \pi(\cdot \mid s_t)}[\,\mathbb{E}[R \mid s_t, a]\,].$$
*As measured:* the only unbiased estimator available without a learned critic is Monte Carlo — branch $k$ independent rollouts from $s_t$ under the same policy and same environment seed, and take the difference of empirical success rates. This is what VinePPO does; the measurement cost is $k \cdot T$ extra rollouts per state.

**Uniform broadcast (GRPO).** Sample $G$ rollouts from $s_1$, set for every token in rollout $i$
$$\hat{A}_i = \frac{R_i - \mathrm{mean}(R_{1:G})}{\mathrm{std}(R_{1:G})}.$$
This is unbiased for the *trajectory-level* gradient but assigns identical credit to causally decisive and causally irrelevant turns.

**Process reward.** A PRM $r_\phi(s_t, a_t) \to [0,1]$ trained either on human step labels (Lightman et al., 2023) or on MC branching labels (Math-Shepherd; Wang et al., 2024). *As measured:* the label for step $t$ is the empirical completion rate $\hat{p}_t = \frac{1}{k}\sum_j \mathbb{1}[R(\tau^{(j)}) = 1]$ from $k$ continuations — so the PRM regresses onto a value function, not onto causal contribution, and $\hat p_t$ has standard error $\sqrt{p(1-p)/k}$, roughly $0.16$ at $k=10$, $p=0.5$.

**Assumptions, and which break.**
1. *Markov state* — holds trivially, since $s_t$ is the whole context, but makes the state space enormous and every state effectively unique. Value-function generalization across $s_t$ is therefore untested, not merely imperfect.
2. *Stationary, resettable environment* — violated for real agentic tasks: web pages change, tool latency varies, stateful sandboxes cannot be forked cheaply. MC branching estimators assume resettability.
3. *Reward measures the task* — violated: SWE-bench-style test-pass rewards admit reward hacking, and $\tau$-bench-style checkers score final database state, not policy quality.
4. *Independent turns* — violated: an early bad tool call poisons the entire remaining context, so $A(s_t,a_t)$ is not additively decomposable.

## 3. State of the Art

**Empirical SOTA (established).**
- Outcome-only RL with uniform broadcast — GRPO (Shao et al., 2024), and its use in DeepSeek-R1 (2025) — is the strongest *reproduced* recipe for single-turn long-CoT reasoning. Its success at $T=1$ is established; it is the default carried into multi-turn agents by inertia, not by evidence.
- PRM-guided **inference-time search**: Lightman et al. (2023) report 78.2% vs 72.4% (ORM) on a 500-problem MATH subset under best-of-1860 reranking. Established for reranking; that is a *verification* result, not a training-time credit-assignment result.
- Multi-turn agentic RL frameworks — Search-R1 (Jin et al., 2025), RAGEN (Wang et al., 2025), ArCHer (Zhou et al., ICML 2024) — report gains on their own environments.

**Claimed but unablated.**
- That hierarchical turn-level critics (ArCHer) give order-of-magnitude sample-efficiency gains: reported on small language-game environments with sub-1B–7B policies, not replicated at frontier scale or on real tool environments.
- That per-step advantage estimation beats uniform broadcast at long horizon: VinePPO (Kazemnejad et al., 2024) shows it for math ($T$ = a handful of reasoning chunks), not for $T \gtrsim 30$ tool-use trajectories.
- Most agentic-RL papers report a benchmark number against a base-model or SFT arm, **not** against a compute-matched GRPO arm. That makes the credit-assignment claim unidentified.

**Theory SOTA.** Return decomposition with proven bias/variance benefits under delayed reward: RUDDER (Arjona-Medina et al., NeurIPS 2019); randomized return decomposition (Ren et al., ICLR 2022); hindsight credit assignment (Harutyunyan et al., NeurIPS 2019); counterfactual credit assignment (Mesnard et al., ICML 2021). None has been instantiated at LLM scale with a working ablation.

## 4. What Is Known

- **Delayed reward is provably bad for both standard estimators.** RUDDER (2019) shows TD-style bias decays only as a factor per update step in the delay $\Delta$, while MC return variance grows with the number of intervening random variables. In an agentic trajectory with $T=50$ turns and ~2k tokens each, the "delay" is $10^5$ tokens.
- **Process supervision beats outcome supervision for verification at 34B–175B scale** (Lightman et al., 2023; Uesato et al., 2022 at 70B). Both papers measure reranking/selection, and Uesato et al. found final-answer accuracy comparable between process and outcome supervision — the process advantage was in *reasoning* error rate, not answer rate.
- **Automatically-labelled PRMs work without human labels.** Math-Shepherd (Wang et al., ACL 2024) matches human-labelled PRM quality on GSM8K/MATH using MC completion rates at 7B.
- **A PRM optimized for accuracy is not optimal for RL.** Setlur et al. (ICLR 2025) show the useful signal is *progress* — the change in success probability under a prover policy — and report ~5–6× better compute efficiency for RL with progress-based verifiers than with outcome rewards on MATH-scale tasks.
- **Simple baselines are strong at $T=1$.** RLOO (Ahmadian et al., ACL 2024) shows a leave-one-out group baseline matches or beats PPO with a learned critic at 7B–70B, i.e. the learned value function contributed little.
- **Multi-turn reliability is the failure mode.** On $\tau$-bench (Yao et al., 2024), frontier models score roughly 60% pass@1 on retail but ~25% pass^8 (all 8 independent trials correct) — consistency across a trajectory collapses far faster than single-shot accuracy.

## 5. What Is Not Known

- **Theoretically open.** No sample-complexity separation between uniform-broadcast policy gradient and per-turn advantage estimation for the nested token/turn semi-MDP. Nobody has proved a bound of the form "success requires $\Omega(f(T))$ more rollouts under broadcast", nor a matching upper bound showing broadcast suffices when the fraction of decisive turns is $\Theta(1)$.
- **Empirically open.** Whether any per-turn credit method beats compute-matched GRPO on $T \ge 30$ real tool environments at $\ge 30$B scale. The experiment is runnable today; the published comparisons are almost all against non-compute-matched arms.
- **Methodologically blocked.** "Contribution of turn $t$" has no ground truth. MC branching measures a value difference under the *current* policy in a *resettable* environment; neither condition holds for stateful web/OS agents. Until there is an agreed measurement, PRM quality for agents cannot be scored, only its downstream proxy.

## 6. Why It Is Hard

The obstruction is **non-identifiability compounded by branching cost**.

Per-turn credit is only identified by intervention — re-running from $s_t$ with a different $a_t$. Observationally, in any group of rollouts sharing a prefix, the contribution of turn 3 and turn 9 enter the outcome through the same scalar; no amount of passive data separates them, because the policy chooses $a_3$ and $a_9$ dependently through the shared context (confounding by the policy itself). The intervention that would break the confound costs $k$ full rollouts per probed state; at $T=50$ and $k=8$ that is 400 extra trajectories per training trajectory, each with real tool latency, and many environments are not resettable at all.

Secondary: the terminal reward often does not measure the named quantity. A test-suite reward scores the diff, not the debugging policy; a database-state checker scores the last write, not the 20 turns before it.

## 7. Current Research (as of 2026)

- **Group-relative methods with intra-trajectory structure.** Extending GRPO with step-level groups formed by anchoring on repeated environment states, so a within-trajectory baseline is available at no extra rollout cost — GiGPO and successors *(frontier — verify: several 2025–2026 preprints, results mostly on ALFWorld/WebShop at 7B)*.
- **Turn-level critics and hierarchy.** ArCHer-lineage work at Berkeley (Levine group) and follow-ons.
- **Automatic process verifiers as advantage estimators** rather than rerankers — CMU/Google (Setlur, Kumar) progress-based verifiers.
- **Agentic RL environments at scale**: Search-R1 (UIUC), RAGEN, SWE-Gym-style trainable software environments, and internal frontier-lab agentic RL stacks whose credit-assignment details are undisclosed *(frontier — verify)*.
- **Rubric and LLM-judge step rewards** for non-verifiable domains — widely deployed, but judge bias as a function of trajectory length is largely unmeasured *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does per-turn credit assignment beat uniform broadcast at long horizon, at matched rollout compute?

**Scale.** A 32B open-weights instruct model. Environment: $\tau$-bench-airline plus a stateful SWE sandbox, filtered to trajectories with $T \ge 30$ tool calls. Budget: 200k training trajectories total per arm, identical KL coefficient $\beta$, identical prompt set, identical decoding temperature, 3 seeds.

**Arms.**
- *Control (must be present):* GRPO with uniform broadcast, $G=16$ rollouts per prompt.
- *Treatment A:* MC per-turn advantage (VinePPO-style), $G=4$ prompts-level rollouts and $k=4$ branch rollouts at 3 uniformly sampled turns — **same total rollout count as control**.
- *Treatment B:* PRM trained on the Treatment-A branch labels, used as a dense advantage with no extra online branching.
- *Placebo:* GRPO with per-turn advantages drawn as random permutations of Treatment A's, same magnitudes. This detects "dense noise acts as a regularizer".

**Deciding number.** Pass^4 on a held-out 400-task split at equal total rollouts. Treatment A or B must exceed control by $\ge 5$ absolute points with non-overlapping 95% bootstrap intervals across 3 seeds, *and* exceed the placebo by $\ge 3$ points. If A and B fail to beat the placebo, the reported gains in the literature are attributable to variance shaping, not credit assignment.

**Secondary readout.** Spearman correlation between PRM step scores and held-out MC branch success rates at $k=32$, computed on 1,000 turns. Below $\rho = 0.4$, the PRM is not measuring progress.

## 9. Key References

- **[Foundational]** Sutton, Precup & Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence 112(1–2), 1999.
- **[Foundational]** Arjona-Medina, Gillhofer, Widrich, Unterthiner, Brandstetter & Hochreiter. *RUDDER: Return Decomposition for Delayed Rewards.* NeurIPS, 2019. — arXiv:1806.07857
- **[Foundational]** Harutyunyan, Dabney, Mesnard, Azar, Piot, Heess, van Hasselt, Wayne, Singh, Precup & Munos. *Hindsight Credit Assignment.* NeurIPS, 2019. — arXiv:1912.02503
- **[Foundational]** Mesnard, Weber, Viola, Thakoor, Saade, Harutyunyan, Dabney, Stepleton, Heess, Guez, Moulines, Hutter, Buesing & Munos. *Counterfactual Credit Assignment in Model-Free Reinforcement Learning.* ICML, 2021. — arXiv:2011.09464
- **[SOTA]** Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu & Guo. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300 (GRPO)
- **[SOTA]** Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever & Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[SOTA]** Wang, Li, Shao, Xu, Dai, Li, Chen, Wu & Sui. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL, 2024. — arXiv:2312.08935
- **[SOTA]** Kazemnejad, Aghajohari, Portelance, Sordoni, Reddy, Courville & Le Roux. *VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment.* 2024. — arXiv:2410.01679
- **[SOTA]** Setlur, Nagpal, Fisch, Geng, Eisenstein, Agarwal, Agarwal, Berant & Kumar. *Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning.* ICLR, 2025. — arXiv:2410.08146
- **[SOTA]** Zhou, Zanette, Pan, Levine & Kumar. *ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL.* ICML, 2024. — arXiv:2402.19446
- **[SOTA]** Ahmadian, Cremer, Gallé, Fadaee, Kreutzer, Pietquin, Üstün & Hooker. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* ACL, 2024. — arXiv:2402.14740
- **[Benchmark]** Yao, Shinn, Razavi & Narasimhan. *$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[Benchmark]** Jimenez, Yang, Wettig, Yao, Pei, Press & Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Benchmark]** Abdulhai, White, Snell, Sun, Hong, Zhai, Xu & Levine. *LMRL-Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models.* 2023. — arXiv:2311.18232
- **[Survey]** Pignatelli, Ferret, Geist, Mesnard, van Hasselt, Pietquin & Toni. *A Survey of Temporal Credit Assignment in Deep Reinforcement Learning.* TMLR, 2024. — arXiv:2312.01072
- **[Context]** Uesato, Kushman, Kumar, Song, Siegel, Wang, Creswell, Irving & Higgins. *Solving math word problems with process- and outcome-based feedback.* 2022. — arXiv:2211.14275

## 10. Worked Example

A 12-turn airline-rebooking task. Group of $G=8$ GRPO rollouts; 3 succeed, 5 fail. Group mean $\bar R = 0.375$, population std $= 0.484$.

Broadcast advantages: success $\hat A = (1-0.375)/0.484 = +1.29$; failure $\hat A = -0.775$. Every token of all 12 turns in a successful rollout is reinforced at $+1.29$, including the 4 turns that were pure redundant lookups.

Now score two specific actions by their *observed* average credit:

| Action | Occurrences in group | Outcomes | Mean credit |
|---|---|---|---|
| $a_3$ = correct `get_reservation` before edit (genuinely helpful) | 4 | 2 succeed, 2 fail | $(2(1.29)+2(-0.775))/4 = +0.257$ |
| $a_5'$ = redundant `list_flights` (causally neutral) | 3 | 0 succeed, 3 fail | $-0.775$ |
| $a_9$ = the decisive `update_reservation` with correct payment id | 3 | 3 succeed | $+1.29$ |

The estimator ranks $a_9 > a_3 > a_5'$, which is directionally right. But the noise: per-occurrence credit has sd $\approx 1.03$, so the standard error on $a_3$'s $+0.257$ at $n=4$ is $1.03/\sqrt{4} = 0.52$ — **twice the signal**. To resolve a $+0.257$ effect at $2\sigma$ needs $n \approx (2 \times 1.03/0.257)^2 \approx 64$ occurrences of that action in that context.

With 12 turns and a realistic branching factor of ~8 plausible tool calls per turn, contexts recur rarely; reaching 64 occurrences per (context, action) pair requires on the order of $64 \times 8^{?}$ rollouts — empirically thousands per prompt, against the 8 actually drawn. And $a_5'$, which is neutral, receives $-0.775$ purely because it happened to co-occur with failures caused at turn 9: the policy's own correlation between $a_5'$ and later mistakes is the confound, and it cannot be removed by collecting more *passive* rollouts, only by branching at turn 5.

That is the obstruction in one table: the credit signal per action is ~4× below the noise floor at practical group sizes, and the cheap fix (more rollouts) buys resolution at $\sqrt{n}$ while the intervention that would actually identify the effect costs $k$ full environment rollouts per probed turn.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*