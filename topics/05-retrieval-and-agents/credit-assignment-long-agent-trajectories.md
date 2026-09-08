---
id: 05-retrieval-and-agents/credit-assignment-long-agent-trajectories
title: "Credit Assignment over Long Agent Trajectories"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Credit Assignment over Long Agent Trajectories

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/credit-assignment-long-agent-trajectories` · **Status:** open

## 1. Problem Statement

An LLM agent runs a trajectory of tool calls, retrievals, and reasoning steps — 30 to 3,000 steps, tens to hundreds of thousands of tokens — and receives one scalar at the end: test suite passed, ticket resolved, answer correct. The problem is to attribute that scalar to the individual decisions that produced it.

Three variants, routinely conflated:

- **Measurement.** Given a trajectory $\tau$ and a terminal reward $R(\tau)$, estimate the *causal contribution* of step $t$: how much would $\mathbb{E}[R]$ change had the policy acted differently at $t$, holding everything else fixed? Solving this means producing per-step credit that agrees with an interventional ground truth.
- **Method.** Given only outcome labels, train a policy whose sample efficiency at horizon $T$ does not degrade as $\Theta(T)$ or worse. Solving this means a training algorithm that reaches a target success rate with a budget sublinear in horizon relative to outcome-only REINFORCE/GRPO.
- **Theory.** Characterize when per-step credit is *identifiable* from trajectory-level rewards alone, and bound the gradient variance of estimators that avoid learning a value function.

Failure to distinguish these is the standard reason the literature appears to disagree: process reward models improve *reranking* (a measurement-adjacent task) without demonstrating improved *sample efficiency* at long horizons (the method task).

## 2. Formal Setting

Model the agent as a POMDP-shaped token-level MDP. State $s_t$ is the full context (system prompt, history, retrieved documents); action $a_t$ is a step — either a token, or a semantic unit (one tool call plus its reasoning preamble). A trajectory is $\tau = (s_0, a_0, \dots, s_{T-1}, a_{T-1}, s_T)$ with terminal reward $R(\tau) \in \{0,1\}$ and $r_t = 0$ for $t < T$.

**Measured quantities.**

- $T$: **measured** as number of assistant turns that contain a tool call, plus one for the final answer. Token horizon $L = \sum_t |a_t|$ is reported separately; the two differ by 2–3 orders of magnitude and results do not transfer between them.
- $R(\tau)$: **measured** by the harness's own verifier (`pytest` exit code, string match, DB state check). Not the true task objective — verifier false-positive rate is itself unmeasured on most benchmarks.
- Advantage of step $t$: $A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$, with $V^\pi(s) = \mathbb{E}_\pi[R \mid s]$. **Measured** by Monte Carlo rollout: from prefix $s_t$, resample $K$ continuations under $\pi$ and average. Cost is $K \cdot T$ extra rollouts per trajectory; standard error of $\hat V$ is $\sqrt{p(1-p)/K}$, so $K=16$ gives $\pm 0.125$ at $p=0.5$ — larger than most reported per-step effects.
- Policy gradient: $\nabla_\theta J = \mathbb{E}\big[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\, A^\pi(s_t,a_t)\big]$. With outcome-only credit, $A$ is replaced by the constant $R - b$, and $$\mathrm{Var}\!\left[\textstyle\sum_t \nabla \log \pi (R-b)\right] = \Theta(T)\ \text{under mixing},\ \Theta(T^2)\ \text{when per-step scores are correlated,}$$ the correlated case being the realistic one for a single coherent plan.

**Assumptions, and which are violated.**

1. *Markov state.* Violated in practice only mildly — context is carried explicitly — but violated hard when the environment has hidden state (a mutated database, a rate limiter).
2. *Stationary environment.* Violated: web environments, flaky tests, and nondeterministic tools mean $R$ is stochastic given $\tau$. Re-running the same SWE-bench trajectory does not always give the same verdict.
3. *Counterfactual resampling is valid.* MC advantage estimation assumes you can resample from $s_t$ under the same environment. Violated whenever a step had a side effect (file written, email sent, cart purchased) — the prefix is not restorable without a snapshottable environment.
4. *Reward is the objective.* Violated by reward hacking: the verifier is a proxy.

## 3. State of the Art

**Established (with ablations).**

- **Process supervision beats outcome supervision for reranking.** Lightman et al., *Let's Verify Step by Step* (ICLR 2024): a process reward model (PRM) trained on PRM800K solves **78.2%** of a 500-problem MATH subset under best-of-1860 search, vs **72.4%** for an outcome-supervised RM and **69.6%** for majority vote, at GPT-4-scale base models. This is a *search/reranking* result, not a policy-optimization result.
- **MC-estimated per-step values beat a learned critic for math RL.** Kazemnejad et al., *VinePPO* (2024/2025): replacing PPO's value network with MC rollout-based value estimates improves MATH/GSM8K accuracy at equal or fewer gradient steps; the paper's ablation shows the learned critic's value estimates are barely better than chance at ranking states.
- **Outcome-only RL works at scale for reasoning.** DeepSeek-R1 (Nature, 2025) trains with GRPO on verifiable outcome rewards, no PRM, no value network — and reports that PRM-based approaches were tried and abandoned as prone to reward hacking. This is the strongest existing evidence that per-step credit is *not required* at horizons of a few thousand tokens.

**Claimed but unablated.**

- Automatic step-label generation (Math-Shepherd, Wang et al., ACL 2024) as a substitute for human step labels at agentic horizons: demonstrated on math, transferred to tool-use trajectories only by assertion.
- LLM-as-judge per-step critique ("Agent-as-a-Judge", Zhuge et al., 2024) as a credit signal: reported agreement with human judgment, but no experiment showing the resulting credit improves policy learning.
- Hierarchical value estimation for LLM agents (ArCHer, Zhou et al., ICML 2024): a genuine utterance-level/token-level two-level critic with reported gains on text games; not reproduced at SWE-bench-scale horizons.

**Benchmark-number-only.** Nearly all long-horizon agent claims — SWE-bench Verified resolve rates, τ-bench pass^k, WebArena success — are single aggregate numbers. They tell you nothing about *which* step failed, and there is no public benchmark that scores per-step credit against interventional ground truth at horizons above ~20 steps.

## 4. What Is Known

- **PRM step labels are expensive and finite.** PRM800K is ~800K step-level human labels over ~75K solutions — for math problems averaging under 20 steps. Nothing of comparable size exists for tool-use trajectories.
- **Step-level error localization is unsolved even for math.** ProcessBench (Zheng et al., ACL 2025), 3,400 test cases: the task is to find the *first* erroneous step. Strong PRMs of the 2024 generation lag general reasoning models substantially; critic-style models trained only on math step data generalize poorly to harder splits.
- **Variance scaling is a theorem, not a hypothesis.** For REINFORCE with a constant terminal reward and a state-independent baseline, gradient variance grows linearly in $T$ under independence and quadratically under correlated scores. This is why long horizons are hard *independently* of any LLM specifics.
- **Classical fixes exist and are proven.** Hindsight Credit Assignment (Harutyunyan et al., NeurIPS 2019) gives an unbiased estimator conditioning on future outcomes; Counterfactual Credit Assignment (Mesnard et al., ICML 2021) uses future-conditional baselines with a proven unbiasedness condition; RUDDER (Arjona-Medina et al., NeurIPS 2019) proves return decomposition makes the expected delay of a reward zero. None has been demonstrated on an LLM agent above toy scale.
- **Horizon degradation is measurable.** Across agent benchmarks, per-step accuracy $p$ compounds: $0.99^{100} = 0.366$. Empirically, agent success rate falls off faster than any single reported per-step accuracy predicts, implying correlated failures — but the correlation structure is unmeasured.

## 5. What Is Not Known

- **Theoretically open.** Whether per-step causal credit is *identifiable* from a finite set of (trajectory, terminal reward) pairs when the policy is near-deterministic. With low action entropy, the off-support counterfactual is never sampled; no proof exists either way about what class of credit functions is recoverable, or of a lower bound on samples needed at horizon $T$.
- **Empirically open.** Whether *any* per-step credit method beats outcome-only GRPO on sample efficiency at $T > 50$ tool calls. Runnable today — snapshottable environments plus MC rollouts — but nobody has published the head-to-head at that horizon with matched compute.
- **Methodologically blocked.** There is no accepted ground truth for "step $t$ deserved credit $c$" in agentic tasks. Human step labels measure *plausibility*, not causal contribution; MC advantage measures contribution *under the current policy*, which changes as training proceeds. Until a counterfactual-intervention protocol is standardized, PRM quality numbers are not comparable across papers.

## 6. Why It Is Hard

The obstruction is **absent interventional ground truth compounded by non-restorable state**.

To know whether step 17 mattered, you must rerun from step 17 with a different action and compare outcomes. That requires (a) an environment snapshot at step 17 and (b) enough resamples to beat the binomial noise. For a 100-step trajectory with $K=16$ resamples per step, one credit-labeled trajectory costs 1,600 additional rollouts — roughly $100\times$ the cost of collecting the trajectory itself. At SWE-bench-like costs of ~$1–5 per rollout, one labeled trajectory is $10^3$–$10^4$ dollars.

Secondary obstruction: **the evaluation does not measure what it names.** PRM benchmarks score agreement with human "is this step correct" annotations. Correctness and causal contribution come apart — a correct-but-useless step deserves zero credit; a locally wrong step that triggers a productive correction may deserve positive credit. Optimizing agreement with plausibility labels is therefore optimizing a different objective than the one the method claims.

## 7. Current Research (as of 2026)

- **Outcome-only RL at scale** — DeepSeek, Qwen, and open replications (verl, OpenRLHF): scale the group-relative baseline instead of learning per-step credit. Established direction.
- **MC/vine-style value estimation** — Mila and follow-ups to VinePPO: pay rollout cost to buy unbiased advantages. Bottleneck is environment snapshotting.
- **Execution-grounded credit for code agents** — SWE-RL (Wei et al., 2025) and successors use test-suite deltas as dense-ish intermediate reward; the credit signal comes from the environment, not a model. *(frontier — verify)*
- **Trajectory-level self-critique into training** — Reflexion-style verbal feedback (Shinn et al., NeurIPS 2023) folded back as a training signal rather than an inference-time hint. Widely claimed, thinly ablated. *(frontier — verify)*
- **Attention/influence-based attribution** as a cheap proxy for causal credit. No published evidence it correlates with MC advantage above chance at long horizons. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does per-step credit buy sample efficiency at $T \approx 100$, or is outcome-only sufficient?

**Scale.** A 7–14B open model on a snapshottable agentic environment (containerized SWE-bench-style repos or a forkable web sandbox), 3,000 training tasks with a binary verifier, trajectories filtered to $T \in [60, 150]$ tool calls. Fixed compute budget: $2 \times 10^5$ rollouts total per arm, counting *all* rollouts including those used for credit estimation. This is the critical accounting rule and the reason prior comparisons are uninformative.

**Arms.**
1. **Control:** GRPO, outcome-only, group size 8.
2. MC advantage (VinePPO-style), $K=4$ resamples per step at 10 uniformly sampled steps per trajectory.
3. PRM-scored steps from an LLM judge, no extra rollouts.

**Deciding number.** Held-out success rate on 500 unseen tasks at the point each arm has consumed $2\times10^5$ rollouts. Decision rule: per-step credit is worth it iff arm 2 or 3 exceeds the control by **≥5 absolute points** (roughly $3\sigma$ for $n=500$ near $p=0.3$: $\sigma \approx 2.0$ points). A tie or a loss is the more informative outcome — it means the field's credit-assignment machinery does not pay for its rollout cost at $T=100$, and no current paper reports enough to rule that out.

**Secondary readout.** Spearman correlation between arm 3's PRM scores and arm 2's MC advantages on the same 200 trajectories. If $\rho < 0.2$, the PRM is not measuring credit, whatever its ProcessBench score.

## 9. Key References

- **[Foundational]** Richard S. Sutton. *Temporal Credit Assignment in Reinforcement Learning.* PhD thesis, University of Massachusetts Amherst, 1984.
- **[Foundational]** Anna Harutyunyan, Will Dabney, Thomas Mesnard, et al. *Hindsight Credit Assignment.* NeurIPS, 2019.
- **[Foundational]** Jose A. Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes Brandstetter, Sepp Hochreiter. *RUDDER: Return Decomposition for Delayed Rewards.* NeurIPS, 2019. — arXiv:1806.07857
- **[Foundational]** Thomas Mesnard, Théophane Weber, Fabio Viola, et al. *Counterfactual Credit Assignment in Model-Free Reinforcement Learning.* ICML, 2021. — arXiv:2011.09464
- **[SOTA]** Hunter Lightman, Vineet Kosaraju, Yura Burda, et al. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[SOTA]** Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, et al. *VinePPO: Refining Credit Assignment in RL Training of LLMs.* 2024/2025. — arXiv:2410.01679
- **[SOTA]** Peiyi Wang, Lei Li, Zhihong Shao, et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL, 2024.
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025.
- **[SOTA]** Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, Aviral Kumar. *ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL.* ICML, 2024.
- **[Benchmark]** Chujie Zheng, Zhenru Zhang, Beichen Zhang, et al. *ProcessBench: Identifying Process Errors in Mathematical Reasoning.* ACL, 2025.
- **[Benchmark]** Carlos E. Jimenez, John Yang, Alexander Wettig, et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024.
- **[Benchmark]** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-Domain Tasks.* 2024.
- **[Survey]** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS, 2023.

## 10. Worked Example

A repo-repair agent runs 84 tool calls: 31 reads, 12 greps, 9 edits, 30 test runs. Final `pytest` exits nonzero. $R=0$. Which of the 9 edits was wrong?

**Outcome-only GRPO.** Group of 8 rollouts, 2 succeed. Every token in the 6 failures gets advantage $-0.65$; every token in the 2 successes gets $+1.94$. The 31 read calls — identical across all 8 rollouts, since the agent always starts by reading the same files — receive nonzero advantage of both signs. The estimator is unbiased, so this washes out in expectation. The question is how many samples "in expectation" costs. With $\approx 2\times10^4$ tokens per trajectory and correlated per-token scores, the variance term scales as $L^2$; a naive estimate says roughly $10^2$–$10^4\times$ more trajectories than a per-step method would need.

**MC advantage.** Snapshot at edit #5, resample $K=16$ continuations. Result: 3/16 succeed, so $\hat V(s_5) = 0.19 \pm 0.10$. Snapshot at edit #6: 1/16, $\hat V(s_6) = 0.06 \pm 0.06$. Estimated advantage of edit #6 is $-0.13 \pm 0.12$. **The 95% interval contains zero.** To resolve a true effect of $-0.13$ at $\alpha=0.05$ with 80% power needs roughly $K \approx 190$ per state. Across 9 edits that is ~1,700 rollouts — for *one* training trajectory.

**PRM.** A judge model scores edit #6 as "correct" — it is syntactically valid, matches the surrounding style, and is a locally reasonable fix. The actual failure is that edit #3, made 40 steps earlier, changed a shared fixture so the test edit #6 targets can no longer pass. The judge assigns edit #3 a high score too: read in isolation, it looks fine.

The obstruction is visible in the gap between the three arms. Outcome-only is unbiased but has variance scaling with $L^2$. MC is the ground truth but its confidence interval at affordable $K$ is wider than the effect. The PRM is affordable and confident and wrong, because it scores local plausibility while the credit lives in a 40-step-range interaction. There is currently no fourth option, and no benchmark that would have caught the PRM's error.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*