---
id: 18-rl-for-llms/credit-assignment-long-cot
title: "Credit Assignment for Long Chains of Thought"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Credit Assignment for Long Chains of Thought

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/credit-assignment-long-cot` · **Status:** open

## 1. Problem Statement

An LLM trained with RL on verifiable tasks emits a chain of thought of $10^3$–$10^5$ tokens and receives **one** scalar at the end: correct or not. The problem is to attribute that scalar to the parts of the trace that caused it.

Three variants, routinely conflated:

- **Measurement.** Given a trace and a reward, estimate the causal contribution of each token, step, or span. Requires a definition of "contribution" that is identifiable from data.
- **Method.** Build an RL algorithm whose per-token advantage estimates are better than the uniform sequence-level baseline, and show the improvement survives ablation against a compute-matched control.
- **Theory.** Bound the sample complexity of policy improvement as a function of trace length $T$ under terminal-only reward, and characterise when finer-grained credit provably helps.

Solving it means: a per-token or per-step credit signal that (a) is estimated at cost comparable to outcome-only RL, (b) beats outcome-only RL at matched total FLOPs, and (c) does not degrade under optimisation pressure (no reward hacking of the credit signal itself).

## 2. Formal Setting

A token-level MDP. State $s_t = (x, y_{<t})$ for prompt $x$ and generated prefix $y_{<t}$; action $a_t = y_t \in \mathcal{V}$; deterministic transition $s_{t+1} = s_t \oplus y_t$. Horizon $T$ = number of generated tokens, terminating on EOS or a cap $T_{\max}$. Reward is terminal only:

$$r(s_t, a_t) = 0 \ \ \forall t < T, \qquad r(s_T, a_T) = R(x, y) \in \{0,1\}$$

measured by a verifier (exact-match on a boxed answer, unit tests, or a proof checker). Objective $J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}[R(x,y)]$, usually with a KL penalty $\beta\,\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$.

The quantity to estimate is the token-level advantage, which with $\gamma=1$ and terminal reward equals

$$A^\pi(s_t, a_t) = V^\pi(s_t \oplus a_t) - V^\pi(s_t), \qquad V^\pi(s) = \mathbb{E}_{y \sim \pi(\cdot|s)}[R(x, s \oplus y)] .$$

**How each quantity is actually measured.**

- $V^\pi(s_t)$: either a learned critic head (PPO), or Monte Carlo — sample $K$ completions from $s_t$, take $\hat V = \frac{1}{K}\sum_k R_k$, standard error $\sqrt{\hat V(1-\hat V)/K}$ (VinePPO).
- $A_t$ under GRPO/RLOO: **no** per-token estimate at all. For a group of $G$ rollouts, $\hat A_t = (R_i - \mu_G)/\sigma_G$ for every $t$ in rollout $i$ — one number broadcast over all $T_i$ tokens.
- "Step" boundaries: newline or sentence splits, or model-emitted delimiters. Not part of the MDP; an annotation convention.
- PRM score: a classifier $p_\phi(\text{step } j \text{ is good} \mid x, y_{\le j})$, trained on human labels (PRM800K) or on MC rollout success rates (Math-Shepherd).

**Assumptions and their status.**

- *Markov in $s_t$* — holds by construction (state is the full prefix), but makes $V^\pi$ a function on an exponentially large space; no generalisation guarantee.
- *Reward measures the target* — violated. Verifiers admit false positives (right answer, wrong reasoning) and length/format hacking.
- *Steps are causally separable* — violated. Later steps rewrite earlier ones; self-correction means a "bad" step can be net-positive.
- *$\pi$ fixed during MC value estimation* — violated on-policy; $\hat V$ is stale within a batch.
- *Bounded $T$* — violated in practice; R1-style training grows traces monotonically, so the credit-assignment horizon is itself a moving target.

## 3. State of the Art

**Empirical SOTA is outcome-only.** DeepSeek-R1 (DeepSeek-AI, *Nature*, 2025; arXiv:2501.12948) trains with GRPO on binary verifier reward and no process reward, no critic, and no value network. Its report states explicitly that PRMs and MCTS were tried and abandoned: PRMs were vulnerable to reward hacking and expensive to retrain. DAPO (Yu et al., 2025, arXiv:2503.14476) and Kimi k1.5 (Kimi Team, 2025, arXiv:2501.12599) likewise use sequence-level advantages. **Established:** frontier reasoning results are reachable with the crudest possible credit assignment.

**Process supervision helps at inference, less clearly in RL.** Lightman et al. (*Let's Verify Step by Step*, ICLR 2024, arXiv:2305.20050) is established for *reranking*: a PRM reaches 78.2% on a 500-problem MATH subset under best-of-1860 versus 72.4% for an outcome RM. That is a verifier result, not a credit-assignment result — it does not show PRM-shaped advantages improve policy gradients.

**Claimed but under-ablated.** VinePPO (Kazemnejad et al., ICML 2025, arXiv:2410.01679) replaces the critic with MC value estimates at sampled states and reports reaching PPO's peak MATH/GSM8K accuracy in fewer gradient steps on RhoMath-1.1B and DeepSeekMath-7B; the comparison is against PPO with a weak critic, not against a FLOP-matched GRPO arm. Setlur et al. (*Rewarding Progress*, ICLR 2025, arXiv:2410.08146) define process advantage verifiers (PAVs) measuring progress under a *prover* policy distinct from the base policy, and report >5× sample efficiency and >6% accuracy over ORM-based search; ablations are on ≤9B models and math only. Wang et al. (Math-Shepherd, ACL 2024, arXiv:2312.08935) get PRM labels from MC rollouts without human annotation; Zhang et al. (arXiv:2501.07301, Qwen) then showed MC-derived PRM labels are substantially noisier than human ones and that PRM benchmarks reward classifier calibration rather than policy improvement.

**Theory SOTA** is thin: no LLM-specific bound. The closest is Laidlaw et al. (*Bridging RL Theory and Practice with the Effective Horizon*, NeurIPS 2023 outstanding paper, arXiv:2304.09853) — hardness scales with the *effective* horizon, not $T$, when the random policy's Q-function is greedily near-optimal.

## 4. What Is Known

- Outcome-only GRPO takes a 671B-parameter model from 15.6% to 71.0% pass@1 on AIME 2024 (86.7% with 64-sample majority vote) while mean trace length grows past 8,000 tokens (DeepSeek-R1-Zero, 2025). Credit assignment over $\sim 10^4$ tokens from one bit is *sufficient* for large gains.
- Process feedback reduces reasoning errors more than answer errors: Uesato et al. (arXiv:2211.14275, 70B Chinchilla, GSM8K) found final-answer error rates close between outcome- and process-supervised models while trace error dropped roughly 14% → 4%. Established, independently consistent with Lightman et al.
- pass@k crossover: Yue et al. (arXiv:2504.13837) find RLVR-trained models beat base models at $k=1$ but are matched or beaten at $k\gtrsim 128$ across math/code/vision benchmarks up to 32B. Interpretation contested; the measurement has been reproduced.
- Gradient concentration: Wang et al. (arXiv:2506.01939) report that restricting policy-gradient updates to the top-20% highest-entropy tokens matches or beats full-token RLVR on Qwen3-32B (AIME'24/'25). Suggests most tokens carry no usable credit — one run family, not yet independently replicated.
- Reward-model overoptimisation follows a predictable KL-indexed curve (Gao, Schulman, Hilton, ICML 2023, arXiv:2210.10760); PRMs inherit this and R1's report cites hacking as the reason PRMs were dropped.
- MC value estimation cost is explicit: $K=9$ rollouts per sampled state in VinePPO, at 1.1B–7B scale, on traces of a few hundred tokens.

## 5. What Is Not Known

**Theoretically open.** No sample-complexity separation between terminal-reward and dense-reward policy gradients for the deterministic-transition token MDP. Nobody has shown a class of reasoning tasks where per-step credit provably reduces the number of trajectories needed by more than a constant factor — nor the converse impossibility.

**Empirically open.** The decisive experiment is runnable and unrun: process-shaped advantages versus outcome-only GRPO, **FLOP-matched**, at $\ge$30B parameters with $\ge$8k-token traces. Every existing PRM-in-RL comparison is at $\le$9B, on short traces, or unmatched on compute.

**Methodologically blocked.** "The contribution of step $j$" has no agreed estimand. $A^\pi(s_j)$ is policy-dependent, so a step is "good" only relative to the continuation policy — Setlur et al. make this explicit with the prover policy, but which prover is the right measurement instrument is undefined. Counterfactual definitions (Mesnard et al., *Counterfactual Credit Assignment*, ICML 2021; Harutyunyan et al., *Hindsight Credit Assignment*, NeurIPS 2019) are non-identifiable here because states are never revisited: every prefix is unique, so no two trajectories share a state past the first divergence.

## 6. Why It Is Hard

**Non-identifiability plus a variance wall, not compute alone.**

Per-token advantage under terminal binary reward has magnitude $O(1/T)$ for a typical token, while a single MC estimate of $V^\pi$ has standard deviation up to $0.5$. Resolving an advantage difference $\epsilon$ needs $K \approx 0.25/\epsilon^2$ rollouts *per state*. For $\epsilon = 0.05$ that is 100 rollouts at one state; for all $T=8{,}000$ states, $8\times10^5$ rollouts of $\sim 8$k tokens each — $\sim 6\times 10^9$ generated tokens **per prompt, per policy update**. The policy changes every step, so the estimates cannot be amortised.

Learned critics avoid the sampling cost but transfer the problem to generalisation over an exponentially large prefix space, and empirically the critic is the weakest component of PPO at this scale.

Compounding it: the ground truth does not exist. A step that looks wrong may trigger the self-correction that produces the right answer, so human step labels (PRM800K) measure *local validity*, not *causal contribution* — an evaluation that does not measure the thing it names. ProcessBench (Zheng et al., arXiv:2412.06559) scores error localisation, which is again validity, not credit.

## 7. Current Research (as of 2026)

- **Critic-free sequence-level methods** (DeepSeek, Moonshot, ByteDance-Seed, Alibaba-Qwen): GRPO/DAPO/RLOO variants, token-level loss normalisation, clip-higher, dynamic sampling. The working assumption is that credit assignment is *not* the bottleneck — data difficulty and entropy collapse are.
- **Implicit PRMs**: Yuan et al., *Free Process Rewards without Process Labels* (arXiv:2412.01981) — an outcome-trained RM parameterised as a log-likelihood ratio yields per-token rewards for free. Cheap enough to test at scale; RL-side ablations still small. *(frontier — verify)*
- **Advantage-shaped process rewards** — CMU/Google follow-ups to PAVs, on choosing the prover policy. *(frontier — verify)*
- **Entropy- and token-selection-based credit** (Alibaba, Tsinghua): restrict updates to forking tokens.
- **Monitorability as a constraint**: Baker et al. (OpenAI, arXiv:2503.11926) show that optimising against a CoT monitor drives obfuscated hacking — a direct argument against dense CoT-level rewards.
- **Long-horizon/agentic RL**, where traces span tool calls and $T > 10^5$; here outcome-only credit visibly degrades and step-level attribution is being revisited.

## 8. Concrete Next Experiment

**Question:** does per-step credit beat uniform sequence-level credit at matched compute, on long traces?

- **Scale.** Qwen3-32B-Base (or Llama-3.1-70B-Base), RLVR on ~40k verifiable math + code prompts, $T_{\max} = 16{,}384$, 400 policy steps, rollout batch 512 prompts × 16 samples. Eval: AIME 2025, HMMT 2025, LiveCodeBench, held-out GPQA-Diamond for transfer.
- **Arms, all matched to the same total generated-token budget** (this is the discipline every prior comparison lacks):
  1. **Control** — GRPO, outcome-only, group size 16.
  2. **MC-vine** — group size 12, remaining tokens spent on $K=8$ MC rollouts at 16 uniformly sampled prefix states per trace.
  3. **Implicit PRM** — outcome-RM-derived token rewards (Yuan et al.), group size 16, RM forward passes charged against the budget.
  4. **Entropy-gated** — GRPO restricted to top-20% entropy tokens.
- **Deciding number.** AIME 2025 pass@1 (32-sample average) at equal cumulative generated tokens. A treatment arm must exceed the control by **≥ 4.0 points**, with the gap holding across 3 seeds ($\ge 2\sigma$), to count. Secondary, and the more interesting number: pass@256 — if dense credit only improves pass@1 while pass@256 falls relative to control, it is sharpening, not credit assignment.
- **Cost estimate.** ~4×10^12 generated tokens per arm-seed; roughly 20k–40k H100-hours for the full 12-run grid. Within a single well-funded lab's budget, which is why its absence is a gap and not an excuse.

## 9. Key References

- **[Foundational]** Sutton, R., Precup, D., Singh, S. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence 112(1–2), 1999.
- **[Foundational]** Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. — arXiv:1506.02438
- **[Foundational]** Arjona-Medina, J. et al. *RUDDER: Return Decomposition for Delayed Rewards.* NeurIPS 2019. — arXiv:1806.07857
- **[Foundational]** Harutyunyan, A. et al. *Hindsight Credit Assignment.* NeurIPS 2019. — arXiv:1912.02503
- **[Foundational]** Mesnard, T. et al. *Counterfactual Credit Assignment in Model-Free Reinforcement Learning.* ICML 2021. — arXiv:2011.09464
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948
- **[SOTA]** Yu, Q. et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[SOTA]** Setlur, A., Nagpal, C., Fisch, A., Geng, X., Eisenstein, J., Agarwal, R., Agarwal, A., Berant, J., Kumar, A. *Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning.* ICLR 2025. — arXiv:2410.08146
- **[SOTA]** Kazemnejad, A. et al. *VinePPO: Refining Credit Assignment in RL Training of LLMs.* ICML 2025. — arXiv:2410.01679
- **[Method]** Lightman, H. et al. *Let's Verify Step by Step.* ICLR 2024. — arXiv:2305.20050
- **[Method]** Uesato, J. et al. *Solving math word problems with process- and outcome-based feedback.* 2022. — arXiv:2211.14275
- **[Method]** Wang, P. et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL 2024. — arXiv:2312.08935
- **[Method]** Shao, Z. et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024 (GRPO). — arXiv:2402.03300
- **[Method]** Ahmadian, A. et al. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* ACL 2024. — arXiv:2402.14740
- **[Analysis]** Yue, Y. et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[Analysis]** Zhang, Z. et al. *The Lessons of Developing Process Reward Models in Mathematical Reasoning.* 2025. — arXiv:2501.07301
- **[Analysis]** Gao, L., Schulman, J., Hilton, J. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. — arXiv:2210.10760
- **[Analysis]** Baker, B. et al. *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.* 2025. — arXiv:2503.11926
- **[Theory]** Laidlaw, C., Russell, S., Dragan, A. *Bridging RL Theory and Practice with the Effective Horizon.* NeurIPS 2023. — arXiv:2304.09853

## 10. Worked Example

One AIME-style problem, group size $G=16$, GRPO, traces averaging $T=8{,}000$ tokens.

Suppose 4 of 16 rollouts are correct. Then $\mu_G = 0.25$, $\sigma_G = 0.433$, and

$$\hat A_i = \frac{R_i - 0.25}{0.433} = \begin{cases} +1.73 & \text{correct} \\ -0.577 & \text{incorrect}\end{cases}$$

Every one of the $\sim 8{,}000$ tokens in a correct trace receives $+1.73$; every token in an incorrect trace receives $-0.577$. The batch delivers $16 \times 8{,}000 = 128{,}000$ token-level advantage values carrying exactly **one bit** of task information.

Now try to extract real credit for one specific span — say a "wait, let me recheck" backtracking phrase. It occurs in 3 correct and 5 incorrect rollouts. Its naive attribution is

$$\bar A = \frac{3(1.73) + 5(-0.577)}{8} = \frac{5.19 - 2.885}{8} = +0.288$$

positive, so the update pushes the phrase up. But this is pure selection: the phrase appears in traces that were already going to succeed. To get a causal estimate you must branch at the state $s_j$ preceding the phrase and compare $\hat V(s_j \oplus \text{phrase})$ against $\hat V(s_j \oplus \text{alternative})$.

Cost of that one comparison at a resolution of $\epsilon = 0.05$: $K = 0.25/\epsilon^2 = 100$ rollouts per branch, 200 total, each $\sim 6{,}000$ remaining tokens $\Rightarrow 1.2\times 10^6$ tokens — to license a single token-span's advantage. Doing it at 16 sampled states per trace, 512 prompts per step, 400 steps costs $\sim 4\times 10^{12}$ tokens, comparable to the *entire* outcome-only training run.

And even after paying it, the estimate is $A^{\pi_\theta}$ for the current $\pi_\theta$ only. One gradient step later the continuation policy has moved; the phrase that was worth $+0.05$ under a policy that backtracks poorly may be worth $-0.02$ under a policy that has learned to backtrack well. **The obstruction is not that the measurement is expensive — it is that the estimand is policy-indexed and expires faster than it can be measured.**

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*