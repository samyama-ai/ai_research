---
id: 04-alignment/process-versus-outcome-supervision
title: "Process versus Outcome Supervision Superiority"
topic: 04-alignment
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Process versus Outcome Supervision Superiority

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/process-versus-outcome-supervision` · **Status:** empirically-open

## 1. Problem Statement

A model produces a reasoning trace $y = (y_1,\dots,y_T)$ and a final answer $a(y)$. Two supervision signals are available:

- **Outcome supervision:** one scalar per trace, $r_{\text{out}} \in \{0,1\}$, from checking $a(y)$ against ground truth.
- **Process supervision:** one label per step, $\ell_t \in \{-1,0,+1\}$, from a human or an automatic judge.

**The question.** Under a *matched budget* — equal annotation cost and equal training compute — does process supervision produce a better final policy than outcome supervision, and if so on which axis (final accuracy, trace validity, out-of-distribution generalization, resistance to reward hacking)?

Three variants that are routinely conflated:

- **Measurement variant.** Is there a metric on which process supervision is better that is not an artifact of the evaluation protocol? Best-of-$N$ reranking with a process reward model (PRM) is the standard evidence, and it is known to be biased toward the PRM's own training distribution.
- **Method variant.** Given a labeling budget, what is the optimal *mix* of dense and sparse labels? This is a resource-allocation question, not a binary.
- **Theory variant.** Is there a class of tasks on which dense per-step reward provably reduces sample complexity for policy improvement, and is that class non-empty for autoregressive reasoning?

Solving it means: a stated regime (task family, model scale, budget) plus a measured, replicated inequality between the two arms with the confound of labeling cost removed.

## 2. Formal Setting

Model the generation as an MDP $M = (\mathcal{S}, \mathcal{A}, P, r, \gamma=1)$ where a state $s_t = (x, y_{1:t})$ is the prompt plus the prefix, and an action is the next step $y_{t+1}$. The policy $\pi_\theta$ is the LM. Episodes terminate at $T \le T_{\max}$.

**Outcome reward, as measured.** $r_{\text{out}}(x,y) = \mathbf{1}[a(y) \equiv a^\star(x)]$ under a symbolic checker (`sympy` equivalence, unit test pass, exact match). Measured cost per label: one checker call, $c_o \approx 10^{-4}$ USD.

**Step value, as measured.** The only operational definition of "this step is correct" that scales is the Monte Carlo completion estimate under a *prover* policy $\mu$:

$$\hat q^\mu_t = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}\big[a(y_{1:t} \oplus z^{(k)}) \equiv a^\star(x)\big], \quad z^{(k)} \sim \mu(\cdot \mid x, y_{1:t}).$$

The induced process reward is the advantage $A^\mu_t = q^\mu_t - q^\mu_{t-1}$. Human step labels $\ell_t$ (PRM800K-style) instead measure *logical validity*, a different quantity: a step can be valid and have $A^\mu_t < 0$, and vice versa.

**Matched-budget predicate.** Let $B$ be total annotation spend and $C$ training FLOPs. Process supervision wins iff

$$\mathbb{E}\big[J(\pi_{\text{proc}}) \mid B, C\big] - \mathbb{E}\big[J(\pi_{\text{out}}) \mid B, C\big] > 0,$$

where $J$ is measured as pass@1 on a held-out task family, not as best-of-$N$ accuracy under the trained verifier. MC labeling costs $c_p \approx K \cdot T \cdot c_{\text{gen}}$ — for $K=8$, $T=10$ that is roughly $10^2\times$ the outcome label.

**Assumptions, and which are violated.**

1. *Steps are a well-defined decomposition.* Violated: newline-delimited steps in long chain-of-thought are arbitrary, and reasoning models emit backtracking traces where a "wrong" step is functionally useful.
2. *Step labels are Markov in the prefix.* Violated: $q^\mu_t$ depends on $\mu$'s ability to recover, so the label is a property of the prover, not the step.
3. *The checker is sound.* Violated at a measurable rate — false positives (right answer, invalid trace) occur; Uesato et al. measured trace-error rate 14.0% among correct-answer GSM8K solutions.
4. *Annotation cost is comparable.* Strongly violated, and most published comparisons do not control for it.

## 3. State of the Art

**Established (ablated, replicated).**

- Uesato et al. (2022) ran the only clean matched comparison: on GSM8K with a 70B Chinchilla-class model, ORM and PRM reranking reached statistically indistinguishable final-answer error (12.7% vs 13.4%), while process supervision cut trace errors from 14.0% to 3.4%. The established claim is *trace validity*, not accuracy.
- Automatic MC-derived process labels work without humans: Math-Shepherd (Wang et al., ACL 2024) reached 84.1% GSM8K / 33.0% MATH with Mistral-7B verification, matching human-labeled PRM quality on those benchmarks.

**Claimed but unablated.**

- Lightman et al. (2023) report a PRM solving 78.2% of a 500-problem MATH subset at best-of-1860 versus 72.4% for the ORM. The arms are not budget-matched (PRM800K is 800K human step labels) and the generator was trained on MATH-adjacent data; the number is a benchmark result under a reranking protocol, not a controlled causal estimate.
- Setlur et al. (2024) report ~5–6$\times$ sample efficiency and >6-point accuracy gains from process advantage verifiers (PAVs) in RL. Directionally supported by theory in the same paper, replicated by few outside groups.

**Counter-evidence at frontier scale.** DeepSeek-R1 (2025) reports abandoning PRMs in large-scale RL: step definition was ambiguous, automated labeling unreliable, and the PRM was reward-hacked. Final training used rule-based outcome rewards. This is the single most important datum and it is a systems report, not a controlled ablation.

## 4. What Is Known

| Result | Scale measured at | Number |
|---|---|---|
| Process ≈ outcome on final answer, better on traces | 70B, GSM8K | 12.7% vs 13.4% error; trace error 3.4% vs 14.0% |
| PRM beats ORM under best-of-$N$ reranking | GPT-4-class generator, MATH500 | 78.2% vs 72.4% at $N=1860$ |
| Automatic MC labels are competitive with human labels | 7B, GSM8K/MATH | 84.1% / 33.0% |
| PRMs are poor at *locating* the first error | Open PRMs vs QwQ-32B-Preview, ProcessBench | Prompted critics beat trained PRMs on F1 |
| MC-estimated labels are noisier than LLM-judge + human labels; BoN evaluation is biased toward the PRM | Qwen2.5-Math-PRM-7B/72B | Reported in Zhang et al. (2025) |
| Implicit PRMs come free from ORM training | 7B | Yuan et al. (2024) derive per-step rewards from an outcome-trained DPO-style model |

The last row matters: if a dense signal is recoverable from outcome-only training, the dichotomy is partly false.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published experiment compares process and outcome supervision at *equal annotation dollars and equal training FLOPs*, with pass@1 on a held-out family as the endpoint, at $\ge$7B scale. The experiment is runnable today for well under $10^5$ USD. Every headline comparison confounds label budget with label type.
- **Empirically open.** Whether process supervision's trace-validity advantage survives into long-CoT reasoning models that backtrack, where "step correctness" and "step usefulness" diverge.
- **Theoretically open.** No lower bound separating the two regimes for autoregressive policies. Setlur et al. give an upper-bound argument for prover-advantage rewards; there is no proof that *no* outcome-only estimator achieves the same rate, and implicit-PRM results suggest one might.
- **Methodologically blocked.** "Step correctness" has no prover-independent definition. $q^\mu_t$ changes with $\mu$; human validity labels measure something else. Until the target is defined, PRM-vs-ORM comparisons compare estimators of different estimands.

## 6. Why It Is Hard

Three specific obstructions, in order of severity.

1. **The evaluation does not measure what it names.** Best-of-$N$ with the PRM as reranker scores the verifier on its own training distribution and rewards it for correlating with the generator's failure modes. Zhang et al. (2025) show BoN rankings do not transfer to error-localization ability. A PRM can win BoN by 6 points and lose ProcessBench.
2. **Non-identifiability of the label.** The MC label $\hat q^\mu_t$ is a joint property of step and prover. Change $\mu$ from the base model to the RL'd policy mid-training and the labels move; this is exactly the drift that produces reward hacking in long RL runs.
3. **Cost confound.** $c_p/c_o \approx 10^2$. Any PRM advantage must be compared against spending the same money on $100\times$ more outcome-labeled problems — which no published study does.

Absent ground truth compounds all three: for a step in the middle of a 4,000-token trace, no cheap oracle says whether it was a good move.

## 7. Current Research (as of 2026)

- **Automatic process labels at scale.** OmegaPRM-style tree search over MC estimates (Google DeepMind, 2024) and Qwen's consensus filtering of MC labels against an LLM judge (Alibaba, 2025) — both aimed at label noise, not at the budget-matched question.
- **Implicit / free process rewards.** Deriving token-level rewards from outcome-trained models (Yuan et al., 2024); actively extended to GRPO-style pipelines *(frontier — verify)*.
- **Generative verifiers.** Framing verification as next-token prediction with CoT (Zhang et al., 2024), which reframes the PRM as a critic and sidesteps the step-boundary problem.
- **Outcome-only RL at frontier scale.** DeepSeek, and open replications (Open-R1, TÜLU-style pipelines), treating rule-based outcome reward as the default. The strongest current empirical position is "outcome supervision plus verifiable rewards is sufficient" *(frontier — verify; based on systems reports, not ablations)*.

## 8. Concrete Next Experiment

**Claim to test.** At matched annotation spend and matched RL compute, process supervision improves held-out pass@1 by $\ge 3$ points.

**Scale.** One 8B base model (e.g. Llama-3.1-8B or Qwen2.5-7B). Fixed annotation budget $B = 40{,}000$ USD of inference. Fixed RL budget: $2\times10^{20}$ FLOPs per arm. Total: 4 arms $\times$ 3 seeds = 12 runs, roughly 3,000 H100-hours.

**Arms (all at the same $B$):**
1. **Control:** outcome RL. Spend $B$ on outcome labels only — about $4\times10^5$ problems with checker verification.
2. **PRM-MC:** spend $B$ on $K=8$ MC rollouts per step over $\sim4\times10^3$ problems; train a PRM; dense reward in RL.
3. **PRM-implicit:** outcome labels only, dense reward derived implicitly from the ORM (Yuan et al.). Costs the same as arm 1.
4. **Mixed:** 90% of $B$ on outcome, 10% on process, dense reward only on steps with $|\hat A_t| > 0.3$.

**Evaluation.** Held-out family not used in training: e.g. Omni-MATH or an AIME-style set plus one non-math verifiable domain (competitive programming). Report pass@1, not best-of-$N$. Secondary: trace-error rate on 200 correct-answer samples, human-graded.

**The deciding number.** $\Delta = \text{pass@1}(\text{arm 2}) - \text{pass@1}(\text{arm 1})$ on the held-out family, with a bootstrap 95% CI over seeds. $\Delta \ge 3$ points with CI excluding 0 supports process superiority at matched budget; $|\Delta| < 1$ with a tight CI refutes it and reduces the question to trace validity. Arm 3 beating arm 2 refutes the dichotomy outright.

## 9. Key References

- **[Foundational]** Cobbe, Kosaraju, Bavarian, et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[Foundational]** Uesato, Kushman, Kumar, et al. *Solving math word problems with process- and outcome-based feedback.* 2022. — arXiv:2211.14275
- **[SOTA]** Lightman, Kosaraju, Burda, et al. *Let's Verify Step by Step.* ICLR 2024. — arXiv:2305.20050
- **[SOTA]** Wang, Li, Shao, et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL 2024. — arXiv:2312.08935
- **[SOTA]** Setlur, Nagpal, Fisch, et al. *Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning.* ICLR 2025. — arXiv:2410.08146
- **[SOTA]** Luo, Li, Wu, et al. *Improve Mathematical Reasoning in Language Models by Automated Process Supervision.* 2024. — arXiv:2406.06592
- **[Benchmark]** Zheng, Zhang, Zhang, et al. *ProcessBench: Identifying Process Errors in Mathematical Reasoning.* 2024. — arXiv:2412.06559
- **[Negative result]** Zhang, Zhang, Wang, et al. *The Lessons of Developing Process Reward Models in Mathematical Reasoning.* 2025. — arXiv:2501.07301
- **[Counter-evidence]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. — arXiv:2501.12948
- **[Method]** Yuan, Li, Wang, et al. *Free Process Rewards without Process Labels.* 2024. — arXiv:2412.01981
- **[Survey]** Casper, Davies, Shi, et al. *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* TMLR, 2023. — arXiv:2307.15217

## 10. Worked Example

A GSM8K-style problem, 4 steps, prover $\mu$ = the 8B base policy, $K=8$ rollouts per prefix.

| $t$ | Step | $\hat q^\mu_t$ | $\hat A_t$ | Human label |
|---|---|---|---|---|
| 1 | Set up total = 3 boxes $\times$ 12 | 0.50 | +0.10 | valid |
| 2 | Compute 3 $\times$ 12 = 36 | 0.75 | +0.25 | valid |
| 3 | Subtract the 4 broken items: 36 − 4 = 30 | 0.38 | −0.37 | **invalid** (arithmetic) |
| 4 | Answer: 30 | 0.38 | 0.00 | invalid |

The labels agree here. Now change one thing: make $\mu$ a stronger prover that self-corrects arithmetic. Rerunning gives $\hat q^\mu_3 = 0.81$, so $\hat A_3 = +0.06$ — the *same wrong step* is now labeled positive, because a strong prover recovers from it. The human label is unchanged.

**The obstruction, made numeric.** The step label flipped sign (−0.37 → +0.06) with no change to the step. In an RL run, $\mu$ drifts toward the policy being trained, so the PRM's targets drift with it. Meanwhile the cost: 4 steps $\times$ 8 rollouts $\times$ ~150 tokens = 4,800 generated tokens per problem for process labels, versus ~1 checker call for the outcome label. At $\$0.30$ per $10^6$ tokens that is $\$1.4\times10^{-3}$ vs $\$10^{-4}$ — so the same $\$40{,}000$ buys either $2.8\times10^7$ process-labeled problems' worth of rollouts spread over $\sim3\times10^4$ problems, or $4\times10^5$ outcome-labeled problems. The reported 5.8-point BoN gap (78.2 vs 72.4) was never charged this $100\times$ price difference, and the sign of the label is not stable under the one thing RL is guaranteed to change.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*