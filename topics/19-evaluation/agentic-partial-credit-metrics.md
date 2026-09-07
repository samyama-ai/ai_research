---
id: 19-evaluation/agentic-partial-credit-metrics
title: "Measuring Agentic Task Success Beyond Binary Completion"
topic: 19-evaluation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Measuring Agentic Task Success Beyond Binary Completion

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/agentic-partial-credit-metrics` · **Status:** methodologically-blocked

## 1. Problem Statement

Agent benchmarks score a rollout with one bit: did the final state satisfy a validator. Two failed rollouts — one that edited the wrong file, one that fixed the bug but broke a regression test — receive the same score, 0. The problem is to define and validate a **partial-credit function** on agent trajectories that carries information the bit does not.

Three variants, different difficulties:

- **Measurement variant (the blocked one).** Define $\phi(\tau) \in [0,1]$ over trajectories such that $\phi$ is *externally valid*: it predicts a quantity someone cares about — residual human effort to finish, probability of success given $k$ more steps, or dollars of downstream damage. Without an anchor, $\phi$ is an arbitrary rubric.
- **Method variant.** Given an accepted $\phi$, build cheap estimators (checkpoint validators, process reward models, LLM judges) that approximate it at benchmark scale. This is engineering and is partly solved.
- **Theory variant.** Characterize when a per-step credit assignment is identifiable from outcome-only data. This is a credit-assignment question and is theoretically open.

Solving it means: a metric that (a) reduces to the binary score on solved tasks, (b) ranks failures in an order that matches an independently measured cost, and (c) is not gameable by trajectory padding.

## 2. Formal Setting

A task instance $t$ drawn from benchmark distribution $\mathcal{D}$ has an initial environment state $s_0$ and a validator $V_t: \mathcal{S} \to \{0,1\}$ (a test suite, a database-state comparison, an exact-match string). An agent policy $\pi$ produces a trajectory
$$\tau = (s_0, a_1, s_1, \dots, a_T, s_T), \qquad T \le T_{\max},$$
at token cost $C(\tau) = \sum_i c_i$ and wall-clock $W(\tau)$. The binary score is $S(\tau) = V_t(s_T)$, and the reported headline is $\widehat{p} = \frac{1}{n}\sum_{j=1}^n S(\tau_j)$ over $n$ instances.

**Partial credit.** A metric $\phi_t: \mathcal{T} \to [0,1]$ with the consistency constraint $S(\tau)=1 \Rightarrow \phi_t(\tau)=1$. Three families, each with a concrete measurement recipe:

1. **Sub-goal / milestone decomposition.** A fixed ordered set $G_t = (g_1,\dots,g_m)$ of predicates $g_i:\mathcal{S}\to\{0,1\}$, hand-written per instance. Measured as $\phi^{\mathrm{prog}}(\tau) = \frac{1}{m}\max_{k\le T}\sum_{i} g_i(s_k)$, or as furthest-milestone-reached $\max\{i : g_i \text{ satisfied}\}/m$.
2. **Validator-internal fraction.** Where $V_t$ is a test suite of $q$ tests, $\phi^{\mathrm{test}}(\tau) = \frac{1}{q}\sum_{j} \mathbb{1}[\text{test}_j \text{ passes at } s_T]$. Measured directly from the harness; no extra annotation.
3. **Judge-assigned credit.** $\phi^{\mathrm{judge}}(\tau) = f_\theta(\tau, t)$ from an LLM judge or process reward model, measured as the judge's scalar output, calibrated against held-out labels.

**Validity anchor.** Let $H(\tau) \in \mathbb{R}_{\ge 0}$ be the *residual completion cost*: minutes of expert work to take $s_T$ to a state satisfying $V_t$. A partial-credit metric is valid on $\mathcal{D}$ at strength $\rho$ if
$$\rho = -\,\tau_b\!\left(\phi(\tau),\, H(\tau)\right) \ \ \text{(Kendall's } \tau_b \text{ over failed trajectories, } S(\tau)=0),$$
i.e. more credit means less residual work. Note the binary metric has $\rho = 0$ on this set by construction — it is constant there.

**Assumptions, and which are violated.**

- *Validator soundness*, $V_t(s)=1 \iff$ task truly done. Violated: SWE-bench's original test suites admit solutions that pass without fixing the issue, which is why SWE-bench Verified exists.
- *Milestone completeness*, that $G_t$ covers the paths a competent agent takes. Violated whenever an agent finds a shortcut; $\phi^{\mathrm{prog}}$ then scores a correct trajectory below a canonical wrong one.
- *Monotone progress*, that state advances toward the goal. Violated in environments with irreversible destructive actions, where $\max_k$ over the trajectory credits a state the agent later destroyed.
- *Cost-independence*, that $\phi$ is comparable across agents at different $C(\tau)$. Violated: an agent that spends $10\times$ the tokens touches more milestones.
- *Instance independence*, needed for the standard binomial error bar $\sqrt{\widehat p(1-\widehat p)/n}$. Violated when instances share a repository or a template.

## 3. State of the Art

**Established (reproduced, ablated).**

- Binary-outcome agent benchmarks with executable validators are the operational standard: SWE-bench (Jimenez et al., ICLR 2024), WebArena (Zhou et al., ICLR 2024), OSWorld (Xie et al., NeurIPS 2024), GAIA (Mialon et al., ICLR 2024).
- Process supervision beats outcome supervision for selecting correct math solutions — Lightman et al., *Let's Verify Step by Step* (ICLR 2024) — establishing that dense per-step signal is learnable and useful *when step labels exist*. 800K human step labels (PRM800K) were required.
- Repeated-sampling reliability metrics: $\text{pass}^k$ (all of $k$ i.i.d. rollouts succeed) in $\tau$-bench (Yao et al., 2024) exposes variance that $\text{pass}@1$ hides.
- Cost must be a reported axis, not an afterthought — Kapoor, Stroebl et al., *AI Agents That Matter* (2024): accuracy-only leaderboards reward retry-until-pass strategies.

**Claimed but unablated.**

- **Progress rate** as a fine-grained agent metric (AgentBoard, Ma et al., NeurIPS 2024 D&B). The construct is well-specified and the sub-goals are hand-annotated across 9 environments; what has *not* been shown is that progress rate predicts anything external — no study anchors it to residual human effort or to success-under-more-compute.
- **Agent-as-a-Judge** (Zhuge et al., 2024) scores intermediate requirements on the DevAI set (55 tasks, 365 hierarchical requirements) and reports higher agreement with human labels than LLM-as-a-Judge. Single dataset, one domain, no cross-lab replication.
- Test-pass fraction as partial credit on SWE-bench-style tasks: reported in individual system papers, **exists only as a benchmark number**; no published validation against repair effort.

**Theory SOTA** is essentially absent: no identifiability result tells you when per-step credit is recoverable from outcome-only rollouts.

## 4. What Is Known

- **Binary headline numbers are low and leave most of the distribution unresolved.** OSWorld at release: humans 72.36% vs. best agent 12.24% on 369 real-computer tasks — i.e. ~88% of rollouts collapse to a single undifferentiated "0". WebArena's initial GPT-4 agent scored 14.41% on 812 tasks against a 78.24% human rate.
- **Reliability, not capability, is often the binding constraint.** On $\tau$-bench (retail, ~115 tasks), a frontier model's $\text{pass}^1$ near 60% falls to roughly a quarter of that at $\text{pass}^8$ — an axis binary $\text{pass}@1$ actively hides.
- **Validators are wrong often enough to matter.** SWE-bench Verified (OpenAI, 2024) is a 500-instance human-filtered subset built because a substantial fraction of the original 2,294 instances had underspecified issues or overly narrow tests; scores roughly double when moving to the filtered set for the same system.
- **Step labels are expensive.** PRM800K: ~800K step-level labels for one narrow domain (MATH). No agentic benchmark has annotation at that density.
- **Eval error bars are usually absent or wrong.** Miller (2024) shows standard leaderboard gaps at $n \approx 500$ are frequently within clustered-sampling error.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no agreed anchor $H(\tau)$. Nobody has published a dataset of failed agent trajectories with measured expert repair effort, so no proposed $\phi$ has ever been validated rather than asserted. Every partial-credit metric currently in use is defined by fiat.
- **Empirically open.** Whether $\phi^{\mathrm{prog}}$ (milestones), $\phi^{\mathrm{test}}$ (test fraction), and $\phi^{\mathrm{judge}}$ agree on agent *rankings*. Measurable today on any existing benchmark; the Kendall $\tau_b$ between the three ranking vectors has not been reported. If it is high, the choice of $\phi$ does not matter; if low, published progress-rate comparisons are rubric artifacts.
- **Empirically open.** Whether partial credit predicts success at larger $T_{\max}$ — does $\phi$ at 30 steps forecast $S$ at 300?
- **Theoretically open.** Identifiability: given only outcome labels over a rollout distribution, when is a per-step credit function unique up to potential-shaping $\Phi(s') - \Phi(s)$? Reward shaping theory (Ng, Harada, Russell, ICML 1999) says policy-invariant shapings form an equivalence class — so credit is *not* identifiable without extra assumptions. What those assumptions are for agent evaluation is unstudied.
- **Theoretically open.** Gaming bounds: no result characterizes the maximum $\phi$ a policy can obtain while having $\Pr[S=1]=0$.

## 6. Why It Is Hard

Three named obstructions, in order of bite.

1. **Absent ground truth.** Partial credit is a claim about counterfactual remaining work. That quantity is only observable by paying a human to finish the trajectory, and it is path-dependent — repairing a half-done wrong refactor can cost more than starting over, so $H$ is not monotone in "amount of correct work done."
2. **Non-identifiability.** Ng et al. (1999) implies the space of credit assignments consistent with a fixed outcome distribution is at least as large as the space of potential functions on states. Two rubrics can be equally consistent with all observed outcomes and induce opposite agent rankings.
3. **The metric does not measure what it names.** "Progress" computed as $\max_k$ over a trajectory rewards exploration breadth: an agent that touches many files accumulates milestone hits without approaching the goal, while a one-shot correct patch touches almost none. Absent a cost control, $\phi^{\mathrm{prog}}$ partially measures $C(\tau)$.

Compute cost is *not* the obstruction here — the rollouts already exist. Annotation cost is.

## 7. Current Research (as of 2026)

- **Rubric-and-checkpoint benchmarks.** AgentBoard-style progress rates and Agent-as-a-Judge-style requirement trees are being folded into new agentic suites; the direction is active across academic groups (Shanghai AI Lab / HKU for AgentBoard, KAUST/Meta for Agent-as-a-Judge).
- **Reliability metrics over point accuracy.** $\text{pass}^k$ (Sierra), cost-controlled Pareto reporting and the Holistic Agent Leaderboard line from Princeton (Kapoor, Stroebl, Narayanan).
- **Process reward models transferred from math to code and tool use**, with automatic step labels from rollout-completion (MCTS-style) rather than humans *(frontier — verify: transfer results outside math remain thin)*.
- **Long-horizon time-based scoring** — expressing capability as the human time-length of tasks an agent completes at 50% reliability (METR, 2025) — is the most serious attempt at an external anchor, but it anchors *task difficulty*, not *partial progress within a failed task*.
- **Item response theory for evals** (tinyBenchmarks, Polo et al., ICML 2024) gives a latent-ability estimate from binary responses; extending IRT to graded responses over agent trajectories is unexplored *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does any existing partial-credit metric predict residual human effort better than chance?

**Scale.** SWE-bench Verified, $n = 500$. Run three agent scaffolds of clearly different quality (e.g. a strong frontier scaffold, a mid-tier one, a deliberately weak single-shot one), 3 seeds each. Retain failed trajectories only; sample 200 of them, stratified across scaffolds.

**Anchor.** For each sampled failure, give an expert Python developer the repo at final state $s_T$ plus the agent's diff and log, and measure $H(\tau)$ = minutes to reach a passing state, capped at 60 (censored). Two annotators on a 40-instance overlap for inter-rater reliability. Budget: ~200 × 25 min ≈ 85 expert-hours.

**Arms.** $\phi^{\mathrm{test}}$ (fraction of FAIL_TO_PASS + PASS_TO_PASS tests passing), $\phi^{\mathrm{prog}}$ (5-milestone hand rubric: right file located, right function edited, patch applies, no regression, target test attempted), $\phi^{\mathrm{judge}}$ (frontier LLM scoring the trajectory 0–1).

**Control arm.** A **cost-only predictor** $\phi^{\mathrm{cost}}(\tau) = -C(\tau)$, plus a **scaffold-identity predictor** (which of the three agents produced it). If a rubric cannot beat these, it is measuring effort or brand, not progress.

**Deciding number.** Partial Kendall $\tau_b$ between $\phi$ and $-H$, controlling for $C(\tau)$ and scaffold identity. **Ship the metric if $\tau_b \ge 0.35$ with a bootstrap 95% CI excluding 0.15; declare it invalid if the CI includes 0.** With $n=200$, the standard error on $\tau_b$ is roughly $0.05$, so the design separates 0.35 from 0.15 comfortably.

**Secondary readout.** Kendall $\tau_b$ between the three metrics' *agent rankings*. If below 0.6, published progress-rate comparisons are rubric-dependent and should not be compared across papers.

## 9. Key References

- **[Foundational]** Andrew Y. Ng, Daishi Harada, Stuart Russell. *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML, 1999.
- **[Foundational]** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Foundational]** Shuyan Zhou et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR, 2024. — arXiv:2307.13854
- **[SOTA]** Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, Junxian He. *AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2401.13178
- **[SOTA]** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[SOTA]** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan. *$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[SOTA]** Tianbao Xie et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS, 2024. — arXiv:2404.07972
- **[SOTA]** Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, Yangyang Shi, Vikas Chandra, Jürgen Schmidhuber. *Agent-as-a-Judge: Evaluate Agents with Agents.* 2024. — arXiv:2410.10934
- **[Survey/Position]** Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, Arvind Narayanan. *AI Agents That Matter.* 2024. — arXiv:2407.01502
- **[Method]** Evan Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024. — arXiv:2411.00640
- **[Method]** Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, Mikhail Yurochkin. *tinyBenchmarks: Evaluating LLMs with Fewer Examples.* ICML, 2024. — arXiv:2402.14992
- **[Context]** Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, Thomas Scialom. *GAIA: A Benchmark for General AI Assistants.* ICLR, 2024. — arXiv:2311.12983

## 10. Worked Example

Take one SWE-bench Verified instance whose validator has $q = 14$ tests (2 FAIL_TO_PASS, 12 PASS_TO_PASS). Three failed trajectories:

| Trajectory | Tests passing at $s_T$ | $\phi^{\mathrm{test}}$ | Milestones hit (of 5) | $\phi^{\mathrm{prog}}$ | Steps / tokens | Measured $H$ (min) |
|---|---|---|---|---|---|---|
| A — one-line patch, wrong sign | 12/14 | 0.857 | 5 | 1.00 | 11 / 40K | 3 |
| B — correct fix, breaks 3 regressions | 11/14 | 0.786 | 4 | 0.80 | 24 / 120K | 8 |
| C — no patch, explored 19 files | 12/14 (baseline; nothing changed) | 0.857 | 2 | 0.40 | 61 / 310K | 45 |

The obstruction is visible in one column. $\phi^{\mathrm{test}}$ scores **C exactly equal to A** at 0.857, because a no-op agent inherits the repository's pre-existing 12 passing tests. Test-pass fraction has a large task-dependent floor $q_{\text{pass}}(s_0)/q$ — here $12/14 = 0.857$ — so its usable dynamic range on this instance is $[0.857, 1.0]$, i.e. 14% of the nominal $[0,1]$. Across a suite where that floor varies from 0.2 to 0.98, averaging $\phi^{\mathrm{test}}$ over instances mostly averages the floors, not the agents.

Rescaling to $\left(\phi^{\mathrm{test}} - \text{floor}\right)/(1-\text{floor})$ fixes the floor but flips the ordering: A gets $0$ (it changed nothing that the suite detects, having passed 12 before and 12 after), B gets $-1.0$ (it *lost* a test), C gets $0$. Now the trajectory that is 3 minutes from done and the one that is 45 minutes from done are again tied, and the genuinely closest-to-correct trajectory scores worst.

$\phi^{\mathrm{prog}}$ orders them A > B > C, which matches $H$ (3 < 8 < 45) — but only because the 5-milestone rubric was written by someone who already knew the reference patch. Its rank correlation with $H$ is confounded with $-C(\tau)$: token counts 40K < 120K < 310K give the *same* ordering. On this instance the control arm of §8 is indistinguishable from the rubric. That is the entire problem in three rows: no proposed $\phi$ has yet been shown to beat "the agent that spent fewer tokens was closer."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*