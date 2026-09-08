---
id: 35-world-models/goal-conditioned-planning-without-reward-engineering
title: "Goal-Conditioned Planning Without Reward Engineering"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Goal-Conditioned Planning Without Reward Engineering

> **Topic:** World Models & Planning · **ID:** `35-world-models/goal-conditioned-planning-without-reward-engineering` · **Status:** open

## 1. Problem Statement

**Input.** A learned or learnable world model of an environment, an unlabeled interaction dataset $\mathcal{D}$ (offline trajectories, or an online budget of environment steps), and at test time a goal specification $g$ — a target observation, a target image, a language string, or a set of states.

**Output.** A policy or planner $\pi(a \mid s, g)$ that reaches $g$ from arbitrary start states.

**Constraint.** No hand-designed reward, shaping term, distance metric, or per-task success predicate may be supplied for the test goals. The only supervision is the goal itself plus reward-free interaction.

**Decision predicate.** The problem is solved when a single agent, trained without task-specific reward, matches a per-task oracle trained with a hand-engineered dense reward on the same tasks, at the same environment-step budget, on tasks whose horizon exceeds the average trajectory length in $\mathcal{D}$ (stitching is required, not memorization).

Three variants pull apart:

- **Measurement.** How do you score "reached $g$" without a hand-written success predicate? Currently unresolved — every benchmark ships a hand-tuned threshold.
- **Method.** Given a success predicate, can planning in a learned model reach distant goals as well as reward-shaped RL? Empirically open at long horizon.
- **Theory.** Under what model-error and coverage conditions does goal-conditioned value learning give a bounded-suboptimality plan? Partially characterized, with the interesting regime uncovered.

## 2. Formal Setting

A reward-free controlled Markov process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, \rho_0, \gamma)$, plus a goal space $\mathcal{G}$ and a goal map $\phi: \mathcal{S} \to \mathcal{G}$. Goals are drawn from an evaluation distribution $p_{\text{eval}}(g)$.

The induced sparse reward is
$$r_g(s) = \mathbb{1}\!\left[\, d_{\mathcal{G}}(\phi(s), g) \le \epsilon \,\right],$$
and the objective is the discounted goal-hitting value
$$V^\pi(s, g) = \mathbb{E}_\pi\!\left[ \gamma^{\,T_g} \right], \quad T_g = \inf\{t : d_{\mathcal{G}}(\phi(s_t), g) \le \epsilon\}.$$

**The crux:** $d_{\mathcal{G}}$ and $\epsilon$ *are* the reward engineering. Claiming "no reward" while shipping a hand-tuned $(\phi, d_{\mathcal{G}}, \epsilon)$ moves the design work rather than removing it.

**Quantities as measured.**

- **Success rate** $\hat{S} = \frac{1}{N}\sum_i \mathbb{1}[T_{g_i} \le H]$ over $N \ge 50$ evaluation episodes per task, horizon $H$ fixed. Report a binomial CI; at $N=50$ the $\pm$ half-width near $\hat S = 0.5$ is $\approx 14$ points.
- **Optimal cost-to-go** $d^*(s,g) = \min_\pi \mathbb{E}[T_g]$, the ground-truth quasimetric. Measurable only in environments with a planner oracle (grid mazes, kinematic arms).
- **Plan suboptimality** $\Delta = \mathbb{E}_{s,g}[\hat{T}_g - d^*(s,g)]$, in environment steps.
- **Stitching load** $\kappa = d^*(s,g) / \bar{L}$, where $\bar{L}$ is mean dataset trajectory length. $\kappa > 1$ means no single training trajectory covers the task.
- **Model error** $\varepsilon_{\text{model}} = \mathbb{E}_{s,a \sim \mu} \, \mathrm{TV}(\hat{P}(\cdot|s,a), P(\cdot|s,a))$, estimated only under the data distribution $\mu$ — never under the planner's own visitation, which is the distribution that matters.
- **Compute budget** $C$: environment steps and gradient steps, both reported.

**Assumptions, and which break.**

| Assumption | Status in practice |
|---|---|
| $\phi$ is given and semantically correct | **Violated** — pixel goals make $\phi = \mathrm{id}$, and pixel distance is not task distance |
| Every state is reachable from every other ($\mathcal{M}$ communicating) | **Violated** — irreversible states (dropped object, fallen humanoid) |
| $\mathcal{D}$ has full coverage of the goal-relevant manifold | **Violated** — offline datasets are narrow; this is what makes stitching necessary |
| $p_{\text{eval}} = p_{\text{train}}$ over goals | Usually violated by design in benchmarks |
| Model error is uniform over the planning distribution | **Violated** — planners exploit exactly where $\hat P$ is wrong |

## 3. State of the Art

**Established (independently reproduced).**

- **Hindsight relabeling.** Andrychowicz et al., *Hindsight Experience Replay*, NeurIPS 2017. Turning any visited state into a goal makes sparse-reward manipulation learnable at all. Reproduced in dozens of codebases; it is the load-bearing trick in essentially every subsequent method.
- **Universal value approximation.** Schaul et al., *Universal Value Function Approximators*, ICML 2015; the idea traces to Kaelbling, *Learning to Achieve Goals*, IJCAI 1993.
- **Goal-conditioned supervised learning works without any value function.** Ghosh et al., *Learning to Reach Goals via Iterated Supervised Learning*, ICLR 2021 — but only when the goal is within the data's trajectory span.
- **Graph search over replay buffers beats flat policies at long horizon.** Savinov et al. (SPTM, ICLR 2018); Eysenbach et al., *Search on the Replay Buffer*, NeurIPS 2019.

**Claimed but not fully ablated.**

- **Contrastive RL** (Eysenbach et al., NeurIPS 2022) frames goal-reaching as classification of future states; reported roughly $2\times$ improvement over prior image-based goal-reaching baselines. The ablation separating representation quality from the relabeling scheme is thin.
- **Quasimetric value learning** (Wang et al., *Optimal Goal-Reaching RL via Quasimetric Learning*, ICML 2023) imposes the triangle inequality on $V$, which is the correct structural prior; gains are reported mostly on offline benchmark suites.
- **HIQL** (Park et al., *Hierarchical Implicit Q-Learning*, NeurIPS 2023) reports that a high-level subgoal policy read off a flat value function outperforms flat extraction; the mechanism claim ("the value gradient is noisy but its direction is not") is argued more than isolated.
- **Model-based variants:** LEXA (Mendonca et al., *Discovering and Achieving Goals via World Models*, NeurIPS 2021) trains explorer and achiever in a Dreamer world model over ~40 goals; Director (Hafner et al., *Deep Hierarchical Planning from Pixels*, NeurIPS 2022) plans in latent subgoal space. Both are reported as benchmark numbers on self-selected task sets.

**Benchmark-number-only.** OGBench (Park, Frans, Eysenbach, Levine; ICML 2025) is the current standard offline goal-conditioned suite and shows the field's actual state: existing methods degrade sharply on `humanoidmaze` and combinatorial `puzzle` tasks where $\kappa \gg 1$. No method dominates across the suite.

## 4. What Is Known

- **Sparse goal reward is trainable with relabeling.** On Fetch manipulation (7-DoF arm, 50-step episodes), DDPG alone reaches near-0% on Push and Pick-and-Place; DDPG+HER reaches high success on Push and non-trivial success on Pick-and-Place (Andrychowicz et al. 2017). Scale: ~$10^6$–$10^7$ environment steps, single robot arm, state observations.
- **Horizon is the failure axis, not dimensionality.** AntMaze-large ($\sim$700–1000-step tasks, D4RL) is where flat offline GCRL methods collapse and hierarchical/graph methods recover; OGBench's humanoid mazes push horizon further and knock all published methods down again. Scale: $10^6$-transition offline datasets.
- **Structural priors on $V$ help measurably.** Enforcing $V(s,g) \ge V(s,w) + V(w,g)$-style quasimetric constraints improves long-horizon offline results over unconstrained MLP critics (Wang et al. 2023).
- **Latent subgoals beat raw-pixel subgoals.** Director on pixel-based sparse tasks (Egocentric Ant Maze, visual pinpad) solves tasks that flat Dreamer-style agents do not.
- **Video/vision-pretrained distances are usable but weak reward proxies.** VIP (Ma et al., ICLR 2023) and R3M (Nair et al., CoRL 2022) produce embeddings whose distance correlates with progress on real-robot tasks; the correlation is far from monotone on failure trajectories.

## 5. What Is Not Known

- **Methodologically blocked — the measurement.** There is no goal-success predicate that is not itself engineered. Every reported "reward-free" success rate is computed with a hand-chosen $(\phi, d_{\mathcal{G}}, \epsilon)$. Until success is defined without a designer-chosen threshold, "without reward engineering" is unfalsifiable as stated.
- **Empirically open — parity with dense reward.** No published head-to-head at matched environment-step budget shows a goal-conditioned agent equalling a dense-reward per-task oracle on tasks with $\kappa \ge 3$. The experiment is runnable today.
- **Empirically open — does the world model help?** Whether latent planning beats model-free goal-conditioned value learning at equal compute is untested with the model as the only varied factor.
- **Theoretically open — sample complexity.** No tight bound on the number of reward-free samples needed to learn an $\alpha$-approximate quasimetric $\hat d$ such that greedy control is $O(\alpha)$-suboptimal, under partial coverage. Reward-free exploration bounds (Jin et al., ICML 2020) cover tabular/linear settings, not the quasimetric-with-function-approximation regime.
- **Theoretically open — identifiability.** Given only reward-free trajectories, the pair (goal metric, dynamics) is not uniquely determined; which extra assumption restores identifiability is unknown.

## 6. Why It Is Hard

**Primary obstruction: non-identifiability of the goal metric.** Reward-free data constrains $P$, not $d_{\mathcal{G}}$. For any strictly increasing $f$, the metric $f \circ d_{\mathcal{G}}$ induces the same ordering but a *different* $\epsilon$-ball, hence a different success set and a different optimal policy near the goal. The data cannot pick $f$. In pixel space this bites hard: two frames differing by a moved distractor can be farther apart in $\ell_2$ than a frame where the task object is misplaced.

**Secondary obstruction: an evaluation that does not measure what it names.** A benchmark "success rate" is the indicator of a threshold the benchmark author chose to make some reference method look reasonable. Tightening $\epsilon$ by $2\times$ can move reported success by tens of points without any change to the agent. Cross-paper comparisons therefore compare thresholds as much as agents.

**Tertiary: planner-induced distribution shift.** $\varepsilon_{\text{model}}$ is estimated on $\mu$; the planner searches for the trajectory that maximizes predicted goal-reaching, which is an adversarial search against $\hat P$. Error compounds as roughly $O(\varepsilon_{\text{model}} H^2)$ over an $H$-step rollout, so the failure is worst exactly where the problem is interesting — long $H$.

## 7. Current Research (as of 2026)

- **Benchmark repair.** OGBench (Berkeley RAIL / Princeton) is the coordination point; the live question is whether its evaluation protocol fixes the threshold problem or just standardizes it. *(frontier — verify)*
- **Quasimetric and contrastive representations** as the value backbone — Eysenbach's group (Princeton), Wang & Isola (MIT).
- **Foundation-model goal specification:** VLM-scored goal satisfaction replacing $\epsilon$-balls (e.g. success detectors built from vision-language models). Moves the engineering into pretraining rather than removing it. *(frontier — verify)*
- **Video-generation planners:** UniPi (Du et al., NeurIPS 2023) and SuSIE (Black et al., ICLR 2024) generate subgoal images and track them, sidestepping value learning entirely; long-horizon compounding is untested.
- **Hierarchical latent planning** — Director-lineage work at Google DeepMind; TD-MPC2 (Hansen et al., ICLR 2024) as the strong model-based control baseline to beat, though it is reward-driven.

## 8. Concrete Next Experiment

**Question.** At matched environment-step budget, does any reward-free goal-conditioned agent match a dense-reward oracle when $\kappa \ge 3$?

**Scale.** OGBench `antmaze-large` and `humanoidmaze-medium`, state observations. $2\times10^6$ offline transitions; $\bar{L} = 200$ steps by construction; select 40 evaluation goals with $d^*(s,g) \in [600, 1000]$ so $\kappa \in [3,5]$. $N = 100$ episodes per goal, $H = 2000$. 5 seeds. Cost estimate: ~600 GPU-hours on A100-class hardware for all arms.

**Arms.**
1. *Control (oracle):* per-goal SAC/IQL with a hand-shaped dense reward $-d^*(s_t, g)$ from the environment's true planner, same $2\times10^6$ steps.
2. Flat goal-conditioned: contrastive RL.
3. Hierarchical: HIQL.
4. Model-based: latent subgoal planner (Director-style) in a learned world model.
5. *Threshold-sensitivity probe:* rerun every arm's evaluation at $\epsilon$, $\epsilon/2$, $2\epsilon$.

**Deciding number.** The success-rate gap $G = \hat S_{\text{oracle}} - \max_{i \in \{2,3,4\}} \hat S_i$ at the benchmark's default $\epsilon$. $G \le 5$ points (with 5-seed CI excluding 10) means reward-free parity at $\kappa \ge 3$ and the method variant is closed. $G \ge 25$ points confirms the horizon gap is real and not a tuning artifact.

**Second number, and the one that matters more:** $\mathrm{range}_\epsilon(\hat S_i)$ across the three thresholds. If any arm's success rate moves more than $G$ itself when $\epsilon$ is halved, the benchmark is measuring the threshold, not the agent, and the measurement variant must be fixed before the method variant is meaningful.

## 9. Key References

- **[Foundational]** Kaelbling, L. P. *Learning to Achieve Goals.* IJCAI, 1993.
- **[Foundational]** Schaul, T., Horgan, D., Gregor, K., Silver, D. *Universal Value Function Approximators.* ICML, 2015.
- **[Foundational]** Andrychowicz, M. et al. *Hindsight Experience Replay.* NeurIPS, 2017. — arXiv:1707.01495
- **[Method]** Ghosh, D. et al. *Learning to Reach Goals via Iterated Supervised Learning.* ICLR, 2021.
- **[Method]** Eysenbach, B., Salakhutdinov, R., Levine, S. *Search on the Replay Buffer: Bridging Planning and Reinforcement Learning.* NeurIPS, 2019.
- **[Method]** Mendonca, R., Rybkin, O., Daniilidis, K., Hafner, D., Pathak, D. *Discovering and Achieving Goals via World Models.* NeurIPS, 2021.
- **[Method]** Hafner, D., Lee, K.-H., Fischer, I., Abbeel, P. *Deep Hierarchical Planning from Pixels.* NeurIPS, 2022.
- **[SOTA]** Eysenbach, B., Zhang, T., Levine, S., Salakhutdinov, R. *Contrastive Learning as Goal-Conditioned Reinforcement Learning.* NeurIPS, 2022.
- **[SOTA]** Wang, T., Torralba, A., Isola, P., Zhang, A. *Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning.* ICML, 2023.
- **[SOTA]** Park, S., Ghosh, D., Eysenbach, B., Levine, S. *HIQL: Offline Goal-Conditioned RL with Latent States as Actions.* NeurIPS, 2023.
- **[Benchmark]** Park, S., Frans, K., Eysenbach, B., Levine, S. *OGBench: Benchmarking Offline Goal-Conditioned RL.* ICML, 2025.
- **[Related]** Ma, Y. J. et al. *VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training.* ICLR, 2023.
- **[Related]** Du, Y. et al. *Learning Universal Policies via Text-Guided Video Generation.* NeurIPS, 2023.
- **[Theory]** Jin, C., Krishnamurthy, A., Simchowitz, M., Yu, T. *Reward-Free Exploration for Reinforcement Learning.* ICML, 2020.

## 10. Worked Example

**Setup.** A $50 \times 50$ discrete maze with walls, 4 actions, deterministic. State $= (x,y)$, goal $= (x_g, y_g)$, $\phi = \mathrm{id}$. Offline data: $10{,}000$ random-walk trajectories of length $\bar L = 200$. Test goal at $d^*(s_0, g) = 620$ steps, so $\kappa = 3.1$.

**Step 1 — no trajectory contains the answer.** A 200-step random walk in 2D covers a radius of about $\sqrt{200} \approx 14$ cells. The start–goal pair is 620 steps apart. Probability that any single trajectory contains both: effectively zero. Every method must stitch across $\ge 4$ trajectories.

**Step 2 — the metric is not identified.** The agent learns $\hat d$ from data. Suppose it learns Euclidean distance in $(x,y)$ instead of maze geodesic distance — perfectly consistent with local transition data, since locally the two agree. Take a state $s^\dagger$ that is 8 cells from $g$ in $\ell_2$ but on the far side of a wall, 210 steps away by geodesic. Greedy descent on $\hat d$ walks into the wall and stalls. Nothing in the reward-free data distinguishes the two metrics without global coverage of the wall structure.

**Step 3 — the threshold does the work.** Set $\epsilon = 1$ cell: greedy-on-$\ell_2$ success $= 0\%$ (it never crosses the wall). Set $\epsilon = 10$ cells: the same policy is scored *successful* from $s^\dagger$, because it terminates 8 cells away. Reported success jumps $0\% \to 61\%$ with no change to the agent.

**Step 4 — the number.** Oracle (BFS on the true maze): $\hat S = 100\%$, $\Delta = 0$. Greedy-on-learned-$\hat d$ at $\epsilon=1$: $\hat S \approx 39\%$, $G = 61$ points. At $\epsilon = 10$: $\hat S \approx 91\%$, $G = 9$ points. $\mathrm{range}_\epsilon(\hat S) = 52 \gg G$ at the loose threshold.

**What this makes visible.** Two of the three headline numbers — the method gap and the "reward-free parity" claim — are functions of $\epsilon$, a quantity the designer picked. That is the obstruction: the engineered reward was not removed, it was compressed into a scalar and moved into the evaluation harness, where it is no longer audited.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*