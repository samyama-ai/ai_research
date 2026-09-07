---
id: 35-world-models/action-space-abstraction-continuous-control
title: "Action Space Abstraction for Continuous Control Planning"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Action Space Abstraction for Continuous Control Planning

> **Topic:** World Models & Planning · **ID:** `35-world-models/action-space-abstraction-continuous-control` · **Status:** open

## 1. Problem Statement

Planning in a learned world model over raw continuous actions costs samples exponential in horizon × action dimension. A 38-DoF quadruped planned 3 steps ahead is a 114-dimensional optimization solved from scratch every control step. The proposed fix is an **action space abstraction**: replace the raw action space $\mathcal{A}\subset\mathbb{R}^{d_a}$ with a smaller latent space $\mathcal{Z}$ of temporally extended skills, plan in $\mathcal{Z}$, decode to raw actions.

The open problem: **does any learned abstraction buy a strict improvement in the return-per-unit-planning-compute frontier on continuous control, and can we say in advance which one?** Three variants, of different difficulty:

- **Measurement.** Define a comparison that isolates the abstraction from the confounds it travels with (extra pretraining data, changed exploration, changed effective horizon, changed action repeat). Currently not well posed.
- **Method.** Learn $(\mathcal{Z}, \text{decoder})$ from data such that planning in $\mathcal{Z}$ at budget $B$ beats planning in $\mathcal{A}$ at budget $B$ — same total compute, same data.
- **Theory.** Bound the return loss from restricting the policy class to decoder-realizable policies, in terms of measurable quantities of the abstraction, and bound the planning-cost saving. Both halves exist separately; a two-sided statement that predicts the net does not.

Solving it means: a stated abstraction-quality functional, computable from data before deployment, that predicts sign and magnitude of the planning-compute/return change on held-out tasks.

## 2. Formal Setting

MDP $M=(\mathcal{S},\mathcal{A},P,r,\gamma)$, $\mathcal{A}=[-1,1]^{d_a}$, learned model $\hat{P}_\theta$, learned reward $\hat r_\theta$, value $\hat V_\theta$. Measured quantities:

- $d_a$: raw action dimension. Measured by reading the environment spec (DMC `dog-run`: $d_a=38$; `humanoid`: $21$; Meta-World: $4$).
- Planner: sampling MPC (CEM/MPPI) with horizon $H$, $N$ trajectory samples, $I$ refinement iterations. **Planning budget** $B = N\cdot I\cdot H$ model rollout steps per control step — counted, not estimated, by instrumenting the rollout call.
- Abstraction $\mathcal{Z}\subset\mathbb{R}^{d_z}$ with decoder $\pi_\psi(a_t\mid s_t,z,\tau)$ executed for $c$ steps ($\tau=0..c-1$). Skill-level model $\hat{P}^{\mathcal{Z}}$ predicts $s_{t+c}$ from $(s_t,z)$. Abstract budget $B^{\mathcal{Z}}=N\cdot I\cdot H_z$ with $H_z=H/c$; the wall-clock comparison needs decoder forward passes counted too.
- **Realizable policy class** $\Pi_\psi=\{\,s\mapsto \pi_\psi(\cdot\mid s,z,\tau) : z\in\mathcal{Z}\,\}^{\text{(concatenated)}}$. The abstraction gap is
$$\Delta_{\text{repr}} \;=\; \max_{\pi\in\Pi}J(\pi)\;-\;\max_{\pi\in\Pi_\psi}J(\pi),$$
measured by *training to convergence* inside $\Pi_\psi$ with unlimited planning, versus a flat SAC/TD-MPC2 ceiling. This is the only honest estimator and it is expensive.
- **Planning gap** $\Delta_{\text{plan}}(B)=\max_{\pi\in\Pi_\psi}J(\pi)-J(\pi^{\text{MPC}}_{B})$: what the finite-budget planner loses relative to its own class.

Total: $J(\pi^\star)-J(\pi^{\text{MPC}}_B)=\Delta_{\text{repr}}+\Delta_{\text{plan}}(B)$. The abstraction wins iff $\Delta_{\text{repr}}$ grows slower than $\Delta_{\text{plan}}$ shrinks. **The whole problem is that only the sum is observed.**

Assumptions, with the ones violated in practice flagged:

1. $\hat P_\theta$ error is horizon-independent — **violated**; compounding error is why $H\le 5$ in practice, and it is exactly what skill-level models are claimed to fix.
2. The skill decoder is Markov in $(s,z,\tau)$ — **violated** when skills are learned from offline demonstrations with unmodelled history.
3. $\mathcal{Z}$ is identifiable — **violated**: for any invertible $g$, $(g(\mathcal{Z}),\pi_\psi\circ g^{-1})$ induces the same $\Pi_\psi$ but different planner geometry, so measured planner performance depends on a quantity the objective never constrained.
4. Reward is Markov at the skill timescale — violated for tasks with sub-$c$ contact events.

## 3. State of the Art

**Flat planning (the arm to beat).** TD-MPC2 (Hansen, Su, Wang, ICLR 2024) plans with MPPI at $H=3$ over raw actions in a learned latent model; single agent across 80+ tasks up to $d_a=38$ (DMC dog). *Established*: it is the strongest single-model result across DMC/Meta-World/MyoSuite at that scale. DreamerV3 (Hafner, Pasukonis, Ba, Lillicrap, *Nature* 2025) reaches comparable breadth with a learned actor and no online search.

**Hierarchical/abstract planning.** Director (Hafner et al., NeurIPS 2022, "Deep Hierarchical Planning from Pixels") plans over discrete latent *goals* (a goal autoencoder, abstract step $K=8$) inside a world model; solves sparse-reward Egocentric Ant Maze and visual pin-pad tasks where flat DreamerV2 gets ~0. *Established* for sparse-reward long-horizon tasks; *unablated* as a compute-matched claim on dense-reward DMC — Director does not beat flat agents there.

**Offline skill priors.** SPiRL (Pertsch, Lee, Lim, CoRL 2020): 10-step action sequences → 10-D latent; OPAL (Ajay, Kumar, Agrawal, Levine, Nachum, ICLR 2021): $c=10$ primitives, improves D4RL antmaze. *Established* as offline-RL improvements. *Claimed but confounded*: the gain is attributed to the abstraction, but the skill prior also injects demonstration data the flat baseline never sees.

**Goal-space abstraction.** HIRO (Nachum, Gu, Lee, Levine, NeurIPS 2018): high level emits raw state-space subgoals every $c=10$ steps; large gains on ant-maze-type tasks over flat baselines at ~10M steps.

**Temporal abstraction without new action spaces.** Fixed action repeat (DMC standard: 2) and TempoRL (Biedenkapp et al., ICML 2021) which learns a repetition count. This is the cheapest abstraction and the most frequently omitted control arm.

## 4. What Is Known

- **Sampling-planner cost scales with $H\cdot d_a$.** CEM/MPPI variance-based refinement needs sample counts growing with search dimension; iCEM (Pinneri et al., CoRL 2020) cut required samples by roughly an order of magnitude on 20–30-D MuJoCo tasks via colored-noise and memory, without changing the action space. Measured at $H=30$, $N\sim 10^2$–$10^3$.
- **Representation quality bounds sub-optimality.** Nachum, Gu, Lee, Levine (ICLR 2019, "Near-Optimal Representation Learning for Hierarchical RL") prove that the loss from restricting to a goal representation is bounded by $O\!\big(\tfrac{\gamma}{(1-\gamma)^2}\big)$ times an expected KL/total-variation mismatch between the true dynamics and the dynamics recoverable through the representation — a bound on $\Delta_{\text{repr}}$ only, with no companion bound on $\Delta_{\text{plan}}$.
- **Options can strictly reduce planning iterations.** Mann & Mannor (ICML 2014) show approximate value iteration with options converges in fewer iterations when option durations are long and models accurate. This is the theory side of the win.
- **Temporal abstraction can also strictly hurt.** Jong, Hester, Stone (AAMAS 2008, "The Utility of Temporal Abstraction in RL") showed options degrade learning when they truncate exploration of the primitive space — a reproduced negative result.
- **Hierarchy's measured benefit is mostly exploration, not planning.** Nachum et al. (2019, "Why Does Hierarchy (Sometimes) Work So Well in RL?") found the gains of HIRO-style hierarchy on ant-maze tasks were largely reproducible by giving a flat agent equivalent exploration/goal-relabeling, at ~10M-step scale.
- **Humanoid-scale control is not solved flat.** HumanoidBench (Sferrazza et al., 2024) reports DreamerV3 and flat model-free baselines near-zero on many of its 27+ whole-body tasks at 10M+ steps; the benchmark ships a hierarchical baseline with a pretrained low-level policy that does better — but with extra pretraining data.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No accepted compute-matched protocol. Published hierarchical results vary pretraining data, exploration, effective horizon and action repeat at once, so $\Delta_{\text{repr}}$ and $\Delta_{\text{plan}}$ are never separated. There is no standard estimator for $\Delta_{\text{repr}}$ at all.
- **Theoretically open.** A two-sided bound: no theorem states, for a learned $(\mathcal{Z},\pi_\psi)$ with measurable statistics, that $\Delta_{\text{plan}}(B^{\mathcal{Z}})+\Delta_{\text{repr}} < \Delta_{\text{plan}}(B)$. The two halves (Nachum 2019; Mann & Mannor 2014) have never been composed under one set of assumptions.
- **Empirically open.** Whether *any* learned abstraction beats flat TD-MPC2 on dense-reward DMC `dog`/`humanoid` at strictly equal rollout budget and equal data. Runnable today; ~$10^3$ GPU-hours; nobody has published it as a clean matched comparison.
- **Empirically open.** Whether learned skills beat the trivial abstraction (action repeat $c$, i.e. $\mathcal{Z}=\mathcal{A}$ with $d_z=d_a$) once budget is matched.

## 6. Why It Is Hard

**Confounded measurement plus non-identifiability.**

1. *Confound.* Changing the action space changes the exploration distribution, the effective planning horizon ($H_z\cdot c \ne H$), and the model's error profile at once. A reported win is compatible with $\Delta_{\text{repr}}<0$ being impossible (it cannot be) and with the entire gain coming from exploration — which is what Nachum et al. (2019) found when they checked.
2. *Non-identifiability.* The decoder objective (behaviour cloning, VAE ELBO, mutual-information skill discovery) is invariant to reparameterization of $\mathcal{Z}$, but the CEM/MPPI planner is not: Gaussian search in $\mathcal{Z}$ depends on the metric. Two abstractions with identical $\Pi_\psi$ and identical loss give different planning returns. So the learning objective does not target the quantity that decides the outcome.
3. *Absent ground truth.* $\Delta_{\text{repr}}$ requires knowing $\max_{\pi\in\Pi_\psi}J(\pi)$, which requires solving a restricted RL problem to convergence — as expensive as the original problem, per abstraction.
4. *Evaluation mismatch.* Skill-quality metrics in use (state coverage, mutual information $I(z;s_{t+c})$, reconstruction MSE) measure diversity, not planning gain. DADS (Sharma et al., ICLR 2020) maximizes exactly such an objective; high mutual information does not imply low $\Delta_{\text{repr}}$ on the reward at hand.

## 7. Current Research (as of 2026)

- **Whole-body humanoid control with pretrained low-level primitives** — HumanoidBench-derived hierarchical stacks and physics-based motion-prior methods (Berkeley RAIL, NVIDIA robotics). Strong empirical results, still data-confounded. *(frontier — verify)*
- **Skill spaces from large robot datasets** (Open X-Embodiment-scale action chunking; chunked action prediction as an implicit abstraction with $c\approx 8$–$50$). Action chunking is the abstraction that actually deployed, and its planning-compute effect is essentially unmeasured. *(frontier — verify)*
- **Scaling flat planners instead** — TD-MPC2-lineage work (Hansen et al.) arguing a single latent model with raw actions is sufficient given scale.
- **Metric-aware latent spaces** (bisimulation- and Lipschitz-regularized skill encoders) aimed directly at the non-identifiability in §6.2. Early, few compute-matched results. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** at strictly equal rollout budget and equal data, does a learned skill space beat raw actions and beat action repeat?

- **Scale.** 4 DMC tasks spanning $d_a$: `walker-run` (6), `humanoid-run` (21), `dog-run` (38), plus `humanoid-stand`. 3M environment steps, 5 seeds. Base agent: TD-MPC2 (same encoder, same latent model, same value function) — only the planner's search space changes. ~600–1000 A100-hours.
- **Arms** (all at identical measured budget $B=N I H=6144$ model rollout steps/decision, decoder passes included in the count):
  - **A. Flat control arm.** MPPI over raw actions, $H=3$, $N=512$, $I=4$.
  - **B. Trivial abstraction.** Raw actions with repeat $c=4$, $H_z=3$ → 12-step lookahead, budget reallocated to $N=512$, $I=4$.
  - **C. Learned skills.** $c=4$, $d_z=8$ VAE over action sequences trained on arm A's own replay buffer (no extra data), $H_z=3$, budget matched.
  - **D. C + isotropy regularizer** (unit-Lipschitz decoder w.r.t. $z$), to test §6.2.
- **Also measure $\Delta_{\text{repr}}$ directly:** for arms C and D, train SAC in the frozen latent space to convergence with no search. That ceiling is the estimator for $\max_{\pi\in\Pi_\psi}J(\pi)$.
- **Deciding number.** Normalized return on `dog-run` at 3M steps: **$J_C - J_A$, with a pre-registered threshold of $+50$ points (out of 1000), 5 seeds, bootstrap 95% CI excluding 0.** Secondary predicate: if $J_C \le J_B$, learned skills add nothing over action repeat and the field's positive results are horizon effects.

## 9. Key References

- **[Foundational]** R. Sutton, D. Precup, S. Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence 112(1–2), 1999.
- **[Foundational]** T. Mann, S. Mannor. *Scaling Up Approximate Value Iteration with Options: Better Policies with Fewer Iterations.* ICML 2014.
- **[Theory]** O. Nachum, S. Gu, H. Lee, S. Levine. *Near-Optimal Representation Learning for Hierarchical Reinforcement Learning.* ICLR 2019.
- **[Negative result]** N. Jong, T. Hester, P. Stone. *The Utility of Temporal Abstraction in Reinforcement Learning.* AAMAS 2008.
- **[Negative result]** O. Nachum, H. Tang, X. Lu, S. Gu, H. Lee, S. Levine. *Why Does Hierarchy (Sometimes) Work So Well in Reinforcement Learning?* 2019.
- **[SOTA, flat]** N. Hansen, H. Su, X. Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024.
- **[SOTA, hierarchical]** D. Hafner, K. Lee, I. Fischer, P. Abbeel. *Deep Hierarchical Planning from Pixels.* NeurIPS 2022.
- **[SOTA, model]** D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering Diverse Control Tasks through World Models.* Nature, 2025.
- **[Skills]** K. Pertsch, Y. Lee, J. Lim. *Accelerating Reinforcement Learning with Learned Skill Priors.* CoRL 2020.
- **[Skills]** A. Ajay, A. Kumar, P. Agrawal, S. Levine, O. Nachum. *OPAL: Offline Primitive Discovery for Accelerating Offline Reinforcement Learning.* ICLR 2021.
- **[Skills]** A. Sharma, S. Gu, S. Levine, V. Kumar, K. Hausman. *Dynamics-Aware Unsupervised Discovery of Skills.* ICLR 2020.
- **[Planner]** C. Pinneri et al. *Sample-efficient Cross-Entropy Method for Real-time Planning.* CoRL 2020.
- **[Benchmark]** C. Sferrazza et al. *HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation.* 2024.

## 10. Worked Example

DMC `dog-run`: $d_a=38$, episode 1000 steps, control at 50 Hz.

**Flat.** $H=3$, search dimension $3\times 38 = 114$. TD-MPC2 uses $N=512$, $I=6$ → $B=512\cdot6\cdot3=9216$ latent rollout steps per decision. Lookahead: 3 steps = 60 ms of physical time. A dog stride is ~400 ms, so the planner never sees a full gait cycle; the value function carries everything beyond 60 ms.

**Abstraction.** $c=4$, $d_z=8$. Search dimension $H_z\cdot d_z = 3\times 8 = 24$ — a **4.75×** reduction. Lookahead becomes 12 steps = 240 ms. Rollout budget at $N=512,I=6,H_z=3$ is 9216 *skill* steps, but each requires 4 decoder passes to score at raw resolution, so the honest matched setting is $N=512, I=6, H_z=3$ against flat $H=3$ with a decoder-cost surcharge of roughly $4\times$ on the low-level network. **Half the published comparisons omit this surcharge**; correcting for it typically halves $N$ and erases a 5–10% return gap.

**Where the obstruction bites.** Suppose arm C scores 520 and arm A scores 480 (normalized return, 5 seeds). Reading this as "the abstraction wins" requires that $\Delta_{\text{repr}}\approx 0$. Now run the direct estimator: train SAC to convergence inside the frozen 8-D skill space with no search. If that ceiling is 610 while flat TD-MPC2's asymptote is 800, then $\Delta_{\text{repr}}\approx 190$ — the abstraction has thrown away a fifth of achievable return, and the +40 observed is $\Delta_{\text{plan}}$ shrinking by ~230 while the ceiling dropped by 190. The measured $+40$ is a small difference of two large, oppositely-signed effects, each with seed noise of ±30. **The sign of the headline number is not stable under the confound it hides**, which is why the direct $\Delta_{\text{repr}}$ measurement in §8 — not the return comparison — is the part of the experiment that actually decides anything.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*