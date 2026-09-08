---
id: 35-world-models/hierarchical-planning-learned-subgoal-spaces
title: "Hierarchical Planning Over Learned Subgoal Spaces"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hierarchical Planning Over Learned Subgoal Spaces

> **Topic:** World Models & Planning · **ID:** `35-world-models/hierarchical-planning-learned-subgoal-spaces` · **Status:** open

## 1. Problem Statement

Given a learned world model of a long-horizon environment, learn a **subgoal space** $\mathcal{G}$ and an abstract transition model over it, such that planning in $\mathcal{G}$ is both (a) cheaper than planning in the primitive action space by a large factor and (b) near-optimal with respect to the primitive MDP. Nobody has shown a learned $\mathcal{G}$ that delivers both at once outside toy or hand-shaped domains.

Three variants, routinely conflated:

- **Method.** Produce an algorithm that discovers $\mathcal{G}$ from data (no privileged coordinates, no hand-designed goal projection) and beats a strong flat model-based agent on tasks with $\ge 10^3$-step horizons.
- **Measurement.** Define a metric that isolates the *hierarchy's* contribution from the exploration bonus, representation learning, and extra parameters it comes bundled with. Currently there is no accepted one; most reported gains are attributable to exploration (§4).
- **Theory.** Bound the suboptimality of the induced hierarchical policy as a function of measurable abstraction error, without assuming the subgoal map is bisimulation-preserving.

Solving it means: on a fixed environment suite, a learned-$\mathcal{G}$ agent matches or beats the flat control arm at equal environment steps *and* equal planning FLOPs, with an ablation showing the gain survives when the flat arm gets the same exploration bonus.

## 2. Formal Setting

MDP $M = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$, horizon $H$. A learned world model $\hat{P}_\theta$ operates on latents $z_t \in \mathcal{Z}$ from encoder $q_\phi(z_t \mid o_{\le t})$.

**Subgoal map.** $f_\psi : \mathcal{Z} \to \mathcal{G}$, with $\mathcal{G}$ either continuous ($\mathbb{R}^d$) or discrete (e.g. $C$ categoricals of $K$ classes, $|\mathcal{G}| = K^C$). Measured as: the actual output dimension and, for continuous $\mathcal{G}$, the participation ratio of the covariance eigenvalues over a held-out rollout buffer — the *effective* $d$, which is usually well below nominal $d$.

**Low-level policy.** $\pi_{\text{lo}}(a \mid z, g)$ run for $K$ steps. Its competence is the empirical **subgoal reachability**
$$p_{\text{reach}} = \Pr\big[\, \| f_\psi(z_{t+K}) - g \| \le \epsilon_g \,\big],$$
measured over $\ge 10^3$ sampled $(z_t, g)$ pairs drawn from the manager's own visitation distribution, not from a uniform prior — the distribution shift between these two is itself a large effect and is rarely reported.

**Abstract model.** $\bar{P}(g' \mid g, \bar{a})$ over $\mathcal{G}$, with abstract reward $\bar{r}(g,g') = \mathbb{E}\big[\sum_{i=0}^{K-1}\gamma^i r_{t+i}\big]$. This is the options/SMDP construction of Sutton, Precup & Singh (1999) with the option set induced by $f_\psi$.

**Objective.** With $\bar\pi$ the manager and $\pi = \bar\pi \circ \pi_{\text{lo}}$ the induced flat policy,
$$\Delta = \max_{s} \big| V^{*}_{M}(s) - V^{\pi}_{M}(s) \big|,$$
and the compute ratio $\rho = \text{FLOPs}_{\text{flat plan}} / \text{FLOPs}_{\text{hier plan}}$. Solving means large $\rho$ at small $\Delta$.

**Abstraction error.** Define $\varepsilon_{\text{abs}} = \sup_{z,\bar a} D_{\mathrm{KL}}\!\big(P(f_\psi(z')\mid z,\bar a)\,\|\,\bar P(\cdot \mid f_\psi(z),\bar a)\big)$. Standard state-abstraction results (Li, Walsh & Littman 2006; Ferns et al. 2004) give $\Delta \lesssim \frac{2\gamma}{(1-\gamma)^2}\,\varepsilon$ under a *model-similarity* abstraction.

**Assumptions, and which break.**
1. $\mathcal{G}$ is Markov — **violated**: learned $f_\psi$ discards history that $\bar P$ needs; the abstract process is typically non-Markov and this is not measured.
2. $\pi_{\text{lo}}$ is fixed while the manager trains — **violated** in every joint-training method; HIRO's off-policy relabeling exists precisely because it fails.
3. Every $g \in \mathcal{G}$ is reachable — **violated**: managers routinely emit off-manifold goals; the fraction is rarely reported.
4. $K$ is a good temporal scale — **violated**: $K$ is a hand-set hyperparameter in almost all deep HRL, which quietly reimports the human prior the method claims to remove.

## 3. State of the Art

**Empirical SOTA.** *Director* (Hafner, Lee, Fischer & Abbeel, NeurIPS 2022) plans in the discrete latent code space of a DreamerV2 world model: a manager picks goals every $K=8$ steps, a goal autoencoder keeps them on-manifold, and the worker is trained on a goal-reaching reward. It solves egocentric visual Ant Maze and a visual pin-pad memory task where flat model-based baselines score near zero. *Established*: the sparse-reward egocentric mazes are not solved by the flat Dreamer arm. *Claimed but unablated*: that the win comes from hierarchy rather than from the goal-autoencoder's exploration pressure — no arm gives the flat agent an equivalent intrinsic bonus.

**Skill-space planning.** OPAL (Ajay et al., ICLR 2021) and SPiRL (Pertsch et al., CoRL 2020) learn a continuous latent skill space from offline data and plan/RL over it; both report large gains on long-horizon D4RL kitchen/maze tasks. *Established for offline data with demonstration structure*; the skill prior is doing much of the work, and neither shows the benefit persists when the offline data is not demonstration-shaped.

**Flat control arm.** TD-MPC2 (Hansen, Su & Wang, ICLR 2024) is the honest control: a single flat latent-space MPC agent covering 104 tasks across 4 domains. Any hierarchy claim should be measured against it, and most are not.

**Symbolic SOTA.** Konidaris, Kaelbling & Lozano-Pérez (JAIR 2018) prove that for a given option set, the *provably sufficient* symbols for planning are the preconditions and image sets of the options — a rare case where the abstraction is derived, not chosen. It requires the option set as input, so it does not solve discovery.

**Theory SOTA.** Nachum et al. (ICLR 2019) bound hierarchical suboptimality by a representation-error term and derive a learning objective from the bound; the bound is loose and the empirical gains it explains are small.

## 4. What Is Known

- **Options provably reduce planning iterations** when they are long and have low value error (Mann & Mannor, ICML 2014): approximate value iteration with options converges in fewer iterations, with an error floor set by option quality. The tradeoff is explicit — longer options, faster convergence, worse floor.
- **Most measured HRL gains are exploration, not modularity.** Nachum et al. (2019, arXiv:1909.10618) ablated HIRO-style agents on Ant Maze/Push/Fall and found the benefit is largely explained by better exploration and multi-task-like training signal; giving a flat agent a comparable exploration mechanism closed much of the gap. This is the single most important reproduced negative result in the area.
- **Goal relabeling is necessary for joint training.** HIRO (Nachum et al., NeurIPS 2018) shows off-policy correction of manager transitions is required because $\pi_{\text{lo}}$ is non-stationary; HAC (Levy et al., ICLR 2019) reaches the same conclusion via hindsight at 3 levels.
- **Learned $\mathcal{G}$ underperforms hand-specified $\mathcal{G}$.** In HIRO the goal space is the raw $x,y$ (or full state) coordinates — privileged. Replacing it with a learned map costs performance; Nachum et al. (ICLR 2019) recover part but not all of it at the scale of $\sim 10^7$ Ant Maze steps.
- **Non-hierarchical replay-graph search works.** SoRB (Eysenbach et al., NeurIPS 2019) does Dijkstra over replay-buffer states with a learned distance and beats flat goal-conditioned RL on long-horizon navigation — a subgoal method that avoids learning $\mathcal{G}$ at all, which is evidence that the *space* is not where the value is.
- **Scale of all of the above:** $10^6$–$10^8$ environment steps, MuJoCo locomotion / DMLab-scale pixels, single-digit task suites. No result here is at the scale of a general video world model.

## 5. What Is Not Known

- **Methodologically blocked (the core one).** There is no accepted measurement that separates the hierarchy's contribution from the exploration bonus, the added parameters, and the auxiliary representation loss it ships with. Until $\rho$ and $\Delta$ are reported jointly at matched exploration, "hierarchy helps" is not a testable claim. This is why the 2019 negative result has never been decisively answered.
- **Empirically open.** Whether any learned $\mathcal{G}$ beats a privileged hand-specified $\mathcal{G}$ on the same task. Runnable today; nobody has run it as a clean head-to-head at $\ge 10^8$ steps.
- **Empirically open.** Whether discovered temporal scale ($K$ learned, not set) ever matches hand-tuned $K$.
- **Theoretically open.** Any bound on $\Delta$ for a *non-Markov* learned $\mathcal{G}$ — i.e. when assumption 1 fails, which is the practical case. Existing bounds all assume a model-similarity or bisimulation-style abstraction.
- **Theoretically open.** Identifiability: whether the subgoal space is determined at all by the objective (§6).

## 6. Why It Is Hard

**Non-identifiability.** For a fixed low-level policy class, many subgoal maps $f_\psi$ induce the same optimal flat policy and the same return. The learning objective — return, or reconstruction, or reachability — is invariant to a large family of reparameterizations of $\mathcal{G}$ (any diffeomorphism composed with a matched $\pi_{\text{lo}}$). So gradient descent has no pressure toward the *planning-useful* $\mathcal{G}$ specifically, only toward some $\mathcal{G}$ that supports the current task. This is why learned goal spaces transfer poorly and why ablations of $\mathcal{G}$ are usually flat.

**Confounded measurement.** Every deep HRL method changes at least four things at once relative to its flat baseline: exploration distribution, parameter count, auxiliary losses, and the action space of the outer loop. Published comparisons vary all four. The 2019 ablation is the only serious attempt to hold them fixed, and it came back negative.

**Compounding reachability.** $\Delta$ is not linear in $p_{\text{reach}}$ — it is exponential in the number of subgoals (§10). A subgoal space is only useful if $\pi_{\text{lo}}$ is nearly perfect, and $\pi_{\text{lo}}$ is trained on the manager's own shifting goal distribution.

## 7. Current Research (as of 2026)

- **World-model-native hierarchy.** Director-lineage work inside DreamerV3-scale models; the open question is whether discrete latent codes are a usable subgoal space at video scale *(frontier — verify)*.
- **LLM-proposed subgoals.** Using a language model to emit subgoal text, grounded by a learned reachability critic — sidesteps discovery by importing a human prior. Widely deployed in embodied agents; whether it constitutes *learned* $\mathcal{G}$ is contested *(frontier — verify)*.
- **Bilevel planning with invented predicates** (Silver, Chitnis et al., AAAI 2023 and follow-ons at MIT/Brown): learn symbolic operators plus continuous samplers, then plan symbolically. Strongest current answer to identifiability, because the predicates are scored by *planning* utility rather than return.
- **Diffusion planners** as an alternative to hierarchy: generate whole trajectories, no subgoal space needed. If these keep improving, the case for learned $\mathcal{G}$ weakens rather than strengthens.

## 8. Concrete Next Experiment

**Question.** Does a *learned* subgoal space contribute anything beyond exploration?

**Scale.** Egocentric visual Ant Maze XL and DMLab-30-style long-horizon navigation, $3\times 10^7$ environment steps, 5 seeds, ~2–4 GPU-days per run at DreamerV3-XS size. Fits on 8 A100s in under a week.

**Arms** (identical parameter count, identical world model, identical wall-clock budget):
1. Director-style learned $\mathcal{G}$ (goal autoencoder over latent codes).
2. **Control arm:** flat DreamerV3 agent with an intrinsic bonus matched to arm 1's exploration — matched by measured state-visitation entropy over a fixed discretization, not by hyperparameter guess.
3. Privileged $\mathcal{G}$: goals are ground-truth $(x,y)$.
4. Random-projection $\mathcal{G}$: $f_\psi$ frozen at a random linear map of the latent, everything else identical.

**Deciding number.** The gap in final success rate between arm 1 and arm 4, at matched visitation entropy. If arm 1 minus arm 4 is $< 5$ points (with 5 seeds, ~ the noise floor), then the *learning* of the subgoal space contributes nothing measurable and the field's premise is wrong; the gain is temporal abstraction plus exploration, obtainable with a random projection. If the gap exceeds 15 points and arm 1 approaches arm 3, subgoal-space learning is real and the target becomes closing the remaining gap to privileged goals.

## 9. Key References

- **[Foundational]** R. Sutton, D. Precup, S. Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence 112(1–2), 1999.
- **[Foundational]** P. Dayan, G. Hinton. *Feudal Reinforcement Learning.* NIPS, 1992.
- **[Foundational]** L. Li, T. Walsh, M. Littman. *Towards a Unified Theory of State Abstraction for MDPs.* ISAIM, 2006.
- **[Foundational]** N. Ferns, P. Panangaden, D. Precup. *Metrics for Finite Markov Decision Processes.* UAI, 2004.
- **[SOTA]** D. Hafner, K.-H. Lee, I. Fischer, P. Abbeel. *Deep Hierarchical Planning from Pixels.* NeurIPS, 2022. — arXiv:2206.04114
- **[SOTA]** O. Nachum, S. Gu, H. Lee, S. Levine. *Data-Efficient Hierarchical Reinforcement Learning.* NeurIPS, 2018. — arXiv:1805.08296
- **[Theory]** O. Nachum, S. Gu, H. Lee, S. Levine. *Near-Optimal Representation Learning for Hierarchical Reinforcement Learning.* ICLR, 2019. — arXiv:1810.01257
- **[Negative result]** O. Nachum, H. Tang, X. Lu, S. Gu, H. Lee, S. Levine. *Why Does Hierarchy (Sometimes) Work So Well in Reinforcement Learning?* arXiv:1909.10618, 2019.
- **[Theory]** T. Mann, S. Mannor. *Scaling Up Approximate Value Iteration with Options: Better Policies with Fewer Iterations.* ICML, 2014.
- **[Symbolic]** G. Konidaris, L. Kaelbling, T. Lozano-Pérez. *From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning.* JAIR 61, 2018.
- **[Related]** A. Levy, G. Konidaris, R. Platt, K. Saenko. *Learning Multi-Level Hierarchies with Hindsight.* ICLR, 2019. — arXiv:1712.00948
- **[Related]** B. Eysenbach, R. Salakhutdinov, S. Levine. *Search on the Replay Buffer: Bridging Planning and Reinforcement Learning.* NeurIPS, 2019. — arXiv:1906.05253
- **[Related]** A. Ajay, A. Kumar, P. Agrawal, S. Levine, O. Nachum. *OPAL: Offline Primitive Discovery for Accelerating Offline Reinforcement Learning.* ICLR, 2021. — arXiv:2010.13611
- **[Control arm]** N. Hansen, H. Su, X. Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR, 2024. — arXiv:2310.16828
- **[Survey]** S. Pateria, B. Subagdja, A.-H. Tan, C. Quek. *Hierarchical Reinforcement Learning: A Comprehensive Survey.* ACM Computing Surveys 54(5), 2021.

## 10. Worked Example

Egocentric Ant Maze XL. Episode length $H = 1000$ primitive steps, $K = 25$, so the manager issues $N = 40$ subgoals per episode.

**The compute win is real.** A flat CEM planner over 1000 steps with 5 CEM iterations and 500 samples costs $\approx 2.5\times 10^6$ model rollout-steps per replan. Planning over 40 subgoals with the same budget costs $\approx 10^5$: $\rho \approx 25$. This is the number hierarchy is sold on.

**The correctness loss is the obstruction.** Take a measured $p_{\text{reach}}$ from the manager's own goal distribution. Success requires the ant to reach essentially every subgoal (failing one strands it in a maze corridor):

| $p_{\text{reach}}$ | $p_{\text{reach}}^{40}$ = episode success |
|---|---|
| 0.85 | 0.0015 |
| 0.90 | 0.0148 |
| 0.95 | 0.129 |
| 0.98 | 0.446 |
| 0.99 | 0.669 |

Published low-level policies on Ant navigation sit around $p_{\text{reach}} \approx 0.85$–$0.95$ for *in-distribution* goals, and lower for the off-manifold goals a freely-parameterized manager emits. To get a usable 45% task success at $N=40$ the worker must hit 0.98 — a regime nobody reports.

**Why this is the obstruction and not a tuning issue.** The two knobs fight. Raise $K$ to 100, and $N$ drops to 10, so 0.90 reachability gives $0.9^{10} = 0.35$ success — better. But longer options raise the value-error floor exactly as Mann & Mannor's bound predicts, and raise $\varepsilon_{\text{abs}}$ because a 100-step abstract transition is far less Markov in $\mathcal{G}$ than a 25-step one. Lower $K$ and $\Delta$ falls but $\rho \to 1$ and hierarchy buys nothing. The learned $f_\psi$ is supposed to break this tradeoff by making subgoals both long-range and reliably reachable — and there is no objective in current practice that scores $f_\psi$ on that joint property, only on downstream return, which is invariant to reparameterizations of $\mathcal{G}$ (§6). That is the gap.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*