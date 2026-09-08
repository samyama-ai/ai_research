---
id: 35-world-models/discovering-temporal-abstractions-without-options
title: "Discovering Temporal Abstractions Without Hand-Specified Options"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Discovering Temporal Abstractions Without Hand-Specified Options

> **Topic:** World Models & Planning · **ID:** `35-world-models/discovering-temporal-abstractions-without-options` · **Status:** open

## 1. Problem Statement

**Input.** A stream of interaction with an environment (observations, actions, and either a reward signal or none), plus a compute budget. No human-authored subgoals, no hand-drawn option initiation sets, no task decomposition, no privileged state variables such as "agent $(x,y)$" or "door open".

**Output.** A set of temporally extended behaviors — options, skills, or a latent action space over a learned world model — together with a model of their outcomes, that a planner can search over.

**Objective.** Solving the problem means: on a held-out family of tasks in the same environment, planning or learning *with* the discovered abstractions beats planning with primitive actions at matched compute, and the margin grows with horizon rather than shrinking.

Three variants that are routinely conflated:

- **Measurement.** Define a scalar that says an option set is good, computable without knowing the downstream tasks. Currently the weakest link.
- **Method.** Produce options that improve planning under an *agreed* measure. Many algorithms exist; almost none are ablated against a primitive-action control at matched compute.
- **Theory.** Characterize when a compact option set exists that reduces planning time, and whether it is learnable from data. Partly answered, and the answers are negative.

## 2. Formal Setting

An MDP $M = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$. An **option** (Sutton, Precup, Singh 1999) is $o = (\mathcal{I}_o, \pi_o, \beta_o)$: initiation set $\mathcal{I}_o \subseteq \mathcal{S}$, policy $\pi_o$, termination $\beta_o: \mathcal{S} \to [0,1]$. Options induce a semi-MDP with multi-time model

$$P_o(s' \mid s) = \sum_{k=1}^{\infty} \gamma^k \Pr(s_k = s', \text{terminate at } k \mid s, o), \qquad R_o(s) = \mathbb{E}\Big[\textstyle\sum_{k=0}^{\tau-1} \gamma^k r_{t+k} \,\Big|\, s, o\Big].$$

**Quantities as measured.**

- **Planning cost** $C(\mathcal{O})$: number of Bellman backups (or wall-clock GPU-seconds for a learned-model planner) until $\|V_k - V^*\|_\infty \le \varepsilon$, with $\varepsilon = 0.01 \cdot V_{\max}$. Requires $V^*$, so measured only in environments where value iteration on the ground-truth MDP is feasible.
- **Transfer gain** $G$: area under the learning curve on $N$ held-out tasks with $\mathcal{O}$ minus the same with primitives only, at equal environment steps *and* equal parameter count. Both arms must include the pretraining interactions used to discover $\mathcal{O}$; omitting them is the single most common measurement error.
- **Option duration** $\bar{\tau} = \mathbb{E}[\tau]$, measured empirically over rollouts. Reported values below 2 primitive steps mean the abstraction is not temporal.
- **Model compounding error** for a latent world model $\hat{P}$: $e(H) = \mathbb{E}\|\hat{z}_H - \phi(s_H)\|_2$ under open-loop rollout, plotted against $H$.

**Assumptions, and their status.**

1. *Stationary, fixed $\mathcal{S}$ and $\mathcal{A}$.* Holds in benchmarks; violated in embodied and web settings.
2. *Tasks share transition dynamics, differ only in $R$.* This is what makes options reusable. Violated in most realistic transfer settings, where dynamics shift too.
3. *Options terminate.* Deep option-discovery methods routinely collapse to $\beta_o \approx 1$ everywhere (degenerate primitives) or $\beta_o \approx 0$ (one option runs forever). Known violated; Harb et al. (2018) add a deliberation cost specifically to suppress it.
4. *The measure of a good option set is task-independent.* Not established. Options optimal for one reward family can be worse than primitives for another.

## 3. State of the Art

**Theory SOTA.** Jinnai, Abel, Hershkowitz, Littman, Konidaris, *Finding Options that Minimize Planning Time* (ICML 2019): choosing $k$ options to minimize iterations of value iteration is NP-hard, and hard to approximate within a factor better than $2$ under standard assumptions; they give an approximation algorithm for a restricted covering formulation. Mann & Mannor (ICML 2014) prove options accelerate approximate value iteration when option durations are long *and* models are accurate — a conditional result, not an unconditional one.

**Empirical SOTA, established.**
- **Option-Critic** (Bacon, Harb, Precup, AAAI 2017): end-to-end option learning by policy gradient. Established that the gradients are correct and options are learnable; the widely reproduced finding is that without a termination penalty options degenerate to near-primitive length.
- **Deliberation cost** (Harb, Bacon, Klissarov, Precup, AAAI 2018): adding a per-switch cost demonstrably lengthens options. Established via ablation.
- **Eigenoptions / covering options** (Machado, Bellemare, Bowling, ICML 2017; Jinnai et al., ICML 2019): options from the graph Laplacian's eigenvectors reduce the expected cover time of the state graph. This is a *proved* property of the construction, and the strongest task-independent grounding available.
- **Deep Skill Chaining** (Bagaria & Konidaris, ICLR 2020): backward-chained initiation classifiers from a goal; works on continuous navigation.

**Claimed but unablated.**
- Unsupervised skill discovery by mutual-information maximization — DIAYN (Eysenbach et al., ICLR 2019), DADS (Sharma et al., ICLR 2020), Variational Intrinsic Control (Gregor et al., 2016). Eysenbach et al. (ICLR 2022) show MI objectives are equivalent to a specific geometric optimization and do *not* by themselves imply usefulness for any downstream reward. The downstream numbers are largely benchmark numbers on locomotion suites, rarely with a matched-compute primitive-action arm.
- **Director** (Hafner et al., NeurIPS 2022) plans in latent space with a learned goal-generating manager; results are strong on egocentric mazes but reported as benchmark returns, with no isolation of what the temporal abstraction contributes.
- **Meta-learned subgoals** (Veeriah et al., NeurIPS 2021) optimize options directly for downstream learning speed — the right objective, demonstrated at small scale.

## 4. What Is Known

- Options help *when the model is given*. Sutton et al. (1999) and Precup's thesis (2000) show correct semi-MDP planning; speedups in tabular four-rooms are on the order of a $2$–$10\times$ reduction in backups, at $\sim10^2$ states.
- Eigenoptions cut expected cover time in four-rooms-scale grids ($\sim100$ states) by roughly an order of magnitude versus random walk (Machado et al. 2017).
- Option collapse is real and reproduced: Option-Critic on Atari with 4–8 options yields $\bar{\tau}$ near 1–2 frames absent a deliberation cost (Bacon et al. 2017; Harb et al. 2018), at $\sim10^7$ frames.
- DIAYN learns tens of visually distinct skills (order 20–50) on MuJoCo locomotion at $\sim10^6$–$10^7$ steps, but hierarchical use of those skills gives gains mainly on goal-reaching tasks aligned with the skills' own diversity axis.
- Flat model-based agents are strong baselines: DreamerV3 (Hafner et al., *Nature*, 2025) solves 150+ tasks including Minecraft diamond collection with fixed hyperparameters and **no** explicit options. Any option-discovery claim must beat this control.
- Optimality is not free: Solway et al. (*PLoS Computational Biology*, 2014) formalize an optimal behavioral hierarchy as Bayesian model selection over task distributions — the optimum is defined *relative to a task distribution*, confirming assumption 4 above is false in general.

## 5. What Is Not Known

- **Theoretically open.** Whether a polynomial-time algorithm exists that, from trajectory data alone, returns an option set within a constant factor of the planning-time optimum for a *distribution* over rewards (the single-reward case is NP-hard; the distributional case has no hardness result either way). Also open: sample complexity of option discovery in continuous state spaces.
- **Empirically open.** Whether any discovery method beats a matched-compute flat model-based agent (DreamerV3-class) on a held-out task family at $\ge 10^8$ environment steps. The experiment is runnable today; the matched-compute control arm is usually missing.
- **Methodologically blocked.** There is no accepted task-independent scalar for option-set quality. Cover time, empowerment, MI, and deliberation cost are all proxies with known counterexamples. Until this is fixed, "better options" is not a measurable claim, only a per-benchmark one.

## 6. Why It Is Hard

Three obstructions, in order of severity.

1. **Non-identifiability.** Infinitely many option sets induce the same optimal value function. Without a task distribution the objective has no unique maximizer, so different methods optimize incompatible proxies and their numbers are not comparable. This is not a compute problem; it is a definition problem.
2. **Confounded measurement.** Reported hierarchical gains bundle (a) temporal abstraction, (b) extra parameters, (c) pretraining interaction, (d) exploration bonus from the intrinsic objective. Removing (b)–(d) — same parameter count, pretraining steps charged to both arms, intrinsic bonus given to the flat baseline too — removes most of the reported gap in the cases where it has been checked.
3. **Compute.** The regime where abstraction should pay — long horizons, $10^8$+ steps, sparse reward — is exactly where a full four-arm ablation costs on the order of thousands of GPU-hours per environment, so it is rarely run.

## 7. Current Research (as of 2026)

- **Reward-respecting subtasks** (Sutton, Machado, Holland, Szepesvári, Timbers, Tanner, White, *Artificial Intelligence*, 2023): discover options whose subtask rewards are derived from the main reward's own value function. Directly attacks non-identifiability by refusing to be task-agnostic.
- **Successor and forward-backward representations** as an option substitute: Touati & Ollivier, *Does Zero-Shot Reinforcement Learning Exist?* (ICLR 2023) get zero-shot task transfer with no options at all — arguably the strongest current argument that explicit options may be unnecessary.
- **Latent action / video world models.** Learning discrete latent actions from unlabeled video (Genie, Bruce et al., ICML 2024) yields action abstractions without any option formalism. Whether these latents are *temporally* abstract or merely per-frame is not settled *(frontier — verify)*.
- **LLM-proposed subgoals** as a substitute for discovery, in embodied and web agents. Widely deployed, essentially unablated against learned abstraction *(frontier — verify)*.
- Groups: Mila/Precup and Bacon (option-critic lineage), UAlberta/Sutton and Machado, Brown/Konidaris and Abel, Google DeepMind (Hafner, Veeriah), Meta AI (Ollivier, Touati).

## 8. Concrete Next Experiment

**Question.** Does discovered temporal abstraction beat a flat world model at matched compute, and does the margin grow with horizon?

**Scale.** Crafter and one egocentric 3D maze suite; $2 \times 10^8$ environment steps per arm; 5 seeds; held-out set of 20 tasks in the same dynamics, never seen during discovery. Roughly 3–5k A100-hours total.

**Arms.**
1. *Control:* DreamerV3, flat, no options, full $2 \times 10^8$ steps.
2. *Control+:* DreamerV3 plus the intrinsic objective (e.g. DIAYN reward) as an exploration bonus only — no hierarchy. This isolates the exploration confound.
3. *Treatment:* identical world model, options discovered in the first $10^8$ steps, hierarchical policy over options for the second $10^8$. Parameter count matched to arm 1 within 5%.
4. *Ceiling:* hand-specified options from environment internals, same budget.

**Deciding number.** Median normalized return on the 20 held-out tasks, plotted against task horizon $H$ bucketed at $\{<50, 50\text{–}200, >200\}$ primitive steps. The claim is supported only if arm 3 exceeds arm 2 by $\ge 10$ normalized points in the $H > 200$ bucket, with the gap monotonically increasing across buckets and non-overlapping 95% bootstrap CIs over seeds. Report $\bar{\tau}$ alongside; if $\bar{\tau} < 5$, the result is not about temporal abstraction whatever the returns say.

## 9. Key References

- **[Foundational]** R. Sutton, D. Precup, S. Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence 112(1–2), 1999.
- **[Foundational]** G. Konidaris, A. Barto. *Skill discovery in continuous reinforcement learning domains using skill chaining.* NeurIPS, 2009.
- **[SOTA-theory]** Y. Jinnai, D. Abel, D. E. Hershkowitz, M. Littman, G. Konidaris. *Finding Options that Minimize Planning Time.* ICML, 2019.
- **[SOTA-method]** P.-L. Bacon, J. Harb, D. Precup. *The Option-Critic Architecture.* AAAI, 2017. — arXiv:1609.05140
- **[SOTA-method]** J. Harb, P.-L. Bacon, M. Klissarov, D. Precup. *When Waiting Is Not an Option: Learning Options with a Deliberation Cost.* AAAI, 2018.
- **[SOTA-method]** M. C. Machado, M. G. Bellemare, M. Bowling. *A Laplacian Framework for Option Discovery in Reinforcement Learning.* ICML, 2017.
- **[SOTA-method]** R. Sutton, M. C. Machado, G. Z. Holland, D. Szepesvári, F. Timbers, B. Tanner, A. White. *Reward-Respecting Subtasks for Model-Based Reinforcement Learning.* Artificial Intelligence, 2023.
- **[Baseline]** D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering diverse control tasks through world models.* Nature, 2025.
- **[Related]** D. Hafner, K.-H. Lee, I. Fischer, P. Abbeel. *Deep Hierarchical Planning from Pixels.* NeurIPS, 2022.
- **[Related]** B. Eysenbach, A. Gupta, J. Ibarz, S. Levine. *Diversity is All You Need: Learning Skills without a Reward Function.* ICLR, 2019.
- **[Related]** A. Touati, Y. Ollivier. *Does Zero-Shot Reinforcement Learning Exist?* ICLR, 2023.
- **[Survey/Analysis]** A. Solway, C. Diuk, N. Córdova, D. Yee, A. Barto, Y. Niv, M. Botvinick. *Optimal Behavioral Hierarchy.* PLoS Computational Biology, 2014.

## 10. Worked Example

**Setting.** Four-rooms, $104$ reachable states, 4 primitive actions, $\gamma = 0.99$.

**Hand-specified options.** The 8 classical "go to hallway" options. Value iteration to $\varepsilon = 0.01$ from a corner start: about $70$ sweeps with primitives, about $12$ with primitives + options — roughly a $5.8\times$ reduction in backups. This is the number that made options famous.

**Now discover them.** Run eigenoptions: build the state-transition graph, take the second-smallest Laplacian eigenvector, define an option that ascends it and one that descends it. With 8 eigenoptions, cover time drops from about $\sim 500$ steps (random walk) to about $\sim 50$. The options do *not* terminate at hallways; they terminate at eigenvector extrema, which sit near room corners.

**The obstruction, made visible.** Place the goal in a room corner: the eigenoptions beat the hallway options, because their terminal states *are* corners. Place the goal just past a hallway: the hallway options win by a wide margin, and the eigenoptions can be *slower than primitives*, because each option commits $\sim 15$ steps in the wrong direction before terminating and the semi-MDP has no way to interrupt it mid-flight.

Both option sets score well on their own proxy — hallway options on planning-time reduction for the reward that generated them, eigenoptions on cover time — and each is worse than primitives on some reward in the family. At $104$ states, with $V^*$ computable exactly, this is a two-minute calculation. At $10^6$ latent states in a learned world model, the same failure is invisible: $V^*$ is unavailable, cover time is undefined, and the only signal is a benchmark return that also moves when parameter count or pretraining budget changes. That is why the problem is methodologically blocked before it is empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*