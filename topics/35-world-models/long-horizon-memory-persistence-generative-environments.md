---
id: 35-world-models/long-horizon-memory-persistence-generative-environments
title: "Long-Horizon Memory Persistence in Generative Environments"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Horizon Memory Persistence in Generative Environments

> **Topic:** World Models & Planning · **ID:** `35-world-models/long-horizon-memory-persistence-generative-environments` · **Status:** open

## 1. Problem Statement

A generative environment is a model $G_\theta$ that maps an action stream to an observation stream, playable in a loop: the user acts, the model renders, the user acts again. Genie, GameNGen and Oasis are instances. All of them forget. Walk away from a wall, turn around after 20 seconds, and the wall is a different wall.

The problem: **make $G_\theta$ preserve the state of the world across horizons long enough for planning, and measure whether it has.** Three variants, of very different difficulty:

- **Measurement.** Define a statistic that separates *the model remembered the world* from *the model resampled a plausible world*. Two renders of the same corridor can be equally plausible and equally wrong; per-frame perceptual distance does not distinguish them.
- **Method.** Build an architecture whose persistence horizon grows sublinearly in inference cost — explicit spatial memory, retrieval over past frames, or a persistent latent state — rather than by extending the attention window.
- **Theory.** Characterise what a next-frame predictor trained on an i.i.d. clip distribution can persist at all. No lower bound exists on the state capacity needed for $\epsilon$-consistent revisits after $\Delta$ steps.

Solving it means: an agent can leave a location, act elsewhere for minutes, return, and find the environment as it left it — including changes the agent itself made — with a stated error tolerance and a cost that does not grow quadratically in elapsed time.

## 2. Formal Setting

Let the true environment be a POMDP $(\mathcal{S}, \mathcal{A}, T, \Omega, O)$ with latent state $s_t$, action $a_t$, observation $x_t \in \mathbb{R}^{H\times W\times 3}$. The generative environment is $G_\theta(x_{1:t}, a_{1:t}) \to \hat{x}_{t+1}$, with an internal carrier $m_t$ (KV cache, recurrent state, or retrieval index) of size $|m_t|$ bytes.

**Revisit pairs.** A trajectory $\tau$ contains a revisit $(t_1, t_2)$ if the agent pose returns: $\|p_{t_2} - p_{t_1}\| \le \delta_p$ and $|\phi_{t_2} - \phi_{t_1}| \le \delta_\phi$, with gap $\Delta = t_2 - t_1$. In a generated environment the true pose is unavailable, so $p_t$ is either **dead-reckoned from the action stream** (valid only if the model's own dynamics are faithful) or **estimated by an external visual odometry model** (valid only if the render is metric). Both are approximations; state which one a reported number used.

**Persistence horizon.** For a perceptual distance $d$ (LPIPS, or DINOv2 patch-feature cosine):

$$C(\Delta) \;=\; \mathbb{E}_{\tau,\,(t_1,t_2):\,t_2-t_1=\Delta}\big[\, d(\hat{x}_{t_2},\, \hat{x}_{t_1}) \,\big], \qquad H_\epsilon \;=\; \max\{\Delta : C(\Delta) \le \epsilon\}.$$

$C(\Delta)$ alone is uninterpretable. It must be bracketed by two measured constants:

$$C_{\text{floor}} = C(\Delta \!\to\! 0) \quad\text{(pose-jitter and sampling noise)}, \qquad C_{\text{amn}} = \mathbb{E}\big[d(\hat{x}_{t_2}, \tilde{x})\big]$$

where $\tilde{x}$ is a render of the same pose from an **independently resampled** rollout — the amnesic control. The reportable quantity is the normalised memory score

$$M(\Delta) \;=\; \frac{C_{\text{amn}} - C(\Delta)}{C_{\text{amn}} - C_{\text{floor}}} \in [0,1],$$

$M=1$ meaning perfect persistence, $M=0$ meaning the model resampled. Cost is $\kappa(\Delta) = $ FLOPs per frame at elapsed time $\Delta$; the target is $\kappa$ sublinear in $\Delta$ at fixed $M$.

**Assumptions, and which are violated.** (i) *Static environment between visits* — violated by any dynamic entity, which is why $C_{\text{amn}}$ must be estimated per-scene, not globally. (ii) *Pose is recoverable* — violated: generated video is not metrically consistent, and odometry error itself grows with $\Delta$, so $C_{\text{floor}}$ is not constant. (iii) *$d$ is monotone in semantic identity* — violated: LPIPS between two different-but-similar corridors is often below LPIPS between the same corridor at 2° of pose error. (iv) *Actions are executed* — violated: the model may simply not move the agent as commanded, converting a memory failure into a control failure.

## 3. State of the Art

**Systems / empirical.**
- **GameNGen** (Valevski et al., ICLR 2025) runs DOOM at 20 FPS from a diffusion model conditioned on the previous 64 frames — a 3.2 s context. Established: human raters distinguished real from simulated clips only 58% (1.6 s) and 60% (3.2 s) of the time. Not established: anything beyond 3.2 s; the evaluation horizon *is* the context length, so the study is silent on persistence.
- **Genie** (Bruce et al., ICML 2024), 11B parameters, learns a latent action space from unlabelled video at 16-frame context. **Genie 3** (DeepMind, Aug 2025) claims minutes of consistency at 720p/24 FPS and "visual memory extending about one minute into the past". This is a blog claim with no paper, no ablation, and no released protocol — a benchmark-number-only result, and not even that, since the number is qualitative.
- **Oasis** (Decart/Etched, 2024) is an open real-time Minecraft model; community testing shows objects change identity within seconds of occlusion. Systems result, no controlled measurement.
- **WORLDMEM** (Xiao et al., 2025) adds an explicit memory bank of past frames with pose keys and attends over it, reporting improved revisit consistency in Minecraft. This is the clearest *method* attack on the problem; the reported gains are not yet independently reproduced.
- **Diffusion Forcing** (Chen et al., NeurIPS 2024) gives per-token independent noise levels and stabilises rollouts beyond the training horizon. Established for stability; it does not add state, so it does not by itself add persistence.

**Theory.** There is essentially no SOTA. Standard sequence-model capacity results (attention needs $\Theta(\Delta)$ cache for exact recall; fixed-size recurrent state induces a $\log$-scale information bottleneck) transfer informally, but no bound is stated for the *spatially indexed, partially observed* revisit setting.

## 4. What Is Known

- **Context windows in deployed interactive world models are seconds, not minutes.** Genie: 16 frames. GameNGen: 64 frames at 20 FPS = 3.2 s. Oasis: order 10s of frames. Measured at 0.5B–11B parameters.
- **Human-realism evals do not test memory.** GameNGen's 58–60% rater accuracy was measured on clips of 1.6 s and 3.2 s — shorter than one occlusion event.
- **Long-term memory is hard even for agents with privileged state.** Memory Maze (Pasukonis, Lillicrap, Hafner, ICLR 2023) shows Dreamer-class agents degrade sharply as maze size grows from 9×9 to 15×15, staying well below human players, at horizons of thousands of steps.
- **Video models learn in-distribution dynamics, not laws.** Kang et al. (2024) show diffusion video models generalise well for interpolation of physical scenarios but fail out-of-distribution even with large data and model scale — so persistence cannot be assumed to emerge from scale on video alone.
- **Attention cost is the binding constraint.** At 720p with a typical $\sim$1024-token-per-frame tokeniser, 60 s at 24 FPS is $1.47\times10^6$ context tokens. Full attention over that at 24 FPS is not real-time on current accelerators.

## 5. What Is Not Known

- **Methodologically blocked (the core of it).** There is no agreed persistence metric with a published floor-and-amnesic-control bracket. Papers report LPIPS/PSNR/FVD on short clips; none report $M(\Delta)$ or anything equivalent, so "Genie 3 remembers for a minute" and "Oasis forgets in seconds" are not comparable claims.
- **Empirically open.** Does persistence improve with parameters at fixed context, or only with context/memory mechanism? Runnable today at 1B–10B scale; nobody has published the scaling sweep with the horizon held as the dependent variable.
- **Empirically open.** Do *agent-caused* edits (a placed block, an opened door) persist as well as passive scene content? Likely worse — edits are rare in training data — but unmeasured.
- **Theoretically open.** Minimum state size for $\epsilon$-consistent revisits after $\Delta$ steps in a POMDP with $k$ distinguishable locations. No lower bound, no matching construction.

## 6. Why It Is Hard

**The measurement is confounded by pose non-identifiability.** To ask "is this the same wall?", you must render from the same pose. In a generated environment there is no ground-truth pose; you infer it from the action stream or from the model's own frames. Both estimators drift with $\Delta$ — precisely the axis you are measuring. So $C_{\text{floor}}$ grows with $\Delta$ at an unknown rate, and $C(\Delta) - C_{\text{floor}}$ — the memory signal — is confounded with odometry error. A model that remembers perfectly but drifts spatially scores identically to one that forgets.

Second, **absent ground truth**: unlike a game engine, the generative environment has no reference state to compare against, so every metric is self-referential. Third, **compute**: the honest fix (full attention over minutes) costs $O(\Delta^2)$, and the cheap fixes (fixed recurrent state, sliding window) are exactly the ones that cannot store a map.

## 7. Current Research (as of 2026)

- **Explicit memory banks with geometric keys** — WORLDMEM-style retrieval over pose-tagged past frames. Main live direction. *(frontier — verify)* Several follow-ups combine this with 3D Gaussian or voxel caches.
- **Google DeepMind Genie line** — scaling interactive world models; persistence claimed, protocol unpublished. *(frontier — verify)*
- **Long-context autoregressive video** (e.g. FAR, Yu et al. 2025) — short-term/long-term context split with distinct tokenisation rates.
- **Driving world models** (Wayve GAIA-1; NVIDIA Cosmos) — shorter horizons but real pose ground truth from vehicle odometry, which makes them the best available testbed for calibrated persistence measurement.
- **Agent-side memory** — Dreamer-lineage work on hierarchical latent state, and Memory Maze as the standing benchmark.

## 8. Concrete Next Experiment

**The revisit-and-edit probe with an amnesic control.**

- **Scale.** One 1–3B interactive world model trained on Minecraft-style data at 256×256, 10 FPS, 32-frame context. Two arms beyond the base: (a) +memory bank of 256 pose-keyed retrieved frames; (b) +context extended to 512 frames. Budget: order $10^3$ H100-hours, i.e. reproducible by a single academic group.
- **Protocol.** 500 scripted trajectories. Agent observes a landmark, optionally *edits* it (places a coloured block), leaves, executes a fixed distractor loop of length $\Delta \in \{50, 100, 200, 400, 800, 1600\}$ frames, returns along the reverse path to the same commanded pose.
- **Control arm (mandatory).** The **amnesic resample**: same seed, same landmark scene, but the return frame generated from a rollout that never saw the landmark. This gives $C_{\text{amn}}$. The **floor** is $\Delta=0$ re-render with the same commanded pose, giving $C_{\text{floor}}$.
- **Deciding number.** $\Delta^{\ast} = H_{0.5}$: the largest gap at which $M(\Delta) \ge 0.5$, reported separately for passive scene content and for agent edits, with 95% bootstrap CIs over trajectories. Publish $C_{\text{floor}}(\Delta)$ alongside, so the confound is visible.
- **What it settles.** If $\Delta^{\ast}$ for the memory-bank arm exceeds the extended-context arm at equal inference FLOPs, retrieval beats context and the field should stop buying horizon with attention. If $\Delta^{\ast}_{\text{edit}} \ll \Delta^{\ast}_{\text{scene}}$ in all arms, the problem is training-distribution coverage of edits, not architecture.

## 9. Key References

- **[Foundational]** David Ha, Jürgen Schmidhuber. *World Models.* NeurIPS, 2018. — arXiv:1803.10122
- **[Foundational]** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. *Mastering Diverse Domains through World Models (DreamerV3).* Nature, 2025. — arXiv:2301.04104
- **[SOTA]** Jake Bruce et al. *Genie: Generative Interactive Environments.* ICML, 2024. — arXiv:2402.15391
- **[SOTA]** Dani Valevski, Yaniv Leviathan, Moab Arar, Shlomi Fruchter. *Diffusion Models Are Real-Time Game Engines (GameNGen).* ICLR, 2025. — arXiv:2408.14837
- **[SOTA]** Boyuan Chen et al. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion.* NeurIPS, 2024. — arXiv:2407.01392
- **[SOTA]** Zeqi Xiao et al. *WorldMem: Long-term Consistent World Simulation with Memory.* Preprint, 2025. (identifier omitted — verify before citing)
- **[Benchmark]** Jurgis Pasukonis, Timothy Lillicrap, Danijar Hafner. *Evaluating Long-Term Memory in 3D Mazes (Memory Maze).* ICLR, 2023. — arXiv:2210.13383
- **[Evidence]** Bingyi Kang et al. *How Far is Video Generation from World Model: A Physical Law Perspective.* Preprint, 2024. — arXiv:2411.02385
- **[Metric]** Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).* CVPR, 2018. — arXiv:1801.03924
- **[Systems]** Anthony Hu et al. *GAIA-1: A Generative World Model for Autonomous Driving.* Preprint, 2023. — arXiv:2309.17080

## 10. Worked Example

Take a 1B model, 256×256, 10 FPS, 32-frame context. The agent faces a red-brick wall with a distinctive white streak, walks away for $\Delta = 600$ frames (60 s), and returns.

Measured LPIPS (illustrative magnitudes of the kind this probe returns):

```
C_floor  (Δ=0 re-render, commanded pose)         0.081
C(600)   (model, revisit)                        0.312
C_amn    (amnesic resample, same pose)           0.339
```

Then

$$M(600) = \frac{0.339 - 0.312}{0.339 - 0.081} = \frac{0.027}{0.258} = 0.105.$$

The model retains about 10% of the recoverable signal — near-total forgetting. But the obstruction is the denominator's other end. Rerun the odometry: at $\Delta=600$ the commanded pose and the visually estimated pose differ by 1.9° yaw and 0.4 m. Re-rendering the *ground-truth* scene at that pose offset alone gives LPIPS $\approx 0.09$–$0.12$. So the pose-induced floor at $\Delta=600$ is not 0.081 but somewhere in $[0.09, 0.12]$, and the numerator — the entire memory signal, 0.027 — is smaller than the uncertainty in the floor.

That is the problem in one line: **the memory effect is smaller than the pose-estimation error used to measure it.** No amount of extra rollouts fixes it, because the error is systematic in $\Delta$, not random. Progress needs either an environment with true pose ground truth (driving logs, or a paired game engine that the generative model was distilled from) or a pose-invariant identity metric — matching the white streak as an object rather than the frame as an image. Until one of those is standard, reported persistence horizons for generative environments are not comparable across papers.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*