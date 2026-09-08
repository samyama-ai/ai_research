---
id: 35-world-models/data-scaling-laws-world-model-accuracy
title: "Data Scaling Laws for World Model Prediction Accuracy"
topic: 35-world-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data Scaling Laws for World Model Prediction Accuracy

> **Topic:** World Models & Planning · **ID:** `35-world-models/data-scaling-laws-world-model-accuracy` · **Status:** empirically-open

## 1. Problem Statement

A world model is a learned conditional predictor of future observations given past observations and actions. The question: **how does prediction accuracy — and the downstream control value of that accuracy — scale with the amount and composition of training data?**

Three variants, of very different difficulty:

- **Measurement variant.** Define an accuracy metric for a world model that (a) is comparable across model families, (b) is monotone in downstream planning value, and (c) does not saturate on the axis that matters. Currently unsolved: per-frame pixel losses saturate long before rollout dynamics become correct.
- **Method variant.** Given a fixed compute budget $C$, choose the data allocation — total frames, action-labelled fraction, environment diversity, trajectory length — that minimises downstream regret. Empirically open.
- **Theory variant.** Prove that world-model rollout error follows a power law in dataset size $D$ with exponent determined by properties of the environment (intrinsic dimension of the state manifold, mixing time, action-space entropy). No proof either way.

Solving it means: given a target planning performance and an environment class, predict the required data budget to within a factor of 2 *before* collecting it.

## 2. Formal Setting

Environment: a controlled Markov process with latent state $s_t \in \mathcal{S}$, action $a_t \in \mathcal{A}$, observation $o_t = \phi(s_t)$, transition kernel $P(s_{t+1} \mid s_t, a_t)$.

Dataset $\mathcal{D}$ of $N$ trajectories of length $T$; **measured as** $D = NT$ frames (the unit reported by video world models) or $D = NT \cdot |\text{tokens/frame}|$ (the unit that enters compute-optimal fits). These differ by 2–3 orders of magnitude and are routinely conflated.

Model $p_\theta(o_{t+1:t+h} \mid o_{\leq t}, a_{t:t+h-1})$, parameters $\theta \in \mathbb{R}^P$.

**One-step loss (measured):** held-out negative log-likelihood in nats per token, on trajectories from environments disjoint from training:
$$L_1(D,P) = -\tfrac{1}{|\mathcal{D}_{\text{test}}|}\sum \log p_\theta(o_{t+1}\mid o_{\leq t}, a_t).$$

**Horizon-$h$ rollout error (measured):** autoregressive open-loop rollout, scored against ground truth by a *state-recoverable* statistic $\psi$ (object positions, contact events, task-relevant scalars), not by pixels:
$$E_h(D) = \mathbb{E}\big[\lVert \psi(\hat o_{t+h}) - \psi(o_{t+h}) \rVert\big].$$

**Downstream regret (measured):** plan in the model, act in the true environment:
$$R(D) = J^\star - J\big(\pi_{\text{MPC}}[p_\theta]\big),$$
with $J$ the true-environment return under a fixed planner (fixed horizon, fixed sample count, fixed seeds).

**Hypothesised law:**
$$L_1(D) = L_\infty + \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad R(D) = R_\infty + \left(\frac{D_r}{D}\right)^{\alpha_R}.$$

The open quantity is the relation between $\alpha_D$ and $\alpha_R$, and whether $R_\infty > 0$ (irreducible planning regret from an accurate model).

**Assumptions, with violation status:**
- *IID sampling of $\mathcal{D}$* — violated: trajectories are temporally correlated, so effective sample size $\ll NT$.
- *Test distribution = train distribution* — violated by construction: world models are used off-policy, on states the data collector never visited.
- *Single power-law regime* — violated: broken/multi-regime scaling is documented (Caballero et al., ICLR 2023).
- *$\psi$ exists and is measurable* — violated for open-domain video, where no ground-truth state extractor exists. This is the methodological block.

## 3. State of the Art

**Established (ablated, reproduced):**
- Power-law $L(D)$, $L(P)$, $L(C)$ for autoregressive modelling across modalities *including video*: Henighan et al., 2020 (arXiv:2010.14701), up to ~1B parameters.
- Compute-optimal token/parameter balance $D \propto P$: Hoffmann et al., NeurIPS 2022 (Chinchilla) — for text loss, not for control.
- Data-scaling power laws in robot imitation learning over *environments* and *objects*, with diminishing returns per additional demonstration within a scene: Lin et al., 2024 (arXiv:2410.18647). Reproduced in-lab; single embodiment.
- Scaling laws for imitation loss and for game score in single-agent games: Tuyls et al., TMLR 2023 (arXiv:2307.09423) — reports that *return* follows a power law in compute with a **different exponent** than the imitation loss.

**Claimed but unablated:**
- "Scaling improves world simulation" (OpenAI Sora technical report, 2024): three compute points shown as video samples, no loss curve, no held-out metric, no seeds. This is an existence demonstration, not a scaling law.
- Genie / Genie 2 / Genie 3 (DeepMind, ICML 2024 and subsequent releases): Genie 1 reports a clean parameter-scaling curve at fixed data; the *data* axis is not swept. Genie 2/3 quality claims exist only as demos.
- NVIDIA Cosmos world foundation models (2025, arXiv:2501.03575): 20M hours of video reported; no controlled data-ablation curve published.

**Benchmark-number-only results:** GAIA-1 (Hu et al., 2023, arXiv:2309.17080) driving-world-model FVD figures; DIAMOND (Alonso et al., NeurIPS 2024) Atari-100k score of 1.46 human-normalised mean. Both are single-budget points, not curves.

## 4. What Is Known

- $L(D)$ power laws hold for video-token prediction over ~3 orders of magnitude of data, exponent $\alpha_D \approx 0.05$–$0.15$ in nats per token at $\leq$1B parameters (Henighan et al., 2020). The exponent is small — an order of magnitude more data buys ~10–30% loss reduction.
- Exponents track **intrinsic data dimension**: $\alpha \approx 4/d$ for a manifold of dimension $d$ (Sharma & Kaplan, JMLR 2022; consistent with Bahri et al., PNAS 2024). Video state manifolds have large $d$, predicting small $\alpha$ — which is what is observed.
- Data *pruning* can beat power-law scaling, turning $\alpha$ into a faster decay when the pruning metric is good (Sorscher et al., NeurIPS 2022; ImageNet scale). No world-model replication.
- Sample-efficient world models reach Atari-100k human-normalised mean 1.04 (IRIS, Micheli et al., ICLR 2023) and 1.46 (DIAMOND, 2024) at **400k frames** — showing that in low-diversity environments the data requirement is small and the binding constraint is model class, not data.
- DreamerV3 (Hafner et al., *Nature*, 2025) shows *monotone improvement with model size* (12M→400M) at fixed data budget across >150 tasks — a parameter law, explicitly not a data law.
- In imitation, environment/object diversity contributes more per unit data than additional demonstrations in seen environments: ~generalisation to new environments saturates around 32 environments × 32 objects in the Lin et al. (2024) grid, at ~10^3 total demonstrations.

## 5. What Is Not Known

- **Empirically open:** the joint law $R(D, P, \text{diversity})$ for a world model used for planning. The experiment is runnable today in simulation; nobody has run a full 2-D data×diversity sweep with regret (not loss) as the dependent variable. This is the central gap.
- **Empirically open:** whether $\alpha_R = \alpha_D$. Tuyls et al. (2023) suggest not, in imitation; unreplicated for model-based planning.
- **Empirically open:** the exchange rate between action-labelled and passive (action-free) frames. Latent-action methods (LAPO; Genie) make passive video usable, but no paper reports "$k$ passive frames $\approx$ 1 labelled frame" as a measured constant.
- **Theoretically open:** whether rollout error compounds as $O(h)$, $O(h^2)$, or $\exp(h)$ under a model with one-step TV error $\epsilon$ *and* a data-dependent $\epsilon(D)$. The $O(\epsilon h^2)$ behavioural-cloning bound (Ross & Bagnell, 2010) is the closest analogue; its transfer to generative world models is assumed, not proved.
- **Methodologically blocked:** "prediction accuracy" for open-domain video. FVD is sensitive to the feature extractor and is not monotone in physical correctness; physics-probe benchmarks (VideoPhy, Bansal et al. 2024; Physics-IQ, Motamed et al. 2025) show high-visual-quality models failing physical-consistency probes. Until $\psi$ is defined, the $y$-axis of the scaling plot is unspecified.

## 6. Why It Is Hard

Four named obstructions:

1. **Confounded measurement.** Every frontier data increase co-varies with diversity, resolution, curation and action-label density. No public frontier result varies $D$ alone. The reported "scaling" is a bundle.
2. **The evaluation does not measure what it names.** Per-frame likelihood and FVD are dominated by texture, which saturates early; the dynamics content is a small fraction of the bits. A model can improve 0.05 nats/token and get *worse* at predicting a collision.
3. **Non-identifiability of $L_\infty$.** Fitting $L_\infty + (D_c/D)^{\alpha}$ from 3–4 budget points gives $\alpha$ confidence intervals wide enough to admit both 0.05 and 0.15 — a factor-of-1000 difference in the data needed for a target loss. Robust fits need ≥6 points spanning ≥3 decades.
4. **Compute cost of the right experiment.** The dependent variable is planning regret, which requires a full planner rollout per checkpoint per seed. A 5×6 model×data grid with 5 seeds is 150 evaluation runs on top of 30 training runs.

## 7. Current Research (as of 2026)

- **Interactive-video world models at scale** — DeepMind (Genie line), OpenAI, NVIDIA (Cosmos), Runway, World Labs. Data volumes are reported; controlled data ablations are not published *(frontier — verify)*.
- **Robot data scaling** — Open X-Embodiment consortium, DROID, Physical Intelligence. Diversity-vs-volume trade-offs are the live question; cross-embodiment transfer laws are being fit *(frontier — verify)*.
- **Latent action learning from passive video** — LAPO-style approaches, to convert unlabelled video into a usable data axis; the exchange rate is the open number.
- **Physics-grounded evaluation** — VideoPhy, Physics-IQ, and simulator-backed probes; the enabling work for the measurement variant.
- **Theory** — data-manifold-dimension explanations of exponents (Bahri, Sharma & Kaplan lineage), not yet extended to autoregressive rollout error.

## 8. Concrete Next Experiment

**Environment.** Procedurally generated, unlimited data, ground-truth state available: Craftax or a Mujoco/Isaac procedural manipulation suite with a state extractor $\psi$ (object poses, contact flags).

**Scale.** 5 model sizes $P \in \{15\text{M}, 50\text{M}, 150\text{M}, 400\text{M}, 1.2\text{B}\}$ × 6 data budgets $D \in \{10^6, 10^{6.7}, 10^{7.3}, 10^8, 10^{8.7}, 10^{9.3}\}$ frames. 30 training runs; ~2–4k A100-hours total at 64×64 resolution.

**Control arm.** Two data curricula at *matched frame count*: (A) **volume** — fixed 32 procedural seeds, more trajectories each; (B) **diversity** — frames spread over $\sqrt{D}$ distinct seeds. Same tokenizer, same optimizer, same schedule. Held-out evaluation on unseen seeds only.

**Measurements per checkpoint.** $L_1$ (nats/token); $E_h$ for $h \in \{1,5,20,50\}$ using $\psi$; regret $R(D)$ with a fixed CEM planner (horizon 20, 512 samples, 5 seeds).

**The deciding number.** The ratio
$$\rho = \frac{\alpha_R}{\alpha_D}$$
fit over the 6 data points at the compute-optimal model size, with bootstrap CIs, reported separately for arms A and B.

- $\rho \approx 1$ with overlapping CIs across arms ⟹ likelihood scaling is a valid proxy; data budgeting can be planned from loss curves alone.
- $\rho < 0.5$, or $\rho$ differing by >2× between arms ⟹ loss-based scaling laws do not predict control value, and every existing frontier scaling claim is measuring the wrong axis.

A secondary number worth extracting from the same runs: the diversity exchange rate $\kappa$ such that arm B at $D$ frames matches arm A at $\kappa D$ frames on regret.

## 9. Key References

- **[Foundational]** Ha, D., Schmidhuber, J. *World Models.* NeurIPS, 2018. — arXiv:1803.10122
- **[Foundational]** Kaplan, J. et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Henighan, T. et al. *Scaling Laws for Autoregressive Generative Modeling.* 2020. — arXiv:2010.14701
- **[Foundational]** Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Theory]** Sharma, U., Kaplan, J. *Scaling Laws from the Data Manifold Dimension.* JMLR, 2022.
- **[Theory]** Bahri, Y., Dyer, E., Kaplan, J., Lee, J., Sharma, U. *Explaining Neural Scaling Laws.* PNAS, 2024.
- **[Theory]** Caballero, E., Gupta, K., Rish, I., Krueger, D. *Broken Neural Scaling Laws.* ICLR, 2023. — arXiv:2210.14891
- **[Theory]** Ross, S., Bagnell, D. *Efficient Reductions for Imitation Learning.* AISTATS, 2010.
- **[SOTA]** Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. *Mastering Diverse Control Tasks through World Models.* Nature, 2025.
- **[SOTA]** Bruce, J. et al. *Genie: Generative Interactive Environments.* ICML, 2024. — arXiv:2402.15391
- **[SOTA]** Alonso, E. et al. *Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND).* NeurIPS, 2024.
- **[SOTA]** Micheli, V., Alonso, E., Fleuret, F. *Transformers are Sample-Efficient World Models (IRIS).* ICLR, 2023.
- **[SOTA]** Hu, A. et al. *GAIA-1: A Generative World Model for Autonomous Driving.* 2023. — arXiv:2309.17080
- **[Empirical]** Lin, F. et al. *Data Scaling Laws in Imitation Learning for Robotic Manipulation.* 2024. — arXiv:2410.18647
- **[Empirical]** Tuyls, J., Madeka, D., Foster, D., Narasimhan, K. *Scaling Laws for Imitation Learning in Single-Agent Games.* TMLR, 2023. — arXiv:2307.09423
- **[Empirical]** Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., Morcos, A. *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS, 2022. — arXiv:2206.14486
- **[Evaluation]** Bansal, H. et al. *VideoPhy: Evaluating Physical Commonsense for Video Generation.* 2024.
- **[Survey]** NVIDIA. *Cosmos World Foundation Model Platform for Physical AI.* 2025. — arXiv:2501.03575

## 10. Worked Example

Take a driving world model at 256×256, 8 fps, tokenized at 1024 tokens/frame. Suppose observed $\alpha_D = 0.08$ with $L_\infty = 1.20$ nats/token, fit from three budgets:

| Hours of video | Frames $D$ | Tokens | $L_1$ (nats/token) |
|---|---|---|---|
| 1k | $2.9\times10^7$ | $3.0\times10^{10}$ | 1.62 |
| 10k | $2.9\times10^8$ | $3.0\times10^{11}$ | 1.55 |
| 100k | $2.9\times10^9$ | $3.0\times10^{12}$ | 1.49 |

Fit gives $\alpha_D = 0.079$. To reach $L_1 = 1.30$: $(D_c/D)^{0.079} = 0.10 \Rightarrow D = D_c \cdot 10^{1/0.079} = D_c \cdot 10^{12.7}$. Extrapolating from the third point, that is roughly $10^{5}\times$ more data — **10 billion hours of driving video**, which does not exist.

Now the obstruction. Refit with the same three points but $L_\infty$ free in $[1.05, 1.35]$: $\alpha_D$ ranges over $0.05$–$0.19$, and the data required for $L_1 = 1.30$ ranges over 6 orders of magnitude. Three budget points cannot identify the asymptote.

Worse, the quantity that matters is unchanged by any of this. Score the same three checkpoints on a rollout probe — "does the model predict the lead vehicle stopping within 20 frames of a brake event" — and typical behaviour is 71% → 74% → 75%. That is a $\rho \approx 0.3$ regime: a 0.13 nats/token gain (8% of the loss range) bought 4 points of the metric anyone cares about, and the curve is flattening faster than the loss curve. The loss law is real, well-fit, and nearly uninformative about the model's value as a simulator. That mismatch — not the compute bill — is what keeps this problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*