---
id: 35-world-models/capacity-allocation-encoder-versus-dynamics
title: "Optimal Model Capacity Allocation Between Encoder and Dynamics"
topic: 35-world-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Model Capacity Allocation Between Encoder and Dynamics

> **Topic:** World Models & Planning · **ID:** `35-world-models/capacity-allocation-encoder-versus-dynamics` · **Status:** empirically-open

## 1. Problem Statement

A learned world model factors into at least two parameterized parts: an **encoder** $e_\phi$ mapping observations to latents, and a **dynamics/transition** model $d_\theta$ predicting the next latent given an action. Given a fixed total parameter (or FLOP) budget $N$, how should it be split between $\phi$ and $\theta$?

Three variants, with different difficulty:

- **Measurement.** For a fixed architecture family and task distribution, does downstream control return depend on the split $\rho = N_\phi / N$ in a way that is reproducible, or is the loss surface in $\rho$ flat within seed noise? No published result establishes either.
- **Method.** Produce an allocation rule $\rho^\star(N, \text{task})$ — analogous to Chinchilla's parameters-vs-tokens rule (Hoffmann et al., 2022) — that is predictive out of sample.
- **Theory.** Prove that under stated assumptions on observation complexity and transition complexity, the optimal split takes a particular form (e.g. that $\rho^\star$ is scale-invariant, or that it shifts toward dynamics as $N$ grows).

Solving it means: a rule that, given a budget and cheap task statistics, predicts the split within the accuracy needed to beat a uniform-split baseline by a stated margin at a scale where the baseline was itself tuned.

## 2. Formal Setting

A POMDP $(\mathcal{O}, \mathcal{A}, \mathcal{S}, T, R, \gamma)$. The model is

$$z_t = e_\phi(o_{\leq t}), \qquad \hat z_{t+1} \sim d_\theta(\cdot \mid z_t, a_t), \qquad \hat r_t = r_\psi(z_t), \qquad \hat o_t \sim g_\xi(\cdot\mid z_t).$$

**Budget.** $N = N_\phi + N_\theta + N_\psi + N_\xi$, counted as trainable parameters excluding embeddings and normalization; $\rho = N_\phi/(N_\phi+N_\theta)$ with heads held fixed. In practice compute, not parameters, is the binding constraint: measure $C \approx 6 N D$ FLOPs for $D$ training tokens/frames, and note that encoders over high-resolution pixels have much higher FLOPs-per-parameter than dynamics transformers, so equal-$\rho$ is not equal-compute. Report both.

**Objective.** The quantity of interest is not model loss but control return:

$$J(\phi,\theta) = \mathbb{E}\Big[\textstyle\sum_t \gamma^t r_t\Big] \quad \text{under the policy planned or trained inside the model,}$$

and the problem is $\rho^\star(N) = \arg\max_\rho \; \mathbb{E}_{\text{seeds}}\,[\,J \mid N, \rho\,]$. Measured as: mean over $\ge 5$ seeds of normalized score at a fixed environment-step budget, with a bootstrap interval (Agarwal et al., NeurIPS 2021, on RL evaluation statistics).

**Diagnostic split.** Decompose one-step error into a representation term and a dynamics term:

$$\underbrace{\mathbb{E}\,\|e_\phi(o_{t+1}) - d_\theta(e_\phi(o_t),a_t)\|^2}_{\text{latent prediction error}}, \qquad \underbrace{\mathbb{E}\,\|o_{t+1} - g_\xi(e_\phi(o_{t+1}))\|^2}_{\text{reconstruction error}}.$$

Assumptions, and where they fail:
- **Separability** — that "encoder capacity" and "dynamics capacity" are distinct resources. Violated by RSSM and by decoder-free models where the encoder is trained through the dynamics loss and absorbs transition structure.
- **Fixed latent width.** $\rho$ is usually varied by changing depth/width, which also changes latent dimension $\dim(z)$ — a third confound.
- **Loss–return monotonicity.** Violated: value-equivalent models (Grimm et al., 2020) show lower predictive loss need not mean higher return.
- **Stationary data.** Violated in on-policy model-based RL, where the data distribution is a function of the model being compared.

## 3. State of the Art

**Established.**
- **TD-MPC2** (Hansen, Su, Wang, ICLR 2024; arXiv:2310.16828) trains a single agent on 80+ continuous-control tasks and scales the model from roughly 1M to 317M parameters across five sizes, reporting monotone improvement in aggregate normalized score. The scaling is of the whole model; the encoder/dynamics split is not swept.
- **DreamerV3** (Hafner, Pasukonis, Ba, Lillicrap; *Nature*, 2025; arXiv:2301.04104) reports a model-size sweep from roughly 10M to 200M parameters showing both final score and data efficiency improving with size, under fixed hyperparameters. Again the split is fixed by the architecture recipe.

**Claimed but unablated.**
- **Genie** (Bruce et al., ICML 2024; arXiv:2402.15391) reports an 11B-parameter system in which the great majority of capacity sits in the dynamics model, with a comparatively small video tokenizer and latent-action model, plus a scaling study over model size. The allocation is a design choice, not an ablated variable — no arm with a large tokenizer and small dynamics is reported.
- Decoder-free / JEPA-style stacks (V-JEPA, Bardes et al., 2024) argue implicitly for encoder-heavy allocation, but compare against different objectives, not against the same objective at a different split.

**Benchmark-number-only.** Atari-100k results — IRIS 1.046 mean human-normalized score (Micheli et al., ICLR 2023; arXiv:2209.00588), DIAMOND 1.46 (Alonso et al., NeurIPS 2024; arXiv:2405.12399) — differ in encoder type, dynamics type, parameter count and training recipe simultaneously. They are not evidence about $\rho$.

## 4. What Is Known

- **Whole-model scaling helps in MBRL, at small absolute scale.** TD-MPC2: 1M→317M parameters, ~100 tasks, single agent, aggregate score rising with size (ICLR 2024). DreamerV3: ~10M→200M, 150+ tasks, same hyperparameters throughout (Nature 2025). Both are 2–3 orders of magnitude below LLM scaling-law regimes.
- **Compute-optimal allocation rules exist in a neighbouring setting and are non-trivial.** Chinchilla (Hoffmann et al., NeurIPS 2022; arXiv:2203.15556) overturned Kaplan et al. (2020; arXiv:2001.08361) on the parameters-vs-tokens ratio using ~400 runs up to 16B parameters — direct evidence that allocation questions are settled by dense sweeps, not by single large runs.
- **Not all model error matters equally.** Value equivalence (Grimm et al., NeurIPS 2020; arXiv:2011.03506) formalizes model classes indistinguishable with respect to a value-function set; MuZero (Schrittwieser et al., *Nature* 2020) reaches state of the art with a model that never reconstructs observations. This is the theoretical reason encoder capacity spent on reconstruction can be pure waste.
- **Representation-side capacity has a well-defined sufficiency notion.** Bisimulation metrics (Ferns et al., UAI 2004) and DBC (Zhang et al., ICLR 2021; arXiv:2006.10742) show that a control-sufficient encoder can discard most observation content, so encoder needs scale with task-relevant, not pixel, complexity.
- **Distractors shift the burden.** Denoised-MDP-style work (Wang et al., ICML 2022) shows reconstruction-driven encoders degrade sharply under exogenous visual noise, while reward/value-driven ones degrade less — evidence that $\rho^\star$ is task-distribution dependent.

## 5. What Is Not Known

- **Empirically open (primary).** No published iso-FLOP sweep over $\rho$ at $\ge 3$ budgets in any world-model family. The experiment is runnable today on Atari-100k or DMC at 10–500M parameters; nobody has reported it. Whether $J$ is even non-flat in $\rho$ beyond seed noise is unmeasured.
- **Empirically open (secondary).** Whether $\rho^\star$ drifts with $N$, with observation dimensionality, with horizon, or with distractor level.
- **Theoretically open.** No bound relating optimal encoder capacity to a task-intrinsic quantity (bisimulation-metric covering number, or the entropy of the value-relevant sigma-algebra) and dynamics capacity to transition complexity. Ferns' and Grimm's results characterize sufficiency, not the capacity needed to realize it.
- **Methodologically blocked.** "Encoder capacity" is not well defined in architectures where the encoder is trained end-to-end through the dynamics loss (RSSM, MuZero): parameters sit in the encoder but represent transition structure. Until an attribution measure exists — e.g. per-module influence on latent prediction error under module-wise capacity ablation — the independent variable is ill-posed for exactly the models people use.

## 6. Why It Is Hard

**Confounded measurement plus a resolution problem.** Changing $\rho$ at fixed $N$ changes latent width, effective receptive field, optimizer conditioning, and FLOPs-per-parameter simultaneously. The effect size being sought is plausibly 5–15% in normalized score, while seed variance on Atari-100k across 5 seeds is routinely 20–30% of the mean for individual games. Distinguishing a 10% allocation effect therefore requires enough seeds and enough tasks that the sweep costs more than a single frontier training run — and the returns are measured through a policy-learning stage whose own hyperparameters interact with model size. Add non-identifiability: two splits can induce the same value-equivalent model class, so loss-based selection cannot rank them.

## 7. Current Research (as of 2026)

- **MBRL scaling.** Follow-ups to TD-MPC2 and DreamerV3 on multi-task scaling; the tunable of record is total size, not split.
- **Generative/interactive video world models.** Genie-lineage systems (DeepMind) and diffusion world models (DIAMOND lineage) push almost all capacity into dynamics with a fixed, comparatively small tokenizer — an unstated allocation prior *(frontier — verify)*.
- **Decoder-free representation learning.** JEPA-family encoders (Meta FAIR) used as frozen front-ends with small trained dynamics heads — effectively $\rho \to 1$ in pretraining, $\rho \to 0$ in adaptation *(frontier — verify)*.
- **Tokenizer-capacity studies in video generation.** Growing evidence that tokenizer quality bounds downstream dynamics quality; the transfer of this to control is unpublished *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** DreamerV3 on Atari-100k (26 games) plus 10 DMC tasks, three iso-FLOP budgets: 25M, 100M, 400M total parameters. At each budget, five splits $\rho \in \{0.15, 0.3, 0.5, 0.7, 0.85\}$ with latent width $\dim(z)$ held fixed at 1024 (vary depth/width only elsewhere), 5 seeds. 3 × 5 × 36 × 5 = 2,700 runs; at roughly 8 GPU-hours per Atari-100k run this is ~20k A100-hours — comparable to one mid-size pretraining run.

**Control arm.** The published DreamerV3 recipe at each budget (its native $\rho$), retuned only for learning rate, so the comparison is against a tuned baseline rather than a strawman.

**Deciding number.** $\Delta = \max_\rho \overline{\text{IQM}}(\rho) - \overline{\text{IQM}}(\rho_{\text{baseline}})$, IQM of human-normalized score with stratified bootstrap 95% CI. **If $\Delta < 0.05$ IQM with the CI excluding 0.10 at all three budgets, allocation is a second-order knob and the field should stop tuning it.** If $\Delta \ge 0.10$ and $\arg\max_\rho$ shifts monotonically with $N$, the fitted $\rho^\star(N)$ is the first allocation law for world models.

Secondary readout: does $\arg\max_\rho$ move toward the encoder when Atari observations are corrupted with a natural-video background distractor? That separates "task complexity" from "observation complexity".

## 9. Key References

- **[Foundational]** Ha, Schmidhuber. *World Models.* NeurIPS 2018. — arXiv:1803.10122
- **[Foundational]** Ferns, Panangaden, Precup. *Metrics for Finite Markov Decision Processes.* UAI 2004.
- **[Foundational]** Grimm, Barreto, Singh, Silver. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* NeurIPS 2020. — arXiv:2011.03506
- **[Foundational]** Schrittwieser et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature, 2020. — arXiv:1911.08265
- **[SOTA]** Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Control Tasks through World Models.* Nature, 2025. — arXiv:2301.04104
- **[SOTA]** Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024. — arXiv:2310.16828
- **[SOTA]** Bruce et al. *Genie: Generative Interactive Environments.* ICML 2024. — arXiv:2402.15391
- **[SOTA]** Alonso et al. *Diffusion for World Modeling: Visual Details Matter in Atari.* NeurIPS 2024. — arXiv:2405.12399
- **[Method]** Micheli, Alonso, Fleuret. *Transformers are Sample-Efficient World Models.* ICLR 2023. — arXiv:2209.00588
- **[Method]** Zhang, McAllister, Calandra, Gal, Levine. *Learning Invariant Representations for Reinforcement Learning without Reconstruction.* ICLR 2021. — arXiv:2006.10742
- **[Method]** Wang, Zhu, Torralba, Isola. *Denoised MDPs: Learning World Models Better Than the World Itself.* ICML 2022.
- **[Scaling]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Scaling]** Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Evaluation]** Agarwal, Schwarzer, Castro, Courville, Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021. — arXiv:2108.13379
- **[Related]** Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA).* 2024.

## 10. Worked Example

Take a 100M-parameter budget on Atari-100k, 64×64×3 frames.

**Arm A (encoder-heavy, $\rho=0.8$):** 80M CNN encoder, 20M dynamics. FLOPs per frame are dominated by the convolutional stack at full spatial resolution — call it $\sim$4 GFLOPs — while the dynamics operates on a 32×32 token grid at $\sim$0.4 GFLOPs. **Arm B (dynamics-heavy, $\rho=0.2$):** 20M encoder, 80M dynamics: $\sim$1 GFLOPs encoder, $\sim$1.6 GFLOPs dynamics.

Same parameter count; Arm A costs roughly 1.7× the training FLOPs of Arm B. Run them iso-parameter and you have not run a controlled experiment — you compared 100M-at-4.4-GFLOPs against 100M-at-2.6-GFLOPs. Run them iso-FLOP instead and Arm A must shrink to about 60M parameters, so the parameter budget is no longer fixed. **There is no setting in which both are held constant.**

Now the effect size. Suppose the true allocation effect is +8% IQM human-normalized score. On Atari-100k, per-game scores across 5 seeds routinely have coefficient of variation 0.2–0.3 (Agarwal et al., 2021, is the standard reference for why 3-seed comparisons at this scale are uninterpretable). Aggregating 26 games recovers roughly a $\sqrt{26}\approx 5\times$ reduction in the aggregate standard error, giving $\sim$4–6% — the same order as the effect. Two arms, 26 games, 5 seeds is 260 runs, and the 95% interval still straddles zero.

The obstruction is visible in both halves: the independent variable cannot be isolated (parameters and FLOPs move together), and the dependent variable's noise floor sits at the effect size. That is why the answer is empirically open rather than merely unpublished — and why the deciding experiment in §8 is specified as three budgets × five splits × five seeds rather than a single head-to-head.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*