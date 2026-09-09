---
id: 35-world-models/measuring-world-model-consistency
title: "Measuring World Model Consistency Independent of Task Reward"
topic: 35-world-models
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Measuring World Model Consistency Independent of Task Reward

> **Topic:** World Models & Planning · **ID:** `35-world-models/measuring-world-model-consistency` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a learned world model $\hat M$ — anything that maps a history of observations and actions to a predicted future — produce a scalar or vector **consistency score** that (i) is computable without any task reward, (ii) is invariant to reparametrization of the model's latent space, and (iii) predicts downstream planning competence on reward functions not seen during training or scoring.

Three variants, of different difficulty:

- **Measurement variant (the blocked one).** Define the score. Nobody has a definition that survives both the invariance requirement and the coverage requirement simultaneously. Latent-space metrics fail invariance; observation-space rollout error fails to separate "the model is inconsistent" from "the decoder is blurry"; behavioral probes fail coverage, because the histories on which a model is inconsistent are exactly the ones absent from the evaluation distribution.
- **Method variant.** Given a fixed score, train models to optimize it. Tractable but downstream of the measurement.
- **Theory variant.** Prove whether any reward-free consistency functional can be a *sufficient* statistic for planning return over a reward class $\mathcal{R}$. Value-equivalence theory (Grimm et al., NeurIPS 2020) says the answer depends entirely on $\mathcal{R}$; a tight characterization is open.

Solving it means: a published metric, computed on a zoo of world models, that rank-orders them the same way held-out-reward planning return does, and does not move when you apply a random invertible map to the latents.

## 2. Formal Setting

Environment: a POMDP $\mathcal{E}=(\mathcal{S},\mathcal{A},T,\Omega,O,R,\gamma)$. A world model is a tuple $\hat M=(\mathcal{Z}, e_\phi, f_\theta, d_\psi)$ with encoder $e_\phi:\mathcal{H}\to\mathcal{Z}$ over histories $h=(o_0,a_0,\dots,o_t)$, latent transition $f_\theta:\mathcal{Z}\times\mathcal{A}\to\mathcal{Z}$, decoder $d_\psi:\mathcal{Z}\to\Delta(\Omega)$. Write $f_\theta(z,a_{1:k})$ for the $k$-step composition.

**Candidate 1 — path-independence error.** Let $\Pi$ be a set of action-sequence pairs known to be equivalent in $\mathcal{E}$ (commuting moves, provable no-ops, loops that close). Measured as
$$C_{\text{path}}=\mathbb{E}_{h\sim\mathcal{D},\,(u,u')\sim\Pi}\big[d_{\mathcal Z}\big(f_\theta(e_\phi(h),u),\,f_\theta(e_\phi(h),u')\big)\big].$$
Requires no reward and no simulator, but requires $\Pi$ — which is domain knowledge, not data.

**Candidate 2 — encoder/transition commutation (latent cycle error).** Roll the real environment one step to get $h'=hao'$; measure $\mathbb{E}[d_{\mathcal Z}(f_\theta(e_\phi(h),a),\,e_\phi(h'))]$. Needs a simulator and is pinned to $\mathcal{D}$.

**Candidate 3 — Myhill–Nerode precision.** For deterministic sequence domains, two histories are equivalent iff they accept the same suffixes. Vafa et al. (NeurIPS 2024) measure **compression precision** (fraction of model-merged history pairs that are truly equivalent) and **distinction precision** (fraction of model-separated pairs that are truly distinct). Exact ground truth; only defined where the environment is a known DFA.

**Candidate 4 — value-equivalence residual.** For policy set $\Pi_v$ and value class $\mathcal{V}$, $\;\sup_{\pi\in\Pi_v,v\in\mathcal{V}}\|\mathcal{T}^\pi_{\hat M}v-\mathcal{T}^\pi_{M}v\|_\infty$. Reward-free only if $R\equiv 0$, in which case it collapses to agreement of discounted occupancy operators.

**Non-identifiability.** For any invertible $g:\mathcal{Z}\to\mathcal{Z}$, the model $(g\circ e_\phi,\, g f_\theta g^{-1},\, d_\psi\circ g^{-1})$ is behaviorally identical but changes $C_{\text{path}}$ and the latent cycle error by an arbitrary factor. Any metric using $d_{\mathcal Z}$ measures the coordinate system as much as the model.

**Assumptions, and which are violated.** (a) Determinism — violated for video and for any stochastic environment; path-independence must then be stated between distributions, and the choice of divergence changes the ranking. (b) Known equivalence set $\Pi$ — violated outside synthetic domains. (c) Evaluation coverage: $\mathcal{D}$ must include histories where inconsistency manifests — systematically violated, since models are inconsistent off-distribution and $\mathcal{D}$ is on-distribution. (d) A fixed $\mathcal{Z}$ geometry — violated by construction (above).

## 3. State of the Art

**Established.**
- *Value equivalence* (Grimm, Barreto, Singh, Silver, NeurIPS 2020): models that are wrong about dynamics can be exactly right for planning over a restricted $(\Pi_v,\mathcal{V})$; the value-equivalent set shrinks monotonically as those sets grow. This is a theorem, and it is the reason reward-free consistency is not automatically the right target.
- *Objective mismatch* (Lambert, Wilcox, Zhang, Pister, Calandra, L4DC 2020): lower one-step model likelihood does not imply higher control return; shown empirically on PETS-style continuous control.
- *Value-aware model loss* (Farahmand, Barreto, Nikovski, AISTATS 2017): the correct model loss for control is reward-weighted, not reward-free.
- *Probe confounding* (Hewitt & Liang, EMNLP 2019): probe accuracy without a control-task selectivity baseline does not establish that a representation encodes a variable. Most "world model probing" papers still report accuracy alone.

**Claimed but unablated.** That rollout-consistency or "physics-plausibility" scores on video generators measure a world model. Physics-IQ (Motamed et al., 2025) and VideoPhy (Bansal et al., 2024) are benchmark numbers: they report which generations look physical to a classifier or human, with no ablation showing the score predicts planning transfer, and no invariance analysis.

**Benchmark-only results.** Genie-family interactive video world models (DeepMind, 2024–2025) report long-horizon "consistency" qualitatively and by human preference. No public reward-free metric, no model zoo, no rank-correlation study. *(frontier — verify)*

## 4. What Is Known

- **High next-step accuracy is compatible with an incoherent map.** Vafa et al. (NeurIPS 2024) trained transformers on NYC taxi trajectories: near-perfect next-turn prediction, yet compression/distinction precision is low and the reconstructed street graph contains streets that do not exist. Accuracy degrades sharply when detours are inserted. Scale: ~100M-parameter models, millions of trajectories.
- **Probe class changes the verdict.** Othello-GPT (Li et al., ICLR 2023, 25M params, 20M synthetic games) has 0.01% illegal-move rate; a nonlinear board probe reaches ~1.7% error while a linear probe in the naive black/white basis reaches ~20–26%. Nanda et al. (BlackboxNLP 2023) showed a *linear* probe in a "mine vs. theirs" frame is near-perfect. The same model was "not linearly readable" and "linearly readable" depending on coordinates — a direct instance of the non-identifiability problem.
- **Reconstruction is not required for competence.** MuZero (Schrittwieser et al., Nature 2020) plans with a latent trained only on reward, value and policy targets; it reaches superhuman Atari and Go while its latent has no defined observation-space error. Any consistency metric requiring a decoder is undefined for it.
- **Scaling does not fix out-of-distribution physics.** Kang et al. (2024, arXiv:2411.02385) trained video diffusion models on synthetic collision/parabola/uniform-motion data: in-distribution error falls with data and parameters; out-of-distribution error is flat, and models resolve OOD cases by nearest-neighbour retrieval of training clips rather than by rule. Scale: up to ~300M-param DiT, ~6M videos.
- **Physics-IQ (2025)** scores the best evaluated video generator far below the physical-variance ceiling (reported ≈24 on a 0–100 scale). This is a benchmark number, not a validated consistency measure.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no reward-free consistency functional that is (i) reparametrization-invariant, (ii) defined for decoder-free latent models like MuZero, and (iii) computable off the training distribution. Every published candidate fails at least one. The measurement is not yet well defined, so no amount of compute settles it.
- **Empirically open.** Nobody has built a world-model zoo ($N\gtrsim 30$) on one environment and reported rank correlation between candidate reward-free metrics and held-out-reward planning return. The experiment is cheap at MiniGrid/Crafter scale and has not been run.
- **Theoretically open.** Whether there exists a reward-free functional $C$ such that, for a stated reward class $\mathcal{R}$, $C(\hat M)\le\epsilon$ implies a uniform planning-suboptimality bound over $\mathcal{R}$ with constants independent of $|\mathcal{S}|$. Bisimulation metrics (Ferns, Panangaden, Precup, UAI 2004) give such bounds but are reward-dependent by construction; the reward-free quotient is coarser and its sufficiency is unproven.

## 6. Why It Is Hard

Three named obstructions, all of which bite at once.

1. **Non-identifiability of the latent metric.** The latent space is defined only up to invertible reparametrization. Any score built on $d_{\mathcal Z}$ can be inflated or deflated arbitrarily without changing a single model prediction.
2. **Absent ground truth off-distribution.** Consistency violations concentrate where the training distribution is thin. There, the environment's true equivalence relation is unknown unless you have a simulator — and if you have a simulator you did not need the learned model.
3. **Evaluation that does not measure what it names.** "Consistency" benchmarks for video and text world models score plausibility to a judge. A model that memorizes plausible continuations scores well while retrieving, not simulating — exactly the failure Kang et al. isolated.

Compute is *not* the obstruction. The relevant experiments are sub-1000-GPU-hour.

## 7. Current Research (as of 2026)

- **Automata-theoretic evaluation.** Vafa, Chen, Rambachan, Kleinberg, Mullainathan (Harvard/MIT) — Myhill–Nerode metrics, and inductive-bias probes (*What Has a Foundation Model Found?*, ICML 2025) that test whether a model's fine-tuning bias matches the true state variable rather than whether a probe can read it out.
- **Value-equivalence and proper models.** DeepMind lineage from MuZero through the value-equivalence principle; the reward-free limit is under-explored.
- **Consistency checks without ground truth.** Fluri, Paleka, Tramèr (ETH Zürich, SaTML 2024) — evaluating superhuman models by internal contradiction. Directly transferable to world models and largely untried there.
- **Interactive video world models.** DeepMind Genie 3, and open replications, report long-horizon coherence; metric definitions remain proprietary or qualitative *(frontier — verify)*.
- **Cognitive-map evaluation.** Momennejad et al. (CogEval, NeurIPS 2023) — graph-structured probes of LLM planning, reward-free by design.

## 8. Concrete Next Experiment

**The metric-validation zoo.**

- **Scale.** One environment with a factored state (Crafter, or a $16\times16$ MiniGrid with objects). Train $N=40$ world models spanning architectures (RSSM, transformer, CSWM-style contrastive, MuZero-style value-only) and training budgets, each ~10M parameters, $10^7$ env steps. ~150 GPU-hours total.
- **Held-out rewards.** $K=20$ reward functions never used in model training or scoring. Plan with a fixed MPC/MCTS procedure; record mean normalized return $\bar G_i$ per model.
- **Metrics under test.** $C_{\text{path}}$, latent cycle error, compression/distinction precision, $n$-step decoded rollout error, and a contradiction-rate score.
- **Control arms.** (a) **Reward-informed control:** the same value-equivalence residual computed with access to *one* of the 20 rewards. (b) **Reparametrization control:** apply a random invertible linear $g$ to each latent space and recompute every metric. (c) **Selectivity control** in the Hewitt–Liang sense for any probe-based metric.
- **The deciding number.** Kendall $\tau$ between each reward-free metric and $\bar G$ across the 40 models. **A reward-free metric is validated if $\tau\ge0.7$ and its value shifts by $<1\%$ under $g$.** If the reward-informed control reaches $\tau\ge0.7$ and no reward-free metric does, the field's premise — that consistency can be scored without reward — is empirically refuted at this scale, and the page moves from *methodologically blocked* to *partially answered, negatively*.

## 9. Key References

- **[Foundational]** Ha, D., Schmidhuber, J. *World Models.* NeurIPS, 2018. — arXiv:1803.10122
- **[Foundational]** Ferns, N., Panangaden, P., Precup, D. *Metrics for Finite Markov Decision Processes.* UAI, 2004.
- **[Theory SOTA]** Grimm, C., Barreto, A., Singh, S., Silver, D. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* NeurIPS, 2020.
- **[Theory]** Farahmand, A., Barreto, A., Nikovski, D. *Value-Aware Loss Function for Model-based Reinforcement Learning.* AISTATS, 2017.
- **[SOTA — measurement]** Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., Mullainathan, S. *Evaluating the World Model Implicit in a Generative Model.* NeurIPS, 2024. — arXiv:2406.03689
- **[SOTA — measurement]** Vafa, K., Chang, P. G., Rambachan, A., Mullainathan, S. *What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models.* ICML, 2025.
- **[Empirical]** Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., Wattenberg, M. *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task.* ICLR, 2023. — arXiv:2210.13382
- **[Empirical]** Nanda, N., Lee, A., Wattenberg, M. *Emergent Linear Representations in World Models of Self-Supervised Sequence Models.* BlackboxNLP @ EMNLP, 2023.
- **[Empirical]** Lambert, N., Wilcox, A., Zhang, H., Pister, K., Calandra, R. *Objective Mismatch in Model-based Reinforcement Learning.* L4DC, 2020.
- **[Empirical]** Kang, B., et al. *How Far is Video Generation from World Model: A Physical Law Perspective.* 2024. — arXiv:2411.02385
- **[Methodology]** Hewitt, J., Liang, P. *Designing and Interpreting Probes with Control Tasks.* EMNLP, 2019.
- **[Methodology]** Fluri, L., Paleka, D., Tramèr, F. *Evaluating Superhuman Models with Consistency Checks.* IEEE SaTML, 2024.
- **[Systems]** Schrittwieser, J., et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature 588, 2020.
- **[Survey]** Belinkov, Y. *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1), 2022.

## 10. Worked Example

Environment: a $5\times5$ **torus** gridworld, 25 states, actions $\{E,W,N,S\}$, deterministic, fully observed, no reward. Training data: every action sequence of length $\le 3$ from a fixed start state — 84 trajectories, complete coverage of that horizon.

- **Model A** learns the true torus: latent $=\mathbb{Z}_5^2$.
- **Model B** learns an unwrapped plane: latent $=\mathbb{Z}^2$, no modular arithmetic.

**Both models are exactly correct on the training distribution.** Within 3 steps the agent cannot leave $[-3,3]^2$, so no wrap occurs. Next-observation accuracy: 100.0% for both. Path-independence error over commuting pairs of length $\le 3$ (e.g. $EN$ vs $NE$): $C_{\text{path}}=0.000$ for both.

**Where they differ.** Over the $4^5=1024$ action sequences of length 5, a wrap occurs iff $|\Delta x|\ge3$ or $|\Delta y|\ge3$. Counting multinomials: $|\Delta x|\ge3$ holds for 112 sequences, $|\Delta y|\ge3$ for another 112, and both cannot hold at once (that needs 6 steps). So **224/1024 = 21.9%** of length-5 sequences separate A from B — and **0/84** of the length-$\le3$ sequences do.

**The obstruction, made visible.**
1. *Coverage.* Any consistency metric evaluated on $\mathcal{D}$ reports A and B as equally consistent. The 21.9% signal exists only one step beyond the data horizon, and in a real environment you cannot certify what the correct answer is out there.
2. *Non-identifiability.* Give B a latent embedding $z\mapsto Rz$ with $R$ a random $2\times2$ rotation with condition number 8. Behavior is unchanged — still 100% on $\mathcal{D}$, still wrong on 21.9% of 5-step sequences — but a latent-MSE consistency score computed on 5-step loops moves from 0.03 to 0.41, a $13\times$ swing driven purely by coordinates.
3. *Reward-dependence.* If the task reward only ever places goals within 2 steps of the start, B is a *perfect* model for planning: value-equivalent to A on the whole reward class. Calling B "inconsistent" is only meaningful once you name the reward class you refuse to condition on — which is exactly the thing the problem asks you to do without.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*