---
id: 35-world-models/latent-state-identifiability-world-models
title: "Latent State Identifiability in Recurrent World Models"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latent State Identifiability in Recurrent World Models

> **Topic:** World Models & Planning · **ID:** `35-world-models/latent-state-identifiability-world-models` · **Status:** open

## 1. Problem Statement

A recurrent world model consumes an interaction stream $(o_1,a_1,\dots,o_T)$ and learns a latent state $\hat z_t$ with a transition rule and a decoder. It is trained on a *predictive* objective — reconstruct $o_{t+1}$, or predict a value/reward. The question: **does fitting the observable stream pin down the latent state, or only pin down its predictions?**

Three variants, routinely conflated:

- **Theory variant.** Given the true generative process $(z_t, a_t, o_t)$, for which model classes and which intervention/action distributions is $\hat z_t = h(z_t)$ forced, with $h$ in a stated equivalence class (permutation + elementwise reparameterization, affine map, or bisimulation quotient)? A solution is a theorem with checkable premises.
- **Measurement variant.** Given a trained model and no ground-truth latents, decide whether the learned state is a faithful state — not merely a sufficient statistic for the training-distribution prediction loss. A solution is an estimator with a control arm that does not confound *probe capacity* with *state content*.
- **Method variant.** Design an objective or data-collection scheme whose optimum is identifiable at scale, without destroying control performance.

Solving it means: an identifiability theorem whose assumptions hold in a real pixel-based control domain, plus a measurement that certifies it post hoc.

## 2. Formal Setting

Ground-truth process: a controlled state-space model with latent $z_t \in \mathcal{Z} \subseteq \mathbb{R}^n$, action $a_t \in \mathcal{A}$, observation $o_t \in \mathcal{O}$,

$$z_t = f(z_{t-1}, a_{t-1}, \epsilon_t), \qquad o_t = g(z_t) + \eta_t,$$

with $g$ injective (an assumption, see below) and $\epsilon_t$ mutually independent across coordinates given $(z_{t-1},a_{t-1})$.

The learner fits $\hat\theta = (\hat f, \hat g, q_\phi)$ maximizing an ELBO or a multi-step prediction loss

$$\mathcal{L}(\theta) = \mathbb{E}_{\pi}\Big[\sum_{t}\; -\log p_\theta(o_{t+1:t+H} \mid o_{\le t}, a_{\ge t})\Big],$$

under behaviour policy $\pi$ and horizon $H$.

**Identifiability.** Let $\mathcal{P}_\theta^\pi$ be the induced law over observable trajectories. Define the *observational equivalence class* $[\theta] = \{\theta' : \mathcal{P}_{\theta'}^{\pi} = \mathcal{P}_{\theta}^{\pi}\}$. The model is $\mathcal{E}$-identifiable if $\theta' \in [\theta] \Rightarrow \hat z' = h(\hat z)$ for some $h \in \mathcal{E}$. Standard choices: $\mathcal{E}_{\mathrm{perm}}$ (permutation + monotone elementwise), $\mathcal{E}_{\mathrm{aff}}$ (invertible affine), $\mathcal{E}_{\mathrm{bisim}}$ (bisimulation quotient, Ferns et al. 2004).

**Measured quantities, as actually computed.**

- *MCC* — fit the Hungarian-matched maximum of mean absolute Pearson (or Spearman) correlation between $\hat z$ and $z$ coordinates, on **held-out** trajectories, after fitting any per-coordinate map on a disjoint split. Reporting MCC fit and evaluated on the same split is the single most common inflation.
- *Probe $R^2$*: $R^2 = 1 - \mathbb{E}\|z - m(\hat z)\|^2 / \mathrm{Var}(z)$, with $m$ linear or a fixed-capacity MLP. Only interpretable against a **control task** (Hewitt & Liang, EMNLP 2019): the same probe on randomly-permuted labels; report selectivity, not raw accuracy.
- *Seed dispersion*: $D = \mathbb{E}_{i\ne j}\,[1 - \mathrm{MCC}(\hat z^{(i)}, \hat z^{(j)})]$ over training seeds. $D>0$ with equal validation loss is direct evidence of non-identifiability, and needs no ground truth.
- *Off-policy divergence*: $\Delta = \mathbb{E}_{\pi'}[\mathcal{L}] - \mathbb{E}_{\pi}[\mathcal{L}]$ for a shifted policy $\pi'$ — the operational cost of an unidentified state.

**Assumptions known to be violated in practice.** (i) $g$ injective — partial observability and occlusion break it in every pixel domain. (ii) Independent noise coordinates — DreamerV3's categorical latents are trained with a KL to a learned prior, not an independent one. (iii) Stationary $\pi$ — RL data is collected under a policy that changes during training, so $\mathcal{P}^\pi$ is a moving target. (iv) Exact optimum — theorems are about the population argmin; SGD returns one point in a basin.

## 3. State of the Art

**Theory SOTA (established).**
- Hyvärinen & Pajunen (*Neural Networks*, 1999): unconditional nonlinear ICA is unidentifiable — infinitely many independent-component solutions exist.
- Locatello et al., ICML 2019 (best paper): unsupervised disentanglement is impossible without inductive bias, for any factorized prior.
- The escape route is auxiliary structure: Hyvärinen & Morioka (NeurIPS 2016, time-contrastive learning), Hyvärinen–Sasaki–Turner (AISTATS 2019), Khemakhem et al. iVAE (AISTATS 2020) — identifiability up to permutation and elementwise transform given a conditionally-factorized exponential-family prior with sufficient variability in the auxiliary variable.
- Temporal/interventional extensions: LEAP (Yao et al., ICLR 2022), TDRL (Yao et al., NeurIPS 2022), CITRIS (Lippe et al., ICML 2022) and BISCUIT (Lippe et al., UAI 2023) — identifiability from temporal sequences with known or binary interactions; mechanism-sparsity (Lachapelle et al., CLeaR 2022).
- Discrete case: finite-state HMMs are identifiable up to label permutation under Kruskal rank conditions (Allman, Matias & Rhodes, *Annals of Statistics*, 2009); Hsu, Kakade & Zhang (*JCSS*, 2012) give a spectral estimator with sample complexity polynomial in $1/\sigma_m$, the $m$-th singular value of the joint observation matrix.

**Established that identifiability is not required for control.** Grimm et al. (NeurIPS 2020) formalize the *value-equivalence principle*: models agreeing on Bellman backups for a policy/function set are interchangeable for planning. MuZero (Schrittwieser et al., *Nature*, 2020) is the existence proof — a latent with no reconstruction constraint at all.

**Empirical SOTA.** DreamerV3 (Hafner et al., *Nature*, 2025; arXiv:2301.04104) with fixed hyperparameters across 150+ tasks; TD-MPC2 (Hansen et al., ICLR 2024); IRIS (Micheli et al., ICLR 2023); Genie (Bruce et al., ICML 2024). None of these report an identifiability measurement. Their latents are evaluated by return and by video rollout quality only.

**Claimed but unablated.** That scaling a world model makes its latent "more veridical." No paper reports MCC or seed dispersion as a function of parameter count for a control world model. Rollout FID/PSNR is a benchmark number, not evidence about state content.

## 4. What Is Known

- **Non-identifiability is the default, and is quantified at small scale.** Locatello et al. trained over 12,000 models across seven datasets (dSprites, Shapes3D, etc., $64\times64$ images) and found disentanglement metrics uncorrelated with unsupervised model-selection criteria; seed variance dominated hyperparameter choice.
- **Predictive accuracy does not certify a correct model.** Vafa et al. (NeurIPS 2024) train transformers on Manhattan taxi routes to near-perfect next-turn prediction; the recovered map, reconstructed from the model, contains streets that do not exist, and detour-robustness collapses. Their Myhill–Nerode-based metrics separate the two.
- **A world model can be present and linearly readable.** Othello-GPT (Li et al., ICLR 2023): an 8-layer GPT with 0.01% illegal-move rate; nonlinear probes recover board state at ~1.7% error while linear probes fail (~20%+ error). Nanda et al. (BlackboxNLP 2023) show the state *is* linear in the mine/yours/blank basis, with near-ceiling probe accuracy. **Lesson: probe failure is evidence about the probe's basis, not about the model.**
- **Weak supervision buys identifiability cheaply.** Locatello et al. (ICML 2020) show paired observations sharing an unknown subset of factors suffice, empirically raising disentanglement scores substantially on the same $64\times64$ benchmarks.
- **Objective mismatch is real.** Lambert et al. (L4DC 2020): model validation loss and control return are weakly, sometimes negatively, correlated on MuJoCo tasks.

## 5. What Is Not Known

- **Theoretically open.** No identifiability theorem covers the actual DreamerV3 architecture: categorical (non-exponential-family-continuous) latents, learned non-factorized prior, recurrent deterministic carry $h_t$ alongside stochastic $z_t$, and a non-stationary data policy. Whether *action variability* alone — actions as the auxiliary variable in the iVAE sense — suffices for $\mathcal{E}_{\mathrm{perm}}$-identifiability under a policy that becomes deterministic as it improves is unproven either way.
- **Empirically open.** Seed dispersion $D$ versus scale for a pixel control world model: runnable today on DMC or Crafter with ~10 seeds × 4 model sizes, never reported.
- **Methodologically blocked.** "Faithful latent state" has no ground-truth-free definition on domains where ground truth is unavailable (video, robotics). MCC needs $z$; probes need labels; bisimulation distance needs the true MDP. Seed dispersion is the only ground-truth-free proxy, and it detects non-identifiability without certifying identifiability.

## 6. Why It Is Hard

**Non-identifiability is an obstruction of the objective, not of the optimizer.** The training loss is a functional of $\mathcal{P}^\pi_\theta$ alone. Every $\theta'\in[\theta]$ is an exact global optimum. No amount of compute, data from $\pi$, or architecture search separates them: the information required is not in the objective.

**Compounding this: the measurement is confounded.** Probe accuracy mixes (a) whether the information is present, (b) whether it is in the probe's hypothesis class, and (c) probe overfitting. Othello-GPT shows (b) alone can flip the verdict. Without a control task, a "the model has no world model" result and a "the probe was linear" result are indistinguishable.

**And the natural fix is expensive.** Breaking the equivalence class needs off-policy or interventional data — deliberately visiting states a good policy avoids. That is exactly the data an RL agent is optimized not to collect, so identifiability and sample-efficient control pull in opposite directions.

## 7. Current Research (as of 2026)

- **Causal representation learning from interaction** — Lippe, Magliacane, Gavves (Amsterdam); Zhang and Yao (CMU/MBZUAI); Lachapelle and Lacoste-Julien (Mila); Ahuja et al. (interventional CRL, ICML 2023). Direction: weaken "known interventions" to "unknown, sparse, action-induced". Extension to high-dimensional video remains at benchmark-toy scale *(frontier — verify)*.
- **World-model evaluation beyond return** — Vafa, Rambachan, Mullainathan (Harvard/MIT) on Myhill–Nerode-style recovery metrics; extension to continuous-state control is being attempted *(frontier — verify)*.
- **Mechanistic probing of large video/sequence models** — Bau's group, Nanda's interpretability group; question is whether scaling video generators (Genie-class) yields linearly-decodable object state.
- **Value-equivalent models at scale** — the counter-position: identifiability is the wrong target; certify planning-sufficiency instead.

## 8. Concrete Next Experiment

**Question.** Does a recurrent world model's latent become more identifiable with scale, or only more predictive?

**Scale.** DeepMind Control Suite, 5 tasks with known ground-truth state $z_t$ (cartpole-swingup $n=4$, walker-walk $n=18$, cheetah-run $n=17$, reacher-hard, finger-spin), pixel observations $64\times64$. DreamerV3 at four sizes (12M / 25M / 75M / 200M parameters), 10 seeds each, 1M environment steps. Total 200 runs, ~2–4 GPU-hours per run on an A100 ≈ 600 GPU-hours.

**Measurements per run.** (i) return; (ii) MCC$(\hat z, z)$ with Hungarian matching, fit on 80% and evaluated on a held-out 20% of trajectories; (iii) seed dispersion $D$ over the 10 seeds; (iv) linear-probe $R^2$ with a control-task selectivity baseline.

**Control arm.** Two, both required. **(a)** A random-init frozen encoder with the same architecture, probed identically — bounds how much $R^2$ comes from the probe. **(b)** A *shuffled-action* model: identical training but with actions replaced by an independent draw from the same marginal, killing the auxiliary variable that identifiability theory relies on. If the real model is identified *because of* action conditioning, arm (b) must show strictly worse MCC at equal reconstruction loss.

**Deciding number.** **Seed dispersion $D$ at 200M parameters, among the seeds whose final return is within 5% of each other.** If $D < 0.10$ (i.e. cross-seed MCC $> 0.90$), scale drives the latent toward a canonical representation and the empirical question closes in the affirmative. If $D > 0.30$ and flat in model size, predictive scaling does not buy identifiability, and the theory variant is the only route. A monotone decrease in $D$ with size that has not plateaued is the interesting third outcome and sets the next scale point.

## 9. Key References

- **[Foundational]** A. Hyvärinen, P. Pajunen. *Nonlinear independent component analysis: Existence and uniqueness results.* Neural Networks 12(3), 1999.
- **[Foundational]** D. Ha, J. Schmidhuber. *World Models.* NeurIPS, 2018. — arXiv:1803.10122
- **[Foundational]** F. Locatello, S. Bauer, M. Lucic, G. Rätsch, S. Gelly, B. Schölkopf, O. Bachem. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML, 2019. — arXiv:1811.12359
- **[Foundational]** I. Khemakhem, D. Kingma, R. Monti, A. Hyvärinen. *Variational Autoencoders and Nonlinear ICA: A Unifying Framework.* AISTATS, 2020. — arXiv:1907.04809
- **[Foundational]** E. Allman, C. Matias, J. Rhodes. *Identifiability of parameters in latent structure models with many observed variables.* Annals of Statistics 37(6A), 2009.
- **[Foundational]** D. Hsu, S. Kakade, T. Zhang. *A spectral algorithm for learning hidden Markov models.* Journal of Computer and System Sciences 78(5), 2012.
- **[SOTA]** D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering diverse control tasks through world models.* Nature, 2025. — arXiv:2301.04104
- **[SOTA]** P. Lippe, S. Magliacane, S. Löwe, Y. Asano, T. Cohen, E. Gavves. *CITRIS: Causal Identifiability from Temporal Intervened Sequences.* ICML, 2022.
- **[SOTA]** P. Lippe et al. *BISCUIT: Causal Representation Learning from Binary Interactions.* UAI, 2023.
- **[SOTA]** W. Yao, G. Chen, K. Zhang. *Temporally Disentangled Representation Learning.* NeurIPS, 2022.
- **[SOTA]** C. Grimm, A. Barreto, S. Singh, D. Silver. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* NeurIPS, 2020.
- **[Measurement]** K. Vafa, J. Chen, A. Rambachan, J. Kleinberg, S. Mullainathan. *Evaluating the World Model Implicit in a Generative Model.* NeurIPS, 2024.
- **[Measurement]** J. Hewitt, P. Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP, 2019.
- **[Measurement]** K. Li, A. Hopkins, D. Bau, F. Viégas, H. Pfister, M. Wattenberg. *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task.* ICLR, 2023.
- **[Survey]** B. Schölkopf, F. Locatello, S. Bauer, N. R. Ke, N. Kalchbrenner, A. Goyal, Y. Bengio. *Toward Causal Representation Learning.* Proceedings of the IEEE 109(5), 2021.

## 10. Worked Example

Take cartpole-swingup: true state $z = (x, \dot x, \theta, \dot\theta)$, $n=4$, pixel observations. Suppose two runs, seed A and seed B, both reach return 860 and identical held-out reconstruction NLL to three digits.

Construct the obstruction explicitly. Let $\hat z^{B}_t = R\,\hat z^{A}_t$ with $R$ a fixed rotation in the $(\theta,\dot\theta)$ plane, and set $\hat f^{B} = R \hat f^{A} R^{-1}$, $\hat g^{B} = \hat g^{A} R^{-1}$. Then for every trajectory,

$$p_{\theta_B}(o_{1:T}\mid a_{1:T}) = p_{\theta_A}(o_{1:T}\mid a_{1:T})$$

exactly. Both are global optima. The loss cannot prefer either.

Now the numbers. Per-coordinate MCC against ground truth for seed A might be $(0.94, 0.88, 0.91, 0.83)$, mean $0.89$. Under a $45^\circ$ rotation mixing $\theta$ and $\dot\theta$, seed B's Hungarian-matched MCC on those two coordinates drops to roughly $|\cos 45^\circ|\cdot 0.91 \approx 0.64$ and $0.59$, mean over four coordinates $\approx 0.76$. **A 0.13 MCC gap at identical loss and identical return.** Cross-seed dispersion $D \approx 0.24$.

Two things this makes visible. First, a paper reporting only "MCC = 0.89, seed A" is reporting a draw from a distribution whose spread is larger than most claimed improvements in the disentanglement literature. Second — and this is the trap — a *nonlinear* probe recovers $z$ from $\hat z^{B}$ with $R^2 \approx 0.99$, because $R$ is invertible and the information is fully present. So the probe says "perfect world model" while MCC says "0.76". They are measuring different things: **content** versus **canonical form**. Only the second is identifiability, and only the second is what breaks when you compose the latent with a downstream module, transfer it, or try to read a causal graph off it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*