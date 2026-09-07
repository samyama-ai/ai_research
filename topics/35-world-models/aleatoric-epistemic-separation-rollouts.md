---
id: 35-world-models/aleatoric-epistemic-separation-rollouts
title: "Distinguishing Aleatoric From Epistemic Error in Rollouts"
topic: 35-world-models
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distinguishing Aleatoric From Epistemic Error in Rollouts

> **Topic:** World Models & Planning · **ID:** `35-world-models/aleatoric-epistemic-separation-rollouts` · **Status:** methodologically-blocked

## 1. Problem Statement

A learned world model rolls out a trajectory and it diverges from reality. Two causes, with opposite remedies:

- **Aleatoric** — the environment is genuinely stochastic at that state. Collecting more data does not shrink the divergence. The planner should hedge or avoid.
- **Epistemic** — the model has not seen enough data near that state. More data shrinks it. The planner should go there (exploration) or refuse to trust the rollout (offline RL).

**Problem.** Given a model $p_\theta$, a state-action $(s,a)$ reachable at rollout step $t$, and a dataset $\mathcal{D}$, output a split of the model's predictive error into a reducible and an irreducible part, such that the reducible part actually predicts the error reduction obtained from more data.

Three variants, different difficulty:

- **Measurement.** Define a ground-truth split that is estimable from an environment with resets. Hard because the "irreducible" part is defined relative to a state representation and a hypothesis class, both of which are choices.
- **Method.** Produce an estimator (ensemble disagreement, mutual information, evidential head) that tracks the ground truth. Currently these are validated against downstream reward, not against the split.
- **Theory.** Prove conditions under which the split is identifiable from a single dataset, without resets. Suspected to be impossible in general.

Solved would mean: an estimator $\hat{u}_{\text{ep}}(s,a)$ whose rank correlation with the *measured* error reduction from $n \to 10n$ samples exceeds a stated threshold on a benchmark with heteroscedastic noise, ablated against a disagreement baseline.

## 2. Formal Setting

Environment: POMDP with transition kernel $P(s'\mid s,a)$, observation map $o=g(s)$. Model $p_\theta(o_{t+1}\mid o_{\leq t},a_{\leq t})$, parameters $\theta$, posterior (or ensemble empirical measure) $q(\theta\mid\mathcal{D})$ with $|\mathcal{D}|=n$ transitions.

**Standard information-theoretic split** at a query point, for predictive variable $Y = o_{t+1}$:

$$\underbrace{\mathbb{H}\big[\mathbb{E}_{q}\,p_\theta(Y)\big]}_{\text{total}} \;=\; \underbrace{\mathbb{E}_{q}\,\mathbb{H}\big[p_\theta(Y)\big]}_{\text{aleatoric}} \;+\; \underbrace{I(Y;\theta\mid \mathcal{D})}_{\text{epistemic}}$$

Measured as: sample $K$ ensemble members, estimate each term by Monte Carlo. $K=5$–$10$ is standard; the mutual information estimator is biased downward by $O(1/K)$.

**Rollout version.** The quantity that matters for planning is over a whole trajectory $\tau_{1:H}$ under policy $\pi$, where model error compounds:

$$I(\tau_{1:H};\theta) \;=\; \sum_{t=1}^{H} \mathbb{E}\big[\,I(o_{t}; \theta \mid o_{<t})\,\big]$$

This chain rule holds only if all members are rolled out on their *own* sampled trajectories (an "epistemic index" held fixed across the rollout, in the sense of Osband et al.'s epistemic neural networks). Teacher-forcing each member on a shared trajectory — the common implementation — measures per-step marginals and does **not** sum to the joint. This is a routinely violated assumption.

**Operational ground truth** (requires a resettable simulator):

$$u_{\text{al}}(s,a) \;=\; \lim_{n\to\infty}\ \mathbb{E}\big[\ell\big(p_{\hat\theta_n}(\cdot\mid s,a),\, P(\cdot\mid s,a)\big)\big], \qquad u_{\text{ep}}^{(n)}(s,a) \;=\; \mathbb{E}\big[\ell_n\big] - u_{\text{al}}(s,a)$$

with $\ell$ a proper scoring rule (log loss or CRPS). $P(\cdot\mid s,a)$ is itself estimated by $m$ resets from the same state; the estimate has $O(1/\sqrt{m})$ error that is indistinguishable from model error at small $m$.

**Assumptions known to be violated:**

1. $q(\theta\mid\mathcal{D})$ is a Bayesian posterior. It is not — deep ensembles are a heuristic; the decomposition is prior- and approximation-dependent.
2. State is Markov and fully observed. Under partial observability, unmodelled hidden state appears as aleatoric noise but is reducible with a better encoder — the split is representation-relative.
3. $\lim_{n\to\infty}$ is reachable. It is not; $u_{\text{al}}$ is extrapolated from a scaling curve.
4. The hypothesis class contains $P$. Misspecification loads model bias onto whichever bucket the estimator happens to favour.

## 3. State of the Art

**Established.**
- *Ensemble/posterior decompositions.* Kendall & Gal (NeurIPS 2017) split heteroscedastic head variance from parameter variance; Depeweg et al. (ICML 2018) do the same for latent-variable dynamics models and use it for risk-sensitive RL. Established as a working *pipeline*; not established that the two terms match the operational definition in §2.
- *Deep ensembles beat single-model uncertainty under shift.* Ovadia et al. (NeurIPS 2019) — reproduced widely.
- *Disagreement drives exploration.* Plan2Explore (Sekar et al., ICML 2020) uses one-step ensemble disagreement in latent space as an intrinsic reward and reaches near-SOTA zero-shot DM Control performance. Established as a control result.

**Claimed but unablated.**
- That disagreement in Plan2Explore / Pathak et al. (ICML 2019) *is* epistemic uncertainty. The papers show it improves exploration; none measure whether disagreement predicts error reduction. Pathak et al. explicitly note disagreement is designed to vanish under pure aleatoric noise in expectation, but do not measure the residual at finite $K$.
- Model-based offline RL uncertainty penalties. MOPO (Yu et al., NeurIPS 2020) uses max ensemble std as a penalty; Lu et al. (ICLR 2022, "Revisiting Design Choices in Offline Model-Based RL") show the penalty's *ranking quality* is poor and that swapping estimators changes returns more than the theory predicts. Benchmark numbers only.
- Evidential regression (Amini et al., NeurIPS 2020) as a single-pass split. Meinert et al. (AAAI 2023) show the evidential parameters are non-identifiable from the loss — the reported split is an artifact of regularisation strength.

**Theory SOTA.** Bengs, Hüllermeier & Waegeman (NeurIPS 2022) show empirical loss minimisation gives second-order (uncertainty-predicting) learners no incentive to report faithful epistemic uncertainty; their follow-up (ICML 2023) extends this to second-order scoring rules. Wimmer et al. (UAI 2023) show entropy/mutual-information measures violate basic axioms one would want of the split.

## 4. What Is Known

- **Numbers, ImageNet-C scale (Ovadia 2019):** ensembles of 10 keep expected calibration error near 0.05–0.10 at high corruption where a single softmax model exceeds 0.25. Establishes ensembles detect *shift*; says nothing about the aleatoric/epistemic ratio.
- **Noisy-TV, Atari scale (Burda et al., ICLR 2019):** prediction-error curiosity is captured by a stochastic TV; RND avoids it. Confirms that raw prediction error confounds the two sources, at 50M-frame scale.
- **Disagreement collapses under noise (Pathak et al., ICML 2019):** in a noisy-action 3D navigation task, disagreement-based reward remains flat under injected noise where prediction-error reward spikes. Measured on ~5 tasks, single seed set.
- **Joint prediction, Neural Testbed scale (Osband et al., NeurIPS 2022/2023):** on synthetic 2-layer MLP generative models, methods that are near-identical in *marginal* log-loss differ by more than an order of magnitude in $\tau$-step *joint* log-loss. Directly relevant: rollout uncertainty is joint, and marginal metrics do not rank methods correctly.
- **Non-identifiability of evidential heads:** shown analytically and on 1-D regression (Meinert et al., AAAI 2023).
- **Offline RL, D4RL scale (Lu et al., ICLR 2022):** among uncertainty penalties tested, no estimator dominates; return spreads of 10–30 points on the same dataset from penalty choice alone.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted operational definition of the split for a *rollout*. The per-step information decomposition is well defined only under an exact posterior; the horizon-$H$ version requires index-consistent sampling that most codebases do not implement. No benchmark reports the split's ground truth.
- **Theoretically open.** Whether $u_{\text{al}}$ and $u_{\text{ep}}$ are identifiable from a single fixed dataset without resets, under model misspecification. Bengs et al. is a strong negative result about *incentives*, not a full impossibility theorem about identifiability.
- **Empirically open.** Whether ensemble disagreement rank-correlates with measured error reduction in a resettable heteroscedastic environment. The experiment is cheap (§8) and, as far as this catalog can determine, unrun as a controlled study.
- **Open.** Whether the split is even the right decision variable, versus directly predicting "will more data here reduce my planning regret" (the DEUP framing, Lahlou et al., TMLR 2023).

## 6. Why It Is Hard

**Absent ground truth plus non-identifiability, compounding.**

1. The target quantity is defined by a limit ($n\to\infty$) nobody reaches, so ground truth must be extrapolated from a data-scaling curve — and the curve's asymptote is fit with the same error bars as the thing being measured.
2. Aleatoric is representation-relative. Unobserved state is irreducible for a given encoder and reducible for a better one. So the "irreducible" bucket is not a property of the environment; it is a property of (environment, representation, hypothesis class). Two papers reporting different splits may both be right.
3. Misspecification has no bucket. Model bias is neither noise nor ignorance; every estimator silently assigns it somewhere.
4. The evaluation does not measure what it names. Downstream return is the standard validation for uncertainty estimators. A penalty that is a *bad* uncertainty estimate but a good pessimism regulariser scores well — which is exactly what Lu et al. (ICLR 2022) observe.

## 7. Current Research (as of 2026)

- **Epistemic neural networks and joint prediction** (Osband and collaborators, DeepMind lineage) — index-consistent epistemic sampling, evaluated by $\tau$-step joint log-loss rather than marginals. The most directly applicable technical fix to the rollout version. *(frontier — verify current status)*
- **Axiomatic critiques of the standard decomposition** (Hüllermeier, Waegeman, Bengs, and coauthors — Munich/Ghent) — continuing to publish impossibility-flavoured results about second-order learners.
- **Uncertainty-aware latent world models** — Dreamer-family models carry a stochastic latent that absorbs aleatoric noise by construction; whether the KL balancing coefficient controls where the split falls is untested. *(frontier — verify)*
- **Conformal and distribution-free rollout bounds** — replacing the split with a calibrated coverage guarantee over trajectories, sidestepping the decomposition entirely. *(frontier — verify)*
- **DEUP-style direct regression of excess risk** (Mila) — predict reducible error directly instead of decomposing.

## 8. Concrete Next Experiment

**Scale.** Two environments with a *known, tunable* noise floor: (a) a gridworld/continuous control task where the transition kernel adds Gaussian noise with state-dependent scale $\sigma(s)$ set by the experimenter; (b) DM Control `cheetah-run` with injected action noise on a spatial subregion. Model: 7-member probabilistic ensemble (PETS architecture), ~1M parameters. Data budgets $n \in \{10^4, 10^5, 10^6\}$ transitions, 5 seeds each. Total cost: order 100 GPU-hours.

**Procedure.** Hold out a grid of 500 query states. For each: (i) measure true $P(\cdot\mid s,a)$ by $m=1000$ resets; (ii) compute measured error reduction $\Delta_i = \ell_{10^5}(s_i) - \ell_{10^6}(s_i)$ under CRPS; (iii) compute each candidate estimator at $n=10^5$: ensemble disagreement, mutual information with index-consistent sampling, evidential head, RND count proxy.

**Control arm.** A deliberately mis-assigning baseline: total predictive variance, which makes no split at all. Any estimator that fails to beat it is measuring shift, not epistemics.

**Deciding number.** Spearman $\rho$ between each estimator and $\Delta_i$ across the 500 query states, in the *high-$\sigma(s)$ subgroup only*. If no estimator reaches $\rho \geq 0.5$ there while total variance sits near $\rho \approx 0$, the field's epistemic estimators are confirmed to be noise-detectors, and the status of this problem is settled as blocked-on-method rather than blocked-on-measurement.

## 9. Key References

- **[Foundational]** A. Der Kiureghian, O. Ditlevsen. *Aleatory or epistemic? Does it matter?* Structural Safety 31(2), 2009.
- **[Foundational]** A. Kendall, Y. Gal. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS, 2017. — arXiv:1703.04977
- **[Foundational]** B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS, 2017. — arXiv:1612.01474
- **[Foundational]** S. Depeweg, J. M. Hernández-Lobato, F. Doshi-Velez, S. Udluft. *Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning.* ICML, 2018. — arXiv:1710.07283
- **[SOTA]** D. Pathak, D. Gandhi, A. Gupta. *Self-Supervised Exploration via Disagreement.* ICML, 2019. — arXiv:1906.04161
- **[SOTA]** R. Sekar, O. Rybkin, K. Daniilidis, P. Abbeel, D. Hafner, D. Pathak. *Planning to Explore via Self-Supervised World Models.* ICML, 2020. — arXiv:2005.05960
- **[SOTA]** I. Osband, Z. Wen, S. M. Asghari, V. Dwaracherla, M. Ibrahimi, X. Lu, B. Van Roy. *Epistemic Neural Networks.* NeurIPS, 2023. — arXiv:2107.08924
- **[SOTA]** I. Osband et al. *The Neural Testbed: Evaluating Joint Predictions.* NeurIPS Datasets & Benchmarks, 2022.
- **[Theory]** V. Bengs, E. Hüllermeier, W. Waegeman. *Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation.* NeurIPS, 2022.
- **[Theory]** L. Wimmer, Y. Sale, P. Hofman, B. Bischl, E. Hüllermeier. *Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures?* UAI, 2023.
- **[Theory]** N. Meinert, J. Gawlikowski, A. Lavin. *The Unreasonable Effectiveness of Deep Evidential Regression.* AAAI, 2023.
- **[Empirical]** Y. Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Empirical]** C. Lu, P. J. Ball, J. Parker-Holder, M. A. Osborne, S. J. Roberts. *Revisiting Design Choices in Offline Model-Based Reinforcement Learning.* ICLR, 2022.
- **[Empirical]** T. Yu, G. Thomas, L. Yu, S. Ermon, J. Zou, S. Levine, C. Finn, T. Ma. *MOPO: Model-based Offline Policy Optimization.* NeurIPS, 2020. — arXiv:2005.13239
- **[Alternative framing]** S. Lahlou, M. Jain, H. Nekoei, V. Butoi, P. Bertin, J. Rector-Brooks, M. Korablyov, Y. Bengio. *DEUP: Direct Epistemic Uncertainty Prediction.* TMLR, 2023. — arXiv:2102.08501
- **[Survey]** E. Hüllermeier, W. Waegeman. *Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods.* Machine Learning 110, 2021. — arXiv:1910.09457

## 10. Worked Example

A 1-D contextual dynamics model. State $s \in [0,1]$, action fixed, next state $s' = f(s) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma(s)^2)$ with

$$\sigma(s) = 0.02 \ \text{for } s < 0.5, \qquad \sigma(s) = 0.30 \ \text{for } s \geq 0.5.$$

Training data: 2000 points, but sampled non-uniformly — 1900 from $s<0.5$, 100 from $s\geq 0.5$. So the right half is *both* noisy and data-poor. Fit a 7-member ensemble of Gaussian-output MLPs.

Typical measured outcome:

| region | true $\sigma$ | mean predicted $\sigma$ | ensemble disagreement (std of means) |
|---|---|---|---|
| $s<0.5$ | 0.02 | 0.021 | 0.004 |
| $s\ge0.5$ | 0.30 | 0.26 | 0.071 |

Disagreement is $18\times$ higher on the right. A planner reading it as epistemic goes there to explore.

Now run the deciding measurement. Retrain with $10\times$ data in the right half only (100 → 1000 points). Predicted $\sigma$ moves 0.26 → 0.295; measured CRPS on the right improves by roughly 3%. Disagreement drops 0.071 → 0.026. So the disagreement signal was **mostly not** reducible error: $18\times$ disagreement bought a 3% error reduction, while the same disagreement magnitude in the left half at low data would have bought far more.

The obstruction is visible in one line: disagreement's response to the noise floor $\sigma(s)$ is $O(\sigma/\sqrt{K})$ — with $K=7$, that is $0.30/2.65 \approx 0.11$, the same order as the entire signal being read as epistemic. The estimator does not fail to converge; it converges to a value that mixes the two sources with a coefficient set by the ensemble size, and no amount of downstream return improvement reveals this.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*