---
id: 35-world-models/learning-reward-functions-inside-world-models
title: "Learning Reward Functions Inside World Models"
topic: 35-world-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning Reward Functions Inside World Models

> **Topic:** World Models & Planning · **ID:** `35-world-models/learning-reward-functions-inside-world-models` · **Status:** open

## 1. Problem Statement

A latent world model predicts future latent states from actions. To plan or to train a policy *in imagination*, it also needs a **reward head**: a learned function that scores states the environment never labelled, because those states exist only inside the model. The problem is that this head is trained on on-policy environment data and then queried far off that distribution by an optimizer whose explicit job is to find its maximum.

Three variants, routinely conflated:

- **Measurement.** Given a trained world model, quantify how much of the policy's imagined return is real. Currently there is no accepted metric that separates "the dynamics head hallucinated a good state" from "the reward head mislabelled a real state."
- **Method.** Train a reward head $\hat r_\phi$ whose induced optimal policy in the model is near-optimal in the environment, under a fixed interaction budget and without extra reward labels off-distribution.
- **Theory.** Characterize when the reward function is *identifiable* from the data the world model was trained on, up to transformations that preserve the optimal policy under the *truncated, bootstrapped* imagination objective actually used — not under the infinite-horizon objective for which the classical invariance theorems hold.

Solving it means: a stated procedure with a bound, or a reproducible empirical law, relating reward-head error off-distribution to real-environment return loss.

## 2. Formal Setting

MDP $\mathcal{M}=(\mathcal S,\mathcal A,T,r,\gamma)$, observations $o_t$. The world model is $(e_\theta, T_\theta, \hat r_\phi, \hat v_\psi)$: encoder $z_t = e_\theta(o_{\le t}, a_{<t})$, latent transition $\hat T_\theta(z_{t+1}\mid z_t,a_t)$, reward head $\hat r_\phi(z_t,a_t)\in\mathbb R$, bootstrap value $\hat v_\psi$.

**Reward head loss, as measured.** In DreamerV3 the target is a two-hot encoding of $\mathrm{symlog}(r)$ over $B=255$ bins and the loss is cross-entropy; in MuZero it is cross-entropy over a 601-bin categorical support under $h(x)=\mathrm{sign}(x)(\sqrt{|x|+1}-1)+\epsilon x$. Both are reported as a single scalar averaged over the replay buffer:
$$\mathcal L_r(\phi)=\mathbb E_{(z,a,r)\sim \mathcal D_{\text{replay}}}\big[\mathrm{CE}\big(\hat r_\phi(z,a),\, \mathrm{twohot}(g(r))\big)\big].$$
This expectation is over the *replay distribution*, which is the object of interest's complement.

**Imagined return.** From $z_0\sim\mathcal D$, roll $H$ steps under $\pi$ and $\hat T_\theta$:
$$\hat J_H(\pi)=\mathbb E\Big[\sum_{t=0}^{H-1}\gamma^t \hat r_\phi(z_t,a_t)+\gamma^H \hat v_\psi(z_H)\Big],\quad H=15\ \text{(DreamerV3)},\ H=5\ \text{(MuZero unroll)}.$$

**The quantity that matters** is the *deployment gap* $\Delta = J(\pi^\star_{\mathcal M}) - J(\hat\pi)$ where $\hat\pi=\arg\max \hat J_H$ and $J$ is true environment return. Measured by running $\hat\pi$ in the real environment for $N$ episodes; the standard error is $\mathrm{sd}(G)/\sqrt N$ and is rarely reported per-seed.

**Off-distribution reward error**, the causal driver, would be measured as
$$\mathcal E(\pi)=\mathbb E_{(z,a)\sim d^{\hat T}_\pi}\big[|\hat r_\phi(z,a)-r(\mathrm{dec}(z),a)|\big],$$
which requires decoding an imagined latent to a real state and querying the true reward. This is only possible in simulators with a programmatic reward function (DMC, Atari via emulator state, Crafter, MuJoCo). **It is not measurable at all** for preference-learned rewards or real robots — that is the methodological block.

**Assumptions and their violations.**
- *Markov reward in latent space*: $r$ is a function of $(z,a)$. Violated whenever the encoder discards reward-relevant detail; Abel et al. (NeurIPS 2021) show some tasks admit no Markov reward at all in a given state space.
- *Replay coverage*: $d^{\hat T}_{\hat\pi} \ll \mathcal D_{\text{replay}}$. Violated by construction — imagination optimization moves mass to where $\hat r_\phi$ is largest, which is where it is least supervised.
- *Potential-shaping invariance*: $r' = r+\gamma\Phi(s')-\Phi(s)$ preserves the optimal policy (Ng, Harada & Russell, ICML 1999). Violated under truncated imagination: the telescoping sum leaves a residual $\gamma^H\Phi(z_H)-\Phi(z_0)$ that is *not* absorbed unless $\hat v_\psi$ is exactly consistent with $\hat r_\phi$.

## 3. State of the Art

**Established (ablated, reproduced).**
- Joint end-to-end training of reward head with dynamics is the default and works at scale: MuZero (Schrittwieser et al., *Nature* 2020) and DreamerV3 (Hafner et al., arXiv:2301.04104; *Nature* 2025). DreamerV3's ablations show the symlog/two-hot reward parameterization is load-bearing for the fixed-hyperparameter claim across domains.
- Value-equivalence: Grimm et al. (NeurIPS 2020) prove a model need only preserve Bellman backups on a chosen policy/value set, so the reward head need not be pointwise accurate. This is a theorem, and it is the reason "reward MSE" is a poor target.
- Reward-free exploration is provably sufficient in tabular/linear settings: Jin et al. (ICML 2020) give $\tilde O(S^2A/\epsilon^2)$ sample complexity to collect a dataset from which *any* later-specified reward yields an $\epsilon$-optimal policy.

**Claimed but unablated.**
- That video-prediction likelihood is a usable reward (VIPER, Escontrela et al., NeurIPS 2023) — strong on 28 DMC/Atari/RLBench tasks, but no ablation isolates reward-head off-distribution error from dynamics error.
- That scaling the world model shrinks the reward-gaming gap. TD-MPC2 (Hansen et al., ICLR 2024) scales to 317M parameters over 104 tasks and reports monotone gains; the reward head's contribution is not separated.

**Benchmark-number-only.** Atari-100k means (IRIS 1.046 mean HNS, ICLR 2023; DIAMOND 1.46 mean HNS, NeurIPS 2024) are aggregate returns. Neither paper reports imagined-vs-real reward divergence.

## 4. What Is Known

- **Overoptimization follows a law in the LLM setting.** Gao, Schulman & Hilton (ICML 2023): gold reward vs. proxy-reward KL follows $d(\alpha-\beta d)$ with $d=\sqrt{\mathrm{KL}}$; the coefficients shift with reward-model size (3M–3B parameters) but the functional form is stable. This is the closest thing to a quantitative overoptimization law, and it was measured with a *fixed* dynamics (language decoding), not a learned one.
- **Reward is not identifiable from behaviour.** Ng & Russell (ICML 2000): degenerate solutions ($r\equiv 0$) explain any policy. Skalse et al. (ICML 2023) characterize exactly which reward-function ambiguities each data source leaves, and show the ambiguity class is generally larger than potential shaping.
- **Sparsity is extreme at the scale these models train at.** Under Atari-100k (100k environment steps, ~400k frames), hard-exploration games give a nonzero-reward base rate below $10^{-3}$ per transition.
- **Imagination horizon is short and tuned.** $H=15$ for DreamerV3, unroll $K=5$ for MuZero — chosen empirically because longer horizons degrade, which is itself evidence the reward+value composite is unreliable off-distribution.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the reward-ambiguity class under the *truncated bootstrapped* objective $\hat J_H$. Potential shaping is known to be safe at $H=\infty$; nothing states the invariance class at finite $H$ with a learned $\hat v_\psi$. No regret bound of the form $\Delta \le f(\mathcal E, H, \gamma, \text{coverage})$ exists for deep latent models.
- **Empirically open.** Whether an overoptimization law of Gao et al.'s form holds when *dynamics are also learned*. The experiment is runnable today on Crafter or DMC with ground-truth reward queryable — nobody has published the curve. Also open: whether reward-head capacity or dynamics capacity dominates $\Delta$ at fixed total parameters.
- **Methodologically blocked.** Measuring $\mathcal E(\pi)$ requires decoding imagined latents to states with a queryable true reward. For preference-based or VLM-based rewards, no ground truth exists at any price, so "reward hallucination" is currently undefined for exactly the systems where it matters most.

## 6. Why It Is Hard

**Non-identifiability compounded by adversarial query distribution.** The reward head is fit on $\mathcal D_{\text{replay}}$, where reward is near-constant zero, and then evaluated at $\arg\max$ over latents chosen by an optimizer. Two failure modes are statistically indistinguishable in the training loss: (a) the head is right and the dynamics invented a rewarding state; (b) the dynamics are right and the head mislabelled. Both show as high $\hat J_H$ and low $J$. Ablating them apart requires a decoder-to-true-reward oracle that exists only in simulators — **absent ground truth**, not compute cost, is the binding constraint. Second, the aggregate scalar loss is dominated by the zero-reward majority class, so the metric reported in every paper is nearly insensitive to the errors that determine $\Delta$: **an evaluation that does not measure what it names.**

## 7. Current Research (as of 2026)

- **Reward-free / decoupled world models**: pretrain dynamics on unlabelled video, attach reward later. Genie-style action-free video models and LAPO-style latent-action inference push this direction *(frontier — verify current reward-attachment results)*.
- **VLM- and video-likelihood rewards** inside imagination (VIPER lineage; Berkeley/DeepMind). Attractive because they generalize off-distribution; unfalsifiable at present because no ground truth exists to score them.
- **Ensemble/uncertainty penalties on the reward head** (pessimism transplanted from offline RL). Reported to reduce imagination exploitation; ablations separating reward-head from dynamics uncertainty are thin.
- **Reward-model interpretability and STARC-style reward distances** (Skalse et al., ICLR 2024) — provides a policy-order-preserving metric on reward functions that could replace MSE as the reported number. Not yet adopted by world-model papers *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Crafter (22 achievements, programmatic reward) plus 6 DMC tasks. DreamerV3 XS/S/M (~8M/18M/37M parameters), 1M environment steps, 5 seeds. Total ~90 runs, roughly 300 GPU-hours on A100-class hardware — small enough for one lab.

**Instrument.** Crafter and DMC both expose the true reward function. Decode each imagined latent $z_t$ with the model's own decoder, re-encode into the simulator's state where possible (Crafter: symbolic grid; DMC: qpos/qvel head trained supervised), query true $r$. Record $\mathcal E(\pi)$ and $\mathrm{KL}(d_{\hat\pi}\,\|\,d_{\text{replay}})$ every 50k steps.

**Control arm.** Identical model, identical seeds, but the reward head replaced by the *oracle* environment reward evaluated on the decoded imagined state. Everything else — dynamics, value, actor, horizon — held fixed. The difference in final return isolates the reward head's contribution to $\Delta$; a second arm with oracle *dynamics* and learned reward isolates the other direction.

**The deciding number.** The fraction of the deployment gap attributable to the reward head:
$$\rho = \frac{J(\pi_{\text{oracle-}r}) - J(\pi_{\text{learned}})}{J(\pi_{\text{oracle-}r,\text{oracle-}T}) - J(\pi_{\text{learned}})}.$$
If $\rho > 0.5$ at 1M steps with non-overlapping 95% CIs across 5 seeds, reward learning — not dynamics — is the bottleneck in imagination-based RL, and the field's parameter budget is misallocated. If $\rho < 0.2$, the reward head is a solved subproblem and this entry should be downgraded.

**Secondary readout.** Fit $J(\hat\pi) = J_0 + d(\alpha-\beta d)$, $d=\sqrt{\mathrm{KL}(d_{\hat\pi}\|d_{\text{replay}})}$. Whether $\beta$ scales with reward-head size the way Gao et al. found for LLM reward models is the first test of whether that law survives learned dynamics.

## 9. Key References

- **[Foundational]** Ng, A. Y., Harada, D., Russell, S. *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML, 1999.
- **[Foundational]** Ng, A. Y., Russell, S. *Algorithms for Inverse Reinforcement Learning.* ICML, 2000.
- **[Foundational]** Oh, J., Singh, S., Lee, H. *Value Prediction Networks.* NeurIPS, 2017. — arXiv:1707.03497
- **[Foundational]** Schrittwieser, J. et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature 588, 2020. — arXiv:1911.08265
- **[Theory]** Grimm, C., Barreto, A., Singh, S., Silver, D. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* NeurIPS, 2020. — arXiv:2011.03506
- **[Theory]** Jin, C., Krishnamurthy, A., Simchowitz, M., Yu, T. *Reward-Free Exploration for Reinforcement Learning.* ICML, 2020. — arXiv:2002.02794
- **[Theory]** Abel, D., Dabney, W., Harutyunyan, A., Ho, M. K., Littman, M. L., Precup, D., Singh, S. *On the Expressivity of Markov Reward.* NeurIPS, 2021 (outstanding paper). — arXiv:2111.00876
- **[Theory]** Cao, H., Cohen, S., Szpruch, L. *Identifiability in Inverse Reinforcement Learning.* NeurIPS, 2021.
- **[Theory]** Skalse, J., Farrugia-Roberts, M., Russell, S., Abate, A., Gleave, A. *Invariance in Policy Optimisation and Partial Identifiability in Reward Learning.* ICML, 2023. — arXiv:2203.07475
- **[Theory]** Skalse, J. et al. *STARC: A General Framework For Quantifying Differences Between Reward Functions.* ICLR, 2024.
- **[SOTA]** Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. *Mastering Diverse Domains through World Models* (DreamerV3). arXiv:2301.04104, 2023; published in Nature, 2025.
- **[SOTA]** Hansen, N., Su, H., Wang, X. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR, 2024. — arXiv:2310.16828
- **[SOTA]** Micheli, V., Alonso, E., Fleuret, F. *Transformers are Sample-Efficient World Models* (IRIS). ICLR, 2023. — arXiv:2209.00588
- **[SOTA]** Alonso, E. et al. *Diffusion for World Modeling: Visual Details Matter in Atari* (DIAMOND). NeurIPS, 2024. — arXiv:2405.12399
- **[Empirical]** Gao, L., Schulman, J., Hilton, J. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[Empirical]** Escontrela, A. et al. *Video Prediction Models as Rewards for Reinforcement Learning* (VIPER). NeurIPS, 2023. — arXiv:2305.14343
- **[Empirical]** Christiano, P. et al. *Deep Reinforcement Learning from Human Preferences.* NeurIPS, 2017. — arXiv:1706.03741
- **[Survey]** Moerland, T., Broekens, J., Plaat, A., Jonker, C. *Model-based Reinforcement Learning: A Survey.* Foundations and Trends in Machine Learning, 2023.

## 10. Worked Example

**Setting.** A sparse Atari-like task at the Atari-100k budget: 100k transitions in replay, reward $\in\{0,1\}$, base rate $p = 10^{-3}$ (100 positive transitions in the whole buffer). DreamerV3 imagination horizon $H=15$, $\gamma=0.997$ so $\gamma^t\approx 1$ over the horizon.

**A well-fit reward head.** Suppose $\hat r_\phi$ achieves recall $0.5$ on true-positive transitions and a false-positive rate of $10^{-3}$ on true-negatives — plausible with 100 positive examples. Classification accuracy is
$$1 - \big(p\cdot 0.5 + (1-p)\cdot 10^{-3}\big) = 1 - (5\times10^{-4} + 10^{-3}) \approx 99.85\%.$$
Reward MSE against a constant-zero baseline: baseline MSE $=10^{-3}$; the head's MSE $\approx 5\times10^{-4}+10^{-3}=1.5\times10^{-3}$. **The learned head is worse than predicting zero everywhere on the reported metric**, yet it is the only head that can drive learning.

**What the actor optimizes.** Over one 15-step imagined rollout, expected *true* reward captured is $15\times10^{-3}\times0.5=7.5\times10^{-3}$; expected *spurious* reward is $15\times(1-10^{-3})\times10^{-3}\approx1.5\times10^{-2}$. The false-positive mass is **2× the true signal** before the actor has done anything.

**Then the actor does something.** Gradient ascent on $\hat J_{15}$ shifts the imagined state distribution toward the head's false positives. Even a mild shift raising the false-positive rate on-visited-states from $10^{-3}$ to $10^{-2}$ — a 1% error rate, unremarkable for a network queried off-distribution — gives imagined return $\hat J_{15}\approx 0.15$ against a true return of $7.5\times10^{-3}$: a **20× overestimate**.

**The obstruction, visible.** Every number a practitioner sees moves the right way: cross-entropy loss falls, imagined return rises, dynamics reconstruction error falls. Nothing in the reported instrumentation distinguishes "the world model imagined a real rewarding state" from "the reward head fired on a latent that decodes to nothing." Separating them needs $\mathcal E(\pi)$ from §2, which needs a decoder-to-true-reward oracle. In Crafter that oracle exists; on a robot, or with a preference-trained reward, it does not — and that is why the problem is open rather than merely unmeasured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*