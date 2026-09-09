---
id: 35-world-models/latent-collapse-self-predictive-objectives
title: "Latent Space Collapse Under Self-Predictive Objectives"
topic: 35-world-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latent Space Collapse Under Self-Predictive Objectives

> **Topic:** World Models & Planning · **ID:** `35-world-models/latent-collapse-self-predictive-objectives` · **Status:** partially-solved

## 1. Problem Statement

A self-predictive world model learns an encoder $\phi$ and a latent dynamics model $f$ by predicting its *own* future embeddings — no pixel reconstruction, no negatives. The objective $\|f(\phi(o_t),a_t) - \mathrm{sg}[\phi_{\text{targ}}(o_{t+1})]\|^2$ has a global minimum at $\phi \equiv c$: a constant encoder makes the prediction exact and the loss zero. **Collapse** is convergence toward that degenerate family — fully (rank 0), or *dimensionally* (rank $\ll$ latent width, the practically common case).

Three variants, routinely conflated:

- **Measurement.** Define a statistic $C(\phi)$ on a trained model that separates "collapsed" from "correctly compressed". Not settled: discarding dimensions is the encoder's *job*.
- **Method.** Find the minimal set of stabilisers (stop-gradient, EMA target, predictor head, variance floor, reward/auxiliary loss) that provably avoids the constant solution while keeping control performance.
- **Theory.** Prove, for a stated architecture and optimiser, that the collapsed manifold is not reached from generic init — or that it is.

Solved: fully-collapsed configurations are diagnosable and avoidable in practice. Open: whether the *dimensional* collapse routinely observed in trained agents is a cause of control failure or a benign consequence of low task intrinsic dimension.

## 2. Formal Setting

MDP $(\mathcal{O},\mathcal{A},P,r,\gamma)$; encoder $\phi_\theta:\mathcal{O}\to\mathbb{R}^d$; latent dynamics $f_\psi:\mathbb{R}^d\times\mathcal{A}\to\mathbb{R}^d$; target encoder $\phi_{\bar\theta}$ with EMA rate $\tau$: $\bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta$. Loss over a replay buffer $\mathcal{D}$ and horizon $H$:

$$\mathcal{L}(\theta,\psi) = \mathbb{E}_{\mathcal{D}}\sum_{k=1}^{H}\big\| g_\psi^{(k)}\!\left(z_t, a_{t:t+k-1}\right) - \mathrm{sg}\big[\phi_{\bar\theta}(o_{t+k})\big] \big\|_2^2, \quad z_t=\phi_\theta(o_t).$$

**Measured quantities.** Draw $N=10^4$ observations from the current policy's replay distribution. Form $Z\in\mathbb{R}^{N\times d}$, centre it, and let $\sigma_1\ge\cdots\ge\sigma_d$ be singular values, $p_i=\sigma_i/\sum_j\sigma_j$.

- **Effective rank** (Roy & Vetterli, 2007): $\mathrm{erank}(Z)=\exp\!\big(-\sum_i p_i\log p_i\big)$. Continuous, scale-invariant, in $[1,d]$.
- **srank$_\delta$** (Kumar et al., ICLR 2021): smallest $k$ with $\sum_{i\le k}\sigma_i^2 \ge (1-\delta)\sum_i\sigma_i^2$, $\delta=0.01$.
- **Feature std**, the SimSiam collapse probe: $\frac{1}{d}\sum_i \mathrm{std}_n(z_{ni}/\|z_n\|_2)$; equals $1/\sqrt{d}$ for isotropic embeddings, $0$ at full collapse.
- **Behavioural sufficiency**: bisimulation-style residual $\Delta = \mathbb{E}|r(o,a)-\hat r(\phi(o),a)| + W_1$-error of the next-latent distribution. This is the quantity collapse is *supposed* to damage; it is almost never reported.

**Assumptions, with the violated ones flagged.** (i) The replay distribution is stationary — **violated**: it shifts with the policy, so $\mathrm{erank}$ measured at step $t$ is over a moving input distribution. (ii) The target is a slowly-moving copy — **violated** at $\tau$ near 0 and under aggressive resets. (iii) Optimisation is gradient flow — **violated**: Adam's preconditioner changes the eigenvalue dynamics that every closed-form analysis relies on. (iv) The encoder is linear — the assumption under which all the collapse theorems hold, and false everywhere.

## 3. State of the Art

**Established (ablated, independently reproduced).**
- Stop-gradient is load-bearing. SimSiam (Chen & He, CVPR 2021): removing it collapses training within epochs.
- BYOL (Grill et al., NeurIPS 2020) works without negatives; its predictor and EMA target are each necessary in the reported ablations.
- Batch-norm is *not* the hidden contrastive term. Richemond et al. (2020) reproduced BYOL to within ~0.4 points of baseline using group norm + weight standardisation, refuting the widely-circulated "BN provides implicit negatives" claim.
- Latent self-prediction improves sample-efficient RL: SPR (Schwarzer et al., ICLR 2021) on Atari-100k; TD-MPC2 (Hansen et al., ICLR 2024) across 100+ continuous-control tasks.

**Theory SOTA.** Tian, Chen & Ganguli (ICML 2021) give closed-form eigenvalue dynamics for two-layer linear BYOL and derive **DirectPred**, setting the predictor analytically from the feature correlation matrix — competitive with learned predictors on ImageNet. Tang et al. (ICML 2023) analyse self-predictive RL as a two-timescale ODE and show stop-gradient plus a slower encoder timescale makes the non-collapsed solution locally stable in the linear case. Ni et al. (ICLR 2024) unify state and history self-prediction and show reward prediction is not required for a self-predictive representation to be a sufficient statistic. Khetarpal et al. (2024) extend to action-conditional self-prediction.

**Claimed but unablated.** That JEPA-style architectures (LeCun's position paper, 2022; I-JEPA, Assran et al., CVPR 2023) avoid collapse *by design*. I-JEPA reports strong ImageNet linear-probe numbers, but the anti-collapse claim rests on the same empirical stabilisers (EMA target, asymmetric predictor); no theorem covers the deep nonlinear case. That "collapse causes" RL plateaus is asserted in many papers and demonstrated causally in almost none — the evidence is a correlation between falling rank and falling return.

**Benchmark-number-only.** Most reported "no collapse" claims are a single feature-std or erank curve on one seed of one task, not a controlled comparison.

## 4. What Is Known

- **Full collapse is trivially inducible and trivially detectable.** SimSiam ImageNet-1k, ResNet-50, 100-epoch pretrain: 67.7% linear-probe top-1 with stop-gradient; **0.1%** (chance) without. Loss reaches its minimum $-1$ within tens of epochs when it collapses.
- **BYOL scale.** 74.3% top-1 (ResNet-50, 300 epochs, ImageNet-1k); 79.6% at ResNet-200(×2). Removing the predictor or freezing the target to the online network drops performance to near-chance in the paper's ablation.
- **Dimensional collapse is real and separate from full collapse.** Jing et al. (ICLR 2022) show contrastive embeddings occupy a strict subspace of the projector output even when training looks healthy, and attribute it to implicit regularisation along low-curvature directions.
- **Rank loss correlates with RL failure.** Kumar et al. (ICLR 2021) measure srank collapsing over training in offline DQN on Atari with a matching return drop; Lyle et al. (ICLR 2022, ICML 2023) reproduce capacity/plasticity loss and show rank-preserving interventions recover part of the gap on Atari and DMC.
- **Explicit anti-collapse regularisers work without negatives.** VICReg (Bardes, Ponce & LeCun, ICLR 2022) and Barlow Twins (Zbontar et al., ICML 2021) reach BYOL-comparable ImageNet linear-probe accuracy using only variance/covariance terms — so the stop-gradient/EMA mechanism is sufficient but not necessary.

## 5. What Is Not Known

- **Theoretically open.** No collapse-avoidance theorem for a *deep nonlinear* encoder under Adam. Every existing guarantee (Tian et al. 2021; Tang et al. 2023) is two-layer linear or a timescale-separated ODE. Whether stop-gradient + EMA excludes the constant solution from generic initialisation in the nonlinear case: no proof either way.
- **Methodologically blocked.** There is no accepted definition of *pathological* rank. A walker task with 24 true state dimensions *should* yield $\mathrm{erank}\approx 24$ in a 512-d latent. Without a task-intrinsic-dimension baseline, "erank dropped" is uninterpretable, so the entire correlational literature rests on an undefined threshold.
- **Empirically open.** The causal test — restore rank without changing anything else, measure return — is runnable on a few thousand GPU-hours and has not been run as a clean multi-task, multi-seed ablation. Also open: whether horizon $H>1$ multi-step prediction resists collapse better than $H=1$, at matched gradient count.

## 6. Why It Is Hard

**Confounded measurement plus absent ground truth.** Rank is measured on a distribution the policy itself controls. A better policy visits a narrower, more structured state distribution, which *lowers* measured erank; a collapsing encoder also lowers erank. The two are entangled by construction, and the standard protocol (measure on the replay buffer) cannot separate them. There is no ground-truth "correct dimensionality" to compare against, because the minimal sufficient latent dimension for a given reward and horizon is itself unknown for every benchmark task.

Second obstruction: **non-identifiability of the mechanism.** Stop-gradient, EMA rate, predictor conditioning, weight decay, and Adam's $\epsilon$ all move the same eigenvalue dynamics. Ablating one changes the effective learning rate of the others, so single-factor ablations do not isolate the cause.

## 7. Current Research (as of 2026)

- **Analytic predictors.** DirectPred-style closed-form predictors extended to action-conditional latent dynamics; the ICML-2023/ICLR-2024 RL theory line (DeepMind, Mila, CMU) continues here.
- **Regulariser-first world models.** Replacing EMA stabilisation with VICReg-style variance floors inside model-based agents, to decouple "does not collapse" from "has a target network".
- **Discrete bottlenecks as collapse insurance.** DreamerV3 (Hafner et al., *Nature*, 2025) uses categorical latents; codebook usage is a direct, interpretable rank proxy. Whether discreteness *prevents* collapse or merely renames it as codebook死 collapse is unresolved *(frontier — verify)*.
- **Plasticity/rank interventions.** Periodic resets, layer norm, and regenerative regularisation applied to world-model encoders specifically *(frontier — verify)*.
- **V-JEPA-line video world models** (Meta AI) — anti-collapse evidence remains empirical probes, not proofs.

## 8. Concrete Next Experiment

**Question.** Is dimensional collapse causal for control performance, or epiphenomenal?

**Scale.** 15 DMC tasks (state and pixel), 1M env steps, TD-MPC2-class agent, $d=512$, 10 seeds per arm. ~4 arms × 15 tasks × 10 seeds ≈ 600 runs; roughly 2–4k A100-hours. Report IQM with stratified bootstrap CIs (Agarwal et al., NeurIPS 2021).

**Arms.**
1. *Baseline*: stop-gradient + EMA, no rank regulariser.
2. *Control (matched-loss)*: identical, plus a **rank-neutral** regulariser of equal gradient norm (random-projection L2 penalty) — the placebo that rules out "any extra loss term helps".
3. *Treatment*: plus VICReg variance floor, tuned only to raise terminal erank to the level of arm 1's best-performing seed.
4. *Reference ceiling*: encoder frozen at a supervised state-predictive initialisation.

**Instrumentation.** Log $\mathrm{erank}(Z)$ every 10k steps on a **fixed held-out observation set** (not the replay buffer — this removes the policy-distribution confound), plus the behavioural residual $\Delta$ from §2.

**The deciding number.** Let $G$ be the IQM return gap between arm 1 and arm 4. Compute $\rho = \big(\mathrm{IQM}_3 - \mathrm{IQM}_2\big)/G$.
- $\rho \ge 0.5$ → collapse is causal; rank regularisation belongs in every self-predictive world model.
- $\rho \le 0.1$ with erank in arm 3 raised $\ge 3\times$ → rank is epiphenomenal, and the correlational literature is measuring the policy's state distribution, not the encoder.

## 9. Key References

- **[Foundational]** Grill, Strub, Altché, Tallec, Richemond, et al. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* NeurIPS, 2020. — arXiv:2006.07733
- **[Foundational]** Chen, He. *Exploring Simple Siamese Representation Learning.* CVPR, 2021. — arXiv:2011.10566
- **[Theory SOTA]** Tian, Chen, Ganguli. *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs.* ICML, 2021. — arXiv:2102.06810
- **[Theory SOTA]** Tang, Guo, Richemond, Pires, Chandak, Munos, et al. *Understanding Self-Predictive Learning for Reinforcement Learning.* ICML, 2023. — arXiv:2212.03319
- **[SOTA]** Ni, Eysenbach, Seyedsalehi, Ma, Gehring, Mahajan, Bacon. *Bridging State and History Representations: Understanding Self-Predictive RL.* ICLR, 2024. — arXiv:2401.08898
- **[SOTA]** Schwarzer, Anand, Goel, Hjelm, Courville, Bachman. *Data-Efficient Reinforcement Learning with Self-Predictive Representations.* ICLR, 2021. — arXiv:2007.05929
- **[SOTA]** Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR, 2024. — arXiv:2310.16828
- **[SOTA]** Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR, 2022. — arXiv:2105.04906
- **[Analysis]** Jing, Vincent, LeCun, Tian. *Understanding Dimensional Collapse in Contrastive Self-Supervised Learning.* ICLR, 2022. — arXiv:2110.09348
- **[Analysis]** Kumar, Agarwal, Ghosh, Levine. *Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning.* ICLR, 2021. — arXiv:2010.14498
- **[Analysis]** Lyle, Rowland, Dabney. *Understanding and Preventing Capacity Loss in Reinforcement Learning.* ICLR, 2022. — arXiv:2204.09560
- **[Analysis]** Richemond, Grill, Altché, Tallec, et al. *BYOL Works Even Without Batch Statistics.* NeurIPS SSL Workshop, 2020. — arXiv:2010.10241
- **[Position]** LeCun. *A Path Towards Autonomous Machine Intelligence.* OpenReview, 2022.
- **[Survey]** Balestriero, Ibrahim, Sobal, Morcos, et al. *A Cookbook of Self-Supervised Learning.* 2023. — arXiv:2304.12210

## 10. Worked Example

**Setup.** DMC `walker-walk`, proprioceptive state dimension 24, action dimension 6. Latent width $d=512$. Suppose a self-predictive agent, trained 500k steps, ends with:

| quantity | step 50k | step 500k |
|---|---|---|
| latent loss | 0.31 | 0.004 |
| $\mathrm{erank}(Z)$, replay buffer | 137 | 11 |
| $\mathrm{erank}(Z)$, fixed held-out set | 141 | 38 |
| episode return | 260 | 780 |

**The arithmetic.** Erank fell $137\to11$ — a 12× drop that the standard protocol would report as severe collapse. But the deciding comparison is the task's intrinsic dimension. The controllable state is 24-d, and a $k$-step-sufficient latent needs at most $24 + 6H$ dimensions in the worst case; at $H=1$ that is 30. So $\mathrm{erank}=38$ on the *fixed* set is **above** the sufficiency floor — not collapsed, compressed correctly. The $11$ measured on the replay buffer is an artefact: the converged policy walks a near-periodic gait, so its own state distribution has low rank, and 27 of the 38 usable dimensions are simply never excited.

**What this makes visible.** Return went *up* by 3× while the headline collapse statistic went down 12×. Two of the three numbers in the table (loss, replay-buffer erank) are monotone in the direction of "collapse", and both are wrong about the model. The only measurement that would settle it — the behavioural residual $\Delta$ against a known-sufficient reference encoder — requires a ground truth for minimal sufficient dimension that no benchmark supplies. That missing baseline, not the optimisation dynamics, is the current obstruction: the field can detect *full* collapse reliably and cannot decide *dimensional* collapse at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*