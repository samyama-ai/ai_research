---
id: 18-rl-for-llms/group-relative-vs-learned-value-baselines
title: "Group-Relative Baselines Versus Learned Value Functions"
topic: 18-rl-for-llms
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Group-Relative Baselines Versus Learned Value Functions

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/group-relative-vs-learned-value-baselines` · **Status:** empirically-open

## 1. Problem Statement

Policy-gradient post-training of LLMs needs a baseline to reduce the variance of the gradient estimate. Two families are in production use:

- **Group-relative (critic-free):** sample $G$ completions per prompt, subtract a statistic of the group's rewards (GRPO, RLOO, ReMax, Dr. GRPO). Every token in a trajectory receives the same advantage.
- **Learned value (critic-based):** train $V_\phi(s_t)$ on-policy and form per-token advantages via GAE (PPO, VC-PPO, VAPO). Tokens within one trajectory receive different advantages.

**The decision predicate.** At a fixed total training FLOP budget $C$ and a fixed prompt set, does the critic-based estimator reach higher terminal task reward than the best group-relative estimator? Formally: is there a compute budget at which per-token credit assignment beats trajectory-level credit assignment, and does that crossover move with sequence length $T$, group size $G$, or reward sparsity?

Three variants, different difficulty:
- **Measurement:** compare the two at *matched* FLOPs rather than matched gradient steps. Currently almost never done.
- **Method:** build a value estimator whose bias–variance profile beats the group mean at long $T$ — VinePPO's Monte-Carlo values and VAPO's value pretraining are attempts.
- **Theory:** characterize when a *state-dependent* baseline strictly dominates a *prompt-dependent* one for the gradient's variance, given that in the LLM setting reward is terminal and deterministic given the completion.

## 2. Formal Setting

A prompt $x \sim \mathcal{D}$, a policy $\pi_\theta$ generating $o = (o_1,\dots,o_T)$ autoregressively, and a terminal scalar reward $R(x,o) \in \mathbb{R}$ (verifier output, $\{0,1\}$ for math/code; a reward-model score otherwise). State $s_t = (x, o_{<t})$.

The score-function gradient:
$$\nabla_\theta J = \mathbb{E}_{x,o}\Big[\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(o_t \mid s_t)\,\big(R(x,o) - b(s_t)\big)\Big].$$

**Group-relative.** Draw $\{o^{(i)}\}_{i=1}^{G} \sim \pi_\theta(\cdot\mid x)$. GRPO uses
$$\hat{A}^{\text{GRPO}}_i = \frac{R_i - \bar{R}}{\mathrm{std}(R)},\qquad \bar R = \tfrac1G\sum_j R_j,$$
constant over $t$. RLOO uses the leave-one-out mean $b_i = \frac{1}{G-1}\sum_{j\neq i} R_j$; Dr. GRPO drops the $\mathrm{std}$ divisor and the $1/|o_i|$ length normalization.

**Critic-based.** $V_\phi$ trained by regression on $\lambda$-returns; $\delta_t = R\,\mathbb{1}[t{=}T] + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$; $\hat A_t^{\text{GAE}} = \sum_{l\ge0}(\gamma\lambda)^l \delta_{t+l}$.

**Quantities as measured.**
- Compute: $C = N_{\text{steps}}\big(c_{\text{gen}} G T + c_{\text{fwd-bwd}} G T + \mathbb{1}[\text{critic}]\,c_{\text{critic}} G T\big)$ in FLOPs, plus wall-clock and peak HBM (the critic adds a second full-size or shrunken network; group methods add $G{-}1$ rollouts).
- Gradient noise: $\mathrm{tr\,Cov}(\hat g)/\|\mathbb{E}\hat g\|^2$, estimated by splitting each batch into $K$ shards and comparing shard gradients — the "gradient noise scale" of McCandlish et al. (2018).
- Credit-assignment quality: rank correlation between $\hat A_t$ and a ground-truth $V^{\pi}(s_t)$ obtained by $M{\ge}64$ MC rollouts from $s_t$ (VinePPO's protocol). This is the only direct measurement of the thing the critic is supposed to supply.

**Assumptions, and which are violated.**
1. *Baseline independent of the sampled action* — required for unbiasedness. Violated by GRPO: $\bar R$ contains $R_i$, giving an $O(1/G)$ bias; RLOO satisfies it.
2. *Scale-invariance of the estimator* — violated by dividing by $\mathrm{std}(R)$, which upweights prompts whose group is nearly all-correct or all-wrong (Liu et al., 2025).
3. *$V_\phi$ is on-policy* — violated after every optimizer step and by the multi-epoch inner loop of PPO; the critic lags the policy it scores.
4. *Reward is terminal and deterministic* — true for verifiers, false for RM-scored chat, where reward noise changes the optimal baseline.
5. *$\gamma = \lambda = 1$* (usual in LLM RL) — makes GAE collapse to $R - V_\phi(s_t)$, so the critic's entire contribution is the per-token offset.

## 3. State of the Art

**Established.**
- GRPO at 7B (DeepSeekMath, Shao et al. 2024) and at 671B (DeepSeek-R1, 2025) produces large reasoning gains without a critic. That critic-free RL *works* at frontier scale is established.
- RLOO (Ahmadian et al., ACL 2024) shows on 6–7B models and TL;DR/HH that an unbiased leave-one-out baseline with $k \in \{2,4\}$ matches or beats PPO while dropping the critic entirely — with per-run compute reported.
- Dr. GRPO (Liu et al., 2025) isolates two GRPO biases (std normalization, token-length normalization) and shows removing them improves token efficiency at 1.5B–7B. This is a real ablation, not a benchmark number.

**Claimed but unablated.**
- VAPO (ByteDance Seed, 2025) reports 60.4 on AIME 2024 with Qwen2.5-32B base, above DAPO's reported 50.0, and attributes the gap to value-model fixes (value pretraining, decoupled GAE, length-adaptive $\lambda$). The comparison is against a different codebase and a different data pipeline; the critic's isolated contribution is not separated from the other changes.
- VinePPO (Kazemnejad et al., 2024) claims MC value estimates beat a learned critic at equal *gradient steps* and reach PPO's peak accuracy in far fewer steps at 1.1B–7B. Equal-FLOP accounting is weaker: MC value estimation costs extra rollouts.
- Widespread practitioner claims that critics are "unnecessary for verifiable rewards" rest mostly on GRPO's popularity, not on matched-compute head-to-heads.

**Where results are only benchmark numbers:** nearly all 32B-scale comparisons (DAPO, VAPO, and vendor RL reports) are single-seed, single-checkpoint AIME/AMC scores with $n \le 30$ problems.

## 4. What Is Known

- **Optimal-baseline theory.** The variance-minimizing baseline is not $\mathbb{E}[R]$ but the $\|\nabla\log\pi\|^2$-weighted mean of returns (Weaver & Tao, UAI 2001; Greensmith, Bartlett & Baxter, JMLR 2004). Both the group mean and $V_\phi$ are approximations to different objects, and neither is the optimum.
- **Leave-one-out is unbiased; group-mean-including-self is not.** Kool et al. (ICLR 2019 workshop) and Ahmadian et al. (2024) give the estimator and the bias term.
- **Numbers.** DeepSeekMath-7B-RL: GSM8K 82.9 → 88.2, MATH 46.8 → 51.7 top-1 after GRPO on the instruct checkpoint (Shao et al., 2024). DAPO: 50 points on AIME 2024 with Qwen2.5-32B base at ~50% of the training steps of the DeepSeek-R1-Zero-Qwen-32B recipe (47 points).
- **Critics are miscalibrated on long CoT.** VinePPO measures PPO's critic against MC ground truth at 1.1B–7B and finds it barely above chance at ranking states within a trajectory; VC-PPO independently identifies value-initialization bias and reward-decay bias in long sequences.
- **Cost.** A same-size critic roughly doubles optimizer memory and adds ~30–50% step time; going from $G=1$ to $G=8$ multiplies generation cost ~8×. So critic-free is not automatically cheaper per unit of *learning*.

## 5. What Is Not Known

- **Empirically open (primary).** No public, matched-FLOP, multi-seed comparison of GRPO/RLOO against a well-tuned critic at $\ge$ 30B parameters with $T \ge 16$k tokens. The experiment is runnable today on a few hundred GPU-days; nobody has published it with controls.
- **Empirically open.** Whether the critic's advantage (if any) grows with $T$. Both camps assert a $T$-dependence in opposite directions: critics for finer credit assignment on long chains, group methods because critics degrade on long chains.
- **Theoretically open.** No variance bound comparing $\mathrm{Var}[\hat g^{\text{RLOO}}(G)]$ to $\mathrm{Var}[\hat g^{\text{GAE}}(V_\phi)]$ under a realistic model of $V_\phi$'s error. The $G$-sample group estimator's variance scales as $O(1/G)$ in the between-trajectory term but does nothing for the within-trajectory term; the size of the within-trajectory term for autoregressive LLMs with terminal reward is unquantified.
- **Methodologically blocked.** "Credit assignment quality" has no agreed metric. MC-rollout $V^\pi$ is itself policy-dependent and expensive ($M{\times}$ generation), and any correlation-to-$V^\pi$ score presumes $V^\pi$ is the right target rather than, say, a counterfactual token-influence measure.

## 6. Why It Is Hard

**Confounded measurement, compounded by unequal compute.** Every published comparison varies at least three things at once: the estimator, the clipping/regularization scheme (GRPO ships with a KL term and token-level ratios that PPO baselines often lack), and the data mix. Papers report equal *gradient steps* or equal *epochs*, not equal FLOPs — which systematically favors whichever method spends more compute per step. A critic-based run at $G=1$ and a group run at $G=8$ differ by ~8× in generation cost; comparing them at equal steps measures the budget, not the estimator.

**Absent ground truth for $V^\pi$.** Validating a critic needs the true value function, obtainable only by expensive MC rollouts that are themselves stale the moment the policy updates.

**Non-identifiability at $\gamma=\lambda=1$.** With terminal reward and no discounting, $\hat A_t^{\text{GAE}} = R - V_\phi(s_t)$ — a per-token *shift*. Since the sum of score functions over a trajectory has zero mean under $\pi_\theta$, part of that shift is a pure baseline (variance-only) and part is genuine credit reallocation. No published protocol separates the two, so a win for the critic cannot be attributed.

**Small-$n$ evaluation.** AIME 2024 has 30 problems; a 3-problem swing is 10 points. Seed variance in LLM RL runs is comparable to the reported method gaps.

## 7. Current Research (as of 2026)

- **Critic rehabilitation for long CoT:** ByteDance Seed's VC-PPO/VAPO line (value pretraining, decoupled GAE $\lambda$ for actor and critic, length-adaptive $\lambda$). *(frontier — verify whether the value-specific ablations hold outside their data pipeline.)*
- **Estimator hygiene in the group family:** Dr. GRPO's unbiased loss, DAPO's dynamic sampling (dropping all-correct/all-wrong groups) and clip-higher, and follow-on "tricks or traps" studies systematically ablating normalization choices at 4B–32B (Alibaba ROLL/Qwen-adjacent groups). *(frontier — verify.)*
- **Process reward models and MC-based per-step advantages** as a third path that supplies token-level signal without a bootstrapped critic (VinePPO lineage, PRM work from Shanghai AI Lab and OpenAI's earlier `PRM800K`).
- **Open infrastructure** (verl, OpenRLHF, TRL) now implements both families in one codebase, which is what finally makes a clean matched-FLOP study cheap. As of this writing no such study has been published.

## 8. Concrete Next Experiment

**Scale.** Qwen3-8B-Base (or Llama-3.1-8B), verifiable-math prompt set of ~40k problems, max generation length 16k tokens, 3 seeds per arm, ~$1.5\times10^{21}$ training FLOPs per arm (roughly 400 H100-hours), single codebase (verl), identical KL coefficient, clipping, and data order.

**Arms (all budget-matched in FLOPs, not steps):**
1. **RLOO**, $G=8$, unbiased leave-one-out, no std normalization.
2. **PPO + learned critic**, $G=1$, $\gamma=\lambda=1$, critic initialized from the SFT backbone with a value head, 20-step value warmup.
3. **Control arm — "shift-only critic":** PPO with the critic's output *averaged over the trajectory*, $\hat A_t = R - \frac1T\sum_u V_\phi(s_u)$. This keeps the critic's variance reduction but destroys its per-token credit assignment. It is the arm that separates "the critic is a better baseline" from "per-token credit assignment matters."
4. **RLOO + shift-only critic** (both baselines composed), to test additivity.

**The deciding number.** $\Delta = \text{acc}_{\text{arm 2}} - \max(\text{acc}_{\text{arm 1}}, \text{acc}_{\text{arm 3}})$ on a held-out 500-problem verifiable set at equal FLOPs, averaged over 3 seeds, with a paired bootstrap CI. If $\Delta \le 1.0$ point with the CI excluding $+2$, per-token credit assignment buys nothing at 8B/16k and the group family is the correct default. If $\Delta \ge 3$ points *and* arm 3 tracks arm 1, the gain is credit assignment, not variance reduction — which is the result that would justify the critic's memory cost. Secondary readout: Spearman $\rho$ between $\hat A_t$ and 64-rollout MC values at steps 0/100/400, to check whether any win coincides with a calibrated critic.

## 9. Key References

- **[Foundational]** Lex Weaver, Nigel Tao. *The Optimal Reward Baseline for Gradient-Based Reinforcement Learning.* UAI, 2001.
- **[Foundational]** Evan Greensmith, Peter Bartlett, Jonathan Baxter. *Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning.* JMLR 5, 2004.
- **[Foundational]** John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR, 2016. — arXiv:1506.02438
- **[Foundational]** John Schulman et al. *Proximal Policy Optimization Algorithms.* 2017. — arXiv:1707.06347
- **[Foundational]** Wouter Kool, Herke van Hoof, Max Welling. *Buy 4 REINFORCE Samples, Get a Baseline for Free!* ICLR Workshop on Deep RL Meets Structured Prediction, 2019.
- **[SOTA]** Zhihong Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300 (introduces GRPO)
- **[SOTA]** Arash Ahmadian et al. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* ACL, 2024. — arXiv:2402.14740 (RLOO)
- **[SOTA]** Amirhossein Kazemnejad et al. *VinePPO: Unlocking RL Potential for LLM Reasoning Through Refined Credit Assignment.* 2024. — arXiv:2410.01679
- **[SOTA]** Zichen Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. — arXiv:2503.20783 (Dr. GRPO)
- **[SOTA]** Qiying Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[SOTA]** ByteDance Seed. *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks.* 2025. (value-based counter-argument; arXiv preprint)
- **[Survey]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948

## 10. Worked Example

Take one prompt with $\pi_\theta$'s pass rate $p = 0.5$, binary reward, $G = 8$, $T = 4000$ tokens.

**Group-relative variance.** The between-trajectory term of the advantage has variance $p(1-p) = 0.25$. With RLOO and $G = 8$ the effective baseline error contributes a factor $\big(\tfrac{G}{G-1}\big)^2/G \approx 0.163$ relative to a single unbaselined sample — a ~6× variance reduction, for 8× the generation FLOPs.

**Critic.** With $\gamma=\lambda=1$, $\hat A_t = R - V_\phi(s_t)$. A perfectly calibrated critic at $t=0$ gives $V_\phi(s_0) = 0.5$ — exactly the group mean, at $G=1$ generation cost. So *at the first token the critic is the group baseline, for free.* Its only extra content is how $V_\phi(s_t)$ moves for $t > 0$.

**Now the obstruction.** Measure that movement. VinePPO-style probing at 7B finds learned critics on long math CoT rank within-trajectory states near chance; suppose the measured Spearman $\rho$ between $V_\phi(s_t)$ and 64-rollout $V^\pi(s_t)$ is $0.15$. Then across $T = 4000$ tokens the critic injects a nearly-random per-token offset $\epsilon_t$ with $\mathrm{Var}(\epsilon_t) \approx (1-\rho^2)\,\mathrm{Var}(V^\pi) \approx 0.98 \times 0.25$. Summed over the trajectory this *adds* variance in the same order as the $6\times$ reduction RLOO bought — but it is not a bias, since the offset is state-dependent and the score functions still have zero mean.

The consequence: the two effects (variance added by a noisy critic, variance removed by a correct one) are the same size at plausible parameter values, and their net sign depends on $\rho$, which nobody measures during production runs. That is why the question is empirically open rather than settled by argument — and why the "shift-only critic" control arm in §8 is necessary: without it, any observed win is attributable to either mechanism.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*