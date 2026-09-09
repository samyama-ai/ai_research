---
id: 18-rl-for-llms/optimizer-stability-large-policy-models
title: "Optimizer Stability for Very Large Policy Models"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimizer Stability for Very Large Policy Models

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/optimizer-stability-large-policy-models` · **Status:** open

## 1. Problem Statement

RL post-training of a large language model (policy gradient on sampled rollouts, scored by a verifier or reward model) collapses at long horizons in ways pretraining does not. Runs that log a healthy reward curve for 500 steps then lose entropy, blow up gradient norm, drift KL, and emit degenerate text. Practitioners patch this with clipping, KL penalties, reference resets, and optimizer-state resets — none of which has a principled selection rule at the scale where it matters ($10^{11}$–$10^{12}$ parameters, mixture-of-experts routing, thousands of optimizer steps).

Three variants, different difficulty:

- **Measurement.** Given a run, decide *before* collapse whether it is on a divergent trajectory. Input: the streamed telemetry $(\nabla, m, v, H_\pi, \mathrm{KL}, \text{reward})$. Output: a binary predicate with a stated lead time. Currently there is no validated leading indicator.
- **Method.** Produce an optimizer + update rule for which the probability of catastrophic divergence over $T$ steps is below some $\delta$ *without* the reward ceiling dropping. The tradeoff is the whole difficulty: every known stabilizer is also a learning-rate cut in disguise.
- **Theory.** Prove a non-asymptotic bound on Adam/Muon-family updates under *non-stationary, self-generated, partly off-policy* data with a bounded-support reward. No such bound exists.

Solving it means: a recipe whose failure rate at $10^{12}$ parameters over $10^4$ RL steps is measured, not anecdotal, and whose stabilizer settings transfer from a proxy scale.

## 2. Formal Setting

Policy $\pi_\theta$, $\theta \in \mathbb{R}^d$. At step $t$, sample a prompt batch $\mathcal{B}_t \sim \mathcal{D}$, generate $G$ responses per prompt from an *inference-engine* policy $\pi_{\theta_{\text{inf}}}$, score with reward $r \in [0,1]$, and form group-relative advantages
$$A_{i} = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon_\sigma}.$$
The surrogate is a clipped importance-weighted objective with ratio $\rho_i = \pi_\theta(y_i|x)/\pi_{\theta_{\text{inf}}}(y_i|x)$, taken per-token (PPO/GRPO) or per-sequence (GSPO).

**Measured quantities.**

| Symbol | Definition | How it is actually read |
|---|---|---|
| $g_t$ | minibatch gradient | post-all-reduce, pre-clip global $\ell_2$ norm, logged per step |
| $\hat\kappa_t$ | update ratio | $\|\Delta\theta_t\|_2 / \|\theta_t\|_2$, per parameter block |
| $H_t$ | policy entropy | mean token entropy over the *sampled* rollouts, not the full vocabulary distribution over a held-out set |
| $\mathrm{KL}_t$ | drift from reference | $k_3$ estimator $\;\mathbb{E}[\rho^{-1}-1-\log\rho^{-1}]$ on rollout tokens |
| $\Delta_t$ | train/infer mismatch | $\mathbb{E}\,|\log \pi_\theta - \log \pi_{\theta_{\text{inf}}}|$ at $\theta_{\text{inf}} = \theta$ |
| $S_t$ | Adam second-moment floor | fraction of coordinates with $\sqrt{v_t} < \varepsilon$ |

Divergence event: $\tau = \min\{t: \hat\kappa_t > 10^2 \cdot \mathrm{median}_{s<t}\,\hat\kappa_s \ \text{or}\ \text{reward}_t < \tfrac12 \max_{s<t} \text{reward}_s\}$. The target object is $P(\tau < T)$ as a function of $d$, $T$, and the stabilizer settings.

**Assumptions, and which are false.**

- *$L$-smoothness of the surrogate.* Violated — transformer losses show norm-dependent smoothness; the effective $L$ grows with the attention-logit scale (Wortsman et al., ICLR 2024).
- *Stationary gradient noise.* Violated by construction: the data distribution is $\pi_\theta$ itself, and reward variance collapses as the policy sharpens, so $\sigma^2_{\text{grad}} \to 0$ on solved prompts and the batch becomes dominated by a shrinking hard set.
- *$\rho_i \approx 1$ at collection time.* Violated. Kernel-level differences between the training and inference stacks make $\Delta_t > 0$ even at identical weights, so "on-policy" RL is silently off-policy.
- *Fixed loss geometry.* Violated for MoE: the routing function is a discrete latent that changes which experts receive gradient, so $d_{\text{eff}}$ is time-varying.

## 3. State of the Art

**Empirical/systems SOTA.**

- *Clip-and-shape family.* DAPO (Yu et al., 2025, arXiv:2503.14476) decouples the clip range (clip-higher), removes the KL penalty, and filters all-correct/all-wrong groups; reported 50 on AIME'24 with Qwen2.5-32B. Established: the entropy-preservation effect of clip-higher is ablated in the paper. Unablated: whether it stabilizes runs beyond the ~$10^3$-step horizon reported.
- *Estimator-level fixes.* Dr. GRPO (Liu et al., 2025, arXiv:2503.20783) shows the length and std normalizers in GRPO are biased and produce a response-length blow-up that is often *misread as* an emergent reasoning behavior. This is established with a clean control. GSPO (Zheng et al., Qwen, 2025, arXiv:2507.18071) moves the ratio to sequence level and reports that it removes MoE routing-induced collapse without the "routing replay" hack. The MoE claim is the strongest single stability result in the literature and has not been independently replicated at another lab's scale.
- *Precision fixes.* Computing the final logits/log-probs in FP32 is repeatedly reported to remove a large part of the train/infer mismatch. In *The Art of Scaling RL Compute for LLMs* (Meta, 2025), FP32 logits is one of the components of the ScaleRL recipe, and the paper fits sigmoidal compute–performance curves whose asymptote is predicted from early training. Established as a fit; the extrapolation claim is the interesting part and rests on their own runs.
- *Optimizer-level.* MuonClip in Kimi K2 (Moonshot AI, 2025) applies qk-clip to bound attention logits and is reported to give a spike-free 15.5T-token pretraining run. This is a pretraining result imported into the RL setting by assumption, not by measurement.

**Theory SOTA.** Convergence results for policy gradient with adaptive optimizers exist only under assumptions the setting violates (bounded gradients, stationary noise, exact on-policy sampling). Molybog et al. (2023, arXiv:2304.09871) give the one mechanistic account with teeth: Adam's update becomes unstable when the time-domain correlation between $m_t$ and $\sqrt{v_t}$ breaks after a long no-gradient interval for a coordinate — an $\varepsilon$-floor story, and RL's sparse reward signal makes such intervals common.

## 4. What Is Known

- **Length inflation is an artifact, not a capability.** Removing GRPO's $1/|y|$ and group-std normalizers removes most of the response-length growth without hurting accuracy — measured on Qwen2.5-Math-7B (Liu et al., 2025).
- **Entropy collapse is monotone and fast.** Across 1.5B–32B models, mean rollout entropy falls by roughly an order of magnitude within the first few hundred GRPO steps under symmetric clipping; pass@$k$ at large $k$ degrades while pass@1 improves.
- **Train/infer log-prob mismatch is real and non-trivially large.** Independent reports put $\Delta_t$ at the $10^{-2}$–$10^{-1}$ nat range per token for BF16 inference engines vs. the training forward pass at identical weights, enough to make truncated importance sampling a measurable improvement.
- **Small-scale proxies reproduce some instabilities.** Attention-logit growth and output-logit divergence appear at ~$10^8$ parameters under high LR and are fixed by qk-layernorm and z-loss; $\mu$P-style LR transfer holds for these (Wortsman et al., ICLR 2024, arXiv:2309.14322; Yang et al., Tensor Programs V, arXiv:2203.03466).
- **MoE routing is a distinct failure channel.** Qwen report per-token clipped objectives failing on their MoE models where the dense counterparts were fine.

## 5. What Is Not Known

- **Theoretically open.** Any non-asymptotic bound on $P(\tau < T)$ for Adam or Muon under self-generated, non-stationary data with group-relative advantages. Also open: whether entropy collapse is a fixed point of the *estimator* or of the *optimizer* — no separation theorem exists.
- **Empirically open.** Whether stabilizer hyperparameters (clip-higher $\epsilon_{\text{high}}$, KL coefficient, $\varepsilon_{\text{Adam}}$, gradient-clip threshold) transfer across scale the way $\mu$P transfers LR. The experiment is a 3-scale sweep; nobody has published it. Also open: whether the ScaleRL sigmoid extrapolation survives outside Meta's recipe family.
- **Methodologically blocked.** "Stability" has no agreed operational definition. Papers report smooth reward curves as evidence, but reward is the quantity being optimized and a collapsed policy can hold reward while losing diversity. There is no accepted leading indicator with a published ROC curve and lead time.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a factorial cost**. Divergence is rare per run and only appears in the long-horizon, large-$d$ regime; a single decisive run at $10^{11}$ parameters over $10^4$ steps costs $10^5$–$10^6$ GPU-hours, so nobody runs the $n \geq 20$ seeds needed to estimate a failure probability. What gets published instead is one run per configuration, where a "stable" outcome is unfalsifiable and any fix is confounded with the effective learning-rate reduction it induces. Second, the reward signal is not ground truth for stability: reward hacking and entropy collapse both *raise* the logged number. Third, the train/infer mismatch means the measured ratio $\rho$ conflates a real policy update with a kernel-level numerical artifact — the two are not separately identifiable from the training log alone.

## 7. Current Research (as of 2026)

- **Estimator geometry.** Sequence-level vs. token-level ratios (Qwen/GSPO), and softened clipping such as MiniMax's CISPO. Active in Chinese frontier labs; dense–MoE contrasts are the discriminating evidence.
- **Numerical determinism.** Making the inference and training forward passes bitwise-consistent, so $\Delta_t = 0$ by construction rather than corrected by importance weights. Thinking Machines' batch-invariant-kernel work is the clearest public statement of the mechanism. *(frontier — verify: whether any lab has shipped a fully bitwise-matched RL loop at MoE scale.)*
- **Predictive scaling of RL compute.** Fitting sigmoidal reward-vs-compute curves and using the fit to reject recipes early (Meta ScaleRL). *(frontier — verify: independent replication.)*
- **Optimizer-side control.** Muon/MuonClip and second-moment-free updates in the RL loop rather than pretraining only. *(frontier — verify.)*
- **Asynchronous/off-policy RL.** Staleness-tolerant pipelines (AReaL, asynchronous RLHF) trade $\Delta_t$ for throughput, making the mismatch question central rather than incidental.

## 8. Concrete Next Experiment

**Question.** Do stabilizer settings transfer across scale, and does any telemetry channel predict collapse with usable lead time?

**Scale.** Three dense policies at 1.5B / 8B / 32B, same tokenizer family, same verifiable-math prompt set, GRPO with $G=16$, 3,000 optimizer steps, **5 seeds per cell** — seeds are the point, not scale.

**Arms.** (a) Control: symmetric clip $\epsilon=0.2$, GRPO normalizers as published, BF16 logits, KL coefficient $10^{-3}$. (b) Clip-higher $\epsilon_{\text{high}}=0.28$. (c) Dr. GRPO normalizers. (d) FP32 logits + truncated importance sampling. (e) All of (b)–(d).

**Instrument.** Log $\hat\kappa_t$, $S_t$, $\Delta_t$, $H_t$ per step; every 100 steps evaluate pass@1 and pass@256 on a held-out set.

**Deciding number.** The **AUC of a single-channel collapse detector at 200-step lead time**, and, separately, the **rank correlation of per-arm empirical divergence rate $\hat P(\tau<3000)$ across the three scales**. Transfer is supported if Spearman $\rho \geq 0.8$ between the 1.5B and 32B arm rankings; it is refuted if $\rho \leq 0.3$. If the best detector's AUC $< 0.7$, the measurement variant is confirmed blocked and no method work should be trusted until it is fixed. Cost estimate: ~75 runs, order $3\times10^4$ H100-hours — within one academic-consortium budget, which is why the absence of this experiment is a gap rather than an impossibility.

## 9. Key References

- **[Foundational]** Shao, Z., Wang, P., Zhu, Q., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300
- **[Foundational]** Molybog, I., Albert, P., Chen, M., et al. *A Theory on Adam Instability in Large-Scale Machine Learning.* 2023. — arXiv:2304.09871
- **[Foundational]** Wortsman, M., Liu, P. J., Xiao, L., et al. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Foundational]** Yang, G., Hu, E. J., Babuschkin, I., et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[SOTA]** Yu, Q., Zhang, Z., Zhu, R., et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[SOTA]** Liu, Z., Chen, C., Li, W., et al. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. — arXiv:2503.20783
- **[SOTA]** Zheng, C., Liu, S., Li, M., et al. *Group Sequence Policy Optimization.* Qwen Team, Alibaba, 2025. — arXiv:2507.18071
- **[SOTA]** Khatri, D., et al. *The Art of Scaling Reinforcement Learning Compute for LLMs.* Meta / collaborators, 2025. (arXiv preprint; identifier omitted — verify before citing.)
- **[SOTA]** Moonshot AI. *Kimi K2: Open Agentic Intelligence.* Technical report, 2025. (MuonClip / qk-clip; identifier omitted.)
- **[Survey]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025.

## 10. Worked Example

Take an 8B dense policy, GRPO, $G=8$, 512 prompts/step, mean response 2,000 tokens — about $8\times10^6$ tokens of gradient signal per step. Reward is binary verifier output.

By step 400, 60% of prompts are solved by all 8 samples and 15% by none. Both groups have $\sigma_{\text{group}} = 0$, so with the standard $\sigma + \epsilon_\sigma$ normalizer their advantages are zero. The **effective batch is 25% of the nominal batch**, and gradient variance per step rises by roughly $1/0.25 = 4\times$ while the logged gradient norm *falls*, because 75% of the terms are exactly zero. A monitor watching $\|g_t\|$ sees a calm curve at the exact moment the signal-to-noise ratio drops fourfold.

Simultaneously, the solved prompts' tokens stop producing gradient. For a coordinate touched only by those tokens, Adam's $v_t$ decays as $\beta_2^{\,k}$; with $\beta_2 = 0.95$, after $k=200$ dry steps $v_t$ has shrunk by $0.95^{200} \approx 3.5\times10^{-5}$. When a hard prompt finally activates that coordinate, the update is $m_t/(\sqrt{v_t}+\varepsilon)$ with a $\sqrt{v_t}$ that has collapsed toward $\varepsilon = 10^{-8}$ — the ratio saturates at $|m_t|/\varepsilon$ and the step is clipped only by the global norm clip, which is being met by one coordinate block. That is the Molybog mechanism, and RL's sparse-reward structure manufactures the dry intervals that pretraining rarely produces.

The obstruction is visible here: the two effects — shrinking effective batch and second-moment floor — have *opposite* signatures in gradient norm, and the only channel that sees both, $\hat\kappa_t$ per block, is not routinely logged. Fixing this by raising $\epsilon_\sigma$ or filtering degenerate groups also changes the effective learning rate, so a single run showing "it stabilized" cannot distinguish the mechanism from the LR cut. The experiment in §8 exists precisely to break that confound with seeds.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*