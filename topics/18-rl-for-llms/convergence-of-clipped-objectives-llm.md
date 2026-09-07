---
id: 18-rl-for-llms/convergence-of-clipped-objectives-llm
title: "Provable Convergence of Clipped Policy-Gradient Objectives for LLMs"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Convergence of Clipped Policy-Gradient Objectives for LLMs

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/convergence-of-clipped-objectives-llm` · **Status:** open

## 1. Problem Statement

PPO-style clipped surrogates (PPO, GRPO, DAPO, GSPO and variants) are the workhorse of RLHF and RLVR. No one has a convergence theorem for the objective that is actually run: a clipped ratio, estimated with a small group of sampled completions, on a token-level MDP with a terminal-only reward, under a *stale* behaviour policy (generation happens minibatches or steps before the update), with per-token normalisation and length bias, optimised by Adam on a transformer.

Three variants, different difficulty:

- **Theory.** Give conditions on clip range $\epsilon$, group size $G$, staleness $\tau$, and step size $\eta$ under which the clipped iterate sequence converges to a stationary point (or global optimum, for softmax tabular) of the *true* KL-regularised RL objective — not of the surrogate.
- **Method.** Design a clipped estimator that keeps PPO's empirical stability but has a bound. Candidates: sequence-level ratios, mirror-descent forms with an explicit proximal term, unbiased truncated importance sampling.
- **Measurement.** Decide empirically whether the runs that are done today are converging at all, or merely stopping. There is currently no accepted diagnostic distinguishing "converged" from "entropy collapsed" from "clipping killed the gradient".

Solved means: a theorem covering nonzero clip, $G<\infty$, $\tau>0$, whose assumptions are checkable on a real run, plus a measured instance where the predicted rate matches observed progress within a stated factor.

## 2. Formal Setting

Prompt $x\sim\mathcal{D}$, completion $y=(y_1,\dots,y_T)$, policy $\pi_\theta(y\mid x)=\prod_{t}\pi_\theta(y_t\mid x,y_{<t})$ over vocabulary $\mathcal{V}$, $|\mathcal{V}|\approx 1.5\times10^5$. Reward $r(x,y)\in\{0,1\}$ for verifiable tasks (measured: unit-test pass or exact-match on a parsed answer), or a learned scalar $r_\phi$ for RLHF.

Objective:
$$J(\theta)=\mathbb{E}_{x,\,y\sim\pi_\theta}\big[r(x,y)\big]-\beta\,\mathrm{KL}\!\left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right).$$

Token ratio $\rho_t(\theta)=\pi_\theta(y_t\mid x,y_{<t})/\pi_{\theta_{\mathrm{old}}}(y_t\mid x,y_{<t})$. Clipped surrogate:
$$L(\theta)=\mathbb{E}\Big[\tfrac{1}{|y|}\textstyle\sum_t \min\big(\rho_t A,\ \mathrm{clip}(\rho_t,1-\epsilon_{\text{lo}},1+\epsilon_{\text{hi}})A\big)\Big].$$

GRPO advantage from a group of $G$ completions per prompt: $A_i=(r_i-\bar r)/\mathrm{std}(r)$, broadcast to all tokens of $y_i$; Dr. GRPO drops the $\mathrm{std}$ and the $1/|y|$.

Measured quantities, as instrumented in practice:
- **Clip fraction** $c=\frac{\\#\{t:\rho_t\notin[1-\epsilon_{\text{lo}},1+\epsilon_{\text{hi}}]\}}{\\#\{t\}}$ — typically $10^{-3}$–$10^{-1}$.
- **Staleness** $\tau$ = optimiser steps between the sampling of $y$ and the update using it; $\tau\in\{0,\dots,8\}$ in synchronous PPO, up to hundreds in async pipelines.
- **KL** estimated by the $k_3$ estimator $\rho-1-\log\rho$ on sampled tokens, not computed exactly.
- **Entropy** $H=-\frac1{|y|}\sum_t\sum_v \pi_\theta(v\mid\cdot)\log\pi_\theta(v\mid\cdot)$, full-vocabulary, per token.

Assumptions standard in the theory, and their status:
- *Exact or unbiased gradients* — violated: $G\in[4,64]$, and the $\mathrm{std}$ normaliser makes $A$ a biased, prompt-difficulty-dependent statistic.
- *Softmax tabular / linear-MDP parameterisation* — violated: transformer, shared parameters across all states.
- *On-policy sampling* ($\theta_{\mathrm{old}}=\theta$) — violated by design; clipping exists only because it is violated.
- *Bounded reward, bounded gradient, $L$-smoothness* — plausible but unverified; smoothness constants for LLM policy gradients have never been measured.
- *Fixed inference numerics* — violated: the sampler (vLLM/SGLang, fp16/bf16 kernels) and the trainer compute different $\log\pi_{\theta_{\mathrm{old}}}$ for the same tokens, so $\rho\neq1$ even at $\tau=0$.

## 3. State of the Art

**Theory SOTA (established).**
- Softmax policy gradient with exact gradients converges to the global optimum, at $O(1/t)$ for true PG and $O(e^{-ct})$ with entropy regularisation (Mei, Xiao, Szepesvári, Schuurmans, ICML 2020).
- NPG achieves $O(1/t)$ dimension-free convergence; PG-with-function-approximation gets $\sqrt{}$-style rates with an approximation-error floor (Agarwal, Kakade, Lee, Mahajan, JMLR 2021). None of this covers clipping.
- TRPO's monotonic-improvement lemma (Schulman et al., ICML 2015) bounds $J(\pi')-J(\pi)$ via a max-KL penalty. PPO's clip is a heuristic surrogate for that bound and does **not** inherit it.
- The one direct result: PPO-Clip global optimality via a hinge-loss reformulation with neural function approximation, under assumptions (regularity of the clipping-induced update, overparameterised networks) that no LLM run satisfies — Huang, Hsieh, Ho, Wu, AAAI 2024.

**Empirical SOTA (claimed, mostly unablated).**
- GRPO (Shao et al., DeepSeekMath, 2024) — critic-free, group-normalised. No convergence claim.
- DAPO (Yu et al., 2025): decoupled clip ($\epsilon_{\text{lo}}=0.2,\epsilon_{\text{hi}}=0.28$), dynamic sampling, token-level loss. Reports AIME'24 avg@32 $=50$ on Qwen2.5-32B. This is a **benchmark number**; the clip-higher change is not isolated from dynamic sampling in a controlled ablation at fixed compute.
- Dr. GRPO (Liu et al., 2025) identifies the $1/|y|$ and $\mathrm{std}$ terms as optimisation biases that inflate wrong-answer length; the fix is argued from the estimator's algebra plus small-scale runs, not proved to change the fixed point.
- Sequence-level ratios (GSPO, Qwen 2025; CISPO in MiniMax-M1) claim stability for MoE training where token ratios explode. Claimed, not independently reproduced at matched compute.

## 4. What Is Known

- **Clipping does not enforce a trust region.** Measured on MuJoCo/Atari-scale PPO: the ratio routinely leaves $[1-\epsilon,1+\epsilon]$ during the inner epochs because clipping zeroes the gradient only for samples on the wrong side of the advantage sign; total-variation and KL between $\pi_\theta$ and $\pi_{\theta_{\text{old}}}$ grow past the nominal bound (Ilyas et al., ICLR 2020; Hsu, Mendler-Dünner, Hardt, NeurIPS 2020 workshop line of work; Wang et al., "Truly PPO", UAI 2019).
- **Most of PPO's reported gain is implementation, not the clip.** Engstrom et al. (ICLR 2020) show advantage normalisation, value clipping, orthogonal init and LR annealing account for the bulk of the PPO-over-TRPO gap on continuous control.
- **The clipped objective is discontinuous in $\theta$-space in its gradient** and is not the gradient of any smooth function; PPO's update is a *pseudo-gradient*.
- **LLM-scale regularities.** Single-epoch, near-on-policy RLVR runs ($\tau\le1$) with $G=8$–$16$ show clip fractions under $1\%$ and monotone reward on math benchmarks over $\sim$500 steps; entropy falls from $\sim0.6$ to $<0.1$ nats/token within a few hundred steps on 7B models, after which pass@$k$ for large $k$ stops improving or degrades. Reported repeatedly across the 2025 open-recipe literature (Qwen2.5-7B/32B, Llama-3.1-8B class).
- **Zero-variance groups waste compute:** when all $G$ samples share the reward, $A_i=0$ and the prompt contributes nothing; DAPO reports this fraction growing large enough late in training to motivate resampling.

## 5. What Is Not Known

**Theoretically open.** No convergence theorem for clipped surrogates with (i) finite group size, (ii) $\tau>0$ staleness, (iii) multiple inner epochs, under any parameterisation matching an LLM. Not even the tabular softmax case with $\epsilon>0$ and stochastic $G$-sample advantages has a rate. It is unproven whether the clipped update's fixed points coincide with stationary points of $J$; the natural conjecture — that clipping introduces a bias $O(\epsilon\cdot c)$ in the stationary point — has no proof or counterexample.

**Empirically open.** Whether clip range and staleness trade off along a predictable frontier at 7B–70B. Runnable today: nobody has published a matched-compute sweep over $(\epsilon_{\text{hi}},\tau,G)$ with seeds.

**Methodologically blocked.** "Convergence" is not defined for these runs. Benchmark accuracy saturates while $\|\nabla J\|$ is unmeasured, entropy is still moving, and the reward is a proxy. There is no agreed stationarity certificate for a $10^{10}$-parameter policy under a non-differentiable surrogate.

## 6. Why It Is Hard

The binding obstruction is **non-identifiability of the failure mode from the observable trace**. When a run plateaus, at least four mechanisms produce the same curve: (a) clipped gradient mass — the informative samples are exactly the ones outside the trust region; (b) entropy collapse — $\pi_\theta$ is near-deterministic, so $\mathrm{std}(r)\to0$ and $A$ is undefined or amplified noise; (c) reward saturation on a finite verifier; (d) numerical mismatch between the inference and training log-probs making $\rho$ systematically biased. Each implies a different fix, and the standard telemetry (reward, KL, clip fraction) does not separate them: clip fraction rises under (a) and (d) alike, entropy falls under (a) and (b) alike.

Second: cost. One clean $(\epsilon,\tau,G)$ cell at 7B with 3 seeds is $\sim$2k–5k H100-hours dominated by generation. A $4\times3\times3$ grid with seeds is a 100k-GPU-hour experiment — affordable to perhaps twenty labs, and none has published it.

## 7. Current Research (as of 2026)

- **Sequence-level and length-normalised ratio design** — Qwen (GSPO), MiniMax (CISPO), motivated by MoE routing instability where token ratios diverge. *(frontier — verify)*
- **Asynchronous/off-policy RLVR** (AReaL-style pipelines, ByteDance/verl, open slime-class stacks): explicit staleness bounds plus decoupled PPO objectives. The theory here is the nearest live target — staleness is a parameter a theorem could take. *(frontier — verify)*
- **Entropy control** as a substitute for a trust region: clip-higher, entropy bonuses, covariance-based selective updates. Several 2025–26 papers argue entropy collapse, not clipping, is the true limit on RLVR gains.
- **Estimator debiasing**: Dr. GRPO and successors; unbiased truncated importance weighting borrowed from off-policy evaluation (Ionides-style truncation, doubly-robust corrections).
- **Theory groups** (Hsieh at NYCU; Agarwal/Kakade lineage; mirror-descent RL, e.g. Tomar et al. MDPO, ICLR 2022) continue on clipping-as-mirror-descent, still under tabular/linear assumptions.

## 8. Concrete Next Experiment

**Question.** Does clipping bias the fixed point, or only the transient?

**Scale.** Qwen2.5-Math-7B, RLVR on a fixed 40k-problem math set with an exact-match verifier, $G=16$, 4096-token generations, 600 optimiser steps, 3 seeds. Roughly 2.5k H100-hours per arm, 5 arms + control = $\sim$15k H100-hours.

**Arms.** $\epsilon_{\text{hi}}\in\{0.2,0.28,0.5\}$ at $\tau=1$; $\epsilon_{\text{hi}}=0.28$ at $\tau\in\{4,16\}$.

**Control arm.** Fully on-policy, single inner epoch, $\tau=0$, *no clipping* (plain REINFORCE with group-mean baseline, no $\mathrm{std}$, no $1/|y|$) — with the inference and training log-prob mismatch removed by recomputing $\log\pi_{\theta_{\text{old}}}$ in the trainer. This arm has an unbiased gradient by construction; every other arm's deviation from it is attributable to clipping or staleness.

**Deciding number.** The **terminal policy gap** $\Delta = J_{\text{ctrl}}(\theta_\infty) - J_{\text{arm}}(\theta_\infty)$, where $J$ is held-out verifier accuracy at temperature 1.0 averaged over 32 samples on 1000 unseen problems, measured after both arms' entropy has been *held fixed* by an entropy-target controller (so mechanism (b) is pinned). If $|\Delta| < 1$ point for all arms while transient reward curves differ by $>5$ points, clipping is a transient-speed knob and the fixed point is clip-invariant. If $\Delta$ grows monotonically with clip fraction $c$ — specifically if a regression of $\Delta$ on mean $c$ has slope significantly $>0$ across 15 runs — clipping biases the optimum and the theory must target the biased fixed point, not $J$.

## 9. Key References

- **[Foundational]** J. Schulman, S. Levine, P. Moritz, M. Jordan, P. Abbeel. *Trust Region Policy Optimization.* ICML, 2015. — arXiv:1502.05477
- **[Foundational]** J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* 2017. — arXiv:1707.06347
- **[Foundational]** A. Agarwal, S. M. Kakade, J. D. Lee, G. Mahajan. *On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift.* JMLR 22, 2021.
- **[Foundational]** J. Mei, C. Xiao, C. Szepesvári, D. Schuurmans. *On the Global Convergence Rates of Softmax Policy Gradient Methods.* ICML, 2020.
- **[SOTA-theory]** N.-C. Huang, P.-C. Hsieh, K.-H. Ho, I.-C. Wu. *PPO-Clip Attains Global Optimality: Towards Deeper Understandings of Clipping.* AAAI, 2024.
- **[SOTA-empirical]** Z. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300 (introduces GRPO)
- **[SOTA-empirical]** Q. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[SOTA-empirical]** Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, M. Lin. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. — arXiv:2503.20783 (Dr. GRPO)
- **[Diagnostic]** A. Ilyas, L. Engstrom, S. Santurkar, D. Tsipras, F. Janoos, L. Rudolph, A. Mądry. *A Closer Look at Deep Policy Gradients.* ICLR, 2020.
- **[Diagnostic]** L. Engstrom, A. Ilyas, S. Santurkar, D. Tsipras, F. Janoos, L. Rudolph, A. Mądry. *Implementation Matters in Deep RL: A Case Study on PPO and TRPO.* ICLR, 2020.
- **[Related]** Y. Wang, H. He, X. Tan. *Truly Proximal Policy Optimization.* UAI, 2019.
- **[Related]** M. Tomar, L. Shani, Y. Efroni, M. Ghavamzadeh. *Mirror Descent Policy Optimization.* ICLR, 2022.

## 10. Worked Example

Two-token bandit; $\mathcal{V}=\{a,b\}$, softmax logits $\theta=(\theta_a,\theta_b)$, reward $r(a)=1$, $r(b)=0$, $G=2$, $\epsilon=0.2$, no KL term.

Start at $\pi_{\theta_{\text{old}}}(a)=0.5$. A group of two draws $\{a,b\}$ gives raw rewards $(1,0)$, mean $0.5$, std $0.5$, so $A_a=+1$, $A_b=-1$.

- Un-clipped gradient at $\theta=\theta_{\text{old}}$: $\nabla_\theta L = A_a\pi(b)\,e_a$-direction $=1\cdot0.5=0.5$ on the $a$-logit gap.
- After one step the ratio for $a$ is $\rho_a=\pi_\theta(a)/0.5$. Clipping at $1+\epsilon$ freezes the objective once $\pi_\theta(a)>0.6$. In the same minibatch, the $b$ sample has $A_b<0$, and $\min$ takes the *unclipped* branch for $\rho_b<1$ — so the negative-advantage term keeps pushing $\pi_\theta(b)$ down without limit within the epoch.

Run the inner loop 4 epochs at $\eta$ chosen so each epoch moves the logit gap by $0.3$: $\pi_\theta(a)$ goes $0.5\to0.57\to0.63\to0.69\to0.74$. Nominal trust region: $\pi(a)\le0.6$. Achieved: $0.74$. Measured $\mathrm{KL}(\pi_\theta\|\pi_{\theta_{\text{old}}})=0.74\log\frac{0.74}{0.5}+0.26\log\frac{0.26}{0.5}=0.290-0.170=0.120$ nats — versus $0.020$ nats at the nominal edge, a **6× overshoot** of the trust region that the clip was supposed to impose.

Now the group statistic bites. At $\pi(a)=0.74$, the chance both samples are $a$ is $0.55$; then $\mathrm{std}(r)=0$, $A$ is $0/0$, and every implementation substitutes $A=0$ (or divides by $\mathrm{std}+10^{-4}$ and emits an advantage of magnitude $\sim10^4$, which the clip does *not* bound — the clip bounds $\rho$, not $A$).

The obstruction is visible in three lines: the clip failed to bound the policy change, the bound it did impose was asymmetric in the sign of $A$, and the estimator became undefined exactly where the policy was improving fastest. A convergence theorem must handle all three at once. None does.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*