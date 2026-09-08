---
id: 18-rl-for-llms/entropy-collapse-policy-gradient
title: "Entropy Collapse in Policy Gradient Fine-Tuning"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Entropy Collapse in Policy Gradient Fine-Tuning

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/entropy-collapse-policy-gradient` · **Status:** open

## 1. Problem Statement

Policy-gradient fine-tuning of a language model — RLHF with a learned reward model, or RLVR with a binary verifier — reliably drives the policy's output entropy toward zero within a few hundred optimizer steps. Sampling temperature stops mattering, rollouts within a group become near-identical, the gradient signal vanishes, and the run plateaus long before the compute budget is spent.

Three distinct variants:

- **Measurement.** Is the entropy that collapses the quantity that actually matters? Token-level conditional entropy, sequence-level entropy, and behavioural diversity (distinct solution strategies, pass@$k$ at large $k$) are different objects that current work uses interchangeably. Which one predicts loss of headroom?
- **Method.** Find an intervention that holds entropy above a floor **and** raises the asymptotic reward, rather than trading pass@1 for pass@$k$ at a fixed frontier.
- **Theory.** Prove or refute: under a mean-zero-advantage policy gradient on a softmax policy with no explicit entropy term, is entropy collapse *necessary*, or an artifact of the estimator (clipping asymmetry, group baselines, off-policy staleness)?

A solution to the method variant is an algorithm that, at matched compute and matched pass@1, delivers strictly higher pass@$k$ at $k \geq 128$ *and* a higher reward ceiling than an entropy-bonus baseline, reproduced at ≥7B on two model families.

## 2. Formal Setting

Policy $\pi_\theta(y \mid x)$ over token sequences, prompts $x \sim \mathcal{D}$, vocabulary $\mathcal{V}$, verifier or reward model $r(x,y)$. Objective:

$$J(\theta) = \mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}\big[r(x,y)\big] - \beta\,\mathbb{E}_x\big[\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})\big].$$

**Entropy, as measured.** Almost all reported curves are the token-averaged conditional entropy over *on-policy sampled* prefixes at temperature 1:

$$\widehat{H}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{|y^{(i)}|}\sum_{t=1}^{|y^{(i)}|} \Big(-\sum_{v\in\mathcal{V}} \pi_\theta(v \mid x^{(i)}, y^{(i)}_{<t}) \log \pi_\theta(v \mid x^{(i)}, y^{(i)}_{<t})\Big),$$

with $y^{(i)} \sim \pi_\theta$. This is *not* the sequence entropy $-\mathbb{E}[\log \pi_\theta(y|x)]$, which is the sum (not mean) of the same terms and therefore confounds with response length. Reported "entropy" also changes if prefixes are teacher-forced from a fixed corpus, if sampling used $T \neq 1$ or top-$p < 1$, or if the sum is truncated to the top-$k$ logits — all four are common and rarely stated.

**Advantage.** GRPO-style: for a group $\{y^{(1)},\dots,y^{(G)}\}$ from one prompt, $A^{(i)} = (r^{(i)} - \mu_r)/\sigma_r$, so $\mathbb{E}[A] \approx 0$ within a group.

**Collapse dynamics.** The key identity (Cui et al., 2025): for a softmax policy under a small natural-gradient-like step of size $\eta$, the one-step change in entropy at a state is

$$\Delta H \approx -\eta\,\mathrm{Cov}_{a\sim\pi}\big(\log \pi(a),\, A(a)\big).$$

Entropy falls exactly when high-probability actions carry above-average advantage — the generic case once the policy is better than chance. Collapse is thus a *property of the correlation between confidence and correctness*, not of the loss's explicit terms.

**Assumptions known to be violated.** (i) $\mathbb{E}[A] = 0$ — false under length normalization, filtered groups, and asymmetric clipping. (ii) On-policy sampling — false with $\geq 2$ inner epochs or async rollout workers. (iii) Tabular softmax — false; parameter sharing across contexts means an entropy-raising update at one state can lower entropy at another. (iv) $r$ is the true objective — false under reward-model overoptimization.

## 3. State of the Art

**Empirical SOTA (established, ablated).**
- **Clip-higher** (DAPO; Yu et al., 2025) decouples the PPO clip bounds, $\epsilon_{\text{high}} > \epsilon_{\text{low}}$, so low-probability tokens can still gain mass. Ablated against symmetric clipping in the paper; entropy stays non-zero over the reported run. Widely reproduced.
- **Covariance-targeted clipping** — Clip-Cov and KL-Cov (Cui et al., 2025) suppress or KL-penalize the small fraction of tokens with the largest $\mathrm{Cov}(\log\pi, A)$. Reported gains: ~+2.0% average on Qwen2.5-7B and ~+6.4% on Qwen2.5-32B across math benchmarks versus GRPO, with larger AIME deltas. Ablated against the entropy-bonus baseline.

**Claimed but unablated / benchmark-only.**
- The exponential fit $R = -a\exp(H) + b$ relating validation reward to policy entropy, used to *predict* a run's ceiling from its first ~200 steps. Fitted per-run on Qwen and Mistral families; the claim that $b - a$ is a genuine ceiling rather than a description of the observed trajectory has not been tested by an intervention that changes $H$ mid-run.
- **Entropy bonus** ($+\lambda H$ in the loss). Standard, and it works in the sense that it holds entropy up — but no published ablation shows a $\lambda$ that raises the reward ceiling rather than trading it. Practitioners report a narrow band between "no effect" and "entropy explosion."
- **Selective updates on high-entropy tokens** (Wang et al., 2025): updating only the top-20% highest-entropy tokens matches or beats full-token RLVR on Qwen2.5-32B. Benchmark numbers; mechanism unablated.
- **KL-reset / reference-policy refresh** (ProRL, Liu et al., 2025) to sustain thousand-step runs. Reported as a systems recipe; contribution of the reset alone is not isolated.

**Theory SOTA.** Mei et al. (ICML 2020): true-gradient softmax PG converges at $O(1/t)$; *entropy-regularized* softmax PG converges linearly, $O(e^{-ct})$. Agarwal et al. (JMLR 2021) give global-convergence and approximation results for softmax and NPG. Both concern tabular/exact-gradient settings and say nothing about entropy under stochastic, clipped, group-baselined estimators on shared parameters.

## 4. What Is Known

- Collapse is fast and monotone. On Qwen2.5-7B / Qwen2.5-32B under GRPO on math corpora, token entropy falls from roughly $0.4$–$0.6$ nats to under $0.1$ nats within ~200 steps; most downstream benchmark gain accrues in that same window (Cui et al., 2025).
- The covariance identity holds empirically: measured $\mathrm{Cov}(\log\pi, A)$ is positive and large early, then decays to near zero as entropy floors — consistent with a self-terminating process rather than an external constraint.
- Entropy loss is heavy-tailed in tokens. A small minority of tokens (branch points: "wait", "alternatively", connective and decision tokens) carry most of the entropy; ~20% of tokens by entropy rank suffice to reproduce full-token RLVR performance at 32B (Wang et al., 2025).
- RL narrows the sampling frontier. Yue et al. (2025) report that RLVR-trained models beat base models at pass@1 but are matched or overtaken at large $k$ (e.g. $k = 256$) on math and code benchmarks at 7B–32B — RL sharpens the base model's distribution rather than adding solutions.
- Clip asymmetry is causal, not incidental: raising $\epsilon_{\text{high}}$ alone measurably slows collapse at fixed data and seed (DAPO ablation, 32B).

## 5. What Is Not Known

- **Theoretically open.** Whether collapse is *necessary* for any mean-zero-advantage PG on a softmax-parameterized LM with bounded reward, or is contingent on the estimator. No theorem either way outside the tabular exact-gradient case. Also open: whether an entropy floor is compatible with convergence to the optimal deterministic-verifier policy at all.
- **Empirically open.** Whether $b - a$ in the exponential fit is an intervenable ceiling. The experiment — force entropy up at step 150 by a mechanism orthogonal to reward, then compare asymptotic reward against a matched control — is runnable at 7B on a few thousand GPU-hours and has not been published.
- **Methodologically blocked.** What "diversity worth preserving" means. Token entropy, sequence entropy, distinct-solution count, and pass@$k$ dissociate: a policy can hold token entropy at 0.3 nats by hedging on formatting tokens while emitting one reasoning strategy. No accepted measure of *strategy-level* diversity exists, so "the intervention preserved diversity" is currently unfalsifiable.

## 6. Why It Is Hard

The obstruction is **non-identifiability between two mechanisms with the same signature**. Entropy falling and reward plateauing co-occur under (A) the policy exhausting its reachable solution set — collapse is the *consequence* of convergence — and (B) premature commitment destroying gradient signal — collapse is the *cause* of the plateau. Both produce identical entropy and reward curves. Distinguishing them requires an intervention on entropy that does not itself alter the reward landscape, and every available lever (entropy bonus, clip bounds, KL coefficient, temperature) changes the objective or the update distribution as well.

Compounding it: pass@$k$ at $k \geq 128$ is the natural discriminator, and it costs $128\times$ the inference of pass@1 per checkpoint, which is why most papers report $k \leq 16$ — a range where the two mechanisms are indistinguishable.

## 7. Current Research (as of 2026)

- **Covariance-targeted regularization** — Clip-Cov / KL-Cov lines from Shanghai AI Lab and Tsinghua collaborators; extensions that gate on per-token covariance rather than entropy rank *(frontier — verify)*.
- **Token-selective updates** — Alibaba Qwen and academic groups following the high-entropy-minority-token result; open question is whether the 20% set is stable across checkpoints.
- **Long-horizon RL recipes** — NVIDIA (ProRL) and frontier labs sustaining $10^3$+ step runs via reference-policy resets and periodic data refresh; entropy management is a stated component but not separately ablated *(frontier — verify)*.
- **Diversity-aware objectives** — pass@$k$-as-reward and determinantal/coverage rewards; early results, no independent reproduction *(frontier — verify)*.
- **Theory** — extensions of the softmax-PG convergence literature to clipped surrogate objectives with group baselines; no published theorem covering the LLM setting.

## 8. Concrete Next Experiment

**Question decided:** is the entropy floor causal for the reward ceiling, or merely correlated with it?

- **Scale.** Qwen2.5-Math-7B and Llama-3.1-8B (two families, to rule out a Qwen-specific effect), GRPO on DAPO-Math-17k, 400 steps, group size 16, 3 seeds. ~2,000 A100-hours per family including evaluation.
- **Arms.** (1) Control: vanilla GRPO, no entropy term. (2) Entropy bonus $\lambda$ tuned to hold $\widehat{H} \geq 0.25$ nats. (3) Clip-higher, $\epsilon_{\text{high}} = 0.28$. (4) **Late intervention:** run vanilla to step 150 (entropy already $< 0.1$ nats), then switch on the arm-2 bonus for steps 150–400. Arm 4 is the discriminator: mechanism (A) predicts it recovers nothing, mechanism (B) predicts it recovers a measurable fraction of the arm-2/3 gap.
- **Deciding number.** **pass@256 on AIME-2025 at step 400, at matched pass@1 (± 1 point) between arms.** If arms 2–4 exceed the control by $\geq 5$ points pass@256 with non-overlapping seed ranges, the floor is causal and the exponential-fit ceiling is intervenable. If arm 4 lands within 1 point of control while arm 2 exceeds it, collapse is irreversible past step 150 — which relocates the problem to *initialization and early-step* entropy management.
- Report $\widehat{H}$, sequence entropy, and distinct-solution counts (judge-clustered, $n=64$) at every checkpoint, so the measurement variant gets data too.

## 9. Key References

- **[Foundational]** Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, Irving. *Fine-Tuning Language Models from Human Preferences.* arXiv, 2019. — arXiv:1909.08593
- **[Foundational]** Ouyang, Wu, Jiang, et al. *Training language models to follow instructions with human feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[Foundational]** Mei, Xiao, Szepesvári, Schuurmans. *On the Global Convergence Rates of Softmax Policy Gradient Methods.* ICML, 2020. — arXiv:2005.06392
- **[Foundational]** Agarwal, Kakade, Lee, Mahajan. *On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift.* JMLR 22, 2021. — arXiv:1908.00261
- **[Foundational]** Shao, Wang, Zhu, et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv, 2024 (GRPO). — arXiv:2402.03300
- **[SOTA]** Cui, Zhang, Zhang, et al. *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* arXiv, 2025 (Clip-Cov, KL-Cov, $R = -a e^{H} + b$). — arXiv:2505.22617
- **[SOTA]** Yu, Zhang, Wang, et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv, 2025 (clip-higher). — arXiv:2503.14476
- **[SOTA]** Wang, Yu, He, et al. *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.* arXiv, 2025. — arXiv:2506.01939
- **[Counterpoint]** Yue, Chen, Lu, et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv, 2025. — arXiv:2504.13837
- **[Related]** Haarnoja, Zhou, Abbeel, Levine. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor.* ICML, 2018. — arXiv:1801.01290

## 10. Worked Example

One branch point, carried through.

A model at a reasoning fork has three continuations with $\pi = (0.7, 0.2, 0.1)$, so $\log\pi = (-0.357, -1.609, -2.303)$ and $H = 0.802$ nats. GRPO group returns mean-zero advantages $A = (+0.2, -0.3, -0.8)$ (check: $0.7(0.2) + 0.2(-0.3) + 0.1(-0.8) = 0$).

$$\mathrm{Cov}(\log\pi, A) = 0.7(-0.357)(0.2) + 0.2(-1.609)(-0.3) + 0.1(-2.303)(-0.8) = 0.231.$$

With effective step $\eta = 0.01$, $\Delta H \approx -0.0023$ nats/step. The sign is forced: the *already-likely* branch is the correct one, so every step deepens it. Nothing in the loss asked for this.

Now the ceiling question. Take an illustrative fit with $a = 0.25$, $b = 1.00$ (coefficients are fitted per run; these are of the reported order). At the run's start, $H = 0.45 \Rightarrow R = 1.00 - 0.25 e^{0.45} = 0.608$. At the floor, $H = 0.05 \Rightarrow R = 0.737$. The asymptote is $b - a = 0.750$.

So at step 200 the run has already banked 0.737 of an available 0.750 — **1.3 points of headroom left, from 12.9 points of total gain.** 90% of the achievable improvement was spent in the first fifth of the schedule.

Here is the obstruction, visible: the fit says the run is finished, but it cannot say *why*. If the intervenable object is $H$, then pushing entropy back to 0.45 buys another 12.9 points. If the fit is just a reparameterization of the convergence trajectory, then $b - a$ is fixed by the base model's reachable solution set and raising $H$ moves you *backward along the same curve* — costing pass@1 and buying nothing. Both readings fit every published entropy-versus-reward plot identically. Only arm 4 of §8 separates them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*