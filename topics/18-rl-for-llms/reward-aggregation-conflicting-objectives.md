---
id: 18-rl-for-llms/reward-aggregation-conflicting-objectives
title: "Reward Aggregation Across Conflicting Objectives"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Reward Aggregation Across Conflicting Objectives

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/reward-aggregation-conflicting-objectives` · **Status:** open

## 1. Problem Statement

RLHF pipelines optimize one scalar. Real deployments care about several things that trade off: helpfulness, harmlessness, factuality, verbosity, instruction adherence, latency-proxied brevity. The standard fix is to train $K$ reward models and combine them, almost always by a weighted sum. The problem is that no principled procedure exists for choosing the combination, and the usual choice is not even well-typed.

Three variants, with different difficulty:

- **Measurement.** Given $K$ learned reward models, are their outputs on a common scale, so that "$+1$ helpfulness $= -1$ harmlessness" means anything? Currently no. Bradley–Terry rewards are identified only up to a per-prompt shift and a global scale set by annotator noise, not by any fact about the objectives.
- **Method.** Find an aggregation rule $A: \mathbb{R}^K \to \mathbb{R}$ (or a constrained program) plus a training procedure that reaches a chosen operating point on the true Pareto front, reproducibly, at a fixed KL budget. Solving means: specify a target trade-off in user-facing units and hit it within tolerance.
- **Theory.** Characterize which aggregation rules are compatible with which axioms (Pareto efficiency, independence, non-dictatorship, scale invariance) when the $K$ objectives come from distinct annotator populations. Arrow-style impossibility applies; the open part is what the LLM-specific escape route is.

## 2. Formal Setting

Prompts $x \sim \mathcal{D}$, responses $y \in \mathcal{Y}$, policy $\pi_\theta(y \mid x)$, reference $\pi_{\mathrm{ref}}$. Vector reward $\mathbf{r}(x,y) = (r_1, \dots, r_K) \in \mathbb{R}^K$.

**How each $r_k$ is measured.** Objective $k$ has a preference dataset $\mathcal{D}_k = \{(x, y^+, y^-)\}$ from annotator pool $k$. Fit by Bradley–Terry maximum likelihood:

$$\Pr(y^+ \succ_k y^- \mid x) = \sigma\!\big(r_k(x,y^+) - r_k(x,y^-)\big), \qquad \sigma(z) = (1+e^{-z})^{-1}.$$

The fitted $r_k$ is invariant to $r_k \mapsto c_k r_k + b_k(x)$ **only** for $c_k = 1$; the scale $c_k$ is pinned by the empirical annotator agreement rate on $\mathcal{D}_k$, i.e. by label noise. The additive $b_k(x)$ is entirely free.

**Aggregation and training.** Linear scalarization with weights $w \in \Delta^{K-1}$:

$$\max_\theta \ \mathbb{E}_{x,\,y\sim\pi_\theta}\Big[\textstyle\sum_k w_k r_k(x,y)\Big] - \beta \, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big).$$

The constrained alternative (Safe RLHF, constrained RLHF) maximizes $r_1$ subject to $\mathbb{E}[r_k] \ge \tau_k$ via a Lagrangian with learned multipliers $\lambda_k$.

**Front.** $\mathcal{P} = \{\mathbf{v} \in \mathbb{R}^K : \exists \theta,\ \mathbf{v} = \mathbb{E}[\mathbf{r}(x,y)],\ \nexists \theta' \text{ dominating}\}$, measured empirically as a set of $(\text{objective-}k \text{ win rate vs. } \pi_{\mathrm{ref}})_{k=1}^K$ tuples at matched $\mathrm{KL}$.

**Assumptions, and which are violated.**

| Assumption | Status |
|---|---|
| $r_k$ commensurable across $k$ | **Violated.** Scales are set by annotator noise per dataset. |
| BT preferences transitive, single latent utility per pool | **Violated.** Annotator disagreement is structured, not i.i.d. noise (Siththaranjan et al. 2024). |
| Pareto front convex, so every point is reachable by some $w$ | **Unverified for LLMs.** Known false in general MORL. |
| $r_k$ valid off the training distribution | **Violated.** All $r_k$ degrade under optimization pressure (Gao et al. 2023). |
| Objectives independent | **Violated.** Verbosity correlates with helpfulness scores across essentially all public RMs. |

## 3. State of the Art

**Established (ablated, reproduced).**
- Separate RMs beat one merged RM when objectives conflict. Llama 2 (Touvron et al., 2023) trains helpfulness and safety RMs separately and combines them with a *piecewise switch*, not a weighted sum — an admission that no weight worked.
- Weight-space interpolation gives a cheap Pareto front. Rewarded Soups (Ramé et al., NeurIPS 2023) fine-tunes one model per reward and linearly interpolates parameters; the interpolated front dominates the front from interpolating rewards, at ~$1/N$ the training cost.
- Fine-grained, densely decomposed rewards outperform a single holistic reward on long-form generation (Wu et al., NeurIPS 2023), with per-objective ablations.
- Constrained optimization beats fixed scalarization for safety. Safe RLHF (Dai et al., ICLR 2024) and Constrained RLHF (Moskovitz et al., ICLR 2024) both show a learned $\lambda$ tracks a target constraint where a fixed $w$ drifts.

**Claimed but under-ablated.**
- Conditioning on the desired trade-off in-context — Rewards-in-Context (Yang et al., ICML 2024), Directional Preference Alignment (Wang et al., ACL 2024), Controllable Preference Optimization (Guo et al., EMNLP 2024), Panacea (Zhong et al., NeurIPS 2024). All report Pareto fronts; none isolate whether the gain comes from the conditioning mechanism or from the extra multi-objective data.
- **Benchmark-number-only.** ArmoRM (Wang et al., EMNLP 2024) reports 90.4 on RewardBench with a 19-dimensional reward and a gating mixture-of-experts. RewardBench measures ranking accuracy on held-out pairs, not downstream trade-off quality after RL. No published run takes ArmoRM's gating weights through PPO and measures the resulting front.

**Theory SOTA.** MaxMin-RLHF (Chakraborty et al., ICML 2024) proves a single BT reward cannot represent a mixture of divergent user utilities and gives a non-vanishing alignment gap; it proposes egalitarian max-min aggregation instead. Ge et al. (NeurIPS 2024) import social-choice axioms into RLHF and show standard pipelines fail several.

## 4. What Is Known

- **Overoptimization is quantitative and per-objective.** Gao, Schulman & Hilton (ICML 2023): with $d = \sqrt{\mathrm{KL}}$, gold reward follows $d(\alpha - \beta \ln d)$ for RL and $d(\alpha - \beta d)$ for best-of-$n$, measured with proxy RMs from 3M to 3B parameters against a 6B gold RM. $\alpha, \beta$ differ per objective, so under a fixed $w$ the *effective* trade-off drifts as KL grows even though $w$ is constant.
- **BT under hidden context implements Borda count** (Siththaranjan, Laidlaw & Hadfield-Menell, ICLR 2024) — a specific, known-flawed voting rule, not utilitarian aggregation. Demonstrated on synthetic and on Anthropic HH data.
- **Linear scalarization reaches only the convex hull** of the Pareto front. Standard MORL result (Roijers et al., JAIR 2013; Vamplew et al., JAAMAS 2022). Any concave region is unreachable by *any* $w$.
- **The tension is scale-dependent.** Bai et al. (2022, Anthropic HH) found helpfulness and harmlessness trade off sharply at small model scale and much less at 52B after RLHF on combined data.
- **Ensembling mitigates but does not remove hacking.** Coste et al. (ICLR 2024) show ensembles help; Eisenstein et al. (COLM 2024) show all members of a pretrain-seed ensemble can hack together — measured up to 8B.

## 5. What Is Not Known

- **Methodologically blocked:** whether $r_1$ and $r_2$ units are comparable at all. There is no accepted normalization. Candidate fixes (z-scoring on a shared prompt set, quantile mapping, calibrating to human win-probability) have never been compared head to head with the resulting policies evaluated. Until this is fixed, "we used $w = (0.5, 0.5)$" is uninterpretable.
- **Empirically open:** is the LLM helpfulness–harmlessness front convex? Runnable today at 8B with a $w$-sweep versus a $\tau$-sweep; nobody has published the comparison at matched KL.
- **Theoretically open:** whether any aggregation rule satisfies Pareto efficiency, scale invariance, and non-dictatorship simultaneously when $r_k$ are learned from separate BT fits. Arrow's theorem does not directly apply (cardinal information is available), Harsanyi's theorem forces linearity but presumes interpersonal comparability the pipeline cannot supply. The gap between the two is unresolved for this setting.
- **Empirically open:** whether $w$ chosen offline on the RM front transfers to the post-RL policy front. Assumed everywhere, tested nowhere.

## 6. Why It Is Hard

**Non-identifiability of scale.** This is the specific obstruction. A BT fit determines $r_k$ up to a multiplicative constant tied to how noisy annotators were on dataset $k$. Two objectives labeled by pools with different agreement rates come out on different scales, and the difference is a property of the labeling process, not of the objectives. So $w_k$ has no units and cannot be elicited from a stakeholder — "weight safety twice as much" does not map to any $w$.

Two amplifiers: (i) the per-prompt free shift $b_k(x)$ means only *within-prompt* differences are meaningful, but scalarization sums across objectives whose shifts differ; (ii) the front is measured with the same RMs being optimized, so the reported trade-off is confounded with per-objective overoptimization rates — an evaluation that does not measure what it names.

## 7. Current Research (as of 2026)

- **Constrained over scalarized.** Lagrangian/threshold formulations are the direction of travel for safety, following Safe RLHF and constrained RLHF; thresholds are elicitable in a way weights are not.
- **Social choice for alignment.** Conitzer et al. (ICML 2024 position) and the follow-on axiomatic line; the open question is which rule survives when votes are learned reward models rather than ballots.
- **Interpretable multi-attribute RMs.** HelpSteer2 / ArmoRM (NVIDIA) push per-attribute regression rewards with explicit gating *(frontier — verify whether gating weights have been validated post-RL rather than on RewardBench)*.
- **Game-theoretic alternatives that sidestep aggregation.** Nash Learning from Human Feedback (Munos et al., ICML 2024) and SPO (Swamy et al., ICML 2024) work with preferences directly, avoiding a scalar reward — the multi-objective extension is largely unexplored *(frontier — verify)*.
- **Personalization as $K$-of-one.** Per-user trade-off conditioning is the commercial pull; the research risk is that it hides the aggregation problem inside a prompt.

## 8. Concrete Next Experiment

**Question: is the helpfulness–harmlessness front convex, and does raw-scale scalarization sit on it?**

- **Scale.** One 8B instruct policy. Two RMs of the same architecture, trained on Anthropic HH helpfulness and harmlessness splits. PPO, KL matched to $\mathrm{KL} = 30$ nats for every arm. ~13 runs, single 8×H100 node-week.
- **Arms.** (a) $w$-sweep: $w_{\text{harm}} \in \{0, 0.1, \dots, 1.0\}$ on raw RM outputs. (b) **Control arm 1:** identical sweep after z-scoring each RM on a common 10k-prompt sample from $\pi_{\mathrm{ref}}$. (c) **Control arm 2:** Lagrangian sweep, maximize helpfulness s.t. $\mathbb{E}[r_{\text{harm}}] \ge \tau$ for 6 values of $\tau$ spanning the same range.
- **Evaluation.** Held-out *gold* judges, not the training RMs: human or a frozen larger model, reporting win rate vs. $\pi_{\mathrm{ref}}$ per objective.
- **Deciding number.** $G = \max_\tau \big[\text{helpfulness win rate (arm c)} - \text{helpfulness win rate (best arm a/b point at the same harmlessness win rate)}\big]$. If $G > 3$ points (above the $\pm 1.5$-point bootstrap CI at $n=2000$ comparisons), the front is concave in that region and linear scalarization is provably leaving policies unreachable — the field should move to constrained formulations. If $G \le 3$, scalarization is adequate and the open problem collapses to choosing $w$, i.e. to the normalization question, which arm (b) minus arm (a) then quantifies directly.

## 9. Key References

- **[Foundational]** Bai, Y. et al. *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* Anthropic, 2022. — arXiv:2204.05862
- **[Foundational]** Gao, L., Schulman, J., Hilton, J. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[Foundational]** Harsanyi, J. *Cardinal Welfare, Individualistic Ethics, and Interpersonal Comparisons of Utility.* Journal of Political Economy, 1955.
- **[Theory]** Chakraborty, S. et al. *MaxMin-RLHF: Alignment with Diverse Human Preferences.* ICML, 2024. — arXiv:2402.08925
- **[Theory]** Siththaranjan, A., Laidlaw, C., Hadfield-Menell, D. *Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF.* ICLR, 2024. — arXiv:2312.08358
- **[Theory]** Ge, L. et al. *Axioms for AI Alignment from Human Feedback.* NeurIPS, 2024. — arXiv:2405.14758
- **[SOTA]** Ramé, A. et al. *Rewarded Soups: Towards Pareto-Optimal Alignment by Interpolating Weights Fine-Tuned on Diverse Rewards.* NeurIPS, 2023. — arXiv:2306.04488
- **[SOTA]** Dai, J. et al. *Safe RLHF: Safe Reinforcement Learning from Human Feedback.* ICLR, 2024. — arXiv:2310.12773
- **[SOTA]** Moskovitz, T. et al. *Confronting Reward Model Overoptimization with Constrained RLHF.* ICLR, 2024. — arXiv:2310.04373
- **[SOTA]** Zhou, Z. et al. *Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization.* Findings of ACL, 2024. — arXiv:2310.03708
- **[SOTA]** Wang, H. et al. *Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts.* EMNLP, 2024. — arXiv:2406.12845
- **[Empirical]** Wu, Z. et al. *Fine-Grained Human Feedback Gives Better Rewards for Language Model Training.* NeurIPS, 2023. — arXiv:2306.01693
- **[Empirical]** Eisenstein, J. et al. *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking.* COLM, 2024. — arXiv:2312.09244
- **[Survey]** Roijers, D. et al. *A Survey of Multi-Objective Sequential Decision-Making.* JAIR, 2013.
- **[Position]** Conitzer, V. et al. *Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback.* ICML, 2024. — arXiv:2404.10271

## 10. Worked Example

Two RMs, both BT-fitted. Suppose the harmlessness pool agrees on 85% of pairs and the helpfulness pool on 70% — a realistic spread; helpfulness is the harder judgment.

Under BT, the fitted reward gap for a typical pair satisfies $\sigma(\Delta r) = p$, so:

$$\Delta r_{\text{harm}} = \mathrm{logit}(0.85) = 1.734, \qquad \Delta r_{\text{help}} = \mathrm{logit}(0.70) = 0.847.$$

The two RMs therefore emit reward differences on scales differing by a factor of $1.734 / 0.847 = 2.05$ for comparably-sized *perceptual* differences. Setting $w = (0.5, 0.5)$ — the choice an engineer reads as "equal weight" — implements an effective weighting of about $2:1$ in favor of harmlessness once rewards are expressed in units of human preference probability. To actually weight equally in those units you need $w \approx (0.67, 0.33)$.

Now make the obstruction visible. The gradient of the scalarized objective is $0.5\nabla r_{\text{help}} + 0.5\nabla r_{\text{harm}}$. Because the harmlessness RM has ~2× the output scale, it dominates the update wherever the two disagree. Run the same recipe with a re-annotated harmlessness set that happens to agree 75% instead of 85% — same objectives, same policy, more annotator disagreement — and the ratio falls to $1.100/0.847 = 1.30$. The operating point moves substantially, with $w$ unchanged and nothing about the *objectives* changed. The trade-off was set by annotator agreement rates, not by any decision anyone made.

Note what does **not** change: the Pareto front itself is invariant to rescaling. Only the map $w \mapsto \text{point on the front}$ moves. That is why the problem is not solved by picking a better $w$ by hand — the parameterization is unstable across data refreshes — and why the experiment in §8 targets reachability (is the front concave?) rather than weight tuning.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*