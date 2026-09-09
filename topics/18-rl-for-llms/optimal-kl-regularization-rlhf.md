---
id: 18-rl-for-llms/optimal-kl-regularization-rlhf
title: "Optimal KL Regularization Strength in RLHF"
topic: 18-rl-for-llms
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal KL Regularization Strength in RLHF

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/optimal-kl-regularization-rlhf` · **Status:** empirically-open

## 1. Problem Statement

RLHF optimizes a learned reward model $\hat r$ while penalizing divergence from a reference policy $\pi_{\mathrm{ref}}$ (usually the SFT checkpoint). The penalty weight $\beta$ — or equivalently a target KL budget — is a free scalar that every practitioner sets by hand. Too small and the policy exploits errors in $\hat r$ (reward hacking, mode collapse, degenerate formatting); too large and the policy barely moves.

Three distinct variants, routinely conflated:

- **Measurement.** Given a trained policy, report the divergence from $\pi_{\mathrm{ref}}$ that actually constrained it. Open because sequence-level KL is estimated, not computed, and its unit (per token vs. per sequence) is not standardized.
- **Method.** Given a task, reward model, and compute budget, *predict* $\beta^\star$ before training rather than sweeping. Solving it means a rule $\beta^\star = f(N_{\mathrm{RM}}, N_{\pi}, |\mathcal{D}_{\mathrm{pref}}|, \text{task})$ whose predictions match a sweep to within the noise of the sweep.
- **Theory.** Prove that the KL-optimal operating point is characterized by a computable quantity — e.g. the curvature of reward-model error along the policy's own distribution shift — rather than only fit post hoc.

A solution to the method variant would eliminate the $\beta$ sweep, currently the single most expensive hyperparameter in the RLHF pipeline.

## 2. Formal Setting

Prompts $x \sim \mathcal{D}$, responses $y = (y_1,\dots,y_T)$ generated autoregressively. The KL-regularized objective:

$$J(\pi) = \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi(\cdot|x)}\big[\hat r(x,y)\big] - \beta\, \mathbb{E}_x\big[ D_{\mathrm{KL}}\!\left(\pi(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\right)\big].$$

Its exact maximizer is the Gibbs tilt $\pi^\star(y|x) \propto \pi_{\mathrm{ref}}(y|x)\exp(\hat r(x,y)/\beta)$ (Ziegler et al. 2019; Korbak et al. 2022 recast this as Bayesian conditioning). Equivalently, a constrained form: maximize $\mathbb{E}[\hat r]$ subject to $\mathbb{E}_x D_{\mathrm{KL}} \le \kappa$ nats, with $\beta$ the Lagrange multiplier.

**How each quantity is actually measured.**

- **Sequence KL.** Not computed in closed form. Estimated on-policy from sampled $y$ as $\widehat{D}_{\mathrm{KL}} = \frac{1}{|B|}\sum_{y\in B}\sum_{t}\log\frac{\pi(y_t|x,y_{<t})}{\pi_{\mathrm{ref}}(y_t|x,y_{<t})}$ (the "k1" estimator: unbiased, high variance, can go negative). Many implementations instead use Schulman's k3 estimator $\mathbb{E}_{\pi}[\rho - \log\rho - 1]$ with $\rho = \pi_{\mathrm{ref}}/\pi$ — non-negative, lower variance, and *this is a different quantity being penalized in the loss*. GRPO (Shao et al. 2024) uses k3 as a loss term, not a reward term; PPO-RLHF conventionally uses k1 inside the per-token reward.
- **Units.** $\widehat{D}_{\mathrm{KL}}$ above is per *sequence*. Dividing by $T$ gives per-token KL. Since RLHF systematically changes $T$ (verbosity inflation), a fixed per-token $\beta$ is a *drifting* sequence-level constraint. Papers frequently do not say which they report.
- **Distance coordinate.** Gao et al. (2023) reparameterize by $d = \sqrt{D_{\mathrm{KL}}}$, in which reward curves are close to linear near the origin.
- **Gold reward.** $r^\star$ is unobservable. Proxied by a held-out larger RM, or by human/LLM-judge win rate against $\pi_{\mathrm{ref}}$.
- **Best-of-$n$ reference scale.** $D_{\mathrm{KL}}(\pi_{\text{BoN}}\|\pi_{\mathrm{ref}}) = \log n - \frac{n-1}{n}$ under continuous-reward/no-ties assumptions; Beirami et al. (2024) show this is an upper bound that is loose when duplicate samples occur.

**Assumptions known to be violated.** (i) $\hat r$ is accurate on-support — false off the SFT distribution, which is exactly where RL goes. (ii) Reverse KL is the right geometry — it is mode-seeking, so it prices diversity loss at nearly zero (Kirk et al. 2024). (iii) A single scalar $\beta$ suffices — reward error is prompt-dependent, so the optimal budget is per-prompt, not global. (iv) $\pi_{\mathrm{ref}}$ is fixed — iterative/online RLHF resets it, silently redefining $\kappa$.

## 3. State of the Art

**Established (ablated, reproduced).**

- *Square-root reward law.* Bai et al. (2022, Anthropic HH) found RM score grows roughly linearly in $\sqrt{D_{\mathrm{KL}}}$ across model scales — the empirical basis for the $d=\sqrt{\mathrm{KL}}$ coordinate.
- *Overoptimization scaling laws.* Gao, Schulman & Hilton (ICML 2023) fit, with $d=\sqrt{D_{\mathrm{KL}}}$: BoN gold score $\;d(\alpha_{\text{bon}} - \beta_{\text{bon}} d)$; RL gold score $\;d(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}}\log d)$. Coefficients vary smoothly with RM parameter count (3M–3B) and RM data size; the *policy* size shifts the curve's height but barely its peak location. This is the strongest existing handle on $\kappa^\star$ and is explicitly synthetic-gold-RM.
- *Adaptive KL control.* Ziegler et al. (2019) targeting a KL setpoint; still the default in TRL/OpenRLHF.

**Claimed but unablated.**

- *Dropping the KL term entirely* for RLVR (verifiable rewards): DAPO (Yu et al. 2025) removes it; Open-Reasoner-Zero and several 2025 reasoning runs follow. The claim that this is safe rests on rule-based rewards being unhackable, which is asserted rather than measured across tasks.
- *Reference-free DPO variants* (SimPO, Meng et al. NeurIPS 2024) — reported as win-rate benchmark numbers, with no paired study of what divergence budget replaced the removed one.
- *Per-prompt / constrained multiplier schemes* (Moskovitz et al., ICLR 2024) — correct in principle, demonstrated at small scale only.

**Benchmark-number-only results.** Nearly all published $\beta$ choices (PPO $\beta \approx 0.02$ in InstructGPT; DPO $\beta = 0.1$ default, $0.01$ in Zephyr) are reported as configuration lines with an AlpacaEval/MT-Bench number attached, not as sweeps with confidence intervals.

## 4. What Is Known

- Gold-vs-proxy divergence is real and reliably reproduced: in Gao et al. (2023), proxy RM score rises monotonically in $d$ while gold score peaks and falls, at policy scale 1.2B with RMs 3M–3B and gold RM 6B. The peak moves to larger $d$ as RM size grows, with $\beta_{\mathrm{RL}}$ shrinking roughly log-linearly in RM parameters.
- BoN and RL trace *different* gold-reward curves at equal KL; BoN is more KL-efficient at small budgets. Same study.
- Ties break the BoN KL formula: Beirami et al. (2024) show $\log n - (n-1)/n$ over-states the true divergence when the sampler repeats.
- KL-regularized MDPs enjoy error-averaging: Vieillard et al. (NeurIPS 2020) prove that KL regularization makes the performance gap depend on the *average* of approximation errors, not their sum — a theoretical reason regularization helps under noisy $\hat r$, at tabular/deep-RL scale, not LLM scale.
- With deterministic preferences, the DPO implicit KL constraint degenerates: Azar et al. (AISTATS 2024, IPO) show $\beta$ stops binding as the preference probability $\to 1$, so DPO's $\beta$ is not comparable to PPO's $\beta$.
- Direct alignment algorithms exhibit the same overoptimization shape (Rafailov et al. 2024), measured up to 7B policies.
- RLHF reduces output diversity at fixed benchmark score (Kirk et al., ICLR 2024) — a cost the reverse-KL penalty does not price.

## 5. What Is Not Known

- **Empirically open.** Whether Gao's coefficient fits extrapolate above ~1B policies with human-verified gold reward. Nobody has run the $\beta$ sweep with human raters as gold at 70B+. Runnable; costs a few hundred GPU-days plus annotation.
- **Empirically open.** Whether $\beta^\star$ is predictable from cheap pre-RL statistics — RM held-out accuracy, RM ensemble disagreement, or reward variance under $\pi_{\mathrm{ref}}$.
- **Theoretically open.** No proof characterizing $\kappa^\star$ from properties of $\hat r$'s error field. Xiong et al. (ICML 2024) give KL-constrained RLHF sample-complexity bounds under coverage conditions, but the bounds do not select $\beta$.
- **Theoretically open.** Whether reverse KL is optimal among $f$-divergences for this trade-off; $\chi^2$ and forward-KL alternatives (Wang et al., ICLR 2024; Go et al., ICML 2023) are explored, not settled.
- **Methodologically blocked.** Comparing $\beta$ across algorithms. PPO's $\beta$ (reward-shaping, k1, per-token), GRPO's $\beta$ (k3, loss term), and DPO's $\beta$ (implicit, sequence-level, non-binding under deterministic preferences) are not the same parameter. There is no agreed normalization, so "optimal $\beta$" is not yet a well-posed cross-method question.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth compounded by non-identifiability of the gold signal**. The quantity to be maximized, $\mathbb{E}[r^\star]$, is measured by a proxy that is itself a reward model or an LLM judge with the same failure modes as the model being optimized. When a larger gold RM is used, the measured $\kappa^\star$ is a property of the *gold-proxy pair*, not of the policy. When humans are used, the per-comparison cost makes the 6–8 point $\beta$ sweep needed to locate a peak — with enough samples per point to separate curves whose gold scores differ by ~1 win-rate point — cost more than the training itself.

Secondary, and independent: **confounded measurement**. Changing $\beta$ changes response length, which changes both the sequence KL and every length-biased judge. A reported $\beta^\star$ therefore mixes the true regularization optimum with a length-preference artifact.

## 7. Current Research (as of 2026)

- **RLVR without KL.** Removing the penalty for verifiable-reward tasks and relying on the task checker to bound hacking (DAPO, Yu et al. 2025; reasoning-RL groups at DeepSeek, Moonshot, ByteDance). *(frontier — verify whether entropy collapse, not reward hacking, becomes the binding failure.)*
- **Entropy-based control.** Replacing or supplementing the KL budget with a policy-entropy target, on the argument that entropy collapse is the observable that actually predicts run failure. *(frontier — verify.)*
- **Uncertainty-scaled penalties.** Setting $\beta$ per prompt from RM ensemble disagreement, following Coste et al. (ICLR 2024) and Eisenstein et al. (COLM 2024), who showed ensembles mitigate but do not eliminate hacking.
- **Divergence geometry.** $f$-divergence and forward-KL objectives to retain diversity (Wang et al. 2024; Go et al. 2023).
- **Open recipes.** Tülu 3 (Lambert et al. 2024) and OpenRLHF publish full configs, making sweeps reproducible for the first time at 8B–70B.

## 8. Concrete Next Experiment

**Question.** Does $\kappa^\star$ (the KL budget maximizing *human-judged* quality) follow the RM-size scaling predicted by Gao et al. at 8B policy scale?

**Scale.** Llama-3.1-8B-Instruct as $\pi_{\mathrm{ref}}$, on a fixed 20k-prompt helpfulness set. Two reward models trained on identical preference data at 1B and 8B. GRPO or PPO, 8 KL setpoints $\kappa \in \{1,2,4,8,16,32,64,128\}$ nats sequence-level, enforced by an adaptive controller (not a fixed $\beta$), 2 seeds each → 32 runs, roughly 1.5k H100-hours.

**Control arm.** Best-of-$n$ from $\pi_{\mathrm{ref}}$ with the *same* RMs at matched KL, using $n$ chosen so $\log n - (n-1)/n = \kappa$. BoN needs no training, so it isolates how much of the gold-score curve is the RM's error field versus the RL optimizer's exploitation of it.

**Deciding number.** $\Delta = \sqrt{\kappa^\star_{8\text{B RM}}} - \sqrt{\kappa^\star_{1\text{B RM}}}$, where $\kappa^\star$ is the arg-max of length-controlled human win rate against $\pi_{\mathrm{ref}}$ (400 comparisons per point, ±2.5 pp at 95%). Gao's fits predict $\Delta > 0$ and of order the RM-size log ratio. $\Delta \le 0$, or a peak that is flat within the ±2.5 pp band across a 128× range of $\kappa$, falsifies the transfer of RM-size scaling to human-gold at 8B — and means the sweep cannot be replaced by a rule.

## 9. Key References

- **[Foundational]** Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, Irving. *Fine-Tuning Language Models from Human Preferences.* 2019. — arXiv:1909.08593
- **[Foundational]** Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano. *Learning to Summarize with Human Feedback.* NeurIPS 2020. — arXiv:2009.01325
- **[Foundational]** Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022. — arXiv:2203.02155
- **[SOTA]** Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. — arXiv:2210.10760
- **[SOTA]** Bai et al. *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* 2022. — arXiv:2204.05862
- **[SOTA]** Rafailov, Sharma, Mitchell, Ermon, Manning, Finn. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. — arXiv:2305.18290
- **[Theory]** Azar, Guo, Piot, Munos, Rowland, Valko, Calandriello. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. — arXiv:2310.12036
- **[Theory]** Vieillard, Kozuno, Scherrer, Pietquin, Munos, Geist. *Leverage the Average: an Analysis of KL Regularization in Reinforcement Learning.* NeurIPS 2020.
- **[Theory]** Korbak, Perez, Buckley. *RL with KL Penalties is Better Viewed as Bayesian Inference.* Findings of EMNLP 2022. — arXiv:2205.11275
- **[Theory]** Xiong, Dong, Ye, Wang, Zhong, Jiang, Zhang. *Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint.* ICML 2024. — arXiv:2312.11456
- **[Theory]** Beirami, Agarwal, Berant, D'Amour, Eisenstein, Nagpal, Suresh. *Theoretical Guarantees on the Best-of-n Alignment Policy.* 2024. — arXiv:2401.01879
- **[Empirical]** Coste, Anwar, Kirk, Krueger. *Reward Model Ensembles Help Mitigate Overoptimization.* ICLR 2024. — arXiv:2310.02743
- **[Empirical]** Eisenstein et al. *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking.* COLM 2024. — arXiv:2312.09244
- **[Empirical]** Kirk, Mediratta, Nalmpantis, Luketina, Hambro, Grefenstette, Raileanu. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR 2024. — arXiv:2310.06452
- **[Empirical]** Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300
- **[Frontier]** Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[Survey]** Kaufmann, Weng, Bengs, Hüllermeier. *A Survey of Reinforcement Learning from Human Feedback.* 2023. — arXiv:2312.14925

## 10. Worked Example

Take the Gao et al. RL functional form for gold reward in the coordinate $d = \sqrt{D_{\mathrm{KL}}}$:

$$R_{\mathrm{gold}}(d) = d\,(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}}\log d).$$

Setting $R'_{\mathrm{gold}}(d) = \alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}}\log d - \beta_{\mathrm{RL}} = 0$ gives a closed form for the optimal budget:

$$d^\star = \exp\!\left(\frac{\alpha_{\mathrm{RL}}}{\beta_{\mathrm{RL}}} - 1\right), \qquad \kappa^\star = (d^\star)^2.$$

Illustrative coefficients of the magnitude Gao et al. fit: with $\alpha_{\mathrm{RL}} = 0.92$, $\beta_{\mathrm{RL}} = 0.30$, $d^\star = e^{2.07} = 7.9$, so $\kappa^\star \approx 63$ nats. Now grow the RM by 10×; their fits have $\beta_{\mathrm{RL}}$ falling roughly log-linearly, say to $0.26$. Then $\alpha/\beta - 1 = 2.54$, $d^\star = 12.7$, $\kappa^\star \approx 161$ nats.

**Where the obstruction becomes visible.** A 13% change in one fitted coefficient moved the recommended KL budget by 2.5×, because $\kappa^\star$ is exponential in $\alpha/\beta$. So $\kappa^\star$ is only as identified as $\beta_{\mathrm{RL}}$ is — and $\beta_{\mathrm{RL}}$ is estimated by regressing against a *synthetic gold RM*, not humans. Two further facts make this worse:

- Near the peak the curve is flat. At $\kappa = 63$ vs. $\kappa = 161$ with the second coefficient set, $R_{\mathrm{gold}}$ differs by under 3%. Distinguishing them with human raters at a realistic effect size needs on the order of $10^3$ comparisons per point — the peak is *statistically* hard to locate exactly where it is *practically* least costly to miss.
- The measurement itself moves. If mean response length rises from 180 to 320 tokens between the low-$\kappa$ and high-$\kappa$ runs — routine — then a controller holding *per-token* KL fixed has silently widened the sequence budget by 1.8×, and the two arms are not at the KL values their labels claim.

Together: the objective's optimum is exponentially sensitive to a coefficient fit against a proxy, and the x-axis of the fit is not measured in a stable unit. That is why the sweep has not been replaced by a rule.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*