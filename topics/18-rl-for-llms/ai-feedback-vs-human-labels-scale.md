---
id: 18-rl-for-llms/ai-feedback-vs-human-labels-scale
title: "Constitutional Feedback Versus Human Labels at Scale"
topic: 18-rl-for-llms
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Constitutional Feedback Versus Human Labels at Scale

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/ai-feedback-vs-human-labels-scale` · **Status:** empirically-open

## 1. Problem Statement

Preference labels for RLHF can come from paid human annotators or from a model prompted with a written rule set (a "constitution"). The question: **at what label budget, and on which axes, does constitutional/AI feedback stop substituting for human labels?**

Three distinct variants, routinely conflated:

- **Measurement.** Given a fixed policy-training recipe, estimate the win rate of a policy trained on $N$ AI labels against one trained on $N$ human labels, under a judge that is *not* the AI labeler. Solving = a reliable, judge-independent estimate with confidence intervals.
- **Method.** Find the mixing rule $\alpha(N)$ — what fraction of the budget should be human — that maximizes gold-standard quality at fixed dollar cost. Solving = a demonstrated frontier that dominates both pure arms.
- **Theory.** Determine whether AI feedback can raise the ceiling of the policy above the labeler's own capability, or only transfer the labeler's ordering. Solving = a separation result or an impossibility proof under stated assumptions.

The measurement variant is the blocker: nearly all published comparisons use an LLM judge drawn from the same family as the labeler.

## 2. Formal Setting

Prompts $x \sim \mathcal{D}$; responses $y \sim \pi_{\text{sft}}(\cdot\mid x)$. A preference is a triple $(x, y_1, y_2)$ with a binary label $c \in \{1,2\}$.

**Label sources.** Human labeler $h$ and AI labeler $a$ (a model $\pi_L$ prompted with constitution $C$, scoring $\log p(\text{"A"}) - \log p(\text{"B"})$ with position-swap averaging). Both are stochastic: $p_h(c \mid x,y_1,y_2)$, $p_a(c \mid x,y_1,y_2; C)$.

**Gold utility.** An unobserved $u^\star(x,y) \in \mathbb{R}$. Everything measurable is a proxy. Define agreement of source $s$ with a *held-out human panel* (measured as raw accuracy against majority vote of $k \ge 3$ fresh annotators, not against the training labels):
$$\mathrm{Agr}(s) = \mathbb{E}_{(x,y_1,y_2)}\big[\Pr\nolimits_{c\sim p_s}[\,c = c_{\text{panel}}\,]\big].$$
The ceiling is human–human agreement $\mathrm{Agr}(h)$, typically $0.7$–$0.8$, not $1.0$.

**Training.** Reward model $r_\phi$ fit by Bradley–Terry NLL on $N$ labels; policy $\pi_\theta$ by
$$\max_\theta\; \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] - \beta\,\mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{sft}}\big).$$

**Quantity of interest.** With $\mathrm{Gold}(\pi) = \mathbb{E}_{x}\mathbb{E}_{y\sim\pi}[u^\star]$, measured as the win rate against a fixed reference completion under blinded human raters, define the **substitution ratio**
$$\rho(N) \;=\; \frac{\mathrm{Gold}(\pi^{\text{AI}}_N) - \mathrm{Gold}(\pi_{\text{sft}})}{\mathrm{Gold}(\pi^{\text{human}}_N) - \mathrm{Gold}(\pi_{\text{sft}})},$$
and the **exchange rate** $N_h(N_a) = \min\{N : \mathrm{Gold}(\pi^{\text{human}}_N) \ge \mathrm{Gold}(\pi^{\text{AI}}_{N_a})\}$ — how many human labels one AI-labeled budget of $N_a$ buys. The open question is the shape of $\rho(N)$ as $N \to 10^6$.

**Assumptions, and which fail.**
1. *Labels are i.i.d. draws from a fixed source.* Violated: human panels drift with rater cohort and instructions; AI labels shift with prompt template and decoding temperature.
2. *The judge is independent of the labeler.* Violated whenever an LLM judge shares a base model with $\pi_L$ — Panickssery et al. (2024) show evaluators favor their own generations.
3. *Bradley–Terry transitivity.* Violated: AI labelers exhibit position and length bias; raw position-flip disagreement of 20–30% is routine.
4. *A single scalar $u^\star$ exists.* Violated for helpfulness–harmlessness trade-offs, where preferences are genuinely multi-objective.

## 3. State of the Art

**Established.**
- Constitutional AI (Bai et al., Anthropic, 2022) trains harmlessness entirely from AI-generated preferences plus critique-revision SFT, using human labels only for helpfulness. That the *harmlessness* half can run without human preference labels is reproduced practice, not a claim.
- RLAIF (Lee et al., ICML 2024) reports on summarization, helpfulness and harmlessness that RLAIF and RLHF are statistically indistinguishable under human evaluation, both beating SFT.
- Overoptimization is lawful: Gao, Schulman & Hilton (ICML 2023) fit gold reward as a function of $d=\sqrt{\mathrm{KL}}$, with proxy–gold divergence growing predictably and shrinking with RM size and label count (RMs 3M–3B params, up to ~100k labels).

**Claimed but unablated.**
- "AI feedback matches human feedback at scale." The published comparisons top out around $10^4$–$10^5$ preference pairs. No public run isolates label source at $10^6$.
- Self-rewarding loops (Yuan et al., 2024) report AlpacaEval 2.0 length-controlled gains over three iterations of Llama-2-70B self-labeling. This exists as a **benchmark number under an LLM judge**, with no blinded human panel and no third iteration ceiling analysis.

**Contrary evidence, established.** Sharma et al., *A Critical Evaluation of AI Feedback for Aligning LLMs* (2024): most of the reported RLAIF gain is attributable to a weak SFT teacher. When the SFT stage already uses data from a model as strong as the AI labeler (GPT-4), the additional RLAIF step yields little or negative gain. This is the single most load-bearing ablation in the area.

## 4. What Is Known

- **AI-judge/human agreement.** GPT-4 agrees with human majority ~85% on MT-Bench pairwise judgments, above the 81% human–human agreement (Zheng et al., NeurIPS 2023 D&B). Ties excluded; agreement drops on math and reasoning prompts.
- **RLAIF vs RLHF.** Human evaluators preferred RLAIF over SFT 71% and RLHF over SFT 73% on TL;DR summarization; head-to-head RLAIF vs RLHF was ~50/50. Harmlessness: 88% harmless for RLAIF vs 76% RLHF vs 64% SFT (Lee et al., ICML 2024, PaLM 2 scale).
- **Human labels are noisy.** Inter-annotator agreement on InstructGPT preference data was ~72–77% (Ouyang et al., NeurIPS 2022). The human arm is not a clean reference.
- **Cost.** Human preference pairs cost roughly $0.3–$1 each at vendor rates; an AI label at 2024–2026 API prices costs $10^{-3}$–$10^{-2}$. Three orders of magnitude.
- **Reward models saturate.** RewardBench (Lambert et al., 2024) shows top RMs above 90% overall but near chance on adversarially perturbed reasoning subsets — accuracy is concentrated on easy pairs.
- **Scale of open recipes.** Tülu 3 (Lambert et al., 2024) uses on the order of $10^5$ synthetic preference pairs; label-source ablation at that scale is partial.

## 5. What Is Not Known

- **Empirically open.** The shape of $\rho(N)$ beyond $\sim10^5$ labels. The experiment is runnable today — cost is dominated by the human arm ($\sim$$300k for $10^6$ pairs) — and nobody has published it with a blinded judge.
- **Empirically open.** Whether a mixed budget ($\alpha$ human, $1-\alpha$ AI) strictly dominates both pure arms, and whether the optimal $\alpha$ decreases with $N$ (AI labels for coverage, human labels for the hard tail) or increases.
- **Theoretically open.** Whether AI feedback can lift $\mathrm{Gold}(\pi)$ above $\mathrm{Agr}(a)$-implied ceilings. Weak-to-strong generalization (Burns et al., 2023) shows a strong student can exceed a weak supervisor by recovering a large fraction of the performance gap on NLP tasks — but no theorem states when this holds for preference supervision.
- **Methodologically blocked.** $\mathrm{Gold}$ itself. There is no accepted estimator of policy quality that is simultaneously (a) not an LLM judge correlated with the labeler and (b) cheap enough to run at every point of a scaling sweep.

## 6. Why It Is Hard

**Confounded measurement, specifically judge–labeler correlation.** The cheap evaluator (an LLM judge) shares pretraining data, tokenizer, and often weights with the AI labeler. Any measured $\rho(N) \approx 1$ could mean "AI labels are as good as human labels" or "the judge shares the labeler's blind spots." Panickssery et al. (2024) measured self-preference directly: evaluators assign higher scores to their own generations, and the effect scales with the model's ability to recognize its own text. Removing the confound requires human panels at every sweep point, which reintroduces exactly the cost the experiment is trying to eliminate.

Second obstruction: **non-identifiability of the label-source effect from the SFT effect**. Sharma et al. showed the two are entangled — the measured RLAIF advantage moves with the SFT teacher, not just the labeler. Any comparison that does not hold the SFT stage fixed against the strongest available teacher is uninterpretable.

## 7. Current Research (as of 2026)

- Anthropic: constitution-conditioned reward models and Collective Constitutional AI (public input into the rule set); rule-following as a training target rather than a fixed preference set.
- OpenAI: rule-based rewards for safety behavior (Mu et al., 2024) — explicit propositions graded by a model, replacing human safety preference data.
- Google DeepMind: online AI feedback for direct alignment (Guo et al., 2024) — AI labels sampled on-policy each step, which sidesteps the offline-dataset staleness that afflicts DPO.
- AI2 / open-recipe groups: RewardBench, Tülu, UltraFeedback (Cui et al., 2024) — public synthetic preference corpora at $\sim$$10^5$ scale, the substrate any scaling sweep would use.
- *(frontier — verify)* Reported work on debate- and verifier-based feedback, and on constitution-as-context reward models that can be re-specified without retraining, has not yet been ablated against human labels at matched budget.

## 8. Concrete Next Experiment

**Scale.** One base model, 7–8B, fixed SFT stage distilled from the *same* model used as AI labeler (removes the Sharma confound). Sweep $N \in \{10^4, 3\times10^4, 10^5, 3\times10^5, 10^6\}$ preference pairs on a fixed prompt set.

**Arms.**
1. AI labels from a constitution-prompted labeler, all $N$.
2. **Control arm:** human labels, all $N$, same prompts, same pairs, same RM and PPO hyperparameters.
3. Mixed: $\alpha \in \{0.1, 0.3\}$ human, remainder AI, matched *dollar* cost to arm 2 rather than matched label count.

**Evaluation.** Blinded pairwise human panel, $k=5$ raters, 1,000 held-out prompts per sweep point, against a fixed reference. Report an LLM-judge win rate alongside — the gap between the two is itself a result.

**The deciding number.** $\rho(10^6)$ under the human panel, with a 95% CI narrower than $\pm 0.10$. $\rho \ge 0.9$ means AI feedback substitutes at the frontier of practical budgets; $\rho \le 0.6$ with $\rho(10^4) \approx 1$ means AI feedback saturates and the exchange rate collapses at scale. Human-panel cost: ~$0.4 \times 10^6 \approx$ $400k for the control arm; ~$120k for evaluation. Roughly $600k total — inside a single lab's budget, which is why "nobody ran it" is the finding.

## 9. Key References

- **[Foundational]** Bai, Y., Kadavath, S., Kundu, S., et al. *Constitutional AI: Harmlessness from AI Feedback.* Anthropic, 2022. — arXiv:2212.08073
- **[Foundational]** Ouyang, L., Wu, J., Jiang, X., et al. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[SOTA]** Lee, H., Phatale, S., Mansoor, H., et al. *RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback.* ICML, 2024. — arXiv:2309.00267
- **[SOTA]** Sharma, A., Keh, S., Mitchell, E., Finn, C., Arora, K., Kollar, T. *A Critical Evaluation of AI Feedback for Aligning Large Language Models.* NeurIPS, 2024. — arXiv:2402.12366
- **[SOTA]** Guo, S., Zhang, B., Liu, T., et al. *Direct Language Model Alignment from Online AI Feedback.* 2024. — arXiv:2402.04792
- **[Theory]** Gao, L., Schulman, J., Hilton, J. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[Theory]** Burns, C., Izmailov, P., Kirchner, J. H., et al. *Weak-to-Strong Generalization.* ICML, 2024. — arXiv:2312.09390
- **[Measurement]** Zheng, L., Chiang, W.-L., Sheng, Y., et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.05685
- **[Measurement]** Panickssery, A., Bowman, S. R., Feng, S. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS, 2024. — arXiv:2404.13076
- **[Measurement]** Lambert, N., Pyatkin, V., Morrison, J., et al. *RewardBench: Evaluating Reward Models for Language Modeling.* 2024. — arXiv:2403.13787
- **[Survey]** Casper, S., Davies, X., Shi, C., et al. *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* TMLR, 2023. — arXiv:2307.15217

## 10. Worked Example

Take the RLAIF summarization result and try to convert it into an exchange rate.

Reported: RLAIF wins 71% against SFT; RLHF wins 73%. Difference: 2 points. With $n=1{,}000$ rated prompts, the standard error on each win rate is $\sqrt{0.71 \times 0.29 / 1000} \approx 1.4\%$; on the difference, $\approx 2.0\%$. So the 2-point gap has a 95% CI of roughly $[-2, 6]$ — consistent with AI labels being 8% *worse* or 3% *better* in relative uplift terms.

Convert to $\rho$: uplift over SFT is $0.71 - 0.5 = 0.21$ (AI) versus $0.73 - 0.5 = 0.23$ (human), so $\hat\rho \approx 0.91$, with CI roughly $[0.74, 1.09]$ propagating the same standard errors. That interval is wide enough to contain both "full substitution" and "AI buys you three-quarters of a human label."

Now the obstruction. To narrow the CI on $\rho$ to $\pm 0.05$ you need the difference in win rates measured to about $\pm 1$ point, which needs $n \approx 10^4$ rated prompts per sweep point — 5 sweep points $\times$ 3 arms $\times$ $10^4$ $\times$ 5 raters $= 7.5 \times 10^5$ human judgments, before any labels are bought. Substituting an LLM judge cuts that to near zero dollars, but the judge shares a base model with the labeler; Panickssery et al.'s self-preference effect is on the order of several points, i.e. **larger than the effect being measured**. The measurement is not slightly noisy — its cheap version has a bias of the same magnitude as the signal. That is why $\rho(10^6)$ is empirically open rather than merely unmeasured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*