---
id: 04-alignment/preference-aggregation-disagreeing-annotators
title: "Preference Aggregation Across Disagreeing Annotators"
topic: 04-alignment
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Preference Aggregation Across Disagreeing Annotators

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/preference-aggregation-disagreeing-annotators` · **Status:** open

## 1. Problem Statement

RLHF pipelines collect pairwise comparisons from many annotators, pool them into one dataset, and fit a single scalar reward model. Annotators disagree on 20–40% of pairs. The pooling step is a **social choice rule**, chosen implicitly by the loss function rather than deliberately, and nobody has established which rule it is defensible to use.

- **Input:** a set of prompts $x$, response pairs $(a, b)$, and labels $y_{i}(x,a,b) \in \{a, b\}$ from annotators $i \in [N]$, with each item labeled by $k \ll N$ annotators.
- **Output:** a policy $\pi_\theta$, or the reward $r_\theta$ that induces it.
- **Objective:** a rule $F$ mapping the profile of individual preference orders to a collective order, with stated axiomatic properties, plus an estimator of $F$ that is consistent from $k$-sparse, non-uniformly-assigned labels.

Three variants, of different difficulty:

- **Measurement.** Decompose observed disagreement into annotator noise (correctable by more labels), genuine value pluralism (not correctable), and item ambiguity. Currently confounded; this is the binding constraint.
- **Method.** Given the decomposition, fit a reward model that respects a chosen aggregation rule. Partly solved for known rules and known groups.
- **Theory.** Characterize which aggregation rules are achievable by gradient-trained reward models under sparse comparison data, and what Arrow-style impossibilities apply. Largely open.

**Solved** would mean: a pipeline that (i) reports which rule it implements, (ii) certifies that rule against named axioms, and (iii) shows on held-out annotators that no group's satisfaction is worse than a stated floor.

## 2. Formal Setting

Annotator $i$ has utility $u_i: \mathcal{X} \times \mathcal{A} \to \mathbb{R}$. Annotators are drawn from a population with measure $\mathcal{P}$. Under Bradley–Terry (BT), annotator $i$ labels

$$\Pr[y_i = a \mid x, a, b] = \sigma\big(u_i(x,a) - u_i(x,b)\big), \quad \sigma(t) = (1+e^{-t})^{-1}.$$

**Measured quantities.**

- *Observed disagreement* $\delta$: over items with $k \ge 2$ labels, $\delta = \Pr[y_i \ne y_j]$ for a random pair of distinct labelers on the same item. Estimated as the pooled per-item pair-discordance rate; report Krippendorff's $\alpha$ alongside, since $\delta$ depends on class balance.
- *Test–retest noise* $\delta_{\text{self}}$: the same annotator, same item, re-shown after a delay. This is the only direct upper bound on the noise component and is almost never collected.
- *Between-annotator variance* $\sigma_{\text{bet}}^2 = \mathrm{Var}_{i}\!\left[\mathbb{E}[\,y_i \mid x,a,b\,]\right]$, identified only when the same items are labeled by many annotators.
- *Group regret* for group $g$: $R_g = \mathbb{E}_{x}\!\left[\max_{a} \bar u_g(x,a) - \bar u_g(x, \pi_\theta(x))\right]$ with $\bar u_g$ the within-group mean utility.

**The pooled objective.** Standard RLHF minimizes $\mathcal{L}(\theta) = -\sum_{(x,a,b,y)} \log \sigma\big(r_\theta(x,a) - r_\theta(x,b)\big)$ over the pooled dataset, discarding annotator identity. The implicit rule is not utilitarian. Siththaranjan, Laidlaw & Hadfield-Menell (ICLR 2024) prove that when preferences depend on hidden context $z$ (annotator identity being one instance), the BT population minimizer is the **Borda count**:

$$r^\star(x,a) \;=\; \Phi\Big(\Pr_{b \sim \mu,\, z}\big[a \succ_z b \mid x\big]\Big),$$

monotone in the probability that $a$ beats a randomly drawn alternative. Utility magnitudes cancel; only win rates survive.

**Assumptions, and which fail.**

| Assumption | Status |
|---|---|
| Labels are i.i.d. draws from one BT model | Violated — $\sigma_{\text{bet}}^2 > 0$ on every subjective task measured |
| Annotator assignment independent of item | Violated — crowd platforms assign by availability, self-selection, and language |
| Utilities are interpersonally comparable on one scale | Unfalsifiable from comparison data alone; formally, non-identifiable |
| Disagreement is noise, reducible by more labels | Violated — Pavlick & Kwiatkowski (TACL 2019) show stable multi-modal NLI judgments |
| The annotator pool represents the deployment population | Violated — PRISM (Kirk et al. 2024) shows large cross-country divergence |

## 3. State of the Art

**Established.**

- *Borda characterization* of pooled BT under hidden context — proved, with matching synthetic experiments (Siththaranjan et al., ICLR 2024). Their Distributional Preference Learning (DPL) predicts a distribution over reward per response rather than a point, and detects hidden context via predicted variance.
- *Axiomatic failure of standard RLHF.* Ge, Procaccia et al., *Axioms for AI Alignment from Human Feedback* (NeurIPS 2024): BT-based reward learning violates majority consistency; a family of linear social-choice rules satisfies the axioms they propose, with sample-complexity guarantees.
- *Impossibility for a single reward.* Chakraborty et al., *MaxMin-RLHF* (ICML 2024): a single reward model cannot align to a population with sufficiently diverse utilities; they give an egalitarian (max-min over a learned mixture) alternative.
- *Multi-annotator modeling beats majority vote on calibration.* Davani, Díaz & Prabhakaran (TACL 2022): multi-task heads per annotator match or beat majority-vote baselines on hate-speech and emotion tasks while additionally recovering disagreement.

**Claimed but unablated.**

- Personalization methods — Rewarded Soups (Rame et al., NeurIPS 2023), Personalized Soups (Jang et al. 2023), Variational Preference Learning (Poddar et al., NeurIPS 2024) — report gains on synthetic or attribute-labeled preference splits. Whether they recover *real* annotator heterogeneity is untested; the group labels are researcher-constructed.
- MaxMin-RLHF's reported win-rate gains for minority groups rest on simulated group structure, not measured annotator clusters.

**Benchmark-number-only.** RewardBench (Lambert et al. 2024) scores reward models against single gold labels, so it cannot distinguish a model that tracks the majority from one that tracks the population distribution. Reported leaderboard deltas carry no information about aggregation quality.

## 4. What Is Known

- **Agreement is low and stable across labs.** Stiennon et al. (NeurIPS 2020, TL;DR summarization, ~64k comparisons): labeler–labeler agreement 73%, researcher–researcher 73%. Ouyang et al. (InstructGPT, NeurIPS 2022): inter-annotator agreement 72.6 ± 1.5% on training labelers, 77.3 ± 1.3% on held-out. Chance is 50%. Anthropic's HH-RLHF (Bai et al. 2022, ~161k comparisons) reports comparable rates.
- **Disagreement is partly irreducible.** ChaosNLI (Nie, Zhou & Bansal, EMNLP 2020) collected 100 labels each on 4,645 NLI items; label distributions are frequently multi-modal, and the 100-annotator majority disagrees with the original 5-annotator gold on a substantial fraction. Extra annotation shrinks the error bars, not the spread.
- **Populations differ systematically.** PRISM (Kirk et al., NeurIPS D&B 2024): 1,500 participants across 75 countries, 8,011 conversations, with demographic and stated-value metadata — the first dataset that permits group-conditional reward estimation on real annotators.
- **Aggregation rules disagree with each other.** Classical social choice: Arrow (1951) rules out any rank aggregation over $\ge 3$ alternatives satisfying unrestricted domain, Pareto, IIA and non-dictatorship. Borda violates IIA; max-min violates Pareto-efficiency in the strong form.
- **Annotator misspecification is a density-estimation error.** Dumoulin et al. (TMLR 2024) show that fitting one BT model to a mixture of annotator models yields a reward that matches neither the mean nor the mode of the underlying utilities.

## 5. What Is Not Known

**Methodologically blocked (the binding gap).** No accepted estimator separates $\delta$ into noise, ambiguity, and pluralism. Doing so requires test–retest data ($\delta_{\text{self}}$) at scale, which no public RLHF dataset contains. Without it, "the model should respect disagreement" is not a testable claim — every observed heterogeneity has a noise explanation of equal fit.

**Theoretically open.** Which social-choice rules are *representable* by a scalar reward model trained by gradient descent on sparse pairwise data? Borda is (Siththaranjan et al.). Whether max-min, Kemeny, or proportional rules admit consistent estimators under $k$-sparse, non-random annotator assignment is unproved. No Arrow-style impossibility has been stated in the reward-model-plus-KL-regularized-policy setting.

**Empirically open.** Nobody has trained a frontier-scale policy under two different aggregation rules on the *same* annotator pool and measured per-group regret on held-out annotators. The experiment is runnable — PRISM supplies the annotator metadata — and costs one extra RLHF run.

## 6. Why It Is Hard

**Non-identifiability, and an evaluation that does not measure what it names.**

From pairwise comparisons alone, $u_i$ is identified only up to a monotone transform per annotator. Interpersonal comparison — "annotator 1's mild preference versus annotator 2's strong one" — has no empirical content in the data actually collected. Utilitarian aggregation needs cardinal, comparable utilities; the data support only ordinal, per-annotator ones. This is not a sample-size problem: more comparisons do not create the missing scale.

Compounding it: the standard evaluation is win rate against a gold label produced by the same pooling rule under test. A model that reproduces the majority scores well by construction; a model that correctly represents a 55/45 split scores worse. The metric rewards the failure mode.

## 7. Current Research (as of 2026)

- **Social-choice-theoretic alignment.** Conitzer, Procaccia, Dai and collaborators (CMU, Duke, Harvard) — *Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback* (ICML 2024) and the NeurIPS 2024 axioms paper. Direction: import voting-rule axioms into reward learning.
- **Distributional and latent-variable reward models.** Hadfield-Menell's group (MIT) on DPL; Poddar/Levine/Fazel-style variational personalization (UW/Berkeley).
- **Deliberative aggregation.** DeepMind's *Fine-tuning language models to find agreement among humans with diverse preferences* (Bakker et al., NeurIPS 2022) — consensus statements rated by ~1,500 UK participants. Anthropic's Collective Constitutional AI (2023, ~1,000 US participants) extends this to constitution drafting.
- **Human-label-variation as a first-class object.** Plank (LMU Munich), Aroyo & Welty, the LeWiDi shared tasks — modeling label distributions rather than gold labels.
- *(frontier — verify)* Reports of production RLHF pipelines retaining annotator IDs and fitting per-annotator random effects; per-cohort reward heads for jurisdictional variation. Not documented in public model cards.

## 8. Concrete Next Experiment

**Question.** Does the choice of aggregation rule change deployed behavior more than the noise floor of the training run?

**Scale.** One 8B-parameter base model. Training data: PRISM (8,011 conversations, 1,500 annotators with country and stated-value metadata) plus 30k pooled preference pairs for coverage. Three RLHF arms, identical hyperparameters, three seeds each — 9 runs, roughly 3k GPU-hours on H100s.

- **Arm A (control):** standard pooled BT reward, annotator IDs discarded. This is the incumbent, and by the Borda result is a known rule, not an unknown one.
- **Arm B:** utilitarian per-annotator model — BT with annotator random effects, aggregate by mean predicted utility.
- **Arm C:** egalitarian — max-min over $K=5$ annotator clusters recovered by $k$-means on per-annotator reward embeddings.

**Held-out evaluation.** 300 annotators withheld from training, stratified by cluster, each judging 100 prompts across arms.

**The deciding number.** $\Delta = \max_g R_g^{(A)} - \max_g R_g^{(C)}$: the reduction in worst-group regret from egalitarian aggregation, measured as win-rate difference against a fixed reference policy within the worst-off held-out cluster. Compare to the seed-to-seed standard deviation $s$ within Arm A.

- $\Delta > 3s$ and worst-group win rate rises by $\ge 5$ points: aggregation rule is a first-class design decision and pooled BT is leaving welfare on the table.
- $\Delta < s$: annotator heterogeneity in this pool is dominated by noise; the pooling debate is not empirically live at this scale, and effort belongs in the measurement variant.

**Required addition.** Collect $\delta_{\text{self}}$ on 2,000 items — same annotator, same item, 14-day gap. Without it $\Delta$ cannot be attributed to pluralism rather than label noise.

## 9. Key References

- **[Foundational]** Arrow, K. *Social Choice and Individual Values.* Wiley, 1951.
- **[Foundational]** Bradley, R. & Terry, M. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika, 1952.
- **[Foundational]** Dawid, A. P. & Skene, A. M. *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* JRSS Series C, 1979.
- **[Foundational]** Pavlick, E. & Kwiatkowski, T. *Inherent Disagreements in Human Textual Inferences.* TACL, 2019.
- **[SOTA]** Siththaranjan, A., Laidlaw, C. & Hadfield-Menell, D. *Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF.* ICLR, 2024. — arXiv:2312.08358
- **[SOTA]** Ge, L., Halpern, D., Micha, E., Procaccia, A. D., Shapira, I., Vorobeychik, Y. & Wu, J. *Axioms for AI Alignment from Human Feedback.* NeurIPS, 2024. — arXiv:2405.14758
- **[SOTA]** Chakraborty, S. et al. *MaxMin-RLHF: Alignment with Diverse Human Preferences.* ICML, 2024. — arXiv:2402.08925
- **[SOTA]** Poddar, S., Wan, Y., Ivison, H., Gupta, A. & Jaques, N. *Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning.* NeurIPS, 2024. — arXiv:2408.10075
- **[Data]** Kirk, H. R. et al. *The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of LLMs.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2404.16019
- **[Data]** Nie, Y., Zhou, X. & Bansal, M. *What Can We Learn from Collective Human Opinions on Natural Language Inference Data?* EMNLP, 2020.
- **[Method]** Davani, A. M., Díaz, M. & Prabhakaran, V. *Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations.* TACL, 2022.
- **[Method]** Bakker, M. et al. *Fine-Tuning Language Models to Find Agreement Among Humans with Diverse Preferences.* NeurIPS, 2022.
- **[Survey]** Plank, B. *The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation.* EMNLP, 2022. — arXiv:2211.02570
- **[Survey]** Casper, S. et al. *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* TMLR, 2023. — arXiv:2307.15217
- **[Position]** Conitzer, V. et al. *Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback.* ICML, 2024. — arXiv:2404.10271

## 10. Worked Example

Three responses to "Should I tell my colleague their work is bad?", a population split 60/40 between a directness-preferring group $G_1$ and a harmony-preferring group $G_2$.

| Response | $u_{G_1}$ | $u_{G_2}$ |
|---|---|---|
| $a$: blunt critique | $+10$ | $-3$ |
| $b$: hedged, mostly praise | $-2$ | $+9$ |
| $c$: neutral, vague, unhelpful | $+1$ | $+1$ |

**Utilitarian.** $0.6(10)+0.4(-3) = 4.8$ for $a$; $0.6(-2)+0.4(9) = 2.4$ for $b$; $1.0$ for $c$. Winner: $a$.

**Max-min.** Worst-group value: $a \to -3$, $b \to -2$, $c \to +1$. Winner: $c$.

**Pooled BT (what RLHF actually does).** Pairwise win probabilities, with logistic noise, are dominated by sign: $G_1$ ranks $a \succ c \succ b$, $G_2$ ranks $b \succ c \succ a$. Borda scores over the three pairs — $a$: $0.6 \times 2 + 0.4 \times 0 = 1.2$; $b$: $0.6\times 0 + 0.4 \times 2 = 0.8$; $c$: $0.6\times1 + 0.4\times1 = 1.0$. Winner: $a$, but $c$ beats $b$ despite $b$ carrying more than three times $c$'s total utility. The $+9$ and $-2$ are erased; only the win indicator survives.

**Where the obstruction becomes visible.** All three rules disagree, and the observable data — a 60/40 split of pairwise labels on every pair — is *identical* under this utility table and under a table where $G_2$'s magnitudes are $+1/-0.2$ instead of $+9/-3$. In the second table, utilitarian and max-min both pick $a$. The comparison data cannot tell the two tables apart, so no amount of additional pairwise annotation selects between "utilitarian says $a$" and "utilitarian says $c$". Choosing a rule therefore requires either cardinal elicitation (ratings, budget allocation, willingness-to-wait) or an explicit, unfalsifiable normative assumption — and today's pipelines make that assumption silently, in the loss function, defaulting to Borda.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*