---
id: 04-alignment/optimal-kl-coefficient-rlhf
title: "Optimal KL Regularization Coefficient in RLHF"
topic: 04-alignment
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal KL Regularization Coefficient in RLHF

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/optimal-kl-coefficient-rlhf` · **Status:** open

## 1. Problem Statement

RLHF optimizes a proxy reward model against a KL penalty to a reference policy. The penalty weight $\beta$ is set by hand, tuned by sweep, or controlled adaptively to hit a target KL. The problem: **is there a principled rule that selects $\beta$ (or the target KL) from measurable properties of the reward model, the reference policy, and the preference dataset — without a gold-reward sweep?**

Three variants, different difficulty:

- **Measurement.** Given a run, estimate the KL actually incurred and the true (gold) utility at that KL. Blocked by the absence of gold reward outside synthetic setups.
- **Method.** Produce a controller that sets $\beta$ online and lands within some tolerance of the gold-optimal point. Empirically open; several candidates exist, none validated against real human gold utility at frontier scale.
- **Theory.** Prove a bound of the form: for reward model error $\varepsilon$ under a specified error model, the regret of $\beta$ relative to $\beta^\star$ is at most $f(\varepsilon,\beta)$, and derive $\beta^\star$. Open except under strong, known-false assumptions.

A solution is a rule $\hat\beta$ computable before or during training whose gold-utility regret against the sweep-optimal $\beta^\star$ is small and stays small as model scale, RM scale, and prompt distribution change.

## 2. Formal Setting

Prompts $x\sim\mathcal D$, responses $y$, reference policy $\pi_{\text{ref}}$ (the SFT checkpoint). Proxy reward $r_\phi:\mathcal X\times\mathcal Y\to\mathbb R$ trained by Bradley–Terry maximum likelihood on preference pairs. The objective:

$$\max_{\pi}\ \mathbb E_{x\sim\mathcal D,\,y\sim\pi(\cdot|x)}\big[r_\phi(x,y)\big]\;-\;\beta\,\mathbb E_{x}\big[\mathrm{KL}\!\left(\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right)\big].$$

The unconstrained maximizer is the Gibbs tilt

$$\pi^\star_\beta(y|x)=\tfrac{1}{Z_\beta(x)}\,\pi_{\text{ref}}(y|x)\exp\!\big(r_\phi(x,y)/\beta\big),$$

so $\beta$ indexes a one-parameter family, and any $\beta$ corresponds to a KL budget $D(\beta)=\mathbb E_x\,\mathrm{KL}(\pi^\star_\beta\|\pi_{\text{ref}})$, monotone decreasing in $\beta$.

**How each quantity is measured.**
- $\mathrm{KL}$: sequence-level, $\frac{1}{N}\sum_i\big[\log\pi_\theta(y_i|x_i)-\log\pi_{\text{ref}}(y_i|x_i)\big]$ over on-policy samples, in nats per response. Implementations differ: the $k_3$ estimator $\mathbb E_{\pi_\theta}[\rho-1-\log\rho]$ with $\rho=\pi_{\text{ref}}/\pi_\theta$ is lower-variance and non-negative; token-level vs. sequence-level and length normalization change reported numbers by factors of the mean length. **Cross-paper KL numbers are not comparable without knowing the convention.**
- $r_\phi$: in RM units. Bradley–Terry fixes only differences, and the *scale* is set by fit quality; it drifts across RM training runs. Most pipelines whiten $r_\phi$ to zero mean and unit variance on a prompt batch — that whitening, not $\beta$, silently sets the effective temperature.
- Gold utility $u^\star$: human preference win rate against $\pi_{\text{ref}}$, or a held-out "gold" RM in synthetic-supervision setups.

**Assumptions and their status.**
1. *$r_\phi$ approximates $u^\star$ uniformly over the reachable policy set.* Violated — error grows with KL, which is exactly the overoptimization phenomenon.
2. *The regularizer is the KL from the SFT policy.* Violated in intent: practitioners want closeness in capability/diversity, and KL is a poor proxy for both.
3. *A single scalar $\beta$ suffices.* Violated when the reward is a weighted mixture (helpful/harmless/verbosity); per-component overoptimization onsets differ.
4. *PPO reaches $\pi^\star_\beta$.* Violated — clipping, value-function error, and finite steps mean the realized policy tracks a KL trajectory, not the tilt at any fixed $\beta$.

## 3. State of the Art

**Established.**
- Gao, Schulman & Hilton (*Scaling Laws for Reward Model Overoptimization*, ICML 2023) fit, with $d=\sqrt{\mathrm{KL}}$, gold reward $R(d)=d(\alpha-\beta_{\text{bon}}d)$ for best-of-$n$ and $R(d)=d(\alpha-\beta_{\text{RL}}\log d)$ for PPO; coefficients vary smoothly with RM parameter count and log-linearly with preference-dataset size. This is the strongest existing quantitative handle on where to stop.
- Best-of-$n$ has closed-form KL $\log n-\frac{n-1}{n}$ (Beirami et al., 2024, correcting the folklore $\log n$), giving an exact KL-budget dial to compare against RL.
- Korbak, Perez & Buckley (EMNLP Findings 2022) show KL-regularized RL is exactly Bayesian inference with $\pi_{\text{ref}}$ as prior — $\beta$ is a prior-strength parameter, not a free knob.

**Claimed but unablated.**
- Adaptive KL controllers (Ziegler et al., 2019) that adjust $\beta$ to hit a target KL are near-universal in implementations. They convert the problem into choosing the target; no paper shows the target itself is chosen well.
- Constrained-RLHF (Moskovitz et al., ICLR 2024) sets per-reward-model Lagrange multipliers by detecting each component's proxy-point. Demonstrated on composite rewards at modest scale; not validated against human gold utility.
- Reward-model ensembles (Coste et al., ICLR 2024) push the turnover point outward; Eisenstein et al. (COLM 2024) show ensembles *mitigate but do not eliminate* hacking, because ensemble members share pretraining-induced error.

**Benchmark-number-only.** Nearly all "we used $\beta=0.01$/$0.05$/$0.1$" reports, including the DPO/IPO literature's $\beta$ sweeps on AlpacaEval and MT-Bench. These are single-seed, single-scale points with no gold reference.

## 4. What Is Known

- **The turnover is real and scale-dependent.** Gao et al. (policy 1.2B; RMs 3M–3B parameters; synthetic 6B gold RM) find gold reward rises, peaks, then falls with KL; larger RMs peak later and higher, and the drop-off coefficient $\beta_{\text{RL}}$ shrinks with RM size. Policy size shifted the curves much less than RM size.
- **Order of magnitude of useful KL.** Published RLHF runs report KL budgets of roughly 5–30 nats per response at the operating point; Ziegler et al. (2019) targeted single-digit nats on summarization/stylistic tasks. InstructGPT (Ouyang et al., 2022; 1.3B–175B) used a small fixed KL reward coefficient ($\beta\approx0.02$ in whitened RM units) plus a pretraining-mixture term.
- **KL is not a sufficient statistic for degradation.** Kirk et al. (ICLR 2024) measure RLHF at 7B–70B and find output diversity collapses (large drops in distinct-$n$ and semantic diversity across seeds) at KL levels where win rate is still improving. Diversity loss and reward gain are not co-monotone in $\beta$.
- **Regularization strength interacts with preference determinism.** Azar et al. (AISTATS 2024) prove that when preferences are near-deterministic, DPO's implicit KL regularization degenerates: the optimum drives $\pi/\pi_{\text{ref}}$ ratios to extremes regardless of $\beta$. IPO's bounded objective fixes this case.
- **Direct alignment overoptimizes too.** Rafailov et al. (2024) report the same inverted-U for DPO/IPO/SLiC as a function of measured KL, at 1B–7B — so the phenomenon is not a PPO artifact.

## 5. What Is Not Known

- **Theoretically open.** No bound gives $\beta^\star$ from measurable RM properties. Under a uniform-error model ($\|r_\phi-u^\star\|_\infty\le\varepsilon$) the regularized objective is trivially $2\varepsilon$-robust for any $\beta$, which is vacuous — it does not produce a turnover. The needed ingredient is a *KL-dependent* error model $\varepsilon(D)$ with an empirically identified functional form; none has been derived from RM training rather than fitted post hoc.
- **Empirically open.** Whether the Gao et al. scaling coefficients extrapolate to $\ge$70B policies, real (non-synthetic) human gold utility, and modern iterative/online pipelines. The experiment is runnable — it costs a sweep of full RLHF runs plus human evaluation at each point, roughly $10$–$10^2$ full training runs.
- **Methodologically blocked.** "Optimal" has no agreed objective. Gold win rate, capability retention, diversity, and safety peak at different $\beta$, and there is no accepted scalarization. Separately, reported KL is not comparable across papers (Section 2), so no meta-analysis of existing runs can even be assembled.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the reward scale combined with an absent gold signal**. $\beta$ only ever acts through the ratio $r_\phi/\beta$, and $r_\phi$'s scale is a free parameter of RM training that whitening resets per run. So $\beta$ is not transferable between runs; only the induced KL is, and KL is measured with incompatible conventions. Meanwhile the thing $\beta$ trades against — true utility — is unobservable at the sample sizes needed: distinguishing two $\beta$ values whose human win rates differ by 2 points at 95% confidence needs on the order of $10^4$ pairwise human judgments per arm. Synthetic gold RMs sidestep the cost but share architecture and pretraining data with the proxy, so their disagreement understates real error (Eisenstein et al., 2024). The result: the quantity to optimize is unmeasurable, and the knob is not comparable across the runs you would pool.

## 7. Current Research (as of 2026)

- **Adaptive/constrained schedules.** Lagrangian and proxy-point-detection controllers descended from Moskovitz et al.; extensions to per-component budgets in multi-objective RLHF. *(frontier — verify)* Several industrial pipelines reportedly schedule $\beta$ downward over training rather than holding it fixed.
- **KL-aware theory of online RLHF.** Xiong et al. (ICML 2024) give regret guarantees for iterative preference learning under an explicit KL constraint — the closest existing link between the constraint level and a sample-complexity statement.
- **Alternative regularizers.** $\chi^2$ / $f$-divergence and pessimism-based substitutes, motivated by KL's failure to penalize the low-probability tails where hacking lives.
- **Inference-time comparators.** Best-of-$n$ and its distillation as a KL-matched control arm, using the exact $\log n-\frac{n-1}{n}$ budget (Beirami et al.; Yang et al., ISIT 2024).
- **Groups.** OpenAI (Gao/Schulman lineage), Google DeepMind (Beirami, Eisenstein), Anthropic, Stanford (Rafailov, Finn), UIUC/Xiong, Cohere and Allen AI on open replications.

## 8. Concrete Next Experiment

**Question.** Does a $\beta$ selected by a cheap, run-local diagnostic match the gold-optimal $\beta$?

**Setup.** Policy: one open 7B and one open 70B instruction-tuned base. Proxy RM: 7B, trained on 100k real preference pairs. Sweep $\beta\in\{0.005,0.01,0.02,0.05,0.1,0.2\}$, 3 seeds, PPO, fixed step budget, logging sequence-level KL with the $k_3$ estimator, reported in nats per response with the whitening statistics published.

**Control arm.** Best-of-$n$ from $\pi_{\text{ref}}$ with the *same* RM, $n$ chosen so $\log n-\frac{n-1}{n}$ equals each run's realized KL. This isolates "did RL find better policy mass at this budget" from "is this budget right".

**Diagnostic under test.** $\hat\beta$ chosen as the largest $\beta$ (smallest KL) at which the disagreement rate between the proxy RM and a held-out RM trained on a disjoint preference split first exceeds its $\pi_{\text{ref}}$ baseline by 5 percentage points — computable during training, no humans.

**Deciding number.** Human win rate against $\pi_{\text{ref}}$, 5,000 pairwise judgments per arm (±1.4 pp at 95%). Report $\Delta = u^\star(\beta^\star) - u^\star(\hat\beta)$. **If $\Delta \le 2$ pp at both 7B and 70B, the diagnostic is a usable rule; if $\Delta > 5$ pp at either scale, or if $\beta^\star$ itself moves by more than one sweep step between 7B and 70B, no scale-free rule of this form exists and the target must be re-derived per scale.**

## 9. Key References

- **[Foundational]** Daniel Ziegler, Nisan Stiennon, Jeffrey Wu, Tom Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving. *Fine-Tuning Language Models from Human Preferences.* 2019. — arXiv:1909.08593
- **[Foundational]** Nisan Stiennon et al. *Learning to Summarize with Human Feedback.* NeurIPS 2020. — arXiv:2009.01325
- **[Foundational]** Long Ouyang et al. *Training language models to follow instructions with human feedback.* NeurIPS 2022. — arXiv:2203.02155
- **[SOTA]** Leo Gao, John Schulman, Jacob Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. — arXiv:2210.10760
- **[SOTA]** Ted Moskovitz, Aaditya Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhutdinov, Anca Dragan, Stephen McAleer. *Confronting Reward Model Overoptimization with Constrained RLHF.* ICLR 2024. — arXiv:2310.04373
- **[SOTA]** Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, Bradley Knox, Chelsea Finn, Scott Niekum. *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms.* NeurIPS 2024. — arXiv:2406.02900
- **[Theory]** Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. — arXiv:2310.12036
- **[Theory]** Tomasz Korbak, Ethan Perez, Christopher Buckley. *RL with KL penalties is better viewed as Bayesian inference.* Findings of EMNLP 2022. — arXiv:2205.11275
- **[Theory]** Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D'Amour, Jacob Eisenstein, Chirag Nagpal, Ananda Theertha Suresh. *Theoretical guarantees on the best-of-n alignment policy.* 2024. — arXiv:2401.01879
- **[Theory]** Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, Tong Zhang. *Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint.* ICML 2024. — arXiv:2312.11456
- **[Empirical]** Jacob Eisenstein et al. *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking.* COLM 2024. — arXiv:2312.09244
- **[Empirical]** Thomas Coste, Usman Anwar, Robert Kirk, David Krueger. *Reward Model Ensembles Help Mitigate Overoptimization.* ICLR 2024. — arXiv:2310.02743
- **[Empirical]** Robert Kirk et al. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR 2024. — arXiv:2310.06452
- **[Survey]** Stephen Casper et al. *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* TMLR 2023. — arXiv:2307.15217

## 10. Worked Example

Take the Gaussian-tilt case, where everything is closed form. Suppose under $\pi_{\text{ref}}$ the whitened proxy reward of a sampled response is $r\sim\mathcal N(0,\sigma^2)$ with $\sigma=1$. The Gibbs tilt $\pi^\star_\beta\propto\pi_{\text{ref}}e^{r/\beta}$ is then a mean shift of $\sigma^2/\beta$ with

$$\mathrm{KL}(\pi^\star_\beta\|\pi_{\text{ref}})=\frac{\sigma^2}{2\beta^2}\quad\Longrightarrow\quad \beta=\frac{\sigma}{\sqrt{2\,\mathrm{KL}}}.$$

Now impose an inverted-U in $d=\sqrt{\mathrm{KL}}$, Gao-style. Take illustrative coefficients: proxy $R_p(d)=d(0.95-0.05d)$, gold $R_g(d)=d(0.85-0.16d)$.

- Proxy peak: $d_p=0.95/(2\cdot0.05)=9.5$, i.e. $\mathrm{KL}=90$ nats, $\beta=1/\sqrt{180}=0.075$.
- Gold peak: $d_g=0.85/(2\cdot0.16)=2.66$, i.e. $\mathrm{KL}=7.1$ nats, $\beta=1/\sqrt{14.2}=0.265$.

Training to the proxy optimum overshoots the gold optimum by **13× in KL** and lands at $R_g(9.5)=-6.4$ — worse than $\pi_{\text{ref}}$ itself, whose gold score is $0$. Choosing $\beta$ by "maximize the reward we can see" is not merely suboptimal; it is worse than not training.

**Where the obstruction shows.** Both $\beta$ values above were computed with $\sigma=1$. Suppose a second RM run fits better and its unwhitened reward has $\sigma=2.5$. The same $\beta=0.265$ now yields $\mathrm{KL}=\sigma^2/(2\beta^2)=17.8$ nats — 2.5× the gold-optimal budget — with no change to any hyperparameter in the config file. The transferable quantity is the KL, not $\beta$; and the curve $R_g$ that determines the right KL is exactly the one you cannot measure without gold labels. That is the whole problem in two lines.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*