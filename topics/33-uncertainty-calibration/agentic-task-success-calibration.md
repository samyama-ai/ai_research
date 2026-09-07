---
id: 33-uncertainty-calibration/agentic-task-success-calibration
title: "Calibration of Long-Horizon Agentic Task Success Predictions"
topic: 33-uncertainty-calibration
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration of Long-Horizon Agentic Task Success Predictions

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/agentic-task-success-calibration` · **Status:** empirically-open

## 1. Problem Statement

An agent is given a task $x$ (a repository and an issue, a browser goal, a research subtask) and a budget. Before or during execution it emits a probability $p \in [0,1]$ that its own eventual trajectory will be scored a success. The outcome $y \in \{0,1\}$ arrives hours later, once. The question: can an agent be made **calibrated** on its own long-horizon success — $\Pr[y=1 \mid p] = p$ — and does calibration survive the conditioning that actually matters (task family, budget, whether the forecast is used to gate an action)?

Three variants, with different difficulty:

- **Measurement.** Define a success-forecast calibration error that is estimable from the few hundred long tasks a lab can afford to run, where each task costs minutes to hours of tool-calling. Currently blocked: standard ECE estimators are biased and the sample sizes are two orders of magnitude too small.
- **Method.** Produce a forecast head — verbalized, logit-based, or a learned probe over trajectory state — with lower calibration error than the strongest cheap baseline (a per-task-family base rate) under distribution shift to unseen task families.
- **Theory.** Characterize what calibration is even achievable when the agent's forecast *causes* the outcome: if $p$ is low and the scaffold aborts, $y$ is never observed. This is calibration under a decision-dependent, censored outcome, not the i.i.d. setting of Dawid (1982) / Foster–Vohra (1998).

Solving it means: a deployed agent whose stated 20% is a real 20% across task families it has not seen, with a confidence interval tight enough to act on.

## 2. Formal Setting

Task $x \sim \mathcal{D}$ over a task distribution; policy $\pi_\theta$; scaffold budget $B$ (wall-clock seconds, tool calls, or dollars). A rollout produces trajectory $\tau = (a_1, o_1, \dots, a_T, o_T)$ with $T \le B$. The grader $g$ is a program or human returning $y = g(x,\tau) \in \{0,1\}$.

**Forecast.** At step $t$, $p_t = f(x, \tau_{1:t})$. The pre-execution forecast is $p_0 = f(x)$. Measured as: the model's stated probability parsed from text (verbalized), or $\sigma(w^\top h_t)$ for a probe on hidden state $h_t$, or the normalized logit on "yes" for a "will you succeed?" query (Kadavath-style $P(\text{IK})$).

**Calibration error.** For a partition into $M$ bins $B_m$ of the $[0,1]$ interval,

$$\widehat{\mathrm{ECE}} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\bigl|\,\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\,\bigr|,$$

with $\mathrm{acc}(B_m) = |B_m|^{-1}\sum_{i \in B_m} y_i$ and $\mathrm{conf}(B_m) = |B_m|^{-1}\sum_{i \in B_m} p_i$. This estimator is **biased upward** at small $n$ (Kumar–Liang–Ma, NeurIPS 2019); at $n = 200$ tasks and $M = 10$ bins the bias is comparable to the signal. Brier score $\frac{1}{n}\sum (p_i - y_i)^2$ is a strictly proper scoring rule (Gneiting–Raftery, JASA 2007) and decomposes into calibration + refinement; report both.

**Horizon coupling.** Success is budget-dependent. Write $S(x, B) = \Pr[y=1 \mid x, B]$. Empirically $\Pr[y = 1 \mid \text{task length } \ell]$ is well fit by a logistic in $\log \ell$ (METR, 2025), giving a *time horizon* $H_{50}$ at which success probability is $0.5$. A forecast is only well-posed relative to a stated $B$; $p$ without $B$ is not a probability of anything.

**Assumptions, and which are violated.**
1. *i.i.d. tasks.* Violated: benchmark tasks are clustered by repository, tool, and author; SWE-bench instances share repos.
2. *Grader is ground truth.* Violated: unit-test graders admit reward hacking and false negatives; agreement between grader and careful human review is itself uncertain.
3. *Outcome observed for all $p$.* Violated whenever the forecast gates execution — censoring is by construction.
4. *Stationarity.* Violated: scaffold, tool versions, and model checkpoints change between the calibration set and deployment.
5. *Binary success.* Violated for partial-credit tasks; forcing binarization moves calibration error into the threshold choice.

## 3. State of the Art

**Established.**
- Verbalized confidence beats raw sequence likelihood for calibration on short-form QA in RLHF'd models. Tian et al. (EMNLP 2023) report verbalized numeric confidence with lower ECE than conditional-probability readouts on GPT-4-class models across TriviaQA/TruthfulQA-style sets.
- Self-evaluation $P(\text{True})$ is reasonably calibrated on single-step tasks and improves with scale (Kadavath et al., 2022, up to 52B parameters).
- Post-hoc recalibration (temperature scaling, Guo et al., ICML 2017) reliably reduces ECE *in-distribution* and does nothing under shift.
- Task duration for human experts predicts agent success with a logistic law; METR (2025) fit $H_{50}$ across models and report a doubling time of roughly 7 months over 2019–2025.

**Claimed but unablated.**
- That an agent's mid-trajectory self-assessment is a usable stopping signal. Reported in scaffolds and agent papers as an ablation-free design choice; no paper isolates the counterfactual (abort-on-low-confidence vs. abort-at-fixed-budget) with matched compute.
- That LLM-judge confidence transfers across task families. Cross-family evaluation is nearly always absent.

**Benchmark number only.** Every published agentic "calibration" figure is a single ECE or AUROC on one benchmark split (SWE-bench Verified, $\tau$-bench, AgentBench, RE-Bench), typically $n \le 500$, without a bias-corrected estimator, without a per-family breakdown, and without an interval. Treat these as descriptive, not as evidence of calibration.

**Theory SOTA.** Asymptotic calibration is achievable by randomized forecasting against adversarial sequences (Foster–Vohra, *Biometrika* 1998). Distance-to-calibration has a clean theory of equivalent measures (Błasiok, Gopalan, Hu, Nakkiran, STOC 2023). Multicalibration (Hébert-Johnson et al., ICML 2018) gives calibration simultaneously over a family of subgroups with sample complexity polynomial in the family's complexity — the right formal target for "calibrated per task family," and not yet applied to agents.

## 4. What Is Known

- **Scale helps self-knowledge.** $P(\text{IK})$ calibration improves monotonically from 800M to 52B parameters (Kadavath et al., 2022); measured on short QA, not agentic rollouts.
- **RLHF degrades likelihood calibration.** The GPT-4 technical report (OpenAI, 2023) shows a pre-trained checkpoint near-calibrated on MMLU and the post-RLHF model substantially miscalibrated on the same items — the single cleanest published before/after, at frontier scale.
- **ECE is estimator-sensitive.** Kumar, Liang, Ma (NeurIPS 2019) show plugin ECE understates true calibration error for continuous-output recalibrators and give a debiased estimator; the fix matters most exactly in the small-$n$ regime agentic evaluation lives in.
- **Long-horizon success falls off predictably in duration.** METR (2025) measure $H_{50}$ on 170 tasks spanning seconds to ~8 hours of human time; success at tasks above ~4 hours human-equivalent was near zero for the models tested, and the logistic-in-$\log\ell$ fit explains most cross-task variance. This makes a *task-length* base rate a strong, cheap calibration baseline.
- **Agent evaluation is noisy at the scale used.** Kapoor et al. (2024) document that headline agent accuracies are unstable under cost control and re-runs; single-seed differences of several points on SWE-bench-scale sets are within run-to-run noise.

## 5. What Is Not Known

- **Empirically open.** Whether any forecast head beats the two cheap baselines — (a) global base rate, (b) logistic in $\log$(estimated human task time) — on held-out *task families*, at $n$ large enough for a confidence interval narrower than the effect. Runnable today; nobody has paid for it. Requires ~2,000 graded long rollouts.
- **Empirically open.** Whether mid-trajectory forecasts are *sharper* than pre-execution ones by more than grader noise, and where in the trajectory the information arrives.
- **Methodologically blocked.** No agreed calibration metric under partial credit, budget-varying success, and grader error. With grader false-negative rate $\varepsilon$, observed calibration error and true calibration error differ by an amount of order $\varepsilon$ that no published agent result estimates.
- **Theoretically open.** Calibration guarantees under decision-dependent censoring: if forecasts gate execution, low-$p$ bins are never populated, and the standard online-calibration constructions do not apply. No known algorithm attains vanishing calibration error in this feedback model.
- **Theoretically open.** Sample complexity of multicalibration over the *combinatorially large* family of agentic task attributes (repo, tool, length, language) at realistic $n$.

## 6. Why It Is Hard

The specific obstruction is **cost-limited sample size against an outcome that is itself noisy**. To resolve a 5-point ECE difference at 95% confidence for binary outcomes you need $n$ on the order of $10^3$ independent tasks. One long-horizon task is 10 minutes to 8 hours of tool-calling plus a grader that must be written, so a family of 1,000 tasks is a multi-month engineering program before a single forecast is scored. Compounding it:

- **Confounded measurement.** Task-family clustering means effective $n$ is the number of *families*, often 10–30, not the number of tasks. Reported intervals that assume independence are too narrow by roughly $\sqrt{1 + (\bar{m}-1)\rho}$ for intra-family correlation $\rho$.
- **Absent ground truth.** The grader is a proxy. Reward hacking makes $y=1$ without task completion; brittle tests make $y=0$ despite it.
- **Evaluation that does not measure what it names.** "Agent calibration" numbers are usually accuracy-of-self-report on the *same* benchmark used for selection, so recalibration on that split is fitting, not generalization.

## 7. Current Research (as of 2026)

- **Horizon-law measurement.** METR continues extending time-horizon estimates and task suites; the logistic-in-duration model is the de facto reference curve *(frontier — verify current $H_{50}$ values, which move every few months)*.
- **Verbalized and probe-based uncertainty.** Follow-ons to Lin–Hilton–Evans (TMLR 2022), Tian et al. (EMNLP 2023), and Xiong et al. (ICLR 2024) are being pushed from QA onto multi-step tool use; results so far are single-benchmark.
- **Conformal wrappers for agents.** Split-conformal (Vovk et al., 2005; Angelopoulos–Bates, 2023) gives distribution-free coverage for "will this trajectory succeed" as a set-valued abstention rule, at the price of exchangeability — which task-family shift violates. Active *(frontier — verify)*.
- **Multicalibration for LLM evaluation.** Early work applying Hébert-Johnson-style subgroup calibration to model evaluation; not yet on agentic rollouts *(frontier — verify)*.
- **Cost-controlled agent benchmarking.** Kapoor et al. and successors push Pareto reporting (accuracy vs. dollars), which is the substrate any calibration study needs.

## 8. Concrete Next Experiment

**Question.** Does any learned success forecaster beat the task-duration base rate out-of-family?

**Scale.** 40 task families × 50 tasks = 2,000 long-horizon tasks, human-time-labeled, spanning 5 minutes to 6 hours. One frontier model, one fixed scaffold, budget $B$ fixed at 4× median human time. 3 seeds per task on a 400-task subset to estimate grader/run noise. Estimated cost: order $10^4$ USD of inference plus grading.

**Arms.**
1. **Control (must be beaten):** $\hat p = \sigma(a + b\log \ell)$, fit on training families, where $\ell$ is estimated human task time. No model introspection.
2. Verbalized pre-execution forecast, temperature-scaled on training families.
3. Linear probe on hidden state at $t = 0$ and at $t = 0.5T$.
4. $P(\text{True})$ self-evaluation at $t = 0.5T$.

**Split.** Leave-8-families-out, 5 folds. Never split within a family.

**Deciding number.** Difference in **debiased ECE** (Kumar–Liang–Ma estimator, 15 equal-mass bins) between the best model-based arm and the control, on held-out families, with a family-level bootstrap 95% interval. Decision rule: an interval excluding zero with a point difference $\ge 0.03$ absolute ECE says model introspection carries out-of-family calibration signal. An interval containing zero at this $n$ says the duration prior is all that is currently measurable — a publishable negative. Report Brier alongside so a sharpness gain masked by miscalibration is visible.

## 9. Key References

- **[Foundational]** A. P. Dawid. *The Well-Calibrated Bayesian.* Journal of the American Statistical Association, 1982.
- **[Foundational]** D. Foster, R. Vohra. *Asymptotic Calibration.* Biometrika, 1998.
- **[Foundational]** T. Gneiting, A. Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA, 2007.
- **[Foundational]** C. Guo, G. Pleiss, Y. Sun, K. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Theory]** A. Kumar, P. Liang, T. Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[Theory]** Ú. Hébert-Johnson, M. Kim, O. Reingold, G. Rothblum. *Multicalibration: Calibration for the (Computationally-Identifiable) Masses.* ICML, 2018.
- **[Theory]** J. Błasiok, P. Gopalan, L. Hu, P. Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC, 2023. — arXiv:2211.16886
- **[SOTA]** S. Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[SOTA]** K. Tian, E. Mitchell, A. Zhou, A. Sharma, R. Rafailov, H. Yao, C. Finn, C. Manning. *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* EMNLP, 2023. — arXiv:2305.14975
- **[SOTA]** S. Lin, J. Hilton, O. Evans. *Teaching Models to Express Their Uncertainty in Words.* TMLR, 2022. — arXiv:2205.14334
- **[SOTA]** M. Xiong, Z. Hu, X. Lu, Y. Li, J. Fu, J. He, B. Hooi. *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.* ICLR, 2024. — arXiv:2306.13063
- **[Measurement]** T. Kwa, B. West, J. Becker, et al. (METR). *Measuring AI Ability to Complete Long Tasks.* 2025. — arXiv:2503.14499
- **[Measurement]** C. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, K. Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Measurement]** S. Yao, N. Shinn, P. Razavi, K. Narasimhan. *$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[Survey/critique]** S. Kapoor, B. Stroebl, Z. Siegel, N. Nadgir, A. Narayanan. *AI Agents That Matter.* 2024. — arXiv:2407.01502
- **[Survey]** A. Angelopoulos, S. Bates. *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in Machine Learning, 2023.

## 10. Worked Example

Take a 200-task agentic set, 20 repositories × 10 tasks, one frontier model, observed overall success 0.40. The agent verbalizes $p_0$ per task. Plugin ECE with 10 equal-width bins comes out at 0.061 — reported as "well calibrated."

Now do the arithmetic the report skips.

**Bin occupancy.** With 200 tasks over 10 bins, average bin has 20 items; the mass concentrates in the $[0.6,0.8]$ bins (models are over-confident), so several bins hold $\le 5$. For a bin with $n_m = 5$ and true rate $q=0.4$, the sampling s.d. of $\mathrm{acc}(B_m)$ is $\sqrt{0.4 \cdot 0.6/5} = 0.219$. Its expected absolute deviation from $q$ is about $0.8 \times 0.219 \approx 0.175$ **even if the forecaster is perfect**. Weighted across bins, the noise floor of plugin ECE here is roughly $\sqrt{\bar{q}(1-\bar q)/\bar{n}_m} \cdot 0.8 \approx 0.8\sqrt{0.24/20} \approx 0.088$.

**The measured 0.061 is below the noise floor of its own estimator.** It is not evidence of calibration; it is evidence that $n$ is too small to distinguish this forecaster from a perfect one *or* from one with 0.09 true ECE.

**Clustering makes it worse.** Effective sample size is not 200. With 10 tasks per repo and intra-repo outcome correlation $\rho = 0.3$, the design effect is $1 + (10-1)(0.3) = 3.7$, so $n_{\text{eff}} \approx 200/3.7 \approx 54$. Redo the floor with $\bar n_m \approx 5.4$: noise floor $\approx 0.8\sqrt{0.24/5.4} \approx 0.169$.

**Grader error on top.** A 5% false-negative rate on tests shifts every bin's accuracy down by ~0.05 · (bin accuracy), adding a systematic ~0.02 to ECE in the high-confidence bins where it does the most damage.

**Conclusion the number cannot support.** Any true ECE between 0 and about 0.17 is consistent with the observation. To separate a 0.03 improvement from zero you need roughly $n_{\text{eff}} \sim 10^3$, i.e. ~40 families of 50 tasks under the same $\rho$ — which is exactly the design in §8. The obstruction is not that agents are hard to introspect. It is that at the sample sizes long-horizon evaluation can afford, the calibration statistic has a noise floor larger than every effect anyone has claimed to measure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*