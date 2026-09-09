---
id: 26-code-generation/maintainability-machine-generated-code
title: "Maintainability of Machine-Generated Code Over Time"
topic: 26-code-generation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Maintainability of Machine-Generated Code Over Time

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/maintainability-machine-generated-code` · **Status:** methodologically-blocked

## 1. Problem Statement

**Input.** A repository whose commits carry a provenance label: which lines were authored by a model, which by a human, which jointly.

**Output.** An estimate of the causal effect of model authorship on the cost of *future* changes to that repository — bug fixes, feature additions, dependency migrations, security patches — over horizons of months to years.

**Decision predicate.** Does raising the model-authored fraction of a codebase from $\alpha$ to $\alpha'$ increase the expected engineer-hours per future change request, holding functionality and team constant?

Three variants, with very different difficulty:

- **Measurement.** Define a maintainability quantity that is (a) predictive of real future effort and (b) not a restatement of code size. This is the blocked variant. Existing proxies — Maintainability Index, code smell counts, cyclomatic complexity, churn — either fail to predict effort once size is controlled, or are themselves altered by the generator in ways unrelated to effort.
- **Method.** Given a working measurement, build generators or review pipelines that reduce it. Runnable today against proxies; unvalidated because the proxies are.
- **Theory.** Is there a formal reason a next-token-trained generator produces code with worse *long-horizon* editability than short-horizon correctness would suggest? No proof either way.

Solving it means: a preregistered, provenance-controlled study reporting an effort ratio with a confidence interval, replicated on a second codebase.

## 2. Formal Setting

Let a repository be a sequence of states $R_0, R_1, \dots$ under commits $c_t$. Each line $\ell$ carries provenance $p(\ell) \in \{\text{human}, \text{model}, \text{joint}\}$. Define the **model-authored fraction**

$$\alpha(R_t) = \frac{\sum_{\ell \in R_t} \mathbb{1}[p(\ell) = \text{model}]}{|R_t|}.$$

*Measured as:* IDE telemetry logging accepted completions plus post-acceptance edit distance, or commit trailers. Both are lossy — a human who edits three tokens of a 40-line suggestion is unclassifiable, and the joint class typically dominates.

A **change request** $q \sim \mathcal{Q}$ (feature, fix, migration) is resolved at cost $C(q, R_t) \in \mathbb{R}_{>0}$, engineer-hours to a merged, review-passing, test-passing patch. *Measured as:* wall-clock from first-touch to merge minus idle gaps, or task-timer instrumentation in a controlled study. Both are noisy at $\sigma/\mu \approx 1$ per task.

**Maintainability** of a state is the expected cost over the future request distribution:

$$M(R_t) = \mathbb{E}_{q \sim \mathcal{Q}_t}\left[ C(q, R_t) \right], \qquad \text{normalized: } \tilde{M}(R_t) = M(R_t) / |R_t|.$$

The estimand is the contrast under an intervention on authorship, with the counterfactual repository $R_t^{\alpha'}$ that implements the same functional specification:

$$\Delta(\alpha \to \alpha') = \mathbb{E}\left[ M(R_t^{\alpha'}) - M(R_t^{\alpha}) \right].$$

Proxies stand in for $M$: Maintainability Index $\mathrm{MI} = 171 - 5.2\ln V - 0.23 G - 16.2 \ln L$ (Halstead volume $V$, cyclomatic complexity $G$, lines $L$; Coleman et al. 1994); smell counts from static analysis; duplication rate (fraction of lines inside $k$-line clone blocks, $k=5$ typical); **churn** $\kappa_w$ = fraction of added lines deleted or rewritten within $w$ days.

Assumptions, with the ones known violated flagged:

1. $\mathcal{Q}_t$ is stationary. **Violated** — request mix shifts as a product matures.
2. Provenance is observable and stable. **Violated** — labels degrade under refactoring; a model-written function later hand-edited retains the model label in most telemetry schemes.
3. Proxy monotonicity: $\mathrm{MI}\downarrow \Rightarrow M\uparrow$. **Violated** — see §4; smells lose significance once file size is controlled.
4. No selection on task difficulty. **Violated** — developers route easy, boilerplate-shaped tasks to the model, so $\alpha$ correlates with task simplicity.
5. Unit-of-analysis independence. **Violated** — commits within a file are strongly correlated.

## 3. State of the Art

**Established.**

- *Short-horizon correctness* is measured well. HumanEval (Chen et al., 2021) and its de-flaked successor EvalPlus (Liu et al., NeurIPS 2023) give reproducible pass@$k$; SWE-bench (Jimenez et al., ICLR 2024) gives repository-level patch resolution. None of these measure anything downstream of the merge.
- *Short-horizon productivity* has a clean RCT: Peng et al. (2023), 95 developers, single HTTP-server task, 55.8% faster with Copilot. It does not extend to maintenance.
- *The direction reverses on real repositories.* METR (2025) ran an RCT with 16 experienced open-source maintainers over 246 tasks in their own mature repositories: AI-allowed tasks took **19% longer**, while the same developers predicted a 24% speedup. This is the strongest evidence that self-report and proxy metrics diverge from measured effort.

**Claimed but unablated.**

- GitClear's industry reports (2024, 2025) analyze roughly 153M and later ~211M changed lines and report rising short-window churn and a large increase in duplicated 5-line blocks alongside falling "moved" (refactored) lines. Provenance is inferred from calendar year and org-level tool adoption, not per-line labels; there is no control arm. Treat as hypothesis-generating.
- Vendor and consultancy claims that assistants "improve code quality" rest on MI or smell deltas — proxies whose link to effort is the disputed object.

**Security is the one adjacent measurement that works.** Pearce et al. (IEEE S&P 2022) found ~40% of Copilot completions in security-relevant scenarios contained a CWE; Perry et al. (CCS 2023), $n=47$, found AI-assisted participants wrote less secure code *and* were more confident it was secure. Security has ground truth (a CWE either is or is not present); maintainability does not.

## 4. What Is Known

- **Smells do not predict effort once size is controlled.** Sjøberg et al. (IEEE TSE 39(8), 2013) instrumented six professional developers maintaining four functionally equivalent Java systems across ~3,148 recorded work hours. Of 12 smell types, most had no significant effect on maintenance effort after adjusting for file size; some were associated with *less* effort. Scale: four systems, industrial developers, real tasks.
- **File size is the dominant confounder.** In the same study and in Yamashita & Moonen (ICSM 2012), lines of code absorbs most of the variance any smell metric claims.
- **Defect labels are ~1/3 wrong.** Herzig, Just & Zeller (ICSE 2013) manually reclassified over 7,000 issue reports across five projects: 33.8% were misclassified, and 39% of files marked defect-prone had never had a bug fixed in them. Any maintainability study keyed on issue trackers inherits this noise.
- **Relative churn predicts defect density.** Nagappan & Ball (ICSE 2005), Windows Server 2003 — relative churn measures explain a large share of system defect density variance. Churn is therefore a defensible *outcome*, and a poor *treatment-free* proxy for quality.
- **Effort distributions are heavy-tailed.** Per-task engineer-hours routinely show $\sigma \approx \mu$; METR's 246-task RCT needed that many tasks to resolve a 19% effect.
- **Lehman's laws** (Proc. IEEE, 1980): complexity increases unless work is explicitly done to reduce it. The AI-specific claim is that generators shift the balance between the two forces — this remains unmeasured.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no validated maintainability measurement independent of code size. Every candidate — MI, smells, duplication, entropy — either correlates with $L$ or has been falsified as an effort predictor. Until a proxy is validated against instrumented effort, $\Delta(\alpha \to \alpha')$ cannot be estimated from repository mining at all.
- **Empirically open.** The provenance-labelled longitudinal cohort does not exist. Nobody has followed matched repositories with per-line authorship labels for 18+ months and measured effort per merged change. Runnable now; requires instrumentation, not new science.
- **Theoretically open.** Whether next-token objectives systematically under-weight properties that only pay off across future edits (abstraction reuse, naming consistency, interface stability) has no formal statement, let alone a proof. There is no known separation theorem between "correct on the test suite" and "cheap to change."
- **Unknown mechanism split.** If an effect exists, it may be the code, the *reviewer*'s reduced comprehension of code they did not write, or task-routing selection. These are not currently identifiable.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability of provenance**.

1. *The named quantity is not the measured one.* "Maintainability" scores measure size and syntactic shape. Sjøberg et al. showed the link to effort fails empirically. An evaluation that does not measure the thing it names produces confidently wrong deltas in either direction.
2. *Provenance decays.* After two rounds of human editing, the model-authored fraction of a line is not defined. $\alpha$ is a fuzzy treatment, so the causal contrast has no sharp treatment arm.
3. *Selection dominates.* Developers accept suggestions for boilerplate and reject them for hard code. $\alpha$ is therefore an inverse proxy for task difficulty, biasing observational estimates toward "AI code is easy to maintain."
4. *Cost, not compute.* The blocking resource is instrumented engineer-hours. Detecting a 15% effort difference with $\sigma/\mu \approx 1$ needs on the order of 700 tasks per arm — roughly 10–20 engineer-years if tasks average a day.

## 7. Current Research (as of 2026)

- **Instrumented RCTs on real repositories.** METR's design — experienced maintainers, own codebases, randomized per task — is the template others are extending to longer horizons *(frontier — verify)*.
- **Provenance standards.** Commit-trailer and SPDX-style AI-contribution metadata proposals are circulating in open-source governance; adoption is thin *(frontier — verify)*.
- **Agentic repository benchmarks.** SWE-bench successors (multi-file, longer-horizon, verified variants) measure resolution rate but still terminate at merge; extending them to *sequences* of dependent tasks on the same repo is an active direction.
- **Empirical software engineering groups** (Zeller/Just lineage on label noise; Nagappan/Bird lineage on repository mining at Microsoft) supply the methodological corrections any credible study must adopt.
- **Static-analysis vendors** market "AI code quality" dashboards built on smells and duplication; none published a validation against effort.

## 8. Concrete Next Experiment

**Question.** Does model-authored code cost more engineer-hours per subsequent change than human-authored code in the same repository?

**Design — retrospective task replay with a within-repository control.**

- **Scale.** Five mature repositories, 100k–500k LOC, with 18 months of per-line provenance from IDE telemetry. Sample 800 closed change requests, stratified so that half touch files with $\alpha > 0.5$ and half $\alpha < 0.1$, **matched on file size, file age, prior churn, and request type** (the four confounders in §6).
- **Control arm.** Human-authored files matched by propensity score on those four covariates. Second control: pre-adoption tasks from the same repositories, same matching, to absorb team-level drift.
- **Outcome.** Instrumented engineer-hours to merged, review-approved patch, plus 30-day post-merge churn $\kappa_{30}$ as a secondary.
- **Validation sub-study (run first).** On a 150-task subset, regress measured hours on MI, smell count, and duplication with $\ln L$ included. If no proxy retains significance, publish that as the finding — it closes the methodological question and invalidates every dashboard built on those proxies.
- **The deciding number.** The size-adjusted effort ratio $\hat{\rho} = \tilde{M}_{\text{model}} / \tilde{M}_{\text{human}}$ with a bootstrap 95% CI. $\hat{\rho} > 1.15$ with the interval excluding 1 is a real maintainability tax; an interval inside $[0.95, 1.05]$ retires the concern at this scale. 800 matched tasks give ~80% power at $\rho = 1.15$ under $\sigma/\mu = 1$.

## 9. Key References

- **[Foundational]** Lehman, M. M. *Programs, Life Cycles, and Laws of Software Evolution.* Proceedings of the IEEE, 1980.
- **[Foundational]** Cunningham, W. *The WyCash Portfolio Management System.* OOPSLA Experience Report, 1992.
- **[Foundational]** Coleman, D., Ash, D., Lowther, B., Oman, P. *Using Metrics to Evaluate Software System Maintainability.* IEEE Computer, 1994.
- **[Key negative result]** Sjøberg, D. I. K., Yamashita, A., Anda, B. C. D., Mockus, A., Dybå, T. *Quantifying the Effect of Code Smells on Maintenance Effort.* IEEE Transactions on Software Engineering 39(8), 2013.
- **[Methodology]** Herzig, K., Just, S., Zeller, A. *It's Not a Bug, It's a Feature: How Misclassification Impacts Bug Prediction.* ICSE, 2013.
- **[Methodology]** Nagappan, N., Ball, T. *Use of Relative Code Churn Measures to Predict System Defect Density.* ICSE, 2005.
- **[Methodology]** Yamashita, A., Moonen, L. *Do Code Smells Reflect Important Maintainability Aspects?* ICSM, 2012.
- **[SOTA — effort]** METR. *Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity.* Technical report, 2025.
- **[SOTA — productivity]** Peng, S., Kalliamvakou, E., Cihon, P., Demirer, M. *The Impact of AI on Developer Productivity: Evidence from GitHub Copilot.* arXiv, 2023. — arXiv:2302.06590
- **[SOTA — correctness]** Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[SOTA — correctness]** Liu, J., Xia, C. S., Wang, Y., Zhang, L. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS, 2023. — arXiv:2305.01210
- **[Adjacent — security]** Pearce, H., Ahmad, B., Tan, B., Dolan-Gavitt, B., Karri, R. *Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions.* IEEE S&P, 2022.
- **[Adjacent — security]** Perry, N., Srivastava, M., Kumar, D., Boneh, D. *Do Users Write More Insecure Code with AI Assistants?* ACM CCS, 2023.
- **[Foundational — capability]** Chen, M. et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374
- **[Industry report, uncontrolled]** GitClear. *Coding on Copilot / AI Copilot Code Quality* research reports, 2024 and 2025.

## 10. Worked Example

A 220k-LOC Python service. Telemetry over 14 months labels 31% of added lines as model-authored. The team's dashboard reports:

| Cohort | Files | Median LOC | Mean MI | Smells/KLOC | 5-line clone rate |
|---|---|---|---|---|---|
| $\alpha > 0.5$ | 412 | 148 | 71.4 | 4.1 | 12.8% |
| $\alpha < 0.1$ | 906 | 263 | 63.9 | 6.7 | 5.2% |

Naive read: model-heavy files score **7.5 MI points better** and carry 39% fewer smells. Ship more AI.

Now apply the size term. $\mathrm{MI}$ contains $-16.2 \ln L$. From $L = 263$ to $L = 148$:

$$\Delta \mathrm{MI}_{\text{size}} = -16.2\,(\ln 148 - \ln 263) = -16.2 \times (-0.575) = +9.3.$$

The size term alone predicts $+9.3$ MI. The observed gap is $+7.5$. Once file length is held fixed, model-heavy files are **1.8 MI points worse**, not 7.5 better. The dashboard measured the fact that models emit shorter files.

Smells/KLOC moves the same way — it is a density, and Sjøberg et al. found smell effects vanish under size adjustment. The one metric pointing the other direction, clone rate at 12.8% vs 5.2%, is the one with no validated link to effort at all.

Then the effort data arrives: 96 matched change requests, 48 per cohort.

$$\tilde{M}_{\text{model}} = 5.9\ \text{h/task}, \quad \tilde{M}_{\text{human}} = 5.4\ \text{h/task}, \quad \hat{\rho} = 1.09,\ \text{95\% CI } [0.81,\ 1.47].$$

The interval spans both "12% cheaper" and "47% more expensive." Three metrics gave three confident answers in two directions; the one measurement that actually names effort is uninformative at $n = 96$. That is the obstruction — not that the answer is bad news, but that the cheap measurements are decided by file length and the honest one needs roughly 800 tasks to say anything.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*