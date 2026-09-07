---
id: 22-safety-robustness/adaptive-attack-evaluation-bias
title: "Adaptive Attack Evaluation Without Attacker Adaptation Bias"
topic: 22-safety-robustness
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adaptive Attack Evaluation Without Attacker Adaptation Bias

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/adaptive-attack-evaluation-bias` · **Status:** methodologically-blocked

## 1. Problem Statement

A defense is evaluated by running attacks against it. The attacks are written by people, with finite effort, and their effort is not uniformly distributed: they try hard against defenses they expect to break and stop early against defenses that look solid. The reported robust accuracy is therefore a function of *attacker effort allocation*, not only of the defense. Call this **attacker adaptation bias**: the gap between measured robustness $\hat{R}$ and worst-case robustness $R^\star$ over the true threat model, where the gap is determined by an unobserved and defense-dependent human variable.

Three variants, with different difficulty:

- **Measurement.** Produce an estimator of $R^\star$ whose bias is bounded, or at least reported, given a fixed budget of attacker effort. Currently no accepted estimator exists — this is the blocked variant.
- **Method.** Build an attack search procedure whose success does not depend on a human noticing the right defense-specific trick (e.g. replacing a non-differentiable component with a smooth surrogate).
- **Theory.** Prove, for some non-trivial defense class, an upper bound on $R^\star$ that no attack can beat — certification. Solved for restricted classes ($\ell_p$ balls, randomized smoothing), open for the defenses people actually deploy (LLM guardrails, filters, agent sandboxes).

Solving it means: two independent red teams, given the same effort budget and the same defense, report robust accuracies within a stated tolerance, and neither is later falsified by a third team.

## 2. Formal Setting

Let $f_\theta$ be a defended system mapping input $x \in \mathcal{X}$ to output. Let $\mathcal{T}$ be the threat model: a set-valued map $\mathcal{T}(x) \subseteq \mathcal{X}$ of admissible perturbations, plus an oracle $\mathrm{harm}(y) \in \{0,1\}$ judging whether an output is a successful attack.

**Worst-case robustness** on distribution $\mathcal{D}$:

$$R^\star(f_\theta) = \mathbb{E}_{x\sim\mathcal{D}}\Big[\,1 - \max_{x' \in \mathcal{T}(x)} \mathrm{harm}\big(f_\theta(x')\big)\Big].$$

**Measured robustness** under an attack portfolio $\mathcal{A} = \{A_1,\dots,A_k\}$ with per-example compute $c$:

$$\hat{R}_{\mathcal{A},c}(f_\theta) = \mathbb{E}_{x}\Big[\,1 - \max_{i\le k}\mathrm{harm}\big(f_\theta(A_i(x; \theta, c))\big)\Big] \;\ge\; R^\star(f_\theta).$$

The bias is $B = \hat{R}_{\mathcal{A},c} - R^\star \ge 0$, one-sided: measurement always overstates robustness.

How each quantity is actually measured:

- $\mathrm{harm}$: a human rubric, a classifier (HarmBench's fine-tuned Llama-2-13B judge), or string matching for refusal prefixes. Judge disagreement with humans is typically 5–15 points of attack success rate (ASR).
- $c$: reported as attack iterations or queries. Human hours spent designing $A_i$ are **not** reported anywhere, and that is the dominant term.
- $\mathcal{T}$: for vision, $\|x'-x\|_\infty \le 8/255$ — crisp. For LLMs, "any prompt a user could send" — not a set anyone can enumerate, so $\max_{x'\in\mathcal{T}(x)}$ is not computable even in principle.

Assumptions and their status:

| Assumption | Status |
|---|---|
| $\mathcal{T}$ is explicitly specified | Holds for $\ell_p$ vision; **violated** for LLM/agent defenses |
| $\mathrm{harm}$ is a fixed oracle | **Violated** — judge model version changes ASR by several points |
| Attack effort is comparable across defenses | **Violated** — this is the bias itself |
| $\hat{R}$ is monotone in $c$ | Holds empirically; no proof of convergence rate |

## 3. State of the Art

**Established.** Adaptive-attack methodology is the accepted standard, not an option. Athalye, Carlini & Wagner (ICML 2018) broke 7 of 9 ICLR 2018 defenses by identifying obfuscated gradients (BPDA, EOT, reparameterization). Tramèr, Carlini, Brendel & Madry (NeurIPS 2020) broke all 13 defenses they examined, each published after adaptive-evaluation guidance existed. AutoAttack (Croce & Hein, ICML 2020) — an ensemble of APGD-CE, APGD-T, FAB-T and Square Attack — is a fixed, hyperparameter-free portfolio and is the RobustBench standard.

**Claimed but unablated.** That automated ensembles substitute for human adaptivity. AutoAttack is a *lower bound on breakage* by construction; it has itself been beaten on individual defenses by hand-built attacks. The claim "our defense survives AutoAttack" is a benchmark number, not evidence about $R^\star$.

**Benchmark-number-only results.** LLM jailbreak leaderboards (HarmBench, JailbreakBench, AgentDojo) report ASR against a fixed attack list. When the defense is public, later attacks routinely move the number by tens of points; the leaderboard rank at time $t$ has no demonstrated predictive value for rank at $t+1$.

**Theory SOTA.** Randomized smoothing (Cohen, Rosenfeld & Kolter, ICML 2019) gives certificates immune to attacker effort entirely — but only for $\ell_2$ balls, at a large clean-accuracy cost. There is no certification for semantic or natural-language threat models.

## 4. What Is Known

- **Adaptive attacks reduce reported robustness to near zero for most defenses.** Tramèr et al. (2020): 13/13 defenses, published robust accuracies typically 40–60% on CIFAR-10 at $\epsilon=8/255$, driven below the authors' claims — several to $0\%$. Scale: CIFAR-10, ResNet-scale models.
- **Genuine robustness moves slowly.** RobustBench CIFAR-10 $\ell_\infty$ $\epsilon=8/255$ leaderboard went from ~53% (Madry-style adversarial training, 2018) to roughly 70–73% AutoAttack accuracy by 2023–2024, achieved mainly by adding generated data and scale — no defense-mechanism trick has stuck.
- **Effort is the covariate.** Carlini's LLM-assisted break of AI-Guardian (2023) reduced a claimed 8% attack success to 98% — same defense, same threat model, more attacker attention.
- **Simple adaptivity beats sophisticated automation.** Andriushchenko, Croce & Flammarion (2024) reached ~100% ASR on several leading safety-tuned LLMs using random search over an adversarial suffix plus a hand-chosen prompt template — a method with almost no machinery, but with a human in the loop choosing the template.
- **Scaling attack samples alone works.** Best-of-$N$ jailbreaking (Hughes et al., 2024) reports ASR climbing with sampled augmentations under a power-law-like trend, meaning any single reported ASR is a point on a curve whose $x$-axis (budget) is usually unstated.
- **Judges are a measurement channel.** HarmBench (Mazeika et al., ICML 2024) documents materially different ASRs across classifiers on identical generations.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No accepted way to measure or report attacker effort. Human design hours, prior familiarity with the defense, and number of abandoned attack ideas are never logged, so $B$ cannot be estimated. Every headline robustness number is conditioned on an unrecorded variable.
- **Methodologically blocked.** For LLMs, $\mathcal{T}$ has no formal definition, so $R^\star$ is not a well-defined target. Debate over "is this prompt in-distribution for the threat model" is not resolvable by experiment.
- **Empirically open.** Whether a pre-registered, effort-matched red-team protocol yields reproducible ASR across independent teams. Runnable today with 4–6 teams; nobody has run it.
- **Theoretically open.** Whether any computable statistic of $(f_\theta, \mathcal{A}, c)$ upper-bounds $B$ without certification. Related to the fact that verifying ReLU network properties is NP-complete (Katz et al., CAV 2017), which rules out cheap exact verification but not bounded-bias estimation.

## 6. Why It Is Hard

**The obstruction is non-identifiability plus a one-sided, unlogged confound.** $\hat{R}$ is jointly determined by defense quality and attacker effort; only $\hat{R}$ is observed and only the defense is varied. Two defenses with the same $R^\star$ report different $\hat{R}$ if one attracted more attention, and no post-hoc statistic distinguishes the cases. The confound is one-sided — no attack can *understate* robustness — so errors never cancel across a leaderboard; they accumulate in one direction. Worse, effort is anti-correlated with reported robustness: a defense that looks strong gets attacked less, which makes it look stronger. The standard fix for confounding, randomization, is unavailable because you cannot randomize how hard a human tries, and the attacker's knowledge of the defense is exactly what makes the attack adaptive.

## 7. Current Research (as of 2026)

- **Fixed standardized portfolios.** RobustBench/AutoAttack (Croce, Hein, EPFL/Tübingen) as a floor on breakage; explicitly not a claim about $R^\star$.
- **Compute-scaled automated red-teaming.** Attack success reported as a curve in queries rather than a single number — the Best-of-$N$ line of work (Anthropic and collaborators) and GCG-style optimization (Zou et al., CMU).
- **Public bounty red-teaming with logged budget.** Anthropic's constitutional-classifiers program (Sharma et al., 2025) reported ~3,000 hours of human red-teaming with no universal jailbreak found during the program, then subsequent public breaks — the clearest natural experiment on effort-dependence to date.
- **Agentic threat models.** AgentDojo (Debenedetti et al., NeurIPS 2024 Datasets & Benchmarks) makes $\mathcal{T}$ partially explicit for tool-using agents by fixing injection sites.
- *(frontier — verify)* Pre-registration and effort-logging norms for red teams, and third-party audit protocols under EU AI Act / NIST-adjacent evaluation guidance. Proposed in policy drafts; no published empirical study of inter-team reproducibility.

## 8. Concrete Next Experiment

**Inter-rater reliability for red teams.**

- **Scale.** 6 independent red teams (3–4 people each), 40 hours per team, 4 defended LLM systems held fixed and given to all teams: (a) an aligned base model, (b) base + input/output classifier, (c) base + paraphrase preprocessing, (d) base + a *deliberately weak* filter (regex refusal-keyword blocklist). Same 200-behavior harm set, same frozen judge checkpoint, same query cap ($10^4$ model calls per behavior).
- **Control arm.** Defense (d), whose $R^\star$ is known to be near zero by construction — every team must find it. Any team reporting non-trivial robustness for (d) reveals its own effort floor, calibrating that team's readings on (b) and (c).
- **Deciding number.** The between-team standard deviation of ASR on defense (b), $\sigma_{\text{between}}$, compared with the within-team standard deviation across three disjoint 100-behavior halves, $\sigma_{\text{within}}$. If $\sigma_{\text{between}} \le 5$ ASR points and $\sigma_{\text{between}}/\sigma_{\text{within}} \le 1.5$, effort-matched protocols make red-team numbers a measurement. If $\sigma_{\text{between}} \ge 20$ points, single-team ASR should not be published as a scalar and the field must report effort-conditioned curves instead.
- **Cost.** ~1,440 human hours plus inference; roughly one large-lab quarter, or one shared multi-institution workshop challenge.

## 9. Key References

- **[Foundational]** Carlini, N., Wagner, D. *Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods.* AISec @ CCS, 2017. — arXiv:1705.07263
- **[Foundational]** Athalye, A., Carlini, N., Wagner, D. *Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples.* ICML, 2018. — arXiv:1802.00420
- **[Foundational]** Tramèr, F., Carlini, N., Brendel, W., Madry, A. *On Adaptive Attacks to Adversarial Example Defenses.* NeurIPS, 2020. — arXiv:2002.08347
- **[Methodology]** Carlini, N., Athalye, A., Papernot, N., Brendel, W., Rauber, J., Tsipras, D., Goodfellow, I., Madry, A., Kurakin, A. *On Evaluating Adversarial Robustness.* arXiv preprint, 2019. — arXiv:1902.06705
- **[SOTA]** Croce, F., Hein, M. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks.* ICML, 2020. — arXiv:2003.01690
- **[Benchmark]** Croce, F., Andriushchenko, M., Sehwag, V., Debenedetti, E., Flammarion, N., Chiang, M., Mittal, P., Hein, M. *RobustBench: a standardized adversarial robustness benchmark.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2010.09670
- **[SOTA]** Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J.Z., Fredrikson, M. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv preprint, 2023. — arXiv:2307.15043
- **[SOTA]** Andriushchenko, M., Croce, F., Flammarion, N. *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.* ICLR, 2025 (arXiv 2024). — arXiv:2404.02151
- **[Benchmark]** Mazeika, M. et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML, 2024. — arXiv:2402.04249
- **[Theory]** Cohen, J., Rosenfeld, E., Kolter, J.Z. *Certified Adversarial Robustness via Randomized Smoothing.* ICML, 2019. — arXiv:1902.02918
- **[Theory]** Katz, G., Barrett, C., Dill, D., Julian, K., Kochenderfer, M. *Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.* CAV, 2017.
- **[Case study]** Carlini, N. *A LLM Assisted Exploitation of AI-Guardian.* arXiv preprint, 2023. — arXiv:2307.15008
- **[Frontier]** Sharma, M. et al. *Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming.* arXiv preprint, 2025. — arXiv:2501.18837
- **[Survey]** Wei, A., Haghtalab, N., Steinhardt, J. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS, 2023. — arXiv:2307.02483

## 10. Worked Example

**AI-Guardian, carried end to end.**

- Defense published with a reported attack success rate of **8%** under the authors' evaluation. Read as $\hat{R} = 0.92$.
- One researcher, using an LLM as a coding assistant, spent on the order of a few days reverse-engineering the defense's secret mask and rebuilding the attack around it. Reported attack success: **98%**. Now $\hat{R} = 0.02$.
- The defense parameters $\theta$ did not change. $R^\star$ did not change. The measured quantity moved 90 points.

Bias decomposition for this instance:

$$B_{\text{authors}} = \hat{R}_{\text{authors}} - R^\star \approx 0.92 - 0.02 = 0.90,$$

taking the later attack as the best available proxy for $R^\star$. The obstruction is that $B_{\text{authors}} = 0.90$ was invisible *at publication time*, and no statistic available then — number of attacks tried, iterations per attack, gradient-norm diagnostics — would have flagged it. Compare the constitutional-classifiers case: ~3,000 logged red-team hours produced no universal jailbreak, then public attempts after release surfaced breaks. Both stories have the same shape and differ only in how much effort preceded the published number.

The practical consequence: two numbers, $0.92$ and $0.02$, are both correct reports of $\hat{R}_{\mathcal{A},c}$ under different unrecorded $c$. Until $c$ includes human effort and is reported, a robustness leaderboard ranks defenses partly by how little attention they received.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*