---
id: 26-code-generation/calibration-confidence-program-correctness
title: "Calibration of Model Confidence on Program Correctness"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration of Model Confidence on Program Correctness

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/calibration-confidence-program-correctness` · **Status:** open

## 1. Problem Statement

A code model emits a program $y$ for a specification $x$ and, alongside it, a number $c \in [0,1]$ claimed to be the probability that $y$ is correct. The problem is to make $c$ mean what it says: among all emissions with $c \approx 0.7$, about 70% should in fact be correct.

Three variants, routinely conflated:

- **Measurement.** Correctness is a property of a program against a specification, and specifications in practice are test suites. Any $c$ is calibrated *against whatever oracle you used*. Defining the target event well enough to score $c$ is itself unsolved for anything past self-contained functions.
- **Method.** Given a fixed oracle, produce $c$ that is calibrated and *sharp* (concentrated near 0 and 1). Trivially calibrated predictors exist — emit the base rate always — so calibration alone is not the objective.
- **Theory.** Characterize what is achievable. Distribution-free calibration is impossible for continuous-valued predictors without discretization (Gupta et al., NeurIPS 2020), and correctness against a general specification is undecidable (Rice), so the theory question is which restricted settings admit guarantees.

Solving it means: a deployable estimator whose calibration error is small on *held-out task distributions it was not tuned on*, at useful sharpness, with the oracle stated.

## 2. Formal Setting

Let $x \sim \mathcal{D}$ be a task, $y \sim p_\theta(\cdot \mid x)$ a sampled program, and $S(x)$ a specification. Define the correctness indicator
$$Z = \mathbb{1}[\,y \models S(x)\,] \in \{0,1\}.$$
**As measured**, $\models$ is never semantic entailment; it is a finite test suite $T(x) = \{(i_j, o_j)\}_{j=1}^{m}$ with
$$\hat{Z} = \prod_{j=1}^{m} \mathbb{1}[\,\mathrm{exec}(y, i_j) = o_j\,],$$
executed under a timeout $\tau$ (a timeout scores as failure). $\hat{Z} \geq Z$ pointwise up to flaky tests: weak suites produce false positives.

A confidence estimator is $c = f(x, y, \text{internals}) \in [0,1]$. Candidate instantiations, each measured differently:

- **Sequence likelihood:** $c = \exp\!\big(\tfrac{1}{|y|}\sum_t \log p_\theta(y_t \mid y_{<t}, x)\big)$ — length-normalized, else $c$ collapses with program length.
- **Self-evaluation:** $c = p_\theta(\text{``True''} \mid x, y, \text{prompt})$, the P(True) construction of Kadavath et al. (2022).
- **Verbalized:** $c$ parsed from generated text ("Confidence: 0.8"), quantized to a coarse grid in practice.
- **Sample agreement:** draw $n$ programs, cluster by input–output behavior on generated inputs, take $c = $ cluster mass — execution-space semantic entropy.

Calibration error uses the standard binned estimator (Naeini et al., AAAI 2015): partition $[0,1]$ into $B$ bins $\mathcal{B}_b$,
$$\widehat{\mathrm{ECE}} = \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{N}\,\big|\,\mathrm{acc}(\mathcal{B}_b) - \mathrm{conf}(\mathcal{B}_b)\,\big|,$$
plus Brier score $\frac1N\sum (c_i - \hat{Z}_i)^2$ and its Murphy decomposition into reliability, resolution, uncertainty.

**Assumptions, with the violated ones flagged:**

| Assumption | Status |
|---|---|
| $\hat Z = Z$ (test suite is the specification) | **Violated.** EvalPlus showed HumanEval suites admit wrong programs. |
| Deployment tasks are i.i.d. from the calibration distribution | **Violated.** Calibrators fit on HumanEval-like function synthesis are applied to repository edits. |
| Execution is deterministic | **Violated** for concurrency, network, time, floating point. |
| $\widehat{\mathrm{ECE}}$ is an unbiased estimate of ECE | **Violated.** Binned ECE is biased downward and depends on $B$ (Vaicenavicius et al., AISTATS 2019). |
| Benchmark tasks are not in pretraining | **Violated** for HumanEval/MBPP at current data scales. |

## 3. State of the Art

**Established.**
- Length-normalized log-likelihood ranks candidate programs above random but is a weak absolute probability. Reported since Codex (Chen et al., 2021), reproduced widely.
- Execution-based signals dominate likelihood signals. CodeT (Chen et al., ICLR 2023) raised HumanEval pass@1 for `code-davinci-002` from 47.0% to 65.8% by dual-execution agreement on generated tests. LEVER (Ni et al., ICML 2023) and Fault-Aware Rankers (Inala et al., NeurIPS 2022) show the same direction on semantic parsing and MBPP-style tasks.
- Post-hoc rescaling (Platt scaling, temperature scaling) reduces measured calibration error for code models. Spiess et al. (ICSE 2025) is the most direct study: across several models and code tasks, raw confidence signals were poorly calibrated and rescaling improved Brier/ECE substantially.
- PAC-style prediction sets for code models are constructible with finite-sample coverage guarantees (Khakhar, Mell, Bastani, ICML 2023) — guarantee is on *set coverage*, not on a scalar $c$.

**Claimed but unablated.**
- That P(True) self-evaluation calibration "improves with scale" for code specifically. Kadavath et al. (2022) show the trend on a broad task mix at up to 52B parameters; the code-only slice is not independently reproduced at frontier scale.
- That RLHF/instruction tuning destroys calibration for code. The GPT-4 report shows this for MMLU multiple-choice; the code analogue is asserted more often than measured.
- That verbalized confidence is usable. Xiong et al. (ICLR 2024) find it overconfident and coarsely quantized on QA; code-specific replication is thin.

**Benchmark-number-only.** AlphaCode's 34.2% solve rate on Codeforces (Science 2022) comes from filtering and clustering 100k+ samples; it is a selection result, not a calibration result — no reliability diagram is reported.

## 4. What Is Known

- **Sharpness beats calibration for reranking.** A miscalibrated but well-ordered score still recovers most of the pass@$k$ → pass@1 gap. CodeT: +18.8 absolute points pass@1 at HumanEval scale (164 tasks, 100 samples/task).
- **Oracle weakness is large and measured.** EvalPlus (Liu et al., NeurIPS 2023) added ~80× more tests to HumanEval; pass@1 for then-SOTA models dropped by up to ~24 percentage points. Any $c$ calibrated against original HumanEval is calibrated against an oracle that is wrong ~1 time in 5 on failures.
- **Scale of existing calibration studies is small.** Function-level benchmarks are 164 (HumanEval), 500 (MBPP test), ~2,300 (SWE-bench full). With $N=164$, per-bin counts at $B=10$ are ~16, and the standard error on a bin's accuracy is ~12 points — larger than most reported ECE improvements.
- **Neural nets are overconfident by default** (Guo et al., ICML 2017); temperature scaling with a single scalar fixes most of it in-distribution and none of it out-of-distribution.
- **Distribution-free calibration is impossible** for a predictor with continuous output without discretizing into bins (Gupta et al., NeurIPS 2020) — the guarantee attaches to the bin, not the number.
- **Displayed uncertainty does not automatically help users.** Vasconcelos et al. found token-probability highlighting in a completion UI did not reliably improve developer outcomes.

## 5. What Is Not Known

- **Methodologically blocked (dominant).** There is no agreed definition of the target event $Z$ beyond self-contained functions. For a repository patch, "correct" is a mixture of tests passing, no regressions, and unstated intent. Until $Z$ is pinned down for agentic/repo-level tasks, $c$ cannot be scored there at all.
- **Empirically open.** Whether any confidence signal transfers: fit a calibrator on MBPP, evaluate ECE on SWE-bench-style tasks. Runnable today; not run at scale with a proper control.
- **Empirically open.** Whether execution-agreement confidence remains calibrated when the generated tests come from the same model that wrote the code (shared-error correlation). Sharpness gains are documented; reliability under correlated failure is not.
- **Theoretically open.** Whether a nontrivial calibration guarantee exists for a decidable fragment (e.g. loop-free programs over bounded integers with a bounded-size specification) — no proof either way.
- **Theoretically open.** Whether calibration is achievable simultaneously with sharpness above the level attainable by the ranking-only estimator, given adversarial (pretraining-contaminated) task distributions.

## 6. Why It Is Hard

The specific obstruction is **absent and adversarially weak ground truth, compounded by measurement bias in the metric**.

1. $\hat Z$ is a proxy that the model has partly optimized against, since public test suites are in pretraining. A calibrator fit on $\hat Z$ learns to predict "passes the known tests", which is not the named quantity.
2. $\widehat{\mathrm{ECE}}$ is a biased, binning-dependent estimator, and the benchmark sizes are 100–1,000 examples. The measurement noise floor exceeds the reported effect sizes, so "improved calibration" claims are frequently unfalsifiable as stated.
3. Non-identifiability of the failure source: when $c=0.8$ and the program fails, you cannot separate "model was overconfident" from "test was flaky/wrong/over-strict" without manual adjudication.

This is not "hard because it matters" — it is hard because the label is wrong at a rate comparable to the quantity being estimated.

## 7. Current Research (as of 2026)

- **Execution-grounded confidence.** Self-generated tests, dual agreement, and behavioral clustering; the semantic-entropy idea (Kuhn et al., ICLR 2023) transplanted from NL to input–output equivalence classes. Active at Meta AI/CMU (Coder-Reviewer lineage), Microsoft Research.
- **Calibration for code specifically.** Devanbu's group at UC Davis with Pradel (Stuttgart) and collaborators — Spiess et al. (ICSE 2025) is the anchor; follow-ups extend to line-level and repair confidence.
- **Conformal / PAC prediction sets for code.** Bastani's group at Penn.
- **Agentic self-assessment.** Whether a coding agent's stated "I'm confident this is fixed" on SWE-bench-style tasks is calibrated against maintainer-written tests *(frontier — verify; results here are mostly vendor evaluations without reliability diagrams)*.
- **Contamination-resistant benchmarks** (LiveCodeBench-style rolling collection) as a way to get an uncontaminated calibration distribution *(frontier — verify for calibration-specific use)*.

## 8. Concrete Next Experiment

**Question.** Does any confidence signal transfer across task distributions, or is code calibration purely in-distribution curve-fitting?

**Scale.** One open-weights model (e.g. 30B-class instruct) and one API frontier model. Fit set: 500 MBPP tasks × 20 samples = 10,000 (program, $\hat Z$) pairs. Test sets, each ≥ 1,000 programs: (a) EvalPlus-strengthened HumanEval+, (b) a rolling contamination-free set collected after the model's cutoff, (c) SWE-bench-Verified patches scored by maintainer tests.

**Arms.** Four estimators — length-normalized likelihood, P(True), verbalized confidence, execution-agreement over 20 self-generated tests — each with and without a Platt scaler fit only on MBPP.

**Control arm.** The constant predictor $c = \bar{Z}_{\text{fit}}$ (MBPP base rate). This is perfectly sharpness-free and its ECE on each test set is the number every method must beat.

**Deciding number.** $\Delta = \widehat{\mathrm{ECE}}_{\text{method}} - \widehat{\mathrm{ECE}}_{\text{constant}}$ on test set (c), with 15-bin equal-mass binning and bootstrap 95% CI over tasks. If no arm achieves $\Delta < -0.05$ with a CI excluding 0, transfer fails and the field should report only in-distribution calibration plus ranking metrics. Report AUROC alongside, so a method that loses on ECE but wins on ordering is not discarded.

## 9. Key References

- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* 2021. — arXiv:2107.03374
- **[SOTA]** Claudio Spiess, David Gros, Kunal Suresh Pai, Michael Pradel, Md Rafiqul Islam Rabin, Amin Alipour, Susmit Jha, Prem Devanbu, Toufique Ahmed. *Calibration and Correctness of Language Models for Code.* ICSE, 2025. — arXiv:2402.02047
- **[SOTA]** Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, Weizhu Chen. *CodeT: Code Generation with Generated Tests.* ICLR, 2023. — arXiv:2207.10397
- **[SOTA]** Tianyi Zhang, Tao Yu, Tatsunori Hashimoto, Mike Lewis, Wen-tau Yih, Daniel Fried, Sida I. Wang. *Coder Reviewer Reranking for Code Generation.* ICML, 2023. — arXiv:2211.16490
- **[SOTA]** Adam Khakhar, Stephen Mell, Osbert Bastani. *PAC Prediction Sets for Large Language Models of Code.* ICML, 2023. — arXiv:2302.08703
- **[Method]** Ansong Ni, Srini Iyer, Dragomir Radev, Ves Stoyanov, Wen-tau Yih, Sida I. Wang, Xi Victoria Lin. *LEVER: Learning to Verify Language-to-Code Generation with Execution.* ICML, 2023. — arXiv:2302.08468
- **[Method]** Jeevana Priya Inala, Chenglong Wang, Mei Yang, Andres Codas, Mark Encarnación, Shuvendu Lahiri, Madanlal Musuvathi, Jianfeng Gao. *Fault-Aware Neural Code Rankers.* NeurIPS, 2022. — arXiv:2206.03865
- **[Measurement]** Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS, 2023. — arXiv:2305.01210
- **[Theory]** Chirag Gupta, Aleksandr Podkopaev, Aaditya Ramdas. *Distribution-free binary classification: prediction sets, confidence intervals and calibration.* NeurIPS, 2020. — arXiv:2006.10564
- **[Theory]** Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, Thomas B. Schön. *Evaluating model calibration in classification.* AISTATS, 2019. — arXiv:1902.06977
- **[Related]** Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023. — arXiv:2302.09664
- **[Related]** Miao Xiong et al. *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.* ICLR, 2024. — arXiv:2306.13063
- **[Survey]** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770

## 10. Worked Example

Take HumanEval (164 tasks), one sample per task, confidence $c$ = length-normalized sequence probability, bucketed into 10 equal-width bins. Suppose the model's original-suite pass@1 is 82% (135/164).

The top bin, $c \in [0.9, 1.0]$, holds 42 programs; 40 pass. Measured accuracy 0.952, mean confidence 0.94, bin contribution to ECE: $\frac{42}{164}\times 0.012 = 0.003$. Summed over bins you get $\widehat{\mathrm{ECE}} = 0.041$ — publishable as "well calibrated".

Now swap the oracle for HumanEval+. EvalPlus-scale test strengthening removes on the order of 15–24% of previously-passing solutions. Apply a 16% failure rate to the top bin: 40 passes become ~34. The bin is now accuracy 0.810 at confidence 0.94 — contribution $\frac{42}{164}\times 0.130 = 0.033$, ten times larger. Total ECE moves to roughly 0.12–0.15. Nothing about the model or the confidence signal changed. The entire calibration result was a property of the test suite.

Then check whether the difference is even resolvable. The top bin has $n=42$; the standard error on its accuracy is $\sqrt{0.81 \cdot 0.19 / 42} = 0.061$. A paper reporting "ECE reduced from 0.041 to 0.028 by temperature scaling" is reporting a shift of 0.013 against per-bin noise of 6 points and an oracle-induced bias of 13 points.

That is the obstruction in one calculation: the oracle error and the estimator noise both exceed the effect. Fixing it needs a stronger $Z$ and $N$ in the thousands — not a better scaling function.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*