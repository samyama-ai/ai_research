---
id: 26-code-generation/calibrated-confidence-program-correctness
title: "Calibrated Confidence for Generated Program Correctness"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibrated Confidence for Generated Program Correctness

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/calibrated-confidence-program-correctness` · **Status:** open

## 1. Problem Statement

A model emits a program $y$ for a specification $x$. The question is not whether $y$ is correct but whether the system can say *how likely* it is to be correct, in a number a caller can act on.

- **Input:** specification $x$ (docstring, issue text, type signature, tests-withheld task), sampled program $y \sim p_\theta(\cdot \mid x)$, and any artifacts the system may compute without the hidden oracle (self-generated tests, execution traces, static analysis, log-probs, a second model's judgment).
- **Output:** a confidence $c(x,y) \in [0,1]$.
- **Objective:** $c$ should be *calibrated* against the semantic-correctness event $C(x,y) \in \{0,1\}$ — among all cases where the system says $0.9$, about $90\%$ should be correct — and *sharp* (concentrated near $0$ and $1$), because the constant predictor $c \equiv \Pr[C=1]$ is perfectly calibrated and useless.

Three variants, routinely conflated:

- **Measurement variant.** Is $C(x,y)$ even observable? Test-suite pass is a proxy; EvalPlus showed the proxy is loose. Calibration error measured against a weak oracle is calibration to the oracle's bias, not to correctness.
- **Method variant.** Given a fixed correctness oracle and i.i.d. data, produce $c$ with low calibration error and high sharpness at deployment cost. Largely an empirical engineering problem; partially solved in-distribution.
- **Theory variant.** Obtain a *distribution-free* guarantee — e.g. selective prediction with risk $\le \epsilon$ — for programs, where the label depends on an unbounded input space and the exchangeability assumption is broken by repository-specific and time-shifted tasks.

Solved would mean: on a held-out, uncontaminated task stream, a system abstains or accepts such that accepted programs are correct at a stated rate $1-\epsilon$ (measured, not assumed), with coverage materially above what a log-prob threshold achieves, and the guarantee surviving a shift to a new repository or benchmark generation.

## 2. Formal Setting

Tasks $(x, O) \sim \mathcal{D}$, where $O$ is a hidden oracle: $O(y)=1$ iff $y$ is semantically correct for $x$. Correctness is measured through a finite test suite $T$:

$$\hat{C}_T(x,y) = \prod_{t \in T} \mathbb{1}[\,\text{exec}(y, t) = \text{expected}(t)\,], \qquad \hat{C}_T \ge C \text{ pointwise (one-sided error)}.$$

$\hat{C}_T$ has false positives (weak tests) and near-zero false negatives modulo flaky execution. Define the **proxy gap** $\delta_T = \Pr[\hat{C}_T = 1] - \Pr[C = 1]$; EvalPlus measures $\delta_T$ empirically by strengthening $T$.

Confidence $c: \mathcal{X}\times\mathcal{Y} \to [0,1]$. Perfect calibration:

$$\Pr\big[\,C = 1 \mid c(X,Y) = v\,\big] = v \quad \forall v \in [0,1].$$

Measured with the binned estimator over $B$ equal-mass bins, $n$ samples:

$$\widehat{\mathrm{ECE}} = \sum_{b=1}^{B} \frac{|I_b|}{n}\Big|\,\overline{C}_b - \overline{c}_b\,\Big|, \qquad \overline{C}_b = \tfrac{1}{|I_b|}\sum_{i\in I_b}\hat{C}_T(x_i,y_i).$$

$\widehat{\mathrm{ECE}}$ is biased upward at finite $n$; with $n=164$ (HumanEval) and $B=10$ the bin noise is $\pm 0.12$ at $p=0.5$, comparable to the effect sizes reported. Brier score $\mathrm{BS} = \frac1n\sum_i (c_i - \hat{C}_i)^2$ decomposes into calibration minus refinement plus irreducible variance, and is the safer headline number.

Selective prediction: with accept rule $\mathbb{1}[c \ge \tau]$, coverage $\mathrm{cov}(\tau)=\Pr[c\ge\tau]$ and selective risk $R(\tau)=\Pr[C=0 \mid c \ge \tau]$. The operational goal is $\max \mathrm{cov}(\tau)$ subject to $R(\tau)\le \epsilon$.

Split conformal on a calibration set of size $m$ gives, under **exchangeability** of $(x_i,y_i)$ with the test point, a prediction set $\mathcal{S}(x)$ with $\Pr[C(x,y)=1 \text{ for some } y \in \mathcal{S}(x)] \ge 1-\alpha$, the guarantee being marginal over the draw of the calibration set.

Assumptions and their status in practice:

| Assumption | Status |
|---|---|
| $\hat{C}_T \approx C$ | **Violated.** Test suites pass semantically wrong programs; HumanEval's suites are thin. |
| Exchangeability of calibration and test tasks | **Violated.** Deployment is repo-specific, time-shifted, and prompt-distribution-shifted. |
| Calibration set is uncontaminated by pretraining | **Usually violated or unverifiable** for public benchmarks. |
| Deterministic execution | Mostly holds; flaky/timeout tests inject label noise of order $10^{-2}$. |
| Marginal coverage suffices | Weak. Callers want conditional coverage per repository or per difficulty band. |

## 3. State of the Art

**Established (ablated, reproduced):**

- *Execution-based reranking beats likelihood.* CodeT (Chen et al., ICLR 2023) uses model-generated tests plus dual execution agreement; LEVER (Ni et al., ICML 2023) trains a verifier on execution results. Both improve pass@1 over log-prob reranking by large margins on MBPP/Spider-class tasks, and both ablate the execution signal specifically.
- *Self-evaluation carries real signal.* Kadavath et al. (2022) show P(True) — asking the model whether its own answer is correct — is meaningfully calibrated and improves with scale, across tasks including code.
- *Raw token probabilities are poorly calibrated for code, and post-hoc rescaling helps.* Spiess et al. (ICSE 2025) evaluate log-prob, verbalized, and reflective confidence over multiple code models and tasks; Platt scaling improves Brier/ECE substantially, while verbalized ("say a number") confidence is the weakest signal.

**Claimed but unablated / benchmark-number-only:**

- Agentic self-repair loops reporting high accept rates on SWE-bench-family tasks report *accuracy*, not calibration; no confidence curve is published for most of them. Treat any implied confidence as unmeasured.
- Semantic-entropy-style clustering (Kuhn et al., ICLR 2023; Farquhar et al., Nature 2024) transfers to code only if semantic equivalence is defined — equivalence-by-test is decidable, equivalence-in-general is not. Reported code results exist but the equivalence relation differs across papers, so numbers are not comparable.
- LLM-as-judge confidence on code: benchmark numbers only, with judge–generator correlation uncontrolled.

**Theory SOTA:** PAC prediction sets for code (Khakhar, Mell, Bastani, ICML 2023) give distribution-free finite-sample guarantees for LLM code output under exchangeability — the only construction in this area with a proof attached. It buys a guarantee at the cost of set-valued output and i.i.d. assumptions the deployment breaks.

## 4. What Is Known

- **Sampling makes correctness cheap to buy but not to identify.** Codex 12B: pass@1 $= 28.8\%$, pass@100 $= 72.3\%$ on HumanEval (164 problems, Chen et al. 2021). The gap is exactly the confidence problem: one of the 100 is right, and mean log-prob picks it far below oracle rate.
- **Test-suite proxies overstate correctness.** EvalPlus (Liu et al., NeurIPS 2023) adds ~80× more tests to HumanEval; reported pass@1 falls by roughly 10–15 points for strong models (e.g. GPT-4-class scores drop from the high-80s to the mid-70s). Every ECE number computed on original HumanEval inherits that bias.
- **Scale improves self-knowledge more than it improves calibration of log-probs.** Kadavath et al. measured this on models up to 52B; the trend is monotone in scale for P(True).
- **Verbalized confidence is coarse.** Lin, Hilton, Evans (TMLR 2022) get non-trivial calibrated verbalized uncertainty on arithmetic; on code, Spiess et al. find it clusters at round numbers (0.8, 0.9) and is close to uninformative after binning.
- **Contamination moves the number.** LiveCodeBench (Jain et al., ICLR 2025) shows measurable accuracy drops on post-cutoff problems for several model families — confidence fitted on pre-cutoff data is fitted on partly memorized data.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted operational definition of $C$ for repository-scale tasks. SWE-bench uses the maintainer's test patch; a patch that passes it can still be wrong. Until $C$ is defined independently of the tests used to grade it, "calibration error for program correctness" names a quantity nobody has measured.
- **Empirically open.** Whether a confidence head calibrated on repository $A$ retains $\mathrm{ECE} \le 0.05$ on repository $B$ of a different language and test culture. The experiment is runnable today; no published run isolates repository shift from difficulty shift.
- **Empirically open.** Whether execution-derived confidence (self-generated tests) beats internal-state probes at matched compute. Present comparisons differ in inference budget by 10–100×.
- **Theoretically open.** Whether any distribution-free guarantee survives when the calibration label is itself a biased proxy $\hat{C}_T \ge C$. Conformal machinery assumes exchangeable *labels*; a one-sided, task-dependent label bias breaks the coverage proof, and no corrected construction is known.
- **Theoretically open.** Non-identifiability: given only $x$, $y$ and finite tests, correctness on unseen inputs is not determined; how much of the residual is irreducible for a given test-generation budget is unquantified.

## 6. Why It Is Hard

The obstruction is **confounded measurement with absent ground truth**, in a specific form: the label used to fit the confidence and the label used to score it come from the same test suite. Any $c$ trained to predict test-pass learns test-suite idiosyncrasy, and the evaluation cannot detect this because it uses the same oracle. Strengthening the oracle (EvalPlus-style mutation, differential testing) changes the labels and thereby changes reported ECE by more than the difference between competing methods — a 10-point shift in base rate against a 2–5 point method effect. Second obstruction: **exchangeability fails by construction.** Deployment tasks arrive grouped by repository, author, and time; the calibration set is public benchmarks. Third: **finite-sample noise.** HumanEval's $n=164$ cannot resolve $\mathrm{ECE}$ differences below ~0.1, so most published comparisons on it are underpowered.

## 7. Current Research (as of 2026)

- Execution-grounded verification: test generation as the confidence signal (CodeT/LEVER lineage), now extended to repository tasks via generated regression tests. *(frontier — verify)* SWT-Bench-style test-generation benchmarks (Mündler et al., NeurIPS 2024) are becoming the substrate for this.
- Conformal and PAC prediction sets for code (Bastani's group at Penn, and follow-ons), pushing toward group-conditional coverage per repository.
- Internal-state probes: linear probes on hidden activations predicting correctness, reported to beat log-probs at negligible inference cost. *(frontier — verify)* — most results are on single-model, single-benchmark setups.
- Contamination-controlled evaluation (LiveCodeBench, rolling benchmarks) as the substrate for honest calibration numbers.
- Verifier-model scaling: training dedicated correctness critics rather than reusing the generator. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does any confidence signal maintain calibration under repository shift, once the oracle is strengthened?

- **Scale.** 2,000 tasks: 1,000 from a *calibration* pool (5 Python repositories, SWE-bench-style issue→patch), 1,000 from a *test* pool of 5 disjoint repositories, all with commit dates after the model's cutoff. One model, 20 samples per task, temperature 0.8 — about $2000 \times 20 = 40{,}000$ generations, roughly 400–800 GPU-hours for a 70B-class open model, or a few thousand dollars via API.
- **Oracle.** Two labels per program: $\hat{C}_{T_{\text{orig}}}$ (maintainer tests) and $\hat{C}_{T^+}$ (maintainer tests plus 50 LLM-generated-and-human-audited differential tests). Report everything twice.
- **Arms.** (1) mean token log-prob; (2) P(True) self-evaluation; (3) generated-test agreement (CodeT-style); (4) linear probe on final-layer activations; (5) split-conformal set from the calibration pool.
- **Control arm.** Platt scaling of mean log-prob fitted on the calibration pool. This is the cheapest thing that already works in-distribution; any method that does not beat it under shift is not progress.
- **Deciding number.** Selective risk at 50% coverage on the *test* pool under the strengthened oracle $T^+$. A method wins only if $R(\tau_{50})$ is at least 5 percentage points below the control's, with a bootstrap 95% CI excluding zero ($n=1000$ gives roughly $\pm 3$ points at $R \approx 0.3$). Secondary: the ECE increase from calibration pool to test pool — if it exceeds 0.10 for every arm, the honest conclusion is that repository-conditional calibration is currently unattainable, which is itself the result.

## 9. Key References

- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374
- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* arXiv, 2022. — arXiv:2207.05221
- **[Foundational]** Vladimir Vovk, Alexander Gammerman, Glenn Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[SOTA]** Claudio Spiess et al. *Calibration and Correctness of Language Models for Code.* ICSE, 2025. — arXiv:2402.02047
- **[SOTA]** Bei Chen et al. *CodeT: Code Generation with Generated Tests.* ICLR, 2023. — arXiv:2207.10397
- **[SOTA]** Ansong Ni et al. *LEVER: Learning to Verify Language-to-Code Generation with Execution.* ICML, 2023. — arXiv:2302.08468
- **[SOTA]** Adam Khakhar, Stephen Mell, Osbert Bastani. *PAC Prediction Sets for Large Language Models of Code.* ICML, 2023.
- **[SOTA]** Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS, 2023. — arXiv:2305.01210
- **[SOTA]** Naman Jain et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR, 2025. — arXiv:2403.07974
- **[Related]** Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023. — arXiv:2302.09664
- **[Related]** Stephanie Lin, Jacob Hilton, Owain Evans. *Teaching Models to Express Their Uncertainty in Words.* TMLR, 2022. — arXiv:2205.14334
- **[Related]** Carlos E. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Survey]** Anastasios N. Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv, 2021. — arXiv:2107.07511

## 10. Worked Example

Take HumanEval-style task `has_close_elements(numbers, threshold)` — return `True` if any two numbers are closer than `threshold`. Sample 20 completions at $T=0.8$. Suppose 14 pass the original three-assert suite; mean log-prob ranks them and the confidence assigned to the top sample is $c = 0.92$.

Now strengthen the oracle. The original suite never tests the *empty list* or a list of length 1. A common completion is:

```python
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

This compares each element to itself, so it returns `True` whenever `threshold > 0` and the list is non-empty. It passes the original suite only if every original test case happens to contain a genuine close pair or an expected-`True` answer — which for several HumanEval problems it does. Under EvalPlus-style inputs it fails immediately.

Carry the numbers. Under $T_{\text{orig}}$: 14/20 pass, base rate $0.70$, and the log-prob-ranked top-1 gets $c=0.92$; the bin at $c \in [0.9,1.0]$ has empirical accuracy $0.90$, so $\widehat{\mathrm{ECE}}$ contribution $\approx 0.02$. Looks calibrated. Under $T^+$: 9/20 pass, base rate $0.45$, and the same high-confidence bin now has empirical accuracy $0.62$ — contribution $\approx 0.28$. The confidence function did not change. The measurement did.

That is the obstruction in one instance: the reported ECE moved by $0.26$ from a change in the oracle alone, an order of magnitude larger than the 2–5 point gaps that separate published confidence methods. Any ranking of methods computed against a weak suite is dominated by $\delta_T$, not by the methods. The experiment in §8 is designed so that $\delta_T$ is measured rather than absorbed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*