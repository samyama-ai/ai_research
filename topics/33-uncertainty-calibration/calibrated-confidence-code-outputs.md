---
id: 33-uncertainty-calibration/calibrated-confidence-code-outputs
title: "Calibrated Confidence for Code and Formal Outputs"
topic: 33-uncertainty-calibration
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibrated Confidence for Code and Formal Outputs

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/calibrated-confidence-code-outputs` · **Status:** empirically-open

## 1. Problem Statement

A code model emits a program $y$ for a specification $x$ and attaches a confidence $c \in [0,1]$. The claim is: "with probability $c$, $y$ satisfies $x$." The problem is to make that claim true, and to know when it is.

Three variants, routinely conflated:

- **Measurement.** Code correctness is not a single label. Passing the visible tests, passing a held-out test suite, passing a mutation-strengthened suite, and being semantically equivalent to a reference are four different events with four different calibration curves. Choosing one silently fixes the answer.
- **Method.** Produce a scalar $c$ — from token likelihood, self-report, sampling agreement, or a trained verifier — whose reliability diagram is diagonal on a target distribution, and whose ranking is useful (high AUC-ROC), and which survives distribution shift from HumanEval-style functions to repository-scale edits.
- **Theory.** For formal outputs, correctness is decidable by a checker (type checker, proof kernel, test oracle). Does that decidability let one build distribution-free guarantees stronger than the exchangeability-only bounds of conformal prediction? Open.

Solving it means: a deployed confidence signal with expected calibration error under 0.05 and abstention curves that dominate the no-confidence baseline, on repository-scale tasks, without access to the hidden tests.

## 2. Formal Setting

Let $x \sim \mathcal{D}$ be a task (prompt, repository state, spec), $y = f_\theta(x)$ a sampled program, and $V$ a verifier. Correctness is $Z = V(x, y) \in \{0,1\}$.

**Correctness as measured.** $V$ is never the ideal oracle. Write $V_T$ for a finite test suite $T$:
$$Z_T = \mathbb{1}\big[\forall t \in T:\ y(t) = y^\star(t)\big], \qquad \Pr[Z_T = 1 \mid Z^\star = 0] = \beta(T) > 0,$$
where $Z^\star$ is semantic correctness and $\beta(T)$ is the false-accept rate of the suite. EvalPlus measured $\beta$ directly: adding generated tests to HumanEval dropped reported pass rates by up to 19.3 points, so the original suite accepted roughly one in five wrong programs it scored as right.

**Confidence.** Candidate estimators, all computable at inference:
- Sequence likelihood, length-normalised: $c_{\text{lik}} = \exp\!\big(\tfrac{1}{|y|}\sum_i \log p_\theta(y_i \mid y_{<i}, x)\big)$.
- Verbalized: $c_{\text{verb}}$, the model's stated percentage, parsed from text.
- P(True): $c_{\text{true}} = p_\theta(\text{"True"} \mid x, y, \text{"Is this correct?"})$ (Kadavath et al., 2022).
- Agreement: $c_{\text{agr}} = \frac{1}{n}\sum_{j} \mathbb{1}[y^{(j)} \equiv y]$ over $n$ samples, with $\equiv$ = agreement on generated inputs, not string equality.

**Calibration error.** With $B$ equal-mass bins,
$$\widehat{\mathrm{ECE}} = \sum_{b=1}^{B} \frac{|I_b|}{N}\Big| \frac{1}{|I_b|}\!\!\sum_{i \in I_b}\! Z_i \; - \; \frac{1}{|I_b|}\!\!\sum_{i \in I_b}\! c_i \Big| .$$
This estimator is biased downward and its bias grows with $B$ at fixed $N$ (Vaicenavicius et al., AISTATS 2019); at $N=164$ (HumanEval) and $B=10$ the bias is the same order as the effect being reported.

**Selective prediction.** For threshold $\tau$, coverage $\kappa(\tau) = \Pr[c \ge \tau]$, risk $R(\tau) = \Pr[Z=0 \mid c \ge \tau]$. The deployment-relevant number is area under the risk–coverage curve, not ECE.

**Assumptions, and which fail.**
1. *$Z$ is well defined* — fails: $Z_T \ne Z^\star$ with $\beta(T)$ unmeasured outside curated benchmarks.
2. *Exchangeability of calibration and test tasks* — fails for conformal methods the moment tasks come from a different repository or language.
3. *Independence of $c$ and $V$* — fails when the model generates its own tests: $c_{\text{agr}}$ and $Z_T$ share the failure mode of misreading the spec.
4. *One program per task* — fails at repository scale, where a patch touches several files and correctness is partially ordered.

## 3. State of the Art

**Established.**
- Post-hoc rescaling works on next-token-style signals. Guo et al. (ICML 2017) showed temperature scaling — one parameter — removes most classifier miscalibration; Spiess et al. (ICSE 2025) carried this to code, finding raw likelihood and verbalized confidence both poorly calibrated for code generation and repair, with Platt scaling giving the largest reliable ECE reduction across models and tasks. This is the strongest ablated result in the area.
- Execution-based reranking beats likelihood. CodeT (Chen et al., ICLR 2023) and LEVER (Ni et al., ICML 2023) both improve pass@1 by executing generated tests; LEVER reports gains of ~6–8 points over greedy decoding across Spider, MBPP and GSM8k. These are *ranking* results, not calibration results — neither paper reports ECE.
- Distribution-free coverage exists for code under exchangeability. Khakhar, Mell & Bastani (ICML 2023) build PAC prediction sets for code models over partial-program spaces, with proven coverage on the calibration distribution.

**Claimed but unablated.**
- That self-consistency across samples is a calibrated probability. It is a good *ranker*; the mapping from agreement fraction to correctness probability is fitted per benchmark and has not been shown to transfer.
- That verbalized confidence, elicited well, is calibrated for code. Tian et al. (EMNLP 2023) showed verbalized beats likelihood for RLHF'd models on QA; the code replication in Spiess et al. does not reproduce that ordering.

**Benchmark-number-only.** Every reported ECE for code confidence known to us is on HumanEval ($N=164$), MBPP ($N \approx 500$), or a similar function-level set. No published ECE on SWE-bench-style repository tasks.

## 4. What Is Known

- **Base models are near-calibrated on multiple choice; alignment destroys it.** The GPT-4 technical report (OpenAI, 2023) shows the pre-RLHF model near-diagonal on MMLU and the post-RLHF model badly overconfident — the figure's reported ECE moves from about 0.007 to about 0.074, a factor of ten, from post-training alone.
- **Self-evaluation scales.** Kadavath et al. (2022, arXiv:2207.05221) found P(True) calibration improves monotonically with model size across 800M–52B, and that models score their *own* samples better than others'.
- **Test suites lie at a measurable rate.** EvalPlus (Liu et al., NeurIPS 2023): $80\times$ more tests reduced HumanEval pass@1 for evaluated models by up to 19.3 points — a direct estimate of $\beta(T)$ on the canonical benchmark.
- **Semantic-equivalence clustering beats raw entropy** for free-form correctness detection (Kuhn et al., ICLR 2023; Farquhar et al., *Nature* 2024, AUROC gains of roughly 0.05–0.10 over naive entropy on QA). Ported to code only as heuristic input-agreement.
- **Small-$N$ ECE is unreliable.** Kumar, Liang & Ma (NeurIPS 2019) show binned ECE systematically understates true calibration error and give a debiased estimator; the correction is rarely applied in code papers.

## 5. What Is Not Known

- **Empirically open.** Does any confidence signal stay calibrated from function-level tasks to repository-level patches? The experiment is runnable today — SWE-bench + a fixed elicitation protocol + a debiased ECE estimator — and has not been run at $N$ large enough to resolve a 0.05 ECE difference.
- **Empirically open.** Which of $c_{\text{lik}}, c_{\text{verb}}, c_{\text{true}}, c_{\text{agr}}$ dominates on the risk–coverage curve at fixed inference cost. Existing comparisons vary sampling budget across arms, so cost is confounded with method.
- **Theoretically open.** Whether decidable verification admits a calibration guarantee strictly stronger than conformal exchangeability bounds — e.g. a bound on $\Pr[Z^\star=0 \mid c, V_T(y)=1]$ using structure of $T$ rather than an i.i.d. calibration set. No proof either way.
- **Methodologically blocked.** Calibration against *semantic* correctness. Without a decidable oracle for general programs (Rice's theorem), $Z^\star$ is unobservable; every reported ECE is ECE against a proxy with unknown $\beta$. For proof assistants this block lifts — the kernel decides — which is why Lean/Coq settings are the cleanest testbed.

## 6. Why It Is Hard

The specific obstruction is **label noise with unknown, confidence-correlated rate**. Miscalibration measured against $Z_T$ decomposes as true miscalibration plus $\beta(T)$ times the model's tendency to fail exactly where the tests are weak — and those two are not independent. A model that writes plausible-but-wrong code writes code that passes shallow tests. So the proxy oracle inflates measured calibration precisely on the confident-and-wrong cases that matter, and the sign of the bias is unknown without measuring $\beta$ per task.

Second obstruction: **estimator variance at benchmark scale.** Distinguishing ECE 0.04 from ECE 0.08 with 10 bins needs $N$ in the low thousands. HumanEval has 164 problems and SWE-bench Verified has 500. Most published differences are inside the noise band.

## 7. Current Research (as of 2026)

- **Calibration-for-code as its own line.** Devanbu's group (UC Davis) with Pradel (Stuttgart), Alipour (Auburn) and SRI — the ICSE 2025 study is the reference point; follow-ups extend to code summarization and repair.
- **Conformal generation.** Bastani/Mell (Penn) on PAC sets for code; Barzilay/Jaakkola's group (MIT) on conformal language modeling stopping rules.
- **Verifier-as-confidence.** Process- and outcome-reward models used as scalar confidence rather than rerankers; the open question is whether reward-model scores are probabilities. *(frontier — verify)*
- **Proof assistants as clean oracles.** LeanDojo (Yang et al., NeurIPS 2023), Baldur (First et al., FSE 2023) give decidable $Z^\star$; calibration studies on Lean proof search are the obvious next step and largely unpublished. *(frontier — verify)*
- **Agentic self-report.** Whether an SWE-agent's stated confidence after running its own tests is calibrated — reported anecdotally in system cards, not measured. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does confidence calibrated on function-level tasks transfer to repository-level tasks?

**Scale.** 2,000 tasks: 1,000 function-level (MBPP + HumanEval+, EvalPlus test suites) and 1,000 repository-level (SWE-bench full, subsampled). One frontier model, temperature 0.8, $n=10$ samples per task — 20,000 generations, roughly $10^3$ GPU-hours or a few hundred dollars of API spend. This $N$ resolves an ECE gap of 0.03 at 10 bins.

**Arms.** Four confidence estimators ($c_{\text{lik}}, c_{\text{verb}}, c_{\text{true}}, c_{\text{agr}}$), each fitted with Platt scaling on the function-level half, evaluated on the repository half.

**Control arm.** The same estimator refit on a repository-level calibration split. The control isolates transfer failure from estimator failure — without it, a bad number is unattributable.

**Deciding number.** $\Delta = \widehat{\mathrm{ECE}}_{\text{transferred}} - \widehat{\mathrm{ECE}}_{\text{refit}}$, using the debiased estimator of Kumar et al. (2019) with bootstrap CIs. If $\Delta < 0.02$ for any estimator, function-level calibration transfers and the field can keep using cheap benchmarks. If $\Delta > 0.05$ for all four, every published code-calibration number is scoped to toy tasks and must be re-measured.

## 9. Key References

- **[Foundational]** C. Guo, G. Pleiss, Y. Sun, K. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Foundational]** S. Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[SOTA]** C. Spiess, D. Gros, K. S. Pai, M. Pradel, M. R. I. Rabin, A. Alipour, S. Jha, P. Devanbu, T. Ahmed. *Calibration and Correctness of Language Models for Code.* ICSE, 2025. — arXiv:2402.02047
- **[SOTA]** J. Liu, C. S. Xia, Y. Wang, L. Zhang. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS, 2023. — arXiv:2305.01210
- **[SOTA]** A. Khakhar, S. Mell, O. Bastani. *PAC Prediction Sets for Large Language Models of Code.* ICML, 2023. — arXiv:2302.08703
- **[Method]** A. Kumar, P. Liang, T. Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[Method]** L. Kuhn, Y. Gal, S. Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023. — arXiv:2302.09664
- **[Method]** S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 2024.
- **[Method]** K. Tian et al. *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* EMNLP, 2023. — arXiv:2305.14975
- **[Related]** B. Chen et al. *CodeT: Code Generation with Generated Tests.* ICLR, 2023. — arXiv:2207.10397
- **[Related]** A. Ni et al. *LEVER: Learning to Verify Language-to-Code Generation with Execution.* ICML, 2023. — arXiv:2302.08468
- **[Related]** C. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Survey]** A. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021. — arXiv:2107.07511
- **[Measurement]** J. Vaicenavicius et al. *Evaluating Model Calibration in Classification.* AISTATS, 2019. — arXiv:1902.06977

## 10. Worked Example

Take HumanEval, $N=164$. Suppose a model reports $c=0.9$ on 60 problems and, scored against the *original* suite, 54 of them pass. Measured on that bin: mean confidence 0.90, accuracy 0.90, contribution to ECE zero. The reliability diagram is diagonal. Publishable.

Now rescore the same 60 with EvalPlus tests. EvalPlus's headline effect is up to 19.3 points of pass@1 lost; take a conservative 10 points on this bin, so 48 of 60 pass. Accuracy is 0.80, confidence 0.90, and the bin contributes $\frac{60}{164}\times 0.10 = 0.037$ to ECE on its own. A model reported as perfectly calibrated is overconfident by 10 points.

Two things make the obstruction visible.

**The bias has a direction.** The 6 programs that flipped are not random: they pass shallow tests and fail edge cases, which is the failure mode of confident-looking code. The proxy oracle preferentially forgives errors inside the high-confidence bin, so measured ECE is biased toward zero exactly where the deployment cost lives.

**The noise swamps the signal.** With 60 samples in the bin, the standard error on accuracy at $p=0.85$ is $\sqrt{0.85 \cdot 0.15/60} \approx 0.046$. The 95% interval on that bin is about $\pm 0.09$ — wider than the 0.037 ECE contribution being estimated. So even the corrected number is unresolved at HumanEval scale.

Both problems are fixed the same way: a decidable oracle ($\beta = 0$) and $N$ in the thousands. Lean proof search gives the first for free — the kernel accepts or it does not — which is why the cleanest version of this experiment is not on Python at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*