---
id: 19-evaluation/eval-under-benchmark-use-distribution-shift
title: "Evaluation Under Distribution Shift Between Benchmark and Use"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Evaluation Under Distribution Shift Between Benchmark and Use

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/eval-under-benchmark-use-distribution-shift` · **Status:** open

## 1. Problem Statement

A benchmark score is measured on distribution $P$. The decision it informs — deploy, rank, gate a release — concerns performance on a use distribution $Q \neq P$. The problem: bound or estimate the quantity that matters, $R_Q(f)$, from data drawn from $P$, plus whatever unlabeled or partially labeled evidence about $Q$ is available.

Three variants, with different difficulty:

- **Measurement.** Given a labeled benchmark $\{(x_i,y_i)\}\sim P$, unlabeled deployment traffic $\{\tilde x_j\}\sim Q$, and a model $f$, produce an interval $[\ell, u]$ with a stated coverage guarantee for $R_Q(f)$. Solved is: coverage holds empirically on held-out labeled deployment data across many $(P,Q,f)$ triples, and the interval is narrow enough to separate the candidate models.
- **Method.** Build benchmarks whose score *transfers* — where the induced ranking over models on $P$ matches the ranking on $Q$. Solved is: rank correlation above a stated threshold on held-out deployment tasks, not on the benchmark's own splits.
- **Theory.** Characterize the conditions on $(P, Q, \mathcal{F})$ under which $R_Q$ is identifiable from $P$-samples plus unlabeled $Q$-samples. Partially settled, and the settled part is negative (§4).

The measurement variant is the one that is actually blocked, because "the use distribution" is rarely a distribution anyone has sampled.

## 2. Formal Setting

Let $\mathcal{X}$ be inputs, $\mathcal{Y}$ outputs, $f:\mathcal{X}\to\mathcal{Y}$ the system under test, $\ell:\mathcal{Y}\times\mathcal{Y}\to[0,1]$ a bounded loss. Define

$$R_P(f) = \mathbb{E}_{(x,y)\sim P}[\ell(f(x),y)], \qquad R_Q(f) = \mathbb{E}_{(x,y)\sim Q}[\ell(f(x),y)].$$

**As measured.** $R_P$ is estimated by $\hat R_P = \frac1n\sum_i \ell(f(x_i),y_i)$ over the $n$ benchmark items — $n = 14{,}042$ for MMLU test, $n = 500$ for SWE-bench Verified, $n = 1{,}319$ for GSM8K test. $Q$ is *not* sampled; what exists is a log of inputs $\{\tilde x_j\}_{j=1}^{m}$ with no labels, often filtered by a product's own routing.

**Importance weighting.** If $Q \ll P$ (absolute continuity) with $w(x,y) = dQ/dP$,

$$R_Q(f) = \mathbb{E}_P[w(x,y)\,\ell(f(x),y)], \qquad \widehat{R}_Q^{\mathrm{IW}} = \frac1n\sum_i \hat w_i\, \ell(f(x_i),y_i).$$

The precision of this estimator is governed by the effective sample size

$$n_{\mathrm{eff}} = \frac{\left(\sum_i \hat w_i\right)^2}{\sum_i \hat w_i^2} \le n,$$

so the reported $\pm$ on a shifted score should use $n_{\mathrm{eff}}$, not $n$.

**Divergence bound.** Under covariate shift ($P(y\mid x)=Q(y\mid x)$), for hypothesis class $\mathcal{H}$ with $\mathcal{H}\Delta\mathcal{H}$-divergence $d_{\mathcal{H}\Delta\mathcal{H}}(P,Q)$ (Ben-David et al., *Machine Learning*, 2010):

$$R_Q(h) \;\le\; R_P(h) + \tfrac12 d_{\mathcal{H}\Delta\mathcal{H}}(P,Q) + \lambda, \qquad \lambda = \min_{h'\in\mathcal H}\left[R_P(h')+R_Q(h')\right].$$

**Assumptions and their violation status.**

| Assumption | Status in practice |
|---|---|
| $Q \ll P$ (benchmark covers use support) | **Violated.** Deployment prompts routinely have no benchmark analogue; $w$ is then undefined, not merely large. |
| Covariate shift, $P(y\mid x)=Q(y\mid x)$ | **Violated.** Deployment labels are user acceptance, not gold answers; the labeling function itself changes. |
| Benchmark items i.i.d. and unseen in training | **Violated.** Contamination and item reuse break exchangeability (§4). |
| $f$ fixed and independent of the benchmark | **Violated.** Models are selected, tuned, and sometimes trained against the benchmark, making $\hat R_P$ optimistically biased. |
| $\lambda$ small | **Unverifiable** without target labels. |

## 3. State of the Art

**Theory SOTA (established).** Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan (2010) give the $\mathcal{H}\Delta\mathcal{H}$ bound above; Ben-David, Lu, Luu, Pál (AISTATS 2010) give matching impossibility results — small covariate shift plus a small hypothesis class is still insufficient without target labels. Zhao, des Combes, Zhang, Gordon (ICML 2019) prove a lower bound on the *joint* source-target error that grows with the shift in the label marginal, ruling out invariant-representation methods as a general fix. Tibshirani, Barber, Candès, Ramdas (NeurIPS 2019) extend conformal prediction to known-likelihood-ratio covariate shift with finite-sample coverage. Angelopoulos et al. (*Science*, 2023) give prediction-powered inference: valid confidence intervals from a small labeled target sample plus a large unlabeled one.

**Empirical SOTA (established).** Recht, Roelofs, Schmidt, Shankar (ICML 2019) rebuilt ImageNet and CIFAR-10 test sets. Taori et al. (NeurIPS 2020) evaluated 200+ ImageNet models across 213 test conditions and formalized *effective robustness*. Miller et al. (ICML 2021) established "accuracy-on-the-line". Baek, Jiang, Raghunathan, Kolter (NeurIPS 2022) extend it to *agreement*-on-the-line, letting OOD accuracy be predicted from unlabeled data when the agreement line is tight. Koh et al. (ICML 2021) is the reference benchmark suite (WILDS).

**Claimed but unablated.** Unlabeled accuracy estimators — Average Thresholded Confidence (Garg et al., ICLR 2022), Difference of Confidences (Guillory et al., ICCV 2021), Mandoline (Chen et al., ICML 2021) — report low mean absolute error on curated shift suites, but the suites are the same ones used to develop them; out-of-suite calibration has not been independently reproduced at frontier-LLM scale. Chatbot Arena Elo is treated as a proxy for real use; that transfer claim exists only as a leaderboard number, never as a measured rank correlation against a deployment outcome.

## 4. What Is Known

- **Reproduction shift is large and model-independent.** ImageNet-v2: top-1 drops **11–14 points** across 30+ models; CIFAR-10: **3–15 points**. The *ranking* was largely preserved (Recht et al. 2019, ICML). Scale: ~2,000 new test images per set.
- **Effective robustness is near zero for almost all interventions.** Across 213 shifts and 200+ ImageNet models, no robustness intervention except training on more diverse data moved a model off the $P$–$Q$ accuracy line (Taori et al. 2020).
- **The line is not universal.** Accuracy-on-the-line fails on Camelyon17-WILDS and on some synthetic shifts, where correlation is near zero or negative (Miller et al. 2021).
- **Benchmark-specific inflation is measurable.** GSM1k, a held-out re-creation of GSM8K, showed drops of up to **13 points** for some open model families and near-zero drops for others, at $n = 1{,}250$ (Zhang et al., 2024, arXiv:2405.00332).
- **Leaderboard sampling is unequal.** On Chatbot Arena, a small number of providers received roughly 20% of all battle data each, and pre-release private variants were tested and withdrawn selectively (Singh et al., 2025, arXiv:2504.20879) — an $\hat R_P$ that is not an unbiased estimate of anything.
- **Benchmark noise is often larger than reported gaps.** At $n=1{,}000$ and 80% accuracy, the binomial standard error is **1.3 points**; many published rankings turn on smaller gaps (Miller, *Adding Error Bars to Evals*, 2024, arXiv:2411.00640).

## 5. What Is Not Known

- **Theoretically open.** No characterization of when $R_Q$ is identifiable under *simultaneous* covariate and concept shift with only unlabeled target data. The known results (Ben-David et al. 2010; Zhao et al. 2019) are impossibility statements for special cases, not a complete boundary.
- **Empirically open.** Whether agreement-on-the-line holds for generative LLM tasks with non-scalar outputs (agentic coding, long-form assistance). The experiment is runnable — it needs a labeled deployment slice and 30+ models — and nobody has published it at that scale.
- **Methodologically blocked.** *"The use distribution"* has no accepted operational definition. Deployment traffic is shaped by the product's own routing, by user adaptation to model behavior, and by the retention policy of the log. There is no ground truth $Q$ to sample from, so estimator coverage cannot be checked against anything. This is the binding constraint, not the theory.

## 6. Why It Is Hard

The obstruction is **non-identifiability under support mismatch, compounded by absent ground truth for $Q$**.

If $Q$ puts mass where $P$ puts none, $R_Q$ is not a functional of $P$ and the model's behavior there — every reweighting estimator returns an undefined or arbitrary value, and more benchmark data does not help. Where support does overlap, weights concentrate, and $n_{\mathrm{eff}}$ collapses (§10), so the interval widens faster than the benchmark can be grown.

Second, the target is a moving one that the measurement perturbs: publishing a benchmark causes training on it, which changes $\hat R_P$ without changing $R_Q$. Contamination makes $\hat R_P$ a biased estimator of $R_P$ *itself*, before shift is even considered.

Third, deployment "labels" are user acceptance or reward-model score, not the benchmark's gold label. So even a perfectly executed reweighting estimates $R_Q$ under the *wrong loss*. This is the "evaluation does not measure the thing it names" failure, and no amount of statistics repairs it.

## 7. Current Research (as of 2026)

- **Unlabeled performance estimation.** CMU (Kolter, Raghunathan, Lipton, Garg) on agreement-on-the-line and its failure modes; extension to generative outputs is active *(frontier — verify)*.
- **Contamination-resistant evaluation.** Private held-out splits and canary-string detection (Oren et al., ICLR 2024, *Proving Test Set Contamination in Black Box Language Models*); dynamic/regenerated benchmarks (LiveBench, LiveCodeBench). Whether regeneration preserves difficulty is itself unmeasured.
- **Prediction-powered inference applied to evals.** Berkeley (Angelopoulos, Jordan, Zrnic) — small human-labeled deployment samples used to debias large model-judged samples. Adoption inside eval pipelines is early *(frontier — verify)*.
- **Shift decomposition.** Columbia (Namkoong) and collaborators, *Diagnosing Model Performance Under Distribution Shift* — splitting a performance drop into covariate shift, concept shift, and label shift components.
- **Deployment-grounded suites.** WILDS-style construction extended to LLM agents; a public benchmark with a matched, labeled deployment slice does not yet exist.

## 8. Concrete Next Experiment

**Question.** Does benchmark rank predict use rank, and by how much does the answer change when you reweight?

**Scale.** Pick one product with retained logs (e.g. an IDE code-assistant). Sample $m = 3{,}000$ real deployment requests, stratified by request type. Human-label the correct completion for each. Evaluate $K = 20$ models spanning a 25-point range on the public benchmark (SWE-bench Verified, $n=500$, plus HumanEval).

**Control arm.** The same 20 models scored on the public benchmark alone, with binomial error bars at $n=500$. A second control: 3,000 *benchmark* items relabeled by the same human protocol, isolating label-protocol shift from input shift.

**Deciding number.** Kendall's $\tau$ between the benchmark ranking and the deployment ranking of the 20 models, with a bootstrap CI. **$\tau \ge 0.8$** means benchmark rank transfers and the field's practice is sound. **$\tau \le 0.5$** means it does not, and every deployment decision made on benchmark rank alone is unsupported. Secondary readout: $n_{\mathrm{eff}}/n$ for the importance-weighted estimator and the fraction of deployment requests with $\hat w$ undefined (no benchmark neighbor) — this quantifies the support-mismatch obstruction directly.

Cost: dominated by human labeling of 3,000 items, roughly 500 annotator-hours. Compute is negligible.

## 9. Key References

- **[Foundational]** Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan. *A Theory of Learning from Different Domains.* Machine Learning 79(1–2), 2010.
- **[Foundational]** Ben-David, Lu, Luu, Pál. *Impossibility Theorems for Domain Adaptation.* AISTATS, 2010.
- **[Foundational]** Shimodaira. *Improving Predictive Inference under Covariate Shift by Weighting the Log-Likelihood Function.* Journal of Statistical Planning and Inference, 2000.
- **[Foundational]** Torralba, Efros. *Unbiased Look at Dataset Bias.* CVPR, 2011.
- **[SOTA]** Recht, Roelofs, Schmidt, Shankar. *Do ImageNet Classifiers Generalize to ImageNet?* ICML, 2019. — arXiv:1902.10811
- **[SOTA]** Taori, Dave, Shankar, Carlini, Recht, Schmidt. *Measuring Robustness to Natural Distribution Shifts in Image Classification.* NeurIPS, 2020. — arXiv:2007.00644
- **[SOTA]** Miller, Taori, Raghunathan, Sagawa, Koh, Shankar, Liang, Carmon, Schmidt. *Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization.* ICML, 2021. — arXiv:2107.04649
- **[SOTA]** Baek, Jiang, Raghunathan, Kolter. *Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift.* NeurIPS, 2022.
- **[SOTA]** Zhao, des Combes, Zhang, Gordon. *On Learning Invariant Representations for Domain Adaptation.* ICML, 2019.
- **[SOTA]** Angelopoulos, Bates, Fannjiang, Jordan, Zrnic. *Prediction-Powered Inference.* Science 382(6671), 2023.
- **[SOTA]** Tibshirani, Barber, Candès, Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS, 2019.
- **[Benchmark]** Koh et al. *WILDS: A Benchmark of in-the-Wild Distribution Shifts.* ICML, 2021. — arXiv:2012.07421
- **[Benchmark]** Zhang et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* 2024. — arXiv:2405.00332
- **[Survey]** Liao, Taori, Raji, Schmidt. *Are We Learning Yet? A Meta Review of Evaluation Failures Across Machine Learning.* NeurIPS Datasets & Benchmarks, 2021.
- **[Survey]** Raji, Bender, Paullada, Denton, Hanna. *AI and the Everything in the Whole Wide World Benchmark.* NeurIPS Datasets & Benchmarks, 2021.
- **[Practice]** Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* 2024. — arXiv:2411.00640

## 10. Worked Example

A benchmark has $n = 500$ items in five equal strata ($p_k = 0.2$, 100 items each). Deployment traffic has $q = (0.01, 0.02, 0.07, 0.20, 0.70)$ — most real requests fall in the hardest stratum, which the benchmark treats as one-fifth of the world.

Per-stratum accuracy:

| Stratum | $p_k$ | $q_k$ | $w_k$ | Model A | Model B |
|---|---|---|---|---|---|
| 1 | 0.20 | 0.01 | 0.05 | 0.80 | 0.50 |
| 2 | 0.20 | 0.02 | 0.10 | 0.75 | 0.55 |
| 3 | 0.20 | 0.07 | 0.35 | 0.70 | 0.60 |
| 4 | 0.20 | 0.20 | 1.00 | 0.60 | 0.65 |
| 5 | 0.20 | 0.70 | 3.50 | 0.50 | 0.70 |

Benchmark scores: $\hat R_P(A) = 0.67$, $\hat R_P(B) = 0.60$. **A leads by 7 points.**

Reweighted: $\hat R_Q(A) = 0.542$, $\hat R_Q(B) = 0.678$. **B leads by 13.6 points.** The ranking reverses, with no change to either model.

Now the precision. $\overline{w^2} = (0.0025 + 0.01 + 0.1225 + 1 + 12.25)/5 = 2.677$, so

$$n_{\mathrm{eff}} = \frac{500}{2.677} \approx 187.$$

The 500-item benchmark buys the precision of 187 items. At $\hat R_Q \approx 0.6$ the standard error rises from 2.2 to 3.6 points; stratum 5 alone (100 items, 70% of the weight) contributes $0.70 \times 5.0 = 3.5$ points of that.

The obstruction is the next step. $q$ was assumed known. In practice it is estimated from logs — and if even 5% of deployment requests fall in a sixth stratum the benchmark never sampled, $p_6 = 0$, $w_6 = \infty$, and $\hat R_Q$ is undefined. The honest bound is then $[\hat R_Q^{\text{overlap}} \cdot 0.95, \ \hat R_Q^{\text{overlap}} \cdot 0.95 + 0.05]$ — a 5-point interval that no additional benchmark sampling can shrink, because the missing region is missing by construction.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*