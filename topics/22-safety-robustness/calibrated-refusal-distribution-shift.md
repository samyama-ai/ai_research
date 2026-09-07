---
id: 22-safety-robustness/calibrated-refusal-distribution-shift
title: "Calibrated Refusal Under Distribution Shift"
topic: 22-safety-robustness
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibrated Refusal Under Distribution Shift

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/calibrated-refusal-distribution-shift` · **Status:** empirically-open

## 1. Problem Statement

A deployed language model must decide, per input, whether to answer or refuse. Refusal is asked to carry two different jobs at once: **competence abstention** (the model does not know the answer) and **policy refusal** (the answer would be harmful). The problem is to build a refusal rule whose stated risk guarantee still holds when the input distribution moves away from the one used to set the rule.

- **Input:** prompt $x$, model $f$, a refusal policy $r_\theta(x) \in \{0,1\}$ tuned on a calibration set from $P_{\text{cal}}$.
- **Output:** answer $f(x)$ when $r_\theta(x)=0$, refusal otherwise.
- **Predicate for a solution:** for a deployment distribution $Q \neq P_{\text{cal}}$ drawn from a stated shift class, the *selective risk* on answered inputs stays at or below the target $\alpha$, while the refusal rate on benign inputs stays at or below a budget $\beta$ — with a certificate computed *before* seeing $Q$'s labels.

Three variants, different difficulty:

- **Measurement:** define "correct refusal" so that over-refusal and under-refusal are on one scale. Currently blocked — harmfulness labels and correctness labels come from different annotation processes.
- **Method:** build the selector. Conformal and selective-classification machinery exists; it is untested at the shift magnitudes deployment actually sees.
- **Theory:** characterise which shift classes admit a distribution-free risk certificate for a two-sided (refuse-too-little / refuse-too-much) loss. Open.

## 2. Formal Setting

Let $\mathcal{X}$ be prompts, $\mathcal{Y}$ responses. Two latent labels per prompt:

- $h(x) \in \{0,1\}$: whether a compliant answer is harmful (policy ground truth, obtained by human annotation against a written policy; measured as majority of $k \geq 3$ annotators, with inter-annotator agreement $\kappa$ reported).
- $c(x, y) \in \{0,1\}$: whether response $y$ is correct/acceptable (measured by task-specific grader — exact match, rubric, or LLM judge with human-verified agreement rate).

Loss on an answered prompt:
$$\ell(x,y) = \lambda_h \, h(x) + \lambda_c \, \big(1 - h(x)\big)\big(1 - c(x,y)\big),$$
with $\lambda_h \gg \lambda_c$. Refusal costs $\lambda_r \, (1 - h(x))$ — refusing a benign prompt is the only refusal that costs anything.

The selector uses a score $s(x) \in \mathbb{R}$ (max token logprob, semantic entropy, a trained probe, a judge score) and threshold $\tau$: $r_\tau(x) = \mathbb{1}[s(x) > \tau]$. Define coverage and selective risk under $Q$:
$$\mathrm{cov}_Q(\tau) = \Pr_{Q}[s(x) \le \tau], \qquad R_Q(\tau) = \frac{\mathbb{E}_{Q}\big[\ell(x, f(x)) \, \mathbb{1}[s(x) \le \tau]\big]}{\mathrm{cov}_Q(\tau)}.$$
$\tau$ is chosen on $n$ calibration points as $\hat\tau = \min\{\tau : \hat{R}_{P}(\tau) + \epsilon(n,\delta) \le \alpha\}$, where $\epsilon$ is a finite-sample bound (Wilson or the SGR bound of Geifman & El-Yaniv).

Under covariate shift with density ratio $w(x) = dQ/dP(x)$ and bounded $\|w\|_\infty$, weighted conformal prediction gives coverage guarantees (Tibshirani et al., 2019). Assumptions actually required, and their status:

| Assumption | Status in practice |
|---|---|
| Exchangeability of calibration and test prompts | **Violated.** Deployment traffic is adversarially selected and time-correlated. |
| $Y \mid X$ unchanged (covariate shift only) | **Violated.** Policy definitions of harm change; the model itself is updated. |
| $w(x)$ known or estimable | **Violated.** Estimating $w$ over prompt space is as hard as the original problem. |
| $\ell$ observable at calibration time | **Partly violated.** $h(x)$ is contested for ~10–30% of borderline prompts. |
| Score $s$ is shift-stable | **Unverified.** Score distributions move under shift; this is the effect being certified against. |

## 3. State of the Art

**Theory SOTA (established).** Chow's rule (1970) is Bayes-optimal for reject-option classification under a known posterior. El-Yaniv & Wiener (JMLR 2010) give the risk–coverage framework; Geifman & El-Yaniv (NeurIPS 2017) give a finite-sample selective-risk bound at a chosen coverage. Vovk et al. (2005) and Angelopoulos & Bates (2021) give distribution-free coverage under exchangeability; Tibshirani et al. (NeurIPS 2019) extend to covariate shift with known $w$; Barber et al. (Ann. Statist. 2023) give coverage-gap bounds proportional to total-variation distance when exchangeability fails; Gibbs & Candès (NeurIPS 2021) give online adaptive conformal inference that recovers long-run coverage without distributional assumptions but with no per-window guarantee.

**Empirical SOTA (established, single-model or single-benchmark).** Kamath, Jia & Liang (ACL 2020) show a calibrator trained on in-domain plus a small out-of-domain sample raises selective QA accuracy under domain shift versus a maxprob baseline. Kuhn et al. (ICLR 2023) and Farquhar et al. (Nature 2024) show semantic entropy beats token-level entropy for detecting confabulations. Quach et al. (ICLR 2024) apply conformal calibration to LM generation sets.

**Claimed but unablated.** That RLHF'd assistants "know what they don't know" (Kadavath et al., 2022) is supported for multiple-choice self-evaluation, not for open-ended generation under shift. That refusal training generalises to novel harm categories is asserted in system cards; the ablation separating memorised refusal triggers from generalising ones is not public. **Benchmark-number-only:** XSTest (Röttger et al., NAACL 2024) and OR-Bench (Cui et al., 2024) report over-refusal rates but do not certify anything, and models are frequently tuned against them, so the numbers no longer track the underlying capability.

## 4. What Is Known

- **Calibration degrades monotonically with shift severity.** Ovadia et al. (NeurIPS 2019), across ImageNet-C/CIFAR-10-C and ~10 methods at ResNet scale: ECE rises several-fold from clean to severity-5 corruption for every method tested; deep ensembles degrade least. This is the most reproduced regularity in the area — and it was measured on vision classifiers, not LMs.
- **RLHF worsens calibration.** GPT-4 technical report (OpenAI, 2023): MMLU calibration ECE $\approx 0.007$ pre-RLHF, $\approx 0.074$ post-RLHF — a ~10× degradation at frontier scale.
- **Over-refusal is large and model-specific.** XSTest (250 safe prompts, NAACL 2024): Llama-2-70B-chat fully refused about 38% of clearly safe prompts; GPT-4 was in the low single digits. OR-Bench (2024) constructs ~80k seemingly-toxic-but-safe prompts and an OR-Bench-Hard-1K subset where leading models refuse the majority.
- **Refusal is shallow.** Wei, Haghtalab & Steinhardt (NeurIPS 2023) show competing-objective and mismatched-generalisation jailbreaks; Zou et al. (2023) show gradient-optimised suffixes transferring across models. A distribution shift as small as a base64 encoding flips refusal.
- **Exchangeability failure has a bound.** Barber et al. (2023): coverage loss is at most the average total-variation distance between calibration and test — vacuous exactly when shift is large.

## 5. What Is Not Known

- **Methodologically blocked:** a single scale on which over-refusal and under-refusal trade off. Setting $\lambda_h/\lambda_r$ requires a harm-per-benign-refusal exchange rate no public policy states, and $h(x)$ itself has no ground truth for the borderline prompts that dominate the decision region.
- **Empirically open:** whether *any* selector — semantic entropy, linear probe on residual stream, or a separate judge model — holds selective risk within $2\times$ its calibrated target across a shift ladder of stated severity at 70B+ scale. The experiment is runnable today; nobody has published it with a control arm.
- **Theoretically open:** whether a distribution-free certificate exists for the two-sided loss under an *adversarially chosen* $Q$ within a bounded-$f$-divergence ball, when $w$ must be estimated from unlabelled deployment traffic. Known results assume $w$ given.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the label under shift, compounded by adversarial selection**. Conformal guarantees convert an assumption about $Q$ into a certificate; the assumption available (bounded density ratio) is unverifiable, and the case that matters — jailbreak traffic — is precisely the case where $\|w\|_\infty$ is unbounded because the attacker picks $x$ to maximise it. Second obstruction: **the evaluation does not measure what it names**. XSTest and OR-Bench measure refusal-string frequency on a fixed prompt set, not risk at a stated coverage; a model can score well by pattern-matching benign-sounding phrasings. Third: labelling $h(x)$ on 10k borderline shifted prompts at $\kappa > 0.7$ costs roughly 200–400 annotator-hours per shift condition, which is why the ladder has not been built.

## 7. Current Research (as of 2026)

- Conformal risk control for generation — Angelopoulos, Bates, Candès and collaborators (Berkeley/Stanford); extensions to multi-objective and shifted settings *(frontier — verify)*.
- Semantic-entropy and probe-based hallucination detection — Gal's group (OMLat, Oxford), following Farquhar et al. (Nature 2024).
- Contextual noncompliance taxonomies — Allen Institute for AI (Brahman et al., NeurIPS 2024 D&B), which separates refusal types rather than collapsing them into one rate.
- Refusal-direction and representation-level analysis of why refusal fails off-distribution *(frontier — verify)*; several 2024–2025 preprints report a low-dimensional refusal direction in residual space, with ablations of varying quality.
- Frontier-lab robustness work (Anthropic, OpenAI, GDM) reports jailbreak-resistance metrics in system cards; per-coverage selective-risk certificates are not reported by any lab as of this writing.

## 8. Concrete Next Experiment

**Question:** does a selector calibrated in-distribution hold its selective-risk target under a graded shift, and does any method beat a fixed-threshold control?

- **Scale:** one open 70B-class instruct model. Calibration set $n = 5{,}000$ prompts from a fixed benign+harmful mixture (60/40), each labelled for $h$ by 3 annotators and for $c$ by rubric. Five shift rungs, 1,000 labelled prompts each: (1) paraphrase, (2) non-English (3 languages), (3) domain change (medical/legal/chemistry), (4) encoded/obfuscated (base64, leetspeak, role-play framing), (5) optimised adversarial suffixes (GCG-style, held-out from any training). Total ~10k labelled prompts.
- **Arms:** (a) **control** — fixed threshold on max-token-logprob calibrated on $P_{\text{cal}}$ to $\alpha = 0.05$ at coverage 0.80; (b) semantic entropy; (c) trained residual-stream probe; (d) separate judge model; (e) weighted conformal with $w$ estimated by a domain classifier on unlabelled shifted prompts.
- **Deciding number:** the **worst-rung selective-risk inflation ratio** $\max_k R_{Q_k}(\hat\tau) / \alpha$ at matched coverage $\ge 0.75$. If no arm keeps this below 2.0 on rungs 1–4, the "method" variant is empirically refuted at this scale and the problem moves to the score-design layer. Report benign refusal rate $\beta$ at the same $\tau$ as the second axis; an arm that hits the risk target by refusing 60% of benign traffic does not count.
- Cost estimate: ~10k prompts × 3 annotations + 5 arms × 10k generations — order 500 annotator-hours and a few thousand GPU-hours.

## 9. Key References

- **[Foundational]** C. K. Chow. *On Optimum Recognition Error and Reject Tradeoff.* IEEE Transactions on Information Theory, 1970.
- **[Foundational]** R. El-Yaniv, Y. Wiener. *On the Foundations of Noise-free Selective Classification.* JMLR, 2010.
- **[Foundational]** Y. Geifman, R. El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS, 2017. — arXiv:1705.08500
- **[Foundational]** V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[SOTA — theory]** R. J. Tibshirani, R. Foygel Barber, E. Candès, A. Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS, 2019. — arXiv:1904.06019
- **[SOTA — theory]** R. Foygel Barber, E. Candès, A. Ramdas, R. J. Tibshirani. *Conformal Prediction Beyond Exchangeability.* Annals of Statistics, 2023. — arXiv:2202.13415
- **[SOTA — theory]** I. Gibbs, E. Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS, 2021. — arXiv:2106.00170
- **[SOTA — empirical]** A. Kamath, R. Jia, P. Liang. *Selective Question Answering under Domain Shift.* ACL, 2020. — arXiv:2006.09462
- **[SOTA — empirical]** S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature, 2024.
- **[SOTA — empirical]** L. Kuhn, Y. Gal, S. Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023. — arXiv:2302.09664
- **[Evidence]** Y. Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[Evidence]** S. Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Evidence]** A. Wei, N. Haghtalab, J. Steinhardt. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS, 2023. — arXiv:2307.02483
- **[Evidence]** A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, M. Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[Benchmark]** P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, D. Hovy. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* NAACL, 2024. — arXiv:2308.01263
- **[Benchmark]** J. Cui, W.-L. Chiang, I. Stoica, C.-J. Hsieh. *OR-Bench: An Over-Refusal Benchmark for Large Language Models.* 2024. — arXiv:2405.20947
- **[Benchmark]** M. Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML, 2024. — arXiv:2402.04249
- **[Survey]** A. N. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021. — arXiv:2107.07511
- **[Survey]** F. Brahman et al. *The Art of Saying No: Contextual Noncompliance in Language Models.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2407.12043

## 10. Worked Example

Calibrate on $n = 2{,}000$ English prompts, score $s$ = negative max-token-logprob, target $\alpha = 0.05$ at coverage $0.80$. Empirically $\hat\tau$ answers 1,600 prompts with 78 losses: $\hat R_P = 78/1600 = 0.049$. The Wilson 95% upper bound at $n=1600$ is about $0.049 + 1.96\sqrt{0.049 \cdot 0.951/1600} \approx 0.060$. Certificate reads: risk $\le 0.06$ at 80% coverage.

Now apply the same $\hat\tau$ to 1,000 base64-encoded versions of the same prompts. Encoding raises token-level surprisal for benign and harmful prompts alike, so the score distribution shifts *right*: coverage falls to, say, $0.72$. But the harmful prompts that survive the threshold are the ones the model handles fluently — and refusal training keyed on plaintext triggers no longer fires. Suppose 101 of the 720 answered prompts incur loss: $R_Q = 0.140$. Inflation ratio $0.140/0.05 = 2.8$.

Try to repair it with weighted conformal. It needs $w(x) = dQ/dP$. A domain classifier separates plaintext from base64 essentially perfectly, so $\hat w$ is near 0 on the calibration support and near $\infty$ on the test support — the effective sample size $\big(\sum_i \hat w_i\big)^2 / \sum_i \hat w_i^2$ collapses toward 1, and the interval widens to the trivial one. The certificate does not become wrong; it becomes vacuous.

That is the obstruction in one number. Coverage went *down* (the model got less confident) while risk went *up* by $2.8\times$ — so refusal rate, the quantity every public benchmark reports, moved in the direction that looks safer while the quantity that matters got worse. Any evaluation that tracks refusal rate rather than selective risk at matched coverage would score this shift as an improvement.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*