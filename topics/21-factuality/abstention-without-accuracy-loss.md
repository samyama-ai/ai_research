---
id: 21-factuality/abstention-without-accuracy-loss
title: "Abstention Without Accuracy Loss"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Abstention Without Accuracy Loss

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/abstention-without-accuracy-loss` · **Status:** open

## 1. Problem Statement

A language model that says "I don't know" on every hard question is safe and useless. A model that never abstains hallucinates. The open problem is the corner of that trade-off that nobody has reached: **an abstention policy that removes a large fraction of the model's wrong answers while leaving its correct answers essentially untouched.**

Input: a fixed generator $M$ and a query $x$. Output: a decision $g(x, M) \in \{\text{answer}, \text{abstain}\}$, possibly using $M$'s internals, samples, or auxiliary calls. Objective: maximize the fraction of errors suppressed subject to a hard constraint that the count of correct answers retained falls by at most $\epsilon$ (e.g. $\epsilon = 1\%$).

Three variants, routinely conflated:

- **Measurement.** Does any deployed system actually achieve high error suppression at $\ge 99\%$ correct-answer retention on open-domain generation? Blocked mostly by the absence of a reliable per-claim correctness oracle at scale.
- **Method.** Build $g$. This is a selective-prediction problem where the score must rank *the model's own errors*, not instance difficulty.
- **Theory.** Characterize the achievable frontier: given a generator whose error is partly irreducible (the query's answer is simply not in the weights), what is the best possible error-suppression-at-fixed-retention, and when is it $1$?

The three have different difficulty. The theory variant has a clean answer under a strong assumption (Chow, 1970) and no answer without it.

## 2. Formal Setting

Let $P$ be a query distribution over $\mathcal{X}$, $M$ a generator producing $y = M(x)$, and $c(x,y) \in \{0,1\}$ a correctness oracle. Define base accuracy
$$a = \mathbb{E}_{x \sim P}\big[c(x, M(x))\big].$$

A selector $g: \mathcal{X} \to \{0,1\}$ ($1$ = answer) induces **coverage** $\phi = \mathbb{E}[g]$ and **selective risk**
$$R(g) = \frac{\mathbb{E}\big[g(x)\,(1 - c(x,M(x)))\big]}{\mathbb{E}[g(x)]}.$$

Two conditional quantities carry the problem:
$$\kappa_+ = \Pr\big[g = 1 \mid c = 1\big] \quad \text{(correct-answer retention)}, \qquad
\kappa_- = \Pr\big[g = 1 \mid c = 0\big] \quad \text{(error leakage)}.$$
The objective is $\min \kappa_-$ subject to $\kappa_+ \ge 1 - \epsilon$. Note $\phi = a\kappa_+ + (1-a)\kappa_-$, so at fixed $a$ the pair $(\kappa_+, \kappa_-)$ is a full ROC point; the target is the top-left corner. Sweeping a score $s(x)$ with threshold $\tau$ traces a curve whose area is AURC (area under risk-coverage) or, equivalently, AUROC of $s$ against $c$.

**How each quantity is actually measured.**
- $c$: for short-form QA, exact/alias match or an LLM judge against a gold string (SimpleQA-style grading into correct / incorrect / not-attempted). For long-form, per-claim decomposition and retrieval-based verification (FActScore). Judge agreement with humans is typically 90–95%, so measured $\kappa_-$ carries a systematic error of the same order as the effects being reported.
- $s$: sequence log-probability, token min-probability, verbalized confidence, $P(\text{IK})$ probe heads, or semantic entropy over $K$ samples, $H_{\text{sem}} = -\sum_j p_j \log p_j$ over meaning-equivalence clusters $j$.
- Abstention rate $1-\phi$: counted from the output text, which requires a classifier for "this is a refusal", itself imperfect.

**Assumptions and their violations.**
1. *$c$ is binary and well-defined.* Violated for long-form output: an answer is 70% correct.
2. *$P$ at evaluation matches deployment.* Violated — benchmarks are error-enriched relative to real traffic (Kamath et al., ACL 2020, is entirely about this).
3. *$M$ is fixed under $g$.* Violated whenever abstention is trained into the model: RLHF-style abstention tuning changes the generator, so $a$ itself moves and $\kappa_+$ is no longer identifiable from post-hoc measurement.
4. *Errors are detectable from the model's own signal.* Violated for confidently-wrong parametric recall, the case that matters most.

## 3. State of the Art

**Established (ablated, reproduced).**
- Selective classification with a softmax-response or SelectiveNet head reaches near-perfect selective accuracy at moderate coverage on closed-set vision tasks: Geifman & El-Yaniv (NeurIPS 2017) report VGG-16 on CIFAR-10 at $\approx 99.9\%$ selective accuracy with $\approx 72\%$ coverage, with guaranteed-risk bounds.
- Semantic entropy beats raw likelihood for hallucination detection on short-form QA; Farquhar et al. (*Nature*, 2024) report AUROC around $0.79$ vs $0.69$ for naive entropy across several models and datasets.
- Self-evaluation ($P(\text{True})$) and $P(\text{IK})$ probes are calibrated *in-distribution* and degrade out-of-distribution (Kadavath et al., 2022, up to 52B).
- Chow's rule is Bayes-optimal for the reject option **given the true posterior** (Chow, 1970; formalized for learned selectors by El-Yaniv & Wiener, JMLR 2010; Cortes, DeSalvo & Mohri, ALT 2016).

**Claimed but unablated.**
- That abstention fine-tuning (R-Tuning, NAACL 2024; various "know what you don't know" tunings) suppresses hallucination *without* accuracy loss. The published tables usually report accuracy on the answered subset and abstention rate; the decomposition into $\kappa_+$ and $\kappa_-$, which is what the claim requires, is rarely reported.
- That RLHF-era models "know" their limits. Yin et al. (ACL Findings 2023) and Cheng et al. (ICML 2024) show partial self-knowledge; neither isolates it from topic-level difficulty priors.

**Benchmark-number-only.** SimpleQA's "not attempted" rate, XSTest over-refusal counts, and most abstention leaderboards are single scalars at one operating point. They do not give a risk-coverage curve, so they cannot distinguish a good selector from a model that simply abstains more.

## 4. What Is Known

- **Frontier LMs are badly wrong-but-confident on long-tail facts.** On SimpleQA (Wei et al., 2024, ~4.3k adversarially-filtered short questions), leading 2024-era models scored roughly 40% correct or below, and stated confidences far exceeded accuracy.
- **Verbalized confidence is coarse but not useless.** Tian et al. (EMNLP 2023) found verbalized probabilities from RLHF'd GPT-4-class models better calibrated (lower ECE) than conditional token likelihoods on QA — a reversal of the pre-RLHF ordering.
- **Sampling-based consistency is the strongest cheap signal.** Lin, Trivedi & Sun (TMLR 2024) and semantic-entropy work put black-box AUROC in the 0.75–0.85 band on short-form QA at 5–10 samples.
- **Abstention transfers poorly across distributions.** Kamath et al. (ACL 2020) trained a calibrator on SQuAD; selective accuracy held in-domain and dropped sharply on out-of-domain QA at matched coverage.
- **Calibration and honesty conflict.** Kalai & Vempala (STOC 2024) prove that for facts appearing once in training ("singletons"), a calibrated generative model must hallucinate at a rate lower-bounded by the singleton fraction — an information-theoretic floor of a few percent for arbitrary-fact domains. Kalai et al. (2025) argue the residual is sustained by binary scoring rules that reward guessing.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the achievable $(\kappa_+, \kappa_-)$ frontier for a *fixed* generator whose errors mix aleatoric (fact absent from weights) and epistemic (fact present, decoding failed) causes. Chow-optimality assumes the true posterior; nothing tells you how much of the error mass is separable from correct answers by *any* function of the model's activations.
- **Empirically open.** Whether any method achieves $\kappa_- \le 0.4$ at $\kappa_+ \ge 0.99$ on open-domain short-form QA at frontier scale. The experiment is runnable today; published work reports AUROC, not this constrained corner, and AUROC $=0.8$ is compatible with terrible behavior at $\kappa_+ = 0.99$.
- **Methodologically blocked.** Long-form abstention. There is no agreed definition of "abstained without accuracy loss" when the model could have hedged one clause of a ten-claim paragraph. Partial abstention has no accepted metric.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**.

1. *Confounding.* Every abstention score correlates with question difficulty, and difficulty correlates with correctness. A selector that abstains on rare entities looks good on AUROC while destroying exactly the rare-entity answers the model got right — the loss shows up in $\kappa_+$, which almost nobody reports.
2. *Non-identifiability under training.* When abstention is fine-tuned in, $M$ and $g$ change together. Post-hoc, you cannot tell "abstained on an item it would have missed" from "lost the knowledge". Recovering $\kappa_+$ requires the pre-tuning model's per-item correctness, a control arm that costs a second full evaluation and is usually skipped.
3. *Oracle noise floor.* At $\kappa_+ = 0.99$ you are measuring a 1% effect with a judge whose disagreement with humans is 5–10%. The measurement error exceeds the constraint.
4. Not primarily compute: the blocking cost is human-verified labels, not GPU hours.

## 7. Current Research (as of 2026)

- **Uncertainty estimation for generation** — semantic entropy and successors (OUI, Oxford; Gal group), cheap probe-based approximations to avoid $K$ samples.
- **Abstention tuning and honesty alignment** — R-Tuning-style SFT, HonestLLM/"I don't know" datasets, multi-LLM collaboration for knowledge-gap identification (Feng et al., ACL 2024).
- **Scoring-rule reform** — proposals to grade benchmarks with explicit confidence targets and negative marking so abstention is not penalized (Kalai et al., 2025). *(frontier — verify uptake by major leaderboards.)*
- **Retrieval-conditioned abstention** — abstain when retrieval support is insufficient rather than when the model feels unsure; sidesteps parametric self-knowledge but changes the system, not the model.
- **Survey anchor:** Wen et al., *Know Your Limits: A Survey of Abstention in Large Language Models*, TACL 2025.

## 8. Concrete Next Experiment

**The Retention-Constrained Abstention Benchmark.**

- **Scale.** 3,000 short-form questions: 1,500 from SimpleQA (long-tail, low base accuracy) and 1,500 from Natural Questions (head, high base accuracy), so the head/tail confound is measurable. Three generators spanning capability (an 8B open model, a 70B open model, one frontier API model). Correctness graded by an LLM judge, with 400 items double-annotated by humans to bound judge error.
- **Arms.** (a) sequence log-prob threshold; (b) verbalized confidence; (c) semantic entropy, $K=10$; (d) trained probe on final-layer activations; (e) abstention-tuned generator.
- **Control arm.** The *same* generator with abstention disabled, evaluated item-by-item, giving the per-item correctness vector $c_i$. Every arm's $\kappa_+$ and $\kappa_-$ is computed against this vector. For arm (e), the control is the pre-tuning checkpoint — without it $\kappa_+$ is not identifiable.
- **Deciding number.** $\kappa_-$ at the threshold where $\kappa_+ = 0.99$, reported separately for head and tail slices, with bootstrap CIs. A method is a genuine advance if $\kappa_- \le 0.40$ at $\kappa_+ \ge 0.99$ on the tail slice — i.e. 60% of errors removed for 1% of correct answers. Current expectation, extrapolating from AUROC $\approx 0.8$: $\kappa_- \in [0.85, 0.95]$. That gap is the problem.

## 9. Key References

- **[Foundational]** C. K. Chow. *On Optimum Recognition Error and Reject Tradeoff.* IEEE Transactions on Information Theory, 1970.
- **[Foundational]** Ran El-Yaniv, Yair Wiener. *On the Foundations of Noise-free Selective Classification.* JMLR, 2010.
- **[Foundational]** Corinna Cortes, Giulia DeSalvo, Mehryar Mohri. *Learning with Rejection.* ALT, 2016.
- **[Foundational]** Yonatan Geifman, Ran El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS, 2017. — arXiv:1705.08500
- **[SOTA]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature, 2024.
- **[SOTA]** Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023. — arXiv:2302.09664
- **[SOTA]** Zhen Lin, Shubhendu Trivedi, Jimeng Sun. *Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models.* TMLR, 2024. — arXiv:2305.19187
- **[Empirical]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* 2022. — arXiv:2207.05221
- **[Empirical]** Amita Kamath, Robin Jia, Percy Liang. *Selective Question Answering under Domain Shift.* ACL, 2020.
- **[Empirical]** Katherine Tian et al. *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* EMNLP, 2023.
- **[Empirical]** Hanning Zhang et al. *R-Tuning: Instructing Large Language Models to Say 'I Don't Know'.* NAACL, 2024.
- **[Theory]** Adam Tauman Kalai, Santosh S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Benchmark]** Jason Wei et al. *Measuring Short-form Factuality in Large Language Models (SimpleQA).* OpenAI, 2024.
- **[Survey]** Bingbing Wen et al. *Know Your Limits: A Survey of Abstention in Large Language Models.* TACL, 2025. — arXiv:2407.18418

## 10. Worked Example

Take a generator with $a = 0.40$ on 1,000 SimpleQA-style items: 400 correct, 600 wrong. Suppose a semantic-entropy selector has AUROC $= 0.80$ against correctness — at the top of the published range.

Model the entropy score for correct items as $\mathcal{N}(0,1)$ and for wrong items as $\mathcal{N}(\mu,1)$; AUROC $=\Phi(\mu/\sqrt{2}) = 0.80 \Rightarrow \mu = 1.19$.

Set the threshold to retain 99% of correct answers: abstain when $s > \tau$ with $\tau = \Phi^{-1}(0.99) = 2.33$. Then
$$\kappa_- = \Pr[s \le 2.33 \mid \text{wrong}] = \Phi(2.33 - 1.19) = \Phi(1.14) = 0.873.$$

Outcome: 396 correct answers kept (4 lost), 524 of 600 errors still emitted. Total abstentions: 80, of which 76 are errors. Selective risk moves from $0.600$ to $524/920 = 0.570$ — a 3-point improvement for a system that has "hallucination detection AUROC 0.80".

Push to $\kappa_+ = 0.90$ ($\tau = 1.28$): $\kappa_- = \Phi(0.09) = 0.536$; 360 correct kept, 322 errors emitted, selective risk $0.472$. You bought a 13-point risk reduction by deleting 40 correct answers — exactly the accuracy loss the problem forbids.

**What this makes visible:** the obstruction is not the average-case ranking quality. AUROC $0.80$ is respectable and still leaves 87% of errors in place at the 99%-retention operating point, because the constraint lives in the extreme tail of the score distribution where the two class-conditionals are nearly identical. Reaching $\kappa_- = 0.40$ at $\kappa_+ = 0.99$ under this Gaussian model needs $\mu = 2.58$, i.e. AUROC $\approx 0.966$ — a regime no published hallucination detector approaches on open-domain generation. Reporting AUROC hides that the required improvement is not incremental.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*