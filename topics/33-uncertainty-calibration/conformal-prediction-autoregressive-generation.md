---
id: 33-uncertainty-calibration/conformal-prediction-autoregressive-generation
title: "Conformal Prediction for Autoregressive Generation"
topic: 33-uncertainty-calibration
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Conformal Prediction for Autoregressive Generation

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/conformal-prediction-autoregressive-generation` · **Status:** open

## 1. Problem Statement

Conformal prediction turns any heuristic score into a set-valued predictor with a finite-sample marginal coverage guarantee, assuming exchangeable data. The open problem is whether this machinery gives anything useful for autoregressive generation, where the output is a variable-length sequence from a combinatorial space and the "correct answer" is a set, not a point.

Three variants, of different difficulty:

- **Measurement.** What is the target set? A set of token continuations, a set of whole sequences, or a set of *claims* inside one generation. Each yields a different guarantee, and only the last is usually what a user wants.
- **Method.** Given a target, construct a calibrated wrapper on a frozen LLM that is small enough to be useful: coverage $1-\alpha$ at a set size or claim-retention rate a user would accept.
- **Theory.** Autoregressive decoding violates exchangeability by construction — the score for token $t$ depends on tokens $1{:}t-1$ that the model itself produced. Establish which guarantees survive, and with what coverage gap.

Solving it means: a wrapper on a frozen model that, for a specified user-visible correctness predicate, delivers $\ge 1-\alpha$ coverage under distribution shift at test time, with an efficiency (set size / retained content) penalty a practitioner accepts, and a coverage gap bound that does not degenerate to vacuity.

## 2. Formal Setting

Prompt $X \in \mathcal{X}$, generation $Y = (y_1,\dots,y_T) \in \mathcal{V}^{\le T_{\max}}$, model $p_\theta(y_t \mid x, y_{<t})$. Calibration set $\{(X_i, Y_i)\}_{i=1}^n$; test point $(X_{n+1}, Y_{n+1})$.

**Admission predicate.** $A(x, y) \in \{0,1\}$ — measured, not assumed. In practice it is a human label, an exact-match check against a reference, or an LLM-judge call. Its noise rate is part of the guarantee.

**Nonconformity score.** $s: \mathcal{X}\times\mathcal{Y}\to\mathbb{R}$. Measured candidates: negative sequence log-likelihood $-\log p_\theta(y\mid x)$; length-normalized $-\frac1T\log p_\theta$; self-consistency frequency over $m$ samples; a trained score head.

**Split conformal.** With $\hat q = $ the $\lceil (n+1)(1-\alpha)\rceil$-th smallest of $\{s(X_i,Y_i)\}$,
$$C(x) = \{y : s(x,y)\le \hat q\}, \qquad 1-\alpha \;\le\; \Pr[Y_{n+1}\in C(X_{n+1})] \;\le\; 1-\alpha+\tfrac{1}{n+1}.$$

**Sampling-based sets.** $C(x)$ is not enumerable, so it is realized as a stopping rule over $m$ sampled generations: emit $\hat{\mathcal{C}}_\lambda(x)\subseteq\{y^{(1)},\dots,y^{(m)}\}$, and calibrate $\lambda$ for
$$\mathbb{E}\big[\ell(\hat{\mathcal{C}}_\lambda(X_{n+1}), Y_{n+1})\big] \le \alpha, \quad \ell = \mathbb{1}\{\nexists\, y\in\hat{\mathcal{C}}_\lambda : A(X,y)=1\},$$
which is conformal risk control, not vanilla conformal prediction.

**Claim-level (backoff).** Decompose $y$ into subclaims $c_1,\dots,c_K$; output $F_\lambda(y)$ = the subset with confidence $\ge\lambda$. Target $\Pr[\text{every retained claim is true}]\ge 1-\alpha$. Efficiency is **retention** $\mathbb{E}[|F_\lambda(y)|/K]$.

**Assumptions and their status:**

| Assumption | Status in practice |
|---|---|
| Exchangeability of $(X_i,Y_i)$ | Violated: deployment prompts drift; $Y$ is model-generated, so scores shift with any decoder or checkpoint change |
| $A$ observed without noise | Violated: LLM-judge agreement with humans is typically 0.7–0.85 |
| Single $\alpha$ per user | Violated: coverage is marginal over prompts, not per-prompt |
| Frozen $p_\theta$ during calibration | Holds only until the next model update |

Exact **conditional** coverage $\Pr[Y\in C(x)\mid X=x]\ge 1-\alpha$ for all $x$ is impossible distribution-free with nontrivial sets (Vovk 2012; Lei & Wasserman 2014; Barber et al. 2021).

## 3. State of the Art

**Established.**
- *Conformal Language Modeling* (Quach, Fisch, Schuster, Yala, Sohn, Jaakkola, Barzilay, ICLR 2024) — calibrated sampling-with-rejection stopping rules; guarantees at least one admissible output per prompt with high probability, via Learn-then-Test multiple testing. Validated on MIMIC-CXR report generation, CNN/DM, TriviaQA.
- *Conformal Factuality* (Mohri & Hashimoto, ICML 2024) — claim backoff with a coverage guarantee on the filtered output. This is the first construction whose guarantee is about the text a user reads.
- *Conformal Risk Control* (Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024) — extends coverage to any bounded monotone loss; the enabling machinery for generation.
- *Conformal prediction beyond exchangeability* (Barber, Candès, Ramdas, Tibshirani, Annals of Statistics 2023) — weighted conformal with an explicit coverage gap in total-variation terms.

**Claimed but unablated.**
- Length-normalized log-likelihood as a nonconformity score is used almost universally without an ablation against a learned score at matched compute.
- *Non-Exchangeable Conformal Language Generation with Nearest Neighbors* (Ulmer, Zerva, Martins, Findings of EACL 2024) applies Barber et al.'s weighting at the token level; the empirical coverage improvement is reported, the bound is not tight enough to certify it.
- *Conformal Nucleus Sampling* (Ravfogel, Goldberg, Goldberger, Findings of ACL 2023) — calibrates top-$p$ by entropy bucket; a decoding improvement, whose sequence-level implication is unestablished.

**Benchmark-number-only.** Nearly all reported efficiency figures (set sizes, retention rates) are single-dataset, single-model, and single-judge. No cross-model, cross-judge reproduction of an efficiency number exists.

## 4. What Is Known

- **Finite-sample coverage is exact and tight.** For split conformal with $n$ calibration points, coverage lies in $[1-\alpha,\ 1-\alpha+\frac{1}{n+1}]$. At $n=1000$, $\alpha=0.1$, the quantile index is 901 and the overshoot ceiling is $0.001$. This is a theorem, not an empirical claim, and it holds for LLMs.
- **Multiple-choice QA is solved-ish.** Over a fixed label set of 4 options, conformal sets on softmax scores give exact coverage; measured average set size grows with model uncertainty. Demonstrated across ~10 open LLMs in the 7B–70B range on MMLU-style benchmarks (Ye et al., NeurIPS 2024 Datasets & Benchmarks). The finite label space is why it works — it does not transfer to open generation.
- **Claim backoff costs a lot of content.** Reported retention falls steeply as $\alpha$ tightens; at $\alpha \approx 0.1$ on FActScore-style biography generation, a large fraction of subclaims is removed, to the point where the output's usefulness is contested. The exact fraction is model- and judge-dependent and has not been reproduced across labs.
- **Non-exchangeability bound.** Barber et al. (2023): with weights $w_i$, the coverage gap is at most $\frac{\sum_i w_i\, d_{\mathrm{TV}}(\text{data}, \text{swapped})}{1+\sum_i w_i}$. For LLM deployment shift, $d_{\mathrm{TV}}$ is not measurable, so the bound is correct and empty.
- **Adaptive conformal (ACI, Gibbs & Candès, NeurIPS 2021)** guarantees long-run coverage $|\frac1T\sum_t \mathrm{err}_t - \alpha| = O(1/(\gamma T))$ with *no* distributional assumption — but only in the online, feedback-available regime.

## 5. What Is Not Known

- **Theoretically open.** Whether any nonvacuous, *computable* coverage gap exists for conformal sets over model-generated sequences, where the score distribution shifts because the model produced the conditioning tokens. Also open: whether token-level per-step guarantees can be composed into a sequence-level guarantee better than the union bound $T\alpha_{\text{step}}$.
- **Empirically open.** Whether a learned nonconformity score beats length-normalized log-likelihood on efficiency at matched inference cost, at 70B+ scale, across ≥3 task families. Runnable today; unrun at that scale.
- **Methodologically blocked.** "Correctness" of a free-form generation. Every guarantee is conditional on $A$, and $A$ is an LLM judge or a noisy human label. A guarantee of 90% coverage w.r.t. a judge that agrees with humans 80% of the time is not a 90% guarantee about truth, and no accepted correction for judge noise exists.

## 6. Why It Is Hard

The specific obstruction is **guarantee/target mismatch compounded by an unmeasured admission predicate**.

1. *Combinatorial output space.* $C(x)$ over $\mathcal{V}^{T}$ cannot be enumerated; it is approximated by $m$ samples, so the guarantee silently becomes "the sampler found an admissible output", not "the set covers the truth." Per-token sets of size $k$ compose to $k^T$ sequences — at $k=5, T=20$ that is $10^{14}$, so token-level conformal is unusable at the sequence level.
2. *Marginal, not conditional.* Coverage averages over prompts. A wrapper can hit 90% marginal coverage while covering 99% of easy prompts and 60% of hard ones — exactly inverted from what a user needs. Distribution-free conditional coverage is provably unattainable.
3. *Absent ground truth.* $A$ is not observed. Judge noise propagates into the calibration quantile in a way nobody has bounded.
4. *Self-induced shift.* The calibration scores are computed on generations from a model that changes with every decoder, prompt-template, or checkpoint change, invalidating exchangeability without any distribution shift in the user's data.

## 7. Current Research (as of 2026)

- **Claim-level and reasoning-level guarantees.** Stanford (Hashimoto, Candès groups) — *Large language model validity via enhanced conformal prediction* (Cherian, Gibbs, Candès, NeurIPS 2024) adds conditional-coverage-like behavior via a level-adaptive filter. *Conformal Language Model Reasoning with Coherent Factuality* (Rubin-Toles et al., ICLR 2025) extends backoff from independent claims to dependency graphs over reasoning steps — necessary because removing a claim in a proof breaks the ones downstream.
- **Black-box / API-only settings.** Calibration from sampled outputs without logits (Su et al., Findings of EMNLP 2024). Practically important; the efficiency cost of dropping logit access is not characterized.
- **Conformal control for agents and RAG.** Calibrated abstain/escalate thresholds under risk control, rather than set prediction. *(frontier — verify)*
- **Judge-noise-aware calibration.** Treating $A$ as a noisy label and correcting the quantile. Active, no accepted method. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does conformal claim-backoff deliver its guarantee under a realistic prompt shift, and does a learned score buy efficiency?

- **Scale.** One open 70B-class instruct model, frozen. Three task families (biography generation, medical QA long-form, multi-hop QA), 2,000 prompts each: 1,000 calibration, 1,000 test. Two judges (a strong LLM judge and 300 human-labeled prompts per family for judge-noise estimation). Total compute: roughly 12k generations plus judge calls — under 500 A100-hours.
- **Control arm.** Split conformal claim backoff with length-normalized log-likelihood as the score, calibrated and tested **in-distribution** (i.i.d. split of the same pool). This is the arm that is guaranteed to work.
- **Treatment arms.** (a) same score, calibrated on family A, tested on family B (shift); (b) learned score head calibrated on A, tested on A and B.
- **Deciding number.** **Empirical claim-level coverage on the shifted test set at $\alpha=0.1$, at matched retention of 0.60.** If shifted coverage stays $\ge 0.88$, the guarantee is robust enough to deploy. If it drops below $0.80$, conformal generation guarantees are in-distribution artifacts and the field should report shifted coverage as the headline metric. Secondary number: retention at matched coverage 0.90, learned vs. log-likelihood score — a gain under 5 points means the score choice is not the bottleneck.

## 9. Key References

- **[Foundational]** Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[Foundational]** Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA, 2018.
- **[Foundational]** Barber, Candès, Ramdas, Tibshirani. *Conformal prediction beyond exchangeability.* Annals of Statistics, 2023.
- **[Foundational]** Barber, Candès, Ramdas, Tibshirani. *The limits of distribution-free conditional predictive inference.* Information and Inference, 2021.
- **[SOTA]** Quach, Fisch, Schuster, Yala, Sohn, Jaakkola, Barzilay. *Conformal Language Modeling.* ICLR, 2024.
- **[SOTA]** Mohri, Hashimoto. *Language Models with Conformal Factuality Guarantees.* ICML, 2024.
- **[SOTA]** Cherian, Gibbs, Candès. *Large language model validity via enhanced conformal prediction methods.* NeurIPS, 2024.
- **[SOTA]** Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.* ICLR, 2024.
- **[SOTA]** Gibbs, Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS, 2021.
- **[Related]** Ulmer, Zerva, Martins. *Non-Exchangeable Conformal Language Generation with Nearest Neighbors.* Findings of EACL, 2024.
- **[Related]** Ravfogel, Goldberg, Goldberger. *Conformal Nucleus Sampling.* Findings of ACL, 2023.
- **[Related]** Rubin-Toles et al. *Conformal Language Model Reasoning with Coherent Factuality.* ICLR, 2025.
- **[Survey]** Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* Foundations and Trends in Machine Learning, 2023.

## 10. Worked Example

Take $n = 1000$ calibration prompts, $\alpha = 0.1$. The quantile index is $\lceil 1001 \times 0.9\rceil = 901$: use the 901st smallest calibration score as $\hat q$. Marginal coverage is then provably in $[0.900, 0.901]$. That part is airtight.

Now split the same test set by a property nobody calibrated on — say, whether the prompt asks about an entity with a Wikipedia page. Suppose 700 test prompts are "head" entities and 300 are "tail". Marginal coverage of 0.90 is consistent with

$$0.7 \times 0.97 + 0.3 \times 0.74 = 0.679 + 0.222 = 0.901.$$

The guarantee holds exactly. Tail-entity users get 74% coverage — the users most in need of a hedge get the weakest one. No distribution-free fix exists (Barber et al. 2021); conditioning on the subgroup requires knowing it in advance and spending calibration data on it, at $n=300$ pushing the overshoot ceiling to $1/301 \approx 0.003$ and widening the variance of $\hat q$.

Layer on judge noise. If the LLM judge $\hat A$ agrees with human $A$ 85% of the time and its errors are correlated with claim obscurity — false-"true" on rare claims — then $\hat q$ is calibrated against an $A$ that is systematically lenient exactly where the model hallucinates. A certified 90% under $\hat A$ can sit near 80% under $A$, with no way to detect it from the calibration set.

That is the obstruction, visible in arithmetic: the theorem is exact, and the two quantities that decide whether it means anything — the conditioning subgroup and the admission predicate — sit outside it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*