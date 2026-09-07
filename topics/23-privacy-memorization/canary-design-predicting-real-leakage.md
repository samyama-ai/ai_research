---
id: 23-privacy-memorization/canary-design-predicting-real-leakage
title: "Canary Design That Predicts Real Data Leakage"
topic: 23-privacy-memorization
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Canary Design That Predicts Real Data Leakage

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/canary-design-predicting-real-leakage` · **Status:** methodologically-blocked

## 1. Problem Statement

A **canary** is an artificial record inserted into training data so that its later recoverability can be measured. Canaries are the standard instrument for privacy auditing: labs insert them, measure exposure or attack success, and report the result as evidence about the leakage of *real* user data. The problem is that nobody has established the transfer function from canary measurement to real-data risk.

Three variants, of very different difficulty:

- **Measurement variant (the blocked one).** Given a canary family $\mathcal{C}$ and a leakage statistic $L$, define and estimate the map $L(\mathcal{C}) \mapsto L(\mathcal{D}_{\text{real}})$, where $\mathcal{D}_{\text{real}}$ is the natural training corpus. Solving it means: a canary design plus an estimator such that the canary-derived prediction of real-data leakage is calibrated — neither systematically conservative nor optimistic — with a stated error bar, verified against direct measurement on real records.
- **Method variant.** Construct canaries maximally *predictive* rather than maximally *extractable*. Current practice optimizes the second (out-of-distribution, high-entropy strings that are the easiest possible targets), which is the wrong objective if the goal is estimating typical-case risk.
- **Theory variant.** Prove a bound relating canary extraction rate to extraction rate of a natural record with matched duplication count, perplexity under the pretrained prior, and context length. No such bound exists even under strong assumptions.

## 2. Formal Setting

Training corpus $\mathcal{D} = \{x_1,\dots,x_N\}$, tokens from vocabulary $V$. Model $\theta = \mathcal{A}(\mathcal{D} \cup \mathcal{C})$ from randomized algorithm $\mathcal{A}$ (SGD or DP-SGD). Canary set $\mathcal{C}$, $|\mathcal{C}| = m$, each canary $c = (p, s)$ with prefix $p$ and secret $s \in \mathcal{S}$ (the *canary space*, e.g. all 9-digit strings, $|\mathcal{S}| = 10^9$).

**Exposure** (Carlini et al., 2019), as actually computed: rank $s$ among all $s' \in \mathcal{S}$ by $-\log P_\theta(s' \mid p)$, take rank $r$, and

$$\text{exposure}(c) = \log_2 |\mathcal{S}| - \log_2 r .$$

For $|\mathcal{S}|$ too large to enumerate, $r$ is estimated by fitting a skew-normal to a sample of $10^5$–$10^6$ alternatives — an approximation that is *itself* untested in the tail, where the answer matters.

**Membership statistic** as measured in practice: a score $f(\theta, x)$ — loss, zlib-normalized loss, min-$k$% prob, or a LiRA likelihood ratio against $n$ shadow models — thresholded at a fixed false-positive rate $\alpha$. The reported number is $\text{TPR}@\alpha$, typically $\alpha \in \{10^{-3}, 10^{-2}\}$.

**Extraction rate** as measured: fraction of records $x$ with a $k$-token prefix ($k=50$ standard) for which greedy decoding reproduces the continuation verbatim.

**The predictive quantity.** For risk statistic $L$ and a matching function $\phi$ mapping real records to canary designs,

$$\text{Gap}(\phi) \;=\; \mathbb{E}_{x \sim \mathcal{D}_{\text{real}}}\big[L(x)\big] \;-\; \mathbb{E}_{c \sim \phi(\mathcal{D}_{\text{real}})}\big[L(c)\big].$$

A canary design is **calibrated** if $|\text{Gap}| \le \varepsilon_{\text{cal}}$ for a pre-declared $\varepsilon_{\text{cal}}$, and **conservative** if $\text{Gap} \le 0$ uniformly over the record distribution — not just on average, which is a strictly weaker and much less useful guarantee.

**Assumptions, and which are violated.**

1. *Canaries are exchangeable with real records under $\mathcal{A}$.* Violated: canaries are near-uniform in $\mathcal{S}$ and therefore maximal-loss under the pretrained prior, so they receive larger gradients early in training than typical text.
2. *Insertion does not perturb the measurand.* Violated at high $m$: inserting thousands of canaries changes the loss landscape and the deduplication statistics of the corpus.
3. *Duplication count is known.* Violated: near-duplicate counts in web corpora are estimated by MinHash at a similarity threshold, and the true count for a given real record is unknown to within an order of magnitude.
4. *One statistic suffices.* Violated: exposure, TPR@low-FPR, and verbatim extraction rank canary designs differently.
5. *DP accounting bounds the worst case.* Holds for the released $\varepsilon$, but the released $\varepsilon$ for large-model training is often $\ge 10$ or absent, so the bound is non-binding.

## 3. State of the Art

**Established.**

- *Secret Sharer* (Carlini, Liu, Erlingsson, Kos, Song, USENIX Security 2019): exposure as a metric, extraction by shortest-path search over $\mathcal{S}$. Demonstrated on character LSTMs and a Google production email model.
- *Quantifying Memorization* (Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang, ICLR 2023): extraction from real (not canary) data scales log-linearly in model size, duplication count, and prefix length. This is the strongest existing statement about real-data leakage, and it is measured *without* canaries.
- *Auditing with one training run* (Steinke, Nasr, Jagielski, NeurIPS 2023): $O(1)$-run auditing via many independent canary inclusion coins, with valid confidence intervals. This fixed the compute cost of auditing, not the design question.
- *Evaluations of Machine Learning Privacy Defenses Are Misleading* (Aerni, Zhang, Tramèr, ACM CCS 2024): average-case evaluation systematically understates risk; hand-crafted worst-case canaries and, separately, naturally atypical real examples both leak far more than the mean. This is the clearest published evidence that the canary-to-real gap is large and signed in a design-dependent direction.

**Claimed but unablated.** That inserting $k$ high-entropy canaries yields a *conservative* (upper-bounding) estimate of real leakage. This is folklore repeated in model cards and audit reports; no paper establishes it. The counter-evidence runs both ways: canaries are more extractable than typical text (optimistic direction violated) but *less* extractable than a duplicated, structurally predictable real secret such as a repeated API key in a code corpus.

**Benchmark-number-only.** Copyright-trap results (Meeus, Shilov, Faysse, de Montjoye, ICML 2024) report detection AUC for injected sequences in a 1.3B-parameter LLM trained from scratch; the numbers are real but characterize trap detectability, not the leakage of the surrounding natural corpus.

## 4. What Is Known

- **Duplication dominates.** Kandpal, Wallace, Raffel (ICML 2022): sequences duplicated ~10× in the corpus are emitted roughly $10^3$× more often than singletons at 1.5B scale. Lee et al. (ACL 2022) show deduplication cuts memorized emission by about 10×.
- **Extraction is real and cheap.** Carlini et al. (USENIX Security 2021) extracted hundreds of verbatim training sequences from GPT-2 1.5B. Nasr et al. (2023) recovered several megabytes of training text from production models including ChatGPT for roughly $200 of queries.
- **Scale.** Carlini et al. (ICLR 2023): for a 6B-parameter GPT-Neo, on the order of 1% of the training corpus is extractable with a 50-token prefix; the fraction rises log-linearly with model size across 125M–6B.
- **Auditing gap.** Empirical $\varepsilon$ lower bounds from canary auditing typically land several-fold below the analytic $\varepsilon$ (e.g. Jagielski, Ullman, Oprea, NeurIPS 2020; Nasr et al., IEEE S&P 2021; Steinke et al., NeurIPS 2023). The gap is design-dependent, which is exactly the problem.
- **MIAs are weak on LLM pretraining.** Duan et al. (COLM 2024): membership inference on pretrained LLMs performs near chance (AUC ≈ 0.5–0.6) once member/non-member sets are properly distribution-matched — so the statistic used to convert canary signal into risk is itself fragile.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted definition of "the real-data leakage a canary is supposed to predict." Extraction rate, TPR@$10^{-3}$, and exposure are not monotone transformations of one another, and each depends on an arbitrary choice (prefix length, FPR operating point, canary space size). Until $L$ is fixed and its estimator's variance characterized, "calibrated canary" has no truth value.
- **Theoretically open.** No bound of the form $L(x_{\text{real}}) \le g\big(L(c)\big)$ for canaries matched on duplication and perplexity, under any non-vacuous assumption on $\mathcal{A}$. DP gives a worst-case bound over all records but says nothing about the canary-to-typical-record ratio.
- **Empirically open.** No study inserts a *graded family* of canaries spanning the perplexity × duplication × length grid into a single pretraining run at $\ge$ 1B parameters and regresses real-record leakage on canary leakage. This is runnable today; it has not been run.

## 6. Why It Is Hard

**Absent ground truth, compounded by confounded measurement.** To validate a canary you must know the true leakage of real records — but the real-record label ("was this memorized because it was in training, or because it is predictable?") is precisely what memorization measurement cannot supply. Verbatim emission of `MIT License\n\nCopyright (c)` is not leakage; verbatim emission of a phone number may be. Every candidate ground truth is either (a) another canary, which is circular, or (b) an MIA whose own reliability on pretraining data is near chance (Duan et al., 2024).

**Non-identifiability.** Canary extractability varies along at least four axes — token entropy, duplication count, prefix distinctiveness, position in the training order — that co-vary in real data and are set independently in canary design. A single scalar exposure cannot identify which axis drove the number, so it cannot be transported to a real record that differs on all four.

**Sign of the bias is not fixed.** High-entropy canaries over-estimate leakage relative to typical prose and under-estimate it relative to duplicated structured secrets. So the common defense "canaries are conservative" is not merely unproven, it is false in at least one regime of practical interest.

## 7. Current Research (as of 2026)

- **Auditing under one run** — Steinke/Nasr/Jagielski line at Google DeepMind, extended to LLM fine-tuning: canaries in gradient space and in input space, with the finding that input-space canaries badly under-audit relative to gradient-space ones. *(frontier — verify the LLM-scale numbers.)*
- **Data watermarks / copyright traps** — Wei, Wang, Jia (ACL 2024); Meeus et al. (ICML 2024). Both design injected sequences for hypothesis-testable detection at controlled false-positive rate. The statistical machinery is directly reusable for the calibration question; nobody has repointed it there.
- **Worst-case evaluation** — Tramèr group (ETH Zürich), following the CCS 2024 result, pushing toward canaries drawn from the *tail of the real distribution* rather than from a synthetic space. This is the most promising route to closing the gap.
- **Unlearning audits** reuse canaries as verification targets, inheriting the same unvalidated transfer assumption. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does canary exposure predict real-record extraction rate, and with what error?

**Scale.** Pretrain a 1.4B-parameter decoder on a 40B-token deduplicated corpus (single run, ~2k A100-hours). Before training, construct a $4\times4$ grid: canary perplexity under a held-out 160M reference model $\in \{5, 20, 100, 500\}$ nats/token-band, duplication count $\in \{1, 4, 16, 64\}$; 64 canaries per cell, 1024 total, each 64 tokens.

**Control arm.** From the *same* corpus, select 1024 real 64-token records matched cell-by-cell on measured reference perplexity and MinHash-derived duplication count. These are not inserted — they are already in the data. Their leakage is measured the same way as the canaries'.

**Measurement.** For every canary and matched real record, compute (i) verbatim greedy extraction at 32-token prefix and (ii) exposure with $|\mathcal{S}|$ fixed by the cell.

**The deciding number.** Per-cell ratio $\rho = \Pr[\text{extract} \mid \text{canary}] / \Pr[\text{extract} \mid \text{matched real}]$. Report $\max_{\text{cell}} \rho / \min_{\text{cell}} \rho$ — the spread of the transfer factor across the grid. If that spread is $< 2$, canaries are usable as a calibrated proxy after a single multiplicative correction, and the problem is largely solved. If it exceeds 10 — the outcome we expect — no scalar transfer function exists, and every audit reporting a single canary number is reporting an uninterpretable quantity. Secondary check: the sign of $\log \rho$ per cell tells whether "canaries are conservative" holds anywhere.

## 9. Key References

- **[Foundational]** Carlini, Liu, Erlingsson, Kos, Song. *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks.* USENIX Security, 2019. — arXiv:1802.08232
- **[Foundational]** Carlini, Tramèr, Wallace, Jagielski, Herbert-Voss, Lee, Roberts, Brown, Song, Erlingsson, Oprea, Raffel. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[SOTA]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Steinke, Nasr, Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[SOTA]** Aerni, Zhang, Tramèr. *Evaluations of Machine Learning Privacy Defenses Are Misleading.* ACM CCS, 2024. — arXiv:2404.17399
- **[Related]** Jagielski, Ullman, Oprea. *Auditing Differentially Private Machine Learning: How Private is Private SGD?* NeurIPS, 2020. — arXiv:2006.07709
- **[Related]** Nasr, Songi, Thakurta, Papernot, Carlini. *Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning.* IEEE S&P, 2021. — arXiv:2101.04535
- **[Related]** Carlini, Chien, Nasr, Song, Terzis, Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[Related]** Kandpal, Wallace, Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- **[Related]** Duan, Suri, Mireshghallah, Min, Shi, Zettlemoyer, Tsvetkov, Choi, Evans, Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Related]** Meeus, Shilov, Faysse, de Montjoye. *Copyright Traps for Large Language Models.* ICML, 2024.
- **[Related]** Wei, Wang, Jia. *Proving Membership in LLM Pretraining Data with Data Watermarks.* ACL, 2024.
- **[Related]** Nasr, Carlini, Hayase, Jagielski, Cooper, Ippolito, Choquette-Choo, Wallace, Tramèr, Lee. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035

## 10. Worked Example

Insert one canary into a 1B-token fine-tuning corpus: `"The access code is 4831-9920-7715"`, repeated 8 times.

*Canary side.* $|\mathcal{S}| = 10^{12}$, so $\log_2|\mathcal{S}| \approx 39.9$ bits. After training, the canary secret ranks $r = 3$ among sampled alternatives. Exposure $= 39.9 - \log_2 3 \approx 38.3$ bits — near-maximal. The audit reports "high memorization."

*Real side, same run.* Take a real record from the corpus: an internal email footer containing an employee's direct line, appearing 8 times, reference-model perplexity ~12 nats/token — six times more predictable than the canary's near-uniform digits. Greedy decoding from a 32-token prefix reproduces the footer boilerplate but emits a *different* plausible phone number. Extraction: 0. The audit's headline number says maximal leakage; the record it was meant to stand in for did not leak at all.

*Now flip it.* A second real record — a hardcoded AWS key in a vendored config file, present 41 times across near-duplicate forks that MinHash counted as 3 — is emitted verbatim from a 16-token prefix. Extraction: 1. The canary, at 8 duplications, gave no warning that a 41-duplicate secret would go.

The two errors have opposite signs in the same training run. Exposure 38.3 bits was simultaneously an over-estimate for the phone number and an under-estimate for the key, and nothing in the canary measurement distinguishes the cases. That is the obstruction: the canary reports a number, the number is real, and it is not an estimate of anything about the corpus.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*