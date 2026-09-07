---
id: 23-privacy-memorization/canary-auditing-external-validity
title: "Canary-Based Auditing Validity for Natural Data"
topic: 23-privacy-memorization
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Canary-Based Auditing Validity for Natural Data

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/canary-auditing-external-validity` · **Status:** methodologically-blocked

## 1. Problem Statement

Privacy audits of language models are almost always run with **canaries**: synthetic records (random digit strings, fabricated names, templated secrets) inserted into training data at a controlled rate, then probed after training. The audit reports an exposure, an extraction rate, or an empirical $\varepsilon$ lower bound. The claim the reader takes away is about *real* training records — a patient note, a leaked API key, a copyrighted paragraph.

The problem: **there is no established transfer function from canary leakage to natural-record leakage.**

Three variants, differing in difficulty:

- **Measurement variant.** Define a quantity $\Delta$ that relates canary risk to natural-record risk, and specify how to estimate it without knowing which natural records are sensitive. Currently not well defined — this is why the page is *methodologically blocked*.
- **Method variant.** Design canaries whose leakage is provably conservative (an upper bound) or provably calibrated for a named natural subpopulation. Partially attacked, unsolved.
- **Theory variant.** Prove conditions on the data distribution and training procedure under which canary-derived $\varepsilon_{\text{emp}}$ lower-bounds the leakage of natural records. Open.

A solution is a procedure that, given a model and a canary audit, outputs a number about natural data with stated coverage — and is validated against a ground truth that does not itself come from canaries.

## 2. Formal Setting

Training set $D = \{x_1, \dots, x_n\}$ drawn i.i.d. from $\mathcal{P}$; training algorithm $\mathcal{A}$ producing $\theta = \mathcal{A}(D)$. Canary set $C$ inserted at multiplicity $m$ (number of copies).

**Per-record leakage.** For record $x$, the auditable quantity is the membership advantage of the best available attack $\mathcal{T}$ at a fixed low false-positive rate $\alpha$:

$$\mathrm{TPR}_\alpha(x) = \Pr_{\theta \sim \mathcal{A}(D \cup \{x\})}[\mathcal{T}(\theta, x) = \text{in}] \quad \text{s.t.} \quad \Pr_{\theta \sim \mathcal{A}(D)}[\mathcal{T}(\theta, x) = \text{in}] \le \alpha .$$

*As measured:* $\mathcal{T}$ is LiRA (Carlini et al., S&P 2022) or a loss-ratio variant; the two probabilities are estimated from $N$ shadow models with $x$ in/out, so the estimator has resolution $\approx 1/N$ at the tail. With $N = 256$ shadow models you cannot resolve $\alpha = 10^{-3}$ per record without pooling.

**Exposure** (Carlini et al., USENIX Sec. 2019), for a canary drawn from a known format space $\mathcal{R}$ of size $|\mathcal{R}|$:

$$\mathrm{expo}(c) = \log_2 |\mathcal{R}| - \log_2 \mathrm{rank}_\theta(c),$$

with $\mathrm{rank}$ the position of $c$ when all of $\mathcal{R}$ is sorted by model perplexity. *As measured:* computable exactly only because $\mathcal{R}$ is enumerable or log-perplexity is approximately Gaussian over $\mathcal{R}$. **No such $\mathcal{R}$ exists for a natural record** — this is the core measurement gap.

**The transfer quantity.** For a natural subpopulation $S \subset \mathrm{supp}(\mathcal{P})$,

$$\Delta_\alpha(S) = \frac{\mathbb{E}_{c \sim C}[\mathrm{TPR}_\alpha(c)]}{\mathbb{E}_{x \sim S}[\mathrm{TPR}_\alpha(x)]} .$$

The audit is *conservative* for $S$ iff $\Delta_\alpha(S) \ge 1$. Auditing practice implicitly assumes $\Delta_\alpha(S) \ge 1$ for every $S$ anyone cares about, and reports no estimate of $\Delta$.

**Assumptions, and which are violated:**

1. *Canaries are worst-case.* Violated in both directions: canaries are out-of-distribution (high loss → easy to detect, inflating $\Delta$) but also unduplicated and unstructured (no paraphrase support, deflating $\Delta$ for records with near-duplicates).
2. *Insertion does not perturb the rest of training.* Approximately holds at $|C|/n \le 10^{-4}$; unverified at the multiplicities $m = 10$–$1000$ used to get signal.
3. *Records are independent.* Violated — natural corpora contain near-duplicates, so removing $x$ leaves the information in $D$ (Kandpal et al., ICML 2022).
4. *Shadow-model retraining is affordable.* False above $\sim$1B parameters; audits there use one-run methods (Steinke et al., NeurIPS 2023) with a looser bound.

## 3. State of the Art

**Established.**
- Exposure-based auditing (Carlini et al., *The Secret Sharer*, USENIX Security 2019) — reproduced widely; detects unintended memorization of inserted secrets and is the basis of most deployed audits.
- LiRA (Carlini et al., S&P 2022) — per-example likelihood-ratio MIA, the reference attack at low FPR; independently reproduced.
- One-run auditing (Steinke, Nasr, Jagielski, NeurIPS 2023) — $\varepsilon$ lower bounds from a single training run via many independent canaries, with a valid finite-sample bound. The bound is a *theorem*, and it is about the algorithm, not about any natural record.
- Tight auditing under a strong adversary (Nasr et al., USENIX Security 2023) — canary gradients chosen adversarially recover nearly the theoretical $\varepsilon$ in the white-box, worst-case-dataset setting.

**Claimed but unablated.**
- That canary leakage upper-bounds natural leakage. Stated or implied in most deployment reports; no paper presents an ablation over canary design measuring $\Delta_\alpha(S)$ for natural $S$.
- That "realistic" canaries (name–SSN templates, phished secrets à la Panda et al., ICLR 2024) close the gap. Panda et al. show a stronger *attack*, not a calibration.

**Benchmark-number only.**
- Every LLM MIA leaderboard result (WikiMIA and successors). Das, Zhang, Tramèr (2024) show blind baselines using publication date beat these attacks, so the numbers measure distribution shift between member/non-member splits, not membership. Meeus et al. (SaTML 2025) reach the same conclusion for the benchmark family as a whole.

## 4. What Is Known

- **Memorization scales log-linearly** in model size, duplication count, and prompt length (Carlini et al., ICLR 2023, *Quantifying Memorization Across Neural Language Models*), measured on GPT-Neo 125M–6B over The Pile: the 6B model emits verbatim continuations for roughly an order of magnitude more sequences than the 125M model under the same 50-token-prefix protocol.
- **Duplication dominates.** Deduplicating training data cuts regurgitation of memorized sequences by roughly $10\times$ (Kandpal, Wallace, Raffel, ICML 2022, on Pile-trained models up to 1.5B). Canaries inserted at $m=1$ therefore sit at the low end of a curve most natural sensitive text does not sit on.
- **Average-case evaluation understates worst-case leakage.** Aerni, Zhang, Tramèr (CCS 2024) show that privacy-defense evaluations reporting average-case MIA accuracy hide vulnerable subpopulations; measuring on deliberately vulnerable examples raises TPR at low FPR by more than an order of magnitude on CIFAR-10-scale image models. This is direct evidence that $\Delta_\alpha(S)$ varies by orders of magnitude across $S$.
- **Auditing bounds are loose in the realistic regime.** Steinke et al. (NeurIPS 2023) report an empirical lower bound of $\varepsilon_{\text{emp}} \approx 1.3$ for a CIFAR-10 model trained to theoretical $\varepsilon = 4.0$ — a factor-of-3 gap in the black-box setting, with the gap closing only under white-box, worst-case-dataset assumptions (Nasr et al., 2023).
- **Standard MIAs fail on LLMs.** Duan et al. (COLM 2024) report AUC near chance ($\approx 0.5$–$0.6$) across Pythia models 160M–12B on Pile members vs. non-members, attributed to single-epoch training and near-identical member/non-member distributions.
- **Real extraction happens anyway.** Nasr et al. (2023) extracted thousands of unique memorized training strings from a production chat model with a divergence attack — evidence that low measured MIA AUC does not imply low natural-record risk.

## 5. What Is Not Known

- **Methodologically blocked:** the definition of the natural-record counterpart to exposure. Exposure requires an enumerable format space $\mathcal{R}$; a sentence has no such space. Counterfactual memorization (Zhang et al., NeurIPS 2023) is the closest well-posed substitute, but it costs $O(N)$ retrainings and has only been computed at sub-billion scale. Until a natural-record risk score is defined that is (a) computable without leave-one-out retraining and (b) validated, $\Delta_\alpha(S)$ cannot be estimated at all.
- **Theoretically open:** whether any distribution-free condition on $(\mathcal{P}, \mathcal{A})$ implies $\Delta_\alpha(S) \ge 1$. No proof either way. Under DP the algorithm-level bound holds for all records, but the *empirical* canary bound is a lower bound on $\varepsilon$, and lower bounds do not transfer across records.
- **Empirically open:** the sign and magnitude of $\Delta_\alpha(S)$ at 1B+ scale for named subpopulations (rare names, medical identifiers, low-frequency code secrets). Runnable today with shadow models at 1B; nobody has published it.
- **Empirically open:** whether canary multiplicity $m$ can be calibrated so that canary leakage matches the leakage of natural records at the same duplication count.

## 6. Why It Is Hard

**Absent ground truth, compounded by non-identifiability.** To validate $\Delta$ you need per-record natural leakage $\mathrm{TPR}_\alpha(x)$, whose gold-standard estimator is leave-one-out shadow training: $N \ge 128$ models per record for a usable tail estimate. At 1B parameters and 20B tokens, one model is $\sim$$10^{21}$ FLOP; 128 of them is $\sim$$10^{23}$ FLOP — for *one* record's confidence interval, before averaging over a subpopulation.

The cheap substitute — reference-model MIA on public member/non-member splits — is confounded: temporal or topical shift between the splits produces attack signal with no membership content, which is exactly what blind baselines expose (Das et al., 2024). So the affordable measurement does not measure the thing it names, and the measurement that does is not affordable. That is the block, not the importance.

Second obstruction: **near-duplicates break the counterfactual.** If $x$ has a paraphrase in $D$, removing $x$ barely changes $\theta$, so $\mathrm{TPR}_\alpha(x)$ is small even though the *content* of $x$ is fully extractable. Canaries have no paraphrases by construction. The two settings differ in the quantity being defined, not just in its value.

## 7. Current Research (as of 2026)

- **One-run and few-run auditing.** Steinke/Nasr/Jagielski line at Google DeepMind, extending to LLM fine-tuning; work on canary designs with tighter per-run bounds.
- **Benchmark hygiene for LLM MIA.** Tramèr's group (ETH Zurich) on blind baselines and misleading defense evaluations; Meeus/de Montjoye (Imperial) on randomized-control MIA benchmark construction (SaTML 2025). Direction: replace observational member/non-member splits with injected randomized controls — which reintroduces the canary-validity question one level up. *(frontier — verify)*
- **Realistic canaries.** "Phishing" canaries (Panda et al., ICLR 2024) and structured PII templates; open question whether these are calibrated or merely stronger.
- **Counterfactual and influence-based memorization at scale.** Attempts to approximate leave-one-out with influence functions or training-run checkpointing rather than retraining. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** is $\Delta_\alpha(S) \ge 1$ for natural sensitive text?

- **Scale.** Pythia-1.4B architecture, trained from scratch on a 20B-token Pile subset. $N = 128$ shadow models with independently randomized in/out assignment over a held-out pool of 2,000 *natural* records (the "targets"), drawn to span duplication counts $m \in \{1, 2, 4, \dots, 64\}$ measured by exact 50-gram match. Budget $\approx 1.3 \times 10^{23}$ FLOP; feasible on a 512-H100 cluster in roughly two weeks.
- **Treatment arm.** In the same runs, insert 2,000 standard canaries (random 10-digit secrets in a fixed template) matched one-to-one to the targets' duplication counts.
- **Control arm.** A duplication-matched set of 2,000 natural records drawn from the *same* documents as the targets but never inserted/removed (always-out), to calibrate the FPR curve and to expose any split-level distribution shift; a blind baseline (no model access, features from length/date/topic only) must score AUC $\le 0.52$ on the target set or the experiment is void.
- **Deciding number.** The per-duplication-stratum ratio $\hat{\Delta}_{0.001}(m) = \mathrm{TPR}_{0.001}(\text{canaries at } m) / \mathrm{TPR}_{0.001}(\text{natural at } m)$ with a bootstrap 95% CI. **If the lower CI bound on $\min_m \hat\Delta_{0.001}(m)$ exceeds 1, canary audits are conservative in this regime; if the upper bound falls below 1 for any $m$, every deployed canary audit understates natural-record risk by that factor.** With 2,000 records pooled per stratum, $\alpha = 10^{-3}$ is resolvable.

## 9. Key References

- **[Foundational]** N. Carlini, C. Liu, Ú. Erlingsson, J. Kos, D. Song. *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks.* USENIX Security, 2019. — arXiv:1802.08232
- **[Foundational]** R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017. — arXiv:1610.05820
- **[SOTA]** N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[SOTA]** T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[SOTA]** M. Nasr, J. Hayes, T. Steinke, B. Balle, F. Tramèr, M. Jagielski, N. Carlini, A. Terzis. *Tight Auditing of Differentially Private Machine Learning.* USENIX Security, 2023. — arXiv:2302.07956
- **[Key evidence]** M. Aerni, J. Zhang, F. Tramèr. *Evaluations of Machine Learning Privacy Defenses are Misleading.* ACM CCS, 2024. — arXiv:2404.17399
- **[Key evidence]** M. Duan, A. Suri, N. Mireshghallah, S. Min, W. Shi, L. Zettlemoyer, Y. Tsvetkov, Y. Choi, D. Evans, H. Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Key evidence]** D. Das, J. Zhang, F. Tramèr. *Blind Baselines Beat Membership Inference Attacks for Foundation Models.* 2024. — arXiv:2406.16201
- **[Scaling]** N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramèr, C. Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Scaling]** N. Kandpal, E. Wallace, C. Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- **[Definition]** C. Zhang, D. Ippolito, K. Lee, M. Jagielski, F. Tramèr, N. Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023. — arXiv:2112.12938
- **[Attack]** A. Panda, C. A. Choquette-Choo, Z. Zhang, Y. Yang, P. Mittal. *Teach LLMs to Phish: Stealing Private Information from Language Models.* ICLR, 2024. — arXiv:2403.00871
- **[Survey]** M. Meeus, I. Shilov, S. Jain, M. Faysse, M. Rei, Y.-A. de Montjoye. *SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It).* IEEE SaTML, 2025. — arXiv:2406.17975
- **[Extraction]** M. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A. Choquette-Choo, E. Wallace, F. Tramèr, K. Lee. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035

## 10. Worked Example

A hospital fine-tunes a 7B model on 500k clinical notes and audits with 100 canaries of the form `Patient ID: <9 random digits>`, each inserted once.

**Canary side.** $|\mathcal{R}| = 10^9$, so $\log_2|\mathcal{R}| = 29.9$ bits. Suppose the median canary ranks 40,000th out of $10^9$ by perplexity:

$$\mathrm{expo} = 29.9 - \log_2(4\times10^4) = 29.9 - 15.3 = 14.6 \text{ bits}.$$

Extraction requires $\mathrm{expo} \approx \log_2|\mathcal{R}| = 29.9$. The audit reports "no canary extractable" and the deployment proceeds.

**Natural side.** Take the sentence *"Mrs. Adaeze Okonkwo-Bhattacharya was admitted for aplastic anemia on 14 March."* Three properties break the transfer:

1. **No format space.** $\mathcal{R}$ is undefined, so $\mathrm{expo}$ is not computable. There is no number to compare to 14.6.
2. **Duplication.** The name appears in 9 notes for the same patient; the canary appeared once. On the log-linear duplication curve from Carlini et al. (ICLR 2023), a $9\times$ multiplicity shift moves memorization by roughly an order of magnitude — the audit measured a point the target does not occupy.
3. **Non-identifiability of the counterfactual.** Because 9 near-duplicate notes exist, deleting any one changes $\theta$ negligibly, so $\mathrm{TPR}_\alpha(x)$ — the quantity the audit is supposed to bound — is *small* while the name is still emitted verbatim under the prompt `"Mrs. Adaeze"`. Membership risk and extraction risk point opposite ways on the same record.

**The obstruction, visible.** The audit produced 14.6 bits about a record type that does not exist in the corpus, and the deciding statistic for the record type that does exist is either undefined (exposure) or misleadingly near zero (membership advantage). Estimating $\Delta_{0.001}$ here needs 128 retrainings of a 7B model per stratum — about $10^{24}$ FLOP — which is why the number in Section 8 has never been reported.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*