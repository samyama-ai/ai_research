---
id: 21-factuality/conformal-factuality-long-form-generation
title: "Conformal Factuality for Free-Form Long Text"
topic: 21-factuality
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Conformal Factuality for Free-Form Long Text

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/conformal-factuality-long-form-generation` · **Status:** partially-solved

## 1. Problem Statement

Given a language model $M$ and a user prompt $x$, produce a long-form answer $\hat y$ together with a **distribution-free finite-sample guarantee** that $\hat y$ contains no false assertion, at a user-chosen error level $\alpha$ — while keeping $\hat y$ informative.

Three variants, with sharply different difficulty:

- **Measurement.** Define, for a paragraph of free text, a claim-level truth predicate that a calibration set can actually score. This is the binding constraint. A paragraph is not a label.
- **Method.** Given such a predicate and an exchangeable calibration set, output a filtered/hedged $\hat y$ with $\Pr[\hat y \text{ contains a false claim}] \le \alpha$. Largely **solved** by conformal risk control over a nested family of back-offs.
- **Theory.** Obtain the guarantee (i) conditionally on the prompt or on prompt subgroups, (ii) under the distribution shift between calibration prompts and deployment prompts, and (iii) with the guarantee attaching to the *meaning of the whole text* rather than to a bag of extracted claims. Open.

Solving it means: a deployed system where a user sets $\alpha = 0.05$, receives text, and the empirical rate of answers containing at least one false claim is $\le 5\%$ on *their* prompt stream — with retained content measurably above a trivial abstain/hedge baseline.

## 2. Formal Setting

Let $(X_i, Y_i)_{i=1}^n$ be calibration prompts with model outputs $Y_i \sim M(\cdot\mid X_i)$, and $(X_{n+1}, Y_{n+1})$ a test pair.

**Decomposition.** A decomposer $D$ maps text to subclaims, $D(y) = \{c_1,\dots,c_{m}\}$. *Measured as:* an LLM prompted to emit atomic, context-independent statements; $m \approx 25$–$60$ for a 300-word biography under the FActScore protocol (Min et al., EMNLP 2023).

**Truth predicate.** $T(c) \in \{0,1\}$, measured by a retrieval-grounded judge (FActScore's Wikipedia-conditioned scorer; SAFE's Google-Search agent, Wei et al. 2024) or by human annotation. $T$ is a noisy proxy, not ground truth.

**Scoring and back-off.** Each claim gets a confidence $s(c) \in \mathbb{R}$ (self-consistency frequency across $k$ samples, verbalized probability, or mean token logprob). The filtered output is
$$\hat y_\lambda = D^{-1}\big(\{c \in D(y) : s(c) \ge \lambda\}\big),$$
re-rendered as prose. The family is **nested**: $\lambda \le \lambda' \Rightarrow \hat y_{\lambda'} \subseteq \hat y_\lambda$, with $\hat y_{\lambda_{\max}}$ the empty/maximally hedged answer.

**Loss.** $L(x,y,\lambda) = \mathbb{1}\big[\exists c \in D(\hat y_\lambda): T(c)=0\big]$, monotone non-increasing in $\lambda$ and equal to $0$ at $\lambda_{\max}$.

**Guarantee.** Conformal risk control (Angelopoulos et al., ICLR 2024) picks
$$\hat\lambda = \inf\Big\{\lambda : \tfrac{1}{n+1}\Big(\sum_{i=1}^n L(X_i,Y_i,\lambda) + 1\Big) \le \alpha\Big\},\qquad \mathbb{E}\big[L(X_{n+1},Y_{n+1},\hat\lambda)\big] \le \alpha.$$

**Utility.** Retention $U(\lambda) = |D(\hat y_\lambda)| / |D(y)|$, or a human preference win-rate against the unfiltered answer. Reporting $\alpha$ without $U$ is meaningless: $U \equiv 0$ satisfies every $\alpha$.

**Assumptions, and their status in practice.**
1. *Exchangeability of $(X_i,Y_i)$ with the test point.* Violated — calibration sets are FActScore biographies or MATH; deployment prompts are not drawn from that distribution. Barber et al. (Ann. Statist. 2023) bound the coverage gap by the total-variation distance, which is unestimated here.
2. *$T$ is the truth.* Violated — SAFE agrees with human annotators on ~72% of individual facts. Calibration is against the judge, so the guarantee is "no claim the judge would reject."
3. *Claim independence / decomposition faithfulness.* Violated — $D$ drops presuppositions and discourse relations, so $\bigwedge_i T(c_i) = 1$ does not imply $\hat y$ is true (Rubin-Toles et al., ICLR 2025).
4. *Model fixed during calibration.* Violated by any post-calibration update to $M$, $D$, or the retrieval index.

## 3. State of the Art

**Established (with ablations).**
- **Conformal factuality** (Mohri & Hashimoto, ICML 2024): back-off over subclaim removal under split conformal / CRC gives a marginal high-probability guarantee on FActScore biographies and MATH. Established: the guarantee holds empirically on held-out data from the same pool; better confidence scores (self-consistency frequency) dominate logprob-based scores in retention at fixed $\alpha$.
- **Conditional boosting** (Cherian, Gibbs & Candès, NeurIPS 2024): learns a score by boosting to maximize retained claims subject to conditional validity via the Gibbs–Candès conditional-conformal framework. Established: strictly more claims retained than Mohri–Hashimoto at matched $\alpha$ on the same biography benchmark, with approximate group-conditional coverage.
- **Conformal language modeling** (Quach et al., ICLR 2024): calibrated stopping rules for sampling sets, using Learn-then-Test (Angelopoulos, Bates, Candès, Jordan, Lei, JMLR 2025) to control the probability that a returned set contains an acceptable answer.
- **Coherent factuality** (Rubin-Toles, Bhatt, Cherian, et al., ICLR 2025): filters over a *deducibility graph* rather than a claim set, so retained subclaims remain jointly supported. Established on MATH-style reasoning that graph-aware filtering retains more content at equal coherent-factuality level than claim-bag filtering.

**Claimed but unablated.** That these procedures transfer to open-domain assistant traffic. Every published calibration set is a narrow pool (biographies, MATH, medical Q&A). No paper reports the guarantee measured on a prompt distribution disjoint from the calibration distribution.

**Benchmark-number-only.** Retention/utility figures are all judge-scored on 100–1,000 prompts; human verification of the filtered outputs is small-sample or absent.

## 4. What Is Known

- CRC's bound $\mathbb{E}[L] \le \alpha$ is a theorem for any bounded monotone loss and exchangeable data (Angelopoulos et al., ICLR 2024). It is *marginal in expectation*, not per-prompt.
- Exact conditional validity is impossible without distributional assumptions (Vovk 2012; Lei & Wasserman 2014; Barber, Candès, Ramdas, Tibshirani, *The limits of distribution-free conditional predictive inference*, Info. & Inference 2021) — nontrivial finite-length prompt-conditional guarantees do not exist distribution-free.
- Judge quality, measured: FActScore's retrieval+LM estimator has <2% error against human FActScore on ~500 annotated biographies; SAFE matches human labels on ~72% of ~16k facts from 496 LongFact responses, and was preferred over humans in 76% of 100 sampled disagreements, at ~20× lower cost (Wei et al., 2024).
- Base rates the guarantee must fight: FActScore on person biographies reports ~58% claim precision for ChatGPT and ~42% for InstructGPT (EMNLP 2023) — at $\alpha=0.05$ on a 40-claim answer, most claims must be dropped unless the score is near-perfectly ranked.
- Non-exchangeable conformal (Barber et al., Ann. Statist. 2023) gives coverage $\ge 1-\alpha - \sum_i w_i \,d_{\mathrm{TV}}$-style slack; the slack term has not been instantiated for LLM prompt shift.

## 5. What Is Not Known

- **Methodologically blocked.** A truth predicate for free text that is not a judge model. All guarantees are conditional on $T$; nobody has defined how to price judge error into $\alpha$ without a second, unavailable, ground truth. Likewise, no accepted utility metric makes hedging degeneracy detectable — "was born in the 20th century" is true and near-vacuous, and retention counts it as full credit.
- **Empirically open.** Whether the guarantee survives deployment shift. The experiment — calibrate on pool A, measure on pool B — is cheap and unrun at scale. Also unrun: whether users prefer $\alpha$-guaranteed filtered text over unfiltered text with inline uncertainty markers.
- **Theoretically open.** A guarantee attached to the entailment closure of the whole passage rather than to a claim set; sharp conditional-validity rates under bounded shift for text; and whether score-learning on calibration data (as in boosting) can be made valid without sample splitting.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth compounded by an evaluation that does not measure what it names**. The pipeline calibrates $\Pr[\text{judge rejects a retained claim}] \le \alpha$ and reports it as $\Pr[\text{output contains a falsehood}] \le \alpha$. With a judge at ~72% human agreement, the gap is larger than every $\alpha$ anyone targets. Second obstruction: **non-identifiability of the object being certified** — $D$ is model-dependent and lossy, so "the set of claims in $\hat y$" is not a well-defined function of the text, and the same paragraph decomposed by two decomposers yields different loss values. Third: **the hedging escape hatch** makes the guarantee free, so all difficulty migrates into a utility metric no one has fixed.

## 7. Current Research (as of 2026)

- Stanford (Hashimoto, Candès groups) — conditional and boosted conformal for LLMs, level-adaptive $\alpha$, coherent/graph-based filtering. Continuous line from Mohri–Hashimoto through Cherian et al. to Rubin-Toles et al.
- CMU / Berkeley (Ramdas, Angelopoulos, Bates) — risk control beyond exchangeability, e-value and betting-based sequential validity, which fits streaming deployment better than split conformal *(frontier — verify current LLM instantiations)*.
- Google DeepMind — long-form factuality evaluation (LongFact/SAFE) supplying the judges these methods calibrate against.
- Emerging: attaching conformal filtering to retrieval-augmented generation, where the retrieved evidence set makes $T$ auditable; and abstention-aware utility metrics that penalize vacuity *(frontier — verify)*.

## 8. Concrete Next Experiment

**The shift-transfer test.**

- **Scale.** Calibrate on $n=1{,}000$ FActScore-style biography prompts with SAFE-scored claims, using the Mohri–Hashimoto back-off with self-consistency scores ($k=8$ samples), targeting $\alpha=0.10$. Evaluate on 500 held-in biography prompts and 500 **out-of-pool** prompts drawn from real assistant traffic categories (product/technical/how-to/current-events), all scored by the same judge, plus 200 of the out-of-pool responses re-scored by 3 human annotators.
- **Control arms.** (a) In-distribution held-out evaluation — the number every paper reports; (b) unfiltered generation; (c) fixed-threshold filtering tuned to match arm (a)'s retention, with no conformal step.
- **Deciding number.** The **out-of-pool empirical error rate** $\hat e_{\mathrm{OOP}} = $ fraction of answers containing $\ge 1$ human-judged false retained claim. If $\hat e_{\mathrm{OOP}} \le 0.15$ (i.e. within $1.5\alpha$) at retention $\ge 0.5$, conformal factuality transfers and the field should move to conditional validity. If $\hat e_{\mathrm{OOP}} \ge 0.30$ — 3× the nominal level — the published guarantees are pool-specific artifacts and the open problem is shift, not scoring. Report $\hat e$ under the judge and under humans separately; their difference is the size of the measurement block in Section 6.

## 9. Key References

- **[Foundational]** Vladimir Vovk, Alexander Gammerman, Glenn Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[Foundational]** Rina Foygel Barber, Emmanuel Candès, Aaditya Ramdas, Ryan Tibshirani. *The limits of distribution-free conditional predictive inference.* Information and Inference, 2021.
- **[Foundational]** Rina Foygel Barber, Emmanuel Candès, Aaditya Ramdas, Ryan Tibshirani. *Conformal prediction beyond exchangeability.* Annals of Statistics, 2023.
- **[Foundational]** Anastasios Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster. *Conformal Risk Control.* ICLR, 2024. — arXiv:2208.02814
- **[Foundational]** Anastasios Angelopoulos, Stephen Bates, Emmanuel Candès, Michael Jordan, Lihua Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* JMLR, 2025.
- **[SOTA]** Christopher Mohri, Tatsunori Hashimoto. *Language Models with Conformal Factuality Guarantees.* ICML, 2024. — arXiv:2402.10978
- **[SOTA]** John Cherian, Isaac Gibbs, Emmanuel Candès. *Large language model validity via enhanced conformal prediction methods.* NeurIPS, 2024.
- **[SOTA]** Maxon Rubin-Toles, Maya Gambhir, Keshav Ramji, Aaron Roth, Surbhi Goel. *Conformal Language Model Reasoning with Coherent Factuality.* ICLR, 2025.
- **[SOTA]** Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi Jaakkola, Regina Barzilay. *Conformal Language Modeling.* ICLR, 2024.
- **[Measurement]** Sewon Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Measurement]** Jerry Wei et al. *Long-form factuality in large language models.* NeurIPS, 2024. — arXiv:2403.18802
- **[Survey]** Anastasios Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* Foundations and Trends in Machine Learning, 2023.

## 10. Worked Example

One biography prompt: *"Tell me a bio of Anna May Wong."* The model emits 40 atomic claims. Assume ChatGPT-level precision, ~58% (FActScore, EMNLP 2023): about 23 true, 17 false.

Target $\alpha = 0.05$ on the **answer-level** loss (no false claim retained). Suppose the self-consistency score is well calibrated but not perfectly ranked: retaining the top-$r$ claims keeps them all true with probability $p(r)$. To get $\Pr[\text{all retained true}] \ge 0.95$ with per-claim retained-truth probability $q$, we need $q^r \ge 0.95$, so
$$r \le \frac{\ln 0.95}{\ln q}.$$
At $q = 0.99$ (the score's top decile), $r \le 5$. At $q = 0.97$, $r \le 1.7$, i.e. **one claim**. From 40 claims the certified answer is "Anna May Wong was an American actress" — 2.5–12% retention.

Now the obstruction. The calibration used SAFE, which agrees with humans on ~72% of facts. A retained claim the judge calls true is human-true with probability well below $0.99$ unless judge errors are strongly anti-correlated with high confidence scores — which is unmeasured. If the judge's false-accept rate on high-confidence claims is even 3%, then $q_{\text{human}} \le 0.97$ regardless of the model, and the $\alpha = 0.05$ answer-level guarantee is unreachable for $r > 1$ **no matter how good the generator gets**. The bound is set by the measuring instrument, not the model. That is why the page is *partially-solved*: the theorem is done, the thermometer is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*