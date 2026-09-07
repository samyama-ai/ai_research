---
id: 19-evaluation/annotator-disagreement-signal-versus-noise
title: "Inter-Annotator Disagreement as Signal Versus Noise"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Inter-Annotator Disagreement as Signal Versus Noise

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/annotator-disagreement-signal-versus-noise` · **Status:** open

## 1. Problem Statement

When $m$ annotators label the same item and do not agree, the disagreement has at least three sources: (a) the item genuinely admits more than one correct label (**signal** — irreducible label variation), (b) annotators differ systematically in how they read the guideline or in what their social position makes salient (**structured signal** — a per-annotator effect), (c) slips, satisficing, and interface noise (**noise**). The catalog problem: given a labeled corpus, **decompose observed disagreement into these components, and decide what the benchmark's target should be**.

Three variants that are usually conflated:

- **Measurement.** Input: an item–annotator label matrix. Output: an estimate of the per-item label distribution $p_i$ and of the annotator-noise fraction. Solved iff the estimator is consistent and its assumptions are checkable from the data.
- **Method.** Input: the same matrix. Output: a training and scoring rule (hard majority, soft label, per-annotator head, jury) that maximises the downstream decision quality. Solved iff a rule dominates majority-vote across tasks under ablation.
- **Theory.** Is the decomposition identifiable at all from cross-sectional annotation (each annotator labels each item once)? Solved iff there is a proof of identifiability under stated conditions, or an impossibility result.

The theory variant is the load-bearing one. Most method papers assume the measurement variant is solved.

## 2. Formal Setting

Items $x_1,\dots,x_n$, label set $\mathcal{Y}$ with $|\mathcal{Y}|=K$, annotator pool $\mathcal{A}$. Annotator $a$ on item $i$ emits $y_{ia}\in\mathcal{Y}$. Posit a latent item distribution $p_i\in\Delta^{K-1}$ and an annotator channel $C_a\in\mathbb{R}^{K\times K}$ with $C_a[k,l]=\Pr(\text{emit } l \mid \text{perceive } k)$, so

$$\Pr(y_{ia}=l) = \sum_{k} p_i[k]\, C_a[k,l].$$

**Measured quantities.**

- Empirical distribution $\hat p_i = \frac{1}{m_i}\sum_{a\in A_i}\mathbf{1}[y_{ia}=\cdot]$, where $A_i$ is the set that actually saw item $i$ and $m_i=|A_i|$.
- Observed agreement $A_o = \frac{1}{n}\sum_i \frac{1}{m_i(m_i-1)}\sum_{a\neq b}\mathbf{1}[y_{ia}=y_{ib}]$; Krippendorff's $\alpha = 1 - D_o/D_e$ with $D_e$ the disagreement under label-permuted pairing.
- Item entropy $H(\hat p_i) = -\sum_k \hat p_i[k]\log \hat p_i[k]$. This is **downward biased** by roughly $(K-1)/(2m_i)$ nats (Miller–Madow); at $m_i=5$, $K=3$ that is $0.20$ nats against a $\log 3 = 1.10$ ceiling — 18% of the range.
- Test–retest self-agreement $s_a = \Pr(y_{ia}^{(t_1)} = y_{ia}^{(t_2)})$, measurable only if items are re-shown to the same annotator after a washout.
- Model score under soft labels: $\mathrm{JSD}(q_\theta(\cdot\mid x_i)\,\|\,\hat p_i)$ or $\mathrm{KL}(\hat p_i \| q_\theta)$.

**Target decomposition.** With $\bar C = \mathbb{E}_a[C_a]$, total disagreement splits into item dispersion $\mathbb{E}_i[1-\|p_i\|_2^2]$, annotator-effect variance $\mathrm{Var}_a(C_a)$, and within-annotator instability $1-s_a$.

**Assumptions, and which fail.**

1. *Conditional independence of annotators given $p_i$.* Violated: guideline training, shared demographics, and anchoring on item order correlate errors (Reidsma & Carletta 2008).
2. *$C_a$ is item-independent.* Violated: annotator competence varies by subdomain; a hate-speech annotator's channel differs for in-group vs out-group targets (Sap et al. 2022).
3. *$m_i \ge 5$ suffices to estimate $p_i$.* Violated: SNLI-style 5-way validation cannot distinguish $p_i=(0.6,0.4,0)$ from $(0.8,0.2,0)$ at any useful confidence; the 95% CI on a proportion at $m=5$ is $\pm 0.43$.
4. *Annotators are exchangeable draws from the deployment population.* Violated: crowd pools are demographically skewed and non-random.

## 3. State of the Art

**Established.**
- *Latent-class annotation models.* Dawid & Skene (1979) EM over $(p_i, C_a)$; MACE (Hovy et al., NAACL 2013); the item-response model of Passonneau & Carpenter (TACL 2014). These recover annotator quality better than majority vote when the conditional-independence assumption holds, verified on synthetic and on small real corpora.
- *Human label variation is reproducible.* Pavlick & Kwiatkowski (TACL 2019) collected 50 annotations for each of ~500 NLI items and showed multi-modal label distributions that persist under re-collection — the dispersion is a property of the item, not the sample.
- *Distributional targets are learnable.* Peterson et al. (ICCV 2019) released CIFAR-10H (≈511,400 judgments, ~50 per image over 10,000 test images) and showed training against soft labels improves out-of-distribution robustness relative to hard labels.

**Claimed but unablated.**
- Per-annotator multi-task heads beating majority vote (Davani et al., TACL 2022) — gains reported on a handful of subjective corpora, not ablated against a matched-capacity ensemble or against simply collecting more annotations per item.
- Jury learning (Gordon et al., CHI 2022) — a mechanism for choosing *whose* labels count; the claim that jury composition changes downstream harm is argued, not measured against a deployment outcome.
- LLM-as-annotator "matches crowd agreement" claims — typically a benchmark number ($\kappa$ against a majority vote) with no test of whether the model reproduces the *distribution*.

**Benchmark-number-only results.** ChaosNLI (Nie, Zhou & Bansal, EMNLP 2020): 100 annotations each for 4,645 SNLI/MNLI/$\alpha$NLI items. Models near-ceiling on old accuracy still show large JSD/KL to the human distribution. That is a leaderboard fact; no ablation isolates *why* (calibration, tokenizer, or objective).

## 4. What Is Known

- **Disagreement is large and not eliminable by more training.** ChaosNLI: on the MNLI subset, the majority label from 100 annotators differs from the original gold label on ~20% of items; over 30% of items have entropy above 0.9 nats out of $\log 3 = 1.10$ (scale: 4,645 items, 100 annotations each).
- **Benchmark gold labels contain real errors distinct from ambiguity.** Northcutt, Athalye & Mueller (NeurIPS Datasets & Benchmarks 2021) estimate ~3.4% average label error across 10 major test sets, ~6% on ImageNet validation (scale: 10 datasets, crowd re-validation of algorithmically flagged candidates). Gema et al. (2024) hand-audited MMLU and found error rates as high as 57% in the Virology subset (scale: 3,000 re-annotated questions).
- **Calibration metrics break under disagreement.** Baan et al. (EMNLP 2022) show ECE against a single gold label is not interpretable when $H(p_i)>0$, and propose human-uncertainty-aware calibration; demonstrated on ChaosNLI.
- **Annotator identity predicts labels.** Sap et al. (NAACL 2022) show annotator attitudes (measured by a survey instrument) predict toxicity ratings of African-American English text at effect sizes comparable to the text features themselves (scale: ~600 annotators, ~15k ratings).
- **Guideline paradigm changes the answer.** Röttger et al. (NAACL 2022) contrast *prescriptive* (one right answer, enforce guideline) with *descriptive* (capture belief) annotation; the same items yield materially different agreement under each. Which paradigm a corpus used is usually undocumented.

## 5. What Is Not Known

- **Theoretically open.** Whether $(p_i, \{C_a\})$ is identifiable from cross-sectional data when $C_a$ may depend on item content. Kruskal-style identifiability results for Dawid–Skene require conditional independence and item-independent channels; with item-dependent channels, item ambiguity and annotator bias are **not separable** — no proof of identifiability and no impossibility theorem covering the realistic case.
- **Empirically open.** Nobody has run the test–retest arm at scale: re-show the *same* annotator the *same* items after a washout, on $\ge 2{,}000$ items with $\ge 50$ annotators, to measure $1-s_a$ directly and subtract within-annotator instability from the disagreement budget. Cost is a few tens of thousands of dollars — runnable, unrun.
- **Empirically open.** Whether soft-label training gains (CIFAR-10H) survive at LLM scale and on text, matched for annotation budget: 50 annotations on 1,000 items versus 5 annotations on 10,000 items.
- **Methodologically blocked.** There is no agreed target for "the right label distribution." Distribution over *whom*? A demographic-balanced population, the deployment user base, or expert consensus give different $p_i$, and no measurement selects among them. This blocks any claim that a model "matches human uncertainty."

## 6. Why It Is Hard

**Non-identifiability under cross-sectional sampling.** With one label per annotator per item, the observed cell counts have $n(K-1)$ degrees of freedom from $p$ plus $|\mathcal{A}|K(K-1)$ from the channels, but only $nm$ observations. Two generative stories — an ambiguous item with clean annotators, and a clean item with biased annotators — produce identical count matrices. The literature resolves this by *assumption* (Dawid–Skene fixes $C_a$ to be item-independent), not by measurement. Repeated measures per annotator per item would break the tie; almost no corpus collects them.

Second obstruction: **an evaluation that does not measure what it names.** "Inter-annotator agreement $\alpha=0.8$" is reported as data quality, but $\alpha$ is a monotone function of both item ambiguity and annotator noise and cannot separate them; a corpus of unambiguous items with sloppy annotators and one of ambiguous items with careful annotators can report the same $\alpha$.

Third: **entropy estimation bias at small $m$.** At the $m=3$–$5$ typical of benchmarks, the plug-in entropy estimator's bias is the same order as the between-item differences one wants to detect.

## 7. Current Research (as of 2026)

- **Learning-from-disagreement** as a standing programme: the LeWiDi shared tasks (SemEval 2023, and follow-ons) supply soft-label corpora and score models on distributional metrics. Groups: Poesio and colleagues (Queen Mary), Plank's MaiNLP lab (LMU Munich), Uma et al.'s JAIR survey lineage.
- **Sociodemographic conditioning and pluralistic alignment** — steering models to per-group distributions rather than a single consensus (Sorensen et al., "A Roadmap to Pluralistic Alignment", ICML 2024).
- **LLM annotators and disagreement collapse** — evidence that LLM label distributions are systematically lower-entropy than human ones, so replacing crowds with models erases the signal being studied. *(frontier — verify: the effect is repeatedly reported but the sampling-temperature confound is rarely controlled.)*
- **Annotation-error detection** at benchmark scale (Cleanlab lineage; MMLU/GPQA audits) — separating error from ambiguity is the explicit goal, and the two are still conflated in most audit protocols.

## 8. Concrete Next Experiment

**Question.** What fraction of observed disagreement is within-annotator instability rather than item ambiguity or between-annotator bias?

**Scale.** 2,000 items sampled from an existing high-annotation corpus (ChaosNLI MNLI subset, stratified by entropy decile). 60 annotators, each labels all 2,000 items. **Each annotator re-labels a random 400 of them after a $\ge 14$-day washout, with item order re-randomised.** Total 132,000 judgments; at $\$0.06$/judgment that is ≈ $\$8$k plus overhead.

**Control arm.** The same 60 annotators, same interface, on 200 items pre-screened by three experts as unambiguous (unanimous expert agreement, single-clause entailment). Within-annotator flip rate on this arm is the pure instability floor $1-s_a^{\text{ctrl}}$.

**Deciding number.** The ratio
$$R = \frac{1-s_a^{\text{treat}} - (1-s_a^{\text{ctrl}})}{\,\overline{D_o}\,},$$
the share of total observed disagreement attributable to *item-driven* within-annotator instability. If $R < 0.10$, disagreement on ambiguous items is stable per annotator, so it is between-annotator structure (signal) and per-annotator modelling is the right method. If $R > 0.30$, a third or more of the "signal" is a single annotator flipping on re-exposure — it is noise, and soft labels are fitting a coin. Report $R$ with a bootstrap CI over annotators; the experiment is decisive if the CI width is under $0.08$, which 60 annotators supplies.

## 9. Key References

- **[Foundational]** A. P. Dawid, A. M. Skene. *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* Journal of the Royal Statistical Society C, 1979.
- **[Foundational]** R. Artstein, M. Poesio. *Inter-Coder Agreement for Computational Linguistics.* Computational Linguistics 34(4), 2008.
- **[Foundational]** D. Reidsma, J. Carletta. *Reliability Measurement Without Limits.* Computational Linguistics 34(3), 2008.
- **[Foundational]** L. Aroyo, C. Welty. *Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation.* AI Magazine 36(1), 2015.
- **[SOTA]** E. Pavlick, T. Kwiatkowski. *Inherent Disagreements in Human Textual Inferences.* TACL 7, 2019.
- **[SOTA]** Y. Nie, X. Zhou, M. Bansal. *What Can We Learn from Collective Human Opinions on Natural Language Inference Data?* EMNLP 2020.
- **[SOTA]** A. M. Davani, M. Díaz, V. Prabhakaran. *Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotation.* TACL 10, 2022.
- **[SOTA]** J. Peterson, R. Battleday, T. Griffiths, O. Russakovsky. *Human Uncertainty Makes Classification More Robust.* ICCV 2019.
- **[SOTA]** J. Baan, W. Aziz, B. Plank, R. Fernández. *Stop Measuring Calibration When Humans Disagree.* EMNLP 2022.
- **[SOTA]** C. Northcutt, A. Athalye, J. Mueller. *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* NeurIPS Datasets and Benchmarks, 2021.
- **[Survey]** A. Uma, T. Fornaciari, D. Hovy, S. Paun, B. Plank, M. Poesio. *Learning from Disagreement: A Survey.* JAIR 72, 2021.
- **[Survey]** B. Plank. *The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation.* EMNLP 2022.
- **[Context]** P. Röttger, B. Vidgen, D. Hovy, J. Pierrehumbert. *Two Contrasting Data Annotation Paradigms for Subjective NLP Tasks.* NAACL 2022.
- **[Context]** M. Sap, S. Swayamdipta, L. Vianna, X. Zhou, Y. Choi, N. A. Smith. *Annotators with Attitudes: How Annotator Beliefs and Identities Bias Toxic Language Detection.* NAACL 2022.
- **[Context]** M. L. Gordon, M. S. Lam, J. S. Park, K. Patel, J. Hancock, T. Hashimoto, M. S. Bernstein. *Jury Learning: Integrating Dissenting Voices into Machine Learning Models.* CHI 2022.

## 10. Worked Example

Take one MNLI premise–hypothesis pair with 100 ChaosNLI annotations distributed $(E,N,C) = (48, 45, 7)$, so $\hat p = (0.48, 0.45, 0.07)$, $H(\hat p) = 0.90$ nats.

**Step 1 — what a normal benchmark sees.** The original SNLI/MNLI protocol drew $m=5$. A binomial draw from $\hat p$ at $m=5$ gives majority label E with probability ≈ 0.40, N with ≈ 0.36, and no majority (2-2-1) with ≈ 0.24. The gold label on this item is a coin flip. Accuracy on it is not a measurement of the model.

**Step 2 — the plug-in entropy at $m=5$.** Averaging $H$ over those draws gives $\mathbb{E}[H(\hat p^{(5)})] \approx 0.72$ nats against the true 0.90 — a 0.18 nat downward bias, 20% of $\log 3$. So "this item is ambiguous" is under-detected exactly where it matters.

**Step 3 — the obstruction.** Two generative stories fit the 100-annotation counts equally:

| Story | $p_i$ (item) | annotator channel | predicted counts |
|---|---|---|---|
| A: ambiguous item, clean annotators | $(0.48,0.45,0.07)$ | $C_a \approx I$ | $(48,45,7)$ |
| B: clean item, split annotators | $(1,0,0)$ | half the pool maps E→N | $(48,45,7)$ |

Cross-sectional data cannot distinguish them. Under A, a model should output $(0.48,0.45,0.07)$ and soft-label training is correct. Under B, the target is $(1,0,0)$ and soft labels teach the model a population artifact. Dawid–Skene picks A only because its item-independent-channel assumption forbids the E→N map from being content-specific — the assumption, not the data, decides.

**Step 4 — what breaks the tie.** Re-show the item to the same 100 annotators after washout. Story A with stable annotators predicts self-agreement $s_a \approx 1$; Story B also predicts $s_a \approx 1$; but a third story — annotators individually indifferent, flipping at random — predicts $s_a \approx 0.5$. That third story is currently indistinguishable from both, and it is the one under which the whole soft-label programme is fitting noise. No published corpus reports $s_a$ at scale. That missing number is the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*