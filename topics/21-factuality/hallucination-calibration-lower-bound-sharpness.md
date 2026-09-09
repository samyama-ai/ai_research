---
id: 21-factuality/hallucination-calibration-lower-bound-sharpness
title: "Sharp Characterization of Hallucination as Calibration Error"
topic: 21-factuality
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sharp Characterization of Hallucination as Calibration Error

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/hallucination-calibration-lower-bound-sharpness` · **Status:** partially-solved

## 1. Problem Statement

Kalai and Vempala (STOC 2024) proved a *lower* bound: a language model that is calibrated on its own training distribution must hallucinate at a rate at least the fraction of facts that appeared exactly once in training, minus its miscalibration. The open problem is the **converse and the sharpness**: how much of the hallucination actually produced by a deployed model is explained by this calibration floor, and how much is residual error the theory does not touch.

Three variants, with different difficulty:

- **Theory variant.** Is the bound tight? Find a matching upper bound: a hypothesis class, distribution family, and estimator such that hallucination rate $\le$ monofact rate $+$ miscalibration $+ o(1)$. Or exhibit a family where the gap is $\Omega(1)$ and characterize what closes it.
- **Measurement variant.** Decompose an observed hallucination rate into (i) an information-theoretic floor from missing mass, (ii) a calibration-error term, (iii) a residual (retrieval failure, reasoning error, sycophancy, decoding artifacts). Currently nobody can produce this decomposition for a real model because the training-corpus fact multiplicity is not observable post hoc.
- **Method variant.** If the floor is real, the only escape is abstention — the model must be allowed to answer "I don't know", which breaks calibration in the density-estimation sense. Design a training objective whose optimum trades a controlled amount of calibration for a quantified reduction in hallucination, and prove the trade curve.

Solving it means: given a model and a corpus, predicting the hallucination rate to within a stated tolerance from calibration and corpus statistics alone.

## 2. Formal Setting

Let $\mathcal{X}$ be the set of strings. A *fact distribution* $p$ over a set of plausible statements induces a partition $\mathcal{X} = F \sqcup E$ into **facts** (valid) and **errors** (invalid). Hallucination rate of model $\hat p$:

$$\mathrm{hal}(\hat p) \;=\; \sum_{x \in E} \hat p(x) \;=\; \hat p(E).$$

*As measured:* sample $n$ generations, adjudicate each against a reference source, report the fraction judged invalid. The adjudicator is a human or an LLM judge; inter-annotator agreement on "is this a hallucination" is typically $\kappa \approx 0.6$–$0.8$, so $\mathrm{hal}$ carries an irreducible measurement error of a few points.

Training corpus $T$ of $|T| = N$ i.i.d. draws from $p$. The **monofact rate** is the Good–Turing missing-mass estimator:

$$\widehat{\mathrm{MF}} \;=\; \frac{|\{x : \mathrm{count}_T(x) = 1\}|}{N}.$$

*As measured:* requires a fact-level deduplication of the pretraining corpus — count how many distinct atomic facts occur exactly once. This is the quantity nobody computes at scale.

**Miscalibration.** Kalai–Vempala's notion is not per-token ECE. Partition generations into bins by predicted probability; for a bin $B$,

$$\mathrm{miscal}(\hat p) \;=\; \sum_{B} \Big| \hat p(B) - p(B) \Big|,$$

i.e. total-variation-style deviation between the model's mass on a probability bin and the true mass. The plug-in binned estimator $\widehat{\mathrm{ECE}}$ is **upward biased** at finite sample: bias scales roughly as $O(\sqrt{B/n})$ for $B$ bins and $n$ samples (Kumar–Liang–Ma, NeurIPS 2019; Roelofs et al., AISTATS 2022).

The theorem then has the shape

$$\mathrm{hal}(\hat p) \;\gtrsim\; \widehat{\mathrm{MF}} \;-\; \mathrm{miscal}(\hat p) \;-\; \frac{|\text{distinct facts seen}|}{N} \;-\; \tilde O\!\left(\sqrt{\tfrac{\log N}{N}}\right).$$

**Assumptions, and which are violated:**

1. *Facts are i.i.d. draws.* Violated — corpora are deduplicated, curated, and heavily correlated; near-duplicate paraphrases mean the true multiplicity of a fact is far above its string count.
2. *Binary valid/invalid partition.* Violated — factuality is graded, context-dependent, and time-varying.
3. *Model matches training distribution.* Violated by design — RLHF, instruction tuning, and safety training move $\hat p$ far off the pretraining distribution.
4. *No abstention.* Violated — deployed models emit hedges and refusals, which sit outside $F \sqcup E$.
5. *Single-fact generations.* Violated — a paragraph carries many facts with correlated errors.

## 3. State of the Art

**Theory SOTA (established).** Kalai & Vempala, *Calibrated Language Models Must Hallucinate* (STOC 2024, arXiv:2311.14648). The lower bound above is proved, for arbitrary models, with no assumption on architecture. Its content is genuinely information-theoretic: it is a Good–Turing argument, not an empirical claim. The follow-up, Kalai, Nachum, Vempala & Zhang, *Why Language Models Hallucinate* (2025, arXiv:2509.04664), reduces generative error to a binary classification problem ("Is-It-Valid") and shows generative error rate is at least roughly twice the misclassification rate of the corresponding classifier — plus a post-training argument that binary-graded benchmarks reward guessing over abstention.

**Established vs. claimed.** Established: the lower bound and the reduction. *Claimed but unablated:* that this bound explains a material share of hallucination in frontier models. No paper has measured $\widehat{\mathrm{MF}}$ on a frontier pretraining corpus and compared it to a measured $\mathrm{hal}$. The monofact numbers cited in discussion are illustrative estimates over reference-style facts, not corpus-wide measurements.

**Empirical SOTA.** Calibration of base models is good and post-training destroys it: the GPT-4 Technical Report (OpenAI, 2023) reports MMLU ECE of about $0.007$ for the pre-trained model and about $0.074$ after RLHF — a 10× degradation. Kadavath et al. (*Language Models (Mostly) Know What They Know*, 2022) show few-shot calibration on multiple choice is near-diagonal at 52B and improves with scale. Hallucination detection SOTA is semantic entropy (Farquhar et al., *Nature* 2024), which uses model uncertainty and reaches AUROC around $0.75$–$0.8$ on QA confabulation detection — a benchmark number, not a decomposition.

**Contrast.** Xu, Jain & Kankanhalli (*Hallucination is Inevitable*, arXiv:2401.11817, 2024) argue inevitability from computability. That result is much weaker in practice: it constrains no rate.

## 4. What Is Known

- The lower bound holds unconditionally for calibrated models. Scale: it is a theorem, scale-free.
- Good–Turing missing mass concentrates within $O(1/\sqrt{N})$ of the singleton fraction (Good, *Biometrika* 1953; McAllester–Schapire, COLT 2000). So $\widehat{\mathrm{MF}}$ is estimable to a fraction of a point at corpus scale — *if* the fact-level count is available.
- RLHF trades calibration for helpfulness: ECE $0.007 \to 0.074$ on MMLU at GPT-4 scale (OpenAI 2023). This is the single most reproduced calibration regularity.
- Abstention escapes the bound. A model outputting "I don't know" with probability $q$ has hallucination rate scaled by roughly $(1-q)$ and is no longer calibrated to $p$. This is a theorem-level observation, not a conjecture.
- Distance-to-calibration is the well-behaved measurement: Błasiok, Gopalan, Hu & Nakkiran (STOC 2023) show ECE is not robust and give a smooth calibration error that is polynomially related to true distance from calibration.
- Fact multiplicity drives extraction, not just hallucination: Allen-Zhu & Li (*Physics of Language Models 3.1*, ICML 2024) show synthetic biographies must be seen with multiple *paraphrases* before knowledge is extractable — single-exposure facts are near-unextractable at 100M–1B parameter scale.

## 5. What Is Not Known

- **Theoretically open.** No matching upper bound. Nobody has shown a nontrivial class where $\mathrm{hal} \le \widehat{\mathrm{MF}} + \mathrm{miscal} + o(1)$. The bound may be loose by a large constant or an additive $\Omega(1)$; either would change its interpretation entirely.
- **Theoretically open.** The abstention/calibration trade curve. There is no characterization of the Pareto frontier between abstention rate $q$, calibration error, and hallucination rate.
- **Empirically open.** $\widehat{\mathrm{MF}}$ for a real pretraining corpus at fact granularity. This is runnable — it needs an entity-relation extraction pass over a few trillion tokens plus semantic dedup — and nobody has published it.
- **Empirically open.** Whether varying singleton fraction in a controlled corpus moves measured hallucination one-for-one. Runnable at 1B scale for well under $\$50\mathrm{k}$.
- **Methodologically blocked.** The *decomposition itself*. Splitting an observed 12% hallucination rate into "floor", "miscalibration", and "residual" requires a definition of the fact partition $F \sqcup E$ that is stable across annotators and across the training/eval boundary. That definition does not exist for open-ended generation.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the fact multiplicity from the observable corpus.** The theorem's driving quantity is how often an *atomic fact* appears, but corpora contain paraphrases, translations, tables, and templated restatements. A fact with one string occurrence may have 40 semantic occurrences, or one. Choosing the semantic-equivalence threshold moves $\widehat{\mathrm{MF}}$ by tens of points, and there is no principled way to set it — the "right" granularity is whatever granularity the model generalizes over, which is a property of the trained model, not the corpus. So the predictor and the prediction target are entangled.

Second obstruction, compounding: **the evaluation does not measure what it names.** Reported ECE is a binned plug-in estimator on multiple-choice benchmarks; the theorem's $\mathrm{miscal}$ is a total-variation deviation over generation-level probability bins on the open-ended distribution. These are different numbers. Plugging MMLU ECE into the bound is not licensed.

## 7. Current Research (as of 2026)

- **OpenAI (Kalai, Nachum, Zhang) and Georgia Tech (Vempala)** — extending the classification reduction and pushing benchmark redesign toward explicit confidence thresholds that score abstention neutrally rather than as a miss.
- **Harvard/Apple/UC Berkeley theory (Błasiok, Gopalan, Nakkiran, Hu)** — calibration distance, multicalibration, and proper-loss conditions under which optimization yields calibration; the natural source of a sharper $\mathrm{miscal}$ term.
- **OATML Oxford (Gal, Farquhar, Kossen)** — semantic-entropy family; entropy-based estimators that could serve as a proxy for the missing-mass term without corpus access. *(frontier — verify: whether semantic entropy has been formally linked to the Kalai–Vempala floor.)*
- **Synthetic-corpus interpretability (Allen-Zhu, Li; and the growing "Physics of LMs" replication community)** — controlled fact-multiplicity pretraining, the closest existing infrastructure to the deciding experiment.
- *(frontier — verify)* Several groups report that reinforcement learning with an explicit abstention reward reduces hallucination without proportionate accuracy loss; ablations isolating calibration change are not yet public.

## 8. Concrete Next Experiment

**Goal.** Measure the slack in the lower bound as a function of the singleton fraction.

**Scale.** Pretrain a 1.4B-parameter decoder from scratch on a synthetic biography corpus of $\approx 30$B tokens, in the style of Allen-Zhu & Li: $10^6$ synthetic individuals, each with 6 attributes, all facts generated from templates so that *fact multiplicity is exact by construction* — this removes the non-identifiability obstruction of §6.

**Arms.** Sweep the singleton fraction $s \in \{0.00, 0.05, 0.10, 0.20, 0.40\}$ — the fraction of individuals whose attributes appear exactly once, with the remainder appearing $\ge 5$ times in $\ge 5$ distinct paraphrases. Five runs, $\approx 1.5\times10^{21}$ FLOPs total, roughly 4 GPU-weeks on 64 H100s.

**Control arm.** $s = 0.00$. The bound predicts a hallucination floor of $0$. Whatever hallucination is measured here is the residual — retrieval, interference, decoding — and must be subtracted from every other arm.

**Protocol.** Query each model on held-out attribute questions with greedy decoding. Exact-match against ground truth gives $\mathrm{hal}$ directly (no judge, no annotator noise). Estimate $\mathrm{miscal}$ with a debiased binned estimator (Kumar–Liang–Ma) at $B = 15$ bins, $n = 10^5$ queries.

**The deciding number.** The slack

$$\Delta(s) \;=\; \big[\mathrm{hal}(s) - \mathrm{hal}(0)\big] \;-\; \big[s - \mathrm{miscal}(s)\big].$$

- $\Delta(s) \le 0.05$ across the sweep $\Rightarrow$ the bound is **sharp**; hallucination on unseen-once facts is exactly the calibration floor, and mitigation must be abstention.
- $\Delta(0.40) \ge 0.30 \Rightarrow$ the bound is **loose**; most hallucination is residual, and the calibration framing is a small part of the story.

Report $\Delta(s)$ with bootstrap CIs. One number, five runs, a month of compute.

## 9. Key References

- **[Foundational]** Adam Tauman Kalai, Santosh S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC 2024. — arXiv:2311.14648
- **[SOTA]** Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang. *Why Language Models Hallucinate.* 2025. — arXiv:2509.04664
- **[Foundational]** I. J. Good. *The Population Frequencies of Species and the Estimation of Population Parameters.* Biometrika 40(3–4), 1953.
- **[Theory]** Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran. *A Unifying Theory of Distance from Calibration.* STOC 2023. — arXiv:2211.16886
- **[Method]** Ananya Kumar, Percy Liang, Tengyu Ma. *Verified Uncertainty Calibration.* NeurIPS 2019. — arXiv:1909.10155
- **[Method]** Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, Michael C. Mozer. *Mitigating Bias in Calibration Error Estimation.* AISTATS 2022.
- **[Empirical]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Empirical]** OpenAI. *GPT-4 Technical Report.* 2023. — arXiv:2303.08774
- **[Empirical]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 2024.
- **[Empirical]** Zeyuan Allen-Zhu, Yuanzhi Li. *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction.* ICML 2024. — arXiv:2309.14316
- **[Contrast]** Ziwei Xu, Sanjay Jain, Mohan Kankanhalli. *Hallucination is Inevitable: An Innate Limitation of Large Language Models.* 2024. — arXiv:2401.11817
- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. — arXiv:1706.04599
- **[Survey]** Lei Huang et al. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025. — arXiv:2311.05232

## 10. Worked Example

Take bibliographic citation generation — the canonical case where the theory should bite.

Suppose a corpus of $N = 10^7$ paper-reference mentions. Count distinct references: 4.2M appear once, 1.1M appear twice, the rest more. Good–Turing missing mass:

$$\widehat{\mathrm{MF}} \;=\; \frac{4.2 \times 10^6}{10^7} \;=\; 0.42.$$

Assume the model's generation-level miscalibration is $\mathrm{miscal} = 0.05$. The bound predicts

$$\mathrm{hal} \;\gtrsim\; 0.42 - 0.05 \;=\; 0.37.$$

Now measure. A frontier model asked for references in a niche subfield fabricates, say, 30% of them. **The bound is violated — which means one of the inputs is wrong, not the theorem.**

Where it breaks, concretely:

- The model is *not* calibrated to $p$. Post-training taught it to hedge and to prefer high-frequency references. It answers a different distribution, so the bound simply does not apply, and the $0.37$ prediction is void.
- The multiplicity count is wrong. A reference appearing once as a full citation string may appear 30 times as "Vaswani et al." in running text. Recount at a semantic granularity and the singleton fraction drops from $0.42$ to perhaps $0.15$ — but that threshold was chosen by hand.
- $\mathrm{miscal} = 0.05$ was imported from an MMLU multiple-choice ECE. The theorem needs deviation over open-ended generation bins, a quantity that was never measured.

Every one of the three inputs is either unmeasured or measured on the wrong distribution. That is the obstruction in one paragraph: **the theorem is proved, and none of its three quantities is currently observable on a deployed model.** The synthetic experiment in §8 is the only setting where all three are pinned by construction — which is exactly why it is the next experiment.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*