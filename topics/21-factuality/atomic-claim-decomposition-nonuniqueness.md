---
id: 21-factuality/atomic-claim-decomposition-nonuniqueness
title: "Atomic Claim Decomposition Is Not Unique"
topic: 21-factuality
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Atomic Claim Decomposition Is Not Unique

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/atomic-claim-decomposition-nonuniqueness` · **Status:** methodologically-blocked

## 1. Problem Statement

Nearly every long-form factuality metric — FActScore, SAFE, VeriScore, FactCheck-GPT — works in two stages: decompose a generated passage into a set of "atomic facts", then verify each one against evidence and report the supported fraction. The metric is the mean over a set the pipeline itself constructed.

The problem: **the decomposition is not a function of the passage.** Different decomposers, or the same decomposer at a different temperature, return sets of different cardinality and different granularity from the same text, and the resulting precision changes by amounts comparable to the model gaps the metric is used to adjudicate.

Three variants, with different difficulty:

- **Measurement.** Define a decomposition-invariant factuality score, or bound the score's sensitivity to the decomposer. Currently open and the binding one.
- **Method.** Build a decomposer whose output is stable under paraphrase, reseeding, and model swap, at fixed verification cost. Partially addressed (Claimify, VeriScore).
- **Theory.** Show whether a canonical atomic decomposition exists at all — i.e. whether "atomicity" picks out a unique partition of propositional content. There is strong reason to think it does not.

Solving it means: two independent labs, using different decomposers on the same 500 generations, report factuality scores that agree within the width of their own confidence intervals, and preserve system ranking.

## 2. Formal Setting

Let $y$ be a generated passage, $\mathcal{C}$ the space of natural-language claims. A **decomposer** is a stochastic map $D: \mathcal{Y} \to 2^{\mathcal{C}}$, sampled as $S = D(y; \theta, \tau, \pi)$ where $\theta$ is the extractor model, $\tau$ temperature, $\pi$ the prompt. Measured as: run the extractor, collect the emitted claim list, $n(S) = |S|$.

A **verifier** is $V: \mathcal{C} \times \mathcal{E} \to \{0,1\}$ against evidence corpus $\mathcal{E}$ (retrieval + NLI, or a search-augmented LLM judge). Measured as: the pipeline's per-claim supported/unsupported label.

The reported score is

$$F(y) = \frac{1}{n(S)} \sum_{c \in S} V(c, \mathcal{E}).$$

Define the **decomposition sensitivity** of a passage over a family $\mathcal{D}$ of admissible decomposers:

$$\Delta(y) = \sup_{D_1, D_2 \in \mathcal{D}} \big| F_{D_1}(y) - F_{D_2}(y) \big|,$$

and the **rank instability** over systems $M_1, M_2$ with corpora $Y_1, Y_2$:

$$R = \Pr_{D \sim \mathcal{D}}\big[\operatorname{sign}(\bar F_D(Y_1) - \bar F_D(Y_2)) \neq \operatorname{sign}(\bar F_{D^\star}(Y_1) - \bar F_{D^\star}(Y_2))\big],$$

with $D^\star$ a designated reference. $R$ is measured by resampling decomposers and counting sign flips.

Two coverage/faithfulness quantities, both measurable by human annotation:

- **Coverage** $\kappa(S,y) = $ fraction of the passage's checkable content entailed by $\bigwedge_{c\in S} c$.
- **Extraction faithfulness** $\phi(S,y) = \frac{1}{n}\sum_c \mathbb{1}[y \models c]$ — claims not entailed by the source are extractor hallucinations.

Assumptions the pipeline rests on, and their status:

1. **Atomicity is well defined** — that each $c$ carries one indivisible fact. *Violated.* "Barack Obama was born in Honolulu in 1961" splits into 1, 2, or 3 claims, all defensible.
2. **Claims are independent** so the mean is a meaningful aggregate. *Violated.* Splitting one compound sentence into three duplicates its evidential weight threefold; $\mathbb{E}[F]$ is a weighted average whose weights the decomposer chooses.
3. **Claims are self-contained** after decontextualization. *Violated.* Pronoun and entity resolution failures produce claims that are unverifiable in principle, which verifiers then score as unsupported.
4. **$V$ is granularity-invariant.** *Violated.* Finer claims are easier to support; verification precision rises with atomicity while informativeness falls.

## 3. State of the Art

**Established.**

- FActScore (Min et al., EMNLP 2023) fixed the two-stage template: InstructGPT decomposition of biography generations, retrieval against Wikipedia, supported-fraction score. Its own automatic estimator was validated against human annotation on ~500 people-biography generations.
- Decontextualization as a separate, formalizable step (Choi et al., TACL 2021) — a sentence is rewritten to stand alone, with a feasible/infeasible label. This is the only part of the pipeline with a clean task definition and human agreement numbers.
- The sensitivity itself is established, not conjectural. Wanner et al., *A Closer Look at Claim Decomposition* (\*SEM 2024) showed FActScore values move materially with the decomposition method on the same generations and proposed DecompScore to control for claim count.
- Molecular Facts (Gunjal & Durrett, EMNLP 2024) established the trade-off direction: over-decontextualization adds context that makes claims trivially verifiable; under-decontextualization makes them ambiguous. They give explicit desiderata (minimality plus decontextuality) rather than an "atomic" primitive.

**Claimed but unablated.**

- VeriScore (Song, Kim, Iyyer, 2024) extracts only *verifiable* claims and reports better correlation with human judgment across domains. The claim that its extractor is stable under reseeding is not ablated.
- Claimify (Metropolitansky & Larson, Microsoft Research, 2025) adds explicit disambiguation and abstention when a sentence's referents are underdetermined, reporting higher entailment and coverage than prior extractors. Reported as benchmark numbers on their evaluation set; no cross-lab replication.
- SAFE (Wei et al., 2024, LongFact) reports superhuman agreement with crowdworkers on individual facts, but the decomposition stage is a single prompted Gemini/GPT call and is not varied in the ablations.

**Benchmark-number-only.** Every published "our decomposer is better" result is a single-lab number on a single extractor family. No shared decomposition benchmark with held-out human partitions exists.

## 4. What Is Known

- Claim counts per passage vary by roughly a factor of 2–3 across published extractors on the same inputs (FActScore-style exhaustive splitting vs. VeriScore-style verifiable-only extraction), at the scale of a few hundred biography- or LongFact-style generations. Verifiable-only extraction discards a large fraction of the exhaustive set.
- Score shifts from swapping the decomposer alone are on the order of several points on a 0–100 scale — the same order as reported gaps between adjacent frontier models on FActScore. Measured on hundreds of generations, not thousands *(verify exact magnitudes against Wanner et al. 2024 and VeriScore 2024 tables before citing a specific number)*.
- Extractor hallucination is nonzero: prompted decomposers emit claims not entailed by the source passage, and this is the failure Claimify's disambiguation stage is built to reduce.
- Decontextualization has human ceiling numbers: Choi et al. (TACL 2021) report substantial but not near-perfect annotator agreement on whether a sentence can be decontextualized at all — the ambiguity is in the task, not only the model.
- Longer generations get systematically better FActScore-style precision when claim counts are unnormalized, because later sentences are hedgier and split into fewer checkable atoms. This is why DecompScore and length-controlled variants exist.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no definition of "atomic claim" that is operational enough to write an inter-annotator agreement protocol against. Without it, $\Delta(y)$ has no denominator: we cannot say a decomposer is *wrong*, only *different*. Every downstream factuality number inherits this.
- **Theoretically open.** Whether any decomposition functional satisfying (i) coverage, (ii) source-entailment, (iii) minimality, and (iv) uniqueness up to logical equivalence exists. Informally, natural-language propositional content is not a lattice with unique atoms — conjunction, presupposition, and gradable predicates all block it — but no impossibility theorem has been written down for this setting.
- **Empirically open.** The rank-instability number $R$ at scale. Nobody has run $k \ge 10$ decomposers $\times$ $\ge 5$ systems $\times$ $\ge 1{,}000$ generations with a fixed verifier and reported how often the system ranking flips. This is a single well-scoped run, and its absence is the largest cheap gap in the area.
- **Empirically open.** Whether decomposition-induced variance is larger or smaller than verifier-induced variance. Both are confounded in every published number.

## 6. Why It Is Hard

**Non-identifiability with absent ground truth.** The obstruction is not compute. It is that $F(y)$ is a mean over a set with no canonical cardinality, so the metric has a free parameter — the granularity — that the experimenter sets implicitly by prompt choice. Splitting a claim into two supported halves raises the score; merging two claims where one is false raises it too. The map from "how factual is this text" to a number is many-to-one in the wrong direction: many numbers, one text.

This compounds with **confounded measurement**: decomposition error and verification error are observed only through their composition. A claim scored unsupported may be false, may be underspecified by the extractor, or may be true but unretrievable. Published pipelines report one number for all three.

And the evaluation does not measure what it names: "factual precision" names a property of the text, but is computed as a property of the (text, decomposer) pair.

## 7. Current Research (as of 2026)

- **Granularity-controlled scoring.** DecompScore-style normalization and length control (JHU CLSP — Van Durme, Dredze and collaborators).
- **Verifiability-gated extraction.** VeriScore (UMass Amherst — Iyyer group): extract only claims a search engine could adjudicate, sidestepping unverifiable atoms rather than defining them away.
- **Disambiguation-and-abstain extraction.** Claimify (Microsoft Research): decompose only where referents are determined; abstain otherwise. Directly targets extractor hallucination.
- **Dependency-aware verification.** DnDScore and related work on scoring claims conditioned on their siblings rather than in isolation, attacking assumption 2 *(frontier — verify)*.
- **Decomposition-free factuality.** Passage-level entailment or citation-precision metrics that never atomize *(frontier — verify)*; the trade is losing the per-claim diagnostics that made the atomic pipelines useful.

## 8. Concrete Next Experiment

**The decomposer-swap ranking audit.**

- **Scale.** $N = 1{,}000$ generations (500 LongFact prompts, 500 FActScore-style biography prompts) from $m = 5$ models spanning a real quality range. $k = 12$ decomposers: 4 prompt styles (FActScore-exhaustive, VeriScore-verifiable, Claimify-disambiguated, sentence-as-claim) $\times$ 3 extractor backbones. **Verification is frozen** — one retrieval corpus, one verifier, one seed — so all variance is attributable to decomposition.
- **Control arm.** The same 12-cell grid with the *decomposer* frozen and the *verifier* varied 12 ways (retrieval depth, NLI model, judge model). This isolates whether decomposition variance exceeds verification variance, which no published result separates.
- **Deciding number.** $R$, the fraction of the $\binom{5}{2}=10$ system pairs whose sign flips across the 12 decomposers. If $R > 0.1$ — one pair in ten reverses under a decomposer swap — atomic-decomposition factuality scores cannot be used to rank systems without publishing the decomposer as part of the metric, and the field should report $F$ with a decomposition-variance error bar. If $R < 0.02$ and $\operatorname{sd}_D(\bar F) < \operatorname{sd}_V(\bar F)$, the non-uniqueness is real but benign for ranking, and the problem downgrades to a calibration nuisance.
- **Cost.** ~12,000 decompositions and ~1–2 M verification calls. Days, not months, on a single lab budget. The absence of this run is the point.

## 9. Key References

- **[Foundational]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. — arXiv:2305.14251
- **[Foundational]** Eunsol Choi, Jennimaria Palomaki, Matthew Lamm, Tom Kwiatkowski, Dipanjan Das, Michael Collins. *Decontextualization: Making Sentences Stand-Alone.* TACL, 2021.
- **[SOTA]** Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, Quoc V. Le. *Long-form Factuality in Large Language Models.* 2024. — arXiv:2403.18802
- **[SOTA]** Yixiao Song, Yekyung Kim, Mohit Iyyer. *VeriScore: Evaluating the Factuality of Verifiable Claims in Long-Form Text Generation.* 2024. — arXiv:2406.19276
- **[SOTA]** Dasha Metropolitansky, Jonathan Larson. *Towards Effective Extraction and Evaluation of Factual Claims.* Microsoft Research, 2025.
- **[Analysis]** Miriam Wanner, Seth Ebner, Zhengping Jiang, Mark Dredze, Benjamin Van Durme. *A Closer Look at Claim Decomposition.* \*SEM 2024.
- **[Analysis]** Anisha Gunjal, Greg Durrett. *Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification.* EMNLP Findings 2024.
- **[Related]** Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, Kelvin Guu. *RARR: Researching and Revising What Language Models Say, Using Language Models.* ACL 2023.

## 10. Worked Example

Passage sentence: *"Obama, who was born in Honolulu in 1961, later taught constitutional law at the University of Chicago for twelve years."*

Assume ground truth: the birthplace and year are correct; the tenure was 1992–2004, so "twelve years" is right, but suppose the generated passage instead said *"for fifteen years"* — one false element.

**Decomposer A (exhaustive, FActScore-style)** — 5 claims:
1. Obama was born in Honolulu. ✓
2. Obama was born in 1961. ✓
3. Obama taught constitutional law. ✓
4. Obama taught at the University of Chicago. ✓
5. Obama taught there for fifteen years. ✗

$F_A = 4/5 = 0.80$.

**Decomposer B (coarse, clause-level)** — 2 claims:
1. Obama was born in Honolulu in 1961. ✓
2. Obama taught constitutional law at the University of Chicago for fifteen years. ✗

$F_B = 1/2 = 0.50$.

**Decomposer C (verifiability-gated, VeriScore-style)** — drops the duration as not cleanly adjudicable by a single search, keeps 4 claims, all supported. $F_C = 4/4 = 1.00$.

One sentence, one error, three defensible decompositions, $\Delta = 0.50$ on this passage.

Now the aggregate effect. Take two systems on 100 sentences each. System $M_1$ writes compound, detail-dense sentences (mean 5 atoms under A) with 1 error per 5 atoms. System $M_2$ writes short, hedged sentences (mean 2 atoms under A) with 1 error per 4 atoms — genuinely worse per fact.

- Under A: $\bar F_1 = 0.80$, $\bar F_2 = 0.75$. $M_1$ wins.
- Under B, where $M_1$'s compound sentences collapse to 2 claims and its single error taints one of them, while $M_2$'s already-short sentences are near-unchanged: $\bar F_1 = 0.50$, $\bar F_2 = 0.72$. $M_2$ wins.

The ranking reverses with no change to either model, either passage, or the verifier. That reversal — not the per-passage $\Delta$ — is the obstruction, and it is exactly the $R$ that Section 8 asks someone to measure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*