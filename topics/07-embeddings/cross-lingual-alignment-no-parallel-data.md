---
id: 07-embeddings/cross-lingual-alignment-no-parallel-data
title: "Cross-Lingual Representation Alignment Without Parallel Data"
topic: 07-embeddings
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Representation Alignment Without Parallel Data

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/cross-lingual-alignment-no-parallel-data` · **Status:** partially-solved

## 1. Problem Statement

**Input.** Two monolingual corpora $C_s, C_t$ in languages $s,t$, with no sentence pairs, no seed dictionary, no shared script requirement. From them, representation spaces $X \subset \mathbb{R}^d$ (source) and $Y \subset \mathbb{R}^d$ (target) — static word vectors, or contextual encoder states.

**Output.** A map $f: \mathbb{R}^d \to \mathbb{R}^d$ such that $f(x_i)$ lands near $y_j$ exactly when $(i,j)$ are translation-equivalent.

**Objective.** Maximise precision@1 on a held-out bilingual lexicon (bilingual lexicon induction, BLI), or downstream zero-shot transfer accuracy: train a classifier on $s$-language labels, evaluate on $t$.

Three variants, of very different difficulty:

- **Method.** Find $f$ that works. Largely solved for high-resource, typologically close pairs; fails unpredictably elsewhere.
- **Theory.** State conditions on $(C_s, C_t)$ under which $f$ is *identifiable* from monolingual statistics alone. Open. No known distribution-level assumption makes the recovery of the true correspondence provable rather than empirical.
- **Measurement.** Decide whether an alignment is good without a dictionary — the ground truth the setting forbids. Currently circular: every published evaluation uses parallel or dictionary supervision at test time.

Solving it means: for an arbitrary pair including one low-resource language, produce $f$ *and* an a priori certificate that $f$ is right, without any cross-lingual signal.

## 2. Formal Setting

Let $V_s, V_t$ be vocabularies, $|V_s|=n_s$. Embeddings $X \in \mathbb{R}^{n_s \times d}$, $Y \in \mathbb{R}^{n_t \times d}$ from fastText/word2vec on $C_s, C_t$; $d=300$ standard, vectors $\ell_2$-normalised and mean-centred (the normalisation is not cosmetic — it changes reported P@1 by several points).

**Alignment.** Restrict to orthogonal $W \in O(d)$ and a partial permutation $P \in \{0,1\}^{n_s \times n_t}$ (at most one 1 per row/column). The Wasserstein–Procrustes objective:

$$\min_{W \in O(d),\, P \in \mathcal{P}} \; \lVert XW - PY \rVert_F^2 .$$

For fixed $P$, $W$ has the closed form $W = UV^\top$ where $U\Sigma V^\top = \mathrm{SVD}(X^\top P Y)$ (orthogonal Procrustes, Schönemann 1966). For fixed $W$, $P$ is an optimal-transport problem. The joint problem is non-convex and NP-hard in general.

**Measured quantities.**

- $\mathrm{P@}k$ — fraction of source test words whose gold translation is in the top $k$ of $\arg\max_j \, \mathrm{CSLS}(f(x_i), y_j)$. CSLS (cross-domain similarity local scaling) corrects the hubness of cosine retrieval: $\mathrm{CSLS}(a,b) = 2\cos(a,b) - r_t(a) - r_s(b)$, with $r$ the mean cosine to the $K{=}10$ nearest neighbours in the other space.
- **Isometry gap.** Gromov–Hausdorff distance between the top-$m$ ($m{\approx}5000$) frequency slices of $X$ and $Y$ under cosine, or eigenvector similarity of the two $k$-NN graph Laplacians (sum of squared differences of the leading eigenvalues chosen to capture 90% of variance).
- **Unsupervised validation criterion** (used for model selection in MUSE): mean CSLS over the top-10k induced pairs.

**Assumptions, and their status.**

1. *Approximate isomorphism* — $X$ and $Y$ are related by a near-isometry. **Violated.** Vulić et al. (EMNLP 2020) show even two spaces for the *same* language trained on different corpora are not isometric; the gap grows with typological distance and shrinks with corpus size.
2. *Comparable corpora* — $C_s, C_t$ cover similar domains. **Violated** whenever the low-resource side is Bible/Wikipedia and the high-resource side is CommonCrawl.
3. *Orthogonality suffices.* Length-preserving maps are a modelling choice, not a fact; relaxations help on distant pairs but lose the closed form.
4. *Frequency-rank correspondence* — the $k$-th most frequent words are roughly translations. Weak, and it is what unsupervised seeding silently leans on.

## 3. State of the Art

**Established (reproduced independently).**
- Orthogonal Procrustes with a seed dictionary (Xing et al., NAACL 2015; Artetxe et al., EMNLP 2016) — the workhorse; needs supervision.
- **VecMap** (Artetxe, Labaka, Agirre, ACL 2018): unsupervised initialisation from the similarity-matrix distribution plus robust self-learning. Reported successful on all pairs tried including en–fi, where the adversarial method returns near-zero.
- **MUSE** (Conneau et al., ICLR 2018): adversarial init + Procrustes refinement + CSLS. First convincing fully unsupervised BLI; en–es P@1 ≈ 81.7, matching the supervised arm.
- **Gromov–Wasserstein alignment** (Alvarez-Melis & Jaakkola, EMNLP 2018) and **Wasserstein-Procrustes** (Grave, Joulin, Berthet, AISTATS 2019) — optimal-transport formulations, competitive without adversarial training and with fewer hyperparameters.
- **Joint multilingual pretraining** (mBERT; XLM-R, Conneau et al., ACL 2020): no explicit alignment step at all; shared subwords plus multilingual MLM produce transferable representations. This is the practical SOTA and has displaced mapping methods for downstream tasks.

**Claimed but unablated / benchmark-only.**
- "Unsupervised matches supervised" holds only on the MUSE/Dinu test dictionaries for high-resource pairs; Glavaš et al. (ACL 2019) show the ranking of methods changes with the evaluation dictionary and the retrieval rule.
- Every unsupervised method uses *some* cross-lingual signal for model selection or early stopping. Vulić et al. (EMNLP 2019) argue that a weakly-supervised arm with ~500–1000 pairs (or identical strings) beats fully unsupervised methods on most pairs, at negligible cost.
- mBERT cross-lingual ability is *not* explained by shared vocabulary: K et al. (ICLR 2020) find depth and parameter sharing matter, lexical overlap does not. Artetxe, Ruder, Yogatama (ACL 2020) show a monolingual model with relearned embeddings transfers, so joint pretraining is not necessary.

## 4. What Is Known

- **Numbers, static embeddings, fastText Wikipedia, $d{=}300$, 200k vocab, MUSE test dictionaries (1500 pairs):** en–es unsupervised P@1 ≈ 81.7 vs supervised ≈ 81.4; en–ru ≈ 51; en–zh ≈ 32. Adversarial init on en–fi and en–et fails to converge (P@1 < 1) in Søgaard et al. (ACL 2018); VecMap's self-learning recovers en–fi to ≈ 37 (ACL 2018 reported numbers).
- **Isomorphism is the predictor.** Søgaard et al. (2018) report a strong correlation between eigenvector similarity of the two $k$-NN graphs and BLI accuracy across pairs; Patra et al. (ACL 2019) reach the same conclusion with Gromov–Hausdorff distance.
- **Corpus, not just language, drives failure.** Same-language spaces trained on different domains (Wikipedia vs. Common Crawl) already show measurable isometry gaps (Vulić et al., EMNLP 2020) — the "language distance" story is partly a corpus-mismatch story.
- **Contextual scale:** XLM-R large, 2.5 TB CommonCrawl, 100 languages, reaches ≈ 80.9 average XNLI zero-shot accuracy vs ≈ 65 for mBERT (Conneau et al., ACL 2020) — no parallel data, no alignment objective. The *curse of multilinguality*: past ~100 languages at fixed capacity, per-language quality drops.

## 5. What Is Not Known

- **Theoretically open.** No identifiability theorem. There is no known condition on the two corpus-generating distributions under which the translation correspondence is the unique optimum of any monolingual-only objective. All existing guarantees are for the *supervised* Procrustes subproblem.
- **Theoretically open.** Why multilingual MLM produces aligned subspaces at all. No account predicts which languages will align, or the exact form of the curse-of-multilinguality tradeoff.
- **Methodologically blocked.** Unsupervised *validation*. There is no criterion that certifies an alignment is correct using only $C_s, C_t$. Mean-CSLS correlates with accuracy on pairs that work and is uninformative on pairs that fail — precisely where it is needed.
- **Empirically open.** Whether isometry-promoting monolingual training (IsoVec, Marchisio et al., EMNLP 2022) closes the low-resource BLI gap when the low-resource corpus is genuinely small (< 10M tokens) and out-of-domain. Runnable on one GPU; not run systematically across 50+ pairs.
- **Empirically open.** Whether unsupervised alignment adds anything on top of a strong multilingual encoder for truly unseen languages, controlled for tokenizer coverage.

## 6. Why It Is Hard

**Non-identifiability, plus a circular evaluation.**

The objective $\min_{W,P}\lVert XW - PY\rVert_F^2$ has many near-optimal solutions when the point clouds have approximate symmetries; distributional geometry does not distinguish the semantically correct permutation from a structurally equivalent wrong one. Adding data does not help — it is not a variance problem. Under exact isometry the correspondence would be recoverable up to the symmetry group; under the ~10–20% isometry gap actually observed for distant pairs, the objective's global optimum is simply not the truth.

Compounding it: the field cannot tell a good alignment from a bad one without a dictionary, so every "unsupervised" pipeline smuggles supervision into hyperparameter choice, seed strategy, or stopping. The reported gap between unsupervised and weakly-supervised is therefore not a measurement of what the label says.

## 7. Current Research (as of 2026)

- **Isometry-aware monolingual training** — shaping $X$ during training so a later map exists (IsoVec line, Johns Hopkins).
- **Optimal transport with relaxed geometry** — Gromov–Wasserstein and unbalanced OT for non-comparable corpora (MIT/Inria lineage).
- **Alignment inside multilingual LLMs** — locating and editing language-agnostic subspaces in decoder-only models; steering vectors for language identity *(frontier — verify)*.
- **Extremely low-resource extension** — adding a 1000th language to an existing encoder using only religious-domain text; the standard evaluation is FLORES-200 / SIB-200 rather than BLI *(frontier — verify)*.
- **Unsupervised model selection** — self-consistency and back-translation-style criteria for certifying alignments; still no criterion that flags catastrophic failure reliably.

## 8. Concrete Next Experiment

**Question.** Is there any *monolingual-only* statistic that predicts alignment failure before the dictionary is seen?

**Scale.** 30 languages × fastText, two corpus conditions each (Wikipedia; a ≤ 10M-token religious/news corpus) = 60 spaces, 30 pairs against English. Static, $d{=}300$. Total compute: order 200 CPU-hours plus a single GPU-day. Deliberately small — this is a measurement experiment, not a scaling one.

**Arms.**
1. VecMap unsupervised self-learning.
2. **Control:** identical pipeline seeded with 1000 gold pairs (weak supervision), per Vulić et al. 2019.
3. **Control:** identity-string seeding only.

**Predictors, computed with zero cross-lingual information:** Gromov–Hausdorff distance and $k$-NN-graph eigenvector similarity on the top-5000 slices; corpus size; type-token ratio gap; mean-CSLS validation score.

**Deciding number.** Spearman $\rho$ between each predictor and the unsupervised-minus-weakly-supervised P@1 gap, across the 30 pairs. **$\rho \ge 0.8$** for any single monolingual predictor would give the field its missing certificate and turn a methodological block into an engineering step. **$\rho \le 0.5$** for all of them — the expected outcome given that mean-CSLS is known to be uninformative on failing pairs — establishes that unsupervised validation needs a genuinely new object, not a better geometric distance.

## 9. Key References

- **[Foundational]** Tomas Mikolov, Quoc V. Le, Ilya Sutskever. *Exploiting Similarities among Languages for Machine Translation.* arXiv, 2013. — arXiv:1309.4168
- **[Foundational]** Chao Xing, Dong Wang, Chao Liu, Yiye Lin. *Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation.* NAACL-HLT, 2015.
- **[SOTA]** Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou. *Word Translation Without Parallel Data.* ICLR, 2018. — arXiv:1710.04087
- **[SOTA]** Mikel Artetxe, Gorka Labaka, Eneko Agirre. *A Robust Self-Learning Method for Fully Unsupervised Cross-Lingual Mappings of Word Embeddings.* ACL, 2018.
- **[SOTA]** Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL, 2020. — arXiv:1911.02116
- **[Critique]** Anders Søgaard, Sebastian Ruder, Ivan Vulić. *On the Limitations of Unsupervised Bilingual Dictionary Induction.* ACL, 2018.
- **[Critique]** Ivan Vulić, Goran Glavaš, Roi Reichart, Anna Korhonen. *Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?* EMNLP-IJCNLP, 2019.
- **[Critique]** Goran Glavaš, Robert Litschko, Sebastian Ruder, Ivan Vulić. *How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions.* ACL, 2019.
- **[Evidence]** Ivan Vulić, Sebastian Ruder, Anders Søgaard. *Are All Good Word Vector Spaces Isomorphic?* EMNLP, 2020.
- **[Method]** Edouard Grave, Armand Joulin, Quentin Berthet. *Unsupervised Alignment of Embeddings with Wasserstein Procrustes.* AISTATS, 2019.
- **[Method]** David Alvarez-Melis, Tommi Jaakkola. *Gromov-Wasserstein Alignment of Word Embedding Spaces.* EMNLP, 2018.
- **[Method]** Kelly Marchisio, Neha Verma, Kevin Duh, Philipp Koehn. *IsoVec: Controlling the Relative Isomorphism of Word Embedding Spaces.* EMNLP, 2022.
- **[Analysis]** Mikel Artetxe, Sebastian Ruder, Dani Yogatama. *On the Cross-lingual Transferability of Monolingual Representations.* ACL, 2020.
- **[Analysis]** Karthikeyan K, Zihan Wang, Stephen Mayhew, Dan Roth. *Cross-Lingual Ability of Multilingual BERT: An Empirical Study.* ICLR, 2020.
- **[Survey]** Sebastian Ruder, Ivan Vulić, Anders Søgaard. *A Survey of Cross-lingual Word Embedding Models.* Journal of Artificial Intelligence Research, 2019.

## 10. Worked Example

**Instance.** English–Finnish, fastText Wikipedia vectors, 200k vocab, $d{=}300$, MUSE test dictionary.

1. Run MUSE adversarial init. The discriminator loss converges; the induced dictionary looks internally consistent; mean-CSLS over the top-10k induced pairs is a plausible number. **P@1 measured against the gold dictionary: under 1%.** The run produced a $W$ that maps the English cloud onto the Finnish cloud with low Frobenius error and near-zero semantic correctness.
2. Run VecMap self-learning from similarity-distribution init. **P@1 ≈ 37%.**
3. Now compute what an operator without a dictionary could see. Both runs give an orthogonal $W$ with comparable residual. Both give a validation CSLS in the same range. The isometry diagnostic — eigenvector similarity of the two 5-NN graphs on the top-5000 words — flags en–fi as far worse than en–es, which is useful, but it flags *the pair*, not *the run*: it does not separate arm 1 from arm 2, and it is the same number in both.

**The obstruction, made visible.** Two runs on the same inputs differ by ~36 P@1 points. Every quantity computable from $C_s$ and $C_t$ alone is nearly identical between them. The objective value does not rank them correctly, because the global optimum of a Frobenius-orthogonal fit is not the translation map when the isometry assumption fails by the ~15% seen here. Adding 1000 gold pairs — about 30 minutes of a bilingual speaker's time — lifts the pair to the 40%+ range *and* makes the runs separable. That cost asymmetry, not the theory, is why "fully unsupervised" remains partially-solved rather than solved: the last increment of supervision removed is the one that carried the certificate.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*