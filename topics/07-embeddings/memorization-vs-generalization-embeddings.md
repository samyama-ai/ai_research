---
id: 07-embeddings/memorization-vs-generalization-embeddings
title: "Memorization versus Generalization in Retrieval Embedding Spaces"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memorization versus Generalization in Retrieval Embedding Spaces

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/memorization-vs-generalization-embeddings` · **Status:** open

## 1. Problem Statement

A dense retriever maps queries and documents into a shared vector space and ranks by inner product. When it returns the right document, two mechanisms could be responsible:

- **Memorization** — the training set contained this query–document pair (or a near-duplicate), and the encoder stored the association as a point-specific direction in the space.
- **Generalization** — the encoder learned a relevance function that transfers to pairs it never saw.

The problem: **separate the two, per test instance, for a fixed trained encoder.**

Three variants, of very different difficulty:

- **Measurement variant.** Given encoders $f, g$, training corpus $\mathcal{D}$, and a test query $q$, output a scalar $\mathrm{mem}(q) \in [0,1]$ that is the causal contribution of $q$'s training neighbors to its retrieval success. Blocked mainly on cost and on what counts as a "neighbor".
- **Method variant.** Train an encoder whose out-of-distribution retrieval quality is not predicted by train–test overlap. Empirically open.
- **Theory variant.** Given a corpus of $N$ documents and a relevance relation $R$, bound the embedding dimension $d$ needed to represent $R$ exactly, and characterize when a low-$d$ encoder must fall back on instance memorization to fit the training relation. Partly answered (§4) and partly open.

A solution to the measurement variant would let one state, for a benchmark score, what fraction of it survives if every training near-duplicate is deleted.

## 2. Formal Setting

Encoders $f_\theta: \mathcal{Q} \to \mathbb{R}^d$, $g_\theta: \mathcal{P} \to \mathbb{R}^d$, score $s_\theta(q,p) = \langle f_\theta(q), g_\theta(p)\rangle$. Training set $S = \{(q_i, p_i^+, \{p_{ij}^-\})\}_{i=1}^n$, InfoNCE loss

$$\mathcal{L}(\theta) = -\sum_{i=1}^{n} \log \frac{e^{s_\theta(q_i,p_i^+)/\tau}}{e^{s_\theta(q_i,p_i^+)/\tau} + \sum_j e^{s_\theta(q_i,p_{ij}^-)/\tau}}.$$

**Retrieval utility.** $U(q, p^\star; \theta) = \mathbf{1}[\,p^\star \in \mathrm{top}\text{-}k_\theta(q, \mathcal{C})\,]$ over corpus $\mathcal{C}$ of size $N$. Measured as recall@$k$ or nDCG@10 on a fixed corpus — not on a re-ranked candidate pool, which changes the quantity.

**Memorization score (Feldman-style, counterfactual).** For a training index $i$,

$$\mathrm{mem}(i) = \Pr_{\theta \sim A(S)}\big[U_i = 1\big] - \Pr_{\theta \sim A(S \setminus i)}\big[U_i = 1\big],$$

with $A$ the (stochastic) training algorithm. **As measured:** subsample $M$ training runs on random $\rho$-fractions of $S$ and take the difference of empirical means between runs that included $i$ and runs that did not (Feldman & Zhang, NeurIPS 2020, used $M \approx 2000$, $\rho = 0.7$ for image classifiers). For a test query $q$ and training index $i$, the analogous **influence** $\mathrm{infl}(i, q)$ is the same difference computed on $q$'s utility.

**Overlap slicing (the cheap proxy).** Partition the test set by $\mathrm{sim}(q, S)$:

$$\mathrm{sim}(q, S) = \max_{i} \; \mathrm{J}(q, q_i), \qquad \mathrm{J} = \text{token-level Jaccard or } \langle f_\theta(q), f_\theta(q_i)\rangle .$$

Slices: exact question overlap, answer-only overlap, no overlap (Lewis et al., EACL 2021). **Reported gap** $\Delta = U_{\text{overlap}} - U_{\text{no-overlap}}$.

**Capacity.** For a binary relevance matrix $R \in \{0,1\}^{|\mathcal{Q}| \times N}$, the minimum $d$ admitting an exact inner-product decomposition with a per-query threshold is governed by the **sign rank** of $2R - \mathbf{1}$.

**Assumptions, and which are violated.**
- *Independent train/test draws* — violated. Web-scale pretraining corpora contain BEIR and MS MARCO text; contamination is the default state, not the exception.
- *"Near-duplicate" is well defined* — violated. Lexical and semantic duplicate detectors disagree on the majority of borderline pairs; there is no agreed threshold.
- *$A$ is stable enough for $M$ subsample runs to estimate $\mathrm{mem}$* — questionable. Contrastive training with in-batch negatives makes the loss for point $i$ depend on batch composition, so removing $i$ perturbs other points' gradients too; the leave-one-out counterfactual is not clean.
- *Fixed corpus $\mathcal{C}$* — violated in practice; corpus size changes recall non-monotonically.

## 3. State of the Art

**Empirical SOTA (established).** Overlap slicing on open-domain QA. Lewis, Stenetorp & Riedel (EACL 2021) show that on Open-NQ, a large majority of test questions have their answer present in some training answer, and roughly a third are near-paraphrases of a training question; DPR's exact-match accuracy on the question-overlap slice is roughly **triple** its accuracy on the no-overlap slice (≈69% vs ≈25% at their scale, single retriever, NQ-only). This is a slice comparison, not a counterfactual — *established as a correlation, unablated as a cause.*

**Established stress test.** Sciavolino et al. (EMNLP 2021, EntityQuestions) show DPR trained on NQ loses badly to BM25 on simple entity-centric questions, with per-relation gaps exceeding 50 points top-20 recall. Ablated: fine-tuning on the failing relations recovers much of the gap, implicating memorized entity coverage rather than a missing relational skill.

**Capacity SOTA (theory, recent).** Weller et al., *On the Theoretical Limitations of Embedding-Based Retrieval* (2025), tie representable top-$k$ sets to the sign rank of the qrel matrix and build **LIMIT**, a 46-document instance where strong commercial embedders score under ~20% recall@100. *Established as a construction*; the claim that natural corpora sit near this bound is **claimed but unablated**.

**Scaling as a partial answer (benchmark number only).** Ni et al., GTR (EMNLP 2022) report that scaling the dual encoder while holding $d = 768$ improves BEIR out-of-domain averages. This is a leaderboard delta; no accompanying contamination audit, so the improvement cannot be attributed to generalization rather than to broader pretraining coverage.

**Leakage direction.** Song & Raghunathan (CCS 2020) and Morris et al. (vec2text, EMNLP 2023) show embeddings retain enough of the input to reconstruct much of it — evidence of high information retention, but *not* evidence about which retrieval decisions depend on it.

## 4. What Is Known

- **Overlap explains a large share of QA retrieval scores.** ≈3× accuracy gap between overlap and no-overlap slices on Open-NQ; measured at DPR-base scale, single dataset (Lewis et al., 2021).
- **Lexical baselines beat dense retrievers out of domain.** BEIR (Thakur et al., NeurIPS 2021 D&B): BM25 outperforms most dense models on the majority of its 18 datasets in zero-shot nDCG@10, at the scale of BERT-base bi-encoders.
- **Memorization is not overfitting.** Tirumala et al. (NeurIPS 2022) show LM memorization rises before validation loss degrades; Carlini et al. (ICLR 2023) show extractable memorization grows log-linearly in model size, data duplication, and prompt context length. Both at LM scale (125M–12B), not measured for retrieval encoders.
- **Long-tail memorization can be near-optimal.** Feldman (STOC 2020): under a long-tailed label prior, fitting singleton examples is necessary to achieve close-to-optimal generalization error. This makes "memorization is bad" false as a blanket claim.
- **Dimension bites.** Reimers & Gurevych (ACL 2021) show dense retrieval quality degrades as index size grows at fixed $d$; the sign-rank argument (2025) gives a matching worst-case reason.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted definition of a train–test "duplicate" for retrieval. Lexical overlap, embedding cosine from the model under test (circular), and LLM-judged paraphrase all give different partitions; the reported $\Delta$ is a function of the chosen threshold, and nobody publishes the threshold sweep.
- **Empirically open.** No published leave-$k$-out counterfactual memorization estimate for a modern retrieval encoder. Feldman–Zhang influence estimation has been run for CIFAR/ImageNet classifiers, never for a contrastively trained dual encoder at ≥1B tokens of training pairs. The compute is affordable at the 100M-parameter scale (§8); it has not been spent.
- **Theoretically open.** Whether the sign-rank lower bound is *attained* by natural corpora, or whether real relevance matrices have low sign rank and the observed failures are optimization artifacts. Also open: whether contrastive training with in-batch negatives admits a stable influence functional at all.
- **Open, mechanistic.** Whether memorized pairs occupy identifiable subspaces (a "lookup table" direction) separable from a general relevance metric.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement via pretraining contamination.** The counterfactual $S \setminus i$ removes a fine-tuning pair but not the pretraining exposure to the same text. The measured $\mathrm{mem}(i)$ is therefore a lower bound with unknown slack, and the slack is largest exactly for popular entities — the cases that dominate benchmark scores.
2. **Non-identifiability of the mechanism.** A high-scoring $\langle f(q), g(p)\rangle$ is a single scalar. Memorized association and learned semantic similarity produce the same scalar; the space has no privileged basis that separates them, and rotations of $(f,g)$ leave all scores invariant.
3. **In-batch negatives break the leave-one-out counterfactual.** Removing example $i$ changes the negative distribution for every example sharing its batch, so $A(S \setminus i)$ differs from $A(S)$ in ways unrelated to $i$'s content. Feldman–Zhang subsampling assumes per-example losses; contrastive training violates it.

Cost is secondary but real: $M = 1000$ retrainings of a 110M-parameter encoder on 500k pairs is roughly $10^3 \times 10^2$ GPU-hours, order 100k A100-hours.

## 7. Current Research (as of 2026)

- **Capacity-driven benchmark design.** LIMIT-style adversarial corpora built from combinatorics rather than topic shift (Google DeepMind, Weller et al.) — the most active line, since it makes the failure mode reproducible in 46 documents.
- **Contamination auditing for embedding leaderboards.** MTEB/MMTEB maintainers (Muennighoff et al., EACL 2023; Enevoldsen et al., ICLR 2025) have moved toward held-out and multilingual splits partly for this reason; systematic per-model contamination reports are still not standard. *(frontier — verify)*
- **Hybrid and lexically-aware dense retrieval** (SPAR-style lexical distillation, Chen et al., Findings of ACL 2022) as an explicit patch for the entity-memorization failure.
- **Influence estimation at scale** for LLMs (gradient-similarity approximations) being ported to retrieval encoders. *(frontier — verify)*
- **Membership inference as a memorization proxy** for embedding models, following Song & Raghunathan; complicated by Duan et al. (COLM 2024) showing MIA on LLMs is near-chance under proper controls.

## 8. Concrete Next Experiment

**Question:** what fraction of a dense retriever's benchmark recall is caused by specific training pairs?

**Scale.** Encoder: BERT-base bi-encoder, $d = 768$. Training set $S$: 500k MS MARCO pairs. $M = 512$ independent runs, each on an i.i.d. 70% subsample of $S$ (Feldman–Zhang protocol). Each run ≈4 A100-hours → ≈2k A100-hours total, ≈$4k at spot pricing. Evaluate every run on a fixed 5k-query test set over the full 8.8M-passage corpus.

**Estimator.** For each test query $q$ and each training index $i$, $\widehat{\mathrm{infl}}(i,q) = \bar{U}_q^{(i \in)} - \bar{U}_q^{(i \notin)}$. Define $q$ as **memorization-driven** if $\max_i \widehat{\mathrm{infl}}(i,q) > 0.25$ with the sign test significant at $p < 0.01$ over the 512 runs.

**Control arms (both required).**
1. *Batch-composition control.* Repeat with a loss using only fixed pre-mined negatives (no in-batch negatives), removing obstruction 3. If the memorization-driven fraction moves by more than 5 points between arms, the InfoNCE estimator is measuring batch effects, not memorization.
2. *Contamination control.* Repeat with an encoder pretrained on a corpus filtered against the MS MARCO test passages (e.g. a from-scratch 110M model), isolating fine-tuning memorization from pretraining exposure.

**The deciding number.** $\phi$ = fraction of test queries that are memorization-driven, reported alongside recall@100. If $\phi < 0.10$, benchmark scores are mostly generalization and the overlap-slice literature is overstating the effect. If $\phi > 0.40$, dense-retrieval leaderboards are largely measuring training-set coverage, and every zero-shot claim needs a contamination-adjusted number.

## 9. Key References

- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[Foundational]** Vitaly Feldman, Chiyuan Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS, 2020. — arXiv:2008.03703
- **[Foundational]** Vladimir Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP, 2020. — arXiv:2004.04906
- **[SOTA / diagnostic]** Patrick Lewis, Pontus Stenetorp, Sebastian Riedel. *Question and Answer Test-Train Overlap in Open-Domain Question Answering Datasets.* EACL, 2021. — arXiv:2008.02637
- **[SOTA / diagnostic]** Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, Danqi Chen. *Simple Entity-Centric Questions Challenge Dense Retrievers.* EMNLP, 2021. — arXiv:2109.08535
- **[SOTA / theory]** Orion Weller et al. *On the Theoretical Limitations of Embedding-Based Retrieval.* Preprint, 2025. — arXiv:2508.21038
- **[SOTA]** Jianmo Ni et al. *Large Dual Encoders Are Generalizable Retrievers.* EMNLP, 2022. — arXiv:2112.07899
- **[Benchmark]** Nandan Thakur et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2104.08663
- **[Benchmark]** Niklas Muennighoff et al. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316
- **[Memorization]** Nicholas Carlini et al. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Memorization]** Kushal Tirumala et al. *Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models.* NeurIPS, 2022. — arXiv:2205.10770
- **[Leakage]** Congzheng Song, Ananth Raghunathan. *Information Leakage in Embedding Models.* ACM CCS, 2020. — arXiv:2004.00053
- **[Leakage]** John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, Alexander M. Rush. *Text Embeddings Reveal (Almost) As Much As Text.* EMNLP, 2023. — arXiv:2310.06816
- **[Capacity]** Nils Reimers, Iryna Gurevych. *The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.* ACL, 2021.
- **[Caution]** Michael Duan et al. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024.

## 10. Worked Example

Take a single Open-NQ test question: *"who played the mother in the sound of music"*. NQ's training set contains several questions about the same film and cast. A DPR-style retriever returns the right Wikipedia passage at rank 1.

**Slice attribution.** Token-Jaccard against the nearest training question, *"who played the father in the sound of music"*, is $8/10 = 0.80$. At a threshold of $0.6$ this query lands in the *overlap* slice; at $0.85$ it lands in *no overlap*. Nothing about the model changed — only the threshold. Under the Lewis et al. numbers, the overlap slice scores ≈69% EM and the no-overlap slice ≈25%; the same query therefore inherits a predicted accuracy of either 69% or 25% depending on a hyperparameter of the *measurement*.

**Counterfactual attribution.** Run the §8 estimator with $M = 512$. Suppose the "father" pair appears in 358 of the subsampled training sets. If the query is retrieved correctly in 341/358 runs that include it and 296/154 → 0.62 of runs that exclude it, then

$$\widehat{\mathrm{infl}} = 0.953 - 0.620 = 0.333 \pm 0.041 \;(\text{s.e.}),$$

above the 0.25 threshold: memorization-driven. But the *same* estimator run under the no-in-batch-negatives control arm can move this by ±0.1 purely because removing one pair reshapes 127 other batches.

**The obstruction, made visible.** The slice method gives an answer that is a function of a threshold nobody agrees on (methodologically blocked). The counterfactual method gives a causal answer, but its error bar is comparable to the effect size unless the batch confound is controlled, and it still cannot see the pretraining corpus, where "The Sound of Music" cast appears thousands of times. Both routes return a number; neither returns the number the field reports as "zero-shot retrieval quality".

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*