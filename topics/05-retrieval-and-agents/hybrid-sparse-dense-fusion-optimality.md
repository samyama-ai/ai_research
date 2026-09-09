---
id: 05-retrieval-and-agents/hybrid-sparse-dense-fusion-optimality
title: "Hybrid Sparse-Dense Fusion Optimality"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hybrid Sparse-Dense Fusion Optimality

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/hybrid-sparse-dense-fusion-optimality` · **Status:** open

## 1. Problem Statement

Given a lexical retriever (BM25, SPLADE, or another sparse scorer) and a dense bi-encoder, both run over the same corpus, produce a single ranking that is at least as good as either and as close as possible to the best achievable ranking from those two signals.

Three variants, routinely conflated:

- **Method.** Find a fusion function $f$ mapping the two systems' outputs to one ranking that maximizes expected nDCG@10. In practice this reduces to a choice between rank-based fusion (RRF), score-based convex combination, and a learned reranker over the union.
- **Theory.** Characterize the regret of a restricted fusion class against the Bayes-optimal ranker under a stated generative model. Specifically: how much is lost by discarding scores and keeping only ranks, and when is a *fixed* mixing weight $\alpha$ optimal versus a per-query $\alpha(q)$?
- **Measurement.** Decide whether observed hybrid gains are real retrieval improvements or artifacts of score normalization, candidate-list depth, and incomplete relevance judgments.

Solving it means: a fusion rule with a stated regret bound, plus an evaluation showing the bound is not slack on TREC-DL and BEIR under complete-enough judgments.

## 2. Formal Setting

Corpus $\mathcal{D}$, query $q \sim P_Q$, binary or graded relevance $y(q,d) \in \{0,1,2,3\}$. Two scorers:

$$s_L(q,d) \in \mathbb{R} \quad \text{(lexical)}, \qquad s_D(q,d) = \langle \phi(q), \psi(d)\rangle \in \mathbb{R} \quad \text{(dense, } \phi,\psi: \to \mathbb{R}^m\text{)}.$$

Each returns a top-$k$ list, $R_L^k(q)$ and $R_D^k(q)$. **Measured as:** $s_L$ is the raw Lucene/Anserini BM25 score with $k_1=0.9, b=0.4$; $s_D$ is the raw inner product or cosine from the ANN index at recall target $\ge 0.99$ against exact search. $k$ is the *retrieval depth*, typically 100 or 1000 — a system parameter, not a property of the query.

Fusion candidates over the union $U(q) = R_L^k \cup R_D^k$:

$$f_{\text{cc}}(q,d) = \alpha\,\tilde{s}_D(q,d) + (1-\alpha)\,\tilde{s}_L(q,d), \qquad f_{\text{rrf}}(q,d) = \sum_{i\in\{L,D\}} \frac{1}{\eta + r_i(q,d)}$$

with $r_i$ the rank in list $i$ ($r_i = \infty$ if absent, contributing 0), $\eta = 60$ by convention. $\tilde{s}$ is a normalizer, almost always min–max over the retrieved list:

$$\tilde{s}_i(q,d) = \frac{s_i(q,d) - \min_{d'\in R_i^k(q)} s_i(q,d')}{\max_{d'} s_i(q,d') - \min_{d'} s_i(q,d')}.$$

Objective: $\alpha^\star = \arg\max_\alpha \mathbb{E}_{q}\!\left[\mathrm{nDCG}@10(f_{\text{cc}}^\alpha, q)\right]$, estimated on a query sample of size $n$ (TREC-DL: $n=43$–$54$; BEIR test sets: $n=50$–$10{,}000$).

**Assumptions, and their status:**

1. *Scores are comparable after normalization.* Violated. Min–max makes $\tilde{s}$ a function of $k$: the same document pair can invert when $k$ goes 100 → 1000 (Section 10).
2. *A single $\alpha$ transfers across domains.* Violated. $\alpha$ tuned on MS MARCO is not optimal on BEIR's out-of-domain sets, where the lexical/dense skill gap swings by tens of nDCG points per dataset.
3. *Judgments are complete over $U(q)$.* Violated by construction. TREC pools were built from runs that did not include modern hybrid systems; unjudged-as-nonrelevant penalizes exactly the documents fusion surfaces.
4. *The two scorers' errors are conditionally independent given relevance.* Assumed by every additive fusion rule, never tested. SPLADE and dense bi-encoders share pretraining corpora and MS MARCO training triples.

## 3. State of the Art

**Theory SOTA.** Bruch, Gai, and Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval* (ACM TOIS, 2023; arXiv:2210.11934) is the only substantial formal treatment. Established there: RRF is not monotone in the underlying scores and is sensitive to $\eta$; convex combination with a theoretically-motivated normalization (TM2C2 — min–max with the theoretical BM25 minimum) dominates RRF and transfers zero-shot better than tuned min–max. Luan, Eisenstein, Toutanova, and Collins, *Sparse, Dense, and Attentional Representations for Text Retrieval* (TACL 2021), gives the complementary capacity result: fixed-dimension dense encoders need $m$ growing with document length to match sparse exact-match fidelity — a reason hybrids exist, not a fusion rule.

**Empirical SOTA.** Tuned convex combination over BM25 + a strong dense retriever, or over SPLADE + dense; in production, RRF (Elasticsearch, Vespa, Weaviate all ship it as default). BGE-M3 (Chen et al., ACL Findings 2024) trains one model to emit sparse, dense, and multi-vector scores and fuses them internally, sidestepping cross-system calibration.

**Claimed but unablated.** That RRF's parameter-free character makes it robust — RRF's $\eta=60$ comes from Cormack, Clarke, and Buettcher (SIGIR 2009) tuned on TREC-3 through TREC-5 ad hoc data; it is a tuned constant carried forward for 17 years without re-tuning. **Benchmark-number-only:** most reported hybrid gains on BEIR are single-configuration nDCG@10 deltas with no depth sweep, no normalization ablation, and no per-dataset $\alpha$ reported. Whether the gain survives holding $|U(q)|$ fixed is, for most published hybrids, not shown.

## 4. What Is Known

- **Hybrid > either component, at MS MARCO/NQ scale.** Ma et al., *A Replication Study of Dense Passage Retriever* (arXiv:2104.05740, 2021): on Natural Questions (3,610 test questions), DPR top-20 accuracy ≈ 79%, BM25 ≈ 63%, linear-combination hybrid ≈ 82–83%. The gain is a few points, not a regime change.
- **BM25 is not dominated out of domain.** Thakur et al., BEIR (NeurIPS Datasets & Benchmarks, 2021), 18 datasets: BM25 average nDCG@10 ≈ 0.43, beating most dense retrievers of that era zero-shot. Per-dataset spread is the whole story — BM25 ≈ 0.44 on Touché-2020 where dense models fell near 0.20, and far below dense on Quora.
- **RRF is beaten by tuned score fusion.** Bruch et al. (TOIS 2023) report convex combination with proper normalization above RRF on MS MARCO Passage (dev, 6,980 queries) and across BEIR subsets, with RRF's advantage confined to the case where scores are untrusted.
- **Lexical signal is not recoverable by scale alone.** Luan et al. (TACL 2021) give the dimension lower bound; Weller et al., *On the Theoretical Limitations of Embedding-Based Retrieval* (2025), construct the LIMIT dataset where state-of-the-art embedders score under 20 recall@100 on trivially easy queries while BM25 solves them — a communication-complexity argument, not an optimization failure.
- **Cost is asymmetric.** Learned sparse retrieval over inverted indexes is now competitive in latency (Bruch, Nardini, Rulli, Venturini, *Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations*, SIGIR 2024 — Seismic), so "hybrid is too slow" is no longer the binding constraint.

## 5. What Is Not Known

- **Theoretically open.** No regret bound for any fusion class against the Bayes-optimal ranker under a realistic model of correlated scorer errors. The independence-of-errors assumption underlying additive fusion has no proof and no counterexample construction. Also open: the exact loss from rank-only fusion — how much nDCG is discarded by throwing away score margins.
- **Empirically open.** The size of the per-query adaptivity gap. Nobody has published $\mathbb{E}_q[\mathrm{nDCG}@10]$ under an oracle per-query $\alpha(q)$ against the best fixed $\alpha$ at BEIR scale. The experiment is a grid sweep and costs a few thousand GPU-hours; it has not been run and reported as a headline number.
- **Methodologically blocked.** Whether hybrid gains on TREC/BEIR are real. Fusion changes $U(q)$, so it surfaces documents outside the judgment pool. Under unjudged-as-nonrelevant, a hybrid that retrieves better unjudged documents scores *worse*. No published hybrid comparison uses condensed-list or bpref-style measures that are robust to this.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names**, compounded by **non-identifiability of $\alpha$**.

Min–max normalization makes $\tilde{s}$ depend on the retrieval depth $k$ and on the query's score distribution, both of which carry no relevance information. So the fused ranking is a function of a system knob. Two papers reporting "convex combination, $\alpha=0.5$" at $k=100$ and $k=1000$ are running different rankers. Meanwhile $\alpha$ and the normalizer are not separately identifiable from nDCG alone: a change in $\alpha$ and a change in the normalization scale produce the same ranking, so a tuned $\alpha$ is not a transferable quantity and cannot be compared across papers.

On top of that, the judgment pools were built before hybrid systems existed. The measurement penalizes the behavior being tested.

## 7. Current Research (as of 2026)

- **Unified single-model fusion.** BGE-M3-style models that emit sparse and dense scores from one backbone, making the scores calibrated by construction (BAAI). This dissolves the cross-system calibration problem but not the weighting problem.
- **Theory of fusion functions.** Bruch and collaborators (Pinecone, ISTI-CNR Pisa) continue on normalization theory and on sparse–dense MIPS as one index rather than two lists. *(frontier — verify)*
- **Query routing instead of fusion.** Predict per query whether lexical or dense should dominate, then skip the other. Sits between fusion and cascades; results so far are single-benchmark. *(frontier — verify)*
- **Rerankers as the fusion function.** In agentic RAG stacks, a cross-encoder or LLM reranker over $U(q)$ makes first-stage fusion weight nearly irrelevant, moving the question to recall of the union rather than order within it. This is the dominant production answer and it makes the theory question less urgent, not answered.

## 8. Concrete Next Experiment

**Question:** how much nDCG is left on the table by using one global $\alpha$ instead of a per-query $\alpha(q)$?

- **Scale.** 13 BEIR datasets with public judgments, plus TREC-DL 2019+2020 (97 queries with deep graded judgments). Retrievers: BM25 (Anserini, $k_1=0.9,b=0.4$) and a single fixed dense model (e.g. GTR-base or bge-base). Retrieval depth fixed at $k=1000$ for every arm — this is the control that most published comparisons omit. Sweep $\alpha \in \{0, 0.05, \dots, 1\}$, 21 points.
- **Control arms.** (a) Best single retriever per dataset; (b) RRF with $\eta=60$; (c) best *fixed* $\alpha$ chosen on MS MARCO dev and frozen; (d) best fixed $\alpha$ chosen per dataset (in-domain oracle).
- **Treatment.** Oracle per-query $\alpha(q)$ — pick the $\alpha$ from the grid maximizing that query's nDCG@10. This is an upper bound, not a method.
- **Deciding number.** $\Delta = \mathrm{nDCG}@10(\text{oracle per-query } \alpha) - \mathrm{nDCG}@10(\text{best fixed } \alpha \text{ per dataset})$, averaged over datasets. If $\Delta < 1.0$ point, per-query fusion is a dead end and the field should stop tuning $\alpha$ and put the compute into the reranker. If $\Delta > 3.0$ points, an $\alpha$-predictor is worth building and the fixed-$\alpha$ literature is measuring the wrong object.
- **Required robustness check.** Recompute everything with unjudged documents removed from the ranking (condensed lists). If the sign of the RRF-vs-convex-combination comparison flips under condensation, the methodological block in Section 5 is confirmed and no fusion conclusion from BEIR is safe.

## 9. Key References

- **[Foundational]** Fox, E. A., Shaw, J. A. *Combination of Multiple Searches.* TREC-2, 1994. — CombSUM/CombMNZ, the origin of score fusion.
- **[Foundational]** Cormack, G. V., Clarke, C. L. A., Buettcher, S. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009.
- **[SOTA / Theory]** Bruch, S., Gai, S., Ingber, A. *An Analysis of Fusion Functions for Hybrid Retrieval.* ACM Transactions on Information Systems, 2023. — arXiv:2210.11934
- **[Foundational]** Luan, Y., Eisenstein, J., Toutanova, K., Collins, M. *Sparse, Dense, and Attentional Representations for Text Retrieval.* TACL 9, 2021.
- **[Benchmark]** Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks, 2021.
- **[Empirical]** Ma, X., Sun, K., Pradeep, R., Lin, J. *A Replication Study of Dense Passage Retriever.* arXiv:2104.05740, 2021.
- **[SOTA]** Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z. *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* Findings of ACL 2024.
- **[Systems]** Bruch, S., Nardini, F. M., Rulli, C., Venturini, R. *Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations.* SIGIR 2024.
- **[Limits]** Weller, O., Boratko, M., Naim, I., Lee, J. *On the Theoretical Limitations of Embedding-Based Retrieval.* 2025. (LIMIT dataset.)
- **[Survey]** Bruch, S. *Foundations of Vector Retrieval.* Springer, 2024.
- **[Hybrid, early]** Gao, L., Dai, Z., Chen, Z., Fan, Z., Van Durme, B., Callan, J. *Complement Lexical Retrieval Model with Semantic Residual Embeddings (CLEAR).* ECIR 2021.

## 10. Worked Example

Two documents, $X$ and $Y$, both retrieved by both systems. Raw scores:

| doc | dense | BM25 |
|---|---|---|
| $X$ | 0.81 | 8.0 |
| $Y$ | 0.80 | 9.0 |

Both systems' list maxima: dense 0.82, BM25 18.0. The minima depend on depth:

| depth $k$ | dense min | BM25 min |
|---|---|---|
| 100 | 0.78 | 4.0 |
| 1000 | 0.30 | 1.0 |

Min–max normalize and fuse with $\alpha = 0.5$.

At $k=100$: $\tilde{s}_D(X) = (0.81-0.78)/0.04 = 0.750$, $\tilde{s}_D(Y) = 0.500$; $\tilde{s}_L(X) = 4/14 = 0.286$, $\tilde{s}_L(Y) = 5/14 = 0.357$.
$$f(X) = 0.518, \quad f(Y) = 0.429 \Rightarrow X \succ Y.$$

At $k=1000$: $\tilde{s}_D(X) = 0.51/0.52 = 0.981$, $\tilde{s}_D(Y) = 0.962$; $\tilde{s}_L(X) = 7/17 = 0.412$, $\tilde{s}_L(Y) = 8/17 = 0.471$.
$$f(X) = 0.697, \quad f(Y) = 0.717 \Rightarrow Y \succ X.$$

Nothing about the query, the documents, or the retrievers changed. Only the number of candidates fetched changed, and the top-2 order inverted. A paper reporting "$\alpha=0.5$ convex combination" has not specified its ranker.

RRF is immune to this — ranks are unchanged, so with $\eta=60$, $X$ scores $1/62 + 1/63 = 0.03200$ and $Y$ scores $1/63 + 1/62 = 0.03200$, an exact tie at both depths. But that immunity is purchased by discarding the one piece of evidence that mattered: at $k=100$ the dense scores were tightly clustered (0.78–0.82), meaning the dense model had almost no confidence in its own ordering, while BM25 separated $X$ and $Y$ by a full point. RRF cannot see either fact.

That is the obstruction in one instance. Score fusion is depth-dependent and therefore not well defined; rank fusion is well defined and throws away the margins that would decide the case. Neither is known to be near-optimal, and the benchmark that would adjudicate them has judgment pools that predate both.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*