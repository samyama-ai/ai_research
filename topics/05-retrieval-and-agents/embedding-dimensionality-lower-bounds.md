---
id: 05-retrieval-and-agents/embedding-dimensionality-lower-bounds
title: "Embedding Dimensionality Lower Bounds for Retrieval Tasks"
topic: 05-retrieval-and-agents
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Embedding Dimensionality Lower Bounds for Retrieval Tasks

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/embedding-dimensionality-lower-bounds` · **Status:** partially-solved

## 1. Problem Statement

A single-vector retriever maps queries and documents into $\mathbb{R}^k$ and ranks by inner product. Fix a corpus of $n$ documents and a set of queries with a known relevance relation. **How large must $k$ be for *some* assignment of vectors to reproduce that relation exactly under top-$k$ ranking?**

Three variants, routinely conflated:

- **Theory.** Given a binary relevance matrix $A$, compute or bound the minimum embedding dimension $\mathrm{rop}(A)$ that admits an order-preserving inner-product representation. This is a sign-rank-type quantity.
- **Measurement.** Given a *real* retrieval task (BEIR, MS MARCO, an agent's tool corpus), estimate the required $k$. Blocked by the fact that graded, incomplete, partly wrong qrels do not define a matrix whose sign-rank is meaningful.
- **Method.** Given a fixed budget $k$ and a trainable encoder, how close to the information-theoretic optimum does gradient descent on contrastive loss actually get? The gap between "a vector configuration exists" and "an encoder learns it" is unquantified.

Solving it means: a function $k^\*(n, r, \epsilon)$ — dimension needed for $n$ documents, $r$ relevant per query, recall $\ge 1-\epsilon$ — with matching upper and lower bounds, plus evidence that trained encoders track it within a stated factor.

## 2. Formal Setting

Let $Q$ be queries, $D$ documents, $|D| = n$, and $A \in \{0,1\}^{|Q| \times n}$ the relevance matrix, $A_{ij}=1$ iff $d_j$ is relevant to $q_i$. An embedding is $\phi: Q \to \mathbb{R}^k$, $\psi: D \to \mathbb{R}^k$ with score $s_{ij} = \langle \phi(q_i), \psi(d_j)\rangle$.

**Order-preserving representation.** $(\phi,\psi)$ *realizes* $A$ if for every $i$ there is a threshold $\tau_i$ with
$$s_{ij} > \tau_i \iff A_{ij} = 1 .$$
The minimum such $k$ is the **row-wise order-preserving rank** $\mathrm{rop}(A)$. Absorbing $\tau_i$ into one extra coordinate gives
$$\mathrm{sign\text{-}rank}(2A - \mathbf{1}) - 1 \;\le\; \mathrm{rop}(A) \;\le\; \mathrm{sign\text{-}rank}(2A-\mathbf{1}),$$
where $\mathrm{sign\text{-}rank}(M) = \min\{\mathrm{rank}(N) : \mathrm{sign}(N_{ij}) = M_{ij}\}$.

**Measured quantities.**
- *Critical $n$*, $n_c(k)$: the largest corpus size for which free optimization of $\phi,\psi$ (direct SGD on the $|Q|\times n$ Gram matrix, no encoder) attains 100% recall of all relevant sets at dimension $k$. Measured by binary search over $n$, 10+ random seeds, declaring success only if every query's relevant set occupies the top ranks.
- *Recall@$m$*: $\frac{1}{|Q|}\sum_i \frac{|\text{top-}m(s_i) \cap \{j: A_{ij}=1\}|}{\sum_j A_{ij}}$ — the deployed metric, strictly weaker than exact realization.
- *Margin*: $\gamma = \min_i (\min_{j: A_{ij}=1} s_{ij} - \max_{j: A_{ij}=0} s_{ij})$ after unit-normalizing all vectors. Finite-precision retrieval needs $\gamma \gtrsim 2^{-b}$ for $b$-bit scores.

**Assumptions, and where they break.**
1. *Binary, complete relevance.* Violated everywhere: BEIR/MS MARCO qrels are sparse and graded; unjudged relevant documents flip $A_{ij}$ from 1 to 0 and can change sign-rank arbitrarily.
2. *Exact realization is the target.* Violated: production systems need recall@100, not exact order. Approximate realization admits much smaller $k$ and has weaker known lower bounds.
3. *Free embeddings are reachable.* Violated: encoders are constrained maps of text, trained on finite data. $n_c(k)$ from free optimization is an **upper bound on achievable capability** — real encoders can only do worse.
4. *Query-agnostic document vectors.* This is the actual restriction being bounded. Multi-vector (ColBERT) and cross-encoder scorers are outside the model and are not bounded by $\mathrm{rop}(A)$.
5. *Unbounded precision.* Violated by int8/binary quantization; bounded-precision sign-rank is a different, larger quantity.

## 3. State of the Art

**Theory (established).**
- Sign-rank of a random $n\times n$ sign matrix is $\Theta(n)$ (Alon, Frankl, Rödl 1985) — most relevance matrices are not embeddable at any practical $k$.
- Forster's theorem (2002): $\mathrm{sign\text{-}rank}(M) \ge \sqrt{mn}/\|M\|_2$ for $M \in \{\pm1\}^{m\times n}$. For an $n\times n$ Hadamard matrix this gives $\ge \sqrt{n}$ — the first explicit linear-algebraic lower bound usable on a constructed corpus.
- Ben-David, Eiron, Simon (JMLR 2002): almost all concept classes over $n$ points require Euclidean half-space dimension $\Omega(n)$, ruling out generic low-dimensional embedding of arbitrary query semantics.
- Larsen & Nelson (FOCS 2017): the Johnson–Lindenstrauss bound $k = \Theta(\epsilon^{-2}\log n)$ is optimal. This governs *distance preservation*, not *rank order*, and is the bound most often misapplied to retrieval.

**Empirical (established at small scale).** Weller, Boratko, Naim, Lee (2025), *On the Theoretical Limitations of Embedding-Based Retrieval*, connect retrieval to sign-rank, measure $n_c(k)$ by free optimization for $k \in \{4,\dots,32\}$, and build **LIMIT** — a corpus where every 2-subset of a small document set is the relevant set for some query. On LIMIT, strong MTEB-leading embedders score in the low single digits to under ~20% recall@100 despite the corpus being tiny (~46 core documents; 50k-document variant), while BM25 and multi-vector retrievers do far better. This is a *constructed* adversarial instance, and its transfer to natural corpora is unestablished.

**Claimed but unablated.** The extrapolation from measured $n_c(k)$ at $k \le 32$ to $k = 4096$ via polynomial fit predicts critical corpus sizes in the hundreds of millions. The fit is over a narrow range with a low-degree polynomial and no independent replication; treat the extrapolated numbers as illustrative, not as a bound.

**Benchmark-number-only results.** Vendor claims that truncating `text-embedding-3-large` from 3072 to 256 dimensions costs roughly 2–3 MTEB points (~64.6 → ~62.0) come from a single provider's evaluation with no released ablation of training-time versus truncation-time effects. MTEB averages also weight tasks that are not retrieval.

## 4. What Is Known

- **Rank ceiling.** If $A$ has rank $r$ over $\mathbb{R}$, then $\mathrm{rop}(A) \le r+1$. A corpus with block-structured topical relevance is cheap to embed; the expensive case is combinatorial relevance (arbitrary subsets), which is exactly what agentic retrieval over tool/document sets produces.
- **The all-2-subsets construction.** For $n$ documents and all $\binom{n}{2}$ possible relevant pairs, no fixed $k \ll n$ suffices; this is the LIMIT design and the reason a 46-document corpus defeats 4096-dimensional retrievers.
- **Measured $n_c(k)$** (free-embedding optimization, Weller et al. 2025, single-node GPU scale): monotone increasing in $k$, in the hundreds-to-low-thousands of documents for $k$ in the range 4–32. These are *best-case* numbers; no encoder was involved.
- **Matryoshka training** (Kusupati et al., NeurIPS 2022) shows nested-dimension supervision recovers most of the accuracy of full-width vectors at small $k$ — reported up to ~14× smaller embeddings at matched ImageNet-1K top-1 versus fixed-width baselines. This measures *method* efficiency, not the theoretical floor.
- **Multi-vector escapes the bound.** ColBERT-style late interaction (Khattab & Zaharia, SIGIR 2020) is not a single inner product and is not subject to $\mathrm{rop}(A)$; LIMIT results are consistent with this.

## 5. What Is Not Known

- **Theoretically open.** No non-trivial lower bound on $\mathrm{rop}(A)$ for the *approximate* objective (recall@$m \ge 1-\epsilon$ with $m \gg r$). All strong bounds are for exact sign realization. Also open: sign-rank of matrices drawn from realistic relevance-generating processes (topic models, power-law co-occurrence) — the random-matrix $\Theta(n)$ result is almost certainly pessimistic for these.
- **Empirically open.** $n_c(k)$ has never been measured for $k \ge 512$ at $n \ge 10^6$. The experiment is a large but ordinary optimization job; nobody has run it. Consequently every statement about "how many documents a 1536-dimensional index can serve" is extrapolation.
- **Methodologically blocked.** Estimating the required $k$ for a *natural* corpus. There is no ground-truth $A$ for MS MARCO: unjudged-but-relevant documents are pervasive, and sign-rank is not robust to entry flips. Until relevance is either complete or the bound is restated for noisy $A$, the question "what dimension does web search need?" is not well posed.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the target matrix combined with non-robustness of the bound**. Sign-rank can jump by $\Omega(n)$ under a single entry flip, while real qrels have flip rates plausibly above 20% (unjudged relevants in pooled collections). So the quantity the theory bounds cannot be estimated from the data practitioners have.

Second obstruction: **the evaluation does not measure what it names**. BEIR/MTEB scores confound representational capacity with training-data coverage, tokenizer, and instruction tuning. A model scoring 5% on LIMIT and 60% on BEIR tells you nothing about which effect dominates on your corpus.

Third: **computing $\mathrm{rop}(A)$ is intractable.** Sign-rank is $\exists\mathbb{R}$-complete in general; the practical proxy is stochastic search, which yields only upper bounds on $n_c$ and is confounded by optimizer failure — a negative result cannot distinguish "no configuration exists" from "SGD did not find it."

## 7. Current Research (as of 2026)

- **Google DeepMind** (Weller, Boratko, Naim, Lee and collaborators): sign-rank framing, LIMIT, and follow-on work extending the construction to instruction-following and reasoning-style retrieval *(frontier — verify)*.
- **Multi-vector and late-interaction efficiency** (Stanford/UWaterloo lineage: ColBERTv2, PLAID): treating the dimensionality bound as an argument for late interaction rather than wider single vectors.
- **Matryoshka and adaptive-width representations** (Kusupati et al. and successors, now standard in commercial APIs): reduces deployed $k$ without addressing the floor.
- **Sparse-dense hybrids**: BM25's strength on LIMIT has revived interest in lexical fallback as capacity insurance rather than a legacy baseline.
- **Theory side**: sign-rank versus VC dimension (Alon, Moran, Yehudayoff 2016) separates learnability from embeddability; the retrieval-specific corollaries are unwritten.

## 8. Concrete Next Experiment

**Question:** does the trained-encoder dimension requirement track the free-embedding floor by a constant factor, or a growing one?

**Scale.** Fix $k \in \{64, 256, 1024, 4096\}$. For each $k$, binary-search the critical corpus size $n_c$ over $n \in [10^3, 10^6]$, with $r=2$ relevant documents per query and the query set covering a random sample of $10^5$ 2-subsets. Two arms per cell:

- **Arm A (free embeddings, the floor).** Optimize $\phi \in \mathbb{R}^{|Q|\times k}$, $\psi \in \mathbb{R}^{n \times k}$ directly with InfoNCE, 10 seeds, until no improvement for 5k steps. Success = 100% of queries have both relevants in top-2. Cost: dominated by an $|Q| \times n$ score matrix; ~a few hundred GPU-hours total at $n=10^6$, $k=4096$ with chunked scoring.
- **Arm B (trained encoder, the achievable).** Same corpora rendered as text (document = a unique short identifier string plus filler); fine-tune a standard 7B-class embedder with output width $k$; same success criterion.

**Control arm.** BM25 and a ColBERT-style late-interaction model on the identical corpora. Both should be near-100% at all $n$; if they are not, the corpus construction leaked lexical signal and the run is void.

**Deciding number.** The ratio
$$\rho(k) = \frac{n_c^{\text{Arm B}}(k)}{n_c^{\text{Arm A}}(k)}.$$
If $\rho(k)$ is roughly constant across the four $k$ values (say within 2×), the theoretical floor is the binding constraint and dimension is the right lever. If $\rho(k)$ decays with $k$ — trained encoders falling further behind as width grows — then optimization, not representational capacity, is what limits deployed retrievers, and widening embeddings is the wrong intervention. One number, four measurements, unambiguous.

## 9. Key References

- **[Foundational]** N. Alon, P. Frankl, V. Rödl. *Geometrical realization of set systems and probabilistic communication complexity.* FOCS, 1985.
- **[Foundational]** J. Forster. *A linear lower bound on the unbounded error probabilistic communication complexity.* Journal of Computer and System Sciences, 2002 (conf. CCC 2001).
- **[Foundational]** S. Ben-David, N. Eiron, H. U. Simon. *Limitations of learning via embeddings in Euclidean half spaces.* Journal of Machine Learning Research, 2002.
- **[Foundational]** W. B. Johnson, J. Lindenstrauss. *Extensions of Lipschitz mappings into a Hilbert space.* Contemporary Mathematics, 1984.
- **[Theory SOTA]** K. G. Larsen, J. Nelson. *Optimality of the Johnson–Lindenstrauss Lemma.* FOCS, 2017. — arXiv:1609.02094
- **[Theory SOTA]** N. Alon, S. Moran, A. Yehudayoff. *Sign rank versus VC dimension.* COLT, 2016. — arXiv:1503.07648
- **[SOTA]** O. Weller, M. Boratko, I. Naim, J. Lee. *On the Theoretical Limitations of Embedding-Based Retrieval.* 2025. — arXiv:2508.21038
- **[SOTA]** A. Kusupati et al. *Matryoshka Representation Learning.* NeurIPS, 2022. — arXiv:2205.13147
- **[SOTA]** O. Khattab, M. Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR, 2020. — arXiv:2004.12832
- **[Survey]** N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2104.08663
- **[Survey]** N. Muennighoff, N. Tazi, L. Magne, N. Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316

## 10. Worked Example

Take $n = 46$ documents and let the query set be **every** 2-subset: $\binom{46}{2} = 1035$ queries. $A$ is $1035 \times 46$ with exactly two ones per row.

Naive intuition says a 3072-dimensional embedder has ample room: $3072 \gg 46$, and $A$ has rank at most 46. Rank is not the binding quantity. The requirement is that for each of 1035 queries, a single vector $\phi(q)$ must score its two documents above the other 44 — 1035 simultaneous linear separations over the same 46 fixed document vectors.

Count the constraints. Each query imposes $2 \times 44 = 88$ strict inequalities, so $1035 \times 88 = 91{,}080$ inequalities on $46k$ document parameters plus $1035k$ query parameters. Parameter count is not the problem either; the geometry is. Fixing $\psi(d_1),\dots,\psi(d_{46})$, the set of 2-subsets realizable as "top-2 of some query direction" is bounded by the number of 2-faces on the convex position of those 46 points. In low dimension that number is far below 1035, so most pairs are simply unreachable — no query vector, at any precision, puts exactly those two on top.

This is what LIMIT measures. Embedders with $k$ from 768 to 4096 recover only a small fraction of the pairs at recall@100 — recall on the order of a few percent to under ~20% — on a corpus small enough to fit on one screen. BM25, which is not a fixed-dimension inner product, and late-interaction retrievers do far better.

**Where the obstruction becomes visible:** the failure is not a training deficiency you can fix with more data, and it is invisible on BEIR, where relevance is topical and block-structured so $\mathrm{rop}(A)$ stays small. The same 3072-dimensional model is near-optimal on one corpus and near-useless on another two orders of magnitude smaller — and no current measurement, applied to a real corpus with incomplete qrels, tells you in advance which case you are in.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*