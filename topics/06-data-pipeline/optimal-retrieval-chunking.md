---
id: 06-data-pipeline/optimal-retrieval-chunking
title: "Optimal Chunking for Retrieval Corpora"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Chunking for Retrieval Corpora

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/optimal-retrieval-chunking` · **Status:** empirically-open

## 1. Problem Statement

Given a corpus of documents and a fixed retrieval-plus-generation stack, decide how to cut the documents into indexed units. The input is a document set $\mathcal{D}$, an embedding model $\phi$, a reader LLM $g$ with a context budget, and a task distribution over queries. The output is a segmentation policy $C$ — where the boundaries fall, how long the pieces are, how much they overlap, and what context is prepended to each. The objective is end-task utility, not retrieval score.

Three variants, routinely conflated:

- **Measurement.** Is there a metric under which "chunk quality" is well defined independent of the reader? Retrieval recall against human-annotated gold spans is the usual proxy, and it is not the objective.
- **Method.** Given a fixed utility metric, find $C$ that maximizes it. This is a search problem over a large discrete space with an expensive oracle (re-index + re-evaluate).
- **Theory.** Is there any structure — submodularity, a bias–variance decomposition, a bound relating chunk length to embedding fidelity — that makes the optimum predictable from corpus statistics rather than found by search?

Solving it means: a rule that maps measurable corpus and model properties to a chunking policy, and beats a tuned fixed-size baseline by a margin that survives re-running with a different embedding model and reader.

## 2. Formal Setting

Corpus $\mathcal{D} = \{d_1,\dots,d_N\}$, document $d$ a token sequence of length $|d|$. A chunking policy is a map
$$C : d \mapsto \{c_1,\dots,c_{m(d)}\},\quad c_i = d[s_i : e_i],$$
with $s_i < e_i$, $\bigcup_i [s_i,e_i) \supseteq [0,|d|)$ (coverage) and overlap $o_i = e_{i-1} - s_i \ge 0$. Write $L = \mathbb{E}[e_i - s_i]$ for mean chunk length in tokens (measured with the reader's tokenizer, not words — a 100-word DPR passage is roughly 130–150 BPE tokens).

Index size $M(C) = \sum_d m(d)$; redundancy $\rho(C) = \sum_i (e_i-s_i) / \sum_d |d| \ge 1$, measured directly from offsets. Embedding cost is $\Theta(\rho \cdot \sum_d |d|)$; ANN memory is $\Theta(M \cdot \dim \phi)$.

Retrieval returns the top $k$ chunks by $\langle \phi(q), \phi(c)\rangle$. The reader consumes $k$ chunks, so the fair comparison holds the **token budget** $B \approx kL$ fixed, not $k$. Utility:
$$U(C) = \mathbb{E}_{q\sim Q}\big[\, u\big(g(q, R_k(q; C))\big) \,\big], \qquad R_k(q;C)=\operatorname{top-}k_{c \in C(\mathcal{D})} \langle \phi(q),\phi(c)\rangle,$$
with $u$ the task metric (exact match, span-F1, faithfulness judged by a rubric). The optimization is $\max_C U(C)$ s.t. $M(C) \le M_{\max}$.

The retrieval-only surrogate uses gold evidence spans $A_q \subset d$:
$$\mathrm{Rec}@k = \mathbb{E}_q\Big[\tfrac{1}{|A_q|}\sum_{a \in A_q}\mathbf{1}\{\exists c \in R_k(q): a \subseteq c\}\Big],$$
which is the quantity most chunking papers report. Note $\mathbf{1}\{a \subseteq c\}$ is discontinuous in boundaries: moving one boundary by one token can flip a query from hit to miss.

Assumptions, with the ones known to fail marked:

1. **Evidence is contiguous.** Violated: multi-hop and comparison queries need $\ge 2$ spans, often in different documents; $\mathrm{Rec}@k$ with $|A_q|>1$ rewards short chunks purely for combinatorial reasons.
2. **Chunk utility is additive across retrieved chunks.** Violated: readers degrade with distractors (Cuconasu et al. 2024) and are position-sensitive (Liu et al. 2024), so $u$ is not a sum over chunks.
3. **$\phi$ is length-invariant.** Violated: bi-encoders pool into a single vector; longer chunks mix more topics into one vector, and most encoders were contrastively trained near 100–300 tokens, so behaviour beyond training length is extrapolation.
4. **Queries are i.i.d. from a stationary $Q$.** Violated in deployment; the optimal $L$ for lookup queries ("what is the deductible") differs from summarization queries.

## 3. State of the Art

**Systems/empirical SOTA.** Tuned fixed-size overlapping windows remain the operating point most production stacks use: roughly 200–500 tokens with 10–20% overlap. Chroma's evaluation of chunking strategies (Smith & Troynikov, technical report, 2024) found small recursive/fixed chunkers competitive with or better than more elaborate semantic splitters at matched token budget — a benchmark number, not an ablation isolating why.

Beyond fixed-size, four established lines:

- **Proposition-level indexing.** *Dense X Retrieval* (Chen et al., EMNLP 2024) decomposes passages into atomic propositions and indexes those; reported gains on open-domain QA recall at small $k$, largest for entity-centric queries. Established as a benchmark result; the index-size and LLM-decomposition cost are reported but the comparison is not compute-matched against simply retrieving more sentences.
- **Contextualized chunk embeddings.** *Late chunking* (Günther et al., 2024) embeds the whole document with a long-context encoder, then pools per chunk, so each chunk vector carries document context at no extra index size. Ablated against naive chunking on BEIR subsets with consistent but modest gains; not ablated against simply prepending a document title.
- **LLM-decided boundaries.** *LumberChunker* (Duarte et al., EMNLP Findings 2024) asks an LLM where content shifts; reported improvements on long-narrative QA. Cost is one LLM pass per document — usually not reported as a compute-matched baseline.
- **Hierarchy instead of a single granularity.** *RAPTOR* (Sarthi et al., ICLR 2024) builds a recursive summary tree and retrieves at mixed levels, sidestepping the choice of a single $L$.

**Theory SOTA.** Effectively none for chunking specifically. The nearest formal result is capacity: Luan et al. (TACL 2021) show single-vector bi-encoders have a fidelity limit that degrades with passage length and that multi-vector representations recover it — an argument that $L$ should be bounded, not a prediction of the optimum.

**Claimed but unablated.** That "semantic" boundaries beat arbitrary ones. Qu, Tu & Bao (2024) tested this directly and found semantic chunking's gains inconsistent and not worth the cost across several retrieval benchmarks — the most direct negative result available.

## 4. What Is Known

- **Passage-level retrieval beats document-level.** Callan (SIGIR 1994) and Kaszkiel & Zobel (SIGIR 1997) established on TREC that fixed-length overlapping windows outperform both whole documents and author-defined paragraph units. Reproduced across decades and both lexical and dense retrievers.
- **A working default exists.** DPR (Karpukhin et al., EMNLP 2020) split Wikipedia into 100-word passages (~21M chunks) and this granularity has been copied into hundreds of systems; it was chosen, not optimized, and its ablation in the paper is limited.
- **Optimum is interior.** Very short chunks lose the context needed to answer; very long chunks dilute the embedding and burn budget. Every study that sweeps $L$ finds a broad interior plateau rather than a sharp peak — typically flat over roughly a 2× range in $L$, which is why many chunkers appear tied.
- **More retrieved context is not monotonically better.** Distractor passages measurably lower answer accuracy (Cuconasu et al., SIGIR 2024), and mid-context evidence is used less well than evidence at the ends (Liu et al., TACL 2024). Both break the "just retrieve more" escape from the chunking question.
- **Long-context readers do not remove the problem.** Xu et al. (ICLR 2024) found retrieval augmentation still helps a 32k-context model, i.e. the corpus still has to be cut.
- **Semantic splitting is not reliably better.** Qu et al. (2024), across multiple BEIR-style datasets.

## 5. What Is Not Known

- **Empirically open (the main gap).** Nobody has published a compute-matched, budget-matched sweep of chunking policies evaluated on *end-task* utility, across ≥3 embedding models × ≥3 readers × ≥3 corpus genres. Every existing study varies one axis and fixes the others, so no result is known to transfer. The experiment is runnable today for well under $50k of inference.
- **Empirically open.** Whether the optimal $L$ is a property of the *encoder's training distribution* or of the *corpus*. These are confounded in every published sweep.
- **Methodologically blocked.** Chunk-level ground truth. Gold evidence spans in NQ, HotpotQA and similar sets were annotated against a particular passage segmentation; scoring a different segmentation against them measures agreement with the annotation convention as much as retrieval quality. Until utility is defined without reference to a pre-existing segmentation, $\mathrm{Rec}@k$ comparisons across chunkers are not commensurable.
- **Theoretically open.** Whether $U(C)$ has exploitable structure. No submodularity result, no bias–variance decomposition separating "context lost at boundaries" from "topic dilution inside a chunk", no bound of the form $U \ge f(L, \dim\phi, \text{topic entropy})$.

## 6. Why It Is Hard

Three named obstructions.

- **Confounded measurement.** Changing $C$ changes the index, the number of vectors, the token budget per retrieved item and the reader's prompt simultaneously. A chunker that halves $L$ doubles $M$ and doubles $k$ at fixed budget; the observed delta mixes granularity, index size and distractor count. Almost no published comparison controls all three.
- **Absent ground truth.** Per §5, gold spans are annotation-convention-bound. There is no segmentation-free oracle for "was the right evidence retrieved".
- **Compute cost of the oracle.** Each candidate $C$ requires re-embedding the corpus and re-running generation and judging. At 10M tokens and $10^3$ eval queries that is roughly an hour and tens of dollars per point — tolerable once, prohibitive for a search over boundary policies, so nobody searches; they test 4–6 hand-picked settings.
- **Non-identifiability of the plateau.** Because $U$ is flat over a wide range of $L$, typical eval sets ($n \approx 500$–2000 queries, $\sigma \approx 1.5$ pp on EM) cannot resolve 1–2 pp differences. Most "chunker A beats B" claims sit inside their own noise band.

## 7. Current Research (as of 2026)

- **Contextualization over segmentation.** Attaching document-level context to each chunk — late chunking (Jina AI), and LLM-written per-chunk context prefixes (Anthropic's "contextual retrieval" engineering report, 2024) — is the most active direction, because it decouples "unit of retrieval" from "unit of meaning".
- **Multi-vector and late-interaction retrieval** (ColBERT lineage, Khattab & Zaharia, SIGIR 2020) reduces sensitivity to $L$ by not compressing a chunk to one vector. *(frontier — verify)* whether late interaction makes chunk length nearly irrelevant is being probed but not settled.
- **Hierarchical / multi-granularity indexes** (RAPTOR descendants) — retrieve at several granularities and let a reranker choose.
- **Agentic and query-time chunking** — fetch the parent document and re-cut it conditioned on the query. *(frontier — verify)*; almost no controlled evaluations.
- **Evaluation infrastructure.** Efforts to build segmentation-free evidence annotation (character-offset gold spans rather than passage IDs) are the prerequisite for unblocking §5's methodological gap.

## 8. Concrete Next Experiment

**Question.** Is optimal chunk length a property of the encoder or of the corpus?

**Scale.** Three corpora with different structure (Wikipedia ~5M tokens sample; SEC 10-K filings ~5M; a clinical or legal set with long clause-level structure ~5M). Three encoders spanning training-context regimes (a 512-token bi-encoder, an 8k-context bi-encoder, one late-interaction model). Two readers (one ~8B, one frontier-class). Sweep $L \in \{64,128,256,512,1024\}$ tokens at fixed overlap 15%, **holding the reader token budget $B = 2048$ constant** by setting $k = \lfloor B/L \rfloor$. 2000 queries per corpus with character-offset gold evidence, annotated once, independent of any segmentation. Total: $3\times3\times5 = 45$ indexes, 2 readers, ~$10^5$ generations.

**Control arm.** Fixed 256-token chunks, 15% overlap, no semantic boundaries, no contextualization — the incumbent default — run inside every cell so cells are comparable.

**Deciding number.** $\Delta = L^\star_{\text{corpus}}$ variance vs. $L^\star_{\text{encoder}}$ variance in a two-way ANOVA on end-task EM, where $L^\star$ is the argmax over the sweep. If the encoder main effect explains $>2\times$ the variance of the corpus main effect (with $\pm1.5$ pp per-cell noise, $n=2000$ gives resolution of ~1 pp), chunking is an encoder-tuning problem and should ship as a per-model constant. If the corpus effect dominates, it is a curation problem and per-corpus search is justified. If neither exceeds the interaction term, the plateau is real and the field should stop tuning $L$ and spend the effort on contextualization.

## 9. Key References

- **[Foundational]** J. P. Callan. *Passage-Level Evidence in Document Retrieval.* SIGIR, 1994.
- **[Foundational]** M. Kaszkiel, J. Zobel. *Passage Retrieval Revisited.* SIGIR, 1997.
- **[Foundational]** M. A. Hearst. *TextTiling: Segmenting Text into Multi-Paragraph Subtopic Passages.* Computational Linguistics, 1997.
- **[Foundational]** V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, W. Yih. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP, 2020. — arXiv:2004.04906
- **[Foundational]** P. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS, 2020. — arXiv:2005.11401
- **[Theory]** Y. Luan, J. Eisenstein, K. Toutanova, M. Collins. *Sparse, Dense, and Attentional Representations for Text Retrieval.* TACL, 2021. — arXiv:2005.00181
- **[SOTA]** T. Chen, H. Wang, S. Chen, W. Yu, K. Ma, X. Zhao, H. Zhang, D. Yu. *Dense X Retrieval: What Retrieval Granularity Should We Use?* EMNLP, 2024. — arXiv:2312.06648
- **[SOTA]** M. Günther, I. Mohr, D. Williams, B. Wang, H. Xiao. *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models.* Jina AI, 2024. — arXiv:2409.04701
- **[SOTA]** A. V. Duarte, J. Marques, M. Graça, M. Freire, L. Li, A. L. Oliveira. *LumberChunker: Long-Form Narrative Document Segmentation.* Findings of EMNLP, 2024. — arXiv:2406.17526
- **[SOTA]** P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, C. D. Manning. *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* ICLR, 2024. — arXiv:2401.18059
- **[Negative result]** R. Qu, R. Tu, F. Bao. *Is Semantic Chunking Worth the Computational Cost?* 2024. — arXiv:2410.13070
- **[Context]** F. Cuconasu et al. *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR, 2024. — arXiv:2401.14887
- **[Context]** N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, P. Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Context]** P. Xu et al. *Retrieval Meets Long Context Large Language Models.* ICLR, 2024. — arXiv:2310.03025
- **[Benchmark]** N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2104.08663
- **[Report]** B. Smith, A. Troynikov. *Evaluating Chunking Strategies for Retrieval.* Chroma technical report, 2024. — non-peer-reviewed; benchmark numbers only.
- **[Survey]** Y. Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey.* 2023/2024. — arXiv:2312.10997

## 10. Worked Example

A 10-K filing, 120k tokens. One query: *"What was the change in the effective tax rate, and why?"* The answer needs a number from a table on page 41 and a one-sentence cause from MD&A on page 27 — two spans, 14k tokens apart.

Budget $B = 2048$ reader tokens.

| $L$ | $k=\lfloor B/L\rfloor$ | chunks in doc ($\rho=1.15$) | both spans retrievable? |
|---|---|---|---|
| 128 | 16 | 1078 | needs 2 of 16 slots to hit 2 of 1078 |
| 256 | 8 | 539 | 2 of 8 slots, 2 of 539 |
| 512 | 4 | 269 | 2 of 4 slots, 2 of 269 |
| 1024 | 2 | 135 | 2 of 2 slots — must be perfect |

Assume each span is independently retrieved with probability $p(L)$, rising with $k$ (more slots) and falling with $L$ (diluted chunk vector). Take a plausible $p$: $0.62, 0.71, 0.68, 0.44$ across the four rows. The query needs **both**, so success $\approx p^2$: $0.38, 0.50, 0.46, 0.19$. Optimum at $L=256$, and the 256-vs-512 gap is 4 pp.

Now the obstruction. With 500 eval queries, a 4 pp difference has standard error $\sqrt{0.5\cdot0.5/500} \approx 2.2$ pp, so the 95% CI on the gap is roughly $\pm 6$ pp — the "winner" is inside the noise. To resolve 4 pp at 95% confidence needs about 2400 queries. Meanwhile the $L=1024$ collapse is not a chunking fact at all: it is $k=2$ leaving no room for two-hop evidence, i.e. the budget constraint, not the segmentation. And $p(L)$ itself was set by the encoder's 512-token training window — swap in a late-interaction retriever and the $L=1024$ row plausibly recovers.

Three different causes — slot count, encoder training length, span dispersion — all show up as "chunk size matters", and the standard experiment separates none of them. That is the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*