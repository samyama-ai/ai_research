---
id: 05-retrieval-and-agents/index-freshness-continual-updates
title: "Index Freshness under Continual Corpus Updates"
topic: 05-retrieval-and-agents
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Index Freshness under Continual Corpus Updates

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/index-freshness-continual-updates` · **Status:** partially-solved

## 1. Problem Statement

A retrieval index is built once over a corpus. The corpus then changes forever: documents are inserted, edited, and deleted, and the embedding model itself is eventually replaced. The problem is to keep the served index equivalent to a from-scratch rebuild, at bounded cost, without ever taking the index offline.

Three variants, with very different difficulty:

- **Measurement.** Define and compute *staleness* of a live index. What is the gap between what the index returns at time $t$ and what an oracle rebuilt at time $t$ would return — and how does that gap translate into downstream answer error? Largely unresolved; most systems report only recall against a static snapshot.
- **Method.** Build an update algorithm whose recall does not decay over an unbounded update stream, with amortized per-update cost far below rebuild. Partially solved for vector indexes (FreshDiskANN, SPFresh, IP-DiskANN); unsolved for learned/generative indexes and for embedding-model version changes.
- **Theory.** Prove a bound on recall degradation as a function of the number of updates absorbed since the last rebuild, for a named graph or partition index. Open. No known non-trivial guarantee for HNSW or Vamana under adversarial or even i.i.d. deletion.

Solving it means: an index maintained for $10^9$ updates whose recall@10 stays within $\epsilon$ of rebuild, with a proof or a measured curve showing the drift does not compound, and a visibility lag ceiling that holds at the tail.

## 2. Formal Setting

Corpus is a stream. At time $t$, $C_t \subseteq \mathcal{D}$ is the live document set, produced by an update sequence $u_1,\dots,u_t$ with $u_i \in \{\texttt{ins}(d), \texttt{del}(d), \texttt{upd}(d,d')\}$.

An index maintenance algorithm is $A$ with state $I_t = A(I_{t-1}, u_t)$. The **oracle** is $I_t^\star = \text{Build}(C_t)$, a full rebuild.

**Recall.** For query $q$ and exact top-$k$ neighbour set $G_k(q, C_t)$ under distance $\rho$ (brute-force computed, the only defensible ground truth),
$$R_k(I_t, q) = \frac{|\,\text{Ret}_k(I_t, q) \cap G_k(q, C_t)\,|}{k}.$$

**Staleness / drift.** Measured, not defined away:
$$\Delta_t = \mathbb{E}_{q\sim Q_t}\!\left[R_k(I_t^\star,q) - R_k(I_t,q)\right].$$
$\Delta_t > 0$ means the maintained index lost quality *relative to rebuild*, isolating maintenance damage from index-family error.

**Visibility lag.** For each document, $L(d) = t_{\text{queryable}}(d) - t_{\text{ingest}}(d)$. Report $p50$ and $p99$, never the mean — refresh is batched, so the distribution is a sawtooth, and the mean hides the tail. In Lucene/Elasticsearch this is governed by `refresh_interval` (default 1 s).

**Correctness of deletion.** $\text{Del}_t$ = fraction of returned results referencing documents already deleted. A tombstone-filtered system has $\text{Del}_t = 0$ but pays a **memory amplification** $M_t = |I_t| / |C_t|$ that grows with deletions until compaction.

**Cost.** Amortized per-update work $c = \frac{1}{T}\sum_t \text{cost}(A, u_t)$ against rebuild cost $c^\star = \text{cost}(\text{Build}(C_T))/T$. The maintenance ratio $\gamma = c/c^\star$ is the number that decides whether maintenance beats periodic rebuild.

**Objective.**
$$\min_A \ \gamma \quad \text{s.t.} \quad \sup_{t \le T} \Delta_t \le \epsilon, \quad p99\,L \le \tau, \quad \sup_t M_t \le \mu.$$

**Assumptions, and which break.**
- *Updates i.i.d. from the base distribution* — violated. Real deletions are correlated (a whole tenant, a crawl batch), and inserts are topically bursty (news, incidents).
- *Query distribution independent of updates* — violated hard. Fresh documents attract disproportionate query mass, so uniform-query recall understates the cost of staleness.
- *Embedding function fixed* — violated on a months timescale. A model swap makes $\rho$ itself change, and no incremental algorithm covers it: re-embedding is $O(|C_t|)$.
- *Ground truth computable* — brute-force $G_k$ at $10^9 \times$ 100 dims costs hours of GPU per query batch, so most evaluations use sampled or approximate ground truth and inherit its bias.

## 3. State of the Art

**Systems/empirical SOTA (established, with ablations).**
- **FreshDiskANN** (Singh, Subramanya, Krishnaswamy, Simhadri, 2021, arXiv:2105.09613) — first graph index with an explicit delete-consolidation algorithm ("FreshVamana") and a long-stream evaluation rather than a single-snapshot number. Established: recall stays near rebuild across streams of tens of millions of operations on SIFT/DEEP-scale data.
- **SPFresh** (Xu et al., SOSP 2023) — in-place LIRE protocol for partition-based indexes: split, merge, and reassign only the postings whose boundaries moved. Established by ablation that reassignment, not splitting, is what preserves recall.
- **IP-DiskANN** — in-place deletion for graph indexes without periodic global consolidation (Zhang et al., 2025). Claimed to remove the rebuild pause; long-horizon drift under adversarial deletion patterns is *not* ablated.
- **Ada-IVF** (Mohoney et al., 2024) — incremental IVF maintenance driven by per-partition access and drift statistics; reports large maintenance-throughput gains over uniform reindexing. Reported as benchmark numbers on a small set of streams, not independently reproduced.

**Benchmark-only results.** The NeurIPS'23 Big-ANN **streaming track** (Simhadri et al., 2024, arXiv:2409.17424) defines a runbook of interleaved insert/delete/search on MS-Turing-30M with a fixed memory budget. Everything reported there is a leaderboard number under one runbook; the runbooks are synthetic (sliding-window and expiration patterns), so ranking transfer to production churn is unestablished.

**Learned-index side.** **DSI++** (Mehta et al., EMNLP 2023, arXiv:2212.09744) shows generative retrieval forgets: adding documents by continued training degrades retrieval on the original set. **IncDSI** (Kishore et al., ICML 2023) adds documents in ~milliseconds by solving a constrained optimization for a new document vector instead of retraining — established for insertion only, no deletion story.

**Theory SOTA.** Essentially none for the dynamic case. HNSW (Malkov & Yashunin, TPAMI 2020) and DiskANN (Subramanya et al., NeurIPS 2019) have no proven recall guarantee even statically at practical parameters; dynamic guarantees inherit that vacuum. Dynamic-graph maintenance results exist for exact structures (e.g. cover trees, navigating nets) but at dimension-dependent costs that are impractical.

## 4. What Is Known

- **Graph indexes degrade under deletion if edges are not repaired.** FreshDiskANN's ablation shows that tombstoning without consolidation drops recall progressively as the deleted fraction grows; the consolidation step is what recovers it. Measured at 100M-point scale on SIFT-derived data.
- **Insertion alone is comparatively benign.** Incremental HNSW/Vamana insertion tracks rebuild recall closely for corpus growth of a few multiples; the hard case is deletion and *update* (delete+insert of a shifted vector).
- **Partition indexes drift through imbalance, not connectivity.** SPFresh's measurements attribute recall loss to postings whose centroids no longer represent their contents; repairing boundaries restores recall at a fraction of rebuild cost (SOSP 2023).
- **Continual training forgets.** DSI++ measures forgetting of previously indexed documents when new ones are added by further training, and mitigates but does not eliminate it (EMNLP 2023, on NQ/MS MARCO subsets).
- **Downstream freshness is a separate, measurable failure.** FreshQA (Vu et al., 2023, arXiv:2310.03214), StreamingQA (Liska et al., ICML 2022) and RealTime QA (Kasai et al., NeurIPS 2023 D&B) each show large accuracy gaps on time-sensitive questions, and that retrieval augmentation shrinks but does not close them. These measure the *answer*, not the index.
- **CRUD-RAG** (Lyu et al., 2024, arXiv:2401.17043) supplies a create/read/update/delete-structured RAG benchmark — the closest existing artifact to a freshness evaluation at the application layer.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form "recall@$k$ after $m$ updates $\ge$ recall of rebuild $- f(m, \text{deleted fraction})$" for any deployed graph index. Also open: whether the FreshVamana consolidation rule is enough to keep the graph's expansion properties bounded over an unbounded stream, or whether drift is unbounded for some adversarial delete order.
- **Empirically open.** Whether $\Delta_t$ compounds or plateaus over $10^9$ updates. Every published stream is $10^7$–$10^8$ operations. The experiment is runnable — it needs machine-months, not new ideas.
- **Empirically open.** Whether recall drift of a few points causes measurable end-task error. Nobody has run the $\Delta_t$-vs-answer-accuracy curve on a real agentic workload.
- **Methodologically blocked.** *Freshness itself has no agreed metric.* Recall-vs-rebuild, visibility lag, and stale-result rate are three different things, reported inconsistently, and none accounts for query mass concentrating on new documents. Also blocked: how to compare indexes across an embedding-model version change, where the distance function is not the same object before and after.

## 6. Why It Is Hard

**The obstruction is absent ground truth at the scale where the effect lives, plus a confounded metric.** Computing $G_k(q, C_t)$ exactly requires brute force over the live corpus at *every* evaluation timestamp — the corpus is different at each one, so ground truth cannot be amortized across the stream the way it is for a static benchmark. At $10^9$ vectors and 100 evaluation points, that is a bigger compute bill than the index maintenance being studied, which is why published streams stop at $10^7$–$10^8$.

Second obstruction: the reported metric does not measure the named thing. "Recall@10 = 0.94 on the streaming runbook" mixes index-family error with maintenance damage, and is averaged over a query distribution that is uniform in the benchmark and heavily fresh-skewed in production. A system can rank first on the runbook and be worse where the queries actually are.

## 7. Current Research (as of 2026)

- **In-place graph maintenance** — the FreshDiskANN → IP-DiskANN line at Microsoft Research India and collaborators, aimed at removing the stop-the-world consolidation pause.
- **Statistics-driven partition maintenance** — Ada-IVF-style access-aware reindexing (Wisconsin/Mohoney and co-authors), being absorbed into production vector stores. *(frontier — verify which stores shipped it.)*
- **Big-ANN streaming track** continuation and harder runbooks with correlated deletions. *(frontier — verify 2026 edition scope.)*
- **Filtered + streaming jointly** — maintaining recall when attribute filters and updates interact; the two features degrade each other and are usually benchmarked separately.
- **Incremental re-embedding** — projecting old vectors into a new model's space to avoid full re-encode. *(frontier — verify; published results are small-scale and quality loss is not well characterized.)*
- **Agentic/memory freshness** — long-lived agents whose retrieved memory must reflect their own recent writes; visibility lag becomes a correctness bug, not a quality knob.

## 8. Concrete Next Experiment

**Question:** does maintenance drift $\Delta_t$ plateau or compound?

- **Scale.** 100M vectors, 768-d (e.g. MS MARCO passages embedded once and up-sampled, or MS-Turing-100M). Run a stream of $10^9$ operations: 50% insert, 40% delete, 10% update, with deletions drawn *correlated* (delete whole crawl batches / whole clusters), not uniformly.
- **Arms.**
  - *Maintained:* FreshDiskANN, SPFresh, IP-DiskANN, incremental HNSW.
  - *Control (mandatory):* full rebuild $I_t^\star$ at 20 logarithmically spaced checkpoints, evaluated with **exact** brute-force ground truth on a fixed 10k-query set, plus a second 10k-query set sampled with recency-weighted mass to model real query skew.
- **Deciding number.** The slope of $\Delta_t$ against $\log$(operations absorbed) between the $10^8$ and $10^9$ checkpoints, on the recency-weighted query set. **Slope $\le 0.002$ recall points per decade $\Rightarrow$ drift plateaus and maintenance is a solved engineering problem at this scale. Slope $\ge 0.01$ $\Rightarrow$ drift compounds, and periodic rebuild is not an optimization but a correctness requirement** — which in turn sets the rebuild interval as a function of churn rate.
- **Cost estimate.** Ground truth dominates: 20 checkpoints × 20k queries × 100M × 768-d ≈ $3\times10^{16}$ multiply-adds, roughly a few GPU-days on A100-class hardware with a batched exact search — affordable, and the reason the experiment is *empirically open* rather than blocked.

## 9. Key References

- **[Foundational]** Yu A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 2020. — arXiv:1603.09320
- **[Foundational]** Suhas Jayaram Subramanya, Fnu Devvrit, Rohan Kadekodi, Ravishankar Krishnaswamy, Harsha Vardhan Simhadri. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS, 2019.
- **[SOTA]** Aditi Singh, Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, Harsha Vardhan Simhadri. *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search.* 2021. — arXiv:2105.09613
- **[SOTA]** Yuming Xu, Hengyu Liang, Jin Li, Shuotao Xu, Qi Chen, Qianxi Zhang, Cheng Li, Ziyue Yang, Fan Yang, Yuqing Yang, Peng Cheng, Mao Yang. *SPFresh: Incremental In-Place Update for Billion-Scale Vector Search.* SOSP, 2023.
- **[SOTA]** Qi Chen et al. *SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search.* NeurIPS, 2021.
- **[SOTA]** Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc Najork, Emma Strubell, Donald Metzler. *DSI++: Updating Transformer Memory with New Documents.* EMNLP, 2023. — arXiv:2212.09744
- **[SOTA]** Varsha Kishore, Chao Wan, Justin Lovelace, Yoav Artzi, Kilian Q. Weinberger. *IncDSI: Incrementally Updatable Document Retrieval.* ICML, 2023.
- **[Benchmark]** Harsha Vardhan Simhadri et al. *Results of the Big ANN: NeurIPS'23 Competition.* 2024. — arXiv:2409.17424
- **[Benchmark]** Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, Thang Luong. *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.* 2023. — arXiv:2310.03214
- **[Benchmark]** Adam Liska et al. *StreamingQA: A Benchmark for Adaptation to New Knowledge over Time in Question Answering Models.* ICML, 2022. — arXiv:2205.11388
- **[Benchmark]** Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, Kentaro Inui. *RealTime QA: What's the Answer Right Now?* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2207.13332
- **[Survey/Benchmark]** Yuanjie Lyu et al. *CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models.* 2024. — arXiv:2401.17043

## 10. Worked Example

A support-knowledge-base RAG system. 2M articles, 768-d embeddings, HNSW with $M=32$, `efConstruction=200`. Churn: 4,000 edits/day, 1,000 deletions/day, 1,500 insertions/day — about 0.3% of the corpus per day.

Deletions are handled by tombstone: the vector stays in the graph as a routing node, filtered out of results. After 180 days, deleted nodes are $180{,}000 / 2{,}000{,}000 = 9\%$ of graph nodes. Edits are delete+insert, so the true dead fraction is $180 \times 5{,}000 / 2{,}000{,}000 = 45\%$.

Now the failure. A search with `efSearch = 64` visits a candidate list of 64 nodes. If 45% are tombstones, the *effective* candidate list is about $64 \times 0.55 \approx 35$ live nodes. The operator has not changed a single parameter, but the index is running at roughly `efSearch = 35`. Measured recall@10 on this configuration typically falls from ~0.95 to ~0.88 — and the dashboard shows nothing, because the dashboard reports latency (which *improved*, since tombstone filtering is cheap) and query volume.

Worse: the loss is not uniform. Deleted articles cluster — an entire deprecated product line was removed. In that region of the space, the live fraction of the candidate list is near zero, and queries about the *replacement* product route through a graph neighbourhood made of dead nodes. Recall there is closer to 0.5. Aggregate recall of 0.88 is the average of 0.95 almost everywhere and 0.5 in the region that gets the most traffic — the new product is what people ask about.

**The obstruction, visible.** To detect this the operator needs $\Delta_t$: recall against a rebuild, on the *current* corpus, with the *current* query distribution. The rebuild costs 2M × 200 × 32 distance computations — cheap here, hours at $10^9$. The exact ground truth costs 10k queries × 2M × 768 ≈ $1.5\times10^{13}$ multiply-adds — also cheap here. At 100M vectors both terms grow 50×, and at $10^9$ with 20 checkpoints they grow past what anyone budgets for an evaluation. So the effect is trivially measurable at the scale where it does not matter much, and unmeasured at the scale where it does. That gap is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*