---
id: 07-embeddings/long-document-embedding-without-chunking
title: "Long-Document Embedding Without Chunking"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Document Embedding Without Chunking

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/long-document-embedding-without-chunking` · **Status:** open

## 1. Problem Statement

Given a document of $10^4$–$10^6$ tokens, produce a **single fixed-size vector** that supports retrieval, clustering and similarity at least as well as the standard pipeline: split the document into passages, embed each, and score by max or sum over passages.

Three variants, routinely conflated:

- **Measurement.** Does any current encoder actually *use* its full advertised context window when producing a document embedding? The claim "8192-token embedding model" is a claim about the input tensor shape, not about information flow into the output vector.
- **Method.** Build an encoder whose whole-document vector beats chunk-and-max on long-document retrieval at equal or lower index cost, with no per-query passage rescoring.
- **Theory.** Determine the information-theoretic price of the single-vector constraint. A $k$-dimensional vector cannot represent every relevance pattern over a corpus; the question is where the ceiling sits as a function of $k$, document length $n$, and query complexity.

Solving it means: one vector per document, $k \le 1024$, that matches multi-vector or chunked baselines on a long-document benchmark whose relevance genuinely requires evidence spread across positions.

## 2. Formal Setting

A document is a token sequence $d = (x_1,\dots,x_n)$, $x_i \in \mathcal{V}$. An encoder $f_\theta: \mathcal{V}^{\le N} \to \mathbb{R}^k$ maps it to a unit-norm vector; $N$ is the architectural context limit, $k$ the embedding width. Query $q$ scores as

$$s(q,d) = \langle f_\theta(q),\, f_\theta(d)\rangle .$$

The **chunked control** partitions $d$ into $m = \lceil n/L \rceil$ windows $c_1,\dots,c_m$ of length $L$ (typically $L=512$) with stride $L - o$, and scores

$$s_{\mathrm{chunk}}(q,d) = \max_{j \le m} \langle f_\theta(q), f_\theta(c_j)\rangle .$$

Index cost is $m$ vectors of $k$ floats versus $1$; the ratio $m$ is the quantity the single-vector approach is buying down.

**Effective context, as measured.** Advertised $N$ is not usable context. Measure by perturbation: let $d^{(i)}$ be $d$ with a span at position $i$ replaced by an in-distribution distractor of equal length. Define positional sensitivity

$$\alpha_i \;=\; \mathbb{E}_{d}\big[\,\|f_\theta(d) - f_\theta(d^{(i)})\|_2\,\big],\qquad \bar\alpha_i = \alpha_i / \textstyle\sum_{j}\alpha_j ,$$

and the **effective context length** $n^\ast(\tau)$ as the smallest prefix length $p$ with $\sum_{i \le p} \bar\alpha_i \ge \tau$, $\tau = 0.9$. A model that ignores its tail has $n^\ast \ll N$. The operational analogue is needle-in-a-haystack recall: insert a gold sentence at relative depth $\rho \in [0,1]$ and report Recall@1 as $R(\rho)$; $R$ should be flat in $\rho$.

**Capacity.** Fix a corpus $D$ and a query set $Q$ with binary relevance $r(q,d)$. Single-vector retrieval realises $r$ only if there exist $\{u_q\},\{v_d\} \subset \mathbb{R}^k$ and thresholds with $\mathrm{sign}(\langle u_q,v_d\rangle - t_q) = r(q,d)$. The minimum such $k$ is the sign-rank of the $|Q| \times |D|$ relevance matrix, so the constraint is a rank condition, not a training-quality condition.

**Assumptions known to be violated.**
1. *Relevance is chunk-local* — assumed by the max-over-chunks control, false for queries requiring aggregation across sections.
2. *Positional uniformity* — assumed by "8K model" claims, false: encoders are measurably front-biased.
3. *Train/test length match* — most contrastive training pairs are 100–500 tokens; inference at 8K–100K is extrapolation.
4. *Benchmark documents are long* — BEIR corpora average a few hundred tokens, so BEIR cannot separate the variants at all.

## 3. State of the Art

**Established.**
- Sparse/windowed attention makes 4K–16K encoding tractable: Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., NeurIPS 2020). Established as *compute*, not as *embedding quality*.
- RoPE interpolation extends context post-hoc: Position Interpolation (Chen et al., 2023), YaRN (Peng et al., ICLR 2024). LongEmbed (Zhu et al., EMNLP 2024) showed these transfer to embedding models plug-and-play, with no retraining.
- Multi-vector late interaction (ColBERT, Khattab & Zaharia, SIGIR 2020; ColBERTv2, NAACL 2022) reliably beats single-vector on hard retrieval — but it is the thing this problem is trying to avoid.

**Claimed, weakly ablated.**
- Jina Embeddings v2 (Günther et al., 2023), nomic-embed-text-v1 (Nussbaum et al., 2024) and BGE-M3 (Chen et al., ACL Findings 2024) ship 8K windows. The evidence that the 8K vector beats the same model's chunked output on the *same* corpus is thin; most reported gains are benchmark deltas on LoCo or MLDR without a matched chunk-and-max arm at equal budget.
- M2-BERT retrieval + LoCo (Saad-Falcon et al., ICML 2024) is the cleanest long-context single-vector result, but LoCo's documents come from summarization corpora (SummScreenFD, QMSum, GovReport) where the query is often a summary — relevance is diffuse by construction, which *favours* whole-document pooling. This is a benchmark number, not a general finding.
- Late chunking (Günther et al., 2024) — encode the full document, pool per chunk afterwards — improves over naive chunking on several BEIR subsets. It is a hybrid: it keeps $m$ vectors, so it does not solve the stated problem, but it is the strongest evidence that long-context conditioning helps at all.

## 4. What Is Known

- **Front bias is real and measured.** Coelho et al. (ACL 2024) found dense retrieval encoders concentrate representational mass in early positions; relevant content late in a document is systematically under-weighted. Measured on retrieval encoders at 512–2048 tokens.
- **Effective $\ne$ advertised.** LongEmbed (EMNLP 2024, 6 datasets to 32K) found models with 4K–32K windows whose retrieval accuracy collapses well before the limit; RoPE-scaling recovered a large part of the gap on E5-Mistral without any training. Analogous to RULER (Hsieh et al., COLM 2024) for generative models, where several "128K" models fall below 32K effective.
- **"Lost in the middle"** (Liu et al., TACL 2024): U-shaped position–performance curve in long-context LMs. The same shape appears in $R(\rho)$ for embedding models, though with fewer independent replications.
- **Single-vector retrieval has a hard capacity ceiling.** Weller et al. (2025, LIMIT) construct tiny corpora — dozens of documents — where no embedding of practical width can express the required relevance pattern, and show top models scoring near zero while BM25 and multi-vector methods succeed. Scale: $\sim$50 documents, embeddings up to 4096-d.
- **Chunk-and-max is a strong baseline.** Across BEIR-style evaluation, chunking a long document at $L=512$ with overlap and taking the max rarely loses to whole-document pooling on single-fact queries — it wins, because it avoids dilution.

## 5. What Is Not Known

- **Theoretically open.** The trade-off curve $k$ vs. $n$ vs. query complexity. LIMIT shows a ceiling exists; no theorem gives the required $k$ as a function of document length and the number of independent "facts" per document that a query may condition on. No lower bound separates single-vector from max-over-$m$-vectors as a function of $m$.
- **Empirically open.** Whether a whole-document encoder trained *with long-document contrastive pairs at scale* (not the usual 100–500-token pairs) closes the gap. The experiment is runnable — it needs a long-pair corpus and a few thousand GPU-hours — and has not been run with a matched-budget chunked control.
- **Methodologically blocked.** There is no long-document retrieval benchmark whose relevance provably requires cross-position aggregation. LoCo and MLDR conflate "long document" with "diffuse query"; needle tests conflate it with "single-fact recall". Until a benchmark exists where chunk-and-max is *provably* upper-bounded below 1.0, no measured win can be attributed to the architecture rather than the corpus.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names**, compounded by **dilution as a fixed-point property of pooling**.

Every long-document benchmark in use is solvable by chunk-and-max to within a few points, because relevance in those corpora is locally witnessed. So the benchmark cannot reward the capability the method claims. Meanwhile mean-pooling over $m$ segments shrinks any single segment's contribution to $O(1/m)$ of the vector while distractor documents with uniformly mild relevance keep theirs — so the pooled representation is not just lossy, it is *adversarially* lossy against exactly the documents a ranker must separate (see §10). Attention pooling does not remove this: it moves the problem to learning a selector under a training distribution whose documents are 20× shorter than test documents.

Secondary: the compute cost of training. Long-document contrastive learning needs large in-batch negative counts *and* long sequences; the two multiply against memory, which is why nearly all embedding models are trained short and evaluated long.

## 7. Current Research (as of 2026)

- **Post-hoc context extension of encoders** — RoPE/NTK scaling, SelfExtend-style tricks applied to embedding models (Zhu et al. and successors). Cheap; effect size is real but caps out.
- **Hybrid late interaction with compression** — ColBERT-style token vectors with aggressive pooling/quantization (Stanford, JHU, Vespa/LightOn groups). Reduces $m$ rather than reaching $m=1$.
- **Sub-quadratic sequence models as encoders** — M2-BERT, Mamba-style encoders (Stanford Hazy Research, CMU/Princeton). *(frontier — verify)* whether state-space encoders show flatter $R(\rho)$ than attention encoders is being tested but not settled.
- **Capacity-aware retrieval** — following LIMIT, work on when to spend more dimensions vs. more vectors (Google DeepMind). *(frontier — verify)*
- **Benchmark construction for cross-position relevance** — several groups are building multi-hop-within-document retrieval sets. This is the bottleneck item and the least crowded.

## 8. Concrete Next Experiment

**Question.** Does a whole-document embedding ever beat chunk-and-max when relevance provably requires two spans separated by more than one chunk?

**Construction (scale).** Build 5,000 synthetic-but-natural documents of 16K tokens each by concatenating real Wikipedia sections. For each, plant two atomic facts $A$ and $B$ at token offsets drawn so their separation exceeds $2L = 1024$. Queries are conjunctions ("the entity that has property $A$ and property $B$"), 5,000 of them, one gold document each, with distractor documents containing $A$ or $B$ alone. This makes the chunked ceiling explicit: no 512-token chunk contains both facts.

**Arms.**
1. *Control:* the same encoder, chunked at $L=512$, stride 256, scored by max (and by sum, as a second control — sum can in principle aggregate).
2. *Treatment:* whole-document single vector, $k=1024$, encoder with 16K context, fine-tuned on 200K long conjunctive pairs.
3. *Upper reference:* ColBERT-style multi-vector on the same documents.

**Deciding number.** nDCG@10 of arm 2 minus arm 1-max, at equal index bytes per document. Arm 1 stores $\sim$63 vectors/doc; equalise by giving arm 2 the same byte budget only if it wants it, otherwise report the 63× storage saving alongside. **Threshold: a $\ge$ +5.0 nDCG@10 gain for arm 2 over arm 1-sum (not just arm 1-max), on held-out documents with separation $> 4096$ tokens, would establish that whole-document encoding buys something chunking cannot.** A gain over max but not over sum means the result is about aggregation, not context.

## 9. Key References

- **[Foundational]** Iz Beltagy, Matthew E. Peters, Arman Cohan. *Longformer: The Long-Document Transformer.* arXiv preprint, 2020. — arXiv:2004.05150
- **[Foundational]** Manzil Zaheer et al. *Big Bird: Transformers for Longer Sequences.* NeurIPS, 2020. — arXiv:2007.14062
- **[Foundational]** Omar Khattab, Matei Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR, 2020. — arXiv:2004.12832
- **[SOTA]** Jon Saad-Falcon et al. *Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT.* ICML, 2024. — arXiv:2402.07440
- **[SOTA]** Dawei Zhu et al. *LongEmbed: Extending Embedding Models for Long Context Retrieval.* EMNLP, 2024. — arXiv:2404.12096
- **[SOTA]** Michael Günther et al. *Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents.* arXiv preprint, 2023. — arXiv:2310.19923
- **[SOTA]** Michael Günther et al. *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models.* arXiv preprint, 2024. — arXiv:2409.04701
- **[SOTA]** Zach Nussbaum et al. *Nomic Embed: Training a Reproducible Long Context Text Embedder.* arXiv preprint, 2024. — arXiv:2402.01613
- **[Limits]** Orion Weller et al. *On the Theoretical Limitations of Embedding-Based Retrieval.* Google DeepMind, 2025. (LIMIT benchmark.)
- **[Analysis]** João Coelho et al. *Dwell in the Beginning: How Language Models Embed Long Documents for Dense Retrieval.* ACL, 2024. — arXiv:2404.04163
- **[Analysis]** Nelson F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Analysis]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Survey/Benchmark]** Nandan Thakur et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2104.08663
- **[Survey/Benchmark]** Niklas Muennighoff et al. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316

## 10. Worked Example

A 61,440-token technical report, $L=512$, so $m=120$ segments. The whole-document vector is the mean of the 120 unit-norm segment embeddings.

Query $q$ matches exactly one segment: $\langle q, v_{j^\ast}\rangle = 0.85$. The other 119 segments are off-topic: $\langle q, v_j\rangle = 0.10$. Segments within a document are topically correlated; take mean pairwise cosine $\rho = 0.30$.

Numerator (mean similarity):

$$\langle q, \bar v\rangle = \frac{0.85 + 119(0.10)}{120} = \frac{12.75}{120} = 0.1063 .$$

Norm of the mean:

$$\|\bar v\| = \sqrt{\tfrac{1}{m} + \tfrac{m-1}{m}\rho} = \sqrt{0.00833 + 0.2975} = 0.553 .$$

So the cosine score of the true positive is $0.1063 / 0.553 = \mathbf{0.192}$.

Now a distractor document, same length, no segment strongly relevant, every segment at $0.15$, same $\rho$. Its score is $0.15/0.553 = \mathbf{0.271}$.

**The distractor outranks the document that actually answers the query, by 0.079 cosine.** Chunk-and-max scores them 0.85 vs 0.15 and gets it right without effort.

The obstruction is visible here: the failure is not a training deficiency, it is arithmetic on the pooling operator. Fixing it needs a pooler that suppresses the 119 irrelevant segments *without knowing the query* — the vector is computed at index time. A learned attention pooler can only do that by betting in advance on which spans queries will care about, which for a 61K-token document with hundreds of distinct claims is exactly the capacity ceiling LIMIT describes. Raising $k$ from 1024 to 4096 changes the norm arithmetic not at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*