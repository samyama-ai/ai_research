---
id: 07-embeddings/ann-recall-adversarial-distributions
title: "Approximate Nearest Neighbor Recall Under Adversarial Distributions"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Approximate Nearest Neighbor Recall Under Adversarial Distributions

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/ann-recall-adversarial-distributions` · **Status:** open

## 1. Problem Statement

Every production vector index — HNSW, IVF-PQ, DiskANN, ScaNN — is tuned and reported on *average-case* recall over a fixed query set drawn from the same distribution as the base set. The question is what happens when that assumption fails: when the query distribution is different from the base distribution (out-of-distribution, OOD), when the base set is chosen or poisoned by an adversary, or when queries are chosen adaptively after seeing the index.

Three variants, of very different difficulty:

- **Measurement.** Define a recall guarantee that is *distribution-free*: a number $\rho$ such that the index returns the true $k$ nearest neighbors with probability $\ge \rho$ for **every** query, not for a query sampled from the test file. No current benchmark reports such a number, and it is not obvious what the right conditioning is.
- **Method.** Build an index with sublinear query time whose recall degrades gracefully — not catastrophically — under (a) OOD queries, (b) an adaptive adversary issuing $Q$ queries and observing results, (c) insertion of $m \ll n$ adversarial base points.
- **Theory.** Prove or refute: for graph-based indices with $O(\log n)$ average degree built by greedy/robust-prune heuristics, is there a family of instances on $n$ points in $\mathbb{R}^d$ where greedy search visits $\Omega(n^{\alpha})$ nodes, $\alpha > 0$, before reaching the true nearest neighbor? (Partially answered — see §4.)

Solving it means: an index that reports a worst-case recall certificate, and a benchmark that measures it.

## 2. Formal Setting

Base set $P = \{p_1,\dots,p_n\} \subset \mathbb{R}^d$, metric $\mathrm{dist}$ (Euclidean or angular). For query $q$, let $N_k(q)$ be the true $k$ nearest neighbors and $\hat N_k(q)$ the index's output.

**Recall, as measured.** ANN-Benchmarks computes
$$\mathrm{rec}@k(q) = \frac{|\hat N_k(q) \cap N_k(q)|}{k}, \qquad \overline{\mathrm{rec}} = \frac{1}{|Q_{\text{test}}|}\sum_{q \in Q_{\text{test}}} \mathrm{rec}@k(q),$$
with $|Q_{\text{test}}|$ typically 1,000–10,000 held-out vectors and ground truth by exact brute force. This is a Monte Carlo estimate of $\mathbb{E}_{q \sim \mathcal{D}_Q}[\mathrm{rec}@k]$, nothing more.

**The quantity that is actually wanted.**
$$\rho_{\min} = \inf_{q \in \mathbb{R}^d} \mathrm{rec}@k(q), \qquad \rho_\epsilon = \sup\{\rho : \Pr_{q\sim\mathcal{D}}[\mathrm{rec}@k(q) \ge \rho] \ge 1-\epsilon\}.$$
$\rho_{\min}$ is 0 for every deployed graph index (a query can be planted in a region the graph does not connect). $\rho_\epsilon$ at $\epsilon = 10^{-4}$ needs $\ge 10^5$ queries to estimate and is essentially never reported.

**Cost.** Distance computations $C(q)$ per query, or the *visited fraction* $C(q)/n$. Latency is a proxy contaminated by cache and SIMD effects; distance count is the portable measure.

**Distribution shift.** Let $\mathcal{D}_P$ be the base distribution and $\mathcal{D}_Q$ the query distribution. In-distribution benchmarks set $\mathcal{D}_Q = \mathcal{D}_P$. OOD is the real case: in Text2Image-1B the base set is image embeddings and queries are text embeddings from a different encoder tower, so $\mathcal{D}_Q \ne \mathcal{D}_P$ by construction.

**Local intrinsic dimensionality (LID)** at $q$, the Hill estimator over the $j$ nearest distances $r_1 \le \dots \le r_j$:
$$\widehat{\mathrm{LID}}_j(q) = -\left(\frac{1}{j}\sum_{i=1}^{j}\ln \frac{r_i}{r_j}\right)^{-1}.$$
This is the standard per-query hardness proxy. It is an estimator with heavy variance at small $j$; treat it as measured, not exact.

**Assumptions known to be violated in practice.**
1. *Queries are i.i.d. from the test file.* False in RAG, recommendation, and dedup, where queries are user- or attacker-controlled and repeated.
2. *The base set is fixed and trusted.* False under corpus poisoning — anyone who can write a document can insert base vectors.
3. *Bounded doubling dimension.* Real embedding sets have LID varying by 3–5× across regions of the same corpus; a single global constant does not hold.
4. *Ground truth is available.* True at benchmark scale (brute force over $10^9$ is affordable once), false for the adversarial variant, where the adversary's optimum is itself an optimization problem.

## 3. State of the Art

**Theory SOTA.** Andoni–Laarhoven–Razenshteyn–Waingarten (SODA 2017) give optimal time–space trade-offs for data-dependent LSH; $c$-ANN in $\ell_2$ with query exponent $\rho = 1/(2c^2-1)$. These bounds are worst-case over *data* and hold under adversarial input — this is their point, and it is the only branch of the field with a genuine distribution-free guarantee. Rubinstein (STOC 2018) shows conditional hardness: under SETH, $(1+\epsilon)$-ANN in $\ell_2$ cannot be solved in truly subquadratic total time for $n$ queries as $\epsilon \to 0$, ruling out a "fast and exact-enough for all inputs" index.

For graph indices, Indyk & Xu (NeurIPS 2023) is the key result: the *slow-preprocessing* variant of DiskANN has a proved guarantee on datasets of bounded doubling dimension, while HNSW, NSG, and fast-preprocessing DiskANN admit constructed instances with no such guarantee. Prokhorenkova & Shekhovtsov (ICML 2020) give a graph-search analysis on the sphere with an $O(n^{\rho})$-type dependence, but under a uniform-on-sphere assumption that no embedding set satisfies.

**Empirical SOTA.** HNSW and DiskANN dominate ANN-Benchmarks and the Big-ANN NeurIPS'21 and NeurIPS'23 competitions at recall@10 $\ge 0.9$. On OOD specifically, OOD-DiskANN (Jaiswal et al., 2022) builds the graph using *sample queries* rather than base points as navigation targets and reports large latency reductions at fixed recall on Text2Image-1B; the NeurIPS'23 OOD track produced further entries in the same family.

**Established vs. claimed.** Established: relative ranking of index families at fixed in-distribution recall, reproduced across independent ANN-Benchmarks runs; the Indyk–Xu separation, which is a theorem. Claimed but unablated: that OOD-aware construction improves *worst-case* rather than average recall — the reported numbers are mean recall on one query file (Text2Image queries), so they are benchmark numbers, not robustness results. No index ships a worst-case recall certificate.

## 4. What Is Known

- **Sublinear-time distribution-free ANN exists, at a price.** Data-dependent LSH: query exponent $\rho = 1/(2c^2-1)$ for $c$-approximation in $\ell_2$ (Andoni et al., SODA 2017). At $c = 2$ this is $\rho = 1/7$ — but the constant factors and the $n^{1+\rho}$ space make it uncompetitive with graphs on $10^9$-point benchmarks.
- **Greedy graph search has no unconditional guarantee.** Indyk & Xu (NeurIPS 2023) exhibit instances where popular implementations fail to achieve their claimed behavior, and prove the positive result only for slow-preprocessing DiskANN under bounded doubling dimension.
- **OOD costs recall in practice.** Big-ANN'21 introduced Text2Image-1B ($n = 10^9$, $d = 200$) precisely because in-distribution indices did poorly on it; the OOD track was retained in the 2023 competition. Scale: $10^9$ base, $10^5$ query.
- **Per-query hardness tracks LID.** Aumüller & Ceccarello (Information Systems, 2021) show high-LID queries have systematically lower recall at fixed budget; on standard million-scale sets (GloVe-100, SIFT-1M) the hardest LID decile can require an order of magnitude more distance computations than the median for equal recall.
- **A tiny adversarial base insertion changes retrieval.** Zhong et al. (EMNLP 2023) show ~50 adversarially optimized passages inserted into a corpus of millions can be retrieved for a large fraction of held-out queries by dense retrievers. That is a statement about the *embedding geometry*, and it is exactly the regime where the index's recall claim is also unaudited.
- **Beyer et al. (ICDT 1999):** under broad conditions, as $d \to \infty$ the ratio of farthest to nearest distance $\to 1$, so "nearest neighbor" loses meaning. Modern embeddings survive this only because their intrinsic dimension is far below $d$ — a property an adversary can attack.

## 5. What Is Not Known

- **Theoretically open.** Whether HNSW-style hierarchical greedy search on $n$ points admits a *lower* bound $\Omega(n^\alpha)$ visited nodes for a natural instance family, versus being merely unproved-above. Also open: whether any index with $O(n)$ space and $\mathrm{polylog}(n)$ query time can give $\rho_{\min} > 0$ for exact $k$-NN under adversarial data (Rubinstein's SETH bound constrains but does not close this).
- **Empirically open.** Nobody has measured $\rho_\epsilon$ at $\epsilon = 10^{-4}$ for a production index at $10^8$–$10^9$ scale. The experiment is entirely runnable: it costs one brute-force ground-truth pass over $10^6$ adversarially chosen queries. It has not been run because benchmarks reward mean recall.
- **Empirically open.** Whether an *adaptive* adversary — one who issues queries, observes returned IDs, and adapts — can drive recall@10 to near zero on HNSW within $10^3$ queries. Plausible, unmeasured.
- **Methodologically blocked.** The adversarial recall metric itself. If the adversary may place base points, "true nearest neighbor" is defined relative to a corpus the adversary shaped, so recall can be 1.0 while retrieval is entirely attacker-controlled. Recall is the wrong name for the thing that matters, and no accepted replacement exists.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names**, compounded by **absent ground truth in the adversarial regime**.

"Recall@10 = 0.95" names a property of the index but measures a property of the *(index, query file)* pair. The tail is unmeasured by construction: with $10^4$ test queries you cannot resolve behavior below $\epsilon = 10^{-3}$, and the failures that matter in deployment — a poisoned region, one adaptively-found blind spot — live at $\epsilon \le 10^{-5}$. Increasing $|Q_{\text{test}}|$ does not help unless the queries are drawn adversarially, and drawing them adversarially requires exact ground truth, which costs $O(nd)$ per query: at $n = 10^9$, $d = 200$, one exact query is ~$2\times10^{11}$ FLOPs, so $10^6$ certificate queries is ~$2\times10^{17}$ FLOPs — GPU-days, not minutes. That is the compute wall behind the empirical gap.

Non-identifiability makes it worse: when recall drops on an OOD query set, the cause could be the index's routing, the encoder's geometry (query and base towers occupying different cones), or genuinely higher LID. Current benchmarks cannot separate the three because they vary together.

## 7. Current Research (as of 2026)

- **OOD-aware graph construction.** The OOD-DiskANN line (Microsoft Research India and collaborators) and successors from the Big-ANN'23 OOD track. Direction: use query samples during construction. Open question they do not answer: robustness to a query distribution unseen at build time.
- **Worst-case analysis of practical graphs.** Indyk & Xu (MIT) and follow-ups; the live question is tight lower bounds for HNSW rather than existence of bad instances. *(frontier — verify)*
- **Retrieval poisoning and RAG security.** PoisonedRAG (Zou et al., USENIX Security 2025) and the corpus-poisoning line; mostly treats the retriever as exact, so index-level effects are unstudied. Merging this with ANN robustness is the obvious unclaimed intersection.
- **Filtered and streaming ANN** (Big-ANN'23 tracks) — predicate filtering creates effectively adversarial sub-distributions, since a filter can isolate an arbitrary sparse subset of the graph. Whether recall guarantees survive filtering is actively studied and unresolved. *(frontier — verify)*
- **Certified retrieval.** Early proposals to attach per-query certificates (e.g. a proof that no unvisited region can contain a closer point). No scalable system yet. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does adversarial query selection reduce HNSW recall@10 by more than distribution shift alone?

**Scale.** DEEP-100M or Text2Image-100M ($n = 10^8$, $d = 96$/$200$). One HNSW index at $M=32$, $\mathrm{efC}=500$, and one slow-preprocessing DiskANN index at matched build cost. Query budget: $10^5$ queries per arm, exact ground truth by GPU brute force (~$10^{16}$ FLOPs total; hours on 8 A100s).

**Arms.**
1. *Control A (in-distribution):* $10^5$ held-out base-distribution queries. Expected $\overline{\mathrm{rec}}@10 \approx 0.95$ at $\mathrm{ef}=100$.
2. *Control B (natural OOD):* $10^5$ genuine text-tower queries. Measures shift without adversary.
3. *Treatment (adaptive adversary):* a search that, given black-box access returning only IDs, does local perturbation on query vectors to minimize recall, with a budget of 100 index probes per adversarial query. Constrain adversarial queries to the convex hull of the base set to prevent trivial off-manifold wins.

**The deciding number.** $\rho_{0.01}$ — the 1st-percentile per-query recall@10 — in each arm, at fixed distance-computation budget $C = 2{,}000$ per query. If treatment $\rho_{0.01}$ is $< 0.2$ while Control B is $> 0.7$, adversarial degradation is a distinct phenomenon from distribution shift and worst-case ANN needs its own metric and defenses. If the two are within 0.1 of each other, the problem collapses into OOD robustness and needs no separate research program. Report LID of adversarial queries alongside, to check the adversary is not merely a high-LID sampler.

## 9. Key References

- **[Foundational]** Piotr Indyk, Rajeev Motwani. *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality.* STOC, 1998.
- **[Foundational]** Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, Uri Shaft. *When Is "Nearest Neighbor" Meaningful?* ICDT, 1999.
- **[Theory SOTA]** Alexandr Andoni, Thijs Laarhoven, Ilya Razenshteyn, Erik Waingarten. *Optimal Hashing-based Time-Space Trade-offs for Approximate Near Neighbors.* SODA, 2017. — arXiv:1608.03580
- **[Theory SOTA]** Aviad Rubinstein. *Hardness of Approximate Nearest Neighbor Search.* STOC, 2018. — arXiv:1803.00904
- **[SOTA]** Piotr Indyk, Haike Xu. *Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations.* NeurIPS, 2023. — arXiv:2310.19126
- **[SOTA]** Yu. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 2020. — arXiv:1603.09320
- **[SOTA]** Suhas Jayaram Subramanya, Devvrit, Rohan Kadekodi, Ravishankar Krishnaswamy, Harsha Vardhan Simhadri. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS, 2019.
- **[SOTA]** Shikhar Jaiswal, Ravishankar Krishnaswamy, Ankit Garg, Harsha Vardhan Simhadri, Sheshansh Agrawal. *OOD-DiskANN: Efficient and Scalable Graph ANNS for Out-of-Distribution Queries.* 2022. — arXiv:2211.12850
- **[Benchmark]** Martin Aumüller, Erik Bernhardsson, Alexander Faithfull. *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* Information Systems, 2020. — arXiv:1807.05614
- **[Benchmark]** Harsha Vardhan Simhadri et al. *Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search.* PMLR (NeurIPS Competition Track), 2022. — arXiv:2205.03763
- **[Benchmark]** Harsha Vardhan Simhadri et al. *Results of the Big ANN: NeurIPS'23 Competition.* 2024. — arXiv:2409.17424
- **[Analysis]** Martin Aumüller, Matteo Ceccarello. *The Role of Local Dimensionality Measures in Benchmarking Nearest Neighbor Search.* Information Systems, 2021.
- **[Analysis]** Liudmila Prokhorenkova, Aleksandr Shekhovtsov. *Graph-based Nearest Neighbor Search: From Practice to Theory.* ICML, 2020. — arXiv:1907.00845
- **[Adversarial]** Zexuan Zhong, Ziqing Huang, Alexander Wettig, Danqi Chen. *Poisoning Retrieval Corpora by Injecting Adversarial Passages.* EMNLP, 2023. — arXiv:2310.19156
- **[Adversarial]** Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia. *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models.* USENIX Security, 2025. — arXiv:2402.07867
- **[Survey]** Alexandr Andoni, Piotr Indyk, Ilya Razenshteyn. *Approximate Nearest Neighbor Search in High Dimensions.* Proceedings of the ICM, 2018. — arXiv:1806.09823

## 10. Worked Example

Take SIFT-1M: $n = 10^6$, $d = 128$, HNSW with $M = 16$, $\mathrm{efC} = 200$, query $\mathrm{ef} = 64$. Reported mean recall@10 on the standard 10,000-query file is about 0.97 at roughly 1,500 distance computations per query — a number reproduced many times on ANN-Benchmarks.

Now decompose that mean instead of reporting it. With 10,000 queries and $\overline{\mathrm{rec}} = 0.97$, the missing 0.03 is $3{,}000$ missed neighbors. Two very different worlds produce the same headline:

- **World A:** every query misses 0.3 of one neighbor on average — recall is uniformly 0.97, nothing is broken.
- **World B:** 9,700 queries have recall 1.0 and 300 queries have recall 0.0. Mean is still 0.97. But $\rho_{0.01}$, the 1st percentile, is 0.0 — three percent of the query space is a blind spot.

The benchmark reports the same 0.97 in both cases and never distinguishes them. In practice SIFT-1M sits closer to World A, but that is a property of a query file drawn from the base distribution, not of the index.

Now make the obstruction visible. Suppose an adversary inserts $m = 100$ points into the $10^6$ base set, placed to be reachable only through a single high-degree hub node that greedy search enters late. The insertion changes the index's *mean* recall by at most $m/n = 10^{-4}$ measured on the original query file — invisible, well inside run-to-run noise. But for queries near the inserted cluster, recall@10 goes to near zero because greedy search terminates before crossing the hub. To detect this you must sample queries near those 100 points; the chance a uniform 10,000-query sample lands there is about $10^{-4} \cdot 10^4 = 1$ query. One query. Even if it is sampled, its contribution to the mean is $10^{-4}$ — below reporting precision.

That is the whole problem in one calculation: **the failure mode is $10^4$ times smaller than the metric's resolution, and $10^4$ times more important than its average.** Fixing it requires either adversarially drawn queries with exact ground truth (compute-bound: $O(nd)$ per query) or a per-query certificate no deployed index produces.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*