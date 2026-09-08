---
id: 05-retrieval-and-agents/cost-optimal-retrieval-generation-allocation
title: "Cost-Optimal Allocation Between Retrieval and Generation"
topic: 05-retrieval-and-agents
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cost-Optimal Allocation Between Retrieval and Generation

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/cost-optimal-retrieval-generation-allocation` · **Status:** empirically-open

## 1. Problem Statement

A retrieval-augmented system spends a fixed budget across four sinks: datastore construction (indexing, embedding, chunking), retrieval at query time (ANN search, reranking), context occupancy (prefill tokens), and generation (decode tokens, sampling, tool loops). The question: **given a total cost budget $C$, what split across these sinks maximizes task accuracy — and does that split follow a stable law, as Chinchilla's does for parameters versus training tokens?**

Three variants, of very different difficulty:

- **Measurement variant.** Can the marginal accuracy per dollar of retrieval and of generation be measured on the same axis, at fixed everything else? Currently blocked by the absence of a shared cost unit (index build is amortized capital; decode is per-query marginal).
- **Method variant.** Build a router/controller that picks per query how much to retrieve and how much to generate, beating any fixed allocation on an accuracy-vs-cost Pareto frontier. Partially solved for two-point choices (retrieve/don't, small model/large model).
- **Theory variant.** Is there an exponent law $A \approx A_\infty - a C_R^{-\alpha} - b C_G^{-\beta}$ with $\alpha,\beta$ stable across tasks, such that the optimum is $C_R/C_G = \text{const}$? Open; no derivation, no falsification.

Solved would mean: a fitted allocation rule that predicts the optimal split on a held-out task family within the noise of the accuracy metric.

## 2. Formal Setting

Query $q \sim \mathcal{D}$, datastore $\mathcal{S}$ of $N$ tokens, retriever $R_\theta$ returning $k$ chunks of $\ell$ tokens each, generator $G_\phi$ producing answer $y$ under a test-time compute policy (samples $s$, reasoning tokens $r$, tool-call rounds $t$).

**Cost, as actually metered:**

$$C(q) = \underbrace{\frac{C_{\text{idx}}(N)}{Q}}_{\text{amortized}} + \underbrace{c_{\text{ann}}\log N + k\,c_{\text{rerank}}}_{\text{retrieval}} + \underbrace{p_{\text{in}}\,(k\ell + |q|)}_{\text{prefill}} + \underbrace{p_{\text{out}}\, s\,(r + |y|)}_{\text{decode}}$$

Every term is measurable: $C_{\text{idx}}$ as GPU-hours to embed and build the HNSW/IVF index; $Q$ as the expected query count over the index's lifetime before refresh; $p_{\text{in}}, p_{\text{out}}$ as \$/Mtok list prices or as measured joules/token under a fixed serving stack (vLLM, fixed batch size). Report both: prices move, FLOPs do not.

Define $C_R$ = amortized index + retrieval + prefill attributable to retrieved tokens; $C_G$ = decode + prefill attributable to the prompt and generated reasoning. Accuracy $A(C_R, C_G) = \mathbb{E}_{q}[\,\text{score}(y,q)\,]$ under a fixed grader.

**Objective:** $\max A$ s.t. $\mathbb{E}_q[C(q)] \le C$. At an interior optimum with differentiable $A$, the KKT condition is equal marginal returns:

$$\frac{\partial A}{\partial C_R} = \frac{\partial A}{\partial C_G}.$$

**Assumptions, and which break:**
1. *Separability* — $A$ decomposes into independent retrieval and generation terms. **Violated**: long contexts degrade generation (position effects, Liu et al. 2024), so the cross term is real and signed negative.
2. *Monotonicity in $k$* — more retrieved chunks never hurt. **Violated**: distractor passages reduce accuracy for high-popularity entities (Mallen et al. 2023).
3. *Amortization is well posed* — $Q$ is known. **Violated** for freshness-sensitive corpora where the index is rebuilt on an unknown schedule.
4. *Stationary $\mathcal{D}$* — query mix fixed. **Violated** under adaptive users and agentic self-generated subqueries.
5. *Grader is cost-independent.* **Violated** when an LLM judge favors longer, retrieval-cited answers.

## 3. State of the Art

**Established.**
- *Datastore scaling.* Shao et al., *Scaling Retrieval-Based Language Models with a Trillion-Token Datastore* (NeurIPS 2024, arXiv:2407.12854) build MassiveDS (1.4T tokens) and fit compute-optimal curves showing a smaller LM with a larger datastore can dominate a larger LM at equal total compute on knowledge-intensive QA. This is the closest thing to a retrieval-vs-parameters scaling law, and it is ablated across datastore sizes.
- *Routing works at two points.* FrugalGPT (Chen, Zaharia, Zou, arXiv:2305.05176) and RouteLLM (Ong et al., arXiv:2406.18665) show cascades reach comparable quality at large cost reductions; Hybrid LLM (Ding et al., ICLR 2024) does the quality-aware routing version.
- *Retrieval is not always positive.* Mallen et al. (ACL 2023) — adaptive retrieval keyed on entity popularity beats always-retrieve on PopQA.

**Claimed but unablated.**
- Self-Route (Li et al., EMNLP 2024 Industry, arXiv:2407.16833) routes between RAG and long-context and reports large cost reductions at comparable accuracy — but the router threshold's sensitivity, and the result's transfer off LongBench/∞Bench, are not ablated.
- Adaptive-RAG (Jeong et al., NAACL 2024) trains a query-complexity classifier; the classifier's labels come from the pipeline's own success, which is circular.
- GraphRAG (Edge et al., arXiv:2404.16130) reports quality gains for global sensemaking; the index-build cost is reported, but no Pareto comparison against spending the same compute on generation.

**Benchmark-number-only.** Anthropic's contextual retrieval result (49% reduction in retrieval failure rate; blog, 2024) is a vendor engineering report, not a peer-reviewed ablation.

## 4. What Is Known

- **Datastore size buys accuracy sublinearly and cheaply.** MassiveDS: retrieval-based LMs improve monotonically as the datastore grows toward $1.4\times10^{12}$ tokens on knowledge tasks, with no observed saturation; measured at LM scales up to 8B.
- **Context length is not free accuracy.** Lost-in-the-Middle (Liu et al., TACL 2024): accuracy on multi-document QA drops by roughly 20 points between gold-document-first and gold-document-middle placements, at 10–30 documents, measured on GPT-3.5-class and Claude-1-class models.
- **Test-time compute substitutes for parameters, within limits.** Snell et al. (arXiv:2408.03314): on MATH with PaLM-2-S*, optimal test-time-compute allocation beats a best-of-$N$ baseline at up to $4\times$ less compute, and can beat a $14\times$ larger model on easy/medium questions — but not on the hardest bin.
- **Retrieval can hurt.** PopQA: for entities above a popularity threshold, non-parametric retrieval underperforms the parametric model; measured with GPT-3 davinci-003 and smaller open models.
- **Retrieval cost is dominated by prefill, not search.** HNSW search over $10^8$ vectors is $\mathcal{O}(\text{ms})$; 20 chunks × 400 tokens = 8k prefill tokens, which at typical \$/Mtok is 1–2 orders of magnitude more than the ANN query itself.

## 5. What Is Not Known

- **Theoretically open.** Whether $A(C_R,C_G)$ admits a Chinchilla-style power-law fit with transferable exponents. No one has proposed and falsified a functional form. The cross term (context dilution) means the standard additive-separable ansatz is likely wrong, and no alternative is fitted.
- **Empirically open.** The full 2-D sweep — datastore size × retrieval depth × sampling budget × reasoning length, at matched total FLOPs, on a task family with clean grading — is runnable today on a few thousand GPU-hours. Nobody has published it. All existing work sweeps one axis with the others pinned.
- **Methodologically blocked.** A single cost unit that legitimately compares amortized index build against per-query decode. $Q$ (queries per index lifetime) is a business parameter, not a scientific one, and the optimum swings by orders of magnitude across plausible $Q$. Until pages report the $Q$ they assume, cross-paper cost comparisons are not meaningful.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus a free parameter that dominates the answer**.

1. *Amortization is a hidden knob.* Optimal $C_R/C_G$ is a function of $Q$. Doubling assumed index lifetime halves the effective retrieval cost. Papers rarely state $Q$, so reported "cost-optimal" points are not comparable.
2. *Non-identifiability of the cross term.* Adding retrieved tokens simultaneously adds evidence (helps) and distractors plus positional dilution (hurts). Observing net accuracy at each $k$ does not separate the two effects; you cannot fit $\partial A/\partial C_R$ without an intervention that adds tokens with zero evidence content (a distractor-only control), which almost no study runs.
3. *Grader-cost coupling.* LLM judges reward citation-bearing answers; that inflates the measured return on retrieval by an amount nobody has quantified.
4. *Price non-stationarity.* Inference \$/Mtok has fallen faster than embedding and storage costs; an allocation fitted in 2024 dollars does not transfer to 2026 dollars. FLOP-denominated results transfer; dollar-denominated ones do not.

## 7. Current Research (as of 2026)

- **Datastore scaling laws** — follow-ups to MassiveDS extending to reasoning-heavy tasks and to multi-hop retrieval (UW / AI2 lineage).
- **Cost-aware routing** — Berkeley LMSYS-lineage routing work (RouteLLM), Microsoft (Hybrid LLM), and vendor "router" products that pick model and retrieval depth per query. *(frontier — verify)* Several 2025–2026 systems papers claim joint model-and-retrieval routing; ablations separating the two decisions are scarce.
- **Agentic search vs. static RAG** — deep-research agents trade many cheap retrieval rounds against long reasoning traces; the allocation question reappears as "search rounds vs. thinking tokens" and is largely untuned. *(frontier — verify)*
- **Context caching as a third sink** — prefix caching makes retrieved-context cost sub-linear in reuse, which shifts the optimum toward retrieval for repeated corpora. Under-studied as an allocation variable.

## 8. Concrete Next Experiment

**Scale.** One 8B open-weight model, fixed serving stack (vLLM, fixed batch, measured joules/token). Datastore: Wikipedia + C4 subsets at $\{10^7, 10^8, 10^9, 10^{10}\}$ tokens. Tasks: Natural Questions + HotpotQA + PopQA, 3,000 queries each, exact-match grading only (no LLM judge — removes grader-cost coupling).

**Grid.** Retrieval depth $k \in \{0,1,2,5,10,20,50\}$; generation budget = self-consistency samples $s \in \{1,2,4,8\}$ × reasoning cap $r \in \{128, 512, 2048\}$. Meter every cell in FLOPs and in joules. Amortize index build at three explicit lifetimes $Q \in \{10^4, 10^6, 10^8\}$.

**Control arms.** (a) **Distractor-only arm**: retrieve $k$ chunks, then replace the gold-containing chunk with a topically matched non-answer chunk. This isolates the dilution term from the evidence term — the missing intervention in every prior sweep. (b) **Fixed-allocation baseline**: the single best $(k,s,r)$ triple, chosen on a dev split, applied to all queries.

**The deciding number.** Fit iso-cost contours and report $\rho^\star = C_R^\star / C_G^\star$ at each budget. **If $\rho^\star$ varies by less than a factor of 2 across the four datastore sizes, three tasks, and three $Q$ values, an allocation law exists and is worth naming. If it varies by more than $10\times$ — particularly if it flips with $Q$ — the law does not exist and per-query routing is the only correct framing.** Secondary number: the dilution slope $\partial A/\partial k$ from the distractor-only arm, in accuracy points per 1k prefill tokens.

## 9. Key References

- **[Foundational]** Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS, 2020. — arXiv:2005.11401
- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Shao, He, Asai, Shi, Dettmers, Min, Zettlemoyer, Koh. *Scaling Retrieval-Based Language Models with a Trillion-Token Datastore.* NeurIPS, 2024. — arXiv:2407.12854
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Li, Li, Lee, Zhang, Lee, Zhang, et al. *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach.* EMNLP Industry Track, 2024. — arXiv:2407.16833
- **[Method]** Chen, Zaharia, Zou. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* 2023. — arXiv:2305.05176
- **[Method]** Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica. *RouteLLM: Learning to Route LLMs with Preference Data.* 2024. — arXiv:2406.18665
- **[Method]** Jeong, Baek, Cho, Hwang, Park. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.* NAACL, 2024. — arXiv:2403.14403
- **[Empirical]** Mallen, Asai, Zhong, Das, Khashabi, Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL, 2023. — arXiv:2212.10511
- **[Empirical]** Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Survey]** Gao, Xiong, Gao, Jia, Pan, Bi, Dai, Sun, Wang, Wang. *Retrieval-Augmented Generation for Large Language Models: A Survey.* 2023. — arXiv:2312.10997

## 10. Worked Example

A support-QA deployment. Corpus $N = 5\times10^8$ tokens. Embedding at 0.3 GFLOP/token → index build $\approx 1.5\times10^{17}$ FLOPs. Serving an 8B model: prefill $\approx 2\times8\times10^9 = 1.6\times10^{10}$ FLOP/token; decode the same per token.

Two candidate configurations at roughly equal per-query cost:

| | $k$ | prefill tokens | samples $s$ | decode tokens | retrieval FLOPs/query | generation FLOPs/query |
|---|---|---|---|---|---|---|
| A (retrieval-heavy) | 20 × 400 | 8,000 | 1 | 300 | $1.3\times10^{14}$ | $4.8\times10^{12}$ |
| B (generation-heavy) | 3 × 400 | 1,200 | 4 | 4×300 | $1.9\times10^{13}$ | $1.9\times10^{13}$ |

Now amortize the index. At $Q = 10^6$ queries, index build adds $1.5\times10^{11}$ FLOPs/query — 0.1% of A's retrieval cost, negligible. At $Q = 10^4$, it adds $1.5\times10^{13}$ FLOPs/query, which **exceeds all of B's generation compute** and roughly doubles B's total. The same accuracy numbers therefore rank A above B at $Q=10^6$ and B above A at $Q=10^4$, with no change to the model, the corpus, or the questions.

That is the obstruction in one table. The ranking is decided by $Q$ — a deployment parameter almost never reported — not by anything the experiment measured. And the accuracy side is equally underdetermined: A's 8,000 prefill tokens carry both the answer and 19 distractors, and without the distractor-only control arm there is no way to attribute A's observed accuracy to evidence rather than to the model's tolerance for dilution. Two free parameters, one reported number, no identification.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*