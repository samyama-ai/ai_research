---
id: 26-code-generation/repository-context-selection-token-budget
title: "Repository Context Selection Under a Fixed Token Budget"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Repository Context Selection Under a Fixed Token Budget

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/repository-context-selection-token-budget` · **Status:** empirically-open

## 1. Problem Statement

Given a repository $R$ (thousands of files, often $10^6$–$10^8$ tokens), a task $q$ (complete this line, fix this issue, add this feature), and a hard budget of $B$ prompt tokens, choose the subset of repository material to place in the prompt so that the generated patch is correct.

Three variants, routinely conflated:

- **Measurement.** Is there a well-defined *oracle* context set — a minimal $S^\star \subseteq R$ with $|S^\star| \le B$ under which the model succeeds? Can it be identified from the ground-truth patch, and is it stable across models?
- **Method.** Build a selector $\pi: (R, q, B) \mapsto S$ that beats the strongest cheap baselines (BM25 over sliding-window chunks; the file the cursor is in plus its imports; "put the whole repo in a 1M-token window").
- **Theory.** Under what structure on the utility $U(S)$ is budgeted selection tractable? If $U$ is monotone submodular, greedy gives $1-1/e$ (Nemhauser–Wolsey–Fisher 1978) and $1-1/e$ is optimal for max-coverage unless P = NP (Feige, JACM 1998). Code utility is **not** submodular: a call site is worthless without its callee's signature, so marginal gains can *increase* — a complementarity that breaks every coverage-style guarantee.

Solving it means: a selector that, at fixed $B$, matches or beats full-repository context on a held-out execution-based benchmark, with an ablation isolating *selection* from retrieval-index quality, model long-context ability, and prompt ordering.

## 2. Formal Setting

Repository as a set of atomic snippets $R = \{c_1, \dots, c_N\}$ (function, class, or fixed-size chunk), each with token cost $t_i = |\mathrm{tok}(c_i)|$ measured by the *deployed* tokenizer, not a proxy. Budget $B$ counts prompt tokens *after* template, system prompt, and the in-file prefix/suffix, i.e. the free budget $B = B_{\text{ctx}} - B_{\text{fixed}} - B_{\text{gen}}$.

A selection is $S \subseteq [N]$ with a rendering $\rho(S)$ (order, file headers, elisions). Utility is the *execution* probability:

$$U(S) \;=\; \mathbb{E}_{y \sim p_\theta(\cdot \mid \rho(S), q)}\big[\mathbb{1}[\,\mathrm{tests}(y) = \text{pass}\,]\big]$$

estimated as $\hat U(S) = \frac{1}{k}\sum_{j=1}^k \mathbb{1}[\text{pass}]$ over $k$ samples at temperature $T$; standard error $\sqrt{\hat U(1-\hat U)/k}$, so $k \ge 10$ per instance is needed to resolve 5-point differences on a 500-instance set. The problem:

$$S^\star \;=\; \arg\max_{S \subseteq [N]} U(\rho(S)) \quad \text{s.t.} \quad \sum_{i \in S} t_i \le B$$

**Efficiency frontier.** Report $U$ as a function of $B$, not at one point: $\mathcal{F}(B) = \max_{|S|\le B} \hat U(S)$. A selector is only interesting if it dominates baselines across $B \in \{2\text{k}, 8\text{k}, 32\text{k}, 128\text{k}\}$.

**Oracle-recall surrogate.** Where execution is unavailable, define $\mathrm{Rec}@B = |S \cap G| / |G|$ for $G$ the set of snippets whose spans overlap the ground-truth patch or its one-hop callees. This is the standard proxy and it is a *bound, not a measure*: high recall with bad ordering can score worse than low recall.

**Assumptions, and which are violated.**
1. *Additive cost.* $\text{cost}(S) = \sum t_i$ — violated by tokenizer merges across concatenation boundaries (small, $<1\%$) and by rendering overhead (headers, separators: 5–15%).
2. *Order-invariance.* $U$ depends on $S$, not $\rho$ — **badly violated**; "Lost in the Middle" (Liu et al., TACL 2024) shows accuracy drops when the needed item sits mid-context, and repo-completion work finds the nearest-cursor snippet should sit last.
3. *Submodularity.* Violated by definitional complementarity (see §1).
4. *Stationary repository.* Benchmarks freeze a commit; real selection happens against a dirty working tree with a half-written edit.
5. *Model-independence of $S^\star$.* Assumed by every shared retrieval corpus; untested.

## 3. State of the Art

**Established (ablated, reproduced):**
- Sliding-window BM25 / lexical retrieval over the repo beats no-retrieval by a wide margin on infilling. RepoCoder (Zhang et al., EMNLP 2023) formalizes iterative retrieval-generation — retrieve, draft, re-retrieve using the draft — and shows consistent gains over one-shot retrieval on RepoEval.
- Cross-file context matters more than model size at the margin: CrossCodeEval (Ding et al., NeurIPS 2023 D&B) reports large exact-match gains from adding retrieved cross-file context across Python, Java, TypeScript, C#, and shows the gap to an oracle-retrieval ceiling remains open.
- Structure helps over pure lexical similarity. DraCo (Cheng et al., ACL 2024) uses dataflow-guided retrieval on an extended dependency graph; repo-level prompt generation with static-analysis-proposed context sources (Shrivastava, Larochelle, Tarlow, ICML 2023) beats a Codex baseline by learning *which* proposal to use.

**Claimed but unablated:**
- That agentic search (an LLM issuing `grep`/`find` calls) beats static retrieval. Agentless (Xia et al., 2024) is the honest counterpoint: a fixed three-phase localize→repair→validate pipeline is competitive with agent scaffolds on SWE-bench Verified at a fraction of the cost, which suggests the agent's advantage is in *localization*, not open-ended search.
- That long-context models make selection obsolete. Reported as benchmark deltas only; the compute-matched comparison (128k of retrieved context vs. 128k of contiguous repo dump, same model, same $k$) is rarely published.

**Benchmark-number-only results:** most SWE-bench leaderboard entries. They vary scaffold, model, retrieval, and reranking simultaneously; no line item attributes score to context selection.

## 4. What Is Known

- **Oracle gap is large.** On CrossCodeEval (~10k instances, 4 languages), retrieval-augmented completion improves substantially over in-file-only, yet the reported oracle/gold-context arm remains clearly above the best retriever — the selection problem is unsolved, not saturated, at 2k–8k budgets.
- **Position effects are first-order.** TACL 2024 ("Lost in the Middle", Liu et al.) measures a U-shaped curve over document position in 20-document QA; the same shape has been reported informally for code. Consequence: two selectors with identical $\mathrm{Rec}@B$ can differ by several points of pass rate purely by ordering.
- **Iteration beats one-shot at fixed budget.** RepoCoder's gains hold with the *same* $B$, so the win is from a better query (the draft), not more tokens.
- **Localization dominates repair on issue-level tasks.** On SWE-bench Verified (500 human-validated instances, subset of the 2,294-instance SWE-bench, Jimenez et al., ICLR 2024), pipelines given gold *file* localization score far above the same pipeline doing its own file retrieval. The bottleneck is finding the file, not editing it.
- **Repo-map heuristics are strong baselines.** Tree-sitter symbol extraction ranked by PageRank over the reference graph (the `aider` repo map, open-source, widely used) delivers a usable whole-repo skeleton in a few thousand tokens. It is rarely used as a control arm in papers.

## 5. What Is Not Known

- **Empirically open.** Does any selector dominate whole-repo dumping at equal *cost* (tokens × price), across $B$? Runnable today: needs ~500 instances × 4 budgets × 4 selectors × $k=10$ samples ≈ 80k generations. Nobody has published the full grid with confidence intervals.
- **Empirically open.** Is $S^\star$ model-transferable? Compute $S^\star$ by greedy hill-climbing against model A's pass rate, then evaluate under model B. No published transfer matrix exists.
- **Methodologically blocked.** There is no accepted definition of the oracle context set. Patch-overlap $G$ is circular (the patch is one solution among many); dependency-closure $G$ blows past any budget on real repos. Until $G$ is defined non-circularly, $\mathrm{Rec}@B$ numbers across papers are not comparable.
- **Theoretically open.** No approximation guarantee for budgeted selection under a utility with *positive* complementarity. Supermodular-ish maximization under a knapsack is inapproximable in general; what structural restriction on code dependency graphs (bounded treewidth of the call graph? bounded fan-in of definitional dependencies?) restores a constant factor is unknown.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement compounded by an absent ground truth.** Every reported context-selection improvement bundles four independent factors: (i) which snippets were chosen, (ii) how they were ordered and formatted, (iii) how the query was built, (iv) which model consumed them. Standard papers vary at least two. Because $U$ is order-sensitive (§4), a selector can be credited for a formatting effect.

Underneath, the ground truth is missing: $S^\star$ is defined by the model's own behavior, so it is not a property of the repository. Estimating it requires $O(2^N)$ evaluations, and greedy estimation costs $O(N \cdot k)$ executed test runs per instance — at $N \approx 10^4$ snippets and 30 s per test run, one instance's oracle costs ~35 CPU-days. That is why nobody has the oracle, and without the oracle the selection metric everyone reports ($\mathrm{Rec}@B$ against patch overlap) is measuring "did you retrieve the file that got edited", which is not the thing named.

## 7. Current Research (as of 2026)

- **Structure-aware retrieval**: dependency-graph and dataflow retrieval (DraCo line, ACL 2024; GraphCoder-style call-graph traversal). Active in academic groups in China and at JetBrains Research (Long Code Arena, Bogomolov et al., 2024, gives repo-scale evaluation suites).
- **Learned rerankers over static-analysis proposals**, descending from the ICML 2023 repo-level prompt-generation work.
- **Context compression**: summarize-then-select, skeletonization, and KV-cache reuse to make large $B$ cheap. *(frontier — verify)* — claims that compressed context matches raw context at 4× reduction are mostly single-benchmark.
- **Agentic localization vs. static pipelines**: the Agentless-style rebuttal is being re-run on newer models; whether the gap closed is *(frontier — verify)*.
- **Industrial**: Cursor, GitHub Copilot Workspace, and Sourcegraph all ship proprietary selectors with internal A/B numbers that are not public. This is the largest body of evidence and it is unavailable.

## 8. Concrete Next Experiment

**The budget-matched selection grid.**

- **Scale.** SWE-bench Verified (500 instances) + CrossCodeEval Python (~2.6k instances). Two models (one ~30B open-weight, one frontier API). Budgets $B \in \{4\text{k}, 16\text{k}, 64\text{k}\}$. $k = 10$ samples, $T = 0.8$. ≈ 300k generations total; a few thousand GPU-hours plus API spend on the order of $10$–$20$k.
- **Arms.** (1) BM25 chunk retrieval. (2) Tree-sitter + PageRank repo map. (3) Dependency-closure of the cursor file / issue-mentioned symbols. (4) LLM reranker over arms 1–3. (5) **Control arm: contiguous whole-repo prefix truncated to $B$** — no selection at all, just as many tokens as fit. (6) **Ceiling: gold-patch files at $B$.**
- **Fixed across arms.** Identical rendering template, identical ordering rule (retrieved snippets ascending by relevance, cursor file last), identical query. Only membership varies. This is the ablation the field is missing.
- **The deciding number.** $\Delta = \hat U_{\text{best selector}}(B{=}16\text{k}) - \hat U_{\text{control}}(B{=}16\text{k})$, with 95% bootstrap CI over instances. If the CI excludes 0 and $\Delta \ge 5$ points, selection has demonstrated value at that budget. If the CI contains 0, the honest conclusion is that at 16k, dumping beats retrieving — and the research target shifts entirely to $B \le 4\text{k}$ (latency-bound completion) and $B \ge 128\text{k}$ (cost-bound agents).

## 9. Key References

- **[Foundational]** Nemhauser, Wolsey, Fisher. *An analysis of approximations for maximizing submodular set functions—I.* Mathematical Programming, 1978.
- **[Foundational]** Feige. *A threshold of $\ln n$ for approximating set cover.* JACM, 1998.
- **[SOTA]** Zhang, Chen, Zhang, Xu, Lou, Shi, Duan, Chen. *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation.* EMNLP 2023.
- **[SOTA]** Ding et al. *CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion.* NeurIPS 2023 Datasets & Benchmarks.
- **[SOTA]** Cheng, Wu, Hu. *Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion.* ACL 2024.
- **[SOTA]** Shrivastava, Larochelle, Tarlow. *Repository-Level Prompt Generation for Large Language Models of Code.* ICML 2023.
- **[Benchmark]** Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
- **[Benchmark]** Liu, Xu, Zhang, McAuley et al. *RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems.* ICLR 2024.
- **[Benchmark]** Bogomolov et al. *Long Code Arena: a Set of Benchmarks for Long-Context Code Models.* 2024.
- **[SOTA]** Xia, Deng, Dunn, Zhang. *Agentless: Demystifying LLM-based Software Engineering Agents.* 2024.
- **[Foundational]** Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024.
- **[SOTA]** Bairi et al. *CodePlan: Repository-Level Coding using LLMs and Planning.* FSE 2024.

## 10. Worked Example

A Django-sized repository: 2,800 Python files, 1.9M tokens. Task: an issue whose gold patch touches 2 files, 41 changed lines. Budget $B = 16{,}000$ tokens free.

Snippet inventory at function granularity: $N = 41{,}000$, mean $t_i = 46$ tokens. So $B$ buys about **350 functions out of 41,000 — 0.85% of the repo.**

- **BM25 over the issue text** puts the two gold files' functions at ranks 4 and 1,190. Rank 1,190 is at cumulative cost ~54k tokens — **outside the budget**. $\mathrm{Rec}@16\text{k} = 0.5$.
- **Dependency closure** of the symbols named in the issue: one-hop = 63 functions (2.9k tokens), and it captures gold file 1. Two-hop = 5,900 functions (271k tokens) — **17× over budget**, and it does capture gold file 2. There is no budget-feasible hop count that gets both. This is the obstruction in one line: the relevant unit sits at hop 2, and hop 2 costs 17 budgets.
- **Repo map (tree-sitter + PageRank)** at 4k tokens lists both files' class signatures but no bodies. Under it the model localizes correctly and then writes a patch calling a helper whose signature it never saw — a plausible-looking wrong edit.

Now the measurement problem. Suppose selector A scores $\hat U = 0.34$ and selector B scores $0.29$ at $k=10$, 500 instances. Per-instance SE at $\hat U \approx 0.3$ with $k=10$ is $\sqrt{0.3 \cdot 0.7/10} = 0.145$; across 500 instances the SE on the mean is ~0.0065, so a 5-point gap is ~7.7 SE — resolvable. But swap the *ordering* rule alone (cursor file first vs. last) and published position effects predict a swing of comparable magnitude. **The 5-point gap is real and the attribution to selection is not** — unless the ordering is held fixed, which is exactly what §8's arm design forces.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*