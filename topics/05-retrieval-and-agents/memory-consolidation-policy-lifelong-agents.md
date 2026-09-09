---
id: 05-retrieval-and-agents/memory-consolidation-policy-lifelong-agents
title: "Memory Consolidation Policy for Lifelong Agents"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memory Consolidation Policy for Lifelong Agents

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/memory-consolidation-policy-lifelong-agents` · **Status:** open

## 1. Problem Statement

An agent runs for months. It sees far more experience than it can store, and far more stored material than it can put in a context window. A **consolidation policy** decides, online, what to write, what to rewrite into a more compact form, what to link, and what to delete — under a hard storage budget and a hard retrieval budget.

- **Input:** a stream of episodes $e_1, e_2, \dots$ (dialogue turns, tool traces, observations), arriving without knowing which future queries will need them.
- **Output:** a memory state $M_t$ of bounded size, plus a retrieval function that returns $k$ items per query.
- **Objective:** maximise long-horizon task utility, not retrieval recall on the episode that happened to be written.

Three variants, routinely conflated:

- **Measurement.** Is there a benchmark where a consolidation policy's *deletion and rewriting* decisions change the score, holding the retriever and the base model fixed? Mostly no — current suites reward retrieval, and a policy that never deletes wins by default.
- **Method.** Build a policy that beats "append everything, retrieve top-$k$" at a fixed byte budget over $10^5$+ episodes.
- **Theory.** Is bounded-memory lifelong consolidation competitive with the offline optimum? For pure eviction under known utility this is paging (Sleator & Tarjan 1985). With *rewriting* — merging episodes into summaries that are lossy and non-invertible — no competitive-ratio result is known.

Solving it means: a policy with a stated competitive or regret bound, plus an empirical win at a byte budget $\le 1\%$ of the raw stream on a task where forgetting is penalised.

## 2. Formal Setting

**Stream and memory.** Episodes $e_t \in \mathcal{E}$ arrive for $t = 1..T$. Memory is a set of items $M_t = \{m_1,\dots,m_{n_t}\}$, each item a token sequence. The budget is measured in **stored tokens** (tokenizer-specific, so report the tokenizer):
$$\mathrm{size}(M_t) = \sum_{i} |\mathrm{tok}(m_i)| \le B .$$
Report $B$ as a *compression ratio* $\rho = B / \sum_{t\le T}|\mathrm{tok}(e_t)|$; papers that report only "number of memories" are not comparable, since a summary item and a raw turn differ by 10–100× in tokens.

**Policy.** $\pi: (M_{t-1}, e_t) \mapsto M_t$, decomposed into write / merge / link / evict. Its cost is measured as LLM calls and tokens per episode, $c_\pi$, amortised: a policy that re-summarises all of $M$ every step is $O(B)$ per episode and does not scale.

**Retrieval and utility.** A query $q$ arrives with the true answer $a^\star$. Retrieval returns $R(q, M_t)$ with $|\mathrm{tok}(R)| \le K$ (the context budget). Utility is end-task:
$$U(\pi) = \mathbb{E}_{(q,a^\star)\sim \mathcal{Q}_T}\big[\,s\big(f_\theta(q, R(q,M_T)),\, a^\star\big)\,\big],$$
with $s$ an exact-match or judge score and $f_\theta$ a **frozen** base model. Freezing $\theta$ is what separates this problem from continual fine-tuning.

**Regret.** Against the clairvoyant offline policy that knows $\mathcal{Q}_T$ in advance:
$$\mathcal{R}(\pi) = U(\pi^{\mathrm{OPT}}_{B}) - U(\pi_B).$$
$\pi^{\mathrm{OPT}}_B$ is Belady-like and not computable at scale; in practice it is approximated by an oracle that keeps exactly the gold evidence spans, which is why evidence-annotated benchmarks matter.

**Assumptions, and where they break.**
- *Utility is set-modular or submodular in $M$* — assumed by every greedy selector, and false: multi-hop answers need two items jointly, so marginal value of either alone is zero (supermodular).
- *Queries are drawn i.i.d. from a stationary $\mathcal{Q}$* — false; user interests drift, and facts get **updated** (old value becomes wrong, not merely stale).
- *Items are independent* — false; merging duplicates changes the calibration of frequency-based salience.
- *Retrieval is fixed while the policy varies* — required for attribution, violated whenever a paper changes memory format and retriever together.

## 3. State of the Art

**Systems / empirical.**
- **MemGPT** (Packer et al., 2023) — OS-style paging between a context window and external store, with the model issuing its own page-in/page-out calls. Established: the interface works and extends effective context. Not ablated: whether *learned* eviction beats recency eviction at fixed budget.
- **Generative Agents** (Park et al., UIST 2023) — memory stream scored by recency + importance + relevance, plus periodic "reflection" into higher-level statements. The ablation removing reflection/importance degrades believability ratings; this is a human-rating result on 25 agents, not a task-accuracy result.
- **A-MEM** (Xu et al., 2025) — Zettelkasten-style linking and note evolution; reports gains on LoCoMo multi-hop. Claimed-but-unablated: whether the gain comes from linking or from the extra LLM passes over the text.
- **Mem0** (Chhikara et al., 2025) — extract/update/delete pipeline; reports ~26% relative improvement over a commercial assistant memory on LOCOMO with ~90% token reduction and large p95-latency reduction. These are benchmark numbers on one suite; the deletion path is not separately ablated.
- **HippoRAG** (Gutiérrez et al., NeurIPS 2024) — PageRank over an open-KG index; up to ~20% improvement over strong RAG on multi-hop QA, at a fraction of iterative-retrieval cost. It is an *indexing* result, not a *forgetting* result — nothing is discarded.

**Theory.** Paging: LRU is $k$-competitive and no deterministic online algorithm does better; randomised marking achieves $2H_k$ (Sleator & Tarjan 1985; Fiat et al. 1991). Streaming submodular maximisation gives $1/2-\varepsilon$ in one pass (Badanidiyuru et al., KDD 2014), $1-1/e$ offline (Nemhauser–Wolsey–Fisher 1978). Knoblauch, Husain & Diethe (ICML 2020) show optimal continual learning requires perfect memory and is NP-hard — the closest thing to a hardness result for the parametric analogue.

No theory covers *lossy rewriting* under an unknown future query distribution.

## 4. What Is Known

- **Long-horizon memory is unsolved at benchmark scale.** LongMemEval (Wu et al., ICLR 2025) reports ~30% accuracy drop for commercial chat assistants on sustained long-term memory tasks versus short-context conditions; instances reach ~115k tokens of history.
- **LoCoMo** (Maharana et al., ACL 2024) conversations average ~300 turns / ~9k tokens across up to 35 sessions; LLMs lag humans substantially, with temporal-reasoning and multi-hop the weakest categories.
- **Small replay buffers are surprisingly strong** in the parametric setting: tiny episodic memories of ~1 example per class recover much of the multi-task gap (Chaudhry et al., 2019), and GEM-style constraints beat naive fine-tuning (Lopez-Paz & Ranzato, NeurIPS 2017). Scale: CIFAR/MiniImageNet, not agents.
- **Reflection/self-critique helps in-episode**, not across-episode: Reflexion (Shinn et al., NeurIPS 2023) reaches 91% pass@1 on HumanEval with verbal feedback within a task — an episodic-buffer result that does not demonstrate consolidation across $10^4$ episodes.
- **Skill libraries compound.** Voyager (Wang et al., TMLR 2024) reports ~3.3× more unique items and 2.3× longer tech-tree traversal than baselines from a growing, deduplicated skill store. Again: monotone growth, no eviction pressure.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no benchmark whose score *drops* when a policy keeps too much. Every public suite (LoCoMo, LongMemEval) is scored per-query with an unbounded store, so "never delete" is optimal by construction. Until budget $B$ is part of the protocol, consolidation cannot be measured.
- **Empirically open.** Does any consolidation policy beat recency+frequency eviction at $\rho \le 0.01$ over $\ge 10^5$ episodes with a frozen retriever? Runnable today; the run costs LLM calls, not new science.
- **Theoretically open.** Competitive ratio for online *lossy rewriting* with supermodular multi-hop utility. Also open: whether any online policy achieves $o(T)$ regret when facts are updated (a stored item becomes actively wrong) rather than merely superseded.
- **Open.** Whether episodic/semantic separation (Tulving 1972; McClelland et al. 1995) buys anything measurable in agents beyond acting as a prompt-engineering scaffold.

## 6. Why It Is Hard

Three named obstructions.

1. **The evaluation does not measure what it names.** "Memory benchmarks" measure retrieval recall on an append-only log. The policy's deletion decisions are unobserved because nothing forces deletion. This is a protocol defect, not a modelling one.
2. **Non-identifiability of the credit.** A consolidation system changes the item format, the index, the retriever's embedding distribution, and the number of LLM passes over the text — simultaneously. A gain of 26% on LOCOMO is not attributable to consolidation unless the retriever and pass count are held fixed. Almost no published ablation does this.
3. **Absent ground truth for future value.** $\pi^{\mathrm{OPT}}$ requires knowing $\mathcal{Q}_T$. Constructing an oracle needs evidence-span annotation over $10^5$ episodes, which no public dataset has; synthetic streams provide it but drift-free synthetic streams make forgetting trivially easy.

Compute is a secondary cost: a $10^5$-episode run with per-episode consolidation is $\sim 10^5$–$10^6$ LLM calls per arm, roughly $10^2$–$10^3$ USD — affordable, which is why this is empirically open rather than blocked by budget.

## 7. Current Research (as of 2026)

- **Agentic memory as a product surface.** Mem0, Zep/Graphiti, LangMem, and the memory features in commercial assistants ship extract–update–delete pipelines. Evaluation remains LOCOMO/LongMemEval-shaped. *(frontier — verify current numbers.)*
- **Graph-structured consolidation.** HippoRAG lineage (Ohio State) and temporal knowledge-graph memories; the open question is edge-set growth under eviction.
- **Sleep-time / offline consolidation.** Batch reprocessing of the day's episodes into semantic notes, sometimes called "sleep-time compute." Plausible and under-ablated. *(frontier — verify.)*
- **Budgeted memory benchmarks.** Several 2025 suites add incremental multi-turn protocols; none that I can verify enforce a token budget as a scored constraint. *(frontier — verify.)*
- **CoALA** (Sumers, Yao, Narasimhan & Griffiths, TMLR 2024) remains the standard framing that separates episodic / semantic / procedural memory for language agents.

## 8. Concrete Next Experiment

**Budgeted Consolidation Benchmark, single-variable.**

- **Scale.** $T = 10^5$ episodes synthesised as a 12-month simulated assistant log: 40% chit-chat distractors, 30% durable facts, 20% **fact updates** (address changes, preference reversals), 10% multi-hop-linkable pairs. Every fact carries a gold evidence-span id, so $\pi^{\mathrm{OPT}}_B$ is computable. Query set $\mathcal{Q}_T$: 5,000 queries, half issued $>60$ simulated days after the evidence.
- **Held fixed across all arms.** Base model (frozen), retriever (one embedding model, top-$k$ with $K=4{,}000$ context tokens), and consolidation compute $c_\pi \le 1$ LLM call per episode. Budgets swept at $\rho \in \{0.3, 0.1, 0.03, 0.01\}$.
- **Control arm.** Append-everything + LRU eviction to budget, top-$k$ retrieval. This is the arm every paper skips.
- **Treatment arms.** (a) importance-scored eviction; (b) merge-on-write summarisation; (c) graph-linked notes; (d) $\pi^{\mathrm{OPT}}_B$ oracle upper bound.
- **Deciding number.** Accuracy gap over LRU at $\rho = 0.01$, reported with the oracle gap as denominator: $\Delta = \frac{U(\pi) - U(\mathrm{LRU})}{U(\pi^{\mathrm{OPT}}) - U(\mathrm{LRU})}$. If no published policy reaches $\Delta \ge 0.25$ with 95% CI excluding 0, the field has no demonstrated consolidation policy — only retrieval engineering.
- **Secondary number to report:** accuracy on the *fact-update* slice, where keeping the old item is actively harmful. Predicted result: append-style policies lose here, and it is the only slice where deletion pays.

## 9. Key References

- **[Foundational]** McClelland, McNaughton & O'Reilly. *Why there are complementary learning systems in the hippocampus and neocortex.* Psychological Review, 1995.
- **[Foundational]** Sleator & Tarjan. *Amortized efficiency of list update and paging rules.* CACM, 1985.
- **[Foundational]** Nemhauser, Wolsey & Fisher. *An analysis of approximations for maximizing submodular set functions—I.* Mathematical Programming, 1978.
- **[Theory]** Knoblauch, Husain & Diethe. *Optimal Continual Learning has Perfect Memory and is NP-hard.* ICML 2020.
- **[Theory]** Badanidiyuru, Mirzasoleiman, Karbasi & Krause. *Streaming Submodular Maximization: Massive Data Summarization on the Fly.* KDD 2014.
- **[Foundational]** Lopez-Paz & Ranzato. *Gradient Episodic Memory for Continual Learning.* NeurIPS 2017. — arXiv:1706.08840
- **[SOTA]** Packer et al. *MemGPT: Towards LLMs as Operating Systems.* 2023. — arXiv:2310.08560
- **[SOTA]** Park, O'Brien, Cai, Morris, Liang & Bernstein. *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. — arXiv:2304.03442
- **[SOTA]** Gutiérrez, Shu, Gu, Yasunaga & Su. *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* NeurIPS 2024. — arXiv:2405.14831
- **[SOTA]** Chhikara et al. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.* 2025. — arXiv:2504.19413
- **[Benchmark]** Maharana, Lee, Tulyakov, Bansal, Barbieri & Fang. *Evaluating Very Long-Term Conversational Memory of LLM Agents.* ACL 2024. — arXiv:2402.17753
- **[Benchmark]** Wu et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* ICLR 2025. — arXiv:2410.10813
- **[Survey]** Sumers, Yao, Narasimhan & Griffiths. *Cognitive Architectures for Language Agents.* TMLR 2024. — arXiv:2309.02427
- **[Survey]** Zhang et al. *A Survey on the Memory Mechanism of Large Language Model based Agents.* 2024. — arXiv:2404.13501

## 10. Worked Example

A 12-month assistant. 300 sessions, 40 turns each, 60 tokens/turn: $300\times40\times60 = 720{,}000$ raw tokens. Budget $\rho = 0.01 \Rightarrow B = 7{,}200$ stored tokens, about 90 summary items at 80 tokens each. Retrieval budget $K = 4{,}000$ tokens, so roughly half of memory can enter context anyway — the *store* budget binds, the retrieval budget does not.

Now the fact-update case. Month 2: "I'm vegetarian." Month 7: "I eat fish now." Month 11: query "book me dinner."

- **Append + LRU.** Both items present; LRU keeps both since both were touched. Retrieval returns both. The frozen model must resolve the contradiction from timestamps. Measured behaviour in this shape of test: models pick the wrong one a large fraction of the time — LongMemEval's ~30% drop is concentrated in exactly these knowledge-update and temporal categories.
- **Merge-on-write.** The consolidator sees the month-7 turn, retrieves the vegetarian note, rewrites to "eats fish, no meat." Correct — *if* the retrieval at write time surfaced the old note. At $B = 7{,}200$ and a 5-month gap, the old note is one of ~90 items; a top-5 write-time retrieval over an embedding index misses it whenever the surface forms differ ("vegetarian" vs "I eat fish now" are not near-neighbours). One miss and the memory silently holds a false fact with no error signal.

**Where the obstruction becomes visible.** Score this on LoCoMo or LongMemEval as published and both arms look similar, because scoring is per-query over an unbounded store and the contradiction is a single item in thousands. The failure only becomes a *number* when (i) $B$ is enforced, and (ii) the update slice is scored separately. Absent both, "our memory system improved 26%" is a statement about retrieval quality, and the consolidation policy — the write, merge and delete decisions — has never been measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*