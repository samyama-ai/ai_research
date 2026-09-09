---
id: 19-evaluation/long-context-eval-beyond-needle
title: "Evaluating Long-Context Models Beyond Needle Retrieval"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Evaluating Long-Context Models Beyond Needle Retrieval

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/long-context-eval-beyond-needle` · **Status:** open

## 1. Problem Statement

Vendors report context windows of $128\mathrm{K}$–$10\mathrm{M}$ tokens and near-perfect "needle-in-a-haystack" (NIAH) scores. NIAH inserts a verbatim sentence into filler text and asks the model to repeat it. It measures verbatim lexical lookup and nothing else. The problem is to produce a measurement that reports, for a model $M$ and length $n$, how much of the context $M$ can actually *use* — aggregate, compose, track state over, and reason across — with the length effect separated from task difficulty.

Three variants, different difficulty:

- **Measurement.** Define an *effective context length* $n^*$ that is (a) monotone-comparable across models, (b) not saturated by frontier systems, (c) predictive of downstream long-input task performance. Open.
- **Method.** Build architectures/training that close the gap between claimed and effective length. Partially addressed; progress is not measurable without the measurement variant.
- **Theory.** Prove separations: tasks solvable by a transformer with $n$-token context but not by any retrieval-plus-short-context pipeline of comparable compute. Largely open; the closest results are circuit-complexity bounds on single-pass transformers, not statements about benchmark tasks.

## 2. Formal Setting

A long-context instance is a triple $(q, c, y)$: query $q$, context $c = (c_1,\dots,c_n)$ of $n$ tokens, gold output $y$. A model $M$ scores $s(M; q,c,y) \in [0,1]$ under a metric (exact match, F1, pairwise accuracy).

**Dispersion.** Let $S \subseteq \{1,\dots,n\}$ be the *support*: the minimal token set from which $y$ is derivable. Two quantities, both measurable by construction of the dataset:

$$\rho = \frac{|S|}{n} \quad\text{(support density)}, \qquad \delta = \frac{\max S - \min S}{n} \quad\text{(support spread)}.$$

NIAH is the corner $\rho \approx 10^{-4}$, $\delta \approx 0$, with $S$ lexically overlapping $q$. Real tasks (summarization, code refactoring, novel-scale QA) sit near $\rho \gtrsim 0.1$, $\delta \approx 1$.

**Effective context length.** Fix a task family $\mathcal{T}$ whose instances can be generated at any $n$ with the answer invariant to $n$ (a *metamorphic* family: padding changes nothing that should change the answer). Let $\bar s_M(n)$ be mean score at length $n$ and $\bar s_M(n_0)$ the short-context baseline at $n_0 = 1\mathrm{K}$. Then

$$n^*_\tau(M) = \max\{\, n : \bar s_M(m) \ \ge\ \tau \cdot \bar s_M(n_0)\ \ \forall m \le n \,\},$$

with $\tau$ typically $0.85$ or $0.5$. The $\forall m \le n$ clause matters: scores are non-monotone in $n$, so a threshold crossing alone is not a length.

**Assumptions, and how they break.**
1. *Answer invariance under padding.* Violated whenever filler is drawn from a distribution that shares entities with the needle — the padding becomes distractors and difficulty rises with $n$ for reasons unrelated to memory.
2. *$n_0$ baseline is the right control.* Violated when the short version is a different task (a 1K summary is not a truncated 100K summary).
3. *Support $S$ is identifiable.* Violated for aggregation tasks ("how many times does X occur"), where $S$ is the whole context and $\rho = 1$ by definition, collapsing the coordinate.
4. *No contamination.* Violated for published novels, arXiv papers, and GitHub repos — the standard long-document sources.
5. *Metric is length-blind.* Violated for generative metrics: LLM judges and ROUGE both degrade as reference length grows.

## 3. State of the Art

**Established (with ablations).**
- **RULER** (Hsieh et al., COLM 2024) generalizes NIAH along controlled axes — multi-key, multi-value, multi-query, variable tracking, frequent-word extraction — holding task identity fixed while $n$ varies. This is the cleanest realization of $n^*_\tau$ in the literature.
- **NoLiMa** (Modarressi et al., ICML 2025) removes lexical overlap between query and needle, forcing associative rather than string-match retrieval. It ablates the overlap directly, so the drop is attributable.
- **HELMET** (Yen et al., ICLR 2025) evaluates seven application-shaped categories at matched lengths and reports that synthetic recall tasks rank models differently from downstream tasks — the core evidence that NIAH is not a proxy.

**Claimed but under-ablated.**
- Vendor NIAH heatmaps (Gemini 1.5 technical report, 2024; similar figures from other labs) report $>99\%$ recall at $1\mathrm{M}$+. These are benchmark numbers on a single task shape with no dispersion control and no distractor ablation. They do not license any claim about usable context.
- LongBench, L-Eval, LooGLE, $\infty$Bench aggregate heterogeneous real tasks. Useful for ranking, but length and difficulty are confounded across subsets, so no $n^*$ can be read off them.
- **Michelangelo** (Vodrahalli et al., 2024, Google DeepMind) proposes "latent structure queries" (Latent List, MRCR, IDK) designed to be contamination-proof by construction. The synthesis-based contamination argument is asserted more than tested.

**No SOTA exists** for the theory variant: there is no proven task family separating true long-context computation from retrieval-augmented short context at equal compute.

## 4. What Is Known

Numbers, with the scale they were measured at:

- **Claimed $\gg$ effective.** RULER, 10 models at claimed $32\mathrm{K}$+: only about half hold performance to $32\mathrm{K}$; GPT-4's effective length lands near $64\mathrm{K}$ against a claimed $128\mathrm{K}$. Several open models claiming $32\mathrm{K}$ fall below their $4\mathrm{K}$ baseline by $16\mathrm{K}$.
- **Lexical overlap carries most of NIAH.** NoLiMa, 12 models at $32\mathrm{K}$: 10 of 12 fall below $50\%$ of their $1\mathrm{K}$ baseline; GPT-4o goes from $99.3\%$ at $1\mathrm{K}$ to $69.7\%$ at $32\mathrm{K}$ — the same models that score $>95\%$ on standard NIAH at that length.
- **Position matters.** *Lost in the Middle* (Liu et al., TACL 2024), multi-document QA at 10–30 documents: accuracy is U-shaped in gold-document position, with mid-context placement costing roughly $20$ points and sometimes falling below the closed-book baseline.
- **Length hurts even trivial reasoning.** FLenQA (Levy et al., ACL 2024): a two-fact deduction whose difficulty is constant by construction; GPT-4 drops from about $0.92$ at $250$ tokens to about $0.68$ at $3\mathrm{K}$. The failure starts three orders of magnitude below advertised windows.
- **Genuine global comprehension is far from solved.** NoCha (Karpinska et al., EMNLP 2024), 1,001 true/false claim pairs over recently published novels ($\sim$127K tokens median): best model $55.8\%$ pairwise against a $50\%$ random floor and $\sim$97% human agreement.
- **Aggregation is worse than retrieval.** SummHay (Laban et al., EMNLP 2024): systems score around $20\%$ joint coverage/citation against a $56\%$ human reference on synthesizing repeated insights from a haystack.
- **Reasoning-in-a-haystack collapses early.** BABILong (Kuratov et al., NeurIPS 2024 D&B): most models use only $10$–$20\%$ of available context effectively; performance on bAbI-style multi-fact tasks degrades sharply past $10\mathrm{K}$.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed definition of "effective context length." $n^*_\tau$ depends on $\mathcal{T}$, on $\tau$, on filler distribution, and on the metric; changing filler from repeated Paul Graham essays to in-domain text moves the number by tens of thousands of tokens. Until $\mathcal{T}$ is standardized, cross-paper $n^*$ values are not comparable. This is the primary blocker.
- **Methodologically blocked.** Contamination control for naturally long documents. NoCha's answer — use novels published after the cutoff — has a shelf life of months and does not scale.
- **Empirically open.** Whether long-context reasoning failures are a *memory* limit or an *attention-allocation* limit. The distinguishing experiment (oracle-compressed context vs. full context at matched token count) is runnable today and has not been run systematically across model families.
- **Empirically open.** Whether any published task requires $n$-token context rather than retrieval over chunks at equal inference FLOPs. Goldman et al. (2024) argue much of the benchmark suite does not.
- **Theoretically open.** No lower bound showing a natural task family is solvable in one $n$-token pass but not by $k$ rounds of retrieval plus $m$-token context, for $km \ll n$.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability of the support**.

To isolate a length effect you must vary $n$ while holding difficulty fixed. For retrieval you can: pad with unrelated text. For *reasoning over dispersed evidence* you cannot, because the evidence is what you would have to pad with. Adding tokens either (a) adds distractors, raising difficulty for reasons independent of memory, or (b) adds irrelevant filler, which reduces the task back to retrieval. There is no third option. Every non-retrieval long-context benchmark therefore either loses length control (LongBench, $\infty$Bench) or loses realism (RULER's variable tracking).

Second obstruction: **absent ground truth at scale**. Annotating the support $S$ for a $200\mathrm{K}$-token instance requires a human to read $200\mathrm{K}$ tokens. NoCha needed annotators who had read entire novels; that is why it has 1,001 items and not 100,000. Synthetic generation restores scale and destroys ecological validity.

## 7. Current Research (as of 2026)

- **Metamorphic and controlled-dispersion suites.** RULER-style generation extended to reasoning; HELMET's matched-length protocol is becoming the default reporting format for academic long-context papers (Princeton NLP).
- **Contamination-resistant synthesis.** Michelangelo-style latent-structure queries and MRCR variants (Google DeepMind, OpenAI evals). *(frontier — verify)* Whether synthesis genuinely defeats contamination is untested.
- **Long-output evaluation.** The dual problem — long generation, not long input — via long procedural generation and long-form consistency benchmarks. Much less developed than the input side.
- **Community leaderboards** on non-retrieval comprehension (e.g. fiction-comprehension leaderboards showing frontier models degrading between $16\mathrm{K}$ and $128\mathrm{K}$). *(frontier — verify: these are leaderboards, not peer-reviewed artifacts, and item pools are small.)*
- **RAG-vs-long-context compute-matched comparisons.** Active; results so far are mixed and sensitive to retriever quality, which is itself an uncontrolled variable.

## 8. Concrete Next Experiment

**Question.** Is measured long-context failure a memory limit or an attention-allocation limit?

**Design.** Build 500 metamorphic instances from a single task family — multi-hop aggregation over $k=8$ facts, generated synthetically so $S$ is known exactly. Instantiate each at $n \in \{2\mathrm{K}, 8\mathrm{K}, 32\mathrm{K}, 128\mathrm{K}, 512\mathrm{K}\}$ with $\delta = 1$ (facts spread uniformly). Three arms, all with identical gold answers:

1. **Full** — the $n$-token context.
2. **Oracle-compressed control** — only the $|S|\approx 400$ support tokens, order preserved. Isolates reasoning difficulty from length.
3. **Distractor-matched control** — $n$ tokens where the filler is topically matched and entity-overlapping, but $S$ is unchanged. Isolates distraction from length.

**Scale.** 5 models spanning families (one $\ge 1\mathrm{M}$ claimed window), 500 items × 5 lengths × 3 arms = 7,500 calls per model; roughly $10^9$ input tokens total at the long end. Order of $\$10^4$ in API spend.

**Deciding number.** $\Delta = \big[\bar s(\text{oracle}) - \bar s(\text{full at }512\mathrm{K})\big] - \big[\bar s(\text{oracle}) - \bar s(\text{distractor-matched at }32\mathrm{K})\big]$.

If $\Delta \le 0.05$, the loss at $512\mathrm{K}$ is explained by distraction, not length — long-context work should target attention allocation and evaluation should report distractor density, not window size. If $\Delta \ge 0.20$, there is a length-specific degradation beyond distraction, and $n^*$ is a real quantity worth standardizing.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schütze. *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* ICML, 2025. — arXiv:2502.05167
- **[SOTA]** Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, Danqi Chen. *HELMET: How to Evaluate Long-context Language Models Effectively and Thoroughly.* ICLR, 2025. — arXiv:2410.02694
- **[Benchmark]** Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer. *One Thousand and One Pairs: A "novel" challenge for long-context language models.* EMNLP, 2024. — arXiv:2406.16264
- **[Benchmark]** Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, Mikhail Burtsev. *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.10149
- **[Benchmark]** Philippe Laban, Alexander R. Fabbri, Caiming Xiong, Chien-Sheng Wu. *Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems.* EMNLP, 2024. — arXiv:2407.01370
- **[Analysis]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Analysis]** Omer Goldman, Alon Jacovi, Aviv Slobodkin, Aviya Maimon, Ido Dagan, Reut Tsarfaty. *Is It Really Long Context if All You Need Is Retrieval? Towards Genuinely Difficult Long Context NLP.* EMNLP, 2024. — arXiv:2407.00402
- **[SOTA]** Kiran Vodrahalli et al. *Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries.* Google DeepMind, 2024. — arXiv:2409.12640
- **[Survey]** Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* ACL, 2024. — arXiv:2308.14508

## 10. Worked Example

Take GPT-4o at $n = 32\mathrm{K}$ and three published measurements of the *same model at the same length*:

| Benchmark | Task shape | Score at 32K |
|---|---|---|
| Standard NIAH | verbatim needle, lexical overlap | $>95\%$ |
| NoLiMa | needle requires one associative hop | $69.7\%$ (from $99.3\%$ at 1K) |
| NoCha (≈127K, nearest comparable) | global claim verification over a novel | $55.8\%$ vs $50\%$ floor |

Now compute $n^*_{0.85}$ for this one model under each family, using the definition in §2 with $n_0 = 1\mathrm{K}$:

- Under NIAH: score never drops below $0.85 \times 0.99 = 0.84$ at any tested length, so $n^*_{0.85} \ge 128\mathrm{K}$ — the window is "fully effective."
- Under NoLiMa: $0.697 < 0.84$ already at $32\mathrm{K}$, and the published curve crosses the threshold between $2\mathrm{K}$ and $8\mathrm{K}$, so $n^*_{0.85} \approx 4\mathrm{K}$.

**One model, one length, a 32× disagreement in effective context.** The gap is not noise and not model improvement — it is entirely a choice of $\mathcal{T}$. Neither number is wrong under its own definition, which is exactly the problem: $n^*$ is currently a property of the benchmark, not of the model.

Push further and the obstruction becomes visible. Try to fix this by making $\mathcal{T}$ "realistic" — say, NoCha. Now you cannot vary $n$ at all: a novel is the length it is. You cannot pad it (padding a novel with unrelated text changes the task), and you cannot truncate it (the claims reference the whole book). So the benchmark with ecological validity has no length axis, and the benchmark with a clean length axis measures string matching. Section 8's oracle-compressed control is an attempt to buy back a length axis by subtraction rather than padding; whether it survives contact with non-synthetic tasks is itself unresolved.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*