---
id: 26-code-generation/long-context-utilization-whole-repository
title: "Long-Context Attention Utilization for Whole-Repository Reasoning"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Attention Utilization for Whole-Repository Reasoning

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/long-context-utilization-whole-repository` · **Status:** empirically-open

## 1. Problem Statement

A model with a 1M-token context window can hold a mid-size repository in one prompt. The question is whether it *uses* it.

- **Input:** a repository snapshot $R$ (files, build config, tests) serialized to a token sequence, plus a task $q$ — complete a function, fix a failing test, rename an API across call sites.
- **Output:** a patch or completion $y$.
- **Decision predicate:** does the model's accuracy on tasks whose ground truth depends on a span located at repository position $p$ stay flat in $p$ and in total context length $N$, once the span is in context?

Three variants, routinely conflated:

- **Measurement.** Define a utilization statistic that separates "the evidence was not attended to" from "the evidence was attended to and the reasoning failed." Currently under-specified.
- **Method.** Build a model or a context-assembly policy whose whole-repo accuracy matches its oracle-context accuracy. Currently unachieved.
- **Theory.** Prove whether a depth-$d$, $H$-head transformer can perform $k$-hop cross-file dependency resolution over $N$ tokens without width or depth growing with $N$. Partially answered, negatively, for related tasks.

Solving it means: for a fixed model, $\mathrm{Acc}(\text{full repo in context}) \ge \mathrm{Acc}(\text{oracle-minimal context})$ on a task suite with verified dependency labels, at $N \ge 500\text{K}$ tokens.

## 2. Formal Setting

Let $R$ be a repository, $C = \mathrm{serialize}(R, \pi)$ its token sequence under file order $\pi$, $|C| = N$. A task instance is $(q, C, y^\*)$ with a verifier $V(y) \in \{0,1\}$ (tests pass / exact-match patch).

**Dependency set.** $D^\* \subseteq \{1,\dots,N\}$ is the set of token positions belonging to spans a correct solution must consult — obtained from a static call/type graph or from human annotation, not from the model.

**Necessity (measured by ablation).** For a span $s$,
$$\mathrm{Nec}(s) \;=\; \Pr\big[V(y)=1 \mid C\big] \;-\; \Pr\big[V(y)=1 \mid C \setminus s\big],$$
estimated over $n$ samples at fixed temperature; $C \setminus s$ replaces $s$ with length-matched filler to hold $N$ constant.

**Utilization gap.** With $C_{\mathrm{oracle}}$ the concatenation of $D^\*$ spans only,
$$\Delta(N) \;=\; \mathrm{Acc}(C_{\mathrm{oracle}}) - \mathrm{Acc}(C_{\mathrm{full}}), \qquad |C_{\mathrm{full}}| = N.$$
$\Delta > 0$ is the distractor penalty: paying for context makes the model worse.

**Effective context length.** $L_{\mathrm{eff}}(\tau) = \max\{N : \mathrm{Acc}(C_{\mathrm{full}}) \ge \tau\cdot\mathrm{Acc}(C_{\mathrm{oracle}})\}$, conventionally $\tau=0.85$.

**Positional sensitivity.** Let $\rho \in [0,1]$ be the relative position of the needed span under a random permutation $\pi$. Define $\sigma_{\mathrm{pos}} = \mathrm{sd}_\rho\big[\mathrm{Acc}(\rho)\big]$ over binned $\rho$; a perfectly utilizing model has $\sigma_{\mathrm{pos}} = 0$.

**Attention mass** $A(s) = \sum_{\ell,h} \sum_{i \in s} \alpha^{(\ell,h)}_{\mathrm{out},i}$ is reported in the literature but is *not* a utilization measure — attention weight and causal contribution dissociate (Jain & Wallace 2019).

**Assumptions, and which break.**
1. $D^\*$ is unique and complete — **violated**: repositories admit multiple valid evidence paths; static graphs miss dynamic dispatch, reflection, config-driven wiring.
2. Ablation is additive, so per-span $\mathrm{Nec}$ composes — **violated**: redundant copies of the same logic mask each other, giving $\mathrm{Nec}(s)\approx 0$ for genuinely used spans.
3. Filler is inert — **violated**: filler changes positional-encoding statistics and can act as a distractor itself.
4. No train-set contamination — **violated for public repos**; the model may reconstruct $y^\*$ without reading $C$.
5. Serialization order $\pi$ is nuisance — **violated**: real agents see a natural order correlated with relevance, so uniform-random $\pi$ measures a different quantity than deployment.

## 3. State of the Art

**Established (ablated, reproduced independently).**
- Position-dependent degradation is real and architecture-general: *Lost in the Middle* (Liu et al., TACL 2024) shows U-shaped accuracy in multi-document QA across GPT-3.5, Claude, and open models.
- Synthetic long-context benchmarks with proper controls show effective length far below claimed length: RULER (Hsieh et al., COLM 2024) finds most models claiming $\ge 32$K fall below their own short-context baseline well before 32K.
- Literal string overlap inflates scores: NoLiMa (Modarressi et al., ICML 2025) removes lexical matches between query and needle; at 32K, 10 of 12 tested models fall below 50% of their $<1$K baseline, and GPT-4o drops from 99.3% to 69.7%.
- Retrieval is carried by a sparse subnetwork: *Retrieval Head Mechanistically Explains Long-Context Factuality* (Wu et al., ICLR 2025) identifies a small head set (a few percent of heads) whose masking collapses needle retrieval while perplexity barely moves.

**Claimed but unablated.**
- Frontier "1M-token" reports (e.g. the Gemini 1.5 technical report, 2024) present near-perfect single-needle recall at 1M. Needle recall is not multi-hop code reasoning; no public ablation ties 1M recall to $\Delta(N)$ on repository tasks.
- Agent scaffolds report SWE-bench Verified scores above 70% while placing only a handful of files in context. This is a **benchmark number**: it shows retrieval+localization works, and says nothing about whole-repo context utilization, because the full repo was never in the prompt.

**Empirical SOTA on repo-scale code specifically.** RepoBench (Liu et al., ICLR 2024), CrossCodeEval (Ding et al., NeurIPS 2023), and Long Code Arena (Bogomolov et al., 2024) are the standard suites. All three find retrieval-augmented short context competitive with or better than naive long context. Agentless (Xia et al., 2024) reached SWE-bench-lite performance comparable to complex agents using a fixed localize-repair-validate pipeline — evidence that selection, not context length, is doing the work.

**Theory SOTA.** Sanford, Hsu & Telgarsky (NeurIPS 2023) prove a one-layer attention layer needs width $\tilde\Omega(N)$ for sparse averaging over $N$ items, while two layers need $\tilde O(1)$. Peng, Narayanan & Papadimitriou (COLM 2024) give a communication-complexity argument that a single-layer, $H$-head, width-$d$ transformer cannot compose two functions over a domain of size $n$ when $Hd = o(n)$. Neither is stated for the repository task, but both bound $k$-hop resolution over long inputs.

## 4. What Is Known

- **Degradation begins far below the window.** Levy, Jacoby & Goldberg (ACL 2024) show reasoning accuracy on a controlled task drops sharply from about 3K input tokens, on models with 16K–128K windows.
- **Effective $\ll$ claimed.** RULER, at 4K–128K, on 10+ open and closed models: effective lengths typically 2–4$\times$ below advertised.
- **Distractors cost more than distance.** Across *Lost in the Middle* and NoLiMa, adding irrelevant context of the same length hurts more than moving the needle, i.e. $\Delta(N) > 0$ dominates $\sigma_{\mathrm{pos}}$ at large $N$.
- **Retrieval beats stuffing on code.** RepoCoder (Zhang et al., EMNLP 2023) and CrossCodeEval both show iterative/BM25 retrieval over a short window outperforming longer undifferentiated context at equal or lower cost.
- **Long context and RAG trade off by budget.** Li et al. (EMNLP 2024 industry track) find long-context models outperform RAG when cost is ignored, and a routing hybrid recovers most of the gain at a fraction of the tokens — measured on QA, not repositories.
- **Training can partly close the middle gap.** FILM-7B (An et al., NeurIPS 2024) closes much of the positional dip via information-intensive training at 32K, without loss on short tasks.

## 5. What Is Not Known

- **Methodologically blocked:** there is no agreed utilization statistic for code. $D^\*$ is not well defined when multiple evidence paths exist, and ablation-based necessity is confounded by redundancy in real repositories. Every number in §4 comes from synthetic needles or short QA, not verified repository dependencies.
- **Empirically open:** $\Delta(N)$ and $L_{\mathrm{eff}}(0.85)$ have never been measured for frontier models at $N \ge 500$K on a decontaminated repository suite with human-verified dependency labels. The experiment is runnable today; the cost is the barrier, not the science.
- **Empirically open:** whether the utilization gap is a *retrieval* failure or a *composition* failure. No study conditions on "the needed span was retrievable" and then measures multi-hop success.
- **Theoretically open:** whether $k$-hop cross-file dependency resolution over $N$ tokens is achievable at depth $O(\log k)$ and width independent of $N$. The existing lower bounds cover one-layer or two-function-composition cases only; no matching bound or construction exists for the repository setting.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth.** $D^\*$ requires a sound, complete dependency label. Static analysis for Python/JS/TS is unsound under dynamic dispatch and dependency injection; human annotation of "what a fix must consult" disagrees between annotators. Without $D^\*$, $\Delta(N)$ cannot be computed.
2. **Confounded measurement via contamination.** Every public repository benchmark risks the model reproducing $y^\*$ from pretraining. A model that scores well may have read nothing. Decontamination requires post-cutoff private repositories, which are hard to obtain at scale with passing test suites.
3. **Compute cost.** One 1M-token forward pass per instance, $n=5$ samples, 500 instances, across 6 context lengths and 8 position bins, is on the order of $10^{11}$ input tokens per model. At commercial rates this is a five-to-six-figure USD experiment per model — which is why it has not been run.

Add a fourth, softer one: SWE-bench-style suites are named as repository reasoning benchmarks but reward localization plus a local edit. They do not measure whole-repo attention utilization, so high scores have been read as evidence for a claim they do not test.

## 7. Current Research (as of 2026)

- **Sparse and hierarchical attention for code**: retrieval-head-aware KV eviction and query-conditioned compression, so 1M tokens can be held at a fraction of KV cost. Active at NVIDIA (RULER lineage), Microsoft Research, and DeepSeek *(frontier — verify)*.
- **Repo-scale benchmarks with dependency labels**: JetBrains Research (Long Code Arena) and the SWE-bench group are extending toward multi-file, cross-repo edits with explicit evidence annotation.
- **Context engineering as the deployed answer**: agent frameworks default to retrieve-then-read with sub-agent context isolation rather than whole-repo stuffing — an admission that $\Delta(N) > 0$ in production.
- **Mechanistic work on retrieval heads and induction circuits** extended to code-specific structures (import resolution, type propagation) *(frontier — verify)*.
- **Theory**: depth-vs-context separations for multi-hop retrieval, following Sanford et al. and Peng et al.

## 8. Concrete Next Experiment

**"Oracle-versus-repo" utilization curve.**

- **Scale.** 300 task instances from 40 private or post-cutoff repositories (Python + TypeScript), each with a failing test and a human-verified dependency set $D^\*$ (two annotators, adjudicated). Context lengths $N \in \{8\text{K}, 32\text{K}, 128\text{K}, 512\text{K}, 1\text{M}\}$ built by adding *real files from the same repository* — never synthetic filler. Position of $D^\*$ spans binned into 5 relative-position buckets. $n=5$ samples, temperature 0.2. Two frontier long-context models. Roughly $2\times10^{11}$ input tokens total.
- **Control arm.** $C_{\mathrm{oracle}}$: the $D^\*$ spans alone (typically 3K–15K tokens), same prompt template, same sampling. Second control: $C_{\mathrm{oracle}}$ plus random same-repo files padded to $N$ — separates "long" from "distracting."
- **Deciding number.** $\Delta(1\text{M}) = \mathrm{Acc}(C_{\mathrm{oracle}}) - \mathrm{Acc}(C_{\mathrm{full}})$ in test-pass rate, with a bootstrap 95% CI. $\Delta(1\text{M}) \le 2$ points means whole-repo context is utilized and retrieval scaffolds are unnecessary overhead. $\Delta(1\text{M}) \ge 10$ points means long context is actively harmful at repo scale and the field's context-engineering practice is correct for a measurable reason. Report $L_{\mathrm{eff}}(0.85)$ as the secondary headline.

Contamination check: for each instance, a closed-book arm with $q$ and file paths only. Instances the model solves closed-book are dropped before computing $\Delta$.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schütze. *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* ICML, 2025. — arXiv:2502.05167
- **[Foundational]** Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu. *Retrieval Head Mechanistically Explains Long-Context Factuality.* ICLR, 2025. — arXiv:2404.15574
- **[SOTA]** Tianyang Liu, Canwen Xu, Julian McAuley. *RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems.* ICLR, 2024. — arXiv:2306.03091
- **[SOTA]** Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang. *CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2310.11248
- **[SOTA]** Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, Weizhu Chen. *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation.* EMNLP, 2023. — arXiv:2303.12570
- **[SOTA]** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[SOTA]** Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, Lingming Zhang. *Agentless: Demystifying LLM-based Software Engineering Agents.* 2024. — arXiv:2407.01489
- **[Theory]** Clayton Sanford, Daniel Hsu, Matus Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS, 2023. — arXiv:2306.02896
- **[Theory]** Binghui Peng, Srini Narayanan, Christos Papadimitriou. *On Limitations of the Transformer Architecture.* COLM, 2024. — arXiv:2402.08164
- **[Empirical]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: The Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Method]** Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, Weizhu Chen. *Make Your LLM Fully Utilize the Context.* NeurIPS, 2024. — arXiv:2404.16811
- **[Survey/Benchmark]** Egor Bogomolov, Aleksandra Eliseeva, Timur Galimzyanov, Evgeniy Glukhov, Anton Shapkin, Maria Tigina, Yaroslav Golubev, Alexander Kovrigin, Arie van Deursen, Maliheh Izadi, Timofey Bryksin. *Long Code Arena: a Set of Benchmarks for Long-Context Code Models.* 2024. — arXiv:2406.11612
- **[Comparison]** Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky. *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach.* EMNLP (industry track), 2024. — arXiv:2407.16833
- **[Caution]** Sarthak Jain, Byron C. Wallace. *Attention is not Explanation.* NAACL, 2019. — arXiv:1902.10186

## 10. Worked Example

A TypeScript monorepo, 420K tokens serialized. Task: a test in `packages/api/__tests__/billing.test.ts` fails after a currency field was added. The correct fix edits `packages/core/src/money.ts`, and correctness depends on three spans:

1. the `Money` interface in `packages/core/src/types.ts` (position 0.11 of the file order),
2. a rounding convention in `packages/core/src/round.ts` (position 0.47),
3. a re-export in `packages/api/src/index.ts` (position 0.93).

$|D^\*| \approx 900$ tokens out of $N = 420{,}000$ — a density of $0.21\%$.

Run the two arms, $n=5$, one model:

| Arm | Context | Test-pass rate |
|---|---|---|
| Oracle | 900 tokens ($D^\*$ only) | 5/5 |
| Oracle + 419K same-repo padding | 420K | 2/5 |
| Full repo, natural order | 420K | 1/5 |

$\Delta(420\text{K}) = 5/5 - 1/5 = 0.80$, or 80 points. Failures are informative: 3 of 4 failed runs edit a *different* `Money` type in `packages/legacy/src/money.ts` — a redundant, stale definition at position 0.62.

Now the obstruction. Compute per-span necessity by ablation. Delete span 1 from the full-repo context: pass rate stays 1/5. Necessity is measured as $0$. The model was not using it — but it was also not using the correct definition at all; it was using the legacy duplicate. Delete the legacy file instead and the full-repo rate goes to 4/5, so the *highest-necessity* edit is deleting a span that is not in $D^\*$.

That is the blockage in one instance: with duplicated definitions, $\mathrm{Nec}(s)$ does not identify whether the model attended to the right evidence, $D^\*$ is not the set that changes the outcome, and no attention statistic distinguishes "read `types.ts` and reasoned wrongly" from "read `legacy/money.ts` and reasoned correctly about the wrong thing." The 80-point $\Delta$ is a solid measurement. Its *cause* is not identifiable with the current toolkit — which is why the problem is methodologically blocked before it is empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*