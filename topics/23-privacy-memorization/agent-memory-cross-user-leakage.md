---
id: 23-privacy-memorization/agent-memory-cross-user-leakage
title: "Agent Memory Stores and Cross-User Leakage"
topic: 23-privacy-memorization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Agent Memory Stores and Cross-User Leakage

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/agent-memory-cross-user-leakage` · **Status:** empirically-open

## 1. Problem Statement

Deployed LLM agents keep persistent memory: summaries, extracted facts, embedded transcript chunks, tool-call traces, and cached artifacts, written during one session and retrieved in later ones. When one deployment serves many principals — users of a consumer assistant, employees in a shared workspace, tenants of a hosted agent platform — the question is whether content written by principal $u$ can surface in a response served to principal $u' \neq u$.

Three variants, with different difficulty:

- **Measurement.** Given a deployed agent with a memory subsystem, estimate the probability that a given piece of $u$'s private content appears in a response to $u'$, under a stated adversary model. No agreed estimator exists. This is the binding variant.
- **Method.** Build a memory architecture whose cross-user leakage is provably bounded (by namespace isolation, per-principal encryption, DP-noised shared statistics, or a policy-checked read gate) without destroying the utility that shared memory provides — collaborative recall, org-wide knowledge, deduplicated tool results.
- **Theory.** Characterize the achievable utility/leakage frontier for a retrieval-augmented memory serving $n$ principals. Unlike training-time memorization, memory leakage is not a property of parameters; it is a property of a retrieval and prompt-assembly policy, so classical DP-SGD accounting does not transfer.

Solving it means: an estimator with stated false-negative behaviour, plus an architecture that keeps the estimate below a declared bound at fixed task utility.

## 2. Formal Setting

Principals $\mathcal{U}$, $|\mathcal{U}| = n$. Session $t$ for principal $u$ produces transcript $x_{u,t}$. A write policy $W$ maps transcripts to memory items:
$$M \leftarrow M \cup W(x_{u,t}), \qquad m = (\text{text}_m,\ e_m \in \mathbb{R}^d,\ \text{owner}(m) = u,\ \text{scope}(m))$$

A read policy $R$ takes query $q$ from principal $u'$ and returns $R(q, u', M) \subseteq M$, typically top-$k$ by cosine similarity plus a filter. The agent's response is $y = f(q, R(q,u',M))$.

**Leakage event.** Fix a secret $s$ occurring in $x_{u,t}$ (measured as a canary string, a PII span from an NER tagger, or a fact tuple). Define the exposure predicate $\mathrm{Leak}(s, y) = 1$ if $y$ contains $s$ under a matching function — exact substring, normalized-edit-distance $\le \tau$, or an LLM judge scoring semantic disclosure. **Cross-user leakage rate** under adversary distribution $\mathcal{Q}$ over queries:
$$\mathrm{CUL} = \Pr_{q \sim \mathcal{Q},\ u' \ne \mathrm{owner}(s)}\big[\mathrm{Leak}(s, f(q, R(q,u',M))) = 1\big]$$

Decompose into a retrieval term and a generation term:
$$\mathrm{CUL} \le \underbrace{\Pr[\exists m \in R(q,u',M): s \subseteq \text{text}_m]}_{\text{retrieval breach } \rho} \cdot \underbrace{\Pr[\mathrm{Leak} \mid \text{breach}]}_{\text{emission } \varepsilon} + \underbrace{\Pr[\mathrm{Leak} \mid \neg\text{breach}]}_{\text{parametric/summary path } \pi}$$

$\rho$ is measured by instrumenting the retriever and string-matching canaries in the returned items. $\varepsilon$ is measured by conditioning on breaches and grading outputs. $\pi$ — leakage with no retrieval hit, via cross-user summaries, shared embeddings caches, or fine-tuning on logs — is the term nobody measures well, because it requires proving the retriever *did not* supply the content.

**Counterfactual/indistinguishability form.** Let $M_{-u}$ be the store with all of $u$'s writes removed. A memory system is $(\epsilon,\delta)$-**principal-isolating** if for all $u' \ne u$ and all measurable $S$,
$$\Pr[f(q, R(q,u',M)) \in S] \le e^{\epsilon}\Pr[f(q,R(q,u',M_{-u})) \in S] + \delta$$
Measured by paired-run estimation: run the same query set against $M$ and $M_{-u}$, compare output distributions. This is the honest definition, and also the expensive one — it needs a rebuilt index per held-out principal.

**Assumptions, and which break.** (a) *Ownership is a function.* Violated: shared documents, group chats, and forwarded content have several rightful owners. (b) *Secrets are extractable spans.* Violated: leakage is often inferential — an agent reveals that a colleague is job-hunting without quoting anything. (c) *Adversary queries are i.i.d. from $\mathcal{Q}$.* Violated: real extraction is adaptive and multi-turn. (d) *The store is append-only and static during measurement.* Violated: production stores are continuously written, so $\mathrm{CUL}$ is non-stationary.

## 3. State of the Art

**Established.** Namespace partitioning — one index per principal, enforced by a metadata filter at query time — reduces $\rho$ to the rate of filter bugs, and is what commercial vector stores (Pinecone, Weaviate, pgvector with row-level security) implement. Its correctness is an access-control property, not an ML one; where it holds, retrieval-path leakage is ruled out by construction. **Nothing establishes that $\pi$ is zero under partitioning**, because shared summarizers, shared caches, and log-based fine-tuning route around the partition.

**Established for the adjacent RAG setting.** Zeng et al. (*The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation*, ACL Findings 2024) show targeted prompts extract verbatim retrieved-corpus content. Qi et al. (*Follow My Instruction and Spill the Beans: Scalable Data Extraction from RAG Applications*, 2024) show instruction-following prompts extract large fractions of a retrieval datastore. Both measure a single shared corpus — a *proxy* for, not an instance of, cross-user leakage.

**Claimed but unablated.** Production memory products (OpenAI ChatGPT memory, Mem0, Zep, LangMem) state per-user scoping. No public red-team artifact reports a measured $\mathrm{CUL}$ with a stated adversary and matching function for any of them. Memory-benchmark numbers (LOCOMO, Maharana et al., ACL 2024; Mem0's reported accuracy gains) are *utility* numbers on single-user long conversations; they carry no isolation evidence.

**Benchmark-only results.** AgentDojo (Debenedetti et al., NeurIPS 2024 Datasets & Benchmarks) and PrivacyLens (Shao et al., NeurIPS 2024 D&B) report agent privacy/injection failure rates on synthetic suites. These are benchmark numbers on constructed environments, not deployment estimates.

## 4. What Is Known

- **Indirect prompt injection reaches agent memory.** Greshake et al. (AISec@CCS 2023) demonstrated that retrieved content can carry instructions the agent executes. If injected text is written into memory, the injection persists across sessions — the mechanism is demonstrated; the persistence rate at scale is not measured.
- **Models leak context under contextual-integrity pressure.** Mireshghallah et al. (ConfAIde, ICLR 2024): GPT-4 and ChatGPT reveal private information in inappropriate contexts in roughly 20–40% of tier-3/tier-4 scenarios, at a scale of a few hundred hand-built vignettes. This bounds $\varepsilon$ from below: even with correct retrieval, the generator will disclose.
- **Agents leak under benign task pressure.** PrivacyLens (2024) reports agent trajectories leaking sensitive information in a substantial minority of cases even when the model answers privacy-probe questions correctly — the gap between stated norm and executed behaviour is the finding, measured on ~500 seed scenarios.
- **Training-time memorization scales log-linearly** in model size, duplication count and prompt length (Carlini et al., *Quantifying Memorization Across Neural Language Models*, ICLR 2023; measured at 125M–6B on The Pile). Relevant only for the $\pi$ path when memory logs are used for fine-tuning; the coefficients do not transfer to retrieval.
- **Embeddings are invertible enough to matter.** Morris et al. (*Text Embeddings Reveal (Almost) As Much As Text*, EMNLP 2023) recover 32-token inputs exactly ~92% of the time on some corpora. A leaked or shared embedding index is therefore near-equivalent to leaked text.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted estimator for $\mathrm{CUL}$ in a live multi-tenant deployment. Canary insertion measures verbatim leakage of artificial strings and misses inferential leakage; LLM-judge disclosure scoring has unmeasured false-negative rates; the paired-run $(\epsilon,\delta)$ definition requires index rebuilds nobody performs. Until a matching function with known sensitivity exists, reported leakage rates are not comparable across systems.
- **Empirically open.** The magnitude of $\pi$ — leakage through cross-user summarization, shared caches, and log-derived fine-tuning — at $n \ge 10^5$ principals. Runnable today on any hosted agent with canaries; unrun publicly.
- **Empirically open.** Whether write-time redaction, read-time policy gating, or hard namespace partitioning sits on the utility/leakage Pareto frontier for shared-workspace agents where some sharing is desired.
- **Theoretically open.** Whether any non-trivial utility is achievable under a strict per-principal DP guarantee for *retrieval* over a store where each item comes from one principal. The natural construction (DP top-$k$ selection) has unbounded sensitivity when a single item is dispositive.

## 6. Why It Is Hard

**The obstruction is absent ground truth plus a non-identifiable emission path.** For any observed disclosure in a deployed agent, you cannot tell whether the content came from (i) a retrieval breach, (ii) a summary written by a cross-user consolidation job, (iii) parametric memorization from log fine-tuning, or (iv) coincidence — the model guessing a common name or plausible fact. Only (i) is observable by instrumenting the retriever. Distinguishing (iii) from (iv) requires a counterfactual model trained without $u$'s logs; distinguishing (ii) requires provenance tracking through summarization that no production system emits.

Compounding: the base rate is tiny. If true $\mathrm{CUL} = 10^{-4}$, detecting it at 20% relative precision needs $\sim 10^6$ adversarial queries per secret class — a real compute and API cost — and the store is non-stationary over the collection window, so the estimand moves while you estimate it.

## 7. Current Research (as of 2026)

- **Contextual-integrity agents.** AirGapAgent (Bagdasaryan et al., CCS 2024) minimizes context released to a task via a separate gatekeeper model; extending this to inter-principal boundaries rather than task boundaries is active *(frontier — verify)*.
- **Capability-based memory.** Information-flow-control designs where each memory item carries a label and the prompt assembler enforces a lattice — descendants of the CaMeL-style planner/executor split. Practical enforcement through free-text summarization remains unsolved *(frontier — verify)*.
- **RAG membership inference and datastore extraction.** Growing line following Zeng et al. and Qi et al.; the multi-tenant variant (infer *which principal* contributed an item) is emerging *(frontier — verify)*.
- **Agent security benchmarks** — AgentDojo, ToolEmu (Ruan et al., ICLR 2024), PrivacyLens — being extended to persistent-memory settings. Groups: ETH Zürich (Tramèr), Google DeepMind, UIUC, CMU, Anthropic/OpenAI internal red teams.

## 8. Concrete Next Experiment

**Question:** in a shared-memory agent, does namespace partitioning drive $\mathrm{CUL}$ to zero, or does the summarization path ($\pi$) dominate?

**Scale.** $n = 1{,}000$ simulated principals, 30 sessions each (30k sessions), on an open memory stack (Mem0 or LangMem + pgvector) with a mid-size open model as the agent. Inject one unique canary per principal — a 24-bit random token phrase bound to a fact ("my account PIN is `qz-8417-mrv`") — plus one *inferential* secret with no verbatim form ("planning to leave the company"). A nightly cross-user consolidation job (the realistic feature: "what are common questions this week?") writes org-level summaries.

**Arms.**
1. **Treatment:** partitioned retrieval + cross-user consolidation enabled.
2. **Control A:** partitioned retrieval, consolidation **disabled**. Isolates $\pi$.
3. **Control B:** unpartitioned shared index. Upper bound on $\rho$; validates the attack suite is potent.

**Probe set.** 200 adversarial queries per principal (direct ask, injection via a document written into memory, indirect elicitation), 200k queries per arm.

**Deciding number.** $\hat{\pi} = $ verbatim canary emission rate in Arm 1 minus Arm 2, with retrieval instrumented so every returned item is logged. **If $\hat{\pi} > 10^{-3}$ (i.e. >200 leaks in 200k probes) with a 95% CI excluding zero, namespace partitioning is insufficient and consolidation is the dominant leak channel — the field's default mitigation is wrong.** If $\hat{\pi} < 10^{-4}$ and Arm 3 shows $\rho > 0.1$, partitioning is confirmed sufficient for the verbatim path, and the open problem narrows to inferential leakage, reported separately as LLM-judge disclosure rate on the non-verbatim secrets.

## 9. Key References

- **[Foundational]** Helen Nissenbaum. *Privacy as Contextual Integrity.* Washington Law Review 79(1), 2004.
- **[Foundational]** Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[Foundational]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, Yejin Choi. *Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory.* ICLR, 2024. — arXiv:2310.17884
- **[SOTA]** Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu, Diyi Yang. *PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action.* NeurIPS Datasets & Benchmarks, 2024.
- **[SOTA]** Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, et al. *The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG).* Findings of ACL, 2024.
- **[SOTA]** Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade, Himabindu Lakkaraju. *Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems.* 2024.
- **[SOTA]** Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, et al. *AirGapAgent: Protecting Privacy-Conscious Conversational Agents.* ACM CCS, 2024.
- **[SOTA]** John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, Alexander M. Rush. *Text Embeddings Reveal (Almost) As Much As Text.* EMNLP, 2023. — arXiv:2310.06816
- **[Survey/Benchmark]** Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS Datasets & Benchmarks, 2024.
- **[Benchmark]** Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz. *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* AISec@CCS, 2023.
- **[Systems]** Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, et al. *MemGPT: Towards LLMs as Operating Systems.* 2023.
- **[Benchmark]** Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang. *Evaluating Very Long-Term Conversational Memory of LLM Agents.* ACL, 2024.

## 10. Worked Example

A workspace assistant serves one company: $n = 5{,}000$ employees, 20 sessions/employee/week. Retrieval is namespace-partitioned — a `WHERE owner_id = :caller` filter, audited, no bugs. So $\rho \approx 0$ on the direct path.

The product also runs a weekly consolidation: for each of 40 teams, an LLM summarizes that team's 2,000 sessions into a 500-token "team context" note, stored with `scope = team`, readable by every team member. This is the feature users asked for.

Employee A tells the assistant, in a 1:1 session: *"Don't put this in any shared note — I'm interviewing at a competitor, so schedule my Thursdays light."* The consolidation job sees 2,000 transcripts, including A's. Its instruction is "summarize recurring themes and scheduling constraints." It writes: *"One team member has recurring Thursday-afternoon external commitments through Q3 and has asked for lighter Thursday load."*

Now employee B, A's manager, asks: *"Who on the team has standing Thursday conflicts?"* Retrieval returns the team note — correctly, by policy. The agent answers with the summary line and, because B's own calendar memory contains only one person with blocked Thursdays, adds *"that looks like A."*

Run the numbers on the measurement. The canary estimator sees nothing: the string `interviewing at a competitor` never appears in any output. Verbatim $\mathrm{CUL} = 0$. The retrieval instrumentation logs a clean, policy-compliant read — no breach, so $\rho = 0$. An LLM judge scoring "did the response disclose A's private information?" scores this a leak — but on a 200-item hand-labeled calibration set of similar cases, the same judge has a false-negative rate that has never been reported for this task, so its estimate of $\pi$ carries no error bar.

Assume 1 in 500 sessions contains a similarly sensitive off-the-record disclosure. Over 100,000 sessions/week, that is 200 candidate items entering consolidation. If the summarizer preserves the inferential content of 5% of them — a plausible rate, unmeasured — that is 10 leaked-in-summary items per week, each readable by ~125 teammates. Verbatim measurement reports zero.

**The obstruction, made visible:** the access-control layer is provably correct, the canary estimator returns zero, and content still crosses the principal boundary — through a summarizer that strips provenance and an inference the model completes from two individually permitted facts. Fixing this needs a matching function with known sensitivity to *inferential* disclosure. That function does not yet exist, which is why this problem is methodologically blocked before it is empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*