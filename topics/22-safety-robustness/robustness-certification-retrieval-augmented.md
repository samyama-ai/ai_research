---
id: 22-safety-robustness/robustness-certification-retrieval-augmented
title: "Robustness Certification for Retrieval-Augmented Systems"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Robustness Certification for Retrieval-Augmented Systems

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/robustness-certification-retrieval-augmented` · **Status:** open

## 1. Problem Statement

A retrieval-augmented generation (RAG) system answers a query by retrieving passages from a corpus and conditioning a language model on them. An attacker who can write to the corpus — a wiki edit, an indexed web page, a shared drive, a poisoned package README — controls part of the model's input at inference time. The problem is to issue a **certificate**: a statement, provable rather than empirical, that the system's answer to a query is unchanged by any attacker-controlled corpus modification within a stated budget.

Three variants, with different difficulty:

- **Measurement.** Define a threat model and a robustness quantity that can be estimated from a finite benchmark without the estimate being an artifact of the benchmark's corpus size. Currently the weakest link.
- **Method.** Build a pipeline whose output is provably invariant to $\le k'$ corrupted passages *among the retrieved set*. Largely solved, at high accuracy cost (§3).
- **Theory.** Bridge a **corpus-level** budget ("attacker injects $m$ of $N$ documents") to a **retrieval-level** budget ("at most $k'$ of the top-$k$ are attacker-controlled"). This bridge is where the field is open. Certifying the generator is not certifying the system.

Solving it means: a deployable procedure that, for a given query and a corpus-level budget $m$, outputs either an answer plus a proof of invariance, or an abstention — with certified accuracy within a few points of undefended clean accuracy.

## 2. Formal Setting

Corpus $\mathcal{C} = \{d_1,\dots,d_N\}$, query $q$, retriever $R_\mathcal{C}: q \mapsto \mathcal{T}_k(q) \subseteq \mathcal{C}$, $|\mathcal{T}_k| = k$, generator $G$. System output $f(q,\mathcal{C}) = G(q, \mathcal{T}_k(q))$.

**Threat model (corpus-level).** $\mathcal{A}_m(\mathcal{C}) = \{\mathcal{C}\cup \mathcal{P} : |\mathcal{P}| \le m\}$ — injection of at most $m$ arbitrary documents of bounded length. Measured as: $m$ = number of rows the attacker writes to the index; $N$ = index cardinality reported by the vector store. Report $m$ and the ratio $m/N$; the ratio is what generalizes across corpora.

**Certified accuracy.** For a labelled set $\{(q_i, y_i)\}_{i=1}^n$,
$$\mathrm{CA}(m) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\Big[\forall\, \mathcal{C}' \in \mathcal{A}_m(\mathcal{C}):\ f(q_i,\mathcal{C}') = y_i\Big].$$
Measured not by search over $\mathcal{C}'$ — that is infeasible — but by a sound sufficient condition evaluated on the clean corpus. $\mathrm{CA}(m)$ is a lower bound on true robust accuracy; the gap between $\mathrm{CA}(m)$ and empirical attacked accuracy is the certificate's slack.

**Retrieval-level budget.** Existing certificates assume
$$|\mathcal{T}_k(q) \cap \mathcal{P}| \le k' \quad \text{for all } \mathcal{P}, |\mathcal{P}|\le m. \tag{A1}$$
Define the **amplification factor**
$$\rho(m) = \max_{|\mathcal{P}|\le m}\ \mathbb{E}_{q\sim \mathcal{Q}}\big[|\mathcal{T}_k(q)\cap\mathcal{P}|\big].$$
If retrieval were content-blind, $\rho(m) \approx km/N$. Adaptive attacks make $\rho$ many orders of magnitude larger.

**Isolate-then-aggregate certificate.** Compute $r_j = G(q, d_j)$ for each $d_j \in \mathcal{T}_k$ and aggregate by a function $\mathrm{Agg}$ that is provably stable to $k'$ arbitrary substitutions. For majority vote over answers with margin $\Delta$ between top-1 and runner-up counts, the certificate holds iff $\Delta > 2k'$. Cost: $k$ generator calls per query, versus 1.

**Assumptions known to be violated in practice.**
- *(A1) with small $k'$.* Adaptive corpus poisoning drives $k'\to k$ for targeted queries (§4). This is the load-bearing violation.
- *Passage independence.* Isolation destroys multi-hop answers that require two passages jointly; abstention is scored as failure, not as safe behaviour.
- *Static corpus.* Real indices are re-embedded and re-chunked; a certificate computed at time $t$ says nothing at $t+1$.
- *Answer equality.* Free-form generation has no exact-match equivalence; certificates are computed over normalized short answers or keywords, so the certified object is not the deployed output.

## 3. State of the Art

**Method SOTA (established).** *RobustRAG* (Xiang, Wu, Zhong, Wagner, Chen, Mittal, 2024) — isolate-then-aggregate with keyword and decoding-based aggregation, giving the first certifiable robust accuracy for RAG under (A1). The certification argument is sound and the code is public. Its scope is exactly the retrieval-level threat model; it certifies nothing about $\rho(m)$.

**Inherited machinery (established).** Randomized ablation and partition aggregation (Levine & Feizi, AAAI 2020; ICLR 2021) give $\ell_0$-style certificates for poisoning; erase-and-check (Kumar et al., 2023) certifies prompt-injection safety for bounded adversarial suffixes. All assume the adversarial fraction of the *model input* is bounded — the same assumption RAG cannot discharge.

**Claimed but unablated.** Defenses that filter poisoned passages by outlier detection, cross-passage consistency, or internal-knowledge conflict (e.g. Astute RAG, 2024; various "TrustRAG"-style pipelines, 2025) report large drops in attack success rate but issue no certificate, and are typically evaluated against the *non-adaptive* attacks they were designed on. There is no published adaptive-attacker ablation strong enough to establish that these survive an attacker who optimizes against the filter.

**Benchmark-number-only results.** RobustRAG's certifiable robust accuracies on RealtimeQA / NQ / Bio (7B–8B open models, $k=10$, $k'=1$) exist as a single table; independent replication at other $k$, other retrievers, and other corpora is thin. Treat the specific percentages as reported, not as reproduced regularities.

## 4. What Is Known

- **Corpus poisoning is cheap and scales sublinearly with corpus size.** Zhong, Huang, Mittal & Chen (EMNLP 2023) show a few tens of gradient-optimized adversarial passages, inserted into a Wikipedia-scale corpus ($\sim$21M passages), enter the top-$k$ for a large fraction of unseen NQ/MS MARCO queries against dense retrievers (Contriever, DPR). Transfer to BM25 is much weaker. Scale: 21M passages, $\le$500 injected.
- **Targeted poisoning needs $\sim$5 documents.** *PoisonedRAG* (Zou, Geng, Wang & Jia, USENIX Security 2025) reports $\approx$90% attack success rate injecting 5 crafted passages *per target question* into corpora of millions, for both black-box and white-box retrievers, across NQ, HotpotQA and MS MARCO with GPT-4 and open 7B generators. So $m/N \approx 10^{-6}$ suffices to set $k' \ge 1$ reliably.
- **Availability attacks work too.** *Machine Against the RAG* (Shafran, Schuster & Shmatikov, USENIX Security 2025) shows single "blocker" documents that induce refusal, a failure mode majority-vote aggregation does not certify against.
- **Indirect prompt injection through retrieval is a real deployment channel,** not a lab construct (Greshake et al., AISec 2023).
- **Certification costs accuracy and compute.** Isolate-then-aggregate multiplies generator calls by $k$ (10× at $k=10$) and drops certified accuracy well below clean accuracy, because single-passage answers are individually weak and multi-hop questions become uncertifiable by construction.
- **Position effects confound aggregation.** Generators weight passage position, not only content (Liu et al., TACL 2024), so a per-passage marginal is not the counterfactual the certificate implicitly assumes.

## 5. What Is Not Known

- **Theoretically open.** No bound on $\rho(m)$ for any deployed dense retriever. There is no theorem of the form "for embedding model $E$ with property $P$, at most $g(m)$ injected passages can enter the top-$k$ of any query." Without one, no corpus-level certificate exists — only conditional ones. Whether such a bound is achievable at all for unconstrained-text injection, or requires a retriever redesign (e.g. provenance-partitioned indices, Lipschitz-constrained encoders), is unresolved.
- **Empirically open.** Nobody has measured $\rho(m)$ as a function of $m/N$ across retriever families at web scale under a genuinely adaptive attacker. Runnable today; not run. Also open: whether hybrid BM25+dense retrieval reduces $\rho$ by orders of magnitude or by a constant.
- **Methodologically blocked.** Certifying free-form generation. $\mathrm{CA}(m)$ requires an equivalence relation on outputs; exact match on short answers is not it, and no semantic equivalence predicate is both decidable and faithful. Every published RAG certificate silently substitutes a keyword or short-answer proxy for the deployed output. Abstention accounting is also unsettled: a certified abstention is safe but useless, and current tables mix the two.

## 6. Why It Is Hard

The specific obstruction is a **threat-model mismatch that makes the evaluation not measure what it names.** "Certified robustness for RAG" is computed under (A1), an assumption about the *retrieved* set, while the attacker's actual budget is over the *corpus*. Retrieval is an adversarially optimizable, content-dependent selection step with no known stability guarantee: the attacker optimizes passage embeddings directly against the encoder, so the mapping from $m$ to $k'$ is under attacker control, not defender control. A certificate for $k'=1$ is therefore vacuous against an attacker who spends 5 documents to reach $k'=3$ — and PoisonedRAG shows 5 documents is the real price.

Secondary: **non-identifiability of the corrupted set.** Detection-based defenses cannot distinguish an injected passage from a genuinely unusual true passage without external ground truth about provenance, which the benchmark corpora do not carry. And **compute**: an honest adaptive evaluation requires re-optimizing the attack against each defense, which costs retriever-gradient runs per defense per corpus — the reason adaptive ablations are missing.

## 7. Current Research (as of 2026)

- **Certified aggregation for RAG** — Princeton/Berkeley line (Xiang, Mittal, Chen, Wagner) extending isolate-then-aggregate to longer-form outputs and to $k' > 1$ with tighter margins.
- **Provenance- and trust-weighted retrieval**, where the index is partitioned by source authority and aggregation is over partitions rather than passages — the RAG analogue of deep partition aggregation. *(frontier — verify)*
- **Adaptive attack benchmarks** for RAG defenses, standardizing the attacker's compute budget so filter-based defenses can be falsified. *(frontier — verify)*
- **Retriever-side certification**: encoders with certified embedding-space margins, so that a bound on injected-passage similarity translates into a bound on $k'$. Early, and in tension with retrieval quality. *(frontier — verify)*
- **Agentic RAG**, where retrieved content feeds tool calls, widening the consequence of a single corrupted passage beyond a wrong answer.

## 8. Concrete Next Experiment

**Measure $\rho(m)$ — the corpus-to-retrieval amplification factor.** This is the missing quantity that makes every existing RAG certificate conditional.

- **Scale.** Wikipedia-scale corpus, $N \approx 21$M passages (BEIR/NQ standard split). Two retrievers: Contriever-MS MARCO (dense) and a BM25+dense hybrid with reciprocal-rank fusion. $k=10$. 1,000 held-out NQ queries, none seen by the attack.
- **Attack arm.** Corpus-poisoning attack of Zhong et al. (gradient-based passage optimization against the encoder), sweeping $m \in \{10, 50, 250, 1000\}$, i.e. $m/N$ from $5\times10^{-7}$ to $5\times10^{-5}$. Attacker sees the encoder but not the query set.
- **Control arm.** The same $m$ passages sampled from a distractor corpus (non-adaptive injection), which pins $\rho$ at the content-blind baseline $km/N \le 5\times10^{-4}$.
- **Deciding number.** $\Pr_q[\,|\mathcal{T}_{10}(q)\cap\mathcal{P}| \ge 2\,]$ at $m = 250$ ($m/N \approx 10^{-5}$). If this exceeds **10%**, every published $k'=1$ RAG certificate is vacuous at a poisoning rate of one in a hundred thousand documents, and the field must certify the retriever, not the generator. If it stays below **0.1%** for the hybrid retriever, hybrid retrieval is a practical bridge from corpus-level to retrieval-level budgets and $k'=1$ certificates become meaningful.

Cost estimate: single 8×A100 node, days not weeks — encoder-gradient optimization plus 4 index builds. The experiment is unblocked; it is simply unrun at this scale with this control.

## 9. Key References

- **[Foundational]** J. Steinhardt, P. W. Koh, P. Liang. *Certified Defenses for Data Poisoning Attacks.* NeurIPS, 2017. — arXiv:1706.03691
- **[Foundational]** J. Cohen, E. Rosenfeld, J. Z. Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML, 2019. — arXiv:1902.02918
- **[Foundational]** A. Levine, S. Feizi. *Deep Partition Aggregation: Provable Defenses against General Poisoning Attacks.* ICLR, 2021. — arXiv:2006.14768
- **[SOTA]** C. Xiang, T. Wu, Z. Zhong, D. Wagner, D. Chen, P. Mittal. *Certifiably Robust RAG against Retrieval Corruption.* 2024. — arXiv:2405.15556
- **[SOTA]** W. Zou, R. Geng, B. Wang, J. Jia. *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models.* USENIX Security, 2025. — arXiv:2402.07867
- **[Attack]** Z. Zhong, Z. Huang, A. Mittal, D. Chen. *Poisoning Retrieval Corpora by Injecting Adversarial Passages.* EMNLP, 2023. — arXiv:2310.19156
- **[Attack]** A. Shafran, R. Schuster, V. Shmatikov. *Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents.* USENIX Security, 2025. — arXiv:2406.05870
- **[Attack]** K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, M. Fritz. *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* ACM AISec, 2023. — arXiv:2302.12173
- **[Related certification]** A. Kumar, C. Agarwal, S. Srinivas, S. Feizi. *Certifying LLM Safety against Adversarial Prompting.* COLM, 2024. — arXiv:2309.02705
- **[Context effects]** N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, P. Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172

## 10. Worked Example

A support assistant retrieves $k=10$ passages from an index of $N = 2\times10^6$ internal documents. Defense: isolate-then-aggregate with majority vote, certified against $k'=1$.

Query: *"What is the maximum refund window?"* True answer: 30 days. Clean per-passage votes: seven passages say "30 days", one says "60 days" (a stale doc), two abstain. Margin $\Delta = 7 - 1 = 6 > 2k' = 2$. **Certificate issued.**

Now the attacker injects $m = 5$ passages, each optimized against the encoder to score highly on refund-phrased queries, each asserting "90 days". Cost: 5 rows, $m/N = 2.5\times10^{-6}$ — the regime where PoisonedRAG reports $\approx$90% success. Suppose 4 of the 5 enter the top-10, displacing 4 true passages. New votes: three "30 days", four "90 days", one "60 days", two abstain. Majority output: **90 days**. Wrong.

Make the failure visible: the certificate was never violated. It said *"correct if at most 1 of the retrieved 10 is corrupted."* The attacker set $k' = 4$ for 5 documents. To certify against $k'=4$ the defense needs $\Delta > 8$ — at $k=10$ with two abstentions, at most 8 votes exist, so **no query in this configuration is certifiable at $k'=4$**, whatever the model.

The arithmetic that decides deployability is not the aggregation margin. It is $\rho(5)$ — how many of 5 injected documents reach the top-10. Content-blind, the expected count is $km/N = 10\cdot 5/2\times10^6 = 2.5\times10^{-5}$. Observed under adaptive attack: $\approx 4$. That is an amplification of $\sim 10^5$, and it is the number nobody has systematically measured. §8 measures it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*