---
id: 05-retrieval-and-agents/negative-result-detection-retrieval
title: "Negative Result Detection in Retrieval"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Negative Result Detection in Retrieval

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/negative-result-detection-retrieval` · **Status:** open

## 1. Problem Statement

A retriever always returns $k$ documents. It returns $k$ documents when the answer is on page one, and it returns $k$ documents when the corpus does not contain the answer at all. **Negative result detection** is the task of deciding, from the query and the returned list, which of those two worlds you are in — before a generator writes a fluent answer out of irrelevant context.

- **Input:** query $q$, corpus $\mathcal{C}$, the top-$k$ list $R_k(q)$ with scores.
- **Output:** a decision $\hat{y} \in \{\textsf{sufficient}, \textsf{insufficient}\}$, or a score to threshold.
- **Objective:** minimise abstention-weighted risk; equivalently, dominate the risk–coverage curve of the no-detector baseline.

Three variants, routinely conflated:

- **Measurement variant.** Can we even label ground truth? Deciding "the answer is not in $\mathcal{C}$" is a universally quantified claim over the whole corpus, and every benchmark approximates it by pooling a handful of systems' top-$k$. Currently the binding constraint.
- **Method variant.** Given labels, build a detector that beats max-score thresholding. Runnable now; results are weak.
- **Theory variant.** Is a calibrated per-query sufficiency score identifiable from a contrastively trained retriever at all? Argued below to be **not** identifiable without extra supervision.

Solved would mean: a detector that, at 90% coverage on an out-of-domain corpus, cuts unsupported-generation rate by half against a tuned max-score threshold, with the threshold transferring across corpora without per-corpus tuning.

## 2. Formal Setting

Corpus $\mathcal{C} = \{d_1,\dots,d_N\}$, scorer $s_\theta(q,d) \in \mathbb{R}$, ranked list $R_k(q)$ with scores $s_{(1)} \ge \dots \ge s_{(k)}$.

**Answerability.** Define the latent label
$$y(q) = \mathbb{1}\!\left[\exists\, E \subseteq \mathcal{C},\ |E| \le m,\ \textsf{entails}(E, a^\star(q))\right],$$
with $m$ the evidence budget (1 for single-hop, 2–4 for multi-hop). *As measured*, $\textsf{entails}$ is a human or LLM judgment over the **pooled** union of top-$k$ lists from a fixed system set $\mathcal{S}$, never over $\mathcal{C}$. So the observed label is $\tilde{y}(q) = \mathbb{1}[\exists E \subseteq \bigcup_{S\in\mathcal{S}} R_k^S(q)]$, and $\tilde{y} \le y$ always. Every reported "unanswerable" count is an upper bound on true unanswerability.

**Detector and risk.** Detector $d_\phi(q, R_k(q)) \in [0,1]$, abstain when $d_\phi < \tau$. With downstream answer correctness $c(q) \in \{0,1\}$ and abstention cost $\lambda$:
$$\mathcal{R}(\tau) = \mathbb{E}_q\big[\mathbb{1}[d_\phi \ge \tau](1 - c(q)) + \lambda\,\mathbb{1}[d_\phi < \tau]\big].$$
Report as coverage $\kappa(\tau) = \Pr[d_\phi \ge \tau]$ against selective risk $\mathcal{R}_{\text{sel}}(\tau) = \mathcal{R}_{\text{err}}(\tau)/\kappa(\tau)$; summarise by area under the risk–coverage curve (AURC). AUROC on $\tilde{y}$ alone is not sufficient — it ignores $\lambda$ and the cost asymmetry.

**The identifiability obstruction.** Dense retrievers are trained with in-batch InfoNCE:
$$\mathcal{L} = -\log \frac{\exp s_\theta(q,d^+)}{\sum_{j} \exp s_\theta(q,d_j)}.$$
For any per-query function $c(q)$, the map $s \mapsto s + c(q)$ leaves $\mathcal{L}$ exactly invariant. The training objective therefore identifies scores **only up to an arbitrary per-query shift**. "$s_{(1)} = 0.71$" carries no cross-query meaning by construction. Any threshold on raw score is fitting a nuisance parameter the loss never constrained.

**Assumptions, and which are violated.**
1. *A unique answer exists or does not.* Violated: ambiguous and time-dependent queries have answers that exist only relative to a reading (AmbigQA, FreshQA).
2. *Pooled judgments approximate corpus-level truth.* Violated at scale — pooling bias grows with $N$ and with retriever diversity.
3. *Scores are exchangeable across queries.* Violated by construction (above).
4. *Sufficiency is a property of the list.* Violated: a strong generator can answer from parametric memory with an irrelevant list, so downstream correctness is not a clean proxy for retrieval sufficiency.

## 3. State of the Art

**Established (reproduced, ablated).**
- *Query performance prediction (QPP)* is the mature ancestor: clarity score (Cronen-Townsend et al., SIGIR 2002), WIG and NQC score-distribution predictors (Zhou & Croft, SIGIR 2007; Shtok et al., TOIS 2012). These correlate with retrieval quality but were built for lexical rankers; Faggioli et al. (ECIR 2023) showed the same predictors degrade markedly on neural rankers, and that QPP evaluation itself is unstable.
- *Ranked-list truncation* — deciding $k$ per query — has trained models that beat fixed cutoffs (Lien et al., ICTIR 2019; Bahri et al., *Choppy*, SIGIR 2020). Established, but the objective is $F_1$ of the returned set, not sufficiency.
- *Extractive unanswerability* is largely solved when the negative is a single paragraph: SQuAD 2.0 (Rajpurkar et al., ACL 2018) moved from ~66 $F_1$ at release to above human agreement with pretrained encoders. This does **not** transfer to corpus-level negatives.

**Claimed but unablated.**
- Self-RAG's `ISREL` / `ISSUP` reflection tokens (Asai et al., ICLR 2024) and CRAG's lightweight retrieval evaluator with confidence thresholds (Yan et al., 2024) both report end-task gains, but neither isolates negative-detection AUROC from generation-quality gains, and both tune thresholds in-domain.
- LLM-as-judge sufficiency graders are widely deployed; their agreement with human sufficiency labels on *hard* negatives (topically on-target, evidentially empty) is largely a benchmark number, not an ablated finding.

**Benchmark-number-only.** NoMIRACL (Thakur et al., EMNLP Findings 2024) supplies a non-relevant subset per language and reports high hallucination rates for strong LLMs given only irrelevant passages. It is the cleanest existing probe, but its negatives are constructed, not naturally occurring.

## 4. What Is Known

- **Generators do not abstain on their own.** Given a retrieved set with no supporting evidence, frontier LLMs answer anyway at high rates — NoMIRACL reports hallucination rates on the non-relevant subset in the tens of percent for GPT-4-class models and substantially worse for smaller open models, across 18 languages (*approximate; direction is robust, exact figures vary by prompt*).
- **Popularity predicts retrieval need.** Mallen et al. (ACL 2023) showed on PopQA (14k entity questions) that parametric accuracy falls monotonically with entity popularity, and that adaptive retrieval gated on popularity beats always-retrieve. This is the strongest existing evidence that a *cheap query-side* signal carries real information about sufficiency.
- **Position matters more than presence.** Liu et al. (TACL 2024) found U-shaped accuracy in context position over 10–30 documents: evidence present but mid-context is nearly as bad as evidence absent. Sufficiency and usability come apart.
- **Out-of-domain score thresholds do not transfer.** BEIR (Thakur et al., NeurIPS 2021 Datasets) established across 18 datasets that dense retriever *ranking* quality drops out-of-domain; score *scales* shift with it, so a threshold tuned on MS MARCO is uncalibrated on, e.g., BioASQ or SciFact.
- **Conformal wrappers give guarantees on the wrong quantity.** TRAQ (Li et al., NAACL 2024) applies conformal prediction to RAG and obtains marginal coverage guarantees on answer sets; the guarantee is over exchangeable draws from the calibration distribution, which is exactly the assumption that fails on a new corpus.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no benchmark whose negatives are certified against the *full corpus*. Every "unanswerable" label is pool-derived, so a detector that flags a true negative the pool missed is scored as a false positive. Until $y$ and $\tilde{y}$ are separated, detector rankings are not trustworthy.
- **Theoretically open.** Whether a per-query calibrated sufficiency score is recoverable from an InfoNCE-trained retriever using unlabelled corpus statistics alone. The shift invariance above shows the *training loss* does not pin it down; it does not prove no consistent estimator exists from the learned geometry (e.g. local density of $\mathcal{C}$ around $q$). No result either way.
- **Empirically open.** Whether a detector using the full score *distribution* over the top-1000 plus corpus-density features beats tuned max-score thresholding on out-of-domain corpora. Runnable today on BEIR; nobody has published the head-to-head at 18-corpus scale with a shared protocol.
- **Empirically open.** Whether detection is better placed pre-retrieval (query-only) or post-retrieval. Mallen et al. is evidence for pre-; Self-RAG/CRAG assume post-; no controlled comparison exists.

## 6. Why It Is Hard

Two obstructions, both specific.

**Absent ground truth with a systematic direction.** Proving "not in $\mathcal{C}$" costs $O(N)$ verification per query. At $N = 10^7$ this is not annotatable, so labels come from pools of size $10^2$–$10^3$. The error is one-sided: pooling manufactures false negatives, and manufactured negatives are exactly the easy ones (topically off-target), while the negatives that matter in production are near-misses. Detectors are therefore trained and scored on the easy half of the distribution.

**Non-identifiability of the score scale.** The per-query shift invariance of the contrastive loss means "how confident is this retrieval" is not a quantity the model was ever asked to represent. Reported gains from score thresholds are gains from a nuisance parameter that happened to be stable within one training distribution.

A third, weaker: **the evaluation does not measure what it names.** End-task accuracy conflates retrieval sufficiency with parametric recall. A system that answers correctly from memory with an empty context scores as a detection success.

## 7. Current Research (as of 2026)

- **Reflection- and critic-token training** (Asai and collaborators, UW/AI2 lineage) — supervising abstention inside generation rather than as a separate gate.
- **Corrective / adaptive RAG** — routing to web search or re-query on a low evaluator score (CRAG lineage); most deployed RAG stacks now ship some variant.
- **Neural QPP** (Faggioli, Arabzadeh, Bagheri, Culpepper and co-authors) — the IR community's direct attack, converging with the LLM-side work under different vocabulary. This convergence is the most promising live thread.
- **Conformal and selective-prediction wrappers for RAG** — distribution-free coverage under exchangeability; the open question is corpus-shift-robust variants. *(frontier — verify)*
- **Corpus-certified negatives via exhaustive LLM verification** on small corpora ($N \le 10^5$) to build a gold-negative set. *(frontier — verify; I am not aware of a released dataset doing this.)*

## 8. Concrete Next Experiment

**Build the first corpus-certified negative set, then re-rank the detectors on it.**

- **Scale.** One corpus of $N = 10^5$ passages (e.g. the SciFact or NFCorpus BEIR collection, or a Wikipedia subset). 1,000 queries. For each query, run **exhaustive** sufficiency verification: score all $10^5$ passages with a strong LLM judge in a cheap first pass, human-adjudicate the top 200 by judge score. Cost: $10^8$ judge calls is infeasible — instead use a high-recall union of 8 heterogeneous retrievers at $k=1000$ (BM25, three dense, two late-interaction, two rewrite-then-retrieve), giving pools of $\sim$5k, and report the pool-size saturation curve to bound residual pooling bias.
- **Arms.** (A) Control: tuned max-score threshold $s_{(1)} > \tau$, $\tau$ fitted on in-domain dev. (B) Score-distribution detector: features $\{s_{(1)}, s_{(1)}-s_{(2)}, \text{NQC over top-100}, \text{entropy of softmax}(s_{1:100})\}$, logistic head. (C) Query-only detector (no retrieval). (D) LLM sufficiency judge over the top-10.
- **The deciding number.** AURC on the certified labels, at fixed abstention cost $\lambda = 0.5$. Secondary, and the real payoff: **the flip rate** — the fraction of queries whose label changes between pool-derived $\tilde{y}$ (pool size 100) and certified $y$ (pool size 5000). If the flip rate exceeds 10%, every published negative-detection AUROC on pooled benchmarks is uninterpretable, and the field's ordering of methods (B) vs (D) should be recomputed. If it is below 2%, the measurement objection is retired and the problem reduces to the method variant.

## 9. Key References

- **[Foundational]** Steve Cronen-Townsend, Yun Zhou, W. Bruce Croft. *Predicting Query Performance.* SIGIR, 2002.
- **[Foundational]** Anna Shtok, Oren Kurland, David Carmel, Fiana Raiber, Gad Markovits. *Predicting Query Performance by Query-Drift Estimation.* ACM TOIS, 2012.
- **[Foundational]** Pranav Rajpurkar, Robin Jia, Percy Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL, 2018. — arXiv:1806.03822
- **[Foundational]** Tom Kwiatkowski et al. *Natural Questions: A Benchmark for Question Answering Research.* TACL, 2019.
- **[SOTA]** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR, 2024. — arXiv:2310.11511
- **[SOTA]** Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling. *Corrective Retrieval Augmented Generation.* 2024. — arXiv:2401.15884
- **[SOTA]** Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing Chen, Mehdi Rezagholizadeh, Jimmy Lin. *NoMIRACL: Knowing When You Don't Know for Robust Multilingual Retrieval-Augmented Generation.* EMNLP Findings, 2024.
- **[SOTA]** Dara Bahri, Yi Tay, Che Zheng, Donald Metzler, Andrew Tomkins. *Choppy: Cut Transformer for Ranked List Truncation.* SIGIR, 2020.
- **[Empirical]** Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, Hannaneh Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL, 2023. — arXiv:2212.10511
- **[Empirical]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Benchmark]** Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks, 2021. — arXiv:2104.08663
- **[Method]** Shuo Li, Sangdon Park, Insup Lee, Osbert Bastani. *TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction.* NAACL, 2024.
- **[Survey]** Guglielmo Faggioli, Thibault Formal, Stefano Marchesin, Stéphane Clinchant, Nicola Ferro, Benjamin Piwowarski. *Query Performance Prediction for Neural IR: Are We There Yet?* ECIR, 2023.
- **[Survey]** David Carmel, Elad Yom-Tov. *Estimating the Query Difficulty for Information Retrieval.* Synthesis Lectures on Information Concepts, Retrieval, and Services, Morgan & Claypool, 2010.

## 10. Worked Example

Corpus: an enterprise wiki, $N = 240{,}000$ passages. Query: *"What is the retention period for customer voice recordings in the EU?"* The wiki documents retention for **chat transcripts** (EU, 90 days) and for **voice recordings** in the **US** (7 years). Nothing covers EU voice. True label $y = 0$.

Retrieval, a standard dense bi-encoder, cosine scores:

```
rank  score   passage
 1    0.742   "EU data retention: chat transcripts are held 90 days..."
 2    0.731   "Voice recording retention (US): 7 years from call date..."
 3    0.706   "EU GDPR overview: data minimisation principles..."
 4    0.689   "Voice recording storage architecture (all regions)..."
 5    0.671   "Customer data deletion requests, EU process..."
```

Calibration on 500 in-domain queries with pooled labels gave threshold $\tau = 0.62$; $s_{(1)} = 0.742 \gg \tau$, so the gate passes. Score gap $s_{(1)} - s_{(2)} = 0.011$, softmax entropy over the top-100 near-maximal — the distribution says "many things are equally on-topic", which is what a true negative looks like, but the max-score rule never sees it. The generator then composes 90 days (from #1) with voice (from #2) and outputs "90 days" with citations to two real passages. This is the failure mode that matters: **fully attributed and wrong**.

Now the obstruction. To score a detector on this query, the benchmark needs $y=0$. The pool of eight retrievers at $k=1000$ returns 4,300 distinct passages; none entail an answer, so $\tilde{y} = 0$ — the label happens to be right. But consider the sibling query *"...for customer voice recordings in Germany?"*, where a single 2019 DPA-compliance page buried at pooled rank 3,900 does state 6 months. At pool size 100 that page is absent and $\tilde{y} = 0$; at pool size 5,000 it is present and $y = 1$. A detector that correctly said "sufficient" is scored as a false positive at the small pool and as a true positive at the large one — **the same prediction, opposite grade, because of the pool**. Until the flip rate in §8 is measured, that is the noise floor under every published number in this area.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*