---
id: 05-retrieval-and-agents/corpus-poisoning-detection-retrieval-time
title: "Corpus Poisoning Detection at Retrieval Time"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Corpus Poisoning Detection at Retrieval Time

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/corpus-poisoning-detection-retrieval-time` · **Status:** open

## 1. Problem Statement

A retrieval-augmented generation (RAG) system serves a query $q$ by retrieving $k$ passages from a corpus $\mathcal{C}$ and conditioning a generator on them. An adversary who can write into $\mathcal{C}$ — via a wiki edit, a scraped web page, an indexed support ticket, an agent's shared memory — injects a small set of passages crafted to be retrieved for chosen queries and to steer the answer.

**The problem:** given only the query and the $k$ retrieved passages, at serving time, decide which (if any) are adversarial, without a clean reference copy of the corpus and without materially degrading answers on clean queries.

Three variants, of very different difficulty:

- **Measurement.** Define a detection task whose reported numbers predict deployed behaviour. Requires fixing the conditional base rate (fraction of poisoned passages *among retrieved ones*, not among the corpus), the adaptive-attacker assumption, and the utility cost of false positives. Currently the weakest link.
- **Method.** Build a detector $D$ with high true-positive rate at a false-positive rate low enough that clean nDCG@$k$ is unharmed, against attacks that know $D$.
- **Theory.** Bound the achievable detection rate as a function of the attacker's injection budget $m$ and the retriever's embedding geometry — or prove that for $m \ge 1$ and a naturalness-constrained attacker, no query-local detector can beat chance.

Solving it means: a serving-time filter that cuts end-to-end attack success rate (ASR) by an order of magnitude against an adaptive attacker while costing under 1 point of clean nDCG@10 and under ~50 ms per query.

## 2. Formal Setting

Corpus $\mathcal{C} = \mathcal{C}_{\text{clean}} \cup \mathcal{A}$, with $N = |\mathcal{C}_{\text{clean}}|$ and $m = |\mathcal{A}|$ injected passages; injection budget $\rho = m/N$, measured as passages written divided by corpus size at index time.

Dual-encoder retriever: $E_q, E_d : \Sigma^* \to \mathbb{R}^h$, score $s(q,d) = \langle E_q(q), E_d(d)\rangle$. Retrieved set
$$\mathcal{R}_k(q) = \operatorname*{arg\,top-}k_{d \in \mathcal{C}} \; s(q,d).$$

Attacker picks a target query distribution $\mathcal{Q}_t$ and solves, for each adversarial passage $a$,
$$\max_{a \in \Sigma^L} \; \mathbb{E}_{q\sim\mathcal{Q}_t}\big[s(q,a)\big] \quad \text{s.t.} \quad \Phi(a) \le \tau,$$
where $\Phi$ is a naturalness constraint — measured in practice as generator perplexity $\text{PPL}(a)$ under a held-out LM, or as the passage being fluent LLM output by construction.

Detector $D(q, \mathcal{R}_k(q)) \to [0,1]^k$, thresholded at $t$. Measured quantities:

- $\text{TPR}(\alpha)$ — fraction of adversarial passages flagged, at the threshold $t$ where the flag rate on a held-out clean corpus equals $\alpha$. Both estimated on the *retrieved* distribution, not a balanced sample.
- $\text{ASR} = \Pr[\text{generator emits attacker target} \mid q \sim \mathcal{Q}_t]$, judged by exact substring match or a fixed LLM judge with the prompt published.
- Utility cost $\Delta = \text{nDCG@}k^{\text{clean}} - \text{nDCG@}k^{\text{filtered}}$ on the unpoisoned benchmark.
- Latency $\ell$ — added wall-clock ms per query at the serving batch size.

Assumptions, with those known to break in practice flagged:

1. *$\mathcal{C}_{\text{clean}}$ contains no adversarial-looking text.* **Violated.** Real corpora carry SEO spam, keyword-stuffed boilerplate, and template text with the same high-perplexity, high-query-similarity signature.
2. *Detector sees passages i.i.d.* **Violated.** Poisoned passages arrive in correlated clusters — several near-duplicates targeting one query.
3. *Attacker does not know $D$.* **Violated by assumption of the threat model**; nearly all published defenses are evaluated non-adaptively.
4. *A clean reference corpus or trusted subset exists.* Usually **unavailable** for open-web or user-contributed indices.
5. *Poisoning is rare.* True corpus-wide ($\rho \sim 10^{-6}$) but false conditionally: the attack's whole purpose is to make $\Pr[\text{poisoned} \mid d \in \mathcal{R}_k(q_t)]$ large. Detectors tuned on the marginal rate are tuned on the wrong distribution.

## 3. State of the Art

**Attacks (established, reproduced).** Zhong et al. (EMNLP 2023) show HotFlip-style gradient optimization of ~50 passages, inserted into MS MARCO's 8.8M-passage corpus, is retrieved for >94% of NQ test queries against Contriever, with cross-domain transfer. Zou et al. (PoisonedRAG, USENIX Security 2025) show 5 fluent LLM-written passages per target question — $\rho \approx 10^{-6}$ on NQ/HotpotQA/MS MARCO — reach ~90% ASR against GPT-4-class generators. Chaudhari et al. (Phantom, 2024) and Chen et al. (AgentPoison, NeurIPS 2024) extend this to trigger-conditioned attacks and to agent memory/knowledge bases; AgentPoison reports >80% retrieval success from poisoning under 0.1% of the memory store.

**Defenses.**
- *Established:* perplexity filtering (the baseline of Jain et al., 2023) reliably catches gradient-optimized token soup — Zhong et al.'s passages have PPL orders of magnitude above corpus median — and reliably fails on fluent LLM-written passages.
- *Established, with a proof:* RobustRAG (Xiang, Wu, Zhong, Wagner, Chen, Mittal, 2024) isolates each retrieved passage, generates per-passage, then aggregates by keyword or decoding-level voting, giving a certifiable lower bound on accuracy under up to $k'$ corrupted passages. This is *robustness*, not detection, and it costs $k$ generator calls per query.
- *Claimed but unablated:* clustering/self-assessment filters such as TrustRAG (2025) and knowledge-conflict resolvers such as AstuteRAG (Wang et al., 2024) report large ASR reductions, but almost entirely against non-adaptive attackers, on the same three QA benchmarks, at balanced or near-balanced poison ratios in the retrieved set.
- *Benchmark-number-only:* most reported detection AUROCs come from held-out sets constructed by mixing a fixed number of known adversarial passages with clean ones. No published number is a TPR at a deployment-realistic FPR on an untouched production index.

## 4. What Is Known

- **Tiny budgets suffice.** 5 passages against a corpus of $\sim2.7\times10^6$ (NQ, PoisonedRAG) — $\rho \approx 1.9\times10^{-6}$ — give ~90% ASR. Scale: three standard QA corpora, Contriever/ANCE/DPR retrievers, $k=5$.
- **Optimized passages are geometric outliers.** Zhong et al. report adversarial passage embeddings sit anomalously close to the centroid of the query embedding distribution; this is what makes them retrieve for *many* queries and also what makes them detectable.
- **Naturalness kills that signal.** PoisonedRAG-style passages are ordinary LLM prose; perplexity and embedding-norm filters lose most of their separation. No independently reproduced detector achieves >90% TPR at <1% FPR on them.
- **Certified defense is expensive and weak in coverage.** RobustRAG's certificates hold for extractive/short-form QA with $k$ separate generations; certified accuracy degrades sharply as the corrupted fraction of $\mathcal{R}_k$ grows past ~20%.
- **Jamming needs even less.** Shafran, Schuster, Shmatikov (USENIX Security 2025) show a *single* blocker document can suppress an answer — a denial-of-service variant with $m=1$.

## 5. What Is Not Known

- **Methodologically blocked (the main gap).** There is no agreed evaluation protocol that fixes the conditional base rate, includes an adaptive attacker, and reports the utility cost. Published AUROCs are therefore not comparable to each other and do not predict deployed FPR. Until this is fixed, "detector A beats detector B" is not a well-posed claim.
- **Empirically open.** Whether any query-local detector reaches ≥90% TPR at ≤0.1% FPR against fluent, adaptively-optimized passages, at $N \ge 10^7$ with real web noise. The experiment is runnable today on public corpora; nobody has published it.
- **Theoretically open.** No lower bound on detectability as a function of $(\rho, k, h, \tau)$. Specifically: is there an $L$-token, PPL-bounded passage that is $\epsilon$-close in embedding space to the clean top-$k$ manifold for target $q$ yet still ranks first? If yes for all reasonable $\tau$, query-local detection is information-theoretically hopeless and defense must move to provenance or aggregation.

## 6. Why It Is Hard

**Base-rate mismatch between the evaluation and the deployment.** Detectors are scored on sets where poisoned passages are 10–50% of the sample, because that is what makes AUROC informative. Deployment has two regimes: corpus-scan, where the poison rate is $10^{-6}$ and a 1% FPR yields ~$10^4$ false alarms per true one; and retrieval-time on a *targeted* query, where the rate can exceed 50%. A single AUROC number averages over two regimes that differ by five orders of magnitude in prior, so it predicts neither.

Secondary, but real: **absent ground truth** — no production index has verified poison labels, so FPR is estimated against a corpus that may already be poisoned; and **non-identifiability** — a fluent passage asserting a false fact that is highly relevant to $q$ is, in feature space, the same object as a fluent passage asserting a true minority-view fact. Any detector separating them is doing fact-checking, not anomaly detection.

## 7. Current Research (as of 2026)

- **Certified aggregation.** Extending RobustRAG-style isolate-then-aggregate beyond short-form QA to long-form and tool-calling agents (Princeton, Berkeley) — the certificate is the only guarantee currently on offer. *(frontier — verify current scope.)*
- **Token-level and gradient-based screening.** Masked-token-probability screens that flag passages whose tokens are improbable given the rest of the passage, aimed at catching optimized triggers that survive whole-passage perplexity. Reported at ACL-venue scale in 2025. *(frontier — verify.)*
- **Provenance over content.** Signed corpora, per-document trust scores, write-path rate limiting — sidesteps detection entirely; adoption-limited, not research-limited.
- **Agent memory poisoning.** Post-AgentPoison work on persistent poisoning of agent scratchpads and shared vector memory (UIUC, UChicago). Detection here is harder: the corpus is written by the agent itself, so provenance gives nothing.
- **Adaptive-attack audits.** A small but growing set of papers re-running published defenses against attackers optimizing through the defense; early results are consistently that non-adaptive ASR reductions shrink substantially. *(frontier — verify magnitudes.)*

## 8. Concrete Next Experiment

**Question:** does any query-local detector survive an adaptive attacker at a deployment-realistic false-positive rate?

**Scale.** MS MARCO passage corpus, $N = 8.84\times10^6$, Contriever-MS MARCO retriever, $k=10$. 500 target queries drawn from NQ. Injection budget $m = 5$ passages per target query ($\rho = 2.8\times10^{-4}$ total). Three attack arms: (a) HotFlip-optimized (Zhong et al.), (b) fluent LLM-written (PoisonedRAG), (c) **adaptive** — fluent passages optimized with the detector's score added to the attacker objective as a penalty, $\max_a s(q,a) - \lambda D(q,a)$.

**Control arms.** (i) Unpoisoned index, same 500 queries, to measure $\Delta$nDCG@10 from filtering alone. (ii) A deliberately weak detector — random flagging at matched FPR — to confirm any ASR reduction is not just from dropping passages. (iii) A "spam-only" arm: 5 real SEO-spam passages from the corpus per query, never adversarial, to measure how much of the TPR is actually spam detection.

**Decision number.** **TPR at FPR = 0.1%**, where the FPR threshold is calibrated on 100k *retrieved* clean passages from the unpoisoned control, reported separately for each attack arm. A detector clearing **TPR ≥ 0.90 on arm (c)** with $\Delta$nDCG@10 $\le 1.0$ point moves the problem from open to partially solved. Below TPR 0.50 on arm (c), query-local detection should be declared a dead end and effort redirected to aggregation and provenance. Secondary readouts: end-to-end ASR before/after, and $\ell$ in ms.

Cost estimate: one 8×A100-day for indexing plus attack optimization; the generator calls dominate at roughly $10^4$ queries × 3 arms.

## 9. Key References

- **[Foundational]** Zexuan Zhong, Ziqing Huang, Alexander Wettig, Danqi Chen. *Poisoning Retrieval Corpora by Injecting Adversarial Passages.* EMNLP, 2023. — arXiv:2310.19156
- **[Foundational/SOTA attack]** Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia. *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models.* USENIX Security, 2025. — arXiv:2402.07867
- **[SOTA defense]** Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, Prateek Mittal. *Certifiably Robust RAG against Retrieval Corruption.* 2024. — arXiv:2405.15556
- **[Attack]** Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, Bo Li. *AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.* NeurIPS, 2024.
- **[Attack]** Avital Shafran, Roei Schuster, Vitaly Shmatikov. *Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents.* USENIX Security, 2025.
- **[Attack]** Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea. *Phantom: General Trigger Attacks on Retrieval Augmented Language Generation.* 2024. — arXiv:2405.20485
- **[Defense baseline]** Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein. *Baseline Defenses for Adversarial Attacks Against Aligned Language Models.* 2023. — arXiv:2309.00614
- **[Defense]** Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, Sercan Ö. Arık. *Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models.* 2024. — arXiv:2410.07176
- **[Survey]** Bang An et al. *Towards Trustworthy Retrieval-Augmented Generation: A Survey.* 2025. — verify identifier before citing.

## 10. Worked Example

Take NQ with Contriever, $N = 2.68\times10^6$ passages, $k=10$. PoisonedRAG injects $m=5$ passages for the target query $q_t$ = *"who wrote the book the origin of species"*, each of the form "*[query restatement]. The Origin of Species was written by Alfred Russel Wallace in 1859…*". All 5 land in the top-10.

Now run a detector with **AUROC 0.95** — a strong published-looking number — on two regimes, using the operating point TPR = 0.95 at FPR = 0.01.

**Retrieval-time, targeted query.** 5 poisoned, 5 clean in $\mathcal{R}_{10}$.
$$\text{TP} = 0.95\times5 = 4.75,\quad \text{FP} = 0.01\times5 = 0.05,\quad \text{precision} = \tfrac{4.75}{4.80} = 0.99.$$
Excellent. This is the number papers report.

**Corpus scan, same detector.** 5 poisoned in $2.68\times10^6$:
$$\text{TP} = 4.75,\quad \text{FP} = 0.01\times(2.68\times10^6) = 26{,}800,\quad \text{precision} = \tfrac{4.75}{26{,}805} = 1.8\times10^{-4}.$$
5,600 false alarms per true detection.

**Retrieval-time, the other 99.9% of traffic.** Clean queries retrieve 10 clean passages; expected FP $= 0.1$ per query. At 100k queries/day, that is **10,000 clean passages dropped per day**, and $1 - 0.99^{10} = 9.6\%$ of all queries lose at least one top-10 passage. On MS MARCO, dropping a random top-10 passage costs roughly 0.3–0.5 nDCG@10 points in expectation — comparable to the entire gap between successive retriever generations.

**The obstruction, made visible.** The same detector, unchanged, is 99% precise and 0.018% precise depending only on which prior you evaluate it under. AUROC 0.95 is simultaneously a deployable defense and a useless one. And the fix is not a better threshold: to get FP $\le 0.001$ per clean query you need FPR $\le 10^{-4}$, four orders below where these detectors are actually characterized — and no published paper reports TPR at that FPR, against any attack, let alone an adaptive one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*