---
id: 23-privacy-memorization/group-privacy-duplicated-entities
title: "Group Privacy Degradation for Duplicated Entities"
topic: 23-privacy-memorization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Group Privacy Degradation for Duplicated Entities

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/group-privacy-duplicated-entities` · **Status:** open

## 1. Problem Statement

Differentially private training protects a *unit* — usually one training example, sometimes one user's shard. A real-world entity (a person, a medical record, a leaked credential) does not respect that unit. It appears in $k$ documents scattered across a web crawl, written by different authors, in different words. The formal guarantee for that entity is the **group privacy** guarantee at group size $k$, which degrades linearly in $\varepsilon$-DP and quadratically in $\rho$-zCDP. At the $k$ values that occur in web-scale corpora ($k$ in the hundreds to millions for any named public figure), the bound is numerically vacuous.

**Input:** a corpus $D$, a training algorithm $\mathcal{A}$ with a proven per-example guarantee, and an entity $e$ occurring in $k_e$ documents.
**Output:** a bound on what an adversary learns about $e$ from $\mathcal{A}(D)$.
**Solved** = either (a) a bound that stays non-vacuous as $k_e$ grows under a realistic (not worst-case) model of how duplicates relate, or (b) a demonstration that the worst-case bound is tight for entities in real corpora, so nothing better is possible.

Three variants that get conflated:

- **Measurement.** Define and estimate per-entity leakage $L(e)$ from a trained model, given that $k_e$ is itself unknown and requires entity resolution over the corpus. Currently ill-posed.
- **Method.** Train under an entity-level guarantee without knowing $k_e$ in advance. Requires either a cap on $k_e$ (which deletes the head of the distribution) or an adaptive-sensitivity mechanism.
- **Theory.** Is $k\varepsilon$ improvable when the $k$ records are *paraphrases* of one fact rather than $k$ arbitrary records? Worst-case answer is no; the conditional answer is open.

## 2. Formal Setting

Corpus $D = \{x_1,\dots,x_n\}$ of documents. An entity $e$ induces an occurrence set $S_e \subseteq D$ with $k_e = |S_e|$, measured by running an entity linker over $D$ and counting linked documents — not by string match, since aliases and paraphrases are the point. $k_e$ is therefore an *estimate* with its own error, typically 5–15% F1 loss on standard entity linking benchmarks.

Two databases are $k$-neighbours, $D \sim_k D'$, if they differ in at most $k$ records. The mechanism $\mathcal{A}$ satisfies $(\varepsilon,\delta)$-DP at $k=1$. Group privacy (Dwork, McSherry, Nissim, Smith, TCC 2006; Dwork & Roth 2014, Thm 2.2) gives, for $D \sim_k D'$ and all measurable $T$:

$$\Pr[\mathcal{A}(D)\in T] \le e^{k\varepsilon}\Pr[\mathcal{A}(D')\in T] + \frac{e^{k\varepsilon}-1}{e^{\varepsilon}-1}\,\delta .$$

Under zero-concentrated DP (Bun & Steinke, TCC 2016), $\rho$-zCDP implies $k^2\rho$-zCDP for groups of size $k$ — quadratic. Under Gaussian DP (Dong, Roth, Su, JRSS-B 2022), $\mu$-GDP implies $k\mu$-GDP; since $\mu \approx \sqrt{2\rho}$, this is the same statement.

Leakage is measured, not assumed. Define per-entity extraction leakage at prefix length $p$:

$$L_{\text{ext}}(e) = \Pr_{x \sim S_e}\big[\, \mathcal{M}(\text{prefix}_p(x)) = \text{suffix}(x) \,\big],$$

greedy decoding, exact match — the Carlini et al. (ICLR 2023) discoverable-memorization operationalization. Define per-entity membership advantage as the LiRA (Carlini et al., IEEE S&P 2022) AUC of a likelihood-ratio test on held-out shadow models, computed over *all* of $S_e$ jointly rather than per document.

Assumptions, with the ones known to be violated marked:

1. Records are exchangeable and the adversary's uncertainty factorizes. **Violated** — Kifer & Machanavajjhala (SIGMOD 2011) show correlated records make the marginal DP semantics weaker than the parameter suggests.
2. $k_e$ is known at training time. **Violated** — entity resolution over a 10T-token crawl is itself an unsolved, expensive problem.
3. Removing $S_e$ removes the fact. **Violated** — a fact deducible from held-out documents survives removal; the counterfactual is not the fact's absence.
4. The training pipeline's dedup step makes duplicates rare. **Partially violated** — near-duplicate paraphrases survive MinHash and exact-substring dedup.

## 3. State of the Art

**Theory SOTA (established).** The $k\varepsilon$ / $k^2\rho$ conversions above, and the fact that they are tight in the worst case: for randomized response there is an explicit $k$-neighbour pair achieving the bound. Kasiviswanathan & Smith (*J. Privacy and Confidentiality*, 2014) give the Bayesian semantics showing what does and does not survive correlation. No non-vacuous entity-level bound exists that exploits duplicate structure.

**Systems SOTA (established).** User-level DP training: McMahan et al. (ICLR 2018) for RNN LMs; Levy et al. (NeurIPS 2021) for user-level sample complexity; Charles et al. (2024) for user-level DP fine-tuning of LLMs. These bound the group correctly *when the group is a user account*. No production LLM pretraining run has a published entity-level guarantee.

**Mitigation SOTA (established for verbatim, unablated for entities).** Deduplication reduces measured memorization sharply (Lee et al., ACL 2022; Kandpal et al., ICML 2022). Claimed but unablated: that dedup reduces *entity-level* leakage. Ippolito et al. (INLG 2023) show verbatim-blocking filters give a false sense of privacy — paraphrased regurgitation survives.

**Benchmark-number-only results.** Min-K% Prob (Shi et al., ICLR 2024) and related pretraining-data detectors report AUCs on WikiMIA-style splits; Duan et al. (COLM 2024) show these splits are confounded by temporal distribution shift and that MIA on LLM pretraining is near chance once the confound is removed. Treat any per-entity leakage number derived from these detectors as unvalidated.

## 4. What Is Known

- **Memorization grows superlinearly in duplication.** Kandpal, Wallace & Raffel (ICML 2022): sequences duplicated 10× in the training set are regenerated roughly $10^3$× more often than sequences seen once — measured on 1.5B-parameter models trained on C4/Wiki-scale data.
- **Log-linear scaling in three variables.** Carlini et al. (ICLR 2023), GPT-Neo 125M–6B on the Pile: memorization increases log-linearly with model size, with number of duplicates, and with prefix length. At 6B and a 50-token prefix, roughly 1% of tested sequences are discoverably memorized.
- **Predictability across scale.** Biderman et al. (NeurIPS 2023), Pythia 70M–12B: which sequences get memorized is only weakly predictable from small models — low-precision transfer, so cheap proxies do not settle entity questions.
- **Extraction is cheap at production scale.** Nasr et al. (2023) extracted several thousand unique memorized training strings from a deployed chat model for roughly $200 of API spend.
- **Duplication also drives utility.** Kandpal et al. (ICML 2023): QA accuracy on an entity rises roughly log-linearly with the number of pretraining documents supporting it. Suppressing head entities costs capability, which is why capping $k_e$ is not free.
- **Group privacy arithmetic.** $\varepsilon = 8$ at the example level and $k = 200$ gives $\varepsilon_{\text{group}} = 1600$; under zCDP the multiplier is $k^2 = 4\times10^4$.

## 5. What Is Not Known

- **Theoretically open.** Whether any *conditional* group-privacy bound — assuming duplicates are paraphrases drawn from a common latent fact rather than arbitrary records — beats $k\varepsilon$. No proof either way. The no-free-lunch results rule out unconditional improvement, not conditional.
- **Empirically open.** The exponent $\alpha$ in $L(e) \propto k_e^{\alpha}$ for *semantic* (paraphrase) duplication, as opposed to verbatim duplication where $\alpha > 1$ is established. Runnable at 1–7B scale with injected canaries; not yet run.
- **Empirically open.** Whether example-level DP-SGD at practical $\varepsilon$ (1–10) reduces entity-level extraction at all for high-$k$ entities, or whether the empirical protection also collapses with $k$ the way the bound does.
- **Methodologically blocked.** Per-entity leakage $L(e)$ itself. There is no agreed estimator that separates *memorization of $S_e$* from *inference of $e$'s attributes from correlated evidence*. Counterfactual memorization (Zhang, Ippolito et al., NeurIPS 2023) is the right shape of definition but requires retraining held-out-$S_e$ models — $O(m)$ pretraining runs for $m$ entities.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the counterfactual combined with the cost of the only estimator that resolves it**. To attribute leakage to $S_e$ you must compare against a model trained on $D \setminus S_e$. For a high-$k$ public entity, $D \setminus S_e$ still supports the fact through implication, so the difference measures nothing clean; for a low-$k$ entity, the difference is real but you need one full pretraining run per entity to see it. Both failure modes point at the same gap: **there is no ground truth for "what the model learned about $e$"**, only for "what strings the model reproduces."

Second obstruction: **the evaluation does not measure what it names**. Verbatim discoverable memorization is what everyone reports, because it is cheap. Entity privacy is about facts, and facts survive paraphrase. Ippolito et al. (INLG 2023) is the direct demonstration.

## 7. Current Research (as of 2026)

- User- and group-level DP for LLM fine-tuning: Google (Ponomareva, Charles, McMahan and collaborators) — the user-level line is the most mature and is where an entity-level extension would land.
- Unlearning as a proxy for retroactive entity removal: TOFU (Maini et al., COLM 2024) and successors. Widely reported as brittle; unlearning a high-$k$ entity is not established to work. *(frontier — verify)*
- Contextual-integrity framings of what the unit of privacy should be for text: Brown, Bun, Feldman, Smith, Talwar (FAccT 2022) is the reference statement; follow-on work formalizing it is active. *(frontier — verify)*
- Semantic (non-verbatim) memorization metrics using entailment or QA probes rather than string match. Several groups; no consensus estimator. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does entity leakage scale superlinearly in $k$ under *paraphrase-only* duplication, and does example-level DP-SGD flatten that curve?

**Scale.** Pythia-1.4B architecture, trained from scratch on the deduplicated Pile (≈300B tokens, ~1 week on 64 A100s per arm — 4 arms). Inject 2,000 synthetic entities (fictitious name + 5 attributes each). Assign duplication counts $k \in \{1,2,4,\dots,512\}$, ~200 entities per bucket. Every occurrence is an independently LLM-generated paraphrase in a distinct document context; **zero verbatim overlap**, verified by exact-substring dedup before training.

**Arms.**
1. Non-private baseline.
2. **Control arm:** same entities, same $k$, but occurrences are verbatim copies. Isolates paraphrase versus string repetition.
3. DP-SGD at example level, $\varepsilon = 8$, $\delta = 10^{-8}$.
4. **Null control:** 200 entities generated but *not* inserted — gives the false-positive floor for the attribute-extraction probe.

**Measurement.** For each entity, a 20-question attribute-recall probe (exact-match on the 5 attributes, 4 phrasings each), scored against the null-control floor.

**The deciding number.** Fit $\log \text{recall}(k) = \alpha \log k + c$ over $k \in [1,512]$ in arm 1 and report $\alpha_{\text{para}}$ with a bootstrap CI.

- $\alpha_{\text{para}} > 1$ (CI excludes 1) ⇒ semantic leakage compounds superlinearly, dedup cannot help, and group-privacy accounting is *not* conservative for entities — the pessimistic reading is the correct one.
- $\alpha_{\text{para}} \le 1$ ⇒ the $k\varepsilon$ bound is loose for paraphrase structure, and the conditional theory question in §5 is worth attacking.

Secondary number: $\alpha_{\text{para}}^{\text{DP}}$ from arm 3. If it is statistically indistinguishable from $\alpha_{\text{para}}$, then $\varepsilon = 8$ example-level DP buys nothing for high-$k$ entities in practice, not just in the bound.

## 9. Key References

- **[Foundational]** Cynthia Dwork, Frank McSherry, Kobbi Nissim, Adam Smith. *Calibrating Noise to Sensitivity in Private Data Analysis.* TCC, 2006. — group privacy conversion.
- **[Foundational]** Cynthia Dwork, Aaron Roth. *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in TCS, 2014. — Theorem 2.2.
- **[Foundational]** Daniel Kifer, Ashwin Machanavajjhala. *No Free Lunch in Data Privacy.* SIGMOD, 2011.
- **[Foundational]** Mark Bun, Thomas Steinke. *Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds.* TCC, 2016. — $k^2\rho$ group bound.
- **[Foundational]** Shiva Kasiviswanathan, Adam Smith. *On the 'Semantics' of Differential Privacy: A Bayesian Formulation.* Journal of Privacy and Confidentiality, 2014.
- **[SOTA]** Nikhil Kandpal, Eric Wallace, Colin Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022.
- **[SOTA]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023.
- **[SOTA]** Stella Biderman et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023.
- **[SOTA]** Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023.
- **[SOTA]** Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022.
- **[SOTA]** H. Brendan McMahan, Daniel Ramage, Kunal Talwar, Li Zhang. *Learning Differentially Private Recurrent Language Models.* ICLR, 2018.
- **[Position]** Hannah Brown, Katherine Bun, Vitaly Feldman, Adam Smith, Kunal Talwar. *What Does it Mean for a Language Model to Preserve Privacy?* FAccT, 2022.
- **[Negative result]** Daphne Ippolito, Florian Tramèr, Milad Nasr, Chiyuan Zhang, Matthew Jagielski, Katherine Lee, Christopher Choquette-Choo, Nicholas Carlini. *Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy.* INLG, 2023.
- **[Negative result]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024.
- **[Survey]** Natalia Ponomareva, Hussein Hazimeh, Alex Kurakin, Zheng Xu, Carson Denison, H. Brendan McMahan, Sergei Vassilvitskii, Steve Chien, Abhradeep Guha Thakurta. *How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy.* JAIR, 2023.

## 10. Worked Example

Take a single individual, "J. Doe," a mid-profile person whose name appears in $k = 200$ crawled documents: a personal site, a company bio, 40 forum posts, 150 syndicated copies of one news item. Suppose the model was trained with DP-SGD at $\varepsilon = 8$, $\delta = 10^{-8}$ — already at the loose end of what practitioners deploy.

**Step 1 — the bound.** Group privacy at $k=200$:

$$\varepsilon_{\text{group}} = k\varepsilon = 1600, \qquad \delta_{\text{group}} = \frac{e^{1600}-1}{e^{8}-1}\cdot 10^{-8}.$$

$e^{1600}$ is about $10^{695}$. The multiplicative bound permits a likelihood ratio of $10^{695}$; the additive term exceeds 1. The guarantee for J. Doe is formally *nothing*. Under zCDP the story is the same by a different route: $\varepsilon=8,\delta=10^{-8}$ is roughly $\rho \approx 0.5$, and $k^2\rho = 2\times10^4$.

**Step 2 — how much noise would fix it.** To hold $\varepsilon_{\text{group}} = 8$ at $k=200$ you need per-example $\varepsilon = 0.04$. For DP-SGD the noise multiplier scales roughly as $1/\varepsilon$ in this regime, so that is a ~200× increase in $\sigma$. To hold utility you compensate with batch size, which scales as $\sigma^2$: a 4-million-fold increase in compute per step. That is the honest price of an entity-level guarantee at $k=200$ — and $k=200$ is small.

**Step 3 — where the obstruction becomes visible.** Neither number tells you what actually leaks. Run the counterfactual: retrain on $D \setminus S_{\text{Doe}}$ and probe for Doe's employer. The model still answers correctly — because the employer's own staff page, which never names Doe in the removed 200 documents, lists the team, and the news item was syndicated under a co-author's byline that *was* retained. So $L_{\text{ext}}$ under the counterfactual is near zero while the attribute is fully recoverable.

The bound says the leakage is unbounded. The verbatim metric says it is zero. The truth — that the attribute is recoverable but not *from* $S_{\text{Doe}}$ — is not expressible in either. That gap, not the arithmetic of $k\varepsilon$, is what keeps the problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*