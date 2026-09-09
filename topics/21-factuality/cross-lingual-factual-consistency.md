---
id: 21-factuality/cross-lingual-factual-consistency
title: "Cross-Lingual Consistency of Factual Assertions"
topic: 21-factuality
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Consistency of Factual Assertions

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/cross-lingual-factual-consistency` · **Status:** empirically-open

## 1. Problem Statement

A single model, asked the same factual question in English, Hindi, Swahili and Japanese, returns different answers. Not merely different surface forms — different entities. The problem is to say why, to measure it in a way that is not confounded by translation and tokenization artifacts, and to build models whose factual assertions are invariant to the language of the query.

Three variants, of very different difficulty:

- **Measurement.** Given a model $M$, a set of facts, and a set of languages $L$, produce a score that is high iff $M$ asserts the same fact in every language — and that does not move when you change the prompt template, the tokenizer, or the translation direction. Currently the hardest of the three, because most published scores conflate consistency with per-language competence.
- **Method.** Given a fact learned or edited in language $\ell_1$, make it retrievable in $\ell_2$ without retraining. Knowledge editing propagates across languages far worse than within one; the engineering question is whether this is fixable at the editing layer or requires a different pretraining objective.
- **Theory.** Under what conditions does a shared multilingual representation force factual agreement? No non-trivial theorem exists. It is not known whether "the model has one fact store queried through language-specific interfaces" and "the model has $|L|$ partly-overlapping fact stores" are distinguishable from behaviour alone.

Solving it means: a metric that separates *consistency* from *accuracy*, plus a model at frontier scale whose cross-lingual agreement on verifiable facts is within a few points of its within-language paraphrase agreement.

## 2. Formal Setting

Let $\mathcal{K} = \{(s, r, o)\}$ be a set of triples over an entity vocabulary with language-independent identifiers (Wikidata QIDs), $L$ a language set, and $T_\ell(r)$ a set of verbalization templates for relation $r$ in language $\ell$. A query is $q = T_\ell(r)[s_\ell]$, where $s_\ell$ is the label of $s$ in $\ell$.

**Answer as measured.** The model emits a string; scoring requires mapping it to an entity. Define the grounding map $g_\ell: \Sigma^* \to \mathcal{E} \cup \{\bot\}$ by exact or alias match against the Wikidata label set for $\ell$. Then
$$\hat{o}_\ell(q) = g_\ell\big(\arg\max_{y} P_M(y \mid q)\big).$$
For masked or constrained scoring, restrict to a candidate set $C \subset \mathcal{E}$ and rank by length-normalized log-likelihood $\frac{1}{|y|}\log P_M(y \mid q)$.

**Accuracy** in language $\ell$: $A_\ell = \frac{1}{|\mathcal{K}|}\sum \mathbb{1}[\hat{o}_\ell = o]$.

**Naive agreement** between $\ell_1, \ell_2$: $\mathrm{Agr} = \frac{1}{|\mathcal{K}|}\sum \mathbb{1}[\hat{o}_{\ell_1} = \hat{o}_{\ell_2}]$. This is the number most papers report and it is not a consistency measure: it is bounded above by $\min(A_{\ell_1}, A_{\ell_2}) + (\text{shared errors})$, so a model that is uniformly wrong scores high and a model strong in one language and weak in another scores low for reasons that have nothing to do with representation sharing.

**Ranking consistency (RankC, Qi et al. 2023)** avoids the tie to correctness by comparing ranked candidate lists rather than top-1 entities. For ranked lists $R_{\ell_1}, R_{\ell_2}$ over $C$,
$$\mathrm{RankC}(\ell_1,\ell_2) = \frac{1}{|\mathcal{K}|}\sum_{q}\frac{\sum_{k=1}^{|C|} P@k(q)\cdot w_k}{\sum_k w_k},\qquad P@k = \frac{|R^{(k)}_{\ell_1}\cap R^{(k)}_{\ell_2}|}{k},$$
with $w_k$ weighting shallower ranks. It is symmetric-ish and correctness-free.

**Template variance control.** Report the within-language paraphrase agreement $\mathrm{Agr}(\ell,\ell)$ over distinct templates $T_\ell(r)$. The quantity of interest is the *excess* inconsistency
$$\Delta = \mathrm{Agr}(\ell,\ell) - \mathrm{Agr}(\ell_1,\ell_2),$$
because a model that is 30% inconsistent under English paraphrase is not additionally interesting for being 35% inconsistent across languages.

**Assumptions, and which are violated.**
1. *Facts are language-independent.* Violated for a substantial slice: legal status, place names, disputed borders, and culturally indexed relations ("national dish") have genuinely different correct answers per locale.
2. *$g_\ell$ is surjective onto answers.* Violated — alias coverage in Wikidata is far thinner for low-resource languages, so ungrounded outputs ($\bot$) are systematically more frequent there, and $\bot$ is silently scored as an error.
3. *Templates are meaning-equivalent across $\ell$.* Violated by translation drift; a cloze template translated by MT changes the relation ("place of birth" → "hometown").
4. *Tokenization is comparable.* Violated: byte-level fertility for Amharic or Telugu is several times English, so length-normalized ranking is not calibrated across $\ell$.
5. *Test facts are not in the pretraining corpus in one language only.* Almost always violated and almost never checked.

## 3. State of the Art

**Established (measurement).**
- X-FACTR (Jiang, Anastasopoulos, Araki, Neubig, EMNLP 2020) and mLAMA (Kassner, Dufter, Schütze, EACL 2021) established that multilingual masked LMs retrieve far more facts in English than in other languages, using the same triples. This is reproduced across model families.
- Fierro & Søgaard (Findings of ACL 2022) extended ParaRel-style paraphrase consistency to 45 languages and found multilingual models are *less* consistent than the English-only models, on the same relations.
- RankC (Qi, Fernández, Bisazza, EMNLP 2023) is the field's best correctness-decoupled metric and is the load-bearing contribution: it showed cross-lingual consistency does not rise with model size within the families tested, while accuracy does.

**Established (mechanism, partial).** Wendler, Veselovsky, Monea, West (ACL 2024) showed via logit-lens that Llama-2 passes through an English-token-aligned intermediate space on non-English prompts. Dumas et al. (2024) showed activation patching can transplant a "concept" across languages while leaving the output language intact. Together these support a partially shared latent store — but neither measures factual agreement, so the link to consistency is inferential.

**Claimed but unablated.**
- That RLHF or instruction tuning on multilingual data improves consistency. Reported as benchmark deltas without an arm that controls for the tuning data containing the test facts.
- That "translate-then-answer" pipelines fix the problem. They raise accuracy in low-resource languages, but no published ablation separates gains from translation quality versus gains from consistency, and the pipeline trivially maximizes agreement by construction — it is a degenerate solution, not a measurement.

**Benchmark-number-only.** Multilingual hallucination scores from Mu-SHROOM (SemEval-2025 Task 3, Vázquez et al.) and cross-lingual editing scores on Bi-ZsRE (Wang et al., ACL 2024) exist as leaderboard numbers with no independent reproduction of the ranking at the time of writing.

## 4. What Is Known

- **English dominance is large and stable.** mLAMA on mBERT (110M params, 53 languages): English precision-at-1 several times that of the median language on identical triples. Reproduced on mT5 and XLM-R.
- **Consistency does not scale with parameters.** Qi et al. measured RankC on XLM-R (up to 550M), mT5 (up to 3.7B) and BLOOM (560M–3B, plus 176B in follow-ups): accuracy rose monotonically with scale, RankC did not. Most language pairs sat well below 0.5; typologically and script-distant pairs lowest. This is the single most important known regularity, and it is *scale-refuting*, not scale-confirming.
- **Consistency tracks language relatedness and script sharing, not per-language accuracy.** Same source.
- **Editing propagates poorly.** ROME/MEMIT-style edits applied in one language transfer to another at rates strongly correlated with the pre-existing RankC of that pair — i.e. the edit follows shared structure that already existed rather than creating it. Measured at ≤7B scale.
- **Cross-lingual editing benchmarks (Bi-ZsRE, MLaKE) show large drops** between same-language and cross-language reliability for every editor tested, at 7B scale.
- **Multilingual hallucination detection is weaker than English.** HalOmi (Dale et al., EMNLP 2023) showed detectors tuned on high-resource pairs degrade sharply on low-resource ones.

## 5. What Is Not Known

- **Methodologically blocked.** Whether observed inconsistency is factual inconsistency or grounding/tokenization artifact. No published benchmark jointly controls alias coverage, tokenizer fertility, template translation fidelity, and the within-language paraphrase baseline $\Delta$. Until it does, every cross-lingual consistency number has an unbounded confound.
- **Methodologically blocked.** Which facts are legitimately locale-dependent. No dataset labels this; it is currently handled by hoping the relation set avoids it.
- **Empirically open.** Whether consistency scales at frontier size. The scale-refutation above tops out around 3B–7B dense models. Nobody has run RankC-style measurement on a 2026-class frontier model with the confound controls in place. The experiment is runnable today; it costs inference, not training.
- **Empirically open.** Whether pretraining-data language-balance or a parallel-fact objective raises $\Delta$-corrected consistency, at fixed accuracy.
- **Theoretically open.** Whether a shared-fact-store model and a multi-store model are behaviourally distinguishable without interpretability access. No identifiability result either way.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by absent ground truth for the confound**.

Every reported cross-lingual disagreement decomposes into at least five terms: (a) genuine divergence of the stored fact; (b) the model does not know the fact in $\ell_2$ at all; (c) the target entity's label in $\ell_2$ is missing from the alias table, so a correct answer scores as $\bot$; (d) the translated template asks a slightly different relation; (e) generic prompt brittleness the model also exhibits within English. Only (a) is the phenomenon. Terms (c) and (d) have no ground truth — you would need per-language human verification of alias coverage and template fidelity, which is exactly the resource that low-resource languages lack, so the measurement error is *largest precisely where the effect is claimed to be largest*.

Secondary obstruction: non-identifiability. "One store, weak interface" and "many stores, partly aligned" predict the same behavioural signature, so the causal question cannot be settled by benchmarking alone.

## 7. Current Research (as of 2026)

- **Correctness-decoupled metrics.** The Groningen group (Bisazza and colleagues) around RankC and its successors remains the reference line of work on measuring rather than improving consistency.
- **Latent-language mechanism.** EPFL (West group) and follow-ups on the English-pivot finding; extensions to whether the pivot is causal for factual recall rather than merely visible in the logit lens *(frontier — verify)*.
- **Cross-lingual knowledge editing.** Bi-ZsRE, MLaKE and successors; the open question is whether editing in the shared latent space rather than in language-specific MLP rows removes the transfer gap *(frontier — verify)*.
- **Multilingual hallucination detection.** Mu-SHROOM-derived span-level detection across ~14 languages; consistency-as-a-detector (disagreement across languages as an uncertainty signal) is an active and underexplored idea.
- **Culturally-indexed facts.** GeoMLAMA-style work (Yin et al., EMNLP 2022) on geo-diverse knowledge is the closest thing to a treatment of the locale-dependence confound, and it is still small.

## 8. Concrete Next Experiment

**Question.** Does cross-lingual factual consistency improve with frontier scale once the measurement confounds are removed?

**Scale.** 3,000 Wikidata triples over 10 relations chosen to be locale-invariant (date of birth, capital of, official currency, chemical symbol, …), in 12 languages spanning 4 scripts and 3 resource tiers. Five human-verified templates per relation per language — verified, not MT'd, at roughly 600 template-language cells, which is a few days of annotator time. Models: one small (1B), one mid (8B), one 70B, one frontier API model. Inference only; total cost is on the order of $10^6$–$10^7$ forward tokens, i.e. hundreds of dollars.

**Controls.**
1. *Within-language arm.* Same 3,000 facts, 5 English paraphrases. Gives $\mathrm{Agr}(\text{en},\text{en})$, the ceiling.
2. *Grounding arm.* For every ungrounded output, a human decides whether it is a correct answer missing from the alias table. Report the corrected score alongside the raw one; the gap between them is the size of confound (c).
3. *Contamination arm.* Restrict to facts created in Wikidata after the model cutoff, or with single-language-only Wikipedia coverage, and report separately.

**Deciding number.** The excess inconsistency
$$\Delta = \mathrm{Agr}(\text{en},\text{en}) - \overline{\mathrm{Agr}(\ell_1,\ell_2)}$$
after grounding correction, plotted against parameter count. If $\Delta$ falls below 5 points at frontier scale, cross-lingual inconsistency is a scale-solved artifact of small models and the problem downgrades to low-resource coverage. If $\Delta$ stays above 15 points at frontier scale while English accuracy exceeds 80%, the multi-store hypothesis survives and pretraining-objective work is warranted. Anything between is the interesting case and demands the interpretability arm.

## 9. Key References

- **[Foundational]** Zhengbao Jiang, Antonios Anastasopoulos, Jun Araki, Haibo Ding, Graham Neubig. *X-FACTR: Multilingual Factual Knowledge Retrieval from Pretrained Language Models.* EMNLP 2020.
- **[Foundational]** Nora Kassner, Philipp Dufter, Hinrich Schütze. *Multilingual LAMA: Investigating Knowledge in Multilingual Pretrained Language Models.* EACL 2021.
- **[Foundational]** Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, Yoav Goldberg. *Measuring and Improving Consistency in Pretrained Language Models.* TACL 2021.
- **[SOTA]** Jirui Qi, Raquel Fernández, Arianna Bisazza. *Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models.* EMNLP 2023. — introduces RankC.
- **[SOTA]** Constanza Fierro, Anders Søgaard. *Factual Consistency of Multilingual Pretrained Language Models.* Findings of ACL 2022.
- **[Mechanism]** Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024.
- **[Editing]** Jiaan Wang et al. *Cross-Lingual Knowledge Editing in Large Language Models.* ACL 2024. — Bi-ZsRE.
- **[Related]** Da Yin, Hritik Bansal, Masoud Monajatipoor, Liunian Harold Li, Kai-Wei Chang. *GeoMLAMA: Geo-Diverse Commonsense Probing on Multilingual Pre-Trained Language Models.* EMNLP 2022.
- **[Related]** David Dale, Elena Voita, Janice Lam, Prangthip Hansanti, Christophe Ropers, Elahe Kalbassi, Cynthia Gao, Loïc Barrault, Marta R. Costa-jussà. *HalOmi: A Manually Annotated Benchmark for Multilingual Hallucination and Omission Detection in Machine Translation.* EMNLP 2023.
- **[Survey]** Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, Pascale Fung. *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys, 2023.

## 10. Worked Example

One relation, `capital-of`, one fact: `(Kazakhstan, capital, Astana)` — renamed Nur-Sultan in 2019, reverted to Astana in 2022.

Ask four languages:

| Language | Raw output | Grounded $\hat{o}$ | Verdict |
|---|---|---|---|
| English | "Astana" | Q1520 | correct |
| Russian | "Нур-Султан" | Q1520 (alias) | stale fact |
| Kazakh | "Астана" | Q1520 | correct |
| Swahili | "Astana, mji mkuu wa Kazakhstan" | $\bot$ | ungrounded |

Naive agreement over the six pairs: the Swahili row fails all three of its pairs on grounding alone, and the Russian row agrees with English only because Wikidata carries "Нур-Султан" as an alias of the same QID. Change one alias table entry and the score moves from $4/6$ to $1/6$. That is a 50-point swing in the headline metric produced by a lexicon, not by the model.

Now the temporal layer: Russian is not inconsistent, it is *out of date*, and English is right for a reason that has nothing to do with representation sharing — post-2022 English text about the reversion is abundant, post-2022 Russian text is more likely to be pre-reversion. So the disagreement is a pretraining-corpus recency effect wearing a cross-lingual costume.

Scale this to 3,000 facts and the two effects — alias coverage and per-language corpus recency — do not average out; both are monotone in resource tier, so they bias the score in the same direction as the hypothesis under test. This is the obstruction in one instance: without the grounding arm and the contamination arm from §8, a published cross-lingual consistency number cannot distinguish "the model stores different facts per language" from "the evaluation harness knows fewer names in Swahili."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*