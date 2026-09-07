---
id: 33-uncertainty-calibration/ambiguity-versus-ignorance-refusals
title: "Distinguishing Ambiguity From Ignorance in Model Refusals"
topic: 33-uncertainty-calibration
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distinguishing Ambiguity From Ignorance in Model Refusals

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/ambiguity-versus-ignorance-refusals` · **Status:** methodologically-blocked

## 1. Problem Statement

A model declines to answer. Two distinct causes produce the same surface behaviour:

- **Ambiguity (aleatoric).** The question admits several valid readings. The correct action is to *ask a clarifying question* or enumerate readings. More model capacity does not help; more context does.
- **Ignorance (epistemic).** The question has one intended reading and a determinate answer the model does not have. The correct action is to *abstain, retrieve, or defer*. More capacity or retrieval helps; clarification does not.

**Input:** a query $x$ and a model $p_\theta$ that emits a refusal. **Output:** a label in $\{\textsf{ambiguous}, \textsf{ignorant}, \textsf{unanswerable}, \textsf{policy}\}$, or a scalar $s(x) \in [0,1]$ ranking ambiguity against ignorance. **Decision predicate:** the label must select the action that maximises expected task utility — clarify vs. abstain — under a cost model where a wrong clarification wastes a turn and a wrong abstention loses an answerable query.

Three variants, different difficulty:

- **Measurement.** Is there a well-defined target quantity at all? Ambiguity is defined relative to a population of askers, which is unobserved. This is where the problem is blocked.
- **Method.** Given a target, can a score be computed from $p_\theta$ cheaply? Partially solved: clarification ensembling and semantic clustering give usable signals.
- **Theory.** Is the decomposition identifiable from the predictive distribution alone? Answer: no, without further assumptions (§6).

## 2. Formal Setting

Let $x$ be a query, $\mathcal{Y}$ the answer space, and $C$ a latent **intent** variable: the disambiguating context the asker holds but did not type. The asker population induces $p^\star(c \mid x)$. Ground truth is a conditional $p^\star(y \mid x, c)$.

Total predictive uncertainty of the model, measured over sampled generations, decomposes as

$$H[Y \mid x] \;=\; \underbrace{\mathbb{E}_{c \sim q(c\mid x)}\,H[Y \mid x, c]}_{\text{residual: ignorance}} \;+\; \underbrace{I(Y; C \mid x)}_{\text{ambiguity}}.$$

**How each quantity is measured.**

- $H[Y \mid x]$: **semantic entropy**. Sample $K$ generations at temperature $T$, cluster by bidirectional entailment with an NLI model, and take $-\sum_j \hat{p}_j \log \hat{p}_j$ over cluster masses $\hat{p}_j = n_j/K$ (Kuhn et al., ICLR 2023). Typical $K \in [5, 20]$, $T = 1.0$.
- $I(Y; C \mid x)$: **input-clarification ensembling** (Hou et al., ICML 2024). Generate $M$ candidate clarifications $c_1..c_M$ with a separate LLM, sample answers under each, and take the Jensen–Shannon-style gap between the pooled and per-clarification semantic entropies.
- Decision utility: with clarify cost $\lambda$ and answer reward $1$, choose $\textsf{clarify}$ iff $\mathbb{E}[\text{gain from } C] > \lambda$.

**Assumptions, and which are violated.**

1. *$q(c\mid x) \approx p^\star(c\mid x)$* — the generated clarification set matches real asker intents. **Violated:** clarifications are produced by an LLM with the same knowledge gaps; AmbigQA (Min et al., EMNLP 2020) shows real intent distributions are long-tailed and often not enumerable from the question text alone.
2. *Entailment clustering is a valid equivalence relation.* **Violated:** NLI models are not transitive; cluster counts shift with the NLI backbone.
3. *A single gold answer exists per $(x,c)$.* **Violated** for questions whose ground truth is time- or place-indexed (SituatedQA, Zhang & Choi, EMNLP 2021).
4. *Refusal text is causally downstream of internal uncertainty.* **Violated:** RLHF-tuned refusal is partly a style prior, fired by surface features of the prompt independent of answer-distribution entropy.

## 3. State of the Art

**Established.**

- Semantic entropy beats token-level entropy and $p(\text{true})$ baselines at detecting confabulation. Farquhar et al. (*Nature*, 2024) report AUROC around $0.79$ averaged over six QA/biography datasets versus $\approx 0.69$ for naive entropy — measured on models up to LLaMA-2 70B and GPT-4-class systems. This separates *high total uncertainty* from *low*; it does not separate the two causes.
- Selective prediction over ambiguous questions is improved by sampling-based repetition scores rather than logit confidence (Cole et al., "Selectively Answering Ambiguous Questions", EMNLP 2023) — measured on AmbigQA and their ambiguity-annotated NQ subsets.
- Abstention can be trained in: R-Tuning (Zhang et al., NAACL 2024) fine-tunes on a knowledge-boundary split and raises abstention accuracy on held-out unanswerable sets.

**Claimed but unablated.**

- That input-clarification ensembling *decomposes* uncertainty into aleatoric and epistemic parts (Hou et al., ICML 2024). The decomposition is validated by downstream selective-prediction gains, not against annotated ambiguity/ignorance labels; no ablation isolates whether the clarifier's own ignorance contaminates the aleatoric term.
- That models "know what they don't know". Yin et al. (ACL Findings, 2023) show partial self-knowledge on **SelfAware**, but the benchmark's unanswerable class mixes future events, opinions, and genuinely ambiguous items — a benchmark number, not a decomposition.

**Benchmark-only results.** CLAMBER (ACL 2024) reports LLM performance at identifying and clarifying ambiguous requests; scores are dataset-specific and no cross-dataset transfer has been demonstrated.

## 4. What Is Known

- **Ambiguity is the common case, not the corner case.** AmbigQA: over half of Natural Questions open-domain questions are ambiguous under annotation by two independent annotators with search access ($14{,}042$ questions annotated).
- **Unanswerability is learnable but shallow.** SQuAD 2.0 (Rajpurkar et al., ACL 2018) added $53{,}775$ unanswerable questions; models closed most of the human gap within two years, then transferred poorly to open-domain unanswerability.
- **Calibration exists at scale for multiple choice.** Kadavath et al. (2021) show a 52B model is well calibrated on multiple-choice formats but degrades sharply on free-form generation — the format where the ambiguity/ignorance distinction actually bites.
- **The aleatoric/epistemic split is not model-free.** Hüllermeier & Waegeman (*Machine Learning*, 2021) show the split depends on the chosen hypothesis space; the same predictive distribution decomposes differently under different priors.
- **Refusal has a low-dimensional internal correlate.** Arditi et al. (NeurIPS 2024) find refusal in chat models is mediated by a single direction in residual stream activations across 13 open models to 72B — evidence that refusal firing is partly decoupled from answer-distribution uncertainty.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed operationalisation of "this query is ambiguous" independent of an annotator panel. Annotator disagreement about *whether* a question is ambiguous is itself unmeasured at scale; no public dataset reports inter-annotator agreement on the ambiguity/ignorance *cause* label for refusals.
- **Theoretically open.** Whether $I(Y;C\mid x)$ is identifiable from $p_\theta$ plus a finite clarification generator, without access to $p^\star(c\mid x)$. No impossibility proof and no identifiability result exists for the LLM setting.
- **Empirically open.** Whether an internal probe (linear read-out on residual activations at the refusal token) separates the two causes better than any behavioural sampling score. Runnable today on 7B–70B open models; not run with cause-labelled data.
- **Empirically open.** Whether the distinction transfers across languages and across retrieval-augmented vs. closed-book settings.

## 6. Why It Is Hard

**Non-identifiability, compounded by absent ground truth.**

Two generative stories produce the same sampled answer distribution. Story A: the asker population splits over three readings, and the model knows each reading's answer. Story B: one reading, and the model's posterior over the answer has three spurious modes. Marginally, both give three semantic clusters with similar masses. The distinguishing variable $C$ is latent and never appears in the observation. Recovering it requires a prior over asker intent — which is exactly the object no dataset supplies.

The standard fix — generate clarifications and condition on them — replaces the unknown $p^\star(c\mid x)$ with a model-generated $q(c\mid x)$ produced by a system with correlated ignorance. When the model does not know a domain, it also cannot enumerate that domain's disambiguations, so ignorance is systematically re-scored as low ambiguity. The error is not noise; it is aligned with the quantity being estimated.

Compounding: refusal strings are near-identical across causes ("I don't have enough information to answer that"), so any evaluation that scores refusal text rather than cause is measuring style, not epistemics.

## 7. Current Research (as of 2026)

- **Clarification-conditioned decomposition.** Follow-ons to Hou et al. (UCSB / MIT-IBM) extending input-clarification ensembling to multi-turn and agentic settings *(frontier — verify)*.
- **Semantic-entropy probes.** OATML (Oxford, Gal group) after the *Nature* 2024 result: cheap linear probes reproducing semantic entropy from hidden states in one forward pass. Whether such probes carry cause information is untested.
- **Interpretability of refusal.** Work following Arditi et al. on refusal directions; the open question is whether the "unsure" direction is distinct from the "underspecified" direction.
- **Abstention surveys and taxonomies.** Wen et al., *The Art of Refusal: A Survey of Abstention in Large Language Models* (2024), gives the cleanest cause taxonomy in print; it is a survey, not a measurement.
- **Interactive benchmarks.** CLAMBER-style and clarification-question suites are being extended toward agent tool-use, where a wrong clarify/abstain choice has measurable downstream cost *(frontier — verify)*.

## 8. Concrete Next Experiment

**Cause-labelled refusal corpus with an intent-panel ground truth.**

- **Scale.** $2{,}000$ queries that elicit a refusal from one 70B-class open model, stratified: $1{,}000$ drawn from AmbigQA-style ambiguous items, $1{,}000$ from long-tail factual items whose answers are verifiable in a held-out corpus. Cause label obtained not from a single annotator but from an **intent panel**: 10 independent annotators each write the answer they would have wanted *before* seeing the model output. Ambiguity score $=$ normalised entropy of the panel's intent clusters; ignorance score $=$ fraction of panel intents whose gold answer exists in the corpus but not in any model sample. Cost: roughly 200 annotator-hours.
- **Arms.** (i) Semantic entropy, $K=20$. (ii) Input-clarification ensembling, $M=5$. (iii) Linear probe on residual activations at the refusal token, layer swept. **Control arm:** a prompt-surface-features-only classifier (question length, wh-word, presence of a named entity, presence of a superlative). This control is the one that matters — much of the reported signal in existing abstention benchmarks is recoverable from surface form.
- **Deciding number.** AUROC for ambiguity-vs-ignorance on the held-out half, against the panel label. **Decision rule:** any method that does not beat the surface-feature control by $\ge 0.10$ AUROC is not measuring epistemics. Secondary: report panel inter-annotator agreement (Krippendorff's $\alpha$) on the cause label — if $\alpha < 0.6$, the problem stays methodologically blocked and no method number is interpretable.

## 9. Key References

- **[Foundational]** Min, Michael, Hajishirzi, Zettlemoyer. *AmbigQA: Answering Ambiguous Open-domain Questions.* EMNLP, 2020. — arXiv:2004.10645
- **[Foundational]** Rajpurkar, Jia, Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL, 2018. — arXiv:1806.03822
- **[Foundational]** Hüllermeier, Waegeman. *Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods.* Machine Learning 110(3), 2021. — arXiv:1910.09457
- **[SOTA]** Farquhar, Kossen, Kuhn, Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 2024.
- **[SOTA]** Kuhn, Gal, Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023. — arXiv:2302.09664
- **[SOTA]** Hou, Liu, Qian, Andreas, Chang, Zhang. *Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling.* ICML, 2024. — arXiv:2311.08718
- **[SOTA]** Cole, Zhang, Gillick, Eisenschlos, Dhingra, Eisenstein. *Selectively Answering Ambiguous Questions.* EMNLP, 2023. — arXiv:2305.14613
- **[Related]** Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic technical report, 2022. — arXiv:2207.05221
- **[Related]** Yin, Sun, Guo, Wu, Qiu, Huang. *Do Large Language Models Know What They Don't Know?* Findings of ACL, 2023. — arXiv:2305.18153
- **[Related]** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS, 2024. — arXiv:2406.11717
- **[Survey]** Wen, Yao, Feng, Evans, Huang, Xiong, Tsvetkov. *The Art of Refusal: A Survey of Abstention in Large Language Models.* 2024. — arXiv:2407.18418
- **[Survey]** Baan, Daheim, Ilia, Ulmer, Li, Fernández, Plank, Sennrich, Zerva, Aziz. *Uncertainty in Natural Language Generation: From Theory to Applications.* 2023. — arXiv:2307.15703

## 10. Worked Example

Two queries, $K=10$ samples each, semantic clustering by bidirectional entailment.

**Query A (ambiguous).** *"What is the population of Springfield?"* There are 30+ US Springfields. Sampled clusters: Illinois $\times 4$, Missouri $\times 3$, Massachusetts $\times 2$, Oregon $\times 1$.

$$H_A = -\left(0.4\ln 0.4 + 0.3\ln 0.3 + 0.2\ln 0.2 + 0.1\ln 0.1\right) = 1.28\ \text{nats}.$$

**Query B (ignorance).** *"Who chaired the 1978 committee that drafted the municipal drainage code for Springfield, Illinois?"* Determinate answer; model has no coverage. Sampled clusters: name-1 $\times 5$, name-2 $\times 5$ — two confidently hallucinated modes.

$$H_B = -\left(0.5\ln 0.5 + 0.5\ln 0.5\right) = 0.69\ \text{nats}.$$

**The obstruction, made visible.** Ranking by semantic entropy gives $H_A > H_B$, so a threshold at $1.0$ nats routes A to "uncertain" and B to "answer confidently" — B gets answered with a fabricated name. Ranking by *any monotone function of total uncertainty* cannot fix this, because the ordering is wrong, not the calibration: an ambiguous question with many readings and a confidently wrong question with few hallucinated modes sit on opposite sides of every threshold, while the required actions (clarify A, abstain on B) are also opposite.

Now the clarification-ensembling repair. For A, the clarifier proposes {Illinois, Missouri, Massachusetts}; per-clarification entropies drop to about $0.1$ nats, so estimated ambiguity $\approx 1.28 - 0.1 = 1.18$ nats. Correct. For B, the clarifier — same knowledge gap — proposes only cosmetic variants ("the drainage code committee", "the 1978 committee"); per-clarification entropy stays at $0.69$, so estimated ambiguity $\approx 0.00$. Also correct, and this is the method working as designed.

But swap B for *"Who chaired the 1978 drainage committee in Springfield?"* — genuinely ambiguous over cities **and** outside the model's knowledge for all of them. The clarifier proposes three cities; per-clarification entropy stays near $0.69$ because the model hallucinates two names under each. Estimated ambiguity $\approx 0$, label $=$ ignorance, action $=$ abstain. The correct action was to clarify first, then abstain on the specific city. The estimator failed exactly where the two causes co-occur — the case that dominates real long-tail queries, and the case no current benchmark labels.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*