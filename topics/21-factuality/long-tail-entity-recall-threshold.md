---
id: 21-factuality/long-tail-entity-recall-threshold
title: "Entity Popularity Threshold for Reliable Recall"
topic: 21-factuality
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Entity Popularity Threshold for Reliable Recall

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/long-tail-entity-recall-threshold` · **Status:** empirically-open

## 1. Problem Statement

A language model answers factual questions about famous entities reliably and about obscure ones unreliably. The transition is empirically smooth in aggregate. The question is whether a **usable threshold** exists: a popularity or pretraining-frequency level $\tau$ above which recall accuracy for an entity is at least $\alpha$ (say $0.95$) with bounded per-entity variance, and how $\tau$ moves with model size $N$, token budget $D$, and occurrence count $k$.

Three variants, different difficulty:

- **Measurement.** Given a model, a corpus, and an entity set, estimate $\tau_\alpha$ and its confidence interval. Blocked mainly on being able to count entity occurrences in the actual pretraining corpus.
- **Method.** Build a predictor $\hat{p}(e)$ of per-entity recall reliability that works *before* the question is asked, so a system can route to retrieval or abstain. Runnable today; the open part is whether any predictor beats the popularity prior by a useful margin on held-out entities.
- **Theory.** Prove that accuracy is a monotone function of occurrence count under a stated memorization model, and derive $\tau_\alpha(N, D)$. Currently open; the closest results bound aggregate hallucination rate, not per-entity recall.

Solving it means: a calibrated, model-specific function from a cheaply computable entity statistic to a recall probability, validated out-of-distribution across relation types and languages.

## 2. Formal Setting

Let $\mathcal{E}$ be an entity set (e.g. Wikidata items) and $\mathcal{R}$ a relation set. A fact is a triple $f = (e, r, o)$. A query template $q_r$ maps $(e,r)$ to a natural-language question with gold answer set $O(e,r)$ (all accepted surface forms, including aliases).

**Accuracy.** For model $M$ with decoding $\pi$,
$$a_M(e,r) = \Pr_{y \sim \pi(\cdot \mid q_r(e))}\big[\mathrm{match}(y, O(e,r))\big],$$
measured as the fraction of $n$ samples ($n \ge 8$, temperature $0$ plus $n-1$ at $T=0.7$ to expose variance) whose normalized form contains a gold alias. `match` is substring-after-normalization — the standard PopQA/TriviaQA criterion, and a known source of both false positives (answer appears inside a hedge) and false negatives (correct but unlisted alias).

**Popularity proxies.** Two distinct quantities, routinely conflated:
$$\pi(e) = \text{monthly Wikipedia pageviews}, \qquad c(e) = \big|\{d \in \mathcal{D} : e \text{ mentioned in } d\}\big|,$$
and the salient variant $c(e,o)$ = documents containing *both* the subject and the object entity, which is the quantity Kandpal et al. show actually predicts accuracy. $c$ requires corpus access and an entity linker; $\pi$ does not. Correlation between $\log \pi$ and $\log c$ is strong but far from 1 and is relation-dependent.

**Threshold.** Fit a monotone (isotonic) regression $\hat{a}(\cdot)$ of $a_M$ on $\log_{10} c(e,o)$ and define
$$\tau_\alpha(M) = \inf\{t : \hat{a}(t') \ge \alpha \ \ \forall t' \ge t\}.$$
The threshold is *useful* only if the conditional dispersion above it is small: require $\Pr[a_M(e,r) < \alpha' \mid c \ge \tau_\alpha] \le \delta$ for a per-entity floor $\alpha' $, not merely a mean above $\alpha$. A mean-only threshold is compatible with 20% of above-threshold entities being answered at chance.

**Assumptions, and which are violated.**

1. *Monotonicity of $a_M$ in $c$.* Violated locally: high-frequency entities with many near-duplicate distractors (common names, franchise reboots) show accuracy dips.
2. *$\pi \propto c$.* Violated for non-English entities, recently popular entities, and entities whose fame postdates the pretraining cutoff.
3. *Question difficulty independent of popularity.* Violated by construction — obscure entities get asked about via templated relations, famous ones via naturally occurring questions.
4. *Exact string counting identifies mentions.* Violated by coreference, aliases, and translation; string counts undercount by a relation-dependent factor.
5. *No contamination.* Violated for every public benchmark of appreciable age.

## 3. State of the Art

**Established (with ablations).**

- Kandpal, Deng, Roberts, Wallace, Raffel, *Large Language Models Struggle to Learn Long-Tail Knowledge* (ICML 2023, arXiv:2211.08411). Entity-links the pretraining corpora of GPT-Neo/BLOOM/GPT-3 models and shows QA accuracy on TriviaQA and Natural Questions is strongly log-linear in the number of pretraining documents containing the salient question–answer entity pair. The causal ablation — removing the relevant documents and retraining — is the strongest evidence in the literature that the relation is not merely correlational.
- Mallen, Asai, Zhong, Das, Khashabi, Hajishirzi, *When Not to Trust Language Models* (ACL 2023, arXiv:2212.10511). Introduces PopQA (14k Wikidata questions with pageview-based popularity) and shows a *crossing point*: below a popularity level, retrieval augmentation helps; above it, retrieval can hurt. Adaptive retrieval keyed on popularity beats always-retrieve on both accuracy and cost. This is the closest existing thing to a threshold result.
- Allen-Zhu, Li, *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws* (arXiv:2404.05405, 2024). In synthetic biographies with controlled exposure counts, models store about 2 bits of knowledge per parameter at 1000 exposures per fact, dropping to roughly 1 bit/param at 100 exposures. Direct evidence that exposure count, not just parameter count, sets what is recallable.

**Claimed but unablated / benchmark-only.**

- Head-to-Tail (Sun et al., NAACL 2024, arXiv:2308.10168) reports large head/torso/tail accuracy gaps across 16 LLMs. The gap is a benchmark number; the tail bucket is defined by a heuristic popularity split, not by measured pretraining counts, so it does not isolate frequency from difficulty.
- Many RAG papers cite "long-tail entities" as motivation and report aggregate gains without stratifying by any frequency statistic. Those numbers say nothing about $\tau$.

**Theory SOTA.** Kalai and Vempala, *Calibrated Language Models Must Hallucinate* (STOC 2024, arXiv:2311.14648): a calibrated model's hallucination rate on arbitrary facts is lower-bounded, roughly, by the Good–Turing "monofact" rate — the fraction of facts appearing exactly once in training — minus calibration error. This is an aggregate bound over a fact distribution, not a per-entity threshold, and it constrains generation rather than recall.

## 4. What Is Known

- **Log-linear scaling.** Kandpal et al.: accuracy rises approximately linearly in $\log_{10} c(e,o)$ across four orders of magnitude, measured at 125M–176B parameters (GPT-Neo family, BLOOM, GPT-3 via API) on TriviaQA and NQ. The slope shrinks with model scale but does not flatten.
- **Extrapolated parameter cost.** The same paper's fit implies a model would need on the order of $10^{18}$ parameters to reach high accuracy on the rarest bucket by scale alone. This is an extrapolation off the fitted line, not a measurement, and should be quoted as such.
- **Popularity crossing point.** Mallen et al.: on PopQA, the popularity level at which retrieval stops helping varies by roughly two orders of magnitude across the 16 relation types. There is no single threshold even within one benchmark.
- **Memorization scales log-linearly** in model size, data duplication, and prompt context length (Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang, *Quantifying Memorization Across Neural Language Models*, ICLR 2023, arXiv:2202.07646), measured on the GPT-Neo suite up to 6B with exact pretraining-corpus access.
- **Duplication drives it.** Lee et al., *Deduplicating Training Data Makes Language Models Better* (ACL 2022, arXiv:2107.06499), and Biderman et al., *Pythia* (ICML 2023, arXiv:2304.01373) with checkpointed, order-known training data, make repeated-exposure effects directly measurable.
- **Corpus counting is now feasible** for open corpora: Elazar et al., *What's In My Big Data?* (ICLR 2024, arXiv:2310.20707) provides count infrastructure over C4, The Pile, RedPajama and others.

## 5. What Is Not Known

- **Empirically open.** Whether a per-entity (not mean) threshold with bounded tail risk exists for any frontier-scale model. Nobody has run the stratified measurement with $n \ge 8$ samples per entity, real corpus counts, and per-relation breakdown at $\ge$70B scale on a decontaminated entity set. The experiment is runnable on any fully open-data model (Pythia, OLMo 2, DCLM) today.
- **Empirically open.** How $\tau_\alpha$ moves under post-training. Instruction tuning and RLHF shift abstention behavior; whether they move the recall threshold or only the reporting threshold is unmeasured.
- **Theoretically open.** No derivation of $\tau_\alpha(N,D,k)$ from a memorization model. The Kalai–Vempala bound is aggregate; capacity results (2 bits/param) give a budget but not an allocation rule across entities.
- **Methodologically blocked.** "Popularity" has no agreed operationalization. Pageviews, inlink count, corpus mention count, and salient-pair count give different orderings and different thresholds on the same entity set. Until one is fixed, cross-paper threshold numbers are not comparable.

## 6. Why It Is Hard

**Confounded measurement, in a specific way.** Popularity is entangled with question difficulty, answer-set size, and alias density. Obscure entities tend to have fewer aliases and more ambiguous names; famous entities attract more distractors. A measured accuracy drop at low $c$ therefore mixes at least four effects: fewer training exposures, harder templates, worse alias coverage in the gold set, and higher name ambiguity. Only the first is the phenomenon of interest, and none of the widely used benchmarks separates them.

**Absent ground truth on counts for the models people care about.** The frontier models with the interesting behavior have undisclosed pretraining corpora, so $c(e,o)$ is unmeasurable and $\pi(e)$ substitutes for it. The models with measurable corpora are small enough that the tail is uniformly bad, compressing the dynamic range the threshold is supposed to live in.

**Non-identifiability of the failure mode.** A wrong answer can be a storage failure (the fact was never encoded), a retrieval failure (encoded but not elicited by this prompt), or a scoring failure (correct but unmatched). Prompt-format sensitivity alone moves factual accuracy by tens of points, so a single-prompt measurement of $a_M$ estimates a lower bound of unknown tightness.

## 7. Current Research (as of 2026)

- **Adaptive retrieval and abstention routing.** Direct descendants of Mallen et al.: Self-RAG (Asai, Wu, Wang, Sil, Hajishirzi, ICLR 2024, arXiv:2310.11511) and subsequent confidence-gated retrieval work at AI2 and UW.
- **Open-data pretraining suites** (OLMo 2 at AI2; DCLM) make count-conditioned analysis tractable at 7B–13B. Groups are beginning to publish frequency-stratified factuality curves on these. *(frontier — verify)*
- **Training-data attribution at scale** — influence-function and datamodel approaches applied to specific factual outputs, aiming to identify the documents responsible for a recalled fact rather than counting mentions. *(frontier — verify)*
- **Hallucination-as-incentive framing.** Kalai, Nachum, Vempala, Zhang, *Why Language Models Hallucinate* (arXiv:2509.04664, 2025), arguing that binary scoring rewards guessing over abstention — relevant because it predicts post-training moves the *reporting* threshold independently of the recall threshold.

## 8. Concrete Next Experiment

**Scale.** OLMo 2 7B and 13B (or Pythia 6.9B/12B), which have public, indexable pretraining corpora. Sample 6,000 Wikidata triples across 12 relations, stratified into 6 log-spaced buckets of measured salient-pair count $c(e,o) \in [10^0, 10^5]$, 1,000 per bucket, all with entity pages created before the pretraining cutoff.

**Protocol.** For each triple, generate 5 paraphrased question templates; sample $n=8$ per template ($T=0.7$). Score with alias-expanded matching plus a human-audited 300-item subsample to estimate scoring error. Report per-entity $a_M(e,r)$, not bucket means.

**Control arm.** A matched set where $c(e,o)$ is held fixed but Wikipedia pageview popularity $\pi$ varies by $\ge 2$ orders of magnitude. If accuracy tracks $\pi$ at fixed $c$, the phenomenon is not frequency-driven and the whole threshold framing is measuring difficulty, not memorization.

**The deciding number.** $\hat{\delta} = \Pr\big[a_M(e,r) < 0.5 \mid c(e,o) \ge \tau_{0.95}\big]$ — the fraction of above-threshold entities that are still unreliable. If $\hat{\delta} \le 0.05$ with a 95% CI width under 0.02, a usable threshold exists and popularity-gated routing is justified. If $\hat{\delta} \ge 0.20$, the threshold is a property of bucket means only, and per-entity routing must use a different signal.

**Cost.** About $6{,}000 \times 5 \times 8 = 240{,}000$ generations per model, two models — under 100 GPU-hours on 8×A100.

## 9. Key References

- **[Foundational]** Kandpal, Deng, Roberts, Wallace, Raffel. *Large Language Models Struggle to Learn Long-Tail Knowledge.* ICML, 2023. — arXiv:2211.08411
- **[Foundational]** Mallen, Asai, Zhong, Das, Khashabi, Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL, 2023. — arXiv:2212.10511
- **[SOTA]** Allen-Zhu, Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[SOTA]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Theory]** Kalai, Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Theory]** Kalai, Nachum, Vempala, Zhang. *Why Language Models Hallucinate.* 2025. — arXiv:2509.04664
- **[Benchmark]** Sun, Xu, Wang, et al. *Head-to-Tail: How Knowledgeable are Large Language Models? A.K.A. Will LLMs Replace Knowledge Graphs?* NAACL, 2024. — arXiv:2308.10168
- **[Infrastructure]** Biderman, Schoelkopf, Anthony, et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML, 2023. — arXiv:2304.01373
- **[Infrastructure]** Elazar, Bhagia, Magnusson, et al. *What's In My Big Data?* ICLR, 2024. — arXiv:2310.20707
- **[Related]** Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL, 2022. — arXiv:2107.06499
- **[Related]** Asai, Wu, Wang, Sil, Hajishirzi. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR, 2024. — arXiv:2310.11511
- **[Related]** Min, Krishna, Lyu, et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251

## 10. Worked Example

Take relation `P50` (author of a work) and two entities with the *same* measured salient-pair count $c(e,o) = 40$ in the pretraining corpus.

- $e_1$ = a mid-list novelist, pageviews $\pi \approx 1{,}200$/month, one canonical name, no namesakes.
- $e_2$ = a novelist sharing a surname with a footballer, pageviews $\pi \approx 30{,}000$/month, the pageviews driven mostly by disambiguation traffic.

Sample $n=8$ answers each across 5 templates ($40$ generations per entity). Illustrative outcome consistent with reported dispersion: $a_M(e_1) = 0.88$, $a_M(e_2) = 0.35$, with $e_2$'s errors being the footballer's biography facts. Both entities sit in the same frequency bucket. A bucket-mean threshold fit would place both above $\tau_{0.95}$ if the bucket mean is $0.96$ — which it can be, if the other 998 entities in the bucket average $0.96$.

Now the arithmetic that makes the obstruction visible. Suppose bucket mean accuracy is $0.96$ and the per-entity distribution is bimodal: 92% of entities at $0.99$, 8% at $0.60$. Mean $= 0.92 \times 0.99 + 0.08 \times 0.60 = 0.96$. The threshold is "satisfied" at $\alpha = 0.95$, yet $\hat\delta$ — the fraction below $0.5$ — is only computable from per-entity samples, and a system that trusts the model above $\tau$ still hallucinates on roughly 1 entity in 12.

The obstruction is not that the curve is noisy. It is that **the published quantity (bucket mean accuracy versus $\log c$) is the wrong functional of the per-entity distribution** for the decision the threshold is meant to support. Every threshold number in the literature is a level set of a conditional mean; every deployment decision needs a conditional quantile. Recovering the quantile requires $n \gtrsim 8$ samples per entity across paraphrases — roughly $40\times$ the generation budget of the standard single-sample protocol, which is why it has not been run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*