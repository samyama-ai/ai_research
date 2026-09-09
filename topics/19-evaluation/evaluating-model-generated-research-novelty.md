---
id: 19-evaluation/evaluating-model-generated-research-novelty
title: "Evaluating Scientific Novelty of Model-Generated Research"
topic: 19-evaluation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Evaluating Scientific Novelty of Model-Generated Research

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/evaluating-model-generated-research-novelty` · **Status:** methodologically-blocked

## 1. Problem Statement

**Input.** A research artifact $a$ produced by a model — an idea sketch, a hypothesis, a full paper with code and results — together with a declared knowledge cutoff and, ideally, the corpus the model was trained on.

**Output.** A scalar or ordinal judgment of *scientific novelty*: whether $a$ contributes something not already in the literature, and how much.

**Decision predicate.** Given two artifacts $a_1, a_2$, decide $\mathrm{Nov}(a_1) > \mathrm{Nov}(a_2)$ with a reliability that exceeds human reviewer noise on the same pair.

Three variants, of very different difficulty:

- **Measurement.** Define $\mathrm{Nov}$ so that repeated independent application yields the same answer. Currently unresolved — this is why the page is marked methodologically blocked.
- **Method.** Build an automatic scorer (retrieval, LLM judge, citation-graph statistic) that tracks whatever expert judgment exists. Runnable today; every existing scorer is validated against a noisy target.
- **Theory.** Characterize whether novelty relative to a training corpus is identifiable at all from the artifact alone, without access to that corpus. Open.

**What counts as solving it.** An instrument whose test–retest and inter-rater reliability on model-generated artifacts exceeds $0.7$, whose scores are invariant to paraphrase and to author identity (human vs. model), and which is *not* satisfiable by retrieving a paraphrase of a paper in the training set.

## 2. Formal Setting

Let $K_t$ be the knowledge state of a field at time $t$: a set of published claims. Let $C_\theta \subseteq K_{t_0}$ be the pretraining corpus of model $\theta$ with cutoff $t_0$. An artifact is $a$; let $\phi(a)$ be its claim set.

Ideal novelty is set-theoretic:

$$\mathrm{Nov}(a \mid K_t) = \big|\{c \in \phi(a) : c \notin \mathrm{Cl}(K_t)\}\big|$$

where $\mathrm{Cl}$ is deductive-plus-trivial-variation closure. **Neither $\phi$ nor $\mathrm{Cl}$ is computable**, so every practical instrument is a proxy. The four in use:

**(a) Expert Likert.** $R$ reviewers each give $s_{r,a} \in \{1,\dots,10\}$. The measured quantity is $\bar{s}_a$; its reliability is
$$\mathrm{ICC} = \frac{\sigma^2_{\text{idea}}}{\sigma^2_{\text{idea}} + \sigma^2_{\text{reviewer}} + \sigma^2_{\varepsilon}}.$$
Measured, not assumed: on NLP idea review, $\sigma^2_{\text{reviewer}} + \sigma^2_\varepsilon$ dominates.

**(b) Retrieval distance.** With embedding $e(\cdot)$ and corpus $D$, $\mathrm{Nov}_{\text{ret}}(a) = 1 - \max_{d \in D} \cos(e(a), e(d))$. Measured by running a fixed retriever over a fixed snapshot; sensitive to both.

**(c) Combinatorial atypicality** (Uzzi et al.). For each reference pair $(j,k)$ in $a$, compute observed co-citation frequency against a degree-preserving rewired null; novelty is the $10$th-percentile $z$-score of pair frequencies.

**(d) Disruption / CD index** (Funk & Owen-Smith). For focal work $f$ with successors $S$,
$$\mathrm{CD}_5(f) = \frac{1}{|S|}\sum_{i \in S} \frac{-2 n^f_i n^b_i + n^f_i}{n^f_i + n^b_i + n^r_i},$$
requiring five years of forward citations — unavailable for an artifact generated today.

**Assumptions, with the violated ones flagged:**

1. *$C_\theta$ is known.* **Violated** for all frontier models; contamination cannot be ruled out, only bounded by membership-inference proxies.
2. *Reviewers score novelty independently of quality, writing, and perceived authorship.* **Violated** — halo effects and anti-AI/pro-AI priors both appear once provenance is guessable.
3. *The retrieval corpus $D$ covers $K_t$.* **Violated**: preprints, negative results, and non-English work are missing, so $\mathrm{Nov}_{\text{ret}}$ over-credits.
4. *Novelty is one-dimensional.* **Violated**: problem-novelty, method-novelty, and result-novelty dissociate; a single Likert item silently averages them.

## 3. State of the Art

**Empirical SOTA — established.** Si, Yang & Hashimoto (ICLR 2025) ran the only large blinded head-to-head: 49 ideas per condition, 79 expert reviewers, 298 reviews, standardized topic and template. LLM-generated ideas scored higher on novelty than expert-written ones ($5.64$ vs. $4.84$ on a 1–10 scale, $p<0.05$). The *design* is established; the *conclusion* is scoped to idea text, not executed research.

**Established caveat from the same line of work.** The follow-up ideation–execution study (Si, Yang & Hashimoto, 2025) had researchers execute both idea sets over weeks; the novelty advantage shrank and overall scores converged or reversed after execution. This is the strongest evidence that pre-execution novelty scores do not predict post-execution value.

**Systems SOTA.** *The AI Scientist* (Lu et al., 2024) and *AI Scientist-v2* (Yamada et al., 2025) generate end-to-end papers; v2 reports one workshop submission passing peer review. That is a single accept, not a novelty measurement, and the review pool was a workshop.

**Claimed but unablated.** Novelty-optimizing generators — SciMON (Wang et al., ACL 2024), ResearchAgent (Baek et al., NAACL 2025) — report gains on LLM-judge or retrieval-distance novelty scores. No ablation shows those gains survive a blinded expert panel, and both optimize against the same family of scorer they are evaluated by. Where numbers exist, they exist only as benchmark numbers.

**LLM-as-reviewer.** Liang et al. (NEJM AI, 2024) found GPT-4 feedback overlapped human feedback at rates comparable to human–human overlap on paper-level critique — but on *weaknesses*, not on novelty specifically.

## 4. What Is Known

- **Human novelty judgment is noisy at a measured rate.** NeurIPS 2014: 10% of submissions were reviewed by two independent committees; ~57% of papers accepted by one were rejected by the other at a ~22.5% accept rate (Cortes & Lawrence, 2021). NeurIPS 2021 replication (Beygelzimer et al., 2023) at ~10× the submission volume found roughly half of accepted papers would have been rejected by the second committee. Scale: thousands of submissions, both years.
- **Generator diversity collapses.** Si et al. found that generating ~4,000 seed ideas on one topic yielded only a few hundred non-duplicates — duplication rises sharply with sampling. Scale: one LLM, seven NLP topics.
- **Reviewers cannot reliably detect provenance,** which is what makes the blinded design work — but ratings shift when they think they can.
- **Bibliometric novelty measures disagree with each other and with expert judgment.** Atypicality (Uzzi et al., *Science*, 2013; 17.9M papers) and the CD index (Park, Leahey & Funk, *Nature*, 2023; reported declines of 91.9% for papers and 78.7% for patents, 1945–2010) are computed on overlapping corpora yet rank differently; the CD-index decline is contested as partly an artifact of reference-list growth and database coverage.
- **Novelty is penalized by reviewers.** Wang, Veugelers & Stephan (*Research Policy*, 2017): highly novel papers are more likely to be top-1% cited long-run *and* more likely to be under-cited early and published in lower-impact venues.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no novelty instrument with published test–retest reliability on model-generated artifacts. No one has reported ICC for novelty ratings separately from overall-score ICC on the same items. Until that number exists, every "model X is more novel" claim is unfalsifiable at the measurement layer.
- **Methodologically blocked.** Distinguishing *novel to the field* from *novel to this model's training corpus* requires corpus access that closed labs do not provide; membership-inference proxies (e.g. Min-K% Prob, Shi et al., ICLR 2024) are calibrated for verbatim text, not for paraphrased ideas.
- **Empirically open.** Whether any automatic scorer predicts *post-execution* expert value. Runnable — it needs ~100 executed projects and about a year, not new theory.
- **Empirically open.** Whether novelty-optimizing generators beat a strong retrieval-plus-recombination baseline under blind expert review at $n \geq 100$ ideas.
- **Theoretically open.** Whether $\mathrm{Nov}(a \mid C_\theta)$ is identifiable from $(a, \theta)$ alone without $C_\theta$ — i.e. whether interpolation within the corpus and genuine extrapolation induce distinguishable output distributions.

## 6. Why It Is Hard

**Absent ground truth compounded by confounded measurement.** The target quantity is defined against a closure of all published knowledge that no one can enumerate, so the reference standard is expert judgment — and expert judgment on this exact construct has a measured disagreement rate near chance-corrected zero for accept/reject decisions.

Two specific obstructions follow:

1. **The instrument's noise floor exceeds the effect size.** Reported novelty gaps are ~0.8 points on a 10-point scale. With per-review standard deviation near 1.5–2 and reviewer variance dominating, a study needs hundreds of reviews to resolve the gap — which is exactly why the one well-powered study cost 79 experts.
2. **Non-identifiability against contamination.** A model reproducing a 2019 paper it memorized and a model deriving the same idea independently emit the same tokens. Nothing in the artifact separates them. Any scorer that does not condition on $C_\theta$ measures *unfamiliarity to the reviewer*, not novelty — an evaluation that does not measure what it names.

## 7. Current Research (as of 2026)

- **Stanford NLP (Si, Yang, Hashimoto)** — extending blinded ideation studies through execution; the ideation–execution gap is the load-bearing result.
- **Sakana AI + Oxford/UBC (Lu, Lange, Foerster, Clune, Ha)** — end-to-end paper generation; evaluation remains venue-acceptance-based.
- **Allen Institute for AI / academic scientometrics** — retrieval-grounded novelty checking against live literature snapshots rather than static benchmarks *(frontier — verify)*.
- **Science-of-science groups (Northwestern, Chicago, Michigan)** — repairing CD-index and atypicality measures against coverage artifacts; directly relevant since these are the only novelty measures with decades of validation.
- **Registered-report and consistency-experiment designs at ML venues** — several 2025–2026 conferences ran or planned reviewer-consistency arms *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does *any* current novelty instrument have reliability above reviewer noise on model-generated ideas?

**Scale.** 200 idea documents on 5 fixed topics: 100 model-generated (2 model families × 50), 100 human-written, all normalized to one template and one length band by a separate rewriting model to strip style cues. Each idea rated by 4 independent domain experts on three separate items — problem-novelty, method-novelty, result-novelty — plus one overall item. Total 800 reviews. Re-rate a random 50-idea subset with the *same* reviewers after 6 weeks for test–retest.

**Control arms.** (i) **Plagiarism arm**: 20 ideas that are close paraphrases of real pre-2021 papers, injected blind. (ii) **Retrieval baseline arm**: 20 ideas produced by naive retrieve-and-recombine over the same literature, no novelty optimization.

**Deciding number.** The intraclass correlation $\mathrm{ICC}(2,1)$ for the method-novelty item. **If $\mathrm{ICC} < 0.4$, every published novelty comparison in this literature is under-powered by construction** and the field must move to forced-choice pairwise designs. Secondary decider: the paraphrase arm's mean novelty score — if it does not land in the bottom quartile, the instrument is measuring reviewer unfamiliarity, not novelty.

**Cost.** ~800 expert-hours; roughly the same order as Si et al., so demonstrably feasible.

## 9. Key References

- **[Foundational]** Brian Uzzi, Satyam Mukherjee, Michael Stringer, Ben Jones. *Atypical Combinations and Scientific Impact.* Science, 2013.
- **[Foundational]** Russell Funk, Jason Owen-Smith. *A Dynamic Network Measure of Technological Change.* Management Science, 2017.
- **[SOTA]** Chenglei Si, Diyi Yang, Tatsunori Hashimoto. *Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers.* ICLR 2025. — arXiv:2409.04109
- **[SOTA]** Chenglei Si, Tatsunori Hashimoto, Diyi Yang. *The Ideation–Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas.* 2025. (identifier omitted — verify)
- **[Systems]** Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha. *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.* 2024. — arXiv:2408.06292
- **[Systems]** Yutaro Yamada et al. *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search.* 2025. — arXiv:2504.08066
- **[Measurement]** Corinna Cortes, Neil D. Lawrence. *Inconsistency in Conference Peer Review: Revisiting the 2014 NeurIPS Experiment.* 2021. — arXiv:2109.09774
- **[Measurement]** Alina Beygelzimer, Yann Dauphin, Percy Liang, Jennifer Wortman Vaughan. *Has the Machine Learning Review Process Become More Arbitrary as the Field Has Grown? The NeurIPS 2021 Consistency Experiment.* 2023. — arXiv:2306.03262
- **[Empirical]** Michael Park, Erin Leahey, Russell Funk. *Papers and Patents Are Becoming Less Disruptive Over Time.* Nature, 2023.
- **[Empirical]** Jian Wang, Reinhilde Veugelers, Paula Stephan. *Bias Against Novelty in Science: A Cautionary Tale for Users of Bibliometric Indicators.* Research Policy, 2017.
- **[Method]** Qingyun Wang et al. *SciMON: Scientific Inspiration Machines Optimized for Novelty.* ACL 2024. — arXiv:2305.14259
- **[Method]** Jinheon Baek et al. *ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models.* NAACL 2025. — arXiv:2404.07738
- **[Related]** Weijia Shi et al. *Detecting Pretraining Data from Large Language Models.* ICLR 2024. — arXiv:2310.16789
- **[Related]** Weixin Liang et al. *Can Large Language Models Provide Useful Feedback on Research Papers? A Large-Scale Empirical Analysis.* NEJM AI, 2024. — arXiv:2310.01783

## 10. Worked Example

**Instance.** Take one model-generated idea from a 2025-era system: *"Reduce hallucination in multi-hop QA by having the model generate the answer in a low-resource language first, then translate back, using the language switch as a consistency check."*

**Score it with each instrument.**

- **Retrieval distance.** Embed the idea; run against a 2024 Semantic Scholar snapshot. Nearest neighbours are cross-lingual consistency papers at $\cos \approx 0.71$, giving $\mathrm{Nov}_{\text{ret}} \approx 0.29$ — above the median for the topic. Verdict: novel.
- **Expert Likert, 4 reviewers.** Scores $\{7, 6, 3, 4\}$. Mean $5.0$, SD $1.83$. The two low scorers cite a specific 2023 cross-lingual self-consistency paper; the two high scorers do not know it. With 4 reviewers, the standard error on the mean is $1.83/\sqrt{4} = 0.92$. The 95% CI is $[3.2, 6.8]$ — wider than the entire 0.8-point gap that the field's headline result rests on.
- **CD index.** Undefined. The artifact has zero forward citations and will for five years.
- **Contamination check.** The 2023 paper is in the model's pretraining window. Min-K% Prob on the idea text returns a score inside the distribution of known-member documents, but so does a fresh paraphrase of any well-covered topic. The check cannot separate recall from rediscovery.

**Where the obstruction becomes visible.** Three instruments give three incomparable answers: retrieval says novel (0.29), experts say ambiguous ($5.0 \pm 0.92$), disruption says undefined. The expert disagreement is not about the idea's merit — it is a *coverage* disagreement about what exists, and it splits 2–2. Adding reviewers narrows the CI on $\bar{s}$ but does not resolve the coverage question, because the reviewers who know the 2023 paper and those who do not are sampling from different $K_t$. The measurement is not noisy around a well-defined value; the value itself is reviewer-relative. That is what "methodologically blocked" means here.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*