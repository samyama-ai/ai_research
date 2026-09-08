---
id: 21-factuality/citation-fabrication-detection-without-corpus
title: "Detecting Fabricated Citations Without an External Corpus"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Fabricated Citations Without an External Corpus

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/citation-fabrication-detection-without-corpus` · **Status:** open

## 1. Problem Statement

A language model emits a bibliographic reference — authors, title, venue, year, sometimes an identifier. Some such references correspond to real publications; some are fluent composites of real fragments. **Lookup solves the easy case**: query Crossref, Semantic Scholar, or OpenAlex and check. This entry is about the case where lookup is unavailable, incomplete, or too slow: no retrieval, no index, no web. The detector sees only the reference string, the prompt, and the generating model (weights, logits, activations, and the ability to resample).

- **Input:** a prompt $x$, a generated reference string $c$, and white- or grey-box access to the generator $p_\theta$.
- **Output:** a score $s(c, x, \theta) \in \mathbb{R}$, thresholded to a decision *real* / *fabricated*.
- **Objective:** maximize AUROC, and more usefully, maximize true-positive rate at a fixed false-positive rate (flagging real citations is expensive for the user).

Three variants, of very different difficulty:

- **Measurement.** Can we even build a clean labelled test set? Ground truth "this reference does not exist" is a negative existential over an incomplete union of catalogs.
- **Method.** Given labels, does any internal signal — token entropy, self-consistency, probe on hidden states — separate fabricated from real *at matched popularity*?
- **Theory.** Is the separation information-theoretically possible at all? A model that never stored a rare-but-real citation and a model that composed a fake one may be in the same epistemic state.

Solving it means: an AUROC $\ge 0.9$ **on the long-tail stratum** (references cited fewer than ~10 times), with no corpus access, reproduced across at least two model families.

## 2. Formal Setting

Let $\mathcal{C}$ be the set of syntactically well-formed references. Let $\mathcal{R} \subset \mathcal{C}$ be the real ones. The generator induces $p_\theta(c \mid x)$ over continuations. Define the ground-truth label
$$y(c) = \mathbb{1}[c \in \mathcal{R}].$$

**As measured**, $\mathcal{R}$ is replaced by $\hat{\mathcal{R}}_K$, the union of $K$ catalogs (Crossref $\approx 1.6\times10^8$ records, OpenAlex $\approx 2.5\times10^8$ works), plus fuzzy matching: $c$ counts as real if some record $r$ satisfies $\mathrm{sim}(c,r) \ge \tau$ under normalized title edit distance with author-surname overlap. $\tau$ is a free parameter and the labels move with it.

Detector families:

- **Sequence uncertainty.** Length-normalized negative log-likelihood $\;u(c) = -\tfrac{1}{|c|}\sum_t \log p_\theta(c_t \mid c_{<t}, x)$.
- **Self-consistency.** Draw $c^{(1)},\dots,c^{(m)} \sim p_\theta(\cdot \mid x')$ under a paraphrased or *indirect* prompt $x'$ (e.g. "who wrote *T*?"), and score field agreement $\;a(c) = \tfrac{1}{m}\sum_i \mathbb{1}[\mathrm{field}(c^{(i)}) = \mathrm{field}(c)]$.
- **Semantic entropy.** Cluster samples by bidirectional entailment into classes $\mathcal{K}$ and compute $\;H_{\mathrm{sem}} = -\sum_{k \in \mathcal{K}} p(k)\log p(k)$ (Farquhar et al., *Nature* 2024).
- **Probes.** $g_\phi(h_\ell(c))$, a linear or low-rank readout on layer-$\ell$ activations, trained on labelled references and evaluated out-of-distribution.

The confound to be controlled is training-corpus frequency. Let $n(c)$ be the number of occurrences of $c$ in the pretraining data. Report **stratified** AUROC: $\mathrm{AUROC}\big(s \mid n(c) \in [1,10]\big)$, not the pooled number.

**Assumptions, and their status:**

1. $\hat{\mathcal{R}}_K \approx \mathcal{R}$ — *violated*. Preprints, theses, non-English and pre-1960 work are patchily indexed; "not found" and "not real" are conflated.
2. $n(c)$ is observable — *violated for every frontier model*; pretraining data is closed. Proxies (citation count, Google Scholar hits) are correlated but not equal.
3. A reference is atomically real or fake — *violated*. Real paper with wrong year, right authors with wrong title, and right title attributed to the wrong venue are distinct error modes with different detectability.
4. Samples $c^{(i)}$ are i.i.d. draws from the model's belief — *violated* by decoding temperature, prompt-order effects, and RLHF-induced mode collapse.

## 3. State of the Art

**Established (ablated, reproduced):**

- *Indirect queries beat direct queries.* Agrawal et al. (Findings of EACL 2024, arXiv:2305.18248) show that asking the model a downstream question about the reference ("who are the authors of *T*?") and measuring cross-sample consistency detects fabricated references better than asking "does this paper exist?". The direct query is contaminated by sycophancy and by the model's prior on assent.
- *Sampling-based consistency beats raw likelihood.* SelfCheckGPT (Manakul et al., EMNLP 2023) and semantic entropy (Farquhar et al., *Nature* 630, 2024) both establish that resampling-plus-agreement dominates token-probability baselines for free-form factuality; semantic entropy reports AUROC $\approx 0.79$ vs $\approx 0.70$ for naive predictive entropy on short-form QA.
- *Calibrated models must fabricate.* Kalai & Vempala (STOC 2024) prove that for facts appearing exactly once in the training corpus ("monofacts"), a calibrated generator's hallucination rate is lower-bounded, to leading order, by the monofact fraction. Citations are the canonical monofact-heavy domain.

**Claimed but unablated:**

- Linear probes on hidden states as lie/hallucination detectors (Azaria & Mitchell, Findings of EMNLP 2023; CCS, Burns et al., ICLR 2023). High in-distribution accuracy; transfer to *citation existence* specifically, at matched popularity, has not been ablated.
- "Model knows internally that the citation is fake." Attractive, and consistent with the probe literature, but no published study separates *the model has no stored record* from *the model has a record and is being incoherent*.

**Benchmark-number-only:** CiteME (Press et al., NeurIPS 2024) reports GPT-4o-class agents at ~27% accuracy on attributing a described paper vs ~69% for a human expert — a retrieval-enabled attribution number, not a corpus-free detection number. It is frequently miscited as evidence about fabrication detection; it is not.

## 4. What Is Known

- **Base rates, 2023 scale.** Walters & Wilder (*Scientific Reports* 13, 2023) hand-checked ChatGPT-generated bibliographies: ~55% of GPT-3.5 citations and ~18% of GPT-4 citations were fabricated; among the non-fabricated, a substantial minority carried substantive metadata errors. Medical-domain replications (Bhattacharyya et al., *Cureus* 2023) found comparable or worse rates on a few hundred references.
- **Fabrication rate falls with scale and with RAG, but does not vanish.** Every reported 2024–2025 measurement leaves a non-zero residue in the long tail.
- **Long-tail knowledge is the driver.** Kandpal et al. (ICML 2023) show QA accuracy scales roughly log-linearly with the number of pretraining documents supporting a fact; models need orders of magnitude more parameters to compensate for rarity. Carlini et al. (ICLR 2023) show verbatim memorization increases with duplication count. A citation string is exactly a rare, near-verbatim string.
- **Uncertainty methods work better on short-form than on structured composites.** Reported AUROCs cluster in 0.75–0.85 for entity-level QA; published citation-specific numbers are sparser and generally lower, and are almost never stratified by popularity.

## 5. What Is Not Known

- **Methodologically blocked (the dominant blockage).** There is no agreed operationalization of "fabricated" that survives catalog incompleteness. Every public evaluation conflates *absent from $\hat{\mathcal{R}}_K$* with *nonexistent*, and none report inter-annotator agreement on the residual cases. Until labels are defined, AUROC differences below ~0.05 are uninterpretable.
- **Empirically open.** Nobody has published a *popularity-stratified* corpus-free detection curve. The experiment is runnable today on an open-data model (OLMo 2, Pythia) where $n(c)$ is directly countable. This is the single largest cheap gain available.
- **Theoretically open.** Whether fabricated and rare-but-real references are distinguishable from $p_\theta$ alone. Kalai & Vempala bound the *generation* rate; they say nothing about *post-hoc detectability*. A separation theorem — or an impossibility result showing the two induce identical output distributions under a stated model of memorization — does not exist.

## 6. Why It Is Hard

**Non-identifiability under an absent ground truth.** The signal every corpus-free detector actually measures is *the model's uncertainty about the reference*. Uncertainty is high in two disjoint cases: (i) the reference was invented; (ii) the reference is real but appeared once or twice in pretraining. Case (ii) is not a rare corner — the citation distribution is heavy-tailed, and by Kalai & Vempala the monofact mass is precisely where fabrication concentrates. The detector is therefore asked to split a class it has no feature for.

This is compounded by a measurement confound: pooled AUROC on a naively sampled reference set is dominated by popular real citations (easy positives) and by wildly implausible fakes (easy negatives). A detector scoring 0.88 pooled can score near chance on the $n(c) \le 10$ stratum, which is the only stratum a user needs help with. Reported numbers do not name the stratum, so **the evaluation does not measure the thing it names**.

## 7. Current Research (as of 2026)

- **Cheap internal-state detectors.** Semantic entropy probes (Kossen et al., 2024, OATML Oxford) approximate sampling-based entropy from a single forward pass; extension to structured bibliographic fields is active *(frontier — verify)*.
- **Open-data auditing.** Groups with pretraining-corpus access (AI2 on OLMo/Dolma; EleutherAI on Pythia) can compute $n(c)$ exactly. Frequency-conditioned hallucination studies exist for QA; citation-specific replication is the obvious next step *(frontier — verify)*.
- **Attribution-first generation.** ALCE (Gao et al., EMNLP 2023) and successors sidestep detection by forcing generation from retrieved passages. This is the deployed answer, and it changes the problem rather than solving it: the corpus-free case remains for offline, air-gapped, and paywalled settings.
- **Publisher-side screening.** Journals and preprint servers running automated reference checks report fabricated-citation interception; these pipelines are corpus-based and thus outside this entry's scope, but they are the source of the best labelled data.

## 8. Concrete Next Experiment

**Question:** does any corpus-free detector beat a popularity-only baseline in the long tail?

- **Scale.** One open-data model with a countable corpus (OLMo 2 7B or Pythia 12B). Build 4,000 references: 2,000 real, sampled from OpenAlex and **stratified by exact pretraining count** $n(c) \in \{[1,3],[4,10],[11,100],[101,\infty)\}$ at 500 each; 2,000 fabricated, generated by prompting the model for citations on 200 topics and verified absent from the union of Crossref + OpenAlex + arXiv, with 200 borderline cases adjudicated by two annotators (report $\kappa$).
- **Arms.** (1) length-normalized NLL; (2) indirect-query consistency, $m=20$ samples; (3) semantic entropy; (4) linear probe on mid-layer activations, trained on the $n \ge 101$ stratum only; (5) **control arm — a popularity oracle** that scores each reference by $\log n(c)$ alone, using nothing else. Arm 5 has zero knowledge of existence; it only knows rarity.
- **Deciding number.** $\Delta = \mathrm{AUROC}_{\text{best detector}} - \mathrm{AUROC}_{\text{popularity oracle}}$, computed **on the $n(c) \in [1,3]$ stratum only**, with a bootstrap 95% CI. If $\Delta \le 0.03$, corpus-free detection is measuring rarity, not fabrication, and the field should stop reporting pooled AUROC. If $\Delta \ge 0.10$, a genuine existence signal is present in the weights and the theory question becomes urgent.
- **Cost.** ~$10^5$ generations at $m=20$; single 8×A100 node, under 48 hours.

## 9. Key References

- **[Foundational]** Kalai, A. T., & Vempala, S. S. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Foundational]** Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. *Large Language Models Struggle to Learn Long-Tail Knowledge.* ICML, 2023. — arXiv:2211.08411
- **[SOTA]** Agrawal, A., Suzgun, M., Mackey, L., & Kalai, A. T. *Do Language Models Know When They're Hallucinating References?* Findings of EACL, 2024. — arXiv:2305.18248
- **[SOTA]** Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 2024.
- **[SOTA]** Manakul, P., Liusie, A., & Gales, M. J. F. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP, 2023. — arXiv:2303.08896
- **[Empirical]** Walters, W. H., & Wilder, E. I. *Fabrication and Errors in the Bibliographic Citations Generated by ChatGPT.* Scientific Reports 13, 2023.
- **[Empirical]** Press, O., Hochlehnert, A., Prabhu, A., Udandarao, V., Press, O., & Bethge, M. *CiteME: Can Language Models Accurately Cite Scientific Claims?* NeurIPS, 2024.
- **[Related]** Azaria, A., & Mitchell, T. *The Internal State of an LLM Knows When It's Lying.* Findings of EMNLP, 2023. — arXiv:2304.13734
- **[Related]** Min, S., et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Survey]** Ji, Z., et al. *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys 55(12), 2023.

## 10. Worked Example

Two references, one fabricated, one real-but-rare. Numbers below are a stipulated but realistic instantiation of the arm-2 protocol, chosen to expose the obstruction.

- $c_1$ (fabricated composite): *"Bengio, Y., Courville, A., Vincent, P. — Deep Sparse Rectifier Representations for Transfer Learning. NeurIPS, 2012."* Every fragment is real; the paper is not.
- $c_2$ (real, long tail): a 1997 workshop paper with 4 lifetime citations, present once in the pretraining corpus.

Run the indirect query "In what venue and year did *T* appear?" with $m = 20$ samples at $T = 1.0$. Modal-answer agreement:

| | modal venue | agreement $a$ | field entropy $H$ (nats) |
|---|---|---|---|
| $c_1$ fabricated | NeurIPS | 6/20 = 0.30 | 1.71 |
| $c_2$ real, $n=1$ | (correct workshop) | 7/20 = 0.35 | 1.66 |
| $c_3$ real, $n>10^3$ (Adam, ICLR 2015) | ICLR | 20/20 = 1.00 | 0.00 |

A threshold at $a \ge 0.5$ separates $c_3$ from both others perfectly. It does not separate $c_1$ from $c_2$ at all: the gap is 0.05 in agreement and 0.05 nats in entropy, well inside the sampling noise of $m=20$ (binomial SE at $p\approx0.3$ is $\approx 0.10$).

Now aggregate. If the evaluation set is 70% popular real citations and 30% obvious fakes, pooled AUROC lands around 0.86 and the method looks solved. Restrict to $n(c) \le 3$ and it falls toward 0.55–0.60 — indistinguishable from the popularity oracle, which knows nothing about existence.

That is the obstruction in one table. The detector is a rarity meter wearing an existence label, and the pooled metric hides it. The next experiment's only job is to publish the stratified column.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*