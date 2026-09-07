---
id: 21-factuality/attribution-citation-support-verification
title: "Faithful Attribution: Do Citations Support the Cited Text"
topic: 21-factuality
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Faithful Attribution: Do Citations Support the Cited Text

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/attribution-citation-support-verification` · **Status:** partially-solved

## 1. Problem Statement

A system answers a question and attaches citations to spans of its answer. The question is whether each cited source actually supports the span it is attached to — not whether the span is true, and not whether the source is relevant.

- **Input:** a generated response $y$, a corpus $\mathcal{D}$ of retrievable passages, and a citation assignment mapping response spans to subsets of $\mathcal{D}$.
- **Output:** for each (span, citation-set) pair, a support label in $\{\text{full}, \text{partial}, \text{none}\}$, plus response-level precision/recall aggregates.
- **Solved** means: an automatic judge whose per-pair labels agree with careful adjudicated human labels within a stated tolerance, on out-of-distribution generators and domains, with the disagreement characterized rather than merely averaged away.

Three variants, routinely conflated:

- **Measurement.** Can we *score* attribution reliably? Currently the binding constraint.
- **Method.** Can we *build* generators whose citations are supporting? Partly solved for short-form QA, unsolved for long-form and multi-document synthesis.
- **Theory.** Is there a decomposition of free text into atomic, independently verifiable claims that is unique up to a stated equivalence? No such theory exists; claim decomposition is currently an empirical convention.

## 2. Formal Setting

Let $x$ be a query, $y$ a response, and $\mathcal{D}$ a corpus. A **decomposition** operator $\Delta$ maps $y$ to claims $\Delta(y) = (c_1,\dots,c_n)$ with citation sets $C_i \subseteq \mathcal{D}$.

**Ground-truth support** is the AIS predicate (Rashkin et al., 2023): $\phi(c, S) = 1$ iff a generic hearer, given only $S$, would affirm "According to $S$, $c$", with $c$ interpretation-resolved in the context of $x$ and preceding claims.

$$\text{Prec}(y) = \frac{1}{|\{i : C_i \neq \emptyset\}|}\sum_{i: C_i \neq \emptyset} \phi(c_i, C_i), \qquad \text{Rec}(y) = \frac{1}{n}\sum_{i=1}^{n} \phi(c_i, C_i)$$

ALCE-style **citation precision** additionally penalizes redundant citations: a citation $d \in C_i$ is precise iff $\phi(c_i, C_i) = 1$ and $\phi(c_i, C_i \setminus \{d\}) = 0$ — i.e. removing it breaks support.

**As actually measured**, $\phi$ is replaced by a judge $\hat{\phi}_\theta$ (an NLI model or a prompted LLM) and $\Delta$ by a prompted decomposer $\hat{\Delta}$. The reported number is therefore

$$\widehat{\text{Rec}}(y) = \mathbb{E}_{\hat\Delta}\Big[\tfrac{1}{n}\textstyle\sum_i \hat\phi_\theta(c_i, C_i)\Big] = \text{Rec}(y) + \underbrace{b_\theta(y)}_{\text{judge bias}} + \underbrace{b_\Delta(y)}_{\text{decomposition bias}}.$$

Neither bias term is estimated in most published results; they are assumed zero.

Assumptions, with the ones known to be violated marked:

1. **Claim independence** — each $c_i$ is verifiable alone. *Violated*: pronouns, ellipsis, and discourse-scoped quantifiers make many spans unverifiable out of context (WiCE, Kamoi et al., 2023).
2. **Decomposition uniqueness** — $\hat\Delta$ is stable. *Violated*: different decomposers produce different claim counts for the same text, and finer decompositions mechanically raise measured support rates.
3. **Judge independence from generator** — $\hat\phi_\theta$ is not correlated with the generator's errors. *Violated* whenever judge and generator share a base model; self-preference inflates precision.
4. **Corpus-closed truth** — support is decidable from $C_i$ alone. *Violated* by claims requiring arithmetic, temporal reasoning, or numeric aggregation over the source.
5. **Binary support** — *violated*: partial support is the modal failure in long-form answers.

## 3. State of the Art

**Established (independently reproduced):**

- The AIS framework (Rashkin et al., *Computational Linguistics* 2023) gives an annotation protocol with reported inter-annotator agreement in the substantial range, and is the definitional basis for essentially all later work.
- ALCE (Gao et al., EMNLP 2023) is the standard automatic benchmark: NLI-based citation precision/recall over ASQA, QAMPARI, ELI5. Its central finding — that strong LLMs leave a large fraction of statements not fully supported by their own citations — has been reproduced across model families.
- Fine-tuned lightweight entailment checkers match large prompted judges on grounded fact-checking. MiniCheck (Tang et al., EMNLP 2024) reports a 770M-class model reaching GPT-4-level balanced accuracy on the 10-dataset LLM-AggreFact benchmark at roughly $400\times$ lower inference cost.
- Sentence-level entailment aggregation beats document-level scoring for long inputs (SummaC, Laban et al., *TACL* 2022).

**Claimed but unablated:**

- That LLM judges "align with humans" on attribution. Reported agreement is nearly always against single-annotator labels on in-distribution data; the cross-generator and cross-domain transfer ablation is usually missing.
- That citation-training (RLHF or SFT on cited data) improves *support* rather than *citation density*. Most papers report precision/recall under the same automatic judge used during development.

**Benchmark-number-only:** generative-search-engine audits. Liu, Zhang & Liang (EMNLP Findings 2023) hand-annotated four commercial systems and report on average only about half of generated sentences fully supported by their citations, and about three-quarters of citations supporting their associated sentence. These are single-snapshot measurements of products that have since changed; they have not been re-run under a fixed protocol.

## 4. What Is Known

- **Human-verified support is far below citation coverage.** Liu et al. (2023), ~1,450 human-annotated sentences across four commercial systems: citation recall ≈ 51.5%, citation precision ≈ 74.5%, and fluency/utility correlated *negatively* with verifiability.
- **Expert domains are worse.** ExpertQA (Malaviya et al., NAACL 2024), 2,177 expert-written questions with expert verification, finds attribution failures at higher rates than in open-domain QA, including for retrieval-augmented systems.
- **Automatic attribution judges are brittle.** AttrScore (Yue et al., EMNLP Findings 2023) shows prompted GPT-4 scoring well on synthetically perturbed pairs but degrading substantially on human-annotated generative-search outputs, particularly on the *extrapolatory* class where the source is topically right but does not entail the claim.
- **Partial support dominates.** WiCE (Kamoi et al., EMNLP 2023), built on real Wikipedia citations, finds a large share of claims are only partly supported; binary labels force an arbitrary rounding.
- **Decomposition granularity moves the score.** FActScore (Min et al., EMNLP 2023) established atomic-fact decomposition for biography generation and reported large precision differences across models at fixed decomposer; the converse — score sensitivity to decomposer choice at fixed model — is documented but not systematically bounded.
- **Post-hoc revision works partially.** RARR (Gao et al., ACL 2023) improves attribution while largely preserving the original text, but is evaluated with the same class of automatic judge it optimizes against.

## 5. What Is Not Known

- **Methodologically blocked:** whether attribution scores are comparable across papers at all. Without a stable $\hat\Delta$ and a bias-corrected $\hat\phi$, $\widehat{\text{Rec}}$ is a property of the evaluation pipeline as much as of the generator. The decomposition-invariance requirement has no accepted formalization.
- **Methodologically blocked:** the semantics of partial support. There is no accepted rule for scoring a claim whose numeric or temporal component is unsupported but whose main assertion is.
- **Empirically open:** the size and sign of judge bias $b_\theta$ when judge and generator share a base model. Runnable today; not run at scale with adjudicated labels.
- **Empirically open:** whether citation-optimized training improves human-verified support or only judge-measured support. Requires a held-out human evaluation with a judge the training never saw.
- **Theoretically open:** whether there exists a decomposition operator with a provable guarantee that response-level support is invariant to granularity under a stated claim-equivalence relation.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement over an unidentified decomposition**. Two distinct freedoms — how text is cut into claims, and how support is judged — multiply, and both are chosen by the evaluator. Making claims finer raises measured support because short claims are easier to entail; making the judge stronger changes which claims count as supported. Neither knob is reported as a confound, so a 5-point improvement in citation recall is not identifiable as a property of the generator.

Second obstruction: **absent ground truth at scale**. AIS annotation costs minutes per claim and requires reading the full cited passage; published human sets are in the low thousands of claims, while benchmark leaderboards score hundreds of thousands. The gap is filled by an automatic judge whose error is assumed unbiased and never measured on the generator being scored.

## 7. Current Research (as of 2026)

- **Cheap specialized verifiers.** Continued work on small entailment checkers in the MiniCheck/AlignScore line, evaluated on LLM-AggreFact-style aggregates (CMU, Salesforce Research, academic groups).
- **Claim decomposition as a first-class object.** Decontextualization-before-verification pipelines and studies of decomposition sensitivity *(frontier — verify: no consensus protocol has emerged)*.
- **Inline attribution in deployed assistants.** Commercial search-augmented assistants ship citations by default; independent audits under a fixed protocol remain rare *(frontier — verify)*.
- **Attribution as training signal.** Reward models keyed on citation support rather than preference; the open concern is reward hacking against the judge *(frontier — verify)*.
- **Regulatory pressure.** Provenance and source-disclosure requirements are pushing attribution from a research metric toward a compliance one, which raises the cost of an unvalidated metric.

## 8. Concrete Next Experiment

**Question:** how large is judge bias $b_\theta$, and does it depend on generator identity?

- **Scale.** 3,000 (claim, citation-set) pairs: 3 generator families $\times$ 2 domains (open-domain QA, expert/biomedical) $\times$ 500 pairs. Every pair labeled by 2 trained annotators under the AIS protocol with a third-annotator adjudication of disagreements. Estimated cost: ~600 annotator-hours.
- **Judges scored on the same pairs:** (a) a small fine-tuned entailment checker; (b) a prompted large judge from a family *disjoint* from all generators; (c) a prompted judge from the *same* family as generator 1.
- **Control arm.** The same 3,000 claims with citation sets permuted within the retrieved pool for that query — topically related, non-supporting sources. True precision on this arm is near the base rate; any judge scoring it materially above base rate is detecting topical relatedness, not entailment.
- **Deciding number.** The signed bias $\hat{b}_\theta = \widehat{\text{Prec}}_\theta - \text{Prec}_{\text{human}}$ per judge per generator. If $|\hat b_\theta| \le 2$ points uniformly and the same-family judge shows no excess bias on its own generator ($\Delta \le 1$ point), automatic attribution scores are usable for cross-system comparison. If the same-family excess exceeds 5 points, every self-judged attribution result in the literature is uninterpretable as a cross-system comparison.

Secondary readout: recompute $\widehat{\text{Rec}}$ under three decomposers at fixed generator and judge. The spread is a direct estimate of $b_\Delta$.

## 9. Key References

- **[Foundational]** Rashkin, Nikolaev, Lamm, Aroyo, Collins, Das, Petrov, Tomar, Turc, Reitter. *Measuring Attribution in Natural Language Generation Models.* Computational Linguistics, 2023. — arXiv:2112.12870
- **[Foundational]** Bohnet, Tran, Verga, Aharoni, Andor, Baldini Soares, et al. *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models.* 2022. — arXiv:2212.08037
- **[SOTA / benchmark]** Gao, Yen, Yu, Chen. *Enabling Large Language Models to Generate Text with Citations.* EMNLP, 2023. — arXiv:2305.14627
- **[SOTA / audit]** Liu, Zhang, Liang. *Evaluating Verifiability in Generative Search Engines.* Findings of EMNLP, 2023. — arXiv:2304.09848
- **[SOTA / verifier]** Tang, Laban, Durrett. *MiniCheck: Efficient Fact-Checking of LLM Outputs on Grounding Documents.* EMNLP, 2024. — arXiv:2404.10774
- **[Method]** Gao, Dai, Pasupat, Chen, Chaganty, Fan, Zhao, Lao, Lee, Juan, Guu. *RARR: Researching and Revising What Language Models Say, Using Language Models.* ACL, 2023. — arXiv:2210.08726
- **[Evaluation of evaluators]** Yue, Chen, Zhang, Sun, Huang. *Automatic Evaluation of Attribution by Large Language Models.* Findings of EMNLP, 2023. — arXiv:2305.06311
- **[Decomposition]** Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Partial support]** Kamoi, Goyal, Diego Rodriguez, Durrett. *WiCE: Real-World Entailment for Claims in Wikipedia.* EMNLP, 2023. — arXiv:2303.01432
- **[Expert domains]** Malaviya, Lee, Chen, Sieber, Yatskar, Roth. *ExpertQA: Expert-Curated Questions and Attributed Answers.* NAACL, 2024. — arXiv:2309.07852
- **[Aggregation]** Laban, Schnabel, Bennett, Hearst. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL, 2022. — arXiv:2111.09525

## 10. Worked Example

Query: *"How much did global installed solar PV capacity grow in 2023?"* The system returns four sentences, each with one citation.

| # | Claim | Cited passage says | AIS (human) | LLM judge |
|---|---|---|---|---|
| 1 | Global solar PV additions reached a record in 2023. | "2023 saw record solar additions." | full | full |
| 2 | Additions were about 375 GW. | "Solar additions exceeded 340 GW." | none (number unsupported) | full |
| 3 | China accounted for over half of new capacity. | "China installed 216 GW of the global total." | partial (needs the global total, in another passage) | full |
| 4 | This growth was driven by falling module prices. | Passage discusses module prices falling; asserts no causal link. | none (extrapolatory) | full |

Human adjudicated precision: $1/4 = 0.25$ under strict full-support scoring, $0.375$ if partial counts as half. Judge precision: $4/4 = 1.00$. Bias on this response: $b_\theta = +0.75$.

Now the decomposition confound. Split claim 2 into "Additions were large" + "Additions were about 375 GW". The first is supported, so measured recall rises from $0.25$ to $0.40$ with **no change to the generator or the sources**. Split claim 3 into "China installed 216 GW" + "that was over half", and measured recall rises again.

The obstruction is visible in one line: two of the four judge errors (claims 3 and 4) are the judge rewarding *topical relatedness* — exactly what the permuted-citation control arm in §8 is designed to catch — and the score movement from $0.25$ to $0.40$ came from re-cutting the text, which no published attribution number controls for.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*