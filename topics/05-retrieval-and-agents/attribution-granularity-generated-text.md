---
id: 05-retrieval-and-agents/attribution-granularity-generated-text
title: "Attribution Granularity for Generated Text"
topic: 05-retrieval-and-agents
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attribution Granularity for Generated Text

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/attribution-granularity-generated-text` · **Status:** methodologically-blocked

## 1. Problem Statement

A retrieval-augmented system emits text $y$ and attaches citations to spans of it. **At what granularity should a span be attributed, and how do we score attribution in a way that is invariant to that choice?**

Three variants, with different difficulty:

- **Measurement.** Given $y$, a corpus $\mathcal{D}$, and a citation assignment, produce a score that ranks systems the same way regardless of whether the unit is a sentence, a clause, an atomic claim, or a token span. No current metric has this property. This is the blocked variant.
- **Method.** Build a generator that chooses its own attribution units — citing per-clause when a sentence fuses two sources, per-passage when a paragraph paraphrases one. Runnable today; blocked only because there is no metric to optimize against.
- **Theory.** Is there a canonical decomposition of a text into attributable units, unique up to some equivalence? Open, and probably false: decontextualization (making a span standalone) is not unique, and the number of "atomic facts" in a sentence is not a well-defined quantity.

Solving it means: an attribution score $A(y, \mathcal{D})$ whose system ranking is stable under decomposition changes, with a stated invariance property and a human-agreement floor.

## 2. Formal Setting

Let $y = (y_1,\dots,y_n)$ be the generated tokens and $\mathcal{D} = \{d_1,\dots,d_m\}$ the retrieved passages.

**Decomposition.** A decomposition operator $\pi$ maps $y$ to units $\pi(y) = \{u_1,\dots,u_k\}$, $u_i$ a contiguous span or a synthesized standalone claim. Measured in practice by prompting an LLM ("break this into independent facts"), so $\pi$ is a stochastic function of the prompt, model, and temperature — $k$ is not a property of $y$.

**Citation map.** $c: \pi(y) \to 2^{\mathcal{D}}$, read off the emitted markers `[1][3]`.

**Support predicate.** $S(u, C) \in \{0,1\}$: does the concatenation of $C \subseteq \mathcal{D}$ entail $u$? Measured by an NLI model or an LLM judge, not by a human at scale.

**Citation precision / recall,** the ALCE definitions:
$$\mathrm{Rec} = \frac{1}{k}\sum_{i} S\!\left(u_i, c(u_i)\right), \qquad \mathrm{Prec} = \frac{\sum_i \sum_{d \in c(u_i)} \mathbb{1}[S(u_i, c(u_i)) \wedge \neg S(u_i, c(u_i)\setminus d)]}{\sum_i |c(u_i)|}.$$
Precision penalizes a citation that can be dropped without losing support.

**The invariance we want.** For two decompositions $\pi, \pi'$ and systems $f, g$:
$$\mathrm{sign}\big(A_\pi(f) - A_\pi(g)\big) = \mathrm{sign}\big(A_{\pi'}(f) - A_{\pi'}(g)\big).$$
This is rank-invariance, not value-invariance — the weaker and still-unmet target.

**Assumptions, and how they break:**

1. *Support is monotone in the evidence set* — adding a passage never destroys support. Violated: contradictory passages flip an NLI verdict.
2. *Units are independent* — $S(u_i,\cdot)$ does not depend on $u_{j\ne i}$. Violated by pronouns, ellipsis, and scope ("the drug" / "in that trial"). Decontextualization (Choi et al., TACL 2021) is the patch, and it is itself ambiguous.
3. *$k$ is decomposition-invariant up to constants.* Violated: reported atomic-fact counts for the same biography vary by roughly $2\times$ across prompts.
4. *Entailment is binary.* Violated by partial support, numeric rounding, and hedged claims.

## 3. State of the Art

**Empirical SOTA (metrics).**
- **ALCE** (Gao, Yen, Yu, Chen, EMNLP 2023) — the standard citation precision/recall harness, sentence-level units, NLI (TRUE/T5-11B) as $S$. Established: it separates systems reproducibly. Unablated: nobody has shown its ranking survives a change of unit.
- **FActScore** (Min et al., EMNLP 2023) — LLM decomposition into atomic facts, then per-fact retrieval-grounded verification. Established: the estimator tracks human judgment closely on biographies. Its granularity dependence was not the object of study.
- **AIS** (Rashkin et al., *Computational Linguistics* 2023) — the human protocol: "is this sentence attributable to the cited source, read standalone?" Established as a definition; it is a human annotation frame, not an automatic metric.
- **MiniCheck** (Tang et al., EMNLP 2024) — a 770M-class checker matching GPT-4-level grounding accuracy on LLM-AggreFact at a small fraction of the cost. Established on that benchmark; a benchmark number, not a proof of judge validity on new domains.
- **ContextCite** (Cohen-Wang et al., NeurIPS 2024) — attribution by ablating context and fitting a surrogate; gives continuous per-source weights instead of discrete citations. Sidesteps unit choice for sources but not for the generated side.

**Direct evidence on the problem itself.** Wanner et al. (*SEM 2024, "A Closer Look at Claim Decomposition") show FActScore-style scores shift materially with the decomposition method — the single clearest published statement that the metric is granularity-dependent. Gunjal & Durrett (EMNLP 2024, "Molecular Facts") argue atomic facts are *too* atomic to be verifiable and propose larger self-contained units; this is a proposal with supporting experiments, not a settled standard.

**Theory SOTA.** None. There is no formal treatment of decomposition equivalence for attribution.

## 4. What Is Known

- **Deployed systems attribute poorly.** Liu, Zhang & Liang (EMNLP Findings 2023) audited four generative search engines (Bing Chat, NeevaAI, perplexity.ai, YouChat) on 1,450 queries: average citation **precision 74.5%**, **recall 51.5%**; only a minority of sentences are fully supported by their own citations. Scale: commercial systems, human annotation.
- **Fluency and attribution are anticorrelated** in that same audit — the most fluent, useful-seeming answers had the lowest support rates.
- **Sentence-level citation recall on ASQA/ELI5 in ALCE** sat well below human ceiling for GPT-3.5-class models at the time of publication (2023); later models improved but the harness's unit was never varied.
- **Sub-sentence supervision changes the verdict.** WICE (Kamoi et al., EMNLP 2023) shows real-world Wikipedia claims are frequently only *partially* supported by their cited evidence — a fact invisible to a sentence-level binary metric.
- **Decomposition is non-unique.** Reported atomic-fact counts for the same input vary substantially across prompts and models (Wanner et al. 2024) — the mechanism behind granularity sensitivity.
- **Automatic judges are cheap and reasonably accurate on in-distribution grounding data** (MiniCheck, LLM-AggreFact, 2024).

## 5. What Is Not Known

- **Methodologically blocked (the core).** No metric with a stated invariance under change of attribution unit exists, and no protocol defines what "the right unit" is independent of a scoring choice. Every reported citation-precision number is conditional on an unreported $\pi$.
- **Empirically open.** Whether current system *rankings* actually flip under $\pi \to \pi'$, or merely shift in level. This is runnable — one benchmark, three decompositions, five systems — and, to our knowledge, has not been run and published as its own result.
- **Empirically open.** Whether granularity sensitivity is judge-driven (NLI model brittleness on short claims) or genuinely semantic. Separable by holding $\pi$ fixed and swapping $S$.
- **Theoretically open.** Whether any $A$ can be simultaneously rank-invariant, sensitive to partial support, and computable from pairwise entailment calls. We suspect an impossibility result is available; none is published.

## 6. Why It Is Hard

**The obstruction is absent ground truth for the unit, compounded by a confounded measurement.** The quantity "number of attributable claims in this paragraph" has no observer-independent value. Two annotators shown the same sentence produce different atomic-fact sets — not from error but because atomicity is a choice about how much context travels with a claim.

That makes the denominator of every recall-style metric a free parameter. And it interacts with the judge: shorter units strip context, which raises NLI false-negatives; longer units bundle a supported and an unsupported fact, which a binary $S$ scores as unsupported (or, worse, as supported if the judge anchors on the first clause). So decomposition and judge error move together, and no published experiment separates them. The evaluation named "citation precision" measures *precision under one unrecorded segmentation by one judge* — not the property its name asserts.

## 7. Current Research (as of 2026)

- **Self-contained-unit design** — "molecular facts" style work (Gunjal & Durrett, UT Austin) making units large enough to be verifiable and small enough to be atomic. Active.
- **Cheap grounding judges** — MiniCheck/AlignScore lineage (Salesforce Research, UNC); the LLM-AggreFact leaderboard is the shared target.
- **Continuous attribution** — ContextCite (MIT, Madry group) and gradient/ablation attribution replacing discrete citations with weights; the open question is whether users can act on a weight.
- **Decomposition-robust scoring** — averaging over sampled decompositions, or scoring at multiple granularities and reporting a curve *(frontier — verify)*.
- **Agentic verification loops** — generate, decompose, retrieve, revise, in the RARR (Gao et al., ACL 2023) lineage; now standard inside deep-research products, with granularity fixed by engineering convenience rather than by evidence *(frontier — verify)*.

## 8. Concrete Next Experiment

**The granularity rank-flip test.**

- **Scale.** 500 questions from ALCE (ASQA + ELI5), 5 systems spanning the quality range (e.g. a strong closed model with citations, an open 8B RAG baseline, a self-RAG variant, a post-hoc citation attacher, and a deliberately over-citing control). ~2,500 answers. Cost: judge calls dominate; with MiniCheck-class checkers this is a few GPU-hours, not a training run.
- **Arms.** Three decompositions of every answer: $\pi_1$ sentence, $\pi_2$ clause (syntactic split, no rewriting), $\pi_3$ LLM atomic facts with decontextualization. Two judges ($S$): an NLI checker and an LLM judge. Six cells per system.
- **Control arm.** The over-citing system — cites every retrieved passage on every unit. Its recall must be near 1 and its precision near $1/|\mathcal{D}|$ in *every* cell. If it is not, the metric is broken independently of granularity.
- **Deciding number.** Kendall's $\tau$ between the 5-system rankings under $\pi_1$ and $\pi_3$, judge held fixed. **$\tau \geq 0.9$**: granularity is a nuisance parameter, publish the constant and move on. **$\tau \leq 0.6$**: citation precision/recall as currently reported does not identify a system property, and every leaderboard using it needs its $\pi$ stated. A secondary number — the $\tau$ across judges with $\pi$ fixed — attributes the instability to judge or to decomposition.

## 9. Key References

- **[Foundational]** Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, David Reitter. *Measuring Attribution in Natural Language Generation Models.* Computational Linguistics, 2023.
- **[Foundational]** Eunsol Choi, Jennimaria Palomaki, Matthew Lamm, Tom Kwiatkowski, Dipanjan Das, Michael Collins. *Decontextualization: Making Sentences Stand-Alone.* TACL, 2021.
- **[SOTA]** Tianyu Gao, Howard Yen, Jiatong Yu, Danqi Chen. *Enabling Large Language Models to Generate Text with Citations.* EMNLP, 2023. — arXiv:2305.14627
- **[SOTA]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[SOTA]** Liyan Tang, Philippe Laban, Greg Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.* EMNLP, 2024.
- **[SOTA]** Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, Aleksander Madry. *ContextCite: Attributing Model Generation to Context.* NeurIPS, 2024.
- **[Evidence]** Nelson F. Liu, Tianyi Zhang, Percy Liang. *Evaluating Verifiability in Generative Search Engines.* Findings of EMNLP, 2023.
- **[Evidence]** Miriam Wanner, Seth Ebner, Zhengping Jiang, Mark Dredze, Benjamin Van Durme. *A Closer Look at Claim Decomposition.* *SEM, 2024.
- **[Evidence]** Anisha Gunjal, Greg Durrett. *Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification.* Findings of EMNLP, 2024.
- **[Evidence]** Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, Greg Durrett. *WICE: Real-World Entailment for Claims in Wikipedia.* EMNLP, 2023.
- **[Related]** Luyu Gao, Zhuyun Dai, Panupong Pasupat, et al. *RARR: Researching and Revising What Language Models Say, Using Language Models.* ACL, 2023.
- **[Related]** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR, 2024.

## 10. Worked Example

One generated sentence, cited:

> "Metformin was first marketed in France in 1979 and is now the most prescribed oral antidiabetic worldwide. [1]"

Passage [1] is a review stating metformin's French launch year and that it is first-line therapy for type 2 diabetes. It does **not** state "most prescribed worldwide."

**Sentence granularity ($\pi_1$).** $k=1$. The judge must decide support for a conjunction where one conjunct is grounded and one is not. Strict entailment gives $S=0$: recall $0/1 = 0$, precision $0/1 = 0$.

**Clause granularity ($\pi_2$).** $k=2$: "Metformin was first marketed in France in 1979" ($S=1$) and "is now the most prescribed oral antidiabetic worldwide" ($S=0$). Recall $1/2 = 0.5$, precision $1/2 = 0.5$.

**Atomic + decontextualized ($\pi_3$).** A typical LLM decomposition yields three units: *marketed in France*, *marketed in 1979*, *most prescribed oral antidiabetic worldwide*. Recall $2/3 \approx 0.67$.

**The obstruction, visible.** The same sentence, the same source, the same judge — recall $0$, $0.5$, or $0.67$ depending only on how it was cut. The spread is $0.67$, which is larger than the entire gap between the best and worst system in most published citation-recall tables. Worse, the direction is not a constant offset: a system that writes *short, single-fact sentences* scores identically under all three $\pi$, while a system that writes *fused, information-dense sentences* gains up to $+0.67$ by moving from $\pi_1$ to $\pi_3$. So the granularity choice does not merely shift the scale — it rewards a specific writing style. Two systems separated by $0.1$ recall in an ALCE table can swap order under a different unit, and no published table records which unit was used beyond "sentence."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*