---
id: 21-factuality/context-parametric-knowledge-conflict-resolution
title: "Retrieval Augmentation Under Conflicting Evidence"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Retrieval Augmentation Under Conflicting Evidence

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/context-parametric-knowledge-conflict-resolution` · **Status:** open

## 1. Problem Statement

A retrieval-augmented model receives a query $q$ and a retrieved context $C = \{c_1,\dots,c_k\}$. The context may contradict the model's parametric memory (context–memory conflict), and the passages may contradict each other (inter-context conflict). The model must emit an answer, and ideally a decision: **trust the context**, **trust memory**, or **abstain / report the conflict**.

Three variants, with different difficulty:

- **Measurement.** Given a deployed system, estimate the rate at which it resolves conflicts *correctly* — not merely the rate at which it follows context. Following context is trivially maximized by ignoring memory; that is not the goal when the context is a poisoned or stale passage.
- **Method.** Build a decoder, training objective, or routing policy that raises correct-resolution rate on both arms (adopt-when-context-is-right, resist-when-context-is-wrong) without trading one against the other.
- **Theory.** Characterize when the correct source is *identifiable at all* from $(q, C, \theta)$. If a counterfactual passage is textually indistinguishable from a true update, no decision rule can separate them, and the ceiling is set by the evidence, not the model.

Solving it means: a system whose correct-resolution rate exceeds the best single-policy baseline (always-context, always-memory) by a margin that survives an adversarial conflict set, plus calibrated abstention when the conflict is genuinely unresolvable.

## 2. Formal Setting

Let $\theta$ be model parameters, $q$ a query with gold answer $a^\star$ under a reference knowledge state $K$ at time $t$. Define:

- **Parametric answer** $a_\theta(q) = \arg\max_a p_\theta(a \mid q)$, measured with the empty context and a fixed prompt template.
- **Contextual answer** $a_C(q) = \arg\max_a p_\theta(a \mid q, C)$.
- **Conflict indicator** $\kappa(q,C) = \mathbb{1}[a_\theta(q) \neq a_{\text{ctx}}(C,q)]$, where $a_{\text{ctx}}$ is the answer *entailed by* $C$ — measured by an NLI model or by construction when $C$ is synthetically substituted.

On the conflict subset $\mathcal{D}_\kappa = \{(q,C) : \kappa = 1\}$, partition by which source is right: $\mathcal{D}^{+}$ (context correct, memory stale/wrong) and $\mathcal{D}^{-}$ (memory correct, context corrupted). Define

$$\text{ACR} = \tfrac{1}{2}\Big(\Pr_{\mathcal{D}^{+}}[\hat a = a^\star] + \Pr_{\mathcal{D}^{-}}[\hat a = a^\star]\Big),$$

the balanced **adjudicated conflict resolution** rate. Always-context gives $\text{ACR} = 0.5$; so does always-memory. The **context reliance rate** used in most papers, $\text{CRR} = \Pr_{\mathcal{D}_\kappa}[\hat a = a_{\text{ctx}}]$, is *not* an accuracy measure and is maximized by a degenerate policy.

Context-aware decoding (Shi et al., 2024) reweights logits:
$$\log \tilde p(y_t) \propto (1+\alpha)\log p_\theta(y_t \mid C, q, y_{<t}) - \alpha \log p_\theta(y_t \mid q, y_{<t}),$$
a fixed global amplification of the context–memory difference with no per-instance evidence assessment.

Assumptions, and their status in practice:

1. **Single gold answer per query.** Violated: temporal and multi-answer questions have valid alternatives (Chen et al., 2022).
2. **$a_\theta(q)$ is a stable read of memory.** Violated: parametric answers move with prompt phrasing and few-shot format; the same model gives different "priors" under paraphrase (Du et al., 2024).
3. **Conflict is detectable by entailment.** Violated for numeric drift, partial overlap, and implicit premises.
4. **Synthetic entity substitution matches natural conflict.** Violated: substituted counterfactuals are often type-consistent but corpus-implausible, so measured resistance overstates real-world resistance (Kortukov et al., 2024).

## 3. State of the Art

**Empirical / systems.**
- *Context-aware decoding* (Shi et al., NAACL 2024): training-free contrastive decoding; reported large gains on knowledge-conflict QA and summarization faithfulness across LLaMA sizes. Established as a CRR-raiser; **not ablated** on $\mathcal{D}^{-}$ — the paper does not show it preserves resistance to corrupted context.
- *Context-faithful prompting* (Zhou et al., Findings EMNLP 2023): opinion-framing plus counterfactual demonstrations. Gains are prompt-template-sensitive; effect size drops sharply with template changes in follow-ups.
- *Self-RAG* (Asai et al., ICLR 2024) and *retrieval-robust training* (Yoran et al., ICLR 2024): train reflection/critique tokens or NLI-filtered irrelevance robustness. Established for irrelevant context; **claimed but unablated** for *plausible contradictory* context.
- *PH3* (Jin et al., Findings ACL 2024): prunes attention heads identified as memory-carrying, shifting the balance toward context. Mechanistic, but the head set is model-specific and the reported gains exist only as benchmark numbers on constructed conflict sets.

**Theory.** Essentially none specific to this problem. There is no identifiability result, no regret bound for a source-selection policy, and no lower bound on ACR under an adversary who may inject $m$ of $k$ passages.

**Benchmarks.** KC (Longpre et al., 2021), ConflictQA/Xie et al. (2024), ConflictingQA (Wan et al., 2024), ConflictBank (Su et al., 2024), FaithEval (Ming et al., 2025), Kortukov et al.'s real-document suite (2024). Most report CRR or context-following, not ACR.

## 4. What Is Known

- **Entity substitution flips answers unreliably.** Longpre et al. (EMNLP 2021), on NQ with T5-scale and DPR-based readers: models frequently retain the original memorized entity rather than the substituted one, and retrieval-heavy training raises context reliance. The measured quantity is a memorization ratio, not accuracy.
- **Coherence and volume beat truth.** Xie et al. (ICLR 2024), GPT-3.5/GPT-4/PaLM2 scale: models are highly receptive to a single coherent counter-memory passage, but exhibit strong confirmation bias when memory-consistent and memory-inconsistent evidence appear together; increasing the *number* of supporting passages shifts the answer regardless of source quality.
- **Popularity gates the useful direction.** Mallen et al. (ACL 2023), PopQA, GPT-Neo through GPT-3 scale: retrieval helps on low-popularity entities and *hurts* on high-popularity ones, so a fixed always-context policy is provably suboptimal on the mixture.
- **Surface features drive persuasion.** Wan et al. (ACL 2024), ConflictingQA: models are moved far more by textual relevance than by source authority, publication venue, or presence of scientific references — the features a human adjudicator would weight most.
- **Priors are measurable but unstable.** Du et al. (ACL 2024) formalize persuasion and susceptibility scores and find susceptibility varies systematically with entity frequency and prompt form.
- **Real updates behave differently from synthetic ones.** Kortukov et al. (COLM 2024): with genuine updated documents, models update far more readily than synthetic-counterfactual results suggest; the dominant failure is reversion to a *parametric* answer that was never in the context.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed operationalization of "correct resolution". Nearly all headline numbers are CRR, which a degenerate policy maximizes. $\mathcal{D}^{-}$ (memory-correct, context-corrupted) is under-built, so ACR is not reportable on most benchmarks. Also blocked: a stable estimator of $a_\theta(q)$ under paraphrase.
- **Empirically open.** Whether any current method raises ACR above 0.5 by a meaningful margin under a matched adversary; whether the effect scales with model size; whether long-context (100k+) retrieval changes the conflict calculus relative to $k \le 10$ passages. All runnable today at ≤70B scale for well under $50k of compute — nobody has published the balanced two-arm result.
- **Theoretically open.** No identifiability characterization: given only $(q, C, \theta)$ with no provenance channel, under what conditions is the correct source recoverable? No lower bound on achievable ACR as a function of the fraction of corrupted passages. No proof that CAD-style fixed-$\alpha$ reweighting cannot be Pareto-optimal on both arms.

## 6. Why It Is Hard

Two obstructions, both concrete.

**Confounded measurement.** Context reliance and correctness are conflated. A method that raises context-following raises the score on every benchmark built from counterfactual substitution, *because those benchmarks define the substituted answer as gold*. The same method necessarily lowers accuracy on corrupted-context inputs, which those benchmarks do not contain. So the field's primary metric rewards exactly the behavior that makes RAG poisoning work.

**Non-identifiability.** For a query with a genuine post-training update, the correct context passage and an adversarially fabricated passage can be word-for-word equally plausible: same style, same entity types, same citation form. Wan et al. (2024) show models do not use the features that would break the tie, and it is unclear those features are present in text at all. Absent a provenance or timestamp channel, part of the residual error is information-theoretically unremovable — but nobody has quantified that floor, so we cannot tell how much of the current gap is method failure.

## 7. Current Research (as of 2026)

- **Balanced conflict benchmarks.** FaithEval (Salesforce AI Research) adds unanswerable and inconsistent-context arms; ConflictBank separates misinformation, temporal, and semantic conflict causes. Movement toward two-arm evaluation is visible but ACR-style balanced reporting is still rare *(frontier — verify)*.
- **Mechanistic localization.** Competition-of-mechanisms work (Ortu et al., ACL 2024) and head-level pruning (Jin et al., 2024) treat conflict as a competition between attention circuits; the open question is whether the identified components transfer across models.
- **Provenance-conditioned generation.** Attaching timestamps, source reliability priors, and retrieval-corpus metadata to passages, then training the model to condition on them. Early and mostly proprietary *(frontier — verify)*.
- **Abstention and conflict-reporting.** Wang et al. (COLM 2024) argue for detect-then-report-then-resolve pipelines rather than single-answer output; adoption in production RAG remains limited.

## 8. Concrete Next Experiment

**Question:** does any current method beat 0.5 ACR?

**Scale.** 3 open models spanning size (8B, 70B, and one ~400B-class open-weights model), 2,000 queries drawn from PopQA and NQ, split 1,000 / 1,000:
- $\mathcal{D}^{+}$: queries where the model's parametric answer is *wrong* and a retrieved passage carries the correct answer (verified by human or strong-judge check).
- $\mathcal{D}^{-}$: queries where the parametric answer is *right* and one injected passage carries a fluent, type-consistent false answer, style-matched to the true passages.

**Arms.** (1) always-context (greedy RAG); (2) always-memory (closed book); (3) CAD at $\alpha \in \{0.5, 1.0\}$; (4) Self-RAG-style critique; (5) a provenance oracle that reveals which passage is injected — this upper-bounds what any text-only method could reach.

**Deciding number.** Balanced ACR. Baselines 1–2 sit at 0.5 by construction. A method is a real advance only if $\text{ACR} > 0.60$ with a 95% bootstrap CI excluding 0.55. The gap between the best text-only arm and the oracle arm quantifies the identifiability floor from §6.

Cost: roughly 30k generations per arm-model pair; single-node inference, days not weeks.

## 9. Key References

- **[Foundational]** S. Longpre, K. Perisetla, A. Chen, N. Ramesh, C. DuBois, S. Singh. *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP, 2021. — arXiv:2109.05052
- **[Foundational]** A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, H. Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL, 2023. — arXiv:2212.10511
- **[SOTA]** W. Shi, X. Han, M. Lewis, Y. Tsvetkov, L. Zettlemoyer, S. W. Yih. *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding.* NAACL, 2024. — arXiv:2305.14739
- **[SOTA]** J. Xie, K. Zhang, J. Chen, R. Lou, Y. Su. *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.* ICLR, 2024. — arXiv:2305.13300
- **[SOTA]** A. Asai, Z. Wu, Y. Wang, A. Sil, H. Hajishirzi. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR, 2024. — arXiv:2310.11511
- **[Method]** O. Yoran, T. Wolfson, O. Ram, J. Berant. *Making Retrieval-Augmented Language Models Robust to Irrelevant Context.* ICLR, 2024. — arXiv:2310.01558
- **[Method]** A. Neeman, R. Aharoni, O. Honovich, L. Choshen, I. Szpektor, O. Abend. *DisentQA: Disentangling Parametric and Contextual Knowledge with Counterfactual Question Answering.* ACL, 2023. — arXiv:2211.05655
- **[Analysis]** A. Wan, E. Wallace, D. Klein. *What Evidence Do Language Models Find Convincing?* ACL, 2024. — arXiv:2402.11782
- **[Analysis]** K. Du, V. Hernández-Cerezo, et al. *Context versus Prior Knowledge in Language Models.* ACL, 2024.
- **[Analysis]** E. Kortukov, A. Rubinstein, E. Nguyen, S. J. Oh. *Studying Large Language Model Behaviors Under Context-Memory Conflicts With Real Documents.* COLM, 2024. — arXiv:2404.16032
- **[Benchmark]** H. Ming, et al. *FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon Is Made of Marshmallows".* ICLR, 2025. — arXiv:2410.03727
- **[Survey]** R. Xu, Z. Qi, Z. Guo, C. Wang, H. Wang, Y. Zhang, W. Xu. *Knowledge Conflicts for LLMs: A Survey.* EMNLP, 2024. — arXiv:2403.08319

## 10. Worked Example

Query: *"Who is the CEO of Twitter?"* Suppose the model's parametric answer is `Jack Dorsey` (pre-2021 corpus). Retrieval returns $k=5$ passages; one, $c_3$, says the CEO is `Linda Yaccarino`.

**Arm $\mathcal{D}^{+}$.** $c_3$ is true. Context-following is correct. CAD with $\alpha=1$ amplifies $\log p(\cdot\mid C) - \log p(\cdot \mid \emptyset)$, and since `Yaccarino` has near-zero parametric mass, its amplified score dominates. Correct.

**Arm $\mathcal{D}^{-}$.** Now inject $c_3'$: *"In a filing dated March 2024, X Corp named **Marcus Feld** interim chief executive."* Same length, same register, same date form. `Feld` is a fabricated name with near-zero parametric mass — so CAD amplifies it *by exactly the same mechanism*. The contrastive term cannot distinguish "new true fact" from "new false fact"; both are low-prior tokens made salient by context.

Numerically: suppose the model assigns $p(\text{Dorsey}\mid q)=0.6$, $p(\text{Feld}\mid q)=10^{-6}$, and with context $p(\text{Feld}\mid q,C)=0.4$, $p(\text{Dorsey}\mid q,C)=0.3$. CAD scores are
$$2\log 0.4 - \log 10^{-6} = -1.83 + 13.8 = 11.97 \quad\text{vs}\quad 2\log 0.3 - \log 0.6 = -2.41 + 0.51 = -1.90.$$
The fabrication wins by ~14 nats. The same $\alpha$ that fixes arm $\mathcal{D}^{+}$ breaks arm $\mathcal{D}^{-}$, and a benchmark reporting only CRR scores this a success. That is the obstruction: the metric and the method share a blind spot, so improvement on the published number is not evidence of improvement on the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*