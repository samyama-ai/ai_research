---
id: 28-knowledge-editing/bidirectional-consistency-after-editing
title: "Bidirectional Consistency After Editing"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Bidirectional Consistency After Editing

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/bidirectional-consistency-after-editing` · **Status:** open

## 1. Problem Statement

A knowledge edit installs a new fact in the forward direction: after editing, the model completes "The Eiffel Tower is located in ___" with "Rome". The question is whether the inverse query moves with it — "Which famous tower is located in Rome?" or "Rome is the location of ___".

Three variants, of very different difficulty:

- **Measurement.** Define bidirectional consistency so that a score is not gameable by prompt format, by the model's prior over entity names, or by relation cardinality. A one-to-many relation (`country → cities`) has no single correct inverse answer, so "the reverse must return $o_{new}$" is the wrong predicate for most triples.
- **Method.** Given an editor $E$ and a triple $(s, r, o_{new})$, produce parameters such that both $p(o_{new} \mid s, r)$ and the inverse query $p(s \mid o_{new}, r^{-1})$ reflect the edit, without degrading unrelated facts.
- **Theory.** Establish whether a *local, rank-limited* parameter edit at a single MLP layer can in principle be bidirectionally consistent, given that pretraining itself does not produce reverse generalization (the reversal curse). If forward and reverse facts are stored in disjoint circuits, no localized forward edit can reach the reverse.

Solving it means: an editor that raises reverse accuracy to within a stated margin of forward efficacy on a benchmark where the reverse predicate is well posed, without loss on locality/specificity, at $\geq 7$B scale.

## 2. Formal Setting

Let $f_\theta$ be an autoregressive LM. A fact is a triple $(s, r, o)$ with a forward prompt template $\pi_r(s)$ and an inverse template $\pi_{r^{-1}}(o)$. An edit request is $(s, r, o_{old} \to o_{new})$; an editor produces $\theta' = E(\theta, s, r, o_{new})$.

**Forward efficacy** — measured as teacher-forced argmax over a candidate set $C$, not free generation:
$$\mathrm{Eff} = \mathbb{E}\big[\mathbb{1}\{\, p_{\theta'}(o_{new} \mid \pi_r(s)) > p_{\theta'}(o_{old} \mid \pi_r(s)) \,\}\big]$$

**Reverse efficacy** for a relation whose inverse is functional ($|\{s : (s,r,o)\}| = 1$):
$$\mathrm{Rev} = \mathbb{E}\big[\mathbb{1}\{\, p_{\theta'}(s \mid \pi_{r^{-1}}(o_{new})) > p_{\theta'}(s' \mid \pi_{r^{-1}}(o_{new})) \ \forall s' \in C \setminus \{s\} \,\}\big]$$

For non-functional inverses the predicate must be *set membership*: $s \in \mathrm{top}_k$ of the inverse distribution, with $k$ fixed to the true fan-out. Reporting a single "reverse accuracy" over mixed-cardinality relations conflates two different questions; this is the main measurement defect in the literature.

**Consistency gap**, the quantity this page is about:
$$\Delta_{\leftrightarrow} = \mathrm{Eff} - \mathrm{Rev} \in [-1, 1]$$
An editor is bidirectionally consistent when $\Delta_{\leftrightarrow} \approx 0$ *and* $\mathrm{Eff}$ is high — $\Delta_{\leftrightarrow}=0$ at $\mathrm{Eff}=\mathrm{Rev}=0$ is trivially achieved by doing nothing.

**Locality** must be measured on the inverse side too:
$$\mathrm{Loc}_{rev} = \mathbb{E}_{(o'', s'') \notin \text{edit}}\big[\mathbb{1}\{\arg\max p_{\theta'}(\cdot \mid \pi_{r^{-1}}(o'')) = \arg\max p_{\theta}(\cdot \mid \pi_{r^{-1}}(o''))\}\big]$$
because an editor can raise $\mathrm{Rev}$ by making $s$ the answer to *every* inverse query of relation $r^{-1}$.

**Assumptions, with the violated ones flagged:**
1. *A fact has a canonical surface form.* Violated — entity aliasing ("Paris" / "the French capital") changes scores by tens of points.
2. *Inverse relations are functional.* Violated for the majority of Wikidata relations.
3. *The pre-edit model knows the reverse fact.* Often violated — if $p_\theta(s \mid \pi_{r^{-1}}(o_{old}))$ was already wrong, $\mathrm{Rev}$ measures pretraining coverage, not the editor. A pre-edit reverse-competence filter is mandatory and frequently absent.
4. *Edits are independent.* Violated at batch scale; sequential edits degrade each other.

## 3. State of the Art

**Empirical SOTA (established).** Locate-then-edit methods — ROME (Meng et al., NeurIPS 2022) and MEMIT (Meng et al., ICLR 2023) — reach forward efficacy above 95% on CounterFact at GPT-2 XL (1.5B) and GPT-J (6B) scale. This is reproduced widely, including in the EasyEdit/KnowEdit framework (Zhang et al., 2024). Hypernetwork editors MEND (Mitchell et al., ICLR 2022) and memory-based SERAC (Mitchell et al., ICML 2022) are the main alternatives; in-context editing (IKE) is a strong non-parametric baseline.

**Bidirectional SOTA (claimed, largely unablated).** BAKE (Ma et al., *Untying the Reversal Curse via Bidirectional Language Model Editing*, 2023) is the only benchmark built specifically for this predicate. It reports near-total failure of ROME/MEMIT/MEND on reverse queries despite high forward efficacy, and proposes BIRD, which optimizes the edited weight so that the object representation also predicts the subject. BIRD's reported gains exist as benchmark numbers on GPT-2 XL and GPT-J only; there is no independent replication at $\geq 7$B, and no ablation separating "the reverse fact was installed" from "the subject token's unconditional probability rose".

**Adjacent SOTA.** RippleEdits (Cohen et al., TACL 2024) measures ripple effects including *Logical Generalization*, which subsumes inversion; it is the broader, better-controlled benchmark but does not isolate cardinality. MQuAKE (Zhong et al., EMNLP 2023) shows multi-hop failure, a superset symptom. Reverse Training (Golovneva et al., COLM 2024) fixes reversal at *pretraining* time by training on token-reversed text — not an editing method, but the strongest existence proof that the deficit is fixable in principle.

## 4. What Is Known

- **The pretraining baseline is near zero.** Berglund et al. (ICLR 2024) finetuned models up to GPT-3 175B on synthetic "A is B" facts: forward accuracy near 100%, reverse accuracy ~0%, with reverse log-probability indistinguishable from a random control name. The effect held for Llama-1 7B/13B/30B under repeated finetuning.
- **Editing does not repair it.** Every bidirectional evaluation published to date reports a large positive $\Delta_{\leftrightarrow}$ for ROME/MEMIT at 1.5B–6B scale: forward efficacy $>95\%$, reverse accuracy in the low tens of percent or worse. Direction and magnitude are consistent across BAKE and RippleEdits; exact values differ by template and filtering.
- **Ripple failure is general.** Cohen et al. report that even the best editors capture well under half of the entailed consequences of an edit on GPT-2 XL, GPT-J and Llama-2 7B, while efficacy remains near ceiling — i.e. the gap is not specific to inversion.
- **Edits do not compose.** MQuAKE shows accuracy on 2–4 hop questions collapses after edits that individually succeed, at GPT-J 6B scale.
- **Sequential editing degrades the model.** Gupta et al. (ACL Findings 2024) show gradual then catastrophic forgetting as edit count grows into the thousands, which bounds how much any bidirectional repair can be applied in bulk.
- **Reverse generalization is achievable by data.** Reverse Training (Golovneva et al., 2024) substantially closes the reversal gap at 1.4B–7B pretraining scale, at the cost of a modified pretraining run.

## 5. What Is Not Known

- **Theoretically open.** Whether a rank-one or rank-$k$ update to a single MLP down-projection can, in general, induce the inverse association — i.e. whether forward and reverse retrieval share sufficient parameter support for a localized edit to reach both. There is no separation theorem and no impossibility proof. Related: no identifiability result saying which parameters *are* the fact.
- **Empirically open.** Whether $\Delta_{\leftrightarrow}$ shrinks with scale. All bidirectional editing numbers are at $\leq 7$B. The experiment at 70B and beyond is runnable today with MEMIT and off-the-shelf checkpoints; nobody has published it with pre-edit reverse-competence filtering.
- **Empirically open.** Whether editing *both* directions explicitly (two edits per fact) is stable, or whether the second edit erases the first — a two-line change to any editing pipeline, unreported at scale.
- **Methodologically blocked.** A cardinality-aware reverse predicate. Until $\mathrm{Rev}$ is defined separately for functional and non-functional inverses, aggregate reverse accuracy is not comparable across benchmarks, and "BIRD beats ROME by $X$ points" is not a well-typed claim.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth for the inverse**.

- Reverse accuracy mixes three causes: the editor failed; the model never knew the reverse fact pre-edit; the inverse relation has no unique answer. Published scores rarely separate them, so a low $\mathrm{Rev}$ is uninformative about the editor.
- Optimizing $\mathrm{Rev}$ is trivially gameable. Raising $p(s \mid \cdot)$ globally scores well on $\mathrm{Rev}$ and is caught only by $\mathrm{Loc}_{rev}$, which most papers do not report.
- Non-identifiability: causal tracing localizes *where information flows*, not *where a fact is stored*. Hase et al. (NeurIPS 2023) showed edit success is largely uncorrelated with the traced layer, so "edit the site the trace found, in both directions" has no principled target on the reverse side.

Compute is not the obstruction. A full ROME/MEMIT bidirectional sweep at 7B is single-GPU-days.

## 7. Current Research (as of 2026)

- **Cardinality-aware ripple benchmarks** extending RippleEdits with typed inverse predicates *(frontier — verify)*.
- **Bidirectional objectives inside the editor** — BIRD-style symmetric losses, and null-space-constrained editors (AlphaEdit and successors) applied to reverse queries *(frontier — verify)*.
- **Retrieval/context editing as the pragmatic fix.** IKE-style and SERAC-style non-parametric editing sidesteps the parameter-locality problem: the edited fact is in context, so both directions can be attended. The open question is whether this holds under multi-fact interference.
- **Mechanistic work on relation representation** — Hernandez et al. (ICLR 2024) show many relations are approximately linear maps on subject representations; whether the inverse map is recoverable from the same weights is the theory-side entry point (Northeastern / David Bau's group and collaborators).
- Groups active on the editing side: Zhejiang University NLP (EasyEdit/KnowEdit), USTC (BAKE/BIRD), Northeastern (ROME/MEMIT lineage), Tel Aviv (RippleEdits).

## 8. Concrete Next Experiment

**Question.** Does the bidirectional consistency gap $\Delta_{\leftrightarrow}$ shrink with model scale, or is it scale-invariant?

**Scale.** Llama-3.1 8B and 70B (two points), 1,000 edit requests drawn only from *functionally invertible* Wikidata relations (`P26 spouse`, `P112 founded-by`, `P50 author`), filtered so that the pre-edit model answers *both* directions of the original fact correctly under 5 paraphrase templates. Editors: MEMIT, ROME, and IKE (in-context).

**Control arms.** (a) *Unedited model* on the same reverse queries with $o_{new}$ substituted — measures how much reverse "success" is prompt-driven guessing. (b) *Random-subject control*: measure $p_{\theta'}(s \mid \pi_{r^{-1}}(o''))$ for 1,000 unrelated $o''$, to detect global subject-probability inflation. (c) *Double-edit arm*: apply the forward edit and an explicit reverse edit, and re-measure forward efficacy.

**Deciding number.** $\Delta_{\leftrightarrow}(70\text{B}) - \Delta_{\leftrightarrow}(8\text{B})$, reported with bootstrap CIs, subject to $\mathrm{Loc}_{rev} \geq 0.95$ in both. If the difference is $< 5$ points, the gap is scale-invariant and the problem is architectural, not a data-coverage artifact — which promotes the theory variant. If it is $> 15$ points, scale is a partial fix and the priority moves to characterizing the curve.

Cost estimate: ~200 A100-hours including the 70B MEMIT covariance pass.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[Foundational]** Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, Owain Evans. *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A".* ICLR 2024. — arXiv:2309.12288
- **[SOTA]** Jun-Yu Ma, Jia-Chen Gu, Zhen-Hua Ling, Quan Liu, Cong Liu. *Untying the Reversal Curse via Bidirectional Language Model Editing.* Preprint, 2023. (BAKE benchmark and BIRD method.)
- **[SOTA]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[SOTA]** Olga Golovneva, Zeyuan Allen-Zhu, Jason Weston, Sainbayar Sukhbaatar. *Reverse Training to Nurse the Reversal Curse.* COLM 2024. — arXiv:2403.13799
- **[Method]** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning. *Fast Model Editing at Scale.* ICLR 2022. — arXiv:2110.11309
- **[Method]** Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn. *Memory-Based Model Editing at Scale.* ICML 2022. — arXiv:2206.06520
- **[Analysis]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Analysis]** Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, Danqi Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP 2023. — arXiv:2305.14795
- **[Analysis]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024.
- **[Analysis]** Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, David Bau. *Linearity of Relation Decoding in Transformer Language Models.* ICLR 2024. — arXiv:2308.09124
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172
- **[Survey/Benchmark]** Ningyu Zhang et al. *A Comprehensive Study of Knowledge Editing for Large Language Models.* Preprint, 2024. (KnowEdit benchmark, EasyEdit toolkit.) — arXiv:2401.01286

## 10. Worked Example

Take the triple $(\text{Lionel Messi}, \text{plays-for}, \text{Inter Miami})$ and edit it to $o_{new} = \text{Real Madrid}$ with MEMIT on an 8B model.

**Forward.** $\pi_r(s) =$ "Lionel Messi plays for". Post-edit $p(\text{Real Madrid}) \approx 0.87$ vs $p(\text{Inter Miami}) \approx 0.01$. $\mathrm{Eff} = 1$.

**Reverse.** $\pi_{r^{-1}}(o_{new}) =$ "A famous player who plays for Real Madrid is". The top prediction is whatever the model's prior favours — Vinícius, Mbappé, Bellingham. $p(\text{Messi})$ moves from $0.004$ pre-edit to $0.011$ post-edit: a 2.8× relative rise, and still rank ~9. Under a strict argmax predicate, $\mathrm{Rev} = 0$ and $\Delta_{\leftrightarrow} = 1$.

**Where the obstruction becomes visible.** That verdict is not interpretable, for three separate reasons at once:

1. `plays-for` has fan-out ~25. The correct predicate is $s \in \mathrm{top}_{25}$, under which rank 9 *passes* and $\mathrm{Rev} = 1$, $\Delta_{\leftrightarrow} = 0$. The same edit scores 0 or 1 depending on a predicate choice that no paper standardizes. Two published bidirectional results can therefore disagree by 100 points on identical model behaviour.
2. The pre-edit check is the one that matters. Ask "A famous player who plays for Inter Miami is" *before* the edit: if the base model does not put Messi first, then the post-edit reverse failure was never the editor's doing, and reporting it as such attributes a pretraining gap to MEMIT.
3. The 2.8× rise is the gameable part. Measure $p(\text{Messi} \mid$ "A famous player who plays for Bayern Munich is"$)$ post-edit. If that also rose ~2–3×, the editor inflated the token "Messi" globally rather than installing an inverse association — and any headline $\mathrm{Rev}$ improvement built on such rises is an artifact. This control ($\mathrm{Loc}_{rev}$) is cheap, is the single most diagnostic measurement on the page, and is missing from nearly every reported bidirectional result.

The obstruction, concretely: with one triple and one 8B model, the answer ranges over $\{0, 1\}$ for $\mathrm{Rev}$ and the deciding factor is measurement convention, not the editor.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*