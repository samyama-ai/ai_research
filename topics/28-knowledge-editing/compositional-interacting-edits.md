---
id: 28-knowledge-editing/compositional-interacting-edits
title: "Compositional Edits That Must Interact"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compositional Edits That Must Interact

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/compositional-interacting-edits` · **Status:** open

## 1. Problem Statement

Batch knowledge editing evaluates edits as if they were independent: apply $n$ facts, check each one back individually, check that unrelated facts moved as little as possible. Real updates are not independent. Two edits can **compose** ("the CEO of X is now Y" and "Y was born in Z" jointly entail "the CEO of X was born in Z"), **conflict** ("X's capital is Y" then "X's capital is W"), or **interfere** through shared subject/relation representations without any logical relation at all.

The problem: given an edit set whose elements must interact, produce a post-edit model whose answers to *derived* queries match the answers of a model that had held the edited facts all along.

Three variants, different difficulty:

- **Measurement.** Define a compositional success criterion that is not gameable by memorising the query template, and that separates "the edit did not propagate" from "the base model could not do the reasoning anyway."
- **Method.** Build an editor whose joint accuracy on derived queries does not collapse as the number of interacting edits grows.
- **Theory.** Decide whether a fixed-rank, closed-form parameter update to a feed-forward key–value memory can represent the transitive closure of an edit set at all, or whether propagation requires inference-time computation.

## 2. Formal Setting

Let $f_\theta$ be an autoregressive LM. An atomic edit is a triple $e=(s,r,o^*)$ replacing $o$ in $(s,r,o)$. An edit batch is $E=\{e_1,\dots,e_n\}$.

**Derived queries.** Fix a rule set $\mathcal{R}$ (composition, inversion, aliasing, two-hop). The *closure* $\mathrm{cl}(K \setminus K_E \cup E)$ under $\mathcal{R}$ over the base knowledge base $K$ gives the intended post-edit answers. A derived query $q$ is one whose closure answer $a^*(q)$ depends on $\ge 2$ edits.

**Measured quantities** (all as exact-match or top-1 argmax over a fixed answer vocabulary, greedy decoding, one prompt template family with $m\ge 3$ paraphrases):

$$\mathrm{ES}(E)=\frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\!\left[\arg\max_o f_{\theta'}(o\mid s_i,r_i)=o_i^*\right]$$

$$\mathrm{CS}_k(E)=\mathbb{E}_{q \in Q_k}\ \mathbb{1}\!\left[\hat a_{\theta'}(q)=a^*(q)\right],\quad Q_k=\{q: |\mathrm{dep}(q)|=k\}$$

where $\mathrm{dep}(q)$ is the set of edits the closure answer depends on. The central statistic is the **interaction gap**

$$\Delta_k(E)=\prod_{i\in \mathrm{dep}(q)}\mathrm{ES}(e_i)\Big|_{\text{avg over }Q_k}-\ \mathrm{CS}_k(E),$$

the shortfall of joint compositional accuracy relative to the independence prediction from marginal edit success. $\Delta_k>0$ means edits fail to compose; $\Delta_k<0$ means the benchmark is solvable without the edits (leakage).

**Base-competence control.** $\mathrm{CS}_k$ is uninterpretable without $\mathrm{CS}_k^{\mathrm{oracle}}$: the same queries answered by the *unedited* model with the edited facts supplied in context. Report $\mathrm{CS}_k/\mathrm{CS}_k^{\mathrm{oracle}}$.

**Assumptions, and which break.**
1. *A unique closure answer exists.* Violated when edits conflict; closure under an inconsistent set is trivial. Conflict cases need a separate order-dependent semantics.
2. *Edits are logically independent unless $\mathcal{R}$ says so.* Violated: MEMIT-style updates share the same $W_{\text{proj}}$ null space, so unrelated edits interfere numerically (Gupta et al., 2024).
3. *Exact match is a faithful readout.* Violated for aliased entities and for models that hedge; requires alias sets.
4. *$K$ is known.* Violated — the base model's actual belief set is not enumerable, so "was $a^*(q)$ already known?" is estimated, not measured.

## 3. State of the Art

**Established (independently reproduced).**
- **ROME** (Meng et al., NeurIPS 2022) and **MEMIT** (Meng et al., ICLR 2023): rank-one / spread-out closed-form updates to mid-layer MLPs. MEMIT scales to 10,000 edits on GPT-J (6B) and GPT-NeoX (20B) with high per-edit efficacy. Reproduced widely.
- **Failure to propagate.** MQuAKE (Zhong et al., EMNLP 2023) shows parameter editors that score high on single-hop recall collapse on multi-hop questions built from the same edits. RippleEdits (Cohen et al., TACL 2024) reports the same across six ripple criteria including two compositionality axes.
- **Sequential collapse.** Gupta et al. (ACL Findings 2024; EMNLP 2024) show ROME-style sequential editing produces gradual forgetting and, past a threshold, "disabling edits" that break the model outright.

**Claimed but unablated.**
- Null-space-projected editing (**AlphaEdit**, Fang et al., ICLR 2025) improves sequential-edit retention; its effect specifically on *interacting* edits is reported as benchmark deltas, not as an ablation of the interaction gap $\Delta_k$.
- Memory/retrieval editors (**MeLLo**, Zhong et al. 2023; **GRACE**, Hartvigsen et al., NeurIPS 2023; **WISE**, Wang et al., NeurIPS 2024) score far better on multi-hop benchmarks, but they move the composition into the prompt or a retrieval step — this is a change of problem, not a solution to parameter-space composition.

**Benchmark-number-only.** Nearly all compositional results exist as leaderboard rows on MQuAKE, RippleEdits, ConflictEdit/RoundEdit (Li et al., ICLR 2024). MQuAKE in particular has a known shortcut: Yang et al. (2024) showed retrieval-style methods can score highly by exploiting question structure, so raw MQuAKE numbers are not evidence of propagation.

## 4. What Is Known

- MEMIT on GPT-J-6B: >90% single-edit efficacy at batch sizes up to $10^4$; multi-hop accuracy on MQuAKE-CF for the same edited models is in the single-digit-to-low-teens percent range, against >80% for in-context/memory-based MeLLo at comparable scale (Zhong et al., 2023).
- RippleEdits (5K edits, GPT-2 XL / GPT-J / LLaMA-family): ROME, MEMIT and MEND all show large drops from subject-aliasing accuracy to compositionality accuracy; a simple in-context-editing baseline beats every parameter editor on the ripple axes (Cohen et al., 2024).
- ConflictEdit (Li et al., ICLR 2024): when two logically dependent edits are applied in sequence, editors frequently retain the *superseded* fact — the measured "conflict magnitude" is nonzero for ROME/MEMIT/MEND at GPT-2 XL and GPT-J scale, and "knowledge distortion" (edited facts' distributions bleeding into neighbours) rises with round-edit count.
- Specificity is fragile: Hoelscher-Obermaier et al. (ACL Findings 2023) show CounterFact's specificity metric overstates locality; a harder prompt set exposes large unrelated-fact drift for ROME/MEMIT at GPT-2 XL scale.
- Localization does not predict editability: Hase et al. (NeurIPS 2023) show causal-tracing layer attribution is uncorrelated with which layer edits best — so "edit where the fact lives" is not a principle you can compose over.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted way to separate "the edit failed to propagate" from "the model never had the reasoning ability." Almost no paper reports the in-context oracle control $\mathrm{CS}_k^{\mathrm{oracle}}$ per query, so published compositional numbers conflate editor failure with base incompetence. Until that ratio is standard, the interaction gap $\Delta_k$ is not measured anywhere.
- **Empirically open.** Whether $\Delta_k$ shrinks with model scale. All the collapse evidence is at 1.5B–20B. Nobody has run a controlled interacting-edit suite across a scale ladder (1B → 70B+) with fixed edit sets and fixed prompts.
- **Empirically open.** Whether propagation failure is an *editor* property or a *representation* property: does a model finetuned on the edited facts (not edited) compose them, at matched single-hop accuracy?
- **Theoretically open.** No result characterises which closures a rank-$r$ linear associative-memory update can realise. Composition of two edits requires the model to chain two retrievals at inference; whether any static weight perturbation of bounded rank can install a chained association that was not already latent is unproven either way.

## 6. Why It Is Hard

**Confounded measurement is the primary obstruction.** $\mathrm{CS}_k$ mixes four causes — edit didn't take, edit took but didn't propagate, model can't do $k$-hop reasoning, benchmark leaks the answer — and the standard benchmarks report one number. MQuAKE's shortcut vulnerability (Yang et al., 2024) is the concrete demonstration: a method can move the headline number without touching the phenomenon.

**Second: absent ground truth for the closure.** $\mathrm{cl}(\cdot)$ requires knowing $K$, the model's actual belief set. It is approximated by a KB (Wikidata) that the model neither exactly holds nor exactly lacks. Every derived-query label inherits that mismatch.

**Third: non-identifiability of interference.** Two edits sharing an MLP key subspace interfere for numerical reasons indistinguishable, from the outside, from logical conflict. Nothing in current evaluation separates them.

## 7. Current Research (as of 2026)

- **Null-space and projection-constrained editing** to make batch edits non-interfering by construction (AlphaEdit line, ICLR 2025; follow-ups extending the projection to sequential regimes) — *(frontier — verify)* whether the constraint helps composition or only preservation.
- **Memory- and adapter-based editors** (GRACE, WISE) plus retrieval-augmented editing, which sidestep parameter composition; the open question is whether the memory itself needs to be closed under the same rules.
- **Composable interventions**: Kolbeinsson et al. (ICLR 2025) study composing editing with compression and unlearning and find order effects — the same framing applied to edit-edit composition is the natural extension.
- **Benchmark repair**: harder specificity sets, leakage audits of MQuAKE, and conflict/distortion suites from the ZJU/KnowEdit group (Zhang, Li et al.).

## 8. Concrete Next Experiment

**Question.** Is the compositional failure of parameter editors an editor artifact or a base-model limit?

**Scale.** Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct. Build 1,000 two-edit chains from Wikidata: $(s,r_1,o^*)$ and $(o^*,r_2,z^*)$, where $z^*$ is derived and never stated. Filter so the base model answers neither hop and the composed query correctly at $<5\%$. Editors: MEMIT, AlphaEdit, GRACE, and in-context editing.

**Control arms (both required).**
1. *Oracle-context arm*: unedited model, both facts in the prompt, same composed question. This is $\mathrm{CS}_2^{\mathrm{oracle}}$ — the ceiling set by reasoning ability.
2. *Single-edit arm*: apply edit 1, supply fact 2 in context (and vice versa). Isolates which of the two positions in the chain fails to be retrieved from weights.

**Deciding number.** The normalised gap $1-\mathrm{CS}_2/\mathrm{CS}_2^{\mathrm{oracle}}$ for MEMIT at 8B, with three paraphrases per query and a leakage screen. If it is $<0.15$, propagation is largely a reasoning-ceiling artifact and the field's framing is wrong. If it is $>0.5$ — the outcome the MQuAKE/RippleEdits evidence predicts — parameter editing genuinely does not install composable facts, and the single-edit arm says whether the failure sits at hop 1 or hop 2.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[SOTA / benchmark]** Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, Danqi Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP, 2023. — arXiv:2305.14795
- **[SOTA / benchmark]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[SOTA]** Zhoubo Li, Ningyu Zhang, Yunzhi Yao, Mengru Wang, Xi Chen, Huajun Chen. *Unveiling the Pitfalls of Knowledge Editing for Large Language Models.* ICLR, 2024. — arXiv:2310.02129
- **[SOTA]** Junfeng Fang et al. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR, 2025. — arXiv:2410.02355
- **[Analysis]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[Analysis]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024. — arXiv:2401.07453
- **[Analysis]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL, 2023. — arXiv:2305.17553
- **[Related]** Arinbjörn Kolbeinsson et al. *Composable Interventions for Language Models.* ICLR, 2025.
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023. — arXiv:2305.13172

## 10. Worked Example

Two edits on GPT-J-6B, MEMIT, applied as one batch:

- $e_1$: (Tim Cook, CEO of, **Nintendo**) — replacing Apple.
- $e_2$: (Nintendo, headquartered in, **Reykjavík**) — replacing Kyoto.

Post-edit, both single-hop probes succeed: "Tim Cook is the CEO of →" gives *Nintendo*; "Nintendo is headquartered in →" gives *Reykjavík*. So $\mathrm{ES}(e_1)=\mathrm{ES}(e_2)=1$, and the independence prediction for the composed query is $1\times1=1$.

The composed probe, "The company Tim Cook runs is headquartered in the city of →", returns *Cupertino*. This is the diagnostic: not *Kyoto* (which would mean hop 1 propagated and hop 2 did not), but *Cupertino* — the pre-edit answer to the **composed** query. The rank-one association was installed at the surface form (Tim Cook, CEO-of) and the surface form (Nintendo, HQ), while the composed path retains an independent, unedited route from *Tim Cook* to *Cupertino* that neither edit touched.

Interaction gap for this instance: $\Delta_2 = 1 - 0 = 1$.

Now the obstruction. Suppose instead the answer had been a refusal or a wrong city. Was the failure the editor's, or can this model do two-hop composition at all? Run the oracle control — same unedited model, both facts in context — and on a filtered set of two-hop Wikidata chains a 6B-class model answers correctly well under half the time. So a measured $\mathrm{CS}_2 = 0.05$ against an unmeasured $\mathrm{CS}_2^{\mathrm{oracle}}$ that might be $0.35$ supports a completely different conclusion than the same $0.05$ against an oracle of $0.95$. Published compositional editing numbers almost never report the denominator. That missing denominator, not the editing algorithm, is what currently blocks the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*