---
id: 28-knowledge-editing/compositional-editing-relational-schemas
title: "Compositional Editing of Relational Schemas"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compositional Editing of Relational Schemas

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/compositional-editing-relational-schemas` · **Status:** open

## 1. Problem Statement

A knowledge edit is usually specified as a single triple rewrite: $(s, r, o) \to (s, r, o^*)$. Real knowledge is *relational and schema-constrained*. Changing "Chile's capital is Santiago" to "Chile's capital is Valparaíso" entails changes to the inverse relation (`capital_of`), to the functional constraint (Santiago is no longer *a* capital of Chile), to compositions (`mayor_of(capital_of(Chile))`), and to every downstream fact whose derivation passes through the edited edge.

**The problem.** Given a base model $f_\theta$, a relational schema $\mathcal{S}$ (relation signatures, functionality/inverse/transitivity axioms, type constraints), and an edit set $E$, produce $\theta^*$ such that the model's behaviour agrees with the *deductive closure* of the edited knowledge base under $\mathcal{S}$ — not merely with $E$ itself — while leaving facts outside the closure unchanged.

Three variants, of very different difficulty:

- **Measurement.** Build an evaluation whose "correct post-edit answer" is the schema closure, and which separates *propagation failure* from *the model never having known the bridge fact*. Currently the weakest link.
- **Method.** Find an editor that achieves closure-consistency without enumerating the closure at edit time (the closure of a single edit over Wikidata-scale relations can be $10^3$–$10^6$ triples).
- **Theory.** Determine whether closure-consistency is achievable by *localized parameter edits at all*, or whether it necessarily requires either retrieval/context or a training-scale update.

Solving it means: an editor that, given $E$ and $\mathcal{S}$ only (no enumerated consequences), matches an oracle retrieval baseline on multi-hop and inverse-relation queries while matching a no-edit baseline on unrelated facts.

## 2. Formal Setting

Let $\mathcal{E}$ be entities, $\mathcal{R}$ relations, and $K \subseteq \mathcal{E}\times\mathcal{R}\times\mathcal{E}$ the base knowledge base. A schema $\mathcal{S}$ is a set of Horn rules
$$\mathcal{S} = \{\, \forall \bar{x}\; \varphi_1(\bar x) \wedge \dots \wedge \varphi_k(\bar x) \Rightarrow \psi(\bar x) \,\}$$
covering inverses ($r(x,y)\Rightarrow r^{-1}(y,x)$), functionality ($r(x,y)\wedge r(x,z)\Rightarrow y=z$), and composition ($r_1(x,y)\wedge r_2(y,z)\Rightarrow r_3(x,z)$).

An edit set $E=\{(s_i,r_i,o_i\to o_i^*)\}_{i=1}^m$ induces a revised base $K'=(K\setminus E^-)\cup E^+$, and the target is the closure $\mathrm{Cl}_\mathcal{S}(K')$ under chase semantics, with functionality violations resolved by retraction of the older tuple.

**Measured quantities.** Queries are natural-language templates $q(\cdot)$ with a verbalizer; $f_\theta(q)$ is the greedy decode.

- **Edit success:** $\mathrm{ES} = \frac{1}{m}\sum_i \mathbb{1}[f_{\theta^*}(q(s_i,r_i)) = o_i^*]$.
- **Closure accuracy at depth $d$:** over derived triples $t \in \mathrm{Cl}_\mathcal{S}(K')\setminus K'$ whose shortest derivation uses $d$ rule applications,
$$\mathrm{CA}_d = \frac{1}{|T_d|}\sum_{t\in T_d}\mathbb{1}[f_{\theta^*}(q(t))=o(t)].$$
- **Retraction rate:** fraction of $t\in \mathrm{Cl}_\mathcal{S}(K)\setminus\mathrm{Cl}_\mathcal{S}(K')$ that the model no longer asserts. Retraction is the half almost never measured.
- **Locality / specificity:** agreement with $f_\theta$ on a control set $N$ disjoint from both closures, $\mathrm{LOC}=\frac{1}{|N|}\sum \mathbb{1}[f_{\theta^*}(q)=f_\theta(q)]$.
- **Capability drag:** $\Delta$ on a held-out general benchmark (MMLU, perplexity on WikiText) after $m$ sequential edits.

**Bridge-knowledge control.** $\mathrm{CA}_d$ is uninterpretable without conditioning on whether the base model knew the bridge facts. Define $B$ = set of derived triples where $f_\theta$ answered all $d$ intermediate hops correctly pre-edit; report $\mathrm{CA}_d\mid B$.

**Assumptions, and where they break.**
1. *Closure is finite and computable.* Holds for Horn schemas; breaks once soft/defeasible relations enter (`likely_capital_of`).
2. *A single verbalizer measures belief.* Known violated — edit success varies by $>20$ points across paraphrases and under prompt reordering.
3. *Facts are atomic and independent.* Violated: parametric edits leak along embedding similarity, not along schema edges.
4. *Ground truth post-edit is unique.* Violated for counterfactual edits that contradict physical or temporal constraints; annotators disagree on which entailments should survive.

## 3. State of the Art

**Parametric editors (systems SOTA).** ROME (Meng et al., NeurIPS 2022) and MEMIT (Meng et al., ICLR 2023) do rank-one / batched rank-$m$ updates to MLP down-projections; MEMIT edits $10^4$ facts in GPT-J (6B) and GPT-NeoX (20B) with high single-hop success. AlphaEdit (Fang et al., ICLR 2025) projects the update onto the null space of preserved-knowledge keys and reports large sequential-editing gains. GRACE (Hartvigsen et al., NeurIPS 2023) uses a discrete codebook adapter for long edit streams. *Established:* single-hop rewrite and paraphrase generalization. *Claimed but unablated for this problem:* that any of these propagate to inverses or compositions — none takes $\mathcal{S}$ as input, and none reports retraction rate.

**Retrieval / in-context editors.** MeLLo (with MQuAKE, Zhong et al., EMNLP 2023) decomposes multi-hop questions and checks each hop against an edit memory; IKE-style in-context editing (Zheng et al., EMNLP Findings 2023) prepends demonstrations. On ripple-effect evaluation, in-context editing beat parametric editors on most criteria (Cohen et al., TACL 2024). This is the practical SOTA — and it is not a model edit; the schema consequences are computed at inference time by the prompt, not stored.

**Benchmarks.** MQuAKE (multi-hop, 3k/9k instances over Wikidata relations); RippleEdits (Cohen et al., TACL 2024) with six criteria including Logical Generalization, Compositionality I/II, Subject Aliasing, Forgetfulness; CounterFact and the tightened CounterFact+ (Hoelscher-Obermaier et al., ACL Findings 2023). Most published "compositionality" numbers exist **only as benchmark scores**, with no ablation isolating propagation from bridge-knowledge absence or from prompt-format effects.

## 4. What Is Known

- **Single-hop success does not transfer to multi-hop.** On MQuAKE (GPT-J 6B, Vicuna-7B), parametric editors including MEMIT and ROME score in the low single digits to ~10% multi-hop accuracy on multi-edit instances while single-hop edit success exceeds 90%. MeLLo raises multi-hop accuracy substantially at the same edit budget. The gap is the finding; the exact points move with prompt format.
- **Ripple effects mostly fail.** RippleEdits (GPT-2 XL 1.5B, GPT-J 6B, LLaMA-2 7B, GPT-3): average accuracy across the six criteria is roughly 40–50% for ROME/MEMIT/MEND, with Compositionality II and Forgetfulness among the worst; in-context editing scored highest overall.
- **Specificity is over-reported.** CounterFact+ showed ROME's near-perfect specificity on CounterFact collapses once the neighbourhood prompts share context with the edited subject.
- **Sequential editing degrades the model.** Gupta et al. (ACL Findings 2024) report gradual then catastrophic forgetting for ROME/MEMIT under sequential edits; Gu et al. (2024) show measurable drops in general abilities (reasoning, summarization) after modest edit counts on LLaMA-1 7B / GPT-2 XL. AlphaEdit's null-space projection is the strongest published mitigation.
- **Localization does not predict editability.** Hase et al. (NeurIPS 2023): causal-tracing layer localization is essentially uncorrelated with where an edit succeeds — so "edit the right site and consequences follow" has no empirical support.
- **Injected entities do not propagate.** Onoe et al. (ACL 2023): after injecting a new entity's description, models answer only weakly on inferences the description entails.

## 5. What Is Not Known

- **Theoretically open.** Whether closure-consistency for a schema of depth $d$ is achievable by a rank-$O(m)$ parameter update at all. No lower bound exists relating $\mathrm{rank}(\Delta W)$ to $|\mathrm{Cl}_\mathcal{S}(K')\setminus K'|$, and no proof that transformers store relations in a form supporting composition of edited edges. Whether *retraction* (making the model stop asserting a derivable-before fact) is representationally distinct from *insertion* is also unproven.
- **Empirically open.** Whether closure accuracy improves with model scale conditioned on bridge knowledge. No study has run the $\mathrm{CA}_d\mid B$ conditioning across a 7B→70B→400B ladder. Also open: whether schema-aware batched editing (materializing depth-1 closure into the edit batch) closes the multi-hop gap, or whether interference among the $10^2$–$10^3$ induced edits destroys locality.
- **Methodologically blocked.** The measurement. Existing multi-hop benchmarks confound (a) propagation failure, (b) missing bridge facts, (c) prompt-format brittleness, (d) benchmark artifacts — MQuAKE's construction has been reported to allow order/consistency exploits that inflate scores *(frontier — verify: MQuAKE-Remastered line of work, 2024–2025)*. Retraction is essentially unmeasured: no widely used benchmark scores whether the model *stops* asserting consequences of the retracted tuple.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of the target**.

1. *Confound.* A wrong multi-hop answer has four causes and the standard metric distinguishes none. Without the $\mid B$ conditioning, a 5% score is compatible with "propagation is impossible" and with "the base model knew 6% of the bridge facts".
2. *Non-identifiability.* For a counterfactual edit, the schema closure is not the only defensible target. Should "Chile's capital is Valparaíso" change the population figures attributed to the capital? Annotators disagree; the closure is defined only relative to a chosen $\mathcal{S}$, and no benchmark publishes its $\mathcal{S}$ as a machine-checkable rule set.
3. *Scaling wall.* Materializing the closure is the obvious fix and it defeats the purpose: edits scale as $O(|\mathrm{Cl}|)$, and sequential-editing degradation is superlinear in edit count in published curves.

None of this is compute-bound. A 7B-scale decisive experiment fits on one 8×A100 node.

## 7. Current Research (as of 2026)

- **Null-space and projection-constrained editors** — AlphaEdit and successors, aimed at sequential-edit stability rather than composition. The link to schema closure is unexplored *(frontier — verify)*.
- **Retrieve-then-edit hybrids** — MeLLo, DeepEdit, and later planner-based decomposers; strongest on MQuAKE but shift the work to inference.
- **Benchmark repair** — reliability audits of MQuAKE and RippleEdits; explicit schema/rule annotation *(frontier — verify)*.
- **Mechanistic work on relation representation** — linear-relational-embedding results (Hernandez et al., ICLR 2024) suggest many relations are approximately affine maps in representation space, which is the strongest existing handle on why composition might be editable.
- **Tooling** — EasyEdit (Wang et al.) is the de facto shared implementation, which helps reproducibility but propagates the same metric definitions.

## 8. Concrete Next Experiment

**Question.** Is the multi-hop gap propagation failure, or missing bridge knowledge?

**Scale.** LLaMA-3 8B and 70B (or equivalent open pair). Build 1,000 two-hop instances $r_3(x,z) \Leftarrow r_1(x,y)\wedge r_2(y,z)$ from Wikidata, editing only the first hop. Budget: ~200 GPU-hours on 8×A100.

**Filter (the key step).** Pre-edit, query each model on both hops separately with 5 paraphrases each. Keep only instances where the model answers *both* hops correctly on $\geq 4/5$ paraphrases. This yields the bridge-known subset $B$; expect $|B|\approx 250$–450 of 1,000.

**Arms.**
1. MEMIT, edit hop-1 only.
2. AlphaEdit, edit hop-1 only.
3. **Control A (oracle closure):** MEMIT with the *entailed* hop-3 fact added explicitly to the edit batch — upper bound on what parametric editing can do.
4. **Control B (retrieval):** unedited model with the edit in context — practical ceiling.
5. **Control C (no edit):** measures benchmark artifacts; should score at the pre-edit-answer rate, not at the post-edit rate.

**Deciding number.** $\mathrm{CA}_2 \mid B$ for arm 1, reported with locality $\mathrm{LOC}$ held above 95%. If $\mathrm{CA}_2\mid B < 25\%$ while Control A exceeds 85%, parametric single-edge editing does not compose and the field should stop reporting single-hop success as evidence of knowledge update. If $\mathrm{CA}_2\mid B > 60\%$, the published multi-hop failures were largely a bridge-knowledge artifact and the benchmarks need rebuilding. Secondary number: retraction rate on the displaced consequence $r_3(x,z_{\text{old}})$ — predicted near 0% in all parametric arms.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[SOTA / benchmark]** Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, Danqi Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP, 2023. — arXiv:2305.14795
- **[SOTA / benchmark]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[SOTA]** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Xiang Wang, Xiangnan He, Tat-Seng Chua. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR, 2025. — arXiv:2410.02355
- **[SOTA]** Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi. *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.* NeurIPS, 2023. — arXiv:2211.11031
- **[Critique]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[Critique]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* ACL Findings, 2023. — arXiv:2305.17553
- **[Critique]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* ACL Findings, 2024. — arXiv:2401.07453
- **[Related]** Yasumasa Onoe, Michael J.Q. Zhang, Shankar Padmanabhan, Greg Durrett, Eunsol Choi. *Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge.* ACL, 2023. — arXiv:2305.01651
- **[Mechanism]** Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, David Bau. *Linearity of Relation Decoding in Transformer Language Models.* ICLR, 2024. — arXiv:2308.09124
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023. — arXiv:2305.13172

## 10. Worked Example

**Edit.** `capital_of_country(Chile) = Santiago → Valparaíso`, applied to LLaMA-2 7B with MEMIT.

**Schema.** Three rules: inverse `capital_of(x,y) ⇒ is_capital_city_of(y,x)`; functionality (Chile has one capital); composition `capital_of(c,y) ∧ located_in_region(y,z) ⇒ capital_region(c,z)`.

**Closure of this one edit** over a Wikidata-shaped neighbourhood: 1 asserted triple, 1 inverse, 1 retraction (`is_capital_city_of(Santiago, Chile)` must go), and 14 depth-2 compositions (region, timezone, mayor, population-of-capital, etc.). Target set: 17 behaviours from 1 edit.

**Typical observed pattern** (numbers of this order are what the RippleEdits/MQuAKE literature reports for ROME/MEMIT at 7B; the exact split is what the §8 experiment would pin down):

| Behaviour | Count | Post-edit correct |
|---|---|---|
| Asserted edit + paraphrases | 1 | ✅ ~95% |
| Inverse (`is_capital_city_of(Valparaíso, Chile)`) | 1 | ✗ often unchanged |
| Retraction (`is_capital_city_of(Santiago, Chile)` still asserted) | 1 | ✗ near-0% retracted |
| Depth-2 compositions | 14 | ✗ ~1–3 of 14 |

Amplification: $17\times$ target, $\approx 3$–$4$ achieved. Edit success reads 95%; closure consistency reads $\approx 20\%$.

**Now the obstruction.** Query the *unedited* model: "Which region is Chile's capital located in?" It answers correctly. "Which region is Valparaíso located in?" — at 7B this is answered correctly only some of the time. So for a depth-2 item where the model never knew `located_in_region(Valparaíso, ·)`, a post-edit failure is *not* a propagation failure. Without the $\mid B$ filter the two are summed into one number, and every published comparison between editors on multi-hop accuracy is partly a comparison of how many bridge facts happened to be known. That is why the field's headline numbers do not settle the question, and why the deciding experiment is a filter, not a new editor.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*