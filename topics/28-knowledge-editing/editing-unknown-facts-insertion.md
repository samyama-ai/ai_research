---
id: 28-knowledge-editing/editing-unknown-facts-insertion
title: "Editing Facts the Model Never Knew"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Editing Facts the Model Never Knew

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editing-unknown-facts-insertion` · **Status:** open

## 1. Problem Statement

Model editing is normally posed as *rewriting*: the model already associates subject $s$ and relation $r$ with some object $o$, and the edit swaps $o \to o^*$. **Insertion** is the other case: the model has no association for $(s, r)$ at all — the entity postdates the pretraining cutoff, or is private, or is simply absent from the corpus. The task is to install $(s, r, o^*)$ so the model uses it like knowledge it learned in pretraining, without retraining.

Three variants, which are usually conflated:

- **Measurement.** Decide whether a given fact is *unknown* to $\theta$ before editing, and whether it is *known* after — as opposed to merely recited under the edit prompt. Both directions are unresolved.
- **Method.** Build an editor whose paraphrase generalization, multi-hop portability, and locality on unknown facts match its numbers on known facts.
- **Theory.** Determine whether rank-one / low-rank interventions on MLP weights can install an association whose key direction did not previously exist, or whether insertion requires a representational change that such interventions cannot express.

Solving it means: an editor whose *known-vs-unknown gap* on generalization and portability is under measurement noise, at fixed efficacy and locality.

## 2. Formal Setting

Autoregressive LM $p_\theta$ over vocabulary $V$. A fact is a triple $f = (s, r, o^*)$ with a prompt set $T(s,r) = \{t_1, \dots, t_m\}$ of paraphrases and a held-out neighborhood $N(f)$ of prompts that must not change.

**Knownness, as measured.** Following the SliCK protocol (Gekhman et al., EMNLP 2024), sample $k$ generations at temperature $\tau$ for each of $m$ few-shot-prompted paraphrases and take
$$\widehat{P}_{\text{corr}}(f;\theta) = \frac{1}{mk}\sum_{i=1}^{m}\sum_{j=1}^{k}\mathbb{1}\!\left[y_{ij} \equiv o^*\right],\quad y_{ij}\sim p_\theta(\cdot \mid t_i).$$
$f$ is **Unknown** iff $\widehat{P}_{\text{corr}} = 0$ with greedy decoding also wrong, at $m=4$, $k=16$, $\tau=0.5$. This is an operational definition, not a statement about $\theta$'s internals.

**Edit operator.** $\theta' = E(\theta, f)$. Metrics, all measured on held-out strings:
$$\mathrm{ES} = \mathbb{1}[p_{\theta'}(o^*\!\mid t_1) > p_{\theta'}(o_c\!\mid t_1)],\quad \mathrm{PS} = \tfrac{1}{m-1}\!\sum_{i\ge2}\mathbb{1}[p_{\theta'}(o^*\!\mid t_i) > p_{\theta'}(o_c\!\mid t_i)],$$
with $\mathrm{NS}$ the same indicator with sign flipped over $N(f)$, and portability $\mathrm{PT}$ = accuracy on two-hop questions whose bridge entity is $o^*$ (MQuAKE protocol). For insertion there is no natural competitor $o_c$, since no prior object exists; the substitute is the model's pre-edit argmax completion, which is typically a type-plausible hallucination — this substitution is the first place the measurement leaks.

**The quantity of interest** is the gap at matched efficacy:
$$\Delta_{\mathrm{PS}} = \mathbb{E}_{f \in \mathcal{K}}[\mathrm{PS}] - \mathbb{E}_{f \in \mathcal{U}}[\mathrm{PS}] \quad \text{subject to } \mathbb{E}[\mathrm{ES}] \ge 0.99 \text{ in both arms},$$
where $\mathcal{K}$/$\mathcal{U}$ are known/unknown fact sets matched on relation, object type, and subject token length.

**Locate-then-edit form.** ROME writes MLP down-projection $W$ as linear associative memory and applies
$$\hat W = W + \frac{(v_* - Wk_*)\,(C^{-1}k_*)^\top}{k_*^\top C^{-1} k_*},\qquad C = \mathbb{E}_{k\sim \mathcal{D}}[kk^\top],$$
with $k_*$ the layer-$\ell$ key at the last subject token, averaged over random prefixes.

**Assumptions, and their status.**
1. *A stable subject key $k_*$ exists.* Violated for novel entities: multi-subword names have no consolidated mid-layer subject state, and prefix-to-prefix variance of $k_*$ is larger.
2. *$C$ estimated from Wikipedia covers the edited direction.* Violated by construction — unknown entities are out-of-distribution for $C$.
3. *Causal-trace localization identifies where to edit.* Falsified: edit success is largely uncorrelated with traced layer (Hase et al., NeurIPS 2023).
4. *A single $(s,r,o^*)$ update suffices for downstream use.* Violated: ripple effects and multi-hop composition fail (Cohen et al., TACL 2024).

## 3. State of the Art

**Systems/empirical SOTA.** ROME (Meng et al., NeurIPS 2022) and MEMIT (ICLR 2023) dominate on CounterFact; MEND (ICLR 2022) and SERAC (ICML 2022) are hypernetwork/retrieval alternatives; GRACE (NeurIPS 2023) uses an explicit discrete key–value codebook and is the only method whose mechanism does not assume a pre-existing association. Retrieval / in-context editing is the strongest baseline on ripple-effect and multi-hop tests — often beating all weight editors.

**Established.** ROME's near-100% efficacy on CounterFact is *analytically guaranteed*, not empirical: the rank-one solve makes $\hat W k_* = v_*$ exactly (§10). MEMIT scales to $10^4$ simultaneous edits on GPT-J-6B with aggregate score holding up. Causal-trace-guided site selection does not explain edit success.

**Claimed but unablated.** That these methods "insert" knowledge. CounterFact, zsRE, and most editing suites are built from facts the model already holds — an insertion arm is usually absent, and where unknown-entity facts appear they are not matched to known controls on relation and object type. No published editor reports $\Delta_{\mathrm{PS}}$ as defined above.

**Benchmark-number-only results.** RippleEdits and MQuAKE scores are aggregate leaderboard figures without per-fact knownness stratification; they tell you editors fail at composition, not whether they fail *differently* on unknown facts.

## 4. What Is Known

- **Fine-tuning on unknown facts is slow and harmful.** On PaLM-2-S with closed-book QA, examples the model does not already know are fitted substantially more slowly than known ones, and once fitted, they raise hallucination rates on *unrelated* questions roughly linearly in the number of unknown examples fitted (Gekhman et al., EMNLP 2024). This is the strongest evidence that insertion is mechanistically different from rewriting.
- **Injected knowledge does not become extractable by default.** In controlled synthetic biographies, models trained on facts without paraphrase augmentation reach near-chance extraction accuracy under QA-style probes despite near-perfect memorization of the training strings (Allen-Zhu & Li, ICML 2024, "Physics of Language Models 3.1"). Scale: 100M–1B parameter GPT-2-class models, millions of synthetic profiles.
- **Definition-based injection barely propagates.** Injecting an entity definition and querying entailed consequences yields small gains over doing nothing; distillation-based propagation helps but does not close the gap (Onoe et al., ACL 2023; Padmanabhan et al., NeurIPS 2023). Scale: GPT-Neo 1.3B–2.7B, ECBD entities postdating the cutoff.
- **Locality metrics overstate.** Under a harder specificity benchmark, ROME's apparent locality collapses — nearby prompts sharing the edited object are corrupted (Hoelscher-Obermaier et al., Findings of ACL 2023), GPT-2-XL and GPT-J scale.
- **Sequential editing degrades the model.** Repeated edits cause gradual then catastrophic forgetting, at low hundreds to thousands of edits on 1B–7B models (Gupta et al., Findings of ACL 2024).

## 5. What Is Not Known

- **Empirically open (dominant).** The matched known-vs-unknown comparison at fixed efficacy has not been run for any weight editor at 7B+ scale. $\Delta_{\mathrm{PS}}$, $\Delta_{\mathrm{PT}}$, and $\Delta_{\mathrm{NS}}$ are simply unmeasured. Every ingredient exists: ECBD-style post-cutoff entities, SliCK for knownness, EasyEdit for the editors. Cost is a few thousand GPU-hours.
- **Methodologically blocked.** Whether a fact is "known" is defined by a sampling procedure with free parameters $(m,k,\tau)$ and few-shot prompt choice; a fact can be Unknown at $\tau=0.5$ and Known at $\tau=0$ with a different template. There is no ground truth for knownness, so the partition of the dataset — the independent variable — is itself measurement-dependent.
- **Theoretically open.** Whether a rank-one update can create a *new* key direction. If $k_*$ for a novel entity lies near the span of existing keys, the update necessarily bleeds into them (locality loss); if it lies far outside, $C^{-1}k_*$ is dominated by low-variance directions and the edit is fragile. No theorem quantifies this tradeoff.

## 6. Why It Is Hard

**The obstruction is that efficacy is a tautology, so the headline metric does not measure the thing it names.** For rank-one editors, $\mathrm{ES}$ is the verification that a linear system was solved (§10) — it carries no information about knowledge acquisition, and it is identical for known and unknown facts. The informative signal lives entirely in $\mathrm{PS}$ and $\mathrm{PT}$, which depend on how tightly paraphrase keys concentrate around $k_*$. For pretrained entities that concentration was *produced by pretraining*; for novel entities there is nothing that produced it, so the editor's generalization is inherited from a structure the unknown case lacks. Compounding this: the control quantity $o_c$ does not exist for insertion, and the known/unknown partition is defined by a decoding procedure rather than by ground truth.

## 7. Current Research (as of 2026)

- **Retrieval and parametric–non-parametric hybrids** (SERAC, GRACE lineage; Hartvigsen and collaborators) — sidestep insertion by keeping new facts external. Strong on ripple tests, weak on the claim of true integration.
- **Knowledge-boundary and knownness estimation** (Google Research / Technion line following Gekhman et al.) — better estimators of $\mathcal{K}$ vs $\mathcal{U}$, which is the prerequisite measurement.
- **Continued-pretraining with synthetic augmentation** as the insertion baseline, following the Physics-of-LMs finding that paraphrase diversity, not repetition, drives extractability.
- **Editor evaluation reform** — matched-control designs and cross-lingual/multi-hop consistency suites (Zhejiang / EasyEdit group, Geva's group at Tel Aviv). *(frontier — verify)*
- Claims that in-context or LoRA-based insertion closes the gap at 70B scale are circulating but I have not seen a matched-control ablation. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** GPT-J-6B and Llama-3-8B (both with open checkpoints and known cutoffs). $|\mathcal{U}| = 500$ facts about entities whose Wikipedia page first appeared after the cutoff (ECBD construction), verified Unknown by SliCK at $m=4$, $k=16$, $\tau=0.5$. $|\mathcal{K}| = 500$ facts verified Known, matched one-to-one on relation, object type, and subject subword count.

**Arms.** ROME, MEMIT, MEND, full fine-tune, LoRA, and prepend-the-fact in context. Each arm tuned so mean $\mathrm{ES} \ge 0.99$ on *both* fact sets — matched efficacy is what makes the comparison legitimate.

**Control arm.** The matched $\mathcal{K}$ set edited counterfactually ($o \to o^*$, standard CounterFact-style rewrite). This is the arm every published number already covers; it calibrates the instrument.

**Decisive number.** $\Delta_{\mathrm{PS}} = \mathrm{PS}(\mathcal{K}) - \mathrm{PS}(\mathcal{U})$, with 95% bootstrap CI over facts.
- $\Delta_{\mathrm{PS}} < 5$ points for ROME/MEMIT ⟹ insertion is not a separate problem; the field can stop treating it as one.
- $\Delta_{\mathrm{PS}} > 20$ points ⟹ insertion is a distinct failure mode and all editor benchmarks built on known facts are reporting inflated generalization.

Secondary readout: $\rho = (k_*^\top C^{-1} k')/(k_*^\top C^{-1} k_*)$ averaged over paraphrase keys $k'$, reported per arm. It predicts $\mathrm{PS}$ mechanistically and costs nothing extra to compute.

## 9. Key References

- **[Foundational]** Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[SOTA]** Meng, Sharma, Andonian, Belinkov, Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[SOTA]** Hartvigsen, Sankaranarayanan, Palangi, Kim, Ghassemi. *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.* NeurIPS 2023. — arXiv:2211.11031
- **[Key evidence]** Gekhman, Yona, Aharoni, Eyal, Feder, Reichart, Herzig. *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* EMNLP 2024. — arXiv:2405.05904
- **[Key evidence]** Allen-Zhu, Li. *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction.* ICML 2024. — arXiv:2309.14316
- **[Key evidence]** Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Evaluation]** Cohen, Biran, Yoran, Globerson, Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. — arXiv:2307.12976
- **[Evaluation]** Zhong, Wu, Manning, Potts, Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP 2023. — arXiv:2305.14795
- **[Evaluation]** Hoelscher-Obermaier, Persson, Kran, Konstas, Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL 2023.
- **[Unknown entities]** Onoe, Zhang, Choi, Durrett. *Entity Cloze By Date: What LMs Know About Unseen Entities.* Findings of NAACL 2022. — arXiv:2205.02832
- **[Unknown entities]** Onoe, Zhang, Padmanabhan, Durrett, Choi. *Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge.* ACL 2023.
- **[Propagation]** Padmanabhan, Onoe, Zhang, Durrett, Choi. *Propagating Knowledge Updates to LMs Through Distillation.* NeurIPS 2023.
- **[Degradation]** Gupta, Rao, Anumanchipalli. *Model Editing at Scale Leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024.
- **[Method]** Mitchell, Lin, Bosselut, Finn, Manning. *Fast Model Editing at Scale.* ICLR 2022. — arXiv:2110.11309
- **[Survey]** Yao, Wang, Tian, Cheng, Xi, Deng, Chen, Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172

## 10. Worked Example

Insert into GPT-J-6B: *"The Kessler–Ohara Telescope is operated by ___ → the Atacama Consortium."* The subject is post-cutoff; SliCK returns $\widehat{P}_{\text{corr}} = 0/64$. Pre-edit greedy completion is "NASA" — a type-plausible hallucination, which becomes $o_c$ by default.

**Step 1 — efficacy is free.** ROME computes $k_*$ (dim 4096) at layer 5's MLP, solves for $v_*$ at the 16384-dim hidden site, and applies the rank-one update. Then
$$\hat W k_* = W k_* + \frac{(v_* - W k_*)(k_*^\top C^{-1} k_*)}{k_*^\top C^{-1}k_*} = v_*$$
**exactly**. $\mathrm{ES} = 1$ is an algebraic identity, holding equally for a known-subject rewrite and this unknown-subject insertion. The metric that anchors every editing leaderboard carries zero bits about whether insertion worked.

**Step 2 — generalization is where the difference lives.** For a paraphrase key $k'$,
$$\hat W k' = W k' + (v_* - Wk_*)\,\rho, \qquad \rho = \frac{k_*^\top C^{-1} k'}{k_*^\top C^{-1} k_*}.$$
The edit transfers in proportion to $\rho$. ROME reports paraphrase scores around the mid-90s on CounterFact (GPT-J-6B), i.e. $\rho$ is close to 1 for pretrained subjects — because pretraining built a consolidated subject representation that makes all mentions of "Eiffel Tower" map to nearly the same key. "Kessler–Ohara" tokenizes into 5+ subwords with no such consolidation, so $\rho$ under paraphrase is expected to fall, and $\mathrm{PS}$ falls with it.

**Step 3 — the obstruction.** You cannot read this off any published table, because the two conditions are never run side by side at matched efficacy. And you cannot cleanly define the unknown arm's baseline: $o_c$ = "NASA" is an artifact of decoding, so a large $\mathrm{PS}$ could mean the edit generalized or merely that "NASA" was a weak competitor. Both problems are fixed by the same design — matched controls plus the $\rho$ readout — and neither has been run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*