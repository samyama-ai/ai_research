---
id: 20-interpretability/circuit-discovery-distribution-shift
title: "Circuit Discovery Under Distribution Shift"
topic: 20-interpretability
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Circuit Discovery Under Distribution Shift

> **Topic:** Interpretability · **ID:** `20-interpretability/circuit-discovery-distribution-shift` · **Status:** empirically-open

## 1. Problem Statement

A *circuit* is a sparse subgraph of a network's computational graph claimed to carry out a specific behaviour. Circuits are discovered on a task distribution $\mathcal{D}$ — a template family, a prompt set, a counterfactual pairing — and then reported as facts about the model. The question here is what survives when the distribution moves.

Input: a model $M$, a discovery distribution $\mathcal{D}_{\text{id}}$, a shifted distribution $\mathcal{D}_{\text{ood}}$ that a competent human would call the *same task*. Output: a circuit $C$ and a claim about its behaviour on $\mathcal{D}_{\text{ood}}$. Decision predicate: is $C$ discovered on $\mathcal{D}_{\text{id}}$ still faithful on $\mathcal{D}_{\text{ood}}$, and does the *rank ordering* of discovery methods by in-distribution faithfulness predict their ordering out of distribution?

Three variants, different difficulty:

- **Measurement.** Define a faithfulness score comparable across two distributions. Currently not well posed: every ablation-based score is defined relative to a corruption distribution that itself moves under shift.
- **Method.** Build a discovery algorithm whose output is stable under shift, or which reports its own domain of validity.
- **Theory.** Give conditions on $M$ and the shift under which a faithful circuit on $\mathcal{D}_{\text{id}}$ is provably faithful on $\mathcal{D}_{\text{ood}}$.

Solving it means: a faithfulness metric with a fixed reference that does not move with the eval set, plus a demonstration at $\geq$1B parameters that circuits selected under it transfer.

## 2. Formal Setting

Let $M$ have computational graph $G=(V,E)$: $V$ = attention heads, MLPs, or SAE latents at token positions; $E$ = the paths between them under a chosen decomposition. A circuit is $C \subseteq E$.

**Metric.** A task defines a scalar readout $m(\cdot)$ — usually a logit difference, $m(x) = z_{\text{correct}}(x) - z_{\text{wrong}}(x)$, measured in logits.

**Ablation.** Discovery needs a counterfactual. Write $M_{C}^{\mathcal{A}}(x)$ for the model run on $x$ with every edge $e \notin C$ replaced by its value under ablation policy $\mathcal{A}$: zero, resample from $x' \sim \mathcal{D}_{\text{corrupt}}$, or mean over $\mathcal{D}$. Faithfulness as normally reported:

$$F_{\mathcal{D}}(C) = \frac{\mathbb{E}_{x\sim\mathcal{D}}\big[m(M_{C}^{\mathcal{A}}(x))\big] - \mathbb{E}_{x}\big[m(M_{\emptyset}^{\mathcal{A}}(x))\big]}{\mathbb{E}_{x}\big[m(M(x))\big] - \mathbb{E}_{x}\big[m(M_{\emptyset}^{\mathcal{A}}(x))\big]}$$

Numerator and denominator are both in logits; $F=1$ means the circuit alone reproduces full performance, $F=0$ means it reproduces the fully-ablated baseline.

**The measurement defect.** $\mathcal{A}$ depends on $\mathcal{D}$. Mean ablation over $\mathcal{D}_{\text{ood}}$ is a different intervention from mean ablation over $\mathcal{D}_{\text{id}}$, and the denominator is re-normalised too. So $F_{\mathcal{D}_{\text{id}}}(C)$ and $F_{\mathcal{D}_{\text{ood}}}(C)$ are not two readings of one quantity. Any transfer gap $\Delta F = F_{\text{id}} - F_{\text{ood}}$ mixes a real change in mechanism with a change in the metric's own zero point.

**Discovery.** ACDC greedily prunes $e$ while $\mathrm{KL}\big(M(x)\,\|\,M_{C\setminus e}^{\mathcal{A}}(x)\big) < \tau$. Edge attribution patching (EAP) scores each edge by a first-order estimate $\hat{s}(e) = (h_{e}^{\text{corrupt}} - h_{e}^{\text{clean}})^\top \nabla_{h_e} m$ and keeps the top $K$; EAP-IG replaces the gradient with an integrated-gradients path average.

**Shift taxonomy.** $\mathcal{D}_{\text{ood}}$ can differ by surface template (ABBA→BABA name order in IOI), by lexicon (frequent→rare names), by length, by language, or by an adversarial constructor chosen to maximise $\Delta F$.

**Assumptions known violated in practice.** (i) *Sparsity* — that a small $C$ suffices; violated by backup/self-repair behaviour, where ablating a head causes downstream heads to increase their contribution (the Hydra effect, McGrath et al. 2023). (ii) *Linearity of the patching estimate* — EAP's first-order approximation is poor for attention-pattern edges; this is exactly what EAP-IG was introduced to fix. (iii) *Off-distribution ablation validity* — ablated activations put the model in states it never occupies, so $m(M_C^{\mathcal{A}}(x))$ may not measure the mechanism at all. (iv) *Task identity across shift* — that $\mathcal{D}_{\text{id}}$ and $\mathcal{D}_{\text{ood}}$ pose the same problem to the model, which is a human judgement, not a measured one.

## 3. State of the Art

**Established.**
- Manual circuit discovery works and is reproducible on small models: the IOI circuit in GPT-2 small, 26 of 144 attention heads across 7 classes (Wang et al., ICLR 2023); the greater-than circuit (Hanna et al., NeurIPS 2023).
- Automation recovers hand-found circuits approximately, not exactly. ACDC (Conmy et al., NeurIPS 2023) reports ROC AUC against canonical circuits well below 1 and misses components a human found. EAP with integrated gradients beats plain EAP and ACDC on circuit-level faithfulness at fixed edge budget (Hanna et al., COLM 2024).
- Component reuse across tasks is real: the IOI name-mover/inhibition machinery is recruited by the Colored Objects task in GPT-2 medium, and editing it changes that task's behaviour (Merullo et al., ICLR 2024).
- Simplified mechanistic explanations that match the full model in-distribution diverge out of distribution (Friedman et al., ICML 2024).

**Claimed but unablated.**
- That EAP-IG's in-distribution ranking of methods carries to shifted evaluation. No paper ablates method rank *under shift*.
- That SAE-based feature circuits are more transferable than component circuits because features are more "natural" units. Sparse feature circuits (Marks et al., ICLR 2025) show a downstream generalisation win — the SHIFT procedure removes a spurious gender feature and raises worst-group accuracy on ambiguous Bias-in-Bios by tens of points — but this is one task, and it is a *debiasing* result, not a circuit-transfer measurement.

**Benchmark-number-only.** MIB (Mueller et al., ICML 2025) supplies a standardised circuit-localisation leaderboard across several models and tasks. Its scores are within-task; no split is a declared distribution shift, so a high MIB number is not evidence of transfer.

## 4. What Is Known

- **Scale of hand-verified circuits.** GPT-2 small, 117M parameters: IOI circuit recovers roughly 87% of the full model's logit difference with 26 heads under mean ablation over the ABC-corrupted distribution (Wang et al. 2023). Everything about circuit transfer is anchored to models of this size.
- **Faithfulness is metric-fragile.** Miller, Chughtai and Saunders (COLM 2024) show that reported faithfulness of the IOI circuit swings across the plausible space of ablation choices, corruption sets and metric definitions — including regimes where the circuit appears to exceed 100% of full-model performance. The variation from methodological choices is comparable to the variation between different candidate circuits.
- **Self-repair is large.** Ablating a strong component is partly compensated downstream (McGrath et al. 2023), so a component's measured necessity understates its causal role — and the size of the compensation is itself distribution-dependent.
- **Adversarially chosen inputs degrade published circuits.** Adversarial circuit evaluation (uit de Bos and Garriga-Alonso, 2024) re-evaluates the IOI, docstring and greater-than circuits on inputs constructed to break them and finds substantially higher KL divergence between circuit and full model than on the original distributions.
- **Illusions predate circuits.** Bolukbasi et al. (2021) documented BERT neuron interpretations that hold on one corpus and fail on another.

## 5. What Is Not Known

- **Methodologically blocked.** A faithfulness score comparable across distributions. Because the ablation reference and the normalising denominator are both functions of the eval distribution, $\Delta F$ has no clean interpretation. Until a fixed reference exists (a frozen ablation bank, or a metric defined against a distribution-independent null), "the circuit transferred" is not a measurable claim.
- **Empirically open.** Whether discovery methods rank the same in and out of distribution. Every ingredient exists — ACDC, EAP, EAP-IG, sparse feature circuits, MIB tasks, shifted splits — and nobody has run the crossed comparison at $\geq$1B parameters. Also open: whether SAE-latent circuits transfer better than component circuits, at matched edge budget.
- **Theoretically open.** No condition on $(M, \mathcal{D}_{\text{id}}, \mathcal{D}_{\text{ood}})$ under which faithfulness is guaranteed to be preserved. Non-identifiability makes this hard to even state: many subgraphs achieve near-equal $F$ on $\mathcal{D}_{\text{id}}$, and nothing in the objective selects the one that transfers.

## 6. Why It Is Hard

The specific obstruction is **a metric whose reference point moves with the thing being measured**. Faithfulness is defined by ablation against a corruption distribution drawn from the eval set. Change the eval set and you change the intervention, the baseline $M_\emptyset$, and the normaliser — so the same circuit gets a different score for reasons that have nothing to do with its mechanism.

Two secondary obstructions compound it. **Non-identifiability**: at a fixed edge budget the top-$K$ set is not unique, and near-tied subgraphs on $\mathcal{D}_{\text{id}}$ can diverge arbitrarily on $\mathcal{D}_{\text{ood}}$; a transfer failure may reflect an arbitrary tie-break, not a wrong theory. **Absent ground truth**: outside a handful of GPT-2-small tasks there is no reference circuit, so "did it transfer?" reduces to "did the score hold up?" — circular given obstruction one.

## 7. Current Research (as of 2026)

- **Benchmarking circuit localisation.** MIB (Mueller, Belinkov, Bau, Marks and collaborators; Technion / Northeastern / EleutherAI orbit) is the standard target. Extending it with explicit shift splits is an obvious next step *(frontier — verify whether such splits have landed)*.
- **Faithfulness metric reform.** Follow-ons to Miller et al. (2024) and to causal scrubbing (Chan et al., Redwood Research, 2022) aim at ablation policies that do not push activations off-manifold.
- **Feature-level circuits.** Sparse feature circuits and transcoder-based circuit tracing (Anthropic, Google DeepMind, EleutherAI) shift the node type from components to latents. Whether this improves transfer is untested *(frontier — verify)*.
- **Circuit reuse and universality.** Merullo/Pavlick (Brown) on cross-task reuse; universality work on whether the same circuits recur across seeds and scales.

## 8. Concrete Next Experiment

**Question.** Does in-distribution circuit faithfulness predict out-of-distribution faithfulness?

**Scale.** Two models: GPT-2 small (117M, where hand-verified circuits exist) and Llama-3.1-8B or Gemma-2-9B (where they do not). Four tasks from MIB. For each task, one shifted split that preserves the task and changes surface form — for IOI: rare/non-Anglophone names, three-name distractors, 2× length.

**Arms.** Discovery methods: ACDC, EAP, EAP-IG, sparse-feature circuits, at matched edge budgets $K \in \{50, 200, 1000\}$.

**Control arm — this is the part that makes it interpretable.** A *frozen ablation bank*: all ablation values sampled once from a fixed generic corpus and reused for both $\mathcal{D}_{\text{id}}$ and $\mathcal{D}_{\text{ood}}$ evaluation, so the intervention does not move with the eval set. Second control: random subgraphs of size $K$ matched on layer/head-type marginals, giving the null $\Delta F$ attributable to metric drift alone.

**Deciding number.** Spearman $\rho$ between per-circuit $F_{\text{id}}$ and $F_{\text{ood}}$ under the frozen bank, pooled over methods, budgets and tasks. $\rho \geq 0.7$ means in-distribution discovery is a usable proxy and the field's practice is sound. $\rho \leq 0.3$ means published circuits are claims about a template family, not about the model, and every circuit paper needs a shift split. Report with bootstrap CIs over tasks, and report the random-subgraph $\Delta F$ alongside — if the null drift exceeds the real circuits' drift, the metric, not the circuit, is what failed.

## 9. Key References

- **[Foundational]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023. — arXiv:2211.00593
- **[Foundational]** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023. — arXiv:2304.14997
- **[SOTA]** Hanna, Pezzelle, Belinkov. *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms.* COLM 2024. — arXiv:2403.17806
- **[SOTA]** Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* 2023. — arXiv:2310.10348
- **[SOTA]** Marks, Rager, Michaud, Belinkov, Bau, Mueller. *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* ICLR 2025. — arXiv:2403.19647
- **[Critique]** Miller, Chughtai, Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM 2024. — arXiv:2407.08734
- **[Critique]** Friedman, Lampinen, Dixon, Chen, Ghandeharioun. *Interpretability Illusions in the Generalization of Simplified Models.* ICML 2024.
- **[Critique]** uit de Bos, Garriga-Alonso. *Adversarial Circuit Evaluation.* 2024.
- **[Related]** Merullo, Eickhoff, Pavlick. *Circuit Component Reuse Across Tasks in Transformer Language Models.* ICLR 2024. — arXiv:2310.08744
- **[Related]** Hanna, Liu, Variengien. *How Does GPT-2 Compute Greater-Than? Interpreting Mathematical Abilities in a Pre-trained Language Model.* NeurIPS 2023. — arXiv:2305.00586
- **[Related]** McGrath, Rahtz, Kramár, Mikulik, Legg. *The Hydra Effect: Emergent Self-Repair in Language Model Computations.* 2023. — arXiv:2307.15771
- **[Related]** Bolukbasi, Pearce, Yuan, Coenen, Reif, Viégas, Wattenberg. *An Interpretability Illusion for BERT.* 2021. — arXiv:2104.07143
- **[Benchmark]** Mueller et al. *MIB: A Mechanistic Interpretability Benchmark.* ICML 2025.

## 10. Worked Example

Take IOI in GPT-2 small. In-distribution: 15-token ABBA templates, common English first names, mean ablation over the ABC-corrupted set. The published 26-head circuit recovers about 87% of the full-model logit difference — call it $F_{\text{id}} = 0.87$, with the full model at roughly 3.5 logits of clean logit difference and the fully-ablated baseline near 0.

Now shift: replace names with rare multi-token names (*Anastasiya*, *Oluwaseun*), keeping template and length. Run the *same* 26 heads and compute $F_{\text{ood}}$ the way papers do — re-derive the mean-ablation values from the new distribution, re-derive the normaliser.

Three things move at once:

1. The clean logit difference drops, say 3.5 → 2.1 logits, because the task itself is harder with rare names. The denominator shrinks by 40%.
2. The mean-ablation baseline shifts, because activation means over rare-name prompts differ from means over common-name prompts. The zero point moves an unknown amount.
3. Name-mover heads now attend to a *second* token position, so any circuit defined at the token-position granularity of the original is edge-mismatched before mechanism is even considered.

Suppose the measurement returns $F_{\text{ood}} = 0.62$. The 0.25 drop admits at least three readings that this experiment cannot separate: the circuit genuinely does less of the work; the mechanism is intact but the shrunken denominator amplifies a fixed absolute residual; or backup heads self-repair differently under the new ablation values, inflating $M_\emptyset$ and compressing the whole scale.

The random-subgraph control makes the problem concrete. Score 26 randomly chosen heads matched on layer distribution. If those go from $F_{\text{id}} = 0.10$ to $F_{\text{ood}} = 0.05$, the drift is small and the 0.25 drop is probably mechanism. If they go $0.10 \to 0.35$ — plausible, since ablation values drawn from a distribution the model handles poorly can make *any* subgraph look load-bearing — then the metric moved further than the circuit did, and the published number says nothing about whether the mechanism transferred. That is the obstruction: without a frozen ablation reference, the transfer question has no answer, only a number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*