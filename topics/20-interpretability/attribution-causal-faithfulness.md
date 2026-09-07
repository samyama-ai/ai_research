---
id: 20-interpretability/attribution-causal-faithfulness
title: "Causal Faithfulness of Attribution Methods"
topic: 20-interpretability
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Causal Faithfulness of Attribution Methods

> **Topic:** Interpretability · **ID:** `20-interpretability/attribution-causal-faithfulness` · **Status:** partially-solved

## 1. Problem Statement

An attribution method takes a model $f$, an input $x$, and a set of parts (input features, neurons, attention heads, edges of the computational graph) and returns a score per part. The claim implicitly attached to that score is **causal**: the part's score tracks how much the model's output depends on that part's computation.

The problem: state a definition of causal faithfulness that is (a) satisfiable by some method, (b) measurable without ground truth about the model's internals, and (c) not gameable by the metric's own free parameters.

Three variants, of very different difficulty:

- **Measurement.** Given $f$, an attribution $\phi$, and a task, output a number that increases if and only if $\phi$ better matches the model's causal structure. Currently unsolved: every deployed metric depends on an intervention distribution that changes the ranking of methods.
- **Method.** Produce an attribution that is faithful under a fixed, agreed metric at a cost sub-linear in the number of parts. Partially solved for circuit-level attribution (attribution patching, AtP\*); unsolved for input-feature attribution.
- **Theory.** Prove which classes of attribution can, in principle, certify statements about model behaviour. Partially resolved in the negative (Bilodeau et al., PNAS 2024).

Solving it means: a metric $M$ with a proof that $M(\phi)$ is high only if $\phi$ approximates ground-truth interventional effects, plus a method that scores high on $M$ at a cost that scales.

## 2. Formal Setting

Let $f:\mathcal{X}\to\mathbb{R}$ be a scalar readout (e.g. logit difference between a correct and a counterfactual token). Let the model be a computational DAG with nodes $V$ (components) and let $z_v(x)$ be the activation of node $v$ on input $x$.

**Ground-truth interventional effect.** For node $v$, a *clean* input $x$, and a *counterfactual* input $x'$ drawn from an intervention distribution $\mathcal{D}$:

$$\mathrm{IE}(v;x,\mathcal{D}) \;=\; \mathbb{E}_{x'\sim\mathcal{D}}\Big[f(x) - f\big(x \,\|\, \mathrm{do}(z_v = z_v(x'))\big)\Big]$$

where $x \,\|\, \mathrm{do}(\cdot)$ is a forward pass with $z_v$ overwritten. This is the quantity all attribution claims are compared against, and it is **not intrinsic to the model**: it is a function of $\mathcal{D}$. Choosing $\mathcal{D}$ = point mass at zero gives zero ablation, $\mathcal{D}$ = task distribution gives mean ablation, $\mathcal{D}$ = paired counterfactuals gives resample/interchange ablation. Measured cost: one forward pass per node per sample, $O(|V| \cdot n)$.

**Attribution.** $\phi: V \to \mathbb{R}$, e.g. gradient$\times$activation $\phi_v = \nabla_{z_v} f \cdot (z_v(x)-z_v(x'))$ (attribution patching — a first-order Taylor estimate of $\mathrm{IE}$ costing two forward passes and one backward pass total).

**Faithfulness metrics as measured.**
- *Correlation form:* $\rho\big(\phi(\cdot), \mathrm{IE}(\cdot;\mathcal{D})\big)$ over nodes.
- *Circuit form:* for a selected subgraph $C\subseteq V$, $\;\mathrm{Faith}(C) = \frac{f_C - f_\emptyset}{f_V - f_\emptyset}$, where $f_C$ is the readout with everything outside $C$ ablated under $\mathcal{D}$. Reported as "the circuit recovers $x\%$ of the logit difference".
- *Deletion form (input features):* area under the curve of $f$ as top-$k$ features by $\phi$ are removed; ROAR additionally **retrains** on the deleted data to control for distribution shift.

**Assumptions, and their status.**
1. *Ablation is off-distribution-safe* — violated. Zero-ablating a residual stream puts the network far outside its training distribution; measured effects then reflect OOD behaviour, not the mechanism.
2. *Effects are approximately additive / no self-repair* — violated. The Hydra effect (McGrath et al., 2023): ablating an attention head causes downstream heads to increase their contribution, so single-node $\mathrm{IE}$ underestimates and the sum of individual effects does not equal the joint effect.
3. *Local linearity* (required for attribution patching) — violated at attention softmax and at LayerNorm, where the Taylor error is large for direct-path patches.
4. *A unique ground-truth circuit exists* — unproven; superposition and distributed features make node-level decomposition non-identifiable.

## 3. State of the Art

**Theory (established).** Bilodeau, Jaques, Koh, Kim, *Impossibility theorems for feature attribution* (PNAS, 2024): for complete linear attributions — Integrated Gradients, SHAP, LIME, gradient$\times$input — there exist model classes for which the attribution provides no better-than-random performance on the task of inferring model behaviour under perturbation. This is a genuine impossibility result, not a benchmark number.

**Method (established for circuits).** Attribution patching / edge attribution patching (Syed, Rager, Conmy, 2023) approximates activation patching at $O(1)$ passes and recovers comparable circuits to ACDC (Conmy et al., NeurIPS 2023) at far lower cost. AtP\* (Kramár, Lieberum, Shah, Nanda, DeepMind, 2024) fixes the two dominant failure modes of naive attribution patching — attention-softmax saturation and cancellation — and is reported to dominate naive AtP and subsampling baselines at matched compute. Sparse feature circuits (Marks et al., 2024) move the node set from neurons to SAE latents.

**Claimed but unablated.** That circuit faithfulness scores reflect the model rather than the experimenter's choices. Miller, Chughtai, Saunders, *Transformer circuit faithfulness metrics are not robust* (COLM, 2024) show the reported faithfulness of published circuits (including IOI) moves substantially with ablation type and with how much of the task-irrelevant computation is held fixed — the score is a property of (circuit, ablation distribution), reported as if a property of the circuit.

**Benchmark-only results.** ROAR (Hooker et al., NeurIPS 2019) rankings, deletion/insertion AUCs, and pointing-game scores are all benchmark numbers whose ordering of methods is metric-dependent and does not transfer.

## 4. What Is Known

- **Some methods are causally inert.** Adebayo et al. (NeurIPS 2018): Guided Backprop and Guided Grad-CAM saliency maps are visually and quantitatively near-unchanged after randomizing the weights of Inception v3 on ImageNet — rank correlations close to 1 against the trained model. Sixt, Granz, Landgraf (ICML 2020) prove why: modified-backprop methods converge to a class-*insensitive* map because the product of positive-projected matrices loses class information.
- **Deletion metrics are confounded by distribution shift.** ROAR, on ImageNet/ResNet-50 and Birdsnap/Food101: after retraining, removing the top 10–90% of pixels ranked by gradients, IG, or gradient$\times$input degrades accuracy no more than removing random pixels; only SmoothGrad-Squared and VarGrad beat the random control. Scale: ImageNet-1k, ResNet-50, full retrain per fraction (hundreds of training runs).
- **Metric disagreement is the norm.** Han, Srinivas, Lakkaraju (NeurIPS 2022) show LIME/SHAP/IG/SmoothGrad are all local function approximators differing only in neighbourhood; which one wins depends on the evaluation neighbourhood, which is a free parameter.
- **Attention weights are not attributions.** Jain & Wallace (NAACL 2019) find adversarial attention distributions with near-identical predictions; Wiegreffe & Pinter (EMNLP 2019) show the conclusion depends on what "explanation" is asked to mean — the disagreement is definitional, not empirical.
- **Patching works but is fragile.** Zhang & Nanda (ICLR 2024) show corrupted-vs-clean direction and the choice of metric (logit difference vs probability) change which components appear causal in GPT-2 small IOI. Makelov, Lange, Nanda (ICLR 2024) exhibit an *interpretability illusion*: subspace activation patching can produce a large, apparently mechanistic effect via dormant pathways not used on the clean distribution.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no ablation-distribution-invariant definition of the causal contribution of a component. Every measured $\mathrm{IE}$ is conditional on $\mathcal{D}$, and no principle selects $\mathcal{D}$. Until this is fixed, "faithfulness" numbers are not comparable across papers.
- **Theoretically open.** Whether the Bilodeau impossibility extends to *non-complete* or non-linear attributions, and whether any tractable attribution class admits a positive certification theorem ("score $>\tau$ implies bounded error on interventional effects"). No proof either way.
- **Theoretically open.** Whether a unique minimal faithful circuit exists for a given task and tolerance, or whether the set of $\epsilon$-faithful subgraphs is exponentially large and mutually inconsistent (non-identifiability under superposition).
- **Empirically open.** Whether attribution-patching fidelity degrades with scale. Published AtP/AtP\* validation is concentrated at $\le 70$B and mostly on small transformers; the correlation between first-order estimates and exhaustive patching at $\ge 100$B on multi-token tasks has not been measured because the exhaustive arm is expensive, not because it is impossible.
- **Empirically open.** Whether self-repair magnitude grows or shrinks with model scale; this determines whether single-node attribution becomes more or less misleading in large models.

## 6. Why It Is Hard

The obstruction is **an evaluation that does not measure the thing it names, because the ground truth is defined by the evaluation's own free parameter.** $\mathrm{IE}(v;\mathcal{D})$ is the reference signal, and $\mathcal{D}$ is chosen by the experimenter. Zero ablation measures OOD sensitivity; mean ablation measures deviation from a task average that the model never sees; resample ablation measures effects relative to a counterfactual set whose construction encodes the hypothesis being tested. A method can be tuned to score well under one $\mathcal{D}$ and fail under another, and there is no third signal to adjudicate.

Compounding it: **self-repair** breaks the additivity that would let single-node effects compose into a circuit-level claim, and **superposition** means the node basis itself is a modelling choice. Absent ground truth is only partly the issue — synthetic models with known mechanisms exist, but they are the models where every method already works.

## 7. Current Research (as of 2026)

- **Causal abstraction as the formal frame.** Geiger, Icard, Potts and collaborators: distributed alignment search and interchange-intervention accuracy define faithfulness as the existence of a homomorphism from the network to a high-level causal model. This gives a principled target; the cost is that the high-level model must be specified in advance.
- **SAE-based node bases.** Sparse feature circuits and successors replace neurons with learned sparse latents to make the decomposition less arbitrary. *(frontier — verify)* Evidence that SAE circuits are more faithful than neuron circuits under matched ablation budgets remains contested; several 2025 evaluations report SAE latents underperforming simple baselines on downstream causal tasks.
- **Robustness of faithfulness metrics.** Follow-ups to Miller et al. (2024) pushing for reporting faithfulness curves across ablation distributions rather than a single number. *(frontier — verify)*
- **Scalable patching.** DeepMind (AtP\* line), Redwood/ARC-adjacent groups, and EleutherAI on cheap estimators with error bars rather than point estimates.

## 8. Concrete Next Experiment

**Question.** Is a published attribution method's ranking of components stable across the ablation distributions that all count as legitimate?

- **Scale.** GPT-2 small (124M) and Pythia-2.8B, three tasks with established circuits: IOI, greater-than, docstring. Node set: all 144 (resp. all) attention heads and MLPs. $n=1000$ paired prompts per task.
- **Arms.** Compute exhaustive $\mathrm{IE}$ per node under four intervention distributions: zero, mean-over-task, resample-from-paired-counterfactual, and resample-from-unrelated-corpus. Compute attribution patching and AtP\* estimates under each.
- **Control arm.** A random attribution $\phi_{\text{rand}}$, and a *permuted-node* control that shuffles true $\mathrm{IE}$ values across nodes — this fixes the marginal distribution of scores and isolates ranking information.
- **Deciding number.** The minimum pairwise Spearman $\rho$ of $\mathrm{IE}$ rankings across the four $\mathcal{D}$'s, restricted to the top-30 nodes. If $\min \rho \ge 0.8$, ablation choice is second-order and single-number faithfulness reporting is defensible. If $\min \rho \le 0.4$ — the outcome the Miller et al. results predict — then no published faithfulness score means anything without its $\mathcal{D}$, and the field should report the four-point curve as the primary artifact.
- **Cost.** $\approx 4 \times |V| \times n$ forward passes per task; hours on one A100 at GPT-2 scale, low tens of GPU-hours at 2.8B.

## 9. Key References

- **[Foundational]** M. Sundararajan, A. Taly, Q. Yan. *Axiomatic Attribution for Deep Networks.* ICML, 2017. — arXiv:1703.01365
- **[Foundational]** S. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS, 2017. — arXiv:1705.07874
- **[Foundational]** J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, B. Kim. *Sanity Checks for Saliency Maps.* NeurIPS, 2018. — arXiv:1810.03292
- **[Foundational]** S. Hooker, D. Erhan, P.-J. Kindermans, B. Kim. *A Benchmark for Interpretability Methods in Deep Neural Networks.* NeurIPS, 2019. — arXiv:1806.10758
- **[SOTA/theory]** B. Bilodeau, N. Jaques, P. W. Koh, B. Kim. *Impossibility Theorems for Feature Attribution.* PNAS, 2024. — arXiv:2212.11870
- **[SOTA]** L. Sixt, M. Granz, T. Landgraf. *When Explanations Lie: Why Many Modified BP Attributions Fail.* ICML, 2020. — arXiv:1912.09818
- **[SOTA]** A. Conmy, A. Mavor-Parker, A. Lynch, S. Heimersheim, A. Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS, 2023. — arXiv:2304.14997
- **[SOTA]** J. Kramár, T. Lieberum, R. Shah, N. Nanda. *AtP\*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components.* 2024. — arXiv:2403.00745
- **[SOTA]** K. Wang, A. Variengien, A. Conmy, B. Shlegeris, J. Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR, 2023. — arXiv:2211.00593
- **[Critique]** J. Miller, B. Chughtai, W. Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM, 2024. — arXiv:2407.08734
- **[Critique]** A. Makelov, G. Lange, N. Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR, 2024. — arXiv:2311.17030
- **[Critique]** T. McGrath, M. Rahtz, J. Kramár, V. Mikulik, S. Legg. *The Hydra Effect: Emergent Self-repair in Language Model Computations.* 2023. — arXiv:2307.15771
- **[Method]** A. Syed, C. Rager, A. Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* 2023. — arXiv:2310.10348
- **[Method]** S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, A. Mueller. *Sparse Feature Circuits.* 2024. — arXiv:2403.19647
- **[Survey]** F. Zhang, N. Nanda. *Towards Best Practices of Activation Patching in Language Models.* ICLR, 2024. — arXiv:2309.16042
- **[Survey]** T. Han, S. Srinivas, H. Lakkaraju. *Which Explanation Should I Choose? A Function Approximation Perspective.* NeurIPS, 2022. — arXiv:2206.01254

## 10. Worked Example

Take GPT-2 small on IOI: "When Mary and John went to the store, John gave a drink to ___". Readout $f$ = logit(Mary) − logit(John), about $+3.5$ nats on clean prompts.

Consider Name Mover Head L9H9, the component with the largest reported direct effect.

- **Resample ablation** (patch in activations from the ABC counterfactual, where the third name breaks the pattern): $\Delta f \approx -1.5$ nats. Large — the head looks essential.
- **Zero ablation**: $\Delta f$ is larger still, but part of that is the residual stream being pushed off-distribution; the same zero-ablation applied to a head with no IOI role also moves $f$, so the number is not attributable to mechanism.
- **The self-repair correction.** Ablate L9H9 and re-measure the direct effect of Negative Name Mover L10H7. Its contribution *shifts in the compensating direction*, recovering a substantial fraction of the ablated effect (McGrath et al., 2023). So $\mathrm{IE}(\text{L9H9})$ measured alone understates the head's role in the intact model, while the sum $\sum_v \mathrm{IE}(v)$ over the 26-head circuit overstates the joint effect — the two errors have opposite sign and do not cancel.
- **The attribution-patching estimate.** Gradient$\times$activation on the same node gives a number correlated with the resample-ablation value across heads, but underestimates heads whose attention pattern is saturated: the softmax gradient is near zero while the actual patch flips the pattern. AtP\* exists precisely to patch this failure.

**Where the obstruction becomes visible.** Three defensible protocols give three different $\mathrm{IE}$ values for L9H9 — differing by more than the gap between L9H9 and the next-ranked head under some of them. The "faithfulness = 87% of logit difference recovered" style of claim is therefore a statement about a (circuit, $\mathcal{D}$, readout) triple. Report the triple, or report nothing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*