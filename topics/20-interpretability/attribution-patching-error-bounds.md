---
id: 20-interpretability/attribution-patching-error-bounds
title: "Attribution Patching Approximation Error"
topic: 20-interpretability
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attribution Patching Approximation Error

> **Topic:** Interpretability · **ID:** `20-interpretability/attribution-patching-error-bounds` · **Status:** partially-solved

## 1. Problem Statement

Activation patching measures the causal effect of a model component by replacing its activation with one from a corrupted run and re-running the forward pass. It costs one forward pass per component, so a full sweep over a frontier model's nodes or edges is $10^5$–$10^8$ forward passes. Attribution patching (AtP) replaces that sweep with a first-order Taylor estimate: two forward passes and one backward pass give an approximate effect for *every* component at once.

The problem is the error of that approximation, in three variants that are routinely conflated:

- **Measurement.** Given a model, a task, and a corruption distribution, what is the distribution of the per-node error $\varepsilon(n)$, and which summary of it matters — Pearson/Spearman correlation with true effects, top-$k$ recall, or the faithfulness of the resulting circuit? These give different answers on the same run.
- **Method.** Can we cheaply detect and repair the cases where AtP fails, without paying the $|N|$-forward-pass cost that AtP exists to avoid? AtP*, GradDrop, and EAP-IG are partial answers.
- **Theory.** Is there any *a priori* bound on $|\varepsilon(n)|$ — computable from quantities available in the two passes AtP already runs — that is non-vacuous for a real transformer? No such bound exists. This is the open core.

Solving it means: a certificate, computed at $O(1)$ passes, that bounds the false-negative rate of an AtP-derived ranking at a stated confidence, validated against exhaustive patching at a scale where exhaustive patching is still feasible.

## 2. Formal Setting

Let $M$ be a transformer, $x_{\text{cl}}$ a clean prompt, $x_{\text{cr}}$ a corrupted counterpart, and $L$ a scalar metric — in practice the logit difference $L = z_{\text{correct}} - z_{\text{incorrect}}$, measured on the final position. Let $N$ be the set of patchable sites (attention head outputs, MLP outputs, residual-stream slices, SAE features, or edges in a computational graph). For $n \in N$, write $a_n \in \mathbb{R}^{d_n}$ for its clean activation and $a_n'$ for the corrupted one.

**True effect** (what AtP approximates), measured by one forward pass with a hook:
$$\mathcal{I}(n) \;=\; L\big(x_{\text{cl}} \,\|\, \mathrm{do}(a_n \!=\! a_n')\big) \;-\; L(x_{\text{cl}}).$$

**AtP estimate**, measured from one clean forward, one corrupted forward, one clean backward:
$$\hat{\mathcal{I}}(n) \;=\; (a_n' - a_n)^{\!\top} \nabla_{a_n} L\big(x_{\text{cl}}\big).$$

**Error**, the Taylor remainder along the segment $a_n(t) = a_n + t\,\Delta_n$, $\Delta_n = a_n' - a_n$:
$$\varepsilon(n) \;=\; \mathcal{I}(n) - \hat{\mathcal{I}}(n) \;=\; \tfrac{1}{2}\,\Delta_n^{\!\top} \nabla^2_{a_n} L\big(a_n(\xi)\big)\, \Delta_n, \quad \xi \in (0,1).$$

**Integrated-gradients variant** (EAP-IG), $m$ steps, cost $m$ backward passes:
$$\hat{\mathcal{I}}_{\text{IG}}(n) = \Delta_n^{\!\top} \cdot \frac{1}{m}\sum_{k=1}^{m} \nabla_{a_n} L\big(a_n + \tfrac{k}{m}\Delta_n\big),$$
which is exact as $m \to \infty$ for the *single-node* effect, and still biased for joint effects.

**Reported quantities.** Rank agreement is Spearman $\rho$ between $\{\hat{\mathcal{I}}(n)\}$ and $\{\mathcal{I}(n)\}$ over $N$. Recall at $k$ is $|\mathrm{top}_k(\hat{\mathcal{I}}) \cap \mathrm{top}_k(\mathcal{I})|/k$. Circuit faithfulness is $\big(L(C) - L(\emptyset)\big)/\big(L(M) - L(\emptyset)\big)$ where $L(C)$ is the metric with all nodes outside circuit $C$ mean-ablated or patched.

**Assumptions, and which are violated.**
1. *Local linearity of $L$ in $a_n$ over $[a_n, a_n']$.* Violated routinely: $\|\Delta_n\|$ is a full clean-vs-corrupted difference, not infinitesimal.
2. *Non-saturated attention.* Violated whenever a head's softmax is near one-hot: $\nabla \approx 0$ while the true effect is large. This is the dominant false-negative mechanism.
3. *No internal cancellation.* Violated when a node's gradient contributions through different downstream paths have opposite sign and cancel, hiding a real effect.
4. *Additivity across nodes.* Violated whenever the circuit contains backup/self-repair behaviour; the sum of single-node AtP scores is not the joint effect.
5. *Frozen LayerNorm/RMSNorm denominator.* An implementation choice, not a property of the model; different codebases make different choices and the resulting scores differ.

## 3. State of the Art

**Method SOTA.** *AtP\** (Kramár, Lieberum, Shah, Nanda, 2024, arXiv:2403.00745) is the strongest node-level estimator: it adds a **QK-fix** (recompute attention softmax with the corrupted query, rather than trusting the gradient through a saturated softmax) and **GradDrop** (compute a separate gradient with each downstream residual contribution zeroed, to expose cancelling paths). Both are targeted repairs for assumptions 2 and 3, and both are *established* — the paper ablates each fix and shows each recovers a distinct set of false negatives.

**Edge SOTA.** *EAP* (Syed, Rager, Conmy, 2023, arXiv:2310.10348; BlackboxNLP 2024) applies the same linearization to edges of the computational graph and recovers circuits comparable to ACDC (Conmy et al., NeurIPS 2023) at a small constant number of passes rather than one per edge. *EAP-IG* (Hanna, Pezzelle, Belinkov, COLM 2024, arXiv:2403.17806) shows plain EAP circuits can be *unfaithful* even when they overlap heavily with a known reference circuit, and that a handful of integrated-gradient steps repairs faithfulness.

**Claimed but unablated.** The claim that AtP-derived rankings are adequate for SAE-feature graphs at the scale of frontier models rests on downstream plausibility, not on comparison against exhaustive patching — exhaustive patching over $10^6$ features has not been run. Sparse Feature Circuits (Marks et al., 2024, arXiv:2403.19647) uses a linear approximation for exactly this reason and validates by intervention on a handful of circuits, not by error measurement.

**Benchmark-number-only results.** Reported Spearman correlations and top-$k$ recalls for AtP are tied to a specific model, task, corruption scheme, and metric. There is no cross-paper protocol; numbers are not comparable across papers.

## 4. What Is Known

- AtP's cost is $\Theta(1)$ passes versus $\Theta(|N|)$ for exhaustive patching. On Pythia-12B the node set is $\sim\!10^4$–$10^5$ per prompt position, so the saving is 4–5 orders of magnitude (AtP*, 2024).
- Two failure modes are identified, reproduced, and separately fixable: **attention softmax saturation** (false negatives at heads with near-one-hot attention) and **cancellation across downstream paths** (fixed by GradDrop). Measured on Pythia models from 410M to 12B.
- AtP* Pareto-dominates AtP, and both dominate a **subsampling** baseline (random subsets patched jointly, effects attributed by regression) on the cost-versus-recall curve; subsampling is the honest control because it is unbiased in expectation.
- On IOI, greater-than, and docstring tasks in GPT-2 small (117M) and Pythia-1.4B, EAP recovers circuits with node overlap comparable to ACDC at roughly two orders of magnitude less compute (Syed et al., 2023).
- High overlap with a reference circuit does not imply faithfulness: EAP circuits on the greater-than task can score high overlap and low faithfulness simultaneously (Hanna et al., COLM 2024). EAP-IG with a small number of steps ($m \approx 5$) closes much of that gap.
- Faithfulness itself is fragile: Miller, Chughtai, Saunders (COLM 2024, arXiv:2407.08734) show circuit faithfulness scores swing substantially with ablation choice (mean vs. resample vs. zero) and with which nodes are counted as "outside" the circuit. Any error metric built on faithfulness inherits that fragility.
- The error is *signed and structured*, not noise: AtP under-reports saturated heads and mis-signs nodes with cancelling paths. It is not well modelled as isotropic error around the truth.

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous, computable bound on $|\varepsilon(n)|$ exists. The Lagrange remainder requires $\sup_{t}\|\nabla^2 L(a_n(t))\|$ along the segment; for a transformer with saturating softmaxes and normalization layers this is neither bounded a priori nor cheaply estimable. Whether a bound exists that is (a) computable in $O(1)$ passes and (b) tighter than "the effect could be anything up to the full metric range" is unproven either way.
- **Theoretically open.** Whether AtP false negatives can be *certified absent* — i.e. whether some cheap statistic implies "no node outside the top $k$ has true effect $> \tau$".
- **Empirically open.** The error distribution has never been measured exhaustively above ~1B parameters. Running all $|N|$ patches on a 7B model over a few hundred prompts is feasible on a modest cluster; nobody has published it.
- **Empirically open.** How the error scales with model size, with $\|\Delta_n\|$, and with SAE feature sparsity. Plausible hypotheses (error grows with $\|\Delta_n\|^2$; error shrinks for sparse features because $\Delta$ is smaller) are untested at scale.
- **Methodologically blocked.** There is no agreed definition of "the" ground-truth effect. Single-node patching, joint patching, mean-ablation, and resample-ablation give different $\mathcal{I}(n)$ for the same node. Until the target is pinned, "approximation error" is not a well-posed quantity — this is the deepest blocker, and it is why the status is *partially-solved* rather than *empirically-open*.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth compounded by non-identifiability of the target**. AtP is validated against exhaustive activation patching, but exhaustive patching is itself a *choice of intervention*, not a ground truth: the effect of a node depends on what the rest of the model is doing, and mean-ablation, zero-ablation, and resample-ablation of the same node yield different numbers with different signs in known cases. So the "error" being bounded is error against a moving reference.

The second obstruction is **compute cost at exactly the scale where the answer changes**. The failure modes AtP* identifies (softmax saturation, path cancellation) get *more* prevalent as models get deeper and heads specialize — which is where exhaustive validation becomes infeasible. The regime where you can measure the error is not the regime where you need the estimator.

The third is that **the standard evaluation does not measure what it names**: circuit-overlap-with-a-known-circuit is reported as evidence that attribution is accurate, but Hanna et al. showed overlap and faithfulness dissociate.

## 7. Current Research (as of 2026)

- **Google DeepMind (Nanda's mechanistic interpretability team, Kramár, Lieberum).** AtP* and successors; extension of the QK-fix/GradDrop machinery to SAE-feature graphs at frontier scale *(frontier — verify)*.
- **Anthropic interpretability.** Attribution graphs over cross-layer transcoders, with linear attribution used as the graph-construction primitive; error is handled by intervention spot-checks rather than by bounds.
- **Hanna, Belinkov, and collaborators (Technion / Amsterdam).** Faithfulness-first evaluation of edge attribution; EAP-IG and successors.
- **Bau Lab / Northeastern, and Marks et al.** Sparse feature circuits; error control by ablation validation on discovered circuits.
- **Edge pruning and optimization-based circuit discovery** (Princeton/Bhaskar et al.) as an alternative that sidesteps linearization by directly optimizing a discrete mask — expensive per task, but supplies a *better reference* against which AtP error can be measured *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** What fraction of genuinely high-effect nodes does AtP\* miss, at 7B scale, and is that fraction predicted by a cheap statistic?

**Scale.** Pythia-6.9B or Llama-3-8B. 300 prompts from three tasks (IOI-style name resolution, greater-than, a factual-recall task). Node set: all attention head outputs and MLP outputs at all positions, $|N| \approx 5\times10^4$ per prompt. Exhaustive resample-ablation patching: $|N| \times 300 \approx 1.5\times10^7$ forward passes, batched — roughly a few thousand A100-hours. This is the whole cost of the experiment and it is affordable.

**Arms.**
1. AtP (plain first-order).
2. AtP\* (QK-fix + GradDrop).
3. EAP-IG with $m = 5$.
4. **Control arm: subsampling** at matched compute — random node subsets patched jointly, effects recovered by ridge regression. This is unbiased, so it isolates linearization bias from sampling noise.
5. Second control: exhaustive patching under a *different* ablation (mean instead of resample), to quantify how much the "ground truth" itself moves.

**Deciding number.** **False-negative rate at $k = 100$: the fraction of nodes in the true top 100 (by $|\mathcal{I}|$, resample-ablation) that AtP\* ranks outside its top 500.** Report it alongside the same rate computed between the two ablation schemes — call that $r_{\text{gt}}$, the irreducible floor. If AtP\*'s false-negative rate is below $r_{\text{gt}}$, the linearization error is smaller than the ambiguity in the target and the method problem is effectively closed at this scale. If it exceeds $2 r_{\text{gt}}$, linearization is the binding error and the theory variant matters.

**Secondary output.** Regress $|\varepsilon(n)|$ on cheap per-node statistics — $\|\Delta_n\|$, attention entropy, GradDrop variance across drops. An $R^2 > 0.5$ would give the first empirical error predictor, which is the practical substitute for the missing bound.

## 9. Key References

- **[Foundational]** Neel Nanda. *Attribution Patching: Activation Patching at Industrial Scale.* Blog post, 2023. — the original proposal; not peer-reviewed.
- **[Foundational]** Mukund Sundararajan, Ankur Taly, Qiqi Yan. *Axiomatic Attribution for Deep Networks.* ICML, 2017. — arXiv:1703.01365
- **[Foundational]** Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS, 2023. — arXiv:2304.14997
- **[SOTA]** János Kramár, Tom Lieberum, Rohin Shah, Neel Nanda. *AtP\*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components.* 2024. — arXiv:2403.00745
- **[SOTA]** Aaquib Syed, Can Rager, Arthur Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* BlackboxNLP, 2024. — arXiv:2310.10348
- **[SOTA]** Michael Hanna, Sandro Pezzelle, Yonatan Belinkov. *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms.* COLM, 2024. — arXiv:2403.17806
- **[Related]** Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller. *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* 2024. — arXiv:2403.19647
- **[Critique]** Joseph Miller, Bilal Chughtai, William Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM, 2024. — arXiv:2407.08734
- **[Survey]** Stefan Heimersheim, Neel Nanda. *How to Use and Interpret Activation Patching.* 2024. — arXiv:2404.15255

## 10. Worked Example

Take a single attention head $h$ in GPT-2 small on the IOI task, on a prompt where the corruption swaps the indirect object. Suppose $h$ is a name-mover head whose softmax over source positions is near one-hot: attention weight $0.97$ on the correct name token, $0.01$ elsewhere.

**True effect.** Patching $h$'s output with the corrupted-run output moves the logit difference from $+3.4$ to $+0.6$, so $\mathcal{I}(h) = -2.8$ — a large, real effect.

**AtP estimate.** The gradient $\nabla_{a_h} L$ is computed through the clean forward pass. Because AtP's linearization propagates through the softmax at its clean operating point, and the softmax Jacobian at a near-one-hot distribution has entries of order $p(1-p) = 0.97 \times 0.03 \approx 0.029$, the gradient signal routed through the attention pattern is suppressed by roughly $30\times$ relative to the unsaturated case. If the direct-path term is small, $\hat{\mathcal{I}}(h)$ can come out at $-0.15$.

**Error.** $\varepsilon(h) = -2.8 - (-0.15) = -2.65$, i.e. AtP recovers about 5% of the true effect. In a ranked list over $5\times10^4$ nodes, a node with true rank 3 lands somewhere past rank 1,000. Nothing in the two passes AtP ran flags this: the estimate is small, the gradient is small and smooth, and there is no residual to inspect.

**What the fix costs.** AtP\*'s QK-fix recomputes the attention softmax using the corrupted query for this head, at the cost of one extra partial forward through the attention layers — a small constant, not $|N|$. Doing that here recovers an estimate near $-2.5$, and the node returns to the top of the ranking.

**Where the obstruction shows.** The fix works because we *knew* which failure mode to look for. Nothing in the procedure certifies that no *other* head is similarly hidden by a mechanism nobody has named yet. Confirming absence still requires the $|N|$ forward passes AtP was built to avoid — and even then, running the same patch with mean-ablation instead of resample-ablation gives $\mathcal{I}(h) = -1.9$ rather than $-2.8$, a 32% shift in the very number the error is measured against. The bound is missing and the target moves.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*