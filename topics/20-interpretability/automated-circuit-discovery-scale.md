---
id: 20-interpretability/automated-circuit-discovery-scale
title: "Automated Circuit Discovery at Frontier Scale"
topic: 20-interpretability
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Automated Circuit Discovery at Frontier Scale

> **Topic:** Interpretability · **ID:** `20-interpretability/automated-circuit-discovery-scale` · **Status:** empirically-open

## 1. Problem Statement

**Input:** a trained transformer $M$, a task distribution $\mathcal{D}$ of paired prompts $(x, x')$ that differ in one task-relevant feature, and a behavioral metric $m$ (logit difference, KL to the full model).

**Output:** a subgraph $C$ of $M$'s computational graph that (a) reproduces $M$'s behavior on $\mathcal{D}$ when the rest of the graph is ablated, (b) is much smaller than $M$, and (c) admits a human-checkable description of what each retained component does.

Three variants, routinely conflated:

- **Measurement.** Given a candidate $C$, decide whether it is faithful. Blocked on ablation choice: the "faithfulness" number moves by tens of percentage points depending on the counterfactual distribution.
- **Method.** Search the $10^7$–$10^9$ edge space for a minimal faithful $C$ within a compute budget sublinear in $|E|$. This is where automation lives.
- **Theory.** Does a small faithful $C$ exist at all for a given behavior, or is the computation spread over a dense, low-rank-superposed graph with no sparse subgraph? Open.

**Solved would mean:** for a $\geq 70$B-parameter model and a behavior not chosen by the method's authors, an automated pipeline returns a circuit of $\leq 10^{-3}$ of the graph that (i) holds $\geq 90\%$ of the metric under ablation of the complement, (ii) survives a *different* ablation distribution than the one it was fit on, and (iii) supports a prediction about out-of-distribution behavior that is confirmed.

## 2. Formal Setting

Let $M$ have computational graph $G=(V,E)$. Nodes $V$ are writers to and readers from the residual stream: embeddings, attention heads (one writer, three readers $q,k,v$), MLPs, unembedding. For a model with $L$ layers and $H$ heads, $|V_{\text{write}}| = LH + L + 1$, $|V_{\text{read}}| = 3LH + L + 1$, and $|E| \approx \tfrac12 |V_{\text{write}}||V_{\text{read}}|$ under the causal-order constraint.

A circuit is $C \subseteq E$. Ablation replaces the activation on each $e \notin C$ with a value from a corrupt distribution: zero ($a_e \!\leftarrow\! 0$), mean ($a_e \!\leftarrow\! \mathbb{E}_{\mathcal{D}}[a_e]$), or resample ($a_e \!\leftarrow\! a_e(x')$). Write $M_{C}$ for the ablated model.

**Faithfulness, as measured:**
$$F(C) \;=\; \frac{\mathbb{E}_{(x,x')\sim\mathcal{D}}\,[\,m(M_C(x)) - m(M_\emptyset(x))\,]}{\mathbb{E}\,[\,m(M(x)) - m(M_\emptyset(x))\,]}$$
with $m$ typically the logit difference between the correct and counterfactual token. $F=1$ is exact recovery; $F>1$ means the complement *hurt* the behavior.

**Edge attribution** (the linear surrogate that makes search tractable): for edge $e$ with activation $a_e$,
$$\hat{\Delta}_e \;=\; \big(a_e(x') - a_e(x)\big)^{\!\top} \nabla_{a_e} m\big(M(x)\big),$$
a first-order estimate of the true ablation effect $\Delta_e = m(M) - m(M_{E\setminus\{e\}})$. All $|E|$ values of $\hat\Delta_e$ come from one forward and one backward pass. **EAP-IG** replaces the single gradient with an integrated gradient over $k$ interpolation steps between $a_e(x)$ and $a_e(x')$.

**Assumptions, and where they break:**

1. *Linearity of the ablation effect* ($\hat\Delta_e \approx \Delta_e$). Violated at attention softmax saturation and at layer-norm, where $\nabla_{a_e} m \approx 0$ while $\Delta_e$ is large — the known failure mode that motivated EAP-IG.
2. *Edge independence* (effects are additive, so top-$k$ by $\hat\Delta$ is near-optimal). Violated whenever components back up for one another; ablating one head causes another to take over.
3. *Off-distribution ablations are neutral.* Violated: zero and mean ablation push the residual stream off the data manifold, so $F$ measures both the circuit and the model's OOD response.
4. *The node basis is the right basis.* Heads and MLPs are architectural, not functional; feature-level methods (SAEs, transcoders) replace them but add a lossy approximation whose error is not accounted for in $F$.

## 3. State of the Art

**Established.**
- **ACDC** (Conmy et al., NeurIPS 2023) — greedy recursive edge pruning, one ablation forward pass per edge, $O(|E|)$ cost. Recovers a large fraction of the hand-found IOI circuit in GPT-2 small; ROC AUC against ground-truth circuits is well below 1 on every task tested.
- **EAP / attribution patching** (Syed, Rager, Conmy, BlackboxNLP 2024) — matches or beats ACDC's node/edge recovery at a cost of 2 forward + 1 backward pass total, independent of $|E|$. This is the result that made scaling conceivable.
- **EAP-IG** (Hanna, Pezzelle, Belinkov, COLM 2024) — integrated gradients over $k \approx 5$–$10$ steps; recovers higher-faithfulness circuits than EAP at the same asymptotic cost, and shows edge *overlap* with a reference circuit is a poor proxy for faithfulness.
- **Edge Pruning** (Bhaskar, Wettig, Friedman, Chen, NeurIPS 2024) — gradient-based optimization of continuous edge masks with an $L_0$ penalty. Scaled to a 13B model (CodeLlama-13B), the largest published edge-level circuit discovery.

**Claimed but not independently ablated.**
- **Attribution graphs via cross-layer transcoders** (Ameisen, Lindsey et al., Transformer Circuits Thread, 2025) applied to Claude 3.5 Haiku. The companion "Biology" paper reports that the method yields an interpretable account for roughly a quarter of the prompts attempted — the honest number, but self-reported, and the CLT's reconstruction error is not folded into the faithfulness claim.
- **Sparse feature circuits** (Marks et al., ICLR 2025) demonstrated at Pythia-70M/GPT-2 scale with a downstream editing result (SHIFT); the editing gain has not been reproduced at $\geq 7$B.

**Benchmark-number-only.** **MIB** (Mueller et al., 2025) gives a circuit-localization leaderboard over GPT-2, Qwen-2.5, Gemma-2 and Llama-3.1-8B. Its headline finding — that gradient-based mask optimization beats greedy ACDC-style search — exists as a leaderboard ranking, not as a mechanism-level explanation of *why*.

## 4. What Is Known

- **Hand-found circuits exist and are partial.** The IOI circuit (Wang et al., ICLR 2023) is 26 heads of 144 in GPT-2 small ($\approx 18\%$ of heads); its own authors document backup name-mover heads that fire only when the primary heads are ablated.
- **Gradient approximation is cheap and lossy.** EAP costs $O(1)$ passes vs. ACDC's $O(|E|)$ — a $\sim\!10^4$–$10^7\times$ saving at GPT-2-small-to-70B scale — while EAP-IG's advantage over EAP shows the first-order error is not negligible.
- **Faithfulness is ablation-dependent.** Miller, Chughtai and Saunders (COLM 2024) show reported faithfulness for published GPT-2-small circuits swings materially with the choice of corrupt distribution and metric; some circuits score $F > 1$.
- **Scale ceiling.** Published edge-level discovery tops out at 13B parameters (Edge Pruning). Feature-level attribution graphs have run on a production-scale model (Claude 3.5 Haiku) but with pruned, hand-inspected graphs rather than an automated faithfulness sweep.
- **Circuits are task-local.** No published circuit found on one task has been shown to predict component roles on an unrelated task at $\geq 7$B.

## 5. What Is Not Known

- **Empirically open.** Does the EAP-IG / edge-pruning family retain its GPT-2-scale faithfulness at 70B+? Runnable today on 8×H100 for a single task; nobody has published the sweep. This is the core gap.
- **Empirically open.** Does a circuit found under resample ablation on $\mathcal{D}$ transfer to a held-out counterfactual distribution $\mathcal{D}'$? The experiment is trivial and almost never reported.
- **Methodologically blocked.** There is no agreed definition of faithfulness that is invariant to ablation choice. Until there is, cross-paper $F$ numbers are not comparable, and "the circuit" is under-specified.
- **Theoretically open.** No result says whether a behavior implemented in superposition admits *any* sparse faithful subgraph. Non-identifiability is unproven either way: multiple disjoint $C$ with $F \approx 1$ are observed empirically (backup heads), but no theorem characterizes when the minimal faithful circuit is unique.

## 6. Why It Is Hard

**Primary obstruction: absent ground truth compounded by a confounded measurement.** There is no gold circuit at frontier scale, so $F$ is the only score — and $F$ is a function of the ablation distribution, which the researcher chooses. A method can be tuned until $F$ is high without the circuit being the mechanism; backup components guarantee that ablating the complement of a *wrong* circuit can still leave behavior intact.

**Secondary obstruction: compute is superlinear in the wrong place.** Discovery is cheap ($O(1)$ passes with attribution); *validation* is not. Each candidate threshold requires an ablated forward pass over the dataset, and the Pareto sweep over thresholds is where the budget goes — see §10.

**Tertiary: basis mismatch.** Head-level graphs are the wrong granularity for polysemantic components; feature-level graphs replace the model with an approximation whose reconstruction error is not in the faithfulness denominator.

## 7. Current Research (as of 2026)

- **Anthropic interpretability team** — cross-layer transcoders and attribution graphs on production models; graphs released via Neuronpedia for external audit.
- **Northeastern (Bau) / Technion (Belinkov) / Mueller** — MIB as a shared benchmark; the circuit-localization track is the current coordination point.
- **Princeton (Chen) and collaborators** — mask-optimization circuit discovery, scaling past 13B *(frontier — verify)*.
- **Apollo Research / Goodfire** — parameter-space decomposition as an alternative to activation-space circuits, aiming at ablation-free identifiability *(frontier — verify)*.
- **EleutherAI / open replication efforts** — reproducing attribution-graph pipelines on open weights (Gemma-2, Llama-3.1) *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does attribution-based circuit discovery survive an ablation-distribution shift at scale?

- **Scale:** Llama-3.1-70B, one task with an unambiguous counterfactual (greater-than / year comparison), $n=500$ prompt pairs, 8×H100 for $\sim\!3$ days.
- **Method arm:** EAP-IG with $k=10$, sweep the edge threshold to produce circuits at $|C| \in \{10^3, 10^4, 10^5\}$ edges, fit under *resample* ablation from distribution $\mathcal{D}$.
- **Control arm (essential):** identical pipeline with $\hat\Delta_e$ replaced by $|\hat\Delta_e|$-matched random edges — same size, same layer histogram, random selection. This controls for "any sufficiently large subgraph is faithful."
- **The deciding number:** the transfer gap
$$G \;=\; F_{\mathcal{D}}(C) - F_{\mathcal{D}'}(C)$$
where $\mathcal{D}'$ is a *different* counterfactual family (e.g. corrupt the subject noun instead of the year). Report $G$ at $|C| = 10^4$.

**Decision rule:** if $G < 0.1$ and $F_{\mathcal{D}'}(C) - F_{\mathcal{D}'}(\text{random}) > 0.3$, automated discovery scales and the field should move to 70B by default. If $G > 0.3$, published circuits are artifacts of their fitting distribution and the measurement problem must be solved before any further scaling.

## 9. Key References

- **[Foundational]** Elhage, Nanda, Olsson, et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
- **[Foundational]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023. — arXiv:2211.00593
- **[Foundational]** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023. — arXiv:2304.14997
- **[SOTA]** Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* BlackboxNLP @ EMNLP 2024. — arXiv:2310.10348
- **[SOTA]** Hanna, Pezzelle, Belinkov. *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms.* COLM 2024.
- **[SOTA]** Bhaskar, Wettig, Friedman, Chen. *Finding Transformer Circuits with Edge Pruning.* NeurIPS 2024.
- **[SOTA]** Marks, Rager, Michaud, Belinkov, Bau, Mueller. *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* ICLR 2025. — arXiv:2403.19647
- **[SOTA]** Dunefsky, Chlenski, Nanda. *Transcoders Find Interpretable LLM Feature Circuits.* NeurIPS 2024. — arXiv:2406.11944
- **[SOTA]** Ameisen, Lindsey, Pearce, et al. *Circuit Tracing: Revealing Computational Graphs in Language Models.* Transformer Circuits Thread, 2025.
- **[SOTA]** Lindsey, Gurnee, Ameisen, et al. *On the Biology of a Large Language Model.* Transformer Circuits Thread, 2025.
- **[Critique]** Miller, Chughtai, Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM 2024.
- **[Benchmark]** Mueller et al. *MIB: A Mechanistic Interpretability Benchmark.* ICML 2025.
- **[Survey]** Heimersheim, Nanda. *How to Use and Interpret Activation Patching.* 2024. — arXiv:2404.15255

## 10. Worked Example

**Graph size, GPT-2 small vs. Llama-3.1-70B.**

| | $L$ | $H$ | writers | readers | $|E| \approx$ |
|---|---|---|---|---|---|
| GPT-2 small | 12 | 12 | 157 | 445 | $3.5\times10^4$ |
| Llama-3.1-70B | 80 | 64 | 5,201 | 15,441 | $4.0\times10^7$ |

**ACDC cost at 70B.** One ablated forward pass on a 100-token sequence costs $\approx 2 \times 7\times10^{10} \times 100 = 1.4\times10^{13}$ FLOP. Over $n=100$ prompts and $4.0\times10^7$ edges:
$$1.4\times10^{13} \times 100 \times 4.0\times10^{7} \;=\; 5.6\times10^{22}\ \text{FLOP}.$$
At an effective $4\times10^{14}$ FLOP/s per H100, that is $1.4\times10^{8}$ s $\approx$ **1,600 GPU-days for one circuit on one task**. ACDC is also inherently sequential, so it does not parallelize cleanly.

**EAP-IG cost at 70B.** $k=10$ integrated-gradient steps, forward+backward $\approx 3\times$ forward: $1.4\times10^{13} \times 3 \times 10 \times 100 = 4.2\times10^{17}$ FLOP $\approx$ **18 GPU-minutes**. Five orders of magnitude cheaper.

**Where the obstruction becomes visible.** The cheap number is not the whole cost. Attribution gives a *ranking*, not a circuit; you still sweep the threshold and ablation-test each candidate. Twenty thresholds × 100 prompts × 2 ablation distributions = 4,000 ablated forward passes $\approx 5.6\times10^{16}$ FLOP — still minutes. So compute is *not* the binding constraint at 70B; the binding constraint is that after those minutes you hold twenty subgraphs, each with an $F$ value, and no ground truth to say which one is the mechanism. If the $10^4$-edge circuit scores $F_{\mathcal{D}} = 0.94$ and the size-matched random control scores $0.71$ — a plausible outcome given GPT-2-scale results — the method has explained $23$ points of a $100$-point scale whose zero point is itself a modeling choice. That gap, not the FLOP count, is what §8 is designed to measure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*