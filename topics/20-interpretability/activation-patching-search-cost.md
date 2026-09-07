---
id: 20-interpretability/activation-patching-search-cost
title: "Activation Patching Hypothesis Space Explosion"
topic: 20-interpretability
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Activation Patching Hypothesis Space Explosion

> **Topic:** Interpretability · **ID:** `20-interpretability/activation-patching-search-cost` · **Status:** solved-but-impractical

## 1. Problem Statement

Activation patching answers a causal question by substitution: run the model on a clean input, run it on a corrupted input, copy one internal activation from one run into the other, and measure how much a task metric moves. The **algorithm is exact and settled**. The problem is that the hypothesis space — which activation, at which token, on which edge, in which basis — grows faster than any budget for evaluating it.

Three variants, with different difficulty:

- **Method variant (solved-but-impractical).** Given a model $M$, a task distribution of clean/corrupt pairs, and a metric, find the minimal subgraph whose patched behaviour reproduces $M$'s behaviour. Exhaustive search is well defined and correct; it is exponential in the number of edges, and even the greedy single-edge relaxation is $\Theta(|E|)$ forward passes per threshold.
- **Measurement variant (partly blocked).** Define "the effect of component $c$" so that the number is stable under the choice of corruption distribution, patch direction (noising vs. denoising), and token alignment. It is not currently stable under any of the three.
- **Theory variant (open).** Prove conditions under which a cheap first-order surrogate (attribution patching) has bounded error against the exact patch, or prove that no such condition holds for saturated attention.

A solution to the method variant is a search procedure whose cost is sub-linear in $|E|$ and whose recovered circuit is faithful under a pre-registered faithfulness test, at a scale of $\geq 10^{10}$ parameters.

## 2. Formal Setting

Let $M$ be a transformer with $L$ layers, $H$ heads per layer, and $T$ token positions. Its computation is a DAG $G=(V,E)$ over a residual-stream decomposition: source nodes are the embedding, the $LH$ attention heads and the $L$ MLPs; sink slots are each head's $q,k,v$ inputs, each MLP input, and the logits. Then

$$|V| = LH + L + 1, \qquad |E| \;\approx\; \frac{L(3H+1)\cdot LH}{2}.$$

For GPT-2 small ($L{=}12,H{=}12$): $|V|=157$, $|E|\approx 3.2\times 10^{4}$. Positional refinement multiplies by $T$.

Let $x$ be clean, $\tilde{x}$ corrupt, and $m:\mathbb{R}^{|\mathcal{V}|}\to\mathbb{R}$ the metric (usually logit difference between the correct and a distractor token). Write $M(x \mid \mathrm{do}(a_c \leftarrow v))$ for the run on $x$ with node $c$'s activation overwritten by $v$. The **indirect effect** of $c$ is measured as

$$\mathrm{IE}(c) \;=\; \mathbb{E}_{(x,\tilde x)}\Big[\, m\big(M(\tilde x \mid \mathrm{do}(a_c \leftarrow a_c(x)))\big) - m\big(M(\tilde x)\big) \Big],$$

the *denoising* direction. The *noising* direction swaps the roles of $x$ and $\tilde x$ and is not the negative of the above. Each $\mathrm{IE}(c)$ costs one forward pass per pair; with $N$ pairs, an exhaustive node sweep is $N|V|T$ passes and an exhaustive edge sweep $N|E|$.

**Attribution patching** replaces the sweep with a first-order Taylor expansion around the corrupt run:

$$\widehat{\mathrm{IE}}(c) \;=\; \big(a_c(x) - a_c(\tilde x)\big)^{\!\top} \nabla_{a_c} m\big(M(\tilde x)\big),$$

computable for all $c$ simultaneously in two forwards and one backward.

Faithfulness of a candidate circuit $C \subseteq E$ is measured by ablating $E \setminus C$ to a baseline and reporting the recovered fraction $m(M_C)/m(M)$.

**Assumptions, and which are violated.**
1. *Additivity/linearity of the residual stream in $m$.* Violated: attention softmax saturates, so $\nabla_{a_c} m \approx 0$ where the true patch effect is large.
2. *Independence of components.* Violated by self-repair — downstream heads compensate for ablated upstream heads (the Hydra effect, backup name-mover heads), so $\sum_c \mathrm{IE}(c) \neq \mathrm{IE}(\{c\})$.
3. *Corrupt run is on-distribution.* Violated by Gaussian-noise corruption, which pushes activations off the data manifold.
4. *Token alignment between $x$ and $\tilde x$.* Violated whenever clean and corrupt prompts differ in tokenisation length.

## 3. State of the Art

**Established.**
- *Exhaustive greedy edge search:* ACDC (Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso, NeurIPS 2023) prunes edges below a threshold $\tau$ in reverse topological order. Cost $\Theta(N|E|)$ forward passes per $\tau$. Correct, and the reference point everything else is measured against.
- *Gradient surrogate:* attribution patching (Nanda, 2023, blog) and edge attribution patching, EAP (Syed, Rager, Conmy, 2023; BlackboxNLP 2024) reduce the sweep to $O(1)$ passes — a $10^{4}$–$10^{5}\times$ speedup at GPT-2 scale.
- *Integrated-gradient correction:* EAP-IG (Hanna, Pezzelle, Belinkov, NeurIPS 2024) integrates the gradient along the clean→corrupt path, recovering faithfulness EAP loses at zero-gradient points, for $\sim$5–10× EAP's cost.
- *Optimisation instead of search:* Edge Pruning (Bhaskar, Wettig, Friedman, Chen, NeurIPS 2024) learns continuous edge masks by gradient descent, scaling to CodeLlama-13B — the largest published edge-level circuit discovery.

**Claimed but unablated.**
- That discovered circuits are *the* mechanism rather than *a* sufficient subgraph. Shi et al. (*Hypothesis Testing the Circuit Hypothesis in LLMs*, NeurIPS 2024) test this explicitly and find partial support only.
- That sparse-autoencoder feature graphs (Marks, Rager, Michaud, Belinkov, Mueller, Bau, 2024) reduce the search space in a way that survives feature-splitting. The claimed reduction is real; its stability under a different SAE seed is a benchmark number, not an ablation.

**Benchmark-number-only.** ACDC's headline comparison against subnetwork probing and head-importance ranking is an ROC-AUC table over five tasks (IOI, docstring, greater-than, induction, tracr). No method dominates across all five, and the ranking is sensitive to the ablation baseline.

## 4. What Is Known

- **Scale of the space.** GPT-2 small: $\approx 3.2\times 10^{4}$ edges, $157$ nodes; subset search is $2^{32000}$. Llama-3-70B ($L{=}80,H{=}64$): $\approx 4\times 10^{7}$ edges, $1.2\times 10^{3}\times$ GPT-2's count, with each forward $565\times$ more expensive.
- **Circuits are findable at small scale.** The IOI circuit in GPT-2 small (Wang, Variengien, Conmy, Shlegeris, Steinhardt, ICLR 2023) comprises 26 heads in 7 classes, found by hand-guided path patching over weeks, not by search.
- **Patch direction matters.** Zhang and Nanda (*Towards Best Practices of Activation Patching*, ICLR 2024) show that noising and denoising identify different component sets on the same task, and that Gaussian-noise corruption yields conclusions that symmetric-token corruption contradicts.
- **Faithfulness metrics are fragile.** Miller, Chughtai, Saunders (*Transformer Circuit Faithfulness Metrics Are Not Robust*, COLM 2024) show reported faithfulness of published circuits moves substantially with ablation-set and baseline choices that the papers treat as incidental.
- **Subspace patching can be illusory.** Makelov, Lange, Nanda (ICLR 2024) construct subspaces that produce a large, clean patching effect while being causally dormant in the unpatched model — activating a *different* pathway than the one named.
- **Self-repair is quantitatively large.** McGrath et al. (*The Hydra Effect*, 2023) find ablating an important attention head is compensated by downstream heads, so single-node effects systematically understate importance.

## 5. What Is Not Known

- **Theoretically open.** No bound on $|\widehat{\mathrm{IE}}(c) - \mathrm{IE}(c)|$ in terms of curvature of $m$ along the clean→corrupt segment. No proof that a minimal faithful subgraph is unique, or a counterexample construction showing the family of minimal faithful subgraphs has exponential size.
- **Empirically open.** Whether any $O(1)$-pass method's recovered circuit remains faithful at $\geq 70$B. Edge Pruning reached 13B; nothing edge-level has been validated above that. Runnable today; nobody has run it.
- **Methodologically blocked.** "The right corruption distribution" is undefined. Faithfulness is scored against an ablation baseline (zero, mean, resample) whose choice changes the score by tens of percentage points, so two papers reporting "90% faithful" are not comparable. Until the baseline is fixed by something other than convention, the search's objective function is not a fixed function.

## 6. Why It Is Hard

Two obstructions, both specific.

**Compute cost with a superlinear scaling exponent.** Edge count scales as $\Theta(L^2H^2)$ while per-pass cost scales as $\Theta(LH d)$. Exhaustive edge search therefore scales roughly as the *cube* of depth-width product — a factor of $\sim 7\times10^{5}$ from GPT-2 small to 70B (see §10).

**Non-identifiability under a moving objective.** Even with infinite compute the search does not have a unique answer: self-repair means the effect of a set is not the sum of its parts, so the greedy order determines which of several sufficient subgraphs you land on; and the faithfulness score that would arbitrate between them is itself baseline-dependent (Miller et al. 2024). This is the case where *the evaluation does not measure the thing it names*: "faithfulness" names mechanism recovery and measures sufficiency-under-a-chosen-ablation.

## 7. Current Research (as of 2026)

- **Mask learning over search.** Edge Pruning-style continuous relaxations, Princeton NLP and successors — the main route past greedy's $\Theta(|E|)$.
- **Feature-level graphs.** Sparse feature circuits and transcoder-based circuit tracing (Bau Lab / Northeastern, Anthropic interpretability, Google DeepMind), aiming to trade a large edge space for a sparse, more interpretable one. Attribution-graph tooling is now open-sourced. *(frontier — verify current scale claims.)*
- **Faithfulness standardisation.** Pre-registered ablation baselines and circuit benchmarks. *(frontier — verify.)*
- **Cheap-surrogate error theory.** Integrated-gradient and higher-order corrections to EAP; no published bound yet.

## 8. Concrete Next Experiment

**Question.** Does a gradient-based edge surrogate stay faithful when the model is large enough that exhaustive verification is impossible — and can we bound the error where it is still possible?

**Scale.** Three models: GPT-2 small (124M), Pythia-2.8B, Llama-3-8B. One task family with an unambiguous metric: indirect object identification, $N=200$ clean/corrupt pairs with symmetric token replacement (not Gaussian noise).

**Arms.**
1. *Reference:* exhaustive ACDC edge sweep — feasible on GPT-2 small only ($\approx 3.2\times10^4 \times 200$ passes).
2. *Treatment:* EAP-IG, 8 integration steps, on all three models.
3. *Control:* random edge subsets matched in size to the treatment circuit, plus a degree-matched control that preserves the graph's layer profile. This control is the arm usually omitted, and it is what separates "found a circuit" from "found enough edges".

**Deciding number.** Rank correlation (Spearman $\rho$) between EAP-IG edge scores and exact ACDC edge scores on GPT-2 small, together with the faithfulness gap $\Delta = \text{faith}(C_{\text{EAP-IG}}) - \text{faith}(C_{\text{random-matched}})$ at each scale. **Decision rule:** if $\rho \geq 0.8$ on GPT-2 small *and* $\Delta \geq 0.3$ holds at 8B with the same circuit size fraction, the surrogate transfers and the search cost problem is practically retired for this task class. If $\Delta$ decays below $0.1$ by 8B while $\rho$ stays high at 124M, the surrogate is validated only where verification was already cheap — which is the current suspicion, and would make the problem methodologically blocked rather than merely expensive.

Estimated cost: arm 1 $\approx$ 10 GPU-hours; arms 2–3 $\approx$ 100 GPU-hours total on 8×A100.

## 9. Key References

- **[Foundational]** Vig, Gehrmann, Belinkov, Qian, Nevo, Singer, Shieber. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS, 2020.
- **[Foundational]** Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.* ICLR, 2023. — arXiv:2211.00593
- **[SOTA]** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS, 2023. — arXiv:2304.14997
- **[SOTA]** Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* BlackboxNLP, 2024. — arXiv:2310.10348
- **[SOTA]** Hanna, Pezzelle, Belinkov. *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms.* NeurIPS, 2024.
- **[SOTA]** Bhaskar, Wettig, Friedman, Chen. *Finding Transformer Circuits with Edge Pruning.* NeurIPS, 2024.
- **[Method]** Zhang, Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR, 2024. — arXiv:2309.16042
- **[Critique]** Makelov, Lange, Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR, 2024. — arXiv:2311.17030
- **[Critique]** Miller, Chughtai, Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM, 2024.
- **[Critique]** McGrath, Rahtz, Kramar, Mikulik, Legg. *The Hydra Effect: Emergent Self-repair in Language Model Computations.* 2023. — arXiv:2307.15771
- **[Survey]** Marks, Rager, Michaud, Belinkov, Bau, Mueller. *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* 2024. — arXiv:2403.19647

## 10. Worked Example

**Setup.** IOI on GPT-2 small, $T=15$ tokens, $N=100$ pairs. Forward cost per sequence $\approx 2 \cdot 1.24\times10^{8} \cdot 15 \approx 3.7\times10^{9}$ FLOPs.

**Exhaustive edge sweep.**

$$3.7\times10^{9} \times 100 \times 3.2\times10^{4} \;\approx\; 1.2\times10^{16}\ \text{FLOPs}.$$

At 20% utilisation of one A100 ($6\times10^{13}$ FLOP/s effective) that is $\approx$ 200 s of pure math — hours in practice with per-edge hook overhead. Tractable. This is why the problem looks solved.

**Now Llama-3-70B, same task.** Edges $\approx 4\times10^{7}$ ($1.2\times10^{3}\times$), per-pass cost $565\times$:

$$1.2\times10^{16} \times 1.2\times10^{3} \times 565 \;\approx\; 8\times10^{21}\ \text{FLOPs}.$$

For comparison, training a 7B model on 200B tokens is $\approx 2\cdot 7\times10^{9}\cdot 2\times10^{11} = 2.8\times10^{21}$ FLOPs. **One exhaustive edge sweep on one task costs about three times the training run of a 7B model.**

**Where the obstruction becomes visible.** Suppose you accept EAP-IG's $O(1)$-pass surrogate instead, at $\sim 10^{13}$ FLOPs — a $10^{9}\times$ saving. You now have an edge ranking you cannot check, because the check is the $8\times10^{21}$ sweep you just avoided. And on GPT-2 small, where you *can* check, the thing you would be validating against is itself unstable: swap the ablation baseline from mean-resample to zero and published faithfulness scores move by tens of points (Miller et al. 2024); swap denoising for noising and the identified head set changes (Zhang and Nanda 2024). So the cheap method is unverifiable at the scale where it matters, and the expensive method's verdict is baseline-dependent at the scale where it is affordable. The search cost is the visible problem; the missing fixed objective is the one that makes the saving unbankable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*