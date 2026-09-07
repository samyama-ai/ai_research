---
id: 02-attention/attention-circuit-universality
title: "Attention Circuit Universality Across Random Seeds"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Circuit Universality Across Random Seeds

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-circuit-universality` · **Status:** empirically-open

## 1. Problem Statement

Train the same transformer architecture on the same data with only the random seed changed (initialization, data order, dropout mask). Do the resulting models implement *the same attention circuits* — the same decomposition of a task into head-level computations wired the same way — or merely the same input-output function realized by different mechanisms?

Three variants, of very different difficulty:

- **Measurement.** Define a circuit-similarity statistic $U \in [0,1]$ between two independently seeded models that is invariant to the known nuisance symmetries (head permutation within a layer, orthogonal rotation inside a head's $QK$/$OV$ subspaces, per-head scale) and is *not* saturated by trivially matching behavior. This is the binding constraint and is currently unsettled.
- **Method.** Given such a statistic, produce the matching: an assignment of heads and edges in model $A$ to heads and edges in model $B$ that maximizes $U$, with a null distribution to test against.
- **Theory.** Prove or refute: for a task family and architecture, the set of loss-minimizing circuit implementations reachable by SGD has cardinality $1$ up to symmetry (strong universality), finite and small (weak universality), or grows with width (no universality).

Solving it means: a statistic with a calibrated null, a matching algorithm, and a number for a named task at a named scale — plus evidence the number tracks a *causal* claim (ablating matched heads in $A$ and $B$ produces the same behavioral change), not just representational correlation.

## 2. Formal Setting

Two models $\theta_A, \theta_B$ from seeds $s_A \neq s_B$, identical architecture: $L$ layers, $H$ heads/layer, head dim $d_h$, residual width $d$. Head $(\ell,h)$ has $W_Q, W_K, W_V \in \mathbb{R}^{d\times d_h}$, $W_O \in \mathbb{R}^{d_h \times d}$. Following the low-rank bilinear form (Elhage et al. 2021), the head is characterized by
$$W_{QK}^{(\ell,h)} = W_Q W_K^\top \in \mathbb{R}^{d\times d},\qquad W_{OV}^{(\ell,h)} = W_V W_O \in \mathbb{R}^{d\times d},$$
both invariant to $W_Q \mapsto W_Q R$, $W_K \mapsto W_K R^{-\top}$ for $R \in GL(d_h)$. These products, not the factors, are the measurable objects.

**Circuit as a graph.** Fix a task distribution $\mathcal{D}$ (clean prompts $x$, counterfactual prompts $x'$, metric $m$, e.g. logit difference). The circuit is a subgraph $C \subseteq G$ of the computational graph whose nodes are heads and MLPs and whose edges are residual-stream writes/reads. Edge importance is measured by activation patching:
$$\tau_e = \mathbb{E}_{(x,x')\sim\mathcal{D}}\big[\, m(x) - m\big(x \,\|\, \mathrm{do}(e \leftarrow a_e(x'))\big)\,\big],$$
i.e. run on $x$ but splice in edge $e$'s activation from $x'$. $C_\epsilon = \{e : |\tau_e| > \epsilon\}$. Faithfulness of $C$: $F(C) = m(x \| \mathrm{do}(\bar{C} \leftarrow \text{mean}))/m(x)$, target $F \approx 1$.

**Universality statistic.** Let $\pi$ range over permutations of heads *within* each layer (the only exact architectural symmetry; cross-layer matching is not a symmetry and must be scored, not quotiented). Define
$$U(A,B) = \max_{\pi} \frac{\sum_{e \in C^A_\epsilon} \mathbb{1}[\pi(e) \in C^B_\epsilon]\cdot \rho\big(\tau^A_e, \tau^B_{\pi(e)}\big)}{|C^A_\epsilon \cup \pi^{-1}(C^B_\epsilon)|},$$
with $\rho$ a per-edge agreement term (e.g. sign agreement times attention-pattern correlation on held-out prompts). Report $U$ against a null $U_0$ from *shuffled* head assignments and against a *within-seed* ceiling $U_1$ from two checkpoints of the same run.

**Assumptions, and which are violated.**
1. *Head permutation is the only symmetry.* Violated: the residual stream admits no canonical basis, so any invertible $M$ acting as $W_{OV}\mapsto M^{-1}W_{OV}M$ leaves the function unchanged; and LayerNorm folding shifts weight mass between components.
2. *Circuits are sparse and node-localized.* Violated: self-repair / the Hydra effect (McGrath et al. 2023) means ablating a head recruits backups, so $\tau_e$ under-reports importance and is non-additive.
3. *One circuit per task.* Violated: modular addition admits at least two distinct algorithms across seeds (Zhong et al. 2023).
4. *$\mathcal{D}$'s counterfactual isolates the intended variable.* Often violated — patching distributions leak distributional shift, which is scored as circuit signal.

## 3. State of the Art

**Established.**
- *Induction heads* form in essentially every decoder-only transformer with $\geq 2$ layers, at a sharp phase change, across 13M–13B params (Olsson et al., Anthropic, 2022). This is the strongest universality claim in the field and it is at the level of a *circuit family*, not a specific head index.
- *Automated circuit discovery* is routine: ACDC (Conmy et al., NeurIPS 2023) and attribution patching / EAP (Syed, Rager, Conmy, 2023) recover known circuits (IOI, greater-than, docstring) at recall competitive with hand analysis, at ~$10^3$–$10^5\times$ fewer forward passes for EAP.
- *Cross-model circuit reuse*: Merullo, Eickhoff, Pavlick (ICLR 2024) show GPT-2 medium reuses the IOI circuit's induction/name-mover components on the unrelated Colored Objects task — reuse *within* a model.
- *Cross-scale consistency*: Tigges et al. (2024) find IOI and greater-than circuits are qualitatively stable across Pythia checkpoints and 70M–2.8B scale, with component roles preserved.

**Claimed but unablated.**
- That $U$ high $\Rightarrow$ same algorithm. No paper has shown a matched-head correspondence between independent seeds that also transfers a *causal intervention* (e.g. a steering direction found in $A$ works, via the matching, in $B$).
- That sparse-autoencoder features are seed-universal. Feature-level diffing via crosscoders (Lindsey et al., Anthropic, 2024) is a method note; the seed-universality number is not established.

**Benchmark-number-only.** Reported circuit "overlap" percentages between models are almost always node-set Jaccard at a hand-tuned $\epsilon$, with no null distribution. Treat as descriptive statistics, not evidence.

## 4. What Is Known

- **Neuron-level universality is small but nonzero.** Gurnee et al. (2024), across five independently seeded GPT-2 medium models, find roughly **1–5%** of MLP neurons are "universal" (max pairwise activation correlation above threshold across all five seeds). These few are interpretable (entropy neurons, prediction/suppression neurons). Scale: 355M params, 5 seeds.
- **Non-universality has an existence proof.** Zhong et al. (NeurIPS 2023): one-layer transformers on modular addition learn *two* algorithms — "Clock" (Fourier multiplication, as in Nanda et al., ICLR 2023) and "Pizza" — selected by seed and hyperparameters. Same loss, same task, different circuits. Scale: toy, $p \approx 59$–$113$.
- **Universality of *representation class*, not implementation.** Chughtai, Chan, Nanda (ICML 2023): networks learning group composition consistently use irreducible representations, but *which* irreps varies by seed. This is the pattern to expect: family-level universality, instance-level variation.
- **Convergent learning at the feature level, in vision.** Li, Yosinski, Clune, Hinton, Lipson (ICLR 2016): many conv1 filters match one-to-one across seeds; the fraction falls sharply with depth. CKA (Kornblith et al., ICML 2019) finds high cross-seed representational similarity for wide networks, low for narrow ones.
- **Loss-basin symmetry.** Entezari et al. (ICLR 2022) conjecture, and Git Re-Basin (Ainsworth et al., ICLR 2023) empirically supports, that SGD solutions are linearly mode-connected *after* permutation alignment for MLPs/CNNs — with reported barriers near zero for wide ResNets on CIFAR-10, but non-trivial residual barriers for transformers.
- **Self-repair is large.** McGrath et al. (2023): ablating a name-mover head in GPT-2 recovers a substantial fraction of its direct effect through downstream compensating heads. Any $\tau_e$-based statistic inherits this bias.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed, symmetry-correct, null-calibrated $U$. Node-set Jaccard, CKA, and patching-score correlation each answer different questions and disagree; none quotient the residual-stream basis ambiguity. Until $U$ is fixed, "circuits are universal" is not a falsifiable claim.
- **Empirically open.** Nobody has trained $\geq 10$ seeds of a $\geq 1$B-param LM to a fixed token budget and run identical automated circuit discovery on a fixed task suite. The compute exists; the run has not been reported. Existing multi-seed work is at 355M (Gurnee) or toy scale (Zhong).
- **Theoretically open.** No theorem bounds the number of SGD-reachable circuit equivalence classes for attention on a nontrivial task. Even for induction, there is no proof that the two-head QK-composition solution is the unique minimum-loss attainable construction.
- **Open and under-asked.** Whether universality is a property of the *task* (algorithmically rigid tasks like induction universal; underdetermined tasks like modular addition not) rather than of the architecture.

## 6. Why It Is Hard

**Non-identifiability of the residual basis, compounded by confounded measurement.** Two obstructions, both specific:

1. *No canonical basis.* Head $(\ell,h)$ in model $A$ and head $(\ell',h')$ in $B$ can implement the same computation while $W_{OV}$ matrices are related by an unknown invertible $M$ that also depends on which downstream heads read the subspace. Matching is a quadratic assignment problem — NP-hard in general — over a group that is *larger* than the one we can quotient exactly. Low $U$ is therefore not evidence of non-universality; it may be search failure.
2. *Patching scores are not the causal quantity they name.* $\tau_e$ measures the effect of an intervention *in a model that repairs itself*. Self-repair makes $\tau_e$ sub-additive and threshold-sensitive: shifting $\epsilon$ by a factor of 2 can change $|C_\epsilon|$ by an order of magnitude, and $U$ with it. Two labs can report 20% and 70% overlap on the same models from this alone.

Compute is a secondary cost (10 seeds $\times$ 1B params $\approx$ 10 full pretrains), but it is not the reason the problem is open — the definitional problem is.

## 7. Current Research (as of 2026)

- **Anthropic interpretability**: cross-model diffing with crosscoders — sparse dictionaries trained jointly on two models' activations, yielding shared vs. model-specific features. The natural substrate for a seed-universality number at feature level *(frontier — verify current results)*.
- **EleutherAI / Pythia ecosystem**: multi-seed Pythia variants (deduped, seed-varied) make cross-seed circuit comparison feasible at 70M–1.4B without new pretraining.
- **DeepMind and academic groups (Brown, Northeastern, MATS cohorts)**: circuit reuse, cross-task and cross-scale stability, attribution-graph methods.
- **Weight-space symmetry / model merging**: Git Re-Basin descendants extended to attention, where head-permutation plus within-head rotation must be handled jointly *(frontier — verify)*.
- **Sparse feature circuits** (Marks et al.) as an alternative node basis that may be better-identified than heads, since dictionary features are sparse and less rotation-ambiguous.

## 8. Concrete Next Experiment

**Scale.** Pretrain $n=8$ seeds of a 160M-param decoder-only transformer (12 layers, 12 heads, $d=768$) on 20B tokens of a fixed, fixed-order-per-seed corpus. Cost: roughly $8 \times 10^{19}$ FLOPs total, a few hundred A100-hours — within a single academic budget.

**Procedure.** On a suite of 4 tasks (IOI, greater-than, induction on random repeated tokens, subject-verb agreement), run EAP-IG circuit discovery with a *fixed* faithfulness target $F = 0.85$ (not a fixed $\epsilon$) so circuit size is determined by behavior, not threshold choice. Match heads across seed pairs by maximizing $U$ over within-layer permutations, initialized from $W_{OV}$-subspace principal-angle similarity.

**Control arms (three, all required).**
1. *Null:* $U_0$ from random within-layer head permutations.
2. *Ceiling:* $U_1$ between two checkpoints of the *same* seed at 19B and 20B tokens.
3. *Causal transfer:* take a steering/ablation intervention tuned on seed 1; apply it through the matching to seeds 2–8; record behavioral change.

**The deciding number.** The normalized universality index
$$\hat{U} = \frac{\mathbb{E}_{A\neq B}[U(A,B)] - U_0}{U_1 - U_0}.$$
$\hat U > 0.7$ *and* causal transfer recovering $>70\%$ of the seed-1 effect size $\Rightarrow$ strong universality at this scale. $\hat U < 0.2$ with matched behavior ($<0.05$ nats loss spread across seeds) $\Rightarrow$ non-universality, and the field's shared-circuit assumption is wrong. The interval between is the honest current prior, and reporting it with error bars over 28 seed pairs is itself the contribution.

## 9. Key References

- **[Foundational]** Elhage, Nanda, Olsson, et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, Anthropic, 2021.
- **[Foundational]** Olsson, Elhage, Nanda, et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895
- **[SOTA]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023. — arXiv:2211.00593
- **[SOTA]** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023. — arXiv:2304.14997
- **[SOTA]** Gurnee, Horsley, Guo, Kheirkhah, Sun, Hathaway, Nanda, Bertsimas. *Universal Neurons in GPT2 Language Models.* 2024. — arXiv:2401.12181
- **[SOTA]** Zhong, Liu, Tegmark, Andreas. *The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks.* NeurIPS 2023. — arXiv:2306.17844
- **[SOTA]** Chughtai, Chan, Nanda. *A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations.* ICML 2023. — arXiv:2302.03025
- **[SOTA]** Merullo, Eickhoff, Pavlick. *Circuit Component Reuse Across Tasks in Transformer Language Models.* ICLR 2024. — arXiv:2310.08744
- **[Related]** Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR 2023. — arXiv:2209.04836
- **[Related]** Entezari, Sedghi, Saukh, Neyshabur. *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks.* ICLR 2022. — arXiv:2110.06296
- **[Related]** McGrath, Rahtz, Kramar, Mikulik, Legg. *The Hydra Effect: Emergent Self-repair in Language Model Computations.* 2023. — arXiv:2307.15771
- **[Related]** Li, Yosinski, Clune, Hinton, Lipson. *Convergent Learning: Do different neural networks learn the same representations?* ICLR 2016. — arXiv:1511.07543
- **[Survey]** Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network Representations Revisited.* ICML 2019. — arXiv:1905.00414

## 10. Worked Example

Take IOI in GPT-2 small: "When Mary and John went to the store, John gave a drink to ___". The published circuit has 26 heads in 7 classes; name-mover heads include L9H9, L9H6, L10H0, with negative name movers L10H7, L11H10.

Now suppose seed $B$ of the same architecture. Two candidate outcomes:

- Its strongest name mover sits at **L9H4** and the S-inhibition heads at **L7H2, L8H11**. Layer indices and class structure match; head indices do not. Node-set Jaccard on raw indices $\approx 0.1$. Jaccard after within-layer permutation matching $\approx 0.8$. Same evidence, two conclusions differing by $8\times$ — purely a choice of statistic.

Then the threshold effect. In GPT-2 small on IOI, the top 3 heads by direct logit attribution carry most of the effect; the tail is long and shallow. Sketch the count of edges above threshold:

```
epsilon (logit-diff)   |C_eps|   faithfulness F
0.40                       9            0.62
0.20                      31            0.85
0.10                      88            0.94
0.05                     240            0.98
```

$|C_\epsilon|$ grows $\approx 27\times$ as $\epsilon$ falls $8\times$, while $F$ moves 0.62 → 0.98. Overlap statistics computed on the 9-edge circuit are dominated by the three name movers and will look universal; computed on the 240-edge circuit they are dominated by the noise tail and will look non-universal. This is the obstruction in one table: **the answer is a function of $\epsilon$, and nothing in the literature fixes $\epsilon$.** Section 8's fix — hold $F$ constant, not $\epsilon$ — is the minimum needed to make the question well posed.

And self-repair makes even that fragile: ablating L9H9 alone recovers much of its direct effect through L10H7 becoming less negative, so $\tau_{L9H9}$ measured by single-edge patching understates it. If seed $B$ happens to have weaker negative name movers, its $\tau$ values are systematically larger — and $B$'s circuit looks bigger and less overlapping, for reasons that have nothing to do with whether the algorithm is the same.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*