---
id: 02-attention/attention-head-superposition
title: "Attention Head Superposition and Polysemanticity"
topic: 02-attention
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Head Superposition and Polysemanticity

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-head-superposition` · **Status:** open

## 1. Problem Statement

A trained transformer appears to compute more distinct attention "behaviors" (induction, name-mover, duplicate-token, successor, syntactic-agreement, …) than it has attention heads. Two explanations compete:

- **Polysemanticity:** a single head implements several unrelated behaviors, selected by context.
- **Head superposition:** a behavior is not carried by any single head; it is a direction in the space spanned by several heads' OV/QK circuits, and heads interfere with each other at a controlled rate.

The problem has three variants with different difficulty.

- **Measurement variant:** given a model $M$ and a set of behaviors $\mathcal{B}$, decide whether the head-to-behavior map is one-to-one, many-to-one, one-to-many, or many-to-many — with a statistic that is not an artifact of how $\mathcal{B}$ was enumerated. Currently the weakest link.
- **Method variant:** find a basis change (rotation, sparse dictionary, head-decomposition) in which behaviors become monosemantic units, or prove no such basis exists at fixed layer width.
- **Theory variant:** for a stated data distribution and training objective, predict when the loss-minimizing solution puts $k$ behaviors into $h < k$ heads, and at what interference cost.

Solved would mean: a decomposition of an attention layer of a $\geq$1B-parameter LM into units that are (i) individually interpretable, (ii) causally sufficient for the layer's output on held-out text to within a stated loss recovery, and (iii) stable across seeds.

## 2. Formal Setting

Layer $\ell$ has $H$ heads. For head $i$, residual stream $X \in \mathbb{R}^{T \times d}$:

$$A^{(i)} = \mathrm{softmax}\!\left(\frac{X W_Q^{(i)} W_K^{(i)\top} X^\top}{\sqrt{d_h}} + \text{mask}\right), \qquad \mathrm{out} = \sum_{i=1}^{H} A^{(i)} X W_{OV}^{(i)},\; W_{OV}^{(i)} = W_V^{(i)} W_O^{(i)}.$$

The layer output is a *sum* over heads, so heads are interchangeable up to any invertible mixing that preserves the sum only if the softmax nonlinearity is respected — the QK circuits are not linearly mixable, which is what makes head superposition different from neuron superposition.

**Behavior.** A behavior $b$ is a triple (prompt distribution $D_b$, causal target $\mathcal{T}_b$, metric $m_b$). Measured as a causal effect under activation patching: patch head $i$'s output from a clean run into a corrupted run and record

$$\mathrm{CE}(i,b) = \mathbb{E}_{x \sim D_b}\big[ m_b(M_{\text{patch}(i)}(x)) - m_b(M_{\text{corrupt}}(x)) \big] \big/ \mathbb{E}\big[m_b(M_{\text{clean}}) - m_b(M_{\text{corrupt}})\big].$$

**Polysemanticity index.** For head $i$ over behavior set $\mathcal{B}$, with $p_b = |\mathrm{CE}(i,b)| / \sum_{b'} |\mathrm{CE}(i,b')|$:

$$P(i) = \exp\!\Big(-\sum_{b \in \mathcal{B}} p_b \log p_b\Big) \in [1, |\mathcal{B}|],$$

the effective number of behaviors per head. $P(i)=1$ is monosemantic.

**Superposition index.** For behavior $b$, let $v_b \in \mathbb{R}^H$ be the vector $(\mathrm{CE}(1,b),\dots,\mathrm{CE}(H,b))$. Superposition is the failure of $v_b$ to be sparse *and* the failure of ablating any single head to destroy $b$:

$$S(b) = \frac{\|v_b\|_1^2}{\|v_b\|_2^2 \cdot H} \in [1/H, 1], \qquad \text{plus the requirement } \max_i \mathrm{CE}(i,b) < \tau \text{ while } \sum_i \mathrm{CE}(i,b) \approx 1.$$

**Interference.** With $k$ behaviors in $h$ heads, the residual noise injected into behavior $b$ by behaviors $b' \neq b$ sharing OV subspace: $\varepsilon_b = \sum_{b' \neq b} \langle u_b, W_{OV} u_{b'}\rangle^2$, measurable as the logit shift on $b$'s target when only $b'$-triggering tokens are present.

**Assumptions, and which are violated.**
1. *$\mathcal{B}$ is a complete enumeration of the layer's behaviors.* **Violated** — $\mathcal{B}$ is hand-curated; $P(i)$ is bounded above by $|\mathcal{B}|$ by construction.
2. *Patching is additive and non-interacting across heads.* **Violated** — Merullo et al. (NeurIPS 2024) show inter-head composition through low-rank residual subspaces; single-head CE undercounts.
3. *Corruption distribution is off-manifold-safe.* **Violated** — mean-ablation and zero-ablation give different CE rankings on the same circuit (Wang et al., ICLR 2023, Appendix on ablation choice).
4. *Heads are the natural unit.* Assumed, not established; the $d_h$-dimensional split is an architectural convention, not a fact about the learned function.

## 3. State of the Art

**Established (reproduced, ablated).**
- Head *redundancy* at inference: Michel et al. (NeurIPS 2019) removed all but one head in most layers of a WMT En–De transformer with small BLEU loss. Voita et al. (ACL 2019) pruned 38 of 48 encoder heads with a 0.15 BLEU drop.
- Head *specialization for some behaviors*: induction heads in 2-layer attention-only models are near-monosemantic and causally necessary (Olsson et al., 2022).
- Neuron-level superposition is real and quantitatively predicted in toy models by feature sparsity (Elhage et al., *Toy Models of Superposition*, 2022).

**Claimed but not fully ablated.**
- *Attention head superposition proper.* Anthropic's Transformer Circuits "Circuits Updates" (2023, Jermyn, Olah and colleagues) constructed toy models where two heads jointly implement a set of skip-trigrams that neither implements alone. Real-model evidence is suggestive, not decisive.
- Attention-output SAEs (Kissane, Krzyzanowski, Bloom, Conmy, Nanda, 2024) train sparse dictionaries on $z$ concatenated over heads in GPT-2 Small and report the majority of features human-rated interpretable, with features spanning multiple heads. The multi-head spanning is reported; the counterfactual that a per-head SAE would do as well is only partially run.

**Benchmark-number-only results.** Loss-recovered and L0 figures for attention SAEs, and "% interpretable features" from LM-autointerp, are benchmark numbers with no ground truth: they do not establish that the recovered features are the model's units.

## 4. What Is Known

- Toy models: superposition onset is governed by feature sparsity and importance; with feature sparsity above ~$1-1/n$, models represent more features than dimensions in structured polytopes (Elhage et al., 2022, models with $n \le 400$ features, $m \le 20$ dims).
- Scherlis, Sachan, Jermyn, Shlegeris, Hadfield-Menell (arXiv:2210.01892, 2022) define *capacity* as fraction of a dimension per feature and prove, for a quadratic-loss toy model, that features are either fully represented, fully dropped, or share dimensions — polysemanticity appears in a specific regime of the sparsity/importance plane.
- IOI in GPT-2 Small (117M): a 26-head circuit with named roles; name-mover heads 9.6 and 9.9 carry most direct effect, and backup name-mover heads *increase* their effect when the primary is ablated (Wang et al., ICLR 2023). This self-repair is the single strongest evidence that behavior is not localized to one head.
- Docstring/greater-than and successor-head studies at 117M–7B find heads that fire on several unrelated distributions.
- ACDC (Conmy et al., NeurIPS 2023) recovers known circuits at up to ~80% edge-level agreement on IOI-style tasks in GPT-2 Small; agreement degrades on larger models.
- Lieberum et al. (2023) scaled circuit analysis to Chinchilla 70B multiple-choice and found identifiable "correct letter" heads — specialization survives to 70B for at least one task.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no seed-independent, enumeration-independent definition of "the set of behaviors a head implements". $P(i)$ and $S(b)$ above both depend on the hand-chosen $\mathcal{B}$. Until a behavior set can be *derived* from the model rather than supplied, "how polysemantic is this head" is not a measurable quantity.
- **Empirically open.** Whether attention-output SAE features are genuinely cross-head or whether cross-head loading is an artifact of training a single dictionary on concatenated $z$. Runnable today on GPT-2 Small and Pythia-1.4B; not run as a matched control.
- **Empirically open.** Whether head polysemanticity increases, decreases, or is flat in $H$ at fixed parameter count. No head-count sweep at $\geq$1B scale exists.
- **Theoretically open.** No analogue of the Elhage/Scherlis capacity results for the softmax QK circuit. The known theory covers linear-plus-ReLU feature encoding, not attention pattern selection. No proof that a transformer layer *must* superpose $k > H$ QK behaviors under any stated distribution.

## 6. Why It Is Hard

**Non-identifiability plus absent ground truth.** The layer output is $\sum_i A^{(i)} X W_{OV}^{(i)}$. Even with the softmax fixing the QK basis, we lack a criterion that says which decomposition of this sum is *the* model's own. Any dictionary with enough atoms and enough sparsity pressure will produce human-readable features; interpretability of the output does not distinguish a true decomposition from a plausible re-parameterization.

Compounding this: **self-repair confounds the measurement.** Backup name-movers mean single-head ablation systematically understates a head's role, so $\mathrm{CE}(i,b)$ is biased downward exactly in the regime where superposition is being tested — the observation "no single head is necessary" is predicted by both the superposition hypothesis and the redundancy hypothesis. The evaluation does not measure the thing it names.

## 7. Current Research (as of 2026)

- **Attention-output dictionary learning.** Kissane/Conmy/Nanda line, extended to transcoders and cross-layer dictionaries; groups at Google DeepMind and MATS-affiliated researchers. *(frontier — verify current model coverage.)*
- **Weights-based decomposition** — factoring $W_{QK}$ and $W_{OV}$ directly rather than activations, to sidestep the enumeration problem. *(frontier — verify.)*
- **Attribution-patching and edge-level circuit discovery at scale** (successors to ACDC), aimed at making $\mathrm{CE}$ cheap enough for exhaustive head sweeps at 7B+.
- **Toy models of QK superposition** — extending Elhage-style phase diagrams to attention-pattern selection rather than feature encoding. Small literature; the theory gap in §5 is live.

## 8. Concrete Next Experiment

**Question.** Is cross-head feature loading in attention SAEs a property of the model or of the dictionary?

**Scale.** Pythia-1.4B, all 24 layers, 24 heads/layer. ~500M tokens of the Pile for dictionary training. Cost order: a few hundred A100-hours — reachable by a single lab.

**Arms.**
1. *Joint:* one SAE on the concatenated $z \in \mathbb{R}^{24 \cdot d_h}$ per layer, expansion 32×.
2. *Control (per-head):* 24 separate SAEs, one per head, each with $1/24$ of the joint dictionary size, so total atoms and total parameters match arm 1 exactly.
3. *Null control:* arm 1 trained on a model with heads randomly re-partitioned (permute $d_h$ columns across heads before splitting) — destroys head structure but preserves the layer function.

**Deciding number.** Fraction of layer-output variance explained at matched L0 (target L0 = 30), reported as $\Delta = \mathrm{FVE}_{\text{joint}} - \mathrm{FVE}_{\text{per-head}}$, averaged over layers 8–16.

- $\Delta \le 0.01$ (1 percentage point): head boundaries carry the structure; cross-head features are a dictionary artifact; superposition-across-heads is not supported.
- $\Delta \ge 0.05$ **and** arm 3 shows no such gap: the joint dictionary is exploiting genuine cross-head structure — evidence for head superposition.
- $\Delta \ge 0.05$ **and** arm 3 shows the same gap: the gain is from dictionary capacity allocation, not head structure. Null result, and a warning about every existing attention-SAE claim.

Secondary readout: per-feature head-participation entropy, $\exp(-\sum_i q_i \log q_i)$ with $q_i$ the fraction of a feature's decoder norm in head $i$; report the median over the top 5,000 features by activation frequency.

## 9. Key References

- **[Foundational]** Elhage, Nanda, Olsson, et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
- **[Foundational]** Elhage, Hume, Olsson, et al. *Toy Models of Superposition.* Transformer Circuits Thread, 2022. — arXiv:2209.10652
- **[Foundational]** Olsson, Elhage, Nanda, et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, 2022. — arXiv:2209.11895
- **[Theory]** Scherlis, Sachan, Jermyn, Shlegeris, Hadfield-Menell. *Polysemanticity and Capacity in Neural Networks.* 2022. — arXiv:2210.01892
- **[Foundational]** Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019. — arXiv:1905.10650
- **[Foundational]** Voita, Talbot, Moiseev, Sennrich, Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL 2019. — arXiv:1905.09418
- **[SOTA]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023. — arXiv:2211.00593
- **[SOTA]** Kissane, Krzyzanowski, Bloom, Conmy, Nanda. *Interpreting Attention Layer Outputs with Sparse Autoencoders.* 2024. — arXiv:2406.17759
- **[SOTA]** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023. — arXiv:2304.14997
- **[SOTA]** Lieberum, Rahtz, Kramár, et al. *Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla.* 2023. — arXiv:2307.09458
- **[Related]** Merullo, Eickhoff, Pavlick. *Talking Heads: Understanding Inter-Layer Communication in Transformer Language Models.* NeurIPS 2024.
- **[Related]** Cunningham, Ewart, Riggs, Huben, Sharkey. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR 2024. — arXiv:2309.08600
- **[Survey]** Bereska, Gavves. *Mechanistic Interpretability for AI Safety — A Review.* TMLR, 2024. — arXiv:2404.14082

## 10. Worked Example

Take GPT-2 Small head 9.9 (layer 9, head 9), a name-mover in the IOI circuit, and $\mathcal{B} = \{$IOI name-moving, duplicate-token detection, previous-token attention, successor (month/number increment)$\}$, each with 512 prompts.

Suppose measured normalized CEs are $(0.42, 0.05, 0.02, 0.11)$. Then $p = (0.70, 0.08, 0.03, 0.18)$, entropy $= 0.85$ nats, so

$$P(9.9) = e^{0.85} = 2.34.$$

Read literally: "head 9.9 does about 2.3 things." Now the obstruction, in two moves.

**Move 1 — enumeration dependence.** Add four more behaviors that head 9.9 does not participate in. Their CEs are near zero, the distribution barely changes, $P$ moves to ~2.4. Add instead four behaviors it *does* touch weakly (0.08 each): $P$ rises to ~3.3. The number is a function of the analyst's list, not of the model. There is no principled stopping rule for $\mathcal{B}$, so $P$ has no fixed value.

**Move 2 — self-repair.** Ablate head 9.9 and re-measure the IOI logit difference. Wang et al. report that backup name-mover heads (e.g. 10.0, 10.10, 11.2) increase their direct effect when 9.9 is removed, so the observed drop in logit difference is markedly smaller than 9.9's direct-effect contribution. Concretely: if 9.9's direct effect is 0.42 of the clean logit difference but ablating it costs only ~0.15, then $\sum_i \mathrm{CE}(i, \text{IOI}) \gg 1$ and the CE vector is not a partition of credit at all.

Both hypotheses predict this. **Superposition:** IOI is a direction in head-space, no head is individually necessary. **Redundancy:** several heads independently compute IOI, each sufficient. The ablation number $0.15$ is consistent with both; distinguishing them requires simultaneous multi-head ablation over $\binom{H}{k}$ subsets, which is $\sim 10^5$ forward-pass sweeps per layer at $H=12, k\le 5$ — and that is GPT-2 Small, the smallest model anyone cares about. That combinatorial cost, sitting on top of a metric whose value depends on the analyst's behavior list, is the obstruction.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*