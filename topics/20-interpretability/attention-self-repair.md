---
id: 20-interpretability/attention-self-repair
title: "Attention Head Ablation Redundancy and Self-Repair"
topic: 20-interpretability
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Head Ablation Redundancy and Self-Repair

> **Topic:** Interpretability · **ID:** `20-interpretability/attention-self-repair` · **Status:** partially-solved

## 1. Problem Statement

Ablating an attention head that a circuit analysis says is load-bearing often costs far less loss than the head's measured direct effect predicts. Downstream components change behaviour and recover part of the lost function. This is **self-repair** (also "the Hydra effect", "backup heads").

Three variants, routinely conflated:

- **Measurement.** Given model $M$, component $c$, and distribution $D$, define an importance score for $c$ that is invariant to which counterfactual you substitute when you ablate. Solving it means: two labs using zero-ablation and resample-ablation agree on the ranking of heads to within a stated tolerance.
- **Method.** Build a circuit-discovery procedure whose recovered subgraph is faithful *and* stable under self-repair — i.e. it does not drop a head because a backup masked it, and does not include a backup that is inert in the unablated model.
- **Theory.** Explain *why* self-repair exists. Is it (a) a mechanical artifact of LayerNorm rescaling and softmax renormalisation, (b) dropout-like or noise-driven redundancy learned in training, (c) an anti-correlated regulatory motif (copy suppression, negative heads) that exists for the clean forward pass and merely *appears* compensatory under ablation? These predict different scaling behaviour and different fixes.

Solving the theory variant means a quantitative predictor: given a head's clean-run direct effect and the LayerNorm/softmax state, predict the fraction of that effect that returns after ablation, with error small enough to be useful (say $R^2 > 0.8$ across models and prompt distributions).

## 2. Formal Setting

Transformer $M$ with residual stream $x^{(\ell)} \in \mathbb{R}^{d}$, heads indexed $h = (\ell, i)$. Let $z_h$ be the head's output written into the residual stream. Metric $m$ (logit of the correct token, logit difference between correct and a distractor, or cross-entropy) on prompt distribution $D$.

**Ablation.** Replace $z_h$ with $\tilde z_h$ and rerun:
$$\mathrm{AE}(h) \;=\; \mathbb{E}_{x \sim D}\big[m(M(x)) - m(M_{z_h \leftarrow \tilde z_h}(x))\big].$$
Choices of $\tilde z_h$: zero, the mean $\mathbb{E}_D[z_h]$, or a resample from a counterfactual distribution $D'$ (Wang et al.'s "path patching" / activation patching). These are *not* interchangeable; zero-ablation moves the residual stream off-distribution and inflates $\mathrm{AE}$ for reasons unrelated to the head's function.

**Direct effect.** With logit lens / direct logit attribution, unembedding $W_U$ and final LayerNorm scale $\sigma(x)$:
$$\mathrm{DE}(h) \;=\; \mathbb{E}_D\!\left[\frac{\langle W_U[t^\ast] - W_U[t^-],\; z_h\rangle}{\sigma(x)}\right],$$
measured by projecting only $z_h$ through the unembedding, holding all downstream nonlinearities at their clean values.

**Self-repair.** The gap between what the head contributes directly and what removing it costs:
$$\mathrm{SR}(h) \;=\; \mathrm{DE}(h) - \mathrm{AE}(h), \qquad r(h) \;=\; \mathrm{SR}(h)/\mathrm{DE}(h).$$
$r(h) = 0$ means no repair; $r(h) = 1$ means the ablation is fully absorbed. $r$ is reported per-head, per-prompt, and is heavy-tailed — the mean and median differ by a lot, so report both.

**Decomposition.** With downstream components $\{c_j\}$ after $h$,
$$\mathrm{SR}(h) \;=\; \underbrace{\sum_j \Delta \mathrm{DE}(c_j)}_{\text{component repair}} \;+\; \underbrace{\Big(\tfrac{1}{\sigma'} - \tfrac{1}{\sigma}\Big)\textstyle\sum_j \langle \cdot, z_{c_j}\rangle}_{\text{LayerNorm rescaling}} \;+\; \varepsilon,$$
where $\sigma'$ is the post-ablation final-LayerNorm scale. The second term is not a mechanism, it is arithmetic: removing a large-norm head shrinks $\sigma$, which multiplicatively amplifies everything else.

**Assumptions, and which are violated.**
1. *Ablation is a valid intervention* — violated. Zero and mean ablation put activations outside the training manifold; downstream MLPs see inputs they were never fit on.
2. *Direct effect is additive through LayerNorm* — violated by construction; $\sigma$ depends on the ablated component.
3. *$D$ is representative* — violated. Almost all quantitative self-repair numbers come from narrow templated distributions (IOI, induction, factual recall), not natural text.
4. *Independence of heads* — violated; QK attention patterns of downstream heads shift when an upstream head is removed, so the "backup" is a different computation, not a copy.

## 3. State of the Art

**Established.**
- *Redundancy under pruning.* Michel, Levy & Neubig, "Are Sixteen Heads Really Better than One?" (NeurIPS 2019) and Voita et al., "Analyzing Multi-Head Self-Attention" (ACL 2019) established that a large majority of heads can be removed with small metric loss in encoder-decoder MT and BERT-scale models.
- *Backup heads.* Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small" (ICLR 2023) found Backup Name Mover Heads that increase their name-moving behaviour precisely when the primary Name Mover Heads are ablated. This is the canonical existence proof and it has been reproduced many times.
- *Cross-model generality.* McGrath et al., "The Hydra Effect: Emergent Self-repair in Language Model Computations" (2023, arXiv:2307.15771), on Chinchilla-family models up to 70B, showed layer-level ablation triggers compensatory increase in other layers plus downstream erasure by late MLPs.
- *Partial mechanism.* Rushing & Nanda, "Explorations of Self-Repair in Language Models" (ICML 2024, arXiv:2402.15390) decomposed self-repair in GPT-2 and Pythia (70M–2.8B) into LayerNorm rescaling, anti-erasure by copy-suppression heads, and a residual of genuine backup behaviour, and reported that no single mechanism dominates across heads.
- *One head understood.* McDougall et al., "Copy Suppression: Comprehensively Understanding an Attention Head" (2023, arXiv:2310.04625) characterised GPT-2 small L10H7 as suppressing tokens already predicted, and reported that this account covers the large majority of the head's effect on OpenWebText. Copy suppression mechanically produces apparent self-repair: remove the thing being suppressed and the suppressor stops firing.

**Claimed but unablated.** That self-repair is a *learned robustness* property (e.g. induced by dropout or by training noise) is asserted often and has no controlled training-time ablation at scale behind it. That self-repair grows with model size is suggested by the Hydra results but has not been measured on a fixed distribution with a fixed metric across a clean parameter sweep.

**Benchmark-number-only.** Circuit-discovery faithfulness scores — ACDC (Conmy et al., NeurIPS 2023) and attribution patching / EAP (Syed, Rager & Conmy, 2024) — are reported as recovered-subgraph metrics on IOI, Greater-Than, Docstring. They are not evidence about self-repair mechanism; they inherit the ablation-choice sensitivity described above.

## 4. What Is Known

- Voita et al. (ACL 2019): 38 of 48 encoder heads pruned from an 8-layer WMT transformer for about $0.15$ BLEU loss.
- Michel et al. (NeurIPS 2019): most layers of a trained WMT/BERT model tolerate reduction to a single head at test time with small degradation; importance is highly non-uniform across heads.
- Wang et al. (ICLR 2023): in GPT-2 small (117M) on IOI, knocking out the three Name Mover Heads recovers a large fraction of logit difference through Backup Name Movers; the circuit is 26 heads out of 144.
- McGrath et al. (2023): in Chinchilla 7B–70B, ablating the single highest-impact attention layer at a given token is compensated substantially by downstream layers; the effect is present across scales, not a small-model artifact.
- Rushing & Nanda (ICML 2024): in GPT-2 small and Pythia up to 2.8B, LayerNorm rescaling alone accounts for a sizable share of measured self-repair on some heads and near none on others; self-repair is stronger for heads with larger direct effects and is prompt-dependent, not a fixed per-head constant.
- Ablation-method sensitivity is a reproduced regularity: zero-ablation systematically overstates importance relative to resample-ablation (Heimersheim & Nanda, "How to use and interpret activation patching", 2024).

## 5. What Is Not Known

- **Methodologically blocked.** There is no ablation-invariant definition of component importance. Since $r(h)$ depends on $\tilde z_h$, "how much self-repair is there" is not yet a well-posed question. Until the counterfactual is pinned by a stated criterion, cross-paper numbers are not comparable.
- **Theoretically open.** No proof or model predicting the *magnitude* of $r(h)$ from clean-run quantities. No theory saying whether self-repair is a necessary consequence of training a residual network with LayerNorm on next-token loss, or an incidental property.
- **Empirically open.** Whether $r(h)$ increases, saturates, or decreases with parameters, data, and depth on a fixed natural-text distribution. Runnable today on Pythia (70M–12B) and OLMo checkpoints; nobody has published the clean sweep with a single metric and a single ablation rule.
- **Empirically open.** Whether self-repair emerges during training at a specific point (e.g. alongside induction heads) or grows smoothly. Pythia's 154 checkpoints make this cheap and it is unrun at full scale.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by absent ground truth**.

Confounded: the quantity $\mathrm{SR}(h)$ mixes at least three things that are not separable by the measurement that defines it — the arithmetic LayerNorm rescale, the loss of a suppression signal (copy suppression heads stop suppressing when their target is gone, which *looks* like compensation but is the head doing its normal job), and genuine redundant computation. Any single scalar $r(h)$ silently sums them.

Absent ground truth: there is no independent oracle for "this head is important." Importance is *defined* by the ablation, so a disagreement between two ablation methods cannot be adjudicated — there is nothing to be right about. This is different from, say, probing, where downstream task accuracy provides an external check.

Compute is not the binding constraint; a full head-level sweep on GPT-2 small is 144 forward-pass sets and runs on one GPU in hours.

## 7. Current Research (as of 2026)

- **Mechanism decomposition.** Continuation of the Rushing & Nanda line: separating LayerNorm, anti-erasure, and true backup contributions per head, and testing whether the split is stable across prompt distributions. *(frontier — verify current status)*
- **Ablation-free attribution.** SAE-based and gradient-based attribution (attribution patching, EAP-IG) aimed at estimating importance without off-distribution interventions. Whether these dodge self-repair or merely relocate the confound is contested. *(frontier — verify)*
- **Training-dynamics.** Checkpoint-series studies (Pythia, OLMo) asking when backup behaviour appears relative to induction-head formation.
- **Safety-relevant use.** Whether self-repair defeats ablation-based capability removal / unlearning — i.e. an ablated capability returning through backup pathways. This is the applied motivation and is actively pursued at Anthropic, DeepMind, and in the open-source interpretability community. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does self-repair scale with model size, or is it a small-model artifact of LayerNorm rescaling?

**Scale.** Pythia 160M, 410M, 1.4B, 2.8B, 6.9B, 12B — same tokenizer, same data order, six points. Distribution $D$: 5,000 natural-text contexts from the Pile validation set (not templated tasks), plus IOI as a second arm for comparability with prior work.

**Procedure.** For every head $h$: compute $\mathrm{DE}(h)$ by direct logit attribution; compute $\mathrm{AE}(h)$ by mean-ablation over $D$ (not zero); form $r(h) = 1 - \mathrm{AE}(h)/\mathrm{DE}(h)$ on the top-50 heads by $|\mathrm{DE}|$ per model.

**Control arm — this is the point of the experiment.** Recompute $\mathrm{AE}(h)$ with the final LayerNorm scale $\sigma$ **frozen at its clean-run value** during the ablated forward pass. Call the resulting ratio $r_{\text{froz}}(h)$. The difference $r(h) - r_{\text{froz}}(h)$ is the LayerNorm-arithmetic share; $r_{\text{froz}}(h)$ is the non-arithmetic share. Second control: ablate a *randomly chosen* head of matched output norm, to establish the baseline $r$ produced by norm perturbation alone.

**Deciding number.** The slope of median $r_{\text{froz}}$ on the natural-text arm against $\log_{10}(\text{parameters})$ across the six models. Slope $> +0.05$ per decade with a 95% bootstrap CI excluding zero: self-repair is a growing, genuine mechanism and ablation-based interpretability and unlearning degrade with scale. Slope indistinguishable from zero, or negative, while $r - r_{\text{froz}}$ carries most of the total: self-repair is largely normalisation arithmetic and the field should report $r_{\text{froz}}$, not $r$.

Cost estimate: about $6 \times 50 \times 3$ ablated evaluation passes over 5,000 contexts — a few hundred GPU-hours on A100s. This is affordable and unrun.

## 9. Key References

- **[Foundational]** Paul Michel, Omer Levy, Graham Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS, 2019. — arXiv:1905.10650
- **[Foundational]** Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL, 2019. — arXiv:1905.09418
- **[Foundational]** Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* ICLR, 2023. — arXiv:2211.00593
- **[SOTA]** Thomas McGrath, Matthew Rahtz, János Kramár, Vladimir Mikulik, Shane Legg. *The Hydra Effect: Emergent Self-repair in Language Model Computations.* 2023. — arXiv:2307.15771
- **[SOTA]** Cody Rushing, Neel Nanda. *Explorations of Self-Repair in Language Models.* ICML, 2024. — arXiv:2402.15390
- **[SOTA]** Callum McDougall, Arthur Conmy, Cody Rushing, Thomas McGrath, Neel Nanda. *Copy Suppression: Comprehensively Understanding an Attention Head.* 2023. — arXiv:2310.04625
- **[Method]** Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS, 2023. — arXiv:2304.14997
- **[Method]** Aaquib Syed, Can Rager, Arthur Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* 2024. — arXiv:2310.10348
- **[Survey]** Stefan Heimersheim, Neel Nanda. *How to use and interpret activation patching.* 2024. — arXiv:2404.15255
- **[Context]** Catherine Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895

## 10. Worked Example

GPT-2 small (117M), IOI prompt: *"When Mary and John went to the store, John gave a drink to ___"*. Metric: logit difference between " Mary" and " John".

Clean run: logit difference $\approx 3.5$. Name Mover Head L9H9 has direct effect $\mathrm{DE} \approx 1.2$ — it moves the " Mary" direction into the residual stream.

Mean-ablate L9H9. Naive prediction: logit difference falls to $\approx 2.3$. Observed: it falls to roughly $2.7$–$2.9$. So $\mathrm{AE} \approx 0.7$, $\mathrm{SR} \approx 0.5$, $r \approx 0.4$.

Where does the $0.5$ come from? Decompose the downstream change:

| Source | share of $\mathrm{SR}$ (order of magnitude) |
|---|---|
| Backup Name Movers (L10H0, L10H10, L11H2) increasing " Mary" attribution | roughly half |
| Copy-suppression head L10H7 suppressing " Mary" *less*, because L9H9 no longer boosted it | a substantial minority |
| Final LayerNorm scale $\sigma$ shrinking, multiplicatively amplifying every remaining contribution | the remainder |

The obstruction is visible in row two and row three. Row three is arithmetic — nothing in the network "repaired" anything; the denominator moved. Row two is the *absence* of a normal computation, not the presence of a compensating one: L10H7's job is to suppress tokens already predicted, and L9H9 was the thing making " Mary" predicted. Removing the cause removes the suppression. Only row one is redundancy in the intuitive sense.

Now switch to zero-ablation instead of mean-ablation. The residual stream loses the head's mean output too, $\sigma$ moves further, and the measured $\mathrm{AE}$ rises — so $r$ *falls*, with no change to the model. The same head, the same prompt, the same metric, two defensible interventions, materially different self-repair fractions and no principle to choose between them. That is the methodological block in section 5, in one prompt.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*