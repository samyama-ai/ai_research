---
id: 28-knowledge-editing/causal-tracing-does-not-localize-edit-sites
title: "Causal Tracing Does Not Localize Editable Sites"
topic: 28-knowledge-editing
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Causal Tracing Does Not Localize Editable Sites

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/causal-tracing-does-not-localize-edit-sites` · **Status:** partially-solved

## 1. Problem Statement

Causal tracing (activation patching over corrupted-prompt runs) produces a heatmap over $(\text{layer}, \text{token})$ positions showing where a fact "lives" in a transformer. ROME used that heatmap to pick the layer it edits. The empirical finding is that the heatmap does not predict where an edit works: editing at the traced peak layer is no better than editing at many other layers, and per-fact tracing strength does not predict per-fact edit success.

Three separable variants:

- **Measurement.** Does the causal-tracing statistic, as defined, estimate anything about parameter-space editability? What is the estimand? Currently undefined — tracing measures a property of activations at inference, editing changes weights.
- **Method.** Is there *any* cheap localization statistic $L$ over sites such that choosing $\arg\max_s L(s)$ improves edit quality over a fixed-layer default? Partially answered: no known $L$ does, for ROME-family edits.
- **Theory.** Is representational localization *in principle* informative about weight editability, or are the two provably decoupled under overparameterization?

Solving it means either exhibiting a localization statistic with measured predictive validity for edit outcomes, or proving decoupling and retiring localization as an edit-site selector.

## 2. Formal Setting

Model $f_\theta$, autoregressive, $L$ layers. A fact is a triple $(s,r,o)$ realized as prompt $x$ with target $o$.

**Causal tracing.** Run clean forward pass on $x$; cache hidden states $h^{(\ell)}_t$. Run corrupted pass $x'$ with subject-token embeddings perturbed by Gaussian noise $\epsilon \sim \mathcal{N}(0,\nu^2 I)$, $\nu$ set to 3× the empirical embedding std. Restore a single state and measure recovery:

$$\mathrm{IE}(\ell,t) \;=\; \mathbb{P}_{\text{corrupt}+h^{(\ell)}_t}[o] \;-\; \mathbb{P}_{\text{corrupt}}[o].$$

Measured as a probability difference on the single target token, averaged over noise seeds (ROME uses 10). The *traced site* is $s^\star=\arg\max_{\ell,t}\mathrm{IE}$, usually restricted to last-subject-token MLP outputs.

**Editing.** ROME solves a rank-one constrained least-squares update at layer $\ell$'s MLP down-projection $W^{(\ell)}\in\mathbb{R}^{d\times m}$:

$$\hat W = \arg\min_W \|WK - V\|_F \quad \text{s.t.}\quad Wk_\star = v_\star,$$

closed form $\hat W = W + (v_\star - Wk_\star)\frac{(C^{-1}k_\star)^\top}{(C^{-1}k_\star)^\top k_\star}$, $C=\mathbb{E}[kk^\top]$ from Wikipedia activations.

**Edit quality**, all measured as rates on CounterFact: efficacy $E$ (new object outranks old on the edit prompt), generalization $G$ (same on paraphrases), specificity $S$ (neighborhood prompts unchanged), plus fluency and essence-drift terms.

**The predicate at issue.** Let $\rho$ be the rank correlation, across facts, between $\mathrm{IE}(s^\star)$ and edit score at $s^\star$; and let $\Delta(\ell)$ be edit score as a function of layer. Localization is informative iff $\rho$ is materially positive *and* $\arg\max_\ell \Delta(\ell) \approx \mathbb{E}[\ell^\star]$.

**Assumptions, with violation status.**
1. *Corruption removes only subject information.* Violated — Gaussian noise pushes activations off-manifold; restoring any state partly restores generic fluency.
2. *Single-state restoration is additively decomposable.* Violated — patching effects are non-additive across sites; total effect ≠ sum of individual effects.
3. *Activation-level necessity implies weight-level sufficiency for change.* Not established; this is the core gap.
4. *Facts are stored, not computed.* Violated for facts recoverable by co-occurrence heuristics; tracing peaks appear for prompts the model answers by lexical association rather than lookup.

## 3. State of the Art

**Established (independently reproduced).**
- Hase, Bansal, Kim, Ghandeharioun, *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing* (NeurIPS 2023) — ROME edit success is essentially flat across a wide band of layers in GPT-J-6B, and per-fact tracing effect explains near-zero variance in per-fact edit success. Their conclusion — tracing does not tell you where to edit — has not been overturned.
- Zhang & Nanda, *Towards Best Practices of Activation Patching in Language Models* (ICLR 2024) — patching results flip qualitatively with corruption type (Gaussian noise vs. symmetric token substitution) and with the metric (logit difference vs. probability vs. KL). Methodological choices, not model structure, drive a large share of the reported localization.

**Claimed but unablated.**
- That MEMIT's chosen critical layer range transfers across models; the range is re-tuned per model without an ablation isolating tracing's contribution.
- That "knowledge neurons" identify storage sites — challenged by Niu et al., *What does the Knowledge Neuron Thesis Have to do with Knowledge?* (ICLR 2024), which finds the identified neurons track token expression more than factual content.

**Benchmark-number-only.** CounterFact/zsRE efficacy scores near 100% for ROME/MEMIT are benchmark artifacts of a prompt-matching metric; Hoelscher-Obermaier et al., *Detecting Edit Failures in Large Language Models* (ACL 2023), show large specificity failures once the evaluation adds distracting context, with no corresponding drop in the headline number.

## 4. What Is Known

At the scales measured (GPT-2 XL 1.5B, GPT-J 6B, sometimes Llama-2 7B; CounterFact, ~21k facts):

- ROME reports >95% efficacy on CounterFact at its traced layer in GPT-2 XL; MEMIT scales to ~10k simultaneous edits in GPT-J with efficacy above 90%.
- Editing at layers *away* from the traced peak yields comparable efficacy and generalization; the layer-vs-score curve is broad and flat over roughly the first third to half of the network, then collapses in the last layers.
- Tracing peaks concentrate at the last subject token, mid-early MLPs — a *token-position* result that is robust; the *layer* result is what fails to transfer to editing.
- Corruption-type sensitivity is large: replacing Gaussian noise with symmetric token substitution moves the apparent critical layer.
- Sequential editing degrades models: Gupta, Rao, Anumanchipalli and follow-ups report progressive collapse over hundreds to thousands of sequential ROME edits, independent of site choice.
- Edit locality failures are systematic: edits leak to related subjects and to the same subject under added context.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed estimand connecting an inference-time intervention to a weight-space counterfactual. Until "editable site" is defined as a property with a ground-truth referent, $\rho$ has no target value. This is the dominant blocker.
- **Theoretically open.** No theorem states when representational necessity implies weight-update sufficiency, or exhibits a family where they provably decouple. A plausible route — showing the flat layer profile follows from rank-one updates plus $C^{-1}$ whitening being expressive at any sufficiently wide layer — is unproven.
- **Empirically open.** Whether the null result holds at 70B+ and for MoE/GQA architectures, and whether a *different* localization statistic (path patching, attribution patching with gradients, sparse-autoencoder feature attribution) recovers predictive validity. Runnable today; not run at scale with a proper control arm.

## 6. Why It Is Hard

**Absent ground truth plus confounded measurement.** No fact has a known storage location, so localization methods can only be validated against downstream proxies — and the only available proxy (edit success) is itself dominated by a metric that a rank-one update can satisfy from almost any wide layer. That makes the comparison non-identifiable: a flat $\Delta(\ell)$ curve is equally consistent with (a) knowledge being distributed, (b) tracing being wrong, and (c) the edit objective being too easy. Compounding it, tracing's output is not invariant to the corruption distribution or the readout metric, so "the localization" is a family of statistics, and a negative result about one member is weak evidence about the family.

## 7. Current Research (as of 2026)

- **Mechanistic interpretability with sparse autoencoders** as a replacement localization primitive — feature-level rather than layer-level attribution, then editing the identified features. Predictive validity for editing is not yet demonstrated *(frontier — verify)*.
- **Better edit evaluation**: ripple effects (Cohen et al., *Evaluating the Ripple Effects of Knowledge Editing*, TACL 2024), portability and multi-hop consistency; these give localization a harder target to predict.
- **Lifelong/sequential editing** (GRACE-style external memory, and null-space projection methods such as AlphaEdit) that sidestep site selection entirely by not choosing a "true" site — implicit evidence that localization is not the operative variable.
- **Patching methodology standardization** following Zhang & Nanda; groups at DeepMind, Northeastern (Bau lab), UNC and Technion are the visible contributors.

## 8. Concrete Next Experiment

**Question.** Does *any* localization statistic predict per-fact edit outcome once the edit metric is hard enough to discriminate?

**Scale.** Llama-3 8B and one 70B model. 2,000 CounterFact facts. For each fact, compute four localization statistics: (i) ROME Gaussian-noise tracing, (ii) symmetric-token-substitution patching, (iii) attribution patching, (iv) SAE feature attribution. For each fact, run ROME at every candidate layer $\ell \in \{1..L\}$ — $2000 \times L$ edits, each a closed-form rank-one update, feasible on 8×A100 in days.

**Control arm.** Two: (a) a *random* layer selector; (b) a *fixed* layer selector at the population-best layer. Both are what tracing must beat.

**Deciding number.** Spearman $\rho$ between each localization statistic's argmax-layer rank and the per-fact best-edit layer, and the gap in mean **ripple-effect-adjusted composite score** between tracing-selected and fixed-layer selection. Decision rule: tracing is informative iff that gap exceeds $+2$ points absolute with 95% CI excluding zero. Any statistic clearing it revives localization-guided editing; all four failing converts this entry from partially-solved to a settled negative for layer-granularity localization.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[SOTA]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[SOTA]** Fred Zhang, Neel Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR 2024. — arXiv:2309.16042
- **[Evaluation]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL 2023.
- **[Evaluation]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024.
- **[Critique]** Jingcheng Niu, Andrew Liu, Zining Zhu, Gerald Penn. *What does the Knowledge Neuron Thesis Have to do with Knowledge?* ICLR 2024.
- **[Mechanism]** Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP 2021.
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023.

## 10. Worked Example

Fact: *"The Space Needle is located in the city of"* → *Seattle*.

1. Clean run: $\mathbb{P}[\text{Seattle}] \approx 0.90$ in GPT-2 XL.
2. Corrupt the three subject tokens with $\nu\!=\!3\sigma$ noise: $\mathbb{P}[\text{Seattle}]$ drops to roughly $0.02$.
3. Restore last-subject-token MLP output, layer 17: recovery to $\approx 0.6$, so $\mathrm{IE}(17,t_{\text{subj}})\approx 0.58$ — the tracing peak. Restore at layer 40: $\mathrm{IE}\approx 0.03$.
4. ROME-edit the fact to *Paris*. At layer 17: efficacy 1, paraphrase generalization holds.
5. **Now the control.** Edit at layer 5, where $\mathrm{IE}\approx 0.10$ — six times weaker. Efficacy is still 1, generalization comparable. Repeat at layers 3, 8, 12: still successful.

The obstruction is visible in step 5. $\mathrm{IE}$ varies by a factor of ~6 across layers where the edit outcome is indistinguishable. The tracing statistic has real dynamic range; the edit metric has none over the same interval. So the correlation is not "small because the signal is noisy" — it is undefined-in-practice because the dependent variable is saturated. Any experiment that reports $\rho \approx 0$ using CounterFact efficacy is measuring the ceiling of the benchmark, not the failure of localization. That is why §8 makes the composite ripple-adjusted score, not efficacy, the deciding number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*