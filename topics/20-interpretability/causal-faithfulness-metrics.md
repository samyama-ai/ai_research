---
id: 20-interpretability/causal-faithfulness-metrics
title: "Causal Faithfulness Metrics for Explanations"
topic: 20-interpretability
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Causal Faithfulness Metrics for Explanations

> **Topic:** Interpretability · **ID:** `20-interpretability/causal-faithfulness-metrics` · **Status:** methodologically-blocked

## 1. Problem Statement

An explanation $e$ of a model $f$ on input $x$ claims that some part of the computation is responsible for the output. **Faithfulness** is the degree to which that claim is true of $f$'s actual mechanism, not of a plausible story a human accepts. The problem is to define and compute a scalar $\mathrm{Faith}(e, f, \mathcal{D}) \in [0,1]$ that is (i) causal — grounded in interventions on $f$, not in correlations; (ii) comparable across explanation types (token attributions, circuits, features, natural-language rationales, chain-of-thought); and (iii) not gameable by explanations that score well while describing nothing real.

Three variants, of very different difficulty:

- **Measurement.** Given $f$, $e$, and an intervention budget, output a faithfulness score whose ordering over explanations matches ground-truth mechanism recovery. *This is the blocked variant: current scores depend on arbitrary choices (ablation distribution, output metric, dataset) in ways that reverse rankings.*
- **Method.** Produce explanations that maximise an agreed score. Tractable — and the reason the measurement variant matters, since optimising a bad metric is cheap.
- **Theory.** Prove identifiability: conditions under which a faithfulness functional has a unique maximiser at the true causal structure, and impossibility results where it does not. Largely untouched.

Solved would mean: a metric with a stated invariance class, a proof or strong evidence that it is not maximised by known illusions, and independently reproduced agreement with ground truth on models where ground truth exists.

## 2. Formal Setting

Let $f: \mathcal{X} \to \Delta(\mathcal{Y})$ be a network with computational graph $G = (V, E)$; $V$ are nodes (attention heads, MLP neurons, residual-stream subspaces, SAE latents) with activations $h_v(x) \in \mathbb{R}^{d_v}$. An explanation is a hypothesis $H = (S, \phi)$: a node subset $S \subseteq V$ and an optional interpretation map $\phi$ assigning each $v \in S$ a high-level variable.

**Intervention.** $\mathrm{do}(h_S \leftarrow a)$ replaces activations on $S$ with $a$ and runs the rest of $f$ forward. The **counterfactual** $f_{S \leftarrow a}(x)$ is measured by a single forward pass; cost is $O(|S|)$ passes for node-wise sweeps, $O(|E|)$ for edge-level attribution.

**Ablation faithfulness.** With an ablation distribution $\mathcal{A}$ (zero, mean over $\mathcal{D}$, resample from a counterfactual prompt distribution, or Gaussian noise) and an output metric $m$ (logit difference, probability, KL, task accuracy):

$$\mathrm{Faith}_{\mathcal{A},m}(S) \;=\; 1 - \frac{\mathbb{E}_{x\sim\mathcal{D},\,a\sim\mathcal{A}}\big[\,\big|\,m(f(x)) - m(f_{\bar S \leftarrow a}(x))\,\big|\,\big]}{\mathbb{E}_{x,a}\big[\,\big|\,m(f(x)) - m(f_{V \leftarrow a}(x))\,\big|\,\big]},$$

where $\bar S = V \setminus S$: *keep the claimed circuit, ablate everything else*. Note the metric is a function of three free choices $(\mathcal{A}, m, \mathcal{D})$, none of which the explanation itself specifies.

**Interchange-intervention accuracy (IIA).** For a high-level causal model $M$ with variable $\Pi$ and alignment $\phi$, IIA is the fraction of source/base pairs $(x, x')$ on which patching $\phi^{-1}(\Pi)$ from $x'$ into $x$ reproduces $M$'s counterfactual output:

$$\mathrm{IIA}(\phi) = \Pr_{x,x'}\Big[\, f_{\phi^{-1}(\Pi) \leftarrow h(x')}(x) = M_{\Pi \leftarrow \Pi(x')}(x) \,\Big].$$

**Assumptions, and their status.**
1. *Ablation stays on-manifold.* Violated. Zero- and mean-ablation put the residual stream far outside its empirical support; measured effects then mix "this component mattered" with "the model was pushed off distribution".
2. *Node independence / no self-repair.* Violated. Backup heads and layer-norm rescaling restore the output after ablation, so a genuinely causal component can score near zero.
3. *A single output metric ranks explanations consistently.* Violated — logit difference and probability disagree in sign on published circuit claims.
4. *The interpretation map $\phi$ is fixed before scoring.* Routinely violated: $\phi$ is optimised (e.g. distributed alignment search) on the same data used to report IIA, so IIA is a training score unless a held-out task distribution is used.

## 3. State of the Art

**Established.**
- *Sanity checks* (Adebayo et al., NeurIPS 2018): several saliency methods are invariant to randomising model parameters or labels — they cannot be faithful, whatever their human plausibility.
- *ROAR* (Hooker et al., NeurIPS 2019): if you remove top-ranked pixels and **retrain**, most attribution methods degrade accuracy no more than a random ranking. Establishes that unretrained deletion curves conflate importance with distribution shift.
- *Activation patching / causal mediation* (Vig et al., NeurIPS 2020; Meng et al., NeurIPS 2022) reliably localises effects that later edits reproduce.
- *Causal abstraction* (Geiger et al., NeurIPS 2021; Geiger et al., CLeaR 2024) gives the only fully specified formal criterion: $f$ is faithful to $M$ under $\phi$ iff all interchange interventions commute.
- *Metric sensitivity* (Zhang & Nanda, ICLR 2024): patching conclusions depend on the choice of corruption and output metric; documented reversals.

**Claimed but unablated.** Comprehensiveness/sufficiency scores from ERASER (DeYoung et al., ACL 2020) are widely reported as "faithfulness" without evidence that they order explanations correctly — they are deletion scores under assumption 1. Circuit papers report "the circuit recovers $X\%$ of the logit difference" as a benchmark number with no calibration of what fraction a wrong circuit would recover. Causal scrubbing (Chan et al., Redwood Research, 2022) is a well-specified protocol but its pass/fail threshold is a chosen constant.

**Benchmark-number-only.** Almost all cross-method faithfulness leaderboards. There is no dataset of models with known ground-truth mechanisms on which competing metrics have been compared head to head at scale.

## 4. What Is Known

- **Randomisation failures.** Guided BackProp and Guided Grad-CAM produce visually unchanged maps after full parameter randomisation of Inception v3 on ImageNet (Adebayo et al., 2018).
- **Retraining flips rankings.** On ImageNet/ResNet-50, ROAR at 10–90% feature removal found gradient, integrated-gradients and guided-backprop rankings no better than random; only ensembling variants (SmoothGrad-Squared, VarGrad) beat the random control (Hooker et al., 2019).
- **Circuit metrics are not robust.** Re-evaluations of the GPT-2 small IOI circuit (Wang et al., ICLR 2023; 26 heads) show reported faithfulness moves substantially with ablation type and with whether ablation is node- or edge-level (Miller, Chughtai & Saunders, COLM 2024).
- **Subspace patching can succeed on a dormant direction.** Makelov, Lange & Nanda (ICLR 2024) show a subspace can pass interchange-intervention tests on IOI in GPT-2 small while the model does not use it in the unintervened forward pass — a metric maximised by a non-mechanism.
- **Verbalised reasoning is measurably unfaithful.** Turpin et al. (NeurIPS 2023): adding an answer-ordering bias to few-shot prompts drops GPT-3.5/Claude 1.0 accuracy by up to ~36 points on BBH tasks while chain-of-thought never mentions the bias. Lanham et al. (Anthropic, 2023) find early-answering and paraphrase tests give faithfulness that *decreases* with model scale on models up to 175B.
- **Self-repair is real.** Ablating a name-mover head in GPT-2 small activates backup heads that partially restore the logit difference — so single-node ablation understates causal role.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No agreed definition of the reference distribution $\mathcal{A}$. Every score above is a score of *(explanation, ablation choice)*, and the choices demonstrably reorder methods. Until $\mathcal{A}$ is pinned by a stated invariance requirement, "faithfulness" names a family, not a quantity.
- **Methodologically blocked.** No ground-truth corpus. Faithfulness metrics are validated against other faithfulness metrics.
- **Theoretically open.** Identifiability: is the faithfulness functional maximised uniquely at the true abstraction, or is the illusion of Makelov et al. generic? No impossibility theorem, no positive identifiability result under stated conditions.
- **Theoretically open.** Composition: whether faithfulness of parts implies faithfulness of the whole under self-repair.
- **Empirically open.** Whether any current metric transfers above chance from models with planted ground truth to frontier-scale LLMs. Runnable today; unrun at scale.

## 6. Why It Is Hard

**Absent ground truth compounded by confounded measurement.** For real networks nobody knows the true mechanism, so the metric cannot be validated; and the only available probe — ablation — necessarily moves activations off the data manifold, so every score mixes *causal importance* with *distribution shift*. Retraining removes the shift (ROAR) but then measures a different model. Self-repair adds a second confound in the opposite direction: true components can be ablated with no output change. The two error sources have opposite signs and no independent estimate, so a score of 0.8 cannot be decomposed. Finally, when $\phi$ is fitted, the metric is optimised over the same hypothesis space it is meant to test — a non-identifiability, not a compute problem.

## 7. Current Research (as of 2026)

- **Causal abstraction and alignment search** — Geiger, Potts, Icard and collaborators (Stanford): IIA, distributed alignment search, and the "right mediator" framing (Mueller et al., 2024 survey).
- **Circuit-metric robustness** — Nanda's group and independent replications; standardising ablation type, edge- vs node-level scoring, and held-out counterfactual distributions.
- **SAE-based faithfulness** — whether sparse-autoencoder latents give an ablation basis that stays on-manifold (Anthropic, EleutherAI, Goodfire). Evidence is mixed; reconstruction error itself perturbs the forward pass. *(frontier — verify)*
- **CoT faithfulness under RL training** — whether outcome-based RL makes verbalised reasoning less load-bearing; measured by paraphrase, truncation and hint-insertion tests. *(frontier — verify)*
- **Planted-mechanism benchmarks** — small transformers trained on tasks with a known algorithm, used as ground truth for metric validation. Still small-scale.

## 8. Concrete Next Experiment

**Question.** Do faithfulness metrics rank explanations correctly when the true mechanism is known?

**Scale.** Train 20 two-layer, 4-head transformers ($\sim$3M params) on synthetic tasks with a *specified* algorithm (modular addition, IOI-style template matching, hierarchical bracket matching) using interchange-intervention training, so the ground-truth circuit is known by construction. Cost: under 200 GPU-hours on one A100.

**Arms.** For each model generate: (a) the true circuit; (b) 50 decoys — random subsets matched on size and layer profile; (c) an adversarial decoy built to maximise IIA on a dormant subspace, following the Makelov construction. Score all under the $4\times4$ grid of $\mathcal{A} \in \{$zero, mean, resample, Gaussian$\}$ × $m \in \{$logit diff, prob, KL, accuracy$\}$, plus causal scrubbing and IIA on a *held-out* task distribution.

**Control arm.** Random size-matched subsets — the null any metric must beat.

**Deciding number.** Rank correlation (Kendall's $\tau$) between metric score and ground-truth overlap, and separately, the **adversarial separation gap** $\Delta = \mathrm{Faith}(\text{true}) - \mathrm{Faith}(\text{adversarial decoy})$. A metric is usable if $\tau > 0.8$ *and* $\Delta > 0$ under **all 16** $(\mathcal{A}, m)$ settings. Report the fraction of the 16 settings in which each metric's ranking of true vs. decoy flips: any metric with flip rate $>0$ is a family, not a measurement. Prediction: every current metric has flip rate $>0$.

## 9. Key References

- **[Foundational]** Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim. *Sanity Checks for Saliency Maps.* NeurIPS 2018. — arXiv:1810.03292
- **[Foundational]** Hooker, Erhan, Kindermans, Kim. *A Benchmark for Interpretability Methods in Deep Neural Networks.* NeurIPS 2019. — arXiv:1806.10758
- **[Foundational]** Jacovi, Goldberg. *Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?* ACL 2020. — arXiv:2004.03685
- **[Foundational]** Vig, Gehrmann, Belinkov, Qian, Nevo, Singer, Shieber. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS 2020.
- **[SOTA]** Geiger, Lu, Icard, Potts. *Causal Abstractions of Neural Networks.* NeurIPS 2021.
- **[SOTA]** Geiger, Wu, Potts, Icard, Goodman. *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations.* CLeaR 2024.
- **[SOTA]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023.
- **[SOTA]** Zhang, Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR 2024.
- **[SOTA]** Makelov, Lange, Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR 2024.
- **[SOTA]** Miller, Chughtai, Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM 2024.
- **[Empirical]** Turpin, Michael, Perez, Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS 2023.
- **[Empirical]** Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic technical report, 2023.
- **[Benchmark]** DeYoung, Jain, Rajani, Lehman, Xiong, Socher, Wallace. *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* ACL 2020.
- **[Survey]** Mueller, Geiger, Wiegreffe, et al. *The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability.* 2024.
- **[Survey]** Räuker, Ho, Casper, Hadfield-Menell. *Toward Transparent AI: A Survey on Interpreting the Inner Workings of Neural Networks.* IEEE SaTML 2023.

## 10. Worked Example

**Instance.** GPT-2 small, IOI task: *"When Mary and John went to the store, John gave a drink to ___"*. Metric $m$ = logit difference between " Mary" and " John"; clean value $\approx 3.5$ nats averaged over templates.

Take the published 26-head circuit $S$ and score it by keeping $S$ and ablating $\bar S$.

| Ablation $\mathcal{A}$ | Recovered logit diff | $\mathrm{Faith}$ |
|---|---|---|
| Resample from ABC-distractor prompts | $\approx 3.2$ | $\approx 0.9$ |
| Mean over the IOI distribution | can exceed clean, $>3.5$ | $>1$ |
| Zero-ablation | collapses, near $0$ | $\approx 0$ |

Three numbers, one circuit, one model, one metric $m$ — spanning the whole $[0,1]$ range and overshooting it. Nothing about the explanation changed; only $\mathcal{A}$ did.

Now the confound has a visible direction. Under zero-ablation the residual stream is off-manifold and layer-norm rescales what remains, so the low score is measuring distribution shift, not absence of mechanism. Under mean-ablation the mean is computed over IOI prompts and already carries task information, so ablated components leak the answer and the score inflates past 1 — the circuit appears to explain *more* than the full model. Resampling from matched counterfactual prompts avoids both, but "matched" is a modelling choice made by the researcher, and swapping the distractor distribution moves the number again.

Add the adversarial case: Makelov et al. exhibit a residual-stream subspace on this exact task that scores highly under interchange intervention while carrying no signal in the unintervened forward pass. So the metric assigns a near-passing score to a component the model does not use.

The obstruction is now explicit. To pick among the three rows you need to know which one reflects the mechanism — which is the thing the metric was supposed to tell you. The measurement presupposes its own answer.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*