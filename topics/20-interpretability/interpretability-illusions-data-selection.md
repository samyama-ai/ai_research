---
id: 20-interpretability/interpretability-illusions-data-selection
title: "Interpretability Illusions From Cherry-Picked Data"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Interpretability Illusions From Cherry-Picked Data

> **Topic:** Interpretability · **ID:** `20-interpretability/interpretability-illusions-data-selection` · **Status:** open

## 1. Problem Statement

Most interpretability claims about a component (a neuron, an SAE latent, an attention head, a residual-stream direction, a circuit) are induced from a small, non-random sample of inputs: the top-$k$ activating examples, a curated probing set, a templated task like IOI, or a hand-built counterfactual pair set. The claim is then stated as if it held over the deployment distribution. An **interpretability illusion** is the gap between the two.

Three variants, of different difficulty:

- **Measurement.** Define a statistic that detects the illusion: given an interpretation $h$ induced from selected data $S$, quantify how much of $h$'s apparent support is an artifact of how $S$ was chosen. Requires a faithfulness metric that is itself not selection-dependent.
- **Method.** Build selection procedures and estimators whose explanations transfer: same explanation, new corpus, no loss of predictive or causal accuracy.
- **Theory.** Characterize when top-$k$ selection is identifiable — when the maximum-activation subset determines the function on the rest of the support, and when it provably does not.

Solving it means: an interpretability result ships with a selection-robustness number, and that number is predictive of behaviour on data the analyst never saw.

## 2. Formal Setting

Model $M$ with hidden activation $a_v(x) \in \mathbb{R}$ for component $v$ at token/input $x$, deployment distribution $\mathcal{D}$ over inputs.

**Selection operator.** $\sigma: \mathcal{D}^n \to 2^{\mathcal{X}}$. Top-$k$: $S_{\text{top}} = \arg\max_{|S|=k} \sum_{x\in S} a_v(x)$ over a corpus draw $X \sim \mathcal{D}^n$. Measured as: sample $n$ tokens (typical $n \in [10^6, 10^8]$), keep the $k$ (typical $k \in [10,50]$) with largest $a_v$.

**Explanation.** $e = E(S)$, a natural-language string or a predicate $\hat{p}_e: \mathcal{X} \to [0,1]$. Measured as: a scoring LLM's predicted activation, rescaled.

**Observational faithfulness** on distribution $Q$:
$$F_{\text{obs}}(e; Q) \;=\; \mathrm{corr}_{x\sim Q}\!\left[\hat{p}_e(x),\, a_v(x)\right].$$

**Causal faithfulness**: with intervention $\mathrm{do}(a_v \leftarrow c)$ and task metric $\ell$,
$$F_{\text{cau}}(e; Q) \;=\; \mathbb{E}_{x\sim Q}\Big[\ell\big(M_{\mathrm{do}(a_v\leftarrow c)}(x)\big) - \ell\big(M(x)\big)\Big] \text{ restricted to } \{x : \hat{p}_e(x) > \tau\},$$
compared against the same quantity on $\{\hat{p}_e \le \tau\}$. Measured by activation patching or clamping with a fixed $c$ (usually $c = 0$ or the max observed).

**Illusion magnitude.**
$$\Delta(\sigma) \;=\; F(e; Q_\sigma) - F(e; \mathcal{D}), \qquad Q_\sigma = \text{empirical law of } \sigma(X).$$

**Selection-null control.** $\Delta_0$ = the same quantity for a random direction $v' \sim \mathcal{U}(S^{d-1})$ (or a randomly initialized $M$). Only $\Delta(\sigma) - \Delta_0$ is evidence about $v$.

**Activation-mass coverage.** $C(S) = \sum_{x \in S} a_v(x) \big/ \mathbb{E}_{\mathcal{D}}[n\, a_v]$ — the fraction of the component's total activation the explained set accounts for.

Assumptions, and their status:

| Assumption | Status |
|---|---|
| Corpus sample is i.i.d. from deployment $\mathcal{D}$ | **Violated** — pretraining shards ≠ chat/agent traffic |
| $\hat{p}_e$ evaluated by an LLM is unbiased | **Violated** — scorers reward fluent, generic explanations |
| Component has a single functional role | **Violated** — polysemanticity, feature absorption |
| Zero/mean ablation is on-distribution | **Violated** — off-manifold; triggers compensation (Hydra effect) |
| Explanation quality is monotone in $k$ | Unverified |

## 3. State of the Art

**Established (replicated, with controls).**
- Randomization sanity checks: saliency maps that survive randomizing model parameters or labels are not explanations (Adebayo et al., NeurIPS 2018). Guided BackProp and Guided Grad-CAM fail.
- ROAR (Hooker et al., NeurIPS 2019): retrain-after-removal on ImageNet/ResNet-50 shows several attribution methods are no better than a random ranking; the original evaluation had not held retraining fixed.
- Bolukbasi et al. (2021, arXiv:2104.07143): in BERT, top-$k$ activating sets for a direction yield coherent but corpus-dependent stories, and *random* directions in the same space produce top-$k$ sets that look as interpretable as neuron-aligned ones.
- Makelov, Lange & Nanda (ICLR 2024, arXiv:2311.17030): subspace activation patching on IOI in GPT-2 small can produce a large, apparently clean causal effect by activating a dormant parallel pathway rather than by hitting the circuit under study.
- Hase et al. (NeurIPS 2023, arXiv:2301.04213): causal-tracing localization does not predict where an edit succeeds in GPT-J 6B on CounterFact — the two rankings are near-uncorrelated.

**Claimed but unablated / benchmark-only.**
- Auto-interp explanation scores for SAE latents (from Bills et al., OpenAI 2023 onward) are reported as headline numbers; the random-direction and random-init-model control arm is usually absent. Heap et al. (2025, arXiv:2501.17727) report that SAEs trained on *randomly initialized* transformers yield auto-interp scores in the same range as SAEs on trained models — a direct hit on the metric, replicated at small scale only.
- SAEBench (Karvonen et al., 2025) aggregates multiple SAE metrics but does not report a per-metric selection-robustness number.
- Claims that scaling improves interpretability: Zimmermann et al. (NeurIPS 2023) find no such gain in vision models — a counterexample, not a general theorem.

**Theory SOTA.** Essentially none. There is no identifiability result stating conditions under which the top-$k$ preimage determines the component's function elsewhere.

## 4. What Is Known

- Selection dominates the induced story. Bolukbasi et al. (BERT-base, 110M, Wikipedia vs. other corpora): the same direction supports mutually inconsistent human-written descriptions when the source corpus changes.
- Random directions clear the interpretability bar. Same paper; the effect is at BERT scale and reappears in Heap et al. at Pythia/GPT-2 scale for SAEs on untrained models.
- Explanation scores collapse off the selected set. Bills et al. (GPT-2 XL, 6,000+ MLP neurons) report an average top-and-random explanation score near $0.15$; the random-only regime scores far lower, and human-written explanations do not close the gap. Huang et al. (BlackboxNLP 2023) find generated neuron explanations largely fail both observational and causal tests when evaluated outside the selection window.
- Simplified models are illusory OOD. Friedman et al. (ICML 2024, arXiv:2312.03656): pruned/discretized "interpretable" surrogates match the base model in-distribution and diverge sharply on held-out distributions — the simplification exploits the evaluation set.
- Ablation is not a clean readout. McGrath et al. (2023, arXiv:2307.15771) show downstream layers compensate for ablated components in Chinchilla 70B, so a null causal effect on the selected set does not license "this component does nothing".
- Probe accuracy needs a control task. Hewitt & Liang (EMNLP 2019) introduced selectivity precisely because high probe accuracy on curated data is achievable without the property being encoded.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed selection-robust faithfulness metric. $F_{\text{obs}}$ and $F_{\text{cau}}$ are both defined relative to a distribution $Q$; nobody has fixed a canonical $\mathcal{D}$ for a frontier model, so $\Delta(\sigma)$ is not currently computable in a way two labs would agree on. The auto-interp scorer is itself a model with its own selection biases; no scorer-independent ground truth exists.
- **Empirically open.** Whether $\Delta(\sigma)$ shrinks with model scale, SAE width, or $k$. Runnable today on Gemma-2 / Llama-3 class models for under a few thousand GPU-hours; not run as a controlled sweep with the random-direction arm.
- **Empirically open.** Whether any published circuit (IOI, greater-than, docstring, indirect-object) retains its measured causal effect size on a corpus drawn from real chat traffic rather than templates.
- **Theoretically open.** Identifiability: for which function classes and activation distributions does $\{(x, a_v(x)) : x \in S_{\text{top}}\}$ constrain $a_v$ on $\mathcal{D} \setminus S_{\text{top}}$? Heavy-tailed activation distributions plausibly make $k$ needed for a given error bound grow with support size; there is no bound either way.

## 6. Why It Is Hard

**Absent ground truth compounded by a confounded measurement.** There is no reference answer for "what feature $v$ represents", so faithfulness is scored by a proxy (an LLM scorer, a downstream task metric) that is evaluated on the same selected data that generated the hypothesis. The selection step and the evaluation step share a sampling operator, so the estimator of $F$ is biased upward by construction and the bias is not identifiable from the data used.

Second obstruction: **non-identifiability under heavy tails**. Component activations are extremely sparse and heavy-tailed. Top-$k$ with $k \approx 32$ over $10^7$ tokens conditions on the extreme upper tail, which is exactly the region least informative about the bulk. Fixing this by sampling the bulk destroys the signal-to-noise that made top-$k$ readable in the first place — the analyst faces a genuine variance/bias trade, not a bug.

Third: **the natural control is rarely run**. Comparing to a random direction or a random-init model costs a full re-run of the pipeline and usually lowers the headline number, so publication incentives suppress $\Delta_0$.

## 7. Current Research (as of 2026)

- Anthropic interpretability team: attribution graphs and cross-layer transcoders, with explicit reporting of when a circuit explains only a fraction of the model's behaviour on a prompt.
- Google DeepMind (Nanda's mech-interp team) and EleutherAI: SAEBench-style multi-metric evaluation, and downstream-utility tests — does the interpretation buy anything on a task the explanation was not selected on (Kantamneni et al., sparse-probing case study, 2025).
- Stanford (Potts, Geiger) and collaborators: causal-abstraction and distributed alignment search, which put the faithfulness claim in a formal interchange-intervention framework rather than a similarity score.
- Feature absorption and SAE non-determinism: multiple groups report that SAEs trained with different seeds on the same data learn substantially different latent sets *(frontier — verify the exact reproducibility figures)*.
- Open-problems agenda: Sharkey et al., *Open Problems in Mechanistic Interpretability* (2025, arXiv:2501.16496), lists illusion/validation as a top-tier gap.

## 8. Concrete Next Experiment

**Selection-transfer sweep with a random-direction control.**

- **Scale.** One open model, Gemma-2-9B (or Llama-3-8B), residual stream layer 20. Train one SAE, 65k latents, $L_0 \approx 40$, on 2B tokens of corpus A (web pretraining shard). Sample 2,000 latents.
- **Procedure.** For each latent: induce explanation $e$ from top-32 on corpus A. Score $F_{\text{obs}}$ and $F_{\text{cau}}$ on (i) held-out corpus A tokens, (ii) corpus B = real assistant-chat transcripts, (iii) corpus C = code. Score by *interval sampling* (equal counts from each activation decile), not top-$k$.
- **Control arms.** (1) 2,000 random directions in the same residual space, identical pipeline. (2) An SAE of identical size trained on a randomly initialized copy of the model. (3) Shuffled explanations (latent $i$'s explanation scored against latent $j$'s activations).
- **Deciding number.** The gap $\;\overline{F_{\text{obs}}}(\text{corpus A top-}32) - \overline{F_{\text{obs}}}(\text{corpus B, interval-sampled})\;$ **minus the same gap for the random-direction arm**. If this excess gap is $< 0.05$, current SAE explanations carry essentially no selection-robust information beyond a random direction and the field's headline auto-interp numbers are illusion. If it exceeds $0.25$, top-$k$ induction is defensible and the remaining task is calibration.
- **Cost.** One SAE training run plus ~$4 \times 2{,}000 \times 3$ scorer calls. Under 1,000 A100-hours plus scorer inference — within reach of a single academic group.

## 9. Key References

- **[Foundational]** Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim. *Sanity Checks for Saliency Maps.* NeurIPS, 2018. — arXiv:1810.03292
- **[Foundational]** Hooker, Erhan, Kindermans, Kim. *A Benchmark for Interpretability Methods in Deep Neural Networks.* NeurIPS, 2019. — arXiv:1806.10758
- **[Foundational]** Bolukbasi, Pearce, Yuan, Coenen, Reif, Viégas, Wattenberg. *An Interpretability Illusion for BERT.* 2021. — arXiv:2104.07143
- **[SOTA]** Makelov, Lange, Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR, 2024. — arXiv:2311.17030
- **[SOTA]** Friedman, Lampinen, Dixon, Chen, Ghandeharioun. *Interpretability Illusions in the Generalization of Simplified Models.* ICML, 2024. — arXiv:2312.03656
- **[SOTA]** Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[SOTA]** Heap, Lawson, Farnik, Aitchison. *Sparse Autoencoders Can Interpret Randomly Initialized Transformers.* 2025. — arXiv:2501.17727
- **[Method]** Hewitt, Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP, 2019.
- **[Method]** Huang, Geiger, D'Oosterlinck, Wu, Potts. *Rigorously Assessing Natural Language Explanations of Neurons.* BlackboxNLP @ EMNLP, 2023.
- **[Method]** McGrath, Rahtz, Kramar, Mikulik, Legg. *The Hydra Effect: Emergent Self-repair in Language Model Computations.* 2023. — arXiv:2307.15771
- **[Survey]** Sharkey et al. *Open Problems in Mechanistic Interpretability.* 2025. — arXiv:2501.16496
- **[Survey]** Belinkov. *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1), 2022.

## 10. Worked Example

An SAE latent, call it $v$, on a 9B model, layer 20. Corpus: $10^7$ tokens.

Observed: $v$ fires ($a_v > 0$) on 12,400 tokens — density $1.24 \times 10^{-3}$. The top-32 examples are all molecular-genetics text; the auto-interp explanation is *"DNA and gene sequences"*, scored $0.71$ on the standard top-and-random protocol.

Now compute coverage. Activations follow roughly $a_v \sim$ Pareto with tail index $\alpha \approx 1.8$. Total activation mass $\sum_x a_v(x) = 8{,}900$ (arbitrary units). Top-32 mass $= 340$.
$$C(S_{\text{top-32}}) = 340/8{,}900 = 3.8\%.$$
So 96.2% of the latent's activation mass, and $32/12{,}400 = 0.26\%$ of its firing tokens, played no part in generating the explanation.

Interval-sample instead: 100 tokens drawn uniformly from each activation decile. In deciles 8–10 (highest), 71/300 are genetics. In deciles 1–5, 4/500 are genetics; the rest are dominated by ordinary capitalized acronyms (`FDA`, `NATO`, `HTML`). Recomputed $F_{\text{obs}}$ on the interval sample: $0.19$.

Causal check. Clamp $a_v$ to its 99th-percentile value on 500 chat prompts containing no genetics. Next-token KL shift: $0.41$ nats, and the induced tokens are acronym-like, not genetics-like — the explanation predicts the wrong intervention outcome off the selected set.

Control arm. Run the identical pipeline on a random unit direction $v'$ in the same residual space. Its top-32 set is also thematically coherent (legal boilerplate) and its auto-interp score is $0.58$.

The obstruction is now visible as arithmetic, not opinion. The reported $0.71$ minus the random-direction floor $0.58$ leaves $0.13$ of headroom; the selection-transfer drop is $0.71 \to 0.19$. The explanation was fitted to 3.8% of the mass and does not survive the other 96.2%, and without the $v'$ arm there is no way to tell how much of the original $0.71$ was ever about $v$ at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*