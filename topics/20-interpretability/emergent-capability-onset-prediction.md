---
id: 20-interpretability/emergent-capability-onset-prediction
title: "Emergent Capability Onset Prediction From Internals"
topic: 20-interpretability
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Emergent Capability Onset Prediction From Internals

> **Topic:** Interpretability · **ID:** `20-interpretability/emergent-capability-onset-prediction` · **Status:** empirically-open

## 1. Problem Statement

Given access to the *internals* of a model that does not yet have a capability — activations, weights, attention patterns, sparse-autoencoder (SAE) features, circuit-level structure — predict *when* along the scaling or training axis that capability will appear, before it appears in any downstream metric.

- **Input:** a checkpoint sequence $\{\theta_t\}$ (training axis) or a model ladder $\{\theta^{(N_i)}\}$ (parameter axis), with read access to all intermediate activations on a probe corpus.
- **Output:** a predicted onset point $\hat{c}^\star$ (compute, tokens, or parameters) at which downstream task accuracy $A$ crosses a threshold $\tau$, plus a calibrated interval.
- **Decision predicate:** the prediction is made using only checkpoints where $A \le$ chance, and is scored against the realized crossing $c^\star$.

Three variants, of very different difficulty:

- **Measurement variant.** Define an internal statistic whose trajectory is non-flat while $A$ is still flat. Partly solved (§4).
- **Method variant.** Turn that statistic into a *forecast* with error bars, extrapolating across a compute gap of at least one order of magnitude. Empirically open.
- **Theory variant.** Prove that some class of internal observable must move before behavioral emergence — or exhibit a model family where no polynomial-time internal observable does. Theoretically open.

A solution to the method variant is: a procedure that, held out on capabilities and model families it was not tuned on, predicts $\log_{10} c^\star$ to within a stated tolerance more often than the stated coverage would allow by chance.

## 2. Formal Setting

Let $M_\theta$ be an autoregressive model. Fix a task $T$ with per-example scorer $s(y, y^\star) \in \{0,1\}$ (exact match) and dataset $D_T$. Measured accuracy:

$$A(\theta) = \frac{1}{|D_T|}\sum_{(x,y^\star)\in D_T} s\big(\text{decode}(M_\theta, x), y^\star\big).$$

*As measured:* greedy decoding, fixed prompt template, fixed $k$-shot count, $|D_T| \ge 500$ so the binomial standard error at $A=0.05$ is $\le 0.01$.

Compute $c = 6 N D$ FLOPs ($N$ non-embedding parameters, $D$ tokens). Onset:

$$c^\star(\tau) = \inf\{c : A(\theta_c) \ge \tau\}, \qquad \tau = A_{\text{chance}} + \delta.$$

*As measured:* $\tau$ must be pinned to chance for the task's answer set; $\delta = 0.05$ absolute is the common but arbitrary choice, and $c^\star$ is sensitive to it.

An **internal observable** is any $\phi: \Theta \to \mathbb{R}^k$ computable from weights and activations on a corpus $C$ disjoint from $D_T$, without $s$. Examples used in practice:

- *Linear probe accuracy:* $\phi_{\text{probe}} = \max_{w} \text{acc}(w^\top h^{(\ell)}(x))$ over layer $\ell$, on labels derived from $T$'s latent structure.
- *Induction score:* mean attention mass a head places on the token following the previous occurrence of the current token, in a repeated random sequence.
- *SAE feature presence:* $\phi_{\text{SAE}}$ = whether a dictionary feature with a given causal role exists and its mean activation on $T$-relevant inputs.
- *Continuous surrogate:* $\phi_{\text{Brier}} = \mathbb{E}[p_\theta(y^\star \mid x)]$ or per-token log-likelihood of the gold answer.

The forecasting problem: fit $f$ on $\{(\phi(\theta_c), c) : c \le c_0\}$ with $A(\theta_{c_0}) \le \tau$, and emit $\hat{c}^\star = f(\cdot)$. Score by $|\log_{10} \hat{c}^\star - \log_{10} c^\star|$ and by interval coverage.

**Assumptions, with the ones known to be violated flagged:**

1. *$A$ is monotone in $c$.* Violated — inverse scaling exists (McKenzie et al., TMLR 2023).
2. *A single scalar $c$ orders checkpoints.* Violated — at fixed $c$, data mixture and repetition change $c^\star$ by more than a factor of 2.
3. *$\phi$ is comparable across scales.* Violated in practice: residual-stream norms, layer counts, and SAE dictionaries all change with $N$, so $\phi$ is not the same function on two models in a ladder.
4. *Emergence is a property of the model, not the metric.* Contested — see Schaeffer et al. (§3).
5. *The probe corpus is independent of $D_T$.* Routinely violated by pretraining contamination.

## 3. State of the Art

**Established (reproduced, ablated):**

- **Metric-induced sharpness.** Schaeffer, Miranda & Koyejo, *Are Emergent Abilities of Large Language Models a Mirage?* (NeurIPS 2023, best paper) show that replacing exact-match with a continuous metric (token edit distance, Brier score) turns many sharp curves into smooth ones, and that sharp curves can be *induced* in vision autoencoders by choosing a discontinuous metric. This is the strongest single result in the area and it is properly ablated.
- **Induction heads as a mechanistic phase change.** Olsson et al., *In-context Learning and Induction Heads* (Anthropic, 2022) identify a visible loss bump co-occurring with induction-head formation, and show ablating induction heads removes most of the in-context learning score gain.
- **Hidden progress before behavioral onset.** Barak et al., *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit* (NeurIPS 2022) prove and demonstrate that a Fourier-gap statistic moves monotonically while population accuracy sits at chance.
- **Progress measures for grokking.** Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* (ICLR 2023): on modular addition, restricted-loss and excluded-loss measures rise well before test accuracy, on a fully reverse-engineered circuit.

**Claimed but unablated, or benchmark-number-only:**

- **Emergence prediction by finetuning.** Snell et al., *Predicting Emergent Capabilities by Finetuning* (2024) fit an "emergence law" that shifts onset with finetuning data volume and extrapolate to the few-shot point; reported as accurate up to about $4\times$ of compute on a handful of BIG-bench tasks. Uses behavior, not internals, and has not been independently reproduced across families.
- **Infinite-resolution evaluation.** Hu et al., *Predicting Emergent Abilities with Infinite Resolution* (ICLR 2024) propose PassUntil, a sampling estimator that resolves task performance down to $\sim10^{-5}$, and report smooth power-law fits below the visible onset. This is a measurement improvement; the claim that it removes emergence is task-dependent and only partly ablated.
- **Observational scaling laws.** Ruan, Maddison & Hashimoto (ICML 2024) show ~100 public models' capabilities lie near a low-dimensional PCA subspace of benchmark scores, enabling cheaper prediction. Uses benchmark scores as "internals-adjacent" features, not activations.
- **Skill-mix theory.** Arora & Goyal, *A Theory for Emergence of Complex Skills in Language Models* (2023) derive emergence of $k$-skill composition from scaling laws plus a random bipartite skill graph. Elegant, but the skill graph is not measured in any real model.

No published method predicts onset from *activations or circuits* across a $10\times$ compute gap with calibrated intervals. That is the SOTA gap.

## 4. What Is Known

- **Sharpness is partly a metric artifact.** Schaeffer et al. (2023): across BIG-bench, sharp emergent curves are concentrated in a small number of metrics — multiple-choice grade and exact string match account for the large majority of claimed emergent pairs; ~92% of curves labelled emergent in BIG-bench used one of those two scorers.
- **Latent skill precedes behavior on toy tasks, with margins.** Modular addition, 1-layer transformer, $p=113$: Fourier-basis structure and restricted loss move thousands of optimizer steps before test accuracy leaves chance (Nanda et al., ICLR 2023).
- **Induction bump is narrow and reproducible.** In Anthropic's 2022 runs (models from ~1M to ~13B parameters), the in-context learning score — the difference in loss between the 500th and 50th token of a context — undergoes its jump inside a window of roughly $2$–$5\times10^9$ training tokens, at essentially the *same* token count across model sizes. That scale-invariance is the single most encouraging datum for the method variant.
- **Sub-threshold signal exists at frontier scale.** PassUntil (Hu et al., ICLR 2024) measures nonzero task performance at accuracies around $10^{-5}$ on code-generation tasks in a 2.4B-parameter run family where standard evaluation reads exactly 0.
- **Loss is predictable; task accuracy is much less so.** Chinchilla-style loss laws (Hoffmann et al., NeurIPS 2022) fit within a few percent, while GPT-4's technical report notes that some capabilities (e.g. Hindsight Neglect) were not predicted by loss extrapolation.
- **Some "emergence" is in-context learning.** Lu et al., *Are Emergent Abilities in Large Language Models just In-Context Learning?* (ACL 2024): after controlling instruction tuning and ICL, most of 21 tested emergent abilities on BIG-bench reduce to ICL — with formal reasoning as the residual exception.

## 5. What Is Not Known

- **Empirically open.** Whether an internal observable measured only at $c \le c_0$ predicts $c^\star$ within $\pm 0.3$ dex across a $10\times$ gap on held-out tasks. The experiment is runnable — it needs a dense checkpoint ladder with activations retained, which Pythia and OLMo partly provide — and nobody has run it as a *blind forecast* with pre-registered tolerances.
- **Empirically open.** Whether SAE features that will later carry a capability are already present (at low density) before onset, or are created at onset. Requires cross-checkpoint dictionary matching, which is not standardized.
- **Methodologically blocked.** "The capability" has no metric-independent definition. Since $c^\star$ depends on $\tau$, the scorer, the prompt, and the shot count, the *target of prediction* is not well posed. Until an onset definition is invariant to these, prediction error is partly a measure of the definition.
- **Methodologically blocked.** Cross-scale comparability of $\phi$. There is no accepted way to say a feature in a 410M model "is the same feature" as one in a 12B model.
- **Theoretically open.** No theorem states that a polynomial-time internal observable must be non-constant before a behavioral phase transition. The parity results (Barak et al. 2022) are the closest positive case; whether hard-instance families exist where internals are provably uninformative until onset is unproven either way.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth, at the level of the target.** $c^\star$ is not a physical constant; it is a function of the scorer. Two labs measuring the same model disagree on the onset of the same "capability" by more than the accuracy any predictor claims. You cannot validate a forecaster against a moving target.
2. **Non-identifiability of internals across scale.** Any $\phi$ must be computed on models with different widths, depths, and learned bases. Alignment procedures (SVCCA, dictionary matching, probe transfer) each impose a different correspondence, and the choice changes the extrapolated $\hat{c}^\star$ — so the prediction is not identified by the data.
3. **Compute cost of the only decisive design.** Deciding the question needs ladders where the *only* varied factor is scale, with dense checkpoints and stored activations, across at least two families. A 6-point ladder to 7B with 20 checkpoints each is a multi-hundred-thousand-GPU-hour commitment, which is why the literature reuses Pythia and OLMo rather than running the clean design.

## 7. Current Research (as of 2026)

- **Sparse dictionary learning at scale.** Anthropic (Templeton et al., *Scaling Monosemanticity*, 2024) and the attribution-graph line (Lindsey et al., 2025) give feature-level and circuit-level objects to track; applying them *longitudinally across checkpoints* is the natural next step and is being attempted *(frontier — verify)*.
- **Open checkpoint suites.** EleutherAI's Pythia (Biderman et al., ICML 2023: 154 checkpoints × 8 sizes) and AI2's OLMo remain the only public substrate for this problem.
- **Predictive-evaluation methods.** Berkeley (Snell et al.) on emergence laws; Stanford (Hashimoto group) on observational scaling; Tsinghua/ModelBest on PassUntil-style high-resolution scoring.
- **Developmental interpretability.** Timaeus and collaborators use the local learning coefficient (a singular-learning-theory estimate of effective parameter count) to detect developmental stages; Hoogland et al. report stagewise structure in small transformers. Whether it *forecasts* rather than *labels* stages is unresolved *(frontier — verify)*.

## 8. Concrete Next Experiment

**Blind onset forecast on Pythia, one dex forward.**

- **Scale.** Pythia 160M, 410M, 1B, 1.4B (fit set) and 6.9B, 12B (held out) — a $\ge 10\times$ compute gap. All 154 public checkpoints per model. 12 tasks with documented sharp curves plus 4 smooth controls.
- **Predictor.** For each task, compute two internal observables at every fit-set checkpoint: (a) best-layer linear probe accuracy on the task's latent variable, using a probe corpus disjoint from $D_T$; (b) mean gold-answer log-likelihood. Fit $\hat{c}^\star$ by extrapolating the probe trajectory to the compute at which behavioral $\tau=0.05$ is reached, calibrating the probe-to-behavior offset on the fit set only. Register predictions and $\pm$ intervals *before* evaluating the 6.9B/12B models.
- **Control arms.** (1) Loss-only baseline: Chinchilla fit to validation loss, then a task-accuracy-vs-loss curve from fit-set models. (2) Behavior-only baseline: PassUntil-style high-resolution accuracy from fit-set models extrapolated as a power law. (3) Shuffled-label probe, to confirm the probe carries task-specific and not generic signal.
- **Deciding number.** Mean absolute error in $\log_{10} c^\star$ on the 12 held-out sharp tasks. **Internals win iff MAE $\le 0.3$ dex and strictly below both baselines by $\ge 0.15$ dex, with 80% interval coverage in $[0.7, 0.9]$.** If MAE $> 0.5$ dex or indistinguishable from the loss-only arm, the measurement variant does not yet transfer and the effort should move to fixing the onset definition instead.
- **Cost.** Forward passes only; low tens of thousands of GPU-hours. This is the cheapest decisive design available today.

## 9. Key References

- **[Foundational]** Wei, J. et al. *Emergent Abilities of Large Language Models.* TMLR, 2022. — arXiv:2206.07682
- **[Foundational]** Olsson, C. et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895
- **[SOTA]** Schaeffer, R., Miranda, B., Koyejo, S. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[SOTA]** Hu, S. et al. *Predicting Emergent Abilities with Infinite Resolution.* ICLR, 2024. — arXiv:2310.03262
- **[SOTA]** Snell, C., Kostrikov, I., Su, Y., Yang, M., Levine, S. *Predicting Emergent Capabilities by Finetuning.* 2024. — arXiv:2411.16035
- **[SOTA]** Ruan, Y., Maddison, C. J., Hashimoto, T. *Observational Scaling Laws and the Predictability of Language Model Performance.* ICML, 2024. — arXiv:2405.10938
- **[Theory]** Barak, B., Edelman, B. L., Goel, S., Kakade, S., Malach, E., Zhang, C. *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit.* NeurIPS, 2022. — arXiv:2207.08799
- **[Theory]** Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR, 2023. — arXiv:2301.05217
- **[Theory]** Arora, S., Goyal, A. *A Theory for Emergence of Complex Skills in Language Models.* 2023. — arXiv:2307.15936
- **[Substrate]** Biderman, S. et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML, 2023. — arXiv:2304.01373
- **[Methods]** Templeton, A. et al. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread, Anthropic, 2024.
- **[Survey/Contra]** Lu, S., Bigoulaeva, I., Sachdeva, R., Tayyar Madabushi, H., Gurevych, I. *Are Emergent Abilities in Large Language Models just In-Context Learning?* ACL, 2024. — arXiv:2309.01809
- **[Scaling]** Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556

## 10. Worked Example

**Task:** 3-digit integer addition, 5-shot, exact match. Chance $\approx 10^{-3}$; onset threshold $\tau = 0.05$.

**Step 1 — behavior.** On a Pythia-style ladder, exact-match accuracy reads $0.00$ at 160M and 410M, and crosses $0.05$ somewhere between 1.4B and 6.9B. Take the realized crossing at $N^\star \approx 3\times10^9$ parameters, $D = 300$B tokens: $c^\star = 6ND \approx 5.4\times10^{21}$ FLOPs.

**Step 2 — internals.** At 410M, train a linear probe on the residual stream at layer $\ell$ to read the carry bit of the units column from the prompt representation. Suppose the probe reaches 0.78 accuracy (chance 0.50) while exact match is 0.00. Signal exists sub-threshold — the measurement variant works.

**Step 3 — extrapolate.** Probe accuracy across 160M/410M/1B/1.4B: $0.61, 0.78, 0.86, 0.90$. Fit $\text{acc} = 1 - a c^{-b}$ in log-compute. This saturates toward 1.0 and gives no crossing point by itself; you must additionally posit the probe accuracy $\rho^\star$ at which behavior crosses $\tau$. Calibrate $\rho^\star$ on a *different* task and you get, say, $\rho^\star = 0.94 \Rightarrow \hat{c}^\star = 1.1\times10^{21}$ — a $0.69$ dex underestimate. Calibrate at $\rho^\star = 0.97$ and you get $\hat{c}^\star = 2.0\times10^{22}$, a $0.57$ dex overestimate.

**The obstruction, visible.** A $0.03$ change in a nuisance parameter that is itself estimated from a handful of tasks swings the forecast by 1.3 orders of magnitude — wider than the gap being predicted. Compounding it, the probe's meaning is not fixed across the ladder: layer $\ell$ is layer 12 of 24 at 410M and layer 18 of 36 at 6.9B, and the best-layer choice alone moves the fitted curve by 0.04–0.06 accuracy at each point. The bottleneck is not the absence of internal signal. It is that the map from internal signal to a behavioral threshold is unidentified, and the threshold itself is a convention.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*