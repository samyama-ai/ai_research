---
id: 02-attention/induction-head-phase-transition
title: "Induction Head Formation Phase Transition"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Induction Head Formation Phase Transition

> **Topic:** Attention Mechanisms · **ID:** `02-attention/induction-head-phase-transition` · **Status:** partially-solved

## 1. Problem Statement

During autoregressive training of a transformer, a set of attention heads implementing the rule "find the previous occurrence of the current token, attend to what followed it, predict that" — *induction heads* — appears over a narrow slice of training. In the same slice, in-context learning ability jumps and the loss curve shows a visible bump. The problem is to explain and predict this transition.

Three variants, with different difficulty:

- **Measurement.** Given checkpoints $\{\theta_t\}$ of a training run, produce a scalar $I(\theta_t)$ that is high exactly when the induction mechanism is present and functional, and is not confounded by heads that merely *correlate* with it. Currently a definitional problem, not merely a noisy one.
- **Method.** Predict, from architecture, data distribution, and optimizer alone — without training — the token count $t^\star$ at which the transition occurs, and its width $\Delta t$, to within a stated factor.
- **Theory.** Prove that gradient descent on a transformer with $\geq 2$ attention layers trained on a data distribution with in-context predictable structure exhibits a sharp (in a defined sense) transition in the induction-head order parameter, and characterize $t^\star$ as a function of distribution parameters.

Solving it means: a mechanistic account with a predictive, falsifiable formula for $t^\star$, plus a causal (not correlational) link from the transition to the in-context learning jump at scales above $1$B parameters.

## 2. Formal Setting

Model: decoder-only transformer, $L$ layers, $H$ heads/layer, width $d$, vocabulary $V$, context $T$. Head $h$ at layer $\ell$ has attention pattern $A^{(\ell,h)} \in \Delta^{T}$ per query, with $A^{(\ell,h)}_{ij}$ the weight from query position $i$ to key position $j$.

**Prefix-matching score** (Olsson et al. 2022), the standard order parameter. Feed a sequence of $T$ tokens sampled i.i.d. uniformly from a random subset of $V$ and repeated $k$ times (typically $k=2$–$4$). For query position $i$ holding token $x_i$, let $S_i = \{j < i : x_{j-1} = x_i\}$ be positions whose *predecessor* matches the current token. Then

$$ \mathrm{PM}^{(\ell,h)} \;=\; \mathbb{E}_{\text{seq}}\Big[\tfrac{1}{|Q|}\textstyle\sum_{i \in Q} \sum_{j \in S_i} A^{(\ell,h)}_{ij}\Big], \qquad I(\theta) = \max_{\ell,h} \mathrm{PM}^{(\ell,h)}. $$

Measured on 20–100 random-repeated sequences; the baseline for a uniform-attention head is $|S_i|/i \approx 1/|V_{\text{subset}}|$.

**In-context learning score.** With $\mathcal{L}_t(n)$ the mean loss at context position $n$,

$$ \mathrm{ICL}(\theta_t) = \mathcal{L}_t(500) - \mathcal{L}_t(50), $$

on held-out natural text. More negative is better. The transition is claimed when $\mathrm{ICL}$ drops sharply.

**Sharpness.** Define the transition window as $\Delta t = t_{0.9} - t_{0.1}$, the tokens between $10\%$ and $90\%$ of the asymptotic rise of $I$. The transition is *sharp* if $\Delta t / t^\star \to 0$ as some scale parameter grows. No theorem establishes this limit for a real transformer; it is an empirical description at fixed scale.

**Loss-bump signature.** With $\mathcal{L}(t)$ the training loss, the transition shows as a local maximum in $\tfrac{d}{d\log t}\mathcal{L}$ — a slowdown, sometimes a small increase, then acceleration.

Assumptions and their status:

- *A single head implements the circuit.* Violated. Induction is distributed over many heads at scale, and $\max_{\ell,h}$ discards this.
- *Prefix-matching implies induction behavior.* Violated. Singh et al. (2024) show heads with high prefix-matching whose ablation does not remove copying, and the attention pattern says nothing about the OV (output-value) copying half of the circuit.
- *Random-token probes are in-distribution.* Violated. Repeated-random sequences are far off the natural-text manifold; a head can score high on the probe and be inactive on real text.
- *K-composition through a previous-token head.* Holds in 2-layer attention-only models (Elhage et al. 2021); at scale, the "previous-token head" is often several heads plus MLP contributions.

## 3. State of the Art

**Established.**
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (Transformer Circuits Thread, 2021): full weights-level decomposition of the induction circuit in 2-layer attention-only models — a previous-token head writes into the residual stream, a second-layer head K-composes with it. Verified by direct QK/OV eigenvalue inspection, not just probes.
- Olsson et al., *In-Context Learning and Induction Heads* (2022): the phase change is real and reproducible across model sizes; six lines of evidence for the induction-head/ICL link, of which the *causal* ones (ablation, architectural surgery) run only on small attention-only models.
- Bietti et al., *Birth of a Transformer* (NeurIPS 2023): two-stage learning — global bigram statistics fast, induction slow — derived in an associative-memory model and matched in a controlled 2-layer transformer.
- Chen, Sheen, Wang, Yang (2024) and Nichani, Damian, Lee (ICML 2024): convergence proofs for gradient-based training of two-attention-layer models on Markov/causal-structure data, with a staged dynamics that produces induction.

**Claimed but unablated at scale.** That the induction phase change *causes* the ICL jump in large models. Olsson et al. state the correlation is "suggestive" for large models and explicitly mark causality as unestablished there. Later work has not closed this; ablating induction heads in a $\geq 7$B model and measuring the ICL score has not been published as a controlled, seed-replicated result.

**Benchmark-number-only.** Reported "induction head counts" in open models (Pythia, OLMo) are almost always $\mathrm{PM}$ thresholded at an arbitrary cutoff (commonly $0.3$–$0.5$). The count is a function of the threshold and the probe distribution; treat cross-paper comparisons as incomparable.

## 4. What Is Known

- **Transition location.** In Anthropic's models (Olsson et al. 2022), the loss bump and prefix-matching rise fall between roughly $2.5\times 10^9$ and $5\times 10^9$ tokens, across models from 2 layers to 40+ layers up to $13$B parameters — a striking near-invariance to model size at fixed data distribution and learning-rate schedule.
- **Minimal depth.** Two attention layers are necessary for the K-composition induction circuit; one-layer attention-only transformers implement skip-trigrams instead and cannot do it (Elhage et al. 2021).
- **Data dependence.** Chan et al. (NeurIPS 2022): in-context learning emerges only when the training distribution is bursty, has many rare classes, and has ambiguous/multi-mapped labels — Zipfian structure of natural language. Remove burstiness and the transition does not happen.
- **Sharpness has a mechanism.** Reddy (ICLR 2024): in a minimal in-context classification setting the abruptness comes from the *nested* multiplicative dependence of the loss on the previous-token subcircuit and the matching subcircuit; each is individually smooth, the product is not. Transition timing shifts predictably with burstiness and number of classes.
- **Subcircuit decomposition.** Singh et al. (NeurIPS 2024): the circuit has at least three learned pieces (previous-token, match/QK, copy/OV); clamping any one to its converged value collapses the transition delay, showing the delay is a *competition/coordination* cost, not a single slow gradient.
- **Transience.** Singh et al. (NeurIPS 2023): with continued training past the transition, ICL can *disappear*, replaced by in-weights learning. The transition is not a permanent phase.
- **Progression of statistics.** Edelman et al. (2024): on in-context Markov chains, models pass through uniform → unigram → in-context bigram in discrete stages, with plateaus between.

## 5. What Is Not Known

- **Methodologically blocked.** There is no probe-independent definition of "an induction head exists in $\theta$." $\mathrm{PM}$ measures the QK half on off-distribution inputs; no standard measure combines QK matching, OV copying, and on-distribution activity into one identified quantity. Until this is fixed, "how many induction heads" is not a well-posed question and $\Delta t$ is not comparable across papers.
- **Theoretically open.** No proof that $\Delta t/t^\star \to 0$ in any limit for a transformer trained with Adam on a Zipfian distribution. Existing proofs cover two-layer, often attention-only, models on synthetic Markov data with modified optimizers, and give staged convergence, not a proven sharp transition.
- **Theoretically open.** Why $t^\star \approx 2.5$–$5$B tokens is roughly independent of model size. No account predicts this invariance.
- **Empirically open.** Causal ablation of induction heads at $\geq 7$B parameters, with the ICL score as outcome and multiple seeds. Runnable today on Pythia-12B or OLMo-7B checkpoints; not published.
- **Empirically open.** Whether the transition is one event or several overlapping ones. At scale, the "single bump" may be an envelope of many heads forming at different times.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**. The order parameter $\mathrm{PM}$ is computed on repeated-random-token sequences, a distribution the model never trains on, and it scores only the attention pattern. A head can (i) score high and never fire on text, (ii) score low and still be part of a distributed induction computation, or (iii) score high with a broken OV circuit that copies nothing. Because the circuit is distributed and redundant at scale, single-head ablation under-reads its causal contribution while all-head ablation removes half the model — so neither arm of the causal experiment is clean.

The secondary obstruction is cost with checkpoint granularity. Resolving $\Delta t$ requires checkpoints spaced far below $\Delta t$; public suites (Pythia: 143 checkpoints, log-then-linear spacing over 300B tokens) put only a handful of checkpoints inside a transition of width $\sim 10^9$ tokens. Getting more means re-running pretraining, which no external group does for the multiple seeds needed to separate transition from seed noise.

## 7. Current Research (as of 2026)

- Dynamics proofs for two-layer attention on structured synthetic data — Princeton (Nichani, Damian, Lee), Yale (Chen, Sheen, Wang, Yang), extending toward $n$-gram and multi-task settings.
- Minimal-model sharpness analyses — Reddy (Princeton/NYU); loss-landscape and nested-nonlinearity accounts of abruptness.
- Circuit-formation dynamics — Singh, Moskovitz, Hill, Chan, Saxe (DeepMind/UCL/Gatsby), subcircuit clamping and competition.
- Developmental interpretability / singular learning theory: local learning coefficient as a stage-detection statistic over checkpoints, applied to induction formation (Timaeus and collaborators). *(frontier — verify current results.)*
- Attribution-based replacements for $\mathrm{PM}$ built on cross-layer transcoders and attribution graphs, aiming at an on-distribution circuit measure. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question:** is the induction transition a single coordinated event, and is it causally responsible for the ICL jump at $\geq 1$B scale?

- **Scale.** Train 5 seeds of a $1.4$B-parameter decoder-only model (Pythia-1.4B architecture) on 30B tokens of the Pile, checkpointing every $2\times 10^8$ tokens — $150$ checkpoints, with $\geq 15$ inside the expected $2.5$–$5$B window. About $8\times 10^{20}$ FLOPs per seed; roughly 500 A100-days total.
- **Measurement.** At each checkpoint compute, per head: $\mathrm{PM}$; an OV copying score $C^{(\ell,h)} = $ fraction of vocabulary directions with positive eigenvalue under $W_U W_{OV}^{(\ell,h)} W_E$; and an on-distribution *induction attribution* $\alpha^{(\ell,h)}$ = increase in loss on natural text when that head's output is mean-ablated, restricted to token positions whose correct next token appeared after a previous copy of the current token.
- **Control arm.** Identical runs on a data distribution with burstiness removed — each document's repeated tokens shuffled across documents so within-context repetition matches the corpus unigram rate. Prediction: no transition, no ICL jump.
- **Deciding number.** $\rho = \Delta t_{\text{ens}} / \bar{\Delta t}_{\text{head}}$, the width of the ensemble $\alpha$-transition divided by the mean width of individual heads' $\alpha$-transitions, and the seed-to-seed standard deviation of $t^\star$. If $\rho < 1.5$ and $\mathrm{sd}(t^\star)/t^\star < 0.1$, the transition is one coordinated event and $t^\star$ is a reproducible constant of the training setup — the method variant becomes tractable. If $\rho > 3$, the "phase transition" is an envelope of staggered per-head events and the single-transition framing is wrong.

## 9. Key References

- **[Foundational]** N. Elhage, N. Nanda, C. Olsson, et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, Anthropic, 2021.
- **[Foundational]** C. Olsson, N. Elhage, N. Nanda, et al. *In-Context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895
- **[SOTA]** A. Bietti, V. Cabannes, D. Bouchacourt, H. Jégou, L. Bottou. *Birth of a Transformer: A Memory Viewpoint.* NeurIPS, 2023. — arXiv:2306.00802
- **[SOTA]** G. Reddy. *The mechanistic basis of data dependence and abrupt learning in an in-context classification task.* ICLR, 2024. — arXiv:2312.03002
- **[SOTA]** A. K. Singh, T. Moskovitz, F. Hill, S. C. Y. Chan, A. M. Saxe. *What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation.* ICML, 2024. — arXiv:2404.07129
- **[SOTA]** E. Nichani, A. Damian, J. D. Lee. *How Transformers Learn Causal Structure with Gradient Descent.* ICML, 2024. — arXiv:2402.14735
- **[SOTA]** S. Chen, H. Sheen, T. Wang, Z. Yang. *Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers.* NeurIPS, 2024. — arXiv:2409.10559
- **[Empirical]** S. C. Y. Chan, A. Santoro, A. K. Lampinen, et al. *Data Distributional Properties Drive Emergent In-Context Learning in Transformers.* NeurIPS, 2022. — arXiv:2205.05055
- **[Empirical]** A. K. Singh, S. C. Y. Chan, T. Moskovitz, E. Grant, A. M. Saxe, F. Hill. *The Transient Nature of Emergent In-Context Learning in Transformers.* NeurIPS, 2023. — arXiv:2311.08360
- **[Empirical]** B. L. Edelman, E. Edelman, S. Goel, E. Malach, N. Tsilivis. *The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains.* NeurIPS, 2024. — arXiv:2402.11004
- **[Infrastructure]** S. Biderman, H. Schoelkopf, Q. Anthony, et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML, 2023. — arXiv:2304.01373

## 10. Worked Example

Take Pythia-1.4B, layer 5, head 3 — a head that reaches $\mathrm{PM} \approx 0.55$ on the standard repeated-random probe (uniform baseline for a 50-token subset: $0.02$).

**Step 1 — probe says yes.** Sequence of 50 random tokens repeated twice, $T=100$. At query position $i=73$ holding token $x$, the head puts $0.55$ of its attention mass on position $24$, whose predecessor is also $x$. This is the textbook induction pattern.

**Step 2 — on-distribution says much less.** Restrict to natural-text positions where the induction rule is actually predictive (current token has a prior occurrence, and the token after that occurrence is the correct next token). Mean-ablate the head. Loss on those positions rises by roughly $0.05$–$0.15$ nats — measurable, but a small fraction of the $\sim 0.8$ nat gap between $\mathcal{L}(50)$ and $\mathcal{L}(500)$ that defines the ICL score. Ablating this head does not undo the ICL jump.

**Step 3 — the obstruction.** Ablate all heads with $\mathrm{PM} > 0.3$ (typically 10–20 heads in a 24-layer model) and the ICL gap does collapse — but so does general loss, by more than a nat, because those heads also do other things. Ablate them one at a time and each contributes $< 0.2$ nats: the parts sum to far less than the whole, because the heads are redundant and back off for each other.

**The number that shows the problem.** Single-head ablation attributes $\approx 0.1$ nats; joint ablation removes $\approx 0.8$ nats of ICL gap but $> 1.0$ nats overall. Neither is a clean causal estimate of "the induction mechanism's contribution." $\mathrm{PM}$ ranked the head correctly and still gave a number that does not convert into a causal claim. That gap — a high-confidence order parameter that does not license a causal statement — is why the problem is only partially solved: the mechanism is understood in 2-layer toy models and the phenomenology is reproducible at scale, but the measurement that would connect them does not yet exist.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*