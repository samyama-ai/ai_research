---
id: 09-model-design/attention-sink-architectural-cause
title: "Architectural Cause of the Softmax Attention Sink"
topic: 09-model-design
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Architectural Cause of the Softmax Attention Sink

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/attention-sink-architectural-cause` · **Status:** partially-solved

## 1. Problem Statement

In nearly every trained decoder-only transformer, most attention heads in most layers place a large fraction of their probability mass on the first token of the sequence — usually `<bos>`, or whatever token happens to occupy position 0 — regardless of that token's semantics. The same models carry *massive activations*: a handful of residual-stream coordinates at those positions with magnitudes 100–1000× the median. The two phenomena are correlated and jointly called the **attention sink**.

Three separable variants:

- **Measurement.** Give a definition of "sink" that is not an arbitrary threshold on head-averaged attention mass, and that distinguishes a *load-bearing* sink from a numerical artifact. Currently open.
- **Method.** Produce an architecture that trains to equal or better loss, quantizes to 8-bit weights and activations without outlier-specific tricks, and exhibits no sink. Partially achieved at ≤1B parameters; unreplicated at frontier scale.
- **Theory.** Prove which architectural constraint *causes* the sink. The leading candidate is the normalization of softmax: $\sum_j a_{ij} = 1$ forces every head to emit a convex combination of value vectors, with no way to emit nothing. Competing causes: the residual stream's need for a high-norm reference direction under LayerNorm/RMSNorm, and depth-dependent over-mixing.

Solving it means: a stated causal claim, an intervention that removes the sink by that mechanism, and a demonstration that loss, long-context behavior, and quantizability are unharmed at ≥7B parameters.

## 2. Formal Setting

Let $X \in \mathbb{R}^{T \times d}$ be residual-stream states over $T$ tokens. For head $h$ in layer $\ell$ with causal mask,

$$A^{(\ell,h)}_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_h})}{\sum_{j' \le i} \exp(q_i^\top k_{j'} / \sqrt{d_h})}, \qquad \sum_{j\le i} A^{(\ell,h)}_{ij} = 1 .$$

**Sink rate.** Measured as the fraction of query positions with mass on position 0 above a threshold, averaged over a held-out corpus:

$$\mathrm{Sink}_\varepsilon^{(\ell,h)} = \frac{1}{|\mathcal{D}|}\sum_{x \in \mathcal{D}} \frac{1}{T-1}\sum_{i=1}^{T-1} \mathbb{1}\!\left[A^{(\ell,h)}_{i0}(x) > \varepsilon\right],\quad \varepsilon = 0.3 \text{ by convention (Gu et al., 2025)}.$$

$\varepsilon$ is a free parameter with no principled setting — this is the methodological hole.

**Massive activation.** For residual state $x_t \in \mathbb{R}^d$, coordinate $c$ is massive if $|x_{t,c}| > \kappa \cdot \mathrm{median}_{t',c'}|x_{t',c'}|$; Sun et al. (2024) use $\kappa$ implicitly at $\sim$1000×.

**Load-bearing test (functional, not descriptive).** The sink's causal role is measured by ablation: replace $A_{i0}$ mass with renormalized mass over $j \ge 1$ and measure $\Delta$ perplexity. The *value-weighted* contribution matters, not the attention weight:

$$\text{contribution}_i = \Big\| \sum_{j\le i} A_{ij} W_V x_j \Big\|, \qquad \text{sink share} = \frac{A_{i0}\|W_V x_0\|}{\text{contribution}_i}.$$

**Quantization proxy.** Excess kurtosis of per-channel activations, $\mathrm{kurt}(x_{\cdot,c}) - 3$, predicts INT8 per-tensor quantization error; sinks inflate it.

**Assumptions known to be violated.** (i) That position 0 is special *only* because of position — false: sinks also land on delimiters, newlines, and low-semantic tokens (Sun et al., 2024). (ii) That the sink is one phenomenon — attention concentration, massive activations, and value-state norm collapse are measured separately and dissociate in some heads. (iii) That $\varepsilon = 0.3$ is scale-invariant — sink mass grows with depth and context length, so a fixed threshold changes what it counts across model sizes.

## 3. State of the Art

**Established (ablated, reproduced).**
- *StreamingLLM* (Xiao et al., ICLR 2024, arXiv:2309.17453): keeping 4 initial tokens' KV while sliding the rest stabilizes generation to 4M tokens on Llama-2-7B/13B, MPT, Falcon, Pythia; evicting them makes perplexity diverge. Training a 160M model from scratch with one dedicated learnable sink token reduces the requirement from 4 tokens to 1 — this is a *causal* architectural result, and it replicated.
- *Quantizable Transformers* (Bondarenko et al., NeurIPS 2023, arXiv:2306.12929): clipped softmax and gated attention give heads an explicit "attend to nothing" path; on BERT-base and OPT-125M/350M this removes outliers and improves INT8 post-training quantization while matching FP16 perplexity. Ablated at ≤350M.
- *Massive Activations* (Sun et al., COLM 2024, arXiv:2402.17762): massive activations act as fixed, input-independent biases; zeroing them collapses the model, and replacing attention with explicit learned key/value bias terms removes them without loss penalty. Reproduced across LLaMA-2, Mistral, ViTs.
- *Vision Transformers Need Registers* (Darcet et al., ICLR 2024, arXiv:2309.16588): adding dedicated register tokens removes high-norm artifact tokens in DINOv2/CLIP and improves dense prediction — the closest thing to a clean architectural fix, in vision.

**Claimed but unablated at scale.** That the sink is *necessary* for long-context stability rather than merely load-bearing after the fact. That normalization alone is the cause: Gu et al. (ICLR 2025) show sigmoid attention without normalization trains sink-free at 1B, but no 7B+ replication with matched data, and no LongBench-class evaluation.

**Benchmark-number-only.** Sink-aware KV-cache compression methods report LongBench/RULER deltas without isolating whether the gain comes from sink handling or from the eviction policy.

## 4. What Is Known

- **Magnitude.** LLaMA-2-7B: massive activations reach $\sim$2500 in absolute value against a median activation magnitude near 0.2, confined to about 4 feature dimensions and to the first token plus delimiters; they appear abruptly at layer 2 and persist to the penultimate layer (Sun et al., COLM 2024).
- **Prevalence.** In LLaMA-3.1-405B, roughly 80% of heads place dominant mass on `<bos>`; the fraction rises with model size and with training context length (Barbero et al., 2025, arXiv:2504.02732).
- **Emergence.** Sinks form during optimization, not at initialization, and appear within the first $\sim$10<sup>9</sup> tokens of pretraining; they emerge under Adam-like optimization and are sensitive to learning rate, weight decay, and context length (Gu et al., ICLR 2025, arXiv:2410.10781).
- **Removability at small scale.** Sinks vanish under attention variants that drop the sum-to-one constraint (sigmoid attention without normalization, softmax-off-by-one/clipped softmax) at 60M–1B, with comparable validation loss.
- **Failure on eviction.** Dropping the first 4 tokens from the KV cache in Llama-2-7B raises perplexity by orders of magnitude; the same window with those tokens retained is stable (Xiao et al., ICLR 2024).
- **Mechanistic account.** Extreme-token behavior in small models is explained by *active–dormant* heads: a head is dormant on most inputs, dumping mass on the sink, and activates on a specific trigger (Guo et al., 2024, arXiv:2410.13835). Demonstrated in a Bigram-Backcopy toy task and 1–7B LLMs.

## 5. What Is Not Known

- **Theoretically open.** No theorem states a necessary-and-sufficient architectural condition for sink formation. The strongest formal claim — that sinks slow over-mixing / rank collapse with depth (Barbero et al., 2025) — is a sufficiency argument about signal propagation, not a proof that softmax normalization is necessary for the observed behavior.
- **Empirically open.** Whether a sink-free architecture matches a softmax baseline at ≥7B on matched tokens across loss, RULER-style retrieval at 128k, and INT8/FP8 quantization. The experiment is runnable today; the cost ($\sim$10<sup>4</sup> GPU-hours per arm) is the only barrier.
- **Methodologically blocked.** "Is this head a sink?" has no threshold-free definition. Attention mass, value norm, and residual-stream norm give different head rankings on the same model, and no published work reports agreement rates among them.

## 6. Why It Is Hard

**Non-identifiability.** At least three hypotheses — (a) softmax must sum to 1, so heads need a no-op dump; (b) LayerNorm/RMSNorm requires a high-norm anchor to stabilize scale; (c) deep stacks need attenuated mixing to avoid rank collapse — predict *the same observable*: high mass on a low-value-norm early token. Interventions confound them: adding a register token changes (a), (b), and (c) at once.

**Confounded measurement.** Attention weight is not contribution. A head with $A_{i0}=0.9$ onto a value vector of norm 0.05 moves the residual stream less than a head with $A_{ij}=0.05$ onto norm 1.0. Most sink statistics in the literature are computed on $A$ alone.

**Compute.** The phenomenon's key properties — prevalence, depth-dependence — *grow* with scale, so small-model ablations are structurally unable to settle the question, and the settling experiment requires full pretraining runs, not fine-tunes.

## 7. Current Research (as of 2026)

- Attention variants that relax normalization: sigmoid and off-by-one softmax, gated attention. Groups: Qualcomm AI Research (Bondarenko lineage), NUS/Sea AI Lab (Gu et al.).
- Register/sink tokens as first-class architecture, now standard in some vision backbones and appearing in language pretraining recipes *(frontier — verify which frontier models ship a dedicated sink token; several open-weight releases document one)*.
- Mechanistic interpretability of active–dormant heads and "catch–tag–release" accounts of what the sink stores (Princeton, UW, Meta FAIR contributors).
- Quantization-driven work: FP8/INT8 and KV-cache compression teams treat sinks as the dominant outlier source; outlier-aware formats are the pragmatic workaround rather than a fix.
- Theory of signal propagation and over-mixing in deep attention stacks (Google DeepMind, Oxford).

## 8. Concrete Next Experiment

**Question.** Is softmax normalization the cause, or is a high-norm anchor token sufficient?

**Scale.** Four 1.4B-parameter decoder-only models, identical data (300B tokens, same order), identical optimizer, 8k context, one seed each plus one repeat seed on arms A and C.

**Arms.**
- **A (control):** standard softmax attention, `<bos>` prepended.
- **B:** softmax, plus 4 dedicated learnable register tokens at positions 0–3, never attended by the loss.
- **C:** unnormalized sigmoid attention, $a_{ij} = \sigma(q_i^\top k_j/\sqrt{d_h} + b)$, no denominator.
- **D:** softmax-off-by-one, $\exp(\cdot)/(1+\sum\exp(\cdot))$.

**Measurements.** Validation loss; $\mathrm{Sink}_{0.3}$ averaged over heads; max per-channel excess kurtosis of residual activations; RULER needle-retrieval accuracy at 8k and at 32k after context extension; INT8 per-tensor W8A8 post-training quantization perplexity delta.

**Deciding number.** The INT8 W8A8 perplexity delta $\Delta_{\text{PTQ}} = \mathrm{ppl}_{\text{int8}} - \mathrm{ppl}_{\text{fp16}}$, subject to $|\Delta\text{loss}| < 0.01$ nats versus arm A and RULER within 2 points. If C and D both give $\Delta_{\text{PTQ}} < 0.1$ while A gives $\Delta_{\text{PTQ}} > 1.0$ and B stays high, normalization is the cause. If B also drops to $<0.1$, the anchor-token hypothesis is sufficient and normalization is not the operative constraint.

## 9. Key References

- **[Foundational]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[Foundational]** Bondarenko, Nagel, Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS 2023. — arXiv:2306.12929
- **[SOTA]** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models.* COLM 2024. — arXiv:2402.17762
- **[SOTA]** Gu, Pang, Du, Liu, Wang, Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR 2025. — arXiv:2410.10781
- **[SOTA]** Barbero, Arroyo, Gu, Perivolaropoulos, Bronstein, Veličković, et al. *Why Do LLMs Attend to the First Token?* 2025. — arXiv:2504.02732
- **[Mechanistic]** Guo, Huang, Zhang, Chen, Lee, et al. *Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs.* 2024. — arXiv:2410.13835
- **[Vision analogue]** Darcet, Oquab, Mairal, Bojanowski. *Vision Transformers Need Registers.* ICLR 2024. — arXiv:2309.16588
- **[Related]** Cancedda. *Spectral Filters, Dark Signals, and Attention Sinks.* ACL 2024.

## 10. Worked Example

Take one head in a mid-layer of Llama-2-7B at query position $i = 512$. A representative sink head gives $A_{i0} \approx 0.85$, with the remaining 0.15 spread over 512 positions.

Naively, 85% of the head's output comes from token 0. Compute the value-weighted share instead. Sink positions carry a value-state norm far below typical: take $\|W_V x_0\| \approx 0.05$ against $\|W_V x_j\| \approx 1.0$ for ordinary tokens. Then

$$\text{sink share} = \frac{0.85 \times 0.05}{0.85 \times 0.05 + 0.15 \times 1.0} = \frac{0.0425}{0.1925} \approx 0.22 .$$

So 85% of the *attention* is 22% of the *contribution*. The head is mostly, but not entirely, doing nothing.

Now the obstruction. Two hypotheses fit this identically:

1. **No-op valve.** The head has nothing to retrieve; normalization forbids emitting zero, so it dumps mass on a deliberately low-norm token. Prediction: replacing softmax with an unnormalized nonlinearity removes the pattern.
2. **Anchor.** Token 0's residual state carries a 2500-magnitude activation in $\sim$4 coordinates that acts as a fixed bias; the head is reading that bias, and its small $W_V$ projection is exactly the calibration of bias strength. Prediction: removing normalization changes nothing, because the model will re-create the bias through an explicit path.

Both predict $A_{i0}=0.85$, both predict low value norm, and both predict that evicting token 0 breaks the model — because under (1) the renormalized 0.15 mass gets inflated 6.7× and blows up downstream activations, and under (2) the bias disappears. Measuring $A$, $\|W_V x_0\|$, or eviction perplexity distinguishes neither. Only a from-scratch training intervention that changes one hypothesis' premise while holding the other fixed — arms C versus B in §8 — separates them, and that costs a pretraining run, not a probe.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*