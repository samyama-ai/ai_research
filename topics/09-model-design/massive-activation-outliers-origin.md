---
id: 09-model-design/massive-activation-outliers-origin
title: "Why Residual Streams Develop Massive Activations"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Why Residual Streams Develop Massive Activations

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/massive-activation-outliers-origin` · **Status:** open

## 1. Problem Statement

Trained transformers place a handful of activations in their residual stream that are $10^2$–$10^4$ times larger than the median. In LLaMA-2-7B, two hidden dimensions at the BOS token and at certain delimiter tokens reach magnitude $\approx 2{,}600$ while the median absolute activation is $\approx 0.2$ (Sun et al., COLM 2024). They are not noise: zeroing them collapses the model. The problem is to explain why they appear, and to say whether they are **necessary** or merely the **cheapest available implementation** of a function the architecture makes awkward.

Three variants, of different difficulty:

- **Measurement.** Give a definition of "massive activation" that is invariant to the arbitrary choices the architecture leaves free — RMSNorm gain folding, residual-stream basis rotation, layer indexing — and that separates *outlier dimension* (a fixed channel, all tokens) from *outlier token* (a fixed position, all channels). Currently these are conflated under one name.
- **Method.** Find an architecture or optimizer change that removes massive activations at fixed loss and fixed token budget. The decision predicate: at $\ge 7$B params and $\ge 1$T tokens, does a modified model match baseline validation loss within $0.01$ nats while reducing per-tensor activation kurtosis by $\ge 10\times$?
- **Theory.** Prove or refute: for softmax attention with a normalization layer that divides by activation norm, any model achieving loss below some threshold on natural language must allocate near-constant, large-norm directions in the residual stream. No proof either way exists.

## 2. Formal Setting

Let a decoder-only transformer with $L$ layers, width $d$, and $H$ heads process a sequence $x_{1:T}$. Write the residual stream after layer $\ell$ as $h_i^{(\ell)} \in \mathbb{R}^d$, $i$ the token index. All quantities below are read from a forward pass in fp32 on a held-out corpus $\mathcal{D}$ of $N$ sequences.

**Massive activation (Sun et al. criterion, as measured).** Activation $h_{i,j}^{(\ell)}$ is *massive* if
$$|h_{i,j}^{(\ell)}| > 100 \quad\text{and}\quad |h_{i,j}^{(\ell)}| > 1000 \cdot \operatorname{median}_{i',j'} |h_{i',j'}^{(\ell)}|.$$
Both thresholds are conventions, not derived quantities — this is the measurement weakness.

**Scale-free surrogate.** Per-layer excess kurtosis of the flattened activation tensor,
$$\kappa^{(\ell)} = \frac{\mathbb{E}\big[(h^{(\ell)}-\mu)^4\big]}{\big(\mathbb{E}[(h^{(\ell)}-\mu)^2]\big)^2},$$
used as the outlier metric by He et al. (NeurIPS 2024). Invariant to global rescaling, *not* invariant to basis rotation.

**Channel vs. token decomposition.** Define the channel score $c_j^{(\ell)} = \max_i |h_{i,j}^{(\ell)}|$ and the token score $t_i^{(\ell)} = \|h_i^{(\ell)}\|_\infty$. An *outlier feature* is large $c_j$ for most $i$; an *extreme token* is large $t_i$ for one $i$ (usually $i=1$).

**Attention sink strength.** For head $(\ell,h)$ with attention matrix $A^{(\ell,h)}$,
$$s^{(\ell,h)} = \frac{1}{T-1}\sum_{i=2}^{T} A^{(\ell,h)}_{i,1},$$
the mean mass on the first token. Sink is conventionally declared at $s > 0.3$ (Gu et al., ICLR 2025).

**Causal test.** Ablation loss $\Delta\mathcal{L} = \mathcal{L}(\text{model with } h_{i,j}^{(\ell)}\!\leftarrow\!0) - \mathcal{L}(\text{model})$, against a control where an equal number of the *next*-largest activations are zeroed.

**Assumptions, and which fail.**
- *Basis privilege is meaningful.* Requires per-coordinate operations (Adam, RMSNorm gains, GLU gating) to break rotational symmetry — true in practice (Elhage et al., 2023), but it means $\kappa$ is not a property of the function computed, only of the parameterization. Violated as a measure of "the model needs this."
- *A fixed threshold transfers across models.* Fails: activation scale depends on final-layer norm gain, embedding scale, and whether QK-norm is used. Absolute thresholds are not comparable between LLaMA and, say, Gemma-style architectures with $\sqrt{d}$ embedding scaling.
- *Massive activations are a stable, converged property.* Partly violated: they emerge abruptly during training and their location can shift across restarts.

## 3. State of the Art

**Empirical characterization (established).** Sun et al., *Massive Activations in Large Language Models* (COLM 2024): massive activations exist in nearly all LLMs tested (LLaMA-2 7B/13B/70B, Mistral-7B, Phi-2, MPT, GPT-2, plus ViTs); they occupy a very small fixed set of dimensions; they emerge in early layers and disappear at the last; they behave as an approximately **input-independent bias**. Replacing them with their dataset mean leaves perplexity nearly unchanged; zeroing them destroys the model. Independently confirmed direction-wise by the outlier-feature literature (Dettmers et al., NeurIPS 2022; Kovaleva et al., ACL Findings 2021).

**Mechanistic account (claimed, partly ablated).** Two accounts, mutually compatible:
1. *Softmax must sum to one, so heads need a no-op target.* Bondarenko et al. (NeurIPS 2023) show clipped softmax and gated attention remove outliers in BERT/OPT-125M-scale models. Xiao et al. (ICLR 2024) show four sink tokens suffice for streaming. This is well ablated only below 1B params.
2. *Active–dormant heads with a mutual-reinforcement loop between sink and massive activation.* Guo et al. (2024/2025) give a Bigram-Backcopy toy model where the loop is derived and reproduced, plus matching LLM measurements. The toy derivation is rigorous; the extrapolation to 7B is claimed, not proven.

**Mitigation SOTA.** He et al. (NeurIPS 2024) reduce activation kurtosis by orders of magnitude with an "Outlier Protected" block (no entry-wise normalization) at up to ~1.2B params, with matched loss. Whether this holds at 7B+ and 1T tokens is unreported. Explicit learned attention bias (a per-head extra key/value, as in Sun et al.'s appendix and shipped in OpenAI's gpt-oss, 2025) removes massive activations in small pretraining runs — a **benchmark number, not a controlled scaling study**.

**Downstream-only results.** SmoothQuant (Xiao et al., ICML 2023) and LLM.int8() (Dettmers et al., 2022) do not explain outliers; they route around them. Their success is evidence outliers are *quantization-relevant*, not evidence about origin.

## 4. What Is Known

- **Magnitudes and locations.** LLaMA-2-7B: massive activations at dims 1415 and 2533, magnitude $\approx 2{,}600$ vs. median $\approx 0.2$; present from layer 2 to layer 30 of 32; confined to BOS, ".", and "\n" tokens (Sun et al., 7B–70B scale).
- **Causal necessity.** Zeroing the ~4 massive activations in LLaMA-2-7B raises WikiText perplexity from ~5.5 to $>10^3$; zeroing 3–4 *random large* activations changes it by $<0.1$ (7B scale).
- **Bias equivalence.** Substituting the empirical mean for the massive values restores near-baseline perplexity — they carry almost no token-specific information (7B–13B).
- **Phase transition with scale.** Systematic outlier features emerge sharply around 6.7B params in OPT-family models, at which point they appear in essentially all layers (Dettmers et al., 125M–175B sweep).
- **Emergence during training.** Sinks appear early — within the first few billion tokens — and their presence depends on data distribution, learning rate, and normalization; models trained without softmax normalization (sigmoid attention, no denominator) can reach comparable loss with no sink at ~1B params / tens of billions of tokens (Gu et al., ICLR 2025).
- **Not language-specific.** ViTs develop the same high-norm "register" tokens in low-information patches; adding explicit register tokens removes them and improves dense prediction and attention-map interpretability (Darcet et al., ICLR 2024, ViT-L/DINOv2 scale).
- **Optimizer contribution.** Adam's per-coordinate normalization privileges the standard basis; this is the standard explanation for why outliers are axis-aligned at all (Elhage et al., Transformer Circuits, 2023) — argued and illustrated, not proved.

## 5. What Is Not Known

- **Theoretically open.** No theorem states that softmax attention with entry-wise normalization *requires* large-norm constant directions to reach a given loss. The converse — that a sink-free architecture is equally expressive — is also unproven. The Guo et al. active–dormant analysis is a theorem about a two-state toy process, not about transformers on natural text.
- **Empirically open.** Whether register tokens, explicit attention bias, or the Outlier-Protected block preserve loss at $\ge 7$B params / $\ge 1$T tokens. Every clean ablation is at $\le 1.2$B. Also open: whether removing massive activations costs long-context or streaming quality, which is where the sink is hypothesized to do work.
- **Methodologically blocked.** "Massive activation" has no basis-invariant, scale-invariant definition. Kurtosis is rotation-dependent; the $>100$ threshold is architecture-dependent. Consequently a claim of the form "architecture X has no massive activations" is currently not falsifiable across families — the metric can be moved by folding RMSNorm gains into the next weight matrix without changing the function.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability plus a scale-mismatched control**. The residual stream's basis and scale are gauge freedoms: a rotation $R$ applied to $h$ and absorbed into surrounding weights leaves the function unchanged but can eliminate axis-aligned outliers entirely, while RMSNorm gain folding rescales them arbitrarily. So the observable ("a huge number in coordinate 2533") is a property of the chosen coordinates, not of the computation — yet the *causal* ablation result is real and coordinate-dependent, because Adam and RMSNorm make the coordinates real for the optimizer. Any candidate explanation must therefore separate "the network computes an approximately constant large-norm bias" (gauge-invariant, likely necessary) from "it stores it in two coordinates" (gauge-dependent, likely incidental). No current metric does this.

Compounding it: the phenomenon's sharpest form appears above ~6.7B params, while every controlled architecture ablation is run below 1.2B. That is a 6–100$\times$ compute gap between where the effect is strongest and where the experiments are affordable.

## 7. Current Research (as of 2026)

- **Sink-free architectures at scale.** Explicit learned attention bias (an extra per-head key/value with no value contribution) is now shipped in production models, including OpenAI's gpt-oss (2025), motivated by quantization and long-context stability. Public loss-matched ablations at frontier scale remain unpublished *(frontier — verify)*.
- **First-token analysis.** Barbero et al. (Google DeepMind / Oxford, 2025) argue sinks act as a brake on over-mixing, with sink strength growing with context length — evidence that sinks are functional for depth-robustness, not just a softmax artifact.
- **Mechanistic loops.** Guo, Pai, and collaborators (UC Berkeley / Princeton) on active–dormant heads and mutual reinforcement dynamics; Cancedda (Meta, ACL 2024) on spectral "dark signals" feeding sinks.
- **Normalization-free / outlier-protected blocks.** He and colleagues (ETH / DeepMind lineage) continuing signal-propagation-based designs; overlap with QK-norm and Dynamic Tanh–style replacements for LayerNorm *(frontier — verify whether any 7B+ run has been published)*.
- **Quantization-driven work.** The practical driver: outliers are the main obstacle to W4A4 inference. Rotation-based methods (QuaRot/SpinQuant lineage, 2024–2025) exploit exactly the gauge freedom above, which is indirect evidence that the coordinate-alignment is incidental.

## 8. Concrete Next Experiment

**Question decided:** does removing massive activations cost loss at a scale where they are known to be strong?

- **Scale.** Four 7B-parameter decoder-only models, identical data order, 300B tokens each (~$1.5\times10^{22}$ FLOPs total, ~$4\times$ 7B/300B runs).
- **Arms.**
  1. *Control:* standard pre-norm RMSNorm + softmax attention, BOS always prepended.
  2. *Explicit bias:* per-head learned key with zero value appended to every attention softmax.
  3. *Registers:* 4 learned prefix tokens, gradient-trained, excluded from the loss.
  4. *Outlier-protected:* He et al. block (normalization removed from the entry-wise path), otherwise identical.
- **Measurements.** Validation loss on held-out C4 and on a 32k-token long-context set; $\max_{i,j,\ell}|h^{(\ell)}_{i,j}|$; per-layer $\kappa^{(\ell)}$; sink strength $s^{(\ell,h)}$; W8A8 and W4A8 per-tensor PTQ perplexity.
- **The deciding number.** Validation-loss gap to the control arm. If any arm reaches $\Delta\mathcal{L} \le 0.01$ nats while cutting $\max|h|$ by $\ge 100\times$, massive activations are an artifact of the softmax/normalization design and not a requirement — and the theory variant collapses to a statement about optimizer gauge. If every arm pays $\ge 0.03$ nats, that is the first scale-appropriate evidence that the bias is functionally necessary.
- **Why it is small enough to run.** No new data, no new tokenizer, no hyperparameter search beyond a shared LR sweep at 1B; the only novel cost is the four matched runs.

## 9. Key References

- **[Foundational]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM, 2024. — arXiv:2402.17762
- **[Foundational]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Foundational]** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS, 2022. — arXiv:2208.07339
- **[SOTA]** Yehonathan Refael? — omitted. Yana Bondarenko, Markus Nagel, Tijmen Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023. — arXiv:2306.12929
- **[SOTA]** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR, 2025. — arXiv:2410.10781
- **[SOTA]** Tianyu Guo, Druv Pai, Yu Bai, Jiantao Jiao, Michael I. Jordan, Song Mei. *Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs.* 2024. — arXiv:2410.13835
- **[SOTA]** Bobby He, Lorenzo Noci, Daniele Paliotta, Imanol Schlag, Thomas Hofmann. *Understanding and Minimising Outlier Features in Transformer Training.* NeurIPS, 2024. — arXiv:2405.19279
- **[SOTA]** Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. *Vision Transformers Need Registers.* ICLR, 2024. — arXiv:2309.16588
- **[Related]** Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu. *Why do LLMs attend to the first token?* 2025. — arXiv:2504.02732
- **[Related]** Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, Anna Rumshisky. *BERT Busters: Outlier Dimensions that Disrupt Transformers.* Findings of ACL, 2021. — arXiv:2105.06990
- **[Related]** William Timkey, Marten van Schijndel. *All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality.* EMNLP, 2021. — arXiv:2109.04404
- **[Related]** Nicola Cancedda. *Spectral Filters, Dark Signals, and Attention Sinks.* ACL, 2024. — arXiv:2402.09221
- **[Related]** Nelson Elhage et al. *Privileged Bases in the Transformer Residual Stream.* Transformer Circuits Thread, Anthropic, 2023.
- **[Related]** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML, 2023. — arXiv:2211.10438

## 10. Worked Example

Take LLaMA-2-7B, layer 10, a 512-token WikiText sequence.

1. Measure $|h^{(10)}_{1,2533}| \approx 2{,}600$; median $|h^{(10)}| \approx 0.2$. Ratio $1.3\times10^4$ — massive by the Sun criterion.
2. Zero it. Perplexity goes from $5.5$ to $>10^3$. Control: zero the next three largest non-massive activations — perplexity moves by $<0.1$. Conclusion so far: this coordinate is causally load-bearing.
3. Replace it with its corpus mean instead of zero. Perplexity returns to $\approx 5.5$. So it carries $\approx 0$ bits about the current input: it is a bias.
4. **Now make the obstruction visible.** Pick a random orthogonal $R \in \mathbb{R}^{4096\times4096}$. Rewrite the model so the residual stream carries $\tilde h = R h$: absorb $R^\top$ into every read matrix ($W_Q, W_K, W_V, W_{\text{in}}$, unembedding) and $R$ into every write matrix ($W_O, W_{\text{out}}$, embedding). If normalization is RMSNorm *without* per-channel gain, the function is exactly unchanged — same logits, same perplexity.
5. Re-measure. The mass of the bias vector, previously concentrated in 2 of 4096 coordinates at magnitude 2600, is now spread over all 4096 at typical magnitude $2600/\sqrt{4096} \approx 41$. The Sun criterion ($>100$) no longer fires. Per-layer kurtosis $\kappa^{(10)}$ drops toward 3. The model has "no massive activations" and computes exactly the same function.

What survives the rotation is the gauge-invariant fact: $\|h_1^{(10)}\| \approx 2{,}600$ against a typical token norm of $\approx 13$ — a 200:1 norm ratio at one token position. What does *not* survive is coordinate alignment. Any paper reporting "our architecture removes outliers" by reporting $\max|h|$ or kurtosis is, in principle, reporting a quantity a free rotation can set. Until the metric is fixed at the level of *per-token residual norm* and *sink attention mass* rather than per-coordinate magnitude, method claims and theory claims in this area are not comparable — which is why the problem stays open despite a large and growing empirical literature.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*