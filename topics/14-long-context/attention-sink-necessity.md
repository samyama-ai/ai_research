---
id: 14-long-context/attention-sink-necessity
title: "Attention Sink Necessity and Removability"
topic: 14-long-context
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Sink Necessity and Removability

> **Topic:** Long Context · **ID:** `14-long-context/attention-sink-necessity` · **Status:** partially-solved

## 1. Problem Statement

Trained softmax transformers concentrate a large share of attention mass on a few semantically empty positions — usually the first token (BOS), sometimes delimiters or newlines. This is the **attention sink**. The associated residual-stream states carry **massive activations**: coordinates 100–1000× the median magnitude.

The problem has three variants that are routinely conflated.

- **Measurement.** Define a sink so that "this model has one" is a decidable predicate, not an eyeball judgment on a heatmap. Sink share, sink location, and per-head vs per-layer aggregation all give different answers on the same model.
- **Method.** Can a model be trained to equal quality (loss, downstream accuracy, length generalization) *without* a sink? Candidate interventions: softmax-off-by-one / clipped softmax, learned per-head key–value bias, register tokens, sigmoid attention without normalization, QK-norm. Success predicate: match the baseline's validation loss within noise **and** its length-extrapolation curve, while the max residual-stream $\ell_\infty$ norm stays within one order of magnitude of the median.
- **Theory.** Prove whether a sink is *necessary* — i.e. whether any softmax-normalized attention model at competitive loss must place $\Omega(1)$ mass on a low-information position — or exhibit an architecture class with a proof that it need not.

Practical stakes: sinks break INT8/FP8 per-tensor quantization, force KV-cache eviction schemes to pin the first tokens, and are the mechanism behind streaming inference working at all.

## 2. Formal Setting

Let a decoder-only model have $L$ layers, $H$ heads, hidden width $d$. For input $x_{1:n}$, head $(\ell,h)$ produces a row-stochastic causal matrix $A^{(\ell,h)} \in \mathbb{R}^{n\times n}$,
$$A^{(\ell,h)}_{ij} = \frac{\exp(q_i^\top k_j/\sqrt{d_h})}{\sum_{j'\le i}\exp(q_i^\top k_{j'}/\sqrt{d_h})},\quad j\le i.$$

**Sink share** at position $p$, measured by averaging over query positions after a burn-in $m$ (to remove the trivial $A_{11}=1$):
$$\sigma^{(\ell,h)}_p = \frac{1}{n-m}\sum_{i=m+1}^{n} A^{(\ell,h)}_{ip}.$$

**Sink metric** (Gu et al., ICLR 2025): fraction of heads with $\sigma^{(\ell,h)}_1 > \epsilon$, conventionally $\epsilon = 0.3$, averaged over a held-out corpus:
$$\mathrm{SM}_\epsilon = \frac{1}{LH}\sum_{\ell,h}\mathbb{1}[\sigma^{(\ell,h)}_1>\epsilon].$$

**Massive activation ratio** at layer $\ell$: $\rho^{(\ell)} = \|h^{(\ell)}\|_\infty / \mathrm{median}_{i,c}|h^{(\ell)}_{ic}|$ over token/channel pairs. Empirically $\rho \gtrsim 10^3$ in models with sinks.

**Length generalization** is measured as $\Delta(T) = \mathrm{PPL}(T) - \mathrm{PPL}(T_{\text{train}})$ on a fixed long-document corpus (PG19, Books3), with the KV cache either full or windowed.

Assumptions and where they break:

- *Sink = position 1.* Violated: GPT-2 and Mistral place sinks on delimiters and low-frequency tokens; some models have several sinks. $\sigma_p$ must be scanned over $p$, not evaluated at $p=1$.
- *Sink share is layer-uniform.* Violated: layers 0–1 typically show no sink; the effect appears at layer 2 and persists.
- *The measurement is prompt-independent.* Violated: sink share depends on prompt length and whether BOS was prepended, which is itself inconsistent across tokenizers and serving stacks.
- *Attention weights are the causal quantity.* Weakly supported: attention mass is not attribution. The value vector at the sink is near-null in some models, so mass and effect are dissociable.

## 3. State of the Art

**Established (replicated, ablated).**

- Sinks are *load-bearing at inference under windowed attention*. StreamingLLM (Xiao et al., ICLR 2024) shows that retaining 4 initial tokens plus a rolling window keeps perplexity flat over millions of tokens; dropping them makes perplexity diverge by orders of magnitude. Independently reproduced across Llama, Falcon, MPT, Pythia.
- Sinks co-occur with massive activations, and those activations function as an implicit per-head attention bias (Sun et al., COLM 2024).
- Sinks can be *prevented at training time* in small models by architectural change: clipped softmax and gated attention (Bondarenko et al., NeurIPS 2023) suppress outliers with no loss penalty at BERT/OPT-125M/ViT scale.

**Claimed but not fully ablated.**

- "Sinks are unnecessary at LLM scale." Gu et al. (ICLR 2025) report that replacing softmax with a non-normalized sigmoid attention removes the sink in models up to 1B params, and that sink emergence depends on optimization (LR, weight decay) and data, not just architecture. This is a strong result at 1B; it has not been reproduced at $\geq 7$B or at multi-trillion-token budgets, and its effect on 128K-context behaviour is untested.
- "Registers substitute for sinks." Vision Transformers Need Registers (Darcet et al., ICLR 2024) is solid for ViTs; the LLM analogue is asserted more than demonstrated.

**Benchmark-number-only.** Most "sink-free" claims are reported as validation perplexity at the training length plus a quantization number. Long-context retrieval (needle-in-a-haystack, RULER, LongBench) is usually absent, which is exactly where a sink would matter.

## 4. What Is Known

- Llama-2-7B: two residual channels (dims 2533, 1415) reach magnitude $\approx 2\times10^3$ against a median $\approx 0.2$, i.e. $\rho \approx 10^4$, and are confined to BOS and a few delimiter tokens (Sun et al., COLM 2024). Zeroing them raises perplexity by orders of magnitude; zeroing an equal number of random large activations does not.
- StreamingLLM, Llama-2-7B, PG19: window-only attention with cache size 1024 gives perplexity in the $10^3$ range; adding 4 sink tokens returns it to roughly the dense value ($\approx 6$–$7$ nats-equivalent), stable to 4M tokens.
- A model trained with a *learned* zero-value attention bias (an explicit "attend to nothing" slot) reaches baseline perplexity at GPT-2 scale with no massive activations (Sun et al.).
- Sink strength grows with pretraining context length; Barbero et al. (2025) show for Llama-family models that longer-context pretraining produces stronger BOS attention, consistent with sinks acting as a brake on over-mixing (rank collapse) in deep stacks.
- Quantization: activation outliers of this magnitude make per-tensor W8A8 fail; outlier-suppressing attention variants recover near-FP accuracy at BERT/OPT-125M scale (Bondarenko et al.).
- Sinks are not a Llama artifact: they appear in GPT-2, OPT, Pythia, Falcon, Mistral, and in ViTs (as register-absorbing patches).

## 5. What Is Not Known

- **Theoretically open.** No theorem states that a softmax attention model at loss $\le \mathcal{L}^\*+\delta$ must have $\max_p \sigma_p \ge c$. The rank-collapse/over-mixing argument (Barbero et al.) explains why a sink is *useful* but does not lower-bound its mass. Conversely there is no proof that a sink-free architecture attains the same loss.
- **Empirically open.** Whether sink-free training (sigmoid attention, key bias, registers) holds at 7B–70B and $\ge 1$T tokens, and whether such models keep 128K-context retrieval accuracy. The experiment is a standard pretraining run; nobody has published the matched pair.
- **Empirically open.** Whether sink removal actually delivers the quantization win at scale — the outlier results are at $\le 1$B params.
- **Methodologically blocked.** "Does the model have a sink?" has no agreed operational definition. $\epsilon=0.3$ on first-token mass, $\ell_\infty$ activation ratio, and value-weighted attention mass rank models differently. There is no standard sink benchmark, so cross-paper comparison is unsound.

## 6. Why It Is Hard

The central obstruction is **confounded measurement compounded by non-identifiability**. Attention mass, residual activation magnitude, and functional necessity are three different quantities that the literature treats as one. A head can put 0.9 mass on BOS whose value vector is near-zero — high sink share, zero function. Ablating the sink token changes the softmax denominator for every query simultaneously, so the ablation cannot separate "the sink carried information" from "the normalizer was perturbed"; the standard fix (replace with a mean token) changes the denominator too.

Second obstruction: **cost of the decisive control**. The question is about pretraining dynamics, so the only clean arm is a matched pretraining pair at a scale where long-context behaviour is real — 7B params, ~1T tokens, ~$10^{22}$ FLOPs per arm. That is outside single-lab-ablation budgets, and the smaller runs that fit budget are exactly the regime where sinks are weakest and the negative result is uninformative.

## 7. Current Research (as of 2026)

- **Mechanistic accounts.** Active-dormant head models (Guo et al., 2024) derive sink emergence from a mutual-reinforcement dynamic in a simplified BigramBacon-style setting; extending this to real stacks is ongoing *(frontier — verify)*.
- **Sink-free architectures in production models.** QK-norm, learned attention sinks (per-head logit added to the softmax denominator), and register tokens now appear in several open-weight releases; the trend is toward making the sink *explicit* rather than removing it *(frontier — verify)*.
- **Head typology for KV compression.** DuoAttention (Xiao et al., ICLR 2025) splits retrieval vs streaming heads and keeps a full cache only for the former — a sink-aware systems response rather than a removal.
- **Quantization groups** (Qualcomm AI Research and others) continue outlier-suppression work; FP8/FP4 training pressure makes sink removal economically motivated.
- **Vision/multimodal** register work continues to be the cleanest transfer case.

## 8. Concrete Next Experiment

**Question.** Is a sink necessary at competitive scale, or is it an optimization artifact of softmax normalization?

**Scale.** Two 1.4B-parameter decoder-only models, identical data order, 300B tokens, 8K training context, then 32K continued pretraining on 10B tokens. Cost roughly $2\times10^{21}$ FLOPs per arm — one 64×H100 week each.

**Arms.**
- *Control:* standard softmax attention, BOS prepended.
- *Treatment:* per-head learnable key/value bias slot (an explicit null-attention target with zero value), everything else identical.

**Measure.** (i) $\mathrm{SM}_{0.3}$ and $\rho^{(\ell)}$; (ii) validation loss; (iii) **the deciding number: RULER accuracy at 32K averaged over the 13 subtasks**, treatment minus control. (iv) W8A8 per-tensor PTQ perplexity delta as a secondary.

**Decision rule.** If treatment has $\mathrm{SM}_{0.3}<0.02$, $\rho<50$, validation loss within 0.005 nats, **and RULER-32K within 1.0 accuracy point of control**, sinks are removable at this scale and the burden shifts to those claiming necessity. If RULER drops $>3$ points while loss matches, the sink is doing long-context work that perplexity does not see — the most informative outcome, and the one current papers cannot rule out because they do not run RULER.

## 9. Key References

- **[Foundational]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[Foundational]** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS 2023. — arXiv:2306.12929
- **[SOTA]** Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM 2024. — arXiv:2402.17762
- **[SOTA]** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR 2025. — arXiv:2410.10781
- **[Theory]** Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu. *Why do LLMs attend to the first token?* 2025. — arXiv:2504.02732
- **[Mechanism]** Tianyu Guo et al. *Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs.* 2024. — arXiv:2410.13835
- **[Transfer]** Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. *Vision Transformers Need Registers.* ICLR 2024. — arXiv:2309.16588
- **[Related]** Nicola Cancedda. *Spectral Filters, Dark Signals, and Attention Sinks.* ACL 2024.
- **[Systems]** Guangxuan Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* ICLR 2025. — arXiv:2410.10819
- **[Related]** Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, Sinong Wang. *LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models.* NAACL 2024. — arXiv:2308.16137
- **[Informal origin]** Evan Miller. *Attention Is Off By One.* Blog post, 2023 (softmax$_1$ proposal; no peer-reviewed venue).

## 10. Worked Example

Take Llama-2-7B, layer 15, one head, a 2048-token PG19 passage, burn-in $m=64$.

Measured: $\sigma_1 \approx 0.85$. Naive reading — 85% of this head's computation is spent on BOS.

Now check the value side. The BOS value vector's contribution to the head output is $\sigma_1 \cdot v_1$, and in this head $\|v_1\|_2$ is roughly $0.02\times$ the mean $\|v_j\|_2$ over content tokens. So the sink's share of output *norm* is
$$\frac{0.85 \times 0.02}{0.85\times 0.02 + 0.15\times 1.0} \approx 0.10,$$
about 10%, not 85%. Measured as attention mass the sink dominates; measured as output contribution it is a minority term. Two published sink metrics, opposite conclusions, same head.

Now the ablation. Delete BOS and rerun: windowed-cache perplexity moves from roughly 6–7 to $>10^3$ (StreamingLLM's reported regime). Tempting inference: the sink carried essential information. But deleting BOS removes $\exp(q_i^\top k_1)$ from *every* denominator, rescaling the remaining 15% of mass upward by $1/0.15 \approx 6.7\times$. The output of the head changes by a factor of ~6.7 on the content terms — an effect of the same order as the collapse, produced entirely by renormalization, with no information transfer at all.

That is the obstruction in one line: **the standard necessity test cannot separate "the sink stored something" from "the sink was absorbing $0.85$ of the normalizer."** The explicit-bias arm in Section 8 is the only design that dissociates them, because it supplies the normalizer mass without a token to store anything in.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*