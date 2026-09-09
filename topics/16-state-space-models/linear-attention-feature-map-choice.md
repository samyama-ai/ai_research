---
id: 16-state-space-models/linear-attention-feature-map-choice
title: "Optimal Kernel Feature Maps for Linear Attention"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Kernel Feature Maps for Linear Attention

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/linear-attention-feature-map-choice` · **Status:** open

## 1. Problem Statement

Linear attention replaces $\mathrm{softmax}(q^\top k)$ with $\phi(q)^\top \phi(k)$ for a feature map $\phi:\mathbb{R}^{d}\to\mathbb{R}^{d'}_{\ge 0}$, which makes the layer a linear RNN with a matrix state of size $d' \times d_v$. The question: **which $\phi$ should you choose, and what does the choice buy you at fixed state size?**

Three variants that are usually conflated:

- **Measurement.** Given a fixed recurrent state budget $S = d' d_v$ bytes, which $\phi$ maximizes downstream quality? Solving this means a ranking of feature maps that is stable across scale, data, and the rest of the architecture (gating, decay, normalization).
- **Method.** Construct a $\phi$ that dominates all known maps at equal $S$ and equal throughput. Solving this means a map that wins on associative recall *and* language-model perplexity without a per-task tuned hyperparameter.
- **Theory.** Characterize the approximation/state tradeoff: for a target error $\varepsilon$ against softmax attention over inputs with bounded entries, what is the minimum $d'$, and is any $\phi$ optimal in that sense? Solving this means a matching upper and lower bound on $d'(\varepsilon, n, d)$.

The theory variant is close to resolved *negatively* for exact softmax mimicry. The measurement and method variants are open.

## 2. Formal Setting

Let $X \in \mathbb{R}^{n\times d_{\text{model}}}$ be a sequence of $n$ tokens; per head, $Q,K \in \mathbb{R}^{n\times d}$, $V\in\mathbb{R}^{n\times d_v}$. Causal linear attention:

$$o_t = \frac{\phi(q_t)^\top \sum_{s\le t}\phi(k_s) v_s^\top}{\phi(q_t)^\top \sum_{s\le t}\phi(k_s) + \epsilon}, \qquad S_t = S_{t-1} + \phi(k_t)v_t^\top \in \mathbb{R}^{d'\times d_v}.$$

Quantities, **as measured**:

- **State size** $S = d' \cdot d_v \cdot H$ scalars per layer, times bytes per element. Measure it as peak decode-time KV-state bytes at batch 1, not parameter count. This is the budget that must be held fixed.
- **Kernel error** $\varepsilon_{\text{ker}} = \mathbb{E}_{q,k\sim \mathcal{D}}\big[(\phi(q)^\top\phi(k) - e^{q^\top k/\sqrt d})^2\big]^{1/2}$, estimated on $q,k$ sampled from a *trained* softmax model's activations, not from $\mathcal{N}(0,I)$.
- **Attention-map divergence** $\varepsilon_{\text{att}} = \frac{1}{n}\sum_t \mathrm{KL}(a_t^{\text{softmax}} \Vert a_t^{\phi})$, the quantity Hedgehog optimizes.
- **Recall capacity**: accuracy on MQAR (multi-query associative recall) at $(n, \text{num-KV-pairs})$, swept until accuracy drops below 95%.
- **Quality**: validation perplexity on a fixed corpus at fixed tokens-per-parameter, plus a recall-heavy slice (FDA, SQuAD, SWDE in the Based evaluation suite).
- **Throughput**: prefill tokens/s and decode tokens/s, measured with a chunked kernel, since $d'$ enters compute as $O(nd'd_v)$.

Assumptions and their status:

1. *$\phi$ nonnegative so the denominator is positive* — holds by construction for $\mathrm{elu}{+}1$, Taylor-2, FAVOR+; **violated** by learned maps unless clamped.
2. *Attention entries are bounded, $|q^\top k|/\sqrt d = O(\sqrt{\log n})$* — **violated in practice**; trained transformers exhibit massive-activation outliers, and this is exactly the regime where the hardness results bite.
3. *Feature map choice is separable from gating/decay* — **violated**: gated linear attention and the delta rule change what the state must store, and hence which $\phi$ is best.
4. *Sequence-level normalization is neutral* — **violated**: the $\epsilon$ and the denominator drive most reported instability.

## 3. State of the Art

**Established (ablated, reproduced):**

- $\phi(x)=\mathrm{elu}(x)+1$ (Katharopoulos et al., ICML 2020) is the baseline; it is stable and cheap and is clearly worse than softmax on recall.
- **FAVOR+** positive random features (Choromanski et al., ICLR 2021) give an *unbiased* estimator of the softmax kernel with variance that stays bounded as $q^\top k$ grows, unlike trigonometric random features. This is a proved property, not a benchmark claim.
- **Taylor-2** — $\phi(x) = [1, x, x\otimes x/\sqrt2]$ applied to a low-dim projection (Based; Arora et al., ICML 2024) — is the strongest simple deterministic map at moderate state, and its recall advantage was ablated against equal-state Mamba and sliding-window baselines.
- **Hedgehog** (Zhang et al., ICLR 2024): a learned $\phi(x)=[\exp(Wx), \exp(-Wx)]$ trained to match softmax attention weights by cross-entropy; the diagnosis that linear-attention failure correlates with *low attention entropy* (spikiness) and lost monotonicity is supported by direct measurement.

**Claimed but incompletely ablated:**

- Rankings between Hedgehog, ReBased (Aksenov et al., ACL 2024), PolySketchFormer (Kacham, Mirrokni, Zhong, ICML 2024), and cosFormer (Qin et al., ICLR 2022) are mostly reported at *equal parameters*, not equal state bytes, and mostly below 1.5B parameters.
- "Feature map $X$ closes the gap to softmax" claims are near-universally measured after distillation from a softmax teacher, which is a different problem than from-scratch training.

**Benchmark-number-only:** most cross-paper comparisons. Different tokenizers, corpora, head dims, and normalizations mean the published tables are not directly comparable.

## 4. What Is Known

- **A polynomial-size $\phi$ cannot approximate softmax attention in general.** Alman & Song (NeurIPS 2023) show that under SETH, if entries are bounded by $B = \omega(\sqrt{\log n})$, no $n^{1+o(1)}$-time algorithm approximates attention; Keles, Wei & Chandrasekaran (ALT 2023) give a matching statement. Any fixed $\phi$ with $d' = \mathrm{poly}(d)$ is such an algorithm, so exact mimicry is ruled out in the unbounded-entry regime.
- **Recall costs state, not cleverness.** The Based/Zoology line (Arora et al., ICLR 2024 and ICML 2024) shows MQAR accuracy is primarily a function of recurrent state size: at 360M parameters, sweeping state from ~2^7 to ~2^{13} scalars per head moves real-world recall (FDA/SWDE) monotonically, and Based with Taylor-2 sits on a better recall-throughput frontier than Mamba at matched state.
- **Spikiness and monotonicity are the failure mode.** Hedgehog (ICLR 2024) measures that $\mathrm{elu}{+}1$ attention distributions have far higher entropy than softmax's and reports recovering >99% of a from-scratch Transformer's quality on WikiText-103 at 125M scale after learning $\phi$, versus a several-point perplexity gap for fixed maps.
- **Random features are high-variance at useful $m$.** Performer needs $m$ in the hundreds per head to be competitive; at $m=64$ with $d=64$ the estimator variance dominates, and reported Performer LM perplexity is worse than deterministic Taylor-2 at comparable $d'$.
- **Nonnegativity matters more than kernel fidelity.** cosFormer (ICLR 2022) drops the exponential kernel entirely for ReLU plus a cosine reweighting and is competitive, which shows $\varepsilon_{\text{ker}}$ is not the operative objective.

## 5. What Is Not Known

- **Theoretically open.** Whether, restricted to bounded-entry inputs ($B = O(\sqrt{\log n})$) and to the *distribution of activations trained transformers actually produce*, there is an optimal $\phi$ at a given $d'$ — and what the tight lower bound on $d'$ is for $\varepsilon_{\text{att}} \le \varepsilon$. Upper bounds exist (FAVOR+, polynomial sketch); no matching lower bound in this restricted regime.
- **Empirically open.** Whether any feature-map ranking survives to $\ge 3$B parameters and $\ge 300$B tokens at *matched state bytes*. Every published head-to-head is $\le 1.4$B. The experiment is runnable today on a few hundred GPU-days; nobody has published it.
- **Empirically open.** Whether $\phi$ still matters once the recurrence has gating (GLA) or the delta rule (DeltaNet), which supply the forgetting that a better $\phi$ was partly compensating for.
- **Methodologically blocked.** There is no agreed definition of "equal capacity" for this comparison. Parameter count, state bytes, and prefill FLOPs give three different orderings, and no paper reports all three.

## 6. Why It Is Hard

**The obstruction is non-identifiability under a confounded budget.** Changing $\phi$ changes $d'$, which simultaneously changes (a) recurrent state size, (b) prefill and decode FLOPs, (c) the conditioning of the denominator $\phi(q_t)^\top\sum\phi(k_s)$, and (d) the effective learning rate through activation scale. A reported win for a feature map is therefore never attributable to the map: at equal parameters it is usually a state-size win, and at equal state it is usually a FLOPs or normalization difference. Fixing all four at once forces a joint sweep whose cost is multiplicative, which is why nobody has run it.

Second, the named evaluation does not measure the named thing. $\varepsilon_{\text{ker}}$ — softmax kernel fidelity — is the stated design objective of Performer and Taylor-2, yet cosFormer, which makes no attempt at it, is competitive. The quantity that predicts downstream quality is closer to $\varepsilon_{\text{att}}$ plus state size, and neither is what most papers optimize.

## 7. Current Research (as of 2026)

- **Learned maps under distillation**: converting pretrained softmax LLMs to linear attention (Hedgehog line at Stanford Hazy Research; LoLCATs-style linearization) is the main practical use of feature maps. *(frontier — verify)* The open question there is whether distillation quality transfers to from-scratch training.
- **Feature maps subsumed by richer recurrences**: GLA (Yang, Wang, Shen, Panda, Kim, ICML 2024), DeltaNet and Gated DeltaNet largely abandon exotic $\phi$ in favor of identity or SiLU features plus a better update rule. The live claim — unproven — is that $\phi$ is second-order once the state update is expressive. *(frontier — verify)*
- **Polynomial sketching**: PolySketchFormer (ICML 2024) reduces the $d^p$ blowup of degree-$p$ polynomial kernels by sketching, which is the only line giving explicit $d'$-vs-error bounds.
- **Hardware-shaped choice**: chunked kernels make $d'$ a tiling parameter; some maps are strictly better only at $d'$ values that align with tensor-core tiles. Cited informally, rarely reported.

## 8. Concrete Next Experiment

**Question:** at *matched recurrent state bytes*, does the feature map matter at all?

- **Scale.** 1.3B-parameter decoder, 100B tokens of a fixed corpus (e.g. a Pile or FineWeb-Edu slice), $H=16$ heads, $d_v = 64$. Five arms, all pinned to $S = 128 \times 64$ scalars per head (fp16, 16 KB/head/layer): (1) $\mathrm{elu}{+}1$ with $d=128$; (2) Taylor-2 with projection to $d_p=15$ so $d'=1+15+120=136$, truncated to 128; (3) FAVOR+ with $m=128$; (4) Hedgehog learned map with $d'=128$; (5) ReLU + cosine reweighting at $d'=128$.
- **Control arm.** Softmax attention at the same parameter count (state grows with $n$), plus a *state-matched* sliding-window softmax with window chosen so its KV cache equals 16 KB/head/layer. The sliding-window control is the one that matters: it is the null hypothesis that any linear map is just a lossy window.
- **Deciding number.** MQAR accuracy at $n=2048$ with 256 key-value pairs, reported as the **spread across arms 1–5**. If the spread is $<2$ accuracy points while the softmax control is $>20$ points above all of them, the feature map is not the operative variable and the field should stop tuning it. If the spread exceeds 10 points, the ranking is real and worth pushing to 7B.
- **Secondary readout.** Validation perplexity delta and decode tokens/s at batch 64, $n=8192$, to confirm no arm won by spending FLOPs.

Cost estimate: roughly 7 × 1.3B × 100B-token runs, about 3,000–5,000 H100-hours.

## 9. Key References

- **[Foundational]** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. — arXiv:2006.16236
- **[Foundational]** Krzysztof Choromanski et al. *Rethinking Attention with Performers.* ICLR 2021. — arXiv:2009.14794
- **[Foundational]** Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. — arXiv:2102.11174
- **[Foundational]** Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith, Lingpeng Kong. *Random Feature Attention.* ICLR 2021. — arXiv:2103.02143
- **[SOTA]** Michael Zhang, Kush Bhatia, Hermann Kumbong, Christopher Ré. *The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry.* ICLR 2024. — arXiv:2402.04347
- **[SOTA]** Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, Christopher Ré. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML 2024. — arXiv:2402.18668
- **[SOTA]** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. — arXiv:2312.06635
- **[Theory]** Josh Alman, Zhao Song. *Fast Attention Requires Bounded Entries.* NeurIPS 2023. — arXiv:2302.13214
- **[Theory]** Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, Chinmay Hegde. *On the Computational Complexity of Self-Attention.* ALT 2023. — arXiv:2209.04881
- **[Theory/SOTA]** Praneeth Kacham, Vahab Mirrokni, Peilin Zhong. *PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels.* ICML 2024. — arXiv:2310.01655
- **[Related]** Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, Yiran Zhong. *cosFormer: Rethinking Softmax in Attention.* ICLR 2022. — arXiv:2202.08791
- **[Related]** Yaroslav Aksenov, Nikita Balagansky, Sofia Maria Lo Cicero Vaina, Boris Shaposhnikov, Alexey Gorbatovski, Daniil Gavrilov. *Linear Transformers with Learnable Kernel Functions are Better In-Context Models.* ACL 2024. — arXiv:2402.10644
- **[Survey]** Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, Christopher Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR 2024. — arXiv:2312.04927

## 10. Worked Example

Take one head with $d = 64$, $d_v = 64$, fp16 state.

| Map | $d'$ | State scalars | State bytes |
|---|---|---|---|
| $\mathrm{elu}(x)+1$ | 64 | $64\times64 = 4{,}096$ | 8 KB |
| Taylor-2 on 16-dim projection | $1+16+\binom{17}{2}=153$ | $153\times64 = 9{,}792$ | 19.1 KB |
| FAVOR+, $m=256$ | 256 | $256\times64=16{,}384$ | 32 KB |

At "equal parameters" — the standard comparison — these three arms differ by **4× in state bytes**. Zoology's measured relation is that MQAR accuracy rises roughly with $\log(\text{state})$ until saturation: over that 4× range, at $n=512$ with 64 KV pairs, the difference attributable to state alone is on the order of 20–30 accuracy points at 360M scale. Published feature-map deltas are typically 5–15 points.

So the reported advantage of Taylor-2 or FAVOR+ over $\mathrm{elu}{+}1$ is **smaller than the state-size confound it is measured through**. Re-run the same three arms at a pinned $S = 8{,}192$ scalars ($\mathrm{elu}{+}1$ with $d=128$; Taylor-2 projected to $d_p=14$, $d'=120$, padded; FAVOR+ with $m=128$) and the published ordering has, to date, never been reproduced. That is the obstruction in one table: the experiment everyone cites does not hold fixed the variable that dominates the outcome.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*