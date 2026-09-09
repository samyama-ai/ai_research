---
id: 02-attention/nopos-positional-emergence
title: "Positional Information Emergence Without Explicit Encoding"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Positional Information Emergence Without Explicit Encoding

> **Topic:** Attention Mechanisms · **ID:** `02-attention/nopos-positional-emergence` · **Status:** partially-solved

## 1. Problem Statement

A decoder-only Transformer with no positional encoding (NoPE) — no sinusoidal or learned absolute embedding, no rotary rotation, no relative bias — still learns order-sensitive functions. The question is what mechanism supplies the position signal, how far it extends, and whether it is a substitute for or merely a supplement to explicit encoding.

Three variants, with different difficulty:

- **Measurement.** Given a trained NoPE model and input $x_{1:n}$, how much information about the index $i$ is recoverable from the residual stream at position $i$? Output: a decodability curve over layers and positions. Solving it means a probe whose accuracy is not confounded by token identity or content statistics.
- **Method.** Can NoPE match or beat RoPE/ALiBi/FIRE on perplexity and length generalization at frontier scale ($\geq 10^9$ parameters, $\geq 10^{11}$ tokens, $\geq 32$k context)? Decision predicate: a paired training run where NoPE is within noise on in-distribution loss and strictly better beyond train length.
- **Theory.** Characterize the class of position-dependent functions a causally masked, permutation-equivariant-except-for-the-mask attention stack can express and *learn* at finite depth and precision. Expressivity is settled; learnability is not.

## 2. Formal Setting

Tokens $x_{1:n} \in \mathcal{V}^n$, embeddings $E(x_i) \in \mathbb{R}^d$ with **no** index-dependent term. Layer $\ell$ attention with causal mask $M_{ij} = 0$ for $j \le i$, $-\infty$ otherwise:

$$A^{(\ell)}_{ij} = \frac{\exp\!\big(q_i^\top k_j / \sqrt{d_h} + M_{ij}\big)}{\sum_{j' \le i} \exp\!\big(q_i^\top k_{j'} / \sqrt{d_h} + M_{ij'}\big)}, \qquad h_i^{(\ell)} = \sum_{j \le i} A^{(\ell)}_{ij} v_j .$$

Without the mask the map $x_{1:n} \mapsto h_{1:n}$ is permutation-equivariant, so no position information exists. The mask makes the *cardinality* $|\{j : j \le i\}| = i$ observable: the softmax denominator averages over $i$ terms.

**Measured quantities.**

- **Position decodability.** Train probe $g_\theta: \mathbb{R}^d \to \{1,\dots,n\}$ on frozen $h_i^{(\ell)}$ from held-out text. Report $\mathrm{MAE}^{(\ell)} = \mathbb{E}_i |g_\theta(h_i^{(\ell)}) - i|$ and normalized $\mathrm{MAE}/n$. Baseline control: the same probe on shuffled position labels, and on a bag-of-tokens featurization (content leaks position — sentence-initial tokens are not uniform over $i$).
- **Variance signal.** For i.i.d. values with variance $\sigma^2$ and near-uniform attention, $\mathrm{Var}[h_i] \approx \sigma^2/i$. So $\log \mathrm{Var}[h_i] \approx \log\sigma^2 - \log i$: a $-1$ slope in $\log$–$\log$ is the fingerprint of the counting mechanism. Measured empirically as the across-channel variance of $h_i$ at fixed $i$, averaged over sequences.
- **Length generalization gap.** $\Delta = \mathcal{L}(\text{test at } L_{\text{test}}) - \mathcal{L}(\text{test at } L_{\text{train}})$, with $L_{\text{test}} = \kappa L_{\text{train}}$, $\kappa \in \{2,4,8\}$, on data matched for topic and token distribution.

**Assumptions, and which break.** (i) Attention is near-uniform — violated: trained heads are sharply peaked, and attention sinks on the BOS token dominate. (ii) Values are i.i.d. across positions — violated by content correlation. (iii) Probes measure representation content, not probe capacity — violated for high-capacity probes; requires a selectivity control. (iv) Sequences are packed uniformly — violated: document-boundary packing makes "position in document" and "position in context window" different variables that most papers conflate.

## 3. State of the Art

**Established.** Haviv et al. (EMNLP Findings 2022) trained decoder-only LMs with no positional encoding and showed position is linearly decodable from early layers, and that removing the causal mask destroys both the decodability and the language modeling ability. Chi, Fan, Rudnicky & Ramadge (ACL 2023) gave the mechanism: causal masking alone induces a position-dependent variance in self-attention outputs, present at random initialization, before any training. Kazemnejad et al. (NeurIPS 2023) proved a NoPE decoder can express absolute position (a first layer can write $1/i$ or a monotone function of $i$ into the stream) and, given that, relative position; and measured NoPE beating T5 relative bias, ALiBi and RoPE on length generalization for small decoder-only models on synthetic reasoning tasks.

**Claimed but unablated.** That NoPE is a *drop-in* replacement at scale. The strongest published NoPE-vs-RoPE comparisons are at $\lesssim 1.3$B parameters with modest token budgets; the frontier-scale claim rests on hybrid architectures (a minority of layers with no rotation, the rest RoPE), not pure NoPE. *(frontier — verify)* Interleaved NoPE/RoPE layer schedules appear in several 2024–2025 long-context systems; the reported gains are benchmark numbers on long-context suites, without a layer-schedule ablation isolating what the NoPE layers contribute.

**Benchmark-only results.** Long-context retrieval scores (needle-in-a-haystack, RULER) for hybrid NoPE models are reported as aggregate accuracies. They do not separate "position is representable" from "the retrieval head found it".

## 4. What Is Known

- **Causal masking is sufficient, and necessary, for position in NoPE.** Bidirectional NoPE models collapse to bag-of-words behaviour. Demonstrated at 125M–1.3B parameters (Haviv et al. 2022).
- **The signal exists at initialization.** In an untrained causal Transformer, output variance decays approximately as $1/i$ over the first few hundred positions (Chi et al. 2023). This is a $-1$ slope on $\log \mathrm{Var}$ vs $\log i$, and it is architectural, not learned.
- **Decodability is early and near-perfect for short contexts.** Linear probes recover absolute position from layer-1 hidden states with small normalized error at context lengths of 512–1024 in models of a few hundred million parameters.
- **Perplexity cost is small at small scale.** NoPE reported within roughly one perplexity point of learned absolute encoding at $\leq 1.3$B parameters, standard web-text corpora, contexts $\leq 1024$ (Haviv et al. 2022). Independent reproduction at matched token budgets is thin.
- **Length generalization advantage is real but narrow.** Kazemnejad et al. (2023) measured NoPE best-on-average across addition, SCAN-style, and copying tasks at roughly $10^7$–$10^8$ parameters with $\kappa \approx 2$. The advantage is not robust: Zhou et al. (2024) showed length generalization on addition swings by tens of points with format and seed, and that FIRE plus randomized positions plus reversed formatting reaches roughly $2.5\times$ train length on 40-digit addition — better than any NoPE result at that task.
- **Attention sinks provide a second channel.** A dominant BOS sink gives every position a fixed reference, so $A_{i,1}$ itself is a monotone proxy for $1/i$ under near-uniform competition.

## 5. What Is Not Known

- **Theoretically open.** Whether the $1/i$ counting signal is *learnable* to $\Theta(\log n)$ effective precision at fixed $d$ and finite float precision. Expressivity constructions exist; no lower bound on the depth or precision needed to maintain position resolution as $n$ grows, and no proof that gradient descent finds such a solution.
- **Theoretically open.** Whether NoPE's advantage on length generalization follows from a bias toward relative-position solutions, or is an artifact of explicit encodings being *out of distribution* beyond train length. These predict different behaviour under randomized position training and have not been separated.
- **Empirically open.** Pure NoPE versus RoPE at $\geq 7$B parameters, $\geq 10^{12}$ tokens, $\geq 32$k context, with matched compute. Runnable today; the run costs on the order of $10^{22}$ FLOPs per arm. Nobody has published it.
- **Methodologically blocked.** "How much positional information is present" has no agreed estimator. Probe accuracy conflates representation content with probe capacity and with content-position correlation in natural text. There is no accepted control that removes the second confound, so cross-paper decodability numbers are not comparable.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability.** In natural text, token identity predicts position: `<s>`-adjacent tokens, capitalized openers, and document-final punctuation all carry index information. A probe reading $h_i$ therefore has two sources, and no published protocol separates them at the required resolution. Worse, position-in-context and position-in-document are different variables that packing conflates; a model can score well on one while carrying none of the other.

The second obstruction is **precision, not expressivity**. The counting signal has magnitude $\sim 1/i$. Distinguishing $i = 4000$ from $i = 4001$ requires resolving a difference of order $6 \times 10^{-8}$ in a channel that also carries content and passes through LayerNorm, which rescales. Whether that survives bf16 in a trained model is an empirical question about a quantity that shrinks as the very thing under test — context length — grows.

## 7. Current Research (as of 2026)

- **Hybrid layer schedules.** Interleaving NoPE layers with RoPE layers to get local precision plus unbounded-range behaviour; pursued in long-context industrial models and in follow-up analyses of what rotation does to attention (Barbero et al., 2024, on why RoPE helps). *(frontier — verify the specific schedules; vendor reports rarely ablate them.)*
- **Mechanistic decomposition.** Locating the heads and channels that carry the emergent index, and relating them to attention sinks and to induction-head formation.
- **Position as a learned, content-conditioned quantity.** Contextual Position Encoding (Golovneva et al., 2024) makes position a function of content rather than of token index — the constructive counterpart to the NoPE question.
- **Length-generalization theory.** RASP-L-style characterizations (Zhou et al., 2023) of which algorithms length-generalize, which predicts NoPE's task-dependent successes better than any positional-encoding-specific account.

## 8. Concrete Next Experiment

**Scale.** Two decoder-only models, 1.4B parameters, identical data order, 300B tokens, train context $L_{\text{train}} = 4096$, bf16.

**Arms.** (a) NoPE. (b) Control: identical model with RoPE, $\theta = 10^4$. Second control for the measurement confound: both arms evaluated on a *content-decorrelated* eval set — sequences of tokens sampled i.i.d. from the unigram distribution, so token identity carries zero position information.

**Measurement.** A rank-1-regularized linear probe on layer-1 and mid-stack residuals, trained on the decorrelated set, evaluated at $L_{\text{test}} \in \{4096, 8192, 16384\}$. Report normalized MAE $= \mathrm{MAE}/L_{\text{test}}$ and the selectivity gap against shuffled labels.

**The deciding number.** Normalized position-probe MAE for NoPE at $L_{\text{test}} = 16384$ (i.e. $4\times$ train length) on the decorrelated set. **Below 0.05**: the emergent code is a genuine absolute-position channel that extrapolates, and NoPE's length-generalization advantage has a mechanistic explanation. **Above 0.25**: the emergent code is local and saturating, NoPE's advantage is the absence of an out-of-distribution signal rather than the presence of a good one, and hybrid schedules are the only viable use.

Cost: about $2 \times 2.5 \times 10^{21}$ FLOPs, roughly 4k A100-days total. Within a single academic cluster-month.

## 9. Key References

- **[Foundational]** Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, Omer Levy. *Transformer Language Models without Positional Encodings Still Learn Positional Information.* Findings of EMNLP, 2022. — arXiv:2203.16634
- **[Foundational]** Ta-Chung Chi, Ting-Han Fan, Alexander I. Rudnicky, Peter J. Ramadge. *Latent Positional Information is in the Self-Attention Variance of Transformer Language Models Without Positional Embeddings.* ACL (short), 2023.
- **[SOTA]** Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS, 2023. — arXiv:2305.19466
- **[SOTA]** Yongchao Zhou, Uri Alon, Xinyun Chen, Xuezhi Wang, Rishabh Agarwal, Denny Zhou. *Transformers Can Achieve Length Generalization But Not Robustly.* 2024. — arXiv:2402.09371
- **[Analysis]** Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, Petar Veličković. *Round and Round We Go! What Makes Rotary Positional Encodings Useful?* 2024. — arXiv:2410.06205
- **[Method]** Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar. *Contextual Position Encoding: Learning to Count What's Important.* 2024. — arXiv:2405.18719
- **[Theory]** Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, Preetum Nakkiran. *What Algorithms can Transformers Learn? A Study in Length Generalization.* ICLR, 2024. — arXiv:2310.16028
- **[Related]** Md Amirul Islam, Sen Jia, Neil D. B. Bruce. *How Much Position Information Do Convolutional Neural Networks Encode?* ICLR, 2020. — the same emergence phenomenon from zero-padding rather than causal masking.

## 10. Worked Example

Take a single untrained causal attention head, $d_h = 64$, values $v_j \sim \mathcal{N}(0, I)$ i.i.d., attention approximately uniform over $j \le i$. Then $h_i = \frac{1}{i}\sum_{j\le i} v_j$ and each channel has $\mathrm{Var}[h_{i,c}] = 1/i$, standard deviation $i^{-1/2}$.

| $i$ | $\mathrm{sd}[h_{i,c}] = i^{-1/2}$ | $\mathrm{sd}[h_{i}] - \mathrm{sd}[h_{i+1}]$ |
|---|---|---|
| 4 | 0.5000 | $2.9\times10^{-2}$ |
| 64 | 0.1250 | $9.7\times10^{-4}$ |
| 1024 | 0.03125 | $1.5\times10^{-5}$ |
| 4096 | 0.015625 | $1.9\times10^{-6}$ |
| 16384 | 0.0078125 | $2.4\times10^{-7}$ |

An estimator of $i$ from $d_h = 64$ channels has relative standard error $\approx 1/\sqrt{2 d_h} = 8.8\%$ on the variance, hence about $\pm 18\%$ on $i$ — roughly $\pm 720$ positions at $i = 4000$, from one head, one sample. Averaging over $H = 32$ heads cuts that to about $\pm 3\%$, or $\pm 125$ positions.

Now the obstruction. bf16 has 8 mantissa bits: relative resolution $\approx 2^{-8} = 3.9 \times 10^{-3}$. At $i = 4096$, one position of index costs a relative change in $\mathrm{sd}$ of $\tfrac{1}{2i} = 1.2 \times 10^{-4}$ — thirty times below bf16's representable step. The counting channel cannot distinguish adjacent positions past $i \approx 128$ in bf16 without amplification, and LayerNorm actively rescales away the magnitude the signal lives in.

So the mechanism gives a *coarse, logarithmically compressing* position code: excellent at $i < 100$, degrading as $\sqrt{i}$ in effective resolution, and hitting numerical floor well before 4k context. That is exactly consistent with the observed pattern — NoPE wins on short synthetic tasks and short-context perplexity, and no one has shown it works at 32k. The deciding experiment in §8 measures whether trained models find an amplification path that beats this floor, or merely inherit it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*