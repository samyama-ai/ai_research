---
id: 01-tokenization/softmax-bottleneck-large-vocab
title: "Softmax Bottleneck Under Very Large Vocabularies"
topic: 01-tokenization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Softmax Bottleneck Under Very Large Vocabularies

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/softmax-bottleneck-large-vocab` · **Status:** partially-solved

## 1. Problem Statement

A language model produces next-token distributions by a single linear map from a $d$-dimensional hidden state into $V$ logits, followed by softmax. The matrix of log-probabilities the model can express therefore has rank at most $d+2$. When $V \gg d$ — the regime every 2024–2026 frontier model now sits in (Gemma 3 1B: $d=1152$, $V=262144$; Llama 3 8B: $d=4096$, $V=128256$) — the reachable set is a vanishingly thin slice of the space of distributions.

Three variants, with different difficulty:

- **Theory variant.** Is the true log-probability matrix of natural language high-rank in a way that a rank-$(d{+}2)$ family cannot $\epsilon$-approximate at the $\epsilon$ that matters for loss? Rank is a statement about *exact* representation; loss cares about approximation.
- **Measurement variant.** Given a trained model, decide whether its output head is the binding constraint, separately from data, depth, and optimization. No accepted estimator exists.
- **Method variant.** Build a head that lifts the rank at $V \ge 128\text{k}$ for less than a few percent added FLOPs and memory, and show it wins on a compute-matched control.

Solving it means: a compute-matched training run at frontier vocabulary where a rank-lifted head beats a plain softmax head on a vocabulary-independent metric (bits per byte), reproduced independently.

## 2. Formal Setting

Let $\mathcal{V}$ be the vocabulary, $V=|\mathcal{V}|$, and let $c$ range over $N$ contexts. The ground-truth log-probability matrix is
$$A \in \mathbb{R}^{N\times V},\qquad A_{c,x}=\log P^*(x\mid c).$$
A model with hidden states $H\in\mathbb{R}^{N\times d}$, output embeddings $W\in\mathbb{R}^{V\times d}$, bias $b\in\mathbb{R}^V$ produces
$$\log \hat P = \mathrm{LogSoftmax}\!\left(HW^\top + \mathbf{1}b^\top\right).$$
Softmax is invariant to a per-row shift, so the model matches $A$ iff $HW^\top + \mathbf{1}b^\top = A + \mathbf{c}\mathbf{1}^\top$ for some $\mathbf{c}\in\mathbb{R}^N$. Hence exact fit requires
$$\operatorname{rank}(A) \le d+2 .$$

Quantities as measured:

- **Effective rank of the logit matrix.** Sample $N$ contexts, collect $L = HW^\top$, take singular values $\sigma_1\ge\dots$; report $r_\varepsilon=\\#\{i:\sigma_i>\varepsilon\sigma_1\}$ at a stated $\varepsilon$ (e.g. $10^{-2}$), or the entropy-based effective rank $\exp\big(-\sum_i p_i\log p_i\big)$, $p_i=\sigma_i/\sum_j\sigma_j$ (Roy & Vetterli, 2007). Both need $N \gg d$ to be meaningful and both depend on $\varepsilon$.
- **Loss, vocabulary-independent.** $\mathrm{BPB} = \frac{1}{\ln 2}\sum_c \mathrm{NLL}(c)\,/\,\sum_c \mathrm{bytes}(c)$. Perplexity per token is not comparable across vocabularies and must not be used.
- **Head cost.** Forward FLOPs per token $\approx 2dV$ for one softmax, $\approx 2KdV$ for $K$ mixture components; peak logit memory $= B\!\cdot\!V\!\cdot\!K\!\cdot\!\text{bytes}$ for a microbatch of $B$ tokens.

Assumptions, and which are violated:

1. *$A$ exists and is estimable.* Violated — $P^*$ is never observed; every empirical claim about $\operatorname{rank}(A)$ substitutes a model's own logits or a finite-sample count matrix, which is rank-limited by $N$.
2. *Exact representation is the requirement.* Violated — training minimizes cross-entropy, so only $\epsilon$-approximation matters, and a low-rank matrix approximates a high-rank one well whenever singular values decay fast.
3. *Weight tying is neutral.* Frequently violated — tying input and output embeddings couples the head's geometry to input representation learning, so head ablations are confounded unless tying is held fixed.
4. *The hidden state is unconstrained on $\mathbb{R}^d$.* Violated — final-layer normalization (RMSNorm) restricts $h$ to a sphere-like manifold, shrinking the reachable set further than the rank bound alone implies.

## 3. State of the Art

**Established.** Mixture of Softmaxes (Yang et al., ICLR 2018) — $\log\hat P = \log\sum_{k=1}^{K}\pi_k(c)\,\mathrm{softmax}(h_k W^\top)$ — is the canonical fix and is established at small vocabulary: AWD-LSTM-MoS reaches test perplexity 55.97 on Penn Treebank ($V\!=\!10\text{k}$) and 63.33 on WikiText-2 ($V\!=\!33\text{k}$), against 57.3 / 65.8 for the same AWD-LSTM backbone. Mixtape (Yang et al., NeurIPS 2019) reproduces most of the gain at 20–30% overhead over plain softmax rather than $K\times$.

**Established, negative-side.** Demeter, Kimmel & Downey (ACL 2020) prove that a token whose output embedding lies inside the convex hull of the others can never be the argmax and has a bounded maximum probability. Grivas, Bogoychev & Lopez (ACL 2022) give an exact test for unargmaxable classes and find that released NMT models with $V\gg d$ do contain them, but only a handful — the structural weakness is real and quantitatively small at $V\approx 32\text{k}$.

**Claimed but unablated at scale.** Chang & McCallum (ACL 2022) show the bottleneck prevents representing genuinely multi-modal next-word distributions and propose multi-facet softmax; gains are reported on GPT-2-scale models with $V=50257$, not at $V\ge128\text{k}$. Godey, de la Clergerie & Sagot (2024) attribute performance saturation in small LMs to rank degeneracy of the output head, with the transition around $d\approx1000$ at $V\approx50\text{k}$ — an observational correlation across the Pythia suite, not a controlled head swap.

**Benchmark-number-only.** Every published claim that large vocabularies are beneficial (Tao et al., *Scaling Laws with Vocabulary*, NeurIPS 2024, which predicts an optimal $V\approx216\text{k}$ for a 3B-parameter model trained compute-optimally, versus the 32k actually used in Llama 2) is a loss/benchmark number under a plain softmax head. None isolates whether the head was binding.

## 4. What Is Known

- **The rank bound is a theorem**, not an empirical claim: $\operatorname{rank}(HW^\top+\mathbf{1}b^\top - \mathbf{c}\mathbf{1}^\top)\le d+2$.
- **MoS gains shrink as backbones and data grow.** The 1.3–2.5 perplexity-point PTB/WT2 gains were measured at $d\le1150$, $V\le33\text{k}$, on ~2–100M training tokens. No published replication shows a comparable gain on a $\ge1$B-parameter transformer trained on $\ge100$B tokens.
- **Unargmaxable tokens exist but are rare.** At $V\approx30\text{k}$, $d=512$, counts found in real translation models are in the tens, not thousands (Grivas et al., ACL 2022).
- **Head cost is already a large fraction of small models.** At $d=1152$, $V=262144$: output projection $=302$M parameters, $2dV=604$ MFLOP/token, roughly 30% of a 1B-parameter model's forward compute.
- **Frontier practice moved the ratio the wrong way for the bottleneck.** $V/d$ went from $\approx 65$ (GPT-2 small: 50257/768) to $\approx 228$ (Gemma 3 1B) between 2019 and 2025.

## 5. What Is Not Known

- **Theoretically open.** Whether the singular-value spectrum of natural-language $A$ decays fast enough that rank $d+2$ suffices to within the cross-entropy gap that separates model generations (~0.02 bits/byte). No lower bound of the form "any rank-$r$ head incurs $\ge \delta(r)$ excess cross-entropy on a stated language model of text" exists. This is the crux: the rank theorem constrains exact fit, and nobody has converted it into an approximation-error bound.
- **Empirically open.** Whether MoS/Mixtape/multi-facet heads give any bits-per-byte gain at $V\ge128\text{k}$, $d\ge2048$, $\ge100$B tokens, under compute matching. The run is affordable (~$10^{21}$ FLOPs per arm) and has not been published.
- **Methodologically blocked.** Deciding whether the head is *the* binding constraint in a given trained model. Effective rank of the observed logit matrix is $\varepsilon$-dependent and is upper-bounded by $\min(N,d+2)$ by construction, so it cannot distinguish "the head cannot express more" from "the body does not produce more".

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an untestable counterfactual**. To show the head binds you must observe the distribution the model *would* have produced with an unconstrained head — which does not exist to be measured. Every available proxy (effective rank, unargmaxable-token counts, multi-modality tests) measures a property of the constrained model and is therefore mechanically consistent with the constraint being slack.

Second obstruction: **cost asymmetry**. Lifting rank by $K$ multiplies the single largest activation tensor in the model. At $V=262144$ with a microbatch of $B=8192$ tokens in bf16, one logit tensor is $8192\times262144\times2 = 4.3$ GB; $K=15$ is 64 GB of logits per microbatch, which does not fit and forces chunking that destroys the comparison's throughput parity. So the honest control arm — same FLOPs, same wall clock, same memory — is expensive to construct, and cheap comparisons silently give the rank-lifted arm more compute.

## 7. Current Research (as of 2026)

- **Vocabulary scaling laws.** Tao et al. (NeurIPS 2024) and follow-ups treat $V$ as a jointly optimized variable; the implied 200k+ vocabularies make the bottleneck acute rather than resolving it.
- **Asymmetric input/output vocabularies.** Over-tokenized-transformer style work (ByteDance Seed, 2025) reports that scaling the *input* vocabulary into the millions helps while scaling the *output* vocabulary is limited by exactly this head cost — indirect evidence that the output side is constrained by compute, not expressivity. *(frontier — verify.)*
- **Multi-token and factorized heads.** Multi-token prediction heads (Meta, 2024) and low-rank/product-quantized output factorizations raise the effective number of distributions emitted per position; whether this changes the rank argument is unanalyzed. *(frontier — verify.)*
- **Rank-degeneracy diagnostics for small LMs.** Inria Almanach (Godey et al.) continue the saturation line.

## 8. Concrete Next Experiment

**Scale.** Three transformer decoders, $d=2048$, 24 layers, ~1.4B non-embedding parameters, BPE vocabulary $V=131072$, 100B tokens of a fixed public corpus, identical data order, untied output embeddings in all arms.

**Arms.**
1. **Control:** plain softmax head.
2. **Treatment:** MoS with $K=4$ ($4\times$ head FLOPs), with the body shrunk to 22 layers so total training FLOPs match the control within 1%.
3. **Capacity control (the arm usually omitted):** plain softmax, but the final hidden state is widened to $d_{\text{out}}=8192$ by one extra linear layer before the output projection — this raises head FLOPs by the same factor as arm 2 and raises the rank bound to $8194$, without any mixture. Same total-FLOP matching.

**Deciding number.** Validation **bits per byte** on a held-out 1B-token split. If arm 2 or arm 3 beats arm 1 by $\ge 0.005$ bits/byte (roughly a third of a typical model-generation improvement, and ~5$\times$ the seed-to-seed spread at this scale), the head is binding at frontier vocabulary and the problem moves to method design. If both land within $\pm0.002$ bits/byte of control, the rank bound is slack at $V=131\text{k}$ and the open question narrows to the theory variant. Arm 3 versus arm 2 separates "rank matters" from "extra head compute matters" — if arm 3 matches arm 2, mixtures buy nothing that width does not.

**Cost.** $\approx 3\times 8.4\times10^{20}$ FLOPs; roughly 2,000 H100-hours per arm.

## 9. Key References

- **[Foundational]** Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen. *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.* ICLR 2018. — arXiv:1711.03953
- **[SOTA/efficiency]** Zhilin Yang, Thang Luong, Ruslan Salakhutdinov, Quoc V. Le. *Mixtape: Breaking the Softmax Bottleneck Efficiently.* NeurIPS 2019.
- **[Theory]** David Demeter, Gregory Kimmel, Doug Downey. *Stolen Probability: A Structural Weakness of Neural Language Models.* ACL 2020.
- **[Theory]** Andreas Grivas, Nikolay Bogoychev, Adam Lopez. *Low-Rank Softmax Can Have Unargmaxable Classes in Theory but Rarely in Practice.* ACL 2022.
- **[Method]** Haw-Shiuan Chang, Andrew McCallum. *Softmax Bottleneck Makes Language Models Unable to Represent Multi-mode Word Distributions.* ACL 2022.
- **[Method]** Sekitoshi Kanai, Yasuhiro Fujiwara, Yuki Yamanaka, Shuichi Adachi. *Sigsoftmax: Reanalysis of the Softmax Bottleneck.* NeurIPS 2018.
- **[Method]** Octavian-Eugen Ganea, Sylvain Gelly, Gary Bécigneul, Aliaksei Severyn. *Breaking the Softmax Bottleneck via Learnable Monotonic Pointwise Non-linearities.* ICML 2019.
- **[Empirical]** Nathan Godey, Éric de la Clergerie, Benoît Sagot. *Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck.* 2024.
- **[Scaling]** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS 2024.
- **[Systems]** Édouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou. *Efficient Softmax Approximation for GPUs.* ICML 2017. — arXiv:1609.04309
- **[Measurement]** Olivier Roy, Martin Vetterli. *The Effective Rank: A Measure of Effective Dimensionality.* EUSIPCO 2007.

## 10. Worked Example

Take Gemma 3 1B: $d=1152$, $V=262144$, output projection $1152\times262144=302$M parameters.

**The bound.** The model's reachable log-probability matrices have rank $\le 1154$. As a fraction of the ambient dimension, $1154/262144 = 0.44\%$. This looks fatal and is not, which is the point.

**Why it is not fatal.** Fit a rank-$r$ truncation to the model's own logit matrix on $N=100{,}000$ sampled contexts. The excess cross-entropy of the rank-$r$ reconstruction against the full-rank-1154 logits decays with the singular spectrum. If the spectrum decays as $\sigma_i \propto i^{-1}$ — the typical observed decay for transformer logit matrices — then truncating from 1154 to 512 discards $\sum_{i>512}\sigma_i^2 / \sum_i \sigma_i^2 \approx 0.001$ of the energy. Under that decay, going *up* from 1154 to 2308 would recover an amount of the same order. Converting energy to bits requires the loss curvature, and no published calibration of that conversion exists: **this is exactly the missing approximation bound from §5.**

**The cost that blocks the test.** Run the same model with $K=15$ MoS components. Head FLOPs per token go from $2dV = 604$ MFLOP to $9.1$ GFLOP. Total forward compute per token goes from ~2.0 GFLOP to ~10.5 GFLOP — a $5.3\times$ slowdown for a head change. At that price, MoS with $K=15$ must beat a plain-softmax model trained $5.3\times$ longer, and it does not: a $5.3\times$ token increase at this scale is worth roughly 0.03–0.05 bits/byte on Chinchilla-style curves, against a hypothesized head gain bounded by the $\sim0.001$ energy fraction above.

**What the example makes visible.** The obstruction is not that the rank bound is loose or tight — it is that nobody has measured the constant converting discarded spectral energy into bits, so the only way to decide is a compute-matched training run, and the compute-matched configuration ($K=4$ with a shortened body, §8) is the first one where the treatment can plausibly win.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*