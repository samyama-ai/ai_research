---
id: 01-tokenization/dynamic-token-pooling
title: "Dynamic Token Pooling and Boundary Prediction"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Dynamic Token Pooling and Boundary Prediction

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/dynamic-token-pooling` · **Status:** empirically-open

## 1. Problem Statement

A byte-level language model can learn *where* to segment its own input instead of accepting a fixed BPE vocabulary. **Dynamic token pooling** does this with a hierarchy: a small local network reads raw bytes, a **boundary predictor** emits a binary decision at each position, contiguous runs between boundaries are pooled into a single vector, and a large "global" network runs on the shortened sequence. The pooled representations are then upsampled back to byte resolution for the output head.

- **Input.** A byte string $x_{1:T} \in \{0,\dots,255\}^T$.
- **Output.** A boundary vector $b_{1:T} \in \{0,1\}^T$ and a next-byte distribution $p_\theta(x_t \mid x_{<t})$ that is causal with respect to $b$.
- **Objective.** Minimise bits-per-byte at a fixed training and inference FLOP budget, where the number of global-network steps is $\sum_t b_t$ and is therefore *data-dependent*.

Three variants that are routinely conflated:

- **Method variant (empirically open).** Is there a learned boundary rule that beats both fixed-stride pooling and a frozen BPE tokenizer at matched compute, at $\ge 10^{22}$ FLOPs?
- **Measurement variant (methodologically blocked in part).** What is "matched compute" when the compression rate $T/\sum_t b_t$ is itself an output of the model and varies by document, language, and domain?
- **Theory variant (theoretically open).** Under what conditions on the source is the loss-minimising segmentation identifiable, and does the optimal segmentation coincide with any linguistic or information-theoretic one?

Solving it means: a boundary predictor that is learned end-to-end, needs no tokenizer supervision, and yields a strictly better bits-per-byte / FLOP Pareto frontier than BPE at scale, reproduced by an independent group.

## 2. Formal Setting

Let $h_{1:T} \in \mathbb{R}^{T \times d}$ be local-encoder states. A boundary predictor $g_\phi$ gives $\pi_t = \sigma(g_\phi(h_{\le t})) \in [0,1]$, and $b_t \sim \mathrm{Bernoulli}(\pi_t)$. Boundaries induce segments $S_k = [s_k, e_k]$ with $e_k$ the $k$-th index where $b_t = 1$. Pooling is a map $z_k = \mathrm{pool}(h_{s_k:e_k})$ — mean, last-state, or cross-attention with $m$ latent queries.

**Compression rate**, measured as the ratio over a held-out corpus, not per batch:
$$\rho = \frac{T}{\sum_{t=1}^{T} b_t}.$$

**Compute.** With global width $d_g$, local width $d_\ell$, $L_g$ and $L_\ell$ layers, the forward cost per byte is measured as
$$C_{\text{byte}} \approx 2\left(L_\ell \, c(d_\ell) + \frac{1}{\rho} L_g \, c(d_g)\right), \qquad c(d)=12d^2 + \text{attn}(d,n),$$
with $\mathrm{attn}$ the sequence-dependent term. Report $C_{\text{byte}}$ as measured wall-clock FLOPs from a profiler, not analytic, because ragged segments force padding: the *realised* rate $\tilde\rho$ satisfies $\tilde\rho \le \rho$ and is what the hardware pays.

**Loss.** The training objective is bits-per-byte plus a rate regulariser,
$$\mathcal{L} = \underbrace{-\frac{1}{T\ln 2}\sum_t \log p_\theta(x_t \mid x_{<t}, b)}_{\text{BPB}} + \lambda \left(\frac{1}{T}\sum_t \pi_t - \frac{1}{\rho^\star}\right)^2,$$
with $\rho^\star$ a target rate. Gradients through $b_t$ use Gumbel-sigmoid straight-through (Jang et al., ICLR 2017; Bengio et al., 2013), or the boundary is supplied by a non-differentiable heuristic (entropy spikes, whitespace, a frozen unigram tokenizer).

**Assumptions, and which are violated.**

1. *Pooling is lossless enough that upsampling recovers byte-level detail.* Violated: reconstruction error grows with segment length; this is why residual/skip connections from the local encoder are mandatory in every working system.
2. *$\rho$ is stationary across the corpus.* Violated hard. Entropy-driven boundaries fire far more often on code, non-Latin scripts, and numbers than on English prose; $\rho$ can vary 2–3× across domains within one corpus.
3. *Straight-through gradients are a usable estimator of $\partial \mathcal{L}/\partial \phi$.* Violated in the sense that the bias is unbounded and unmeasured; no paper reports the variance of this estimator.
4. *Causality holds.* A boundary predictor that uses $h_t$ to decide $b_t$ and then pools $h_{s_k:t}$ is causal; one that uses next-byte entropy computed by a separate model is causal only if that model is itself causal.

## 3. State of the Art

**Established (ablated, with controls).**

- **Dynamic Token Pooling** (Nawrot, Chorowski, Łańcucki, Ponti; ACL 2023) is the canonical formulation: it compares learned Gumbel boundaries, entropy-spike boundaries, unigram-tokenizer-supervised boundaries, and fixed-stride pooling in one codebase, at ~40M–100M parameters on text8, wiki40b and a multilingual set. Finding: dynamic pooling beats fixed-stride Hourglass pooling and vanilla byte Transformers on both BPC and wall-clock, but the *supervised* (tokenizer-imitating) boundaries are competitive with or better than the end-to-end learned ones on several languages. That last result is the important one and is often dropped when the paper is cited.
- **Hourglass Transformer** (Nawrot et al., 2021) established that fixed-rate downsample/upsample with shortening factor 2–3 is already a win over flat byte models — the correct control arm.
- **MegaByte** (Yu, Simig, Aghajanyan, Zettlemoyer; NeurIPS 2023) fixed patches of 8 bytes; shows that a fixed rule at $\rho=8$ trains stably to million-byte contexts. No learned boundaries.
- **SpaceByte** (Slagle; NeurIPS 2024) shows a whitespace rule closes most of the byte-vs-BPE gap at ~1B scale. This is the strongest evidence that *learning* boundaries buys little over a trivial rule in English.

**Claimed but not independently ablated.**

- **Byte Latent Transformer** (Pagnoni et al., 2024/2025) uses entropy-based patching from a small auxiliary byte LM, trained to 8B parameters / 4T bytes, and reports parity with a Llama-3-style BPE model at training-FLOP parity plus up to ~50% inference FLOP reduction by growing patch size. The entropy patcher is a *heuristic*, not learned end-to-end, and the ablation isolating patching from the architecture change (local encoder/decoder capacity) is thin.
- **H-Net** (Hwang, Wang, Gu; 2025) learns boundaries via a cosine-similarity routing module with a ratio loss and reports that a 1-stage H-Net matches a BPE Transformer and 2-stage matches at larger data scale, with large gains on Chinese and DNA. Single group, single codebase as of writing.
- **AU-Net** (Nawrot et al., 2025) uses multi-stage autoregressive U-Nets over bytes with rule-based splits.

**Benchmark-number-only.** Most cross-lingual robustness claims for dynamic pooling exist as one table row per language with a single seed. No variance is reported.

## 4. What Is Known

- Fixed-stride pooling at $\rho \in [2,3]$ improves BPC over flat byte models at 40M–150M parameters (Hourglass, 2021; reproduced in Dynamic Token Pooling, ACL 2023).
- Learned boundaries collapse without a rate regulariser. In the ACL 2023 setting the Gumbel predictor drives $\sum_t \pi_t$ to a degenerate extreme unless $\lambda > 0$; $\lambda$ is tuned per corpus.
- Entropy-spike boundaries track whitespace and morpheme edges in English closely enough that whitespace alone recovers most of the benefit at ~1B parameters (SpaceByte, NeurIPS 2024).
- Byte-level hierarchies can be scaled: BLT at 8B parameters / 4T training bytes reaches parity with a tokenizer-based baseline at matched training FLOPs, with mean patch size roughly 4.5–6 bytes.
- Byte models are markedly more robust to character-level noise and to orthographic variation than BPE models at equal size — reported consistently from CANINE (TACL 2022) through BLT.
- Gains are largest where BPE is worst: Chinese, code, and genomic sequence (H-Net, 2025).

## 5. What Is Not Known

- **Empirically open.** Whether *end-to-end learned* boundaries beat a *hand-written* rule (whitespace, entropy threshold) at $\ge 10^{22}$ training FLOPs. Every scaled system that works (BLT, MegaByte, AU-Net) uses a rule; every learned system (DTP, H-Net) is either sub-1B or single-group. The experiment is runnable today.
- **Empirically open.** Whether the pooling advantage survives at long context, where the attention term dominates and $1/\rho$ savings on a quadratic cost behave differently from savings on a linear one.
- **Methodologically blocked.** "Compute-matched" comparison between a variable-$\rho$ model and a fixed-vocabulary model. There is no accepted convention for whether to match analytic FLOPs, realised FLOPs after padding, wall-clock, or FLOPs at the *evaluation* corpus's $\rho$ — and the four give different winners.
- **Theoretically open.** Identifiability of the optimal segmentation. No result characterises whether the BPB-minimising boundary set is unique, or bounds the loss gap between the optimum and a whitespace rule for a given source.
- **Theoretically open.** Whether straight-through Gumbel-sigmoid on boundary variables converges to a stationary point of the true objective under any nontrivial assumptions.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement, not compute.** A dynamic-pooling system changes four things at once relative to a BPE baseline: the input granularity (bytes vs. subwords), the architecture (local encoder + global backbone + upsampler vs. one stack), the output head (256-way vs. 32k-way softmax — a large parameter and FLOP difference), and the compression rate $\rho$. Reported wins are attributed to the boundary predictor, but no published ablation holds the other three fixed. The 256-way head alone frees enough parameters at 1B scale to move BPB.

**Secondary: non-identifiability.** Empirically, many boundary sets with the same $\rho$ give near-identical BPB. The loss surface over $\phi$ is close to flat in the directions that matter, which is why the auxiliary rate loss dominates the learned signal and why supervised boundaries match learned ones.

**Tertiary: absent ground truth.** There is no target segmentation. Morphological gold standards exist for a handful of languages and correlate weakly with what minimises BPB, so "boundary quality" is scored by downstream loss — the same quantity the method is supposed to improve, which makes the diagnosis circular.

## 7. Current Research (as of 2026)

- **Meta FAIR** — BLT line: entropy patching, patch-size scaling as an inference-time knob, and hierarchical byte stacks (AU-Net). *(frontier — verify current status.)*
- **CMU / Cartesia (Gu and collaborators)** — H-Net: end-to-end learned chunking with SSM backbones, multi-stage hierarchies, and emphasis on non-English and DNA. *(frontier — verify.)*
- **Edinburgh / Cambridge (Ponti, Nawrot and collaborators)** — the original dynamic-pooling line, now oriented toward efficiency and multilingual fairness of compression rates.
- Open threads: variable-rate KV-cache and batching kernels for ragged segments; using $\rho$ per language as a fairness metric (tokenizer premium for low-resource scripts); boundary prediction for multimodal byte streams. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does a learned boundary predictor beat a fixed rule at scale, with architecture held constant?

**Scale.** 1.3B total parameters (local encoder 100M, global 1.1B, upsampler 100M), 100B training bytes, ~$8\times10^{21}$ FLOPs. Multilingual: 60% English, 20% code, 10% Chinese, 10% mixed-script.

**Arms — identical architecture, identical parameter count, identical target $\rho^\star = 4.5$, only the boundary source changes:**

1. **Control A (rule):** whitespace + punctuation boundaries, rate-matched by merging/splitting to hit $\rho = 4.5$.
2. **Control B (heuristic):** entropy patching from a 50M byte LM, threshold tuned to $\rho = 4.5$.
3. **Treatment:** Gumbel-sigmoid learned predictor with rate loss $\lambda$ tuned on 1% of budget.
4. **Control C (upper reference):** fixed stride $\rho = 4.5$, no boundary information at all.

**Deciding number.** Held-out **bits-per-byte on the Chinese and code splits, at matched measured wall-clock FLOPs per byte** (profiler-measured, padding included), 3 seeds. Decision rule: the treatment wins only if it beats the *better* of Control A and Control B by $\ge 0.01$ BPB with non-overlapping seed ranges on both splits. Below that threshold the honest conclusion is that learned boundaries are not yet worth their complexity, and the field should report rule-based patching as the default.

**Cost.** ~2,000 A100-days total for four arms × three seeds — within a single well-funded lab's quarterly budget, which is why this counts as empirically open rather than blocked.

## 9. Key References

- **[Foundational]** Nawrot, Tworkowski, Tyrolski, Kaiser, Wu, Szegedy, Michalewski. *Hierarchical Transformers Are More Efficient Language Models.* Findings of NAACL, 2022. — arXiv:2110.13711
- **[Foundational]** Nawrot, Chorowski, Łańcucki, Ponti. *Efficient Transformers with Dynamic Token Pooling.* ACL, 2023. — arXiv:2211.09761
- **[Foundational]** Clark, Garrette, Turc, Wieting. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL, 2022. — arXiv:2103.06874
- **[Foundational]** Tay, Tran, Ruder, Gupta, Chung, Bahri, Qin, Baumgartner, Yu, Metzler. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR, 2022. — arXiv:2106.12672
- **[SOTA]** Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, Li, Zhou, Yu, Weston, Zettlemoyer, Ghosh, Lewis, Holtzman, Iyer. *Byte Latent Transformer: Patches Scale Better Than Tokens.* Meta AI, 2024/2025. — arXiv:2412.09871
- **[SOTA]** Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net).* 2025.
- **[SOTA]** Yu, Simig, Aghajanyan, Zettlemoyer. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS, 2023. — arXiv:2305.07185
- **[Control]** Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* NeurIPS, 2024. — arXiv:2404.14408
- **[Method]** Jang, Gu, Poole. *Categorical Reparameterization with Gumbel-Softmax.* ICLR, 2017. — arXiv:1611.01144
- **[Method]** Bengio, Léonard, Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* 2013. — arXiv:1308.3432
- **[Related]** Wang, Gu et al. *MambaByte: Token-free Selective State Space Model.* COLM, 2024. — arXiv:2401.13660

## 10. Worked Example

Take a 100-byte English fragment and a 100-byte Python fragment, one model, target $\rho^\star = 4.5$.

English (`"the quick brown fox jumps over the lazy dog ..."`): whitespace gives ~19 boundaries per 100 bytes, $\rho \approx 5.3$. An entropy patcher at threshold $\tau$ tuned on English gives ~22, $\rho \approx 4.5$. Agreement between the two boundary sets is high — most entropy spikes land at the byte after a space, where the next character is least predictable.

Python (`"for i in range(len(xs)): xs[i] = xs[i] * 2\n"`): whitespace and punctuation give ~30 boundaries per 100 bytes, $\rho \approx 3.3$. The same entropy threshold $\tau$ fires ~41 times, $\rho \approx 2.4$ — identifier starts, digits and bracket sequences are all high-entropy.

Now count. Suppose the global stack costs 10 GFLOP per step and the local stack 0.6 GFLOP per byte. On English, $C_{\text{byte}} = 0.6 + 10/4.5 = 2.82$ GFLOP. On Python with the same $\tau$, $C_{\text{byte}} = 0.6 + 10/2.4 = 4.77$ GFLOP — **69% more compute per byte on code than on prose, from the same model with no configuration change**.

This is the obstruction, concretely. A paper reporting "BLT-style patching matches BPE at compute parity" fixed parity on a corpus whose mixture determines $\rho$. Shift the evaluation mixture 20 points toward code and the byte model's realised inference cost rises ~20–30% while the BPE baseline's does not — the "parity" claim silently moves. Batch it, and the picture gets worse: a batch of 32 documents with $\rho$ ranging 2.4–5.3 pads to the longest patched sequence, so $\tilde\rho \approx 2.6$, not the 4.5 in the table. The number that decides Section 8 must therefore be profiler-measured wall-clock FLOPs on a *fixed* evaluation mixture, per split — anything else measures the corpus, not the method.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*