---
id: 01-tokenization/learned-end-to-end-segmentation
title: "Learned End-to-End Segmentation That Beats BPE"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learned End-to-End Segmentation That Beats BPE

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/learned-end-to-end-segmentation` · **Status:** open

## 1. Problem Statement

A byte-pair-encoding (BPE) tokenizer is a frozen, corpus-fitted preprocessor: it is trained by a greedy merge count on a static corpus, before the language model exists, and its boundaries never move again. The question is whether a segmentation learned **jointly with the model, by gradient on the model's own loss**, can beat it.

Three variants, different difficulty:

- **Measurement variant.** Define a comparison between a byte-level model with learned segmentation and a BPE model that is not decided by the choice of accounting unit. Bits-per-byte (BPB) is unit-free; perplexity-per-token is not. Open question: which compute-matched protocol is fair when one arm's sequence length is itself a learned quantity.
- **Method variant.** Produce a learned segmenter $S_\theta$ such that, at matched training FLOPs and matched inference FLOPs, the end-to-end system attains strictly lower BPB and higher downstream accuracy than the best BPE baseline, at a scale where the ordering is stable ($\geq 1$B parameters, $\geq 100$B bytes).
- **Theory variant.** Characterize when a learned segmentation *can* beat a frozen one. Is there a class of distributions where every static vocabulary of size $V$ incurs excess loss that an input-dependent segmenter avoids?

Solving it means: a public, independently reproduced result where the learned-segmentation arm wins on compute-matched BPB *and* on a downstream suite, with the segmenter ablated against a non-learned dynamic control (e.g. whitespace patching).

## 2. Formal Setting

**Objects.** A byte string $x \in \Sigma^*$, $\Sigma = \{0,\dots,255\}$, of length $n = |x|$. A segmenter is a map $S_\theta: \Sigma^n \to \{0,1\}^n$ giving boundary indicators $b_t$, inducing $m = \sum_t b_t$ chunks. A backbone $p_\phi$ models the chunk sequence; a decoder returns byte probabilities.

**The objective, as measured.** Bits per byte, the only unit-free loss:

$$\mathrm{BPB}(x) = \frac{-\log_2 p_{\theta,\phi}(x)}{|x|}.$$

For a BPE arm with tokenizer $T$, $-\log_2 p(x) = -\sum_i \log_2 p(t_i \mid t_{<i})$ and the *same* $|x|$ in bytes is the denominator. This is what makes cross-tokenizer comparison possible at all; token-level perplexity is not comparable across arms.

**Compression ratio** $\rho = n/m$ (bytes per chunk), measured on held-out text, not train.

**Compute.** Training FLOPs $C$. For a hierarchical model with byte-level encoder cost $C_{\text{enc}}(n)$ and backbone cost $C_{\text{bb}}(m)$, total $C \approx C_{\text{enc}}(n) + C_{\text{bb}}(n/\rho)$. Because $\rho$ is *learned*, $C$ is a random variable during training — matching FLOPs across arms requires measuring realized $\rho$, not the target.

**The joint problem.**

$$\min_{\theta,\phi}\ \mathbb{E}_{x\sim\mathcal{D}}\big[\mathrm{BPB}(x)\big] \quad \text{s.t.}\quad \mathbb{E}[C(x;\theta)] \le C_0 .$$

The obstruction is structural: $b_t$ is discrete, so $\partial \mathrm{BPB}/\partial\theta$ does not exist through the segmentation. Known relaxations: Gumbel/straight-through (Dynamic Token Pooling), an auxiliary entropy predictor that is *not* differentiated through (BLT), a smoothing/routing module with a ratio-targeting auxiliary loss (H-Net).

**Assumptions, and which are violated.**
1. *Held-out text is drawn from the same $\mathcal{D}$ as training.* Violated for code, non-Latin scripts, and math, where BPE compression ratios differ by 2–4×.
2. *BPB is monotone in downstream utility.* Known violated — Schmidt et al. (EMNLP 2024) show compression and task accuracy dissociate.
3. *FLOP matching implies wall-clock matching.* Violated: variable-length chunking gives ragged batches; realized throughput can be 20–40% below the FLOP-implied figure.
4. *The segmenter's $\rho$ is stable across the run.* Violated without an auxiliary ratio loss; segmenters collapse to all-boundaries or no-boundaries.

## 3. State of the Art

**Empirical SOTA (claimed, partially ablated).**
- **Byte Latent Transformer** (Pagnoni et al., Meta; ACL 2025). Entropy-based dynamic patching: a small byte LM emits $H(x_t\mid x_{<t})$ and a boundary is placed on entropy spikes. Trained to 8B params / 4T bytes; reported FLOP-matched parity with Llama 3 8B plus robustness gains on noisy and character-level tasks. **Established:** the scaling study is FLOP-controlled and public. **Claimed but unablated:** that the entropy segmenter beats a fixed-stride or whitespace control at 8B — the strong control arm is run only at smaller scale. Note also the segmenter is trained by its own next-byte loss, not by the end model's loss, so BLT is *dynamic* but not *end-to-end learned*.
- **H-Net** (Hwang, Wang, Gu; 2025). A routing module scores boundaries from cosine similarity of adjacent encoder states, with a straight-through estimator and a ratio-targeting auxiliary loss — genuinely end-to-end. Reported: 1-stage H-Net matches a BPE Transformer at matched compute; 2-stage exceeds it, with larger margins on Chinese, code, and DNA. **Benchmark-number status:** results are at ~1.3B params on order-$10^{11}$ bytes, from one group; no independent reproduction at that scale as of 2026.
- **SpaceByte** (Slagle, 2024) is the control that matters: patching on whitespace bytes alone — zero learned parameters — recovers most of MegaByte-to-BPE gap at ~1B scale. Any learned segmenter must beat this, and most published work does not report it.

**Theory SOTA.** Whittington, Bachmann & Pimentel, *Tokenisation is NP-Complete* (ACL 2025): finding the vocabulary minimizing tokenized length is NP-complete in both direct and bottom-up formulations. Kozma & Voderholzer (2024): BPE's greedy merges are within a constant factor of optimal compression (approximation ratio bounded below by $0.333$, above by $0.625$). Rajaraman, Jiao & Ramchandran (2024): for $k$-th order Markov sources, a transformer *without* tokenization is driven to near-unigram behaviour, while a suitable tokenizer makes near-optimal cross-entropy reachable — an existence result for why tokenization helps, not a bound on learned segmentation.

## 4. What Is Known

- **BPE is not compression-optimal, but is close.** $0.333 \le$ approximation ratio $\le 0.625$ against the optimal merge sequence (Kozma & Voderholzer 2024). Optimal tokenization is NP-complete (ACL 2025).
- **Compression does not determine downstream quality.** Schmidt et al. (EMNLP 2024) train 350M-parameter models over many tokenizers: corpus compression correlates with, but does not predict, task accuracy; some worse-compressing tokenizers win downstream.
- **Intrinsic tokenizer metrics predict weakly.** Zouhar et al. (ACL 2023) propose Rényi efficiency and report correlation with BLEU on MT; the correlation is not stable across the settings later authors tried.
- **Byte-level models need hierarchy to be affordable.** MegaByte (Yu et al., 2023) with fixed patch size $P{=}8$ and MambaByte (Wang et al., NeurIPS 2024) both show byte modeling is viable at ~1B scale but only under sequence compression.
- **A parameter-free dynamic rule is a strong baseline.** SpaceByte (Slagle, NeurIPS 2024): whitespace-aligned patches, ~1B params, close most of the byte-vs-BPE gap.
- **Vocabulary size is a scaling axis.** Tao et al. (2024) and Dagan et al. (2024) find compute-optimal vocabulary grows with model size; most 7B models are under-vocabularied by roughly 2–3× relative to the fitted optimum. This is a confound: a "learned segmenter beats BPE" result may only be beating a badly sized BPE.

## 5. What Is Not Known

- **Empirically open.** Whether any end-to-end learned segmenter beats a *well-tuned, correctly-sized* BPE at $\geq 7$B params and $\geq 1$T bytes, under both FLOP-matching and wall-clock-matching, with a whitespace-patching control in the same table. Every ingredient is runnable today; the run costs $\sim 10^{22}$–$10^{23}$ FLOPs per arm and nobody has published the full 4-arm grid.
- **Theoretically open.** Whether there exists a distribution class where every static vocabulary of size $V$ has excess cross-entropy $\Omega(f(V))$ that an input-dependent segmenter avoids at equal compute. No separation theorem either way.
- **Methodologically blocked.** What "matched compute" means when $\rho$ is learned and drifts, and how to compare arms whose inference cost is input-dependent. Also blocked: there is no ground-truth segmentation, so segmentation quality can only be scored through end-task loss — which confounds segmenter and backbone.

## 6. Why It Is Hard

The obstruction is **non-identifiability under a confounded measurement**, not compute alone. A hierarchical byte model differs from a BPE model in at least four coupled ways: the segmentation rule, the extra encoder/decoder parameters, the effective sequence length, and the vocabulary-size confound of the baseline. A BPB win is attributable to any of them. The field's usual ablation — swap the learned segmenter for a fixed stride — changes $\rho$ and therefore compute, so it is not a clean swap either.

Second obstruction: the gradient does not exist. Every reported end-to-end result rests on a biased estimator (straight-through) plus an auxiliary ratio loss that *pins* $\rho$ near a hand-chosen target. If $\rho$ is pinned, the segmenter is learning boundary *placement* only, not compression rate — so the headline claim "learned end-to-end" is weaker than it sounds, and no paper has ablated the ratio loss away and shown stability.

## 7. Current Research (as of 2026)

- **Meta FAIR** — BLT line: entropy patching at scale, patch-size scaling laws. *(frontier — verify current status of successor models.)*
- **CMU / Cartesia (Gu and collaborators)** — H-Net and multi-stage dynamic chunking; extensions to DNA and multilingual byte streams *(frontier — verify).*
- **Stanford / ETH / Cambridge (Kallini, Pimentel, Cotterell and co-authors)** — MrT5 (ICLR 2025), token deletion inside a byte encoder; and the tokenization-complexity line.
- **Tokenizer-free multilingual work** — motivated by the measured 2–5× token-cost penalty non-Latin scripts pay under English-fit BPE (Ahia et al., EMNLP 2023; Petrov et al., NeurIPS 2023).

## 8. Concrete Next Experiment

**Scale.** 1.3B non-embedding params, 100B bytes of FineWeb-Edu, four arms, FLOP-matched to within 2% on *realized* compute (measure $\rho$ every 1k steps and log actual FLOPs).

**Arms.**
1. **Control A (the one everyone skips):** whitespace patching, SpaceByte-style, zero learned segmentation parameters.
2. **Control B:** BPE at the compute-optimal vocabulary for 1.3B (fit $V$ per Tao et al., roughly 48k–64k), not the default 32k.
3. **Treatment:** H-Net-style learned router, straight-through, ratio loss targeting the *realized* $\rho$ of Arm A so compression is held fixed and only boundary placement varies.
4. **Treatment−:** same as 3 with the ratio loss removed.

**Deciding number.** $\Delta\mathrm{BPB} = \mathrm{BPB}(\text{Arm 3}) - \mathrm{BPB}(\text{Arm 1})$ on held-out FineWeb-Edu, plus the same on a code and a Chinese held-out set. The claim "learned segmentation beats BPE" survives only if $\Delta\mathrm{BPB} < -0.005$ bits/byte on English with 3 seeds and non-overlapping seed ranges. If $|\Delta\mathrm{BPB}| < 0.005$, the honest conclusion is that the win in the literature is from byte-level hierarchy and vocabulary sizing, not from learning where the boundaries go. Arm 4's realized $\rho$ trajectory answers separately whether the segmenter learns compression or only placement. Cost: roughly $4 \times 10^{21}$ FLOPs total — days on 64 H100s.

## 9. Key References

- **[Foundational]** Sennrich, Haddow & Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[Foundational]** Kudo. *Subword Regularization: Improving NMT Models with Multiple Subword Candidates.* ACL 2018. — arXiv:1804.10959
- **[SOTA]** Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* ACL 2025. — arXiv:2412.09871
- **[SOTA]** Hwang, Wang & Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* 2025. — arXiv:2507.07955
- **[SOTA]** Yu et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS 2023. — arXiv:2305.07185
- **[Baseline]** Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* NeurIPS 2024. — arXiv:2404.14408
- **[Method]** Nawrot, Chorowski, Łańcucki & Ponti. *Efficient Transformers with Dynamic Token Pooling.* ACL 2023. — arXiv:2211.09761
- **[Method]** Tay et al. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR 2022. — arXiv:2106.12672
- **[Method]** Xue et al. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL 2022. — arXiv:2105.13626
- **[Theory]** Whittington, Bachmann & Pimentel. *Tokenisation is NP-Complete.* ACL 2025. — arXiv:2412.15210
- **[Theory]** Kozma & Voderholzer. *Theoretical Analysis of Byte-Pair Encoding.* 2024. — arXiv:2411.08671
- **[Theory]** Rajaraman, Jiao & Ramchandran. *Toward a Theory of Tokenization in LLMs.* 2024. — arXiv:2404.08335
- **[Evaluation]** Schmidt, Reddy, Zhang, Alameddine, Uzan, Pinter & Tanner. *Tokenization Is More Than Compression.* EMNLP 2024. — arXiv:2402.18376
- **[Evaluation]** Zouhar, Meister, Gastaldi, Du, Sachan & Cotterell. *Tokenization and the Noiseless Channel.* ACL 2023. — arXiv:2306.16842
- **[Survey/Context]** Mielke et al. *Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP.* 2021. — arXiv:2112.10508

## 10. Worked Example

Take the string `unbelievability` (15 bytes).

- **GPT-2 BPE** segments it `un|bel|iev|ability` — 4 tokens, $\rho = 3.75$ bytes/token.
- **A morphological ground truth** would be `un|believ|abil|ity` — also 4 pieces, same $\rho$.

Now the point. Suppose both models assign the string the same total $-\log_2 p = 41.2$ bits. Then both have $\mathrm{BPB} = 41.2/15 = 2.75$. **The segmentation difference is invisible in the objective.** Morphological correctness is not what the loss scores; the loss scores only the total code length. So a learned segmenter that finds `believ` gets no credit unless that boundary changes bits elsewhere in the corpus.

Where a difference *could* show up: a rare inflection. `unbelievabilities` under GPT-2 BPE is `un|bel|iev|abilities` (4 tokens); the model must generalize from `abilities` as an atom. A morphemic segmenter gives `un|believ|abil|ities`, sharing `ities` with hundreds of other words. Assume the morphemic arm saves 1.5 bits on this word. English text with such forms at rate $\approx 3\times10^{-4}$ per byte gives corpus-level $\Delta\mathrm{BPB} \approx 4.5\times10^{-4}$ bits/byte.

That is the obstruction, in one number: **$4.5\times10^{-4}$ is an order of magnitude below the $\sim 5\times10^{-3}$ seed-to-seed BPB spread at 1.3B parameters.** The linguistically motivated gain is buried in run-to-run noise. Any reported win larger than that is therefore *not* coming from better morphology — it is coming from the byte-level encoder, the extra parameters, the vocabulary sizing of the baseline, or the sequence-length/compute accounting. Until an experiment separates those four, "learned segmentation beats BPE" is a claim about architecture, not about segmentation.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*