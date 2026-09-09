---
id: 09-model-design/learned-patch-boundaries
title: "Optimal Patch or Chunk Boundary Learning"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Patch or Chunk Boundary Learning

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/learned-patch-boundaries` · **Status:** open

## 1. Problem Statement

Given a raw sequence (bytes, characters, pixels, samples), a model that processes it hierarchically must first decide **where to cut it** into patches/chunks that the expensive backbone sees as single positions. The question: can boundary placement be *learned end-to-end from the modeling loss*, and does a learned segmentation beat a fixed or heuristic one at matched compute?

- **Input:** sequence $x_{1:T}$ over alphabet $\mathcal{V}$ (e.g. 256 bytes), compute budget $C$.
- **Output:** a boundary set $B \subseteq \{1,\dots,T\}$, $|B| = M$, inducing chunks, plus model parameters.
- **Objective:** minimize bits-per-byte $\mathrm{BPB}$ of the full generative model at fixed inference FLOPs/byte.
- **Solved** = a boundary rule that, at matched FLOPs and matched mean chunk length, beats every fixed/heuristic rule by more than seed noise, on ≥3 modalities.

Three variants, different difficulty:

- **Measurement:** is there a segmentation-quality statistic that predicts downstream loss? Currently no reliable one — compression rate and Rényi efficiency are weak proxies.
- **Method:** train the boundary predictor through a discrete decision without collapse to trivial segmentations. Partly achieved (H-Net), at the cost of an auxiliary ratio target.
- **Theory:** characterize the loss-optimal segmentation for a given source class. Essentially untouched beyond Markov-source results on tokenization.

## 2. Formal Setting

Boundary indicators $b_t \in \{0,1\}$, $b_t=1$ marks the start of a chunk. Compression ratio
$$ s \;=\; T \big/ \textstyle\sum_t b_t $$
measured as mean bytes per chunk over a held-out corpus (not the training average — they differ by 3–8% on code).

The model factorizes as encoder $E_{\theta_e}$ (per byte), backbone $G_{\theta_g}$ (per chunk), decoder $D_{\theta_d}$ (per byte). Forward FLOPs per byte, measured by counting matmul FLOPs in a profiler rather than analytically:
$$ F(s) \;\approx\; 2\!\left(N_e + N_d + \frac{N_g}{s}\right) + \text{attn terms} $$

Loss is reported **only** in bits per *raw byte*, never per token:
$$ \mathrm{BPB} = -\frac{1}{T\ln 2}\sum_{t} \ln p_\theta(x_t \mid x_{<t}) $$
This is the one quantity comparable across tokenizers.

The boundary policy is $q_\phi(b_t \mid x_{\le t})$. Because $b$ is discrete, training uses one of: entropy thresholding on an auxiliary byte LM ($b_t = 1$ iff $H(p_{\text{aux}}(\cdot\mid x_{<t})) > \tau$, or the "approximate monotonicity" rule $H_t - H_{t-1} > \tau'$); Gumbel-softmax; straight-through with a routing score; or a cosine-similarity router with a smoothing interpolation.

**Assumptions, and their status:**

1. *Chunks are contiguous and non-overlapping.* Holds by construction; excludes overlapping/multi-scale allocations.
2. *A single $s$ is optimal per corpus.* **Violated** — optimal $s$ differs sharply between English prose, code, and DNA within one training mixture.
3. *The backbone cost dominates, so raising $s$ is free accuracy-per-FLOP.* **Violated** at $s \gtrsim 6$, where $N_e + N_d$ dominates and further compression buys nothing.
4. *Held-out BPB ranks methods the same way downstream tasks do.* **Violated in the tokenizer literature** (Schmidt et al. 2024) — compression and downstream accuracy decouple.
5. *The auxiliary entropy model is a valid stand-in for the trained model's own surprisal.* Untested; the auxiliary model is small and frozen while the main model's surprisal profile shifts during training.

## 3. State of the Art

**Empirical SOTA.**

- **BLT** (Pagnoni et al., Meta, ACL 2025): entropy-based patching from a small byte LM; trained to 8B parameters on 4T bytes. *Established:* byte-level models can be trained at 8B scale with FLOP-controlled scaling curves. *Claimed but weakly ablated:* that entropy patching beats other rules **at matched $s$** — the headline comparisons vary $s$ (4.5 vs ~6+) and model shape together.
- **H-Net** (Hwang, Wang & Gu, 2025): a learned routing module with a smoothing layer, giving gradients to boundaries, plus an auxiliary loss pinning $s$ to a target. *Established:* end-to-end boundary learning trains stably in a multi-stage hierarchy. *Claimed:* outperforming BPE Transformers at matched compute and a reported ~3.6× data-efficiency gain on DNA; these are benchmark numbers, not independently reproduced as of 2026.
- **SpaceByte** (Slagle, NeurIPS 2024): boundaries at space-like bytes — a *rule with no learning at all* — reaches roughly subword-Transformer perplexity in FLOP-controlled comparison, and clearly beats MegaByte's fixed patches. This is the control arm most learned methods fail to beat convincingly.
- **Dynamic Token Pooling** (Nawrot et al., ACL 2023): compared entropy, unigram, Gumbel and end-to-end boundaries directly. Whitespace-derived boundaries were competitive with or better than learned ones on text.

**Theory SOTA.** Rajaraman, Jiao & Ramchandran (2024) prove that on $k$-th order Markov sources, transformers trained on raw symbols drift toward near-unigram predictions, while a *unigram* model over a BPE-style dictionary attains cross-entropy within $(1+\varepsilon)$ of the source entropy rate as dictionary size grows. This justifies chunking in general; it says nothing about *which* boundaries.

## 4. What Is Known

- Fixed-size patching is strictly worse than content-aware patching. MegaByte (1.2B params, patch size 8) is beaten by SpaceByte at matched FLOPs (Slagle 2024, ~1B scale).
- Mean patch size has a broad optimum. In BLT's scaling study (up to 8B params, 4T bytes), $s \in [4.5, 8]$ all train; the inference-FLOP advantage over a Llama-3-style tokenizer baseline is reported up to ~50% at the largest $s$, which is an artifact of the FLOP formula in §2 as much as of segmentation quality.
- Compression is not the objective. Schmidt et al. (EMNLP 2024) trained many LMs at the ~350M class across vocabularies and found the best-compressing tokenizer (PathPiece) was not the best downstream; Zouhar et al. (ACL 2023) found Rényi efficiency a better-than-compression but still partial predictor of BLEU.
- BPE segmentation is suboptimal for pretraining relative to unigram-LM segmentation on downstream tasks (Bostrom & Durrett, Findings of EMNLP 2020, at BERT-base scale) — established, reproduced.
- Boundary learning collapses without a constraint. Every reported end-to-end system (Charformer's GBST, dynamic pooling, H-Net) uses either a soft mixture over fixed block sizes or an explicit ratio-regularizer; unconstrained routers drive $s \to 1$ or $s \to \infty$.
- In vision the analogue is monotone and boring: ViT-B/8 beats ViT-B/16 by roughly 1–3 ImageNet points at ~4× FLOPs; FlexiViT (CVPR 2023) shows one model can serve many patch sizes, but no learned *content-adaptive* image patcher has cleanly beaten uniform grids at matched compute.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the loss-optimal segmentation for any nontrivial source class (hidden semi-Markov, context-free). No proof that the entropy-threshold rule is optimal, near-optimal, or arbitrarily bad for a stated source family. No identifiability result: it is unproven whether the boundary policy is even determined by the training loss, or whether a large equivalence class of segmentations is loss-equivalent.
- **Empirically open.** The matched-$s$, matched-FLOP, matched-data head-to-head of {fixed, whitespace, entropy, learned-router} at ≥1B params on a multi-modality mixture has not been published. Every existing comparison moves at least two variables.
- **Methodologically blocked.** "Segmentation quality" has no accepted intrinsic measure. Compression rate, Rényi efficiency, and boundary-F1 against morphological gold standards each fail to predict downstream loss. Until a proxy exists, every boundary rule must be validated by a full pretraining run.

## 6. Why It Is Hard

The specific obstruction is a **compute confound entangled with non-identifiability**.

Changing the boundary rule changes $s$, which changes FLOPs/byte through $N_g/s$, which changes effective model capacity per byte. Holding FLOPs fixed requires resizing the backbone, which changes the scaling-law position. So the measured BPB gap between two patchers is a sum of (i) segmentation quality, (ii) compute per byte, and (iii) parameter-count effects — and (ii) alone is typically an order of magnitude larger than (i) (see §10).

Layered on top: the loss appears to be nearly flat across a wide family of segmentations. A router can be given a strong training signal and still land anywhere in that flat region, so two runs with different random seeds produce different boundary policies at the same loss. There is no ground truth to appeal to — morphemes are not the right target for a byte LM, and no one has shown they should be.

## 7. Current Research (as of 2026)

- **Meta AI (FAIR)** — byte-level and latent-patch scaling following BLT; extending entropy patching to multimodal streams. *(frontier — verify)*
- **CMU / Cartesia (Gu, Hwang, Wang)** — hierarchical dynamic chunking, multi-stage H-Nets, and pushing routing into SSM backbones.
- **Tokenizer-free evaluation** — groups around Pinter, Cotterell and Zouhar working on intrinsic tokenizer metrics; the direct precursor to a usable segmentation-quality proxy.
- **Vision/video adaptive tokenization** — recurrent/elastic allocation of a variable token budget per image (Duggal et al.; ElasticTok line). *(frontier — verify)* Whether these beat uniform patches at matched FLOPs is unresolved.
- **Theory of tokenization** — Berkeley (Rajaraman, Jiao, Ramchandran) and follow-ons on BPE optimality for Markov sources.

## 8. Concrete Next Experiment

**The matched-$s$ patcher bake-off.**

- **Scale:** 1.0B total parameters (local encoder 80M + decoder 80M + backbone ~840M), 100B bytes, single seed sweep of 3 seeds. ~$2\times10^{21}$ FLOPs per arm; 5 arms × 3 seeds ≈ 3k H100-hours. This is the smallest scale at which byte-model scaling curves have separated in prior work.
- **Arms**, all *calibrated to the same held-out mean patch size $s = 5.0 \pm 0.05$ bytes*, with the backbone width adjusted so profiler-measured FLOPs/byte match within 1%:
  1. fixed stride 5 (MegaByte-style),
  2. whitespace/space-like bytes with forced splits to hit $s=5$ (SpaceByte-style),
  3. entropy threshold from a 50M byte LM (BLT-style),
  4. learned router with ratio loss targeting $s=5$ (H-Net-style),
  5. **oracle upper bound:** boundaries chosen offline by dynamic programming to minimize the *frozen* arm-3 model's BPB, subject to $s=5$.
- **Control arm:** arm 2. Whitespace is the strongest zero-learning rule and the one every learned method must beat.
- **Deciding number:** $\Delta\mathrm{BPB} = \mathrm{BPB}(\text{arm 2}) - \mathrm{BPB}(\text{arm 4})$ on a held-out mixture (50% web text, 25% code, 25% non-space-delimited: Chinese + DNA). Seed noise at this scale is roughly $\pm0.003$ BPB. **If $\Delta\mathrm{BPB} > 0.01$, boundary learning is real.** If $|\Delta\mathrm{BPB}| < 0.005$ while the oracle arm 5 gains $>0.03$, the problem is confirmed as an *optimization/identifiability* failure, not an absence of headroom — a different and more valuable result.

## 9. Key References

- **[Foundational]** Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[Foundational]** Clark, Garrette, Turc, Wieting. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL 2022. — arXiv:2103.06874
- **[Foundational]** Yu, Simig, Flaherty, Aghajanyan, Zettlemoyer, Lewis. *MegaByte: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS 2023. — arXiv:2305.07185
- **[SOTA]** Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* ACL 2025. — arXiv:2412.09871
- **[SOTA]** Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* 2025. — arXiv:2507.07955
- **[SOTA]** Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* NeurIPS 2024. — arXiv:2404.14408
- **[Method]** Nawrot, Chorowski, Łańcucki, Ponti. *Efficient Transformers with Dynamic Token Pooling.* ACL 2023. — arXiv:2211.09761
- **[Method]** Tay et al. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR 2022. — arXiv:2106.12672
- **[Theory]** Rajaraman, Jiao, Ramchandran. *Toward a Theory of Tokenization in LLMs.* 2024. — arXiv:2404.08335
- **[Measurement]** Zouhar, Meister, Gastaldi, Du, Sachan, Cotterell. *Tokenization and the Noiseless Channel.* ACL 2023. — arXiv:2306.16842
- **[Measurement]** Schmidt, Reddy, Zhang, Alameddine, Uzan, Pinter, Tanner. *Tokenization Is More Than Compression.* EMNLP 2024. — arXiv:2402.18376
- **[Measurement]** Bostrom, Durrett. *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP 2020. — arXiv:2004.03720
- **[Vision]** Beyer et al. *FlexiViT: One Model for All Patch Sizes.* CVPR 2023. — arXiv:2212.08013
- **[Related]** Wang et al. *MambaByte: Token-free Selective State Space Model.* COLM 2024. — arXiv:2401.13660

## 10. Worked Example

Take a BLT-shaped model: local encoder + decoder $N_e+N_d = 160$M, backbone $N_g = 1000$M. FLOPs per byte (forward, matmul only, ignoring attention):

| patcher | $s$ (bytes/patch) | $F(s) = 2(160\text{M} + 1000\text{M}/s)$ | measured BPB |
|---|---|---|---|
| entropy, $\tau$ low | 4.5 | 764 MFLOP/byte | 0.850 |
| entropy, $\tau$ high | 6.0 | 653 MFLOP/byte | 0.845 |

Read naively: the higher threshold gives better boundaries *and* 15% fewer FLOPs. That is the confound.

Now correct it. To spend the arm-1 budget at $s=6$, grow the backbone to $N_g' = (764/2 - 160)\times 6 = 1332$M — a $1.33\times$ parameter increase. Using a Chinchilla-style exponent $L_{\text{red}} \propto N^{-0.34}$ on the reducible part of the loss, and taking the irreducible floor at $\approx 0.50$ BPB so $L_{\text{red}} = 0.345$:

$$ \Delta L_{\text{red}} = 0.345\left(1 - 1.33^{-0.34}\right) \approx 0.345 \times 0.0927 \approx 0.032 \ \text{BPB} $$

So a FLOP-matched $s{=}6$ arm should sit near $0.845 - 0.032 = 0.813$ BPB from *capacity alone*. The observed 0.845 is 0.032 BPB **worse** than the compute-matched prediction. The reported "0.005 BPB win for better boundaries" inverts sign once the confound is removed: the higher threshold is buying FLOPs, not better cuts, and is paying for them in segmentation quality.

The obstruction is exactly this: the capacity effect (0.032) is $6\times$ the claimed boundary effect (0.005) and $10\times$ the seed noise (0.003). Any patcher comparison that does not pin $s$ and profiler-measured FLOPs simultaneously is measuring compute allocation and calling it boundary learning.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*