---
id: 24-multimodal/continuous-versus-discrete-visual-tokens
title: "Continuous versus Discrete Visual Tokenization"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continuous versus Discrete Visual Tokenization

> **Topic:** Multimodal Models · **ID:** `24-multimodal/continuous-versus-discrete-visual-tokens` · **Status:** open

## 1. Problem Statement

A multimodal model must convert an image into a sequence the transformer can consume. Two families exist:

- **Discrete:** an encoder maps the image to indices in a finite codebook (VQ-VAE, VQGAN, LFQ, FSQ). The image becomes literal tokens, trainable with the same cross-entropy head as text.
- **Continuous:** the encoder emits real-valued vectors fed directly into the residual stream (LLaVA-style understanding) or modelled with a diffusion/flow head (MAR, Transfusion, Fluid).

**The question:** at a fixed compute budget $C$ and fixed data, does either family dominate the other on joint understanding *and* generation, or is the observed gap an artifact of unmatched tokenizers, heads, and hyperparameters?

Three variants, of different difficulty:

- **Measurement:** define a comparison in which "discrete vs continuous" is the only varying factor. Currently unsolved — bits per image, tokens per image, and FLOPs per image cannot all be held fixed at once.
- **Method:** build a tokenizer that matches continuous reconstruction fidelity while keeping a categorical likelihood. Partially achieved (LFQ, FSQ) at reconstruction; unproven end-to-end.
- **Theory:** bound the loss in achievable rate–distortion–perception from constraining the latent to a finite alphabet at fixed sequence length. Open.

A solution is a scaling-law statement: coefficients $(a, \alpha)$ in $L(C) = L_\infty + aC^{-\alpha}$ for each family, measured on a shared downstream metric, with a crossover point or a proof of no crossover.

## 2. Formal Setting

Image $x \in \mathbb{R}^{H \times W \times 3}$. Encoder $E_\theta$ produces $n$ latents of width $d$. Two quantization regimes:

$$z^{\text{disc}} = \big(q(E_\theta(x)_i)\big)_{i=1}^n \in \{1,\dots,K\}^n, \qquad z^{\text{cont}} = E_\theta(x) \in \mathbb{R}^{n\times d}$$

**Measured quantities:**

- **Rate** $R$ — bits per image. Discrete: $R = n\log_2 K$ upper bound; the operational rate is $n \cdot H(q(E_\theta(X)))$, measured as the empirical entropy of code indices over $\ge 10^4$ held-out images. Continuous: $R$ is not defined without an explicit quantizer; the honest proxy is the Gaussian-VAE rate $R = \mathbb{E}_x\, D_{\mathrm{KL}}(q_\phi(z|x)\,\|\,p(z))/\ln 2$.
- **Distortion** $D = \mathbb{E}\|x - \hat{x}\|_2^2$ (or LPIPS), measured on ImageNet-val at native resolution.
- **Perception** $P = d(p_X, p_{\hat X})$, measured as rFID over 50k reconstructions.
- **Codebook utilization** $U = |\{k : \hat p(k) > 0\}|/K$, and perplexity $2^{H(\hat p)}$, over the same $10^4$ images.
- **Compute** $C$ — training FLOPs, counted as $6ND$ for the backbone *plus* tokenizer and diffusion-head FLOPs. Most published comparisons omit the head.

The Blau–Michaeli theorem (ICML 2019) gives, for a fixed rate, a strict trade-off: forcing $P \to 0$ raises the minimum achievable $D$. Discrete and continuous tokenizers sit at different points on this surface, so a single-metric comparison is under-determined.

**Assumptions, and which are violated:**

1. *The tokenizer is fixed and equally optimized in both arms.* Violated: discrete tokenizers have absorbed years more tuning in the generation literature; continuous encoders have absorbed more in the understanding literature (CLIP/SigLIP).
2. *Matched $n$ implies matched information.* Violated: at $n = 256$, $K = 16384$ carries $\le 3584$ bits; a 16-dim fp16 continuous latent carries $\le 65536$ bits.
3. *Downstream metrics are monotone in rFID.* Violated — MAGVIT-v2 and others report tokenizers with worse rFID but better generation FID after the prior is trained.
4. *Codes are i.i.d. across positions.* Badly violated; spatial redundancy is why $H \ll \log_2 K$ in practice.

## 3. State of the Art

**Established (ablated, reproduced):**

- **FSQ** (Mentzer et al., ICLR 2024) removes the codebook, auxiliary losses and EMA, and matches VQ reconstruction and generation once $K \gtrsim 2^{10}$, while keeping near-100% utilization at $K = 2^{16}$ where VQ collapses. Independently reproduced.
- **Lookup-free quantization** (MAGVIT-v2, Yu et al., ICLR 2024) scales the codebook to $K = 2^{18}$ by dropping the embedding lookup; the paper's central claim — that a language-model prior with a good enough tokenizer beats diffusion on ImageNet and Kinetics — is supported by ablation over codebook size.
- **Continuous AR without VQ** (MAR, Li et al., NeurIPS 2024): a per-token diffusion loss replaces cross-entropy, reaching FID 1.55 on ImageNet 256×256. The ablation isolating the loss (same backbone, VQ vs continuous) is in the paper.

**Claimed but unablated, or benchmark-only:**

- **Transfusion** (Zhou et al., 2024) reports that Chameleon-style discrete training needs many times more compute to reach the same FID. This is a small number of single-seed runs on one data mixture, with the diffusion head's FLOPs not fully charged to the continuous arm. Treat as suggestive, not established.
- **Chameleon** (Meta, 2024) and **Emu3** (BAAI, 2024) show a single discrete vocabulary suffices for competitive understanding *and* generation — but neither runs the matched continuous control.
- **Janus** (Wu et al., 2024) claims decoupling the encoder for understanding from the one for generation beats sharing either. Benchmark numbers only; no compute-matched shared-encoder control at scale.

## 4. What Is Known

- Naive VQ collapses: VQGAN with $K = 16384$ typically uses a small fraction of codes; FSQ at $K=2^{16}$ holds utilization near 100% (ImageNet, $256^2$).
- Reconstruction favours continuous latents at equal $n$: a KL-regularized f8 VAE reaches rFID below 1 on ImageNet-val 256×256; f16 VQGAN with $K=16384$ is around 4–5. The gap shrinks, but does not close, with LFQ/FSQ at large $K$.
- Generation does **not** track that gap. Discrete multi-scale AR (VAR, Tian et al., NeurIPS 2024) reaches FID $\approx 1.7$–1.8 on ImageNet 256×256; continuous MAR reaches 1.55; plain raster discrete AR (LlamaGen) 2.18. The spread within each family exceeds the spread between families.
- Sequence length dominates both: TiTok (Yu et al., NeurIPS 2024) reconstructs and generates from 32 discrete tokens per image, showing the $n$ axis is worth more than the alphabet axis at fixed budget.
- For understanding, continuous CLIP/SigLIP features are the operating point of every top open VLM; Cambrian-1 (Tong et al., NeurIPS 2024) shows vision-centric benchmarks separate encoders that MMBench-style suites do not.

## 5. What Is Not Known

- **Theoretically open:** no bound on the excess distortion, at fixed $n$ and fixed decoder capacity, from restricting to a $K$-ary alphabet versus $\mathbb{R}^d$ — i.e. no rate–distortion–perception separation theorem for finite-alphabet visual latents under a learned prior.
- **Empirically open:** no compute-matched scaling-law sweep ($10^{20}$–$10^{22}$ FLOPs, $\ge 4$ compute points per arm, $\ge 2$ seeds) comparing families on joint understanding + generation. Runnable today; nobody has published it.
- **Empirically open:** whether discrete visual tokens interfere with text loss when sharing a softmax (the "modality competition" claim), measured as text-only perplexity delta at matched vision FLOPs.
- **Methodologically blocked:** the comparison itself. There is no agreed way to equate a categorical token with a continuous latent — token-matched, bit-matched, and FLOP-matched protocols give different winners, and no paper reports all three.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**. Switching families changes at least four things simultaneously: the encoder objective, the output head (softmax vs diffusion/regression), the loss scale, and the effective bits per position. Any observed gap is attributable to any of them, and the published ablations vary one while leaving the others tied to the arm. Worse, the two arms have different optimal hyperparameters — learning rate, head width, sampling temperature vs. CFG scale — so a fair comparison requires a per-arm hyperparameter sweep, multiplying an already $10^{21}$-FLOP experiment by 5–10×. That cost, not conceptual difficulty, is why the decisive run does not exist.

A second obstruction: the headline metric. FID rewards distribution match, not fidelity to the conditioning; it is known to move under sampler and CFG changes that do not alter what a human would call quality. An FID-decided comparison risks measuring sampler tuning rather than tokenizer class.

## 7. Current Research (as of 2026)

- **Unified any-to-any backbones** — Meta (Chameleon/Transfusion line), BAAI (Emu3), DeepSeek (Janus line) — converging on hybrid designs: continuous features for understanding, discrete or diffusion-headed tokens for generation. *(frontier — verify)*
- **Tokenizer compression:** 1-D and few-token tokenizers after TiTok; the working hypothesis is that $n$, not $K$, is the binding constraint.
- **Quantizer-free discretization:** FSQ, LFQ, and residual/binary variants aiming at continuous-level rFID with a categorical likelihood.
- **Semantic-aligned tokenizers** distilling CLIP/DINO features into the codebook so one vocabulary serves both tasks *(frontier — verify)*.
- Largely absent: any group publishing the compute-matched scaling sweep. The incentive is to ship a model, not to settle the axis.

## 8. Concrete Next Experiment

**Scale.** Four compute points per arm: $3\times10^{19}$, $10^{20}$, $3\times10^{20}$, $10^{21}$ FLOPs (roughly 0.3B–3B params, Chinchilla-optimal tokens), 2 seeds each. Total ~$1.1\times10^{22}$ FLOPs including a 5-point learning-rate sweep at the smallest scale only. Feasible on ~256 H100s in under three weeks.

**Arms.** Identical transformer backbone, identical text corpus, identical image corpus (e.g. 400M image–text pairs), identical $n = 256$ visual positions per image.
- *A (discrete):* FSQ tokenizer, $K = 2^{16}$, softmax head shared with text.
- *B (continuous):* same encoder architecture trained with KL regularization to the same rFID target, per-token diffusion head (MAR-style), head FLOPs charged to $C$.
- *Control arm C:* discrete tokenizer, but the same diffusion head applied to the codebook embeddings — this isolates *head* from *alphabet*, the confound no published comparison removes.

**The deciding number.** Fit $L(C) = L_\infty + aC^{-\alpha}$ to a single composite: the sum of (i) held-out text cross-entropy and (ii) generation NLL-proxy converted to bits/image at matched rate. Report $\Delta\alpha = \alpha_B - \alpha_A$ with bootstrap CIs. **If $|\Delta\alpha| < 0.01$ and arm C sits with B rather than A, the alphabet is irrelevant and the head explains the literature's gap.** That single comparison resolves the question as posed.

## 9. Key References

- **[Foundational]** van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS 2017. — arXiv:1711.00937
- **[Foundational]** Esser, Rombach, Ommer. *Taming Transformers for High-Resolution Image Synthesis.* CVPR 2021. — arXiv:2012.09841
- **[Theory]** Blau, Michaeli. *Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff.* ICML 2019. — arXiv:1901.07821
- **[SOTA]** Mentzer, Minnen, Agustsson, Tschannen. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR 2024. — arXiv:2309.15505
- **[SOTA]** Yu, Lezama, Gundavarapu, et al. *Language Model Beats Diffusion — Tokenizer is Key to Visual Generation.* ICLR 2024. — arXiv:2310.05737
- **[SOTA]** Li, Tian, Li, Deng, He. *Autoregressive Image Generation without Vector Quantization.* NeurIPS 2024. — arXiv:2406.11838
- **[SOTA]** Tian, Jiang, Yuan, Peng, Wang. *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.* NeurIPS 2024. — arXiv:2404.02905
- **[SOTA]** Zhou, Wang, Aghajanyan, et al. *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.* 2024. — arXiv:2408.11039
- **[Systems]** Chameleon Team (Meta FAIR). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* 2024. — arXiv:2405.09818
- **[Systems]** Wang, Zhang, Zhang, et al. *Emu3: Next-Token Prediction is All You Need.* 2024. — arXiv:2409.18869
- **[Systems]** Yu, Weber, Deng, Shen, Cremers, Chen. *An Image is Worth 32 Tokens for Reconstruction and Generation.* NeurIPS 2024. — arXiv:2406.07550
- **[Evaluation]** Tong, Brown, Wu, et al. *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.* NeurIPS 2024. — arXiv:2406.16860

## 10. Worked Example

Take one ImageNet image at $256 \times 256$ and $n = 256$ latent positions.

**Discrete arm.** $K = 16384 \Rightarrow \log_2 K = 14$ bits/token, so $R_{\max} = 256 \times 14 = 3584$ bits $= 448$ bytes. Measured code entropy on held-out data is typically well below the ceiling — at $H \approx 9$ bits/token the operational rate is $2304$ bits.

**Continuous arm.** $d = 16$, fp16: $256 \times 16 \times 16 = 65536$ bits nominal. The KL rate is far lower — a typical f8 KL-VAE gives $D_{\mathrm{KL}} \approx 10^4$ nats $\approx 1.4\times10^4$ bits.

**Now try to match.** Three protocols, three verdicts:

| Protocol | Held fixed | Which arm is favoured |
|---|---|---|
| Token-matched | $n = 256$ both | Continuous — carries ~6× the bits |
| Bit-matched | $R \approx 2300$ bits | Discrete — continuous must be quantized to comply, giving up its advantage |
| FLOP-matched | $C$ including head | Ambiguous — the diffusion head adds 10–30% FLOPs per token; the softmax head adds $K \times d_{\text{model}}$ output params |

The obstruction is visible in the table, not the text: the same two systems reverse rank depending on which quantity is pinned. The literature's reported gaps — Transfusion's compute-multiple over Chameleon, VAR's FID 1.73 against MAR's 1.55 — are each measured under a *different* one of these three protocols. Until one paper reports all three for the same pair of models, the question is not open for lack of experiments; it is open for lack of a comparison anyone agrees is fair.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*