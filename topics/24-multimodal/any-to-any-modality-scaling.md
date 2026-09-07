---
id: 24-multimodal/any-to-any-modality-scaling
title: "Any-to-Any Modality Scaling Beyond Two Modalities"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Any-to-Any Modality Scaling Beyond Two Modalities

> **Topic:** Multimodal Models · **ID:** `24-multimodal/any-to-any-modality-scaling` · **Status:** empirically-open

## 1. Problem Statement

A single model consumes any subset of $M$ modalities and emits any subset. Text+image ($M=2$) works: joint training beats separate models on both directions. What happens at $M = 5, 10, 21$ is not established.

Three distinct variants, routinely conflated:

- **Measurement.** Does adding modality $m_{M+1}$ to a fixed compute budget improve, leave unchanged, or degrade the $M(M-1)$ pre-existing generation directions? Reported as a per-direction delta at matched FLOPs, not matched epochs.
- **Method.** Is there an architecture or training schedule whose aggregate loss improves monotonically in $M$ at fixed total compute — i.e. late-stage modality addition without retraining, or capacity allocation (mixture-of-experts, modality-specific adapters) that converts interference into transfer?
- **Theory.** Under what conditions on the joint distribution $p(x_1,\dots,x_M)$ does a shared parameter vector dominate $M$ separate ones? The known scaling-law fits are empirical curve fits with no derivation.

Solved would mean: a scaling law, validated out-of-sample at $M \ge 8$ and $\ge 10^{22}$ FLOPs, that predicts each direction's loss from $(N, D, M, \text{mixture weights})$ within the run-to-run seed noise.

## 2. Formal Setting

Modalities $\mathcal{M} = \{1,\dots,M\}$, each with tokenizer $T_m: \mathcal{X}_m \to V_m^{*}$ into a discrete vocabulary (VQ codes for image/audio, BPE for text). Everything is measured in **tokens after tokenization**, which is where most cross-paper comparisons break: one image is 1024 tokens in Chameleon, 256 in some VQ setups, and a continuous patch embedding in others.

Training corpus is a mixture over subsets $S \subseteq \mathcal{M}$ with weights $\alpha_S \ge 0$, $\sum_S \alpha_S = 1$. Total token budget $D$, parameters $N$, compute $C \approx 6ND$.

Per-direction evaluation: for source set $A$ and target modality $b \notin A$,

$$L_{A \to b}(N, D, \alpha) = \mathbb{E}_{(x_A, x_b) \sim p_{\text{eval}}}\left[-\tfrac{1}{|T_b(x_b)|}\log p_\theta\big(T_b(x_b) \mid T_A(x_A)\big)\right]$$

measured in nats per target token on a held-out set disjoint from training at the document level. There are $\sum_{b}(2^{M-1}-1) \approx M 2^{M-1}$ such directions; the usual reported subset is the $M(M-1)$ single-source ones.

**Competition coefficient.** Following Aghajanyan et al. (2023), define for direction $A \to b$

$$\Delta_{A \to b}(C) = L^{\text{joint}}_{A \to b}(C) - L^{\text{spec}}_{A \to b}(C)$$

where the specialist arm spends the *same* compute $C$ on only the modalities in $A \cup \{b\}$. $\Delta < 0$ is synergy, $\Delta > 0$ is competition. The empirically observed pattern is a **competition barrier**: $\Delta > 0$ below a critical compute $C^{*}(\alpha)$ and $\Delta \le 0$ above it. The open quantity is the growth of $C^{*}$ in $M$.

Assumptions, with the ones known to fail marked:

1. Tokenizers are fixed and lossless enough that $L$ ranks models consistently. **Violated** — VQ reconstruction error puts a modality-specific floor on $L_{A\to b}$, so cross-modality loss comparison is meaningless in absolute terms.
2. $p_{\text{eval}}$ is the same for joint and specialist arms. Holds only if the eval set is built independent of the training mixture; frequently violated when web-scraped pairs leak.
3. Compute-matching implies fair comparison. **Violated** in practice: joint models are almost always given more data, more parameters, or a longer schedule than the specialist baseline, and many papers report no specialist arm at all.
4. Directions are exchangeable enough for a single law. Unlikely to hold — text→image and image→text have asymmetric entropy (image targets carry $\sim$1–3 orders of magnitude more tokens).

## 3. State of the Art

**Empirical / systems SOTA.**

- **4M-21** (Bachmann et al., NeurIPS 2024): 21 modalities (RGB, depth, normals, segmentation, edges, poses, metadata, text), 3B parameters, masked multimodal token prediction. Establishes that $M=21$ *trains stably* and that any-to-any generation is qualitatively coherent. The paper reports no matched-compute specialist arm for each of the 21 modalities, so per-direction competition is not measured.
- **Unified-IO 2** (Lu et al., CVPR 2024): 7B, text/image/audio/action from scratch, single autoregressive stack. Contributes architectural stabilizers (2D RoPE, QK-norm, scaled cosine attention) required to keep multimodal pretraining from diverging — an *established* engineering finding, reproduced by later stacks.
- **Chameleon** (Meta, 2024): early-fusion, fully token-based text+image at 7B/34B; the divergence/stability analysis is the durable contribution. **Transfusion** (Zhou et al., 2024) shows diffusion-on-continuous-image-tokens inside one transformer beats VQ tokenization at matched compute for image generation — an ablated result, but at $M=2$.
- **NExT-GPT** (Wu et al., ICML 2024) and **AnyGPT** (Zhan et al., ACL 2024): $M=4$ (text/image/audio/video) via frozen encoders and decoders with a trained connector. Cheap, and the honest reading is that they scale $M$ by *not* training jointly — cross-modal composition is delegated to frozen adapters. Their any-to-any numbers are benchmark scores, not competition measurements.
- **ImageBind** (Girdhar et al., CVPR 2023): binds 6 modalities to image space using only image-paired data, and reports emergent zero-shot alignment between never-co-observed pairs (e.g. audio→depth). Emergent alignment is established; it is *representation* alignment, not generation.

**Theory SOTA.** Aghajanyan et al., *Scaling Laws for Generative Mixed-Modal Language Models* (ICML 2023): fitted laws over text, image, speech, code combinations up to 30B parameters, introducing an explicit competition/synergy term and the compute-dependent barrier. This is a fit, not a derivation, and covers $M \le 4$ in pairs — not simultaneous high-$M$.

**Claimed but unablated:** that any-to-any models "unify" modalities. Every published $M \ge 6$ any-to-any system reports downstream benchmark numbers (VQA, captioning, audio captioning) rather than matched-compute per-direction loss deltas against specialists.

## 4. What Is Known

- **Mixed-modal competition is real and compute-dependent.** Aghajanyan et al. (ICML 2023), models 8M–30B: joint text+image models underperform matched-compute specialists below a crossover, and the crossover moves with mixture weight. Measured for pairs.
- **Alignment can emerge without paired data.** ImageBind (CVPR 2023), 6 modalities: zero-shot audio→depth retrieval well above chance despite no audio-depth pairs in training. Scale: ViT-H image encoder, standard web-scale image pairings.
- **High-$M$ training is stable with the right normalization.** 4M-21 (3B, 21 modalities) and Unified-IO 2 (7B, 4 modalities) both converge; instability in early-fusion multimodal training is attributable to logit drift and is fixed by QK-norm / z-loss (Chameleon, 2024).
- **Tokenizer choice dominates architecture at $M=2$.** Transfusion (2024) reports roughly an order-of-magnitude compute advantage over VQ-token autoregression for image generation quality at matched parameters — i.e. the discretization decision, not the fusion topology, sets the frontier.
- **The modality gap is a geometric fact.** Liang et al. (NeurIPS 2022): contrastively trained encoders place modalities in disjoint cones of the shared sphere; the gap is set at initialization and preserved by the contrastive objective. Measured on CLIP-scale models.

## 5. What Is Not Known

- **Empirically open (primary).** How $\Delta_{A\to b}$ scales in $M$ at fixed compute. Nobody has run an $M$-sweep ($M \in \{2,4,8,16\}$) with matched FLOPs and per-direction specialist controls. The experiment is runnable today at $\sim$1B parameters; it has not been run because the specialist arm multiplies cost by the number of directions.
- **Empirically open.** Whether $C^{*}(M)$ grows polynomially or exponentially in $M$. If exponential, any-to-any at $M>10$ is permanently compute-dominated by specialists.
- **Empirically open.** Whether modality addition is *incremental* — can modality 22 be added to a trained $M=21$ model without degrading the existing 21·20 directions, at cost sublinear in the original pretraining?
- **Theoretically open.** No theorem gives conditions on $p(x_1,\dots,x_M)$ under which parameter sharing dominates separate models. Even the two-modality case lacks a bound; the scaling laws are fits.
- **Methodologically blocked.** Cross-direction loss comparison. $L_{A\to b}$ in nats/token is not commensurable across $b$ because tokenizers differ in reconstruction floor and sequence length. There is no accepted normalization, so "the model got worse at audio and better at depth" is currently not a well-posed statement.

## 6. Why It Is Hard

Three named obstructions.

1. **Quadratic-to-exponential control cost.** A clean measurement needs one specialist per direction. At $M=8$ that is 56 single-source directions; at matched compute each specialist costs the same as the joint model. The controlled experiment is $\sim$50× the joint run, which is why every paper skips it. This is a compute-cost obstruction with a combinatorial multiplier, not merely a large-model obstruction.
2. **Non-identifiability of interference vs. capacity.** A drop in $L_{\text{text}\to\text{image}}$ when audio is added is consistent with (a) representational interference, (b) fewer text-image tokens seen, (c) parameter capacity exhaustion. Separating these needs a 3-way sweep over $(N, \alpha, M)$; two-way sweeps — what everyone runs — cannot distinguish them.
3. **The evaluation does not measure what it names.** "Any-to-any" is scored on per-modality benchmarks (VQA-v2, COCO captioning, AudioCaps). These test $A\to b$ for a handful of $|A|=1$ pairs. The compositional claim — that the model handles arbitrary subsets $A$ — is untested, and no benchmark exists whose items require $|A| \ge 3$ sources jointly.

## 7. Current Research (as of 2026)

- **Discrete-plus-continuous hybrids.** Transfusion-style single-stack models with continuous image/audio latents and discrete text; the open question is whether the hybrid raises or lowers $C^{*}$ at high $M$. Meta FAIR, and academic follow-ups. *(frontier — verify)*
- **Modality-routed sparsity.** MoE with per-modality expert affinity as an explicit competition-mitigation mechanism; predicted to flatten $C^{*}(M)$ by giving each modality private capacity. Reported in several 2024–2026 systems; a matched-compute ablation isolating routing from added parameters is still missing. *(frontier — verify)*
- **Modality-count scaling laws.** Extending Aghajanyan-style fits to simultaneous $M \ge 6$. EPFL/Apple (the 4M line), Allen AI (Unified-IO line). *(frontier — verify)*
- **Incremental modality grafting.** Adding a modality post-hoc via adapters or continued pretraining with replay; the metric of interest is forgetting on the pre-existing directions.

## 8. Concrete Next Experiment

**The modality-count sweep with specialist controls.**

- **Scale.** Decoder-only transformer, $N = 1.3$B, $C = 3\times10^{21}$ FLOPs per arm ($D \approx 4\times10^{11}$ tokens). Four joint arms at $M \in \{2,4,8,16\}$ drawn from a fixed pool (text, RGB, depth, normals, segmentation, audio, video frames, code, tabular, pose, …), uniform $\alpha$ over available pairs.
- **Control arm.** For six *fixed probe directions* present in all four arms (text→RGB, RGB→text, text→audio, audio→text, RGB→depth, depth→RGB): a specialist trained on only those two modalities at the *same* $3\times10^{21}$ FLOPs. Six specialists, not 240 — this is what makes the experiment affordable ($\sim$10 arms total, roughly $3\times10^{22}$ FLOPs, order 10k H100-days).
- **Controls for confound (2).** Repeat the $M=8$ arm at $N \in \{0.4, 1.3, 4\}$B to separate capacity exhaustion from interference, and at fixed per-direction token count (varying $D$) to separate data dilution.
- **Deciding number.** The slope $\beta$ in $\bar{\Delta}(M) = \frac{1}{6}\sum_{\text{probes}} \Delta_{A\to b} = \beta \log_2 M + c$, in nats per target token, with seed noise estimated from 3 seeds at $M=4$. **$\beta \le 0$: any-to-any scaling is benign and the field should build high-$M$ models. $\beta > 0$ and larger than seed noise: competition compounds with $M$, and the $C^{*}$ needed at $M=16$ should be extrapolated before any further high-$M$ system is trained.** Everything else in the sweep is secondary to that one slope.

## 9. Key References

- **[Foundational]** Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, Luke Zettlemoyer. *Scaling Laws for Generative Mixed-Modal Language Models.* ICML, 2023. — arXiv:2301.03728
- **[Foundational]** Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra. *ImageBind: One Embedding Space To Bind Them All.* CVPR, 2023. — arXiv:2305.05665
- **[Foundational]** Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Zou. *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS, 2022. — arXiv:2203.02053
- **[SOTA]** Roman Bachmann, Oğuzhan Fatih Kar, David Mizrahi, Ali Garjani, Mingfei Gao, David Griffiths, Jiaming Hu, Afshin Dehghan, Amir Zamir. *4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities.* NeurIPS, 2024. — arXiv:2406.09406
- **[SOTA]** David Mizrahi, Roman Bachmann, Oğuzhan Fatih Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, Amir Zamir. *4M: Massively Multimodal Masked Modeling.* NeurIPS, 2023. — arXiv:2312.06647
- **[SOTA]** Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, Aniruddha Kembhavi. *Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action.* CVPR, 2024. — arXiv:2312.17172
- **[SOTA]** Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, Omer Levy. *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.* 2024. — arXiv:2408.11039
- **[SOTA]** Chameleon Team (Meta AI). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* 2024. — arXiv:2405.09818
- **[Method]** Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal. *Any-to-Any Generation via Composable Diffusion (CoDi).* NeurIPS, 2023. — arXiv:2305.11846
- **[Method]** Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, Tat-Seng Chua. *NExT-GPT: Any-to-Any Multimodal LLM.* ICML, 2024. — arXiv:2309.05519
- **[Method]** Jun Zhan et al. *AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling.* ACL, 2024. — arXiv:2402.12226

## 10. Worked Example

Take $M=4$: text (T), RGB (I), depth (D), audio (A). Probe direction T→I.

Fix $C = 3\times10^{21}$ FLOPs, $N=1.3$B, so $D \approx 3.8\times10^{11}$ tokens. Under uniform $\alpha$ over the 6 unordered pairs, the T–I pair receives $1/6$ of the budget: $6.4\times10^{10}$ tokens. The T+I specialist at the same FLOPs receives all $3.8\times10^{11}$ — **6× more T–I data**.

Chinchilla-style single-modality slopes give roughly $L \propto D^{-0.28}$ in the data-limited regime, so from data dilution alone the joint model's T→I loss is worse by a factor $6^{0.28} \approx 1.63$ on the reducible part. If the specialist's reducible loss is 0.40 nats/token above an irreducible floor of 2.10 (typical for VQ image tokens at 1024 tokens/image), the joint model lands near $2.10 + 0.65 = 2.75$, i.e. $\Delta_{T\to I} \approx +0.25$ nats/token.

Now the obstruction. Suppose the measured $\Delta$ is $+0.31$. The 0.06 nat excess over the dilution prediction is the interference signal — and it sits inside the error bars of the exponent 0.28, which is itself fitted, modality-specific, and not known to better than $\pm 0.03$. Propagating that uncertainty gives a dilution prediction of $2.75 \pm 0.05$ nats. **The confound is the same size as the effect.** That is why the sweep in §8 requires the $N$-sweep and the fixed-per-direction-token arm: without them, every reported "modality interference" number at $M>2$ is indistinguishable from having simply shown the model less of the relevant data.

Extend to $M=16$: the T–I pair gets $1/120$ of the budget, a 120× dilution, predicted penalty $120^{0.28} \approx 3.8\times$ on reducible loss. Any interference term must be measured against a 3.8× baseline shift. The measurement problem gets *harder* exactly as $M$ grows, which is the reason the question is still open despite $M=21$ systems existing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*