---
id: 24-multimodal/early-versus-late-fusion-scaling
title: "Early Fusion versus Late Fusion at Scale"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Early Fusion versus Late Fusion at Scale

> **Topic:** Multimodal Models · **ID:** `24-multimodal/early-versus-late-fusion-scaling` · **Status:** empirically-open

## 1. Problem Statement

Two architectures dominate multimodal training:

- **Late fusion** (also "compositional"): a pretrained unimodal encoder per modality (e.g. a ViT trained with CLIP-style contrastive loss) feeds a pretrained text LLM through a connector — linear projection, MLP, Q-Former, or cross-attention. LLaVA, BLIP-2, Flamingo, and most production VLMs are here.
- **Early fusion** (also "native"): one transformer consumes interleaved tokens from all modalities from random initialization, with no modality-specific pretrained encoder. Chameleon, Fuyu, Transfusion, Emu3.

The decision predicate: **fix a training compute budget $C$ and a target task distribution; which architecture family attains lower loss, and does the answer flip as $C$ grows?**

Three variants, with different difficulty:

- **Measurement.** Do the two families have different scaling exponents, or only different offsets? An offset difference is a fixed multiplicative compute penalty; an exponent difference means the ranking flips at some crossover budget $C^\star$. Estimating $C^\star$ requires fitting two scaling laws, not comparing two models.
- **Method.** Given a budget, what is the compute-optimal allocation — parameters vs tokens, and within parameters, dense vs modality-aware sparse (MoE)?
- **Theory.** Is there any reason a shared representation should be *worse* than a modular one at fixed capacity, beyond optimization difficulty? No formal result either way.

Solving it means: a published, reproduced pair of scaling laws with confidence intervals on $C^\star$ that hold on held-out downstream tasks, not just on pretraining loss.

## 2. Formal Setting

Let $\mathcal{D}$ be a distribution over interleaved documents $x = (x_1,\dots,x_T)$ whose tokens carry modality labels $m_t \in \{\text{text},\text{image}\}$. Training compute is measured as $C \approx 6 N D$ FLOPs for dense models with $N$ non-embedding parameters and $D$ training tokens (Kaplan et al. 2020); for sparse models $N$ is replaced by active parameters $N_{\text{act}}$, which is the first place the comparison gets slippery.

Per-modality loss, measured on a held-out shard:

$$L_m(N,D) = -\frac{1}{|\mathcal{T}_m|}\sum_{t \in \mathcal{T}_m} \log p_\theta(x_t \mid x_{<t}), \qquad \mathcal{T}_m = \{t : m_t = m\}.$$

The aggregate objective is $L = \sum_m w_m L_m$ with mixing weights $w_m$ fixed by the data recipe. Fit, per architecture $a \in \{\text{early},\text{late}\}$, the Chinchilla parametric form (Hoffmann et al. 2022):

$$L_a(N,D) = E_a + \frac{A_a}{N^{\alpha_a}} + \frac{B_a}{D^{\beta_a}},$$

and the compute-optimal frontier $N_a^{\text{opt}}(C) \propto C^{p_a}$, $D_a^{\text{opt}}(C) \propto C^{1-p_a}$. The crossover budget is

$$C^\star = \inf\{C : L_{\text{early}}(C) < L_{\text{late}}(C)\}.$$

**Compute accounting for late fusion.** The honest budget must include the encoder's own pretraining: $C_{\text{late}} = C_{\text{train}} + \gamma\, C_{\text{enc}}$, where $C_{\text{enc}}$ is the FLOPs that produced the vision encoder and $\gamma \in [0,1]$ amortizes it over the models that reuse it. Every published comparison implicitly picks a $\gamma$ and almost none state it. $\gamma = 0$ (the usual default) means late fusion is scored on borrowed compute; $\gamma = 1$ charges a single downstream model for a CLIP run reused thousands of times. $C^\star$ is a function of $\gamma$, so a single scalar answer does not exist.

**Assumptions, and which are violated:**

1. *Loss is comparable across architectures.* Violated when tokenizers differ. A VQ image tokenizer (Chameleon) and continuous patch embeddings (LLaVA) do not define the same $L_{\text{image}}$; cross-entropy over an 8192-entry codebook is not on the same scale as a diffusion or regression head. Only $L_{\text{text}}$ is directly comparable, and only when the text tokenizer matches.
2. *$C \approx 6ND$.* Breaks for high image-token counts where attention is a non-trivial FLOP share, and for encoder-based models whose encoder runs at a different resolution than the LM.
3. *Data recipe is architecture-independent.* Violated: early-fusion runs typically use far more raw interleaved data, late-fusion runs use curated instruction data.
4. *Pretraining loss predicts downstream accuracy monotonically.* Holds within a family; poorly established across families with different image tokenizations.

## 3. State of the Art

**Established (with ablations).**

- Laurençon et al., *What matters when building vision-language models?* (NeurIPS 2024, arXiv:2405.02246) is the cleanest controlled ablation of connector design at fixed data. Its main finding: with a **frozen** LM, cross-attention fusion beats a fully-autoregressive projection; when the LM is **unfrozen**, the ranking reverses and the simpler projection wins. This is an interaction effect, not a main effect — and it means single-configuration comparisons are uninformative.
- Cherti et al., *Reproducible scaling laws for contrastive language-image learning* (CVPR 2023, arXiv:2212.07143) established clean power-law scaling for the late-fusion *encoder* alone across 3 orders of magnitude of compute on LAION.

**Claimed, partially ablated.**

- Shukor, Fini, Turrisi da Costa, Cord, Susskind, El-Nouby, *Scaling Laws for Native Multimodal Models* (2025) is the only large sweep aimed directly at this question: hundreds of models trained from scratch, spanning roughly $0.3$B–$8$B parameters. Reported conclusions: early- and late-fusion scaling exponents are close; early fusion is *better* at small $N$ and lower training/inference cost; late fusion's advantage, where present, is an offset not a slope; and modality-aware sparsity (MoE) improves early fusion substantially. The sweep does not charge $\gamma>0$ for encoder pretraining and does not extend past $\sim$8B, so the interesting extrapolation is unverified.
- Lin et al., *MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts* (2024, arXiv:2407.21770) reports up to $\sim$3.7$\times$ pretraining FLOPs savings at 1.4B active parameters (about 5.2$\times$ on image tokens, 2.6$\times$ on text) at matched loss versus a dense early-fusion baseline.

**Benchmark numbers only, no matched-compute control.** Chameleon (Meta, 2024, arXiv:2405.09818) at 7B/34B and Transfusion (Zhou et al., 2024, arXiv:2408.11039) both report strong mixed-modal results, but neither is paired with a late-fusion arm at equal $C$, equal data, and equal tokenizer. Their numbers are evidence that early fusion *works* at scale, not evidence about the ranking.

## 4. What Is Known

- **Both families scale as power laws.** Contrastive late-fusion encoders: clean fits over $\sim$3 decades of compute up to ViT-G/14 on 2B samples seen (Cherti et al. 2023). Early-fusion native models: Chinchilla-form fits reported up to 8B parameters and $\sim$10$^2$B tokens (Shukor et al. 2025).
- **Compute-optimal allocation is close to Chinchilla in both families.** Reported $N^{\text{opt}} \propto C^{p}$ with $p \approx 0.5$; early fusion trends slightly more token-hungry as the image fraction of the mixture rises.
- **Sparsity helps early fusion more than late fusion.** MoMa: 3.7$\times$ FLOPs saving at 1.4B active params; the gain is concentrated on image tokens (5.2$\times$), consistent with modality-specific features being cheaply separable inside a shared trunk.
- **Freezing changes the answer.** The frozen-vs-unfrozen LM interaction (Laurençon et al. 2024) is reproduced across at least two architecture families.
- **Early fusion is cheaper at inference at matched quality in the reported range** (no separate encoder forward pass, fewer parameters at matched loss below $\sim$8B) — measured, not extrapolated.

## 5. What Is Not Known

- **Empirically open (primary).** Whether the small-$N$ early-fusion advantage survives past $\sim$10B parameters and $\sim$1T tokens. Nobody has run matched-compute arms at $\ge$30B. The experiment is entirely runnable; it costs on the order of $10^{23}$ FLOPs per arm.
- **Empirically open.** The value of $C^\star$ as a function of $\gamma$ (encoder-amortization). No published sweep varies $\gamma$ at all.
- **Methodologically blocked.** A modality-agnostic loss that lets $L_{\text{image}}$ be compared across discrete-token, continuous-embedding, and diffusion-head models. Without it, cross-family comparison is restricted to $L_{\text{text}}$ and downstream accuracy — both of which are weak, noisy proxies for the quantity in question.
- **Theoretically open.** No separation result. There is no theorem stating that a shared trunk of width $d$ can or cannot represent the same function class as two width-$d/2$ trunks plus a connector, under any realistic assumption on the interleaved data distribution. Nor any result on whether joint training induces gradient conflict that grows with scale.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiable compute accounting.**

Every published comparison varies at least four things at once: tokenizer, data mixture, initialization (pretrained vs random), and effective compute. Attributing the loss gap to "fusion depth" requires holding the other three fixed, which is impossible in the strict sense — late fusion is *defined* by having a pretrained encoder, so "same initialization" cannot be enforced. The best available control is to charge the encoder's pretraining FLOPs, and that reintroduces the free parameter $\gamma$, which no experiment can pin down: it depends on how many downstream models will reuse the encoder, a deployment fact, not a scientific one.

Secondary obstruction: cost. Resolving an exponent difference of $\Delta\alpha \approx 0.02$ needs at least a decade of compute with $\ge$5 points per arm and seed replicates. At frontier scale that is a multi-million-dollar sweep whose payoff is a methodological answer, not a shippable model — which is why labs that could run it publish single flagship models instead.

## 7. Current Research (as of 2026)

- **Apple (Shukor, El-Nouby, Susskind and collaborators)** — native multimodal scaling laws; the closest thing to a direct attack on $C^\star$.
- **Meta FAIR** — Chameleon/MoMa line: early fusion plus modality-aware sparsity; the open question they are pushing is whether expert routing removes the modality-interference term entirely *(frontier — verify current status)*.
- **Hybrid objectives** — Transfusion-style models combining next-token prediction on text with diffusion on continuous image latents. These sit between the two families and make the loss-comparability problem worse, not better.
- **Any-to-any and audio/video extension** — whether the early-fusion advantage grows with the number of modalities (more encoders to amortize) is being asked but not yet answered with matched-compute arms *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Six points per arm on a log-spaced compute ladder: $N \in \{0.5, 1, 3, 8, 15, 30\}$B, each trained Chinchilla-optimally ($D \approx 20N$), on one frozen data mixture (say 60% text / 40% interleaved image-text), one shared BPE text tokenizer, and one shared image representation for both arms — e.g. the same VQ tokenizer, used as input to the early-fusion trunk and as the target for the late-fusion arm's projector. Two seeds at the three smallest points to estimate fit noise. Total $\approx 3\times10^{23}$ FLOPs per arm.

**Control arm.** Late fusion with a *from-scratch* ViT of matched parameter count, trained jointly, with its FLOPs fully charged ($\gamma = 1$). This is the arm nobody runs, and it is the only one that makes the comparison identifiable. Two ancillary arms at $\gamma \in \{0, 0.1\}$ reusing a public CLIP encoder bracket the deployment-realistic range.

**The deciding number.** Fit $L_a(C)$ on each arm and report

$$\Delta\alpha = \alpha_{\text{early}} - \alpha_{\text{late}}$$

with a bootstrap 95% CI on held-out $L_{\text{text}}$ (the only cross-family-comparable loss). **If the CI on $\Delta\alpha$ excludes 0, the ranking flips at a finite $C^\star$ and the field should report $C^\star$, not architecture preferences. If the CI contains 0 and $|\Delta\alpha| < 0.01$, the families differ only by an offset, the whole question collapses to "how large is the constant-factor penalty", and the answer is a single multiplicative number reportable in one line.** Secondary readout: the same fit at $\gamma = 0$, to quantify how much of late fusion's reputation is borrowed compute.

## 9. Key References

- **[Foundational]** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Alec Radford, Jong Wook Kim, Chris Hallacy, et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. — arXiv:2103.00020
- **[Foundational]** Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS, 2022. — arXiv:2204.14198
- **[SOTA]** Mustafa Shukor, Enrico Fini, Victor Guilherme Turrisi da Costa, Matthieu Cord, Joshua Susskind, Alaaeldin El-Nouby. *Scaling Laws for Native Multimodal Models.* Apple, 2025. (identifier omitted — verify before citing)
- **[SOTA]** Chameleon Team (Meta FAIR). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* 2024. — arXiv:2405.09818
- **[SOTA]** Xi Victoria Lin, Akshat Shrivastava, Liang Luo, et al. *MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts.* 2024. — arXiv:2407.21770
- **[SOTA]** Chunting Zhou, Lili Yu, Arun Babu, et al. *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.* ICLR, 2025. — arXiv:2408.11039
- **[Ablation]** Hugo Laurençon, Léo Tronchon, Matthieu Cord, Victor Sanh. *What matters when building vision-language models?* NeurIPS, 2024. — arXiv:2405.02246
- **[Empirical]** Mehdi Cherti, Romain Beaumont, Ross Wightman, et al. *Reproducible scaling laws for contrastive language-image learning.* CVPR, 2023. — arXiv:2212.07143
- **[Survey]** Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* ICML, 2023. — arXiv:2301.12597
- **[Survey]** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee. *Visual Instruction Tuning.* NeurIPS, 2023. — arXiv:2304.08485

## 10. Worked Example

Take a 7B late-fusion VLM: a 6.7B LM plus a 0.3B ViT-L encoder, fine-tuned on $D = 5\times10^{10}$ multimodal tokens.

Nominal training compute, encoder excluded:

$$C_{\text{train}} \approx 6 \times 7\times10^9 \times 5\times10^{10} = 2.1\times10^{21}\ \text{FLOPs}.$$

Now charge the encoder. A CLIP ViT-L run of the scale used in OpenCLIP reproductions sees on the order of $3.4\times10^{10}$ samples-seen-equivalents; at roughly $2\times10^{11}$ FLOPs per image forward-backward at 224px, that is $C_{\text{enc}} \approx 7\times10^{21}$ FLOPs — **more than three times the LM-side budget it is being bolted onto**.

So:

| $\gamma$ | $C_{\text{late}}$ (FLOPs) | ratio to matched early-fusion arm at $2.1\times10^{21}$ |
|---|---|---|
| 0 (standard reporting) | $2.1\times10^{21}$ | 1.0$\times$ |
| 0.1 (reused by ~10 models) | $2.8\times10^{21}$ | 1.3$\times$ |
| 1.0 (single use) | $9.1\times10^{21}$ | 4.3$\times$ |

Suppose the late-fusion model beats the early-fusion model by 1.5 points on an aggregate VQA suite. Using the reported early-fusion fit, roughly $L \propto C^{-0.05}$ near this budget, a 4.3$\times$ compute increase buys the early-fusion arm about a $7\%$ loss reduction — comfortably more than 1.5 points on a suite where 1 point of accuracy costs well under $2\times$ compute in this range.

**The obstruction, made visible:** the *same measured 1.5-point gap* supports "late fusion wins" at $\gamma = 0$ and "early fusion wins by a wide margin" at $\gamma = 1$. The free parameter is not measurable from the experiment — it is a fact about how many models will reuse the encoder in deployment. Until papers report $\gamma$ explicitly, or run the from-scratch-encoder control arm of Section 8, the published comparisons are not comparisons.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*