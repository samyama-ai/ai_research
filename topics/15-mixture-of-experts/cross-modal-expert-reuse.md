---
id: 15-mixture-of-experts/cross-modal-expert-reuse
title: "Expert Reuse Across Modalities"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expert Reuse Across Modalities

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/cross-modal-expert-reuse` · **Status:** empirically-open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) layer routes each token to $k$ of $E$ experts. When the token stream carries more than one modality — image patches, text subwords, audio frames, video tubelets — the router may send tokens of different modalities to the *same* expert. The question is whether that sharing does any work.

Three variants, routinely conflated:

- **Measurement.** Given a trained multimodal MoE, quantify how much of an expert's parameters are *causally* used by more than one modality. Routing overlap (how often modalities co-visit an expert) is the observable; causal reuse is the quantity of interest, and they are not the same thing.
- **Method.** Design routing, initialization, and auxiliary losses that maximize genuine cross-modal reuse at fixed active-parameter and FLOP budget — i.e. beat a modality-siloed control (hard-partitioned experts, as in VLMo's mixture-of-modality-experts) on both modalities at once.
- **Theory.** Characterize when a single expert function $f_e$ can serve two input distributions better than two dedicated experts of half the width each. This is a question about shared low-rank structure across modality-conditional feature distributions, and there is no theorem either way.

Solving it means: a reuse metric that is invariant to router temperature and load-balancing pressure, plus a demonstration that the metric predicts the loss cost of removing shared experts.

## 2. Formal Setting

Modalities $m \in \mathcal{M}$, $|\mathcal{M}| = M$. A token is $(x, m)$ with $x \in \mathbb{R}^d$. MoE layer $\ell$ has experts $\{f_{\ell,e}\}_{e=1}^{E}$ and router $g_\ell: \mathbb{R}^d \to \Delta^{E-1}$; top-$k$ gives $\mathcal{T}_\ell(x) \subset [E]$, $|\mathcal{T}_\ell(x)| = k$, and
$$y = \sum_{e \in \mathcal{T}_\ell(x)} \frac{g_{\ell,e}(x)}{\sum_{e' \in \mathcal{T}_\ell(x)} g_{\ell,e'}(x)} f_{\ell,e}(x).$$

**Measured quantities.**

- *Modality-conditional expert usage.* Over a held-out corpus, $p_\ell(e \mid m) = \frac{1}{N_m}\sum_{(x,m)} \mathbb{1}[e \in \mathcal{T}_\ell(x)]$, normalized to sum to $k$; measured with routers frozen, no auxiliary loss, batch composition matched to pretraining ratios.
- *Routing overlap.* $O_\ell = 1 - \mathrm{JS}\big(p_\ell(\cdot\mid m_1)\,\|\,p_\ell(\cdot\mid m_2)\big)/\log 2 \in [0,1]$. Purely correlational.
- *Causal reuse.* For expert $e$, zero its output (route-around, renormalize the remaining $k-1$ gates) and measure per-modality loss deltas $\delta_{\ell,e}^{(m)} = L_m(\text{ablated}) - L_m(\text{full})$ in nats/token. Define
$$C_{\ell,e} = \frac{\min_m \delta_{\ell,e}^{(m)}}{\max_m \delta_{\ell,e}^{(m)}} \in [0,1],$$
with $C=1$ meaning both modalities depend on $e$ equally and $C \approx 0$ meaning one modality merely *visits* it. Layer-level reuse is the usage-weighted mean $\bar{C}_\ell$.
- *Transfer utility.* $\Delta_m = L_m(\text{siloed}) - L_m(\text{shared})$ at matched active parameters, total parameters, FLOPs, tokens, and per-modality token counts. Positive $\Delta_m$ for all $m$ is the only unambiguous evidence that reuse pays.

**Assumptions, and which break.**

1. *Routing is stable under ablation.* Violated: removing an expert shifts the gate distribution, so single-expert ablation confounds the expert's function with router recalibration.
2. *Loss deltas are comparable across modalities.* Violated: image-patch reconstruction/contrastive losses and text cross-entropy live on different scales; $C_{\ell,e}$ needs per-modality normalization (e.g. by $L_m$ of a modality-only control) or it is meaningless.
3. *Modality identity is a partition of tokens.* Violated for OCR-heavy images, speech transcripts, and interleaved documents, where "modality" is a label on the encoder, not on the information.
4. *Load-balancing loss is neutral.* Violated by construction: an auxiliary balance term with coefficient $\alpha$ pushes $p_\ell(\cdot \mid m)$ toward uniform, which inflates $O_\ell$ regardless of function. Any reuse number must be reported with $\alpha$.

## 3. State of the Art

**Established.**
- **VLMo** (Bao et al., NeurIPS 2022) — mixture-of-modality-experts: FFN experts *hard-assigned* by modality, self-attention shared. Works well, and is the natural control arm: it demonstrates that most of the cross-modal benefit can be obtained without any shared FFN capacity.
- **LIMoE** (Mustafa et al., NeurIPS 2022) — the first large sparse MoE trained on image–text contrastively with a single shared expert pool. Requires two new auxiliary losses (local and global entropy) plus per-modality priors specifically to *prevent* modality collapse, where one modality monopolizes the experts. LIMoE-H/14 reaches 84.1% zero-shot ImageNet top-1. Established: without the entropy losses training collapses. Also established: experts specialize, some strongly by modality.
- **Sparse upcycling** (Komatsuzaki et al., ICLR 2023) — dense checkpoints converted to MoE by copying the FFN into $E$ experts; the standard initialization for multimodal MoE work.

**Claimed but unablated.**
- **Uni-Perceiver-MoE** (Zhu et al., NeurIPS 2022) attributes gains to reduced task interference via conditional routing; the interference mechanism is inferred from benchmark deltas, not from causal ablation of shared experts.
- **Mod-Squad** (Chen et al., CVPR 2023) uses a mutual-information objective between tasks and experts to shape sharing on Taskonomy; multi-task, not strictly multi-modal, and the MI objective is optimized rather than measured post hoc.
- **MoE-LLaVA** (Lin et al., 2024) and **CuMo** (Li et al., 2024) report expert-routing distributions over image versus text tokens as evidence of sharing. These are routing-overlap plots — benchmark numbers plus a heatmap, with no ablation linking overlap to loss.
- **VL-MoE / scaling VLMs with sparse MoE** (Shen et al., Findings of EMNLP 2023) reports favorable quality-per-FLOP; the reuse mechanism is not isolated.

No published result reports $C_{\ell,e}$-style causal reuse against a FLOP-matched siloed control at $\geq 10$B total parameters.

## 4. What Is Known

- **Modality collapse is real and needs explicit correction.** LIMoE (ViT-H/14 scale, image–text contrastive) requires entropy-based auxiliary losses; without them, routing degenerates. Measured at up to $E = 32$ experts per MoE layer.
- **Hard modality partitioning is competitive.** VLMo (base and large, ~10–200M image–text pairs) achieves strong VQA/NLVR2/retrieval with zero shared FFN capacity across modalities, which bounds how much unique value shared experts can be adding at that scale.
- **Some experts are strongly modality-pure.** LIMoE's published expert-usage analysis shows a subset of experts receiving near-exclusively one modality; the remainder are mixed. The mixed fraction is reported as a routing statistic only.
- **Load balancing dominates routing statistics.** Switch Transformer (Fedus et al., JMLR 2022) and ST-MoE (Zoph et al., 2022) show routing distributions move substantially with the auxiliary-loss coefficient and with router z-loss — established at 1B–1.6T total parameters, single modality.
- **Shared-expert designs help in unimodal LLMs.** DeepSeekMoE (Dai et al., ACL 2024) isolates always-on shared experts and reports gains at 2B and 16B; this is the closest thing to a positive control for "a shared expert can carry common structure", but it is monolingual-modality.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of cross-modal reuse that is invariant to load-balancing pressure and router recalibration under ablation. Every published "sharing" number is a routing statistic, which assumption (4) above shows is partly a function of $\alpha$, not of the model's computation.
- **Empirically open.** Whether a shared-expert multimodal MoE beats a FLOP-, parameter-, and token-matched modality-siloed MoE on *both* modalities at $\geq 10$B total parameters. The experiment is runnable — roughly a few thousand accelerator-days — and has not been published with matched controls.
- **Empirically open.** Whether reuse increases or decreases with $E$ and with scale. Two plausible stories (more experts → finer, purer specialization; more scale → more abstract, modality-agnostic features) predict opposite signs; no scaling sweep tests them.
- **Theoretically open.** No result gives conditions under which one width-$w$ expert serving two modality-conditional distributions beats two width-$w/2$ experts. The natural framing — shared subspace dimension between $\Sigma_{m_1}$ and $\Sigma_{m_2}$ in the layer's input covariance — has no theorem attached in the MoE setting.

## 6. Why It Is Hard

Three named obstructions.

1. **The evaluation does not measure what it names.** "Expert sharing" is reported as routing overlap. Overlap is a property of the router; reuse is a property of the expert. A router pushed toward uniform by the balance loss produces high overlap in a model where each modality uses disjoint subspaces of the same weight matrix — sharing the parameters, not the computation.
2. **Non-identifiability under ablation.** Deleting expert $e$ changes the renormalized gates for every token that touched it, so $\delta^{(m)}_{\ell,e}$ mixes "this expert computed something for modality $m$" with "the router's remaining mass is now miscalibrated". Ablation with router recalibration (re-fit gates, frozen experts) and without give different answers, and neither is obviously correct.
3. **Compute cost of the matched control.** The only clean answer needs a siloed arm matched on active params, total params, FLOPs, tokens, *and* per-modality token ratio — a full second pretraining run at the scale where the effect is supposed to appear. Small-scale runs are not informative because modality collapse behaviour changes with $E$ and depth.

## 7. Current Research (as of 2026)

- **Multimodal MoE LLMs** — MoE-LLaVA, CuMo, Uni-MoE-style unified audio–image–video–text MoEs; the direction is production-oriented, with routing plots as evidence *(frontier — verify current bests)*.
- **Shared + routed expert hybrids** — DeepSeekMoE-style always-on shared experts imported into vision–language stacks, with the shared expert implicitly cast as the cross-modal carrier. Whether the shared expert actually encodes modality-agnostic structure is untested *(frontier — verify)*.
- **Interpretability of expert function** — probing experts as feature dictionaries rather than as routing targets; would supply the missing causal metric. Sparse-autoencoder-style analysis applied per-expert is the obvious method and, to our knowledge, unpublished for multimodal MoE *(frontier — verify)*.
- **Modality-aware routing regularizers** — entropy, mutual-information (Mod-Squad lineage), and prior-matching losses that *target* a chosen reuse level rather than letting it emerge.

## 8. Concrete Next Experiment

**Scale.** Two pretraining runs, ~3B active / ~20B total parameters, $E = 32$ experts, $k = 2$, MoE every second FFN block, trained on 300B interleaved image–text tokens with a fixed 40% image / 60% text token ratio.

**Arms.**
- *Shared*: one expert pool, all modalities routed by a single learned router, load-balance coefficient $\alpha$ swept over $\{0.001, 0.01, 0.1\}$.
- *Control (siloed)*: experts hard-partitioned 16 image / 16 text, VLMo-style, identical active params, total params, FLOPs, data order, and steps.

**Measurement.** On held-out data, per-modality validation loss normalized by a modality-only reference run; then per-expert route-around ablation with two protocols (frozen gates and gates re-fit on 10M tokens) to bracket obstruction 2. Report $\bar{C}_\ell$ per layer and $O_\ell$ per layer, both as a function of $\alpha$.

**The deciding number.** $\min_m \Delta_m$ — the worse of the two per-modality loss improvements of shared over siloed, in nats/token, at the best $\alpha$ for each arm. If $\min_m \Delta_m \leq 0.005$ nats/token (below typical seed noise at this scale), cross-modal expert sharing buys nothing beyond parameter-count bookkeeping and the field should default to modality-siloed experts. If $\min_m \Delta_m \geq 0.02$ nats/token, sharing is real and the follow-up is whether $\bar{C}_\ell$ (not $O_\ell$) predicts it: regress $\Delta$ on both across layers and report $R^2$.

## 9. Key References

- **[Foundational]** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[SOTA]** Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, Neil Houlsby. *Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts.* NeurIPS, 2022. — arXiv:2206.02770
- **[SOTA]** Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Songhao Piao, Furu Wei. *VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts.* NeurIPS, 2022. — arXiv:2111.02358
- **[SOTA]** Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, Neil Houlsby. *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints.* ICLR, 2023. — arXiv:2212.05055
- **[Method]** Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[Method]** Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik Learned-Miller, Chuang Gan. *Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners.* CVPR, 2023. — arXiv:2212.08066
- **[Method]** Jinguo Zhu, Xizhou Zhu, Wenhai Wang, Xiaohua Wang, Hongsheng Li, Xiaogang Wang, Jifeng Dai. *Uni-Perceiver-MoE: Learning Sparse Generalist Models with Conditional MoEs.* NeurIPS, 2022. — arXiv:2206.04674
- **[Empirical]** Sheng Shen, Zhewei Yao, Chunyuan Li, Trevor Darrell, Kurt Keutzer, Yuxiong He. *Scaling Vision-Language Models with Sparse Mixture of Experts.* Findings of EMNLP, 2023. — arXiv:2303.07226
- **[Empirical]** Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, et al. *MoE-LLaVA: Mixture of Experts for Large Vision-Language Models.* 2024. — arXiv:2401.15947
- **[Survey]** Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906

## 10. Worked Example

One MoE layer, $E = 8$, $k = 1$, image and text tokens in a 50/50 held-out batch of 1M tokens each. Suppose expert $e_3$ receives 41% of image tokens and 36% of text tokens. Routing overlap for that expert is near-maximal, and this is exactly the plot that multimodal MoE papers publish.

Now ablate $e_3$ (route-around, frozen gates), and normalize each modality's loss delta by that modality's own single-modality reference loss:

| quantity | image | text |
|---|---|---|
| reference loss $L_m$ (nats/token) | 2.30 | 2.90 |
| $\delta_{e_3}^{(m)}$ raw (nats/token) | 0.052 | 1.90 |
| $\delta / L_m$ | 0.023 | 0.655 |

$C_{\ell,e_3} = 0.023/0.655 = 0.035$. Overlap says "fully shared"; causal reuse says the expert is a text expert that image tokens pass through at essentially no benefit. (These are illustrative numbers chosen to make the arithmetic concrete — no published paper reports this pair of measurements on the same expert, which is the point.)

Then re-fit the router on 10M tokens with $e_3$ removed and experts frozen. If $\delta^{(\text{image})}$ falls to 0.004 and $\delta^{(\text{text})}$ to 1.10, both numbers moved by more than the gap they were meant to establish. That is obstruction 2 in a single table: the ablation measures the router as much as the expert, and until the protocol is pinned down, no reuse number — including $C$ — is comparable across papers.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*