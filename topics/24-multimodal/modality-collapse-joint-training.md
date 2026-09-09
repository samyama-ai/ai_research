---
id: 24-multimodal/modality-collapse-joint-training
title: "Modality Collapse in Joint Training"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Modality Collapse in Joint Training

> **Topic:** Multimodal Models · **ID:** `24-multimodal/modality-collapse-joint-training` · **Status:** open

## 1. Problem Statement

Train one network on paired inputs from $M$ modalities with a single joint objective. Often the trained model behaves as if a subset of modalities were absent: the encoder for modality $m$ carries little task-relevant information, ablating $m$ at test time barely changes the loss, and the joint model can be *worse* than the best model trained on one modality alone. This is **modality collapse** (also called modality laziness, modality imbalance, or greedy learning).

Three distinct problems get conflated:

- **Measurement.** Given a trained checkpoint, decide whether modality $m$ has collapsed, and quantify by how much. There is no agreed estimator; input ablation, encoder gradient norms, and linear probes disagree on the same checkpoint.
- **Method.** Produce a training procedure whose joint model dominates every unimodal model and every post-hoc ensemble of unimodal models, on held-out data, without per-dataset tuning.
- **Theory.** Prove for a stated data model and optimizer whether collapse is a property of the *optimization path* (avoidable by reweighting) or of the *loss landscape / statistics* (the joint optimum genuinely underuses $m$).

Solved would mean: an estimator that is invariant to encoder reparameterization and agrees across seeds, plus a method that beats the "late-fusion ensemble of separately trained unimodal encoders" control at matched compute on a benchmark suite the method was not tuned on.

## 2. Formal Setting

Data $(x^{(1)},\dots,x^{(M)}, y) \sim \mathcal{D}$. Encoders $f_m(\cdot;\theta_m) \in \mathbb{R}^{d}$, fusion head $g(\cdot;\phi)$, prediction $\hat y = g(f_1,\dots,f_M)$, joint risk

$$\mathcal{L}(\theta,\phi) = \mathbb{E}_{\mathcal D}\big[\ell\big(g(f_1(x^{(1)}),\dots,f_M(x^{(M)})),\,y\big)\big].$$

**Measured quantities.**

- *Ablation gap.* $\Delta_m = \mathcal{L}(\text{replace } x^{(m)} \text{ by } \tilde x^{(m)}) - \mathcal{L}$, with $\tilde x^{(m)}$ resampled from the marginal (not zeroed — zeroing puts the encoder off-distribution and inflates $\Delta_m$).
- *Unimodal reference.* $\mathcal{L}^\star_m = \min_{\theta_m,\phi_m} \mathbb{E}[\ell(g_m(f_m(x^{(m)})),y)]$, trained from scratch at the same token/step budget.
- *Collapse indicator.* $C_m = 1$ if $\Delta_m < \varepsilon$ while $\mathcal{L}^\star_m \ll \mathcal{L}^\star_{\text{chance}}$ — modality $m$ is predictive alone but unused jointly.
- *Competition.* $\Gamma = \mathcal{L} - \min_m \mathcal{L}^\star_m$. $\Gamma > 0$ is the strong failure: joint training loses to a single modality.
- *Encoder logit contribution.* for a linear head $\phi = [W_1,\dots,W_M]$, $s_m = \mathbb{E}\|W_m f_m(x^{(m)})\|$, and its gradient analogue $\mathbb{E}\|\nabla_{\theta_m}\ell\|$.
- *Conditional utilization rate* (Wu et al., ICML 2022): accuracy gained by adding $m$ to a model already using the rest, measured by retraining a head on frozen features.

**Assumptions and where they break.** (i) $s_m$ and $\|\nabla_{\theta_m}\ell\|$ are treated as usage measures — but both are scale-covariant: multiplying $f_m$ by $c$ and $W_m$ by $1/c$ leaves the function unchanged and changes both, so **neither is a well-defined functional of the model**. Batch/layer norm partly but not fully removes this. (ii) $\Delta_m$ assumes the head is optimal for the ablated input; it is not, so $\Delta_m$ overstates usage. (iii) Matched-compute unimodal references are usually not run; papers compare against unimodal baselines trained to convergence with different schedules. (iv) Late-fusion is assumed representative; early-fusion and token-concatenation transformers have different failure modes.

## 3. State of the Art

**Established (ablated, reproduced across labs).**

- **Gradient-Blending** (Wang, Tran, Feiszli, CVPR 2020): reweight per-modality and joint losses by an estimated overfitting-to-generalization ratio. First clean demonstration that naive audio-visual late fusion underperforms the video-only model on Kinetics, and that reweighting fixes it.
- **OGM-GE** (Peng et al., CVPR 2022): monitor per-modality logit contribution ratio, damp the dominant modality's gradient, add Gaussian noise for enhancement. Reproduced on CREMA-D, Kinetics-Sounds, VGGSound by several later groups.
- **Greedy-learning diagnosis** (Wu, Jastrzębski, Cho, Geras, ICML 2022): conditional utilization rate shows networks lock onto the faster-learning modality early; per-modality learning-rate rebalancing (Guided Learning) recovers most of the gap.
- **Theory** (Huang et al., ICML 2022): in a two-modality late-fusion model with a stated signal-plus-noise data generator, gradient descent provably enters a winner-take-all regime — one encoder learns the feature, the other stays near its initialization — and the winner is set by early signal-to-noise, not by asymptotic usefulness.

**Claimed but not adequately ablated.**

- Most rebalancing methods (PMR, CVPR 2023; MLA alternating unimodal adaptation, CVPR 2024; diagnose-and-relearn, ECCV 2024) report gains of 1–5 points on CREMA-D / Kinetics-Sounds / UCF101-class benchmarks. The **missing control is almost always the same**: a late-fusion ensemble of independently trained unimodal encoders at equal total compute. Where that control has been run, it is competitive.
- Generative side: multimodal VAEs. Impartial optimization for modality collapse (Javaloy, Meghdadi, Valera, ICML 2022) treats collapse as gradient conflict; Daunhawer et al. (ICLR 2022) argue the mixture-based objectives have a *generative-quality* limitation independent of optimization. These two explanations have not been separated experimentally.
- Large-scale mixed-modal LLMs: reported instabilities and modality-specific loss plateaus in early-fusion training (e.g. Chameleon, 2024) are **benchmark/loss-curve observations**, not controlled collapse measurements.

## 4. What Is Known

- Naive joint training can lose to unimodal. Wang et al. (CVPR 2020) report audio-RGB late fusion on Kinetics below the RGB-only network; gradient blending recovers several points. Scale: ~240k-clip video classification, ResNet-class encoders.
- Ordering effects are large at small scale. On CREMA-D (~7.4k clips, 6 classes), concat-fusion audio+video baselines sit in the low 60s % accuracy and rebalancing methods report high 60s to ~80% depending on backbone — a spread larger than most claimed method deltas, which is why cross-paper number comparison is unreliable.
- The dominant modality is dataset-determined, not modality-determined: audio dominates on CREMA-D, video dominates on Kinetics-Sounds, with the same architecture.
- Mixed-modal scaling laws (Aghajanyan et al., 2023) fit a competition/synergy term whose sign flips with model scale and mixing ratio: at small scale modalities compete, at larger scale the fitted interaction becomes less negative. Measured on text+image autoregressive models up to ~7B parameters.
- Contrastive two-tower training does not collapse in this sense but exhibits a **modality gap**: image and text embeddings occupy disjoint cones, present at initialization and preserved by the contrastive loss (Liang et al., NeurIPS 2022). This is a distinct phenomenon and should not be measured with the same estimators.

## 5. What Is Not Known

- **Methodologically blocked.** There is no reparameterization-invariant usage measure. Gradient-norm and logit-contribution estimators change under a rescaling that leaves the function identical, so a large fraction of the rebalancing literature optimizes a quantity that is not a property of the model. This blocks even stating "method A reduces collapse more than method B" cleanly.
- **Empirically open.** Whether any rebalancing method beats a matched-compute unimodal ensemble across ≥5 datasets it was not tuned on. Runnable today on 8 GPUs; nobody has published the full grid.
- **Empirically open.** Whether collapse persists at frontier scale in early-fusion autoregressive models, or is a small-data/short-schedule artifact. The scaling-law evidence points to weakening competition with scale but does not measure per-modality usage directly.
- **Theoretically open.** Whether collapse is optimization-path or landscape. Huang et al. prove a winner-take-all dynamic in a specific two-layer model; no result characterizes when the *global* joint optimum underuses a modality, nor whether any per-modality reweighting schedule provably escapes the bad basin for a general data model.

## 6. Why It Is Hard

Three concrete obstructions, in order of severity.

1. **Non-identifiability of the measurement.** Usage is not identifiable from weights: encoder scale and head scale trade off exactly. Any estimator built on gradient magnitude or logit norm measures a gauge choice, not a function. Ablation-based estimators are gauge-invariant but confounded by head suboptimality under ablation.
2. **Confounded control arm.** Joint models are compared to unimodal models trained with different schedules, augmentations, and epoch counts. A 2-point joint gain and a 2-point schedule artifact are indistinguishable in the published numbers.
3. **Absent ground truth.** For real data nobody knows the true per-modality Bayes-optimal contribution, so "the model should use audio more" is unfalsifiable. Synthetic data with known contributions exists but does not reproduce the encoder-capacity mismatch that likely drives collapse in practice.

## 7. Current Research (as of 2026)

- Gradient-surgery and reweighting descendants (Renmin University / Di Hu's group and collaborators; on-the-fly modulation, diagnose-and-relearn). Mature; diminishing returns on the standard four benchmarks.
- Alternating or decoupled unimodal training with a shared head (MLA line, UNC/Mohit Bansal and collaborators) — reframes the problem as avoiding joint optimization rather than fixing it.
- Mixture-of-experts and modality-routed early fusion in native-multimodal LLMs, where per-modality experts sidestep parameter competition. Evidence is loss-curve level, not usage level. *(frontier — verify)*
- Information-theoretic usage estimators (partial information decomposition: redundancy / uniqueness / synergy) applied to multimodal models — the most promising route to a gauge-invariant measurement, currently limited by estimator variance in high dimension. *(frontier — verify)*
- Modality-collapse analysis in multimodal generative models (VAEs, diffusion with multiple conditioning streams), still largely separate from the discriminative literature.

## 8. Concrete Next Experiment

**Question decided:** does any published rebalancing method beat matched-compute decoupled training, once usage is measured gauge-invariantly?

- **Scale.** Six datasets (CREMA-D, Kinetics-Sounds, VGGSound-subset, UCF101 RGB+flow, MM-IMDb image+text, Food-101 image+text). Fixed backbones (ResNet-18 per stream, BERT-base for text). 5 seeds each. About 180 runs, ~8 A100-days total.
- **Control arm (the one usually missing).** Train each modality's encoder separately to the *same total FLOP budget* as the joint model, freeze, then fit a late-fusion head on the concatenated features. Call its accuracy $A_{\text{ens}}$.
- **Treatment arms.** Naive joint concat-fusion; gradient blending; OGM-GE; alternating unimodal adaptation.
- **Measurement.** Report $\Delta_m$ with marginal resampling *and head refit after ablation* (this is the gauge-invariant version), not gradient norms.
- **Deciding number.** $\delta = \mathbb{E}_{\text{datasets}}[A_{\text{method}} - A_{\text{ens}}]$ with a paired bootstrap 95% CI over the 6 datasets × 5 seeds. If no method achieves $\delta > 0$ with a CI excluding zero, the rebalancing literature is measuring schedule effects and the problem should be restated as "when is joint training worth it at all". If some method achieves $\delta > 1.0$ point, joint training has a real, method-recoverable advantage and the theory question becomes the live one.

## 9. Key References

- **[Foundational]** Weiyao Wang, Du Tran, Matt Feiszli. *What Makes Training Multi-Modal Classification Networks Hard?* CVPR, 2020. — arXiv:1905.12681
- **[Theory]** Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, Longbo Huang. *Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning? (Provably).* ICML, 2022. — arXiv:2203.12221
- **[SOTA]** Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, Di Hu. *Balanced Multimodal Learning via On-the-fly Gradient Modulation.* CVPR, 2022. — arXiv:2203.15332
- **[Diagnosis]** Nan Wu, Stanisław Jastrzębski, Kyunghyun Cho, Krzysztof J. Geras. *Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks.* ICML, 2022. — arXiv:2202.05306
- **[Generative]** Adrián Javaloy, Maryam Meghdadi, Isabel Valera. *Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization.* ICML, 2022. — arXiv:2206.04496
- **[Generative]** Imant Daunhawer, Thomas M. Sutter, Kieran Chin-Cheong, Emanuele Palumbo, Julia E. Vogt. *On the Limitations of Multimodal VAEs.* ICLR, 2022. — arXiv:2110.04121
- **[Scale]** Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, Luke Zettlemoyer. *Scaling Laws for Generative Mixed-Modal Language Models.* ICML, 2023. — arXiv:2301.03728
- **[Related]** Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Zou. *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS, 2022. — arXiv:2203.02053
- **[SOTA]** Xiaohui Zhang, Jaehong Yoon, Mohit Bansal, Huaxiu Yao. *Multimodal Representation Learning by Alternating Unimodal Adaptation.* CVPR, 2024. — arXiv:2311.10707
- **[Survey]** Tadas Baltrušaitis, Chaitanya Ahuja, Louis-Philippe Morency. *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE TPAMI, 2019.

## 10. Worked Example

CREMA-D, 6-way emotion classification, audio + face video, ResNet-18 per stream, linear concat head.

Suppose a run gives audio-only 61%, video-only 55%, joint concat-fusion 63%. The joint model's measured logit contributions are $s_{\text{audio}} = 8.4$, $s_{\text{video}} = 1.9$, ratio $4.4$. The standard reading: video has collapsed; damp audio gradients.

Now apply the gauge transform. Rescale the video encoder's final BN-free projection by $c = 4.4$ and its head block $W_{\text{video}}$ by $1/4.4$. The function is bit-identical — same predictions, same loss, same accuracy — and now $s_{\text{video}} = 8.4$, ratio $1.0$. **The collapse metric reads "balanced" on a model that has not changed.** OGM-GE's modulation coefficient, computed from that ratio, would apply a different gradient scaling to the same function.

Contrast with the gauge-invariant measurement. Resample video from the marginal, refit the head on frozen features: joint accuracy falls 63% → 62.1%, so $\Delta_{\text{video}} = 0.9$ points. Resample audio the same way: 63% → 55.8%, $\Delta_{\text{audio}} = 7.2$ points. Video really is contributing under 1 point.

Then the control: independently trained audio and video encoders at the same total FLOPs, frozen, late-fusion head — 63.4%. The ensemble matches or beats the joint model, and beats it without any rebalancing. $\Gamma = \mathcal{L} - \min_m\mathcal{L}^\star_m$ is near zero, so this run is not even the strong failure case.

The obstruction is visible in three lines: the field's most-used collapse metric is not a function of the model, the gauge-invariant replacement says the effect here is under one accuracy point, and the control arm that would tell you whether joint training helps at all was not run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*