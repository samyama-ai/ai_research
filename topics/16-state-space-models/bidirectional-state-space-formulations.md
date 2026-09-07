---
id: 16-state-space-models/bidirectional-state-space-formulations
title: "Bidirectional and Non-Causal State-Space Formulations"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Bidirectional and Non-Causal State-Space Formulations

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/bidirectional-state-space-formulations` · **Status:** open

## 1. Problem Statement

State-space models (SSMs) are defined by a causal recurrence: the state at position $t$ depends only on positions $\le t$. Many domains — masked language modeling, image and video encoding, DNA, audio enhancement, protein structure — have no natural arrow of time, and the causal constraint is a modeling artifact rather than a requirement. The problem is to define, analyze, and evaluate SSM sequence mixers that condition on the full sequence.

Three variants, with different difficulty:

- **Method.** What is the right non-causal generalization? The default in practice is *run the SSM forward, run it backward, add*. Is that the correct closure of the operator class, or an arbitrary point in it?
- **Theory.** Does bidirectionality buy expressive power, or only a constant-factor of statistical efficiency? Causal SSMs with finite state are known to sit in $\mathsf{TC}^0$; adding a backward pass is a second $\mathsf{TC}^0$ computation, so the complexity-class answer is likely "no". The open question is the finer-grained one: for a fixed state size $N$ and depth $L_{\text{depth}}$, is there a function family separating bidirectional from causal SSMs by more than a constant factor in $N$?
- **Measurement.** A bidirectional layer, naively built, has $2\times$ the SSM parameters and $2\times$ the scan FLOPs of its causal twin. Nearly every reported gain is confounded with that. **Solving the measurement variant means: a protocol that isolates the value of non-causality from the value of extra compute.**

A solution would be (a) a characterization of the operator class that non-causal SSMs span, (b) an algorithm that realizes it at causal cost, and (c) a compute-matched benchmark showing where the gain is real.

## 2. Formal Setting

Write a sequence mixer as a matrix. For input $x \in \mathbb{R}^{L \times d}$ and a per-channel mixing matrix $M \in \mathbb{R}^{L\times L}$, one head computes $y = Mx$.

A causal SSM with state dimension $N$, matrices $(A_t, B_t, C_t)$, has
$$h_t = A_t h_{t-1} + B_t x_t,\qquad y_t = C_t^\top h_t,$$
so $M_{ij} = C_i^\top A_{i:j} B_j$ for $i \ge j$ and $0$ otherwise, where $A_{i:j} = A_i \cdots A_{j+1}$. $M$ is **lower-triangular semiseparable of order $N$**: every submatrix strictly below the diagonal has rank $\le N$ (Dao & Gu, ICML 2024).

**Naive bidirectional.** $M^{\text{bi}} = M^{\rightarrow} + \Pi M^{\leftarrow} \Pi$ with $\Pi$ the reversal permutation. This ties the diagonal (it is summed twice) and needs two independent parameter sets.

**Quasiseparable.** The closure is the class $\mathcal{Q}_N$: $M_{ij} = \bar C_i^\top \bar A_{i:j} \bar B_j$ for $i>j$, $\check C_i^\top \check A_{j:i}\check B_j$ for $i<j$, and a *free* diagonal $\delta_i$ for $i=j$. Every submatrix strictly above or strictly below the diagonal has rank $\le N$. Quasiseparable strictly contains the naive sum (which forces $\delta_i$ to be a specific function of the two branches) and admits an $O(LN)$ algorithm (Hydra, NeurIPS 2024).

**Quantities as measured.**
- Parameters $P$: counted per-layer, SSM projections included, embedding excluded.
- Training FLOPs $F = 6PT$ for $T$ tokens; report measured wall-clock separately since scans are memory-bound.
- **Bidirectional gain** $\Delta = \mathcal{M}(\text{bi}) - \mathcal{M}(\text{causal})$ under a stated matching: $\Delta_P$ (params matched), $\Delta_F$ (FLOPs matched), $\Delta_{\text{wall}}$ (wall-clock matched). These three differ and are routinely reported without saying which.
- Task metric $\mathcal{M}$: GLUE average, ImageNet top-1, MCC on Nucleotide Transformer tasks, PESQ for enhancement.

**Assumptions, and where they break.**
1. *Per-layer non-causality is what bidirectionality means.* Violated: a causal SSM stacked with any global op (pooling, one attention layer, `[CLS]`-style readout) is already non-causal end-to-end. Most "causal baselines" in vision papers are not causal models.
2. *The two directions are symmetric.* Violated for DNA (reverse-complement is the true symmetry, not reversal — Caduceus, ICML 2024) and for 2D images, where "backward" is one of many raster orders.
3. *Selectivity commutes with reversal.* Violated: in Mamba, $A_t, B_t, C_t$ are input-dependent, so the backward pass sees a different gating schedule; the operator is not the transpose of the forward one.
4. *Bidirectional layers cost $2\times$.* Approximately true in FLOPs, false in wall-clock — the reverse scan is often overlapped, measured overhead is typically $1.2$–$1.6\times$, not $2\times$.

## 3. State of the Art

**Established.**
- Semiseparable/SSD characterization of causal SSM mixers, and the $O(LN)$ quasiseparable algorithm for the bidirectional case (Dao & Gu 2024; Hwang et al., Hydra, NeurIPS 2024). This is a theorem plus an implementation, not a benchmark claim.
- Reverse-complement *equivariance* by construction in Caduceus (Schiff et al., ICML 2024) — an architectural symmetry property that holds by construction, independent of any benchmark.
- Bidirectional SSMs run at $O(L)$ and beat quadratic attention on memory at long $L$: Vision Mamba reports ~$2.8\times$ faster inference and large GPU-memory savings versus DeiT at $1248\times1248$ resolution. The asymptotics are not in doubt.

**Claimed but under-ablated.**
- Hydra reports gains over matched BERT and ViT baselines (roughly $+0.8$ GLUE average, and a multi-point ImageNet top-1 gain over ViT-B under its stated training recipe). These are the paper's own numbers; there is no independent replication at a second lab, and the ablation isolating *quasiseparability* from *the free diagonal* from *extra parameters* is partial.
- Vision Mamba's and VMamba's multi-directional scans (bi-scan, cross-scan/SS2D) are reported as necessary in each paper's ablation, but the ablations are internal and the baselines differ across papers.
- Bi-directional Mamba variants for speech enhancement and time series report gains; almost none report $\Delta_P$ and $\Delta_F$ separately.

**Counter-evidence.** MambaOut (Yu & Wang, CVPR 2025) shows plain gated-CNN blocks — no SSM, no scan — match or beat vision Mamba models on ImageNet-1K classification, while the gap persists on detection/segmentation. MambaVision (Hatamizadeh & Kautz, CVPR 2025) argues the symmetric bidirectional scan is the wrong primitive for vision and replaces it with a hybrid design. Both are direct evidence that the reported bidirectional gain in vision is at least partly a training-recipe and parameter-count artifact.

## 4. What Is Known

- **Expressivity ceiling.** Log-precision SSMs of the S4/Mamba family cannot solve $\mathsf{NC}^1$-hard problems such as word problems over $S_5$ at fixed depth; they lie in $\mathsf{TC}^0$ (Merrill, Petty, Sabharwal, ICML 2024). Two passes do not escape the class.
- **Copying/recall gap.** Transformers beat SSMs at copying with a separation that grows with input length; measured at model scales up to ~1B on synthetic copy and on natural-language recall (Jelassi et al., ICML 2024). Bidirectionality does not close this — the bottleneck is the $O(N)$ state, and running it twice gives $2N$.
- **Algorithmic cost.** Quasiseparable matrix–vector multiply is $O(LN)$ sequentially, with an associative-scan form of depth $O(\log L)$ — the same asymptotics as one causal scan (Hydra, 2024).
- **Scale of the empirical record.** Vision Mamba: ImageNet-1K, Vim-Ti at ~7M params reaching ~76% top-1 versus DeiT-Ti ~72%, with the two models differing in more than directionality. Hydra: BERT-base scale, ~110M params, C4/Wikipedia-scale pretraining. Caduceus: ~$10^5$–$10^6$ params on human-genome pretraining, outperforming a 500M-param Nucleotide Transformer on several downstream tasks. **No compute-matched bidirectional-versus-causal SSM study exists above ~1B parameters.**

## 5. What Is Not Known

- **Theoretically open.** Whether there is a function family that a depth-$L_{\text{depth}}$ quasiseparable SSM of state $N$ solves but a depth-$L_{\text{depth}}$ causal SSM of state $2N$ cannot (i.e. a separation not explained by doubling the state). No proof either way. Also open: whether the free diagonal in $\mathcal{Q}_N$ confers any representational power beyond a residual connection.
- **Empirically open.** The compute-matched scaling law. Fit $\mathcal{M}(F)$ for bidirectional and causal encoders under identical data, tokenizer, and FLOPs, at $F$ spanning $10^{19}$–$10^{21}$. Runnable today on a few hundred GPU-days; nobody has published it.
- **Methodologically blocked.** What "the causal control" means for a *non-generative* task. A causal SSM encoder with a mean-pool head already sees the whole sequence. Until the community fixes a control — causal-with-pooling? causal-with-one-attention-layer? forward-only-with-doubled-state? — $\Delta$ is not a well-defined number, and papers reporting it are not measuring the same quantity.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a non-identifiable control**. Every published bidirectional gain varies at least three things at once relative to its baseline: (i) parameter count, (ii) per-layer FLOPs, (iii) the training recipe (vision Mamba baselines are DeiT recipes; Hydra's baseline is a BERT recipe). Because the natural causal control is itself non-causal end-to-end once a pooling head is attached, there is no canonical zero point against which to measure the effect. This is not a compute problem — BERT-scale ablations are cheap — it is a definitional one, and it explains why five years of bidirectional-SSM papers have produced a monotone stream of positive results with no consensus on whether the mechanism does anything.

Secondary: selective SSMs make the forward and backward operators non-transposes, so the clean linear-algebraic story ($M$ quasiseparable) is exact only for time-invariant SSMs. For Mamba-style models the analysis is heuristic.

## 7. Current Research (as of 2026)

- **Matrix-mixer unification.** Extending the semiseparable/quasiseparable framing to other structured classes (Toeplitz, low-rank-plus-diagonal, butterfly) — Dao and Gu's groups at Princeton/CMU and collaborators. Established framing; the empirical payoff is still open.
- **Symmetry-correct non-causality.** Equivariant SSMs beyond reversal: reverse-complement for genomics, dihedral for images, permutation for sets (Cornell/Kuleshov, Stanford). *(frontier — verify)*
- **Hybrid encoders.** Replacing symmetric bi-scan with sparse global attention on top of causal scans (NVIDIA MambaVision line, and several 2025–26 vision backbones). Convergent evidence that the pure bidirectional scan is not the winning vision primitive.
- **Diffusion/masked-LM decoders.** Non-causal SSM backbones for discrete diffusion language models, where the whole sequence is visible by construction. *(frontier — verify)*
- **Negative results.** MambaOut-style deletion studies. Under-supplied relative to positive claims.

## 8. Concrete Next Experiment

**Question.** Is there a compute-matched gain from per-layer non-causality in a masked encoder?

**Scale.** Four arms, each ~110M non-embedding params, each trained on the *same* 30B tokens of a fixed corpus (e.g. C4), same tokenizer, same masking rate (30%), same LR schedule, 3 seeds. ~$2\times10^{20}$ FLOPs total; ~200 A100-days.

**Arms.**
1. **Control A (causal + pool):** forward-only Mamba-2 encoder, state $N=64$, with a mean-pool/`[CLS]` head. *This is the correct zero point.*
2. **Control B (state-matched):** forward-only, state $N=128$ — same SSM FLOPs as the bidirectional arm, still causal per layer.
3. **Treatment C (naive bi):** forward+backward, $N=64$ each, summed.
4. **Treatment D (quasiseparable):** Hydra mixer, order $N=64$, free diagonal.

Layer counts adjusted so all four match on total training FLOPs, not on depth.

**Deciding number.** $\Delta_F = \text{GLUE}_{\text{avg}}(\text{D}) - \text{GLUE}_{\text{avg}}(\text{B})$, with a seed-variance-derived confidence interval. GLUE-average seed noise at BERT-base scale is roughly $\pm 0.4$ points. **If $\Delta_F > 0.8$ points with the interval excluding zero, non-causality is real and not a compute artifact. If $|\Delta_F| < 0.4$, the published gains are parameter-count effects and the field should say so.** The secondary number, $\text{D} - \text{C}$, isolates whether quasiseparability beats the naive sum; the Hydra paper predicts it does, at roughly $+0.3$–$0.5$.

## 9. Key References

- **[Foundational]** M. Schuster, K. Paliwal. *Bidirectional Recurrent Neural Networks.* IEEE Transactions on Signal Processing, 1997.
- **[Foundational]** A. Gu, K. Goel, C. Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[SOTA / theory]** T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** S. Hwang, A. Lahoti, T. Dao, A. Gu. *Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers.* NeurIPS 2024. — arXiv:2407.09941
- **[SOTA]** L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, X. Wang. *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model.* ICML 2024. — arXiv:2401.09417
- **[SOTA]** Y. Liu et al. *VMamba: Visual State Space Model.* NeurIPS 2024. — arXiv:2401.10166
- **[SOTA]** Y. Schiff, C.-H. Kao, A. Gokaslan, T. Dao, A. Gu, V. Kuleshov. *Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling.* ICML 2024. — arXiv:2403.03234
- **[Limits]** W. Merrill, J. Petty, A. Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Limits]** S. Jelassi, D. Brandfonbrener, S. Kakade, E. Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024. — arXiv:2402.01032
- **[Negative result]** W. Yu, X. Wang. *MambaOut: Do We Really Need Mamba for Vision?* CVPR 2025. — arXiv:2405.07992
- **[Alternative design]** A. Hatamizadeh, J. Kautz. *MambaVision: A Hybrid Mamba-Transformer Vision Backbone.* CVPR 2025. — arXiv:2407.08083

## 10. Worked Example

Take $L=8$, state $N=1$, time-invariant, $A=a$, $B=C=1$. The causal mixer is $M^{\rightarrow}_{ij} = a^{\,i-j}$ for $i\ge j$. With $a=0.9$:

$$M^{\rightarrow} = \begin{pmatrix}1&&&\\0.9&1&&\\0.81&0.9&1&\\ \vdots& & &\ddots\end{pmatrix}$$

Naive bidirectional gives $M^{\text{bi}}_{ij} = a^{|i-j|}$ for $i\ne j$ and $M^{\text{bi}}_{ii}=2$. The diagonal is *forced* to $2$ — twice the off-diagonal-adjacent weight $0.9$ — purely because both scans emit their own $t=t$ term. That is a modeling artifact, not a choice. Quasiseparable frees $\delta_i$: the same operator class with one extra scalar per position.

Now the obstruction. Count parameters and FLOPs for the three arms of §8 at $d=768$, $N=64$, $L=512$:

| Arm | SSM params/layer | Scan FLOPs/layer/token | Free diagonal |
|---|---|---|---|
| Causal, $N=64$ | $\approx 2dN = 98\text{k}$ | $\approx 2dN = 98\text{k}$ | no |
| Causal, $N=128$ | $\approx 197\text{k}$ | $\approx 197\text{k}$ | no |
| Naive bi, $2\times N{=}64$ | $\approx 197\text{k}$ | $\approx 197\text{k}$ | no |
| Hydra, $\mathcal{Q}_{64}$ | $\approx 197\text{k} + d$ | $\approx 197\text{k}$ | yes ($+768$) |

The naive bidirectional arm and the doubled-state causal arm are **identical in parameters and FLOPs to within 0.4%**. Every published comparison of "bidirectional Mamba" against "Mamba" instead compares row 3 against row 1 — a 2× compute gap. A $+2$-point ImageNet or $+0.8$-point GLUE gain measured that way is indistinguishable from the gain you would get by simply doubling $N$ in a causal model, which costs nothing conceptually and requires no reverse scan. Until someone runs row 3 against row 2, the field does not know whether non-causality has ever been shown to do anything at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*