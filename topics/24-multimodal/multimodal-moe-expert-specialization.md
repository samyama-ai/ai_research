---
id: 24-multimodal/multimodal-moe-expert-specialization
title: "Modality-Specific Sparsity in Multimodal Mixture-of-Experts"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Modality-Specific Sparsity in Multimodal Mixture-of-Experts

> **Topic:** Multimodal Models · **ID:** `24-multimodal/multimodal-moe-expert-specialization` · **Status:** empirically-open

## 1. Problem Statement

A sparse mixture-of-experts (MoE) layer routes each token to a small subset of parameter blocks. In a multimodal model the token stream carries a modality label (image patch, text token, audio frame). The question: **does the router learn modality-specific experts, should it, and does forcing or forbidding that specialization change loss at matched activated FLOPs?**

Three variants, routinely conflated:

- **Measurement.** Given a trained multimodal MoE, produce a number that says how modality-specialized its experts are, invariant to expert permutation and to modality base rates. No agreed metric exists.
- **Method.** Design a routing objective that gets the best loss-per-activated-FLOP. Modality-specific sparsity is a design knob: hard partition (one expert set per modality), soft (shared router, emergent split), or none.
- **Theory.** Under what data distribution is a modality-partitioned router optimal for a fixed capacity budget? Unaddressed.

Solved means: a permutation- and base-rate-invariant specialization measure with a *causal* validation (ablating a "text expert" hurts text loss and not image loss), plus a controlled scaling comparison of partitioned vs. emergent routing at matched activated parameters.

## 2. Formal Setting

Token $x_t$ with modality label $m_t \in \mathcal{M} = \{v, \ell\}$. An MoE layer has $E$ experts $\{f_1,\dots,f_E\}$, router $g(x_t) \in \Delta^{E-1}$, top-$k$ gate. Assignment indicator $A_{t,e} = \mathbb{1}[e \in \text{top-}k(g(x_t))]$.

**Routing mass.** Measured by accumulating counters over a held-out set $\mathcal{D}$ at inference, per layer:
$$p(e \mid m) = \frac{\sum_{t: m_t = m} A_{t,e}}{k \cdot |\{t : m_t = m\}|}, \qquad \pi_m = \frac{|\{t: m_t = m\}|}{|\mathcal{D}|}.$$

**Correlational specialization.** Normalized mutual information between modality and expert:
$$S_{\mathrm{NMI}} = \frac{I(M; E)}{\min(H(M), H(E))} \in [0,1].$$
$S_{\mathrm{NMI}}$ is permutation-invariant, unlike raw per-expert purity $\max_m p(m \mid e)$, which is dominated by $\pi_m$ when modalities are imbalanced.

**Causal specialization.** For expert $e$, ablate it (replace output with zero, or reroute to the next-ranked expert) and measure cross-entropy change on each modality:
$$\delta_m(e) = \mathcal{L}_m(\text{ablate } e) - \mathcal{L}_m, \qquad S_{\mathrm{causal}} = \frac{1}{E}\sum_e \frac{|\delta_v(e) - \delta_\ell(e)|}{\delta_v(e) + \delta_\ell(e)}.$$
Units: nats/token. This is the quantity that matters; $S_{\mathrm{NMI}}$ is a proxy.

**The objective.** Standard training minimizes $\mathcal{L} + \alpha \mathcal{L}_{\mathrm{bal}} + \beta \mathcal{L}_z$ where the Switch-style balance loss (Fedus et al., JMLR 2022) is
$$\mathcal{L}_{\mathrm{bal}} = E \sum_{e=1}^{E} \Big(\tfrac{1}{|\mathcal{D}_B|}\textstyle\sum_t A_{t,e}\Big)\Big(\tfrac{1}{|\mathcal{D}_B|}\sum_t g_e(x_t)\Big),$$
computed over a batch $\mathcal{D}_B$ **pooled across modalities**.

Assumptions, and which fail:

1. *Modality labels are clean.* Fails: OCR-heavy image patches, rendered text, and speech transcripts are semantically cross-modal.
2. *Token counts are comparable across modalities.* Badly violated — LLaVA-1.5 emits 576 image tokens per image against instructions of tens of tokens (Liu et al., CVPR 2024).
3. *Experts are exchangeable, so the router is identified up to permutation.* Fails at capacity limits: token dropping makes assignment order-dependent within a batch.
4. *Ablation is a valid causal probe.* Partially fails — zeroing an expert is off-distribution for the downstream layer norm; rerouting to rank-$(k{+}1)$ is the better control.

## 3. State of the Art

**Emergent routing (established).** LIMoE (Mustafa et al., NeurIPS 2022) trains a single one-tower MoE on image–text contrastive data and reports experts that are near-exclusively image or text, plus mixed experts. Establishing this required two new auxiliary losses (local and global entropy losses) on top of standard balancing; without them the model collapses to modality-imbalanced routing or drops one modality's tokens entirely. This is the strongest evidence that emergent modality sparsity is *achievable* — and simultaneously that it is not free.

**Hard partition (established, but not ablated against emergent at scale).** VLMo (Bao et al., NeurIPS 2022) uses mixture-of-modality-experts: a vision FFN, a language FFN, and a vision-language FFN, selected by modality label with no learned router. Effective, but the paper does not run a matched-FLOP comparison against a learned router.

**Benchmark-number-only claims.** MoE-LLaVA (Lin et al., 2024) reports a sparse VLM with ~3B activated parameters matching LLaVA-1.5-7B on several visual benchmarks and beating it on POPE hallucination. Aria (Li et al., 2024; 25.3B total / 3.9B activated) and DeepSeek-VL2 report expert-specialization figures. In all three, specialization is shown as routing-distribution plots on benchmark data; none reports a causal ablation or a matched-FLOP dense/partitioned control. Treat these as benchmark numbers, not ablations.

**Counter-evidence from text-only MoE.** Mixtral 8x7B (Jiang et al., 2024) analyzes router assignments across Pile subsets and finds **no** clear domain specialization; what it does find is syntactic and positional structure, including consecutive-token repetition well above chance. If domain does not induce specialization in text, modality may be doing less work than the plots suggest.

## 4. What Is Known

- **Sparse routing scales.** V-MoE (Riquelme et al., NeurIPS 2021) reaches 90.35% ImageNet top-1 at 15B total parameters with sparse activation; Switch Transformer (Fedus et al., JMLR 2022) reports ~7x pretraining speedup at matched FLOPs vs. T5-Base. Scale: 1B–1.6T parameters.
- **LIMoE-H/14 reaches 84.1% zero-shot ImageNet top-1**, one-tower, trained from scratch — competitive with two-tower CLIP-style baselines of the era. Scale: ~5.6B total parameters.
- **Balancing is load-bearing.** LIMoE's entropy losses are required for stability; removing them degrades or collapses training. Reproduced qualitatively in expert-choice routing (Zhou et al., NeurIPS 2022), where per-expert token selection removes dropping entirely and gives >2x convergence speedup at matched compute in the text setting.
- **Fine-grained experts increase measured specialization.** DeepSeekMoE (Dai et al., ACL 2024) splits experts and isolates shared experts, reporting higher expert redundancy-free specialization at 2B and 16B scale — text-only.
- **Router choice matters less than expected in vision.** "Routers in Vision Mixture of Experts: An Empirical Study" (Liu et al., TMLR 2024) finds token-choice and expert-choice routers close in quality once capacity is matched.

## 5. What Is Not Known

- **Methodologically blocked:** what "modality specialization" *means*. Published figures report per-expert modality fractions, which are confounded by $\pi_m$. No paper reports $S_{\mathrm{causal}}$ for a multimodal MoE. Without a causal probe, a "vision expert" may simply be an expert that receives many tokens.
- **Empirically open:** whether hard modality partition (VLMo-style) beats emergent routing at matched activated FLOPs, matched total parameters, and matched data. The experiment is runnable at 1–3B total parameters on ~100 GPUs. Nobody has published it with both arms.
- **Empirically open:** whether measured specialization survives modality-conditional load balancing (balance computed within each modality rather than pooled). Currently confounded, see §6.
- **Theoretically open:** no result gives conditions on a mixture data distribution under which the compute-optimal expert partition aligns with modality. The MoE approximation-theory literature does not treat labeled sub-distributions with unequal token counts.

## 6. Why It Is Hard

**The obstruction is that the load-balancing loss and the modality token-count imbalance jointly forbid the outcome being measured.** $\mathcal{L}_{\mathrm{bal}}$ is computed over pooled batches and pushes every expert toward $1/E$ of all tokens. If modality $\ell$ supplies less than $1/E$ of tokens, *no* expert can specialize to $\ell$ without incurring balance penalty. So the measurement — routing purity — is a function of the auxiliary-loss coefficient $\alpha$ and the vision/text token ratio, not only of what the model "wants". Published purity plots do not report $\alpha$ or $\pi_m$ alongside, making them non-comparable across papers.

Secondary: expert permutation symmetry means two runs from different seeds produce incomparable per-expert statistics; only permutation-invariant aggregates ($S_{\mathrm{NMI}}$, $S_{\mathrm{causal}}$) are stable, and neither is standard.

## 7. Current Research (as of 2026)

- **Fine-grained + shared-expert architectures carried into VLMs** — DeepSeek-VL2, Aria, Uni-MoE (Li et al., IEEE TPAMI 2025). Direction: many small experts plus always-on shared experts, so modality-general computation is factored out and residual experts can specialize. *(frontier — verify the specialization claims; they are routing plots.)*
- **Modality-aware auxiliary losses** — per-modality balancing and entropy regularizers, descendants of LIMoE. Google DeepMind, and several academic groups. *(frontier — verify)*
- **Router interpretability** — extending Mixtral-style assignment analysis to vision-language streams, including positional and OCR-token effects. *(frontier — verify)*
- **Upcycling** — converting dense VLMs into MoEs by expert cloning, which sets a strong specialization prior at initialization and is cheap enough to run the controls this problem needs.

## 8. Concrete Next Experiment

**Scale.** Decoder-only multimodal MoE: 24 layers, $E = 8$, top-$k = 2$, ~2.8B total / ~0.8B activated parameters. 60B tokens of interleaved image–text. ~64 H100-days per arm; four arms.

**Arms.**
- **A (control):** standard pooled Switch balance loss, $\alpha = 0.01$.
- **B:** modality-conditional balance — $\mathcal{L}_{\mathrm{bal}}$ computed separately within image tokens and within text tokens, then summed.
- **C:** hard partition, VLMo-style, 4 vision experts / 4 text experts, no learned router.
- **D:** no balance loss, z-loss only.

All arms matched on activated FLOPs/token, total parameters, data order, and seed count ($n=3$).

**Reported quantities.** $S_{\mathrm{NMI}}$, $S_{\mathrm{causal}}$ (rank-$(k{+}1)$ reroute ablation), per-modality validation cross-entropy.

**The deciding number.** Image-token validation cross-entropy of **B minus A**, nats/token, at equal activated FLOPs. If $\Delta \le 0$ while $S_{\mathrm{causal}}$ rises from A's value to $>0.3$, then modality specialization is *free* and was suppressed by the pooled balance loss — pooled balancing is a bug in every multimodal MoE recipe. If $\Delta \ge +0.02$, modality mixing buys real capacity sharing and hard partitions (C) should lose too; the field's current default is correct and the specialization plots are cosmetic.

## 9. Key References

- **[Foundational]** Noam Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23, 2022. — arXiv:2101.03961
- **[Foundational]** Dmitry Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[SOTA]** Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, Neil Houlsby. *Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts.* NeurIPS, 2022. — arXiv:2206.02770
- **[SOTA]** Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Furu Wei. *VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts.* NeurIPS, 2022. — arXiv:2111.02358
- **[SOTA]** Carlos Riquelme et al. *Scaling Vision with Sparse Mixture of Experts.* NeurIPS, 2021. — arXiv:2106.05974
- **[SOTA]** Yanqi Zhou et al. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[SOTA]** Damai Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[Empirical]** Albert Q. Jiang et al. *Mixtral of Experts.* Preprint, 2024. — arXiv:2401.04088
- **[Empirical]** Bin Lin et al. *MoE-LLaVA: Mixture of Experts for Large Vision-Language Models.* Preprint, 2024. — arXiv:2401.15947
- **[Survey]** Tianlin Liu, Mathieu Blondel, Carlos Riquelme, Joan Puigcerver. *Routers in Vision Mixture of Experts: An Empirical Study.* TMLR, 2024.

## 10. Worked Example

Take a LLaVA-1.5-style stream: 576 image tokens per image, average instruction + response ≈ 64 text tokens. Then
$$\pi_v = \frac{576}{640} = 0.90, \qquad \pi_\ell = 0.10.$$

With $E = 8$ experts, the pooled balance loss is minimized when each expert receives $1/8 = 0.125$ of routed token-slots. A **text-pure** expert can receive at most all text tokens, $0.10 < 0.125$. So a text-pure expert is structurally under-loaded, and the balance term penalizes it — before any consideration of what would lower cross-entropy. The optimum of the auxiliary loss requires every expert to absorb image tokens; the maximum achievable $p(\ell \mid e)$ under exact balance is $0.10/0.125 = 0.80$, and only for one expert, with the other seven at $p(\ell\mid e)=0$.

Now compute the metric that gets published. Perfect balance with maximal text concentration gives $H(M) = 0.325$ nats, and $I(M;E) = 0.125 \cdot D_{\mathrm{KL}}(0.8\|0.1) \approx 0.125 \times 1.32 = 0.165$ nats, so $S_{\mathrm{NMI}} \approx 0.51$. A figure captioned "experts specialize by modality" is compatible with a router that is doing nothing more than satisfying the auxiliary loss.

Change one number — pool 4 images per sample instead of 1, so $\pi_\ell = 0.027$ — and the ceiling drops to $p(\ell\mid e) \le 0.22$, $S_{\mathrm{NMI}} \le 0.14$. Same model, same router behaviour, specialization "disappears". **The published measure moves with the image-to-text token ratio, which papers do not report.** That is the obstruction, and it is why §8's deciding number is a loss difference under modality-conditional balancing rather than another purity plot.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*