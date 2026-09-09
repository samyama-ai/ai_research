---
id: 02-attention/gqa-quality-loss-scaling
title: "Grouped-Query Attention Quality Loss Scaling"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Grouped-Query Attention Quality Loss Scaling

> **Topic:** Attention Mechanisms · **ID:** `02-attention/gqa-quality-loss-scaling` · **Status:** empirically-open

## 1. Problem Statement

Grouped-query attention (GQA) shares one key/value head across a group of query heads, cutting the KV cache by the sharing factor $H/G$. Every frontier open-weight model since Llama 2 70B uses it, almost always with $G=8$. Nobody has published the curve that would justify that choice.

The problem: **determine how the quality cost of KV-head sharing scales with model size, training tokens, and context length, at a matched budget.**

- **Measurement variant.** Given a fixed training recipe, measure $\Delta(N, D, G)$ — the excess validation loss of a $G$-group model over the $G=H$ (multi-head) model — under a stated matching convention (iso-parameter, iso-FLOP, or iso-KV-cache). These three conventions give different answers and are routinely conflated.
- **Method variant.** Find the allocation of a fixed serving budget across $(L, H, G, d_h)$ that minimises loss. GQA is one point in that space; cross-layer sharing and latent compression are others.
- **Theory variant.** Prove whether sharing KV heads costs representational capacity that additional width or depth cannot recover, or whether the multi-head KV budget is asymptotically redundant.

A solution to the measurement variant is a fitted law with error bars that predicts, out of sample, the loss penalty of $G=8$ at a scale not used to fit it. Nothing published meets that bar.

## 2. Formal Setting

A decoder layer has $H$ query heads, $G$ KV groups with $G \mid H$, head dimension $d_h$, model width $d = H d_h$, and $L$ layers. Group $g$ serves query heads $\{h : \lceil hG/H \rceil = g\}$. $G=H$ is multi-head attention (MHA); $G=1$ is multi-query attention (MQA).

**Measured quantities.**

- *KV cache bytes per token*, the quantity actually paid for at serving time:
$$M_{\text{kv}} = 2\,L\,G\,d_h\,b$$
with $b$ bytes per element ($b=2$ for fp16/bf16). Measure it as peak allocator bytes under a fixed batch and sequence length, not from the formula — paged allocators and quantised caches diverge from it.
- *Parameter count*: attention contributes $L(2Hd_hd + 2Gd_hd)$ weights, so shrinking $G$ removes parameters. Report the total from the checkpoint, not the nominal size class.
- *Quality*: token-averaged validation cross-entropy $\mathcal{L}$ on a held-out corpus disjoint from training, in nats/token, reported as mean over $\geq 3$ seeds with a seed standard deviation. Downstream benchmark averages are a secondary readout and are noisier by roughly an order of magnitude.
- *The penalty*, defined per matching convention $c \in \{\text{param}, \text{flop}, \text{kv}\}$:
$$\Delta_c(N, D, G) \;=\; \mathcal{L}_c(N, D, G) \;-\; \mathcal{L}_c(N, D, H)$$
where under $c=\text{param}$ the parameters freed by sharing are returned to the FFN so both arms have equal $N$.

**Candidate law.** The natural parametric form, by analogy with Chinchilla-style fits (Hoffmann et al., 2022):
$$\Delta_{\text{param}}(N, D, G) \;\approx\; A \, N^{-\alpha} D^{-\gamma} \left(\log_2 \tfrac{H}{G}\right)^{\beta}$$
The decisive parameter is $\alpha$. If $\alpha > 0$, the GQA penalty vanishes at scale and $G=8$ is safe by extrapolation. If $\alpha \le 0$ it does not, and current practice is borrowing quality from larger models.

**Assumptions, and which are violated.**

- *Separability of $N$ and $G$ effects* — assumed, untested; the sharing penalty plausibly interacts with $d_h$ and $L$.
- *Iso-parameter matching is achievable* — violated in practice. Published GQA models redistribute nothing; the freed parameters are simply absent.
- *Validation loss ranks architectures the same way downstream tasks do* — known to be violated. Tay et al. (2022) show architecture rankings on upstream perplexity do not transfer to downstream fine-tuning quality, and reverse with scale.
- *Fixed context length* — violated. The penalty is measured at 2–8K contexts and applied at 128K+, where attention entropy and retrieval behaviour differ.

## 3. State of the Art

**Established.** Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (EMNLP 2023, arXiv:2305.13245) introduced GQA and the uptraining recipe: convert an MHA checkpoint by mean-pooling KV heads within a group, then continue pre-training for a small fraction $\alpha \approx 0.05$ of the original compute. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (arXiv:1911.02150, 2019) is the MQA antecedent.

**Claimed but unablated.** The near-parity of GQA-8 with MHA rests on a single T5-XXL uptraining comparison at one scale, one seed, and one task family (summarisation/QA). It is a benchmark number, not an ablation: no seed variance, no from-scratch arm, no parameter-matched arm, no scale sweep. Every subsequent adoption — Llama 2 70B, Llama 3 (all sizes), Mistral 7B, Qwen2, Gemma 2 — cites this result and none re-ran it.

**Genuine architecture sweeps.** Brandon et al., *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention* (arXiv:2405.12981, 2024) is the closest thing to a controlled study: 1B-parameter models trained from scratch, sweeping $G$ and cross-layer KV sharing, plotting accuracy against KV-cache bytes rather than against $G$. It establishes a Pareto frontier at one scale.

**Competing designs.** DeepSeek-V2 (arXiv:2405.04434, 2024) reports that multi-head latent attention (MLA), which compresses KV into a low-rank latent, beats MHA quality with a smaller cache. The comparison is against internally trained baselines; the claim that MLA dominates GQA at matched budget across scales is unablated by third parties.

## 4. What Is Known

- **Uptraining works cheaply.** Converting T5-XXL (11B) MHA to GQA-8 and continuing for 5% of pre-training compute recovers essentially all quality on the reported average (≈47.1 vs ≈47.2 for MHA), while inference time per sample drops from ≈1.5s to ≈0.3s — close to the MQA figure (Ainslie et al., 2023, T5-XXL scale).
- **MQA ($G=1$) does degrade measurably.** The same study puts MQA-XXL below MHA-XXL by roughly 0.6 points on the same average, and MQA is known to be unstable in long-context training. The penalty is not zero at the extreme.
- **KV cache dominates serving memory at long context.** For Llama-3-8B at 8K context, the KV cache is ~1 GiB per sequence against 16 GiB of bf16 weights; at $G=32$ it would be ~4 GiB (arithmetic from the published config, §10).
- **Architecture rankings move with scale.** Tay et al., *Scaling Laws vs Model Architectures* (arXiv:2207.10551, 2022), across ten architectures and multiple sizes: upstream perplexity ordering does not predict downstream ordering, and some architectures cross over. This is the strongest evidence that a single-scale GQA result cannot be extrapolated.
- **Heads have a low-rank bottleneck.** Bhojanapalli et al., *Low-Rank Bottleneck in Multi-head Attention Models* (ICML 2020) show per-head expressivity is capped by $d_h$ independently of $H$ — relevant because GQA changes the ratio of query heads to KV subspaces without changing $d_h$.

## 5. What Is Not Known

- **Empirically open (primary).** The sign of $\alpha$ — whether the sharing penalty grows, shrinks, or is flat in $N$. The experiment is a two-scale, five-value sweep, runnable for roughly $10^{22}$ FLOPs, well within a mid-sized academic or industry cluster. Nobody has published it.
- **Empirically open.** Whether $G=8$ is optimal or merely inherited. No paper compares $G \in \{2,4,8,16\}$ at matched KV budget across more than one scale.
- **Empirically open.** Whether the penalty grows with context length. All controlled measurements are at $\le 8$K; deployment is at 128K–1M.
- **Methodologically blocked.** The matching convention. There is no agreed answer to "what is held fixed when comparing GQA to MHA", so published comparisons are not commensurable — some hold width fixed (GQA is smaller), some hold total parameters fixed (GQA has a wider FFN). Until the convention is fixed, $\Delta$ is not a well-defined number.
- **Theoretically open.** Whether KV-head sharing costs representational capacity recoverable by width. No separation theorem and no expressivity-equivalence proof exists for $G < H$.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by an effect size below seed noise.**

Two confounds run in opposite directions. Reducing $G$ removes parameters (helping the MHA arm) and reduces per-token FLOPs (letting the GQA arm train on more tokens for the same budget). Published comparisons control neither, so the reported difference is a sum of three effects with unknown signs.

The effect size then makes it worse. The headline GQA-8-vs-MHA gap is ~0.1 benchmark points. Seed-to-seed standard deviation on multi-task benchmark averages at this scale is typically 0.3–0.5 points. **The evidence the entire field's KV-cache design rests on is smaller than its own noise floor**, and no confidence interval was published. Resolving a 0.1-point effect requires either many seeds — multiplying an already large training cost — or switching to validation loss, which Tay et al. show may not rank the architectures the way deployment does. That is the bind: the cheap measurement is not decision-relevant, and the decision-relevant measurement is not affordable at the seed count needed.

## 7. Current Research (as of 2026)

- **Cache-shape co-design.** Cross-layer attention (Brandon et al., MIT/Meta), MLA (DeepSeek), and sliding-window/global hybrids (Gemma, Character.AI) treat $G$ as one axis among several. The framing has shifted from "how much GQA" to "what is the Pareto frontier in KV bytes vs loss".
- **Non-uniform grouping.** WGQA (Chinnakonduru & Mohapatra, arXiv:2407.10855, 2024) learns weighted rather than mean pooling when converting MHA→GQA; QCQA and related work allocate different $G$ per layer. Gains reported at ≤1B scale only.
- **Post-hoc conversion.** Uptraining MHA checkpoints to GQA/MLA without full retraining is an active line, motivated by cheaply retrofitting existing open weights *(frontier — verify: several 2025 preprints claim MHA→MLA conversion at 7B scale; third-party replication is thin)*.
- **Long-context stress.** Whether low $G$ hurts needle-retrieval specifically, rather than average loss, is being probed but with retrieval benchmarks whose difficulty is not calibrated across models *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Two model sizes, 400M and 3B non-embedding parameters, trained from scratch at Chinchilla-optimal tokens (8B and 60B). $H = 16$ and $32$ respectively, $d_h = 128$, context 8192.

**Arms.** $G \in \{1, 2, 4, 8, H\}$ at each scale, 3 seeds each — 30 runs, ~$1.2 \times 10^{22}$ FLOPs total, roughly 3k H100-days.

**Control arm.** The $G = H$ MHA run at each scale, **iso-parameter**: every reduced-$G$ arm returns its freed KV-projection parameters to the FFN hidden dimension so all arms at a scale have total parameter counts within 0.5%. Report the iso-FLOP variant as a secondary readout.

**The deciding number.** The ratio
$$R \;=\; \frac{\Delta_{\text{param}}(N{=}3\text{B}, G{=}8)}{\Delta_{\text{param}}(N{=}400\text{M}, G{=}8)}$$
with a bootstrap 95% CI over seeds, $\Delta$ in nats/token of validation loss.

- $R < 1$ with the CI excluding 1: the penalty shrinks with scale; $G=8$ is safe and the field's convention is vindicated.
- $R > 1$ with the CI excluding 1: the penalty grows; every frontier model is paying an unmeasured and increasing quality tax.
- CI containing 1 at these scales: the effect is below resolution at $10^{22}$ FLOPs, which is itself a publishable and decision-relevant result — it means the question cannot be settled below frontier scale.

## 9. Key References

- **[Foundational]** Noam Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. — arXiv:1911.02150
- **[Foundational / SOTA]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[SOTA]** William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, Jonathan Ragan-Kelley. *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention.* NeurIPS 2024. — arXiv:2405.12981
- **[SOTA]** DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024. — arXiv:2405.04434
- **[Method]** Yi Tay, Mostafa Dehghani, Samira Abnar, Hyung Won Chung, William Fedus, Jinfeng Rao, Sharan Narang, Vinh Q. Tran, Dani Yogatama, Donald Metzler. *Scaling Laws vs Model Architectures: How Does Inductive Bias Influence Scaling?* Findings of EMNLP 2023. — arXiv:2207.10551
- **[Theory]** Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank Reddi, Sanjiv Kumar. *Low-Rank Bottleneck in Multi-head Attention Models.* ICML 2020. — arXiv:2002.07028
- **[Scaling]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Systems]** Woosuk Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. — arXiv:2309.06180
- **[Method]** Sai Sena Chinnakonduru, Astarag Mohapatra. *Weighted Grouped Query Attention in Transformers.* 2024. — arXiv:2407.10855

## 10. Worked Example

Take Llama-3-8B as published: $L=32$, $H=32$, $G=8$, $d_h=128$, $d=4096$, bf16.

**Cache saved.** Per token, $M_{\text{kv}} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes = 128 KiB. The $G=32$ counterfactual is $512$ KiB/token. At 8192 context: **1 GiB vs 4 GiB per sequence.** On an 80 GiB H100 holding 16 GiB of weights, that is ~62 concurrent sequences instead of ~15 — a 4× throughput difference. The engineering case for GQA is not in dispute.

**The confound made visible.** KV projections per layer: MHA needs $2 d^2 = 33.6$M; GQA-8 needs $2 d \cdot (G d_h) = 2 \times 4096 \times 1024 = 8.4$M. Difference $25.2$M per layer, $\times 32 = $ **806M parameters**. The shipped 8.03B GQA model is compared, implicitly, against an MHA model that would carry ~8.84B parameters — 10% more. Any measured quality gap therefore contains a 10%-parameter effect that nobody subtracted. At Chinchilla scaling those 806M parameters are worth roughly a 3–4% loss reduction if spent on the FFN, plausibly larger than the sharing penalty itself.

**The noise floor made visible.** The only published head-to-head at scale reports ≈47.1 (GQA-8) vs ≈47.2 (MHA) on a T5-XXL benchmark average — a 0.1-point gap, one seed each, no interval. To detect a true 0.1-point effect against a per-seed standard deviation of ~0.4 at 80% power requires about $n \approx 250$ seeds per arm. At 11B parameters that is roughly $10^{25}$ FLOPs of training — more than was used to train the model whose design decision it is meant to justify.

That is the obstruction in one line: the field made a universal architecture choice on a difference it could not have resolved, and the affordable version of the experiment (§8, validation loss at 400M/3B) answers a question that Tay et al. show may not be the deployment question.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*