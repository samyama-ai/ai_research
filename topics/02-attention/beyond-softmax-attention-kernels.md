---
id: 02-attention/beyond-softmax-attention-kernels
title: "Optimal Attention Kernel Beyond Exponential Softmax"
topic: 02-attention
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Attention Kernel Beyond Exponential Softmax

> **Topic:** Attention Mechanisms · **ID:** `02-attention/beyond-softmax-attention-kernels` · **Status:** open

## 1. Problem Statement

Softmax attention normalizes similarity scores with $\exp(\cdot)$. That choice is inherited from the 2017 architecture, not derived. The problem: **is the exponential kernel optimal, and optimal for what objective?**

Three variants, with different difficulty:

- **Measurement variant.** Define a metric under which one kernel beats another that is not just "downstream loss at one scale". Candidates: loss at matched FLOPs, length-generalization gap, retrieval accuracy at $n \gg$ training length, attention-entropy stability during training. No agreed metric exists.
- **Method variant.** Find a similarity function $k(q, k)$ and normalization that dominates softmax on loss-per-FLOP at $\geq 7$B parameters. Empirically open; hundreds of candidates proposed, none has displaced softmax in a frontier model.
- **Theory variant.** Prove that softmax is or is not the minimizer of some natural objective (max-entropy under a moment constraint, Bayes-optimal retrieval under a generative model of key–query geometry). Theoretically open.

A solution to the method variant is a kernel plus a hardware-realizable kernel implementation plus an ablation at scale. A solution to the theory variant is a theorem naming the objective softmax optimizes and the assumptions under which it does.

## 2. Formal Setting

Attention is a kernel smoother (Tsai et al., EMNLP 2019). For queries $Q \in \mathbb{R}^{n \times d}$, keys $K$, values $V$:

$$\mathrm{Attn}(Q,K,V)_i \;=\; \sum_{j=1}^{n} \frac{k(q_i, k_j)}{\sum_{l=1}^{n} k(q_i, k_l)} \, v_j, \qquad k_{\mathrm{softmax}}(q,k) = \exp\!\left(\frac{\langle q,k\rangle}{\sqrt{d}}\right).$$

Write $p_{ij} = k(q_i,k_j)/\sum_l k(q_i,k_l)$ for the attention weights.

**Quantities as measured.**

- **Loss at matched compute.** $L(C)$ = validation cross-entropy in nats/token at total training FLOPs $C$, with $C \approx 6ND$ ($N$ non-embedding params, $D$ tokens) plus the attention term $\approx 12 L_{\text{layers}} n d$ per token. Two kernels differ in per-token cost; comparing at equal parameters is the wrong control — equal $C$ is the right one.
- **Attention entropy.** $H_i = -\sum_j p_{ij}\log p_{ij}$, measured per head per layer on a held-out batch. Entropy collapse ($H_i \to 0$) predicts training instability (Zhai et al., ICML 2023).
- **Sharpness / dispersion.** $\max_j p_{ij}$ as a function of context length $n$. Softmax with bounded logits has $\max_j p_{ij} \to 0$ as $n \to \infty$ — the coefficients provably disperse (Veličković et al., 2024).
- **Throughput.** Tokens/s and HBM bytes moved at fixed batch and sequence length, on a named accelerator. A kernel that is not fusable into a FlashAttention-style tiled loop loses on wall-clock even at equal FLOPs.
- **Length generalization gap.** Accuracy on a synthetic retrieval task (needle-in-haystack, associative recall) at $n = 4\times$ the training context.

**Assumptions and which are violated.**

1. *Logits are bounded.* Assumed by dispersion results and by Alman–Song complexity. Violated in practice: trained models grow logit magnitudes, and attention sinks (BOS tokens absorbing large mass) create extreme entries.
2. *Heads are interchangeable.* Violated — induction heads, retrieval heads and sink heads have different entropy profiles, so a single global kernel choice may be wrong for all of them.
3. *Kernel choice is separable from positional encoding and normalization.* Violated: sigmoid attention needs a length-dependent bias $b \approx -\log n$ to work at all (Ramapuram et al., 2024), so "kernel" and "normalization" are not independent knobs.
4. *Downstream loss is monotone in kernel quality.* Unverified — small-scale loss differences of $<0.01$ nats routinely fail to survive scaling.

## 3. State of the Art

**Established (ablated, reproduced).**

- **Sparse normalizers.** sparsemax (Martins & Astudillo, ICML 2016) and $\alpha$-entmax (Peters, Niculae & Martins, ACL 2019) produce exactly-zero weights and match or slightly exceed softmax BLEU on machine translation. Entmax-1.5 is a drop-in and is reproduced across several codebases. It has never been shown to help at LLM scale.
- **Sigmoid attention.** Ramapuram et al. (2024) show $\sigma(\langle q,k\rangle/\sqrt d + b)$, with $b = -\log n$, matches softmax on language modeling and ViT across 85M–1B, and give FLASHSIGMOID, reporting ~17% inference kernel speedup on H100. This is the strongest evidence that the *exponential* specifically is not required.
- **Linear/feature-map kernels.** $k(q,k)=\phi(q)^\top\phi(k)$ gives $O(n)$ attention: Katharopoulos et al. (ICML 2020), Performer/FAVOR+ (Choromanski et al., ICLR 2021). Established that they are strictly worse in quality at matched params without gating; gated linear attention (Yang et al., ICML 2024) closes much of the gap on ≤1.3B models.
- **Softmax mimicry.** Hedgehog (Zhang et al., ICLR 2024) shows the recoverable property of softmax is *low-entropy, spiky, monotonic* weights; learning a feature map to match softmax's attention distribution recovers most of the quality gap.

**Claimed but unablated.** That entmax's sparsity gives interpretability benefits; that polynomial kernels (PolySketchFormer, Kacham et al., ICML 2024) are Pareto-superior beyond the reported benchmark scale. Both exist mainly as benchmark numbers on ≤1.3B models with a single seed.

**Theory SOTA.** Alman & Song (NeurIPS 2023): with entries bounded by $o(\sqrt{\log n})$, softmax attention admits an $n^{1+o(1)}$ algorithm; above $\Theta(\sqrt{\log n})$ no subquadratic algorithm exists under SETH. Keles, Wijewardena & Hegde (ALT 2023) give a matching hardness result. Neither says the exponential is *optimal* — only that its approximability has a sharp threshold.

## 4. What Is Known

- Rank collapse: pure self-attention without skips/MLPs converges to rank-1 **doubly exponentially in depth** (Dong, Cordonnier & Loukas, ICML 2021). This is a property of the row-stochastic averaging, not of $\exp$ specifically.
- Entropy collapse: in ViT training, mean attention entropy falling below ~0.5 nats coincides with loss divergence; $\sigma$Reparam fixes it (Zhai et al., ICML 2023, ViT-B/16 scale).
- Dispersion: for any fixed bounded logit range, softmax weights over $n$ items converge to uniform as $n\to\infty$ at rate $O(1/n)$; sharp OOD decisions therefore fail at long context (Veličković et al., 2024, demonstrated on max-retrieval tasks with $n$ up to $\sim$10$\times$ training length).
- Sigmoid $\approx$ softmax on loss at 1B params (Ramapuram et al., 2024) — the largest careful non-softmax kernel ablation published.
- Linear attention at 1.3B: ungated linear attention loses roughly 0.1–0.2 nats vs softmax; gating recovers most of it (Yang et al., ICML 2024).

## 5. What Is Not Known

- **Theoretically open.** No theorem states an objective for which $\exp$ is the unique optimum. Max-entropy under a linear moment constraint yields softmax, but no one has shown that constraint is the right model of retrieval. No lower bound says a non-exponential kernel *cannot* be strictly better at fixed FLOPs.
- **Empirically open.** No kernel ablation exists at $\geq 7$B with matched compute, multiple seeds, and long-context evaluation. The experiment is runnable today for ~$10^5$ GPU-hours; nobody has published it.
- **Methodologically blocked.** "Which kernel is better" has no agreed metric. Loss-at-matched-FLOPs, length generalization and entropy stability disagree in sign on known cases: sigmoid matches softmax on loss but changes the entropy profile; entmax improves sparsity metrics with no loss change. Until the metric is fixed, the comparison is not well posed.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by hardware coupling**. A kernel change is never evaluated alone: it interacts with the positional encoding, the normalization ($b=-\log n$), the initialization scale, and the learning rate. Reported wins are typically single-seed at ≤1B, where seed variance on validation loss is $\pm 0.01$–0.02 nats — the same magnitude as the claimed effect.

Second obstruction: **the winner is decided by memory bandwidth, not FLOPs.** Softmax's advantage is partly that it is fusable — the online-softmax rescaling trick makes FlashAttention possible. A kernel with a non-associative normalizer cannot be tiled the same way and loses 2–3× wall-clock even at identical arithmetic cost. Any candidate must therefore be co-designed with its kernel, which raises the cost of a fair test by an order of magnitude and biases the field toward whatever already has a fast implementation.

## 7. Current Research (as of 2026)

- Adaptive-temperature and sharpness-preserving normalizers, following Veličković et al. — targeting the dispersion result directly *(frontier — verify)*.
- Gated linear attention and hybrid stacks (a few full-softmax layers interleaved with linear layers), pursued at MIT/Tsinghua/Flash-Linear-Attention community and in several open-weights model families.
- Polynomial and sketch-based kernels (Google Research: PolySketchFormer line).
- Sigmoid attention follow-ups at Apple; kernel-level work on fused non-softmax normalizers.
- Theory: attention as a Hopfield/energy retrieval model, asking which kernel is Bayes-optimal under a stated key-geometry prior *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does any non-exponential kernel beat softmax on loss-per-FLOP at scale, or is it a wash?

- **Scale.** 1.4B parameters, 30B tokens (Chinchilla-ish), 8k training context. Three seeds per arm. ~2×10<sup>21</sup> FLOPs per arm; ~4k H100-hours per arm, ~28k total.
- **Arms.** (1) softmax — control; (2) sigmoid with $b=-\log n$; (3) entmax-1.5; (4) adaptive-temperature softmax; (5) $\mathrm{ReLU}(\langle q,k\rangle)^2$ polynomial, degree 2; (6) gated linear attention. All arms share tokenizer, data order, RoPE, optimizer, and are **tuned separately for learning rate** over a 3-point sweep — otherwise the control is favored by construction.
- **Matched compute, not matched params.** Report FLOPs measured, not estimated.
- **Deciding number.** Validation loss in nats/token at fixed $C = 2\times10^{21}$, reported as $\Delta L$ vs control with a seed-variance interval. **Decision rule: a kernel wins only if $\Delta L \leq -0.02$ nats with the 3-seed interval excluding zero, AND it does not lose more than 10% throughput on a fused kernel at $n{=}8$k.** Secondary: needle-retrieval accuracy at $n = 32$k (4× train).

Expected outcome, stated in advance: arms 2–4 land within $\pm 0.01$ nats of control (a wash on loss) while differing by $>15$ points on the 32k retrieval probe. That result would move the field's metric from loss to length generalization — which is the actual contribution.

## 9. Key References

- **[Foundational]** Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[Foundational]** Tsai, Bai, Yamada, Morency, Salakhutdinov. *Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel.* EMNLP, 2019.
- **[Foundational]** Martins, Astudillo. *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification.* ICML, 2016. — arXiv:1602.02068
- **[SOTA]** Peters, Niculae, Martins. *Sparse Sequence-to-Sequence Models.* ACL, 2019. — arXiv:1905.05702
- **[SOTA]** Ramapuram, Danieli, Dhekane, Weers, Busbridge, Ablin, Likhomanenko, Digani, Gu, Zhang, Cuturi. *Theory, Analysis, and Best Practices for Sigmoid Self-Attention.* 2024. — arXiv:2409.04431
- **[SOTA]** Veličković, Perivolaropoulos, Barbero, Pascanu. *softmax is not enough (for sharp out-of-distribution).* 2024. — arXiv:2410.01104
- **[SOTA]** Zhang, Backurs, Arora, Ré et al. *The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry.* ICLR, 2024.
- **[SOTA]** Yang, Wang, Shen, Panda, Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML, 2024. — arXiv:2312.06635
- **[Theory]** Alman, Song. *Fast Attention Requires Bounded Entries.* NeurIPS, 2023. — arXiv:2302.13214
- **[Theory]** Keles, Wijewardena, Hegde. *On the Computational Complexity of Self-Attention.* ALT, 2023.
- **[Theory]** Dong, Cordonnier, Loukas. *Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.* ICML, 2021. — arXiv:2103.03404
- **[Empirical]** Zhai et al. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML, 2023. — arXiv:2303.06296
- **[Survey]** Tay, Dehghani, Bahri, Metzler. *Efficient Transformers: A Survey.* ACM Computing Surveys, 2022. — arXiv:2009.06732

## 10. Worked Example

Take one retrieval head with $d=64$, and a context where exactly one key matches. Suppose the matching key has logit $s^\star = 6$ and all $n-1$ distractors have logit $0$ (a generous 6-nat margin).

Softmax mass on the correct key:

$$p^\star = \frac{e^{6}}{e^{6} + (n-1)} = \frac{403.4}{403.4 + n - 1}.$$

| $n$ | $p^\star$ |
|---|---|
| 512 | 0.441 |
| 4,096 | 0.090 |
| 32,768 | 0.012 |
| 262,144 | 0.0015 |

At 32k context the correct key holds 1.2% of the mass; the other 98.8% is distractor value vectors averaged into the output. The head has not gotten worse — the margin is unchanged — the normalizer dilutes it. This is the dispersion result made concrete: to hold $p^\star \geq 0.5$ the logit must grow as $\log n$, so a model trained at 4k and run at 32k is 3 nats short.

Now the obstruction. Swap in entmax-1.5, which zeros the distractors outright and gives $p^\star \approx 1$. Retrieval is fixed. But: (a) the top-$k$ support must be computed by a sort or a threshold search, which does not fuse into the FlashAttention online-rescaling loop, so measured throughput drops ~2.5× at $n{=}32$k; (b) at $n{=}4$k, where training happens, softmax already gives $p^\star = 0.09$ and *validation loss is essentially identical* — the gain is invisible to the metric the field uses to select kernels.

So the better kernel is not selected, because the number that would show the improvement (32k retrieval) is not the number the comparison is run on (4k loss), and the number it does lose on (throughput) is an artifact of which kernel got a fused implementation first. That is the problem in one table.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*