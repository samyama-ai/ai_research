---
id: 12-quantization-compression/optimal-rounding-approximability
title: "Optimal Rounding is NP-Hard but Approximable"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Rounding is NP-Hard but Approximable

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/optimal-rounding-approximability` · **Status:** open

## 1. Problem Statement

Given a trained weight matrix and a calibration set, post-training quantization must pick, for each weight, one of the two neighbouring grid points (or one of $2^b$ grid points). Rounding each weight to its nearest grid point (RTN) is provably not optimal: the weights interact through the input covariance, so a coordinated set of "wrong" roundings can cancel. The catalog problem is the gap between three statements that are routinely conflated:

- **Theory variant.** Is the layerwise rounding problem NP-hard, and what multiplicative approximation ratio is achievable in polynomial time? The decision version is: given $H \succeq 0$, $w$, step $s$, and $\tau > 0$, does a grid point $q$ exist with $(w-q)^\top H (w-q) \le \tau$?
- **Method variant.** Do the deployed heuristics (AdaRound, GPTQ/OBQ, LDLQ, GPFQ) achieve a guarantee on the *layerwise* objective, and is that guarantee multiplicative, additive, or only in expectation over a randomization?
- **Measurement variant.** Does minimizing the layerwise proxy $(w-q)^\top H (w-q)$ minimize end-task loss? The proxy is a second-order Taylor surrogate; the quantity anyone cares about is perplexity or task accuracy.

Solving the problem means: (a) a hardness result and matching approximation ratio for the layerwise objective under a stated grid model, and (b) evidence that closing the remaining proxy gap changes the end-task number.

## 2. Formal Setting

Layer weights $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, calibration activations $X \in \mathbb{R}^{d_{\text{in}} \times m}$ collected by running $m$ tokens of a held-out corpus through the frozen model and caching the layer input. Measured Hessian:

$$H \;=\; \tfrac{1}{m} X X^\top + \lambda I, \qquad \lambda = 10^{-2}\cdot \tfrac{1}{d_{\text{in}}}\mathrm{tr}(H_0)$$

$\lambda$ is the damping actually used in GPTQ-family code, not a theoretical device — without it $H$ is singular whenever $m < d_{\text{in}}$.

Grid: $\mathcal{G}(s, z) = \{\, s(k - z) : k \in \{0,\dots,2^b-1\}\,\}$ with scale $s$ and zero-point $z$ fixed per group of $g$ columns (typically $g = 128$). The **rounding-only** problem restricts each coordinate to its two neighbours, $q_i \in \{s\lfloor w_i/s\rfloor, s\lceil w_i/s\rceil\}$, giving $2^{d_{\text{in}}}$ candidates. Rows decouple, so per row $w \in \mathbb{R}^{d_{\text{in}}}$:

$$\textsf{OPT}(w,H,s) \;=\; \min_{q \in \mathcal{G}^{d_{\text{in}}}} \; (w-q)^\top H (w-q)$$

Write $e = w - q$ and $\delta_i = e_i - \mathrm{RTN}_i$ so the rounding-only problem is a Boolean quadratic minimization in $b \in \{0,1\}^{d_{\text{in}}}$.

End-task quantity, as measured: $\Delta \mathrm{PPL} = \mathrm{PPL}(\hat\theta) - \mathrm{PPL}(\theta)$ on WikiText-2 at sequence length 4096, or $\Delta$accuracy averaged over a fixed zero-shot suite.

Assumptions, with the ones known to fail marked:

1. $H$ is the true curvature. **Violated** — $H$ is the Gauss-Newton block of a *layerwise* reconstruction loss, not the network Hessian; cross-layer terms are dropped entirely.
2. Layers are independent. **Violated** — quantization error propagates; BRECQ's block reconstruction exists precisely because of this.
3. $H$ estimated from $m \approx 128$ sequences equals the deployment-distribution Hessian. **Violated** — the calibration set is a few hundred thousand tokens against a deployment distribution nobody samples.
4. $s, z$ fixed before rounding. Violated by any method that jointly searches scales.

## 3. State of the Art

**Theory SOTA (established).** The rounding-only problem is an instance of Boolean quadratic minimization with PSD matrix; the full-grid problem is a bounded-box closest vector problem (CVP) in the lattice $s\mathbb{Z}^{d}$ under the metric $H$. CVP is NP-hard (van Emde Boas, 1981) and NP-hard to approximate within $n^{c/\log\log n}$ (Dinur, Kindler, Raz, Safra, *Combinatorica* 2003). No polynomial-time constant-factor multiplicative approximation for the layerwise objective is known, and none is expected. What *is* established positively: unbiased stochastic rounding achieves the additive bound $\mathbb{E}[(w-q)^\top H(w-q)] \le \tfrac{s^2}{4}\mathrm{tr}(H)$, because unbiasedness kills every off-diagonal term.

**Method SOTA with a proof (established).** QuIP (Chee, Cai, Kuleshov, De Sa, NeurIPS 2023) proves LDLQ — adaptive rounding with linear feedback, the family containing GPTQ — is optimal *within that family* for a worst-case and average-case proxy loss, and that RTN and stochastic rounding are not. GPFQ (Lybrand & Saab, *JMLR* 2021; Zhang, Zhou, Saab, *SIMODS* 2023) gives relative error decaying as $\tilde{O}(\sqrt{\log N / m})$ for a single layer with random or bounded-moment input, under assumptions on the data distribution that transformer activations do not satisfy.

**Empirical SOTA (benchmark numbers only).** GPTQ (Frantar et al., ICLR 2023), QuIP#/QTIP (Tseng et al., ICML 2024), and QuaRot/SpinQuant achieve 3–4 bit weight-only quantization with small perplexity deltas. These are benchmark numbers on WikiText-2 and a zero-shot suite; **none is accompanied by a measurement of how close the achieved layerwise objective is to $\textsf{OPT}$**, because $\textsf{OPT}$ is never computed. Claimed-but-unablated: that the ordering of methods by perplexity matches their ordering by proxy loss.

## 4. What Is Known

- **The gap between rounding choices is large at real scale.** AdaRound (Nagel et al., ICML 2020) sampled 100 random roundings of a single ResNet18 layer at 4 bits and found a spread of more than $10\%$ top-1 accuracy; the best sampled rounding beat nearest rounding by roughly $10$ points. Scale: ResNet18, ImageNet, W4A32.
- **Optimizing the proxy pays, on CNNs.** AdaRound reports W4A32 ResNet18 at $68.71\%$ top-1 versus $\approx 52\%$ for nearest rounding and $69.68\%$ FP32 — a ~16-point recovery from rounding order alone, no weight training.
- **Optimizing the proxy pays, on LLMs.** GPTQ quantizes OPT-175B and BLOOM-176B to 3–4 bits in about 4 GPU-hours with near-lossless perplexity, where RTN at 3 bits diverges. OBC/OBQ (Frantar & Alistarh, NeurIPS 2022) is the exact-greedy predecessor at $O(d^3)$ per row.
- **Incoherence processing helps and is provable.** QuIP shows that random orthogonal (later Hadamard) conditioning of $W$ and $H$ improves the LDLQ bound and enables usable 2-bit quantization of OPT/Llama-family models.
- **The additive bound is tight and weak.** $\tfrac{s^2}{4}\mathrm{tr}(H)$ is achieved exactly by stochastic rounding and can exceed $\textsf{OPT}$ by an unbounded factor (Section 10).

## 5. What Is Not Known

- **Theoretically open.** No published NP-hardness proof for the *rounding-only* (two-neighbour, PSD $H$) layerwise problem specifically — the hardness is inherited from CVP for the full-grid case, but the box-restricted binary version with a PSD, damped, empirically-structured $H$ has no matching reduction in the quantization literature. Likewise no inapproximability result stated in the quantization setting, and no algorithm with a multiplicative guarantee better than trivial.
- **Empirically open.** Nobody has computed $\textsf{OPT}$ exactly for a real transformer row ($d_{\text{in}} = 4096$) by branch-and-bound or a CVP solver and reported the ratio $L_{\text{GPTQ}}/\textsf{OPT}$. The experiment is runnable — it needs a lattice solver and patience, not new theory.
- **Methodologically blocked.** Whether "optimal rounding" is the right target at all. The proxy-to-perplexity map is not characterized: no one has shown the derivative of end-task loss with respect to layerwise proxy loss is positive in the regime where good methods already operate.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure the thing it names**, compounded by **absent ground truth**.

- Ground truth is absent: $\textsf{OPT}$ is never computed, so every claim of "near-optimal rounding" is relative to another heuristic. The field's ordering is by perplexity, which conflates rounding quality with scale search, incoherence processing, and calibration-set choice.
- The named quantity is layerwise, the measured quantity is end-to-end. A method can cut proxy loss by 30% and move perplexity by 0.01 — that has been observed informally and never ablated cleanly, because published comparisons change three things at once.
- Compute is a secondary obstruction: exact CVP at $d = 4096$ over a $2^{4096}$ box is out of reach in general, but is tractable for $d \le 64$ sub-blocks, which is enough to bound the ratio empirically.

## 7. Current Research (as of 2026)

- **Incoherence and rotation.** QuIP# / QTIP (Cornell, De Sa group), QuaRot and SpinQuant (ETH/ISTA and Meta) push Hadamard-conditioned lattice codebooks; the theory is about conditioning $H$, not about approximating $\textsf{OPT}$.
- **Vector and trellis quantization.** Replacing coordinate rounding with lattice/trellis codes, which changes the combinatorial problem from Boolean to a shortest-path over a trellis and is therefore exactly solvable per block — an under-remarked case where the NP-hardness is sidestepped by changing the code, not the algorithm.
- **Provable greedy quantization.** Saab and collaborators continue extending GPFQ-style error bounds to multi-layer and non-Gaussian input *(frontier — verify)*.
- **Hardness-aware formulations.** SDP and Goemans-Williamson-style relaxations of the rounding problem are discussed but, to our knowledge, not yet reported at transformer scale *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** how far from optimal is GPTQ on the objective it optimizes, and does closing that gap matter?

- **Scale.** Llama-3.1-8B, one MLP down-projection layer, $d_{\text{in}} = 14336$. Take 200 randomly chosen rows. Partition each row into 224 blocks of $d = 64$ columns with the corresponding $64\times64$ diagonal blocks of $H$ (calibration: 128 sequences × 4096 tokens of C4). At $d = 64$ and $b = 3$, compute $\textsf{OPT}$ exactly per block by branch-and-bound over the box-CVP instance (seconds per block with a Schnorr–Euchner enumeration and an RTN incumbent).
- **Control arms.** (i) RTN, (ii) stochastic rounding, (iii) GPTQ restricted to the same block, (iv) exhaustive optimum.
- **Deciding number.** The median ratio $\rho = L_{\text{GPTQ}}/\textsf{OPT}$ over the 44,800 blocks. If $\rho < 1.05$, adaptive rounding is essentially solved on the proxy and further work belongs in the objective, not the search. If $\rho > 1.5$, there is real headroom, and the follow-up is mandatory: replace GPTQ with the exact block optimum on *every* layer and report $\Delta$WikiText-2 perplexity at 3 bits. If perplexity moves by less than $0.05$ while $\rho > 1.5$, the proxy is refuted as the target and the problem shifts to Section 5's methodological gap.

## 9. Key References

- **[Foundational]** M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos, T. Blankevoort. *Up or Down? Adaptive Rounding for Post-Training Quantization.* ICML 2020. — arXiv:2004.10568
- **[Foundational]** P. van Emde Boas. *Another NP-complete problem and the complexity of computing short vectors in a lattice.* Technical Report 81-04, University of Amsterdam, 1981.
- **[Foundational]** I. Dinur, G. Kindler, R. Raz, S. Safra. *Approximating CVP to within almost-polynomial factors is NP-hard.* Combinatorica 23(2), 2003.
- **[SOTA]** E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. — arXiv:2210.17323
- **[SOTA]** J. Chee, Y. Cai, V. Kuleshov, C. De Sa. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees.* NeurIPS 2023. — arXiv:2307.13304
- **[SOTA]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. — arXiv:2402.04396
- **[Theory]** E. Lybrand, R. Saab. *A Greedy Algorithm for Quantizing Neural Networks.* JMLR 22(156), 2021.
- **[Theory]** J. Zhang, Y. Zhou, R. Saab. *Post-training Quantization for Neural Networks with Provable Guarantees.* SIAM Journal on Mathematics of Data Science, 2023.
- **[Related]** E. Frantar, D. Alistarh. *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning.* NeurIPS 2022.
- **[Related]** Y. Li et al. *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction.* ICLR 2021.
- **[Survey]** A. Gholami, S. Kim, Z. Dong, Z. Yao, M. Mahoney, K. Keutzer. *A Survey of Quantization Methods for Efficient Neural Network Inference.* 2021. — arXiv:2103.13630

## 10. Worked Example

Take $d = 100$, step $s = 1$, grid $\{0,1\}$, and $w_i = 0.49$ for all $i$. Let $H = vv^\top$ with $v = \mathbf{1}$ — a rank-one Hessian, the limit of a calibration set whose activations are perfectly correlated. The objective is $(\mathbf{1}^\top e)^2$ with $e_i = w_i - q_i \in \{0.49, -0.51\}$.

- **RTN.** Every coordinate rounds down, $e_i = 0.49$, so $\mathbf{1}^\top e = 49$ and the loss is $49^2 = 2401$.
- **Optimum.** Round 49 coordinates up and 51 down: $\mathbf{1}^\top e = 51(0.49) - 49(0.51) = 24.99 - 24.99 = 0$. So $\textsf{OPT} = 0$.
- **Stochastic rounding.** Round up with $p_i = 0.49$. Unbiased, so the expected loss is $\sum_i \mathrm{Var}(e_i) = 100 \times 0.49 \times 0.51 = 24.99$, matching the bound $\tfrac{s^2}{4}\mathrm{tr}(H) = \tfrac{1}{4}\cdot 100 = 25$.

The obstruction is visible in three numbers: $2401$, $24.99$, $0$. Stochastic rounding is 96× better than RTN and honours its additive guarantee, yet its ratio to the optimum is infinite — no multiplicative approximation of any factor exists for this instance in this family. Meanwhile the additive bound $25$ is the *only* thing a guarantee-carrying method can promise, and it is independent of how good the true optimum is. Real Hessians are not rank one, but a 4096-dimensional attention-projection $H$ estimated from 128 sequences has an effective rank in the low hundreds, so the near-degenerate directions where this cancellation lives are exactly the ones present in practice. Reporting perplexity tells you nothing about which of the three regimes a method is in.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*