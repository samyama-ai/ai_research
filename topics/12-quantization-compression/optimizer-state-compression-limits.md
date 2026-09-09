---
id: 12-quantization-compression/optimizer-state-compression-limits
title: "Gradient and Optimizer State Compression Limits"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Gradient and Optimizer State Compression Limits

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/optimizer-state-compression-limits` · **Status:** open

## 1. Problem Statement

Adam carries two state tensors per parameter plus a gradient buffer. In fp32 that is 12 bytes/parameter of transient state against 4 bytes of weights. The question: **how few bits per coordinate can the gradient and the optimizer state be stored or transmitted in before the training run measurably degrades — and does that critical bit-rate move with model scale?**

Three variants, routinely conflated:

- **Measurement.** Define "degrades". Final validation loss at a fixed token budget is the usual proxy, but it confounds a compression penalty with a learning-rate mistuning. The measurement variant asks for a degradation metric that is invariant to hyperparameter search effort.
- **Method.** Build a quantizer/sparsifier whose critical rate $B^\star$ is low. Established practice: 8 bits works, 4 bits works for fine-tuning, 1 bit works for gradients *with error feedback*.
- **Theory.** Prove a lower bound on $B$ below which no compressor, biased or unbiased, can match uncompressed convergence for a given function class. Distributed mean estimation has such bounds; stateful adaptive optimization does not.

Solving it means: a scaling law $B^\star(N, D)$ for the critical rate as a function of parameters $N$ and tokens $D$, plus a matching lower bound.

## 2. Formal Setting

Parameters $\theta_t \in \mathbb{R}^d$, stochastic gradient $g_t$, optimizer state $s_t \in \mathbb{R}^{kd}$ ($k=2$ for Adam: first moment $m_t$, second moment $v_t$).

A compressor is a map $\mathcal{C}: \mathbb{R}^{n} \to \{0,1\}^{\ell}$ with decoder $\mathcal{D}$. Its **measured rate** is
$$B_{\text{eff}} = \frac{\ell}{n} = B_{\text{payload}} + \frac{B_{\text{meta}}}{G},$$
where $G$ is the block size and $B_{\text{meta}}$ the bits of per-block metadata. This is the quantity to report: block-wise int4 with one fp16 scale per 128 elements is $4 + 16/128 = 4.125$ bits, not 4. Papers that quote the payload only understate rate by 3–13%.

Two standard compressor classes:
- **Unbiased, variance $\omega$:** $\mathbb{E}[\mathcal{D}\mathcal{C}(x)] = x$, $\mathbb{E}\|\mathcal{D}\mathcal{C}(x)-x\|^2 \le \omega\|x\|^2$ (QSGD, rand-$k$).
- **Contractive, factor $\delta$:** $\mathbb{E}\|\mathcal{D}\mathcal{C}(x)-x\|^2 \le (1-\delta)\|x\|^2$ (top-$k$, sign, low-rank projection). Biased; requires error feedback for convergence.

**Degradation.** With $L(N,D;B)$ the validation loss (nats/token) of a run at rate $B$, define
$$\Delta L(N,D;B) = L(N,D;B) - L(N,D;\infty),$$
each arm's learning rate independently tuned over the same grid. The **critical rate** is $B^\star(\epsilon) = \min\{B : \Delta L \le \epsilon\}$. The scale-sensitivity is the slope $\kappa = \partial \Delta L / \partial \log_{10} N$ in nats/decade — the number that decides whether a small-scale result transfers.

**Memory-matched frontier.** At fixed optimizer-state bytes $M$, a run may spend them on $N = M \cdot 8 / (k B_{\text{eff}})$ parameters. The fair question is not "does 4-bit hurt" but "does 4-bit at $7.8\times$ the parameters beat fp32".

**Assumptions, and where they fail.**
- *Bounded gradient variance $\sigma^2$.* Violated: transformer gradient noise is heavy-tailed, which is part of why adaptive methods beat SGD on attention models (Zhang et al., NeurIPS 2020).
- *$L$-smoothness with a global constant.* Violated at loss spikes; smoothness is state-dependent.
- *Coordinates exchangeable.* Violated: outlier feature dimensions carry systematically larger magnitudes, the same phenomenon that forces mixed-precision handling in LLM.int8() (Dettmers et al., NeurIPS 2022).
- *Quantization error independent of the signal.* Violated for $v_t$, whose dynamic range spans several orders of magnitude within one tensor and correlates with which coordinates matter.
- *Convergence-rate equivalence implies loss equivalence.* Violated in practice: EF21-style theory bounds $\min_t\|\nabla f\|^2$, which is not the quantity a pretraining run is scored on.

## 3. State of the Art

**Systems/empirical SOTA — established.**
- **8-bit Adam** (Dettmers et al., ICLR 2022): block-wise dynamic quantization of both moments, matched 32-bit results across GLUE, WMT'16, ImageNet/ResNet-50, MoCo v2, and GPT-2 pretraining up to 1.5B. Multiple independent reproductions; shipped in `bitsandbytes` and used by default in many training stacks. This is the most solidly established point on the curve.
- **Adafactor** (Shazeer & Stern, ICML 2018): factored second moment, $O(n+m)$ instead of $O(nm)$ per matrix. Established at T5 scale.
- **Error feedback for 1-bit gradients** (Seide et al., INTERSPEECH 2014; Karimireddy et al., ICML 2019): sign compression alone can fail to converge; with error feedback it recovers SGD rates.

**Claimed but unablated.**
- **4-bit optimizers** (Li, Chen, Zhu, NeurIPS 2023): 4-bit $m$ and $v$ with a rank-1 normalization for the second moment, "lossless" on NLU, machine translation and LLaMA-7B instruction fine-tuning. The evidence is fine-tuning and short-horizon; there is no from-scratch pretraining arm at multi-billion scale with a tuned fp32 control.
- **GaLore** (Zhao et al., ICML 2024): projects gradients to a low-rank subspace, cutting optimizer state ~65%; reports LLaMA-7B pretraining on a 24 GB GPU. The headline comparisons are at fixed token budget with a shared LR grid; the low-rank arm's advantage under a fully tuned full-rank control remains contested, and rank is a compression axis with different failure modes from bit-width.
- **MicroAdam** (Modoranu et al., NeurIPS 2024) — sparse gradients plus compressed error feedback, with a convergence proof; benchmark numbers only at ≤7B fine-tuning.
- **4-bit Shampoo** (Wang et al., NeurIPS 2024) — quantizes preconditioner eigenvector matrices; vision-scale evidence.

**Theory SOTA.** Communication-constrained distributed mean estimation has tight bounds (Suresh et al., ICML 2017; Zhang et al., NIPS 2013). Biased compression is characterized for convex/nonconvex SGD (Beznosikov et al., JMLR 2023; Richtárik et al., EF21, NeurIPS 2021). None of this covers a *stateful* optimizer whose preconditioner is itself the compressed object — Adam's $v_t$ is an accumulator, so quantization error compounds across steps rather than averaging out.

## 4. What Is Known

- **8 bits is free at ≤1.5B.** 8-bit Adam matched 32-bit GPT-2 perplexity and downstream scores; the paper's own ablation shows block-wise quantization (not just dynamic exponent) is the necessary ingredient — non-blockwise 8-bit diverges. Measured at 1.5B params.
- **Arithmetic of the saving.** fp32 Adam state = 8 B/param; 8-bit block-wise ($8 + 16/2048$ bits) = 2.0 B/param. For 1B parameters that is 8.0 GB → 2.0 GB.
- **1-bit gradients work with error feedback, not without.** signSGD without feedback has explicit divergent counterexamples (Karimireddy et al., ICML 2019). 1-bit Adam (Tang et al., ICML 2021) needs a full-precision warmup phase because Adam's variance term is not stable enough early to freeze.
- **Sparsification tolerates extreme rates.** Deep Gradient Compression (Lin et al., ICLR 2018) reported 270–600× gradient compression with momentum correction and warmup, on ResNet/LSTM scale — a regime with far more gradient redundancy than an LLM at Chinchilla-optimal token counts.
- **Second moment is more compressible than first.** Multiple works converge on the same asymmetry: $v_t$ tolerates factorization (Adafactor) and even per-block sharing (Adam-mini, Zhang et al. 2024, reports ~45–50% total optimizer memory cut with matched loss at ≤7B), while aggressive compression of $m_t$ costs more.
- **Low-rank is not the same as low-bit.** LoRA-style rank restriction changes the solution's spectral structure, not just its precision (Shuttleworth et al. 2024, "intruder dimensions"; Biderman et al., TMLR 2024, "LoRA learns less and forgets less"). Results transfer between the two axes only by assumption.

## 5. What Is Not Known

- **Empirically open.** $\kappa = \partial \Delta L/\partial \log_{10} N$ at $B \in \{4,3,2\}$ bits. Every sub-8-bit result is at ≤7B and mostly fine-tuning. Whether the 4-bit penalty is 0.00 nats at 7B and 0.05 nats at 70B is runnable today and unrun. Cost: ~$10^{21}$ FLOPs for a clean two-scale sweep.
- **Empirically open.** Whether the memory-matched frontier ever favors uncompressed state. No paper reports "$N$-larger model at $B$ bits vs $N$-smaller at 32 bits, both compute-matched".
- **Theoretically open.** A lower bound on bits/coordinate for stateful adaptive optimization. Existing bounds treat the compressor as memoryless mean estimation; the accumulator structure of $v_t$ means the relevant object is a bit-rate–constrained filter, and no rate–distortion result covers it.
- **Methodologically blocked.** $\Delta L$ is not identified without specifying the hyperparameter-tuning budget for each arm. A compressed arm that is under-tuned looks lossy; one whose LR grid is centered after seeing the compressed results looks lossless. No community standard fixes this. AlgoPerf (Dahl et al. 2023) is the closest existing protocol and does not cover state precision.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a compute cliff**. The effect being measured, $\Delta L \approx 0.01$ nats, is smaller than the seed-to-seed spread of a single pretraining run at 1B params and comparable to the effect of a 20% learning-rate change. Distinguishing it requires either multiple seeds per arm — multiplying an already expensive sweep — or a tuning protocol that the field has not agreed on. Meanwhile the *interesting* regime is exactly the one where a single run costs six figures, so the incentive is to publish the 7B fine-tuning number and assert extrapolation. Secondary obstruction: quantization error in $v_t$ is signal-correlated and accumulating, so the standard unbiased-compressor analysis does not apply, and the biased-compressor analysis bounds gradient norm rather than loss.

## 7. Current Research (as of 2026)

- **Sub-4-bit optimizer state** with learned or per-tensor-adaptive codebooks; extensions of the Dettmers/Chen–Zhu line *(frontier — verify)*.
- **Structural rather than numeric compression:** Adam-mini, CAME (Luo et al., ACL 2023), and factored-preconditioner methods — reducing the number of state entries instead of bits per entry.
- **Gradient low-rank plus quantization stacking** (Q-GaLore and successors), where the two compressions interact and are rarely ablated apart *(frontier — verify)*.
- **Optimizer benchmarking under matched tuning**, e.g. Zhao et al., "Deconstructing What Makes a Good Optimizer for Language Models" (2024), which is the methodology this problem needs applied to precision.
- **Rate–distortion framing of training**: treating the optimizer as a lossy channel and asking for the distortion–convergence tradeoff. Little published; a natural home for the missing lower bound *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Two model sizes, 350M and 1.3B decoder-only, on FineWeb/C4 at 20 tokens/param (7B and 26B tokens). Total ≈ $2.6\times10^{20}$ FLOPs per arm-pair; ~5 arms × 2 scales × 3 seeds ≈ 8 GPU-months on H100.

**Arms.** $B_{\text{eff}} \in \{32, 8.008, 4.125, 3.125, 2.125\}$ bits/coordinate, block-wise dynamic-exponent quantization applied to both Adam moments, block size 128 for the sub-8-bit arms.

**Control.** fp32 Adam, learning rate tuned over an identical 5-point log grid *per arm* (so no arm is advantaged by tuning effort), same data order, same seeds. Report per-arm best-of-grid and the grid's location — if any arm's optimum is at a grid edge, the run is void.

**Deciding number.** $\kappa_B = [\Delta L(1.3\text{B};B) - \Delta L(350\text{M};B)] / \log_{10}(1.3\text{B}/350\text{M})$, in nats/decade, with seed variance reported.

- $\kappa_{4.125} \le 0.005$ nats/decade → 4-bit state is scale-safe; the memory should be spent on parameters.
- $\kappa_{4.125} \ge 0.02$ nats/decade → the published 4-bit "lossless" claims are artifacts of small scale, and extrapolation to 70B predicts ≥0.03 nats of loss — enough to erase the memory-matched gain.

## 9. Key References

- **[Foundational]** Seide, Fu, Droppo, Li, Yu. *1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs.* INTERSPEECH, 2014.
- **[Foundational]** Alistarh, Grubic, Li, Tomioka, Vojnovic. *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding.* NeurIPS, 2017. — arXiv:1610.02132
- **[Foundational]** Shazeer, Stern. *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.* ICML, 2018. — arXiv:1804.04235
- **[Foundational]** Suresh, Yu, Kumar, McMahan. *Distributed Mean Estimation with Limited Communication.* ICML, 2017. — arXiv:1611.00429
- **[Theory]** Karimireddy, Rebjock, Stich, Jaggi. *Error Feedback Fixes SignSGD and other Gradient Compression Schemes.* ICML, 2019. — arXiv:1901.09847
- **[Theory]** Richtárik, Sokolov, Fatkhullin. *EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback.* NeurIPS, 2021. — arXiv:2106.05203
- **[Theory]** Beznosikov, Horváth, Richtárik, Safaryan. *On Biased Compression for Distributed Learning.* JMLR, 2023. — arXiv:2002.12410
- **[SOTA]** Dettmers, Lewis, Shleifer, Zettlemoyer. *8-bit Optimizers via Block-wise Quantization.* ICLR, 2022. — arXiv:2110.02861
- **[SOTA]** Li, Chen, Zhu. *Memory Efficient Optimizers with 4-bit States.* NeurIPS, 2023. — arXiv:2309.01507
- **[SOTA]** Zhao, Zhang, Chen, Wang, Anandkumar, Tian. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* ICML, 2024. — arXiv:2403.03507
- **[SOTA]** Modoranu, Safaryan, Malinovsky, Kurtic, Robert, Richtárik, Alistarh. *MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence.* NeurIPS, 2024. — arXiv:2405.15593
- **[Related]** Lin, Han, Mao, Wang, Dally. *Deep Gradient Compression.* ICLR, 2018. — arXiv:1712.01887
- **[Related]** Zhang, Chen, Zhu, Sun, Luo, Ruan, Xiao, Yin. *Adam-mini: Use Fewer Learning Rates To Gain More.* 2024. — arXiv:2406.16793
- **[Methodology]** Dahl, Schneider, Nado, et al. *Benchmarking Neural Network Training Algorithms.* 2023. — arXiv:2306.07179

## 10. Worked Example

A 1.3B-parameter model. Adam state in fp32: $2 \times 4 \times 1.3\times10^9 = 10.4$ GB.

Block-wise 4-bit with one fp16 scale per 128 elements: $B_{\text{eff}} = 4.125$ bits, so $2 \times 4.125/8 \times 1.3\times10^9 = 1.34$ GB. Saving: 9.06 GB.

Now spend that memory instead of banking it. At a fixed 10.4 GB state budget:
$$N = \frac{10.4\times10^9 \times 8}{2 \times 4.125} \approx 1.01\times10^{10}\ \text{parameters},$$
a $7.8\times$ increase. Using the Chinchilla parametric fit $L = 1.69 + 406.4\,N^{-0.34} + 410.7\,D^{-0.28}$ (Hoffmann et al., 2022), the parameter term falls from $406.4/(1.3\times10^9)^{0.34} = 0.322$ to $406.4/(1.01\times10^{10})^{0.34} = 0.161$ nats.

**The prize is 0.16 nats. The claimed 4-bit penalty is ~0.01 nats.** A $16\times$ margin — which is why the field has effectively assumed the answer.

Here is the obstruction. That 0.01 nats was measured at ≤7B, largely in fine-tuning, where the second moment is already well-conditioned by a pretrained initialization. The memory-matched argument above requires the penalty at 10B *from scratch*. If $\Delta L$ grows at even $\kappa = 0.05$ nats/decade — a slope nobody has excluded, because nobody has measured two scales with a tuned control — then between 1.3B and 10B the penalty rises from 0.01 to $0.01 + 0.05\times0.89 = 0.054$ nats, and the case for 4-bit weakens from decisive to marginal. At $\kappa = 0.15$ it inverts.

The whole argument turns on a slope that costs about 8 GPU-months to measure and has not been measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*