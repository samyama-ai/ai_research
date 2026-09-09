---
id: 12-quantization-compression/low-rank-plus-quantized-decomposition
title: "Low-Rank Plus Quantized Decomposition Optimality"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Low-Rank Plus Quantized Decomposition Optimality

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/low-rank-plus-quantized-decomposition` · **Status:** open

## 1. Problem Statement

Given a trained weight matrix $W$, represent it as a sum $W \approx Q + L$, where $Q$ is a low-bit quantized matrix and $L = AB$ is a low-rank correction stored at higher precision. Every method in this family — LoRC, LoftQ, LQ-LoRA, LQER, CALDERA, ZeroQuant-V2 — instantiates this template. None of them solve it optimally, and it is not known what optimal means at scale.

Three variants, with different difficulty:

- **Theory variant.** For a fixed bit budget, a fixed quantization grid, and a fixed rank, is there a polynomial-time algorithm returning a decomposition within a constant factor of the best possible reconstruction error? Or is the joint problem hard, and by how much do the standard alternating heuristics lose?
- **Method variant.** Given a budget of $b$ effective bits per weight, how should it be *split* between $Q$ (more bits per entry) and $L$ (more rank)? Every deployed system picks this by grid search on one benchmark. No allocation rule is derived.
- **Measurement variant.** Reconstruction error in $\|\cdot\|_F$ is not the objective; end-task loss is. It is not established that the layerwise proxy actually used (input-weighted error) orders methods the same way end-task loss does at the 2-bit frontier.

Solving it means: a decomposition algorithm with a proven approximation guarantee against the joint optimum, plus a budget-allocation rule that predicts the empirically best $(b_Q, r)$ split without a sweep.

## 2. Formal Setting

Let $W \in \mathbb{R}^{m \times n}$ be one linear layer's weights. Let $\mathcal{G}_b \subset \mathbb{R}$ be a $2^b$-point quantization grid (uniform with scale $s$ and zero-point $z$, or a learned codebook / lattice as in QuIP#). Let $\mathcal{Q}_b^{m\times n}$ be matrices with all entries in $\mathcal{G}_b$, possibly with per-group scales.

**Objective, as measured.** Layerwise calibration uses a batch $X \in \mathbb{R}^{n \times N}$ of $N$ activation vectors drawn from a calibration corpus, giving the second-moment matrix $H = \frac{1}{N} X X^\top \in \mathbb{R}^{n \times n}$. The decomposition problem is

$$\min_{Q \in \mathcal{Q}_{b_Q}^{m\times n},\; A \in \mathbb{R}^{m\times r},\; B \in \mathbb{R}^{r \times n}} \; \big\| (W - Q - AB) H^{1/2} \big\|_F^2 .$$

With $H = I$ this is the unweighted problem; with $H$ estimated from data it is the "activation-aware" problem (GPTQ, AWQ, ASVD, SVD-LLM all use some form of $H$).

**Bit budget, as measured.** Storing $Q$ at $b_Q$ bits and the factors $A, B$ at $b_L$ bits gives effective bits per weight

$$b_{\text{eff}} = b_Q + r\left(\tfrac{1}{m} + \tfrac{1}{n}\right) b_L + \frac{\text{scale overhead}}{mn}.$$

For $m=n=4096$, $r=64$, $b_L=16$: the low-rank term costs $0.5$ bits/weight — comparable to moving $Q$ from 2-bit to 2.5-bit. The comparison is only meaningful when $b_{\text{eff}}$ is matched to two decimal places, including group scales and zero-points.

**Assumptions, and which are violated.**

1. *Layerwise decomposability* — minimizing per-layer error minimizes end-task loss. Violated: errors compound and partially cancel across depth; sequential propagation (GPTQ-style) exists precisely because this fails.
2. *$H$ is stationary* — the calibration second moment matches deployment. Violated: $H$ shifts with domain and sequence length, and $Q$ changes the activations that later layers see.
3. *Frobenius error is the right loss* — violated: KL divergence to the FP16 model is the quantity people actually care about, and it is not a quadratic in $W$ except to second order.
4. *Grid is fixed before decomposition* — violated by design in methods that re-fit scales after subtracting $L$.
5. *Rank is a free real parameter* — violated: hardware wants $r$ a multiple of the tensor-core tile, and a rank-64 GEMM has latency out of proportion to its FLOPs.

## 3. State of the Art

**Theory SOTA.** There is no approximation guarantee for the joint problem. Two adjacent hardness facts are established: weighted low-rank approximation is NP-hard (Gillis & Glineur, *SIMAX* 2011), and nearest-point decoding in a general lattice is NP-hard (van Emde Boas 1981; Ajtai 1998 under randomized reductions). Fixing $L$ makes $Q$ a per-entry rounding problem that is trivial for $H=I$ and NP-hard in general for arbitrary $H$; fixing $Q$ makes $L$ an Eckart–Young problem, solvable exactly by SVD for $H=I$ (Eckart & Young 1936). The joint problem inherits both. **Established:** alternating minimization converges to a stationary point. **Not established:** any bound on the gap to the global optimum.

CALDERA (Saha et al., NeurIPS 2024) gives the strongest analysis in this family — error bounds for a randomized low-precision low-rank factorization, building on LPLR (Saha et al., NeurIPS 2023). These bound the error of *their* estimator relative to a rank-$r$ spectral quantity, not the gap to the joint optimum.

**Empirical SOTA.** At sub-3-bit, CALDERA reports Llama-2 7B/13B and Llama-3 8B at roughly 2.1–2.5 effective bits, beating QuIP# (Tseng et al., ICML 2024) at matched budget; the gain is largest at the 2-bit end and shrinks toward 3 bits. **This is a benchmark number** (WikiText-2 perplexity plus zero-shot suites), not a mechanism ablation: the paper does not isolate how much of the gain comes from the low-rank term versus the improved quantizer inside it.

**Claimed but unablated.** (a) That initializing LoRA factors from the quantization residual — LoftQ (Li et al., ICLR 2024) — helps *because* it reduces $\|W - Q - AB\|_F$, rather than because it perturbs the fine-tuning trajectory; the two are confounded whenever $L$ is subsequently trained. (b) That activation-aware weighting $H$ is the right weighting for the *low-rank* part as opposed to the quantized part; ASVD (Yuan et al., 2023) and SVD-LLM (Wang et al., ICLR 2025) assume it without a matched-budget test against $H=I$. (c) That rank should be uniform across layers.

## 4. What Is Known

- **Exact solution of each half.** For $H=I$, $\arg\min_{\text{rank} \le r}\|W - Q - AB\|_F$ is the truncated SVD of $W-Q$ (Eckart–Young 1936). Rounding-to-nearest is optimal for $Q$ given $L$ when $H=I$.
- **The residual is close to full-rank.** Quantization residuals $W-Q$ from round-to-nearest have near-flat spectra: capturing a fixed fraction of residual energy needs rank proportional to the matrix dimension. This is why $r$ in deployed systems is 16–256 for $4096$-dimensional layers and captures only a small share of residual Frobenius energy, yet still moves perplexity — evidence that Frobenius energy is the wrong accounting.
- **Order matters, and neither order dominates.** SVD-first-then-quantize and quantize-first-then-SVD give different errors on the same matrix; §10 exhibits a $2\times2$ instance where greedy SVD-first is arbitrarily worse than the joint optimum.
- **Scale of measured effects.** LoftQ's reported gains over QLoRA (Dettmers et al., NeurIPS 2023) are small at 4-bit and large at 2-bit on DeBERTa/BART/Llama-2 fine-tuning — the pattern is consistent across papers: the low-rank term buys little above 3 bits and a lot below 2.5.
- **Zero-shot accuracy is a low-resolution instrument.** At 7B scale, common-sense zero-shot suites separate methods by 1–3 points with run-to-run and calibration-set variance of a similar order; perplexity is tighter but less aligned with use.

## 5. What Is Not Known

- **Theoretically open.** Whether the joint problem admits a constant-factor approximation, and whether alternating minimization has any bounded gap. No hardness proof for the joint formulation exists either — only for its two relaxations separately. Also open: the optimal $(b_Q, r)$ trade-off curve for any nontrivial random matrix ensemble.
- **Empirically open.** Whether, at matched $b_{\text{eff}}$, spending bits on rank ever beats spending them on the quantizer *at 70B+ scale*. Nearly all matched-budget comparisons are at 7B–13B. The experiment is runnable today; nobody has published it cleanly at 70B with $b_{\text{eff}}$ matched to 0.05 bits.
- **Empirically open.** Whether per-layer rank allocation (more rank in attention output / MLP down-projection) beats uniform rank at fixed total budget.
- **Methodologically blocked.** The objective itself. Layerwise $\|(W-Q-AB)H^{1/2}\|_F$ is a proxy whose correlation with end-task KL divergence has not been characterized at the 2-bit frontier, where the proxy's second-order assumption is weakest. Until a proxy with a measured rank correlation to end-task loss exists, "optimal decomposition" names an objective nobody has validated.

## 6. Why It Is Hard

**The specific obstruction is non-identifiability of credit between the two terms, compounded by a proxy that is not known to order methods correctly.**

Given $(Q, L)$ with good end-task loss, there is no way to attribute the gain. Adding $L$ changes three things at once: it reduces reconstruction error, it re-shapes the residual so the quantizer's scales get re-fit, and it adds a full-precision path through the layer that the optimizer can exploit during any subsequent fine-tuning. A matched-budget baseline that raises $b_Q$ instead changes only the first. So the standard comparison "$Q$ at 2 bits plus rank 64" versus "$Q$ at 2.5 bits" is not a clean contrast of *where the bits went*; it is a contrast of two different function classes.

Second obstruction: **compute cost of the decisive measurement.** Deciding the rank-versus-bits question requires a full sweep over $(b_Q, r)$ at 70B, each point needing calibration, evaluation on several benchmarks, and multiple calibration seeds to beat the noise floor. That is tens of thousands of GPU-hours for a negative-result-shaped answer, which is why it stays unrun.

## 7. Current Research (as of 2026)

- **Joint optimization with better quantizers.** Combining incoherence processing / lattice codebooks (QuIP, QuIP#, Tseng et al.) with a low-rank term — the CALDERA line (Princeton / Stanford, Pilanci, Goldstein). Active direction: replacing the alternating loop with a convex relaxation. *(frontier — verify)*
- **Data-aware rank allocation.** Assigning per-layer rank from a sensitivity score derived from $H$ rather than uniformly. Reported in several 2025 preprints; matched-budget ablations are thin. *(frontier — verify)*
- **Quantization-aware LoRA initialization.** Descendants of LoftQ/LQ-LoRA (Guo et al., ICLR 2024) that fold the fine-tuning objective into the decomposition rather than fitting the residual first.
- **Hardware co-design.** Whether a rank-$r$ side path is worth its latency; fused kernels that compute $Qx + A(Bx)$ in one pass. The rank at which the low-rank path stops being free is a systems number that varies by kernel and has not been standardized.
- **Theory.** Hardness and approximability of quantized-plus-low-rank matrix approximation. No published result yet resolves it.

## 8. Concrete Next Experiment

**Question:** at a fixed effective bit budget, does spending bits on rank beat spending them on the quantizer?

**Scale.** Llama-3.1 70B (and 8B as a scaling control). One calibration corpus (RedPajama, 2048 sequences × 4096 tokens), three calibration seeds.

**Arms**, all matched to $b_{\text{eff}} = 2.50 \pm 0.02$ bits/weight including all scales and zero-points:

| Arm | $Q$ | $L$ |
|---|---|---|
| A (control) | 2.50-bit lattice quantizer, no low-rank | $r=0$ |
| B | 2.25-bit, same quantizer family | $r=32$, FP16 |
| C | 2.00-bit | $r=64$, FP16 |
| D (null) | 2.25-bit | $r=32$, factors **random** and frozen |

Arm D is the essential control: it holds the function class and parameter count fixed while destroying the fitted content of $L$. Arm A holds bits fixed while removing the low-rank path.

**Deciding number.** Mean token-level KL divergence from the FP16 model over a held-out 10M-token mixed corpus, reported with a seed-to-seed standard deviation. The question resolves *yes* if $\mathrm{KL}(C) < \mathrm{KL}(A)$ by more than $3\sigma$ **and** $\mathrm{KL}(C) < \mathrm{KL}(D)$ by more than $3\sigma$. If C beats A but not D, the low-rank path is buying capacity, not reconstruction — a different and more interesting result.

**Secondary readout.** Spearman correlation between layerwise $\|(W-Q-AB)H^{1/2}\|_F$ and end-task KL across all arms and layers. A correlation below about $0.7$ would establish that the field's optimization objective does not order methods the way the deployment metric does — which would make the measurement variant of §1 the binding problem.

Cost estimate: about 12 decomposition runs at 70B plus evaluation; on the order of $10^3$–$10^4$ A100-hours.

## 9. Key References

- **[Foundational]** Carl Eckart, Gale Young. *The approximation of one matrix by another of lower rank.* Psychometrika, 1936.
- **[Foundational]** Nicolas Gillis, François Glineur. *Low-rank matrix approximation with weights or missing data is NP-hard.* SIAM Journal on Matrix Analysis and Applications, 2011.
- **[Foundational]** Emmanuel Candès, Xiaodong Li, Yi Ma, John Wright. *Robust Principal Component Analysis?* Journal of the ACM, 2011. — the low-rank-plus-structured template this problem generalizes
- **[SOTA]** Rajarshi Saha, Naomi Sagan, Varun Srivastava, Andrea Goldsmith, Mert Pilanci. *Compressing Large Language Models using Low Rank and Low Precision Decomposition (CALDERA).* NeurIPS, 2024. — arXiv:2405.18886
- **[SOTA]** Rajarshi Saha, Varun Srivastava, Mert Pilanci. *Matrix Compression via Randomized Low Rank and Low Precision Factorization.* NeurIPS, 2023. — arXiv:2310.11028
- **[SOTA]** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML, 2024. — arXiv:2402.04396
- **[SOTA]** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees.* NeurIPS, 2023. — arXiv:2307.13304
- **[Method]** Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao. *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models.* ICLR, 2024. — arXiv:2310.08659
- **[Method]** Han Guo, Philip Greengard, Eric P. Xing, Yoon Kim. *LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning.* ICLR, 2024. — arXiv:2311.12023
- **[Method]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- **[Method]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[Method]** Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, Guangyu Sun. *ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models.* 2023. — arXiv:2312.05821
- **[Method]** Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang. *SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression.* ICLR, 2025. — arXiv:2403.07378
- **[Survey]** Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang. *A Survey on Model Compression for Large Language Models.* Transactions of the ACL, 2024.
- **[Survey]** Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer. *A Survey of Quantization Methods for Efficient Neural Network Inference.* 2021. — arXiv:2103.13630

## 10. Worked Example

Take $m=n=2$, rank budget $r=1$, and a fixed ternary grid $\mathcal{G} = \{-3, 0, +3\}$ (scale fixed by absmax calibration on a wider tensor, as in group-wise quantization). Let

$$W = \begin{bmatrix} 3.4 & 2.6 \\ 2.6 & 3.4 \end{bmatrix} = \underbrace{\begin{bmatrix} 3 & 3 \\ 3 & 3\end{bmatrix}}_{\text{on the grid}} + \underbrace{\begin{bmatrix} 0.4 & -0.4 \\ -0.4 & 0.4\end{bmatrix}}_{\text{rank }1}.$$

**Greedy SVD-first (the LoRC / ASVD order).** $W$ has singular values $6$ and $0.8$, with top left/right singular vector $(1,1)/\sqrt{2}$. The rank-1 truncation is $L = \begin{bmatrix}3&3\\3&3\end{bmatrix}$. Residual $R = W - L = \begin{bmatrix}0.4&-0.4\\-0.4&0.4\end{bmatrix}$. Rounding $R$ to $\mathcal{G}$ gives $Q = 0$ (every entry is nearer $0$ than $\pm 3$). Final error $\|W - Q - L\|_F^2 = 4(0.4)^2 = 0.64$.

**Joint optimum.** Set $Q = \begin{bmatrix}3&3\\3&3\end{bmatrix}$ — legal, all entries in $\mathcal{G}$ — and $L = \begin{bmatrix}0.4&-0.4\\-0.4&0.4\end{bmatrix}$, which is exactly rank 1. Error $= 0$.

**The obstruction, made visible.** The greedy algorithm loses by an *unbounded* ratio ($0.64$ versus $0$), and it loses for a structural reason: it spent its single unit of rank on a direction the quantizer could have represented for free, leaving the one component the quantizer *cannot* represent unrepresented. The top singular direction is exactly the wrong thing to give to $L$ whenever it lies on the grid.

**And the fix does not generalize.** Reverse the order — quantize $W$ first, then SVD the residual — and this instance is solved exactly. But now perturb the example: with grid $\{-3,0,3\}$ and $W' = \begin{bmatrix}1.6 & 1.4\\ 1.4 & 1.6\end{bmatrix}$, rounding first sends every entry to $0$ (all are nearer $0$ than $3$), leaving residual $W'$ itself, whose best rank-1 approximation $\begin{bmatrix}1.5&1.5\\1.5&1.5\end{bmatrix}$ leaves error $4(0.1)^2 = 0.04$; whereas taking the rank-1 term first gives the same $0.04$. Shift the scale so the grid becomes $\{-1.5, 0, 1.5\}$ and quantize-first is exact while SVD-first is not. **Neither order dominates, the winner depends on the grid's alignment with the spectrum of $W$, and no known algorithm decides which order to use without trying both.** At $4096\times4096$ with $r=64$ this choice is made once, globally, by convention — and the loss from making it wrong is exactly the quantity §8 measures.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*