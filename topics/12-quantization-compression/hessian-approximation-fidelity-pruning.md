---
id: 12-quantization-compression/hessian-approximation-fidelity-pruning
title: "Hessian Approximation Fidelity in One-Shot Pruning"
topic: 12-quantization-compression
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hessian Approximation Fidelity in One-Shot Pruning

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/hessian-approximation-fidelity-pruning` · **Status:** partially-solved

## 1. Problem Statement

One-shot pruning removes a fraction of a trained network's weights without retraining, using a local quadratic model of the loss. Every method in the Optimal Brain Surgeon (OBS) lineage — WoodFisher, oBERT, Optimal Brain Compression, SparseGPT — picks weights and computes compensating updates from an *approximation* $\hat{H}$ to a curvature matrix $H$. The problem: **does higher fidelity in $\hat{H}$ buy lower end-task loss, and if so, along which axis of fidelity?**

Three variants, with different difficulty:

- **Measurement.** Given a pruned model, attribute its loss gap to (a) the layer-wise proxy objective, (b) the curvature approximation inside that proxy, (c) calibration sampling noise, (d) the dampener. No accepted decomposition exists.
- **Method.** Build $\hat{H}$ that is cheaper *and* better than the layer-wise Gauss–Newton $2XX^\top$ used by SparseGPT, measured by perplexity at matched sparsity and matched calibration budget.
- **Theory.** Bound the excess model-level loss of a mask/update chosen under $\hat{H}$ in terms of $\|\hat{H} - H\|$ and the pruning perturbation norm. No such bound is known for the layer-wise decomposition actually used.

A solution to the method variant is a curvature estimator that beats $2XX^\top$ by a margin exceeding calibration-seed variance. A solution to the theory variant is a non-vacuous excess-risk bound.

## 2. Formal Setting

Model $f_\theta$, $\theta \in \mathbb{R}^d$, loss $\mathcal{L}(\theta) = \mathbb{E}_{x\sim\mathcal{D}}[\ell(f_\theta(x))]$. Pruning perturbs $\theta \to \theta + \delta$ with $\delta$ constrained so $(\theta+\delta)_{\mathcal{M}^c} = 0$ for mask complement $\mathcal{M}^c$, $|\mathcal{M}^c| = s\,d$.

**Second-order model.** At a stationary point ($\nabla\mathcal{L}=0$, assumed),
$$\Delta\mathcal{L} \approx \tfrac12 \delta^\top H \delta, \qquad H = \nabla^2\mathcal{L}(\theta).$$
OBS solves $\min_\delta \frac12\delta^\top H\delta$ s.t. $e_q^\top\delta + \theta_q = 0$, giving saliency and update
$$\rho_q = \frac{\theta_q^2}{2\,[H^{-1}]_{qq}}, \qquad \delta = -\frac{\theta_q}{[H^{-1}]_{qq}}\,H^{-1}e_q .$$

**What is actually measured.** Nobody forms $H \in \mathbb{R}^{d\times d}$ for $d\sim10^{11}$. The objective is replaced by a **per-layer reconstruction** loss. For layer $\ell$ with weights $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ and calibration activations $X\in\mathbb{R}^{d_\text{in}\times N}$,
$$\min_{\widehat{W}} \|WX - \widehat{W}X\|_F^2, \qquad H_\ell = 2XX^\top + \lambda I .$$
$N = n_{\text{seq}}\cdot T$ token columns; SparseGPT uses $n_{\text{seq}}=128$, $T=2048$, so $N = 262{,}144$, drawn from C4. $\lambda = c\cdot \frac{1}{d_\text{in}}\mathrm{tr}(2XX^\top)$ with $c = 10^{-2}$.

Cheaper estimators sit on a fidelity ladder:

| Estimator | Object | Cost per layer |
|---|---|---|
| Magnitude | $\hat H = I$ | $0$ |
| Wanda | $\hat H = \mathrm{diag}(XX^\top)$; score $|W_{ij}|\cdot\|X_j\|_2$ | $O(N d_\text{in})$ |
| SparseGPT / OBC | $\hat H = 2XX^\top+\lambda I$, exact inverse | $O(d_\text{in}^3 + N d_\text{in}^2)$ |
| K-FAC | $A\otimes G$ across layers | $O(d_\text{in}^3 + d_\text{out}^3)$ |
| Exact $\nabla^2\mathcal{L}$ | full Hessian | infeasible |

**Assumptions, and which are violated.**

1. *$\nabla\mathcal{L}=0$.* Violated — LLMs are stopped far from a stationary point; the linear term is nonzero and unmodelled.
2. *Quadratic model valid at the perturbation scale.* Violated at $s\geq0.5$: $\|\delta\|$ is a large fraction of $\|\theta\|$.
3. *Layer-wise separability* — that minimising per-layer $\|\Delta WX\|_F^2$ minimises $\Delta\mathcal{L}$. Violated; SparseGPT propagates already-pruned activations forward, which partially but not fully corrects this.
4. *$H \succ 0$.* The true Hessian is indefinite (Sagun et al., 2018); $2XX^\top$ is PSD by construction, so the Gauss–Newton surrogate discards negative curvature.
5. *Calibration $\mathcal{D}$ matches deployment.* Violated; C4 calibration is used for downstream tasks.

## 3. State of the Art

**Systems/empirical SOTA.** SparseGPT (Frantar & Alistarh, ICML 2023) prunes OPT-175B to 50% unstructured in ~4 GPU-hours with near-zero perplexity change. Wanda (Sun et al., ICLR 2024) discards the inverse entirely and the compensating update entirely, keeping only the diagonal, and matches SparseGPT at 50% on LLaMA-7B/13B/30B/65B. ALPS (Meng et al., NeurIPS 2024) replaces the greedy column-wise solve with ADMM on the same $2XX^\top$ and reports gains that grow with sparsity. CHITA (Benbaki et al., ICML 2023) uses combinatorial optimization over an empirical-Fisher surrogate at ResNet scale.

**Established:** the layer-wise Gauss–Newton objective plus *any* reasonable weighting beats magnitude pruning by a large margin at $\ge50\%$ sparsity in LLMs. **Claimed but unablated:** that the *inverse* — the off-diagonal, compensation-carrying part — is what does the work. Wanda is the direct counterevidence and was published as such.

**Theory SOTA.** OBS optimality holds only for a single weight under an exact PSD $H$ (Hassibi & Stork, 1993). Choosing the optimal mask of size $k$ is combinatorial; no approximation guarantee is known for the greedy column-order schedule SparseGPT uses. Kunstner, Balles & Hennig (NeurIPS 2019) prove the empirical Fisher does not converge to the Hessian away from a minimum with a well-specified model — directly relevant, and largely ignored by the pruning literature.

**Benchmark-number-only results:** essentially all LLM pruning comparisons. Reported perplexities are single-seed, single calibration draw, on WikiText-2 with C4 calibration.

## 4. What Is Known

- **50% unstructured, LLaMA-7B, WikiText-2 ppl:** dense $\approx5.68$; magnitude $\approx17.3$; SparseGPT $\approx7.22$; Wanda $\approx7.26$. The full-inverse method and the diagonal-only method differ by $\approx0.04$ ppl — smaller than the spread from changing calibration set. (7B, reported in Sun et al., ICLR 2024.)
- **OPT-175B, 50% unstructured:** SparseGPT recovers dense-level raw WikiText-2 perplexity ($\approx8.2$ vs $8.34$ dense); magnitude pruning diverges to $\gg10^2$. (175B, Frantar & Alistarh, ICML 2023.)
- **The gap opens with structure and sparsity.** At 2:4 semi-structured and at 60–70% unstructured, full-inverse methods and ADMM solvers separate from Wanda by $\ge0.3$–$1$ ppl at 7B (Sun et al. 2024; Meng et al. 2024).
- **Calibration data matters as much as curvature form.** Williams & Aletras (ACL 2024) show downstream-task variation across calibration corpora comparable to the method gaps above, at 7B.
- **Curvature spectrum.** LLM/DNN Hessians are dominated by a small outlier subspace of dimension $\approx$ number of classes/clusters, with a bulk near zero and a negative tail (Sagun et al. 2018; Ghorbani et al., ICML 2019, at ResNet/CIFAR scale). The pruning surrogate models only the PSD bulk.
- $2XX^\top$ is full-rank here ($N=262{,}144 \gg d_\text{in}=4096$) but ill-conditioned by outlier feature channels; the dampener $\lambda$ is load-bearing, not cosmetic.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $\Delta\mathcal{L}(\hat{H}) - \Delta\mathcal{L}(H) \le g(\|\hat H - H\|, \|\delta\|)$ for the layer-wise decomposition. No approximation ratio for greedy mask selection under correlated $H$. No characterisation of when the Gauss–Newton surrogate's missing negative curvature changes the optimal mask.
- **Empirically open.** Whether a *better* curvature estimator exists at the same cost — e.g. K-FAC-style cross-layer coupling, or activation-covariance shrinkage instead of ridge dampening — evaluated at $\ge$7B with multi-seed calibration. Runnable today; not run at that scale.
- **Methodologically blocked.** The fidelity/accuracy attribution itself. "Fidelity" has no agreed metric: relative Frobenius error, subspace overlap of top eigenvectors, and induced-mask agreement rank estimators differently, and no ground-truth $H$ exists at 7B to measure against.

## 6. Why It Is Hard

**Confounded measurement, plus absent ground truth.** The layer-wise proxy and the curvature estimate are varied together and evaluated by one scalar (perplexity), so the improvement cannot be attributed. Worse, they *trade off*: a more faithful $\hat H_\ell$ makes the layer-wise reconstruction tighter, but the layer-wise objective is not the model loss, so tightening it can move end-task loss the wrong way. This is the mechanism that lets Wanda tie SparseGPT — perfect optimisation of the wrong objective.

**Non-identifiability of $\lambda$.** The dampener is not a hyperparameter of the problem; it is a hyperparameter of the estimator. Two methods with different $\lambda$ are not comparing curvature forms. Papers rarely re-tune $\lambda$ per baseline.

**No reference Hessian.** At 7B, $H$ has $\sim4\times10^{19}$ entries. Hessian-vector products give top eigenpairs (PyHessian, Yao et al. 2020), not $[H^{-1}]_{qq}$ for $10^9$ coordinates. So "fidelity" cannot be measured directly at the scale where it matters — only at $10^6$-parameter proxies where the conclusion may not transfer.

## 7. Current Research (as of 2026)

- **IST Austria (Alistarh group)** — successors to SparseGPT/OBC; sparse-plus-quantized joint compression and better layer-wise solvers.
- **MIT ORC (Mazumder group)** — ALPS/CHITA line: exact ADMM and combinatorial solvers on the fixed $2XX^\top$, isolating solver quality from curvature quality. This is the cleanest existing ablation of the two factors.
- **Calibration-robustness work** — shrinkage and multi-corpus averaging of $XX^\top$ instead of a single 128-sequence draw *(frontier — verify)*.
- **Cross-layer curvature** — K-FAC-style block coupling and global saliency budgets replacing uniform per-layer sparsity *(frontier — verify)*.
- **Curvature-free structured pruning** — SliceGPT (Ashkboos et al., ICLR 2024) removes the question by rotating into a PCA basis, sidestepping rather than solving it.

## 8. Concrete Next Experiment

**Question:** at fixed calibration budget, does off-diagonal curvature information change end-task loss beyond seed noise?

- **Scale:** LLaMA-2-7B and LLaMA-2-13B, 50% and 70% unstructured, plus 2:4.
- **Arms** (identical masks-per-layer budget, identical $N=128\times2048$ C4 draw, $\lambda$ re-tuned per arm over $c\in\{10^{-3},10^{-2},10^{-1}\}$):
  1. **Control:** SparseGPT, full $(2XX^\top+\lambda I)^{-1}$.
  2. Diagonal-only $\hat H$, *with* the SparseGPT compensation step (isolates saliency from update — Wanda drops both).
  3. Full $\hat H$ for saliency, no compensation update (the mirror ablation).
  4. Ledoit–Wolf shrinkage $\hat H = (1-\alpha)XX^\top + \alpha\,\bar{\sigma}I$ in place of ridge.
- **Seeds:** 5 independent calibration draws per arm.
- **Deciding number:** the mean WikiText-2 perplexity gap between arm 1 and arm 2 at 70% sparsity, against the across-seed standard deviation within arm 1. If $|\Delta\text{ppl}| < 2\sigma_{\text{seed}}$, off-diagonal curvature is not doing the work and the field's second-order framing is decorative. If $\Delta\text{ppl} > 2\sigma_{\text{seed}}$ at 70% but not 50%, the answer is sparsity-dependent and the 50%-only literature is under-powered. Cost: ~40 pruning runs, well under 200 A100-hours.

## 9. Key References

- **[Foundational]** Y. LeCun, J. Denker, S. Solla. *Optimal Brain Damage.* NeurIPS 1989.
- **[Foundational]** B. Hassibi, D. Stork. *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon.* NeurIPS 1992.
- **[SOTA]** E. Frantar, D. Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML 2023. — arXiv:2301.00774
- **[SOTA]** M. Sun, Z. Liu, A. Bair, J. Z. Kolter. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR 2024. — arXiv:2306.11695
- **[SOTA]** X. Meng, K. Behdin, H. Wang, R. Mazumder. *ALPS: Improved Optimization for Highly Sparse One-Shot Pruning for Large Language Models.* NeurIPS 2024. — arXiv:2406.07831
- E. Frantar, D. Alistarh. *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning.* NeurIPS 2022. — arXiv:2208.11580
- S. P. Singh, D. Alistarh. *WoodFisher: Efficient Second-Order Approximation for Neural Network Compression.* NeurIPS 2020. — arXiv:2004.14340
- E. Kurtic et al. *The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models.* EMNLP 2022. — arXiv:2203.07259
- R. Benbaki et al. *Fast as CHITA: Neural Network Pruning with Combinatorial Optimization.* ICML 2023. — arXiv:2302.14623
- **[Theory]** F. Kunstner, L. Balles, P. Hennig. *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent.* NeurIPS 2019. — arXiv:1905.12558
- J. Martens, R. Grosse. *Optimizing Neural Networks with Kronecker-factored Approximate Curvature.* ICML 2015. — arXiv:1503.05671
- L. Sagun, U. Evci, V. U. Güney, Y. Dauphin, L. Bottou. *Empirical Analysis of the Hessian of Over-Parametrized Neural Networks.* ICLR 2018 Workshop. — arXiv:1706.04454
- B. Ghorbani, S. Krishnan, Y. Xiao. *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML 2019. — arXiv:1901.10159
- M. Williams, N. Aletras. *On the Impact of Calibration Data in Post-training Quantization and Pruning.* ACL 2024. — arXiv:2311.09755
- **[Survey]** D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, J. Guttag. *What is the State of Neural Network Pruning?* MLSys 2020. — arXiv:2003.03033

## 10. Worked Example

Three weights in one output row, $w = (0.95,\ 1.0,\ 1.0)$, with normalised layer Hessian
$$H = \begin{pmatrix}1 & 0 & 0\\ 0 & 1 & 0.9\\ 0 & 0.9 & 1\end{pmatrix},$$
i.e. inputs 2 and 3 are near-duplicate channels ($r=0.9$); input 1 is independent. Prune one weight.

**Diagonal estimator (Wanda-style).** Score $w_i^2 H_{ii} = (0.9025,\ 1.0,\ 1.0)$. Prunes **weight 1**. True cost, with optimal compensation, $\rho_1 = w_1^2/(2[H^{-1}]_{11}) = 0.9025/2 = 0.4513$.

**Full inverse (OBS).** The $2\times2$ block has $\det = 0.19$, so $[H^{-1}]_{22}=[H^{-1}]_{33}=1/0.19=5.263$, $[H^{-1}]_{11}=1$. Saliencies $\rho = (0.4513,\ 0.0950,\ 0.0950)$. Prunes **weight 2**, at $4.75\times$ lower layer-wise cost, because weight 3 absorbs it via $\delta_3 = +0.9$.

The masks disagree, and on the proxy objective the full inverse is 4.75× better. **Now the obstruction.** Apply SparseGPT's dampener: $\mathrm{mean}(\mathrm{diag}\,H)=1$, so $\lambda = 0.01$. The damped block has $\det = 1.01^2-0.81 = 0.2101$, giving $[H^{-1}]_{22} = 1.01/0.2101 = 4.807$ and $\rho_2 = 0.1040$ — a **9.5% shift in saliency from a 1% regularizer**, because the block's small eigenvalue $0.1$ moves to $0.11$. The saliency is a function of the estimator's hyperparameter as much as of the curvature.

And at 7B scale the 4.75× proxy win does not appear in the metric: Wanda's masks, which make exactly this kind of "wrong" choice everywhere, land within $0.04$ WikiText-2 perplexity of SparseGPT at 50%. Either near-duplicate channel pairs are rare in real layers, or the layer-wise objective the 4.75× is measured in is not the objective that matters. **Distinguishing those two explanations is the open problem**, and no published ablation does it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*