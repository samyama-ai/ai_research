---
id: 13-parameter-efficient-adaptation/gradient-free-adaptation-complexity
title: "Gradient-Free Adaptation Sample Complexity"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Gradient-Free Adaptation Sample Complexity

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/gradient-free-adaptation-complexity` · **Status:** open

## 1. Problem Statement

Adapting a pretrained model without backpropagation — using only forward passes (zeroth-order gradient estimates, evolution strategies, or API-only prompt/prefix search) — costs memory roughly equal to inference and works through black-box endpoints. The open question is what it costs in *queries*.

**Input.** A frozen model $f_\theta$, a task dataset $D$, an adaptation parameter space $\mathcal{P} \subseteq \mathbb{R}^p$ (LoRA weights, prefix vectors, soft prompts, or a random projection thereof), and a budget of $N$ forward passes.

**Output.** $\hat{z} \in \mathcal{P}$ and the loss gap to first-order fine-tuning under an equal *compute* budget.

**Decision predicate.** Does the forward-pass count $N(\epsilon)$ needed to reach excess loss $\epsilon$ scale with the ambient trainable dimension $p$, or with some smaller intrinsic quantity — and what is that quantity?

Three variants, routinely conflated:

- **Theory.** Prove a rate for $N(\epsilon)$ on non-convex objectives whose dimension dependence is a *measurable* property of the loss landscape, not an unverifiable assumption.
- **Method.** Build a gradient-free adapter whose query multiplier over first-order tuning is $O(1)$–$O(10)$ rather than $O(10^2)$.
- **Measurement.** Define the comparison fairly. Steps are not comparable across optimizers; only forward-equivalent FLOPs and wall-clock under matched memory are.

## 2. Formal Setting

Loss $\mathcal{L}(z) = \mathbb{E}_{(x,y)\sim D}[\ell(f_{\theta \oplus z}(x), y)]$, where $\oplus$ merges adaptation parameters $z \in \mathbb{R}^p$ into the frozen $\theta \in \mathbb{R}^d$.

**Simultaneous perturbation estimator** (SPSA; Spall 1992), the basis of MeZO:

$$\widehat{\nabla}\mathcal{L}(z) = \frac{\mathcal{L}(z + \mu u; B) - \mathcal{L}(z - \mu u; B)}{2\mu}\, u, \qquad u \sim \mathcal{N}(0, I_p)$$

with smoothing radius $\mu$ and minibatch $B$. **Measured as:** two forward passes per step, same $B$ and same $u$ on both, $u$ regenerated from a stored seed so no $p$-sized buffer is held.

**Query budget.** $N$ = total forward passes. **Compute budget** in forward-equivalents: a first-order step costs $\approx 3$ (one forward $\approx 1$, backward $\approx 2$), a ZO step costs exactly $2$. So the honest multiplier is

$$M \;=\; \frac{2\,T_{\mathrm{ZO}}}{3\,T_{\mathrm{FO}}}$$

for step counts $T$ reaching equal validation loss. Reporting $T_{\mathrm{ZO}}/T_{\mathrm{FO}}$ instead inflates the gap by $1.5\times$; reporting peak memory instead of FLOPs hides it entirely.

**Effective rank.** Malladi et al. (2023) replace ambient $p$ with the local effective rank $r$ of the Hessian: on a region containing the trajectory, $\nabla^2 \mathcal{L}(z) \preceq H(z)$ with $\mathrm{tr}(H)/\|H\|_{op} \le r$. **Measured as:** stochastic Lanczos or Hutchinson trace estimates on the actual adapter subspace along the actual trajectory — which is what nobody reports at scale.

**Assumptions and their violations.**
- *$L$-smoothness of $\mathcal{L}$ in $z$.* Violated: transformer losses have unbounded local curvature; layer-norm and attention softmax produce sharp regions.
- *Unbiasedness of $\widehat{\nabla}$ as $\mu \to 0$.* Violated in bf16/fp16 — for small $\mu$ the numerator is dominated by rounding, so $\mu$ is set by numerical precision (typically $10^{-3}$), not by bias-variance optimality.
- *Constant effective rank along the trajectory.* Unverified; $r$ is defined on a region assumed to contain the whole optimization path.
- *PL / strong-growth condition* used for the global-convergence result. Not testable at billion-parameter scale.

## 3. State of the Art

**Theory SOTA (established).**
- Nesterov & Spokoiny (*Found. Comput. Math.*, 2017): random gradient-free methods on convex $\mathcal{L}$ incur a factor $\approx p$ over first-order complexity.
- Ghadimi & Lan (*SIAM J. Optim.*, 2013): stochastic ZO on smooth non-convex objectives reaches an $\epsilon$-stationary point in $O(p/\epsilon^2)$ queries.
- Duchi, Jordan, Wainwright & Wibisono (*IEEE Trans. Inf. Theory*, 2015) and Jamieson, Nowak & Recht (NeurIPS 2012): matching $\Omega(\sqrt{p/T})$ lower bounds. The dimension factor is **not** removable in the worst case.
- Malladi et al. (NeurIPS 2023): under a local-effective-rank assumption plus PL, MeZO's rate depends on $r$, not $p$. This is a **conditional** dimension-free result; the assumption is asserted, not measured at the scale of the experiments.

**Empirical SOTA.**
- **MeZO** (Malladi et al., NeurIPS 2023): OPT-13B, SuperGLUE, 16-shot-per-class prompted setting, reported within ~1 point of full fine-tuning on several tasks with up to $12\times$ memory reduction; runs OPT-30B on one 80 GB A100. Established: the memory claim. **Claimed but unablated at parity:** that the accuracy is reached at comparable compute — MeZO runs $\sim$100K steps against fine-tuning's $\sim$1K.
- **BBT / BBTv2** (Sun et al., ICML 2022 / EMNLP 2022): CMA-ES over a 500-dimensional random projection of a soft prompt, RoBERTa-large, ~8K API queries, 16-shot; BBT reports ~89% on SST-2, above few-shot model tuning in that regime. Benchmark number only: it does not transfer to generative tasks or to full-data regimes.
- **ZO benchmark** (Zhang et al., *Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark*, ICML 2024): six-plus ZO variants across model families; finds strong task- and prompt-dependence and high seed variance.
- **Forward gradients** (Baydin et al. 2022; Ren et al., ICLR 2023): unbiased directional-derivative estimators degrade badly with width unless local losses shrink the effective dimension — independent evidence that dimension, not bias, is the binding constraint.

## 4. What Is Known

- **Memory is genuinely solved.** ZO fine-tuning holds inference-sized state: no activations, no optimizer moments. $12\times$ reduction measured on OPT-13B (Malladi et al. 2023).
- **The worst-case $p$ factor is tight.** $\Omega(\sqrt{p/T})$ holds even for strongly convex objectives with two function evaluations per step (Duchi et al. 2015).
- **Prompting matters more than the optimizer.** MeZO's ablations show non-prompted ZO fine-tuning is far weaker; the pretrained prompt narrows the search basin.
- **Step-count multipliers are large.** $\sim$100K ZO steps vs $\sim$1K first-order steps on OPT-13B $\Rightarrow$ $M \approx 67$ in forward-equivalents. Wall-clock favors ZO only when the first-order run does not fit in memory at all.
- **Query-efficient black-box adaptation exists in a narrow regime**: $\le 10^4$ queries, encoder models $\le$ 355M parameters, $\le 500$-dimensional search space, classification only (BBT).
- **PEFT parameterization changes $p$ by orders of magnitude** without changing accuracy much in the first-order setting (LoRA, Hu et al., ICLR 2022) — giving a ready-made dial for testing dimension dependence that has not been systematically swept for ZO.

## 5. What Is Not Known

- **Theoretically open.** Whether any *verifiable* landscape property of pretrained transformers implies sub-linear-in-$p$ ZO complexity. Effective rank is sufficient if it holds; no one has proven it holds, and no lower bound rules out the reverse. Also open: whether the $r$-dependent bound survives removal of the PL condition.
- **Empirically open.** The scaling of $M$ with trainable-parameter count $p$, model size $d$, and task difficulty. Runnable today: sweep LoRA rank $\times$ model size $\times$ seeds. Nobody has published the clean grid. Also open: whether $M$ grows or shrinks with model scale — the two published points (RoBERTa-large, OPT-13B) use different tasks, budgets and estimators, so they cannot be joined into a trend.
- **Methodologically blocked.** The effective rank $r$ is defined on "a region containing the trajectory" whose extent is unspecified. Two analysts computing $r$ on the same run can differ by orders of magnitude by choosing different regions and different Hessian upper bounds $H$. Until $r$ has a fixed estimation protocol, "MeZO converges at rate governed by $r$" is not falsifiable.

## 6. Why It Is Hard

**Confounded measurement, plus non-identifiability of the dimension term.**

Every reported ZO-vs-FO comparison varies at least four things at once: step count, batch size, prompt template, and trainable-parameter count. Accuracy differences of 1–2 points on 16-shot SuperGLUE sit inside the seed variance the benchmark itself reports, so the comparison cannot resolve the quantity of interest ($M$) at the precision needed.

The theory obstruction is separate. The bound has the form $N \propto r/\epsilon^2$ with $r$ unmeasured. Any observed $N$ is consistent with the theory for *some* $r$ — so the theory currently cannot be refuted by an experiment, only fitted to one. Distinguishing "dimension-free with large constant" from "dimension-dependent with small constant" requires varying $p$ over $\ge 2$ decades at fixed everything else, and each cell of that grid is a full 100K-step run.

Compute cost compounds it: a single MeZO run on OPT-13B is $\sim 2\times10^5$ forward passes. A $4 \times 3 \times 5$ grid (rank $\times$ size $\times$ seed) is on the order of $10^7$ forward passes of a 13B model — days to weeks on a modest cluster. Affordable, but nobody's headline result depends on it, so it goes unrun.

## 7. Current Research (as of 2026)

- **Variance reduction and structure.** Sparse perturbation (Sparse-MeZO), low-rank perturbation subspaces (LOZO), and Hessian-informed preconditioning (HiZOO) all aim to cut the effective $p$ in the estimator. Reported gains are on the same 16-shot SuperGLUE protocol, so they inherit its variance problem. *(frontier — verify each ablation.)*
- **Hybrid first-/zeroth-order.** Adding a small number of true gradient steps or gradient-based warm start to a mostly-ZO run (Addax and similar). Promising because it directly targets $M$ rather than memory. *(frontier — verify.)*
- **Gradient-free training from scratch.** DeepZero (Chen et al., ICLR 2024) shows ZO training of small CNNs to non-trivial accuracy via coordinate-wise sparse estimation — evidence about what dimension reduction buys.
- **API-only adaptation.** Continuing work on black-box prompt search for closed models, now aimed at generative tasks rather than classification.
- **Groups.** Princeton (Malladi, Chen, Arora and collaborators) on MeZO and successors; Michigan State / IBM (Sijia Liu's group) on the ZO benchmark and DeepZero; Fudan (Xipeng Qiu's group) on BBT-line black-box tuning.

## 8. Concrete Next Experiment

**The dimension-dependence sweep.**

- **Scale.** OPT-1.3B and OPT-13B. Tasks: SST-2 (easy, classification), RTE (hard, classification), SQuAD (generative). LoRA adapters at rank $\in \{1, 4, 16, 64, 256\}$, giving trainable $p$ from $\sim 4\times10^4$ to $\sim 10^7$ — over two decades at fixed model, fixed prompt, fixed data. 5 seeds per cell.
- **Protocol.** For each cell, run MeZO to a fixed target validation loss $\epsilon^\star$ (set by the first-order arm), recording $T_{\mathrm{ZO}}$; cap at $5\times10^5$ steps and record censoring. Measure the Hessian effective rank $r$ by Hutchinson trace + power-iteration top eigenvalue on the LoRA subspace, at 10 checkpoints along each trajectory, with the estimation protocol fixed in advance and published.
- **Control arm.** AdamW LoRA fine-tuning at the *same* rank, same data order, same prompt, same 5 seeds, to the same $\epsilon^\star$, giving $T_{\mathrm{FO}}$. This is the control that makes the comparison about the optimizer rather than the parameterization.
- **Deciding number.** The slope $\alpha$ of $\log M$ against $\log p$, where $M = 2T_{\mathrm{ZO}}/(3T_{\mathrm{FO}})$, fitted per model and task with seed variance as the error bar.
  - $\alpha \approx 0$ (95% CI excluding 0.3): the effective-rank picture holds; ZO complexity is set by landscape structure and the engineering target is the constant.
  - $\alpha \approx 0.5$–$1.0$: ZO adaptation is dimension-bound, the dimension-free result does not describe practice, and gradient-free PEFT must be confined to the smallest workable $p$.
  - Secondary check: does measured $r$ track $M$ across cells ($R^2$ of $\log M$ on $\log r$)? If not, effective rank is the wrong quantity regardless of $\alpha$.

Cost estimate: $\approx 10^7$ forward-equivalents at 13B for the ZO arms — roughly 1–2 weeks on 8 A100s. This is the smallest grid that separates the two hypotheses.

## 9. Key References

- **[Foundational]** James C. Spall. *Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation.* IEEE Transactions on Automatic Control, 1992.
- **[Foundational]** Yurii Nesterov, Vladimir Spokoiny. *Random Gradient-Free Minimization of Convex Functions.* Foundations of Computational Mathematics, 2017.
- **[Foundational]** Saeed Ghadimi, Guanghui Lan. *Stochastic First- and Zeroth-order Methods for Nonconvex Stochastic Programming.* SIAM Journal on Optimization, 2013.
- **[Lower bound]** John Duchi, Michael I. Jordan, Martin Wainwright, Andre Wibisono. *Optimal Rates for Zero-Order Convex Optimization: The Power of Two Function Evaluations.* IEEE Transactions on Information Theory, 2015.
- **[Lower bound]** Kevin Jamieson, Robert Nowak, Benjamin Recht. *Query Complexity of Derivative-Free Optimization.* NeurIPS, 2012.
- **[SOTA]** Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora. *Fine-Tuning Language Models with Just Forward Passes.* NeurIPS, 2023. — arXiv:2305.17333
- **[SOTA]** Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, Xipeng Qiu. *Black-Box Tuning for Language-Model-as-a-Service.* ICML, 2022.
- **[SOTA]** Tianxiang Sun, Zhengfu He, Hong Qian, Yunhua Zhou, Xuanjing Huang, Xipeng Qiu. *BBTv2: Towards a Gradient-Free Future with Large Language Models.* EMNLP, 2022.
- **[Benchmark]** Yihua Zhang et al. *Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark.* ICML, 2024.
- **[Related]** Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu. *DeepZero: Scaling Up Zeroth-Order Optimization for Deep Model Training.* ICLR, 2024.
- **[Related]** Atılım Güneş Baydin, Barak A. Pearlmutter, Don Syme, Frank Wood, Philip Torr. *Gradients without Backpropagation.* arXiv preprint, 2022.
- **[Related]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.

## 10. Worked Example

**Setup.** OPT-1.3B, RTE, LoRA rank 16 on query/value projections of all 24 layers: $p = 2 \times 24 \times 2 \times 16 \times 2048 \approx 3.1$M trainable parameters.

**What the worst-case theory predicts.** Ghadimi–Lan gives $N = O(p/\epsilon^2)$. Take a first-order run reaching the target in $T_{\mathrm{FO}} = 1{,}000$ steps. The dimension-scaled prediction is $T_{\mathrm{ZO}} \sim p \cdot T_{\mathrm{FO}} = 3.1\times10^9$ steps — $6.2\times10^9$ forward passes. At 25 ms per forward pass with batch 16, that is **4.9 years on one GPU.** ZO fine-tuning of LLMs would be impossible.

**What actually happens.** MeZO-style runs converge in $\sim 10^5$ steps. So the observed multiplier is

$$M = \frac{2 \times 10^5}{3 \times 10^3} \approx 67,$$

not $3.1\times10^6$. The gap between prediction and practice is a factor of $\sim 5\times10^4$.

**Where the obstruction becomes visible.** Fit the effective-rank story to this: $M \approx 67$ implies $r \approx 10^2$, i.e. the loss behaves as if it had ~100 relevant directions in a 3.1M-dimensional adapter space. Nothing rules that out — and nothing confirms it, because no published run measures $r$ on the trajectory. The same $M = 67$ is equally consistent with:

| Hypothesis | Implied form | Prediction at rank 256 ($p \approx 50$M) |
|---|---|---|
| Dimension-free, $r \approx 10^2$ | $M$ constant in $p$ | $M \approx 67$ |
| $M \propto \sqrt{p}$ with small constant | $M \approx 0.038\sqrt{p}$ | $M \approx 270$ |
| $M \propto p$ with tiny constant | $M \approx 2\times10^{-5} p$ | $M \approx 1{,}000$ |

All three fit the single published point exactly. They differ by $15\times$ at rank 256 — a difference that is trivially measurable and has never been measured. That is the whole problem: the theory is unfalsifiable at one point, the second point costs a week of GPU time, and the field has spent its compute on memory numbers instead.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*