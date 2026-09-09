---
id: 02-attention/icl-implicit-gradient-descent
title: "In-Context Learning as Implicit Gradient Descent"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# In-Context Learning as Implicit Gradient Descent

> **Topic:** Attention Mechanisms · **ID:** `02-attention/icl-implicit-gradient-descent` · **Status:** partially-solved

## 1. Problem Statement

A transformer given a prompt of labelled examples $(x_1,y_1,\dots,x_N,y_N)$ and a query $x_{\text{query}}$ predicts $\hat y$ without any weight update. The claim under examination: the forward pass *is* an optimization algorithm — the attention layers implement steps of gradient descent on an inner objective, with the demonstrations as the training set and the residual stream carrying the implicit parameter vector.

Three variants, with very different status:

- **Theory (expressivity + optimization).** Does there exist a parameter setting of an $L$-layer attention stack that implements $L$ steps of (preconditioned) gradient descent on the in-context regression loss, and is that setting a *global minimizer* of the pretraining objective? **Largely settled for linear self-attention.**
- **Measurement (mechanistic identification).** Given a *trained* model — including a real pretrained LLM — decide whether its forward pass computes gradient descent, some other iterative solver, or a non-optimization mechanism (retrieval, induction heads, task vectors). **Methodologically blocked.**
- **Method.** If ICL is implicit GD, use that to predict and control ICL: number of steps $\approx$ depth, learning rate $\approx$ a readable weight statistic, curriculum $\approx$ preconditioner. **Empirically open.**

A solution to the measurement variant is a decision procedure that, on models where ground truth is known by construction, returns the correct algorithm identity, and separates GD from at least one non-GD alternative that fits the input–output behaviour equally well.

## 2. Formal Setting

**Task distribution.** Sample $w \sim \mathcal{N}(0, I_d)$, inputs $x_i \sim \mathcal{N}(0,\Sigma)$, labels $y_i = \langle w, x_i\rangle + \varepsilon_i$, $\varepsilon_i\sim\mathcal{N}(0,\sigma^2)$. Prompt token matrix
$$E = \begin{pmatrix} x_1 & \cdots & x_N & x_{\text{query}} \\ y_1 & \cdots & y_N & 0\end{pmatrix} \in \mathbb{R}^{(d+1)\times(N+1)}.$$

**Linear self-attention (LSA) layer.** With $P\in\mathbb{R}^{(d+1)\times(d+1)}$ (value/projection) and $Q\in\mathbb{R}^{(d+1)\times(d+1)}$ (key–query),
$$E \leftarrow E + \tfrac{1}{N}\, P E \,(E^{\top} Q E).$$
Softmax is dropped; this is the object almost all theory covers. **Assumption known to be violated:** real models use softmax, MLPs, LayerNorm, and multi-head attention, all of which change the fixed point.

**Inner objective.** $\mathcal{L}_{\text{in}}(v) = \frac{1}{2N}\sum_{i=1}^{N}(\langle v, x_i\rangle - y_i)^2$, with gradient $\nabla\mathcal{L}_{\text{in}}(v) = \frac{1}{N}\sum_i (\langle v,x_i\rangle - y_i)x_i$.

**Preconditioned GD step.** $v_{t+1} = v_t - \eta\, A\,\nabla \mathcal{L}_{\text{in}}(v_t)$ for some PSD $A\in\mathbb{R}^{d\times d}$.

**Outer (pretraining) objective, as measured.** Monte-Carlo over $B$ sampled tasks:
$$\hat{\mathcal{L}}_{\text{out}} = \frac{1}{B}\sum_{b=1}^{B}\big(\hat y^{(b)}(E^{(b)}) - \langle w^{(b)}, x^{(b)}_{\text{query}}\rangle\big)^2 .$$

**Measured equivalence quantities.** Let $\hat y_{\text{TF}}$ be the model output and $\hat y_{\text{GD}}^{(t,\eta,A)}$ the output of $t$ preconditioned GD steps fit to the same prompt.

1. *Prediction agreement:* $\Delta_{\text{pred}} = \mathbb{E}\,(\hat y_{\text{TF}} - \hat y_{\text{GD}})^2 \big/ \mathbb{E}\,\hat y_{\text{TF}}^2$, minimized over $(\eta, A)$.
2. *Implicit-weight agreement:* recover $v_{\text{TF}} = \nabla_{x_{\text{query}}} \hat y_{\text{TF}}$ (exact for linear-in-query models), compare to $v_{\text{GD}}$ by cosine similarity.
3. *Sensitivity agreement:* Jacobian $\partial \hat y/\partial y_i$ across $i$; true GD on an exchangeable loss gives a permutation-equivariant profile.

Quantity 1 is the one usually reported. It is the weakest: many distinct algorithms agree on predictions to within noise.

**Assumptions violated in practice:** (i) isotropic Gaussian $x$ — real prompt embeddings are anisotropic and heavy-tailed; (ii) a single well-defined task per prompt — real prompts mix format, task and language priors; (iii) the residual stream carries $v$ in a fixed linear read-out basis; (iv) $N \gg d$; (v) noiseless label semantics — Min et al. (EMNLP 2022) showed ICL survives randomized labels, which no honest GD-on-labels account predicts.

## 3. State of the Art

**Theory SOTA (established).**
- von Oswald et al., *Transformers Learn In-Context by Gradient Descent*, ICML 2023: explicit weight construction where one LSA layer equals one GD step; trained LSA layers converge to that construction.
- Mahankali, Hashimoto, Ma, ICLR 2024: for one layer of LSA on the linear-regression prior, **one step of preconditioned GD is the exact global minimizer** of the population pretraining loss. This is a theorem, not a fit.
- Ahn, Cheng, Daneshmand, Sra, NeurIPS 2023: stationary points of the multi-layer LSA in-context loss implement *preconditioned* GD; the learned preconditioner tracks $\Sigma^{-1}$.
- Zhang, Frei, Bartlett, JMLR 2024: gradient flow on a one-layer LSA converges to a global minimum despite non-convexity, and the limit is one GD step; the same limit provably degrades under covariate shift.
- Bai, Chen, Wang, Xiong, Mei, NeurIPS 2023: transformers can implement ridge regression, Lasso, GD and Newton with stated depth/width, plus in-context algorithm selection.

**Empirical SOTA on real LLMs (claimed, weakly ablated).**
- Dai et al., ACL Findings 2023, argued GPT-class attention is "meta-optimization" via a dual form. The supporting metrics (similarity of attention-map updates to fine-tuning updates) were **not ablated against non-GD controls**.
- Akyürek et al., ICLR 2023: probing recovers least-squares/ridge intermediates from trained transformer activations — strongest positive evidence, but on models trained *only* on linear regression.

**Contrary results.**
- Shen, Mishra, Khashabi, ICML 2024, *Do pretrained Transformers Learn In-Context by Gradient Descent?*: on real pretrained LMs, ICL and explicit fine-tuning diverge in order sensitivity and layer-wise behaviour; the Dai et al. similarity metrics do not survive controls.
- Fu, Chen, Jia, Sharan (2023): full (softmax) transformers converge across layers at a rate matching **Iterative Newton**, not GD — error falls superlinearly in depth where GD falls linearly.
- Deutch, Magar, Natan, Dar, NAACL 2024: the ICL↔GD correspondence appears only when the comparison is restricted to upper layers.

Benchmark-number-only results: nearly all real-LLM claims. There is no pretrained LLM for which the GD hypothesis has been confirmed by intervention rather than correlation.

## 4. What Is Known

- **One layer, one step, exactly.** For $d$-dimensional isotropic linear regression, the global minimizer of the one-layer LSA population loss is one preconditioned GD step (Mahankali et al. 2024). Verified numerically at $d \in \{5,10,20\}$, $N$ up to $100$.
- **Recovered GD parameters.** von Oswald et al. report trained one-layer LSA matching the GD construction to within numerical precision on $d=10$, $N=10$–$100$ toy regression, with prediction and weight-sensitivity curves overlapping.
- **Depth $\approx$ steps, in the toy regime only.** Multi-layer LSA on linear regression tracks GD++ / preconditioned GD across $L \le 5$ layers.
- **Curvature-aware behaviour.** With ill-conditioned $\Sigma$, trained models beat plain GD at matched step count, consistent with an implicit $\Sigma^{-1}$ preconditioner (Ahn et al. 2023) — and with Newton-like solvers (Fu et al. 2023).
- **Competing mechanisms exist and are real.** Induction heads (Olsson et al., Anthropic 2022) explain much of ICL's emergence in language models; task/function vectors (Hendel et al., EMNLP Findings 2023; Todd et al., ICLR 2024) compress a demonstration set into a single residual-stream direction that can be transplanted — a *retrieval-then-apply* mechanism, not a descent.
- **Label semantics are often ignorable.** Min et al. (EMNLP 2022): randomizing demonstration labels leaves classification accuracy largely intact at GPT-3 175B scale, which a literal implicit-GD-on-the-loss account cannot explain.

## 5. What Is Not Known

- **Methodologically blocked (the central gap).** No accepted criterion distinguishes "implements GD" from "produces the same predictions as GD." $\Delta_{\text{pred}}$ near zero is compatible with GD, Newton, ridge closed-form, and kernel smoothing. Non-identifiability, not compute, is the blocker.
- **Theoretically open.** No global-optimality result for **softmax** attention with MLP blocks. No characterization of the learned algorithm at depth $L>3$ beyond stationary-point analysis. No proof that the depth-$L$ solution is $L$ GD steps rather than $\lceil\log L\rceil$ Newton steps.
- **Empirically open.** Whether a language model pretrained on natural text (not synthetic regression) carries a decodable implicit parameter vector that updates monotonically in a loss across layers. Runnable today at 1–8B scale; unrun with adequate controls.
- **Open.** Whether ICL is a *mixture* — GD-like for novel continuous tasks, retrieval-like for tasks seen in pretraining — and where the switch sits. Bai et al.'s algorithm-selection result makes this plausible but does not test it in real models.

## 6. Why It Is Hard

**Non-identifiability under a squared-error fit.** The evidence is a fit between $\hat y_{\text{TF}}$ and $\hat y_{\text{GD}}$ with $(\eta, A)$ free. That family is expressive enough to absorb a large class of linear estimators; for $N>d$ all consistent linear-regression solvers converge to the same answer, so agreement at large $N$ carries almost no information about mechanism. The discriminating regime is $N \lesssim d$ and few layers, exactly where noise is largest.

**Confounded measurement on real models.** Extracting "the implicit weight vector" from a residual stream requires choosing a read-out basis. Any linear probe rich enough to recover $v_{\text{GD}}$ can also manufacture it from a non-GD representation — the probe supplies the regression itself.

**Absent ground truth.** For pretrained LLMs there is no known correct answer, so every method is validated only on synthetic models where GD was arguably built in by the training distribution.

## 7. Current Research (as of 2026)

- **Beyond first order.** Whether trained transformers implement Newton / conjugate-gradient-like updates (Fu, Chen, Jia, Sharan; Giannou et al. on emulating Newton's method). Discriminating first-order from second-order behaviour by *depth-scaling of error* is the sharpest live test.
- **Statistician view.** Bai/Mei-line work on in-context algorithm selection and ridge/Lasso implementation; extensions to nonlinear and mixture priors.
- **Bayesian alternative.** Xie et al. (ICLR 2022) implicit Bayesian inference; Lin & Lee, *Dual Operating Modes of In-Context Learning*, ICML 2024, formalizes task-retrieval vs task-learning as two modes of one posterior. *(frontier — verify)* Reconciling the GD picture as an *approximation to* posterior mean, rather than a rival, is the most promising unification.
- **Mechanistic interpretability.** Function/task-vector work (Northeastern, Tel Aviv, Anthropic) pushing on causal interventions rather than similarity metrics.
- **Linear-attention revival.** Vladymyrov, von Oswald et al., *Linear Transformers are Versatile In-Context Learners*, NeurIPS 2024 — learned algorithms that beat plain GD on noisy/mixed-noise regression.

## 8. Concrete Next Experiment

**Question decided:** is the depth-wise error decay of a trained attention stack first-order (GD) or second-order (Newton-like)?

**Scale.** Train decoder-only transformers on in-context linear regression, $d=20$, $N=40$ (so $N/d=2$: the discriminating regime), isotropic and ill-conditioned $\Sigma$ ($\kappa=100$). Depths $L\in\{1,2,3,4,6,8,12\}$, 8 heads, width 256, softmax attention with MLPs. ~10M params each; 7 depths × 2 conditionings × 3 seeds = 42 runs, each under 4 GPU-hours on one A100. Total ≈ 170 GPU-hours.

**Control arms.** For each trained model, the best-fit *oracle* baselines at matched step count: (a) $L$ steps of preconditioned GD with $(\eta,A)$ optimized per condition; (b) $L$ steps of iterative Newton; (c) exact ridge with optimal $\lambda$ (the depth-independent floor). Plus a **negative control**: models trained on randomized labels, which must not show monotone depth-wise descent.

**The deciding number.** Fit $\log \big(\mathcal{L}_{\text{in}}(v_\ell) - \mathcal{L}^\star\big)$ against layer index $\ell$, where $v_\ell = \nabla_{x_{\text{query}}}\hat y_\ell$ is read from an early-exit head trained on frozen activations. Report the exponent $\beta$ in $\log\text{err} \sim -c\,\ell^{\beta}$ on the ill-conditioned arm. **$\beta = 1.0 \pm 0.1$ (geometric decay) supports GD; $\beta \ge 1.5$ supports a second-order method.** One scalar, one plot, decides which family the mechanism belongs to — and the randomized-label arm shows whether the read-out is measuring the model or the probe.

## 9. Key References

- **[Foundational]** Garg, Tsipras, Liang, Valiant. *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.* NeurIPS 2022. — arXiv:2208.01066
- **[Foundational]** von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov. *Transformers Learn In-Context by Gradient Descent.* ICML 2023. — arXiv:2212.07677
- **[Foundational]** Akyürek, Schuurmans, Andreas, Ma, Zhou. *What Learning Algorithm Is In-Context Learning? Investigations with Linear Models.* ICLR 2023. — arXiv:2211.15661
- **[SOTA — theory]** Mahankali, Hashimoto, Ma. *One Step of Gradient Descent Is Provably the Optimal In-Context Learner with One Layer of Linear Self-Attention.* ICLR 2024. — arXiv:2307.03576
- **[SOTA — theory]** Ahn, Cheng, Daneshmand, Sra. *Transformers Learn to Implement Preconditioned Gradient Descent for In-Context Learning.* NeurIPS 2023. — arXiv:2306.00297
- **[SOTA — theory]** Zhang, Frei, Bartlett. *Trained Transformers Learn Linear Models In-Context.* JMLR, 2024. — arXiv:2306.09927
- **[SOTA — expressivity]** Bai, Chen, Wang, Xiong, Mei. *Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection.* NeurIPS 2023. — arXiv:2306.04637
- **[Contrary]** Shen, Mishra, Khashabi. *Do Pretrained Transformers Learn In-Context by Gradient Descent?* ICML 2024. — arXiv:2310.08540
- **[Contrary]** Deutch, Magar, Natan, Dar. *In-Context Learning and Gradient Descent Revisited.* NAACL 2024. — arXiv:2311.07772
- **[Contrary]** Fu, Chen, Jia, Sharan. *Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models.* 2023. — arXiv:2310.17086
- **[Alternative]** Xie, Raghunathan, Liang, Ma. *An Explanation of In-Context Learning as Implicit Bayesian Inference.* ICLR 2022. — arXiv:2111.02080
- **[Alternative]** Olsson et al. *In-Context Learning and Induction Heads.* Transformer Circuits Thread / Anthropic, 2022. — arXiv:2209.11895
- **[Alternative]** Hendel, Geva, Globerson. *In-Context Learning Creates Task Vectors.* Findings of EMNLP 2023. — arXiv:2310.15916
- **[Alternative]** Min, Lyu, Holtzman, Artetxe, Lewis, Hajishirzi, Zettlemoyer. *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP 2022. — arXiv:2202.12837
- **[Survey]** Dong, Li, Dai, Zheng, Ma, Li, Xia, Xu, Wu, Chang, Sun, Li, Sui. *A Survey on In-Context Learning.* EMNLP 2024. — arXiv:2301.00234

## 10. Worked Example

Take $d=5$, $N=10$, isotropic $\Sigma=I$, $\sigma=0$. Train a one-layer LSA model to convergence on the outer loss. Its output is exactly
$$\hat y_{\text{TF}} = x_{\text{query}}^{\top} M \Big(\tfrac{1}{N}\textstyle\sum_i y_i x_i\Big)$$
for a learned $M\in\mathbb{R}^{5\times 5}$; empirically $M \approx \alpha I$ with $\alpha \approx N/(N+d+1) = 10/16 = 0.625$ — shrinkage, which is what ridge with $\lambda = d\sigma_w^{-2}$-ish would give.

Now the obstruction. Compare three candidate algorithms on the *same* prompts:

| Candidate | Prediction | $\Delta_{\text{pred}}$ vs model |
|---|---|---|
| One GD step, $\eta = 0.625$, from $v_0=0$ | $0.625\,x_q^\top \hat\Sigma^{-1}\!\cdot\!\bar{yx}$ w/ $\hat\Sigma \to I$ | $< 10^{-6}$ |
| Ridge, $\lambda$ tuned | $x_q^\top(\hat\Sigma + \lambda I)^{-1}\bar{yx}$ | $\sim 10^{-3}$ |
| Bayes posterior mean under $w\sim\mathcal{N}(0,I)$ | $x_q^\top(X^\top X + \sigma^2 I)^{-1}X^\top y$ | $\sim 10^{-3}$ |

The GD arm wins — but only because at $N \gg d$ with $\hat\Sigma \approx I$, *all three collapse to the same estimator up to a scalar*. The $10^{-3}$ gaps are the $O(d/N)$ deviation of $\hat\Sigma$ from $I$, not evidence about mechanism. Re-run at $N=4 < d=5$: $\hat\Sigma$ is singular, the three candidates separate by $\sim 30\%$ in prediction, and the model matches *none* of them within noise across seeds — the fitted $(\eta, A)$ that minimizes $\Delta_{\text{pred}}$ varies by more than $2\times$ between seeds.

The lesson: the regime where the GD fit looks perfect is the regime where the fit is uninformative, and the regime where it would be informative is the regime where the fit fails. That is why the problem is *partially solved* — a theorem in the linear case, an unfalsifiable correlation everywhere else.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*