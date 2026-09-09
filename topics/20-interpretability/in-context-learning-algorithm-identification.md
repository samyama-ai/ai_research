---
id: 20-interpretability/in-context-learning-algorithm-identification
title: "In-Context Learning Algorithm Identification"
topic: 20-interpretability
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# In-Context Learning Algorithm Identification

> **Topic:** Interpretability · **ID:** `20-interpretability/in-context-learning-algorithm-identification` · **Status:** partially-solved

## 1. Problem Statement

A transformer shown $(x_1,y_1),\dots,(x_n,y_n),x_{n+1}$ in its context predicts $\hat y_{n+1}$ without a weight update. The problem: **name the algorithm the forward pass runs**, and prove the name is right.

Three variants, routinely conflated:

- **Measurement.** Given a trained model and a task family, decide which member of a candidate algorithm set $\mathcal{A}=\{\text{OLS}, \text{ridge}_\lambda, k\text{-step GD}, \text{Newton}, \text{kNN}, \text{Bayes posterior mean},\dots\}$ the model is behaviorally closest to, with a calibrated notion of "closest" and an accept/reject decision, not a ranking.
- **Method.** Recover the algorithm from weights and activations — a circuit whose intermediate states correspond to the named algorithm's intermediate states, validated causally (ablate the claimed carrier, the claimed algorithmic step breaks and only it).
- **Theory.** Prove that the training objective's minimizer (or the gradient-flow limit) over a stated architecture class *is* a specific algorithm, with an approximation and optimization guarantee.

Solved would mean: for at least one non-toy model and task family, a statement of the form "layers $\ell_1..\ell_2$ implement one step of preconditioned gradient descent with preconditioner $P$ read off the weights", supported by (i) prediction agreement below a stated tolerance under distribution shift chosen to separate candidates, and (ii) causal edits that transfer as the algorithm predicts.

## 2. Formal Setting

Task distribution $\mathcal{T}$; a task $w\sim\mathcal{T}$ induces $y=f_w(x)+\varepsilon$, $x\sim\mathcal{D}_x$, $\varepsilon\sim\mathcal{N}(0,\sigma^2)$. Prompt $P_n=(x_1,y_1,\dots,x_n,y_n,x_{n+1})$. The model is $T_\theta:P_n\mapsto \hat y_{n+1}\in\mathbb{R}$ (regression readout) or a distribution over tokens (language model). Pretraining loss

$$\mathcal{L}(\theta)=\mathbb{E}_{w,\{x_i\},\varepsilon}\Big[\tfrac{1}{n}\sum_{j=1}^{n}\big(T_\theta(P_j)-f_w(x_{j+1})\big)^2\Big].$$

**Candidate algorithm.** $A:\ \big((x_i,y_i)_{i\le n},x_{n+1}\big)\mapsto\hat y^A_{n+1}$, a deterministic function with hyperparameters $\phi$ (ridge $\lambda$, GD step size $\eta$ and count $k$).

**Behavioral distance,** as measured: draw $M$ prompts from an evaluation distribution $\mathcal{Q}$ and compute

$$\widehat{d}_{\mathcal{Q}}(T_\theta,A_\phi)=\Big(\tfrac{1}{M}\sum_{m=1}^{M}\big(T_\theta(P^{(m)})-A_\phi(P^{(m)})\big)^2\Big)^{1/2},\qquad \phi^\star=\arg\min_\phi \widehat d_{\mathcal{Q}}.$$

Reported normalized: $\widehat d_\mathcal{Q}/\mathrm{sd}(y)$. Fitting $\phi$ on the same $\mathcal{Q}$ used for the verdict is the standard leak; $\phi$ must be fit on $\mathcal{Q}_{\text{fit}}$ and scored on a disjoint $\mathcal{Q}_{\text{test}}$.

**Separation.** Two candidates are distinguishable on $\mathcal{Q}$ only if $\Delta_{\mathcal{Q}}(A,B)=\big(\mathbb{E}_\mathcal{Q}(A-B)^2\big)^{1/2}$ exceeds the model's own irreducible noise floor $\epsilon_{\text{seed}}$ — the spread across pretraining seeds. Identification is only meaningful when $\Delta_\mathcal{Q}(A,B)\gg\epsilon_{\text{seed}}$.

**Mechanistic claim,** as measured: a decoder $g$ from residual-stream activations $h^{(\ell)}$ to the algorithm's state $s_\ell$ (e.g. the iterate $w_k$), scored by $R^2$ of the probe **and** by causal transfer — patch $g^{-1}(s'_\ell)$ into layer $\ell$ and check the output moves to $A$'s output under $s'_\ell$.

**Assumptions, and which fail.** (i) $\mathcal{A}$ contains the true algorithm — almost surely false; the model may run something unnamed. (ii) $\mathcal{D}_x$ isotropic Gaussian, $f_w$ linear — violated for every language model. (iii) One algorithm for all $n$ — violated: models switch regime with context length and with task ambiguity. (iv) Linear attention / no softmax, used by most theory — violated by real architectures.

## 3. State of the Art

**Theory (established).** Bai et al., *Transformers as Statisticians* (NeurIPS 2023): explicit constructions in which a transformer implements ridge regression, gradient descent, Lasso via proximal steps, and performs in-context *algorithm selection*, with error bounds. Ahn et al. (NeurIPS 2023) and Mahankali, Hashimoto & Ma (ICLR 2024): for a **single layer of linear self-attention**, the global minimizer of the population ICL loss on linear regression is exactly one step of preconditioned gradient descent — an identification theorem, not an analogy. Zhang, Frei & Bartlett (JMLR 2024): gradient flow on one-layer linear attention converges to that solution. All are restricted to linear attention and linear tasks.

**Empirical (established).** Garg et al. (NeurIPS 2022): 12-layer GPT-2-scale models trained from scratch match OLS on 20-dim noiseless linear regression across $n$, and degrade gracefully under shift. Akyürek et al. (ICLR 2023): trained models track ridge at small $n$ and OLS at large $n$; probes recover $\hat w$ and moment matrices from activations. Olsson et al. (2022): induction heads, with an ablation-backed phase-change story for pattern-completion ICL in language models.

**Claimed but unablated.** "LLMs perform implicit gradient descent as meta-optimizers" (Dai et al., ACL Findings 2023) rests on a linear-attention identity plus correlational similarity metrics; Shen et al. (ICML 2024) and Deutch et al. (NAACL 2024) re-ran the comparisons on real pretrained LMs and found the claimed GD correspondence does not hold once controls and metrics are fixed. Fu et al. (NeurIPS 2024) argue the layer-wise error curve matches **Iterative Newton**, not GD — same behavior, different name, showing the earlier verdict was under-determined.

**Benchmark-number-only.** Task-vector and function-vector results (Hendel et al., EMNLP Findings 2023; Todd et al., ICLR 2024) report causal task-transfer accuracies but do not name an algorithm; they localize *what* is computed, not *how*.

## 4. What Is Known

- Scale 12-layer, 256-dim, 20-dim linear regression, noiseless: from-scratch transformers reach squared error within a few percent of OLS for $n\ge d$ and beat OLS for $n<d$ by behaving like ridge (Garg et al. 2022; Akyürek et al. 2023).
- One-layer linear attention on isotropic linear regression: the optimum is one preconditioned GD step; proven, closed form (Ahn et al. 2023; Mahankali et al. 2024).
- Layer-wise convergence in deep linear-attention models is **superlinear** in depth on linear regression — inconsistent with fixed-step GD, consistent with a second-order method (Fu et al. 2024).
- ICL is not stable across training: it emerges and then **decays** with continued training under some data distributions (Singh et al., NeurIPS 2023), and is driven by burstiness and a Zipfian label distribution (Chan et al., NeurIPS 2022). "The algorithm" is therefore a function of the training checkpoint.
- In real LLMs at 1B–70B, replacing demonstration labels with random labels degrades accuracy far less than a learning-algorithm account predicts (Min et al., EMNLP 2022) — evidence that much of ICL in language is format/task-location, not regression.

## 5. What Is Not Known

- **Theoretically open.** Whether the loss minimizer over *softmax* attention with $L\ge2$ layers is any named algorithm. All identification theorems assume linear attention or hand-built weights. No lower bound rules out a non-nameable minimizer.
- **Theoretically open.** Identifiability itself: no theorem states conditions on $(\mathcal{Q},\mathcal{A})$ under which behavioral agreement below $\epsilon$ implies mechanistic equivalence.
- **Empirically open.** Whether deep-transformer ICL on linear regression is Newton-like or GD-like once tested on a distribution *designed* to separate them (ill-conditioned $\mathcal{D}_x$), with seed noise measured. Runnable today at ~$10^8$ parameters and modest cost; not run at the right scale with the right control.
- **Methodologically blocked.** ICL in pretrained language models. There is no agreed task family, no ground-truth algorithm, and no candidate set $\mathcal{A}$ with more than one plausible member. The measurement is undefined, not merely unrun.

## 6. Why It Is Hard

**Non-identifiability is the obstruction.** Algorithms that differ mechanistically can agree behaviorally to within measurement noise on the very distribution used to train and test them. OLS, ridge with small $\lambda$, $k$-step GD with large $k$, and Newton all converge to the same estimator on well-conditioned isotropic data — exactly the data the field uses. The distinguishing observable (convergence *rate* as a function of depth, or behavior under conditioning shift) is off-distribution, where the model's own behavior degrades for reasons unrelated to which algorithm it runs. Secondary: seed-to-seed variation in $T_\theta$ is often the same order as $\Delta_\mathcal{Q}$ between candidates, so agreement scores are reported without an error bar that would make them falsifiable. Third: probes fit to predict $\hat w$ from activations succeed partly because $\hat w$ is a smooth function of the inputs the layer already carries — probe $R^2$ without causal transfer does not license the claim.

## 7. Current Research (as of 2026)

- Provable ICL beyond linear attention — softmax, multi-layer, non-Gaussian covariates (groups around Song Mei, Suvrit Sra, Peter Bartlett). *(frontier — verify current results)*
- Bayesian/task-mixture accounts as the null hypothesis: ICL as posterior inference over a pretraining task prior (Xie et al. 2022; Lin & Lee, ICML 2024; Panwar et al., ICLR 2024), with the dual-mode picture — retrieval of a pretrained task vs. learning a new one.
- Higher-order-method hypotheses following Fu et al. 2024, testing depth-vs-accuracy curves as the discriminating signal.
- Function/task vectors and their causal transfer as a mechanism-level substrate for language ICL (Todd et al.; Hendel et al.).
- Training-dynamics accounts: transience, data distributional drivers, and the induction-head phase change as the thing to be explained. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** is deep-transformer ICL on linear regression GD-like or Newton-like?

- **Scale.** 12-layer, $d_{\text{model}}=256$ decoder (~$2\times10^7$ params), Garg et al. setup, $d=20$, $n\le40$, **5 pretraining seeds** — the seed spread is the error bar and must be reported. Cost: single-GPU-days.
- **Manipulation.** Evaluate on covariates with condition number $\kappa\in\{1,10,10^2,10^3\}$ (anisotropic $\Sigma$), a regime where GD's error decays as $(1-1/\kappa)^k$ and Newton's as $\kappa$-independent quadratic. Fit each candidate's hyperparameters on $\mathcal{Q}_{\text{fit}}$, score on disjoint $\mathcal{Q}_{\text{test}}$.
- **Control arm.** The same $\widehat d$ computed between two *independently seeded* models, $\epsilon_{\text{seed}}$. Any candidate whose distance is below $\epsilon_{\text{seed}}$ is not distinguishable from the model and must be reported as such.
- **Deciding number.** $R=\widehat d_{\mathcal{Q}_{\text{test}}}(T_\theta,\text{GD}_{k^\star})/\widehat d_{\mathcal{Q}_{\text{test}}}(T_\theta,\text{Newton}_{k^\star})$ at $\kappa=10^3$. $R>3$ with both distances above $\epsilon_{\text{seed}}$ decides for Newton; $R<1/3$ for GD; $R\in[1/3,3]$, or either distance below $\epsilon_{\text{seed}}$, is a **null result and should be published as one** — it says the setup cannot separate the two.

## 9. Key References

- **[Foundational]** Garg, Tsipras, Liang, Valiant. *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.* NeurIPS 2022. — arXiv:2208.01066
- **[Foundational]** Akyürek, Schuurmans, Andreas, Ma, Zhou. *What learning algorithm is in-context learning? Investigations with linear models.* ICLR 2023. — arXiv:2211.15661
- **[Foundational]** von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov. *Transformers learn in-context by gradient descent.* ICML 2023. — arXiv:2212.07677
- **[Foundational]** Xie, Raghunathan, Liang, Ma. *An Explanation of In-context Learning as Implicit Bayesian Inference.* ICLR 2022. — arXiv:2111.02080
- **[SOTA, theory]** Bai, Chen, Wang, Xiong, Mei. *Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection.* NeurIPS 2023. — arXiv:2306.04637
- **[SOTA, theory]** Ahn, Cheng, Daneshmand, Sra. *Transformers learn to implement preconditioned gradient descent for in-context learning.* NeurIPS 2023. — arXiv:2306.00297
- **[SOTA, theory]** Mahankali, Hashimoto, Ma. *One Step of Gradient Descent is Provably the Optimal In-Context Learner with One Layer of Linear Self-Attention.* ICLR 2024. — arXiv:2307.03576
- **[SOTA, empirical]** Fu, Chen, Jia, Sharan. *Transformers Learn to Achieve Second-Order Convergence Rates for In-Context Linear Regression.* NeurIPS 2024. — arXiv:2310.17086
- **[Negative result]** Shen, Mishra, Khashabi. *Do pretrained Transformers Learn In-Context by Gradient Descent?* ICML 2024. — arXiv:2310.08540
- **[Negative result]** Min, Lyu, Holtzman, Artetxe, Lewis, Hajishirzi, Zettlemoyer. *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP 2022. — arXiv:2202.12837
- **[Mechanistic]** Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895
- **[Mechanistic]** Todd, Li, Sharma, Mueller, Wallace, Bau. *Function Vectors in Large Language Models.* ICLR 2024. — arXiv:2310.15213
- **[Dynamics]** Chan, Santoro, Lampinen, Wang, Singh, Richemond, McClelland, Hill. *Data Distributional Properties Drive Emergent In-Context Learning in Transformers.* NeurIPS 2022. — arXiv:2205.05055
- **[Dynamics]** Singh, Chan, Moskovitz, Grant, Saxe, Hill. *The Transient Nature of Emergent In-Context Learning in Transformers.* NeurIPS 2023. — arXiv:2311.08360

## 10. Worked Example

$d=20$, isotropic $\mathcal{D}_x=\mathcal{N}(0,I)$, noiseless linear targets, $n=40$ in-context pairs. Candidates: OLS and ridge$_\lambda$ with $\lambda=0.01$.

With $X\in\mathbb{R}^{40\times20}$ isotropic, the sample eigenvalues of $X^\top X$ concentrate near $n=40$. Ridge shrinks each coordinate by $s^2/(s^2+\lambda)$, so the relative deviation from OLS is about

$$\frac{\|\hat w_{\text{ridge}}-\hat w_{\text{OLS}}\|}{\|\hat w_{\text{OLS}}\|}\approx\frac{\lambda}{n}=\frac{0.01}{40}=2.5\times10^{-4}.$$

With $\mathrm{sd}(y)\approx\sqrt{d}=4.5$, the induced prediction gap is $\Delta_\mathcal{Q}(\text{OLS},\text{ridge}_{0.01})\approx 10^{-3}$ in normalized units.

Now the control: retrain the same architecture with a different seed and compute model-to-model distance. In published replications of this setup, normalized model-vs-OLS error sits around $10^{-2}$, and seed-to-seed spread is of the same order. So $\Delta_\mathcal{Q}=10^{-3}$ sits **an order of magnitude below** $\epsilon_{\text{seed}}\approx10^{-2}$.

The obstruction is now visible as a number: on the canonical isotropic benchmark, no amount of evaluation data can decide between OLS and ridge$_{0.01}$, because the two candidates differ by less than the model's own seed noise. A paper reporting "the transformer implements OLS" on this distribution has reported that its measurement lacks the resolution to say otherwise. Raising $\kappa$ to $10^3$ inflates $\Delta_\mathcal{Q}$ by roughly $\kappa$ into the $10^{-1}$ range — above the noise floor — which is why the experiment in §8 changes the covariate conditioning rather than collecting more prompts.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*