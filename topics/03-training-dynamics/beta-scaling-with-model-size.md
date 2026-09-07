---
id: 03-training-dynamics/beta-scaling-with-model-size
title: "Optimizer Hyperparameter Beta Scaling with Model Size"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimizer Hyperparameter Beta Scaling with Model Size

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/beta-scaling-with-model-size` · **Status:** empirically-open

## 1. Problem Statement

Adam and AdamW carry two exponential-moving-average (EMA) decay rates, $\beta_1$ (first moment) and $\beta_2$ (second moment). Learning rate has a widely used transfer rule across model width ($\mu$P). The betas have none. Practice instead uses fixed folklore constants — $(0.9, 0.999)$ at small scale, $(0.9, 0.95)$ for most large language models — with no derivation of why the constant changes.

**The question.** For a fixed architecture family, fixed token budget rule, and fixed batch size *measured in tokens*, does the loss-optimal $(\beta_1^\star, \beta_2^\star)$ depend on model size $N$ (width $n$, depth $L$)? If yes, with what exponent?

Three variants, different difficulty:

- **Measurement.** Estimate $\beta_2^\star(N)$ with error bars at 2–4 scales, holding batch size and token budget fixed. Runnable today; nobody has published the clean grid.
- **Method.** Give a transfer rule $\beta_2(N) = f(\beta_2^{\text{proxy}}, N)$ that makes a small-model tune land on the large-model optimum, as $\mu$P does for learning rate. Requires the measurement first.
- **Theory.** Derive the width/depth dependence of the optimal second-moment timescale from a model of gradient-noise autocorrelation. Open; existing SDE theory covers batch size only.

**Solved** means: a published exponent $b$ with a confidence interval such that $1-\beta_2^\star \propto N^{b}$, plus a held-out run at a scale outside the fitting range where the predicted $\beta_2$ beats the folklore constant on final loss.

## 2. Formal Setting

Parameters $\theta \in \mathbb{R}^P$. At step $t$, minibatch $B_t$ of $B$ tokens, gradient $g_t = \nabla_\theta \mathcal{L}(\theta_t; B_t)$. AdamW:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2},$$
$$\theta_{t+1} = \theta_t - \eta_t \left( \hat m_t / (\sqrt{\hat v_t} + \epsilon) + \lambda \theta_t \right).$$

**Quantities as measured.**

- **Model size.** Width $n$ = residual stream dimension (measured: `d_model`). Depth $L$ = block count. $N$ = non-embedding parameter count.
- **EMA timescale in steps.** $\tau_2 = 1/(1-\beta_2)$, $\tau_1 = 1/(1-\beta_1)$. Measured directly from the config, not fitted.
- **EMA timescale in tokens.** $T_2 = B\,\tau_2$. This is the scale-free version; any claim about "$\beta_2$ scaling" that does not hold $B$ fixed is a claim about $T_2$, not $\beta_2$.
- **Optimum.** $\beta_2^\star(N) = \arg\min_{\beta_2} \mathbb{E}[\mathcal{L}_{\text{val}}(N, C, \beta_2)]$ at compute $C$, with $\eta$ *re-tuned at each $\beta_2$* (joint, not coordinate-wise — the $(\eta,\beta_2)$ ridge is strongly correlated).
- **Gradient noise.** Per-coordinate signal-to-noise $\rho_i = \mu_i^2 / \sigma_i^2$ where $\mu_i = \mathbb{E}[g_i]$, $\sigma_i^2 = \mathrm{Var}[g_i]$ over minibatches. Measured by splitting a large batch into microbatches and taking the sample moments — expensive and itself batch-size dependent.

**Assumptions, and which are violated.**

1. *Stationary gradient noise over the EMA window.* Violated: gradient norms drop by orders of magnitude across warmup and again at loss spikes.
2. *A unique interior optimum in $\beta_2$.* Violated in the instability regime — the loss surface in $(\eta,\beta_2)$ has a divergence cliff, not a smooth basin (Wortsman et al., 2024).
3. *Separability of $\beta_2$ from $\epsilon$.* Violated: $\epsilon$ interacts with $\sqrt{v_t}$ whenever $v_t \to 0$, and Everett et al. (2024) show $\epsilon$ itself must be scaled down with width.
4. *Batch size held fixed while $N$ varies.* Violated in essentially every published large run — batch size grows with model size, confounding the two.

## 3. State of the Art

**Established (theory).**
- Malladi et al. (NeurIPS 2022) derive from an SDE approximation of Adam/RMSProp that when batch size is scaled $B \to \kappa B$, the correct scaling is $\eta \to \sqrt{\kappa}\,\eta$ (square-root LR rule) and $1-\beta \to \kappa(1-\beta)$ for both betas — i.e. **constant EMA timescale in tokens**. This is a batch-size rule, derived under a fixed model.
- Busbridge et al. (NeurIPS 2023), "How to Scale Your EMA", give the matching rule for a general model EMA: $\beta \to \beta^{\kappa}$, agreeing to first order.
- Défossez et al. (TMLR 2022) prove Adam/Adagrad convergence bounds whose rate degrades as $O\!\big((1-\beta_2)^{-1}\big)$-type factors in the non-convex smooth setting — a bound on $\beta_2$'s effect, with no width dependence anywhere in it.

**Established (empirical).**
- $\mu$P (Yang & Hu, ICML 2021; Yang et al., Tensor Programs V, 2022) transfers learning rate across width. Betas are held *fixed* in $\mu$P transfer, and this is an assumption of the recipe, not a result of it.
- Everett et al. (ICML 2024), "Scaling Exponents Across Parameterizations and Optimizers", sweep parameterizations and per-layer LR exponents up to ~1.2B parameters and show $\epsilon$ must shrink with scale (motivating their Adam-atan2). Betas were not the swept axis.

**Claimed but unablated.**
- The industry drift from $\beta_2 = 0.999$ to $0.95$ for large LMs (GPT-3, Brown et al. 2020; LLaMA, Touvron et al. 2023) is reported as a config line, never ablated at the target scale. No published paper shows $0.95$ beats $0.999$ at 70B with batch size held fixed.
- Marek et al. (2025) argue $\beta_2$ should be set from a token-timescale rule and that small-batch training is far more robust than believed when $\beta_2$ is adjusted; the evidence is at sub-billion scale.

**Benchmark-number-only.** Li et al. (2025), "Predictable Scale / Step Law", fit optimal $\eta$ and batch size as power laws in $N$ and $D$ over ~1M GPU-hours. The betas are held constant throughout; their surfaces are therefore conditional on folklore betas and cannot answer this question.

## 4. What Is Known

- **Batch-size rule holds empirically at small scale.** Malladi et al. validate the $\kappa$-scaling of $1-\beta$ on ResNet-50/ImageNet and BERT-base pretraining across batch $\kappa$ up to $\sim 8$–$16\times$; loss curves overlay within noise.
- **LR–$\beta_2$ coupling is strong.** In the AdamW update, the effective step scales roughly as $\eta$ times a $\beta_2$-dependent noise-attenuation factor. Coordinate-wise tuning of $\beta_2$ at fixed $\eta$ therefore mismeasures the optimum; this is visible in the 2D sweeps of Wortsman et al. (2024) at 20M–4.8B parameters.
- **Low $\beta_2$ suppresses loss spikes.** Wortsman et al. (2024) reproduce attention-logit-growth and output-logit-divergence instabilities in models as small as 20M–100M and show they track the LR at which large models fail; lowering $\beta_2$ (equivalently shortening $\tau_2$) is one of the mitigations practitioners report.
- **Critical batch size grows with data, not mainly with model size.** Zhang et al. (ICLR 2025), "How Does Critical Batch Size Scale in Pre-training?", find CBS scales primarily with data size $D$ at 85M–1.2B. This matters because it says the batch-size confound is *not* a pure function of $N$ — so the two axes are separable in principle.
- **The GPT-2 → GPT-3 shift is roughly, but not exactly, the batch-size rule.** See §10: the token-timescale rule explains a factor of ~6 of an observed ~50× change in $1-\beta_2$.

## 5. What Is Not Known

- **Empirically open (the core gap).** No published experiment varies width over $\ge 8\times$ with batch size *fixed in tokens*, token budget fixed by a Chinchilla-style rule, and $(\eta, \beta_2)$ swept jointly. The experiment costs a few thousand GPU-hours; it is runnable and unrun.
- **Empirically open.** Whether $\beta_1^\star$ moves with depth. Deep-network signal-propagation results (Depth-$\mu$P, Yang et al. 2023) suggest depth-dependent scaling for residual branches; nothing analogous exists for momentum.
- **Theoretically open.** No derivation of a width-dependent optimal $\tau_2$. The SDE analysis of Malladi et al. is exact only in the small-LR limit and treats the model as a black box; it has no $n$ in it.
- **Methodologically blocked.** "Optimal $\beta_2$" is ill-posed under warmup and cosine decay, because $\tau_2 = 1/(1-\beta_2)$ is constant in steps while the gradient distribution is strongly non-stationary. A well-posed target would be a *schedule* $\beta_2(t)$, and no agreed parameterization of that schedule exists.

## 6. Why It Is Hard

**The specific obstruction is non-identifiability in the existing record, compounded by a 3-parameter joint sweep.**

Every large-scale run that changed $\beta_2$ also changed batch size, model width, depth, token budget, and often the parameterization. GPT-3 versus GPT-2 differs on all five. Since a batch-size effect on $1-\beta_2$ is theoretically predicted and a width effect is not, any observed shift is absorbed by the batch-size explanation with no residual anyone is obliged to account for. The record cannot distinguish $b = 0$ from $b = 0.15$.

Second: the decision requires a *joint* $(\eta, \beta_2, \epsilon)$ sweep, because $\epsilon$ is known to need scaling with width (Everett et al., 2024) and $\eta$ is known to trade off against $\beta_2$. A 3D grid at 4 scales with 3 seeds is $\sim 10^2$–$10^3$ runs. At the width where an effect would be detectable ($\ge 2048$), that is real money — and at smaller widths the effect, if it exists, is smaller than seed noise ($\pm 0.005$ nats typical for 100M-parameter runs).

## 7. Current Research (as of 2026)

- **Parameterization groups** (Google DeepMind — Everett, Pennington; Microsoft/xAI — Yang) continue to extend $\mu$P-style transfer to $\epsilon$, weight decay, and per-layer exponents. Betas are still explicitly outside the transferred set.
- **Matrix/norm-based optimizers** (Muon: Jordan et al. 2024; Bernstein & Newhouse, "Old Optimizer, New Norm", 2024; Moonshot's Moonlight, 2025) partly sidestep the question — Muon's Newton–Schulz orthogonalization makes the update scale-invariant, leaving only a single momentum $\beta$. Whether *that* $\beta$ transfers across width is itself untested. *(frontier — verify)*
- **Token-timescale framing** (Marek, Lotfi, Wilson, Goldblum and collaborators, 2025) is the most direct current attack: set $\beta_2$ from a target half-life in tokens rather than steps.
- **Industrial hyperparameter scaling laws** (StepFun's Step Law 2025; DeepSeek LLM 2024) fit $\eta$ and $B$ surfaces at scale but freeze betas; extending their grids by one axis is the cheapest path to an answer for a lab that already owns the infrastructure. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Decoder-only transformers in $\mu$P, depth $L = 24$ fixed, widths $n \in \{256, 512, 1024, 2048\}$ (64× parameter range). Token budget $D = 20N$ (Chinchilla). **Batch size fixed at $B = 2^{18} = 262{,}144$ tokens for all four widths** — this is the whole point; it removes the confound. Fixed WSD schedule, 2% warmup, no gradient clipping (clipping hides $\beta_2$ effects).

**Grid.** $1-\beta_2 \in \{0.001, 0.003, 0.01, 0.03, 0.1\}$ × $\eta \in$ 5 values spanning $8\times$ around the $\mu$P-transferred base LR, $\beta_1 = 0.9$ fixed, $\epsilon = 10^{-15}$ (small enough to be inert). 3 seeds at the argmin cell only. $\approx 100$ runs + 12 seed reruns; the $n=2048$ arm dominates cost at roughly $6\times10^{19}$ FLOPs total.

**Control arm.** The same four widths run at fixed $\beta_2 = 0.95$ with $\eta$ re-tuned. This isolates "does re-tuning $\beta_2$ per width buy anything beyond re-tuning $\eta$".

**The deciding number.** Fit $\log(1-\beta_2^\star) = a + b \log n$ over the four widths, with $\beta_2^\star$ taken from a quadratic fit to the loss along the $\beta_2$ axis at the per-$\beta_2$ optimal $\eta$. **Report $b$ and its 95% CI.**

- $|b| < 0.1$ with CI excluding $0.25$ → betas do not need width scaling; the folklore constant is defensible and the field should stop worrying.
- $b$ significantly positive (predicted sign if wider models have noisier per-coordinate gradients) → publish the exponent and validate at $n = 4096$ held out: the predicted $\beta_2$ must beat $0.95$ by more than $2\times$ the seed standard deviation (target: $>0.01$ nats).

## 9. Key References

- **[Foundational]** Kingma, D. P., Ba, J. *Adam: A Method for Stochastic Optimization.* ICLR, 2015. — arXiv:1412.6980
- **[Foundational]** Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.* ICLR, 2019. — arXiv:1711.05101
- **[Theory SOTA]** Malladi, S., Lyu, K., Panigrahi, A., Arora, S. *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* NeurIPS, 2022. — arXiv:2205.10287
- **[Theory]** Busbridge, D., Ramapuram, J., Ablin, P., Likhomanenko, T., Dhekane, E. G., Suau, X., Webb, R. *How to Scale Your EMA.* NeurIPS, 2023. — arXiv:2307.13813
- **[Theory]** Défossez, A., Bottou, L., Bach, F., Usunier, N. *A Simple Convergence Proof of Adam and Adagrad.* TMLR, 2022. — arXiv:2003.02395
- **[SOTA]** Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., Gao, J. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021 / arXiv 2022. — arXiv:2203.03466
- **[SOTA]** Everett, K., Xiao, L., Wortsman, M., Alemi, A. A., Novak, R., Liu, P. J., Gur, I., Sohl-Dickstein, J., Kaelbling, L. P., Lee, J., Pennington, J. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024. — arXiv:2407.05872
- **[SOTA]** Wortsman, M., Liu, P. J., Xiao, L., Everett, K., Alemi, A., Adlam, B., Co-Reyes, J. D., Gur, I., Kumar, A., Novak, R., Pennington, J., Sohl-Dickstein, J., Xu, K., Lee, J., Gilmer, J., Kornblith, S. *Small-scale Proxies for Large-scale Transformer Training Instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Empirical]** Zhang, H., Morwani, D., Vyas, N., Wu, J., Zou, D., Ghosh, U., Ghosh, M., Brandfonbrener, D., Kakade, S. *How Does Critical Batch Size Scale in Pre-training?* ICLR, 2025. — arXiv:2410.21676
- **[Empirical]** Shallue, C. J., Lee, J., Antognini, J., Sohl-Dickstein, J., Frostig, R., Dahl, G. E. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[Empirical]** Porian, T., Wortsman, M., Jitsev, J., Schmidt, L., Carmon, Y. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* NeurIPS, 2024. — arXiv:2406.19146
- **[Context]** Brown, T. B., et al. *Language Models are Few-Shot Learners.* NeurIPS, 2020 (GPT-3 config: $\beta_2 = 0.95$). — arXiv:2005.14165
- **[Context]** Touvron, H., et al. *LLaMA: Open and Efficient Foundation Language Models.* 2023 ($\beta_1=0.9$, $\beta_2=0.95$). — arXiv:2302.13971
- **[Survey]** Bernstein, J., Newhouse, L. *Old Optimizer, New Norm: An Anthology.* 2024. — arXiv:2409.20325

## 10. Worked Example

**Can the batch-size rule alone explain the GPT-2 → GPT-3 beta change?**

| | GPT-2 (124M) | GPT-3 (175B) |
|---|---|---|
| $\beta_2$ | 0.999 | 0.95 |
| $1-\beta_2$ | $10^{-3}$ | $5\times10^{-2}$ |
| $\tau_2$ (steps) | 1000 | 20 |
| Batch $B$ (tokens) | $\approx 5.2\times10^5$ | $3.2\times10^6$ |
| Width $n$ | 768 | 12288 |

Malladi's rule: $1-\beta_2$ should scale linearly with $B$. Observed batch ratio $\kappa = 3.2\times10^6 / 5.2\times10^5 = 6.2$. Predicted:

$$(1-\beta_2)^{\text{pred}} = 10^{-3} \times 6.2 = 6.2\times10^{-3} \implies \beta_2^{\text{pred}} = 0.9938.$$

Observed: $0.95$. The **residual is a factor of $5\times10^{-2}/6.2\times10^{-3} = 8.1$** in $1-\beta_2$, unexplained by batch size.

Now compare token timescales: $T_2^{\text{GPT-2}} = 1000 \times 5.2\times10^5 = 5.2\times10^8$ tokens; $T_2^{\text{GPT-3}} = 20 \times 3.2\times10^6 = 6.4\times10^7$ tokens. The second-moment memory *shortened by 8×* in token terms while the model got 1400× larger and 16× wider.

**Where the obstruction becomes visible.** That residual factor of 8.1 is exactly the quantity this problem is about — and it is not attributable. Fitting it to width gives $b = \log 8.1 / \log 16 = 0.75$; fitting it to parameter count gives $b = \log 8.1 / \log(1.4\times10^3) = 0.29$; attributing it to token budget $D$ (3.0×10^{11} vs 10^{10}, ratio 30) gives $0.61$. Three different exponents, all fitting the same two points, all consistent with the published record. And a fourth explanation — that GPT-3's $0.95$ was chosen for stability against loss spikes rather than final loss, and is *not* the loss-optimum at all — fits equally well and is what practitioners actually report.

Two data points, four free explanations, zero error bars. This is why the answer requires the fixed-batch width sweep in §8 and cannot be extracted from anything already published.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*