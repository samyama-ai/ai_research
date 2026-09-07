---
id: 34-diffusion-generative/compute-optimal-train-sample-allocation
title: "Compute-Optimal Allocation Between Diffusion Training and Sampling Steps"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Allocation Between Diffusion Training and Sampling Steps

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/compute-optimal-train-sample-allocation` · **Status:** empirically-open

## 1. Problem Statement

A diffusion model spends compute twice: once to fit the score network, and again — per sample, forever — to integrate the reverse process. Chinchilla answered the analogous question for autoregressive LMs on the *training* side only (parameters vs. tokens). For diffusion the budget is three-way: parameters $N$, training tokens/images $D$, and denoiser evaluations per sample $T$ (NFE), the last multiplied by the number of samples ever drawn.

Three variants, different difficulty:

- **Measurement.** Given a fixed total FLOP budget covering training *and* a specified deployment volume, which $(N, D, T)$ minimizes a stated sample-quality functional? Runnable today; the obstruction is that the quality functional (FID and friends) does not cleanly order models across different $T$.
- **Method.** Produce a fitted allocation rule $N^\star(C), D^\star(C), T^\star(C)$ with exponents, validated by held-out extrapolation to a budget $\ge 10\times$ the largest fit point.
- **Theory.** Prove that under a score-approximation error model $\varepsilon_{\text{score}}(N,D)$ and a sampler discretization error $\varepsilon_{\text{disc}}(T)$, the optimal allocation has a specific scaling form — in particular whether $T^\star$ is asymptotically constant, or grows with $C$.

A solution is a rule that, at a budget outside the fitted range, picks an allocation beating the best hand-tuned baseline on a pre-registered metric.

## 2. Formal Setting

Data $x_0 \sim p_{\text{data}}$ on $\mathbb{R}^d$. Forward process $x_t = \alpha_t x_0 + \sigma_t \epsilon$. Score network $s_\theta$, $|\theta| = N$, trained on $D$ examples (or $D$ latent tokens) with the denoising objective
$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \pi, x_0, \epsilon}\big[ w(t)\,\| s_\theta(x_t,t) - \nabla_{x_t}\log p_t(x_t\mid x_0) \|_2^2 \big].$$

**Measured quantities.**

- Training compute $C_{\text{train}} \approx 6 N D$ FLOPs (forward+backward; measure it, do not assume it — attention at $16{,}384$ latent tokens breaks the linear-in-$N$ approximation).
- Sampling compute per sample $C_{\text{samp}} \approx 2 N T \cdot g$, with $g \ge 1$ the classifier-free guidance factor ($g = 2$ when guidance runs conditional and unconditional branches).
- Total: $C(V) = 6ND + 2NTgV$ for deployment volume $V$ samples.
- Score error, the only trainable term: $\varepsilon_{\text{score}}^2 = \mathbb{E}_t\, w(t)\,\mathbb{E}_{x_t}\|s_\theta - \nabla \log p_t\|^2$. Measurable only up to the intractable constant $\mathbb{E}\|\nabla\log p_t\|^2$; in practice one reports validation loss differences, not absolute error.
- Quality $Q$: FID, FD$_{\text{DINOv2}}$, precision/recall, or a likelihood proxy. All are estimator-biased at finite sample count; FID at $50$k reference images has a bias of order $1$ FID unit between $10$k and $50$k generated samples.

**The optimization.**
$$\min_{N,D,T} \; Q\big(\varepsilon_{\text{score}}(N,D),\, \varepsilon_{\text{disc}}(T)\big) \quad \text{s.t.} \quad 6ND + 2NTgV \le C.$$

**Assumptions, and which fail.**
1. *Error separability* — $Q$ decomposes into a training term plus a sampler term. Violated: a better-trained score is smoother and tolerates larger steps, so $\varepsilon_{\text{disc}}$ depends on $\theta$.
2. *$Q$ monotone in $T$.* Violated: with guidance, FID is U-shaped in $T$ for some samplers; more steps can integrate a biased ODE more faithfully and score worse.
3. *Fixed sampler.* Violated: DPM-Solver++, EDM's Heun stepping, and distillation each shift the whole $T$ axis by $5$–$50\times$.
4. *$C_{\text{train}}$ dominates.* Violated for any deployed model: at $V = 10^9$ samples, sampling FLOPs exceed training FLOPs by orders of magnitude, which moves $T^\star$ but is almost never in the plotted objective.

## 3. State of the Art

**Established.** DiT (Peebles & Xie, ICCV 2023) showed FID decreases monotonically with training Gflops across DiT-S/B/L/XL at matched training steps, with DiT-XL/2 (675M) reaching FID $2.27$ on ImageNet $256^2$ at $250$ DDPM steps with guidance — a clean parameter-scaling result, but at *fixed, large* $T$. EDM (Karras et al., NeurIPS 2022) established the sampler-side frontier by ablation: second-order Heun with the $\sigma$-schedule $\sigma_i \propto (\sigma_{\max}^{1/\rho} + \tfrac{i}{n-1}(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}))^\rho$, $\rho = 7$, reaching CIFAR-10 FID $1.79$ at $35$ NFE.

**Closest direct attack.** "Bigger is not Always Better: Scaling Properties of Latent Diffusion Models" (Mei et al., TMLR 2024) trained LDMs from $39$M to $5$B parameters and found that at *matched sampling compute*, smaller models frequently beat larger ones — the crossover is real, and it is the core evidence that the three-way allocation is non-trivial. What that paper does not do: fit an allocation law, or vary $D$ independently of $N$.

**Claimed but unablated.** "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps" (Ma et al., 2025) shows search over noise seeds under a verifier beats adding denoising steps at equal NFE. The result is a benchmark number under specific verifiers (including ones related to the evaluation metric); the verifier-metric independence is not established, so part of the gain may be metric-hacking.

**Theory SOTA (different question).** Chen et al. (ICLR 2023) and Benton et al. (ICLR 2024) bound the steps needed for TV/KL convergence given an $L^2$-accurate score — nearly linear in $d$ — but treat score error as an exogenous constant, so they cannot trade it against $N$ or $D$.

## 4. What Is Known

- **Parameter scaling at fixed $T$ is smooth.** DiT: FID-50K falls $\sim 68 \to 2.27$ across S/2 to XL/2 at 400K–7M steps, ImageNet $256^2$.
- **Text-to-image scaling is component-specific.** Li et al. (CVPR 2024) scaled SD-style UNets and found the cross-attention/transformer blocks carry the gain; naive width scaling is inefficient. Measured at $\sim 0.4$–$1.7$B params, LAION-scale data.
- **Rectified-flow transformers scale predictably.** Esser et al. (ICML 2024, SD3) report validation loss decreasing as a power law from $0.15$B to $8$B params, with loss correlating with human preference — but $T$ is held at $50$ steps throughout.
- **Sampler compute is compressible by $\sim 10\times$ without retraining.** DPM-Solver (Lu et al., NeurIPS 2022): comparable FID at $10$–$20$ NFE where DDIM needs $100$–$250$.
- **Distillation moves the frontier by $\sim 100\times$.** Consistency distillation (Song et al., ICML 2023): CIFAR-10 FID $3.55$ at $1$ NFE, $2.93$ at $2$ NFE, versus $\sim 2$ at $35$ NFE for the teacher — a large quality-per-NFE gain purchased with extra *training* compute, which is exactly the trade the allocation law should price and no published law does.
- **The metric is not neutral.** FID depends on ImageNet class structure in the Inception features (Kynkäänniemi et al., ICLR 2023) and mis-ranks diffusion models relative to human judgment (Stein et al., NeurIPS 2023).

## 5. What Is Not Known

- **Empirically open.** No published IsoFLOP study varies $N$, $D$, and $T$ jointly with $V$ as an explicit parameter. The experiment is runnable at $\le 10^{21}$ FLOPs on ImageNet-scale data; nobody has run it and reported exponents.
- **Empirically open.** Whether $T^\star$ is constant in $C$. Practice suggests $T^\star$ is roughly flat ($20$–$50$ NFE) across four orders of magnitude of model scale, which if true means the diffusion allocation problem collapses to Chinchilla plus a fixed sampler constant. Untested directly.
- **Theoretically open.** No theorem relating $\varepsilon_{\text{score}}(N,D)$ to the achievable $\varepsilon_{\text{disc}}(T)$ — i.e., no proof that better scores admit coarser discretization, despite it being the mechanism everyone assumes.
- **Methodologically blocked.** Comparing allocations requires a $Q$ that is monotone and unbiased across $T$. FID is neither. Without it, "compute-optimal" is defined against a ruler that bends with the variable being optimized.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by guidance**. Classifier-free guidance scale $w$ trades FID against precision/recall, and the FID-optimal $w$ shifts with both $N$ and $T$. So any $(N,D,T)$ comparison is really a comparison over $(N,D,T,w)$, and the optimum over $w$ must be re-searched at every grid point — turning a 3-D IsoFLOP sweep into a 4-D one, roughly $5$–$10\times$ the compute. Second: distillation makes the frontier non-convex. A point at $T=4$ reachable only via a distillation stage costs training compute that the $6ND$ term does not model, so the feasible set is not a simple budget simplex. Third: the deployment volume $V$ that determines whether sampling FLOPs matter is a business parameter, not a scientific one — the "optimal" allocation is only defined relative to a $V$ that papers do not state.

## 7. Current Research (as of 2026)

- **Diffusion scaling laws.** Groups at ByteDance, Google DeepMind, and academic labs have posted fits for diffusion transformers relating loss to $N$ and $D$; most hold $T$ fixed. *(frontier — verify)*
- **Inference-time search.** Xie's group at NYU and collaborators (Ma et al.) on verifier-guided search over the noise space as an alternative axis to $T$.
- **Few-step generators.** Distillation (consistency, adversarial, distribution-matching) at Stability, NVIDIA, MIT; the practical effect is to make $T$ nearly free, which if it holds retires the problem's sampling axis and replaces it with a distillation-compute axis.
- **Metric repair.** FD$_{\text{DINOv2}}$ and human-preference models as replacements for FID; adoption is partial, so cross-paper comparison remains broken. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** DiT/SiT on ImageNet $256^2$ latents. Grid: $N \in \{33\text{M}, 130\text{M}, 458\text{M}, 675\text{M}\}$ (S/B/L/XL), $D$ set to hit IsoFLOP training budgets $C_{\text{train}} \in \{3, 10, 30\} \times 10^{19}$ FLOPs, $T \in \{4, 8, 16, 32, 64, 128, 250\}$ with EDM Heun and DPM-Solver++, guidance $w$ re-optimized per cell over $\{1.0, 1.5, 2.0, 3.0\}$. Total $\approx 4{,}000$ A100-days including guidance search — large but not frontier.

**Control arm.** DiT-XL/2 trained at the largest $C_{\text{train}}$ and sampled at $T=250$, $w=1.5$ — the published default allocation.

**The deciding number.** Report $\arg\min_T$ FD$_{\text{DINOv2}}$ as a function of $C_{\text{train}}$, and fit $T^\star \propto C_{\text{train}}^{\beta}$. The question resolves on $\beta$: if the 95% CI contains $0$, $T^\star$ is scale-free and the three-way problem reduces to Chinchilla with a constant sampler tax. If $\beta$ is significantly nonzero, an allocation law is required and its exponent is now measured. Pre-register FD$_{\text{DINOv2}}$ at $50$k/$50$k samples to avoid the FID monotonicity failure.

## 9. Key References

- **[Foundational]** Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. — arXiv:2006.11239
- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Karras, Aittala, Aila, Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS, 2022. — arXiv:2206.00364
- **[SOTA]** Peebles, Xie. *Scalable Diffusion Models with Transformers.* ICCV, 2023. — arXiv:2212.09748
- **[SOTA]** Mei, Delbracio, Talebi, Tu, Patel, Milanfar. *Bigger is not Always Better: Scaling Properties of Latent Diffusion Models.* TMLR, 2024.
- **[SOTA]** Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML, 2024. — arXiv:2403.03206
- **[SOTA]** Ma, Tong, Jia, Xu, Xie et al. *Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps.* 2025. — arXiv:2501.09732
- **[Theory]** Chen, Chewi, Li, Li, Salim, Zhang. *Sampling is as Easy as Learning the Score.* ICLR, 2023. — arXiv:2209.11215
- **[Theory]** Benton, De Bortoli, Doucet, Deligiannidis. *Nearly $d$-Linear Convergence Bounds for Diffusion Models via Stochastic Localization.* ICLR, 2024.
- **[Method]** Lu, Zhou, Bao, Chen, Li, Zhu. *DPM-Solver.* NeurIPS, 2022. — arXiv:2206.00927
- **[Method]** Song, Dhariwal, Chen, Sutskever. *Consistency Models.* ICML, 2023. — arXiv:2303.01469
- **[Method]** Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR, 2022. — arXiv:2202.00512
- **[Measurement]** Kynkäänniemi, Karras, Aittala, Aila, Lehtinen. *The Role of ImageNet Classes in Fréchet Inception Distance.* ICLR, 2023.
- **[Measurement]** Stein et al. *Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models.* NeurIPS, 2023.
- **[Analogue]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314

## 10. Worked Example

Two candidates at a fixed total budget, ImageNet $256^2$, deployment $V = 10^7$ samples, guidance factor $g=2$.

- **A:** DiT-XL/2, $N = 6.75\times10^8$, trained $7$M steps at batch $256$ → $D \approx 1.8\times10^9$ images. $C_{\text{train}} = 6ND \approx 7.3\times10^{18}$ FLOPs. Sampling at $T = 250$: $C_{\text{samp}} = 2NTgV \approx 2(6.75\times10^8)(250)(2)(10^7) = 6.8\times10^{18}$ FLOPs. Total $\approx 1.4\times10^{19}$.
- **B:** DiT-L/2, $N = 4.58\times10^8$, same $D$ → $C_{\text{train}} \approx 4.9\times10^{18}$. Spend the saved $2.4\times10^{18}$ on more training *and* sample at $T=250$: $C_{\text{samp}} \approx 4.6\times10^{18}$. Total $\approx 9.5\times10^{18}$ — B is $32\%$ cheaper overall.

Sampling is $47\%$ of A's lifetime FLOPs, not a rounding error. Now the obstruction. Published FID-50K: XL/2 $\approx 2.27$, L/2 $\approx 2.91$ (guidance $1.5$, $T=250$). To decide whether B's $32\%$ saving is worth $+0.64$ FID you must compare across a metric that is (i) biased at finite sample count by $\sim 0.1$–$1$ unit, (ii) known to mis-rank diffusion models against human preference, and (iii) re-optimized at a different $w$ for each model — L/2's FID-optimal guidance is not XL/2's. Re-running the comparison at $T=32$ with DPM-Solver++ cuts both sampling terms by $\sim 8\times$, which flips the total-cost ratio to $\approx 1.05$ in A's favor and reverses the decision entirely.

The allocation answer is not stable under a change of sampler or a change of metric. That instability, not the compute, is what makes the problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*