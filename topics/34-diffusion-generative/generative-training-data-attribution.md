---
id: 34-diffusion-generative/generative-training-data-attribution
title: "Exact Attribution of a Generated Sample to Training Examples"
topic: 34-diffusion-generative
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exact Attribution of a Generated Sample to Training Examples

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/generative-training-data-attribution` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a generative model $\theta$ trained on a dataset $D = \{x_1,\dots,x_n\}$ and a sample $\hat{x}$ that the model produced, decide which training examples caused $\hat{x}$ — and by how much.

Three variants, of very different difficulty:

- **Measurement.** Define a ground-truth attribution $\tau^\star(\hat{x}, x_i)$ that is well posed. This is the blocked one. A generated sample is a function of both the weights and the sampling noise; remove $x_i$ from $D$, retrain, and the "same" sample no longer exists, so the counterfactual that attribution is supposed to estimate has no referent.
- **Method.** Given some accepted target, compute per-example scores over $n \sim 10^9$ examples at inference cost far below retraining. Partly solved (TRAK, EK-FAC influence functions); accuracy is modest.
- **Theory.** Prove identifiability: conditions on $D$ and the training map under which $\tau^\star$ is unique. Open, and likely false without strong assumptions — duplicated and near-duplicate images make the credit assignment non-identifiable by construction.

Solving it means: a score $\tau$ computable in $O(n)$ forward-pass-equivalents whose predictions match retraining counterfactuals with correlation near $1$, for a definition of "counterfactual" that survives the fact that generation is stochastic. Nobody currently has the third clause.

## 2. Formal Setting

Let $\mathcal{A}$ be the training algorithm (data order, seed, schedule) and $\theta(S) = \mathcal{A}(S)$ for $S \subseteq D$. Let $g(\theta, z)$ be the deterministic sampler (DDIM, or the probability-flow ODE) applied to initial noise $z \sim \mathcal{N}(0,I)$, so $\hat{x} = g(\theta(D), z)$.

**Target function.** Attribution requires a scalar $f(\hat{x}; \theta)$ to attribute. In practice it is never "the sample"; it is a proxy, most often the simple diffusion loss at the sample,
$$f_{\mathrm{loss}}(\hat x;\theta) \;=\; \sum_{t \in T} \mathbb{E}_{\epsilon}\big\|\epsilon_\theta(\sqrt{\bar\alpha_t}\hat x + \sqrt{1-\bar\alpha_t}\epsilon,\, t) - \epsilon\big\|_2^2 ,$$
measured by Monte Carlo over a fixed grid $T$ of timesteps and a fixed set of $\epsilon$ draws (typically 10–100 per timestep). Alternatives: the ELBO, or $\|g(\theta,z)-\hat x\|$ for the fixed seed $z$ (Journey-TRAK, Georgiev et al. 2023).

**Ground truth.** The counterfactual attribution of $x_i$ is
$$\tau^\star(\hat x, x_i) \;=\; \mathbb{E}_{\mathcal{A}}\big[f(\hat x;\theta(D))\big] - \mathbb{E}_{\mathcal{A}}\big[f(\hat x;\theta(D\setminus\{x_i\}))\big],$$
measured by retraining $m$ models per condition and averaging; the outer expectation is over seeds and data order.

**Evaluation.** The linear datamodeling score (LDS, Ilyas et al. 2022; Park et al. 2023): sample $K$ random subsets $S_1,\dots,S_K$ of size $\alpha n$, retrain $m$ models on each, and report
$$\mathrm{LDS}(\tau) \;=\; \rho_{\text{Spearman}}\Big(\big\{\textstyle\sum_{i \in S_k} \tau(\hat x, x_i)\big\}_{k=1}^K,\ \big\{\mathbb{E}_\mathcal{A} f(\hat x;\theta(S_k))\big\}_{k=1}^K\Big).$$
Measured cost: $Km$ full training runs, typically $K=64$–$128$, $m=5$–$20$, $\alpha=0.5$.

**Assumptions, and which are violated.**
1. *Additivity* — subset effect is the sum of per-example effects. Violated: near-duplicates substitute for each other, so removing one changes nothing while removing all changes a lot.
2. *Low seed variance* — $\mathrm{Var}_\mathcal{A}[f]$ is small next to the leave-one-out effect. Violated at $n \gtrsim 10^5$, where the LOO signal falls below run-to-run noise and $m$ must grow as the square of the inverse effect size.
3. *The proxy tracks the sample* — $f_{\mathrm{loss}}$ ranks data the way perceptual identity of $\hat x$ would. Not established; see §4.
4. *Convexity / unique optimum*, inherited from influence-function derivations. Violated for all deep nets (Basu et al. 2021; Bae et al. 2022).

## 3. State of the Art

**Method SOTA (established).** TRAK (Park, Georgiev, Ilyas, Leclerc, Madry, ICML 2023) — random-projected empirical-NTK gradient features, averaged over an ensemble of independently trained models; the standard scalable estimator. EK-FAC influence functions scaled to 52B-parameter LLMs (Grosse et al. 2023) and to diffusion models with K-FAC approximations (Mlodozeniec et al., ICLR 2025). D-TRAK (Zheng, Pang, Du, Jiang, Lin, ICLR 2024) reports that replacing the theoretically motivated loss gradient with a *different* function (e.g. $\|\epsilon_\theta\|^2$) improves LDS — a result that is reproducible but explicitly undermines the theory the method is derived from.

**Measurement SOTA (claimed, weakly ablated).** "Evaluating Data Attribution for Text-to-Image Models" (Wang, Efros, Zhu, Zhang, ICCV 2023) builds synthetic ground truth by fine-tuning a model on a single known exemplar, then asking whether attribution recovers it. Established as a retrieval benchmark; unablated as a proxy for attribution in *pretraining*, where no single example dominates.

**Benchmark-number-only results.** Most reported diffusion-attribution numbers are LDS on CIFAR-2/CIFAR-10-scale DDPMs with $n \le 5{,}000$. Nothing at LAION scale has a retraining-based ground truth of any kind; text-to-image "attribution" results there are retrieval scores against human or CLIP similarity judgments, which is a different quantity.

## 4. What Is Known

- **Memorization is real but rare.** Carlini et al. (USENIX Security 2023) recovered on the order of $10^2$ near-verbatim training images from Stable Diffusion by generating 500 samples for each of the 350,000 most-duplicated captions ($1.75\times10^8$ samples). Duplication in the training set was the dominant predictor. Somepalli et al. (CVPR 2023) found replication rates of a few percent on LAION-scale models under a copy-detection retrieval metric.
- **Influence estimates are fragile.** Basu, Pope, Feizi (ICLR 2021): influence-function estimates on deep nets vary strongly with depth, width, weight decay and damping; correlation with LOO retraining is often near zero. Bae et al. (NeurIPS 2022): influence functions approximate the *proximal Bregman response function*, not leave-one-out retraining — the target was misnamed, not just mis-estimated.
- **LDS values are low in absolute terms.** On CIFAR-scale DDPMs ($n=5{,}000$, $K$ in the dozens), reported LDS for gradient-based attribution sits in roughly the $0.2$–$0.4$ Spearman range, with D-TRAK's ablated variant at the top of that band. A score of $0.3$ means the method explains a minority of the subset-to-subset variation it is meant to predict.
- **Ensembling is required, not optional.** TRAK's gains come substantially from averaging over independently trained models; single-model attribution is markedly worse (ICML 2023 ablations).
- **Long-tail structure.** Feldman & Zhang (NeurIPS 2020) established at ImageNet scale that memorization concentrates on rare examples — consistent with attribution being sharp for singletons and degenerate for duplicated data.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted ground truth for attributing a *generated image* as opposed to a *held-out image's loss*. Retraining changes which images the model generates, so $\tau^\star$ as written in §2 is defined only on a fixed external $\hat x$, not on the generative act. Whether $f_{\mathrm{loss}}$-attribution and perceptual-origin attribution agree has not been measured, at any scale. This is the load-bearing gap.
- **Theoretically open.** No identifiability theorem: given a set of $k$ exact duplicates, every convex combination of credit assignments is consistent with all counterfactuals, and no proof exists that a canonical choice (uniform, Shapley) is the right one for copyright or scientific purposes. Also open: whether any $O(n)$ estimator can achieve LDS $\to 1$ for non-convex training.
- **Empirically open.** Whether LDS $\approx 0.3$ at $n=5{,}000$ improves or collapses at $n=10^6$ is runnable (a few thousand GPU-hours on ImageNet-scale latent diffusion) and unrun.

## 6. Why It Is Hard

Three named obstructions, in order of severity.

1. **Absent ground truth for the actual object.** The quantity everyone wants — "this image came from those training photos" — has no counterfactual definition, because the sample is not preserved under retraining. Every benchmark silently substitutes a loss on a fixed image. The evaluation does not measure the thing it names.
2. **Non-identifiability under duplication.** For $k$ identical or near-identical training images, all LOO effects are $\approx 0$ while the group effect is large. Attribution scores are then determined by the estimator's inductive bias, not by the data, and no experiment can distinguish among the choices.
3. **Compute.** Ground truth costs $Km$ retrainings. At Stable-Diffusion scale (order $10^5$ A100-hours per run), $K m = 10^3$ implies $\sim 10^8$ A100-hours — roughly $10^8$–$10^9$ USD at commodity rates. The measurement is not merely expensive; it is out of reach by four orders of magnitude.

Seed variance compounds all three: as $n$ grows, per-example effect shrinks as $\approx 1/n$ while run-to-run noise does not, so $m$ must grow to keep signal-to-noise fixed.

## 7. Current Research (as of 2026)

- **Scalable curvature.** K-FAC/EK-FAC influence for diffusion (Mlodozeniec, Eschenhagen, Bae, Immer, Krueger, Turner — Cambridge/Vector, ICLR 2025); unrolled-differentiation estimators (Bae et al., 2024).
- **Loss-proxy design.** D-TRAK's finding that non-gradient-theoretic target functions win has turned "which $f$?" into an open sub-problem (Sea AI Lab / Pang, Du, Lin).
- **Copyright-facing metrics.** Near-access-free / differential-privacy-style guarantees as an alternative to per-sample attribution — attribution replaced by a bound on influence. *(frontier — verify current results.)*
- **Group and Shapley-style attribution** to sidestep duplication non-identifiability. *(frontier — verify.)*
- **Provenance by construction:** watermarking generations with the retrieved/conditioning shards, sidestepping post-hoc attribution entirely. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does loss-proxy attribution agree with perceptual-origin attribution for generated samples?

**Scale.** Latent diffusion on ImageNet-64, $n = 128{,}000$ (100 examples/class × 1,280 classes), 30M-parameter UNet, $\sim$4 A100-hours per training run. Train $K=128$ models on random 50% subsets, $m=4$ seeds each: $512$ runs $\approx 2{,}000$ A100-hours ($\approx$ \$4k).

**Procedure.** Fix 200 noise seeds. For each seed, generate under each of the 512 models. For each generated image $\hat x$ from the full-data model, compute (a) TRAK/D-TRAK scores against $f_{\mathrm{loss}}$; (b) the *perceptual* counterfactual: fraction of subsets $S_k$ containing $x_i$ under which the seed-$z$ generation lands in the same DINOv2 cluster as $\hat x$.

**Control arm.** Identical pipeline with $\hat x$ replaced by a *held-out real* image. This isolates whether the difficulty is attribution or generativity.

**Deciding number.** Spearman correlation between the loss-proxy ranking and the perceptual-counterfactual ranking, over the top-100 attributed examples per sample. If $\rho > 0.7$ the field's proxy is validated and the problem downgrades to empirically open. If $\rho < 0.3$ — while the held-out control reproduces the published LDS $\approx 0.3$ — then every published diffusion attribution number is measuring a quantity unrelated to sample provenance, and the problem is confirmed methodologically blocked.

## 9. Key References

- **[Foundational]** Pang Wei Koh, Percy Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017. — arXiv:1703.04730
- **[Foundational]** Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry. *Datamodels: Predicting Predictions from Training Data.* ICML 2022. — arXiv:2202.00622
- **[SOTA]** Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry. *TRAK: Attributing Model Behavior at Scale.* ICML 2023. — arXiv:2303.14186
- **[SOTA]** Xiaosen Zheng, Tianyu Pang, Chao Du, Jing Jiang, Min Lin. *Intriguing Properties of Data Attribution on Diffusion Models.* ICLR 2024. — arXiv:2311.00500
- **[SOTA]** Bruno Mlodozeniec, Runa Eschenhagen, Juhan Bae, Alexander Immer, David Krueger, Richard Turner. *Influence Functions for Scalable Data Attribution in Diffusion Models.* ICLR 2025. — arXiv:2410.13850
- **[SOTA]** Kristian Georgiev, Joshua Vendrow, Hadi Salman, Sung Min Park, Aleksander Madry. *The Journey, Not the Destination: How Data Guides Diffusion Models.* 2023. — arXiv:2312.06205
- **[Evaluation]** Sheng-Yu Wang, Alexei A. Efros, Jun-Yan Zhu, Richard Zhang. *Evaluating Data Attribution for Text-to-Image Models.* ICCV 2023.
- **[Critique]** Samyadeep Basu, Phil Pope, Soheil Feizi. *Influence Functions in Deep Learning Are Fragile.* ICLR 2021.
- **[Critique]** Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, Roger Grosse. *If Influence Functions are the Answer, Then What is the Question?* NeurIPS 2022.
- **[Context]** Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, Eric Wallace. *Extracting Training Data from Diffusion Models.* USENIX Security 2023. — arXiv:2301.13188
- **[Context]** Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein. *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models.* CVPR 2023.
- **[Context]** Vitaly Feldman, Chiyuan Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS 2020.
- **[Scale]** Roger Grosse et al. *Studying Large Language Model Generalization with Influence Functions.* Anthropic, 2023. — arXiv:2308.03296

## 10. Worked Example

**Setup.** DDPM on CIFAR-2 (classes *car* and *horse*, $n = 5{,}000$), the standard diffusion-attribution testbed. One training run $\approx 1$ A100-hour. Ground truth by LDS with $K = 64$, $m = 10$: $640$ A100-hours $\approx$ \$1,300.

**Fix a seed** $z_0$ and generate $\hat x_0$ from the full-data model — a red car on grass. TRAK against $f_{\mathrm{loss}}(\hat x_0)$ returns a top-10 list; suppose 7 of the 10 are red cars.

**Now the obstruction.** Take the 64 half-subset models and generate from the *same* $z_0$. The images are not perturbations of $\hat x_0$; they are different cars, and in a minority of cases different classes. Measure it: mean LPIPS between $\hat x_0$ and the 64 subset generations lands near the LPIPS between two unrelated CIFAR images ($\approx 0.6$–$0.7$), not near the $\approx 0.1$ that would indicate a stable sample identity. The counterfactual "what happens to $\hat x_0$ if we drop $x_i$" therefore has no image-space answer.

**What the benchmark does instead.** It holds $\hat x_0$ fixed as an external image and asks how $f_{\mathrm{loss}}(\hat x_0;\theta(S_k))$ moves. That is a *density* question — how much probability mass the retrained model puts near $\hat x_0$ — not a *provenance* question. Reported LDS $\approx 0.3$ answers the density question, moderately.

**The arithmetic that closes it.** Suppose CIFAR-2 contains 5 near-duplicate red cars. Removing any one shifts $f_{\mathrm{loss}}(\hat x_0)$ by roughly $\Delta/5$ where $\Delta$ is the all-five effect; with $m = 10$ seeds the standard error on the LOO estimate is $\sigma/\sqrt{10}$. Empirically $\sigma$ is comparable to $\Delta$ itself at this scale, so the per-duplicate signal-to-noise is about $0.2/\sqrt{10} \approx 0.06$. To resolve one duplicate's contribution at $3\sigma$ you need $m \approx 2{,}500$ seeds per subset — $1.6\times10^5$ A100-hours on a 5,000-image dataset. Scale the same requirement to LAION-2B and the cost exceeds $10^8$ A100-hours before the identifiability problem is even addressed. The obstruction is not that attribution is inaccurate; it is that the quantity it estimates is both unmeasurable at scale and not the quantity anyone asked about.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*