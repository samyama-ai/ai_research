---
id: 23-privacy-memorization/gradient-inversion-large-batches
title: "Gradient Inversion at Realistic Federated Batch Sizes"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Gradient Inversion at Realistic Federated Batch Sizes

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/gradient-inversion-large-batches` · **Status:** partially-solved

## 1. Problem Statement

A federated client holds a private batch and sends the server an update. **Input:** the update $g$, the model architecture and weights $\theta$, and any public prior the attacker cares to use. **Output:** the client's raw examples. **Decision predicate:** does there exist an attack that recovers a non-trivial fraction of individual examples when the update is the *realistic* one — a FedAvg update aggregating tens to hundreds of examples over multiple local steps — against an **honest-but-curious** server that cannot modify $\theta$?

Three variants, routinely conflated:

- **Method.** Build an attack that works at batch size $B \gtrsim 64$ on high-resolution data. Largely *solved* if the server is malicious (it can plant parameters that make the update near-linear in one example); *open* if the server is honest-but-curious.
- **Measurement.** Decide when a reconstruction counts as a privacy breach. Current practice reports mean PSNR/LPIPS over the batch, which credits the attack for whatever the image prior already knew. There is no accepted prior-matched control.
- **Theory.** Characterise when the map $\text{batch} \mapsto g$ is injective on the data manifold, and give a bound on recoverable bits as a function of $B$, parameter count $d$, local steps $E$, and DP noise $\sigma$. Open beyond the single-layer case.

Solving it means: a stated $B^*(\text{arch}, \text{resolution}, E)$ above which honest-but-curious inversion provably or reliably fails, measured against a prior-only control.

## 2. Formal Setting

Client data $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^{B}$, $x_i \in \mathbb{R}^{n}$ ($n = 3\times224\times224 = 150{,}528$ for ImageNet). Model $f_\theta$, $\theta \in \mathbb{R}^d$, loss $\ell$. The single-step (FedSGD) update is

$$g = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell(f_\theta(x_i), y_i) \in \mathbb{R}^d .$$

The FedAvg update after $E$ epochs of local SGD with step $\eta$ and $T$ minibatch steps is $\Delta\theta = \theta_T - \theta_0$, which is *not* a sum of per-example gradients at a fixed $\theta$ — the composition of $T$ nonlinear steps is what makes the realistic case hard.

**Attack.** Minimise a matching objective plus a prior:

$$\hat{\mathcal{B}} = \arg\min_{\{\tilde x_i\},\{\tilde y_i\}} \; \mathcal{D}\!\left(\nabla_\theta \tfrac{1}{B}\textstyle\sum_i \ell(f_\theta(\tilde x_i), \tilde y_i),\; g\right) + \lambda \mathcal{R}(\{\tilde x_i\}),$$

with $\mathcal{D}$ cosine distance (Geiping et al. 2020) or $\ell_2$ (Zhu et al. 2019), and $\mathcal{R}$ total variation, BN-statistics matching, or a GAN/diffusion prior.

**Measured quantities.**
- *Reconstruction quality:* per-image PSNR $=10\log_{10}(1/\mathrm{MSE})$ and LPIPS, after solving the optimal assignment $\pi \in S_B$ between $\hat{\mathcal{B}}$ and $\mathcal{B}$ (Hungarian matching on LPIPS). Order is unidentifiable, so an assignment step is mandatory and is itself an oracle the real attacker lacks.
- *Attack success rate (ASR):* fraction of $i$ with $\mathrm{LPIPS}(\hat x_{\pi(i)}, x_i) < \tau$. $\tau = 0.3$ is common; the threshold is a convention, not a calibrated breach criterion.
- *Prior-matched gain:* $\Delta = \mathrm{LPIPS}(\text{prior-only best match}) - \mathrm{LPIPS}(\text{gradient-conditioned})$. Rarely reported.

**Assumptions, and which are violated.**
1. *Batch labels known or recoverable.* Recoverable exactly when labels are unique in the batch (Yin et al. 2021); violated at $B > $ #classes and with label smoothing/mixup.
2. *BatchNorm statistics available.* Assumed by most high-$B$ image results; violated in practice (FedAvg clients send weights, not running stats; GroupNorm is the standard fix).
3. *Single local step.* Violated: production FedAvg uses $E \ge 1$ epoch, $T$ in the tens.
4. *Untrained or early-training $\theta$.* Attacks degrade sharply on converged models; violated whenever the round is late.
5. *No secure aggregation, no DP noise, no update compression.* All three are deployed in real systems (Bonawitz et al. 2017; McMahan et al. 2018).

## 3. State of the Art

**Established (honest-but-curious, image).** Geiping et al. (NeurIPS 2020) recover single ImageNet images at high fidelity from a trained ResNet-18 and show degradation with $B$; at $B = 100$ on CIFAR-100 with a trained network, reconstructions are recognisable only in aggregate, not per-image. Yin et al., *GradInversion* (CVPR 2021), reach $B = 48$ on ImageNet with ResNet-50 — but with BN statistics and a group-consistency regulariser over 32 seeds; it is the strongest honest-but-curious image result and it consumes hours of GPU per batch.

**Established (malicious server).** *Robbing the Fed* (Fowl et al., ICLR 2022) prepends a linear "imprint" module and recovers a large majority of individual images from batches of hundreds — exactly, not approximately. *Fishing* (Wen et al., ICML 2022) rescales class-specific parameters so one example dominates the aggregate, recovering targets from batches up to $B=256$ and through secure aggregation over many users. *Decepticons* (Fowl et al., ICLR 2023) and Gupta et al. (NeurIPS 2022) do the analogue for text. These attacks change the threat model: they answer "can a malicious server steal data?" (yes) not "does an aggregate gradient leak?".

**Claimed but unablated.** Diffusion- and GAN-prior attacks report improved LPIPS at $B \ge 32$, but almost none report a **prior-only control** — a sample from the same prior conditioned only on the label, with no gradient term. Without it, the metric cannot separate "the gradient leaked this face" from "the prior draws faces". Treat all high-$B$ generative-prior numbers as benchmark numbers, not evidence of leakage.

**Text.** LAMP (Balunović et al., NeurIPS 2022) recovers ~50–80% token overlap on short sequences from BERT-base at $B \le 4$ with a language-model prior; performance falls off fast with sequence length and batch size. TAG (Deng et al., EMNLP Findings 2021) is the earlier baseline.

**Defence side.** Huang et al. (NeurIPS 2021) is the reference ablation: many published defences (Soteria, PRECODE-style perturbations, pruning) lose most of their claimed protection under adaptive attacks, while modest DP noise and larger $B$ do the actual work. Yue et al. (USENIX Security 2023) reach the same conclusion for gradient obfuscation broadly.

## 4. What Is Known

- **Single-layer identifiability.** For a fully-connected layer with bias, an input is recoverable in closed form from the gradient ratio $\partial \ell/\partial W_j \,/\, \partial \ell/\partial b_j$ whenever exactly one example activates neuron $j$ (Geiping et al. 2020; R-GAP, Zhu & Blaschko, ICLR 2021). This is the mechanism behind essentially every large-batch attack.
- **Monotone degradation in $B$.** Across CIFAR-10/100 and ImageNet, mean PSNR falls roughly monotonically with $B$; Huang et al. (2021) report attacks becoming unusable on ImageNet-scale images at $B = 32$ under realistic settings (no BN stats, trained model).
- **Untrained ≫ trained.** Reconstruction quality is far higher at initialisation than after convergence — measured at CIFAR scale, ResNet-18.
- **Local steps hurt the attacker.** Dimitrov et al. (TMLR 2022) formalise FedAvg leakage and show recovery quality drops as $T$ grows; their attack still works but at markedly reduced fidelity relative to FedSGD.
- **Malicious modification collapses the batch-size barrier.** $B = 512$ with near-exact recovery (Fowl et al., ICLR 2022) at ImageNet resolution.
- **Aggregation is not a barrier by itself.** Cocktail Party Attack (Kariyappa et al., ICML 2023) recovers images from aggregated fully-connected gradients via ICA, at CIFAR/Tiny-ImageNet scale.

## 5. What Is Not Known

- **Methodologically blocked.** Whether reported high-$B$ reconstructions carry information from the gradient at all. No standard prior-only control arm exists, and the Hungarian matching step gives the attacker an assignment oracle. Until $\Delta$ (§2) is reported, "ASR at $B=128$" is uninterpretable.
- **Theoretically open.** No bound of the form "at most $k(B, d, E, \sigma)$ bits of the batch are recoverable from $\Delta\theta$" for deep networks. The single-layer result does not compose. Non-identifiability at $B > $ layer width is conjectured, not proved.
- **Empirically open.** The honest-but-curious sweep over $(B, E, \text{norm layer}, \text{training stage})$ at ImageNet resolution with a fixed compute budget per batch has not been run at scale. Nobody has published $B^*$ for GroupNorm ResNet-50 under FedAvg with $E = 1$.
- **Open.** Whether text inversion degrades with $B$ on the same curve as images, given that token embeddings are much lower-entropy targets.

## 6. Why It Is Hard

The central obstruction is **confounded measurement compounded by non-identifiability**.

- *Non-identifiability:* $g$ has $d$ coordinates but the attacker must fit $Bn$ unknowns plus permutation. For ResNet-18 ($d \approx 11.7$M) and ImageNet at $B = 128$, $Bn \approx 19.3$M $> d$. The system is underdetermined; the prior supplies the missing constraints — so the prior, not the gradient, determines much of the output.
- *Confounded metric:* PSNR/LPIPS against ground truth cannot distinguish gradient-derived detail from prior-derived detail, and Hungarian matching hands the attacker information a deployed attacker never has.
- *Compute:* GradInversion-style attacks use multiple restarts × thousands of optimisation steps per batch; a full $(B, E, \text{norm}, \text{stage})$ grid at ImageNet resolution is thousands of GPU-hours, which is why the grid does not exist.
- *Moving threat model:* every negative result at high $B$ is answered with "but a malicious server can", which is true and answers a different question.

## 7. Current Research (as of 2026)

- **Diffusion-prior inversion** at $B \ge 32$ — strong reported numbers, weak controls. *(frontier — verify)*
- **Malicious-parameter attacks against secure aggregation**, extending Fishing/Robbing-the-Fed to user-level disaggregation (Maryland/Goldstein group, Vector Institute/Boenisch–Papernot line).
- **Certified/analytic leakage bounds** for linear and attention layers (SRI Lab, ETH Zürich — Vechev group, continuing the LAMP/FedAvg-leakage line).
- **LLM federated fine-tuning**: whether LoRA updates leak more than full updates, since the low-rank factor is small relative to the sequence. Early work exists; the batch-size curve is not established. *(frontier — verify)*
- **Auditing framing**: treating gradient inversion as a lower bound for empirical DP auditing rather than as an attack in itself.

## 8. Concrete Next Experiment

**Question:** at what $B$ does an honest-but-curious attack stop extracting *any* information beyond its prior?

- **Scale.** ResNet-50 with **GroupNorm** (no BN statistics leak), ImageNet $224\times224$, $B \in \{1, 8, 32, 128, 512\}$, two training stages (random init; 90%-converged), two update types (FedSGD; FedAvg with $E=1$, $T=20$, $\eta=0.01$). 20 batches per cell, fixed budget of 4 GPU-hours per batch (A100). ~40 cells → ~3,200 GPU-hours.
- **Attack arm.** Cosine-matching (Geiping) with a class-conditional diffusion prior, 8 restarts, group consensus.
- **Control arm (the point of the experiment).** Identical prior, identical restarts, identical budget, **gradient term deleted** ($\lambda_{\text{match}} = 0$) — samples conditioned only on the recovered labels. Same Hungarian matching against ground truth.
- **Deciding number.** The prior-matched gain
$$\Delta(B) = \mathbb{E}\big[\mathrm{LPIPS}_{\text{control}}\big] - \mathbb{E}\big[\mathrm{LPIPS}_{\text{attack}}\big].$$
Report $B^* = \min\{B : \Delta(B) < 0.05 \text{ with } 95\% \text{ CI excluding } 0.05\}$.
- **Interpretation.** If $B^* \le 32$ for GroupNorm + FedAvg, the honest-but-curious threat is bounded by batch size alone and defence effort should move entirely to malicious-server detection. If $\Delta(128) > 0.1$, the field's degradation claims are wrong and per-example DP is required even at large $B$.

## 9. Key References

- **[Foundational]** Zhu, Liu, Han. *Deep Leakage from Gradients.* NeurIPS 2019. — arXiv:1906.08935
- **[Foundational]** Geiping, Bauermeister, Dröge, Moeller. *Inverting Gradients — How easy is it to break privacy in federated learning?* NeurIPS 2020. — arXiv:2003.14053
- **[SOTA, honest-but-curious]** Yin, Mallya, Vahdat, Alvarez, Kautz, Molchanov. *See through Gradients: Image Batch Recovery via GradInversion.* CVPR 2021. — arXiv:2104.07586
- **[SOTA, malicious]** Fowl, Geiping, Czaja, Goldblum, Goldstein. *Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models.* ICLR 2022. — arXiv:2110.13057
- **[SOTA, malicious]** Wen, Geiping, Fowl, Goldblum, Goldstein. *Fishing for User Data in Large-Batch Federated Learning via Gradient Magnification.* ICML 2022. — arXiv:2202.00580
- **[Ablation]** Huang, Gupta, Song, Li, Arora. *Evaluating Gradient Inversion Attacks and Defenses in Federated Learning.* NeurIPS 2021. — arXiv:2112.00059
- **[Theory]** Zhu, Blaschko. *R-GAP: Recursive Gradient Attack on Privacy.* ICLR 2021. — arXiv:2010.07733
- **[FedAvg]** Dimitrov, Balunović, Konstantinov, Vechev. *Data Leakage in Federated Averaging.* TMLR 2022. — arXiv:2206.12395
- **[Text]** Balunović, Dimitrov, Jovanović, Vechev. *LAMP: Extracting Text from Gradients with Language Model Priors.* NeurIPS 2022. — arXiv:2202.08827
- **[Aggregation]** Kariyappa, Guo, Maeng, Xiong, Suh, Qureshi, Lee. *Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning using Independent Component Analysis.* ICML 2023.
- **[Defence audit]** Yue, Klabjan, et al. *Gradient Obfuscation Gives a False Sense of Security in Federated Learning.* USENIX Security 2023.
- **[Survey]** Kairouz, McMahan, et al. *Advances and Open Problems in Federated Learning.* Foundations and Trends in ML, 2021. — arXiv:1912.04977

## 10. Worked Example

ResNet-18, ImageNet, $B = 64$, GroupNorm, converged checkpoint, FedSGD.

**Counting.** Unknowns: $64 \times 150{,}528 = 9{,}633{,}792$ pixel values plus a permutation over 64 items ($\log_2 64! \approx 296$ bits). Constraints: $d = 11{,}689{,}512$ gradient coordinates, in float32 — but the update is dominated by the final layers, and after gradient clipping and typical 8-bit quantisation the *effective* information in $g$ is far below $32d$ bits. Nominally $d > Bn$, so the system looks solvable; in practice the informative coordinates are the ~513k parameters of the last block, giving roughly 0.05 informative coordinates per unknown pixel.

**Where the pixels come from.** Run the attack, get mean LPIPS $= 0.42$ over the batch after Hungarian matching. That number looks like a breach. Now run the control: same diffusion prior, same 8 restarts, gradient term removed, conditioned only on the 64 recovered labels, then Hungarian-matched. It returns mean LPIPS $= 0.47$. The gradient bought $\Delta = 0.05$ — a fifth of a JND on LPIPS, and less than the seed-to-seed spread of the attack itself ($\pm 0.03$).

**The obstruction, visible.** The reconstruction that "recovers" a face is mostly the prior drawing a face of the right class. The reported metric credits the attack for it. Contrast the malicious-server arm on the same batch: with a Robbing-the-Fed imprint module, the recovered images are exact for ~50 of the 64 examples — LPIPS near 0, $\Delta$ near 0.47, and no prior needed. The gap between $\Delta = 0.05$ and $\Delta = 0.47$ is the whole content of the problem, and current papers report the same headline metric for both.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*