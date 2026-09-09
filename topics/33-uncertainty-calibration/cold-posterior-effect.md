---
id: 33-uncertainty-calibration/cold-posterior-effect
title: "The Cold Posterior Effect Explanation"
topic: 33-uncertainty-calibration
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# The Cold Posterior Effect Explanation

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/cold-posterior-effect` · **Status:** partially-solved

## 1. Problem Statement

Bayesian neural networks (BNNs) should predict best when averaging over the exact posterior $p(\theta \mid \mathcal{D})$. Wenzel et al. (ICML 2020) reported the opposite: raising the posterior to a power $1/T$ with $T < 1$ — "cooling" it — improves test accuracy and negative log-likelihood (NLL) on standard image and text benchmarks. Averaging over the *correct* posterior is worse than averaging over a sharpened, non-Bayesian one.

Three distinct variants, routinely conflated:

- **Measurement.** Given a dataset, architecture and inference algorithm, estimate $T^\star = \arg\min_T \mathcal{L}_{\text{test}}(T)$ and a confidence interval on it. Solving this means separating true tempering from MCMC bias, prior scale, and augmentation.
- **Method.** Produce a model where the untempered posterior ($T=1$) is at least as good as any tempered one, without hand-tuning $T$. This means fixing the misspecification, not compensating for it.
- **Theory.** Prove, for a stated likelihood/prior/data-generating triple, that $T^\star < 1$ (or $=1$), and identify which modelling assumption forces it.

## 2. Formal Setting

Data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, parameters $\theta \in \mathbb{R}^d$, likelihood $p(y \mid x, \theta)$, prior $p(\theta)$. Define the energy

$$U(\theta) = -\sum_{i=1}^{n} \log p(y_i \mid x_i, \theta) - \log p(\theta).$$

**Fully tempered posterior:** $p_T(\theta \mid \mathcal{D}) \propto \exp(-U(\theta)/T)$.
**Likelihood-tempered ("cold likelihood") posterior:** $q_T(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)^{1/T} p(\theta)$.

These are not the same object, and papers differ in which they use. Under a Gaussian prior $p(\theta) = \mathcal{N}(0, \alpha^2 I)$,

$$\exp(-U(\theta)/T) \;\propto\; p(\mathcal{D}\mid\theta)^{1/T}\,\mathcal{N}(\theta; 0, \alpha^2 T\, I),$$

so **full tempering at temperature $T$ is exactly likelihood tempering with the prior variance rescaled to $\alpha^2 T$.** $T$ and prior scale are non-identifiable in the fully tempered family. Any claim that "the CPE is a prior problem" versus "a temperature problem" must fix a parameterisation first.

**Measured quantity.** With $S$ posterior samples $\theta_1,\dots,\theta_S \sim p_T$, the Bayesian model average predicts $\hat p(y\mid x) = \frac1S\sum_s p(y\mid x,\theta_s)$, and

$$\mathcal{L}_{\text{test}}(T) = -\frac{1}{m}\sum_{j=1}^{m} \log \hat p(y_j \mid x_j), \qquad T^\star = \arg\min_{T} \mathcal{L}_{\text{test}}(T).$$

The *cold posterior effect* is the event $T^\star < 1$ by more than sampling error. The effect size is $\Delta = \mathcal{L}_{\text{test}}(1) - \mathcal{L}_{\text{test}}(T^\star)$ in nats/example.

**Assumptions, and which are violated.**
1. *i.i.d. sampling from the target population.* Violated: CIFAR-10 and ImageNet labels are curated — ambiguous images were filtered or resolved by annotator majority (Aitchison, ICLR 2021).
2. *Categorical likelihood with no aleatoric noise* — one hard label per image. Violated: CIFAR-10H human soft labels show real per-image label entropy (Peterson et al., ICCV 2019).
3. *The training objective is a log-likelihood.* Violated by data augmentation: $\sum_i \frac{1}{K}\sum_{k}\log p(y_i \mid t_k(x_i),\theta)$ is not the log-density of any generative model over $\mathcal{D}$.
4. *Samples come from $p_T$.* Violated: SG-MCMC with minibatch gradient noise and a finite step size targets a biased stationary distribution.

## 3. State of the Art

**Established.** Data augmentation is a *sufficient* cause of a large CPE. Removing augmentation shrinks or eliminates it across several independent studies (Izmailov et al., ICML 2021; Fortuin et al., ICLR 2022; Noci et al., NeurIPS 2021). Bachmann et al. (ICML 2022) give the mechanism: an augmented objective over $K$ transformations behaves like a likelihood evaluated on an inflated effective sample size, and tempering by roughly $T \approx 1/K$ restores a consistent scale.

**Established (negative).** Augmentation is *not necessary*. Nabarro et al. (UAI 2022) build a principled augmented likelihood over a finite augmentation orbit — a genuine density — and still observe a CPE. Noci et al. (NeurIPS 2021) report CPE in augmentation-free settings when the prior is misspecified.

**Established (algorithmic).** Wenzel et al. ran extensive diagnostics ruling out minibatch-noise bias as the explanation. Izmailov et al. ran full-batch HMC on CIFAR-10 (ResNet-20-FRN, hundreds of TPU-chip-days) and found essentially **no** CPE without augmentation — $T=1$ was optimal. This is the strongest evidence that CPE is a model-specification phenomenon, not an SG-MCMC artifact.

**Claimed but unablated.** That likelihood misspecification of aleatoric uncertainty is the *dominant* cause (Adlam et al., 2020; Kapoor et al., NeurIPS 2022) rests on Dirichlet/noisy-label likelihoods removing CPE on CIFAR-scale benchmarks; it has not been ablated jointly against curation and prior scale in one factorial design. Aitchison's curation theory is validated by a single decisive but narrow result (CIFAR-10H, below). PAC-Bayes accounts (Pitas & Arbel) explain why $T<1$ optimises a *bound*, not why it optimises test NLL.

**Benchmark-number-only.** Nearly all reported $T^\star$ values are point minima on a coarse log grid ($T \in \{10^{-4},\dots,1\}$) from one or two seeds. Almost no paper reports a confidence interval on $T^\star$.

## 4. What Is Known

- **Wenzel et al. 2020**, ResNet-20 on CIFAR-10, SG-MCMC with augmentation: test accuracy rises from roughly $88\%$ at $T=1$ to roughly $91\%$ near $T\!\sim\!10^{-2}$; CNN-LSTM on IMDB shows a larger relative gap (roughly $81\% \to 86\%$). Scale: $n = 50{,}000$, $d \approx 2.7\times10^5$.
- **Izmailov et al. 2021**, full-batch HMC, ResNet-20-FRN, CIFAR-10, **no** augmentation: $T=1$ is optimal; the tempered curve is flat-to-worse below 1. Same scale, exact-inference arm.
- **Aitchison 2021**: replacing CIFAR-10's hard labels with CIFAR-10H human soft labels (10,000 test images, ~2,500 annotators, ~50 labels/image) removes the CPE. This is the single cleanest causal demonstration that a curation/aleatoric mismatch produces it.
- **Fortuin et al. 2022**: heavy-tailed weight priors matching empirical SGD weight distributions remove CPE for fully-connected nets on MNIST/FashionMNIST but **not** for CNNs, where spatially correlated priors were needed. Scale: $n \le 60{,}000$, $d \le 10^6$.
- **Effect size**: reported $\Delta$ is typically $0.02$–$0.15$ nats/example, i.e. the same order as seed-to-seed variance in some setups. Few papers quantify that variance.
- **Non-identifiability** (Section 2) is an algebraic fact, not an empirical one.

## 5. What Is Not Known

- **Theoretically open.** No theorem stating necessary and sufficient conditions on $(p(y\mid x,\theta), p(\theta), \text{data process})$ under which $T^\star<1$ for *test NLL* (as opposed to a PAC-Bayes bound or a generalized-Bayes risk). Grünwald's safe-Bayes results prove $T<1$ can be needed under misspecification for *consistency* in regression; the transfer to deep classifiers is unproven.
- **Empirically open.** Whether CPE survives at ImageNet scale under exact inference. Full-batch HMC has been run at CIFAR scale only. No one has run the 2×2×2 factorial (soft labels × augmentation × exact/approximate inference) in a single codebase with matched priors and seeds.
- **Methodologically blocked.** "The temperature at which the posterior is best" is not well defined until the prior parameterisation is fixed, because of the $T \leftrightarrow \alpha^2 T$ equivalence. Cross-paper comparisons of $T^\star$ are, strictly, not comparable.

## 6. Why It Is Hard

Three specific obstructions, in order of bite:

1. **Non-identifiability.** $T$ and prior scale are the same knob under full tempering with a Gaussian prior. "The posterior is misspecified" and "the temperature is wrong" are the same statement re-parameterised, so no experiment on the fully tempered family can distinguish them.
2. **Compute cost of the control arm.** The only inference method without stationary-distribution bias at these scales is full-batch HMC, which cost hundreds of TPU-chip-days for a ResNet-20 on CIFAR-10 (Izmailov et al.). At ImageNet scale it is out of reach, so the decisive control cannot currently be run where the effect matters most.
3. **Absent ground truth for aleatoric noise.** Testing the "misspecified aleatoric uncertainty" hypothesis needs the true conditional $p(y\mid x)$. CIFAR-10H is the only large soft-label set at this scale, and it covers the *test* split — the hypothesis concerns the *training* likelihood.

A fourth, minor: the evaluation does not measure what it names. $T^\star$ is chosen on test NLL, so it is a selected statistic; reported $\Delta \sim 0.05$ nats is often within seed noise that is not reported.

## 7. Current Research (as of 2026)

- **Generalized/robust Bayes.** Recasting $T$ as a learned robustness parameter (safe-Bayes, $\beta$-divergence and Gibbs posteriors) rather than an inference hack — Grünwald's line, continued in the Bayesian-deep-learning community.
- **Likelihood repair.** Dirichlet, noisy-label and per-example-noise likelihoods that make $T=1$ optimal by construction (Kapoor, Wilson and collaborators, NYU/Columbia).
- **Prior design.** Function-space and correlated weight priors matched to trained-network statistics (Fortuin, Helsinki/ETH; van der Wilk, Oxford).
- **Diagnosis via bounds.** PAC-Bayes framings in which the CPE is read as *underfitting* of the tempered predictive rather than overconfidence (Pitas & Arbel, TMLR) *(frontier — verify)*.
- **Scaling.** Whether the CPE appears at all for fine-tuned foundation models with low-rank Bayesian posteriors, where $d_{\text{eff}} \ll n$ *(frontier — verify)*.

## 8. Concrete Next Experiment

**Design.** A $2\times2\times2$ factorial on CIFAR-10, ResNet-20-FRN ($d\approx2.7\times10^5$), one codebase, 5 seeds per cell:

- Factor A: hard labels vs. human soft labels on the *training* split (collect ~20 annotations for 5,000 training images; ~100k annotations, roughly \$3–5k on a crowd platform — this is the only new data cost).
- Factor B: augmentation on / off.
- Factor C: inference = full-batch HMC (control arm) vs. cyclical SG-MCMC.

Prior fixed to $\mathcal{N}(0,\alpha^2 I)$ with $\alpha$ **tuned separately at each $T$** so that the prior-scale confound is removed by construction. Temperature grid $T \in \{0.05, 0.1, 0.2, 0.5, 1, 2\}$, dense near 1.

**Control arm.** Cell (soft labels, no augmentation, HMC) — every known misspecification removed.

**Deciding number.** The bootstrap 95% CI on $T^\star$ in the control arm. If it contains $1$, and the CPE reappears when any single factor is reverted, the effect is fully explained by curation + augmentation and the problem moves to *solved*. If $T^\star < 0.5$ with the CI excluding 1, an unidentified cause remains and the theory variant stays open. Secondary readout: $\Delta$ in nats/example against the seed-to-seed standard deviation of $\mathcal{L}_{\text{test}}(1)$; a claim is only meaningful if $\Delta > 3\sigma_{\text{seed}}$.

## 9. Key References

- **[Foundational]** F. Wenzel, K. Roth, B. S. Veeling, J. Świątkowski, L. Tran, S. Mandt, J. Snoek, T. Salimans, R. Jenatton, S. Nowozin. *How Good is the Bayes Posterior in Deep Neural Networks Really?* ICML, 2020. — arXiv:2002.02405
- **[SOTA]** P. Izmailov, S. Vikram, M. D. Hoffman, A. G. Wilson. *What Are Bayesian Neural Network Posteriors Really Like?* ICML, 2021. — arXiv:2104.14421
- **[Theory]** L. Aitchison. *A Statistical Theory of Cold Posteriors in Deep Neural Networks.* ICLR, 2021. — arXiv:2008.05912
- **[Ablation]** L. Noci, K. Roth, G. Bachmann, S. Nowozin, T. Hofmann. *Disentangling the Roles of Curation, Data-Augmentation and the Prior in the Cold Posterior Effect.* NeurIPS, 2021. — arXiv:2106.06596
- **[Theory]** G. Bachmann, L. Noci, T. Hofmann. *How Tempering Fixes Data Augmentation in Bayesian Neural Networks.* ICML, 2022. — arXiv:2205.13900
- **[Ablation]** S. Nabarro, S. Ganev, A. Garriga-Alonso, V. Fortuin, M. van der Wilk, L. Aitchison. *Data Augmentation in Bayesian Neural Networks and the Cold Posterior Effect.* UAI, 2022. — arXiv:2106.05586
- **[SOTA]** S. Kapoor, W. J. Maddox, P. Izmailov, A. G. Wilson. *On Uncertainty, Tempering, and Data Augmentation in Bayesian Neural Networks.* NeurIPS, 2022. — arXiv:2203.16481
- **[Priors]** V. Fortuin, A. Garriga-Alonso, S. W. Ober, F. Wenzel, G. Rätsch, R. E. Turner, M. van der Wilk, L. Aitchison. *Bayesian Neural Network Priors Revisited.* ICLR, 2022. — arXiv:2102.06571
- **[Foundational, statistics]** P. Grünwald, T. van Ommen. *Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It.* Bayesian Analysis, 2017.
- **[Data]** J. C. Peterson, R. M. Battleday, T. L. Griffiths, O. Russakovsky. *Human Uncertainty Makes Classification More Robust.* ICCV, 2019. (CIFAR-10H)
- **[Context]** A. G. Wilson, P. Izmailov. *Bayesian Deep Learning and a Probabilistic Perspective of Generalization.* NeurIPS, 2020. — arXiv:2002.08791
- **[Bounds]** K. Pitas, J. Arbel. *Cold Posteriors through PAC-Bayes.* 2022. — arXiv:2206.11173

## 10. Worked Example

Take the augmentation mechanism concretely. CIFAR-10, $n = 50{,}000$, random crop + horizontal flip. Treat the augmentation set as a finite orbit of size $K$; count the distinct crops from 4-pixel padding ($9\times9=81$) times 2 flips, so $K = 162$.

The SGD/SG-MCMC objective averages over augmentations:

$$\tilde\ell(\theta) = \sum_{i=1}^{n} \frac{1}{K}\sum_{k=1}^{K} \log p(y_i \mid t_k(x_i), \theta).$$

If instead you treat each augmented pair as an independent observation — which is what the *gradient* of a per-epoch augmented pass behaves like when the model can fit each view — the effective log-likelihood is $K$ times larger, i.e. $n_{\text{eff}} = Kn$. To recover a posterior with the concentration the modeller intended for $n$ observations you must divide the energy by $K$: $T \approx 1/K \approx 6\times10^{-3}$.

Compare with measurement: Wenzel et al.'s CIFAR-10 ResNet-20 optimum sits near $T \sim 10^{-2}$. Same order of magnitude. The mechanism predicts the number.

Now the obstruction. Under a Gaussian prior, the fully tempered posterior at $T = 1/162$ is *identically* the untempered-likelihood posterior with prior variance $\alpha^2/162$ — a prior standard deviation shrunk by $12.7\times$. So the same data support two incompatible stories with identical predictions:

| Story | Fix |
|---|---|
| Augmentation inflates $n_{\text{eff}}$ by $K=162$ | Divide energy by $K$ |
| The Gaussian prior is $12.7\times$ too wide | Shrink $\alpha$ |

No amount of sampling from this family separates them, because they are the same distribution. This is why the experiment in Section 8 tunes $\alpha$ independently at each $T$: only then does a residual $T^\star < 1$ carry information about the *likelihood* rather than about a prior scale nobody set carefully in the first place.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*