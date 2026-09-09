---
id: 33-uncertainty-calibration/deep-ensembles-versus-bayesian-posteriors
title: "Deep Ensembles Versus Bayesian Posteriors"
topic: 33-uncertainty-calibration
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Deep Ensembles Versus Bayesian Posteriors

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/deep-ensembles-versus-bayesian-posteriors` · **Status:** open

## 1. Problem Statement

A deep ensemble trains $M$ copies of the same network from independent random initializations and averages their predictive distributions. It beats almost every approximate-Bayesian method on calibration and shift robustness benchmarks. The problem is to say what it is doing.

Three variants, routinely conflated:

- **Theory.** Is the deep-ensemble predictive $\bar p_M$ an approximation to a Bayesian posterior predictive $p_{\text{Bayes}}$ under some prior, or is it a different estimator that happens to score well? Does $\bar p_M$ converge to anything as $M \to \infty$, and to what?
- **Measurement.** Given samples from a reference posterior and from an ensemble, what statistic decides whether they represent the same predictive belief? Marginal metrics (NLL, ECE, AUROC) are not sufficient: two procedures can match on every single-input marginal and disagree on every joint.
- **Method.** If the ensemble is *not* Bayesian, is that a defect or a feature? Under distribution shift, does the gap favour the ensemble, and is the mechanism identifiable?

Solving it means: a reference posterior at nontrivial scale, a joint-predictive statistic that discriminates, and a measured answer with matched compute.

## 2. Formal Setting

Data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, network $f_\theta: \mathcal{X} \to \mathbb{R}^K$, $\theta \in \mathbb{R}^d$, likelihood $p(y \mid x, \theta) = \mathrm{softmax}(f_\theta(x))_y$, prior $p(\theta)$ (in practice $\mathcal{N}(0, \sigma^2 I)$ with $\sigma^2$ set to match weight decay).

**Tempered posterior**, the object actually sampled:
$$p_T(\theta \mid \mathcal{D}) \propto \exp\!\left(\tfrac{1}{T}\big[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\big]\right), \quad T > 0,$$
with $T=1$ the Bayes posterior. Posterior predictive $p_{\text{Bayes}}(y \mid x) = \mathbb{E}_{\theta \sim p_1}[p(y\mid x,\theta)]$, measured as a Monte Carlo average over $S$ retained HMC states after discarding burn-in, with $\hat R < 1.1$ across chains as the (weak) convergence check.

**Deep ensemble.** $\theta^{(m)} = \mathcal{A}(\mathcal{D}, \xi_m)$ where $\mathcal{A}$ is SGD with augmentation and $\xi_m$ is the seed (init, shuffling, augmentation draws). Predictive $\bar p_M(y \mid x) = \frac{1}{M}\sum_m p(y \mid x, \theta^{(m)})$. The induced $q(\theta) = \frac{1}{M}\sum_m \delta_{\theta^{(m)}}$ has no closed form — it is defined only by the algorithm.

**Discriminating statistic.** Marginal NLL $-\log \bar p(y\mid x)$ cannot separate the two. Use the joint log-loss over a $\tau$-tuple $x_{1:\tau}$ drawn dyadically (two anchor inputs, $\tau$ draws with replacement):
$$\mathcal{L}_\tau = -\mathbb{E}\left[\log \textstyle\int \prod_{t=1}^{\tau} p(y_t \mid x_t, \theta)\, dq(\theta)\right],$$
which is sensitive to the *dependence* the belief induces across inputs — the part a posterior encodes and a marginal metric discards (Osband et al., NeurIPS 2022).

**Functional distance.** Compare in function space, not weight space: $d(q, p) = \mathbb{E}_{x}\, \mathrm{JSD}\big(\mathbb{E}_q[p(\cdot\mid x,\theta)] \,\|\, \mathbb{E}_p[\cdot]\big)$, plus the disagreement rate $\mathbb{E}_{x}\mathbb{E}_{\theta,\theta'}[\mathbb{1}\{\arg\max f_\theta(x) \neq \arg\max f_{\theta'}(x)\}]$.

**Assumptions known to be violated.** (i) The likelihood is not a likelihood: data augmentation multiplies the effective sample size, so the "posterior" being sampled is not $p(\theta\mid\mathcal{D})$ (Nabarro et al., UAI 2022; Kapoor et al., NeurIPS 2022). (ii) $T=1$ is rarely the best-performing temperature — the cold posterior effect (Wenzel et al., ICML 2020). (iii) Gaussian priors on weights induce function-space priors nobody has validated. (iv) $\hat R$ diagnostics on weight coordinates are meaningless under the permutation symmetry of hidden units; the posterior is non-identifiable up to a group of order $\sim \prod_l h_l!$.

## 3. State of the Art

**Empirical SOTA (established).** Deep ensembles remain the strongest practical uncertainty method on shifted data. Ovadia et al. (NeurIPS 2019) measured across MNIST/CIFAR-10/ImageNet plus text and genomics: ensembles of $M=5$ had the lowest ECE at every corruption level, and no approximate-Bayesian method (MC dropout, SVI, last-layer methods, temperature scaling) closed the gap. Independently reproduced by Ashukha et al. (ICLR 2020) and Gustafsson et al. (CVPR-W 2020).

**Theory SOTA.** Three partial links, each with a caveat:
- He, Lakshminarayanan & Teh (NeurIPS 2020): in the infinite-width NTK limit, ensembles trained with *randomized prior functions* and a modified loss produce exact posterior samples from the corresponding GP. Established, but the construction is not what practitioners run and the limit removes feature learning.
- D'Angelo & Fortuin (NeurIPS 2021): adding a kernelized repulsion term makes the ensemble members a Wasserstein-gradient-flow discretization of the posterior. Established for the modified algorithm; says nothing about the vanilla one.
- Wild, Ghalebikesabi, Sejdinovic & Knoblauch (NeurIPS 2023): vanilla deep ensembles are exact minimizers of a *generalized* variational objective over the family of Dirac mixtures. Established, but generalized-Bayes objectives are broad enough that the label "Bayesian" loses discriminating power.

**Claimed but unablated.** That ensembles "explore multiple modes" while HMC/VI stay in one basin is a widely repeated explanation traced to Fort, Hu & Lakshminarayanan (2019). Their evidence — function-space diversity and loss-landscape geometry — is real; the causal claim that mode multiplicity *is* the source of the calibration gain has not been isolated by ablation.

**Benchmark-number-only.** Most reported ensemble-vs-Bayes comparisons hold $M$ fixed rather than compute fixed. Ashukha et al. (ICLR 2020) is the exception, and their deep-ensemble-equivalent score showed most cheap Bayesian approximations are worth fewer than 2 independently trained networks.

## 4. What Is Known

- **The reference posterior exists at CIFAR-10 scale.** Izmailov et al. (ICML 2021) ran full-batch HMC on ResNet-20-FRN, CIFAR-10, at $T=1$ with no augmentation: hundreds of TPU-core-days per chain, 3 chains. Single-chain test accuracy $\approx 89.6\%$, chain-mixture $\approx 90.9\%$, above the SGD baseline and above a comparable deep ensemble.
- **The true posterior is worse under shift.** In the same study, HMC's advantage inverts on CIFAR-10-C: under corruption it falls below both SGD and deep ensembles. This is the single most load-bearing datum in the debate — being *more* Bayesian hurt robustness.
- **Cold posteriors are real and partly explained.** Wenzel et al. (ICML 2020) found test error decreasing monotonically as $T$ falls from 1 toward $10^{-2}$ on ResNet-20/CIFAR-10 and CNN-LSTM/IMDB. Much of the effect is attributable to augmentation-inflated likelihoods and dataset curation (Aitchison, 2021; Noci et al., NeurIPS 2021; Nabarro et al., UAI 2022) — but not all of it in every setting.
- **Diversity source.** Fort et al. (2019): ensembling over random inits gives far higher function-space diversity than subspace/within-basin sampling; SWAG-style single-basin approximations recover only a fraction of the ensemble gain (Maddox et al., NeurIPS 2019).
- **Ensembles are not uniquely necessary.** Abe et al. (NeurIPS 2022) showed a substantial part of ensemble gain on CIFAR-10/100 and ImageNet is matched by a single model of comparable capacity, and that ensemble gains do not require multi-modality in the way usually assumed.
- **Marginal metrics saturate.** On the Neural Testbed (Osband et al., NeurIPS 2022), methods indistinguishable on marginal NLL differ by large margins on joint NLL at $\tau=10$.

## 5. What Is Not Known

- **Theoretically open.** Whether the vanilla-SGD ensemble measure $q_M$ converges, as $M\to\infty$ at finite width and finite $n$, to any $p_T(\theta\mid\mathcal{D})$ for a data-dependent $T$ and prior. No proof either way. Also open: whether the generalized-VI identification of Wild et al. (2023) implies anything about predictive behaviour under shift.
- **Empirically open.** Whether the HMC-vs-ensemble ordering at CIFAR-10 scale survives (a) augmentation-corrected likelihoods, (b) matched compute rather than matched $M$, and (c) architectures at ImageNet or LLM-finetuning scale. Runnable; unrun, because a single reference posterior costs $10^2$–$10^3$ accelerator-days.
- **Methodologically blocked.** "Is method $A$ closer to the posterior than method $B$?" has no agreed metric. Weight-space divergences are meaningless under permutation symmetry; function-space divergences require choosing an input distribution, and the answer changes when that distribution shifts — which is exactly the regime of interest. Joint NLL is the best candidate but the choice of $\tau$ and of the input-tuple sampler is unstandardized.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth at scale.** Above ~$10^7$ parameters no one can produce samples certified to come from $p(\theta\mid\mathcal{D})$. Every comparison at ImageNet scale compares two approximations with unknown bias.
2. **Non-identifiability.** Hidden-unit permutation and scaling symmetries mean the posterior has astronomically many equivalent modes. "Ensembles find different modes" is therefore not automatically a statement about different functions, and weight-space convergence diagnostics ($\hat R$, ESS) are invalid as written.
3. **An evaluation that does not measure what it names.** ECE and NLL are marginal; a Bayesian posterior is a claim about the *joint* over unseen inputs. Ranking methods by ECE and calling it a test of Bayesian-ness is a category error. This is why the debate has run a decade without resolution: the leaderboard was never measuring the disputed quantity.

## 7. Current Research (as of 2026)

- **Reference-posterior infrastructure.** Extending the Izmailov/Wilson-line HMC chains to larger models and to augmented likelihoods; the released CIFAR-10 chains remain the community's only certified reference. Groups: NYU (Wilson), Google DeepMind.
- **Joint-predictive evaluation.** Epistemic neural networks and the Neural Testbed line (Osband and collaborators, DeepMind) push joint NLL as the primary metric; adoption outside RL is still partial *(frontier — verify)*.
- **Generalized-Bayes framing.** Knoblauch, Fortuin, Sejdinovic and collaborators: treat deep ensembles as legitimate generalized-Bayes objects rather than failed approximations, and study which loss/divergence pair they optimize.
- **Scale-out to LLMs.** Whether ensembles of fine-tuned LLMs, LoRA ensembles, or checkpoint ensembles carry epistemic signal usable for abstention. Papamarkou et al. (ICML 2024 position paper) argue the case; empirical resolution at frontier scale is not public *(frontier — verify)*.
- **Cold-posterior closure.** Whether a correctly specified augmentation likelihood eliminates the effect entirely, or a residual remains attributable to prior misspecification.

## 8. Concrete Next Experiment

**Question.** At fixed compute, does a deep ensemble's predictive belief differ from the true posterior's in a way that a joint metric detects, and does that difference explain the shift-robustness gap?

**Scale.** CIFAR-10, ResNet-20-FRN — the one setting where a certified reference posterior already exists (Izmailov et al., ICML 2021, chains publicly released). No new HMC run required for the main arm.

**Arms.**
1. Reference: $S=240$ retained HMC states, $T=1$, no augmentation.
2. Treatment: deep ensemble, $M \in \{5, 10, 20, 40\}$, *no augmentation* (matching the HMC likelihood exactly — this control is usually skipped and is the reason existing comparisons are confounded).
3. Control arm: HMC states thinned to $M$ samples, so ensemble and posterior are compared at identical predictive-average cardinality.
4. Second control: single model with $M\times$ width, matched parameter count (the Abe et al. confound).

**Measurement.** Joint NLL $\mathcal{L}_\tau$ at $\tau=10$ under dyadic input sampling, on (a) clean CIFAR-10 test, (b) CIFAR-10-C at severity 3, (c) CIFAR-10.1.

**Deciding number.** $\Delta = \mathcal{L}_{10}^{\text{ens},M=20} - \mathcal{L}_{10}^{\text{HMC-thinned},M=20}$ on CIFAR-10-C severity 3, in nats per 10-input tuple, with bootstrap CI over 2000 tuples. If $\Delta < -0.05$ nats with the CI excluding zero, the ensemble encodes a *better* joint belief than the true posterior under shift, and "ensembles approximate the posterior" is falsified as an explanation of their success. If $|\Delta| < 0.02$, the ensemble is behaviourally posterior-equivalent and the remaining gap in published results is attributable to augmentation, not to Bayes.

**Cost estimate.** ~600 GPU-hours for arms 2–4 on A100s; arm 1 is a download.

## 9. Key References

- **[Foundational]** Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS, 2017. — arXiv:1612.01474
- **[Foundational]** Andrew Gordon Wilson, Pavel Izmailov. *Bayesian Deep Learning and a Probabilistic Perspective of Generalization.* NeurIPS, 2020. — arXiv:2002.08791
- **[SOTA]** Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, Andrew Gordon Wilson. *What Are Bayesian Neural Network Posteriors Really Like?* ICML, 2021. — arXiv:2104.14421
- **[SOTA]** Florian Wenzel, Kevin Roth, Bastiaan S. Veeling, Jakub Świątkowski, Linh Tran, Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton, Sebastian Nowozin. *How Good is the Bayes Posterior in Deep Neural Networks Really?* ICML, 2020. — arXiv:2002.02405
- **[SOTA]** Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS, 2019. — arXiv:1906.02530
- **[SOTA]** Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, Dmitry Vetrov. *Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning.* ICLR, 2020. — arXiv:2002.06470
- **[Theory]** Bobby He, Balaji Lakshminarayanan, Yee Whye Teh. *Bayesian Deep Ensembles via the Neural Tangent Kernel.* NeurIPS, 2020. — arXiv:2007.05864
- **[Theory]** Francesco D'Angelo, Vincent Fortuin. *Repulsive Deep Ensembles are Bayesian.* NeurIPS, 2021. — arXiv:2106.11642
- **[Theory]** Veit Wild, Sahra Ghalebikesabi, Dino Sejdinovic, Jeremias Knoblauch. *A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods.* NeurIPS, 2023. — arXiv:2305.15027
- **[Measurement]** Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, Botao Hao, Morteza Ibrahimi, Dieterich Lawson, Xiuyuan Lu, Brendan O'Donoghue, Benjamin Van Roy. *The Neural Testbed: Evaluating Joint Predictions.* NeurIPS, 2022. — arXiv:2110.04629
- **[Ablation]** Taiga Abe, Estefany Kelly Buchanan, Geoff Pleiss, Richard Zemel, John P. Cunningham. *Deep Ensembles Work, But Are They Necessary?* NeurIPS, 2022. — arXiv:2202.06985
- **[Confound]** Seth Nabarro, Stoil Ganev, Adrià Garriga-Alonso, Vincent Fortuin, Mark van der Wilk, Laurence Aitchison. *Data Augmentation in Bayesian Neural Networks and the Cold Posterior Effect.* UAI, 2022. — arXiv:2106.05586
- **[Diversity]** Stanislav Fort, Huiyi Hu, Balaji Lakshminarayanan. *Deep Ensembles: A Loss Landscape Perspective.* 2019. — arXiv:1912.02757
- **[Survey/Position]** Theodore Papamarkou, Maria Skoularidou, Konstantina Palla, Laurence Aitchison, Julyan Arbel, David Dunson, Maurizio Filippone, Vincent Fortuin, Philipp Hennig, José Miguel Hernández-Lobato, Aliaksandr Hubin, Alexander Immer, Theofanis Karaletsos, Mohammad Emtiyaz Khan, Agustinus Kristiadi, Yingzhen Li, Stephan Mandt, Christopher Nemeth, Michael A. Osborne, Tim G. J. Rudner, David Rügamer, Yee Whye Teh, Max Welling, Andrew Gordon Wilson, Ruqi Zhang. *Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI.* ICML, 2024.

## 10. Worked Example

Take two-way disagreement on a single CIFAR-10 test image under fog corruption, severity 3. Suppose:

- HMC ($M=20$ thinned samples): 14 samples predict *ship*, 6 predict *airplane*. Marginal predictive $\bar p(\text{ship}) = 0.70$, $\bar p(\text{airplane}) = 0.30$.
- Deep ensemble ($M=20$): 14 members predict *ship*, 6 predict *airplane*. Same marginal, $0.70/0.30$.

Marginal NLL is identical: $-\log 0.70 = 0.357$ nats for either, if the label is *ship*. ECE contribution is identical. On every single-input metric the two are the same object.

Now take a $\tau=2$ tuple of two fog-corrupted ships, $x_1$ and $x_2$, and ask for the joint. Suppose the HMC samples are *coupled* — the 6 airplane-voting samples are the same 6 parameter draws on both images (a genuine shared hypothesis: "under fog, hull edges look like wings"). Then
$$\hat p_{\text{HMC}}(\text{ship},\text{ship}) = \tfrac{1}{20}\textstyle\sum_s p_s(\text{ship}|x_1)p_s(\text{ship}|x_2) \approx 0.70,$$
joint NLL $= 0.357$ nats. Suppose the ensemble members are *decoupled* — a different 6 members dissent on each image. Then the joint factorizes, $\approx 0.70 \times 0.70 = 0.49$, joint NLL $= 0.713$ nats.

The gap is $0.36$ nats per 2-tuple, on a pair where every marginal metric reported zero difference. Extended to $\tau=10$ with dyadic sampling the gap compounds roughly linearly in $\tau$ under full decoupling.

This is the obstruction made concrete. The correlation structure — whether dissent is a shared hypothesis or independent noise — is the entire content of the Bayesian claim, and it is exactly what the benchmarks the field ranks on integrate away. Whichever direction the sign actually falls in (and it is not obvious the ensemble is the decoupled one; multi-basin members may share more corruption-specific structure than thinned HMC states within a basin), the number has not been measured on the released CIFAR-10 chains. That measurement is a few hundred GPU-hours away and would settle more than a decade of leaderboard argument.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*