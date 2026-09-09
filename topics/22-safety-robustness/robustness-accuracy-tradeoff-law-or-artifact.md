---
id: 22-safety-robustness/robustness-accuracy-tradeoff-law-or-artifact
title: "The Robustness-Accuracy Tradeoff as a Law or Artifact"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# The Robustness-Accuracy Tradeoff as a Law or Artifact

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/robustness-accuracy-tradeoff-law-or-artifact` · **Status:** open

## 1. Problem Statement

Every adversarially trained image classifier loses clean accuracy relative to a standard classifier of the same architecture and data budget. The open question is **why**.

Three distinct variants, routinely conflated:

- **Theory variant.** For a given data distribution $\mathcal{D}$, perturbation set $B_\varepsilon$, and hypothesis class $\mathcal{H}$, is $\max_{h \in \mathcal{H}}$ clean accuracy subject to $\varepsilon$-robustness *strictly below* unconstrained max clean accuracy? I.e. is the tradeoff a property of the distribution, or of the estimator?
- **Measurement variant.** Given only an observed pair (clean accuracy, robust accuracy), decompose the clean-accuracy drop into (a) Bayes-optimal conflict, (b) finite-sample estimation error, (c) inner-maximization surrogate error, (d) capacity shortfall, (e) optimization/overfitting artifacts. No accepted decomposition exists.
- **Method variant.** Does there exist a training procedure that attains standard clean accuracy *and* nontrivial certified or empirical $\varepsilon$-robustness at fixed data and compute?

**Solved** would mean: a decomposition of the observed drop whose terms are separately measurable, plus either a lower bound showing the residual is irreducible for realistic $(\mathcal{D}, \varepsilon)$, or a method that removes it.

## 2. Formal Setting

Data $(x, y) \sim \mathcal{D}$ on $\mathcal{X} \times [K]$. Perturbation set $B_\varepsilon(x) = \{x' : \|x' - x\|_p \le \varepsilon\}$.

**Standard risk** and **robust risk** of classifier $f$:

$$R(f) = \Pr_{\mathcal{D}}[f(x) \neq y], \qquad R_\varepsilon(f) = \Pr_{\mathcal{D}}\!\left[\exists x' \in B_\varepsilon(x): f(x') \neq y\right].$$

*Measured as:* $R$ by top-1 error on a held-out split ($n = 10{,}000$ for CIFAR-10). $R_\varepsilon$ is **not** measured — only upper-bounded by an attack. The reported number is $\hat R_{\mathcal{A}}(f) = \frac{1}{n}\sum_i \mathbb{1}[f(\mathcal{A}(x_i)) \ne y_i]$ for attack ensemble $\mathcal{A}$ (AutoAttack, Croce & Hein, ICML 2020), so $\hat R_{\mathcal{A}} \le R_\varepsilon$ with unknown slack.

**Astuteness** at radius $\varepsilon$: $\text{ast}_\varepsilon(f) = 1 - R_\varepsilon(f)$. Define the frontier

$$\Phi_\varepsilon(\mathcal{H}, n) = \max\{1 - R(f) : f \in \mathcal{H}_n,\ R_\varepsilon(f) \le R(f) + \delta\}.$$

The "law" claim is $\Phi_\varepsilon(\mathcal{H}, \infty) < \Phi_0(\mathcal{H}, \infty)$ for image distributions. The "artifact" claim is that the gap lives in $\mathcal{H}_n$ — the *reachable* set under SGD with finite $n$.

**$r$-separation.** $\mathcal{D}$ is $r$-separated if for all $x, x'$ with $y \ne y'$ in the support, $\|x - x'\|_p > 2r$. If $\mathcal{D}$ is $r$-separated and $\varepsilon < r$, a classifier with $\text{ast}_\varepsilon = 1$ exists (Yang et al., NeurIPS 2020). This makes the theory variant non-vacuous only when separation fails.

Adversarial training (Madry et al., ICLR 2018) optimizes $\min_\theta \mathbb{E}\,\max_{x' \in B_\varepsilon(x)} \ell(f_\theta(x'), y)$; the inner max is approximated by $k$-step PGD, so the realized objective is a lower bound on the true one.

**Assumptions known to be violated in practice:**
- *Test labels are the Bayes label.* False near class boundaries; CIFAR-10 has ~3% label noise (Northcutt et al., NeurIPS D&B 2021), which alone caps clean accuracy and interacts with $\varepsilon$.
- *$B_\varepsilon$ is label-preserving.* At ImageNet $\varepsilon = 4/255$ this is plausible; at CIFAR-10 $\varepsilon = 8/255$ on $32\times32$ images it is not uniformly true.
- *Inner max is solved.* PGD-10 leaves measurable slack; AutoAttack re-evaluations have cut reported robustness by 10–20 points on dozens of published defenses.
- *Equal class priors / balanced costs.* Dobriban et al. (IEEE Trans. Inf. Theory, 2023) show the tradeoff magnitude depends explicitly on class imbalance.

## 3. State of the Art

**Theory SOTA (established).**
- Tsipras et al. (ICLR 2019) construct a distribution where *any* $\varepsilon$-robust classifier has clean accuracy at most $1/\!\left(1 + \text{(weak-feature advantage)}\right)$ — a genuine theorem, but on a hand-built Gaussian-mixture with one strong and $d$ weak features. It proves the tradeoff is *possible*, not that it holds for images.
- Zhang et al. (TRADES, ICML 2019) give a decomposition of robust error into natural error plus a boundary term, with a surrogate bound. Established.
- Bhagoji et al. (NeurIPS 2019) and Pydi & Jog (ICML 2020) give optimal-transport lower bounds on achievable robust risk for any classifier. These are distribution-dependent and computable only for low-dimensional or binary cases.
- Yang et al. (NeurIPS 2020) measure $r$-separation on real data and conclude no tradeoff is *necessary* on MNIST/CIFAR-10/SVHN/ResImageNet at standard $\varepsilon$. Established and the strongest "artifact" evidence.
- Raghunathan et al. (ICML 2020) show adversarial training can hurt *generalization* in a well-specified linear setting even when robust and accurate predictors coexist — the drop is an estimation effect, removable with more data.

**Empirical SOTA (RobustBench, Croce et al., NeurIPS D&B 2021).**
- CIFAR-10, $\ell_\infty$, $\varepsilon = 8/255$: best AutoAttack robustness ~71% at ~93.3% clean (Wang et al., *Better Diffusion Models Further Improve Adversarial Training*, ICML 2023; Peng et al., BMVC 2023). Standard WRN-28-10 clean accuracy is ~96%.
- ImageNet, $\ell_\infty$, $\varepsilon = 4/255$: ~59.6% robust at ~78.9% clean (Liu et al., *A Comprehensive Study on Robustness of Image Classification Models*, CVPR 2023, Swin-L). Non-robust Swin-L is ~87% clean.

**Claimed but unablated.** That the residual ~3-point CIFAR-10 gap is irreducible. Nobody has ablated it against label noise, $\varepsilon$ exceeding local separation, or augmentation-schedule differences between the robust and standard arms. The AdvProp result (Xie et al., CVPR 2020) — adversarial examples *raising* ImageNet clean accuracy by ~0.7 points at small $\varepsilon$ with separate BatchNorm — is a benchmark number whose mechanism remains unisolated.

## 4. What Is Known

- **Magnitude shrank as data grew, at fixed $\varepsilon$.** CIFAR-10 $\varepsilon=8/255$: Madry 2018, 87.3% clean / 44.0% AutoAttack; Carmon et al. 2019 (+500k unlabeled TinyImages), 89.7% / 59.5%; Gowal et al. 2021 (100M DDPM-generated), 88.7% / 66.1%; Wang et al. 2023 (EDM-generated), 93.3% / 70.7%. Clean *and* robust accuracy rose together across four generations. A fixed Pareto law cannot produce that.
- **Sample complexity is provably higher.** Schmidt et al. (NeurIPS 2018): Gaussian-mixture separation needs $\Theta(\sqrt{d})$ more samples for robust than standard generalization. Scale: $d = 3072$ for CIFAR-10, predicting a ~50× data multiplier.
- **Robust overfitting is real and large.** Rice et al. (ICML 2020): robust test error degrades after the first LR decay; early stopping recovers ~5–8 points of robust accuracy on CIFAR-10 PreActResNet-18. Part of the measured gap is optimization, not distribution.
- **Separation measurements.** Yang et al. report minimum $\ell_\infty$ inter-class test distance ~0.21 on CIFAR-10 versus the standard $\varepsilon = 0.031$, and ~0.74 on MNIST versus $\varepsilon = 0.1$ — roughly $7\times$ and $7\times$ headroom.
- **Scaling behaves lawfully but saturates slowly.** Bartoldson et al. (ICML 2024) fit robustness scaling laws in compute and data and project that reaching ~90% AutoAttack robustness at CIFAR-10 $\varepsilon=8/255$ needs compute orders of magnitude beyond current runs.
- **Robust features transfer better.** Salman et al. (NeurIPS 2020): ImageNet-robust backbones beat standard ones on downstream transfer, so "robustness destroys useful features" is false as stated.

## 5. What Is Not Known

- **Theoretically open.** Whether $\Phi_\varepsilon(\mathcal{H}, \infty) < \Phi_0(\mathcal{H}, \infty)$ for any real image distribution at deployed $\varepsilon$. No proof either way. Existing positive results are on synthetic distributions; the OT lower bounds are not computable at $d = 3072$.
- **Empirically open.** Whether the residual CIFAR-10 gap (93.3 vs 96.0) vanishes under unbounded generated data. Runnable — scale the Wang et al. recipe by $10$–$100\times$ generated images — but unrun.
- **Methodologically blocked.** The decomposition itself. $R_\varepsilon$ is only ever upper-bounded by an attack, and the Bayes-optimal astute risk for images is unestimable, so "irreducible conflict" has no measurable definition. Detecting adversarial examples is about as hard as classifying them robustly (Tramèr, ICML 2022), which closes the obvious shortcut of measuring conflict via a separate detector.

## 6. Why It Is Hard

**The primary obstruction is non-identifiability under a confounded measurement.** The observed clean-accuracy drop is a sum of five terms and every published experiment varies at least two of them at once. Adversarial training changes the objective, the effective augmentation distribution, the optimal LR schedule, the early-stopping point, and the effective batch statistics simultaneously. There is no control arm that isolates "same procedure, $\varepsilon = 0$" — setting $\varepsilon = 0$ recovers standard training and removes the mechanism under test.

Secondary: the ground truth is absent. Deciding whether a specific perturbed $x'$ should keep label $y$ requires a human oracle, and human labels on $\varepsilon = 8/255$ CIFAR-10 perturbations are themselves noisy. So the denominator of "accuracy" is not fixed across the two arms.

Third: compute. The data-scaling arm that would settle the empirical variant costs roughly $10^3$ GPU-hours per point on CIFAR-10 WRN-70-16 with 100M synthetic images, and the relevant question is about ImageNet.

## 7. Current Research (as of 2026)

- **Synthetic-data scaling** for robust training — diffusion-generated data as the lever (Gowal/Rebuffi/Qin at DeepMind; Wang, Pang, Lin at Sea AI Lab). Mainstream and reproduced.
- **Robustness scaling laws** (Bartoldson, Kailkhura et al., LLNL) — fitting $R_\varepsilon$ against compute/data to extrapolate whether the frontier converges. *(frontier — verify)*
- **Separation-aware and instance-adaptive $\varepsilon$** — making the radius respect local margin rather than a global constant, descended from Yang et al. *(frontier — verify)*
- **Certified training** (randomized smoothing, IBP successors) where the tradeoff is measured against a *proved* bound rather than an attack, removing the slack term at the cost of much weaker accuracy.
- **Transfer to LLMs** — whether jailbreak robustness trades against capability is the same question in a setting where $B_\varepsilon$ is not even metric. Measurement is worse defined than in vision.

## 8. Concrete Next Experiment

**Question to decide:** is the residual CIFAR-10 clean-accuracy gap an estimation artifact?

**Scale.** CIFAR-10, WRN-28-10, $\ell_\infty$, $\varepsilon = 8/255$. Five synthetic-data budgets: $\{1, 5, 20, 50, 200\}$M EDM-generated images, matching the Wang et al. (ICML 2023) recipe. ~5 runs × ~400 GPU-hours ≈ 2,000 A100-hours.

**Control arm (the part usually missing).** At each budget, train a *standard* model on the **identical** image stream, identical schedule, identical early-stopping criterion, with $\varepsilon = 0$ — and additionally a third arm with $\varepsilon = 8/255$ but the inner max replaced by a *random* point in $B_\varepsilon$ (matched augmentation strength, no adversarial structure). The random-$B_\varepsilon$ arm separates "training on perturbed inputs" from "training on worst-case inputs".

**Deciding number.** The clean-accuracy gap $\Delta(n) = \text{Acc}_{\text{std}}(n) - \text{Acc}_{\text{AT}}(n)$ fitted as $\Delta(n) = \Delta_\infty + c\,n^{-\alpha}$. **If $\Delta_\infty < 0.5$ points with a 95% CI excluding 1 point, the tradeoff at this $\varepsilon$ is an estimation artifact. If $\Delta_\infty > 2$ points, it is a law at this radius and hypothesis class.** Current data gives three points on this curve ($\Delta \approx 8.7, 7.3, 2.7$) — enough to suspect decay, not enough to fit $\Delta_\infty$.

## 9. Key References

- **[Foundational]** D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, A. Madry. *Robustness May Be at Odds with Accuracy.* ICLR 2019. — arXiv:1805.12152
- **[Foundational]** A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018. — arXiv:1706.06083
- **[Foundational]** H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, M. Jordan. *Theoretically Principled Trade-off between Robustness and Accuracy.* ICML 2019. — arXiv:1901.08573
- **[Key counter-result]** Y.-Y. Yang, C. Rashtchian, H. Zhang, R. Salakhutdinov, K. Chaudhuri. *A Closer Look at Accuracy vs. Robustness.* NeurIPS 2020. — arXiv:2003.02460
- **[Key counter-result]** A. Raghunathan, S. M. Xie, F. Yang, J. Duchi, P. Liang. *Understanding and Mitigating the Tradeoff Between Robustness and Accuracy.* ICML 2020. — arXiv:2002.10716
- **[Sample complexity]** L. Schmidt, S. Santurkar, D. Tsipras, K. Talwar, A. Madry. *Adversarially Robust Generalization Requires More Data.* NeurIPS 2018. — arXiv:1804.11285
- **[SOTA]** Z. Wang, X. Pang, C. Du, M. Lin, W. Liu, S. Yan. *Better Diffusion Models Further Improve Adversarial Training.* ICML 2023. — arXiv:2302.04638
- **[SOTA]** S. Gowal, S.-A. Rebuffi, O. Wiles, F. Stimberg, D. A. Calian, T. Mann. *Improving Robustness using Generated Data.* NeurIPS 2021. — arXiv:2110.09468
- **[Measurement]** F. Croce, M. Hein. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks.* ICML 2020. — arXiv:2003.01690
- **[Benchmark]** F. Croce, M. Andriushchenko, V. Sehwag, E. Debenedetti, N. Flammarion, M. Chiang, P. Mittal, M. Hein. *RobustBench: a standardized adversarial robustness benchmark.* NeurIPS Datasets & Benchmarks 2021. — arXiv:2010.09670
- **[Optimization artifact]** L. Rice, E. Wong, J. Z. Kolter. *Overfitting in Adversarially Robust Deep Learning.* ICML 2020. — arXiv:2002.11569
- **[Theory]** E. Dobriban, H. Hassani, D. Hong, A. Robey. *Provable Tradeoffs in Adversarially Robust Classification.* IEEE Transactions on Information Theory, 2023. — arXiv:2006.05161
- **[Theory]** A. N. Bhagoji, D. Cullina, P. Mittal. *Lower Bounds on Adversarial Robustness from Optimal Transport.* NeurIPS 2019. — arXiv:1909.12272
- **[Counterexample to the simple story]** C. Xie, M. Tan, B. Gong, J. Wang, A. Yuille, Q. V. Le. *Adversarial Examples Improve Image Recognition.* CVPR 2020. — arXiv:1911.09665
- **[Scaling]** B. R. Bartoldson, J. Diffenderfer, K. Parasyris, B. Kailkhura. *Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies.* ICML 2024. — arXiv:2404.09349
- **[Survey]** A. Chakraborty, M. Alam, V. Dey, A. Chattopadhyay, D. Mukhopadhyay. *A Survey on Adversarial Attacks and Defences.* CAAI Transactions on Intelligence Technology, 2021.

## 10. Worked Example

Take CIFAR-10, $\ell_\infty$, $\varepsilon = 8/255 = 0.031$.

**Step 1 — existence.** Yang et al. measure the minimum inter-class $\ell_\infty$ distance in the CIFAR-10 test set at ~$0.21$. Since $0.031 < 0.21/2 = 0.105$, a classifier exists with $\text{ast}_\varepsilon = 1$ on the test set: 100% clean, 100% robust. The information-theoretic frontier at this radius, *measured on the data we actually evaluate on*, has no tradeoff at all.

**Step 2 — achieved.** Best measured: 93.25% clean / 70.69% AutoAttack (Wang et al. 2023, WRN-70-16, 50M EDM images). Standard WRN-70-16: ~96.5% clean.

**Step 3 — the arithmetic.**

```
existence frontier   : clean 100.0   robust 100.0
standard training    : clean  96.5   robust   0.0
best robust training : clean  93.3   robust  70.7
clean gap (Δ)        :        3.2 points
robust gap to exist. :       29.3 points
```

**Step 4 — where the obstruction becomes visible.** $\Delta = 3.2$ is the entire empirical basis for calling the tradeoff a law at this radius, and it cannot be attributed. Three candidate explanations each account for all of it:

- **Label noise.** CIFAR-10 test-set label error is ~0.5–3%. Robust training smooths decision boundaries and so cannot fit mislabeled points that standard training memorizes. This predicts $\Delta \approx$ the noise rate and predicts $\Delta \to$ noise rate, not $0$, under infinite data — indistinguishable from a law by a single measurement.
- **Estimation error.** $\Delta$ fell $8.7 \to 7.3 \to 2.7 \to 3.2$ as the data budget went $50\text{k} \to 550\text{k} \to 100\text{M}$ (with architecture also changing). Consistent with $\Delta_\infty = 0$.
- **Attack slack.** The 70.69% is an upper bound on $1 - R_\varepsilon$. If AutoAttack leaves 5 points of slack, the true frontier point is 93.3/65.7, and a *weaker* tradeoff in the reported numbers reflects a *weaker attack*, not a better model.

No existing experiment separates these, because none holds the image stream, the schedule, and the evaluation oracle fixed while varying only the inner maximization. That is the experiment in Section 8, and until it runs the field's headline number — 3.2 points — is a quantity whose units are unknown.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*