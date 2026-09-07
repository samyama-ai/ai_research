---
id: 22-safety-robustness/certified-robustness-beyond-smoothing-radii
title: "Provable Certified Robustness Beyond Randomized Smoothing Radii"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Certified Robustness Beyond Randomized Smoothing Radii

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/certified-robustness-beyond-smoothing-radii` · **Status:** open

## 1. Problem Statement

Randomized smoothing certifies a ball around an input in which a classifier's prediction provably cannot change. On ImageNet it delivers $\ell_2$ radii of roughly $0.5$–$2.0$ at usable accuracy, which converts to an $\ell_\infty$ radius near $1/255$. Empirical attacks operate at $\ell_\infty = 4/255$–$8/255$ and, increasingly, outside any $\ell_p$ ball at all. The problem is to obtain **sound certificates at threat-model scales that matter**, on models at deployment scale.

Three variants, with different difficulty:

- **Theory.** Is the current radius a fundamental limit of noise-based certification, or an artifact of Gaussian smoothing and the Neyman–Pearson analysis? For $\ell_\infty$ in dimension $d$ there is a partial impossibility result; for $\ell_2$ and for non-i.i.d. smoothing measures there is not.
- **Method.** Build a certifier — deterministic, probabilistic, or hybrid — that certifies a nontrivial fraction of ImageNet at $\ell_\infty \geq 4/255$ without collapsing clean accuracy.
- **Measurement.** Define a certificate for threat models people actually care about (semantic edits, patch, prompt-space perturbation) such that the certified quantity is the quantity of interest. Currently it is not.

A solution to the method variant: $\geq 30\%$ certified top-1 accuracy on ImageNet at $\ell_\infty = 4/255$, with a machine-checkable soundness argument and certification cost $\leq 10$ s/image.

## 2. Formal Setting

Classifier $f: \mathbb{R}^d \to \mathcal{Y}$, input $x$, label $y$. A **certificate** at $x$ is a radius $R(x) \ge 0$ with the soundness guarantee

$$\forall \delta,\ \|\delta\|_p \le R(x) \implies g(x+\delta) = g(x),$$

for the certified predictor $g$. **Certified accuracy at radius $r$** is measured as $\frac{1}{n}\sum_i \mathbb{1}[g(x_i) = y_i \wedge R(x_i) \ge r]$ over the full test set, counting abstentions as errors — the only honest accounting, and one that some reported numbers do not use.

**Randomized smoothing** (Cohen–Rosenfeld–Kolter, ICML 2019). With base classifier $F$ and $\varepsilon \sim \mathcal{N}(0,\sigma^2 I_d)$, define $g(x) = \arg\max_c \Pr[F(x+\varepsilon)=c]$ and $p_A = \Pr[F(x+\varepsilon)=y]$. Then

$$R(x) = \sigma\,\Phi^{-1}(p_A),$$

and this is tight for the class of all base classifiers with that $p_A$.

**What is actually measured.** $p_A$ is never observed. It is replaced by a Clopper–Pearson lower confidence bound $\underline{p_A}$ from $n$ Monte Carlo samples at level $\alpha$, giving $\hat R = \sigma\Phi^{-1}(\underline{p_A})$. Two consequences that are properties of the *measurement*, not the method:

$$\underline{p_A} \le \alpha^{1/n} \quad\Longrightarrow\quad \hat R \le \sigma\,\Phi^{-1}(\alpha^{1/n}),$$

so the radius is capped at roughly $3.8\sigma$ for Cohen's standard $n=10^5$, $\alpha=10^{-3}$; and the guarantee is **probabilistic** — it holds with probability $1-\alpha$ over the certification randomness, so a $10^{-3}$-level certificate applied to 50,000 images is expected to be wrong on $\sim 50$ of them.

**Norm conversion.** $\|\delta\|_\infty \le \|\delta\|_2 \le \sqrt{d}\,\|\delta\|_\infty$, so an $\ell_2$ certificate of $R$ yields only $\ell_\infty$ radius $R/\sqrt{d}$; at ImageNet $d = 3\times224^2 = 150{,}528$, $\sqrt{d} \approx 388$.

**Deterministic verification.** For a ReLU network, verify $\min_{\|\delta\|_p \le \epsilon} \big(z_y(x+\delta) - \max_{c\ne y} z_c(x+\delta)\big) > 0$. Exact solution is NP-complete (Katz et al., CAV 2017). Practical verifiers solve a relaxation (interval bound propagation, CROWN linear bounds, branch-and-bound), so *incompleteness* is the measured quantity: a "not verified" result mixes genuine vulnerability with relaxation slack.

**Assumptions known violated in practice.** (i) i.i.d. Monte Carlo samples — implementations reuse a batched RNG stream and often reuse the selection sample for estimation; (ii) exact arithmetic — floating-point bound propagation is unsound unless intervals are directionally rounded, and most verifiers do not; (iii) the $\ell_p$ ball is the threat model — it is not, for any deployed system; (iv) $g$, not $F$, is deployed — practitioners often report $F$'s clean accuracy alongside $g$'s certificates.

## 3. State of the Art

**Established (reproduced, ablated).**
- Gaussian smoothing with a noise-trained base classifier: 49% certified top-1 on ImageNet at $\ell_2 = 0.5$ (Cohen et al., ICML 2019). Adversarial training of the smoothed classifier (SmoothAdv, Salman et al., NeurIPS 2019) and consistency/MACER regularizers (Zhai et al., ICLR 2020; Jeong & Shin, NeurIPS 2020) improve this by 5–15 points depending on radius.
- Denoised smoothing with an off-the-shelf diffusion model plus a strong classifier: 71% certified top-1 on ImageNet at $\ell_2=0.5$ (Carlini et al., ICLR 2023). Independently reproduced and extended (DensePure, Xiao et al., ICLR 2023). This is the current $\ell_2$ SOTA and it uses **no** robust training.
- Deterministic $\ell_\infty$ on CIFAR-10 at $\epsilon=8/255$: certified accuracy in the 35–39% band with clean accuracy 48–55%, via IBP-family training (Shi et al., NeurIPS 2021: 34.6% certified / 48.9% clean; SABR, Müller et al., ICLR 2023: ~35% / ~52%; MTL-IBP, De Palma et al., ICLR 2024, at the top of the band). These are benchmark numbers on one dataset at one $\epsilon$.
- $\alpha,\beta$-CROWN (Wang et al., NeurIPS 2021) has won VNN-COMP repeatedly. Its scale ceiling is networks of $\sim10^5$–$10^7$ parameters; it does not run on ImageNet-scale backbones.

**Claimed but unablated.** Certification for semantic and non-$\ell_p$ threat models (rotation, brightness, patch, text substitution) is reported per-paper on bespoke benchmarks with no shared protocol; cross-paper numbers are not comparable. Claims that diffusion-based denoising "solves" certification rest entirely on $\ell_2$ radii $\le 3.0$ and do not transfer to $\ell_\infty$.

**Benchmark-number-only.** Most ImageNet certified accuracies are reported on a 500-image subsample of the validation set, not all 50,000, because certification costs $\sim10^5$ forward passes per image.

## 4. What Is Known

- $\sigma\Phi^{-1}(p_A)$ is **tight** for Gaussian smoothing: no better radius is extractable from $p_A$ alone (Cohen et al., 2019).
- The optimal smoothing distribution for a given norm is characterized: Gaussian for $\ell_2$, and Wulff-crystal-shaped measures generally (Yang et al., *Randomized Smoothing of All Shapes and Sizes*, ICML 2020). Discrete/uniform smoothing gives $\ell_0$ and $\ell_1$ certificates (Lee et al., NeurIPS 2019; Levine & Feizi).
- **$\ell_\infty$ barrier.** For i.i.d. noise, certified $\ell_\infty$ radius degrades as $\Omega(1/\sqrt{d})$ relative to $\ell_2$ unless clean accuracy drops sharply: Blum, Dick, Manoj & Zhang (JMLR 2020) and Kumar, Levine, Feizi & Goldstein (ICML 2020) both prove that certifying $\ell_\infty$ radius $\epsilon$ in dimension $d$ requires $\sigma \gtrsim \epsilon\sqrt{d}$, at which point the base classifier's accuracy under noise collapses. Two independent proofs, same conclusion.
- **Relaxation barrier.** Any verifier using the standard single-neuron convex relaxation has an accuracy ceiling that cannot be closed by better solving (Salman et al., *A Convex Relaxation Barrier to Tight Robustness Verification*, NeurIPS 2019).
- **Exact verification is NP-complete** for ReLU networks (Katz et al., CAV 2017; hardness also in Weng et al., ICML 2018).
- **Smoothing has side effects.** It shrinks decision regions unevenly across classes, degrading worst-class accuracy — the "hidden cost" (Mohapatra et al., AISTATS 2021).

## 5. What Is Not Known

- **Theoretically open.** Whether the $\Omega(\sqrt{d})$ $\ell_\infty$ barrier extends to *non-i.i.d.*, input-dependent, or learned smoothing measures. The existing proofs assume a fixed product measure. No lower bound rules out a data-dependent certification scheme achieving $\ell_\infty = 4/255$ on natural images.
- **Theoretically open.** Whether *any* polynomial-time sound certifier can beat the convex-relaxation ceiling on general networks, or whether that ceiling is intrinsic to architectures trained without a verification-friendly inductive bias.
- **Empirically open.** Whether IBP-family certified training scales to ImageNet. The experiment is runnable — it costs roughly one ImageNet training run per configuration — but no group has published a full sweep at $\epsilon = 4/255$.
- **Empirically open.** Whether certified accuracy at fixed $\epsilon$ improves with base-model scale. Denoised smoothing suggests yes for $\ell_2$; there is no scaling curve at all for deterministic $\ell_\infty$.
- **Methodologically blocked.** Certification for the threat models that motivate the field. There is no accepted definition of a sound certificate over "semantically equivalent images" or over token-level perturbations of an LLM prompt, because the perturbation set has no formal description. Papers substitute a parameterized family (rotation angle, brightness) and certify that instead — a different quantity than the one named.

## 6. Why It Is Hard

Three distinct obstructions.

1. **Finite-sample cap, not model quality.** The certified radius is bounded by $\sigma\Phi^{-1}(\alpha^{1/n})$ regardless of how good $F$ is. Doubling the radius requires roughly squaring $n$. This is a measurement limit, and it is why no smoothing paper reports radii above $\sim 4\sigma$.
2. **Dimensional conversion.** $\ell_2 \to \ell_\infty$ costs a factor of $\sqrt{d} \approx 388$ at ImageNet resolution, and the two proofs above show this loss is not an artifact of the analysis for i.i.d. noise. The barrier scales *against* progress: better sensors mean higher $d$ means worse certificates.
3. **Incompleteness confounds evaluation.** A deterministic verifier returning "unverified" does not distinguish a genuinely attackable input from relaxation slack. Comparing two verifiers on certified accuracy therefore compares a mixture of model robustness and solver tightness, and the mixing proportion is unmeasured. There is no ground-truth robust-accuracy oracle above toy scale, because computing it is NP-complete.

## 7. Current Research (as of 2026)

- **Diffusion-denoised smoothing.** Carlini/Tramèr and follow-ons; extension to higher $\sigma$ and to multi-step consistency denoisers. Established for $\ell_2$; no $\ell_\infty$ path.
- **Verification-aware training.** SABR / STAPS / MTL-IBP lineage (ETH Zurich SRI, Oxford). Direction: close the clean-accuracy gap so certified training is not a 25-point tax. *(frontier — verify: ImageNet-scale results.)*
- **Branch-and-bound at scale.** $\alpha,\beta$-CROWN group (CMU/UIUC/Northeastern), GPU-parallel bounding, VNN-COMP as the shared harness.
- **Beyond $\ell_p$.** Certified patch defenses (Levine & Feizi; Xiang et al.), semantic certification via parameterized transformation intervals, and early work on certified guarantees for LLM prompt perturbations via smoothing over token deletions. The last is *(frontier — verify)*: soundness depends on a perturbation set definition that is currently stipulated, not derived.
- **Floating-point-sound verification.** Small but real line of work on directed-rounding verifiers; most published certificates are not FP-sound.

## 8. Concrete Next Experiment

**Question.** Does certified $\ell_\infty$ accuracy at a fixed, meaningful $\epsilon$ improve with model scale, or is it flat?

**Setup.** Downscaled ImageNet ($64\times64$, $d=12{,}288$), full 50,000-image validation set. Train four CNNs at $2\times10^6$, $8\times10^6$, $3\times10^7$, $1.2\times10^8$ parameters with an identical IBP-family certified-training recipe (Shi et al. warmup schedule + SABR-style small-box propagation) at $\epsilon = 4/255$. Verify with $\alpha,\beta$-CROWN, 60 s/image budget, FP-sound rounding enabled.

**Control arm.** The same four architectures trained with standard cross-entropy and certified by Gaussian randomized smoothing at $\sigma = 4/255 \cdot \sqrt{d} \approx 1.73$ (the $\sigma$ needed for the $\ell_\infty$ radius), $n = 10^5$, $\alpha = 10^{-3}$.

**Deciding number.** Certified accuracy at $\epsilon = 4/255$, abstentions counted as errors, as a function of $\log(\text{params})$. A slope $\geq 5$ points per $4\times$ parameter increase over the full range says deterministic certification scales and the ImageNet run is worth its cost. A slope $\leq 1$ point says the ceiling is the relaxation, not the model, and effort belongs on tighter relaxations instead. The control arm is expected to be near 0% — that expectation being confirmed is itself the reusable result.

Cost: roughly 3,000 A100-hours training plus 800 GPU-hours verification.

## 9. Key References

- **[Foundational]** Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML 2019. — arXiv:1902.02918
- **[Foundational]** Guy Katz, Clark Barrett, David Dill, Kyle Julian, Mykel Kochenderfer. *Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.* CAV 2017. — arXiv:1702.01135
- **[Barrier]** Avrim Blum, Travis Dick, Naren Manoj, Hongyang Zhang. *Random Smoothing Might be Unable to Certify $\ell_\infty$ Robustness for High-Dimensional Images.* JMLR, 2020.
- **[Barrier]** Aounon Kumar, Alexander Levine, Soheil Feizi, Tom Goldstein. *Curse of Dimensionality on Randomized Smoothing for Certifiable Robustness.* ICML 2020.
- **[Barrier]** Hadi Salman, Greg Yang, Huan Zhang, Cho-Jui Hsieh, Pengchuan Zhang. *A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks.* NeurIPS 2019.
- **[Theory]** Greg Yang, Tony Duan, J. Edward Hu, Hadi Salman, Ilya Razenshteyn, Jerry Li. *Randomized Smoothing of All Shapes and Sizes.* ICML 2020.
- **[SOTA]** Nicholas Carlini, Florian Tramèr, Krishnamurthy Dvijotham, Leslie Rice, Mingjie Sun, J. Zico Kolter. *(Certified!!) Adversarial Robustness for Free!* ICLR 2023.
- **[SOTA]** Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, J. Zico Kolter. *Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification.* NeurIPS 2021.
- **[SOTA]** Mark Niklas Müller, Franziska Eckert, Marc Fischer, Martin Vechev. *Certified Training: Small Boxes are All You Need.* ICLR 2023.
- **[Method]** Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sébastien Bubeck, Greg Yang. *Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers.* NeurIPS 2019.
- **[Method]** Zhouxing Shi, Yihan Wang, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh. *Fast Certified Robust Training with Short Warmup.* NeurIPS 2021.
- **[Caveat]** Aounon Kumar, et al. / Mohapatra et al. *The Hidden Cost of Randomized Smoothing.* AISTATS 2021.
- **[Survey]** Linyi Li, Tao Xie, Bo Li. *SoK: Certified Robustness for Deep Neural Networks.* IEEE Symposium on Security and Privacy, 2023.

## 10. Worked Example

One ImageNet image, standard Cohen protocol: $\sigma = 0.5$, $n = 10^5$, $\alpha = 10^{-3}$, $d = 150{,}528$.

**Best case.** All $10^5$ noisy samples classify correctly. Clopper–Pearson gives

$$\underline{p_A} = \alpha^{1/n} = \exp(-6.9078/10^5) = 0.9999309,$$
$$\Phi^{-1}(0.9999309) = 3.81, \qquad \hat R_2 = 0.5 \times 3.81 = 1.91.$$

This is the *ceiling*: a perfect base classifier at this $\sigma$ and $n$ cannot certify further.

**Convert to $\ell_\infty$.**

$$\hat R_\infty = \frac{1.91}{\sqrt{150{,}528}} = \frac{1.91}{388} = 0.0049 = \frac{1.26}{255}.$$

The empirical benchmark is $4/255$. The best possible certificate is $3.2\times$ short.

**Cost of closing the gap by sampling.** To reach $\ell_\infty = 4/255$ at $\sigma = 0.5$ we need $R_2 = 4/255 \times 388 = 6.09$, i.e. $\Phi^{-1}(\underline{p_A}) = 12.2$, i.e. $\underline{p_A} = 1 - 1.6\times10^{-34}$. From $\alpha^{1/n} \ge \underline{p_A}$:

$$n \ge \frac{-\ln \alpha}{1-\underline{p_A}} = \frac{6.91}{1.6\times10^{-34}} \approx 4\times10^{34}\ \text{forward passes}.$$

At $10^4$ inferences/second that is $10^{23}$ years, per image.

**Cost of closing it by raising $\sigma$.** Take $\sigma = 1.73$ so that $R_2 = 4/255\cdot\sqrt d$ is reachable at modest $p_A$. Gaussian noise at $\sigma = 1.73$ on $[0,1]$-scaled pixels has standard deviation $1.7\times$ the full dynamic range of the image. Empirically, ImageNet top-1 under $\sigma = 1.0$ noise is already in the low teens; at $1.73$ the base classifier is near chance, so $p_A < 0.5$ and no certificate is issued at all. This is exactly the trade the Blum et al. and Kumar et al. theorems formalize.

**What the example shows.** The gap between certified and empirical threat scales is not a modeling deficit that a better backbone closes. It is the product of two hard multipliers — a $\sqrt{d} = 388$ norm conversion and a Monte Carlo radius cap at $\sim 3.8\sigma$ — and both are properties of the certification procedure, not of the network being certified. Any solution must change the procedure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*