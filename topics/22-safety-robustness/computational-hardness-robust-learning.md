---
id: 22-safety-robustness/computational-hardness-robust-learning
title: "Computational Hardness of Robust Learning"
topic: 22-safety-robustness
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Computational Hardness of Robust Learning

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/computational-hardness-robust-learning` · **Status:** partially-solved

## 1. Problem Statement

Given a concept class and a data distribution for which accurate classification is *easy* (learnable in polynomial time and samples), is classification that remains accurate under a bounded adversarial perturbation also easy? Or is there an intrinsic computational price — a separation where a robust classifier provably exists but no efficient algorithm can find it?

Three variants, routinely conflated:

- **Theory variant.** Does there exist a concept class $\mathcal{C}$, distribution family $\mathcal{D}$, and perturbation budget $\epsilon$ such that $\mathcal{C}$ is efficiently PAC-learnable over $\mathcal{D}$ but not efficiently *robustly* learnable, under standard complexity or cryptographic assumptions? **Largely settled affirmatively** (§4) — hence status `partially-solved`.
- **Method variant.** Is the observed 20–30 point robust-accuracy deficit of real networks a consequence of that hardness, or of insufficient data, capacity, or optimization? Open.
- **Measurement variant.** Given a trained model, compute its true robust error $R_\epsilon(f)$. Exact computation is NP-hard for ReLU nets; every reported number is either an attack-based upper bound on robustness or a verifier-based lower bound, and the two disagree by tens of points.

Solving the problem means: an unconditional or standard-assumption separation *that predicts the empirical gap*, plus an experiment distinguishing "hard" from "under-resourced".

## 2. Formal Setting

Instance space $\mathcal{X} \subseteq \mathbb{R}^d$, labels $\mathcal{Y}=\{\pm1\}$, distribution $D$ over $\mathcal{X}\times\mathcal{Y}$. Perturbation set $\mathcal{U}(x) = \{x' : \|x'-x\|_p \le \epsilon\}$.

**Robust risk.**
$$R_\epsilon(f) \;=\; \mathbb{E}_{(x,y)\sim D}\Big[\sup_{x'\in\mathcal{U}(x)} \mathbb{1}\{f(x')\neq y\}\Big].$$

**As measured.** The $\sup$ is never computed. Two estimators are reported:
$$\hat R^{\mathrm{atk}}_\epsilon = \frac{1}{n}\sum_i \mathbb{1}\{\exists\, x'\in A(x_i): f(x')\neq y_i\}, \qquad \hat R^{\mathrm{cert}}_\epsilon = \frac{1}{n}\sum_i \mathbb{1}\{V(f,x_i,\epsilon)=\text{unverified}\},$$
with $A$ a finite attack (AutoAttack: 100-step APGD-CE + APGD-T + FAB-T + Square, $5\times10^3$ queries) and $V$ an incomplete verifier (bound propagation / branch-and-bound with a time budget, typically 100–1000 s per input). Always $\hat R^{\mathrm{atk}}_\epsilon \le R_\epsilon(f) \le \hat R^{\mathrm{cert}}_\epsilon$. The interval width is the measurement error, and it is not small (§10).

**Robust learnability.** $\mathcal{C}$ is efficiently robustly learnable over $\mathcal{D}$ if some algorithm running in $\mathrm{poly}(d,1/\delta,1/\gamma)$ time outputs $f$ with $R_\epsilon(f) \le \min_{c\in\mathcal{C}} R_\epsilon(c) + \gamma$ w.p. $1-\delta$, for all $D\in\mathcal{D}$ with $\inf_{c} R_\epsilon(c)=0$ (realizable case).

**Compute budget.** Adversarial training with $k$-step PGD costs $(k{+}1)\times$ the forward/backward work of standard ERM; the empirical SOTA arm below is $\approx 10^{19}$–$10^{20}$ FLOPs for CIFAR-10 including synthetic-data generation.

**Assumptions, and which are violated.**
1. *The threat model is the $\ell_p$ ball.* Violated: real threats (patches, spatial transforms, semantic edits) are not $\ell_p$-bounded, so $R_\epsilon$ names "robustness" while measuring one ball.
2. *Realizability* — some $c\in\mathcal{C}$ has zero robust risk. Violated for images: classes at $\epsilon=8/255$ are not perfectly robustly separable in the Bayes sense, and no ground-truth robust labeling function is available.
3. *i.i.d. sampling.* Holds for benchmarks, not for deployment.
4. *Proper learning.* Violated by the fix — the known positive result is improper (§4), so the output is not in $\mathcal{C}$ and is not interpretable as a member of the hypothesis class.

## 3. State of the Art

**Theory SOTA (established).**
- Bubeck, Price, Razenshteyn (ICML 2019) and Bubeck, Lee, Price, Razenshteyn (COLT 2019): a distribution efficiently learnable in the statistical-query model non-robustly, for which robust learning needs exponentially many SQs; plus a cryptographic separation.
- Degwekar, Nakkiran, Vaikuntanathan (COLT 2019): assuming standard cryptography, tasks exist where a robust classifier is in the class but no polynomial-time learner finds it — with a "win-win": if you can break robustness you get a cryptographic primitive.
- Garg, Jha, Mahloujifar, Mahmoody (ALT 2020): hardness of robust learning under computationally bounded adversaries, and a converse — hardness can be *used* defensively.
- Montasser, Hanneke, Srebro (COLT 2019): every class of finite VC dimension is robustly PAC-learnable, but only **improperly**; proper robust learning of some finite-VC classes is impossible.
- Gourdeau, Kanade, Kwiatkowska, Worrell (NeurIPS 2019; JMLR 2021): monotone conjunctions are not robustly learnable under log-Lipschitz distributions once the robustness radius exceeds $\omega(\log n)$, while being trivially learnable non-robustly.

**Empirical SOTA (benchmark numbers only).** RobustBench CIFAR-10 $\ell_\infty$, $\epsilon=8/255$: Peng et al. (2023) 93.27% clean / 71.07% AutoAttack robust; Wang et al. (ICML 2023, diffusion-generated data) 93.25% / 70.69%. ImageNet $\ell_\infty$, $\epsilon=4/255$: Liu et al. (2023, Swin-L) 78.92% clean / 59.56% robust. These are leaderboard entries against a fixed attack ensemble, not certified quantities.

**Claimed but unablated.** That robustness gains from synthetic data reflect a genuine sample-complexity fix rather than distributional overlap with the test set — the diffusion model is trained on the same CIFAR-10 training split, and no ablation isolates novel-information gain from memorized re-sampling. Also unablated: whether the residual 29-point gap on CIFAR-10 is attributable to any hardness result, none of which applies to convolutional nets on images.

## 4. What Is Known

- **Separation exists.** Efficient non-robust learnability does not imply efficient robust learnability, under standard cryptographic assumptions (Degwekar et al. 2019) and unconditionally in the SQ model (Bubeck et al. 2019).
- **Statistical hardness is distinct from computational hardness.** Schmidt et al. (NeurIPS 2018): in a $d$-dimensional Gaussian mixture, non-robust learning needs $O(1)$ samples while $\ell_\infty$-robust learning needs $\Omega(\sqrt{d})$ — a purely information-theoretic gap.
- **Improper learning suffices.** Finite VC dimension $\Rightarrow$ robust PAC learnability improperly, with sample complexity $\tilde O(\mathrm{VC}(\mathcal{C})\cdot \mathrm{VC}^*(\mathcal{C})/\gamma)$ (Montasser et al. 2019).
- **Verification is NP-hard.** Exact $\ell_\infty$ robustness checking for ReLU networks is NP-complete (Katz et al., CAV 2017; Weng et al., ICML 2018).
- **A capacity law.** Bubeck & Sellke (NeurIPS 2021): fitting $n$ points from an isoperimetric distribution with an $O(1)$-Lipschitz network requires $\approx nd$ parameters — smooth interpolation costs a factor $d$ over the $n$ parameters needed for plain interpolation.
- **Robustness costs clean accuracy at current scale.** CIFAR-10: 99.5% clean for a non-robust WideResNet vs 93.3% clean for the best robust model — a 6-point drop, measured at $\sim3\times10^7$ parameters with 1M–50M synthetic images.
- **Adversarial training is the only reliable empirical method.** PGD adversarial training (Madry et al., ICLR 2018) and randomized smoothing (Cohen, Rosenfeld, Kolter, ICML 2019; 49% certified accuracy at $\ell_2$ radius 0.5 on ImageNet) survived the 2018–2020 wave of broken defenses; nearly everything else fell to adaptive attacks (Athalye et al., ICML 2018; Tramèr et al., NeurIPS 2020).

## 5. What Is Not Known

- **Theoretically open.** Whether any hardness result applies to *natural* image distributions and neural hypothesis classes. All separations use engineered distributions (pseudorandom function constructions, log-Lipschitz product measures). No theorem lower-bounds robust learning for, say, convolutional nets on a distribution with realistic image statistics.
- **Theoretically open.** Whether efficient *proper* robust learning is possible for specific natural classes (halfspaces under Gaussians beyond known regimes), and whether the improper Montasser learner has any polynomial-time implementation.
- **Empirically open.** Whether the CIFAR-10 gap closes with scale. Nobody has run adversarial training at $10^{22}$+ FLOPs with $\ge 10^9$ genuinely novel images, so "hard" and "under-resourced" remain observationally equivalent.
- **Methodologically blocked.** The true robust error of any deployed model. $\hat R^{\mathrm{atk}}$ and $\hat R^{\mathrm{cert}}$ bracket it with a 30+ point interval on CIFAR-10 at $\epsilon=8/255$; there is no estimator of $R_\epsilon$ itself.
- **Methodologically blocked.** Robustness outside $\ell_p$ balls: no ground-truth definition of the perturbation set for semantic threats, so nothing can be scored.

## 6. Why It Is Hard

The central obstruction is **non-identifiability of the measurement**. A robust-accuracy number is a bound of unknown tightness from two directions: a stronger attack lowers $\hat R^{\mathrm{atk}}$, a better verifier raises the certified floor, and no experiment currently separates "the model is robust" from "the attack is weak". Every headline robustness claim is therefore conditional on the attack suite, and history says attacks improve — Athalye et al. (2018) broke 7 of 9 ICLR-2018 defenses; Tramèr et al. (2020) broke 13 more.

Second: the theory measures a different object than the practice. The separations are statements about worst-case concept classes; the benchmark is a fixed dataset with no known member of any concept class as its labeling function. There is no ground truth for "the robust Bayes classifier on CIFAR-10", so the theory cannot be falsified by the benchmark and the benchmark cannot be explained by the theory.

Third: compute. PGD-10 adversarial training is $\approx 11\times$ ERM per step, and the SOTA arms add diffusion sampling of $10^6$–$10^8$ images. The scaling experiment that would decide the empirical variant costs roughly two orders of magnitude more than any published robust-training run.

## 7. Current Research (as of 2026)

- **Robust learning theory.** Montasser, Hanneke, Srebro and collaborators on sample-complexity and improper-learning structure; Gourdeau, Kanade, Kwiatkowska, Worrell (Oxford) on distributional assumptions and query models that restore tractability (e.g. local membership queries, NeurIPS 2022).
- **Cryptographic separations.** Mahloujifar and Mahmoody's line on computationally bounded adversaries and defensive uses of hardness.
- **Verification.** The annual VNN-COMP community (α,β-CROWN and successors) pushing complete branch-and-bound to larger nets; scaling remains the binding constraint. *(frontier — verify current-year winners.)*
- **Data-scaling defenses.** Diffusion-generated training data for adversarial training (Wang et al. 2023 and follow-ups), now the dominant CIFAR-10 recipe.
- **LLM robustness.** Jailbreak-resistance framed as robust learning over discrete token perturbations, where the perturbation set is defined by a judge model rather than a norm ball — this makes $R_\epsilon$ *undefined* rather than merely hard. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Is the CIFAR-10 robust-accuracy deficit compute/data-limited or intrinsic?

**Scale.** Adversarial-train a fixed architecture family (WideResNet-70-16, $2.7\times10^8$ params) with PGD-10 at $\epsilon=8/255$ across a 4-point data ladder using **CIFAR-5m** (6M real CIFAR-like images from Tiny-Images, Nakkiran et al. 2021) — not diffusion samples, to avoid the memorization confound: $n \in \{5\times10^4, 5\times10^5, 2\times10^6, 6\times10^6\}$. Total $\approx 5\times10^{20}$ FLOPs.

**Control arm.** Identical ladder with standard (non-robust) ERM training, same architecture, same $n$, same schedule. This isolates the *robustness* scaling exponent from the generic data-scaling exponent.

**Deciding number.** Fit $\hat R^{\mathrm{atk}}_\epsilon(n) = a\,n^{-\alpha_{\mathrm{rob}}} + R_\infty$ and the analogous curve for clean error, exponent $\alpha_{\mathrm{std}}$. **The decision statistic is $R_\infty$, the fitted irreducible robust error, with a bootstrap 95% CI.** If $R_\infty \le 0.05$ (CI upper bound), the gap is data-limited and no hardness explanation is needed. If $R_\infty \ge 0.20$ with $\alpha_{\mathrm{rob}} < \alpha_{\mathrm{std}}$, the deficit persists under a 120× data increase and the intrinsic-hardness hypothesis survives its first real test.

**Cost of not doing it:** every current claim about "robustness requires more data" rests on a 2-point extrapolation.

## 9. Key References

- **[Foundational]** A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR, 2018. — arXiv:1706.06083
- **[Foundational]** S. Bubeck, E. Price, I. Razenshteyn. *Adversarial Examples from Computational Constraints.* ICML, 2019. — arXiv:1805.10204
- **[Foundational]** A. Degwekar, P. Nakkiran, V. Vaikuntanathan. *Computational Limitations in Robust Classification and Win-Win Results.* COLT, 2019. — arXiv:1902.01086
- **[Foundational]** O. Montasser, S. Hanneke, N. Srebro. *VC Classes are Adversarially Robustly Learnable, but Only Improperly.* COLT, 2019. — arXiv:1902.04217
- **[Foundational]** P. Gourdeau, V. Kanade, M. Kwiatkowska, J. Worrell. *On the Hardness of Robust Classification.* NeurIPS, 2019; JMLR, 2021.
- **[Foundational]** L. Schmidt, S. Santurkar, D. Tsipras, K. Talwar, A. Madry. *Adversarially Robust Generalization Requires More Data.* NeurIPS, 2018. — arXiv:1804.11285
- **[Theory SOTA]** S. Bubeck, M. Sellke. *A Universal Law of Robustness via Isoperimetry.* NeurIPS, 2021 (Outstanding Paper). — arXiv:2105.12806
- **[Theory]** S. Garg, S. Jha, S. Mahloujifar, M. Mahmoody. *Adversarially Robust Learning Could Leverage Computational Hardness.* ALT, 2020.
- **[Verification]** G. Katz, C. Barrett, D. Dill, K. Julian, M. Kochenderfer. *Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.* CAV, 2017. — arXiv:1702.01135
- **[Empirical SOTA]** Z. Wang, T. Pang, C. Du, M. Lin, W. Liu, S. Yan. *Better Diffusion Models Further Improve Adversarial Training.* ICML, 2023. — arXiv:2302.04638
- **[Benchmark]** F. Croce, M. Andriushchenko, V. Sehwag, E. Debenedetti, N. Flammarion, M. Chiang, P. Mittal, M. Hein. *RobustBench: A Standardized Adversarial Robustness Benchmark.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2010.09670
- **[Certification]** J. Cohen, E. Rosenfeld, J. Z. Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML, 2019. — arXiv:1902.02918
- **[Survey / cautionary]** F. Tramèr, N. Carlini, W. Brendel, A. Madry. *On Adaptive Attacks to Adversarial Example Defenses.* NeurIPS, 2020. — arXiv:2002.08347

## 10. Worked Example

Take CIFAR-10, $\ell_\infty$, $\epsilon=8/255$, and carry one model's robustness measurement end to end.

**Upper bound on robustness (attack).** Wang et al. (2023), WRN-70-16 with diffusion data: AutoAttack robust accuracy 70.69%, i.e. $\hat R^{\mathrm{atk}}_\epsilon = 0.293$. This is a *lower* bound on the true robust error.

**Lower bound on robustness (certification).** The best *certified* CIFAR-10 accuracy at the same $\epsilon$ is far lower — certified-training work in the fast-IBP line (Shi et al., NeurIPS 2021) reports $\approx 34.6\%$ verified accuracy at 48.9% clean, i.e. $\hat R^{\mathrm{cert}}_\epsilon \approx 0.65$ for a model that is itself much weaker. Even running a complete branch-and-bound verifier on the *adversarially trained* WRN-70-16 does not close this: the network is not Lipschitz-regularized, bound propagation blows up, and per-input timeouts dominate.

**The interval.** For the class of models we actually deploy, the true robust error satisfies
$$0.29 \;\le\; R_\epsilon(f) \;\le\; 1.00 \quad\text{(no non-trivial certificate at }\epsilon=8/255\text{ for this architecture)}.$$
Restricting to models we *can* certify, the interval is $0.65 \le R_\epsilon \le 0.65$ — tight, but at 48.9% clean accuracy, which is unusable.

**What the obstruction looks like.** Suppose a new attack drops the WRN-70-16 to 55% robust accuracy tomorrow. Nothing in the training pipeline changed; the model's $R_\epsilon$ was always what it was. The 15-point move is entirely measurement. Now ask the paper's question — "does hardness explain the gap?" — with $R_\epsilon$ known only to $\pm 0.35$. It cannot be asked. The theory says robust learning can be exponentially harder than learning for *some* distribution; the benchmark says a number that is an artifact of the current attack suite for *this* distribution. Nothing connects them, and the connection cannot be built until either (a) complete verification scales to $10^8$-parameter nets at $\epsilon=8/255$, or (b) the scaling experiment in §8 produces an $R_\infty$ that is robust to attack-suite version, because a fitted asymptote is far less sensitive to a uniform shift in the estimator than a single point is.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*