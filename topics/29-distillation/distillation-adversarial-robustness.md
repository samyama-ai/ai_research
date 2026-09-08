---
id: 29-distillation/distillation-adversarial-robustness
title: "Distillation of Adversarial Robustness"
topic: 29-distillation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distillation of Adversarial Robustness

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/distillation-adversarial-robustness` · **Status:** empirically-open

## 1. Problem Statement

Given a robust teacher $T$ — a network with certified or empirically measured accuracy under a bounded adversary — and a student architecture $S$ with a fraction of the teacher's parameters and FLOPs, can the teacher's robustness be transferred to the student at a cost far below training the student adversarially from scratch?

Three variants, routinely conflated:

- **Measurement.** Does a given distillation procedure produce a student whose robustness survives an adaptive attack, not just the attack used during training? Broken defenses in this area are almost always measurement failures (§3).
- **Method.** Find a distillation loss and attack schedule that closes the robustness gap between a distilled student and an adversarially-trained-from-scratch student of the same architecture, at lower total compute.
- **Theory.** Is robustness a *distillable* property at all? Standard distillation transfers a function's values on a data manifold; robustness is a property of the function on $\varepsilon$-balls off the manifold, whose volume grows exponentially in input dimension. No theorem says finite soft-label supervision determines it.

Solving it means: a student with $\le 1/10$ the teacher's inference FLOPs, within 1 point of AutoAttack accuracy of the same student adversarially trained from scratch, at $\le 1/3$ the training compute, reproduced on two datasets by an independent group.

## 2. Formal Setting

Data $(x,y)\sim\mathcal{D}$ on $\mathcal{X}\subseteq[0,1]^d$, labels $[K]$. Threat model $\mathcal{B}(x)=\{x': \|x'-x\|_p\le\varepsilon\}$; canonical CIFAR-10 setting $p=\infty$, $\varepsilon=8/255$.

Robust risk of classifier $f$:
$$R_{\mathrm{rob}}(f)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[\max_{x'\in\mathcal{B}(x)}\mathbf{1}\{f(x')\neq y\}\Big].$$

**As measured:** the inner max is intractable, so the reported quantity is
$$\widehat{R}_{\mathcal{A}}(f)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\{f(\mathcal{A}(f,x_i,y_i))\neq y_i\},$$
an *upper bound on robust accuracy* for attack ensemble $\mathcal{A}$. The de facto standard is AutoAttack (APGD-CE, APGD-T, FAB-T, Square; Croce & Hein, ICML 2020) on the full CIFAR-10 test set, $n=10{,}000$. Any number from PGD-20 alone is not a robustness measurement, it is a lower-bounded guess.

Distillation objective, in the general form covering ARD/RSLAD/IAD/AdaAD:
$$\min_{\theta}\ \mathbb{E}\Big[(1-\lambda)\,\ell_{\mathrm{CE}}\big(S_\theta(x),y\big)+\lambda\,\tau^{2}\,\mathrm{KL}\big(\sigma(T(\tilde x)/\tau)\,\|\,\sigma(S_\theta(x^{\mathrm{adv}})/\tau)\big)\Big],$$
where $x^{\mathrm{adv}}=\arg\max_{x'\in\mathcal{B}(x)}\mathcal{L}_{\mathrm{inner}}(S_\theta,x',\cdot)$ is generated against the **student**, $\tilde x\in\{x,x^{\mathrm{adv}}\}$ selects whether the teacher is queried on clean or perturbed inputs, and $\tau$ is temperature. Methods differ in $\tilde x$, in $\mathcal{L}_{\mathrm{inner}}$ (CE vs. KL-to-teacher), and in whether $\lambda$ is fixed or a per-sample function of teacher confidence.

**Compute, as measured:** $C=(\text{steps})\times(\text{fwd+bwd cost})$. Adversarial training with $k$-step PGD costs $\approx(k{+}1)\times$ standard training on the student. Distillation adds $\ge 1$ teacher forward per step, and $k$ teacher forwards if the teacher is queried on each inner-loop iterate — so "cheap distillation" claims must state which.

**Assumptions, and which fail.**
1. *The teacher's robustness is real.* Holds only for RobustBench-verified teachers; violated whenever a teacher's number comes from a weak attack.
2. *Soft labels encode the teacher's local decision geometry.* Violated: a $K$-vector at one point cannot determine behaviour over a $d$-dimensional ball, $d=3072$ for CIFAR-10.
3. *Student and teacher share a loss surface geometry the inner max can exploit.* Violated across architecture families (CNN teacher, ViT student) — gradient alignment is low and unmeasured in most papers.
4. *Test distribution matches training.* Violated by the synthetic-data regimes (§4) that produce the strongest teachers.

## 3. State of the Art

**Established.**
- *Defensive distillation is broken.* Papernot et al. (IEEE S&P 2016) distilled at high temperature and reported near-zero attack success; Carlini & Wagner (IEEE S&P 2017) reduced it to 0% robust accuracy. The mechanism was gradient masking, later generalized by Athalye et al. (ICML 2018). This is settled and is the reason every claim here must be adaptively attacked.
- *ARD* (Goldblum et al., AAAI 2020) established that a robust teacher plus student-generated attacks beats adversarial training alone for small students.
- *RSLAD* (Zi et al., ICCV 2021): use the teacher's **robust soft labels** for both the inner maximization and the outer loss. Reported ResNet-18 on CIFAR-10: ~83.4% clean, ~51.5% AutoAttack, from a WideResNet-34-10 TRADES teacher (~53.1% AA on RobustBench). This is the canonical result and has been reproduced by follow-on work.
- *IAD* (Zhu et al., ICLR 2022): teachers are unreliable on adversarial inputs; down-weight teacher supervision where teacher confidence on $x^{\mathrm{adv}}$ is low.
- *AdaAD* (Huang et al., CVPR 2023): adaptively search the inner maximum by maximizing student–teacher prediction discrepancy rather than CE.

**Claimed but unablated.**
- That the teacher supplies robustness rather than a *label-smoothing / regularization* effect. Almost no paper runs the control arm of a non-robust teacher of the same capacity plus matched label smoothing at matched compute.
- That gains hold under distribution shift or at ImageNet scale. Most numbers are CIFAR-10/100, ResNet-18 or MobileNetV2 students.
- Cross-architecture (CNN→ViT) robustness distillation results exist mainly as single benchmark rows, without adaptive-attack audits.

**Benchmark-number-only.** Every "+2 points AA over RSLAD" claim in the 2022–2025 literature rests on one seed, one teacher, one $\varepsilon$. Seed-to-seed AA variance on CIFAR-10 ResNet-18 is roughly 0.3–0.6 points; many reported margins are inside it.

## 4. What Is Known

- **Robustness costs capacity.** Madry et al. (ICLR 2018) showed robust training needs substantially larger models than standard training; the effect is monotone in width on CIFAR-10 at $\varepsilon=8/255$.
- **The frontier is data-bound.** Schmidt et al. (NeurIPS 2018) proved a sample-complexity separation for robust generalization in a Gaussian model. Empirically: Gowal et al. (2020) reached ~65.9% AA on CIFAR-10 with WRN-70-16 plus extra data; Wang et al. (ICML 2023) reached ~70.7% AA / ~93.3% clean with diffusion-generated data — the largest single jump in the benchmark's history, and it came from data, not from architecture or loss.
- **Teacher robustness caps student robustness in practice.** Across published ARD/RSLAD/IAD tables, no small student exceeds its teacher's AA number.
- **Distillation students beat scratch adversarial training at small scale.** ResNet-18 CIFAR-10: TRADES-from-scratch ≈ 49% AA; RSLAD ≈ 51.5% AA. A 2–3 point gap, measured with AutoAttack at $n=10{,}000$.
- **Non-robust features transfer.** Ilyas et al. (NeurIPS 2019) showed a model trained on a "robustified" dataset inherits robustness with no adversarial training — evidence that robustness can move through *data* even when it does not move through logits.

## 5. What Is Not Known

- **Theoretically open.** No result characterizes when $\varepsilon$-ball behaviour is identifiable from finite soft-label supervision at clean points. No lower bound of the form "any student of capacity $c$ distilled from a teacher of robust risk $r$ has robust risk $\ge g(c,r)$".
- **Empirically open.** (a) Does robustness distillation help at ImageNet-1k scale with a modern robust teacher? The run is affordable (order $10^3$–$10^4$ A100-hours) and has not been done as a controlled comparison. (b) Does the teacher contribute robustness beyond regularization? The non-robust-teacher control arm is runnable in days on CIFAR-10 and is essentially absent from the literature. (c) Cross-family transfer (robust CNN → ViT student) under AutoAttack.
- **Methodologically blocked.** "Robustness transferred" has no accepted metric beyond a single AA scalar. AA accuracy conflates decision-boundary margin, gradient obfuscation residue, and clean accuracy. There is no standard, attack-independent measure of how much of the *teacher's* robust geometry the student inherited.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a self-referential attack loop**. The inner maximization is run against the student, whose loss surface distillation is actively smoothing. Smoothing the surface lowers attack success without necessarily enlarging the margin — exactly the failure mode of 2016 defensive distillation. AutoAttack's Square component (gradient-free) catches gross masking, but partial masking that shifts AA's effective strength by 1–2 points is inside the range of every headline improvement in this subfield. So the field cannot currently tell a 2-point method gain from a 2-point measurement artifact.

Secondary: **absent ground truth.** $R_{\mathrm{rob}}$ is not computable at these scales; certified bounds (IBP, randomized smoothing) sit far below empirical numbers and use a different threat model, so they cannot arbitrate.

## 7. Current Research (as of 2026)

- Adaptive/discrepancy-driven inner maximization, post-AdaAD — improving where the student is attacked rather than what it is told *(frontier — verify)*.
- Distilling from diffusion-data-trained teachers (Wang et al. line), asking whether the data advantage survives compression *(frontier — verify)*.
- Robustness transfer to ViT and hybrid students; groups at EPFL/Tübingen (Croce, Hein lineage) maintain the evaluation infrastructure (RobustBench) that any such claim must pass.
- Robust fine-tuning of foundation-model backbones as an alternative to distillation — if a robust CLIP-scale backbone fine-tunes robustly, the distillation route becomes economically uninteresting.

## 8. Concrete Next Experiment

**The teacher-necessity ablation, at CIFAR-10 scale, with a matched-compute control.**

- **Scale.** CIFAR-10, $\varepsilon_\infty=8/255$. Student ResNet-18 (11.2M params). Teacher WRN-70-16 from Wang et al. (2023), AA ≈ 70.7%. 5 seeds per arm. Roughly 200–400 GPU-hours total on A100s.
- **Arms** (all at *equal wall-clock student compute*, teacher forwards counted):
  1. RSLAD from the robust teacher.
  2. **Control A:** same distillation loss, *non-robust* WRN-70-16 teacher (standard training, ~96% clean).
  3. **Control B:** no teacher — TRADES from scratch with label smoothing tuned to match arm 1's mean logit entropy.
  4. **Control C:** arm 1's compute spent entirely on more TRADES epochs.
- **Deciding number.** AutoAttack accuracy on the full 10,000-image test set, reported as mean ± std over 5 seeds. The question is settled if $\mathrm{AA}(1)-\max\{\mathrm{AA}(2),\mathrm{AA}(3),\mathrm{AA}(4)\} > 1.0$ point with non-overlapping std bands. If it is not, the robust teacher is not the active ingredient and the subfield's premise fails at its best-studied scale.
- **Mandatory audit.** Report AA component-wise; if Square accuracy is more than 3 points below APGD-T accuracy in any arm, that arm is masking gradients and its number is void.

## 9. Key References

- **[Foundational]** Papernot, McDaniel, Wu, Jha, Swami. *Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks.* IEEE S&P, 2016. — arXiv:1511.04508
- **[Foundational]** Carlini, Wagner. *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P, 2017. — arXiv:1608.04644
- **[Foundational]** Madry, Makelov, Schmidt, Tsipras, Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR, 2018. — arXiv:1706.06083
- **[Foundational]** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Method]** Goldblum, Fowl, Feizi, Goldstein. *Adversarially Robust Distillation.* AAAI, 2020. — arXiv:1905.09747
- **[SOTA]** Zi, Zhu, Ma, Wang, Xia. *Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better.* ICCV, 2021.
- **[SOTA]** Zhu, Yao, Han, Liu, Zhou, Niu, Tsang, Yang. *Reliable Adversarial Distillation with Unreliable Teachers.* ICLR, 2022.
- **[SOTA]** Huang, Chen, Li, Wang, Zhang, Wang. *Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation.* CVPR, 2023.
- **[Evaluation]** Croce, Hein. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks.* ICML, 2020. — arXiv:2003.01690
- **[Evaluation]** Croce, Andriushchenko, Sehwag, Debenedetti, Flammarion, Chiang, Mittal, Hein. *RobustBench: a standardized adversarial robustness benchmark.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2010.09670
- **[Evaluation]** Tramèr, Carlini, Brendel, Madry. *On Adaptive Attacks to Adversarial Example Defenses.* NeurIPS, 2020. — arXiv:2002.08347
- **[Context]** Zhang, Yu, Jiao, Xing, El Ghaoui, Jordan. *Theoretically Principled Trade-off between Robustness and Accuracy.* ICML, 2019. — arXiv:1901.08573
- **[Context]** Schmidt, Santurkar, Tsipras, Talwar, Mądry. *Adversarially Robust Generalization Requires More Data.* NeurIPS, 2018. — arXiv:1804.11285
- **[Context]** Wang, Pang, Du, Lin, Liu, Yan. *Better Diffusion Models Further Improve Adversarial Training.* ICML, 2023. — arXiv:2302.04638
- **[Context]** Ilyas, Santurkar, Tsipras, Engstrom, Tran, Mądry. *Adversarial Examples Are Not Bugs, They Are Features.* NeurIPS, 2019. — arXiv:1905.02175

## 10. Worked Example

Teacher: WRN-70-16, CIFAR-10, ~70.7% AA. Student: ResNet-18, ~11.2M params, roughly $30\times$ fewer FLOPs.

Budget accounting, 200 epochs, 10-step PGD inner loop, student forward+backward cost normalized to 1:
- TRADES from scratch: $200\times 11\times 1 = 2200$ units.
- RSLAD with teacher queried once per step: $2200 + 200\times 1\times 8 \approx 3800$ units (teacher forward $\approx 8\times$ student forward).
- RSLAD with teacher queried on every inner iterate: $2200 + 200\times 10\times 8 = 18{,}200$ units — **8× the scratch baseline**.

Outcome, from published tables: distilled ResNet-18 lands near 51.5% AA; scratch TRADES ResNet-18 near 49%. The gap is ~2.5 points.

Now make the obstruction visible. Ask what a 2.5-point gap buys per unit of compute. Arm 4 of §8 — spending the same 3800 units on ~345 epochs of plain TRADES — is never reported, and longer adversarial training on CIFAR-10 is known to move AA by roughly 0.5–1.5 points before robust overfitting sets in. Arm 3 — label smoothing at matched logit entropy — is known to move AA by a similar order. Two unrun controls, each plausibly worth 1 point, against a 2.5-point claimed effect with per-seed noise of 0.3–0.6 points.

The teacher may be contributing 2.5 points, or 0.5 points plus regularization, or nothing beyond a smoother surface that AutoAttack partially fails to penetrate. The published numbers cannot distinguish these three worlds. That is why the status is *empirically open* and not *partially solved*: the decisive experiment is cheap, well-defined, and has not been run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*