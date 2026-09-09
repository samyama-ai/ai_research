---
id: 22-safety-robustness/robust-overfitting-adversarial-training
title: "Robust Overfitting in Adversarial Training"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Robust Overfitting in Adversarial Training

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/robust-overfitting-adversarial-training` · **Status:** open

## 1. Problem Statement

In standard (non-adversarial) training of deep networks, test error decreases roughly monotonically and then plateaus; training past the interpolation point rarely hurts. In adversarial training it does. After the first learning-rate decay, **robust test accuracy peaks and then degrades by 5–10 points while robust training accuracy continues to rise toward 100%**. Rice, Wong & Kolter (ICML 2020) named this *robust overfitting*. Clean test accuracy over the same epochs is flat or improving, so this is not ordinary overfitting.

Three distinct problems are usually conflated:

- **Measurement.** Is the peak-then-decay curve a property of the robust risk, or an artifact of the attack used to evaluate it (a fixed-budget PGD whose success rate drifts with the loss surface)? Solving this means an evaluation of robust risk that is attack-independent enough that the decay is reproduced under a strictly stronger attack.
- **Method.** Find a training procedure whose *final* checkpoint matches the *early-stopped* checkpoint's robust accuracy, without early stopping on held-out robust accuracy, and without giving up peak robustness. Currently unsolved: every known mitigation either recovers only part of the gap or trades peak accuracy for stability.
- **Theory.** Explain why the adversarial empirical risk minimizer generalizes worse with more optimization while the standard one does not. Solving this means a bound whose gap term grows with training epochs at $\epsilon > 0$ and does not at $\epsilon = 0$, with matching lower bound.

## 2. Formal Setting

Data $(x,y) \sim \mathcal{D}$ over $\mathcal{X} \times [K]$, $\mathcal{X} \subseteq [0,1]^d$. Threat model: $\ell_p$ ball $B_\epsilon(x) = \{x' : \|x'-x\|_p \le \epsilon\}$; the canonical instance is CIFAR-10, $p=\infty$, $\epsilon = 8/255$. Model $f_\theta$, loss $\ell$. The adversarial (robust) risk and its empirical counterpart on $S = \{(x_i,y_i)\}_{i=1}^n$:

$$R_\epsilon(\theta) = \mathbb{E}_{\mathcal{D}} \max_{x' \in B_\epsilon(x)} \ell(f_\theta(x'), y), \qquad \hat{R}_\epsilon(\theta, S) = \frac{1}{n}\sum_{i=1}^n \max_{x' \in B_\epsilon(x_i)} \ell(f_\theta(x'), y_i).$$

Adversarial training (Madry et al., ICLR 2018) runs SGD on $\hat{R}_\epsilon$ with the inner max approximated by $k$-step PGD. **Robust overfitting** is the event that, for the trajectory $\theta_1,\dots,\theta_T$,

$$\Delta_{\mathrm{ro}} = \max_{t \le T} A_\epsilon(\theta_t) - A_\epsilon(\theta_T) \gg 0, \quad A_\epsilon(\theta) = \Pr_{\mathcal{D}}\!\left[\forall x' \in B_\epsilon(x): \arg\max f_\theta(x') = y\right].$$

**How each quantity is actually measured.**
- $A_\epsilon$ is *not* computable. It is estimated by a surrogate attack: $\hat{A}_\epsilon^{\mathcal{A}}(\theta) = 1 - \frac{1}{m}\sum_j \mathbb{1}[\mathcal{A}(\theta,x_j,y_j) \text{ succeeds}]$, an **upper bound** on $A_\epsilon$ only if $\mathcal{A}$ is sound. AutoAttack (Croce & Hein, ICML 2020) — APGD-CE, APGD-T, FAB-T, Square — is the community standard; PGD-10/PGD-20 overestimates by 1–4 points.
- $\max_t$ is measured over epoch checkpoints, and the argmax is selected on a **held-out split** (Rice et al. hold out 1,000 CIFAR-10 training images). Selecting on the test set inflates the peak.
- Robust *training* accuracy uses the same $k$-step attack, so "100% robust train accuracy" means "PGD-10 finds no adversarial example", not $A_\epsilon = 1$ on the training set.

**Assumptions, and which are violated.** (i) The inner max is solved — violated; PGD is a local ascent, and its slack is itself epoch-dependent. (ii) Labels are correct within every $\epsilon$-ball, i.e. $B_\epsilon(x_i) \cap B_\epsilon(x_j) = \emptyset$ for $y_i \ne y_j$ — violated at $\epsilon=8/255$ for near-duplicate CIFAR-10 images, which makes $\hat{R}_\epsilon$ have a nonzero irreducible floor and turns fitting it into label-noise memorization. (iii) Train and test are i.i.d. — approximately true for CIFAR-10, but the 80M-TI and diffusion-generated augmentation sets used by SOTA break it.

## 3. State of the Art

**Established (reproduced, ablated).**
- *Early stopping* on held-out robust accuracy is still the strongest single mitigation. Rice et al. (ICML 2020) showed it matches or beats every regularizer they tested ($\ell_2$, cutout, mixup, larger models) across CIFAR-10/100 and SVHN.
- *Adversarial Weight Perturbation* (Wu, Xia, Wang, NeurIPS 2020): minimize $\max_{\|v\| \le \gamma\|\theta\|} \hat R_\epsilon(\theta+v)$. Flattens the weight-loss landscape, adds roughly +1.5 to +3 points final robust accuracy, and visibly shrinks the peak-to-final decay. Reproduced widely; part of most RobustBench top entries.
- *Weight averaging* (SWA/EMA) plus *knowledge distillation* (Chen et al., ICLR 2021, "Robust Overfitting may be mitigated by properly learned smoothening") recovers a large share of the gap on CIFAR-10 ResNet-18.
- *More data* is the only intervention that raises the peak and suppresses the decay simultaneously. Gowal et al. (2020) with 80M-TI; Gowal et al. (NeurIPS 2021) and Wang et al. (ICML 2023) with diffusion-generated data.

**Claimed but not fully ablated.**
- The *label-noise* account (Dong, Liu & Shang, NeurIPS 2022, "Label Noise in Adversarial Training") argues robust overfitting is memorization of the implicit label noise induced by $\epsilon$-ball overlap. The mechanism is argued with synthetic noise injection, not with a measurement of the actual induced noise rate on CIFAR-10.
- The *flat-minima* account (Stutz, Hein & Schiele, ICCV 2021) correlates robust generalization with flatness in the weight loss landscape. Correlational; flatness measures are reparameterization-sensitive.
- MLCAT / small-loss-data reweighting (Yu et al., ICML 2022) attributes the decay to small-loss training points and reports gains; independent replication at other scales is thin.

**Benchmark-number-only.** RobustBench CIFAR-10 $\ell_\infty$, $\epsilon=8/255$ leaderboard entries are single AutoAttack numbers on final checkpoints; they do not report $\Delta_{\mathrm{ro}}$, so the leaderboard tells you nothing about whether a method fixed robust overfitting or just early-stopped well.

## 4. What Is Known

- **The canonical gap.** PreAct ResNet-18, CIFAR-10, $\ell_\infty$ $\epsilon=8/255$, PGD-10 training, piecewise LR decay at epochs 100/150: best robust test accuracy $\approx 53\%$ (PGD-20), final $\approx 46\text{–}47\%$; $\Delta_{\mathrm{ro}} \approx 6$–$7$ points. Clean test accuracy over the same window changes by $<1$ point. (Rice et al., ICML 2020.)
- **It is dataset- and architecture-general.** Present on SVHN, CIFAR-100, Tiny-ImageNet and ImageNet-scale AT, and for TRADES (Zhang et al., ICML 2019) as well as PGD-AT.
- **Onset is tied to the learning-rate decay**, not to a fixed epoch count. Cyclic schedules shift the peak but do not remove it.
- **Data scaling suppresses it.** WRN-70-16 with 50M EDM-generated images: 70.69% AutoAttack robust accuracy at 92.44% clean on CIFAR-10 $\epsilon=8/255$ (Wang et al., ICML 2023), versus 60.75% with CutMix + weight averaging and no extra data (Rebuffi et al., NeurIPS 2021) and 57.20% for the best 2020 no-extra-data model (Gowal et al., 2020). The decay shrinks as the generated-data multiplier grows.
- **Sample complexity is provably higher.** Schmidt et al. (NeurIPS 2018) exhibit a Gaussian model where robust learning needs $\Theta(\sqrt{d})$ times more samples than standard learning. Yin, Ramchandran & Bartlett (ICML 2019) show adversarial Rademacher complexity for linear classes carries an explicit dimension dependence absent in the standard case.
- **Uniform stability degrades under AT.** Xing, Song & Cheng (NeurIPS 2021) and Xiao et al. (NeurIPS 2022) prove stability-based generalization bounds for adversarial training that are looser than the standard-training analogues and grow with the number of SGD steps.
- **CIFAR-10 human label error is ~0.5%** in the test set (Northcutt, Athalye & Mueller, NeurIPS 2021 Datasets & Benchmarks). Clean label noise alone is far too small to explain a 6-point robust gap.

## 5. What Is Not Known

- **Theoretically open.** No bound separates $\epsilon>0$ from $\epsilon=0$ in the *direction of the observed effect*: existing stability bounds grow with $T$ for both, just with worse constants for AT, and none has a matching lower bound that forces the non-monotone test curve. Whether robust overfitting is a necessary consequence of ERM on $\hat R_\epsilon$ with finite $n$, or an artifact of the specific optimizer, is unproved.
- **Empirically open.** Whether $\Delta_{\mathrm{ro}} \to 0$ as $n \to \infty$ with fixed architecture, or converges to a positive constant. Runnable today with EDM-generated data at 10M/50M/200M multipliers; nobody has published the $\Delta_{\mathrm{ro}}$-versus-$n$ curve under AutoAttack at fixed compute-per-example.
- **Methodologically blocked.** The induced label-noise rate $\eta_\epsilon = \Pr[\exists x' \in B_\epsilon(x): x' \in B_\epsilon(\tilde x), \tilde y \ne y]$ is the quantity the label-noise theory needs, and it is not measurable — it requires the robust Bayes classifier. Every current "noise rate" is a proxy computed from a trained model, which is circular.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus absent ground truth**, not compute.

$A_\epsilon$ is only ever seen through an attack. When robust test accuracy falls 6 points late in training, two hypotheses are observationally close: the model genuinely became less robust, or the model's loss surface became easier for APGD to ascend (gradient masking in reverse). AutoAttack narrows this but does not close it — it is still a fixed ensemble of first-order and query-limited attacks, and no verified bound is available at $\epsilon = 8/255$ on CIFAR-10 for the WideResNets that show the effect. Certified methods (randomized smoothing, IBP) give sound numbers but at a different threat model and 20–40 points lower accuracy, so they cannot audit the $\ell_\infty$ PGD-AT curve.

Second, the leading mechanistic explanation — memorizing $\epsilon$-induced label noise — depends on a quantity ($\eta_\epsilon$) that cannot be measured without solving the problem. So the theory is currently unfalsifiable on real data and is tested only by injecting synthetic noise, which begs the question.

## 7. Current Research (as of 2026)

- **Generated-data scaling.** DeepMind (Gowal, Rebuffi, Croce) and the Wang et al. line push diffusion-augmented AT; the open question has shifted from "does data fix it" to "at what rate", and the scaling exponent for $\Delta_{\mathrm{ro}}$ is unpublished.
- **Landscape and stability.** Tübingen (Hein, Croce), CISPA (Stutz), and the algorithmic-stability line (Xing/Song/Cheng; Xiao et al.) continue to push flatness-based and stability-based bounds; the gap between bound and observation remains multiplicative.
- **Label-centric mitigations.** Self-distillation, temporal ensembling of soft labels, and per-example $\epsilon$ schedules. *(frontier — verify)* Several 2024–2025 papers claim near-elimination of $\Delta_{\mathrm{ro}}$ with soft-label AT; independent AutoAttack re-evaluation of these claims is the missing step.
- **NTK / wide-network analyses** of robust overfitting for infinitely wide networks. *(frontier — verify)* Results exist but are for two-layer or lazy-regime models and do not obviously transfer.

## 8. Concrete Next Experiment

**Question decided:** is robust overfitting memorization of $\epsilon$-induced label noise, or an optimization/stability phenomenon?

**Scale.** CIFAR-10, PreAct ResNet-18 (11M params), $\ell_\infty$ $\epsilon = 8/255$, PGD-10 training, SGD 0.1 with decay at 100/150, 200 epochs. ~6 GPU-hours per run on one A100; 5 seeds × 4 arms ≈ 120 GPU-hours. Evaluate every checkpoint after epoch 90 with **AutoAttack on a fixed 2,000-image test subset**, and select the peak on a 1,000-image held-out split carved from train.

**Arms.**
1. *Control:* standard PGD-AT, hard labels.
2. *Soft-label:* identical, but targets are the temporally-averaged (EMA, $\tau=0.9$) robust predictions of the same run — removes the pressure to fit an inconsistent hard label inside an overlapping $\epsilon$-ball without changing the optimizer.
3. *Optimization control:* hard labels + AWP ($\gamma = 5\times10^{-3}$) — changes stability, not labels.
4. *Placebo:* soft labels drawn from a *fixed uniform-over-top-2* distribution, matched in entropy to arm 2 but carrying no per-example information. Rules out "label smoothing of any kind is enough".

**Deciding number.** $\Delta_{\mathrm{ro}}^{\mathrm{AA}} = \hat{A}^{\mathrm{AA}}_\epsilon(\theta_{t^\star}) - \hat{A}^{\mathrm{AA}}_\epsilon(\theta_T)$, mean over 5 seeds, with the side constraint that peak robust accuracy $\hat{A}^{\mathrm{AA}}_\epsilon(\theta_{t^\star})$ must not drop by more than 0.5 points versus control.

**Interpretation.** Control is expected at $\Delta_{\mathrm{ro}}^{\mathrm{AA}} \approx 5\text{–}6$ points. If arm 2 reaches $\Delta_{\mathrm{ro}}^{\mathrm{AA}} < 1.0$ while arm 4 stays $> 3$, the label-noise account is supported and the mitigation is label-side. If arm 2 and arm 4 are within 1 point of each other, the effect is entropy/regularization, not information — and the label-noise account is not what is doing the work.

## 9. Key References

- **[Foundational]** Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR, 2018. — arXiv:1706.06083
- **[Foundational]** Leslie Rice, Eric Wong, J. Zico Kolter. *Overfitting in Adversarially Robust Deep Learning.* ICML, 2020. — arXiv:2002.11569
- **[Foundational]** Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, Aleksander Madry. *Adversarially Robust Generalization Requires More Data.* NeurIPS, 2018. — arXiv:1804.11285
- **[SOTA]** Zekai Wang, Tianyu Pang, Chao Du, Min Lin, Weiwei Liu, Shuicheng Yan. *Better Diffusion Models Further Improve Adversarial Training.* ICML, 2023. — arXiv:2302.04638
- **[SOTA]** Sylvestre-Alvise Rebuffi, Sven Gowal, Dan A. Calian, Florian Stimberg, Olivia Wiles, Timothy Mann. *Data Augmentation Can Improve Robustness.* NeurIPS, 2021. — arXiv:2103.01946
- **[SOTA]** Dongxian Wu, Shu-Tao Xia, Yisen Wang. *Adversarial Weight Perturbation Helps Robust Generalization.* NeurIPS, 2020. — arXiv:2004.05884
- **[Method]** Tianlong Chen, Zhenyu Zhang, Sijia Liu, Shiyu Chang, Zhangyang Wang. *Robust Overfitting may be mitigated by properly learned smoothening.* ICLR, 2021.
- **[Mechanism]** Chengyu Dong, Liyuan Liu, Jingbo Shang. *Label Noise in Adversarial Training: A Novel Perspective to Study Robust Overfitting.* NeurIPS, 2022. — arXiv:2110.03135
- **[Mechanism]** David Stutz, Matthias Hein, Bernt Schiele. *Relating Adversarially Robust Generalization to Flat Minima.* ICCV, 2021. — arXiv:2104.04448
- **[Theory]** Yue Xing, Qifan Song, Guang Cheng. *On the Algorithmic Stability of Adversarial Training.* NeurIPS, 2021.
- **[Theory]** Dong Yin, Kannan Ramchandran, Peter Bartlett. *Rademacher Complexity for Adversarially Robust Generalization.* ICML, 2019. — arXiv:1810.11914
- **[Evaluation]** Francesco Croce, Matthias Hein. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks.* ICML, 2020. — arXiv:2003.01690
- **[Survey/Benchmark]** Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, Matthias Hein. *RobustBench: A Standardized Adversarial Robustness Benchmark.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2010.09670

## 10. Worked Example

Take the canonical run: PreAct ResNet-18, CIFAR-10 ($n = 50{,}000$), $\epsilon = 8/255$, PGD-10.

| Epoch | Robust train acc (PGD-10) | Robust test acc (PGD-20) | Clean test acc |
|---|---|---|---|
| 99 (pre-decay) | ~44% | ~44% | ~78% |
| 105 (peak) | ~75% | **~53%** | ~82% |
| 200 (final) | ~96% | **~47%** | ~84% |

The robust generalization gap goes from ~0 to ~49 points while the clean gap stays near 5. Now ask whether label noise explains the 6-point decay. Suppose a fraction $\eta_\epsilon$ of training points are unlearnable because their $\epsilon$-ball overlaps another class. Fitting them costs test accuracy roughly in proportion, so a first-order account needs $\eta_\epsilon \gtrsim 0.06$, i.e. **~3,000 of 50,000 CIFAR-10 images**.

Now try to measure that. Human label error in CIFAR-10 is ~0.5% — an order of magnitude too small. So $\eta_\epsilon$ must come from geometric overlap: pairs $(x_i,x_j)$ with $y_i \ne y_j$ and $\|x_i - x_j\|_\infty \le 2\epsilon = 16/255$. Counting those pairs directly in pixel space on CIFAR-10 returns essentially zero — natural images of different classes are not within 16/255 in $\ell_\infty$. The overlap that matters is in *feature* space, under the representation the network happens to learn, which is exactly the object being trained. **The noise rate is defined only relative to the model whose overfitting it is supposed to explain.**

That is the obstruction in one line: the 6-point number is solid and reproducible, and the leading explanation for it depends on a quantity that no current method can measure without assuming the answer. This is why the experiment in §8 is built around a *placebo arm* — with $\eta_\epsilon$ unmeasurable, the only available test is whether per-example label information does something that entropy-matched noise does not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*