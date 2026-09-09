---
id: 08-loss-and-heads/optimal-label-smoothing-schedule
title: "Optimal Label Smoothing Schedule"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Label Smoothing Schedule

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/optimal-label-smoothing-schedule` · **Status:** open

## 1. Problem Statement

Label smoothing replaces the one-hot target with a mixture of the one-hot target and a fixed prior, controlled by a scalar $\varepsilon$. In practice $\varepsilon$ is a constant, almost always $0.1$, carried unchanged from Szegedy et al. (2016) into image classifiers, machine translation, and speech models. The problem: **is a constant the right function of training time, and if not, what schedule $\varepsilon_t$ is optimal for a stated objective?**

Three variants, of different difficulty:

- **Measurement.** Given a model family, dataset, and compute budget, does *any* non-constant $\varepsilon_t$ beat the best constant $\varepsilon^\star$ on a pre-registered metric, by more than seed noise? This is a runnable experiment and is mostly unrun at scale.
- **Method.** Produce a schedule rule — a function of step, loss, or measured confidence — that transfers across datasets and scales without per-task tuning. Adaptive and online variants exist; none has an independently reproduced transfer claim.
- **Theory.** Characterise $\varepsilon^\star_t$ from properties of the data (label noise rate, class count, Bayes error) and the optimiser. Only partial results exist, and they are for convex or last-layer settings.

Solving it means: a schedule rule, a proof or reliable empirical law for when it beats constant $\varepsilon$, and an account of which downstream metric it optimises — accuracy, calibration, distillation quality, and transfer do **not** move together under smoothing.

## 2. Formal Setting

Let $\mathcal{D}$ be a distribution over $(x,y)$ with $y\in\{1,\dots,K\}$. A model $f_\theta$ emits logits $z\in\mathbb{R}^K$ and $p_\theta(\cdot\mid x)=\mathrm{softmax}(z)$. The smoothed target at step $t$ is

$$q^{(t)}_k(y) = (1-\varepsilon_t)\,\mathbb{1}[k=y] + \varepsilon_t\, u_k,$$

with $u$ a prior, usually uniform $u_k=1/K$. The loss is $\mathcal{L}_t(\theta)=\mathbb{E}\big[\mathrm{CE}(q^{(t)}(y),p_\theta(\cdot\mid x))\big]$, which decomposes as

$$\mathcal{L}_t = (1-\varepsilon_t)\,\mathrm{CE}(\delta_y,p_\theta) + \varepsilon_t\,\mathrm{KL}(u\,\|\,p_\theta) + \varepsilon_t H(u).$$

The schedule is $\varepsilon:\{1,\dots,T\}\to[-a,1)$ ($\varepsilon<0$ is admissible — see Wei et al. 2022). Quantities as measured:

- **Population optimum.** For separable data the smoothed loss is minimised at a *finite* logit gap: correct-class probability $1-\varepsilon+\varepsilon/K$, so the gap to any wrong class is $\Delta^\star(\varepsilon)=\log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}$. At $K=1000,\varepsilon=0.1$, $\Delta^\star\approx\log(0.9001/0.0001)=9.1$ nats. At $\varepsilon=0$, $\Delta^\star=\infty$. This finite target is the mechanism behind every reported effect.
- **Accuracy.** Top-1 on a held-out split, reported as mean ± std over $\geq 5$ seeds. Below ~5 seeds the ImageNet seed std (~$0.1$–$0.15\%$ for ResNet-50) swamps typical smoothing effects.
- **Calibration.** ECE with $M=15$ equal-width bins, $\mathrm{ECE}=\sum_m \frac{|B_m|}{n}\,|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)|$. Bin count is a free parameter and ECE is a biased estimator; report adaptive-bin ECE alongside.
- **Distillation value.** Student top-1 after KD from the teacher, at fixed temperature $\tau$ — not a property of the teacher alone.
- **Transfer.** Linear-probe accuracy on held-out tasks (Kornblith et al. 2021 protocol).

Assumptions, and where they break:

1. *Uniform prior $u$ is the correct target for non-target mass.* Violated: classes are semantically nested (ImageNet dog breeds), so uniform mass is a mis-specified prior.
2. *Label noise is symmetric.* Violated in every real annotation pipeline; noise is instance-dependent and class-conditional.
3. *A single $\varepsilon$ serves all examples.* Violated: the Bayes-optimal smoothing depends on per-example ambiguity.
4. *The training objective's optimum is reached.* Violated: at typical budgets the model is far from the population optimum, which is precisely why the *schedule* rather than the *value* can matter.

## 3. State of the Art

**Established.** Constant $\varepsilon=0.1$ is the operative SOTA. Szegedy et al. (CVPR 2016) introduced it; Vaswani et al. (NeurIPS 2017) adopted $\varepsilon_{ls}=0.1$ for Transformers and stated the trade explicitly — it *hurts* perplexity while *improving* BLEU. Müller, Kornblith & Hinton (NeurIPS 2019) established two robust facts: smoothing improves calibration, and it destroys within-class logit structure, degrading the teacher's value for distillation. Pereyra et al. (ICLR Workshop 2017) established that the confidence penalty $-\beta H(p_\theta)$ is a near-substitute, and Meister, Salesky & Cotterell (ICML 2020) generalised this: over a family of entropy regularisers the *strength* dominates the *form*, so the object worth scheduling is the regularisation magnitude, not the smoothing formula.

**Theory SOTA.** Xu et al. (2020, *Towards Understanding Label Smoothing*) give the only convergence-rate argument for a *schedule*: label smoothing reduces gradient variance and accelerates the early phase, but biases the stationary point; their two-stage TSLA drops smoothing after a switch point and improves the stochastic-optimisation rate over pure CE in their analysed setting. The analysis assumes convexity or a linear last layer.

**Claimed but unablated.** Online Label Smoothing (Zhang et al., *Delving Deep into Label Smoothing*, IEEE TIP 2021) replaces $u$ with an accumulated prediction histogram and reports gains on CIFAR/ImageNet; the improvement is confounded with the changed prior, so it does not isolate a schedule effect. Adaptive per-example smoothing papers generally report a single benchmark number per dataset without seed variance, which is below the resolution needed. Chandrasegaran et al. (ICML 2022) is the main correction to Müller: LS teachers are usable for KD at low temperature, so the "LS breaks distillation" claim is temperature-dependent, not absolute.

**Frontier LLMs.** Modern decoder-only pretraining recipes largely omit label smoothing, using a z-loss / logit-norm penalty instead. There is no public scaling study of $\varepsilon_t$ for LLM pretraining.

## 4. What Is Known

- $\varepsilon=0.1$ on ILSVRC-2012 with Inception-v2/v3 gives roughly **0.2% absolute** top-1 and top-5 improvement (Szegedy et al. 2016, $K=1000$, ~1.2M images). The effect is small enough that fewer than ~5 seeds cannot resolve it.
- Transformer-base, WMT14 EN–DE, 28.4 BLEU: $\varepsilon_{ls}=0.1$ **increases** validation perplexity and **increases** BLEU (Vaswani et al. 2017). Established that the accuracy metric and the likelihood metric move in opposite directions.
- Smoothing sharply reduces ECE across ImageNet CNNs and NMT models, moving models from over- to near-calibrated at $\varepsilon\approx0.1$ (Müller et al. 2019), consistent with the miscalibration baseline in Guo et al. (ICML 2017).
- Penultimate-layer representations collapse toward equidistant class-template clusters under smoothing; the erased information is exactly the inter-class similarity structure KD exploits (Müller et al. 2019, ImageNet/CIFAR scale).
- Under symmetric label noise, smoothing helps and acts like a shrinkage/denoising estimator; Lukasik et al. (ICML 2020) show multi-point gains on CIFAR at 20–40% injected symmetric noise. Wei et al. (ICML 2022) show the sign flips at high noise rates: **negative** smoothing ($\varepsilon<0$) is preferable there.
- Better-accuracy losses including label smoothing yield **worse** linear-probe transfer (Kornblith et al. 2021), an order-1% downstream cost for sub-1% ImageNet gain.

The consistent shape: at fixed budget, $\varepsilon\approx0.1$ buys calibration and a small accuracy gain, and sells likelihood, transferable features, and teacher quality.

## 5. What Is Not Known

- **Empirically open.** Whether any $\varepsilon_t$ — linear decay to 0, step-off at 60% of training, warm-up from 0 — beats the best constant on ImageNet-scale top-1 by more than seed noise. TSLA-style two-stage schedules are the obvious candidate and have never been run with adequate seeds at ImageNet or LLM scale. Also open: whether the $\varepsilon^\star$ that maximises accuracy coincides with the one that minimises ECE at any $t$.
- **Theoretically open.** No characterisation of $\varepsilon^\star_t$ for non-convex deep networks. No proof that a constant is or is not optimal within the class of schedules for a fixed step budget. The relation between $\varepsilon^\star$ and the label-noise rate $\eta$ is known only in restricted symmetric-noise models.
- **Methodologically blocked.** "Optimal" is undefined until the objective is chosen, and the objectives conflict (accuracy vs. perplexity vs. ECE vs. transfer vs. KD). ECE itself is bin-dependent and biased, so "the schedule that best calibrates" is not a well-posed target with current estimators.

## 6. Why It Is Hard

Two specific obstructions.

**Effect size below measurement noise, at high compute cost.** The reported constant-$\varepsilon$ accuracy gain is ~0.2–0.5% absolute; the ImageNet seed std for ResNet-50 is ~0.1–0.15%. Resolving a *schedule*'s advantage over the best constant — a second-order effect, plausibly under 0.2% — needs perhaps 5–10 seeds per arm across a grid of schedules, i.e. tens of full ImageNet runs before any single conclusion. That is why the experiment is unrun rather than negative.

**Confounded objective / evaluation that does not measure what it names.** Smoothing is nearly always tuned on top-1 and then justified by calibration, but Müller et al. show the same intervention degrades distillation and Kornblith et al. show it degrades transfer. A schedule tuned on one axis is not known to be good on the others, and no paper reports the full four-axis profile for a single schedule sweep. On top of this, smoothing interacts with weight decay, mixup/CutMix (which already softens targets), and learning-rate decay — a decaying $\varepsilon_t$ correlated with a decaying LR is not identifiable from LR schedule effects without an explicit crossed design.

## 7. Current Research (as of 2026)

- **Noise-aware smoothing.** Extending negative label smoothing (Wei et al. 2022) into $\eta$-dependent schedules; the mapping from estimated noise rate to $\varepsilon^\star_t$ is the active question.
- **Distillation-compatible smoothing.** Following Chandrasegaran et al. (ICML 2022), tuning teacher $\varepsilon$ jointly with KD temperature rather than treating them separately.
- **Regularisation-strength scheduling generally.** Treating $\varepsilon_t$, weight decay, and dropout as one scheduled regularisation budget, per the Meister et al. equivalence result.
- **LLM output-head regularisation.** z-loss, logit soft-capping, and entropy penalties are the LLM-era substitutes; whether a smoothing schedule adds anything over z-loss at $10^{9}$–$10^{11}$ parameters is unstudied in public *(frontier — verify)*.
- **Per-example adaptive $\varepsilon$** conditioned on model confidence or annotator disagreement, most active in medical imaging and ASR *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** ResNet-50 on ImageNet-1k, 90 epochs, standard recipe, 5 seeds per arm. ~8 GPU-days per arm on 8×A100; 5 arms → ~200 GPU-days total. Cheap enough to actually run, large enough that $K=1000$ makes smoothing non-trivial.

**Arms.**
1. **Control:** constant $\varepsilon=0.1$ (the incumbent).
2. $\varepsilon=0$ (calibration floor / accuracy reference).
3. **Step-off (TSLA):** $\varepsilon=0.1$ for epochs 1–60, $\varepsilon=0$ thereafter.
4. **Linear decay:** $\varepsilon_t = 0.1\,(1-t/T)$.
5. **Warm-up:** $\varepsilon_t = 0.1\,(t/T)$ — tests whether the benefit is late, not early.

Freeze the LR schedule identically across arms so $\varepsilon_t$ is not confounded with LR decay. Report the four-axis profile per arm: top-1, 15-bin and adaptive-bin ECE, KD student top-1 at $\tau\in\{1,4\}$, and mean linear-probe accuracy over 5 downstream tasks.

**Deciding number.** $\Delta = \overline{\mathrm{top1}}(\text{arm 3}) - \overline{\mathrm{top1}}(\text{arm 1})$, with a pre-registered threshold of $\Delta > 0.3\%$ absolute and a two-sample $t$-test at $p<0.01$ over 5 seeds (seed std ~0.12% ⇒ SE ~0.054%, so 0.3% is ~5.5 SE — detectable). $\Delta > 0.3\%$ with ECE not worse than arm 1 by more than $0.01$ moves the field's default off constant smoothing. $|\Delta| < 0.1\%$ is a real negative result and should be published as one: it closes the empirical variant for vision.

## 9. Key References

- **[Foundational]** C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. *Rethinking the Inception Architecture for Computer Vision.* CVPR, 2016. — arXiv:1512.00567
- **[Foundational]** G. Pereyra, G. Tucker, J. Chorowski, Ł. Kaiser, G. Hinton. *Regularizing Neural Networks by Penalizing Confident Output Distributions.* ICLR Workshop, 2017. — arXiv:1701.06548
- **[Foundational]** A. Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[SOTA]** R. Müller, S. Kornblith, G. Hinton. *When Does Label Smoothing Help?* NeurIPS, 2019. — arXiv:1906.02629
- **[SOTA]** M. Lukasik, S. Bhojanapalli, A. K. Menon, S. Kumar. *Does Label Smoothing Mitigate Label Noise?* ICML, 2020. — arXiv:2003.02819
- **[SOTA]** Y. Xu, Y. Xu, Q. Qian, H. Li, R. Jin. *Towards Understanding Label Smoothing.* 2020. — arXiv:2006.11653
- **[SOTA]** J. Wei, H. Liu, T. Liu, G. Niu, M. Sugiyama, Y. Liu. *To Smooth or Not? When Label Smoothing Meets Noisy Labels.* ICML, 2022.
- **[SOTA]** K. Chandrasegaran, N.-T. Tran, Y. Zhao, N.-M. Cheung. *Revisiting Label Smoothing and Knowledge Distillation Compatibility: What was Missing?* ICML, 2022.
- **[Theory]** C. Meister, E. Salesky, R. Cotterell. *Generalized Entropy Regularization, or: There's Nothing Special about Label Smoothing.* ACL, 2020. — arXiv:2005.00820
- **[Context]** C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[Context]** S. Kornblith, T. Chen, H. Lee, M. Norouzi. *Why Do Better Loss Functions Lead to Less Transferable Features?* NeurIPS, 2021. — arXiv:2010.16402
- **[Method]** C.-B. Zhang, P.-T. Jiang, Q. Hou, Y. Wei, Q. Han, Z. Li, M.-M. Cheng. *Delving Deep into Label Smoothing.* IEEE TIP, 2021. — arXiv:2011.12562

## 10. Worked Example

Take $K=1000$ (ImageNet) and $\varepsilon=0.1$. The smoothed target is $0.9001$ on the true class and $10^{-4}$ elsewhere. The population-optimal logit gap is

$$\Delta^\star = \log\frac{0.9001}{0.0001} \approx 9.11 \text{ nats}.$$

A trained ResNet-50 with $\varepsilon=0$ typically reaches a mean correct-class softmax probability of ~0.95 on training data late in training, i.e. an effective gap of roughly $\log(0.95/(0.05/999)) \approx 9.85$ nats. The two numbers differ by about **0.7 nats**. That is the entire size of the intervention: constant label smoothing at $\varepsilon=0.1$ asks the network to sit ~0.7 nats less confident than it otherwise would.

Now ask what a schedule changes. A step-off at epoch 60 releases that 0.7-nat constraint for the final 30 epochs. Under the TSLA argument the early phase keeps the variance-reduction benefit and the late phase removes the bias — so the predicted gain is the *difference between two already-small effects*.

The obstruction is visible in the arithmetic. The constant-$\varepsilon$ effect on ImageNet top-1 is ~0.2–0.5% absolute. A schedule can only recover a fraction of the residual bias, so a plausible prior on $\Delta$ is 0.1–0.3%. ResNet-50 seed std is ~0.12%. With 5 seeds the SE is 0.054%; a 0.15% true effect is under 3 SE, and a 0.10% effect is under 2 SE. **Detecting the effect you actually expect requires close to the maximum number of seeds anyone runs — which is why every published schedule claim rests on 1–3 runs and none has been independently reproduced.** The problem is open not because the experiment is exotic, but because the effect size sits at the edge of the noise floor and nobody has paid for the seeds.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*