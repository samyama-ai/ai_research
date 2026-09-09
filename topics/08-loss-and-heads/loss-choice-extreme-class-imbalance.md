---
id: 08-loss-and-heads/loss-choice-extreme-class-imbalance
title: "Loss Function Choice for Extreme Class Imbalance"
topic: 08-loss-and-heads
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Function Choice for Extreme Class Imbalance

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/loss-choice-extreme-class-imbalance` · **Status:** empirically-open

## 1. Problem Statement

Given a training set whose class frequencies span three or more orders of magnitude ($\pi_{\max}/\pi_{\min} \ge 10^3$), decide which training loss to use. Candidates: plain softmax cross-entropy (CE), focal loss, class-balanced reweighting, LDAM, logit adjustment, Dice/Tversky (segmentation), or CE plus a post-hoc threshold shift.

Three distinct variants, routinely conflated:

- **Method variant.** Does any loss modification beat CE trained normally and then *post-hoc corrected* (logit shift by $-\tau\log\hat\pi_y$, or threshold tuning on a validation set)? This is the practical question and it is empirically open.
- **Theory variant.** For a given target metric $M$ (balanced accuracy, macro-F1, worst-group recall, Dice), which surrogate losses are **$M$-consistent** — minimising the surrogate over all measurable functions maximises $M$? Partially settled for linear-fractional metrics, open for margin-modified losses under finite capacity.
- **Measurement variant.** Under $\pi_{\min}\sim10^{-4}$, the test set contains $O(10)$ minority examples. The metric's standard error can exceed the reported gap between methods. Much of the literature's ranking is not identified.

Solving it means: a decision rule mapping (imbalance ratio, sample count of the rarest class, capacity, target metric) to a loss, validated with confidence intervals that exclude the CE + post-hoc baseline.

## 2. Formal Setting

Data $(x,y)\sim\mathcal{D}$ over $\mathcal{X}\times[K]$, priors $\pi_k = \Pr[y=k]$, measured as empirical training counts $n_k/n$ — not as test priors, which may differ (this is the first violated assumption). Imbalance ratio $\rho = \max_k n_k/\min_k n_k$; "extreme" means $\rho\ge10^3$. Model $f_\theta:\mathcal{X}\to\mathbb{R}^K$ producing logits; predicted label $\arg\max_k f_k(x)$.

Losses, all on logits:

$$\ell_{\mathrm{CE}} = -\log p_y,\qquad p_k=\frac{e^{f_k}}{\sum_j e^{f_j}}$$

$$\ell_{\mathrm{focal}} = -(1-p_y)^\gamma\log p_y,\qquad \ell_{\mathrm{CB}} = -\frac{1-\beta}{1-\beta^{n_y}}\log p_y$$

$$\ell_{\mathrm{LA}} = -\log\frac{e^{f_y+\tau\log\pi_y}}{\sum_j e^{f_j+\tau\log\pi_j}},\qquad \ell_{\mathrm{LDAM}} = -\log\frac{e^{f_y-\Delta_y}}{e^{f_y-\Delta_y}+\sum_{j\ne y}e^{f_j}},\ \Delta_y \propto n_y^{-1/4}$$

Post-hoc correction is the map $f_k \mapsto f_k - \tau\log\pi_k$ applied at inference to a CE-trained model; $\tau$ is chosen on validation data, so it costs one scalar sweep and zero retraining. This is the control arm the field under-uses.

Target metrics as measured: balanced accuracy $\mathrm{BA}=\frac1K\sum_k \widehat{\mathrm{rec}}_k$ where $\widehat{\mathrm{rec}}_k = \frac{1}{m_k}\sum_{i:y_i=k}\mathbb{1}[\hat y_i=k]$ over $m_k$ test examples of class $k$. Its variance is dominated by the rarest class: $\mathrm{Var}(\mathrm{BA})\ge \frac{1}{K^2}\cdot\frac{r(1-r)}{m_{\min}}$. With $m_{\min}=10$ and $r=0.5$, the per-class standard error is $0.158$ — larger than nearly every headline improvement in the long-tail literature.

Assumptions known to be violated: (i) train and test priors equal (long-tail benchmarks deliberately break this — test is uniform); (ii) label noise independent of frequency (rare-class labels are noisier; Van Horn & Perona, 2017); (iii) i.i.d. sampling (rare classes are often geographically or temporally clustered); (iv) separability, which most margin theory assumes and which fails for the head classes at realistic capacity.

## 3. State of the Art

**Established (theory).** Logit adjustment is Fisher-consistent for balanced error: the Bayes-optimal balanced-error rule is $\arg\max_k \Pr[y=k\mid x]/\pi_k$, and the additive shift realises it exactly (Menon et al., ICLR 2021). Kini et al. (NeurIPS 2021) prove that in the separable, overparameterised regime, *additive* logit margins do not change the terminal direction of gradient descent — the implicit bias washes them out — while *multiplicative* logit scaling does. This is a genuine theorem and it explains why several published additive-margin gains vanish with longer training. Fang et al. (PNAS 2021) prove "minority collapse": under the layer-peeled model, once $\rho$ exceeds a threshold, minority-class classifier vectors converge to each other, making them indistinguishable regardless of the reweighting applied.

**Established (empirical).** Decoupling: features learned with plain CE on the imbalanced data, followed by classifier retraining or $\tau$-normalisation, matches or beats end-to-end reweighted training (Kang et al., ICLR 2020).

**Claimed but unablated.** Focal loss's original justification — down-weighting easy examples fixes foreground/background imbalance in dense detection (Lin et al., ICCV 2017) — was never separated from the paper's prior-initialised bias term, and RetinaNet's gain over its own CE ablation is confounded with that initialisation. Charoenphakdee et al. (CVPR 2021) show focal loss is not classification-calibrated for posterior estimation in general, so its probabilities are not the thing they are read as. Mukhoti et al. (NeurIPS 2020) report focal loss *improves* calibration (ECE), which is compatible only because ECE rewards the underconfidence focal induces — a different claim than posterior correctness.

**Benchmark-number-only results.** Most long-tail leaderboard entries (CIFAR-100-LT, ImageNet-LT, iNaturalist-2018) are single-seed, single-metric top-1 numbers with no interval. The commonly cited ordering CE < focal < CB < LDAM < LA rests on these.

## 4. What Is Known

- **Post-hoc beats retraining, at scale.** On ImageNet-LT ($\rho=256$, 115.8k images, 1000 classes) logit adjustment reports ~$51$–$52\%$ top-1 vs ~$45\%$ for CE, and the *post-hoc* variant is within roughly a point of the loss-modified variant (Menon et al., 2021). The extra training-time machinery buys little.
- **Importance weighting decays.** Byrd & Lipton (ICML 2019) show that for separable data with unregularised deep nets, the effect of importance weights on the learned function vanishes as training proceeds; measured on CIFAR-scale nets, weighted and unweighted models converge to near-identical decision boundaries.
- **Weight decay matters more than the loss.** Alshammari et al. (CVPR 2022) reach ~$53.9\%$ on ImageNet-LT with plain CE plus tuned per-layer weight decay and a MaxNorm constraint — matching or exceeding several specialised losses.
- **Dice-family losses optimise their own metric.** Eelbode et al. (IEEE TMI 2020) show soft-Dice and soft-Jaccard are monotonically linked to their hard counterparts; Ma et al. (Medical Image Analysis 2021, "Loss odyssey") evaluate ~20 segmentation losses across 4 tasks and find no loss wins on more than a subset — compound CE+Dice is the safe default, not a winner.
- **Calibration degrades with rarity.** Wallace & Dahabreh (ICDM 2012) show class-probability estimates on the minority class are systematically biased low, and that resampling changes the bias without removing it.

## 5. What Is Not Known

- **Empirically open.** No study varies $\rho$ (from $10^1$ to $10^4$), $n_{\min}$ (from $10^0$ to $10^3$), and capacity independently while holding optimiser, schedule, augmentation and weight decay fixed, with $\ge5$ seeds and reported intervals. Every ingredient exists; the factorial has not been run. This is the central gap.
- **Theoretically open.** Whether any training-time loss modification can beat CE-plus-optimal-post-hoc-shift when the hypothesis class is misspecified (the well-specified case gives them the same Bayes rule, so the whole question lives in the approximation/estimation trade-off). No proof either way.
- **Theoretically open.** Sample complexity for the rarest class: how many examples of class $k$ are needed for $\widehat{\mathrm{rec}}_k$ to exceed a target, as a function of the representation learned from the *other* classes. Transfer from head to tail is unquantified.
- **Methodologically blocked.** "Which loss is better for rare classes" is not well posed until the metric is fixed and its estimator has usable variance. With $m_{\min}\le 30$ test examples, macro-F1 differences below ~3 points are not resolvable from a single test set.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**.

1. *The evaluation does not measure what it names.* Balanced accuracy on a long-tail benchmark is an average of per-class recalls whose tail terms are estimated from a handful of examples. The reported quantity is mostly noise from the tail and mostly signal from the head, in a mixture nobody reports.
2. *The loss is not identified from the result.* Changing the loss also changes effective learning rate on tail gradients, effective regularisation, and early-stopping point. Alshammari et al. show weight decay alone spans the range that separates published losses — so a loss-vs-loss comparison at fixed hyperparameters is a comparison of hyperparameter luck.
3. *Implicit bias erases the intervention.* Byrd & Lipton and Kini et al. together mean the training-time signal you inject can be asymptotically discarded by gradient descent. A short-schedule experiment and a long-schedule one can produce opposite rankings, both correct.

Cost is not the obstruction: the decisive experiment fits on a handful of GPUs.

## 7. Current Research (as of 2026)

- **Post-hoc and decoupled correction as default.** Google Research (Menon, Jayasumana, Kumar and collaborators) continue on consistency-based adjustment and distillation for long tails.
- **Neural-collapse-based analysis of imbalance.** Layer-peeled and unconstrained-features models extending Fang et al.; groups at Penn, Stanford, EPFL.
- **Loss-landscape-free accounts.** Regularisation-first explanations (weight decay, MaxNorm, feature-norm balancing) after Alshammari et al.
- **Long-tail in foundation models.** Whether pretrained representations dissolve the problem — fine-tuning a CLIP/DINOv2 backbone on a $\rho=10^3$ task and finding the loss choice irrelevant *(frontier — verify; scattered reports, no controlled study)*.
- **Extreme multi-label** (millions of labels, $\pi_{\min}<10^{-6}$): negative sampling and propensity-scored losses; here the loss choice interacts with the sampler and cannot be studied alone.

## 8. Concrete Next Experiment

**Scale.** CIFAR-100-LT and ImageNet-LT, plus a synthetic $\rho=10^4$ variant of ImageNet-LT ($n_{\min}=1$–$5$). ResNet-32 and ResNet-50. Grid: $\rho\in\{10,10^2,10^3,10^4\}$ × loss $\in$ {CE, focal($\gamma{=}2$), CB($\beta{=}0.9999$), LDAM, LA} × 5 seeds. Weight decay and learning rate tuned *per cell* by equal-budget random search (20 trials), so no method wins by hyperparameter accident. Two schedules: 200 epochs and 600 epochs, to expose implicit-bias decay. Cost: about 1,600 short runs ≈ 2–3k A100-hours.

**Control arm.** CE trained with the same tuned hyperparameters, then post-hoc logit shift $f_k - \tau\log\hat\pi_k$ with $\tau$ selected on a held-out split. No retraining.

**The deciding number.** $\Delta = \mathrm{BA}(\text{best loss}) - \mathrm{BA}(\text{CE + post-hoc})$, with a bootstrap 95% CI over seeds *and* test resamples. Decision rule: if the CI for $\Delta$ contains 0 at every $\rho$, training-time loss modification is empirically dead for this regime and the catalog entry closes on the method variant. If $\Delta > 0$ with the CI excluding 0 at some $\rho^\star$, report $\rho^\star$ — the imbalance ratio at which post-hoc correction stops sufficing. That single threshold is the deliverable.

Report per-class test counts alongside every number; refuse to report a metric whose tail standard error exceeds $\Delta$.

## 9. Key References

- **[Foundational]** Charles Elkan. *The Foundations of Cost-Sensitive Learning.* IJCAI, 2001.
- **[Foundational]** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. *Focal Loss for Dense Object Detection.* ICCV, 2017. — arXiv:1708.02002
- **[SOTA]** Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar. *Long-tail learning via logit adjustment.* ICLR, 2021. — arXiv:2007.07314
- **[SOTA]** Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, Yannis Kalantidis. *Decoupling Representation and Classifier for Long-Tailed Recognition.* ICLR, 2020. — arXiv:1910.09217
- **[SOTA]** Shaden Alshammari, Yu-Xiong Wang, Deva Ramanan, Shu Kong. *Long-Tailed Recognition via Weight Balancing.* CVPR, 2022. — arXiv:2203.14197
- **[Theory]** Ganesh Ramachandra Kini, Orestis Paraskevas, Samet Oymak, Christos Thrampoulidis. *Label-Imbalanced and Group-Sensitive Classification under Overparameterization.* NeurIPS, 2021. — arXiv:2103.01550
- **[Theory]** Cong Fang, Hangfeng He, Qi Long, Weijie J. Su. *Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training.* PNAS, 2021.
- **[Theory]** Jonathon Byrd, Zachary C. Lipton. *What is the Effect of Importance Weighting in Deep Learning?* ICML, 2019. — arXiv:1812.03372
- **[Theory]** Nontawat Charoenphakdee, Jayakorn Vongkulbhisal, Nuttapong Chairatanakul, Masashi Sugiyama. *On Focal Loss for Class-Posterior Probability Estimation: A Theoretical Perspective.* CVPR, 2021.
- **[Method]** Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie. *Class-Balanced Loss Based on Effective Number of Samples.* CVPR, 2019. — arXiv:1901.05555
- **[Method]** Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma. *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.* NeurIPS, 2019. — arXiv:1906.07413
- **[Calibration]** Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip H.S. Torr, Puneet K. Dokania. *Calibrating Deep Neural Networks using Focal Loss.* NeurIPS, 2020. — arXiv:2002.09437
- **[Segmentation]** Jun Ma et al. *Loss odyssey in medical image segmentation.* Medical Image Analysis, 2021.
- **[Segmentation]** Tom Eelbode et al. *Optimization for Medical Image Segmentation: Theory and Practice When Evaluating With Dice Score or Jaccard Index.* IEEE Transactions on Medical Imaging, 2020.
- **[Survey]** Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, Jiashi Feng. *Deep Long-Tailed Learning: A Survey.* IEEE TPAMI, 2023. — arXiv:2110.04596

## 10. Worked Example

Binary detection, $\pi_1 = 10^{-4}$, $n = 10^6$ training examples so $n_1 = 100$ positives. Test set $2\times10^5$ examples, so $m_1 = 20$ positives.

Suppose the model's true positive-class recall is $r = 0.40$ and specificity is $0.999$.

- Observed positives detected: $\mathrm{Bin}(20, 0.40)$, mean $8$, sd $\sqrt{20\cdot0.4\cdot0.6}=2.19$. So $\widehat{\mathrm{rec}}_1 = 0.40 \pm 0.11$ (1 sd).
- Balanced accuracy $= (0.40 + 0.999)/2 = 0.6995$, with sd $\approx 0.11/2 = 0.055$.

Now run focal loss and see $\widehat{\mathrm{rec}}_1 = 0.55$ (11 of 20 detected) versus CE's $0.40$ (8 of 20). Headline: "+15 points recall, +7.5 points balanced accuracy." Fisher's exact test on $8/20$ vs $11/20$ gives $p \approx 0.53$. The difference is three test examples. The 95% CI on the recall gap is roughly $[-0.16, +0.45]$.

To resolve a true 15-point recall gap at $\alpha=0.05$, $80\%$ power, you need about $m_1 \approx 170$ positives per arm — a test set of $1.7\times10^6$ examples at $\pi_1=10^{-4}$, i.e. 8.5× the one you have.

Second half of the example: take the CE model and sweep the decision threshold instead. Moving the operating point to match focal's positive-prediction rate takes CE from $8/20$ to $11/20$ as well, at a specificity cost of $0.999\to0.9986$. The loss change and a threshold change produced the same measured effect.

**What the obstruction looks like:** the reported gain is (a) inside the noise, and (b) reproducible by a free post-hoc knob. Both must be excluded before a loss can be credited, and almost no published comparison excludes either.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*