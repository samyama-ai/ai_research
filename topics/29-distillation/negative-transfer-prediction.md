---
id: 29-distillation/negative-transfer-prediction
title: "Negative Transfer Prediction"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Negative Transfer Prediction

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/negative-transfer-prediction` · **Status:** open

## 1. Problem Statement

Given a source artifact (a dataset, a pretrained checkpoint, an intermediate task, or a teacher model) and a target task, decide **before paying for the transfer** whether transferring will make the target worse than not transferring at all.

Three variants, with different difficulty:

- **Measurement.** Define negative transfer so that the sign of the effect is stable across seeds, budgets, and target metrics. Currently under-specified: the same source/target pair flips sign when the fine-tuning budget or learning rate changes.
- **Method.** Build a predictor $\hat{\Delta}$ that costs $\ll$ one transfer run and gets the *sign* right. Most published transferability scores are evaluated on *ranking* (Kendall $\tau$), not on sign, and ranking accuracy does not imply sign accuracy.
- **Theory.** Characterize the conditions on source and target distributions under which transfer provably cannot hurt, or bound the harm. Ben-David-style divergence bounds give a sufficient condition for benefit; no matching necessary condition exists at realistic scale.

Solving it means: a cheap score with calibrated sign accuracy and bounded regret against the brute-force oracle, on a held-out family of source/target pairs it was not tuned on.

## 2. Formal Setting

Target task $T$ with distribution $\mathcal{D}_T$, labelled sample $S_T \sim \mathcal{D}_T^{n}$, loss $\ell$. Source artifact $S$. Learning algorithm $\mathcal{A}$ with compute budget $B$ (steps $\times$ tokens or FLOPs), hyperparameters $\theta$, seed $r$.

Baseline (no transfer): $h_0 = \mathcal{A}(S_T; \theta, r, B)$. Transfer: $h_1 = \mathcal{A}(S_T \mid S; \theta', r, B)$.

**Transfer gain**, measured as a difference of held-out risks on a fixed test split $S_T^{\text{test}}$:

$$\Delta(S \to T) \;=\; \mathbb{E}_{r,\theta}\big[\hat{R}(h_0)\big] \;-\; \mathbb{E}_{r,\theta}\big[\hat{R}(h_1)\big], \qquad \hat{R}(h) = \tfrac{1}{m}\sum_{i=1}^{m} \ell(h(x_i), y_i).$$

**Negative transfer** is the event $\Delta < 0$. It is a decision predicate, not a scalar: the object to predict is $\mathrm{sign}(\Delta)$.

**Measured quantities.**
- $\hat\Delta$ is estimated from $k$ seeds per arm; the seed standard deviation $\sigma_r$ must be reported, since $|\Delta|$ is often $O(\sigma_r)$.
- $\theta'$ must be tuned *per arm* — comparing a tuned transfer arm to an untuned baseline is the most common way a positive result is manufactured.
- **Negative transfer rate** over a pair family $\mathcal{P}$: $\mathrm{NTR} = \frac{1}{|\mathcal{P}|}\sum_{(S,T)} \mathbf{1}[\hat\Delta < 0]$.
- **Predictor quality**: sign accuracy $\Pr[\mathrm{sign}(\hat{\Delta}_{\text{pred}}) = \mathrm{sign}(\Delta)]$, and **regret** $\Delta(S^\star \to T) - \Delta(\hat{S} \to T)$ where $S^\star$ is the oracle-best source and $\hat{S}$ the predicted-best. Regret is the number that matters; $\tau$ is not.

**Theory anchor.** Ben-David et al. (2010) bound target risk of a hypothesis trained on source-weighted data by
$$\epsilon_T(h) \le \epsilon_S(h) + \tfrac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda,$$
with $\lambda$ the joint optimal risk. This is an upper bound only: a large $d_{\mathcal{H}\Delta\mathcal{H}}$ permits harm but does not predict it.

**Assumptions known to be violated.** (i) Fixed $\mathcal{A}$ and $\theta$ across arms — violated whenever transfer changes the optimal learning rate or schedule. (ii) A single scalar target metric — violated when transfer helps in-distribution and hurts out-of-distribution (Kumar et al., ICLR 2022). (iii) Additivity across sources — violated in multi-source mixtures, where pairwise gains do not compose. (iv) Feature-space overlap between source and target, assumed by every $\mathcal{H}\Delta\mathcal{H}$-style bound and routinely absent.

## 3. State of the Art

**Theory SOTA (established).** Ben-David et al. (*Machine Learning*, 2010) — divergence upper bound; sufficient, not necessary. Menon et al. (ICML 2021) give a bias–variance decomposition for distillation showing a teacher helps only when its class-probability estimate has lower variance than the one-hot label, which is the cleanest existing *sufficient condition* for a teacher not to hurt.

**Empirical SOTA (established, reproduced).**
- **LogME** (You et al., ICML 2021): marginal-likelihood score of target labels under source features; roughly $10^3\times$ cheaper than fine-tuning. Reproduced across vision and NLP checkpoint pools.
- **LEEP** (Nguyen et al., ICML 2020) and **NCE** (Tran et al., ICCV 2019): label-space transferability from a source classifier's predictions on target data.
- **Task2Vec / task embeddings** (Achille et al., ICCV 2019; Vu et al., EMNLP 2020): Fisher-information embeddings predict transfer better than dataset-size heuristics.
- **TAG** (Fifty et al., NeurIPS 2021): predicts multi-task groupings from inter-task gradient effects in a single training run.

**Claimed but unablated.** Most transferability scores report Kendall $\tau$ against a fine-tuning oracle on a curated pool of *good* checkpoints where the NT rate is low; sign accuracy in the negative regime is usually not reported. Agostinelli et al. (ECCV 2022) show these evaluations are unstable — the ranking of transferability metrics changes with the checkpoint pool and the target-set split. **Benchmark-number-only:** almost all published $\tau$ values, since none carry a regret figure against brute force.

## 4. What Is Known

- **Negative transfer is common, not rare.** Pruksachatkun et al. (ACL 2020) ran 11 intermediate tasks $\times$ 10 target tasks with RoBERTa-large; a large fraction of pairs were neutral-to-negative, and probing-task correlations explained little of the variance. Vu et al. (EMNLP 2020), 33 NLP tasks, found the sign depends heavily on target-set size.
- **Bigger teachers can hurt students.** Cho & Hariharan (ICCV 2019) showed on CIFAR-100 and ImageNet that student accuracy is non-monotone in teacher accuracy: a stronger teacher can produce a worse student because of the capacity gap. Mirzadeh et al. (AAAI 2020) reproduced this and inserted a teacher assistant to close the gap.
- **Distillation often fails to match the teacher even when it "works."** Stanton et al. (NeurIPS 2021) show student–teacher agreement stays low on ImageNet-scale models even when student accuracy improves — an optimization failure, not a capacity failure.
- **Capacity contention at scale.** The "curse of multilinguality" (Conneau et al., ACL 2020, XLM-R): past a language count fixed by model capacity, adding languages lowers per-language performance. Negative transfer here is a resource-allocation effect, not a distribution-mismatch effect.
- **Transfer gain follows a power law in target data.** Hernandez et al. (2021, arXiv:2102.01293) fit "effective data transferred" as a power law in target dataset size and pretraining size; the benefit shrinks as the target set grows, and they document *ossification*, where pretraining slows later target learning.
- **Fine-tuning can be locally harmful.** Kumar et al. (ICLR 2022): full fine-tuning distorts pretrained features and underperforms linear probing out-of-distribution; LP-FT recovers both. Same checkpoint, same target — sign of $\Delta$ depends on which distribution you evaluate on.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed definition of $\Delta$ that fixes hyperparameter tuning per arm, seed count, and the evaluation distribution. Without that, published NT rates are not comparable across papers. This is the binding constraint.
- **Empirically open.** No study reports **sign accuracy and regret** of transferability scores on a pair family deliberately enriched in negative pairs. The experiment is runnable today; nobody has run it because building the oracle costs $|\mathcal{P}| \times k$ fine-tuning runs.
- **Empirically open.** Whether pairwise $\Delta$ predicts mixture-level $\Delta$ in multi-source pretraining. DoReMi (Xie et al., NeurIPS 2023) and DsDm (Engstrom et al., ICML 2024) optimize mixtures without ever testing pairwise additivity.
- **Theoretically open.** No necessary-and-sufficient condition for $\Delta < 0$ for gradient-trained overparameterized networks. Existing bounds are one-sided; no lower bound on harm exists outside convex/linear settings.
- **Theoretically open.** Whether sign prediction is achievable at cost $o(\text{one transfer run})$ in the worst case, or whether it is as hard as running the transfer.

## 6. Why It Is Hard

**The obstruction is that the effect size is comparable to the measurement noise, and the ground truth is defined by the thing you are trying to avoid computing.**

Concretely:
1. **Non-identifiability of the arm.** $\Delta$ is defined relative to a baseline that itself depends on $\theta$. A negative $\Delta$ under one learning rate becomes positive under another, so $\mathrm{sign}(\Delta)$ is a property of the (source, target, algorithm, budget) tuple, not of the pair. Predictors that take only $(S,T)$ as input are predicting an underdetermined quantity.
2. **Seed variance swamps the signal.** Typical reported gains on small GLUE-scale targets are 0.5–2 points against seed standard deviations of similar magnitude. Detecting the sign needs tens of seeds per arm.
3. **Absent ground truth at scale.** The oracle is brute force. For $N$ sources it is $O(N)$ full runs, and for mixtures $O(2^N)$ — the reason Standley et al. (ICML 2020) could only exhaust groupings over five Taskonomy tasks.
4. **Evaluation mismatch.** Transferability metrics are scored by $\tau$ over mostly-positive pools. A metric can hold $\tau = 0.8$ and still be at chance on the sign in the negative regime — which is the only regime the practitioner needs.

## 7. Current Research (as of 2026)

- **Data-attribution routes to transfer prediction.** Datamodels (Ilyas et al., ICML 2022), DsDm (Engstrom et al., ICML 2024), LESS (Xia et al., ICML 2024) predict per-target effects of training subsets, which is negative-transfer prediction at the example level. MIT Madry Lab; Stanford (Percy Liang's group). *(frontier — verify: whether example-level attributions aggregate to source-level sign predictions.)*
- **Mixture-law fitting.** Scaling-law-style regressions that predict target loss from mixture weights, extending DoReMi. *(frontier — verify.)*
- **Distillation-side capacity-gap theory.** Formal accounts of when a teacher's soft labels reduce student variance, extending Menon et al. (ICML 2021).
- **Model-merging interference.** Task-arithmetic and merging work (TIES, DARE lineage) measures parameter-space interference, an alternative negative-transfer proxy that skips the fine-tuning oracle. *(frontier — verify.)*
- **Reliability audits** of transferability metrics following Agostinelli et al. (ECCV 2022).

## 8. Concrete Next Experiment

**Question.** Does any cheap transferability score beat the trivial "always transfer" rule on *sign accuracy* in a negative-enriched pair family?

**Scale.** RoBERTa-base (125M). $12$ intermediate source tasks $\times$ $6$ target tasks $= 72$ pairs, targets deliberately including small/high-variance ones (CoLA, RTE, MRPC) where NT is frequent. Per arm: **20 seeds**, learning rate tuned independently per arm over a fixed 4-point grid. Total $\approx (72+6) \times 20 \times 4 \approx 6{,}240$ target-side fine-tuning runs; at ~10 GPU-minutes each on an A100 this is ~1,000 GPU-hours — one week on 8 GPUs. Report $\hat\Delta$ with bootstrap CIs; label a pair negative only when the 95% CI excludes 0.

**Control arms.** (a) Always transfer. (b) Random sign at the base rate. (c) Source-size heuristic. Candidate predictors: LogME, LEEP, NCE, Task2Vec cosine, TAG-style gradient similarity.

**The deciding number.** **Balanced sign accuracy on the confidently-labelled pairs.** If no score exceeds the always-transfer control by $\ge 10$ points balanced accuracy with a 95% CI excluding zero, then transferability metrics do not predict negative transfer — they rank good sources among good sources — and the field should stop reporting $\tau$ as evidence they do. Secondary number: regret against the oracle-best source, in target metric points.

## 9. Key References

- **[Foundational]** Rosenstein, Marx, Kaelbling, Dietterich. *To Transfer or Not To Transfer.* NIPS 2005 Workshop on Inductive Transfer.
- **[Foundational]** Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan. *A Theory of Learning from Different Domains.* Machine Learning 79(1–2), 2010.
- **[Foundational]** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NIPS 2014 Deep Learning Workshop — arXiv:1503.02531.
- **[SOTA]** You, Liu, Wang, Long. *LogME: Practical Assessment of Pre-trained Models for Transfer Learning.* ICML 2021 — arXiv:2102.11005.
- **[SOTA]** Nguyen, Hassner, Seeger, Archambeau. *LEEP: A New Measure to Evaluate Transferability of Learned Representations.* ICML 2020 — arXiv:2002.12462.
- **[SOTA]** Fifty, Amid, Zhao, Yu, Anil, Finn. *Efficiently Identifying Task Groupings for Multi-Task Learning.* NeurIPS 2021 — arXiv:2109.04617.
- **[SOTA]** Engstrom, Feldmann, Madry. *DsDm: Model-Aware Dataset Selection with Datamodels.* ICML 2024 — arXiv:2401.12926.
- **[Empirical]** Pruksachatkun, Phang, Liu, et al. *Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work?* ACL 2020.
- **[Empirical]** Vu, Wang, Munkhdalai, et al. *Exploring and Predicting Transferability across NLP Tasks.* EMNLP 2020.
- **[Empirical]** Cho, Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV 2019.
- **[Empirical]** Stanton, Izmailov, Kirichenko, Alemi, Wilson. *Does Knowledge Distillation Really Work?* NeurIPS 2021 — arXiv:2106.05945.
- **[Empirical]** Kumar, Raghunathan, Jones, Ma, Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022 — arXiv:2202.10054.
- **[Empirical]** Hernandez, Kaplan, Henighan, McCandlish. *Scaling Laws for Transfer.* 2021 — arXiv:2102.01293.
- **[Theory]** Menon, Rawat, Reddi, Kim, Kumar. *A Statistical Perspective on Distillation.* ICML 2021.
- **[Reliability]** Agostinelli, Pándy, Uijlings, Mensink, Ferrari. *How Stable are Transferability Metrics Evaluations?* ECCV 2022.
- **[Survey]** Zhang, Deng, Lu, Wu. *A Survey on Negative Transfer.* IEEE/CAA Journal of Automatica Sinica, 2023 — arXiv:2009.00909.
- **[Survey]** Wang, Dai, Póczos, Carbonell. *Characterizing and Avoiding Negative Transfer.* CVPR 2019 — arXiv:1811.09751.

## 10. Worked Example

**Setting.** Target: CoLA, metric Matthews correlation (MCC), $n = 8.5$k train. Source candidate: an intermediate MNLI-tuned RoBERTa-base checkpoint. Question: is $\Delta(\text{MNLI}\to\text{CoLA}) < 0$?

**Step 1 — the effect we care about.** Practitioners switch sources for gains of about 1 MCC point. So set the target effect $\delta = 1.0$.

**Step 2 — the noise.** CoLA MCC seed standard deviation for RoBERTa-base is roughly $\sigma_r \approx 1.5$ points across random restarts (fine-tuning instability on small GLUE tasks is well documented). So $\delta / \sigma_r \approx 0.67$.

**Step 3 — seeds needed.** Two-sample comparison, $\alpha = 0.05$, power $0.8$:

$$k \;=\; \frac{2(z_{0.975}+z_{0.80})^2 \sigma_r^2}{\delta^2} \;=\; \frac{2 \times (1.96+0.84)^2 \times 2.25}{1.0} \;\approx\; 35.3 \;\Rightarrow\; k = 36 \text{ seeds per arm}.$$

**Step 4 — cost of one ground-truth label.** Two arms $\times$ 36 seeds $= 72$ runs, $\times 4$ learning rates for honest per-arm tuning $= 288$ runs. At ~10 GPU-minutes each: **~48 GPU-hours to determine the sign for a single (source, target) pair.**

**Step 5 — the obstruction made visible.** LogME scores this pair in about 3 GPU-seconds — roughly $6 \times 10^4\times$ cheaper. But the label it is being validated against costs 48 GPU-hours, so published validations use $k = 3$ seeds instead of 36. At $k = 3$ the standard error of $\hat\Delta$ is $\sigma_r\sqrt{2/3} \approx 1.2$ points, wider than the 1.0-point effect. **The ground-truth labels used to train and evaluate negative-transfer predictors are themselves at chance on the effects those predictors are meant to detect.** A score that reports $\tau = 0.8$ against such labels has demonstrated agreement with noise-dominated targets in the near-zero regime, and nothing about the sign.

That is the loop to break: either drive $\sigma_r$ down (fixed init, deterministic data order, averaged checkpoints) so $k = 3$ suffices, or accept the ~1,000 GPU-hour bill in §8 and build one properly-powered oracle set that everyone reuses.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*