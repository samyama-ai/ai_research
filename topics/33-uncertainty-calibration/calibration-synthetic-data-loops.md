---
id: 33-uncertainty-calibration/calibration-synthetic-data-loops
title: "Calibration Under Self-Training and Synthetic Data Loops"
topic: 33-uncertainty-calibration
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration Under Self-Training and Synthetic Data Loops

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/calibration-synthetic-data-loops` · **Status:** empirically-open

## 1. Problem Statement

A model generates data; that data (filtered, curated, or raw) enters the training set of the next model; repeat. The question is what happens to **calibration** — the agreement between a model's stated confidence and its empirical accuracy — across such generations, and whether the degradation is separable from the degradation in accuracy.

Three variants, different difficulty:

- **Measurement.** Given a chain of models $M_0, M_1, \dots, M_T$ trained under a self-consumption protocol, does calibration error grow, and at what rate in $T$? This is runnable today and mostly unrun at frontier scale.
- **Method.** Is there a training-time or data-curation intervention (accumulate rather than replace, verifier filtering, confidence-aware relabeling, abstention targets) that keeps calibration error bounded as $T \to \infty$ while accuracy still improves?
- **Theory.** Under what conditions on the sampling operator does the fixed point of the self-training map preserve the conditional distribution $p(y \mid x)$, as opposed to only the marginal support? Model collapse theory to date bounds *distributional* distance and *test loss*; it does not bound calibration error.

Solving it means: a protocol with a proof or a reproduced empirical law giving $\mathrm{ECE}(M_T)$ as a function of $T$, synthetic fraction, and filtering strength, plus a demonstration that a named intervention flattens that curve without costing accuracy.

## 2. Formal Setting

Let $p^\*(x,y)$ be the real data distribution. Generation $t$ has model $M_t$ with predictive distribution $q_t(y \mid x)$ and a *confidence* $c_t(x) = \max_y q_t(y\mid x)$ for classification, or a verbalized/token-probability score for open generation.

**Self-consumption map.** Training set at step $t+1$:
$$\mathcal{D}_{t+1} = \underbrace{(1-\alpha)\,\mathcal{D}^{\mathrm{real}}}_{\text{fresh real data}} \;\cup\; \alpha \cdot \Phi\!\left(\{(x_i, \hat y_i) : \hat y_i \sim q_t(\cdot \mid x_i)\}\right)$$
with $\alpha \in [0,1]$ the **synthetic fraction**, and $\Phi$ a **curation operator** (identity = raw loop; top-$k$ by verifier score, majority vote over $n$ samples, or reward-model threshold = curated loop). *Replacement* regimes discard $\mathcal{D}_{\le t}$; *accumulation* regimes keep it — a distinction that changes the asymptotics (§4).

**Calibration, as measured.** Binned expected calibration error with $B$ equal-mass bins over $N$ held-out points:
$$\widehat{\mathrm{ECE}}_B = \sum_{b=1}^{B} \frac{|I_b|}{N}\left| \mathrm{acc}(I_b) - \mathrm{conf}(I_b) \right|, \quad \mathrm{acc}(I_b)=\frac{1}{|I_b|}\sum_{i \in I_b}\mathbf{1}[\hat y_i = y_i]$$
Report alongside it: Brier score $\frac{1}{N}\sum_i (c_i - \mathbf{1}[\hat y_i = y_i])^2$, its Murphy decomposition into calibration and refinement terms, and negative log-likelihood — because ECE alone is estimator-biased and monotone in $B$.

**The quantity that actually matters.** Define generation-$T$ calibration drift $\Delta_T = \mathrm{ECE}(M_T) - \mathrm{ECE}(M_0)$ measured on a **fixed real-data probe set** held out of every $\mathcal{D}_t$. Distinguish from *sharpness collapse*: a chain can improve ECE by getting less confident and less accurate together.

**Assumptions, and which break.**
1. *The probe set stays out-of-distribution-free.* Violated once $M_t$'s outputs seed public web data that later re-enters evaluation corpora.
2. *Labels $y$ are well defined.* Holds for MMLU-style multiple choice; fails for open generation, where "correctness" is itself a model judgment and the judge shares the loop's biases.
3. *$\Phi$ is independent of $q_t$.* Violated by design in any verifier-filtered loop — the filter is usually the model itself.
4. *IID sampling of $x_i$.* Violated when prompts are themselves synthesized.

## 3. State of the Art

**Established (theory).** Shumailov et al. (*Nature*, 2024) show recursive training on generated data drives tail loss and variance collapse in Gaussian and language settings. Dohmatob et al. (*A Tale of Tails*, ICML 2024) derive modified scaling laws under synthetic contamination — a finite-loss floor that fresh-data scaling cannot cross. Gerstgrasser et al. (2024, arXiv:2404.01413) prove that **accumulating** real and synthetic data bounds the test error, whereas *replacing* diverges linearly in $T$ for linear regression. Bertrand et al. (ICLR 2024) give stability conditions for iterative retraining under a sufficient real-data fraction. All bound *loss or distributional distance*; **none states a calibration bound**.

**Established (empirical, calibration-adjacent).** Guo et al. (ICML 2017) established that modern networks are systematically overconfident and that temperature scaling recovers most of it. The GPT-4 technical report (OpenAI, 2023) reports MMLU ECE of $0.007$ for the pre-trained model and $\approx 0.074$ post-RLHF — a 10× degradation from a *post-training* loop that shares the structure of self-training.

**Claimed but unablated.** That self-improvement pipelines (STaR, Zelikman et al., NeurIPS 2022; Huang et al., EMNLP 2023) preserve calibration — these papers report accuracy gains and do not report ECE or Brier at all. That verifier filtering "fixes" collapse: filtering is shown to preserve or improve *quality* (Ferbach et al., NeurIPS 2024, on curated self-consuming loops optimizing preferences) but curation provably sharpens toward the reward's argmax, which is a mechanism for *worse* calibration, unmeasured.

**Benchmark-number-only.** Reported diversity collapse under synthetic text training (Guo et al., *The Curious Decline of Linguistic Diversity*, NAACL Findings 2024) is a set of corpus-level diversity metrics on specific models; it has no calibration axis and does not generalize across scales as stated.

## 4. What Is Known

- **Replacement diverges, accumulation does not.** Linear-regression analysis (Gerstgrasser et al. 2024): test error under replacement grows linearly in the number of generations; under accumulation it is bounded by a constant independent of $T$. Verified empirically at GPT-2 scale (125M) on language modeling and for diffusion models on CIFAR-style data.
- **Collapse is measurable in ~5–10 generations at small scale.** Shumailov et al. fine-tuned OPT-125M on wikitext2 across generations and observed rising perplexity and disappearing distribution tails within single-digit generation counts. This is the scale nearly every collapse result is measured at: $10^8$ parameters, not $10^{11}$.
- **Post-training loops damage calibration in the direction predicted.** GPT-4: ECE $0.007 \to 0.074$ on MMLU after RLHF (OpenAI 2023, Figure 8). Single generation, one model, no ablation isolating which stage did it.
- **ECE is a biased estimator.** Kumar, Liang, Ma (*Verified Uncertainty Calibration*, NeurIPS 2019) show binned ECE systematically underestimates true calibration error and that the bias depends on $B$ and $N$ — so cross-generation ECE comparisons with differing accuracy are not automatically comparable.
- **Conformal prediction gives distribution-free coverage** (Vovk et al. 2005; Angelopoulos & Bates 2023) — but only under exchangeability between calibration and test sets, which a self-training loop breaks by construction.

## 5. What Is Not Known

- **Theoretically open.** No theorem bounds $\Delta_T$ under any self-consumption map. The accumulation results bound $L_2$ risk; risk can stay bounded while the calibration component of the Brier decomposition grows and the refinement component shrinks in compensation. Whether curated loops ($\Phi \neq \mathrm{id}$) have a fixed point that preserves $p(y\mid x)$ rather than only $\arg\max_y p(y\mid x)$ is open.
- **Empirically open (the main gap).** Nobody has run a controlled generation chain at $\geq 7$B parameters with calibration as the *primary* logged metric, holding accuracy and data volume fixed across arms. The experiment is entirely runnable; it costs GPU-months, and the incentive gradient in the field points at accuracy.
- **Methodologically blocked.** Calibration for open-ended generation. Verbalized confidence (Tian et al., EMNLP 2023) and self-evaluation logits (Kadavath et al., 2022) both need a correctness label that, at generation $t$, is supplied by a model inside the loop. There is no agreed loop-independent correctness oracle for free-form text.

## 6. Why It Is Hard

**Confounded measurement, three ways.**

1. *Accuracy–calibration entanglement.* $\mathrm{ECE}$ is not scale-free in accuracy. If $M_T$ is less accurate, an unchanged confidence distribution mechanically yields worse ECE. Attributing $\Delta_T$ to the loop rather than to accuracy loss requires matched-accuracy arms — which requires deliberately handicapping the control, which nobody budgets for.
2. *The curation operator is the model.* $\Phi$ built from $M_t$'s own scores makes the training distribution a function of the very confidences under study. Non-identifiable from observational chains: you cannot tell "confidence became miscalibrated" from "the data was selected by confidence" without randomizing $\Phi$.
3. *No clean probe set.* The obvious probes (MMLU, ARC, GSM8K) are in the pretraining corpora of every model in the chain, and by 2026 partly synthetic themselves. Contamination moves ECE in an unknown direction.

Plus a cost floor: an $N$-generation chain at 7B costs $N$ full pretraining or fine-tuning runs, and the effect being measured (ECE moving from 0.02 to 0.06) is smaller than run-to-run seed variance at small $N$.

## 7. Current Research (as of 2026)

- **Collapse theory groups** — Dohmatob/Feng/Kempe (NYU, Meta FAIR) on modified scaling laws; the Stanford/Constellation line on accumulation vs. replacement; Alemohammad et al. (Rice, *Self-Consuming Generative Models Go MAD*, ICLR 2024) on the image side. Extension of these results to a calibration functional is the obvious open lane. *(frontier — verify current status)*
- **Curated / preference-shaped loops** — Ferbach, Bertrand, Gidel et al. on self-consuming models with curated data; the theory says curation steers toward preference maxima, which predicts sharpening. Nobody has instrumented that prediction with ECE. *(frontier — verify)*
- **Hallucination-as-miscalibration** — Kalai and Vempala's 2025 argument that evaluation scoring which penalizes abstention trains overconfidence. This is the mechanism most likely to compound across generations: if generation $t$'s outputs are filtered by a scorer that rewards guessing, generation $t+1$ inherits the guess-rather-than-abstain prior. *(frontier — verify)*
- **Provenance and watermarking** as an experimental control — labeling synthetic fraction in real corpora so $\alpha$ becomes observable rather than assumed.

## 8. Concrete Next Experiment

**Scale.** Pythia-1.4B or Llama-3.2-1B as $M_0$. Five generations. Task: 4-way multiple choice over a held-out probe of 20k questions (ARC-Challenge + MMLU subsets, decontaminated by 13-gram overlap against every $\mathcal{D}_t$). Each generation: fine-tune on 2B tokens. Total ≈ 4 arms × 5 generations × 1 GPU-week ≈ 20 GPU-weeks on 8×A100.

**Arms.**
- **A (control): fresh real data only**, 2B new real tokens per generation. Isolates drift from repeated fine-tuning alone.
- **B: replacement loop**, $\alpha = 1$, $\Phi = \mathrm{id}$.
- **C: accumulation loop**, $\alpha = 0.5$, data accumulated, $\Phi = \mathrm{id}$.
- **D: curated loop**, $\alpha = 0.5$, $\Phi = $ majority-vote-of-8 self-consistency filter.

All arms matched on token count and optimizer steps. Log per generation: accuracy, $\widehat{\mathrm{ECE}}_{15}$ (equal-mass), Brier with Murphy decomposition, mean confidence, and the debiased calibration estimator of Kumar et al. (2019). Three seeds.

**The deciding number.** The **matched-accuracy calibration gap at generation 5**: $\Delta_5^{\mathrm{match}} = \mathrm{ECE}(D_5) - \mathrm{ECE}(A_5)$, evaluated on the subset of the probe where arms A and D have equal accuracy (±1 point), so accuracy cannot explain the difference. If $\Delta_5^{\mathrm{match}} > 0.03$ with seed spread under $0.01$, curated self-training degrades calibration independent of accuracy, and the field's default "filter and iterate" recipe carries a hidden cost. If $\Delta_5^{\mathrm{match}} \approx 0$ while B collapses, then the problem reduces to the already-known replacement/accumulation distinction and calibration needs no separate theory.

## 9. Key References

- **[Foundational]** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.
- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[SOTA]** Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* 2024. — arXiv:2404.01413
- **[SOTA]** Elvis Dohmatob, Yunzhen Feng, Pu Yang, François Charton, Julia Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML, 2024. — arXiv:2402.07043
- **[SOTA]** Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, Richard G. Baraniuk. *Self-Consuming Generative Models Go MAD.* ICLR, 2024. — arXiv:2307.01850
- **[SOTA]** Ananya Kumar, Percy Liang, Tengyu Ma. *Verified Uncertainty Calibration.* NeurIPS, 2019. — arXiv:1909.10155
- **[Method]** Quentin Bertrand, Avishek Joey Bose, Alexandre Duplessis, Marco Jiralerspong, Gauthier Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* ICLR, 2024.
- **[Method]** Damien Ferbach, Quentin Bertrand, Avishek Joey Bose, Gauthier Gidel. *Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences.* NeurIPS, 2024.
- **[Method]** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS, 2022. — arXiv:2203.14465
- **[Empirical]** Saurav Kadavath, Tom Conerly, Amanda Askell, et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Empirical]** Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, et al. *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* EMNLP, 2023.
- **[Empirical]** OpenAI. *GPT-4 Technical Report.* 2023. — arXiv:2303.08774 (MMLU ECE 0.007 pre-training vs. 0.074 post-RLHF)
- **[Survey]** Anastasios N. Angelopoulos, Stephen Bates. *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in Machine Learning, 2023.
- **[Related]** Yanzhu Guo, Guokan Shang, Michalis Vazirgiannis, Chloé Clavel. *The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text.* Findings of NAACL, 2024.

## 10. Worked Example

A 4-way multiple-choice probe, 10,000 questions. Generation 0: accuracy $0.60$, mean confidence $0.62$, $\widehat{\mathrm{ECE}}_{15} = 0.02$.

Run a curated loop: sample 8 answers per training question, keep the majority answer, fine-tune. Majority voting is a sharpening operator — at accuracy $0.60$ per sample, the majority-of-8 label is right about $70\%$ of the time, but it arrives as a **hard label**, so the next model is trained toward confidence $1.0$ on data that is $30\%$ wrong.

Generation 1: accuracy rises to $0.65$ (self-consistency distillation works). Mean confidence rises to $0.80$. Now
$$\widehat{\mathrm{ECE}} \approx |0.65 - 0.80| = 0.15$$
if the gap is roughly uniform across bins. Accuracy is **up 5 points** and calibration error is **up 7.5×**.

The obstruction is visible right here. Every metric the pipeline's authors report — accuracy, majority-vote agreement, downstream task score — improved. The one that got worse is not logged. And you cannot recover the truth by comparing to generation 0, because accuracy changed: some of the $0.15$ is the loop making the model overconfident, some is arithmetic from a different accuracy level. Separating them needs the matched-accuracy subset of §8. Without it, the effect is real, unmeasured, and invisible to the pipeline that caused it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*