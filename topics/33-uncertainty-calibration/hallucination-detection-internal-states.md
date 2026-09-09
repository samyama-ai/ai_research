---
id: 33-uncertainty-calibration/hallucination-detection-internal-states
title: "Detecting Confidently Wrong Hallucinations from Internal States"
topic: 33-uncertainty-calibration
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Confidently Wrong Hallucinations from Internal States

> **Topic:** Uncertainty & Calibration · **ID:** `33-uncertainty-calibration/hallucination-detection-internal-states` · **Status:** empirically-open

## 1. Problem Statement

A language model produces a fluent, factually false statement and assigns it high likelihood. The question: do the model's hidden activations contain a linearly or cheaply decodable signal that separates these **confidently wrong** generations from confidently correct ones — a signal not already available from the output distribution?

The sharpening matters. Almost every published "internal states know when it's lying" result is scored on a pool that mixes low-confidence and high-confidence errors. Token log-probability alone separates that pool well. The open problem is the **residual**: detection power on the confidence-matched subset where output-level uncertainty is uninformative.

Three variants, different difficulty:

- **Measurement.** Define "hallucination" so a label is assignable per generation, per claim, at scale, without human annotation of every item. Currently the weakest link.
- **Method.** Build $g(h) \to [0,1]$ from activations that beats a logit/sampling baseline out of distribution. Empirically open.
- **Theory.** Prove whether a truth-relevant direction must exist in the residual stream of a model trained by next-token prediction on a corpus containing both true and false text, and whether it is identifiable without labels. Theoretically open; the unsupervised case has a known negative result on identifiability.

Solved would mean: a detector, trained on one task family, that adds $\ge 0.05$ AUROC over the best output-level baseline on confidence-matched held-out errors, across $\ge 3$ model families, reproduced independently.

## 2. Formal Setting

Model $M_\theta$, prompt $x$, greedy or sampled generation $y = (y_1,\dots,y_T)$. Residual-stream activation at layer $\ell$, position $t$: $h_t^{(\ell)} \in \mathbb{R}^d$, $\ell \in \{1,\dots,L\}$. The detector reads a fixed pooling $\phi(y,x) \in \mathbb{R}^{d}$ — commonly the last-prompt-token state, the last generated token, or the **exact-answer token** (the span carrying the factual claim, located by a separate extraction step).

**Label.** An oracle $O(x,y) \in \{0,1\}$, $1$ = unsupported. As measured, $O$ is one of: exact match against a QA gold set; FActScore-style per-atomic-fact NLI against a retrieved corpus (Min et al., EMNLP 2023); or an LLM judge. All three are noisy; label noise is *correlated with* the feature being probed, since judges and generators share pretraining data.

**Output-level confidence.**
$$c_{\text{lp}}(y) = \tfrac{1}{T}\sum_{t=1}^{T}\log p_\theta(y_t \mid x, y_{<t}), \qquad c_{\text{pT}}(y) = p_\theta(\texttt{"True"} \mid \texttt{prompt}(x,y))$$
and semantic entropy over $K$ samples clustered by bidirectional entailment into classes $c \in \mathcal{C}$:
$$H_{\text{sem}}(x) = -\sum_{c \in \mathcal{C}} \hat p(c\mid x)\log \hat p(c \mid x), \quad \hat p(c\mid x) = \tfrac{1}{K}\sum_{k=1}^K \mathbb{1}[y^{(k)} \in c].$$

**Detector and target.** $g_w(\phi) = \sigma(w^\top \phi + b)$, fit by logistic regression on $n$ labelled generations. The headline quantity is not AUROC but the **confidence-matched increment**
$$\Delta\text{AUROC} = \text{AUROC}\big(g_w; \mathcal{D}_\tau\big) - \max_{b \in \mathcal{B}} \text{AUROC}\big(b; \mathcal{D}_\tau\big), \quad \mathcal{D}_\tau = \{(x,y) : c_{\text{lp}}(y) > \tau\},$$
with $\tau$ the top-quartile threshold and $\mathcal{B} = \{c_{\text{lp}}, c_{\text{pT}}, -H_{\text{sem}}\}$. Deployment metric is area under the risk–coverage curve: at coverage $\kappa$, selective risk $R(\kappa) = \mathbb{E}[O \mid g_w \text{ in bottom } \kappa]$.

**Assumptions, and which fail.**
1. *Binary truth.* Fails for hedged, partially-true, and underspecified generations — a large fraction of long-form output.
2. *Label independence from the probe.* Fails when the judge is the same model family.
3. *A single truth direction transfers across tasks.* Known violated: probes trained on one dataset lose most of their advantage on another (Orgad et al., ICLR 2025).
4. *Errors are knowledge failures.* Fails: some errors are sampling accidents where the model does know the answer (Simhi et al., 2025), and these have different internal signatures.

## 3. State of the Art

**Established (reproduced, ablated).**
- Linear probes on hidden states beat chance at true/false classification on curated statement sets — SAPLMA (Azaria & Mitchell, EMNLP Findings 2023), *Geometry of Truth* (Marks & Tegmark, COLM 2024). Marks & Tegmark show the direction is causally usable: patching along it flips model assertions.
- Self-evaluation $p(\text{True})$ is well calibrated at scale (Kadavath et al., 2022), and improves with model size.
- Semantic entropy beats naive sequence entropy for detecting confabulations (Farquhar et al., *Nature* 630, 2024).
- Probes trained to predict semantic entropy from a single forward pass (SEPs; Kossen et al., 2024) retain most of semantic entropy's power at ~$1/K$ the cost, and degrade more gracefully out of distribution than accuracy probes.

**Claimed but unablated, or benchmark-number-only.**
- INSIDE / EigenScore (Chen et al., ICLR 2024): covariance-eigenvalue score over sampled generations' embeddings, reported to beat SelfCheckGPT on CoQA/NQ with LLaMA-7B/13B. Exists as a benchmark table; not confidence-matched, limited independent replication.
- MIND (Su et al., 2024): unsupervised real-time detection from internal states, auto-labelled training data. Same caveat.
- "Exact answer token" probing (Orgad et al., ICLR 2025): large in-task AUROC gains, but the same paper reports the generalization failure — this is the most honest current picture.
- Entity-awareness directions (Ferrando et al., ICLR 2025): SAE latents encoding "I know this entity", causally steerable to induce or suppress refusal. Mechanistic, single-model-family.

**Negative result.** Levinstein & Herrmann (*Philosophical Studies*, 2024): SAPLMA-style probes fail systematically on negation, and CCS (Burns et al., ICLR 2023) is not identified — the consistency objective admits many non-truth features as solutions.

## 4. What Is Known

- **Scale of the linear signal.** SAPLMA reports ~0.71–0.83 accuracy across topic-held-out statement sets on OPT-6.7B and LLaMA-30B, versus a BERT-embedding baseline near chance-plus.
- **Semantic entropy.** Farquhar et al. (2024) report mean AUROC ≈ 0.79 versus ≈ 0.69 for naive entropy, averaged over ~30 task/model combinations spanning LLaMA-2-70B, Falcon-40B and Mistral-7B, with $K \approx 10$ samples.
- **Layer/position structure.** Truth signal peaks in middle-to-late layers (roughly $0.5L$–$0.8L$) and concentrates on the answer span, not the sequence end — replicated in Marks & Tegmark (COLM 2024) and Orgad et al. (ICLR 2025) at 7B–70B.
- **Generalization gap.** Orgad et al. measure probe transfer across TriviaQA, HotpotQA, Winobias, math and NLI tasks on Mistral-7B and Llama-3-8B: within-task probing is strong, cross-task transfer drops to near the output-baseline level except between tasks sharing skill type.
- **Error taxonomy matters.** The model's internals can encode the correct answer while the sampled output is wrong — Orgad et al. show a probe can select the right answer at rates above the model's own generation accuracy, meaning "knows-but-says-otherwise" is a real and separable class.

## 5. What Is Not Known

- **Empirically open.** No published study reports $\Delta\text{AUROC}$ on a confidence-matched, out-of-distribution subset across three model families at $\ge 30$B. The experiment is runnable today on open weights; nobody has run it in this form. This is the central gap.
- **Empirically open.** Whether $\Delta\text{AUROC}$ grows, shrinks, or is flat with model scale from 7B to 400B. Two plausible stories (bigger models represent truth more explicitly / bigger models are better calibrated so the residual shrinks) both have advocates and no data.
- **Theoretically open.** Whether next-token pretraining on a mixed-veracity corpus *must* induce a low-rank truth-relevant subspace. No existence proof, no impossibility proof.
- **Theoretically open (partly resolved negatively).** Unsupervised identification: CCS's objective is provably underdetermined; whether *any* label-free objective identifies truth rather than a correlated feature (plausibility, sentiment, "the kind of thing the prompt wants") is unresolved.
- **Methodologically blocked.** A per-claim hallucination label for long-form generation that is cheap, reproducible, and not produced by a model sharing the generator's priors. FActScore needs a retrieval corpus and a judge; both leak.

## 6. Why It Is Hard

**Confounded measurement, primarily.** Reported AUROCs are computed on pools where output-level confidence is already highly predictive. A probe reading hidden states partly reconstructs $c_{\text{lp}}$ — logits are a linear readout of the final residual stream — so probe score and confidence are correlated by construction. The published number therefore does not isolate the claimed capability. Almost no paper reports the baseline-conditioned increment.

**Absent ground truth** compounds it. The oracle is a judge model or a retrieval-plus-NLI pipeline with its own error rate of several percent — comparable to the effect size being measured on the high-confidence subset, where the positive base rate is under 10%.

**Non-identifiability** blocks the unsupervised route: consistency-style objectives have many minima, and the found direction may encode "asserted with conviction" rather than "true".

Compute is *not* the obstruction. The deciding experiment costs on the order of a few thousand GPU-hours.

## 7. Current Research (as of 2026)

- **Probe generalization and error taxonomy.** Technion / Google (Orgad, Geva and collaborators) and follow-ups on separating ignorance from sampling error.
- **Cheap surrogates for sampling-based uncertainty.** Oxford OATML (Gal, Farquhar, Kossen) — semantic entropy probes and successors.
- **Mechanistic knowledge-awareness.** SAE-based entity-recognition latents and refusal directions (Ferrando et al.; representation-engineering line from Zou et al.).
- **Streaming / prefix detection.** Detecting the hallucination before the span is emitted, from prompt-side states (Gottesman & Geva, EMNLP 2024; Snyder et al., KDD 2024).
- *(frontier — verify)* Probes on reasoning traces of long-chain-of-thought models, where the confidently-wrong case is a confidently-wrong *intermediate step*; early reports suggest step-level probes transfer worse than answer-level ones. Treat as unsettled.

## 8. Concrete Next Experiment

**Question.** Does an internal-state probe add detection power over output-level confidence, on errors that confidence cannot flag, out of distribution?

**Scale.** Three families, three sizes: Llama-3.1-8B, Qwen2.5-32B, Llama-3.1-70B (open weights, ~2–4k A100-hours total including $K{=}10$ sampling).

**Data.** Train probes on TriviaQA, 8k generations, labels by exact match, features = exact-answer-token residual at layers $\{0.4L, 0.6L, 0.8L\}$. Evaluate on three held-out distributions: FActScore biographies (per-claim labels), a knowledge-conflict set, and GSM8K final answers. 2k items each.

**Confidence matching.** Restrict every evaluation to $\mathcal{D}_\tau$ = top-quartile $c_{\text{lp}}$. Report positive base rate; require $\ge 150$ positives per cell for a usable CI.

**Control arm.** Logistic regression on $(c_{\text{lp}}, c_{\text{pT}}, -H_{\text{sem}})$ — the same classifier form, same training data, output-level features only. Second control: probe on a *shuffled-label* target, to bound feature-dimension overfitting at $d \approx 8192$ with $n = 8000$.

**Deciding number.** $\Delta\text{AUROC}$ on $\mathcal{D}_\tau$, bootstrap 95% CI over 1000 resamples. **Decision rule: $\Delta \ge 0.05$ with CI excluding 0 in all three families and at least two of three OOD sets $\Rightarrow$ internal states carry usable residual signal. $\Delta \le 0.02$ or CI spanning 0 $\Rightarrow$ published gains are confidence reconstruction, and the field should report $\Delta$, not AUROC.**

## 9. Key References

- **[Foundational]** Amos Azaria, Tom Mitchell. *The Internal State of an LLM Knows When It's Lying.* Findings of EMNLP, 2023. — arXiv:2304.13734
- **[Foundational]** Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt. *Discovering Latent Knowledge in Language Models Without Supervision.* ICLR, 2023. — arXiv:2212.03827
- **[Foundational]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[SOTA]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 2024.
- **[SOTA]** Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov. *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations.* ICLR, 2025. — arXiv:2410.02707
- **[SOTA]** Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal. *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs.* 2024. — arXiv:2406.15927
- **[SOTA]** Chao Chen et al. *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.* ICLR, 2024. — arXiv:2402.03744
- **[Method]** Samuel Marks, Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in LLM True/False Datasets.* COLM, 2024. — arXiv:2310.06824
- **[Method]** Javier Ferrando, Oscar Obeso, Senthooran Rajamanoharan, Neel Nanda. *Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models.* ICLR, 2025. — arXiv:2411.14257
- **[Negative]** B. A. Levinstein, Daniel A. Herrmann. *Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks.* Philosophical Studies, 2024. — arXiv:2307.00175
- **[Evaluation]** Sewon Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Baseline]** Potsawee Manakul, Adian Liusie, Mark Gales. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection.* EMNLP, 2023. — arXiv:2303.08896

## 10. Worked Example

Llama-3.1-8B on 10,000 TriviaQA-style questions. Accuracy 70%, so 3,000 errors overall. A probe reports **AUROC 0.85** on the full pool. The output-level baseline $c_{\text{lp}}$ alone reports **0.80** on the same pool.

Now restrict to the top confidence quartile, $N = 2{,}500$. Error rate there is 8%: **200 positives, 2,300 negatives**. On this subset the probe's AUROC falls to **0.70** and the baseline to **0.66** — the probe's advantage is $\Delta = 0.04$, below the decision threshold.

Operating cost at that AUROC. Set the flag threshold to catch half the errors, TPR $= 0.50$; on a typical ROC of area 0.70 that costs FPR $\approx 0.20$.

$$\text{TP} = 0.50 \times 200 = 100, \qquad \text{FP} = 0.20 \times 2{,}300 = 460$$
$$\text{Precision} = \frac{100}{100 + 460} = 17.9\%$$

So the system interrupts the user 560 times to catch 100 real errors, and still misses 100. Selective risk at 78% coverage falls from 8.0% to about 5.1% — a 36% relative reduction in error, bought with a 22% abstention rate.

**Where the obstruction becomes visible.** The headline 0.85 is mostly the probe rediscovering $c_{\text{lp}}$, which is a linear function of the final residual stream the probe reads. Strip that shared component and 0.85 − 0.80 = 0.05 on the full pool collapses to 0.04 on the subset where it would matter, with a bootstrap CI over 200 positives of roughly $\pm 0.05$ — wide enough to include zero. The experiment as usually reported cannot distinguish "internal states encode truth" from "internal states encode the logits", and the confidently-wrong regime — the only regime where a detector adds anything — is exactly where the sample size is smallest and the label noise largest.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*