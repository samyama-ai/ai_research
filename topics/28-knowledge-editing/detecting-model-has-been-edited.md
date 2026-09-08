---
id: 28-knowledge-editing/detecting-model-has-been-edited
title: "Detecting Whether a Model Has Been Edited"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Whether a Model Has Been Edited

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/detecting-model-has-been-edited` · **Status:** open

## 1. Problem Statement

Given a deployed model, decide whether its weights have been altered by a targeted post-training edit — ROME, MEMIT, MEND, a LoRA patch, a GRACE codebook entry, a BadEdit backdoor — and if so, recover *what* was edited.

Three variants, with very different difficulty:

- **Measurement.** Define an observable that separates edited from unedited models. Requires stating the access model: black-box queries only, next-token logits, full weights, or full weights *plus* a trusted reference checkpoint $\theta_0$.
- **Method.** Build a detector $D$ with useful true-positive rate at a fixed false-positive rate against an *adaptive* editor who knows $D$.
- **Theory.** Determine whether edit-detection is possible in principle, or whether for any editing target there exists a fine-tuning trajectory producing an indistinguishable checkpoint — the identifiability question.

Solving it means: a detector with TPR $\geq 0.9$ at FPR $\leq 0.01$ on held-out edit methods *not seen during detector training*, on models the detector was not calibrated on, against an editor allowed to regularize against the detector. Nothing published meets this bar.

## 2. Formal Setting

Base model $\theta_0 \in \mathbb{R}^d$. An edit request is $e = (s, r, o^\ast)$ — subject, relation, target object — with a scope set $S(e)$ of prompts the edit should change and a complement $\bar{S}(e)$ it should not. An editor $\mathcal{E}$ produces $\theta_1 = \mathcal{E}(\theta_0, e)$.

**Detector.** $D: \mathcal{O} \to \{0,1\}$, where the observation $\mathcal{O}$ depends on access level $\mathcal{A}$:

$$\mathcal{A}_{\text{bb}} = \{(x_i, y_i)\}_{i=1}^n,\quad \mathcal{A}_{\text{logit}} = \{p_\theta(\cdot \mid x_i)\}, \quad \mathcal{A}_{\text{w}} = \theta, \quad \mathcal{A}_{\text{ref}} = (\theta, \theta_0).$$

**Measured quantities.**

- **Detection power.** $\mathrm{TPR}(\alpha) = \Pr_{\theta_1}[D=1]$ subject to $\Pr_{\theta_0}[D=1] \le \alpha$. Measured empirically over a paired corpus of $N$ (edited, unedited) checkpoints; the FPR estimate has standard error $\sqrt{\alpha(1-\alpha)/N}$, so $\alpha = 0.01$ needs $N \gtrsim 10^3$ clean checkpoints to be meaningful. Most published numbers use $N < 50$.
- **Edit magnitude.** For rank-one methods, $\Delta = \theta_1 - \theta_0$ is supported on one MLP down-projection $W_l \in \mathbb{R}^{d_h \times d_m}$ with $\Delta W_l = (v^\ast - W_l k^\ast) \frac{(C^{-1}k^\ast)^\top}{(C^{-1}k^\ast)^\top k^\ast}$, $C = \mathbb{E}[kk^\top]$ the covariance of MLP keys. Measured as relative Frobenius norm $\rho = \|\Delta W_l\|_F / \|W_l\|_F$.
- **Behavioural signature.** Edit-induced overconfidence: $\kappa(x) = \log p_\theta(o^\ast \mid x) - \log p_\theta(o^{(2)} \mid x)$, the logit margin over the runner-up. Measured by teacher-forcing on $|S(e)|$ paraphrases.
- **Ripple deficit.** $R(e) = \mathrm{acc}(\text{logical consequences of } e)$ under $\theta_1$ — the RippleEdits construction. A genuinely retrained model has $R \approx$ its accuracy on comparable unedited facts; an edited model does not.

**Assumptions, and which fail.**

1. *A trusted reference $\theta_0$ exists.* Violated for any model released only post-edit — the deployment case that matters.
2. *The edit is low-rank and localized.* Violated by full fine-tuning, sequential edits ($10^3$+ accumulate into a dense $\Delta$), and GRACE-style adapters that add parameters rather than change them.
3. *Edited and clean checkpoints are exchangeable apart from the edit.* Violated in practice: published "clean" controls differ from edited models in optimizer state, dtype, and provenance, so any classifier can learn the confound instead of the edit.
4. *The editor is non-adaptive.* Violated by construction in the threat model that motivates the problem.

## 3. State of the Art

**Established.**

- *Edits are behaviourally conspicuous when you know what to probe.* RippleEdits (Cohen et al., TACL 2024) shows edited models fail logical consequences of the injected fact at rates far above unedited models on facts they know. This is a real, reproduced signal — but it is a per-fact probe, not a whole-model detector, and it presumes you already guessed $e$.
- *Edits degrade general ability at scale.* Gupta et al. (Findings of ACL 2024) show sequential ROME/MEMIT editing on GPT-2-XL and Llama-2-7B causes gradual then catastrophic forgetting; Gu et al. (EMNLP 2024) show downstream-task collapse. Degradation is detectable — but only after hundreds to thousands of edits.
- *Specificity failures are measurable.* CounterFact+ (Hoelscher-Obermaier et al., Findings of ACL 2023) shows ROME's reported specificity was inflated by a benchmark that did not probe adjacent subjects; measured properly, ROME leaks into unrelated prompts.

**Claimed but unablated.**

- Hidden-state classifiers that detect edited knowledge (Youssef et al., *Detecting Edited Knowledge in Language Models*, 2024/25) report high in-distribution AUC. What is not ablated: transfer across edit method, across model family, and against an adaptive editor. Treat the numbers as in-distribution benchmark figures.
- Weight-space anomaly detection — flagging MLP layers whose singular-value spectrum has an outlier consistent with a rank-one insert — is a natural attack and is discussed informally. No paper establishes its FPR against models that were merely fine-tuned. *(frontier — verify)*

**Adjacent SOTA that is genuinely solid.** Instructional fingerprinting (Xu et al., NAACL 2024) shows a *cooperative* owner can plant a durable, queryable signature; this solves attribution, not adversarial edit-detection.

## 4. What Is Known

- ROME achieves ~99–100% efficacy on CounterFact for GPT-J 6B and GPT-2-XL 1.5B with a single rank-one update to one layer (Meng et al., NeurIPS 2022). The perturbed parameter fraction is $\sim 10^{-4}$ of the model; $\rho$ is small enough that generic weight-norm checks do not flag it.
- MEMIT scales to 10,000 simultaneous edits on GPT-J 6B and GPT-NeoX 20B (Meng et al., ICLR 2023), still confined to a handful of MLP layers.
- BadEdit (Li et al., ICLR 2024) injects a functioning backdoor using ~15 data samples and edits taking under a minute, with reported clean-task performance preserved — i.e. the standard benchmark suite is *not* a detector.
- Localization does not predict edit site: Hase et al. (NeurIPS 2023) show causal-tracing localization and successful edit location are largely uncorrelated on GPT-J 6B. Detection heuristics keyed to "where facts live" therefore have no principled anchor.
- Unlearning-verification forging (Thudi et al., USENIX Security 2022) proves that for a given final checkpoint one can construct an alternative training trajectory of comparable plausibility. The construction is about unlearning proofs, but the mechanism — trajectory non-uniqueness — applies directly here.

## 5. What Is Not Known

- **Theoretically open.** Whether, for a given $(\theta_0, e)$, there exists a benign fine-tuning run reaching a checkpoint $\theta_2$ with $\|\theta_2 - \theta_1\|$ below any detector's resolution. No lower bound on detectability, and no impossibility theorem, exists for edit-detection specifically. Thudi et al. is suggestive, not a proof for this setting.
- **Empirically open.** Cross-method, cross-scale detector transfer. Nobody has trained a detector on ROME/MEMIT edits of a 7B model and tested it on MEND/GRACE/LoRA edits of a 70B model with $N \geq 10^3$ clean controls. The experiment is runnable today for well under $10^4$ GPU-hours.
- **Methodologically blocked.** The base rate. "Fraction of deployed models that have been edited" is unknown and unestimable, so any reported TPR/FPR pair cannot be converted into a posterior. A detector at TPR 0.9 / FPR 0.01 is near-useless if the prior is $10^{-4}$.

## 6. Why It Is Hard

**Non-identifiability plus an absent control distribution.** Two obstructions, both concrete:

1. *Edit and fine-tune are not disjoint hypotheses.* Every production model is fine-tuned. The map from "$\Delta$ that changes one fact" to "$\Delta$ produced by RLHF touching that fact" is not injective, and no one has characterized the overlap. The null hypothesis "unedited" has no canonical distribution to test against.
2. *No paired ground truth at scale.* Detection requires a corpus of matched (edited, clean) checkpoints differing *only* by the edit. Constructing $10^3$ such pairs at 7B costs storage on the order of 14 TB in bf16 before any detector training. Existing work uses tens of checkpoints, so FPR at $\alpha = 0.01$ is not resolvable — the reported operating point is below the measurement floor.

Compounding both: the evaluation does not measure what it names. "Detects edits" in current papers means "separates edited from clean on the same benchmark that generated the edits", which is a within-distribution classification result, not a detection guarantee.

## 7. Current Research (as of 2026)

- **Provenance-first approaches.** Fingerprinting and weight-signature schemes (Xu et al.; HuRef, Zeng et al., NeurIPS 2024) sidestep detection by making tampering *break* a signature. Deployable now for cooperative owners; useless against a hostile editor who controls the release.
- **Behavioural auditing.** Ripple-effect and consistency probing extended to whole-model sweeps rather than per-fact checks (Technion/Google, Zhejiang/EasyEdit line). *(frontier — verify)*
- **Weight-forensics.** Spectral and covariance-residual analysis of MLP projections looking for rank-one inserts. *(frontier — verify)*
- **Safety-motivated tampering detection.** Sleeper Agents (Hubinger et al., 2024) established that a planted conditional behaviour survives standard safety training; the detection question it raises is the same one here, and remains open.

## 8. Concrete Next Experiment

**Question.** Does an edit-detector transfer across edit method?

**Scale.** Llama-3-8B. Build 500 edited checkpoints — 125 each ROME, MEMIT, MEND, LoRA-patch — each injecting 1 to 10 CounterFact edits. Cost: LoRA/MEND deltas stored as adapters, ROME/MEMIT as the modified layer only; total under 1 TB. Edit compute $< 200$ GPU-hours on 8×H100.

**Control arm — the part usually skipped.** 500 *clean* checkpoints that are not $\theta_0$: each is $\theta_0$ fine-tuned on 1–10 documents drawn from a held-out corpus for 1–20 steps at LR $10^{-6}$–$10^{-5}$. This makes the null "was fine-tuned but not edited", not "is pristine". Any detector that only beats pristine $\theta_0$ is measuring a confound.

**Protocol.** Train the detector (spectral features of every MLP down-projection + logit-margin $\kappa$ over 5,000 generic factual probes) on ROME + MEMIT only. Test on MEND + LoRA.

**Deciding number.** TPR at FPR $= 0.01$ on the held-out methods, with the 500 fine-tuned controls as the null. $\geq 0.80$ means detection generalizes and the problem is tractable. $\leq 0.30$ — chance-adjacent given the method gap — means detection is method-specific and the field should pivot to provenance. Anything between is the informative middle and licenses the adaptive-editor round.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[SOTA]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[SOTA]** Yanzhou Li et al. *BadEdit: Backdooring Large Language Models by Model Editing.* ICLR, 2024. — arXiv:2403.13355
- **[SOTA]** Paul Youssef, Zhixue Zhao, Jörg Schlötterer, Christin Seifert. *Detecting Edited Knowledge in Language Models.* 2024/2025. (identifier uncertain — omitted)
- **[Theory-adjacent]** Anvith Thudi, Hengrui Jia, Ilia Shumailov, Nicolas Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022.
- **[Empirical]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024. — arXiv:2401.07453
- **[Empirical]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[Empirical]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL, 2023.
- **[Defense]** Jiashu Xu, Fei Wang, Mingyu Derek Ma, Pang Wei Koh, Chaowei Xiao, Muhao Chen. *Instructional Fingerprinting of Large Language Models.* NAACL, 2024. — arXiv:2401.12255
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023. — arXiv:2305.13172
- **[Context]** Evan Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* 2024. — arXiv:2401.05566

## 10. Worked Example

Take GPT-J 6B and the ROME edit "The Eiffel Tower is located in **Rome**".

**Weight side.** The update touches layer 5's MLP down-projection, $W_5 \in \mathbb{R}^{4096 \times 16384}$ — 67M of 6B parameters, 1.1%. The update is rank one, so $\|\Delta W_5\|_F / \|W_5\|_F$ is on the order of $10^{-2}$ and $\|\Delta\theta\|_F/\|\theta\|_F \approx 10^{-3}$. Now fine-tune the *same* base model for 10 steps on 8 unrelated documents at LR $2\times10^{-5}$. The resulting global relative change is the same order. A weight-norm threshold that catches the edit catches the fine-tune. The only surviving discriminator is *rank*: the edit's $\Delta W_5$ has one dominant singular value, the fine-tune's does not.

**The obstruction, made visible.** That discriminator is one line for an adaptive editor to defeat. Replace the rank-one edit with the same $v^\ast$ target reached by $k=8$ steps of gradient descent on the edit prompt plus a spectral penalty $\lambda \sum_{i>1}\sigma_i(\Delta W_5)^{-1}$ that forces the update to spread across singular directions. Efficacy on the edit prompt is unchanged — the target logit is a scalar constraint, and there are $\sim 10^7$ parameters in the layer to satisfy it with. The detector's one feature is gone at no cost to the attacker.

**The base-rate arithmetic.** Suppose the spectral detector reaches TPR 0.9 / FPR 0.01 against non-adaptive edits. Screen 10,000 Hugging Face checkpoints, of which 10 are genuinely backdoored ($10^{-3}$ prior). Expected flags: $0.9 \times 10 = 9$ true, $0.01 \times 9990 \approx 100$ false. Precision 8%. Twelve manual audits per real hit — and the audit itself has no ground truth, because there is no trusted $\theta_0$ to diff against. Detection power is not the binding constraint; the missing reference checkpoint and the unknown base rate are.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*