---
id: 28-knowledge-editing/formal-unlearning-guarantee-editing
title: "Formal Guarantee of Unlearning by Editing"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Formal Guarantee of Unlearning by Editing

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/formal-unlearning-guarantee-editing` · **Status:** open

## 1. Problem Statement

A weight-editing method (ROME, MEMIT, RMU, task-vector negation, targeted fine-tuning) takes a trained model $\theta$ and a target set of facts or documents $D_f$ and returns $\theta'$ that no longer emits the target content. The open problem: **produce a certificate that $\theta'$ is indistinguishable from a model that never saw $D_f$**, under a stated adversary class, with the certificate holding for non-convex transformer training.

Three variants, different difficulty:

- **Measurement.** Given $(\theta, \theta', D_f)$, decide whether $D_f$'s influence is gone. Currently blocked: no accepted estimator of the indistinguishability parameter, and Thudi et al. (2022) show weights alone cannot carry the proof.
- **Method.** Build an editing operator whose output provably satisfies an $(\varepsilon,\delta)$ removal guarantee at 7B+ scale without retraining. Open; all current certified constructions require convexity or a noise budget that destroys utility at scale.
- **Theory.** Prove or refute: for some non-trivial edit class over non-convex $\theta$, a bounded-cost edit achieves $(\varepsilon,\delta)$-removal with utility loss $o(1)$. Theoretically open.

Solving it means: a checkable statement of the form "no adversary in class $\mathcal{A}$ with budget $B$ can distinguish $\theta'$ from retrain-without-$D_f$ with advantage $>\alpha$", with $\alpha$ either proved or lower-bounded by an audit whose power is itself quantified.

## 2. Formal Setting

Let $D = \{z_1,\dots,z_n\}$ be the training corpus, $A$ a randomized training algorithm, $\theta = A(D)$. Let $D_f \subset D$ be the forget set, $|D_f| = m$, and $D_r = D \setminus D_f$. An **unlearning-by-editing** operator is $U(\theta, D_f, D_r) \to \theta'$ with cost $C(U) \ll C(A)$ (typically $10^{-4}$–$10^{-6}$ of pretraining FLOPs).

**Certified removal** (Guo et al., ICML 2020). $U$ is $(\varepsilon,\delta)$-certified if for all measurable $S \subseteq \Theta$,

$$\Pr[U(A(D),D_f,D_r) \in S] \le e^{\varepsilon}\Pr[A(D_r) \in S] + \delta$$

and symmetrically. The retrain-from-scratch distribution $A(D_r)$ is the reference; $\varepsilon$ is not observed but bounded analytically.

**Gradient residual.** For empirical risk $L(\theta;D_r)=\sum_{z\in D_r}\ell(\theta;z)$, the certificate in Guo et al. runs a Newton step $\theta' = \theta + H^{-1}\nabla$ and bounds $\|\nabla L(\theta';D_r)\|_2 \le \gamma$ (measured directly by one backward pass over $D_r$), then adds noise $b \sim \mathcal{N}(0,\sigma^2 I)$ with $\sigma = \gamma\sqrt{2\ln(1.5/\delta)}/\varepsilon\lambda$. $\gamma$ is the only empirical quantity; everything else is closed-form, and $\gamma \le \frac{m^2 C^2 L_H}{\lambda^2 n^2}$ requires $\ell$ to be $\lambda$-strongly convex with $L_H$-Lipschitz Hessian.

**Deletion capacity** (Sekhari et al., NeurIPS 2021): the largest $m$ such that excess population risk stays $\le 0.01$; for convex losses $m = \Omega\!\left(n\varepsilon/\sqrt{d\log(1/\delta)}\right)$, with $d$ the parameter count.

**Measured audit lower bound.** In practice one estimates, not $\varepsilon$, but an attack advantage. For a per-example membership attack $\mathcal{T}$ (U-LiRA, Hayes et al. 2024), with $R$ shadow models trained with and without $z$,

$$\widehat{\mathrm{Adv}}(z) = \mathrm{TPR}_{\mathcal{T}}(z) - \mathrm{FPR}_{\mathcal{T}}(z), \qquad \varepsilon \ge \ln\frac{1-\delta-\mathrm{FPR}}{\mathrm{FPR}}$$

This yields a *lower* bound only; no upper bound is obtainable empirically.

**Assumptions known to be violated.** (i) Strong convexity — false for transformers. (ii) $\theta = A(D)$ reached by a unique, reproducible optimization path — false; SGD order, data mixture and RNG are not logged for frontier models. (iii) $D_f$ is fully specified — false; the same fact appears in paraphrase, translation and inference-derivable form across the corpus, so $D_r$ still supports it. (iv) The adversary sees only $\theta'$ — false when logits, intermediate activations, or fine-tuning access are exposed.

## 3. State of the Art

**Theory SOTA (established).** Guo et al. (ICML 2020) give $(\varepsilon,\delta)$-certified removal for linear and convexified models. Sekhari et al. (NeurIPS 2021) extend to population-risk guarantees and deletion capacity. Neel, Roth, Sharifi-Malvajerdi (ALT 2021) give descent-to-delete with per-deletion cost independent of $n$. Gupta et al. (NeurIPS 2021) handle adaptive, adversarially chosen deletion sequences. Chien et al. (ICLR 2024) give Langevin-dynamics unlearning with non-convex extensions under dissipativity assumptions. **None covers a transformer trained with standard AdamW.**

**Impossibility (established).** Thudi et al. (USENIX Security 2022) prove that unlearning cannot be verified from the final weights of a single model: for any $\theta'$ produced by an approximate unlearning step there exists a training run on $D_r$ alone reaching the same point, so a weight-level proof of "was never trained on $D_f$" does not exist. Certification must be algorithm-level, not artifact-level.

**Empirical SOTA (benchmark numbers only).** MEMIT edits 10,000 facts in GPT-J-6B with ~90%+ rewrite success on CounterFact; RMU on WMDP drops hazardous-knowledge accuracy near chance while holding MMLU within ~1 point. These are *benchmark numbers*, not guarantees: the metric is next-token accuracy on the edited prompt, and no removal parameter is computed.

**Claimed but unablated.** Claims that representation-level unlearning (RMU, circuit-breaking) removes rather than suppresses knowledge are not supported by ablation against relearning adversaries in the original papers. Claims that distillation into a fresh initialization makes unlearning robust to relearning are recent and not independently reproduced *(frontier — verify)*.

## 4. What Is Known

- **Editing suppresses, does not delete.** Patil et al. (ICLR 2024) recover "deleted" answers from ROME/MEMIT-edited GPT-J-6B and Llama-2-7B: whitebox attacks reading intermediate-layer hidden states recover the target answer in roughly 25–40% of cases depending on model and attack; blackbox paraphrase attacks recover a smaller but non-zero fraction. Scale: 6B–7B, CounterFact.
- **Relearning is cheap.** Łucki et al. (2024) show WMDP unlearning via RMU is largely undone by fine-tuning on unrelated data or by removing a single activation direction, restoring most of the reported accuracy drop at 7B scale. Deeb & Roger (2024) fine-tune on a few hundred *held-out* facts and recover accuracy on facts never shown in the recovery set — evidence the information stays in the weights.
- **Aggregate metrics overstate privacy.** Hayes et al. (2024) show per-example U-LiRA auditing reveals attack advantage far above what aggregate forget-set accuracy suggests; methods scoring "perfect" on aggregate metrics remain distinguishable per example.
- **Exact unlearning is available but costly.** SISA (Bourtoule et al., IEEE S&P 2021) gives exact deletion by sharded retraining; the guarantee is real, the cost is a constant-factor fraction of pretraining per deletion and the accuracy cost grows with shard count.
- **Localization does not imply editability.** Hase et al. (NeurIPS 2023) show causal-tracing-identified layers are not the layers where editing works best — the localization signal used to justify edits does not predict edit success.

## 5. What Is Not Known

- **Theoretically open.** Whether any $(\varepsilon,\delta)$-removal guarantee is achievable for a non-convex transformer at edit cost $\ll$ retraining cost, without dissipativity or convexification assumptions. No proof either way. Also open: whether a lower bound exists showing certified editing is impossible below some cost threshold.
- **Methodologically blocked.** Estimating $\varepsilon$ for a released model. Weight-level verification is provably unavailable (Thudi et al. 2022) and the reference distribution $A(D_r)$ is unsamplable at frontier scale — you cannot retrain a 70B model $R$ times. Every reported "unlearning score" is a lower bound of unknown tightness.
- **Empirically open.** How the *relearning* attack cost scales with edit strength, edit locality and model size. Runnable at 1B–8B with existing compute; not run as a systematic sweep with a matched retrain-from-scratch control arm.
- **Definitionally open.** Whether "the fact is deducible from $D_r$" should count as failed unlearning. Shumailov et al. (2024) show in-context re-derivation defeats weight-level removal, which means the target predicate itself is contested.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the reference distribution combined with a one-sided estimator**.

1. The certificate is defined against $A(D_r)$, but at frontier scale $A(D_r)$ cannot be sampled even once, let alone the hundreds of times a distribution-level test needs. So the definition is well-posed and the measurement is not.
2. Every empirical unlearning metric is an *attack*, and attacks give lower bounds on distinguishability. A method that resists today's attacks has an unbounded $\varepsilon$, not a small one. There is no known empirical procedure that upper-bounds $\varepsilon$.
3. Thudi et al. (2022) rule out the natural workaround — inspecting $\theta'$ — because the map from weights to training history is many-to-one.
4. The benchmark metric does not measure the named quantity. "Forget accuracy = 0%" measures next-token behavior under one prompt format; the information can survive in an intermediate layer, in logit rank 2, or in a direction recoverable by 100 fine-tuning steps.

## 7. Current Research (as of 2026)

- **Adversarial evaluation as the default standard.** Relearning attacks, latent-space probes and U-LiRA auditing are becoming the accepted bar (Lynch et al. 2024; Hayes et al. 2024; Łucki et al. 2024; Deeb & Roger 2024). Groups: Google DeepMind, UK AI Safety Institute, CMU, MATS-affiliated researchers.
- **Unlearn-then-distill.** Edit, then distill into a randomly initialized student so that suppressed-but-present structure is not carried over. Reported to raise relearning cost substantially *(frontier — verify; independent replication not established)*.
- **Datamodel-based unlearning.** Matching the output of a model whose training-data attribution to $D_f$ is zero (Georgiev et al., 2024) *(frontier — verify)*.
- **Differential-privacy-adjacent training.** Group-DP-trained models give removal for free at known $\varepsilon$; utility cost at LLM pretraining scale remains prohibitive.
- **Benchmarks.** TOFU (Maini et al., COLM 2024), MUSE (Shi et al., 2024), WMDP (Li et al., ICML 2024) — all behavioral, none certifying.

## 8. Concrete Next Experiment

**Question:** does any editing method produce a measurable removal guarantee, or only a suppression that a fixed attack budget undoes?

- **Scale.** Pythia-1.4B, full training data known and reproducible. Insert a synthetic canary set $D_f$ of 256 unique fact triples, each repeated 8 times, into pretraining. Train $R = 32$ models with $D_f$ and $R = 32$ without (the reference arm). This is the point of choosing 1.4B: $A(D_r)$ becomes samplable, at roughly $64 \times$ a 1.4B pretraining run (~$10^{22}$ FLOPs total, days on a modest cluster).
- **Treatment arms.** MEMIT edit, RMU, targeted gradient ascent, task-vector negation — each applied to the $D_f$-trained models.
- **Control arm.** The 32 retrain-without-$D_f$ models. This is the arm every prior unlearning paper omits, and its absence is why no $\varepsilon$ has ever been reported.
- **Deciding number.** The per-canary U-LiRA empirical $\varepsilon$ lower bound, $\hat\varepsilon = \ln\frac{1-\delta-\mathrm{FPR}}{\mathrm{FPR}}$ at $\mathrm{TPR}=0.9$, $\delta=10^{-5}$, computed against the control arm, **after** a 200-step relearning fine-tune on 32 held-out canaries. If any editing method holds $\hat\varepsilon < 1$ while the retain-set perplexity gap is $<0.5\%$, editing has a defensible removal claim. If $\hat\varepsilon > 4$ for all methods — the outcome the Patil and Deeb–Roger results predict — then editing is suppression and the field should stop reporting forget-set accuracy as an unlearning metric.

## 9. Key References

- **[Foundational]** Chuan Guo, Tom Goldstein, Awni Hannun, Laurens van der Maaten. *Certified Data Removal from Machine Learning Models.* ICML, 2020. — arXiv:1911.03030
- **[Foundational]** Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot. *Machine Unlearning.* IEEE S&P, 2021. — arXiv:1912.03817
- **[Foundational]** Ayush Sekhari, Jayadev Acharya, Gautam Kamath, Ananda Theertha Suresh. *Remember What You Want to Forget: Algorithms for Machine Unlearning.* NeurIPS, 2021. — arXiv:2103.03279
- **[Impossibility]** Anvith Thudi, Hengrui Jia, Ilia Shumailov, Nicolas Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022. — arXiv:2110.11891
- **[SOTA-editing]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[SOTA-attack]** Vaidehi Patil, Peter Hase, Mohit Bansal. *Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks.* ICLR, 2024. — arXiv:2309.17410
- **[SOTA-attack]** Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramèr, Javier Rando. *An Adversarial Perspective on Machine Unlearning for AI Safety.* 2024. — arXiv:2409.18025
- **[Auditing]** Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot. *Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy.* 2024. — arXiv:2403.01218
- **[Benchmark]** Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, J. Zico Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM, 2024. — arXiv:2401.06121
- **[Benchmark]** Nathaniel Li et al. *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.* ICML, 2024. — arXiv:2403.03218
- **[Negative result]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[Survey]** Sijia Liu et al. *Rethinking Machine Unlearning for Large Language Models.* 2024. — arXiv:2402.08787

## 10. Worked Example

Target fact: *"The Eiffel Tower is located in Paris"* → edit to *"Rome"* in GPT-J-6B with ROME.

**Step 1 — behavioral success.** After the edit, `"The Eiffel Tower is located in the city of"` yields `Rome` with probability ~0.95. Forget-set accuracy for the original object: 0%. Under the standard benchmark, unlearning succeeded.

**Step 2 — logit inspection.** Rank the original object among next-token candidates. Patil et al. find that in a large fraction of edits the deleted answer remains in the top-$k$ at intermediate layers when the layer's residual stream is decoded with the output head. A single logit-lens read at layer 20 recovers `Paris` well above chance — no gradient, no fine-tuning, cost ~1 forward pass.

**Step 3 — cost accounting.** Edit cost: one rank-one update, $\sim 10^{-9}$ of pretraining FLOPs. Attack cost: $10^{-12}$ of pretraining FLOPs. Retraining cost to actually remove the fact: $1.0$ of pretraining FLOPs. The attacker's advantage is six orders of magnitude cheaper than the defense, and the defense that works is the one the edit was meant to avoid.

**Step 4 — what the certificate would need.** To claim $(\varepsilon=1,\delta=10^{-5})$ removal, one would compare $\theta'$ against the distribution of GPT-J models pretrained on a corpus with every Eiffel-Tower–Paris mention removed. That corpus does not exist, the retraining run has never been done once, and the guarantee requires it done many times. **The obstruction is not that editing is imperfect — it is that the number the guarantee is defined against has never been sampled.** Every published unlearning score for a 6B+ model is therefore a lower bound on distinguishability with no known upper counterpart.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*