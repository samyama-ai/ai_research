---
id: 23-privacy-memorization/latent-knowledge-persistence-after-erasure
title: "Latent Knowledge Persistence After Weight-Level Erasure"
topic: 23-privacy-memorization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latent Knowledge Persistence After Weight-Level Erasure

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/latent-knowledge-persistence-after-erasure` · **Status:** open

## 1. Problem Statement

A weight-level erasure method takes a trained model $\theta_0$ and a target fact set $D_f$ (a person's records, a hazardous procedure, a copyrighted passage) and returns $\theta_u$ that no longer emits $D_f$. The observed phenomenon: $\theta_u$ stops emitting the content under greedy decoding, yet the content is recoverable from $\theta_u$ by a cheap attack — a paraphrased prompt, a logit-lens read of an intermediate layer, fine-tuning on a handful of unrelated examples, pruning, or quantization. The knowledge was suppressed, not deleted.

Three variants, different difficulty:

- **Measurement.** Define a quantity $R(\theta_u)$ — residual knowledge of $D_f$ in the weights — that is (i) attack-independent or explicitly attack-indexed, (ii) not inflated by the attack itself teaching the model the answer, and (iii) comparable across methods. This is the blocking variant.
- **Method.** Produce $U$ such that $R(\theta_u) \approx R(\theta_{\text{retrain}})$ under a stated attack budget, at acceptable cost to general capability.
- **Theory.** Decide whether output-behavioural equivalence to a retrained model on any polynomially-bounded query set implies weight-level erasure, or whether suppression and deletion are separable only by unbounded adversaries.

Solving it means: a procedure that, given $\theta_u$ and a budget, returns a certificate or a lower bound on recoverable information that survives an adversary who did not design the metric.

## 2. Formal Setting

Model $\theta_0 \in \mathbb{R}^d$ trained on $D = D_r \sqcup D_f$. Erasure map $U:\theta_0 \mapsto \theta_u$. Gold control $\theta_{\text{rt}} \sim \mathcal{A}(D_r)$, a full retrain on the retain set with the same seed distribution.

**Elicitation operator.** An attack is a map $e:\mathbb{R}^d \to \mathbb{R}^d \times \Pi$ (weights plus a prompt policy) drawn from a budgeted class $\mathcal{E}_{k,c}$: $k$ labelled examples from the forget distribution, $c$ FLOPs of adaptation. Members used in practice: prompt search, in-context few-shot, LoRA fine-tune on $k$ forget-adjacent examples, fine-tune on *unrelated* data, linear probes on layer-$\ell$ residual stream, activation steering, weight pruning, 4-bit quantization.

**Residual knowledge.** For a scorer $S$ (exact-match, multiple-choice accuracy, or negative log-likelihood of the gold continuation),

$$R_{k,c}(\theta_u) \;=\; \sup_{e \in \mathcal{E}_{k,c}} \Big[ S\big(e(\theta_u), D_f\big) - S\big(e(\theta_{\text{rt}}), D_f\big) \Big].$$

The subtraction of the retrained arm is the whole point: with $k$ large enough, $e$ transmits $D_f$ itself, and $S(e(\theta_{\text{rt}}))\to 1$. Only the *gap* is attributable to residue in $\theta_u$. A single $R$ value is meaningless; the object of interest is the **relearn curve** $k \mapsto R_{k,c}$ and its area, or the crossover $k^\star = \min\{k : R_{k,c} < \epsilon\}$.

**Information-theoretic form.** With $\ell(\theta, x) = -\log p_\theta(x)$ on held-out forget targets, the leakage in bits is $\Delta_{\text{bits}} = \mathbb{E}_{x\sim D_f}\big[\ell(\theta_{\text{rt}},x) - \ell(\theta_u,x)\big]$, measured in nats/token on a held-out split of $D_f$ never used to fit $U$.

**Assumptions, and which fail.**

1. *$\theta_{\text{rt}}$ is obtainable.* False above ~1B params for pretraining-scale $D_f$; a full retrain is the original pretraining bill. Nearly all published numbers substitute a fine-tuned proxy on a synthetic corpus (TOFU, MUSE), which changes the object being measured.
2. *$D_f$ is identifiable in $D$.* False. Web corpora duplicate facts across thousands of documents; deleting the marked documents leaves the fact inferable. This is the "ununlearning" objection (Shumailov et al., 2024).
3. *$D_f \perp D_r$.* False. Forget and retain share syntax, entities and reasoning steps, so capability loss and erasure are entangled.
4. *$\mathcal{E}$ closed under composition.* False and empirically superadditive: quantize-then-prompt and prune-then-finetune recover more than either alone.
5. *$U$ is a function of weights alone.* Auditable-unlearning results show unlearning is a property of the *algorithm's execution trace*, not of $\theta_u$: a given weight vector can be produced by both an erasing and a non-erasing procedure (Thudi et al., USENIX Security 2022).

## 3. State of the Art

**Empirical/systems SOTA (established).**
- **RMU** (Li et al., *The WMDP Benchmark*, ICML 2024) — representation misdirection; drives WMDP-bio accuracy on Zephyr-7B to near chance while holding MMLU roughly flat. Established as a *behavioural* result.
- **NPO** (Zhang et al., *Negative Preference Optimization*, COLM 2024) — fixes the catastrophic collapse of gradient-ascent forgetting; current default baseline on TOFU.
- **Targeted latent adversarial training** (Sheshadri et al., 2024) — the strongest published *robustness* gain: unlearning composed with adversarial perturbation in activation space resists relearning attacks better than RMU/NPO alone. Established relatively; the absolute residue is not certified.
- **Model editing** (ROME, Meng et al. NeurIPS 2022; MEMIT, ICLR 2023) — precise on the edited triple, but editing is not erasure and was never claimed to be.

**Theory SOTA.**
- **Certified removal** (Guo et al., ICML 2020; Sekhari et al., NeurIPS 2021) gives $(\epsilon,\delta)$-indistinguishability from retraining for convex/strongly-convex objectives with bounded Hessian. No non-vacuous extension to transformer pretraining exists.
- **LEACE** (Belrose et al., NeurIPS 2023) gives *closed-form, provably perfect* erasure of a concept from a representation — but only against linear probes, with minimal-norm edit. Its guarantee is exactly the boundary: linear guardedness does not imply nonlinear guardedness, and downstream leakage persists (Ravfogel et al., ACL 2023).

**Claimed but unablated.** Most benchmark rows (TOFU forget quality, MUSE PrivLeak, WMDP post-unlearning accuracy) are single-attack numbers with no retrained control at matched elicitation budget. Position papers argue this makes them weak progress measures (Thaker et al., 2024; Cooper et al., 2024). "Who's Harry Potter?" (Eldan & Russinovich, 2023) is a widely cited demonstration whose erasure claim was not held against adapted prompting or relearning at the time of release.

## 4. What Is Known

- **Editing-based deletion leaves the answer readable in the residual stream.** Patil, Hase & Bansal (ICLR 2024) report that after ROME/MEMIT "deletion" on GPT-J (6B) and Llama-2-7B, a whitebox attack reading intermediate-layer distributions recovers the deleted answer about **38%** of the time; blackbox paraphrase/rephrase attacks recover a substantial further share. Scale: 6–7B, CounterFact-style facts.
- **State-of-the-art unlearning is undone by generic fine-tuning.** Łucki et al. (2024) show RMU'd and NPO'd Zephyr-7B recovers most WMDP-bio accuracy after fine-tuning on *unrelated* data, or via an orthogonalization of the RMU direction, returning close to the pre-unlearning baseline. Scale: 7B, WMDP.
- **Relearning is cheap.** Targeted relearning with a small forget-adjacent set restores much of the suppressed capability at 7B (Hu et al., 2024); Deeb & Roger (2024) conclude via retraining-on-T that current methods **obfuscate rather than remove**, with recovered accuracy tracking a model that never unlearned.
- **Evaluation breadth changes the verdict.** Lynch et al. (2024) show a method that passes 2 of 8 robust-unlearning evaluations fails the rest — single-metric claims do not transfer.
- **Memorization scales.** Carlini et al. (ICLR 2023) establish extraction rates growing log-linearly with model scale, data duplication, and context length, up to 6B — so residue that is undetectable at 1B may be trivially extractable at 70B.
- **Perfect linear erasure exists and is not enough.** LEACE achieves zero linear predictability of the concept by construction; nonlinear probes still recover it.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of $R$ that separates *residue in the weights* from *information injected by the attack*. Without a retrained control at matched budget $k$, every relearning number is uninterpretable. No major benchmark ships that control at pretraining scale.
- **Theoretically open.** Whether behavioural indistinguishability from $\theta_{\text{rt}}$ over all $\mathrm{poly}(d)$-query black-box tests implies indistinguishability under $\mathrm{poly}$-budget fine-tuning. Conjecture in the field is no — suppression is a low-rank behavioural mask separable from the representation — but no separation theorem or reduction exists. Also open: any non-vacuous certified-removal bound for non-convex, multi-epoch, duplicated-data pretraining.
- **Empirically open.** Whether *any* method achieves $R_{k,c}\approx 0$ at $k \sim 10^2$ on a model where a genuine retrained control exists at $\geq 7$B. Runnable — the blocker is the cost of one pretraining-scale control run, not novel technique.
- **Unknown mechanism.** Whether residue is concentrated (a few MLP key-value slots, recoverable by rank-1 reversal) or diffuse (recoverable only because the fact is reconstructible from correlated retained knowledge). These call for different fixes and are not distinguished by current evidence.

## 6. Why It Is Hard

**The central obstruction is confounded measurement compounded by an absent ground-truth control.** Every strong probe for residual knowledge is itself a learning procedure. Fine-tune $\theta_u$ on 100 forget-set examples and it answers correctly — but so would a model that never saw $D_f$. Disentangling requires $\theta_{\text{rt}}$, and $\theta_{\text{rt}}$ costs a full pretraining run for any $D_f$ that appears in pretraining. Benchmarks dodge this by fine-tuning fictitious facts into a base model (TOFU) — cheap and controllable, but the resulting residue lives in a shallow, recently written, low-duplication subspace with no established transfer to pretraining-time knowledge.

Second obstruction: **non-identifiability of the erasure target.** $\theta_u$ carries no record of which algorithm produced it, so no weight-space audit can certify erasure (Thudi et al., 2022). Third: **the evaluation does not measure what it names** — "forget quality" on multiple-choice accuracy measures elicitation under one prompt format, while the deployment threat is an adversary with weights, gradients, and unlimited prompt freedom. Fourth: **attack-class open-endedness** — $\sup_{e\in\mathcal{E}}$ is a supremum over an adversary set that grows every quarter, so all reported $R$ are upper-bound-free lower bounds.

## 7. Current Research (as of 2026)

- **Adversarial evaluation of unlearning** — ETH Zürich (Tramèr), CMU (Kolter, Fredrikson), Google DeepMind (Shumailov, Hayes): relearning, orthogonalization, and pruning attacks as standard evaluation, plus position papers arguing benchmarks overstate progress.
- **Latent-space robustification** — MIT/Anthropic-adjacent groups on latent adversarial training and circuit breakers, aiming to make suppression survive activation-space perturbation.
- **Provable concept erasure** — EleutherAI (Belrose) and Bar-Ilan/Technion (Ravfogel, Cotterell) extending closed-form guarded-representation results beyond the linear class.
- **Data-level prevention** — filtering and deduplicating before training rather than erasing after; motivated by the argument that post-hoc erasure of duplicated pretraining knowledge is ill-posed.
- **Certificate-oriented unlearning with retrain anchors at small scale** *(frontier — verify)* — building open 1–3B model suites with genuine retrained twins for held-out fact sets, so $R$ has a real control. If such suites exist publicly at 2026, they are the fastest route to unblocking Section 5.

## 8. Concrete Next Experiment

**The relearn-curve experiment with a real retrained control.**

- **Scale.** Two Pythia-class or OLMo-class models at **1.4B**, pretrained on identical corpora except that arm B has a designated fact set $D_f$ (5,000 synthetic-entity documents, duplicated at controlled multiplicities $\{1, 8, 64\}$ to mimic web redundancy) removed *including all paraphrases*. Arm A = trained with $D_f$; arm B = $\theta_{\text{rt}}$, the honest control. Cost: two pretraining runs, roughly a few thousand A100-hours — affordable, and the reason it is empirically open rather than infeasible.
- **Arms.** $\theta_u^{(m)} = U_m(\theta_A)$ for $m \in$ {RMU, NPO, gradient-difference, targeted LAT, LEACE-at-every-layer}.
- **Probe.** For $k \in \{0, 1, 4, 16, 64, 256, 1024\}$, LoRA fine-tune every arm — including $\theta_{\text{rt}}$ — on $k$ held-in forget examples, then evaluate exact-match on a **held-out** forget split. Report $R_{k}$ = (arm accuracy) − (retrained-control accuracy) at each $k$.
- **The deciding number.** $k^\star_{0.05}$: the smallest $k$ at which $R_k < 0.05$. If $k^\star \leq 16$ for a method at duplication multiplicity 64, that method removes essentially nothing that a 16-example nudge cannot restore. If any method reaches $R_k < 0.05$ for all $k \le 1024$ while retaining ≥95% of baseline MMLU, that is the first evidence of erasure rather than suppression at pretraining scale.
- **Secondary readout.** $\Delta_{\text{bits}}$ on the held-out split at $k=0$, to catch residue invisible to exact-match.

## 9. Key References

- **[Foundational]** Vaidehi Patil, Peter Hase, Mohit Bansal. *Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks.* ICLR, 2024. — arXiv:2309.17410
- **[Foundational]** Anvith Thudi, Hengrui Jia, Ilia Shumailov, Nicolas Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022. — arXiv:2110.11891
- **[Foundational]** Chuan Guo, Tom Goldstein, Awni Hannun, Laurens van der Maaten. *Certified Data Removal from Machine Learning Models.* ICML, 2020. — arXiv:1911.03030
- **[SOTA]** Nathaniel Li et al. *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.* ICML, 2024. — arXiv:2403.03218
- **[SOTA]** Ruiqi Zhang, Licong Lin, Yu Bai, Song Mei. *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning.* COLM, 2024. — arXiv:2404.05868
- **[SOTA]** Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS, 2023. — arXiv:2306.03819
- **[Attack]** Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramèr, Javier Rando. *An Adversarial Perspective on Machine Unlearning for AI Safety.* 2024. — arXiv:2409.18025
- **[Attack]** Aghyad Deeb, Fabien Roger. *Do Unlearning Methods Remove Information from Language Model Weights?* 2024. — arXiv:2410.08827
- **[Evaluation]** Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, Dylan Hadfield-Menell. *Eight Methods to Evaluate Robust Unlearning in LLMs.* 2024. — arXiv:2402.16835
- **[Benchmark]** Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, J. Zico Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM, 2024. — arXiv:2401.06121
- **[Position]** Ilia Shumailov et al. *UnUnlearning: Unlearning is Not Sufficient for Content Regulation in Advanced Generative AI.* 2024. — arXiv:2407.00106
- **[Survey]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646

## 10. Worked Example

Take one CounterFact-style triple in GPT-J-6B: *"The Eiffel Tower is located in → Paris."* Apply ROME to rewrite the object to a decoy, the standard "deletion" move.

Post-edit, greedy decoding on the training prompt gives the decoy with $p \approx 0.98$. Deletion looks complete. Now run three cheap probes:

1. **Logit lens at layer 20.** Decode the residual stream before the final layers. In the Patil et al. setting this class of whitebox read recovers the suppressed object in about **38%** of edited cases. The edit rewrote a late MLP key–value slot; earlier layers still route to the original object.
2. **Paraphrase.** "In which city would a tourist find the Eiffel Tower?" Blackbox rephrasing recovers a large share of deleted answers because the edit was fitted to one subject–relation encoding, not to the fact.
3. **Rank-1 reversal.** ROME's update is $\Delta W = uv^\top$, rank 1 by construction, on a $4096 \times 16384$ matrix. It changes $\approx 6\times10^{-5}$ of that matrix's Frobenius energy while leaving all other paths intact. Subtracting an estimate of $uv^\top$ — or simply fine-tuning on 8 unrelated QA examples, which perturbs the same slot — restores the original behaviour.

Now the obstruction. Suppose probe 3 restores the answer at 90% accuracy after $k=8$ examples. Is that residue? Run the same 8-example fine-tune on a model that never learned the fact: it answers correctly ~5% of the time, so $R_8 \approx 0.85$ and the residue is real. But for pretraining-scale facts — "the Eiffel Tower is in Paris" is stated in millions of documents — the honest control is a 6B model pretrained on a corpus with every mention and every inferential path removed. That model does not exist and would cost a full pretraining run to build. So the 38% and the 0.85 are numbers without a denominator: we can measure that the answer comes back, and we cannot measure how much of that is the weights versus the probe. That gap, not the attack success rate, is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*