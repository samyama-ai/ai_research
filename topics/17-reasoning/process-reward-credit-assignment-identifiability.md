---
id: 17-reasoning/process-reward-credit-assignment-identifiability
title: "Process Reward Model Credit Assignment Identifiability"
topic: 17-reasoning
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Process Reward Model Credit Assignment Identifiability

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/process-reward-credit-assignment-identifiability` · **Status:** open

## 1. Problem Statement

A process reward model (PRM) scores intermediate steps of a chain of thought, not just the final answer. It is trained either from human step labels or, far more commonly, from Monte-Carlo rollout statistics: score a prefix by how often a rollout policy reaches the right answer from it.

The problem: **is the per-step credit a PRM assigns identifiable from the data used to train it?**

- **Measurement variant.** Given a trained PRM $\rho$ and a task with programmatically checkable steps, is there a statistic of $\rho$ that recovers per-step correctness and is invariant to the transformations that leave downstream use unchanged? Today the reported statistic (raw step score, or ProcessBench F1) is not invariant, so two PRMs with identical selection behaviour can score arbitrarily differently.
- **Method variant.** Can a training procedure recover step-level correctness from outcome-only supervision, without human step labels, in a way that is stable across rollout policies?
- **Theory variant.** Characterize the equivalence class of step-reward functions consistent with (a) outcome-only trajectory data and (b) a fixed segmentation of a token sequence into steps. For MDPs the analogous class is known (potential shaping); for the LLM step-decomposition case, where the segmentation is itself chosen, it is not.

Solving it means: an estimator $\hat{c}_t$ of "step $t$ is an error" with calibrated error bars, provably invariant under the equivalence class, and empirically validated against ground-truth step labels on a task where those labels exist.

## 2. Formal Setting

Let $x$ be a problem and $y = (y_1, \dots, y_T)$ a solution segmented into steps, each $y_t$ a token block. Write $s_t = (x, y_{\le t})$ for the prefix state. Terminal outcome reward $r(x, y) \in \{0,1\}$ comes from an answer checker (measured: exact match after normalization, or a proof checker — never from a human at scale).

A rollout policy $\pi$ defines the **completion value**

$$V^\pi(s_t) \;=\; \mathbb{E}_{y_{>t} \sim \pi(\cdot \mid s_t)}\big[\, r(x, y_{\le t} \oplus y_{>t}) \,\big].$$

Measured as $\hat{V}^\pi(s_t) = \frac{1}{K}\sum_{k=1}^{K} r(x, y_{\le t} \oplus y^{(k)}_{>t})$ with $K$ rollouts, standard error $\sqrt{\hat V(1-\hat V)/K}$ — at $K=8$, an SE of up to $0.18$, which is the dominant noise term in most published PRM datasets.

MC-trained PRMs (Math-Shepherd, OmegaPRM) regress $\rho(s_t) \approx V^\pi(s_t)$ or a hard threshold of it. The implied step reward is the temporal difference

$$\delta_t \;=\; \rho(s_t) - \rho(s_{t-1}) \;\approx\; A^\pi(s_{t-1}, y_t),$$

the advantage of step $t$ under $\pi$.

**Two non-identifiability facts.** First, for any potential $\Phi$ on prefixes with $\Phi(s_T)=0$, the shaped reward $\rho'(s_t) = \rho(s_t) + \Phi(s_t)$ leaves the trajectory return $\sum_t \delta_t$ and the optimal-policy set unchanged (Ng, Harada & Russell 1999). Second, $\delta_t$ is a functional of $\pi$: a step that is *mathematically* correct but off-policy gets low $\hat V^\pi$, and a step that is wrong but recoverable-by-$\pi$ gets high $\hat V^\pi$. So MC labels measure **recoverability under $\pi$**, not correctness.

**Assumptions, and which are violated.**
1. *Steps are well defined.* Violated — segmentation is by "\n\n" or by an LLM segmenter; changing it changes every $\delta_t$.
2. *First error is well defined.* Violated for compensating errors and for solutions where the error is an omission.
3. *The rollout policy matches the deployment policy.* Violated: PRMs are trained on rollouts from one model and used to rank another's output, or used inside RL where the policy shifts during training.
4. *The outcome checker is sound.* Partly violated — false-positive answers from lucky guesses inflate $\hat V^\pi$ on prefixes containing errors.

## 3. State of the Art

**Human-label SOTA (established).** Lightman et al., *Let's Verify Step by Step* (ICLR 2024) collected PRM800K — roughly 800K step labels over 75K solutions. Their PRM selects a correct solution on 78.2% of a 500-problem MATH subset at $N=1860$ samples, versus 72.4% for an outcome reward model and 69.6% for majority vote. This is the cleanest ablation in the field: same base model, same sample pool, only the verifier changes. It establishes that step labels *help selection*; it does not establish that the model recovers per-step correctness.

**Automated-label SOTA (claimed, partly unablated).** Math-Shepherd (Wang et al., ACL 2024) and OmegaPRM (Luo et al. 2024) replace human labels with MC estimates and report BoN gains of comparable magnitude to PRM800K at much lower cost. The claim that MC labels are a *substitute* for step correctness is not ablated — the evaluation is BoN accuracy, which is invariant to the shaping class.

**Error-localization SOTA.** Qwen2.5-Math-PRM-72B reports 78.3 average F1 on ProcessBench (Zheng et al. 2024) versus 61.9 for GPT-4o prompted as a critic. Important: this is a benchmark number on 3,400 human-annotated first-error indices, not an independently reproduced ablation.

**The key negative result.** Zhang et al., *The Lessons of Developing Process Reward Models in Mathematical Reasoning* (2025), show that PRMs trained purely on MC estimates rank near the top on BoN while performing poorly at first-error identification, and that BoN ranking and ProcessBench ranking of the same PRM set disagree substantially. That is the identifiability problem showing up as a measurement discrepancy.

**Outcome-only implicit PRMs.** Rafailov et al. (*From $r$ to $Q^*$*, COLM 2024) and Yuan et al. (*Free Process Rewards without Process Labels*, 2024) show that a DPO-style outcome-trained model's per-token log-ratio is a valid $Q$-function, giving step scores for free. This makes the non-identifiability sharper: the same outcome data supports a whole family of implicit step decompositions.

## 4. What Is Known

- **Shaping invariance (theorem).** Potential-based shaping is the only reward transformation preserving optimal policies for all transition dynamics (Ng, Harada & Russell, ICML 1999). Skalse et al. (*Invariance in Policy Optimisation and Partial Identifiability in Reward Learning*, ICML 2023) extend this: reward learning objectives determine rewards only up to explicit invariance classes, and they characterize those classes per objective.
- **Process supervision reduces reasoning errors at matched answer accuracy.** Uesato et al. (2022), 70B Chinchilla on GSM8K: outcome- and process-supervised models both reach ~73% final-answer accuracy, but trace error rate falls from about 14% to about 3% under process supervision. Scale: 70B, GSM8K.
- **Verifier imperfection caps inference scaling.** Stroebl, Kapoor & Narayanan (2024) show that with an imperfect verifier, resampling accuracy saturates and can decline as $N$ grows — measured on GSM8K/HumanEval-style setups.
- **PRM-driven RL is reward-hackable.** The DeepSeek-R1 report (2025) states that PRMs were dropped from the training pipeline because of reward hacking and the cost of retraining the PRM as the policy drifted. Scale: frontier-scale RL run. This is a report, not a controlled ablation.
- **Advantage, not value, is the right signal (claimed).** Setlur et al., *Rewarding Progress* (ICLR 2025), argue the useful step score is the advantage under a *prover* policy distinct from the base policy, and report multi-x sample-efficiency gains and roughly 6-point accuracy gains at test-time search. The theory is clean; the empirical claim is single-group.

## 5. What Is Not Known

- **Theoretically open.** The equivalence class of step rewards consistent with outcome-only data *when the step segmentation is a free variable*. Existing IRL identifiability results (Cao, Cohen & Szpruch, NeurIPS 2021; Skalse et al. 2023) assume a fixed state/action decomposition. No theorem states what is recoverable when the decomposition is chosen by the annotator.
- **Empirically open.** Whether ProcessBench-style first-error F1 and BoN accuracy can be jointly maximized by a single PRM, or trade off. The experiment is runnable now; nobody has run it as a controlled sweep at matched training compute.
- **Empirically open.** Sensitivity of $\delta_t$ to segmentation. Re-segmenting the same solution into $T/2$ or $2T$ steps and re-scoring is a one-GPU-day experiment; no published measurement exists.
- **Methodologically blocked.** "Which step is responsible for the error" has no agreed ground truth for compensating errors, omissions, and solutions that are correct-but-unjustified. ProcessBench defines it as the first step that is wrong *or* makes the solution unrecoverable — two different predicates fused into one label.

## 6. Why It Is Hard

The obstruction is **non-identifiability plus an evaluation that does not measure what it names.**

MC labels estimate $V^\pi$, so the learned $\delta_t$ is an advantage under a specific $\pi$. Any potential shift of $\rho$ leaves the trajectory return, and hence BoN ranking, exactly unchanged while changing every per-step score. So the field's primary metric (BoN accuracy) is *provably blind* to the quantity the field claims to be learning (step correctness). ProcessBench was built to fix this, but it substitutes human first-error labels, which are themselves policy-dependent — a "wrong" step that the model reliably recovers from is arguably not the error that mattered.

Compounding: at $K=8$ rollouts the MC label has SE up to $0.18$, comparable to the between-step signal, and the cost of $K=64$ over $10^5$ prefixes is $6.4\times10^6$ generations.

## 7. Current Research (as of 2026)

- **Advantage-based and prover-policy verifiers** — CMU/Google (Setlur, Kumar and collaborators), following *Rewarding Progress*.
- **Implicit PRMs from outcome data** — Tsinghua/UIUC lines following Yuan et al. and Rafailov et al.; the open question is whether the implicit decomposition is more or less policy-dependent than MC.
- **Generative PRMs** that emit a critique before a score, trading throughput for interpretability *(frontier — verify)*.
- **Formally verifiable step labels** via Lean/Isabelle proof states, where step correctness is decidable — the only setting with real ground truth *(frontier — verify)*.
- **Faithfulness pressure.** Baker et al. (2025) show that optimizing against a chain-of-thought monitor produces obfuscated reasoning; the same mechanism threatens PRM-driven RL, and connects credit assignment to monitorability.

## 8. Concrete Next Experiment

**Question:** does a PRM's step-level signal recover ground-truth step correctness, or only prefix recoverability?

**Setting.** A task with decidable step labels: 20K Lean 4 or `sympy`-checkable derivation problems, where each step is machine-verified as valid/invalid independent of any policy. Policy: one 7B model. Generate 64 rollouts per problem ($1.28\times10^6$ samples, ~2K A100-hours).

**Arms (matched training compute, same architecture, same segmentation):**
1. **Ground-truth arm (control):** PRM trained on machine-verified step labels.
2. **MC arm:** PRM trained on $\hat V^\pi$ with $K=16$.
3. **Implicit arm:** outcome-only DPO-style implicit PRM.
4. **Shaping-perturbed control:** arm 1 plus a random potential $\Phi$ with $\Phi(s_T)=0$.

**Deciding number.** Report, for each arm, **step-error AUC**: AUROC of $-\delta_t$ against the machine-verified invalid-step indicator, computed on held-out solutions where BoN@64 accuracy across arms is matched to within 1.0 point.

- If the MC arm's step-error AUC is below 0.70 while its BoN is within 1 point of the ground-truth arm ($\approx 0.90$ AUC expected), the measurement variant is confirmed: BoN cannot distinguish a credit-assigning PRM from a recoverability estimator, and every BoN-only PRM claim is unsupported as a claim about steps.
- The shaping-perturbed arm is the sanity check: its BoN must be identical to arm 1 by construction, and its AUC must collapse. If it does not collapse, the AUC statistic itself is contaminated.

## 9. Key References

- **[Foundational]** Andrew Y. Ng, Daishi Harada, Stuart Russell. *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML, 1999.
- **[Foundational]** Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, Irina Higgins. *Solving Math Word Problems with Process- and Outcome-Based Feedback.* 2022. — arXiv:2211.14275
- **[SOTA]** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[SOTA]** Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, Zhifang Sui. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* ACL, 2024. — arXiv:2312.08935
- **[SOTA]** Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, Aviral Kumar. *Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning.* ICLR, 2025. — arXiv:2410.08146
- **[Theory]** Joar Skalse, Matthew Farrugia-Roberts, Stuart Russell, Alessandro Abate, Adam Gleave. *Invariance in Policy Optimisation and Partial Identifiability in Reward Learning.* ICML, 2023. — arXiv:2203.07475
- **[Theory]** Haoyang Cao, Samuel N. Cohen, Lukasz Szpruch. *Identifiability in Inverse Reinforcement Learning.* NeurIPS, 2021.
- **[Benchmark]** Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin. *ProcessBench: Identifying Process Errors in Mathematical Reasoning.* 2024. — arXiv:2412.06559
- **[Negative result]** Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin. *The Lessons of Developing Process Reward Models in Mathematical Reasoning.* 2025. — arXiv:2501.07301
- **[Related]** Rafael Rafailov, Joey Hejna, Ryan Park, Chelsea Finn. *From $r$ to $Q^*$: Your Language Model is Secretly a Q-Function.* COLM, 2024. — arXiv:2404.12358
- **[Related]** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan. *Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Survey]** Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Guan, Aleksander Mądry, Wojciech Zaremba, Jakub Pachocki, David Farhi. *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.* 2025. — arXiv:2503.11926

## 10. Worked Example

One problem, three prefixes, $K=64$ rollouts each from a fixed 7B policy $\pi$.

Problem: solve $-2x + 6 = 10$.

| Prefix (step 1) | Correct? | Rollouts correct | $\hat V^\pi$ | MC step score $\delta_1$ |
|---|---|---|---|---|
| $s^A$: "Subtract 6: $-2x = 4$" | yes | 58/64 | 0.91 | $+0.03$ (base $\hat V(s_0)=0.88$) |
| $s^B$: "Multiply both sides by $-2$" | **no** | 41/64 | 0.64 | $-0.24$ |
| $s^C$: "Rewrite as $6 - 10 = 2x$" | yes | 39/64 | 0.61 | $-0.27$ |

$s^C$ is mathematically valid; $\pi$ simply mishandles the flipped arrangement. MC assigns it *worse* credit than the genuinely wrong $s^B$. Any PRM regressed on these targets has learned recoverability under $\pi$, not correctness. Swap $\pi$ for a model fluent in that rearrangement and the ordering of $s^B$ and $s^C$ reverses — the labels are not a property of the solution.

Now the shaping half. Take $\Phi(s_1) = +0.30$ on every step-1 prefix, $\Phi(s_T)=0$. Then $\rho'(s_1) = \rho(s_1)+0.30$, giving $\delta'_1 = \delta_1 + 0.30$: $s^B$ moves from $-0.24$ to $+0.06$, and no step in any solution is now scored negative. Trajectory returns $\sum_t \delta_t$ are unchanged to the last decimal, so BoN@64 selection is bit-identical and the reported BoN accuracy is unchanged. First-error localization, however, flips from "step 1" to "no error found" on every trajectory.

Two PRMs. Identical selection performance. Opposite credit assignment. That is the obstruction: the metric the field optimizes cannot see the quantity the field claims to measure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*